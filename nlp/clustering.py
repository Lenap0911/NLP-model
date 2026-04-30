import os
import sys
import logging
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

config = importlib.import_module('config.nlp_config')
logger = logging.getLogger(__name__)

# function-word stopwords per language — prevents c-TF-IDF surfacing articles/prepositions
# as topic keywords instead of flood-relevant content words
_STOPWORDS: dict[str, list[str]] = {
    'es': [
        'de','la','el','en','y','a','los','del','se','las','por','un','con',
        'una','su','al','lo','le','da','ha','que','no','es','pero','más',
        'esto','este','esta','han','sus','como','para','también','son','fue',
        'si','ya','todo','hay','sobre','cuando','donde','después','durante',
        'antes','mientras','entre','sin','hasta','desde','porque','aunque',
    ],
    'pt': [
        'de','da','do','das','dos','em','no','na','nos','nas','ao','à',
        'pelo','pela','pelos','pelas','a','o','as','os','e','ou','que',
        'com','para','por','mais','mas','não','já','também','só','assim',
        'ainda','se','como','porque','quando','onde','este','esta','esse',
        'essa','isso','isto','aquele','aquela','seu','sua','seus','suas',
        'um','uma','foi','são','tem','era','ser','estar','ter','há','haver',
    ],
    'en': [
        'the','a','an','in','of','to','and','or','is','are','was','were',
        'be','been','being','have','has','had','do','does','did','will',
        'would','could','should','may','might','that','this','these','those',
        'it','its','by','with','for','on','at','from','as','but','not',
    ],
}


def run_bertopic(
    texts: list,
    embeddings: np.ndarray = None,
    language: str = 'multilingual',
    languages: list = None,
) -> tuple:
    """
    fitting BERTopic on article texts with precomputed LaBSE embeddings
    following Dujardin et al. (2024) who used BERTopic for the 2021 EU floods
    passing precomputed embeddings avoids re-encoding (uses LaBSE output directly)

    BERTopic runs UMAP → HDBSCAN internally — no separate reduction/clustering step needed
    languages: ISO 639-1 codes present in the corpus — used to select stopwords
    returns (topic_model, topics, probs)
    topics is a list of topic IDs per document (-1 = outlier)
    """
    try:
        from bertopic import BERTopic
    except ImportError:
        raise ImportError('bertopic not installed — run: pip install bertopic')

    from sklearn.feature_extraction.text import CountVectorizer

    active_langs = languages or list(_STOPWORDS.keys())
    combined_stopwords = sorted({w for lang in active_langs for w in _STOPWORDS.get(lang, [])})
    logger.info(f'vectorizer stopwords: {len(combined_stopwords)} words for langs {active_langs}')
    vectorizer = CountVectorizer(
        stop_words=combined_stopwords,
        ngram_range=config.BERTOPIC_NGRAM_RANGE,
        min_df=config.BERTOPIC_MIN_DF,
    )

    logger.info('fitting BERTopic...')
    try:
        from hdbscan import HDBSCAN
        hdbscan_model = HDBSCAN(
            min_cluster_size=config.BERTOPIC_MIN_TOPIC_SIZE,
            min_samples=config.HDBSCAN_MIN_SAMPLES,
            cluster_selection_epsilon=config.HDBSCAN_CLUSTER_SELECTION_EPSILON,
            prediction_data=True,
            random_state=42,
        )
        topic_model = BERTopic(
            language=language,
            min_topic_size=config.BERTOPIC_MIN_TOPIC_SIZE,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            calculate_probabilities=True,
            verbose=True,
        )
    except ImportError:
        topic_model = BERTopic(
            language=language,
            min_topic_size=config.BERTOPIC_MIN_TOPIC_SIZE,
            vectorizer_model=vectorizer,
            calculate_probabilities=True,
            verbose=True,
        )

    if embeddings is not None:
        topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)
    else:
        topics, probs = topic_model.fit_transform(texts)

    n_total = len(topics)
    n_outliers_before = sum(1 for t in topics if t == -1)
    logger.info(
        f'BERTopic initial: {len(topic_model.get_topic_info()) - 1} topics, '
        f'{n_outliers_before}/{n_total} outliers ({100 * n_outliers_before // n_total}%)'
    )

    # reassign outliers to nearest topic by embedding distance (recommended for small corpora)
    n_outliers_final = n_outliers_before
    if n_outliers_before > 0 and embeddings is not None:
        try:
            topics = topic_model.reduce_outliers(
                texts, topics, strategy='embeddings', embeddings=embeddings
            )
            topic_model.update_topics(texts, topics=topics)
            n_outliers_final = sum(1 for t in topics if t == -1)
            logger.info(
                f'outlier reduction: {n_outliers_before} → {n_outliers_final} outliers '
                f'({n_outliers_before - n_outliers_final} reassigned)'
            )
        except Exception as e:
            logger.warning(f'outlier reduction failed ({e}) — keeping original topics')

    topic_info = topic_model.get_topic_info()
    n_topics_final = len(topic_info) - 1   # excludes the outlier row (-1)
    outlier_rate   = n_outliers_final / n_total if n_total > 0 else 0.0

    logger.info(
        f'BERTopic quality — topics: {n_topics_final}, '
        f'outlier rate: {n_outliers_final}/{n_total} ({outlier_rate:.0%})'
    )
    if outlier_rate > 0.50:
        logger.warning(
            f'outlier rate {outlier_rate:.0%} exceeds 50% — clustering is unreliable; '
            f'consider lowering BERTOPIC_MIN_TOPIC_SIZE or HDBSCAN_CLUSTER_SELECTION_EPSILON'
        )
    elif n_topics_final < 3:
        logger.warning(
            f'only {n_topics_final} topic(s) found — increase corpus size or lower '
            f'BERTOPIC_MIN_TOPIC_SIZE for more granular clustering'
        )
    else:
        logger.info(f'clustering quality: OK ({n_topics_final} topics, {outlier_rate:.0%} outliers)')

    return topic_model, topics, probs


def run_clustering(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    main entry point: running full clustering pipeline
    BERTopic's internal UMAP → HDBSCAN is the single clustering step (Dujardin et al. 2024)
    umap_cluster: raw HDBSCAN cluster labels (-1 = noise)
    topic_id: BERTopic semantic topic ID (-1 = outlier)
    adds columns: umap_cluster, topic_id to df
    """
    texts = df['embed_text'].tolist()
    languages = df['language'].dropna().unique().tolist() if 'language' in df.columns else None
    topic_model, topics, _ = run_bertopic(texts, embeddings=embeddings, languages=languages)

    df['umap_cluster'] = topic_model.hdbscan_model.labels_
    df['topic_id'] = topics

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(config.TOPICS_PATH, index=False)
    logger.info(f'topic info saved to {config.TOPICS_PATH}')

    return df
