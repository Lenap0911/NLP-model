import os
import sys
import logging
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

config = importlib.import_module('config.nlp_config')
logger = logging.getLogger(__name__)


def run_bertopic(
    texts: list,
    embeddings: np.ndarray = None,
    language: str = 'multilingual',
) -> tuple:
    """
    fitting BERTopic on article texts with precomputed LaBSE embeddings
    following Dujardin et al. (2024) who used BERTopic for the 2021 EU floods
    passing precomputed embeddings avoids re-encoding (uses LaBSE output directly)

    BERTopic runs UMAP → HDBSCAN internally — no separate reduction/clustering step needed
    returns (topic_model, topics, probs)
    topics is a list of topic IDs per document (-1 = outlier)
    """
    try:
        from bertopic import BERTopic
    except ImportError:
        raise ImportError('bertopic not installed — run: pip install bertopic')

    from sklearn.feature_extraction.text import CountVectorizer

    # Spanish stopwords — prevents function words dominating topic representations
    # BERTopic's default c-TF-IDF picks up 'de', 'la', 'el' without this
    _ES_STOPWORDS = [
        'de','la','el','en','y','a','los','del','se','las','por','un','con',
        'una','su','al','lo','le','da','ha','que','no','es','pero','más',
        'esto','este','esta','han','sus','como','para','también','son','fue',
        'si','ya','todo','hay','sobre','cuando','donde','después','durante',
        'antes','mientras','entre','sin','hasta','desde','porque','aunque',
    ]
    vectorizer = CountVectorizer(
        stop_words=_ES_STOPWORDS,
        ngram_range=(1, 2),
        min_df=2,
    )

    logger.info('fitting BERTopic...')
    try:
        from hdbscan import HDBSCAN
        hdbscan_model = HDBSCAN(
            min_cluster_size=config.BERTOPIC_MIN_TOPIC_SIZE,
            min_samples=1,
            cluster_selection_epsilon=0.5,
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
    if n_outliers_before > 0 and embeddings is not None:
        try:
            topics = topic_model.reduce_outliers(
                texts, topics, strategy='embeddings', embeddings=embeddings
            )
            topic_model.update_topics(texts, topics=topics)
            n_outliers_after = sum(1 for t in topics if t == -1)
            logger.info(
                f'outlier reduction: {n_outliers_before} → {n_outliers_after} outliers '
                f'({n_outliers_before - n_outliers_after} reassigned)'
            )
        except Exception as e:
            logger.warning(f'outlier reduction failed ({e}) — keeping original topics')

    topic_info = topic_model.get_topic_info()
    logger.info(f'BERTopic final: {len(topic_info) - 1} topics (excl. outlier topic -1)')
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
    topic_model, topics, _ = run_bertopic(texts, embeddings=embeddings)

    df['umap_cluster'] = topic_model.hdbscan_model.labels_
    df['topic_id'] = topics

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(config.TOPICS_PATH, index=False)
    logger.info(f'topic info saved to {config.TOPICS_PATH}')

    return df
