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


def _translate_keywords(kw_str: str, source_lang: str) -> str:
    """
    Translate a comma-separated keyword string from source_lang to English.
    Falls back to the original string if deep-translator is not installed or
    the request fails — translation is best-effort, not a hard requirement.
    """
    if source_lang == 'en' or not kw_str or kw_str == 'outlier':
        return kw_str
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source=source_lang, target='en').translate(kw_str)
        return result if result else kw_str
    except ImportError:
        logger.warning(
            'deep-translator not installed — topic keywords will not be translated. '
            'Run: pip install deep-translator'
        )
        return kw_str
    except Exception as e:
        logger.warning(f'keyword translation failed ({source_lang}→en): {e}')
        return kw_str


def _run_bertopic_for_language(
    lang: str,
    texts: list[str],
    embs: np.ndarray,
) -> tuple[dict[int, str], dict[int, str], list[int]]:
    """
    Run BERTopic on a single-language corpus slice and translate topic keywords.

    Running BERTopic per language means c-TF-IDF operates on monolingual text,
    so the extracted keywords are clean and interpretable in the source language
    rather than a mix of Spanish/English words within a single topic label.

    Returns:
        kw_src  — topic_id → top-5 keywords in source language
        kw_en   — topic_id → top-5 keywords translated to English
        topics  — per-document topic assignment (same order as input texts)
    """
    if len(texts) < config.BERTOPIC_MIN_TOPIC_SIZE:
        logger.warning(
            f'[{lang}] only {len(texts)} docs, min_topic_size={config.BERTOPIC_MIN_TOPIC_SIZE} '
            f'— skipping BERTopic for this language'
        )
        return {-1: 'outlier'}, {-1: 'outlier'}, [-1] * len(texts)

    topic_model, topics, _ = run_bertopic(texts, embs, languages=[lang])

    kw_src: dict[int, str] = {}
    kw_en: dict[int, str] = {}
    for topic_id in set(topics):
        if topic_id == -1:
            kw_src[-1] = 'outlier'
            kw_en[-1] = 'outlier'
            continue
        words = topic_model.get_topic(topic_id) or []
        kw_str = ', '.join(w for w, _ in words[:5])
        kw_src[topic_id] = kw_str
        kw_en[topic_id] = _translate_keywords(kw_str, lang)

    return kw_src, kw_en, topics


def _run_cross_lingual_hdbscan(
    embeddings: np.ndarray,
    n_docs: int,
) -> np.ndarray:
    """
    UMAP dimension reduction followed by HDBSCAN on the full LaBSE embedding matrix.

    Because LaBSE maps semantically equivalent content to nearby vectors regardless
    of source language, this step produces language-agnostic cluster IDs — an English
    article and a Spanish article about the same flood event will land in the same
    cluster even though they were never in the same per-language BERTopic run.

    Returns an array of cluster labels (int), -1 = noise/outlier.
    """
    try:
        import umap as umap_lib
        from hdbscan import HDBSCAN
    except ImportError as e:
        logger.warning(f'cross-lingual clustering skipped ({e}) — install umap-learn and hdbscan')
        return np.full(n_docs, -1, dtype=int)

    n_neighbors = min(config.UMAP_N_NEIGHBORS, n_docs - 1)
    logger.info(
        f'UMAP: {embeddings.shape} → {config.UMAP_N_COMPONENTS}d '
        f'(n_neighbors={n_neighbors}, metric=cosine)'
    )
    reducer = umap_lib.UMAP(
        n_components=config.UMAP_N_COMPONENTS,
        n_neighbors=n_neighbors,
        random_state=42,
        metric='cosine',
    )
    reduced = reducer.fit_transform(embeddings)

    clusterer = HDBSCAN(
        min_cluster_size=config.BERTOPIC_MIN_TOPIC_SIZE,
        min_samples=config.HDBSCAN_MIN_SAMPLES,
        cluster_selection_epsilon=config.HDBSCAN_CLUSTER_SELECTION_EPSILON,
    )
    labels: np.ndarray = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = int((labels == -1).sum())
    outlier_rate = n_outliers / len(labels) if len(labels) > 0 else 0.0
    logger.info(
        f'cross-lingual HDBSCAN: {n_clusters} clusters, '
        f'{n_outliers}/{len(labels)} outliers ({outlier_rate:.0%})'
    )
    if outlier_rate > 0.5:
        logger.warning(
            f'cross-lingual outlier rate {outlier_rate:.0%} > 50% — '
            f'consider lowering BERTOPIC_MIN_TOPIC_SIZE or HDBSCAN_CLUSTER_SELECTION_EPSILON'
        )

    return labels


def run_clustering(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    Two-stage language-aware clustering pipeline.

    Stage 1 — Per-language BERTopic:
      BERTopic is fit separately on each language's article slice, using the
      corresponding rows of the LaBSE embedding matrix. c-TF-IDF extracts topic
      keywords from monolingual text, so labels are clean and in the source language.
      Keywords are then translated to English via deep-translator (best-effort).

    Stage 2 — Cross-lingual HDBSCAN:
      UMAP + HDBSCAN runs on the full LaBSE embedding matrix (all languages together).
      LaBSE already maps semantically equivalent content to nearby vectors across
      languages, so this step produces language-agnostic cluster IDs without any
      translation. An EN article and an ES article about the same flood event will
      share the same cross_cluster_id.

    Output columns added to df:
      lang_topic_id       — BERTopic topic ID within per-language model (-1 = outlier)
      lang_topic_keywords — top-5 keywords in source language
      topic_keywords_en   — top-5 keywords translated to English
      cross_cluster_id    — language-agnostic HDBSCAN cluster (-1 = noise/outlier)
    """
    df = df.copy()
    df['lang_topic_id'] = -1
    df['lang_topic_keywords'] = ''
    df['topic_keywords_en'] = ''

    # ── Stage 1: per-language BERTopic ────────────────────────────────────────
    logger.info('=== Clustering Stage 1: per-language BERTopic ===')
    languages = sorted(df['language'].dropna().unique().tolist())

    topic_summary_rows: list[dict] = []

    for lang in languages:
        # use positional indices for numpy slicing, label index for df assignment
        positions = np.where((df['language'] == lang).values)[0]
        idx = df.index[positions]
        lang_texts = df.iloc[positions]['embed_text'].tolist()
        lang_embs = embeddings[positions]

        logger.info(f'[{lang}] running BERTopic on {len(lang_texts)} articles')
        kw_src, kw_en, topics = _run_bertopic_for_language(lang, lang_texts, lang_embs)

        df.loc[idx, 'lang_topic_id'] = topics
        df.loc[idx, 'lang_topic_keywords'] = [kw_src.get(t, '') for t in topics]
        df.loc[idx, 'topic_keywords_en'] = [kw_en.get(t, '') for t in topics]

        for topic_id, kw_str in kw_src.items():
            if topic_id == -1:
                continue
            n = sum(1 for t in topics if t == topic_id)
            topic_summary_rows.append({
                'language': lang,
                'lang_topic_id': topic_id,
                'lang_topic_keywords': kw_str,
                'topic_keywords_en': kw_en.get(topic_id, ''),
                'n_articles': n,
            })

    if topic_summary_rows:
        topic_info = pd.DataFrame(topic_summary_rows)
        topic_info.to_csv(config.TOPICS_PATH, index=False)
        logger.info(f'per-language topic info saved → {config.TOPICS_PATH}')

    # ── Stage 2: cross-lingual HDBSCAN ────────────────────────────────────────
    logger.info('=== Clustering Stage 2: cross-lingual HDBSCAN ===')
    cross_labels = _run_cross_lingual_hdbscan(embeddings, n_docs=len(df))
    df['cross_cluster_id'] = cross_labels

    return df
