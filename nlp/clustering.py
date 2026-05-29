import os
import sys
import logging
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

config = importlib.import_module('config.nlp_config')
logger = logging.getLogger(__name__)

# ── actionability feature columns used for data-driven clustering ─────────────
# tries all of these; uses whichever are present in the input df
_SCORE_COLS = [
    'actionability_percentage',
    'mean_actionability_probability',
    'mean_imperative_count',
    'mean_short_term_count',
    'mean_long_term_count',
    'mean_spatial_count',
    'mean_advice',
    'mean_srl_complete',
    'mean_has_agent',
    'mean_has_action',
    'mean_has_location',
]

# function-word stopwords per language for BERTopic c-TF-IDF (topic modeling only)
_STOPWORDS: dict[str, list[str]] = {
    'es': [
        'de','la','el','en','y','a','los','del','se','las','por','un','con',
        'una','su','al','lo','le','da','ha','que','no','es','pero','más',
        'esto','este','esta','han','sus','como','para','también','son','fue',
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


# ── Stage 1a: Global North / South assignment ─────────────────────────────────

def assign_global_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns each article to 'Global North' or 'Global South' based on country.
    Uses GLOBAL_NORTH_COUNTRIES from config; everything else is Global South.
    Adds column: global_region
    """
    if 'country' not in df.columns:
        logger.warning('no country column — global_region set to unknown')
        df['global_region'] = 'unknown'
        return df

    df['global_region'] = df['country'].apply(
        lambda c: 'Global North' if str(c).strip() in config.GLOBAL_NORTH_COUNTRIES
        else 'Global South'
    )
    counts = df['global_region'].value_counts().to_dict()
    logger.info(f'global region assignment: {counts}')
    return df


# ── Stage 1b: Predefined group summary tables ─────────────────────────────────

def _group_stats(df: pd.DataFrame, group_col: str, score_col: str = 'actionability_percentage') -> pd.DataFrame:
    """
    Computes distribution stats of score_col within each value of group_col.
    Returns a summary df with count, mean, median, std, min, max per group,
    plus share of total articles.
    """
    if score_col not in df.columns:
        logger.warning(f'{score_col} not in df — cannot compute group stats for {group_col}')
        return pd.DataFrame()

    stats = (
        df.groupby(group_col)[score_col]
        .agg(count='count', mean='mean', median='median', std='std',
             min='min', max='max')
        .reset_index()
    )
    stats['pct_of_total'] = (stats['count'] / len(df) * 100).round(1)
    stats = stats.sort_values('mean', ascending=False).reset_index(drop=True)
    return stats


def compute_group_distributions(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Stage 1: computes actionability score distributions for three predefined groupings:
      - global_region  (Global North / Global South)
      - domain         (website domain extracted from url, or domain column if present)
      - country

    Returns a dict of summary DataFrames keyed by group name.
    Each is also saved to output/ as a CSV.
    """
    # extract domain from url if a domain column isn't already present
    if 'domain' not in df.columns and 'url' in df.columns:
        df['domain'] = df['url'].str.extract(r'https?://(?:www\.)?([^/]+)/')
        logger.info('domain column extracted from url')

    groups = {}
    for col in ('global_region', 'domain', 'country', 'language'):
        if col not in df.columns:
            logger.warning(f'grouping column {col!r} not found — skipping')
            continue
        stats = _group_stats(df, col)
        if stats.empty:
            continue
        groups[col] = stats
        out_path = os.path.join(config.CLUSTER_STATS_DIR, f'group_stats_{col}.csv')
        stats.to_csv(out_path, index=False, encoding='utf-8')
        logger.info(f'group stats ({col}) saved → {out_path}')
        logger.info(f'  {col} breakdown:\n{stats.to_string(index=False)}')

    return groups


# ── Stage 2: data-driven HDBSCAN on actionability features ────────────────────

def _build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Builds a normalised feature matrix from whatever actionability score columns
    are present in the df. Requires at least actionability_score.
    Returns (matrix, used_column_names).
    """
    available = [c for c in _SCORE_COLS if c in df.columns]
    if not available:
        raise ValueError(
            'no actionability score columns found — '
            'run_actionability must produce at least actionability_score'
        )
    if len(available) == 1:
        logger.warning(
            f'only one feature column ({available[0]}) available — '
            'data-driven clustering will be low-quality; '
            'provide sub-score columns for better results'
        )

    X = df[available].fillna(0).astype(float).values
    # z-score normalise so no single column dominates by scale
    std = X.std(axis=0)
    std[std == 0] = 1  # avoid division by zero for constant columns
    X = (X - X.mean(axis=0)) / std
    return X, available


def run_data_driven_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2: HDBSCAN on normalised actionability feature vectors.

    No embeddings or dimensionality reduction — features are already low-dimensional
    (up to ~6 score columns). HDBSCAN finds natural groupings in the actionability
    space, e.g. a cluster of high-spatial / high-imperative articles (evacuation orders)
    vs a cluster of high-long_term / low-imperative articles (policy reporting).

    Adds column: data_cluster_id  (-1 = noise / outlier)
    Also saves cluster_summary.csv with mean feature profile per cluster.
    """
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        logger.warning('hdbscan not installed — skipping data-driven clustering')
        df['data_cluster_id'] = -1
        return df

    X, feature_cols = _build_feature_matrix(df)
    logger.info(f'data-driven clustering on {X.shape[0]} articles × {len(feature_cols)} features: {feature_cols}')

    clusterer = HDBSCAN(
        min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=config.HDBSCAN_MIN_SAMPLES,
        cluster_selection_epsilon=config.HDBSCAN_CLUSTER_SELECTION_EPS,
    )
    labels = clusterer.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = int((labels == -1).sum())
    logger.info(f'HDBSCAN: {n_clusters} clusters, {n_outliers}/{len(labels)} noise points')

    df = df.copy()
    df['data_cluster_id'] = labels

    # cluster summary: mean feature profile + dominant country/language per cluster
    summary_rows = []
    for cid in sorted(set(labels)):
        mask = labels == cid
        label = 'noise' if cid == -1 else f'cluster_{cid}'
        row = {'cluster': label, 'n_articles': int(mask.sum())}

        # mean actionability features
        for col in feature_cols:
            if col in df.columns:
                row[col] = round(float(df.loc[mask, col].mean()), 4)

        # dominant metadata
        for meta_col in ('country', 'language', 'global_region', 'domain'):
            if meta_col in df.columns:
                top = df.loc[mask, meta_col].value_counts()
                row[f'top_{meta_col}'] = top.index[0] if len(top) else ''
                row[f'top_{meta_col}_pct'] = round(top.iloc[0] / mask.sum() * 100, 1) if len(top) else 0.0

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    out_path = os.path.join(config.CLUSTER_STATS_DIR, 'cluster_summary.csv')
    summary.to_csv(out_path, index=False, encoding='utf-8')
    logger.info(f'cluster summary saved → {out_path}')
    logger.info(f'\n{summary.to_string(index=False)}')

    return df


# ── Stage 3 (optional): BERTopic topic modeling ───────────────────────────────

def run_topic_modeling(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    Optional secondary analysis — only call this explicitly if you want topic labels.
    Runs per-language BERTopic using precomputed LaBSE embeddings.
    Adds columns: lang_topic_id, lang_topic_keywords, topic_keywords_en

    This is NOT part of the main run_clustering() call.
    Call separately: df = run_topic_modeling(df, embeddings)
    """
    try:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError:
        logger.warning('bertopic not installed — topic modeling skipped')
        return df

    df = df.copy()
    df['lang_topic_id'] = -1
    df['lang_topic_keywords'] = ''
    df['topic_keywords_en'] = ''

    text_col = 'embed_text' if 'embed_text' in df.columns else 'clean_text'
    languages = sorted(df['language'].dropna().unique().tolist())

    for lang in languages:
        positions = np.where((df['language'] == lang).values)[0]
        if len(positions) < config.BERTOPIC_MIN_TOPIC_SIZE:
            logger.warning(f'[{lang}] {len(positions)} docs < min_topic_size — skipping')
            continue

        idx = df.index[positions]
        lang_texts = df.iloc[positions][text_col].tolist()
        lang_embs = embeddings[positions]

        stopwords = sorted({w for l in [lang] for w in _STOPWORDS.get(l, [])})
        vectorizer = CountVectorizer(
            stop_words=stopwords,
            ngram_range=config.BERTOPIC_NGRAM_RANGE,
            min_df=config.BERTOPIC_MIN_DF,
        )
        try:
            from hdbscan import HDBSCAN as _HDBSCAN
            hdbscan_model = _HDBSCAN(
                min_cluster_size=config.BERTOPIC_MIN_TOPIC_SIZE,
                min_samples=1,
                prediction_data=True,
            )
            model = BERTopic(
                language='multilingual',
                min_topic_size=config.BERTOPIC_MIN_TOPIC_SIZE,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer,
                calculate_probabilities=False,
                verbose=False,
            )
        except ImportError:
            model = BERTopic(
                language='multilingual',
                min_topic_size=config.BERTOPIC_MIN_TOPIC_SIZE,
                vectorizer_model=vectorizer,
                calculate_probabilities=False,
                verbose=False,
            )

        logger.info(f'[{lang}] BERTopic on {len(lang_texts)} articles')
        topics, _ = model.fit_transform(lang_texts, embeddings=lang_embs)

        kw_src: dict[int, str] = {}
        for tid in set(topics):
            if tid == -1:
                kw_src[-1] = 'outlier'
                continue
            words = model.get_topic(tid) or []
            kw_src[tid] = ', '.join(w for w, _ in words[:5])

        df.loc[idx, 'lang_topic_id'] = topics
        df.loc[idx, 'lang_topic_keywords'] = [kw_src.get(t, '') for t in topics]

        # best-effort translation
        def _translate(s: str) -> str:
            if lang == 'en' or not s or s == 'outlier':
                return s
            try:
                from deep_translator import GoogleTranslator
                result = GoogleTranslator(source=lang, target='en').translate(s)
                return result or s
            except Exception:
                return s

        df.loc[idx, 'topic_keywords_en'] = [_translate(kw_src.get(t, '')) for t in topics]

    topic_path = os.path.join(config.CLUSTER_STATS_DIR, 'topic_model_results.csv')
    df[['lang_topic_id', 'lang_topic_keywords', 'topic_keywords_en']].to_csv(
        topic_path, index=False, encoding='utf-8'
    )
    logger.info(f'topic model results saved → {topic_path}')
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def run_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-stage clustering pipeline. Takes output_actionability dataframe directly —
    no embeddings required.

    Stage 1 — Predefined categorical grouping:
      Assigns each article to Global North / Global South, then computes actionability
      score distributions by global_region, country, domain, and language.
      Saves group_stats_<group>.csv for each grouping.

    Stage 2 — Data-driven HDBSCAN:
      Clusters articles on normalised actionability feature vectors to find natural
      groupings in the actionability space (e.g. evacuation-heavy vs recovery-heavy).
      Saves cluster_summary.csv with feature profiles per cluster.

    For optional BERTopic topic modeling, call run_topic_modeling(df, embeddings)
    separately after this function.

    Output columns added to df:
      global_region    — 'Global North' or 'Global South'
      data_cluster_id  — HDBSCAN cluster id (-1 = noise/outlier)
    """
    logger.info('=== Clustering Stage 1: predefined group distributions ===')
    df = assign_global_region(df)
    compute_group_distributions(df)

    logger.info('=== Clustering Stage 2: data-driven HDBSCAN ===')
    df = run_data_driven_clustering(df)

    return df
