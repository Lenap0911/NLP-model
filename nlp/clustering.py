# nlp/clustering.py
# semantic clustering of flood articles in embedding space
# theoretical basis:
#   Sit et al. (2020): UMAP dimensionality reduction + DBSCAN for spatial clusters
#   Dujardin et al. (2024): BERTopic for temporal-spatial topic discovery
#   Xu & Qiang (2022): distance-decay in information diffusion — clusters reflect proximity

import logging
import importlib

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

config = importlib.import_module('config.nlp_config')
logger = logging.getLogger(__name__)


def reduce_with_umap(embeddings: np.ndarray, n_components: int = None) -> np.ndarray:
    """
    reducing LaBSE embeddings from 768 dims to n_components
    before applying DBSCAN — following Sit et al. (2020) pipeline:
    LSTM embeddings → UMAP → DBSCAN to identify impact areas

    UMAP preserves local structure (cluster topology) better than PCA
    for high-dimensional sentence embeddings
    """
    try:
        import umap
    except ImportError:
        raise ImportError('umap-learn not installed — run: pip install umap-learn')

    n_components = n_components or config.UMAP_N_COMPONENTS
    logger.info(f'reducing embeddings to {n_components} dims with UMAP...')
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=config.UMAP_N_NEIGHBORS,
        metric='cosine',
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)
    logger.info(f'UMAP complete: shape {reduced.shape}')
    return reduced


def cluster_with_dbscan(
    reduced_embeddings: np.ndarray,
    eps: float = None,
    min_samples: int = None,
) -> np.ndarray:
    """
    clustering articles in UMAP-reduced embedding space using DBSCAN
    rationale for DBSCAN over K-means:
    - flood article clusters have irregular density (Sit et al. 2020)
    - articles from the same geographic impact zone cluster tightly
    - outlier articles (noise, -1 label) are meaningfully different —
      genuinely atypical coverage that doesn't fit any cluster
    - no need to pre-specify K: number of topics emerges from data
    """
    eps         = eps or config.DBSCAN_EPS
    min_samples = min_samples or config.DBSCAN_MIN_SAMPLES

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(reduced_embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    logger.info(f'DBSCAN: {n_clusters} clusters, {n_noise} noise articles (eps={eps}, min_samples={min_samples})')

    if n_clusters > 1:
        non_noise = labels != -1
        sil = silhouette_score(reduced_embeddings[non_noise], labels[non_noise])
        logger.info(f'silhouette score (non-noise): {sil:.4f}')

    return labels


def run_bertopic(
    texts: list,
    embeddings: np.ndarray = None,
    language: str = 'multilingual',
) -> tuple:
    """
    fitting BERTopic on article texts with precomputed LaBSE embeddings
    following Dujardin et al. (2024) who used BERTopic for the 2021 EU floods
    passing precomputed embeddings avoids re-encoding (uses LaBSE output directly)

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
        ngram_range=(1, 2),   # unigrams + bigrams for richer topic labels
        min_df=2,             # token must appear in at least 2 docs
    )

    logger.info('fitting BERTopic...')
    try:
        from hdbscan import HDBSCAN
        hdbscan_model = HDBSCAN(
            min_cluster_size=config.BERTOPIC_MIN_TOPIC_SIZE,
            min_samples=1,
            cluster_selection_epsilon=0.5,
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

    topic_info = topic_model.get_topic_info()
    logger.info(f'BERTopic found {len(topic_info) - 1} topics (excl. outlier topic -1)')
    return topic_model, topics, probs


def run_clustering(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    main entry point: running full clustering pipeline
    1. UMAP reduction (768 → 5 dims)
    2. DBSCAN clustering on reduced embeddings
    3. BERTopic topic modelling on texts with LaBSE embeddings
    adds columns: umap_cluster, topic_id to df
    """
    # UMAP + DBSCAN following Sit et al. (2020)
    reduced       = reduce_with_umap(embeddings)
    cluster_labels = cluster_with_dbscan(reduced)
    df['umap_cluster'] = cluster_labels

    # BERTopic following Dujardin et al. (2024)
    texts = df['embed_text'].tolist()
    topic_model, topics, _ = run_bertopic(texts, embeddings=embeddings)
    df['topic_id'] = topics

    # saving topic info separately
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(config.TOPICS_PATH, index=False)
    logger.info(f'topic info saved to {config.TOPICS_PATH}')

    return df
