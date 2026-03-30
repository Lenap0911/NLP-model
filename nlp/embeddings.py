# nlp/embeddings.py
# generating multilingual semantic embeddings using LaBSE
# theoretical basis:
#   El Ouadi (2025): LaBSE for cross-lingual flood article comparison
#   Sit et al. (2020): embeddings as input to spatial clustering (DBSCAN/UMAP)
#   Dujardin et al. (2024): BERTopic requires dense sentence embeddings as input

import logging
import importlib

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

config = importlib.import_module('config.nlp_config')
logger = logging.getLogger(__name__)


def load_model(model_name: str = None) -> SentenceTransformer:
    """
    loading the LaBSE model from sentence-transformers
    LaBSE chosen over mBERT because:
    - trained specifically for cross-lingual sentence similarity (El Ouadi 2025)
    - maps english and spanish into a shared 768-dim vector space
    - allows direct cosine similarity comparison across languages
    """
    model_name = model_name or config.EMBEDDING_MODEL
    logger.info(f'loading embedding model: {model_name}')
    model = SentenceTransformer(model_name)
    return model


def generate_embeddings(
    texts: list,
    model: SentenceTransformer = None,
    batch_size: int = None,
    show_progress: bool = True,
) -> np.ndarray:
    """
    encoding a list of texts into LaBSE embeddings
    returns numpy array of shape (n_articles, 768)

    batching reduces memory pressure on large corpora —
    batch_size comes from config.EMBEDDING_BATCH (default 64)
    """
    if model is None:
        model = load_model()
    batch_size = batch_size or config.EMBEDDING_BATCH

    logger.info(f'encoding {len(texts)} texts in batches of {batch_size}...')
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # L2 normalise so cosine sim = dot product
        convert_to_numpy=True,
    )
    logger.info(f'embeddings shape: {embeddings.shape}')
    return embeddings


def save_embeddings(embeddings: np.ndarray, path: str = None) -> None:
    """saving embeddings to .npy so they don't need to be recomputed"""
    path = path or config.EMBEDDINGS_PATH
    np.save(path, embeddings)
    logger.info(f'embeddings saved to {path}')


def load_embeddings(path: str = None) -> np.ndarray:
    """loading precomputed embeddings from disk"""
    path = path or config.EMBEDDINGS_PATH
    embeddings = np.load(path)
    logger.info(f'loaded embeddings from {path}, shape: {embeddings.shape}')
    return embeddings


def cross_lingual_similarity(
    embeddings: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    computing pairwise cosine similarity between english and spanish articles
    grounding in El Ouadi (2025): cross-lingual comparison of flood coverage
    returns a summary dataframe of top en-es similar pairs

    this is the key analytical step for the americas bilingual comparison —
    finding articles in english and spanish that describe the same flood event
    from different media perspectives (Khawaja et al. 2025: Global North/South framing)
    """
    en_mask = df['language'] == 'en'
    es_mask = df['language'] == 'es'

    en_emb = embeddings[en_mask.values]
    es_emb = embeddings[es_mask.values]

    if en_emb.shape[0] == 0 or es_emb.shape[0] == 0:
        logger.info('cross-lingual similarity skipped: dataset has no articles in one or both languages')
        return pd.DataFrame(columns=['en_idx', 'es_idx', 'similarity', 'en_url', 'es_url'])

    # cosine similarity matrix (normalised embeddings → dot product)
    sim_matrix = en_emb @ es_emb.T   # shape (n_en, n_es)

    en_idx = df[en_mask].index.tolist()
    es_idx = df[es_mask].index.tolist()

    # collecting top matches above similarity threshold
    pairs = []
    threshold = 0.75
    for i, ei in enumerate(en_idx):
        top_j = np.argmax(sim_matrix[i])
        score  = sim_matrix[i, top_j]
        if score >= threshold:
            pairs.append({
                'en_idx':     ei,
                'es_idx':     es_idx[top_j],
                'similarity': float(score),
                'en_url':     df.loc[ei, 'url'] if 'url' in df.columns else '',
                'es_url':     df.loc[es_idx[top_j], 'url'] if 'url' in df.columns else '',
            })

    if not pairs:
        logger.info('no cross-lingual pairs found (dataset may be single-language)')
        return pd.DataFrame(columns=['en_idx', 'es_idx', 'similarity', 'en_url', 'es_url'])

    result = pd.DataFrame(pairs).sort_values('similarity', ascending=False)
    logger.info(f'found {len(result)} high-similarity cross-lingual pairs (threshold={threshold})')
    return result


def run_embeddings(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    main entry point: generating embeddings for preprocessed dataframe
    returns (df with embed_text confirmed, embeddings array)
    """
    model = load_model()
    texts = df['embed_text'].tolist()
    embeddings = generate_embeddings(texts, model=model)
    save_embeddings(embeddings)
    return df, embeddings
