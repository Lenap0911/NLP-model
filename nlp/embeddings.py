# nlp/embeddings.py
# generating multilingual semantic embeddings using LaBSE
# theoretical basis:
#   El Ouadi (2025): LaBSE for cross-lingual flood article comparison
#   Sit et al. (2020): embeddings as input to spatial clustering (DBSCAN/UMAP)
#   Dujardin et al. (2024): BERTopic requires dense sentence embeddings as input

import logging
import importlib
import os

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


def isotropy_correction(embeddings: np.ndarray, d: int = None) -> np.ndarray:
    """
    All-but-Top postprocessing (Mu & Viswanath 2018):
    sentence embedding spaces are anisotropic — vectors cluster in a narrow cone,
    making cosine similarity artificially inflated and poorly discriminative.
    Fix: subtract the corpus mean, then remove the top-D dominant directions via SVD.
    Re-normalise to unit length so cosine sim = dot product is restored.
    Applied corpus-wide after encoding, before similarity thresholding and clustering.
    """
    d = d if d is not None else config.ISOTROPY_D
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    top_directions = Vt[:d]
    for direction in top_directions:
        centered -= centered.dot(direction)[:, None] * direction
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    corrected = centered / np.maximum(norms, 1e-9)
    logger.info(f'isotropy correction applied (d={d}): embeddings re-normalised')
    return corrected


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


def save_embedding_cache(
    embeddings: np.ndarray,
    doc_ids: list,
    path: str = None,
) -> None:
    """
    Save embeddings with doc_id index to .npz for incremental cache lookups.
    Allows new articles to be embedded without re-encoding the entire corpus.
    path defaults to EMBEDDINGS_PATH with .npz extension.
    """
    base_path = path or config.EMBEDDINGS_PATH
    npz_path = str(base_path).replace('.npy', '_cache.npz')
    np.savez(npz_path, embeddings=embeddings, doc_ids=np.array(doc_ids, dtype=object))
    logger.info(f'embedding cache saved to {npz_path} ({len(doc_ids)} docs)')


def load_embedding_cache(path: str = None) -> tuple[np.ndarray, list]:
    """
    Load incremental embedding cache. Returns (embeddings array, doc_ids list).
    Returns (None, []) if cache does not exist.
    """
    base_path = path or config.EMBEDDINGS_PATH
    npz_path = str(base_path).replace('.npy', '_cache.npz')
    if not os.path.exists(npz_path):
        logger.info(f'no embedding cache found at {npz_path}')
        return None, []
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data['embeddings']
    doc_ids = list(data['doc_ids'])
    logger.info(f'loaded embedding cache from {npz_path}, shape: {embeddings.shape}')
    return embeddings, doc_ids


def run_embeddings_incremental(
    df: pd.DataFrame,
    model: SentenceTransformer = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Incremental embedding: load cached embeddings for known doc_ids, encode only
    new articles. Falls back to full re-encoding if cache is empty or mismatched.
    Requires 'doc_id' column in df (or uses integer index as fallback).
    """
    id_col = 'doc_id' if 'doc_id' in df.columns else None
    doc_ids = list(df[id_col]) if id_col else list(df.index.astype(str))

    cached_emb, cached_ids = load_embedding_cache()
    cached_id_set = set(cached_ids)

    new_mask = [did not in cached_id_set for did in doc_ids]
    n_new = sum(new_mask)

    if cached_emb is not None and n_new == 0:
        logger.info('all doc_ids found in cache — skipping encoding')
        id_to_idx = {did: i for i, did in enumerate(cached_ids)}
        ordered_emb = np.array([cached_emb[id_to_idx[did]] for did in doc_ids])
        return df, isotropy_correction(ordered_emb)

    if cached_emb is None or n_new == len(doc_ids):
        logger.info('no usable cache — encoding full corpus')
        return run_embeddings(df, model=model)

    # Encode only new articles
    logger.info(f'encoding {n_new} new articles (cache has {len(cached_ids)} existing)')
    if model is None:
        model = load_model()

    new_indices = [i for i, is_new in enumerate(new_mask) if is_new]
    new_texts   = [df['embed_text'].iloc[i] for i in new_indices]
    new_emb     = generate_embeddings(new_texts, model=model)
    new_ids     = [doc_ids[i] for i in new_indices]

    # Merge cached + new
    all_embeddings = np.vstack([cached_emb, new_emb])
    all_ids        = cached_ids + new_ids

    save_embedding_cache(all_embeddings, all_ids)
    save_embeddings(all_embeddings)  # also update the .npy for backward compat

    # Return in original df order; apply correction on full merged corpus
    id_to_idx = {did: i for i, did in enumerate(all_ids)}
    ordered_emb = np.array([all_embeddings[id_to_idx[did]] for did in doc_ids])
    return df, isotropy_correction(ordered_emb)


def _csls_matrix(emb_a: np.ndarray, emb_b: np.ndarray, k: int = 10) -> np.ndarray:
    """
    CSLS(a, b) = 2·cos(a,b) − r(a) − r(b)
    r(x) = mean cosine of x to its k nearest neighbours in the other language.
    Penalises hub vectors (those unusually close to many others), improving
    cross-lingual retrieval precision by 4–8 pp over raw cosine (Conneau et al. 2018).
    Embeddings must be L2-normalised so cos(a,b) = dot(a,b).
    """
    sim = emb_a @ emb_b.T  # (n_a, n_b)

    k_a = min(k, emb_b.shape[0])
    r_a = np.partition(sim, -k_a, axis=1)[:, -k_a:].mean(axis=1)  # (n_a,)

    k_b = min(k, emb_a.shape[0])
    r_b = np.partition(sim.T, -k_b, axis=1)[:, -k_b:].mean(axis=1)  # (n_b,)

    return 2 * sim - r_a[:, None] - r_b[None, :]


def cross_lingual_similarity(
    embeddings: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute CSLS-scored cross-lingual pairs for all language combinations present.
    Threshold is data-driven: the Nth percentile of best-match CSLS scores across
    the full corpus (config.CROSS_LINGUAL_THRESHOLD_PERCENTILE, default 75).
    This adapts to corpus size, language mix, and topic diversity rather than
    relying on a hardcoded absolute score.

    Grounding: El Ouadi (2025) cross-lingual flood coverage comparison;
    Khawaja et al. (2025) Global North / South media framing.
    Returns a dataframe sorted by similarity descending.
    Columns: lang_a, lang_b, idx_a, idx_b, similarity, url_a, url_b
    """
    percentile = config.CROSS_LINGUAL_THRESHOLD_PERCENTILE
    languages_present = set(df['language'].dropna().unique())
    _EMPTY = pd.DataFrame(columns=['lang_a', 'lang_b', 'idx_a', 'idx_b',
                                   'similarity', 'url_a', 'url_b'])

    lang_pairs = [
        ('en', 'es'),  # English ↔ Spanish (core Americas comparison)
        ('en', 'pt'),  # English ↔ Portuguese (Brazil coverage)
        ('es', 'pt'),  # Spanish ↔ Portuguese (intra-Latin America comparison)
    ]

    # pass 1: build CSLS matrices, collect best-match score per source article
    computed = []
    all_best_scores = []
    for lang_a, lang_b in lang_pairs:
        if lang_a not in languages_present or lang_b not in languages_present:
            logger.debug(f'skipping {lang_a}↔{lang_b}: one or both languages absent')
            continue
        mask_a = df['language'] == lang_a
        mask_b = df['language'] == lang_b
        emb_a = embeddings[mask_a.values]
        emb_b = embeddings[mask_b.values]
        if emb_a.shape[0] == 0 or emb_b.shape[0] == 0:
            continue
        sim_matrix = _csls_matrix(emb_a, emb_b, k=config.CSLS_K)
        idx_a = df[mask_a].index.tolist()
        idx_b = df[mask_b].index.tolist()
        computed.append((lang_a, lang_b, sim_matrix, idx_a, idx_b))
        all_best_scores.extend(sim_matrix.max(axis=1).tolist())

    if not computed:
        logger.info('no cross-lingual pairs found (dataset may be single-language)')
        return _EMPTY

    # compute data-driven threshold from full score distribution
    scores = np.array(all_best_scores)
    threshold = float(np.percentile(scores, percentile))
    logger.info(
        f'CSLS best-match distribution — '
        f'p25={np.percentile(scores, 25):.3f} '
        f'p50={np.percentile(scores, 50):.3f} '
        f'p75={np.percentile(scores, 75):.3f} '
        f'p90={np.percentile(scores, 90):.3f}'
    )
    logger.info(f'cross-lingual threshold: p{percentile} = {threshold:.4f}')

    # pass 2: filter by threshold and build pair records
    all_pairs = []
    for lang_a, lang_b, sim_matrix, idx_a, idx_b in computed:
        pairs = []
        for i, ia in enumerate(idx_a):
            top_j = int(np.argmax(sim_matrix[i]))
            score  = float(sim_matrix[i, top_j])
            if score >= threshold:
                pairs.append({
                    'lang_a':     lang_a,
                    'lang_b':     lang_b,
                    'idx_a':      ia,
                    'idx_b':      idx_b[top_j],
                    'similarity': round(score, 4),
                    'url_a':      df.loc[ia, 'url'] if 'url' in df.columns else '',
                    'url_b':      df.loc[idx_b[top_j], 'url'] if 'url' in df.columns else '',
                })
        logger.info(f'{lang_a}↔{lang_b}: {len(pairs)} pairs above p{percentile} threshold')
        all_pairs.extend(pairs)

    if not all_pairs:
        return _EMPTY

    result = pd.DataFrame(all_pairs).sort_values('similarity', ascending=False)
    logger.info(f'total cross-lingual pairs: {len(result)}')
    return result


def run_embeddings(df: pd.DataFrame, model: SentenceTransformer = None) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Main entry point: generating embeddings for preprocessed dataframe.
    Returns (df with embed_text confirmed, embeddings array).
    Use run_embeddings_incremental() to skip re-encoding known doc_ids.
    """
    if model is None:
        model = load_model()
    texts = df['embed_text'].tolist()
    embeddings = generate_embeddings(texts, model=model)
    embeddings = isotropy_correction(embeddings)
    save_embeddings(embeddings)
    # Also update the incremental cache
    id_col = 'doc_id' if 'doc_id' in df.columns else None
    doc_ids = list(df[id_col]) if id_col else list(df.index.astype(str))
    save_embedding_cache(embeddings, doc_ids)
    return df, embeddings
