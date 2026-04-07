# run_nlp_pipeline.py
# orchestrating the full NLP analysis pipeline for americas flood articles
# run this from the project root:
#   python run_nlp_pipeline.py [--input path/to/data.csv] [--skip-embed]
"""
import sys
import io
import logging
import argparse
import importlib
import os

# adding project root to path so config and nlp modules resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

config = importlib.import_module('config.nlp_config')

_stdout_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(_stdout_utf8),
        logging.FileHandler(os.path.join(config.LOG_DIR, 'nlp_pipeline.log'), encoding='utf-8'),
    ],
    force=True,   # clears handlers added by imported libs (numexpr, sentence_transformers)
)
logger = logging.getLogger('pipeline')
"""


import sys
import io
import logging
import argparse
import importlib
import os
from pathlib import Path

# adding project root to path so config and nlp modules resolve correctly
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

config = importlib.import_module('config.nlp_config')

# Ensure directories exist BEFORE creating FileHandler (works on macOS/Windows)
Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

_stdout_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(_stdout_utf8),
        logging.FileHandler(str(Path(config.LOG_DIR) / 'nlp_pipeline.log'), encoding='utf-8'),
    ],
    force=True,   # clears handlers added by imported libs (numexpr, sentence_transformers)
)
logger = logging.getLogger('pipeline')


def main(input_path: str = None, skip_embed: bool = False):
    from nlp.preprocessing import run_preprocessing
    from nlp.embeddings    import run_embeddings, load_embeddings, cross_lingual_similarity
    from nlp.actionability import run_actionability
    from nlp.clustering    import run_clustering

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # ── step 1: preprocessing ─────────────────────────────────────────────────
    logger.info('=== STEP 1: PREPROCESSING ===')
    df = run_preprocessing(input_path)

    # ── step 2: embeddings ────────────────────────────────────────────────────
    if skip_embed and os.path.exists(config.EMBEDDINGS_PATH):
        logger.info('=== STEP 2: LOADING PRECOMPUTED EMBEDDINGS ===')
        embeddings = load_embeddings()
        # making sure df and embeddings are aligned
        assert len(df) == embeddings.shape[0], \
            f'mismatch: {len(df)} rows but {embeddings.shape[0]} embeddings'
    else:
        logger.info('=== STEP 2: GENERATING LABSE EMBEDDINGS ===')
        df, embeddings = run_embeddings(df)

    # ── step 3: cross-lingual similarity (en ↔ es) ────────────────────────────
    logger.info('=== STEP 3: CROSS-LINGUAL SIMILARITY ===')
    cross_lingual_pairs = cross_lingual_similarity(embeddings, df)
    cross_lingual_pairs.to_csv(
        os.path.join(config.OUTPUT_DIR, 'cross_lingual_pairs.csv'), index=False
    )
    logger.info(f'saved {len(cross_lingual_pairs)} cross-lingual pairs')

    # ── step 4: actionability scoring ────────────────────────────────────────
    logger.info('=== STEP 4: ACTIONABILITY SCORING ===')
    df = run_actionability(df)

    # ── step 5: clustering + topic modelling ──────────────────────────────────
    logger.info('=== STEP 5: CLUSTERING + TOPIC MODELLING ===')
    df = run_clustering(df, embeddings)

    # ── step 6: saving enriched dataset ──────────────────────────────────────
    logger.info('=== STEP 6: SAVING ENRICHED DATASET ===')
    df.to_csv(config.ENRICHED_CSV_PATH, index=False)
    logger.info(f'enriched dataset saved -> {config.ENRICHED_CSV_PATH}')
    logger.info(f'final shape: {df.shape}')
    logger.info(f'columns: {list(df.columns)}')

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='americas flood NLP pipeline')
    parser.add_argument('--input',      type=str,  default=None,
                        help='path to input CSV (overrides config.INPUT_CSV)')
    parser.add_argument('--skip-embed', action='store_true',
                        help='load precomputed embeddings instead of re-encoding')
    args = parser.parse_args()

    main(input_path=args.input, skip_embed=args.skip_embed)
