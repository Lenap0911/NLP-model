# run_nlp_pipeline.py
# orchestrating the full NLP analysis pipeline for americas flood articles
# run this from the project root:
#   python run_nlp_pipeline.py [--input path/to/data.csv] [--skip-embed]


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


def main(input_path: str = None):
    from nlp.preprocessing import run_preprocessing
    from nlp.actionability import run_actionability
    from nlp.authority     import run_authority
    from nlp.framing       import run_framing
    from nlp.clustering    import run_clustering

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Derive output CSV name from input filename so multi-event runs don't
    # overwrite each other (e.g. flood_65_input.csv -> flood_65_enriched.csv)
    if input_path:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        enriched_name = stem.replace('_input', '') + '_enriched.csv'
        config.ENRICHED_CSV_PATH = os.path.join(config.OUTPUT_DIR, enriched_name)
    logger.info(f'enriched output -> {config.ENRICHED_CSV_PATH}')

    # ── step 1: preprocessing ─────────────────────────────────────────────────
    logger.info('=== STEP 1: PREPROCESSING ===')
    df = run_preprocessing(path=input_path)

    # ── step 2: actionability scoring ────────────────────────────────────────
    logger.info('=== STEP 2: ACTIONABILITY SCORING ===')
    df = run_actionability(df)

    # ── step 4: source authority scoring ─────────────────────────────────────
    logger.info('=== STEP 4: SOURCE AUTHORITY SCORING ===')
    df = run_authority(df)

    # ── step 5: frame classification ──────────────────────────────────────────
    logger.info('=== STEP 5: FRAME CLASSIFICATION ===')
    df = run_framing(df)

    # ── step 6: clustering ────────────────────────────────────────────────────
    logger.info('=== STEP 6: CLUSTERING ===')
    df = run_clustering(df)

    # ── step 7: saving enriched dataset ──────────────────────────────────────
    # clustering and visualizations run separately in clustering_analysis.ipynb
    logger.info('=== STEP 7: SAVING ENRICHED DATASET ===')
    df.to_csv(config.ENRICHED_CSV_PATH, index=False)
    logger.info(f'enriched dataset saved -> {config.ENRICHED_CSV_PATH}')
    logger.info(f'final shape: {df.shape}')
    logger.info(f'columns: {list(df.columns)}')

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='americas flood NLP pipeline')
    parser.add_argument('--input',      type=str,  default=None,
                        help='path to input CSV (overrides config.INPUT_CSV)')
    args = parser.parse_args()

    main(input_path=args.input)
