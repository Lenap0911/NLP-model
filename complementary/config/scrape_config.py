# complementary/config/scrape_config.py
# configuration for the targeted scraping pipeline
# mirrors the structure of ../config/nlp_config.py so the two datasets
# can be merged without column renaming

import os

# ── paths ─────────────────────────────────────────────────────────────────────
THIS_DIR     = os.path.dirname(__file__)
COMP_DIR     = os.path.dirname(THIS_DIR)
MODEL_DIR    = os.path.dirname(COMP_DIR)

OUTLETS_JSON     = os.path.join(THIS_DIR, 'outlets.json')
OUTPUT_DIR       = os.path.join(COMP_DIR, 'output')
OUTPUT_CSV       = os.path.join(OUTPUT_DIR, 'complementary_raw.csv')
ANALYSIS_DIR     = os.path.join(COMP_DIR, 'analysis', 'reports')

# parent CC dataset — used for domain coverage analysis
CC_PILOT_CSV     = os.path.join(MODEL_DIR, 'data', 'url_report_pilot.csv')

# ── schema constants (must match parent pipeline) ─────────────────────────────
# ISO 639-2 codes — same as CC pipeline
LANG_ISO2_TO_ISO3 = {'en': 'eng', 'es': 'spa', 'pt': 'por'}
LANG_ISO3_TO_ISO2 = {v: k for k, v in LANG_ISO2_TO_ISO3.items()}

# column order — identical to CC CSV so datasets concatenate cleanly
CSV_COLUMNS = [
    'doc_num', 'flood_id', 'country', 'url', 'domain', 'page_title',
    'pub_date', 'pub_in_window', 'timestamp', 'language_detected',
    'language_match', 'is_relevant', 'flood_term_hits', 'location_term_hits',
    'subnational_hits', 'location_specificity_score', 'word_count',
    'char_count', 'is_content_duplicate', 'signal_many_short_lines',
    'signal_no_long_sentence', 'signal_large_low_flood', 'clean_text_relevant',
]

# ── quality thresholds (mirrors CC pipeline) ──────────────────────────────────
MIN_CHAR_LENGTH  = 100    # articles shorter than this are dropped
MIN_FLOOD_HITS   = 2      # minimum flood keyword matches in full article text
MAX_SHORT_LINE_RATIO = 0.5    # signal_many_short_lines threshold
MIN_LONG_SENTENCE_CHARS = 80  # signal_no_long_sentence: no sentence ≥ this → signal
LARGE_LOW_FLOOD_CHAR_THRESHOLD  = 2000   # signal_large_low_flood: large article...
LARGE_LOW_FLOOD_HIT_THRESHOLD   = 2      # ...with fewer flood hits than this → signal

# ── language filter — complementary dataset targets PT and ES only ─────────────
# EN is already 51% of the CC corpus. Adding more EN articles would worsen the
# imbalance rather than correct it. All EN outlets in outlets.json are skipped.
TARGET_LANGUAGES = ['pt', 'es']

# ── scraping settings ─────────────────────────────────────────────────────────
RSS_MAX_ENTRIES       = 200    # max entries to read per RSS feed per run
ARCHIVE_MAX_PAGES     = 5      # max search result pages to fetch per outlet per query
REQUEST_TIMEOUT_S     = 15     # seconds before giving up on a URL
REQUEST_DELAY_S       = 1.5    # polite delay between consecutive requests (seconds)
MAX_ARTICLES_PER_RUN  = 5000   # safety cap across all outlets

# ── flood events ──────────────────────────────────────────────────────────────
DOC_NUM_START = 100_000

# Dates derived from CC pilot dataset pub_date analysis.
# Floods 2 and 3 are the primary archive scraping targets (PT and ES gaps).
# Windows are wider than the acute event to capture before/during/after coverage.
FLOOD_EVENTS: dict[int, dict] = {
    1: {
        'start_date': '2026-03-01', 'end_date': '2026-04-30',
        'notes': 'Caribbean/St Lucia — EN dominant, skip for complementary',
    },
    2: {
        'start_date': '2021-11-01', 'end_date': '2022-02-28',
        'notes': 'Brazil/Bahia floods 2021-22 — PRIMARY PT TARGET',
        'archive_queries': {
            'pt': ['enchente bahia', 'inundação bahia', 'chuvas bahia',
                   'desastre bahia', 'alagamento bahia'],
        },
    },
    3: {
        'start_date': '2022-04-01', 'end_date': '2022-10-31',
        'notes': 'Colombia floods 2022 — PRIMARY ES TARGET',
        'archive_queries': {
            'es': ['inundaciones colombia', 'lluvias colombia', 'ola invernal colombia',
                   'desastre colombia', 'damnificados colombia'],
        },
    },
    4: {
        'start_date': '2017-07-01', 'end_date': '2017-12-31',
        'notes': 'Panama floods 2017 — secondary ES target',
        'archive_queries': {
            'es': ['inundaciones panama', 'lluvias panama'],
        },
    },
    5: {
        'start_date': '2026-03-01', 'end_date': '2026-04-30',
        'notes': 'Trinidad & Tobago — EN dominant, skip for complementary',
    },
    6: {
        'start_date': '2022-03-01', 'end_date': '2022-09-30',
        'notes': 'Broad EN corpus — EN dominant, skip for complementary',
    },
}

# Flood IDs where the complementary dataset should actively add articles.
# Excludes EN-dominant events (1, 5, 6) where supplementing PT/ES is not the gap.
ACTIVE_FLOOD_IDS = [2, 3, 4]
