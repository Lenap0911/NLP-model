# nlp_config.py
# central config for the americas flood nlp pipeline
# all paths, model names, language settings, and thresholds live here
# changing the dataset only requires editing this file

import os

# ── dataset ──────────────────────────────────────────────────────────────────
# current test dataset: flood-126 (Valencia 2024, Spain), 39 articles
# to switch dataset: update INPUT_CSV — no other file needs changing
# actual CSV columns:
#   doc_num, flood_id, country, url, domain, page_title, pub_date,
#   pub_in_window, timestamp, language_detected, language_match,
#   is_relevant, flood_term_hits, location_term_hits, subnational_hits,
#   location_specificity_score, word_count, char_count,
#   is_content_duplicate, signal_many_short_lines, signal_no_long_sentence,
#   signal_large_low_flood, clean_text_relevant
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
INPUT_CSV  = os.path.join(DATA_DIR, 'url_report_flood_126_relevant_with_text.csv')

# ── column name constants (matches actual CSV schema) ────────────────────────
TEXT_COLUMN     = 'clean_text_relevant'   # pre-cleaned article body
TITLE_COLUMN    = 'page_title'            # article headline
LANGUAGE_COLUMN = 'language_detected'    # ISO 639-2 codes e.g. 'spa', 'eng'

# ── temporal phase reference date ────────────────────────────────────────────
# used when CSV has no flood_date column — pub_date is compared against this
# to assign before / during / after phase (Dujardin et al. 2024)
# update this to the flood onset date for each new dataset
FLOOD_REFERENCE_DATE = '2024-10-29'   # Valencia 2024 DANA flood onset

# ── output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
EMBEDDINGS_PATH   = os.path.join(OUTPUT_DIR, 'labse_embeddings.npy')
ENRICHED_CSV_PATH = os.path.join(OUTPUT_DIR, 'flood_126_enriched.csv')
TOPICS_PATH       = os.path.join(OUTPUT_DIR, 'topic_model_results.csv')

# ── language mapping (ISO 639-2 → ISO 639-1) ─────────────────────────────────
# the CSV stores 3-letter codes ('spa', 'eng'); pipeline internals use 2-letter
# based on Blomeier et al. (2024) multilingual classification approach
# and El Ouadi (2025) LaBSE cross-lingual embeddings
LANGUAGE_CODE_MAP = {
    'spa': 'es',
    'eng': 'en',
    'por': 'pt',
    'fra': 'fr',
}
SUPPORTED_LANGUAGES = ['en', 'es']
LANGUAGE_LABELS     = {'en': 'English (North America)', 'es': 'Spanish (Latin America)'}

# ── embedding model ───────────────────────────────────────────────────────────
# LaBSE: Language-Agnostic BERT Sentence Embedding
# rationale: El Ouadi (2025) — enables cross-lingual similarity search,
# clustering, and comparison across English and Spanish flood articles
# without needing translation; maps both languages into a shared vector space
EMBEDDING_MODEL  = 'sentence-transformers/LaBSE'
EMBEDDING_DIM    = 768
EMBEDDING_BATCH  = 64   # adjust down if running on CPU

# ── preprocessing ─────────────────────────────────────────────────────────────
# flood keyword lexicon path — only used if recomputing hits from scratch
KEYWORD_LEXICON_PATH = os.path.join(
    os.path.dirname(__file__), 'flood_keywords.json'
)
# NOTE: flood_term_hits is pre-computed in the CSV — no recomputation needed
# MIN_FLOOD_HITS kept for fallback validation only
MIN_FLOOD_HITS = 2
# minimum article length in characters
MIN_CHAR_LENGTH = 100
# fields to embed: page_title + clean article body (El Ouadi 2025 approach)
# clean_text is produced by preprocessing.py from clean_text_relevant
FIELDS_TO_EMBED = ['page_title', 'clean_text']

# ── actionability scoring ─────────────────────────────────────────────────────
# inspired by Mostafiz et al. (2022) short-term vs long-term actionability
# and Zade et al. (2018) actionability bias framework
ACTIONABILITY_KEYWORDS = {
    'en': {
        'imperative_verbs': ['evacuate', 'shelter', 'avoid', 'call', 'move',
                             'prepare', 'stay', 'go', 'follow', 'contact'],
        'short_term':       ['now', 'immediately', 'emergency', 'warning',
                             'alert', 'urgent', 'danger', 'rescue'],
        'long_term':        ['recovery', 'rebuild', 'policy', 'resilience',
                             'mitigation', 'adaptation', 'relief', 'fund'],
        'spatial_anchors':  ['road', 'bridge', 'highway', 'shelter', 'zone',
                             'district', 'county', 'city', 'region', 'area'],
    },
    'es': {
        'imperative_verbs': ['evacuar', 'refugiarse', 'evitar', 'llamar',
                             'trasladarse', 'prepararse', 'seguir', 'contactar'],
        'short_term':       ['ahora', 'inmediatamente', 'emergencia', 'alerta',
                             'urgente', 'peligro', 'rescate', 'aviso'],
        'long_term':        ['recuperación', 'reconstruir', 'política',
                             'resiliencia', 'mitigación', 'adaptación', 'ayuda'],
        'spatial_anchors':  ['carretera', 'puente', 'refugio', 'zona',
                             'municipio', 'ciudad', 'región', 'área'],
    }
}

# ── topic modelling ───────────────────────────────────────────────────────────
# BERTopic approach: Dujardin et al. (2024) temporal-spatial topic design
# covers before / during / after disaster phases
BERTOPIC_MIN_TOPIC_SIZE = 5   # lowered from 10 — set to ~10% of corpus size
TEMPORAL_PHASES = ['before', 'during', 'after']  # assigned via pub_date vs flood_date

# ── semantic role labelling ───────────────────────────────────────────────────
# Jurafsky (2014) Chapter 21 — extract AGENT, THEME, GOAL from flood sentences
# spacy model to use per language
SPACY_MODELS = {
    'en': 'en_core_web_sm',
    'es': 'es_core_news_sm',
}

# ── clustering / diffusion ────────────────────────────────────────────────────
# Xu & Qiang (2022) distance-decay; Sit et al. (2020) DBSCAN spatial clusters
DBSCAN_EPS         = 0.7   # euclidean distance in UMAP-reduced space; increase for small datasets
DBSCAN_MIN_SAMPLES = 3     # lowered to allow clusters in small corpora
UMAP_N_COMPONENTS  = 5    # reduce before DBSCAN (Sit et al. 2020)
UMAP_N_NEIGHBORS   = 10   # reduced from 15 — must be < n_samples for small datasets

# ── logging ───────────────────────────────────────────────────────────────────
LOG_DIR  = os.path.join(os.path.dirname(__file__), '..', 'logs')
LOG_LEVEL = 'INFO'
