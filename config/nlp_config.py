# nlp_config.py
# central config for the americas flood nlp pipeline
# all paths, model names, language settings, and thresholds live here
# changing the dataset only requires editing this file

import os

# ── dataset ──────────────────────────────────────────────────────────────────
# verified dataset: 11 flood events (IDs 61,62,79,81,83,104,151,152,160,169,188)
# 524 articles, PT/ES only (no EN)
# to switch dataset: update INPUT_CSV — no other file needs changing
# actual CSV columns:
#   flood_id, country, url, page_title, pub_date, language_detected, clean_text
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
INPUT_CSV  = os.path.join(DATA_DIR, 'verified_articles_clean.csv')

# ── column name constants (matches actual CSV schema) ────────────────────────
TEXT_COLUMN     = 'clean_text'            # pre-cleaned article body
TITLE_COLUMN    = 'page_title'            # article headline
LANGUAGE_COLUMN = 'language_detected'    # ISO 639-2 codes e.g. 'spa', 'eng'

# ── output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
EMBEDDINGS_PATH   = os.path.join(OUTPUT_DIR, 'labse_embeddings.npy')
ENRICHED_CSV_PATH = os.path.join(OUTPUT_DIR, 'enriched.csv')   # overridden at runtime by run_nlp_pipeline.py
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
SUPPORTED_LANGUAGES = ['en', 'es', 'pt']
LANGUAGE_LABELS     = {
    'en': 'English (North America)',
    'es': 'Spanish (Latin America)',
    'pt': 'Portuguese (Brazil)',
}

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
    },
    'pt': {
        # Brazilian Portuguese — follows same taxonomy as EN/ES
        # imperative verbs: direct commands relevant to flood response
        'imperative_verbs': ['evacuar', 'refugiar', 'evitar', 'ligar', 'sair',
                             'preparar', 'ficar', 'seguir', 'contatar'],
        # short-term: immediate danger and response signals
        'short_term':       ['agora', 'imediatamente', 'emergência', 'alerta',
                             'urgente', 'perigo', 'resgate', 'aviso', 'socorro'],
        # long-term: recovery, resilience, policy language
        'long_term':        ['recuperação', 'reconstrução', 'política',
                             'resiliência', 'mitigação', 'adaptação', 'auxílio',
                             'assistência', 'reconstruir', 'reabilitação'],
        # spatial anchors: geographic grounding signals
        'spatial_anchors':  ['estrada', 'ponte', 'abrigo', 'zona', 'bairro',
                             'município', 'cidade', 'região', 'área', 'rua',
                             'rodovia', 'distrito'],
    },
}

# ── global north / global south classification ────────────────────────────────
# used by clustering.py to assign each article's country to a world-system tier
# Global North: high-income OECD countries (US, Canada)
# Global South: all other Americas countries in the dataset
# extend this dict when adding new flood events
GLOBAL_NORTH_COUNTRIES = {
    'United States of America',
    'United States',
    'USA',
    'Canada',
}
# all other countries in the dataset are treated as Global South by default

# ── clustering ────────────────────────────────────────────────────────────────
# K-Means on normalised actionability feature vectors
# tries each k value; saves a separate summary CSV per k + silhouette scores
KMEANS_K_VALUES     = [3, 4, 5]   # k values to try
KMEANS_N_INIT       = 20          # random initialisations per k
KMEANS_RANDOM_STATE = 42

# output paths for per-group summary tables (written alongside enriched CSV)
CLUSTER_STATS_DIR = OUTPUT_DIR   # group summary CSVs go into output/

# ── topic modelling (secondary / optional) ────────────────────────────────────
# BERTopic approach: Dujardin et al. (2024) temporal-spatial topic design
# only runs if embeddings are explicitly passed to run_topic_modeling()
BERTOPIC_MIN_TOPIC_SIZE = 5
BERTOPIC_NGRAM_RANGE   = (1, 2)
BERTOPIC_MIN_DF        = 2

# ── semantic role labelling ───────────────────────────────────────────────────
# Jurafsky (2014) Chapter 21 — extract AGENT, THEME, GOAL from flood sentences
# spacy model to use per language
SPACY_MODELS = {
    'en': 'en_core_web_sm',
    'es': 'es_core_news_sm',
    'pt': 'pt_core_news_sm',   # install: python -m spacy download pt_core_news_sm
}

# ── cross-lingual similarity ──────────────────────────────────────────────────
# CSLS_K: neighbourhood size for Cross-Lingual Similarity Local Scaling.
# Conneau et al. (2018) show k=10 optimal on MUSE benchmarks; lower k for small corpora.
CSLS_K = 10
# CROSS_LINGUAL_THRESHOLD_PERCENTILE: keep the top N% of best-match CSLS scores.
# Data-driven — adapts to corpus size, language mix, and topic diversity.
# 75 = top quartile of matches; raise to 80–90 for stricter pairing.
CROSS_LINGUAL_THRESHOLD_PERCENTILE = 75

# ── isotropy correction ───────────────────────────────────────────────────────
# All-but-Top postprocessing (Mu & Viswanath 2018): subtract corpus mean then
# remove top-D principal components so cosine distances span the full sphere.
# Applied after encoding, before similarity comparison and clustering.
ISOTROPY_D = 3   # 3 = recommended for corpora >200 docs; was 1 for the 39-article test set

# ── clustering / diffusion ────────────────────────────────────────────────────
# Xu & Qiang (2022) distance-decay; Sit et al. (2020) DBSCAN spatial clusters
DBSCAN_EPS         = 0.7   # euclidean distance in UMAP-reduced space; increase for small datasets
DBSCAN_MIN_SAMPLES = 3     # lowered to allow clusters in small corpora
UMAP_N_COMPONENTS  = 5    # reduce before DBSCAN (Sit et al. 2020)
UMAP_N_NEIGHBORS   = 10   # reduced from 15 — must be < n_samples for small datasets

# ── logging ───────────────────────────────────────────────────────────────────
LOG_DIR  = os.path.join(os.path.dirname(__file__), '..', 'logs')
LOG_LEVEL = 'INFO'
