# nlp/preprocessing.py
# handling text cleaning, language detection, and flood relevance filtering
# methods grounded in:
#   - Blomeier et al. (2024): keyword-based relevance filtering with MIN_FLOOD_HITS
#   - El Ouadi (2025): combining title + description + lead sentence for embedding
#   - stage_06_clean_deduplicate.py pattern from the existing CC pipeline

import re
import json
import logging
import hashlib
import importlib

import pandas as pd
import langid

# loading config modularly so any path change only touches nlp_config.py
config = importlib.import_module('config.nlp_config')

logger = logging.getLogger(__name__)


def load_data(path: str = None) -> pd.DataFrame:
    """loading the americas flood csv — uses INPUT_CSV from config if no path given"""
    path = path or config.INPUT_CSV
    logger.info(f'loading data from {path}')
    df = pd.read_csv(path)
    logger.info(f'loaded {len(df)} rows, columns: {list(df.columns)}')
    return df


def load_keyword_lexicon(path: str = None) -> dict:
    """loading the bilingual flood keyword lexicon from config/flood_keywords.json"""
    path = path or config.KEYWORD_LEXICON_PATH
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def clean_text(text: str) -> str:
    """
    normalising raw text before any NLP step:
    - stripping html tags (trafilatura may leave residual markup)
    - collapsing whitespace
    - removing zero-width characters
    mirroring stage_06 normalisation in the CC pipeline
    """
    if not isinstance(text, str):
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)           # stripping html
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)  # zero-width chars
    text = re.sub(r'\s+', ' ', text).strip()        # collapsing whitespace
    return text


def detect_language(text: str) -> str:
    """
    detecting language using langid (same library as CC pipeline stage_06)
    returning iso 639-1 code; falls back to 'unknown' on failure
    """
    try:
        lang, _ = langid.classify(text[:500])  # 500 chars is sufficient for detection
        return lang
    except Exception:
        return 'unknown'


def count_flood_hits(text: str, lang: str, lexicon: dict) -> int:
    """
    counting flood keyword matches using word boundaries to prevent
    substring false positives — same approach as stage_06 in the CC pipeline
    (e.g. avoids 'alud' matching inside 'resultado')
    """
    keywords = lexicon.get(lang, lexicon.get('en', []))
    text_lower = text.lower()
    hits = 0
    for kw in keywords:
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        if re.search(pattern, text_lower):
            hits += 1
    return hits


def build_embed_text(row: pd.Series) -> str:
    """
    constructing the text string to embed per article
    following El Ouadi (2025): title + description + opening sentence
    combined into a single string for LaBSE embedding
    """
    parts = []
    for field in config.FIELDS_TO_EMBED:
        val = row.get(field, '')
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return ' '.join(parts)


def deduplicate(df: pd.DataFrame, text_col: str = 'clean_text') -> pd.DataFrame:
    """
    deduplicating by SHA-256 hash of clean text within each flood event
    mirroring stage_06 deduplication logic from the CC pipeline
    """
    df = df.copy()
    df['_hash'] = df[text_col].apply(
        lambda t: hashlib.sha256(t.encode('utf-8', errors='ignore')).hexdigest()
    )
    before = len(df)
    df = df.drop_duplicates(subset=['flood_id', '_hash'])
    df = df.drop(columns=['_hash'])
    logger.info(f'deduplication: {before} → {len(df)} rows')
    return df


def run_preprocessing(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    running the full preprocessing pipeline:
    1. loading data (if not passed in)
    2. mapping ISO 639-2 language codes → ISO 639-1 (e.g. 'spa' → 'es')
    3. cleaning text from clean_text_relevant column
    4. filtering minimum character length
    5. filtering to supported languages (en, es)
    6. using pre-computed flood_term_hits from CSV (no recomputation needed)
    7. filtering duplicate articles using pre-computed is_content_duplicate flag
    8. building embed_text field (page_title + clean_text) for LaBSE
    returning enriched dataframe ready for embedding
    """
    if df is None:
        df = load_data()

    # mapping ISO 639-2 ('spa', 'eng') → ISO 639-1 ('es', 'en')
    # the CSV stores 3-letter codes in language_detected
    lang_col = config.LANGUAGE_COLUMN   # 'language_detected'
    df['language'] = df[lang_col].map(config.LANGUAGE_CODE_MAP).fillna(df[lang_col])
    logger.info(f'language codes mapped: {df["language"].value_counts().to_dict()}')

    # cleaning text from the pre-cleaned CSV column
    logger.info('cleaning text from clean_text_relevant...')
    df['clean_text'] = df[config.TEXT_COLUMN].apply(clean_text)

    # filtering minimum length
    df = df[df['clean_text'].str.len() >= config.MIN_CHAR_LENGTH].copy()
    logger.info(f'{len(df)} articles after length filter')

    # keeping only supported languages (en, es)
    df = df[df['language'].isin(config.SUPPORTED_LANGUAGES)].copy()
    logger.info(f'{len(df)} articles after language filter (en/es only)')

    # using pre-computed flood_term_hits from CSV (Blomeier et al. 2024)
    # validate the column exists and meets threshold; no recomputation needed
    if 'flood_term_hits' in df.columns:
        df['flood_hits'] = df['flood_term_hits'].astype(int)
        df = df[df['flood_hits'] >= config.MIN_FLOOD_HITS].copy()
        logger.info(f'{len(df)} articles after flood relevance filter (pre-computed hits, min={config.MIN_FLOOD_HITS})')
    else:
        # fallback: recompute from keyword lexicon if column absent
        logger.warning('flood_term_hits column not found — recomputing from lexicon')
        lexicon = load_keyword_lexicon()
        df['flood_hits'] = df.apply(
            lambda r: count_flood_hits(r['clean_text'], r['language'], lexicon), axis=1
        )
        df = df[df['flood_hits'] >= config.MIN_FLOOD_HITS].copy()
        logger.info(f'{len(df)} articles after flood relevance filter (recomputed, min={config.MIN_FLOOD_HITS})')

    # deduplication using pre-computed flag from CSV
    if 'is_content_duplicate' in df.columns:
        before = len(df)
        df = df[df['is_content_duplicate'].astype(str).str.lower() == 'false'].copy()
        logger.info(f'deduplication (pre-computed flag): {before} → {len(df)} rows')
    else:
        # fallback: hash-based deduplication
        df = deduplicate(df, text_col='clean_text')

    # building the combined field for embedding: title + body (El Ouadi 2025)
    df['embed_text'] = df.apply(build_embed_text, axis=1)

    logger.info(f'preprocessing complete: {len(df)} articles ready for embedding')
    return df
