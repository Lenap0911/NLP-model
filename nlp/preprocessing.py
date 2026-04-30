# nlp/preprocessing.py
# handling text cleaning, language detection, and flood relevance filtering
# methods grounded in:
#   - Blomeier et al. (2024): keyword-based relevance filtering with MIN_FLOOD_HITS
#   - El Ouadi (2025): combining title + description + lead sentence for embedding
#   - stage_06_clean_deduplicate.py pattern from the existing CC pipeline

import re
import json
import difflib
import logging
import hashlib
import importlib
from functools import lru_cache

import pandas as pd

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


@lru_cache(maxsize=1)
def _build_lingua_detector():
    """builds lingua detector once, restricted to SUPPORTED_LANGUAGES for higher accuracy"""
    from lingua import Language, LanguageDetectorBuilder
    _lingua_lang_map = {
        'en': Language.ENGLISH,
        'es': Language.SPANISH,
        'pt': Language.PORTUGUESE,
        'fr': Language.FRENCH,
    }
    langs = [_lingua_lang_map[c] for c in config.SUPPORTED_LANGUAGES if c in _lingua_lang_map]
    return LanguageDetectorBuilder.from_languages(*langs).build()


def detect_language(text: str) -> str:
    """
    detecting language using lingua-py — more accurate than langid for regional
    Spanish (Caribbean, Andean, Rioplatense) and Portuguese varieties
    detector is restricted to SUPPORTED_LANGUAGES: narrowing candidates improves accuracy
    returns iso 639-1 code; falls back to 'unknown' if lingua not installed
    install: pip install lingua-language-detector
    """
    _iso_map = {'ENGLISH': 'en', 'SPANISH': 'es', 'PORTUGUESE': 'pt', 'FRENCH': 'fr'}
    try:
        detector = _build_lingua_detector()
        result = detector.detect_language_of(text[:500])
        return _iso_map.get(result.name, 'unknown') if result else 'unknown'
    except ImportError:
        logger.warning('lingua-language-detector not installed — run: pip install lingua-language-detector')
        return 'unknown'
    except Exception:
        return 'unknown'


def count_flood_hits(t ext: str, lang: str, lexicon: dict) -> int:
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


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def _strip_leading_title_repeat(title: str, body: str, threshold: float = 0.85) -> str:
    """
    removes the opening sentence of body when it closely matches title
    prevents title double-counting in LaBSE embedding
    uses startswith fast-path, then SequenceMatcher similarity (>= threshold)
    """
    if not title or not body:
        return body
    m = re.search(r'(?<=[.!?])\s', body)
    first_sent = body[:m.start()] if m else body[:len(title) + 60]
    norm_title = _normalize(title)
    norm_sent = _normalize(first_sent)
    if not norm_title:
        return body
    if norm_sent.startswith(norm_title):
        return body[len(first_sent):].lstrip()
    if difflib.SequenceMatcher(None, norm_title, norm_sent).ratio() >= threshold:
        return body[len(first_sent):].lstrip()
    return body


def build_embed_text(row: pd.Series) -> str:
    """
    constructing the text string to embed per article
    following El Ouadi (2025): title + description + opening sentence
    strips body opening sentence if it duplicates the title (inflates title weight)
    """
    parts = []
    title = ''
    for field in config.FIELDS_TO_EMBED:
        raw = row.get(field, '')
        val = ('' if pd.isna(raw) else str(raw)).strip()
        if not val:
            continue
        if field == config.TITLE_COLUMN:
            title = val
        elif field == 'clean_text':
            val = _strip_leading_title_repeat(title, val)
        if val:
            parts.append(val)
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


def _log_lang_dist(df: pd.DataFrame, step: str) -> None:
    dist = df['language'].value_counts().to_dict() if 'language' in df.columns else {}
    logger.info(f'[{step}] n={len(df)} | lang dist: {dist}')


def run_preprocessing(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    running the full preprocessing pipeline:
    1. loading data (if not passed in)
    2. mapping ISO 639-2 language codes → ISO 639-1 (e.g. 'spa' → 'es')
    3. cleaning text from clean_text_relevant column
    4. filtering minimum character length
    5. filtering to supported languages (en, es, pt)
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
    _log_lang_dist(df, 'after length filter')

    # keeping only supported languages (en, es, pt)
    df = df[df['language'].isin(config.SUPPORTED_LANGUAGES)].copy()
    _log_lang_dist(df, 'after language filter')

    # using pre-computed flood_term_hits from CSV (Blomeier et al. 2024)
    # validate the column exists and meets threshold; no recomputation needed
    if 'flood_term_hits' in df.columns:
        df['flood_hits'] = df['flood_term_hits'].astype(int)
        df = df[df['flood_hits'] >= config.MIN_FLOOD_HITS].copy()
        _log_lang_dist(df, f'after flood filter (pre-computed, min={config.MIN_FLOOD_HITS})')
    else:
        # fallback: recompute from keyword lexicon if column absent
        logger.warning('flood_term_hits column not found — recomputing from lexicon')
        lexicon = load_keyword_lexicon()
        df['flood_hits'] = df.apply(
            lambda r: count_flood_hits(r['clean_text'], r['language'], lexicon), axis=1
        )
        df = df[df['flood_hits'] >= config.MIN_FLOOD_HITS].copy()
        _log_lang_dist(df, f'after flood filter (recomputed, min={config.MIN_FLOOD_HITS})')

    # deduplication using pre-computed flag from CSV
    if 'is_content_duplicate' in df.columns:
        df = df[df['is_content_duplicate'].astype(str).str.lower() == 'false'].copy()
        _log_lang_dist(df, 'after deduplication (pre-computed flag)')
    else:
        # fallback: hash-based deduplication
        df = deduplicate(df, text_col='clean_text')
        _log_lang_dist(df, 'after deduplication (hash-based)')

    # building the combined field for embedding: title + body (El Ouadi 2025)
    df['embed_text'] = df.apply(build_embed_text, axis=1)

    logger.info(f'preprocessing complete: {len(df)} articles ready for embedding')
    return df
