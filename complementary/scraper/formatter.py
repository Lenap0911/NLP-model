# complementary/scraper/formatter.py
# converts scraped article dicts into rows that exactly match the CC CSV schema
# every column in CSV_COLUMNS must be present with the correct dtype
#
# quality signals are computed identically to the CC preprocessing pipeline:
#   signal_many_short_lines   → >50% of newline-split lines are < 50 chars
#   signal_no_long_sentence   → no sentence reaches MIN_LONG_SENTENCE_CHARS chars
#   signal_large_low_flood    → char_count > threshold AND flood_term_hits < threshold

import re
import hashlib
import logging
from datetime import datetime, timezone, date
from typing import Optional

import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from complementary.config import scrape_config as cfg

logger = logging.getLogger(__name__)

# ── flood keyword lexicon (mirrors CC pipeline, simplified inline) ─────────────
# Full lexicon lives in config/flood_keywords.json; this inline version is used
# for hit-counting in the complementary pipeline so the column is comparable.
_FLOOD_KEYWORDS: dict[str, list[str]] = {
    'en': [
        'flood', 'flooding', 'flash flood', 'inundation', 'storm surge',
        'hurricane', 'tropical storm', 'heavy rain', 'downpour', 'deluge',
        'landslide', 'mudslide', 'levee', 'dam', 'overflow', 'swamp',
        'emergency', 'evacuation', 'displaced', 'shelter', 'rescue',
    ],
    'es': [
        'inundación', 'inundaciones', 'crecida', 'desbordamiento', 'riada',
        'lluvia intensa', 'lluvias', 'tormenta', 'huracán', 'deslizamiento',
        'alud', 'emergencia', 'evacuación', 'desastre', 'alerta', 'rescate',
        'afectados', 'desplazados', 'presa', 'represa', 'desborde',
    ],
    'pt': [
        'enchente', 'inundação', 'inundações', 'alagamento', 'transbordamento',
        'chuva forte', 'chuvas', 'tempestade', 'furacão', 'deslizamento',
        'enxurrada', 'emergência', 'evacuação', 'desastre', 'alerta', 'resgate',
        'afetados', 'desalojados', 'barragem', 'represa', 'rompimento',
    ],
}

_LOCATION_KEYWORDS: dict[str, list[str]] = {
    'en': ['city', 'town', 'county', 'state', 'district', 'region', 'area', 'zone', 'road', 'bridge'],
    'es': ['ciudad', 'municipio', 'estado', 'región', 'zona', 'área', 'barrio', 'carretera', 'puente'],
    'pt': ['cidade', 'município', 'estado', 'região', 'zona', 'área', 'bairro', 'rodovia', 'ponte'],
}


def _count_keyword_hits(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(
        1 for kw in keywords
        if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower)
    )


def _detect_language_iso3(text: str, expected_lang: str) -> tuple[str, bool]:
    """
    Detect language using lingua-language-detector (same library as CC pipeline).
    Returns (iso3_code, language_match_bool).
    Falls back to expected_lang if lingua is not installed.
    """
    iso2_to_iso3 = cfg.LANG_ISO2_TO_ISO3
    expected_iso3 = iso2_to_iso3.get(expected_lang, 'eng')

    try:
        from lingua import Language, LanguageDetectorBuilder
        lang_map = {
            'en': Language.ENGLISH,
            'es': Language.SPANISH,
            'pt': Language.PORTUGUESE,
        }
        langs = [lang_map[l] for l in cfg.LANG_ISO2_TO_ISO3.keys() if l in lang_map]
        detector = LanguageDetectorBuilder.from_languages(*langs).build()
        result = detector.detect_language_of(text[:500])
        if result:
            iso2 = {'ENGLISH': 'en', 'SPANISH': 'es', 'PORTUGUESE': 'pt'}.get(result.name, expected_lang)
            iso3 = iso2_to_iso3.get(iso2, expected_iso3)
            return iso3, (iso3 == expected_iso3)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f'language detection failed: {e}')

    return expected_iso3, True


def _compute_quality_signals(text: str, flood_hits: int, char_count: int) -> dict:
    """compute the three boilerplate quality signals used in the CC pipeline"""
    lines = [l for l in text.split('\n') if l.strip()]
    short_line_ratio = (
        sum(1 for l in lines if len(l) < 50) / len(lines)
        if lines else 0.0
    )
    sentences = re.split(r'[.!?]+', text)
    has_long_sentence = any(len(s) >= cfg.MIN_LONG_SENTENCE_CHARS for s in sentences)

    return {
        'signal_many_short_lines': short_line_ratio > cfg.MAX_SHORT_LINE_RATIO,
        'signal_no_long_sentence': not has_long_sentence,
        'signal_large_low_flood': (
            char_count > cfg.LARGE_LOW_FLOOD_CHAR_THRESHOLD
            and flood_hits < cfg.LARGE_LOW_FLOOD_HIT_THRESHOLD
        ),
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()


def _pub_in_window(pub_date, flood_id: int) -> bool:
    """check if article pub_date falls within the flood event window"""
    event = cfg.FLOOD_EVENTS.get(flood_id, {})
    start_str = event.get('start_date')
    end_str = event.get('end_date')
    if pub_date is None or start_str is None or end_str is None:
        return False
    try:
        if isinstance(pub_date, datetime):
            pub = pub_date.date() if pub_date.tzinfo else pub_date.date()
        elif isinstance(pub_date, date):
            pub = pub_date
        else:
            pub = datetime.fromisoformat(str(pub_date)).date()
        start = date.fromisoformat(start_str)
        end = date.fromisoformat(end_str)
        return start <= pub <= end
    except Exception:
        return False


def format_article(
    article: dict,
    flood_id: int,
    doc_num: int,
    seen_hashes: set,
) -> Optional[dict]:
    """
    Convert one scraped article dict into a row matching the CC CSV schema.

    Returns None if the article fails flood relevance gating or is a duplicate.
    Adds the article's hash to seen_hashes if kept.
    """
    text = article.get('clean_text_relevant', '').strip()
    if not text:
        return None

    lang = article.get('language', 'en')
    flood_keywords = _FLOOD_KEYWORDS.get(lang, _FLOOD_KEYWORDS['en'])
    location_keywords = _LOCATION_KEYWORDS.get(lang, _LOCATION_KEYWORDS['en'])

    flood_hits = _count_keyword_hits(text, flood_keywords)
    if flood_hits < cfg.MIN_FLOOD_HITS:
        logger.debug(f'flood_hits={flood_hits} < {cfg.MIN_FLOOD_HITS} — dropping {article["url"]}')
        return None

    h = _sha256(text)
    is_duplicate = h in seen_hashes
    seen_hashes.add(h)

    location_hits = _count_keyword_hits(text, location_keywords)
    char_count = article.get('char_count', len(text))
    word_count = article.get('word_count', len(text.split()))

    # location specificity score: 0–1 normalised hit rate (capped at 1)
    loc_spec = min(1.0, round(location_hits / max(flood_hits, 1), 4))

    quality = _compute_quality_signals(text, flood_hits, char_count)

    pub_date = article.get('pub_date')
    pub_date_str = pub_date.strftime('%Y-%m-%d') if pub_date else None

    detected_iso3, lang_match = _detect_language_iso3(text, lang)

    return {
        'doc_num': doc_num,
        'flood_id': flood_id,
        'country': article.get('country', ''),
        'url': article['url'],
        'domain': article.get('domain', ''),
        'page_title': article.get('page_title', ''),
        'pub_date': pub_date_str,
        'pub_in_window': _pub_in_window(pub_date, flood_id),
        'timestamp': datetime.now(tz=timezone.utc).isoformat(),
        'language_detected': detected_iso3,
        'language_match': lang_match,
        'is_relevant': True,
        'flood_term_hits': flood_hits,
        'location_term_hits': location_hits,
        'subnational_hits': 0,  # subnational keyword list is flood-event-specific; set per event
        'location_specificity_score': loc_spec,
        'word_count': word_count,
        'char_count': char_count,
        'is_content_duplicate': is_duplicate,
        'signal_many_short_lines': quality['signal_many_short_lines'],
        'signal_no_long_sentence': quality['signal_no_long_sentence'],
        'signal_large_low_flood': quality['signal_large_low_flood'],
        'clean_text_relevant': text,
    }


def format_all(
    articles: list[dict],
    flood_id: int,
    doc_num_start: int = cfg.DOC_NUM_START,
) -> pd.DataFrame:
    """
    Format all scraped articles into a DataFrame with the CC column schema.
    Drops duplicates and articles below the flood relevance threshold.
    """
    seen_hashes: set = set()
    rows = []
    doc_num = doc_num_start

    for article in articles:
        row = format_article(article, flood_id, doc_num, seen_hashes)
        if row is not None:
            rows.append(row)
            doc_num += 1

    if not rows:
        logger.warning('no articles passed formatting — output will be empty')
        return pd.DataFrame(columns=cfg.CSV_COLUMNS)

    df = pd.DataFrame(rows, columns=cfg.CSV_COLUMNS)
    logger.info(
        f'formatted {len(df)} articles | '
        f'duplicates dropped: {sum(1 for r in rows if r["is_content_duplicate"])} | '
        f'lang dist: {df["language_detected"].value_counts().to_dict()}'
    )
    return df
