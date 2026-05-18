# complementary/scraper/rss_poller.py
# polls RSS feeds from curated outlets and filters for flood-relevant articles
# returns a list of candidate URLs + metadata to hand off to news_scraper.py

import re
import time
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# flood relevance keywords used for RSS-level pre-filtering
# (full article text is checked again in formatter.py with the complete lexicon)
_FLOOD_RSS_KEYWORDS: dict[str, list[str]] = {
    'en': [
        'flood', 'flooding', 'inundation', 'flash flood', 'storm surge',
        'hurricane', 'tropical storm', 'heavy rain', 'landslide', 'mudslide',
        'evacuation', 'displaced', 'disaster relief',
    ],
    'es': [
        # high-specificity: one hit is enough
        'inundación', 'inundaciones', 'crecida', 'desbordamiento', 'riada',
        'avenida', 'anegamiento', 'aluvión', 'deslizamiento', 'huracán',
        # medium-specificity: still good signals
        'lluvias intensas', 'lluvias torrenciales', 'lluvias fuertes',
        'tormenta tropical', 'ciclón', 'ola invernal', 'damnificados',
        'evacuación', 'desplazados',
        # kept broad terms (require ≥2 hits total via threshold below)
        'lluvias', 'tormenta', 'desastre natural',
    ],
    'pt': [
        # high-specificity: one hit is enough
        'enchente', 'inundação', 'inundações', 'alagamento', 'transbordamento',
        'cheia', 'cheias', 'deslizamento', 'desabamento', 'barragem',
        # medium-specificity
        'chuvas intensas', 'chuvas fortes', 'chuvas torrenciais',
        'tempestade', 'furacão', 'desalojados', 'desabrigados',
        'defesa civil', 'catástrofe',
        # kept broad terms (require ≥2 hits total via threshold below)
        'chuvas', 'evacuação', 'desastre',
    ],
}

# minimum keyword hits in title+summary to pass the RSS pre-filter.
# 2 hits drastically reduces false positives from articles that use
# "chuvas" or "lluvias" as a metaphor or in unrelated weather context.
_RSS_MIN_HITS = 2


def _build_pattern(lang: str) -> re.Pattern:
    """compile a single regex from all RSS-level flood keywords for a language"""
    terms = _FLOOD_RSS_KEYWORDS.get(lang, _FLOOD_RSS_KEYWORDS['en'])
    escaped = [re.escape(t) for t in terms]
    return re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)


def _parse_date(entry) -> Optional[datetime]:
    """extract publish date from a feedparser entry, return UTC-aware datetime or None"""
    if hasattr(entry, 'published_parsed') and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
        try:
            return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    return None


def poll_outlet(outlet: dict, max_entries: int = 200) -> list[dict]:
    """
    Poll one outlet's RSS feed and return flood-relevant candidate articles.
    Each returned dict has: url, page_title, domain, language, country, pub_date, flood_hits_rss

    Returns an empty list if:
    - the outlet has no rss_url
    - feedparser is not installed
    - the feed is unreachable
    - no entries pass the flood relevance filter
    """
    rss_url = outlet.get('rss_url')
    if not rss_url:
        logger.debug(f'[{outlet["domain"]}] no RSS URL — skipping (scrape-only outlet)')
        return []

    try:
        import feedparser
    except ImportError:
        logger.warning('feedparser not installed — run: pip install feedparser')
        return []

    lang = outlet.get('language', 'en')
    pattern = _build_pattern(lang)

    logger.info(f'[{outlet["domain"]}] polling RSS: {rss_url}')
    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
        logger.warning(f'[{outlet["domain"]}] RSS fetch failed: {e}')
        return []

    if not feed.entries:
        logger.warning(f'[{outlet["domain"]}] RSS returned 0 entries')
        return []

    candidates = []
    for entry in feed.entries[:max_entries]:
        url = entry.get('link', '').strip()
        title = entry.get('title', '').strip()
        summary = entry.get('summary', '').strip()

        if not url:
            continue

        combined = f'{title} {summary}'
        hits = len(pattern.findall(combined))
        if hits < _RSS_MIN_HITS:
            continue

        candidates.append({
            'url': url,
            'page_title': title,
            'domain': outlet['domain'],
            'language': lang,
            'country': outlet['country'],
            'pub_date': _parse_date(entry),
            'flood_hits_rss': hits,
            'outlet_type': outlet.get('type', 'unknown'),
        })

    logger.info(f'[{outlet["domain"]}] {len(candidates)}/{min(len(feed.entries), max_entries)} entries passed flood filter')
    return candidates


def poll_all_outlets(
    outlets: list[dict],
    max_entries: int = 200,
    delay_s: float = 1.0,
) -> list[dict]:
    """Poll all outlets with RSS feeds, returning merged candidate list."""
    all_candidates: list[dict] = []
    rss_outlets = [o for o in outlets if o.get('rss_url')]
    logger.info(f'polling {len(rss_outlets)} outlets with RSS feeds')

    for i, outlet in enumerate(rss_outlets):
        candidates = poll_outlet(outlet, max_entries=max_entries)
        all_candidates.extend(candidates)
        if i < len(rss_outlets) - 1:
            time.sleep(delay_s)

    logger.info(f'RSS polling complete: {len(all_candidates)} candidate articles total')
    return all_candidates
