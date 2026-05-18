# complementary/scraper/archive_scraper.py
# date-bounded archive scraping for historical flood events
# used when RSS feeds don't reach back far enough (floods 2 and 3 are 2021-2022)
#
# approach: fetch outlet search pages with date-range URL parameters,
# extract article links with BeautifulSoup, then hand off to news_scraper.py

import re
import time
import logging
from urllib.parse import quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from complementary.config import scrape_config as cfg

logger = logging.getLogger(__name__)

# flood-related terms that appear in article URL slugs
# outlets like G1 and El Tiempo include the headline in the URL path,
# so filtering on slug terms removes the irrelevant sidebar/navigation links
# that get picked up when a search result page is scraped wholesale.
# accents are stripped in URL slugs, so use the unaccented versions.
_FLOOD_SLUG_TERMS: dict[str, list[str]] = {
    'pt': [
        'enchente', 'inundacao', 'inundacoes', 'alagamento', 'chuva', 'chuvas',
        'temporal', 'tempestade', 'deslizamento', 'desastre', 'tragedia',
        'emergencia', 'evacuacao', 'vitimas', 'mortes', 'desabamento',
        'petropolis', 'bahia', 'defesa-civil', 'catastrofe', 'calamidade',
    ],
    'es': [
        'inundacion', 'inundaciones', 'lluvia', 'lluvias', 'desastre',
        'emergencia', 'tormenta', 'evacuacion', 'damnificados', 'tragedia',
        'victimas', 'desbordamiento', 'crecida', 'deslave', 'alud',
        'defensa-civil', 'calamidad', 'catastrofe',
    ],
}


def _url_looks_like_flood(url: str, lang: str) -> bool:
    """
    Check whether the URL slug contains flood-related terms.
    Returns True (keep) or False (discard).
    If no slug terms are defined for the language, defaults to True (keep all).
    """
    terms = _FLOOD_SLUG_TERMS.get(lang)
    if not terms:
        return True
    path = urlparse(url).path.lower()
    return any(term in path for term in terms)


_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'es-419,es;q=0.9,pt-BR;q=0.8,pt;q=0.7,en;q=0.6',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}


def _build_search_url(
    template: str,
    query: str,
    start_date: str,
    end_date: str,
    page: int = 1,
) -> str:
    return template.format(
        query=quote_plus(query),
        start_date=start_date,
        end_date=end_date,
        page=page,
    )


def _is_article_url(href: str, domain: str, must_contain: str, must_exclude: list[str]) -> bool:
    """filter href to keep only likely article URLs for this outlet"""
    if not href or not href.startswith('http'):
        return False
    parsed = urlparse(href)
    if domain not in parsed.netloc:
        return False
    if must_contain and must_contain not in href:
        return False
    path = parsed.path
    # skip very short paths (homepage / section index)
    if len(path.strip('/').split('/')) < 2:
        return False
    for excl in must_exclude:
        if excl in href:
            return False
    return True


def _extract_links(
    html: str,
    base_url: str,
    domain: str,
    search_config: dict,
    lang: str,
) -> tuple[list[str], int]:
    """
    Parse a search result page and return (flood_relevant_urls, total_found).
    Applies two filters:
      1. URL structure filter (_is_article_url) — drops navigation/tag/author links
      2. URL slug filter (_url_looks_like_flood) — drops articles whose slug contains
         no flood-related terms (catches sidebar and related-article noise)
    Returns total_found so the caller can log the filter ratio.
    """
    soup = BeautifulSoup(html, 'lxml')
    must_contain = search_config.get('article_url_contains', domain)
    must_exclude  = search_config.get('article_url_excludes', [])

    all_links = set()
    for tag in soup.find_all('a', href=True):
        href = tag['href'].strip()
        if href.startswith('/'):
            href = urljoin(base_url, href)
        if _is_article_url(href, domain, must_contain, must_exclude):
            all_links.add(href)

    flood_links = [l for l in all_links if _url_looks_like_flood(l, lang)]
    return flood_links, len(all_links)


def search_outlet_archive(
    outlet: dict,
    query: str,
    start_date: str,
    end_date: str,
    max_pages: int = cfg.ARCHIVE_MAX_PAGES,
) -> list[dict]:
    """
    Fetch archive search result pages for one outlet + query and return
    candidate dicts (url, domain, language, country) ready for news_scraper.py.

    Iterates through paginated results up to max_pages.
    Stops early if a page returns no new links (end of results).
    """
    search_cfg = outlet.get('archive_search')
    if not search_cfg or not search_cfg.get('url_template'):
        return []

    domain   = outlet['domain']
    lang     = outlet['language']
    base_url = f'https://{domain}'
    template = search_cfg['url_template']

    all_links: set[str] = set()
    logger.info(
        f'[{domain}] archive search: "{query}" '
        f'({start_date} to {end_date}), max {max_pages} pages'
    )

    for page in range(1, max_pages + 1):
        url = _build_search_url(template, query, start_date, end_date, page)
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=cfg.REQUEST_TIMEOUT_S)
            if resp.status_code == 404:
                logger.debug(f'[{domain}] page {page} returned 404 — stopping')
                break
            if resp.status_code != 200:
                logger.warning(f'[{domain}] page {page} returned HTTP {resp.status_code}')
                break
            links, total_found = _extract_links(resp.text, base_url, domain, search_cfg, lang)
            logger.debug(
                f'[{domain}] page {page}: {len(links)}/{total_found} links passed slug filter'
            )
            new_links = set(links) - all_links
            if not new_links:
                logger.debug(f'[{domain}] page {page}: no new links — stopping')
                break
            all_links |= new_links
            logger.debug(f'[{domain}] page {page}: +{len(new_links)} links ({len(all_links)} total)')
        except requests.RequestException as e:
            logger.warning(f'[{domain}] page {page} request failed: {e}')
            break
        time.sleep(cfg.REQUEST_DELAY_S)

    candidates = [
        {
            'url': link,
            'domain': domain,
            'language': lang,
            'country': outlet['country'],
            'outlet_type': outlet.get('type', 'unknown'),
            'page_title': '',
            'pub_date': None,
            'flood_hits_rss': 0,
        }
        for link in all_links
    ]
    logger.info(f'[{domain}] archive search complete: {len(candidates)} candidate URLs for "{query}"')
    return candidates


def build_archive_candidates(
    outlets: list[dict],
    flood_id: int,
) -> list[dict]:
    """
    Run archive searches for all enabled PT/ES outlets for a given flood event.
    Uses the archive_queries defined in FLOOD_EVENTS[flood_id] in scrape_config.py.

    Returns merged candidate list deduplicated by URL.
    """
    event = cfg.FLOOD_EVENTS.get(flood_id, {})
    start_date = event.get('start_date')
    end_date   = event.get('end_date')
    queries_by_lang: dict[str, list[str]] = event.get('archive_queries', {})

    if not start_date or not end_date:
        logger.warning(
            f'flood_id={flood_id}: start_date/end_date not set in FLOOD_EVENTS — '
            f'update complementary/config/scrape_config.py'
        )
        return []

    if not queries_by_lang:
        logger.info(f'flood_id={flood_id}: no archive_queries defined — skipping archive scrape')
        return []

    enabled = [o for o in outlets if o.get('enabled', True) and o.get('language') in cfg.TARGET_LANGUAGES]
    all_candidates: list[dict] = []
    seen_urls: set[str] = set()

    for outlet in enabled:
        lang = outlet['language']
        queries = queries_by_lang.get(lang, [])
        if not queries:
            continue

        for query in queries:
            candidates = search_outlet_archive(
                outlet, query, start_date, end_date,
                max_pages=cfg.ARCHIVE_MAX_PAGES,
            )
            for c in candidates:
                if c['url'] not in seen_urls:
                    seen_urls.add(c['url'])
                    all_candidates.append(c)
            time.sleep(cfg.REQUEST_DELAY_S)

    logger.info(
        f'flood_id={flood_id}: archive scraping complete — '
        f'{len(all_candidates)} unique candidate URLs'
    )
    return all_candidates
