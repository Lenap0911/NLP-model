# complementary/scraper/news_scraper.py
# extracts full article text from candidate URLs using newspaper3k
# handles download, parse, and basic quality gating before passing to formatter.py

import time
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def _fetch_html(url: str, timeout: int) -> Optional[str]:
    """fetch raw HTML with a browser-like User-Agent; returns None on error"""
    try:
        import requests
        resp = requests.get(
            url,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; FloodNewsBot/1.0)'},
        )
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.warning(f'HTTP fetch failed [{url}]: {e}')
        return None


def _extract_with_trafilatura(html: str, url: str) -> Optional[str]:
    """extract main article text with trafilatura; returns None if unavailable or empty"""
    try:
        import trafilatura
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        return text or None
    except ImportError:
        return None
    except Exception as e:
        logger.debug(f'trafilatura failed [{url}]: {e}')
        return None


def _extract_with_newspaper(url: str, language: str, timeout: int) -> tuple[Optional[str], Optional[str]]:
    """extract text+title with newspaper3k; returns (text, title) or (None, None)"""
    try:
        import newspaper
    except ImportError:
        return None, None
    try:
        art = newspaper.Article(url, language=language, request_timeout=timeout)
        art.download()
        art.parse()
        return (art.text or None), (art.title or None)
    except Exception as e:
        logger.debug(f'newspaper3k failed [{url}]: {e}')
        return None, None


def scrape_article(
    candidate: dict,
    min_chars: int = 100,
    timeout: int = 15,
) -> Optional[dict]:
    """
    Fetch full article text for one candidate URL.

    Tries trafilatura first (better at JS-heavy sites like G1), falls back to
    newspaper3k. candidate must have 'url' and 'language' keys.
    Returns a formatted dict or None if the article is too short.
    """
    url = candidate['url']
    lang = candidate.get('language', 'en')

    html = _fetch_html(url, timeout)
    title = candidate.get('page_title', '')
    text = None

    if html:
        text = _extract_with_trafilatura(html, url)

    if not text:
        np_text, np_title = _extract_with_newspaper(url, lang, timeout)
        text = np_text
        if np_title and not title:
            title = np_title

    if not text or len(text) < min_chars:
        logger.debug(f'too short or empty — dropping {url}')
        return None

    pub_date = candidate.get('pub_date')

    return {
        'url': url,
        'domain': candidate['domain'],
        'page_title': title,
        'clean_text_relevant': text,
        'pub_date': pub_date,
        'language': lang,
        'country': candidate['country'],
        'outlet_type': candidate.get('outlet_type', 'unknown'),
        'word_count': len(text.split()),
        'char_count': len(text),
        'flood_hits_rss': candidate.get('flood_hits_rss', 0),
    }


def scrape_all(
    candidates: list[dict],
    min_chars: int = 100,
    timeout: int = 15,
    delay_s: float = 1.0,
    max_articles: int = 5000,
) -> list[dict]:
    """
    Scrape all candidate URLs, applying quality gating and a polite delay.
    Deduplicates by URL before scraping to avoid redundant requests.
    """
    seen_urls: set[str] = set()
    unique = []
    for c in candidates:
        url = c['url']
        if url not in seen_urls:
            seen_urls.add(url)
            unique.append(c)

    logger.info(f'scraping {len(unique)} unique URLs (max {max_articles})')
    unique = unique[:max_articles]

    articles: list[dict] = []
    for i, candidate in enumerate(unique):
        result = scrape_article(candidate, min_chars=min_chars, timeout=timeout)
        if result:
            articles.append(result)
        if (i + 1) % 50 == 0:
            logger.info(f'  scraped {i + 1}/{len(unique)}, {len(articles)} kept so far')
        if i < len(unique) - 1:
            time.sleep(delay_s)

    logger.info(f'scraping complete: {len(articles)}/{len(unique)} articles kept')
    return articles
