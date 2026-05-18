# complementary/scraper/news_scraper.py
# extracts full article text from candidate URLs using newspaper3k
# handles download, parse, and basic quality gating before passing to formatter.py

import time
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def _get_article(url: str, language: str, timeout: int) -> Optional[object]:
    """download and parse one URL with newspaper3k, return Article or None"""
    try:
        import newspaper
    except ImportError:
        raise ImportError('newspaper3k not installed — run: pip install newspaper3k')

    try:
        article = newspaper.Article(url, language=language, request_timeout=timeout)
        article.download()
        article.parse()
        return article
    except Exception as e:
        logger.warning(f'scrape failed [{url}]: {e}')
        return None


def scrape_article(
    candidate: dict,
    min_chars: int = 100,
    timeout: int = 15,
) -> Optional[dict]:
    """
    Fetch full article text for one candidate URL.

    candidate: dict from rss_poller.poll_outlet — must have 'url' and 'language'
    min_chars: drop articles shorter than this (mirrors CC MIN_CHAR_LENGTH)
    timeout:   request timeout in seconds

    Returns a dict with all scraped fields, or None if the article fails
    quality gating or cannot be downloaded.
    """
    url = candidate['url']
    lang = candidate.get('language', 'en')

    article = _get_article(url, lang, timeout)
    if article is None:
        return None

    text = (article.text or '').strip()
    if len(text) < min_chars:
        logger.debug(f'too short ({len(text)} chars) — dropping {url}')
        return None

    title = (article.title or candidate.get('page_title', '')).strip()

    pub_date = candidate.get('pub_date')
    if pub_date is None and article.publish_date:
        raw = article.publish_date
        if raw.tzinfo is None:
            raw = raw.replace(tzinfo=timezone.utc)
        pub_date = raw

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
