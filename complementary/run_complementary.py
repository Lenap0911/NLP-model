# complementary/run_complementary.py
# main entry point — PT and ES only, no EN
# runs fully standalone — does NOT require the CC dataset
#
# Usage (run from Model/ directory):
#
#   # dry run first: RSS poll + archive search URL extraction, no full scraping
#   python complementary/run_complementary.py --flood-id 2 --dry-run
#
#   # full scrape for Brazil/Bahia floods (primary PT target)
#   python complementary/run_complementary.py --flood-id 2
#
#   # full scrape for Colombia floods (primary ES target)
#   python complementary/run_complementary.py --flood-id 3
#
#   # all active flood events (2, 3, 4)
#   python complementary/run_complementary.py --flood-id all

import sys
import os
import json
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from complementary.config import scrape_config as cfg
from complementary.scraper.rss_poller import poll_all_outlets
from complementary.scraper.archive_scraper import build_archive_candidates, build_cdx_candidates
from complementary.scraper.news_scraper import scrape_all
from complementary.scraper.formatter import format_all

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger('complementary')


def _load_outlets() -> list[dict]:
    with open(cfg.OUTLETS_JSON, encoding='utf-8') as f:
        return json.load(f)['outlets']


def _enabled_pt_es(outlets: list[dict]) -> list[dict]:
    """return only enabled outlets in the target languages (PT and ES)"""
    return [
        o for o in outlets
        if o.get('enabled', True) and o.get('language') in cfg.TARGET_LANGUAGES
    ]


def run(flood_id: int | None, dry_run: bool = False) -> None:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    label = f'flood_id={flood_id}' if flood_id is not None else 'standalone'
    logger.info(f'=== Complementary pipeline ({label}) — languages: {cfg.TARGET_LANGUAGES} ===')

    all_outlets = _load_outlets()
    active = _enabled_pt_es(all_outlets)
    logger.info(
        f'outlets: {len(active)} enabled PT/ES '
        f'({len(all_outlets) - len(active)} EN outlets skipped)'
    )

    # ── step 1: RSS polling (current articles) ────────────────────────────────
    logger.info('=== STEP 1: RSS POLLING ===')
    rss_candidates = poll_all_outlets(
        active,
        max_entries=cfg.RSS_MAX_ENTRIES,
        delay_s=cfg.REQUEST_DELAY_S,
    )
    logger.info(f'RSS: {len(rss_candidates)} candidates')

    # ── step 2a: site-search archive scraping (historical articles) ─────────────
    archive_candidates: list[dict] = []
    if flood_id is not None:
        logger.info('=== STEP 2a: ARCHIVE SCRAPING (site search) ===')
        archive_candidates = build_archive_candidates(active, flood_id=flood_id)
        logger.info(f'archive (site search): {len(archive_candidates)} candidates')
    else:
        logger.info('=== STEP 2a: ARCHIVE SCRAPING — skipped (no flood_id) ===')

    # ── step 2b: Wayback Machine CDX archive scraping ─────────────────────────
    cdx_candidates: list[dict] = []
    if flood_id is not None:
        logger.info('=== STEP 2b: CDX ARCHIVE SCRAPING (Wayback Machine) ===')
        cdx_candidates = build_cdx_candidates(all_outlets, flood_id=flood_id)
        logger.info(f'archive (CDX): {len(cdx_candidates)} candidates')
    else:
        logger.info('=== STEP 2b: CDX ARCHIVE SCRAPING — skipped (no flood_id) ===')

    all_candidates = rss_candidates + archive_candidates + cdx_candidates

    # deduplicate by URL before scraping
    seen: set[str] = set()
    unique_candidates = []
    for c in all_candidates:
        if c['url'] not in seen:
            seen.add(c['url'])
            unique_candidates.append(c)
    logger.info(f'total unique candidates: {len(unique_candidates)}')

    by_lang = {}
    for c in unique_candidates:
        by_lang[c['language']] = by_lang.get(c['language'], 0) + 1
    logger.info(f'language breakdown: {by_lang}')

    suffix = f'flood{flood_id}' if flood_id is not None else 'standalone'

    if dry_run:
        logger.info('dry-run mode — saving candidates, not scraping full articles')
        dry_path = os.path.join(cfg.OUTPUT_DIR, f'candidates_{suffix}.json')
        with open(dry_path, 'w', encoding='utf-8') as f:
            json.dump(unique_candidates, f, default=str, indent=2, ensure_ascii=False)
        logger.info(f'candidates saved → {dry_path}')
        return

    # ── step 3: full article scraping ────────────────────────────────────────
    logger.info('=== STEP 3: ARTICLE SCRAPING ===')
    articles = scrape_all(
        unique_candidates,
        min_chars=cfg.MIN_CHAR_LENGTH,
        timeout=cfg.REQUEST_TIMEOUT_S,
        delay_s=cfg.REQUEST_DELAY_S,
        max_articles=cfg.MAX_ARTICLES_PER_RUN,
    )

    # ── step 4: format to CC schema ───────────────────────────────────────────
    logger.info('=== STEP 4: FORMATTING TO CC SCHEMA ===')
    effective_flood_id = flood_id if flood_id is not None else 0
    df = format_all(articles, flood_id=effective_flood_id, doc_num_start=cfg.DOC_NUM_START)

    if df.empty:
        logger.warning('no articles passed quality gating')
        return

    out_path = os.path.join(cfg.OUTPUT_DIR, f'complementary_{suffix}.csv')
    df.to_csv(out_path, index=False)
    logger.info(f'saved {len(df)} articles → {out_path}')
    logger.info(f'language dist: {df["language_detected"].value_counts().to_dict()}')
    logger.info(f'flood_term_hits mean: {df["flood_term_hits"].mean():.2f}')
    logger.info(f'duplicate rate: {df["is_content_duplicate"].mean():.1%}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Complementary flood dataset — PT and ES only, no EN'
    )
    parser.add_argument(
        '--flood-id',
        default=None,
        help=(
            f'flood event ID to scrape. Active events: {cfg.ACTIVE_FLOOD_IDS}. '
            'Use "all" for all active events. Omit for RSS-only standalone run.'
        ),
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='collect candidate URLs only — do not scrape full articles',
    )
    args = parser.parse_args()

    if args.flood_id == 'all':
        for fid in cfg.ACTIVE_FLOOD_IDS:
            run(flood_id=fid, dry_run=args.dry_run)
    elif args.flood_id is None:
        run(flood_id=None, dry_run=args.dry_run)
    else:
        try:
            fid = int(args.flood_id)
        except ValueError:
            logger.error(f'invalid --flood-id "{args.flood_id}"')
            sys.exit(1)
        run(flood_id=fid, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
