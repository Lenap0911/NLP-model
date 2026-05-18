# complementary/analysis/analyze_cc_coverage.py
# Step 1: understand what CC actually harvested before deciding what to add.
#
# Produces two reports:
#   1. domain_report.csv  — domain-level counts + language + presence in curated outlet list
#   2. gap_report.csv     — curated outlets that are absent from the CC corpus
#
# Run from the Model/ directory:
#   python complementary/analysis/analyze_cc_coverage.py

import os
import sys
import json
import logging

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from complementary.config import scrape_config as cfg

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def _load_cc(path: str) -> pd.DataFrame:
    logger.info(f'loading CC dataset from {path}')
    df = pd.read_csv(path, low_memory=False)
    logger.info(f'loaded {len(df)} rows, {df["language_detected"].value_counts().to_dict()}')
    return df


def _load_outlets(path: str) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        return json.load(f)['outlets']


def analyze_domain_coverage(df: pd.DataFrame, outlets: list[dict]) -> pd.DataFrame:
    """
    Per-domain breakdown of the CC corpus.
    Flags which domains are in the curated outlet list and which are not.
    """
    curated_domains = {o['domain'] for o in outlets}

    domain_counts = (
        df.groupby('domain')
        .agg(
            n_articles=('doc_num', 'count'),
            language=('language_detected', lambda x: x.mode()[0] if len(x) > 0 else 'unknown'),
            country=('country', lambda x: x.mode()[0] if len(x) > 0 else 'unknown'),
            flood_hits_mean=('flood_term_hits', 'mean'),
            flood_hits_min=('flood_term_hits', 'min'),
        )
        .reset_index()
        .sort_values('n_articles', ascending=False)
    )
    domain_counts['in_curated_list'] = domain_counts['domain'].isin(curated_domains)
    domain_counts['flood_hits_mean'] = domain_counts['flood_hits_mean'].round(2)
    return domain_counts


def analyze_language_distribution(df: pd.DataFrame) -> dict:
    """Language share of the CC corpus — the baseline the complementary dataset corrects."""
    total = len(df)
    dist = df['language_detected'].value_counts()
    return {code: {'count': int(n), 'pct': round(100 * n / total, 1)} for code, n in dist.items()}


def find_gaps(domain_report: pd.DataFrame, outlets: list[dict]) -> pd.DataFrame:
    """
    Which curated outlets are completely absent from the CC corpus?
    These are the primary targets for the complementary scraping pipeline.
    """
    cc_domains = set(domain_report['domain'])
    gaps = [
        {
            'domain': o['domain'],
            'name': o['name'],
            'language': o['language'],
            'country': o['country'],
            'type': o['type'],
            'has_rss': o['rss_url'] is not None,
            'reason_missing': 'not in CC corpus',
        }
        for o in outlets
        if o['domain'] not in cc_domains
    ]
    present = [
        {
            'domain': o['domain'],
            'name': o['name'],
            'language': o['language'],
            'country': o['country'],
            'type': o['type'],
            'has_rss': o['rss_url'] is not None,
            'reason_missing': 'present but may be under-represented',
        }
        for o in outlets
        if o['domain'] in cc_domains
    ]
    return pd.DataFrame(gaps + present)


def run_analysis() -> None:
    os.makedirs(cfg.ANALYSIS_DIR, exist_ok=True)

    if not os.path.exists(cfg.CC_PILOT_CSV):
        logger.error(f'CC dataset not found at {cfg.CC_PILOT_CSV}')
        logger.error('Update CC_PILOT_CSV in complementary/config/scrape_config.py')
        return

    df = _load_cc(cfg.CC_PILOT_CSV)
    outlets = _load_outlets(cfg.OUTLETS_JSON)

    # language distribution
    lang_dist = analyze_language_distribution(df)
    logger.info('CC language distribution:')
    for code, stats in lang_dist.items():
        logger.info(f'  {code}: {stats["count"]} articles ({stats["pct"]}%)')

    # domain coverage
    domain_report = analyze_domain_coverage(df, outlets)
    domain_path = os.path.join(cfg.ANALYSIS_DIR, 'domain_report.csv')
    domain_report.to_csv(domain_path, index=False)
    logger.info(f'domain report saved → {domain_path}')
    logger.info(f'unique domains in CC: {len(domain_report)}')
    logger.info(f'curated outlets found in CC: {domain_report["in_curated_list"].sum()}')

    # gap report
    gap_report = find_gaps(domain_report, outlets)
    gap_path = os.path.join(cfg.ANALYSIS_DIR, 'gap_report.csv')
    gap_report.to_csv(gap_path, index=False)

    absent = gap_report[gap_report['reason_missing'] == 'not in CC corpus']
    logger.info(f'{len(absent)} curated outlets absent from CC — gap report saved → {gap_path}')
    logger.info('top absent PT outlets:')
    for _, row in absent[absent['language'] == 'pt'].iterrows():
        logger.info(f'  {row["name"]} ({row["domain"]})')
    logger.info('top absent ES outlets:')
    for _, row in absent[absent['language'] == 'es'].iterrows():
        logger.info(f'  {row["name"]} ({row["domain"]})')


if __name__ == '__main__':
    run_analysis()
