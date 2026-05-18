# Complementary Dataset Pipeline

Targeted scraping pipeline to address the structural under-representation of
Brazilian Portuguese (PT) and regional Spanish (ES) sources in the Common Crawl
(CC) harvest. Produces a CSV that concatenates directly with the CC pilot dataset
because it uses the identical column schema.

## Why this exists

CC's architecture prioritises high-linkage, frequently re-crawled domains,
which structurally over-represents English-language, high-traffic websites.
After preprocessing, the CC pilot sits at EN 51% / ES 36% / PT 13%.
Brazil — arguably the most flood-affected country in the region (Dartmouth
Flood Observatory) — accounts for 13% of articles despite having a substantial
PT journalism ecosystem.

This pipeline curates a list of flood-relevant outlets (see `config/outlets.json`),
polls their RSS feeds, and scrapes full article text using `newspaper3k`. The
output is formatted to match the CC CSV schema so the two datasets can be merged
without column renaming.

## Folder structure

```
complementary/
├── config/
│   ├── outlets.json          ← curated outlet list (EN/ES/PT) with RSS URLs
│   └── scrape_config.py      ← paths, thresholds, flood event dates
├── analysis/
│   ├── analyze_cc_coverage.py ← step 0: domain/language gap analysis
│   └── reports/              ← generated: domain_report.csv, gap_report.csv
├── scraper/
│   ├── rss_poller.py         ← poll RSS feeds, pre-filter by flood keywords
│   ├── news_scraper.py       ← newspaper3k article extraction
│   └── formatter.py          ← compute all CC schema columns + quality signals
├── output/                   ← generated: complementary_flood{id}.csv
├── run_complementary.py      ← main entry point
└── requirements.txt
```

## Setup

Install dependencies (separate from the main pipeline):

```bash
pip install -r complementary/requirements.txt
```

## Running

Run all commands from the `Model/` directory.

**Step 0 — analyze CC coverage gaps:**
```bash
python complementary/analysis/analyze_cc_coverage.py
```
Produces `analysis/reports/domain_report.csv` and `gap_report.csv`.
Check these before scraping to confirm which outlets are actually missing.

**Step 1 — dry run (RSS polling only, no full scraping):**
```bash
python complementary/run_complementary.py --flood-id 1 --dry-run
```
Saves candidate URLs to `output/rss_candidates_flood1.json`. Use this to verify
RSS feeds are reachable and returning flood-relevant articles before committing
to full scraping.

**Step 2 — full scrape for one flood event:**
```bash
python complementary/run_complementary.py --flood-id 1
```

**Step 3 — scrape all flood events:**
```bash
python complementary/run_complementary.py --flood-id all
```

## Before running: update flood event dates

Open `config/scrape_config.py` and fill in `start_date` and `end_date` for each
flood event. These are used to compute `pub_in_window`, which mirrors the CC
pipeline's temporal relevance flag.

```python
FLOOD_EVENTS = {
    1: {'start_date': '2023-04-01', 'end_date': '2023-04-30', 'notes': 'Brazil RS floods'},
    ...
}
```

## Output schema

The output CSV has the same columns as the CC pilot dataset in the same order:

| column | type | notes |
|---|---|---|
| `doc_num` | int | starts at 100,000 to avoid collision with CC doc_nums |
| `flood_id` | int | same flood event IDs as CC (1–6) |
| `country` | str | from outlet metadata |
| `url` | str | article URL |
| `domain` | str | outlet domain |
| `page_title` | str | from newspaper3k |
| `pub_date` | date | from RSS or newspaper3k |
| `pub_in_window` | bool | pub_date within flood event window |
| `timestamp` | datetime | time of scrape |
| `language_detected` | str | ISO 639-2 via lingua-language-detector |
| `language_match` | bool | detected language matches outlet's declared language |
| `is_relevant` | bool | always True (flood filter applied before saving) |
| `flood_term_hits` | int | keyword hits in full article text |
| `location_term_hits` | int | location keyword hits |
| `subnational_hits` | int | 0 by default — set per event with subnational keyword list |
| `location_specificity_score` | float | 0–1 normalised location hit rate |
| `word_count` | int | |
| `char_count` | int | |
| `is_content_duplicate` | bool | SHA-256 hash deduplication within this run |
| `signal_many_short_lines` | bool | boilerplate quality signal |
| `signal_no_long_sentence` | bool | boilerplate quality signal |
| `signal_large_low_flood` | bool | boilerplate quality signal |
| `clean_text_relevant` | str | full article body from newspaper3k |

## Merging with the CC dataset

```python
import pandas as pd

cc  = pd.read_csv('data/url_report_pilot.csv')
comp = pd.read_csv('complementary/output/complementary_flood1.csv')

merged = pd.concat([cc, comp], ignore_index=True)
merged.to_csv('data/merged_pilot.csv', index=False)
```

Both DataFrames have identical columns so `pd.concat` works without alignment.

## Limitations

- `subnational_hits` is set to 0 — subnational keyword lists are flood-event-specific
  and need to be added to `scrape_config.py` per event.
- Outlets behind hard paywalls (e.g. Folha full articles) will return partial text.
  `newspaper3k` extracts whatever is publicly accessible.
- RSS feeds only return recent articles. For historical flood events, RSS alone is
  insufficient — direct URL scraping from outlet archives is needed.
- Rate limiting: `REQUEST_DELAY_S` in `scrape_config.py` controls the polite delay
  between requests. Increase it if outlets return 429 errors.
