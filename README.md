# Americas Flood NLP Pipeline

NLP analysis of flood-related news articles from the Americas (EN, ES, PT), extracted from Common Crawl. The core research goal is to measure **actionability** of flood coverage across regions (North America vs South America) and identify where CC systematically under-represents certain source types.

**Current dataset:** `verified_articles_clean.csv` — 612 rows, 580 after filtering, covering multiple flood events across the Americas (EN, ES, PT).  
**Flood event metadata:** `verified_floods_with_articles.csv` — flood event reference table.

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download pt_core_news_sm
```

---

## Running

```bash
# full pipeline
python run_nlp_pipeline.py

# with a custom input CSV
python run_nlp_pipeline.py --input /path/to/your_data.csv

# visualizations (run after pipeline)
python generate_visualizations.py
```

---

## File structure

```
Model/
├── config/
│   ├── nlp_config.py             ← all constants and paths — edit this first
│   └── flood_keywords.json       ← bilingual keyword lexicon (unused by default)
├── nlp/
│   ├── preprocessing.py          ← language mapping, text cleaning, deduplication
│   ├── actionability.py          ← sentence-level actionability scoring
│   ├── authority.py              ← source authority classification by domain
│   ├── framing.py                ← rule-based frame classification
│   └── clustering.py             ← group distributions + HDBSCAN clustering
├── data/
│   ├── verified_articles_clean.csv        ← current pipeline input (612 articles)
│   └── verified_floods_with_articles.csv  ← flood event metadata
├── output/                       ← pipeline outputs (gitignored)
│   ├── enriched.csv                       ← main output: one row per article
│   ├── sentences_actionability.csv        ← sentence-level detail
│   ├── group_stats_global_region.csv
│   ├── group_stats_country.csv
│   ├── group_stats_domain.csv
│   ├── group_stats_language.csv
│   ├── cluster_summary.csv
│   └── visualizations/                    ← PNG plots from generate_visualizations.py
├── logs/                         ← run logs (gitignored)
├── run_nlp_pipeline.py           ← main entry point
├── generate_visualizations.py    ← standalone visualization script
└── requirements.txt
```

---

## Input CSV schema

| column | type | notes |
|--------|------|-------|
| `article_id` | int | unique article identifier |
| `flood_id` | int | flood event id |
| `iso` | str | country ISO code e.g. `BRA` |
| `country` | str | full country name |
| `americas_region` | str | e.g. `South America` |
| `location` | str | subnational location |
| `river_basin` | str | river basin name |
| `start_date` | date | flood event start date |
| `end_date` | date | flood event end date |
| `language_detected` | str | ISO 639-2 code e.g. `spa`, `por` |
| `url` | str | article URL |
| `page_title` | str | article headline |
| `pub_date` | date | publication date |
| `clean_text` | str | pre-cleaned article body |

---

## Pipeline steps

| step | module | output columns added |
|------|--------|----------------------|
| 1 | `preprocessing.py` | `language` |
| 2 | `actionability.py` | `total_sentences`, `actionability_percentage` |
| 3 | `authority.py` | `domain`, `scope`, `source_type` |
| 4 | `framing.py` | `dominant_frame` |
| 5 | `clustering.py` | `global_region`, `data_cluster_id` |
| 6 | save | `output/enriched.csv` (580 rows × 21 columns) |

Sentence-level detail is saved separately to `output/sentences_actionability.csv`.

---

## Actionability scoring

Scoring operates at the **sentence level** and aggregates back to article level.

Each sentence is scored using:
- Keyword matching (imperative verbs, short-term urgency, long-term recovery, spatial anchors)
- spaCy morphology (imperative/subjunctive verb forms, auxiliary modals)
- SRL-lite (agent, action, location presence)
- Advice detection (`recommend`, `suggest`, `urge` and equivalents in ES/PT)

A weighted density score is normalised to `actionability_probability` (0–1), then binned into `actionability_score` (0/1/2). The article-level `actionability_percentage` is a weighted average across all sentences.

---

## Clustering design

**Stage 1 — Predefined group distributions**  
Computes `actionability_percentage` distributions by global region, country, domain, and language. Saved as `group_stats_*.csv`.

North America = US + Canada. All other Americas countries = South America.  
To change this, edit `NORTH_AMERICA_COUNTRIES` in `config/nlp_config.py`.

**Stage 2 — Data-driven HDBSCAN**  
Clusters articles on normalised actionability feature vectors. Output column: `data_cluster_id` (−1 = noise). Summary saved to `cluster_summary.csv`.

---

## Framing categories

Based on Entman (1993) and disaster journalism literature:

| frame | description |
|-------|-------------|
| `impact` | casualties, damage, displacement |
| `response` | rescue, evacuation, emergency aid |
| `accountability` | government failure, warnings, policy |
| `recovery` | reconstruction, resilience, long-term aid |

---

## Changing the dataset

Edit only `config/nlp_config.py`:
- `INPUT_CSV` — path to new CSV
- `FLOOD_REFERENCE_DATE` — flood onset date for temporal analysis

---

## Literature grounding

| module | key papers |
|--------|-----------|
| `preprocessing.py` | Blomeier et al. 2024 |
| `actionability.py` | Mostafiz et al. 2022; Zade et al. 2018; Jurafsky 2014; Zguir et al. 2025 |
| `authority.py` | Gordon 2000; Khawaja et al. 2025 |
| `framing.py` | Entman 1993; Zade et al. 2018; Khawaja et al. 2025 |
| `clustering.py` | Sit et al. 2020 (HDBSCAN) |
