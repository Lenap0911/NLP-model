# CLAUDE.md — americas flood NLP pipeline

## what this subproject does

NLP analysis of flood-related news articles from the americas (EN, ES, PT),
extracted from Common Crawl. The core research goal is to measure **actionability**
of flood coverage across regions (Global North vs Global South) and identify
where CC systematically under-represents certain source types.

**Current dataset:** verified_articles_clean_text.csv — 524 articles, 11 flood events, PT/ES only.
**Full event metadata:** verified_floods_with_articles.csv — 38 flood events across the Americas.

## setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download pt_core_news_sm
```

## running

```bash
# full pipeline
python run_nlp_pipeline.py

# with a custom dataset path
python run_nlp_pipeline.py --input /path/to/your_data.csv

# skip re-encoding if embeddings already computed
python run_nlp_pipeline.py --skip-embed
```

## file structure

```
Model/
├── config/
│   ├── nlp_config.py          ← ALL constants/paths live here — edit this first
│   └── flood_keywords.json    ← bilingual keyword lexicon (fallback only)
├── nlp/
│   ├── preprocessing.py       ← language mapping, clean, filter, deduplicate
│   ├── embeddings.py          ← LaBSE encoding + cross-lingual similarity
│   ├── actionability.py       ← keyword scoring + SRL + morphology (friend's module)
│   └── clustering.py          ← group distributions + data-driven HDBSCAN
├── data/
│   ├── verified_articles_clean_text.csv   ← current article input
│   └── verified_floods_with_articles.csv  ← flood event metadata (38 events)
├── output/                    ← all pipeline outputs (gitignored)
│   ├── group_stats_global_region.csv
│   ├── group_stats_country.csv
│   ├── group_stats_domain.csv
│   ├── group_stats_language.csv
│   └── cluster_summary.csv
├── logs/                      ← run logs (gitignored)
├── run_nlp_pipeline.py        ← main entry point
└── requirements.txt
```

## CSV schema (actual columns)

| column | type | notes |
|--------|------|-------|
| `flood_id` | int | flood event id |
| `country` | str | country name |
| `url` | str | article URL |
| `page_title` | str | article headline — used in embedding |
| `pub_date` | date | publication date — used for temporal phase |
| `language_detected` | str | **ISO 639-2** code e.g. `spa`, `por` |
| `clean_text` | str | **pre-cleaned article body** — main text input |

## pipeline steps

| step | module | description |
|------|--------|-------------|
| 1 | preprocessing.py | language mapping, text cleaning, flood filter, dedup |
| 2 | embeddings.py | LaBSE encoding (skip with `--skip-embed`) |
| 3 | actionability.py | keyword scoring + SRL + morphology → `output_actionability` |
| 4 | authority.py | source authority scoring |
| 5 | framing.py | frame classification |
| 6 | clustering.py | group distributions + data-driven clustering |

## clustering design

`clustering.py` takes `output_actionability` as input — **no embeddings required**.

### Stage 1 — Predefined categorical groupings
Each article is assigned to a category and actionability score distributions
are computed. Saved as CSV tables in `output/`:

| grouping | output file | what it shows |
|----------|-------------|---------------|
| Global North / Global South | `group_stats_global_region.csv` | does CC coverage in richer countries differ in actionability? |
| Website domain | `group_stats_domain.csv` | which outlets produce more/less actionable content? |
| Country | `group_stats_country.csv` | per-country actionability distribution |
| Language | `group_stats_language.csv` | PT vs ES vs EN actionability gap |

Global North = US + Canada. All other Americas countries = Global South.
To add countries to Global North, edit `GLOBAL_NORTH_COUNTRIES` in config.

### Stage 2 — Data-driven HDBSCAN
Clusters articles on normalised actionability sub-scores (imperative, short-term,
long-term, spatial, past_tense_ratio — whichever are present in `output_actionability`).
Finds natural groupings in actionability space, e.g.:
- high-imperative + high-spatial = evacuation-order articles
- high-long-term + low-imperative = policy/recovery reporting

Output column: `data_cluster_id` (-1 = noise). Summary: `cluster_summary.csv`.

### Stage 3 — BERTopic (optional, secondary)
Only runs if explicitly called: `run_topic_modeling(df, embeddings)`.
Not part of the main pipeline. Adds `lang_topic_id`, `lang_topic_keywords`,
`topic_keywords_en` columns.

## expected input schema for clustering (output_actionability)

`run_clustering()` expects these columns (minimum):

| column | required | notes |
|--------|----------|-------|
| `actionability_score` | yes | composite score per article |
| `country` | yes | for Global North/South assignment |
| `language` | yes | ISO 639-1 |
| `url` or `domain` | yes | domain extracted from url if domain absent |
| `imperative_score` / `imperative_count` | optional | better clustering if present |
| `short_term_score` / `short_term_count` | optional | |
| `long_term_score` / `long_term_count` | optional | |
| `spatial_score` / `spatial_count` | optional | |
| `past_tense_ratio` | optional | |

## key adaptations for current CSV

- **Language codes**: `language_detected` uses ISO 639-2 (`spa`, `por`). Maps to ISO 639-1 via `LANGUAGE_CODE_MAP` in config.
- **Flood hits**: `flood_term_hits` absent → falls back to keyword lexicon recomputation.
- **Deduplication**: `is_content_duplicate` absent → falls back to SHA-256 hash deduplication.

## to change the dataset

**only edit `config/nlp_config.py`**:
- `INPUT_CSV` — path to new CSV
- `FLOOD_REFERENCE_DATE` — flood onset date for temporal phase

## literature grounding

| module | key papers |
|--------|-----------|
| preprocessing.py | Blomeier et al. 2024, El Ouadi 2025 |
| embeddings.py | El Ouadi 2025 (LaBSE), Khawaja et al. 2025 |
| actionability.py | Mostafiz et al. 2022, Zade et al. 2018, Jurafsky 2014, Zguir et al. 2025 |
| clustering.py | Sit et al. 2020 (HDBSCAN), Dujardin et al. 2024 (BERTopic, optional) |

## important constraints

- `config/nlp_config.py` is the single source of truth — never hardcode paths in modules
- `output/` and `logs/` are gitignored — share outputs via team storage
- embeddings take ~20 min on CPU — use `--skip-embed` after first run
- clustering.py does not require embeddings; topic modeling does
