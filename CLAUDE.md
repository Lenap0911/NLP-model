# CLAUDE.md — americas flood NLP pipeline

## what this subproject does

NLP analysis of flood-related news articles from the americas (EN + ES),
extracted from Common Crawl by the parent CC pipeline. The goal is to score
articles for **actionability** and discover **semantic clusters** of coverage,
enabling a bilingual (English / Spanish) comparison.

**Current dataset:** flood-126, Valencia 2024 (Spain) — 39 articles, all Spanish.
This is initial test data; the pipeline is designed for Americas datasets.

## setup

```bash
pip install pandas langid sentence-transformers numpy scikit-learn umap-learn bertopic spacy
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
```

## running

```bash
# full pipeline (uses INPUT_CSV from config/nlp_config.py)
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
│   ├── actionability.py       ← keyword scoring + SRL features + temporal phase
│   └── clustering.py          ← UMAP + DBSCAN + BERTopic
├── data/
│   └── url_report_flood_126_relevant_with_text.csv   ← current input
├── output/                    ← all pipeline outputs (gitignored)
├── logs/                      ← run logs (gitignored)
├── run_nlp_pipeline.py        ← main entry point
└── requirements.txt
```

## CSV schema (actual columns)

| column | type | notes |
|--------|------|-------|
| `doc_num` | int | row identifier |
| `flood_id` | int | flood event id (e.g. 126) |
| `country` | str | country name |
| `url` | str | article URL |
| `domain` | str | publisher domain |
| `page_title` | str | article headline — used in embedding |
| `pub_date` | date | publication date — used for temporal phase |
| `pub_in_window` | bool | article within flood event window |
| `timestamp` | datetime | crawl timestamp |
| `language_detected` | str | **ISO 639-2** code e.g. `spa`, `eng` |
| `language_match` | bool | language matches expected |
| `is_relevant` | bool | relevance flag |
| `flood_term_hits` | int | pre-computed flood keyword count |
| `location_term_hits` | int | pre-computed location keyword count |
| `subnational_hits` | int | sub-national location keyword count |
| `location_specificity_score` | float | 0–1 geographic specificity |
| `word_count` | int | article word count |
| `char_count` | int | article character count |
| `is_content_duplicate` | bool | pre-computed duplicate flag |
| `signal_many_short_lines` | bool | boilerplate quality signal |
| `signal_no_long_sentence` | bool | boilerplate quality signal |
| `signal_large_low_flood` | bool | boilerplate quality signal |
| `clean_text_relevant` | str | **pre-cleaned article body** — main text input |

## key pipeline adaptations for this CSV

- **Language codes**: `language_detected` uses ISO 639-2 (`spa`, `eng`). `preprocessing.py` maps these to ISO 639-1 (`es`, `en`) via `LANGUAGE_CODE_MAP` in config.
- **Text column**: `clean_text_relevant` is the pre-cleaned text (no `raw_text`). Preprocessing uses this directly.
- **Flood hits**: `flood_term_hits` is pre-computed — no recomputation. Falls back to keyword lexicon if column is absent.
- **Deduplication**: `is_content_duplicate` flag is pre-computed — used directly. Falls back to SHA-256 hash if column is absent.
- **Temporal phase**: No `flood_date` column. `pub_date` is compared against `FLOOD_REFERENCE_DATE` in config (set to flood onset date). Update `FLOOD_REFERENCE_DATE` for each new dataset.
- **Embedding input**: `page_title` + `clean_text` (derived from `clean_text_relevant`). No `description` or `lead_sentence` columns.

## to change the dataset

**only edit `config/nlp_config.py`**:
- `INPUT_CSV` — path to new CSV
- `FLOOD_REFERENCE_DATE` — flood onset date for temporal phase calculation
- No other file needs changing (column constants are also in config)
 To switch to an Americas dataset later: only edit INPUT_CSV and FLOOD_REFERENCE_DATE in config/nlp_config.py.

## literature grounding

| module            | key papers                                        |
|-------------------|---------------------------------------------------|
| preprocessing.py  | Blomeier et al. 2024, El Ouadi 2025               |
| embeddings.py     | El Ouadi 2025 (LaBSE), Khawaja et al. 2025        |
| actionability.py  | Mostafiz et al. 2022, Zade et al. 2018, Jurafsky 2014, Zguir et al. 2025 |
| clustering.py     | Sit et al. 2020 (DBSCAN), Dujardin et al. 2024 (BERTopic) |

## important constraints

- `config/nlp_config.py` is the single source of truth — never hardcode paths in modules
- `output/` and `logs/` are gitignored — share outputs via team storage
- embeddings take ~20 min to generate on CPU for ~10k articles — use `--skip-embed` after first run
- always run preprocessing before embeddings — the row order must match
