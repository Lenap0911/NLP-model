# Americas Flood NLP Pipeline

NLP analysis of flood-related news articles from the Americas (EN, ES, PT), extracted from Common Crawl. The core research goal is to measure **actionability** of flood coverage — the degree to which articles contain protective action guidance — across languages, regions, and source types, using the Protective Action Decision Model (PADM) as a theoretical framework.

**Current dataset:** `verified_articles_clean.csv` — 612 rows, 607 after filtering, covering 44 flood events across 17 countries (EN, ES, PT).  
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
# full pipeline — produces enriched.csv and all stats
python run_nlp_pipeline.py

# with a custom input CSV
python run_nlp_pipeline.py --input /path/to/your_data.csv
```

Visualisations, tables, and the pipeline diagram are generated from `notebook_tests/` (see below).

---

## File structure

```
Model/
├── config/
│   ├── nlp_config.py              ← all constants, paths, and keyword lists — edit this first
│   ├── keywords.ipynb             ← exploratory keyword frequency analysis (development only)
│   └── en_search_keywords.txt     ← raw English keyword list used in keywords.ipynb
│
├── nlp/                           ← pipeline modules (one per stage)
│   ├── preprocessing.py           ← language mapping, text cleaning, deduplication
│   ├── actionability.py           ← sentence-level actionability scoring (keywords + spaCy + SRL)
│   ├── authority.py               ← source authority classification by domain lookup
│   ├── framing.py                 ← rule-based frame classification (impact/response/accountability/recovery)
│   └── clustering.py              ← predefined group distributions + K-Means clustering (k=3)
│
├── data/
│   ├── verified_articles_clean.csv        ← pipeline input (612 articles, 607 post-filter)
│   └── verified_floods_with_articles.csv  ← flood event metadata (44 events, used for joining)
│
├── output/                        ← all pipeline outputs
│   ├── enriched.csv               ← main output: one row per article, all pipeline columns added
│   ├── sentences_actionability.csv ← sentence-level detail: one row per sentence with raw scores
│   ├── interpretations_report.md  ← analytical interpretations of all key figures and findings
│   ├── stats/                     ← aggregate statistical summaries
│   │   ├── group_stats_country.csv
│   │   ├── group_stats_domain.csv
│   │   ├── group_stats_language.csv
│   │   ├── group_stats_region.csv
│   │   ├── group_stats_global_region.csv
│   │   ├── cluster_silhouette_scores.csv
│   │   ├── cluster_summary_structural_k3.csv  ← k=3 final (structural features only)
│   │   ├── cluster_summary_full_k3.csv        ← k=3 final (with actionability)
│   │   ├── cluster_summary_structural_k4.csv  ← k=4 exploratory
│   │   ├── cluster_summary_full_k4.csv
│   │   ├── cluster_summary_structural_k5.csv  ← k=5 exploratory
│   │   └── cluster_summary_full_k5.csv
│   ├── visualizations/            ← PNG plots from generate_visualizations.py
│   │   ├── general_graphs/        ← language, region, source type, frame distributions
│   │   └── clustering_graphs/     ← PADM feature heatmaps, cluster profiles
│   └── tables/                    ← appendix tables A1–A6
│       ├── table_A1_corpus.png
│       ├── table_A2_actionability.png
│       ├── table_A3_padm_components.png
│       ├── table_A4_source_region.png
│       ├── table_A5_framing.png
│       ├── table_A6_clustering.png
│       └── appendix_tables.docx   ← editable Word version of A1–A6
│
├── notebook_tests/                ← output generators and development notebooks (see below)
│   ├── generate_visualizations.py ← produces all PNGs in output/visualizations/
│   ├── generate_tables.py         ← produces appendix table PNGs in output/tables/
│   ├── generate_tables_docx.py    ← produces appendix_tables.docx in output/tables/
│   ├── generate_pipeline_diagram.py ← renders pipeline flowchart (Graphviz)
│   ├── statistical_significance.ipynb ← statistical tests (Chi-square, Mann-Whitney, Kruskal-Wallis)
│   ├── actionability_tester.ipynb ← development: testing actionability scoring in isolation
│   └── function_tester.ipynb      ← development: testing individual NLP functions
│
├── logs/                          ← run logs (gitignored)
├── run_nlp_pipeline.py            ← main entry point
└── requirements.txt
```

---

## notebook_tests/

This folder contains two types of files:

**Output generators** — standalone scripts that read from `output/` and produce figures, tables, and diagrams. These are separate from the main pipeline and are run manually after `run_nlp_pipeline.py` completes:

```bash
python notebook_tests/generate_visualizations.py   # all PNG plots → output/visualizations/
python notebook_tests/generate_tables.py           # appendix table PNGs → output/tables/
python notebook_tests/generate_tables_docx.py      # editable Word tables → output/tables/
python notebook_tests/generate_pipeline_diagram.py # pipeline diagram → output/
```

**Development notebooks** — Jupyter notebooks used during development to test and validate individual pipeline components. They are not part of the production pipeline but are included for transparency:

- `statistical_significance.ipynb` — runs and records all statistical significance tests cited in the report (Chi-square, Mann-Whitney U, Kruskal-Wallis H)
- `actionability_tester.ipynb` — isolated testing of the actionability scoring functions
- `function_tester.ipynb` — isolated testing of individual NLP utility functions

---

## Input CSV schema

| column | type | notes |
|--------|------|-------|
| `article_id` | int | unique article identifier |
| `flood_id` | int | flood event id |
| `iso` | str | country ISO code e.g. `BRA` |
| `country` | str | full country name |
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

## Pipeline stages

| stage | module | columns added |
|-------|--------|---------------|
| 1 | `preprocessing.py` | `language` |
| 2 | `actionability.py` | `total_sentences`, `actionability_percentage`, sentence-level features |
| 3 | `authority.py` | `domain`, `scope`, `source_type` |
| 4 | `framing.py` | `dominant_frame` |
| 5 | `clustering.py` | `region`, `data_cluster_id` |
| 6 | save | `output/enriched.csv` (607 rows × 21 columns) |

Sentence-level detail is written separately to `output/sentences_actionability.csv`.

---

## Actionability scoring

Scoring operates at the **sentence level** and aggregates to article level.

Each sentence is scored using:
- Keyword matching — imperative verbs, urgency signals, spatial anchors, recovery terms (language-specific lists for EN/ES/PT in `config/nlp_config.py`)
- spaCy morphology — imperative/subjunctive verb forms, modal auxiliaries
- SRL-lite — presence of agent, action, and location in the same sentence
- Advice detection — institutional recommendation verbs (*recommend*, *suggest*, *urge* and ES/PT equivalents)

The article-level `actionability_percentage` is the weighted proportion of sentences containing actionable content.

---

## Clustering

K-Means (k=3) on five normalised structural features (imperative count, urgency, spatial anchors, advice-framing, SRL completeness). Actionability percentage is **excluded from clustering input** and observed post-hoc.

**k=3 selected** by silhouette score (0.499 vs k=4: 0.286, k=5: 0.208). Cluster summaries for k=4 and k=5 are retained in `output/stats/` for reference.

| cluster | label | n | mean actionability |
|---------|-------|---|--------------------|
| 0 | Descriptive Baseline | 549 (90%) | 0.7% |
| 1 | Actionable Advisory | 39 (6%) | 18.0% |
| 2 | Recovery Discourse | 19 (3%) | 5.3% |

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

## AI use disclosure

AI assistants (Claude, Gemini) were used during the development of this project in the following ways:

- **Code development and debugging** — assisted in writing, structuring, and iterating on pipeline modules, resolving errors, and refining implementation logic. All design decisions, analytical choices, and module architectures are the authors' own.
- **Editorial and linguistic feedback** — used to provide structural recommendations for condensing written sections while preserving core arguments and the authors' voice. The output of these prompts was used as a starting point for manual editing; all final wording, arguments, analysis, and conclusions are entirely the authors' own.
- **Clarification of concepts** — occasionally consulted to clarify technical terminology or confirm interpretations, in the same way one might use a reference tool.

AI-generated suggestions were not adopted verbatim. All core arguments, references, analytical interpretations, and conclusions in the accompanying report are the authors' original work.

---

## Literature grounding

| module | key papers |
|--------|-----------|
| `preprocessing.py` | Blomeier et al. 2025 |
| `actionability.py` | Mostafiz et al. 2022; Zade et al. 2018; Jurafsky 2014; Zguir et al. 2025 |
| `authority.py` | Semetko & Valkenburg 2000; Barocas et al. 2023 |
| `framing.py` | Entman 1993; Klein 2024; Blomeier et al. 2025 |
| `clustering.py` | Sit et al. 2020; Lindell & Perry 2012 |
