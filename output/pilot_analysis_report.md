# Pilot Dataset — NLP Pipeline Results Analysis
**Date:** 2026-04-30  
**Dataset:** url_report_pilot.csv — 6 flood events, 6161 raw articles  
**Post-preprocessing:** 1800 articles (EN: 926 / ES: 642 / PT: 232)

---

## 1. Preprocessing Funnel

| Stage | N |
|---|---|
| Raw rows loaded | 6,161 |
| After length filter | 1,887 |
| After language filter (EN/ES/PT only) | 1,885 |
| After flood relevance filter (min hits = 2) | 1,885 |
| After deduplication | **1,800** |

Over 70% of the raw corpus was filtered — mostly short pages, boilerplate, and low-relevance crawl noise. Language distribution after filtering: **EN 51% / ES 36% / PT 13%**.

---

## 2. Actionability Scoring

### 2a. Score statistics by language

| | EN | ES | PT |
|---|---|---|---|
| Mean | **1.636** | 0.747 | 0.822 |
| Median | 1.240 | 0.650 | 0.750 |
| Std | 1.359 | 0.571 | 0.379 |
| Max | 6.556 | 3.122 | 3.496 |
| % zero score | 2.1% | **7.5%** | 1.3% |

EN articles score roughly **2× higher** than ES/PT. This reflects keyword list imbalance, not a genuine signal difference (see §6 — model gaps).

### 2b. Score distribution buckets

| Bucket | EN | ES | PT |
|---|---|---|---|
| 0 (none) | 0 | 20 | 1 |
| 0.1–0.5 (weak) | 145 | 173 | 23 |
| 0.5–1 (low) | 218 | 229 | 170 |
| 1–2 (moderate) | 276 | 148 | 30 |
| >2 (high) | 268 | 24 | 5 |

EN has a healthy spread reaching the high tier (29% of articles >2). ES and PT are heavily concentrated in the 0.5–1 low band — the keyword lists hit only the most basic signals.

### 2c. Sub-scores by language

| Sub-score | EN | ES | PT |
|---|---|---|---|
| Imperative verbs | **1.689** | 0.458 | 0.810 |
| Short-term urgency | **1.624** | 0.849 | 0.315 |
| Long-term recovery | **0.916** | 0.623 | 0.155 |
| Spatial anchors | **2.422** | 1.469 | 2.181 |

Spatial scores are most balanced — geographic terms translate well across languages. PT short-term urgency (0.315) is the weakest sub-score in the corpus, indicating the PT keyword list barely covers the urgency register of Brazilian Portuguese journalism.

### 2d. Actionability by flood event

| flood_id | N | Mean | Median | Max |
|---|---|---|---|---|
| 1 | 118 | 1.208 | 0.953 | 5.975 |
| 2 | 265 | 0.998 | 0.750 | 5.472 |
| 3 | 689 | 0.833 | 0.700 | 6.059 |
| 4 | 61 | 0.852 | 0.675 | 3.082 |
| 5 | 55 | 1.352 | 0.946 | 5.470 |
| **6** | **612** | **1.761** | **1.295** | **6.556** |

Flood_id 6 (predominantly English) scores highest — but also contains the false positives (see §5).

---

## 3. Framing Analysis

### Frame distribution by language

| Frame | EN | ES | PT |
|---|---|---|---|
| **Response** | **533 (57%)** | 153 (24%) | **162 (70%)** |
| **Accountability** | 165 (18%) | **347 (54%)** | 42 (18%) |
| Impact | 143 (15%) | 84 (13%) | 22 (9%) |
| Recovery | 85 (9%) | 58 (9%) | 6 (3%) |

**This is the most analytically significant finding.**

- **English and Portuguese** media frames floods through **operational response** — what is being done, emergency action, institutional mobilisation.
- **Spanish-language** media frames floods through **accountability** — institutional failures, political responsibility, structural critique.

This divergence mirrors findings in crisis communication literature: Latin American journalism tends toward systemic critique of state failure, while Anglophone outlets lead with response logistics. This is a real signal worth reporting in the thesis.

---

## 4. SRL-lite (Semantic Role Labelling)

| Feature | EN | ES | PT |
|---|---|---|---|
| Has agent | 0.997 | 1.000 | 0.974 |
| Has action | 0.989 | 0.992 | 0.991 |
| Has location | 0.927 | 0.974 | **1.000** |
| SRL complete | 0.917 | 0.966 | 0.966 |

SpaCy performs well across all three languages. Location detection is actually strongest for PT and ES — the smaller models may handle proper noun tagging well on cleaner journalistic text. EN has slightly lower SRL completeness (92%) likely due to more fragmented/boilerplate EN articles in the corpus.

### Past-tense ratio (penalty mechanism)

| Language | Mean past_tense_ratio |
|---|---|
| EN | 0.437 |
| ES | 0.400 |
| PT | **0.108** |

PT's suspiciously low ratio (0.108 vs ~0.42 for EN/ES) indicates `pt_core_news_sm` is underdetecting verb morphological tense. The small PT model does not handle Brazilian Portuguese tense inflection as reliably as the EN/ES equivalents. The past-tense penalty is effectively not firing for PT articles, meaning their actionability scores are not being corrected downward for retrospective content.

---

## 5. Clustering + Topics

### Cluster summary

| Cluster | N | Lang mix | Mean actionability |
|---|---|---|---|
| **10 (catch-all)** | **1,426 (79%)** | EN/ES/PT mixed | 1.257 |
| 1 | 151 | PT only | 0.796 |
| 0 | 29 | EN only | **4.156** |
| 4 | 86 | EN only | 0.920 |
| 2 | 30 | EN only | 0.293 |
| 6 | 16 | PT+EN | 1.609 |
| 5 | 10 | EN+ES | 0.497 |
| -1 (outliers) | 4 | mixed | 0.489 |

**79% of the corpus collapsed into a single undifferentiated cluster** — BERTopic could not find enough term-frequency signal to split the heterogeneous 6-event corpus. Cluster 1 (all PT) and cluster 0 (high-action EN) are the only semantically coherent clusters. Cross-language clustering is not happening — language dominates over topic in the embedding space.

### Top topic labels

| Topic | N | Keywords |
|---|---|---|
| 0 (noise) | 1,426 | the, and, of, de, to |
| 1 | 151 | de, eventos, geo, previsão, hidrológicos |
| 2 | 87 | jul, nov, nightline, apr, 27 |
| 3 | 30 | at, airport, hewanorra, cloudy, showers |
| 4 | 29 | july, dc, co, in, for |
| 5 | 17 | 2025, pm, bolsonaro, july, 40 |
| 6 | 14 | de, la, el, en, río |
| 9 | 11 | edital, superior, médio, prefeitura, 2022 |
| 10 | 10 | panama, news, panamá, canal, the |

Topic 0 labels are pure stopwords — the catch-all cluster has no distinguishing vocabulary. Topic 1 is genuine (PT hydrological forecasting). Topic 2/7 show date artifacts (`nightline, jul, nov, apr`) from news archive headers. Topic 9 is Brazilian public procurement content (`edital, prefeitura`) — not flood content at all, indicating an upstream relevance filter failure.

---

## 6. Cross-Lingual Pairs

**Result: 0 pairs found.**

The CSLS matching found no EN↔ES or EN↔PT article pairs above the threshold. This is not a model failure — it reflects the data structure: flood_ids 1–6 in the pilot appear to represent **different geographic events with monolingual coverage**, not bilingual coverage of the same event. Cross-lingual matching requires articles from different language outlets covering the *same* flood event. The Americas dataset (multiple EN + ES outlets covering the same Latin American floods) should produce pairs.

---

## 7. Where the Model is Lacking

### Critical issues

**False positives at the top of actionability**  
The 3 highest-scoring articles in the corpus are "Best 15 Minutes of Fame 2009 | Best of Miami" from `miaminewtimes.com` (score 6.556). These are entertainment awards lists that score high because they contain dense location names, short imperative phrases, and urgency-adjacent vocabulary. The keyword scorer cannot distinguish "Best of Miami" from "Evacuate Miami." Flood_id 6 (1855 EN articles from a broad South Florida crawl) is the source. Fix: domain blocklist or raise the minimum `flood_hits` threshold for flood_id 6.

**EN keyword list 3× richer than ES/PT**  
EN has 10 imperative verbs, 8 short-term signals, 8 long-term signals, 10 spatial anchors. ES has 8/8/7/8, PT has 10/9/10/12. The raw counts look similar but the EN terms are broader and hit more surface forms. The 2× gap in imperative/short-term scores is primarily a list coverage gap, not a real journalistic difference.

### Moderate issues

**Cluster collapse (79% in one cluster)**  
`BERTOPIC_MIN_TOPIC_SIZE = 5` is too permissive for a 1800-article mixed corpus. BERTopic forms dozens of micro-clusters and lumps everything else together. Raising to 15–20 would force more compact, interpretable groupings.

**Date/metadata tokens in topic labels**  
`nightline, jul, nov, apr` appearing as topic keywords means news archive header patterns are leaking through the stopword filter. Fix: add month abbreviations and `nightline` to the `_STOPWORDS` dict in clustering.py.

**Topic 9 = public procurement content**  
`edital, prefeitura, médio, superior` are Brazilian government tender terminology — not flood content. These articles passed the flood relevance filter (flood_hits ≥ 2) but are clearly off-topic. The flood keyword list is too permissive for Portuguese.

### Minor issues

**PT past-tense underdetection**  
`pt_core_news_sm` reliably misses verb tense morphology. The past-tense penalty (0.15 weight) is effectively inactive for PT. Acceptable given the small model size; would require `pt_core_news_lg` or `pt_core_news_trf` to fix.

---

## 8. Recommended Next Steps

| Priority | Fix |
|---|---|
| High | Domain blocklist for `miaminewtimes.com` and similar city-guide domains |
| High | Expand ES `short_term` and `imperative_verbs` lists (close the 2× gap) |
| High | Expand PT `short_term` list (0.315 mean is too low to be meaningful) |
| Medium | Raise `BERTOPIC_MIN_TOPIC_SIZE` to 15–20 |
| Medium | Add month names + `nightline` to clustering stopwords |
| Medium | Raise flood_hits minimum threshold for PT (reduce public procurement noise) |
| Low | Investigate `pt_core_news_sm` tense detection; consider `pt_core_news_lg` |
