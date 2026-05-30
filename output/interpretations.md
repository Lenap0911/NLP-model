# Pipeline Results — Interpretations

**Dataset:** 580 articles (after filtering) | 11 flood events | EN, ES, PT  
**Pipeline run:** enriched.csv — 580 rows × 33 columns

---

## 1. Actionability by Language

| Language | Articles | Mean actionability | Max |
|----------|----------|--------------------|-----|
| Spanish (ES) | 198 | 3.35% | 60.0% |
| Portuguese (PT) | 240 | 1.69% | 33.3% |
| English (EN) | 139 | 1.45% | 25.0% |

Spanish content is the most actionable on average, though the median is 0% across all three languages — actionability is rare regardless of language. The corpus skews toward Portuguese by article count (41.4%) yet PT produces the lowest mean actionability, reflecting the dominance of low-actionability Brazilian regional outlets. English content, primarily from the United States, scores lowest despite coming from a Global North context.

---

## 2. Actionability by Region

| Region | Articles | Mean actionability | Share of corpus |
|--------|----------|--------------------|-----------------|
| Global South | 448 | 2.38% | 77.2% |
| Global North | 129 | 1.56% | 22.2% |

Global South coverage is marginally more actionable than Global North, which runs counter to the hypothesis that richer-country outlets produce more guidance-oriented reporting. This is driven by the concentration of actionable Mexican national news within the Global South category. Global North coverage (US and Canada) is dominated by descriptive impact reporting and retrospective journalism.

---

## 3. Actionability by Country (top 8)

| Country | Articles | Mean actionability |
|---------|----------|--------------------|
| Mexico | 92 | 4.60% |
| Costa Rica | 3 | 4.17% |
| Colombia | 41 | 3.65% |
| Canada | 14 | 2.83% |
| Peru | 27 | 2.47% |
| Brazil | 240 | 1.69% |
| United States | 115 | 1.41% |
| Uruguay | 28 | 0.42% |

Mexico is the most actionable country in the corpus, driven by national news outlets that use explicit advice-framing language. Brazil, despite contributing 41.4% of all articles, scores below average — confirming that Common Crawl's over-indexing on Brazilian regional outlets inflates corpus volume without proportionate actionability. Several countries (Argentina, Bolivia, Ecuador, Honduras, Guyana, Panama) score 0% across all articles, suggesting either a coverage gap or a consistently non-advisory register in those outlets.

---

## 4. Actionability by Domain (top 10 by mean, min. 5 articles)

| Domain | Articles | Mean actionability |
|--------|----------|--------------------|
| elfinanciero.com.mx | 59 | 5.26% |
| tabascohoy.com | 5 | 6.67% |
| alcalorpolitico.com | 5 | 6.67% |
| elcolombiano.com | 5 | 6.19% |
| semana.com | 8 | 5.63% |
| eje21.com.co | 16 | 3.76% |
| cbc.ca | 11 | 3.60% |
| em.com.br | 32 | 2.98% |
| diariodepernambuco.com.br | 75 | 1.45% |
| folhams.com.br | 18 | 0.00% |

`elfinanciero.com.mx` is both the second largest domain (59 articles, 10.2% of corpus) and among the most actionable (5.26%), suggesting national Mexican financial and general news outlets consistently embed advisory language. `diariodepernambuco.com.br`, the largest single domain (75 articles, 12.9% of corpus), scores only 1.45% — illustrating how CC's crawl concentration in one Brazilian regional outlet shapes the overall corpus character. Several outlets score exactly 0% across all articles, including `folhams.com.br` (18 articles), `pbs.org` (9), and `elobservador.com.uy` (8).

---

## 5. Dominant Frame Distribution

| Frame | Articles | Share |
|-------|----------|-------|
| Impact | 252 | 43.4% |
| Response | 157 | 27.1% |
| Accountability | 147 | 25.3% |
| Recovery | 24 | 4.1% |

The dominant register across the corpus is **impact framing** — describing casualties, damage, and displacement. Together, impact and accountability account for 68.7% of all articles. Response and recovery framing, which are more proximate to actionable guidance, represent only 31.2%. The near-absence of recovery framing (4.1%) suggests flood coverage in this corpus is concentrated in the acute phase and does not extend to long-term rebuilding discourse.

---

## 6. Unsupervised Clustering — Structural K=4

K=4 was selected as the optimal number of clusters based on silhouette scores (structural k=4: 0.332; full k=4: 0.333). Clusters were derived from structural language features only; actionability percentage was observed post-hoc.

### Cluster 3 — Descriptive Narrative (n=402, 69% of corpus)
Near-zero scores across all features. Brazil PT, regional news. The dominant pattern in the corpus — factual event description with no behavioural guidance. Confirms CC's systematic over-indexing on high-volume regional outlets that produce low-actionability content.

### Cluster 1 — Urgency-Spatial Coverage (n=124, 21% of corpus)
Highest spatial anchor density (0.63) and short-term urgency signals (0.36) of any cluster. US/ES content, predominantly unknown source type (consistent with wire services). These articles locate the emergency precisely and convey urgency but still score only 2.9% actionability — the *warns without instructing* pattern. The structural ingredients for actionable advice are present but the advice itself is absent.

### Cluster 0 — Recovery and Policy Discourse (n=18, 3% of corpus)
Elevated long-term keyword density (0.44) — reconstruction, resilience, policy, mitigation. Scores identically to Cluster 1 on actionability (2.9%) but for a different reason: forward-looking institutional language about what will be done rather than what readers should do. Brazil PT, regional news. Structurally distinct from urgency coverage despite similar actionability.

### Cluster 2 — Actionable Advisory Coverage (n=36, 6% of corpus)
The only cluster with meaningful actionability (21%). Uniquely driven by advice-framing verbs (`mean_advice` = 0.21) — *recommends*, *suggests*, *urges* — rather than imperative forms. Predominantly Mexican Spanish national news (38.9% Mexico, 61.1% ES). When advisory content appears in this corpus it comes from national-level Spanish outlets and takes the form of institutional recommendations rather than direct commands.

---

## 7. Key Cross-Cutting Findings

1. **Actionability is structurally rare.** The median actionability score is 0% for every language, region, and country grouping. Across 580 articles and 4,444 sentences, fewer than 6% of articles contain meaningful advisory content.

2. **Source type predicts actionability more than language or region.** National news outlets — particularly Mexican and Colombian — consistently score higher. Regional outlets, which dominate the corpus by volume, consistently score near zero.

3. **Common Crawl's coverage pattern shapes the findings.** A single Brazilian regional outlet (`diariodepernambuco.com.br`) contributes 12.9% of all articles. Brazil accounts for 41.4% of the corpus. This concentration amplifies low-actionability Portuguese content and suppresses the corpus-wide mean.

4. **Advisory content is structurally distinct.** The actionable cluster (Cluster 2) differs not just in score but in mechanism — advice-framing verbs rather than imperative structures. This suggests actionability in this corpus is mediated through institutional recommendation language, not direct public commands.

---

*Generated from pipeline output — enriched.csv, cluster_summary_structural_k4.csv, group_stats_*.csv*
