# Interpretations for Report — Key Analytical Visualizations

**Dataset:** 607 articles | 11 flood events | EN, ES, PT  
**Framework:** Protective Action Decision Model (PADM) — Lindell & Perry (2012)  
**Figures:** output/clustering_graphs/

---

## Figure 1 — Range of Article-Level Actionability
`00_actionability_range.png`

The strip plot establishes the structural context for all subsequent findings. Each dot represents one article plotted against its actionability percentage. The dense vertical cluster at 0% represents 506 articles (83% of the corpus) — every dot in that column scored zero across all sentences. The remaining 101 articles (17%) scatter rightward, with a median of 10.0% among those with any actionability and a long tail extending to 40%.

The key interpretive point is not simply that actionability is low on average, but that the distribution is strongly bimodal: the corpus divides sharply into articles that contain no actionable content at all and a minority that contain some. There is no gradient — an article either triggers advisory features or it does not. This pattern is consistent with the PADM prediction that actionable disaster communication is a deliberate editorial choice rather than a default property of flood coverage, and that the majority of outlets in this corpus do not make that choice.

---

## Figure 2 — PADM Component Presence by Language
`09_padm_components_by_language.png`

This chart shows, for each PADM-relevant linguistic feature, the percentage of articles in which at least one sentence triggered that feature, broken down by language. It directly answers the research question by showing which elements of actionable communication the corpus routinely delivers and which it systematically omits.

**Spatial anchors** are present in 68–92% of articles — flood coverage consistently locates events geographically, with Spanish (88%) and Portuguese (92%) outlets anchoring events more precisely than English (68%). This satisfies one PADM criterion (named location) but not the directive criterion (what to do there).

**Short-term urgency** is present in roughly 51–56% of articles across all three languages, with English slightly lower (51%) than Spanish and Portuguese (both 56%). **Imperative signals** — measured by combining keyword-based detection with morphological POS tagging for Spanish and Portuguese conjugated imperatives — are present in 34% of English articles, 16% of Portuguese, and only 10% of Spanish. The lower Spanish figure reflects that the POS-based detection captures conjugated imperative forms that differ substantially from the keyword list, while English keyword matching covers a broader range of directive verbs.

**Advice-framing** is the rarest component across all languages — present in 13% of Spanish articles, 7% of English, and only 5% of Portuguese. This is the PADM element most directly associated with concrete protective action recommendations (PARs): verbs of institutional recommendation (*recommends*, *urges*, *suggests*) that signal actionable guidance without requiring a grammatical imperative. Its near-absence across the corpus is the single most direct evidence that flood journalism does not meet PADM communication standards.

**Cross-linguistic pattern:** All three language groups share the same structural profile — strong spatial and urgency signals, near-absent advice-framing. The differences are of degree, not kind, and no language group comes close to meeting PADM standards for actionable communication.

---

## Figure 3 — Mean Actionability by Dominant Frame
`10_frame_actionability.png`

The chart plots mean actionability percentage by dominant frame (impact, response, accountability, recovery), with ±1 SD error bars and the corpus mean (2.0%) as a reference line.

**The central finding is that response-framed articles are not more actionable than impact-framed ones.** Impact articles average approximately 1.3% and response articles approximately 2.0% — both at or below the corpus mean and statistically indistinguishable given the overlapping standard deviations. The corpus *responds* to floods without *instructing* anyone. Response framing describes rescue operations, evacuation orders issued, and emergency resources deployed — but describes them as events happening to third parties rather than as guidance directed at the reader.

**Accountability-framed articles score highest** (3.6%), the only frame meaningfully above the corpus mean. Government warnings, official protocols, and criticism of inadequate preparation often contain embedded directives and institutional recommendations as supporting evidence for accountability claims. Mexican and Colombian national news outlets — which produce the bulk of the corpus's actionable content — frequently frame actionable guidance within accountability contexts.

**Recovery framing** averages 0.9% — below even impact framing and the lowest of all four frames (n=26). Despite the intuition that recovery articles embed recommendations and timelines, this corpus's recovery coverage is predominantly descriptive, reporting on damage assessments and funding decisions rather than directing reader behaviour.

**Implication for the RQ:** Dominant frame alone is insufficient to predict actionability. The mechanism driving actionability differences is the specific linguistic features within an article — particularly advice-framing verbs — not the thematic orientation of the piece. An article can be fully response-framed while containing zero actionable sentences.

---

## Figure 3a — Mean Actionability by Frame — English
`10_en_frame_actionability.png`

Among English-language articles (n=94, mean 0.9%), all four frames score below 1.2% and are statistically indistinguishable. Accountability framing averages only 0.35% — the lowest of any frame in any language — reflecting that English-language coverage in this corpus is dominated by US and Canadian regional and wire-service outlets, which report on official actions without embedding actionable directives. The near-zero variance across frames confirms that dominant frame is not a predictor of actionability in English flood journalism.

---

## Figure 3b — Mean Actionability by Frame — Spanish
`10_es_frame_actionability.png`

Spanish-language articles (n=200, mean 3.1%) show the clearest frame differentiation in the corpus. Accountability-framed articles average 6.0% — nearly double the language mean and three times the impact frame (1.7%). This pattern is driven by Mexican and Colombian national news outlets, which produce the majority of Spanish accountability coverage and embed institutional directives within criticism of government preparedness. Response (2.0%) and recovery (1.3%) frames remain below the language mean. The accountability gap in Spanish is the primary driver of the corpus-level finding that accountability framing is the most actionable frame overall.

---

## Figure 3c — Mean Actionability by Frame — Portuguese
`10_pt_frame_actionability.png`

Portuguese-language articles (n=313, mean 1.6%) show a flatter frame profile than Spanish, with all four frames within a narrow band. Response framing is marginally highest (2.3%), followed by accountability (2.0%), impact (1.0%), and recovery (0.7%). The near-equivalence across frames suggests that in Brazilian regional journalism — which dominates the Portuguese corpus — frame choice does not reflect a substantive editorial decision to include or exclude actionable content. Articles across all frames share the same descriptive baseline profile.

---

## Figure 4 — Source Type × Global Region (H₁ Mechanism)
`11_source_type_by_region.png`

This chart directly tests the mechanism behind the regional difference identified in H₁. The y-axis shows mean actionability, the x-axis shows source type ordered by mean score, and the two bars per category separate North America (blue) and South America (orange) outlets.

**The regional gap is real but narrowly sourced.** South America outlets score higher than North America across most source types, but the gap is only substantively large in **national_news** — where South America national news (primarily Mexican and Colombian outlets) averages approximately 4%, while North America national news (US and Canadian outlets) is absent from this category because the corpus contains no national North American news outlets with comparable volume. For every other source type — regional news, local news, NGO, government agency, and unknown — the difference between regions is negligible.

**This finding partially supports H₁ while complicating its framing.** The hypothesis predicts that North America outlets produce less actionable content because institutional alert systems carry the directive function. The data show North America news media are indeed less actionable, but this appears to reflect the *absence* of national-scale advisory journalism in the corpus rather than a systematic editorial choice by North America media. US and Canadian coverage in this corpus is dominated by wire services and regional outlets (classified as unknown and regional_news), neither of which produces the institutional recommendation language that drives actionability scores in Mexican and Colombian national outlets.

**Structural implication:** Actionability in this corpus is a property of *institutional national journalism*, not of geography or language. Where national-scale outlets are present, actionability is higher — regardless of region. The South America advantage is an artifact of corpus composition, not a stable regional pattern. This requires careful qualification when interpreting the regional comparison, as it means H₁ cannot be cleanly confirmed or rejected without controlling for source type.

---

## Figure 5 — Cluster Profiles Across PADM Features (Heatmap)
`12_cluster_padm_heatmap.png`

The heatmap shows the three K-Means clusters (structural features, k=3, silhouette=0.499) against five PADM-relevant structural features. Colour encodes raw mean score per cluster (darker = higher); values are annotated in each cell. Actionability was not used in clustering — the cluster structure was derived from these five features alone, and actionability was observed post-hoc.

**Cluster 1 (Actionable Advisory, n=39, 6% of corpus)** is defined by the highest advice-framing (0.185) and short-term urgency (0.328) of any cluster, combined with elevated spatial anchors (0.599). This cluster scores 18.0% mean actionability — by far the highest of any cluster. That actionability is driven primarily by institutional recommendation language rather than direct commands: articles that *recommend* protective actions through official authority rather than imperative address. This is the only cluster that approaches PADM compliance.

**Cluster 2 (Recovery Discourse, n=19, 3% of corpus)** is defined by elevated long-term recovery keywords (0.456) alongside moderate spatial anchors (0.406) and short-term urgency (0.196), with 5.3% mean actionability. It reflects forward-looking institutional language about reconstruction, policy, and resilience — describing what will be done rather than what readers should do now.

**Cluster 0 (Descriptive Baseline, n=549, 90% of corpus)** scores near zero on advice-framing (0.002) and imperatives (0.015) with a mean actionability of 0.7%, confirming that the overwhelming majority of the corpus is behaviourally inert — geographically located and factually complete, but containing no advisory dimension.

**Overall heatmap interpretation:** The three clusters map onto three distinct relationships with PADM standards: (0) no advisory features; (2) instructional language present but future-oriented, not reader-directed; (1) the only cluster approaching PADM compliance, representing just 6% of the corpus. The concentration of 90% of articles in the descriptive baseline cluster — and the near-absence of advice-framing across all clusters — is the clearest structural evidence that flood journalism in this corpus does not meet PADM communication standards for protective action reporting.

---

## Figure 6 — Actionability by Country

| Country | Articles | Mean actionability | Max |
|---------|----------|--------------------|-----|
| Mexico | 105 | 4.19% | 40.0% |
| Peru | 30 | 3.80% | 25.0% |
| Costa Rica | 3 | 3.70% | 11.1% |
| Colombia | 31 | 2.15% | 20.0% |
| Bolivia | 3 | 2.08% | 6.25% |
| Brazil | 313 | 1.59% | 37.5% |
| Canada | 12 | 1.47% | 9.09% |
| United States | 64 | 0.70% | 20.0% |
| Ecuador | 5 | 0.00% | — |
| Uruguay | 22 | 0.00% | — |
| Guyana | 6 | 0.00% | — |

Mexico is the most actionable country in the corpus, driven by national news outlets that use explicit advice-framing language. Brazil, despite contributing 51.6% of all articles, scores below the corpus mean — confirming that Common Crawl's over-indexing on Brazilian regional outlets inflates corpus volume without proportionate actionability. Several countries (Ecuador, Uruguay, Guyana, Panama, Venezuela) score 0% across all articles.

---

## Figure 7 — Actionability by Domain (top 10 by mean, min. 5 articles)

| Domain | Articles | Mean actionability |
|--------|----------|--------------------|
| tabascohoy.com | 8 | 8.07% |
| expreso.com.pe | 18 | 4.61% |
| elfinanciero.com.mx | 60 | 4.53% |
| eje21.com.co | 10 | 4.43% |
| elpueblo.com.pe | 8 | 3.87% |
| bhaz.com.br | 8 | 3.72% |
| em.com.br | 22 | 3.68% |
| folhavitoria.com.br | 6 | 3.57% |
| ibahia.com | 10 | 2.70% |
| campograndenews.com.br | 7 | 2.68% |

The highest-scoring domains are Mexican and Peruvian national and regional outlets. `elfinanciero.com.mx` (60 articles, 10th largest domain) scores 4.53%, confirming that Mexican national journalism consistently embeds advisory language. The presence of several Brazilian domains in the top 10 reflects that even within the low-actionability Brazilian corpus, a subset of outlets produce above-average advisory content.

---

## Key Cross-Cutting Findings

1. **Actionability is structurally rare.** The median actionability score is 0% for every language, region, and country grouping. Across 607 articles, fewer than 6% contain meaningful advisory content (Cluster 1, k=3).

2. **Source type predicts actionability more than language or region.** National news outlets — particularly Mexican and Colombian — consistently score higher. Regional outlets, which dominate the corpus by volume, consistently score near zero. Cluster membership (η²=0.453) explains variance in actionability far better than frame (η²=0.025) or region alone.

3. **Common Crawl's coverage pattern shapes the findings.** Brazil accounts for 51.6% of the corpus (313 articles). This concentration amplifies low-actionability Portuguese content and suppresses the corpus-wide mean.

4. **Advisory content is structurally distinct.** The actionable cluster (Cluster 1, k=3) is defined by advice-framing verbs rather than imperative structures — institutional recommendation language, not direct public commands. This cluster represents just 6% of the corpus and is concentrated in Spanish national news.

---

*Figures generated from: enriched.csv, cluster_summary_structural_k3.csv*  
*Pipeline: nlp/actionability.py, nlp/clustering.py, generate_visualizations.py*  
*Figures: 00, 09, 10, 10_en, 10_es, 10_pt, 11, 12*
