# Interpretations for Report — Key Analytical Visualizations

**Dataset:** 580 articles | 11 flood events | EN, ES, PT  
**Framework:** Protective Action Decision Model (PADM) — Lindell & Perry (2012)  
**Figures:** output/clustering_graphs/

---

## Figure 1 — Range of Article-Level Actionability
`00_actionability_range.png`

The strip plot establishes the structural context for all subsequent findings. Each dot represents one article plotted against its actionability percentage. The dense vertical cluster at 0% represents 484 articles (84% of the corpus) — every dot in that column scored zero across all sentences. The remaining 93 articles (16%) scatter rightward, clustering between 5–20% with a median of 12.5% and a long tail extending to 60%.

The key interpretive point is not simply that actionability is low on average, but that the distribution is bimodal: the corpus divides sharply into articles that contain no actionable content at all and a small minority that contain some. There is no gradient. This pattern is consistent with the PADM prediction that actionable disaster communication is a deliberate editorial choice rather than a default property of flood coverage — and that most outlets in this corpus do not make that choice.

---

## Figure 2 — PADM Component Presence by Language
`09_padm_components_by_language.png`

This chart shows, for each PADM-relevant linguistic feature, the percentage of articles in which at least one sentence triggered that feature, broken down by language. It directly answers the research question by showing which elements of actionable communication the corpus routinely delivers and which it systematically omits.

**Spatial anchors** are present in 70–81% of articles — flood coverage consistently locates events geographically, which satisfies one PADM criterion (named location) but not the directive criterion (what to do there).

**Imperative signals** and **short-term urgency** are present in roughly 40–52% of articles, with Portuguese significantly lower on imperatives (18%) than English (45%) or Spanish (41%). This cross-linguistic gap reflects the different journalistic registers of Brazilian regional Portuguese outlets, which tend toward descriptive rather than directive language.

**Advice-framing** is the rarest component across all languages — present in fewer than 12% of articles in any language group. This is the PADM element most directly associated with concrete protective action recommendations (PARs): verbs of institutional recommendation (*recommends*, *urges*, *suggests*) that signal actionable guidance without requiring a grammatical imperative. Its near-absence across the corpus is the single most direct evidence that flood journalism does not meet PADM communication standards.

**Cross-linguistic pattern:** The profile is broadly similar across EN, ES, and PT — all three languages are SRL-complete but advice-framing poor. The differences are of degree, not kind. This consistency supports treating the failure as a genre-level property of flood journalism rather than a language-specific artifact.

---

## Figure 3 — Mean Actionability by Dominant Frame
`10_frame_actionability.png`

The chart plots mean actionability percentage by dominant frame (impact, response, accountability, recovery), with ±1 SD error bars and the corpus mean (2.2%) as a reference line.

**The central finding is that response-framed articles are not more actionable than impact-framed ones.** Impact articles average approximately 1.2% and response articles approximately 1.5% — both below the corpus mean and statistically indistinguishable given the overlapping standard deviations. The corpus *responds* to floods without *instructing* anyone. Response framing describes rescue operations, evacuation orders issued, and emergency resources deployed — but describes them as events happening to third parties rather than as guidance directed at the reader.

**Accountability-framed articles score highest** (approximately 3.7%), which is counterintuitive at first glance but interpretable: government warnings, official protocols, and criticism of inadequate preparation often contain embedded directives and institutional recommendations as supporting evidence for accountability claims. Mexican and Colombian national news outlets — which produce the bulk of the corpus's actionable content — frequently frame actionable guidance within accountability contexts.

**Recovery framing** scores approximately 3.0%, above the corpus mean but based on a very small sample (n=24). These articles, which address reconstruction, resilience policy, and long-term planning, tend to embed specific recommendations and timelines.

**Implication for the RQ:** Dominant frame alone is insufficient to predict actionability. The mechanism driving actionability differences is the specific linguistic features within an article — particularly advice-framing verbs and imperative density — not the thematic orientation of the piece. An article can be fully response-framed while containing zero actionable sentences.

---

## Figure 4 — Source Type × Global Region (H₁ Mechanism)
`11_source_type_by_region.png`

This chart directly tests the mechanism behind the regional difference identified in H₁. The y-axis shows mean actionability, the x-axis shows source type ordered by mean score, and the two bars per category separate Global North (blue) and Global South (orange) outlets.

**The regional gap is real but narrowly sourced.** Global South outlets score higher than Global North across most source types, but the gap is only substantively large in **national_news** — where Global South national news (primarily Mexican and Colombian outlets) averages approximately 4%, while Global North national news (US and Canadian outlets) is absent from this category because the corpus contains no national North American news outlets with comparable volume. For every other source type — regional news, local news, NGO, government agency, and unknown — the difference between regions is negligible.

**This finding partially supports H₁ while complicating its framing.** The hypothesis predicts that Global North outlets produce less actionable content because institutional alert systems carry the directive function. The data show Global North news media are indeed less actionable, but this appears to reflect the *absence* of national-scale advisory journalism in the corpus rather than a systematic editorial choice by Global North media. US and Canadian coverage in this corpus is dominated by wire services and regional outlets (classified as unknown and regional_news), neither of which produces the institutional recommendation language that drives actionability scores in Mexican and Colombian national outlets.

**Structural implication:** Actionability in this corpus is a property of *institutional national journalism*, not of geography or language. Where national-scale outlets are present, actionability is higher — regardless of region. The Global South advantage is an artifact of corpus composition, not a stable regional pattern. This requires careful qualification when interpreting the regional comparison, as it means H₁ cannot be cleanly confirmed or rejected without controlling for source type.

---

## Figure 5 — Cluster Profiles Across PADM Features (Heatmap)
`12_cluster_padm_heatmap.png`

The heatmap shows the four K-Means clusters (structural features, k=4) against the six PADM-relevant structural features. Colour encodes z-score (green = above cluster average, red = below); raw means are annotated in each cell. Actionability was not used in clustering — the cluster structure was derived from these six features alone, and actionability was observed post-hoc.

**Cluster 2 (Actionable Advisory, n=36)** is uniquely identified by a single feature: advice-framing (0.209, the highest value in its column and the only strongly green cell in that column). Its imperative signals are low (0.05) and its short-term urgency is moderate (0.097). This means the 21% actionability in this cluster is driven not by direct commands but by institutional recommendation language. These are articles that *recommend* protective actions rather than *ordering* them — consistent with the journalistic register of Mexican and Colombian national news, which frames guidance through official authority rather than direct address.

**Cluster 1 (Urgency-Spatial, n=124)** has the highest imperative signals (0.215) and short-term urgency (0.356) of any cluster, combined with the highest spatial anchor density (0.630). This is the profile most consistent with classic PADM alert communication: locate the threat, signal urgency, issue a directive. Yet this cluster scores only 2.9% actionability — barely above the descriptive clusters. The explanation is SRL incompleteness within individual sentences: the components are present across the article, but they rarely co-occur in the same sentence in a form that constitutes a complete directive.

**Cluster 0 (Recovery Discourse, n=18)** is defined by elevated long-term recovery keywords (0.437) and high SRL completeness (0.844), reflecting forward-looking institutional language about what will be done rather than what readers should do now.

**Cluster 3 (Descriptive Baseline, n=402, 69% of corpus)** scores near zero on all features except SRL completeness (0.736), confirming that the majority of the corpus is factually complete but behaviourally inert — describing events with no advisory dimension.

**Overall heatmap interpretation:** The four clusters map cleanly onto four distinct failure modes relative to PADM: (3) no advisory features at all; (1) urgency and location present but no complete directive; (0) instructional language present but future-oriented, not reader-directed; (2) the only cluster approaching PADM compliance, representing 6% of the corpus. The heatmap makes visible that even the best-performing cluster achieves actionability through advice-framing rather than the full PADM sequence, suggesting that flood journalism in this corpus never fully closes the loop from threat identification to reader-executable protective action.

---

*Figures generated from: enriched.csv, cluster_summary_structural_k4.csv*  
*Pipeline: nlp/actionability.py, nlp/clustering.py, generate_visualizations.py*
