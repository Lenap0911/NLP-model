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

**Spatial anchors** are present in 70–81% of articles — flood coverage consistently locates events geographically, which satisfies one PADM criterion (named location) but not the directive criterion (what to do there).

**Imperative signals** and **short-term urgency** are present in roughly 40–52% of articles, with Portuguese significantly lower on imperatives (18%) than English (45%) or Spanish (41%). This cross-linguistic gap reflects the different journalistic registers of Brazilian regional Portuguese outlets, which tend toward descriptive rather than directive language.

**Advice-framing** is the rarest component across all languages — present in fewer than 12% of articles in any language group. This is the PADM element most directly associated with concrete protective action recommendations (PARs): verbs of institutional recommendation (*recommends*, *urges*, *suggests*) that signal actionable guidance without requiring a grammatical imperative. Its near-absence across the corpus is the single most direct evidence that flood journalism does not meet PADM communication standards.

**Cross-linguistic pattern:** The profile is broadly similar across EN, ES, and PT — all three languages are advice-framing poor while maintaining strong spatial and urgency signals. The differences are of degree, not kind. This consistency supports treating the failure as a genre-level property of flood journalism rather than a language-specific artifact.

---

## Figure 3 — Mean Actionability by Dominant Frame
`10_frame_actionability.png`

The chart plots mean actionability percentage by dominant frame (impact, response, accountability, recovery), with ±1 SD error bars and the corpus mean (2.0%) as a reference line.

**The central finding is that response-framed articles are not more actionable than impact-framed ones.** Impact articles average approximately 1.3% and response articles approximately 2.0% — both at or below the corpus mean and statistically indistinguishable given the overlapping standard deviations. The corpus *responds* to floods without *instructing* anyone. Response framing describes rescue operations, evacuation orders issued, and emergency resources deployed — but describes them as events happening to third parties rather than as guidance directed at the reader.

**Accountability-framed articles score highest** (3.6%), the only frame meaningfully above the corpus mean. Government warnings, official protocols, and criticism of inadequate preparation often contain embedded directives and institutional recommendations as supporting evidence for accountability claims. Mexican and Colombian national news outlets — which produce the bulk of the corpus's actionable content — frequently frame actionable guidance within accountability contexts.

**Recovery framing** averages 0.9% — below even impact framing and the lowest of all four frames (n=26). Despite the intuition that recovery articles embed recommendations and timelines, this corpus's recovery coverage is predominantly descriptive, reporting on damage assessments and funding decisions rather than directing reader behaviour.

**Implication for the RQ:** Dominant frame alone is insufficient to predict actionability. The mechanism driving actionability differences is the specific linguistic features within an article — particularly advice-framing verbs — not the thematic orientation of the piece. An article can be fully response-framed while containing zero actionable sentences.

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

The heatmap shows the three K-Means clusters (structural features, k=3, silhouette=0.499) against five PADM-relevant structural features. Colour encodes raw mean score per cluster (darker = higher); values are annotated in each cell. Actionability was not used in clustering — the cluster structure was derived from these five features alone, and actionability was observed post-hoc.

**Cluster 1 (Actionable Advisory, n=39, 6% of corpus)** is defined by the highest advice-framing (0.185) and short-term urgency (0.328) of any cluster, combined with elevated spatial anchors (0.599). This cluster scores 18.0% mean actionability — by far the highest of any cluster. That actionability is driven primarily by institutional recommendation language rather than direct commands: articles that *recommend* protective actions through official authority rather than imperative address. This is the only cluster that approaches PADM compliance.

**Cluster 2 (Recovery Discourse, n=19, 3% of corpus)** is defined by elevated long-term recovery keywords (0.456) alongside moderate spatial anchors (0.406) and short-term urgency (0.196), with 5.3% mean actionability. It reflects forward-looking institutional language about reconstruction, policy, and resilience — describing what will be done rather than what readers should do now.

**Cluster 0 (Descriptive Baseline, n=549, 90% of corpus)** scores near zero on advice-framing (0.002) and imperatives (0.015) with a mean actionability of 0.7%, confirming that the overwhelming majority of the corpus is behaviourally inert — geographically located and factually complete, but containing no advisory dimension.

**Overall heatmap interpretation:** The three clusters map onto three distinct relationships with PADM standards: (0) no advisory features; (2) instructional language present but future-oriented, not reader-directed; (1) the only cluster approaching PADM compliance, representing just 6% of the corpus. The concentration of 90% of articles in the descriptive baseline cluster — and the near-absence of advice-framing across all clusters — is the clearest structural evidence that flood journalism in this corpus does not meet PADM communication standards for protective action reporting.

---

*Figures generated from: enriched.csv, cluster_summary_structural_k3.csv*  
*Pipeline: nlp/actionability.py, nlp/clustering.py, generate_visualizations.py*
