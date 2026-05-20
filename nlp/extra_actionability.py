def _extract_spacy_features(text: str, lang: str) -> dict:
    """Parse text once with spaCy and extract SRL/NER/tense features from the same Doc.

    This avoids running spaCy 3x per article (SRL + NER + tense), which is a major
    bottleneck on larger datasets.
    """
    import json

    nlp = _get_spacy(lang)
    if nlp is None:
        return {
            # SRL-lite
            'has_agent': 0,
            'has_action': 0,
            'has_location': 0,
            'srl_complete': 0,
            # NER
            'top_locations': json.dumps([]),
            'top_orgs': json.dumps([]),
            # tenses
            'past_tense': 0,
            'present_tense': 0,
            'future_tense': 0,
        }

    sentences = split_into_sentences(text)
    if not sentences:
        sentences = [""]

    # Parse per sentence (no re-joining) and aggregate features across the docs.
    docs = list(nlp.pipe(sentences))

    # --- SRL-lite (Jurafsky WHO did WHAT WHERE) ---
    has_a = 0
    has_v = 0
    has_l = 0

    # --- NER top entities ---
    seen_locs: dict[str, int] = {}
    seen_orgs: dict[str, int] = {}
    for doc in docs:
        agents = [t for t in doc if t.dep_ in ('nsubj', 'nsubjpass')]
        actions = [t for t in doc if t.pos_ == 'VERB' and t.dep_ in ('ROOT', 'ccomp', 'xcomp')]
        locations = [t for t in doc if t.ent_type_ in ('GPE', 'LOC', 'FAC')]

        has_a = int(has_a or len(agents) > 0)
        has_v = int(has_v or len(actions) > 0)
        has_l = int(has_l or len(locations) > 0)

        for ent in doc.ents:
            text_norm = ent.text.strip()
            if not text_norm:
                continue
            if ent.label_ in ('GPE', 'LOC', 'FAC'):
                seen_locs[text_norm] = seen_locs.get(text_norm, 0) + 1
            elif ent.label_ == 'ORG':
                seen_orgs[text_norm] = seen_orgs.get(text_norm, 0) + 1
    top_locs = [k for k, _ in sorted(seen_locs.items(), key=lambda x: -x[1])[:5]]
    top_orgs = [k for k, _ in sorted(seen_orgs.items(), key=lambda x: -x[1])[:5]]

    # --- verb tense counts ---
    tense_counts = {'past_tense': 0, 'present_tense': 0, 'future_tense': 0}
    for doc in docs:
        for token in doc:
            if token.pos_ in ('VERB', 'AUX'):
                morph_tense = token.morph.get('Tense')
                if morph_tense:
                    if 'Past' in morph_tense:
                        tense_counts['past_tense'] += 1
                    elif 'Pres' in morph_tense:
                        if lang == 'en' and token.lemma_ in ('will', 'shall', "'ll"):
                            tense_counts['future_tense'] += 1
                        else:
                            tense_counts['present_tense'] += 1
                    elif 'Fut' in morph_tense:
                        tense_counts['future_tense'] += 1
                else:
                    if lang == 'en' and token.lemma_ in ('will', 'shall', "'ll"):
                        tense_counts['future_tense'] += 1

    return {
        'has_agent': has_a,
        'has_action': has_v,
        'has_location': has_l,
        'srl_complete': int(has_a and has_v and has_l),
        'top_locations': json.dumps(top_locs, ensure_ascii=False),
        'top_orgs': json.dumps(top_orgs, ensure_ascii=False),
        **tense_counts,
    }

########

def extract_named_entities(text: str, lang: str) -> dict:
    """
    Extract top named entities from the article using spacy NER.
    Returns:
        top_locations: JSON-serialisable list of up to 5 unique GPE/LOC/FAC names
        top_orgs:      JSON-serialisable list of up to 5 unique ORG names
    These columns are useful for downstream geographic diffusion analysis (Han et al. 2017)
    and for validating location dictionary coverage.
    """
    feats = _extract_spacy_features(text, lang)
    return {
        'top_locations': feats['top_locations'],
        'top_orgs': feats['top_orgs'],
    }


def count_verb_tenses(text: str, lang: str) -> dict:
    """
    Identifies and counts verb tenses (past, present, future) using SpaCy.
    Accounts for Spanish morphological inflection vs English auxiliary construction.
    """
    feats = _extract_spacy_features(text, lang)
    return {
        'past_tense': feats['past_tense'],
        'present_tense': feats['present_tense'],
        'future_tense': feats['future_tense'],
    }
###########


###################################################
def score_actionability_keywords(text: str, lang: str) -> dict:
    """
    # computing keyword-based actionability sub-scores per dimension
    #following Mostafiz et al. (2022) short-term / long-term distinction
    #and Zguir et al. (2025) three-category taxonomy (supplies, personnel, actions)

    #returns dict with:
     # - imperative_score: presence of action verbs (calls to act)
      # - short_term_score: immediate danger / response language
      # - long_term_score:  recovery / resilience / policy language
      # - spatial_score:    geographic anchoring (Xu & Qiang 2022: spatial explicitness)
      # - total_score:      weighted composite

      # input: text string (article clean_text)
    """
    kw_dict = config.ACTIONABILITY_KEYWORDS.get(lang, config.ACTIONABILITY_KEYWORDS['en'])
    text_lower = text.lower()

    def _hit_count(keyword_list):
        return sum(
            1 for kw in keyword_list
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
        )

    imp   = _hit_count(kw_dict['imperative_verbs'])
    short = _hit_count(kw_dict['short_term'])
    long_ = _hit_count(kw_dict['long_term'])
    spat  = _hit_count(kw_dict['spatial_anchors'])

    # weighting: imperative verbs and short-term signals carry most actionability weight
    # spatial anchors are essential — spatially grounded info spreads further (Xu & Qiang 2022)
    total = (imp * 0.35) + (short * 0.30) + (long_ * 0.15) + (spat * 0.20)

    return {
        'imperative_score': imp,
        'short_term_score': short,
        'long_term_score':  long_,
        'spatial_score':    spat,
        'actionability_score': round(total, 4),
    }



