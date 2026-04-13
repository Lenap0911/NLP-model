# nlp/actionability.py
# scoring articles for actionability — the core analytical dimension of the project
# theoretical grounding:
#   Mostafiz et al. (2022): short-term vs long-term actionability dimensions
#   Zade et al. (2018): actionability bias — actionable info is higher utility in disasters
#   Jurafsky (2014) Ch.21: semantic role labelling — WHO did WHAT to WHOM, WHERE, WHEN
#   Kruspe et al. (2021): keyword + ML hybrid approaches for actionability detection
#   Zguir et al. (2025): taxonomy of actionable requests (supplies, personnel, actions)

import re
import logging
import importlib

import pandas as pd
import spacy

config = importlib.import_module('config.nlp_config')
logger = logging.getLogger(__name__)

# loading spacy models lazily — only once per language
_spacy_models = {}

# Maximum character count for spacy processing — truncated at sentence boundary
# to avoid splitting tokens mid-word (the original text[:1000] approach)
_SPACY_CHAR_LIMIT = 1200


def _truncate_at_sentence(text: str, max_chars: int = _SPACY_CHAR_LIMIT) -> str:
    """
    Truncate text to at most max_chars characters, but always at a sentence boundary.
    Avoids the text[:1000] antipattern that cuts mid-sentence and breaks dependency
    parsing and morphological tagging at the boundary.
    Falls back to hard truncation if no sentence boundary is found within limit.
    """
    if len(text) <= max_chars:
        return text
    # Find the last sentence-ending punctuation before the limit
    window = text[:max_chars]
    last_end = max(
        window.rfind('. '),
        window.rfind('! '),
        window.rfind('? '),
        window.rfind('.\n'),
    )
    if last_end > max_chars // 2:
        return text[:last_end + 1].strip()
    # No good boundary found — fall back to hard cut at max_chars
    return window


def _get_spacy(lang: str):
    """loading spacy model for given language, caching after first load"""
    if lang not in _spacy_models:
        model_name = config.SPACY_MODELS.get(lang)
        if model_name is None:
            return None
        try:
            _spacy_models[lang] = spacy.load(model_name)
        except OSError:
            logger.warning(f'spacy model {model_name} not found — run: python -m spacy download {model_name}')
            _spacy_models[lang] = None
    return _spacy_models[lang]


def score_actionability_keywords(text: str, lang: str) -> dict:
    """
    computing keyword-based actionability sub-scores per dimension
    following Mostafiz et al. (2022) short-term / long-term distinction
    and Zguir et al. (2025) three-category taxonomy (supplies, personnel, actions)

    returns dict with:
      - imperative_score: presence of action verbs (calls to act)
      - short_term_score: immediate danger / response language
      - long_term_score:  recovery / resilience / policy language
      - spatial_score:    geographic anchoring (Xu & Qiang 2022: spatial explicitness)
      - total_score:      weighted composite
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


def extract_srl_features(text: str, lang: str) -> dict:
    """
    lightweight semantic role labelling using spacy dependency parsing
    extracting AGENT (subject), ACTION (verb), THEME (object), LOCATION (prep)
    grounded in Jurafsky (2014) Ch.21 — WHO did WHAT to WHOM WHERE

    not full PropBank SRL — approximating the key roles via dependency labels
    which is sufficient for the actionability signal we need:
    checking if a flood-related action verb has a spatial theme (LOCATION)
    """
    nlp = _get_spacy(lang)
    if nlp is None:
        return {'has_agent': 0, 'has_action': 0, 'has_location': 0, 'srl_complete': 0}

    doc = nlp(_truncate_at_sentence(text))
    agents    = [t for t in doc if t.dep_ in ('nsubj', 'nsubjpass')]
    actions   = [t for t in doc if t.pos_ == 'VERB' and t.dep_ in ('ROOT', 'ccomp', 'xcomp')]
    locations = [t for t in doc if t.ent_type_ in ('GPE', 'LOC', 'FAC')]

    # srl_complete = 1 if article has all three components → most likely to be actionable
    has_a = int(len(agents) > 0)
    has_v = int(len(actions) > 0)
    has_l = int(len(locations) > 0)

    return {
        'has_agent':    has_a,
        'has_action':   has_v,
        'has_location': has_l,
        'srl_complete': int(has_a and has_v and has_l),   # Jurafsky: complete role structure
    }


def extract_named_entities(text: str, lang: str) -> dict:
    """
    Extract top named entities from the article using spacy NER.
    Returns:
        top_locations: JSON-serialisable list of up to 5 unique GPE/LOC/FAC names
        top_orgs:      JSON-serialisable list of up to 5 unique ORG names
    These columns are useful for downstream geographic diffusion analysis (Han et al. 2017)
    and for validating location dictionary coverage.
    """
    import json
    nlp = _get_spacy(lang)
    if nlp is None:
        return {'top_locations': json.dumps([]), 'top_orgs': json.dumps([])}

    doc = nlp(_truncate_at_sentence(text))
    seen_locs: dict[str, int] = {}
    seen_orgs: dict[str, int] = {}

    for ent in doc.ents:
        text_norm = ent.text.strip()
        if not text_norm:
            continue
        if ent.label_ in ('GPE', 'LOC', 'FAC'):
            seen_locs[text_norm] = seen_locs.get(text_norm, 0) + 1
        elif ent.label_ == 'ORG':
            seen_orgs[text_norm] = seen_orgs.get(text_norm, 0) + 1

    # Sort by frequency descending, keep top 5
    top_locs = [k for k, _ in sorted(seen_locs.items(), key=lambda x: -x[1])[:5]]
    top_orgs = [k for k, _ in sorted(seen_orgs.items(), key=lambda x: -x[1])[:5]]

    return {
        'top_locations': json.dumps(top_locs, ensure_ascii=False),
        'top_orgs':      json.dumps(top_orgs, ensure_ascii=False),
    }


def assign_temporal_phase(pub_date, flood_date) -> str:
    """
    assigning article to before / during / after phase
    following Dujardin et al. (2024) temporal design for the 2021 European floods
    and Sit et al. (2020) temporal analysis of Hurricane Irma

    expects pub_date and flood_date as pandas Timestamps or parseable strings
    flood_date is the peak/onset date of the flood event from the events table
    """
    try:
        pub   = pd.to_datetime(pub_date)
        flood = pd.to_datetime(flood_date)
        delta = (pub - flood).days
        if delta < -1:
            return 'before'
        elif delta <= 7:
            return 'during'
        else:
            return 'after'
    except Exception:
        return 'unknown'

def count_verb_tenses(text: str, lang: str) -> dict:
    """
    Identifies and counts verb tenses (past, present, future) using SpaCy.
    Accounts for Spanish morphological inflection vs English auxiliary construction.
    """
    nlp = _get_spacy(lang)
    if nlp is None:
        return {'past_tense': 0, 'present_tense': 0, 'future_tense': 0}

    doc = nlp(_truncate_at_sentence(text))
    
    counts = {'past_tense': 0, 'present_tense': 0, 'future_tense': 0}
    
    for token in doc:
        # We only care about main verbs and auxiliaries
        if token.pos_ in ('VERB', 'AUX'):
            morph_tense = token.morph.get('Tense')
            
            if morph_tense:
                if 'Past' in morph_tense:
                    counts['past_tense'] += 1
                elif 'Pres' in morph_tense:
                    # Intercept English future auxiliary 'will' / 'shall' 
                    # which SpaCy often tags as Present or Finite.
                    if lang == 'en' and token.lemma_ in ('will', 'shall', "'ll"):
                        counts['future_tense'] += 1
                    else:
                        counts['present_tense'] += 1
                elif 'Fut' in morph_tense: 
                    # Natively captures Spanish future tense inflection
                    counts['future_tense'] += 1
            else:
                # Failsafe for English future auxiliaries lacking explicit Tense tags
                if lang == 'en' and token.lemma_ in ('will', 'shall', "'ll"):
                     counts['future_tense'] += 1
                     
    return counts

def run_actionability(df: pd.DataFrame) -> pd.DataFrame:
    """
    running the full actionability enrichment pipeline on preprocessed dataframe
    adds columns: imperative_score, short_term_score, long_term_score,
                  spatial_score, actionability_score,
                  has_agent, has_action, has_location, srl_complete,
                  past_tense, present_tense, future_tense, past_tense_ratio,
                  temporal_phase
    """
    logger.info('scoring actionability...')

    # 1. Keyword-based scoring (fast, no model needed)
    kw_scores = df.apply(
        lambda r: score_actionability_keywords(r['clean_text'], r['language']),
        axis=1,
        result_type='expand'
    )
    df = pd.concat([df, kw_scores], axis=1)

    # 2. Semantic role labelling features (requires spacy)
    logger.info('extracting SRL features...')
    srl_feats = df.apply(
        lambda r: extract_srl_features(r['clean_text'], r['language']),
        axis=1,
        result_type='expand'
    )
    df = pd.concat([df, srl_feats], axis=1)

    # 3. Named entity extraction (Phase 6c)
    logger.info('extracting named entities...')
    ner_feats = df.apply(
        lambda r: extract_named_entities(r['clean_text'], r['language']),
        axis=1,
        result_type='expand'
    )
    df = pd.concat([df, ner_feats], axis=1)

    # 4. Verb tense extraction
    logger.info('extracting verb tenses...')
    tense_feats = df.apply(
        lambda r: count_verb_tenses(r['clean_text'], r['language']),
        axis=1,
        result_type='expand'
    )
    df = pd.concat([df, tense_feats], axis=1)

    # 4. Integrate Past Tense Penalty into Actionability Score
    logger.info('applying past tense penalty to actionability score...')
    
    def calculate_penalty(row):
        total_verbs = row['past_tense'] + row['present_tense'] + row['future_tense']
        if total_verbs == 0:
            return 0.0
        return row['past_tense'] / total_verbs

    df['past_tense_ratio'] = df.apply(calculate_penalty, axis=1)
    
    # Weight for the past tense penalty (adjust this 0.15 based on your data distribution)
    penalty_weight = 0.15 
    df['actionability_score'] = df['actionability_score'] - (penalty_weight * df['past_tense_ratio'])
    
    # Ensure score doesn't drop below zero and round it
    df['actionability_score'] = df['actionability_score'].clip(lower=0).round(4)

    # 5. Temporal phase assignment
    # Priority order:
    #   1. flood_date column (per-row date — set by stage_08 from flood_crawl.csv)
    #   2. FLOOD_REFERENCE_DATES dict (per flood_id — multi-event support)
    #   3. FLOOD_REFERENCE_DATE scalar (single-event fallback)
    if 'flood_date' in df.columns:
        logger.info('assigning temporal phases using flood_date column (per-row)...')
        df['temporal_phase'] = df.apply(
            lambda r: assign_temporal_phase(r.get('pub_date'), r.get('flood_date')),
            axis=1
        )
    elif (
        hasattr(config, 'FLOOD_REFERENCE_DATES')
        and config.FLOOD_REFERENCE_DATES
        and 'flood_id' in df.columns
    ):
        logger.info('assigning temporal phases using FLOOD_REFERENCE_DATES dict (per flood_id)...')
        ref_dates = config.FLOOD_REFERENCE_DATES

        def _phase_from_dict(row):
            fid = row.get('flood_id')
            ref = ref_dates.get(int(fid)) if fid is not None else None
            if ref is None:
                return 'unknown'
            return assign_temporal_phase(row.get('pub_date'), ref)

        df['temporal_phase'] = df.apply(_phase_from_dict, axis=1)
    elif hasattr(config, 'FLOOD_REFERENCE_DATE') and config.FLOOD_REFERENCE_DATE:
        logger.info(f'assigning temporal phases using FLOOD_REFERENCE_DATE={config.FLOOD_REFERENCE_DATE}...')
        df['temporal_phase'] = df['pub_date'].apply(
            lambda pub: assign_temporal_phase(pub, config.FLOOD_REFERENCE_DATE)
        )
    else:
        logger.warning('no flood_date column or FLOOD_REFERENCE_DATE(S) set — skipping temporal phase')
        df['temporal_phase'] = 'unknown'

    logger.info('actionability scoring complete')
    return df
