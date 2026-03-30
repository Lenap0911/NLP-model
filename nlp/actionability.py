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

    doc = nlp(text[:1000])   # 1000 chars sufficient; full text too slow
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


def run_actionability(df: pd.DataFrame) -> pd.DataFrame:
    """
    running the full actionability enrichment pipeline on preprocessed dataframe
    adds columns: imperative_score, short_term_score, long_term_score,
                  spatial_score, actionability_score,
                  has_agent, has_action, has_location, srl_complete,
                  temporal_phase
    """
    logger.info('scoring actionability...')

    # keyword-based scoring (fast, no model needed)
    kw_scores = df.apply(
        lambda r: score_actionability_keywords(r['clean_text'], r['language']),
        axis=1,
        result_type='expand'
    )
    df = pd.concat([df, kw_scores], axis=1)

    # semantic role labelling features (requires spacy)
    logger.info('extracting SRL features...')
    srl_feats = df.apply(
        lambda r: extract_srl_features(r['clean_text'], r['language']),
        axis=1,
        result_type='expand'
    )
    df = pd.concat([df, srl_feats], axis=1)

    # temporal phase assignment
    # uses flood_date column if present; otherwise falls back to
    # config.FLOOD_REFERENCE_DATE (set to the flood onset date for the dataset)
    # pub_date from the CSV is compared against the reference date
    if 'flood_date' in df.columns:
        logger.info('assigning temporal phases using flood_date column...')
        df['temporal_phase'] = df.apply(
            lambda r: assign_temporal_phase(r.get('pub_date'), r.get('flood_date')),
            axis=1
        )
    elif hasattr(config, 'FLOOD_REFERENCE_DATE') and config.FLOOD_REFERENCE_DATE:
        logger.info(f'assigning temporal phases using FLOOD_REFERENCE_DATE={config.FLOOD_REFERENCE_DATE}...')
        df['temporal_phase'] = df['pub_date'].apply(
            lambda pub: assign_temporal_phase(pub, config.FLOOD_REFERENCE_DATE)
        )
    else:
        logger.warning('no flood_date column or FLOOD_REFERENCE_DATE set — skipping temporal phase')
        df['temporal_phase'] = 'unknown'

    logger.info('actionability scoring complete')
    return df
