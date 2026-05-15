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

####        1      ####
def _get_spacy(lang: str):
    """loading spacy model for given language, caching after first load
    input: language code (e.g. 'en', 'es', 'pt')
    output: spacy model or None
    """
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

####        2       ####
def split_into_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences using spaCy Sentencizer.
    input: text string (article clean_text)
    output: Returns a list of sentence strings
    """
    nlp = spacy.blank('xx')
    nlp.add_pipe('sentencizer')
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


#########################################################################################################################

#       3 -  DataFrame Creation ()

#  IMPORTANT: input probaly needs to be changed not sure what
# df_for_actionability is the summarized df to which only final conclusions are added
# df_by_sentence is the expanded df with one row per sentence of each article all the intermediate steps will take place in this df 
#########################################################################################################################

def create_article_df(articles: list[dict]) -> pd.DataFrame: 
    """Create a DataFrame from a list of article metadata dictionaries.

    Creates a table with one row per (doc_num, flood_id, language) and a column
    containing the list of sentence strings for that article.

    Expected keys in each dict (minimum):
      - doc_num
      - flood_id
      - language
      - clean_text  (string paragraph)

    Output columns:
      - doc_num
      - flood_id
      - language
      - list_of_sentences (list[str])
    """
    df = pd.DataFrame(articles)

    # Keep only what we need (and fail loudly if missing)
    required = {'doc_num', 'flood_id', 'language', 'clean_text'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'create_article_df missing required fields: {sorted(missing)}')

    df = df.loc[:, ['doc_num', 'flood_id', 'language', 'clean_text']].copy()
    df['language'] = df['language'].astype('category')

    # Split each article into sentences (spaCy sentencizer)
    df['list_of_sentences'] = df['clean_text'].fillna('').apply(split_into_sentences)

    # One row per (doc_num, flood_id, language)
    df_for_actionability = (
    df.groupby(['doc_num', 'flood_id', 'language'], as_index=False)
      .agg(list_of_sentences=('list_of_sentences',
                              lambda lists: [s for sub in lists for s in sub]))
    )

    return df_for_actionability


def make_sentence_level_df(df_articles: pd.DataFrame) -> pd.DataFrame:
    """Convert an article-level df into a sentence-level df (one row per sentence).

    Expected input columns:
      - doc_num
      - flood_id
      - language
      - list_of_sentences (list[str])

    Output columns:
      - doc_num
      - flood_id
      - language
      - sentence
      - sentence_num (0-based index within article)
    """
    required = {'doc_num', 'flood_id', 'language', 'list_of_sentences'}
    missing = required - set(df_articles.columns)
    if missing:
        raise KeyError(f'make_sentence_level_df missing required columns: {sorted(missing)}')

    df = df_articles.loc[:, ['doc_num', 'flood_id', 'language', 'list_of_sentences']].copy()

    # Ensure list type (avoid explode issues)
    df['list_of_sentences'] = df['list_of_sentences'].apply(
        lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [str(x)])
    )

    # Keep sentence index before exploding
    df['sentence_num'] = df['list_of_sentences'].apply(lambda sents: list(range(len(sents))))
    df = df.explode(['list_of_sentences', 'sentence_num'], ignore_index=True)

    df = df.rename(columns={'list_of_sentences': 'sentence'})

    # Optional: drop empty sentences
    df['sentence'] = df['sentence'].fillna('').astype(str)
    df_by_sentence = df[df['sentence'].str.strip() != ''].reset_index(drop=True)

    return df_by_sentence


#########################################################################################################################

#       4 -  Actionable Keyword count 

#########################################################################################################################

####  Auxiliary function: Keyword Dictionary
def _get_kw_dict(lang: str) -> dict:
    """Return keyword dictionary for language; fall back to English."""
    lang_norm = (lang or 'en').strip().lower()
    kw = config.ACTIONABILITY_KEYWORDS.get(lang_norm) or config.ACTIONABILITY_KEYWORDS['en']
    # Guarantee expected keys exist (empty list fallback)
    return {
        'imperative_verbs': kw.get('imperative_verbs', []),
        'short_term': kw.get('short_term', []),
        'long_term': kw.get('long_term', []),
        'spatial_anchors': kw.get('spatial_anchors', []),
    }


def actionable_keyword_count(df_by_sentence: pd.DataFrame) -> pd.DataFrame:
    """Add keyword hit counts to df_by_sentence (one row per sentence).

    Expected input columns:
      - language (str)
      - sentence (str)

    Adds output columns (int per row):
      - imperative_count
      - short_term_count
      - long_term_count
      - spatial_count
    """
    required = {'language', 'sentence'}
    missing = required - set(df_by_sentence.columns)
    if missing:
        raise KeyError(f'actionable_keyword_count missing required columns: {sorted(missing)}')

    def _count_hits(sentence: str, keywords: list[str]) -> int:
        s = (sentence or "").lower()
        return sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', s))

    def _per_sentence(sentence: str, lang: str) -> dict:
        kw_dict = _get_kw_dict(lang)
        return {
            'imperative_count': _count_hits(sentence, kw_dict['imperative_verbs']),
            'short_term_count': _count_hits(sentence, kw_dict['short_term']),
            'long_term_count': _count_hits(sentence, kw_dict['long_term']),
            'spatial_count': _count_hits(sentence, kw_dict['spatial_anchors']),
        }

    expanded = df_by_sentence.apply(
        lambda r: _per_sentence(r['sentence'], r['language']),
        axis=1,
        result_type='expand'
    )

    for c in expanded.columns:
        df_by_sentence[c] = expanded[c]

    return df_by_sentence

#########################################################################################################################

#           5 -  SLR Features  

#########################################################################################################################


def add_sentence_pos_components(df_by_sentence: pd.DataFrame) -> pd.DataFrame:
    """Add POS component token lists to df_by_sentence (one row per sentence).

    Expected input df columns:
      - language (str)
      - sentence (str)

    Adds output columns (each cell is list[str]):
      - adjective, adposition, adverb, auxiliary, coordinating conjunction,
        determiner, interjection, noun, numeral, particle, pronoun, proper noun,
        punctuation, subordinating conjunction, symbol, verb, other
    """
    required = {'language', 'sentence'}
    missing = required - set(df_by_sentence.columns)
    if missing:
        raise KeyError(f'add_sentence_pos_components missing required columns: {sorted(missing)}')

    pos_map = {
        'ADJ': 'adjective',
        'ADP': 'adposition',
        'ADV': 'adverb',
        'AUX': 'auxiliary',
        'CCONJ': 'coordinating conjunction',
        'DET': 'determiner',
        'INTJ': 'interjection',
        'NOUN': 'noun',
        'NUM': 'numeral',
        'PART': 'particle',
        'PRON': 'pronoun',
        'PROPN': 'proper noun',
        'PUNCT': 'punctuation',
        'SCONJ': 'subordinating conjunction',
        'SYM': 'symbol',
        'VERB': 'verb',
        'X': 'other',
    }
    out_cols = list(pos_map.values())

    # initialize columns so the schema is stable
    for c in out_cols:
        if c not in df_by_sentence.columns:
            df_by_sentence[c] = [[] for _ in range(len(df_by_sentence))]

    # Process by language for efficiency
    for lang, idx in df_by_sentence.groupby('language', dropna=False).groups.items():
        lang_norm = (lang or 'en').strip().lower() if isinstance(lang, str) else 'en'
        nlp = _get_spacy(lang_norm) or spacy.blank('xx')

        sentences = df_by_sentence.loc[idx, 'sentence'].fillna('').astype(str).tolist()
        docs = list(nlp.pipe(sentences)) if sentences else []

        for row_i, doc in zip(idx, docs):
            buckets = {c: [] for c in out_cols}
            for tok in doc:
                if tok.is_space:
                    continue
                col = pos_map.get(tok.pos_)
                if col is not None:
                    buckets[col].append(tok.text)

            for c in out_cols:
                df_by_sentence.at[row_i, c] = buckets[c]
    return df_by_sentence





def extract_srl_features(text: str, lang: str) -> dict:
    """
    lightweight semantic role labelling using spacy dependency parsing
    extracting AGENT (subject), ACTION (verb), THEME (object), LOCATION (prep)
    grounded in Jurafsky (2014) Ch.21 — WHO did WHAT to WHOM WHERE

    not full PropBank SRL — approximating the key roles via dependency labels
    which is sufficient for the actionability signal we need:
    checking if a flood-related action verb has a spatial theme (LOCATION)
    """
    feats = _extract_spacy_features(text, lang)
    return {
        'has_agent': feats['has_agent'],
        'has_action': feats['has_action'],
        'has_location': feats['has_location'],
        'srl_complete': feats['srl_complete'],
    }



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
"""
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

"""
###############################




def run_actionability(df: pd.DataFrame) -> pd.DataFrame:
    """
    running the full actionability enrichment pipeline on preprocessed dataframe
    adds columns: imperative_score, short_term_score, long_term_score,
                  spatial_score, actionability_score,
                  has_agent, has_action, has_location, srl_complete,
                  past_tense, present_tense, future_tense, past_tense_ratio
    """
    logger.info('scoring actionability...')

    # 1. Keyword-based scoring (fast, no model needed)
    kw_scores = df.apply(
        lambda r: score_actionability_keywords(r['clean_text'], r['language']),
        axis=1,
        result_type='expand'
    )
    df = pd.concat([df, kw_scores], axis=1)

    # 2. spaCy-derived features (SRL-lite + NER + verb tenses)
    # Parse once per row and reuse the Doc to avoid 3x repeated spaCy calls.
    logger.info('extracting SRL/NER/tense features (single spaCy pass)...')
    spacy_feats = df.apply(
        lambda r: _extract_spacy_features(r['clean_text'], r['language']),
        axis=1,
        result_type='expand'
    )
    df = pd.concat([df, spacy_feats], axis=1)

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

    logger.info('actionability scoring complete')
    return df
