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

    Creates a table with one row per article (article_id, flood_id, language,) and a column
    containing the list of sentence strings for that article.

    Expected keys in each dict (minimum):
      - flood_id
      - article_id
      - language
      - country
      - clean_text  (string paragraph)
    

    Output columns:
      - flood_id
      - article_id
      - language
      - country
      - list_of_sentences (list[str])
    """
    df = pd.DataFrame(articles)

    # Keep only what we need (and fail loudly if missing)
    required = {'flood_id', 'article_id', 'country', 'language', 'clean_text'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'create_article_df missing required fields: {sorted(missing)}')

    df = df.loc[:, ['article_id', 'flood_id', 'country', 'language', 'clean_text']].copy()
    df['language'] = df['language'].astype('category')

    # Split each article into sentences (spaCy sentencizer)
    df['list_of_sentences'] = df['clean_text'].fillna('').apply(split_into_sentences)

    # Ensure one row per article_id (if duplicates exist, flatten sentence lists)
    df_for_actionability = (
        df.groupby('article_id', as_index=False)
          .agg(
              flood_id=('flood_id', 'first'),
              language=('language', 'first'),
              country=('country', 'first'),
              list_of_sentences=('list_of_sentences', lambda lists: [s for sub in lists for s in sub]),
          )
    )

    return df_for_actionability


def make_sentence_level_df(df_articles: pd.DataFrame) -> pd.DataFrame:
    """Convert an article-level df into a sentence-level df (one row per sentence).

    Expected input columns:
      - flood_id
      - article_id
      - language
      - country
      - list_of_sentences (list[str])

    Expected input columns:
      - flood_id
      - article_id
      - language
      - country
      - sentence
      - sentence_num (0-based index within article)
    """
    required = {'article_id', 'flood_id', 'language', 'country', 'list_of_sentences'}
    missing = required - set(df_articles.columns)
    if missing:
        raise KeyError(f'make_sentence_level_df missing required columns: {sorted(missing)}')

    df = df_articles.loc[:, ['article_id', 'flood_id', 'language', 'country', 'list_of_sentences']].copy()

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
    """Add keyword hit counts to df_by_sentence (one row per sentence), efficiently.

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

    # Ensure we have strings (avoid repeated fillna/astype inside loops)
    df_by_sentence['sentence'] = df_by_sentence['sentence'].fillna('').astype(str)
    df_by_sentence['language'] = df_by_sentence['language'].fillna('en').astype(str)

    # Precompile regex patterns per language + category once
    # (major speedup vs per-row/per-keyword re.search)
    patterns: dict[str, dict[str, re.Pattern[str]]] = {}
    for lang in df_by_sentence['language'].unique():
        kw = _get_kw_dict(lang)
        compiled: dict[str, re.Pattern[str]] = {}

        # Build one alternation regex per category
        for cat, key in [
            ('imperative_count', 'imperative_verbs'),
            ('short_term_count', 'short_term'),
            ('long_term_count', 'long_term'),
            ('spatial_count', 'spatial_anchors'),
        ]:
            words = [w.strip() for w in kw.get(key, []) if isinstance(w, str) and w.strip()]
            if not words:
                compiled[cat] = re.compile(r'(?!)')  # match nothing
            else:
                # Sort longest-first to avoid redundant alternation work (minor)
                words = sorted(set(words), key=len, reverse=True)
                compiled[cat] = re.compile(r'\b(?:' + '|'.join(map(re.escape, words)) + r')\b', flags=re.IGNORECASE)

        patterns[lang] = compiled

    # Allocate output columns
    df_by_sentence['imperative_count'] = 0
    df_by_sentence['short_term_count'] = 0
    df_by_sentence['long_term_count'] = 0
    df_by_sentence['spatial_count'] = 0

    # Compute per language using vectorized str.count
    for lang, idx in df_by_sentence.groupby('language', dropna=False).groups.items():
        lang = str(lang) if lang is not None else 'en'
        pats = patterns.get(lang) or patterns.get('en')
        if pats is None:
            # If even 'en' isn't present for some reason, just keep zeros
            continue

        s = df_by_sentence.loc[idx, 'sentence']
        df_by_sentence.loc[idx, 'imperative_count'] = s.str.count(pats['imperative_count'])
        df_by_sentence.loc[idx, 'short_term_count'] = s.str.count(pats['short_term_count'])
        df_by_sentence.loc[idx, 'long_term_count'] = s.str.count(pats['long_term_count'])
        df_by_sentence.loc[idx, 'spatial_count'] = s.str.count(pats['spatial_count'])

    # Ensure ints (str.count returns int but keep explicit)
    df_by_sentence['imperative_count'] = df_by_sentence['imperative_count'].astype(int)
    df_by_sentence['short_term_count'] = df_by_sentence['short_term_count'].astype(int)
    df_by_sentence['long_term_count'] = df_by_sentence['long_term_count'].astype(int)
    df_by_sentence['spatial_count'] = df_by_sentence['spatial_count'].astype(int)

    return df_by_sentence

def extract_all_actionable_features(df_by_sentence: pd.DataFrame) -> pd.DataFrame:
    """
    Unified extraction using spaCy for both morphology (verbs) and lemmatization (keywords).

    """
    # 1. Prepare Data
    required = {'language', 'sentence'}
    if not required.issubset(df_by_sentence.columns):
        raise KeyError(f'Missing columns: {required - set(df_by_sentence.columns)}')

    extracted_data = []

    # 2. Process by Language Group
    for lang, idx in df_by_sentence.groupby('language', dropna=False).groups.items():
        lang_norm = (lang or 'en').strip().lower() if isinstance(lang, str) else 'en'
        nlp = _get_spacy(lang_norm)
        
        # Load Lemmatized Keyword Dictionaries for this language
        kw_dict = _get_kw_dict(lang_norm)
        # Convert lists to sets for O(1) lookup speed
        short_term_lemmas = set(kw_dict.get('short_term', []))
        long_term_lemmas = set(kw_dict.get('long_term', []))
        spatial_lemmas = set(kw_dict.get('spatial_anchors', []))
        
        if nlp is None or nlp.vocab.length == 0:
            extracted_data.extend([{}] * len(idx))
            continue

        sentences = df_by_sentence.loc[idx, 'sentence'].fillna('').astype(str).tolist()
        
        # 3. Batch Process through spaCy
        for doc in nlp.pipe(sentences):
            row_features = {
                'verbs_imperative': [],
                'verbs_subjunctive': [],
                'auxiliary_modals': [],
                'imperative_count': 0,
                'short_term_count': 0,
                'long_term_count': 0,
                'spatial_count': 0
            }
            
            for tok in doc:
                if tok.is_space or tok.is_punct: 
                    continue
                
                lemma = tok.lemma_.lower()

                # --- A. Evaluate Keywords via Lemmatization ---
                if lemma in spatial_lemmas:
                    row_features['spatial_count'] += 1
                elif lemma in short_term_lemmas:
                    row_features['short_term_count'] += 1
                elif lemma in long_term_lemmas:
                    row_features['long_term_count'] += 1

                # --- B. Evaluate Morphology for Actionability ---
                if tok.pos_ == 'VERB':
                    mood = tok.morph.get('Mood')
                    if 'Imp' in mood:
                        row_features['verbs_imperative'].append(tok.lower_)
                        row_features['imperative_count'] += 1
                    elif 'Sub' in mood:
                        row_features['verbs_subjunctive'].append(tok.lower_)
                        row_features['imperative_count'] += 1 

                elif tok.pos_ == 'AUX':
                    if lemma in ['dever', 'precisar', 'ter', 'must', 'should', 'need',
                        'deber', 'necesitar', 'tener', 'haber']:
                        row_features['auxiliary_modals'].append(lemma)

            extracted_data.append(row_features)

    # 4. Merge Data
    features_df = pd.DataFrame(extracted_data, index=df_by_sentence.index)
    
    # Clean up NA values
    for col in features_df.columns:
        if 'count' in col:
            features_df[col] = features_df[col].fillna(0).astype(int)
        else:
            features_df[col] = features_df[col].apply(lambda x: x if isinstance(x, list) else [])
        
    return pd.concat([df_by_sentence, features_df], axis=1)

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
                df_by_sentence.at[row_i, c] = buckets[c] # not efficient need to fix, will probs create a bottleneck
    return df_by_sentence



def extract_srl_features(df_by_sentence: pd.DataFrame) -> pd.DataFrame:
    """Add SRL-lite features to df_by_sentence (one row per sentence).

    Lightweight semantic role labelling using spaCy dependency parsing extracted in
    `_extract_spacy_features` (agent/action/location + completeness).

    Expected input columns:
      - language (str)
      - sentence (str)

    Adds output columns:
      - has_agent (int: 0/1)
      - has_action (int: 0/1)
      - has_location (int: 0/1)
      - srl_complete (int: 0/1)
    """
    required = {'language', 'sentence'}
    missing = required - set(df_by_sentence.columns)
    if missing:
        raise KeyError(f'extract_srl_features missing required columns: {sorted(missing)}')

    # Ensure we have strings
    df_by_sentence['sentence'] = df_by_sentence['sentence'].fillna('').astype(str)
    df_by_sentence['language'] = df_by_sentence['language'].fillna('en').astype(str)

    # Allocate output columns
    df_by_sentence['has_agent'] = 0
    df_by_sentence['has_action'] = 0
    df_by_sentence['has_location'] = 0
    df_by_sentence['srl_complete'] = 0

    # Process by language for efficiency (reuse the same spaCy model)
    for lang, idx in df_by_sentence.groupby('language', dropna=False).groups.items():
        lang_norm = (lang or 'en').strip().lower() if isinstance(lang, str) else 'en'
        nlp = _get_spacy(lang_norm)

        # If model missing, keep zeros for this group
        if nlp is None or nlp.vocab.length == 0:
            continue

        sentences = df_by_sentence.loc[idx, 'sentence'].tolist()
        docs = list(nlp.pipe(sentences)) if sentences else []

        # Compute SRL-lite per doc (sentence)
        has_agent = []
        has_action = []
        has_location = []
        srl_complete = []

        for doc in docs:
            _has_agent = 0
            _has_action = 0
            _has_location = 0

            for tok in doc:
                # Agent: nominal subject
                if tok.dep_ in ('nsubj', 'nsubj:pass', 'csubj'):
                    _has_agent = 1
                # Action: any verb/aux as a lightweight proxy
                if tok.pos_ in ('VERB', 'AUX'):
                    _has_action = 1
                # Location: prepositional object / oblique nominal (very lightweight)
                # (kept broad because labels vary across languages/models)
                if tok.dep_ in ('pobj', 'obl', 'iobj'):
                    _has_location = 1

            has_agent.append(_has_agent)
            has_action.append(_has_action)
            has_location.append(_has_location)
            srl_complete.append(1 if (_has_agent and _has_action and _has_location) else 0)

        df_by_sentence.loc[idx, 'has_agent'] = has_agent
        df_by_sentence.loc[idx, 'has_action'] = has_action
        df_by_sentence.loc[idx, 'has_location'] = has_location
        df_by_sentence.loc[idx, 'srl_complete'] = srl_complete

    # ensure ints
    for c in ['has_agent', 'has_action', 'has_location', 'srl_complete']:
        df_by_sentence[c] = df_by_sentence[c].fillna(0).astype(int)

    return df_by_sentence






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
