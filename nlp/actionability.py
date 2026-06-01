# nlp/actionability.py
# scoring articles for actionability — the core analytical dimension of the project
# theoretical grounding:
#   Mostafiz et al. (2022): short-term vs long-term actionability dimensions
#   Zade et al. (2018): actionability bias — actionable info is higher utility in disasters
#   Jurafsky (2014) Ch.21: semantic role labelling — WHO did WHAT to WHOM, WHERE, WHEN
#   Kruspe et al. (2021): keyword + ML hybrid approaches for actionability detection
#   Zguir et al. (2025): taxonomy of actionable requests (supplies, personnel, actions)

import os
import re
import logging
import importlib

import numpy as np
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
    """Split a paragraph into sentences with guardrails.

    Extra guardrails:
    - don't allow a split if another sentence-ending punctuation appears within the next 5 chars
      (prevents splitting on abbreviations like "U.S." / "p.m." / "S.P.")
    - only allow a split if the next non-space character starts with a capital letter
      (helps avoid mid-sentence splits, but note: this can be too strict for languages like es/pt)

    Extra tweak:
    - force a sentence break before Spanish inverted question/exclamation if it is glued to previous text
      like: "... circuito.¿Qué ..." -> "... circuito." + "¿Qué ..."
    """
    if text is None:
        return []

    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []

    # --- NEW: force split when inverted punctuation is glued to previous text ---
    # Ensures sentencizer sees a boundary opportunity.
    # Examples fixed:
    #   "...corto circuito.¿Qué hacer...?" -> "...corto circuito." + "¿Qué hacer...?"
    #   "...algo!¿Y ahora...?" -> "...algo!" + "¿Y ahora...?"
    text = re.sub(r"([.!?])(?=(?:¿|¡))", r"\1 ", text)

    # Prevent sentencizer from treating a.m./p.m. as sentence-ending punctuation
    _AMPM_TOKEN = "<AMPM>"
    text = re.sub(r"\b([ap])\.m\.\b", r"\1" + _AMPM_TOKEN, text, flags=re.IGNORECASE)

    nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    doc = nlp(text)

    sents = [s.text.strip() for s in doc.sents if s.text and s.text.strip()]

    # Restore a.m./p.m.
    sents = [
        re.sub(r"\b([ap])" + re.escape(_AMPM_TOKEN) + r"\b", r"\1.m.", s, flags=re.IGNORECASE)
        for s in sents
    ]

    # --- NEW: hard split inside any "sentence" if it still contains a glued inverted punct ---
    # This is a last-resort deterministic split: break at the FIRST occurrence of ¿ or ¡
    # when it's not at the start of the sentence.
    forced: list[str] = []
    for s in sents:
        ss = s.strip()
        if not ss:
            continue
        # split at the first ¿ or ¡ that appears after some text
        m = re.search(r".+?(?=(¿|¡))", ss)
        if m and m.end() > 0 and m.end() < len(ss):
            left = ss[: m.end()].strip()
            right = ss[m.end() :].strip()
            if left:
                forced.append(left)
            if right:
                forced.append(right)
        else:
            forced.append(ss)

    sents = forced

    # --- post-merge "bad boundary" fixer based on original text positions ---
    # NOTE: allow sentence to start with ¿/¡ by skipping them to inspect the next real char.
    _LEADING_QUOTE_CHARS = "\"'“”‘’«»‹›"
    _LEADING_INVERTED_PUNCT = "¿¡"
    _LEADING_BRACKETS = "([{"

    def _next_real_char(start_idx: int) -> str | None:
        """Return next non-space, skipping opening quotes/brackets/inverted punct."""
        if start_idx is None:
            return None

        i = start_idx

        while i < len(text) and text[i].isspace():
            i += 1

        OPENERS = _LEADING_QUOTE_CHARS + _LEADING_BRACKETS + _LEADING_INVERTED_PUNCT
        while i < len(text) and text[i] in OPENERS:
            i += 1
            while i < len(text) and text[i].isspace():
                i += 1

        return None if i >= len(text) else text[i]

    repaired: list[str] = []
    cursor = 0

    for sent in sents:
        pos = text.find(sent.replace(" ", " "), cursor)
        if pos == -1:
            repaired.append(sent)
            continue

        end = pos + len(sent)
        cursor = end

        if not repaired:
            repaired.append(sent)
            continue

        boundary_idx = pos
        lookback = max(0, boundary_idx - 10)

        prev_punct_idx = None
        for j in range(boundary_idx - 1, lookback - 1, -1):
            if text[j] in ".!?":
                prev_punct_idx = j
                break

        merge = False

        if prev_punct_idx is not None:
            window = text[prev_punct_idx + 1 : min(len(text), prev_punct_idx + 1 + 2)]
            if any(ch in ".!?" for ch in window):
                merge = True

        nxt = _next_real_char(boundary_idx)
        if nxt is not None:
            if not re.match(r"[A-ZÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇÑ0-9]", nxt):
                merge = True

        if merge:
            repaired[-1] = f"{repaired[-1]} {sent}".strip()
        else:
            repaired.append(sent)

    sents = repaired

    # --- existing merge rules (abbrev-only, abbrev-end) ---
    ABBREV_ONLY = re.compile(
        r"^(?:"
        r"Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|Gen|Sgt|Capt|Col|Maj|Rep|Sen|Gov|Pres|Rev|Hon|"
        r"Inc|Ltd|Co|Corp|vs|etc|e\.g|i\.e|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|"
        r"a\.m|p\.m"
        r")\.?$",
        flags=re.IGNORECASE,
    )

    ABBREV_END = re.compile(
        r"(?:\b(?:"
        r"Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|Gen|Sgt|Capt|Col|Maj|Rep|Sen|Gov|Pres|Rev|Hon|"
        r"Inc|Ltd|Co|Corp|No|Nos|Art|Sec|Ch|Vol|Fig|Ref|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
        r")\.)$",
        flags=re.IGNORECASE,
    )

    merged: list[str] = []
    i = 0
    while i < len(sents):
        cur = sents[i]
        if ABBREV_ONLY.match(cur) and i + 1 < len(sents):
            sents[i + 1] = f"{cur} {sents[i + 1]}".strip()
            i += 1
            continue
        if ABBREV_END.search(cur) and i + 1 < len(sents):
            sents[i + 1] = f"{cur} {sents[i + 1]}".strip()
            i += 1
            continue
        merged.append(cur)
        i += 1

    # --- existing filters (same as before) ---
    JUNK_FULL = re.compile(
        r"^(?:"
        r"(?:https?://\S+|www\.\S+|pic\.twitter\.com/\S+)"
        r"|fuente\b.*"
        r"|source\b.*"
        r"|mais notícias\b.*"
        r"|más noticias\b.*"
        r"|more news\b.*"
        r")$",
        flags=re.IGNORECASE,
    )


    out: list[str] = []
    for s in merged:
        ss = s.strip().strip("“”\"'`").strip()
        if not ss:
            continue
        if JUNK_FULL.match(ss):
            continue
        if re.search(r"[A-Za-zÀ-ÿ]", ss) is None:
            continue

        if re.search(r"[,;:]\s*$", ss) and len(ss) < 60:
            continue

        letters = len(re.findall(r"[A-Za-zÀ-ÿ]", ss))
        if len(ss) < 20 or letters < 10:
            continue

        out.append(ss)

  
    # If any sentence is > 1000 chars, split it into ~300-char chunks (prefer splitting on whitespace).
    def _chunk_long_sentence(s: str, max_len: int = 1000, chunk_len: int = 300) -> list[str]:
        s = (s or "").strip()
        if len(s) <= max_len:
            return [s] if s else []

        chunks: list[str] = []
        i = 0
        n = len(s)

        while i < n:
            j = min(i + chunk_len, n)
            # try not to cut in the middle of a word: backtrack to last whitespace
            if j < n:
                k = s.rfind(" ", i, j)
                if k != -1 and k > i + 40:  # avoid tiny fragments
                    j = k

            piece = s[i:j].strip()
            if piece:
                chunks.append(piece)

            # advance; if we split at whitespace, skip it
            i = j
            while i < n and s[i].isspace():
                i += 1

        return chunks

    out_chunked: list[str] = []
    for s in out:
        out_chunked.extend(_chunk_long_sentence(s, max_len=1000, chunk_len=300))

    return out_chunked

_WEATHER_TABLEISH_RE = re.compile(
    r"(?is)"
    r"(values represent|national weather service|cooperative observers|24[\s-]*hour|precipitation|station max\s*/\s*min)"
)

def is_tableish_weather_sentence(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return True

    # strong keyword signal
    if _WEATHER_TABLEISH_RE.search(s):
        return True

    # structure: many digits + many slashes/colons (tables)
    digit_count = len(re.findall(r"\d", s))
    sep_count = len(re.findall(r"[/=:]", s))
    upper_tokens = re.findall(r"\b[A-Z]{3,}\b", s)

    if digit_count >= 25 and sep_count >= 12:
        return True
    if len(upper_tokens) >= 12 and sep_count >= 10:
        return True

    return False


#########################################################################################################################

#       3 -  DataFrame Creation ()

#  IMPORTANT: input probaly needs to be changed not sure what
# df_for_actionability is the summarized df to which only final conclusions are added
# df_by_sentence is the expanded df with one row per sentence of each article all the intermediate steps will take place in this df 
#########################################################################################################################


def create_article_df(articles: list[dict]) -> pd.DataFrame:
    """Create a DataFrame from a list of article metadata dictionaries.
    # ...existing docstring...
    """
    df = pd.DataFrame(articles)

    required = {'flood_id', 'article_id', 'country', 'language', 'clean_text'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'create_article_df missing required fields: {sorted(missing)}')

    df = df.loc[:, ['article_id', 'flood_id', 'country', 'language', 'clean_text']].copy()
    df['language'] = df['language'].astype('category')

    # Split each article into sentences
    df['list_of_sentences'] = df['clean_text'].fillna('').apply(split_into_sentences)


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
    # NEW: drop weather-table/bulletin artifacts
    df_by_sentence = df_by_sentence[~df_by_sentence["sentence"].apply(is_tableish_weather_sentence)].reset_index(drop=True)

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
    required = {'language', 'sentence'}
    if not required.issubset(df_by_sentence.columns):
        raise KeyError(f'Missing columns: {required - set(df_by_sentence.columns)}')

    # Pre-create output frame aligned to df_by_sentence index
    # NOTE: imperative_count/short_term_count/long_term_count/spatial_count are already
    # computed by actionable_keyword_count() — do NOT duplicate them here or the concat
    # produces two columns with the same name and the zeros overwrite the real values.
    features_df = pd.DataFrame(index=df_by_sentence.index)
    features_df['verbs_imperative'] = [[] for _ in range(len(df_by_sentence))]
    features_df['verbs_subjunctive'] = [[] for _ in range(len(df_by_sentence))]
    features_df['auxiliary_modals'] = [[] for _ in range(len(df_by_sentence))]
    features_df['subjunctive_count'] = 0
    # NOTE: short_term_count/long_term_count/spatial_count ARE re-computed here via
    # lemmatization (more accurate than regex). imperative_count is NOT added — it is
    # computed by actionable_keyword_count() and must not be overwritten with zeros here.
    features_df['short_term_count'] = 0
    features_df['long_term_count'] = 0
    features_df['spatial_count'] = 0


    for lang, idx in df_by_sentence.groupby('language', dropna=False).groups.items():
        lang_norm = (lang or 'en').strip().lower() if isinstance(lang, str) else 'en'
        nlp = _get_spacy(lang_norm)

        kw_dict = _get_kw_dict(lang_norm)
        short_term_lemmas = set(kw_dict.get('short_term', []))
        long_term_lemmas = set(kw_dict.get('long_term', []))
        spatial_lemmas = set(kw_dict.get('spatial_anchors', []))

        # If model missing, just leave defaults for these rows
        if nlp is None or getattr(nlp, "vocab", None) is None or nlp.vocab.length == 0:
            continue

        # IMPORTANT: preserve row-index order for writing back
        idx_list = list(idx)
        sentences = df_by_sentence.loc[idx_list, 'sentence'].fillna('').astype(str).tolist()

        for row_i, doc in zip(idx_list, nlp.pipe(sentences)):
            row_features = {
                'verbs_imperative': [],
                'verbs_subjunctive': [],
                'auxiliary_modals': [],
                'subjunctive_count': 0,
                'short_term_count': 0,
                'long_term_count': 0,
                'spatial_count': 0,
            }

            for tok in doc:
                if tok.is_space or tok.is_punct:
                    continue

                lemma = (tok.lemma_ or '').lower()

                # keyword counts (lemmatized)
                if lemma in spatial_lemmas:
                    row_features['spatial_count'] += 1
                elif lemma in short_term_lemmas:
                    row_features['short_term_count'] += 1
                elif lemma in long_term_lemmas:
                    row_features['long_term_count'] += 1

                # morphology counts
                if tok.pos_ == 'VERB':
                    mood = tok.morph.get('Mood')
                    if 'Imp' in mood:
                        row_features['verbs_imperative'].append(tok.lower_)

                    elif 'Sub' in mood:
                        row_features['verbs_subjunctive'].append(tok.lower_)
  

                elif tok.pos_ == 'AUX':
                    if lemma in [
                        'dever', 'precisar', 'ter', 'must', 'should', 'need',
                        'deber', 'necesitar', 'tener', 'haber'
                    ]:
                        row_features['auxiliary_modals'].append(lemma)

            # write back aligned by row index
            for k, v in row_features.items():
                features_df.at[row_i, k] = v

    for c in ['verbs_imperative', 'verbs_subjunctive', 'auxiliary_modals']:
        features_df[c] = features_df[c].apply(lambda x: x if isinstance(x, list) else [])

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


#########################################################################################################################

#           6 -  Identify advise   

#########################################################################################################################



def add_advice_flag(df_by_sentence: pd.DataFrame) -> pd.DataFrame:
    """Add a binary 'advice' column (0/1) for sentences that contain advice-like cues.

    Strategy:
      1) Try spaCy lemma match (precision-friendly).
      2) If spaCy unavailable OR lemma match misses, use regex fallback (recall-friendly).

    Output:
      - advice (int 0/1)
    """
    required = {'language', 'sentence'}
    missing = required - set(df_by_sentence.columns)
    if missing:
        raise KeyError(f'add_advice_flag missing required columns: {sorted(missing)}')

    df = df_by_sentence.copy()
    df['language'] = df['language'].fillna('en').astype(str).str.strip().str.lower()
    df['sentence'] = df['sentence'].fillna('').astype(str)

    df['advice'] = 0

    lemma_targets = {
        'en': {'recommend', 'advise', 'suggest', 'urge'},
        'es': {'recomendar', 'aconsejar', 'sugerir'},
        'pt': {'recomendar', 'aconselhar', 'sugerir'},
    }

    # Regex fallback patterns (broader coverage; tune as needed)
    advice_regex = {
        'en': re.compile(r"\b(?:recommend|recommends|recommended|recommending|suggest|suggests|suggested|suggesting|advises|advise|advised|advising|urge|urges|urged|urging)\b", re.IGNORECASE),
        'es': re.compile(r"\b(?:recomend\w*|aconsej\w*|suger\w*)\b", re.IGNORECASE),
        'pt': re.compile(r"\b(?:recomend\w*|aconselh\w*|suger\w*|urgen)\b", re.IGNORECASE),
    }

    for lang, idx in df.groupby('language', dropna=False).groups.items():
        lang_norm = (lang or 'en').strip().lower()
        if lang_norm not in advice_regex:
            continue

        sents = df.loc[idx, 'sentence'].tolist()

        # 1) try spaCy lemma method if available
        lemma_flags = None
        if lang_norm in lemma_targets:
            nlp = _get_spacy(lang_norm)
            if nlp is not None and getattr(nlp, "vocab", None) is not None and nlp.vocab.length > 0:
                targets = lemma_targets[lang_norm]
                flags: list[int] = []
                for doc in nlp.pipe(sents):
                    hit = 0
                    for tok in doc:
                        if tok.is_space or tok.is_punct:
                            continue
                        lemma = (tok.lemma_ or '').lower()
                        if lemma in targets:
                            hit = 1
                            break
                    flags.append(hit)
                lemma_flags = flags

        # 2) regex fallback (and union with lemma_flags if present)
        rgx = advice_regex[lang_norm]
        regex_flags = [1 if rgx.search(s or "") else 0 for s in sents]

        if lemma_flags is None:
            df.loc[idx, 'advice'] = regex_flags
        else:
            # union: mark advice if either method hits
            df.loc[idx, 'advice'] = [int(a or b) for a, b in zip(lemma_flags, regex_flags)]

    df['advice'] = df['advice'].fillna(0).astype(int)
    return df


#########################################################################################################################

#           7 -  run actionability   

#########################################################################################################################

def run_actionability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full actionability pipeline: article df → sentence-level df with actionability scores.

    Input: preprocessed df with columns:
        article_id (auto-generated if absent), flood_id, country, language, clean_text

    Output: df_by_sentence with all feature columns plus:
        - actionability_probability (float 0–1)
        - actionability_score (int: 0=none, 1=low 0–0.5, 2=high >0.5)
    """
    logger.info('running actionability pipeline...')

    # ensure article_id exists
    if 'article_id' not in df.columns:
        df = df.copy()
        df['article_id'] = range(len(df))

    # 1. article-level df with sentence lists
    articles = df.to_dict(orient='records')
    df_articles = create_article_df(articles)

    # 2. sentence-level df
    df_by_sentence = make_sentence_level_df(df_articles)
    print(f'[actionability] sentence df shape: {df_by_sentence.shape}')
    print(f'[actionability] sentence df columns: {list(df_by_sentence.columns)}')

    # 3. load + verify spacy models for every language present
    for lang in df_by_sentence['language'].dropna().unique():
        lang_norm = str(lang).strip().lower()
        model = _get_spacy(lang_norm)
        model_name = config.SPACY_MODELS.get(lang_norm, 'not configured')
        if model is not None:
            print(f'[actionability] spacy model loaded successfully: {model_name} (lang={lang_norm})')
        else:
            print(f'[actionability] WARNING: spacy model unavailable for lang={lang_norm} ({model_name})')

    # 4. keyword counts (regex-based)
    df_by_sentence = actionable_keyword_count(df_by_sentence)

    # 5. POS components
    df_by_sentence = add_sentence_pos_components(df_by_sentence)

    # 6. lemmatized counts + morphology features
    df_by_sentence = extract_all_actionable_features(df_by_sentence)

    # drop duplicate count columns — keep last (lemmatized, from step 6)
    df_by_sentence = df_by_sentence.loc[:, ~df_by_sentence.columns.duplicated(keep='last')]

    # 7. SRL features
    df_by_sentence = extract_srl_features(df_by_sentence)

    # 8. advice features
    df_by_sentence = add_advice_flag(df_by_sentence)

    # 9. confirm all features present
    print(f'[actionability] feature-enriched df shape: {df_by_sentence.shape}')
    print(f'[actionability] feature-enriched df columns: {list(df_by_sentence.columns)}')

    # 10. calculate actionability density: weighted feature sum 
    word_count = (
        df_by_sentence['sentence'].str.split().str.len()
        .fillna(1).clip(lower=1).astype(float)
    )

    def _col(name: str) -> pd.Series:
        if name in df_by_sentence.columns:
            return df_by_sentence[name].fillna(0).astype(float)
        return pd.Series(0.0, index=df_by_sentence.index)

    def _list_len(name: str) -> pd.Series:
        if name in df_by_sentence.columns:
            return df_by_sentence[name].apply(
                lambda x: float(len(x)) if isinstance(x, list) else 0.0
            )
        return pd.Series(0.0, index=df_by_sentence.index)

    density_raw = (
          3.0 * _col('imperative_count')
        + 1.5 * _col('short_term_count')
        + 1.5 * _col('long_term_count')
        + 1.0 * _col('spatial_count')
        + 2.0 * _list_len('verbs_imperative')
        + 1.5 * _list_len('verbs_subjunctive')
        + 1.5 * _list_len('auxiliary_modals')
        + 1.0 * _col('srl_complete')
        + 20.0 * _col('advice')
    ) 
   
    density = density_raw 

    # 10. standardize density → actionability_probability in [0, 1] via min-max
    d_min = density.min()
    d_max = density.max()
    if d_max > d_min:
        prob = ((density - d_min) / (d_max - d_min))
    else:
        prob = pd.Series(0.0, index=df_by_sentence.index)

    df_by_sentence['actionability_probability'] = prob.astype(float)

    # 11. actionability_score: 0 = (0-0.2], 1 = low (0.2–0.7], 2 = high (>0.7)
    df_by_sentence['actionability_score'] = np.select(
        [
            (prob > 0.0) & (prob <= 0.2),
            (prob > 0.2) & (prob <= 0.7),
            prob > 0.7,
        ],
        [0, 1, 2],
        default=0,
    ).astype(int)

    # save sentence-level output to CSV
    sentences_path = os.path.join(config.OUTPUT_DIR, 'sentences_actionability.csv')
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    df_by_sentence.to_csv(sentences_path, index=False)
    logger.info('sentence-level output saved -> %s', sentences_path)

    # aggregate to article level and merge back onto the input df
    df_article_scores = calculate_article_actionability(df_by_sentence)

    # aggregate sub-score columns to article level (mean per article)
    # these feed directly into clustering as the feature matrix
    _sub_score_cols = [
        c for c in [
            'imperative_count', 'short_term_count', 'long_term_count',
            'spatial_count', 'advice', 'srl_complete',
            'has_agent', 'has_action', 'has_location',
            'actionability_probability',
        ]
        if c in df_by_sentence.columns
    ]
    if _sub_score_cols:
        _sub_means = (
            df_by_sentence[['article_id'] + _sub_score_cols]
            .groupby('article_id', as_index=False)
            .mean()
            .rename(columns={c: f'mean_{c}' for c in _sub_score_cols})
        )
        df_article_scores = df_article_scores.merge(_sub_means, on='article_id', how='left')

    df = df.merge(
        df_article_scores.drop(columns=['flood_id', 'country', 'language'], errors='ignore'),
        on='article_id', how='left'
    )

    logger.info('actionability pipeline complete — %d sentences, %d articles', len(df_by_sentence), len(df))
    return df


def calculate_article_actionability(df_by_sentence: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentence-level scores into an article-level actionability percentage.

    Scoring weights:
      - actionability_score == 1  → 0.5 actionable sentence (low)
      - actionability_score == 2  → 1.0 actionable sentence (high)
      - actionability_score == 0  → 0.0

    actionability_percentage = weighted_actionable_count / total_sentences * 100

    Input: df_by_sentence with columns: article_id, actionability_score
    Output: df_for_actionability — one row per article with actionability_percentage
    """
    required = {'article_id', 'actionability_score'}
    missing = required - set(df_by_sentence.columns)
    if missing:
        raise KeyError(f'calculate_article_actionability missing required columns: {sorted(missing)}')

    weight_map = {0: 0.0, 1: 0.5, 2: 1.0}

    df = df_by_sentence.copy()
    df['_weighted'] = df['actionability_score'].map(weight_map).fillna(0.0)

    df_for_actionability = (
        df.groupby('article_id', as_index=False)
          .agg(
              total_sentences=('actionability_score', 'count'),
              weighted_actionable=('_weighted', 'sum'),
          )
    )

    df_for_actionability['actionability_percentage'] = (
        (df_for_actionability['weighted_actionable'] / df_for_actionability['total_sentences'].clip(lower=1))
        * 100
    ).round(2)

    df_for_actionability = df_for_actionability.drop(columns=['weighted_actionable'])

    # carry over article-level metadata if present in df_by_sentence
    meta_cols = [c for c in ('flood_id', 'country', 'language') if c in df_by_sentence.columns]
    if meta_cols:
        meta = (
            df_by_sentence[['article_id'] + meta_cols]
            .drop_duplicates('article_id')
        )
        df_for_actionability = df_for_actionability.merge(meta, on='article_id', how='left')

    logger.info(
        'article actionability calculated — %d articles, mean %.1f%%',
        len(df_for_actionability),
        df_for_actionability['actionability_percentage'].mean(),
    )
    return df_for_actionability
