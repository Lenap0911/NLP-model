# nlp/authority.py
# source authority scoring for flood news articles
# theoretical grounding:
#   Gordon (2000): authority as a dimension of source credibility in news
#   Khawaja et al. (2025): Global North vs Global South media framing
#   El Ouadi (2025): domain-level credibility in cross-lingual flood comparison
#
# Two dimensions scored per article:
#   1. scope:       local / regional / national / international
#   2. credibility: tier 1 (major newswire/public broadcaster) to tier 3 (unknown)
#
# Scores are heuristic and domain-based — sufficient for comparative analysis
# across EN/ES/PT corpora without requiring external APIs.

import re
import importlib
import logging

import pandas as pd

config = importlib.import_module('config.nlp_config')
logger = logging.getLogger(__name__)


# ── Scope classification ───────────────────────────────────────────────────────
# Domains are assigned a scope label.  Any domain not in any set defaults to
# "local" — the most conservative assumption.

_INTERNATIONAL_DOMAINS = frozenset({
    # Newswires + global English
    'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'theguardian.com',
    'aljazeera.com', 'aljazeera.net', 'france24.com', 'dw.com', 'euronews.com',
    'voanews.com', 'rt.com', 'bloomberg.com', 'time.com', 'newsweek.com',
    # International humanitarian / disaster
    'reliefweb.int', 'floodlist.com', 'ifrc.org', 'unocha.org',
    'preventionweb.net', 'gdacs.org',
    # International wire services in Spanish/Portuguese
    'efe.com', 'lusa.pt',
})

_NATIONAL_DOMAINS: dict[str, frozenset] = {
    # USA
    'USA': frozenset({
        'foxnews.com', 'cnn.com', 'nbcnews.com', 'cbsnews.com', 'abcnews.go.com',
        'msnbc.com', 'usatoday.com', 'washingtonpost.com', 'nytimes.com',
        'latimes.com', 'npr.org', 'thehill.com', 'politico.com', 'univision.com',
        'telemundo.com', 'weather.com', 'accuweather.com',
    }),
    # Spain
    'ESP': frozenset({
        'elpais.com', 'elmundo.es', 'abc.es', 'lavanguardia.com',
        '20minutos.es', 'rtve.es', 'elconfidencial.com', 'eldiario.es',
    }),
    # Brazil
    'BRA': frozenset({
        'globo.com', 'uol.com.br', 'folha.uol.com.br', 'g1.globo.com',
        'oglobo.globo.com', 'estadao.com.br', 'r7.com', 'noticias.uol.com.br',
    }),
    # Argentina
    'ARG': frozenset({
        'infobae.com', 'lanacion.com.ar', 'clarin.com', 'ambito.com',
    }),
    # Colombia
    'COL': frozenset({
        'semana.com', 'eltiempo.com', 'elespectador.com', 'caracol.com.co',
    }),
    # Mexico
    'MEX': frozenset({
        'milenio.com', 'eluniversal.com.mx', 'excelsior.com.mx',
        'reforma.com', 'jornada.com.mx',
    }),
    # Peru
    'PER': frozenset({'elcomercio.pe', 'larepublica.pe', 'andina.pe'}),
    # Bolivia
    'BOL': frozenset({'lostiempos.com', 'eldeber.com.bo', 'erbol.com.bo'}),
    # Honduras
    'HND': frozenset({'laprensa.hn', 'elheraldo.hn', 'proceso.hn'}),
    # Costa Rica
    'CRI': frozenset({'nacion.com', 'crhoy.com', 'teletica.com'}),
    # Dominican Republic
    'DOM': frozenset({'listindiario.com', 'diariolibre.com', 'elcaribe.com.do'}),
}

# ── Credibility tiers ─────────────────────────────────────────────────────────
# Tier 1: major newswires, public broadcasters, established nationals
# Tier 2: established regional/local journalism, institutional sources
# Tier 3: unknown or low-signal domains

_TIER1_DOMAINS = frozenset({
    'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'theguardian.com',
    'nytimes.com', 'washingtonpost.com', 'latimes.com', 'npr.org',
    'efe.com', 'lusa.pt', 'france24.com', 'dw.com',
    'rtve.es', 'elpais.com', 'globo.com', 'folha.uol.com.br', 'estadao.com.br',
    'g1.globo.com', 'eltiempo.com', 'elespectador.com', 'semana.com',
    'nacion.com', 'laprensa.hn', 'listindiario.com', 'andina.pe', 'elcomercio.pe',
    'lostiempos.com', 'infobae.com', 'lanacion.com.ar', 'clarin.com',
    'reliefweb.int', 'ifrc.org', 'unocha.org',
})

_TIER2_DOMAINS_RE = re.compile(
    r'\.(gov|gob|edu|org|mil)(\.[a-z]{2})?$', re.IGNORECASE
)


def classify_scope(domain: str, country_iso: str = None) -> str:
    """
    Return scope label for a given domain: 'international', 'national', 'local'.
    country_iso is the ISO code of the flood event (to look up national domain list).
    """
    domain = domain.lower().removeprefix('www.')
    if domain in _INTERNATIONAL_DOMAINS:
        return 'international'
    if country_iso and domain in _NATIONAL_DOMAINS.get(country_iso, frozenset()):
        return 'national'
    # Check all country lists if no ISO provided
    if not country_iso:
        for nat_set in _NATIONAL_DOMAINS.values():
            if domain in nat_set:
                return 'national'
    return 'local'


def classify_credibility(domain: str) -> int:
    """
    Return credibility tier (1, 2, or 3) for a given domain.
    Tier 1: major newswires / established nationals
    Tier 2: government / educational / NGO domains
    Tier 3: unknown
    """
    domain = domain.lower().removeprefix('www.')
    if domain in _TIER1_DOMAINS:
        return 1
    if _TIER2_DOMAINS_RE.search(domain):
        return 2
    return 3


def score_authority(domain: str, country_iso: str = None) -> dict:
    """
    Return authority scores for one article given its domain.
    Returns:
        scope:              'international' | 'national' | 'local'
        scope_score:        3=international, 2=national, 1=local (for numeric analysis)
        credibility_tier:   1 (highest) – 3 (lowest)
        authority_score:    composite = scope_score * (4 - credibility_tier)
                            range [1, 9] where 9 = international tier-1
    """
    scope = classify_scope(domain, country_iso)
    tier  = classify_credibility(domain)

    scope_score = {'international': 3, 'national': 2, 'local': 1}.get(scope, 1)
    authority_score = scope_score * (4 - tier)  # tier 1 → 3pts, tier 3 → 1pt

    return {
        'scope':            scope,
        'scope_score':      scope_score,
        'credibility_tier': tier,
        'authority_score':  authority_score,
    }


def run_authority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add authority columns to the dataframe.
    Expects 'domain' column (added by stage_08). Uses 'country' column for ISO lookup.
    """
    logger.info('scoring source authority...')

    # Build a flood_id -> ISO map from the country column if available
    # We use the domain column directly since that's already extracted
    if 'domain' not in df.columns:
        logger.warning("'domain' column missing — authority scoring skipped")
        df['scope'] = 'unknown'
        df['scope_score'] = 0
        df['credibility_tier'] = 3
        df['authority_score'] = 0
        return df

    # Derive ISO from country column heuristically if needed
    # (country is the full country name from flood_crawl.csv)
    _country_to_iso = {
        'united states of america': 'USA',
        'colombia': 'COL',
        'brazil': 'BRA',
        'argentina': 'ARG',
        'mexico': 'MEX',
        'spain': 'ESP',
        'peru': 'PER',
        'bolivia (plurinational state of)': 'BOL',
        'honduras': 'HND',
        'costa rica': 'CRI',
        'dominican republic': 'DOM',
    }

    def _get_iso(row):
        country = str(row.get('country', '')).lower()
        return _country_to_iso.get(country)

    authority_rows = df.apply(
        lambda r: score_authority(str(r.get('domain', '')), _get_iso(r)),
        axis=1,
        result_type='expand',
    )
    df = pd.concat([df, authority_rows], axis=1)
    logger.info('authority scoring complete')
    return df
