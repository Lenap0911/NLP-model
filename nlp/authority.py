# nlp/authority.py
# source authority classification grounded in the actual dataset domains
# theoretical grounding:
#   Gordon (2000): authority as a dimension of source credibility in news
#   Khawaja et al. (2025): Global North vs Global South media framing
#
# Two dimensions per article:
#   scope:       government | national | regional | local | ngo
#   source_type: government_agency | ngo | national_news | regional_news |
#                local_news | radio | unknown
#
# Domain lookup built from verified_articles_clean_text.csv (34 unique domains).
# Unknown domains fall back to heuristics (.gov/.gob → government_agency,
# .org → ngo, everything else → unknown/local).

import re
import importlib
import logging

import pandas as pd

config = importlib.import_module('config.nlp_config')
logger = logging.getLogger(__name__)


# ── Per-domain lookup (built from actual dataset) ─────────────────────────────
# Keys are bare domains (no www. prefix).
# scope:       government | national | regional | local | ngo
# source_type: government_agency | ngo | national_news | regional_news |
#              local_news | radio | unknown

_DOMAIN_LOOKUP: dict[str, dict[str, str]] = {
    # ── Brazil — government ───────────────────────────────────────────────────
    'www2.cemaden.gov.br': {
        'scope': 'government',
        'source_type': 'government_agency',
        # CEMADEN: national early-warning centre — not journalism
    },
    # ── Panama — government ───────────────────────────────────────────────────
    'miambiente.gob.pa': {
        'scope': 'government',
        'source_type': 'government_agency',
        # Panama Ministry of Environment
    },
    # ── Panama — NGO ──────────────────────────────────────────────────────────
    'cruzroja.org.pa': {
        'scope': 'ngo',
        'source_type': 'ngo',
        # Panama Red Cross
    },
    # ── Brazil — national news ────────────────────────────────────────────────
    'istoedinheiro.com.br': {
        'scope': 'national',
        'source_type': 'national_news',
        # IstoÉ Dinheiro — national business magazine
    },
    # ── Mexico — national news ────────────────────────────────────────────────
    'elfinanciero.com.mx': {
        'scope': 'national',
        'source_type': 'national_news',
        # El Financiero — major national financial daily
    },
    'piedepagina.mx': {
        'scope': 'national',
        'source_type': 'national_news',
        # Pie de Página — national investigative journalism
    },
    # ── Colombia — national news ───────────────────────────────────────────────
    'semana.com': {
        'scope': 'national',
        'source_type': 'national_news',
        # Semana — flagship Colombian weekly news magazine
    },
    'larepublica.co': {
        'scope': 'national',
        'source_type': 'national_news',
        # La República — national business newspaper
    },
    'elespectador.com': {
        'scope': 'national',
        'source_type': 'national_news',
        # El Espectador — one of Colombia's oldest national dailies
    },
    # ── Peru — national news ───────────────────────────────────────────────────
    'expreso.com.pe': {
        'scope': 'national',
        'source_type': 'national_news',
        # Expreso — national daily, Lima
    },
    # ── Brazil — regional news ────────────────────────────────────────────────
    'diariodepernambuco.com.br': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # Diário de Pernambuco — oldest newspaper in the Americas (est. 1825), Pernambuco state
    },
    'em.com.br': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # Estado de Minas — leading Minas Gerais state newspaper
    },
    'folhams.com.br': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # Folha MS — Mato Grosso do Sul state newspaper
    },
    'jornaldaparaiba.com.br': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # Jornal da Paraíba — Paraíba state newspaper
    },
    'agazeta.com.br': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # A Gazeta — Espírito Santo state newspaper
    },
    'ibahia.com': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # iBahia — Bahia state news portal (Grupo A Tarde)
    },
    'diariodonordeste.verdesmares.com.br': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # Diário do Nordeste — major Ceará / northeast Brazil newspaper
    },
    'folhape.com.br': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # Folha de Pernambuco — Pernambuco state newspaper
    },
    'pernambuco.com': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # Pernambuco.com — Pernambuco state news portal
    },
    'opovo.com.br': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # O Povo — leading Ceará state newspaper (est. 1928)
    },
    'hojeemdia.com.br': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # Hoje em Dia — Minas Gerais regional newspaper
    },
    'gp1.com.br': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # GP1 — Piauí state news portal
    },
    'gazetaweb.com': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # Gazeta Web — Alagoas state news portal
    },
    'seculodiario.com.br': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # Século Diário — Espírito Santo regional news portal
    },
    # ── Colombia — regional news ───────────────────────────────────────────────
    'elcolombiano.com': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # El Colombiano — Antioquia/Medellín regional newspaper (large regional)
    },
    # ── Peru — regional news ───────────────────────────────────────────────────
    'elpueblo.com.pe': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # El Pueblo — Arequipa regional newspaper
    },
    # ── Ecuador — radio ───────────────────────────────────────────────────────
    'radiohuancavilca.com.ec': {
        'scope': 'regional',
        'source_type': 'radio',
        # Radio Huancavilca — Guayaquil regional radio station
    },
    # ── Brazil — local news ───────────────────────────────────────────────────
    'bhaz.com.br': {
        'scope': 'local',
        'source_type': 'local_news',
        # BHaz — Belo Horizonte city lifestyle/news portal
    },
    'campograndenews.com.br': {
        'scope': 'local',
        'source_type': 'local_news',
        # Campo Grande News — Campo Grande city news
    },
    'faroldabahia.com.br': {
        'scope': 'local',
        'source_type': 'local_news',
        # Farol da Bahia — Salvador/Bahia local news portal
    },
    'a12.com': {
        'scope': 'local',
        'source_type': 'local_news',
        # A12 — Catholic news portal, São Paulo
    },
    'jornalinterior.com.br': {
        'scope': 'local',
        'source_type': 'local_news',
        # Jornal Interior — São Paulo interior region local newspaper
    },
    # ── Bolivia — local news ───────────────────────────────────────────────────
    'lavozdetarija.com': {
        'scope': 'local',
        'source_type': 'local_news',
        # La Voz de Tarija — Tarija city/department local newspaper
    },
    # ── remaining domains (NaN-country rows in dataset) ───────────────────────
    'folhavitoria.com.br': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # Folha Vitória — Espírito Santo state news portal
    },
    'poder360.com.br': {
        'scope': 'national',
        'source_type': 'national_news',
        # Poder360 — major Brazilian national political journalism portal
    },
    'folhadematogrosso.com.br': {
        'scope': 'regional',
        'source_type': 'regional_news',
        # Folha de Mato Grosso — Mato Grosso state newspaper
    },
    'sudaca.pe': {
        'scope': 'national',
        'source_type': 'national_news',
        # Sudaca — Peruvian investigative/social journalism outlet
    },
    'brasil.elpais.com': {
        'scope': 'national',
        'source_type': 'national_news',
        # El País Brasil — Brazilian edition of Spanish international newspaper
    },
}

# heuristic fallbacks for domains not in the lookup
_GOV_RE  = re.compile(r'\.(gov|gob)(\.[a-z]{2})?$', re.IGNORECASE)
_NGO_RE  = re.compile(r'\.org(\.[a-z]{2})?$', re.IGNORECASE)


def classify_source(domain: str) -> dict[str, str]:
    """
    Return scope and source_type for a domain.
    Uses exact lookup for known dataset domains, heuristics for unknowns.
    """
    domain = domain.lower().strip()
    # strip www / www2 / www20 etc. prefixes
    domain = re.sub(r'^www\d*\.', '', domain)

    if domain in _DOMAIN_LOOKUP:
        return _DOMAIN_LOOKUP[domain]

    # heuristic fallbacks
    if _GOV_RE.search(domain):
        return {'scope': 'government', 'source_type': 'government_agency'}
    if _NGO_RE.search(domain):
        return {'scope': 'ngo', 'source_type': 'ngo'}

    return {'scope': 'local', 'source_type': 'unknown'}


def run_authority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds scope and source_type columns to the dataframe.
    Extracts domain from url if a domain column is not already present.
    """
    logger.info('classifying source authority...')

    df = df.copy()

    if 'domain' not in df.columns:
        if 'url' not in df.columns:
            logger.warning('neither domain nor url column found — authority skipped')
            df['scope'] = 'unknown'
            df['source_type'] = 'unknown'
            return df
        df['domain'] = df['url'].str.extract(r'https?://(?:www\.)?([^/]+)/')

    authority_rows = df['domain'].apply(
        lambda d: classify_source(str(d) if pd.notna(d) else '')
    ).apply(pd.Series)

    df['scope']       = authority_rows['scope']
    df['source_type'] = authority_rows['source_type']

    scope_dist  = df['scope'].value_counts().to_dict()
    type_dist   = df['source_type'].value_counts().to_dict()
    logger.info(f'scope distribution: {scope_dist}')
    logger.info(f'source_type distribution: {type_dist}')
    logger.info('authority classification complete')
    return df
