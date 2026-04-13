# nlp/framing.py
# rule-based frame classifier for flood news articles
# theoretical grounding:
#   Entman (1993): framing as selection and salience — what is defined as the problem,
#                  who caused it, what should be done, and what is the moral judgment
#   Zade et al. (2018): actionability bias — response and recovery frames carry
#                       more operational utility than impact or accountability frames
#   Khawaja et al. (2025): Global North / South divergence in framing of same events
#
# Four primary frames (Entman 1993 + disaster journalism literature):
#   impact:         describes what the flood did (casualties, damage, displacement)
#   response:       describes what is being / should be done (rescue, evacuation, aid)
#   accountability: who is responsible (government failure, policy, warnings missed)
#   recovery:       what happens next (reconstruction, resilience, long-term aid)
#
# Each article receives a dominant frame + sub-scores for all four.
# Bilingual: EN + ES + PT keyword sets.

import re
import importlib
import logging

import pandas as pd

config = importlib.import_module('config.nlp_config')
logger = logging.getLogger(__name__)


# ── Frame keyword lexicons ─────────────────────────────────────────────────────
# Keyword → frame mapping, bilingual (EN / ES / PT).
# Each keyword triggers +1 to the corresponding frame score.
# Designed to be exhaustive for flood journalism; extend from corpus review.

_FRAME_KEYWORDS: dict[str, dict[str, list[str]]] = {
    'en': {
        'impact': [
            'dead', 'died', 'deaths', 'killed', 'casualties', 'missing', 'injured',
            'displaced', 'homeless', 'victims', 'damage', 'destroyed', 'collapsed',
            'submerged', 'flooded', 'inundated', 'swept away', 'landslide', 'mudslide',
            'debris', 'washed out', 'cut off', 'stranded', 'trapped',
        ],
        'response': [
            'rescue', 'evacuate', 'evacuation', 'relief', 'aid', 'shelter', 'deployed',
            'emergency', 'response', 'crews', 'team', 'firefighters', 'military',
            'helicopters', 'boats', 'volunteers', 'search', 'operation', 'effort',
            'distributed', 'supplied', 'food', 'water', 'medicine',
        ],
        'accountability': [
            'government', 'official', 'mayor', 'governor', 'president', 'minister',
            'agency', 'warning', 'failed', 'failure', 'negligence', 'criticism',
            'blamed', 'responsibility', 'policy', 'infrastructure', 'corruption',
            'delayed', 'inadequate', 'unprepared', 'oversight', 'investigation',
        ],
        'recovery': [
            'rebuild', 'reconstruction', 'recovery', 'resilience', 'long-term',
            'fund', 'grant', 'insurance', 'compensation', 'restoration', 'mitigation',
            'adaptation', 'future', 'prevention', 'planning', 'investment', 'program',
        ],
    },
    'es': {
        'impact': [
            'muertos', 'fallecidos', 'víctimas', 'heridos', 'desaparecidos',
            'desplazados', 'damnificados', 'daños', 'destruido', 'derrumbado',
            'inundado', 'anegado', 'arrastrado', 'deslizamiento', 'desbordamiento',
            'aislado', 'atrapado', 'colapso', 'derrumbe',
        ],
        'response': [
            'rescate', 'evacuación', 'ayuda', 'socorro', 'emergencia', 'albergue',
            'refugio', 'bomberos', 'militares', 'voluntarios', 'helicóptero',
            'botes', 'búsqueda', 'operativo', 'distribución', 'asistencia',
            'equipos', 'brigadas', 'despliegue',
        ],
        'accountability': [
            'gobierno', 'funcionarios', 'alcalde', 'gobernador', 'presidente',
            'ministerio', 'alerta', 'fallo', 'negligencia', 'críticas', 'culpa',
            'responsabilidad', 'política', 'infraestructura', 'corrupción',
            'inacción', 'incumplimiento', 'investigación',
        ],
        'recovery': [
            'reconstrucción', 'recuperación', 'resiliencia', 'fondos', 'ayuda',
            'seguro', 'compensación', 'restauración', 'mitigación', 'adaptación',
            'prevención', 'planificación', 'inversión', 'programa', 'largo plazo',
        ],
    },
    'pt': {
        'impact': [
            'mortos', 'mortes', 'vítimas', 'feridos', 'desaparecidos',
            'desalojados', 'desabrigados', 'danos', 'destruído', 'desabamento',
            'inundado', 'alagado', 'arrastado', 'deslizamento', 'transbordamento',
            'isolado', 'preso', 'colapso',
        ],
        'response': [
            'resgate', 'evacuação', 'ajuda', 'socorro', 'emergência', 'abrigo',
            'bombeiros', 'militares', 'voluntários', 'helicóptero', 'barcos',
            'busca', 'operação', 'distribuição', 'assistência', 'equipes',
            'brigadas', 'deslocamento',
        ],
        'accountability': [
            'governo', 'funcionários', 'prefeito', 'governador', 'presidente',
            'ministério', 'alerta', 'falha', 'negligência', 'críticas', 'culpa',
            'responsabilidade', 'política', 'infraestrutura', 'corrupção',
            'omissão', 'descumprimento', 'investigação',
        ],
        'recovery': [
            'reconstrução', 'recuperação', 'resiliência', 'fundos', 'auxílio',
            'seguro', 'compensação', 'restauração', 'mitigação', 'adaptação',
            'prevenção', 'planejamento', 'investimento', 'programa', 'longo prazo',
        ],
    },
}


def _build_pattern(kw: str) -> re.Pattern:
    """Compile a word-boundary pattern for a keyword or phrase."""
    escaped = re.escape(kw)
    # For phrases, don't add \b around spaces
    if ' ' in kw:
        return re.compile(escaped, re.IGNORECASE)
    return re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)


# Pre-compile all patterns at import time for performance
_COMPILED_FRAMES: dict[str, dict[str, list[tuple[str, re.Pattern]]]] = {}
for _lang, _frames in _FRAME_KEYWORDS.items():
    _COMPILED_FRAMES[_lang] = {}
    for _frame, _kws in _frames.items():
        _COMPILED_FRAMES[_lang][_frame] = [(kw, _build_pattern(kw)) for kw in _kws]


def score_frames(text: str, lang: str) -> dict:
    """
    Score all four frames for one article using keyword matching.
    Returns dict with:
        frame_{name}_score: int hit count per frame
        dominant_frame:     name of highest-scoring frame (tie → impact)
        frame_diversity:    number of frames with score > 0 (0–4)
    """
    # Fall back to English if language not in lexicon
    lang_frames = _COMPILED_FRAMES.get(lang, _COMPILED_FRAMES.get('en', {}))
    text_lower  = text.lower()

    scores: dict[str, int] = {}
    for frame_name, kw_patterns in lang_frames.items():
        scores[frame_name] = sum(
            1 for _, pat in kw_patterns if pat.search(text_lower)
        )

    dominant = max(scores, key=lambda k: scores[k]) if scores else 'impact'
    # Tie-break: prefer response > impact > accountability > recovery
    if scores:
        max_score = max(scores.values())
        tiebreak_order = ['response', 'impact', 'accountability', 'recovery']
        for candidate in tiebreak_order:
            if scores.get(candidate, 0) == max_score:
                dominant = candidate
                break

    diversity = sum(1 for s in scores.values() if s > 0)

    return {
        'frame_impact_score':         scores.get('impact', 0),
        'frame_response_score':       scores.get('response', 0),
        'frame_accountability_score': scores.get('accountability', 0),
        'frame_recovery_score':       scores.get('recovery', 0),
        'dominant_frame':             dominant,
        'frame_diversity':            diversity,
    }


def run_framing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add framing columns to the dataframe.
    Expects 'clean_text' and 'language' columns (from preprocessing).
    """
    logger.info('classifying article frames...')

    frame_scores = df.apply(
        lambda r: score_frames(str(r.get('clean_text', '')), str(r.get('language', 'en'))),
        axis=1,
        result_type='expand',
    )
    df = pd.concat([df, frame_scores], axis=1)

    # Summary logging
    if 'dominant_frame' in df.columns:
        dist = df['dominant_frame'].value_counts().to_dict()
        logger.info(f'frame distribution: {dist}')

    logger.info('framing classification complete')
    return df
