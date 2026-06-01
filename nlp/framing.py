# nlp/framing.py
# rule-based frame classifier for flood news articles
# theoretical grounding:
#   Entman (1993): framing as selection and salience — what is defined as the problem,
#                  who caused it, what should be done, and what is the moral judgment
#   Zade et al. (2018): actionability bias — response and recovery frames carry
#                       more operational utility than impact or accountability frames
#   Khawaja et al. (2025): North America / South divergence in framing of same events
#
# Four primary frames (Entman 1993 + disaster journalism literature):
#   impact:         describes what the flood did (casualties, damage, displacement)
#   response:       describes what is being / should be done (rescue, evacuation, aid)
#   accountability: who is responsible (government failure, policy, warnings missed)
#   recovery:       what happens next (reconstruction, resilience, long-term aid)
#
# Input: output_actionability df — each row is one article with a list_of_sentences column
# Output: same df with one added column — dominant_frame

import re
import importlib
import logging

import pandas as pd

config = importlib.import_module('config.nlp_config')
logger = logging.getLogger(__name__)


# ── Frame keyword lexicons ─────────────────────────────────────────────────────

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
            'muertos', 'muerto', 'muerta', 'muertas',
            'fallecidos', 'fallecido', 'fallecida', 'fallecidas', 'fallecimiento',
            'víctimas', 'víctima',
            'heridos', 'herido', 'herida', 'heridas',
            'desaparecidos', 'desaparecido', 'desaparecida', 'desaparecidas',
            'desplazados', 'desplazado', 'desplazada', 'desplazadas',
            'damnificados', 'damnificado', 'damnificada',
            'daños', 'daño',
            'destruido', 'destruidos', 'destruida', 'destruidas',
            'derrumbado', 'derrumbados', 'derrumbada', 'derrumbadas',
            'inundado', 'inundados', 'inundada', 'inundadas',
            'anegado', 'anegados', 'anegada', 'anegadas',
            'arrastrado', 'arrastrados', 'arrastrada', 'arrastradas',
            'deslizamiento', 'deslizamientos',
            'desbordamiento', 'desbordamientos',
            'aislado', 'aislados', 'aislada', 'aisladas',
            'atrapado', 'atrapados', 'atrapada', 'atrapadas',
            'colapso', 'derrumbe', 'derrumbes',
        ],
        'response': [
            'rescate', 'rescates',
            'evacuación', 'evacuaciones', 'evacuado', 'evacuados', 'evacuada', 'evacuadas',
            'ayuda', 'socorro', 'emergencia', 'emergencias',
            'albergue', 'albergues', 'refugio', 'refugios',
            'bomberos', 'militares', 'voluntarios', 'voluntario', 'voluntaria',
            'helicóptero', 'helicópteros', 'botes', 'bote',
            'búsqueda', 'operativo', 'operativos',
            'distribución', 'asistencia', 'equipos', 'equipo',
            'brigadas', 'brigada', 'despliegue',
        ],
        'accountability': [
            'gobierno', 'gobiernos', 'funcionarios', 'funcionario',
            'alcalde', 'alcaldes', 'alcaldesa',
            'gobernador', 'gobernadores', 'presidente', 'presidentes',
            'ministerio', 'ministerios', 'ministro', 'ministra',
            'alerta', 'alertas',
            'fallo', 'fallos', 'falló', 'fallaron',
            'negligencia', 'críticas', 'crítica',
            'culpa', 'culpable', 'culpables',
            'responsabilidad', 'responsable', 'responsables',
            'política', 'políticas',
            'infraestructura', 'corrupción',
            'inacción', 'incumplimiento', 'investigación', 'investigaciones',
        ],
        'recovery': [
            'reconstrucción', 'reconstruir', 'reconstruyendo', 'reconstruido',
            'recuperación', 'recuperar', 'recuperando',
            'resiliencia', 'fondos', 'fondo',
            'ayuda', 'seguro', 'seguros',
            'compensación', 'compensaciones',
            'restauración', 'mitigación', 'adaptación',
            'prevención', 'planificación',
            'inversión', 'inversiones',
            'programa', 'programas', 'largo plazo',
        ],
    },
    'pt': {
        'impact': [
            'mortos', 'mortes', 'morte', 'vítimas', 'vítima',
            'feridos', 'ferido', 'ferida', 'feridas',
            'desaparecidos', 'desaparecido', 'desaparecida', 'desaparecidas',
            'desalojados', 'desalojado', 'desalojada', 'desalojadas',
            'desabrigados', 'desabrigado', 'desabrigada', 'desabrigadas',
            'danos', 'dano',
            'destruído', 'destruídos', 'destruída', 'destruídas',
            'desabamento', 'desabamentos',
            'inundado', 'inundados', 'inundada', 'inundadas',
            'alagado', 'alagados', 'alagada', 'alagadas',
            'arrastado', 'arrastados', 'arrastada', 'arrastadas',
            'deslizamento', 'deslizamentos',
            'transbordamento', 'transbordamentos',
            'isolado', 'isolados', 'isolada', 'isoladas',
            'preso', 'presos', 'presa', 'presas',
            'colapso',
        ],
        'response': [
            'resgate', 'resgates',
            'evacuação', 'evacuações', 'evacuado', 'evacuados', 'evacuada', 'evacuadas',
            'ajuda', 'socorro', 'emergência', 'emergências',
            'abrigo', 'abrigos',
            'bombeiros', 'militares', 'voluntários', 'voluntário', 'voluntária',
            'helicóptero', 'helicópteros', 'barcos', 'barco',
            'busca', 'operação', 'operações',
            'distribuição', 'assistência', 'equipes', 'equipe',
            'brigadas', 'brigada', 'deslocamento',
        ],
        'accountability': [
            'governo', 'governos', 'funcionários', 'funcionário',
            'prefeito', 'prefeitos', 'prefeita', 'prefeitas',
            'governador', 'governadores', 'presidente', 'presidentes',
            'ministério', 'ministérios', 'ministro', 'ministra',
            'alerta', 'alertas',
            'falha', 'falhas', 'falhou', 'falharam',
            'negligência', 'críticas', 'crítica',
            'culpa', 'culpado', 'culpados',
            'responsabilidade', 'responsável', 'responsáveis',
            'política', 'políticas',
            'infraestrutura', 'corrupção',
            'omissão', 'descumprimento', 'investigação', 'investigações',
        ],
        'recovery': [
            'reconstrução', 'reconstruir', 'reconstruindo', 'reconstruído',
            'recuperação', 'recuperar', 'recuperando',
            'resiliência', 'fundos', 'fundo',
            'auxílio', 'auxílios', 'ajuda',
            'seguro', 'seguros', 'compensação', 'compensações',
            'restauração', 'mitigação', 'adaptação',
            'prevenção', 'planejamento',
            'investimento', 'investimentos',
            'programa', 'programas', 'longo prazo',
        ],
    },
}

# tie-break order: response > impact > accountability > recovery
_TIEBREAK = ['response', 'impact', 'accountability', 'recovery']

# pre-compile all patterns at import time
_COMPILED: dict[str, dict[str, list[re.Pattern]]] = {}
for _lang, _frames in _FRAME_KEYWORDS.items():
    _COMPILED[_lang] = {}
    for _frame, _kws in _frames.items():
        _COMPILED[_lang][_frame] = [
            re.compile((r'' if ' ' in kw else r'\b') + re.escape(kw) + (r'' if ' ' in kw else r'\b'),
                       re.IGNORECASE)
            for kw in _kws
        ]


def _dominant_frame(text: str, lang: str) -> str:
    """Score all four frames against text, return the dominant frame label."""
    lang_patterns = _COMPILED.get(lang, _COMPILED.get('en', {}))
    text_lower = text.lower()

    scores = {
        frame: sum(1 for pat in patterns if pat.search(text_lower))
        for frame, patterns in lang_patterns.items()
    }

    if not scores or max(scores.values()) == 0:
        return 'impact'  # default when no keywords match

    max_score = max(scores.values())
    for candidate in _TIEBREAK:
        if scores.get(candidate, 0) == max_score:
            return candidate

    return 'impact'


def run_framing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a dominant_frame column to the dataframe (one label per article).

    Accepts output_actionability format: each row is one article with a
    list_of_sentences column (list[str]). Falls back to clean_text if
    list_of_sentences is absent.

    Dominant frame is one of: impact | response | accountability | recovery
    """
    logger.info('classifying article frames...')

    # resolve text source — join sentence list if available
    def _get_text(row) -> str:
        sentences = row.get('list_of_sentences')
        if isinstance(sentences, list) and sentences:
            return ' '.join(str(s) for s in sentences)
        # fallback to clean_text
        return str(row.get('clean_text', ''))

    lang_col = 'language' if 'language' in df.columns else None

    df = df.copy()
    df['dominant_frame'] = df.apply(
        lambda r: _dominant_frame(
            _get_text(r),
            str(r[lang_col]) if lang_col else 'en',
        ),
        axis=1,
    )

    dist = df['dominant_frame'].value_counts().to_dict()
    logger.info(f'frame distribution: {dist}')
    logger.info('framing classification complete')
    return df
