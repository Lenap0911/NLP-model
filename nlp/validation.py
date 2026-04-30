# nlp/validation.py
# pre-run sanity checks for the NLP pipeline
# verifies that the keyword scorer can correctly rank known-tier texts
# before processing the full dataset — catches broken keyword lists early

import logging
import importlib

config = importlib.import_module('config.nlp_config')
logger = logging.getLogger(__name__)

# Three texts per language with known actionability tiers.
# HIGH:   imperative verbs + urgency signals + spatial anchors
# MEDIUM: urgency/spatial but no direct calls to action
# LOW:    retrospective, recovery language, no calls to action
_CASES = {
    'en': {
        'high':   'Evacuate the area immediately — dangerous flooding in downtown, '
                  'contact emergency services now and move to higher ground.',
        'medium': 'Emergency shelters have been set up in the flood zone for '
                  'displaced residents across the city district.',
        'low':    'Last year the floods caused widespread damage; recovery efforts '
                  'focused on rebuilding homes in the affected region.',
    },
    'es': {
        'high':   'Evacúe inmediatamente — hay peligro de inundación en la zona, '
                  'llame a los servicios de emergencia ahora y trasládese a un lugar seguro.',
        'medium': 'Se han establecido refugios de emergencia en la zona para los '
                  'afectados por las inundaciones en el municipio.',
        'low':    'Las inundaciones del año pasado causaron daños en la región; '
                  'los esfuerzos de recuperación se centran en reconstruir las áreas afectadas.',
    },
    'pt': {
        'high':   'Evacue imediatamente — há perigo de inundação na zona, '
                  'ligue para os serviços de emergência agora e saia de casa.',
        'medium': 'Abrigos de emergência foram criados na zona para os afetados '
                  'pelas inundações no município.',
        'low':    'As inundações do ano passado causaram danos na região; '
                  'os esforços de recuperação focam-se em reconstruir as áreas afetadas.',
    },
}


def validate_scorer() -> bool:
    """
    Sanity-check the keyword scorer against known-tier texts.
    For each supported language, checks: score(high) > score(medium) > score(low).
    Also checks that the high-tier text clears a minimum absolute threshold.

    Returns True if all checks pass, False otherwise.
    Logs PASS/FAIL per language with actual scores — never raises, never halts.
    """
    from nlp.actionability import score_actionability_keywords

    all_pass = True

    for lang, cases in _CASES.items():
        if lang not in config.SUPPORTED_LANGUAGES:
            continue

        scores = {
            tier: score_actionability_keywords(text, lang)['actionability_score']
            for tier, text in cases.items()
        }

        high_ok   = scores['high']   > scores['medium']
        medium_ok = scores['medium'] > scores['low']
        abs_ok    = scores['high']   > 0.3

        passed = high_ok and medium_ok and abs_ok
        status = 'PASS' if passed else 'FAIL'

        logger.info(
            f'  [{status}] {lang.upper()} — '
            f'high={scores["high"]:.3f}  medium={scores["medium"]:.3f}  low={scores["low"]:.3f}'
        )

        if not high_ok:
            logger.warning(
                f'    {lang}: high ({scores["high"]:.3f}) <= medium ({scores["medium"]:.3f})'
                ' — check imperative_verbs / short_term keywords in config'
            )
        if not medium_ok:
            logger.warning(
                f'    {lang}: medium ({scores["medium"]:.3f}) <= low ({scores["low"]:.3f})'
                ' — check short_term / spatial_anchors keywords in config'
            )
        if not abs_ok:
            logger.warning(
                f'    {lang}: high-tier text scored only {scores["high"]:.3f}'
                ' — keyword list may be too sparse to detect actionability'
            )

        all_pass = all_pass and passed

    if all_pass:
        logger.info('  Scorer validation passed — proceeding with pipeline.')
    else:
        logger.warning(
            '  Scorer validation failed for one or more languages. '
            'Actionability results may be unreliable. '
            'Check ACTIONABILITY_KEYWORDS in config/nlp_config.py.'
        )

    return all_pass
