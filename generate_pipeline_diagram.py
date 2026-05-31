"""
Generate a clean pipeline data flow diagram using Graphviz.
Output: output/pipeline_diagram.png
"""

import os
import graphviz

os.environ['PATH'] += r';C:\Program Files (x86)\Graphviz\bin'

dot = graphviz.Digraph(
    'pipeline',
    format='png',
    graph_attr={
        'rankdir':  'TB',
        'splines':  'ortho',
        'nodesep':  '0.4',
        'ranksep':  '0.45',
        'bgcolor':  '#F2F4F7',
        'fontname': 'Arial',
        'pad':      '0.5',
        'dpi':      '160',
    },
    node_attr={
        'shape':    'plaintext',
        'fontname': 'Arial',
        'margin':   '0',
    },
    edge_attr={
        'color':     '#2C3E50',
        'penwidth':  '2.2',
        'arrowsize': '0.9',
    },
)


def col(label, label_col, bg, lines, min_width=200):
    """One horizontal column: label header + bullet lines."""
    bullets = ''.join(
        f'<TR><TD ALIGN="LEFT" CELLPADDING="2">'
        f'<FONT POINT-SIZE="9.5" COLOR="#1A252F">- {l}</FONT>'
        f'</TD></TR>'
        for l in lines
    )
    return (
        f'<TD BGCOLOR="{bg}" CELLPADDING="8" VALIGN="TOP" ALIGN="LEFT">'
        f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="1">'
        f'<TR><TD ALIGN="LEFT" CELLPADDING="3">'
        f'<B><FONT POINT-SIZE="9" COLOR="{label_col}">{label}</FONT></B>'
        f'</TD></TR>'
        f'{bullets}'
        f'</TABLE>'
        f'</TD>'
    )


def divider():
    return '<TD WIDTH="1" BGCOLOR="#BDC3C7" CELLPADDING="0"></TD>'


def step_node(step_name, filename, input_lines, process_lines, output_lines, banner_col):
    return (
        f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" '
        f'BGCOLOR="white" COLOR="#9EAEC0">'

        # banner
        f'<TR><TD COLSPAN="5" BGCOLOR="{banner_col}" CELLPADDING="9" ALIGN="CENTER">'
        f'<B><FONT POINT-SIZE="13" COLOR="white">{step_name}</FONT></B>'
        f'  '
        f'<FONT POINT-SIZE="9" COLOR="#C0E0DC" FACE="Courier New"><I>({filename})</I></FONT>'
        f'</TD></TR>'

        # thin separator
        f'<TR><TD COLSPAN="5" HEIGHT="1" BGCOLOR="#BDC3C7"></TD></TR>'

        # three columns
        f'<TR>'
        f'{col("INPUT",   "#1A5276", "#D6EAF8", input_lines)}'
        f'{divider()}'
        f'{col("PROCESS", "#7D6608", "#FEF9E7", process_lines)}'
        f'{divider()}'
        f'{col("OUTPUT",  "#1E8449", "#E9F7EF", output_lines)}'
        f'</TR>'

        f'</TABLE>>'
    )


def io_node(label, sublabel, bg, border):
    return (
        f'<<TABLE BORDER="2" CELLBORDER="0" CELLSPACING="0" CELLPADDING="10" '
        f'BGCOLOR="{bg}" COLOR="{border}">'
        f'<TR><TD ALIGN="CENTER">'
        f'<B><FONT POINT-SIZE="13" COLOR="#1A252F">{label}</FONT></B>'
        f'</TD></TR>'
        f'<TR><TD ALIGN="CENTER">'
        f'<FONT POINT-SIZE="9.5" COLOR="#566573"><I>{sublabel}</I></FONT>'
        f'</TD></TR>'
        f'</TABLE>>'
    )


def output_node(rows):
    file_rows = ''.join(
        f'<TR>'
        f'<TD ALIGN="LEFT" CELLPADDING="5" BGCOLOR="#D4E6F1">'
        f'<B><FONT POINT-SIZE="9" COLOR="#1A5276" FACE="Courier New">{fname}</FONT></B>'
        f'</TD>'
        f'<TD ALIGN="LEFT" CELLPADDING="5">'
        f'<FONT POINT-SIZE="10" COLOR="#1A252F">{desc}</FONT>'
        f'</TD>'
        f'</TR>'
        for fname, desc in rows
    )
    return (
        f'<<TABLE BORDER="2" CELLBORDER="0" CELLSPACING="2" CELLPADDING="0" '
        f'BGCOLOR="#EBF5FB" COLOR="#1A5276">'
        f'<TR><TD COLSPAN="2" BGCOLOR="#1F618D" CELLPADDING="9" ALIGN="CENTER">'
        f'<B><FONT POINT-SIZE="13" COLOR="white">OUTPUTS</FONT></B>'
        f'</TD></TR>'
        f'{file_rows}'
        f'</TABLE>>'
    )


# ── nodes ─────────────────────────────────────────────────────────────────────

dot.node('input', io_node(
    'INPUT CSV',
    'verified_articles_clean.csv  |  612 rows  |  EN, ES, PT  |  11 flood events',
    '#D6EAF8', '#1F618D',
))

dot.node('step1', step_node(
    'STEP 1: PREPROCESSING', 'preprocessing.py',
    ['612-row CSV with ISO 639-2 codes',
     'and pre-cleaned article text'],
    ['ISO 639-2 to 639-1 language mapping (spa/por/eng)',
     'HTML stripping and whitespace normalisation',
     'Minimum 100-character length filter',
     'SHA-256 deduplication per flood event'],
    ['580 articles retained',
     'clean_text and language columns verified'],
    '#1F618D',
))

dot.node('step2', step_node(
    'STEP 2: ACTIONABILITY SCORING', 'actionability.py',
    ['Cleaned df (580 rows) from Step 1'],
    ['Sentence segmentation via spaCy',
     'Morphological POS tagging: imperative and subjunctive verbs',
     'Trilingual keyword counting: imperative, short-term,',
     '   long-term urgency, spatial anchors',
     'SRL: agent + action + location co-presence per sentence',
     'Advice-framing verb flag (recommends, urges, suggests)',
     'Weighted density score normalised to [0, 1]'],
    ['sentences_actionability.csv',
     'Article means: mean_imperative_count,',
     '   mean_advice, actionability_percentage'],
    '#6C3483',
))

dot.node('step3', step_node(
    'STEP 3: SOURCE AUTHORITY CLASSIFICATION', 'authority.py',
    ['Enriched df from Step 2',
     'Article domain / URL'],
    ['Domain matched against 34-domain lookup table',
     'Fallback: .gov/.gob to government_agency,',
     '   .org to ngo, other to unknown'],
    ['scope: government / national /','   regional / local / ngo',
     'source_type per article'],
    '#1E8449',
))

dot.node('step4', step_node(
    'STEP 4: FRAME CLASSIFICATION', 'framing.py',
    ['Enriched df from Step 3'],
    ['Trilingual keyword lexicons (EN/ES/PT)',
     'Four frames: impact, response,',
     '   accountability, recovery (Entman 1993)',
     'Dominant frame = highest keyword count'],
    ['dominant_frame column per article',
     'Impact assigned on ties'],
    '#BA4A00',
))

dot.node('step5', step_node(
    'STEP 5: CLUSTERING', 'clustering.py',
    ['Fully enriched df from Step 4'],
    ['Global North / South region assignment',
     'Group stats by region, country, domain, language',
     'Z-score normalisation of 6 structural features',
     'K-Means at k = 3, 4, 5 on structural and full sets',
     'Silhouette scoring: k = 4 selected (score 0.332)'],
    ['data_cluster_id per article',
     'cluster_summary_structural_k4.csv',
     'group_stats_*.csv'],
    '#117A65',
))

dot.node('output', output_node([
    ('enriched.csv',                      '580 rows x 33 columns — all pipeline features per article'),
    ('cluster_summary_structural_k4.csv', 'Cluster profiles: mean scores, top country / language / source type'),
    ('group_stats_*.csv',                 'Actionability distributions by region, country, domain, language'),
    ('interpretations.md',                'Written interpretation of cluster structure and findings'),
]))

# ── edges ─────────────────────────────────────────────────────────────────────
for a, b in [('input','step1'), ('step1','step2'), ('step2','step3'),
             ('step3','step4'), ('step4','step5'), ('step5','output')]:
    dot.edge(a, b)

# ── render ────────────────────────────────────────────────────────────────────
os.makedirs('output', exist_ok=True)
out = dot.render(filename='output/pipeline_diagram', cleanup=True)
print(f'saved: {out}')
