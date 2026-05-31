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
        'splines':  'polyline',
        'nodesep':  '0.5',
        'ranksep':  '0.5',
        'bgcolor':  '#F2F4F7',
        'fontname': 'Arial',
        'pad':      '0.6',
        'dpi':      '180',
    },
    node_attr={
        'shape':    'plaintext',
        'fontname': 'Arial',
        'margin':   '0',
    },
    edge_attr={
        'color':     '#2C3E50',
        'penwidth':  '2.2',
        'arrowsize': '1.0',
    },
)


def section(label, label_col, bg, lines):
    """One coloured section block: header row + one row per bullet."""
    bullet_rows = ''.join(
        f'<TR><TD ALIGN="LEFT" CELLPADDING="3">'
        f'<FONT POINT-SIZE="10.5" COLOR="#1A252F">&#x25AA; {l}</FONT>'
        f'</TD></TR>'
        for l in lines
    )
    return (
        f'<TR><TD BGCOLOR="{bg}" CELLPADDING="5" CELLSPACING="0">'
        f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">'
        f'<TR><TD ALIGN="LEFT" CELLPADDING="2">'
        f'<B><FONT POINT-SIZE="9" COLOR="{label_col}">&#9654; {label}</FONT></B>'
        f'</TD></TR>'
        f'{bullet_rows}'
        f'</TABLE>'
        f'</TD></TR>'
    )


def step_node(step_name, filename, input_lines, process_lines, output_lines, banner_col):
    filename_row = (
        f'<TR><TD BGCOLOR="{banner_col}" CELLPADDING="2" ALIGN="CENTER">'
        f'<FONT POINT-SIZE="9.5" COLOR="#A8DADC" FACE="Courier New"><I>{filename}</I></FONT>'
        f'</TD></TR>'
    )
    spacer = '<TR><TD HEIGHT="3" BGCOLOR="white"></TD></TR>'
    return (
        f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" '
        f'BGCOLOR="white" COLOR="#BDC3C7" STYLE="ROUNDED">'

        # banner
        f'<TR><TD BGCOLOR="{banner_col}" CELLPADDING="10" ALIGN="CENTER">'
        f'<B><FONT POINT-SIZE="13.5" COLOR="white">{step_name}</FONT></B>'
        f'</TD></TR>'

        # filename
        f'{filename_row}'

        # sections
        f'{spacer}'
        f'{section("INPUT",   "#1A5276", "#D6EAF8", input_lines)}'
        f'{spacer}'
        f'{section("PROCESS", "#7D6608", "#FEF9E7", process_lines)}'
        f'{spacer}'
        f'{section("OUTPUT",  "#1E8449", "#E9F7EF", output_lines)}'
        f'{spacer}'

        f'</TABLE>>'
    )


def io_node(label, sublabel, bg, border):
    return (
        f'<<TABLE BORDER="2" CELLBORDER="0" CELLSPACING="0" CELLPADDING="10" '
        f'BGCOLOR="{bg}" COLOR="{border}" STYLE="ROUNDED">'
        f'<TR><TD ALIGN="CENTER">'
        f'<B><FONT POINT-SIZE="14" COLOR="#1A252F">{label}</FONT></B>'
        f'</TD></TR>'
        f'<TR><TD ALIGN="CENTER">'
        f'<FONT POINT-SIZE="10" COLOR="#566573"><I>{sublabel}</I></FONT>'
        f'</TD></TR>'
        f'</TABLE>>'
    )


def output_node(rows):
    file_rows = ''.join(
        f'<TR>'
        f'<TD ALIGN="LEFT" CELLPADDING="6" BGCOLOR="#D4E6F1">'
        f'<B><FONT POINT-SIZE="9.5" COLOR="#1A5276" FACE="Courier New">{fname}</FONT></B>'
        f'</TD>'
        f'<TD ALIGN="LEFT" CELLPADDING="6">'
        f'<FONT POINT-SIZE="10.5" COLOR="#1A252F">{desc}</FONT>'
        f'</TD>'
        f'</TR>'
        for fname, desc in rows
    )
    return (
        f'<<TABLE BORDER="2" CELLBORDER="0" CELLSPACING="3" CELLPADDING="0" '
        f'BGCOLOR="#EBF5FB" COLOR="#1A5276" STYLE="ROUNDED">'
        f'<TR><TD COLSPAN="2" BGCOLOR="#1F618D" CELLPADDING="10" ALIGN="CENTER">'
        f'<B><FONT POINT-SIZE="14" COLOR="white">OUTPUTS</FONT></B>'
        f'</TD></TR>'
        f'{file_rows}'
        f'</TABLE>>'
    )


# ── nodes ─────────────────────────────────────────────────────────────────────

dot.node('input', io_node(
    'INPUT CSV',
    'verified_articles_clean.csv   |   612 rows   |   EN, ES, PT   |   11 flood events',
    '#D6EAF8', '#1F618D',
))

dot.node('step1', step_node(
    'STEP 1: PREPROCESSING', 'preprocessing.py',
    ['612-row CSV with ISO 639-2 language codes and pre-cleaned article text'],
    ['ISO 639-2 → ISO 639-1 language mapping  (spa/por/eng → es/pt/en)',
     'HTML stripping, zero-width character removal, whitespace normalisation',
     'Minimum 100-character length filter',
     'SHA-256 deduplication per flood event  (removes duplicate URL crawls)'],
    ['580 articles retained  |  clean_text and language columns verified'],
    '#1F618D',
))

dot.node('step2', step_node(
    'STEP 2: ACTIONABILITY SCORING', 'actionability.py',
    ['Cleaned df (580 rows) from Step 1'],
    ['Sentence segmentation via spaCy  (en_core_web_sm / es_core_news_sm / pt_core_news_sm)',
     'Morphological POS tagging: imperative and subjunctive verb detection per sentence',
     'Trilingual keyword counting: imperative · short-term urgency · long-term recovery · spatial anchors',
     'Semantic Role Labelling: agent + action + location co-presence per sentence',
     'Advice-framing verb flag: recommends / urges / suggests  (distinct from direct imperatives)',
     'Weighted density score → min-max normalised actionability_probability [0, 1]'],
    ['sentences_actionability.csv  |  article means: mean_imperative_count, mean_advice, actionability_percentage'],
    '#6C3483',
))

dot.node('step3', step_node(
    'STEP 3: SOURCE AUTHORITY CLASSIFICATION', 'authority.py',
    ['Enriched df from Step 2  +  article domain / URL'],
    ['Domain matched against verified 34-domain lookup table built from dataset',
     'Fallback heuristics: .gov/.gob → government_agency,   .org → ngo,   other → unknown'],
    ['scope: government / national / regional / local / ngo  per article',
     'source_type: government_agency / national_news / regional_news / local_news / unknown'],
    '#1E8449',
))

dot.node('step4', step_node(
    'STEP 4: FRAME CLASSIFICATION', 'framing.py',
    ['Enriched df from Step 3'],
    ['Trilingual keyword lexicons (EN/ES/PT) count frame-relevant matches across all sentences',
     'Four frames: impact · response · accountability · recovery  (Entman 1993)',
     'Dominant frame = highest keyword count  (impact assigned on ties)'],
    ['dominant_frame column added per article'],
    '#BA4A00',
))

dot.node('step5', step_node(
    'STEP 5: CLUSTERING', 'clustering.py',
    ['Fully enriched df from Step 4  (actionability + authority + frame features)'],
    ['Global North / South region assignment per article  (country lookup)',
     'Group statistics: actionability distributions by region, country, domain, language',
     'Z-score normalisation of 6 structural features  (imperative, short-term, long-term, spatial, advice, SRL)',
     'K-Means at k = 3, 4, 5 on structural and full feature sets',
     'Silhouette scoring per configuration  →  k = 4 selected  (score: 0.332)'],
    ['data_cluster_id per article  |  cluster_summary_structural_k4.csv  |  group_stats_*.csv'],
    '#117A65',
))

dot.node('output', output_node([
    ('enriched.csv',                      '580 rows x 33 columns — all pipeline features per article'),
    ('cluster_summary_structural_k4.csv', 'Cluster profiles: mean feature scores + top country / language / source type'),
    ('group_stats_*.csv',                 'Actionability distributions by region, country, domain, and language'),
    ('interpretations.md',                'Written interpretation of cluster structure and cross-cutting findings'),
]))

# ── edges ─────────────────────────────────────────────────────────────────────
for a, b in [('input','step1'), ('step1','step2'), ('step2','step3'),
             ('step3','step4'), ('step4','step5'), ('step5','output')]:
    dot.edge(a, b)

# ── render ────────────────────────────────────────────────────────────────────
os.makedirs('output', exist_ok=True)
out = dot.render(filename='output/pipeline_diagram', cleanup=True)
print(f'saved: {out}')
