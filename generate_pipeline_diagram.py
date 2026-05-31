import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')

# ── colour palette ────────────────────────────────────────────────────────────
C_INPUT   = '#D6E4F0'   # light blue
C_STEP    = '#FFFFFF'   # white boxes
C_OUTPUT  = '#D5F5E3'   # light green
C_ARROW   = '#5D6D7E'
C_BORDER  = '#2C3E50'
C_LABEL   = '#1A252F'
C_SUB     = '#566573'
C_FILE    = '#7F8C8D'

def box(ax, x, y, w, h, label, sublabel='', file='', color=C_STEP, border=C_BORDER,
        label_size=11, sublabel_size=8.5):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.08',
        facecolor=color,
        edgecolor=border,
        linewidth=1.4,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, y + (0.15 if sublabel else 0), label,
            ha='center', va='center', fontsize=label_size,
            fontweight='bold', color=C_LABEL, zorder=4)
    if sublabel:
        ax.text(x, y - 0.28, sublabel,
                ha='center', va='center', fontsize=sublabel_size,
                color=C_SUB, zorder=4, style='italic')
    if file:
        ax.text(x, y - h/2 - 0.18, file,
                ha='center', va='top', fontsize=7.5,
                color=C_FILE, zorder=4, family='monospace')

def arrow(ax, x, y_top, y_bot, label=''):
    ax.annotate('', xy=(x, y_bot + 0.05), xytext=(x, y_top - 0.05),
                arrowprops=dict(arrowstyle='->', color=C_ARROW,
                                lw=1.6, mutation_scale=14),
                zorder=2)
    if label:
        ax.text(x + 0.18, (y_top + y_bot) / 2, label,
                ha='left', va='center', fontsize=7.5,
                color=C_SUB, style='italic', zorder=4)

def side_tag(ax, x_box_right, y, text, color='#ECF0F1'):
    ax.text(x_box_right + 0.18, y, text,
            ha='left', va='center', fontsize=7.2,
            color='#626567', style='italic', zorder=4)

# ── positions (y increases upward — we'll flip visually top→bottom) ───────────
# Using y values from top (15) to bottom (1)

BOXES = [
    # (x_centre, y_centre, w, h, label, sublabel, file, color)
    (5, 14.5, 6.5, 0.85, 'INPUT CSV', 'verified_articles_clean.csv  |  612 rows', '', C_INPUT),
    (5, 12.8, 6.5, 1.1,  'STEP 1: PREPROCESSING', 'language normalisation · text cleaning · deduplication · length filtering', 'preprocessing.py', C_STEP),
    (5, 10.9, 6.5, 1.1,  'STEP 2: ACTIONABILITY SCORING', 'sentence segmentation · POS tagging · keyword counting · SRL · advice detection', 'actionability.py', C_STEP),
    (5,  9.0, 6.5, 1.1,  'STEP 3: SOURCE AUTHORITY', 'domain lookup · scope classification · source type assignment', 'authority.py', C_STEP),
    (5,  7.1, 6.5, 1.1,  'STEP 4: FRAME CLASSIFICATION', 'impact · response · accountability · recovery (Entman 1993)', 'framing.py', C_STEP),
    (5,  5.2, 6.5, 1.1,  'STEP 5: CLUSTERING', 'z-score normalisation · K-Means k=3,4,5 · silhouette selection · k=4 optimal', 'clustering.py', C_STEP),
    (5,  3.2, 6.5, 0.85, 'OUTPUT', 'enriched.csv  |  580 rows × 33 columns', '', C_OUTPUT),
]

for (x, y, w, h, label, sublabel, file, color) in BOXES:
    box(ax, x, y, w, h, label, sublabel, file, color)

# ── arrows with intermediate labels ──────────────────────────────────────────
ARROWS = [
    (14.07, 13.33, 'cleaned, language-verified df  (580 rows)'),
    (12.35, 11.45, '+ sentence-level features aggregated to article means'),
    (10.45,  9.55, '+ scope, source_type'),
    ( 8.55,  7.65, '+ dominant_frame'),
    ( 6.65,  5.75, '+ data_cluster_id, group_stats'),
    ( 4.75,  3.65, ''),
]

for (y_top, y_bot, lbl) in ARROWS:
    arrow(ax, 5, y_top, y_bot, lbl)

# ── side annotations: key outputs per step ────────────────────────────────────
TAGS = [
    (8.25, 12.8, '→ 580 rows retained'),
    (8.25, 10.9, '→ sentences_actionability.csv'),
    (8.25,  9.0, '→ scope + source_type per article'),
    (8.25,  7.1, '→ dominant_frame per article'),
    (8.25,  5.2, '→ cluster_summary_structural_k4.csv\n      → group_stats_*.csv'),
]

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(5, 15.6, 'Americas Flood NLP Pipeline — Data Flow',
        ha='center', va='center', fontsize=13, fontweight='bold',
        color=C_LABEL, zorder=4)

ax.text(5, 15.15, 'EN · ES · PT  |  11 flood events  |  scikit-learn · spaCy · pandas',
        ha='center', va='center', fontsize=8.5, color=C_FILE, zorder=4)

plt.tight_layout()
out = 'output/pipeline_diagram.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'saved: {out}')
