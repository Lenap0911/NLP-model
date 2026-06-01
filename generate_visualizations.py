# generate_visualizations.py
# produces visualizations from the enriched pipeline output
# run from the project root:  python generate_visualizations.py

import os
import sys
import importlib

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
config = importlib.import_module('config.nlp_config')

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
VIZ_DIR = os.path.join(OUT_DIR, 'visualizations')
os.makedirs(VIZ_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

LANG_LABELS = {'en': 'English', 'es': 'Spanish', 'pt': 'Portuguese'}
LANG_PALETTE = {'en': '#295574', 'es': '#e75e1e', 'pt': '#2ca062'}
PALETTE = sns.color_palette('tab10')


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(VIZ_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {path}')


# ── 1. Articles by language ───────────────────────────────────────────────────
def plot_language_distribution(df: pd.DataFrame) -> None:
    lang_order = ['en', 'es', 'pt']
    counts = df['language'].value_counts().reindex(lang_order).dropna()
    labels = [LANG_LABELS.get(l, l) for l in counts.index]
    colors = [LANG_PALETTE.get(l, '#aaaaaa') for l in counts.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, counts.values, color=colors, edgecolor='white')
    ax.set_title('Articles by language', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Number of articles')
    ax.tick_params(axis='x', rotation=0)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    _save(fig, '01_language_distribution.png')


# ── 2. Articles by country (top 12) ──────────────────────────────────────────
def plot_country_distribution(df: pd.DataFrame) -> None:
    counts = df['country'].value_counts().head(12)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot.bar(ax=ax, color=PALETTE[0], edgecolor='white', alpha=0.85)
    ax.set_title('Articles by country (top 12)', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Number of articles')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    _save(fig, '02_country_distribution.png')


# ── 3. Actionability by language ─────────────────────────────────────────────
def plot_actionability_by_language(df: pd.DataFrame) -> None:
    df = df.copy()
    df['language_label'] = df['language'].map(LANG_LABELS).fillna(df['language'])
    order = df.groupby('language_label')['actionability_percentage'].mean().sort_values(ascending=False).index
    palette = {LANG_LABELS.get(k, k): v for k, v in LANG_PALETTE.items()}

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x='language_label', y='actionability_percentage',
                order=order, palette=palette, ax=ax)
    ax.set_title('Actionability by language', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Actionability (%)')
    plt.tight_layout()
    _save(fig, '03_actionability_by_language.png')


# ── 4. Actionability by Global North / South ─────────────────────────────────
def plot_actionability_by_region(df: pd.DataFrame) -> None:
    if 'global_region' not in df.columns:
        print('global_region column not found — skipping plot 4')
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Actionability: Global North vs Global South', fontsize=12, fontweight='bold')

    # boxplot
    sns.boxplot(data=df, x='global_region', y='actionability_percentage',
                palette=['#4e79a7', '#f28e2b'], ax=axes[0])
    axes[0].set_title('Distribution')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Actionability (%)')

    # article count
    counts = df['global_region'].value_counts()
    counts.plot.bar(ax=axes[1], color=['#4e79a7', '#f28e2b'], edgecolor='white')
    axes[1].set_title('Article count')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Number of articles')
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    _save(fig, '04_actionability_by_region.png')


# ── 5. Actionability by source type ──────────────────────────────────────────
def plot_actionability_by_source(df: pd.DataFrame) -> None:
    if 'source_type' not in df.columns:
        print('source_type column not found — skipping plot 5')
        return

    order = df.groupby('source_type')['actionability_percentage'].mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x='source_type', y='actionability_percentage',
                order=order, palette='tab10', ax=ax, errorbar='sd')
    ax.set_title('Mean actionability by source type', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Actionability (%) — mean ± SD')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    _save(fig, '05_actionability_by_source_type.png')


# ── 6. Frame distribution ─────────────────────────────────────────────────────
def plot_frame_distribution(df: pd.DataFrame) -> None:
    if 'dominant_frame' not in df.columns:
        print('dominant_frame column not found — skipping plot 6')
        return

    frame_colors = {
        'impact': '#e15759', 'response': '#4e79a7',
        'accountability': '#f28e2b', 'recovery': '#59a14f',
    }
    counts = df['dominant_frame'].value_counts()
    colors = [frame_colors.get(f, '#aaaaaa') for f in counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot.bar(ax=ax, color=colors, edgecolor='white')
    ax.set_title('Dominant frame distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Number of articles')
    ax.tick_params(axis='x', rotation=0)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    _save(fig, '06_frame_distribution.png')


# ── 7. Frame × Global region (normalised to proportions) ─────────────────────
def plot_frame_by_region(df: pd.DataFrame) -> None:
    if 'dominant_frame' not in df.columns or 'global_region' not in df.columns:
        print('dominant_frame or global_region missing — skipping plot 7')
        return

    frame_region = (
        df.groupby(['global_region', 'dominant_frame'])
        .size().reset_index(name='count')
    )
    totals = frame_region.groupby('global_region')['count'].transform('sum')
    frame_region['pct'] = frame_region['count'] / totals * 100
    pivot = frame_region.pivot(index='dominant_frame', columns='global_region', values='pct').fillna(0)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot.bar(ax=ax, color=['#4e79a7', '#f28e2b'], edgecolor='white', alpha=0.88)
    ax.set_title('Frame distribution by global region (% of articles within region)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('% of articles in region')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Region')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=8, padding=2)
    plt.tight_layout()
    _save(fig, '07_frame_by_region.png')


# ── 9. PADM component presence by language ────────────────────────────────────
def plot_padm_components(df: pd.DataFrame) -> None:
    components = {
        'Imperative signals':  'mean_imperative_count',
        'Short-term urgency':  'mean_short_term_count',
        'Spatial anchors':     'mean_spatial_count',
        'Advice-framing':      'mean_advice',
    }
    missing = [c for c in components.values() if c not in df.columns]
    if missing:
        print(f'PADM columns missing: {missing} — skipping plot 9')
        return

    lang_order = ['en', 'es', 'pt']
    lang_labels = [LANG_LABELS.get(l, l) for l in lang_order]
    x = range(len(components))
    width = 0.25
    comp_labels = list(components.keys())

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, lang in enumerate(lang_order):
        sub = df[df['language'] == lang]
        pcts = []
        for col in components.values():
            if col == 'mean_imperative_count' and 'mean_verbs_imperative_count' in sub.columns:
                # combine keyword-based (all languages) + POS-based (ES/PT morphology)
                present = (sub['mean_imperative_count'] > 0) | (sub['mean_verbs_imperative_count'] > 0)
                pcts.append(present.mean() * 100)
            else:
                pcts.append((sub[col] > 0).mean() * 100)
        bars = ax.bar(
            [xi + i * width for xi in x], pcts,
            width=width, label=LANG_LABELS.get(lang, lang),
            color=LANG_PALETTE[lang], edgecolor='white', alpha=0.9
        )

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(comp_labels, fontsize=10)
    ax.set_ylabel('% of articles with component present')
    ax.set_title(
        'PADM component presence by language\n'
        '(% of articles where at least one sentence triggered each feature)',
        fontsize=12, fontweight='bold'
    )
    ax.legend(title='Language')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    _save(fig, '09_padm_components_by_language.png')


# ── 10. Frame × Actionability ─────────────────────────────────────────────────
def plot_frame_actionability(df: pd.DataFrame) -> None:
    if 'dominant_frame' not in df.columns:
        print('dominant_frame missing — skipping plot 10')
        return

    frame_order = ['impact', 'response', 'accountability', 'recovery']
    frame_colors = {
        'impact': '#e15759', 'response': '#4e79a7',
        'accountability': '#f28e2b', 'recovery': '#59a14f',
    }

    stats = (
        df.groupby('dominant_frame')['actionability_percentage']
        .agg(['mean', 'median', 'count', 'std'])
        .reindex(frame_order).reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [frame_colors.get(f, '#aaa') for f in stats['dominant_frame']]
    bars = ax.bar(stats['dominant_frame'], stats['mean'], color=colors,
                  edgecolor='white', alpha=0.9, zorder=3)
    ax.errorbar(stats['dominant_frame'], stats['mean'],
                yerr=stats['std'], fmt='none', color='#333333',
                capsize=4, linewidth=1.2, zorder=4)

    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + row['std'] + 0.3,
                f"n={int(row['count'])}", ha='center', va='bottom', fontsize=8.5)

    ax.axhline(df['actionability_percentage'].mean(), color='#555', linewidth=1.2,
               linestyle='--', label=f"Corpus mean ({df['actionability_percentage'].mean():.1f}%)")
    ax.set_title('Mean actionability by dominant frame\n(error bars = ±1 SD)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Mean actionability (%)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, '10_frame_actionability.png')


# ── 10b. Frame × Actionability by language ───────────────────────────────────
def plot_frame_actionability_by_language(df: pd.DataFrame) -> None:
    if 'dominant_frame' not in df.columns:
        print('dominant_frame missing — skipping frame by language plots')
        return

    frame_order = ['impact', 'response', 'accountability', 'recovery']
    frame_colors = {
        'impact': '#e15759', 'response': '#4e79a7',
        'accountability': '#f28e2b', 'recovery': '#59a14f',
    }
    lang_names = {'en': 'English', 'es': 'Spanish', 'pt': 'Portuguese'}

    for lang, label in lang_names.items():
        sub = df[df['language'] == lang]
        if sub.empty:
            continue

        stats = (
            sub.groupby('dominant_frame')['actionability_percentage']
            .agg(['mean', 'count', 'std'])
            .reindex(frame_order).reset_index()
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = [frame_colors.get(f, '#aaa') for f in stats['dominant_frame']]
        bars = ax.bar(stats['dominant_frame'], stats['mean'], color=colors,
                      edgecolor='white', alpha=0.9, zorder=3)
        ax.errorbar(stats['dominant_frame'], stats['mean'],
                    yerr=stats['std'], fmt='none', color='#333333',
                    capsize=4, linewidth=1.2, zorder=4)

        for bar, (_, row) in zip(bars, stats.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (row['std'] or 0) + 0.1,
                    f"n={int(row['count'])}" if not pd.isna(row['count']) else '',
                    ha='center', va='bottom', fontsize=8.5)

        ax.axhline(sub['actionability_percentage'].mean(), color='#555', linewidth=1.2,
                   linestyle='--', label=f"Language mean ({sub['actionability_percentage'].mean():.1f}%)")
        ax.set_title(f'Mean actionability by dominant frame — {label}\n(error bars = ±1 SD)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Mean actionability (%)')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        plt.tight_layout()
        _save(fig, f'10_{lang}_frame_actionability.png')


# ── 11. Source type × Region (H₁ mechanism) ──────────────────────────────────
def plot_source_region(df: pd.DataFrame) -> None:
    if 'source_type' not in df.columns or 'global_region' not in df.columns:
        print('source_type or global_region missing — skipping plot 11')
        return

    type_order = (
        df.groupby('source_type')['actionability_percentage']
        .mean().sort_values(ascending=False).index.tolist()
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=df, x='source_type', y='actionability_percentage',
        hue='global_region', order=type_order,
        palette={'Global North': '#4e79a7', 'Global South': '#f28e2b'},
        ax=ax, errorbar='sd', alpha=0.9
    )
    ax.set_title(
        'Mean actionability by source type and region\n'
        'Global South advantage is concentrated in national news — not a regional pattern',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel('')
    ax.set_ylabel('Mean actionability (%) — mean ± SD')
    ax.tick_params(axis='x', rotation=25)
    ax.legend(title='Region')
    plt.tight_layout()
    _save(fig, '11_source_type_by_region.png')


# ── 12. Cluster × PADM components heatmap ────────────────────────────────────
def plot_cluster_padm_heatmap(df: pd.DataFrame) -> None:
    cluster_csv = os.path.join(OUT_DIR, 'cluster_summary_structural_k3.csv')
    if not os.path.exists(cluster_csv):
        print('cluster_summary_structural_k3.csv not found — skipping plot 12')
        return

    cs = pd.read_csv(cluster_csv)
    cs['label'] = cs['cluster'].str.extract(r'_c(\d+)$')[0].astype(int).apply(
        lambda x: f'Cluster {x}'
    )
    cs = cs.sort_values('label').set_index('label')

    feature_cols = {
        'mean_imperative_count': 'Imperative\nsignals',
        'mean_short_term_count': 'Short-term\nurgency',
        'mean_long_term_count':  'Long-term\nrecovery',
        'mean_spatial_count':    'Spatial\nanchors',
        'mean_advice':           'Advice-\nframing',
    }
    available = {k: v for k, v in feature_cols.items() if k in cs.columns}
    heat = cs[[*available.keys()]].rename(columns=available).astype(float)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        heat, ax=ax, cmap='YlOrRd',
        annot=heat.round(3), fmt='g',
        linewidths=0.5, linecolor='#dddddd',
        cbar_kws={'label': 'Mean score per cluster'},
    )
    ax.set_title(
        'Cluster profiles across PADM structural features (k=3, structural clustering)\n'
        'Colour = raw mean score per cluster',
        fontsize=11, fontweight='bold'
    )
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    _save(fig, '12_cluster_padm_heatmap.png')


# ── 8. Cluster profiles ───────────────────────────────────────────────────────
def plot_cluster_profiles(df: pd.DataFrame) -> None:
    cluster_csv = os.path.join(OUT_DIR, 'cluster_summary_structural_k3.csv')
    if not os.path.exists(cluster_csv):
        print('cluster_summary_structural_k3.csv not found — skipping plot 8')
        return

    cs = pd.read_csv(cluster_csv)

    # extract short cluster ID from the label string e.g. 'structural_k4_c2' -> 'Cluster 2'
    cs['label'] = cs['cluster'].str.extract(r'_c(\d+)$')[0].astype(int).apply(
        lambda x: f'Cluster {x}'
    )
    cs = cs.sort_values('label').reset_index(drop=True)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(cs))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('K-Means cluster analysis (structural features, k=3)', fontsize=12, fontweight='bold')

    axes[0].bar(cs['label'], cs['actionability_percentage_mean'], color=colors, edgecolor='white')
    axes[0].set_title('Mean actionability per cluster')
    axes[0].set_ylabel('Actionability (%)')
    axes[0].tick_params(axis='x', rotation=15)

    axes[1].bar(cs['label'], cs['n_articles'], color=colors, edgecolor='white')
    axes[1].set_title('Article count per cluster')
    axes[1].set_ylabel('Number of articles')
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    _save(fig, '08_cluster_profiles.png')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    enriched_path = os.path.join(OUT_DIR, 'enriched.csv')
    if not os.path.exists(enriched_path):
        print(f'enriched.csv not found at {enriched_path} — run the pipeline first')
        sys.exit(1)

    print('loading enriched.csv...')
    df = pd.read_csv(enriched_path)
    print(f'{len(df)} articles | columns: {list(df.columns)}')

    # run clustering if columns not already present
    if 'global_region' not in df.columns or 'data_cluster_id' not in df.columns:
        print('running clustering...')
        from nlp.clustering import run_clustering
        df = run_clustering(df)

    print('generating visualizations...')
    plot_language_distribution(df)
    plot_country_distribution(df)
    plot_actionability_by_language(df)
    plot_actionability_by_region(df)
    plot_actionability_by_source(df)
    plot_frame_distribution(df)
    plot_frame_by_region(df)
    plot_cluster_profiles(df)
    plot_padm_components(df)
    plot_frame_actionability(df)
    plot_frame_actionability_by_language(df)
    plot_source_region(df)
    plot_cluster_padm_heatmap(df)

    print(f'\nall outputs saved to: {VIZ_DIR}')
    for f in sorted(os.listdir(VIZ_DIR)):
        print(f'  {f}')
