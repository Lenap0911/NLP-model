# generate_visualizations.py
# produces visualizations from the enriched pipeline output
# run from the project root:  python generate_visualizations.py

import os
import sys
import importlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
config = importlib.import_module('config.nlp_config')

OUT_DIR     = os.path.join(ROOT, 'output')
STATS_DIR   = os.path.join(OUT_DIR, 'stats')
VIZ_DIR     = os.path.join(OUT_DIR, 'visualizations')
GENERAL_DIR = os.path.join(VIZ_DIR, 'general_graphs')
CLUSTER_DIR = os.path.join(VIZ_DIR, 'clustering_graphs')

for d in (VIZ_DIR, GENERAL_DIR, CLUSTER_DIR):
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

LANG_LABELS = {'en': 'English', 'es': 'Spanish', 'pt': 'Portuguese'}
LANG_PALETTE = {'en': '#295574', 'es': '#e75e1e', 'pt': '#2ca062'}
PALETTE = sns.color_palette('tab10')


def _save(fig: plt.Figure, name: str, folder: str = VIZ_DIR) -> None:
    path = os.path.join(folder, name)
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
    _save(fig, '01_language_distribution.png', GENERAL_DIR)


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
    _save(fig, '02_country_distribution.png', GENERAL_DIR)


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
    _save(fig, '03_actionability_by_language.png', GENERAL_DIR)


# ── 4. Actionability by region ────────────────────────────────────────────────
def plot_actionability_by_region(df: pd.DataFrame) -> None:
    if 'region' not in df.columns:
        print('region column not found — skipping plot 4')
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Actionability: North America vs South America', fontsize=12, fontweight='bold')

    sns.boxplot(data=df, x='region', y='actionability_percentage',
                palette=['#4e79a7', '#f28e2b'], ax=axes[0])
    axes[0].set_title('Distribution')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Actionability (%)')

    counts = df['region'].value_counts()
    counts.plot.bar(ax=axes[1], color=['#4e79a7', '#f28e2b'], edgecolor='white')
    axes[1].set_title('Article count')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Number of articles')
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    _save(fig, '04_actionability_by_region.png', GENERAL_DIR)


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
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    _save(fig, '05_actionability_by_source_type.png', GENERAL_DIR)


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
    _save(fig, '06_frame_distribution.png', GENERAL_DIR)


# ── 7. Frame × region ────────────────────────────────────────────────────────
def plot_frame_by_region(df: pd.DataFrame) -> None:
    if 'dominant_frame' not in df.columns or 'region' not in df.columns:
        print('dominant_frame or region missing — skipping plot 7')
        return

    frame_region = (
        df.groupby(['region', 'dominant_frame'])
        .size().reset_index(name='count')
    )
    totals = frame_region.groupby('region')['count'].transform('sum')
    frame_region['pct'] = frame_region['count'] / totals * 100
    pivot = frame_region.pivot(index='dominant_frame', columns='region', values='pct').fillna(0)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot.bar(ax=ax, color=['#4e79a7', '#f28e2b'], edgecolor='white', alpha=0.88)
    ax.set_title('Frame distribution by region', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('% of articles in region')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Region')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=8, padding=2)
    plt.tight_layout()
    _save(fig, '07_frame_by_region.png', GENERAL_DIR)


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
    ax.set_title('Mean actionability by dominant frame', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Mean actionability (%)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, '10_frame_actionability.png', GENERAL_DIR)


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
        ax.set_title(f'Mean actionability by dominant frame — {label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Mean actionability (%)')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        plt.tight_layout()
        _save(fig, f'10_{lang}_frame_actionability.png', GENERAL_DIR)


# ── 13b. Actionability range histogram ───────────────────────────────────────
def plot_actionability_range_bar(df: pd.DataFrame) -> None:
    s = df['actionability_percentage'].dropna()

    mean_val   = s.mean()
    median_val = s.median()
    q25, q75   = s.quantile(0.25), s.quantile(0.75)
    q10, q90   = s.quantile(0.10), s.quantile(0.90)

    bin_edges = np.arange(0, s.max() + 2, 2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(s, bins=bin_edges, color='#4e79a7', edgecolor='white', alpha=0.85, zorder=2)
    ax.axvspan(q25, q75, alpha=0.15, color='#4e79a7', label=f'IQR (Q1={q25:.1f}%–Q3={q75:.1f}%)', zorder=1)
    ax.axvspan(q10, q90, alpha=0.08, color='#4e79a7', label=f'10th–90th pct ({q10:.1f}%–{q90:.1f}%)', zorder=1)
    ax.axvline(mean_val,   color='#e15759', linewidth=2, linestyle='--', label=f'Mean = {mean_val:.2f}%',    zorder=3)
    ax.axvline(median_val, color='#f28e2b', linewidth=2, linestyle=':',  label=f'Median = {median_val:.1f}%', zorder=3)

    ax.set_yscale('log')
    for patch in ax.patches:
        h = patch.get_height()
        if h > 0:
            ax.text(patch.get_x() + patch.get_width() / 2, h * 1.15,
                    str(int(h)), ha='center', va='bottom', fontsize=8, color='#333333')

    ax.set_xlabel('Article-level actionability (%)', fontsize=12, labelpad=8)
    ax.set_ylabel('Number of articles (log scale)', fontsize=12, labelpad=8)
    ax.set_title(f'Distribution of article-level actionability  (N={len(s):,} articles)',
                 fontsize=13, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    _save(fig, '13b_actionability_range_histogram.png', GENERAL_DIR)


# ── 15. National news vs Other outlets ───────────────────────────────────────
def plot_national_vs_other(df: pd.DataFrame) -> None:
    if 'source_type' not in df.columns:
        print('source_type missing — skipping plot 15')
        return

    df = df.copy()
    df['source_group'] = df['source_type'].apply(
        lambda x: 'National news' if x == 'national_news' else 'Other outlets'
    )
    group_colors = {'National news': '#4e79a7', 'Other outlets': '#aaaaaa'}
    order = ['National news', 'Other outlets']

    means   = df.groupby('source_group')['actionability_percentage'].mean()
    medians = df.groupby('source_group')['actionability_percentage'].median()
    counts  = df.groupby('source_group')['actionability_percentage'].count()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Actionability: National news vs Other outlets', fontsize=13, fontweight='bold')

    sns.boxplot(data=df, x='source_group', y='actionability_percentage',
                order=order, palette=group_colors, ax=axes[0], width=0.5)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Actionability (%)')
    axes[0].set_title('Distribution')
    for i, grp in enumerate(order):
        axes[0].text(i, axes[0].get_ylim()[1] * 0.97,
                     f'mean={means[grp]:.2f}%\nmedian={medians[grp]:.1f}%',
                     ha='center', va='top', fontsize=9, color='#333333')

    means_sd = df.groupby('source_group')['actionability_percentage'].agg(['mean', 'std']).reindex(order)
    bars = axes[1].bar(order, means_sd['mean'],
                       yerr=means_sd['std'], capsize=5,
                       color=[group_colors[g] for g in order],
                       edgecolor='white', width=0.5, error_kw={'elinewidth': 1.2})
    for bar, grp in zip(bars, order):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + means_sd.loc[grp, 'std'] + 0.3,
                     f'n={counts[grp]}', ha='center', va='bottom', fontsize=9, color='#333333')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Mean actionability (%) ± SD')
    axes[1].set_title('Mean ± SD')
    axes[1].set_ylim(bottom=0)
    plt.tight_layout()
    _save(fig, '15_national_vs_other_outlets.png', GENERAL_DIR)


# ── actionability by source type and region, faceted by language ──────────────
def plot_actionability_source_region_by_language(df: pd.DataFrame) -> None:
    if 'source_type' not in df.columns or 'region' not in df.columns:
        print('source_type or region missing — skipping actionability_source_region_by_language')
        return

    region_palette = {'North America': '#4e79a7', 'South America': '#f28e2b'}
    lang_names = {'en': 'English', 'es': 'Spanish', 'pt': 'Portuguese'}
    langs = [l for l in ['en', 'es', 'pt'] if l in df['language'].values]

    fig, axes = plt.subplots(len(langs), 1, figsize=(11, 5 * len(langs)), squeeze=False)

    for ax, lang in zip(axes[:, 0], langs):
        sub = df[df['language'] == lang].copy()
        type_order = (
            sub.groupby('source_type')['actionability_percentage']
            .mean().sort_values(ascending=False).index.tolist()
        )
        sns.barplot(
            data=sub, x='source_type', y='actionability_percentage',
            hue='region', order=type_order,
            palette=region_palette,
            ax=ax, errorbar='sd', alpha=0.9,
        )
        ax.set_title(lang_names[lang], fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Mean actionability (%)')
        ax.tick_params(axis='x', rotation=20)
        ax.legend(title='Region', fontsize=9)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    _save(fig, 'actionability_source_region_by_language.png', GENERAL_DIR)


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
    x = range(len(components))
    width = 0.25
    comp_labels = list(components.keys())

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, lang in enumerate(lang_order):
        sub = df[df['language'] == lang]
        pcts = []
        for col in components.values():
            if col == 'mean_imperative_count' and 'mean_verbs_imperative_count' in sub.columns:
                present = (sub['mean_imperative_count'] > 0) | (sub['mean_verbs_imperative_count'] > 0)
                pcts.append(present.mean() * 100)
            else:
                pcts.append((sub[col] > 0).mean() * 100)
        ax.bar(
            [xi + i * width for xi in x], pcts,
            width=width, label=LANG_LABELS.get(lang, lang),
            color=LANG_PALETTE[lang], edgecolor='white', alpha=0.9,
        )

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(comp_labels, fontsize=10)
    ax.set_ylabel('% of articles with component present')
    ax.set_title('PADM component presence by language', fontsize=12, fontweight='bold')
    ax.legend(title='Language')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    _save(fig, '09_padm_components_by_language.png', CLUSTER_DIR)


# ── 11. Source type × Region ──────────────────────────────────────────────────
def plot_source_region(df: pd.DataFrame) -> None:
    if 'source_type' not in df.columns or 'region' not in df.columns:
        print('source_type or region missing — skipping plot 11')
        return

    type_order = (
        df.groupby('source_type')['actionability_percentage']
        .mean().sort_values(ascending=False).index.tolist()
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=df, x='source_type', y='actionability_percentage',
        hue='region', order=type_order,
        palette={'North America': '#4e79a7', 'South America': '#f28e2b'},
        ax=ax, errorbar='sd', alpha=0.9,
    )
    ax.set_title('Mean actionability by source type and region', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Mean actionability (%) — mean ± SD')
    ax.tick_params(axis='x', rotation=25)
    ax.legend(title='Region')
    plt.tight_layout()
    _save(fig, '11_source_type_by_region.png', CLUSTER_DIR)


# ── 12. Cluster × PADM heatmap ───────────────────────────────────────────────
def plot_cluster_padm_heatmap(df: pd.DataFrame) -> None:
    cluster_csv = os.path.join(STATS_DIR, 'cluster_summary_structural_k3.csv')
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
    ax.set_title('Cluster profiles across PADM structural features', fontsize=11, fontweight='bold')
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    _save(fig, '12_cluster_padm_heatmap.png', CLUSTER_DIR)


# ── 8. Cluster profiles ───────────────────────────────────────────────────────
def plot_cluster_profiles(df: pd.DataFrame) -> None:
    cluster_csv = os.path.join(STATS_DIR, 'cluster_summary_structural_k3.csv')
    if not os.path.exists(cluster_csv):
        print('cluster_summary_structural_k3.csv not found — skipping plot 8')
        return

    cs = pd.read_csv(cluster_csv)
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
    _save(fig, '08_cluster_profiles.png', CLUSTER_DIR)


# ── 12b. Frame distribution by cluster ───────────────────────────────────────
def plot_frame_by_cluster(df: pd.DataFrame) -> None:
    if 'data_cluster_id' not in df.columns or 'dominant_frame' not in df.columns:
        print('data_cluster_id or dominant_frame missing — skipping plot 12b')
        return

    frame_colors = {
        'impact':         '#e15759',
        'response':       '#4e79a7',
        'accountability': '#f28e2b',
        'recovery':       '#59a14f',
    }
    frame_order = ['impact', 'response', 'accountability', 'recovery']

    plot_df = df[['data_cluster_id', 'dominant_frame']].dropna()
    plot_df = plot_df[plot_df['dominant_frame'].isin(frame_order)].copy()
    plot_df['cluster_label'] = plot_df['data_cluster_id'].astype(int).apply(lambda x: f'Cluster {x}')

    ct = (
        plot_df.groupby(['cluster_label', 'dominant_frame'])
               .size().unstack(fill_value=0)
               .reindex(columns=frame_order, fill_value=0)
    )
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct = ct_pct.sort_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = [0.0] * len(ct_pct)
    for frame in frame_order:
        vals = ct_pct[frame].values if frame in ct_pct.columns else [0.0] * len(ct_pct)
        bars = ax.bar(ct_pct.index, vals, bottom=bottom,
                      label=frame.capitalize(), color=frame_colors[frame],
                      edgecolor='white', width=0.55)
        for bar, val in zip(bars, vals):
            if val >= 6:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.0f}%', ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold')
        bottom = [b + v for b, v in zip(bottom, vals)]

    for i, (_, row) in enumerate(ct.iterrows()):
        ax.text(i, 102, f'n={row.sum()}', ha='center', va='bottom', fontsize=9, color='#444444')

    ax.set_xlabel('Structural cluster', fontsize=12, labelpad=8)
    ax.set_ylabel('Share of articles (%)', fontsize=12, labelpad=8)
    ax.set_title('Dominant frame distribution by cluster', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 112)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(title='Frame', bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, '12b_frame_by_cluster.png', CLUSTER_DIR)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    enriched_path = os.path.join(OUT_DIR, 'enriched.csv')
    if not os.path.exists(enriched_path):
        print(f'enriched.csv not found at {enriched_path} — run the pipeline first')
        sys.exit(1)

    print('loading enriched.csv...')
    df = pd.read_csv(enriched_path)
    if 'global_region' in df.columns and 'region' not in df.columns:
        df = df.rename(columns={'global_region': 'region'})
    if 'region' in df.columns:
        df['region'] = df['region'].replace(
            {'Global North': 'North America', 'Global South': 'South America'}
        )
    print(f'{len(df)} articles | columns: {list(df.columns)}')

    if 'region' not in df.columns or 'data_cluster_id' not in df.columns:
        print('running clustering...')
        from nlp.clustering import run_clustering
        df = run_clustering(df)

    print('generating visualizations...')
    # general_graphs
    plot_language_distribution(df)
    plot_country_distribution(df)
    plot_actionability_by_language(df)
    plot_actionability_by_region(df)
    plot_actionability_by_source(df)
    plot_frame_distribution(df)
    plot_frame_by_region(df)
    plot_frame_actionability(df)
    plot_frame_actionability_by_language(df)
    plot_actionability_range_bar(df)
    plot_national_vs_other(df)
    plot_actionability_source_region_by_language(df)
    # clustering_graphs
    plot_padm_components(df)
    plot_source_region(df)
    plot_cluster_padm_heatmap(df)
    plot_cluster_profiles(df)
    plot_frame_by_cluster(df)

    print(f'\nall outputs saved to:')
    for folder in (GENERAL_DIR, CLUSTER_DIR):
        print(f'  {folder}')
        for f in sorted(os.listdir(folder)):
            print(f'    {f}')
