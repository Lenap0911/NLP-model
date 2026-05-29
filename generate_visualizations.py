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


# ── 7. Frame × Global region ──────────────────────────────────────────────────
def plot_frame_by_region(df: pd.DataFrame) -> None:
    if 'dominant_frame' not in df.columns or 'global_region' not in df.columns:
        print('dominant_frame or global_region missing — skipping plot 7')
        return

    frame_region = (
        df.groupby(['global_region', 'dominant_frame'])
        .size().reset_index(name='count')
    )
    pivot = frame_region.pivot(index='dominant_frame', columns='global_region', values='count').fillna(0)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot.bar(ax=ax, color=['#4e79a7', '#f28e2b'], edgecolor='white', alpha=0.88)
    ax.set_title('Frame distribution by global region', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Number of articles')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Region')
    plt.tight_layout()
    _save(fig, '07_frame_by_region.png')


# ── 8. Cluster profiles ───────────────────────────────────────────────────────
def plot_cluster_profiles(df: pd.DataFrame) -> None:
    if 'data_cluster_id' not in df.columns:
        print('data_cluster_id column not found — skipping plot 8')
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('HDBSCAN cluster analysis', fontsize=12, fontweight='bold')

    # mean actionability per cluster
    cluster_means = (
        df.groupby('data_cluster_id')['actionability_percentage']
        .agg(['mean', 'count']).reset_index()
    )
    cluster_means['label'] = cluster_means['data_cluster_id'].apply(
        lambda x: 'Noise' if x == -1 else f'Cluster {x}'
    )
    colors = ['#aaaaaa' if x == -1 else PALETTE[i % len(PALETTE)]
              for i, x in enumerate(cluster_means['data_cluster_id'])]

    axes[0].bar(cluster_means['label'], cluster_means['mean'], color=colors, edgecolor='white')
    axes[0].set_title('Mean actionability per cluster')
    axes[0].set_ylabel('Actionability (%)')
    axes[0].tick_params(axis='x', rotation=15)

    # cluster size
    axes[1].bar(cluster_means['label'], cluster_means['count'], color=colors, edgecolor='white')
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

    print(f'\nall outputs saved to: {VIZ_DIR}')
    for f in sorted(os.listdir(VIZ_DIR)):
        print(f'  {f}')
