"""
period_analysis.py
------------------
Sub-period PCA comparison engine for the Stock PCA Cluster app.

Implements three validation methods:
  1. Side-by-side factor loading bar charts
  2. Procrustes disparity scores (pairwise)
  3. Quadrant migration table (stock-level)

Drop this file into stockapp/ alongside utils.py, visualizations.py, etc.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import procrustes
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Sub-period definitions ────────────────────────────────────────────────────
SUB_PERIODS = {
    'Post-COVID\n(Mar 2021–Jun 2022)':  ('2021-03-31', '2022-06-30'),
    'Rate Shock\n(Jul 2022–Sep 2023)':  ('2022-07-31', '2023-09-30'),
    'Disinflation\n(Oct 2023–Oct 2024)': ('2023-10-31', '2024-10-31'),
}

# Short labels used as dict keys elsewhere
PERIOD_KEYS = list(SUB_PERIODS.keys())

# Human-readable feature names  (must match the order your PCA uses)
FEATURE_DISPLAY_NAMES = {
    'roa':            'ROA',
    'roe':            'ROE',
    'cash_to_debt':   'Cash-to-Debt',
    'vol_60d':        '60d Volatility',
    'sales_to_price': 'Sales/Price',
    'book_to_market': 'Book/Market',
    'earnings_yield': 'Earnings Yield',
    'debt_to_assets': 'Debt/Assets',
    'gross_profit':   'Gross Profitability',
    'momentum_12m':   'Momentum 12M',
    'size':           'Size (log MktCap)',
}


# ── Core: run PCA on one sub-period ──────────────────────────────────────────

def _run_pca_for_period(df: pd.DataFrame, features: list, date_col: str,
                         start: str, end: str, n_components: int = 3):
    """
    Filter df to [start, end], average each ticker, run PCA.

    Returns
    -------
    pca_model   : fitted sklearn PCA object
    scores_df   : DataFrame with columns [ticker, PC1, PC2, PC3, Quadrant]
    loadings_df : DataFrame  (features × PCs)
    feature_list: list of features actually used
    """
    mask = (df[date_col] >= pd.Timestamp(start)) & (df[date_col] <= pd.Timestamp(end))
    sub = df.loc[mask].copy()

    if sub.empty:
        return None, None, None, None

    # Average across time for each ticker
    ticker_col = next((c for c in ['ticker', 'TICKER'] if c in sub.columns), None)
    grp = sub.groupby(ticker_col)[features].mean().dropna()

    if len(grp) < 10:
        return None, None, None, None

    scaler = StandardScaler()
    X = scaler.fit_transform(grp)

    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X)

    scores_df = pd.DataFrame(scores, index=grp.index,
                             columns=[f'PC{i+1}' for i in range(n_components)])
    scores_df.index.name = 'ticker'
    scores_df = scores_df.reset_index()

    # Assign quadrants based on PC1/PC2 signs
    scores_df['Quadrant'] = scores_df.apply(_assign_quadrant, axis=1)

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )

    return pca, scores_df, loadings_df, features


def _assign_quadrant(row):
    if row['PC1'] >= 0 and row['PC2'] >= 0:
        return 'Q1: Profitable Value'
    elif row['PC1'] < 0 and row['PC2'] >= 0:
        return 'Q2: Value Traps/Distressed'
    elif row['PC1'] < 0 and row['PC2'] < 0:
        return 'Q3: Struggling Growth'
    else:
        return 'Q4: Quality Growth'


# ── Method 1: Loading Bar Charts ──────────────────────────────────────────────

def create_loading_comparison_chart(df: pd.DataFrame, features: list,
                                    date_col: str, pc: str = 'PC1') -> go.Figure:
    """
    Side-by-side grouped bar chart of factor loadings for PC1 or PC2
    across the three sub-periods.
    """
    period_loadings = {}
    for label, (start, end) in SUB_PERIODS.items():
        _, _, loadings_df, _ = _run_pca_for_period(df, features, date_col, start, end)
        if loadings_df is not None:
            period_loadings[label] = loadings_df[pc]

    if not period_loadings:
        return go.Figure()

    # Map raw feature names to display names
    display = [FEATURE_DISPLAY_NAMES.get(f, f) for f in features]
    colors = ['#4C78A8', '#F58518', '#54A24B']

    fig = go.Figure()
    for (label, series), color in zip(period_loadings.items(), colors):
        short_label = label.split('\n')[0]   # first line only for legend
        fig.add_trace(go.Bar(
            name=short_label,
            x=display,
            y=series.values,
            marker_color=color,
            opacity=0.85,
        ))

    fig.add_hline(y=0, line_width=1, line_color='white', opacity=0.4)

    fig.update_layout(
        barmode='group',
        title=dict(
            text=f'{pc} Factor Loadings — Sub-Period Comparison',
            font=dict(size=16)
        ),
        xaxis_title='Feature',
        yaxis_title='Loading Weight',
        legend_title='Sub-Period',
        height=450,
        template='plotly_dark',
        margin=dict(t=60, b=80),
    )
    return fig


# ── Method 2: Procrustes Analysis ────────────────────────────────────────────

def compute_procrustes_table(df: pd.DataFrame, features: list,
                              date_col: str) -> pd.DataFrame:
    """
    Run pairwise Procrustes analysis across sub-periods on the PC score matrices.

    Returns a DataFrame with columns:
        Period A | Period B | Common Tickers | Disparity | Interpretation
    """
    # Collect scores for each period
    period_scores = {}
    for label, (start, end) in SUB_PERIODS.items():
        _, scores_df, _, _ = _run_pca_for_period(df, features, date_col, start, end)
        if scores_df is not None:
            period_scores[label] = scores_df.set_index('ticker')[['PC1', 'PC2', 'PC3']]

    results = []
    keys = list(period_scores.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            label_a, label_b = keys[i], keys[j]
            sa = period_scores[label_a]
            sb = period_scores[label_b]

            # Intersection of tickers
            common = sa.index.intersection(sb.index)
            if len(common) < 20:
                results.append({
                    'Period A': label_a.split('\n')[0],
                    'Period B': label_b.split('\n')[0],
                    'Common Tickers': len(common),
                    'Disparity': None,
                    'Interpretation': 'Too few common tickers'
                })
                continue

            mat_a = sa.loc[common].values.astype(float)
            mat_b = sb.loc[common].values.astype(float)

            _, _, disparity = procrustes(mat_a, mat_b)

            if disparity < 0.05:
                interp = '🟢 Very similar factor structure'
            elif disparity < 0.15:
                interp = '🟡 Moderate structural shift'
            elif disparity < 0.30:
                interp = '🟠 Meaningful structural change'
            else:
                interp = '🔴 Major regime change'

            results.append({
                'Period A': label_a.split('\n')[0],
                'Period B': label_b.split('\n')[0],
                'Common Tickers': len(common),
                'Disparity': round(disparity, 4),
                'Interpretation': interp
            })

    return pd.DataFrame(results)


def create_procrustes_heatmap(procrustes_df: pd.DataFrame) -> go.Figure:
    """
    Render a clean heatmap of Procrustes disparity scores.
    """
    periods = list(dict.fromkeys(
        procrustes_df['Period A'].tolist() + procrustes_df['Period B'].tolist()
    ))
    n = len(periods)
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, 0)

    for _, row in procrustes_df.iterrows():
        i = periods.index(row['Period A'])
        j = periods.index(row['Period B'])
        val = row['Disparity'] if row['Disparity'] is not None else 0
        matrix[i][j] = val
        matrix[j][i] = val

    # Annotation text
    text = []
    for i in range(n):
        row_text = []
        for j in range(n):
            if i == j:
                row_text.append('—')
            else:
                row_text.append(f'{matrix[i][j]:.3f}')
        text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=periods,
        y=periods,
        text=text,
        texttemplate='%{text}',
        colorscale='RdYlGn_r',
        zmin=0, zmax=0.5,
        showscale=True,
        colorbar=dict(title='Disparity<br>(0=same, 1=max diff)')
    ))

    fig.update_layout(
        title='Procrustes Disparity — Pairwise Sub-Period Comparison',
        height=350,
        template='plotly_dark',
        margin=dict(t=60),
    )
    return fig


# ── Method 3: Quadrant Migration Table ───────────────────────────────────────

def compute_quadrant_migration(df: pd.DataFrame, features: list,
                                date_col: str) -> tuple:
    """
    For each stock, track which quadrant it occupied in each sub-period.

    Returns
    -------
    migration_df   : wide DataFrame  (ticker | Q per period | changed?)
    summary_df     : summary stats per period-pair
    migration_pct  : % of stocks that changed quadrant across ALL periods
    """
    period_quadrants = {}
    for label, (start, end) in SUB_PERIODS.items():
        _, scores_df, _, _ = _run_pca_for_period(df, features, date_col, start, end)
        if scores_df is not None:
            short = label.split('\n')[0]
            period_quadrants[short] = scores_df.set_index('ticker')['Quadrant']

    if len(period_quadrants) < 2:
        return pd.DataFrame(), pd.DataFrame(), None

    # Build wide table
    wide = pd.DataFrame(period_quadrants)
    wide = wide.dropna()   # only stocks present in ALL periods
    wide.index.name = 'Ticker'

    period_names = list(period_quadrants.keys())

    # Flag any quadrant change across all periods
    wide['Any Change'] = wide.apply(
        lambda r: len(set(r[period_names])) > 1, axis=1
    )

    migration_pct = wide['Any Change'].mean() * 100

    # Pairwise summary
    summary_rows = []
    for i in range(len(period_names) - 1):
        pa, pb = period_names[i], period_names[i + 1]
        changed = (wide[pa] != wide[pb]).sum()
        total   = len(wide)
        summary_rows.append({
            'Transition': f'{pa} → {pb}',
            'Stocks Analyzed': total,
            'Changed Quadrant': changed,
            'Stayed Same': total - changed,
            'Migration Rate': f'{changed / total * 100:.1f}%'
        })

    summary_df = pd.DataFrame(summary_rows)
    return wide.reset_index(), summary_df, migration_pct


def create_migration_sankey(migration_df: pd.DataFrame,
                             period_names: list) -> go.Figure:
    """
    Sankey diagram showing stock flow between quadrants across sub-periods.
    """
    quadrant_order = [
        'Q1: Profitable Value',
        'Q2: Value Traps/Distressed',
        'Q3: Struggling Growth',
        'Q4: Quality Growth',
    ]
    q_colors = {
        'Q1: Profitable Value':       '#54A24B',
        'Q2: Value Traps/Distressed': '#E45756',
        'Q3: Struggling Growth':      '#F58518',
        'Q4: Quality Growth':         '#4C78A8',
    }

    # Build node list: each period × each quadrant
    nodes = []
    node_colors = []
    for period in period_names:
        for q in quadrant_order:
            nodes.append(f'{period}<br>{q.split(":")[0]}')
            node_colors.append(q_colors[q])

    def node_idx(period_idx, q):
        return period_idx * len(quadrant_order) + quadrant_order.index(q)

    source, target, value = [], [], []

    for p_idx in range(len(period_names) - 1):
        pa = period_names[p_idx]
        pb = period_names[p_idx + 1]
        if pa not in migration_df.columns or pb not in migration_df.columns:
            continue
        flow = migration_df.groupby([pa, pb]).size().reset_index(name='count')
        for _, row in flow.iterrows():
            if row[pa] in quadrant_order and row[pb] in quadrant_order:
                source.append(node_idx(p_idx, row[pa]))
                target.append(node_idx(p_idx + 1, row[pb]))
                value.append(row['count'])

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=nodes,
            color=node_colors,
        ),
        link=dict(source=source, target=target, value=value),
    ))

    fig.update_layout(
        title='Quadrant Migration Flow Across Sub-Periods',
        height=500,
        template='plotly_dark',
        margin=dict(t=60),
    )
    return fig


# ── Public entry point ────────────────────────────────────────────────────────

def get_features_from_df(df: pd.DataFrame) -> list:
    """
    Infer which feature columns are present in the data.
    Tries known feature names in order; returns those found.
    """
    candidates = list(FEATURE_DISPLAY_NAMES.keys())
    return [f for f in candidates if f in df.columns]
