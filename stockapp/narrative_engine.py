"""
narrative_engine.py
-------------------
Rule-based narrative interpretation engine for Stock PCA Cluster Analysis.
Generates plain-English analysis from PCA session state data — no API key required.

Produces four analysis sections for any selected stock:
  1. Overall Position Summary  (quadrant + PC1/PC2 plain-English)
  2. Top 3 Factor Strengths & Weaknesses
  3. Time Trajectory Narrative  (from raw_data time series)
  4. Peer Context  (how the stock compares within its quadrant)

Usage in app.py:
    from narrative_engine import generate_narrative
    sections = generate_narrative(
        ticker      = ticker,
        pca_row     = pca_row,        # Single-row Series/dict from pca_df
        percentiles = percentiles,    # Dict {feature: percentile_value}
        factor_data = factor_data,    # Dict {category: {feature: value}}
        peer_df     = peer_df,        # DataFrame of peers in same quadrant
        raw_data    = raw_data,       # Full time-series DataFrame
        loadings    = loadings        # st.session_state.pca_loadings
    )
    # Returns dict with keys: 'summary', 'factors', 'trajectory', 'peers'
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any

# ---------------------------------------------------------------------------
# Feature display name mapping — mirrors config.py / visualizations.py
# ---------------------------------------------------------------------------
FEATURE_DISPLAY_NAMES = {
    'earnings_yield': 'Earnings Yield',
    'bm':             'Book-to-Market',
    'sales_to_price': 'Sales-to-Price',
    'roe':            'Return on Equity (ROE)',
    'roa':            'Return on Assets (ROA)',
    'gprof':          'Gross Profitability',
    'debt_assets':    'Debt-to-Assets',
    'cash_debt':      'Cash-to-Debt',
    'momentum_12m':   '12-Mo. Momentum',
    'vol_60d_ann':    '60-Day Volatility',
    'addv_63d':       'Liquidity',
}

# Quadrant metadata
_QUADRANT_META = {
    'Q1': {
        'name':    'Profitable Value',
        'summary': 'strong operational quality with an attractive valuation',
        'detail':  (
            'This is a relatively rare combination — the stock demonstrates solid '
            'profitability and financial health while trading at valuations that '
            'suggest the market has not yet fully priced in its fundamentals.'
        ),
    },
    'Q2': {
        'name':    'Value Traps / Distressed',
        'summary': 'cheap valuation but weak operational quality',
        'detail':  (
            'Low valuations may look attractive on the surface, but they are '
            'accompanied by weaker profitability and financial health metrics. '
            'Caution is warranted — low prices can reflect genuine fundamental risk.'
        ),
    },
    'Q3': {
        'name':    'Struggling Growth',
        'summary': 'elevated growth-style valuation but below-average operational quality',
        'detail':  (
            'The market is pricing in future potential, but current operational '
            'metrics have not yet validated that premium. This is typical of '
            'early-stage or turnaround situations where execution risk remains high.'
        ),
    },
    'Q4': {
        'name':    'Quality Growth',
        'summary': 'high operational quality combined with a growth-oriented valuation',
        'detail':  (
            'The stock earns its premium valuation through demonstrated profitability '
            'and financial strength. Investors are paying up for quality — a '
            'defensible position as long as those fundamentals remain intact.'
        ),
    },
}


# ===========================================================================
# SECTION 1 — OVERALL POSITION SUMMARY
# ===========================================================================

def _describe_pc1(pc1: float) -> str:
    """Plain-English description of PC1 score (Profitability & Operational Quality)."""
    if pc1 >= 1.5:
        return (
            f"a PC1 score of {pc1:.2f} — well above average — signals exceptionally strong "
            "profitability and operational quality. ROA, ROE, and cash-to-debt ratios are "
            "all pointing to a high-quality business with durable financial health."
        )
    elif pc1 >= 0.5:
        return (
            f"a PC1 score of {pc1:.2f} indicates above-average operational quality. "
            "Profitability metrics (ROA, ROE) and financial strength (cash-to-debt) are "
            "solid, placing this stock in favorable territory on the quality axis."
        )
    elif pc1 >= -0.5:
        return (
            f"a PC1 score of {pc1:.2f} reflects near-average operational quality — "
            "neither a standout on profitability nor a concern on financial health. "
            "Performance on ROA, ROE, and cash-to-debt is in line with the broader universe."
        )
    elif pc1 >= -1.5:
        return (
            f"a PC1 score of {pc1:.2f} indicates below-average operational quality. "
            "Profitability (ROA, ROE) and/or financial health (cash-to-debt) are weaker "
            "relative to peers, which may warrant closer scrutiny."
        )
    else:
        return (
            f"a PC1 score of {pc1:.2f} — well below average — reflects material weakness "
            "in operational quality. Low profitability and/or elevated financial risk "
            "(captured by ROA, ROE, and cash-to-debt) place this stock in the bottom tier."
        )


def _describe_pc2(pc2: float) -> str:
    """Plain-English description of PC2 score (Valuation Style: Value vs Growth)."""
    if pc2 >= 1.5:
        return (
            f"on the valuation axis (PC2: {pc2:.2f}), the stock screens as deep value — "
            "high sales-to-price, book-to-market, and earnings yield ratios suggest the "
            "market is pricing it at a significant discount relative to its fundamentals."
        )
    elif pc2 >= 0.5:
        return (
            f"on the valuation axis (PC2: {pc2:.2f}), the stock leans value-oriented. "
            "Metrics like sales-to-price and book-to-market are above average, indicating "
            "a relatively modest market valuation relative to its financials."
        )
    elif pc2 >= -0.5:
        return (
            f"on the valuation axis (PC2: {pc2:.2f}), the stock sits near the blend point — "
            "neither a clear value play nor a pronounced growth name. Valuation multiples "
            "are broadly in line with the market average."
        )
    elif pc2 >= -1.5:
        return (
            f"on the valuation axis (PC2: {pc2:.2f}), the stock carries a growth-oriented "
            "premium. Lower book-to-market and earnings yield suggest the market is pricing "
            "in future growth expectations beyond current financials."
        )
    else:
        return (
            f"on the valuation axis (PC2: {pc2:.2f}), the stock is a pronounced growth name. "
            "The market is pricing significant future potential — current earnings, book value, "
            "and sales yield look low relative to price, which is typical of high-growth stories."
        )


def generate_summary(ticker: str, pca_row: Any) -> str:
    """Section 1: Overall position summary."""
    pc1      = float(pca_row.get('PC1', 0) if isinstance(pca_row, dict) else pca_row['PC1'])
    pc2      = float(pca_row.get('PC2', 0) if isinstance(pca_row, dict) else pca_row['PC2'])
    if pc1 >= 0 and pc2 >= 0:
        quadrant = 'Q1'
    elif pc1 < 0 and pc2 >= 0:
        quadrant = 'Q2'
    elif pc1 < 0 and pc2 < 0:
        quadrant = 'Q3'
    else:
        quadrant = 'Q4'

    meta = _QUADRANT_META.get(quadrant, _QUADRANT_META['Q3'])

    lines = [
        f"**{ticker}** sits in **{quadrant} — {meta['name']}**, reflecting {meta['summary']}.",
        meta['detail'],
        "",
        "**Profitability & Quality (PC1):** " + _describe_pc1(pc1),
        "",
        "**Valuation Style (PC2):** " + _describe_pc2(pc2),
    ]
    return "\n".join(lines)


# ===========================================================================
# SECTION 2 — TOP 3 STRENGTHS & WEAKNESSES
# ===========================================================================

def generate_factor_highlights(
    ticker: str,
    percentiles: Dict[str, float],
    factor_data: Optional[Dict] = None,
) -> str:
    """Section 2: Top 3 factor strengths and weaknesses by percentile rank."""
    if not percentiles:
        return f"Percentile data is not available for {ticker}."

    raw_vals: Dict[str, float] = {}
    if factor_data:
        for _, feats in factor_data.items():
            for feat, val in feats.items():
                raw_vals[feat] = val

    sorted_pcts = sorted(percentiles.items(), key=lambda x: x[1], reverse=True)
    strengths   = sorted_pcts[:3]
    weaknesses  = sorted_pcts[-3:][::-1]

    def _row(feat: str, pct: float) -> str:
        label = FEATURE_DISPLAY_NAMES.get(feat, feat)
        raw   = raw_vals.get(feat)
        raw_str = f" (raw: {raw:.3f})" if raw is not None else ""
        bar_filled = int(pct / 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)
        return f"- **{label}**: {pct:.0f}th percentile  `{bar}`{raw_str}"

    top_feat = FEATURE_DISPLAY_NAMES.get(strengths[0][0], strengths[0][0])
    bot_feat = FEATURE_DISPLAY_NAMES.get(weaknesses[0][0], weaknesses[0][0])
    top_pct  = strengths[0][1]
    bot_pct  = weaknesses[0][1]

    lines = [
        "**🟢 Top 3 Strengths** *(highest percentile ranks vs. quadrant peers)*",
        *[_row(f, p) for f, p in strengths],
        "",
        "**🔴 Top 3 Weaknesses** *(lowest percentile ranks vs. quadrant peers)*",
        *[_row(f, p) for f, p in weaknesses],
        "",
        f"*{ticker}'s clearest edge is **{top_feat}** ({top_pct:.0f}th percentile). "
        f"Its most notable drag is **{bot_feat}** ({bot_pct:.0f}th percentile).*",
    ]
    return "\n".join(lines)


# ===========================================================================
# SECTION 3 — TIME TRAJECTORY NARRATIVE
# ===========================================================================

def _classify_direction(start: float, end: float, std: float) -> str:
    delta = end - start
    if std == 0:
        return "flat"
    magnitude = abs(delta) / std
    if magnitude < 0.3:
        return "flat"
    elif delta > 0:
        return "rising" if magnitude < 1.0 else "sharply rising"
    else:
        return "declining" if magnitude < 1.0 else "sharply declining"


def _detect_steps(series: pd.Series, threshold_std: float = 1.2) -> int:
    """Count discrete step-changes (e.g., cash-to-debt financing jumps)."""
    if len(series) < 4:
        return 0
    diffs = series.diff().dropna()
    rolling_std = diffs.rolling(window=4, min_periods=2).std()
    return int((diffs.abs() > threshold_std * rolling_std).sum())


def generate_trajectory_narrative(
    ticker: str,
    raw_data: pd.DataFrame,
    loadings: Optional[Dict] = None,
) -> str:
    """Section 3: Time trajectory narrative from raw time-series data."""
    ticker_upper = ticker.upper()
    tick_col = 'ticker' if 'ticker' in raw_data.columns else 'TICKER'
    if tick_col not in raw_data.columns:
        return f"Ticker column not found — trajectory analysis unavailable."

    stock_df = raw_data[raw_data[tick_col].str.upper() == ticker_upper].copy()
    if stock_df.empty:
        return f"No time-series data found for {ticker}."

    date_col = next((c for c in ['public_date', 'date', 'datadate'] if c in stock_df.columns), None)
    if date_col is None:
        return "Date column not found — trajectory analysis unavailable."

    stock_df['_date'] = pd.to_datetime(stock_df[date_col])
    stock_df = stock_df.sort_values('_date').reset_index(drop=True)

    date_min  = stock_df['_date'].min().strftime('%b %Y')
    date_max  = stock_df['_date'].max().strftime('%b %Y')
    n_periods = len(stock_df)

    # Default driver features; override from loadings if available
    pc1_features = ['roa', 'roe', 'cash_debt']
    pc2_features = ['sales_to_price', 'bm', 'earnings_yield']

    if loadings:
        if 'PC1' in loadings and 'positive' in loadings['PC1']:
            pc1_features = list(loadings['PC1']['positive'].keys())[:3]
        if 'PC2' in loadings and 'positive' in loadings['PC2']:
            pc2_features = list(loadings['PC2']['positive'].keys())[:3]

    def _trend_parts(features):
        parts = []
        for feat in features:
            if feat not in stock_df.columns:
                continue
            series = stock_df[feat].dropna()
            if len(series) < 3:
                continue
            label = FEATURE_DISPLAY_NAMES.get(feat, feat)

            # Special handling: cash-to-debt stepped pattern
            if feat == 'cash_debt':
                steps = _detect_steps(series)
                if steps >= 2:
                    parts.append(
                        f"**{label}** shows {steps} discrete step-changes — "
                        f"consistent with specific financing events (debt issuances, "
                        f"equity raises, or acquisitions) rather than gradual operational drift"
                    )
                    continue

            direction = _classify_direction(series.iloc[0], series.iloc[-1], series.std())
            if direction != "flat":
                parts.append(f"**{label}** has been {direction}")
        return parts

    pc1_trends = _trend_parts(pc1_features)
    pc2_trends = _trend_parts(pc2_features)

    lines = [
        f"**Time Range:** {date_min} → {date_max}  ({n_periods} data points)",
        "",
        "**Quality & Profitability Trajectory (PC1 drivers):** " + (
            "; ".join(pc1_trends) + "."
            if pc1_trends else
            "Factor levels have been broadly stable — no significant directional "
            "shifts detected in ROA, ROE, or Cash-to-Debt."
        ),
        "",
        "**Valuation Style Trajectory (PC2 drivers):** " + (
            "; ".join(pc2_trends) + "."
            if pc2_trends else
            "Valuation multiples (Sales-to-Price, Book-to-Market, Earnings Yield) "
            "have remained relatively stable over the observation window."
        ),
        "",
        "*Note: Stepped changes in Cash-to-Debt reflect discrete corporate financing "
        "decisions rather than gradual operational changes — a pattern observed "
        "consistently across stocks in this dataset.*",
    ]
    return "\n".join(lines)


# ===========================================================================
# SECTION 4 — PEER CONTEXT
# ===========================================================================

def generate_peer_context(
    ticker: str,
    pca_row: Any,
    peer_df: pd.DataFrame,
    percentiles: Dict[str, float],
) -> str:
    """Section 4: Peer context — how the stock compares within its quadrant."""
    if peer_df is None or peer_df.empty:
        return f"Peer data is not available for {ticker}."

    n_peers  = len(peer_df)
    pc1      = float(pca_row.get('PC1', 0) if isinstance(pca_row, dict) else pca_row['PC1'])
    pc2      = float(pca_row.get('PC2', 0) if isinstance(pca_row, dict) else pca_row['PC2'])
    pc1_val  = pc1
    pc2_val  = pc2
    if pc1_val >= 0 and pc2_val >= 0:
        quadrant = 'Q1'
    elif pc1_val < 0 and pc2_val >= 0:
        quadrant = 'Q2'
    elif pc1_val < 0 and pc2_val < 0:
        quadrant = 'Q3'
    else:
        quadrant = 'Q4'
    meta     = _QUADRANT_META.get(quadrant, _QUADRANT_META['Q3'])

    pc1_pct_rank = ((peer_df['PC1'] < pc1).sum() / n_peers * 100) if 'PC1' in peer_df.columns else None
    pc2_pct_rank = ((peer_df['PC2'] < pc2).sum() / n_peers * 100) if 'PC2' in peer_df.columns else None

    # Find most differentiated factors (furthest from 50th percentile)
    standouts = []
    for feat, pct in sorted(percentiles.items(), key=lambda x: abs(x[1] - 50), reverse=True)[:3]:
        label = FEATURE_DISPLAY_NAMES.get(feat, feat)
        if pct >= 80:
            standouts.append(f"**{label}** ranks in the top {100 - pct:.0f}% of the group ({pct:.0f}th percentile)")
        elif pct <= 20:
            standouts.append(f"**{label}** ranks in the bottom {pct:.0f}% of the group ({pct:.0f}th percentile)")

    lines = [
        f"**{ticker}** is one of **{n_peers} stocks** in the **{quadrant} — {meta['name']}** quadrant.",
        "",
    ]

    if pc1_pct_rank is not None:
        quality_pos = (
            "near the top" if pc1_pct_rank >= 70 else
            "above average" if pc1_pct_rank >= 55 else
            "near the middle" if pc1_pct_rank >= 40 else
            "below average" if pc1_pct_rank >= 25 else
            "near the bottom"
        )
        lines.append(
            f"On **operational quality (PC1)**, it ranks {quality_pos} of its quadrant peers — "
            f"better than approximately {pc1_pct_rank:.0f}% of the group."
        )

    if pc2_pct_rank is not None:
        value_pos = (
            "among the most value-oriented" if pc2_pct_rank >= 70 else
            "slightly value-leaning" if pc2_pct_rank >= 55 else
            "near the blend point" if pc2_pct_rank >= 40 else
            "slightly growth-leaning" if pc2_pct_rank >= 25 else
            "among the most growth-oriented"
        )
        lines.append(
            f"On **valuation style (PC2)**, it is {value_pos} within the group."
        )

    if standouts:
        lines += ["", "**Notable factor differentiators vs. quadrant peers:**"]
        lines += [f"- {s}" for s in standouts]

    return "\n".join(lines)


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def generate_narrative(
    ticker: str,
    pca_row: Any,
    percentiles: Dict[str, float],
    factor_data: Optional[Dict] = None,
    peer_df: Optional[pd.DataFrame] = None,
    raw_data: Optional[pd.DataFrame] = None,
    loadings: Optional[Dict] = None,
) -> Dict[str, str]:
    """
    Generate all four narrative sections for a selected stock.

    Args:
        ticker:      Stock ticker symbol (e.g. 'GE').
        pca_row:     Single-row Series or dict from pca_df for this stock.
                     Must contain keys: PC1, PC2, quadrant, cluster.
        percentiles: Dict {feature_code: percentile_value} vs. quadrant peers.
        factor_data: Dict {category: {feature: raw_value}} for factor highlights.
        peer_df:     DataFrame of all stocks in the same quadrant (peer context).
        raw_data:    Full time-series DataFrame — all tickers/dates (trajectory).
        loadings:    st.session_state.pca_loadings (for trajectory driver labels).

    Returns:
        Dict with keys:
            'summary'    -> Section 1 markdown string
            'factors'    -> Section 2 markdown string
            'trajectory' -> Section 3 markdown string
            'peers'      -> Section 4 markdown string
    """
    return {
        'summary':    generate_summary(ticker, pca_row),
        'factors':    generate_factor_highlights(ticker, percentiles, factor_data),
        'trajectory': (
            generate_trajectory_narrative(ticker, raw_data, loadings)
            if raw_data is not None
            else "Time-series data was not passed to the narrative engine."
        ),
        'peers': (
            generate_peer_context(ticker, pca_row, peer_df, percentiles)
            if peer_df is not None
            else "Peer data was not passed to the narrative engine."
        ),
    }
