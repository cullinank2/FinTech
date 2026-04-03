"""
narrative_engine.py
-------------------
Rule-based narrative interpretation engine for Stock PCA Cluster Analysis.
Generates plain-English analysis from PCA session state data — no API key required.

Produces five analysis sections for any selected stock:
  1. Overall Position Summary  (quadrant + PC1/PC2 plain-English)
  2. Top 3 Factor Strengths & Weaknesses
  3. Time Trajectory Narrative  (from raw_data time series)
  4. Peer Context  (how the stock compares within its quadrant)
  5. Structural Context  (KG regime chain — Tier 1 governance artifact)
     Populated only when a KnowledgeGraph instance and current_regime are passed.
     When absent, section returns a graceful placeholder — no errors, no crashes.

Usage in app.py:
    from narrative_engine import generate_narrative
    sections = generate_narrative(
        ticker         = ticker,
        pca_row        = pca_row,
        percentiles    = percentiles,
        factor_data    = factor_data,
        peer_df        = peer_df,
        raw_data       = raw_data,
        loadings       = loadings,
        kg             = kg_instance,       # optional KnowledgeGraph
        current_regime = "Disinflation",    # optional str
    )
    # Returns dict with keys: 'summary', 'factors', 'trajectory', 'peers', 'structural'

Tier 1 governance note:
    Section 5 is deterministic. It calls kg.query_crowding_chain() and
    kg.get_structural_drift_summary() — pure graph traversal, no LLM.
    Its output is a reportable governance artifact identical in status to
    Sections 1–4. The Tier 2 chatbot is a separate system (chatbot.py).
"""

import pandas as pd
from typing import Dict, Optional, Any

from factor_registry import (
    FEATURE_DISPLAY_NAMES,
    FEATURE_ECONOMIC_MEANINGS,
    PCA_DRIVER_GROUPS,
    REGIME_ORDER,
)

# ---------------------------------------------------------------------------
# Feature display name mapping — mirrors config.py / visualizations.py
# ---------------------------------------------------------------------------

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

# Canonical regime ordering (use registry directly — no local alias)


# ===========================================================================
# SECTION 1 — OVERALL POSITION SUMMARY
# ===========================================================================

def _describe_pc1(pc1: float) -> str:
    """Plain-English description of PC1 score (Profitability & Operational Quality)."""
    if pc1 >= 1.5:
        return (
            f"a PC1 score of {pc1:.2f} — well above average — signals exceptionally strong "
            f"profitability and operational quality. {', '.join([FEATURE_DISPLAY_NAMES.get(f, f) for f in PCA_DRIVER_GROUPS.get('PC1', [])])} ratios are "
            "all pointing to a high-quality business with durable financial health."
        )
    elif pc1 >= 0.5:
        return (
            f"a PC1 score of {pc1:.2f} indicates above-average operational quality. "
            f"Profitability metrics ({', '.join([FEATURE_DISPLAY_NAMES.get(f, f) for f in PCA_DRIVER_GROUPS.get('PC1', [])])}) are "
            "solid, placing this stock in favorable territory on the quality axis."
        )
    elif pc1 >= -0.5:
        return (
            f"a PC1 score of {pc1:.2f} reflects near-average operational quality — "
            "neither a standout on profitability nor a concern on financial health. "
            f"Performance on {', '.join([FEATURE_DISPLAY_NAMES.get(f, f) for f in PCA_DRIVER_GROUPS.get('PC1', [])])} is in line with the broader universe."
        )
    elif pc1 >= -1.5:
        return (
            f"a PC1 score of {pc1:.2f} indicates below-average operational quality. "
            f"Profitability and financial health ({', '.join([FEATURE_DISPLAY_NAMES.get(f, f) for f in PCA_DRIVER_GROUPS.get('PC1', [])])}) are weaker "
            "relative to peers, which may warrant closer scrutiny."
        )
    else:
        return (
            f"a PC1 score of {pc1:.2f} — well below average — reflects material weakness "
            "in operational quality. Low profitability and/or elevated financial risk "
            f"(captured by {', '.join([FEATURE_DISPLAY_NAMES.get(f, f) for f in PCA_DRIVER_GROUPS.get('PC1', [])])}) place this stock in the bottom tier."
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


def _describe_pc3(pc3: float) -> str:
    """Plain-English description of PC3 score (Leverage & Risk Profile)."""
    if pc3 >= 1.5:
        return (
            f"a PC3 score of {pc3:.2f} — well above average — indicates a heavily "
            "leveraged, asset-intensive balance sheet. Debt-to-assets is elevated and "
            "the stock screens as deep value on an asset basis, typical of capital-heavy industries."
        )
    elif pc3 >= 0.5:
        return (
            f"a PC3 score of {pc3:.2f} indicates above-average leverage and asset intensity. "
            "The balance sheet carries more debt relative to assets than typical peers, "
            "though not at an extreme level."
        )
    elif pc3 >= -0.5:
        return (
            f"a PC3 score of {pc3:.2f} reflects a near-average leverage and risk profile — "
            "debt levels and asset intensity are broadly in line with the wider stock universe."
        )
    elif pc3 >= -1.5:
        return (
            f"a PC3 score of {pc3:.2f} indicates a relatively asset-light, lower-leverage profile. "
            "The business is more capital efficient than average, with stronger profitability "
            "relative to book value."
        )
    else:
        return (
            f"a PC3 score of {pc3:.2f} — well below average — reflects a highly asset-light, "
            "capital-efficient business. Low leverage combined with strong profitability "
            "relative to assets places this stock at the growth end of the PC3 spectrum."
        )


def generate_summary(ticker: str, pca_row: Any, show_pc3: bool = False) -> str:
    """Section 1: Overall position summary."""
    pc1 = float(pca_row.get('PC1', 0) if isinstance(pca_row, dict) else pca_row['PC1'])
    pc2 = float(pca_row.get('PC2', 0) if isinstance(pca_row, dict) else pca_row['PC2'])
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

    if show_pc3:
        pc3 = float(pca_row.get('PC3', 0) if isinstance(pca_row, dict) else pca_row.get('PC3', 0))
        lines += ["", "**Leverage & Risk Profile (PC3):** " + _describe_pc3(pc3)]

    try:
        return "\n".join(lines)
    except Exception:
        return "Narrative generation error in factor highlights section."

# ---------------------------------------------------------------------------
# FACTOR-LEVEL REGIME INTERPRETATION
# ---------------------------------------------------------------------------

def _get_factor_regime_notes(kg, current_regime: str) -> dict:
    """
    Returns factor-specific warnings based on regime transitions.
    Focuses on instability, reversals, and crowding.
    """
    if kg is None or not current_regime:
        return {}

    try:
        drift = kg.get_structural_drift_summary(current_regime)

        regime_order = REGIME_ORDER
        idx = regime_order.index(current_regime) if current_regime in regime_order else -1
        prior = regime_order[idx - 1] if idx > 0 else None

        notes = []

        # --- Global instability ---
        if drift.get("early_warning_triggered"):
            notes.append(
                "⚠️ Factor relationships are currently unstable — rankings may shift rapidly."
            )

        # --- Crowding ---
        if drift.get("crowding_risk") == "High":
            notes.append(
                "🔴 High factor crowding detected — top-ranked signals may be crowded trades."
            )

        # --- Factor reversals ---
        if prior:
            chain = kg.query_crowding_chain(prior, current_regime)
            reversals = [
                r for r in chain.get("factor_rotations", [])
                if r.get("pc2_stability_class") == "reversed"
            ]

            if reversals:
                names = [r.get("display_name", r.get("factor")) for r in reversals[:3]]
                notes.append(
                    f"🔄 Key factor reversals detected: {', '.join(names)} — "
                    "interpret value/growth signals with caution."
                )

        return {"notes": notes}

    except Exception:
        return {}

# ===========================================================================
# SECTION 2 — TOP 3 STRENGTHS & WEAKNESSES
# ===========================================================================

def generate_factor_highlights(
    ticker: str,
    percentiles: Dict[str, float],
    factor_data: Optional[Dict] = None,
    structural_drivers: Optional[list] = None,
    kg: Any = None,
    current_regime: Optional[str] = None,
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

    def _ordinal(n: float) -> str:
        i = int(n)
        if 11 <= (i % 100) <= 13:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(i % 10, 'th')
        return f"{i}{suffix}"

    def _row(feat: str, pct: float) -> str:
        label     = FEATURE_DISPLAY_NAMES.get(feat, feat)
        raw       = raw_vals.get(feat)
        raw_str   = f" (raw: {raw:.3f})" if raw is not None else ""
        bar_filled = int(pct / 10)
        bar       = "█" * bar_filled + "░" * (10 - bar_filled)
        return f"- **{label}**: {_ordinal(pct)} percentile  `{bar}`{raw_str}"

    top_feat = FEATURE_DISPLAY_NAMES.get(strengths[0][0], strengths[0][0])
    bot_feat = FEATURE_DISPLAY_NAMES.get(weaknesses[0][0], weaknesses[0][0])
    top_pct  = strengths[0][1]
    bot_pct  = weaknesses[0][1]

    regime_notes = _get_factor_regime_notes(kg, current_regime)

    structural_driver_lines = []
    structural_driver_summary = ""
    if structural_drivers:
        structural_driver_lines = []

        summary_parts = []

        for driver in structural_drivers[:3]:
            factor_key = driver.get("factor_name")
            factor_name = driver.get("display_name") or factor_key or "Unknown Driver"

            economic_meaning = FEATURE_ECONOMIC_MEANINGS.get(factor_key, factor_name)
            direction = driver.get("direction", "").strip().lower()
            strength = driver.get("strength", "").strip().lower()

            descriptor_parts = [part for part in [strength, direction] if part]
            descriptor = " ".join(descriptor_parts).strip()

            if "positive" in direction:
                interpretation = "increasing exposure"
            elif "negative" in direction:
                interpretation = "introducing structural headwinds"
            else:
                interpretation = "shaping current positioning"

            if "strong" in strength:
                intensity = "a primary structural driver of positioning"
            elif "moderate" in strength:
                intensity = "a meaningful structural contributor"
            else:
                intensity = "a secondary structural influence"

            if descriptor:
                structural_driver_lines.append(
                    f"- **{factor_name}** ({economic_meaning}) is {intensity}, {interpretation}."
                )
            else:
                structural_driver_lines.append(
                    f"- **{factor_name}** ({economic_meaning}) is {intensity}, shaping current positioning."
                )

            summary_parts.append(f"**{factor_name}**")

        if len(summary_parts) == 1:
            structural_driver_summary = (
                f"**Structural Drivers:** Current positioning is being defined primarily by {summary_parts[0]}."
            )
        elif len(summary_parts) == 2:
            structural_driver_summary = (
                f"**Structural Drivers:** Current positioning is being defined primarily by {summary_parts[0]} and {summary_parts[1]}."
            )
        elif len(summary_parts) >= 3:
            structural_driver_summary = (
                f"**Structural Drivers:** Current positioning is being defined primarily by {summary_parts[0]}, "
                f"{summary_parts[1]}, and {summary_parts[2]}."
            )

    lines = [
        f"**Factor Positioning Summary:** {ticker}'s clearest edge is **{top_feat}** "
        f"({_ordinal(top_pct)} percentile), while its most notable drag is "
        f"**{bot_feat}** ({_ordinal(bot_pct)} percentile).",
        "",
        "**Key Strengths:**",
        *[_row(f, p) for f, p in strengths],
        "",
        "**Key Weaknesses:**",
        *[_row(f, p) for f, p in weaknesses],
    ]

# --- NEW: Cross-link strengths/weaknesses with structural positioning ---
    if strengths or weaknesses:
        try:
            strength_names = [
                FEATURE_DISPLAY_NAMES.get(f, f) for f, _ in strengths[:2]
            ]
            weakness_names = [
                FEATURE_DISPLAY_NAMES.get(f, f) for f, _ in weaknesses[:2]
            ]

            strength_text = ", ".join([f"**{s}**" for s in strength_names]) if strength_names else ""
            weakness_text = ", ".join([f"**{w}**" for w in weakness_names]) if weakness_names else ""

            if strength_text and weakness_text:
                linkage_sentence = (
                    f"*This positioning is supported by strength in {strength_text}, "
                    f"while pressure from {weakness_text} is acting as a counterbalance.*"
                )
            elif strength_text:
                linkage_sentence = (
                    f"*This positioning is supported primarily by strength in {strength_text}.*"
                )
            elif weakness_text:
                linkage_sentence = (
                    f"*This positioning faces pressure primarily from {weakness_text}.*"
                )
            else:
                linkage_sentence = ""

            if linkage_sentence:
                lines += ["", linkage_sentence]

        except Exception:
            pass

    if structural_driver_lines:
        lines += [
            "",
            structural_driver_summary,
            "",
            *structural_driver_lines,
        ]

    # Append regime-aware interpretation if present
    if regime_notes and regime_notes.get("notes"):
        lines += [
            "",
            "**⚠️ Regime Context (Factor Reliability):**",
            *[f"- {n}" for n in regime_notes["notes"]],
        ]

    return "\n".join(lines)


def generate_trajectory_narrative(
    ticker: str,
    raw_data: pd.DataFrame,
    loadings: Optional[Dict] = None,
    show_pc3: bool = False,
) -> str:
    """Section 3: Time trajectory narrative from raw time-series data."""
    ticker_upper = ticker.upper()
    tick_col     = 'ticker' if 'ticker' in raw_data.columns else 'TICKER'
    if tick_col not in raw_data.columns:
        return "Ticker column not found — trajectory analysis unavailable."

    stock_df = raw_data[raw_data[tick_col].str.upper() == ticker_upper].copy()
    if stock_df.empty:
        return f"No time-series data found for {ticker}."

    date_col = next(
        (c for c in ['public_date', 'date', 'datadate'] if c in stock_df.columns), None
    )
    if date_col is None:
        return "Date column not found — trajectory analysis unavailable."

    stock_df['_date'] = pd.to_datetime(stock_df[date_col])
    stock_df          = stock_df.sort_values('_date').reset_index(drop=True)

    date_min  = stock_df['_date'].min().strftime('%b %Y')
    date_max  = stock_df['_date'].max().strftime('%b %Y')
    n_periods = len(stock_df)

    # Registry-driven PCA driver groups (no hard-coded axis keys)
    pc1_features = list(PCA_DRIVER_GROUPS.get("PC1", []))
    pc2_features = list(PCA_DRIVER_GROUPS.get("PC2", []))
    pc3_features = list(PCA_DRIVER_GROUPS.get("PC3", []))
    
    if loadings:
        if 'PC1' in loadings and 'positive' in loadings['PC1']:
            pc1_features = list(loadings['PC1']['positive'].keys())[:3]
        if 'PC2' in loadings and 'positive' in loadings['PC2']:
            pc2_features = list(loadings['PC2']['positive'].keys())[:3]
        if 'PC3' in loadings and 'positive' in loadings['PC3']:
            pc3_features = list(loadings['PC3']['positive'].keys())[:3]

    def _trend_parts(features):
        parts = []
        for feat in features:
            if feat not in stock_df.columns:
                continue
            series = stock_df[feat].dropna()
            if len(series) < 3:
                continue
            label = FEATURE_DISPLAY_NAMES.get(feat, feat)
            if feat == 'cash_debt':
                diffs = series.diff().dropna()
                rolling_std = diffs.rolling(window=4, min_periods=2).std()

                step_flags = (diffs.abs() > 1.2 * rolling_std)
                steps = int(step_flags.sum()) if not step_flags.empty else 0

                if steps >= 2:
                    parts.append(
                        f"**{label}** shows {steps} discrete step-changes — "
                        "consistent with specific financing events (debt issuances, "
                        "equity raises, or acquisitions) rather than gradual operational drift"
                    )
                    continue
            start_val = series.iloc[0]
            end_val = series.iloc[-1]
            std_val = series.std()

            if std_val == 0:
                direction = "flat"
            else:
                delta = end_val - start_val
                magnitude = abs(delta) / std_val

                if magnitude < 0.3:
                    direction = "flat"
                elif delta > 0:
                    direction = "rising" if magnitude < 1.0 else "sharply rising"
                else:
                    direction = "declining" if magnitude < 1.0 else "sharply declining"

            if direction != "flat":
                parts.append(f"**{label}** has been {direction}")
        return parts

    pc1_trends = _trend_parts(pc1_features)
    pc2_trends = _trend_parts(pc2_features)
    pc3_trends = _trend_parts(pc3_features) if show_pc3 else []

    lines = [
        f"**Time Range:** {date_min} → {date_max}  ({n_periods} data points)",
        "",
        "**Quality & Profitability Trajectory (PC1 drivers):** " + (
            "; ".join(pc1_trends) + "."
            if pc1_trends else
            "Factor levels have been broadly stable — no significant directional "
            f"shifts detected in {', '.join([FEATURE_DISPLAY_NAMES.get(f, f) for f in PCA_DRIVER_GROUPS.get('PC1', [])])}."
        ),
        "",
        "**Valuation Style Trajectory (PC2 drivers):** " + (
            "; ".join(pc2_trends) + "."
            if pc2_trends else
            f"Valuation multiples ({', '.join([FEATURE_DISPLAY_NAMES.get(f, f) for f in PCA_DRIVER_GROUPS.get('PC2', [])])}) "
            "have remained relatively stable over the observation window."
        ),
    ]

    if show_pc3:
        lines += [
            "",
            "**Leverage & Risk Profile Trajectory (PC3 drivers):** " + (
                "; ".join(pc3_trends) + "."
                if pc3_trends else
                "Leverage and risk metrics (Debt-to-Assets, Volatility, Gross Profitability) "
                "have remained broadly stable over the observation window."
            ),
        ]

    lines += [
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
    gics_sector: str = 'N/A',
    kg: Any = None,
    current_regime: Optional[str] = None,
) -> str:
    """Section 4: Peer context — how the stock compares within its quadrant."""
    if peer_df is None or peer_df.empty:
        return f"Peer data is not available for {ticker}."

    n_peers = len(peer_df)
    pc1     = float(pca_row.get('PC1', 0) if isinstance(pca_row, dict) else pca_row['PC1'])
    pc2     = float(pca_row.get('PC2', 0) if isinstance(pca_row, dict) else pca_row['PC2'])

    if pc1 >= 0 and pc2 >= 0:
        quadrant = 'Q1'
    elif pc1 < 0 and pc2 >= 0:
        quadrant = 'Q2'
    elif pc1 < 0 and pc2 < 0:
        quadrant = 'Q3'
    else:
        quadrant = 'Q4'

    meta        = _QUADRANT_META.get(quadrant, _QUADRANT_META['Q3'])
    pc1_pct_rank = ((peer_df['PC1'] < pc1).sum() / n_peers * 100) if 'PC1' in peer_df.columns else None
    pc2_pct_rank = ((peer_df['PC2'] < pc2).sum() / n_peers * 100) if 'PC2' in peer_df.columns else None

    standouts = []
    for feat, pct in sorted(percentiles.items(), key=lambda x: abs(x[1] - 50), reverse=True)[:3]:
        label = FEATURE_DISPLAY_NAMES.get(feat, feat)

        def _ordinal(n: float) -> str:
            i = int(n)
            if 11 <= (i % 100) <= 13:
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(i % 10, 'th')
            return f"{i}{suffix}"

        if pct >= 80:
            standouts.append(
                f"**{label}** ranks in the top {100 - int(pct)}% of the group "
                f"({_ordinal(pct)} percentile)"
            )
        elif pct <= 20:
            standouts.append(
                f"**{label}** ranks in the bottom {int(pct)}% of the group "
                f"({_ordinal(pct)} percentile)"
            )

    lines = [
        f"**{ticker}** is one of **{n_peers} stocks** in the **{gics_sector}** sector "
        f"in **{quadrant} — {meta['name']}** quadrant.",
        "",
    ]

    # --- Peer dispersion metric (institutional diagnostic) ---
    if 'PC1' in peer_df.columns and 'PC2' in peer_df.columns:
        pc1_disp = peer_df['PC1'].std()
        pc2_disp = peer_df['PC2'].std()

        def _disp_label(val):
            if val < 0.5:
                return "low (crowded)"
            elif val < 1.0:
                return "moderate"
            else:
                return "high (well differentiated)"

        lines += [
            "",
            "**Peer Dispersion (Structural Spread):**",
            f"- PC1 dispersion: {pc1_disp:.2f} → {_disp_label(pc1_disp)}",
            f"- PC2 dispersion: {pc2_disp:.2f} → {_disp_label(pc2_disp)}",
        ]    

    if pc1_pct_rank is not None:
        quality_pos = (
            "near the top"    if pc1_pct_rank >= 70 else
            "above average"   if pc1_pct_rank >= 55 else
            "near the middle" if pc1_pct_rank >= 40 else
            "below average"   if pc1_pct_rank >= 25 else
            "near the bottom"
        )
        lines.append(
            f"On **operational quality (PC1)**, it ranks {quality_pos} of its quadrant peers — "
            f"better than approximately {pc1_pct_rank:.0f}% of the group."
        )

    if pc2_pct_rank is not None:
        value_pos = (
            "among the most value-oriented"   if pc2_pct_rank >= 70 else
            "slightly value-leaning"          if pc2_pct_rank >= 55 else
            "near the blend point"            if pc2_pct_rank >= 40 else
            "slightly growth-leaning"         if pc2_pct_rank >= 25 else
            "among the most growth-oriented"
        )
        lines.append(
            f"On **valuation style (PC2)**, it is {value_pos} within the group."
        )

    # --- Regime-aware peer context (KG overlay) ---
    if kg is not None and current_regime:
        try:
            drift = kg.get_structural_drift_summary(current_regime)

            if drift.get("crowding_risk") == "High":
                lines += [
                    "",
                    "⚠️ **Peer context warning:** High factor crowding in this regime — "
                    "cross-sectional differentiation may be compressed.",
                ]

            if drift.get("migration_pct", 0) > 30:
                lines += [
                   "",
                   "🔀 **Peer instability:** Elevated quadrant migration suggests peer group "
                   "composition is shifting meaningfully.",
                ]

        except Exception:
            pass

    if standouts:
        lines += ["", "**Notable factor differentiators vs. GICS sector peers:**"]
        lines += [f"- {s}" for s in standouts]

    return "\n".join(lines)


# ===========================================================================
# SECTION 5 — STRUCTURAL CONTEXT  (Tier 1 KG integration)
# ===========================================================================

def _severity_emoji(severity: str) -> str:
    return {"Major": "🔴", "Meaningful": "🟠", "Detectable": "🟡",
            "Negligible": "🟢"}.get(severity, "⚪")


def _crowding_emoji(risk: str) -> str:
    return {"High": "🔴", "Elevated": "🟡", "Normal": "🟢"}.get(risk, "⚪")


def _stability_emoji(stability: str) -> str:
    return {"reversed": "🔴", "rotated": "🟡", "stable": "🟢"}.get(stability, "⚪")


def generate_structural_context(
    ticker: str,
    current_regime: str,
    kg: Any,                        # KnowledgeGraph instance — typed as Any to avoid
    quadrant_history: Optional[list] = None,  # import cycle
    kg_peers: Optional[list] = None,
) -> str:
    """
    Section 5: Structural context from the Knowledge Graph.

    Tier 1 governance artifact — deterministic, auditable, no LLM.

    Produces three sub-sections:
      5a. Regime structural state (crowding, Procrustes from prior, early warning)
      5b. Regime transition chain (narrative_chain from query_crowding_chain)
      5c. Stock-specific KG context (quadrant history + KG structural peers)

    Parameters
    ----------
    ticker          : stock ticker
    current_regime  : "Post-COVID" | "Rate Shock" | "Disinflation"
    kg              : KnowledgeGraph instance from kg_interface.py
    quadrant_history: pre-fetched list from kg.get_quadrant_history(ticker)
                      — pass None to fetch inside this function
    kg_peers        : pre-fetched list from kg.get_peers(ticker, current_regime)
                      — pass None to fetch inside this function

    Returns
    -------
    Markdown string — Section 5 of the narrative report.
    """
    if kg is None:
        return (
            "**Structural Context (Section 5):** Knowledge Graph not available. "
            "Run Period Comparison to build the structural graph, then reload the stock card."
        )

    lines = ["## Structural Context (Knowledge Graph — Tier 1 Governance Output)", ""]

    # ── 5a: Regime structural state ───────────────────────────────────────────
    try:
        drift = kg.get_structural_drift_summary(current_regime)
        if "error" in drift:
            lines.append(f"*Regime node not found: {drift['error']}*")
        else:
            sev_e  = _severity_emoji(drift.get("severity_from_prior", ""))
            crd_e  = _crowding_emoji(drift.get("crowding_risk", "Normal"))
            ew_str = "⚠️ **TRIGGERED**" if drift.get("early_warning_triggered") else "✅ Not triggered"
            prior  = drift.get("prior_regime", "N/A") or "N/A"

            lines += [
                f"### 5a — Regime: {current_regime}",
                f"**Date range:** {drift.get('start_date', '')} → {drift.get('end_date', '')}",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Crowding Score | {crd_e} {drift.get('crowding_score', 'N/A')} "
                f"({drift.get('crowding_risk', 'N/A')}) |",
                f"| Procrustes vs. {prior} | "
                f"{sev_e} {float(drift.get('procrustes_from_prior', 0.0)):.3f} "
                f"({drift.get('severity_from_prior', 'N/A')}) |",
                f"| Quadrant Migration Rate | "
                f"{drift.get('migration_pct', 0):.1f}% "
                f"({drift.get('stocks_changed', 0):,} of "
                f"{drift.get('stocks_analyzed', 0):,} stocks) |",
                f"| Early Warning | {ew_str} |",
                "",
            ]
    except Exception as e:
        lines.append(f"*Regime structural state unavailable: {e}*\n")

    # ── 5b: Regime transition chain ───────────────────────────────────────────
    try:
        regime_idx  = REGIME_ORDER.index(current_regime) if current_regime in REGIME_ORDER else -1
        prior_regime = REGIME_ORDER[regime_idx - 1] if regime_idx > 0 else None

        if prior_regime:
            chain = kg.query_crowding_chain(prior_regime, current_regime)
            lines += [
                f"### 5b — Structural Transition: {prior_regime} → {current_regime}",
                "",
            ]

            for step in chain.get("narrative_chain", []):
                step_type = step.get("type", "")
                interp    = step.get("interpretation", "")

                if step_type == "regime_transition":
                    emoji = _severity_emoji(step.get("severity", ""))
                elif step_type == "factor_rotation":
                    emoji = "🔄"
                elif step_type == "crowding_shift":
                    emoji = _crowding_emoji(step.get("risk_after", "Normal"))
                elif step_type == "early_warning":
                    emoji = "⚠️" if step.get("triggered") else "✅"
                elif step_type == "quadrant_migration":
                    emoji = "🔀"
                else:
                    emoji = "•"

                lines.append(f"**Step {step.get('step', '?')}** {emoji}  {interp}")

            # Factor rotation summary
            rotations = chain.get("factor_rotations", [])
            # Use pc2_stability_class for display — overall stability_class may reflect
            # a PC3 reversal that is invisible in the PC2 display context.
            reversals = [r for r in rotations
                         if r.get("pc2_stability_class", r.get("stability_class")) == "reversed"]
            if reversals:
                lines += [
                    "",
                    f"**Factor sign reversals on PC2 ({len(reversals)} detected):**",
                ]
                for rev in reversals:
                    disp = rev.get("display_name", rev.get("factor", ""))
                    pc2f = rev.get("pc2_from", 0.0)
                    pc2t = rev.get("pc2_to", 0.0)
                    src  = " *(Appendix B)*" if rev.get("data_source") == "appendix_b_fallback" else ""
                    lines.append(
                        f"- **{disp}**: PC2 {pc2f:+.3f} → {pc2t:+.3f} 🔴 reversed{src}"
                    )
            lines.append("")
        else:
            lines += [
                "### 5b — Structural Transition",
                f"*{current_regime} is the first regime — no prior transition to compare.*",
                "",
            ]
    except Exception as e:
        lines.append(f"*Transition chain unavailable: {e}*\n")

    # ── 5c: Stock-specific KG context ────────────────────────────────────────
    lines.append(f"### 5c — {ticker}: Structural Position in {current_regime}")
    lines.append("")

    # Quadrant history
    try:
        history = quadrant_history if quadrant_history is not None else \
                  kg.get_quadrant_history(ticker)

        if history:
            lines.append("**Quadrant trajectory (chronological):**")
            lines.append("")
            prev_q = None
            for h in history:
                regime   = h.get("regime", "")
                q_id     = h.get("quadrant_id", "")
                q_name   = h.get("quadrant_name", "")
                pc1      = h.get("pc1", 0.0)
                pc2      = h.get("pc2", 0.0)
                migrated = (prev_q is not None and prev_q != q_id)
                flag     = " 🔀 *migrated*" if migrated else ""
                is_cur   = " ← **current**" if regime == current_regime else ""
                lines.append(
                    f"- **{regime}**: {q_id} — {q_name} "
                    f"(PC1: {pc1:+.3f}, PC2: {pc2:+.3f}){flag}{is_cur}"
                )
                prev_q = q_id
            lines.append("")
        else:
            lines.append(
                f"*No quadrant history found for {ticker} in the Knowledge Graph. "
                "Run Period Comparison with equity nodes enabled to populate.*\n"
            )
    except Exception as e:
        lines.append(f"*Quadrant history unavailable: {e}*\n")

    # KG structural peers
    try:
        peers = kg_peers if kg_peers is not None else \
                kg.get_peers(ticker, current_regime, max_results=5)

        if peers:
            lines += [
                f"**Closest structural peers in {current_regime} "
                f"(by PCA cluster proximity):**",
                "",
            ]
            for p in peers:
                sector = p.get("gics_sector", "Unknown")
                lines.append(
                    f"- **{p['ticker']}** — distance {p['distance']:.3f} | "
                    f"{p.get('quadrant', 'N/A')} | {sector}"
                )
            lines.append("")
        else:
            lines.append(
                f"*No KG structural peers found for {ticker} in {current_regime}. "
                "Equity nodes are populated when Period Comparison runs with "
                "'Live Equity Nodes' enabled in the Knowledge Graph tab.*\n"
            )
    except Exception as e:
        lines.append(f"*KG peer lookup unavailable: {e}*\n")

    # Governance footer
    lines += [
        "---",
        "*Section 5 is a Tier 1 governance output — deterministic KG path traversal, "
        "no language model involved. All values sourced from live pipeline outputs "
        "or Appendix B fallbacks where labeled.*",
    ]

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
    gics_sector: str = 'N/A',
    show_pc3: bool = False,
    structural_drivers: Optional[list] = None,
    # ── KG parameters (Tier 1 Section 5) ─────────────────────────────────────
    kg: Optional[Any] = None,
    current_regime: Optional[str] = None,
    kg_quadrant_history: Optional[list] = None,
    kg_peers: Optional[list] = None,
) -> Dict[str, str]:
    """
    Generate all five narrative sections for a selected stock.

    Args:
        ticker:               Stock ticker symbol (e.g. 'GE').
        pca_row:              Single-row Series or dict from pca_df for this stock.
                              Must contain keys: PC1, PC2, quadrant, cluster.
        percentiles:          Dict {feature_code: percentile_value} vs. quadrant peers.
        factor_data:          Dict {category: {feature: raw_value}} for factor highlights.
        peer_df:              DataFrame of all stocks in the same quadrant (peer context).
        raw_data:             Full time-series DataFrame — all tickers/dates (trajectory).
        loadings:             st.session_state.pca_loadings (for trajectory driver labels).
        gics_sector:          GICS sector label for peer context header.
        show_pc3:             Whether to include PC3 axis descriptions.
        structural_drivers:   Optional precomputed structural driver list from app.py.
                              Tier 1 deterministic input only; no LLM dependency.
        kg:                   KnowledgeGraph instance from kg_interface.py (Tier 1).
                              Pass None to skip Section 5 gracefully.
        current_regime:       Active regime string — "Post-COVID" | "Rate Shock" |
                              "Disinflation". Required when kg is provided.
        kg_quadrant_history:  Pre-fetched result of kg.get_quadrant_history(ticker).
                              Pass None to fetch inside generate_structural_context().
        kg_peers:             Pre-fetched result of kg.get_peers(ticker, regime).
                              Pass None to fetch inside generate_structural_context().

    Returns:
        Dict with keys:
            'summary'    -> Section 1 markdown string
            'factors'    -> Section 2 markdown string
            'trajectory' -> Section 3 markdown string
            'peers'      -> Section 4 markdown string
            'structural' -> Section 5 markdown string (KG governance output)
    """
    structural_section = generate_structural_context(
        ticker          = ticker,
        current_regime = current_regime,
        kg              = kg,
        quadrant_history = kg_quadrant_history,
        kg_peers        = kg_peers,
    ) if (kg is not None and current_regime) else (
        "**Structural Context (Section 5):** "
        "Pass `kg=` and `current_regime=` to `generate_narrative()` to enable "
        "this Tier 1 governance section."
    )

    return {
        'summary': generate_summary(ticker, pca_row, show_pc3),
        'factors': generate_factor_highlights(
    ticker=ticker,
    percentiles=percentiles,
    factor_data=factor_data,
    structural_drivers=structural_drivers,
    kg=kg,
    current_regime=current_regime,
),
        'trajectory': (
            generate_trajectory_narrative(ticker, raw_data, loadings, show_pc3)
            if raw_data is not None
            else "Time-series data was not passed to the narrative engine."
        ),
        'peers': (
            generate_peer_context(
            ticker,
            pca_row,
            peer_df,
            percentiles,
            gics_sector,
            kg=kg,
            current_regime=current_regime,
)
            if peer_df is not None
            else "Peer data was not passed to the narrative engine."
        ),
        'structural': structural_section,
    }
