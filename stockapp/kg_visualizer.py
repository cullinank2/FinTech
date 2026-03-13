"""
kg_visualizer.py  —  ESDS Knowledge Graph: Phase 3
====================================================
Renders the ESDS knowledge graph as an interactive Pyvis network
in a dedicated Streamlit tab alongside Cluster Overview and Period Comparison.

Default view: Static ontology — regimes, factors, mechanisms, quadrants,
and the structural relationships that define the ESDS architecture.
No ticker nodes are shown unless the user explicitly requests them.

Usage (from app.py):
    from kg_visualizer import render_kg_tab

Architecture:
    - Imports kg_schema (Phase 1) for node/edge type definitions
    - Imports kg_builder (Phase 2) for live graph construction
    - Pyvis renders to a temporary HTML file, served via st.components.v1.html
    - Node color and size encode layer / node type
    - Edge width and style encode relationship strength
    - Interactive controls: zoom, pan, hover tooltips, filter sidebar

Pyvis install:
    pip install pyvis  (add to requirements.txt)
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

try:
    from kg_schema import (
        NodeType, EdgeType,
        REGIME_NODES, FACTOR_NODES, QUADRANT_NODES,
        MECHANISM_NODES, EMPIRICAL_ANCHORS,
    )
    KG_SCHEMA_AVAILABLE = True
except Exception:
    KG_SCHEMA_AVAILABLE = False

try:
    from kg_builder import build_kg, KGResult
    KG_BUILDER_AVAILABLE = True
except Exception as _kg_builder_err:
    KG_BUILDER_AVAILABLE = False


# ── Visual design constants ───────────────────────────────────────────────────

# Node colors by layer (hex, no #)
LAYER_COLORS = {
    "regime":     "#00b4d8",   # teal-blue  — macro context
    "factor":     "#90e0ef",   # light teal — factor signals
    "quadrant":   "#f4a261",   # amber      — quadrant positions
    "mechanism":  "#e63946",   # red        — structural mechanisms
    "metric":     "#2ec4b6",   # green-teal — empirical metrics
    "platform":   "#6d6875",   # purple     — institutional layer
    "default":    "#adb5bd",   # grey       — fallback
}

# Node sizes by type
LAYER_SIZES = {
    "regime":     40,
    "factor":     24,
    "quadrant":   30,
    "mechanism":  36,
    "metric":     20,
    "platform":   32,
    "default":    18,
}

# Edge colors by relationship class
EDGE_COLORS = {
    "structural":   "#00b4d8",
    "temporal":     "#f4a261",
    "crowding":     "#e63946",
    "membership":   "#90e0ef",
    "governance":   "#6d6875",
    "default":      "#495057",
}

# ── Live diagnostics helpers ─────────────────────────────────────────────────

def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(str(value).replace("%", "").replace(",", "").strip())
    except Exception:
        return default


def _get_procrustes_map() -> dict:
    df = st.session_state.get("procrustes_results")
    if df is None or len(df) == 0:
        return {}

    out = {}
    for _, row in df.iterrows():
        a = row.get("Period A")
        b = row.get("Period B")
        if a and b:
            out[(a, b)] = row
    return out


def _lookup_procrustes(a: str, b: str):
    pro_map = _get_procrustes_map()
    return pro_map.get((a, b)) or pro_map.get((b, a))


def _get_migration_map() -> dict:
    df = st.session_state.get("migration_summary")
    if df is None or len(df) == 0:
        return {}

    out = {}
    for _, row in df.iterrows():
        key = row.get("Transition")
        if key:
            out[key] = row
    return out


def _get_crowding_map() -> dict:
    df = st.session_state.get("crowding_df") or st.session_state.get("crowding_results")
    if df is None or len(df) == 0:
        return {}

    out = {}
    for _, row in df.iterrows():
        period = row.get("period") or row.get("Period")
        if period:
            out[period] = row
    return out


def _crowding_score(period_name: str):
    crowding = _get_crowding_map()
    row = crowding.get(period_name)
    if row is None:
        return None
    return _safe_float(row.get("crowding_score"))


def _migration_row(transition: str):
    return _get_migration_map().get(transition)


# ── Node definitions for the static ontology ─────────────────────────────────

def _regime_nodes() -> list[dict]:
    pc_rs = _lookup_procrustes("Post-COVID", "Rate Shock")
    pc_d = _lookup_procrustes("Post-COVID", "Disinflation")
    rs_d = _lookup_procrustes("Rate Shock", "Disinflation")

    pc_rs_mig = _migration_row("Post-COVID → Rate Shock")
    rs_d_mig = _migration_row("Rate Shock → Disinflation")

    pc_crowd = _crowding_score("Post-COVID")
    rs_crowd = _crowding_score("Rate Shock")
    d_crowd = _crowding_score("Disinflation")

    pc_rs_text = (
        f"Procrustes vs Rate Shock: {pc_rs['Disparity']:.3f} "
        f"({int(pc_rs['Common Tickers']):,} common stocks)"
        if pc_rs is not None else "Procrustes vs Rate Shock: unavailable"
    )
    pc_d_text = (
        f"Procrustes vs Disinflation: {pc_d['Disparity']:.3f} "
        f"({int(pc_d['Common Tickers']):,} common stocks)"
        if pc_d is not None else "Procrustes vs Disinflation: unavailable"
    )
    rs_d_text = (
        f"Procrustes vs Disinflation: {rs_d['Disparity']:.3f} "
        f"({int(rs_d['Common Tickers']):,} common stocks)"
        if rs_d is not None else "Procrustes vs Disinflation: unavailable"
    )

    pc_mig_text = (
        f"Migration to Rate Shock: {pc_rs_mig['Migration Rate']} "
        f"({int(pc_rs_mig['Changed Quadrant']):,}/{int(pc_rs_mig['Stocks Analyzed']):,} stocks)"
        if pc_rs_mig is not None else "Migration to Rate Shock: unavailable"
    )
    rs_mig_text = (
        f"Migration to Disinflation: {rs_d_mig['Migration Rate']} "
        f"({int(rs_d_mig['Changed Quadrant']):,}/{int(rs_d_mig['Stocks Analyzed']):,} stocks)"
        if rs_d_mig is not None else "Migration to Disinflation: unavailable"
    )

    return [
        {
            "id": "regime_postcovid",
            "label": "Post-COVID\n(Mar 2021–Jun 2022)",
            "layer": "regime",
            "tooltip": (
                "Post-COVID regime\n"
                f"{pc_rs_text}\n"
                f"Crowding score: {pc_crowd:.1f}" if pc_crowd is not None else "Crowding score: unavailable"
            ) + "\n" + pc_mig_text,
        },
        {
            "id": "regime_rateshock",
            "label": "Rate Shock\n(Jul 2022–Sep 2023)",
            "layer": "regime",
            "tooltip": (
                "Rate Shock regime\n"
                f"{rs_d_text}\n"
                f"Crowding score: {rs_crowd:.1f}" if rs_crowd is not None else "Crowding score: unavailable"
            ) + "\n" + rs_mig_text,
        },
        {
            "id": "regime_disinflation",
            "label": "Disinflation\n(Oct 2023–Oct 2024)",
            "layer": "regime",
            "tooltip": (
                "Disinflation regime\n"
                f"{pc_d_text}\n"
                f"Crowding score: {d_crowd:.1f}" if d_crowd is not None else "Crowding score: unavailable"
            ),
        },
    ]


def _factor_nodes() -> list[dict]:
    factors = [
        ("factor_ey",  "Earnings\nYield",        "Fundamental: sign reversal in Disinflation (PC2 flip: +0.211 → −0.364)"),
        ("factor_btm", "Book-to-\nMarket",        "Fundamental: stable value anchor across regimes (PC2: +0.571 → +0.507 → +0.478)"),
        ("factor_sp",  "Sales-to-\nPrice",        "Fundamental: strongest PC2 driver all three regimes"),
        ("factor_roa", "ROA",                     "Profitability: primary PC1 signal"),
        ("factor_gp",  "Gross\nProfitability",    "Profitability: reverses sign on PC3 in Rate Shock / Disinflation"),
        ("factor_ctd", "Cash-to-\nDebt",          "Quality: top PC1 driver (0.387–0.427); near-zero on PC3 in later regimes"),
        ("factor_dta", "Debt-to-\nAssets",        "Leverage: dominant PC3 driver; flips sign Post-COVID → Rate Shock (−0.466 → +0.723)"),
        ("factor_mom", "12-Mo\nMomentum",         "Behavioral: included as structural complement"),
        ("factor_vol", "60-Day\nVolatility",      "Behavioral: dominant PC3 in Post-COVID (+0.572); attenuates in later regimes"),
        ("factor_liq", "Liquidity",               "Behavioral: complement signal"),
        ("factor_pe",  "P/E Ratio",               "Behavioral: complement signal"),
    ]
    return [
        {
            "id": fid,
            "label": label,
            "layer": "factor",
            "tooltip": tip,
        }
        for fid, label, tip in factors
    ]


def _quadrant_nodes() -> list[dict]:
    return [
        {
            "id": "q1",
            "label": "Q1\nValue Trap",
            "layer": "quadrant",
            "tooltip": "PC1 < 0, PC2 > 0 — low profitability, value-tilted",
        },
        {
            "id": "q2",
            "label": "Q2\nDistressed\nGrowth",
            "layer": "quadrant",
            "tooltip": "PC1 < 0, PC2 < 0 — low profitability, growth-tilted",
        },
        {
            "id": "q3",
            "label": "Q3\nQuality\nValue",
            "layer": "quadrant",
            "tooltip": "PC1 > 0, PC2 > 0 — high profitability, value-tilted",
        },
        {
            "id": "q4",
            "label": "Q4\nQuality\nGrowth",
            "layer": "quadrant",
            "tooltip": "PC1 > 0, PC2 < 0 — high profitability, growth-tilted (e.g. GE: PC1 0.412, PC2 −0.659)",
        },
    ]


def _structural_edges() -> list[dict]:
    edges = []

    pc_rs = _lookup_procrustes("Post-COVID", "Rate Shock")
    pc_d = _lookup_procrustes("Post-COVID", "Disinflation")
    rs_d = _lookup_procrustes("Rate Shock", "Disinflation")

    def transition_edge(src_id, tgt_id, row):
        if row is None:
            return {
                "from": src_id,
                "to": tgt_id,
                "label": "",
                "color": EDGE_COLORS["temporal"],
                "width": 1.5,
            }

        disparity = _safe_float(row.get("Disparity"), 0.0)
        common = int(row.get("Common Tickers", 0))
        return {
            "from": src_id,
            "to": tgt_id,
            "label": f"{disparity:.3f}",
            "title": f"{disparity:.3f} | {common:,} common stocks",
            "color": EDGE_COLORS["temporal"],
            "width": 1.5 + disparity * 6,
        }

    edges += [
        transition_edge("regime_postcovid", "regime_rateshock", pc_rs),
        transition_edge("regime_postcovid", "regime_disinflation", pc_d),
        transition_edge("regime_rateshock", "regime_disinflation", rs_d),
    ]

    edges += [
        {"from": "mech_pca", "to": "q1", "label": "", "color": EDGE_COLORS["structural"], "width": 1.5},
        {"from": "mech_pca", "to": "q2", "label": "", "color": EDGE_COLORS["structural"], "width": 1.5},
        {"from": "mech_pca", "to": "q3", "label": "", "color": EDGE_COLORS["structural"], "width": 1.5},
        {"from": "mech_pca", "to": "q4", "label": "", "color": EDGE_COLORS["structural"], "width": 1.5},
    ]

    pc1_drivers = ["factor_ctd", "factor_roa", "factor_ey"]
    pc2_drivers = ["factor_sp", "factor_btm", "factor_ey"]
    pc3_drivers = ["factor_dta", "factor_gp", "factor_vol"]

    for f in pc1_drivers:
        edges.append({"from": f, "to": "mech_pca", "label": "PC1", "color": EDGE_COLORS["structural"], "width": 2})
    for f in pc2_drivers:
        edges.append({"from": f, "to": "mech_pca", "label": "PC2", "color": EDGE_COLORS["membership"], "width": 2})
    for f in pc3_drivers:
        edges.append({"from": f, "to": "mech_pca", "label": "PC3", "color": EDGE_COLORS["governance"], "width": 2})

    edges += [
        {"from": "mech_procrustes", "to": "regime_postcovid", "label": "", "color": EDGE_COLORS["structural"], "width": 1.5},
        {"from": "mech_procrustes", "to": "regime_rateshock", "label": "", "color": EDGE_COLORS["structural"], "width": 1.5},
        {"from": "mech_procrustes", "to": "regime_disinflation", "label": "", "color": EDGE_COLORS["structural"], "width": 1.5},
    ]

    for period_id, period_name in [
        ("regime_postcovid", "Post-COVID"),
        ("regime_rateshock", "Rate Shock"),
        ("regime_disinflation", "Disinflation"),
    ]:
        score = _crowding_score(period_name)
        label = f"{score:.1f}" if score is not None else ""
        width = 1.5 + (score / 25 if score is not None else 0)
        edges.append({
            "from": "mech_crowding",
            "to": period_id,
            "label": label,
            "title": f"Crowding score: {score:.1f}" if score is not None else "Crowding unavailable",
            "color": EDGE_COLORS["crowding"],
            "width": width,
        })

    edges += [
        {"from": "plat_esds", "to": "mech_pca", "label": "", "color": EDGE_COLORS["governance"], "width": 2},
        {"from": "plat_esds", "to": "mech_procrustes", "label": "", "color": EDGE_COLORS["governance"], "width": 2},
        {"from": "plat_esds", "to": "mech_crowding", "label": "", "color": EDGE_COLORS["governance"], "width": 2},
        {"from": "plat_esds", "to": "mech_kmeans", "label": "", "color": EDGE_COLORS["governance"], "width": 2},
        {"from": "plat_esds", "to": "mech_quadrant", "label": "", "color": EDGE_COLORS["governance"], "width": 2},
        {"from": "plat_esds", "to": "plat_barra", "label": "complements", "color": EDGE_COLORS["governance"], "width": 2},
        {"from": "plat_esds", "to": "plat_narrative", "label": "Tier 1", "color": EDGE_COLORS["governance"], "width": 1.5},
        {"from": "plat_esds", "to": "plat_chatbot", "label": "Tier 2", "color": EDGE_COLORS["governance"], "width": 1.5},
    ]

    return edges


def _mechanism_nodes() -> list[dict]:
    return [
        {
            "id": "mech_pca",
            "label": "PCA\nStructural\nCoordinate System",
            "layer": "mechanism",
            "tooltip": "Principal component geometry defining factor space structure",
        },
        {
            "id": "mech_procrustes",
            "label": "Procrustes\nDisparity",
            "layer": "mechanism",
            "tooltip": "Rotation-invariant distance between factor spaces across regimes",
        },
        {
            "id": "mech_crowding",
            "label": "Geometric\nCrowding",
            "layer": "mechanism",
            "tooltip": "Spatial compression signal within PCA factor space",
        },
        {
            "id": "mech_kmeans",
            "label": "KMeans\nClustering",
            "layer": "mechanism",
            "tooltip": "Structural peer groups based on factor geometry",
        },
        {
            "id": "mech_quadrant",
            "label": "Quadrant\nClassification",
            "layer": "mechanism",
            "tooltip": "PC1 / PC2 quadrant classification of securities",
        },
    ]


def _platform_nodes() -> list[dict]:
    return [
        {
            "id": "plat_esds",
            "label": "ESDS\nStructural\nVisibility Layer",
            "layer": "platform",
            "tooltip": (
                "Equity Structural Diagnostics System\n"
                "Complement to Barra / Aladdin / Axioma — not a replacement\n"
                "~1,738 U.S. equities | Three macro regimes\n"
                "Two-tier AI: Narrative Engine (governance) + Chatbot (practitioner)"
            ),
        },
        {
            "id": "plat_barra",
            "label": "Barra / Aladdin\n/ Axioma",
            "layer": "platform",
            "tooltip": (
                "Incumbent institutional risk platforms\n"
                "Measure factor exposures; do not monitor structural character of factor space\n"
                "ESDS sits alongside — not competing"
            ),
        },
        {
            "id": "plat_narrative",
            "label": "Tier 1\nNarrative Engine\n(Deterministic)",
            "layer": "platform",
            "tooltip": (
                "Rules-based governance artifact — no API key required\n"
                "Versioned, auditable, CRO-level outputs\n"
                "Not a generative chatbot — structural interpretive translation"
            ),
        },
        {
            "id": "plat_chatbot",
            "label": "Tier 2\nAI Chatbot\n(Configurable)",
            "layer": "platform",
            "tooltip": (
                "Configurable practitioner tool — requires OpenAI API key\n"
                "Multi-turn conversational analysis\n"
                "Separated from governance layer by design"
            ),
        },
    ]


# ── Pyvis graph builder ───────────────────────────────────────────────────────

def _build_pyvis_network(height: str = "700px") -> Network:
    """Build and return a configured Pyvis Network for the static ontology."""
    net = Network(
        height=height,
        width="100%",
        bgcolor="#0d1117",
        font_color="#e6edf3",
        directed=True,
    )

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -120,
          "centralGravity": 0.005,
          "springLength": 180,
          "springConstant": 0.04,
          "damping": 0.5,
          "avoidOverlap": 0.5
        },
        "stabilization": {
          "enabled": true,
          "iterations": 250,
          "updateInterval": 25
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "navigationButtons": true,
        "keyboard": { "enabled": true }
      },
      "edges": {
        "smooth": { "enabled": true, "type": "dynamic" },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } },
        "font": { "size": 10, "color": "#adb5bd", "face": "Courier New" }
      },
      "nodes": {
        "font": { "size": 12, "color": "#e6edf3", "face": "Calibri" },
        "borderWidth": 2,
        "shadow": { "enabled": true, "size": 8, "x": 2, "y": 2 }
      }
    }
    """)

    # Add all node groups
    all_node_groups = [
        _regime_nodes(),
        _factor_nodes(),
        _quadrant_nodes(),
        _mechanism_nodes(),
        _platform_nodes(),
    ]

    for group in all_node_groups:
        for node in group:
            layer = node.get("layer", "default")
            color = LAYER_COLORS.get(layer, LAYER_COLORS["default"])
            size  = LAYER_SIZES.get(layer, LAYER_SIZES["default"])
            net.add_node(
                node["id"],
                label=node["label"],
                title=node.get("tooltip", node["label"]),
                color={
                    "background": color,
                    "border": "#ffffff",
                    "highlight": {"background": "#ffffff", "border": color},
                    "hover":     {"background": "#ffffff", "border": color},
                },
                size=size,
                shape="dot",
                mass=1.5,
            )

    # Add all edges
    for edge in _structural_edges():
        net.add_edge(
            edge["from"],
            edge["to"],
            label=edge.get("label", ""),
            color=edge.get("color", EDGE_COLORS["default"]),
            width=edge.get("width", 1.5),
            title=edge.get("title", edge.get("label", "")),
        )

    return net


def _render_pyvis_to_html(net: Network) -> str:
    """Return PyVis graph HTML directly."""
    return net.generate_html()


@st.cache_data()
def _build_equity_graph_html_from_pipeline() -> str:
    """
    Build equity-augmented KG by running PCA for each period directly.
    Mirrors the pattern used by the crowding module in app.py.
    Cached so it only runs once per session.
    """
    if not KG_BUILDER_AVAILABLE:
        return _build_cached_graph_html()

    try:
        from period_analysis import _run_pca_for_period
        from config import FEATURE_COLUMNS

        raw_data = st.session_state.get("raw_data")
        if raw_data is None or raw_data.empty:
            return _build_cached_graph_html()

        date_col = next(
            (c for c in ["public_date", "date", "datadate"] if c in raw_data.columns),
            None,
        )
        if date_col is None:
            return _build_cached_graph_html()

        features = [c for c in FEATURE_COLUMNS if c in raw_data.columns]

        period_label_map = {
            "Post-COVID":   ("2021-03-01", "2022-06-30"),
            "Rate Shock":   ("2022-07-01", "2023-09-30"),
            "Disinflation": ("2023-10-01", "2024-10-31"),
        }

        period_data = {}
        for period_name, (start, end) in period_label_map.items():
            mask = (raw_data[date_col] >= start) & (raw_data[date_col] <= end)
            period_slice = raw_data[mask]
            if len(period_slice) < 10:
                continue
            try:
                _, scores_df, _, _ = _run_pca_for_period(
                    period_slice, features, date_col, start, end
                )
                if scores_df is not None and not scores_df.empty:
                    # Normalize column names for kg_builder
                    if "Quadrant" in scores_df.columns and "quadrant" not in scores_df.columns:
                        scores_df = scores_df.copy()
                        scores_df["quadrant"] = scores_df["Quadrant"]
                    period_data[period_name] = scores_df
            except Exception:
                continue

        if not period_data:
            return _build_cached_graph_html()

        return _build_equity_graph_html(period_data, migration_df=None)

    except Exception as exc:
        st.warning(f"Live equity graph failed, showing static ontology: {exc}")
        return _build_cached_graph_html()


# ── Legend builder ────────────────────────────────────────────────────────────

def _render_legend() -> None:
    """Render a compact color legend below the graph."""
    st.markdown(
        """
        <div style="display:flex; flex-wrap:wrap; gap:16px; margin-top:8px; font-size:13px;">
          <span><span style="color:#00b4d8;">●</span> Macro Regime</span>
          <span><span style="color:#90e0ef;">●</span> Factor Signal</span>
          <span><span style="color:#f4a261;">●</span> Quadrant</span>
          <span><span style="color:#e63946;">●</span> Core Mechanism</span>
          <span><span style="color:#6d6875;">●</span> Platform / Governance</span>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:16px; margin-top:6px; font-size:12px; color:#6c757d;">
          <span>Edge width = relationship strength &nbsp;|&nbsp;
          Hover nodes for empirical detail &nbsp;|&nbsp;
          Scroll to zoom &nbsp;|&nbsp; Drag to pan</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Key metrics panel ─────────────────────────────────────────────────────────

def _render_metrics_panel() -> None:
    """Three-column empirical anchors strip above the graph."""
    col1, col2, col3 = st.columns(3)

    procrustes = st.session_state.get("procrustes_results")
    migration_summary = st.session_state.get("migration_summary")
    migration_pct = st.session_state.get("migration_pct")

    with col1:
        if procrustes is not None and len(procrustes) >= 3:
            row1 = procrustes.iloc[0]
            row2 = procrustes.iloc[1]
            row3 = procrustes.iloc[2]

            st.metric("Procrustes — PC→RS", f"{row1['Disparity']:.3f}")
            st.caption(f"{int(row1['Common Tickers']):,} overlapping stocks")

            st.metric("Procrustes — PC→D", f"{row2['Disparity']:.3f}")
            st.caption(f"{int(row2['Common Tickers']):,} overlapping stocks")

            st.metric("Procrustes — RS→D", f"{row3['Disparity']:.3f}")
            st.caption(f"{int(row3['Common Tickers']):,} overlapping stocks")
        else:
            st.info("Run Period Comparison to populate structural diagnostics.")

    with col2:
        pc_crowd = _crowding_score("Post-COVID")
        rs_crowd = _crowding_score("Rate Shock")
        d_crowd = _crowding_score("Disinflation")

        if pc_crowd is not None:
            st.metric("Crowding — Post-COVID", f"{pc_crowd:.1f}")
            st.caption("Live crowding score")
        else:
            st.metric("Crowding — Post-COVID", "N/A")

        if rs_crowd is not None:
            st.metric("Crowding — Rate Shock", f"{rs_crowd:.1f}")
            st.caption("Live crowding score")
        else:
            st.metric("Crowding — Rate Shock", "N/A")

        if d_crowd is not None:
            st.metric("Crowding — Disinflation", f"{d_crowd:.1f}")
            st.caption("Live crowding score")
        else:
            st.metric("Crowding — Disinflation", "N/A")

    with col3:
        pc_rs = _migration_row("Post-COVID → Rate Shock")
        rs_d = _migration_row("Rate Shock → Disinflation")

        if pc_rs is not None:
            st.metric("Migration PC→RS", str(pc_rs["Migration Rate"]))
            st.caption(
                f"{int(pc_rs['Changed Quadrant']):,} / {int(pc_rs['Stocks Analyzed']):,} stocks"
            )
        else:
            st.metric("Migration PC→RS", "N/A")

        if rs_d is not None:
            st.metric("Migration RS→D", str(rs_d["Migration Rate"]))
            st.caption(
                f"{int(rs_d['Changed Quadrant']):,} / {int(rs_d['Stocks Analyzed']):,} stocks"
            )
        else:
            st.metric("Migration RS→D", "N/A")

        if migration_pct is not None and migration_summary is not None and len(migration_summary) > 0:
            tracked = int(migration_summary.iloc[0]["Stocks Analyzed"])
            changed_any = int(round(tracked * float(migration_pct) / 100.0))
            st.metric("Changed quadrant", f"{migration_pct:.1f}%")
            st.caption(f"{changed_any:,} / {tracked:,} — governance scale")
        else:
            st.metric("Changed quadrant", "N/A")


# ── Filter sidebar ────────────────────────────────────────────────────────────

def _render_kg_filters() -> dict:
    """
    Sidebar controls for the KG tab.
    Returns a dict of filter state that future phases can use
    to add ticker ego-networks or layer toggling.
    """
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🧠 Knowledge Graph")

        show_factors    = st.checkbox("Show Factor Nodes",    value=True)
        show_quadrants  = st.checkbox("Show Quadrant Nodes",  value=True)
        show_mechanisms = st.checkbox("Show Mechanism Nodes", value=True)
        show_platforms  = st.checkbox("Show Platform Nodes",  value=True)

        st.markdown("---")

    return {
        "show_factors":    show_factors,
        "show_quadrants":  show_quadrants,
        "show_mechanisms": show_mechanisms,
        "show_platforms":  show_platforms,
    }


@st.cache_data()
def _build_cached_graph_html() -> str:
    """Build the static PyVis ontology graph once and cache the HTML."""
    net = _build_pyvis_network(height="680px")
    return _render_pyvis_to_html(net)


def _build_equity_graph_html(period_data: dict, migration_df) -> str:
    """
    Build a PyVis graph that includes live equity nodes from the pipeline.
    Called only when session state has loaded pipeline data.
    Not cached — equity data is session-specific.
    """
    if not KG_BUILDER_AVAILABLE:
        return _build_cached_graph_html()

    try:
        kg_result = build_kg(
            period_data=period_data,
            migration_df=migration_df,
            include_equity_nodes=True,
        )
        G = kg_result.graph

        net = Network(
            height="680px",
            width="100%",
            bgcolor="#0d1117",
            font_color="#e6edf3",
            directed=True,
        )
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -120,
              "centralGravity": 0.005,
              "springLength": 180,
              "springConstant": 0.04,
              "damping": 0.5,
              "avoidOverlap": 0.5
            },
            "stabilization": { "enabled": true, "iterations": 250 }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "navigationButtons": true,
            "keyboard": { "enabled": true }
          },
          "edges": {
            "smooth": { "enabled": true, "type": "dynamic" },
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } },
            "font": { "size": 10, "color": "#adb5bd", "face": "Courier New" }
          },
          "nodes": {
            "font": { "size": 12, "color": "#e6edf3", "face": "Calibri" },
            "borderWidth": 2,
            "shadow": { "enabled": true, "size": 8, "x": 2, "y": 2 }
          }
        }
        """)

        # Node type → visual properties
        node_type_style = {
            "regime":           ("#00b4d8", 40),
            "factor":           ("#90e0ef", 24),
            "quadrant":         ("#f4a261", 30),
            "axis":             ("#2ec4b6", 22),
            "category":         ("#48cae4", 20),
            "crowding":         ("#e63946", 28),
            "transition":       ("#f4a261", 26),
            "structural_break": ("#e63946", 34),
            "early_warning":    ("#e63946", 32),
            "stock":            ("#b7e4c7", 10),
            "cluster":          ("#52b788", 18),
            "default":          ("#adb5bd", 16),
        }

        for node_id, attrs in G.nodes(data=True):
            ntype = attrs.get("node_type", "default")
            color, size = node_type_style.get(ntype, node_type_style["default"])
            label = attrs.get("label", str(node_id))
            # Truncate long stock labels
            if ntype == "stock" and len(label) > 6:
                label = label[:6]
            tooltip = "\n".join(
                f"{k}: {v}" for k, v in attrs.items()
                if k not in ("label", "node_type") and v is not None
            )
            net.add_node(
                str(node_id),
                label=label,
                title=tooltip or label,
                color={
                    "background": color,
                    "border": "#ffffff",
                    "highlight": {"background": "#ffffff", "border": color},
                    "hover":     {"background": "#ffffff", "border": color},
                },
                size=size,
                shape="dot",
                mass=1.5,
            )

        for src, tgt, attrs in G.edges(data=True):
            etype = attrs.get("edge_type", "")
            edge_color_map = {
                "regime_transition":    "#f4a261",
                "crowding_level":       "#e63946",
                "factor_loading":       "#90e0ef",
                "belongs_to_category":  "#48cae4",
                "triggers_break":       "#e63946",
                "triggers_warning":     "#e63946",
                "quadrant_assignment":  "#b7e4c7",
                "cluster_membership":   "#52b788",
                "belongs_to":           "#52b788",
                "migrates_to":          "#f4a261",
            }
            ecolor = edge_color_map.get(etype, "#495057")
            width  = attrs.get("width", 1.0)
            if etype == "regime_transition":
                d = attrs.get("procrustes_disparity", 0)
                width = 1.5 + d * 6
            elif etype == "crowding_level":
                width = 1.5 + attrs.get("score", 0) / 25
            net.add_edge(
                str(src), str(tgt),
                label=attrs.get("label", ""),
                color=ecolor,
                width=width,
                title=attrs.get("label", etype),
            )

        return net.generate_html()

    except Exception as exc:
        # Fallback to static graph on any error
        import streamlit as _st
        _st.warning(f"Equity graph build failed, showing static ontology: {exc}")
        return _build_cached_graph_html()


# ── Main entry point ──────────────────────────────────────────────────────────

def render_kg_tab() -> None:
    """
    Main entry point called from app.py inside the Knowledge Graph tab.
    Renders the full interactive Pyvis ontology with metrics and legend.
    """
    st.subheader("🧠 ESDS Knowledge Graph — Structural Ontology")
    st.caption(
        "Interactive map of the ESDS framework: five core mechanisms, "
        "eleven factor signals, three macro regimes, four quadrant positions, "
        "and the institutional governance layer. "
        "Hover any node for empirical detail. "
        "Edge width reflects relationship strength."
    )

    # ── Dependency check ──────────────────────────────────────────────────────
    if not PYVIS_AVAILABLE:
        st.error(
            "**Pyvis not installed.** Add `pyvis` to `requirements.txt` and redeploy.\n\n"
            "```\npip install pyvis\n```"
        )
        return

    # ── Empirical anchors strip ───────────────────────────────────────────────
    with st.expander("📊 Empirical Anchors", expanded=False):
        _render_metrics_panel()

    st.markdown("---")

    # ── Graph ─────────────────────────────────────────────────────────────────
    view_mode = st.radio(
        "Graph view",
        ["Static Ontology", "Live Equity Nodes"],
        horizontal=True,
        help="Static Ontology: framework architecture only. Live Equity Nodes: adds ~1,738 tickers, clusters, and quadrant assignments from the pipeline.",
    )

    with st.spinner("Building structural knowledge graph…"):
        try:
            if view_mode == "Live Equity Nodes" and KG_BUILDER_AVAILABLE:
                html = _build_equity_graph_html_from_pipeline()
            else:
                html = _build_cached_graph_html()
            components.html(html, height=700, scrolling=False)
        except Exception as exc:
            st.error(f"Knowledge graph render error: {exc}")
            st.info(
                "If this is a first-run import error, confirm that "
                "`kg_schema.py` and `kg_builder.py` are in the same "
                "directory as `app.py`."
            )

    _render_legend()

    # ── Interpretation notes ──────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📖 How to read this graph"):
        st.markdown(
            """
**Node layers**
- 🔵 **Teal — Macro Regimes**: Post-COVID, Rate Shock, Disinflation. The three structural windows that define the ESDS analysis universe.
- 🩵 **Light teal — Factor Signals**: The 11 retained signals across Fundamental, Profitability, Quality, Leverage, and Behavioral domains.
- 🟠 **Amber — Quadrants**: The four structural positions on the PC1/PC2 plane (Quality Growth, Quality Value, Value Trap, Distressed Growth).
- 🔴 **Red — Core Mechanisms**: The five proprietary methods — PCA as structural geometry, KMeans clustering, Procrustes disparity, geometric crowding, quadrant classification.
- 🟣 **Purple — Governance / Platform**: ESDS as a whole, its two-tier AI architecture, and the incumbent platforms it complements.

**Edge encoding**
- **Edge width** encodes relationship strength. The thick red edge from Crowding to Disinflation (67.9, Elevated) is intentionally prominent.
- **Procrustes edges** between regimes encode structural distance — the 0.459 Post-COVID → Disinflation edge is thicker than the 0.186 Rate Shock → Disinflation edge, reflecting greater structural discontinuity.
- **PC1/PC2/PC3 labels** on factor → PCA edges identify which principal component each factor primarily drives.

**Zero prior art nodes** (hover for detail)
- Procrustes Disparity — no prior application to equity factor spaces
- Geometric Crowding — spatial PCA compression, not correlation-based
- PCA as Governance Diagnostic — structural framing without precedent
            """
        )

