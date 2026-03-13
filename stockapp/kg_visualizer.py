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
except ImportError:
    KG_SCHEMA_AVAILABLE = False

try:
    from kg_builder import build_static_ontology_graph, KGResult
    KG_BUILDER_AVAILABLE = True
except ImportError:
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

# ── Node definitions for the static ontology ─────────────────────────────────

def _regime_nodes() -> list[dict]:
    return [
        {
            "id": "regime_postcovid",
            "label": "Post-COVID\n(Mar 2021–Jun 2022)",
            "layer": "regime",
            "tooltip": (
                "Post-COVID regime\n"
                "Procrustes vs Rate Shock: 0.342 (322 common tickers)\n"
                "Crowding score: 28.3 (Normal)\n"
                "Migration to Rate Shock: 43.7% (138/316 stocks)"
            ),
        },
        {
            "id": "regime_rateshock",
            "label": "Rate Shock\n(Jul 2022–Sep 2023)",
            "layer": "regime",
            "tooltip": (
                "Rate Shock regime\n"
                "Procrustes vs Disinflation: 0.186 (1,590 common tickers)\n"
                "Crowding score: 30.1 (Normal)\n"
                "Migration to Disinflation: 30.1% (95/316 stocks)"
            ),
        },
        {
            "id": "regime_disinflation",
            "label": "Disinflation\n(Oct 2023–Oct 2024)",
            "layer": "regime",
            "tooltip": (
                "Disinflation regime\n"
                "Crowding score: 67.9 (ELEVATED — compression signal)\n"
                "PC1 variance: 28.3% | PC2: 14.4% | PC3: 12.0%\n"
                "60.1% of tracked stocks changed quadrant at least once"
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


def _mechanism_nodes() -> list[dict]:
    return [
        {
            "id": "mech_pca",
            "label": "PCA\nStructural\nCoordinate System",
            "layer": "mechanism",
            "tooltip": (
                "Structural geometry — not return prediction\n"
                "PC1: 28.3% variance (Profitability & Quality)\n"
                "PC2: 14.4% variance (Valuation Style)\n"
                "PC3: 12.0% variance (Leverage & Asset Intensity)\n"
                "Zero prior art as governance diagnostic framing"
            ),
        },
        {
            "id": "mech_procrustes",
            "label": "Procrustes\nDisparity\n(Regime Break Detector)",
            "layer": "mechanism",
            "tooltip": (
                "Rotation-invariant structural distance between factor spaces\n"
                "0.342: Post-COVID → Rate Shock (322 tickers) — EXCEEDS 0.30 threshold\n"
                "0.459: Post-COVID → Disinflation (316 tickers) — EXCEEDS 0.30 threshold\n"
                "0.186: Rate Shock → Disinflation (1,590 tickers) — adjacent continuity\n"
                "ZERO PRIOR ART in finance"
            ),
        },
        {
            "id": "mech_crowding",
            "label": "Geometric\nCrowding\nDetection",
            "layer": "mechanism",
            "tooltip": (
                "Spatial compression in PCA factor space — prospective signal\n"
                "Scores: 28.3 (Post-COVID) / 30.1 (Rate Shock) / 67.9 (Disinflation)\n"
                "Formula: 0.6 × Concentration + 0.4 × (100 − Dispersion Normalized)\n"
                "ZERO PRIOR ART as geometric crowding measure"
            ),
        },
        {
            "id": "mech_kmeans",
            "label": "KMeans\nClustering",
            "layer": "mechanism",
            "tooltip": "Cuts across GICS sector lines — structural peer groups by factor geometry",
        },
        {
            "id": "mech_quadrant",
            "label": "Quadrant\nClassification",
            "layer": "mechanism",
            "tooltip": (
                "Four-quadrant grid on PC1/PC2 plane\n"
                "316 stocks tracked continuously across all three regimes\n"
                "60.1% changed quadrant at least once — governance event at scale"
            ),
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


# ── Edge definitions for the static ontology ─────────────────────────────────

def _structural_edges() -> list[dict]:
    edges = []

    # Regime temporal chain
    edges += [
        {"from": "regime_postcovid",  "to": "regime_rateshock",    "label": "Procrustes 0.342", "color": EDGE_COLORS["temporal"],   "width": 3},
        {"from": "regime_postcovid",  "to": "regime_disinflation", "label": "Procrustes 0.459", "color": EDGE_COLORS["temporal"],   "width": 4},
        {"from": "regime_rateshock",  "to": "regime_disinflation", "label": "Procrustes 0.186", "color": EDGE_COLORS["temporal"],   "width": 1.5},
    ]

    # PCA produces quadrant axes
    edges += [
        {"from": "mech_pca", "to": "q1", "label": "", "color": EDGE_COLORS["structural"], "width": 1.5},
        {"from": "mech_pca", "to": "q2", "label": "", "color": EDGE_COLORS["structural"], "width": 1.5},
        {"from": "mech_pca", "to": "q3", "label": "", "color": EDGE_COLORS["structural"], "width": 1.5},
        {"from": "mech_pca", "to": "q4", "label": "", "color": EDGE_COLORS["structural"], "width": 1.5},
    ]

    # Key factor → PCA loading relationships (dominant drivers only)
    pc1_drivers = ["factor_ctd", "factor_roa", "factor_ey"]
    pc2_drivers = ["factor_sp", "factor_btm", "factor_ey"]
    pc3_drivers = ["factor_dta", "factor_gp", "factor_vol"]

    for f in pc1_drivers:
        edges.append({"from": f, "to": "mech_pca", "label": "PC1", "color": EDGE_COLORS["structural"], "width": 2})
    for f in pc2_drivers:
        edges.append({"from": f, "to": "mech_pca", "label": "PC2", "color": EDGE_COLORS["membership"],  "width": 2})
    for f in pc3_drivers:
        edges.append({"from": f, "to": "mech_pca", "label": "PC3", "color": EDGE_COLORS["governance"],  "width": 2})

    # Procrustes compares regimes
    edges += [
        {"from": "mech_procrustes", "to": "regime_postcovid",   "label": "", "color": EDGE_COLORS["structural"], "width": 1.5},
        {"from": "mech_procrustes", "to": "regime_rateshock",   "label": "", "color": EDGE_COLORS["structural"], "width": 1.5},
        {"from": "mech_procrustes", "to": "regime_disinflation","label": "", "color": EDGE_COLORS["structural"], "width": 1.5},
    ]

    # Crowding scores per regime
    edges += [
        {"from": "mech_crowding", "to": "regime_postcovid",    "label": "28.3 Normal",   "color": EDGE_COLORS["crowding"], "width": 1.5},
        {"from": "mech_crowding", "to": "regime_rateshock",    "label": "30.1 Normal",   "color": EDGE_COLORS["crowding"], "width": 1.5},
        {"from": "mech_crowding", "to": "regime_disinflation", "label": "67.9 ELEVATED", "color": EDGE_COLORS["crowding"], "width": 3},
    ]

    # ESDS mechanisms
    edges += [
        {"from": "plat_esds", "to": "mech_pca",        "label": "", "color": EDGE_COLORS["governance"], "width": 2},
        {"from": "plat_esds", "to": "mech_procrustes", "label": "", "color": EDGE_COLORS["governance"], "width": 2},
        {"from": "plat_esds", "to": "mech_crowding",   "label": "", "color": EDGE_COLORS["governance"], "width": 2},
        {"from": "plat_esds", "to": "mech_kmeans",     "label": "", "color": EDGE_COLORS["governance"], "width": 2},
        {"from": "plat_esds", "to": "mech_quadrant",   "label": "", "color": EDGE_COLORS["governance"], "width": 2},
    ]

    # ESDS complements incumbents
    edges += [
        {"from": "plat_esds",  "to": "plat_barra",    "label": "complements", "color": EDGE_COLORS["governance"], "width": 2},
        {"from": "plat_esds",  "to": "plat_narrative","label": "Tier 1",      "color": EDGE_COLORS["governance"], "width": 1.5},
        {"from": "plat_esds",  "to": "plat_chatbot",  "label": "Tier 2",      "color": EDGE_COLORS["governance"], "width": 1.5},
    ]

    return edges


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
            title=edge.get("label", ""),
        )

    return net


def _render_pyvis_to_html(net: Network) -> str:
    """Return PyVis graph HTML directly."""
    return net.generate_html()


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

    with col1:
        st.metric("Procrustes — PC→RS",   "0.342", delta="322 tickers | EXCEEDS 0.30")
        st.metric("Procrustes — PC→D",    "0.459", delta="316 tickers | EXCEEDS 0.30")
        st.metric("Procrustes — RS→D",    "0.186", delta="1,590 tickers | continuity")

    with col2:
        st.metric("Crowding — Post-COVID",   "28.3", delta="Normal")
        st.metric("Crowding — Rate Shock",   "30.1", delta="Normal")
        st.metric("Crowding — Disinflation", "67.9", delta="ELEVATED ▲", delta_color="inverse")

    with col3:
        st.metric("Migration PC→RS",   "43.7%", delta="138 / 316 stocks")
        st.metric("Migration RS→D",    "30.1%", delta="95 / 316 stocks")
        st.metric("Changed quadrant",  "60.1%", delta="190 / 316 — governance scale")


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


@st.cache_data
def _build_cached_graph_html() -> str:
    """Build the PyVis graph once and cache the HTML."""
    net = _build_pyvis_network(height="680px")
    return _render_pyvis_to_html(net)


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
    with st.spinner("Building structural knowledge graph…"):
        try:
            html = _build_cached_graph_html()
            components.html(html, height=700, scrolling=False)
        except Exception as exc:
            st.error(f"Knowledge graph render error: {exc}")
            st.info(
                "If this is a first-run import error, confirm that "
                "`kg_schema.py` and `kg_builder.py` are in the same "
                "directory as `app.py` and that `pyvis` is in `requirements.txt`."
            )
            return

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

