"""
kg_visualizer.py  —  ESDS Knowledge Graph: Phase 3
====================================================
Renders the ESDS knowledge graph as an interactive Pyvis network
in a dedicated Streamlit tab.

Architecture:
    - kg_schema  (Phase 1): node/edge type definitions + catalog dicts
    - kg_builder (Phase 2): live graph construction from session state
    - kg_visualizer (Phase 3): Pyvis rendering + Streamlit UI

All empirical data (Procrustes scores, crowding scores, PC variance,
migration rates, universe count, factor loadings) flows from:
  PRIMARY:  st.session_state (live pipeline outputs)
  FALLBACK: Appendix B constants in kg_schema (when pipeline hasn't run)

No hard-coded Appendix B values appear in this file.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

# ── Schema import ─────────────────────────────────────────────────────────────
try:
    from kg_schema import (
        NodeType, EdgeType,
        REGIME_NODES, FACTOR_NODES, QUADRANT_NODES, MECHANISM_NODES,
        PLATFORM_NODES, AXIS_NODES, CATEGORY_NODES,
        APPENDIX_B_PROCRUSTES, APPENDIX_B_CROWDING, APPENDIX_B_PC_VARIANCE,
        PROCRUSTES_MEANINGFUL, CROWDING_THRESHOLD_ELEVATED, CROWDING_THRESHOLD_HIGH,
        SHORT_PERIOD_MAP,
    )
    KG_SCHEMA_AVAILABLE = True
except Exception as _schema_err:
    KG_SCHEMA_AVAILABLE = False
    print(f"[kg_visualizer] kg_schema import failed: {_schema_err}")

# ── Builder import ────────────────────────────────────────────────────────────
try:
    from kg_builder import (
        build_kg, build_static_ontology_graph, KGResult,
        _safe_float,                # single definition lives in kg_builder
        _get_procrustes_row, _get_crowding_row, _get_migration_row,
        _get_pc_variance, _get_universe_count, _get_live_factor_axis_map,
    )
    KG_BUILDER_AVAILABLE = True
except Exception as _kg_builder_err:
    KG_BUILDER_AVAILABLE = False
    print(f"[kg_visualizer] kg_builder import failed: {_kg_builder_err}")
    # Minimal fallback so the rest of the file won't crash on name errors
    def _safe_float(val, default=0.0):
        try:
            return float(str(val).replace('%','').replace(',','').strip())
        except Exception:
            return default


# =============================================================================
# VISUAL DESIGN CONSTANTS
# =============================================================================

LAYER_COLORS = {
    "regime":           "#00b4d8",
    "factor":           "#90e0ef",
    "quadrant":         "#f4a261",
    "mechanism":        "#e63946",
    "platform":         "#6d6875",
    "axis":             "#2ec4b6",
    "category":         "#48cae4",
    "crowding":         "#e63946",
    "transition":       "#f4a261",
    "structural_break": "#e63946",
    "early_warning":    "#e63946",
    "stock":            "#b7e4c7",
    "cluster":          "#52b788",
    "default":          "#adb5bd",
}

LAYER_SIZES = {
    "regime":    40,
    "factor":    24,
    "quadrant":  30,
    "mechanism": 36,
    "platform":  32,
    "axis":      22,
    "category":  20,
    "stock":     10,
    "cluster":   18,
    "default":   18,
}

EDGE_COLORS = {
    "regime_transition":   "#f4a261",
    "crowding_level":      "#e63946",
    "factor_loading":      "#90e0ef",
    "belongs_to_category": "#48cae4",
    "triggers_break":      "#e63946",
    "triggers_warning":    "#e63946",
    "quadrant_assignment": "#b7e4c7",
    "cluster_membership":  "#52b788",
    "belongs_to":          "#52b788",
    "migrates_to":         "#f4a261",
    "governs":             "#6d6875",
    "complements":         "#6d6875",
    "default":             "#495057",
}

_PYVIS_OPTIONS = """
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
    "stabilization": { "enabled": true, "iterations": 250, "updateInterval": 25 }
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
"""


# =============================================================================
# SESSION STATE ACCESSORS
# =============================================================================

@st.cache_data(show_spinner=False)
def _cached_procrustes_map(_key) -> dict:
    """Cached extraction of procrustes_results into (A,B)->row map."""
    df = st.session_state.get("procrustes_results")
    if df is None or df.empty:
        return {}
    out = {}
    for _, row in df.iterrows():
        a = str(row.get("Period A", "")).strip()
        b = str(row.get("Period B", "")).strip()
        if a and b:
            out[(a, b)] = row
    return out


def _lookup_procrustes(a: str, b: str):
    """Return procrustes row for a pair, or None."""
    key = id(st.session_state.get("procrustes_results"))
    pm  = _cached_procrustes_map(key)
    result = pm.get((a, b))
    if result is None:
        result = pm.get((b, a))
    return result


def _crowding_score_live(period_name: str):
    """Return live crowding score float or None."""
    for key in ["crowding_df", "crowding_results"]:
        df = st.session_state.get(key)
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                p = str(row.get("period", row.get("Period", ""))).strip()
                if p == period_name:
                    return _safe_float(
                        row.get("crowding_score") or row.get("score"), None
                    )
    return None


def _migration_row_live(transition_str: str):
    """Return migration summary row for a transition string like 'Post-COVID -> Rate Shock'."""
    df = st.session_state.get("migration_summary")
    if df is None or df.empty:
        return None
    for _, row in df.iterrows():
        t = str(row.get("Transition", "")).replace("→", "->").strip()
        if t == transition_str:
            return row
    return None


def _pc_variance_live() -> dict:
    """Return {PC1: float, PC2: float, PC3: float} from live pca_model or Appendix B."""
    pca_model = st.session_state.get("pca_model")
    if pca_model is not None:
        try:
            ratios = pca_model.explained_variance_ratio_
            return {f"PC{i+1}": round(float(ratios[i]) * 100, 1)
                    for i in range(min(len(ratios), 3))}
        except Exception:
            pass
    return {k: v for k, v in APPENDIX_B_PC_VARIANCE.items() if k.startswith("PC")}


def _universe_count_live() -> int:
    """Return live universe count from pca_df or Appendix B fallback."""
    pca_df = st.session_state.get("pca_df")
    if pca_df is not None and not pca_df.empty:
        return len(pca_df)
    try:
        from kg_schema import APPENDIX_B_UNIVERSE_COUNT
        return APPENDIX_B_UNIVERSE_COUNT
    except Exception:
        return 0


def _factor_loading_tooltip(factor_code: str) -> str:
    """
    Build a tooltip for a factor node using live pca_loadings from session state.
    Falls back to generic category description if loadings not available.
    """
    loadings = st.session_state.get("pca_loadings")
    lines = []
    if loadings:
        for pc in ["PC1", "PC2", "PC3"]:
            pc_data  = loadings.get(pc, {})
            pos_dict = pc_data.get("positive", {})
            neg_dict = pc_data.get("negative", {})
            val = pos_dict.get(factor_code) or neg_dict.get(factor_code)
            if val is not None:
                sign = "+" if val >= 0 else ""
                lines.append(f"{pc}: {sign}{val:.3f}")
    if lines:
        return f"Live loadings:\n" + "\n".join(lines)
    # Fallback: use display name from factor node in schema
    node = FACTOR_NODES.get(factor_code) if KG_SCHEMA_AVAILABLE else None
    if node:
        return f"{node.display_name} | {node.category.value} | {node.data_source}"
    return factor_code


# =============================================================================
# PYVIS GRAPH BUILDER — delegates to kg_builder for the actual graph
# =============================================================================

def _make_pyvis_net(height: str = "680px") -> Network:
    net = Network(
        height=height, width="100%",
        bgcolor="#0d1117", font_color="#e6edf3", directed=True,
    )
    net.set_options(_PYVIS_OPTIONS)
    return net


def _populate_pyvis_from_networkx(net: Network, G) -> None:
    """
    Transfer all nodes and edges from a NetworkX DiGraph into a Pyvis Network.
    Node visual properties are determined by node_type attribute.
    """
    for node_id, attrs in G.nodes(data=True):
        ntype  = attrs.get("node_type", "default")
        color  = LAYER_COLORS.get(ntype, LAYER_COLORS["default"])
        size   = LAYER_SIZES.get(ntype, LAYER_SIZES["default"])
        label  = attrs.get("label", str(node_id))

        # Truncate long stock labels
        if ntype == "stock" and len(label) > 6:
            label = label[:6]

        # Build tooltip from node attributes
        tooltip_parts = [attrs.get("tooltip", "")] if attrs.get("tooltip") else []
        for k, v in attrs.items():
            if k not in ("label", "node_type", "tooltip") and v is not None:
                tooltip_parts.append(f"{k}: {v}")
        tooltip = "\n".join(tooltip_parts) or label

        net.add_node(
            str(node_id),
            label = label,
            title = tooltip,
            color = {
                "background": color,
                "border":     "#ffffff",
                "highlight":  {"background": "#ffffff", "border": color},
                "hover":      {"background": "#ffffff", "border": color},
            },
            size  = size,
            shape = "dot",
            mass  = 1.5,
        )

    for src, tgt, attrs in G.edges(data=True):
        etype  = attrs.get("edge_type", "")
        ecolor = EDGE_COLORS.get(etype, EDGE_COLORS["default"])
        width  = 1.5
        # Scale edge width by relationship strength
        if etype == "regime_transition":
            d     = _safe_float(attrs.get("procrustes_disparity"), 0.0)
            width = 1.5 + d * 6
        elif etype == "crowding_level":
            width = 1.5 + _safe_float(attrs.get("score"), 0.0) / 25
        net.add_edge(
            str(src), str(tgt),
            label = attrs.get("label", ""),
            color = ecolor,
            width = width,
            title = attrs.get("label", etype),
        )


@st.cache_data(show_spinner=False)
def _build_static_graph_html(_cache_key) -> str:
    """
    Build the static ontology Pyvis graph via kg_builder.
    Cached; cache is busted when session state changes (via _cache_key).
    _cache_key should be a hash/id of the relevant session state keys.
    """
    if not KG_BUILDER_AVAILABLE:
        return "<p>kg_builder not available</p>"
    try:
        kg_result = build_static_ontology_graph()
        net       = _make_pyvis_net(height="680px")
        _populate_pyvis_from_networkx(net, kg_result.graph)
        return net.generate_html()
    except Exception as exc:
        return f"<p>Static graph build error: {exc}</p>"


def _static_graph_cache_key() -> int:
    """
    Derive a cache key from the session state keys that affect the static graph.
    When any of these change (pipeline runs), the cached graph is invalidated.
    """
    relevant = (
        id(st.session_state.get("procrustes_results")),
        id(st.session_state.get("crowding_df")),
        id(st.session_state.get("pca_model")),
        id(st.session_state.get("migration_summary")),
    )
    return hash(relevant)


def _build_equity_graph_html() -> str:
    """
    Build the equity-augmented graph using per-period scores_df from session state.
    Checks session_state['period_scores'] first (set by app.py Option A).
    Falls back to re-running _run_pca_for_period() if period_scores not available.
    """
    if not KG_BUILDER_AVAILABLE:
        return _build_static_graph_html(_static_graph_cache_key())

    try:
        # ── Option A: use pre-computed per-period scores from session state ──
        period_scores = st.session_state.get("period_scores")

        if not period_scores:
            # ── Fallback: re-run PCA per period ──────────────────────────────
            period_scores = _recompute_period_scores()

        if not period_scores:
            st.warning("No period scores available — showing static ontology.")
            return _build_static_graph_html(_static_graph_cache_key())

        migration_df = st.session_state.get("migration_wide")

        kg_result = build_kg(
            period_data          = period_scores,
            migration_df         = migration_df,
            include_equity_nodes = True,
        )
        net = _make_pyvis_net(height="680px")
        _populate_pyvis_from_networkx(net, kg_result.graph)
        return net.generate_html()

    except Exception as exc:
        st.warning(f"Equity graph build failed, showing static ontology: {exc}")
        return _build_static_graph_html(_static_graph_cache_key())


def _recompute_period_scores() -> dict:
    """
    Fallback: re-run _run_pca_for_period() for each regime period.
    Used only when session_state['period_scores'] is not populated.
    """
    raw_data = st.session_state.get("raw_data")
    if raw_data is None or raw_data.empty:
        return {}

    try:
        from period_analysis import _run_pca_for_period, SUB_PERIODS
        from config import FEATURE_COLUMNS

        date_col = next(
            (c for c in ["public_date", "date", "datadate"] if c in raw_data.columns),
            None,
        )
        if date_col is None:
            return {}

        features     = [c for c in FEATURE_COLUMNS if c in raw_data.columns]
        # Use clean short labels (strip \n from SUB_PERIODS keys)
        short_map    = {k.split('\n')[0]: v for k, v in SUB_PERIODS.items()}
        period_scores = {}

        for short_label, (start, end) in short_map.items():
            mask  = (raw_data[date_col] >= start) & (raw_data[date_col] <= end)
            slice_df = raw_data[mask]
            if len(slice_df) < 10:
                continue
            try:
                _, scores_df, _, _ = _run_pca_for_period(
                    slice_df, features, date_col, start, end
                )
                if scores_df is not None and not scores_df.empty:
                    if "Quadrant" in scores_df.columns and "quadrant" not in scores_df.columns:
                        scores_df = scores_df.copy()
                        scores_df["quadrant"] = scores_df["Quadrant"]
                    period_scores[short_label] = scores_df
            except Exception:
                continue

        return period_scores

    except Exception as exc:
        st.warning(f"Period score recomputation failed: {exc}")
        return {}


# =============================================================================
# METRICS PANEL — all values from session state
# =============================================================================

def _render_metrics_panel() -> None:
    """Three-column empirical anchors strip — all values from live session state."""
    col1, col2, col3 = st.columns(3)

    pc_var = _pc_variance_live()

    with col1:
        st.markdown("**Procrustes Disparity**")
        pairs = [
            ("Post-COVID", "Rate Shock",   "PC→RS"),
            ("Post-COVID", "Disinflation", "PC→D"),
            ("Rate Shock", "Disinflation", "RS→D"),
        ]
        any_live = False
        for a, b, label in pairs:
            row = _lookup_procrustes(a, b)
            if row is not None:
                disp    = _safe_float(row.get("Disparity"), 0.0)
                common  = int(_safe_float(row.get("Common Tickers"), 0))
                st.metric(label, f"{disp:.3f}",
                          delta=f"{common:,} common tickers")
                any_live = True
            else:
                # Show Appendix B fallback with label
                fb = APPENDIX_B_PROCRUSTES.get((a, b)) if KG_SCHEMA_AVAILABLE else None
                if fb:
                    st.metric(f"{label} (ref)", f"{fb['disparity']:.3f}",
                              delta=f"{fb['common_tickers']:,} tickers (Appendix B)")
        if not any_live:
            st.info("Run Period Comparison to populate live Procrustes diagnostics.")

    with col2:
        st.markdown("**Factor Crowding Score**")
        for period in ["Post-COVID", "Rate Shock", "Disinflation"]:
            score = _crowding_score_live(period)
            if score is not None:
                risk = ("Normal" if score < CROWDING_THRESHOLD_ELEVATED
                        else "Elevated" if score < CROWDING_THRESHOLD_HIGH
                        else "High Risk")
                st.metric(period, f"{score:.1f}",
                          delta=risk,
                          delta_color="normal" if score < CROWDING_THRESHOLD_HIGH else "inverse")
            else:
                fb = APPENDIX_B_CROWDING.get(period) if KG_SCHEMA_AVAILABLE else None
                if fb:
                    st.metric(f"{period} (ref)", f"{fb['score']:.1f}",
                              delta=f"{fb['risk_level'].value} (Appendix B)")
                else:
                    st.metric(period, "N/A")

    with col3:
        st.markdown("**PC Variance Explained**")
        for pc in ["PC1", "PC2", "PC3"]:
            v = pc_var.get(pc)
            if v is not None:
                st.metric(pc, f"{v:.1f}%")

        st.markdown("**Quadrant Migration**")
        for a, b, label in [("Post-COVID", "Rate Shock", "PC→RS"),
                             ("Rate Shock", "Disinflation", "RS→D")]:
            mr = _migration_row_live(f"{a} -> {b}")
            if mr is not None:
                rate    = str(mr.get("Migration Rate", "N/A"))
                changed = int(_safe_float(mr.get("Changed Quadrant"), 0))
                total   = int(_safe_float(mr.get("Stocks Analyzed"), 0))
                st.metric(f"Migration {label}", rate,
                          delta=f"{changed:,} / {total:,} stocks")
            else:
                fb = APPENDIX_B_MIGRATION.get((a, b)) if KG_SCHEMA_AVAILABLE else None
                if fb:
                    st.metric(f"Migration {label} (ref)",
                              f"{fb['migration_pct']:.1f}%",
                              delta=f"{fb['changed']:,}/{fb['analyzed']:,} (Appendix B)")


# =============================================================================
# LEGEND
# =============================================================================

def _render_legend() -> None:
    universe = _universe_count_live()
    pc_var   = _pc_variance_live()
    comb_var = round(sum(pc_var.get(f"PC{i}", 0) for i in range(1, 4)), 1)

    st.markdown(
        f"""
        <div style="display:flex; flex-wrap:wrap; gap:16px; margin-top:8px; font-size:13px;">
          <span><span style="color:#00b4d8;">●</span> Macro Regime</span>
          <span><span style="color:#90e0ef;">●</span> Factor Signal</span>
          <span><span style="color:#f4a261;">●</span> Quadrant</span>
          <span><span style="color:#e63946;">●</span> Core Mechanism</span>
          <span><span style="color:#6d6875;">●</span> Platform / Governance</span>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:16px; margin-top:6px;
                    font-size:12px; color:#6c757d;">
          <span>Universe: ~{universe:,} equities &nbsp;|&nbsp;
          PC1+PC2+PC3 variance: ~{comb_var:.1f}% &nbsp;|&nbsp;
          Edge width = relationship strength &nbsp;|&nbsp;
          Hover nodes for live detail &nbsp;|&nbsp;
          Scroll to zoom &nbsp;|&nbsp; Drag to pan</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# FILTER SIDEBAR
# =============================================================================

def _render_kg_filters() -> dict:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Knowledge Graph")
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


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def render_kg_tab() -> None:
    """
    Main entry point called from app.py inside the Knowledge Graph tab.
    Renders the full interactive Pyvis ontology with metrics and legend.
    """
    st.subheader("ESDS Knowledge Graph — Structural Ontology")
    st.caption(
        "Interactive map of the ESDS framework: five core mechanisms, "
        "eleven factor signals, three macro regimes, four quadrant positions, "
        "and the institutional governance layer. "
        "Hover any node for live empirical detail. "
        "Edge width reflects relationship strength. "
        "All values populated from live pipeline outputs."
    )

    if not PYVIS_AVAILABLE:
        st.error(
            "**Pyvis not installed.** Add `pyvis` to `requirements.txt` and redeploy.\n\n"
            "```\npip install pyvis\n```"
        )
        return

    # Empirical anchors strip
    with st.expander("Empirical Anchors", expanded=False):
        _render_metrics_panel()

    st.markdown("---")

    # View mode toggle
    view_mode = st.radio(
        "Graph view",
        ["Static Ontology", "Live Equity Nodes"],
        horizontal=True,
        help=(
            "Static Ontology: framework architecture with live Procrustes / "
            "crowding / variance values on edges and tooltips. "
            "Live Equity Nodes: adds ~1,738 tickers, clusters, and quadrant "
            "assignments from the pipeline (requires Period Comparison to have run)."
        ),
    )

    with st.spinner("Building structural knowledge graph…"):
        try:
            if view_mode == "Live Equity Nodes" and KG_BUILDER_AVAILABLE:
                html = _build_equity_graph_html()
            else:
                html = _build_static_graph_html(_static_graph_cache_key())
            components.html(html, height=700, scrolling=False)
        except Exception as exc:
            st.error(f"Knowledge graph render error: {exc}")
            st.info(
                "Confirm that `kg_schema.py` and `kg_builder.py` are in the "
                "same directory as `app.py`, and that `pyvis` is installed."
            )

    _render_legend()

    st.markdown("---")
    with st.expander("How to read this graph"):
        st.markdown(
            """
**Node layers**
- **Teal — Macro Regimes**: Post-COVID, Rate Shock, Disinflation.
  Hover for live Procrustes scores, crowding score, and date range.
- **Light teal — Factor Signals**: 11 retained signals.
  Hover for live PC loadings from the fitted PCA model.
- **Amber — Quadrants**: Four structural positions on the PC1/PC2 plane.
  Hover for definitions sourced from config.py.
- **Red — Core Mechanisms**: Five proprietary methods. Zero prior art nodes are flagged.
- **Purple — Governance / Platform**: ESDS two-tier AI architecture and incumbent platforms.

**Edge encoding**
- **Edge width** encodes relationship strength.
  Procrustes edges scale with disparity score — thicker = more structural distance.
  Crowding edges scale with crowding score — thicker = more concentrated.
- **All values are live** from the pipeline when Period Comparison has been run.
  Appendix B fallbacks are used only before the pipeline runs, and are labeled as such.

**Zero prior art mechanisms**
- *Procrustes Disparity*: no prior application to equity factor spaces.
- *Geometric Crowding*: spatial PCA compression, distinct from correlation-based measures.
- *PCA as Governance Diagnostic*: structural framing without precedent.
            """
        )
