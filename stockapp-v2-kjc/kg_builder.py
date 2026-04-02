"""
kg_builder.py  —  ESDS Knowledge Graph: Phase 2
================================================
Constructs a NetworkX knowledge graph from kg_schema.py canonical catalogs.

Exported symbols used by kg_visualizer.py:
    build_static_ontology_graph() -> KGResult
    build_kg(period_data, migration_df, include_equity_nodes) -> KGResult
    KGResult
    get_structural_summary(G) -> dict
    _safe_float(val, default) -> float        (imported by kg_visualizer)

All empirical values (Procrustes scores, crowding scores, PC variance,
migration rates, universe count) are drawn from:
  1. Streamlit session state (live pipeline outputs) — primary source
  2. Appendix B fallback constants in kg_schema — only when pipeline hasn't run

Standalone validation:
    python kg_builder.py
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    sys.path.insert(0, str(Path(__file__).parent))
except NameError:
    import os
    sys.path.insert(0, os.getcwd())

import networkx as nx

# ── Schema import ─────────────────────────────────────────────────────────────
try:
    from kg_schema import (
        # Catalog dicts
        REGIME_NODES, FACTOR_NODES, QUADRANT_NODES, AXIS_NODES,
        CATEGORY_NODES, MECHANISM_NODES, PLATFORM_NODES,
        TRANSITION_PAIRS, SHORT_PERIOD_MAP,
        # Appendix B fallbacks (display only, never used for graph construction)
        APPENDIX_B_PROCRUSTES, APPENDIX_B_CROWDING,
        APPENDIX_B_PC_VARIANCE, APPENDIX_B_MIGRATION,
        APPENDIX_B_UNIVERSE_COUNT,
        # Enums / types
        RegimeName, QuadrantID, RiskLevel, StructuralBreakSeverity,
        NodeType, EdgeType,
        # Methodology constants
        PROCRUSTES_MEANINGFUL, CROWDING_THRESHOLD_ELEVATED, CROWDING_THRESHOLD_HIGH,
    )
    KG_SCHEMA_AVAILABLE = True
except Exception as e:
    KG_SCHEMA_AVAILABLE = False
    print(f"[kg_builder] Warning: kg_schema import failed: {e}")


# ── _safe_float (single definition — imported by kg_visualizer) ───────────────

def _safe_float(val, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    Strips % signs and commas for display strings.
    """
    try:
        if val is None:
            return default
        return float(str(val).replace('%', '').replace(',', '').strip())
    except (TypeError, ValueError):
        return default


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class KGResult:
    graph: nx.DiGraph
    n_nodes: int = 0
    n_edges: int = 0
    n_tickers: int = 0
    n_clusters: int = 0
    regime_coverage: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "── ESDS Knowledge Graph ─────────────────────────────────────",
            f"  Nodes          : {self.n_nodes:,}",
            f"  Edges          : {self.n_edges:,}",
            f"  Tickers        : {self.n_tickers:,}",
            f"  Clusters       : {self.n_clusters:,}",
            f"  Regimes wired  : {', '.join(self.regime_coverage) or 'static only'}",
        ]
        for w in self.warnings[:5]:
            lines.append(f"  WARNING: {w}")
        lines.append("─" * 60)
        return "\n".join(lines)


# ── Session state helpers ─────────────────────────────────────────────────────

def _get_session_state():
    """Return st.session_state if Streamlit is available, else empty dict."""
    try:
        import streamlit as st
        return st.session_state
    except Exception:
        return {}


def _get_procrustes_row(ss, period_a: str, period_b: str):
    """
    Look up a Procrustes row from session_state['procrustes_results'].
    Falls back to Appendix B if session state not populated.
    Returns a dict with keys: disparity, common_tickers, interpretation.
    """
    proc_df = ss.get("procrustes_results")
    if proc_df is not None and not proc_df.empty:
        for _, row in proc_df.iterrows():
            pa = str(row.get("Period A", "")).strip()
            pb = str(row.get("Period B", "")).strip()
            if (pa == period_a and pb == period_b) or (pa == period_b and pb == period_a):
                return {
                    "disparity":       _safe_float(row.get("Disparity"), 0.0),
                    "common_tickers":  int(_safe_float(row.get("Common Tickers"), 0)),
                    "interpretation":  str(row.get("Interpretation", "")),
                    "source":          "live",
                }
    # Fallback to Appendix B
    fb = APPENDIX_B_PROCRUSTES.get((period_a, period_b)) or \
         APPENDIX_B_PROCRUSTES.get((period_b, period_a))
    if fb:
        return {**fb, "source": "appendix_b_fallback"}
    return {"disparity": 0.0, "common_tickers": 0, "interpretation": "", "source": "missing"}


def _get_crowding_row(ss, period: str):
    """
    Look up a crowding row from session_state['crowding_df'].
    Falls back to Appendix B if not populated.
    Returns dict with keys: score, risk_level, largest_cluster_pct, centroid_dispersion.
    """
    for key in ["crowding_df", "crowding_results"]:
        df = ss.get(key)
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                p = str(row.get("period", row.get("Period", ""))).strip()
                if p == period:
                    return {
                        "score":               _safe_float(row.get("crowding_score", row.get("score")), 0.0),
                        "risk_level":          str(row.get("risk_level", "Normal")),
                        "largest_cluster_pct": _safe_float(row.get("largest_cluster_pct"), 0.0),
                        "centroid_dispersion": _safe_float(row.get("centroid_dispersion"), 0.0),
                        "source":              "live",
                    }
    fb = APPENDIX_B_CROWDING.get(period)
    if fb:
        return {
            "score":               fb["score"],
            "risk_level":          fb["risk_level"].value,
            "largest_cluster_pct": 0.0,
            "centroid_dispersion": 0.0,
            "source":              "appendix_b_fallback",
        }
    return {"score": 0.0, "risk_level": "Normal", "largest_cluster_pct": 0.0,
            "centroid_dispersion": 0.0, "source": "missing"}


def _get_migration_row(ss, period_a: str, period_b: str):
    """
    Look up a migration row from session_state['migration_summary'].
    Falls back to Appendix B if not populated.
    Returns dict with keys: migration_pct, changed, analyzed.
    """
    df = ss.get("migration_summary")
    transition_key = f"{period_a} -> {period_b}"
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            t = str(row.get("Transition", "")).replace("→", "->").strip()
            if t == transition_key:
                analyzed = int(_safe_float(row.get("Stocks Analyzed"), 0))
                changed  = int(_safe_float(row.get("Changed Quadrant"), 0))
                pct_str  = str(row.get("Migration Rate", "0%"))
                pct      = _safe_float(pct_str, 0.0)
                return {"migration_pct": pct, "changed": changed,
                        "analyzed": analyzed, "source": "live"}
    fb = APPENDIX_B_MIGRATION.get((period_a, period_b))
    if fb:
        return {**fb, "source": "appendix_b_fallback"}
    return {"migration_pct": 0.0, "changed": 0, "analyzed": 0, "source": "missing"}


def _get_pc_variance(ss):
    """
    Get PC variance ratios from session_state['pca_model'].
    Falls back to Appendix B values.
    Returns dict {PC1: float, PC2: float, PC3: float}.
    """
    pca_model = ss.get("pca_model")
    if pca_model is not None:
        try:
            ratios = pca_model.explained_variance_ratio_
            return {
                f"PC{i+1}": round(float(ratios[i]) * 100, 1)
                for i in range(min(len(ratios), 3))
            }
        except Exception:
            pass
    return {k: v for k, v in APPENDIX_B_PC_VARIANCE.items() if k.startswith("PC")}


def _get_universe_count(ss):
    """
    Get universe count from session_state['pca_df'].
    Falls back to Appendix B.
    """
    pca_df = ss.get("pca_df")
    if pca_df is not None and not pca_df.empty:
        return len(pca_df)
    return APPENDIX_B_UNIVERSE_COUNT


def _get_live_factor_axis_map(ss):
    """
    Build factor -> [axis_id, ...] mapping from live pca_loadings in session state.
    Uses top 3 positive loading factors per PC.
    Falls back to a config-derived static map if loadings not available.

    Returns dict: {"axis:PC1": [factor_ids], "axis:PC2": [...], "axis:PC3": [...]}
    """
    loadings = ss.get("pca_loadings")
    if loadings:
        result = {}
        for pc_key in ["PC1", "PC2", "PC3"]:
            axis_id = f"axis:{pc_key}"
            pc_data = loadings.get(pc_key, {})
            # Top 3 positive loadings
            pos = pc_data.get("positive", {})
            top_factors = [f"factor:{code}" for code in list(pos.keys())[:3]]
            if top_factors:
                result[axis_id] = top_factors
        if result:
            return result

    # Static fallback derived from config (no Appendix B loading values)
    try:
        from config import FACTOR_CATEGORIES
        # Use known category-PC relationships from the ESDS methodology
        # PC1 = Profitability/Quality -> Quality + Financial Strength factors
        # PC2 = Valuation Style -> Value factors
        # PC3 = Leverage/Risk -> Financial Strength + Risk factors
        quality_factors   = [f"factor:{c}" for c in FACTOR_CATEGORIES.get("Quality", [])[:3]]
        value_factors     = [f"factor:{c}" for c in FACTOR_CATEGORIES.get("Value", [])[:3]]
        fin_str_factors   = [f"factor:{c}" for c in FACTOR_CATEGORIES.get("Financial Strength", [])[:2]]
        risk_factors      = [f"factor:{c}" for c in FACTOR_CATEGORIES.get("Risk/Volatility", [])[:2]]
        return {
            "axis:PC1": quality_factors + fin_str_factors[:1],
            "axis:PC2": value_factors,
            "axis:PC3": fin_str_factors[1:] + risk_factors,
        }
    except Exception:
        return {}


# ── Severity classifier ───────────────────────────────────────────────────────

def _classify_severity(disparity: float) -> StructuralBreakSeverity:
    if disparity >= PROCRUSTES_MEANINGFUL:
        return StructuralBreakSeverity.MAJOR
    elif disparity >= 0.15:
        return StructuralBreakSeverity.MEANINGFUL
    elif disparity >= 0.05:
        return StructuralBreakSeverity.DETECTABLE
    return StructuralBreakSeverity.NEGLIGIBLE


def _classify_risk(score: float) -> RiskLevel:
    if score >= CROWDING_THRESHOLD_HIGH:
        return RiskLevel.HIGH
    elif score >= CROWDING_THRESHOLD_ELEVATED:
        return RiskLevel.ELEVATED
    return RiskLevel.NORMAL


# ── Phase A: static nodes ─────────────────────────────────────────────────────

def _wire_static_nodes(G: nx.DiGraph, ss: dict) -> None:
    """Add all static + session-state-enriched nodes from schema catalogs."""

    pc_variance = _get_pc_variance(ss)
    universe    = _get_universe_count(ss)

    # Regime nodes
    for short, node in REGIME_NODES.items():
        crowding = _get_crowding_row(ss, short)
        G.add_node(
            node.node_id,
            label                = node.name.value,
            node_type            = NodeType.REGIME.value,
            start_date           = node.start_date,
            end_date             = node.end_date,
            universe_count       = universe,
            crowding_score       = crowding["score"],
            crowding_risk        = crowding["risk_level"],
            crowding_source      = crowding["source"],
        )

    # Factor nodes
    for code, node in FACTOR_NODES.items():
        G.add_node(
            node.node_id,
            label        = node.display_name,
            node_type    = NodeType.FACTOR.value,
            category     = node.category.value,
            data_source  = node.data_source,
            description  = node.description,
        )

    # Quadrant nodes
    for qid, node in QUADRANT_NODES.items():
        G.add_node(
            node.node_id,
            label       = f"{qid}: {node.name}",
            node_type   = NodeType.QUADRANT.value,
            pc1_sign    = node.pc1_sign,
            pc2_sign    = node.pc2_sign,
            description = node.description,
        )

    # Axis nodes — variance from live pca_model
    for pc, node in AXIS_NODES.items():
        live_var = pc_variance.get(pc, node.variance_explained)
        G.add_node(
            node.node_id,
            label              = f"{pc}: {node.name}",
            node_type          = NodeType.AXIS.value,
            variance_explained = live_var,
            high_meaning       = node.high_meaning,
            low_meaning        = node.low_meaning,
        )

    # Category nodes
    for cat, node in CATEGORY_NODES.items():
        G.add_node(
            node.node_id,
            label     = cat,
            node_type = NodeType.CATEGORY.value,
            members   = list(node.members),
        )

    # Mechanism nodes
    for key, node in MECHANISM_NODES.items():
        G.add_node(
            node.node_id,
            label          = node.label,
            node_type      = NodeType.MECHANISM.value,
            tooltip        = node.tooltip,
            zero_prior_art = node.zero_prior_art,
        )

    # Platform nodes — enrich ESDS tooltip with live universe count
    for key, node in PLATFORM_NODES.items():
        tooltip = node.tooltip
        if key == "esds":
            tooltip = tooltip.replace(
                "Universe count: populated live from session state.",
                f"Universe: ~{universe:,} U.S. equities."
            )
        G.add_node(
            node.node_id,
            label     = node.label,
            node_type = NodeType.PLATFORM.value,
            tooltip   = tooltip,
        )


# ── Phase B: structural edges ─────────────────────────────────────────────────

def _wire_structural_edges(G: nx.DiGraph, ss: dict) -> None:
    """Wire all structural relationships using live session state values."""

    # Regime -> Crowding edges
    for short in REGIME_NODES:
        regime_id   = f"regime:{short}"
        crowding_id = f"crowding:{short}"  # crowding nodes added in Phase A via regime attrs
        crowding = _get_crowding_row(ss, short)
        # Attach crowding data as edge from regime to itself as attribute
        # (crowding is encoded directly on regime node; edge is to mechanism node)
        if G.has_node(regime_id) and G.has_node("mechanism:crowding"):
            G.add_edge(
                regime_id, "mechanism:crowding",
                edge_type    = EdgeType.CROWDING_LEVEL.value,
                score        = crowding["score"],
                risk_level   = crowding["risk_level"],
                label        = f"{crowding['score']:.1f}",
                data_source  = crowding["source"],
            )

    # Regime -> Regime (Procrustes transition edges)
    for (r_from, r_to) in TRANSITION_PAIRS:
        src = f"regime:{r_from}"
        tgt = f"regime:{r_to}"
        if not (G.has_node(src) and G.has_node(tgt)):
            continue
        proc = _get_procrustes_row(ss, r_from, r_to)
        mig  = _get_migration_row(ss, r_from, r_to)
        disp = proc["disparity"]
        sev  = _classify_severity(disp)
        G.add_edge(
            src, tgt,
            edge_type            = EdgeType.REGIME_TRANSITION.value,
            procrustes_disparity = disp,
            common_tickers       = proc["common_tickers"],
            interpretation       = proc["interpretation"],
            migration_pct        = mig["migration_pct"],
            stocks_changed       = mig["changed"],
            stocks_analyzed      = mig["analyzed"],
            severity             = sev.value,
            is_major_break       = disp >= PROCRUSTES_MEANINGFUL,
            label                = f"{disp:.3f}",
            data_source          = proc["source"],
        )

    # Factor -> Axis (live loadings or config-derived fallback)
    factor_axis_map = _get_live_factor_axis_map(ss)
    for axis_id, factor_ids in factor_axis_map.items():
        for factor_id in factor_ids:
            if G.has_node(factor_id) and G.has_node(axis_id):
                pc_label = axis_id.replace("axis:", "")
                G.add_edge(
                    factor_id, axis_id,
                    edge_type = EdgeType.FACTOR_LOADING.value,
                    label     = pc_label,
                )

    # Factor -> Category
    for cat, node in CATEGORY_NODES.items():
        cat_id = node.node_id
        for member_code in node.members:
            factor_id = f"factor:{member_code}"
            if G.has_node(factor_id) and G.has_node(cat_id):
                G.add_edge(factor_id, cat_id,
                           edge_type = EdgeType.BELONGS_TO_CATEGORY.value)

    # ESDS -> Mechanisms
    if G.has_node("platform:esds"):
        for mech_id in [n for n in G.nodes if G.nodes[n].get("node_type") == NodeType.MECHANISM.value]:
            G.add_edge("platform:esds", mech_id,
                       edge_type = EdgeType.GOVERNS.value, label="")

    # ESDS -> Barra (complements)
    if G.has_node("platform:esds") and G.has_node("platform:barra"):
        G.add_edge("platform:esds", "platform:barra",
                   edge_type = EdgeType.COMPLEMENTS.value, label="complements")

    # ESDS -> Tier 1/2 platforms
    for plat in ["platform:narrative", "platform:chatbot"]:
        if G.has_node("platform:esds") and G.has_node(plat):
            tier = "Tier 1" if "narrative" in plat else "Tier 2"
            G.add_edge("platform:esds", plat,
                       edge_type = EdgeType.GOVERNS.value, label=tier)

    # PCA mechanism -> Quadrants
    if G.has_node("mechanism:pca"):
        for qid in QUADRANT_NODES:
            q_node_id = f"quadrant:{qid}"
            if G.has_node(q_node_id):
                G.add_edge("mechanism:pca", q_node_id,
                           edge_type = EdgeType.GOVERNS.value, label="")


# ── Phase C: dynamic equity nodes ─────────────────────────────────────────────

def _wire_equity_nodes(G: nx.DiGraph, period_data: dict, result: KGResult) -> None:
    ticker_seen  = set()
    cluster_seen = set()
    n_clusters   = 0

    for period_label, df in period_data.items():
        regime_id = f"regime:{period_label}"
        if not G.has_node(regime_id) or df is None or df.empty:
            result.warnings.append(f"Skipping '{period_label}' — no data or regime node missing.")
            continue

        df = df.copy()
        if "Quadrant" in df.columns and "quadrant" not in df.columns:
            df["quadrant"] = df["Quadrant"]

        if not {"ticker", "quadrant"}.issubset(df.columns):
            result.warnings.append(f"'{period_label}' missing ticker or quadrant columns.")
            continue

        result.regime_coverage.append(period_label)

        for _, row in df.iterrows():
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker:
                continue

            stock_id = f"stock:{ticker}"
            if ticker not in ticker_seen:
                G.add_node(
                    stock_id,
                    label      = ticker,
                    node_type  = NodeType.STOCK.value,
                    gics_sector= str(row.get("gics_sector", row.get("gicdesc", "Unknown"))),
                )
                ticker_seen.add(ticker)

            raw_q     = str(row.get("quadrant", ""))
            q_short   = raw_q.split(":")[0].strip()
            quadrant_id = f"quadrant:{q_short}"
            if G.has_node(quadrant_id):
                G.add_edge(
                    stock_id, quadrant_id,
                    edge_type = EdgeType.QUADRANT_ASSIGNMENT.value,
                    regime    = period_label,
                    pc1       = _safe_float(row.get("PC1", 0)),
                    pc2       = _safe_float(row.get("PC2", 0)),
                )

            cluster_label = row.get("cluster")
            if cluster_label is not None:
                cluster_id = f"cluster:{period_label}:{cluster_label}"
                if cluster_id not in cluster_seen:
                    G.add_node(
                        cluster_id,
                        label         = f"{period_label} Cluster {cluster_label}",
                        node_type     = NodeType.CLUSTER.value,
                        regime        = period_label,
                        cluster_label = str(cluster_label),
                    )
                    G.add_edge(cluster_id, regime_id,
                               edge_type = EdgeType.BELONGS_TO.value)
                    cluster_seen.add(cluster_id)
                    n_clusters += 1
                G.add_edge(stock_id, cluster_id,
                           edge_type = EdgeType.CLUSTER_MEMBERSHIP.value,
                           regime    = period_label)

    result.n_tickers  = len(ticker_seen)
    result.n_clusters = n_clusters


# ── Phase D: quadrant migration edges ────────────────────────────────────────

def _wire_migration_edges(G: nx.DiGraph, migration_df, result: KGResult) -> None:
    if migration_df is None or migration_df.empty:
        result.warnings.append("No migration_df — migration edges skipped.")
        return

    period_cols = [c for c in ["Post-COVID", "Rate Shock", "Disinflation"]
                   if c in migration_df.columns]

    for i in range(len(period_cols) - 1):
        col_from, col_to = period_cols[i], period_cols[i + 1]
        sub     = migration_df[[col_from, col_to]].dropna()
        grouped = sub.groupby([col_from, col_to]).size().reset_index(name="count")
        for _, row in grouped.iterrows():
            q_from = str(row[col_from]).split(":")[0].strip()
            q_to   = str(row[col_to]).split(":")[0].strip()
            src    = f"quadrant:{q_from}"
            tgt    = f"quadrant:{q_to}"
            if G.has_node(src) and G.has_node(tgt):
                G.add_edge(
                    src, tgt,
                    edge_type   = EdgeType.MIGRATES_TO.value,
                    count       = int(row["count"]),
                    from_period = col_from,
                    to_period   = col_to,
                )


# ── Master build functions ────────────────────────────────────────────────────

def build_static_ontology_graph() -> KGResult:
    """
    Build static ESDS ontology (no equity nodes).
    All empirical values drawn from session state where available,
    falling back to Appendix B constants only when pipeline hasn't run.
    Used by kg_visualizer.py for the Static Ontology view.
    """
    if not KG_SCHEMA_AVAILABLE:
        return KGResult(graph=nx.DiGraph(), warnings=["kg_schema not available"])

    ss = _get_session_state()
    G  = nx.DiGraph(name="ESDS_Knowledge_Graph", version="3.0")
    result = KGResult(graph=G)

    _wire_static_nodes(G, ss)
    _wire_structural_edges(G, ss)

    result.n_nodes = G.number_of_nodes()
    result.n_edges = G.number_of_edges()
    return result


def build_kg(
    period_data: Optional[dict] = None,
    migration_df=None,
    include_equity_nodes: bool = True,
) -> KGResult:
    """
    Build full ESDS KG with optional live equity nodes.
    period_data: dict of {period_label: scores_df} from session_state['period_scores']
    migration_df: from session_state['migration_wide']
    """
    if not KG_SCHEMA_AVAILABLE:
        return KGResult(graph=nx.DiGraph(), warnings=["kg_schema not available"])

    ss = _get_session_state()
    G  = nx.DiGraph(name="ESDS_Knowledge_Graph", version="3.0")
    result = KGResult(graph=G)

    _wire_static_nodes(G, ss)
    _wire_structural_edges(G, ss)

    if include_equity_nodes and period_data:
        _wire_equity_nodes(G, period_data, result)

    if migration_df is not None:
        _wire_migration_edges(G, migration_df, result)

    result.n_nodes = G.number_of_nodes()
    result.n_edges = G.number_of_edges()
    return result


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_procrustes_chain(G: nx.DiGraph) -> list:
    transitions = []
    for src, tgt, data in G.edges(data=True):
        if data.get("edge_type") == EdgeType.REGIME_TRANSITION.value:
            transitions.append({
                "from":             G.nodes[src].get("label", src),
                "to":               G.nodes[tgt].get("label", tgt),
                "procrustes_score": data.get("procrustes_disparity", 0.0),
                "common_tickers":   data.get("common_tickers", 0),
                "is_major_break":   data.get("is_major_break", False),
                "data_source":      data.get("data_source", "unknown"),
            })
    return sorted(transitions, key=lambda x: x["procrustes_score"], reverse=True)


def get_structural_summary(G: nx.DiGraph) -> dict:
    node_type_counts: dict = {}
    for _, attrs in G.nodes(data=True):
        nt = attrs.get("node_type", "unknown")
        node_type_counts[nt] = node_type_counts.get(nt, 0) + 1
    edge_type_counts: dict = {}
    for _, _, attrs in G.edges(data=True):
        et = attrs.get("edge_type", "unknown")
        edge_type_counts[et] = edge_type_counts.get(et, 0) + 1
    return {
        "total_nodes":      G.number_of_nodes(),
        "total_edges":      G.number_of_edges(),
        "node_type_counts": node_type_counts,
        "edge_type_counts": edge_type_counts,
        "procrustes_chain": get_procrustes_chain(G),
    }


# ── Standalone validation ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[kg_builder] Phase 2 validation\n")
    result = build_static_ontology_graph()
    print(result.summary())
    summary = get_structural_summary(result.graph)
    print("\nNode types:")
    for nt, count in sorted(summary["node_type_counts"].items(), key=lambda x: -x[1]):
        print(f"  {nt:<30} {count:>4}")
    print("\nEdge types:")
    for et, count in sorted(summary["edge_type_counts"].items(), key=lambda x: -x[1]):
        print(f"  {et:<35} {count:>4}")
    print("\nProcrustes chain:")
    for t in summary["procrustes_chain"]:
        flag = " <- MAJOR BREAK" if t["is_major_break"] else ""
        src  = "(appendix B fallback)" if t["data_source"] == "appendix_b_fallback" else "(live)"
        print(f"  {t['from']:<20} -> {t['to']:<20}  score={t['procrustes_score']:.3f}{flag}  {src}")
    assert result.n_nodes >= 15, f"Expected >= 15 nodes, got {result.n_nodes}"
    assert result.n_edges > 0, "Expected structural edges"
    print(f"\n[OK] {result.n_nodes} nodes, {result.n_edges} edges — Phase 2 complete.")
