"""
kg_builder.py  —  ESDS Knowledge Graph: Phase 2
================================================
Constructs a NetworkX knowledge graph from kg_schema.py canonical catalogs.
Imports only names that actually exist in kg_schema.py.

Exported symbols used by kg_visualizer.py:
    build_static_ontology_graph() -> KGResult
    build_kg(period_data, migration_df, include_equity_nodes) -> KGResult
    KGResult
    get_structural_summary(G) -> dict

Standalone validation:
    python kg_builder.py
"""

from __future__ import annotations

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
import pandas as pd

# ── Phase 1 ontology import (only real exports) ───────────────────────────────
try:
    from kg_schema import (
        REGIME_NODES,
        FACTOR_NODES,
        QUADRANT_NODES,
        AXIS_NODES,
        CATEGORY_NODES,
        CROWDING_NODES,
        TRANSITION_NODES,
        STRUCTURAL_BREAK_NODES,
        EARLY_WARNING_NODES,
        APPENDIX_B_PROCRUSTES,
        APPENDIX_B_CROWDING,
        APPENDIX_B_PC_VARIANCE,
        RegimeName,
        QuadrantID,
        RiskLevel,
        StructuralBreakSeverity,
    )
    KG_SCHEMA_AVAILABLE = True
except Exception as e:
    KG_SCHEMA_AVAILABLE = False
    print(f"[kg_builder] Warning: kg_schema import failed: {e}")


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
            "── ESDS Knowledge Graph ────────────────────────────────────────",
            f"  Nodes          : {self.n_nodes:,}",
            f"  Edges          : {self.n_edges:,}",
            f"  Tickers        : {self.n_tickers:,}",
            f"  Clusters       : {self.n_clusters:,}",
            f"  Regimes wired  : {', '.join(self.regime_coverage) or 'static only'}",
        ]
        for w in self.warnings[:5]:
            lines.append(f"  ⚠  {w}")
        lines.append("─" * 64)
        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ── Phase A: static nodes ─────────────────────────────────────────────────────

def _wire_static_nodes(G: nx.DiGraph) -> None:
    """Add all static nodes from kg_schema canonical catalogs."""

    for key, node in REGIME_NODES.items():
        G.add_node(
            node.node_id,
            label=node.name.value,
            node_type="regime",
            start_date=node.start_date,
            end_date=node.end_date,
            universe_count=node.universe_count,
            crowding_score=node.crowding_score,
            is_break_from_prior=node.is_break_from_prior,
        )

    for code, node in FACTOR_NODES.items():
        G.add_node(
            node.node_id,
            label=node.display_name,
            node_type="factor",
            category=node.category.value,
            data_source=node.data_source,
            description=node.description,
        )

    for qid, node in QUADRANT_NODES.items():
        G.add_node(
            node.node_id,
            label=f"{qid}: {node.name}",
            node_type="quadrant",
            pc1_sign=node.pc1_sign,
            pc2_sign=node.pc2_sign,
            description=node.description,
        )

    for pc, node in AXIS_NODES.items():
        G.add_node(
            node.node_id,
            label=f"{pc}: {node.name}",
            node_type="axis",
            variance_explained=node.variance_explained,
            high_meaning=node.high_meaning,
            low_meaning=node.low_meaning,
        )

    for cat, node in CATEGORY_NODES.items():
        G.add_node(
            node.node_id,
            label=cat,
            node_type="category",
            members=list(node.members),
        )

    for regime, node in CROWDING_NODES.items():
        G.add_node(
            node.node_id,
            label=f"Crowding: {regime}",
            node_type="crowding",
            score=node.score,
            risk_level=node.risk_level.value,
            largest_cluster_pct=node.largest_cluster_pct,
            centroid_dispersion=node.centroid_dispersion,
        )

    for (r_from, r_to), node in TRANSITION_NODES.items():
        G.add_node(
            node.node_id,
            label=f"{r_from} → {r_to}",
            node_type="transition",
            procrustes_disparity=node.procrustes_disparity,
            common_ticker_count=node.common_ticker_count,
            migration_pct=node.migration_pct,
            severity=node.severity.value,
            interpretation=node.interpretation,
        )

    for key, node in STRUCTURAL_BREAK_NODES.items():
        G.add_node(
            node.node_id,
            label=f"Major Break: {key}",
            node_type="structural_break",
            disparity=node.disparity,
            severity=node.severity.value,
            recalibration_flag=node.recalibration_flag,
        )

    for regime, node in EARLY_WARNING_NODES.items():
        G.add_node(
            node.node_id,
            label=f"Warning: {regime}",
            node_type="early_warning",
            triggered=node.triggered,
            crowding_elevated=node.crowding_elevated,
            procrustes_elevated=node.procrustes_elevated,
            composite_risk_level=node.composite_risk_level.value,
            alert_description=node.alert_description,
        )


# ── Phase B: structural edges ─────────────────────────────────────────────────

def _wire_structural_edges(G: nx.DiGraph) -> None:

    # Regime → Crowding
    for regime in CROWDING_NODES:
        regime_id   = f"regime:{regime}"
        crowding_id = f"crowding:{regime}"
        if G.has_node(regime_id) and G.has_node(crowding_id):
            score = APPENDIX_B_CROWDING[regime]["score"]
            risk  = APPENDIX_B_CROWDING[regime]["risk_level"].value
            G.add_edge(regime_id, crowding_id,
                       edge_type="crowding_level",
                       score=score,
                       label=f"{score} {risk}")

    # Regime → Regime (Procrustes)
    for (r_from, r_to), node in TRANSITION_NODES.items():
        src = f"regime:{r_from}"
        tgt = f"regime:{r_to}"
        if G.has_node(src) and G.has_node(tgt):
            G.add_edge(src, tgt,
                       edge_type="regime_transition",
                       procrustes_disparity=node.procrustes_disparity,
                       common_tickers=node.common_ticker_count,
                       migration_pct=node.migration_pct,
                       severity=node.severity.value,
                       label=f"Procrustes {node.procrustes_disparity:.3f}",
                       is_major_break=node.procrustes_disparity >= 0.30)

    # Factor → Axis
    pc_factor_map = {
        "axis:PC1": ["factor:roa", "factor:cash_debt", "factor:roe", "factor:gprof"],
        "axis:PC2": ["factor:sales_to_price", "factor:bm", "factor:earnings_yield"],
        "axis:PC3": ["factor:debt_assets", "factor:vol_60d_ann", "factor:gprof"],
    }
    for axis_id, factor_ids in pc_factor_map.items():
        for factor_id in factor_ids:
            if G.has_node(factor_id) and G.has_node(axis_id):
                G.add_edge(factor_id, axis_id,
                           edge_type="factor_loading",
                           label=axis_id.replace("axis:", ""))

    # Factor → Category
    for cat, node in CATEGORY_NODES.items():
        cat_id = node.node_id
        for member_code in node.members:
            factor_id = f"factor:{member_code}"
            if G.has_node(factor_id) and G.has_node(cat_id):
                G.add_edge(factor_id, cat_id, edge_type="belongs_to_category")

    # Transition → Structural break
    for key, break_node in STRUCTURAL_BREAK_NODES.items():
        if G.has_node(break_node.node_id) and G.has_node(break_node.transition_node_id):
            G.add_edge(break_node.transition_node_id, break_node.node_id,
                       edge_type="triggers_break", label="major break")

    # Regime → Early warning
    for regime, warning_node in EARLY_WARNING_NODES.items():
        regime_id = f"regime:{regime}"
        if G.has_node(regime_id) and G.has_node(warning_node.node_id):
            G.add_edge(regime_id, warning_node.node_id,
                       edge_type="triggers_warning", label="elevated risk")


# ── Phase C: dynamic equity nodes ─────────────────────────────────────────────

def _wire_equity_nodes(G, period_data, result):
    ticker_seen = set()
    cluster_seen = set()
    n_clusters = 0

    for period_label, df in period_data.items():
        regime_id = f"regime:{period_label}"
        if not G.has_node(regime_id) or df is None or df.empty:
            result.warnings.append(f"Skipping '{period_label}' — no data.")
            continue

        df = df.copy()
        if "Quadrant" in df.columns and "quadrant" not in df.columns:
            df["quadrant"] = df["Quadrant"]

        if not {"ticker", "quadrant"}.issubset(df.columns):
            result.warnings.append(f"'{period_label}' missing required columns.")
            continue

        result.regime_coverage.append(period_label)

        for _, row in df.iterrows():
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker:
                continue

            stock_id = f"stock:{ticker}"
            if ticker not in ticker_seen:
                G.add_node(stock_id, label=ticker, node_type="stock",
                           gics_sector=str(row.get("gics_sector", row.get("gicdesc", "Unknown"))))
                ticker_seen.add(ticker)

            raw_q = str(row.get("quadrant", ""))
            q_short = raw_q.split(":")[0].strip()
            quadrant_id = f"quadrant:{q_short}"
            if G.has_node(quadrant_id):
                G.add_edge(stock_id, quadrant_id,
                           edge_type="quadrant_assignment",
                           regime=period_label,
                           pc1=_safe_float(row.get("PC1", 0)),
                           pc2=_safe_float(row.get("PC2", 0)))

            cluster_label = row.get("cluster")
            if cluster_label is not None:
                cluster_id = f"cluster:{period_label}:{cluster_label}"
                if cluster_id not in cluster_seen:
                    G.add_node(cluster_id,
                               label=f"{period_label} Cluster {cluster_label}",
                               node_type="cluster",
                               regime=period_label,
                               cluster_label=str(cluster_label))
                    G.add_edge(cluster_id, regime_id, edge_type="belongs_to")
                    cluster_seen.add(cluster_id)
                    n_clusters += 1
                G.add_edge(stock_id, cluster_id,
                           edge_type="cluster_membership", regime=period_label)

    result.n_tickers = len(ticker_seen)
    result.n_clusters = n_clusters


# ── Phase D: quadrant migration edges ─────────────────────────────────────────

def _wire_migration_edges(G, migration_df, result):
    if migration_df is None or migration_df.empty:
        result.warnings.append("No migration_df — migration edges skipped.")
        return

    period_cols = [c for c in ["Post-COVID", "Rate Shock", "Disinflation"]
                   if c in migration_df.columns]

    for i in range(len(period_cols) - 1):
        col_from, col_to = period_cols[i], period_cols[i + 1]
        sub = migration_df[[col_from, col_to]].dropna()
        grouped = sub.groupby([col_from, col_to]).size().reset_index(name="count")
        for _, row in grouped.iterrows():
            q_from = str(row[col_from]).split(":")[0].strip()
            q_to   = str(row[col_to]).split(":")[0].strip()
            src = f"quadrant:{q_from}"
            tgt = f"quadrant:{q_to}"
            if G.has_node(src) and G.has_node(tgt):
                G.add_edge(src, tgt,
                           edge_type="migrates_to",
                           count=int(row["count"]),
                           from_period=col_from,
                           to_period=col_to)


# ── Master build functions ────────────────────────────────────────────────────

def build_static_ontology_graph() -> KGResult:
    """Build static ESDS ontology (no equity nodes). Used by kg_visualizer.py."""
    if not KG_SCHEMA_AVAILABLE:
        return KGResult(graph=nx.DiGraph(), warnings=["kg_schema not available"])
    G = nx.DiGraph(name="ESDS_Knowledge_Graph", version="2.1")
    result = KGResult(graph=G)
    _wire_static_nodes(G)
    _wire_structural_edges(G)
    result.n_nodes = G.number_of_nodes()
    result.n_edges = G.number_of_edges()
    return result


def build_kg(
    period_data: Optional[dict] = None,
    migration_df=None,
    include_equity_nodes: bool = True,
) -> KGResult:
    """Build full ESDS KG with optional equity nodes."""
    if not KG_SCHEMA_AVAILABLE:
        return KGResult(graph=nx.DiGraph(), warnings=["kg_schema not available"])
    G = nx.DiGraph(name="ESDS_Knowledge_Graph", version="2.1")
    result = KGResult(graph=G)
    _wire_static_nodes(G)
    _wire_structural_edges(G)
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
        if data.get("edge_type") == "regime_transition":
            transitions.append({
                "from": G.nodes[src].get("label", src),
                "to":   G.nodes[tgt].get("label", tgt),
                "procrustes_score": data.get("procrustes_disparity", 0.0),
                "common_tickers":   data.get("common_tickers", 0),
                "is_major_break":   data.get("is_major_break", False),
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
        flag = " ← MAJOR BREAK" if t["is_major_break"] else ""
        print(f"  {t['from']:<20} → {t['to']:<20}  score={t['procrustes_score']:.3f}{flag}")
    assert result.n_nodes >= 20, f"Expected ≥20 nodes, got {result.n_nodes}"
    assert result.n_edges > 0, "Expected structural edges"
    print(f"\n✓ {result.n_nodes} nodes, {result.n_edges} edges — Phase 2 complete.")
