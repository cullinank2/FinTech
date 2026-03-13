"""
kg_builder.py  —  ESDS Knowledge Graph: Phase 2
================================================
Constructs a live NetworkX knowledge graph from ESDS pipeline outputs.
Imports the Phase 1 ontology (kg_schema.py) and populates it with:
  - Static institutional nodes (factors, regimes, quadrants, mechanisms)
  - Dynamic equity nodes (tickers, clusters, sector memberships)
  - All typed edges (factor loadings, Procrustes transitions, crowding
    scores, quadrant assignments, cluster memberships, migrations)

Usage (standalone validation):
    python kg_builder.py

Usage (from app.py or period_analysis.py):
    from kg_builder import build_kg, KGResult
    result = build_kg(period_data_dict, migration_df)

Dependencies:
    networkx, pandas, numpy  (all already in ESDS environment)
    kg_schema               (Phase 1 — must be in same directory)
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Ensure kg_schema resolves regardless of kernel/app working directory
sys.path.insert(0, str(Path(__file__).parent))

import networkx as nx
import numpy as np
import pandas as pd

# ── Phase 1 ontology import ───────────────────────────────────────────────────
try:
    from kg_schema import (
        KGNode, KGEdge, NodeType, EdgeType,
        REGIME_NODES, FACTOR_NODES, QUADRANT_NODES, MECHANISM_NODES,
        GOVERNANCE_NODES, MARKET_CONTEXT_NODES, ALL_STATIC_NODES,
        EMPIRICAL_ANCHORS,
        PROCRUSTES_PAIRS, CROWDING_SCORES,
        PC1_VARIANCE, PC2_VARIANCE, PC3_VARIANCE,
    )
except ImportError as e:
    sys.exit(
        f"[kg_builder] Cannot import kg_schema: {e}\n"
        "  Ensure kg_schema.py is in the same directory as kg_builder.py."
    )

# ── Config: quadrant label mapping ───────────────────────────────────────────
QUADRANT_LABEL_MAP: dict[str, str] = {
    "Q1: Profitable Value":      "Q1",
    "Q2: Value Traps/Distressed":"Q2",
    "Q3: Struggling Growth":     "Q3",
    "Q4: Quality Growth":        "Q4",
}

# ── Factor → PC contribution weights (from Appendix B ground truth) ──────────
# Format: factor_id -> {PC1_loading, PC2_loading, PC3_loading}
# Sign conventions: Earnings Yield reverses to negative on PC2 in Disinflation
FACTOR_PC_WEIGHTS: dict[str, dict[str, dict[str, float]]] = {
    "post_covid": {
        "book_to_market":        {"PC1": +0.42, "PC2": +0.38, "PC3": -0.15},
        "earnings_yield":        {"PC1": +0.38, "PC2": +0.35, "PC3": -0.10},
        "sales_to_price":        {"PC1": +0.40, "PC2": +0.32, "PC3": -0.08},
        "debt_to_assets":        {"PC1": -0.28, "PC2": -0.05, "PC3": -0.25},
        "gross_profitability":   {"PC1": -0.30, "PC2": +0.10, "PC3": +0.36},
        "asset_turnover":        {"PC1": -0.18, "PC2": +0.12, "PC3": +0.38},
        "accruals":              {"PC1": +0.12, "PC2": -0.55, "PC3": +0.15},
        "net_stock_issuance":    {"PC1": +0.08, "PC2": -0.48, "PC3": +0.12},
        "return_on_equity":      {"PC1": -0.25, "PC2": +0.22, "PC3": +0.42},
        "revenue_growth":        {"PC1": -0.20, "PC2": +0.18, "PC3": +0.35},
        "price_momentum":        {"PC1": -0.15, "PC2": -0.10, "PC3": +0.28},
    },
    "rate_shock": {
        "book_to_market":        {"PC1": +0.45, "PC2": +0.40, "PC3": -0.12},
        "earnings_yield":        {"PC1": +0.41, "PC2": +0.38, "PC3": -0.08},
        "sales_to_price":        {"PC1": +0.43, "PC2": +0.35, "PC3": -0.06},
        "debt_to_assets":        {"PC1": -0.30, "PC2": -0.08, "PC3": -0.22},
        "gross_profitability":   {"PC1": -0.32, "PC2": +0.12, "PC3": +0.34},
        "asset_turnover":        {"PC1": -0.20, "PC2": +0.14, "PC3": +0.36},
        "accruals":              {"PC1": +0.10, "PC2": -0.52, "PC3": +0.18},
        "net_stock_issuance":    {"PC1": +0.06, "PC2": -0.46, "PC3": +0.14},
        "return_on_equity":      {"PC1": -0.28, "PC2": +0.20, "PC3": +0.44},
        "revenue_growth":        {"PC1": -0.22, "PC2": +0.16, "PC3": +0.38},
        "price_momentum":        {"PC1": -0.18, "PC2": -0.08, "PC3": +0.30},
    },
    "disinflation": {
        "book_to_market":        {"PC1": +0.44, "PC2": +0.35, "PC3": -0.14},
        "earnings_yield":        {"PC1": +0.39, "PC2": -0.28, "PC3": -0.09},  # sign reversal on PC2
        "sales_to_price":        {"PC1": +0.42, "PC2": +0.33, "PC3": -0.07},
        "debt_to_assets":        {"PC1": -0.31, "PC2": -0.07, "PC3": -0.24},
        "gross_profitability":   {"PC1": -0.31, "PC2": +0.15, "PC3": +0.35},
        "asset_turnover":        {"PC1": -0.19, "PC2": +0.16, "PC3": +0.37},
        "accruals":              {"PC1": +0.11, "PC2": -0.50, "PC3": +0.20},
        "net_stock_issuance":    {"PC1": +0.07, "PC2": -0.44, "PC3": +0.16},
        "return_on_equity":      {"PC1": -0.27, "PC2": +0.22, "PC3": +0.46},
        "revenue_growth":        {"PC1": -0.21, "PC2": +0.20, "PC3": +0.40},
        "price_momentum":        {"PC1": -0.16, "PC2": -0.06, "PC3": +0.32},
    },
}

# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class KGResult:
    """Returned by build_kg(); provides graph + summary statistics."""
    graph: nx.DiGraph
    n_nodes: int = 0
    n_edges: int = 0
    n_tickers: int = 0
    n_clusters: int = 0
    regime_coverage: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "── ESDS Knowledge Graph (Phase 2) ──────────────────────────────",
            f"  Nodes          : {self.n_nodes:,}",
            f"  Edges          : {self.n_edges:,}",
            f"  Tickers        : {self.n_tickers:,}",
            f"  Clusters       : {self.n_clusters:,}",
            f"  Regimes wired  : {', '.join(self.regime_coverage)}",
        ]
        if self.warnings:
            lines.append(f"  Warnings       : {len(self.warnings)}")
            for w in self.warnings[:5]:
                lines.append(f"    ⚠  {w}")
        lines.append("─" * 64)
        return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _add_node(G: nx.DiGraph, node: KGNode) -> None:
    """Add a KGNode to the graph, converting dataclass to dict attrs."""
    G.add_node(node.id, **{
        "label":       node.label,
        "node_type":   node.node_type.value,
        "description": node.description,
        **node.properties,
    })


def _add_edge(G: nx.DiGraph, edge: KGEdge) -> None:
    """Add a KGEdge to the graph."""
    G.add_edge(edge.source, edge.target, **{
        "edge_type": edge.edge_type.value,
        **edge.properties,
    })


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ── Phase A: populate static ontology nodes ───────────────────────────────────

def _wire_static_nodes(G: nx.DiGraph) -> None:
    """Add all 35 static nodes from Phase 1 schema."""
    for node in ALL_STATIC_NODES:
        _add_node(G, node)


# ── Phase B: wire static structural edges ─────────────────────────────────────

def _wire_structural_edges(G: nx.DiGraph) -> None:
    """
    Wire edges between static nodes:
      - Regime → PC variance nodes
      - Factor → Domain nodes
      - Mechanism → Regime nodes (operates_in)
      - Procrustes pair edges (regime_transition)
      - Crowding score edges (crowding_level)
    """
    # ── Regime → PC variance explained ──
    regime_pc_variance = {
        "regime_post_covid": {
            "pc1_axis": PC1_VARIANCE,
            "pc2_axis": PC2_VARIANCE,
            "pc3_axis": PC3_VARIANCE,
        },
        "regime_rate_shock": {
            "pc1_axis": PC1_VARIANCE,
            "pc2_axis": PC2_VARIANCE,
            "pc3_axis": PC3_VARIANCE,
        },
        "regime_disinflation": {
            "pc1_axis": PC1_VARIANCE,
            "pc2_axis": PC2_VARIANCE,
            "pc3_axis": PC3_VARIANCE,
        },
    }
    for regime_id, pc_map in regime_pc_variance.items():
        for pc_id, variance in pc_map.items():
            if G.has_node(regime_id) and G.has_node(pc_id):
                _add_edge(G, KGEdge(
                    source=regime_id, target=pc_id,
                    edge_type=EdgeType.STRUCTURAL_RELATIONSHIP,
                    properties={"variance_explained": variance, "label": "pca_coordinate"},
                ))

    # ── Factor → PC loading (per regime) ──
    factor_id_map = {
        "book_to_market":     "factor_book_to_market",
        "earnings_yield":     "factor_earnings_yield",
        "sales_to_price":     "factor_sales_to_price",
        "debt_to_assets":     "factor_debt_to_assets",
        "gross_profitability":"factor_gross_profitability",
        "asset_turnover":     "factor_asset_turnover",
        "accruals":           "factor_accruals",
        "net_stock_issuance": "factor_net_stock_issuance",
        "return_on_equity":   "factor_return_on_equity",
        "revenue_growth":     "factor_revenue_growth",
        "price_momentum":     "factor_price_momentum",
    }
    regime_key_map = {
        "regime_post_covid":    "post_covid",
        "regime_rate_shock":    "rate_shock",
        "regime_disinflation":  "disinflation",
    }
    pc_node_map = {"PC1": "pc1_axis", "PC2": "pc2_axis", "PC3": "pc3_axis"}

    for regime_node_id, regime_key in regime_key_map.items():
        weights = FACTOR_PC_WEIGHTS.get(regime_key, {})
        for factor_key, pc_loadings in weights.items():
            factor_node_id = factor_id_map.get(factor_key)
            if factor_node_id and G.has_node(factor_node_id):
                for pc_label, loading in pc_loadings.items():
                    pc_node_id = pc_node_map.get(pc_label)
                    if pc_node_id and G.has_node(pc_node_id):
                        _add_edge(G, KGEdge(
                            source=factor_node_id, target=pc_node_id,
                            edge_type=EdgeType.FACTOR_LOADING,
                            properties={
                                "regime": regime_node_id,
                                "loading": loading,
                                "abs_loading": abs(loading),
                                "direction": "positive" if loading > 0 else "negative",
                                "label": f"{factor_key}→{pc_label}",
                            },
                        ))

    # ── Procrustes regime-transition edges ──
    for pair in PROCRUSTES_PAIRS:
        src = pair["source_regime"]
        tgt = pair["target_regime"]
        if G.has_node(src) and G.has_node(tgt):
            interpretation = pair.get("interpretation", "")
            _add_edge(G, KGEdge(
                source=src, target=tgt,
                edge_type=EdgeType.REGIME_TRANSITION,
                properties={
                    "procrustes_score":   pair["procrustes_score"],
                    "common_tickers":     pair["common_tickers"],
                    "interpretation":     interpretation,
                    "label": f"procrustes_{pair['procrustes_score']}",
                    "is_major_break": pair["procrustes_score"] >= 0.40,
                },
            ))

    # ── Crowding score edges (regime → crowding_mechanism) ──
    crowding_node_id = "mechanism_crowding"
    for cs in CROWDING_SCORES:
        regime_id = cs["regime"]
        if G.has_node(regime_id) and G.has_node(crowding_node_id):
            _add_edge(G, KGEdge(
                source=regime_id, target=crowding_node_id,
                edge_type=EdgeType.CROWDING_LEVEL,
                properties={
                    "score":             cs["score"],
                    "label":             cs["label"],
                    "is_elevated":       cs["score"] >= 60,
                    "label_str": f"crowding_{cs['label'].lower().replace(' ', '_')}",
                },
            ))


# ── Phase C: dynamic equity nodes from pipeline data ─────────────────────────

def _wire_equity_nodes(
    G: nx.DiGraph,
    period_data: dict[str, pd.DataFrame],
    result: KGResult,
) -> None:
    """
    Build dynamic Stock and Cluster nodes from period pipeline DataFrames.

    Expected DataFrame columns (from ESDS utils.py output):
        ticker, gics_sector, quadrant, cluster_label,
        pc1_score, pc2_score, [pc3_score]

    period_data: dict mapping regime key → processed DataFrame
        keys should be: 'Post-COVID', 'Rate Shock', 'Disinflation'
    """
    regime_key_map = {
        "Post-COVID":   "regime_post_covid",
        "Rate Shock":   "regime_rate_shock",
        "Disinflation": "regime_disinflation",
    }

    ticker_seen: set[str] = set()
    cluster_seen: set[str] = set()
    n_clusters = 0

    for period_label, df in period_data.items():
        regime_node_id = regime_key_map.get(period_label)
        if regime_node_id is None:
            result.warnings.append(
                f"Unknown period label '{period_label}' — skipping equity wiring."
            )
            continue
        if df is None or df.empty:
            result.warnings.append(f"Empty DataFrame for period '{period_label}' — skipping.")
            continue

        required_cols = {"ticker", "quadrant"}
        missing = required_cols - set(df.columns)
        if missing:
            result.warnings.append(
                f"Period '{period_label}' DataFrame missing columns: {missing} — skipping."
            )
            continue

        result.regime_coverage.append(period_label)

        for _, row in df.iterrows():
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker:
                continue

            # ── Stock node (once per ticker globally) ──
            if ticker not in ticker_seen:
                sector = str(row.get("gics_sector", "Unknown"))
                G.add_node(
                    f"stock_{ticker}",
                    label=ticker,
                    node_type=NodeType.STOCK.value,
                    gics_sector=sector,
                    description=f"Equity: {ticker} ({sector})",
                )
                ticker_seen.add(ticker)

            stock_node_id = f"stock_{ticker}"

            # ── Quadrant assignment edge (stock → quadrant, per regime) ──
            raw_quadrant = str(row.get("quadrant", ""))
            q_key = QUADRANT_LABEL_MAP.get(raw_quadrant, raw_quadrant)
            quadrant_node_id = f"quadrant_{q_key.lower().replace(': ', '_').replace(' ', '_')}"
            if not G.has_node(quadrant_node_id):
                quadrant_node_id = f"quadrant_{q_key}"  # fallback

            if G.has_node(quadrant_node_id):
                _add_edge(G, KGEdge(
                    source=stock_node_id, target=quadrant_node_id,
                    edge_type=EdgeType.QUADRANT_ASSIGNMENT,
                    properties={
                        "regime": regime_node_id,
                        "period": period_label,
                        "pc1_score": _safe_float(row.get("pc1_score")),
                        "pc2_score": _safe_float(row.get("pc2_score")),
                        "pc3_score": _safe_float(row.get("pc3_score", 0.0)),
                    },
                ))

            # ── Cluster node + membership edge ──
            cluster_label = row.get("cluster_label") or row.get("cluster")
            if cluster_label is not None:
                cluster_node_id = f"cluster_{period_label.lower().replace(' ', '_')}_{cluster_label}"
                if cluster_node_id not in cluster_seen:
                    G.add_node(
                        cluster_node_id,
                        label=f"{period_label} Cluster {cluster_label}",
                        node_type=NodeType.CLUSTER.value,
                        regime=regime_node_id,
                        period=period_label,
                        cluster_id=str(cluster_label),
                        description=f"KMeans cluster {cluster_label} in {period_label}",
                    )
                    # Cluster belongs_to regime
                    if G.has_node(regime_node_id):
                        _add_edge(G, KGEdge(
                            source=cluster_node_id, target=regime_node_id,
                            edge_type=EdgeType.BELONGS_TO,
                            properties={"label": "cluster_in_regime"},
                        ))
                    cluster_seen.add(cluster_node_id)
                    n_clusters += 1

                _add_edge(G, KGEdge(
                    source=stock_node_id, target=cluster_node_id,
                    edge_type=EdgeType.CLUSTER_MEMBERSHIP,
                    properties={
                        "regime": regime_node_id,
                        "period": period_label,
                    },
                ))

    result.n_tickers = len(ticker_seen)
    result.n_clusters = n_clusters


# ── Phase D: quadrant migration edges ─────────────────────────────────────────

def _wire_migration_edges(
    G: nx.DiGraph,
    migration_df: Optional[pd.DataFrame],
    result: KGResult,
) -> None:
    """
    Add MIGRATES_TO edges between quadrant nodes based on migration_df.

    migration_df: long-form DataFrame with columns:
        ticker, period_from, period_to, quadrant_from, quadrant_to
    OR wide-form with columns:
        ticker, Post-COVID, Rate Shock, Disinflation
    """
    if migration_df is None or migration_df.empty:
        result.warnings.append("No migration_df provided — migration edges skipped.")
        return

    # ── Detect long vs. wide format ──
    has_long = {"period_from", "period_to", "quadrant_from", "quadrant_to"}.issubset(
        migration_df.columns
    )
    has_wide = {"Post-COVID", "Rate Shock", "Disinflation"}.issubset(migration_df.columns)

    if has_long:
        _wire_migration_long(G, migration_df, result)
    elif has_wide:
        _wire_migration_wide(G, migration_df, result)
    else:
        result.warnings.append(
            "migration_df columns not recognized — expected long or wide format. "
            f"Got: {list(migration_df.columns)[:8]}"
        )


def _quadrant_node_id(label: str) -> str:
    """Normalize quadrant label to node id."""
    short = QUADRANT_LABEL_MAP.get(label, label)
    return f"quadrant_{short.lower()}"


def _wire_migration_long(
    G: nx.DiGraph,
    df: pd.DataFrame,
    result: KGResult,
) -> None:
    grouped = df.groupby(["quadrant_from", "quadrant_to"]).size().reset_index(name="count")
    for _, row in grouped.iterrows():
        src = _quadrant_node_id(row["quadrant_from"])
        tgt = _quadrant_node_id(row["quadrant_to"])
        if G.has_node(src) and G.has_node(tgt):
            _add_edge(G, KGEdge(
                source=src, target=tgt,
                edge_type=EdgeType.MIGRATES_TO,
                properties={"count": int(row["count"]), "label": "quadrant_migration"},
            ))


def _wire_migration_wide(
    G: nx.DiGraph,
    df: pd.DataFrame,
    result: KGResult,
) -> None:
    period_cols = ["Post-COVID", "Rate Shock", "Disinflation"]
    available = [c for c in period_cols if c in df.columns]
    for i in range(len(available) - 1):
        col_from = available[i]
        col_to = available[i + 1]
        sub = df[[col_from, col_to]].dropna()
        grouped = sub.groupby([col_from, col_to]).size().reset_index(name="count")
        for _, row in grouped.iterrows():
            src = _quadrant_node_id(row[col_from])
            tgt = _quadrant_node_id(row[col_to])
            if G.has_node(src) and G.has_node(tgt):
                _add_edge(G, KGEdge(
                    source=src, target=tgt,
                    edge_type=EdgeType.MIGRATES_TO,
                    properties={
                        "count":       int(row["count"]),
                        "from_period": col_from,
                        "to_period":   col_to,
                        "label":       "quadrant_migration",
                    },
                ))


# ── Master build function ─────────────────────────────────────────────────────

def build_kg(
    period_data: Optional[dict[str, pd.DataFrame]] = None,
    migration_df: Optional[pd.DataFrame] = None,
    include_equity_nodes: bool = True,
) -> KGResult:
    """
    Build the ESDS Knowledge Graph.

    Parameters
    ----------
    period_data : dict[str, pd.DataFrame], optional
        Mapping of period label → processed DataFrame from ESDS pipeline.
        Keys: 'Post-COVID', 'Rate Shock', 'Disinflation'
        If None, builds schema-only graph (static nodes + structural edges).

    migration_df : pd.DataFrame, optional
        Migration data in long or wide format (see _wire_migration_edges).

    include_equity_nodes : bool
        If False, skip dynamic stock/cluster nodes (useful for fast structural
        queries or when period_data is unavailable).

    Returns
    -------
    KGResult
        Container with graph, statistics, and any build warnings.
    """
    G = nx.DiGraph(
        name="ESDS_Knowledge_Graph",
        version="2.0",
        description="Equity Structural Diagnostics System — full knowledge graph",
        empirical_source="Appendix B (authoritative ground truth)",
    )

    result = KGResult(graph=G)

    # Phase A: static ontology nodes
    _wire_static_nodes(G)

    # Phase B: structural edges between static nodes
    _wire_structural_edges(G)

    # Phase C: dynamic equity nodes (optional)
    if include_equity_nodes and period_data:
        _wire_equity_nodes(G, period_data, result)

    # Phase D: migration edges (optional)
    if migration_df is not None:
        _wire_migration_edges(G, migration_df, result)

    # Final statistics
    result.n_nodes = G.number_of_nodes()
    result.n_edges = G.number_of_edges()

    return result


# ── Query helpers (importable by app.py) ─────────────────────────────────────

def get_ticker_regimes(G: nx.DiGraph, ticker: str) -> dict[str, str]:
    """
    Return {period_label: quadrant} for a given ticker across all regimes.
    Returns empty dict if ticker not in graph.
    """
    stock_id = f"stock_{ticker.upper()}"
    if stock_id not in G:
        return {}
    result = {}
    for _, tgt, data in G.out_edges(stock_id, data=True):
        if data.get("edge_type") == EdgeType.QUADRANT_ASSIGNMENT.value:
            period = data.get("period", "Unknown")
            quadrant_label = G.nodes[tgt].get("label", tgt)
            result[period] = quadrant_label
    return result


def get_regime_crowding(G: nx.DiGraph, regime_id: str) -> Optional[dict]:
    """
    Return crowding score dict for a regime, or None if not found.
    regime_id: 'regime_post_covid' | 'regime_rate_shock' | 'regime_disinflation'
    """
    crowding_node = "mechanism_crowding"
    if not G.has_edge(regime_id, crowding_node):
        return None
    return dict(G[regime_id][crowding_node])


def get_procrustes_chain(G: nx.DiGraph) -> list[dict]:
    """
    Return list of all regime-transition edges with Procrustes scores,
    sorted by score descending (largest structural break first).
    """
    transitions = []
    for src, tgt, data in G.edges(data=True):
        if data.get("edge_type") == EdgeType.REGIME_TRANSITION.value:
            transitions.append({
                "from": G.nodes[src].get("label", src),
                "to":   G.nodes[tgt].get("label", tgt),
                "procrustes_score": data.get("procrustes_score", 0.0),
                "common_tickers":   data.get("common_tickers", 0),
                "interpretation":   data.get("interpretation", ""),
                "is_major_break":   data.get("is_major_break", False),
            })
    return sorted(transitions, key=lambda x: x["procrustes_score"], reverse=True)


def get_structural_summary(G: nx.DiGraph) -> dict:
    """
    High-level structural summary for dashboard or narrative engine consumption.
    """
    node_type_counts: dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        nt = attrs.get("node_type", "unknown")
        node_type_counts[nt] = node_type_counts.get(nt, 0) + 1

    edge_type_counts: dict[str, int] = {}
    for _, _, attrs in G.edges(data=True):
        et = attrs.get("edge_type", "unknown")
        edge_type_counts[et] = edge_type_counts.get(et, 0) + 1

    return {
        "total_nodes":       G.number_of_nodes(),
        "total_edges":       G.number_of_edges(),
        "node_type_counts":  node_type_counts,
        "edge_type_counts":  edge_type_counts,
        "procrustes_chain":  get_procrustes_chain(G),
        "empirical_anchors": EMPIRICAL_ANCHORS,
    }


# ── Standalone validation ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[kg_builder] Phase 2 validation — schema-only build (no period data)\n")

    result = build_kg(period_data=None, migration_df=None)
    print(result.summary())

    summary = get_structural_summary(result.graph)
    print("\nNode type breakdown:")
    for nt, count in sorted(summary["node_type_counts"].items(), key=lambda x: -x[1]):
        print(f"  {nt:<30} {count:>4}")

    print("\nEdge type breakdown:")
    for et, count in sorted(summary["edge_type_counts"].items(), key=lambda x: -x[1]):
        print(f"  {et:<35} {count:>4}")

    print("\nProcrustes chain (largest break first):")
    for t in summary["procrustes_chain"]:
        flag = " ← MAJOR BREAK" if t["is_major_break"] else ""
        print(f"  {t['from']:<20} → {t['to']:<20}  score={t['procrustes_score']:.3f}  "
              f"n={t['common_tickers']:,}{flag}")

    # Assertions
    assert result.n_nodes >= 25, f"Expected ≥25 nodes, got {result.n_nodes}"
    assert result.n_edges > 0,   "Expected structural edges to be wired"
    assert len(summary["procrustes_chain"]) == 3, "Expected exactly 3 Procrustes transitions"

    print("\n✓ All Phase 2 assertions passed")
    print(f"✓ {result.n_nodes} nodes, {result.n_edges} edges in schema-only build")
    print("\nPhase 2 complete. Ready for Phase 3 (kg_visualizer.py).")
