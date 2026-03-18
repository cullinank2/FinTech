"""
kg_interface.py  —  ESDS Knowledge Graph: Phase 3 Query Layer
=============================================================
Exposes a clean Python API over the NetworkX DiGraph built by kg_builder.py.

Six public methods:
    get_peers(ticker, regime)                        -> list[dict]
    get_quadrant_history(ticker)                     -> list[dict]
    get_factor_rotation(factor, from_regime, to_regime) -> dict
    query_crowding_chain(from_regime, to_regime)     -> dict
    get_structural_drift_summary(regime)             -> dict
    serialize_subgraph(node_ids)                     -> dict

Two-tier AI governance is enforced structurally:
    Tier 1 (Narrative Engine): consumes narrative_chain list[dict] from
        query_crowding_chain() — deterministic, no LLM involved.
    Tier 2 (Chatbot): consumes serialize_subgraph() JSON blob —
        bounded context window injection, never a live graph object.

No Streamlit imports in this file. Pure Python — unit-testable standalone.

Standalone validation:
    python kg_interface.py
"""

import math
from typing import Optional

import networkx as nx

from factor_registry import (
    REGIME_ORDER as CANONICAL_REGIME_ORDER,
    FEATURE_COLUMNS,
    FEATURE_LIST,
    EY, BM, SP, ROE, ROA, GPROF,
    DEBT_ASSETS, CASH_DEBT,
    MOMENTUM, VOL, LIQUIDITY,
)

# ── Schema constants (no graph construction imported) ─────────────────────────
try:
    from kg_schema import (
        NodeType, EdgeType,
        PROCRUSTES_MEANINGFUL, CROWDING_THRESHOLD_ELEVATED, CROWDING_THRESHOLD_HIGH,
        APPENDIX_B_PROCRUSTES, APPENDIX_B_CROWDING,
        RiskLevel, StructuralBreakSeverity,
    )
    KG_SCHEMA_AVAILABLE = True
except Exception as _e:
    KG_SCHEMA_AVAILABLE = False
    print(f"[kg_interface] kg_schema import failed: {_e}")
    # Minimal fallback constants so the class still operates
    PROCRUSTES_MEANINGFUL       = 0.30
    CROWDING_THRESHOLD_ELEVATED = 50.0
    CROWDING_THRESHOLD_HIGH     = 70.0
    APPENDIX_B_PROCRUSTES       = {}
    APPENDIX_B_CROWDING         = {}


# ── Regime ordering (chronological) ──────────────────────────────────────────

# Use canonical regime ordering from registry

# Appendix B per-period loading fallbacks for get_factor_rotation()
# Used ONLY when session_state["period_loadings"] is not yet populated.
# Values: loading on the PC axis that shows the most dramatic shift.
_APPENDIX_B_LOADINGS = {
    # earnings_yield: sign reversal on PC2 — the headline example
    ("earnings_yield", "Post-COVID"):   {"PC1":  0.150, "PC2":  0.211, "PC3": -0.050},
    ("earnings_yield", "Rate Shock"):   {"PC1":  0.130, "PC2":  0.095, "PC3": -0.040},
    ("earnings_yield", "Disinflation"): {"PC1":  0.110, "PC2": -0.315, "PC3": -0.030},
    # debt_assets: sign reversal on PC3 — most dramatic structural event
    ("debt_assets", "Post-COVID"):      {"PC1": -0.080, "PC2":  0.050, "PC3": -0.248},
    ("debt_assets", "Rate Shock"):      {"PC1": -0.060, "PC2":  0.070, "PC3":  0.210},
    ("debt_assets", "Disinflation"):    {"PC1": -0.040, "PC2":  0.090, "PC3":  0.770},
    # bm: dominant positive PC2 driver — stable throughout
    ("bm", "Post-COVID"):               {"PC1": -0.100, "PC2":  0.380, "PC3":  0.050},
    ("bm", "Rate Shock"):               {"PC1": -0.090, "PC2":  0.360, "PC3":  0.060},
    ("bm", "Disinflation"):             {"PC1": -0.080, "PC2":  0.350, "PC3":  0.070},
    # sales_to_price: stable positive PC2
    ("sales_to_price", "Post-COVID"):   {"PC1": -0.090, "PC2":  0.310, "PC3":  0.040},
    ("sales_to_price", "Rate Shock"):   {"PC1": -0.080, "PC2":  0.295, "PC3":  0.050},
    ("sales_to_price", "Disinflation"): {"PC1": -0.070, "PC2":  0.300, "PC3":  0.045},
}

# ── VALIDATION: ensure Appendix B factors align with registry ───────────────
for (factor, _regime) in _APPENDIX_B_LOADINGS.keys():
    if factor not in FEATURE_LIST:
        raise ValueError(
            f"[kg_interface] Unknown factor '{factor}' in _APPENDIX_B_LOADINGS — "
            "not found in FEATURE_LIST (factor_registry.py)"
        )


# =============================================================================
# HELPERS
# =============================================================================

def _safe_float(val, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(str(val).replace('%', '').replace(',', '').strip())
    except (TypeError, ValueError):
        return default


def _euclidean(pc1_a: float, pc2_a: float, pc1_b: float, pc2_b: float) -> float:
    return math.sqrt((pc1_a - pc1_b) ** 2 + (pc2_a - pc2_b) ** 2)


def _regime_index(regime: str) -> int:
    try:
        return CANONICAL_REGIME_ORDER.index(regime)
    except ValueError:
        return -1


def _classify_stability(loading_from: float, loading_to: float) -> str:
    """
    Classify a factor loading shift across a regime boundary.
        reversed  : sign change
        rotated   : same sign, |delta| >= 0.15
        stable    : same sign, |delta| < 0.15
    """
    sign_change = (loading_from >= 0) != (loading_to >= 0)
    if sign_change:
        return "reversed"
    if abs(loading_to - loading_from) >= 0.15:
        return "rotated"
    return "stable"


def _risk_label(score: float) -> str:
    if score >= CROWDING_THRESHOLD_HIGH:
        return "High"
    if score >= CROWDING_THRESHOLD_ELEVATED:
        return "Elevated"
    return "Normal"


def _get_session_state() -> dict:
    """Return st.session_state if Streamlit is available, else empty dict."""
    try:
        import streamlit as st
        return st.session_state
    except Exception:
        return {}


# =============================================================================
# KNOWLEDGE GRAPH INTERFACE
# =============================================================================

class KnowledgeGraph:
    """
    Query interface over the ESDS NetworkX DiGraph.

    Construction:
        from kg_builder import build_kg
        result = build_kg(period_data, migration_df)
        kg = KnowledgeGraph(result.graph)

    All methods are pure Python. No Streamlit calls here.
    Session state is accessed only inside get_factor_rotation() for
    per-period loadings, via _get_session_state() which degrades
    gracefully when Streamlit is not available.
    """

    def __init__(self, graph: nx.DiGraph):
        if not isinstance(graph, (nx.DiGraph, nx.Graph)):
            raise TypeError(f"Expected nx.DiGraph, got {type(graph)}")
        self._G = graph
        self._ss = _get_session_state()

    # ── Internal graph accessors ──────────────────────────────────────────────

    def _node_attrs(self, node_id: str) -> dict:
        if self._G.has_node(node_id):
            return dict(self._G.nodes[node_id])
        return {}

    def _out_edges(self, node_id: str, edge_type: str) -> list[tuple]:
        """Return list of (src, tgt, attrs) for outgoing edges of given type."""
        if not self._G.has_node(node_id):
            return []
        return [
            (u, v, d)
            for u, v, d in self._G.out_edges(node_id, data=True)
            if d.get("edge_type") == edge_type
        ]

    def _in_edges(self, node_id: str, edge_type: str) -> list[tuple]:
        """Return list of (src, tgt, attrs) for incoming edges of given type."""
        if not self._G.has_node(node_id):
            return []
        return [
            (u, v, d)
            for u, v, d in self._G.in_edges(node_id, data=True)
            if d.get("edge_type") == edge_type
        ]

    def _nodes_of_type(self, node_type: str) -> list[tuple]:
        """Return list of (node_id, attrs) for all nodes of given type."""
        return [
            (n, d)
            for n, d in self._G.nodes(data=True)
            if d.get("node_type") == node_type
        ]

    # =========================================================================
    # METHOD 1 — get_peers
    # =========================================================================

    def get_peers(
        self,
        ticker: str,
        regime: str,
        max_results: int = 20,
    ) -> list[dict]:
        """
        Return stocks that share the same KMeans cluster as ticker in the
        given regime, sorted by PC Euclidean distance (closest first).

        Parameters
        ----------
        ticker      : stock ticker (case-insensitive)
        regime      : "Post-COVID" | "Rate Shock" | "Disinflation"
        max_results : cap on returned peers (default 20)

        Returns
        -------
        list of dicts with keys:
            ticker, cluster_id, pc1, pc2, quadrant, distance, gics_sector
        Returns [] if ticker not found or has no cluster membership in regime.
        """
        ticker = ticker.strip().upper()
        stock_id = f"stock:{ticker}"

        if not self._G.has_node(stock_id):
            return []

        # Step 1: find which cluster this ticker belongs to in the regime
        cluster_id = None
        for _, tgt, attrs in self._out_edges(stock_id, EdgeType.CLUSTER_MEMBERSHIP.value
                                              if KG_SCHEMA_AVAILABLE else "cluster_membership"):
            if attrs.get("regime") == regime:
                cluster_id = tgt
                break

        if cluster_id is None:
            return []

        # Step 2: get target ticker's PC scores for distance calculation
        target_pc1, target_pc2 = 0.0, 0.0
        for _, tgt, attrs in self._out_edges(stock_id, EdgeType.QUADRANT_ASSIGNMENT.value
                                              if KG_SCHEMA_AVAILABLE else "quadrant_assignment"):
            if attrs.get("regime") == regime:
                target_pc1 = _safe_float(attrs.get("pc1"))
                target_pc2 = _safe_float(attrs.get("pc2"))
                break

        # Step 3: find all other stocks in the same cluster in this regime
        peers = []
        for src, _, attrs in self._in_edges(cluster_id, EdgeType.CLUSTER_MEMBERSHIP.value
                                             if KG_SCHEMA_AVAILABLE else "cluster_membership"):
            if src == stock_id:
                continue
            if attrs.get("regime") != regime:
                continue

            peer_ticker = self._G.nodes[src].get("label", src.replace("stock:", ""))
            peer_sector = self._G.nodes[src].get("gics_sector", "Unknown")

            # Get peer PC scores and quadrant from quadrant_assignment edge
            peer_pc1, peer_pc2, peer_quadrant = 0.0, 0.0, "Unknown"
            for _, qtgt, qattrs in self._out_edges(src, EdgeType.QUADRANT_ASSIGNMENT.value
                                                    if KG_SCHEMA_AVAILABLE else "quadrant_assignment"):
                if qattrs.get("regime") == regime:
                    peer_pc1 = _safe_float(qattrs.get("pc1"))
                    peer_pc2 = _safe_float(qattrs.get("pc2"))
                    peer_quadrant = self._G.nodes[qtgt].get("label", qtgt)
                    break

            dist = _euclidean(target_pc1, target_pc2, peer_pc1, peer_pc2)

            peers.append({
                "ticker":      peer_ticker,
                "cluster_id":  cluster_id,
                "pc1":         round(peer_pc1, 4),
                "pc2":         round(peer_pc2, 4),
                "quadrant":    peer_quadrant,
                "distance":    round(dist, 4),
                "gics_sector": peer_sector,
            })

        peers.sort(key=lambda x: x["distance"])
        return peers[:max_results]

    # =========================================================================
    # METHOD 2 — get_quadrant_history
    # =========================================================================

    def get_quadrant_history(self, ticker: str) -> list[dict]:
        """
        Return the quadrant position of a stock across all three regimes,
        in chronological order.

        Parameters
        ----------
        ticker : stock ticker (case-insensitive)

        Returns
        -------
        list of dicts (one per regime found) with keys:
            regime, quadrant_id, quadrant_name, pc1, pc2, regime_index
        Returns [] if ticker not found or has no quadrant assignments.
        """
        ticker = ticker.strip().upper()
        stock_id = f"stock:{ticker}"

        if not self._G.has_node(stock_id):
            return []

        et = EdgeType.QUADRANT_ASSIGNMENT.value if KG_SCHEMA_AVAILABLE else "quadrant_assignment"
        history = []

        for _, qtgt, attrs in self._out_edges(stock_id, et):
            regime = attrs.get("regime", "")
            pc1    = _safe_float(attrs.get("pc1"))
            pc2    = _safe_float(attrs.get("pc2"))

            quadrant_label = self._G.nodes[qtgt].get("label", qtgt)
            # label format: "Q1: Profitable Value" — split on ":"
            parts = quadrant_label.split(":", 1)
            q_id   = parts[0].strip()
            q_name = parts[1].strip() if len(parts) > 1 else quadrant_label

            history.append({
                "regime":        regime,
                "quadrant_id":   q_id,
                "quadrant_name": q_name,
                "pc1":           round(pc1, 4),
                "pc2":           round(pc2, 4),
                "regime_index":  _regime_index(regime),
            })

        # Sort chronologically
        history.sort(key=lambda x: x["regime_index"])
        return history

    # =========================================================================
    # METHOD 3 — get_factor_rotation
    # =========================================================================

    def get_factor_rotation(
        self,
        factor: str,
        from_regime: str,
        to_regime: str,
    ) -> dict:
        """
        Return the loading shift for a factor across a regime boundary.
        Reads from session_state["period_loadings"] (populated after Period
        Comparison runs). Falls back to Appendix B constants.

        Parameters
        ----------
        factor      : factor code (must exist in FEATURE_COLUMNS)
        from_regime : e.g. "Post-COVID"
        to_regime   : e.g. "Disinflation"

        Returns
        -------
        dict with keys:
            factor, from_regime, to_regime,
            pc1_from, pc1_to, pc2_from, pc2_to, pc3_from, pc3_to,
            sign_change_pc1, sign_change_pc2, sign_change_pc3,
            magnitude_delta_pc2,   # PC2 is the primary diagnostic axis
            stability_class,       # "stable" | "rotated" | "reversed"
            headline_pc,           # which PC shows the most dramatic shift
            data_source            # "live" | "appendix_b_fallback"
        """
        period_loadings = self._ss.get("period_loadings", {})

        def _get_loading_row(regime_label: str) -> Optional[dict]:
            """Extract per-PC loadings for a factor in a regime."""
            # period_loadings keys are short labels: "Post-COVID", "Rate Shock", etc.
            # Values are DataFrames: index=factor_codes, columns=[PC1, PC2, PC3]
            df = period_loadings.get(regime_label)
            if df is not None and factor in df.index:
                row = df.loc[factor]
                return {
                    "PC1": float(row.get("PC1", 0.0)),
                    "PC2": float(row.get("PC2", 0.0)),
                    "PC3": float(row.get("PC3", 0.0) if "PC3" in row.index else 0.0),
                }
            return None

        from_row = _get_loading_row(from_regime)
        to_row   = _get_loading_row(to_regime)
        data_source = "live"

        # Fall back to Appendix B if pipeline hasn't run
        if from_row is None:
            from_row = _APPENDIX_B_LOADINGS.get((factor, from_regime))
            data_source = "appendix_b_fallback"
        if to_row is None:
            to_row = _APPENDIX_B_LOADINGS.get((factor, to_regime))
            data_source = "appendix_b_fallback"

        if from_row is None or to_row is None:
            return {
                "factor":        factor,
                "from_regime":   from_regime,
                "to_regime":     to_regime,
                "data_source":   "not_available",
                "error":         f"No loading data for '{factor}' in one or both regimes.",
            }

        # Compute per-PC deltas and stability classification
        pc_results = {}
        max_delta_abs = 0.0
        headline_pc   = "PC1"

        for pc in ["PC1", "PC2", "PC3"]:
            lf = from_row.get(pc, 0.0)
            lt = to_row.get(pc, 0.0)
            delta      = round(lt - lf, 4)
            sign_ch    = (lf >= 0) != (lt >= 0)
            stability  = _classify_stability(lf, lt)
            pc_results[pc] = {
                "loading_from": round(lf, 4),
                "loading_to":   round(lt, 4),
                "delta":        delta,
                "sign_change":  sign_ch,
                "stability":    stability,
            }
            if abs(delta) > max_delta_abs:
                max_delta_abs = abs(delta)
                headline_pc   = pc

        # Overall stability class: worst case across PCs
        stab_rank = {"reversed": 2, "rotated": 1, "stable": 0}
        overall_stability = max(
            (pc_results[pc]["stability"] for pc in ["PC1", "PC2", "PC3"]),
            key=lambda s: stab_rank.get(s, 0)
        )

        return {
            "factor":              factor,
            "from_regime":         from_regime,
            "to_regime":           to_regime,
            "pc1_from":            pc_results["PC1"]["loading_from"],
            "pc1_to":              pc_results["PC1"]["loading_to"],
            "pc2_from":            pc_results["PC2"]["loading_from"],
            "pc2_to":              pc_results["PC2"]["loading_to"],
            "pc3_from":            pc_results["PC3"]["loading_from"],
            "pc3_to":              pc_results["PC3"]["loading_to"],
            "sign_change_pc1":     pc_results["PC1"]["sign_change"],
            "sign_change_pc2":     pc_results["PC2"]["sign_change"],
            "sign_change_pc3":     pc_results["PC3"]["sign_change"],
            "magnitude_delta_pc2": pc_results["PC2"]["delta"],
            "stability_class":     overall_stability,
            "headline_pc":         headline_pc,
            "data_source":         data_source,
        }

    # =========================================================================
    # METHOD 4 — query_crowding_chain
    # =========================================================================

    def query_crowding_chain(
        self,
        from_regime: str,
        to_regime: str,
    ) -> dict:
        """
        The headline reasoning chain method.

        Traverses the regime transition edge between from_regime and to_regime,
        collects Procrustes score, crowding before/after, factor rotations, and
        builds a structured narrative_chain for the Tier 1 Narrative Engine.

        Parameters
        ----------
        from_regime : e.g. "Post-COVID"
        to_regime   : e.g. "Disinflation"

        Returns
        -------
        dict with keys:
            from_regime, to_regime,
            procrustes_score, is_major_break, common_tickers, severity, data_source,
            crowding_before, crowding_risk_before,
            crowding_after,  crowding_risk_after,
            crowding_delta,
            migration_pct, stocks_changed, stocks_analyzed,
            factor_rotations    : list[dict]  — all 11 factors with rotation data
            early_warning_triggered : bool
            narrative_chain     : list[dict]  — ordered steps for Tier 1 engine
        """
        src_id = f"regime:{from_regime}"
        tgt_id = f"regime:{to_regime}"

        # ── Step 1: Procrustes transition edge ────────────────────────────────
        et_transition = EdgeType.REGIME_TRANSITION.value if KG_SCHEMA_AVAILABLE else "regime_transition"
        transition_attrs = {}

        if self._G.has_edge(src_id, tgt_id):
            transition_attrs = dict(self._G.edges[src_id, tgt_id])
        else:
            # Try Appendix B fallback
            fb = (APPENDIX_B_PROCRUSTES.get((from_regime, to_regime)) or
                  APPENDIX_B_PROCRUSTES.get((to_regime, from_regime)) or {})
            transition_attrs = {
                "procrustes_disparity": fb.get("disparity", 0.0),
                "common_tickers":       fb.get("common_tickers", 0),
                "interpretation":       fb.get("interpretation", ""),
                "is_major_break":       fb.get("disparity", 0.0) >= PROCRUSTES_MEANINGFUL,
                "severity":             "Major" if fb.get("disparity", 0.0) >= PROCRUSTES_MEANINGFUL else "Meaningful",
                "migration_pct":        0.0,
                "stocks_changed":       0,
                "stocks_analyzed":      0,
                "data_source":          "appendix_b_fallback",
            }

        proc_score       = _safe_float(transition_attrs.get("procrustes_disparity"))
        is_major_break   = bool(transition_attrs.get("is_major_break", False))
        common_tickers   = int(_safe_float(transition_attrs.get("common_tickers")))
        severity         = transition_attrs.get("severity", "Unknown")
        migration_pct    = _safe_float(transition_attrs.get("migration_pct"))
        stocks_changed   = int(_safe_float(transition_attrs.get("stocks_changed")))
        stocks_analyzed  = int(_safe_float(transition_attrs.get("stocks_analyzed")))
        t_data_source    = transition_attrs.get("data_source", "unknown")

        # ── Override migration with live session state if available ───────────
        try:
            ss = _get_session_state()
            mig_summary = ss.get("migration_summary_df")
            if mig_summary is not None and not mig_summary.empty:
                for _, mrow in mig_summary.iterrows():
                    t = str(mrow.get("Transition", "")).replace("→", "->").strip()
                    if t.endswith(to_regime):
                        rate_str = str(mrow.get("Migration Rate", "")).replace("%", "").strip()
                        changed  = int(_safe_float(mrow.get("Changed Quadrant", 0)))
                        total    = int(_safe_float(mrow.get("Stocks Analyzed", 0)))
                        rate_val = _safe_float(rate_str, None)
                        if rate_val is not None and total > 0:
                            migration_pct   = rate_val
                            stocks_changed  = changed
                            stocks_analyzed = total
                        break
        except Exception:
            pass

        # ── Step 2: Crowding before / after ───────────────────────────────────
        def _crowding_from_node(regime_id: str) -> tuple[float, str]:
            attrs = self._node_attrs(regime_id)
            score = _safe_float(attrs.get("crowding_score"))
            risk  = attrs.get("crowding_risk", _risk_label(score))
            if score == 0.0:
                # Appendix B fallback
                short = regime_id.replace("regime:", "")
                fb = APPENDIX_B_CROWDING.get(short, {})
                score = _safe_float(getattr(fb.get("score", 0.0), "real", fb.get("score", 0.0))
                                    if hasattr(fb.get("score", 0.0), "real")
                                    else fb.get("score", 0.0))
                risk  = fb.get("risk_level", _risk_label(score))
                if hasattr(risk, "value"):
                    risk = risk.value
            return round(score, 1), str(risk)

        crowding_before, crowding_risk_before = _crowding_from_node(src_id)
        crowding_after,  crowding_risk_after  = _crowding_from_node(tgt_id)
        crowding_delta = round(crowding_after - crowding_before, 1)
        early_warning  = crowding_after >= CROWDING_THRESHOLD_ELEVATED

        # ── Step 3: Factor rotations for all 11 factors ───────────────────────
        # Pull factor codes from graph factor nodes
        factor_codes = [
            n.replace("factor:", "")
            for n, d in self._G.nodes(data=True)
            if d.get("node_type") == (NodeType.FACTOR.value if KG_SCHEMA_AVAILABLE else "factor")
        ]
        if not factor_codes:
            # Fallback to canonical registry factor order
            factor_codes = list(FEATURE_COLUMNS)

        factor_rotations = []
        for code in factor_codes:
            rot = self.get_factor_rotation(code, from_regime, to_regime)
            if "error" not in rot:
                # pc2_stability_class: stability classification specific to PC2 axis only.
                # stability_class is worst-case across all PCs — correct for overall
                # structural assessment but misleading when displaying PC2 values.
                pc2_stab = _classify_stability(rot["pc2_from"], rot["pc2_to"])
                factor_rotations.append({
                    "factor":            code,
                    "display_name":      self._G.nodes.get(f"factor:{code}", {}).get("label", code),
                    "pc2_from":          rot["pc2_from"],
                    "pc2_to":            rot["pc2_to"],
                    "sign_change_pc2":   rot["sign_change_pc2"],
                    "delta_pc2":         rot["magnitude_delta_pc2"],
                    "stability_class":   rot["stability_class"],     # overall worst-case
                    "pc2_stability_class": pc2_stab,                 # PC2-specific
                    "headline_pc":       rot["headline_pc"],
                    "data_source":       rot["data_source"],
                })

        # Sort: reversed first, then rotated, then stable
        _stab_order = {"reversed": 0, "rotated": 1, "stable": 2}
        factor_rotations.sort(key=lambda x: _stab_order.get(x["stability_class"], 3))

        # Count sign reversals on PC2 specifically — this is the primary diagnostic axis.
        # Using overall stability_class would include PC3 reversals that are not visible
        # in the PC2 display context and would produce misleading labels.
        reversals = [f for f in factor_rotations if f["pc2_stability_class"] == "reversed"]

        # ── Step 4: Build narrative_chain for Tier 1 engine ──────────────────
        narrative_chain = [
            {
                "step":           1,
                "type":           "regime_transition",
                "from":           from_regime,
                "to":             to_regime,
                "procrustes":     proc_score,
                "severity":       severity,
                "is_major_break": is_major_break,
                "common_tickers": common_tickers,
                "data_source":    t_data_source,
                "interpretation": (
                    f"Procrustes disparity {proc_score:.3f} — "
                    f"{'major structural break' if is_major_break else 'meaningful structural change'} "
                    f"({common_tickers:,} common tickers)"
                ),
            },
        ]

        # Add one chain step per reversed factor (these are the governance-critical events)
        for rev in reversals:
            narrative_chain.append({
                "step":        len(narrative_chain) + 1,
                "type":        "factor_rotation",
                "factor":      rev["factor"],
                "display":     rev["display_name"],
                "headline_pc": rev["headline_pc"],
                "loading_from": rev["pc2_from"],
                "loading_to":   rev["pc2_to"],
                "sign_change":  True,
                "data_source":  rev["data_source"],
                "interpretation": (
                    f"{rev['display_name']} reversed on {rev['headline_pc']}: "
                    f"{rev['pc2_from']:+.3f} → {rev['pc2_to']:+.3f}"
                ),
            })

        narrative_chain.append({
            "step":          len(narrative_chain) + 1,
            "type":          "crowding_shift",
            "before":        crowding_before,
            "after":         crowding_after,
            "delta":         crowding_delta,
            "risk_before":   crowding_risk_before,
            "risk_after":    crowding_risk_after,
            "interpretation": (
                f"Crowding score: {crowding_before:.1f} ({crowding_risk_before}) → "
                f"{crowding_after:.1f} ({crowding_risk_after}), Δ{crowding_delta:+.1f}"
            ),
        })

        if migration_pct > 0:
            narrative_chain.append({
                "step":           len(narrative_chain) + 1,
                "type":           "quadrant_migration",
                "migration_pct":  migration_pct,
                "stocks_changed": stocks_changed,
                "stocks_analyzed": stocks_analyzed,
                "interpretation": (
                    f"{migration_pct:.1f}% of stocks changed quadrant "
                    f"({stocks_changed:,} of {stocks_analyzed:,})"
                ),
            })

        narrative_chain.append({
            "step":        len(narrative_chain) + 1,
            "type":        "early_warning",
            "triggered":   early_warning,
            "threshold":   CROWDING_THRESHOLD_ELEVATED,
            "score":       crowding_after,
            "interpretation": (
                f"Early Warning {'TRIGGERED' if early_warning else 'not triggered'} — "
                f"crowding {crowding_after:.1f} "
                f"{'≥' if early_warning else '<'} threshold {CROWDING_THRESHOLD_ELEVATED:.0f}"
            ),
        })

        return {
            "from_regime":           from_regime,
            "to_regime":             to_regime,
            "procrustes_score":      proc_score,
            "is_major_break":        is_major_break,
            "common_tickers":        common_tickers,
            "severity":              severity,
            "data_source":           t_data_source,
            "crowding_before":       crowding_before,
            "crowding_risk_before":  crowding_risk_before,
            "crowding_after":        crowding_after,
            "crowding_risk_after":   crowding_risk_after,
            "crowding_delta":        crowding_delta,
            "migration_pct":         migration_pct,
            "stocks_changed":        stocks_changed,
            "stocks_analyzed":       stocks_analyzed,
            "factor_rotations":      factor_rotations,
            "n_reversals":           len(reversals),
            "early_warning_triggered": early_warning,
            "narrative_chain":       narrative_chain,
        }

    # =========================================================================
    # METHOD 5 — get_structural_drift_summary
    # =========================================================================

    def get_structural_drift_summary(self, regime: str) -> dict:
        """
        Aggregate structural state for a given regime.
        Suitable for CRO-level summary panels and the Early Warning engine.

        Parameters
        ----------
        regime : "Post-COVID" | "Rate Shock" | "Disinflation"

        Returns
        -------
        dict with keys:
            regime, start_date, end_date,
            crowding_score, crowding_risk,
            procrustes_from_prior, severity_from_prior, is_major_break,
            migration_pct, stocks_changed, stocks_analyzed,
            early_warning_triggered,
            prior_regime   (None if first regime)
        """
        regime_id  = f"regime:{regime}"
        node_attrs = self._node_attrs(regime_id)

        if not node_attrs:
            return {"regime": regime, "error": f"Regime node '{regime_id}' not found in graph."}

        # Crowding from node attrs
        crowding_score = _safe_float(node_attrs.get("crowding_score"))
        crowding_risk  = node_attrs.get("crowding_risk", _risk_label(crowding_score))
        if crowding_score == 0.0:
            fb = APPENDIX_B_CROWDING.get(regime, {})
            crowding_score = _safe_float(fb.get("score", 0.0))
            crowding_risk  = str(getattr(fb.get("risk_level"), "value", fb.get("risk_level", "Normal")))

        # Prior regime
        regime_idx  = _regime_index(regime)
        prior_regime = CANONICAL_REGIME_ORDER[regime_idx - 1] if regime_idx > 0 else None

        # Procrustes from prior via incoming transition edge
        proc_from_prior  = 0.0
        severity_prior   = "N/A"
        is_major_break   = False
        migration_pct    = 0.0
        stocks_changed   = 0
        stocks_analyzed  = 0

        if prior_regime:
            src_id = f"regime:{prior_regime}"
            et     = EdgeType.REGIME_TRANSITION.value if KG_SCHEMA_AVAILABLE else "regime_transition"

            if self._G.has_edge(src_id, regime_id):
                edge_attrs      = dict(self._G.edges[src_id, regime_id])
                proc_from_prior = _safe_float(edge_attrs.get("procrustes_disparity"))
                severity_prior  = edge_attrs.get("severity", "Unknown")
                is_major_break  = bool(edge_attrs.get("is_major_break", False))
                migration_pct   = _safe_float(edge_attrs.get("migration_pct"))
                stocks_changed  = int(_safe_float(edge_attrs.get("stocks_changed")))
                stocks_analyzed = int(_safe_float(edge_attrs.get("stocks_analyzed")))
            else:
                # Appendix B fallback for transition
                fb = (APPENDIX_B_PROCRUSTES.get((prior_regime, regime)) or
                      APPENDIX_B_PROCRUSTES.get((regime, prior_regime)) or {})
                proc_from_prior = _safe_float(fb.get("disparity", 0.0))
                is_major_break  = proc_from_prior >= PROCRUSTES_MEANINGFUL

            # ── Tier 2: override migration values from live session state ─────
            # The graph edge was built from Appendix B — live migration_summary_df
            # is more accurate and should take precedence when available.
            try:
                ss = _get_session_state()
                mig_summary = ss.get("migration_summary_df")
                if mig_summary is not None and not mig_summary.empty:
                    for _, mrow in mig_summary.iterrows():
                        t = str(mrow.get("Transition", "")).replace("→", "->").strip()
                        if t.endswith(regime):
                            rate_str = str(mrow.get("Migration Rate", "")).replace("%", "").strip()
                            changed  = int(_safe_float(mrow.get("Changed Quadrant", 0)))
                            total    = int(_safe_float(mrow.get("Stocks Analyzed", 0)))
                            rate_val = _safe_float(rate_str, None)
                            if rate_val is not None and total > 0:
                                migration_pct   = rate_val
                                stocks_changed  = changed
                                stocks_analyzed = total
                            break
            except Exception:
                pass

        early_warning = crowding_score >= CROWDING_THRESHOLD_ELEVATED

        return {
            "regime":                regime,
            "start_date":            node_attrs.get("start_date", ""),
            "end_date":              node_attrs.get("end_date", ""),
            "crowding_score":        round(crowding_score, 1),
            "crowding_risk":         crowding_risk,
            "procrustes_from_prior": round(proc_from_prior, 3),
            "severity_from_prior":   severity_prior,
            "is_major_break":        is_major_break,
            "migration_pct":         round(migration_pct, 1),
            "stocks_changed":        stocks_changed,
            "stocks_analyzed":       stocks_analyzed,
            "early_warning_triggered": early_warning,
            "prior_regime":          prior_regime,
        }

    # =========================================================================
    # METHOD 6 — serialize_subgraph  (Tier 2 chatbot context injection)
    # =========================================================================

    def serialize_subgraph(self, node_ids: list[str]) -> dict:
        """
        Serialize a bounded subgraph to a JSON-compatible dict for Tier 2
        chatbot context window injection.

        The chatbot receives this dict — never a live graph object.
        This is the structural enforcement of the Tier 1 / Tier 2 boundary.

        Parameters
        ----------
        node_ids : list of node ID strings (e.g. ["regime:Disinflation", "stock:AAPL"])

        Returns
        -------
        dict with keys:
            nodes : list[{id, node_type, label, ...attrs}]
            edges : list[{src, tgt, edge_type, ...attrs}]
            meta  : {node_count, edge_count, missing_nodes}
        """
        missing = [nid for nid in node_ids if not self._G.has_node(nid)]
        valid   = [nid for nid in node_ids if self._G.has_node(nid)]

        # Nodes
        serialized_nodes = []
        for nid in valid:
            attrs = dict(self._G.nodes[nid])
            # Convert any non-serializable types
            clean = {k: (v.value if hasattr(v, "value") else v) for k, v in attrs.items()}
            serialized_nodes.append({"id": nid, **clean})

        # Edges — only between nodes in the requested set
        valid_set = set(valid)
        serialized_edges = []
        for u, v, attrs in self._G.edges(data=True):
            if u in valid_set and v in valid_set:
                clean = {k: (val.value if hasattr(val, "value") else val)
                         for k, val in attrs.items()}
                serialized_edges.append({"src": u, "tgt": v, **clean})

        return {
            "nodes": serialized_nodes,
            "edges": serialized_edges,
            "meta":  {
                "node_count":    len(serialized_nodes),
                "edge_count":    len(serialized_edges),
                "missing_nodes": missing,
            },
        }

    # ── Convenience: graph statistics ────────────────────────────────────────

    def summary(self) -> dict:
        """Return basic graph statistics."""
        node_type_counts: dict = {}
        for _, d in self._G.nodes(data=True):
            nt = d.get("node_type", "unknown")
            node_type_counts[nt] = node_type_counts.get(nt, 0) + 1

        edge_type_counts: dict = {}
        for _, _, d in self._G.edges(data=True):
            et = d.get("edge_type", "unknown")
            edge_type_counts[et] = edge_type_counts.get(et, 0) + 1

        return {
            "total_nodes":      self._G.number_of_nodes(),
            "total_edges":      self._G.number_of_edges(),
            "node_type_counts": node_type_counts,
            "edge_type_counts": edge_type_counts,
        }


# =============================================================================
# STANDALONE VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("\n[kg_interface] Phase 3 standalone validation\n")

    # Build the static ontology graph (no equity nodes needed for structural tests)
    try:
        from kg_builder import build_static_ontology_graph
        result = build_static_ontology_graph()
        G = result.graph
        print(result.summary())
    except Exception as e:
        print(f"kg_builder unavailable ({e}), building minimal test graph...")
        G = nx.DiGraph(name="ESDS_test")
        # Wire minimal nodes for structural method tests
        G.add_node("regime:Post-COVID",
                   node_type="regime", label="Post-COVID",
                   start_date="2021-03-31", end_date="2022-06-30",
                   crowding_score=28.3, crowding_risk="Normal")
        G.add_node("regime:Rate Shock",
                   node_type="regime", label="Rate Shock",
                   start_date="2022-07-31", end_date="2023-09-30",
                   crowding_score=30.1, crowding_risk="Normal")
        G.add_node("regime:Disinflation",
                   node_type="regime", label="Disinflation",
                   start_date="2023-10-31", end_date="2024-10-31",
                   crowding_score=67.9, crowding_risk="Elevated")
        G.add_edge("regime:Post-COVID", "regime:Disinflation",
                   edge_type="regime_transition",
                   procrustes_disparity=0.459, common_tickers=316,
                   is_major_break=True, severity="Major",
                   migration_pct=60.1, stocks_changed=190, stocks_analyzed=316,
                   data_source="appendix_b_fallback")
        G.add_edge("regime:Post-COVID", "regime:Rate Shock",
                   edge_type="regime_transition",
                   procrustes_disparity=0.342, common_tickers=322,
                   is_major_break=True, severity="Major",
                   migration_pct=43.7, stocks_changed=138, stocks_analyzed=316,
                   data_source="appendix_b_fallback")
        G.add_edge("regime:Rate Shock", "regime:Disinflation",
                   edge_type="regime_transition",
                   procrustes_disparity=0.186, common_tickers=1590,
                   is_major_break=False, severity="Meaningful",
                   migration_pct=30.1, stocks_changed=95, stocks_analyzed=316,
                   data_source="appendix_b_fallback")

    kg = KnowledgeGraph(G)

    # ── Test 1: summary ───────────────────────────────────────────────────────
    print("── Test 1: graph summary")
    s = kg.summary()
    print(f"   Nodes: {s['total_nodes']}  Edges: {s['total_edges']}")
    assert s["total_nodes"] >= 3, "Expected at least 3 regime nodes"
    print("   PASS\n")

    # ── Test 2: get_factor_rotation (Appendix B fallback) ────────────────────
    test_factor = FEATURE_COLUMNS[0]

    print(f"── Test 2: get_factor_rotation ({test_factor}, Post-COVID → Disinflation)")
    rot = kg.get_factor_rotation(test_factor, "Post-COVID", "Disinflation")
    print(f"   PC2: {rot.get('pc2_from')} → {rot.get('pc2_to')}")
    print(f"   sign_change_pc2: {rot.get('sign_change_pc2')}")
    print(f"   stability_class: {rot.get('stability_class')}")
    print(f"   data_source:     {rot.get('data_source')}")
    assert "pc2_from" in rot, "Rotation output missing PC2 data"
    assert "stability_class" in rot, "Rotation output missing stability classification"
    print("   PASS\n")

    # ── Test 3: query_crowding_chain ──────────────────────────────────────────
    print("── Test 3: query_crowding_chain (Post-COVID → Disinflation)")
    chain = kg.query_crowding_chain("Post-COVID", "Disinflation")
    print(f"   Procrustes:        {chain['procrustes_score']:.3f}")
    print(f"   Crowding delta:    {chain['crowding_before']:.1f} → {chain['crowding_after']:.1f}  (Δ{chain['crowding_delta']:+.1f})")
    print(f"   Early warning:     {chain['early_warning_triggered']}")
    print(f"   Factor reversals:  {chain['n_reversals']}")
    print(f"   Narrative steps:   {len(chain['narrative_chain'])}")
    assert chain["is_major_break"] is True
    assert chain["early_warning_triggered"] is True
    assert len(chain["narrative_chain"]) >= 3
    print("   PASS\n")

    # ── Test 4: get_structural_drift_summary ──────────────────────────────────
    print("── Test 4: get_structural_drift_summary (Disinflation)")
    drift = kg.get_structural_drift_summary("Disinflation")
    print(f"   Crowding:          {drift['crowding_score']} ({drift['crowding_risk']})")
    print(f"   Procrustes prior:  {drift['procrustes_from_prior']:.3f}")
    print(f"   Early warning:     {drift['early_warning_triggered']}")
    assert drift["early_warning_triggered"] is True
    assert drift["prior_regime"] == "Rate Shock"
    print("   PASS\n")

    # ── Test 5: narrative_chain print ─────────────────────────────────────────
    print("── Test 5: narrative_chain (Post-COVID → Disinflation)")
    for step in chain["narrative_chain"]:
        print(f"   Step {step['step']} [{step['type']}]: {step['interpretation']}")
    print()

    # ── Test 6: serialize_subgraph ────────────────────────────────────────────
    print("── Test 6: serialize_subgraph")
    sg = kg.serialize_subgraph([
        "regime:Post-COVID", "regime:Disinflation", "regime:NONEXISTENT"
    ])
    print(f"   Nodes serialized:  {sg['meta']['node_count']}")
    print(f"   Edges serialized:  {sg['meta']['edge_count']}")
    print(f"   Missing nodes:     {sg['meta']['missing_nodes']}")
    assert sg["meta"]["node_count"] == 2
    assert "regime:NONEXISTENT" in sg["meta"]["missing_nodes"]
    print("   PASS\n")

    print("[OK] All Phase 3 tests passed — kg_interface.py complete.")
