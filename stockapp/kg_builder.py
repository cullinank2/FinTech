"""
kg_schema.py
------------
Phase 1: ESDS Knowledge Graph Ontology

Defines all node types, edge types, and their properties as Python dataclasses.
This is the vocabulary of the KG — no graph construction happens here.

Design principles:
  - Every node property maps to a real field in utils.py, period_analysis.py,
  
    config.py, or an Appendix B empirical anchor.
  - No invented abstractions. If it isn't computed by the existing pipeline,
    it is not a property here.
  - Dataclasses are immutable (frozen=True) to enforce governance integrity.
  - The two-tier AI distinction is encoded structurally:
      Tier 1 (Narrative Engine): uses deterministic KG path traversal only.
      Tier 2 (Chatbot): receives a serialized KG subgraph as context.

Node ID conventions:
  - MarketRegime:       "regime:{name}"          e.g. "regime:Post-COVID"
  - Factor:             "factor:{code}"          e.g. "factor:earnings_yield"
  - FactorAxis:         "axis:PC{n}"             e.g. "axis:PC1"
  - FactorCategory:     "category:{name}"        e.g. "category:Value"
  - Quadrant:           "quadrant:{id}"          e.g. "quadrant:Q4"
  - Cluster:            "cluster:{regime}:{id}"  e.g. "cluster:Disinflation:0"
  - Stock:              "stock:{ticker}"         e.g. "stock:GE"
  - FactorSpace:        "space:{regime}"         e.g. "space:Rate Shock"
  - CrowdingScore:      "crowding:{regime}"      e.g. "crowding:Disinflation"
  - RegimeTransition:   "transition:{A}__{B}"    e.g. "transition:Post-COVID__Rate Shock"
  - StructuralBreak:    "break:{A}__{B}"         e.g. "break:Post-COVID__Disinflation"
  - InstabilitySignal:  "signal:{regime}"        e.g. "signal:Disinflation"
  - EarlyWarning:       "warning:{regime}"       e.g. "warning:Disinflation"

Source references:
  config.py       — FEATURE_COLUMNS, FACTOR_CATEGORIES, QUADRANTS, N_COMPONENTS
  utils.py        — compute_pca_and_clusters(), determine_quadrant(),
                    compute_crowding_scores()
  period_analysis.py — _run_pca_for_period(), compute_procrustes_table(),
                       compute_quadrant_migration(), SUB_PERIODS
  narrative_engine.py — generate_narrative() four-section output
  Appendix B      — all numeric ground-truth values embedded below
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum


# =============================================================================
# ENUMERATIONS
# =============================================================================

class RegimeName(str, Enum):
    """
    The three macro regimes defined in period_analysis.py::SUB_PERIODS.
    Date ranges are the authoritative Appendix A / period_analysis definitions.
    """
    POST_COVID   = "Post-COVID"    # 2021-03-31 → 2022-06-30
    RATE_SHOCK   = "Rate Shock"    # 2022-07-31 → 2023-09-30
    DISINFLATION = "Disinflation"  # 2023-10-31 → 2024-10-31


class QuadrantID(str, Enum):
    """
    Four quadrants defined in config.py::QUADRANTS.
    Assignment rule: determine_quadrant(pc1, pc2) in utils.py.
    """
    Q1 = "Q1"   # PC1 >= 0, PC2 >= 0 → Profitable Value
    Q2 = "Q2"   # PC1 <  0, PC2 >= 0 → Value Traps / Distressed
    Q3 = "Q3"   # PC1 <  0, PC2 <  0 → Struggling Growth
    Q4 = "Q4"   # PC1 >= 0, PC2 <  0 → Quality Growth


class FactorCategoryName(str, Enum):
    """Six factor categories from config.py::FACTOR_CATEGORIES."""
    VALUE            = "Value"
    QUALITY          = "Quality"
    FINANCIAL_STR    = "Financial Strength"
    MOMENTUM         = "Momentum"
    RISK_VOLATILITY  = "Risk/Volatility"
    LIQUIDITY        = "Liquidity"


class RiskLevel(str, Enum):
    """
    Crowding risk thresholds from utils.py::compute_crowding_scores().
    Normal < 50 | Elevated 50-69 | High >= 70
    """
    NORMAL   = "Normal"    # score < 50
    ELEVATED = "Elevated"  # score 50-69
    HIGH     = "High"      # score >= 70


class StructuralBreakSeverity(str, Enum):
    """
    Procrustes disparity thresholds from Appendix A / period_analysis.py.
    <0.05 negligible | 0.05-0.15 detectable | 0.15-0.30 meaningful | >0.30 major
    """
    NEGLIGIBLE  = "Negligible"   # < 0.05
    DETECTABLE  = "Detectable"   # 0.05 – 0.14
    MEANINGFUL  = "Meaningful"   # 0.15 – 0.29
    MAJOR       = "Major"        # >= 0.30  ← recalibration trigger


class AITier(str, Enum):
    """
    Two-tier AI governance distinction — critical for institutional framing.
    Tier 1: deterministic Narrative Engine (governance artifact, reportable).
    Tier 2: configurable Chatbot (practitioner tool, session-specific).
    """
    TIER_1_NARRATIVE = "Tier1_NarrativeEngine"
    TIER_2_CHATBOT   = "Tier2_Chatbot"


# =============================================================================
# ── LAYER 1: MARKET CONTEXT NODES ───────────────────────────────────────────
# =============================================================================

@dataclass(frozen=True)
class MarketRegimeNode:
    """
    One of three macro regimes.
    Source: period_analysis.py::SUB_PERIODS
    Universe size is determined dynamically during graph construction.
    """
    node_id:        str            # "regime:{name}"
    name:           RegimeName
    start_date:     str            # ISO date string, e.g. "2021-03-31"
    end_date:       str
    universe_count: int            # total tickers in period
    # Populated after KG build:
    crowding_score: Optional[float] = None   # from CrowdingScoreNode
    is_break_from_prior: bool = False        # True if Procrustes > 0.30 vs prior


@dataclass(frozen=True)
class FactorSpaceNode:
    """
    The PCA coordinate system for a specific regime.
    Represents the geometric structure of the equity universe in that period.
    Source: period_analysis.py::_run_pca_for_period()
    """
    node_id:          str          # "space:{regime}"
    regime:           RegimeName
    n_components:     int = 3      # config.py::N_COMPONENTS
    pc1_variance_pct: float = 0.0  # populated dynamically from PCA results
    pc2_variance_pct: float = 0.0  # populated dynamically from PCA results
    pc3_variance_pct: float = 0.0  # populated dynamically from PCA results
    combined_variance_pct: float = 0.0  # populated dynamically from PCA results


# =============================================================================
# ── LAYER 2: FACTOR SYSTEM NODES ────────────────────────────────────────────
# =============================================================================

@dataclass(frozen=True)
class FactorNode:
    """
    One of the 11 individual signals from config.py::FEATURE_COLUMNS.
    Individual signals are deliberately NOT composited — this is an ESDS
    design principle (Appendix A: 'preserved individual factors').
    """
    node_id:       str                  # "factor:{code}"
    code:          str                  # e.g. "earnings_yield"
    display_name:  str                  # e.g. "Earnings Yield (V)"
    category:      FactorCategoryName
    data_source:   str                  # "WRDS/Compustat" or "Yahoo Finance"
    description:   str = ""


@dataclass(frozen=True)
class FactorAxisNode:
    """
    A principal component axis — PC1, PC2, or PC3.
    Source: config.py::PC1_INTERPRETATION, PC2_INTERPRETATION, PC3_INTERPRETATION
    Note: axis interpretation can rotate across regimes (Appendix B key finding).
    """
    node_id:            str        # "axis:PC{n}"
    pc_number:          int        # 1, 2, or 3
    name:               str        # "Profitability & Operational Quality"
    variance_explained: float      # full-sample % variance (config.py)
    high_meaning:       str        # plain-English positive-loading pole
    low_meaning:        str        # plain-English negative-loading pole


@dataclass(frozen=True)
class FactorCategoryNode:
    """
    A domain grouping of factors from config.py::FACTOR_CATEGORIES.
    Used for peer context and narrative generation in narrative_engine.py.
    """
    node_id:  str                  # "category:{name}"
    name:     FactorCategoryName
    members:  Tuple[str, ...]      # factor codes (immutable)


# =============================================================================
# ── LAYER 3: MARKET GEOMETRY NODES ──────────────────────────────────────────
# =============================================================================

@dataclass(frozen=True)
class QuadrantNode:
    """
    One of the four structural quadrants.
    Assignment: determine_quadrant(pc1, pc2) in utils.py.
    Source: config.py::QUADRANTS
    """
    node_id:      str          # "quadrant:{Q1|Q2|Q3|Q4}"
    quadrant_id:  QuadrantID
    name:         str          # e.g. "Quality Growth"
    pc1_sign:     str          # "positive" or "negative"
    pc2_sign:     str          # "positive" or "negative"
    description:  str


@dataclass(frozen=True)
class ClusterNode:
    """
    One KMeans cluster (K=4, random_state=42) within a given regime.
    Source: utils.py::compute_pca_and_clusters() → KMeans(n_clusters=4)
    Properties are populated during KG build from per-period PCA runs.
    """
    node_id:           str          # "cluster:{regime}:{cluster_id}"
    regime:            RegimeName
    cluster_id:        int          # 0, 1, 2, or 3
    centroid_pc1:      float = 0.0
    centroid_pc2:      float = 0.0
    centroid_pc3:      float = 0.0
    member_count:      int   = 0
    pct_of_universe:   float = 0.0  # % of period universe in this cluster


# =============================================================================
# ── LAYER 4: SECURITY LAYER NODES ───────────────────────────────────────────
# =============================================================================

@dataclass(frozen=True)
class StockNode:
    """
    An individual equity in the ESDS universe (~1,738 U.S. equities).
    Per-regime scores are populated during KG build.
    Source: utils.py::compute_pca_and_clusters() result DataFrame
    """
    node_id:     str           # "stock:{ticker}"
    ticker:      str
    permno:      str           # CRSP PERMNO (unique identifier)
    gics_sector: Optional[str] = None
    # Full-sample (time-averaged) PCA position:
    pc1:         float = 0.0
    pc2:         float = 0.0
    pc3:         float = 0.0
    quadrant_id: Optional[QuadrantID] = None
    cluster_id:  Optional[int]        = None


@dataclass(frozen=True)
class StockRegimePositionNode:
    """
    A stock's PCA position within a specific regime sub-period.
    Separate from StockNode to allow regime-conditioned comparisons.
    Source: period_analysis.py::_run_pca_for_period() → scores_df
    This node enables the quadrant migration analysis.
    """
    node_id:     str           # "position:{ticker}:{regime}"
    ticker:      str
    regime:      RegimeName
    pc1:         float
    pc2:         float
    pc3:         float
    quadrant_id: QuadrantID
    cluster_id:  int


# =============================================================================
# ── LAYER 5: STRUCTURAL METRICS & EVENTS ────────────────────────────────────
# =============================================================================

@dataclass(frozen=True)
class CrowdingScoreNode:
    """
    Factor Crowding Score for one regime.
    Formula: 0.6 * Concentration + 0.4 * (100 - Dispersion_Normalized)
    Source: utils.py::compute_crowding_scores()
    Appendix B ground truth:
    Values are populated dynamically from utils.py::compute_crowding_scores().
    """
    node_id:              str         # "crowding:{regime}"
    regime:               RegimeName
    score:                float       # 0–100
    risk_level:           RiskLevel
    largest_cluster_pct:  float       # % of universe in biggest cluster
    centroid_dispersion:  float       # mean pairwise centroid distance in PC1/PC2
    dispersion_normalized: float      # 0–100 normalized across periods
    # Thresholds from utils.py::risk_label():
    THRESHOLD_ELEVATED:   float = field(default=50.0, compare=False)
    THRESHOLD_HIGH:       float = field(default=70.0, compare=False)


@dataclass(frozen=True)
class RegimeTransitionNode:
    """
    The transition between two adjacent regime periods.
    Carries Procrustes disparity score and quadrant migration rate.
    Source: period_analysis.py::compute_procrustes_table(),
            compute_quadrant_migration()
    Appendix B ground truth (authoritative):
    Values are populated dynamically from:
    period_analysis.py::compute_procrustes_table()
    and
    period_analysis.py::compute_quadrant_migration().
    """
    node_id:            str          # "transition:{A}__{B}"
    regime_from:        RegimeName
    regime_to:          RegimeName
    procrustes_disparity: float      # Appendix B authoritative value
    common_ticker_count:  int        # intersection of regime universes
    migration_pct:        float      # % stocks that changed quadrant
    stocks_changed:       int        # raw count
    stocks_analyzed:      int        # denominator
    severity:             StructuralBreakSeverity
    interpretation:       str        # from period_analysis.py color-coded labels


@dataclass(frozen=True)
class StructuralBreakNode:
    """
    Fired when Procrustes disparity exceeds the 0.30 major threshold.
    Triggers recalibration recommendation per Appendix A governance rules.
    Source: period_analysis.py::compute_procrustes_table() → '🔴 Major regime change'
    Break nodes are created dynamically when disparity >= 0.30.
    """
    node_id:             str         # "break:{A}__{B}"
    transition_node_id:  str         # FK → RegimeTransitionNode
    disparity:           float       # must be >= 0.30 to exist
    severity:            StructuralBreakSeverity = StructuralBreakSeverity.MAJOR
    recalibration_flag:  bool = True


@dataclass(frozen=True)
class InstabilitySignalNode:
    """
    Intermediate warning: elevated but sub-break Procrustes + rising crowding.
    Represents the 0.15–0.29 'meaningful structural change' zone.
    Signals are generated dynamically when disparity falls within
the 0.15–0.29 meaningful structural change range.
    Source: Appendix A threshold definitions + utils.py crowding thresholds.
    """
    node_id:                str         # "signal:{transition_id}"
    transition_node_id:     str         # FK → RegimeTransitionNode
    procrustes_disparity:   float       # 0.15 – 0.29
    crowding_score_current: float       # from CrowdingScoreNode
    crowding_delta:         float       # change from prior regime
    signal_components:      Tuple[str, ...] = ()  # which signals triggered


@dataclass(frozen=True)
class EarlyWarningNode:
    """
    Composite alert combining multiple structural signals.
    Monitors: rising Procrustes + cluster compression + quadrant migration
              + factor axis rotation.
    Source: Appendix A 'Early Warning Engine' specification.
    Early warnings are generated dynamically based on crowding,
migration, and Procrustes thresholds.
    """
    node_id:              str         # "warning:{regime}"
    regime:               RegimeName
    triggered:            bool
    crowding_elevated:    bool
    procrustes_elevated:  bool
    migration_elevated:   bool
    composite_risk_level: RiskLevel
    alert_description:    str


# =============================================================================
# ── LAYER 6: ANALYTICS LAYER NODES ──────────────────────────────────────────
# =============================================================================

@dataclass(frozen=True)
class PCAModelNode:
    """
    Metadata about the PCA model for a specific regime.
    Source: period_analysis.py::_run_pca_for_period() → pca_model
    """
    node_id:         str         # "pca_model:{regime}"
    regime:          RegimeName
    n_components:    int = 3     # config.py::N_COMPONENTS
    random_state:    int = 42    # from period_analysis.py
    n_features:      int = 11    # len(FEATURE_COLUMNS)
    universe_size:   int = 0     # tickers used in this period's PCA fit


@dataclass(frozen=True)
class KMeansModelNode:
    """
    Metadata about the KMeans clustering model for a specific regime.
    Source: utils.py::compute_pca_and_clusters() → KMeans(n_clusters=4, random_state=42)
    """
    node_id:      str         # "kmeans_model:{regime}"
    regime:       RegimeName
    n_clusters:   int = 4     # config.py::N_CLUSTERS
    random_state: int = 42
    n_init:       int = 10


@dataclass(frozen=True)
class NarrativeOutputNode:
    """
    The four-section narrative output from narrative_engine.py.
    Tier 1 only — deterministic, versioned, governance artifact.
    Each section maps to a generate_*() function in narrative_engine.py.
    NOT used by the Tier 2 chatbot (which receives raw KG subgraph context).
    """
    node_id:        str         # "narrative:{ticker}:{regime}"
    ticker:         str
    regime:         RegimeName
    ai_tier:        AITier = AITier.TIER_1_NARRATIVE
    # Section outputs from narrative_engine.generate_narrative():
    summary:        str = ""    # generate_summary()
    factors:        str = ""    # generate_factor_highlights()
    trajectory:     str = ""    # generate_trajectory_narrative()
    peers:          str = ""    # generate_peer_context()
    version:        str = ""    # e.g. "1.0.0" for governance versioning


# =============================================================================
# ── EDGE TYPES ───────────────────────────────────────────────────────────────
# =============================================================================

@dataclass(frozen=True)
class LoadsOnEdge:
    """
    Factor → FactorAxis (within a given regime).
    Property: loading value from PCA components matrix.
    Source: period_analysis.py::_run_pca_for_period() → loadings_df
    Key finding (Appendix B): Earnings Yield sign reverses (+0.211 → −0.315)
    from Post-COVID to Disinflation — structural opposition to BM / Sales-to-Price.
    """
    edge_id:       str       # "loads_on:{factor}__{axis}__{regime}"
    factor_id:     str       # FK → FactorNode
    axis_id:       str       # FK → FactorAxisNode
    regime:        RegimeName
    loading_value: float     # signed loading from pca.components_.T
    is_dominant:   bool      # True if |loading| is in top 3 for this axis/regime
    sign_reversed_from_prior: bool = False  # True for EY in Rate Shock, Disinflation


@dataclass(frozen=True)
class OccupiesQuadrantEdge:
    """
    Stock → Quadrant (within a given regime).
    Derived from StockRegimePositionNode.quadrant_id.
    Source: determine_quadrant() in utils.py / _assign_quadrant() in period_analysis.py
    """
    edge_id:    str
    stock_id:   str          # FK → StockNode
    quadrant_id: str         # FK → QuadrantNode
    regime:     RegimeName
    pc1:        float        # stock's PC1 score in this regime
    pc2:        float        # stock's PC2 score in this regime


@dataclass(frozen=True)
class MigratedToEdge:
    """
    StockRegimePosition(A) → StockRegimePosition(B).
    Exists only if quadrant changed between regimes.
    Source: period_analysis.py::compute_quadrant_migration() → 'Any Change'
    Migration statistics are computed dynamically using
    period_analysis.py::compute_quadrant_migration().
    """
    edge_id:          str
    position_from_id: str    # FK → StockRegimePositionNode (regime A)
    position_to_id:   str    # FK → StockRegimePositionNode (regime B)
    regime_from:      RegimeName
    regime_to:        RegimeName
    quadrant_from:    QuadrantID
    quadrant_to:      QuadrantID
    pc1_delta:        float  # PC1(B) - PC1(A)
    pc2_delta:        float  # PC2(B) - PC2(A)


@dataclass(frozen=True)
class BelongsToClusterEdge:
    """
    Stock → Cluster (within a given regime).
    Source: utils.py::compute_pca_and_clusters() → result_df['cluster']
    Used for crowding analysis: spatial concentration in PCA factor space.
    """
    edge_id:          str
    stock_id:         str    # FK → StockNode
    cluster_id:       str    # FK → ClusterNode
    regime:           RegimeName
    pc_distance_from_centroid: float = 0.0  # Euclidean distance in PC1/PC2/PC3 space


@dataclass(frozen=True)
class PeersWithEdge:
    """
    Stock ↔ Stock (symmetric, within a given regime and quadrant).
    Structural peers: same quadrant in same regime.
    Source: utils.py::get_stocks_in_same_quadrant()
    Used by: generate_peer_context() in narrative_engine.py (Tier 1)
    Note: this is GICS-agnostic structural peering — a key ESDS differentiator.
    """
    edge_id:           str
    stock_a_id:        str    # FK → StockNode
    stock_b_id:        str    # FK → StockNode
    regime:            RegimeName
    quadrant_id:       QuadrantID
    pc_distance:       float  # Euclidean distance in PC1/PC2 space
    same_cluster:      bool
    same_gics_sector:  bool


@dataclass(frozen=True)
class TransitionsToEdge:
    """
    MarketRegime → MarketRegime.
    Carries the Procrustes structural distance between the two regimes.
    Source: period_analysis.py::compute_procrustes_table()
    Appendix B ground truth embedded in RegimeTransitionNode above.
    """
    edge_id:             str
    regime_from_id:      str   # FK → MarketRegimeNode
    regime_to_id:        str   # FK → MarketRegimeNode
    transition_node_id:  str   # FK → RegimeTransitionNode (full data)
    procrustes_disparity: float
    severity:            StructuralBreakSeverity


@dataclass(frozen=True)
class TriggersEdge:
    """
    StructuralMetric → StructuralEvent.
    e.g. CrowdingScore(Disinflation, 68) → EarlyWarning(Disinflation)
         RegimeTransition(PC→DI, 0.397) → StructuralBreak
    Source: Appendix A threshold definitions.
    """
    edge_id:      str
    source_id:    str   # FK → CrowdingScoreNode or RegimeTransitionNode
    target_id:    str   # FK → EarlyWarningNode or StructuralBreakNode
    threshold:    float
    observed_value: float
    exceeded:     bool


@dataclass(frozen=True)
class GeneratedByEdge:
    """
    NarrativeOutput → Stock + Regime.
    Governance traceability: which data inputs produced which narrative.
    Tier 1 only — each narrative output is versioned and auditable.
    Source: narrative_engine.py::generate_narrative()
    """
    edge_id:         str
    narrative_id:    str   # FK → NarrativeOutputNode
    stock_id:        str   # FK → StockNode
    regime_id:       str   # FK → MarketRegimeNode
    ai_tier:         AITier = AITier.TIER_1_NARRATIVE
    narrative_version: str = ""


# =============================================================================
# NOTE
# =============================================================================
# Empirical statistics such as Procrustes disparity, crowding scores,
# migration percentages, and PCA variance explained are generated
# dynamically by the ESDS analytics pipeline.
#
# These values are computed in:
#   period_analysis.py
#   utils.py
#
# and inserted into the knowledge graph during graph construction
# inside kg_builder.py.


# =============================================================================
# RUNTIME NODE CREATION
# =============================================================================
# Concrete node instances are NOT defined in this schema file.
#
# The ESDS Knowledge Graph is populated dynamically from live
# analytics outputs during graph construction in:
#
#     kg_builder.py
#
# This schema defines only the ontology structure.


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def validate_schema() -> Dict[str, int]:
    """
    Validate that ontology classes and enums exist.

    This file defines only the ESDS knowledge graph schema.
    Node instances are created dynamically in kg_builder.py.
    """

    checks = {
        "RegimeName": len(RegimeName),
        "QuadrantID": len(QuadrantID),
        "FactorCategoryName": len(FactorCategoryName),
        "RiskLevel": len(RiskLevel),
        "StructuralBreakSeverity": len(StructuralBreakSeverity),
        "AITier": len(AITier),
    }

    assert checks["RegimeName"] == 3
    assert checks["QuadrantID"] == 4

    checks["TOTAL_ENUMS"] = sum(checks.values())

    return checks

if __name__ == "__main__":

    counts = validate_schema()

    print("=== ESDS Knowledge Graph Schema ===\n")

    for name, count in counts.items():
        print(f"{name:<25} {count}")

    print("\nSchema validation successful.")
    print("Node instances will be created dynamically in kg_builder.py.")