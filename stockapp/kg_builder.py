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
    Appendix B: universe counts 1,648 / 1,665 / 1,676 per regime.
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
    pc1_variance_pct: float = 0.0  # Appendix B: 28.3% (full-sample)
    pc2_variance_pct: float = 0.0  # Appendix B: 14.4%
    pc3_variance_pct: float = 0.0  # Appendix B: 12.0%
    combined_variance_pct: float = 0.0  # Appendix B: 54.7%


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
      Post-COVID:   score=22, largest_cluster_pct=37.0, dispersion=2.63 → Normal
      Rate Shock:   score=30, largest_cluster_pct=51.7, dispersion=2.51 → Normal
      Disinflation: score=68, largest_cluster_pct=47.1, dispersion=2.37 → Elevated
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
      Post-COVID → Rate Shock:   disparity=0.277, migration_pct=55.0, n=1,620
      Post-COVID → Disinflation: disparity=0.397, migration_pct=N/A,  n=1,591
      Rate Shock → Disinflation: disparity=0.207, migration_pct=30.4, n=1,636
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
    In current data: only Post-COVID → Disinflation (0.397) crosses this threshold.
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
    In current data: Post-COVID → Rate Shock (0.277) sits here.
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
    In current data: Disinflation crowding score of 68 → Elevated alert.
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
    Appendix B: 68% of tracked universe changed quadrant at least once.
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
# GROUND TRUTH CONSTANTS  (Appendix B — single source of empirical truth)
# =============================================================================

APPENDIX_B_PROCRUSTES: Dict[Tuple[str, str], Dict] = {
    ("Post-COVID", "Rate Shock"):   {
        "disparity": 0.277,
        "common_tickers": 1620,
        "severity": StructuralBreakSeverity.MEANINGFUL,
        "interpretation": "Meaningful structural change — approaching major threshold",
    },
    ("Post-COVID", "Disinflation"): {
        "disparity": 0.397,
        "common_tickers": 1591,
        "severity": StructuralBreakSeverity.MAJOR,
        "interpretation": "Major regime change — recalibration trigger crossed",
    },
    ("Rate Shock", "Disinflation"): {
        "disparity": 0.207,
        "common_tickers": 1636,
        "severity": StructuralBreakSeverity.MEANINGFUL,
        "interpretation": "Meaningful structural change — adjacent-period normalization",
    },
}

APPENDIX_B_CROWDING: Dict[str, Dict] = {
    "Post-COVID": {
        "score": 22.0,
        "risk_level": RiskLevel.NORMAL,
        "largest_cluster_pct": 37.0,
        "centroid_dispersion": 2.63,
        "n_stocks": 1648,
    },
    "Rate Shock": {
        "score": 30.0,
        "risk_level": RiskLevel.NORMAL,
        "largest_cluster_pct": 51.7,
        "centroid_dispersion": 2.51,
        "n_stocks": 1665,
    },
    "Disinflation": {
        "score": 68.0,
        "risk_level": RiskLevel.ELEVATED,
        "largest_cluster_pct": 47.1,
        "centroid_dispersion": 2.37,
        "n_stocks": 1676,
    },
}

APPENDIX_B_MIGRATION: Dict[Tuple[str, str], Dict] = {
    ("Post-COVID", "Rate Shock"):   {
        "migration_pct": 55.0,
        "stocks_changed": 875,
        "stocks_analyzed": 1591,
    },
    ("Rate Shock", "Disinflation"): {
        "migration_pct": 30.4,
        "stocks_changed": 484,
        "stocks_analyzed": 1591,
    },
}

APPENDIX_B_MIGRATION_ANY = {
    "pct": 68.0,
    "changed": 1082,
    "total": 1591,
    "description": "68% of tracked universe changed quadrant at least once across all three regimes",
}

APPENDIX_B_PC_VARIANCE = {
    "PC1": 28.3,
    "PC2": 14.4,
    "PC3": 12.0,
    "combined": 54.7,
}

# Key factor loading observations from Appendix B (structural events)
APPENDIX_B_LOADING_EVENTS = {
    "earnings_yield_sign_reversal": {
        "factor": "earnings_yield",
        "axis": "PC2",
        "Post-COVID": +0.211,
        "Rate Shock": -0.364,
        "Disinflation": -0.315,
        "note": "EY reverses to negative loading — structurally opposed to BM and SP in PC2",
    },
    "debt_assets_sign_reversal": {
        "factor": "debt_assets",
        "axis": "PC3",
        "Post-COVID": -0.248,
        "Rate Shock": +0.719,
        "Disinflation": +0.770,
        "note": "Most dramatic PC3 structural event — sign reversal at Post-COVID→Rate Shock boundary",
    },
    "cash_debt_PC1_trajectory": {
        "factor": "cash_debt",
        "axis": "PC1",
        "Post-COVID": +0.329,
        "Rate Shock": +0.427,
        "Disinflation": +0.387,
        "note": "Cash-to-Debt loading on PC1 rises then partially retreats — balance sheet quality signal",
    },
}


# =============================================================================
# CANONICAL NODE CATALOG  (static instances — populated once, referenced always)
# =============================================================================

# ── Market Regimes ────────────────────────────────────────────────────────────
REGIME_NODES: Dict[str, MarketRegimeNode] = {
    "Post-COVID": MarketRegimeNode(
        node_id="regime:Post-COVID",
        name=RegimeName.POST_COVID,
        start_date="2021-03-31",
        end_date="2022-06-30",
        universe_count=1648,
        crowding_score=22.0,
        is_break_from_prior=False,
    ),
    "Rate Shock": MarketRegimeNode(
        node_id="regime:Rate Shock",
        name=RegimeName.RATE_SHOCK,
        start_date="2022-07-31",
        end_date="2023-09-30",
        universe_count=1665,
        crowding_score=30.0,
        is_break_from_prior=False,  # Procrustes 0.277 — meaningful but < 0.30
    ),
    "Disinflation": MarketRegimeNode(
        node_id="regime:Disinflation",
        name=RegimeName.DISINFLATION,
        start_date="2023-10-31",
        end_date="2024-10-31",
        universe_count=1676,
        crowding_score=68.0,
        is_break_from_prior=True,   # Procrustes 0.397 → major break from Post-COVID
    ),
}

# ── Factor Axes ───────────────────────────────────────────────────────────────
AXIS_NODES: Dict[str, FactorAxisNode] = {
    "PC1": FactorAxisNode(
        node_id="axis:PC1",
        pc_number=1,
        name="Profitability & Operational Quality",
        variance_explained=APPENDIX_B_PC_VARIANCE["PC1"],
        high_meaning="Operationally profitable, financially strong, stable cash position",
        low_meaning="Lower profitability, weaker operations, volatile, cash-constrained",
    ),
    "PC2": FactorAxisNode(
        node_id="axis:PC2",
        pc_number=2,
        name="Valuation Style",
        variance_explained=APPENDIX_B_PC_VARIANCE["PC2"],
        high_meaning="Deep value — high book-to-market, sales-to-price, earnings yield",
        low_meaning="Growth premium — trading above fundamental value metrics",
    ),
    "PC3": FactorAxisNode(
        node_id="axis:PC3",
        pc_number=3,
        name="Leverage & Asset Intensity",
        variance_explained=APPENDIX_B_PC_VARIANCE["PC3"],
        high_meaning="Leveraged, asset-intensive, higher debt-to-assets",
        low_meaning="Asset-light, conservative, lower leverage, cleaner balance sheets",
    ),
}

# ── Quadrants ─────────────────────────────────────────────────────────────────
QUADRANT_NODES: Dict[str, QuadrantNode] = {
    "Q1": QuadrantNode(
        node_id="quadrant:Q1",
        quadrant_id=QuadrantID.Q1,
        name="Profitable Value",
        pc1_sign="positive",
        pc2_sign="positive",
        description="Strong profitability trading at value prices — rare combination",
    ),
    "Q2": QuadrantNode(
        node_id="quadrant:Q2",
        quadrant_id=QuadrantID.Q2,
        name="Value Traps / Distressed",
        pc1_sign="negative",
        pc2_sign="positive",
        description="Cheap on traditional metrics but weak operational quality",
    ),
    "Q3": QuadrantNode(
        node_id="quadrant:Q3",
        quadrant_id=QuadrantID.Q3,
        name="Struggling Growth",
        pc1_sign="negative",
        pc2_sign="negative",
        description="Growth premium valuation with below-average operational quality",
    ),
    "Q4": QuadrantNode(
        node_id="quadrant:Q4",
        quadrant_id=QuadrantID.Q4,
        name="Quality Growth",
        pc1_sign="positive",
        pc2_sign="negative",
        description="High operational quality commanding a growth-oriented premium",
    ),
}

# ── Factors ───────────────────────────────────────────────────────────────────
FACTOR_NODES: Dict[str, FactorNode] = {
    "earnings_yield": FactorNode(
        node_id="factor:earnings_yield",
        code="earnings_yield",
        display_name="Earnings Yield (V)",
        category=FactorCategoryName.VALUE,
        data_source="WRDS/Compustat",
        description="Earnings per share / price. Reverses sign on PC2 in Rate Shock and Disinflation regimes.",
    ),
    "bm": FactorNode(
        node_id="factor:bm",
        code="bm",
        display_name="Book-to-Market (V)",
        category=FactorCategoryName.VALUE,
        data_source="WRDS/Compustat",
        description="Book value / market cap. Dominant positive PC2 loader — stable across regimes.",
    ),
    "sales_to_price": FactorNode(
        node_id="factor:sales_to_price",
        code="sales_to_price",
        display_name="Sales-to-Price (V)",
        category=FactorCategoryName.VALUE,
        data_source="WRDS/Compustat",
        description="Sales / market cap. Stable positive PC2 loader across all three regimes.",
    ),
    "roe": FactorNode(
        node_id="factor:roe",
        code="roe",
        display_name="Return on Equity (Q)",
        category=FactorCategoryName.QUALITY,
        data_source="WRDS/Compustat",
        description="Net income / equity. Core PC1 driver — positive loader across all regimes.",
    ),
    "roa": FactorNode(
        node_id="factor:roa",
        code="roa",
        display_name="Return on Assets (Q)",
        category=FactorCategoryName.QUALITY,
        data_source="WRDS/Compustat",
        description="Net income / assets. Dominant positive PC1 loader (+0.55 full-sample).",
    ),
    "gprof": FactorNode(
        node_id="factor:gprof",
        code="gprof",
        display_name="Gross Profitability (Q)",
        category=FactorCategoryName.QUALITY,
        data_source="WRDS/Compustat",
        description="Gross profit / assets. PC1 positive; sign reverses on PC3 at Post-COVID→Rate Shock.",
    ),
    "debt_assets": FactorNode(
        node_id="factor:debt_assets",
        code="debt_assets",
        display_name="Debt-to-Assets (FS)",
        category=FactorCategoryName.FINANCIAL_STR,
        data_source="WRDS/Compustat",
        description="Total debt / assets. Most dramatic structural event: PC3 sign reversal −0.248 → +0.719.",
    ),
    "cash_debt": FactorNode(
        node_id="factor:cash_debt",
        code="cash_debt",
        display_name="Cash-to-Debt (FS)",
        category=FactorCategoryName.FINANCIAL_STR,
        data_source="WRDS/Compustat",
        description="Cash / debt. PC1 loading rises Post-COVID→Rate Shock (+0.329→+0.427). Stepped pattern in time-series.",
    ),
    "momentum_12m": FactorNode(
        node_id="factor:momentum_12m",
        code="momentum_12m",
        display_name="12-Mo. Momentum (R)",
        category=FactorCategoryName.MOMENTUM,
        data_source="Yahoo Finance",
        description="12-month price return. Behavioral complement signal — not composited.",
    ),
    "vol_60d_ann": FactorNode(
        node_id="factor:vol_60d_ann",
        code="vol_60d_ann",
        display_name="60-Day Volatility (R)",
        category=FactorCategoryName.RISK_VOLATILITY,
        data_source="Yahoo Finance",
        description="Annualized 60-day return volatility. PC3 positive loader (+0.52 full-sample).",
    ),
    "addv_63d": FactorNode(
        node_id="factor:addv_63d",
        code="addv_63d",
        display_name="Liquidity (R)",
        category=FactorCategoryName.LIQUIDITY,
        data_source="Yahoo Finance",
        description="63-day average daily dollar volume. Negative PC3 loader (−0.48 full-sample).",
    ),
}

# ── Factor Categories ─────────────────────────────────────────────────────────
CATEGORY_NODES: Dict[str, FactorCategoryNode] = {
    cat: FactorCategoryNode(
        node_id=f"category:{cat}",
        name=FactorCategoryName(cat),
        members=tuple(members),
    )
    for cat, members in {
        "Value":            ["earnings_yield", "bm", "sales_to_price"],
        "Quality":          ["roe", "roa", "gprof"],
        "Financial Strength": ["debt_assets", "cash_debt"],
        "Momentum":         ["momentum_12m"],
        "Risk/Volatility":  ["vol_60d_ann"],
        "Liquidity":        ["addv_63d"],
    }.items()
}

# ── Crowding Score Nodes ──────────────────────────────────────────────────────
CROWDING_NODES: Dict[str, CrowdingScoreNode] = {
    regime: CrowdingScoreNode(
        node_id=f"crowding:{regime}",
        regime=RegimeName(regime),
        score=data["score"],
        risk_level=data["risk_level"],
        largest_cluster_pct=data["largest_cluster_pct"],
        centroid_dispersion=data["centroid_dispersion"],
        dispersion_normalized=0.0,  # populated during KG build (cross-period normalization)
    )
    for regime, data in APPENDIX_B_CROWDING.items()
}

# ── Regime Transition Nodes ───────────────────────────────────────────────────
TRANSITION_NODES: Dict[Tuple[str, str], RegimeTransitionNode] = {
    ("Post-COVID", "Rate Shock"): RegimeTransitionNode(
        node_id="transition:Post-COVID__Rate Shock",
        regime_from=RegimeName.POST_COVID,
        regime_to=RegimeName.RATE_SHOCK,
        procrustes_disparity=0.277,
        common_ticker_count=1620,
        migration_pct=55.0,
        stocks_changed=875,
        stocks_analyzed=1591,
        severity=StructuralBreakSeverity.MEANINGFUL,
        interpretation="Meaningful structural change — approaching but below major threshold",
    ),
    ("Post-COVID", "Disinflation"): RegimeTransitionNode(
        node_id="transition:Post-COVID__Disinflation",
        regime_from=RegimeName.POST_COVID,
        regime_to=RegimeName.DISINFLATION,
        procrustes_disparity=0.397,
        common_ticker_count=1591,
        migration_pct=68.0,    # any-change across all periods
        stocks_changed=1082,
        stocks_analyzed=1591,
        severity=StructuralBreakSeverity.MAJOR,
        interpretation="Major regime change — Procrustes 0.397 exceeds 0.30 recalibration threshold",
    ),
    ("Rate Shock", "Disinflation"): RegimeTransitionNode(
        node_id="transition:Rate Shock__Disinflation",
        regime_from=RegimeName.RATE_SHOCK,
        regime_to=RegimeName.DISINFLATION,
        procrustes_disparity=0.207,
        common_ticker_count=1636,
        migration_pct=30.4,
        stocks_changed=484,
        stocks_analyzed=1591,
        severity=StructuralBreakSeverity.MEANINGFUL,
        interpretation="Meaningful structural change — adjacent-period normalization, larger universe overlap",
    ),
}

# ── Structural Break Node (only one in current data) ─────────────────────────
STRUCTURAL_BREAK_NODES: Dict[str, StructuralBreakNode] = {
    "Post-COVID__Disinflation": StructuralBreakNode(
        node_id="break:Post-COVID__Disinflation",
        transition_node_id="transition:Post-COVID__Disinflation",
        disparity=0.397,
        severity=StructuralBreakSeverity.MAJOR,
        recalibration_flag=True,
    ),
}

# ── Early Warning Node (Disinflation — Elevated crowding) ────────────────────
EARLY_WARNING_NODES: Dict[str, EarlyWarningNode] = {
    "Disinflation": EarlyWarningNode(
        node_id="warning:Disinflation",
        regime=RegimeName.DISINFLATION,
        triggered=True,
        crowding_elevated=True,      # score 68 → Elevated (>=50 threshold)
        procrustes_elevated=True,    # 0.397 major break from Post-COVID
        migration_elevated=True,     # 68% any-change migration rate
        composite_risk_level=RiskLevel.ELEVATED,
        alert_description=(
            "Disinflation regime shows elevated crowding (score 68), "
            "a major structural break from Post-COVID (Procrustes 0.397), "
            "and high quadrant migration (68% of tracked universe). "
            "Recalibration of factor model recommended."
        ),
    ),
}


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def validate_schema() -> Dict[str, int]:
    """
    Count all canonical nodes and verify internal consistency.
    Run this to confirm the schema is self-consistent before building the graph.
    """
    counts = {
        "MarketRegimeNode":      len(REGIME_NODES),
        "FactorAxisNode":        len(AXIS_NODES),
        "QuadrantNode":          len(QUADRANT_NODES),
        "FactorNode":            len(FACTOR_NODES),
        "FactorCategoryNode":    len(CATEGORY_NODES),
        "CrowdingScoreNode":     len(CROWDING_NODES),
        "RegimeTransitionNode":  len(TRANSITION_NODES),
        "StructuralBreakNode":   len(STRUCTURAL_BREAK_NODES),
        "EarlyWarningNode":      len(EARLY_WARNING_NODES),
        # Dynamic nodes (populated during build, not catalogued here):
        # StockNode, StockRegimePositionNode, ClusterNode,
        # PCAModelNode, KMeansModelNode, NarrativeOutputNode
    }
    total_static = sum(counts.values())

    # Consistency checks
    assert len(FACTOR_NODES) == 11, \
        f"Expected 11 factors (FEATURE_COLUMNS), got {len(FACTOR_NODES)}"
    assert len(QUADRANT_NODES) == 4, \
        f"Expected 4 quadrants, got {len(QUADRANT_NODES)}"
    assert len(REGIME_NODES) == 3, \
        f"Expected 3 regimes, got {len(REGIME_NODES)}"
    assert len(AXIS_NODES) == 3, \
        f"Expected 3 PC axes (N_COMPONENTS=3), got {len(AXIS_NODES)}"
    assert len(CROWDING_NODES) == 3, \
        f"Expected one crowding score per regime, got {len(CROWDING_NODES)}"
    assert len(TRANSITION_NODES) == 3, \
        f"Expected 3 pairwise transitions (C(3,2)), got {len(TRANSITION_NODES)}"

    # Verify all factor categories account for all 11 features
    all_categorized = [f for node in CATEGORY_NODES.values() for f in node.members]
    assert len(all_categorized) == 11, \
        f"Category members should total 11, got {len(all_categorized)}"
    assert set(all_categorized) == set(FACTOR_NODES.keys()), \
        "Category members do not match FACTOR_NODES keys"

    # Verify Appendix B values embedded correctly
    assert TRANSITION_NODES[("Post-COVID", "Disinflation")].procrustes_disparity == 0.397
    assert CROWDING_NODES["Disinflation"].score == 68.0
    assert CROWDING_NODES["Disinflation"].risk_level == RiskLevel.ELEVATED

    counts["TOTAL_STATIC_NODES"] = total_static
    return counts


if __name__ == "__main__":
    counts = validate_schema()
    print("=== kg_schema.py — Schema Validation ===\n")
    for name, count in counts.items():
        if name == "TOTAL_STATIC_NODES":
            print(f"\n  {'─'*36}")
        print(f"  {name:<30} {count}")
    print("\n  All assertions passed. Schema is self-consistent.")
    print(f"\n  Dynamic nodes (populated during kg_builder.py):")
    print(f"  {'StockNode':<30} ~1,738 (full universe)")
    print(f"  {'StockRegimePositionNode':<30} ~1,738 × 3 regimes")
    print(f"  {'ClusterNode':<30} 4 clusters × 3 regimes = 12")
    print(f"  {'PCAModelNode':<30} 3 (one per regime)")
    print(f"  {'KMeansModelNode':<30} 3 (one per regime)")
    print(f"  {'NarrativeOutputNode':<30} 1 per stock queried (Tier 1 only)")
