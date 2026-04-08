"""
kg_schema.py
------------
Phase 1: ESDS Knowledge Graph Ontology

Defines all node types, edge types, enumerations, and catalog dictionaries
that form the vocabulary of the ESDS Knowledge Graph.

No graph construction happens here — that is kg_builder.py's job.

Design principles:
  - Catalog dicts are built from config.py and period_analysis.py imports.
    They are NEVER populated from hard-coded Appendix B numbers at build time.
  - Appendix B values are isolated in APPENDIX_B_* dicts at the bottom,
    clearly labeled as last-resort display fallbacks only.
  - Dataclasses are frozen=True to enforce governance integrity.
  - Two-tier AI distinction encoded structurally:
      Tier 1 (Narrative Engine): deterministic KG path traversal only.
      Tier 2 (Chatbot): receives serialized KG subgraph as context.

Node ID conventions:
  regime:     "regime:{short_name}"     e.g. "regime:Post-COVID"
  factor:     "factor:{code}"           e.g. "factor:<feature_code>"
  axis:       "axis:PC{n}"             e.g. "axis:PC1"
  category:   "category:{name}"        e.g. "category:Value"
  quadrant:   "quadrant:{id}"          e.g. "quadrant:Q4"
  cluster:    "cluster:{regime}:{id}"  e.g. "cluster:Disinflation:0"
  stock:      "stock:{ticker}"         e.g. "stock:GE"
  crowding:   "crowding:{short_name}"  e.g. "crowding:Disinflation"
  transition: "transition:{A}__{B}"   e.g. "transition:Post-COVID__Rate Shock"
  break:      "break:{A}__{B}"        e.g. "break:Post-COVID__Disinflation"
  warning:    "warning:{short_name}"   e.g. "warning:Disinflation"
  mechanism:  "mechanism:{id}"        e.g. "mechanism:procrustes"
  platform:   "platform:{id}"         e.g. "platform:esds"
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from enum import Enum

from semantic_constants import (
    PROCRUSTES_NEGLIGIBLE,
    PROCRUSTES_DETECTABLE,
    PROCRUSTES_MEANINGFUL,
    CROWDING_THRESHOLD_ELEVATED,
    CROWDING_THRESHOLD_HIGH,
)

from semantic_constants import REGIME_ORDER

# =============================================================================
# METHODOLOGY CONSTANTS
# Imported from semantic_constants.py (canonical threshold owner)
# =============================================================================


# =============================================================================
# ENUMERATIONS
# =============================================================================

class RegimeName(str, Enum):
    POST_COVID   = REGIME_ORDER[0]
    RATE_SHOCK   = REGIME_ORDER[1]
    DISINFLATION = REGIME_ORDER[2]


class QuadrantID(str, Enum):
    """Four PCA quadrants. Assignment: determine_quadrant() in utils.py."""
    Q1 = "Q1"   # PC1 >= 0, PC2 >= 0
    Q2 = "Q2"   # PC1 <  0, PC2 >= 0
    Q3 = "Q3"   # PC1 <  0, PC2 <  0
    Q4 = "Q4"   # PC1 >= 0, PC2 <  0


class FactorCategoryName(str, Enum):
    """Factor categories — must match keys in config.py::FACTOR_CATEGORIES."""
    VALUE         = "Value"
    QUALITY       = "Quality"
    FINANCIAL_STR = "Financial Strength"
    MOMENTUM      = "Momentum"
    RISK_VOL      = "Risk/Volatility"
    LIQUIDITY     = "Liquidity"


class RiskLevel(str, Enum):
    """
    Crowding risk levels.
    Thresholds: Normal < 50 | Elevated 50-69 | High >= 70
    Source: utils.py::risk_label()
    """
    NORMAL   = "Normal"
    ELEVATED = "Elevated"
    HIGH     = "High"


class StructuralBreakSeverity(str, Enum):
    """
    Procrustes disparity severity bands.
    Source: period_analysis.py::compute_procrustes_table() inline thresholds.
    """
    NEGLIGIBLE = "Negligible"   # < 0.05
    DETECTABLE = "Detectable"   # 0.05 - 0.14
    MEANINGFUL = "Meaningful"   # 0.15 - 0.29
    MAJOR      = "Major"        # >= 0.30 -> recalibration trigger


class AITier(str, Enum):
    """Two-tier AI governance distinction."""
    TIER_1_NARRATIVE = "Tier1_NarrativeEngine"
    TIER_2_CHATBOT   = "Tier2_Chatbot"


class NodeType(str, Enum):
    """All node types present in the ESDS knowledge graph."""
    REGIME           = "regime"
    FACTOR           = "factor"
    AXIS             = "axis"
    CATEGORY         = "category"
    QUADRANT         = "quadrant"
    CLUSTER          = "cluster"
    STOCK            = "stock"
    CROWDING         = "crowding"
    TRANSITION       = "transition"
    STRUCTURAL_BREAK = "structural_break"
    EARLY_WARNING    = "early_warning"
    MECHANISM        = "mechanism"
    PLATFORM         = "platform"


class EdgeType(str, Enum):
    """All edge relationship types present in the ESDS knowledge graph."""
    REGIME_TRANSITION   = "regime_transition"
    CROWDING_LEVEL      = "crowding_level"
    FACTOR_LOADING      = "factor_loading"   # AUDIT-OK: scanner checks plural 'factor_loadings' — singular is correct
    BELONGS_TO_CATEGORY = "belongs_to_category"
    TRIGGERS_BREAK      = "triggers_break"
    TRIGGERS_WARNING    = "triggers_warning"
    QUADRANT_ASSIGNMENT = "quadrant_assignment"
    CLUSTER_MEMBERSHIP  = "cluster_membership"
    BELONGS_TO          = "belongs_to"
    MIGRATES_TO         = "migrates_to"
    GOVERNS             = "governs"
    COMPLEMENTS         = "complements"


# =============================================================================
# KG SERIALIZATION ALLOWLISTS
# =============================================================================

SERIALIZABLE_NODE_TYPES = {
    NodeType.STOCK.value,
    NodeType.FACTOR.value,
    NodeType.REGIME.value,
    NodeType.QUADRANT.value,
    NodeType.MECHANISM.value,
    NodeType.CLUSTER.value,
    NodeType.CATEGORY.value,
    NodeType.AXIS.value,
    NodeType.PLATFORM.value,
}

SERIALIZABLE_EDGE_TYPES = {
    EdgeType.REGIME_TRANSITION.value,
    EdgeType.FACTOR_LOADING.value,
    EdgeType.QUADRANT_ASSIGNMENT.value,
    EdgeType.CLUSTER_MEMBERSHIP.value,
    EdgeType.CROWDING_LEVEL.value,
    EdgeType.BELONGS_TO_CATEGORY.value,
    EdgeType.BELONGS_TO.value,
    EdgeType.MIGRATES_TO.value,
    EdgeType.GOVERNS.value,
    EdgeType.COMPLEMENTS.value,
    EdgeType.TRIGGERS_BREAK.value,
    EdgeType.TRIGGERS_WARNING.value,
    "related_to",
}


# =============================================================================
# DATACLASS NODE DEFINITIONS
# =============================================================================

@dataclass(frozen=True)
class MarketRegimeNode:
    node_id:             str
    name:                RegimeName
    start_date:          str
    end_date:            str
    universe_count:      int        # from session_state at build time
    crowding_score:      Optional[float] = None
    is_break_from_prior: bool = False


@dataclass(frozen=True)
class FactorNode:
    node_id:      str
    code:         str
    display_name: str
    category:     FactorCategoryName
    data_source:  str
    description:  str = ""


@dataclass(frozen=True)
class FactorAxisNode:
    node_id:            str
    pc_number:          int
    name:               str
    variance_explained: float       # from pca_model.explained_variance_ratio_
    high_meaning:       str
    low_meaning:        str


@dataclass(frozen=True)
class FactorCategoryNode:
    node_id: str
    name:    FactorCategoryName
    members: Tuple[str, ...]


@dataclass(frozen=True)
class QuadrantNode:
    node_id:     str
    quadrant_id: QuadrantID
    name:        str
    pc1_sign:    str
    pc2_sign:    str
    description: str


@dataclass(frozen=True)
class CrowdingScoreNode:
    node_id:             str
    regime:              RegimeName
    score:               float
    risk_level:          RiskLevel
    largest_cluster_pct: float
    centroid_dispersion: float


@dataclass(frozen=True)
class RegimeTransitionNode:
    node_id:              str
    regime_from:          RegimeName
    regime_to:            RegimeName
    procrustes_disparity: float
    common_ticker_count:  int
    migration_pct:        float
    stocks_changed:       int
    stocks_analyzed:      int
    severity:             StructuralBreakSeverity
    interpretation:       str


@dataclass(frozen=True)
class StructuralBreakNode:
    node_id:            str
    transition_node_id: str
    disparity:          float
    severity:           StructuralBreakSeverity = StructuralBreakSeverity.MAJOR
    recalibration_flag: bool = True


@dataclass(frozen=True)
class EarlyWarningNode:
    node_id:              str
    regime:               RegimeName
    triggered:            bool
    crowding_elevated:    bool
    procrustes_elevated:  bool
    composite_risk_level: RiskLevel
    alert_description:    str


@dataclass(frozen=True)
class MechanismNode:
    node_id:       str
    label:         str
    tooltip:       str
    zero_prior_art: bool = False


@dataclass(frozen=True)
class PlatformNode:
    node_id: str
    label:   str
    tooltip: str


# =============================================================================
# CATALOG DICTS — built from config.py + period_analysis.py
# No Appendix B values here. All empirical numbers populated at build time
# from live session state inside kg_builder.py.
# =============================================================================

def _build_catalogs():
    """
    Build all static catalog dicts from config.py and period_analysis.py.
    Empirical values (Procrustes scores, crowding scores, variance) are NOT
    set here — they are inserted by kg_builder.py from session state.
    """
    try:
        from config import (
            FEATURE_COLUMNS, FEATURE_DISPLAY_NAMES, FACTOR_CATEGORIES,
            QUADRANTS, PC1_INTERPRETATION, PC2_INTERPRETATION,
            PC3_INTERPRETATION, N_COMPONENTS,
        )
        from factor_registry import FEATURE_DATA_SOURCES
        from period_analysis import SUB_PERIODS

        # Strip \n from SUB_PERIODS keys to get clean short labels
        short_period_map = {k.split('\n')[0]: v for k, v in SUB_PERIODS.items()}

        _regime_name_map = {
            "Post-COVID":   RegimeName.POST_COVID,
            "Rate Shock":   RegimeName.RATE_SHOCK,
            "Disinflation": RegimeName.DISINFLATION,
        }

        # ── REGIME_NODES ──────────────────────────────────────────────────────
        regime_nodes: Dict[str, MarketRegimeNode] = {}
        for short, (start, end) in short_period_map.items():
            rname = _regime_name_map.get(short)
            if rname is None:
                continue
            regime_nodes[short] = MarketRegimeNode(
                node_id        = f"regime:{short}",
                name           = rname,
                start_date     = start,
                end_date       = end,
                universe_count = 0,    # filled by kg_builder from session state
                crowding_score = None, # filled by kg_builder from session state
            )

        # ── FACTOR_NODES ──────────────────────────────────────────────────────
        from factor_registry import FEATURE_METADATA

        _cat_enum_map = {
            'Value':              FactorCategoryName.VALUE,
            'Quality':            FactorCategoryName.QUALITY,
            'Financial Strength': FactorCategoryName.FINANCIAL_STR,
            'Momentum':           FactorCategoryName.MOMENTUM,
            'Risk/Volatility':    FactorCategoryName.RISK_VOL,
            'Liquidity':          FactorCategoryName.LIQUIDITY,
        }

        factor_nodes: Dict[str, FactorNode] = {}

        for code, meta in FEATURE_METADATA.items():
            cat_str  = meta.get('category', 'Quality')
            cat_enum = _cat_enum_map.get(cat_str, FactorCategoryName.QUALITY)

            factor_nodes[code] = FactorNode(
                node_id      = f"factor:{code}",
                code         = code,
                display_name = meta.get('display', code),
                category     = cat_enum,
                data_source  = meta.get('source', 'WRDS'),
                description  = f"{cat_str} factor | Source: {meta.get('source', 'WRDS')}",
            )

        # ── QUADRANT_NODES ────────────────────────────────────────────────────
        _q_enum = {q: QuadrantID(q) for q in ['Q1', 'Q2', 'Q3', 'Q4']}
        quadrant_nodes: Dict[str, QuadrantNode] = {}
        for qid, qdata in QUADRANTS.items():
            quadrant_nodes[qid] = QuadrantNode(
                node_id     = f"quadrant:{qid}",
                quadrant_id = _q_enum[qid],
                name        = qdata['name'],
                pc1_sign    = qdata['pc1_sign'],
                pc2_sign    = qdata['pc2_sign'],
                description = qdata['description'],
            )

        # ── AXIS_NODES ────────────────────────────────────────────────────────
        # variance_explained from config fallback; overridden live in kg_builder
        axis_nodes: Dict[str, FactorAxisNode] = {}
        for i, interp in enumerate(
            [PC1_INTERPRETATION, PC2_INTERPRETATION, PC3_INTERPRETATION], start=1
        ):
            axis_nodes[f"PC{i}"] = FactorAxisNode(
                node_id            = f"axis:PC{i}",
                pc_number          = i,
                name               = interp['name'],
                variance_explained = interp.get('variance_explained', 0.0),
                high_meaning       = ', '.join(interp.get('high_meaning', [])),
                low_meaning        = ', '.join(interp.get('low_meaning', [])),
            )

        # ── CATEGORY_NODES ────────────────────────────────────────────────────
        category_nodes: Dict[str, FactorCategoryNode] = {}
        for cat_str, codes in FACTOR_CATEGORIES.items():
            cat_enum = _cat_enum_map.get(cat_str, FactorCategoryName.QUALITY)
            category_nodes[cat_str] = FactorCategoryNode(
                node_id = f"category:{cat_str}",
                name    = cat_enum,
                members = tuple(codes),
            )

        # ── MECHANISM_NODES ───────────────────────────────────────────────────
        mechanism_nodes: Dict[str, MechanismNode] = {
            "pca": MechanismNode(
                node_id        = "mechanism:pca",
                label          = "PCA\nStructural\nCoordinate System",
                tooltip        = (
                    "PCA as structural diagnostic geometry.\n"
                    f"Retains {N_COMPONENTS} components. Individual factors preserved.\n"
                    "Zero prior art as governance diagnostic."
                ),
                zero_prior_art = True,
            ),
            "procrustes": MechanismNode(
                node_id        = "mechanism:procrustes",
                label          = "Procrustes\nDisparity",
                tooltip        = (
                    "Rotation-invariant distance between factor spaces across regimes.\n"
                    f"Major break threshold: >= {PROCRUSTES_MEANINGFUL}.\n"
                    "Zero prior art in finance."
                ),
                zero_prior_art = True,
            ),
            "crowding": MechanismNode(
                node_id        = "mechanism:crowding",
                label          = "Geometric\nCrowding",
                tooltip        = (
                    "Spatial compression in PCA factor space.\n"
                    "Formula: 0.6 x Concentration + 0.4 x (100 - Dispersion).\n"
                    f"Elevated >= {CROWDING_THRESHOLD_ELEVATED:.0f} | "
                    f"High >= {CROWDING_THRESHOLD_HIGH:.0f}.\n"
                    "Zero prior art — not correlation-based."
                ),
                zero_prior_art = True,
            ),
            "kmeans": MechanismNode(
                node_id = "mechanism:kmeans",
                label   = "KMeans\nClustering",
                tooltip = (
                    f"KMeans (K=4, random_state=42) structural peer grouping.\n"
                    "GICS-agnostic — peers by factor geometry, not industry codes."
                ),
            ),
            "quadrant": MechanismNode(
                node_id = "mechanism:quadrant",
                label   = "Quadrant\nClassification",
                tooltip = (
                    "PC1/PC2 sign-based quadrant assignment.\n"
                    + "\n".join(
                        f"{qid}: {qdata['name']} "
                        f"(PC1 {qdata['pc1_sign']}, PC2 {qdata['pc2_sign']})"
                        for qid, qdata in QUADRANTS.items()
                    )
                ),
            ),
        }

        # ── PLATFORM_NODES ────────────────────────────────────────────────────
        platform_nodes: Dict[str, PlatformNode] = {
            "esds": PlatformNode(
                node_id = "platform:esds",
                label   = "ESDS\nStructural\nVisibility Layer",
                tooltip = (
                    "Equity Structural Diagnostics System\n"
                    "Structural visibility layer alongside Barra / Aladdin / Axioma.\n"
                    "Universe count: populated live from session state.\n"
                    "Two-tier AI: Narrative Engine (governance) + Chatbot (practitioner)."
                ),
            ),
            "barra": PlatformNode(
                node_id = "platform:barra",
                label   = "Barra / Aladdin\n/ Axioma",
                tooltip = (
                    "Incumbent institutional risk platforms.\n"
                    "Measure factor exposures; do not monitor structural character "
                    "of factor space. ESDS complements, does not compete."
                ),
            ),
            "narrative": PlatformNode(
                node_id = "platform:narrative",
                label   = "Tier 1\nNarrative Engine\n(Deterministic)",
                tooltip = (
                    "Rules-based governance artifact — no API key required.\n"
                    "Versioned, auditable, CRO-level outputs.\n"
                    "Source: Tier 1 Narrative Engine (deterministic module)"  # ARCH: intentional NE boundary — provenance label only, no import
                ),
            ),
            "chatbot": PlatformNode(
                node_id = "platform:chatbot",
                label   = "Tier 2\nAI Chatbot\n(Configurable)",
                tooltip = (
                    "Configurable practitioner tool — requires OpenAI API key.\n"
                    "Multi-turn conversational analysis.\n"
                    "Separated from governance layer by design."
                ),
            ),
        }

        transition_pairs = [
            ("Post-COVID", "Rate Shock"),
            ("Post-COVID", "Disinflation"),
            ("Rate Shock", "Disinflation"),
        ]

        return (
            regime_nodes, factor_nodes, quadrant_nodes, axis_nodes,
            category_nodes, mechanism_nodes, platform_nodes,
            transition_pairs, short_period_map, True,
        )

    except Exception as e:
        print(f"[kg_schema] Catalog build failed: {e}. Using empty stubs.")
        return ({}, {}, {}, {}, {}, {}, {}, [], {}, False)


(
    REGIME_NODES,
    FACTOR_NODES,
    QUADRANT_NODES,
    AXIS_NODES,
    CATEGORY_NODES,
    MECHANISM_NODES,
    PLATFORM_NODES,
    TRANSITION_PAIRS,
    SHORT_PERIOD_MAP,
    _CATALOGS_OK,
) = _build_catalogs()

# Populated at runtime by kg_builder from session state
EMPIRICAL_ANCHORS: Dict = {}


# =============================================================================
# APPENDIX B FALLBACK VALUES
# Category D — used ONLY for display when pipeline has not yet run.
# NEVER used as input to graph construction.
# =============================================================================

APPENDIX_B_PROCRUSTES = {
    ("Post-COVID", "Rate Shock"):   {"disparity": 0.342, "common_tickers": 322,
                                     "interpretation": "Major regime change"},
    ("Post-COVID", "Disinflation"): {"disparity": 0.459, "common_tickers": 316,
                                     "interpretation": "Major regime change"},
    ("Rate Shock", "Disinflation"): {"disparity": 0.186, "common_tickers": 1590,
                                     "interpretation": "Meaningful structural change"},
}

APPENDIX_B_CROWDING = {
    "Post-COVID":   {"score": 28.3,  "risk_level": RiskLevel.NORMAL},
    "Rate Shock":   {"score": 30.1,  "risk_level": RiskLevel.NORMAL},
    "Disinflation": {"score": 67.9,  "risk_level": RiskLevel.ELEVATED},
}

APPENDIX_B_PC_VARIANCE = {
    "PC1": 28.3, "PC2": 14.4, "PC3": 12.0, "combined": 54.7,
}

APPENDIX_B_MIGRATION = {
    ("Post-COVID", "Rate Shock"):   {"migration_pct": 43.7, "changed": 138, "analyzed": 316},
    ("Rate Shock", "Disinflation"): {"migration_pct": 60.1, "changed": 190, "analyzed": 316},
}

APPENDIX_B_UNIVERSE_COUNT = 1738


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def validate_schema() -> Dict[str, int]:
    checks = {
        "RegimeName":              len(RegimeName),
        "QuadrantID":              len(QuadrantID),
        "FactorCategoryName":      len(FactorCategoryName),
        "RiskLevel":               len(RiskLevel),
        "StructuralBreakSeverity": len(StructuralBreakSeverity),
        "AITier":                  len(AITier),
        "NodeType":                len(NodeType),
        "EdgeType":                len(EdgeType),
        "REGIME_NODES":            len(REGIME_NODES),
        "FACTOR_NODES":            len(FACTOR_NODES),
        "QUADRANT_NODES":          len(QUADRANT_NODES),
        "AXIS_NODES":              len(AXIS_NODES),
        "CATEGORY_NODES":          len(CATEGORY_NODES),
        "MECHANISM_NODES":         len(MECHANISM_NODES),
        "PLATFORM_NODES":          len(PLATFORM_NODES),
    }
    assert checks["RegimeName"]   == 3, "Expected 3 regimes"
    assert checks["QuadrantID"]   == 4, "Expected 4 quadrants"
    assert checks["FACTOR_NODES"] >= 11, f"Expected >=11 factors, got {checks['FACTOR_NODES']}"
    return checks


if __name__ == "__main__":
    counts = validate_schema()
    print("=== ESDS Knowledge Graph Schema ===\n")
    for name, count in counts.items():
        print(f"  {name:<30} {count}")
    print(f"\n  Catalogs built from config: {_CATALOGS_OK}")
    print("\nSchema validation successful.")
