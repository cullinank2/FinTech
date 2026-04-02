"""
structural_types.py
-------------------
Canonical type definitions for the KG-backed Structural Analyst layer.

Purpose
-------
Provides a single source of truth for:
- supported structural question types
- evidence packet schema
- analysis response schema

This module is intentionally lightweight and dependency-free.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict

# ============================================================
# Canonical question types
# ============================================================

StructuralQuestionType = Literal[
    "structural_drift",
    "regime_transition",
    "peer_comparison",
    "factor_rotation",
    "quadrant_history",
]

ConfidenceLevel = Literal["high", "medium", "low"]


# ============================================================
# Evidence + response schemas
# ============================================================

class EvidenceItem(TypedDict):
    source_type: str
    source_name: str
    fact: str


class SubgraphSnapshotMeta(TypedDict):
    node_count: int
    edge_count: int
    included_node_ids: List[str]


class StructuralEvidencePacket(TypedDict, total=False):
    # Core request identity
    question_type: StructuralQuestionType
    ticker: str
    regime: str

    # Optional request context
    peer_tickers: List[str]
    target_factor: Optional[str]
    from_regime: Optional[str]
    to_regime: Optional[str]

    # Deterministic KG outputs
    structural_drift_summary: Dict[str, Any]
    quadrant_history: List[Dict[str, Any]]
    crowding_chain: Dict[str, Any]
    peer_context: List[Dict[str, Any]]
    factor_rotation: Dict[str, Any]

    # Bounded structural context
    serialized_subgraph: Dict[str, Any]
    subgraph_snapshot_meta: SubgraphSnapshotMeta


class StructuralAnalysisResponse(TypedDict):
    question_type: StructuralQuestionType
    ticker: str
    regime: str
    answer: str
    summary_bullets: List[str]
    evidence: List[EvidenceItem]
    subgraph_snapshot: SubgraphSnapshotMeta
    limits: List[str]
    confidence: ConfidenceLevel
    analysis_mode: str


# ============================================================
# Constants
# ============================================================

STRUCTURAL_ANALYSIS_MODE = "bounded_kg_v1"

VALID_QUESTION_TYPES = {
    "structural_drift",
    "regime_transition",
    "peer_comparison",
    "factor_rotation",
    "quadrant_history",
}

VALID_CONFIDENCE_LEVELS = {"high", "medium", "low"}