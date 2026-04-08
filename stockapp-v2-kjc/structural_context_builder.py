"""
structural_context_builder.py
-----------------------------
Deterministic context assembly layer for the KG-backed Structural Analyst.

Purpose
-------
Builds a bounded evidence packet from kg_interface.py query methods.
The LLM never calls KG methods directly.

Expected KG interface surface
-----------------------------
This module assumes the supplied `kg` object exposes:
- get_peers(ticker, regime) -> list[dict]
- get_quadrant_history(ticker) -> list[dict]
- get_factor_rotation(factor, from_regime, to_regime) -> dict
- query_crowding_chain(from_regime, to_regime) -> dict
- get_structural_drift_summary(regime) -> dict
- serialize_subgraph(node_ids) -> dict

Design notes
------------
- We keep all retrieval deterministic.
- We include only a bounded subgraph.
- We record subgraph metadata for UI and audit display.
"""

from typing import Any, Dict, List, Optional

from structural_types import (
    StructuralEvidencePacket,
    VALID_QUESTION_TYPES,
)


def build_structural_evidence_packet(
    kg: Any,
    ticker: str,
    regime: str,
    question_type: str,
    factor: Optional[str] = None,
    from_regime: Optional[str] = None,
    to_regime: Optional[str] = None,
    max_peers: int = 5,
) -> StructuralEvidencePacket:
    """
    Build a bounded evidence packet for the Structural Analyst.

    Parameters
    ----------
    kg : Any
        Knowledge graph interface object
    ticker : str
        Target stock ticker
    regime : str
        Current regime label
    question_type : str
        One of:
        - structural_drift
        - regime_transition
        - peer_comparison
        - factor_rotation
        - quadrant_history
    factor : str | None
        Required for factor_rotation
    from_regime : str | None
        Used for regime_transition / factor_rotation
    to_regime : str | None
        Used for regime_transition / factor_rotation
    max_peers : int
        Max peers to include in the packet

    Returns
    -------
    StructuralEvidencePacket
    """
    _validate_request_inputs(
        ticker=ticker,
        regime=regime,
        question_type=question_type,
        factor=factor,
        from_regime=from_regime,
        to_regime=to_regime,
        max_peers=max_peers,
    )

    packet: StructuralEvidencePacket = {
        "question_type": question_type,  # type: ignore[typeddict-item]
        "ticker": ticker,
        "regime": regime,
    }

    node_ids: List[str] = _seed_node_ids(
        ticker=ticker,
        regime=regime,
        question_type=question_type,
        factor=factor,
        from_regime=from_regime,
        to_regime=to_regime,
    )

    if question_type == "structural_drift":
        _populate_structural_drift_packet(
            packet=packet,
            kg=kg,
            ticker=ticker,
            regime=regime,
            max_peers=max_peers,
        )

    elif question_type == "peer_comparison":
        _populate_peer_comparison_packet(
            packet=packet,
            kg=kg,
            ticker=ticker,
            regime=regime,
            max_peers=max_peers,
        )

    elif question_type == "quadrant_history":
        _populate_quadrant_history_packet(
            packet=packet,
            kg=kg,
            ticker=ticker,
        )

    elif question_type == "factor_rotation":
        _populate_factor_rotation_packet(
            packet=packet,
            kg=kg,
            factor=factor,
            from_regime=from_regime,
            to_regime=to_regime,
        )

    elif question_type == "regime_transition":
        _populate_regime_transition_packet(
            packet=packet,
            kg=kg,
            from_regime=from_regime,
            to_regime=to_regime,
        )

    serialized = _safe_serialize_subgraph(kg=kg, node_ids=node_ids)

    packet["serialized_subgraph"] = serialized
    packet["subgraph_snapshot_meta"] = {
        "node_count": len(serialized.get("nodes", [])),
        "edge_count": len(serialized.get("edges", [])),
        "included_node_ids": node_ids,
    }

    return packet


# ============================================================
# Internal population helpers
# ============================================================

def _populate_structural_drift_packet(
    packet: StructuralEvidencePacket,
    kg: Any,
    ticker: str,
    regime: str,
    max_peers: int,
) -> None:
    packet["structural_drift_summary"] = _safe_call_dict(
        kg.get_structural_drift_summary,
        regime,
    )
    packet["quadrant_history"] = _safe_call_list(
        kg.get_quadrant_history,
        ticker,
    )
    packet["peer_context"] = _truncate_list(
        _safe_call_list(kg.get_peers, ticker, regime),
        max_peers,
    )


def _populate_peer_comparison_packet(
    packet: StructuralEvidencePacket,
    kg: Any,
    ticker: str,
    regime: str,
    max_peers: int,
) -> None:
    peers = _truncate_list(
        _safe_call_list(kg.get_peers, ticker, regime),
        max_peers,
    )
    packet["peer_context"] = peers
    packet["peer_tickers"] = _extract_peer_tickers(peers)
    packet["quadrant_history"] = _safe_call_list(
        kg.get_quadrant_history,
        ticker,
    )


def _populate_quadrant_history_packet(
    packet: StructuralEvidencePacket,
    kg: Any,
    ticker: str,
) -> None:
    packet["quadrant_history"] = _safe_call_list(
        kg.get_quadrant_history,
        ticker,
    )


def _populate_factor_rotation_packet(
    packet: StructuralEvidencePacket,
    kg: Any,
    factor: Optional[str],
    from_regime: Optional[str],
    to_regime: Optional[str],
) -> None:
    if not factor:
        raise ValueError("factor_rotation requires 'factor'")
    if not from_regime or not to_regime:
        raise ValueError("factor_rotation requires both 'from_regime' and 'to_regime'")

    packet["target_factor"] = factor
    packet["from_regime"] = from_regime
    packet["to_regime"] = to_regime
    packet["factor_rotation"] = _safe_call_dict(
        kg.get_factor_rotation,
        factor,
        from_regime,
        to_regime,
    )


def _populate_regime_transition_packet(
    packet: StructuralEvidencePacket,
    kg: Any,
    from_regime: Optional[str],
    to_regime: Optional[str],
) -> None:
    if not from_regime or not to_regime:
        raise ValueError("regime_transition requires both 'from_regime' and 'to_regime'")

    packet["from_regime"] = from_regime
    packet["to_regime"] = to_regime
    packet["crowding_chain"] = _safe_call_dict(
        kg.query_crowding_chain,
        from_regime,
        to_regime,
    )


# ============================================================
# Validation + utility helpers
# ============================================================

def _validate_request_inputs(
    ticker: str,
    regime: str,
    question_type: str,
    factor: Optional[str],
    from_regime: Optional[str],
    to_regime: Optional[str],
    max_peers: int,
) -> None:
    if not ticker or not isinstance(ticker, str):
        raise ValueError("ticker must be a non-empty string")

    if not regime or not isinstance(regime, str):
        raise ValueError("regime must be a non-empty string")

    if question_type not in VALID_QUESTION_TYPES:
        raise ValueError(
            f"Unsupported question_type: {question_type}. "
            f"Expected one of: {sorted(VALID_QUESTION_TYPES)}"
        )

    if not isinstance(max_peers, int) or max_peers < 1:
        raise ValueError("max_peers must be an integer >= 1")

    if question_type == "factor_rotation":
        if not factor:
            raise ValueError("factor_rotation requires 'factor'")
        if not from_regime or not to_regime:
            raise ValueError("factor_rotation requires 'from_regime' and 'to_regime'")

    if question_type == "regime_transition":
        if not from_regime or not to_regime:
            raise ValueError("regime_transition requires 'from_regime' and 'to_regime'")


def _seed_node_ids(
    ticker: str,
    regime: str,
    question_type: str,
    factor: Optional[str],
    from_regime: Optional[str],
    to_regime: Optional[str],
) -> List[str]:
    """
    Build a bounded node seed list for serialize_subgraph().

    This uses a conservative naming convention aligned to the architecture plan.
    If your KG node IDs differ, adjust only this function.
    """
    node_ids = [
        f"stock:{ticker}",
        f"regime:{regime}",
    ]

    if question_type == "factor_rotation" and factor:
        node_ids.extend([
            f"factor:{factor}",
            f"regime:{from_regime}",
            f"regime:{to_regime}",
        ])

    if question_type == "regime_transition" and from_regime and to_regime:
        node_ids.extend([
            f"regime:{from_regime}",
            f"regime:{to_regime}",
        ])

    return _dedupe_preserve_order(node_ids)


def _safe_serialize_subgraph(kg: Any, node_ids: List[str]) -> Dict[str, Any]:
    """
    Safely serialize a bounded subgraph.
    """
    try:
        result = kg.serialize_subgraph(node_ids)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    return {"nodes": [], "edges": [], "error": "serialize_subgraph_failed"}


def _safe_call_dict(func: Any, *args: Any) -> Dict[str, Any]:
    """
    Safe wrapper around KG methods expected to return dicts.
    """
    try:
        result = func(*args)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return {}


def _safe_call_list(func: Any, *args: Any) -> List[Dict[str, Any]]:
    """
    Safe wrapper around KG methods expected to return list[dict].
    """
    try:
        result = func(*args)
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
    except Exception:
        pass
    return []


def _truncate_list(items: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    return items[:max_items]


def _extract_peer_tickers(peers: List[Dict[str, Any]]) -> List[str]:
    """
    Extract peer tickers from a peer list using a tolerant field search.
    """
    tickers: List[str] = []

    for peer in peers:
        ticker = (
            peer.get("ticker")
            or peer.get("peer_ticker")
            or peer.get("symbol")
        )
        if isinstance(ticker, str) and ticker:
            tickers.append(ticker)

    return _dedupe_preserve_order(tickers)


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result