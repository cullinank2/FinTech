"""
structural_analyst.py
---------------------
AI interpretation layer for the KG-backed Structural Analyst.

Purpose
-------
Consumes a deterministic evidence packet, applies a strict prompt contract,
and returns a validated structured response.

Design goals
------------
- bounded by supplied evidence only
- explicit response schema
- safe fallback if model output is malformed
- decoupled from any specific OpenAI SDK implementation

Important integration note
--------------------------
This module expects an injected LLM callable instead of hard-coding a specific
SDK interface. That makes it easy to plug into:
- your existing chatbot/OpenAI wrapper
- a future Responses API client
- a mock test client

Expected llm_callable signature
-------------------------------
llm_callable(system_prompt: str, user_prompt: str) -> str

It must return raw text containing JSON only.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from structural_prompts import (
    STRUCTURAL_ANALYST_SYSTEM_PROMPT,
    build_structural_user_prompt,
)
from structural_types import (
    STRUCTURAL_ANALYSIS_MODE,
    VALID_CONFIDENCE_LEVELS,
    VALID_QUESTION_TYPES,
    StructuralAnalysisResponse,
    StructuralEvidencePacket,
)


def run_structural_analysis(
    evidence_packet: StructuralEvidencePacket,
    llm_callable: Callable[[str, str], str],
) -> StructuralAnalysisResponse:
    """
    Run KG-bounded structural analysis using the supplied LLM callable.

    Parameters
    ----------
    evidence_packet : StructuralEvidencePacket
        Deterministic context bundle built from structural_context_builder.py
    llm_callable : Callable[[str, str], str]
        Injected model-call function.
        Must accept:
            (system_prompt, user_prompt)
        and return raw text containing JSON.

    Returns
    -------
    StructuralAnalysisResponse
    """
    try:
        _validate_evidence_packet_minimum(evidence_packet)

        system_prompt = STRUCTURAL_ANALYST_SYSTEM_PROMPT
        user_prompt = build_structural_user_prompt(evidence_packet)

        raw = llm_callable(system_prompt, user_prompt)

        # ------------------------------------------------------------
        # Normalize LLM output to raw text
        # ------------------------------------------------------------
        if isinstance(raw, str):
            raw_text = raw

        elif isinstance(raw, dict):
            raw_text = raw.get("output_text", "") or str(raw)

        elif hasattr(raw, "output_text"):
            raw_text = getattr(raw, "output_text", "")

        elif hasattr(raw, "choices"):
            try:
                raw_text = raw.choices[0].message.content
            except Exception:
                raw_text = str(raw)

        else:
            raw_text = str(raw)
        parsed = _parse_response_json(raw_text)
        validated = _validate_structural_response(parsed, evidence_packet)

        return validated

    except Exception as exc:
        return _safe_fallback_response(
            evidence_packet=evidence_packet,
            reason=str(exc),
        )


# ============================================================
# Parsing
# ============================================================

def _parse_response_json(raw_text: str) -> Dict[str, Any]:
    """
    Parse model output as JSON.

    Supports:
    - direct JSON object text
    - fenced JSON block fallback cleanup
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Model returned empty response")

    cleaned = raw_text.strip()

    # Handle accidental fenced output
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON must be an object")

    return parsed


# ============================================================
# Validation
# ============================================================

def _validate_evidence_packet_minimum(
    evidence_packet: StructuralEvidencePacket,
) -> None:
    required_fields = ["question_type", "ticker", "regime"]

    for field in required_fields:
        if field not in evidence_packet:
            raise ValueError(f"Evidence packet missing required field: {field}")

    question_type = evidence_packet["question_type"]
    if question_type not in VALID_QUESTION_TYPES:
        raise ValueError(f"Invalid question_type in evidence packet: {question_type}")


def _validate_structural_response(
    response: Dict[str, Any],
    evidence_packet: StructuralEvidencePacket,
) -> StructuralAnalysisResponse:
    required_fields = [
        "question_type",
        "ticker",
        "regime",
        "answer",
        "summary_bullets",
        "evidence",
        "subgraph_snapshot",
        "limits",
        "confidence",
        "analysis_mode",
    ]

    for field in required_fields:
        if field not in response:
            raise ValueError(f"Missing required response field: {field}")

    if response["question_type"] != evidence_packet["question_type"]:
        raise ValueError("Response question_type mismatch")

    if response["ticker"] != evidence_packet["ticker"]:
        raise ValueError("Response ticker mismatch")

    if response["regime"] != evidence_packet["regime"]:
        raise ValueError("Response regime mismatch")

    if response["confidence"] not in VALID_CONFIDENCE_LEVELS:
        raise ValueError("Invalid confidence value")

    if response["analysis_mode"] != STRUCTURAL_ANALYSIS_MODE:
        raise ValueError("Invalid analysis_mode")

    if not isinstance(response["answer"], str):
        raise ValueError("answer must be a string")

    if not isinstance(response["summary_bullets"], list) or not all(
        isinstance(item, str) for item in response["summary_bullets"]
    ):
        raise ValueError("summary_bullets must be list[str]")

    if not isinstance(response["limits"], list) or not all(
        isinstance(item, str) for item in response["limits"]
    ):
        raise ValueError("limits must be list[str]")

    _validate_evidence_list(response["evidence"])
    _validate_subgraph_snapshot(response["subgraph_snapshot"])

    return {
        "question_type": response["question_type"],
        "ticker": response["ticker"],
        "regime": response["regime"],
        "answer": response["answer"],
        "summary_bullets": response["summary_bullets"],
        "evidence": response["evidence"],
        "subgraph_snapshot": response["subgraph_snapshot"],
        "limits": response["limits"],
        "confidence": response["confidence"],
        "analysis_mode": response["analysis_mode"],
    }


def _validate_evidence_list(evidence: Any) -> None:
    if not isinstance(evidence, list):
        raise ValueError("evidence must be a list")

    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError("Each evidence item must be an object")
        for field in ["source_type", "source_name", "fact"]:
            if field not in item or not isinstance(item[field], str):
                raise ValueError(f"Invalid evidence item field: {field}")


def _validate_subgraph_snapshot(snapshot: Any) -> None:
    if not isinstance(snapshot, dict):
        raise ValueError("subgraph_snapshot must be an object")

    required_fields = ["node_count", "edge_count", "included_node_ids"]
    for field in required_fields:
        if field not in snapshot:
            raise ValueError(f"subgraph_snapshot missing field: {field}")

    if not isinstance(snapshot["node_count"], int):
        raise ValueError("subgraph_snapshot.node_count must be int")

    if not isinstance(snapshot["edge_count"], int):
        raise ValueError("subgraph_snapshot.edge_count must be int")

    if not isinstance(snapshot["included_node_ids"], list) or not all(
        isinstance(item, str) for item in snapshot["included_node_ids"]
    ):
        raise ValueError("subgraph_snapshot.included_node_ids must be list[str]")


# ============================================================
# Safe fallback
# ============================================================

def _safe_fallback_response(
    evidence_packet: StructuralEvidencePacket,
    reason: str,
) -> StructuralAnalysisResponse:
    question_type = evidence_packet.get("question_type", "structural_drift")
    ticker = evidence_packet.get("ticker", "UNKNOWN")
    regime = evidence_packet.get("regime", "UNKNOWN")

    subgraph_snapshot = evidence_packet.get(
        "subgraph_snapshot_meta",
        {
            "node_count": 0,
            "edge_count": 0,
            "included_node_ids": [],
        },
    )

    return {
        "question_type": question_type,  # type: ignore[return-value]
        "ticker": ticker,
        "regime": regime,
        "answer": (
            "A structural interpretation could not be generated from the "
            "supplied evidence packet."
        ),
        "summary_bullets": [],
        "evidence": [],
        "subgraph_snapshot": subgraph_snapshot,
        "limits": [
            "Model output could not be validated",
            f"Processing reason: {reason}",
        ],
        "confidence": "low",
        "analysis_mode": "bounded_kg_v1_fallback",
    }