"""
structural_report_formatter.py
------------------------------
Generates structured, deterministic narrative reports from KG outputs.
No LLM usage. Safe for UI + export layers.
"""

from typing import Dict, List


# ============================================================
# Helpers
# ============================================================

def _title(text: str) -> str:
    return text.replace("_", " ").title() if text else "Unknown"


def _fmt_score(x):
    return f"{round(x,1)}" if isinstance(x, (int, float)) else "N/A"


# ============================================================
# SECTION 1 — Executive Summary
# ============================================================

def build_executive_summary(data: Dict) -> List[str]:
    rt = data.get("regime_transition", {})

    if not rt:
        return ["No structural signal detected."]

    return [
        f"{_title(rt.get('from'))} → {_title(rt.get('to'))} regime transition",
        f"Style shift: {_title(rt.get('quadrant_from'))} → {_title(rt.get('quadrant_to'))}",
        f"Crowding: {_title(rt.get('crowding_label'))} ({_fmt_score(rt.get('crowding_score'))})"
    ]


# ============================================================
# SECTION 2 — Regime Transition
# ============================================================

def build_regime_section(data: Dict) -> List[str]:
    rt = data.get("regime_transition", {})

    return [
        f"Previous regime: {_title(rt.get('from'))}",
        f"Current regime: {_title(rt.get('to'))}",
        f"Crowding score: {_fmt_score(rt.get('crowding_score'))}"
    ]


# ============================================================
# SECTION 3 — Factor Rotation
# ============================================================

def build_factor_rotation_section(data: Dict) -> List[str]:
    rotations = data.get("factor_rotation", [])

    if not rotations:
        return ["No significant factor rotations detected."]

    lines = []
    for r in rotations:
        lines.append(
            f"{_title(r.get('factor'))}: "
            f"{_fmt_score(r.get('from_loading'))} → {_fmt_score(r.get('to_loading'))}"
        )

    return lines


# ============================================================
# SECTION 4 — Crowding & Risk
# ============================================================

def build_crowding_section(data: Dict) -> List[str]:
    chain = data.get("crowding_chain", {})

    return [
        f"Crowding regime: {_title(chain.get('label'))}",
        f"Crowding score: {_fmt_score(chain.get('score'))}",
        f"Interpretation: {chain.get('interpretation', 'N/A')}"
    ]


# ============================================================
# SECTION 5 — Peer Context
# ============================================================

def build_peer_section(data: Dict) -> List[str]:
    peers = data.get("peers", [])

    if not peers:
        return ["No peer context available."]

    lines = []
    for p in peers[:5]:
        lines.append(
            f"{p.get('ticker')} — {_title(p.get('quadrant'))} "
            f"(distance: {_fmt_score(p.get('distance'))})"
        )

    return lines


# ============================================================
# MASTER REPORT BUILDER
# ============================================================

def build_structural_report(data: Dict) -> Dict[str, List[str]]:
    return {
        "Executive Summary": build_executive_summary(data),
        "Regime Transition": build_regime_section(data),
        "Factor Rotation": build_factor_rotation_section(data),
        "Crowding & Risk": build_crowding_section(data),
        "Peer Context": build_peer_section(data),
    }