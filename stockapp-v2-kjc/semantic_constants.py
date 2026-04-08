"""
ARCH: SHARED_SEMANTIC_CONSTANTS
ARCH: CANONICAL_SEMANTIC_LAYER
ARCH: NO_RUNTIME_LOGIC

semantic_constants.py
---------------------
Canonical shared semantic definitions for ESDS.

Purpose
-------
Centralize reusable semantic concepts that should not be duplicated across
config, KG, UI, or analysis layers.

Rules
-----
- Store canonical shared meanings here
- Do not place UI-only copy here
- Do not place runtime/business logic here
- Do not place Tier 2 prompt text here
"""

# =============================================================================
# REGIME SEMANTICS
# =============================================================================

REGIME_ORDER = [
    "Post-COVID",
    "Rate Shock",
    "Disinflation",
]

# =============================================================================
# THRESHOLD SEMANTICS
# =============================================================================

PROCRUSTES_NEGLIGIBLE = 0.05
PROCRUSTES_DETECTABLE = 0.15
PROCRUSTES_MEANINGFUL = 0.30

CROWDING_THRESHOLD_ELEVATED = 50.0
CROWDING_THRESHOLD_HIGH = 70.0

# =============================================================================
# QUADRANT SEMANTICS
# =============================================================================

QUADRANTS = {
    'Q1': {
        'name': 'Profitable Value',
        'pc1_sign': 'positive',
        'pc2_sign': 'positive',
        'description': '<b>THINK:</b> Strong profitability, trading at value prices',
        'characteristics': ['High profitability', 'Stable operations', 'Value priced', 'Quality fundamentals']
    },
    'Q2': {
        'name': 'Value Traps / Distressed',
        'pc1_sign': 'negative',
        'pc2_sign': 'positive',
        'description': '<b>THINK:</b> Weak operations, trading cheap on traditional metrics',
        'characteristics': ['Lower profitability', 'Volatile', 'Deep value priced', 'Potential traps']
    },
    'Q3': {
        'name': 'Struggling Growth',
        'pc1_sign': 'negative',
        'pc2_sign': 'negative',
        'description': '<b>THINK:</b> Unprofitable companies at growth premiums',
        'characteristics': ['Weak profitability', 'Volatile', 'Growth premium pricing', 'Speculative']
    },
    'Q4': {
        'name': 'Quality Growth',
        'pc1_sign': 'positive',
        'pc2_sign': 'negative',
        'description': '<b>THINK:</b> Strong profitability commanding growth premiums',
        'characteristics': ['High profitability', 'Stable operations', 'Growth premium pricing', 'Quality compounders']
    }
}