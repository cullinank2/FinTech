"""
factor_registry.py
------------------
Single source of truth for factor definitions used across the app,
knowledge graph, narrative engine, and period analysis modules.
"""

from types import MappingProxyType

# ============================================================
# Feature Key Constants (true single source of truth)
# ------------------------------------------------------------
# NOTE:
# These are canonical factor identifiers used across the entire system.
# They are intentionally defined as string constants and referenced
# everywhere else (PCA, KG, narrative engine, UI).
#
# Audit scanners may flag these as "hard-coded features" but this is
# expected and correct — this is the authoritative registry layer.
# ============================================================
EY = 'earnings_yield'
BM = 'bm'
SP = 'sales_to_price'
ROE = 'roe'
ROA = 'roa'
GPROF = 'gprof'
DEBT_ASSETS = 'debt_assets'
CASH_DEBT = 'cash_debt'
MOMENTUM = 'momentum_12m'
VOL = 'vol_60d_ann'
LIQUIDITY = 'addv_63d'

# Canonical factor keys (derived from constants)
ALL_FEATURE_KEYS = [
    EY,
    BM,
    SP,
    ROE,
    ROA,
    GPROF,
    DEBT_ASSETS,
    CASH_DEBT,
    MOMENTUM,
    VOL,
    LIQUIDITY
]

# Canonical factor list (backward-compatible alias)
FEATURE_LIST = ALL_FEATURE_KEYS

# PCA feature columns (alias for backward compatibility)
PCA_FEATURES = FEATURE_LIST
FEATURE_COLUMNS = PCA_FEATURES

# Display names used across charts / UI
# ============================================================
# Feature Metadata (single structured definition layer)
# ============================================================

# Internal mutable definition (never import directly)
_FEATURE_METADATA = {
    EY: {
        'display': 'Earnings Yield (V)',
        'category': 'Value',
        'pc': ['PC2'],
        'source': 'WRDS/Compustat',
    },
    BM: {
        'display': 'Book-to-Market (V)',
        'economic_meaning': 'valuation relative to fundamentals',
        'category': 'Value',
        'pc': ['PC2'],
        'source': 'WRDS/Compustat',
    },
    SP: {
        'display': 'Sales-to-Price (V)',
        'economic_meaning': 'revenue-based valuation relative to price',
        'category': 'Value',
        'pc': ['PC2'],
        'source': 'WRDS/Compustat',
    },
    ROE: {
        'display': 'Return on Equity (Q)',
        'economic_meaning': 'shareholder profitability and capital efficiency',
        'category': 'Quality',
        'pc': ['PC1'],
        'source': 'WRDS/Compustat',
    },
    ROA: {
        'display': 'Return on Assets (Q)',
        'economic_meaning': 'operational efficiency of assets',
        'category': 'Quality',
        'pc': ['PC1'],
        'source': 'WRDS/Compustat',
    },
    GPROF: {
        'display': 'Gross Profitability (Q)',
        'economic_meaning': 'core operating profitability before financing effects',
        'category': 'Quality',
        'pc': ['PC3'],
        'source': 'WRDS/Compustat',
    },
    DEBT_ASSETS: {
        'display': 'Debt-to-Assets(FS)',
        'economic_meaning': 'balance sheet leverage and financial risk burden',
        'category': 'Financial Strength',
        'pc': ['PC3'],
        'source': 'WRDS/Compustat',
    },
    CASH_DEBT: {
        'display': 'Cash-to-Debt (FS)',
        'economic_meaning': 'liquidity buffer relative to outstanding debt',
        'category': 'Financial Strength',
        'pc': ['PC1'],
        'source': 'WRDS/Compustat',
    },
    MOMENTUM: {
        'display': '12-Mo. Momentum (R)',
        'economic_meaning': 'price trend persistence and investor sentiment',
        'category': 'Momentum',
        'source': 'WRDS/CRSP',
    },
    VOL: {
        'display': '60-Day Volatility (R)',
        'economic_meaning': 'recent price variability and realized risk',
        'category': 'Risk/Volatility',
        'pc': ['PC3'],
        'source': 'WRDS/CRSP',
    },
    LIQUIDITY: {
        'display': 'Liquidity (R)',
        'economic_meaning': 'trading capacity and ease of entering or exiting positions',
        'category': 'Liquidity',
        'source': 'WRDS/CRSP',
    }
}

# Enforce immutability (read-only registry)
FEATURE_METADATA = MappingProxyType(_FEATURE_METADATA)

# Derived display names (backward-compatible)
FEATURE_DISPLAY_NAMES = {
    k: v['display'] for k, v in FEATURE_METADATA.items()
}

# Derived economic meanings for narrative / structural interpretation
FEATURE_ECONOMIC_MEANINGS = {
    k: v.get('economic_meaning', v['display']) for k, v in FEATURE_METADATA.items()
}

# Display order follows canonical feature list
FEATURE_DISPLAY_ORDER = FEATURE_LIST

# Derived factor categories (backward-compatible)
FACTOR_CATEGORIES = {}
for k, v in FEATURE_METADATA.items():
    cat = v.get('category')
    if cat:
        FACTOR_CATEGORIES.setdefault(cat, []).append(k)

# Derived PCA driver groups (backward-compatible)
PCA_DRIVER_GROUPS = {}
for k, v in FEATURE_METADATA.items():
    for pc in v.get('pc', []):
        PCA_DRIVER_GROUPS.setdefault(pc, []).append(k)

# KG / ontology data-source mapping (derived for backward compatibility)
FEATURE_DATA_SOURCES = {
    k: v.get('source') for k, v in FEATURE_METADATA.items()
}

REGIME_ORDER = ["Post-COVID", "Rate Shock", "Disinflation"]

# =============================================================================
# APPENDIX B LOADINGS (Compatibility Stub for KG Interface)
# =============================================================================

# NOTE:
# This is a fallback mapping used by kg_interface.
# It is intentionally left empty unless Appendix B data is explicitly defined.

APPENDIX_B_FACTOR_LOADINGS = {}