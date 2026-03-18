"""
factor_registry.py
------------------
Single source of truth for factor definitions used across the app,
knowledge graph, narrative engine, and period analysis modules.
"""

# ============================================================
# Feature Key Constants (true single source of truth)
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
FEATURE_METADATA = {
    EY: {'display': 'Earnings Yield (V)'},
    BM: {'display': 'Book-to-Market (V)'},
    SP: {'display': 'Sales-to-Price (V)'},
    ROE: {'display': 'Return on Equity (Q)'},
    ROA: {'display': 'Return on Assets (Q)'},
    GPROF: {'display': 'Gross Profitability (Q)'},
    DEBT_ASSETS: {'display': 'Debt-to-Assets(FS)'},
    CASH_DEBT: {'display': 'Cash-to-Debt (FS)'},
    MOMENTUM: {'display': '12-Mo. Momentum (R)'},
    VOL: {'display': '60-Day Volatility (R)'},
    LIQUIDITY: {'display': 'Liquidity (R)'}
}

# Derived display names (backward-compatible)
FEATURE_DISPLAY_NAMES = {
    k: v['display'] for k, v in FEATURE_METADATA.items()
}

# Display order follows canonical feature list
FEATURE_DISPLAY_ORDER = FEATURE_LIST

FACTOR_CATEGORIES = {
    'Value': [EY, BM, SP],
    'Quality': [ROE, ROA, GPROF],
    'Financial Strength': [DEBT_ASSETS, CASH_DEBT],
    'Momentum': [MOMENTUM],
    'Risk/Volatility': [VOL],
    'Liquidity': [LIQUIDITY]
}

# For narrative / PCA interpretation defaults
PCA_DRIVER_GROUPS = {
    'PC1': [ROA, ROE, CASH_DEBT],
    'PC2': [SP, BM, EY],
    'PC3': [DEBT_ASSETS, VOL, GPROF],
}

# KG / ontology data-source mapping
FEATURE_DATA_SOURCES = {
    EY: 'WRDS/Compustat',
    BM: 'WRDS/Compustat',
    SP: 'WRDS/Compustat',
    ROE: 'WRDS/Compustat',
    ROA: 'WRDS/Compustat',
    GPROF: 'WRDS/Compustat',
    DEBT_ASSETS: 'WRDS/Compustat',
    CASH_DEBT: 'WRDS/Compustat',
    MOMENTUM: 'WRDS/CRSP',
    VOL: 'WRDS/CRSP',
    LIQUIDITY: 'WRDS/CRSP',
}

REGIME_ORDER = ["Post-COVID", "Rate Shock", "Disinflation"]