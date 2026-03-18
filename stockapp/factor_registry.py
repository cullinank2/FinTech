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

# ------------------------------------------------------------
# Attach category metadata to each feature
# ------------------------------------------------------------
FEATURE_METADATA[EY]['category'] = 'Value'
FEATURE_METADATA[BM]['category'] = 'Value'
FEATURE_METADATA[SP]['category'] = 'Value'

FEATURE_METADATA[ROE]['category'] = 'Quality'
FEATURE_METADATA[ROA]['category'] = 'Quality'
FEATURE_METADATA[GPROF]['category'] = 'Quality'

FEATURE_METADATA[DEBT_ASSETS]['category'] = 'Financial Strength'
FEATURE_METADATA[CASH_DEBT]['category'] = 'Financial Strength'

FEATURE_METADATA[MOMENTUM]['category'] = 'Momentum'
FEATURE_METADATA[VOL]['category'] = 'Risk/Volatility'
FEATURE_METADATA[LIQUIDITY]['category'] = 'Liquidity'

# Derived factor categories (backward-compatible)
FACTOR_CATEGORIES = {}
for k, v in FEATURE_METADATA.items():
    cat = v.get('category')
    if cat:
        FACTOR_CATEGORIES.setdefault(cat, []).append(k)

# For narrative / PCA interpretation defaults
# ------------------------------------------------------------
# Attach PCA driver metadata to each feature
# ------------------------------------------------------------
FEATURE_METADATA[ROA].setdefault('pc', []).append('PC1')
FEATURE_METADATA[ROE].setdefault('pc', []).append('PC1')
FEATURE_METADATA[CASH_DEBT].setdefault('pc', []).append('PC1')

FEATURE_METADATA[SP].setdefault('pc', []).append('PC2')
FEATURE_METADATA[BM].setdefault('pc', []).append('PC2')
FEATURE_METADATA[EY].setdefault('pc', []).append('PC2')

FEATURE_METADATA[DEBT_ASSETS].setdefault('pc', []).append('PC3')
FEATURE_METADATA[VOL].setdefault('pc', []).append('PC3')
FEATURE_METADATA[GPROF].setdefault('pc', []).append('PC3')

# Derived PCA driver groups (backward-compatible)
PCA_DRIVER_GROUPS = {}
for k, v in FEATURE_METADATA.items():
    for pc in v.get('pc', []):
        PCA_DRIVER_GROUPS.setdefault(pc, []).append(k)

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