"""
factor_registry.py
------------------
Single source of truth for factor definitions used across the app,
knowledge graph, narrative engine, and period analysis modules.
"""

# Canonical factor list (single source of truth)
FEATURE_LIST = [
    'earnings_yield',
    'bm',
    'sales_to_price',
    'roe',
    'roa',
    'gprof',
    'debt_assets',
    'cash_debt',
    'momentum_12m',
    'vol_60d_ann',
    'addv_63d'
]

# PCA feature columns (alias for backward compatibility)
FEATURE_COLUMNS = FEATURE_LIST

# Display names used across charts / UI
FEATURE_DISPLAY_NAMES = {
    'earnings_yield': 'Earnings Yield (V)',
    'bm': 'Book-to-Market (V)',
    'sales_to_price': 'Sales-to-Price (V)',
    'roe': 'Return on Equity (Q)',
    'roa': 'Return on Assets (Q)',
    'gprof': 'Gross Profitability (Q)',
    'debt_assets': 'Debt-to-Assets(FS)',
    'cash_debt': 'Cash-to-Debt (FS)',
    'momentum_12m': '12-Mo. Momentum (R)',
    'vol_60d_ann': '60-Day Volatility (R)',
    'addv_63d': 'Liquidity (R)'
}

# Display order follows canonical feature list
FEATURE_DISPLAY_ORDER = FEATURE_LIST

FACTOR_CATEGORIES = {
    'Value': ['earnings_yield', 'bm', 'sales_to_price'],
    'Quality': ['roe', 'roa', 'gprof'],
    'Financial Strength': ['debt_assets', 'cash_debt'],
    'Momentum': ['momentum_12m'],
    'Risk/Volatility': ['vol_60d_ann'],
    'Liquidity': ['addv_63d']
}

# For narrative / PCA interpretation defaults
PCA_DRIVER_GROUPS = {
    'PC1': ['roa', 'roe', 'cash_debt'],
    'PC2': ['sales_to_price', 'bm', 'earnings_yield'],
    'PC3': ['debt_assets', 'vol_60d_ann', 'gprof'],
}

# KG / ontology data-source mapping
FEATURE_DATA_SOURCES = {
    'earnings_yield': 'WRDS/Compustat',
    'bm': 'WRDS/Compustat',
    'sales_to_price': 'WRDS/Compustat',
    'roe': 'WRDS/Compustat',
    'roa': 'WRDS/Compustat',
    'gprof': 'WRDS/Compustat',
    'debt_assets': 'WRDS/Compustat',
    'cash_debt': 'WRDS/Compustat',
    'momentum_12m': 'WRDS/CRSP',
    'vol_60d_ann': 'WRDS/CRSP',
    'addv_63d': 'WRDS/CRSP',
}

REGIME_ORDER = ["Post-COVID", "Rate Shock", "Disinflation"]