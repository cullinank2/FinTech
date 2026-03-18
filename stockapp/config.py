"""
Configuration settings and constants for the Stock PCA Cluster Analysis App.
Contains API endpoints, visualization settings, and factor definitions.
"""

# ==========================================================
# MASTER FEATURE REGISTRY (Single Source of Truth)
# ==========================================================

FEATURE_REGISTRY = {
    "earnings_yield": {
        "category": "Value",
        "label": "Earnings Yield",
        "source": "WRDS/Compustat"
    },
    "sales_to_price": {
        "category": "Value",
        "label": "Sales-to-Price",
        "source": "WRDS/Compustat"
    },
    "roe": {
        "category": "Quality",
        "label": "Return on Equity",
        "source": "WRDS/Compustat"
    },
    "roa": {
        "category": "Quality",
        "label": "Return on Assets",
        "source": "WRDS/Compustat"
    },
}

# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================

# GitHub raw URL for the factors dataset
# IMPORTANT: Replace this with your actual GitHub raw URL if different
GITHUB_DATA_URL = "https://raw.githubusercontent.com/DrOsmanDatascience/FinTech/main/data/factors_build_df_dec_10.csv"

# Backup/alternative data path for local development
LOCAL_DATA_PATH = "data/factors_build_df_dec_10.csv"


# =============================================================================
# OPENAI API CONFIGURATION
# =============================================================================

# IMPORTANT: Replace with your actual OpenAI API key or use environment variable
# Recommended: Set OPENAI_API_KEY environment variable instead of hardcoding
OPENAI_API_KEY_PLACEHOLDER = "YOUR_OPENAI_API_KEY_HERE"

# OpenAI model to use for the chatbot
OPENAI_MODEL = "gpt-4o-mini"  # Cost-effective option; use "gpt-4o" for better quality


from factor_registry import (
    FEATURE_COLUMNS,
    FEATURE_DISPLAY_NAMES,
    FEATURE_DISPLAY_ORDER,
    FACTOR_CATEGORIES,
)

# =============================================================================
# PCA AND CLUSTERING CONFIGURATION
# =============================================================================

# Number of principal components to compute
N_COMPONENTS = 3

# Number of clusters for KMeans
N_CLUSTERS = 4

# =============================================================================
# FEATURE CONFIGURATION (Registry-Driven)
# =============================================================================

# NOTE:
# All PCA features, display names, ordering, and categories
# are sourced from factor_registry.py (single source of truth).
# Do NOT define or duplicate feature lists in this file.

# Imported objects:
# - FEATURE_COLUMNS
# - FEATURE_DISPLAY_NAMES
# - FEATURE_DISPLAY_ORDER
# - FACTOR_CATEGORIES

# =============================================================================
# PCA AXIS INTERPRETATIONS (from visualization guide)
# =============================================================================

PC1_INTERPRETATION = {
    'name': 'Profitability & Operational Quality',
    'variance_explained': 37.5,
    'high_meaning': ['Operationally profitable', 'Stable', 'Strong cash position'],
    'low_meaning': ['Lower profitability', 'Weaker operations', 'Volatile', 'Cash-constrained'],
    'high_meaning_shorthand': 'Higher Quality',
    'low_meaning_shorthand': 'Lower Quality',
}

PC2_INTERPRETATION = {
    'name': 'Valuation Style: Growth vs Value',
    'variance_explained': 14.6,
    'high_meaning': ['Deep value stocks', 'High book value', 'High sales relative to price', 'Cheap on traditional metrics'],
    'low_meaning': ['Growth/premium stocks', 'Trading at premium to fundamentals'],
    'high_meaning_shorthand': 'Value',
    'low_meaning_shorthand': 'Growth',
}

PC3_INTERPRETATION = {
    'name': 'Leverage & Asset Intensity',
    'variance_explained': 12.0,
    'high_meaning': ['Leveraged with volatility', 'Higher debt', 'Volatile', 'Maybe decent margins'],
    'low_meaning': ['Conservative/stable', 'Lower leverage', 'Less volatile', 'Cleaner balance sheets'],
}

# =============================================================================
# QUADRANT DEFINITIONS
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


# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# Color palette for clusters (Dark2 colormap) 
CLUSTER_COLORS = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']

# Plot dimensions
PLOT_WIDTH = 900
PLOT_HEIGHT = 700

# Animation settings
ANIMATION_FRAME_DURATION = 500  # milliseconds per frame

# Streamlit page configuration
PAGE_CONFIG = {
    'page_title': 'Stock PCA Cluster Analysis',
    'page_icon': '📊',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}


# =============================================================================
# CHATBOT SYSTEM PROMPT
# =============================================================================

CHATBOT_SYSTEM_PROMPT = """You are a financial analysis assistant specializing in PCA 
(Principal Component Analysis) cluster analysis for stocks. You help users understand:

1. Which cluster a stock belongs to and what that means
2. How stocks compare to others in their cluster or quadrant
3. The meaning of PC1 (Quality/Stability) and PC2 (Size/Leverage) axes
4. Factor breakdowns including Value, Quality, Financial Strength, Momentum, Volatility, and Liquidity

Key interpretations:
- PC1 (X-axis): Quality/Stability - High values = profitable, financially strong; Low values = riskier, volatile
- PC2 (Y-axis): Size/Leverage - High values = large/liquid, leveraged; Low values = cash-rich, efficient

Provide concise, actionable insights for stakeholders. Use the provided context about the 
selected stock and its cluster to give specific, data-driven answers."""

# =============================================================================
# TEMPORARY TEST - DELETE AFTER TESTING
# =============================================================================

    