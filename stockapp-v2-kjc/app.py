#Force rebuild - 2026-02-08
"""
ARCHITECTURE: PCA + Knowledge Graph + Narrative System

ARCH: TIER1_NARRATIVE_ENGINE
ARCH: APP_IS_SOLE_NE_CALLER
ARCH: KG_SERIALIZATION_ONLY

====================================================================
TIER 1 — NARRATIVE ENGINE (Deterministic, Auditable, Reportable)
====================================================================

BOUNDARY:
- The Narrative Engine is the ONLY component allowed to produce
  deterministic, governance-safe explanations.

AUTHORIZED CALL SITE:
- app.py → generate_narrative(...)

RULES:
- Must NOT depend on LLM output
- Must be reproducible across runs
- Must be explainable and auditable

INPUTS:
- PCA outputs (PC scores, loadings)
- Percentiles
- Factor breakdowns
- Knowledge Graph (optional, structured only)

OUTPUT:
- Structured narrative sections:
    - summary
    - factors
    - trajectory
    - peers
    - structural

ENFORCEMENT:
- Narrative Engine is NEVER imported or called outside app.py


====================================================================
DATA FLOW SUMMARY
====================================================================

RAW DATA → PCA → CLUSTERS → FACTORS
                      ↓
              KNOWLEDGE GRAPH
                      ↓
        ┌─────────────┴─────────────┐
        ↓                           ↓
TIER 1 (Narrative Engine)     TIER 2 (Chatbot)
Deterministic Output          LLM Exploration
Auditable                     Non-auditable
"""

import streamlit as st
import pandas as pd

# Import project modules
from config import (
    PAGE_CONFIG,
    GITHUB_DATA_URL,
    FEATURE_COLUMNS,
    FACTOR_CATEGORIES,
    QUADRANTS,
    PC1_INTERPRETATION,
    PC2_INTERPRETATION,
    PC3_INTERPRETATION,
    OPENAI_API_KEY_PLACEHOLDER,
    FEATURE_DISPLAY_NAMES
)
from utils import (
    load_data,
    preprocess_data,
    get_available_tickers,
    validate_stock_input,
    filter_stock_data,
    compute_pca_and_clusters,
    get_pca_loadings,
    determine_quadrant,
    get_stocks_in_same_quadrant,
    compute_percentile_ranks,
    get_cluster_summary,
    prepare_time_series_data,
    get_factor_breakdown,
    compute_crowding_scores
)
from visualizations import (
    create_pca_scatter_plot,
    create_quadrant_comparison_plot,
    create_factor_radar_chart,
    create_percentile_chart,
    create_factor_trend_chart,
    create_timelapse_animation,
    create_timelapse_animation_3d,
    create_3d_pca_plot,
    create_cluster_summary_plot,
    plot_crowding_score
)
from chatbot import create_chatbot, SAMPLE_QUESTIONS
from structural_context_builder import build_structural_evidence_packet
from kg_visualizer import render_kg_tab, render_structural_intelligence_tab
from period_analysis import (
    create_loading_comparison_chart,
    compute_procrustes_table,
    create_procrustes_heatmap,
    compute_quadrant_migration,
    create_migration_sankey,
    get_features_from_df,
    PERIOD_KEYS,
)
from narrative_engine import generate_narrative  # ARCH: intentional NE boundary — app.py is the sole authorized caller


# =============================================================================
# PAGE CONFIGURATION 
# =============================================================================

st.set_page_config(**PAGE_CONFIG)


# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

st.markdown("""
<style>
    .hero-text {
        font-size: 1.2rem;   /* increase size */
        font-weight: 500;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Main header styling */
    .main-header {
        font-size: 1.7rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    /* Info box styling */
    .info-box {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }

    .info-box * {
         color: var(--text-color) !important;
    }

    #.info-box {
    #    background-color: #f0f8ff;
    #    padding: 1rem;
    #    border-radius: 10px;
    #    border-left: 5px solid #1f77b4;
    #    margin: 1rem 0;
    #}
    
    /* Quadrant indicator */
    .quadrant-box {
        background-color: var(--secondary-background-color);   /* theme-aware bg */
        color: var(--text-color);                              /* theme-aware text */
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid rgba(128,128,128,0.25);              /* neutral in both */
    }

    /* Force readable text for all nested elements */
    .quadrant-box * {
        color: var(--text-color) !important;
    }
            
    #.quadrant-box {
    #    background-color: #e8f4e8;
    #    padding: 1.5rem;
    #    border-radius: 10px;
    #    text-align: center;
    #    margin: 0.5rem 0;
    #}
    
    /* Metric styling */
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}

/* USER MESSAGE */
.user-message {
    background-color: var(--secondary-background-color);
    color: var(--text-color) !important;
    border-left: 4px solid #3b82f6;
}

/* ASSISTANT MESSAGE */
.assistant-message {
    background-color: var(--secondary-background-color);
    color: var(--text-color) !important;
    border-left: 4px solid #10b981;
}

/* FORCE TEXT VISIBILITY */
.user-message * {
    color: var(--text-color) !important;
}

.assistant-message * {
    color: var(--text-color) !important;
}

/* DARK MODE FIX */
@media (prefers-color-scheme: dark) {
    .user-message {
        background-color: #1e293b !important;
        color: #ffffff !important;
    }

    .assistant-message {
        background-color: #111827 !important;
        color: #f9fafb !important;
    }

    .user-message * {
        color: #ffffff !important;
    }

    .assistant-message * {
        color: #f9fafb !important;
    }
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# TAB HEADER STYLING (STREAMLIT DOM OVERRIDE)
# =============================================================================

st.markdown("""
<style>

/* ===============================
   THEME VARIABLES
=============================== */
:root {
    --tab-color: #1f77b4;
}

html[data-theme="dark"] {
    --tab-color: #4da3ff;
}
            
/* -------------------------------
   ALL TAB LABELS
--------------------------------*/
button[data-baseweb="tab"] * {
    font-size: 1rem !important;
    color: var(--tab-color) !important;
    font-weight: 500 !important;
}

/* -------------------------------
   ACTIVE TAB
--------------------------------*/
button[aria-selected="true"] * {
    font-size: 1.5rem !important;
    color: var(--tab-color) !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'data_loaded': False,
        'raw_data': None,
        'processed_data': None,
        'pca_df': None,
        'pca_model': None,
        'kmeans_model': None,
        'scaler': None,
        'selected_stock': None,
        'selected_stock_data': None,
        'chatbot': None,
        'chat_history': [],
        'selected_gics_sector': 'All Sectors'
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data(show_spinner=True)
def load_and_process_data():
    """Load and preprocess data with caching."""
    try:
        # Load data from GitHub
        raw_data = load_data(use_github=True)
        
        # Preprocess data
        processed_data = preprocess_data(raw_data)
        
        # Compute PCA and clustering
        pca_df, pca_model, kmeans_model, scaler, loadings, loadings_df = compute_pca_and_clusters(processed_data)
        
        return raw_data, processed_data, pca_df, pca_model, kmeans_model, scaler, loadings, loadings_df, None
        
    except Exception as e:
        return None, None, None, None, None, None, None, None, str(e)


# =============================================================================
# SIDEBAR COMPONENTS 
# =============================================================================

def render_sidebar():
    """Render the sidebar with scope-specific controls."""

    st.sidebar.markdown("## Navigation & Scope")
    st.sidebar.caption(
        "Select the diagnostic workspace. Universe mode monitors system-level structure; "
        "Stock mode evaluates issuer-level position within that structure."
    )

    st.session_state.analysis_scope = st.sidebar.radio(
        "Diagnostic Workspace",
        [
            "Universe / Portfolio Level",
            "Stock / Individual Ticker Level",
        ],
        index=0 if st.session_state.analysis_scope == "Universe / Portfolio Level" else 1,
        help="Switch between market-wide structural diagnostics and single-stock analysis."
    )

    stock_scope_active = (
        st.session_state.analysis_scope == "Stock / Individual Ticker Level"
    )

    # stock_selected will be computed AFTER selection logic
    stock_selected = False

    if stock_scope_active:
        st.sidebar.markdown("---")
        st.sidebar.markdown("## Issuer Selection")
        st.sidebar.caption(
            "Choose the equity to analyze in the Stock / Individual Diagnostics workspace."
        )

        # Stock input
        #st.sidebar.markdown("""
        #<div class="info-box">
        #    Enter stock ticker (e.g., AAPL, MSFT) to analyze.
        #</div>
        #""", unsafe_allow_html=True)

        stock_input = st.sidebar.text_input(
            "Enter Stock Ticker:",
            placeholder="e.g., AAPL or 14593",
            key="stock_input"
        )
    else:
        stock_input = ""
        st.session_state.selected_stock = None
        if 'ticker_dropdown' in st.session_state:
            st.session_state.ticker_dropdown = ''
    
    # Validation and selection
    if stock_input and st.session_state.processed_data is not None:
        is_valid, input_type, normalized_value = validate_stock_input(
            st.session_state.processed_data, 
            stock_input
        )
        
        if is_valid:
            st.sidebar.success(f"✅ Found: {normalized_value} ({input_type})")
            st.session_state.selected_stock = {
                'value': normalized_value,
                'type': input_type
            }
        else:
            st.sidebar.error(f"❌ '{stock_input}' not found in dataset")
            st.session_state.selected_stock = None
    elif not stock_input:
        # Ticker cleared — reset stock selection so GICS filter reactivates
        st.session_state.selected_stock = None
        # Also clear Quick Select dropdown
        if 'ticker_dropdown' in st.session_state:
            st.session_state.ticker_dropdown = ''
    
    # Quick selection dropdown (stock scope only)
    if stock_scope_active:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Quick Select")
        st.sidebar.caption("Use the searchable list below as an alternative to manual ticker entry.")
        
        # Check if data is ready
        data_ready = (
            st.session_state.pca_df is not None and 
            'ticker' in st.session_state.pca_df.columns
        )
        
        # Prepare ticker list
        if data_ready:
            tickers = [''] + sorted(st.session_state.pca_df['ticker'].dropna().unique().tolist())
        else:
            tickers = ['Loading stock list...']
        
        # Selectbox (disabled if data not ready)
        selected_dropdown = st.sidebar.selectbox(
            "Or choose from list:",
            options=tickers,
            key="ticker_dropdown",
            disabled=not data_ready
        )
        
        # Update selected stock only if data is ready and valid selection
        if data_ready and selected_dropdown and selected_dropdown != '':
            st.session_state.selected_stock = {
                'value': selected_dropdown,
                'type': 'ticker'
            }

    # Check if stock is selected (UPDATED AFTER INPUT + DROPDOWN)
    stock_selected = st.session_state.selected_stock is not None

    # Universe-only landing page controls
    if not stock_scope_active:
        st.sidebar.markdown("---")
        st.sidebar.markdown("## Universe Controls")
        st.sidebar.caption(
            "Configure the market-wide dashboard view and sector-level universe focus."
        )
        st.sidebar.markdown("### GICS Sector Filter")

        gics_sectors = [
            "All Sectors",
            "Energy",
            "Materials",
            "Industrials",
            "Consumer Discretionary",
            "Consumer Staples",
            "Health Care",
            "Financials",
            "Information Technology",
            "Communication Services",
            "Utilities",
            "Real Estate"
        ]

        # Build sector options with stock counts
        if st.session_state.pca_df is not None and st.session_state.raw_data is not None:
            pca_tickers = set(st.session_state.pca_df['ticker'].str.upper().tolist())
            raw = st.session_state.raw_data
            all_sectors_count = len(st.session_state.pca_df)
            gics_sectors_with_counts = [f"All Sectors ({all_sectors_count})"]
            for sector in gics_sectors[1:]:  # skip "All Sectors"
                if 'gicdesc' in raw.columns and 'ticker' in raw.columns:
                    count = raw[
                        (raw['gicdesc'] == sector) &
                        (raw['ticker'].str.upper().isin(pca_tickers))
                    ]['ticker'].nunique()
                    gics_sectors_with_counts.append(f"{sector} ({count})")
                else:
                    gics_sectors_with_counts.append(sector)
        else:
            gics_sectors_with_counts = gics_sectors

        # Store options so callback can access them
        st.session_state['_gics_sector_options'] = gics_sectors_with_counts

        def on_sector_change():
            raw_selection = st.session_state.get('gics_sector_filter', 'All Sectors')
            st.session_state.selected_gics_sector = raw_selection.split(" (")[0]

        # Find the index matching the previously selected sector
        current_sector = st.session_state.get('selected_gics_sector', 'All Sectors')
        current_index = 0
        for i, opt in enumerate(gics_sectors_with_counts):
            if opt.split(" (")[0] == current_sector:
                current_index = i
                break

        st.sidebar.selectbox(
            "Filter landing page by sector:",
            options=gics_sectors_with_counts,
            index=current_index,
            key="gics_sector_filter",
            on_change=on_sector_change,
            help="Select a GICS sector to show only that sector's stocks in the Cluster Plot"
        )

    # Visuals tab controls (stock scope only)
    if stock_scope_active:
        st.sidebar.markdown("### Visual Diagnostics")
        st.sidebar.caption(
            "Use the top tabs for page navigation. These controls apply only to the Visuals workspace."
        )
        
        view_options = [
            "🎯 Cluster Plot",
            "👥 Quadrant Peers",
            "📊 Factor Analysis",
            "🕐 2D or 3D Time-Lapse",
            "🌐 3D Cluster View",
            "🌐 3D Quadrant Peers",
        ]
        
        # Preserve existing visualization logic, but reposition mentally as tab-specific
        if 'current_view' not in st.session_state:
            st.session_state.current_view = view_options[0]
        
        current_index = (
            view_options.index(st.session_state.current_view)
            if st.session_state.current_view in view_options
            else 0
        )
        
        selected_view = st.sidebar.selectbox(
            "Visual shown in 📊 Visuals tab:",
            options=view_options,
            index=current_index,
            key="view_selector",
            disabled=not stock_selected
        )
        
        if stock_selected:
            st.session_state.current_view = selected_view

    # GICS Sector filter (stock scope only)
    if stock_scope_active:
        st.sidebar.markdown("### Peer Universe Filter")

        if stock_selected:
            # Get total stock count from PCA dataframe
            total_stocks = len(st.session_state.pca_df) if st.session_state.pca_df is not None else 0
            
            # Get GICS sector stock count from PCA dataframe
            sector_stocks = 0
            if st.session_state.pca_df is not None and st.session_state.raw_data is not None and st.session_state.selected_stock is not None:
                ticker = st.session_state.selected_stock['value']
                if 'ticker' in st.session_state.raw_data.columns and 'gicdesc' in st.session_state.raw_data.columns:
                    ticker_data = st.session_state.raw_data[
                        st.session_state.raw_data['ticker'].str.upper() == ticker.upper()
                    ]
                    if not ticker_data.empty:
                        selected_sector = ticker_data['gicdesc'].iloc[0]
                        pca_tickers_upper = [t.upper() for t in st.session_state.pca_df['ticker'].tolist()]
                        sector_stocks = st.session_state.raw_data[
                            (st.session_state.raw_data['gicdesc'] == selected_sector) &
                            (st.session_state.raw_data['ticker'].str.upper().isin(pca_tickers_upper))
                        ]['ticker'].nunique()
            
            filter_options = [
                f"All Stocks ({total_stocks})",
                f"GICS Sector Only ({sector_stocks})"
            ]
        else:
            # Placeholder options when no stock selected
            filter_options = [
                "All Stocks",
                "GICS Sector Only"
            ]

        selected_filter = st.sidebar.selectbox(
            "Show stocks from:",
            options=filter_options,
            index=1 if stock_selected else 0,
            key="gics_filter",
            disabled=not stock_selected,
            help="Filter visualizations to show all stocks or only stocks in the same GICS sector"
        )
        
        # Extract the filter mode (without the count)
        if "All Stocks" in selected_filter:
            filter_mode = "All Stocks"
        else:
            filter_mode = "GICS Sector Only"
        
        # Store filter selection in session state
        if 'gics_filter_mode' not in st.session_state:
            st.session_state.gics_filter_mode = "GICS Sector Only"
        
        # THIS LINE IS CRITICAL - actually saves the selection!
        st.session_state.gics_filter_mode = filter_mode

    # Axis interpretations (moved below controls for clarity)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Factor Axis Interpretation")
    st.sidebar.caption(
        "Interpret the structural meaning of PCA axes used throughout the diagnostics."
    )

    # Get live variance values from PCA model if available
    pc1_var = PC1_INTERPRETATION['variance_explained']
    pc2_var = PC2_INTERPRETATION['variance_explained']
    pc3_var = PC3_INTERPRETATION['variance_explained']

    if st.session_state.pca_model is not None:
        ratios = st.session_state.pca_model.explained_variance_ratio_
        if len(ratios) >= 1:
            pc1_var = round(ratios[0] * 100, 1)
        if len(ratios) >= 2:
            pc2_var = round(ratios[1] * 100, 1)
        if len(ratios) >= 3:
            pc3_var = round(ratios[2] * 100, 1)

    with st.sidebar.expander(f"PC1 (X-axis): {PC1_INTERPRETATION['name']}"):
        st.markdown(f"""
        **Explains ~{pc1_var}% of variance**
        
        **High values (→ Right):**
        - {', '.join(PC1_INTERPRETATION['high_meaning'])}
        
        **Low values (← Left):**
        - {', '.join(PC1_INTERPRETATION['low_meaning'])}
        """)

    with st.sidebar.expander(f"PC2 (Y-axis): {PC2_INTERPRETATION['name']}"):
        st.markdown(f"""
        **Explains ~{pc2_var}% of variance**
        
        **High values (↑ Up):**
        - {', '.join(PC2_INTERPRETATION['high_meaning'])}
        
        **Low values (↓ Down):**
        - {', '.join(PC2_INTERPRETATION['low_meaning'])}
        """)

    active_visual = st.session_state.get('current_view', '')
    timelapse_is_3d = (
        active_visual == "🕐 2D or 3D Time-Lapse" and
        st.session_state.get('timelapse_view_mode', '2D View') == '3D View'
    )

    if st.session_state.get('selected_stock') is not None and (
        active_visual in ["🌐 3D Cluster View", "🌐 3D Quadrant Peers"] or timelapse_is_3d
    ):
        with st.sidebar.expander(f"PC3 (Z-axis): {PC3_INTERPRETATION['name']}"):
            st.markdown(f"""
            **Explains ~{pc3_var}% of variance**

            **High values (↑ Up):**
            - {', '.join(PC3_INTERPRETATION['high_meaning'])}
        
            **Low values (↓ Down):**
            - {', '.join(PC3_INTERPRETATION['low_meaning'])}
            """)

    # OpenAI API Key input
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Chatbot Settings")
    
    user_api_key = st.sidebar.text_input(
        "OpenAI API Key (optional):",
        type="password",
        placeholder="Enter your own key or leave blank",
        help="If left blank, the app will use its default API key"
    )
    
    st.session_state.chatbot = create_chatbot(user_api_key or None)
    
    if not st.session_state.chatbot.is_available():
        st.sidebar.warning("⚠️ No API key available.")
     


# =============================================================================
# MAIN CONTENT COMPONENTS
# =============================================================================

def call_structural_llm(system_prompt: str, user_prompt: str) -> str:
    """LLM adapter for the KG-backed Structural Analyst."""
    chatbot = st.session_state.get("chatbot")

    if chatbot is None or not chatbot.is_available():
        raise ValueError("Chatbot/OpenAI client is not available")

    response = chatbot.client.responses.create(
        model="gpt-5",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=1200,
    )

    return response.output_text


def render_main_header():
    """Render the main page header."""
    st.markdown("""
    <div class="main-header">
        📈 EQUITY STRUCTURAL DIAGNOSTICS
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero-text">
    Identify emerging structural risk in equities by quantifying factor crowding, regime shifts, 
                and cross-sectional instability using PCA-based diagnostics.
    </div>
    """, unsafe_allow_html=True)


def _build_quadrant_history_html(stock_data: pd.DataFrame, ticker: str) -> str:
    """
    Build HTML for a 3-quarter quadrant history panel.
    Projects raw factor data through the fitted scaler+PCA to get PC1/PC2
    for each historical quarter, then determines quadrant.
    """
    quadrant_colors = {
        'Q1': ('#10B981', '🟢'),
        'Q2': ('#EF4444', '🔴'),
        'Q3': ('#F59E0B', '🟡'),
        'Q4': ('#3B82F6', '🔵'),
    }

    history_rows = []

    scaler  = st.session_state.get('scaler')
    pca_model = st.session_state.get('pca_model')

    date_col = next(
        (c for c in ['public_date', 'date', 'datadate'] if c in stock_data.columns),
        None
    )

    if date_col and scaler is not None and pca_model is not None:
        ts = stock_data[stock_data['ticker'].str.upper() == ticker.upper()].copy()
        ts[date_col] = pd.to_datetime(ts[date_col])
        ts = ts.sort_values(date_col)

        # Keep only feature columns that exist in this data
        available_features = [c for c in FEATURE_COLUMNS if c in ts.columns]

        if len(available_features) >= 2:
            # Fill missing values
            ts[available_features] = ts[available_features].fillna(ts[available_features].median())

            # Project each row through scaler + PCA
            X_scaled = scaler.transform(ts[available_features])
            X_pca    = pca_model.transform(X_scaled)
            ts['_PC1'] = X_pca[:, 0]
            ts['_PC2'] = X_pca[:, 1]

            # Bucket by calendar quarter, take last obs per quarter
            ts['_qtr'] = ts[date_col].dt.to_period('Q')
            qtr_df = (
                ts.groupby('_qtr')
                .last()
                .reset_index()
                .sort_values('_qtr', ascending=False)
            )

            # Skip most recent quarter (= current card), take prior 3
            past_qtrs = qtr_df.iloc[1:4]

            for _, row in past_qtrs.iterrows():
                q = determine_quadrant(row['_PC1'], row['_PC2'])
                q_info = QUADRANTS.get(q, {})
                color, icon = quadrant_colors.get(q, ('#888', '⬜'))
                history_rows.append({
                    'label':    str(row['_qtr']),
                    'quadrant': q,
                    'name':     q_info.get('name', ''),
                    'color':    color,
                    'icon':     icon,
                })

    if not history_rows:
        return """
        <div style="padding:0.6rem 0.8rem; border-radius:8px;
                    border:1px solid rgba(128,128,128,0.25);
                    background:var(--secondary-background-color);">
            <div style="font-size:12px; color:gray; margin-bottom:6px; font-weight:600;
                        text-transform:uppercase; letter-spacing:0.05em;">
                Quadrant History
            </div>
            <div style="font-size:13px; color:gray; font-style:italic;">
                Insufficient historical data
            </div>
        </div>
        """

    rows_html = ""
    for i, r in enumerate(history_rows):
        ago_label = ["1Q ago", "2Q ago", "3Q ago"][i]
        rows_html += f"""
        <div style="display:flex; align-items:center; gap:8px;
                    padding:4px 0; border-bottom:1px solid rgba(128,128,128,0.12);">
            <span style="font-size:12px; color:gray; width:48px; flex-shrink:0;">{ago_label}</span>
            <span style="font-size:11px; color:gray; width:52px; flex-shrink:0;">{r['label']}</span>
            <span style="font-size:13px; font-weight:700; color:{r['color']}; flex-shrink:0;">
                {r['icon']} {r['quadrant']}
            </span>
            <span style="font-size:11px; color:{r['color']}; white-space:nowrap;">
                {r['name']}
            </span>
        </div>
        """

    return f"""
    <div style="padding:0.6rem 0.8rem; border-radius:8px;
                border:1px solid rgba(128,128,128,0.25);
                background:var(--secondary-background-color);">
        <div style="font-size:12px; color:gray; margin-bottom:6px; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.05em;">
            📅 Quadrant History
        </div>
        {rows_html}
    </div>
    """


def render_stock_overview(stock_data: pd.DataFrame, pca_row: pd.Series):
    """Render the stock overview section.""" 
    
    ticker = pca_row.get('ticker', 'N/A')
    permno = pca_row.get('permno', 'N/A')
    cluster = pca_row.get('cluster', 'N/A')
    pc1 = pca_row.get('PC1', 0)
    pc2 = pca_row.get('PC2', 0)
    quadrant = determine_quadrant(pc1, pc2)
    quadrant_info = QUADRANTS.get(quadrant, {})
    
    # Get GICS Sector by filtering raw data for this specific ticker
    gics_sector = 'N/A'
    if 'ticker' in stock_data.columns and 'gicdesc' in stock_data.columns:
        ticker_match = stock_data[stock_data['ticker'].str.upper() == ticker.upper()]
        if not ticker_match.empty:
            gics_sector = ticker_match['gicdesc'].iloc[0]
    
    # Display header with GICS Sector inline
    st.markdown(f"## 📊 Overview: {ticker} &nbsp;&nbsp;&nbsp;&nbsp; **GICS Sector:** {gics_sector}")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ticker", ticker)
    with col2:
        st.metric("PERMNO", permno)
    with col3:
        st.metric("Cluster", f"Cluster {cluster}")
    with col4:
        st.metric("Quadrant", quadrant)
    
    # Quadrant description
    st.markdown(f"""
    <div class="quadrant-box">
        <h4>{quadrant}: {quadrant_info.get('name', 'Unknown')}</h4>
        <p>{quadrant_info.get('description', '')}</p>
        <p><strong>Characteristics:</strong> {', '.join(quadrant_info.get('characteristics', []))}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # PCA scores + quadrant history side by side
    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        col1, col2 = st.columns(2)
        with col1:
            pc1_label = PC1_INTERPRETATION['high_meaning_shorthand'] if pc1 >= 0 else PC1_INTERPRETATION['low_meaning_shorthand']
            pc1_arrow = "↑" if pc1 >= 0 else "↓"
            pc1_color = "#10B981" if pc1 >= 0 else "#EF4444"
            st.metric(f"PC1 Score ({PC1_INTERPRETATION['name']})", f"{pc1:.3f}")
            st.markdown(f"<div style='margin-top:-30px; color:{pc1_color}; font-size:14px'>{pc1_arrow} {pc1_label}</div>", unsafe_allow_html=True)
        with col2:
            pc2_label = PC2_INTERPRETATION['high_meaning_shorthand'] if pc2 >= 0 else PC2_INTERPRETATION['low_meaning_shorthand']
            pc2_arrow = "↑" if pc2 >= 0 else "↓"
            st.metric(f"PC2 Score ({PC2_INTERPRETATION['name']})", f"{pc2:.3f}")
            st.markdown(f"<div style='margin-top:-30px; color:gray; font-size:14px'>{pc2_arrow} {pc2_label}</div>", unsafe_allow_html=True)

    with right_col:
        # --- Quadrant History: last 3 quarters ---
        history_html = _build_quadrant_history_html(st.session_state.raw_data, ticker)
        st.components.v1.html(history_html, height=140, scrolling=False)

    return ticker, permno, cluster, pc1, pc2, quadrant


def filter_by_gics_sector(pca_df: pd.DataFrame, raw_data: pd.DataFrame, ticker: str, filter_mode: str) -> pd.DataFrame:
    """
    Filter PCA dataframe by GICS sector if requested.
    
    Args:
        pca_df: Full PCA DataFrame with all stocks
        raw_data: Raw data containing GICS sector info
        ticker: Selected stock ticker
        filter_mode: "All Stocks" or "GICS Sector Only"
    
    Returns:
        Filtered or full PCA DataFrame
    """
    if filter_mode == "All Stocks":
        return pca_df
    
    # Get selected stock's GICS sector
    if 'ticker' not in raw_data.columns or 'gicdesc' not in raw_data.columns:
        return pca_df  # Fallback to all stocks if columns missing
    
    # Get the selected stock's sector
    ticker_data = raw_data[raw_data['ticker'].str.upper() == ticker.upper()]
    if ticker_data.empty:
        return pca_df
    
    selected_sector = ticker_data['gicdesc'].iloc[0]
    
    # Get all tickers in the same sector from raw_data
    sector_tickers = raw_data[raw_data['gicdesc'] == selected_sector]['ticker'].unique()
    
    # Filter out NaN values and convert to uppercase, handling non-string values
    sector_tickers_upper = [
        str(t).upper() for t in sector_tickers 
        if pd.notna(t) and t != ''
    ]
    
    # Filter PCA dataframe to only include tickers in this sector
    # Use case-insensitive matching
    filtered_pca_df = pca_df[pca_df['ticker'].str.upper().isin(sector_tickers_upper)]
    
    return filtered_pca_df


def render_visualizations(
    pca_df: pd.DataFrame,
    selected_ticker: str,
    pca_row: pd.Series,
    quadrant_peers: pd.DataFrame,
    raw_data: pd.DataFrame,
    pca_model,
    scaler
):
    """Render the active visual inside the Stock Visuals tab.""" 
    
    # Active visual selection for the Visuals tab
    active_visual = st.session_state.get('current_view', '🎯 Cluster Plot')
    
    # Apply GICS sector filter if selected
    filter_mode = st.session_state.get('gics_filter_mode', 'All Stocks')
    filtered_pca_df = filter_by_gics_sector(pca_df, raw_data, selected_ticker, filter_mode)
    
    # Show info about filtering
    if filter_mode == "GICS Sector Only":
        sector_count = len(filtered_pca_df)
        st.info(f"📊 Showing {sector_count} stocks in the same GICS sector as {selected_ticker}")
    
    # Display the selected visualization
    if active_visual == "🎯 Cluster Plot":
        st.markdown("### 🎯 PCA Cluster Visualization")
        
        # Get live variance values
        pc1_var = PC1_INTERPRETATION['variance_explained']  # fallback
        pc2_var = PC2_INTERPRETATION['variance_explained']  # fallback
        if st.session_state.pca_model is not None:
            ratios = st.session_state.pca_model.explained_variance_ratio_
            if len(ratios) >= 1:
                pc1_var = round(ratios[0] * 100, 1)
            if len(ratios) >= 2:
                pc2_var = round(ratios[1] * 100, 1)
        combined_var = round(pc1_var + pc2_var, 1)
        
        st.markdown(f"""
        This plot shows stocks positioned based on their Principal Components (PC) characteristics. Your selected stock is highlighted with a ⭐.
        
        *PC1 explains ~{pc1_var}% of variance (X-axis) and PC2 explains ~{pc2_var}% of variance (Y-axis); Combined variance explained ~{combined_var}%*
        """)
        
        fig = create_pca_scatter_plot(filtered_pca_df, selected_ticker)
        st.plotly_chart(fig, use_container_width=True)
        
        # ============================================================
        # TEMPORARY DEBUG: PCA Loadings Table
        # ============================================================
        st.markdown("---")
        st.markdown("### 🔧 DEBUG: Live PCA Loadings")
        
        if 'pca_loadings' in st.session_state:
            loadings = st.session_state.pca_loadings
            
            # Build debug table
            debug_data = []
            for pc in ['PC1', 'PC2', 'PC3']:
                if pc in loadings:
                    # Add positive loadings
                    if 'positive' in loadings[pc]:
                        for feat, val in loadings[pc]['positive'].items():
                            debug_data.append({
                                'PC': pc,
                                'Type': 'Positive',
                                'Feature': feat,
                                'Display Name': FEATURE_DISPLAY_NAMES.get(feat, feat),
                                'Loading': val
                            })
                    
                    # Add negative loadings
                    if 'negative' in loadings[pc]:
                        for feat, val in loadings[pc]['negative'].items():
                            debug_data.append({
                                'PC': pc,
                                'Type': 'Negative',
                                'Feature': feat,
                                'Display Name': FEATURE_DISPLAY_NAMES.get(feat, feat),
                                'Loading': val
                            })
            
            if debug_data:
                debug_df = pd.DataFrame(debug_data)
                st.dataframe(debug_df, use_container_width=True)
            else:
                st.warning("No loadings data available")
        else:
            st.warning("pca_loadings not found in session state")
    
    elif active_visual == "👥 Quadrant Peers":

        # ---------------------------------------------------------
        # Header Section
        # ---------------------------------------------------------
        st.markdown("### 👥 Quadrant Peer Comparison")

        # Safely extract PC values
        pc1 = pca_row.get('PC1', 0)
        pc2 = pca_row.get('PC2', 0)

        # Determine quadrant label
        if pc1 >= 0 and pc2 >= 0:
            quadrant_label = "Q1"
        elif pc1 < 0 and pc2 >= 0:
            quadrant_label = "Q2"
        elif pc1 < 0 and pc2 < 0:
            quadrant_label = "Q3"
        else:
            quadrant_label = "Q4"

        quadrant_name = QUADRANTS.get(quadrant_label, {}).get("name", "Unknown Quadrant")
        quadrant_desc = QUADRANTS.get(quadrant_label, {}).get("description", "")

        # Recalculate quadrant peers from filtered data
        filtered_quadrant_peers = get_stocks_in_same_quadrant(
            filtered_pca_df, pc1, pc2, exclude_ticker=selected_ticker
        )

        # ---------------------------------------------------------
        # Executive Summary Text Block
        # ---------------------------------------------------------
        st.markdown(f"""
        #### 📌 Quadrant Overview: **{quadrant_label} – {quadrant_name}**

        _Interpretation_: {quadrant_desc}

        ---
        **Peer Count:** {len(filtered_quadrant_peers)} comparable stocks  
        """)

        # ---------------------------------------------------------
        # Plot Section
        # ---------------------------------------------------------
        if not filtered_quadrant_peers.empty:

            fig = create_quadrant_comparison_plot(
                filtered_pca_df,
                selected_ticker,
                filtered_quadrant_peers
            )

            st.plotly_chart(fig, use_container_width=True)

            # ---------------------------------------------------------
            # Peer Table
            # ---------------------------------------------------------
            with st.expander("📋 View Peer Table (Detailed PCA Values)"):

                display_cols = ['ticker', 'permno', 'PC1', 'PC2', 'cluster']
                display_cols = [c for c in display_cols if c in filtered_quadrant_peers.columns]

                st.dataframe(
                   filtered_quadrant_peers[display_cols]
                   .sort_values(by="PC1", ascending=False)
                   .round(3),
                   use_container_width=True
                )

        else:
            st.info("No other stocks found in this quadrant.")
    
    elif active_visual == "🌐 3D Quadrant Peers":
        st.markdown("### 🌐 3D Quadrant Peer Comparison")
        st.markdown(f"""
        Explore quadrant peers in 3D space. The Z-axis (PC3) reveals the 
        {PC3_INTERPRETATION['name']} dimension within your peer group.
        """)
        pc1 = pca_row.get('PC1', 0)
        pc2 = pca_row.get('PC2', 0)
        filtered_quadrant_peers = get_stocks_in_same_quadrant(
            filtered_pca_df, pc1, pc2, exclude_ticker=selected_ticker
        )
        
        if not filtered_quadrant_peers.empty and 'PC3' in filtered_pca_df.columns:
            # Get PC1 + PC2 + PC3 variance
            pc3_variance = ""
            combined_variance = ""
            if st.session_state.pca_model is not None:
                variance_ratios = st.session_state.pca_model.explained_variance_ratio_
                if len(variance_ratios) >= 3:
                    pc3_var = round(variance_ratios[2] * 100, 1)
                    pc1_var = round(variance_ratios[0] * 100, 1)
                    pc2_var = round(variance_ratios[1] * 100, 1)
                    combined = round(pc1_var + pc2_var + pc3_var, 1)
                    pc3_variance = f" &nbsp;|&nbsp; <i>Explains ~{pc3_var}% of variance</i>"
                    combined_variance = f" &nbsp;|&nbsp; <i>Combined PC1+PC2+PC3: ~{combined}%</i>"
            
            # PC3 info box
            pc3_high = ' · '.join(PC3_INTERPRETATION['high_meaning'])
            pc3_low  = ' · '.join(PC3_INTERPRETATION['low_meaning'])
            st.markdown(f"""
                <div style="
                    background-color: var(--secondary-background-color);
                    color: var(--text-color);
                    padding: 0.75rem 1rem;
                    border-radius: 8px;
                    border-left: 3px solid #1f77b4;
                    font-size: 1.0rem;
                    line-height: 2;
                    white-space: nowrap;
                ">
                    <b>📐 PC3: {PC3_INTERPRETATION['name']}:</b>{pc3_variance}{combined_variance}<br>
                    ↑ <b>High PC3 - </b><i>{pc3_high}</i> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                    ↓ <b>Low PC3 - </b><i>{pc3_low}</i>
                </div>
                """, unsafe_allow_html=True)            
            from visualizations import create_3d_quadrant_peers_plot
            fig_3d_peers = create_3d_quadrant_peers_plot(
                filtered_quadrant_peers, selected_ticker, pca_row
            )
            st.plotly_chart(fig_3d_peers, use_container_width=True)
            
            # Peer table
            with st.expander("📋 View Peer Table"):
                display_cols = ['ticker', 'permno', 'PC1', 'PC2', 'PC3', 'cluster']
                display_cols = [c for c in display_cols if c in filtered_quadrant_peers.columns]
                st.dataframe(filtered_quadrant_peers[display_cols].round(3))
        else:
            st.info("No peers found or PC3 data unavailable.")

    elif active_visual == "📊 Factor Analysis":
        st.markdown("### 📊 Factor Breakdown Analysis")
          
        # Recalculate percentiles from filtered quadrant peers
        pc1 = pca_row.get('PC1', 0)
        pc2 = pca_row.get('PC2', 0)
        filtered_quadrant_peers = get_stocks_in_same_quadrant(
            filtered_pca_df, pc1, pc2, exclude_ticker=selected_ticker
        )
        
        # Calculate percentiles BEFORE using them in charts
        available_features = [c for c in FEATURE_COLUMNS if c in pca_row.index]
        percentiles = compute_percentile_ranks(filtered_quadrant_peers, pca_row, available_features)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Radar chart with percentiles
            factor_data = get_factor_breakdown(pca_row)
            fig_radar = create_factor_radar_chart(factor_data, selected_ticker, percentiles)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            # Percentile rankings chart
            if percentiles:
                fig_percentile = create_percentile_chart(percentiles, selected_ticker)
                st.plotly_chart(fig_percentile, use_container_width=True)

                st.caption("(V)=Value   (Q)=Quality   (FS)=Financial Strength   (R)=Risk")
        
        # Factor details table
        with st.expander("📋 Detailed Factor Values"):
            factor_table = []
            for category, features in FACTOR_CATEGORIES.items():
                for feature in features:
                    if feature in pca_row.index:
                        pct = percentiles.get(feature, 'N/A')
                        factor_table.append({
                            'Category': category,
                            'Factor': feature,
                            'Value': pca_row[feature],
                            'Percentile': f"{pct:.1f}%" if isinstance(pct, (int, float)) else pct
                        })
            
            if factor_table:
                st.dataframe(pd.DataFrame(factor_table))
        
        # Factor Trend Charts
        st.markdown("---")
        st.markdown("### 📈 Factor Trends Over Time")
        st.markdown("""
        Visualize how the top PC1 and PC2 drivers have changed over time, 
        helping explain movement on the 2D cluster plot.
        """)
        
        # Time period selector
        col1, col2 = st.columns([1, 3])
        with col1:
            time_period = st.selectbox(
                "Time Period:",
                options=["All", "5Y", "3Y", "1Y"],
                index=0,
                key="factor_trend_period"
            )
        
        # Generate charts if loadings available
        if 'pca_loadings' in st.session_state:
            fig_pc1_trend, fig_pc2_trend = create_factor_trend_chart(
                raw_data, 
                selected_ticker, 
                st.session_state.pca_loadings,
                period=time_period
            )
            
            # Display charts side by side
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_pc1_trend, use_container_width=True)
            with col2:
                st.plotly_chart(fig_pc2_trend, use_container_width=True)
            
            # Explanation
            st.info("💡 **Interpretation:** Rising lines indicate improving metrics. "
                   "PC1 drivers (left) explain horizontal movement. PC2 drivers (right) explain vertical movement.")
        else:
            st.warning("PCA loadings not available.")
    
    elif active_visual == "🕐 2D or 3D Time-Lapse":
        st.markdown("### 🕐 Historical Movement Animation")
        st.markdown("""
        Watch how the stock's position has changed over time in the PCA space.
        Click **Play** to start the animation.
        """)
        
        # Add 2D/3D toggle
        col1, col2 = st.columns([1, 3])
        with col1:
            view_mode = st.radio(
                "View Mode:",
                options=["2D View", "3D View"],
                index=0,
                key="timelapse_view_mode",
                horizontal=True
            )
        
        if st.button("🔄 Generate Time-Lapse Animation", key="timelapse_btn"):
            with st.spinner("Preparing animation..."):
                time_series_data = prepare_time_series_data(
                    raw_data, selected_ticker, pca_model, scaler
                )
                
                if not time_series_data.empty:
                    # Calculate time range
                    start_date = pd.to_datetime(time_series_data['date'].iloc[0])
                    end_date = pd.to_datetime(time_series_data['date'].iloc[-1])
                    num_months = round((end_date - start_date).days / 30.44)
                    num_datapoints = len(time_series_data)
                    
                    st.info(f"📅 **Time Range:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} "
                           f"({num_months} months, {num_datapoints} data points)")
                    
                    if view_mode == "3D View":
                        # Check if PC3 data is available
                        if 'PC3' in time_series_data.columns:
                            from visualizations import create_timelapse_animation_3d
                            fig_animation = create_timelapse_animation_3d(
                                time_series_data, selected_ticker, filtered_pca_df
                            )
                            st.plotly_chart(fig_animation, use_container_width=True)
                        else:
                            st.warning("3D view requires PC3 data. Falling back to 2D view.")
                            fig_animation = create_timelapse_animation(
                                time_series_data, selected_ticker, filtered_pca_df
                            )
                            st.plotly_chart(fig_animation, use_container_width=True)
                    else:
                        # 2D View
                        fig_animation = create_timelapse_animation(
                            time_series_data, selected_ticker, filtered_pca_df
                        )
                        st.plotly_chart(fig_animation, use_container_width=True)
                else:
                    st.warning("Insufficient time-series data for animation.")
    
    elif active_visual == "🌐 3D Cluster View":
        st.markdown("### 🌐 3D PCA Visualization")
        st.markdown("""
        Explore the clusters in 3D space using the first three principal components.
        Drag to rotate, scroll to zoom.
        """)
        
        if 'PC3' in filtered_pca_df.columns:
            # Get PC1 + PC2 + PC3 variance
            pc3_variance = ""
            combined_variance = ""
            if st.session_state.pca_model is not None:
                variance_ratios = st.session_state.pca_model.explained_variance_ratio_
                if len(variance_ratios) >= 3:
                    pc3_var = round(variance_ratios[2] * 100, 1)
                    pc1_var = round(variance_ratios[0] * 100, 1)
                    pc2_var = round(variance_ratios[1] * 100, 1)
                    combined = round(pc1_var + pc2_var + pc3_var, 1)
                    pc3_variance = f" &nbsp;|&nbsp; <i>Explains ~{pc3_var}% of variance</i>"
                    combined_variance = f" &nbsp;|&nbsp; <b>Combined PC1+PC2+PC3: ~{combined}%</b>"
            
            # PC3 info box ABOVE chart
            pc3_high = ' · '.join(PC3_INTERPRETATION['high_meaning'])
            pc3_low  = ' · '.join(PC3_INTERPRETATION['low_meaning'])
            st.markdown(f"""
                <div style="
                    background-color: var(--secondary-background-color);
                    color: var(--text-color);
                    padding: 0.75rem 1rem;
                    border-radius: 8px;
                    border-left: 3px solid #1f77b4;
                    font-size: 1.0rem;
                    line-height: 2;
                    white-space: nowrap;
                ">
                    <b>📐 PC3: {PC3_INTERPRETATION['name']}:</b>{pc3_variance}{combined_variance}<br>
                    ↑ <b>High PC3 - </b><i>{pc3_high}</i> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                    ↓ <b>Low PC3 - </b><i>{pc3_low}</i>
                </div>
                """, unsafe_allow_html=True)
            
            # Chart full width
            fig_3d = create_3d_pca_plot(filtered_pca_df, selected_ticker)
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("3D visualization requires PC3 data.")


def render_full_universe_loadings_table():
    """
    Render Appendix B.1-style full-universe PCA loadings table
    from the main PCA model (time-averaged cross-section).
    """
    st.markdown("### 4 · Full-Universe PC Loadings")
    st.caption(
        "Main PCA — time-averaged cross-section of the full universe. "
        "This is the operative coordinate system used for stock positioning, "
        "quadrant assignment, cluster mapping, and peer comparison."
    )

    pca_model = st.session_state.get("pca_model")
    if pca_model is None:
        st.warning("Main PCA model is not available.")
        return

    if not hasattr(pca_model, "components_"):
        st.warning("PCA components are not available on the current model.")
        return

    if len(FEATURE_COLUMNS) == 0:
        st.warning("No feature columns were found.")
        return

    # Build raw loading matrix: rows = factors, cols = PCs
    n_components = min(3, pca_model.components_.shape[0])
    component_names = [f"PC{i+1}" for i in range(n_components)]

    loadings_df = pd.DataFrame(
        pca_model.components_[:n_components].T,
        index=FEATURE_COLUMNS,
        columns=component_names
    ).reset_index().rename(columns={"index": "Code"})

    # Factor display names
    loadings_df["Factor"] = loadings_df["Code"].map(FEATURE_DISPLAY_NAMES).fillna(loadings_df["Code"])

    # Domain mapping from FACTOR_CATEGORIES
    code_to_domain = {}
    for domain, codes in FACTOR_CATEGORIES.items():
        for code in codes:
            code_to_domain[code] = domain

    loadings_df["Domain"] = loadings_df["Code"].map(code_to_domain).fillna("Other")

    # Clean column order
    ordered_cols = ["Factor", "Code", "Domain"] + component_names
    loadings_df = loadings_df[ordered_cols]

    # Variance explained
    variance_text = ""
    if hasattr(pca_model, "explained_variance_ratio_"):
        ratios = pca_model.explained_variance_ratio_
        pieces = []
        total = 0.0
        for i in range(min(3, len(ratios))):
            pct = round(ratios[i] * 100, 1)
            total += pct
            pieces.append(f"PC{i+1} ≈ {pct}%")
        variance_text = " · ".join(pieces) + f" · Combined ≈ {round(total, 1)}%"

    if variance_text:
        st.markdown(f"**Variance explained:** {variance_text}")

        st.caption(
            "Note: PCA component signs may differ from Appendix B due to sign indeterminacy, "
            "but structural interpretations are identical."
    )

    def _style_loading(val):
        if not isinstance(val, (int, float)):
            return ""
        if abs(val) < 0.08:
            return ""
        if val >= 0.30:
            return "background-color: rgba(84,162,75,0.75); color: white; font-weight: bold;"
        if val > 0:
            return "background-color: rgba(84,162,75,0.30);"
        if val <= -0.30:
            return "background-color: rgba(228,87,86,0.75); color: white; font-weight: bold;"
        return "background-color: rgba(228,87,86,0.30);"

    numeric_cols = [c for c in ["PC1", "PC2", "PC3"] if c in loadings_df.columns]

    styled = (
        loadings_df.style
        .map(_style_loading, subset=numeric_cols)
        .format({col: "{:+.2f}" for col in numeric_cols})
        .set_properties(subset=["Factor", "Code", "Domain"], **{"text-align": "left"})
        .set_properties(subset=numeric_cols, **{"text-align": "center"})
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#1e1e2e"),
                    ("color", "#ffffff"),
                    ("font-size", "12px"),
                    ("text-align", "center"),
                    ("padding", "6px 10px"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("font-size", "12px"),
                    ("padding", "4px 10px"),
                ],
            },
            {
                "selector": "tr:hover td",
                "props": [("background-color", "rgba(255,255,255,0.05)")],
            },
        ])
    )

    st.dataframe(styled, width="stretch")

    with st.expander("📖 How to read this table"):
        st.markdown("""
- **PC1 / PC2 / PC3** are the principal component loadings from the **main PCA**.
- **Positive loading** means the factor pushes stocks higher on that axis.
- **Negative loading** means the factor pushes stocks lower on that axis.
- **Higher absolute values** indicate stronger structural contribution.
- Cells with **|loading| < 0.08** are left unshaded to match your appendix convention.
        """)


def render_narrative_section(
    ticker: str,
    pca_row: pd.Series,
    peer_df: pd.DataFrame,
    gics_sector: str = 'N/A',
):
    """Render the AI Narrative Analysis section (no API key required)."""

    st.markdown("---")
    st.header("📊 AI Narrative Analysis")
    st.caption("Auto-generated plain-English interpretation — no API key required.")

    percentiles  = st.session_state.get('current_percentiles', {})
    factor_data  = st.session_state.get('current_factor_data', {})
    raw_data     = st.session_state.get('raw_data', None)
    loadings     = st.session_state.get('pca_loadings', None)

    # --- STEP 3B: Remove UI dependency from PC3 logic ---
    show_pc3 = False
    try:
        if pca_row is not None and "PC3" in pca_row:
            show_pc3 = True
    except Exception:
        show_pc3 = False

    with st.spinner("Generating narrative analysis..."):
        kg                  = st.session_state.get("kg_instance")
        kg_regime           = st.session_state.get("kg_current_regime")
        structural_drivers  = st.session_state.get("current_structural_drivers")

        # Fallback: compute drivers if not yet available in session
        if not structural_drivers and loadings is not None and pca_row is not None:
            try:
                loadings_df = pd.DataFrame(loadings)

                pc1 = pca_row.get("PC1", 0)
                pc2 = pca_row.get("PC2", 0)

                drivers = []

                for factor in loadings_df.index:
                    pc1_loading = loadings_df.at[factor, "PC1"]
                    pc2_loading = loadings_df.at[factor, "PC2"]

                    directional_score = (pc1_loading * pc1) + (pc2_loading * pc2)

                    direction = "Positive" if directional_score >= 0 else "Negative"

                    abs_score = abs(directional_score)
                    if abs_score >= 0.30:
                        strength = "Strong"
                    elif abs_score >= 0.15:
                        strength = "Moderate"
                    else:
                        strength = "Light"

                    display_name = FEATURE_DISPLAY_NAMES.get(factor, factor)

                    drivers.append({
                        "factor_name": display_name,
                        "direction": direction,
                        "strength": strength,
                        "score": directional_score,
                    })

                # sort + keep top 3
                drivers = sorted(drivers, key=lambda x: abs(x["score"]), reverse=True)[:3]

                structural_drivers = drivers
                st.session_state.current_structural_drivers = drivers

            except Exception:
                structural_drivers = []
        # --- FIX: Align AI peer group with filtered universe ---
        peer_df = peer_df.copy()
        # peer_df already contains quadrant-filtered peers

        sections = generate_narrative(  # ARCH: intentional NE boundary — sole authorized call site
            ticker              = ticker,
            pca_row             = pca_row,
            percentiles         = percentiles,
            factor_data         = factor_data,
            peer_df             = peer_df,
            raw_data            = raw_data,
            loadings            = loadings,
            gics_sector         = gics_sector,
            show_pc3            = show_pc3,
            structural_drivers  = structural_drivers,
            kg                  = kg,
            current_regime      = kg_regime,
        )

    # --- STEP 3A: Narrative decoupled from visualization state ---

    # Always render full deterministic narrative stack
    if sections.get('summary'):
        st.markdown(sections['summary'])

    if sections.get('factors'):
        factors_section = sections['factors']
        if isinstance(factors_section, dict):
            st.markdown(factors_section.get('text', ''))
        else:
            st.markdown(factors_section)

    if sections.get('trajectory'):
        st.markdown(sections['trajectory'])

    if sections.get('peers'):
        st.markdown(sections['peers'])

    if sections.get('factors'):
        st.markdown("---")
        st.subheader("🧾 Deterministic Stock Narrative (Tier 1)")

        factors_section = sections['factors']

        if isinstance(factors_section, dict):
            st.markdown(factors_section.get('text', ''))

            # --- NEW: Persist KG references for downstream use ---
            raw_refs = factors_section.get('kg_references', [])

            # --- NEW: Normalize KG references into canonical node IDs ---
            normalized_refs = []
            try:
                import re
                for ref in raw_refs:
                    match = re.match(r"\[\[KG:(.*?):(.*?)\]\]", ref)
                    if match:
                        node_type, node_id = match.groups()
                        normalized_refs.append(f"{node_type}:{node_id}")
            except Exception:
                normalized_refs = []

            # --- Inject structural anchor nodes ---
            anchors = []

            try:
                if ticker:
                    anchors.append(f"stock:{ticker}")

                if kg_regime:
                    anchors.append(f"regime:{kg_regime}")

                # --- NEW: include current quadrant ---
                quadrant = st.session_state.get("current_quadrant")
                if quadrant:
                    anchors.append(f"quadrant:{quadrant}")

                # --- NEW: include current cluster ---
                cluster = pca_row.get("cluster")
                if cluster is not None and str(cluster) != "":
                    anchors.append(f"cluster:{cluster}")

                # --- NEW: include structural driver factors ---
                drivers = st.session_state.get("current_structural_drivers", [])
                for d in drivers[:3]:
                    factor_code = d.get("factor")

                    # Ensure canonical KG factor ID (lowercase, underscore-safe)
                    if factor_code:
                        anchors.append(f"factor:{str(factor_code).strip().lower()}")

            except Exception:
                pass

            # Combine narrative refs + anchors
            all_refs = list(dict.fromkeys(normalized_refs + anchors))

            st.session_state['last_narrative_kg_refs'] = all_refs

            # --- NEW: Build KG subgraph from narrative references (WITH EXPANSION) ---
            try:
                if kg is not None and all_refs:
                    subgraph = kg.serialize_subgraph(all_refs)
                    st.session_state['last_narrative_subgraph'] = subgraph

                    try:
                        meta = subgraph.get("meta", {}) if isinstance(subgraph, dict) else {}
                        node_count = meta.get("node_count", 0)

                        st.session_state["last_narrative_subgraph_debug"] = {
                            "requested_refs": list(all_refs),
                            "normalized_refs": list(normalized_refs),
                            "anchors": list(anchors),
                            "meta": meta,
                            "subgraph": subgraph,
                        }

                        if node_count == 0:
                            st.warning("Narrative KG subgraph returned 0 nodes.")
                            with st.expander("🔧 Debug: Narrative Subgraph Build", expanded=False):
                                st.markdown("**Requested Refs**")
                                for ref in all_refs:
                                    st.markdown(f"- `{ref}`")

                                missing_nodes = meta.get("missing_nodes", [])
                                if missing_nodes:
                                    st.markdown("**Missing Nodes**")
                                    for ref in missing_nodes:
                                        st.markdown(f"- `{ref}`")
                    except Exception:
                        pass
                else:
                    st.session_state['last_narrative_subgraph'] = None
            except Exception:
                st.session_state['last_narrative_subgraph'] = None

    if sections.get('structural') and kg is not None:
        with st.expander("🧠 Structural Context (Knowledge Graph — Tier 1)", expanded=False):
            st.markdown(sections['structural'])


def render_chatbot_section(
    ticker: str,
    permno: str,
    cluster: int,
    quadrant: str,
    pc1: float,
    pc2: float,
    pca_row: pd.Series,
    percentiles: dict,
    peer_count: int,
    cluster_summary: pd.DataFrame,
    total_universe=None
):
    """Render the AI chatbot section."""
    
    st.markdown("---")
    st.markdown("## 🤖 AI Analysis Assistant")
    
    chatbot = st.session_state.chatbot
    
    if chatbot is None or not chatbot.is_available():
        st.warning("""
        ⚠️ **Chatbot not configured.** 
        
        To enable the AI assistant:
        1. Enter your OpenAI API key in the sidebar
        2. The chatbot will be activated automatically
        
        You can get an API key from [OpenAI's website](https://platform.openai.com/api-keys).
        """)
        return
    
    # Update chatbot context
    factor_data = get_factor_breakdown(pca_row)
    kg = st.session_state.get("kg_instance")
    kg_regime = st.session_state.get("kg_current_regime")
    nearest_peers = st.session_state.get("nearest_peers")
    nearest_peer_tickers = (
        nearest_peers["ticker"].dropna().astype(str).tolist()
        if nearest_peers is not None and "ticker" in nearest_peers.columns
        else []
    )
    kg_subgraph = None
    if kg is not None and kg_regime is not None:
        try:
            kg_subgraph = kg.serialize_subgraph([
                f"regime:{kg_regime}",
                f"stock:{ticker}",
                f"quadrant:{quadrant}",
                f"cluster:{cluster}",
            ])

            # DEBUG
            st.write("KG Subgraph:", kg_subgraph)

            # STORE FOR STRUCTURAL ANALYST
            st.session_state["stock_subgraph"] = kg_subgraph

        except Exception:
            kg_subgraph = None
    chatbot.set_stock_context(
        ticker=ticker,
        permno=permno,
        cluster=cluster,
        quadrant=quadrant,
        pc1=pc1,
        pc2=pc2,
        factor_data=factor_data,
        percentiles=percentiles,
        peer_count=peer_count,
        cluster_summary=cluster_summary,
        kg_subgraph=kg_subgraph,
        total_universe=total_universe,
    )
    
    # Quick analysis button
    if st.button("📝 Get Quick Analysis", key="quick_analysis_btn"):
        analysis = chatbot.get_quick_analysis()
        st.markdown(analysis)
    
    st.markdown("---")
    
    # Sample questions
    st.markdown("### Sample Questions")
    cols = st.columns(4)
    for i, question in enumerate(SAMPLE_QUESTIONS[:4]):
        with cols[i]:
            if st.button(question[:30] + "...", key=f"sample_q_{i}"):
                response = chatbot.get_response(question)
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': question
                })
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
    
    # Chat interface
    st.markdown("### Ask a Question")
    
    user_input = st.text_input(
        "Your question:",
        placeholder="e.g., How does this stock compare to its peers?",
        key="chat_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("Send", key="send_btn", type="primary")
    with col2:
        clear_button = st.button("Clear History", key="clear_btn")
    
    if send_button and user_input:
        with st.spinner("Thinking..."):
            response = chatbot.get_response(user_input)
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
    
    if clear_button:
        st.session_state.chat_history = []
        chatbot.clear_history()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Conversation")
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)


def render_universe_cluster_overview():
    """Render the Universe / Portfolio Level cluster overview tab."""

    st.markdown("### System Structure Dashboard")
    st.caption(
        "Monitor cross-sectional market structure, cluster concentration, and sector-level positioning "
        "through the primary PCA diagnostic view."
    )

    st.info(
        "Use this dashboard to assess how broadly dispersed the current universe is, "
        "where structural crowding may be emerging, and how sector filtering changes the observed market shape."
    )

    # Apply GICS sector filter if selected on landing page
    plot_df = st.session_state.pca_df.copy()
    selected_sector = st.session_state.get('selected_gics_sector', 'All Sectors')

    if selected_sector and selected_sector != "All Sectors" and 'gicdesc' in st.session_state.raw_data.columns:
        sector_tickers = st.session_state.raw_data[
            st.session_state.raw_data['gicdesc'] == selected_sector
        ]['ticker'].unique()
        plot_df = plot_df[plot_df['ticker'].isin(sector_tickers)]

    total_count = len(plot_df)
    cluster_count = plot_df['cluster'].nunique() if 'cluster' in plot_df.columns and not plot_df.empty else 0
    sector_scope = selected_sector if selected_sector else "All Sectors"

    dominant_cluster_share = 0.0
    if 'cluster' in plot_df.columns and not plot_df.empty:
        dominant_cluster_share = (
            plot_df['cluster'].value_counts(normalize=True).iloc[0] * 100
        )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Stocks in View", f"{total_count}")
    metric_col2.metric("Clusters Represented", f"{cluster_count}")
    metric_col3.metric("Sector Scope", sector_scope)
    metric_col4.metric("Largest Cluster Share", f"{dominant_cluster_share:.1f}%")

    if selected_sector and selected_sector != "All Sectors":
        sector_label = f" — {selected_sector} ({total_count})"
    else:
        sector_label = f" — All Sectors ({total_count})"

    st.markdown("---")
    st.markdown(f"#### PCA Cluster Map{sector_label}")
    st.caption(
        "Primary cross-sectional view of market structure, showing relative position and clustering across the current universe."
    )

    fig = create_pca_scatter_plot(plot_df)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Cluster Concentration Summary")
    st.caption(
        "Distribution of names across structural clusters in the current universe view."
    )

    cluster_summary = get_cluster_summary(plot_df)
    fig_summary = create_cluster_summary_plot(cluster_summary)
    st.plotly_chart(fig_summary, use_container_width=True)


def render_universe_period_comparison():
    """Render the Universe / Portfolio Level period comparison tab."""

    st.subheader("🧭 Regime Intelligence")
    st.caption(
        "Detect structural regime shifts across Post-COVID, Rate Shock, and "
        "Disinflation environments."
    )

    st.info(
        "This module reframes period comparison into a regime intelligence view — "
        "highlighting structural breaks, crowding buildup, and stock repositioning "
        "across changing market environments."
    )

    raw_df = st.session_state.raw_data
    if raw_df is None:
        st.info("Data not yet loaded.")
    else:
        date_col = next(
            (c for c in ['date', 'DATE', 'Date', 'period', 'PERIOD', 'datadate', 'yyyymm', 'yearmonth', 'public_date'] if c in raw_df.columns),
            None
        )
        if date_col is None:
            st.error(f"Date column not found. Available columns: {list(raw_df.columns)}")
        else:
            raw_df[date_col] = pd.to_datetime(raw_df[date_col])
            features = get_features_from_df(raw_df)

            if len(features) < 3:
                    st.error(f"Too few feature columns detected: {features}")
            else:
                    # Section 1: Procrustes
                    st.markdown("---")
                    st.markdown("### 1 · Procrustes Disparity Scores")
                    st.caption("0 = identical structure; >0.15 = meaningful regime change.")

                    with st.spinner("Computing Procrustes analysis…"):
                        proc_df  = compute_procrustes_table(raw_df, features, date_col)
                        fig_proc = create_procrustes_heatmap(proc_df)

                    col_heat, col_table = st.columns([1.2, 1])
                    with col_heat:
                        st.plotly_chart(fig_proc, use_container_width=True)
                    with col_table:
                        st.markdown("**Pairwise results**")
                        st.dataframe(proc_df, use_container_width=True, hide_index=True)

                    with st.expander("📖 How to read Procrustes"):
                        st.markdown("""
                        - **< 0.05** 🟢 Fa\tor structure nearly identical across periods.
                        - **0.05–0.15** 🟡 Moderate shift; broadly similar structure.
                        - **0.15–0.30** 🟠 Meaningful regime change; factors rewiring.
                        - **> 0.30** 🔴 Major structural break; treat as distinct regimes.
                        - Procrustes is **rotation-invariant** — accounts for PCA sign flips and axis swaps.
                        """)

                    # ============================================================
                    # CROWDING SCORE MODULE
                    # ============================================================
                    st.markdown("---")
                    st.markdown("### 2 · Factor Crowding Score")
                    st.caption(
                        "Measures how structurally concentrated the equity universe has become "
                        "in PCA space each regime. A rising score signals factor crowding building "
                        "before it becomes a realized risk event."
                    )

                    # Build a combined pca_df across all three periods
                    from period_analysis import _run_pca_for_period
                    all_period_rows = []
                    period_label_map = {
                        'Post-COVID':   ('2021-03-01', '2022-06-30'),
                        'Rate Shock':   ('2022-07-01', '2023-09-30'),
                        'Disinflation': ('2023-10-01', '2024-10-31'),
                    }
                    for period_name, (start, end) in period_label_map.items():
                        period_mask = (raw_df[date_col] >= start) & (raw_df[date_col] <= end)
                        period_slice = raw_df[period_mask]
                        if len(period_slice) < 10:
                            continue
                        try:
                            _, scores_df, _, _ = _run_pca_for_period(
                                period_slice, features, date_col, start, end
                            )
                            if scores_df is None:
                                continue
                            # Store clean per-period scores for Knowledge Graph
                            # (before period/cluster columns are added for crowding)
                            if "period_scores" not in st.session_state:
                                st.session_state["period_scores"] = {}
                            st.session_state["period_scores"][period_name] = scores_df.copy()

                            scores_df['period'] = period_name
                            # compute_crowding_scores needs 'cluster' column
                            # derive it from Quadrant label
                            quad_map = {
                                f"Q1: {QUADRANTS['Q1']['name']}": 0,
                                f"Q2: {QUADRANTS['Q2']['name']}": 1,
                                f"Q3: {QUADRANTS['Q3']['name']}": 2,
                                f"Q4: {QUADRANTS['Q4']['name']}": 3,
                            }
                            scores_df['cluster'] = scores_df['Quadrant'].map(quad_map)
                            all_period_rows.append(scores_df)
                        except Exception:
                            continue

                    if all_period_rows:
                        combined_period_df = pd.concat(all_period_rows, ignore_index=True)
                        crowding_df = compute_crowding_scores(combined_period_df)
                        st.session_state["crowding_df"] = crowding_df
                        st.session_state["crowding_results"] = crowding_df

                        # FORCE rebuild KG after period_scores is created (critical for Tier 2)
                        from kg_builder import build_kg
                        from kg_interface import KnowledgeGraph

                        kg_result = build_kg(
                            period_data=st.session_state.get("period_scores"),
                            migration_df=st.session_state.get("migration_wide"),
                            include_equity_nodes=True,
                        )

                        st.session_state["kg_instance"] = KnowledgeGraph(kg_result.graph)
                        st.session_state["kg_current_regime"] = "Disinflation"

                        if not crowding_df.empty:
                            # Metric cards — one per regime
                            metric_cols = st.columns(len(crowding_df))
                            for i, row in crowding_df.iterrows():
                                with metric_cols[i]:
                                    delta_str = ""
                                    if i > 0:
                                        prev_score = crowding_df.iloc[i - 1]['crowding_score']
                                        delta = row['crowding_score'] - prev_score
                                        delta_str = f"{delta:+.1f} vs prior regime"
                                    st.metric(
                                        label=f"{row['period']}",
                                        value=f"{row['crowding_score']:.0f} / 100",
                                        delta=delta_str,
                                        delta_color="inverse"
                                    )
                                    st.caption(row['risk_level'])

                            # Chart
                            crowding_fig = plot_crowding_score(crowding_df)
                            if crowding_fig:
                                st.plotly_chart(crowding_fig, use_container_width=True)

                            # Detail table
                            with st.expander("📋 Crowding Score Detail Table"):
                                display_df = crowding_df.rename(columns={
                                    'period':               'Regime',
                                    'n_stocks':             'Stocks',
                                    'largest_cluster_pct':  '% in Largest Cluster',
                                    'centroid_dispersion':  'Centroid Dispersion',
                                    'crowding_score':       'Crowding Score',
                                    'risk_level':           'Risk Level'
                                })
                                st.dataframe(display_df, use_container_width=True, hide_index=True)

                            # Plain-English narrative
                            latest   = crowding_df.iloc[-1]
                            earliest = crowding_df.iloc[0]
                            trend    = "increased" if latest['crowding_score'] > earliest['crowding_score'] else "decreased"
                            st.info(
                                f"**Crowding Trend:** Factor crowding has **{trend}** from "
                                f"{earliest['crowding_score']:.0f} in the {earliest['period']} regime "
                                f"to {latest['crowding_score']:.0f} in the {latest['period']} regime. "
                                f"The current {latest['risk_level']} reading reflects that "
                                f"{latest['largest_cluster_pct']:.0f}% of the universe is concentrated "
                                f"in the single largest cluster — a portfolio overweight this cluster "
                                f"is running a structural factor bet that may not be visible in "
                                f"traditional correlation-based risk systems."
                            )
                        else:
                            st.warning("Crowding score could not be computed — check cluster labels.")
                    else:
                        st.warning("Not enough period data to compute crowding scores.")

                    # Section 3: Quadrant Migration
                    st.markdown("---")
                    st.markdown("### 3 · Quadrant Migration")
                    st.caption("Tracks where each stock sat in PCA space across the three regimes.")

                    with st.spinner("Computing quadrant assignments…"):
                        period_names = [k.split('\n')[0] for k in PERIOD_KEYS]
                        migration_df, summary_df, migration_pct = compute_quadrant_migration(
                            raw_df, features, date_col
                        )
                        if summary_df is not None and not summary_df.empty:
                            st.session_state["migration_summary_df"] = summary_df

                    if migration_df is not None and not migration_df.empty:
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Stocks Tracked", f"{len(migration_df):,}")
                        col_b.metric("Changed Quadrant", f"{migration_pct:.1f}%")
                        col_c.metric("Stayed Same", f"{100 - migration_pct:.1f}%")

                        st.markdown("**Migration rates between adjacent periods:**")
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)

                        st.markdown("**Flow diagram:**")
                        fig_sankey = create_migration_sankey(migration_df, period_names)
                        st.plotly_chart(fig_sankey, use_container_width=True)

                        with st.expander("🔍 View stock-level quadrant table"):
                            st.dataframe(
                                migration_df.sort_values('Any Change', ascending=False),
                                use_container_width=True,
                                hide_index=True
                            )
                    else:
                        st.warning("Not enough common tickers across all three sub-periods to compute migration.")

                    # Section 4: Full-Universe PC Loadings (Main PCA)
                    st.markdown("---")
                    render_full_universe_loadings_table()
                    
                    # Section 5: Factor Loadings by Sub-Period
                    st.markdown("---")
                    st.markdown("### 5 · Factor Loadings by Sub-Period")
                    st.caption("If bars change size or flip sign across periods, the factor structure shifted.")

                    pc_choice = st.radio(
                        "Select principal component:",
                        options=["PC1", "PC2", "PC3"],
                        horizontal=True,
                        key="landing_period_pc_choice"
                    )

                    with st.spinner("Running PCA for each sub-period…"):
                        fig_loadings = create_loading_comparison_chart(raw_df, features, date_col, pc=pc_choice)
                    st.plotly_chart(fig_loadings, use_container_width=True)

                    from period_analysis import get_loading_comparison_data
                    loadings_table = get_loading_comparison_data(raw_df, features, date_col, pc=pc_choice)
                    if not loadings_table.empty:
                        with st.expander(f"🔍 View {pc_choice} factor loadings table"):
                            period_cols = ['Post-COVID', 'Rate Shock', 'Disinflation']
                            delta_cols  = [c for c in loadings_table.columns if c.startswith('Δ')]

                            def _style_loading(val):
                                if not isinstance(val, (int, float)):
                                    return ''
                                if val >= 0.30:
                                    return 'background-color: rgba(84,162,75,0.75); color: white; font-weight:bold;'
                                elif val >= 0.10:
                                    return 'background-color: rgba(84,162,75,0.35); color: white;'
                                elif val <= -0.30:
                                    return 'background-color: rgba(228,87,86,0.75); color: white; font-weight:bold;'
                                elif val <= -0.10:
                                    return 'background-color: rgba(228,87,86,0.35); color: white;'
                                return ''

                            def _style_delta(val):
                                if not isinstance(val, (int, float)):
                                    return ''
                                if val > 0.05:
                                    return 'color: #54A24B; font-weight: bold;'
                                elif val < -0.05:
                                    return 'color: #E45756; font-weight: bold;'
                                return 'color: #888888;'

                            styled = (
                                loadings_table.style
                                .map(_style_loading, subset=period_cols)
                                .map(_style_delta,   subset=delta_cols)
                                .format('{:+.3f}', subset=delta_cols)
                                .format('{:.3f}',  subset=period_cols)
                                .set_properties(**{'text-align': 'center'})
                                .set_table_styles([
                                    {'selector': 'th',
                                    'props': [('background-color', '#1e1e2e'),
                                            ('color', '#ffffff'),
                                            ('font-size', '12px'),
                                            ('text-align', 'center'),
                                            ('padding', '6px 10px')]},
                                    {'selector': 'td',
                                    'props': [('font-size', '12px'),
                                            ('padding', '4px 10px')]},
                                    {'selector': 'tr:hover td',
                                    'props': [('background-color', 'rgba(255,255,255,0.05)')]},
                                ])
                            )
                            st.dataframe(styled, use_container_width=True)
                            st.caption(
                                f"Loading weights for **{pc_choice}** across the three sub-periods. "
                                "Bold shading = dominant driver (|loading| ≥ 0.30). "
                                "Δ columns show shift between adjacent regimes; "
                                "green = strengthening, red = weakening."
                            )

                    with st.expander("📖 How to read this chart"):
                        st.markdown("""
                        - **Bar height** = how strongly that feature drives this component in that period.
                        - **Positive loading** = the feature pushes stocks *higher* on this axis.
                        - **Negative loading** = the feature pushes stocks *lower* on this axis.
                        - If a bar **flips sign** between periods, the factor literally reversed its role.
                        - If a bar **shrinks toward zero**, that feature lost explanatory power in that regime.
                        """)


def render_universe_kg_tab():
    """Render the Universe / Portfolio Level knowledge graph tab."""

    render_kg_tab()


def render_universe_structural_tab():
    """Render the Universe / Portfolio Level structural intelligence tab."""

    render_structural_intelligence_tab()


def render_universe_workspace():
    """Render the Universe / Portfolio Level workspace."""
    
    st.markdown("## Universe / Portfolio Diagnostics")
    st.caption(
        "System-level structural dashboard for monitoring factor crowding, regime shifts, "
        "cross-sectional instability, and market-wide positioning dynamics."
    )
    
    # Show overall cluster summary
    if st.session_state.pca_df is not None:

        landing_tab1, landing_tab2, landing_tab3, landing_tab4 = st.tabs([
            "📊 Cluster Overview",
            "📐 Period Comparison",
            "🧠 Knowledge Graph",
            "🔬 Structural Intelligence",
        ])

        with landing_tab1:
            render_universe_cluster_overview()

        with landing_tab2:
            render_universe_period_comparison()

        with landing_tab3:
            render_universe_kg_tab()

        with landing_tab4:
            render_universe_structural_tab()


def render_stock_overview_tab(pca_row: pd.Series):
    """Render the Stock / Individual Ticker Level overview tab."""

    return render_stock_overview(
        st.session_state.raw_data,
        pca_row
    )


def render_stock_visuals_tab(
    pca_df,
    ticker,
    pca_row,
    quadrant_peers
):
    """Render the Stock / Individual Ticker Level visuals tab."""

    st.markdown("### 📊 Visuals")

    render_visualizations(
        pca_df,
        ticker,
        pca_row,
        quadrant_peers,
        st.session_state.raw_data,
        st.session_state.pca_model,
        st.session_state.scaler
    )


def render_stock_peers_tab(
    ticker: str,
    quadrant: str,
    cluster: int,
    pc1: float,
    pc2: float,
    quadrant_peers,
    pca_df,
):
    """Render the stock-level Peers & Positioning tab."""
    st.markdown("### 👥 Peers & Positioning")

    st.markdown(
        f"**{ticker}** is currently positioned in **{quadrant}** and **Cluster {cluster}**."
    )
    st.caption(
        f"PC coordinates: PC1 = {pc1:.2f}, PC2 = {pc2:.2f}"
    )

    same_quadrant_count = len(quadrant_peers) if quadrant_peers is not None else 0

    cluster_peers = pd.DataFrame()
    if "cluster" in pca_df.columns:
        cluster_peers = pca_df[
            (pca_df["cluster"] == cluster) & (pca_df["ticker"] != ticker)
        ].copy()

    same_cluster_count = len(cluster_peers)
    overlap_count = 0
    if not cluster_peers.empty and quadrant_peers is not None and not quadrant_peers.empty:
        overlap_count = len(
            set(cluster_peers["ticker"]).intersection(set(quadrant_peers["ticker"]))
        )

    pos_col1, pos_col2, pos_col3 = st.columns(3)
    with pos_col1:
        st.metric("Same Quadrant", same_quadrant_count)
    with pos_col2:
        st.metric("Same Cluster", same_cluster_count)
    with pos_col3:
        st.metric("Quadrant + Cluster Overlap", overlap_count)

    st.markdown("#### PCA Positioning")

    try:
        fig = create_pca_scatter_plot(
            pca_df,
            selected_ticker=ticker,
            highlight_peers=None
        )
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.warning(f"PCA visualization unavailable: {e}")

    st.markdown("#### Nearest Structural Peers (PCA Distance)")

    try:
        peer_universe = pca_df.copy()

        target_row = peer_universe[peer_universe["ticker"] == ticker]

        if not target_row.empty:
            target_pc1 = target_row.iloc[0]["PC1"]
            target_pc2 = target_row.iloc[0]["PC2"]

            peer_universe["distance"] = (
                (peer_universe["PC1"] - target_pc1) ** 2 +
                (peer_universe["PC2"] - target_pc2) ** 2
            ) ** 0.5

            nearest_peers = (
                peer_universe[peer_universe["ticker"] != ticker]
                .sort_values("distance")
                .head(15)
            )

            # Persist for reuse (narrative, KG, structural analyst)
            st.session_state["nearest_peers"] = nearest_peers.copy()

            # --- Distance Percentile (Crowding Signal) ---
            all_distances = peer_universe["distance"]

            target_distance = peer_universe[
                peer_universe["ticker"] == ticker
            ]["distance"].values[0]

            percentile = (all_distances < target_distance).mean() * 100

            # Interpret crowding
            if percentile <= 10:
                crowd_label = "🔴 Highly Crowded"
                crowd_text = "This stock sits in a densely packed region of PCA space — elevated crowding risk."
            elif percentile <= 40:
                crowd_label = "🟠 Moderately Crowded"
                crowd_text = "This stock has meaningful structural proximity to peers."
            elif percentile <= 70:
                crowd_label = "🟡 Neutral"
                crowd_text = "This stock is neither especially crowded nor especially isolated."
            else:
                crowd_label = "🟢 Structurally Isolated"
                crowd_text = "This stock is more structurally differentiated from peers."

            st.metric(
                "Structural Crowding Percentile",
                f"{percentile:.1f}%",
                help="Lower = more crowded (closer to other stocks)"
            )
            st.caption(crowd_label)
            st.caption(crowd_text)

            # --- Peer Dispersion Metric ---
            avg_peer_distance = nearest_peers["distance"].mean()

            st.metric(
                "Peer Dispersion (Avg Distance)",
                f"{avg_peer_distance:.4f}",
                help="Lower = tightly clustered peers (higher fragility)"
            )

            if avg_peer_distance < 0.05:
                dispersion_label = "🔴 Very Tight Cluster"
                dispersion_text = "Peers are extremely close — elevated fragility and crowding risk."
            elif avg_peer_distance < 0.15:
                dispersion_label = "🟠 Moderate Dispersion"
                dispersion_text = "Peers are somewhat clustered with moderate structural similarity."
            else:
                dispersion_label = "🟢 Broadly Distributed"
                dispersion_text = "Peers are more spread out — lower structural fragility."

            st.caption(dispersion_label)
            st.caption(dispersion_text)

            # --- Structural Risk Score ---
            # Invert percentile (lower percentile = higher risk)
            crowding_risk = 100 - percentile

            # Normalize dispersion (simple scaling assumption)
            dispersion_risk = max(0, min(100, (0.15 - avg_peer_distance) / 0.15 * 100))

            structural_risk_score = 0.6 * crowding_risk + 0.4 * dispersion_risk

            # Define risk label BEFORE using it
            if structural_risk_score >= 75:
                risk_label = "🔴 High Structural Risk"
                bar_color = "red"
            elif structural_risk_score >= 40:
                risk_label = "🟠 Moderate Structural Risk"
                bar_color = "orange"
            else:
                risk_label = "🟢 Low Structural Risk"
                bar_color = "green"

            st.metric(
                "Structural Risk Score",
                f"{structural_risk_score:.1f}",
                help="Composite of crowding + peer dispersion (higher = more structurally risky)"
            )

            st.progress(int(structural_risk_score))

            st.markdown(
                f"<div style='color:{bar_color}; font-weight:600;'>Risk Level: {risk_label}</div>",
                unsafe_allow_html=True
            )

            # --- Executive Summary ---
            if structural_risk_score >= 75:
                summary_text = "This stock exhibits elevated structural risk due to high crowding and tightly aligned peer positioning."
            elif structural_risk_score >= 40:
                summary_text = "This stock shows moderate structural risk, with some crowding and partial peer alignment."
            else:
                summary_text = "This stock is structurally differentiated, with lower crowding and more dispersed peer positioning."

            st.info(summary_text)

            # --- PCA Loading-Based Structural Drivers (ROBUST VERSION) ---
            loadings = st.session_state.get("pca_loadings_df")

            if loadings is not None:
                try:
                    loadings_df = pd.DataFrame(loadings)

                    # 🔧 Fix orientation (handles dict vs DataFrame cases)
                    if "PC1" not in loadings_df.columns:
                        loadings_df = loadings_df.T

                    # 🔧 Force numeric (prevents abs() crash)
                    loadings_df = loadings_df.apply(pd.to_numeric, errors="coerce")

                    # 🔧 Drop empty rows
                    loadings_df = loadings_df.dropna(how="all")

                    # Combine PC1 + PC2 importance
                    loadings_df["importance"] = (
                        loadings_df["PC1"].abs() + loadings_df["PC2"].abs()
                    )

                    top_factors = (
                        loadings_df["importance"]
                        .sort_values(ascending=False)
                        .head(3)
                        .index.tolist()
                    )

                    directional_driver_labels = []

                    for factor in top_factors:
                        display_name = (
                            FEATURE_DISPLAY_NAMES.get(
                                factor,
                                factor.replace("_", " ").title()
                            ) if "FEATURE_DISPLAY_NAMES" in globals()
                            else factor.replace("_", " ").title()
                        )

                        pc1_loading = loadings_df.at[factor, "PC1"] if factor in loadings_df.index else 0
                        pc2_loading = loadings_df.at[factor, "PC2"] if factor in loadings_df.index else 0

                        directional_score = (pc1_loading * pc1) + (pc2_loading * pc2)

                        arrow = "↑" if directional_score >= 0 else "↓"

                        abs_score = abs(directional_score)
                        if abs_score >= 0.30:
                            strength = "Strong"
                        elif abs_score >= 0.15:
                            strength = "Moderate"
                        else:
                            strength = "Light"

                        directional_driver_labels.append(
                            f"{display_name} {arrow} ({strength})"
                        )

                    st.markdown("#### 🔬 Structural Drivers")

                    structural_drivers = []

                    for factor in top_factors:
                        display_name = (
                            FEATURE_DISPLAY_NAMES.get(
                                factor,
                                factor.replace("_", " ").title()
                            ) if "FEATURE_DISPLAY_NAMES" in globals()
                            else factor.replace("_", " ").title()
                        )

                        pc1_loading = loadings_df.at[factor, "PC1"] if factor in loadings_df.index else 0
                        pc2_loading = loadings_df.at[factor, "PC2"] if factor in loadings_df.index else 0

                        directional_score = (pc1_loading * pc1) + (pc2_loading * pc2)

                        direction_word = "Positive" if directional_score >= 0 else "Negative"

                        abs_score = abs(directional_score)
                        if abs_score >= 0.30:
                            strength = "Strong"
                        elif abs_score >= 0.15:
                            strength = "Moderate"
                        else:
                            strength = "Light"

                        structural_drivers.append({
                            "factor": factor,
                            "factor_name": factor,
                            "display_name": display_name,
                            "direction": direction_word,
                            "strength": strength,
                            "directional_score": directional_score,
                            "pc1_loading": pc1_loading,
                            "pc2_loading": pc2_loading,
                        }) 

                        st.markdown(
                            f"- **{display_name}**: {direction_word}, {strength} influence"
                        )

                    st.session_state.current_structural_drivers = structural_drivers

                except Exception as e:
                    st.caption(f"Driver calculation unavailable: {e}")

            else:
                st.caption("PCA loadings not available for structural driver analysis.")

            display_cols = [c for c in ["ticker", "cluster", "PC1", "PC2", "distance"] if c in nearest_peers.columns]

            st.dataframe(
                nearest_peers[display_cols],
                width="stretch",
                hide_index=True
            )
        else:
            st.info("Selected stock not found in PCA dataset.")

    except Exception as e:
        st.warning(f"Nearest peer calculation unavailable: {e}")


def render_stock_narrative_tab(
    ticker: str,
    pca_row: pd.Series,
    narrative_peers: pd.DataFrame,
    gics_sector: str,
):
    """Render the Stock / Individual Ticker Level narrative tab."""

    # Use structurally-derived peers if available
    peer_df = st.session_state.get("nearest_peers", narrative_peers)

    render_narrative_section(
        ticker=ticker,
        pca_row=pca_row,
        peer_df=peer_df,
        gics_sector=gics_sector,
    )

def render_stock_structural_tab(
    ticker: str,
):
    """Render the Stock / Individual Ticker Level structural intelligence tab."""

    # Lazy import to avoid Streamlit module cache issues
    try:
        from structural_analyst import run_structural_analysis
    except Exception as e:
        st.error(f"Structural Analyst failed to load: {e}")
        run_structural_analysis = None

    kg = st.session_state.get("kg_instance")
    kg_regime = st.session_state.get("kg_current_regime")
    nearest_peers = st.session_state.get("nearest_peers")
    nearest_peer_tickers = (
        nearest_peers["ticker"].dropna().astype(str).tolist()
        if nearest_peers is not None and "ticker" in nearest_peers.columns
        else []
    )

    if kg is None or kg_regime is None:
        st.info("Structural Analyst requires Knowledge Graph context.")
        return

    col1, col2 = st.columns([4, 1])

    with col1:
        structural_question = st.text_input(
            "Ask a structural question:",
            placeholder="e.g., What changed structurally in this regime?",
            key="structural_input"
        )

    chatbot = create_chatbot()

    with col2:
        run_structural = st.button(
            "Analyze",
            key="structural_btn",
            disabled=False
        )

    if run_structural and structural_question:
        with st.spinner("Running KG-backed structural analysis..."):
            try:
                evidence_packet = None

                narrative_packet = st.session_state.get("last_narrative_subgraph")

                if narrative_packet:
                    try:
                        evidence_packet = {
                            "question_type": "structural_drift",
                            "ticker": ticker,
                            "regime": kg_regime,
                            "subgraph": narrative_packet,
                            "peer_tickers": nearest_peer_tickers,
                        }
                    except Exception:
                        evidence_packet = None
                else:
                    try:
                        evidence_packet = build_structural_evidence_packet(
                            kg=kg,
                            ticker=ticker,
                            regime=kg_regime,
                            question_type="structural_drift",
                        )

                        # Attach peer context if available
                        if evidence_packet is not None:
                            evidence_packet["peer_tickers"] = nearest_peer_tickers

                    except Exception:
                        evidence_packet = None

                if not evidence_packet:
                    st.error("No stock-centered KG subgraph is available for structural analysis.")
                    result = None
                else:
                    if run_structural_analysis is not None:
                        result = run_structural_analysis(
                            evidence_packet=evidence_packet,
                            llm_callable=chatbot.call_llm_structural,
                        )
                    else:
                        st.error("Structural Analyst unavailable.")
                        result = None

                if result:
                    st.markdown("## 🧠 Structural Analysis (KG-Grounded AI)")
                    st.caption("LLM-based interpretation grounded strictly in Knowledge Graph evidence (Tier 2).")

                    structural_answer = result.get("answer", "No answer returned.")
                    st.markdown(f"**Assessment:** {structural_answer}")

                    if result.get("summary_bullets"):
                        st.markdown("### 🔑 Key Points")
                        unique_bullets = list(dict.fromkeys(result["summary_bullets"]))
                        for b in unique_bullets:
                            st.markdown(f"- {b}")

                    with st.expander("🔍 Evidence"):
                        evidence_label_map = {
                            "structural_drift_summary": "Structural Drift",
                            "quadrant_history_summary": "Quadrant History",
                            "peer_comparison_summary": "Peer Comparison",
                            "factor_rotation_summary": "Factor Rotation",
                            "regime_transition_summary": "Regime Transition",
                        }

                        for e in result.get("evidence", []):
                            if isinstance(e, dict):
                                raw_source = e.get("source_name", "Unknown Source")
                                source = evidence_label_map.get(raw_source, raw_source)
                                fact = e.get("fact", "")
                                st.markdown(f"- **{source}**: {fact}")
                            else:
                                st.markdown(f"- {str(e)}")

                    with st.expander("🕸️ Subgraph Snapshot"):
                        snapshot = result.get("subgraph_snapshot", {})

                        if isinstance(snapshot, dict):
                            node_count = snapshot.get("node_count", 0)
                            edge_count = snapshot.get("edge_count", 0)
                            included_node_ids = snapshot.get("included_node_ids", [])

                            snap_col1, snap_col2 = st.columns(2)
                            with snap_col1:
                                st.metric("Nodes", node_count)
                            with snap_col2:
                                st.metric("Edges", edge_count)

                            if included_node_ids:
                                st.markdown("**Included Node IDs**")
                                for node_id in included_node_ids:
                                    st.markdown(f"- `{node_id}`")
                            else:
                                st.caption("No included node IDs were returned.")
                        else:
                            st.caption("Subgraph snapshot unavailable.")

                    with st.expander("⚠️ Limits"):
                        limits = result.get("limits", [])

                        if isinstance(limits, list):
                            for l in limits:
                                st.markdown(f"- {l}")
                        else:
                            st.markdown(f"- {str(limits)}")

                    confidence = str(result.get("confidence", "unknown")).lower()

                    if confidence == "high":
                        confidence_label = "🟢 High"
                    elif confidence == "medium":
                        confidence_label = "🟡 Medium"
                    elif confidence == "low":
                        confidence_label = "🔴 Low"
                    else:
                        confidence_label = confidence.title()

                    st.caption(f"Confidence: {confidence_label}")

            except Exception as e:
                st.error(f"Structural analysis failed: {str(e)}")


def render_stock_workspace():
    """Render the Stock / Individual Ticker Level workspace."""

    st.markdown("## Stock / Individual Diagnostics")
    st.caption(
        "Issuer-level diagnostic workspace for evaluating structural position, peer-relative exposure, "
        "narrative interpretation, and KG-grounded structural intelligence."
    )

    # Get selected stock data
    stock_info = st.session_state.selected_stock

    if stock_info is None:
        st.info("👆 Select a stock ticker or PERMNO in the sidebar to begin analysis.")
        return
    pca_df = st.session_state.pca_df
    
    # Find stock in PCA DataFrame
    if stock_info['type'] == 'ticker':
        mask = pca_df['ticker'].str.upper() == stock_info['value'].upper()
    else:
        mask = pca_df['permno'] == stock_info['value']
    
    stock_pca_data = pca_df[mask]
    
    if stock_pca_data.empty:
        st.error(f"Could not find {stock_info['value']} in the PCA results.")
        return
    
    pca_row = stock_pca_data.iloc[0]
    stock_tab1, stock_tab2, stock_tab3, stock_tab4, stock_tab5 = st.tabs([
        "📌 Overview",
        "📊 Visuals",
        "👥 Peers & Positioning",
        "🧾 Narrative",
        "🧠 AI / Structural",
    ])
    
    # Render stock overview
    with stock_tab1:
        ticker, permno, cluster, pc1, pc2, quadrant = render_stock_overview_tab(
            pca_row
        )

    # Get quadrant peers (GICS-filtered universe for consistency)
    gics_filtered_pca = filter_by_gics_sector(
        pca_df,
        st.session_state.raw_data,
        ticker,
        st.session_state.get('gics_filter_mode', 'All Stocks')
    )

    quadrant_peers = get_stocks_in_same_quadrant(
        gics_filtered_pca, pc1, pc2, exclude_ticker=ticker
    )
    
    # 📊 Visuals Tab
    with stock_tab2:
        render_stock_visuals_tab(
            pca_df,
            ticker,
            pca_row,
            quadrant_peers
        )
    
    # Build GICS sector filtered peers — used consistently for all percentiles and narratives
    gics_filtered_pca = filter_by_gics_sector(pca_df, st.session_state.raw_data, ticker, "GICS Sector Only")
    narrative_peers = get_stocks_in_same_quadrant(gics_filtered_pca, pc1, pc2, exclude_ticker=ticker)

    # Get cluster summary and percentiles — using GICS sector peers only
    cluster_summary = get_cluster_summary(pca_df)
    available_features = [c for c in FEATURE_COLUMNS if c in pca_row.index]
    percentiles = compute_percentile_ranks(narrative_peers, pca_row, available_features)

    # Store for narrative engine
    st.session_state.current_percentiles = percentiles
    st.session_state.current_factor_data = get_factor_breakdown(pca_row)

    # Get GICS sector name for narrative
    gics_sector = 'N/A'
    if 'gicdesc' in st.session_state.raw_data.columns:
        ticker_rows = st.session_state.raw_data[st.session_state.raw_data['ticker'].str.upper() == ticker.upper()]
        if not ticker_rows.empty:
            gics_sector = ticker_rows['gicdesc'].iloc[0]

    # 👥 Peers & Positioning Tab
    with stock_tab3:
        render_stock_peers_tab(
            ticker=ticker,
            quadrant=quadrant,
            cluster=cluster,
            pc1=pc1,
            pc2=pc2,
            quadrant_peers=quadrant_peers,
            pca_df=pca_df,
        )

    # 🧾 Narrative Tab
    with stock_tab4:
        render_stock_narrative_tab(
            ticker=ticker,
            pca_row=pca_row,
            narrative_peers=narrative_peers,
            gics_sector=gics_sector,
        )

    # 🧠 AI / Structural Tab
    with stock_tab5:
        render_stock_structural_tab(ticker=ticker)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""

    # Initialize session state
    init_session_state()

    # Render main header
    render_main_header()
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading data from GitHub..."):
            result = load_and_process_data()
            raw_data, processed_data, pca_df, pca_model, kmeans_model, scaler, loadings, loadings_df, error = result
            
            if error:
                st.error(f"❌ Failed to load data: {error}")
                st.markdown(f"""
                **Troubleshooting:**
                1. Check your internet connection
                2. Verify the GitHub URL is accessible: `{GITHUB_DATA_URL}`
                3. Try refreshing the page
                """)
                return
            
            st.session_state.raw_data = raw_data
            st.session_state.processed_data = processed_data
            st.session_state.pca_df = pca_df
            st.session_state.pca_model = pca_model
            st.session_state.kmeans_model = kmeans_model
            st.session_state.scaler = scaler
            st.session_state.pca_loadings = loadings
            st.session_state.pca_loadings_df = loadings_df
            st.session_state.data_loaded = True
    
    # Initialize analysis scope (Phase 1)
    if "analysis_scope" not in st.session_state:
        st.session_state.analysis_scope = "Universe / Portfolio Level"
        
    # ============================================================
    # GLOBAL KG BUILD (SHARED ACROSS ALL MODES)
    # ============================================================

    period_scores = st.session_state.get("period_scores")

    if period_scores and (
        "kg_instance" not in st.session_state or st.session_state.kg_instance is None
    ):
        try:
            from kg_builder import build_kg
            from kg_interface import KnowledgeGraph

            kg_result = build_kg(
                period_data=period_scores,
                migration_df=st.session_state.get("migration_wide"),
                include_equity_nodes=True,
            )

            st.session_state.kg_instance = KnowledgeGraph(kg_result.graph)

            if "kg_current_regime" not in st.session_state:
                st.session_state.kg_current_regime = "Disinflation"

        except Exception as e:
            st.warning(f"KG build failed: {e}")
            st.session_state.kg_instance = None

    # Render sidebar AFTER data is loaded so counts and filters work on first click
    render_sidebar()

    # ============================================================
    # MAIN WORKSPACE ROUTING (ARCHITECTURE CLEAN SPLIT)
    # ============================================================

    if st.session_state.analysis_scope == "Universe / Portfolio Level":
        render_universe_workspace()

    else:
        render_stock_workspace()

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: gray; font-size: 0.8rem;">
            Equity Structural Diagnostics System (ESDS) | Institutional PCA-Based Risk Diagnostics |
            Data Source: GitHub Repository
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

