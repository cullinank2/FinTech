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
import os

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
        font-size: 1.25rem;   /* increase size */
        font-weight: 500;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .tab-header {
        font-size: 2rem !important;
        font-weight: 700 !important;
        text-align: center !important;
        padding: 0.5rem 0 !important;
        border-bottom: 3px solid red !important;
        margin-bottom: 1rem !important;
    }

    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
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
# UI HELPERS
# =============================================================================

def render_tab_header(text):
    st.markdown(f"""
    <div class="tab-header">{text}</div>
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
        pca_df, pca_model, kmeans_model, scaler, loadings = compute_pca_and_clusters(processed_data)
        
        return raw_data, processed_data, pca_df, pca_model, kmeans_model, scaler, loadings, None
        
    except Exception as e:
        return None, None, None, None, None, None, None, str(e)


# =============================================================================
# SIDEBAR COMPONENTS
# =============================================================================

def render_sidebar():
    """Render the sidebar with stock selection and controls."""
    
    st.sidebar.markdown("## 📊 Stock Selection")
    
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
    
    # Quick selection dropdown (always visible, disabled until data loads)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Select")
    
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
    
    
    # Check if stock is selected
    stock_selected = st.session_state.selected_stock is not None

    # GICS Sector filter for landing page Cluster Plot
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏭 GICS Sector Filter")

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

    if stock_selected:
        st.sidebar.selectbox(
            "Filter landing page by sector:",
            options=gics_sectors,
            index=0,
            key="gics_sector_filter_disabled",
            disabled=True,
            help="Sector filter is for the landing page only. Clear the ticker to use it."
        )
    else:
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

    # Visualizations dropdown (always visible, disabled until stock selected)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Visualizations")
    
    view_options = [
        "🎯 Cluster Plot",
        "👥 Quadrant Peers",
        "📊 Factor Analysis",
        "🕐 2D or 3D Time-Lapse",
        "🌐 3D Cluster View",
        "🌐 3D Quadrant Peers",
    ]
    
    # Always show dropdown, but disable if no stock selected
    selected_view = st.sidebar.selectbox(
        "Jump to view:",
        options=view_options,
        key="view_selector",
        disabled=not stock_selected
    )
    
    # Only update session state if stock is selected
    if stock_selected:
        # Store selection in session state
        if 'current_view' not in st.session_state:
            st.session_state.current_view = view_options[0]
        
        if selected_view != st.session_state.current_view:
            st.session_state.current_view = selected_view

    # GICS Sector filter (always visible, disabled until stock selected)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Stock Universe Filter")

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

    # Display axis interpretations
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Axis Interpretations")
    
    # Get live variance values from PCA model if available
    pc1_var = PC1_INTERPRETATION['variance_explained']  # fallback to hardcoded
    pc2_var = PC2_INTERPRETATION['variance_explained']  # fallback to hardcoded
    pc3_var = PC3_INTERPRETATION['variance_explained']  # fallback to hardcoded
    
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
    
    # Only show PC3 expander when 3D View is selected
    current_view = st.session_state.get('current_view', '')
    timelapse_is_3d = (
        current_view == "🕐 2D or 3D Time-Lapse" and
        st.session_state.get('timelapse_view_mode', '2D View') == '3D View'
    )
    if stock_selected and (current_view in ["🌐 3D Cluster View", "🌐 3D Quadrant Peers"] or timelapse_is_3d):
        with st.sidebar.expander(f"PC3 (Z-axis): {PC3_INTERPRETATION['name']}"):
            st.markdown(f"""
            **Explains ~{pc3_var}% of variance**

            **High values (↑ Up):**
            - {', '.join(PC3_INTERPRETATION['high_meaning'])}
        
            **Low values (↓ Down):**
            - {', '.join(PC3_INTERPRETATION['low_meaning'])}
            """)            
                        
            #**The cleanest factor in the model**
            
            #**High values (↑ Up):**
            #- Deep value stocks
            #- Asset-heavy companies
            #- Leveraged balance sheets
            
            #**Low values (↓ Down):**
            #- Growth / asset-light companies
            #- Capital efficient businesses
            #- Higher profitability vs book value
            
            #*≈ Momentum vs Profitability · Pure Value vs Growth*
            #""")
    
    # OpenAI API Key input
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Chatbot Settings")
    
    api_key = st.sidebar.text_input(
        "OpenAI API Key (for chatbot):",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key to enable the AI chatbot feature"
    )
    
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        st.session_state.chatbot = create_chatbot(api_key)
     


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

    return response.choices[0].message.content


def render_main_header():
    """Render the main page header."""
    st.markdown("""
    <div class="main-header">
        📈 EQUITY STRUCTURAL DIAGNOSTICS
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero-text">
    Analyze stocks using Principal Component Analysis (PCA) to understand their 
    characteristics across quality, stability, leverage, and size dimensions.
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
    st.markdown(f"## 📊 Analysis: {ticker} &nbsp;&nbsp;&nbsp;&nbsp; **GICS Sector:** {gics_sector}")
    
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
    """Render the visualization sections based on dropdown selection.""" 
    
    # Get current view from session state
    current_view = st.session_state.get('current_view', '🎯 Cluster Plot')
    
    # Apply GICS sector filter if selected
    filter_mode = st.session_state.get('gics_filter_mode', 'All Stocks')
    filtered_pca_df = filter_by_gics_sector(pca_df, raw_data, selected_ticker, filter_mode)
    
    # Show info about filtering
    if filter_mode == "GICS Sector Only":
        sector_count = len(filtered_pca_df)
        st.info(f"📊 Showing {sector_count} stocks in the same GICS sector as {selected_ticker}")
    
    # Display the selected visualization
    if current_view == "🎯 Cluster Plot":
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
    
    elif current_view == "👥 Quadrant Peers":

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
    
    elif current_view == "🌐 3D Quadrant Peers":
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

    elif current_view == "📊 Factor Analysis":
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
    
    elif current_view == "🕐 2D or 3D Time-Lapse":
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
    
    elif current_view == "🌐 3D Cluster View":
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
    current_view = st.session_state.get('current_view', '🎯 Cluster Plot')

    timelapse_is_3d = (
        current_view == "🕐 2D or 3D Time-Lapse" and
        st.session_state.get('timelapse_view_mode', '2D View') == '3D View'
    )
    cluster_3d = current_view in ["🌐 3D Cluster View", "🌐 3D Quadrant Peers"]
    show_pc3 = timelapse_is_3d or cluster_3d

    with st.spinner("Generating narrative analysis..."):
        kg           = st.session_state.get("kg_instance")
        kg_regime    = st.session_state.get("kg_current_regime")
        # --- FIX: Align AI peer group with filtered universe ---
        peer_df = peer_df.copy()
        # peer_df already contains quadrant-filtered peers

        sections = generate_narrative(  # ARCH: intentional NE boundary — sole authorized call site
            ticker          = ticker,
            pca_row         = pca_row,
            percentiles     = percentiles,
            factor_data     = factor_data,
            peer_df         = peer_df,
            raw_data        = raw_data,
            loadings        = loadings,
            gics_sector     = gics_sector,
            show_pc3        = show_pc3,
            kg              = kg,
            current_regime  = kg_regime,
        )

    current_view = st.session_state.get('current_view', '🎯 Cluster Plot')

    if current_view == '🎯 Cluster Plot':
        st.markdown(sections['summary'])
    elif current_view == '📊 Factor Analysis':
        st.markdown(sections['factors'])
    elif current_view == '🕐 2D or 3D Time-Lapse':
        st.markdown(sections['trajectory'])
    elif current_view in ['👥 Quadrant Peers', '🌐 3D Quadrant Peers']:
        st.markdown(sections['peers'])
    else:
        st.markdown(sections['summary'])

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
    kg_subgraph = None
    if kg is not None and kg_regime is not None:
        try:
            kg_subgraph = kg.serialize_subgraph([
                f"regime:{kg_regime}",
                f"stock:{ticker}",
            ])
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
            raw_data, processed_data, pca_df, pca_model, kmeans_model, scaler, loadings, error = result
            
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
            st.session_state.data_loaded = True
    
    # Render sidebar AFTER data is loaded so counts and filters work on first click
    render_sidebar()

    # Check for selected stock
    if st.session_state.selected_stock is None:
        st.info("👆 Enter a stock ticker or PERMNO in the sidebar to begin analysis.")
        
        # Show overall cluster summary
        if st.session_state.pca_df is not None:

            landing_tab1, landing_tab2, landing_tab3, landing_tab4 = st.tabs([
                "📊 Cluster Overview",
                "📐 Period Comparison",
                "🧠 Knowledge Graph",
                "🔬 Structural Intelligence",
            ])

            with landing_tab1:
                render_tab_header("📊 Cluster Overview")

                # Apply GICS sector filter if selected on landing page
                plot_df = st.session_state.pca_df.copy()
                selected_sector = st.session_state.get('selected_gics_sector', 'All Sectors')

                if selected_sector and selected_sector != "All Sectors" and 'gicdesc' in st.session_state.raw_data.columns:
                    sector_tickers = st.session_state.raw_data[
                        st.session_state.raw_data['gicdesc'] == selected_sector
                    ]['ticker'].unique()
                    plot_df = plot_df[plot_df['ticker'].isin(sector_tickers)]

                total_count = len(plot_df)
                if selected_sector and selected_sector != "All Sectors":
                    sector_label = f" — {selected_sector} ({total_count})"
                else:
                    sector_label = f" — All Sectors ({total_count})"
                st.markdown(f"### 📊 Cluster Overview{sector_label}")

                fig = create_pca_scatter_plot(plot_df)
                st.plotly_chart(fig, use_container_width=True)

                cluster_summary = get_cluster_summary(plot_df)
                fig_summary = create_cluster_summary_plot(cluster_summary)
                st.plotly_chart(fig_summary, use_container_width=True)

            with landing_tab2:
                st.subheader("📐 Sub-Period PCA Comparison")
                st.caption(
                    "Validates whether factor structure and stock behavior genuinely differ "
                    "across Post-COVID, Rate Shock, and Disinflation regimes."
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
                                - **< 0.05** 🟢 Factor structure nearly identical across periods.
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

                                # Build KG instance for Tier 1 narrative + Tier 2 chatbot
                                try:
                                    from kg_builder import build_kg
                                    from kg_interface import KnowledgeGraph
                                    kg_result = build_kg(
                                        period_data          = st.session_state.get("period_scores"),
                                        migration_df         = st.session_state.get("migration_wide"),
                                        include_equity_nodes = True,
                                    )
                                    st.session_state["kg_instance"] = KnowledgeGraph(kg_result.graph)
                                    st.session_state["kg_current_regime"] = "Disinflation"
                                except Exception:
                                    pass

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

                            # Section 4: Factor Loadings by Sub-Period
                            st.markdown("---")
                            st.markdown("### 4 · Factor Loadings by Sub-Period")
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
                                        .applymap(_style_loading, subset=period_cols)
                                        .applymap(_style_delta,   subset=delta_cols)
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

        with landing_tab3:
                render_kg_tab()

        with landing_tab4:
                render_structural_intelligence_tab()

        return
    
    # Get selected stock data
    stock_info = st.session_state.selected_stock
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
    
    # Render stock overview
    ticker, permno, cluster, pc1, pc2, quadrant = render_stock_overview(
        st.session_state.raw_data, 
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
    
    # Render visualizations
    render_visualizations(
        pca_df,
        ticker,
        pca_row,
        quadrant_peers,
        st.session_state.raw_data,  # ← CORRECT! This has gicdesc column
        st.session_state.pca_model,
        st.session_state.scaler
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

    # Render narrative engine section
    render_narrative_section(ticker, pca_row, narrative_peers, gics_sector)

    # Render chatbot section
    render_chatbot_section(
    ticker, permno, cluster, quadrant, pc1, pc2,
    pca_row,
    percentiles,
    len(quadrant_peers),
    cluster_summary,
    total_universe=len(gics_filtered_pca)
)

    # ============================================================
    # STRUCTURAL ANALYST (KG-GROUNDED, ZERO-HALLUCINATION MODE)
    # ============================================================

    # Lazy import to avoid Streamlit module cache issues
    try:
        from structural_analyst import run_structural_analysis
    except Exception as e:
        st.error(f"Structural Analyst failed to load: {e}")
        run_structural_analysis = None

    st.markdown("---")
    st.markdown("## 🧠 Structural Analyst (KG-Backed, No Hallucination)")

    kg = st.session_state.get("kg_instance")
    kg_regime = st.session_state.get("kg_current_regime")

    if kg is None or kg_regime is None:
        st.info("Structural Analyst requires Knowledge Graph context.")
    else:
        col1, col2 = st.columns([4, 1])

        with col1:
            structural_question = st.text_input(
                "Ask a structural question:",
                placeholder="e.g., What changed structurally in this regime?",
                key="structural_input"
            )

        # Initialize chatbot for structural analysis
        chatbot = create_chatbot()

        with col2:
            run_structural = st.button("Analyze", key="structural_btn")

        if run_structural and structural_question:

            with st.spinner("Running KG-backed structural analysis..."):

                try:
                    evidence_packet = build_structural_evidence_packet(
                        kg=kg,
                        ticker=ticker,
                        regime=kg_regime,
                        question_type="structural_drift",
                    )

                    if run_structural_analysis is not None:
                        result = run_structural_analysis(
                            evidence_packet=evidence_packet,
                            llm_callable=chatbot.call_llm_structural,
                        )
                    else:
                        st.error("Structural Analyst unavailable.")
                        result = None
                    if result:
                        st.markdown("### 📊 Structural Answer")
                        st.write(result.get("answer", "No answer returned."))

                        if result.get("summary_bullets"):
                            st.markdown("### 🔑 Key Points")

                            # Deduplicate bullets (preserve order)
                            unique_bullets = list(dict.fromkeys(result["summary_bullets"]))

                            for b in unique_bullets:
                                st.markdown(f"- {b}")

                    with st.expander("🔍 Evidence"):
                        for e in result.get("evidence", []):
                            st.markdown(
                                f"- **{e.get('source_name')}**: {e.get('fact')}"
                            )

                    with st.expander("⚠️ Limits"):
                        st.write(result.get("limits", "None stated"))

                    st.caption(f"Confidence: {result.get('confidence')}")

                except Exception as e:
                    st.error(f"Structural analysis failed: {str(e)}")



# Footer
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.8rem;">
        Stock PCA Cluster Analysis | Built with Streamlit | 
        Data Source: GitHub Repository
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

