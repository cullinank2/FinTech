#Force rebuild - 2026-02-08
"""
Stock PCA Cluster Analysis - Streamlit Web Application

A comprehensive interactive dashboard for analyzing stock clusters using
Principal Component Analysis (PCA). Features include:
- Interactive 2D/3D PCA visualizations
- Quadrant-based peer comparison
- Factor breakdown analysis
- Time-lapse animations
- AI-powered chatbot for contextual questions

Author: Beautiful Mind FinTech
Date: 2024
"""

import streamlit as st
import pandas as pd
import os
from typing import Optional

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
    OPENAI_API_KEY_PLACEHOLDER
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
    get_factor_breakdown
)
from visualizations import (
    create_pca_scatter_plot,
    create_quadrant_comparison_plot,
    create_factor_radar_chart,
    create_percentile_chart,
    create_timelapse_animation,
    create_3d_pca_plot,
    create_cluster_summary_plot
)
from chatbot import create_chatbot, SAMPLE_QUESTIONS


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(**PAGE_CONFIG)


# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

st.markdown("""
<style>
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
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
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
        'chat_history': []
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
        return None, None, None, None, None, None, str(e)


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
    
    
    # Visualizations dropdown (always visible, disabled until stock selected)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Visualizations")
    
    # Check if stock is selected
    stock_selected = st.session_state.selected_stock is not None
    
    view_options = [
        "🎯 Cluster Plot",
        "👥 Quadrant Peers",
        "📊 Factor Analysis",
        "🕐 Time-Lapse",
        "🌐 3D Cluster View",
        "🌐 3D Quadrant Peers"
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
    if stock_selected and current_view in ["🌐 3D Cluster View", "🌐 3D Quadrant Peers"]:
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

def render_main_header():
    """Render the main page header."""
    st.markdown("""
    <div class="main-header">
        📈 STOCK FACTOR ANALYTICS
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Analyze stocks using Principal Component Analysis (PCA) to understand their 
    characteristics across quality, stability, leverage, and size dimensions.
    """)


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
    
    # PCA scores
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "PC1 Score (Quality/Stability)", 
            f"{pc1:.3f}",
            delta="Higher Quality" if pc1 >= 0 else "Riskier",
            delta_color="normal" if pc1 >= 0 else "inverse"
        )
    with col2:
        st.metric(
            "PC2 Score (Size/Leverage)", 
            f"{pc2:.3f}",
            delta="Large/Leveraged" if pc2 >= 0 else "Cash-Rich",
            delta_color="off"
        )
    
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
        st.markdown("""
        Explore quadrant peers in 3D space. The Z-axis (PC3) reveals the 
        Value vs Growth dimension within your peer group.
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
                    <b>📐 PC3: Leverage & Risk Profile:</b><i> cleanest factor in the model</i>{pc3_variance}{combined_variance}<br>
                    ↑ <b>High PC3 - </b><i>Deep value · Asset-heavy · Leveraged</i> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                    ↓ <b>Low PC3 - </b><i>Growth · Asset-light · Capital efficient</i>
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
    
    elif current_view == "🕐 Time-Lapse":
        st.markdown("### 🕐 Historical Movement Animation")
        st.markdown("""
        Watch how the stock's position has changed over time in the PCA space.
        Click **Play** to start the animation.
        """)
        
        if st.button("🔄 Generate Time-Lapse Animation", key="timelapse_btn"):
            with st.spinner("Preparing animation..."):
                time_series_data = prepare_time_series_data(
                    raw_data, selected_ticker, pca_model, scaler
                )
                
                if not time_series_data.empty:
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
                    <b>📐 PC3: Leverage & Risk Profile:</b><i> cleanest factor in the model</i>{pc3_variance}{combined_variance}<br>
                    ↑ <b>High PC3 - </b><i>Deep value · Asset-heavy · Leveraged</i> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                    ↓ <b>Low PC3 - </b><i>Growth · Asset-light · Capital efficient</i>
                </div>
                """, unsafe_allow_html=True)
            
            # Chart full width
            fig_3d = create_3d_pca_plot(filtered_pca_df, selected_ticker)
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("3D visualization requires PC3 data.")


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
    cluster_summary: pd.DataFrame
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
        cluster_summary=cluster_summary
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
    
    # Render sidebar
    render_sidebar()
    
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
    
    # Check for selected stock
    if st.session_state.selected_stock is None:
        st.info("👆 Enter a stock ticker or PERMNO in the sidebar to begin analysis.")
        
        # Show overall cluster summary
        if st.session_state.pca_df is not None:
            st.markdown("### 📊 Cluster Overview")
            
            fig = create_pca_scatter_plot(st.session_state.pca_df)
            st.plotly_chart(fig, use_container_width=True)
            
            cluster_summary = get_cluster_summary(st.session_state.pca_df)
            fig_summary = create_cluster_summary_plot(cluster_summary)
            st.plotly_chart(fig_summary, use_container_width=True)
        
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
    
    # Get quadrant peers
    quadrant_peers = get_stocks_in_same_quadrant(
        pca_df, pc1, pc2, exclude_ticker=ticker
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
    
    # Get cluster summary and percentiles for chatbot
    cluster_summary = get_cluster_summary(pca_df)
    available_features = [c for c in FEATURE_COLUMNS if c in pca_row.index]
    percentiles = compute_percentile_ranks(quadrant_peers, pca_row, available_features)
    
    # Render chatbot section
    render_chatbot_section(
        ticker, permno, cluster, quadrant, pc1, pc2,
        pca_row, percentiles, len(quadrant_peers), cluster_summary
    )
    
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
