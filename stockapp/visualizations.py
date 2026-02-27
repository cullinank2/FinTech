"""
Visualization functions for Stock PCA Cluster Analysis.
Creates interactive Plotly charts for the Streamlit dashboard.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple

from config import (
    CLUSTER_COLORS,
    PLOT_WIDTH,
    PLOT_HEIGHT,
    ANIMATION_FRAME_DURATION,
    QUADRANTS,
    FACTOR_CATEGORIES,
    PC1_INTERPRETATION,
    PC2_INTERPRETATION,
    PC3_INTERPRETATION,
    FEATURE_DISPLAY_NAMES,
    FEATURE_DISPLAY_ORDER
)


# =============================================================================
# MAIN PCA SCATTER PLOT (PRIMARY VISUALIZATION)
# =============================================================================

def create_pca_scatter_plot(
    pca_df: pd.DataFrame,
    selected_ticker: Optional[str] = None,
    show_quadrant_labels: bool = True
) -> go.Figure:
    """
    Create the main 2D PCA scatter plot with cluster coloring.
    This is the primary visualization users see after selecting a stock.
    
    Args:
        pca_df: DataFrame with PC1, PC2, and cluster columns
        selected_ticker: Ticker to highlight (optional)
        show_quadrant_labels: Whether to show quadrant labels
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Add quadrant background shading (force full-quadrant coverage)
    x_pad = 0.5
    y_pad = 0.5

    x_lim = max(abs(pca_df["PC1"].min()), abs(pca_df["PC1"].max())) + x_pad
    y_lim = max(abs(pca_df["PC2"].min()), abs(pca_df["PC2"].max())) + y_pad

    x_min, x_max = -x_lim, x_lim
    y_min, y_max = -y_lim, y_lim

    
    # Add quadrant background shading
    #x_min, x_max = pca_df['PC1'].min() - 0.5, pca_df['PC1'].max() + 0.5
    #y_min, y_max = pca_df['PC2'].min() - 0.5, pca_df['PC2'].max() + 0.5
    
    # Quadrant colors (very light)
    quadrant_colors = {
        'Q1': 'rgba(144, 238, 144, 0.1)',  # Light green
        'Q2': 'rgba(255, 182, 193, 0.1)',  # Light pink
        'Q3': 'rgba(255, 255, 224, 0.1)',  # Light yellow
        'Q4': 'rgba(173, 216, 230, 0.1)'   # Light blue
    }
    
    # Add quadrant rectangles (span full axis range)
    fig.add_shape(type="rect", xref="x", yref="y",
              x0=0, y0=0, x1=x_max, y1=y_max,
              fillcolor=quadrant_colors['Q1'],
              line=dict(width=0), layer="below")

    fig.add_shape(type="rect", xref="x", yref="y",
              x0=x_min, y0=0, x1=0, y1=y_max,
              fillcolor=quadrant_colors['Q2'],
              line=dict(width=0), layer="below")

    fig.add_shape(type="rect", xref="x", yref="y",
              x0=x_min, y0=y_min, x1=0, y1=0,
              fillcolor=quadrant_colors['Q3'],
              line=dict(width=0), layer="below")

    fig.add_shape(type="rect", xref="x", yref="y",
              x0=0, y0=y_min, x1=x_max, y1=0,
              fillcolor=quadrant_colors['Q4'],
              line=dict(width=0), layer="below")


    # Add quadrant rectangles
    # Q1: High Quality + Large/Leveraged (top-right)
    #fig.add_shape(type="rect", x0=0, y0=0, x1=x_max, y1=y_max,
    #              fillcolor=quadrant_colors['Q1'], line=dict(width=0))
    # Q2: Lower Quality + Large/Leveraged (top-left)
    #fig.add_shape(type="rect", x0=x_min, y0=0, x1=0, y1=y_max,
    #              fillcolor=quadrant_colors['Q2'], line=dict(width=0))
    # Q3: Lower Quality + Cash-Rich (bottom-left)
    #fig.add_shape(type="rect", x0=x_min, y0=y_min, x1=0, y1=0,
    #              fillcolor=quadrant_colors['Q3'], line=dict(width=0))
    # Q4: High Quality + Cash-Rich (bottom-right)
    #fig.add_shape(type="rect", x0=0, y0=y_min, x1=x_max, y1=0,
    #              fillcolor=quadrant_colors['Q4'], line=dict(width=0))
    
    # ---------------------------------------------------------------------
    # Invisible hover targets for quadrant explanations (reliable version)
    # ---------------------------------------------------------------------

    hover_mark = dict(size=90, color="rgba(0,0,0,0.01)", line=dict(width=0))  # tiny opacity so it can be hovered

    fig.add_trace(go.Scatter(
        x=[x_max - 0.25], y=[y_max - 0.25],   # top-right corner (Q1)
        mode="markers",
        marker=hover_mark,
        hovertemplate=(
            f"<b>Q1: {QUADRANTS['Q1']['name']}</b><br>"
            f"{QUADRANTS['Q1']['description']}<br>"
            f"Characteristics: {', '.join(QUADRANTS['Q1']['characteristics'])}"
            "<extra></extra>"
        ),
        showlegend=False,
        name=""
    ))

    fig.add_trace(go.Scatter(
        x=[x_min + 0.25], y=[y_max - 0.25],   # top-left corner (Q2)
        mode="markers",
        marker=hover_mark,
        hovertemplate=(
            f"<b>Q2: {QUADRANTS['Q2']['name']}</b><br>"
            f"{QUADRANTS['Q2']['description']}<br>"
            f"Characteristics: {', '.join(QUADRANTS['Q2']['characteristics'])}"
            "<extra></extra>"
        ),
        showlegend=False,
        name=""
    ))

    fig.add_trace(go.Scatter(
        x=[x_min + 0.25], y=[y_min + 0.25],   # bottom-left corner (Q3)
        mode="markers",
        marker=hover_mark,
        hovertemplate=(
            f"<b>Q3: {QUADRANTS['Q3']['name']}</b><br>"
            f"{QUADRANTS['Q3']['description']}<br>"
            f"Characteristics: {', '.join(QUADRANTS['Q3']['characteristics'])}"
            "<extra></extra>"
        ),
        showlegend=False,
        name=""
    ))

    fig.add_trace(go.Scatter(
        x=[x_max - 0.25], y=[y_min + 0.25],   # bottom-right corner (Q4)
        mode="markers",
        marker=hover_mark,
        hovertemplate=(
            f"<b>Q4: {QUADRANTS['Q4']['name']}</b><br>"
            f"{QUADRANTS['Q4']['description']}<br>"
            f"Characteristics: {', '.join(QUADRANTS['Q4']['characteristics'])}"
            "<extra></extra>"
        ),
        showlegend=False,
        name=""
    ))
    
    # Add axis lines at origin
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Plot each cluster
    for cluster_id in sorted(pca_df['cluster'].unique()):
        cluster_data = pca_df[pca_df['cluster'] == cluster_id]
        
        # Determine if any point should be highlighted
        is_selected = (
            selected_ticker is not None and 
            'ticker' in cluster_data.columns and
            selected_ticker.upper() in cluster_data['ticker'].str.upper().values
        )
        
        # Create hover text
        hover_text = []
        for _, row in cluster_data.iterrows():
            text = f"<b>{row.get('ticker', 'N/A')}</b><br>"
            text += f"PERMNO: {row.get('permno', 'N/A')}<br>"
            text += f"Cluster: {cluster_id}<br>"
            text += f"PC1: {row['PC1']:.3f}<br>"
            text += f"PC2: {row['PC2']:.3f}"
            hover_text.append(text)
        
        fig.add_trace(go.Scatter(
            x=cluster_data['PC1'],
            y=cluster_data['PC2'],
            mode='markers',
            marker=dict(
                size=10 if not is_selected else 8,
                color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            name=f'Cluster {cluster_id}',
            hovertemplate="%{text}<extra></extra>",
            text=hover_text,
            customdata=cluster_data['ticker'].values
        ))
    
    # Highlight selected stock
    if selected_ticker and 'ticker' in pca_df.columns:
        selected_data = pca_df[pca_df['ticker'].str.upper() == selected_ticker.upper()]
        if not selected_data.empty:
            fig.add_trace(go.Scatter(
                x=selected_data['PC1'],
                y=selected_data['PC2'],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color='red',
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                text=[selected_ticker],
                textposition='top center',
                textfont=dict(size=14, color='red', family='Arial Black'),
                name=f'Selected: {selected_ticker}',
                hovertemplate=f"<b>{selected_ticker}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>"
            ))
    
    # Add quadrant labels
    if show_quadrant_labels:
        labels = [
            dict(x=x_max*0.7, y=y_max*0.9, text=f"Q1: {QUADRANTS['Q1']['name']}",
                showarrow=False, font=dict(size=14, color='green')),
            dict(x=x_min*0.7, y=y_max*0.9, text=f"Q2: {QUADRANTS['Q2']['name']}",
                showarrow=False, font=dict(size=14, color='red')),
            dict(x=x_min*0.7, y=y_min*0.9, text=f"Q3: {QUADRANTS['Q3']['name']}",
                showarrow=False, font=dict(size=14, color='olive')),
            dict(x=x_max*0.7, y=y_min*0.9, text=f"Q4: {QUADRANTS['Q4']['name']}",
                showarrow=False, font=dict(size=14, color='grey'))
        ]
        for label in labels:
            fig.add_annotation(**label)

    # Add axis characteristic labels (ONCE) ✅ OUTSIDE LOOP
    # Theme-aware hover styling (used by all axis hover tooltips)
    theme_base = st.get_option("theme.base")              # 'dark' or 'light'
    is_dark = (theme_base == "dark")                      # True if dark mode

    hover_bg = "rgba(20,20,20,0.95)" if is_dark else "rgba(255,255,255,0.95)"
    hover_font = "white" if is_dark else "black"
    hover_border = "rgba(255,255,255,0.25)" if is_dark else "rgba(0,0,0,0.15)"

    # Build PC1 / PC2 axis texts (what you already show on the chart)
    pc1_high_text = f"→ {'<br>'.join(PC1_INTERPRETATION['high_meaning'])}"
    pc1_low_text  = f"← {'<br>'.join(PC1_INTERPRETATION['low_meaning'])}"
    pc2_high_text = f"↑ {', '.join(PC2_INTERPRETATION['high_meaning'])}"
    pc2_low_text  = f"↓ {', '.join(PC2_INTERPRETATION['low_meaning'])}"

    # Build live loading hover text (PC1  and PC2 positive / negative) if available
    pc1_pos_hover = ""
    pc1_neg_hover = ""
    pc2_pos_hover = ""
    pc2_neg_hover = ""

    if 'pca_loadings' in st.session_state:
        loadings = st.session_state.pca_loadings

        if 'PC1' in loadings and 'positive' in loadings['PC1']:
            top_pos = list(loadings['PC1']['positive'].items())[:3]
            pc1_pos_hover = "<br>".join([
                f"{FEATURE_DISPLAY_NAMES.get(feat, feat)}: {val:.3f}"
                for feat, val in top_pos
            ])

        if 'PC1' in loadings and 'negative' in loadings['PC1']:
            top_neg = list(loadings['PC1']['negative'].items())[:3]
            pc1_neg_hover = "<br>".join([
                f"{FEATURE_DISPLAY_NAMES.get(feat, feat)}: {val:.3f}"
                for feat, val in top_neg
            ])

        if 'PC2' in loadings and 'positive' in loadings['PC2']:
            top_pos = list(loadings['PC2']['positive'].items())[:3]
            pc2_pos_hover = "<br>".join([
                f"{FEATURE_DISPLAY_NAMES.get(feat, feat)}: {val:.3f}"
                for feat, val in top_pos
            ])

        if 'PC2' in loadings and 'negative' in loadings['PC2']:
            top_neg = list(loadings['PC2']['negative'].items())[:3]
            pc2_neg_hover = "<br>".join([
                f"{FEATURE_DISPLAY_NAMES.get(feat, feat)}: {val:.3f}"
                for feat, val in top_neg
            ])

    def _add_hover_target(x, y, hover_html):
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            marker=dict(size=40, color="rgba(0,0,0,0.01)", line=dict(width=0)),  # invisible but hoverable
            showlegend=False,
            hovertemplate=hover_html + "<extra></extra>",
            hoverlabel=dict(
                bgcolor=hover_bg,
                font=dict(size=12, color=hover_font),
                bordercolor=hover_border
            ),
            name=""
        ))

    # ------------------------------------------------------------------
    # 1) Invisible hover targets Pos & Neg PCA Drivers (reliable in dark/light mode)
    # ------------------------------------------------------------------

    # PC1 HIGH hover target (right side)
    _add_hover_target(
        x_max - 0.10, 0,
        f"<b>PC1 High Drivers:</b><br>{pc1_pos_hover}"
        if pc1_pos_hover else
        "<b>PC1 High Drivers:</b><br>(No loadings available)"
    )

    # PC1 LOW hover target (left side)
    _add_hover_target(
        x_min + 0.10, 0,
        f"<b>PC1 Low Drivers:</b><br>{pc1_neg_hover}"
        if pc1_neg_hover else
        "<b>PC1 Low Drivers:</b><br>(No loadings available)"
    )

    # PC2 HIGH hover target (top)
    _add_hover_target(
        0, y_max - 0.10,
        f"<b>PC2 High Drivers:</b><br>{pc2_pos_hover}"
        if pc2_pos_hover else
        "<b>PC2 High Drivers:</b><br>(No loadings available)"
    )

    # PC2 LOW hover target (bottom)
    _add_hover_target(
        0, y_min + 0.10,
        f"<b>PC2 Low Drivers:</b><br>{pc2_neg_hover}"
        if pc2_neg_hover else
        "<b>PC2 Low Drivers:</b><br>(No loadings available)"
    )

    # ------------------------------------------------------------------
    # 2) Visible axis annotation text (no hovertext here)
    # ------------------------------------------------------------------

    fig.add_annotation(
        x=x_max, y=0,
        text=pc1_high_text,
        showarrow=False,
        xanchor='right',
        xshift=-20,
        yshift=15,
        font=dict(size=11, color='gray')
    )

    fig.add_annotation(
        x=x_min, y=0,
        text=pc1_low_text,
        showarrow=False,
        xanchor='left',
        xshift=20,
        yshift=15,
        font=dict(size=11, color='gray')
    )

    fig.add_annotation(
        x=0, y=y_max,
        text=pc2_high_text,
        showarrow=False,
        xshift=60,
        font=dict(size=11, color='gray')
    )

    fig.add_annotation(
        x=0, y=y_min,
        text=pc2_low_text,
        showarrow=False,
        xshift=60,
        font=dict(size=11, color='gray')
    )

    # Update layout (ONCE)
    fig.update_layout(
        title=dict(text='Stock PCA Cluster Analysis', font=dict(size=20)),
        xaxis_title=f"PC1: {PC1_INTERPRETATION['name']}",
        yaxis_title=f"PC2: {PC2_INTERPRETATION['name']}",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        hovermode='closest'
    )

    # Lock axis ranges (ONCE)
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])

    return fig

# =============================================================================
# QUADRANT PEER COMPARISON
# =============================================================================

def create_quadrant_comparison_plot(
    pca_df: pd.DataFrame,
    selected_ticker: str,
    quadrant_peers: pd.DataFrame
) -> go.Figure:
    """
    Create a focused view of the selected stock's quadrant with peers.
    
    Args:
        pca_df: Full PCA DataFrame
        selected_ticker: Selected stock ticker
        quadrant_peers: DataFrame of peers in same quadrant
        
    Returns:
        Plotly Figure object
    """
    if 'ticker' not in pca_df.columns:
        return go.Figure()
    
    selected_data = pca_df[pca_df['ticker'].str.upper() == selected_ticker.upper()]
    if selected_data.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Plot quadrant peers
    if not quadrant_peers.empty:
        hover_text = [
            f"<b>{row.get('ticker', 'N/A')}</b><br>PC1: {row['PC1']:.3f}<br>PC2: {row['PC2']:.3f}"
            for _, row in quadrant_peers.iterrows()
        ]
        
        fig.add_trace(go.Scatter(
            x=quadrant_peers['PC1'],
            y=quadrant_peers['PC2'],
            mode='markers+text',
            marker=dict(size=12, color='lightblue', opacity=0.7,
                       line=dict(width=1, color='darkblue')),
            text=quadrant_peers['ticker'] if 'ticker' in quadrant_peers.columns else None,
            textposition='top center',
            textfont=dict(size=8),
            name='Quadrant Peers',
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_text
        ))
    
    # Highlight selected stock
    fig.add_trace(go.Scatter(
        x=selected_data['PC1'],
        y=selected_data['PC2'],
        mode='markers+text',
        marker=dict(size=25, color='red', symbol='star',
                   line=dict(width=2, color='black')),
        text=[selected_ticker],
        textposition='bottom center',
        textfont=dict(size=14, color='red', family='Arial Black'),
        name=f'Selected: {selected_ticker}',
        hovertemplate=f"<b>{selected_ticker}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>"
    ))
    
    # Add quadrant reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Add invisible hover target for ONLY the subject quadrant
    x_min = min(quadrant_peers['PC1'].min(), selected_data['PC1'].min()) - 0.5
    x_max = max(quadrant_peers['PC1'].max(), selected_data['PC1'].max()) + 0.5
    y_min = min(quadrant_peers['PC2'].min(), selected_data['PC2'].min()) - 0.5
    y_max = max(quadrant_peers['PC2'].max(), selected_data['PC2'].max()) + 0.5

    hover_mark = dict(size=90, color="rgba(0,0,0,0.01)", line=dict(width=0))

    # Determine which quadrant the selected stock is in
    sel_pc1 = selected_data['PC1'].iloc[0]
    sel_pc2 = selected_data['PC2'].iloc[0]

    if sel_pc1 >= 0 and sel_pc2 >= 0:
        # Q1: top-right
        hover_x = x_max - 0.1
        hover_y = y_max - 0.1
        hover_text = (
            f"<b>Q1: {QUADRANTS['Q1']['name']}</b><br>"
            f"{QUADRANTS['Q1']['description']}<br>"
            f"<b>Characteristics:</b> {', '.join(QUADRANTS['Q1']['characteristics'])}"
            "<extra></extra>" 
        )
    elif sel_pc1 < 0 and sel_pc2 >= 0:
        # Q2: top-left
        hover_x = x_min + 0.1
        hover_y = y_max - 0.1
        hover_text = (
            f"<b>Q2: {QUADRANTS['Q2']['name']}</b><br>"
            f"{QUADRANTS['Q2']['description']}<br>"
            f"<b>Characteristics:</b> {', '.join(QUADRANTS['Q2']['characteristics'])}"
            "<extra></extra>"
        )
    elif sel_pc1 < 0 and sel_pc2 < 0:
        # Q3: bottom-left
        hover_x = x_min + 0.1
        hover_y = y_min + 0.1
        hover_text = (
            f"<b>Q3: {QUADRANTS['Q3']['name']}</b><br>"
            f"{QUADRANTS['Q3']['description']}<br>"
            f"<b>Characteristics:</b> {', '.join(QUADRANTS['Q3']['characteristics'])}"
            "<extra></extra>"
        )
    else:
        # Q4: bottom-right
        hover_x = x_max - 0.1
        hover_y = y_min + 0.1
        hover_text = (
            f"<b>Q4: {QUADRANTS['Q4']['name']}</b><br>"
            f"{QUADRANTS['Q4']['description']}<br>"
            f"<b>Characteristics:</b> {', '.join(QUADRANTS['Q4']['characteristics'])}"
            "<extra></extra>"
        )

    fig.add_trace(go.Scatter(
        x=[hover_x], y=[hover_y],
        mode="markers", marker=hover_mark,
        hovertemplate=hover_text,
        showlegend=False, name=""
    ))

    # Set axis ranges to show ONLY the subject quadrant
    if sel_pc1 >= 0 and sel_pc2 >= 0:
        # Q1: top-right - x from 0 to max, y from 0 to max
        x_range = [0, x_max]
        y_range = [0, y_max]
    elif sel_pc1 < 0 and sel_pc2 >= 0:
        # Q2: top-left - x from min to 0, y from 0 to max
        x_range = [x_min, 0]
        y_range = [0, y_max]
    elif sel_pc1 < 0 and sel_pc2 < 0:
        # Q3: bottom-left - x from min to 0, y from min to 0
        x_range = [x_min, 0]
        y_range = [y_min, 0]
    else:
        # Q4: bottom-right - x from 0 to max, y from min to 0
        x_range = [0, x_max]
        y_range = [y_min, 0]

    fig.update_layout(
        title=f'Quadrant Peers for {selected_ticker}',
        xaxis_title=f"PC1: {PC1_INTERPRETATION['name']}",
        yaxis_title=f"PC2: {PC2_INTERPRETATION['name']}",
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        width=PLOT_WIDTH,
        height=500,
        showlegend=True
    )
    
    return fig

# =============================================================================
# FACTOR BREAKDOWN RADAR CHART
# =============================================================================

def create_factor_radar_chart(
    factor_data: Dict[str, Dict[str, float]],
    ticker: str,
    percentiles: Optional[Dict[str, float]] = None
) -> go.Figure:
    """
    Create a radar chart showing the factor breakdown for a stock.
    Uses percentile values for proper normalization (0-100 scale).
    """
    # Flatten factor data WITH DISPLAY NAMES
    categories = []
    values = []
    
    for category, features in factor_data.items():
        for feature, value in features.items():
            # Use .get() with fallback to original name if not found
            display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
            categories.append(display_name)
            values.append(value)
    
    # Use percentiles if provided, otherwise normalize raw values (0-100 scale)
    if percentiles:
        # Extract percentile values in same order as categories
        percentile_values = []
        for category, features in factor_data.items():
            for feature, value in features.items():
                pct = percentiles.get(feature, 50)  # Default to 50th percentile if missing
                percentile_values.append(pct)
        radar_values = percentile_values
    else:
        # Fallback: normalize raw values to 0-100 scale
        if values:
            min_val = min(values)
            max_val = max(values)
            if max_val != min_val:
                radar_values = [((v - min_val) / (max_val - min_val)) * 100 for v in values]
            else:
                radar_values = [50] * len(values)
        else:
            radar_values = []
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=radar_values + [radar_values[0]] if radar_values else [],
        theta=categories + [categories[0]] if categories else [],
        fill='toself',
        fillcolor='rgba(27, 158, 119, 0.3)',
        line=dict(color=CLUSTER_COLORS[0], width=2),
        name=ticker,
        hovertemplate="<b>%{theta}</b><br>Percentile: %{r:.1f}%<br>Raw Value: %{customdata:.4f}<extra></extra>",
        customdata=values + [values[0]] if values else []
    ))
    
    # Theme-aware tick color
    theme_base = st.get_option("theme.base")
    tick_color = "rgba(255,255,255,0.75)" if theme_base == "dark" else "rgba(0,0,0,0.65)"

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='%',
                tickfont=dict(
                    size=10,
                    color=tick_color
                ),
                gridcolor="rgba(128,128,128,0.3)",
                linecolor="rgba(128,128,128,0.3)"
            )
        ),
        title=f'Factor Breakdown: {ticker}',
        width=500,
        height=500,
        showlegend=False
    )
    
    return fig

# =============================================================================
# PERCENTILE RANKING BAR CHART
# =============================================================================

def create_percentile_chart(
    percentiles: Dict[str, float],
    ticker: str
) -> go.Figure:
    """
    Create a horizontal bar chart showing percentile rankings.
    """
    # Sort features according to FEATURE_DISPLAY_ORDER
    ordered_features = [f for f in FEATURE_DISPLAY_ORDER if f in percentiles]
    
    # Convert feature codes to display names in order
    features = [FEATURE_DISPLAY_NAMES.get(f, f) for f in ordered_features]
    values = [percentiles[f] for f in ordered_features]
    
    # Color bars based on percentile (green for high, red for low)
    colors = ['green' if v >= 50 else 'red' for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=features,
        x=values,
        orientation='h',
        marker=dict(
            color=colors,
            opacity=0.7
        ),
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Percentile: %{x:.1f}%<extra></extra>"
    ))
    
    # Add 50th percentile reference line
    fig.add_vline(
        x=50,
        line_dash="dash",
        line_color="gray",
        opacity=0.7,
        annotation_text="50th Percentile",
        annotation_position="top"
    )

    # Base layout (NO annotations list here)
    fig.update_layout(
        title=f'Percentile Rankings vs GICS Sector Peers: {ticker}',
        xaxis_title='Percentile Rank',
        yaxis_title='Factor',
        xaxis=dict(range=[0, 105]),
        yaxis=dict(autorange='reversed'),
        width=600,
        height=max(460, len(features) * 30 + 60),  # give a bit more room
        showlegend=False,
        margin=dict(b=110)  # bigger bottom margin so footnote is visible
    )

    return fig


# =============================================================================
# FACTOR TREND CHART
# =============================================================================

def create_factor_trend_chart(
    raw_data: pd.DataFrame,
    ticker: str,
    loadings_dict: dict,
    period: str = "All"
) -> Tuple[go.Figure, go.Figure]:
    """
    Create dual trend charts showing how top PC1 and PC2 drivers change over time.
    
    Args:
        raw_data: Full raw dataset with time series data
        ticker: Selected stock ticker
        loadings_dict: PCA loadings from session state
        period: Time period filter ("1Y", "3Y", "5Y", "All")
        
    Returns:
        Tuple of (PC1 figure, PC2 figure)
    """
    # Filter for selected ticker
    stock_data = raw_data[raw_data['ticker'].str.upper() == ticker.upper()].copy()
    
    if stock_data.empty:
        return go.Figure(), go.Figure()
    
    # Get date column
    date_col = None
    for col in ['public_date', 'date', 'datadate']:
        if col in stock_data.columns:
            date_col = col
            break
    
    if date_col is None:
        return go.Figure(), go.Figure()
    
    # Sort by date
    stock_data = stock_data.sort_values(date_col)
    stock_data['date'] = pd.to_datetime(stock_data[date_col])
    
    # Filter by period
    if period != "All":
        end_date = stock_data['date'].max()
        if period == "1Y":
            start_date = end_date - pd.DateOffset(years=1)
        elif period == "3Y":
            start_date = end_date - pd.DateOffset(years=3)
        elif period == "5Y":
            start_date = end_date - pd.DateOffset(years=5)
        stock_data = stock_data[stock_data['date'] >= start_date]
    
    # Get top 3 drivers for PC1 and PC2
    pc1_features = []
    pc2_features = []
    
    if 'PC1' in loadings_dict and 'positive' in loadings_dict['PC1']:
        pc1_features = list(loadings_dict['PC1']['positive'].keys())[:3]
    
    if 'PC2' in loadings_dict and 'positive' in loadings_dict['PC2']:
        pc2_features = list(loadings_dict['PC2']['positive'].keys())[:3]
    
    # Create PC1 trend chart
    fig_pc1 = go.Figure()
    
    for feat in pc1_features:
        if feat in stock_data.columns:
            # Normalize to 0-100 percentile scale within this stock's history
            values = stock_data[feat].values
            if len(values) > 0 and values.std() > 0:
                normalized = ((values - values.min()) / (values.max() - values.min())) * 100
            else:
                normalized = [50] * len(values)
            
            display_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
            
            fig_pc1.add_trace(go.Scatter(
                x=stock_data['date'],
                y=normalized,
                mode='lines+markers',
                name=display_name,
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{display_name}</b><br>Date: %{{x|%Y-%m-%d}}<br>Normalized: %{{y:.1f}}<extra></extra>"
            ))
    
    fig_pc1.update_layout(
        title=f'PC1 Driver Trends: {ticker}',
        xaxis_title='Date',
        yaxis_title='Normalized Value (0-100)',
        width=PLOT_WIDTH,
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Create PC2 trend chart
    fig_pc2 = go.Figure()
    
    for feat in pc2_features:
        if feat in stock_data.columns:
            # Normalize to 0-100 percentile scale within this stock's history
            values = stock_data[feat].values
            if len(values) > 0 and values.std() > 0:
                normalized = ((values - values.min()) / (values.max() - values.min())) * 100
            else:
                normalized = [50] * len(values)
            
            display_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
            
            fig_pc2.add_trace(go.Scatter(
                x=stock_data['date'],
                y=normalized,
                mode='lines+markers',
                name=display_name,
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{display_name}</b><br>Date: %{{x|%Y-%m-%d}}<br>Normalized: %{{y:.1f}}<extra></extra>"
            ))
    
    fig_pc2.update_layout(
        title=f'PC2 Driver Trends: {ticker}',
        xaxis_title='Date',
        yaxis_title='Normalized Value (0-100)',
        width=PLOT_WIDTH,
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig_pc1, fig_pc2


# =============================================================================
# TIME-LAPSE ANIMATION
# =============================================================================

def create_timelapse_animation(
    time_series_df: pd.DataFrame,
    ticker: str,
    pca_df: pd.DataFrame
) -> go.Figure:
    """
    Create an animated scatter plot showing stock movement over time.
    
    Args:
        time_series_df: DataFrame with date, PC1, PC2 columns
        ticker: Stock ticker
        pca_df: Full PCA DataFrame for background context
        
    Returns:
        Plotly Figure with animation
    """
    if time_series_df.empty:
        return go.Figure()
    
    # Create figure with frames
    fig = go.Figure()
    
    # Add static background (all other stocks, dimmed)
    fig.add_trace(go.Scatter(
        x=pca_df['PC1'],
        y=pca_df['PC2'],
        mode='markers',
        marker=dict(size=6, color='lightgray', opacity=0.3),
        name='Other Stocks',
        hoverinfo='skip'
    ))
    
    # Add axis lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add the animated trace (current position)
    fig.add_trace(go.Scatter(
        x=[time_series_df['PC1'].iloc[0]],
        y=[time_series_df['PC2'].iloc[0]],
        mode='markers+text',
        marker=dict(size=15, color='red', symbol='star'),
        text=[ticker],
        textposition='top center',
        name=f'{ticker} Position'
    ))
    
    # Add trail trace
    fig.add_trace(go.Scatter(
        x=time_series_df['PC1'].iloc[:1],
        y=time_series_df['PC2'].iloc[:1],
        mode='lines',
        line=dict(color='blue', width=2, dash='dot'),
        name='Historical Path',
        opacity=0.5
    ))
    
    # Create frames for animation
    frames = []
    for i in range(1, len(time_series_df)):
        frame_data = [
            # Background (unchanged)
            go.Scatter(
                x=pca_df['PC1'],
                y=pca_df['PC2'],
                mode='markers',
                marker=dict(size=6, color='lightgray', opacity=0.3)
            ),
            # Current position
            go.Scatter(
                x=[time_series_df['PC1'].iloc[i]],
                y=[time_series_df['PC2'].iloc[i]],
                mode='markers+text',
                marker=dict(size=15, color='red', symbol='star'),
                text=[ticker],
                textposition='top center'
            ),
            # Trail
            go.Scatter(
                x=time_series_df['PC1'].iloc[:i+1],
                y=time_series_df['PC2'].iloc[:i+1],
                mode='lines',
                line=dict(color='blue', width=2, dash='dot'),
                opacity=0.5
            )
        ]
        
        frame_name = str(time_series_df['date'].iloc[i])[:10]
        frames.append(go.Frame(data=frame_data, name=frame_name))
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        title=f'Historical Movement: {ticker}',
        xaxis_title=f"PC1: {PC1_INTERPRETATION['name']}",
        yaxis_title=f"PC2: {PC2_INTERPRETATION['name']}",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.15,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": ANIMATION_FRAME_DURATION, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 200}
                        }]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=12),
                    prefix="Date: ",
                    visible=True,
                    xanchor="center"
                ),
                transition=dict(duration=200),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.05,
                y=0,
                steps=[
                    dict(
                        args=[[f.name], {
                            "frame": {"duration": 200, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 200}
                        }],
                        label=f.name,
                        method="animate"
                    )
                    for f in frames
                ]
            )
        ]
    )
    
    return fig

def create_timelapse_animation_3d(
    time_series_df: pd.DataFrame,
    ticker: str,
    pca_df: pd.DataFrame
) -> go.Figure:
    """
    Create an animated 3D scatter plot showing stock movement over time.
    
    Args:
        time_series_df: DataFrame with date, PC1, PC2, PC3 columns
        ticker: Stock ticker
        pca_df: Full PCA DataFrame for background context
        
    Returns:
        Plotly Figure with 3D animation
    """
    if time_series_df.empty or 'PC3' not in time_series_df.columns:
        return go.Figure()
    
    # Create figure with frames
    fig = go.Figure()
    
    # Add static background (all other stocks, dimmed)
    if 'PC3' in pca_df.columns:
        fig.add_trace(go.Scatter3d(
            x=pca_df['PC1'],
            y=pca_df['PC2'],
            z=pca_df['PC3'],
            mode='markers',
            marker=dict(size=3, color='lightgray', opacity=0.2),
            name='Other Stocks',
            hoverinfo='skip'
        ))
    
    # Add the animated trace (current position)
    fig.add_trace(go.Scatter3d(
        x=[time_series_df['PC1'].iloc[0]],
        y=[time_series_df['PC2'].iloc[0]],
        z=[time_series_df['PC3'].iloc[0]],
        mode='markers+text',
        marker=dict(size=10, color='red', symbol='diamond'),
        text=[ticker],
        textposition='top center',
        name=f'{ticker} Position'
    ))
    
    # Add trail trace
    fig.add_trace(go.Scatter3d(
        x=time_series_df['PC1'].iloc[:1],
        y=time_series_df['PC2'].iloc[:1],
        z=time_series_df['PC3'].iloc[:1],
        mode='lines',
        line=dict(color='blue', width=4),
        name='Historical Path',
        opacity=0.6
    ))
    
    # Create frames for animation
    frames = []
    for i in range(1, len(time_series_df)):
        frame_data = [
            # Background (unchanged)
            go.Scatter3d(
                x=pca_df['PC1'] if 'PC3' in pca_df.columns else [],
                y=pca_df['PC2'] if 'PC3' in pca_df.columns else [],
                z=pca_df['PC3'] if 'PC3' in pca_df.columns else [],
                mode='markers',
                marker=dict(size=3, color='lightgray', opacity=0.2)
            ),
            # Current position
            go.Scatter3d(
                x=[time_series_df['PC1'].iloc[i]],
                y=[time_series_df['PC2'].iloc[i]],
                z=[time_series_df['PC3'].iloc[i]],
                mode='markers+text',
                marker=dict(size=10, color='red', symbol='diamond'),
                text=[ticker],
                textposition='top center'
            ),
            # Trail
            go.Scatter3d(
                x=time_series_df['PC1'].iloc[:i+1],
                y=time_series_df['PC2'].iloc[:i+1],
                z=time_series_df['PC3'].iloc[:i+1],
                mode='lines',
                line=dict(color='blue', width=4),
                opacity=0.6
            )
        ]
        
        frame_name = str(time_series_df['date'].iloc[i])[:10]
        frames.append(go.Frame(data=frame_data, name=frame_name))
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        title=f'3D Historical Movement: {ticker}',
        scene=dict(
            xaxis_title=f"PC1: {PC1_INTERPRETATION['name']}",
            yaxis_title=f"PC2: {PC2_INTERPRETATION['name']}",
            zaxis_title=f"PC3: {PC3_INTERPRETATION['name']}"
        ),
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.15,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": ANIMATION_FRAME_DURATION, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 200}
                        }]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=12),
                    prefix="Date: ",
                    visible=True,
                    xanchor="center"
                ),
                transition=dict(duration=200),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.05,
                y=0,
                steps=[
                    dict(
                        args=[[f.name], {
                            "frame": {"duration": 200, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 200}
                        }],
                        label=f.name,
                        method="animate"
                    )
                    for f in frames
                ]
            )
        ]
    )
    
    return fig


# =============================================================================
# 3D PCA VISUALIZATION
# =============================================================================

def create_3d_pca_plot(
    pca_df: pd.DataFrame,
    selected_ticker: Optional[str] = None
) -> go.Figure:
    """
    Create a 3D PCA scatter plot.
    
    Args:
        pca_df: DataFrame with PC1, PC2, PC3, and cluster columns
        selected_ticker: Ticker to highlight (optional)
        
    Returns:
        Plotly Figure object
    """
    if 'PC3' not in pca_df.columns:
        return go.Figure()
    
    fig = go.Figure()
    
    # Plot each cluster
    for cluster_id in sorted(pca_df['cluster'].unique()):
        cluster_data = pca_df[pca_df['cluster'] == cluster_id]
        
        hover_text = [
            f"<b>{row.get('ticker', 'N/A')}</b><br>Cluster: {cluster_id}"
            for _, row in cluster_data.iterrows()
        ]
        
        fig.add_trace(go.Scatter3d(
            x=cluster_data['PC1'],
            y=cluster_data['PC2'],
            z=cluster_data['PC3'],
            mode='markers',
            marker=dict(
                size=5,
                color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
                opacity=0.7
            ),
            name=f'Cluster {cluster_id}',
            hovertemplate="%{text}<extra></extra>",
            text=hover_text
        ))
    
    # Highlight selected stock
    if selected_ticker and 'ticker' in pca_df.columns:
        selected_data = pca_df[pca_df['ticker'].str.upper() == selected_ticker.upper()]
        if not selected_data.empty:
            fig.add_trace(go.Scatter3d(
                x=selected_data['PC1'],
                y=selected_data['PC2'],
                z=selected_data['PC3'],
                mode='markers+text',
                marker=dict(size=15, color='red', symbol='diamond'),
                text=[selected_ticker],
                name=f'Selected: {selected_ticker}'
            )) 
    
    fig.update_layout(
        title='3D PCA Cluster Visualization',
        scene=dict(
            xaxis_title=f"PC1: {PC1_INTERPRETATION['name']}",
            yaxis_title=f"PC2: {PC2_INTERPRETATION['name']}",
            zaxis_title=f"PC3: {PC3_INTERPRETATION['name']}"
        ),
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        showlegend=True
    )
    
    return fig


# =============================================================================
# 3D QUADRANT PEERS VISUALIZATION
# =============================================================================

def create_3d_quadrant_peers_plot(
    quadrant_peers: pd.DataFrame,
    selected_ticker: str,
    pca_row: pd.Series
) -> go.Figure:
    """
    Create a 3D scatter plot of quadrant peers with selected stock highlighted.
    
    Args:
        quadrant_peers: DataFrame of peers in same quadrant
        selected_ticker: Selected stock ticker
        pca_row: PCA row for selected stock
        
    Returns:
        Plotly Figure object
    """
    if 'PC3' not in quadrant_peers.columns:
        return go.Figure()
    
    fig = go.Figure()
    
    # Plot quadrant peers
    hover_text = [
        f"<b>{row.get('ticker', 'N/A')}</b><br>"
        f"PC1: {row['PC1']:.3f}<br>"
        f"PC2: {row['PC2']:.3f}<br>"
        f"PC3: {row['PC3']:.3f}<br>"
        f"Cluster: {row.get('cluster', 'N/A')}"
        for _, row in quadrant_peers.iterrows()
    ]
    
    fig.add_trace(go.Scatter3d(
        x=quadrant_peers['PC1'],
        y=quadrant_peers['PC2'],
        z=quadrant_peers['PC3'],
        mode='markers+text',
        marker=dict(
            size=6,
            color='lightblue',
            opacity=0.7,
            line=dict(width=1, color='darkblue')
        ),
        text=quadrant_peers['ticker'] if 'ticker' in quadrant_peers.columns else None,
        textposition='top center',
        textfont=dict(size=8),
        name='Quadrant Peers',
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_text
    ))
    
    # Highlight selected stock
    sel_pc3 = pca_row.get('PC3', 0)
    sel_pc1 = pca_row.get('PC1', 0)
    sel_pc2 = pca_row.get('PC2', 0)
    
    fig.add_trace(go.Scatter3d(
        x=[sel_pc1],
        y=[sel_pc2],
        z=[sel_pc3],
        mode='markers+text',
        marker=dict(size=15, color='red', symbol='diamond'),
        text=[selected_ticker],
        textfont=dict(size=12, color='red'),
        name=f'Selected: {selected_ticker}',
        hovertemplate=(
            f"<b>{selected_ticker}</b><br>"
            f"PC1: {sel_pc1:.3f}<br>"
            f"PC2: {sel_pc2:.3f}<br>"
            f"PC3: {sel_pc3:.3f}"
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=f'3D Quadrant Peers for {selected_ticker}',
        scene=dict(
            xaxis_title=f"PC1: {PC1_INTERPRETATION['name']}",
            yaxis_title=f"PC2: {PC2_INTERPRETATION['name']}",
            zaxis_title=f"PC3: {PC3_INTERPRETATION['name']}"
        ),
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        showlegend=True
    )
    
    return fig


# =============================================================================
# CLUSTER SUMMARY VISUALIZATION
# =============================================================================

def create_cluster_summary_plot(cluster_summary: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart comparing cluster characteristics.
    
    Args:
        cluster_summary: DataFrame with cluster statistics
        
    Returns:
        Plotly Figure object
    """
    if cluster_summary.empty:
        return go.Figure()
    
    # Select key metrics for comparison
    key_metrics = ['PC1_mean', 'PC2_mean']
    available_metrics = [m for m in key_metrics if m in cluster_summary.columns]
    
    if not available_metrics:
        return go.Figure()
    
    fig = make_subplots(
        rows=1, cols=len(available_metrics),
        subplot_titles=[m.replace('_mean', ' (Mean)') for m in available_metrics]
    )
    
    for i, metric in enumerate(available_metrics, 1):
        fig.add_trace(
            go.Bar(
                x=[f'Cluster {c}' for c in cluster_summary.index],
                y=cluster_summary[metric],
                marker_color=[CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in cluster_summary.index],
                showlegend=False
            ),
            row=1, col=i
        )
    
    fig.update_layout(
        title='Cluster Comparison',
        width=PLOT_WIDTH,
        height=400
    )
    
    return fig
