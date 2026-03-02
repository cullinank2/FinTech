"""
Utility functions for Stock PCA Cluster Analysis App.
Handles data loading, preprocessing, PCA computation, and clustering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

print("🔴 UTILS.PY LOADED 🔴") 

from config import (
    GITHUB_DATA_URL, 
    LOCAL_DATA_PATH,
    FEATURE_COLUMNS, 
    N_COMPONENTS, 
    N_CLUSTERS,
    FACTOR_CATEGORIES,
    QUADRANTS
)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_data_from_github(url: str = GITHUB_DATA_URL) -> pd.DataFrame:
    """
    Load the factors dataset from GitHub.
    
    Args:
        url: GitHub raw URL for the CSV file
        
    Returns:
        DataFrame with the loaded data
        
    Raises:
        Exception if data cannot be loaded
    """
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        raise Exception(f"Failed to load data from GitHub: {str(e)}")


def load_data_local(path: str = LOCAL_DATA_PATH) -> pd.DataFrame:
    """
    Load the factors dataset from local file.
    
    Args:
        path: Local file path for the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise Exception(f"Failed to load local data: {str(e)}")


def load_data(use_github: bool = True) -> pd.DataFrame:
    """
    Load data with fallback from GitHub to local.
    
    Args:
        use_github: Whether to try loading from GitHub first
        
    Returns:
        DataFrame with the loaded data
    """
    if use_github:
        try:
            return load_data_from_github()
        except:
            print("GitHub load failed, trying local file...")
            return load_data_local()
    else:
        return load_data_local()


# =============================================================================
# DATA PREPROCESSING FUNCTIONS
# =============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw data for PCA analysis.
    
    - Handles missing values
    - Ensures proper data types
    - Creates necessary identifiers
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure ticker and permno columns exist
    if 'ticker' not in df.columns and 'TICKER' in df.columns:
        df['ticker'] = df['TICKER']
    if 'permno' not in df.columns and 'PERMNO' in df.columns:
        df['permno'] = df['PERMNO']
    
    # Convert permno to string for consistent lookups
    if 'permno' in df.columns:
        df['permno'] = df['permno'].astype(str)
    
    # Convert date column if present
    date_cols = ['public_date', 'date', 'datadate']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            break
    
    # Handle missing values in feature columns
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    df[available_features] = df[available_features].fillna(df[available_features].median())
    
    return df


def get_available_tickers(df: pd.DataFrame) -> List[str]:
    """
    Get list of unique tickers from the dataset.
    
    Args:
        df: DataFrame with ticker column
        
    Returns:
        Sorted list of unique tickers
    """
    if 'ticker' in df.columns:
        return sorted(df['ticker'].dropna().unique().tolist())
    return []


def get_available_permnos(df: pd.DataFrame) -> List[str]:
    """
    Get list of unique PERMNOs from the dataset.
    
    Args:
        df: DataFrame with permno column
        
    Returns:
        Sorted list of unique PERMNOs
    """
    if 'permno' in df.columns:
        return sorted(df['permno'].dropna().unique().tolist())
    return []


def validate_stock_input(df: pd.DataFrame, input_value: str) -> Tuple[bool, str, str]:
    """
    Validate user input and determine if it's a ticker or PERMNO.
    
    Args:
        df: DataFrame with stock data
        input_value: User input string
        
    Returns:
        Tuple of (is_valid, input_type, normalized_value)
    """
    input_value = input_value.strip().upper()
    
    # Check if it's a ticker
    if 'ticker' in df.columns and input_value in df['ticker'].str.upper().values:
        return True, 'ticker', input_value
    
    # Check if it's a PERMNO
    if 'permno' in df.columns and input_value in df['permno'].values:
        return True, 'permno', input_value
    
    # Try numeric PERMNO
    try:
        permno_int = str(int(input_value))
        if 'permno' in df.columns and permno_int in df['permno'].values:
            return True, 'permno', permno_int
    except ValueError:
        pass
    
    return False, 'unknown', input_value


def filter_stock_data(df: pd.DataFrame, input_value: str, input_type: str) -> pd.DataFrame:
    """
    Filter DataFrame for a specific stock.
    
    Args:
        df: Full DataFrame
        input_value: Stock identifier value
        input_type: Either 'ticker' or 'permno'
        
    Returns:
        Filtered DataFrame for the selected stock
    """
    if input_type == 'ticker':
        return df[df['ticker'].str.upper() == input_value.upper()]
    elif input_type == 'permno':
        return df[df['permno'] == input_value]
    return pd.DataFrame()


# =============================================================================
# PCA AND CLUSTERING FUNCTIONS
# =============================================================================

def compute_pca_and_clusters(
    df: pd.DataFrame,
    n_components: int = N_COMPONENTS,
    n_clusters: int = N_CLUSTERS
) -> Tuple[pd.DataFrame, PCA, KMeans, StandardScaler]:
    """
    Compute PCA and KMeans clustering on the data.
    
    Args:
        df: Preprocessed DataFrame
        n_components: Number of PCA components
        n_clusters: Number of clusters for KMeans
        
    Returns:
        Tuple of (DataFrame with PCA scores and clusters, PCA object, KMeans object, Scaler)
    """
    # Get available feature columns
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    
    # Aggregate data by ticker (average across time periods)
    if 'ticker' in df.columns:
        agg_df = df.groupby(['permno', 'ticker'])[available_features].mean().reset_index()
    else:
        agg_df = df.groupby('permno')[available_features].mean().reset_index()
    
    # Drop any remaining NaN values
    agg_df = agg_df.dropna(subset=available_features)
    
    # Extract feature matrix
    X = agg_df[available_features].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Compute KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Create loadings dictionary for session state
    loadings_dict = {}
    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    loadings_df = pd.DataFrame(pca.components_.T, index=available_features, columns=pc_cols)
    
    for i in range(n_components):
        pc_name = f'PC{i+1}'
        
        # Get top 5 positive and negative
        positive = loadings_df[pc_name].sort_values(ascending=False).head(5).to_dict()
        negative = loadings_df[pc_name].sort_values(ascending=True).head(5).to_dict()
        
        loadings_dict[pc_name] = {
            'positive': positive,
            'negative': negative
        }
    
    # Create result DataFrame
    result_df = agg_df.copy()
    for i in range(n_components):
        result_df[f'PC{i+1}'] = X_pca[:, i]
    result_df['cluster'] = clusters
    
    return result_df, pca, kmeans, scaler, loadings_dict


def get_pca_loadings(pca: PCA, feature_names: List[str]) -> pd.DataFrame:
    """
    Get PCA loadings as a DataFrame.
    
    Args:
        pca: Fitted PCA object
        feature_names: List of feature names
        
    Returns:
        DataFrame with loadings for each component
    """
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=feature_names
    )
    return loadings


def determine_quadrant(pc1: float, pc2: float) -> str:
    """
    Determine which quadrant a stock falls into based on PC1 and PC2 scores.
    
    Args:
        pc1: PC1 score
        pc2: PC2 score
        
    Returns:
        Quadrant identifier (Q1, Q2, Q3, or Q4)
    """
    if pc1 >= 0 and pc2 >= 0:
        return 'Q1'
    elif pc1 < 0 and pc2 >= 0:
        return 'Q2'
    elif pc1 < 0 and pc2 < 0:
        return 'Q3'
    else:  # pc1 >= 0 and pc2 < 0
        return 'Q4'


def get_stocks_in_same_quadrant(
    pca_df: pd.DataFrame, 
    target_pc1: float, 
    target_pc2: float,
    exclude_ticker: Optional[str] = None
) -> pd.DataFrame:
    """
    Get all stocks in the same quadrant as the target.
    
    Args:
        pca_df: DataFrame with PCA scores
        target_pc1: Target stock's PC1 score
        target_pc2: Target stock's PC2 score
        exclude_ticker: Ticker to exclude from results
        
    Returns:
        DataFrame with stocks in the same quadrant
    """
    quadrant = determine_quadrant(target_pc1, target_pc2)
    
    # Filter based on quadrant
    if quadrant == 'Q1':
        mask = (pca_df['PC1'] >= 0) & (pca_df['PC2'] >= 0)
    elif quadrant == 'Q2':
        mask = (pca_df['PC1'] < 0) & (pca_df['PC2'] >= 0)
    elif quadrant == 'Q3':
        mask = (pca_df['PC1'] < 0) & (pca_df['PC2'] < 0)
    else:
        mask = (pca_df['PC1'] >= 0) & (pca_df['PC2'] < 0)
    
    result = pca_df[mask].copy()
    
    if exclude_ticker and 'ticker' in result.columns:
        result = result[result['ticker'].str.upper() != exclude_ticker.upper()]
    
    return result


def compute_percentile_ranks(
    df: pd.DataFrame,
    target_row: pd.Series,
    feature_columns: List[str]
) -> Dict[str, float]:
    """
    Compute percentile ranks for a stock's features relative to its peers.
    
    Args:
        df: DataFrame with peer stocks
        target_row: Series with target stock's data
        feature_columns: List of feature columns to rank
        
    Returns:
        Dictionary mapping feature names to percentile ranks
    """
    percentiles = {}
    for col in feature_columns:
        if col in df.columns and col in target_row.index:
            values = df[col].dropna()
            target_val = target_row[col]
            percentile = (values < target_val).sum() / len(values) * 100
            percentiles[col] = percentile
    return percentiles


def get_cluster_summary(pca_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for each cluster.
    
    Args:
        pca_df: DataFrame with PCA scores and cluster assignments
        
    Returns:
        DataFrame with cluster summary statistics
    """
    available_features = [col for col in FEATURE_COLUMNS if col in pca_df.columns]
    
    summary = pca_df.groupby('cluster').agg({
        **{col: 'mean' for col in available_features},
        'ticker': 'count' if 'ticker' in pca_df.columns else 'permno'
    }).rename(columns={'ticker': 'count', 'permno': 'count'})
    
    # Add PC means
    summary['PC1_mean'] = pca_df.groupby('cluster')['PC1'].mean()
    summary['PC2_mean'] = pca_df.groupby('cluster')['PC2'].mean()
    
    return summary


# =============================================================================
# TIME SERIES FUNCTIONS FOR ANIMATION
# =============================================================================

def prepare_time_series_data(
    df: pd.DataFrame,
    ticker: str,
    pca: PCA,
    scaler: StandardScaler
) -> pd.DataFrame:
    """
    Prepare time series data for a stock's movement animation.
    
    Args:
        df: Original DataFrame with time-series data
        ticker: Stock ticker
        pca: Fitted PCA object
        scaler: Fitted StandardScaler
        
    Returns:
        DataFrame with time-series PCA scores
    """
    # Filter for the specific ticker
    stock_df = df[df['ticker'].str.upper() == ticker.upper()].copy()
    
    if stock_df.empty:
        return pd.DataFrame()
    
    # Get date column
    date_col = None
    for col in ['public_date', 'date', 'datadate']:
        if col in stock_df.columns:
            date_col = col
            break
    
    if date_col is None:
        return pd.DataFrame()
    
    # Sort by date
    stock_df = stock_df.sort_values(date_col)
    
    # Get available features
    available_features = [col for col in FEATURE_COLUMNS if col in stock_df.columns]
    
    # Extract features and transform
    X = stock_df[available_features].fillna(method='ffill').fillna(method='bfill')
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'date': stock_df[date_col].values,
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1]
    })
    
    if pca.n_components_ >= 3:
        result_df['PC3'] = X_pca[:, 2]
    
    return result_df


def get_factor_breakdown(
    row: pd.Series,
    feature_columns: List[str] = FEATURE_COLUMNS
) -> Dict[str, Dict[str, float]]:
    """
    Get factor breakdown for a stock organized by category.
    
    Args:
        row: Series with stock data
        feature_columns: List of feature columns
        
    Returns:
        Dictionary organized by factor category
    """
    breakdown = {}
    
    for category, features in FACTOR_CATEGORIES.items():
        breakdown[category] = {}
        for feature in features:
            if feature in row.index:
                breakdown[category][feature] = row[feature]
    
    return breakdown
# ============================================================
# CROWDING SCORE MODULE
# ============================================================

def compute_crowding_scores(pca_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a Factor Crowding Score for each market regime.

    Logic:
    - Concentration Score: % of universe in the single largest cluster (0-100)
    - Dispersion Score: mean pairwise distance between cluster centroids in PC1/PC2 space
      (normalized and inverted so low dispersion = high crowding)
    - Crowding Score = 0.6 * Concentration + 0.4 * (100 - NormalizedDispersion)

    Returns a DataFrame with one row per period with columns:
        period, n_stocks, largest_cluster_pct, centroid_dispersion, crowding_score, risk_level
    """
    import numpy as np
    from itertools import combinations

    period_col = 'period' if 'period' in pca_df.columns else None
    if period_col is None:
        return pd.DataFrame()

    results = []

    # Define canonical period order
    period_order = ['Post-COVID', 'Rate Shock', 'Disinflation']
    periods_present = [p for p in period_order if p in pca_df[period_col].unique()]

    for period in periods_present:
        df_p = pca_df[pca_df[period_col] == period].copy()
        n_stocks = len(df_p)

        if n_stocks == 0 or 'cluster' not in df_p.columns:
            continue

        # --- Concentration Score ---
        cluster_counts = df_p['cluster'].value_counts()
        largest_pct = (cluster_counts.iloc[0] / n_stocks) * 100  # % in biggest cluster

        # --- Centroid Dispersion ---
        centroids = df_p.groupby('cluster')[['PC1', 'PC2']].mean()
        if len(centroids) < 2:
            dispersion = 0.0
        else:
            dists = []
            for c1, c2 in combinations(centroids.index, 2):
                p1 = centroids.loc[c1].values
                p2 = centroids.loc[c2].values
                dists.append(np.sqrt(((p1 - p2) ** 2).sum()))
            dispersion = np.mean(dists)

        results.append({
            'period': period,
            'n_stocks': n_stocks,
            'largest_cluster_pct': round(largest_pct, 1),
            'centroid_dispersion': round(dispersion, 2),
        })

    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(results)

    # Normalize dispersion to 0-100 scale across periods
    d_min = df_results['centroid_dispersion'].min()
    d_max = df_results['centroid_dispersion'].max()
    if d_max > d_min:
        df_results['dispersion_normalized'] = (
            (df_results['centroid_dispersion'] - d_min) / (d_max - d_min) * 100
        )
    else:
        df_results['dispersion_normalized'] = 50.0  # flat if only one period

    # Final Crowding Score: high concentration + low dispersion = high crowding
    df_results['crowding_score'] = (
        0.6 * df_results['largest_cluster_pct'] +
        0.4 * (100 - df_results['dispersion_normalized'])
    ).round(1)

    # Risk Level label
    def risk_label(score):
        if score >= 70:
            return '🔴 High'
        elif score >= 50:
            return '🟡 Elevated'
        else:
            return '🟢 Normal'

    df_results['risk_level'] = df_results['crowding_score'].apply(risk_label)

    return df_results[['period', 'n_stocks', 'largest_cluster_pct',
                        'centroid_dispersion', 'crowding_score', 'risk_level']]
