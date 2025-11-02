import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="AI Trading Strategy",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="collapsed"
)

# Custom CSS for amazing UI
st.markdown("""
<style>
    /* Hide sidebar by default */
    [data-testid="collapsedControl"] {
        display: none;
    }
    
    /* Main container */
    .main {
        padding: 0rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main > div {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin-top: 1rem;
    }
    
    /* Hero section */
    .hero {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .hero h1 {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        color: white !important;
    }
    
    .hero p {
        font-size: 1.3rem;
        margin-top: 1rem;
        opacity: 0.95;
    }
    
    /* Control panel */
    .control-panel {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stMetric label {
        color: white !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        border: none;
        padding: 1rem 3rem;
        border-radius: 50px;
        font-size: 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Section headers */
    h2 {
        color: #667eea;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 3rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .info-box h3 {
        margin-top: 0;
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class='hero'>
    <h1>🤖 AI-Powered Trading Strategy</h1>
    <p>Machine Learning Meets Quantitative Finance</p>
</div>
""", unsafe_allow_html=True)

# Predefined stock lists (no Wikipedia needed!)
POPULAR_STOCKS = {
    "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX"],
    "Blue Chips": ["JPM", "JNJ", "PG", "WMT", "V", "UNH", "HD", "DIS"],
    "Growth Stocks": ["TSLA", "NVDA", "AMD", "SHOP", "SQ", "ROKU", "PLTR", "COIN"],
    "S&P 500 Sample": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", 
                       "UNH", "JNJ", "V", "WMT", "JPM", "PG", "MA", "HD", "CVX", "MRK", 
                       "ABBV", "KO", "PEP", "COST", "AVGO", "LLY", "TMO", "MCD", "CSCO",
                       "ACN", "ABT", "ADBE", "DHR", "NKE", "CRM", "TXN", "PM", "NEE",
                       "ORCL", "WFC", "VZ", "BMY", "UPS", "MS", "RTX", "HON", "QCOM",
                       "INTU", "LOW", "AMGN", "T", "IBM", "CAT"]
}

# Control Panel in main area
st.markdown("## ⚙️ Configure Your Strategy")

with st.container():
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        stock_universe = st.selectbox(
            "📊 Stock Universe",
            list(POPULAR_STOCKS.keys()),
            index=3
        )
    
    with col2:
        n_clusters = st.slider("🎯 Number of Clusters", 3, 10, 5)
    
    with col3:
        top_n_stocks = st.slider("💼 Portfolio Size", 5, 30, 15)
    
    with col4:
        lookback_period = st.slider("📅 Lookback (days)", 100, 500, 252)

# Date range
col1, col2, col3 = st.columns([2, 2, 3])
with col1:
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
with col2:
    end_date = st.date_input("End Date", datetime(2024, 1, 1))
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    run_backtest = st.button("🚀 RUN BACKTEST NOW")

# Advanced settings in expander
with st.expander("⚙️ Advanced Settings"):
    col1, col2 = st.columns(2)
    with col1:
        rebalance_freq = st.slider("Rebalance Frequency (days)", 30, 180, 90)
    with col2:
        min_data_threshold = st.slider("Min Data Completeness (%)", 50, 100, 80)

if run_backtest:
    st.session_state.run = True

@st.cache_data(show_spinner=False)
def download_stock_data(tickers, start, end):
    """Download stock data"""
    data = yf.download(tickers, start=start, end=end, progress=False)['Adj Close']
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    data = data.dropna(axis=1, thresh=len(data)*0.8)
    return data

def calculate_features(data):
    """Calculate technical features for each stock"""
    features_dict = {}
    
    for ticker in data.columns:
        prices = data[ticker].dropna()
        
        if len(prices) < 100:
            continue
            
        returns = prices.pct_change()
        
        features_dict[ticker] = {
            'mean_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
            'momentum_1m': prices.pct_change(21).iloc[-1] if len(prices) > 21 else 0,
            'momentum_3m': prices.pct_change(63).iloc[-1] if len(prices) > 63 else 0,
            'momentum_6m': prices.pct_change(126).iloc[-1] if len(prices) > 126 else 0,
            'momentum_12m': prices.pct_change(252).iloc[-1] if len(prices) > 252 else 0,
            'rsi': calculate_rsi(prices),
            'max_drawdown': calculate_max_drawdown(prices)
        }
    
    return pd.DataFrame(features_dict).T

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if len(rsi) > 0 else 50

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def perform_clustering(features, n_clusters):
    """Perform K-Means clustering"""
    features_clean = features.dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_clean)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    features_clean['cluster'] = clusters
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return features_clean, X_pca, kmeans, scaler, pca

def calculate_portfolio_performance(data, selected_stocks):
    """Calculate portfolio returns"""
    portfolio_data = data[selected_stocks].dropna()
    portfolio_returns = portfolio_data.pct_change().mean(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    return portfolio_returns, cumulative_returns

# Main execution
if 'run' in st.session_state and st.session_state.run:
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get tickers
        tickers = POPULAR_STOCKS[stock_universe]
        
        status_text.markdown("### 📊 Downloading stock data...")
        progress_bar.progress(20)
        
        data = download_stock_data(tickers, start_date, end_date)
        
        if data is None or len(data) < 50:
            st.error("❌ Insufficient data. Please adjust date range.")
            st.stop()
        
        status_text.markdown(f"### ✅ Downloaded data for {len(data.columns)} stocks")
        progress_bar.progress(40)
        
        status_text.markdown("### 🔧 Calculating technical features...")
        features = calculate_features(data)
        
        status_text.markdown(f"### ✅ Calculated features for {len(features)} stocks")
        progress_bar.progress(60)
        
        status_text.markdown("### 🤖 Performing K-Means clustering...")
        features_clustered, X_pca, kmeans, scaler, pca = perform_clustering(features, n_clusters)
        
        status_text.markdown(f"### ✅ Clustered stocks into {n_clusters} groups")
        progress_bar.progress(80)
        
        # Select best cluster
        cluster_stats = features_clustered.groupby('cluster').agg({
            'mean_return': 'mean',
            'volatility': 'mean',
            'sharpe_ratio': 'mean',
            'momentum_12m': 'mean'
        })
        cluster_stats['count'] = features_clustered.groupby('cluster').size()
        
        best_cluster = cluster_stats['sharpe_ratio'].idxmax()
        cluster_stocks = features_clustered[features_clustered['cluster'] == best_cluster]
        top_stocks = cluster_stocks.nlargest(top_n_stocks, 'sharpe_ratio').index.tolist()
        
        status_text.markdown("### 💰 Calculating portfolio performance...")
        portfolio_returns, cumulative_returns = calculate_portfolio_performance(data, top_stocks)
        
        # Download benchmark
        sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)['Adj Close']
        sp500_returns = sp500.pct_change()
        sp500_cumulative = (1 + sp500_returns).cumprod()
        
        # Align dates
        common_dates = portfolio_returns.index.intersection(sp500_returns.index)
        portfolio_returns = portfolio_returns.loc[common_dates]
        sp500_returns = sp500_returns.loc[common_dates]
        cumulative_returns = cumulative_returns.loc[common_dates]
        sp500_cumulative = sp500_cumulative.loc[common_dates]
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        st.success("🎉 Analysis Complete!")
        
        # RESULTS
        st.markdown("## 📊 Performance Overview")
        
        # Calculate metrics
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        sp500_total_return = (sp500_cumulative.iloc[-1] - 1) * 100
        
        ann_return = portfolio_returns.mean() * 252 * 100
        ann_vol = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
        
        sp500_ann_return = sp500_returns.mean() * 252 * 100
        sp500_ann_vol = sp500_returns.std() * np.sqrt(252) * 100
        sp500_sharpe = (sp500_returns.mean() / sp500_returns.std()) * np.sqrt(252)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{total_return:.2f}%",
                delta=f"{total_return - sp500_total_return:.2f}% vs S&P500"
            )
        
        with col2:
            st.metric(
                "Annual Return",
                f"{ann_return:.2f}%",
                delta=f"{ann_return - sp500_ann_return:.2f}% vs S&P500"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.3f}",
                delta=f"{sharpe - sp500_sharpe:.3f} vs S&P500"
            )
        
        with col4:
            st.metric(
                "Volatility",
                f"{ann_vol:.2f}%",
                delta=f"{ann_vol - sp500_ann_vol:.2f}% vs S&P500",
                delta_color="inverse"
            )
        
        # Performance Chart
        st.markdown("## 💹 Cumulative Returns")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=(cumulative_returns - 1) * 100,
            name='AI Strategy',
            line=dict(color='#667eea', width=3),
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=sp500_cumulative.index,
            y=(sp500_cumulative - 1) * 100,
            name='S&P 500',
            line=dict(color='#764ba2', width=3, dash='dash')
        ))
        
        fig.update_layout(
            height=500,
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Cluster Statistics")
            cluster_stats_display = cluster_stats.round(4)
            cluster_stats_display.columns = ['Avg Return', 'Avg Vol', 'Avg Sharpe', 'Momentum', 'Count']
            st.dataframe(
                cluster_stats_display.style.background_gradient(cmap='RdYlGn', subset=['Avg Sharpe']),
                use_container_width=True
            )
        
        with col2:
            st.markdown("### 🎯 PCA Visualization")
            
            fig = go.Figure()
            
            for cluster in range(n_clusters):
                mask = features_clustered['cluster'] == cluster
                fig.add_trace(go.Scatter(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    mode='markers',
                    name=f'Cluster {cluster}',
                    marker=dict(size=12, opacity=0.7)
                ))
            
            fig.update_layout(
                height=400,
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Selected Stocks
        st.markdown(f"### 📋 Selected Portfolio ({len(top_stocks)} stocks from Cluster {best_cluster})")
        
        selected_df = cluster_stocks.loc[top_stocks][['mean_return', 'volatility', 'sharpe_ratio', 'momentum_12m']]
        selected_df.columns = ['Annual Return', 'Volatility', 'Sharpe', '12M Momentum']
        selected_df = (selected_df * 100).round(2)
        
        st.dataframe(
            selected_df.style.background_gradient(cmap='RdYlGn', subset=['Sharpe']),
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("See error details"):
            st.exception(e)

else:
    # Landing page
    st.markdown("""
    <div class='info-box'>
        <h3>🚀 Ready to Get Started?</h3>
        <p style='font-size: 1.1rem; margin: 0;'>
            Configure your strategy above and click <b>"RUN BACKTEST NOW"</b> to see the magic happen!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 Machine Learning
        Uses K-Means clustering to group stocks by similar characteristics
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Data-Driven
        Analyzes 9 technical indicators including momentum and RSI
        """)
    
    with col3:
        st.markdown("""
        ### 💹 Proven Strategy
        Systematic selection based on risk-adjusted returns
        """)
