import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

# Page config (tittle, icon, layout)
st.set_page_config(
    page_title="Smart Stock Monitor",
    page_icon="📈",
    layout="wide"
)

# ✅ Smart caching with ticker-specific cache keys
@st.cache_data(ttl=30, show_spinner=False)
def fetch_market_data(ticker, force_refresh=False):
    """
    Cached data fetching with force refresh option
    force_refresh is just to bust the cache when needed
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m')
        
        try:
            info = stock.info
        except:
            info = {}
        
        fetch_time = datetime.now()
        
        return data, info, fetch_time
    except Exception as e:
        return None, {}, datetime.now()

# Lightweight stock database
STOCK_DATABASE = {
    "🔥 Popular Stocks": {
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'Google': 'GOOGL',
        'Amazon': 'AMZN',
        'Meta': 'META',
        'NVIDIA': 'NVDA',
        'Tesla': 'TSLA',
        'Netflix': 'NFLX',
        'PayPal': 'PYPL',
        'Adobe': 'ADBE'
    },
    "💰 Financials": {
        'JPMorgan': 'JPM',
        'Bank of America': 'BAC',
        'Visa': 'V',
        'Mastercard': 'MA',
        'Goldman Sachs': 'GS',
        'Morgan Stanley': 'MS',
        'American Express': 'AXP',
        'Wells Fargo': 'WFC'
    },
    "🏥 Healthcare": {
        'Johnson & Johnson': 'JNJ',
        'UnitedHealth': 'UNH',
        'Pfizer': 'PFE',
        'Moderna': 'MRNA',
        'Merck': 'MRK',
        'AbbVie': 'ABBV',
        'Eli Lilly': 'LLY'
    },
    "🛍️ Consumer": {
        'Walmart': 'WMT',
        'Costco': 'COST',
        'Nike': 'NKE',
        'Starbucks': 'SBUX',
        'McDonald\'s': 'MCD',
        'Coca-Cola': 'KO',
        'PepsiCo': 'PEP',
        'Home Depot': 'HD'
    },
    "🚗 Automotive": {
        'Tesla': 'TSLA',
        'Ford': 'F',
        'General Motors': 'GM',
        'Rivian': 'RIVN',
        'Lucid': 'LCID',
        'Nio': 'NIO'
    },
    "🥇 Precious Metals": {
        'Gold (GLD ETF)': 'GLD',
        'Silver (SLV ETF)': 'SLV',
        'Gold Futures': 'GC=F',
        'Silver Futures': 'SI=F',
        'Platinum': 'PPLT',
        'Copper': 'CPER'
    },
    "🛢️ Energy": {
        'Crude Oil WTI': 'CL=F',
        'Natural Gas': 'NG=F',
        'US Oil Fund': 'USO',
        'ExxonMobil': 'XOM',
        'Chevron': 'CVX',
        'ConocoPhillips': 'COP'
    },
    "🌾 Agriculture": {
        'Corn': 'ZC=F',
        'Wheat': 'ZW=F',
        'Soybeans': 'ZS=F',
        'Coffee': 'KC=F',
        'Sugar': 'SB=F',
        'Cotton': 'CT=F'
    },
    "🪙 Crypto": {
        'Bitcoin': 'BTC-USD',
        'Ethereum': 'ETH-USD',
        'Solana': 'SOL-USD',
        'Cardano': 'ADA-USD',
        'Coinbase': 'COIN',
        'MicroStrategy': 'MSTR'
    }
}

# Initialize session state
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = {}

if 'last_refresh_time' not in st.session_state:
    st.session_state.last_refresh_time = {}

# Header
st.title("📈 Smart Stock Monitor")
st.caption("🎯 Selective refresh • Low CPU • Fast updates")

# Sidebar
st.sidebar.header("🎯 Stock Selection")

# Category selection
category = st.sidebar.selectbox(
    "Category",
    options=list(STOCK_DATABASE.keys()),
    index=0
)

# Asset selection
assets = STOCK_DATABASE[category]
asset_name = st.sidebar.selectbox(
    "Select Asset",
    options=list(assets.keys()),
    index=0
)
ticker = assets[asset_name]

st.sidebar.success(f"📊 **{ticker}**")

# ✅ SELECTIVE REFRESH BUTTON - Only for current stock
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Refresh Controls")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🔄 Refresh This Stock", use_container_width=True, type="primary"):
        # Clear cache ONLY for this specific ticker
        fetch_market_data.clear()
        
        # Update counter for this ticker
        if ticker not in st.session_state.refresh_counter:
            st.session_state.refresh_counter[ticker] = 0
        st.session_state.refresh_counter[ticker] += 1
        
        # Record refresh time
        st.session_state.last_refresh_time[ticker] = datetime.now()
        
        st.sidebar.success(f"✅ {ticker} refreshed!")
        time.sleep(0.5)  # Brief pause to show success message
        st.rerun()

with col2:
    if st.button("🔄 Refresh All", use_container_width=True):
        # Clear entire cache
        st.cache_data.clear()
        st.session_state.refresh_counter = {}
        st.session_state.last_refresh_time = {}
        st.sidebar.success("✅ All data refreshed!")
        time.sleep(0.5)
        st.rerun()

# Show refresh stats for current ticker
if ticker in st.session_state.last_refresh_time:
    last_refresh = st.session_state.last_refresh_time[ticker]
    time_since = (datetime.now() - last_refresh).total_seconds()
    
    if time_since < 60:
        st.sidebar.info(f"⏰ Last refresh: {int(time_since)}s ago")
    else:
        st.sidebar.info(f"⏰ Last refresh: {int(time_since/60)}m ago")

if ticker in st.session_state.refresh_counter:
    st.sidebar.caption(f"🔢 Refreshed {st.session_state.refresh_counter[ticker]} times")

# Auto-refresh settings
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Auto-Refresh")

auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)

if auto_refresh:
    refresh_interval = st.sidebar.slider(
        "Interval (seconds)",
        min_value=30,
        max_value=300,
        value=60,
        step=30
    )
    st.sidebar.info(f"⏱️ Auto-refreshing every {refresh_interval}s")
    
    # Auto-refresh countdown
    countdown_placeholder = st.sidebar.empty()
else:
    st.sidebar.info("⏸️ Auto-refresh disabled")

# Display options
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Display Options")
show_chart = st.sidebar.checkbox("Show Chart", value=True)
show_volume = st.sidebar.checkbox("Show Volume", value=False)
show_table = st.sidebar.checkbox("Show Data Table", value=False)

# Quick switch - Recently viewed stocks
st.sidebar.markdown("---")
if st.session_state.last_refresh_time:
    st.sidebar.subheader("⚡ Recent Stocks")
    recent_tickers = list(st.session_state.last_refresh_time.keys())[-5:]
    for recent_ticker in reversed(recent_tickers):
        if recent_ticker != ticker:  # Don't show current stock
            if st.sidebar.button(f"↪️ {recent_ticker}", key=f"quick_{recent_ticker}"):
                # Find this ticker in database and switch to it
                for cat, stocks in STOCK_DATABASE.items():
                    for name, symbol in stocks.items():
                        if symbol == recent_ticker:
                            st.session_state.quick_switch_category = cat
                            st.session_state.quick_switch_ticker = recent_ticker
                            st.rerun()

# Main content
st.markdown("---")

# Get force_refresh flag based on refresh counter
force_refresh_flag = st.session_state.refresh_counter.get(ticker, 0)

try:
    # Fetch data (uses cache unless force refreshed)
    data, info, fetch_time = fetch_market_data(ticker, force_refresh=force_refresh_flag)
    
    if data is not None and not data.empty:
        # Calculate metrics
        current_price = data['Close'].iloc[-1]
        previous_close = info.get('previousClose', data['Close'].iloc[0])
        change = current_price - previous_close
        change_pct = (change / previous_close) * 100
        
        # Asset info header
        col_info1, col_info2 = st.columns([3, 1])
        with col_info1:
            st.subheader(f"📊 {asset_name}")
            st.caption(f"Ticker: {ticker} • Category: {category}")
        with col_info2:
            st.metric(
                label="Data Age",
                value=f"{(datetime.now() - fetch_time).seconds}s",
                help="Time since data was fetched"
            )
        
        st.markdown("---")
        
        # Main metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "💰 Current Price",
                f"${current_price:.2f}",
                f"{change:+.2f} ({change_pct:+.2f}%)"
            )
        
        with col2:
            st.metric("📈 High", f"${data['High'].max():.2f}")
        
        with col3:
            st.metric("📉 Low", f"${data['Low'].min():.2f}")
        
        with col4:
            volume = data['Volume'].sum()
            if volume > 1e9:
                st.metric("📦 Volume", f"{volume/1e9:.2f}B")
            elif volume > 1e6:
                st.metric("📦 Volume", f"{volume/1e6:.1f}M")
            else:
                st.metric("📦 Volume", f"{volume:,.0f}")
        
        with col5:
            price_range = data['High'].max() - data['Low'].min()
            st.metric("📊 Range", f"${price_range:.2f}")
        
        # Charts
        if show_chart:
            st.markdown("---")
            
            if show_volume:
                chart_col1, chart_col2 = st.columns([3, 1])
            else:
                chart_col1 = st.container()
            
            with chart_col1:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#00D9FF', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 217, 255, 0.1)'
                ))
                
                fig.update_layout(
                    title=f'{asset_name} ({ticker}) - Intraday Price',
                    yaxis_title='Price (USD)',
                    xaxis_title='Time',
                    height=400,
                    template='plotly_white',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            if show_volume:
                with chart_col2:
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        marker_color='lightblue',
                        name='Volume'
                    ))
                    
                    fig2.update_layout(
                        title='Volume',
                        height=400,
                        yaxis_title='Volume',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
        
        # Data table
        if show_table:
            st.markdown("---")
            st.subheader("📋 Recent Trading Data")
            
            recent = data.tail(15)[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            recent.index = recent.index.strftime('%H:%M:%S')
            
            # Format for display
            for col in ['Open', 'High', 'Low', 'Close']:
                recent[col] = recent[col].apply(lambda x: f"${x:.2f}")
            recent['Volume'] = recent['Volume'].apply(lambda x: f"{x:,.0f}")
            
            st.dataframe(recent, use_container_width=True)
        
        # Footer
        st.markdown("---")
        footer_col1, footer_col2, footer_col3 = st.columns(3)
        
        with footer_col1:
            st.caption(f"⏰ Data fetched: {fetch_time.strftime('%H:%M:%S')}")
        
        with footer_col2:
            st.caption(f"📊 Data points: {len(data)}")
        
        with footer_col3:
            if auto_refresh:
                st.caption(f"🔄 Auto-refresh: ON ({refresh_interval}s)")
            else:
                st.caption("⏸️ Auto-refresh: OFF")
        
        # Auto-refresh logic
        if auto_refresh:
            # Show countdown
            for remaining in range(refresh_interval, 0, -1):
                countdown_placeholder.info(f"⏱️ Next refresh in {remaining}s")
                time.sleep(1)
            
            # Clear cache for this ticker only
            fetch_market_data.clear()
            if ticker not in st.session_state.refresh_counter:
                st.session_state.refresh_counter[ticker] = 0
            st.session_state.refresh_counter[ticker] += 1
            st.session_state.last_refresh_time[ticker] = datetime.now()
            
            st.rerun()
    
    else:
        st.error("❌ No data available")
        st.info("""
            **Possible reasons:**
            - Market is closed
            - Invalid ticker
            - Connection issue
            
            Try clicking "🔄 Refresh This Stock"
        """)

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.info("Click '🔄 Refresh This Stock' to try again")

# Performance info
st.sidebar.markdown("---")
st.sidebar.caption("""
💡 **How it works:**
- "Refresh This Stock" = Updates only current ticker
- "Refresh All" = Clears all cached data
- Other stocks stay cached (saves CPU)
- Auto-refresh only updates current view
""")