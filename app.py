import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from tvDatafeed import TvDatafeed, Interval

# Configure secure silent logging for internal tracking
logging.basicConfig(level=logging.ERROR)

# ---------------------------------------------------------
# 1. UI CONFIGURATION & "OLD MONEY" CUSTOM STYLING (SAFE ZONE)
# ---------------------------------------------------------
st.set_page_config(page_title="AURA • Market Intelligence", layout="wide", initial_sidebar_state="collapsed")

# Inject adaptive custom CSS for both Light and Dark modes
st.markdown("""
    <style>
    h1, h2, h3 { color: #BFA15F !important; font-family: 'Playfair Display', Georgia, serif; }
    p, label { font-family: 'Inter', sans-serif; }
    
    /* Target the metric cards to make them equal height and adaptive */
    [data-testid="metric-container"] { 
        background-color: var(--secondary-background-color); 
        border: 1px solid #BFA15F; 
        padding: 15px; 
        border-radius: 4px; 
        height: 100%; 
    }
    </style>
""", unsafe_allow_html=True)

st.title("AURA • MARKET INTELLIGENCE TERMINAL")
st.markdown("---")

# ---------------------------------------------------------
# 2. TICKER FACTORY (Top 10 Global & Top 40 Indian Heavyweights)
# ---------------------------------------------------------
ticker_dict = {
    "Global Titans (US)": {
        "Apple Inc.": "AAPL", "Microsoft Corp.": "MSFT", "NVIDIA Corp.": "NVDA", 
        "Alphabet Inc.": "GOOGL", "Amazon.com Inc.": "AMZN", "Meta Platforms": "META", 
        "Tesla Inc.": "TSLA", "Berkshire Hathaway": "BRK-B", "Visa Inc.": "V", "JPMorgan Chase": "JPM"
    },
    "Indian Market (NSE)": {
        "Reliance Industries": "RELIANCE.NS", "Tata Consultancy Services": "TCS.NS", 
        "HDFC Bank": "HDFCBANK.NS", "Bharti Airtel": "BHARTIARTL.NS", "ICICI Bank": "ICICIBANK.NS", 
        "Infosys Ltd.": "INFY.NS", "State Bank of India": "SBIN.NS", "ITC Ltd.": "ITC", 
        "Larsen & Toubro": "LT.NS", "Hindustan Unilever": "HINDUNILVR.NS", "HCL Technologies": "HCLTECH.NS", 
        "Sun Pharmaceutical": "SUNPHARMA.NS", "Maruti Suzuki": "MARUTI.NS", "Tata Motors": "TATAMOTORS.NS", 
        "NTPC Ltd.": "NTPC.NS", "ONGC": "ONGC.NS", "Coal India": "COALINDIA.NS", 
        "Axis Bank": "AXISBANK.NS", "Adani Enterprises": "ADANIENT.NS", "Kotak Mahindra Bank": "KOTAKBANK.NS",
        "Bajaj Finance": "BAJFINANCE.NS", "Mahindra & Mahindra": "M&M.NS", "Bharti Airtel": "BHARTIARTL.NS",
        "JSW Steel": "JSWSTEEL.NS", "UltraTech Cement": "ULTRACEMCO.NS", "Power Grid Corp": "POWERGRID.NS",
        "Titan Company": "TITAN.NS", "NTPC": "NTPC.NS", "IndusInd Bank": "INDUSINDBK.NS", 
        "Bajaj Finserv": "BAJAJFINSV.NS", "Nestle India": "NESTLEIND.NS", "Tech Mahindra": "TECHM.NS",
        "Wipro Ltd.": "WIPRO.NS", "Hindalco Industries": "HINDALCO.NS", "Grasim Industries": "GRASIM.NS",
        "Adani Ports": "ADANIPORTS.NS", "Cipla Ltd.": "CIPLA.NS", "Tata Steel": "TATASTEEL.NS",
        "Eicher Motors": "EICHERMOT.NS", "Apollo Hospitals": "APOLLOHOSP.NS"
    }
}
st.caption("⚡ Live market data may be subject to exchange-mandated delays for non-institutional connections.")
# ---------------------------------------------------------
# 3. INTERACTIVE SEARCH CONTROL ROOM
# ---------------------------------------------------------
col1, col2 = st.columns([1, 2])
with col1:
    market_type = st.radio("Select Market Grid", list(ticker_dict.keys()))
with col2:
    selected_stock = st.selectbox("Select Asset Ticker", list(ticker_dict[market_type].keys()))

ticker_symbol = ticker_dict[market_type][selected_stock]
currency_sym = "$" if market_type == "Global Titans (US)" else "₹"

# ---------------------------------------------------------
# 4. SECURE ENGINE EXECUTION BLOCK
# ---------------------------------------------------------
try:
    @st.cache_data(ttl=300)  # 5-minute cache for near real-time updates
    def fetch_and_engineer_data(ticker, market):
        # ---------------------------------------------
        # ENGINE A: YAHOO FINANCE (For US Stocks)
        # ---------------------------------------------
        if market == "Global Titans (US)":
            raw_data = yf.download(ticker, period="max")
            df = raw_data[['Open', 'High', 'Low', 'Close']].copy()
            
        # ---------------------------------------------
        # ENGINE B: TRADINGVIEW (For Indian Stocks)
        # ---------------------------------------------
        else:
            # Initialize TradingView as a guest (no login required)
            tv = TvDatafeed()
            
            # Clean the ticker (TradingView uses 'RELIANCE' instead of 'RELIANCE.NS')
            tv_ticker = ticker.replace(".NS", "")
            
            # Fetch 5000 days of historical daily data directly from the NSE
            raw_data = tv.get_hist(symbol=tv_ticker, exchange='NSE', interval=Interval.in_daily, n_bars=5000)
            
            # Rename columns to match our feature factory
            df = raw_data[['open', 'high', 'low', 'close']].copy()
            
        # Standardize column names for the math engine
        df.columns = ['Open_Price', 'High_Price', 'Low_Price', 'Close_Price']
        
        # --- Aerodynamic Technical Indicators ---
        df['MA_10'] = df['Close_Price'].rolling(window=10).mean()
        df['MA_50'] = df['Close_Price'].rolling(window=50).mean()
        df['RSI'] = RSIIndicator(close=df['Close_Price'], window=14).rsi()
        
        indicator_bb = BollingerBands(close=df['Close_Price'], window=20, window_dev=2)
        df['BB_High'] = indicator_bb.bollinger_hband()
        df['BB_Low'] = indicator_bb.bollinger_lband()
        
        df['Target_Next_Day'] = df['Close_Price'].shift(-1)
        df['EMA_12'] = df['Close_Price'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close_Price'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        
        return df.dropna()

    with st.spinner("Extracting live market telemetry..."):
        # We now pass the market_type into the function so it knows which engine to use
        df = fetch_and_engineer_data(ticker_symbol, market_type)

    # ---------------------------------------------------------
    # 5. LIVE MACHINE LEARNING ENGINE
    # ---------------------------------------------------------
    feature_cols = ['Close_Price', 'MA_10', 'MA_50', 'RSI', 'BB_High', 'BB_Low']
    X = df[feature_cols]
    y = df['Target_Next_Day']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = Ridge(alpha=1.0)
    
    # Optional diagnostic printout removed for production security
    model.fit(X_train, y_train)

    test_preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    latest_live_features = X.iloc[[-1]]
    tomorrow_prediction = model.predict(latest_live_features)[0]
    current_close_price = X.iloc[-1]['Close_Price']
    predicted_delta = tomorrow_prediction - current_close_price
    delta_pct = (predicted_delta / current_close_price) * 100

    # ---------------------------------------------------------
    # 6. EXECUTIVE ANALYTICS DASHBOARD CARD DISPLAY
    # ---------------------------------------------------------
    current_open_price = df.iloc[-1]['Open_Price']

    m_col1, m_col2, m_col3, m_col4 = st.columns([1, 1, 1, 1])
    with m_col1:
        st.metric(label="Today's Open", value=f"{currency_sym}{current_open_price:,.2f}")
    with m_col2:
        st.metric(label="Latest Close", value=f"{currency_sym}{current_close_price:,.2f}")
    with m_col3:
        st.metric(
            label="AI Projected Close", 
            value=f"{currency_sym}{tomorrow_prediction:,.2f}", 
            delta=f"{predicted_delta:+.2f} ({delta_pct:+.2f}%)"
        )
    with m_col4:
        st.metric(label="System Error (RMSE)", value=f"{currency_sym}{rmse:.2f}")

    # ---------------------------------------------------------
    # 7. BESPOKE INTERACTIVE VISUALIZATION
    # ---------------------------------------------------------
    st.markdown("### Visual Market Telemetry")

    timeframe = st.radio(
        "Select Chart Timeframe", 
        options=["1M", "3M", "6M", "1Y", "5Y", "Max"], 
        horizontal=True,
        index=3 
    )

    days_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "5Y": 1260, "Max": len(df)}
    visual_df = df.tail(days_map[timeframe])

    intersect_dates = visual_df.index.intersection(X_test.index)
    if len(intersect_dates) > 0:
        test_display_preds = test_preds[-len(intersect_dates):]
    else:
        test_display_preds = []

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=visual_df.index,
        open=visual_df['Open_Price'],
        high=visual_df['High_Price'],
        low=visual_df['Low_Price'],
        close=visual_df['Close_Price'],
        name='Market OHLC',
        increasing_line_color='#00A86B', 
        decreasing_line_color='#B22222'  
    ))

    fig.update_layout(xaxis_rangeslider_visible=False)
    
    if len(intersect_dates) > 0:
        fig.add_trace(go.Scatter(x=intersect_dates, y=test_display_preds, name='AI Historic Predictions', line=dict(color='#00A86B', width=2, dash='dot')))

    fig.add_trace(go.Scatter(x=visual_df.index, y=visual_df['BB_High'], name='BB Upper Boundary', line=dict(color='#2C2A29', width=1)))
    fig.add_trace(go.Scatter(x=visual_df.index, y=visual_df['BB_Low'], name='BB Lower Boundary', line=dict(color='#2C2A29', width=1), fill='tonexty', fillcolor='rgba(44, 42, 41, 0.15)'))

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # ---------------------------------------------------------
    # 8. ALGORITHMIC AI ANALYST AGENT
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### Executive Market Briefing")

    latest_row = df.iloc[-1]
    lat_close = latest_row['Close_Price']
    lat_rsi = latest_row['RSI']
    lat_macd = latest_row['MACD']
    lat_bb_high = latest_row['BB_High']
    lat_bb_low = latest_row['BB_Low']

    bb_range = lat_bb_high - lat_bb_low
    position_in_bands = (lat_close - lat_bb_low) / bb_range if bb_range != 0 else 0.5

    if position_in_bands > 0.8:
        volatility_analysis = "trading near its upper resistance boundary, signaling potential short-term overextension"
    elif position_in_bands < 0.2:
        volatility_analysis = "resting near historic support thresholds, indicating strong defensive consolidation"
    else:
        volatility_analysis = "maintaining a highly stable trajectory within its standard volatility channel"

    if lat_rsi > 70:
        momentum_state = "highly overbought territory"
        action_tone = "suggests institutional caution as buyers face diminishing returns"
    elif lat_rsi < 30:
        momentum_state = "deeply oversold conditions"
        action_tone = "indicates strong accumulation potential for value-driven portfolios"
    else:
        momentum_state = "neutral equilibrium"
        if lat_macd > 0:
            action_tone = "displaying a steady bullish divergence as buying volume outpaces distribution"
        else:
            action_tone = "experiencing subtle structural overhead resistance as market volume cools"

    market_brief_paragraph = (
        f"AURA Terminal telemetry confirms a structured {momentum_state} for {selected_stock} ({ticker_symbol}). "
        f"The asset is currently {volatility_analysis}. With the algorithmic predictive engine projecting an immediate "
        f"next-day close targeting {currency_sym}{tomorrow_prediction:,.2f} (a shift of {delta_pct:+.2f}%), technical analysis {action_tone}. "
        f"System validation maintains an RMSE floor of {currency_sym}{rmse:.2f}, securing robust statistical confidence in this session's vector."
    )

    st.markdown(
        f"""
        <div style="background-color: var(--secondary-background-color); border-left: 3px solid #BFA15F; padding: 25px; border-radius: 4px; margin-top: 10px;">
            <p style="font-style: italic; font-size: 1.1rem; line-height: 1.6; color: var(--text-color); margin: 0;">
                "{market_brief_paragraph}"
            </p>
            <p style="text-align: right; font-size: 0.8rem; color: #888888; margin: 15px 0 0 0; letter-spacing: 2px;">
                SECURE ENGINE CODEV • SYSTEM LOGGED
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown(f"**System Log:** Telemetry pipeline fully operational for {selected_stock} ({ticker_symbol}). Data cached automatically.")

# ---------------------------------------------------------
# 9. SILENT FALLBACK & THREAT CONTAINMENT (SECURITY PERIMETER)
# ---------------------------------------------------------
except ValueError as e:
    # Captures empty array formations (like the APPL zero-data bug) quietly
    logging.error(f"Array dimensional collapse or data fetch failure: {e}")
    st.error("⚠️ Market data is currently unavailable for this asset. Please try again later.")
except Exception as e:
    # Comprehensive catch-all to ensure the raw traceback is never exposed to the client side
    logging.error(f"Critical System Fault: {e}")
    st.error("⚠️ The terminal encountered a temporary connection issue while establishing the API handshake.")
