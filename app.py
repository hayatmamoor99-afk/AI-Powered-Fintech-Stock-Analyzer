import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import datetime
# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Apex Wealth Analytics | Advanced Asset Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- PREMIUM LIGHT THEME CSS INJECTION ---
st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        /* Apply fonts globally */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
        }
        
        /* Main container styling */
        .main {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        }
        
        /* Header bar styling */
        .header-container {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(12px);
            padding: 1.5rem 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
            border: 1px solid rgba(226, 232, 240, 0.8);
            margin-bottom: 2rem;
        }
        
        .header-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }
        
        .header-subtitle {
            font-size: 1rem;
            color: #64748b;
            margin-top: 0.3rem;
            font-weight: 500;
        }
        /* Metric card styling */
        .metric-card {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02);
            border: 1px solid #e2e8f0;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
            border-color: #cbd5e1;
        }
        
        .metric-title {
            font-size: 0.85rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #0f172a;
        }
        
        .metric-delta {
            font-size: 0.95rem;
            font-weight: 600;
            margin-top: 0.4rem;
        }
        
        .delta-positive {
            color: #10b981;
        }
        
        .delta-negative {
            color: #ef4444;
        }
        
        /* Decision panel styling */
        .decision-box {
            border-radius: 20px;
            padding: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.4);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
            margin-bottom: 2rem;
        }
        
        .decision-buy {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 1px solid #bbf7d0;
        }
        
        .decision-sell {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border: 1px solid #fecaca;
        }
        
        .decision-hold {
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border: 1px solid #fde68a;
        }
        
        /* Custom side bar styling */
        .sidebar .sidebar-content {
            background-color: #ffffff;
        }
        
        /* Theory Guide Styles */
        .theory-section {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 1.25rem;
            border-left: 4px solid #3b82f6;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.01);
        }
        
        .theory-header {
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem 0;
            color: #94a3b8;
            font-size: 0.85rem;
            border-top: 1px solid #e2e8f0;
            margin-top: 3rem;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)
# --- HELPER FUNCTIONS FOR CALCULATIONS ---
@st.cache_data(ttl=3600)
def fetch_ticker_data(ticker_symbol):
    """Fetches up to 6 years of historical daily data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Fetching 6 years to ensure we have a solid 5 years of historical base for returns
        df = ticker.history(period="6y")
        if df.empty:
            return None, None
        
        # Resolve company/asset name
        info = ticker.info
        asset_name = info.get('longName') or info.get('shortName') or ticker_symbol
        return df, asset_name
    except Exception as e:
        return None, None
def calculate_returns_distribution(df, years):
    """Filters data for the last N years and computes daily returns."""
    end_date = df.index[-1]
    start_date = end_date - pd.DateOffset(years=years)
    df_filtered = df.loc[start_date:end_date].copy()
    
    # Calculate daily percentage returns
    df_filtered['Daily_Return'] = df_filtered['Close'].pct_change()
    return df_filtered.dropna(subset=['Daily_Return'])
def calculate_investment_simulation(df, years, principal=100.0):
    """Simulates investing $100 exactly N years ago."""
    current_price = df['Close'].iloc[-1]
    end_date = df.index[-1]
    target_date = end_date - pd.DateOffset(years=years)
    
    # Find the closest trading day to the target date
    indexer = df.index.get_indexer([target_date], method='nearest')[0]
    historical_price = df['Close'].iloc[indexer]
    historical_date = df.index[indexer].date()
    
    multiplier = current_price / historical_price
    resulting_val = principal * multiplier
    net_gain = resulting_val - principal
    pct_gain = (multiplier - 1) * 100
    
    return {
        "historical_date": historical_date,
        "historical_price": historical_price,
        "current_price": current_price,
        "resulting_value": resulting_val,
        "net_gain": net_gain,
        "pct_gain": pct_gain
    }
def compute_rsi(df, period=14):
    """Calculates Relative Strength Index."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]
def make_recommendation(df_1y, df_full):
    """Computes recommendation using probability principles & technical metrics."""
    # 1. Historical Daily Returns probability (using 1 year data)
    returns = df_1y['Daily_Return'] * 100 # percentage returns
    mu, std = norm.fit(returns)
    
    # Probability of a positive daily return: P(R > 0)
    # Using the normal survival function (1 - CDF)
    prob_positive_daily = norm.sf(0, loc=mu, scale=std)
    
    # Empirical probability of positive returns
    empirical_prob = (returns > 0).mean()
    
    # Combined probability metric
    prob_metric = (prob_positive_daily + empirical_prob) / 2.0
    
    # 2. Moving Averages (Trend)
    df_full['SMA_50'] = df_full['Close'].rolling(window=50).mean()
    df_full['SMA_200'] = df_full['Close'].rolling(window=200).mean()
    
    current_price = df_full['Close'].iloc[-1]
    sma_50 = df_full['SMA_50'].iloc[-1]
    sma_200 = df_full['SMA_200'].iloc[-1]
    
    trend_bullish = current_price > sma_50 and sma_50 > sma_200
    trend_bearish = current_price < sma_50 and sma_50 < sma_200
    
    # 3. Momentum (RSI)
    rsi_val = compute_rsi(df_full)
    
    # 4. Multi-factor Scoring (0 to 100)
    # Probability component (max 45 pts)
    # Scale a probability range of [0.45, 0.55] to [0, 45] points
    prob_score = min(45, max(0, (prob_metric - 0.45) * 450))
    
    # Trend component (max 30 pts)
    trend_score = 0
    if current_price > sma_50:
        trend_score += 15
    if sma_50 > sma_200:
        trend_score += 15
        
    # RSI component (max 25 pts)
    rsi_score = 0
    if rsi_val < 30: # Oversold (Bullish Buy Opportunity)
        rsi_score = 25
    elif rsi_val >= 30 and rsi_val < 50: # Moderate Bullish
        rsi_score = 20
    elif rsi_val >= 50 and rsi_val < 70: # Moderate Bearish
        rsi_score = 10
    else: # Overbought (Bearish Sell Opportunity)
        rsi_score = 0
        
    total_score = prob_score + trend_score + rsi_score
    
    # Determine Signal
    if total_score >= 65:
        signal = "BUY"
        color_class = "decision-buy"
        bg_color = "#dcfce7"
        text_color = "#15803d"
        desc = "Strong bullish setup driven by strong historical daily win rate, solid upward trend momentum, and favorable relative strength."
    elif total_score >= 48:
        signal = "BUY (MODERATE)"
        color_class = "decision-buy"
        bg_color = "#f0fdf4"
        text_color = "#166534"
        desc = "Moderate buy signal. The asset shows supportive technical foundations but lacks high probabilistic conviction for aggressive entries."
    elif total_score >= 35:
        signal = "DON'T BUY (HOLD)"
        color_class = "decision-hold"
        bg_color = "#fffbeb"
        text_color = "#b45309"
        desc = "Neutral/Hold. Current momentum indicators are mixed. Recommend waiting for a clearer trend breakout or more favorable statistical odds."
    else:
        signal = "DON'T BUY (SELL / AVOID)"
        color_class = "decision-sell"
        bg_color = "#fef2f2"
        text_color = "#b91c1c"
        desc = "Avoid or sell. Technical trend is downward (bearish), momentum is overstretched or weak, and historical return probabilities favor continued losses."
        
    return {
        "score": total_score,
        "signal": signal,
        "color_class": color_class,
        "bg_color": bg_color,
        "text_color": text_color,
        "desc": desc,
        "prob_metric": prob_metric,
        "rsi": rsi_val,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "current_price": current_price
    }
# --- HEADER SECTION ---
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Apex Wealth Analytics</h1>
        <div class="header-subtitle">Advanced Quantitative Asset Intelligence & Risk Modeling</div>
    </div>
""", unsafe_allow_html=True)
# --- SIDEBAR (Educational Corner & Inputs) ---
with st.sidebar:
    st.markdown("### 🔍 Asset Selector")
    
    # Suggestions helper
    popular_symbols = {
        "Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "BRK-B", "JPM", "V"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD"]
    }
    
    symbol_type = st.radio("Asset Type", ["Stock Ticker", "Crypto / Coin Symbol"])
    
    default_ticker = "AAPL" if symbol_type == "Stock Ticker" else "BTC-USD"
    ticker_input = st.text_input(
        "Enter yfinance Ticker Symbol",
        value=default_ticker,
        help="E.g., AAPL for Apple Inc., BTC-USD for Bitcoin."
    ).strip().upper()
    
    st.markdown("💡 **Popular Suggestions:**")
    st.caption(", ".join(popular_symbols["Stocks" if symbol_type == "Stock Ticker" else "Crypto"]))
    
    st.markdown("---")
    
    # THEORETICAL CORNER
    st.markdown("### 🎓 Theoretical Corner")
    st.markdown("Welcome, Newbie! Here is a simple breakdown of the advanced quantitative principles used in this application:")
    
    with st.expander("📊 1. Return Histograms", expanded=True):
        st.markdown("""
        **What it is:** A histogram splits the historical daily returns into range bins and counts how many times returns fell into each range.
        
        **Why it matters:** It visualizes **volatility** and **symmetry**. A wider histogram means high volatility (riskier). If it has a longer right tail, the asset has a positive skew (tends to have large gains).
        
        **Color Coding:** 
        - <span style="color:#10b981; font-weight:bold;">Green bars</span> represent profitable daily returns (> 0%).
        - <span style="color:#ef4444; font-weight:bold;">Red bars</span> represent loss days (< 0%).
        """, unsafe_allow_html=True)
        
    with st.expander("📈 2. Fitted Probability Curve", expanded=False):
        st.markdown("""
        **What it is:** The dashed line is a **Probability Density Function (PDF)** of a fitted Normal Distribution.
        
        **Why it matters:** It shows the statistical probability distribution of daily returns. By calculating the area under the curve to the right of 0%, we find the mathematical probability of a positive return day.
        """)
        
    with st.expander("💵 3. Investment Simulation", expanded=False):
        st.markdown("""
        **What it is:** A simulation of investing a fixed $100 on the exact trading day 1, 2, 3, 4, or 5 years ago.
        
        **Why it matters:** It demonstrates the real-world historical compound growth (or loss) of the asset compared to holding cash.
        """)
    with st.expander("🎯 4. Decision Engine", expanded=False):
        st.markdown("""
        **What it is:** A probability-weighted recommendations matrix using:
        
        1. **Win-rate Probability**: Historical odds of positive return days.
        2. **RSI (Relative Strength Index)**: Identifies if the asset is overbought (>70, risky) or oversold (<30, bargain).
        3. **SMAs (Simple Moving Averages)**: Tells us if the long-term trend is pointing up (50-day average > 200-day average).
        """)
# --- FETCH DATA ---
if ticker_input:
    with st.spinner(f"Loading data for {ticker_input} from Yahoo Finance..."):
        df, asset_name = fetch_ticker_data(ticker_input)
        
    if df is not None and len(df) > 100:
        # Get historical returns for each period
        periods = {
            "5 Years": calculate_returns_distribution(df, 5),
            "4 Years": calculate_returns_distribution(df, 4),
            "3 Years": calculate_returns_distribution(df, 3),
            "2 Years": calculate_returns_distribution(df, 2),
            "1 Year": calculate_returns_distribution(df, 1),
        }
        
        # Current asset summary metrics
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_diff = current_price - prev_price
        price_pct = (price_diff / prev_price) * 100
        
        # --- TOP LEVEL DASHBOARD ROW ---
        col_title, col_metric = st.columns([2, 1])
        with col_title:
            st.markdown(f"## {asset_name} (`{ticker_input}`)")
            st.caption(f"Latest Market Quote: {df.index[-1].strftime('%B %d, %Y')} | Data provided via yfinance API")
        with col_metric:
            delta_class = "delta-positive" if price_diff >= 0 else "delta-negative"
            delta_sign = "+" if price_diff >= 0 else ""
            st.markdown(f"""
                <div class="metric-card" style="padding: 1rem 1.5rem; text-align: right;">
                    <div class="metric-title">Live Closing Price</div>
                    <div class="metric-value">${current_price:,.2f}</div>
                    <div class="metric-delta {delta_class}">{delta_sign}${price_diff:,.2f} ({delta_sign}{price_pct:.2f}%) Today</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.write(" ")
        
        # --- RECOMMENDATION BLOCK ---
        st.markdown("### 🤖 Advanced Probabilistic Decision Matrix")
        rec = make_recommendation(periods["1 Year"], df)
        
        decision_html = f"""
            <div class="decision-box {rec['color_class']}">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span style="font-size: 0.85rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">AI Quantitative Signal</span>
                        <div style="font-size: 2.2rem; font-weight: 800; color: {rec['text_color']}; margin-top: 0.2rem;">{rec['signal']}</div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.85rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Quantitative Score</span>
                        <div style="font-size: 2.2rem; font-weight: 800; color: {rec['text_color']};">{rec['score']:.1f} / 100</div>
                    </div>
                </div>
                <div style="margin-top: 1rem; font-size: 1.05rem; color: #334155; font-weight: 500; line-height: 1.5;">
                    <strong>Decision Basis:</strong> {rec['desc']}
                </div>
                <hr style="margin: 1.25rem 0; border: 0; border-top: 1px solid rgba(0,0,0,0.08);" />
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
                    <div>
                        <span style="font-size: 0.8rem; font-weight: 600; color: #64748b;">Daily Return Win Rate:</span>
                        <span style="font-size: 0.95rem; font-weight: 700; color: #1e293b; margin-left: 0.3rem;">{rec['prob_metric']*100:.2f}%</span>
                    </div>
                    <div>
                        <span style="font-size: 0.8rem; font-weight: 600; color: #64748b;">RSI (14-day):</span>
                        <span style="font-size: 0.95rem; font-weight: 700; color: #1e293b; margin-left: 0.3rem;">{rec['rsi']:.1f}</span>
                    </div>
                    <div>
                        <span style="font-size: 0.8rem; font-weight: 600; color: #64748b;">Moving Averages (50 / 200 SMA):</span>
                        <span style="font-size: 0.95rem; font-weight: 700; color: #1e293b; margin-left: 0.3rem;">
                            ${rec['sma_50']:,.2f} / ${rec['sma_200']:,.2f} 
                            {"(Bullish Golden Cross)" if rec['sma_50'] > rec['sma_200'] else "(Bearish Death Cross)"}
                        </span>
                    </div>
                </div>
            </div>
        """
        st.markdown(decision_html, unsafe_allow_html=True)
        
        # --- $100 INVESTMENT SIMULATION ---
        st.markdown("### 💵 Historical Investment Simulation")
        st.markdown("What would a **$100 investment** made in the past be worth today compared to the current market condition?")
        
        simulations = {}
        for y in [5, 4, 3, 2, 1]:
            simulations[y] = calculate_investment_simulation(df, y)
            
        cols = st.columns(5)
        for i, y in enumerate([1, 2, 3, 4, 5]):
            sim = simulations[y]
            gain_loss_class = "delta-positive" if sim['net_gain'] >= 0 else "delta-negative"
            gain_loss_sign = "+" if sim['net_gain'] >= 0 else ""
            
            with cols[i]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">{y} Year{'s' if y > 1 else ''} Ago ({y*12}M)</div>
                        <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.5rem;">Invested on {sim['historical_date'].strftime('%b %Y')}</div>
                        <div class="metric-value">${sim['resulting_value']:,.2f}</div>
                        <div class="metric-delta {gain_loss_class}">
                            {gain_loss_sign}${abs(sim['net_gain']):,.2f} ({gain_loss_sign}{sim['pct_gain']:.2f}%)
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        st.write(" ")
        st.write(" ")
        
        # --- INTERACTIVE RETURN HISTOGRAMS ---
        st.markdown("### 📊 Interactive Return Distribution Histograms")
        st.markdown(
            "Visualizing the statistical probability density of daily returns. "
            "Use the controls on the upper right of the chart to **zoom in/out** dynamically without any quality loss."
        )
        
        # Create tabs for each year
        hist_tabs = st.tabs(["5 Years Data", "4 Years Data", "3 Years Data", "2 Years Data", "1 Year Data"])
        
        for idx, (label, period_df) in enumerate(periods.items()):
            with hist_tabs[idx]:
                returns = period_df['Daily_Return'] * 100 # convert to percent
                
                # Fit Normal Distribution
                mu, std = norm.fit(returns)
                
                # Plotly Chart Setup
                fig = go.Figure()
                
                # Compute return ranges for bins
                ret_min, ret_max = returns.min(), returns.max()
                
                # Dynamic Bin size
                bin_width = max(0.05, (ret_max - ret_min) / 45)
                
                # Negative Returns Histogram
                fig.add_trace(go.Histogram(
                    x=returns[returns < 0],
                    name="Loss Day (Negative Return)",
                    histnorm='density',
                    xbins=dict(start=ret_min, end=0, size=bin_width),
                    marker=dict(
                        color='rgba(239, 83, 80, 0.75)',  # soft coral red
                        line=dict(color='rgba(239, 83, 80, 1)', width=0.5)
                    ),
                    autobinx=False,
                    hovertemplate="Daily Return Bin: %{x:.2f}%<br>Probability Density: %{y:.4f}<extra></extra>"
                ))
                
                # Positive Returns Histogram
                fig.add_trace(go.Histogram(
                    x=returns[returns >= 0],
                    name="Profit Day (Positive Return)",
                    histnorm='density',
                    xbins=dict(start=0, end=ret_max, size=bin_width),
                    marker=dict(
                        color='rgba(38, 166, 154, 0.75)',  # soft teal green
                        line=dict(color='rgba(38, 166, 154, 1)', width=0.5)
                    ),
                    autobinx=False,
                    hovertemplate="Daily Return Bin: %{x:.2f}%<br>Probability Density: %{y:.4f}<extra></extra>"
                ))
                
                # Add Fitted normal curve PDF
                x_curve = np.linspace(ret_min, ret_max, 300)
                y_curve = norm.pdf(x_curve, mu, std)
                
                fig.add_trace(go.Scatter(
                    x=x_curve,
                    y=y_curve,
                    mode='lines',
                    name='Fitted Probability density (Normal Fit)',
                    line=dict(color='rgba(59, 130, 246, 0.85)', width=3, dash='dash')
                ))
                
                # Layout formatting
                fig.update_layout(
                    barmode='overlay',
                    title=dict(
                        text=f"Daily Returns Distribution - Last {label} (Daily % Change)",
                        font=dict(size=16, family="Plus Jakarta Sans", color="#1e293b")
                    ),
                    xaxis=dict(
                        title="Daily Percentage Return (%)",
                        gridcolor="rgba(226, 232, 240, 0.6)",
                        zerolinecolor="rgba(15, 23, 42, 0.3)",
                        zerolinewidth=1.5
                    ),
                    yaxis=dict(
                        title="Probability Density",
                        gridcolor="rgba(226, 232, 240, 0.6)"
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    plot_bgcolor='rgba(255,255,255,1)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=80, b=40),
                    hovermode='closest'
                )
                
                # Show in Streamlit
                st.plotly_chart(fig, use_container_width=True, config={'responsive': True, 'displaylogo': False})
                
                # Quick period statistics below each chart
                win_rate = (returns >= 0).mean() * 100
                avg_daily = returns.mean()
                ann_volatility = returns.std() * np.sqrt(252)
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Historical Win Rate (Positive Return Days)", f"{win_rate:.2f}%")
                with col_s2:
                    st.metric("Average Daily Return", f"{avg_daily:+.4f}%")
                with col_s3:
                    st.metric("Annualized Historical Volatility", f"{ann_volatility:.2f}%")
                    
    else:
        st.error(f"❌ Could not retrieve active history for symbol '{ticker_input}'. Please check the symbol is correct on Yahoo Finance (e.g. AAPL, MSFT, BTC-USD, ETH-USD) and try again.")
else:
    st.info("ℹ️ Please enter a valid stock or cryptocurrency symbol in the sidebar to run the analysis.")
# --- FOOTER SECTION ---
st.markdown("""
    <div class="footer">
        Designed by <strong>Mamoor Hayat</strong>. All Rights Reserved &copy; 2026.<br>
        <span style="font-size:0.75rem; color:#cbd5e1;">Disclaimer: All analytical computations are probabilistic and for informational purposes only. Do not treat as definitive financial advice.</span>
    </div>
""", unsafe_allow_html=True)
