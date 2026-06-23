"""
╔══════════════════════════════════════════════════════════════════╗
║          FINTECH STOCK & CRYPTO ANALYZER                        ║
║          Designed by: Mamoor Hayat                              ║
║          © All Rights Reserved                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FinScope Pro | Stock & Crypto Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — LIGHT, MODERN, ATTRACTIVE
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #f0f4ff 0%, #fafbff 50%, #f5f0ff 100%);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
    color: #ffffff;
}
section[data-testid="stSidebar"] * { color: #e0e0f0 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #a78bfa !important; }

div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e0e7ff;
    border-radius: 16px;
    padding: 18px 22px;
    box-shadow: 0 4px 20px rgba(99,102,241,0.08);
    transition: transform 0.2s;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(99,102,241,0.15);
}
.streamlit-expanderHeader {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent {
    background: #ffffff;
    border: 1px solid #e0e7ff;
    border-radius: 0 0 10px 10px;
}
.stButton > button {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 15px;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 22px rgba(99,102,241,0.45);
}
.stAlert { border-radius: 12px !important; }
.main-title {
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
    color: #ffffff;
    padding: 28px 36px;
    border-radius: 20px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(99,102,241,0.2);
}
.main-title h1 { color: #a78bfa; margin: 0 0 6px 0; font-size: 2.2rem; }
.main-title p  { color: #c4b5fd; margin: 0; font-size: 0.95rem; }
.section-header {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    padding: 12px 20px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1.05rem;
    margin: 20px 0 14px 0;
}
.verdict-buy {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    border: 2px solid #10b981;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}
.verdict-sell {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}
.verdict-neutral {
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    border: 2px solid #f59e0b;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}
.footer {
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    color: #c4b5fd;
    padding: 18px 28px;
    border-radius: 14px;
    text-align: center;
    margin-top: 40px;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str):
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, auto_adjust=True)
        return df, tk.info
    except Exception:
        return None, {}


def investment_simulation(df_year: pd.DataFrame):
    if df_year is None or df_year.empty:
        return None, None, None
    start_price = df_year['Close'].iloc[0]
    end_price   = df_year['Close'].iloc[-1]
    shares      = 100 / start_price
    final_val   = shares * end_price
    profit_loss = final_val - 100
    pct_return  = (profit_loss / 100) * 100
    return round(final_val, 2), round(profit_loss, 2), round(pct_return, 2)


def calculate_signals(df: pd.DataFrame):
    signals = {}
    close = df['Close']

    ma_20  = close.rolling(20).mean().iloc[-1]
    ma_50  = close.rolling(50).mean().iloc[-1]
    ma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    current = close.iloc[-1]

    signals['MA20']  = "Bullish" if current > ma_20  else "Bearish"
    signals['MA50']  = "Bullish" if current > ma_50  else "Bearish"
    signals['MA200'] = ("Bullish" if current > ma_200 else "Bearish") if ma_200 else "N/A"

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    rsi   = 100 - (100 / (1 + rs)).iloc[-1]
    signals['RSI'] = round(rsi, 1)
    if rsi < 30:   signals['RSI_signal'] = "Oversold (Bullish)"
    elif rsi > 70: signals['RSI_signal'] = "Overbought (Bearish)"
    else:          signals['RSI_signal'] = "Neutral"

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    signals['MACD'] = "Bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "Bearish"

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up  = (bb_mid + 2 * bb_std).iloc[-1]
    bb_low = (bb_mid - 2 * bb_std).iloc[-1]
    if current < bb_low:   signals['Bollinger'] = "Below Lower Band (Bullish)"
    elif current > bb_up:  signals['Bollinger'] = "Above Upper Band (Bearish)"
    else:                  signals['Bollinger'] = "Within Bands (Neutral)"

    daily_ret = close.pct_change().dropna()
    signals['Volatility'] = round(daily_ret.std() * np.sqrt(252) * 100, 2)

    score = 0
    if signals['MA20']  == "Bullish": score += 1
    if signals['MA50']  == "Bullish": score += 1
    if signals['MA200'] == "Bullish": score += 1
    if signals['MACD']  == "Bullish": score += 2
    if "Bullish" in signals['RSI_signal']: score += 2
    if "Bullish" in signals['Bollinger']:  score += 1
    signals['score'] = score
    return signals


def make_histogram(df: pd.DataFrame, ticker: str, period_label: str):
    df = df.copy()
    df['Date']   = df.index
    df['Return'] = df['Close'].pct_change() * 100
    df['Color']  = df['Return'].apply(lambda x: '#10b981' if x >= 0 else '#ef4444')
    max_abs = df['Return'].abs().max() or 1
    df['Opacity'] = df['Return'].abs().apply(lambda x: 0.4 + 0.6 * (x / max_abs))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Close'],
        marker=dict(
            color=df['Color'],
            opacity=df['Opacity'],
            line=dict(color='rgba(0,0,0,0.05)', width=0.3)
        ),
        name='Close Price',
        hovertemplate=(
            '<b>%{x|%Y-%m-%d}</b><br>'
            'Close: $%{y:.2f}<br>'
            'Day Return: %{customdata:.2f}%<extra></extra>'
        ),
        customdata=df['Return']
    ))

    ma20 = df['Close'].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=df['Date'], y=ma20,
        mode='lines', name='20-Day MA',
        line=dict(color='#6366f1', width=2, dash='dot')
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{ticker.upper()} — {period_label} Price Histogram</b>",
            font=dict(size=18, color='#1a1a2e'), x=0.02
        ),
        xaxis=dict(
            title='Date',
            showgrid=True, gridcolor='rgba(99,102,241,0.1)',
            rangeslider=dict(visible=True, thickness=0.06),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label='1M', step='month', stepmode='backward'),
                    dict(count=3, label='3M', step='month', stepmode='backward'),
                    dict(count=6, label='6M', step='month', stepmode='backward'),
                    dict(step='all', label='All')
                ],
                bgcolor='#f0f4ff', activecolor='#6366f1',
                font=dict(color='#1a1a2e')
            ),
            fixedrange=False
        ),
        yaxis=dict(
            title='Price (USD)', showgrid=True,
            gridcolor='rgba(99,102,241,0.1)',
            tickprefix='$', fixedrange=False
        ),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            bgcolor='rgba(255,255,255,0.8)', bordercolor='#e0e7ff', borderwidth=1
        ),
        plot_bgcolor='#fafbff', paper_bgcolor='#ffffff',
        margin=dict(l=60, r=20, t=80, b=60),
        hovermode='x unified', dragmode='zoom', height=430
    )
    return fig


def investment_bar_chart(results: dict):
    periods = list(results.keys())
    finals  = [v['final'] for v in results.values()]
    profits = [v['profit'] for v in results.values()]
    colors  = ['#10b981' if p >= 0 else '#ef4444' for p in profits]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=periods, y=finals,
        marker_color=colors,
        marker_line=dict(color='rgba(0,0,0,0.1)', width=1),
        text=[f"${f:.2f}" for f in finals],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Final: $%{y:.2f}<extra></extra>'
    ))
    fig.add_hline(y=100, line_dash='dash', line_color='#6366f1',
                  annotation_text='Initial $100', annotation_position='right')
    fig.update_layout(
        title=dict(text='<b>$100 Investment Simulation per Period</b>',
                   font=dict(size=17, color='#1a1a2e'), x=0.02),
        xaxis_title='Period',
        yaxis=dict(title='Final Value (USD)', tickprefix='$'),
        plot_bgcolor='#fafbff', paper_bgcolor='#ffffff',
        height=350, margin=dict(l=60, r=20, t=70, b=50), showlegend=False
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 FinScope Pro")
    st.markdown("---")
    ticker_input = st.text_input(
        "Enter Stock / Crypto Ticker",
        value="",
        placeholder="e.g. AAPL, BTC-USD, TSLA"
    ).strip().upper()

    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
**Stocks:** Type the ticker symbol.
- `AAPL` → Apple Inc.
- `TSLA` → Tesla
- `MSFT` → Microsoft
- `GOOGL` → Alphabet
- `AMZN` → Amazon

**Crypto (add -USD):**
- `BTC-USD` → Bitcoin
- `ETH-USD` → Ethereum
- `SOL-USD` → Solana
- `BNB-USD` → Binance Coin

**Find any ticker:** [finance.yahoo.com](https://finance.yahoo.com)
    """)
    st.markdown("---")
    st.markdown("### 🧠 Glossary")
    with st.expander("📘 Key Terms"):
        st.markdown("""
**RSI**: Measures momentum. <30 = oversold (bullish), >70 = overbought (bearish).

**MACD**: Trend-following momentum indicator.

**Moving Average**: Average price over N days. Smooths out noise.

**Bollinger Bands**: Volatility bands ±2σ around a 20-day MA.

**Volatility**: How wildly the price swings (annualised %).

**Bull/Bear**: Bull = rising market; Bear = falling market.
        """)

    st.markdown("---")
    st.markdown("""
<div style='text-align:center; font-size:0.75rem; color:#a78bfa;'>
Designed by <b>Mamoor Hayat</b><br>
© All Rights Reserved
</div>
""", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 Analyze", use_container_width=True)


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.markdown("""
<div class='main-title'>
  <h1>📈 FinScope Pro — Stock & Crypto Analyzer</h1>
  <p>Advanced technical analysis · Investment simulation · Buy/Sell signals · Designed by <b>Mamoor Hayat</b> · © All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)

if not analyze_btn or not ticker_input:
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "📊", "5-Year Histograms",
         "Interactive bar charts with zoom/pan for 1–5 year windows. Green = up days, Red = down days."),
        (c2, "💰", "Investment Simulation",
         "See how $100 invested at the start of each period would have grown or shrunk by today."),
        (c3, "🤖", "Buy / Hold / Sell Signal",
         "6 technical indicators combined into a probability-based recommendation."),
    ]:
        col.markdown(f"""
<div style='background:#fff; border:1px solid #e0e7ff; border-radius:14px;
            padding:22px; text-align:center; box-shadow:0 4px 16px rgba(99,102,241,0.07);'>
<div style='font-size:2.5rem;'>{icon}</div>
<h4 style='color:#6366f1; margin:8px 0 6px;'>{title}</h4>
<p style='color:#64748b; font-size:0.85rem; margin:0;'>{desc}</p>
</div>""", unsafe_allow_html=True)
    st.info("👈  Enter a **stock ticker** or **crypto symbol** in the sidebar and click **🚀 Analyze** to begin!")
    st.markdown("""
<div class='footer'>
  <b>FinScope Pro</b> · Designed & Developed by <b>Mamoor Hayat</b><br>
  © All Rights Reserved · For Educational Purposes Only · Not Financial Advice<br>
  <span style='font-size:0.75rem; color:#7c3aed;'>Powered by yFinance · Plotly · Streamlit</span>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ── FETCH DATA ───────────────────────────────
with st.spinner(f"Fetching data for **{ticker_input}** ..."):
    df_5y, info = fetch_stock_data(ticker_input, "5y")

if df_5y is None or df_5y.empty:
    st.error(f"❌ No data found for **{ticker_input}**. Please check the symbol and try again.")
    st.stop()

# ── COMPANY KPIs ─────────────────────────────
name      = info.get('longName', ticker_input)
sector    = info.get('sector', info.get('category', '—'))
mktcap    = info.get('marketCap')
mktcap_s  = f"${mktcap/1e9:.2f}B" if mktcap else "—"
current_price = df_5y['Close'].iloc[-1]
prev_price    = df_5y['Close'].iloc[-2]
day_change    = ((current_price - prev_price) / prev_price) * 100

st.markdown(f"<div class='section-header'>🏢 {name} ({ticker_input})</div>", unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Current Price",  f"${current_price:.2f}", f"{day_change:+.2f}%")
k2.metric("Market Cap",     mktcap_s)
k3.metric("Sector / Type",  sector)
k4.metric("52W High",       f"${df_5y['Close'].tail(252).max():.2f}")
k5.metric("52W Low",        f"${df_5y['Close'].tail(252).min():.2f}")

# ── BUILD PERIOD SLICES ──────────────────────
today = datetime.now()
periods = {
    "5 Years": df_5y,
    "4 Years": df_5y[df_5y.index >= (today - timedelta(days=4*365))],
    "3 Years": df_5y[df_5y.index >= (today - timedelta(days=3*365))],
    "2 Years": df_5y[df_5y.index >= (today - timedelta(days=2*365))],
    "1 Year":  df_5y[df_5y.index >= (today - timedelta(days=365))],
}

# ─────────────────────────────────────────────
# SECTION 1 — HISTOGRAMS
# ─────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>📊 Interactive Price Histograms "
    "(Scroll to Zoom · Click & Drag to Pan · Range Slider below each chart)</div>",
    unsafe_allow_html=True
)
st.caption("🟢 Green = up day  |  🔴 Red = down day  |  Deeper colour = larger move  |  Purple dotted line = 20-day MA")

tabs = st.tabs(["📅 5 Years", "📅 4 Years", "📅 3 Years", "📅 2 Years", "📅 1 Year"])
for tab, (label, df_p) in zip(tabs, periods.items()):
    with tab:
        if df_p.empty:
            st.warning(f"Not enough data for {label}.")
            continue
        fig = make_histogram(df_p, ticker_input, label)
        st.plotly_chart(fig, use_container_width=True, config={
            'scrollZoom': True,
            'displayModeBar': True,
            'toImageButtonOptions': {'format': 'png', 'scale': 2}
        })

# ─────────────────────────────────────────────
# SECTION 2 — INVESTMENT SIMULATION
# ─────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>💰 $100 Investment Simulation — What if you had invested?</div>",
    unsafe_allow_html=True
)
st.caption("How much would **$100** invested at the **start** of each period be worth **today**?")

sim_results = {}
inv_cols = st.columns(5)
for col, (label, df_p) in zip(inv_cols, periods.items()):
    final, profit, pct = investment_simulation(df_p)
    if final is None:
        continue
    sim_results[label] = {'final': final, 'profit': profit, 'pct': pct}
    emoji = "🟢" if profit >= 0 else "🔴"
    delta_str = f"+${profit:.2f}" if profit >= 0 else f"-${abs(profit):.2f}"
    col.metric(
        label=f"{emoji} {label}",
        value=f"${final:.2f}",
        delta=f"{delta_str} ({pct:+.1f}%)"
    )

if sim_results:
    st.plotly_chart(
        investment_bar_chart(sim_results),
        use_container_width=True,
        config={'scrollZoom': True}
    )

# ─────────────────────────────────────────────
# SECTION 3 — TECHNICAL SIGNALS & BUY/SELL
# ─────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>🤖 Technical Analysis & Buy / Hold / Sell Recommendation</div>",
    unsafe_allow_html=True
)

signals  = calculate_signals(df_5y.tail(500))
score    = signals['score']
pct_bull = round((score / 8) * 100, 1)

if pct_bull >= 65:
    verdict, vc, vi = "BUY 🟢",        "verdict-buy",     "✅"
elif pct_bull <= 35:
    verdict, vc, vi = "AVOID / SELL 🔴","verdict-sell",    "⛔"
else:
    verdict, vc, vi = "HOLD / WAIT 🟡", "verdict-neutral", "⚠️"

bar_color = "#10b981" if pct_bull >= 65 else "#ef4444" if pct_bull <= 35 else "#f59e0b"

c1, c2 = st.columns([1, 2])
with c1:
    st.markdown(f"""
<div class='{vc}'>
  <div style='font-size:1.1rem; font-weight:600; color:#374151;'>Overall Signal</div>
  <div style='font-size:2.2rem; font-weight:800; margin:8px 0;'>{vi} {verdict}</div>
  <div style='font-size:1rem; color:#374151;'>Bullish Score: <b>{score}/8</b> ({pct_bull}%)</div>
  <div style='margin-top:10px;'>
    <div style='background:#e0e7ff; border-radius:20px; height:12px;'>
      <div style='background:{bar_color}; width:{pct_bull}%; height:12px; border-radius:20px;'></div>
    </div>
    <div style='display:flex; justify-content:space-between; font-size:0.7rem; color:#6b7280; margin-top:4px;'>
      <span>Bearish 0%</span><span>Neutral 50%</span><span>Bullish 100%</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

with c2:
    sig_data = {
        'Indicator': ['MA (20-day)', 'MA (50-day)', 'MA (200-day)', 'MACD', 'RSI', 'Bollinger Bands'],
        'Reading': [
            signals['MA20'], signals['MA50'], signals['MA200'],
            signals['MACD'],
            f"{signals['RSI']} — {signals['RSI_signal']}",
            signals['Bollinger']
        ]
    }
    df_sig = pd.DataFrame(sig_data)

    def colour_signal(val):
        if 'Bullish' in str(val): return 'background-color:#d1fae5; color:#065f46'
        if 'Bearish' in str(val): return 'background-color:#fee2e2; color:#991b1b'
        return 'background-color:#fef3c7; color:#92400e'

    st.dataframe(
        df_sig.style.applymap(colour_signal, subset=['Reading']),
        hide_index=True, use_container_width=True
    )
    st.caption(f"🌪️ Annualised Volatility: **{signals['Volatility']}%**")

# ─────────────────────────────────────────────
# SECTION 4 — BEGINNER GUIDE
# ─────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>📘 Complete Beginner's Guide — Everything Explained</div>",
    unsafe_allow_html=True
)

with st.expander("🔍 Understanding the Histograms", expanded=False):
    st.markdown("""
**What is this histogram?**
Each vertical bar represents the *closing price* of the stock on one trading day.

| Colour | Meaning |
|--------|---------|
| 🟢 **Green** | Price closed *higher* than the previous day (UP day) |
| 🔴 **Red**   | Price closed *lower* than the previous day (DOWN day) |
| Deeper shade | Larger price move that day |
| Purple dotted | 20-Day Moving Average (smoothed trend line) |

**Zoom & Pan controls:**
- 🖱️ Scroll wheel → zoom in / out (no quality loss — SVG vector rendering)
- Click & drag → pan left/right
- Range-selector buttons (1M / 3M / 6M / All) → quick jumps
- Drag the slider bar at the bottom → navigate any date range

**Why multiple time periods?**
Looking at 5 years gives the big picture. 1 year shows recent momentum. Comparing them helps you see whether the current trend is an exception or the norm.
    """)

with st.expander("💰 Understanding the Investment Simulation", expanded=False):
    st.markdown("""
**What does this tell me?**
It answers: *"If I had put $100 into this stock/coin at the start of each window, how much would I have today?"*

**The maths (simplified):**
1. Find the *opening price* at the start of the period.
2. Calculate how many shares $100 would have bought (= 100 ÷ start price).
3. Multiply those shares by *today's price* → that's your portfolio value.
4. Subtract $100 → profit or loss.

**Example:** BTC-USD was $20,000 two years ago. $100 buys 0.005 BTC.
Today BTC is $60,000. Your 0.005 BTC = **$300** → profit of **$200 (+200%)**.

⚠️ **Disclaimer:** This ignores brokerage fees, taxes, dividends, and stock splits.
It is illustrative only — not a guarantee of future returns.
    """)

with st.expander("🤖 Understanding Buy / Hold / Sell Signals", expanded=False):
    st.markdown("""
The recommendation is built from **6 technical indicators** totalling a max score of **8 bullish points**.

| Score Band | Signal | Meaning |
|-----------|--------|---------|
| 65–100 % | ✅ BUY | Most indicators point upward |
| 36–64 %  | ⚠️ HOLD/WAIT | Mixed signals — sit on the fence |
| 0–35 %   | ⛔ AVOID/SELL | Most indicators point downward |

---
**MA (Moving Average)**
Average closing price over the last N days. If today's price is *above* the MA, it suggests the stock is in an uptrend. Three MAs are checked: 20-day, 50-day, 200-day.

**RSI (Relative Strength Index)**
Momentum oscillator from 0–100.
- < 30 → Oversold (potential bargain buy)
- > 70 → Overbought (potentially risky to enter)
- 30–70 → Normal range

**MACD (Moving Average Convergence/Divergence)**
Compares two exponential moving averages (12-day vs 26-day).
- MACD line crosses *above* signal line → Bullish momentum
- Crosses *below* → Bearish momentum

**Bollinger Bands**
Bands set ±2 standard deviations from a 20-day MA.
- Price near *lower* band → historically tends to bounce up (Bullish)
- Price near *upper* band → historically tends to pull back (Bearish)

---
> ⚠️ **Important:** Technical analysis is *probabilistic*, not a crystal ball.
> This app is for **educational purposes only** and does **NOT** constitute financial advice.
> Always do your own research (DYOR) and consult a licensed financial advisor before investing.
    """)

# ─────────────────────────────────────────────
# SECTION 5 — FULL PRICE CHART
# ─────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>📉 Full 5-Year Price & Volume Chart</div>",
    unsafe_allow_html=True
)

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=df_5y.index, y=df_5y['Close'],
    mode='lines', name='Close Price',
    line=dict(color='#6366f1', width=2),
    fill='tozeroy', fillcolor='rgba(99,102,241,0.07)'
))
fig_line.add_trace(go.Bar(
    x=df_5y.index, y=df_5y['Volume'],
    name='Volume', marker_color='rgba(139,92,246,0.25)',
    yaxis='y2'
))
fig_line.update_layout(
    yaxis=dict(title='Price (USD)', tickprefix='$', fixedrange=False),
    yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False, fixedrange=False),
    xaxis=dict(
        fixedrange=False,
        rangeslider=dict(visible=True, thickness=0.05),
        rangeselector=dict(
            buttons=[
                dict(count=6, label='6M', step='month', stepmode='backward'),
                dict(count=1, label='1Y', step='year',  stepmode='backward'),
                dict(count=3, label='3Y', step='year',  stepmode='backward'),
                dict(step='all', label='5Y')
            ]
        )
    ),
    plot_bgcolor='#fafbff', paper_bgcolor='#ffffff',
    height=400, margin=dict(l=60, r=60, t=30, b=50),
    legend=dict(orientation='h', y=1.05)
)
st.plotly_chart(fig_line, use_container_width=True,
                config={'scrollZoom': True, 'displayModeBar': True})

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class='footer'>
  <b>FinScope Pro</b> &nbsp;·&nbsp; Designed & Developed by <b>Mamoor Hayat</b><br>
  © All Rights Reserved &nbsp;·&nbsp; For Educational Purposes Only &nbsp;·&nbsp; Not Financial Advice<br>
  <span style='font-size:0.75rem; color:#7c3aed;'>Powered by yFinance · Plotly · Streamlit</span>
</div>
""", unsafe_allow_html=True)
