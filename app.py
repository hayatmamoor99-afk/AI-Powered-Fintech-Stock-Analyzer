"""
StockLens - AI-Powered Fintech Stock Analyzer
=============================================
Requirements:
    pip install streamlit yfinance plotly pandas numpy anthropic

Run:
    streamlit run stocklens_app.py
"""

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Page Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockLens — AI Market Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a0e1a; color: #e2e8f0; }
    .main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

    /* Hide sidebar toggle & default header */
    [data-testid="collapsedControl"] { display: none; }
    #MainMenu, footer, header { visibility: hidden; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 11px !important; }
    [data-testid="stMetricValue"] { color: #e2e8f0 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #111827;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748b;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 10px 28px !important;
        transition: opacity .2s;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* Quick pick buttons */
    div[data-testid="column"] .stButton > button {
        background: #111827 !important;
        border: 1px solid #1e2d45 !important;
        color: #94a3b8 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 6px 10px !important;
    }
    div[data-testid="column"] .stButton > button:hover {
        border-color: #3b82f6 !important;
        color: #60a5fa !important;
    }

    /* Text input */
    .stTextInput > div > div > input {
        background: #111827 !important;
        border: 2px solid #1e2d45 !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 12px 18px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
    }

    .section-header {
        font-size: 12px;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 24px 0 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e2d45;
    }

    .inv-card {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .guide-box {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 8px;
        height: 100%;
    }

    .disclaimer {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.25);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 12px;
        color: #b45309;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────────────────

def get_stock_data(ticker: str) -> dict:
    tk   = yf.Ticker(ticker)
    info = tk.info
    end  = datetime.today()
    hist = tk.history(start=end - timedelta(days=5 * 365 + 30), end=end, interval="1mo")
    if hist.empty:
        raise ValueError(f"No data found for '{ticker}'. Please check the ticker symbol.")
    hist = hist[["Close"]].dropna()
    periods = {
        "5yr": hist,
        "4yr": hist[hist.index >= end - timedelta(days=4 * 365)],
        "3yr": hist[hist.index >= end - timedelta(days=3 * 365)],
        "2yr": hist[hist.index >= end - timedelta(days=2 * 365)],
        "1yr": hist[hist.index >= end - timedelta(days=365)],
    }
    cp = info.get("currentPrice") or info.get("regularMarketPrice") or float(hist["Close"].iloc[-1])
    pc = info.get("previousClose") or info.get("regularMarketPreviousClose") or float(hist["Close"].iloc[-2])
    return {
        "ticker":       ticker.upper(),
        "company_name": info.get("longName") or info.get("shortName") or ticker.upper(),
        "current_price": round(float(cp), 2),
        "prev_close":    round(float(pc), 2),
        "market_cap":    info.get("marketCap"),
        "pe_ratio":      info.get("trailingPE") or info.get("forwardPE"),
        "week52_high":   info.get("fiftyTwoWeekHigh"),
        "week52_low":    info.get("fiftyTwoWeekLow"),
        "avg_volume":    info.get("averageVolume"),
        "dividend":      info.get("dividendYield"),
        "sector":        info.get("sector", "N/A"),
        "ma50":          info.get("fiftyDayAverage"),
        "ma200":         info.get("twoHundredDayAverage"),
        "periods":       periods,
    }


def compute_technicals(hist: pd.DataFrame) -> dict:
    closes = hist["Close"].values.astype(float)
    if len(closes) < 5:
        return {"rsi": 50, "trend": "Neutral", "trend_pct": 0, "volatility": "Unknown", "vol_pct": 0}
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    ag = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
    al = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
    rsi = 100 - (100 / (1 + ag / al)) if al != 0 else 50
    tp  = (closes[-1] - closes[0]) / closes[0] * 100
    trend = ("Strong Uptrend"   if tp > 20 else
             "Uptrend"          if tp > 5  else
             "Strong Downtrend" if tp < -20 else
             "Downtrend"        if tp < -5  else "Sideways")
    returns = np.diff(closes) / closes[:-1]
    vp = float(np.std(returns) * np.sqrt(12) * 100)
    vol = ("Very High" if vp > 60 else "High" if vp > 35 else "Medium" if vp > 15 else "Low")
    return {"rsi": round(float(rsi), 1), "trend": trend,
            "trend_pct": round(float(tp), 2), "volatility": vol, "vol_pct": round(vp, 1)}


def compute_recommendation(tech: dict, data: dict) -> dict:
    score = 50
    rsi, trend = tech["rsi"], tech["trend"]
    if rsi < 30:   score += 20
    elif rsi < 45: score += 10
    elif rsi > 70: score -= 20
    elif rsi > 55: score -= 5
    if "Strong Uptrend"   in trend: score += 20
    elif "Uptrend"        in trend: score += 10
    elif "Strong Downtrend" in trend: score -= 20
    elif "Downtrend"      in trend: score -= 10
    cp, m50, m200 = data["current_price"], data.get("ma50") or data["current_price"], data.get("ma200") or data["current_price"]
    score += 8 if cp > m50  else -8
    score += 8 if cp > m200 else -8
    score = max(0, min(100, score))
    if score >= 62:   rec, color, icon = "BUY",  "#10b981", "🚀"
    elif score <= 38: rec, color, icon = "SELL", "#ef4444", "⛔"
    else:             rec, color, icon = "HOLD", "#f59e0b", "⏸️"
    h = data["periods"]["5yr"]["Close"].values
    return {"rec": rec, "confidence": score, "color": color, "icon": icon,
            "support": round(float(np.percentile(h, 15)), 2),
            "resistance": round(float(np.percentile(h, 85)), 2)}


def simulate_investment(hist: pd.DataFrame, amount: float = 100) -> dict:
    closes = hist["Close"].dropna().values.astype(float)
    if len(closes) < 2:
        return {"value": amount, "gain": 0, "pct": 0}
    value = round((amount / closes[0]) * closes[-1], 2)
    return {"value": value, "gain": round(value - amount, 2),
            "pct": round((closes[-1] - closes[0]) / closes[0] * 100, 2)}


def fmt_large(n) -> str:
    if n is None: return "N/A"
    n = float(n)
    if n >= 1e12: return f"${n/1e12:.2f}T"
    if n >= 1e9:  return f"${n/1e9:.2f}B"
    if n >= 1e6:  return f"${n/1e6:.2f}M"
    return f"{n:,.0f}"


def make_chart(hist: pd.DataFrame, ticker: str, label: str) -> go.Figure:
    closes = hist["Close"].dropna()
    is_up  = float(closes.iloc[-1]) >= float(closes.iloc[0])
    color  = "#10b981" if is_up else "#ef4444"
    fill_c = "rgba(16,185,129,0.08)" if is_up else "rgba(239,68,68,0.08)"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=closes.index, y=closes,
        fill="tozeroy", fillcolor=fill_c,
        line=dict(color=color, width=2.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.2f}<extra></extra>",
        name=ticker,
    ))
    if len(closes) >= 6:
        fig.add_trace(go.Scatter(
            x=closes.index, y=closes.rolling(3).mean(),
            line=dict(color="rgba(139,92,246,0.6)", width=1.5, dash="dot"),
            name="3-mo MA", hoverinfo="skip",
        ))
    fig.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
        font=dict(family="Inter,sans-serif", color="#94a3b8"),
        title=dict(text=f"{ticker} — {label} Price History", font=dict(size=14, color="#e2e8f0")),
        xaxis=dict(gridcolor="#1e2d45", showgrid=True, zeroline=False, tickfont=dict(size=10), showline=False),
        yaxis=dict(gridcolor="#1e2d45", showgrid=True, zeroline=False, tickprefix="$", tickfont=dict(size=10), showline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#94a3b8")),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a2235", font=dict(color="#e2e8f0")),
        height=340,
    )
    return fig


# ── App State ────────────────────────────────────────────────────────────────────
if "ticker" not in st.session_state:
    st.session_state.ticker = ""


# ── Header ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:32px 0 24px">
    <div style="font-size:48px;margin-bottom:8px">📡</div>
    <div style="font-size:36px;font-weight:900;background:linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px">
        StockLens
    </div>
    <div style="color:#64748b;font-size:14px">AI-Powered Market Intelligence · Real yfinance Data</div>
</div>
""", unsafe_allow_html=True)


# ── Search Bar ───────────────────────────────────────────────────────────────────
col_inp, col_btn = st.columns([5, 1])
with col_inp:
    ticker_input = st.text_input(
        "ticker", label_visibility="collapsed",
        placeholder="Enter stock ticker: AAPL, TSLA, MSFT, NVDA, RELIANCE.NS ...",
        value=st.session_state.ticker,
        key="ticker_field",
    )
with col_btn:
    search_clicked = st.button("🔍 Analyze", use_container_width=True)

# ── Quick Picks ──────────────────────────────────────────────────────────────────
st.markdown("<div style='margin:6px 0 4px;font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.8px'>⚡ Quick Picks</div>", unsafe_allow_html=True)
qcols = st.columns(10)
quick = ["AAPL","MSFT","GOOGL","TSLA","AMZN","NVDA","META","JPM","BRK-B","NFLX"]
for col, qt in zip(qcols, quick):
    with col:
        if st.button(qt, key=f"q_{qt}"):
            st.session_state.ticker = qt
            st.rerun()

# ── Resolve ticker ───────────────────────────────────────────────────────────────
final_ticker = (ticker_input or st.session_state.ticker).strip().upper()
should_run   = search_clicked or (st.session_state.ticker and not ticker_input)

if not final_ticker:
    # Feature badges
    st.markdown("""
    <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:40px">
        <span style="background:rgba(59,130,246,.15);color:#60a5fa;padding:8px 20px;border-radius:20px;font-size:13px;font-weight:600">📊 5-Year Charts</span>
        <span style="background:rgba(139,92,246,.15);color:#a78bfa;padding:8px 20px;border-radius:20px;font-size:13px;font-weight:600">💰 $100 Simulator</span>
        <span style="background:rgba(16,185,129,.15);color:#34d399;padding:8px 20px;border-radius:20px;font-size:13px;font-weight:600">🤖 AI Recommendation</span>
        <span style="background:rgba(245,158,11,.15);color:#fbbf24;padding:8px 20px;border-radius:20px;font-size:13px;font-weight:600">⚡ Real yfinance Data</span>
    </div>
    <div style="text-align:center;margin-top:24px;color:#334155;font-size:13px">
        Type a ticker above or click a quick pick to get started
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load Data ────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching real market data for **{final_ticker}**..."):
    try:
        data = get_stock_data(final_ticker)
    except Exception as e:
        st.error(f"⚠️ {e}")
        st.stop()

tech = compute_technicals(data["periods"]["5yr"])
rec  = compute_recommendation(tech, data)

# ── Stock Header ─────────────────────────────────────────────────────────────────
change     = data["current_price"] - data["prev_close"]
change_pct = (change / data["prev_close"] * 100) if data["prev_close"] else 0
is_up      = change >= 0
arrow      = "▲" if is_up else "▼"
delta_col  = "#10b981" if is_up else "#ef4444"

h1, h2 = st.columns([3, 1])
with h1:
    st.markdown(f"""
    <div style="margin-bottom:4px">
        <span style="font-size:36px;font-weight:900;color:#fff;letter-spacing:-1px">{data['ticker']}</span>
        <span style="margin-left:14px;font-size:14px;color:#64748b">{data['company_name']}</span>
    </div>
    <div style="font-size:12px;color:#64748b">Sector: {data['sector']}</div>
    """, unsafe_allow_html=True)
with h2:
    st.markdown(f"""
    <div style="text-align:right">
        <div style="font-size:36px;font-weight:900;color:#fff">${data['current_price']:,.2f}</div>
        <div style="font-size:14px;font-weight:600;color:{delta_col}">
            {arrow} ${abs(change):.2f} ({'+' if is_up else ''}{change_pct:.2f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
mc = st.columns(8)
mets = [
    ("Market Cap",  fmt_large(data["market_cap"])),
    ("P/E Ratio",   f"{data['pe_ratio']:.1f}" if data["pe_ratio"] else "N/A"),
    ("52W High",    f"${data['week52_high']:.2f}" if data["week52_high"] else "N/A"),
    ("52W Low",     f"${data['week52_low']:.2f}"  if data["week52_low"]  else "N/A"),
    ("Avg Volume",  fmt_large(data["avg_volume"])),
    ("MA 50",       f"${data['ma50']:.2f}"  if data["ma50"]  else "N/A"),
    ("MA 200",      f"${data['ma200']:.2f}" if data["ma200"] else "N/A"),
    ("RSI (14)",    str(tech["rsi"])),
]
for col, (label, val) in zip(mc, mets):
    col.metric(label, val)

# ── Charts ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Historical Price Charts</div>', unsafe_allow_html=True)
tabs = st.tabs(["📅 5 Years", "📅 4 Years", "📅 3 Years", "📅 2 Years", "📅 1 Year"])
period_map = [("5yr","5 Year"),("4yr","4 Year"),("3yr","3 Year"),("2yr","2 Year"),("1yr","1 Year")]
for tab, (key, label) in zip(tabs, period_map):
    with tab:
        h = data["periods"][key]
        if h.empty:
            st.warning(f"Not enough data for {label} period.")
        else:
            st.plotly_chart(make_chart(h, data["ticker"], label), use_container_width=True)

# ── $100 Simulator ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">💰 $100 Investment Simulator</div>', unsafe_allow_html=True)
icols = st.columns(5)
for col, (key, label) in zip(icols, period_map):
    inv = simulate_investment(data["periods"][key])
    pos = inv["gain"] >= 0
    c   = "#10b981" if pos else "#ef4444"
    ar  = "▲" if pos else "▼"
    col.markdown(f"""
    <div class="inv-card">
        <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.5px">{label}</div>
        <div style="font-size:11px;color:#64748b;margin-top:2px">Started $100</div>
        <div style="font-size:22px;font-weight:800;color:{c};margin-top:8px">${inv['value']:,.2f}</div>
        <div style="font-size:13px;font-weight:600;color:{c};margin-top:4px">{'+' if pos else ''}${inv['gain']:,.2f}</div>
        <div style="font-size:11px;color:{c};margin-top:2px">{ar} {abs(inv['pct']):.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# ── Recommendation ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🤖 AI Recommendation Engine</div>', unsafe_allow_html=True)
vc  = {"BUY":"rgba(16,185,129,0.08)","SELL":"rgba(239,68,68,0.08)","HOLD":"rgba(245,158,11,0.08)"}[rec["rec"]]
vb  = {"BUY":"rgba(16,185,129,0.3)", "SELL":"rgba(239,68,68,0.3)", "HOLD":"rgba(245,158,11,0.3)"}[rec["rec"]]
bar = "█" * int(rec["confidence"] / 5) + "░" * (20 - int(rec["confidence"] / 5))

r1, r2 = st.columns([2, 1])
with r1:
    st.markdown(f"""
    <div style="background:{vc};border:1px solid {vb};border-radius:12px;padding:24px">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px">
            <div style="font-size:44px">{rec['icon']}</div>
            <div>
                <div style="font-size:11px;font-weight:700;color:{rec['color']};text-transform:uppercase;letter-spacing:1px">AI Verdict</div>
                <div style="font-size:34px;font-weight:900;color:{rec['color']}">{rec['rec']}</div>
            </div>
            <div style="margin-left:auto;text-align:right">
                <div style="font-size:11px;color:#64748b">Confidence</div>
                <div style="font-size:34px;font-weight:900;color:{rec['color']}">{rec['confidence']}%</div>
            </div>
        </div>
        <div style="font-size:11px;color:#64748b;font-family:monospace;letter-spacing:1px;margin-bottom:6px">{bar}</div>
        <div style="font-size:12px;color:#64748b">Based on RSI, trend direction, moving averages & probability scoring</div>
    </div>
    """, unsafe_allow_html=True)

with r2:
    signals = [
        ("Trend",       tech["trend"],         "#e2e8f0"),
        ("RSI",         str(tech["rsi"]),       "#10b981" if tech["rsi"]<30 else "#ef4444" if tech["rsi"]>70 else "#e2e8f0"),
        ("Volatility",  tech["volatility"],     "#e2e8f0"),
        ("Support",     f"${rec['support']:,.2f}", "#10b981"),
        ("Resistance",  f"${rec['resistance']:,.2f}", "#ef4444"),
        ("Trend %",     f"{'+' if tech['trend_pct']>0 else ''}{tech['trend_pct']:.1f}%", "#10b981" if tech["trend_pct"]>0 else "#ef4444"),
    ]
    for label, val, color in signals:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2d45;border-radius:8px;
            padding:10px 14px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.5px">{label}</span>
            <span style="font-size:13px;font-weight:700;color:{color}">{val}</span>
        </div>
        """, unsafe_allow_html=True)

# ── Beginner's Guide ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📚 Beginner\'s Guide</div>', unsafe_allow_html=True)
guides = [
    ("📊 Histogram",   "A line chart of stock price over time. Rising line = stock gaining value."),
    ("💵 $100 Sim",    "Shows what $100 invested at the start would be worth today."),
    ("📈 RSI",         "0–100 scale. Below 30 = oversold (buy signal). Above 70 = overbought (sell signal)."),
    ("⚡ Volatility",  "How wildly the stock swings. High = bigger gains OR losses."),
    ("🎯 Support",     "Price level the stock tends to bounce up from. A floor."),
    ("🚧 Resistance",  "Price level the stock struggles to break above. A ceiling."),
]
gcols = st.columns(6)
for col, (title, body) in zip(gcols, guides):
    col.markdown(f"""
    <div class="guide-box">
        <div style="font-size:12px;font-weight:700;color:#3b82f6;margin-bottom:4px">{title}</div>
        <div style="font-size:11px;color:#64748b;line-height:1.6">{body}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚠️ <b>Disclaimer:</b> Educational purposes only. Data from Yahoo Finance via yfinance.
    Nothing here is financial advice. Consult a licensed financial advisor before investing.
    Past performance does not guarantee future results.
</div>
""", unsafe_allow_html=True)
