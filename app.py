import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="FinScope Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background: #f4f6ff; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1a1a2e,#0f3460) !important;
}
section[data-testid="stSidebar"] * { color: #ddd6fe !important; }
div[data-testid="metric-container"] {
    background: #fff; border: 1px solid #e0e7ff;
    border-radius: 14px; padding: 16px;
    box-shadow: 0 2px 12px rgba(99,102,241,.08);
}
.stButton > button {
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    color: white !important; border: none;
    border-radius: 10px; font-weight: 700;
    box-shadow: 0 4px 14px rgba(99,102,241,.35);
}
.hdr {
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    color: white; padding: 10px 18px; border-radius: 10px;
    font-weight: 700; font-size: 1rem; margin: 18px 0 10px;
}
.buy   { background:#d1fae5; border:2px solid #10b981; border-radius:14px; padding:18px; text-align:center; }
.sell  { background:#fee2e2; border:2px solid #ef4444; border-radius:14px; padding:18px; text-align:center; }
.hold  { background:#fef3c7; border:2px solid #f59e0b; border-radius:14px; padding:18px; text-align:center; }
.footer{
    background:linear-gradient(135deg,#1a1a2e,#0f3460);
    color:#c4b5fd; padding:16px; border-radius:12px;
    text-align:center; margin-top:36px; font-size:.85rem;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_data(ticker):
    tk = yf.Ticker(ticker)
    df = tk.history(period="5y", auto_adjust=True)
    info = {}
    try:
        info = tk.info
    except Exception:
        pass
    return df, info

def sim100(df):
    if df is None or len(df) < 2:
        return None, None, None
    s = df["Close"].iloc[0]
    e = df["Close"].iloc[-1]
    final  = round(100 / s * e, 2)
    profit = round(final - 100, 2)
    pct    = round((profit / 100) * 100, 2)
    return final, profit, pct

def signals(df):
    c = df["Close"]
    cur = c.iloc[-1]
    res = {}
    res["MA20"]  = "Bullish" if cur > c.rolling(20).mean().iloc[-1]  else "Bearish"
    res["MA50"]  = "Bullish" if cur > c.rolling(50).mean().iloc[-1]  else "Bearish"
    res["MA200"] = ("Bullish" if cur > c.rolling(200).mean().iloc[-1] else "Bearish") if len(c)>=200 else "N/A"
    d = c.diff()
    rs = d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0).rolling(14).mean())
    rsi = 100 - 100/(1+rs)
    rsi_val = round(rsi.iloc[-1], 1)
    res["RSI"] = rsi_val
    res["RSI_lbl"] = "Oversold ✅" if rsi_val < 30 else ("Overbought ⚠️" if rsi_val > 70 else "Normal")
    macd  = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig   = macd.ewm(span=9).mean()
    res["MACD"] = "Bullish" if macd.iloc[-1] > sig.iloc[-1] else "Bearish"
    bb_m  = c.rolling(20).mean()
    bb_s  = c.rolling(20).std()
    if cur < (bb_m - 2*bb_s).iloc[-1]:   res["BB"] = "Bullish"
    elif cur > (bb_m + 2*bb_s).iloc[-1]: res["BB"] = "Bearish"
    else:                                  res["BB"] = "Neutral"
    score = sum([
        res["MA20"]=="Bullish", res["MA50"]=="Bullish",
        res["MA200"]=="Bullish", res["MACD"]=="Bullish",
        rsi_val<30, rsi_val<50,
        res["BB"]=="Bullish",
    ])
    res["score"] = score   # out of 7
    res["vol"]   = round(c.pct_change().std() * (252**0.5) * 100, 1)
    return res

def histogram(df, ticker, label):
    df = df.copy()
    df["ret"]  = df["Close"].pct_change()*100
    df["col"]  = df["ret"].apply(lambda x: "#10b981" if x >= 0 else "#ef4444")
    mx = df["ret"].abs().max() or 1
    df["op"]   = df["ret"].abs().apply(lambda x: 0.4 + 0.6*(x/mx))
    ma = df["Close"].rolling(20).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index, y=df["Close"],
        marker=dict(color=df["col"], opacity=df["op"],
                    line=dict(width=0.2, color="rgba(0,0,0,0.05)")),
        name="Close",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:.2f}<br>%{customdata:.2f}%<extra></extra>",
        customdata=df["ret"],
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=ma, mode="lines", name="20d MA",
        line=dict(color="#6366f1", width=1.8, dash="dot"),
    ))
    fig.update_layout(
        title=f"<b>{ticker} — {label}</b>",
        xaxis=dict(
            fixedrange=False, gridcolor="rgba(99,102,241,.1)",
            rangeslider=dict(visible=True, thickness=0.05),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#f0f4ff", activecolor="#6366f1",
            ),
        ),
        yaxis=dict(fixedrange=False, tickprefix="$", gridcolor="rgba(99,102,241,.1)"),
        plot_bgcolor="#fafbff", paper_bgcolor="#fff",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=55, r=15, t=60, b=55),
        height=430, hovermode="x unified",
    )
    return fig

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 FinScope Pro")
    st.markdown("---")
    ticker = st.text_input("Stock / Crypto ticker", placeholder="e.g. AAPL  BTC-USD").strip().upper()
    go_btn = st.button("🚀 Analyze", use_container_width=True)
    st.markdown("---")
    st.markdown("""
**Stocks:** `AAPL` `TSLA` `MSFT` `NVDA` `AMZN`  
**Crypto:** `BTC-USD` `ETH-USD` `SOL-USD`  
**Search more:** [finance.yahoo.com](https://finance.yahoo.com)
    """)
    st.markdown("---")
    with st.expander("📘 Glossary"):
        st.markdown("""
**RSI** — momentum 0–100. <30 = oversold, >70 = overbought.  
**MACD** — trend momentum from two EMAs.  
**Moving Avg** — smoothed price trend line.  
**Bollinger** — volatility bands ±2σ.  
**Bull/Bear** — rising / falling market.
        """)
    st.markdown("""
<div style="text-align:center;font-size:.75rem;color:#a78bfa;margin-top:12px;">
Designed by <b>Mamoor Hayat</b><br>© All Rights Reserved
</div>""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1a1a2e,#0f3460);
            padding:24px 32px;border-radius:18px;margin-bottom:24px;
            box-shadow:0 6px 28px rgba(99,102,241,.2);">
  <h2 style="color:#a78bfa;margin:0 0 4px;">📈 FinScope Pro — Stock & Crypto Analyzer</h2>
  <p style="color:#c4b5fd;margin:0;font-size:.9rem;">
    Technical analysis · Investment simulation · Buy/Sell signals ·
    Designed by <b>Mamoor Hayat</b> · © All Rights Reserved
  </p>
</div>
""", unsafe_allow_html=True)

# ── Landing ───────────────────────────────────────────────────
if not go_btn or not ticker:
    c1, c2, c3 = st.columns(3)
    for col, ic, ttl, dsc in [
        (c1,"📊","Price Histograms","1–5 yr bars, green=up, red=down, scroll to zoom"),
        (c2,"💰","$100 Simulation","What if you invested $100 at the start of each period?"),
        (c3,"🤖","Buy/Hold/Sell","6 indicators → probability-based recommendation"),
    ]:
        col.markdown(f"""
<div style="background:#fff;border:1px solid #e0e7ff;border-radius:14px;
            padding:22px;text-align:center;box-shadow:0 3px 14px rgba(99,102,241,.07);">
<div style="font-size:2.4rem;">{ic}</div>
<h4 style="color:#6366f1;margin:8px 0 4px;">{ttl}</h4>
<p style="color:#64748b;font-size:.85rem;margin:0;">{dsc}</p>
</div>""", unsafe_allow_html=True)
    st.info("👈 Enter a ticker in the sidebar and press **🚀 Analyze**")
    st.markdown("<div class='footer'><b>FinScope Pro</b> · Designed by <b>Mamoor Hayat</b> · © All Rights Reserved · Educational use only · Not financial advice</div>", unsafe_allow_html=True)
    st.stop()

# ── Fetch ─────────────────────────────────────────────────────
with st.spinner(f"Loading {ticker} …"):
    df5, info = get_data(ticker)

if df5 is None or df5.empty:
    st.error(f"No data for **{ticker}**. Check the symbol and try again.")
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────
name   = info.get("longName", ticker)
cur_p  = df5["Close"].iloc[-1]
prev_p = df5["Close"].iloc[-2]
chg    = (cur_p - prev_p) / prev_p * 100
mktcap = info.get("marketCap")
mcs    = f"${mktcap/1e9:.1f}B" if mktcap else "—"
hi52   = df5["Close"].tail(252).max()
lo52   = df5["Close"].tail(252).min()

st.markdown(f"<div class='hdr'>🏢 {name}  ({ticker})</div>", unsafe_allow_html=True)
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Price",    f"${cur_p:.2f}", f"{chg:+.2f}%")
k2.metric("Mkt Cap",  mcs)
k3.metric("Sector",   info.get("sector", info.get("category","—")))
k4.metric("52W High", f"${hi52:.2f}")
k5.metric("52W Low",  f"${lo52:.2f}")

# ── Period slices ─────────────────────────────────────────────
today = datetime.now()
periods = {
    "5 Years": df5,
    "4 Years": df5[df5.index >= today - timedelta(days=4*365)],
    "3 Years": df5[df5.index >= today - timedelta(days=3*365)],
    "2 Years": df5[df5.index >= today - timedelta(days=2*365)],
    "1 Year":  df5[df5.index >= today - timedelta(days=365)],
}

# ── Section 1 — Histograms ────────────────────────────────────
st.markdown("<div class='hdr'>📊 Price Histograms — Scroll to Zoom · Drag to Pan</div>", unsafe_allow_html=True)
st.caption("🟢 Green = up day  🔴 Red = down day  Deeper = bigger move  Purple dotted = 20-day MA")

tabs = st.tabs(["5 Years","4 Years","3 Years","2 Years","1 Year"])
for tab, (lbl, dfp) in zip(tabs, periods.items()):
    with tab:
        if dfp.empty:
            st.warning(f"Not enough data for {lbl}.")
        else:
            st.plotly_chart(
                histogram(dfp, ticker, lbl),
                use_container_width=True,
                config={"scrollZoom": True, "displayModeBar": True,
                        "toImageButtonOptions": {"format":"png","scale":2}},
            )

# ── Section 2 — $100 simulation ───────────────────────────────
st.markdown("<div class='hdr'>💰 $100 Investment Simulation</div>", unsafe_allow_html=True)
st.caption("How much would $100 invested at the start of each period be worth today?")

sim_res = {}
cols = st.columns(5)
for col, (lbl, dfp) in zip(cols, periods.items()):
    f, p, pc = sim100(dfp)
    if f is None: continue
    sim_res[lbl] = {"final":f,"profit":p}
    sign = "🟢" if p >= 0 else "🔴"
    col.metric(f"{sign} {lbl}", f"${f}", f"{'+' if p>=0 else ''}{p} ({pc:+.1f}%)")

if sim_res:
    labels = list(sim_res.keys())
    finals = [v["final"] for v in sim_res.values()]
    profs  = [v["profit"] for v in sim_res.values()]
    clrs   = ["#10b981" if p>=0 else "#ef4444" for p in profs]
    fig_b  = go.Figure(go.Bar(
        x=labels, y=finals, marker_color=clrs,
        text=[f"${v}" for v in finals], textposition="outside",
    ))
    fig_b.add_hline(y=100, line_dash="dash", line_color="#6366f1",
                    annotation_text="$100 invested", annotation_position="right")
    fig_b.update_layout(
        title="$100 Invested → Value Today",
        yaxis=dict(tickprefix="$"),
        plot_bgcolor="#fafbff", paper_bgcolor="#fff",
        height=340, margin=dict(l=50,r=20,t=50,b=40), showlegend=False,
    )
    st.plotly_chart(fig_b, use_container_width=True)

# ── Section 3 — Signals ───────────────────────────────────────
st.markdown("<div class='hdr'>🤖 Technical Signals & Buy / Hold / Sell</div>", unsafe_allow_html=True)

sg    = signals(df5.tail(500))
score = sg["score"]
pct   = round(score/7*100, 1)

if pct >= 65:   verdict, vcls, icon = "BUY",        "buy",  "✅"
elif pct <= 35: verdict, vcls, icon = "AVOID/SELL", "sell", "⛔"
else:           verdict, vcls, icon = "HOLD/WAIT",  "hold", "⚠️"

bar_c = "#10b981" if pct>=65 else "#ef4444" if pct<=35 else "#f59e0b"

c1, c2 = st.columns([1,2])
with c1:
    st.markdown(f"""
<div class='{vcls}'>
  <div style='font-size:1rem;font-weight:600;color:#374151;'>Overall Signal</div>
  <div style='font-size:2rem;font-weight:800;margin:6px 0;'>{icon} {verdict}</div>
  <div style='font-size:.9rem;color:#374151;'>Score: {score}/7 ({pct}%)</div>
  <div style='background:#e0e7ff;border-radius:20px;height:10px;margin-top:8px;'>
    <div style='background:{bar_c};width:{pct}%;height:10px;border-radius:20px;'></div>
  </div>
</div>""", unsafe_allow_html=True)

with c2:
    rows = [
        ["MA 20-day",  sg["MA20"]],
        ["MA 50-day",  sg["MA50"]],
        ["MA 200-day", sg["MA200"]],
        ["MACD",       sg["MACD"]],
        ["RSI",        f"{sg['RSI']} — {sg['RSI_lbl']}"],
        ["Bollinger",  sg["BB"]],
        ["Volatility", f"{sg['vol']}% p.a."],
    ]
    df_sg = pd.DataFrame(rows, columns=["Indicator","Reading"])

    def color_cell(v):
        if "Bullish" in str(v) or "Oversold" in str(v):
            return "background-color:#d1fae5;color:#065f46"
        if "Bearish" in str(v) or "Overbought" in str(v):
            return "background-color:#fee2e2;color:#991b1b"
        return "background-color:#fef3c7;color:#92400e"

    st.dataframe(
        df_sg.style.map(color_cell, subset=["Reading"]),
        hide_index=True, use_container_width=True,
    )

# ── Section 4 — Beginner guide ────────────────────────────────
st.markdown("<div class='hdr'>📘 Beginner's Guide — Everything Explained</div>", unsafe_allow_html=True)

with st.expander("📊 How to read the Histogram"):
    st.markdown("""
Each bar = one trading day's closing price.

| Colour | Meaning |
|--------|---------|
| 🟢 Green | Price went UP vs previous day |
| 🔴 Red | Price went DOWN vs previous day |
| Deeper shade | Bigger price move |
| Purple dotted line | 20-day moving average (trend smoother) |

**Zoom & Pan:** scroll mouse wheel to zoom, click-drag to pan, use range buttons or bottom slider.
    """)

with st.expander("💰 How the $100 simulation works"):
    st.markdown("""
1. Take the opening price at the start of the period.
2. Divide $100 by that price → number of shares bought.
3. Multiply shares × today's price → your portfolio value today.
4. Subtract $100 → your profit or loss.

> ⚠️ Simplified: ignores fees, taxes, dividends. Educational only.
    """)

with st.expander("🤖 How Buy/Hold/Sell is calculated"):
    st.markdown("""
Seven sub-signals are checked. Each bullish signal adds a point. The score is converted to a percentage:

| % Score | Verdict |
|---------|---------|
| ≥ 65% | ✅ BUY — momentum is positive |
| 36–64% | ⚠️ HOLD — mixed signals |
| ≤ 35% | ⛔ AVOID/SELL — momentum is negative |

**RSI** < 30 = oversold (potential buy zone). > 70 = overbought (risky).  
**MACD** above signal line = upward momentum.  
**Moving averages** above price = downtrend and vice versa.  
**Bollinger** near lower band = historically tends to bounce.

> ⚠️ This is educational only — not financial advice. Always do your own research.
    """)

# ── Section 5 — Full chart ────────────────────────────────────
st.markdown("<div class='hdr'>📉 Full 5-Year Price + Volume</div>", unsafe_allow_html=True)
fig_full = go.Figure()
fig_full.add_trace(go.Scatter(
    x=df5.index, y=df5["Close"], mode="lines", name="Close",
    line=dict(color="#6366f1", width=2),
    fill="tozeroy", fillcolor="rgba(99,102,241,.07)",
))
fig_full.add_trace(go.Bar(
    x=df5.index, y=df5["Volume"], name="Volume",
    marker_color="rgba(139,92,246,.2)", yaxis="y2",
))
fig_full.update_layout(
    yaxis=dict(tickprefix="$", fixedrange=False),
    yaxis2=dict(overlaying="y", side="right", showgrid=False, fixedrange=False),
    xaxis=dict(
        fixedrange=False,
        rangeslider=dict(visible=True, thickness=0.05),
        rangeselector=dict(buttons=[
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year",  stepmode="backward"),
            dict(count=3, label="3Y", step="year",  stepmode="backward"),
            dict(step="all", label="5Y"),
        ]),
    ),
    plot_bgcolor="#fafbff", paper_bgcolor="#fff",
    height=400, margin=dict(l=55,r=60,t=30,b=50),
    legend=dict(orientation="h", y=1.04),
)
st.plotly_chart(fig_full, use_container_width=True,
                config={"scrollZoom": True, "displayModeBar": True})

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
  <b>FinScope Pro</b> · Designed & Developed by <b>Mamoor Hayat</b> ·
  © All Rights Reserved · Educational use only · Not financial advice<br>
  <span style='font-size:.75rem;color:#7c3aed;'>Powered by yFinance · Plotly · Streamlit</span>
</div>
""", unsafe_allow_html=True)
