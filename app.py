import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from scipy.stats import skew, kurtosis
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="FinSight AI",
    page_icon="📈",
    layout="wide"
)

# ---------------------------
# CUSTOM CSS
# ---------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stMetric {
    background-color: #1C2333;
    padding:10px;
    border-radius:15px;
}

h1,h2,h3 {
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER
# ---------------------------

st.title("🚀 FinSight AI")
st.subheader("Advanced Stock Analysis Platform")

# ---------------------------
# SIDEBAR EDUCATION PANEL
# ---------------------------

with st.sidebar:

    st.header("📚 Learn")

    st.markdown("""
### Histogram
Shows return distribution.

### Volatility
Measures price fluctuations.

### Standard Deviation
Measures risk.

### Skewness
Shows return asymmetry.

### Kurtosis
Shows extreme events.

### Monte Carlo
Forecasts possible future prices.
""")

# ---------------------------
# STOCK SEARCH
# ---------------------------

ticker = st.text_input(
    "Enter Stock Symbol",
    value="NVDA"
)

if ticker:

    try:

        stock = yf.Ticker(ticker)

        data = stock.history(period="5y")

        if data.empty:
            st.error("No data found.")
            st.stop()

        current_price = data["Close"].iloc[-1]

        st.success(f"Current Price: ${current_price:.2f}")

        # --------------------
        # PRICE CHART
        # --------------------

        fig = px.line(
            data,
            y="Close",
            title=f"{ticker} Price History"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # --------------------
        # INVESTMENT RETURNS
        # --------------------

        st.header("💰 Investment Growth")

        periods = {
            "1 Year":"1y",
            "2 Years":"2y",
            "3 Years":"3y",
            "4 Years":"4y",
            "5 Years":"5y"
        }

        growth_rows = []

        for label, period in periods.items():

            hist = stock.history(period=period)

            if len(hist) > 0:

                start_price = hist["Close"].iloc[0]
                end_price = hist["Close"].iloc[-1]

                value = 100 * (end_price / start_price)

                profit = value - 100

                growth_rows.append([
                    label,
                    round(value,2),
                    round(profit,2)
                ])

        growth_df = pd.DataFrame(
            growth_rows,
            columns=[
                "Period",
                "Current Value of $100",
                "Profit/Loss"
            ]
        )

        st.dataframe(
            growth_df,
            use_container_width=True
        )

# ==========================================
# BEAUTIFUL HISTOGRAM ANALYSIS
# ==========================================

st.header("📊 Return Distribution Intelligence")

years = [1, 2, 3, 4, 5]

for year in years:

    st.markdown("---")

    st.subheader(f"📈 {year}-Year Return Analysis")

    hist = stock.history(period=f"{year}y")

    returns = (
        hist["Close"]
        .pct_change()
        .dropna()
    )

    # -----------------------------
    # STATISTICS
    # -----------------------------

    mean_return = returns.mean() * 100
    std_return = returns.std() * 100
    skew_value = skew(returns)
    kurt_value = kurtosis(returns)

    # -----------------------------
    # BEAUTIFUL HISTOGRAM
    # -----------------------------

    fig = px.histogram(
        returns * 100,
        nbins=50,
        title=f"{year}-Year Daily Return Distribution",
        labels={
            "value": "Daily Return (%)",
            "count": "Frequency"
        },
        marginal="box"
    )

    fig.update_layout(
        height=500,
        template="plotly_dark",
        title_x=0.5,
        font=dict(size=14)
    )

    fig.add_vline(
        x=mean_return,
        line_dash="dash",
        line_width=3,
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------
    # HUMANIZED METRICS
    # -----------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            f"""
🟢 Average Daily Return

{mean_return:.2f}%

Meaning:
On average this stock moved
{mean_return:.2f}% per day.
"""
        )

        risk_text = ""

        if std_return < 1:
            risk_text = "🟢 Low Volatility"

        elif std_return < 3:
            risk_text = "🟡 Moderate Volatility"

        else:
            risk_text = "🔴 High Volatility"

        st.warning(
            f"""
{risk_text}

Standard Deviation:
{std_return:.2f}%

Meaning:
This indicates how much
the stock's returns fluctuate.
"""
        )

    with col2:

        if skew_value > 0:

            skew_text = """
🟢 Positive Skew

Historically this stock has
experienced larger positive
surprises than negative ones.
"""

        elif skew_value < 0:

            skew_text = """
🔴 Negative Skew

Historically this stock has
experienced larger downside
surprises.
"""

        else:

            skew_text = """
⚪ Neutral Skew

Returns are fairly balanced.
"""

        st.info(skew_text)

        if kurt_value > 3:

            kurt_text = """
🔴 High Kurtosis

Most days are normal but
occasional extreme movements
have occurred.
"""

        else:

            kurt_text = """
🟢 Normal Kurtosis

Price movements are generally
more predictable.
"""

        st.error(kurt_text)

    # -----------------------------
    # AI SUMMARY
    # -----------------------------

    if (
        mean_return > 0
        and std_return < 3
        and skew_value > 0
    ):

        summary = """
🤖 AI Analysis

This stock shows healthy historical
returns, manageable volatility and
favorable upside characteristics.

Suitable for investors seeking
growth with moderate risk.
"""

    elif mean_return > 0:

        summary = """
🤖 AI Analysis

The stock has generated positive
returns historically but investors
should monitor volatility closely.
"""

    else:

        summary = """
🤖 AI Analysis

Historical performance has been
weak and risk-adjusted returns
appear less attractive.
"""

    st.info(summary)

        # --------------------
        # BUY / SELL ENGINE
        # --------------------

        st.header("🧠 AI Probability Engine")

        returns = (
            data["Close"]
            .pct_change()
            .dropna()
        )

        positive_days = (
            returns > 0
        ).mean()

        probability_up = (
            positive_days * 100
        )

        annual_return = (
            (
                current_price /
                data["Close"].iloc[0]
            ) - 1
        ) * 100

        volatility = (
            returns.std() * np.sqrt(252)
        ) * 100

        score = 50

        if annual_return > 20:
            score += 15

        if probability_up > 55:
            score += 15

        if volatility < 40:
            score += 10

        score = min(score,100)

        c1,c2,c3 = st.columns(3)

        c1.metric(
            "Probability Up",
            f"{probability_up:.1f}%"
        )

        c2.metric(
            "Annual Return",
            f"{annual_return:.1f}%"
        )

        c3.metric(
            "AI Score",
            score
        )

        if score >= 80:
            st.success(
                "🟢 Strong Buy Candidate"
            )

        elif score >= 65:
            st.info(
                "🔵 Moderate Buy Candidate"
            )

        elif score >= 50:
            st.warning(
                "🟡 Hold / Wait"
            )

        else:
            st.error(
                "🔴 Avoid Currently"
            )

        # --------------------
        # MONTE CARLO
        # --------------------

        st.header("🔮 Monte Carlo Forecast")

        returns = (
            data["Close"]
            .pct_change()
            .dropna()
        )

        simulations = 1000
        days = 252

        mean = returns.mean()
        std = returns.std()

        results = []

        for _ in range(simulations):

            prices = [current_price]

            for _ in range(days):

                shock = np.random.normal(
                    mean,
                    std
                )

                prices.append(
                    prices[-1] *
                    (1 + shock)
                )

            results.append(
                prices[-1]
            )

        expected = np.mean(results)

        best = np.percentile(
            results,
            95
        )

        worst = np.percentile(
            results,
            5
        )

        col1,col2,col3 = st.columns(3)

        col1.metric(
            "Expected Price",
            f"${expected:.2f}"
        )

        col2.metric(
            "Best Case",
            f"${best:.2f}"
        )

        col3.metric(
            "Worst Case",
            f"${worst:.2f}"
        )

        fig = px.histogram(
            results,
            nbins=50,
            title="Future Price Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except Exception as e:

        st.error(e)
