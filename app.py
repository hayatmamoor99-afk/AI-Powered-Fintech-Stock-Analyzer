import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from scipy.stats import skew, kurtosis
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="MASIS AI",
    page_icon="📈",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

.main {
    background-color:#f8fafc;
}

.block-container {
    padding-top:1rem;
}

.title{
    text-align:center;
    color:#0f172a;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#64748b;
    font-size:18px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

.metric-box{
    background:white;
    padding:15px;
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.markdown(
"""
<div class='title'>
MASIS AI
</div>
<div class='subtitle'>
Market Analysis & Strategic Intelligence System
</div>
""",
unsafe_allow_html=True
)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

with st.sidebar:

    st.header("📚 Learning Center")

    st.write("""
### Histogram
Shows distribution of returns.

### CAGR
Annual growth rate.

### Volatility
Measures risk.

### Sharpe Ratio
Risk-adjusted return.

### Monte Carlo
Future simulation based prediction.

### Probability of Gain
Chance that price increases in future.
""")

    st.info("""
Designed by Mamoor Hayat

All Rights Reserved ©
""")

# ------------------------------------------------
# USER INPUT
# ------------------------------------------------

ticker = st.text_input(
    "Enter Stock / Coin Symbol",
    "AAPL"
)

# ------------------------------------------------
# DOWNLOAD DATA
# ------------------------------------------------

if ticker:

    try:

        data = yf.download(
            ticker,
            period="5y",
            auto_adjust=True
        )

        if data.empty:
            st.error("Ticker not found.")
            st.stop()

        close = data["Close"]

        st.success(f"Data loaded successfully for {ticker}")

        # ----------------------------------------
        # INVESTMENT ANALYSIS
        # ----------------------------------------

        st.header("💰 Investment Growth")

        periods = {
            "1 Year":252,
            "2 Years":504,
            "3 Years":756,
            "4 Years":1008,
            "5 Years":1260
        }

        investment_results=[]

        current_price=close.iloc[-1]

        for label,days in periods.items():

            if len(close)>=days:

                old_price=close.iloc[-days]

                value=(100/current_price)*current_price

                growth=(current_price/old_price)*100

                investment_results.append(
                    [label,100,round(growth,2)]
                )

        invest_df=pd.DataFrame(
            investment_results,
            columns=[
                "Period",
                "Investment",
                "Current Value"
            ]
        )

        st.dataframe(
            invest_df,
            use_container_width=True
        )

        # ----------------------------------------
        # HISTOGRAMS
        # ----------------------------------------

        st.header("📊 Return Histograms")

        year_map = {
            "1 Year":252,
            "2 Years":504,
            "3 Years":756,
            "4 Years":1008,
            "5 Years":1260
        }

        for year_name,days in year_map.items():

            if len(close)>=days:

                subset=close.tail(days)

                returns=subset.pct_change().dropna()*100

                colors=np.where(
                    returns>=0,
                    "Profit",
                    "Loss"
                )

                hist_df=pd.DataFrame({
                    "Return":returns,
                    "Type":colors
                })

                fig=px.histogram(
                    hist_df,
                    x="Return",
                    color="Type",
                    nbins=50,
                    title=f"{ticker} - {year_name}",
                    barmode="overlay"
                )

                fig.update_layout(
                    height=500
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # ----------------------------------------
        # RISK METRICS
        # ----------------------------------------

        returns=close.pct_change().dropna()

        cagr=(
            (close.iloc[-1]/close.iloc[0])
            **(1/5)-1
        )*100

        volatility=returns.std()*np.sqrt(252)*100

        sharpe=(returns.mean()/returns.std())*np.sqrt(252)

        st.header("📈 Advanced Analytics")

        col1,col2,col3=st.columns(3)

        col1.metric(
            "CAGR %",
            round(cagr,2)
        )

        col2.metric(
            "Volatility %",
            round(volatility,2)
        )

        col3.metric(
            "Sharpe Ratio",
            round(sharpe,2)
        )

        # ----------------------------------------
        # MONTE CARLO
        # ----------------------------------------

        st.header("🔮 Future Forecast")

        simulations=10000
        horizon=252

        mu=returns.mean()
        sigma=returns.std()

        last_price=close.iloc[-1]

        ending_prices=[]

        for _ in range(simulations):

            future_returns=np.random.normal(
                mu,
                sigma,
                horizon
            )

            price=last_price

            for r in future_returns:
                price*=1+r

            ending_prices.append(price)

        ending_prices=np.array(
            ending_prices
        )

        gain_prob=(
            ending_prices>last_price
        ).mean()*100

        loss_prob=100-gain_prob

        st.metric(
            "Probability of Gain",
            f"{gain_prob:.2f}%"
        )

        st.metric(
            "Probability of Loss",
            f"{loss_prob:.2f}%"
        )

        # ----------------------------------------
        # SIGNAL
        # ----------------------------------------

        st.header("🤖 AI Recommendation")

        score=0

        if cagr>10:
            score+=1

        if sharpe>1:
            score+=1

        if gain_prob>60:
            score+=1

        if volatility<40:
            score+=1

        if score>=3:

            st.success("""
BUY

Strong growth profile with
favorable probability metrics.
""")

        else:

            st.warning("""
WAIT

Risk-return profile not
strong enough currently.
""")

        # ----------------------------------------
        # COPYRIGHT
        # ----------------------------------------

        st.markdown(
        """
        <div class='footer'>
        Designed By Mamoor Hayat<br>
        © 2026 MASIS AI
        All Rights Reserved
        </div>
        """,
        unsafe_allow_html=True
        )

    except Exception as e:

        st.error(str(e))
