import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import random
import json

# ---------------- Mock/Helper Functions ----------------
def generate_health_score():
    """Mock health score."""
    return {
        "score": random.randint(60, 95),
        "details": {
            "Income Stability": random.randint(50, 100),
            "Savings Habits": random.randint(50, 100),
            "Debt Management": random.randint(50, 100),
            "Investment Diversification": random.randint(50, 100),
            "Emergency Preparedness": random.randint(50, 100),
            "Spending Efficiency": random.randint(50, 100),
            "Credit Health": random.randint(50, 100),
            "Overall Resilience": random.randint(50, 100),
        }
    }

def run_simulation(params):
    """Mock future simulation."""
    years = params.get("years", 10)
    inflation = params.get("inflation", 0.02)
    growth = params.get("growth", 0.05)
    initial_wealth = params.get("initial_wealth", 10000)
    monthly_savings = params.get("monthly_savings", 500)
    
    if growth == inflation:
        final = initial_wealth + monthly_savings * 12 * years
        projections = [initial_wealth + monthly_savings * 12 * y for y in range(1, years+1)]
    else:
        factor = (1 + growth - inflation)
        final = initial_wealth * (factor ** years) + monthly_savings * 12 * ((factor ** years) - 1) / (growth - inflation)
        projections = [initial_wealth * (factor ** y) + monthly_savings * 12 * ((factor ** y) - 1) / (growth - inflation) for y in range(1, years+1)]
    return {
        "final_wealth": round(final, 2),
        "projections": [round(p, 2) for p in projections],
        "confidence": random.uniform(0.7, 0.95),
        "risk_metrics": {"Volatility": round(random.uniform(0.1, 0.3), 2)}
    }

def analyze_investment(ticker, period="1y"):
    """Fetch real data from yfinance and compute basic metrics."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return {"error": "No data found for ticker"}
        current_price = hist['Close'].iloc[-1]
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # annualized
        sharpe = (returns.mean() * 252) / volatility if volatility != 0 else 0
        # simple recommendation based on Sharpe
        if sharpe > 1.0:
            rec = "Buy"
        elif sharpe > 0.5:
            rec = "Hold"
        else:
            rec = "Sell"
        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "annual_return": round(returns.mean() * 252, 4),
            "volatility": round(volatility, 4),
            "sharpe_ratio": round(sharpe, 2),
            "recommendation": rec,
            "explanation": f"Based on {period} of data, this asset has a Sharpe ratio of {sharpe:.2f}. {rec}."
        }
    except Exception as e:
        return {"error": str(e)}

def detect_scam(text):
    """Mock scam detection."""
    suspicious = ["urgent", "click here", "verify account", "win", "prize", "inheritance", "bank account", "password", "phishing"]
    found = [kw for kw in suspicious if kw in text.lower()]
    prob = min(0.95, len(found) * 0.1 + random.uniform(0, 0.1))
    return {
        "is_scam": prob > 0.6,
        "probability": prob,
        "detected_indicators": found,
        "explanation": f"Detected suspicious terms: {', '.join(found) if found else 'none'}. Risk: {prob*100:.1f}%"
    }

def get_ai_response(message):
    """Mock AI assistant."""
    responses = [
        "Based on your spending, you could save $200/month by reducing dining out.",
        "Your savings rate is excellent. Consider investing more in a diversified ETF.",
        "Paying off high-interest credit card should be your top priority.",
        "You are on track to reach your retirement goal by age 62."
    ]
    return random.choice(responses)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="FinTwin AI", layout="wide")
st.title("🧠 FinTwin AI – Your Personal Financial Twin")

# Sidebar Navigation
menu = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Financial Health", "Simulator", "Investment Analysis", "Scam Detection", "AI Assistant", "Goals"]
)

# In-memory goals storage (demo)
if "goals" not in st.session_state:
    st.session_state.goals = []

# ---------------- Dashboard ----------------
if menu == "Dashboard":
    st.header("📊 Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Balance", "$12,450")
    with col2:
        st.metric("Monthly Income", "$5,200")
    with col3:
        st.metric("Monthly Expenses", "$3,800")
    st.subheader("Recent Transactions")
    data = pd.DataFrame({
        "Date": pd.date_range(end=datetime.today(), periods=5),
        "Description": ["Groceries", "Rent", "Coffee", "Salary", "Utilities"],
        "Amount": [-150, -1200, -5, 5200, -200],
        "Category": ["Food", "Housing", "Food", "Income", "Bills"]
    })
    st.dataframe(data, use_container_width=True)

# ---------------- Financial Health ----------------
elif menu == "Financial Health":
    st.header("💚 Financial Health Score")
    if st.button("Refresh Score"):
        health = generate_health_score()
        st.session_state.health = health
    if "health" in st.session_state:
        health = st.session_state.health
        st.metric("Overall Score", f"{health['score']}/100")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Score Breakdown")
            for k, v in health["details"].items():
                st.progress(v/100, text=f"{k}: {v}%")
        with col2:
            st.subheader("Recommendations")
            st.write("• Increase emergency fund to cover 6 months of expenses.")
            st.write("• Consider diversifying investments into international markets.")
            st.write("• Reduce dining out by 20% to boost savings.")
    else:
        st.info("Click 'Refresh Score' to generate your Financial Health report.")

# ---------------- Simulator ----------------
elif menu == "Simulator":
    st.header("🔮 Future Financial Simulator")
    with st.form("sim_form"):
        col1, col2 = st.columns(2)
        with col1:
            initial = st.number_input("Initial Wealth ($)", value=10000, step=1000)
            monthly = st.number_input("Monthly Savings ($)", value=500, step=50)
            years = st.slider("Time Horizon (years)", 1, 30, 10)
        with col2:
            growth = st.slider("Expected Annual Return (%)", 0, 20, 5) / 100
            inflation = st.slider("Inflation Rate (%)", 0, 10, 2) / 100
        submitted = st.form_submit_button("Run Simulation")
    
    if submitted:
        params = {
            "initial_wealth": initial,
            "monthly_savings": monthly,
            "years": years,
            "growth": growth,
            "inflation": inflation
        }
        result = run_simulation(params)
        st.success(f"Projected Wealth after {years} years: **${result['final_wealth']:,.2f}**")
        st.info(f"Confidence Level: {result['confidence']*100:.1f}%")
        # Plot projections
        df = pd.DataFrame({
            "Year": list(range(1, years+1)),
            "Projected Wealth": result["projections"]
        })
        fig = px.line(df, x="Year", y="Projected Wealth", title="Wealth Projection Over Time")
        st.plotly_chart(fig, use_container_width=True)
        st.json(result["risk_metrics"])

# ---------------- Investment Analysis ----------------
elif menu == "Investment Analysis":
    st.header("📈 Investment Analytics")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", value="AAPL")
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    if st.button("Analyze"):
        with st.spinner("Fetching data..."):
            result = analyze_investment(ticker, period)
        if "error" in result:
            st.error(result["error"])
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${result['current_price']}")
            col2.metric("Annual Return", f"{result['annual_return']*100:.2f}%")
            col3.metric("Volatility", f"{result['volatility']*100:.2f}%")
            st.metric("Sharpe Ratio", result['sharpe_ratio'])
            st.metric("Recommendation", result['recommendation'])
            st.info(result['explanation'])
            # Plot price history
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                fig = px.line(hist, x=hist.index, y="Close", title=f"{ticker} Price History")
                st.plotly_chart(fig, use_container_width=True)

# ---------------- Scam Detection ----------------
elif menu == "Scam Detection":
    st.header("🛡️ AI Scam Detection")
    text = st.text_area("Paste suspicious email, SMS, or message here:")
    if st.button("Check"):
        if text.strip():
            result = detect_scam(text)
            if result["is_scam"]:
                st.error("⚠️ This message appears to be a scam!")
            else:
                st.success("✅ This message seems safe.")
            st.write(f"**Risk Probability:** {result['probability']*100:.1f}%")
            st.write(f"**Detected Indicators:** {', '.join(result['detected_indicators']) if result['detected_indicators'] else 'None'}")
            st.write(f"**Explanation:** {result['explanation']}")
        else:
            st.warning("Please enter some text to analyze.")

# ---------------- AI Assistant ----------------
elif menu == "AI Assistant":
    st.header("💬 AI Financial Assistant")
    user_question = st.text_input("Ask me anything about your finances:")
    if st.button("Ask"):
        if user_question.strip():
            response = get_ai_response(user_question)
            st.write("**FinTwin AI:**", response)
            st.caption("(This is a mock response. Replace with real OpenAI integration.)")
        else:
            st.warning("Please enter a question.")

# ---------------- Goals ----------------
elif menu == "Goals":
    st.header("🎯 Smart Goal Management")
    with st.form("goal_form"):
        goal_name = st.text_input("Goal Name")
        target = st.number_input("Target Amount ($)", min_value=0.0, step=100.0)
        date = st.date_input("Target Date", min_value=datetime.today())
        category = st.selectbox("Category", ["Retirement", "House", "Education", "Travel", "Emergency Fund", "Other"])
        if st.form_submit_button("Add Goal"):
            st.session_state.goals.append({
                "name": goal_name,
                "target": target,
                "date": date,
                "category": category,
                "progress": random.uniform(0, 100)
            })
            st.success("Goal added!")
    
    if st.session_state.goals:
        df = pd.DataFrame(st.session_state.goals)
        st.dataframe(df, use_container_width=True)
        # progress bars
        for i, row in df.iterrows():
            st.progress(row["progress"]/100, text=f"{row['name']} - {row['progress']:.0f}% complete")
    else:
        st.info("No goals set yet. Add one above!")
