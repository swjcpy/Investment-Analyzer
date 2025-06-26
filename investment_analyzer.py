# investment_analyzer_v2.py
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

st.set_page_config(page_title="Investment Analyzer", layout="wide")
st.title("📊 Investment Analyzer")

# --- Sidebar Input ---
st.sidebar.header("Portfolio Input")
uploaded_file = st.sidebar.file_uploader("Upload your portfolio CSV", type=["csv"])
st.sidebar.markdown("Example format:")
st.sidebar.code("Ticker,Value\nAAPL,30000\nMSFT,30000\nVOO,40000")

# --- Default Example Portfolio ---
default_portfolio = pd.DataFrame({
    "Ticker": ["AAPL", "MSFT", "VOO"],
    "Value": [30000, 30000, 40000]
})

# --- Read uploaded file or use default ---
if uploaded_file:
    try:
        portfolio = pd.read_csv(uploaded_file)
        st.success("Portfolio uploaded successfully.")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
else:
    st.info("Using example portfolio.")
    portfolio = default_portfolio.copy()

# --- Calculate Weights from Values ---
if "Value" not in portfolio.columns or portfolio["Value"].sum() == 0:
    st.error("Portfolio must include 'Value' column with non-zero total.")
    st.stop()
portfolio["Weight"] = portfolio["Value"] / portfolio["Value"].sum()

# --- Fetch Data ---
st.subheader("Portfolio Overview")
results = []
price_data = {}
sectors = []
rolling_data = {}
for ticker in portfolio["Ticker"]:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        info = stock.info

        daily_returns = hist["Close"].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))

        sector = info.get("sector", "N/A")
        sectors.append({"Ticker": ticker, "Sector": sector})

        results.append({
            "Ticker": ticker,
            "Name": info.get("shortName", "N/A"),
            "Current Price": info.get("currentPrice", np.nan),
            "1Y Return %": ((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1) * 100,
            "Volatility (Annualized) %": volatility * 100,
            "Sharpe Ratio": sharpe_ratio,
            "PE Ratio": info.get("trailingPE", np.nan),
            "Dividend Yield %": (info.get("dividendYield", 0) or 0) * 100,
            "Market Cap": info.get("marketCap", np.nan),
        })
        price_data[ticker] = hist["Close"]
    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {e}")

# --- Display Table ---
result_df = pd.DataFrame(results)
st.dataframe(result_df)

# --- AI Analysis ---
st.subheader("🧠 AI Portfolio Insight")
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None
if openai_api_key:
    client = OpenAI(
        api_key=openai_api_key
    )
    
    # Default model; you can change to gpt-4 if you have access
    # MODEL = "gpt-4o-mini"
    MODEL = "gpt-4.1-mini"
    if st.button("🧾 Analyze Portfolio with ChatGPT"):
        prompt = f"Please summarize and give recommendations based on this portfolio data:\n\n{result_df.to_string(index=False)}"
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful financial assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1500
            )
            st.write(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
else:
    st.info("OpenAI API key not configured. Please add it to Streamlit secrets to enable AI insights.")

# --- Warnings ---
st.subheader("⚠️ Portfolio Risk Warnings")
warnings = []

# 1. High Single-Stock Exposure
max_weight = portfolio["Weight"].max()
if max_weight > 0.5:
    ticker = portfolio.loc[portfolio["Weight"].idxmax(), "Ticker"]
    warnings.append(f"⚠️ High exposure: {ticker} has more than 50% of your portfolio.")

# 2. Excessive Volatility
vol_threshold = 40  # 40% annualized volatility
for i, row in result_df.iterrows():
    if row["Volatility (Annualized) %"] > vol_threshold:
        warnings.append(f"⚠️ {row['Ticker']} has high volatility: {row['Volatility (Annualized) %']:.2f}%")

# 3. Missing Data
if result_df.isnull().values.any():
    warnings.append("⚠️ Some stock data is missing or could not be retrieved.")

if warnings:
    for warn in warnings:
        st.warning(warn)
else:
    st.success("✅ No major portfolio warnings detected.")



# --- Historical Performance and Sector Allocation Side-by-Side ---
if price_data:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Historical Performance (1Y) vs SPY")
        df_prices = pd.DataFrame(price_data)
        df_norm = df_prices / df_prices.iloc[0]
        df_weighted = df_norm.multiply(portfolio.set_index("Ticker")["Weight"], axis=1)
        df_portfolio = df_weighted.sum(axis=1)

        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="1y")
        spy_norm = spy_hist["Close"] / spy_hist["Close"].iloc[0]

        fig, ax = plt.subplots()
        for col in df_norm.columns:
            ax.plot(df_norm.index, df_norm[col], alpha=0.4, label=col)
        ax.plot(df_portfolio.index, df_portfolio, color="black", linewidth=2, label="Portfolio")
        ax.plot(spy_norm.index, spy_norm, color="orange", linestyle="--", linewidth=2, label="SPY Benchmark")
        ax.set_title("Normalized Price Performance (1Y)")
        ax.set_ylabel("Growth Factor")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("🧭 Sector Allocation")
        sector_df = pd.DataFrame(sectors).merge(portfolio, on="Ticker")
        fig2, ax2 = plt.subplots()
        sns.heatmap(pd.crosstab(sector_df["Sector"], sector_df["Ticker"], values=sector_df["Weight"], aggfunc='sum').fillna(0), annot=True, cmap="Blues", ax=ax2)
        ax2.set_title("Sector vs Ticker Allocation Heatmap")
        st.pyplot(fig2)

# --- 📉 Max Drawdown and Time to Recovery ---
st.subheader("📉 Max Drawdown Analysis")
roll_max = df_portfolio.cummax()
drawdown = (df_portfolio - roll_max) / roll_max
max_drawdown = drawdown.min()
time_to_recovery = (drawdown == 0).astype(int).diff().eq(1).sum()
st.write(f"**Max Drawdown:** {max_drawdown:.2%}")
st.write(f"**Approx. Time to Recovery Points:** {int(time_to_recovery)}")

fig_dd, ax_dd = plt.subplots()
ax_dd.plot(drawdown, color='red')
ax_dd.set_title("Portfolio Drawdown Over Time")
ax_dd.set_ylabel("Drawdown (%)")
st.pyplot(fig_dd)

# --- 📈 Rolling Sharpe & Volatility ---
st.subheader("📈 Rolling Metrics")
for ticker, df_roll in rolling_data.items():
    fig_roll, ax_roll = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax_roll[0].plot(df_roll.index, df_roll["Rolling Sharpe"], label=f"{ticker} Rolling Sharpe")
    ax_roll[0].legend()
    ax_roll[0].set_ylabel("Sharpe")
    ax_roll[1].plot(df_roll.index, df_roll["Rolling Volatility"], label=f"{ticker} Rolling Volatility", color="orange")
    ax_roll[1].legend()
    ax_roll[1].set_ylabel("Volatility")
    ax_roll[1].set_xlabel("Date")
    fig_roll.suptitle(f"Rolling Metrics: {ticker}")
    st.pyplot(fig_roll)

# --- Monte Carlo Simulation ---
st.subheader("📈 Monte Carlo Simulation: Future Portfolio Projection")
initial_value = st.number_input("Initial Portfolio Value ($)", value=portfolio['Value'].sum(), step=1000)
years = st.slider("Years to Simulate", 5, 50, 30)
sims = st.slider("Number of Simulations", 100, 2000, 500)
target_value = st.number_input("Target Portfolio Value ($)", value=2000000, step=100000)
annual_contribution = st.number_input("Annual Contribution ($)", value=0, step=1000)
annual_withdrawal = st.number_input("Annual Withdrawal ($)", value=0, step=1000)

expected_return = df_portfolio.pct_change().mean() * 252
volatility = df_portfolio.pct_change().std() * np.sqrt(252)

results_mc = np.zeros((sims, years))
final_values = []
for i in range(sims):
    values = [initial_value]
    for _ in range(years):
        growth = np.random.normal(expected_return, volatility)
        new_value = values[-1] * (1 + growth) + annual_contribution - annual_withdrawal
        values.append(max(new_value, 0))
    results_mc[i, :] = values[1:]
    final_values.append(values[-1])

fig3, ax3 = plt.subplots()
ax3.plot(results_mc.T, alpha=0.05, color="blue")
percentiles = np.percentile(results_mc, [10, 25, 50, 75, 90], axis=0)
labels = ["10th", "25th", "Median", "75th", "90th"]
colors = ["#d73027", "#fc8d59", "#4575b4", "#91bfdb", "#fee090"]
for i, label in enumerate(labels):
    ax3.plot(percentiles[i], label=f"{label} Percentile", linewidth=2, color=colors[i])

probability_hit = np.mean(np.array(final_values) >= target_value) * 100
ax3.set_title(f"Monte Carlo Simulation of Portfolio Value\nProbability of Reaching ${target_value:,.0f}: {probability_hit:.2f}%")
ax3.set_xlabel("Year")
ax3.set_ylabel("Portfolio Value ($)")
ax3.grid(True)
ax3.legend()
st.pyplot(fig3)

# --- Download Combined Report ---
st.subheader("📥 Download Data")
report_df = pd.DataFrame(results)
st.download_button("Download CSV Report", report_df.to_csv(index=False), file_name="portfolio_report.csv")
