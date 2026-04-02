"""
Stock Price Forecasting Dashboard
====================================
Author: Hemanta sethy
Run:  streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


# ── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── Title ────────────────────────────────────────────────────────
st.title(" Stock Price Prediction & Forecasting Dashboard")
st.markdown("*Finance & Stock Market Analysis Project — Data Analyst Portfolio*")
st.divider()


# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

STOCK_OPTIONS = {
    "TCS (NSE)":             "TCS.NS",
    "Infosys (NSE)":         "INFY.NS",
    "Reliance (NSE)":        "RELIANCE.NS",
    "Apple (NASDAQ)":        "AAPL",
    "Google (NASDAQ)":       "GOOGL",
    "Microsoft (NASDAQ)":   "MSFT",
}

selected_name   = st.sidebar.selectbox("Select Stock", list(STOCK_OPTIONS.keys()))
ticker          = STOCK_OPTIONS[selected_name]
period_map      = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}
selected_period = st.sidebar.selectbox("Historical Period", list(period_map.keys()), index=3)
model_choice    = st.sidebar.radio("Forecasting Model", ["ARIMA", "Prophet", "Both"])
forecast_days   = st.sidebar.slider("Forecast Days", min_value=7, max_value=90, value=30, step=7)
st.sidebar.divider()
st.sidebar.info(" Data sourced from Yahoo Finance via yfinance")


# ── Helper Functions ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False)
    df.reset_index(inplace=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df.dropna()


def calc_metrics(actual, predicted):
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return round(rmse, 2), round(mae, 2), round(mape, 2)


def run_arima_forecast(series: pd.Series, forecast_days: int):
    split       = int(len(series) * 0.80)
    train, test = series[:split], series[split:]
    order       = (5, 1, 0)

    history     = list(train)
    predictions = []
    for obs in test:
        m    = ARIMA(history, order=order).fit()
        yhat = m.forecast(steps=1)[0]
        predictions.append(yhat)
        history.append(obs)

    rmse, mae, mape = calc_metrics(test.values, np.array(predictions))

    final_model  = ARIMA(series, order=order).fit()
    forecast_val = final_model.forecast(steps=forecast_days)

    return predictions, test, rmse, mae, mape, forecast_val


def run_prophet_forecast(df: pd.DataFrame, forecast_days: int):
    prophet_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    prophet_df["y"] = prophet_df["y"].astype(float)

    split      = int(len(prophet_df) * 0.80)
    train_df   = prophet_df[:split]
    test_df    = prophet_df[split:]

    model = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(train_df)

    future_test  = model.make_future_dataframe(periods=len(test_df), freq="B")
    forecast_all = model.predict(future_test)
    test_forecast = forecast_all[forecast_all["ds"].isin(test_df["ds"])]["yhat"].values

    actual  = test_df["y"].values[:len(test_forecast)]
    rmse, mae, mape = calc_metrics(actual, test_forecast)

    model_full = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model_full.fit(prophet_df)
    future   = model_full.make_future_dataframe(periods=forecast_days, freq="B")
    forecast = model_full.predict(future)

    return test_forecast, test_df, rmse, mae, mape, forecast.tail(forecast_days), model_full, forecast


# ── Fetch Data ───────────────────────────────────────────────────
with st.spinner(f"Fetching data for {selected_name}..."):
    df = fetch_data(ticker, period_map[selected_period])

if df.empty:
    st.error("  No data returned. Check the ticker or try another stock.")
    st.stop()

close = df["Close"].astype(float)

# ── KPI Cards ────────────────────────────────────────────────────
latest     = float(close.iloc[-1])
prev       = float(close.iloc[-2])
change_pct = ((latest - prev) / prev) * 100
high_52w   = float(close.max())
low_52w    = float(close.min())
vol_avg    = int(df["Volume"].astype(float).mean()) if "Volume" in df.columns else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Price",  f"₹{latest:,.2f}" if ".NS" in ticker else f"${latest:,.2f}")
col2.metric("Day Change",     f"{change_pct:+.2f}%", delta=f"{change_pct:+.2f}%")
col3.metric("52W High",       f"{high_52w:,.2f}")
col4.metric("52W Low",        f"{low_52w:,.2f}")
col5.metric("Avg Volume",     f"{vol_avg:,}")

st.divider()

# ── Price Chart ──────────────────────────────────────────────────
st.subheader("  Historical Price Chart")
tab1, tab2, tab3 = st.tabs(["Candlestick", "Line Chart", "Volume"])

with tab1:
    if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        fig = go.Figure(data=[go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"],
            low=df["Low"],  close=df["Close"],
            increasing_line_color="#1D9E75", decreasing_line_color="#E24B4A"
        )])
        fig.update_layout(title=f"{selected_name} — Candlestick Chart",
                          xaxis_rangeslider_visible=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=close,  name="Close Price", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df["Date"], y=ma20,   name="MA 20",       line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=df["Date"], y=ma50,   name="MA 50",       line=dict(dash="dot")))
    fig.update_layout(title=f"{selected_name} — Price with Moving Averages", height=420)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    if "Volume" in df.columns:
        fig = px.bar(df, x="Date", y="Volume", title=f"{selected_name} — Trading Volume",
                     color_discrete_sequence=["#378ADD"])
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Forecasting ──────────────────────────────────────────────────
st.subheader(f"  {forecast_days}-Day Price Forecast")

series = close.reset_index(drop=True)
series.index = pd.to_datetime(df["Date"].values)

# ── ARIMA ────────────────────────────────────────────────────────
if model_choice in ("ARIMA", "Both"):
    st.markdown("####   ARIMA Model")
    with st.spinner("Running ARIMA forecast..."):
        try:
            preds, test_set, rmse, mae, mape, arima_fc = run_arima_forecast(series, forecast_days)

            last_date      = series.index[-1]
            forecast_dates = pd.bdate_range(start=last_date, periods=forecast_days + 1)[1:]

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("RMSE",  rmse)
            col_m2.metric("MAE",   mae)
            col_m3.metric("MAPE",  f"{mape}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index[-90:], y=series.values[-90:],
                                     name="Historical", line=dict(color="#185FA5", width=2)))
            fig.add_trace(go.Scatter(x=forecast_dates, y=arima_fc.values,
                                     name="ARIMA Forecast",
                                     line=dict(color="#E8830C", dash="dash", width=2)))
            fig.update_layout(title=f"ARIMA — {selected_name} ({forecast_days}-day Forecast)",
                              xaxis_title="Date", yaxis_title="Price", height=420)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"ARIMA could not run: {e}. Try a shorter period or different stock.")

# ── Prophet ──────────────────────────────────────────────────────
if model_choice in ("Prophet", "Both"):
    st.markdown("####   Facebook Prophet Model")
    with st.spinner("Running Prophet forecast..."):
        try:
            p_preds, p_test, p_rmse, p_mae, p_mape, p_fc, p_model, full_fc = run_prophet_forecast(df, forecast_days)

            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.metric("RMSE",  p_rmse)
            col_p2.metric("MAE",   p_mae)
            col_p3.metric("MAPE",  f"{p_mape}%")

            hist_tail = df.tail(90)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_tail["Date"], y=hist_tail["Close"],
                                     name="Historical", line=dict(color="#185FA5", width=2)))
            fig.add_trace(go.Scatter(x=p_fc["ds"], y=p_fc["yhat"],
                                     name="Prophet Forecast",
                                     line=dict(color="#1D9E75", dash="dash", width=2)))
            fig.add_trace(go.Scatter(
                x=list(p_fc["ds"]) + list(p_fc["ds"])[::-1],
                y=list(p_fc["yhat_upper"]) + list(p_fc["yhat_lower"])[::-1],
                fill="toself", fillcolor="rgba(29,158,117,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% Confidence Interval",
            ))
            fig.update_layout(title=f"Prophet — {selected_name} ({forecast_days}-day Forecast)",
                              xaxis_title="Date", yaxis_title="Price", height=420)
            st.plotly_chart(fig, use_container_width=True)

            # Components
            with st.expander("  View Prophet Components (Trend & Seasonality)"):
                comp_fig = p_model.plot_components(full_fc)
                st.pyplot(comp_fig)

        except Exception as e:
            st.warning(f"Prophet could not run: {e}. Try a longer period (1Y+).")

st.divider()

# ── Raw Data Table ───────────────────────────────────────────────
with st.expander("  View Raw Data"):
    st.dataframe(df.tail(50), use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, f"{ticker}_data.csv", "text/csv")

st.caption("Built with Python · yfinance · ARIMA · Prophet · Streamlit · Plotly")
