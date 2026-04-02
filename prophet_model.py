"""
Facebook Prophet Stock Price Forecasting Model
================================================
Author: Hemanta sethy
Description: Uses FB Prophet to forecast stock closing prices
             with seasonality, trend, and confidence intervals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


# ── Helper Functions ────────────────────────────────────────────

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Return RMSE, MAE, and MAPE."""
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "MAPE": round(mape, 4)}


def load_and_prepare(filepath: str) -> pd.DataFrame:
    """
    Load CSV and return a Prophet-ready DataFrame with
    columns ['ds', 'y'] (date and target value).
    """
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df[["Date", "Close"]].dropna()
    df.columns = ["ds", "y"]
    df["y"] = df["y"].astype(float)
    return df


# ── Main Forecasting Function ────────────────────────────────────

def run_prophet(filepath: str, ticker: str = "Stock",
                forecast_days: int = 30,
                seasonality_mode: str = "multiplicative"):
    """
    Full Prophet pipeline:
      1. Load & prepare data
      2. Train / test split (80/20)
      3. Fit Prophet and evaluate on test
      4. Forecast next N business days
      5. Plot components and predictions
    """
    print(f"\n{'='*55}")
    print(f"  Prophet Forecast — {ticker}")
    print(f"{'='*55}")

    # 1. Load data
    df = load_and_prepare(filepath)
    print(f"\n  Data range  : {df['ds'].min().date()} → {df['ds'].max().date()}")
    print(f"   Total rows  : {len(df)}")

    # 2. Train / test split
    split      = int(len(df) * 0.80)
    train_df   = df[:split].reset_index(drop=True)
    test_df    = df[split:].reset_index(drop=True)
    print(f"\n  Train size  : {len(train_df)}  |  Test size: {len(test_df)}")

    # 3. Fit Prophet
    print("\n  Fitting Prophet model...")
    model = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(train_df)

    # Predict on test period
    future_test  = model.make_future_dataframe(periods=len(test_df), freq="B")
    forecast_all = model.predict(future_test)

    test_forecast = forecast_all[forecast_all["ds"].isin(test_df["ds"])][["ds", "yhat"]].reset_index(drop=True)
    merged        = pd.merge(test_df, test_forecast, on="ds", how="inner")

    actual      = merged["y"].values
    predicted   = merged["yhat"].values

    # 4. Metrics
    metrics = calculate_metrics(actual, predicted)
    print(f"\n  Model Performance:")
    print(f"   RMSE : {metrics['RMSE']}")
    print(f"   MAE  : {metrics['MAE']}")
    print(f"   MAPE : {metrics['MAPE']}%")

    # 5. Future forecast (full data)
    print(f"\n  Refitting on full data for {forecast_days}-day forecast...")
    model_full = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model_full.fit(df)
    future     = model_full.make_future_dataframe(periods=forecast_days, freq="B")
    forecast   = model_full.predict(future)
    future_only = forecast.tail(forecast_days)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    print(f"\n🔮 Next {forecast_days}-day Forecast (first 5 rows):")
    print(future_only.rename(columns={"ds": "Date", "yhat": "Forecast",
                                       "yhat_lower": "Lower CI",
                                       "yhat_upper": "Upper CI"}).head().to_string(index=False))

    # 6. Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(f"Facebook Prophet — {ticker}", fontsize=14, fontweight="bold")

    # Subplot 1 — Actual vs Predicted (test set)
    axes[0].plot(merged["ds"], actual,    label="Actual",    linewidth=1.8)
    axes[0].plot(merged["ds"], predicted, label="Predicted", linestyle="--", linewidth=1.5)
    axes[0].set_title("Test Set: Actual vs Predicted")
    axes[0].legend()
    axes[0].set_ylabel("Price")
    axes[0].grid(alpha=0.3)

    # Subplot 2 — Future forecast with confidence interval
    hist_tail = df.tail(90)
    axes[1].plot(hist_tail["ds"], hist_tail["y"], label="Historical (last 90 days)", linewidth=1.8)
    axes[1].plot(future_only["ds"], future_only["yhat"],
                 color="orange", linestyle="--", linewidth=1.8, label=f"Forecast ({forecast_days} days)")
    axes[1].fill_between(future_only["ds"],
                          future_only["yhat_lower"],
                          future_only["yhat_upper"],
                          alpha=0.25, color="orange", label="95% Confidence Interval")
    axes[1].set_title(f"Next {forecast_days}-day Forecast with Confidence Interval")
    axes[1].legend()
    axes[1].set_ylabel("Price")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"prophet_{ticker.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()
    print(f"\n  Plot saved as prophet_{ticker.lower().replace(' ', '_')}.png")

    # Prophet component plot
    fig2 = model_full.plot_components(forecast)
    fig2.suptitle(f"Prophet Components — {ticker}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"prophet_components_{ticker.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()

    return metrics, future_only


# ── Entry Point ──────────────────────────────────────────────────

if __name__ == "__main__":
    run_prophet(
        filepath="data/TCS_NS.csv",
        ticker="TCS",
        forecast_days=30,
        seasonality_mode="multiplicative",
    )
