"""
ARIMA Stock Price Forecasting Model
=====================================
Author: Your Name
Description: Trains an ARIMA model on historical stock data
             and forecasts future closing prices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


# ── Helper Functions ────────────────────────────────────────────

def check_stationarity(series: pd.Series) -> None:
    """Run Augmented Dickey-Fuller test and print result."""
    result = adfuller(series.dropna())
    print(f"\n📊 ADF Statistic : {result[0]:.4f}")
    print(f"   p-value       : {result[1]:.4f}")
    if result[1] < 0.05:
        print("   ✅ Series is STATIONARY (good for ARIMA)")
    else:
        print("   ⚠️  Series is NOT stationary — differencing required")


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Return RMSE, MAE, and MAPE."""
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "MAPE": round(mape, 4)}


def load_data(filepath: str, target_col: str = "Close") -> pd.Series:
    """Load CSV and return the target column as a Series with DatetimeIndex."""
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df[target_col].astype(float)


# ── Main Forecasting Function ────────────────────────────────────

def run_arima(filepath: str, ticker: str = "Stock",
              order: tuple = (5, 1, 0), forecast_days: int = 30):
    """
    Full ARIMA pipeline:
      1. Load & inspect data
      2. Stationarity check
      3. Train / test split (80/20)
      4. Fit ARIMA and evaluate
      5. Forecast next N days
      6. Plot results
    """
    print(f"\n{'='*55}")
    print(f"  ARIMA Forecast — {ticker}")
    print(f"{'='*55}")

    # 1. Load data
    series = load_data(filepath)
    print(f"\n📅 Data range  : {series.index[0].date()} → {series.index[-1].date()}")
    print(f"   Total rows  : {len(series)}")

    # 2. Stationarity check
    check_stationarity(series)

    # 3. Train / test split
    split = int(len(series) * 0.80)
    train, test = series[:split], series[split:]
    print(f"\n📂 Train size  : {len(train)}  |  Test size: {len(test)}")

    # 4. Fit ARIMA
    print(f"\n🔧 Fitting ARIMA{order}...")
    model  = ARIMA(train, order=order)
    fitted = model.fit()

    # Rolling one-step predictions on test set
    history     = list(train)
    predictions = []
    for obs in test:
        m    = ARIMA(history, order=order).fit()
        yhat = m.forecast(steps=1)[0]
        predictions.append(yhat)
        history.append(obs)

    predictions = np.array(predictions)
    actual      = test.values

    # 5. Metrics
    metrics = calculate_metrics(actual, predictions)
    print(f"\n📈 Model Performance:")
    print(f"   RMSE : {metrics['RMSE']}")
    print(f"   MAE  : {metrics['MAE']}")
    print(f"   MAPE : {metrics['MAPE']}%")

    # 6. Forecast next N days
    final_model    = ARIMA(series, order=order).fit()
    forecast_res   = final_model.forecast(steps=forecast_days)
    last_date      = series.index[-1]
    forecast_dates = pd.bdate_range(start=last_date, periods=forecast_days + 1)[1:]
    forecast_df    = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast_res.values})

    print(f"\n🔮 Next {forecast_days}-day Forecast (first 5 rows):")
    print(forecast_df.head().to_string(index=False))

    # 7. Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(f"ARIMA{order} — {ticker}", fontsize=14, fontweight="bold")

    # Subplot 1 — Actual vs Predicted (test set)
    axes[0].plot(test.index, actual,      label="Actual",    linewidth=1.8)
    axes[0].plot(test.index, predictions, label="Predicted", linestyle="--", linewidth=1.5)
    axes[0].set_title("Test Set: Actual vs Predicted")
    axes[0].legend()
    axes[0].set_ylabel("Price")
    axes[0].grid(alpha=0.3)

    # Subplot 2 — Future Forecast
    axes[1].plot(series[-90:], label="Historical (last 90 days)", linewidth=1.8)
    axes[1].plot(forecast_df.set_index("Date")["Forecast"],
                 label=f"Forecast ({forecast_days} days)",
                 color="orange", linestyle="--", linewidth=1.8)
    axes[1].set_title(f"Next {forecast_days}-day Price Forecast")
    axes[1].legend()
    axes[1].set_ylabel("Price")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"arima_{ticker.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()
    print(f"\n✅ Plot saved as arima_{ticker.lower().replace(' ', '_')}.png")

    return metrics, forecast_df


# ── Entry Point ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Change the filepath to your downloaded CSV
    run_arima(
        filepath="data/TCS_NS.csv",
        ticker="TCS",
        order=(5, 1, 0),
        forecast_days=30,
    )
