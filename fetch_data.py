"""
Stock Price Data Collection Script
===================================
Author: Hemanta sethy
Description: Downloads historical stock data using yfinance
"""

import yfinance as yf
import pandas as pd
import os

# ── Stocks to download ──────────────────────────────────────────
STOCKS = {
    "TCS.NS":      "Tata Consultancy Services",
    "INFY.NS":     "Infosys",
    "RELIANCE.NS": "Reliance Industries",
    "AAPL":        "Apple Inc.",
    "GOOGL":       "Alphabet (Google)",
}

START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"
SAVE_DIR   = os.path.dirname(__file__)


def download_stock(ticker: str, name: str) -> pd.DataFrame:
    """Download OHLCV data for a single ticker."""
    print(f"  Downloading {name} ({ticker})...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df


def save_stock(df: pd.DataFrame, ticker: str) -> None:
    """Save dataframe to CSV."""
    filename = ticker.replace(".", "_") + ".csv"
    path = os.path.join(SAVE_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved: {filename}  ({len(df)} rows)")


def main():
    print("\n Fetching stock data from Yahoo Finance...\n")
    for ticker, name in STOCKS.items():
        try:
            df = download_stock(ticker, name)
            save_stock(df, ticker)
        except Exception as e:
            print(f"  ERROR for {ticker}: {e}")
    print("\n  All downloads complete!\n")


if __name__ == "__main__":
    main()
