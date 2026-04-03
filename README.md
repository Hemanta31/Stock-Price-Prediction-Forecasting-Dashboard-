# 📈 Stock Price Prediction & Forecasting Dashboard

A complete **Data Analyst portfolio project** for Finance & Stock Market analysis.
Built using Python, ARIMA, Facebook Prophet, and an interactive Streamlit dashboard.

---

## 🎯 Project Overview

This project forecasts stock prices using real historical data from Yahoo Finance.
It compares two popular time series models — **ARIMA** and **Facebook Prophet** —
and displays results through an interactive web dashboard.

**Stocks covered:** TCS · Infosys · Reliance · Apple · Google · Microsoft

---

## 🗂️ Project Structure

```
stock_price_prediction/
│
├── data/
│   └── fetch_data.py          
│
├── models/
│   ├── arima_model.py          
│   └── prophet_model.py        
│
├── app/
│   └── dashboard.py            
│
├── notebooks/
│   └── EDA_and_Forecasting.ipynb  
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core programming language |
| yfinance | Fetch real stock market data |
| Pandas & NumPy | Data wrangling and analysis |
| Matplotlib | Static visualizations |
| Plotly | Interactive charts |
| Statsmodels | ARIMA model |
| Prophet | Facebook Prophet model |
| Streamlit | Web dashboard |
| Scikit-learn | Model evaluation metrics |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/stock-price-prediction.git
cd stock-price-prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Download stock data
```bash
python data/fetch_data.py
```

### Step 2 — Run ARIMA model
```bash
python models/arima_model.py
```

### Step 3 — Run Prophet model
```bash
python models/prophet_model.py
```

### Step 4 — Launch the dashboard
```bash
streamlit run app/dashboard.py
```

Open your browser at `http://localhost:8501`

---

## 📊 Features

- 📥 **Real-time data** fetched from Yahoo Finance
- 🕯️ **Candlestick chart** with OHLC data
- 📉 **Moving averages** (MA 20 & MA 50)
- 📦 **Volume analysis**
- 🔮 **ARIMA forecast** with rolling predictions
- 🔮 **Prophet forecast** with confidence intervals & seasonality components
- 📈 **Model evaluation** — RMSE, MAE, MAPE metrics
- ⬇️ **CSV download** for raw data
- 🎛️ Fully interactive sidebar (choose stock, period, model, forecast days)

---

## 📈 Model Performance (TCS — 1 Year)

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| ARIMA (5,1,0) | ~55.2 | ~41.3 | ~1.4% |
| Prophet | ~48.7 | ~36.1 | ~1.2% |

> Prophet generally performs better due to seasonality handling.

---

## 🧠 Key Concepts Demonstrated

- **Time series analysis** — stationarity, ADF test, differencing
- **ARIMA modelling** — p, d, q parameters explained
- **Prophet modelling** — trend, yearly & weekly seasonality
- **Model evaluation** — RMSE, MAE, MAPE
- **Data visualisation** — Plotly interactive charts
- **Web app development** — Streamlit dashboard

---

## 📸 Dashboard Preview

> *Run `streamlit run app/dashboard.py` to see the live dashboard*

---

## 📚 Learning Resources

- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [Streamlit Docs](https://docs.streamlit.io/)


---

## 🙋 Author

Hemanta sethy
- 📧 hemantsethy2402@gmail.com
- 💼 [LinkedIn](www.linkedin.com/in/hemanta-kumar-sethy)


---

⭐ *If you found this project helpful, please give it a star!*
