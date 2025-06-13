# 📈 Stock Price Prediction with Linear Regression

This project demonstrates a simple stock price prediction pipeline using historical data and a windowed Linear Regression model.

## 📌 Project Summary

- Ticker: `PLAY` (Dave & Buster's Entertainment)
- Source: Yahoo Finance via `yfinance`
- Approach: Sliding window time-series regression
- Model: `sklearn.linear_model.LinearRegression`
- Metrics: MSE, RMSE, MAE

## 📂 Structure

.
├── stock_price_prediction.py
├── models/
│ └── play_stock_model.pkl
├── requirements.txt
└── README.md

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt

```
2. Run the script:

```bash
python stock_price_prediction.py
```
The trained model will be saved in the models/ directory.

📊 Example Output
MSE: 123.4567

RMSE: 11.1111

MAE: 8.8888

🧠 Possible Improvements
Use LSTM or more advanced models

Try different tickers or forecast horizons

Add train/val/test split and grid search




