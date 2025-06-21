# Stock Price Prediction Using Linear Regression on Time Series Data

This project demonstrates a simple approach to predict stock prices using historical closing prices and a linear regression model trained on sliding window time-series data.

---

## 📈 Objective

Predict the next day’s closing price of a stock based on the past `window_size` days of closing prices.

---

## ⚙️ Features

- Download historical stock prices using `yfinance`
- Create windowed input-output pairs for time series regression
- Train/test split without shuffling (to preserve temporal order)
- Linear regression model training and evaluation
- Visualization of price history and predicted vs true prices
- Save trained model and plots to disk

---

## 📂 Project Structure

```

stock-price-prediction/
├── main.py             # Main script for data download, training, evaluation, and plotting
├── models/             # Saved trained model(s)
├── plots/              # Visualizations (price history, prediction vs true)
├── requirements.txt    # Python dependencies
└── README.md           # This file

````

---

## 🛠️ Requirements

Install dependencies via pip:

```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn joblib
````

---

## ▶️ How to Run

Simply run:

```bash
python main.py
```

---

## 📊 Outputs

* `plots/price_history.png` — Historical closing prices chart
* `plots/prediction_vs_true.png` — Plot comparing predicted and true prices on test set
* `models/play_stock_model.pkl` — Saved trained linear regression model

---

## 📉 Evaluation Metrics

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)

---

## License

MIT License

---

