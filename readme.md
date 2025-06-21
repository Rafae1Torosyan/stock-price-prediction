
# 📈 Stock Price Prediction with Linear Regression

This project demonstrates a simple time-series forecasting approach to predict stock prices using historical closing prices and a linear regression model.

---

## ⚙️ How It Works

- Downloads historical stock price data (default ticker: `PLAY`)
- Creates a sliding window dataset to predict the next day’s closing price based on previous days
- Trains a linear regression model on the windowed data
- Evaluates performance with MSE, RMSE, and MAE metrics
- Saves model and plots for visualization

---

## 🚀 How to Run

By default, the script downloads data for ticker `PLAY`:

```bash
python main.py
````

You can specify a different ticker symbol with the `--ticker` argument:

```bash
python main.py --ticker AAPL
```

Replace `AAPL` with any valid stock ticker symbol you want to analyze.

---

## 📉 Evaluation Metrics (example for ticker PLAY):

* MSE: 0.1856
* RMSE: 0.4308
* MAE: 0.3375

---

## 🗂️ Project Structure

```
stock-price-prediction/
├── main.py                  # Main script to run the model
├── models/                  # Saved trained models
├── plots/                   # Generated plots (price history, predictions)
├── requirements.txt         # Dependencies list
└── README.md                # Project overview
```

---

## 🛠️ Dependencies

* yfinance
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* joblib

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## License

MIT License

