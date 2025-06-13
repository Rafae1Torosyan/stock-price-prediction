import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

# Visualization settings
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# 1. Download historical stock price data
ticker = "PLAY"
print(f"Downloading data for {ticker}...")
df = yf.download(ticker, period="5y")[["Close"]].dropna()
print("Data downloaded successfully.\n")

# 2. Plot historical price chart
plt.plot(df.index, df["Close"], label="Close Price")
plt.title(f"{ticker} Stock Price History")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()

# Save the price history plot
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/price_history.png")
plt.show()

# 3. Create windowed dataset for time-series regression
def create_windowed_data(series, window_size):
    """
    Creates input-output pairs using a sliding window over the time series.
    X: sequences of 'window_size' past values
    y: the next value after each sequence
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

window_size = 10  # Number of days used to predict the next day
prices = df["Close"].values
X, y = create_windowed_data(prices, window_size)

# 4. Train-test split (no shuffling for time-series data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")

# 5. Train a simple linear regression model
model = LinearRegression()
model.fit(X_train.reshape(len(X_train), -1), y_train)

# 6. Predict on test set
y_pred = model.predict(X_test.reshape(len(X_test), -1))

# 7. Plot predictions vs true values
plt.plot(range(len(y_test)), y_test, label="True Price")
plt.plot(range(len(y_pred)), y_pred, label="Predicted Price")
plt.title(f"{ticker}: True vs Predicted Prices")
plt.xlabel("Time Step")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()

# Save the predictions vs true prices plot
plt.savefig("plots/prediction_vs_true.png")
plt.show()

# 8. Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📉 MSE:  {mse:.4f}")
print(f"📉 RMSE: {rmse:.4f}")
print(f"📉 MAE:  {mae:.4f}")

# 9. Save the trained model to file
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/play_stock_model.pkl")
print("\n✅ Model saved to models/play_stock_model.pkl")
