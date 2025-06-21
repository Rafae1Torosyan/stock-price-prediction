import os
import argparse
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def create_windowed_data(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)


def main(args):
    print(f"Downloading data for {args.ticker}...")
    df = yf.download(args.ticker, period=args.period)[["Close"]].dropna()
    print("Data downloaded successfully.\n")

    # Plot price history
    plt.plot(df.index, df["Close"], label="Close Price")
    plt.title(f"{args.ticker} Stock Price History")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(args.plot_dir, exist_ok=True)
    price_history_path = os.path.join(args.plot_dir, "price_history.png")
    plt.savefig(price_history_path)
    plt.show()

    # Prepare windowed data
    prices = df["Close"].values
    X, y = create_windowed_data(prices, args.window_size)

    # Split train/test (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, shuffle=False
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")

    # Train model
    model = LinearRegression()
    model.fit(X_train.reshape(len(X_train), -1), y_train)

    # Predict
    y_pred = model.predict(X_test.reshape(len(X_test), -1))

    # Plot predictions vs true
    plt.plot(range(len(y_test)), y_test, label="True Price")
    plt.plot(range(len(y_pred)), y_pred, label="Predicted Price")
    plt.title(f"{args.ticker}: True vs Predicted Prices")
    plt.xlabel("Time Step")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()

    prediction_plot_path = os.path.join(args.plot_dir, "prediction_vs_true.png")
    plt.savefig(prediction_plot_path)
    plt.show()

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n📉 MSE:  {mse:.4f}")
    print(f"📉 RMSE: {rmse:.4f}")
    print(f"📉 MAE:  {mae:.4f}")

    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, f"{args.ticker}_stock_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Price Prediction with Linear Regression")
    parser.add_argument("--ticker", type=str, default="PLAY", help="Stock ticker symbol")
    parser.add_argument("--period", type=str, default="5y", help="Data period for download (e.g., 1y, 5y)")
    parser.add_argument("--window_size", type=int, default=10, help="Window size for time series data")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of test dataset")
    parser.add_argument("--plot_dir", type=str, default="plots", help="Directory to save plots")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save model")

    args = parser.parse_args()
    main(args)
