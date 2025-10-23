# -*- coding: utf-8 -*-
"""
LSTM Stock Price Model Testing Suite - Optimized
Tests trained models and generates predictions
"""

import numpy as np
import pandas as pd  
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

def load_model_and_scaler(ticker):
    """Load trained model and scaler for a ticker"""
    try:
        model = load_model(f'models/{ticker}_lstm_model.h5')
        with open(f'models/{ticker}_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        print(f"[ERROR] Loading {ticker}: {e}")
        return None, None

def prepare_sequences(data, scaler, lookback=60):
    """Prepare sequences from price data"""
    scaled_data = scaler.transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X).reshape(len(X), lookback, 1)
    return X, np.array(y)

def evaluate_model(ticker, period='6mo'):
    """Test model and generate comprehensive results"""
    print(f"\n{'='*60}\n{ticker} - Model Evaluation\n{'='*60}")

    model, scaler = load_model_and_scaler(ticker)
    if model is None:
        return None

    # Fetch and prepare data
    data = yf.Ticker(ticker).history(period=period)
    print(f"[DATA] {len(data)} days | {data.index[0].date()} to {data.index[-1].date()}")

    X_test, y_test = prepare_sequences(data['Close'].values, scaler)

    # Predict
    predictions = scaler.inverse_transform(model.predict(X_test, verbose=0))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(actual, predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100

    # Display results
    print(f"\n[METRICS]")
    print(f"  RMSE: ${rmse:.2f} | MAE: ${mae:.2f}")
    print(f"  R2: {r2:.4f} | MAPE: {mape:.2f}%")
    print(f"  Status: {'[EXCELLENT]' if r2 > 0.9 else '[GOOD]' if r2 > 0.7 else '[MODERATE]' if r2 > 0.5 else '[POOR]'}")

    # Visualize
    dates = data.index[60:]
    errors = actual - predictions

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    # Plot 1: Full comparison
    ax1.plot(dates, actual, 'b-', linewidth=2, label='Actual')
    ax1.plot(dates, predictions, 'r-', linewidth=2, alpha=0.7, label='Predicted')
    ax1.set_title(f'{ticker} - Full Test Period')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Recent 30 days
    ax2.plot(dates[-30:], actual[-30:], 'b-', linewidth=2, marker='o', label='Actual')
    ax2.plot(dates[-30:], predictions[-30:], 'r-', linewidth=2, marker='x', alpha=0.7, label='Predicted')
    ax2.set_title(f'{ticker} - Last 30 Days')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Error over time
    ax3.plot(dates, errors, 'purple', linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='--')
    ax3.fill_between(dates, errors.flatten(), 0, alpha=0.3, color='purple')
    ax3.set_title('Prediction Error')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Error distribution
    ax4.hist(errors, bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_title('Error Distribution')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'models/{ticker}_test_results.png', dpi=150, bbox_inches='tight')
    print(f"[SAVED] models/{ticker}_test_results.png")

    # Save CSV
    pd.DataFrame({
        'Date': dates,
        'Actual': actual.flatten(),
        'Predicted': predictions.flatten(),
        'Error': errors.flatten(),
        'Error_%': (np.abs(errors.flatten()) / actual.flatten() * 100)
    }).to_csv(f'models/{ticker}_test_results.csv', index=False)

    return {'ticker': ticker, 'r2': r2, 'rmse': rmse, 'mape': mape}

def predict_next_day(ticker):
    """Predict next day's closing price"""
    model, scaler = load_model_and_scaler(ticker)
    if model is None:
        return None

    # Get last 60 days
    data = yf.Ticker(ticker).history(period='3mo')
    if len(data) < 60:
        print(f"[ERROR] {ticker}: Need 60 days, got {len(data)}")
        return None

    # Predict
    X = scaler.transform(data['Close'].values[-60:].reshape(-1, 1))
    X = np.array([X]).reshape(1, 60, 1)
    predicted = scaler.inverse_transform(model.predict(X, verbose=0))[0][0]

    current = data['Close'].iloc[-1]
    change = predicted - current
    change_pct = (change / current) * 100

    print(f"\n{ticker}: ${current:.2f} -> ${predicted:.2f} ({change_pct:+.2f}%) [{('UP' if change > 0 else 'DOWN')}]")

    return {'ticker': ticker, 'current': current, 'predicted': predicted, 'change_%': change_pct}

def compare_stocks(tickers):
    """Compare predictions across multiple stocks"""
    print(f"\n{'='*60}\nMulti-Stock Comparison\n{'='*60}")

    results = []
    for ticker in tickers:
        result = predict_next_day(ticker)
        if result:
            results.append(result)

    if not results:
        print("[ERROR] No predictions generated")
        return None

    # Create and sort dataframe
    df = pd.DataFrame(results).sort_values('change_%', ascending=False)

    print(f"\n[SUMMARY]\n{df.to_string(index=False)}")
    df.to_csv('models/multi_stock_predictions.csv', index=False)
    print(f"\n[SAVED] models/multi_stock_predictions.csv")

    # Visualize
    plt.figure(figsize=(14, 8))
    colors = ['green' if x > 0 else 'red' for x in df['change_%']]
    x_pos = np.arange(len(df))

    plt.bar(x_pos, df['change_%'], color=colors, alpha=0.7, edgecolor='black')
    plt.xticks(x_pos, df['ticker'], rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.title('Next Day Price Change Predictions', fontsize=16, fontweight='bold')
    plt.xlabel('Stock Ticker', fontsize=12)
    plt.ylabel('Expected Change (%)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')

    # Add labels
    for i, v in enumerate(df['change_%']):
        plt.text(i, v + 0.1 if v > 0 else v - 0.3, f'{v:.2f}%',
                ha='center', va='bottom' if v > 0 else 'top', fontweight='bold')

    plt.tight_layout()
    plt.savefig('models/multi_stock_comparison.png', dpi=150, bbox_inches='tight')
    print(f"[SAVED] models/multi_stock_comparison.png")

    return df

# Main execution
if __name__ == "__main__":
    STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

    print("\n" + "="*60)
    print("LSTM MODEL TESTING SUITE")
    print("="*60)

    # Test 1: Evaluate each model
    print("\n[TEST 1] Model Evaluation")
    for ticker in STOCKS:
        try:
            evaluate_model(ticker)
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")

    # Test 2: Next day predictions
    print(f"\n{'='*60}\n[TEST 2] Next Day Predictions\n{'='*60}")
    for ticker in STOCKS:
        try:
            predict_next_day(ticker)
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")

    # Test 3: Multi-stock comparison
    print(f"\n{'='*60}\n[TEST 3] Multi-Stock Comparison\n{'='*60}")
    try:
        compare_stocks(STOCKS)
    except Exception as e:
        print(f"[ERROR] Comparison: {e}")

    print(f"\n{'='*60}")
    print("[COMPLETE] All tests finished!")
    print("="*60)
    print("\n[INFO] Results: models/*.png, models/*.csv")
    print("[WARNING] Not financial advice!")
