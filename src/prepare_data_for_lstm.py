import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

def fetch_and_save_training_data(ticker, period='5y'):
    """
    Fetch data and save it for LSTM training
    This bridges your real-time app with the LSTM tutorial
    """
    print(f"\n{'='*60}")
    print(f"Preparing training data for {ticker}")
    print('='*60)
    
    # Fetch data using yfinance (same as your app!)
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    
    print(f" Fetched {len(data)} days of data")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Save raw data
    filename = f'data/{ticker}_raw_data.csv'
    data.to_csv(filename)
    print(f" Saved raw data to {filename}")
    
    # Prepare Close prices for LSTM
    close_prices = data['Close'].values.reshape(-1, 1)
    
    # Scale the data (LSTM requirement)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Save scaler for later use
    scaler_file = f'models/{ticker}_scaler.pkl'
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f" Saved scaler to {scaler_file}")
    
    # Create training sequences (60 days to predict next day)
    X_train = []
    y_train = []
    
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Reshape for LSTM [samples, time_steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    print(f"\n Training Data Shape:")
    print(f"   X_train: {X_train.shape}")
    print(f"   y_train: {y_train.shape}")
    
    # Save prepared data
    np.save(f'data/{ticker}_X_train.npy', X_train)
    np.save(f'data/{ticker}_y_train.npy', y_train)
    np.save(f'data/{ticker}_scaled_data.npy', scaled_data)
    
    print(f"\n All data prepared and saved!")
    print(f"Ready for LSTM training!")
    
    return X_train, y_train, scaled_data, scaler

# Prepare data for your main stocks
stocks_to_train = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

print("\n Starting data preparation for LSTM training...")
print(f"Preparing {len(stocks_to_train)} stocks\n")

for ticker in stocks_to_train:
    try:
        fetch_and_save_training_data(ticker, period='5y')
    except Exception as e:
        print(f"Error with {ticker}: {e}")

print("\n" + "="*60)
print(" DATA PREPARATION COMPLETE!")
print("="*60)
print("\n Files created:")
print("   data/           → Raw CSV files")
print("   data/           → Numpy training files (.npy)")
print("   models/         → Scaler files (.pkl)")
print("\n Next step: Watch Tutorial 2 and train LSTM model!")



