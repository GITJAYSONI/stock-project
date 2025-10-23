import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import pickle

def train_lstm_model(ticker):
    """
    Train LSTM model using the prepared data
    This follows Tutorial 2 but uses YOUR data
    """
    print(f"\n{'='*60}")
    print(f"Training LSTM Model for {ticker}")
    print('='*60)
    
    # Load prepared data
    X_train = np.load(f'data/{ticker}_X_train.npy')
    y_train = np.load(f'data/{ticker}_y_train.npy')
    
    print(f"✅ Loaded training data")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    
    # Split into train and validation (80/20)
    split = int(len(X_train) * 0.8)
    X_train_split = X_train[:split]
    y_train_split = y_train[:split]
    X_val = X_train[split:]
    y_val = y_train[split:]
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(X_train_split)} samples")
    print(f"   Validation: {len(X_val)} samples")
    
    # Build LSTM Model (from Tutorial 2)
    print(f"\n🏗️ Building LSTM Model...")
    
    model = Sequential([
        # First LSTM layer
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        
        # Second LSTM layer
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        
        # Third LSTM layer
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        
        # Output layers
        Dense(units=25),
        Dense(units=1)
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print(f"✅ Model built successfully!")
    print(f"\n📋 Model Summary:")
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        f'models/{ticker}_best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: 50")
    print(f"   Batch size: 32")
    print(f"   This may take 5-10 minutes...\n")
    
    history = model.fit(
        X_train_split, y_train_split,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # Save final model
    model.save(f'models/{ticker}_lstm_model.h5')
    print(f"\n✅ Model saved to models/{ticker}_lstm_model.h5")
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{ticker} - Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'models/{ticker}_training_history.png')
    print(f"✅ Training history plot saved!")
    
    # Evaluate
    train_loss = model.evaluate(X_train_split, y_train_split, verbose=0)
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\n📊 Final Results:")
    print(f"   Training Loss: {train_loss:.6f}")
    print(f"   Validation Loss: {val_loss:.6f}")
    
    if val_loss < train_loss * 1.5:
        print(f"   ✅ Good model! No significant overfitting")
    else:
        print(f"   ⚠️ Warning: Possible overfitting")
    
    return model, history

# Train models for all prepared stocks
if __name__ == "__main__":
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    print("\n LSTM MODEL TRAINING")
    print("="*60)
    print(f"Training {len(stocks)} models")
    print("This will take approximately 30-50 minutes total")
    print("="*60)
    
    for ticker in stocks:
        try:
            model, history = train_lstm_model(ticker)
            print(f"\n {ticker} model training complete!\n")
        except Exception as e:
            print(f"\n Error training {ticker}: {e}\n")
    
    print("\n" + "="*60)
    print(" ALL MODELS TRAINED!")
    print("="*60)
    print("\n Models saved in: models/")
    print("📈 Training plots saved in: models/")
    print("\n Next: Bridge 2 - Integrate predictions into your app!")