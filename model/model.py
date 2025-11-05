import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --------------------
# Device Setup
# --------------------
def setup_device():
    """
    Sets up the device (GPU, NPU, or CPU) for training.

    Returns:
        str: Device identifier string ('/GPU:0', '/NPU:0', or '/CPU:0').
    """
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available. Using GPU for training.")
        return '/GPU:0'
    try:
        devices = tf.config.list_physical_devices()
        for device in devices:
            if 'NPU' in device.device_type:
                print("Intel NPU detected. Using NPU for training.")
                return '/NPU:0'
    except:
        pass
    print("No GPU/NPU found. Using CPU for training.")
    return '/CPU:0'


# --------------------
# Data Aggregation
# --------------------
def preprocess_transaction_data(df, symbol=None):
    """
    Aggregates raw transaction-level stock data into daily summaries.

    Args:
        df (pd.DataFrame): DataFrame containing transaction data.
        symbol (str, optional): Stock symbol to filter data. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with daily summaries of stock data.
    """
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    if symbol:
        df = df[df['symbol'] == symbol]

    aggregation = {
        'rate': 'mean',
        'quantity': 'sum'
    }

    # Only include transaction count if it exists
    if 'transaction' in df.columns:
        aggregation['transaction'] = 'count'

    daily_df = df.groupby(['transaction_date']).agg(aggregation).reset_index()

    # Ensure consistent column naming
    daily_df.rename(columns={
        'rate': 'rate',
        'quantity': 'volume'
    }, inplace=True)

    if 'transaction' in daily_df.columns:
        daily_df.rename(columns={'transaction': 'trades'}, inplace=True)

    return daily_df


# --------------------
# RSI Calculation
# --------------------
def calculate_rsi(prices, period=14):
    """
    Calculates the Relative Strength Index (RSI) for a given price series.

    Args:
        prices (pd.Series): Price series.
        period (int, optional): Period for RSI calculation. Defaults to 14.

    Returns:
        pd.Series: RSI values.
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# --------------------
# Data Preparation
# --------------------
def prepare_data(df, seq_length=60):
    """
    Prepares data for LSTM model training.

    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        seq_length (int, optional): Sequence length for LSTM model. Defaults to 60.

    Returns:
        tuple: X (input sequences), y (target values), and scaler (MinMaxScaler).
    """
    df['SMA_5'] = df['rate'].rolling(window=5).mean()
    df['SMA_20'] = df['rate'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['rate'])
    df['Volatility'] = df['rate'].rolling(window=20).std()

    features = ['rate', 'SMA_5', 'SMA_20', 'RSI', 'Volatility']
    data = df[features].dropna().values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []

    # Ensure enough data exists after dropping NaNs for the sequence length
    if len(scaled_data) < seq_length + 1:
        print(f"Warning: Not enough data ({len(scaled_data)}) for sequence length ({seq_length}). Cannot prepare data.")
        return np.array([]), np.array([]), scaler  # Return empty arrays

    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:(i + seq_length)])
        y.append(scaled_data[i + seq_length, 0])

    return np.array(X), np.array(y), scaler


# --------------------
# Model Creation
# --------------------
def create_model(seq_length, n_features):
    """
    Creates an LSTM model for stock price prediction.

    Args:
        seq_length (int): Sequence length for LSTM model.
        n_features (int): Number of features in the input data.

    Returns:
        tf.keras.models.Sequential: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(256, activation='tanh', return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.4),
        LSTM(128, activation='tanh', return_sequences=True),
        Dropout(0.4),
        LSTM(64, activation='tanh'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005), loss=tf.keras.losses.Huber())
    return model

# --------------------
# Model Training
# --------------------
def train_model(df, seq_length=60, epochs=100, batch_size=32):
    """
    Trains the LSTM model.

    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        seq_length (int, optional): Sequence length for LSTM model. Defaults to 60.
        epochs (int, optional): Number of training epochs. Defaults to 100.
        batch_size (int, optional): Batch size for training. Defaults to 32.

    Returns:
        tuple: Trained LSTM model and scaler (MinMaxScaler).
    """
    device = setup_device()
    with tf.device(device):
        X, y, scaler = prepare_data(df, seq_length)

        if len(X) == 0:
            print("Not enough data to train the model.")
            return None, None # Return None if data preparation failed

        train_size = int(len(X) * 0.8)
        # Ensure train_size is at least 1
        if train_size == 0 and len(X) > 0:
             train_size = 1
        # Ensure there's data for both train and test sets
        if len(X) - train_size < 1:
             print("Not enough data for both training and testing sets.")
             return None, None

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        n_features = X.shape[2]
        model = create_model(seq_length, n_features)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]

        # Adjust batch size for non-CPU devices, but ensure it's not larger than train_size
        current_batch_size = batch_size
        if device != '/CPU:0':
            current_batch_size *= 2
        current_batch_size = min(current_batch_size, train_size)
        if current_batch_size == 0:
            current_batch_size = 1  # Ensure batch size is at least 1

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=current_batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

    return model, scaler

# --------------------
# Future Prediction
# --------------------
def predict_future(model, scaler, last_sequence, num_days=30):
    """
    Predicts future stock prices using the trained LSTM model with adaptive volatility.

    Args:
        model (tf.keras.models.Sequential): Trained LSTM model.
        scaler (MinMaxScaler): MinMaxScaler used for scaling data.
        last_sequence (np.array): Last sequence of data used for prediction.
        num_days (int, optional): Number of days to predict. Defaults to 30.

    Returns:
        np.array: Predicted stock prices for the next num_days.
    """
    device = setup_device()
    with tf.device(device):
        current_sequence = last_sequence.copy()
        seq_length = current_sequence.shape[0]
        n_features = current_sequence.shape[1]

        # --- Calculate historical volatility characteristics ---
        hist_prices = last_sequence[:, 0]  # Historical prices (scaled)
        hist_returns = np.diff(hist_prices)
        
        # Calculate key volatility metrics
        historical_volatility = np.std(hist_returns) if len(hist_returns) > 0 else 0.01
        daily_volatility = np.std(hist_prices)
        mean_absolute_change = np.mean(np.abs(hist_returns)) if len(hist_returns) > 0 else 0.001
        
        # Coefficient of variation (volatility relative to price level)
        hist_mean = np.mean(hist_prices)
        cv = daily_volatility / (hist_mean + 1e-8)
        
        # Determine volatility regime based on coefficient of variation
        if cv < 0.02:  # Very low volatility (like stable stocks)
            volatility_regime = "low"
            noise_scale = 0.3
            shock_probability = 0.02
            momentum_decay = 0.9
        elif cv < 0.05:  # Moderate volatility
            volatility_regime = "moderate"
            noise_scale = 0.6
            shock_probability = 0.05
            momentum_decay = 0.8
        else:  # High volatility
            volatility_regime = "high"
            noise_scale = 1.0
            shock_probability = 0.1
            momentum_decay = 0.7
        
        print(f"Detected volatility regime: {volatility_regime} (CV: {cv:.4f})")
        
        # Calculate trend characteristics
        x = np.arange(len(hist_prices))
        if len(hist_prices) > 1:
            slope, intercept = np.polyfit(x, hist_prices, 1)
            trend_strength = min(1.0, abs(slope) / (daily_volatility + 1e-8))
        else:
            slope = 0
            trend_strength = 0.1
        
        # Adaptive noise parameters based on historical behavior
        base_noise_std = historical_volatility * noise_scale
        trend_persistence = max(0.1, min(0.9, trend_strength * 0.5))
        
        # Initialize prediction variables
        predictions = []
        momentum = 0.0
        trend_direction = 1 if slope > 0 else -1
        
        for i in range(num_days):
            # Get model prediction
            pred_scaled = model.predict(current_sequence.reshape(1, seq_length, n_features), verbose=0)[0, 0]
            
            # --- Apply controlled noise based on volatility regime ---
            
            # 1. Base market noise (reduced and adaptive)
            market_noise = np.random.normal(0, base_noise_std * 0.5)
            
            # 2. Trend continuation with decay
            if i > 0:
                recent_change = predictions[-1] - (predictions[-2] if len(predictions) > 1 else current_sequence[-1, 0])
                momentum = momentum * momentum_decay + recent_change * (1 - momentum_decay)
            else:
                # Initialize momentum from last historical change
                if len(hist_returns) > 0:
                    momentum = hist_returns[-1] * trend_persistence
                else:
                    momentum = 0.0
            
            # 3. Mean reversion force (stronger for low volatility stocks)
            if predictions:
                deviation_from_start = predictions[-1] - current_sequence[-1, 0]
                mean_reversion = -deviation_from_start * (0.05 if volatility_regime == "high" else 0.15)
            else:
                mean_reversion = 0.0
            
            # 4. Occasional shocks (rare and smaller for stable stocks)
            shock = 0.0
            if np.random.random() < shock_probability:
                shock_magnitude = historical_volatility * np.random.uniform(0.5, 1.5)
                if volatility_regime == "low":
                    shock_magnitude *= 0.3  # Reduce shock size for stable stocks
                shock = shock_magnitude * np.random.choice([-1, 1])
            
            # 5. Cyclical component (very subtle)
            cycle_component = 0.0
            if i > 5:  # Only add cycles after some predictions
                cycle_length = 7 + np.random.randint(-2, 3)  # Weekly-ish cycles
                cycle_component = np.sin(2 * np.pi * i / cycle_length) * historical_volatility * 0.1
            
            # --- Combine all factors with appropriate weights ---
            total_change = (
                market_noise * 0.4 +
                momentum * 0.3 +
                mean_reversion * 0.2 +
                shock * 0.8 +
                cycle_component * 0.1
            )
            
            # Apply change to prediction
            pred_with_noise = pred_scaled + total_change
            
            # --- Update sequence for next prediction ---
            current_sequence = np.roll(current_sequence, -1, axis=0)
            
            # Create new row
            new_row_scaled = np.zeros(n_features)
            new_row_scaled[0] = pred_with_noise
            
            # Update other features more conservatively
            if n_features > 1:
                feature_noise = np.random.normal(0, 0.01, n_features - 1)
                new_row_scaled[1:] = current_sequence[-2, 1:] + feature_noise
            
            current_sequence[-1] = new_row_scaled
            predictions.append(pred_with_noise)
            
            # Periodically adjust trend direction based on momentum
            if i > 0 and i % 10 == 0:
                if abs(momentum) > historical_volatility:
                    # Strong momentum, small chance of reversal
                    if np.random.random() < 0.1:
                        trend_direction *= -1
                else:
                    # Weak momentum, higher chance of reversal
                    if np.random.random() < 0.3:
                        trend_direction *= -1

        # --- Inverse transform predictions ---
        # Create dummy array for inverse transformation
        dummy_features = np.zeros((num_days, n_features))
        dummy_features[:, 0] = np.array(predictions)
        
        # Inverse transform to get actual prices
        predictions_transformed = scaler.inverse_transform(dummy_features)
        
        return predictions_transformed[:, 0].reshape(-1, 1)
