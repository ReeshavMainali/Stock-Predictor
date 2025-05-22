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
    Predicts future stock prices using the trained LSTM model.

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
        predictions = []
        current_sequence = last_sequence.copy()
        seq_length = current_sequence.shape[0]  # Get sequence length from input
        n_features = current_sequence.shape[1]  # Get number of features from input

        # --- Parameters for adding fluctuations (Adjust these!) ---
        # Base volatility scaling - how much historical volatility influences noise
        base_volatility_scale = 0.2  # Was 0.2
        # Trend noise scaling - how much random trend influences noise
        trend_noise_scale = 0.2  # Was 0.5
        # Cyclical factor scaling
        cyclical_scale = 0.3  # Was 0.3
        # Momentum/Mean Reversion scaling
        momentum_mr_scale = 0.3  # Was 0.2
        # Shock probability and magnitude scaling
        shock_probability = 0.1  # Was 0.05
        shock_magnitude_scale = 1.0  # Was 0.5-1.5 range, now fixed scale
        # Feature noise scaling - how much noise to add when simulating future features
        feature_noise_scale = 0.02  # New parameter

        # --- Calculate initial historical metrics ---
        # Ensure historical_volatility is not zero
        historical_volatility = np.std(current_sequence[:, 0])
        if historical_volatility < 1e-6:  # Add a small value if volatility is near zero
            historical_volatility = np.mean(current_sequence[:, 0]) * 0.01  # Use a percentage of the price as base volatility
            if historical_volatility < 1e-6:
                historical_volatility = 0.01  # Fallback minimum

        historical_mean_change = np.mean(np.abs(np.diff(current_sequence[:, 0])))
        if historical_mean_change < 1e-6:  # Add a small value if mean change is near zero
            historical_mean_change = historical_volatility * 0.1  # Use a percentage of volatility
            if historical_mean_change < 1e-6:
                historical_mean_change = 0.001  # Fallback minimum

        trend_strength = 0.3  # Keep trend strength parameter
        reversal_probability = 0.15  # Keep reversal probability
        volatility_scaling = np.random.uniform(0.8, 1.2)  # Make volatility scaling fluctuate more
        last_actual_scaled = current_sequence[-1, 0]  # Use scaled value for calculations
        trend_direction = np.random.choice([-1, 1])
        cycle_length = np.random.randint(5, 15)
        cycle_phase = 0

        for i in range(num_days):
            # Predict the next step based on the current sequence
            # Reshape for the model: (batch_size, seq_length, n_features)
            # Ensure the input shape matches the model's expected input shape
            pred_scaled = model.predict(current_sequence.reshape(1, seq_length, n_features), verbose=0)[0, 0]

            # --- Add Fluctuations ---
            # Base random noise scaled by historical volatility and current volatility scaling
            # Ensure noise is added in the scaled space
            base_noise = np.random.normal(0, historical_volatility * volatility_scaling) * base_volatility_scale

            # Trend noise scaled by historical mean change and trend noise scale
            trend_noise = historical_mean_change * np.random.uniform(-1.0, 1.0) * trend_noise_scale  # Increased range

            # Cyclical factor
            cycle_phase += 1
            cyclical_factor = np.sin(2 * np.pi * cycle_phase / cycle_length) * historical_volatility * cyclical_scale

            # Combine noise components
            market_factor = base_noise + trend_noise + cyclical_factor

            # Apply trend direction
            pred_with_noise_scaled = pred_scaled + market_factor * trend_direction

            # Add momentum and mean reversion based on the *scaled* values
            if predictions:
                # Use the last predicted scaled value for momentum/MR
                last_pred_scaled = predictions[-1]
                momentum = (last_pred_scaled - last_actual_scaled) * trend_strength * momentum_mr_scale
                mean_reversion = (last_actual_scaled - last_pred_scaled) * (1 - trend_strength) * momentum_mr_scale
                pred_with_noise_scaled += momentum + mean_reversion
            else:
                # If it's the first prediction, base momentum/MR on the last historical change
                last_historical_change_scaled = current_sequence[-1, 0] - current_sequence[-2, 0] if len(current_sequence) > 1 else 0
                momentum = last_historical_change_scaled * trend_strength * momentum_mr_scale
                mean_reversion = (last_actual_scaled - (last_actual_scaled + last_historical_change_scaled)) * (1 - trend_strength) * momentum_mr_scale  # MR towards last actual
                pred_with_noise_scaled += momentum + mean_reversion

            # Add random shock
            if np.random.random() < shock_probability:
                shock_magnitude = historical_volatility * np.random.uniform(0.8, 1.5) * shock_magnitude_scale  # Increased range
                pred_with_noise_scaled += shock_magnitude * np.random.choice([-1, 1])  # Shock can be up or down

            # --- Update sequence for next prediction ---
            # The model predicts the *next* value (index 0) based on the sequence
            # We need to create the *full* feature vector for the next step
            # Roll the sequence back one step
            current_sequence = np.roll(current_sequence, -1, axis=0)

            # Create a new row for the predicted day
            new_row_scaled = np.zeros(n_features)
            new_row_scaled[0] = pred_with_noise_scaled  # The predicted price is the first feature

            # Simulate other features (SMA, RSI, Volatility) for the new day
            # This is an approximation. We'll add noise to the last known features.
            # Ensure we don't go out of bounds if n_features is only 1 (though unlikely with this model)
            if n_features > 1:
                 # Add noise to features other than the price
                 # Use standard deviation of each feature column for scaling noise
                 feature_stds = np.std(current_sequence[:, 1:], axis=0)
                 # Add a small epsilon to avoid division by zero if a feature has zero std
                 feature_stds[feature_stds < 1e-6] = 1e-6
                 noise_magnitude = feature_stds * feature_noise_scale
                 new_row_scaled[1:] = current_sequence[-2, 1:] + np.random.normal(0, noise_magnitude)

                 # Optional: Add some basic logic to keep features somewhat related to the price
                 # This is complex and might require domain knowledge or more sophisticated simulation
                 # For now, simple noise addition is used.

            # Add the new row to the sequence
            current_sequence[-1] = new_row_scaled

            # Store the predicted *scaled* price
            predictions.append(pred_with_noise_scaled)

            # Update volatility scaling and cycle parameters periodically
            if i % 2 == 0:
                volatility_scaling = np.random.uniform(0.8, 1.2)  # Fluctuate volatility scaling
            if i % cycle_length == 0:
                cycle_length = np.random.randint(5, 15)
                cycle_phase = 0
            if np.random.random() < reversal_probability:
                trend_direction *= -1  # Randomly reverse trend direction

        # --- Inverse transform predictions ---
        # Create a dummy array with the predicted prices and placeholder features
        # The scaler expects an array with the same number of features it was trained on (n_features)
        # We need to create an array of shape (num_days, n_features)
        # Fill the first column with the predicted scaled prices
        # Fill the other columns with dummy values (e.g., zeros or the mean of the scaled features)
        # Using zeros is standard practice for inverse transforming a single feature prediction
        dummy_features = np.zeros((num_days, n_features))
        dummy_features[:, 0] = np.array(predictions)  # Place the scaled predictions in the first column

        # Inverse transform the dummy array
        predictions_transformed = scaler.inverse_transform(dummy_features)

        # Return only the first column, which contains the inverse-transformed prices
        return predictions_transformed[:, 0].reshape(-1, 1)
