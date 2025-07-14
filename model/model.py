import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# --------------------
# Device Setup
# --------------------
def setup_device():
    """
    Sets up the device (GPU, NPU, or CPU) for training with improved GPU configuration.
    
    Returns:
        str: Device identifier string ('/GPU:0', '/NPU:0', or '/CPU:0').
    """
    # Configure GPU memory growth to prevent memory allocation issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU is available. Using GPU for training. Found {len(gpus)} GPU(s).")
            return '/GPU:0'
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Check for NPU
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
# Enhanced Data Preprocessing
# --------------------
def preprocess_transaction_data(df, symbol=None):
    """
    Enhanced aggregation of raw transaction-level stock data into daily summaries
    with additional market microstructure features.
    
    Args:
        df (pd.DataFrame): DataFrame containing transaction data.
        symbol (str, optional): Stock symbol to filter data. Defaults to None.
        
    Returns:
        pd.DataFrame: DataFrame with enhanced daily summaries including OHLCV data.
    """
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    if symbol:
        df = df[df['symbol'] == symbol]
    
    if df.empty:
        return pd.DataFrame()
    
    # Enhanced aggregation with OHLCV data
    daily_agg = df.groupby(['transaction_date']).agg({
        'rate': ['first', 'max', 'min', 'last', 'mean', 'std'],
        'quantity': ['sum', 'mean', 'std', 'count'],
        'amount': ['sum', 'mean'],
        'transaction': 'count'
    }).reset_index()
    
    # Flatten column names
    daily_agg.columns = ['transaction_date', 'open', 'high', 'low', 'close', 'avg_price', 'price_std',
                        'volume', 'avg_volume', 'volume_std', 'trade_count', 'total_amount', 'avg_amount', 'transaction_count']
    
    # Handle missing values
    daily_agg['price_std'] = daily_agg['price_std'].fillna(0)
    daily_agg['volume_std'] = daily_agg['volume_std'].fillna(0)
    
    # Calculate additional market microstructure features
    daily_agg['price_range'] = daily_agg['high'] - daily_agg['low']
    daily_agg['price_range_pct'] = (daily_agg['price_range'] / daily_agg['close']) * 100
    daily_agg['avg_trade_size'] = daily_agg['volume'] / daily_agg['trade_count']
    daily_agg['vwap'] = daily_agg['total_amount'] / daily_agg['volume']  # Volume Weighted Average Price
    daily_agg['price_efficiency'] = abs(daily_agg['close'] - daily_agg['vwap']) / daily_agg['close']
    
    # Sort by date
    daily_agg = daily_agg.sort_values('transaction_date').reset_index(drop=True)
    
    return daily_agg

# --------------------
# Enhanced Technical Indicators
# --------------------
def calculate_technical_indicators(df):
    """
    Calculate comprehensive technical indicators for stock analysis.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data.
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    df = df.copy()
    
    # Price-based indicators
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'])
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # On-Balance Volume (OBV)
    df['obv'] = (df['volume'] * np.sign(df['returns'])).cumsum()
    
    # Average True Range (ATR)
    df['tr'] = np.maximum(df['high'] - df['low'], 
                         np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                   abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized
    
    # Price momentum
    df['momentum'] = df['close'] / df['close'].shift(10) - 1
    
    # Williams %R
    df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
    
    return df

def calculate_rsi(prices, period=14):
    """Enhanced RSI calculation with improved handling of edge cases."""
    if len(prices) < period + 1:
        return pd.Series(index=prices.index, dtype=float)
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Handle division by zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# --------------------
# Enhanced Data Preparation
# --------------------
def prepare_data(df, seq_length=60, prediction_horizon=1, validation_split=0.2):
    """
    Enhanced data preparation with multiple features and proper validation split.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data with technical indicators.
        seq_length (int): Sequence length for LSTM model.
        prediction_horizon (int): Days ahead to predict.
        validation_split (float): Fraction of data for validation.
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names)
    """
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Select features for model
    feature_columns = [
        'close', 'volume', 'high', 'low', 'open', 'vwap',
        'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'ema_20',
        'rsi', 'macd', 'macd_signal', 'bb_position', 'bb_width',
        'stoch_k', 'stoch_d', 'volume_ratio', 'atr', 'volatility',
        'momentum', 'williams_r', 'price_range_pct', 'price_efficiency'
    ]
    
    # Filter existing columns
    available_features = [col for col in feature_columns if col in df.columns]
    
    if not available_features:
        print("No suitable features found in dataframe")
        return None, None, None, None, None, None, None, None
    
    # Prepare data
    data = df[available_features].dropna()
    
    if len(data) < seq_length + prediction_horizon:
        print(f"Insufficient data: {len(data)} rows, need at least {seq_length + prediction_horizon}")
        return None, None, None, None, None, None, None, None
    
    # Scale features
    scaler = StandardScaler()  # Often works better than MinMaxScaler for financial data
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - seq_length - prediction_horizon + 1):
        X.append(scaled_data[i:(i + seq_length)])
        y.append(scaled_data[i + seq_length + prediction_horizon - 1, 0])  # Predict close price
    
    X, y = np.array(X), np.array(y)
    
    # Split data
    train_size = int(len(X) * (1 - validation_split - 0.2))  # 60% train, 20% val, 20% test
    val_size = int(len(X) * validation_split)
    
    X_train = X[:train_size]
    X_val = X[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    
    y_train = y[:train_size]
    y_val = y[train_size:train_size + val_size]
    y_test = y[train_size + val_size:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, available_features

# --------------------
# Enhanced Model Architecture
# --------------------
def create_model(seq_length, n_features, dropout_rate=0.3):
    """
    Create an enhanced LSTM model with improved architecture.
    
    Args:
        seq_length (int): Sequence length for LSTM model.
        n_features (int): Number of features in input data.
        dropout_rate (float): Dropout rate for regularization.
        
    Returns:
        tf.keras.models.Sequential: Enhanced LSTM model.
    """
    model = Sequential([
        # First LSTM layer with return sequences
        Bidirectional(LSTM(128, return_sequences=True, 
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                     input_shape=(seq_length, n_features)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Second LSTM layer
        Bidirectional(LSTM(64, return_sequences=True,
                          kernel_regularizer=tf.keras.regularizers.l2(0.001))),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Third LSTM layer
        LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Dense layers
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(dropout_rate),
        
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Use adaptive learning rate
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, 
                 loss='huber',  # More robust to outliers
                 metrics=['mse', 'mae'])
    
    return model

# --------------------
# Enhanced Training Function
# --------------------
def train_model(df, seq_length=60, epochs=100, batch_size=32, patience=20):
    """
    Enhanced model training with proper validation and callbacks.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        seq_length (int): Sequence length for LSTM model.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for training.
        patience (int): Early stopping patience.
        
    Returns:
        tuple: (model, scaler, history, evaluation_metrics)
    """
    device = setup_device()
    
    with tf.device(device):
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, features = prepare_data(
            df, seq_length
        )
        
        if X_train is None:
            print("Data preparation failed")
            return None, None, None, None
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Features used: {features}")
        
        # Create model
        model = create_enhanced_model(seq_length, X_train.shape[2])
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Adjust batch size for GPU
        if device != '/CPU:0':
            batch_size = min(batch_size * 2, len(X_train))
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        evaluation_metrics = evaluate_model(model, X_test, y_test, scaler)
        
        return model, scaler, history, evaluation_metrics

# --------------------
# Model Evaluation
# --------------------
def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        scaler: Fitted scaler
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Create dummy arrays for inverse scaling
    y_test_scaled = np.zeros((len(y_test), scaler.n_features_in_))
    y_pred_scaled = np.zeros((len(y_pred), scaler.n_features_in_))
    
    y_test_scaled[:, 0] = y_test
    y_pred_scaled[:, 0] = y_pred.flatten()
    
    # Inverse transform
    y_test_actual = scaler.inverse_transform(y_test_scaled)[:, 0]
    y_pred_actual = scaler.inverse_transform(y_pred_scaled)[:, 0]
    
    # Calculate metrics
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_actual, y_pred_actual)
    
    # Calculate percentage errors
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
    
    # Direction accuracy
    actual_direction = np.sign(np.diff(y_test_actual))
    pred_direction = np.sign(np.diff(y_pred_actual))
    direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'direction_accuracy': direction_accuracy
    }
    
    print("\nModel Evaluation Results:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Direction Accuracy: {direction_accuracy:.2f}%")
    
    return metrics

# --------------------
# Enhanced Prediction Function
# --------------------
def predict_future(model, scaler, last_sequence, num_days=30, confidence_intervals=True):
    """
    Enhanced future prediction with confidence intervals and better uncertainty modeling.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        last_sequence: Last sequence of data
        num_days: Number of days to predict
        confidence_intervals: Whether to calculate confidence intervals
        
    Returns:
        dict: Dictionary containing predictions and confidence intervals
    """
    device = setup_device()
    
    with tf.device(device):
        predictions = []
        current_sequence = last_sequence.copy()
        seq_length = current_sequence.shape[0]
        n_features = current_sequence.shape[1]
        
        # For confidence intervals, we'll use Monte Carlo dropout
        if confidence_intervals:
            n_samples = 100
            all_predictions = []
            
            for _ in range(n_samples):
                sample_predictions = []
                temp_sequence = current_sequence.copy()
                
                for day in range(num_days):
                    # Predict with dropout enabled (training=True)
                    pred = model(temp_sequence.reshape(1, seq_length, n_features), training=True)
                    pred_value = pred.numpy()[0, 0]
                    sample_predictions.append(pred_value)
                    
                    # Update sequence
                    new_row = np.zeros(n_features)
                    new_row[0] = pred_value
                    
                    # Estimate other features (simplified)
                    if n_features > 1:
                        new_row[1:] = temp_sequence[-1, 1:] * (1 + np.random.normal(0, 0.001, n_features-1))
                    
                    temp_sequence = np.roll(temp_sequence, -1, axis=0)
                    temp_sequence[-1] = new_row
                
                all_predictions.append(sample_predictions)
            
            # Calculate statistics
            all_predictions = np.array(all_predictions)
            predictions = np.mean(all_predictions, axis=0)
            prediction_std = np.std(all_predictions, axis=0)
            
            # Calculate confidence intervals (95%)
            lower_bound = predictions - 1.96 * prediction_std
            upper_bound = predictions + 1.96 * prediction_std
            
        else:
            # Simple prediction without confidence intervals
            for day in range(num_days):
                pred = model.predict(current_sequence.reshape(1, seq_length, n_features), verbose=0)
                pred_value = pred[0, 0]
                predictions.append(pred_value)
                
                # Update sequence
                new_row = np.zeros(n_features)
                new_row[0] = pred_value
                
                if n_features > 1:
                    new_row[1:] = current_sequence[-1, 1:]
                
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = new_row
        
        # Inverse transform predictions
        dummy_array = np.zeros((len(predictions), scaler.n_features_in_))
        dummy_array[:, 0] = predictions
        predictions_actual = scaler.inverse_transform(dummy_array)[:, 0]
        
        result = {
            'predictions': predictions_actual,
            'dates': pd.date_range(start=pd.Timestamp.now(), periods=num_days, freq='D')
        }
        
        if confidence_intervals:
            dummy_lower = np.zeros((len(lower_bound), scaler.n_features_in_))
            dummy_upper = np.zeros((len(upper_bound), scaler.n_features_in_))
            dummy_lower[:, 0] = lower_bound
            dummy_upper[:, 0] = upper_bound
            
            result['lower_bound'] = scaler.inverse_transform(dummy_lower)[:, 0]
            result['upper_bound'] = scaler.inverse_transform(dummy_upper)[:, 0]
        
        return result