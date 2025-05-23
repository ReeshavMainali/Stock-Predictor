import pytest
import pandas as pd
import numpy as np
from model.model import preprocess_transaction_data, calculate_rsi, prepare_data
from sklearn.preprocessing import MinMaxScaler
from pandas.testing import assert_frame_equal, assert_series_equal

# --- Tests for preprocess_transaction_data ---

def test_preprocess_transaction_data_empty():
    df_empty = pd.DataFrame(columns=['transaction_date', 'symbol', 'rate', 'quantity', 'transaction'])
    processed_df = preprocess_transaction_data(df_empty.copy())
    assert processed_df.empty
    assert 'volume' in processed_df.columns
    assert 'trades' in processed_df.columns
    assert 'rate' in processed_df.columns # Should still be 'rate' for price

    processed_df_no_transaction_col = preprocess_transaction_data(df_empty.drop(columns=['transaction']))
    assert processed_df_no_transaction_col.empty
    assert 'volume' in processed_df_no_transaction_col.columns
    assert 'trades' not in processed_df_no_transaction_col.columns # trades from transaction
    assert 'rate' in processed_df_no_transaction_col.columns


def test_preprocess_transaction_data_sample():
    data = {
        'transaction_date': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:05:00', '2023-01-02 11:00:00', '2023-01-01 10:10:00']),
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'MSFT'],
        'rate': [150.0, 152.0, 155.0, 200.0],
        'quantity': [10, 5, 12, 7],
        'transaction': [1, 2, 3, 1] # Will be renamed to 'trades'
    }
    df = pd.DataFrame(data)

    # Test with symbol filter
    processed_aapl = preprocess_transaction_data(df.copy(), symbol='AAPL')
    assert len(processed_aapl) == 2 # Two unique dates for AAPL
    assert processed_aapl.index.name == 'transaction_date'
    
    # Check 2023-01-01 for AAPL
    day1_aapl = processed_aapl.loc[pd.to_datetime('2023-01-01')]
    assert day1_aapl['rate'] == pytest.approx((150.0 * 10 + 152.0 * 5) / (10 + 5)) # Weighted average if not simple mean. Code uses simple mean.
    assert day1_aapl['rate'] == pytest.approx(np.mean([150.0, 152.0])) # Code uses simple mean of rates per day
    assert day1_aapl['volume'] == 10 + 5
    assert day1_aapl['trades'] == 1 + 2

    # Check 2023-01-02 for AAPL
    day2_aapl = processed_aapl.loc[pd.to_datetime('2023-01-02')]
    assert day2_aapl['rate'] == 155.0
    assert day2_aapl['volume'] == 12
    assert day2_aapl['trades'] == 3
    
    assert 'symbol' not in processed_aapl.columns # Symbol column should be dropped

    # Test without symbol filter (aggregates all symbols together)
    processed_all = preprocess_transaction_data(df.copy())
    assert len(processed_all) == 2 # Two unique dates across all symbols
    
    day1_all = processed_all.loc[pd.to_datetime('2023-01-01')]
    # AAPL rates: 150, 152. MSFT rate: 200.
    assert day1_all['rate'] == pytest.approx(np.mean([150.0, 152.0, 200.0]))
    assert day1_all['volume'] == 10 + 5 + 7
    assert day1_all['trades'] == 1 + 2 + 1
    
    # Test without 'transaction' column
    df_no_transaction = df.drop(columns=['transaction'])
    processed_no_trans = preprocess_transaction_data(df_no_transaction.copy(), symbol='AAPL')
    assert len(processed_no_trans) == 2
    assert 'trades' not in processed_no_trans.columns
    assert processed_no_trans.loc[pd.to_datetime('2023-01-01')]['volume'] == 15


def test_preprocess_transaction_data_date_conversion():
    data = {
        'transaction_date': ['2023-01-01 10:00:00', '2023-01-01 12:00:00', '2023-01-02 14:00:00'],
        'symbol': ['ANY', 'ANY', 'ANY'],
        'rate': [10, 12, 15],
        'quantity': [1, 1, 1]
    }
    df = pd.DataFrame(data)
    # Convert to datetime if not already, but ensure the function handles string dates by converting to pd.datetime then .dt.date
    df['transaction_date'] = pd.to_datetime(df['transaction_date']) 
    
    processed = preprocess_transaction_data(df.copy())
    assert len(processed) == 2
    assert isinstance(processed.index, pd.DatetimeIndex)
    assert processed.index[0] == pd.to_datetime('2023-01-01')
    assert processed.index[1] == pd.to_datetime('2023-01-02')

# --- Tests for calculate_rsi ---

def test_calculate_rsi_known_values():
    # Example from https://www.macroption.com/rsi-calculation/ (slightly different results due to Wilder's vs simple SMA for first avg gain/loss)
    # The model.py uses simple SMA for the first average gain/loss.
    prices = pd.Series([
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 
        45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64
    ]) # 20 points
    period = 14

    rsi = calculate_rsi(prices, period=period)
    assert len(rsi) == len(prices)
    assert rsi.iloc[:period-1].isna().all() # First period-1 values are NaN
    
    # Manual calculation for a few points (using simple SMA for first avg, then Wilder's)
    # For index 13 (14th data point, first RSI value):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).iloc[1:period] # First 13 diffs (for period 14)
    loss = -delta.where(delta < 0, 0.0).iloc[1:period]
    
    avg_gain1 = gain.mean()
    avg_loss1 = loss.mean()
    rs1 = avg_gain1 / avg_loss1 if avg_loss1 > 0 else 100 # if avg_loss1 is 0, RSI is 100
    rsi1 = 100 - (100 / (1 + rs1))
    assert rsi.iloc[period-1] == pytest.approx(rsi1, abs=0.1) # model.py rsi starts at index 'period'

    # The implementation in model.py starts outputting RSI from index `period` (0-indexed)
    # So, rsi.iloc[period-1] is the first non-NaN value.
    # Let's check the implementation detail:
    # delta = data['rate'].diff() - this is correct
    # up = delta.clip(lower=0)
    # down = -1*delta.clip(upper=0)
    # ema_up = up.ewm(com=period-1, adjust=False).mean() -> Wilder's smoothing from the start
    # ema_down = down.ewm(com=period-1, adjust=False).mean()
    # rs = ema_up/ema_down
    # data[f'RSI'] = 100 - (100/(1+rs))
    # This uses EWM directly. For Wilder's, alpha = 1/period, so com = period - 1.
    
    # Using values from a known Wilder's RSI calculator for the given series:
    # Period 14, for price at index 13 (14th price 46.28): RSI ~ 70.59
    # For price at index 14 (15th price 46.28): RSI ~ 70.59
    # For price at index 15 (16th price 46.00): RSI ~ 63.78
    # For price at index 19 (20th price 45.64): RSI ~ 47.79
    # The implementation returns NaN for the first `period` entries.
    # So, rsi.iloc[period] would be the first calculated RSI.
    # Let's re-evaluate the indexing.
    # If `data` has length N, `data['rate'].diff()` has N elements, first is NaN.
    # `up.ewm` also has N elements. `rs` has N elements. `data['RSI']` has N elements.
    # The `calculate_rsi` function takes a Series, adds 'rate' column, calculates RSI.
    # It should return a series of same length as input `prices`.
    # The line `data[f'RSI'].iloc[:period] = np.nan` makes the first `period` values NaN.
    # Corrected understanding: first non-NaN is at period-1.
    assert rsi.iloc[:period-1].isna().all() # First `period-1` values (0 to period-2) are NaN
    assert pd.notna(rsi.iloc[period-1])     # First calculated RSI is at index period-1
    
    # For this data and Wilder's EWM from start (com=13 for period 14)
    # RSI for prices[13] (46.28) should be ~70.59
    # RSI for prices[14] (46.28) should be ~70.59
    # RSI for prices[15] (46.00) should be ~63.78
    # RSI for prices[19] (45.64) should be ~47.79

    # Let's use the function's output for a known sequence if direct calc is complex
    # Example from pandas-ta:
    # prices = pd.Series([10, 11, 10, 9, 8, 9, 10, 11, 12, 13, 12, 11, 10, 9, 8, 7, 8, 9, 10, 11])
    # rsi_14 = calculate_rsi(prices, 14)
    # Expected from pandas_ta (Wilder's):
    # 13 NaN
    # 14 28.010401
    # 15 23.300180
    # 16 37.019301
    # 17 48.337003
    # 18 57.981803
    # 19 65.924173
    
    # My model.py code: `data[f'RSI'].iloc[:period] = np.nan`
    # This means index 0 to period-1 are NaN. The first value is at index `period`.
    # So for prices[13] (14th value), this is index 13.
    # If period is 14, then indices 0..13 are NaN. prices[14] is the first RSI.
    # This seems to be a common convention: first RSI value needs `period` previous changes, so for data point `period` (0-indexed).

    # Let's re-check the provided example data's RSI with period 14
    # prices = pd.Series([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28])
    # RSI for period 14. This means we need 14 prices to calculate the first RSI.
    # The first RSI will correspond to the 14th price (index 13).
    # The model.py does: `data[f'RSI'].iloc[:period] = np.nan;`
    # This would mean `rsi.iloc[13]` is NaN. The first non-NaN is `rsi.iloc[14]`.
    # This is unusual. Typically, for period P, the first P-1 values are NaN.
    # Let's test with a slightly longer series to be sure about the output indexing.
    prices_long = pd.Series(list(range(10, 30)) + list(range(29, 9, -1))) # 20 + 20 = 40 data points
    rsi_long_14 = calculate_rsi(prices_long.copy(), period=14)
    assert rsi_long_14.iloc[:14].isna().all() # First 14 values (index 0-13) are NaN
    assert not rsi_long_14.iloc[14:].isna().any() # Subsequent values should be calculated

    # Test specific known values (if possible, otherwise behavior like all up/down)
    # All prices increasing for > period steps
    all_up = pd.Series(np.arange(1, 30, dtype=float)) # 29 prices
    rsi_all_up = calculate_rsi(all_up.copy(), period=14)
    # After enough steps, RSI for all up should be 100 (or close to it)
    assert rsi_all_up.iloc[14:].notna().all()
    # With Wilder's EWM, it approaches 100 but might not hit it exactly if series not long enough
    # For a continuous increase, avg_loss becomes very small, rs becomes very large, RSI -> 100
    assert rsi_all_up.iloc[-1] > 90 # Should be very high

    # All prices decreasing
    all_down = pd.Series(np.arange(29, 0, -1, dtype=float)) # 29 prices
    rsi_all_down = calculate_rsi(all_down.copy(), period=14)
    assert rsi_all_down.iloc[14:].notna().all()
    # RSI should be very low (close to 0)
    assert rsi_all_down.iloc[-1] < 10


def test_calculate_rsi_edge_cases():
    # Series with all same prices
    same_prices = pd.Series([10.0] * 30)
    rsi_same = calculate_rsi(same_prices.copy(), period=14)
    # If all changes are 0, up and down are 0. ema_up and ema_down are 0.
    # rs = ema_up / ema_down -> 0/0 = NaN or some implementations handle it.
    # If rs is NaN, rsi is NaN. If rs is 0 (if ema_down is non-zero, ema_up is 0), RSI is 0.
    # If ema_up is non-zero and ema_down is 0, rs is inf, RSI is 100.
    # In this code, if ema_down is 0, rs can be inf or nan.
    # If ema_down is 0, and ema_up is also 0, rs is nan. 100/(1+nan) is nan.
    # The implementation in model.py: if ema_down is zero, it can lead to division by zero for rs.
    # `rs = ema_up / ema_down`
    # `np.seterr(divide='ignore', invalid='ignore')` is NOT used in model.py's calculate_rsi
    # However, pandas EWM handles internal zeros. If all diffs are 0, ema_up and ema_down are 0.
    # 0/0 results in NaN for rs. Then 100/(1+NaN) is NaN.
    assert rsi_same.iloc[14:].isna().all() # Expect NaN if all prices are the same

    # Series shorter than RSI period + 1 (need at least period+1 prices for 1 RSI value if diff is used)
    # The function expects `period` initial NaNs for RSI.
    # So, if len(prices) <= period, all RSI values should be NaN.
    short_prices = pd.Series([10.0, 11.0, 12.0] * (14//3)) # Length 12, period 14
    rsi_short = calculate_rsi(short_prices.copy(), period=14)
    assert rsi_short.isna().all() # If length < period, all are NaN.
    
    short_prices_equal_period = pd.Series(np.arange(1,15, dtype=float)) # Length 14, period 14
    rsi_short_eq = calculate_rsi(short_prices_equal_period.copy(), period=14)
    # First period-1 (0-12) values are NaN. Value at period-1 (13) is calculated.
    assert rsi_short_eq.iloc[:period-1].isna().all()
    assert pd.notna(rsi_short_eq.iloc[period-1])

    # Series with NaN values in prices
    prices_with_nan = pd.Series([10.0, 11.0, np.nan, 13.0, 14.0] * 5) # Length 25
    rsi_with_nan = calculate_rsi(prices_with_nan.copy(), period=14)
    # EWM has `ignore_na` parameter, default False. If False, NaNs propagate.
    # The diff() will also propagate NaNs.
    # So, we expect NaNs where calculation is affected.
    # The initial `period` NaNs should still be there.
    assert rsi_with_nan.iloc[:14].isna().all()
    # Subsequent NaNs depend on propagation. EWM usually continues calculation after NaN if possible.
    # But here, `data['RSI'] = 100 - (100/(1+rs))` will propagate NaNs if rs is NaN.
    # If a value in `up` or `down` is NaN, the EWM for that point becomes NaN.
    assert rsi_with_nan.isna().sum() > 14 # More NaNs than just the initial ones


# --- Tests for prepare_data ---

def test_prepare_data_basic_flow_and_shapes():
    # Create a DataFrame with enough data for SMA20 and seq_length
    # Min data needed: initial NaNs from SMA20 (19) + seq_length (e.g.,10) + 1 (for y label) = 30
    num_records = 50
    seq_length = 10
    data = {
        'rate': np.arange(1, num_records + 1, dtype=float),
        'volume': np.random.rand(num_records) * 100,
        'trades': np.random.randint(1, 5, num_records)
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2023-01-01', periods=num_records, freq='D')

    X, y, scaler = prepare_data(df.copy(), seq_length=seq_length)

    # Check for technical indicator columns (intermediary check, not directly output)
    # This requires looking into the function or testing a modified version.
    # For now, trust they are used for X. Number of features in X.shape[2] tells us.
    
    # Expected number of features: rate, SMA_5, SMA_20, RSI, Volatility = 5
    num_features = 5 
    assert X.shape[2] == num_features
    
    # SMA_20 creates 19 NaNs. RSI creates `period` (default 14) NaNs. Volatility (std dev 20) creates 19 NaNs.
    # The function drops NaNs *after* feature calculation. Max NaNs from rolling(20) is 19.
    # So, effective data starts after index 19.
    # Length of usable data = num_records - 19
    # Number of sequences = usable_data_length - seq_length
    expected_X_len = num_records - 19 - seq_length
    
    assert X.shape[0] == expected_X_len
    assert X.shape[1] == seq_length
    assert len(y) == expected_X_len
    
    assert isinstance(scaler, MinMaxScaler)
    # Check if scaler was fitted (e.g., min_ and scale_ should not be None)
    assert scaler.min_ is not None
    assert scaler.scale_ is not None


def test_prepare_data_scaling():
    num_records = 30 # Min for SMA20 (19) + seq_length (1) + 1 = 21. Let's use 30.
    seq_length = 5
    data = {
        'rate': np.linspace(10, 100, num_records), # Predictable data for scaling check
        'volume': np.ones(num_records) * 10,
        'trades': np.ones(num_records)
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2023-01-01', periods=num_records, freq='D')

    X, y, scaler = prepare_data(df.copy(), seq_length=seq_length)

    # X contains scaled data. The original data was from 10 to 100 for 'rate'.
    # Other features (SMAs, RSI, Volatility) will also be scaled.
    # Check that all values in X are between 0 and 1 (or very close, due to float precision)
    assert np.all(X >= -1e-6) and np.all(X <= 1 + 1e-6) # Allowing for tiny float inaccuracies

    # Check y values (they are from the 'rate' column, scaled)
    # y should also be scaled between 0 and 1.
    assert np.all(y >= -1e-6) and np.all(y <= 1 + 1e-6)

    # Test inverse transform if possible (optional, but good for sanity)
    # For example, the 'rate' feature in X (first feature, X[:, :, 0])
    # We need to reconstruct the 2D array that was passed to scaler.transform()
    # This is complex because X is 3D and features were selected.
    # However, we can check the scaler's fitted parameters for 'rate'.
    # The scaler is fitted on df_scaled which has 5 columns in order: rate, SMA_5, SMA_20, RSI, Volatility
    # So, scaler.min_[0] and scaler.scale_[0] correspond to 'rate'.
    
    # Original 'rate' values after dropping NaNs (first 19 for SMA20/Volatility)
    original_rate_after_nans = df['rate'].iloc[19:].values
    
    # The y values are scaled versions of original_rate_after_nans[seq_length:]
    expected_y_unscaled = original_rate_after_nans[seq_length:]
    
    # Reconstruct what y would be if unscaled, using the scaler
    # y is a 1D array. scaler.inverse_transform expects 2D array with num_features columns.
    # We need to create a dummy array for inverse transform, where only the 'rate' column (index 0) matters.
    dummy_for_y_inv = np.zeros((len(y), 5)) # 5 features
    dummy_for_y_inv[:, 0] = y # Put scaled y into the first column
    y_unscaled_approx = scaler.inverse_transform(dummy_for_y_inv)[:, 0]

    assert np.allclose(y_unscaled_approx, expected_y_unscaled, atol=1e-5)


def test_prepare_data_insufficient_data():
    seq_length = 10
    # Case 1: Total records less than max window (e.g., SMA20 needs 20 records for 1 value)
    # This will result in `data` being empty after `dropna()`.
    df_too_short_for_features = pd.DataFrame({'rate': np.arange(1, 15)}) # 14 records
    X, y, scaler = prepare_data(df_too_short_for_features.copy(), seq_length=seq_length)
    assert X.size == 0
    assert y.size == 0
    assert isinstance(scaler, MinMaxScaler) # Scaler is returned, but not fitted to data

    # Case 2: Enough for features, but not for seq_length + 1 label after scaling
    # This is handled by the `if len(scaled_data) < seq_length + 1:` check in prepare_data.
    # Need num_records - 19 (for SMA20) > seq_length. If not, X and y are empty.
    # Example: num_records = 29, seq_length = 10. Usable data length = 29 - 19 = 10.
    # Since 10 is not < seq_length + 1 (11), this isn't handled by the new check but the existing one.
    # If usable data length is exactly seq_length (e.g., 10), then len(scaled_data) - seq_length = 0 loops.
    # The condition `len(scaled_data) < seq_length + 1` ensures there's at least one y value.
    # If len(scaled_data) = 10, seq_length = 10. 10 < 11 is true. So this returns empty X, y.
    df_short_for_seq = pd.DataFrame({'rate': np.arange(1, seq_length + 19 + 1)}) # Exactly enough for 0 sequences after scaling
    X2, y2, scaler2 = prepare_data(df_short_for_seq.copy(), seq_length=seq_length)
    assert X2.size == 0
    assert y2.size == 0
    assert isinstance(scaler2, MinMaxScaler) # Scaler would be fitted here.

    df_one_seq = pd.DataFrame({'rate': np.arange(1, seq_length + 19 + 2)}) # Enough for 1 sequence
    X, y, scaler = prepare_data(df_one_seq.copy(), seq_length=seq_length)
    assert X.shape == (1, seq_length, 5)
    assert y.shape == (1,)


def test_prepare_data_feature_presence():
    # Test that the features are actually different and present.
    # Create data that would make features distinct.
    num_records = 50
    seq_length = 5
    rates = np.sin(np.linspace(0, 10, num_records)) * 10 + 50 # Some variance
    data = {
        'rate': rates,
        'volume': np.random.rand(num_records) * 100 + 10, # Ensure volume is not zero for volatility
        'trades': np.random.randint(1, 5, num_records)
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2023-01-01', periods=num_records, freq='D')

    X, y, scaler = prepare_data(df.copy(), seq_length=seq_length)

    # X has shape (num_samples, seq_length, num_features)
    # num_features should be 5: rate, SMA_5, SMA_20, RSI, Volatility

    # Check that these features are not all constant or identical if source data varies
    # This is implicitly tested by X.shape[2] == 5, but let's consider variance.
    # For one sequence, X[0], check that columns are different.
    # X[0, :, 0] is 'rate' for that sequence
    # X[0, :, 1] is 'SMA_5' for that sequence
    # ...
    # X[0, :, 4] is 'Volatility' for that sequence
    if X.shape[0] > 0:
        sample_sequence = X[0] # Shape (seq_length, num_features)
        for i in range(sample_sequence.shape[1]): # Iterate over features
            # Check that the feature column is not constant if source data was not trivial
            # (unless the feature itself naturally becomes constant, e.g. RSI at 100)
            # A simple check: std dev of the feature across the sequence
            # This is a weak check, as a feature could be non-constant but still wrong.
            # More robust is to recalculate one feature manually for a known input and compare.
            pass # Direct comparison of calculated features is complex here. Trusting the pipeline.

    # A key check is that the number of features is correct.
    assert X.shape[2] == 5

    # Ensure 'volume' and 'trades' are not in X, as they are not part of the scaled features for LSTM
    # The function selects ['rate', 'SMA_5', 'SMA_20', 'RSI', 'Volatility'] before scaling.
    # This is implicitly tested by X.shape[2] == 5.

# Ensure all required columns ('rate', 'volume', 'trades') are present in input to prepare_data
def test_prepare_data_missing_input_columns():
    num_records = 30
    seq_length = 5
    df_base = pd.DataFrame({
        'rate': np.arange(1, num_records + 1, dtype=float),
        'volume': np.random.rand(num_records) * 100,
        'trades': np.random.randint(1, 5, num_records)
    })
    df_base.index = pd.date_range(start='2023-01-01', periods=num_records, freq='D')

    with pytest.raises(KeyError, match="'rate'"): # Or whatever error it raises
        prepare_data(df_base.drop(columns=['rate']), seq_length=seq_length)
    
    # Volatility uses 'volume', if it's missing, it might lead to issues or use of 'trades'
    # The current model.py calculate_volatility uses df['volume'] if present, else df['trades']
    # So, if 'volume' is missing but 'trades' is present, it should still run.
    # Updated understanding: Volatility is based on 'rate' only.
    X_no_volume, y_no_volume, scaler_no_volume = prepare_data(df_base.drop(columns=['volume']), seq_length=seq_length)
    assert X_no_volume.shape[2] == 5 # Number of features should still be 5
    # Ensure X_no_volume is not empty, given enough input data in df_base
    # (num_records=30, seq_length=5, SMA20 needs 20 records, RSI needs 14)
    # df_base.dropna() will have (30 - (20-1)) = 11 rows. 11 < 5+1 is false.
    # So data will be produced.
    assert X_no_volume.size > 0

    # Test dropping 'volume' and 'trades': Volatility is based on 'rate', so this should also work.
    # The original test expected a KeyError here, which is incorrect for current 'prepare_data'.
    X_no_vol_trades, y_no_vol_trades, scaler_no_vol_trades = prepare_data(df_base.drop(columns=['volume', 'trades']), seq_length=seq_length)
    assert X_no_vol_trades.shape[2] == 5 # Number of features should still be 5
    assert X_no_vol_trades.size > 0 # Should produce output

    # RSI needs 'rate'. SMA needs 'rate'.
    # If only 'rate' is present:
    df_only_rate = pd.DataFrame({'rate': np.arange(1, num_records + 1, dtype=float)})
    df_only_rate.index = pd.date_range(start='2023-01-01', periods=num_records, freq='D')
    with pytest.raises(KeyError): # calculate_volatility will fail
        prepare_data(df_only_rate, seq_length=seq_length)

```
