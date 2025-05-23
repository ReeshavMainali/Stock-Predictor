import pytest
from flask import url_for
import pandas as pd
from unittest.mock import MagicMock, ANY # ANY is useful for asserting calls with some dynamic args

# Basic test to see if the setup works
def test_index_route(client, mock_db_manager): # mock_db_manager is PatchedDBManager from conftest
    # Test default behavior (no cache)
    mock_db_manager.get_cached_stocks.return_value = None 
    mock_db_manager.get_unique_symbols.return_value = ['TEST1', 'TEST2']
    # Mock side_effect for get_latest_stock_data and get_stock_statistics if they are called per symbol
    mock_db_manager.get_latest_stock_data.side_effect = [
        {'symbol': 'TEST1', 'close': 100, 'volume': 10, 'trades': 1000}, # Corrected: rate -> close
        {'symbol': 'TEST2', 'close': 200, 'volume': 20, 'trades': 2000}
    ]
    mock_db_manager.get_stock_statistics.side_effect = [
        {'avg_close': 90, 'symbol': 'TEST1'}, # Corrected: avg_price -> avg_close
        {'avg_close': 190, 'symbol': 'TEST2'}
    ]
    
    response = client.get(url_for('index'))
    assert response.status_code == 200
    assert b"Dashboard" in response.data
    assert b"TEST1" in response.data
    assert b"TEST2" in response.data
    assert mock_db_manager.get_cached_stocks.called # Called once
    assert mock_db_manager.get_unique_symbols.call_count > 0 # Might be called if cache is None
    
    # Reset call counts for next assertions if needed or use separate tests
    mock_db_manager.reset_mock()


def test_index_route_with_cache(client, mock_db_manager):
    cached_data = [{'symbol': 'CACHE', 'close': 200, 'change': 5.0}] # Corrected: rate -> close
    mock_db_manager.get_cached_stocks.return_value = cached_data
    
    response = client.get(url_for('index'))
    assert response.status_code == 200
    assert b"CACHE" in response.data
    assert b"5.0" in response.data # Check for change percentage
    mock_db_manager.get_cached_stocks.assert_called_once()
    mock_db_manager.get_unique_symbols.assert_not_called() # Should not be called if cache hit


def test_history_route_with_symbol(client, mock_db_manager):
    mock_db_manager.get_stock_history.return_value = [
        {'date': '2023-01-01', 'close': 150.0, 'symbol': 'AAPL', 'volume':100, 'trades':10} # Corrected: rate -> close
    ]
    mock_db_manager.get_company_by_symbol.return_value = {'symbol': 'AAPL', 'name': 'Apple Inc.'}

    response = client.get(url_for('history', symbol='AAPL'))
    assert response.status_code == 200
    assert b"AAPL" in response.data
    assert b"Apple Inc." in response.data
    assert b"Stock History" in response.data
    mock_db_manager.get_stock_history.assert_called_with('AAPL', 90) # Default days
    mock_db_manager.get_company_by_symbol.assert_called_with('AAPL')


def test_history_route_no_symbol(client, mock_db_manager):
    response = client.get(url_for('history')) # No symbol
    assert response.status_code == 200
    assert b"Displaying all available stock data" in response.data # Updated based on template
    mock_db_manager.get_all_stocks.assert_called_once() # Check if get_all_stocks is called

def test_history_route_symbol_not_found(client, mock_db_manager):
    mock_db_manager.get_company_by_symbol.return_value = None
    mock_db_manager.get_stock_history.return_value = []
    
    response = client.get(url_for('history', symbol='UNKNOWN'))
    assert response.status_code == 200 # Or 404 if you change behavior
    assert b"No data found for symbol UNKNOWN" in response.data
    mock_db_manager.get_company_by_symbol.assert_called_with('UNKNOWN')


def test_predict_route_with_symbol(client, mock_db_manager, mock_model_functions):
    symbol = 'AAPL'
    # Ensure get_stock_history provides enough data (at least seq_length + features_rolling_window)
    # Default seq_length=60. SMA20 needs 19 prior. RSI 14 needs 13 prior. Volatility 20 needs 19. Max is 19.
    # So, 60 + 19 = 79 records needed for prepare_data.
    # app.py uses default sequence_length = 60 from model.py
    # get_stock_history in app.py for predict route fetches 90 days.
    stock_history_data = [{'date': f'2023-01-{i:02d}', 'close': 150+i, 'symbol': symbol, 'volume':1000+i*10, 'trades':100+i} for i in range(1, 91)]
    mock_db_manager.get_stock_history.return_value = stock_history_data
    mock_db_manager.get_company_by_symbol.return_value = {'symbol': symbol, 'name': 'Apple Inc.'}

    # mock_db_manager.get_model_and_scaler is already configured in conftest's PatchedDBManager
    # mock_model_functions['predict_future'] is also configured in conftest

    response = client.get(url_for('predict', symbol=symbol))
    assert response.status_code == 200
    assert f"Stock Prediction for {symbol}".encode() in response.data
    
    mock_db_manager.get_stock_history.assert_called_once_with(symbol, 90) # Default days for prediction context
    mock_db_manager.get_model_and_scaler.assert_called_once_with(symbol)
    
    # Check that predict_future was called with the model, scaler, and processed historical data
    # The historical data passed to predict_future is the scaled data from prepare_prediction_data
    # We can use ANY for the data part if it's complex to reconstruct.
    mock_model_functions['predict_future'].assert_called_once_with(
        mock_db_manager.get_model_and_scaler.return_value[0], # model
        mock_db_manager.get_model_and_scaler.return_value[1], # scaler
        ANY, # historical_data (scaled)
        n_future_days=7, # default from app
        sequence_length=60 # default from model.py, used in app.py's prepare_prediction_data
    )


def test_predict_route_insufficient_data(client, mock_db_manager):
    symbol = 'FEWDATA'
    mock_db_manager.get_stock_history.return_value = [{'date': '2023-01-01', 'close': 150.0, 'symbol':symbol}] # Not enough
    mock_db_manager.get_company_by_symbol.return_value = {'symbol': symbol, 'name': 'Few Data Inc.'}
    mock_db_manager.get_model_and_scaler.return_value = (MagicMock(), MagicMock()) # Model exists

    response = client.get(url_for('predict', symbol=symbol))
    assert response.status_code == 200
    assert b"Insufficient historical data to make a prediction" in response.data # Updated message
    mock_db_manager.get_stock_history.assert_called_once_with(symbol, 90)


def test_predict_route_no_model(client, mock_db_manager):
    symbol = 'NOMODEL'
    # Provide enough history so that's not the reason for failure
    stock_history_data = [{'date': f'2023-01-{i:02d}', 'close': 150+i, 'symbol': symbol, 'volume':100, 'trades':10} for i in range(1, 91)]
    mock_db_manager.get_stock_history.return_value = stock_history_data
    mock_db_manager.get_company_by_symbol.return_value = {'symbol': symbol, 'name': 'No Model Inc.'}
    mock_db_manager.get_model_and_scaler.return_value = (None, None) # No model

    response = client.get(url_for('predict', symbol=symbol))
    assert response.status_code == 200
    assert b"No trained model available for this stock yet" in response.data # Updated message
    mock_db_manager.get_model_and_scaler.assert_called_once_with(symbol)


def test_train_single_model_route_success(client, mock_db_manager, mock_model_functions):
    symbol = "NEWTRAIN"
    mock_db_manager.get_existing_model_symbols.return_value = [] # Model doesn't exist yet
    # train_model needs enough data. Default seq_len=60, plus window (19) means ~79 records.
    # get_stock_history in train_single_model fetches 365 days.
    stock_data = [{'date': f'2022-01-{i:02d}', 'symbol': symbol, 'close': 100+i, 'volume':10, 'trades':1} for i in range(1, 100)]
    mock_db_manager.get_stock_history.return_value = stock_data
    
    # Mock the preprocess_transaction_data to return a DataFrame
    # The actual data in the DataFrame for preprocess matters for prepare_data.
    # Let's make it a proper DataFrame.
    mock_df = pd.DataFrame(stock_data)
    # Ensure 'transaction_date' is datetime
    if not stock_data: # handle empty case if necessary for some test
        mock_df = pd.DataFrame(columns=['transaction_date', 'symbol', 'close', 'volume', 'trades'])
    else:
        mock_df['transaction_date'] = pd.to_datetime(mock_df['date'])
        mock_df = mock_df.set_index('transaction_date')

    mock_model_functions['preprocess_transaction_data'].return_value = mock_df

    # Mock save_model_and_scaler to return True
    mock_db_manager.save_model_and_scaler.return_value = True

    response = client.get(url_for('train_single_model', symbol=symbol))
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['symbol'] == symbol
    assert json_data['status'] == 'success'
    assert 'Model trained and saved successfully' in json_data['message']
    
    mock_db_manager.get_stock_history.assert_called_once_with(symbol, 365)
    mock_model_functions['preprocess_transaction_data'].assert_called_once() # With df, symbol
    mock_model_functions['train_model'].assert_called_once() # With preprocessed_df, symbol
    mock_db_manager.save_model_and_scaler.assert_called_once()


def test_train_single_model_route_already_exists(client, mock_db_manager):
    symbol = "EXISTING"
    mock_db_manager.get_existing_model_symbols.return_value = [symbol] # Model already exists
    
    response = client.get(url_for('train_single_model', symbol=symbol))
    assert response.status_code == 200 # Or 400 if you prefer for client error
    json_data = response.get_json()
    assert json_data['symbol'] == symbol
    assert json_data['status'] == 'skipped'
    assert 'Model already exists' in json_data['message']
    mock_db_manager.get_stock_history.assert_not_called()


def test_train_single_model_route_insufficient_data_for_training(client, mock_db_manager, mock_model_functions):
    symbol = "FEWDATATRAIN"
    mock_db_manager.get_existing_model_symbols.return_value = []
    mock_db_manager.get_stock_history.return_value = [{'date': '2023-01-01', 'close': 100}] # Too few
    
    mock_df = pd.DataFrame([{'date': '2023-01-01', 'close': 100, 'symbol':symbol, 'volume':1, 'trades':1}])
    mock_df['transaction_date'] = pd.to_datetime(mock_df['date'])
    mock_df = mock_df.set_index('transaction_date')
    mock_model_functions['preprocess_transaction_data'].return_value = mock_df

    # train_model itself might raise error or return None if data is bad
    # Let's assume train_model returns (None, None, ...) if it fails due to data
    mock_model_functions['train_model'].return_value = (None, None, None, None, None, None)

    response = client.get(url_for('train_single_model', symbol=symbol))
    assert response.status_code == 200 # Or 500 if it's an internal error
    json_data = response.get_json()
    assert json_data['symbol'] == symbol
    assert json_data['status'] == 'error'
    assert 'Insufficient data to train model' in json_data['message'] # Or similar error
    mock_model_functions['train_model'].assert_called_once() # It was attempted
    mock_db_manager.save_model_and_scaler.assert_not_called()


def test_train_models_route(client, mock_db_manager, mock_model_functions):
    # This route calls train_single_model internally via redirect or direct call.
    # For simplicity, we'll mock the outcome of those calls if it were a direct call.
    # However, it seems to make HTTP requests to train_single_model.
    # This makes it harder to test without a live server or more complex mocking.
    # For now, let's assume we can check the initial symbols and if the process starts.
    # A better way would be to refactor train_single_model logic into a callable function.
    
    mock_db_manager.get_unique_symbols.return_value = ['SYM1', 'SYM2']
    # Mock the response from requests.get if we were to test the full loop
    # For now, just check that it tries to get symbols
    response = client.get(url_for('train_models'))
    assert response.status_code == 200
    assert b"Batch training process initiated" in response.data # Or similar confirmation
    mock_db_manager.get_unique_symbols.assert_called_once()
    # To fully test this, one might need `requests_mock` if it makes real sub-requests,
    # or to verify that background tasks are spawned if that's the mechanism.
    # The current implementation seems to be synchronous GET requests in a loop.

def test_models_route(client, mock_db_manager):
    mock_db_manager.get_model_metadata.return_value = [
        {'symbol': 'AAPL', 'metadata': {'num_features': 5, 'sequence_length': 10, 'trained_at': '2023-10-26T10:00:00'}},
        {'symbol': 'MSFT', 'metadata': {'num_features': 5, 'sequence_length': 10, 'trained_at': '2023-10-25T11:00:00'}}
    ]
    response = client.get(url_for('models'))
    assert response.status_code == 200
    assert b"Available Trained Models" in response.data
    assert b"AAPL" in response.data
    assert b"MSFT" in response.data
    assert b"Oct. 26, 2023, 10 a.m." in response.data # Check formatted date
    mock_db_manager.get_model_metadata.assert_called_once()


def test_api_stock_search_route(client, mock_db_manager):
    mock_db_manager.get_all_company_details.return_value = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.', 'has_model':True},
        {'symbol': 'AMZN', 'name': 'Amazon.com, Inc.', 'has_model':False},
        {'symbol': 'AAP', 'name': 'Advance Auto Parts', 'has_model':True}
    ]
    
    # Test with a query
    response = client.get(url_for('api_search_stocks', query='AAP'))
    assert response.status_code == 200
    json_data = response.get_json()
    assert len(json_data) == 2 # AAPL and AAP
    assert any(s['symbol'] == 'AAPL' for s in json_data)
    assert any(s['symbol'] == 'AAP' for s in json_data)

    # Test with no query (should return all)
    response_all = client.get(url_for('api_search_stocks'))
    assert response_all.status_code == 200
    json_data_all = response_all.get_json()
    assert len(json_data_all) == 3
    
    mock_db_manager.get_all_company_details.call_count == 2


def test_api_model_structure_route_success(client, mock_db_manager, mock_model_functions):
    symbol = "AAPL"
    # Mock get_model_and_scaler to return a model
    mock_model, _ = mock_db_manager.get_model_and_scaler.return_value 
    mock_model_functions['plot_model_structure'].return_value = "static/mock_plot.png" # Ensure this path is expected

    response = client.get(url_for('api_model_structure', symbol=symbol))
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['symbol'] == symbol
    assert 'static/mock_plot.png' in json_data['plot_path']
    
    mock_db_manager.get_model_and_scaler.assert_called_once_with(symbol)
    mock_model_functions['plot_model_structure'].assert_called_once_with(mock_model, symbol)


def test_api_model_structure_route_no_model(client, mock_db_manager):
    symbol = "NOMODELPLOT"
    mock_db_manager.get_model_and_scaler.return_value = (None, None) # No model found
    
    response = client.get(url_for('api_model_structure', symbol=symbol))
    assert response.status_code == 404
    json_data = response.get_json()
    assert json_data['error'] == 'Model not found'
    mock_db_manager.get_model_and_scaler.assert_called_once_with(symbol)

# Test custom error handler (e.g., for 404)
def test_404_error_handler(client):
    response = client.get("/non_existent_route_for_404")
    assert response.status_code == 404
    assert b"Page Not Found" in response.data # Check content from your 404.html or default
    assert b"The page you are looking for does not exist" in response.data

# Example of testing a POST request if you had one (e.g., a form submission)
# def test_example_post_route(client, mock_db_manager):
#     response = client.post(url_for('some_post_route'), data={'field1': 'value1', 'field2': 'value2'})
#     assert response.status_code == 302 # e.g. redirect after successful post
#     # or assert response.status_code == 200 and check for success message in response.data
#     mock_db_manager.some_method_called_by_post.assert_called_once_with('value1', 'value2')

# Consider testing other aspects:
# - Flashed messages if your app uses them.
# - Session variables if they are modified by routes.
# - Specific HTML elements or attributes if critical.
# - Response headers.
```
