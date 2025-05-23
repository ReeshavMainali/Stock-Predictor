import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import app as flask_app # Your Flask app instance
from unittest.mock import MagicMock, patch
import pandas as pd # For mock_model_functions
import numpy as np

@pytest.fixture(scope='session') # Scope app to session for efficiency
def flask_test_app():
    # Flask app configuration for testing
    flask_app.config.update({  # flask_app here refers to the globally imported app instance
        "TESTING": True,
        "WTF_CSRF_ENABLED": False, 
        # "LOGIN_DISABLED": True, 
    })
    yield flask_app

@pytest.fixture() # Default scope is function
def client(flask_test_app): # Use the renamed fixture
    return flask_test_app.test_client()

@pytest.fixture
def mock_db_manager(mocker): # Use mocker fixture for easier patching
    # Create a mock for DatabaseManager
    # This mock will be used by tests to control data flow from the database
    # Patch where DatabaseManager is instantiated or used in app.py
    # Assuming it's imported as `from functions.db_data_manager import DatabaseManager`
    # and then an instance is created, or its methods are called directly if static/class methods
    # If app.py does `import functions.db_data_manager`, then it would be `functions.db_data_manager.DatabaseManager`
    # Based on previous files, app.py likely does: `from functions.db_data_manager import DatabaseManager`
    # And then `db_manager = DatabaseManager()`
    # So, the target for patching is where this instance `db_manager` is located in `app.py`
    # Or, if `DatabaseManager` is instantiated inside routes, patch `app.DatabaseManager`
    
    # Let's assume `db_manager` is an instance available globally in `app.py` (e.g., app.db_manager)
    # Or, more likely, `DatabaseManager` is imported and instantiated in `app.py`'s global scope or app factory.
    # The prompt says `patch('app.DatabaseManager')`, implying `DatabaseManager` is a name in `app.py`'s namespace.

    mock_instance = MagicMock(spec=True) # Use spec for better mocking

    # Configure common return values for methods that are frequently called
    mock_instance.get_all_company_details.return_value = [{'symbol': 'AAPL', 'name': 'Apple Inc.', 'has_model': True}]
    # Ensure stock_history returns data that prepare_prediction_data can process
    mock_instance.get_stock_history.return_value = [
        {'date': '2023-01-01', 'close': 150.0, 'symbol': 'AAPL'},
        {'date': '2023-01-02', 'close': 151.0, 'symbol': 'AAPL'}
    ] * 30 # Ensure enough data for predictions
    mock_instance.get_cached_stocks.return_value = None # Default to no cache
    mock_instance.get_unique_symbols.return_value = ['AAPL', 'MSFT']
    mock_instance.get_latest_stock_data.return_value = {'symbol': 'AAPL', 'close': 155.0, 'volume': 1000, 'trades': 100}
    mock_instance.get_stock_statistics.return_value = {'avg_close': 150.0, 'max_close':160.0, 'min_close':140.0, 'avg_volume':1000}
    
    # For get_model_and_scaler, ensure the mock model has a predict method
    mock_keras_model = MagicMock()
    mock_keras_model.predict.return_value = [[0.9]] * 7 # 7 days prediction, 1 feature
    mock_scaler = MagicMock()
    # Mock the scaler's expected methods if they are called, e.g. inverse_transform
    mock_scaler.inverse_transform.return_value = [[float(i)] for i in range(160, 167)]


    mock_instance.get_model_and_scaler.return_value = (mock_keras_model, mock_scaler)
    mock_instance.get_existing_model_symbols.return_value = ['AAPL']
    # get_model_metadata in app.py is used to get a list of dicts
    mock_instance.get_model_metadata.return_value = [{'symbol': 'AAPL', 'metadata': {'num_features': 5, 'sequence_length': 10, 'trained_at': '2023-01-01'}}]
    mock_instance.save_model_and_scaler.return_value = True # Simulate successful save

    # Patch the db_manager instance within the app module.
    # This assumes app.py has `db_manager = DatabaseManager()` at the module level.
    # If DatabaseManager is instantiated per request or differently, this patching target needs adjustment.
    # Given the structure of typical Flask apps, `app.db_manager` or patching `DatabaseManager`
    # where it's imported by `app.py` are common.
    # The prompt uses `patch('app.DatabaseManager')` - this means `DatabaseManager` is a name in `app.py`
    # that is called to create an instance, or its class/static methods are used.
    
    # If `DatabaseManager` is imported in `app.py` and then instantiated within functions,
    # the patch target should be `app.DatabaseManager`.
    # Example app.py:
    # from functions.db_data_manager import DatabaseManager
    # def some_route():
    #   db = DatabaseManager()
    #   db.get_all_company_details()

    # The prompt `patch('app.DatabaseManager')` suggests it's used like `app.DatabaseManager().method()`
    # or `app.DatabaseManager.static_method()`.
    # Let's stick to the prompt's `patch('app.DatabaseManager')`
    
    # We need to ensure that when `app.DatabaseManager()` is called, it returns our mock_instance.
    # So, `app.DatabaseManager` itself should be a mock that, when called, returns `mock_instance`.
    # Or, if `db_manager` is a global instance in `app.py`, patch `app.db_manager`.
    # Let's assume `app.py` does `from functions.db_data_manager import DatabaseManager` and then it's used.
    # So we patch `'app.DatabaseManager'` to be a mock constructor.
    
    # The provided solution in the prompt uses `with patch('app.DatabaseManager') as MockDBManager: mock_instance = MockDBManager.return_value`
    # This means app.DatabaseManager is treated as the class itself. So when app.DatabaseManager() is called, it returns mock_instance.
    
    # For the purpose of this fixture, we create the mock_instance and then tell the patcher
    # that this should be the return_value of the `DatabaseManager` constructor when called from `app.py`.
    # The actual patching happens in the test functions or a higher-scoped fixture if needed.
    # Here, we are defining what the mock *should be* if `app.DatabaseManager` is patched.
    # A common pattern is to apply the patch directly in the fixture:
    
    patcher = patch('app.DatabaseManager', return_value=mock_instance)
    # It's better to start the patcher here and stop it after the yield if this fixture is responsible for patching.
    # However, the prompt's example test `test_index_route(client, mock_db_manager)` implies `mock_db_manager` IS the patched object.
    # This means the patch should be active for the duration of the test using this fixture.
    
    # Correct way for this fixture to provide the mock and manage the patch:
    # No, the prompt's example `with patch('app.DatabaseManager') as MockDBManager: yield MockDBManager.return_value` is for the fixture itself.
    # Let's re-read the prompt's conftest.py structure.
    # `with patch('app.DatabaseManager') as MockDBManager: mock_instance = MockDBManager.return_value; ...; yield mock_instance`
    # This is the correct structure for the fixture.

    # This fixture should manage the patch's lifecycle.
    # The mock_db_manager IS the mock_instance.
    # The patch should be started here.
    
    # The prompt structure:
    # @pytest.fixture
    # def mock_db_manager():
    #     with patch('app.DatabaseManager') as MockDBManager:
    #         mock_instance = MockDBManager.return_value
    #         ... configure mock_instance ...
    #         yield mock_instance

    # This is what I will implement.
    return mock_instance # This is not right. The patch needs to be active.

@pytest.fixture
def PatchedDBManager(): # Renaming to clarify it's the patch controller
    with patch('app.DatabaseManager') as MockDBManager:
        mock_instance = MockDBManager.return_value
        # Configure common return values
        mock_instance.get_all_company_details.return_value = [{'symbol': 'AAPL', 'name': 'Apple Inc.', 'has_model': True}]
        mock_instance.get_stock_history.return_value = [
            {'date': f'2023-01-{i:02d}', 'close': 150.0+i, 'symbol': 'AAPL', 'volume': 1000+i*10, 'trades': 10+i} for i in range(1,61)
        ]
        mock_instance.get_cached_stocks.return_value = None
        mock_instance.get_unique_symbols.return_value = ['AAPL', 'MSFT']
        mock_instance.get_latest_stock_data.return_value = {'symbol': 'AAPL', 'close': 155.0, 'volume': 1000, 'trades':100}
        mock_instance.get_stock_statistics.return_value = {'avg_close': 150.0, 'max_close':160.0, 'min_close':140.0, 'avg_volume':1000.0}
        
        mock_keras_model = MagicMock(name="MockKerasModel")
        # The model's predict method is called with scaled data.
        # It should return predictions in a way that inverse_transform can handle.
        # If model predicts 1 feature, output shape is (n_preds, 1).
        mock_keras_model.predict.return_value = np.array([[0.5 + i*0.01] for i in range(7)]) # 7 days, 1 feature scaled
        
        mock_scaler = MagicMock(name="MockScaler")
        # inverse_transform is called on the model's predictions.
        # Input to inverse_transform is (n_preds, 1). Output should be (n_preds, 1) with actual prices.
        mock_scaler.inverse_transform.return_value = np.array([[160.0 + i] for i in range(7)])
        # transform is called on historical data (n_samples, n_features)
        # It should return (n_samples, n_features) scaled.
        # Assuming 5 features as in app.py's prepare_prediction_data context
        mock_scaler.transform.return_value = np.random.rand(60, 5) # 60 days, 5 features scaled


        mock_instance.get_model_and_scaler.return_value = (mock_keras_model, mock_scaler)
        mock_instance.get_existing_model_symbols.return_value = ['AAPL']
        mock_instance.get_model_metadata.return_value = [{'symbol': 'AAPL', 'metadata': {'num_features': 5, 'sequence_length': 10, 'trained_at': '2023-01-01T12:00:00'}}]
        mock_instance.save_model_and_scaler.return_value = True
        mock_instance.get_company_by_symbol.return_value = {'symbol': 'AAPL', 'name': 'Apple Inc.'}


        yield mock_instance # This is the mock_db_manager instance

@pytest.fixture
def mock_db_manager(PatchedDBManager): # Keep original name for tests
    return PatchedDBManager


@pytest.fixture
def mock_model_functions():
    # Mock functions from model.py that are imported into app.py
    # Paths should be 'app.function_name'
    with patch('app.train_model') as mock_train, \
         patch('app.predict_future') as mock_predict, \
         patch('app.calculate_rsi') as mock_rsi, \
         patch('app.preprocess_transaction_data') as mock_preprocess, \
         patch('app.prepare_data') as mock_prepare_data, \
         patch('app.plot_model_structure') as mock_plot_model_structure: # Added this
        
        # train_model returns: model, scaler, history_df, X_test, y_test, y_pred
        # For simplicity, only mock model and scaler unless others are used by app routes
        mock_keras_model_trained = MagicMock(name="MockKerasModelTrained")
        mock_scaler_trained = MagicMock(name="MockScalerTrained")
        mock_train.return_value = (mock_keras_model_trained, mock_scaler_trained, pd.DataFrame(), np.array([]), np.array([]), np.array([]))
        
        # predict_future is called with (model, scaler, historical_data, n_future_days, sequence_length)
        # It returns a list of predicted prices (not scaled).
        mock_predict.return_value = [160.0 + i for i in range(7)] # 7 days of predictions
        
        # preprocess_transaction_data returns a DataFrame
        # Passthrough is fine if complex logic isn't needed for the test.
        mock_preprocess.side_effect = lambda df, symbol=None: df 
        
        # calculate_rsi returns a pandas Series
        mock_rsi.return_value = pd.Series([50.0] * 100) # Mock RSI data for 100 days
        
        # prepare_data returns X, y, scaler
        # X: (samples, seq_len, features), y: (samples,), scaler: fitted scaler
        # This is used by train_model, so might not need direct mock if train_model is mocked.
        # However, if app.py calls it directly for some reason:
        mock_X = np.random.rand(50, 10, 5) # 50 samples, 10 seq_len, 5 features
        mock_y = np.random.rand(50)
        mock_scaler_prepared = MagicMock(name="MockScalerPrepared")
        mock_prepare_data.return_value = (mock_X, mock_y, mock_scaler_prepared)

        mock_plot_model_structure.return_value = "mock_plot_path.png" # Path to a dummy image

        yield {
            'train_model': mock_train,
            'predict_future': mock_predict,
            'calculate_rsi': mock_rsi,
            'preprocess_transaction_data': mock_preprocess,
            'prepare_data': mock_prepare_data,
            'plot_model_structure': mock_plot_model_structure
        }
