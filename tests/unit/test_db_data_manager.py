import pytest
from unittest.mock import patch, MagicMock, mock_open
import mongomock
import pickle
import tempfile
import os
from datetime import datetime, timedelta

# Attempt to import TensorFlow and Keras for type hinting and spec.
# If not available, we'll use MagicMock without spec for these.
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model as KerasModel
except ImportError:
    tf = None
    KerasModel = MagicMock # Fallback if TensorFlow is not in the test environment

from functions.db_data_manager import DatabaseManager, MONGODB_URI, DB_NAME
# Assuming a MinMaxScaler-like object for scaler, which is usually from sklearn
# For testing purposes, a MagicMock is sufficient.
# from sklearn.preprocessing import MinMaxScaler # Example scaler

# Constants for testing
TEST_DB_NAME = "test_stock_db"
TEST_MONGO_URI = "mongodb://localhost:27017/" # Dummy URI for mongomock

@pytest.fixture
def mock_mongo_client():
    """Provides a mongomock client instance."""
    return mongomock.MongoClient(TEST_MONGO_URI)

@pytest.fixture
def db_manager(mock_mongo_client):
    """
    Provides a DatabaseManager instance with a mocked MongoDB client.
    Patches MONGODB_URI and DB_NAME used by DatabaseManager.
    Clears relevant collections before each test.
    """
    with patch('functions.db_data_manager.MongoClient', return_value=mock_mongo_client), \
         patch('functions.db_data_manager.MONGODB_URI', TEST_MONGO_URI), \
         patch('functions.db_data_manager.DB_NAME', TEST_DB_NAME):
        
        manager = DatabaseManager()
        
        # Clean up collections before each test to ensure isolation
        manager.db.companies.delete_many({})
        manager.db.stock_data.delete_many({})
        manager.db.models.delete_many({})
        manager.db.cache.delete_many({})
        manager.db.predictions.delete_many({}) # Assuming a predictions collection might exist
        
        yield manager # Use yield to allow cleanup after test if needed, though mongomock is in-memory

# --- Test Company Data ---
def test_get_all_company_details_no_companies(db_manager):
    assert db_manager.get_all_company_details() == []

def test_get_all_company_details_with_data(db_manager):
    db_manager.db.companies.insert_many([
        {"symbol": "AAPL", "name": "Apple Inc"},
        {"symbol": "GOOG", "name": "Alphabet Inc"}
    ])
    db_manager.db.models.insert_one({"symbol": "AAPL"}) # AAPL has a model

    companies = db_manager.get_all_company_details()
    assert len(companies) == 2
    
    aapl = next(c for c in companies if c["symbol"] == "AAPL")
    goog = next(c for c in companies if c["symbol"] == "GOOG")

    assert aapl["name"] == "Apple Inc"
    assert aapl["has_model"] is True
    assert goog["name"] == "Alphabet Inc"
    assert goog["has_model"] is False
    # Test sorting (by symbol)
    assert companies[0]["symbol"] == "AAPL"
    assert companies[1]["symbol"] == "GOOG"

# --- Test Stock Data ---
def test_get_all_stocks_no_data(db_manager):
    assert db_manager.get_all_stocks() == []

def test_get_all_stocks_with_data(db_manager):
    db_manager.db.stock_data.insert_many([
        {"symbol": "AAPL", "close": 150},
        {"symbol": "GOOG", "close": 2500}
    ])
    stocks = db_manager.get_all_stocks()
    assert len(stocks) == 2

def test_get_stock_by_symbol(db_manager):
    db_manager.db.stock_data.insert_one({"symbol": "AAPL", "close": 150, "date": "2023-01-01"})
    
    data = db_manager.get_stock_by_symbol("AAPL")
    assert len(data) == 1
    assert data[0]["close"] == 150
    
    assert db_manager.get_stock_by_symbol("NONE") == []

def test_get_stock_by_date_range(db_manager):
    db_manager.db.stock_data.insert_many([
        {"symbol": "AAPL", "date": "2023-01-01", "close": 100},
        {"symbol": "AAPL", "date": "2023-01-02", "close": 102},
        {"symbol": "AAPL", "date": "2023-01-05", "close": 105},
    ])
    
    data = db_manager.get_stock_by_date_range("AAPL", "2023-01-01", "2023-01-03")
    assert len(data) == 2
    assert data[0]["date"] == "2023-01-01"
    assert data[1]["date"] == "2023-01-02"

def test_get_unique_symbols(db_manager):
    db_manager.db.stock_data.insert_many([
        {"symbol": "AAPL"}, {"symbol": "GOOG"}, {"symbol": "AAPL"}
    ])
    symbols = db_manager.get_unique_symbols()
    assert len(symbols) == 2
    assert "AAPL" in symbols
    assert "GOOG" in symbols

def test_get_latest_stock_data(db_manager):
    db_manager.db.stock_data.insert_many([
        {"symbol": "AAPL", "date": "2023-01-01", "close": 100},
        {"symbol": "AAPL", "date": "2023-01-02", "close": 102}, # Latest for AAPL
        {"symbol": "GOOG", "date": "2023-01-01", "close": 2000},
    ])
    latest_aapl = db_manager.get_latest_stock_data("AAPL")
    assert latest_aapl is not None
    assert latest_aapl["close"] == 102
    
    assert db_manager.get_latest_stock_data("NONE") is None

def test_get_stock_statistics(db_manager):
    # This method uses aggregation, mongomock needs to support it.
    # mongomock's support for complex aggregations can sometimes be limited.
    db_manager.db.stock_data.insert_many([
        {"symbol": "AAPL", "close": 100, "volume": 1000},
        {"symbol": "AAPL", "close": 150, "volume": 2000}, # Max close for AAPL
        {"symbol": "AAPL", "close": 50,  "volume": 500},  # Min close for AAPL
    ])
    stats = db_manager.get_stock_statistics("AAPL")
    assert stats is not None
    assert stats["symbol"] == "AAPL"
    assert stats["avg_close"] == 100
    assert stats["max_close"] == 150
    assert stats["min_close"] == 50
    assert stats["avg_volume"] == (1000 + 2000 + 500) / 3
    
    assert db_manager.get_stock_statistics("NONE") is None


def test_get_stock_history(db_manager):
    db_manager.db.stock_data.insert_many([
        {"symbol": "AAPL", "date": "2023-01-01", "close": 100},
        {"symbol": "AAPL", "date": "2023-01-02", "close": 102},
    ])
    history = db_manager.get_stock_history("AAPL", 30) # Limit to 30 days
    assert len(history) == 2
    assert history[0]["date"] == "2023-01-01" # Sorted by date
    assert history[1]["close"] == 102

# --- Test Caching ---
def test_cache_top_stocks_and_get_cached_stocks(db_manager):
    assert db_manager.get_cached_stocks() is None # Cache empty initially
    
    top_stocks_data = [{"symbol": "AAPL", "change": 5.0}]
    db_manager.cache_top_stocks(top_stocks_data)
    
    cached_data = db_manager.get_cached_stocks()
    assert cached_data is not None
    assert len(cached_data) == 1
    assert cached_data[0]["symbol"] == "AAPL"
    
    # Test cache clearing aspect (implicitly tested by fixture, but can be explicit)
    db_manager.db.cache.delete_many({})
    assert db_manager.get_cached_stocks() is None

# --- Test Model Management ---

# Mock KerasModel correctly if TensorFlow is available
MockKerasModel = KerasModel if tf else MagicMock

@pytest.fixture
def mock_model_scaler():
    """Provides a mock Keras model and a mock scaler."""
    # If tf is not available, KerasModel is MagicMock, so spec=MagicMock() is fine.
    # Otherwise, spec=tf.keras.Model (or the imported KerasModel)
    mock_model = MagicMock(spec=MockKerasModel)
    mock_model.save = MagicMock() # Mock the save method specifically
    
    mock_scaler = MagicMock() # For a scaler, MagicMock is usually sufficient
    # If we needed to test specific scaler methods like fit_transform, inverse_transform:
    # mock_scaler.fit_transform = MagicMock(return_value=...)
    # mock_scaler.inverse_transform = MagicMock(return_value=...)
    return mock_model, mock_scaler

def test_save_model_and_scaler(db_manager, mock_model_scaler):
    mock_model, mock_scaler = mock_model_scaler
    symbol = "TESTMDL"
    
    # We need to mock tempfile.NamedTemporaryFile and os.remove
    # as the DatabaseManager uses them to save the model to a temporary file
    # before putting it into GridFS (or a collection in mongomock's case).
    
    mock_temp_file_instance = MagicMock()
    mock_temp_file_instance.name = "dummy_temp_path.keras" # Provide a name attribute

    with patch('tempfile.NamedTemporaryFile', return_value=mock_temp_file_instance) as mock_tempfile_constructor, \
         patch('os.remove') as mock_os_remove, \
         patch('builtins.open', mock_open(read_data=b"dummy_model_data")) as mock_file_open, \
         patch('pickle.dumps', return_value=b"pickled_scaler_data") as mock_pickle_dumps:

        # Simulate the context manager __enter__ and __exit__ for NamedTemporaryFile
        mock_tempfile_constructor.return_value.__enter__.return_value = mock_temp_file_instance
        
        db_manager.save_model_and_scaler(symbol, mock_model, mock_scaler, {"accuracy": 0.95})
        
        # Verify model.save was called with the temp file's name
        mock_model.save.assert_called_once_with(mock_temp_file_instance.name)
        
        # Verify that open was called to read the temp file
        mock_file_open.assert_called_once_with(mock_temp_file_instance.name, 'rb')
        
        # Verify scaler was pickled
        mock_pickle_dumps.assert_called_once_with(mock_scaler)
        
        # Verify os.remove was called for the temp file
        mock_os_remove.assert_called_once_with(mock_temp_file_instance.name)
        
        # Check data in mongomock
        saved_model_doc = db_manager.db.models.find_one({"symbol": symbol})
        assert saved_model_doc is not None
        assert saved_model_doc["symbol"] == symbol
        assert saved_model_doc["model_data"] == b"dummy_model_data" # GridFSOut in real mongo
        assert saved_model_doc["scaler_data"] == b"pickled_scaler_data"
        assert saved_model_doc["metadata"]["accuracy"] == 0.95
        assert "saved_at" in saved_model_doc

def test_get_model_and_scaler_found(db_manager, mock_model_scaler):
    original_mock_model, original_mock_scaler = mock_model_scaler
    symbol = "TESTMDL"
    model_data_bytes = b"dummy_keras_model_bytes"
    scaler_data_bytes = pickle.dumps(original_mock_scaler) # Use actual pickle for this

    db_manager.db.models.insert_one({
        "symbol": symbol,
        "model_data": model_data_bytes, # In real MongoDB, this would be a GridFS file ID
        "scaler_data": scaler_data_bytes,
        "metadata": {"info": "test_model"},
        "saved_at": datetime.utcnow()
    })

    # Mock load_model which is called by db_manager.get_model_and_scaler
    # It should return a new mock model instance for this test
    loaded_mock_model = MagicMock(spec=MockKerasModel)

    mock_temp_file_instance = MagicMock()
    mock_temp_file_instance.name = "retrieved_temp_path.keras"

    with patch('tensorflow.keras.models.load_model', return_value=loaded_mock_model) as mock_tf_load_model, \
         patch('tempfile.NamedTemporaryFile', return_value=mock_temp_file_instance) as mock_tempfile_constructor, \
         patch('os.remove') as mock_os_remove, \
         patch('builtins.open', mock_open()) as mock_file_open, \
         patch('pickle.loads', return_value=original_mock_scaler) as mock_pickle_loads: # Ensure pickle.loads returns the scaler

        # Simulate the context manager for NamedTemporaryFile
        mock_tempfile_constructor.return_value.__enter__.return_value = mock_temp_file_instance
        
        retrieved_model, retrieved_scaler, metadata = db_manager.get_model_and_scaler(symbol)

        # Verify tempfile was used to write model data
        mock_tempfile_constructor.assert_called_once_with(delete=False, suffix=".keras")
        mock_file_open.assert_called_once_with(mock_temp_file_instance.name, 'wb')
        mock_file_open().write.assert_called_once_with(model_data_bytes)
        
        # Verify load_model was called with the temp file path
        mock_tf_load_model.assert_called_once_with(mock_temp_file_instance.name)
        assert retrieved_model is loaded_mock_model
        
        # Verify scaler was unpickled
        mock_pickle_loads.assert_called_once_with(scaler_data_bytes)
        assert retrieved_scaler is original_mock_scaler # Check if the unpickled object is what we expect
        
        # Verify metadata
        assert metadata["info"] == "test_model"
        
        # Verify temp file was removed
        mock_os_remove.assert_called_once_with(mock_temp_file_instance.name)

def test_get_model_and_scaler_not_found(db_manager):
    model, scaler, metadata = db_manager.get_model_and_scaler("NONEXISTENT")
    assert model is None
    assert scaler is None
    assert metadata is None

def test_get_existing_model_symbols(db_manager):
    db_manager.db.models.insert_many([
        {"symbol": "AAPL", "model_data": b"data", "scaler_data": b"data"},
        {"symbol": "GOOG", "model_data": b"data", "scaler_data": b"data"}
    ])
    symbols = db_manager.get_existing_model_symbols()
    assert len(symbols) == 2
    assert "AAPL" in symbols
    assert "GOOG" in symbols

def test_get_model_metadata(db_manager):
    db_manager.db.models.insert_one({
        "symbol": "TESTMETA", 
        "metadata": {"version": "1.0", "accuracy": 0.99}
    })
    metadata = db_manager.get_model_metadata("TESTMETA")
    assert metadata is not None
    assert metadata["version"] == "1.0"
    assert metadata["accuracy"] == 0.99

    assert db_manager.get_model_metadata("NOMETA") is None


# --- Test Data Insertion/Update (Example for one method) ---
def test_add_stock_data(db_manager):
    sample_data = {"symbol": "NEWCO", "date": "2023-10-26", "close": 100}
    db_manager.add_stock_data(sample_data)
    
    retrieved = db_manager.db.stock_data.find_one({"symbol": "NEWCO"})
    assert retrieved is not None
    assert retrieved["close"] == 100

    # Test update
    updated_data = {"symbol": "NEWCO", "date": "2023-10-26", "close": 105}
    db_manager.add_stock_data(updated_data)
    
    retrieved_updated = db_manager.db.stock_data.find_one({"symbol": "NEWCO"})
    count = db_manager.db.stock_data.count_documents({"symbol": "NEWCO"})
    assert count == 1 # Should update, not insert new
    assert retrieved_updated["close"] == 105


def test_add_company_details(db_manager):
    company_data = {"symbol": "COMP", "name": "Comp Inc", "sector": "Tech"}
    db_manager.add_company_details(company_data)

    retrieved = db_manager.db.companies.find_one({"symbol": "COMP"})
    assert retrieved is not None
    assert retrieved["name"] == "Comp Inc"

    # Test update
    updated_company_data = {"symbol": "COMP", "name": "Comp Inc", "sector": "Technology Updated"}
    db_manager.add_company_details(updated_company_data)
    count = db_manager.db.companies.count_documents({"symbol": "COMP"})
    assert count == 1
    retrieved_updated = db_manager.db.companies.find_one({"symbol": "COMP"})
    assert retrieved_updated["sector"] == "Technology Updated"

# Test for ensure_indexes - more of an operational check
def test_ensure_indexes(db_manager, mock_mongo_client):
    # This test is a bit more involved as it checks for index creation commands.
    # mongomock might not fully simulate index creation details like real MongoDB.
    # We can check if the methods are called on the collection objects.
    
    # Spy on create_index calls
    db_manager.db.stock_data.create_index = MagicMock()
    db_manager.db.companies.create_index = MagicMock()
    db_manager.db.models.create_index = MagicMock()
    db_manager.db.cache.create_index = MagicMock()

    db_manager.ensure_indexes()

    db_manager.db.stock_data.create_index.assert_any_call([("symbol", 1), ("date", -1)], unique=True)
    db_manager.db.companies.create_index.assert_any_call([("symbol", 1)], unique=True)
    db_manager.db.models.create_index.assert_any_call([("symbol", 1)], unique=True)
    db_manager.db.cache.create_index.assert_any_call([("name", 1)], unique=True)
    # Check TTL index for cache (this is harder to assert precisely with mongomock's current capabilities)
    # For now, we'll trust the create_index call for 'expireAt' was made.
    # A more robust test would involve querying mock_mongo_client.index_information() or similar
    # if mongomock provided detailed index info.
    cache_create_index_calls = db_manager.db.cache.create_index.call_args_list
    assert any(
        call[0][0] == [("expireAt", 1)] and call[1].get("expireAfterSeconds") == 0
        for call in cache_create_index_calls
    ), "TTL index on 'expireAt' for cache collection not created as expected."


# Placeholder for GridFS-related tests if used more directly or if mongomock had full GridFS mock
# Currently, model data is stored directly in collection for simplicity with mongomock.
# If DatabaseManager was using GridFS explicitly (e.g. fs = gridfs.GridFS(self.db)),
# then fs.put, fs.get, fs.exists would need to be mocked/tested.
# The current implementation abstracts this by reading the file and putting bytes in the collection.

# Example of testing a method that might interact with predictions
def test_save_prediction(db_manager):
    prediction_data = {"symbol": "PRED", "date": "2023-11-01", "predicted_price": 150}
    db_manager.save_prediction("PRED", prediction_data) # Assuming such a method
    
    saved = db_manager.db.predictions.find_one({"symbol": "PRED"})
    assert saved is not None
    assert saved["predicted_price"] == 150

def test_get_predictions(db_manager):
    db_manager.db.predictions.insert_one({"symbol": "PRED", "date": "2023-11-01", "predicted_price": 150})
    preds = db_manager.get_predictions("PRED") # Assuming such a method
    assert len(preds) == 1
    assert preds[0]["predicted_price"] == 150

# Final check for get_stock_statistics with no data for a symbol
def test_get_stock_statistics_no_data_for_symbol(db_manager):
    # Ensure other data exists to avoid confusion with empty collection
    db_manager.db.stock_data.insert_one({"symbol": "OTHER", "close": 100, "volume": 1000})
    stats = db_manager.get_stock_statistics("NOSUCHSYMBOL")
    assert stats is None

# Test for get_stock_history with limit
def test_get_stock_history_with_limit(db_manager):
    base_date = datetime(2023, 1, 1)
    for i in range(50): # Insert 50 records
        db_manager.db.stock_data.insert_one({
            "symbol": "AAPL", 
            "date": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"), 
            "close": 100 + i
        })
    
    history = db_manager.get_stock_history("AAPL", 20) # Limit to 20 days
    assert len(history) == 20
    # Ensure they are the latest 20
    assert history[0]["close"] == 100 + (50 - 20) # Oldest of the latest 20
    assert history[19]["close"] == 100 + 49      # Latest overall
    
    history_default_limit = db_manager.get_stock_history("AAPL") # Default limit is 90
    assert len(history_default_limit) == 50 # Since we only have 50 records

# Test for get_all_company_details sorting
def test_get_all_company_details_sorting(db_manager):
    db_manager.db.companies.insert_many([
        {"symbol": "MSFT", "name": "Microsoft"},
        {"symbol": "AAPL", "name": "Apple Inc"},
        {"symbol": "GOOG", "name": "Alphabet Inc"}
    ])
    companies = db_manager.get_all_company_details()
    assert len(companies) == 3
    assert companies[0]["symbol"] == "AAPL"
    assert companies[1]["symbol"] == "GOOG"
    assert companies[2]["symbol"] == "MSFT"

# Test for model/scaler saving when tempfile or os operations fail
def test_save_model_and_scaler_tempfile_error(db_manager, mock_model_scaler):
    mock_model, mock_scaler = mock_model_scaler
    symbol = "FAILSYM"

    with patch('tempfile.NamedTemporaryFile', side_effect=IOError("Disk full")) as mock_tempfile:
        with pytest.raises(IOError, match="Disk full"):
            db_manager.save_model_and_scaler(symbol, mock_model, mock_scaler, {})
        mock_tempfile.assert_called_once() # Ensure it was attempted
        assert db_manager.db.models.count_documents({"symbol": symbol}) == 0 # No data saved

def test_save_model_and_scaler_model_save_error(db_manager, mock_model_scaler):
    mock_model, mock_scaler = mock_model_scaler
    symbol = "FAILSYM"
    mock_model.save.side_effect = Exception("Keras save failed") # Simulate error during model.save()

    mock_temp_file_instance = MagicMock()
    mock_temp_file_instance.name = "dummy_temp_path.keras"

    with patch('tempfile.NamedTemporaryFile', return_value=mock_temp_file_instance) as mock_tempfile_constructor, \
         patch('os.remove') as mock_os_remove:
        mock_tempfile_constructor.return_value.__enter__.return_value = mock_temp_file_instance
        
        with pytest.raises(Exception, match="Keras save failed"):
            db_manager.save_model_and_scaler(symbol, mock_model, mock_scaler, {})
        
        mock_model.save.assert_called_once()
        # os.remove should still be called if the temp file was created by 'with ... as ...:'
        # but the actual file content might not have been written or might be corrupted.
        # The important part is that the database operation should not commit inconsistent data.
        mock_os_remove.assert_called_once_with(mock_temp_file_instance.name)
        assert db_manager.db.models.count_documents({"symbol": symbol}) == 0


def test_get_model_and_scaler_load_model_error(db_manager):
    symbol = "LOADFAILS SYM"
    db_manager.db.models.insert_one({
        "symbol": symbol, "model_data": b"fake_model", "scaler_data": pickle.dumps(MagicMock())
    })

    mock_temp_file_instance = MagicMock()
    mock_temp_file_instance.name = "retrieved_temp_path.keras"

    with patch('tensorflow.keras.models.load_model', side_effect=IOError("Cannot load model")) as mock_load, \
         patch('tempfile.NamedTemporaryFile', return_value=mock_temp_file_instance) as mock_tempfile_constructor, \
         patch('os.remove') as mock_os_remove, \
         patch('builtins.open', mock_open()): # Mock open for writing to temp file
        mock_tempfile_constructor.return_value.__enter__.return_value = mock_temp_file_instance

        with pytest.raises(IOError, match="Cannot load model"):
            db_manager.get_model_and_scaler(symbol)
        
        mock_load.assert_called_once()
        mock_os_remove.assert_called_once_with(mock_temp_file_instance.name) # Temp file should still be cleaned up

def test_get_model_and_scaler_pickle_error(db_manager):
    symbol = "PICKLEFAILS SYM"
    db_manager.db.models.insert_one({
        "symbol": symbol, "model_data": b"fake_model", "scaler_data": b"corrupted_pickle_data"
    })
    
    # Mock load_model to return a dummy model to allow execution to reach pickle.loads
    loaded_mock_model = MagicMock(spec=MockKerasModel)
    mock_temp_file_instance = MagicMock()
    mock_temp_file_instance.name = "retrieved_temp_path.keras"

    with patch('tensorflow.keras.models.load_model', return_value=loaded_mock_model), \
         patch('pickle.loads', side_effect=pickle.UnpicklingError("Invalid pickle data")), \
         patch('tempfile.NamedTemporaryFile', return_value=mock_temp_file_instance) as mock_tempfile_constructor, \
         patch('os.remove') as mock_os_remove, \
         patch('builtins.open', mock_open()):
        mock_tempfile_constructor.return_value.__enter__.return_value = mock_temp_file_instance

        with pytest.raises(pickle.UnpicklingError, match="Invalid pickle data"):
            db_manager.get_model_and_scaler(symbol)
        
        mock_os_remove.assert_called_once_with(mock_temp_file_instance.name)
