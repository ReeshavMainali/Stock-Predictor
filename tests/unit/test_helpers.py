import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Assuming DatabaseManager is in functions.db_data_manager
# Adjust the import path if DatabaseManager is located elsewhere
from functions.db_data_manager import DatabaseManager 
from functions.helpers import (
    _calculate_percentage_change,
    _prepare_top_stocks_data,
    _prepare_prediction_data,
)

# Tests for _calculate_percentage_change
def test_calculate_percentage_change_positive():
    assert _calculate_percentage_change(110, 100) == 10.0

def test_calculate_percentage_change_negative():
    assert _calculate_percentage_change(90, 100) == -10.0

def test_calculate_percentage_change_no_change():
    assert _calculate_percentage_change(100, 100) == 0.0

def test_calculate_percentage_change_average_zero():
    assert _calculate_percentage_change(10, 0) == 0.0

def test_calculate_percentage_change_current_zero():
    assert _calculate_percentage_change(0, 100) == -100.0

def test_calculate_percentage_change_both_zero():
    assert _calculate_percentage_change(0, 0) == 0.0


# Tests for _prepare_top_stocks_data
def test_prepare_top_stocks_data_no_symbols():
    mock_db_manager = MagicMock(spec=DatabaseManager)
    mock_db_manager.get_unique_symbols.return_value = []
    
    result = _prepare_top_stocks_data(mock_db_manager)
    assert result == []
    mock_db_manager.get_unique_symbols.assert_called_once()

def test_prepare_top_stocks_data_no_latest_data():
    mock_db_manager = MagicMock(spec=DatabaseManager)
    mock_db_manager.get_unique_symbols.return_value = ["AAPL", "GOOG"]
    mock_db_manager.get_latest_stock_data.return_value = None # Simulate no data
    
    result = _prepare_top_stocks_data(mock_db_manager)
    assert result == []
    mock_db_manager.get_unique_symbols.assert_called_once()
    # assert mock_db_manager.get_latest_stock_data.call_count == 2 # Called for each symbol

def test_prepare_top_stocks_data_success_and_sorting():
    mock_db_manager = MagicMock(spec=DatabaseManager)
    mock_db_manager.get_unique_symbols.return_value = ["AAPL", "GOOG", "MSFT"]
    
    # Simulate data for AAPL
    mock_db_manager.get_latest_stock_data.side_effect = lambda symbol: {
        "AAPL": {"close": 150.0, "symbol": "AAPL", "date": "2023-10-01"},
        "GOOG": {"close": 2500.0, "symbol": "GOOG", "date": "2023-10-01"},
        "MSFT": {"close": 300.0, "symbol": "MSFT", "date": "2023-10-01"},
    }[symbol]

    mock_db_manager.get_stock_statistics.side_effect = lambda symbol: {
        "AAPL": {"avg_close": 140.0},
        "GOOG": {"avg_close": 2600.0}, # Negative change
        "MSFT": {"avg_close": 290.0}, # Positive change, less than AAPL
    }[symbol]

    result = _prepare_top_stocks_data(mock_db_manager)
    
    assert len(result) == 3
    # Expected order: AAPL (7.14%), MSFT (3.45%), GOOG (-3.85%)
    assert result[0]["symbol"] == "AAPL" 
    assert result[1]["symbol"] == "MSFT"
    assert result[2]["symbol"] == "GOOG"
    
    assert result[0]["change"] == pytest.approx(( (150-140)/140 )*100)
    assert result[1]["change"] == pytest.approx(( (300-290)/290 )*100)
    assert result[2]["change"] == pytest.approx(( (2500-2600)/2600 )*100)

def test_prepare_top_stocks_data_max_10_stocks():
    mock_db_manager = MagicMock(spec=DatabaseManager)
    symbols = [f"SYM{i}" for i in range(15)]
    mock_db_manager.get_unique_symbols.return_value = symbols
    
    mock_db_manager.get_latest_stock_data.side_effect = lambda symbol: \
        {"close": 100.0 + symbols.index(symbol), "symbol": symbol, "date": "2023-10-01"}

    mock_db_manager.get_stock_statistics.side_effect = lambda symbol: \
        {"avg_close": 90.0 + symbols.index(symbol)} # All will have positive change

    result = _prepare_top_stocks_data(mock_db_manager)
    assert len(result) == 10 # Should be capped at 10
    # Check if they are sorted by change (which also means by symbol index here due to how data is mocked)
    # Highest change will be SYM14, then SYM13, ...
    for i in range(10):
        expected_symbol_index = 14 - i
        assert result[i]["symbol"] == f"SYM{expected_symbol_index}"


# Tests for _prepare_prediction_data
def test_prepare_prediction_data_empty_inputs():
    result = _prepare_prediction_data([], [])
    assert result["dates"] == []
    assert result["prices"] == []
    assert result["is_prediction"] == []
    assert result["name"] == ""
    assert result["current_price"] is None
    assert result["price_change"] is None
    assert result["percentage_change"] is None
    assert result["min_price"] == 0
    assert result["max_price"] == 0

def test_prepare_prediction_data_with_history_no_predictions():
    historical_data = [
        {"date": "2023-01-01", "close": 100, "symbol": "AAPL"},
        {"date": "2023-01-02", "close": 102, "symbol": "AAPL"},
    ]
    result = _prepare_prediction_data(historical_data, [])
    
    assert len(result["dates"]) == 2
    assert len(result["prices"]) == 2
    assert len(result["is_prediction"]) == 2
    
    assert result["dates"] == ["01/01", "01/02"]
    assert result["prices"] == [100, 102]
    assert result["is_prediction"] == [False, False]
    assert result["name"] == "AAPL"
    assert result["current_price"] == 102
    assert result["price_change"] == 2 # 102 - 100
    assert result["percentage_change"] == pytest.approx(2.0) # (2/100)*100
    assert result["min_price"] == 100
    assert result["max_price"] == 102


def test_prepare_prediction_data_with_history_and_predictions():
    # Prepare 35 days of historical data to test the 30-day slicing
    base_date = datetime(2023, 1, 1)
    historical_data = []
    for i in range(35):
        historical_data.append({
            "date": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
            "close": 100 + i,
            "symbol": "TEST"
        })
    
    # Last historical price is 100 + 34 = 134
    # Predictions should start from the day after the last historical date
    predictions = [135, 136, 137, 138, 139, 140, 141] # 7 predictions
    
    result = _prepare_prediction_data(historical_data, predictions)
    
    # Expected: 30 days of history + 7 days of prediction = 37 data points
    assert len(result["dates"]) == 37
    assert len(result["prices"]) == 37
    assert len(result["is_prediction"]) == 37
    
    assert result["name"] == "TEST"
    assert result["current_price"] == 134 # Last actual price
    
    # Price change and percentage change are based on the 30-day window of historical data
    # Oldest price in the 30-day window: historical_data[5]['close'] = 100 + 5 = 105
    # Newest price (current_price): 134
    expected_price_change = 134 - 105 # 29
    expected_percentage_change = (expected_price_change / 105) * 100
    
    assert result["price_change"] == pytest.approx(expected_price_change)
    assert result["percentage_change"] == pytest.approx(expected_percentage_change)

    # Check is_prediction flags
    assert result["is_prediction"][:30] == [False] * 30
    assert result["is_prediction"][30:] == [True] * 7

    # Check dates format and continuity
    last_hist_date_obj = datetime.strptime(historical_data[-1]["date"], "%Y-%m-%d")
    for i in range(7):
        expected_pred_date = (last_hist_date_obj + timedelta(days=i + 1)).strftime("%m/%d")
        assert result["dates"][30+i] == expected_pred_date
    
    # Check prices
    assert result["prices"][:30] == [h["close"] for h in historical_data[-30:]]
    assert result["prices"][30:] == predictions

    # Check scaling factor logic (min/max)
    all_prices = [h["close"] for h in historical_data[-30:]] + predictions
    assert result["min_price"] == min(all_prices)
    assert result["max_price"] == max(all_prices)


def test_prepare_prediction_data_only_predictions_no_history():
    predictions = [200, 205, 202]
    # If no historical data, symbol name is unknown, current price etc. are None or 0
    result = _prepare_prediction_data([], predictions)
    
    assert len(result["dates"]) == 3 # 3 predictions
    assert len(result["prices"]) == 3
    assert len(result["is_prediction"]) == 3

    # Dates should be "Pred 1", "Pred 2", ...
    assert result["dates"] == ["Pred 1", "Pred 2", "Pred 3"]
    assert result["prices"] == predictions
    assert result["is_prediction"] == [True, True, True]
    
    assert result["name"] == "" # No symbol from history
    assert result["current_price"] is None
    assert result["price_change"] is None
    assert result["percentage_change"] is None
    
    assert result["min_price"] == min(predictions)
    assert result["max_price"] == max(predictions)


def test_prepare_prediction_data_scaling_factor_logic():
    # Test with a small range to make scaling factor more obvious if it were fixed
    historical_data = [
        {"date": "2023-01-01", "close": 1, "symbol": "TINY"},
        {"date": "2023-01-02", "close": 2, "symbol": "TINY"},
    ]
    predictions = [3, 1, 4]
    
    result = _prepare_prediction_data(historical_data, predictions)
    
    all_expected_prices = [1, 2, 3, 1, 4]
    assert result["min_price"] == 1
    assert result["max_price"] == 4
    # This implicitly tests that the y-axis of a graph would scale correctly.


def test_prepare_prediction_data_history_less_than_30_days():
    historical_data = [
        {"date": "2023-01-01", "close": 50, "symbol": "LESS"},
        {"date": "2023-01-02", "close": 55, "symbol": "LESS"},
    ] # 2 days of history
    predictions = [60, 65]
    
    result = _prepare_prediction_data(historical_data, predictions)
    
    assert len(result["dates"]) == 4 # 2 history + 2 prediction
    assert result["prices"] == [50, 55, 60, 65]
    assert result["is_prediction"] == [False, False, True, True]
    
    assert result["name"] == "LESS"
    assert result["current_price"] == 55
    assert result["price_change"] == 5 # 55 - 50
    assert result["percentage_change"] == pytest.approx(10.0) # (5/50)*100
    
    assert result["min_price"] == 50
    assert result["max_price"] == 65

    # Verify correct date formatting for history
    assert result["dates"][0] == "01/01"
    assert result["dates"][1] == "01/02"
    # Verify correct date "Pred X" for predictions when history exists
    # Based on the current implementation, it uses the day after the last historical date
    last_hist_date_obj = datetime.strptime(historical_data[-1]["date"], "%Y-%m-%d")
    expected_pred_date_1 = (last_hist_date_obj + timedelta(days=1)).strftime("%m/%d")
    expected_pred_date_2 = (last_hist_date_obj + timedelta(days=2)).strftime("%m/%d")
    assert result["dates"][2] == expected_pred_date_1
    assert result["dates"][3] == expected_pred_date_2

# It's good practice to also test the patching for _prepare_top_stocks_data
# to ensure the mock is used as expected, especially if DatabaseManager was imported directly.
@patch('functions.helpers.DatabaseManager') # Patch where it's USED
def test_prepare_top_stocks_data_with_explicit_patch(MockDbManager):
    # This test is more about verifying the patching mechanism if you had trouble
    # with direct MagicMock(spec=...) or if DatabaseManager is complex to instantiate.
    
    mock_instance = MockDbManager.return_value # Get the instance created by helpers
    mock_instance.get_unique_symbols.return_value = ["TESTSYM"]
    mock_instance.get_latest_stock_data.return_value = {"close": 110, "symbol": "TESTSYM", "date": "2023-01-01"}
    mock_instance.get_stock_statistics.return_value = {"avg_close": 100}
    
    result = _prepare_top_stocks_data(mock_instance) # Pass the already mocked instance
                                                    # or if helpers.py instantiates it, don't pass it.
                                                    # The current _prepare_top_stocks_data expects an instance.

    assert len(result) == 1
    assert result[0]["symbol"] == "TESTSYM"
    assert result[0]["change"] == 10.0
    
    mock_instance.get_unique_symbols.assert_called_once()
    mock_instance.get_latest_stock_data.assert_called_with("TESTSYM")
    mock_instance.get_stock_statistics.assert_called_with("TESTSYM")

# If _prepare_top_stocks_data instantiates DatabaseManager itself, the patch would look like this:
# @patch('functions.helpers.DatabaseManager') 
# def test_prepare_top_stocks_data_internal_instantiation(MockDbManagerConstructor):
#     # mock_constructor_instance = MockDbManagerConstructor.return_value
#     # mock_constructor_instance.get_unique_symbols.return_value = ...
#     # result = _prepare_top_stocks_data() # Don't pass db_manager
#     # ... asserts
# However, the current _prepare_top_stocks_data takes db_manager as an argument,
# so the previous tests with MagicMock(spec=DatabaseManager) are more direct.
# The test_prepare_top_stocks_data_with_explicit_patch is a bit redundant
# with the MagicMock(spec=...) approach if DatabaseManager is correctly imported for spec.
# I'll keep it for now as an example of patching.
# For DatabaseManager to be correctly used in spec, it needs to be importable.
# Let's assume functions.db_data_manager.DatabaseManager is the correct path.
# If not, the spec might not be as effective.

# One final check for _prepare_top_stocks_data: what if get_stock_statistics returns None?
def test_prepare_top_stocks_data_no_stats():
    mock_db_manager = MagicMock(spec=DatabaseManager)
    mock_db_manager.get_unique_symbols.return_value = ["NOSYMBOL"]
    mock_db_manager.get_latest_stock_data.return_value = {"close": 100.0, "symbol": "NOSYMBOL", "date": "2023-10-01"}
    mock_db_manager.get_stock_statistics.return_value = None # Simulate no stats data

    result = _prepare_top_stocks_data(mock_db_manager)
    assert result == [] # Expect empty if stats are missing, as change cannot be calculated
    mock_db_manager.get_unique_symbols.assert_called_once()
    mock_db_manager.get_latest_stock_data.assert_called_with("NOSYMBOL")
    mock_db_manager.get_stock_statistics.assert_called_with("NOSYMBOL")

def test_prepare_top_stocks_data_stats_missing_avg_close():
    mock_db_manager = MagicMock(spec=DatabaseManager)
    mock_db_manager.get_unique_symbols.return_value = ["NOSYMBOL"]
    mock_db_manager.get_latest_stock_data.return_value = {"close": 100.0, "symbol": "NOSYMBOL", "date": "2023-10-01"}
    # Simulate stats data missing the 'avg_close' key
    mock_db_manager.get_stock_statistics.return_value = {"some_other_stat": 50} 

    result = _prepare_top_stocks_data(mock_db_manager)
    # This should ideally be handled gracefully by _prepare_top_stocks_data
    # Current implementation might raise KeyError. Let's assume it should return empty or skip.
    # Based on current helper: `avg_close = stock_stats.get("avg_close")` then `if avg_close is None or avg_close == 0:`
    # So, if "avg_close" key is missing, avg_close will be None, and it will be skipped.
    assert result == [] 
    mock_db_manager.get_unique_symbols.assert_called_once()
    mock_db_manager.get_latest_stock_data.assert_called_with("NOSYMBOL")
    mock_db_manager.get_stock_statistics.assert_called_with("NOSYMBOL")

def test_prepare_top_stocks_data_latest_data_missing_close():
    mock_db_manager = MagicMock(spec=DatabaseManager)
    mock_db_manager.get_unique_symbols.return_value = ["NOSYMBOL"]
    # Simulate latest stock data missing the 'close' key
    mock_db_manager.get_latest_stock_data.return_value = {"symbol": "NOSYMBOL", "date": "2023-10-01"}
    mock_db_manager.get_stock_statistics.return_value = {"avg_close": 100.0}

    result = _prepare_top_stocks_data(mock_db_manager)
    # Based on current helper: `current_price = latest_data.get("close")` then `if current_price is None:`
    # It will be skipped.
    assert result == []
    mock_db_manager.get_unique_symbols.assert_called_once()
    mock_db_manager.get_latest_stock_data.assert_called_with("NOSYMBOL")
    # get_stock_statistics might not be called if latest_data processing fails early,
    # but in the current structure, it's called before the check for current_price.
    # Let's refine this based on actual helper logic:
    # latest_data = db_manager.get_latest_stock_data(symbol)
    # if not latest_data or latest_data.get("close") is None: continue
    # stock_stats = db_manager.get_stock_statistics(symbol)
    # if not stock_stats or stock_stats.get("avg_close") is None or stock_stats.get("avg_close") == 0: continue
    # So, if latest_data.get("close") is None, get_stock_statistics won't be called for that symbol.
    # This means the mock_db_manager.get_stock_statistics.assert_called_with("NOSYMBOL") would fail.
    # Let's adjust the expectation:
    # mock_db_manager.get_stock_statistics.assert_not_called() # Or check call_count if other symbols are processed

    # For a single symbol failing like this:
    assert mock_db_manager.get_stock_statistics.call_count == 0 # It shouldn't be called.
