from datetime import timedelta
from typing import List , Dict 
from .db_data_manager import DatabaseManager
import pandas as pd

# --------------------------
# Helper Functions
# --------------------------

def _calculate_percentage_change(current: float, average: float) -> float:
    """Calculate percentage change between current and average price.
    
    Args:
        current: Current stock price
        average: Average stock price
        
    Returns:
        Percentage change (rounded to 2 decimal places)
    """
    return round(((current - average) / average) * 100, 2) if average else 0

def _prepare_top_stocks_data(db_manager: DatabaseManager) -> List[Dict]:
    """Prepare top performing stocks data for dashboard.
    
    Args:
        db_manager: DatabaseManager instance
        
    Returns:
        List of top performing stocks sorted by percentage change
    """
    symbols = db_manager.get_unique_symbols()
    top_stocks = []
    
    for symbol in symbols:
        latest_data = db_manager.get_latest_stock_data(symbol)
        if not latest_data:
            continue
            
        stats = db_manager.get_stock_statistics(symbol)
        # Use 'close' for price, and .get for potentially missing keys
        current_close_price = latest_data.get('close') 
        if current_close_price is None: # Ensure price is available
            continue

        # Ensure stats and avg_close are present, otherwise default avg_price to current_close_price
        avg_price = current_close_price
        if stats and 'avg_close' in stats:
            avg_price = stats.get('avg_close', current_close_price)

        top_stocks.append({
            'symbol': symbol,
            'rate': current_close_price,  # Use 'close' here, which is now current_close_price
            'quantity': latest_data.get('quantity', 0), # Use .get for safety
            'amount': latest_data.get('amount', 0),     # Use .get for safety
            'change': _calculate_percentage_change(current_close_price, avg_price)
        })
    
    # Sort by change percentage (descending) and return top 10
    return sorted(top_stocks, key=lambda x: x['change'], reverse=True)[:10]

def _prepare_prediction_data(
    historical_data: List[Dict], 
    predictions: List[float], # This is expected to be a NumPy array or similar, supporting .size and .flatten
    num_days: int
) -> List[Dict]:
    """Prepare combined historical and prediction data for display.
    
    Args:
        historical_data: List of historical stock records (dictionaries)
        predictions: List/array of predicted prices (float values)
        num_days: Number of prediction days
        
    Returns:
        Combined list of historical and prediction data
    """
    if not historical_data: # Guard against empty historical data
        return []

    # Ensure historical_data has 'date' and 'close' keys.
    # The problem description refers to 'transaction_date' and 'rate' for historical data
    # in the context of this function, let's assume it should be 'date' and 'close'
    # to align with typical stock data structures and avoid KeyErrors.
    # If it must be 'transaction_date' and 'rate', this part needs to match that.
    # For now, assuming 'date' and 'close' are the keys in historical_data items.
    
    last_historical_record = historical_data[-1]
    last_date_str = last_historical_record.get('date') # Use .get for safety
    last_price = last_historical_record.get('close')   # Use .get for safety

    if last_date_str is None or last_price is None:
        # Cannot proceed without a valid last date or price from historical data
        # Or handle this case by returning only historical or raising an error
        return historical_data # Or some other error handling

    last_date = pd.to_datetime(last_date_str)
    future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                   for i in range(num_days)]
    
    # Apply scaling factor to predictions if predictions are available
    # The original code had `predictions.size > 0` which implies numpy array.
    # Let's make it more robust for lists too.
    if predictions is not None and len(predictions) > 0 :
        # Assuming predictions is a list of lists or a 2D numpy array like [[val1], [val2]]
        # And that the first element of the first prediction is the one to use for scaling.
        first_predicted_value = predictions[0]
        if isinstance(first_predicted_value, (list, tuple)) and len(first_predicted_value) > 0:
            first_predicted_value = first_predicted_value[0]
        
        if first_predicted_value != 0: # Avoid division by zero
            scaling_factor = last_price / first_predicted_value
            # Apply scaling: if predictions is numpy array, this works. If list of lists, needs a loop.
            if hasattr(predictions, 'flatten'): # Check if it's a NumPy array
                 scaled_predictions = predictions * scaling_factor
                 flat_scaled_predictions = scaled_predictions.flatten()
            else: # Assuming list of simple floats or list of single-element lists
                scaled_predictions = []
                for pred_val_item in predictions:
                    if isinstance(pred_val_item, (list, tuple)):
                        scaled_predictions.append(pred_val_item[0] * scaling_factor)
                    else: # simple float
                        scaled_predictions.append(pred_val_item * scaling_factor)
                flat_scaled_predictions = scaled_predictions
        else: # first predicted value is 0, cannot scale. Use raw predictions.
             if hasattr(predictions, 'flatten'):
                 flat_scaled_predictions = predictions.flatten()
             else:
                flat_scaled_predictions = [p[0] if isinstance(p, (list,tuple)) else p for p in predictions]

    else: # No predictions provided
        flat_scaled_predictions = []

    # Create prediction records
    prediction_data = [{
        'date': date, # Align key name with historical data ('date')
        'close': float(price), # Align key name ('close')
        'is_prediction': True
    } for date, price in zip(future_dates, flat_scaled_predictions)]
    
    # Mark historical data (ensure 'is_prediction' is set)
    # Create new list of dicts for historical data to avoid modifying original list of dicts
    processed_historical_data = []
    for record in historical_data[-30:]: # Take last 30 days
        new_record = record.copy() # Avoid modifying original data items
        new_record['is_prediction'] = False
        # Ensure keys are consistent: 'date' and 'close'
        if 'transaction_date' in new_record and 'date' not in new_record:
            new_record['date'] = new_record.pop('transaction_date')
        if 'rate' in new_record and 'close' not in new_record:
            new_record['close'] = new_record.pop('rate')
        processed_historical_data.append(new_record)
    
    return processed_historical_data + prediction_data
