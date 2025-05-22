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
        current_rate = latest_data['rate']
        avg_price = stats.get('avg_price', current_rate)
        
        top_stocks.append({
            'symbol': symbol,
            'rate': latest_data['rate'],
            'quantity': latest_data['quantity'],
            'amount': latest_data['amount'],
            'change': _calculate_percentage_change(current_rate, avg_price)
        })
    
    # Sort by change percentage (descending) and return top 10
    return sorted(top_stocks, key=lambda x: x['change'], reverse=True)[:10]

def _prepare_prediction_data(
    historical_data: List[Dict], 
    predictions: List[float], 
    num_days: int
) -> List[Dict]:
    """Prepare combined historical and prediction data for display.
    
    Args:
        historical_data: List of historical stock records
        predictions: List of predicted prices
        num_days: Number of prediction days
        
    Returns:
        Combined list of historical and prediction data
    """
    last_date = pd.to_datetime(historical_data[-1]['transaction_date'])
    future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                   for i in range(num_days)]
    
    # Apply scaling factor to predictions
    last_historical_price = historical_data[-1]['rate']
    if predictions.size > 0:
        scaling_factor = last_historical_price / predictions[0][0]
        predictions = predictions * scaling_factor
    
    # Create prediction records
    prediction_data = [{
        'transaction_date': date,
        'rate': float(price),
        'is_prediction': True
    } for date, price in zip(future_dates, predictions.flatten())]
    
    # Mark historical data
    for data in historical_data:
        data['is_prediction'] = False
    
    # Combine last 30 days of history with predictions
    return historical_data[-30:] + prediction_data
