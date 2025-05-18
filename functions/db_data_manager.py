from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime
from typing import List, Dict, Optional, Union
from functions.logger import logger
import pickle



# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

class DatabaseManager:
    def __init__(self):
        """Initialize database connection"""
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client.admin
        self.collection = self.db.scraped_data
        self.companies_collection = self.db.companies
    
    def get_all_company_details(self) -> List[Dict]:
        """Get all company symbols and names from the 'companies' collection"""
        try:
            # Fetch only symbol and name, sort by name
            # Ensure this matches the structure of your 'companies' collection
            companies_cursor = self.companies_collection.find(
                {},
                {'symbol': 1, 'name': 1, '_id': 0}
            ).sort('name', 1)
            return list(companies_cursor)
        except Exception as e:
            logger.error(f"Error fetching company details from MongoDB: {e}")
            return [] # Return empty list on error

    def get_all_stocks(self) -> List[Dict]:
        """Get all stock data from the collection"""
        try:
            return list(self.collection.find())
        except Exception as e:
            logger.error(f"Error fetching stocks: {e}")
            return []

    def get_stock_by_symbol(self, symbol: str) -> List[Dict]:
        """Get stock data for a specific symbol"""
        try:
            return list(self.collection.find({"symbol": symbol}))
        except Exception as e:
            logger.error(f"Error fetching stocks: {e}")
            return []

    def get_stock_by_date_range(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """Get stock data for a specific symbol within a date range"""
        try:
            query = {
                "symbol": symbol,
                "transaction_date": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            return list(self.collection.find(query))
        except Exception as e:
            logger.error(f"Error fetching stock data for date range: {e}")
            return []

    def get_unique_symbols(self) -> List[str]:
        """Get list of unique stock symbols"""
        try:
            return self.collection.distinct("symbol")
        except Exception as e:
            logger.error(f"Error fetching unique symbols: {e}")
            return []

    def get_latest_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get the most recent stock data for a symbol"""
        try:
            return self.collection.find_one(
                {"symbol": symbol},
                sort=[("transaction_date", -1)]
            )
        except Exception as e:
            logger.error(f"Error fetching latest stock data: {e}")
            return None

    def get_stock_statistics(self, symbol: str) -> Dict[str, Union[float, int]]:
        """Get statistical data for a stock symbol"""
        try:
            pipeline = [
                {"$match": {"symbol": symbol}},
                {
                    "$group": {
                        "_id": "$symbol",
                        "avg_price": {"$avg": "$rate"},
                        "max_price": {"$max": "$rate"},
                        "min_price": {"$min": "$rate"},
                        "total_volume": {"$sum": "$quantity"}
                    }
                }
            ]
            result = list(self.collection.aggregate(pipeline))
            return result[0] if result else {}
        except Exception as e:
            logger.error(f"Error calculating stock statistics: {e}")
            return {}

    def get_stock_history(self, symbol: str) -> List[Dict]:
        """Get historical stock data for a specific symbol with average price"""
        try:
            pipeline = [
                {"$match": {"symbol": symbol}},
                {
                    "$group": {
                        "_id": {
                            "symbol": "$symbol",
                            "transaction_date": "$transaction_date"
                        },
                        "rate": {"$last": "$rate"},
                        "quantity": {"$sum": "$quantity"},
                        "amount": {"$sum": "$amount"},
                        "avg_price": {"$avg": "$rate"}
                    }
                },
                {"$sort": {"_id.transaction_date": 1}},
                {
                    "$project": {
                        "_id": 0,
                        "symbol": "$_id.symbol",
                        "transaction_date": "$_id.transaction_date",
                        "rate": 1,
                        "quantity": 1,
                        "amount": 1,
                        "avg_price": 1
                    }
                }
            ]
            return list(self.collection.aggregate(pipeline))
        except Exception as e:
            logger.error(f"Error fetching stock history: {e}")
            return []

    def close_connection(self):
        """Close the MongoDB connection"""
        try:
            self.client.close()
        except Exception as e:
            logger.error(f"Error closing connection: {e}")

    def cache_top_stocks(self, top_stocks: list) -> bool:
        """Cache top performing stocks in a separate collection"""
        try:
            # Get the cache collection
            cache_collection = self.db.top_stocks_cache
            
            # Clear existing cache
            cache_collection.delete_many({})
            
            # Insert new data
            if top_stocks:
                cache_collection.insert_many(top_stocks)
            return True
        except Exception as e:
            logger.error(f"Error caching top stocks: {e}", exc_info=True)
            return False
    
    def get_cached_stocks(self) -> list:
        """Get cached top performing stocks"""
        try:
            cache_collection = self.db.top_stocks_cache
            return list(cache_collection.find())
        except Exception as e:
            logger.error(f"Error fetching cached stocks: {e}", exc_info=True)
            return []

    def save_model_and_scaler(self, symbol, model_data, scaler_data):
        """
        Save trained model and scaler for a symbol in the database
        """
        try:
            # Convert model and scaler to binary format
            model_binary = pickle.dumps(model_data)
            scaler_binary = pickle.dumps(scaler_data)
            
            # Prepare document
            model_doc = {
                'symbol': symbol,
                'model': model_binary,
                'scaler': scaler_binary,
                'updated_at': datetime.now()
            }
            
            # Update or insert the model
            self.db['models'].update_one(
                {'symbol': symbol},
                {'$set': model_doc},
                upsert=True
            )
            
            logger.info(f"Successfully saved model and scaler for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model for {symbol}: {str(e)}")
            return False

    def get_model_and_scaler(self, symbol):
        """
        Retrieve trained model and scaler for a symbol from the database
        """
        try:
            model_doc = self.db['models'].find_one({'symbol': symbol})
            if model_doc:
                model = pickle.loads(model_doc['model'])
                scaler = pickle.loads(model_doc['scaler'])
                return model, scaler
            return None, None
            
        except Exception as e:
            logger.error(f"Error retrieving model for {symbol}: {str(e)}")
            return None, None

    def get_existing_model_symbols(self):
        """Get all symbols that already have trained models"""
        try:
            model_collection = self.db['models']
            # Only fetch the symbol field for efficiency
            existing_models = model_collection.distinct('symbol')
            return existing_models
        except Exception as e:
            logger.error(f"Error fetching existing model symbols: {str(e)}")
            return []

# Usage example:
"""
db_manager = DatabaseManager()

# Get all stocks
stocks = db_manager.get_all_stocks()

# Get specific stock data
apple_stocks = db_manager.get_stock_by_symbol("AHPC")

# Get date range data
range_data = db_manager.get_stock_by_date_range("AHPC", "2014-01-01", "2014-12-31")

# Get unique symbols
symbols = db_manager.get_unique_symbols()

# Get latest stock data
latest = db_manager.get_latest_stock_data("AHPC")

# Get statistics
stats = db_manager.get_stock_statistics("AHPC")

# Close connection when done
db_manager.close_connection()
"""