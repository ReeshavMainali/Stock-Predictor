"""
Database interaction module for stock data and machine learning models.

This module provides a DatabaseManager class that handles all interactions with MongoDB,
including storing/retrieving stock data, company information, and trained ML models.
"""

from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime
from typing import List, Dict, Optional, Union
from functions.logger import logger
import pickle
import tensorflow as tf  # For model serialization/deserialization
import tempfile          # For temporary model file handling


# Load environment variables once at module level
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "admin"


class DatabaseManager:
    """MongoDB manager for stock data and machine learning models.
    
    Handles all database operations including:
    - Stock data retrieval (single/multiple symbols, date ranges)
    - Company information
    - Machine learning model storage/retrieval
    - Caching operations
    
    Attributes:
        client: MongoClient instance
        db: Reference to admin database
        collection: Reference to scraped_data collection
        companies_collection: Reference to companies collection
        models_collection: Reference to models collection
    """
    
    def __init__(self):
        """Initialize MongoDB connection and collection references."""
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client.admin
        self.collection = self.db.scraped_data
        self.companies_collection = self.db.companies
        self.models_collection = self.db.models

    # --------------------------
    # Company Data Methods
    # --------------------------

    def get_all_company_details(self) -> List[Dict]:
        """Get details (symbol and name) for companies with trained models.
        
        Returns:
            List of company dictionaries sorted alphabetically by name, 
            each containing 'symbol' and 'name' keys.
            Empty list if no companies with models found or on error.
        """
        try:
            symbols_with_models = self.get_existing_model_symbols()
            if not symbols_with_models:
                logger.info("No trained models found, returning empty company list.")
                return []

            query = {'symbol': {'$in': symbols_with_models}}
            projection = {'symbol': 1, 'name': 1, '_id': 0}
            
            return list(
                self.companies_collection.find(query, projection)
                .sort('name', 1)  # Sort by name alphabetically
            )
        except Exception as e:
            logger.error(f"Error fetching company details with models: {e}")
            return []

    # --------------------------
    # Stock Data Methods
    # --------------------------

    def get_all_stocks(self) -> List[Dict]:
        """Retrieve all stock data from the collection.
        
        Returns:
            List of all stock documents.
            Empty list on error.
        """
        try:
            return list(self.collection.find())
        except Exception as e:
            logger.error(f"Error fetching all stocks: {e}")
            return []

    def get_stock_by_symbol(self, symbol: str) -> List[Dict]:
        """Get stock data for a specific symbol.
        
        Args:
            symbol: Stock symbol to query
            
        Returns:
            List of stock documents for the given symbol.
            Empty list on error or if symbol not found.
        """
        try:
            return list(self.collection.find({"symbol": symbol}))
        except Exception as e:
            logger.error(f"Error fetching stocks for {symbol}: {e}")
            return []

    def get_stock_by_date_range(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> List[Dict]:
        """Get stock data for a symbol within a date range.
        
        Args:
            symbol: Stock symbol to query
            start_date: Start date (inclusive) as string (YYYY-MM-DD)
            end_date: End date (inclusive) as string (YYYY-MM-DD)
            
        Returns:
            List of stock documents matching the criteria.
            Empty list on error or if no matches found.
        """
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
            logger.error(f"Error fetching {symbol} data from {start_date} to {end_date}: {e}")
            return []

    def get_unique_symbols(self) -> List[str]:
        """Get list of unique stock symbols in the database.
        
        Returns:
            List of unique stock symbols.
            Empty list on error.
        """
        try:
            return self.collection.distinct("symbol")
        except Exception as e:
            logger.error(f"Error fetching unique symbols: {e}")
            return []

    def get_latest_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get the most recent stock data for a symbol.
        
        Args:
            symbol: Stock symbol to query
            
        Returns:
            The most recent stock document for the symbol.
            None if not found or on error.
        """
        try:
            return self.collection.find_one(
                {"symbol": symbol},
                sort=[("transaction_date", -1)]  # Sort by date descending
            )
        except Exception as e:
            logger.error(f"Error fetching latest data for {symbol}: {e}")
            return None

    def get_stock_statistics(self, symbol: str) -> Dict[str, Union[float, int]]:
        """Calculate statistics (avg, max, min price, total volume) for a stock.
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Dictionary with statistics:
            {
                'avg_price': float,
                'max_price': float,
                'min_price': float,
                'total_volume': int
            }
            Empty dict on error.
        """
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
            logger.error(f"Error calculating statistics for {symbol}: {e}")
            return {}

    def get_stock_history(self, symbol: str) -> List[Dict]:
        """Get historical stock data with aggregated daily metrics.
        
        Args:
            symbol: Stock symbol to query
            
        Returns:
            List of daily stock documents with:
            - symbol
            - transaction_date
            - rate (last rate of the day)
            - quantity (total daily volume)
            - amount (total daily amount)
            - avg_price (average daily price)
            Sorted chronologically.
            Empty list on error.
        """
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
            logger.error(f"Error fetching history for {symbol}: {e}")
            return []

    # --------------------------
    # Caching Methods
    # --------------------------

    def cache_top_stocks(self, top_stocks: list) -> bool:
        """Cache top performing stocks in a dedicated collection.
        
        Args:
            top_stocks: List of stock documents to cache
            
        Returns:
            True if operation succeeded, False otherwise
        """
        try:
            cache_collection = self.db.top_stocks_cache
            cache_collection.delete_many({})  # Clear existing cache
            
            if top_stocks:
                cache_collection.insert_many(top_stocks)
            return True
        except Exception as e:
            logger.error(f"Error caching top stocks: {e}", exc_info=True)
            return False

    def get_cached_stocks(self) -> list:
        """Retrieve cached top performing stocks.
        
        Returns:
            List of cached stock documents.
            Empty list on error.
        """
        try:
            return list(self.db.top_stocks_cache.find())
        except Exception as e:
            logger.error(f"Error fetching cached stocks: {e}", exc_info=True)
            return []

    # --------------------------
    # Model Management Methods
    # --------------------------

    def save_model_and_scaler(self, symbol: str, model_data: tf.keras.Model, scaler_data) -> bool:
        """Save trained Keras model and scaler to database.
        
        Args:
            symbol: Stock symbol associated with the model
            model_data: Trained Keras model object
            scaler_data: Scikit-learn scaler object
            
        Returns:
            True if save succeeded, False otherwise
        """
        temp_model_path = None
        try:
            # Save model to temporary file
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
                temp_model_path = tmp_file.name
            model_data.save(temp_model_path)
            
            # Read model binary
            with open(temp_model_path, 'rb') as f:
                model_binary = f.read()

            # Prepare document
            model_doc = {
                'symbol': symbol,
                'model': model_binary,
                'scaler': pickle.dumps(scaler_data),
                'model_format': 'keras_file',
                'tensorflow_version': tf.__version__,
                'updated_at': datetime.now(),
            }
            
            # Upsert operation
            self.models_collection.update_one(
                {'symbol': symbol},
                {'$set': model_doc},
                upsert=True
            )
            
            logger.info(f"Saved model and scaler for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model for {symbol}: {str(e)}")
            return False
        finally:
            # Clean up temporary file
            if temp_model_path and os.path.exists(temp_model_path):
                os.remove(temp_model_path)

    def get_model_and_scaler(self, symbol: str) -> tuple:
        """Retrieve trained model and scaler from database.
        
        Args:
            symbol: Stock symbol associated with the model
            
        Returns:
            Tuple of (model, scaler) if found, (None, None) otherwise
        """
        temp_model_path = None
        try:
            model_doc = self.models_collection.find_one({'symbol': symbol})
            if not model_doc:
                logger.warning(f"No trained model found for {symbol}")
                return None, None

            # Handle different model formats
            if model_doc.get('model_format') == 'keras_file':
                with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
                    temp_model_path = tmp_file.name
                    tmp_file.write(model_doc['model'])
                
                model = tf.keras.models.load_model(temp_model_path)
                logger.info(f"Loaded Keras model for {symbol}")
            else:
                logger.warning(f"Model for {symbol} is in old pickle format")
                model = pickle.loads(model_doc['model'])
            
            scaler = pickle.loads(model_doc['scaler'])
            return model, scaler
            
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {str(e)}")
            return None, None
        finally:
            if temp_model_path and os.path.exists(temp_model_path):
                os.remove(temp_model_path)

    def get_existing_model_symbols(self) -> List[str]:
        """Get symbols of all stocks with trained models.
        
        Returns:
            List of stock symbols with trained models.
            Empty list on error.
        """
        try:
            return self.models_collection.distinct('symbol')
        except Exception as e:
            logger.error(f"Error fetching model symbols: {str(e)}")
            return []

    def get_model_metadata(self) -> List[Dict]:
        """Get metadata for all trained models (excluding binary data).
        
        Returns:
            List of model metadata dictionaries with:
            - symbol
            - model_format
            - tensorflow_version
            Empty list on error.
        """
        try:
            projection = {
                'symbol': 1, 
                'model_format': 1, 
                'tensorflow_version': 1, 
                '_id': 0
            }
            return list(self.models_collection.find({}, projection))
        except Exception as e:
            logger.error(f"Error fetching model metadata: {e}", exc_info=True)
            return []

    # --------------------------
    # Connection Management
    # --------------------------

    def close_connection(self):
        """Close the MongoDB connection."""
        try:
            self.client.close()
        except Exception as e:
            logger.error(f"Error closing connection: {e}")