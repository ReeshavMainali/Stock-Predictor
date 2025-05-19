from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime
from typing import List, Dict, Optional, Union
from functions.logger import logger
import pickle
import tensorflow as tf # For tf.keras.models.load_model and tf.__version__
import tempfile         # For creating temporary files for model saving/loading


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
        self.models_collection = self.db.models # Add reference to models collection
    
    def get_all_company_details(self) -> List[Dict]:
        """
        Get details (symbol and name) for companies that have a trained model.
        Returns the list sorted alphabetically by name.
        """
        try:
            # First, get the list of symbols for which models exist
            symbols_with_models = self.get_existing_model_symbols()
            
            if not symbols_with_models:
                logger.info("No trained models found, returning empty company list.")
                return []

            # Now, fetch company details only for those symbols
            # Use the $in operator to match documents where 'symbol' is in the list
            query = {
                'symbol': { '$in': symbols_with_models }
            }

            companies_cursor = self.companies_collection.find(
                query,
                {'symbol': 1, 'name': 1, '_id': 0}
            ).sort('name', 1) # Sort by name alphabetically
            
            return list(companies_cursor)
        except Exception as e:
            logger.error(f"Error fetching company details with models from MongoDB: {e}")
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
        # model_data is the Keras model object
        # scaler_data is the scikit-learn scaler object
        temp_model_path = None
        try:
            # Save Keras model to a temporary .keras file
            # tempfile.NamedTemporaryFile creates a file that is deleted when closed by default.
            # We need to save to its path, then read it, then it can be deleted.
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
                temp_model_path = tmp_file.name
            
            model_data.save(temp_model_path) # Keras saves in .keras format if path ends with .keras
            
            # Read the binary content of the saved model file
            with open(temp_model_path, 'rb') as f:
                model_binary = f.read()

            scaler_binary = pickle.dumps(scaler_data)
            
            # Prepare document
            model_doc = {
                'symbol': symbol,
                'model': model_binary,
                'scaler': scaler_binary,
                'model_format': 'keras_file', # Indicate the model storage format
                'tensorflow_version': tf.__version__, # Store TF version for debugging
                'updated_at': datetime.now(),
            }
            
            # Update or insert the model
            self.models_collection.update_one( # Use the models_collection attribute
                {'symbol': symbol},
                {'$set': model_doc},
                upsert=True
            )
            
            logger.info(f"Successfully saved model and scaler for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model for {symbol}: {str(e)}")
            return False
        finally:
            # Clean up the temporary model file
            if temp_model_path and os.path.exists(temp_model_path):
                os.remove(temp_model_path)

    def get_model_and_scaler(self, symbol):
        """
        Retrieve trained model and scaler for a symbol from the database
        """
        temp_model_path = None
        try:
            model_doc = self.models_collection.find_one({'symbol': symbol}) # Use the models_collection attribute
            if model_doc:
                # Check the model format
                if model_doc.get('model_format') == 'keras_file':
                    # Write the binary model data to a temporary .keras file
                    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
                        temp_model_path = tmp_file.name
                        tmp_file.write(model_doc['model'])
                    
                    # Load the model using tf.keras.models.load_model
                    # You might need to pass custom_objects if your model uses them,
                    # but your current model in model.py uses standard layers.
                    model = tf.keras.models.load_model(temp_model_path)
                    logger.info(f"Successfully loaded Keras model for {symbol} from file format.")
                else:
                    # Fallback for old pickled models (consider migrating these)
                    logger.warning(f"Model for {symbol} is in old pickle format. Attempting to load with pickle. This may fail.")
                    model = pickle.loads(model_doc['model'])
                
                scaler = pickle.loads(model_doc['scaler'])
                return model, scaler
            
            logger.warning(f"No trained model found in database for symbol {symbol}")
            return None, None
            
        except Exception as e:
            logger.error(f"Error retrieving model for {symbol}: {str(e)}")
            return None, None
        finally:
            # Clean up the temporary model file
            if temp_model_path and os.path.exists(temp_model_path):
                os.remove(temp_model_path)
                
    def get_existing_model_symbols(self) -> List[str]:
        """Get all symbols that already have trained models"""
        try:
            # Use the models_collection attribute
            existing_models = self.models_collection.distinct('symbol')
            return existing_models
        except Exception as e:
            logger.error(f"Error fetching existing model symbols: {str(e)}")
            return []

    def get_model_metadata(self) -> List[Dict]:
        """
        Get metadata (symbol, format, TF version) for all trained models.
        Excludes the binary model and scaler data.
        """
        try:
            # Project only the required fields, exclude the binary data and _id
            projection = {'symbol': 1, 'model_format': 1, 'tensorflow_version': 1, '_id': 0}
            metadata_list = list(self.models_collection.find({}, projection))
            logger.debug(f"Retrieved metadata for {len(metadata_list)} models")
            return metadata_list
        except Exception as e:
            logger.error(f"Error fetching model metadata: {e}", exc_info=True)
            return []
