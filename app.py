"""
Flask application for stock market analysis and prediction.

This application provides:
- Dashboard with top performing stocks
- Historical stock data visualization
- Stock price prediction functionality
- Model training endpoints
- API endpoints for data access
"""

from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from functions.db_data_manager import DatabaseManager
from functions.logger import logger
from model.model import train_model, predict_future, calculate_rsi, preprocess_transaction_data
from datetime import timedelta
import io
import sys
from typing import List, Dict, Optional
from functions.helpers import _calculate_percentage_change , _prepare_prediction_data , _prepare_top_stocks_data

# Disable TensorFlow OneDNN optimizations for compatibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# --------------------------
# Route Handlers
# --------------------------

@app.route('/')
def index() -> str:
    """Render dashboard with top performing stocks.
    
    Supports cache refresh via ?refresh=1 query parameter.
    """
    logger.info("Accessing index page")
    db_manager = DatabaseManager()
    
    try:
        # Check for cached data unless refresh requested
        if request.args.get('refresh', '0') != '1':
            cached_stocks = db_manager.get_cached_stocks()
            if cached_stocks:
                logger.info("Using cached top stocks data")
                return render_template('index.html', top_stocks=cached_stocks)
        
        # Calculate fresh data if no cache or refresh requested
        logger.info("Calculating fresh top stocks data")
        top_stocks = _prepare_top_stocks_data(db_manager)
        db_manager.cache_top_stocks(top_stocks)
        
        return render_template('index.html', top_stocks=top_stocks)
        
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}", exc_info=True)
        return render_template('index.html', top_stocks=[])
    finally:
        db_manager.close_connection()
        logger.debug("Database connection closed")

@app.route('/history/<symbol>')
@app.route('/history')
def history(symbol: Optional[str] = None) -> str:
    """Render stock history page for a specific symbol.
    
    Args:
        symbol: Stock symbol to display history for
        
    Returns:
        Rendered template with historical data
    """
    logger.info("Accessing history page")
    db_manager = DatabaseManager()
    
    try:
        # Get all companies for dropdown
        all_companies = db_manager.get_all_company_details()
        
        # Get symbol from URL or query param
        symbol = symbol or request.args.get('symbol')
        if not symbol:
            logger.warning("No symbol provided for history")
            return render_template('history.html', 
                                companies=all_companies, 
                                symbol=None, 
                                error="No stock symbol provided")
        
        # Fetch and process history data
        logger.info(f"Fetching historical data for {symbol}")
        history_data = db_manager.get_stock_history(symbol)
        logger.info(f"Retrieved {len(history_data)} records for {symbol}")
        
        # Calculate percentage changes
        for data in history_data:
            data['change'] = _calculate_percentage_change(
                data['rate'], 
                data['avg_price']
            )
            
        return render_template('history.html', 
                            companies=all_companies, 
                            symbol=symbol, 
                            history_data=history_data)
        
    except Exception as e:
        logger.error(f"Error fetching history for {symbol}: {str(e)}", exc_info=True)
        return render_template('history.html', 
                            companies=all_companies, 
                            symbol=symbol, 
                            history_data=[])
    finally:
        db_manager.close_connection()
        logger.debug("Database connection closed")

@app.route('/predict/<symbol>')
@app.route('/predict')
def predict(symbol: Optional[str] = None) -> str:
    """Render stock prediction page with forecast data.
    
    Args:
        symbol: Stock symbol to predict
        days: Number of days to predict (query parameter, default=30)
        
    Returns:
        Rendered template with prediction data
    """
    logger.info("Accessing prediction page")
    db_manager = DatabaseManager()
    
    try:
        # Get all companies for dropdown
        all_companies = db_manager.get_all_company_details()
        
        # Get parameters
        symbol = symbol or request.args.get('symbol')
        num_days = int(request.args.get('days', 30))
        
        if not symbol:
            logger.warning("No symbol provided for prediction")
            return render_template('predict.html', 
                                companies=all_companies, 
                                symbol=None, 
                                error="No stock symbol provided")
        
        # Fetch and validate data
        stock_data = db_manager.get_stock_history(symbol)
        if len(stock_data) < 60:
            logger.warning(f"Insufficient data for {symbol}")
            return render_template('predict.html', 
                                companies=all_companies, 
                                symbol=symbol, 
                                error="Insufficient historical data (minimum 60 days required)")
        
        # Check for existing model
        model, scaler = db_manager.get_model_and_scaler(symbol)
        if model is None:
            logger.warning(f"No trained model for {symbol}")
            return render_template('predict.html', 
                                companies=all_companies, 
                                symbol=symbol, 
                                error="No trained model available. Please train first.")
        
        # Prepare data and make predictions
        df = pd.DataFrame(stock_data)
        df = preprocess_transaction_data(df, symbol)
        df = df.sort_values('transaction_date')
        
        # Calculate technical indicators
        df['SMA_5'] = df['rate'].rolling(window=5).mean()
        df['SMA_20'] = df['rate'].rolling(window=20).mean()
        df['RSI'] = calculate_rsi(df['rate'])
        df['Volatility'] = df['rate'].rolling(window=20).std()
        
        # Prepare features and predict
        features = ['rate', 'SMA_5', 'SMA_20', 'RSI', 'Volatility']
        data = df[features].dropna().values
        scaled_data = scaler.transform(data)
        predictions = predict_future(model, scaler, scaled_data[-60:], num_days=num_days)
        
        # Combine historical and prediction data
        display_data = _prepare_prediction_data(stock_data, predictions, num_days)
        
        return render_template('predict.html', 
                            companies=all_companies, 
                            symbol=symbol, 
                            stock_data=display_data, 
                            days=num_days)
        
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {str(e)}", exc_info=True)
        return render_template('predict.html', 
                            companies=all_companies, 
                            symbol=symbol, 
                            error=f"Prediction error: {str(e)}")
    finally:
        db_manager.close_connection()
        logger.debug("Database connection closed")

@app.route('/train_models')
def train_models() -> jsonify:
    """Batch train models for all symbols with sufficient data.
    
    Returns:
        JSON response with training results
    """
    logger.info("Starting batch training")
    db_manager = DatabaseManager()
    
    try:
        symbols = db_manager.get_unique_symbols()
        existing_models = set(db_manager.get_existing_model_symbols())
        training_results = []
        
        for symbol in symbols:
            try:
                if symbol in existing_models:
                    training_results.append({
                        'symbol': symbol,
                        'status': 'skipped',
                        'reason': 'model exists'
                    })
                    continue
                    
                stock_data = db_manager.get_stock_history(symbol)
                if len(stock_data) <= 60:
                    training_results.append({
                        'symbol': symbol,
                        'status': 'skipped',
                        'reason': 'insufficient data',
                        'data_points': len(stock_data)
                    })
                    continue
                    
                # Train and save model
                df = pd.DataFrame(stock_data)
                df = preprocess_transaction_data(df, symbol)
                model, scaler = train_model(df, seq_length=60, epochs=100)
                
                training_results.append({
                    'symbol': symbol,
                    'status': 'success' if db_manager.save_model_and_scaler(symbol, model, scaler) else 'failed',
                    'data_points': len(stock_data)
                })
                
            except Exception as e:
                training_results.append({
                    'symbol': symbol,
                    'status': 'failed',
                    'error': str(e)
                })
                logger.error(f"Training failed for {symbol}: {str(e)}")
                
        return jsonify({
            'status': 'completed',
            'results': training_results
        })
        
    except Exception as e:
        logger.error(f"Batch training error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        db_manager.close_connection()
        logger.debug("Database connection closed")

@app.route('/train/<symbol>')
def train_single_model(symbol: str) -> jsonify:
    """Train model for a single stock symbol.
    
    Args:
        symbol: Stock symbol to train model for
        
    Returns:
        JSON response with training status
    """
    logger.info(f"Training model for {symbol}")
    db_manager = DatabaseManager()
    
    try:
        # Check for existing model
        if symbol in db_manager.get_existing_model_symbols():
            return jsonify({
                'symbol': symbol,
                'status': 'skipped',
                'reason': 'model exists'
            })
            
        # Check data requirements
        stock_data = db_manager.get_stock_history(symbol)
        if len(stock_data) <= 60:
            return jsonify({
                'symbol': symbol,
                'status': 'skipped',
                'reason': 'insufficient data',
                'data_points': len(stock_data)
            })
            
        # Train and save model
        df = pd.DataFrame(stock_data)
        df = preprocess_transaction_data(df, symbol)
        model, scaler = train_model(df, seq_length=60, epochs=100)
        
        if not db_manager.save_model_and_scaler(symbol, model, scaler):
            return jsonify({
                'symbol': symbol,
                'status': 'failed',
                'error': 'Failed to save model'
            }), 500
            
        return jsonify({
            'symbol': symbol,
            'status': 'success',
            'data_points': len(stock_data)
        })
        
    except Exception as e:
        logger.error(f"Training failed for {symbol}: {str(e)}", exc_info=True)
        return jsonify({
            'symbol': symbol,
            'status': 'failed',
            'error': str(e)
        }), 500
    finally:
        db_manager.close_connection()
        logger.debug("Database connection closed")

@app.route('/api/stocks/search')
def search_stocks() -> jsonify:
    """Search stocks by symbol or name.
    
    Returns:
        JSON response with matching stocks (max 10)
    """
    term = request.args.get('term', '').upper()
    db_manager = DatabaseManager()
    
    try:
        all_companies = db_manager.get_all_company_details()
        matches = [
            {'symbol': c['symbol'], 'name': c.get('name', c['symbol'])}
            for c in all_companies
            if term in c['symbol'] or term in c.get('name', '').upper()
        ]
        return jsonify(matches[:10])
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify([])
    finally:
        db_manager.close_connection()

@app.route('/models')
def list_models() -> str:
    """Render page listing all trained models."""
    logger.info("Accessing models page")
    db_manager = DatabaseManager()
    
    try:
        models_metadata = db_manager.get_model_metadata()
        logger.info(f"Found {len(models_metadata)} models")
        return render_template('models.html', models=models_metadata)
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}", exc_info=True)
        return render_template('models.html', models=[])
    finally:
        db_manager.close_connection()

@app.route('/api/models/<symbol>/structure')
def get_model_structure(symbol: str) -> jsonify:
    """Get model structure summary for a stock symbol.
    
    Args:
        symbol: Stock symbol to get model for
        
    Returns:
        JSON response with model summary or error
    """
    logger.info(f"Fetching model structure for {symbol}")
    db_manager = DatabaseManager()
    
    try:
        model, _ = db_manager.get_model_and_scaler(symbol)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
            
        # Capture model summary
        stream = io.StringIO()
        sys.stdout = stream
        model.summary()
        sys.stdout = sys.__stdout__
        
        return jsonify({
            'symbol': symbol,
            'structure': stream.getvalue()
        })
    except Exception as e:
        logger.error(f"Model structure error for {symbol}: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    finally:
        db_manager.close_connection()

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True, port=5000)