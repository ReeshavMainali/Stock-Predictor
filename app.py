from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from functions.db_data_manager import DatabaseManager
from functions.logger import logger
from model.model import train_model, predict_future, calculate_rsi, preprocess_transaction_data
from datetime import datetime, timedelta
import io,sys

# Set the environment variable to disable TensorFlow OneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

@app.route('/')
def index():
    logger.info("Accessing index page")
    db_manager = DatabaseManager()

    try:
        refresh = request.args.get('refresh', '0') == '1'

        if not refresh:
            logger.debug("Attempting to fetch cached top stocks")
            top_stocks = db_manager.get_cached_stocks()
            if top_stocks:
                logger.info("Using cached top stocks data")
                return render_template('index.html', top_stocks=top_stocks)

        logger.info("Calculating fresh top stocks data")
        symbols = db_manager.get_unique_symbols()
        top_stocks = []
        for symbol in symbols:
            latest_data = db_manager.get_latest_stock_data(symbol)
            if latest_data:
                stats = db_manager.get_stock_statistics(symbol)
                current_rate = latest_data['rate']
                avg_price = stats.get('avg_price', current_rate)
                change = ((current_rate - avg_price) / avg_price) * 100 if avg_price else 0
                stock_data = {
                    'symbol': symbol,
                    'rate': latest_data['rate'],
                    'quantity': latest_data['quantity'],
                    'amount': latest_data['amount'],
                    'change': round(change, 2)
                }
                top_stocks.append(stock_data)

        top_stocks.sort(key=lambda x: x['change'], reverse=True)
        top_stocks = top_stocks[:10]
        logger.info("Caching new top stocks data")
        db_manager.cache_top_stocks(top_stocks)

    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}", exc_info=True)
        top_stocks = []
    finally:
        db_manager.close_connection()
        logger.debug("Database connection closed")

    return render_template('index.html', top_stocks=top_stocks)

@app.route('/history/<symbol>')
@app.route('/history')
def history(symbol=None):
    logger.info("Accessing history page")
    db_manager = DatabaseManager()
    all_companies = db_manager.get_all_company_details() # Fetch all companies

    try:
        if symbol is None:
            symbol = request.args.get('symbol')
        if not symbol:
            logger.warning("No symbol provided for history")
            # Pass companies even if no symbol is selected yet
            return render_template('history.html', companies=all_companies, symbol=None, error="No stock symbol provided")

        logger.info(f"Fetching historical data for symbol: {symbol}")
        history_data = db_manager.get_stock_history(symbol)
        logger.info(f"Retrieved {len(history_data)} historical records for {symbol}")

        logger.debug("Calculating change percentages")
        for data in history_data:
            avg_price = data['avg_price']
            current_rate = data['rate']
            change = ((current_rate - avg_price) / avg_price) * 100 if avg_price else 0
            data['change'] = round(change, 2)

    except Exception as e:
        logger.error(f"Error fetching stock history data for {symbol}: {str(e)}", exc_info=True)
        history_data = []
    finally:
        db_manager.close_connection()
        logger.debug("Database connection closed")

    return render_template('history.html', companies=all_companies, symbol=symbol, history_data=history_data)

@app.route('/predict/<symbol>')
@app.route('/predict')
def predict(symbol=None):
    logger.info("Accessing prediction page")
    db_manager = DatabaseManager()
    all_companies = db_manager.get_all_company_details() # Use DatabaseManager

    try:
        if symbol is None:
            symbol = request.args.get('symbol')
        num_days = int(request.args.get('days', 30))


        if not symbol:
            logger.warning("No symbol provided for prediction")
            return render_template('predict.html', companies=all_companies, symbol=None, error="No stock symbol provided")

        logger.debug(f"Fetching stock data for prediction: {symbol}")
        stock_data = db_manager.get_stock_history(symbol)
        logger.info(f"Retrieved {len(stock_data)} records for prediction")

        if len(stock_data) < 60:
            logger.warning(f"Insufficient data for {symbol} (needs at least 60 days)")
            return render_template('predict.html', companies=all_companies, symbol=symbol, error="Insufficient historical data for prediction (minimum 60 days required)")

        df = pd.DataFrame(stock_data)
        df = preprocess_transaction_data(df, symbol)

        model, scaler = db_manager.get_model_and_scaler(symbol)
        if model is None or scaler is None:
            logger.warning(f"No trained model found for symbol {symbol}")
            return render_template('predict.html', companies=all_companies, symbol=symbol, error="No trained model available for this symbol. Please train it first.")

        df = df.sort_values('transaction_date')
        df['SMA_5'] = df['rate'].rolling(window=5).mean()
        df['SMA_20'] = df['rate'].rolling(window=20).mean()
        df['RSI'] = calculate_rsi(df['rate'])
        df['Volatility'] = df['rate'].rolling(window=20).std()
        features = ['rate', 'SMA_5', 'SMA_20', 'RSI', 'Volatility']
        data = df[features].dropna().values
        scaled_data = scaler.transform(data)
        last_sequence = scaled_data[-60:]
        predictions = predict_future(model, scaler, last_sequence, num_days=num_days)

        last_date = df['transaction_date'].iloc[-1]
        future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(num_days)]
        last_historical_price = df['rate'].iloc[-1]

        if len(predictions) > 0:
            scaling_factor = last_historical_price / predictions[0][0]
            predictions = predictions * scaling_factor

        prediction_data = [
            {
                'transaction_date': date,
                'rate': float(price),
                'is_prediction': True
            } for date, price in zip(future_dates, predictions.flatten())
        ]

        for data in stock_data:
            data['is_prediction'] = False
        display_historical = stock_data[-30:]
        display_data = display_historical + prediction_data

        return render_template('predict.html', companies=all_companies, symbol=symbol, stock_data=display_data, days=num_days)

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return render_template('predict.html', companies=all_companies, symbol=symbol, error=f"Error in prediction: {str(e)}")
    finally:
        db_manager.close_connection()
        logger.debug("Database connection closed")

@app.route('/train_models')
def train_models():
    logger.info("Starting batch training for all symbols")
    db_manager = DatabaseManager()

    try:
        symbols = db_manager.get_unique_symbols()
        logger.info(f"Found {len(symbols)} symbols to train")
        existing_models = set(db_manager.get_existing_model_symbols())
        logger.info(f"Found {len(existing_models)} existing models")

        training_results = []
        for symbol in symbols:
            try:
                if symbol in existing_models:
                    logger.info(f"Model for {symbol} already exists. Skipping training.")
                    training_results.append({
                        'symbol': symbol,
                        'status': 'skipped',
                        'reason': 'model already exists'
                    })
                    continue

                stock_data = db_manager.get_stock_history(symbol)

                if len(stock_data) > 60:
                    df = pd.DataFrame(stock_data)
                    df = preprocess_transaction_data(df, symbol)
                    model, scaler = train_model(df, seq_length=60, epochs=100)
                    save_success = db_manager.save_model_and_scaler(symbol, model, scaler)
                    training_results.append({
                        'symbol': symbol,
                        'status': 'success' if save_success else 'failed_to_save',
                        'data_points': len(stock_data)
                    })
                    logger.info(f"Successfully trained model for {symbol}")
                else:
                    training_results.append({
                        'symbol': symbol,
                        'status': 'skipped',
                        'reason': 'insufficient data',
                        'data_points': len(stock_data)
                    })
                    logger.warning(f"Skipped {symbol} due to insufficient data")

            except Exception as e:
                training_results.append({
                    'symbol': symbol,
                    'status': 'failed',
                    'error': str(e)
                })
                logger.error(f"Error training model for {symbol}: {str(e)}")

    except Exception as e:
        logger.error(f"Error in batch training: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        db_manager.close_connection()
        logger.debug("Database connection closed")

    return jsonify({
        'status': 'completed',
        'results': training_results
    })

@app.route('/train/<symbol>')
def train_single_model(symbol):
    logger.info(f"Attempting to train model for single symbol: {symbol}")
    db_manager = DatabaseManager()

    try:
        # Check if a model already exists for this symbol
        existing_models = db_manager.get_existing_model_symbols()
        if symbol in existing_models:
            logger.info(f"Model for {symbol} already exists. Skipping training.")
            return jsonify({
                'symbol': symbol,
                'status': 'skipped',
                'reason': 'model already exists'
            })

        stock_data = db_manager.get_stock_history(symbol)
        logger.info(f"Retrieved {len(stock_data)} records for training {symbol}")

        if len(stock_data) > 60: # Ensure sufficient data
            df = pd.DataFrame(stock_data)
            df = preprocess_transaction_data(df, symbol)

            logger.info(f"Starting training for {symbol}...")
            # Assuming train_model returns model, scaler
            model, scaler = train_model(df, seq_length=60, epochs=100) # Use same parameters as batch training
            logger.info(f"Training completed for {symbol}.")

            save_success = db_manager.save_model_and_scaler(symbol, model, scaler)

            if save_success:
                logger.info(f"Successfully saved model for {symbol}")
                return jsonify({
                    'symbol': symbol,
                    'status': 'success',
                    'data_points': len(stock_data)
                })
            else:
                logger.error(f"Failed to save model for {symbol}")
                return jsonify({
                    'symbol': symbol,
                    'status': 'failed_to_save',
                    'data_points': len(stock_data)
                }), 500 # Indicate server error if saving fails

        else:
            logger.warning(f"Insufficient data for {symbol} (needs at least 60 days)")
            return jsonify({
                'symbol': symbol,
                'status': 'skipped',
                'reason': 'insufficient data',
                'data_points': len(stock_data)
            })

    except Exception as e:
        logger.error(f"Error training model for {symbol}: {str(e)}", exc_info=True)
        return jsonify({
            'symbol': symbol,
            'status': 'failed',
            'error': str(e)
        }), 500 # Indicate server error on training failure
    finally:
        db_manager.close_connection()
        logger.debug("Database connection closed")



@app.route('/api/stocks/search')
def search_stocks():
    term = request.args.get('term', '').upper()
    db_manager = DatabaseManager()
    try:
        # Fetch company details (symbol and name)
        all_company_details = db_manager.get_all_company_details()
        matches = [
            {'symbol': company['symbol'], 'name': company.get('name', company['symbol'])}
            for company in all_company_details
            if term in company['symbol'] or (company.get('name') and term in company.get('name', '').upper())
        ]
        return jsonify(matches[:10])
    except Exception as e:
        logger.error(f"Error searching stocks: {str(e)}")
        return jsonify([])
    finally:
        db_manager.close_connection()

@app.route('/models')
def list_models():
    logger.info("Accessing models list page")
    db_manager = DatabaseManager()
    models_metadata = []
    try:
        # Use the new method to fetch model metadata
        models_metadata = db_manager.get_model_metadata()
        logger.info(f"Found {len(models_metadata)} trained models")
    except Exception as e:
        logger.error(f"Error fetching model metadata: {str(e)}", exc_info=True)
        # models_metadata remains empty list
    finally:
        db_manager.close_connection()
        logger.debug("Database connection closed")

    # Render the new template, passing the list of models
    return render_template('models.html', models=models_metadata)

@app.route('/api/models/<symbol>/structure')
def get_model_structure(symbol):
    logger.info(f"Fetching model structure for symbol: {symbol}")
    db_manager = DatabaseManager()
    try:
        model, _ = db_manager.get_model_and_scaler(symbol)

        if model is None:
            logger.warning(f"Model not found for symbol {symbol}")
            return jsonify({'error': 'Model not found'}), 404

        # Capture model summary output
        stream = io.StringIO()
        sys.stdout = stream
        model.summary()
        sys.stdout = sys.__stdout__ # Restore stdout
        summary_string = stream.getvalue()

        logger.debug(f"Successfully generated model summary for {symbol}")
        return jsonify({'symbol': symbol, 'structure': summary_string})

    except Exception as e:
        logger.error(f"Error getting model structure for {symbol}: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500
    finally:
        db_manager.close_connection()
        logger.debug("Database connection closed")


if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True, port=5000)
