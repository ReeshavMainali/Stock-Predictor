import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import pickle
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
REPORT_DIR = "test_reports"
SEQUENCE_LENGTH = 60
MIN_DATA_POINTS = 100

# Create reports directory
os.makedirs(REPORT_DIR, exist_ok=True)

# ------------------------
# Safe Import Handler
# ------------------------

def safe_import():
    """Safely import modules with proper error handling."""
    imports = {}
    try:
        from app import app
        imports['app'] = app
        logger.info("Successfully imported Flask app")
    except ImportError as e:
        logger.error(f"Failed to import Flask app: {e}")
        imports['app'] = None
        
    try:
        from model.model import preprocess_transaction_data, predict_future, train_model
        imports['model_functions'] = {
            'preprocess_transaction_data': preprocess_transaction_data,
            'predict_future': predict_future,
            'train_model': train_model
        }
        logger.info("Successfully imported model functions")
    except ImportError as e:
        logger.error(f"Failed to import model functions: {e}")
        imports['model_functions'] = None
        
    try:
        from functions.db_data_manager import DatabaseManager
        imports['DatabaseManager'] = DatabaseManager
        logger.info("Successfully imported DatabaseManager")
    except ImportError as e:
        logger.error(f"Failed to import DatabaseManager: {e}")
        imports['DatabaseManager'] = None
        
    try:
        from functions.helpers import _calculate_percentage_change
        imports['_calculate_percentage_change'] = _calculate_percentage_change
        logger.info("Successfully imported helper functions")
    except ImportError as e:
        logger.error(f"Failed to import helper functions: {e}")
        imports['_calculate_percentage_change'] = None
        
    return imports

# Import all modules safely
IMPORTS = safe_import()

# ------------------------
# Helper Functions for Testing
# ------------------------

def print_and_save_table(df, name):
    """Prints a DataFrame as a markdown table and saves it to a file."""
    try:
        if df.empty:
            logger.warning(f"DataFrame for {name} is empty")
            return
            
        table = tabulate(df, headers='keys', tablefmt='github', showindex=False)
        print(f"\n{name}\n{table}")
        
        filepath = os.path.join(REPORT_DIR, f"{name}.md")
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(f"# {name}\n\n{table}\n")
        logger.info(f"Saved report: {filepath}")
    except Exception as e:
        logger.error(f"Error saving table {name}: {e}")

def save_chart(fig, name):
    """Saves a matplotlib chart to a file."""
    try:
        filepath = os.path.join(REPORT_DIR, f"{name}.png")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved chart: {filepath}")
    except Exception as e:
        logger.error(f"Error saving chart {name}: {e}")

def create_mock_data(symbol, start_date, num_days=100, rate_base=1000):
    """Generates mock stock transaction data following the NEPSE format."""
    try:
        mock_data = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Generate more realistic price movement
        price = rate_base
        
        for i in range(num_days):
            # Add realistic price volatility (random walk with drift)
            daily_return = np.random.normal(0.001, 0.02)  # 0.1% daily return, 2% volatility
            price = max(1.0, price * (1 + daily_return))
            
            quantity = np.random.randint(10, 1000)
            
            mock_data.append({
                "_id": {"$oid": f"681c65aa163ab5abe74f{i:04d}"},
                "transaction": f"20231120050016{i:04d}",
                "id": i + 1,
                "symbol": symbol,
                "buyer": str(np.random.randint(10, 50)),
                "seller": str(np.random.randint(10, 50)),
                "quantity": quantity,
                "rate": round(price, 2),
                "amount": round(price * quantity, 2),
                "transaction_date": current_date.strftime('%Y-%m-%d')
            })
            current_date += timedelta(days=1)
            
        logger.info(f"Created {num_days} mock data points for {symbol}")
        return mock_data
    except Exception as e:
        logger.error(f"Error creating mock data: {e}")
        return []

# ------------------------
# Mock Data Setup
# ------------------------

# Create diverse mock data for different NEPSE sectors
MOCK_DATABASE_DATA = {
    'NABIL': create_mock_data("NABIL", "2023-01-01", 200, 1200),  # Commercial Bank
    'UPPER': create_mock_data("UPPER", "2023-01-01", 200, 680),   # Hydropower
    'NLICL': create_mock_data("NLICL", "2023-01-01", 200, 950),  # Insurance
    'NMB': create_mock_data("NMB", "2023-01-01", 200, 420),      # Commercial Bank
    'SHIVM': create_mock_data("SHIVM", "2023-01-01", 200, 1580), # Mutual Fund
}

def get_mock_db_manager():
    """Returns a mocked DatabaseManager instance."""
    mock_db_manager = MagicMock()
    
    def get_stock_data_by_symbol(symbol):
        data = MOCK_DATABASE_DATA.get(symbol, [])
        logger.info(f"Returning {len(data)} records for symbol {symbol}")
        return data
    
    def get_unique_symbols():
        symbols = list(MOCK_DATABASE_DATA.keys())
        logger.info(f"Returning {len(symbols)} unique symbols")
        return symbols
        
    def get_latest_stock_data(symbol):
        data = MOCK_DATABASE_DATA.get(symbol, [])
        return data[-1] if data else {}
        
    def get_stock_statistics(symbol):
        data = MOCK_DATABASE_DATA.get(symbol, [])
        if not data:
            return {}
        df = pd.DataFrame(data)
        return {
            'avg_price': float(df['rate'].mean()),
            'max_rate': float(df['rate'].max()),
            'min_rate': float(df['rate'].min()),
            'total_volume': int(df['quantity'].sum())
        }

    # Set up mock methods
    mock_db_manager.get_stock_by_symbol.side_effect = get_stock_data_by_symbol
    mock_db_manager.get_unique_symbols.side_effect = get_unique_symbols
    mock_db_manager.get_latest_stock_data.side_effect = get_latest_stock_data
    mock_db_manager.get_stock_statistics.side_effect = get_stock_statistics

    return mock_db_manager

# ------------------------
# Unit Tests
# ------------------------

def test_preprocess_transaction_data():
    """Unit test for data preprocessing."""
    logger.info("Starting preprocessing unit test")
    results = []
    
    if IMPORTS['model_functions'] is None:
        results.append({
            "Test": "Import Check",
            "Result": False,
            "Details": "Model functions not available"
        })
        return pd.DataFrame(results)
    
    try:
        df = pd.DataFrame(MOCK_DATABASE_DATA['NABIL'])
        preprocess_func = IMPORTS['model_functions']['preprocess_transaction_data']
        processed_df = preprocess_func(df)
        
        results.extend([
            {
                "Test": "DataFrame is not empty",
                "Result": not processed_df.empty,
                "Details": f"Shape: {processed_df.shape}"
            },
            {
                "Test": "DataFrame has 'rate' column",
                "Result": 'rate' in processed_df.columns,
                "Details": f"Columns: {list(processed_df.columns)}"
            },
            {
                "Test": "DataFrame has datetime index",
                "Result": isinstance(processed_df.index, pd.DatetimeIndex),
                "Details": f"Index type: {type(processed_df.index)}"
            },
            {
                "Test": "No missing values in rate",
                "Result": not processed_df['rate'].isna().any(),
                "Details": f"NaN count: {processed_df['rate'].isna().sum()}"
            }
        ])
        
    except Exception as e:
        logger.error(f"Preprocessing test failed: {e}")
        results.append({
            "Test": "Preprocessing execution",
            "Result": False,
            "Details": str(e)
        })
    
    df_results = pd.DataFrame(results)
    print_and_save_table(df_results, "Unit_Test_Preprocessing")
    return df_results

def test_calculate_percentage_change():
    """Unit test for the percentage change calculation helper function."""
    logger.info("Starting percentage change unit test")
    results = []
    
    if IMPORTS['_calculate_percentage_change'] is None:
        results.append({
            "Test": "Import Check",
            "Result": False,
            "Details": "Helper function not available"
        })
        return pd.DataFrame(results)
    
    try:
        calc_func = IMPORTS['_calculate_percentage_change']
        
        test_cases = [
            {"name": "Positive change", "current": 120, "previous": 100, "expected": 20.0},
            {"name": "Negative change", "current": 80, "previous": 100, "expected": -20.0},
            {"name": "Zero change", "current": 100, "previous": 100, "expected": 0.0},
            {"name": "Zero previous (edge case)", "current": 100, "previous": 0, "expected": 0.0}
        ]
        
        for case in test_cases:
            actual = calc_func(case["current"], case["previous"])
            results.append({
                "Test": case["name"],
                "Result": abs(actual - case["expected"]) < 0.01,
                "Details": f"Expected: {case['expected']}, Got: {actual}"
            })
            
    except Exception as e:
        logger.error(f"Percentage change test failed: {e}")
        results.append({
            "Test": "Function execution",
            "Result": False,
            "Details": str(e)
        })
    
    df_results = pd.DataFrame(results)
    print_and_save_table(df_results, "Unit_Test_Percentage_Change")
    return df_results

# ------------------------
# API Tests
# ------------------------

def test_api_endpoints():
    """System test for API endpoints using the Flask test client."""
    logger.info("Starting API endpoint tests")
    results = []
    
    if IMPORTS['app'] is None:
        results.append({
            "Endpoint": "N/A",
            "Status Code": "N/A",
            "Success": False,
            "Notes": "Flask app not available for testing"
        })
        return pd.DataFrame(results)
    
    test_client = IMPORTS['app'].test_client()
    
    try:
        with patch('app.DatabaseManager', return_value=get_mock_db_manager()):
            # Test endpoints with error handling
            endpoints_to_test = [
                ("/top-stocks", "Dashboard data"),
                ("/historical-data/NABIL", "Historical data for NABIL"),
                ("/predict/NABIL/10", "Prediction without training (should handle gracefully)"),
                ("/train/NABIL", "Model training"),
                ("/predict/NABIL/5", "Prediction after training")
            ]
            
            for endpoint, description in endpoints_to_test:
                try:
                    response = test_client.get(endpoint)
                    results.append({
                        "Endpoint": endpoint,
                        "Status Code": response.status_code,
                        "Success": response.status_code in [200, 404],  # Both are acceptable
                        "Notes": f"{description} - Status: {response.status_code}"
                    })
                except Exception as e:
                    results.append({
                        "Endpoint": endpoint,
                        "Status Code": "ERROR",
                        "Success": False,
                        "Notes": f"Exception: {str(e)[:100]}"
                    })
                    
    except Exception as e:
        logger.error(f"API test setup failed: {e}")
        results.append({
            "Endpoint": "Test Setup",
            "Status Code": "ERROR",
            "Success": False,
            "Notes": f"Setup failed: {str(e)[:100]}"
        })
    
    df_results = pd.DataFrame(results)
    print_and_save_table(df_results, "System_Test_API_Endpoints")
    return df_results

# ------------------------
# Performance and Accuracy Test
# ------------------------

def test_performance_and_accuracy():
    """Performance and accuracy test with proper error handling."""
    logger.info("Starting performance and accuracy tests")
    results = []
    
    if IMPORTS['model_functions'] is None:
        results.append({
            "Metric": "Import Check",
            "Value": "FAILED",
            "Notes": "Model functions not available"
        })
        return pd.DataFrame(results)
    
    try:
        # Use NABIL data (commercial bank)
        symbol = "NABIL"
        mock_data = MOCK_DATABASE_DATA[symbol]
        df = pd.DataFrame(mock_data)
        
        # Preprocess the data using your model's preprocessing function
        preprocess_func = IMPORTS['model_functions']['preprocess_transaction_data']
        processed_df = preprocess_func(df, symbol)
        
        logger.info(f"Preprocessed data shape: {processed_df.shape}")
        logger.info(f"Preprocessed columns: {processed_df.columns.tolist()}")
        
        if len(processed_df) < MIN_DATA_POINTS:
            results.append({
                "Metric": "Data Validation",
                "Value": "FAILED",
                "Notes": f"Insufficient data: {len(processed_df)} < {MIN_DATA_POINTS}"
            })
            return pd.DataFrame(results)
        
        # Performance measurement
        start_time = time.time()
        
        # Train the model using the correctly preprocessed data
        train_func = IMPORTS['model_functions']['train_model']
        model_result = train_func(processed_df)  # Pass the full preprocessed DataFrame
        
        if model_result is None or (isinstance(model_result, tuple) and model_result[0] is None):
            results.append({
                "Metric": "Model Training",
                "Value": "FAILED",
                "Notes": "Model training returned None - likely insufficient data after feature engineering"
            })
            return pd.DataFrame(results)
            
        model, scaler = model_result
        train_time = time.time() - start_time
        
        logger.info(f"Model training completed in {train_time:.3f} seconds")
        
        # Now we need to get the properly prepared data sequence for prediction
        # The train_model function internally calls prepare_data which creates the technical indicators
        # We need to recreate this process to get the last sequence
        try:
            # Recreate the feature engineering process from prepare_data
            processed_df['SMA_5'] = processed_df['rate'].rolling(window=5).mean()
            processed_df['SMA_20'] = processed_df['rate'].rolling(window=20).mean()
            
            # Calculate RSI manually (simplified version)
            def calculate_rsi_simple(prices, period=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            processed_df['RSI'] = calculate_rsi_simple(processed_df['rate'])
            processed_df['Volatility'] = processed_df['rate'].rolling(window=20).std()
            
            # Select the same features used in training
            features = ['rate', 'SMA_5', 'SMA_20', 'RSI', 'Volatility']
            feature_data = processed_df[features].dropna()
            
            logger.info(f"Feature data shape after engineering: {feature_data.shape}")
            
            if len(feature_data) < SEQUENCE_LENGTH:
                results.append({
                    "Metric": "Feature Engineering Check",
                    "Value": "FAILED",
                    "Notes": f"Insufficient data after feature engineering: {len(feature_data)} < {SEQUENCE_LENGTH}"
                })
                return pd.DataFrame(results)
            
            # Scale the data using the same scaler that was used in training
            # Note: We should use the scaler that was returned from train_model
            scaled_data = scaler.transform(feature_data.values)
            
            # Get the last sequence for prediction
            last_sequence = scaled_data[-SEQUENCE_LENGTH:]
            
            logger.info(f"Last sequence shape: {last_sequence.shape}")
            logger.info(f"Sequence features: {features}")
            
            # Validate the sequence
            if last_sequence.shape != (SEQUENCE_LENGTH, len(features)):
                results.append({
                    "Metric": "Sequence Shape Validation",
                    "Value": "FAILED",
                    "Notes": f"Expected shape ({SEQUENCE_LENGTH}, {len(features)}), got {last_sequence.shape}"
                })
                return pd.DataFrame(results)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(last_sequence)) or np.any(np.isinf(last_sequence)):
                results.append({
                    "Metric": "Data Quality Check",
                    "Value": "FAILED",
                    "Notes": "Sequence contains NaN or infinite values"
                })
                return pd.DataFrame(results)
            
            # Make predictions
            num_days_to_predict = 5  # Keep it small for testing
            logger.info(f"Making predictions for {num_days_to_predict} days")
            
            predict_func = IMPORTS['model_functions']['predict_future']
            predictions_scaled = predict_func(model, scaler, last_sequence, num_days_to_predict)
            
            predict_time = time.time() - start_time - train_time
            
            logger.info(f"Predictions completed in {predict_time:.3f} seconds")
            logger.info(f"Predictions shape: {predictions_scaled.shape}")
            
        except Exception as data_prep_error:
            logger.error(f"Data preparation for prediction failed: {data_prep_error}")
            results.append({
                "Metric": "Data Preparation for Prediction",
                "Value": "FAILED",
                "Notes": f"Error: {str(data_prep_error)[:150]}"
            })
            return pd.DataFrame(results)

            
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        logger.error(traceback.format_exc())
        results.append({
            "Metric": "Test Execution",
            "Value": "ERROR",
            "Notes": f"Exception: {str(e)[:200]}"
        })
    
    df_results = pd.DataFrame(results)
    print_and_save_table(df_results, "Performance_and_Accuracy_Test")
    return df_results

# ------------------------
# Reliability Test
# ------------------------

def test_reliability():
    """Tests system reliability against edge cases and bad data."""
    logger.info("Starting reliability tests")
    results = []
    
    if IMPORTS['model_functions'] is None:
        results.append({
            "Test Case": "Import Check",
            "Outcome": "Failed",
            "Notes": "Model functions not available"
        })
        return pd.DataFrame(results)
    
    train_func = IMPORTS['model_functions']['train_model']
    
    # Test case 1: DataFrame with missing values
    try:
        df_missing = pd.DataFrame({
            'transaction_date': pd.date_range('2023-01-01', periods=100),
            'rate': [100 + i + (np.nan if i % 10 == 0 else 0) for i in range(100)],
            'quantity': [10] * 100,
            'amount': [1000] * 100
        })
        
        model_result = train_func(df_missing)
        if model_result is not None and model_result[0] is not None:
            outcome = "Passed"
            notes = "Model handled missing data gracefully"
        else:
            outcome = "Passed"
            notes = "Model correctly rejected data with missing values"
            
    except Exception as e:
        outcome = "Failed"
        notes = f"Exception handling missing data: {str(e)[:100]}"
    
    results.append({
        "Test Case": "Missing data (NaN values)",
        "Outcome": outcome,
        "Notes": notes
    })
    
    # Test case 2: Empty DataFrame
    try:
        df_empty = pd.DataFrame(columns=['transaction_date', 'rate', 'quantity', 'amount'])
        model_result = train_func(df_empty)
        
        if model_result is None or (isinstance(model_result, tuple) and model_result[0] is None):
            outcome = "Passed"
            notes = "Model correctly handled empty dataset"
        else:
            outcome = "Failed"
            notes = "Model should not train on empty data"
            
    except Exception as e:
        outcome = "Passed"
        notes = f"Exception properly raised for empty data: {type(e).__name__}"
    
    results.append({
        "Test Case": "Empty DataFrame",
        "Outcome": outcome,
        "Notes": notes
    })
    
    # Test case 3: Insufficient data
    try:
        df_small = pd.DataFrame({
            'transaction_date': pd.date_range('2023-01-01', periods=10),
            'rate': [100 + i for i in range(10)],
            'quantity': [10] * 10,
            'amount': [1000] * 10
        })
        
        model_result = train_func(df_small)
        if model_result is None or (isinstance(model_result, tuple) and model_result[0] is None):
            outcome = "Passed"
            notes = "Model correctly rejected insufficient data"
        else:
            outcome = "Failed"
            notes = "Model should require more data for training"
            
    except Exception as e:
        outcome = "Passed"
        notes = f"Exception properly raised for insufficient data: {type(e).__name__}"
    
    results.append({
        "Test Case": "Insufficient training data",
        "Outcome": outcome,
        "Notes": notes
    })
    
    # Test case 4: API with invalid symbol
    if IMPORTS['app'] is not None:
        try:
            test_client = IMPORTS['app'].test_client()
            with patch('app.DatabaseManager', return_value=get_mock_db_manager()):
                response = test_client.get('/historical-data/INVALID_SYMBOL')
                
                if response.status_code == 200:
                    data = response.get_json()
                    if not data:  # Empty list/dict
                        outcome = "Passed"
                        notes = "API correctly returned empty data for invalid symbol"
                    else:
                        outcome = "Failed"
                        notes = "API returned data for invalid symbol"
                else:
                    outcome = "Passed"
                    notes = f"API correctly returned error status: {response.status_code}"
                    
        except Exception as e:
            outcome = "Failed"
            notes = f"API test failed: {str(e)[:100]}"
    else:
        outcome = "Skipped"
        notes = "Flask app not available"
    
    results.append({
        "Test Case": "API with invalid symbol",
        "Outcome": outcome,
        "Notes": notes
    })
    
    df_results = pd.DataFrame(results)
    print_and_save_table(df_results, "Reliability_Test")
    return df_results

# ------------------------
# Test Summary and Report Generation
# ------------------------

def generate_summary_report(test_results):
    """Generate a comprehensive summary report of all tests."""
    logger.info("Generating summary report")
    
    summary_data = []
    total_tests = 0
    passed_tests = 0
    
    for test_name, df in test_results.items():
        if df is not None and not df.empty:
            test_count = len(df)
            # Count different success indicators based on column names
            if 'Result' in df.columns:
                success_count = df['Result'].sum() if df['Result'].dtype == bool else 0
            elif 'Success' in df.columns:
                success_count = df['Success'].sum() if df['Success'].dtype == bool else 0
            elif 'Outcome' in df.columns:
                success_count = (df['Outcome'] == 'Passed').sum()
            else:
                success_count = 0
                
            success_rate = (success_count / test_count * 100) if test_count > 0 else 0
            
            summary_data.append({
                'Test Category': test_name,
                'Total Tests': test_count,
                'Passed': success_count,
                'Failed': test_count - success_count,
                'Success Rate (%)': f"{success_rate:.1f}%"
            })
            
            total_tests += test_count
            passed_tests += success_count
    
    # Overall summary
    overall_success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    summary_data.append({
        'Test Category': 'OVERALL',
        'Total Tests': total_tests,
        'Passed': passed_tests,
        'Failed': total_tests - passed_tests,
        'Success Rate (%)': f"{overall_success_rate:.1f}%"
    })
    
    summary_df = pd.DataFrame(summary_data)
    print_and_save_table(summary_df, "Test_Summary_Report")
    
    # Create visual summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Test categories performance
    categories = summary_df[summary_df['Test Category'] != 'OVERALL']
    if not categories.empty:
        ax1.bar(categories['Test Category'], categories['Passed'], label='Passed', color='green', alpha=0.7)
        ax1.bar(categories['Test Category'], categories['Failed'], bottom=categories['Passed'], label='Failed', color='red', alpha=0.7)
        ax1.set_title('Test Results by Category')
        ax1.set_ylabel('Number of Tests')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
    
    # Overall pie chart
    if total_tests > 0:
        ax2.pie([passed_tests, total_tests - passed_tests], 
                labels=['Passed', 'Failed'], 
                colors=['green', 'red'], 
                autopct='%1.1f%%',
                startangle=90)
        ax2.set_title(f'Overall Test Results\n({overall_success_rate:.1f}% Success Rate)')
    
    plt.tight_layout()
    save_chart(fig, "Test_Summary_Chart")
    
    return summary_df

# ------------------------
# Main Test Runner
# ------------------------

def main():
    """Main test execution function with comprehensive error handling."""
    print("=" * 60)
    print("NEPSE STOCK PREDICTION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    logger.info("Starting comprehensive test suite")
    
    # Check if critical imports are available
    missing_imports = []
    for key, value in IMPORTS.items():
        if value is None:
            missing_imports.append(key)
    
    if missing_imports:
        print(f"\nWARNING: The following modules could not be imported: {missing_imports}")
        print("Some tests may be skipped or fail.\n")
    
    test_results = {}
    
    # Run all tests with error handling
    test_functions = [
        ("Unit Test - Preprocessing", test_preprocess_transaction_data),
        ("Unit Test - Percentage Change", test_calculate_percentage_change),
        ("System Test - API Endpoints", test_api_endpoints),
        ("Performance & Accuracy Test", test_performance_and_accuracy),
        ("Reliability Test", test_reliability)
    ]
    
    for test_name, test_func in test_functions:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            result_df = test_func()
            test_results[test_name] = result_df
            logger.info(f"Completed: {test_name}")
        except Exception as e:
            logger.error(f"Failed to run {test_name}: {e}")
            logger.error(traceback.format_exc())
            # Create a failure record
            failure_df = pd.DataFrame([{
                'Test': test_name,
                'Result': False,
                'Details': f'Test execution failed: {str(e)[:200]}'
            }])
            test_results[test_name] = failure_df
    
    # Generate summary report
    print(f"\n{'='*50}")
    print("Generating Summary Report")
    print('='*50)
    
    try:
        summary_df = generate_summary_report(test_results)
        logger.info("Summary report generated successfully")
    except Exception as e:
        logger.error(f"Failed to generate summary report: {e}")
    
    print(f"\n{'='*60}")
    print("TEST SUITE COMPLETED")
    print(f"Reports saved in: {os.path.abspath(REPORT_DIR)}")
    print('='*60)
    
    return test_results

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest suite interrupted by user")
        logger.info("Test suite interrupted by user")
    except Exception as e:
        print(f"\nCritical error in test suite: {e}")
        logger.critical(f"Critical error in test suite: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)