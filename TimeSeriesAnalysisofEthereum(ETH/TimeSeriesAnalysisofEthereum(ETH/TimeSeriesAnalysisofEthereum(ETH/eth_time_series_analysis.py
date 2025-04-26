"""
Time Series Analysis of Ethereum (ETH/USDT) Market Projections using ARIMA

This script performs a comprehensive time series analysis on Ethereum price data,
including data collection, exploratory data analysis, stationarity testing,
ARIMA model development, evaluation, and forecasting.

Author: [Your Name]
Date: April 26, 2025

Note: This analysis is for educational purposes only and should not be used
for financial advice or investment decisions. Cryptocurrency markets are highly
volatile and unpredictable.
"""

# TODO: Clean up some of these imports later if I have time
# Some of these might not be needed for the final version

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
import os
from datetime import datetime, timedelta

# Set plot style and suppress warnings
plt.style.use('seaborn-v0_8-darkgrid')
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 1. Data Collection & Preparation
def collect_data():
    """
    Collect historical Ethereum price data from Yahoo Finance.
    """
    print("Collecting Ethereum price data from Yahoo Finance...")
    
    # Define date range (2 years of historical data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    # Download data from Yahoo Finance
    eth_data = yf.download('ETH-USD', start=start_date, end=end_date)
    
    # Save raw data
    eth_data.to_csv('results/eth_raw_data.csv')
    
    print(f"Data collected from {eth_data.index.min().date()} to {eth_data.index.max().date()}")
    print(f"Total data points: {len(eth_data)}")
    
    return eth_data

def preprocess_data(data):
    """
    Preprocess the collected data for analysis.
    """
    print("\nPreprocessing data...")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    print(f"Missing values in each column:\n{missing_values}")
    
    # Fill missing values if any
    if missing_values.sum() > 0:
        data = data.fillna(method='ffill')
        print("Missing values filled using forward fill method.")
    
    # Focus on 'Close' price for time series analysis
    eth_close = data[['Close']].copy()
    eth_close.rename(columns={'Close': 'Price'}, inplace=True)
    
    # Calculate daily returns
    eth_close['Returns'] = eth_close['Price'].pct_change() * 100
    
    # Calculate rolling statistics
    eth_close['MA7'] = eth_close['Price'].rolling(window=7).mean()
    eth_close['MA30'] = eth_close['Price'].rolling(window=30).mean()
    eth_close['Volatility'] = eth_close['Returns'].rolling(window=30).std()
    
    # Save processed data
    eth_close.to_csv('results/eth_processed_data.csv')
    
    print("Data preprocessing completed.")
    
    return eth_close

# 2. Exploratory Data Analysis (EDA)
def perform_eda(data):
    """
    Perform exploratory data analysis on the preprocessed data.
    """
    print("\nPerforming Exploratory Data Analysis...")
    
    # Statistical summary
    stats_summary = data['Price'].describe()
    print(f"Statistical summary of ETH price:\n{stats_summary}")
    
    # Plot price trends
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Price'], label='ETH Price')
    plt.plot(data.index, data['MA7'], label='7-Day MA', linestyle='--')
    plt.plot(data.index, data['MA30'], label='30-Day MA', linestyle='-.')
    plt.title('Ethereum Price Trend with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)  # I like having grids on my plots - makes it easier to read
    plt.tight_layout()
    plt.savefig('plots/eth_price_trend.png')
    
    # Plot daily returns
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Returns'], color='blue')
    plt.title('Ethereum Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns (%)')
    plt.tight_layout()
    plt.savefig('plots/eth_daily_returns.png')
    
    # Plot volatility
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Volatility'], color='red')
    plt.title('Ethereum 30-Day Rolling Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Std Dev of Returns)')
    plt.tight_layout()
    plt.savefig('plots/eth_volatility.png')
    
    # Distribution of returns
    plt.figure(figsize=(14, 7))
    sns.histplot(data['Returns'].dropna(), kde=True)
    plt.title('Distribution of Ethereum Daily Returns')
    plt.xlabel('Returns (%)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('plots/eth_returns_distribution.png')
    
    # Monthly average prices
    monthly_avg = data['Price'].resample('M').mean()
    plt.figure(figsize=(14, 7))
    plt.plot(monthly_avg.index, monthly_avg.values)
    plt.title('Ethereum Monthly Average Price')
    plt.xlabel('Date')
    plt.ylabel('Average Price (USD)')
    plt.tight_layout()
    plt.savefig('plots/eth_monthly_avg.png')
    
    print("EDA completed and plots saved.")
    
    # Return insights from EDA
    return {
        'mean_price': stats_summary.loc['mean'],
        'max_price': stats_summary.loc['max'],
        'min_price': stats_summary.loc['min'],
        'volatility': data['Volatility'].mean(),
        'avg_return': data['Returns'].mean()
    }

# 3. Stationarity Testing
def test_stationarity(data):
    """
    Test the stationarity of the time series using ADF test.
    
    I'm using the Augmented Dickey-Fuller test here since it's the most common
    approach for testing stationarity in time series. The null hypothesis is that
    the series has a unit root (is non-stationary).
    """
    print("\nTesting stationarity of the time series...")
    
    # Test on raw price data
    price_series = data['Price'].dropna()
    price_adf_result = adfuller(price_series)
    
    print(f"ADF Statistic (Price): {price_adf_result[0]}")
    print(f"p-value (Price): {price_adf_result[1]}")
    print(f"Critical Values (Price):")
    for key, value in price_adf_result[4].items():
        print(f"\t{key}: {value}")
    
    # Plot the price series
    plt.figure(figsize=(14, 7))
    plt.plot(price_series)
    plt.title('Ethereum Price Series')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.tight_layout()
    plt.savefig('plots/eth_price_series.png')
    
    # Test on differenced price data
    diff_series = price_series.diff().dropna()
    diff_adf_result = adfuller(diff_series)
    
    print(f"\nADF Statistic (Differenced Price): {diff_adf_result[0]}")
    print(f"p-value (Differenced Price): {diff_adf_result[1]}")
    print(f"Critical Values (Differenced Price):")
    for key, value in diff_adf_result[4].items():
        print(f"\t{key}: {value}")
    
    # Plot the differenced series
    plt.figure(figsize=(14, 7))
    plt.plot(diff_series)
    plt.title('Differenced Ethereum Price Series')
    plt.xlabel('Date')
    plt.ylabel('Price Change (USD)')
    plt.tight_layout()
    plt.savefig('plots/eth_diff_series.png')
    
    # Determine if differencing is required
    is_stationary = price_adf_result[1] < 0.05
    diff_is_stationary = diff_adf_result[1] < 0.05
    
    if is_stationary:
        print("\nThe original price series is stationary. No differencing required.")
        d_value = 0
    else:
        if diff_is_stationary:
            print("\nThe original price series is non-stationary, but the differenced series is stationary.")
            print("First-order differencing (d=1) is required.")
            d_value = 1
        else:
            print("\nBoth original and first-differenced series are non-stationary.")
            print("Higher-order differencing may be required.")
            d_value = 2
    
    return {
        'price_adf_statistic': price_adf_result[0],
        'price_p_value': price_adf_result[1],
        'diff_adf_statistic': diff_adf_result[0],
        'diff_p_value': diff_adf_result[1],
        'd_value': d_value,
        'price_is_stationary': is_stationary,
        'diff_is_stationary': diff_is_stationary
    }

# 4. ARIMA Model Development
def develop_arima_model(data, d_value):
    """
    Develop ARIMA models for price forecasting.
    
    This was the most time-consuming part of the analysis. Had to try different
    combinations of p and q values to find the best model. I'm limiting to 0-4
    for each parameter to keep computation time reasonable.
    """
    print("\nDeveloping ARIMA models...")
    
    # Prepare data
    price_series = data['Price'].dropna()
    
    # Create training and testing sets (80-20 split)
    train_size = int(len(price_series) * 0.8)
    train, test = price_series[:train_size], price_series[train_size:]
    
    print(f"Training data: {len(train)} observations")
    print(f"Testing data: {len(test)} observations")
    
    # Plot ACF and PACF to determine p and q values
    plt.figure(figsize=(14, 7))
    plt.subplot(211)
    plot_acf(price_series.diff().dropna() if d_value > 0 else price_series, ax=plt.gca(), lags=40)
    plt.subplot(212)
    plot_pacf(price_series.diff().dropna() if d_value > 0 else price_series, ax=plt.gca(), lags=40)
    plt.tight_layout()
    plt.savefig('plots/eth_acf_pacf.png')
    
    # Try different combinations of p and q values
    p_values = range(0, 5)
    q_values = range(0, 5)
    
    best_aic = float('inf')
    best_order = None
    best_model = None
    
    results = []
    
    for p in p_values:
        for q in q_values:
            try:
                model = ARIMA(train, order=(p, d_value, q))
                model_fit = model.fit()
                aic = model_fit.aic
                results.append((p, d_value, q, aic))
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d_value, q)
                    best_model = model_fit
                
                print(f"ARIMA({p},{d_value},{q}) - AIC: {aic}")
            except:
                continue
    
    # Save results of all models
    results_df = pd.DataFrame(results, columns=['p', 'd', 'q', 'AIC'])
    results_df = results_df.sort_values('AIC')
    results_df.to_csv('results/arima_model_results.csv', index=False)
    
    print(f"\nBest ARIMA model: ARIMA{best_order} with AIC: {best_aic}")
    
    # Save the best model summary
    with open('results/best_model_summary.txt', 'w') as f:
        f.write(str(best_model.summary()))
    
    return best_model, best_order, train, test

# 5. Model Evaluation
def evaluate_model(model, train, test, best_order):
    """
    Evaluate the ARIMA model performance.
    """
    print("\nEvaluating model performance...")
    
    # Forecast on test data
    forecast = model.forecast(steps=len(test))
    
    # Calculate error metrics
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = mean_absolute_percentage_error(test, forecast) * 100
    
    print(f"Root Mean Square Error (RMSE): {rmse}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
    
    # Plot actual vs predicted
    plt.figure(figsize=(14, 7))
    plt.plot(test.index, test, label='Actual')
    plt.plot(test.index, forecast, label='Predicted', color='red')
    plt.title(f'Ethereum Price - Actual vs Predicted (ARIMA{best_order})')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/eth_actual_vs_predicted.png')
    
    # Plot residuals
    residuals = test - forecast
    plt.figure(figsize=(14, 7))
    plt.plot(test.index, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Residuals of ARIMA Model')
    plt.xlabel('Date')
    plt.ylabel('Residual Value')
    plt.tight_layout()
    plt.savefig('plots/eth_residuals.png')
    
    # Histogram of residuals
    plt.figure(figsize=(14, 7))
    # Convert residuals to numpy array and check if it has valid values
    residuals_array = residuals.values
    if len(residuals_array) > 1 and not np.isnan(residuals_array).all():
        plt.hist(residuals_array, bins=20, alpha=0.7)
        # Only use kde if we have enough non-nan values
        if np.sum(~np.isnan(residuals_array)) > 5:
            try:
                sns.kdeplot(residuals_array, color='red')
            except:
                pass  # Silently fail if kde still doesn't work
    else:
        plt.text(0.5, 0.5, "Insufficient data for histogram", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('plots/eth_residuals_distribution.png')
    
    return {
        'rmse': rmse,
        'mape': mape,
        'forecast': forecast,
        'residuals': residuals
    }

# 6. Forecasting & Visualization
def generate_forecast(model, data, best_order):
    """
    Generate future price projections for Ethereum.
    """
    print("\nGenerating future price projections...")
    
    # Forecast for the next 30 days
    forecast_steps = 30
    forecast = model.forecast(steps=forecast_steps)
    
    # Create date range for forecast
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Forecasted_Price': forecast
    }, index=forecast_dates)
    
    # Save forecast results
    forecast_df.to_csv('results/eth_forecast.csv')
    
    # Plot the forecast
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[-90:], data['Price'][-90:], label='Historical')
    plt.plot(forecast_df.index, forecast_df['Forecasted_Price'], label='Forecast', color='red')
    
    # Add confidence intervals (assuming normal distribution of errors)
    std_error = np.std(model.resid)
    plt.fill_between(
        forecast_df.index,
        forecast_df['Forecasted_Price'] - 1.96 * std_error,
        forecast_df['Forecasted_Price'] + 1.96 * std_error,
        color='pink', alpha=0.3
    )
    
    plt.title(f'Ethereum 30-Day Price Forecast (ARIMA{best_order})')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/eth_forecast.png')
    
    print(f"Forecast generated for the next {forecast_steps} days.")
    
    # Calculate forecast statistics
    forecast_stats = {
        'start_price': forecast_df['Forecasted_Price'].iloc[0],
        'end_price': forecast_df['Forecasted_Price'].iloc[-1],
        'min_price': forecast_df['Forecasted_Price'].min(),
        'max_price': forecast_df['Forecasted_Price'].max(),
        'price_change': (forecast_df['Forecasted_Price'].iloc[-1] - forecast_df['Forecasted_Price'].iloc[0]) / forecast_df['Forecasted_Price'].iloc[0] * 100
    }
    
    print(f"Forecasted price change over 30 days: {forecast_stats['price_change']:.2f}%")
    
    return forecast_df, forecast_stats

# 7. Generate Report
def generate_report(eda_insights, stationarity_results, best_order, eval_metrics, forecast_stats):
    """
    Generate a comprehensive report of the analysis.
    """
    print("\nGenerating final report...")
    
    # Convert any Series objects to float values
    mean_price = float(eda_insights['mean_price']) if hasattr(eda_insights['mean_price'], 'iloc') else eda_insights['mean_price']
    max_price = float(eda_insights['max_price']) if hasattr(eda_insights['max_price'], 'iloc') else eda_insights['max_price']
    min_price = float(eda_insights['min_price']) if hasattr(eda_insights['min_price'], 'iloc') else eda_insights['min_price']
    avg_return = float(eda_insights['avg_return']) if hasattr(ed
(Content truncated due to size limit. Use line ranges to read in chunks)