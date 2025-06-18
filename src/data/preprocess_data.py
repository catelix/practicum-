"""
Data preprocessing module for cleaning and normalizing market data.
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Constants
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed')
PRICE_COLUMNS = ['Open', 'High', 'Low', 'Close']
VOLUME_COLUMN = 'Volume'

def load_market_data(filepath: str) -> pd.DataFrame:
    """
    Load market data from CSV file.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        DataFrame containing market data
    """
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def detect_outliers(
    data: pd.DataFrame,
    columns: list = PRICE_COLUMNS,
    iqr_multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Detect outliers using IQR method.
    
    Args:
        data: DataFrame containing market data
        columns: List of columns to check for outliers
        iqr_multiplier: Multiplier for IQR to determine outlier bounds
    
    Returns:
        DataFrame with outlier information
    """
    outlier_info = pd.DataFrame()
    
    for ticker in data['Ticker'].unique():
        ticker_data = data[data['Ticker'] == ticker]
        
        for col in columns:
            Q1 = ticker_data[col].quantile(0.25)
            Q3 = ticker_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            outliers = ticker_data[
                (ticker_data[col] < lower_bound) |
                (ticker_data[col] > upper_bound)
            ]
            
            if not outliers.empty:
                outlier_info = pd.concat([
                    outlier_info,
                    pd.DataFrame({
                        'Ticker': ticker,
                        'Column': col,
                        'Date': outliers['Date'],
                        'Value': outliers[col],
                        'Lower_Bound': lower_bound,
                        'Upper_Bound': upper_bound
                    })
                ])
    
    return outlier_info

def interpolate_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate missing values in the dataset.
    
    Args:
        data: DataFrame containing market data
    
    Returns:
        DataFrame with interpolated values
    """
    # Sort by Date and Ticker
    data = data.sort_values(['Date', 'Ticker'])
    
    # Group by Ticker and interpolate
    interpolated_data = []
    for ticker in data['Ticker'].unique():
        ticker_data = data[data['Ticker'] == ticker].copy()
        
        # Set Date as index for time interpolation
        ticker_data = ticker_data.set_index('Date')
        
        # Interpolate numeric columns
        numeric_cols = ticker_data.select_dtypes(include=[np.number]).columns
        ticker_data[numeric_cols] = ticker_data[numeric_cols].interpolate(method='time')
        
        # Forward fill any remaining NaN values
        ticker_data = ticker_data.fillna(method='ffill')
        
        # Reset index to bring Date back as a column
        ticker_data = ticker_data.reset_index()
        
        interpolated_data.append(ticker_data)
    
    return pd.concat(interpolated_data, ignore_index=True)

def normalize_data(
    data: pd.DataFrame,
    columns: list = PRICE_COLUMNS + [VOLUME_COLUMN]
) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: DataFrame containing market data
        columns: List of columns to normalize
    
    Returns:
        Tuple of (normalized DataFrame, scaler dictionary)
    """
    normalized_data = data.copy()
    scalers = {}
    
    for ticker in data['Ticker'].unique():
        ticker_data = data[data['Ticker'] == ticker]
        
        for col in columns:
            scaler = MinMaxScaler()
            values = ticker_data[col].values.reshape(-1, 1)
            
            # Fit and transform
            normalized_values = scaler.fit_transform(values)
            
            # Store scaler
            scalers[f"{ticker}_{col}"] = scaler
            
            # Update values in normalized data
            mask = normalized_data['Ticker'] == ticker
            normalized_data.loc[mask, f"{col}_Normalized"] = normalized_values
    
    return normalized_data, scalers

def save_processed_data(
    data: pd.DataFrame,
    filename: Optional[str] = None
) -> str:
    """
    Save processed data to CSV file.
    
    Args:
        data: DataFrame containing processed market data
        filename: Optional custom filename
    
    Returns:
        Path to saved file
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    if filename is None:
        filename = 'processed_market_data.csv'
    
    filepath = os.path.join(PROCESSED_DATA_DIR, filename)
    data.to_csv(filepath, index=False)
    
    return filepath

def main():
    """Main function to preprocess market data."""
    try:
        # Find the most recent raw data file
        raw_data_dir = os.path.join(os.path.dirname(PROCESSED_DATA_DIR), 'raw')
        raw_files = [f for f in os.listdir(raw_data_dir) if f.startswith('market_data_')]
        if not raw_files:
            raise FileNotFoundError("No raw market data files found")
        
        latest_file = max(raw_files)
        filepath = os.path.join(raw_data_dir, latest_file)
        
        # Load data
        print("Loading market data...")
        data = load_market_data(filepath)
        
        # Detect outliers
        print("Detecting outliers...")
        outliers = detect_outliers(data)
        if not outliers.empty:
            print("\nOutlier Summary:")
            print(outliers.groupby(['Ticker', 'Column']).size())
        
        # Interpolate missing values
        print("\nInterpolating missing values...")
        data = interpolate_missing_values(data)
        
        # Normalize data
        print("Normalizing data...")
        normalized_data, scalers = normalize_data(data)
        
        # Save processed data
        print("Saving processed data...")
        filepath = save_processed_data(normalized_data)
        print(f"Processed data saved to: {filepath}")
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Total records: {len(normalized_data)}")
        print(f"Number of tickers: {normalized_data['Ticker'].nunique()}")
        print("\nNormalized columns:")
        print([col for col in normalized_data.columns if col.endswith('_Normalized')])
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 