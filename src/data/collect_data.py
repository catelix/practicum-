"""
Data collection module for fetching market data using yfinance.
"""

import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Constants
DEFAULT_TICKERS = ['SPY', 'QQQ', 'AAPL', 'MSFT']
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')

def fetch_market_data(
    tickers: List[str] = DEFAULT_TICKERS,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    interval: str = '1d'
) -> pd.DataFrame:
    """
    Fetch market data for specified tickers using yfinance.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval (e.g., '1d' for daily)
    
    Returns:
        DataFrame containing OHLCV data for all tickers
    """
    all_data = []
    
    for ticker in tqdm(tickers, desc="Fetching market data"):
        try:
            # Download data
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            
            # Flatten columns if multi-index
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join([str(i) for i in col if i]).strip('_') for col in data.columns.values]
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Select only the columns for this ticker
            cols = ['Date', f'Open_{ticker}', f'High_{ticker}', f'Low_{ticker}', f'Close_{ticker}', f'Volume_{ticker}']
            data = data[cols]
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            data['Ticker'] = ticker
            
            all_data.append(data)
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            continue
    
    # Combine all data
    if not all_data:
        raise ValueError("No data was successfully fetched")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Ensure column names are consistent
    # Convert column names to strings and title case
    combined_data.columns = [str(col).title() for col in combined_data.columns]
    
    return combined_data

def save_market_data(
    data: pd.DataFrame,
    filename: Optional[str] = None
) -> str:
    """
    Save market data to CSV file.
    
    Args:
        data: DataFrame containing market data
        filename: Optional custom filename
    
    Returns:
        Path to saved file
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'market_data_{timestamp}.csv'
    
    filepath = os.path.join(DATA_DIR, filename)
    data.to_csv(filepath, index=False)
    
    return filepath

def main():
    """Main function to fetch and save market data."""
    try:
        # Fetch data
        print("Fetching market data...")
        market_data = fetch_market_data()
        
        # Save data
        print("Saving market data...")
        filepath = save_market_data(market_data)
        print(f"Data saved to: {filepath}")
        
        # Print summary
        print("\nData Summary:")
        print(f"Total records: {len(market_data)}")
        print("\nTicker counts:")
        print(market_data['Ticker'].value_counts())
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 