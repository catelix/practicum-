"""
Portfolio management module for handling portfolio operations and simulations.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Constants
INITIAL_ALLOCATION = {
    'SPY': 0.40,
    'QQQ': 0.30,
    'AAPL': 0.15,
    'MSFT': 0.15
}
LSTM_THRESHOLD = 0.02  # 2% threshold for LSTM signals
REBALANCE_FREQUENCY = 'W'  # Weekly rebalancing

class PortfolioManager:
    def __init__(
        self,
        initial_allocation: Dict[str, float] = INITIAL_ALLOCATION,
        initial_capital: float = 100000.0
    ):
        """
        Initialize portfolio manager.
        
        Args:
            initial_allocation: Dictionary of initial asset allocations
            initial_capital: Initial portfolio value
        """
        self.initial_allocation = initial_allocation
        self.initial_capital = initial_capital
        self.current_allocation = initial_allocation.copy()
        self.portfolio_value = initial_capital
        self.positions = {}
        self.transaction_history = []
        
    def calculate_position_sizes(
        self,
        prices: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Calculate position sizes based on current allocation and prices.
        
        Args:
            prices: Dictionary of current prices for each asset
        
        Returns:
            Dictionary of position sizes (number of shares)
        """
        position_sizes = {}
        
        for asset, allocation in self.current_allocation.items():
            if asset in prices:
                position_value = self.portfolio_value * allocation
                position_sizes[asset] = int(position_value / prices[asset])
        
        return position_sizes
    
    def rebalance_portfolio(
        self,
        date: datetime,
        prices: Dict[str, float],
        lstm_signals: Optional[Dict[str, float]] = None,
        hmm_states: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        Rebalance portfolio based on signals and states.
        
        Args:
            date: Current date
            prices: Dictionary of current prices
            lstm_signals: Dictionary of LSTM predictions
            hmm_states: Dictionary of HMM volatility states
        
        Returns:
            List of transactions
        """
        transactions = []
        
        # Adjust allocation based on signals
        target_allocation = self.current_allocation.copy()
        
        if lstm_signals is not None:
            for asset, signal in lstm_signals.items():
                if asset in target_allocation:
                    if signal >= LSTM_THRESHOLD:
                        # Increase allocation for positive signals
                        target_allocation[asset] *= 1.2
                    elif signal <= -LSTM_THRESHOLD:
                        # Decrease allocation for negative signals
                        target_allocation[asset] *= 0.8
        
        if hmm_states is not None:
            for asset, state in hmm_states.items():
                if asset in target_allocation:
                    if state == 'High':
                        # Reduce exposure in high volatility
                        target_allocation[asset] *= 0.7
                    elif state == 'Low':
                        # Increase exposure in low volatility
                        target_allocation[asset] *= 1.1
        
        # Normalize allocations
        total_allocation = sum(target_allocation.values())
        target_allocation = {
            asset: allocation / total_allocation
            for asset, allocation in target_allocation.items()
        }
        
        # Calculate target positions
        target_positions = self.calculate_position_sizes(prices)
        
        # Execute trades
        for asset, target_size in target_positions.items():
            current_size = self.positions.get(asset, 0)
            
            if target_size != current_size:
                # Calculate trade
                trade_size = target_size - current_size
                trade_value = trade_size * prices[asset]
                
                # Record transaction
                transaction = {
                    'Date': date,
                    'Asset': asset,
                    'Action': 'BUY' if trade_size > 0 else 'SELL',
                    'Shares': abs(trade_size),
                    'Price': prices[asset],
                    'Value': abs(trade_value),
                    'LSTM_Signal': lstm_signals.get(asset) if lstm_signals else None,
                    'HMM_State': hmm_states.get(asset) if hmm_states else None
                }
                transactions.append(transaction)
                
                # Update positions
                self.positions[asset] = target_size
        
        # Update current allocation
        self.current_allocation = target_allocation
        
        # Update transaction history
        self.transaction_history.extend(transactions)
        
        return transactions
    
    def calculate_portfolio_value(
        self,
        date: datetime,
        prices: Dict[str, float]
    ) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            date: Current date
            prices: Dictionary of current prices
        
        Returns:
            Current portfolio value
        """
        portfolio_value = 0.0
        
        for asset, shares in self.positions.items():
            if asset in prices:
                portfolio_value += shares * prices[asset]
        
        self.portfolio_value = portfolio_value
        return portfolio_value
    
    def get_portfolio_metrics(
        self,
        daily_values: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            daily_values: DataFrame with daily portfolio values
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate returns
        daily_values['Returns'] = daily_values['Portfolio_Value'].pct_change()
        
        # Calculate metrics
        total_return = (daily_values['Portfolio_Value'].iloc[-1] / daily_values['Portfolio_Value'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(daily_values)) - 1
        volatility = daily_values['Returns'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        
        # Calculate maximum drawdown
        daily_values['Cumulative_Max'] = daily_values['Portfolio_Value'].cummax()
        daily_values['Drawdown'] = (daily_values['Portfolio_Value'] - daily_values['Cumulative_Max']) / daily_values['Cumulative_Max']
        max_drawdown = daily_values['Drawdown'].min()
        
        return {
            'Total_Return': total_return,
            'Annualized_Return': annualized_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown
        }

def simulate_portfolio(
    data: pd.DataFrame,
    lstm_predictions: Dict[str, pd.DataFrame],
    hmm_predictions: Dict[str, pd.DataFrame],
    initial_capital: float = 100000.0
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simulate portfolio performance.
    
    Args:
        data: DataFrame with market data
        lstm_predictions: Dictionary of LSTM predictions for each asset
        hmm_predictions: Dictionary of HMM predictions for each asset
        initial_capital: Initial portfolio value
    
    Returns:
        Tuple of (daily portfolio values, performance metrics)
    """
    # Initialize portfolio manager
    manager = PortfolioManager(initial_capital=initial_capital)
    
    # Prepare date range
    dates = pd.date_range(
        start=data['Date'].min(),
        end=data['Date'].max(),
        freq='D'
    )
    
    # Initialize results
    daily_values = []
    
    # Get initial prices for position setup
    initial_data = data[data['Date'] == dates[0]]
    if not initial_data.empty:
        initial_prices = dict(zip(initial_data['Ticker'], initial_data['Close']))
        # Set up initial positions
        manager.positions = manager.calculate_position_sizes(initial_prices)
        # Calculate initial portfolio value
        manager.portfolio_value = manager.calculate_portfolio_value(dates[0], initial_prices)
    
    # Simulate portfolio
    for date in dates:
        # Get current prices
        current_data = data[data['Date'] == date]
        if current_data.empty:
            continue
        
        prices = dict(zip(current_data['Ticker'], current_data['Close']))
        
        # Get signals
        lstm_signals = {}
        hmm_states = {}
        
        for asset in manager.initial_allocation.keys():
            # Get LSTM signal
            if asset in lstm_predictions:
                asset_pred = lstm_predictions[asset]
                pred = asset_pred[asset_pred['Date'] == date]
                if not pred.empty:
                    lstm_signals[asset] = pred['Predicted_Close_Normalized'].iloc[0]
            
            # Get HMM state
            if asset in hmm_predictions:
                asset_pred = hmm_predictions[asset]
                pred = asset_pred[asset_pred['Date'] == date]
                if not pred.empty:
                    hmm_states[asset] = pred['Volatility_Level'].iloc[0]
        
        # Rebalance portfolio (weekly)
        if date.weekday() == 0:  # Monday
            manager.rebalance_portfolio(date, prices, lstm_signals, hmm_states)
        
        # Calculate portfolio value
        portfolio_value = manager.calculate_portfolio_value(date, prices)
        
        # Record daily value
        daily_values.append({
            'Date': date,
            'Portfolio_Value': portfolio_value
        })
    
    # Convert to DataFrame
    daily_values_df = pd.DataFrame(daily_values)
    
    # Calculate metrics
    metrics = manager.get_portfolio_metrics(daily_values_df)
    
    return daily_values_df, metrics

def main():
    """Main function to run portfolio simulation."""
    try:
        # Load data
        processed_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed')
        data_file = os.path.join(processed_data_dir, 'processed_market_data.csv')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError("Processed data file not found")
        
        data = pd.read_csv(data_file)
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Load predictions
        lstm_predictions = {}
        hmm_predictions = {}
        
        for ticker in INITIAL_ALLOCATION.keys():
            # Load LSTM predictions
            lstm_file = os.path.join(processed_data_dir, f'lstm_predictions_{ticker}.csv')
            if os.path.exists(lstm_file):
                lstm_predictions[ticker] = pd.read_csv(lstm_file)
                lstm_predictions[ticker]['Date'] = pd.to_datetime(lstm_predictions[ticker]['Date'])
            
            # Load HMM predictions
            hmm_file = os.path.join(processed_data_dir, f'hmm_predictions_{ticker}.csv')
            if os.path.exists(hmm_file):
                hmm_predictions[ticker] = pd.read_csv(hmm_file)
                hmm_predictions[ticker]['Date'] = pd.to_datetime(hmm_predictions[ticker]['Date'])
        
        # Run simulation
        print("Running portfolio simulation...")
        daily_values, metrics = simulate_portfolio(data, lstm_predictions, hmm_predictions)
        
        # Save results
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save daily values
        daily_values_file = os.path.join(results_dir, 'portfolio_daily_values.csv')
        daily_values.to_csv(daily_values_file, index=False)
        print(f"Daily values saved to: {daily_values_file}")
        
        # Save metrics
        metrics_file = os.path.join(results_dir, 'portfolio_metrics.csv')
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
        print(f"Metrics saved to: {metrics_file}")
        
        # Print metrics
        print("\nPortfolio Performance Metrics:")
        for metric, value in metrics.items():
            if metric in ['Total_Return', 'Annualized_Return', 'Max_Drawdown']:
                print(f"{metric}: {value:.2%}")
            else:
                print(f"{metric}: {value:.4f}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 