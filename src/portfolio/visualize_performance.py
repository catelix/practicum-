"""
Visualization module for generating performance plots.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Constants
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'performance_plots')

def plot_portfolio_value(
    daily_values: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot portfolio value over time.
    
    Args:
        daily_values: DataFrame with daily portfolio values
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot portfolio value
    plt.plot(daily_values['Date'], daily_values['Portfolio_Value'], label='Portfolio Value')
    
    # Add labels and title
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_returns_distribution(
    daily_values: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of daily returns.
    
    Args:
        daily_values: DataFrame with daily portfolio values
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate returns
    returns = daily_values['Portfolio_Value'].pct_change().dropna()
    
    # Plot histogram with KDE
    sns.histplot(returns, kde=True, bins=50)
    
    # Add labels and title
    plt.title('Distribution of Daily Returns')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Add mean and std lines
    mean_return = returns.mean()
    std_return = returns.std()
    
    plt.axvline(mean_return, color='r', linestyle='--', label=f'Mean: {mean_return:.4f}')
    plt.axvline(mean_return + std_return, color='g', linestyle=':', label=f'+1 Std: {mean_return + std_return:.4f}')
    plt.axvline(mean_return - std_return, color='g', linestyle=':', label=f'-1 Std: {mean_return - std_return:.4f}')
    
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_drawdown(
    daily_values: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot portfolio drawdown over time.
    
    Args:
        daily_values: DataFrame with daily portfolio values
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate drawdown
    daily_values['Cumulative_Max'] = daily_values['Portfolio_Value'].cummax()
    daily_values['Drawdown'] = (daily_values['Portfolio_Value'] - daily_values['Cumulative_Max']) / daily_values['Cumulative_Max']
    
    # Plot drawdown
    plt.fill_between(
        daily_values['Date'],
        daily_values['Drawdown'],
        0,
        color='red',
        alpha=0.3
    )
    plt.plot(daily_values['Date'], daily_values['Drawdown'], color='red', label='Drawdown')
    
    # Add labels and title
    plt.title('Portfolio Drawdown Over Time')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True)
    plt.legend()
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_rolling_metrics(
    daily_values: pd.DataFrame,
    window: int = 252,  # 1 year
    save_path: Optional[str] = None
) -> None:
    """
    Plot rolling metrics (volatility and Sharpe ratio).
    
    Args:
        daily_values: DataFrame with daily portfolio values
        window: Rolling window size in days
        save_path: Optional path to save the plot
    """
    # Calculate returns
    daily_values['Returns'] = daily_values['Portfolio_Value'].pct_change()
    
    # Calculate rolling metrics
    rolling_vol = daily_values['Returns'].rolling(window=window).std() * np.sqrt(252)
    rolling_mean = daily_values['Returns'].rolling(window=window).mean() * 252
    rolling_sharpe = rolling_mean / rolling_vol
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot rolling volatility
    ax1.plot(daily_values['Date'], rolling_vol, label='Rolling Volatility')
    ax1.set_title('Rolling Annualized Volatility')
    ax1.set_ylabel('Volatility')
    ax1.grid(True)
    ax1.legend()
    
    # Plot rolling Sharpe ratio
    ax2.plot(daily_values['Date'], rolling_sharpe, label='Rolling Sharpe Ratio', color='green')
    ax2.set_title('Rolling Sharpe Ratio')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.grid(True)
    ax2.legend()
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_metrics_comparison(
    metrics: Dict[str, float],
    benchmark_metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison of performance metrics.
    
    Args:
        metrics: Dictionary of portfolio metrics
        benchmark_metrics: Optional dictionary of benchmark metrics
        save_path: Optional path to save the plot
    """
    # Prepare data
    metrics_data = []
    
    # Add portfolio metrics
    for metric, value in metrics.items():
        metrics_data.append({
            'Metric': metric,
            'Value': value,
            'Strategy': 'Dynamic Portfolio'
        })
    
    # Add benchmark metrics if provided
    if benchmark_metrics:
        for metric, value in benchmark_metrics.items():
            metrics_data.append({
                'Metric': metric,
                'Value': value,
                'Strategy': 'Buy & Hold'
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_data)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot metrics
    sns.barplot(
        data=df,
        x='Metric',
        y='Value',
        hue='Strategy',
        palette='Set2'
    )
    
    # Add labels and title
    plt.title('Performance Metrics Comparison')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.grid(True)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    """Main function to generate performance visualizations."""
    try:
        # Create plots directory
        os.makedirs(PLOTS_DIR, exist_ok=True)
        
        # Load portfolio data
        daily_values_file = os.path.join(RESULTS_DIR, 'portfolio_daily_values.csv')
        metrics_file = os.path.join(RESULTS_DIR, 'portfolio_metrics.csv')
        
        if not os.path.exists(daily_values_file) or not os.path.exists(metrics_file):
            raise FileNotFoundError("Portfolio data files not found")
        
        daily_values = pd.read_csv(daily_values_file)
        daily_values['Date'] = pd.to_datetime(daily_values['Date'])
        
        metrics = pd.read_csv(metrics_file).iloc[0].to_dict()
        
        # Generate plots
        print("Generating performance plots...")
        
        # Portfolio value plot
        plot_portfolio_value(
            daily_values,
            save_path=os.path.join(PLOTS_DIR, 'portfolio_value.png')
        )
        
        # Returns distribution plot
        plot_returns_distribution(
            daily_values,
            save_path=os.path.join(PLOTS_DIR, 'returns_distribution.png')
        )
        
        # Drawdown plot
        plot_drawdown(
            daily_values,
            save_path=os.path.join(PLOTS_DIR, 'drawdown.png')
        )
        
        # Rolling metrics plot
        plot_rolling_metrics(
            daily_values,
            save_path=os.path.join(PLOTS_DIR, 'rolling_metrics.png')
        )
        
        # Metrics comparison plot
        plot_metrics_comparison(
            metrics,
            save_path=os.path.join(PLOTS_DIR, 'metrics_comparison.png')
        )
        
        print(f"Plots saved to: {PLOTS_DIR}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 