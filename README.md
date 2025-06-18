# AI-Based Dynamic Investment Portfolio Framework

## Project Overview
This project is part of the final coursework for the Practicum in the Master’s in Artificial Intelligence for Business. Its goal is to put into practice the main topics learned throughout the program. This version is a work in progress (WIP).
This project implements a dynamic investment recommendation system using advanced time-series AI models (LSTM and HMM) to generate adaptive buy/sell signals. The system analyzes historical stock and ETF data to provide intelligent portfolio rebalancing recommendations, comparing performance against traditional buy-and-hold strategies.

### Key Features
- Data-driven investment signals using LSTM and Hidden Markov Models
- Dynamic portfolio rebalancing based on AI predictions
- Performance comparison against buy-and-hold strategy
- Comprehensive risk and return metrics
- Modular and extensible codebase

## Project Structure
```
investment_portfolio/
├── data/                  # Data storage directory
│   ├── raw/              # Raw market data
│   └── processed/        # Processed datasets
├── notebooks/            # Jupyter notebooks for analysis
├── src/                  # Source code
│   ├── data/            # Data collection and preprocessing 
│   ├── models/          # LSTM and HMM model implementations  (WIP)
│   ├── portfolio/       # Portfolio management and simulation (WIP)
│   └── utils/           # Utility functions
├── tests/               # Unit tests
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- virtualenv (recommended)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd investment_portfolio
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Collection
```bash
python src/data/collect_data.py
```
This script fetches historical market data for specified tickers using yfinance.

### Model Training
```bash
python src/models/train_models.py
```
Trains both LSTM and HMM models on the collected data.

### Portfolio Simulation
```bash
python src/portfolio/simulate_portfolio.py
```
Runs the portfolio simulation with both dynamic and buy-and-hold strategies.

### Performance Analysis
```bash
python src/portfolio/analyze_performance.py
```
Generates performance metrics and visualizations.

## Output Files
- `data/raw/market_data.csv`: Raw market data
- `data/processed/processed_data.csv`: Preprocessed market data
- `models/lstm_model.h5`: Trained LSTM model
- `models/hmm_model.pkl`: Trained HMM model
- `results/portfolio_performance.csv`: Portfolio performance metrics
- `results/performance_plots/`: Directory containing performance visualizations

## Performance Metrics
The system evaluates portfolio performance using:
- Sharpe Ratio
- Maximum Drawdown
- Cumulative Return
- Volatility
- Win Rate

## Author
Caio Teixeira

## License
This project is licensed under the MIT License - see the LICENSE file for details. 
