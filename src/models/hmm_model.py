"""
Hidden Markov Model module for volatility state detection.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib

# Constants
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed')
N_STATES = 3  # Number of hidden states
FEATURE_COLUMNS = ['Close', 'Volume']  # Use raw columns instead of normalized ones

class HMMPredictor:
    def __init__(
        self,
        n_states: int = N_STATES,
        feature_columns: List[str] = FEATURE_COLUMNS
    ):
        """
        Initialize HMM predictor.
        
        Args:
            n_states: Number of hidden states
            feature_columns: List of feature column names
        """
        self.n_states = n_states
        self.feature_columns = feature_columns
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_features(
        self,
        data: pd.DataFrame,
        ticker: str
    ) -> np.ndarray:
        """
        Prepare features for HMM training.
        
        Args:
            data: DataFrame containing market data
            ticker: Ticker symbol
        
        Returns:
            Array of features
        """
        # Filter data for specific ticker
        ticker_data = data[data['Ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date')
        
        # Calculate returns using raw Close prices
        ticker_data['Returns'] = ticker_data['Close'].pct_change()
        
        # Prepare features using raw data
        features = ticker_data[self.feature_columns + ['Returns']].values
        
        # Remove NaN values
        features = features[~np.isnan(features).any(axis=1)]
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled
    
    def train(
        self,
        data: pd.DataFrame,
        ticker: str,
        n_iter: int = 1000,
        random_state: int = 42
    ) -> Dict:
        """
        Train HMM model.
        
        Args:
            data: DataFrame containing market data
            ticker: Ticker symbol
            n_iter: Maximum number of iterations
            random_state: Random state for reproducibility
        
        Returns:
            Dictionary with model parameters
        """
        # Prepare features
        features = self.prepare_features(data, ticker)
        
        # Initialize and train HMM
        model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state
        )
        
        model.fit(features)
        self.model = model
        
        # Calculate state parameters
        state_params = self._calculate_state_parameters(features)
        
        return state_params
    
    def _calculate_state_parameters(
        self,
        features: np.ndarray
    ) -> Dict:
        """
        Calculate parameters for each state.
        
        Args:
            features: Scaled feature array
        
        Returns:
            Dictionary with state parameters
        """
        # Get state sequence
        state_sequence = self.model.predict(features)
        
        # Calculate parameters for each state
        state_params = {}
        for state in range(self.n_states):
            state_mask = state_sequence == state
            state_features = features[state_mask]
            
            if len(state_features) > 0:
                state_params[state] = {
                    'mean': np.mean(state_features, axis=0),
                    'std': np.std(state_features, axis=0),
                    'count': len(state_features),
                    'proportion': len(state_features) / len(features)
                }
        
        return state_params
    
    def predict_states(
        self,
        data: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """
        Predict hidden states for market data.
        
        Args:
            data: DataFrame containing market data
            ticker: Ticker symbol
        
        Returns:
            DataFrame with predicted states
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        features = self.prepare_features(data, ticker)
        
        # Predict states
        state_sequence = self.model.predict(features)
        
        # Create predictions DataFrame
        ticker_data = data[data['Ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date')
        
        # Add Returns column (needed for filtering)
        ticker_data['Returns'] = ticker_data['Close'].pct_change()
        # Remove rows with NaN values to match feature length
        ticker_data = ticker_data[~ticker_data['Returns'].isna()]
        
        pred_df = pd.DataFrame({
            'Date': ticker_data['Date'],
            'Ticker': ticker,
            'State': state_sequence,
            'State_Probability': np.max(self.model.predict_proba(features), axis=1)
        })
        
        return pred_df
    
    def get_volatility_state(
        self,
        state_params: Dict
    ) -> Dict[int, str]:
        """
        Determine volatility level for each state.
        
        Args:
            state_params: Dictionary with state parameters
        
        Returns:
            Dictionary mapping states to volatility levels
        """
        # Calculate average return volatility for each state
        state_volatility = {}
        for state, params in state_params.items():
            # Use the standard deviation of returns (last feature)
            state_volatility[state] = params['std'][-1]
        
        # Sort states by volatility
        sorted_states = sorted(
            state_volatility.items(),
            key=lambda x: x[1]
        )
        
        # Assign volatility levels
        volatility_levels = ['Low', 'Medium', 'High']
        state_volatility_map = {
            state: level
            for (state, _), level in zip(sorted_states, volatility_levels)
        }
        
        return state_volatility_map
    
    def save_model(self, ticker: str) -> str:
        """
        Save trained model to disk.
        
        Args:
            ticker: Ticker symbol
        
        Returns:
            Path to saved model file
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Save model and scaler
        model_file = os.path.join(MODEL_DIR, f'hmm_{ticker}.pkl')
        scaler_file = os.path.join(MODEL_DIR, f'hmm_scaler_{ticker}.pkl')
        
        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)
        
        return model_file
    
    def load_model(self, ticker: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            ticker: Ticker symbol
        """
        model_file = os.path.join(MODEL_DIR, f'hmm_{ticker}.pkl')
        scaler_file = os.path.join(MODEL_DIR, f'hmm_scaler_{ticker}.pkl')
        
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)

def main():
    """Main function to train and evaluate HMM model."""
    try:
        # Load processed data
        data_file = os.path.join(PROCESSED_DATA_DIR, 'processed_market_data.csv')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError("Processed data file not found")
        
        data = pd.read_csv(data_file)
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Train model for each ticker
        tickers = data['Ticker'].unique()
        for ticker in tickers:
            print(f"\nTraining HMM model for {ticker}...")
            
            predictor = HMMPredictor()
            state_params = predictor.train(data, ticker)
            
            # Get volatility states
            volatility_map = predictor.get_volatility_state(state_params)
            
            # Generate state predictions
            predictions = predictor.predict_states(data, ticker)
            
            # Add volatility levels
            predictions['Volatility_Level'] = predictions['State'].map(volatility_map)
            
            # Save predictions
            pred_file = os.path.join(PROCESSED_DATA_DIR, f'hmm_predictions_{ticker}.csv')
            predictions.to_csv(pred_file, index=False)
            
            print(f"Predictions saved to: {pred_file}")
            
            # Print state summary
            print(f"\nState Summary for {ticker}:")
            for state, level in volatility_map.items():
                params = state_params[state]
                print(f"\nState {state} ({level} Volatility):")
                print(f"Proportion: {params['proportion']:.2%}")
                print(f"Mean Returns: {params['mean'][-1]:.6f}")
                print(f"Returns Volatility: {params['std'][-1]:.6f}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 