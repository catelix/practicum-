"""
LSTM model module for time series prediction.
"""

import os

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Constants
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed')
SEQUENCE_LENGTH = 60  # Number of time steps to look back
PREDICTION_HORIZON = 5  # Number of days to predict ahead
FEATURE_COLUMNS = ['Open_Normalized', 'High_Normalized', 'Low_Normalized', 'Close_Normalized', 'Volume_Normalized']
TARGET_COLUMN = 'Close_Normalized'

class LSTMPredictor:
    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        prediction_horizon: int = PREDICTION_HORIZON,
        feature_columns: List[str] = FEATURE_COLUMNS,
        target_column: str = TARGET_COLUMN
    ):
        """
        Initialize LSTM predictor.
        
        Args:
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of days to predict ahead
            feature_columns: List of feature column names
            target_column: Target column name
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.model = None
        self.scalers = {}
        
    def prepare_sequences(
        self,
        data: pd.DataFrame,
        ticker: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            data: DataFrame containing market data
            ticker: Ticker symbol
        
        Returns:
            Tuple of (X, y) arrays for training
        """
        # Filter data for specific ticker
        ticker_data = data[data['Ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date')
        
        # Prepare feature and target arrays
        features = ticker_data[self.feature_columns].values
        targets = ticker_data[self.target_column].values
        
        X, y = [], []
        
        for i in range(len(ticker_data) - self.sequence_length - self.prediction_horizon + 1):
            X.append(features[i:(i + self.sequence_length)])
            y.append(targets[i + self.sequence_length + self.prediction_horizon - 1])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
    
    def train(
        self,
        data: pd.DataFrame,
        ticker: str,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10
    ) -> Dict:
        """
        Train LSTM model.
        
        Args:
            data: DataFrame containing market data
            ticker: Ticker symbol
            validation_split: Validation split ratio
            epochs: Number of training epochs
            batch_size: Batch size
            patience: Early stopping patience
        
        Returns:
            Training history dictionary
        """
        # Prepare sequences
        X, y = self.prepare_sequences(data, ticker)
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            shuffle=False
        )
        
        # Build model if not already built
        if self.model is None:
            self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=os.path.join(MODEL_DIR, f'lstm_{ticker}.h5'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def predict(
        self,
        data: pd.DataFrame,
        ticker: str,
        last_n_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate predictions using trained model.
        
        Args:
            data: DataFrame containing market data
            ticker: Ticker symbol
            last_n_days: Optional number of days to predict
        
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Filter data for specific ticker
        ticker_data = data[data['Ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date')
        
        if last_n_days is not None:
            ticker_data = ticker_data.tail(last_n_days + self.sequence_length)
        
        # Prepare sequences
        features = ticker_data[self.feature_columns].values
        
        # Generate predictions
        predictions = []
        dates = []
        
        for i in range(len(ticker_data) - self.sequence_length):
            X = features[i:(i + self.sequence_length)]
            X = X.reshape(1, self.sequence_length, len(self.feature_columns))
            
            pred = self.model.predict(X, verbose=0)[0][0]
            predictions.append(pred)
            dates.append(ticker_data['Date'].iloc[i + self.sequence_length])
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame({
            'Date': dates,
            'Ticker': ticker,
            'Predicted_Close_Normalized': predictions
        })
        
        return pred_df
    
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
        filepath = os.path.join(MODEL_DIR, f'lstm_{ticker}.h5')
        self.model.save(filepath)
        
        return filepath
    
    def load_model(self, ticker: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            ticker: Ticker symbol
        """
        filepath = os.path.join(MODEL_DIR, f'lstm_{ticker}.h5')
        self.model = load_model(filepath)

def main():
    """Main function to train and evaluate LSTM model."""
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
            print(f"\nTraining LSTM model for {ticker}...")
            
            predictor = LSTMPredictor()
            history = predictor.train(data, ticker)
            
            # Generate predictions
            predictions = predictor.predict(data, ticker)
            
            # Save predictions
            pred_file = os.path.join(PROCESSED_DATA_DIR, f'lstm_predictions_{ticker}.csv')
            predictions.to_csv(pred_file, index=False)
            
            print(f"Predictions saved to: {pred_file}")
            
            # Print training summary
            print(f"\nTraining Summary for {ticker}:")
            print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
            print(f"Final validation MAE: {history['val_mae'][-1]:.6f}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
