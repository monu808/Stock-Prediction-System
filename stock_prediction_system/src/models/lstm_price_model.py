"""LSTM model for stock price prediction"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from datetime import datetime, timedelta

from ..utils.logging_config import LoggingMixin


class LSTMPricePredictor(nn.Module, LoggingMixin):
    """LSTM neural network for stock price prediction"""
    
    def __init__(
        self,
        input_size: int = 5,  # OHLCV
        hidden_sizes: List[int] = [128, 64, 32],
        output_size: int = 1,  # Predicted price
        dropout: float = 0.2,
        num_layers: int = 2
    ):
        super(LSTMPricePredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_sizes[0],
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        )
        
        # Additional LSTM layers
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=hidden_sizes[i-1],
                    hidden_size=hidden_sizes[i],
                    num_layers=1,
                    dropout=0,
                    batch_first=True
                )
            )
        
        # Dropout layers
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(len(hidden_sizes))
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        batch_size = x.size(0)
        
        # Pass through LSTM layers
        lstm_out = x
        for i, lstm_layer in enumerate(self.lstm_layers):
            lstm_out, _ = lstm_layer(lstm_out)
            lstm_out = self.dropout_layers[i](lstm_out)
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Output layer
        output = self.output_layer(lstm_out)
        
        return output


class StockPriceDataset:
    """Dataset class for stock price data"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 60,
        target_column: str = 'close',
        feature_columns: List[str] = ['open', 'high', 'low', 'close', 'volume']
    ):
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.feature_columns = feature_columns
        
        # Prepare data
        self.X, self.y = self._prepare_sequences()
        
        # Scalers
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        # Fit and transform data
        self.X_scaled = self._scale_features(self.X)
        self.y_scaled = self._scale_targets(self.y)
    
    def _prepare_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare input sequences and targets"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(self.data)):
            # Input sequence
            sequence = self.data[self.feature_columns].iloc[i-self.sequence_length:i].values
            X.append(sequence)
            
            # Target (next day's closing price)
            target = self.data[self.target_column].iloc[i]
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale input features"""
        # Reshape for scaling
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        # Fit and transform
        X_scaled = self.feature_scaler.fit_transform(X_reshaped)
        
        # Reshape back
        return X_scaled.reshape(original_shape)
    
    def _scale_targets(self, y: np.ndarray) -> np.ndarray:
        """Scale target values"""
        return self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse transform predictions to original scale"""
        return self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    def get_train_test_split(self, test_size: float = 0.2) -> Tuple:
        """Split data into train and test sets"""
        split_idx = int(len(self.X_scaled) * (1 - test_size))
        
        X_train = self.X_scaled[:split_idx]
        X_test = self.X_scaled[split_idx:]
        y_train = self.y_scaled[:split_idx]
        y_test = self.y_scaled[split_idx:]
        
        return X_train, X_test, y_train, y_test


class LSTMTrainer(LoggingMixin):
    """Trainer class for LSTM model"""
    
    def __init__(
        self,
        model: LSTMPricePredictor,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = 32
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create batches
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # Convert to tensors
            batch_X = torch.FloatTensor(batch_X).to(self.device)
            batch_y = torch.FloatTensor(batch_y).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs.squeeze(), batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32
    ) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_X = X_val[i:i+batch_size]
                batch_y = y_val[i:i+batch_size]
                
                batch_X = torch.FloatTensor(batch_X).to(self.device)
                batch_y = torch.FloatTensor(batch_y).to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self,
        dataset: StockPriceDataset,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """Train the model"""
        self.logger.info(f"Starting training for {epochs} epochs...")
        
        # Split data
        X_train, X_val, y_train, y_val = dataset.get_train_test_split(validation_split)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(X_train, y_train, batch_size)
            
            # Validate
            val_loss = self.validate(X_val, y_val, batch_size)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_checkpoint(f'best_model_epoch_{epoch}.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}/{epochs} - "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )
            
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = outputs.cpu().numpy()
        
        return predictions
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])


class PricePredictionEngine(LoggingMixin):
    """Main engine for stock price prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}  # Symbol -> Model mapping
        self.datasets = {}  # Symbol -> Dataset mapping
        self.trainers = {}  # Symbol -> Trainer mapping
        
        # Model configuration
        self.model_config = config.get('lstm_price', {})
        self.sequence_length = self.model_config.get('sequence_length', 60)
        self.hidden_units = self.model_config.get('hidden_units', [128, 64, 32])
        self.dropout = self.model_config.get('dropout', 0.2)
        self.learning_rate = self.model_config.get('learning_rate', 0.001)
        self.batch_size = self.model_config.get('batch_size', 32)
        self.epochs = self.model_config.get('epochs', 100)
    
    def prepare_symbol_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Prepare data for a specific symbol"""
        # Create dataset
        dataset = StockPriceDataset(
            data=data,
            sequence_length=self.sequence_length
        )
        self.datasets[symbol] = dataset
        
        # Create model
        model = LSTMPricePredictor(
            input_size=5,  # OHLCV
            hidden_sizes=self.hidden_units,
            dropout=self.dropout
        )
        self.models[symbol] = model
        
        # Create trainer
        trainer = LSTMTrainer(
            model=model,
            learning_rate=self.learning_rate
        )
        self.trainers[symbol] = trainer
        
        self.logger.info(f"Prepared data and model for {symbol}")
    
    def train_symbol_model(self, symbol: str) -> Dict[str, Any]:
        """Train model for a specific symbol"""
        if symbol not in self.trainers:
            raise ValueError(f"No trainer found for symbol {symbol}")
        
        trainer = self.trainers[symbol]
        dataset = self.datasets[symbol]
        
        # Train the model
        history = trainer.train(
            dataset=dataset,
            epochs=self.epochs,
            batch_size=self.batch_size
        )
        
        self.logger.info(f"Training completed for {symbol}")
        return history
    
    def predict_next_price(
        self,
        symbol: str,
        recent_data: np.ndarray
    ) -> Tuple[float, float]:
        """Predict next price for a symbol"""
        if symbol not in self.models:
            raise ValueError(f"No model found for symbol {symbol}")
        
        trainer = self.trainers[symbol]
        dataset = self.datasets[symbol]
        
        # Scale the input data
        recent_data_scaled = dataset.feature_scaler.transform(recent_data)
        
        # Make prediction
        prediction_scaled = trainer.predict(
            recent_data_scaled.reshape(1, -1, recent_data_scaled.shape[-1])
        )
        
        # Inverse transform
        prediction = dataset.inverse_transform_predictions(prediction_scaled)[0]
        
        # Calculate confidence (simplified)
        confidence = 0.8  # This would be calculated based on model uncertainty
        
        return prediction, confidence
    
    def get_model_performance(self, symbol: str) -> Dict[str, float]:
        """Get performance metrics for a symbol's model"""
        if symbol not in self.trainers:
            return {}
        
        trainer = self.trainers[symbol]
        
        return {
            'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else 0.0,
            'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else 0.0,
            'best_val_loss': min(trainer.val_losses) if trainer.val_losses else 0.0,
            'epochs_trained': len(trainer.train_losses)
        }
    
    def save_models(self, save_dir: str) -> None:
        """Save all trained models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for symbol, trainer in self.trainers.items():
            filepath = os.path.join(save_dir, f"{symbol}_lstm_model.pth")
            trainer.save_checkpoint(filepath)
            
            # Save dataset scalers
            dataset = self.datasets[symbol]
            scaler_path = os.path.join(save_dir, f"{symbol}_scalers.joblib")
            joblib.dump({
                'feature_scaler': dataset.feature_scaler,
                'target_scaler': dataset.target_scaler
            }, scaler_path)
        
        self.logger.info(f"Models saved to {save_dir}")
    
    def load_models(self, load_dir: str) -> None:
        """Load pre-trained models"""
        import os
        
        # Find all model files
        for filename in os.listdir(load_dir):
            if filename.endswith('_lstm_model.pth'):
                symbol = filename.replace('_lstm_model.pth', '')
                
                # Create model and trainer
                model = LSTMPricePredictor(
                    hidden_sizes=self.hidden_units,
                    dropout=self.dropout
                )
                trainer = LSTMTrainer(model=model, learning_rate=self.learning_rate)
                
                # Load checkpoint
                filepath = os.path.join(load_dir, filename)
                trainer.load_checkpoint(filepath)
                
                self.models[symbol] = model
                self.trainers[symbol] = trainer
                
                # Load scalers
                scaler_path = os.path.join(load_dir, f"{symbol}_scalers.joblib")
                if os.path.exists(scaler_path):
                    scalers = joblib.load(scaler_path)
                    # Create dummy dataset to hold scalers
                    dataset = StockPriceDataset(pd.DataFrame(), sequence_length=self.sequence_length)
                    dataset.feature_scaler = scalers['feature_scaler']
                    dataset.target_scaler = scalers['target_scaler']
                    self.datasets[symbol] = dataset
        
        self.logger.info(f"Models loaded from {load_dir}")