import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
from loguru import logger
from config import Config
import os
from datetime import datetime, timedelta

class StockPricePredictor:
    def __init__(self, lookback_days=Config.LSTM_LOOKBACK_DAYS):
        self.lookback_days = lookback_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.feature_scaler = StandardScaler()
        
    def prepare_lstm_data(self, data, target_column='Close'):
        """Prepare data for LSTM model"""
        # Use multiple features for better prediction
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI', 'MACD']
        
        # Filter available features
        available_features = [col for col in features if col in data.columns]
        
        if not available_features:
            logger.error("No required features found in data")
            return None, None, None
        
        # Prepare feature data
        feature_data = data[available_features].fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(feature_data)
        
        # Scale target
        target_data = data[target_column].values.reshape(-1, 1)
        scaled_target = self.scaler.fit_transform(target_data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback_days, len(scaled_features)):
            X.append(scaled_features[i-self.lookback_days:i])
            y.append(scaled_target[i, 0])
        
        return np.array(X), np.array(y), available_features
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(100, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(50, return_sequences=False)),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )
        
        return model
    
    def train_lstm_model(self, data, epochs=100, batch_size=32, validation_split=0.2):
        """Train LSTM model for price prediction"""
        try:
            # Prepare data
            X, y, features = self.prepare_lstm_data(data)
            
            if X is None:
                return False
            
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build model
            self.model = self.build_lstm_model((X.shape[1], X.shape[2]))
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )
            
            model_checkpoint = ModelCheckpoint(
                'models/best_lstm_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
            
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, model_checkpoint],
                verbose=1
            )
            
            # Evaluate model
            train_loss = self.model.evaluate(X_train, y_train, verbose=0)
            test_loss = self.model.evaluate(X_test, y_test, verbose=0)
            
            logger.info(f"Training Loss: {train_loss}")
            logger.info(f"Test Loss: {test_loss}")
            
            # Save scalers
            joblib.dump(self.scaler, 'models/price_scaler.pkl')
            joblib.dump(self.feature_scaler, 'models/feature_scaler.pkl')
            
            return True
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return False
    
    def predict_price(self, data, days_ahead=Config.PREDICTION_DAYS):
        """Predict future prices"""
        try:
            if self.model is None:
                self.load_model()
            
            # Prepare features
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI', 'MACD']
            available_features = [col for col in features if col in data.columns]
            
            if not available_features:
                logger.error("No required features found for prediction")
                return None
            
            # Get last sequence
            feature_data = data[available_features].fillna(method='ffill').fillna(method='bfill')
            scaled_features = self.feature_scaler.transform(feature_data)
            
            # Create prediction sequence
            last_sequence = scaled_features[-self.lookback_days:]
            predictions = []
            
            # Predict multiple days ahead
            current_sequence = last_sequence.copy()
            
            for _ in range(days_ahead):
                # Reshape for prediction
                pred_input = current_sequence.reshape(1, self.lookback_days, len(available_features))
                
                # Make prediction
                scaled_pred = self.model.predict(pred_input, verbose=0)
                
                # Inverse transform prediction
                pred_price = self.scaler.inverse_transform(scaled_pred)[0, 0]
                predictions.append(pred_price)
                
                # Update sequence for next prediction
                # Note: This is a simplified approach - in practice, you'd want to
                # update all features based on the predicted price
                new_row = current_sequence[-1].copy()
                new_row[available_features.index('Close')] = scaled_pred[0, 0]
                
                current_sequence = np.vstack([current_sequence[1:], new_row])
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting prices: {e}")
            return None
    
    def load_model(self):
        """Load trained model and scalers"""
        try:
            if os.path.exists('models/best_lstm_model.h5'):
                self.model = load_model('models/best_lstm_model.h5')
                self.scaler = joblib.load('models/price_scaler.pkl')
                self.feature_scaler = joblib.load('models/feature_scaler.pkl')
                logger.info("Model loaded successfully")
                return True
            else:
                logger.warning("No trained model found")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

class TradingSignalGenerator:
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.signal_scaler = StandardScaler()
        
    def prepare_signal_features(self, data):
        """Prepare features for trading signal generation"""
        df = data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Price_Change'] = df['Close'] - df['Open']
        df['High_Low_Ratio'] = df['High'] / df['Low']
        
        # Technical indicator signals
        df['SMA_Signal'] = np.where(df['Close'] > df['SMA_20'], 1, 0)
        df['RSI_Signal'] = np.where((df['RSI'] > 30) & (df['RSI'] < 70), 1, 0)
        df['MACD_Signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1, 0)
        df['BB_Signal'] = np.where(
            (df['Close'] > df['BB_Lower']) & (df['Close'] < df['BB_Upper']), 1, 0
        )
        
        # Volume signals
        df['Volume_Signal'] = np.where(df['Volume'] > df['Volume_SMA'], 1, 0)
        
        # Momentum indicators
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        return df
    
    def create_trading_targets(self, data, holding_period=5):
        """Create trading targets based on future returns"""
        df = data.copy()
        
        # Calculate future returns
        future_returns = df['Close'].shift(-holding_period) / df['Close'] - 1
        
        # Create categorical targets
        # 1: Strong Buy (>5% return), 0.5: Buy (2-5%), 0: Hold (-2% to 2%), -0.5: Sell (-5% to -2%), -1: Strong Sell (<-5%)
        targets = np.where(future_returns > 0.05, 1,
                  np.where(future_returns > 0.02, 0.5,
                  np.where(future_returns > -0.02, 0,
                  np.where(future_returns > -0.05, -0.5, -1))))
        
        return targets
    
    def train_signal_models(self, data):
        """Train models for trading signal generation"""
        try:
            # Prepare features
            feature_data = self.prepare_signal_features(data)
            
            # Select feature columns
            feature_columns = [
                'Returns', 'Price_Change', 'High_Low_Ratio', 'SMA_Signal',
                'RSI_Signal', 'MACD_Signal', 'BB_Signal', 'Volume_Signal',
                'Momentum_5', 'Momentum_10', 'Volatility', 'RSI', 'MACD'
            ]
            
            # Filter available features
            available_features = [col for col in feature_columns if col in feature_data.columns]
            
            if not available_features:
                logger.error("No features available for signal generation")
                return False
            
            # Prepare data
            X = feature_data[available_features].fillna(0)
            y = self.create_trading_targets(feature_data)
            
            # Remove rows with NaN targets
            valid_indices = ~np.isnan(y)
            X = X[valid_indices]
            y = y[valid_indices]
            
            if len(X) == 0:
                logger.error("No valid data for training signal models")
                return False
            
            # Scale features
            X_scaled = self.signal_scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train models
            self.rf_model.fit(X_train, y_train)
            self.gb_model.fit(X_train, y_train)
            
            # Evaluate models
            rf_score = self.rf_model.score(X_test, y_test)
            gb_score = self.gb_model.score(X_test, y_test)
            
            logger.info(f"Random Forest R² Score: {rf_score:.4f}")
            logger.info(f"Gradient Boosting R² Score: {gb_score:.4f}")
            
            # Save models
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.rf_model, 'models/rf_signal_model.pkl')
            joblib.dump(self.gb_model, 'models/gb_signal_model.pkl')
            joblib.dump(self.signal_scaler, 'models/signal_scaler.pkl')
            joblib.dump(available_features, 'models/signal_features.pkl')
            
            return True
            
        except Exception as e:
            logger.error(f"Error training signal models: {e}")
            return False
    
    def generate_trading_signal(self, data):
        """Generate trading signal for current market conditions"""
        try:
            # Load models if not already loaded
            if not hasattr(self, 'rf_model') or self.rf_model is None:
                self.load_signal_models()
            
            # Prepare features
            feature_data = self.prepare_signal_features(data)
            available_features = joblib.load('models/signal_features.pkl')
            
            # Get latest data point
            latest_features = feature_data[available_features].iloc[-1:].fillna(0)
            
            # Scale features
            scaled_features = self.signal_scaler.transform(latest_features)
            
            # Generate predictions
            rf_signal = self.rf_model.predict(scaled_features)[0]
            gb_signal = self.gb_model.predict(scaled_features)[0]
            
            # Ensemble prediction
            ensemble_signal = (rf_signal + gb_signal) / 2
            
            # Generate recommendation
            if ensemble_signal > 0.3:
                recommendation = "STRONG_BUY"
                confidence = min(ensemble_signal, 1.0)
            elif ensemble_signal > 0.1:
                recommendation = "BUY"
                confidence = ensemble_signal
            elif ensemble_signal > -0.1:
                recommendation = "HOLD"
                confidence = 1 - abs(ensemble_signal)
            elif ensemble_signal > -0.3:
                recommendation = "SELL"
                confidence = abs(ensemble_signal)
            else:
                recommendation = "STRONG_SELL"
                confidence = min(abs(ensemble_signal), 1.0)
            
            return {
                'signal': ensemble_signal,
                'recommendation': recommendation,
                'confidence': confidence,
                'rf_signal': rf_signal,
                'gb_signal': gb_signal
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return None
    
    def load_signal_models(self):
        """Load trained signal models"""
        try:
            self.rf_model = joblib.load('models/rf_signal_model.pkl')
            self.gb_model = joblib.load('models/gb_signal_model.pkl')
            self.signal_scaler = joblib.load('models/signal_scaler.pkl')
            logger.info("Signal models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading signal models: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # This would typically be run with real stock data
    print("AI Models module loaded successfully")
    print("Use StockPricePredictor for price prediction")
    print("Use TradingSignalGenerator for trading signals") 