import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from loguru import logger
from config import Config
import os
from datetime import datetime, timedelta

class SimpleStockPredictor:
    """
    Simplified stock predictor using only scikit-learn (no TensorFlow required)
    Uses Random Forest and Linear Regression for price prediction
    """
    def __init__(self, lookback_days=Config.LSTM_LOOKBACK_DAYS):
        self.lookback_days = lookback_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.lr_model = LinearRegression()
        
    def prepare_data(self, data, target_column='Close'):
        """Prepare data for machine learning models"""
        # Use multiple features for better prediction
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI', 'MACD']
        
        # Filter available features
        available_features = [col for col in features if col in data.columns]
        
        if not available_features:
            logger.error("No required features found in data")
            return None, None, None
        
        # Prepare feature data
        feature_data = data[available_features].fillna(method='ffill').fillna(method='bfill')
        
        # Create lagged features (simulate sequence data)
        X_features = []
        y_targets = []
        
        for i in range(self.lookback_days, len(feature_data)):
            # Create features from past lookback_days
            sequence_features = []
            for j in range(self.lookback_days):
                sequence_features.extend(feature_data.iloc[i-self.lookback_days+j].values)
            
            X_features.append(sequence_features)
            y_targets.append(feature_data[target_column].iloc[i])
        
        X = np.array(X_features)
        y = np.array(y_targets)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        return X_scaled, y_scaled, available_features
    
    def train_models(self, data, validation_split=0.2):
        """Train prediction models"""
        try:
            # Prepare data
            X, y, features = self.prepare_data(data)
            
            if X is None:
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=validation_split, random_state=42, shuffle=False
            )
            
            # Train Random Forest
            logger.info("Training Random Forest model...")
            self.rf_model.fit(X_train, y_train)
            
            # Train Linear Regression
            logger.info("Training Linear Regression model...")
            self.lr_model.fit(X_train, y_train)
            
            # Evaluate models
            rf_pred = self.rf_model.predict(X_test)
            lr_pred = self.lr_model.predict(X_test)
            
            rf_score = r2_score(y_test, rf_pred)
            lr_score = r2_score(y_test, lr_pred)
            
            logger.info(f"Random Forest R² Score: {rf_score:.4f}")
            logger.info(f"Linear Regression R² Score: {lr_score:.4f}")
            
            # Save models
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.rf_model, 'models/rf_price_model.pkl')
            joblib.dump(self.lr_model, 'models/lr_price_model.pkl')
            joblib.dump(self.scaler, 'models/simple_price_scaler.pkl')
            joblib.dump(self.feature_scaler, 'models/simple_feature_scaler.pkl')
            joblib.dump(features, 'models/simple_features.pkl')
            
            return True
            
        except Exception as e:
            logger.error(f"Error training simple models: {e}")
            return False
    
    def predict_price(self, data, days_ahead=Config.PREDICTION_DAYS):
        """Predict future prices using ensemble of models"""
        try:
            if not self.load_models():
                logger.warning("Models not loaded, using default prediction")
                return None
            
            # Load feature list
            features = joblib.load('models/simple_features.pkl')
            
            # Prepare features
            feature_data = data[features].fillna(method='ffill').fillna(method='bfill')
            
            predictions = []
            
            for day in range(days_ahead):
                # Get last sequence
                if len(feature_data) < self.lookback_days:
                    logger.warning("Not enough data for prediction")
                    break
                
                # Create features from last lookback_days
                sequence_features = []
                for j in range(self.lookback_days):
                    sequence_features.extend(feature_data.iloc[-(self.lookback_days-j)].values)
                
                X_pred = np.array(sequence_features).reshape(1, -1)
                X_pred_scaled = self.feature_scaler.transform(X_pred)
                
                # Make predictions with both models
                rf_pred_scaled = self.rf_model.predict(X_pred_scaled)[0]
                lr_pred_scaled = self.lr_model.predict(X_pred_scaled)[0]
                
                # Ensemble prediction (average)
                ensemble_pred_scaled = (rf_pred_scaled + lr_pred_scaled) / 2
                
                # Inverse transform
                pred_price = self.scaler.inverse_transform([[ensemble_pred_scaled]])[0, 0]
                predictions.append(pred_price)
                
                # Add prediction to data for next iteration (simplified)
                new_row = feature_data.iloc[-1].copy()
                new_row['Close'] = pred_price
                feature_data = pd.concat([feature_data, new_row.to_frame().T], ignore_index=True)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting prices: {e}")
            return None
    
    def load_models(self):
        """Load trained models"""
        try:
            if (os.path.exists('models/rf_price_model.pkl') and 
                os.path.exists('models/lr_price_model.pkl')):
                
                self.rf_model = joblib.load('models/rf_price_model.pkl')
                self.lr_model = joblib.load('models/lr_price_model.pkl')
                self.scaler = joblib.load('models/simple_price_scaler.pkl')
                self.feature_scaler = joblib.load('models/simple_feature_scaler.pkl')
                
                logger.info("Simple prediction models loaded successfully")
                return True
            else:
                logger.warning("No trained simple models found")
                return False
        except Exception as e:
            logger.error(f"Error loading simple models: {e}")
            return False

class TradingSignalGenerator:
    """Trading signal generator using ensemble methods"""
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
        
        # Technical indicator signals (if available)
        if 'SMA_20' in df.columns:
            df['SMA_Signal'] = np.where(df['Close'] > df['SMA_20'], 1, 0)
        else:
            df['SMA_Signal'] = 0
            
        if 'RSI' in df.columns:
            df['RSI_Signal'] = np.where((df['RSI'] > 30) & (df['RSI'] < 70), 1, 0)
        else:
            df['RSI_Signal'] = 0
            
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            df['MACD_Signal_Cross'] = np.where(df['MACD'] > df['MACD_Signal'], 1, 0)
        else:
            df['MACD_Signal_Cross'] = 0
        
        # Volume signals
        if 'Volume_SMA' in df.columns:
            df['Volume_Signal'] = np.where(df['Volume'] > df['Volume_SMA'], 1, 0)
        else:
            df['Volume_Signal'] = 0
        
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
                'RSI_Signal', 'MACD_Signal_Cross', 'Volume_Signal',
                'Momentum_5', 'Momentum_10', 'Volatility'
            ]
            
            # Add RSI and MACD values if available
            if 'RSI' in feature_data.columns:
                feature_columns.append('RSI')
            if 'MACD' in feature_data.columns:
                feature_columns.append('MACD')
            
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
            joblib.dump(self.rf_model, 'models/simple_rf_signal_model.pkl')
            joblib.dump(self.gb_model, 'models/simple_gb_signal_model.pkl')
            joblib.dump(self.signal_scaler, 'models/simple_signal_scaler.pkl')
            joblib.dump(available_features, 'models/simple_signal_features.pkl')
            
            return True
            
        except Exception as e:
            logger.error(f"Error training signal models: {e}")
            return False
    
    def generate_trading_signal(self, data):
        """Generate trading signal for current market conditions"""
        try:
            # Load models if not already loaded
            if not self.load_signal_models():
                return None
            
            # Prepare features
            feature_data = self.prepare_signal_features(data)
            available_features = joblib.load('models/simple_signal_features.pkl')
            
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
            if (os.path.exists('models/simple_rf_signal_model.pkl') and 
                os.path.exists('models/simple_gb_signal_model.pkl')):
                
                self.rf_model = joblib.load('models/simple_rf_signal_model.pkl')
                self.gb_model = joblib.load('models/simple_gb_signal_model.pkl')
                self.signal_scaler = joblib.load('models/simple_signal_scaler.pkl')
                
                logger.info("Simple signal models loaded successfully")
                return True
            else:
                logger.warning("No trained simple signal models found")
                return False
        except Exception as e:
            logger.error(f"Error loading simple signal models: {e}")
            return False

# Backward compatibility aliases
StockPricePredictor = SimpleStockPredictor

# Example usage
if __name__ == "__main__":
    print("Simple AI Models module loaded successfully")
    print("This version uses only scikit-learn (no TensorFlow required)")
    print("Use SimpleStockPredictor for price prediction")
    print("Use TradingSignalGenerator for trading signals") 