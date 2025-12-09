"""
Advanced ML Models for Stock Price Prediction
Ensemble of LSTM, XGBoost, and LightGBM for directional prediction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Run: pip install lightgbm")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
except ImportError:
    HAS_TENSORFLOW = False
    print("Warning: TensorFlow not installed. Run: pip install tensorflow")


@dataclass
class PredictionResult:
    """Results from ML prediction"""
    ticker: str
    direction: str  # "UP", "DOWN", "NEUTRAL"
    confidence: float  # 0.0 to 1.0
    predicted_change_pct: float  # Expected % change
    
    # Individual model predictions
    lstm_prediction: Optional[float] = None
    xgboost_prediction: Optional[float] = None
    lightgbm_prediction: Optional[float] = None
    
    # Model performance metrics
    ensemble_accuracy: float = 0.0
    
    # Feature importances (top 5)
    top_features: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'ticker': self.ticker,
            'direction': self.direction,
            'confidence': self.confidence,
            'predicted_change_pct': self.predicted_change_pct,
            'lstm_pred': self.lstm_prediction,
            'xgb_pred': self.xgboost_prediction,
            'lgb_pred': self.lightgbm_prediction,
            'accuracy': self.ensemble_accuracy
        }


class FeatureEngineer:
    """Creates advanced features for stock prediction."""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_names = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set from OHLCV data.
        
        Features include:
        - Price-based: returns, log returns, momentum
        - Moving averages: SMA, EMA, crossovers
        - Volatility: ATR, Bollinger Bands, historical vol
        - Volume: OBV, volume momentum, VWAP
        - Momentum indicators: RSI, MACD, Stochastic
        - Trend indicators: ADX, Aroon
        - Seasonality: day of week, month, quarter
        """
        data = df.copy()
        
        # Ensure we have required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # =====================
        # PRICE-BASED FEATURES
        # =====================
        
        # Returns
        data['returns'] = data['Close'].pct_change()
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Multi-period returns
        for period in [2, 3, 5, 10, 20]:
            data[f'returns_{period}d'] = data['Close'].pct_change(period)
        
        # Price momentum
        for period in [5, 10, 20, 50]:
            data[f'momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
        
        # =====================
        # MOVING AVERAGES
        # =====================
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 200]:
            data[f'sma_{period}'] = data['Close'].rolling(window=period).mean()
            data[f'close_vs_sma_{period}'] = (data['Close'] - data[f'sma_{period}']) / data[f'sma_{period}']
        
        # Exponential Moving Averages
        for period in [12, 26, 50]:
            data[f'ema_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
        
        # MA Crossovers (binary signals)
        data['sma_5_20_cross'] = (data['sma_5'] > data['sma_20']).astype(int)
        data['sma_20_50_cross'] = (data['sma_20'] > data['sma_50']).astype(int)
        data['ema_12_26_cross'] = (data['ema_12'] > data['ema_26']).astype(int)
        
        # =====================
        # VOLATILITY FEATURES
        # =====================
        
        # Historical volatility
        for period in [5, 10, 20]:
            data[f'volatility_{period}'] = data['returns'].rolling(window=period).std() * np.sqrt(252)
        
        # Average True Range (ATR)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['atr_14'] = true_range.rolling(14).mean()
        data['atr_pct'] = data['atr_14'] / data['Close']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        data['bb_middle'] = data['Close'].rolling(bb_period).mean()
        bb_std_dev = data['Close'].rolling(bb_period).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std_dev * bb_std)
        data['bb_lower'] = data['bb_middle'] - (bb_std_dev * bb_std)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # =====================
        # VOLUME FEATURES
        # =====================
        
        # Volume momentum
        data['volume_sma_20'] = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_sma_20']
        
        # Volume momentum
        for period in [5, 10]:
            data[f'volume_momentum_{period}'] = data['Volume'] / data['Volume'].shift(period)
        
        # On-Balance Volume (simplified)
        data['obv'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        data['obv_sma_20'] = data['obv'].rolling(20).mean()
        data['obv_signal'] = (data['obv'] > data['obv_sma_20']).astype(int)
        
        # Price-Volume trend
        data['pv_trend'] = data['returns'] * data['volume_ratio']
        
        # =====================
        # MOMENTUM INDICATORS
        # =====================
        
        # RSI
        for period in [7, 14, 21]:
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI divergence (price up but RSI down = bearish)
        data['rsi_divergence'] = data['momentum_5'] - (data['rsi_14'].diff(5) / 100)
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        data['macd_cross'] = (data['macd'] > data['macd_signal']).astype(int)
        
        # Stochastic Oscillator
        low_14 = data['Low'].rolling(14).min()
        high_14 = data['High'].rolling(14).max()
        data['stoch_k'] = 100 * (data['Close'] - low_14) / (high_14 - low_14)
        data['stoch_d'] = data['stoch_k'].rolling(3).mean()
        
        # Williams %R
        data['williams_r'] = -100 * (high_14 - data['Close']) / (high_14 - low_14)
        
        # =====================
        # TREND INDICATORS
        # =====================
        
        # ADX (Average Directional Index) - simplified
        plus_dm = data['High'].diff()
        minus_dm = -data['Low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = true_range
        atr14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        data['adx'] = dx.rolling(14).mean()
        data['plus_di'] = plus_di
        data['minus_di'] = minus_di
        
        # Trend strength
        data['trend_strength'] = np.abs(data['close_vs_sma_50']) * data['adx'] / 100
        
        # =====================
        # SEASONALITY FEATURES
        # =====================
        
        if isinstance(data.index, pd.DatetimeIndex):
            data['day_of_week'] = data.index.dayofweek
            data['month'] = data.index.month
            data['quarter'] = data.index.quarter
            data['is_month_start'] = data.index.is_month_start.astype(int)
            data['is_month_end'] = data.index.is_month_end.astype(int)
            data['is_quarter_end'] = data.index.is_quarter_end.astype(int)
        
        # =====================
        # PATTERN FEATURES
        # =====================
        
        # Candlestick patterns (simplified)
        data['body'] = data['Close'] - data['Open']
        data['body_pct'] = data['body'] / data['Open']
        data['upper_shadow'] = data['High'] - data[['Open', 'Close']].max(axis=1)
        data['lower_shadow'] = data[['Open', 'Close']].min(axis=1) - data['Low']
        
        # Doji (small body)
        data['is_doji'] = (np.abs(data['body_pct']) < 0.001).astype(int)
        
        # Gap
        data['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        
        # =====================
        # LAGGED FEATURES
        # =====================
        
        # Lagged returns (to capture autocorrelation)
        for lag in [1, 2, 3, 5]:
            data[f'returns_lag_{lag}'] = data['returns'].shift(lag)
        
        # Store feature names (excluding target and non-features)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
                       'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200', 
                       'ema_12', 'ema_26', 'ema_50', 'bb_middle', 'bb_upper', 'bb_lower',
                       'obv', 'obv_sma_20', 'atr_14']
        self.feature_names = [col for col in data.columns if col not in exclude_cols]
        
        return data
    
    def prepare_data_for_training(
        self,
        df: pd.DataFrame,
        target_days: int = 5,
        binary_target: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare data for model training.
        
        Args:
            df: DataFrame with features
            target_days: Days ahead to predict
            binary_target: If True, predict direction (1=up, 0=down)
            
        Returns:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
        """
        data = self.create_features(df)
        
        # Create target: future return
        data['target'] = data['Close'].shift(-target_days) / data['Close'] - 1
        
        if binary_target:
            # 1 = price goes up, 0 = price goes down
            data['target'] = (data['target'] > 0).astype(int)
        
        # Drop rows with NaN
        data = data.dropna()
        
        # Select features
        feature_cols = [col for col in self.feature_names if col in data.columns]
        
        X = data[feature_cols].values
        y = data['target'].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y, feature_cols


class LSTMModel:
    """LSTM model for time-series prediction."""
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.model = None
        self.scaler = MinMaxScaler()
        
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        X_seq, y_seq = [], []
        for i in range(self.lookback, len(X)):
            X_seq.append(X[i-self.lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build LSTM model architecture."""
        if not HAS_TENSORFLOW:
            return
            
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32) -> dict:
        """Train the LSTM model."""
        if not HAS_TENSORFLOW:
            return {'accuracy': 0.0}
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)
        
        if len(X_seq) < 100:
            return {'accuracy': 0.0}
        
        # Split data
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        # Build model
        self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Train with early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        y_pred = (self.model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_test, y_pred)
        
        return {'accuracy': accuracy}
    
    def predict(self, X: np.ndarray) -> float:
        """Predict probability of price going up."""
        if not HAS_TENSORFLOW or self.model is None:
            return 0.5
        
        # Need last `lookback` samples
        if len(X) < self.lookback:
            return 0.5
            
        X_seq = X[-self.lookback:].reshape(1, self.lookback, -1)
        prob = self.model.predict(X_seq, verbose=0)[0][0]
        return float(prob)


class XGBoostModel:
    """XGBoost model for tabular prediction."""
    
    def __init__(self):
        self.model = None
        self.feature_importance = {}
        
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
        """Train XGBoost model."""
        if not HAS_XGBOOST:
            return {'accuracy': 0.0}
        
        # Split data (time-series aware)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        
        self.model.fit(X_train, y_train)
        
        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(feature_names, importance))
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {'accuracy': accuracy}
    
    def predict(self, X: np.ndarray) -> float:
        """Predict probability of price going up."""
        if not HAS_XGBOOST or self.model is None:
            return 0.5
            
        prob = self.model.predict_proba(X[-1:])
        return float(prob[0][1])  # Probability of class 1 (up)
    
    def get_top_features(self, n: int = 5) -> list:
        """Get top N most important features."""
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]


class LightGBMModel:
    """LightGBM model for fast prediction."""
    
    def __init__(self):
        self.model = None
        self.feature_importance = {}
        
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
        """Train LightGBM model."""
        if not HAS_LIGHTGBM:
            return {'accuracy': 0.0}
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(feature_names, importance))
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {'accuracy': accuracy}
    
    def predict(self, X: np.ndarray) -> float:
        """Predict probability of price going up."""
        if not HAS_LIGHTGBM or self.model is None:
            return 0.5
            
        prob = self.model.predict_proba(X[-1:])
        return float(prob[0][1])


class EnsemblePredictor:
    """
    Ensemble of LSTM, XGBoost, and LightGBM for stock direction prediction.
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.lstm = LSTMModel(lookback=60)
        self.xgboost = XGBoostModel()
        self.lightgbm = LightGBMModel()
        
        # Ensemble weights (can be tuned based on validation performance)
        self.weights = {
            'lstm': 0.35,
            'xgboost': 0.35,
            'lightgbm': 0.30
        }
        
        self.is_trained = False
        self.training_accuracy = {}
        
    def train(self, df: pd.DataFrame, target_days: int = 5) -> dict:
        """
        Train all models in the ensemble.
        
        Args:
            df: DataFrame with OHLCV data (at least 1 year)
            target_days: Days ahead to predict
            
        Returns:
            Dictionary with training metrics
        """
        print("Preparing features...")
        X, y, feature_names = self.feature_engineer.prepare_data_for_training(
            df, 
            target_days=target_days,
            binary_target=True
        )
        
        print(f"Training data shape: {X.shape}")
        print(f"Features: {len(feature_names)}")
        print(f"Positive class ratio: {y.mean():.2%}")
        
        results = {}
        
        # Train LSTM
        if HAS_TENSORFLOW:
            print("Training LSTM...")
            lstm_result = self.lstm.train(X, y)
            results['lstm'] = lstm_result
            print(f"  LSTM Accuracy: {lstm_result['accuracy']:.2%}")
        
        # Train XGBoost
        if HAS_XGBOOST:
            print("Training XGBoost...")
            xgb_result = self.xgboost.train(X, y, feature_names)
            results['xgboost'] = xgb_result
            print(f"  XGBoost Accuracy: {xgb_result['accuracy']:.2%}")
        
        # Train LightGBM
        if HAS_LIGHTGBM:
            print("Training LightGBM...")
            lgb_result = self.lightgbm.train(X, y, feature_names)
            results['lightgbm'] = lgb_result
            print(f"  LightGBM Accuracy: {lgb_result['accuracy']:.2%}")
        
        self.is_trained = True
        self.training_accuracy = results
        
        # Calculate ensemble accuracy (weighted average)
        total_weight = 0
        weighted_acc = 0
        for model, weight in self.weights.items():
            if model in results and results[model]['accuracy'] > 0:
                weighted_acc += weight * results[model]['accuracy']
                total_weight += weight
        
        if total_weight > 0:
            results['ensemble'] = {'accuracy': weighted_acc / total_weight}
            print(f"\nEnsemble Accuracy: {results['ensemble']['accuracy']:.2%}")
        
        return results
    
    def predict(self, df: pd.DataFrame, ticker: str) -> PredictionResult:
        """
        Make prediction using ensemble.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker symbol
            
        Returns:
            PredictionResult with direction and confidence
        """
        # Prepare features
        data = self.feature_engineer.create_features(df)
        data = data.dropna()
        
        feature_cols = [col for col in self.feature_engineer.feature_names if col in data.columns]
        X = data[feature_cols].values
        X = self.feature_engineer.scaler.transform(X)
        
        # Get predictions from each model
        predictions = {}
        
        if HAS_TENSORFLOW and self.lstm.model is not None:
            predictions['lstm'] = self.lstm.predict(X)
        
        if HAS_XGBOOST and self.xgboost.model is not None:
            predictions['xgboost'] = self.xgboost.predict(X)
        
        if HAS_LIGHTGBM and self.lightgbm.model is not None:
            predictions['lightgbm'] = self.lightgbm.predict(X)
        
        # Ensemble prediction (weighted average)
        if not predictions:
            return PredictionResult(
                ticker=ticker,
                direction="NEUTRAL",
                confidence=0.0,
                predicted_change_pct=0.0
            )
        
        weighted_prob = 0
        total_weight = 0
        for model, prob in predictions.items():
            weight = self.weights.get(model, 0.33)
            weighted_prob += weight * prob
            total_weight += weight
        
        final_prob = weighted_prob / total_weight if total_weight > 0 else 0.5
        
        # Determine direction and confidence
        if final_prob > 0.6:
            direction = "UP"
            confidence = (final_prob - 0.5) * 2  # Scale to 0-1
        elif final_prob < 0.4:
            direction = "DOWN"
            confidence = (0.5 - final_prob) * 2
        else:
            direction = "NEUTRAL"
            confidence = 1 - abs(final_prob - 0.5) * 2
        
        # Estimate % change based on recent volatility
        recent_volatility = data['volatility_20'].iloc[-1] if 'volatility_20' in data.columns else 0.02
        predicted_change = (final_prob - 0.5) * recent_volatility * 100 * 5  # 5-day estimate
        
        # Get top features
        top_features = self.xgboost.get_top_features(5) if HAS_XGBOOST else []
        
        return PredictionResult(
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            predicted_change_pct=predicted_change,
            lstm_prediction=predictions.get('lstm'),
            xgboost_prediction=predictions.get('xgboost'),
            lightgbm_prediction=predictions.get('lightgbm'),
            ensemble_accuracy=self.training_accuracy.get('ensemble', {}).get('accuracy', 0),
            top_features=top_features
        )


# Quick test
if __name__ == "__main__":
    import yfinance as yf
    
    print("="*60)
    print("Testing Ensemble Predictor")
    print("="*60)
    
    # Download data
    ticker = "AAPL"
    print(f"\nDownloading {ticker} data...")
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")
    print(f"Downloaded {len(df)} days of data")
    
    # Train ensemble
    ensemble = EnsemblePredictor()
    results = ensemble.train(df, target_days=5)
    
    # Make prediction
    print("\n" + "="*60)
    print("Making Prediction")
    print("="*60)
    prediction = ensemble.predict(df, ticker)
    
    print(f"\nTicker: {prediction.ticker}")
    print(f"Direction: {prediction.direction}")
    print(f"Confidence: {prediction.confidence:.2%}")
    print(f"Predicted Change: {prediction.predicted_change_pct:+.2f}%")
    print(f"\nModel Predictions:")
    print(f"  LSTM: {prediction.lstm_prediction:.2%}" if prediction.lstm_prediction else "  LSTM: N/A")
    print(f"  XGBoost: {prediction.xgboost_prediction:.2%}" if prediction.xgboost_prediction else "  XGBoost: N/A")
    print(f"  LightGBM: {prediction.lightgbm_prediction:.2%}" if prediction.lightgbm_prediction else "  LightGBM: N/A")
    
    if prediction.top_features:
        print(f"\nTop Features:")
        for feat, imp in prediction.top_features:
            print(f"  {feat}: {imp:.4f}")