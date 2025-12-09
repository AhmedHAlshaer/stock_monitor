"""
Regression Analysis Module
Handles trend analysis and Mon-Thu prediction using regression models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TrendAnalysis:
    """Results of trend analysis"""
    ticker: str
    trend_direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    trend_strength: float  # 0-100 scale
    predicted_change_percent: float  # Expected % change Mon-Thu
    confidence: float  # R² score
    support_level: float
    resistance_level: float
    recommendation: str  # "BUY", "SELL", "HOLD"
    reasoning: str
    
    def to_discord_message(self) -> str:
        """Format for Discord display"""
        emoji_map = {
            "BULLISH": "🟢",
            "BEARISH": "🔴", 
            "NEUTRAL": "🟡"
        }
        rec_emoji = {"BUY": "💰", "SELL": "🚪", "HOLD": "⏸️"}
        
        return (
            f"## {self.ticker} Weekly Analysis\n"
            f"**Trend:** {emoji_map.get(self.trend_direction, '⚪')} {self.trend_direction} "
            f"(Strength: {self.trend_strength:.0f}/100)\n"
            f"**Mon-Thu Forecast:** {self.predicted_change_percent:+.2f}%\n"
            f"**Model Confidence:** {self.confidence:.1%}\n"
            f"**Support:** ${self.support_level:.2f} | **Resistance:** ${self.resistance_level:.2f}\n"
            f"**Recommendation:** {rec_emoji.get(self.recommendation, '❓')} {self.recommendation}\n"
            f"*{self.reasoning}*"
        )


class RegressionAnalyzer:
    """Performs regression analysis for stock trend prediction."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for regression from OHLCV data.
        
        Features include:
        - Price momentum (various windows)
        - Volume trends
        - Volatility measures
        - Day of week patterns
        """
        data = df.copy()
        
        # Basic price features
        data['returns'] = data['Close'].pct_change()
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            data[f'sma_{window}'] = data['Close'].rolling(window=window).mean()
            data[f'price_vs_sma_{window}'] = (data['Close'] - data[f'sma_{window}']) / data[f'sma_{window}']
        
        # Momentum indicators
        data['momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
        data['momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
        data['momentum_20'] = data['Close'] / data['Close'].shift(20) - 1
        
        # Volatility
        data['volatility_5'] = data['returns'].rolling(window=5).std()
        data['volatility_20'] = data['returns'].rolling(window=20).std()
        
        # Volume features
        data['volume_sma_20'] = data['Volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_sma_20']
        
        # Range features
        data['daily_range'] = (data['High'] - data['Low']) / data['Close']
        data['range_sma_10'] = data['daily_range'].rolling(window=10).mean()
        
        # RSI-like momentum
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Day of week (for weekly patterns)
        data['day_of_week'] = pd.to_datetime(data.index).dayofweek
        
        return data.dropna()
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> tuple[float, float]:
        """Calculate recent support and resistance levels."""
        recent = df.tail(window)
        support = recent['Low'].min()
        resistance = recent['High'].max()
        return support, resistance
    
    def predict_weekly_trend(self, df: pd.DataFrame, ticker: str) -> Optional[TrendAnalysis]:
        """
        Predict Mon-Thu trend using regression.
        
        Uses historical patterns to predict expected price movement
        for the upcoming Monday to Thursday period.
        """
        if len(df) < 60:  # Need at least 60 days of data
            return None
            
        try:
            # Prepare features
            data = self.prepare_features(df)
            
            # Target: Next 4-day return (Mon-Thu approximation)
            data['target'] = data['Close'].shift(-4) / data['Close'] - 1
            data = data.dropna()
            
            if len(data) < 30:
                return None
            
            # Feature columns
            feature_cols = [
                'momentum_5', 'momentum_10', 'momentum_20',
                'price_vs_sma_5', 'price_vs_sma_20',
                'volatility_5', 'volatility_20',
                'volume_ratio', 'rsi', 'daily_range'
            ]
            
            X = data[feature_cols].values
            y = data['target'].values
            
            # Train/test split (use last 20% for validation)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model (Ridge for stability)
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            r2 = max(0, r2_score(y_test, y_pred))  # Clip negative R²
            
            # Predict next week
            latest_features = data[feature_cols].iloc[-1:].values
            latest_scaled = self.scaler.transform(latest_features)
            predicted_return = model.predict(latest_scaled)[0]
            
            # Calculate trend metrics
            recent_momentum = data['momentum_20'].iloc[-1]
            trend_strength = min(100, abs(recent_momentum) * 500)  # Scale to 0-100
            
            if predicted_return > 0.01:
                trend_direction = "BULLISH"
            elif predicted_return < -0.01:
                trend_direction = "BEARISH"
            else:
                trend_direction = "NEUTRAL"
            
            # Support/Resistance
            support, resistance = self.calculate_support_resistance(df)
            current_price = df['Close'].iloc[-1]
            
            # Generate recommendation
            recommendation, reasoning = self._generate_recommendation(
                trend_direction=trend_direction,
                predicted_return=predicted_return,
                confidence=r2,
                rsi=data['rsi'].iloc[-1],
                price_vs_support=(current_price - support) / support,
                price_vs_resistance=(resistance - current_price) / current_price
            )
            
            return TrendAnalysis(
                ticker=ticker,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                predicted_change_percent=predicted_return * 100,
                confidence=r2,
                support_level=support,
                resistance_level=resistance,
                recommendation=recommendation,
                reasoning=reasoning
            )
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            return None
    
    def _generate_recommendation(
        self,
        trend_direction: str,
        predicted_return: float,
        confidence: float,
        rsi: float,
        price_vs_support: float,
        price_vs_resistance: float
    ) -> tuple[str, str]:
        """Generate buy/sell/hold recommendation with reasoning."""
        
        reasons = []
        
        # Low confidence = HOLD
        if confidence < 0.1:
            return "HOLD", "Model confidence too low for reliable prediction."
        
        # RSI extremes
        if rsi < 30:
            reasons.append("RSI indicates oversold conditions")
        elif rsi > 70:
            reasons.append("RSI indicates overbought conditions")
        
        # Near support/resistance
        if price_vs_support < 0.03:
            reasons.append("Price near support level")
        if price_vs_resistance < 0.03:
            reasons.append("Price near resistance level")
        
        # Trend-based recommendation
        if trend_direction == "BULLISH" and predicted_return > 0.02:
            if rsi < 70:
                rec = "BUY"
                reasons.append(f"Bullish trend with {predicted_return*100:.1f}% predicted upside")
            else:
                rec = "HOLD"
                reasons.append("Bullish but overbought - wait for pullback")
                
        elif trend_direction == "BEARISH" and predicted_return < -0.02:
            if rsi > 30:
                rec = "SELL"
                reasons.append(f"Bearish trend with {abs(predicted_return)*100:.1f}% predicted downside")
            else:
                rec = "HOLD"
                reasons.append("Bearish but oversold - potential bounce")
        else:
            rec = "HOLD"
            reasons.append("No strong directional signal")
        
        reasoning = ". ".join(reasons) + "."
        return rec, reasoning


# Quick test
if __name__ == "__main__":
    import yfinance as yf
    
    # Test with Apple
    ticker = "AAPL"
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    
    analyzer = RegressionAnalyzer()
    result = analyzer.predict_weekly_trend(df, ticker)
    
    if result:
        print(result.to_discord_message())
    else:
        print("Analysis failed")
