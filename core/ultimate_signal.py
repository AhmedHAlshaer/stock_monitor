"""
Ultimate Signal Generator
Combines ML models, sentiment, analysts, insiders, earnings, and seasonality
for the most comprehensive stock analysis possible.
"""

import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import json

import yfinance as yf
import pandas as pd
import numpy as np

# Import our modules
from core.news_sentiment import (
    NewsAnalyzer, MarketDataAnalyzer, 
    NewsSentiment, EarningsInfo, AnalystRatings, InsiderActivity
)
from core.historical_patterns import HistoricalAnalyzer, HistoricalContext
from core.ml_models import EnsemblePredictor, PredictionResult, HAS_TENSORFLOW, HAS_XGBOOST, HAS_LIGHTGBM


@dataclass
class UltimateSignal:
    """The ultimate combined trading signal."""
    ticker: str
    timestamp: datetime
    
    # Final signal
    signal: str  # "STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"
    score: float  # -2 to +2
    confidence: float  # 0 to 1
    
    # Component scores (all -1 to 1)
    ml_score: float = 0.0
    sentiment_score: float = 0.0
    analyst_score: float = 0.0
    insider_score: float = 0.0
    earnings_score: float = 0.0
    seasonality_score: float = 0.0
    technical_score: float = 0.0
    
    # ML model details
    ml_direction: str = "NEUTRAL"
    ml_confidence: float = 0.0
    lstm_pred: Optional[float] = None
    xgb_pred: Optional[float] = None
    lgb_pred: Optional[float] = None
    
    # Earnings details
    earnings_beat_rate: float = 0.0
    earnings_streak: int = 0
    days_until_earnings: Optional[int] = None
    
    # Seasonality details
    current_month_avg: float = 0.0
    current_month_win_rate: float = 0.0
    
    # Analyst details
    analyst_rating: str = "HOLD"
    price_target: Optional[float] = None
    upside_pct: Optional[float] = None
    num_analysts: int = 0
    
    # Key insights
    bullish_factors: List[str] = field(default_factory=list)
    bearish_factors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # News summary
    news_summary: str = ""
    news_count: int = 0
    
    def get_signal_emoji(self) -> str:
        return {
            "STRONG BUY": "🚀",
            "BUY": "🟢",
            "HOLD": "🟡",
            "SELL": "🔴",
            "STRONG SELL": "💀"
        }.get(self.signal, "⚪")
    
    def to_summary(self) -> str:
        """Generate a text summary."""
        lines = [
            f"{self.get_signal_emoji()} {self.ticker} - {self.signal}",
            f"Score: {self.score:+.2f} | Confidence: {self.confidence:.0%}",
            "",
            "Component Scores:",
            f"  ML Models: {self.ml_score:+.2f} ({self.ml_direction})",
            f"  Sentiment: {self.sentiment_score:+.2f}",
            f"  Analysts: {self.analyst_score:+.2f}",
            f"  Insiders: {self.insider_score:+.2f}",
            f"  Earnings: {self.earnings_score:+.2f}",
            f"  Seasonality: {self.seasonality_score:+.2f}",
        ]
        
        if self.bullish_factors:
            lines.append("")
            lines.append("🟢 Bullish:")
            for f in self.bullish_factors[:5]:
                lines.append(f"  • {f}")
        
        if self.bearish_factors:
            lines.append("")
            lines.append("🔴 Bearish:")
            for f in self.bearish_factors[:5]:
                lines.append(f"  • {f}")
        
        if self.warnings:
            lines.append("")
            lines.append("⚠️ Warnings:")
            for w in self.warnings:
                lines.append(f"  • {w}")
        
        return "\n".join(lines)


class UltimateSignalGenerator:
    """
    The Ultimate Stock Signal Generator.
    
    Combines:
    - ML Ensemble (LSTM + XGBoost + LightGBM)
    - News Sentiment (DeepSeek LLM)
    - Analyst Ratings
    - Insider Activity
    - Earnings History
    - Seasonal Patterns
    - Technical Indicators
    """
    
    def __init__(self, deepseek_api_key: Optional[str] = None, use_ml: bool = True):
        """
        Initialize the signal generator.
        
        Args:
            deepseek_api_key: API key for DeepSeek sentiment analysis
            use_ml: Whether to use ML models (requires tensorflow, xgboost, lightgbm)
        """
        # Initialize components
        self.news_analyzer = NewsAnalyzer(api_key=deepseek_api_key)
        self.market_analyzer = MarketDataAnalyzer()
        self.historical_analyzer = HistoricalAnalyzer()
        
        # ML predictor (lazy initialization)
        self.use_ml = use_ml and (HAS_TENSORFLOW or HAS_XGBOOST or HAS_LIGHTGBM)
        self.ml_predictor = None
        self.ml_trained_tickers = set()
        
        # Component weights
        self.weights = {
            'ml': 0.25,          # ML ensemble prediction
            'sentiment': 0.20,   # News sentiment
            'analyst': 0.15,     # Analyst ratings
            'insider': 0.10,     # Insider activity
            'earnings': 0.15,    # Earnings history
            'seasonality': 0.10, # Seasonal patterns
            'technical': 0.05    # Basic technical indicators
        }
        
        print(f"Ultimate Signal Generator initialized")
        print(f"  ML Models: {'Enabled' if self.use_ml else 'Disabled'}")
        print(f"  DeepSeek: {'Enabled' if deepseek_api_key else 'Disabled (keyword fallback)'}")
    
    def _train_ml_if_needed(self, ticker: str, df: pd.DataFrame) -> None:
        """Train ML models if not already trained for this ticker."""
        if not self.use_ml:
            return
        
        if self.ml_predictor is None:
            self.ml_predictor = EnsemblePredictor()
        
        if ticker not in self.ml_trained_tickers:
            print(f"Training ML models for {ticker}...")
            try:
                self.ml_predictor.train(df, target_days=5)
                self.ml_trained_tickers.add(ticker)
            except Exception as e:
                print(f"ML training failed: {e}")
    
    def _calculate_technical_score(self, df: pd.DataFrame) -> Tuple[float, List[str], List[str]]:
        """Calculate basic technical score without ML."""
        bullish = []
        bearish = []
        
        if df.empty or len(df) < 50:
            return 0.0, bullish, bearish
        
        close = df['Close'].iloc[-1]
        
        # SMA crossovers
        sma_20 = df['Close'].rolling(20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(50).mean().iloc[-1]
        
        if close > sma_20 > sma_50:
            bullish.append("Price above rising moving averages")
        elif close < sma_20 < sma_50:
            bearish.append("Price below falling moving averages")
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        if rsi < 30:
            bullish.append(f"RSI oversold ({rsi:.0f})")
        elif rsi > 70:
            bearish.append(f"RSI overbought ({rsi:.0f})")
        
        # Momentum
        momentum_20 = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
        if momentum_20 > 10:
            bullish.append(f"Strong 20-day momentum (+{momentum_20:.1f}%)")
        elif momentum_20 < -10:
            bearish.append(f"Weak 20-day momentum ({momentum_20:.1f}%)")
        
        # Volume trend
        vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
        vol_current = df['Volume'].iloc[-1]
        if vol_current > vol_avg * 1.5:
            if df['Close'].iloc[-1] > df['Close'].iloc[-2]:
                bullish.append("High volume on up day")
            else:
                bearish.append("High volume on down day")
        
        # Calculate score
        score = (len(bullish) - len(bearish)) / 4  # Normalize to roughly -1 to 1
        score = max(-1, min(1, score))
        
        return score, bullish, bearish
    
    def generate_signal(self, ticker: str) -> UltimateSignal:
        """
        Generate the ultimate trading signal.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            UltimateSignal with comprehensive analysis
        """
        ticker = ticker.upper()
        print(f"\n{'='*60}")
        print(f"Generating Ultimate Signal for {ticker}")
        print('='*60)
        
        # Fetch price data
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        
        if df.empty:
            raise ValueError(f"Could not fetch data for {ticker}")
        
        current_price = df['Close'].iloc[-1]
        
        # Initialize result
        result = UltimateSignal(
            ticker=ticker,
            timestamp=datetime.now(),
            signal="HOLD",
            score=0.0,
            confidence=0.0
        )
        
        all_bullish = []
        all_bearish = []
        all_warnings = []
        
        # ==================
        # 1. ML PREDICTION
        # ==================
        print("\n📊 ML Models...")
        if self.use_ml:
            try:
                self._train_ml_if_needed(ticker, df)
                ml_result = self.ml_predictor.predict(df, ticker)
                
                result.ml_direction = ml_result.direction
                result.ml_confidence = ml_result.confidence
                result.lstm_pred = ml_result.lstm_prediction
                result.xgb_pred = ml_result.xgboost_prediction
                result.lgb_pred = ml_result.lightgbm_prediction
                
                # Convert to score
                if ml_result.direction == "UP":
                    result.ml_score = ml_result.confidence
                    all_bullish.append(f"ML models predict UP ({ml_result.confidence:.0%} confidence)")
                elif ml_result.direction == "DOWN":
                    result.ml_score = -ml_result.confidence
                    all_bearish.append(f"ML models predict DOWN ({ml_result.confidence:.0%} confidence)")
                else:
                    result.ml_score = 0.0
                
                print(f"   Direction: {ml_result.direction} ({ml_result.confidence:.0%})")
            except Exception as e:
                print(f"   ML prediction failed: {e}")
                result.ml_score = 0.0
        else:
            print("   Skipped (not available)")
        
        # ==================
        # 2. NEWS SENTIMENT
        # ==================
        print("\n📰 News Sentiment...")
        try:
            articles = self.news_analyzer.fetch_news(ticker)
            sentiment = self.news_analyzer.analyze_sentiment(ticker, articles)
            
            result.sentiment_score = sentiment.sentiment_score
            result.news_summary = sentiment.analysis_summary
            result.news_count = sentiment.news_count
            
            if sentiment.sentiment_score > 0.3:
                all_bullish.append(f"Positive news sentiment ({sentiment.overall_sentiment})")
            elif sentiment.sentiment_score < -0.3:
                all_bearish.append(f"Negative news sentiment ({sentiment.overall_sentiment})")
            
            if sentiment.bullish_factors:
                all_bullish.extend(sentiment.bullish_factors[:2])
            if sentiment.bearish_factors:
                all_bearish.extend(sentiment.bearish_factors[:2])
            
            print(f"   Sentiment: {sentiment.overall_sentiment} ({sentiment.sentiment_score:+.2f})")
        except Exception as e:
            print(f"   Sentiment failed: {e}")
        
        # ==================
        # 3. ANALYST RATINGS
        # ==================
        print("\n🎯 Analyst Ratings...")
        try:
            analyst = self.market_analyzer.get_analyst_ratings(ticker)
            
            result.analyst_rating = analyst.recommendation
            result.price_target = analyst.target_price
            result.upside_pct = analyst.upside_percent
            result.num_analysts = analyst.num_analysts
            
            # Convert to score
            analyst_score = {"BUY": 0.7, "HOLD": 0.0, "SELL": -0.7}.get(analyst.recommendation, 0.0)
            if analyst.upside_percent:
                if analyst.upside_percent > 20:
                    analyst_score += 0.3
                elif analyst.upside_percent < -10:
                    analyst_score -= 0.3
            result.analyst_score = max(-1, min(1, analyst_score))
            
            if analyst.recommendation == "BUY":
                target_str = f" (target ${analyst.target_price:.0f})" if analyst.target_price else ""
                all_bullish.append(f"Analysts bullish{target_str}")
            elif analyst.recommendation == "SELL":
                all_bearish.append("Analysts bearish")
            
            print(f"   Rating: {analyst.recommendation} ({analyst.num_analysts} analysts)")
            if analyst.target_price:
                print(f"   Target: ${analyst.target_price:.2f} ({analyst.upside_percent:+.1f}%)")
        except Exception as e:
            print(f"   Analyst data failed: {e}")
        
        # ==================
        # 4. INSIDER ACTIVITY
        # ==================
        print("\n👔 Insider Activity...")
        try:
            insider = self.market_analyzer.get_insider_activity(ticker)
            
            insider_score = {"BUYING": 0.5, "NEUTRAL": 0.0, "SELLING": -0.3}.get(insider.signal, 0.0)
            result.insider_score = insider_score
            
            if insider.signal == "BUYING":
                all_bullish.append(f"Insider buying (${insider.net_value_3m/1e6:.1f}M net)")
            elif insider.signal == "SELLING":
                all_bearish.append("Insider selling activity")
            
            print(f"   Activity: {insider.signal}")
        except Exception as e:
            print(f"   Insider data failed: {e}")
        
        # ==================
        # 5. EARNINGS HISTORY
        # ==================
        print("\n📊 Earnings History...")
        try:
            earnings = self.historical_analyzer.analyze_earnings_history(ticker)
            
            result.earnings_beat_rate = earnings.beat_rate
            result.earnings_streak = earnings.current_streak
            result.days_until_earnings = earnings.days_until_earnings
            result.earnings_score = earnings.to_signal_score()
            
            if earnings.beat_rate > 0.7:
                all_bullish.append(f"Strong earnings track record ({earnings.beat_rate:.0%} beat rate)")
            elif earnings.beat_rate < 0.4:
                all_bearish.append(f"Weak earnings history ({earnings.beat_rate:.0%} beat rate)")
            
            if earnings.current_streak >= 4:
                all_bullish.append(f"🔥 {earnings.current_streak} quarter beat streak")
            elif earnings.current_streak <= -3:
                all_bearish.append(f"❄️ {abs(earnings.current_streak)} quarter miss streak")
            
            if earnings.days_until_earnings and earnings.days_until_earnings <= 14:
                all_warnings.append(f"Earnings in {earnings.days_until_earnings} days")
            
            print(f"   Beat Rate: {earnings.beat_rate:.0%}")
            print(f"   Streak: {earnings.current_streak}")
        except Exception as e:
            print(f"   Earnings analysis failed: {e}")
        
        # ==================
        # 6. SEASONALITY
        # ==================
        print("\n📅 Seasonality...")
        try:
            seasonality = self.historical_analyzer.analyze_seasonality(ticker)
            
            result.current_month_avg = seasonality.current_month_avg
            result.current_month_win_rate = seasonality.current_month_win_rate
            result.seasonality_score = seasonality.to_signal_score()
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_name = month_names[seasonality.current_month - 1]
            
            if seasonality.current_month_avg > 2 and seasonality.current_month_win_rate > 0.6:
                all_bullish.append(f"Historically strong month ({month_name}: +{seasonality.current_month_avg:.1f}%)")
            elif seasonality.current_month_avg < -2 and seasonality.current_month_win_rate < 0.4:
                all_bearish.append(f"Historically weak month ({month_name}: {seasonality.current_month_avg:.1f}%)")
            
            print(f"   {month_name}: {seasonality.current_month_avg:+.1f}% avg ({seasonality.current_month_win_rate:.0%} win rate)")
        except Exception as e:
            print(f"   Seasonality analysis failed: {e}")
        
        # ==================
        # 7. TECHNICAL INDICATORS
        # ==================
        print("\n📈 Technical Analysis...")
        try:
            tech_score, tech_bullish, tech_bearish = self._calculate_technical_score(df)
            result.technical_score = tech_score
            all_bullish.extend(tech_bullish)
            all_bearish.extend(tech_bearish)
            print(f"   Score: {tech_score:+.2f}")
        except Exception as e:
            print(f"   Technical analysis failed: {e}")
        
        # ==================
        # COMBINE ALL SCORES
        # ==================
        print("\n" + "="*60)
        print("COMBINING SIGNALS")
        print("="*60)
        
        # Weighted combination
        combined_score = (
            self.weights['ml'] * result.ml_score +
            self.weights['sentiment'] * result.sentiment_score +
            self.weights['analyst'] * result.analyst_score +
            self.weights['insider'] * result.insider_score +
            self.weights['earnings'] * result.earnings_score +
            self.weights['seasonality'] * result.seasonality_score +
            self.weights['technical'] * result.technical_score
        )
        
        # Scale to -2 to 2 range
        combined_score *= 2
        
        result.score = combined_score
        result.bullish_factors = all_bullish
        result.bearish_factors = all_bearish
        result.warnings = all_warnings
        
        # Determine signal
        if combined_score >= 0.7:
            result.signal = "STRONG BUY"
        elif combined_score >= 0.3:
            result.signal = "BUY"
        elif combined_score <= -0.7:
            result.signal = "STRONG SELL"
        elif combined_score <= -0.3:
            result.signal = "SELL"
        else:
            result.signal = "HOLD"
        
        # Calculate confidence
        # Higher when signals agree, lower when mixed
        signal_signs = [
            np.sign(result.ml_score) if result.ml_score != 0 else 0,
            np.sign(result.sentiment_score) if result.sentiment_score != 0 else 0,
            np.sign(result.analyst_score) if result.analyst_score != 0 else 0,
            np.sign(result.earnings_score) if result.earnings_score != 0 else 0,
            np.sign(result.seasonality_score) if result.seasonality_score != 0 else 0,
        ]
        non_zero = [s for s in signal_signs if s != 0]
        if non_zero:
            agreement = abs(sum(non_zero)) / len(non_zero)
        else:
            agreement = 0.5
        
        result.confidence = min(0.90, 0.4 + agreement * 0.4)
        
        # Reduce confidence near earnings
        if result.days_until_earnings and result.days_until_earnings <= 7:
            result.confidence *= 0.7
        
        print(f"\nFinal Score: {result.score:+.2f}")
        print(f"Signal: {result.signal}")
        print(f"Confidence: {result.confidence:.0%}")
        
        return result


# Test
if __name__ == "__main__":
    # Initialize generator
    generator = UltimateSignalGenerator(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        use_ml=True
    )
    
    # Test with a few stocks
    tickers = ["NVDA", "AAPL"]
    
    for ticker in tickers:
        try:
            signal = generator.generate_signal(ticker)
            print("\n" + "="*60)
            print("FINAL REPORT")
            print("="*60)
            print(signal.to_summary())
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
