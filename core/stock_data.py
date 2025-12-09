"""
Stock Data Fetcher Module
Handles fetching historical stock data and calculating 52-week metrics.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass


@dataclass
class StockSignal:
    """Represents a stock signal (high/low alert)"""
    ticker: str
    signal_type: str  # "52_WEEK_HIGH" or "52_WEEK_LOW"
    current_price: float
    threshold_price: float  # The 52-week high or low value
    percent_from_threshold: float
    timestamp: datetime
    
    def __str__(self):
        emoji = "🚀" if self.signal_type == "52_WEEK_HIGH" else "📉"
        return (
            f"{emoji} **{self.ticker}** hit a {self.signal_type.replace('_', ' ')}!\n"
            f"Current: ${self.current_price:.2f} | "
            f"52W {'High' if 'HIGH' in self.signal_type else 'Low'}: ${self.threshold_price:.2f}\n"
            f"({self.percent_from_threshold:+.2f}% from threshold)"
        )


class StockDataFetcher:
    """Fetches and analyzes stock data for 52-week signals."""
    
    def __init__(self, tickers: list[str]):
        """
        Initialize with a list of stock tickers to monitor.
        
        Args:
            tickers: List of stock symbols (e.g., ["AAPL", "GOOGL", "MSFT"])
        """
        self.tickers = [t.upper() for t in tickers]
        self._cache = {}  # Cache to avoid repeated API calls
        
    def fetch_52_week_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch 52 weeks of historical data for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        try:
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=52)
            
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"Warning: No data returned for {ticker}")
                return None
                
            self._cache[ticker] = {
                'data': df,
                'fetched_at': datetime.now()
            }
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def get_52_week_range(self, ticker: str) -> Optional[dict]:
        """
        Calculate the 52-week high and low for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with high, low, current price, and percentages
        """
        df = self.fetch_52_week_data(ticker)
        if df is None or df.empty:
            return None
            
        high_52w = df['High'].max()
        low_52w = df['Low'].min()
        current = df['Close'].iloc[-1]
        
        return {
            'ticker': ticker,
            'current_price': current,
            '52_week_high': high_52w,
            '52_week_low': low_52w,
            'percent_from_high': ((current - high_52w) / high_52w) * 100,
            'percent_from_low': ((current - low_52w) / low_52w) * 100,
            'range_position': ((current - low_52w) / (high_52w - low_52w)) * 100  # 0-100%
        }
    
    def check_signals(self, threshold_percent: float = 2.0) -> list[StockSignal]:
        """
        Check all tickers for 52-week high/low signals.
        
        Args:
            threshold_percent: How close to high/low to trigger (default 2%)
            
        Returns:
            List of StockSignal objects for triggered alerts
        """
        signals = []
        
        for ticker in self.tickers:
            metrics = self.get_52_week_range(ticker)
            if metrics is None:
                continue
            
            # Check for 52-week high (within threshold or above)
            if metrics['percent_from_high'] >= -threshold_percent:
                signals.append(StockSignal(
                    ticker=ticker,
                    signal_type="52_WEEK_HIGH",
                    current_price=metrics['current_price'],
                    threshold_price=metrics['52_week_high'],
                    percent_from_threshold=metrics['percent_from_high'],
                    timestamp=datetime.now()
                ))
            
            # Check for 52-week low (within threshold or below)
            if metrics['percent_from_low'] <= threshold_percent:
                signals.append(StockSignal(
                    ticker=ticker,
                    signal_type="52_WEEK_LOW",
                    current_price=metrics['current_price'],
                    threshold_price=metrics['52_week_low'],
                    percent_from_threshold=metrics['percent_from_low'],
                    timestamp=datetime.now()
                ))
        
        return signals
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all monitored stocks.
        
        Returns:
            DataFrame with 52-week metrics for all tickers
        """
        summaries = []
        for ticker in self.tickers:
            metrics = self.get_52_week_range(ticker)
            if metrics:
                summaries.append(metrics)
        
        if not summaries:
            return pd.DataFrame()
            
        return pd.DataFrame(summaries)


# Quick test
if __name__ == "__main__":
    # Test with some popular stocks
    fetcher = StockDataFetcher(["AAPL", "GOOGL", "MSFT", "TSLA"])
    
    print("=== 52-Week Summary ===")
    summary = fetcher.get_summary()
    print(summary.to_string())
    
    print("\n=== Checking Signals (5% threshold) ===")
    signals = fetcher.check_signals(threshold_percent=5.0)
    for signal in signals:
        print(signal)
        print()
