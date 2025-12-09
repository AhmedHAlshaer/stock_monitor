"""
Historical Patterns Module
Analyzes earnings history, seasonality, and past event reactions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import yfinance as yf


@dataclass
class EarningsHistory:
    """Historical earnings performance"""
    ticker: str
    
    # Beat/miss record
    total_quarters: int = 0
    beats: int = 0
    misses: int = 0
    meets: int = 0
    beat_rate: float = 0.0
    
    # Average surprise
    avg_surprise_pct: float = 0.0
    
    # Post-earnings moves
    avg_move_after_earnings: float = 0.0
    avg_move_on_beat: float = 0.0
    avg_move_on_miss: float = 0.0
    
    # Streak
    current_streak: int = 0  # Positive = beats, negative = misses
    streak_type: str = ""  # "BEAT_STREAK", "MISS_STREAK", "NONE"
    
    # Next earnings
    days_until_earnings: Optional[int] = None
    
    def to_signal_score(self) -> float:
        """Convert earnings history to a signal score (-1 to 1)"""
        score = 0.0
        
        # Beat rate contribution (0 to 0.4)
        if self.total_quarters > 0:
            score += (self.beat_rate - 0.5) * 0.8  # -0.4 to +0.4
        
        # Streak contribution (0 to 0.3)
        if self.current_streak >= 4:
            score += 0.3
        elif self.current_streak >= 2:
            score += 0.15
        elif self.current_streak <= -3:
            score -= 0.3
        elif self.current_streak <= -2:
            score -= 0.15
        
        # Average surprise contribution (0 to 0.3)
        if self.avg_surprise_pct > 5:
            score += 0.3
        elif self.avg_surprise_pct > 2:
            score += 0.15
        elif self.avg_surprise_pct < -5:
            score -= 0.3
        elif self.avg_surprise_pct < -2:
            score -= 0.15
        
        return max(-1, min(1, score))
    
    def to_discord_field(self) -> dict:
        emoji = "🟢" if self.beat_rate > 0.6 else "🔴" if self.beat_rate < 0.4 else "🟡"
        
        streak_text = ""
        if self.current_streak >= 2:
            streak_text = f"🔥 {self.current_streak} beat streak"
        elif self.current_streak <= -2:
            streak_text = f"❄️ {abs(self.current_streak)} miss streak"
        
        value = f"{emoji} **{self.beat_rate:.0%}** beat rate ({self.beats}/{self.total_quarters})\n"
        value += f"Avg surprise: {self.avg_surprise_pct:+.1f}%\n"
        value += f"Avg post-earnings move: {self.avg_move_after_earnings:+.1f}%"
        if streak_text:
            value += f"\n{streak_text}"
        
        return {
            "name": "📊 Earnings History",
            "value": value,
            "inline": True
        }


@dataclass
class SeasonalPattern:
    """Seasonal performance patterns"""
    ticker: str
    
    # Monthly returns (average for each month)
    monthly_returns: Dict[int, float] = field(default_factory=dict)
    
    # Current month stats
    current_month: int = 0
    current_month_avg: float = 0.0
    current_month_win_rate: float = 0.0
    
    # Best/worst months
    best_month: int = 0
    best_month_avg: float = 0.0
    worst_month: int = 0
    worst_month_avg: float = 0.0
    
    # Day of week patterns
    best_day: int = 0  # 0=Monday, 4=Friday
    worst_day: int = 0
    
    # Quarter patterns
    quarterly_returns: Dict[int, float] = field(default_factory=dict)
    current_quarter: int = 0
    current_quarter_avg: float = 0.0
    
    # Santa Claus rally (last 5 days of Dec + first 2 of Jan)
    santa_rally_avg: float = 0.0
    
    # January effect
    january_avg: float = 0.0
    
    def to_signal_score(self) -> float:
        """Convert seasonal patterns to signal score"""
        score = 0.0
        
        # Current month contribution
        if self.current_month_avg > 3:
            score += 0.4
        elif self.current_month_avg > 1:
            score += 0.2
        elif self.current_month_avg < -3:
            score -= 0.4
        elif self.current_month_avg < -1:
            score -= 0.2
        
        # Win rate contribution
        if self.current_month_win_rate > 0.7:
            score += 0.3
        elif self.current_month_win_rate > 0.6:
            score += 0.15
        elif self.current_month_win_rate < 0.3:
            score -= 0.3
        elif self.current_month_win_rate < 0.4:
            score -= 0.15
        
        # Quarter contribution
        if self.current_quarter_avg > 5:
            score += 0.3
        elif self.current_quarter_avg > 2:
            score += 0.15
        elif self.current_quarter_avg < -5:
            score -= 0.3
        elif self.current_quarter_avg < -2:
            score -= 0.15
        
        return max(-1, min(1, score))
    
    def to_discord_field(self) -> dict:
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        current_month_name = month_names[self.current_month - 1]
        best_month_name = month_names[self.best_month - 1]
        worst_month_name = month_names[self.worst_month - 1]
        
        emoji = "🟢" if self.current_month_avg > 1 else "🔴" if self.current_month_avg < -1 else "🟡"
        
        value = f"{emoji} **{current_month_name}** historical: {self.current_month_avg:+.1f}% avg\n"
        value += f"Win rate: {self.current_month_win_rate:.0%}\n"
        value += f"Best: {best_month_name} ({self.best_month_avg:+.1f}%)\n"
        value += f"Worst: {worst_month_name} ({self.worst_month_avg:+.1f}%)"
        
        return {
            "name": "📅 Seasonality",
            "value": value,
            "inline": True
        }


@dataclass
class HistoricalContext:
    """Combined historical context for a stock"""
    ticker: str
    earnings: EarningsHistory
    seasonality: SeasonalPattern
    
    # Long-term performance
    return_1y: float = 0.0
    return_3y: float = 0.0
    return_5y: float = 0.0
    
    # Vs market
    vs_spy_1y: float = 0.0
    
    # Volatility percentile
    volatility_percentile: float = 0.0
    
    def combined_score(self) -> float:
        """Get combined historical score"""
        earnings_score = self.earnings.to_signal_score() * 0.5
        seasonal_score = self.seasonality.to_signal_score() * 0.3
        
        # Long-term performance contribution
        performance_score = 0.0
        if self.vs_spy_1y > 20:
            performance_score = 0.2
        elif self.vs_spy_1y > 10:
            performance_score = 0.1
        elif self.vs_spy_1y < -20:
            performance_score = -0.2
        elif self.vs_spy_1y < -10:
            performance_score = -0.1
        
        return earnings_score + seasonal_score + performance_score


class HistoricalAnalyzer:
    """Analyzes historical patterns for stocks."""
    
    def __init__(self):
        self.cache = {}
    
    def analyze_earnings_history(self, ticker: str) -> EarningsHistory:
        """Analyze earnings beat/miss history."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get earnings history
            earnings = stock.earnings_history
            
            result = EarningsHistory(ticker=ticker)
            
            if earnings is None or earnings.empty:
                return result
            
            # Calculate beat/miss stats
            beats = 0
            misses = 0
            meets = 0
            surprises = []
            
            for _, row in earnings.iterrows():
                actual = row.get('epsActual', row.get('Reported EPS', None))
                estimate = row.get('epsEstimate', row.get('EPS Estimate', None))
                
                if actual is not None and estimate is not None and estimate != 0:
                    surprise_pct = ((actual - estimate) / abs(estimate)) * 100
                    surprises.append(surprise_pct)
                    
                    if surprise_pct > 1:  # Beat by more than 1%
                        beats += 1
                    elif surprise_pct < -1:  # Miss by more than 1%
                        misses += 1
                    else:
                        meets += 1
            
            total = beats + misses + meets
            
            if total > 0:
                result.total_quarters = total
                result.beats = beats
                result.misses = misses
                result.meets = meets
                result.beat_rate = beats / total
                
                if surprises:
                    result.avg_surprise_pct = np.mean(surprises)
            
            # Calculate streak
            if len(surprises) > 0:
                streak = 0
                for s in surprises:  # Most recent first
                    if s > 1:  # Beat
                        if streak >= 0:
                            streak += 1
                        else:
                            break
                    elif s < -1:  # Miss
                        if streak <= 0:
                            streak -= 1
                        else:
                            break
                    else:
                        break
                
                result.current_streak = streak
                if streak >= 2:
                    result.streak_type = "BEAT_STREAK"
                elif streak <= -2:
                    result.streak_type = "MISS_STREAK"
                else:
                    result.streak_type = "NONE"
            
            # Get post-earnings price moves
            try:
                hist = stock.history(period="2y")
                if not hist.empty:
                    # Calculate average moves around earnings dates
                    earnings_dates = earnings.index if hasattr(earnings, 'index') else []
                    moves_on_beat = []
                    moves_on_miss = []
                    all_moves = []
                    
                    for i, (date, row) in enumerate(earnings.iterrows()):
                        # Find price change on earnings day
                        try:
                            if date in hist.index:
                                idx = hist.index.get_loc(date)
                                if idx > 0 and idx < len(hist) - 1:
                                    move = (hist['Close'].iloc[idx+1] - hist['Close'].iloc[idx-1]) / hist['Close'].iloc[idx-1] * 100
                                    all_moves.append(move)
                                    
                                    if i < len(surprises):
                                        if surprises[i] > 1:
                                            moves_on_beat.append(move)
                                        elif surprises[i] < -1:
                                            moves_on_miss.append(move)
                        except:
                            continue
                    
                    if all_moves:
                        result.avg_move_after_earnings = np.mean(all_moves)
                    if moves_on_beat:
                        result.avg_move_on_beat = np.mean(moves_on_beat)
                    if moves_on_miss:
                        result.avg_move_on_miss = np.mean(moves_on_miss)
            except:
                pass
            
            # Get next earnings date
            try:
                calendar = stock.calendar
                if calendar is not None and not calendar.empty:
                    if 'Earnings Date' in calendar.index:
                        next_earnings = calendar.loc['Earnings Date']
                        if hasattr(next_earnings, '__iter__') and not isinstance(next_earnings, str):
                            next_earnings = next_earnings.iloc[0] if len(next_earnings) > 0 else None
                        
                        if next_earnings:
                            if hasattr(next_earnings, 'to_pydatetime'):
                                next_earnings = next_earnings.to_pydatetime()
                            result.days_until_earnings = (next_earnings - datetime.now()).days
            except:
                pass
            
            return result
            
        except Exception as e:
            print(f"Error analyzing earnings for {ticker}: {e}")
            return EarningsHistory(ticker=ticker)
    
    def analyze_seasonality(self, ticker: str, years: int = 5) -> SeasonalPattern:
        """Analyze seasonal patterns."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{years}y")
            
            result = SeasonalPattern(ticker=ticker)
            
            if hist.empty:
                return result
            
            # Calculate monthly returns
            hist['month'] = hist.index.month
            hist['year'] = hist.index.year
            hist['quarter'] = hist.index.quarter
            hist['day_of_week'] = hist.index.dayofweek
            hist['daily_return'] = hist['Close'].pct_change() * 100
            
            # Monthly aggregation
            monthly_data = hist.groupby([hist.index.year, hist.index.month])['Close'].agg(['first', 'last'])
            monthly_data['return'] = (monthly_data['last'] - monthly_data['first']) / monthly_data['first'] * 100
            
            # Average return by month
            monthly_returns = {}
            monthly_win_rates = {}
            for month in range(1, 13):
                month_data = monthly_data.xs(month, level=1) if month in monthly_data.index.get_level_values(1) else pd.DataFrame()
                if not month_data.empty:
                    monthly_returns[month] = month_data['return'].mean()
                    monthly_win_rates[month] = (month_data['return'] > 0).mean()
                else:
                    monthly_returns[month] = 0.0
                    monthly_win_rates[month] = 0.5
            
            result.monthly_returns = monthly_returns
            
            # Current month
            current_month = datetime.now().month
            result.current_month = current_month
            result.current_month_avg = monthly_returns.get(current_month, 0.0)
            result.current_month_win_rate = monthly_win_rates.get(current_month, 0.5)
            
            # Best/worst months
            if monthly_returns:
                result.best_month = max(monthly_returns, key=monthly_returns.get)
                result.best_month_avg = monthly_returns[result.best_month]
                result.worst_month = min(monthly_returns, key=monthly_returns.get)
                result.worst_month_avg = monthly_returns[result.worst_month]
            
            # Quarterly patterns
            quarterly_data = hist.groupby([hist.index.year, hist.index.quarter])['Close'].agg(['first', 'last'])
            quarterly_data['return'] = (quarterly_data['last'] - quarterly_data['first']) / quarterly_data['first'] * 100
            
            quarterly_returns = {}
            for q in range(1, 5):
                q_data = quarterly_data.xs(q, level=1) if q in quarterly_data.index.get_level_values(1) else pd.DataFrame()
                if not q_data.empty:
                    quarterly_returns[q] = q_data['return'].mean()
                else:
                    quarterly_returns[q] = 0.0
            
            result.quarterly_returns = quarterly_returns
            result.current_quarter = (datetime.now().month - 1) // 3 + 1
            result.current_quarter_avg = quarterly_returns.get(result.current_quarter, 0.0)
            
            # Day of week patterns
            day_returns = hist.groupby('day_of_week')['daily_return'].mean()
            if not day_returns.empty:
                result.best_day = day_returns.idxmax()
                result.worst_day = day_returns.idxmin()
            
            # January effect
            result.january_avg = monthly_returns.get(1, 0.0)
            
            # Santa Claus rally (simplified - just December avg)
            result.santa_rally_avg = monthly_returns.get(12, 0.0)
            
            return result
            
        except Exception as e:
            print(f"Error analyzing seasonality for {ticker}: {e}")
            return SeasonalPattern(ticker=ticker)
    
    def analyze_long_term_performance(self, ticker: str) -> Dict:
        """Analyze long-term performance vs market."""
        try:
            stock = yf.Ticker(ticker)
            spy = yf.Ticker("SPY")
            
            # Get 5 years of data
            stock_hist = stock.history(period="5y")
            spy_hist = spy.history(period="5y")
            
            if stock_hist.empty or spy_hist.empty:
                return {}
            
            result = {}
            
            # Calculate returns over different periods
            for period_name, days in [('1y', 252), ('3y', 756), ('5y', 1260)]:
                if len(stock_hist) >= days:
                    stock_return = (stock_hist['Close'].iloc[-1] / stock_hist['Close'].iloc[-days] - 1) * 100
                    result[f'return_{period_name}'] = stock_return
                    
                    if len(spy_hist) >= days:
                        spy_return = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[-days] - 1) * 100
                        result[f'vs_spy_{period_name}'] = stock_return - spy_return
            
            # Volatility percentile
            stock_vol = stock_hist['Close'].pct_change().std() * np.sqrt(252) * 100
            result['annualized_volatility'] = stock_vol
            
            return result
            
        except Exception as e:
            print(f"Error analyzing long-term performance for {ticker}: {e}")
            return {}
    
    def get_full_context(self, ticker: str) -> HistoricalContext:
        """Get complete historical context."""
        earnings = self.analyze_earnings_history(ticker)
        seasonality = self.analyze_seasonality(ticker)
        performance = self.analyze_long_term_performance(ticker)
        
        return HistoricalContext(
            ticker=ticker,
            earnings=earnings,
            seasonality=seasonality,
            return_1y=performance.get('return_1y', 0.0),
            return_3y=performance.get('return_3y', 0.0),
            return_5y=performance.get('return_5y', 0.0),
            vs_spy_1y=performance.get('vs_spy_1y', 0.0),
            volatility_percentile=performance.get('annualized_volatility', 0.0)
        )


# Test
if __name__ == "__main__":
    analyzer = HistoricalAnalyzer()
    
    tickers = ["AAPL", "NVDA", "TSLA"]
    
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"Analyzing {ticker}")
        print('='*60)
        
        context = analyzer.get_full_context(ticker)
        
        print(f"\n📊 EARNINGS HISTORY")
        print(f"   Beat Rate: {context.earnings.beat_rate:.0%} ({context.earnings.beats}/{context.earnings.total_quarters})")
        print(f"   Avg Surprise: {context.earnings.avg_surprise_pct:+.1f}%")
        print(f"   Avg Post-Earnings Move: {context.earnings.avg_move_after_earnings:+.1f}%")
        print(f"   Current Streak: {context.earnings.current_streak} ({context.earnings.streak_type})")
        print(f"   Days Until Earnings: {context.earnings.days_until_earnings}")
        print(f"   Signal Score: {context.earnings.to_signal_score():+.2f}")
        
        print(f"\n📅 SEASONALITY")
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        print(f"   Current Month ({month_names[context.seasonality.current_month-1]}): {context.seasonality.current_month_avg:+.1f}%")
        print(f"   Win Rate: {context.seasonality.current_month_win_rate:.0%}")
        print(f"   Best Month: {month_names[context.seasonality.best_month-1]} ({context.seasonality.best_month_avg:+.1f}%)")
        print(f"   Worst Month: {month_names[context.seasonality.worst_month-1]} ({context.seasonality.worst_month_avg:+.1f}%)")
        print(f"   Signal Score: {context.seasonality.to_signal_score():+.2f}")
        
        print(f"\n📈 PERFORMANCE")
        print(f"   1Y Return: {context.return_1y:+.1f}%")
        print(f"   vs SPY (1Y): {context.vs_spy_1y:+.1f}%")
        
        print(f"\n🎯 COMBINED HISTORICAL SCORE: {context.combined_score():+.2f}")
