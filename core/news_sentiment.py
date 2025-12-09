"""
News Sentiment Analysis Module
Fetches news and uses LLM to analyze sentiment for trading signals.
"""

import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import json
import re

import yfinance as yf
from openai import OpenAI  # DeepSeek uses OpenAI-compatible API


# DeepSeek API Configuration
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"  # or "deepseek-reasoner" for R1


@dataclass
class NewsArticle:
    """Represents a news article"""
    title: str
    publisher: str
    link: str
    published: datetime
    summary: Optional[str] = None


@dataclass 
class NewsSentiment:
    """Sentiment analysis result for a stock"""
    ticker: str
    overall_sentiment: str  # "BULLISH", "BEARISH", "NEUTRAL"
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    key_themes: list[str] = field(default_factory=list)
    bullish_factors: list[str] = field(default_factory=list)
    bearish_factors: list[str] = field(default_factory=list)
    news_count: int = 0
    analysis_summary: str = ""
    
    def to_discord_embed_fields(self) -> list[dict]:
        """Return fields for Discord embed"""
        emoji_map = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}
        
        fields = [
            {
                "name": "📰 News Sentiment",
                "value": f"{emoji_map.get(self.overall_sentiment, '⚪')} **{self.overall_sentiment}** ({self.sentiment_score:+.2f})",
                "inline": True
            },
            {
                "name": "📊 Confidence",
                "value": f"{self.confidence:.0%}",
                "inline": True
            },
            {
                "name": "📰 Articles Analyzed",
                "value": str(self.news_count),
                "inline": True
            }
        ]
        
        if self.bullish_factors:
            fields.append({
                "name": "🟢 Bullish Factors",
                "value": "\n".join([f"• {f}" for f in self.bullish_factors[:3]]),
                "inline": False
            })
        
        if self.bearish_factors:
            fields.append({
                "name": "🔴 Bearish Factors", 
                "value": "\n".join([f"• {f}" for f in self.bearish_factors[:3]]),
                "inline": False
            })
            
        return fields


@dataclass
class EarningsInfo:
    """Upcoming earnings information"""
    ticker: str
    earnings_date: Optional[datetime]
    days_until: Optional[int]
    eps_estimate: Optional[float]
    revenue_estimate: Optional[float]
    
    def to_discord_field(self) -> dict:
        if self.earnings_date and self.days_until is not None:
            if self.days_until <= 7:
                warning = "⚠️ "
            else:
                warning = ""
            return {
                "name": "📅 Earnings",
                "value": f"{warning}**{self.days_until} days** ({self.earnings_date.strftime('%b %d')})",
                "inline": True
            }
        return {
            "name": "📅 Earnings",
            "value": "No upcoming date",
            "inline": True
        }


@dataclass
class AnalystRatings:
    """Analyst recommendations summary"""
    ticker: str
    recommendation: str  # "BUY", "HOLD", "SELL"
    target_price: Optional[float]
    current_price: Optional[float]
    num_analysts: int
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0
    
    @property
    def upside_percent(self) -> Optional[float]:
        if self.target_price and self.current_price:
            return ((self.target_price - self.current_price) / self.current_price) * 100
        return None
    
    def to_discord_field(self) -> dict:
        emoji_map = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}
        
        value = f"{emoji_map.get(self.recommendation, '⚪')} **{self.recommendation}**"
        if self.target_price and self.upside_percent:
            value += f"\nTarget: ${self.target_price:.2f} ({self.upside_percent:+.1f}%)"
        value += f"\n{self.num_analysts} analysts"
        
        return {
            "name": "🎯 Analyst Rating",
            "value": value,
            "inline": True
        }


@dataclass
class InsiderActivity:
    """Insider trading activity summary"""
    ticker: str
    net_shares_3m: int  # Net shares bought/sold in 3 months
    net_value_3m: float  # Net dollar value
    buy_transactions: int
    sell_transactions: int
    signal: str  # "BUYING", "SELLING", "NEUTRAL"
    
    def to_discord_field(self) -> dict:
        emoji_map = {"BUYING": "🟢", "SELLING": "🔴", "NEUTRAL": "🟡"}
        
        if self.net_value_3m >= 1_000_000:
            value_str = f"${self.net_value_3m/1_000_000:.1f}M"
        elif self.net_value_3m <= -1_000_000:
            value_str = f"-${abs(self.net_value_3m)/1_000_000:.1f}M"
        else:
            value_str = f"${self.net_value_3m/1000:.0f}K"
            
        return {
            "name": "👔 Insider Activity",
            "value": f"{emoji_map.get(self.signal, '⚪')} **{self.signal}**\n{value_str} (3mo net)",
            "inline": True
        }


@dataclass
class CombinedSignal:
    """Combined trading signal from all factors"""
    ticker: str
    signal: str  # "STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"
    confidence: float
    score: float  # -2 to 2
    
    # Component scores
    technical_score: float
    sentiment_score: float
    analyst_score: float
    insider_score: float
    
    reasoning: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def to_discord_field(self) -> dict:
        emoji_map = {
            "STRONG BUY": "🚀",
            "BUY": "🟢",
            "HOLD": "🟡", 
            "SELL": "🔴",
            "STRONG SELL": "💀"
        }
        
        value = f"{emoji_map.get(self.signal, '⚪')} **{self.signal}**\n"
        value += f"Score: {self.score:+.2f} | Confidence: {self.confidence:.0%}"
        
        if self.warnings:
            value += f"\n⚠️ {self.warnings[0]}"
        
        return {
            "name": "🎯 COMBINED SIGNAL",
            "value": value,
            "inline": False
        }


class NewsAnalyzer:
    """Analyzes news sentiment using LLM."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize with DeepSeek API key.
        Falls back to DEEPSEEK_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=DEEPSEEK_BASE_URL
            )
        else:
            self.client = None
            print("Warning: No DeepSeek API key found. News sentiment will be limited.")
    
    def fetch_news(self, ticker: str, days: int = 7) -> list[NewsArticle]:
        """Fetch recent news for a ticker using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            news_data = stock.news
            
            if not news_data:
                return []
            
            articles = []
            cutoff = datetime.now() - timedelta(days=days)
            
            for item in news_data[:15]:  # Limit to 15 articles
                try:
                    # Handle new yfinance nested format
                    content = item.get('content', item)  # Try nested, fallback to flat
                    
                    # Get title
                    title = content.get('title', item.get('title', ''))
                    if not title:
                        continue
                    
                    # Get publisher
                    provider = content.get('provider', {})
                    if isinstance(provider, dict):
                        publisher = provider.get('displayName', 'Unknown')
                    else:
                        publisher = item.get('publisher', 'Unknown')
                    
                    # Get link
                    canonical = content.get('canonicalUrl', {})
                    if isinstance(canonical, dict):
                        link = canonical.get('url', '')
                    else:
                        link = item.get('link', '')
                    
                    # Get publish time
                    pub_date_str = content.get('pubDate', '')
                    if pub_date_str:
                        # Parse ISO format: '2025-12-09T16:00:00Z'
                        pub_time = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00')).replace(tzinfo=None)
                    else:
                        # Fallback to timestamp
                        timestamp = item.get('providerPublishTime', 0)
                        pub_time = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
                    
                    if pub_time < cutoff:
                        continue
                    
                    # Get summary
                    summary = content.get('summary', item.get('summary', None))
                        
                    articles.append(NewsArticle(
                        title=title,
                        publisher=publisher,
                        link=link,
                        published=pub_time,
                        summary=summary
                    ))
                except Exception as e:
                    print(f"Error parsing article: {e}")
                    continue
                    
            return articles
            
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return []
    
    def analyze_sentiment(self, ticker: str, articles: list[NewsArticle]) -> NewsSentiment:
        """Use LLM to analyze news sentiment."""
        
        if not articles:
            return NewsSentiment(
                ticker=ticker,
                overall_sentiment="NEUTRAL",
                sentiment_score=0.0,
                confidence=0.0,
                news_count=0,
                analysis_summary="No recent news found."
            )
        
        if not self.client:
            # Fallback: simple keyword-based analysis
            return self._simple_sentiment_analysis(ticker, articles)
        
        # Prepare news for LLM
        news_text = "\n".join([
            f"- [{a.publisher}] {a.title}"
            for a in articles
        ])
        
        prompt = f"""Analyze the following news headlines for {ticker} stock and provide a trading sentiment analysis.

NEWS HEADLINES:
{news_text}

Provide your analysis in the following JSON format:
{{
    "overall_sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
    "sentiment_score": <float from -1.0 (very bearish) to 1.0 (very bullish)>,
    "confidence": <float from 0.0 to 1.0 based on news quality/quantity>,
    "key_themes": [<list of 2-3 main themes in the news>],
    "bullish_factors": [<list of positive factors mentioned>],
    "bearish_factors": [<list of negative factors mentioned>],
    "analysis_summary": "<2-3 sentence summary of the news sentiment and its potential impact on the stock>"
}}

Focus on factors that could impact the stock price in the short term (1-2 weeks).
Only return valid JSON, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in news sentiment analysis. Provide objective, data-driven analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text)
            
            return NewsSentiment(
                ticker=ticker,
                overall_sentiment=result.get("overall_sentiment", "NEUTRAL"),
                sentiment_score=float(result.get("sentiment_score", 0)),
                confidence=float(result.get("confidence", 0.5)),
                key_themes=result.get("key_themes", []),
                bullish_factors=result.get("bullish_factors", []),
                bearish_factors=result.get("bearish_factors", []),
                news_count=len(articles),
                analysis_summary=result.get("analysis_summary", "")
            )
            
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return self._simple_sentiment_analysis(ticker, articles)
    
    def _simple_sentiment_analysis(self, ticker: str, articles: list[NewsArticle]) -> NewsSentiment:
        """Fallback keyword-based sentiment analysis."""
        
        bullish_keywords = [
            'surge', 'soar', 'jump', 'gain', 'rise', 'up', 'high', 'record',
            'beat', 'exceed', 'strong', 'growth', 'profit', 'buy', 'upgrade',
            'bullish', 'positive', 'optimistic', 'rally', 'boom'
        ]
        
        bearish_keywords = [
            'drop', 'fall', 'decline', 'down', 'low', 'miss', 'weak', 'loss',
            'sell', 'downgrade', 'bearish', 'negative', 'pessimistic', 'crash',
            'plunge', 'tumble', 'fear', 'concern', 'risk', 'cut', 'layoff'
        ]
        
        bullish_count = 0
        bearish_count = 0
        
        for article in articles:
            text = article.title.lower()
            bullish_count += sum(1 for kw in bullish_keywords if kw in text)
            bearish_count += sum(1 for kw in bearish_keywords if kw in text)
        
        total = bullish_count + bearish_count
        if total == 0:
            score = 0
        else:
            score = (bullish_count - bearish_count) / total
        
        if score > 0.2:
            sentiment = "BULLISH"
        elif score < -0.2:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
        
        return NewsSentiment(
            ticker=ticker,
            overall_sentiment=sentiment,
            sentiment_score=score,
            confidence=min(0.6, len(articles) * 0.1),  # Lower confidence for keyword analysis
            news_count=len(articles),
            analysis_summary=f"Keyword analysis: {bullish_count} bullish, {bearish_count} bearish signals."
        )


class MarketDataAnalyzer:
    """Fetches additional market data for comprehensive analysis."""
    
    def get_earnings_info(self, ticker: str) -> EarningsInfo:
        """Get upcoming earnings date and estimates."""
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            earnings_date = None
            days_until = None
            
            if calendar is not None and not calendar.empty:
                if 'Earnings Date' in calendar.index:
                    dates = calendar.loc['Earnings Date']
                    if hasattr(dates, '__iter__') and not isinstance(dates, str):
                        earnings_date = dates.iloc[0] if len(dates) > 0 else None
                    else:
                        earnings_date = dates
                    
                    if earnings_date:
                        if hasattr(earnings_date, 'to_pydatetime'):
                            earnings_date = earnings_date.to_pydatetime()
                        days_until = (earnings_date - datetime.now()).days
            
            return EarningsInfo(
                ticker=ticker,
                earnings_date=earnings_date,
                days_until=days_until,
                eps_estimate=None,
                revenue_estimate=None
            )
            
        except Exception as e:
            print(f"Error fetching earnings for {ticker}: {e}")
            return EarningsInfo(
                ticker=ticker,
                earnings_date=None,
                days_until=None,
                eps_estimate=None,
                revenue_estimate=None
            )
    
    def get_analyst_ratings(self, ticker: str) -> AnalystRatings:
        """Get analyst recommendations summary."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get recommendations
            rec = stock.recommendations
            info = stock.info
            
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            target_price = info.get('targetMeanPrice')
            
            # Count recommendations - handle new aggregated format
            strong_buy = buy = hold = sell = strong_sell = 0
            
            if rec is not None and not rec.empty:
                # New format: aggregated by period with columns strongBuy, buy, hold, sell, strongSell
                if 'strongBuy' in rec.columns:
                    # Get most recent period (0m = current month)
                    latest = rec.iloc[0]
                    strong_buy = int(latest.get('strongBuy', 0))
                    buy = int(latest.get('buy', 0))
                    hold = int(latest.get('hold', 0))
                    sell = int(latest.get('sell', 0))
                    strong_sell = int(latest.get('strongSell', 0))
                else:
                    # Old format: individual recommendations with 'To Grade' column
                    recent = rec.tail(30)
                    for _, row in recent.iterrows():
                        grade = str(row.get('To Grade', '')).lower()
                        if 'strong buy' in grade:
                            strong_buy += 1
                        elif 'buy' in grade or 'outperform' in grade or 'overweight' in grade:
                            buy += 1
                        elif 'hold' in grade or 'neutral' in grade or 'equal' in grade:
                            hold += 1
                        elif 'strong sell' in grade:
                            strong_sell += 1
                        elif 'sell' in grade or 'underperform' in grade or 'underweight' in grade:
                            sell += 1
            
            total = strong_buy + buy + hold + sell + strong_sell
            
            # Determine overall recommendation
            if total == 0:
                recommendation = "HOLD"
            else:
                buy_pct = (strong_buy + buy) / total
                sell_pct = (strong_sell + sell) / total
                
                if buy_pct > 0.6:
                    recommendation = "BUY"
                elif sell_pct > 0.4:
                    recommendation = "SELL"
                else:
                    recommendation = "HOLD"
            
            return AnalystRatings(
                ticker=ticker,
                recommendation=recommendation,
                target_price=target_price,
                current_price=current_price,
                num_analysts=total,
                strong_buy=strong_buy,
                buy=buy,
                hold=hold,
                sell=sell,
                strong_sell=strong_sell
            )
            
        except Exception as e:
            print(f"Error fetching analyst ratings for {ticker}: {e}")
            return AnalystRatings(
                ticker=ticker,
                recommendation="HOLD",
                target_price=None,
                current_price=None,
                num_analysts=0
            )
    
    def get_insider_activity(self, ticker: str) -> InsiderActivity:
        """Get insider trading activity summary."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get insider transactions
            insider_tx = stock.insider_transactions
            
            buy_count = 0
            sell_count = 0
            net_shares = 0
            net_value = 0.0
            
            if insider_tx is not None and not insider_tx.empty:
                # Look at last 3 months
                cutoff = datetime.now() - timedelta(days=90)
                
                for _, row in insider_tx.iterrows():
                    try:
                        tx_type = str(row.get('Text', '')).lower()
                        shares = row.get('Shares', 0) or 0
                        value = row.get('Value', 0) or 0
                        
                        if 'buy' in tx_type or 'purchase' in tx_type:
                            buy_count += 1
                            net_shares += shares
                            net_value += value
                        elif 'sale' in tx_type or 'sell' in tx_type:
                            sell_count += 1
                            net_shares -= shares
                            net_value -= value
                    except Exception:
                        continue
            
            # Determine signal
            if net_value > 1_000_000:
                signal = "BUYING"
            elif net_value < -5_000_000:
                signal = "SELLING"
            else:
                signal = "NEUTRAL"
            
            return InsiderActivity(
                ticker=ticker,
                net_shares_3m=net_shares,
                net_value_3m=net_value,
                buy_transactions=buy_count,
                sell_transactions=sell_count,
                signal=signal
            )
            
        except Exception as e:
            print(f"Error fetching insider activity for {ticker}: {e}")
            return InsiderActivity(
                ticker=ticker,
                net_shares_3m=0,
                net_value_3m=0,
                buy_transactions=0,
                sell_transactions=0,
                signal="NEUTRAL"
            )


class SignalGenerator:
    """Generates combined trading signals from all data sources."""
    
    def __init__(self, deepseek_api_key: Optional[str] = None):
        self.news_analyzer = NewsAnalyzer(api_key=deepseek_api_key)
        self.market_analyzer = MarketDataAnalyzer()
        
        # Weights for combining signals
        self.weights = {
            'technical': 0.25,
            'sentiment': 0.30,
            'analyst': 0.25,
            'insider': 0.20
        }
    
    def generate_signal(
        self,
        ticker: str,
        technical_score: float = 0.0  # From existing regression analysis
    ) -> tuple[CombinedSignal, NewsSentiment, EarningsInfo, AnalystRatings, InsiderActivity]:
        """
        Generate comprehensive trading signal.
        
        Args:
            ticker: Stock symbol
            technical_score: Score from technical analysis (-1 to 1)
            
        Returns:
            Tuple of (CombinedSignal, NewsSentiment, EarningsInfo, AnalystRatings, InsiderActivity)
        """
        # Fetch all data
        articles = self.news_analyzer.fetch_news(ticker)
        sentiment = self.news_analyzer.analyze_sentiment(ticker, articles)
        earnings = self.market_analyzer.get_earnings_info(ticker)
        analyst = self.market_analyzer.get_analyst_ratings(ticker)
        insider = self.market_analyzer.get_insider_activity(ticker)
        
        # Convert to scores (-1 to 1)
        sentiment_score = sentiment.sentiment_score
        
        analyst_score = {
            "BUY": 0.7,
            "HOLD": 0.0,
            "SELL": -0.7
        }.get(analyst.recommendation, 0.0)
        
        # Add upside potential to analyst score
        if analyst.upside_percent:
            if analyst.upside_percent > 20:
                analyst_score += 0.3
            elif analyst.upside_percent < -10:
                analyst_score -= 0.3
        
        insider_score = {
            "BUYING": 0.5,
            "NEUTRAL": 0.0,
            "SELLING": -0.3
        }.get(insider.signal, 0.0)
        
        # Calculate weighted score
        combined_score = (
            self.weights['technical'] * technical_score +
            self.weights['sentiment'] * sentiment_score +
            self.weights['analyst'] * analyst_score +
            self.weights['insider'] * insider_score
        )
        
        # Generate warnings
        warnings = []
        if earnings.days_until is not None and earnings.days_until <= 7:
            warnings.append(f"Earnings in {earnings.days_until} days - expect volatility")
        if sentiment.confidence < 0.3:
            warnings.append("Low news coverage - limited sentiment data")
        
        # Generate reasoning
        reasoning = []
        if sentiment_score > 0.3:
            reasoning.append("Positive news sentiment")
        elif sentiment_score < -0.3:
            reasoning.append("Negative news sentiment")
        
        if analyst.recommendation == "BUY":
            reasoning.append(f"Analysts bullish (target: ${analyst.target_price:.2f})" if analyst.target_price else "Analysts bullish")
        elif analyst.recommendation == "SELL":
            reasoning.append("Analysts bearish")
        
        if insider.signal == "BUYING":
            reasoning.append("Insider buying activity")
        elif insider.signal == "SELLING":
            reasoning.append("Insider selling activity")
        
        # Determine signal
        if combined_score >= 0.5:
            signal = "STRONG BUY"
        elif combined_score >= 0.2:
            signal = "BUY"
        elif combined_score <= -0.5:
            signal = "STRONG SELL"
        elif combined_score <= -0.2:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        # Reduce confidence if near earnings
        confidence = sentiment.confidence * 0.4 + 0.4  # Base confidence
        if earnings.days_until is not None and earnings.days_until <= 7:
            confidence *= 0.7
        
        combined = CombinedSignal(
            ticker=ticker,
            signal=signal,
            confidence=min(confidence, 0.85),  # Cap at 85%
            score=combined_score,
            technical_score=technical_score,
            sentiment_score=sentiment_score,
            analyst_score=analyst_score,
            insider_score=insider_score,
            reasoning=reasoning,
            warnings=warnings
        )
        
        return combined, sentiment, earnings, analyst, insider


# Test
if __name__ == "__main__":
    generator = SignalGenerator()
    
    ticker = "AAPL"
    print(f"\n{'='*50}")
    print(f"Analyzing {ticker}...")
    print('='*50)
    
    combined, sentiment, earnings, analyst, insider = generator.generate_signal(ticker, technical_score=0.1)
    
    print(f"\n📰 News Sentiment: {sentiment.overall_sentiment} ({sentiment.sentiment_score:+.2f})")
    print(f"   Confidence: {sentiment.confidence:.0%}")
    print(f"   Articles: {sentiment.news_count}")
    if sentiment.bullish_factors:
        print(f"   Bullish: {', '.join(sentiment.bullish_factors[:2])}")
    if sentiment.bearish_factors:
        print(f"   Bearish: {', '.join(sentiment.bearish_factors[:2])}")
    
    print(f"\n📅 Earnings: {earnings.days_until} days away" if earnings.days_until else "\n📅 Earnings: No date found")
    
    print(f"\n🎯 Analyst Rating: {analyst.recommendation}")
    print(f"   Target: ${analyst.target_price:.2f} ({analyst.upside_percent:+.1f}%)" if analyst.target_price else "   No target price")
    print(f"   Analysts: {analyst.num_analysts}")
    
    print(f"\n👔 Insider Activity: {insider.signal}")
    print(f"   Net Value (3mo): ${insider.net_value_3m:,.0f}")
    
    print(f"\n{'='*50}")
    print(f"🎯 COMBINED SIGNAL: {combined.signal}")
    print(f"   Score: {combined.score:+.2f}")
    print(f"   Confidence: {combined.confidence:.0%}")
    if combined.reasoning:
        print(f"   Reasoning: {', '.join(combined.reasoning)}")
    if combined.warnings:
        print(f"   ⚠️ Warnings: {', '.join(combined.warnings)}")
