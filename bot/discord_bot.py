"""
Stock Monitor Discord Bot
Main bot file with commands and scheduled alerts.
"""

import discord
from discord.ext import commands, tasks
from discord import app_commands
import asyncio
import csv
import io
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Optional, Tuple
import os
from dotenv import load_dotenv

# Import our modules
from core.stock_data import StockDataFetcher, StockSignal
from core.regression import RegressionAnalyzer, TrendAnalysis
from core.visualizer import ChartGenerator
from core.news_sentiment import SignalGenerator, NewsAnalyzer, MarketDataAnalyzer
from core.ultimate_signal import UltimateSignalGenerator, UltimateSignal
from core.persistence import Store, StoreConfigurationError
from core.macro import MacroAnalyzer
from core.sec_filings import SECClient, ensure_filing_summaries
from core.positions import PositionTracker
from core.briefing import BriefingAssembler, BriefingReport
from core.synthesis import Synthesizer
from core.alerts import ensure_default_watchlist_threshold, run_event_alert_job
from bot.embeds.briefing import build_briefing_embed

load_dotenv()

_POSITION_CSV_COLUMNS = frozenset({"symbol", "quantity", "cost_basis", "opened_at"})


async def log_error_channel(bot: "StockMonitorBot", message: str) -> None:
    """Post briefing/macro errors to ERROR_CHANNEL_ID, then ALERT_CHANNEL_ID; else print."""
    eid = int(os.getenv("ERROR_CHANNEL_ID", "0") or "0")
    if not eid:
        eid = int(os.getenv("ALERT_CHANNEL_ID", "0") or "0") or int(
            getattr(bot, "alert_channel_id", 0) or 0
        )
    if not eid:
        print(f"[briefing] {message}")
        return
    try:
        ch = bot.get_channel(eid)
        if ch is not None and isinstance(ch, discord.abc.Messageable):
            await ch.send(f"⚠️ `{message[:1900]}`")
    except Exception as exc:
        print(f"[error channel] {exc}")


async def deliver_morning_briefing(
    bot: "StockMonitorBot",
    *,
    skip_dedup: bool = False,
    interaction: Optional[discord.Interaction] = None,
) -> Tuple[Optional[BriefingReport], str]:
    """
    Build and post the morning briefing. When ``skip_dedup`` is False, uses
    ``alerts_sent`` (morning_briefing + today's date) and marks sent after post.

    Returns ``(report, status)`` where status is ``ok``, ``already_sent``, ``no_channel``, or error text.
    """
    if bot.store is None:
        msg = "Supabase not configured."
        if interaction:
            await interaction.followup.send(f"❌ {msg}")
        return None, msg

    ref = date.today().isoformat()
    if not skip_dedup:
        if await bot.store.has_alert_been_sent("morning_briefing", ref):
            return None, "already_sent"

    try:
        assembler = BriefingAssembler(
            bot.store,
            bot.fetcher,
            bot.watchlist,
            synthesizer=Synthesizer(bot.store),
        )
        report = await assembler.build(date.today())
    except Exception as exc:
        await log_error_channel(bot, f"briefing build failed: {exc}")
        if interaction:
            await interaction.followup.send(f"❌ Briefing failed: `{exc}`")
        return None, str(exc)

    embed = build_briefing_embed(report)
    for err in report.section_errors:
        await log_error_channel(bot, str(err))

    try:
        if interaction:
            await interaction.followup.send(embed=embed)
        else:
            cid = int(os.getenv("BRIEFING_CHANNEL_ID") or os.getenv("ALERT_CHANNEL_ID") or "0")
            if not cid:
                await log_error_channel(
                    bot, "No BRIEFING_CHANNEL_ID / ALERT_CHANNEL_ID for morning briefing"
                )
                return report, "no_channel"
            ch = bot.get_channel(cid)
            if not ch:
                await log_error_channel(bot, f"Briefing channel {cid} not found")
                return report, "no_channel"
            await ch.send(embed=embed)
            if not skip_dedup:
                await bot.store.mark_alert_sent(
                    "morning_briefing", ref, channel_id=str(cid)
                )
    except Exception as exc:
        await log_error_channel(bot, f"briefing post failed: {exc}")
        if interaction:
            await interaction.followup.send(f"❌ Post failed: `{exc}`")
        raise

    return report, "ok"


class StockMonitorBot(commands.Bot):
    """Discord bot for stock monitoring and analysis."""
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        
        super().__init__(
            command_prefix="!",
            intents=intents,
            description="Stock Monitor Bot - 52-week signals & trend analysis"
        )
        
        # Default watchlist - customize this!
        self.watchlist = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMD", "META", "AMZN"]
        
        # Components
        self.fetcher = StockDataFetcher(self.watchlist)
        self.analyzer = RegressionAnalyzer()
        self.chart_gen = ChartGenerator(output_dir="./charts")
        self.signal_gen = SignalGenerator(deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"))
        
        # Ultimate Signal Generator (ML + Everything)
        self.ultimate_gen = UltimateSignalGenerator(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            use_ml=True
        )
        
        # Alert channel (set via command or env)
        self.alert_channel_id = int(os.getenv("ALERT_CHANNEL_ID", "0"))
        
        # Signal threshold (how close to 52w high/low to alert)
        self.alert_threshold = 3.0  # percent

        # Supabase store: one async client for the bot lifetime (commands reuse it).
        self.store: Optional[Store] = None

    async def setup_hook(self):
        """Called when bot is ready, before on_ready."""
        try:
            self.store = await Store.from_env()
        except StoreConfigurationError:
            self.store = None
            print(
                "⚠️ Supabase not configured (SUPABASE_URL / SUPABASE_SERVICE_KEY) — "
                "/macro and other DB features disabled until set."
            )

        # Add cog with slash commands
        await self.add_cog(StockCommands(self))
        await self.add_cog(AlertCommandsCog(self))
        await self.add_cog(ScheduledTasksCog(self))

        # Sync commands
        await self.tree.sync()
        print("Slash commands synced!")

    async def on_ready(self):
        print(f"✅ {self.user} is online!")
        print(f"📊 Monitoring: {', '.join(self.watchlist)}")
        print(f"🔔 Alert threshold: {self.alert_threshold}%")


class StockCommands(commands.Cog):
    """Slash commands for stock monitoring."""

    position = app_commands.Group(
        name="position",
        description="Track portfolio holdings, cost basis, and optional profit/stop targets",
    )

    def __init__(self, bot: StockMonitorBot):
        self.bot = bot

    def _price_maps_for_symbols(
        self, symbols: set[str]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Current price and previous close from yfinance (via fetcher), keyed by upper symbol."""
        prices: dict[str, float] = {}
        prev_close: dict[str, float] = {}
        for s in symbols:
            df = self.bot.fetcher.fetch_52_week_data(s)
            if df is None or df.empty:
                continue
            su = s.strip().upper()
            prices[su] = float(df["Close"].iloc[-1])
            if len(df) >= 2:
                prev_close[su] = float(df["Close"].iloc[-2])
        return prices, prev_close
    
    @app_commands.command(name="check", description="Check a stock for 52-week signals")
    @app_commands.describe(ticker="Stock symbol (e.g., AAPL)")
    async def check(self, interaction: discord.Interaction, ticker: str):
        """Check a single stock's 52-week status."""
        await interaction.response.defer(thinking=True)
        
        ticker = ticker.upper()
        metrics = self.bot.fetcher.get_52_week_range(ticker)
        
        if not metrics:
            await interaction.followup.send(f"❌ Could not fetch data for **{ticker}**")
            return
        
        # Determine status emoji
        pos = metrics['range_position']
        if pos >= 90:
            status = "🔥 Near 52-week HIGH"
        elif pos <= 10:
            status = "❄️ Near 52-week LOW"
        elif pos >= 50:
            status = "📈 Upper half of range"
        else:
            status = "📉 Lower half of range"
        
        embed = discord.Embed(
            title=f"📊 {ticker} - 52 Week Analysis",
            color=discord.Color.green() if pos >= 50 else discord.Color.red(),
            timestamp=datetime.now()
        )
        
        embed.add_field(
            name="Current Price",
            value=f"${metrics['current_price']:.2f}",
            inline=True
        )
        embed.add_field(
            name="52W High",
            value=f"${metrics['52_week_high']:.2f} ({metrics['percent_from_high']:+.1f}%)",
            inline=True
        )
        embed.add_field(
            name="52W Low",
            value=f"${metrics['52_week_low']:.2f} ({metrics['percent_from_low']:+.1f}%)",
            inline=True
        )
        embed.add_field(
            name="Range Position",
            value=f"{pos:.1f}%",
            inline=True
        )
        embed.add_field(
            name="Status",
            value=status,
            inline=True
        )
        
        await interaction.followup.send(embed=embed)
    
    @app_commands.command(name="analyze", description="Get trend analysis with chart")
    @app_commands.describe(ticker="Stock symbol (e.g., AAPL)")
    async def analyze(self, interaction: discord.Interaction, ticker: str):
        """Full regression analysis with chart for a stock."""
        await interaction.response.defer(thinking=True)
        
        ticker = ticker.upper()
        
        # Fetch data
        df = self.bot.fetcher.fetch_52_week_data(ticker)
        if df is None or df.empty:
            await interaction.followup.send(f"❌ Could not fetch data for **{ticker}**")
            return
        
        # Run regression analysis
        analysis = self.bot.analyzer.predict_weekly_trend(df, ticker)
        
        if not analysis:
            await interaction.followup.send(f"❌ Not enough data to analyze **{ticker}**")
            return
        
        # Generate chart
        prediction = {
            'direction': analysis.trend_direction,
            'change': analysis.predicted_change_percent,
            'recommendation': analysis.recommendation
        }
        chart_buffer = self.bot.chart_gen.create_chart_bytes(df, ticker, prediction=prediction)
        
        # Create embed
        color_map = {
            "BULLISH": discord.Color.green(),
            "BEARISH": discord.Color.red(),
            "NEUTRAL": discord.Color.gold()
        }
        
        embed = discord.Embed(
            title=f"📈 {ticker} - Weekly Trend Analysis",
            description=analysis.reasoning,
            color=color_map.get(analysis.trend_direction, discord.Color.blue()),
            timestamp=datetime.now()
        )
        
        embed.add_field(
            name="🎯 Trend",
            value=f"{analysis.trend_direction} (Strength: {analysis.trend_strength:.0f}/100)",
            inline=True
        )
        embed.add_field(
            name="📊 Mon-Thu Forecast",
            value=f"{analysis.predicted_change_percent:+.2f}%",
            inline=True
        )
        embed.add_field(
            name="🎲 Confidence",
            value=f"{analysis.confidence:.1%}",
            inline=True
        )
        embed.add_field(
            name="💪 Support",
            value=f"${analysis.support_level:.2f}",
            inline=True
        )
        embed.add_field(
            name="🚧 Resistance",
            value=f"${analysis.resistance_level:.2f}",
            inline=True
        )
        
        rec_emoji = {"BUY": "💰", "SELL": "🚪", "HOLD": "⏸️"}
        embed.add_field(
            name="📌 Recommendation",
            value=f"{rec_emoji.get(analysis.recommendation, '❓')} **{analysis.recommendation}**",
            inline=True
        )
        
        # Send with chart
        file = discord.File(chart_buffer, filename=f"{ticker}_analysis.png")
        embed.set_image(url=f"attachment://{ticker}_analysis.png")
        
        await interaction.followup.send(embed=embed, file=file)
    
    @app_commands.command(name="watchlist", description="View or modify watchlist")
    @app_commands.describe(
        action="add, remove, list, or threshold",
        ticker="Stock symbol (for add/remove/threshold)",
        threshold_pct="Price-move alert threshold %% (threshold action only)",
    )
    async def watchlist(
        self,
        interaction: discord.Interaction,
        action: str = "list",
        ticker: Optional[str] = None,
        threshold_pct: Optional[float] = None,
    ):
        """Manage the stock watchlist."""
        action = action.lower()

        if action == "list":
            embed = discord.Embed(
                title="📋 Stock Watchlist",
                description="\n".join([f"• {t}" for t in self.bot.watchlist]),
                color=discord.Color.blue()
            )
            await interaction.response.send_message(embed=embed)
            
        elif action == "add" and ticker:
            ticker = ticker.upper()
            if ticker not in self.bot.watchlist:
                self.bot.watchlist.append(ticker)
                self.bot.fetcher = StockDataFetcher(self.bot.watchlist)
                if self.bot.store is not None:
                    await ensure_default_watchlist_threshold(self.bot.store, ticker)
                await interaction.response.send_message(f"✅ Added **{ticker}** to watchlist")
            else:
                await interaction.response.send_message(f"⚠️ **{ticker}** already in watchlist")

        elif action == "threshold" and ticker and threshold_pct is not None:
            sym = ticker.strip().upper()
            if self.bot.store is None:
                await interaction.response.send_message(
                    "❌ Supabase not configured — cannot set alert threshold."
                )
                return
            await self.bot.store.set_watchlist_meta(
                sym, alert_threshold_pct=Decimal(str(threshold_pct))
            )
            await interaction.response.send_message(
                f"✅ Alert threshold for **{sym}** set to **{threshold_pct}%** (stored in watchlist_meta)."
            )

        elif action == "remove" and ticker:
            ticker = ticker.upper()
            if ticker in self.bot.watchlist:
                self.bot.watchlist.remove(ticker)
                self.bot.fetcher = StockDataFetcher(self.bot.watchlist)
                await interaction.response.send_message(f"✅ Removed **{ticker}** from watchlist")
            else:
                await interaction.response.send_message(f"⚠️ **{ticker}** not in watchlist")
        else:
            await interaction.response.send_message(
                "Usage: `/watchlist list` | `/watchlist add TICKER` | `/watchlist remove TICKER` | "
                "`/watchlist threshold TICKER PCT`"
            )
    
    @app_commands.command(name="setchannel", description="Set alert channel")
    async def setchannel(self, interaction: discord.Interaction):
        """Set current channel as alert channel."""
        self.bot.alert_channel_id = interaction.channel_id
        await interaction.response.send_message(
            f"✅ Alerts will be sent to {interaction.channel.mention}"
        )
    
    @app_commands.command(name="scan", description="Scan all watchlist for signals now")
    async def scan(self, interaction: discord.Interaction):
        """Manually trigger a scan of all watchlist stocks."""
        await interaction.response.defer(thinking=True)
        
        signals = self.bot.fetcher.check_signals(threshold_percent=self.bot.alert_threshold)
        
        if not signals:
            await interaction.followup.send("✅ No 52-week signals detected in watchlist.")
            return
        
        embed = discord.Embed(
            title="🚨 52-Week Signals Detected",
            color=discord.Color.orange(),
            timestamp=datetime.now()
        )
        
        for signal in signals:
            emoji = "🚀" if signal.signal_type == "52_WEEK_HIGH" else "📉"
            embed.add_field(
                name=f"{emoji} {signal.ticker}",
                value=(
                    f"**{signal.signal_type.replace('_', ' ')}**\n"
                    f"Current: ${signal.current_price:.2f}\n"
                    f"Threshold: ${signal.threshold_price:.2f}\n"
                    f"Distance: {signal.percent_from_threshold:+.1f}%"
                ),
                inline=True
            )
        
        await interaction.followup.send(embed=embed)
    
    @app_commands.command(name="summary", description="Get summary of all watchlist stocks")
    async def summary(self, interaction: discord.Interaction):
        """Summary table of all stocks."""
        await interaction.response.defer(thinking=True)
        
        df = self.bot.fetcher.get_summary()
        
        if df.empty:
            await interaction.followup.send("❌ Could not fetch watchlist data")
            return
        
        # Create text table
        lines = ["```"]
        lines.append(f"{'Ticker':<8} {'Price':>10} {'52W High':>10} {'52W Low':>10} {'Position':>10}")
        lines.append("-" * 55)
        
        for _, row in df.iterrows():
            lines.append(
                f"{row['ticker']:<8} "
                f"${row['current_price']:>8.2f} "
                f"${row['52_week_high']:>8.2f} "
                f"${row['52_week_low']:>8.2f} "
                f"{row['range_position']:>8.1f}%"
            )
        
        lines.append("```")
        
        await interaction.followup.send("\n".join(lines))
    
    @app_commands.command(name="signal", description="Get comprehensive BUY/HOLD/SELL signal with ML + sentiment + analysts")
    @app_commands.describe(ticker="Stock symbol (e.g., AAPL)")
    async def signal(self, interaction: discord.Interaction, ticker: str):
        """Ultimate trading signal combining ML models with all factors."""
        await interaction.response.defer(thinking=True)
        
        ticker = ticker.upper()
        
        try:
            # Generate ultimate signal
            signal = self.bot.ultimate_gen.generate_signal(ticker)
        except Exception as e:
            await interaction.followup.send(f"❌ Error analyzing **{ticker}**: {str(e)}")
            return
        
        # Build embed
        signal_colors = {
            "STRONG BUY": discord.Color.green(),
            "BUY": discord.Color.from_rgb(144, 238, 144),
            "HOLD": discord.Color.gold(),
            "SELL": discord.Color.from_rgb(255, 99, 71),
            "STRONG SELL": discord.Color.red()
        }
        
        embed = discord.Embed(
            title=f"{signal.get_signal_emoji()} {ticker} - {signal.signal}",
            description=f"**Score:** {signal.score:+.2f} | **Confidence:** {signal.confidence:.0%}",
            color=signal_colors.get(signal.signal, discord.Color.blue()),
            timestamp=datetime.now()
        )
        
        # Component scores
        embed.add_field(
            name="🤖 ML Models",
            value=f"{signal.ml_score:+.2f}\n{signal.ml_direction}",
            inline=True
        )
        embed.add_field(
            name="📰 Sentiment",
            value=f"{signal.sentiment_score:+.2f}",
            inline=True
        )
        embed.add_field(
            name="🎯 Analysts",
            value=f"{signal.analyst_score:+.2f}\n{signal.analyst_rating}",
            inline=True
        )
        embed.add_field(
            name="👔 Insiders",
            value=f"{signal.insider_score:+.2f}",
            inline=True
        )
        embed.add_field(
            name="📊 Earnings",
            value=f"{signal.earnings_score:+.2f}\n{signal.earnings_beat_rate:.0%} beat",
            inline=True
        )
        embed.add_field(
            name="📅 Seasonal",
            value=f"{signal.seasonality_score:+.2f}\n{signal.current_month_avg:+.1f}% avg",
            inline=True
        )
        
        # Analyst target
        if signal.price_target and signal.upside_pct:
            embed.add_field(
                name="🎯 Price Target",
                value=f"${signal.price_target:.2f} ({signal.upside_pct:+.1f}%)",
                inline=True
            )
        
        # Earnings warning
        if signal.days_until_earnings:
            warning = "⚠️ " if signal.days_until_earnings <= 14 else ""
            embed.add_field(
                name="📅 Earnings",
                value=f"{warning}{signal.days_until_earnings} days",
                inline=True
            )
        
        # Bullish factors
        if signal.bullish_factors:
            bullish_text = "\n".join([f"• {f}" for f in signal.bullish_factors[:4]])
            embed.add_field(
                name="🟢 Bullish Factors",
                value=bullish_text[:1024],
                inline=False
            )
        
        # Bearish factors
        if signal.bearish_factors:
            bearish_text = "\n".join([f"• {f}" for f in signal.bearish_factors[:4]])
            embed.add_field(
                name="🔴 Bearish Factors",
                value=bearish_text[:1024],
                inline=False
            )
        
        # Warnings
        if signal.warnings:
            warnings_text = "\n".join([f"• {w}" for w in signal.warnings])
            embed.add_field(
                name="⚠️ Warnings",
                value=warnings_text[:1024],
                inline=False
            )
        
        # News summary
        if signal.news_summary:
            embed.add_field(
                name="📰 News Summary",
                value=signal.news_summary[:500],
                inline=False
            )
        
        # ML model details (if available)
        ml_details = []
        if signal.lstm_pred is not None:
            ml_details.append(f"LSTM: {signal.lstm_pred:.0%}")
        if signal.xgb_pred is not None:
            ml_details.append(f"XGB: {signal.xgb_pred:.0%}")
        if signal.lgb_pred is not None:
            ml_details.append(f"LGB: {signal.lgb_pred:.0%}")
        if ml_details:
            embed.set_footer(text=f"ML Predictions: {' | '.join(ml_details)} • Not financial advice")
        else:
            embed.set_footer(text=f"Analysis based on {signal.news_count} news articles • Not financial advice")
        
        await interaction.followup.send(embed=embed)
    
    @app_commands.command(name="news", description="Get latest news and sentiment for a stock")
    @app_commands.describe(ticker="Stock symbol (e.g., AAPL)")
    async def news(self, interaction: discord.Interaction, ticker: str):
        """Get news headlines and sentiment analysis."""
        await interaction.response.defer(thinking=True)
        
        ticker = ticker.upper()
        
        # Fetch news and analyze
        articles = self.bot.signal_gen.news_analyzer.fetch_news(ticker)
        sentiment = self.bot.signal_gen.news_analyzer.analyze_sentiment(ticker, articles)
        
        if not articles:
            await interaction.followup.send(f"📰 No recent news found for **{ticker}**")
            return
        
        # Build embed
        sentiment_colors = {
            "BULLISH": discord.Color.green(),
            "BEARISH": discord.Color.red(),
            "NEUTRAL": discord.Color.gold()
        }
        sentiment_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}
        
        embed = discord.Embed(
            title=f"📰 {ticker} News & Sentiment",
            description=f"{sentiment_emoji.get(sentiment.overall_sentiment, '⚪')} **{sentiment.overall_sentiment}** (Score: {sentiment.sentiment_score:+.2f})",
            color=sentiment_colors.get(sentiment.overall_sentiment, discord.Color.blue()),
            timestamp=datetime.now()
        )
        
        # Add summary
        if sentiment.analysis_summary:
            embed.add_field(
                name="📋 Analysis",
                value=sentiment.analysis_summary,
                inline=False
            )
        
        # Add headlines
        headlines = []
        for i, article in enumerate(articles[:5]):
            age = (datetime.now() - article.published).days
            age_str = f"{age}d ago" if age > 0 else "Today"
            headlines.append(f"**{i+1}.** [{article.title[:60]}...]({article.link})\n*{article.publisher} • {age_str}*")
        
        embed.add_field(
            name=f"📰 Latest Headlines ({len(articles)} total)",
            value="\n\n".join(headlines),
            inline=False
        )
        
        # Add factors
        if sentiment.bullish_factors:
            embed.add_field(
                name="🟢 Bullish Factors",
                value="\n".join([f"• {f}" for f in sentiment.bullish_factors[:3]]),
                inline=True
            )
        if sentiment.bearish_factors:
            embed.add_field(
                name="🔴 Bearish Factors",
                value="\n".join([f"• {f}" for f in sentiment.bearish_factors[:3]]),
                inline=True
            )
        
        await interaction.followup.send(embed=embed)

    @app_commands.command(
        name="macro",
        description="Macro snapshot: yields, VIX, dollar, CPI, labor (FRED data in database)",
    )
    async def macro(self, interaction: discord.Interaction):
        """Show latest macro indicators from stored FRED series."""
        await interaction.response.defer(thinking=True)
        if self.bot.store is None:
            await interaction.followup.send(
                "❌ Supabase is not configured. Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in `.env`."
            )
            return

        analyzer = MacroAnalyzer(self.bot.store)
        changes = await analyzer.get_changes(date.today())
        if not changes:
            await interaction.followup.send("No macro data available.")
            return

        def _field_block(items: list[dict]) -> str:
            lines = [f"**{c['series_id']}** — {c['change_str']}" for c in items]
            text = "\n".join(lines)
            return text if len(text) <= 1024 else text[:1021] + "..."

        rates = [c for c in changes if c["series_id"] in ("DGS10", "DGS2", "DFF")]
        risk = [c for c in changes if c["series_id"] in ("VIXCLS", "DTWEXBGS")]
        labor = [c for c in changes if c["series_id"] in ("CPIAUCSL", "UNRATE", "PAYEMS")]

        embed = discord.Embed(
            title="🌍 Macro snapshot",
            description=(
                "Values are read from your database (loaded via `python -m jobs.update_macro`). "
                "Not prescriptive — context only."
            ),
            color=discord.Color.blue(),
            timestamp=datetime.now(),
        )
        embed.add_field(name="📈 Rates", value=_field_block(rates), inline=False)
        embed.add_field(name="📊 Risk & USD", value=_field_block(risk), inline=False)
        embed.add_field(name="🏛️ Prices & labor", value=_field_block(labor), inline=False)
        embed.set_footer(text="FRED (St. Louis Fed) • macro_series • Not financial advice")

        await interaction.followup.send(embed=embed)

    @app_commands.command(
        name="filings",
        description="Recent SEC EDGAR filings for a symbol (summaries when cached / Haiku)",
    )
    @app_commands.describe(ticker="Stock symbol (e.g. NVDA)")
    async def filings_cmd(self, interaction: discord.Interaction, ticker: str):
        """Fetch recent filings from EDGAR, store metadata, and summarize new items (Haiku cap)."""
        await interaction.response.defer(thinking=True)
        if self.bot.store is None:
            await interaction.followup.send(
                "❌ Supabase is not configured. Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`."
            )
            return

        sym = ticker.strip().upper()
        try:
            client = SECClient()
        except ValueError as exc:
            await interaction.followup.send(f"❌ {exc}")
            return

        try:
            store = self.bot.store
            meta = await store.get_watchlist_meta(sym)
            if meta and meta.cik:
                cik_str = meta.cik
            else:
                cik_str = await client.resolve_cik(sym, store)

            since = datetime.now(timezone.utc) - timedelta(days=400)
            recent = await client.fetch_recent_filings(cik_str, since)
            await ensure_filing_summaries(
                store,
                client,
                sym,
                recent[:50],
                max_haiku_summaries=3,
            )

            rows = await store.list_filings_for_symbol(sym, limit=12)
        except Exception as exc:
            await interaction.followup.send(f"❌ EDGAR error for **{sym}**: `{exc}`")
            return
        finally:
            await client.aclose()

        if not rows:
            await interaction.followup.send(
                f"No tracked filings found for **{sym}** in the lookback window."
            )
            return

        embed = discord.Embed(
            title=f"📁 SEC filings — {sym}",
            description=(
                "Direct from EDGAR (not investment advice). "
                "Summaries use Claude Haiku where enabled; Form 4 uses XML parse."
            ),
            color=discord.Color.dark_teal(),
            timestamp=datetime.now(),
        )

        def _clip(s: Optional[str], n: int = 420) -> str:
            if not s:
                return "—"
            s = s.strip()
            return s if len(s) <= n else s[: n - 1] + "…"

        for r in rows[:6]:
            summ = _clip(r.summary, 400)
            val = (
                f"**Filed:** {r.filed_at.date()} • **CIK** {r.cik}\n"
                f"{summ}\n[View]({r.url})"
            )
            embed.add_field(
                name=f"{r.form_type} — `{r.accession_number}`",
                value=val[:1024],
                inline=False,
            )

        embed.set_footer(text="SEC EDGAR • summaries cached in Supabase • Not financial advice")
        await interaction.followup.send(embed=embed)

    @position.command(name="add", description="Add an open position (cost = average cost per share)")
    @app_commands.describe(
        symbol="Ticker symbol",
        quantity="Number of shares",
        cost="Average cost per share (USD)",
        notes="Optional note",
        target="Optional take-profit alert: +X% vs cost",
        stop="Optional stop-loss alert: X% drawdown vs cost (positive number)",
    )
    async def position_add(
        self,
        interaction: discord.Interaction,
        symbol: str,
        quantity: float,
        cost: float,
        notes: Optional[str] = None,
        target: Optional[float] = None,
        stop: Optional[float] = None,
    ):
        await interaction.response.defer(thinking=True)
        if self.bot.store is None:
            await interaction.followup.send(
                "❌ Supabase is not configured. Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`."
            )
            return

        sym = symbol.strip().upper()
        tracker = PositionTracker(self.bot.store)
        try:
            pos = await tracker.add(
                sym,
                Decimal(str(quantity)),
                Decimal(str(cost)),
                notes=notes,
                target_pct=Decimal(str(target)) if target is not None else None,
                stop_pct=Decimal(str(stop)) if stop is not None else None,
            )
        except Exception as exc:
            await interaction.followup.send(f"❌ Could not add position: `{exc}`")
            return

        lines = [
            f"**{pos.symbol}** × `{pos.quantity}` @ `${pos.cost_basis}`/sh",
            f"`{pos.id}`",
        ]
        if pos.target_pct is not None:
            lines.append(f"Target +{pos.target_pct}%")
        if pos.stop_pct is not None:
            lines.append(f"Stop −{pos.stop_pct}% (threshold)")
        embed = discord.Embed(
            title="✅ Position opened",
            description="\n".join(lines),
            color=discord.Color.green(),
            timestamp=datetime.now(),
        )
        await interaction.followup.send(embed=embed)

    @position.command(name="list", description="Open positions with unrealized P&L")
    async def position_list_cmd(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)
        if self.bot.store is None:
            await interaction.followup.send(
                "❌ Supabase is not configured. Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`."
            )
            return

        tracker = PositionTracker(self.bot.store)
        open_pos = await tracker.list_open()
        if not open_pos:
            await interaction.followup.send("No open positions.")
            return

        symbols = {p.symbol for p in open_pos}
        prices, prev = self._price_maps_for_symbols(symbols)
        lines = tracker.compute_pnl(open_pos, prices, prev)
        embed = discord.Embed(
            title="📋 Open positions",
            color=discord.Color.gold(),
            timestamp=datetime.now(),
        )
        for ln in lines[:12]:
            day = ""
            if ln.day_change_dollar is not None:
                day = f" • Day **${float(ln.day_change_dollar):+.2f}**"
            embed.add_field(
                name=f"{ln.symbol} `{ln.position_id}`",
                value=(
                    f"Qty **{ln.quantity}** @ **${ln.cost_basis_per_share}** → "
                    f"**${ln.current_price:.2f}**{day}\n"
                    f"Unrealized **${float(ln.unrealized_dollar):+.2f}** "
                    f"({float(ln.unrealized_pct):+.2f}%)"
                )[:1024],
                inline=False,
            )
        if len(lines) > 12:
            embed.set_footer(text=f"Showing 12 of {len(lines)} lots")
        await interaction.followup.send(embed=embed)

    @position.command(name="close", description="Close all open lots for a symbol")
    @app_commands.describe(symbol="Ticker symbol")
    async def position_close_cmd(self, interaction: discord.Interaction, symbol: str):
        await interaction.response.defer(thinking=True)
        if self.bot.store is None:
            await interaction.followup.send(
                "❌ Supabase is not configured. Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`."
            )
            return

        sym = symbol.strip().upper()
        tracker = PositionTracker(self.bot.store)
        try:
            closed = await tracker.close(sym)
        except Exception as exc:
            await interaction.followup.send(f"❌ Close failed: `{exc}`")
            return

        if not closed:
            await interaction.followup.send(f"No open positions for **{sym}**.")
            return

        ids = ", ".join(str(p.id) for p in closed)
        await interaction.followup.send(
            f"✅ Closed **{len(closed)}** lot(s) for **{sym}**: `{ids}`"
        )

    @position.command(
        name="import",
        description="Bulk-add from CSV (symbol, quantity, cost_basis, opened_at)",
    )
    @app_commands.describe(
        file="CSV attachment",
    )
    async def position_import_cmd(
        self,
        interaction: discord.Interaction,
        file: discord.Attachment,
    ):
        await interaction.response.defer(thinking=True)
        if self.bot.store is None:
            await interaction.followup.send(
                "❌ Supabase is not configured. Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`."
            )
            return

        if file.size and file.size > 2 * 1024 * 1024:
            await interaction.followup.send("❌ File too large (max 2 MB).")
            return
        if not (file.filename or "").lower().endswith(".csv"):
            await interaction.followup.send("❌ Please upload a `.csv` file.")
            return

        raw = await file.read()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            await interaction.followup.send("❌ Could not decode CSV as UTF-8.")
            return

        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            await interaction.followup.send("❌ CSV has no header row.")
            return

        header_map = {h.strip().lower(): h for h in reader.fieldnames if h and h.strip()}
        missing = _POSITION_CSV_COLUMNS - set(header_map.keys())
        if missing:
            await interaction.followup.send(
                f"❌ Missing column(s): **{', '.join(sorted(missing))}** "
                "(expected `symbol, quantity, cost_basis, opened_at`)."
            )
            return

        tracker = PositionTracker(self.bot.store)
        ok, errs = 0, 0
        err_lines: list[str] = []

        for i, raw_row in enumerate(reader, start=2):
            if not any((v or "").strip() for v in raw_row.values()):
                continue
            try:
                sym = raw_row[header_map["symbol"]].strip().upper()
                qty_s = raw_row[header_map["quantity"]].strip()
                cost_s = raw_row[header_map["cost_basis"]].strip()
                opened_s = raw_row[header_map["opened_at"]].strip()
                try:
                    opened = datetime.fromisoformat(opened_s.replace("Z", "+00:00"))
                except ValueError:
                    opened = datetime.fromisoformat(opened_s[:10] + "T12:00:00+00:00")
                if opened.tzinfo is None:
                    opened = opened.replace(tzinfo=timezone.utc)
                await tracker.add(sym, qty_s, cost_s, opened_at=opened)
                ok += 1
            except Exception as exc:
                errs += 1
                if len(err_lines) < 5:
                    err_lines.append(f"row {i}: {exc}")

        msg = f"✅ Imported **{ok}** position(s)."
        if errs:
            msg += f" ⚠️ **{errs}** row(s) failed."
            if err_lines:
                msg += "\n" + "\n".join(err_lines[:5])
        await interaction.followup.send(msg[:2000])

    @app_commands.command(
        name="pnl",
        description="Portfolio summary: total value, unrealized P&L, day change, top movers",
    )
    async def pnl_cmd(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)
        if self.bot.store is None:
            await interaction.followup.send(
                "❌ Supabase is not configured. Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`."
            )
            return

        tracker = PositionTracker(self.bot.store)
        open_pos = await tracker.list_open()
        if not open_pos:
            await interaction.followup.send("No open positions.")
            return

        symbols = {p.symbol for p in open_pos}
        prices, prev = self._price_maps_for_symbols(symbols)
        summary = await tracker.portfolio_summary(prices, prev)

        day_line = "—"
        if summary.total_day_change_dollar is not None:
            day_line = f"${float(summary.total_day_change_dollar):+.2f}"

        embed = discord.Embed(
            title="💰 Portfolio P&L",
            color=discord.Color.blurple(),
            timestamp=datetime.now(),
        )
        embed.add_field(
            name="Value",
            value=f"${float(summary.total_market_value):,.2f}",
            inline=True,
        )
        embed.add_field(
            name="Cost basis",
            value=f"${float(summary.total_cost_total):,.2f}",
            inline=True,
        )
        embed.add_field(
            name="Unrealized",
            value=(
                f"${float(summary.total_unrealized_dollar):+.2f} "
                f"({float(summary.total_unrealized_pct):+.2f}%)"
            ),
            inline=True,
        )
        embed.add_field(name="Day P&L (est.)", value=day_line, inline=False)

        movers = sorted(
            summary.lines,
            key=lambda ln: abs(float(ln.day_change_dollar or 0)),
            reverse=True,
        )[:5]
        if movers:
            block = "\n".join(
                f"**{m.symbol}** — day ${float(m.day_change_dollar or 0):+.2f} • "
                f"unreal ${float(m.unrealized_dollar):+.2f}"
                for m in movers
            )
            embed.add_field(name="Top day movers (by |day $|)", value=block[:1024], inline=False)

        embed.set_footer(text="Prices from yfinance via StockDataFetcher • Not financial advice")
        await interaction.followup.send(embed=embed)

    @app_commands.command(
        name="briefing",
        description="Run today's morning briefing (informational snapshot)",
    )
    async def briefing_cmd(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True, ephemeral=True)
        await deliver_morning_briefing(self.bot, skip_dedup=True, interaction=interaction)


class AlertCommandsCog(commands.Cog):
    """Phase 6 event alerts — status and routing test (never uses briefing channel)."""

    alerts = app_commands.Group(
        name="alerts",
        description="Event-driven alerts (ALERT_CHANNEL_ID)",
    )

    def __init__(self, bot: StockMonitorBot):
        self.bot = bot

    @alerts.command(name="status", description="Alerts recorded in the last 24 hours")
    async def alerts_status(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True, ephemeral=True)
        if self.bot.store is None:
            await interaction.followup.send("❌ Supabase not configured.")
            return
        rows = await self.bot.store.list_alerts_sent_since(
            hours=24, alert_type_prefix="event_"
        )
        if not rows:
            await interaction.followup.send("No rows in **alerts_sent** for the last 24h.")
            return
        lines: list[str] = []
        for r in rows[:30]:
            lines.append(
                f"`{r.get('alert_type', '?')}` · `{r.get('reference_id', '?')}` · "
                f"{r.get('sent_at', '')}"
            )
        embed = discord.Embed(
            title="Event alerts (24h)",
            description="\n".join(lines)[:3900],
            color=discord.Color(0x00796B),
        )
        await interaction.followup.send(embed=embed)

    @alerts.command(name="test", description="Post a dummy alert to the event channel")
    @app_commands.describe(
        alert_type="One of: 8k, form4, premarket, intraday, target, stop, macro",
    )
    async def alerts_test(self, interaction: discord.Interaction, alert_type: str):
        await interaction.response.defer(thinking=True, ephemeral=True)
        if not self.bot.alert_channel_id:
            await interaction.followup.send(
                "❌ **ALERT_CHANNEL_ID** not set — cannot route test alerts."
            )
            return
        try:
            from bot.embeds.alerts import deliver_event_alert
            from core.alerts import dummy_event_alert

            alert = dummy_event_alert(alert_type)
            await deliver_event_alert(self.bot, alert)
        except ValueError as exc:
            await interaction.followup.send(f"❌ {exc}")
            return
        except RuntimeError as exc:
            await interaction.followup.send(f"❌ {exc}")
            return
        await interaction.followup.send(
            f"✅ Sent test **{alert_type.strip()}** to **ALERT_CHANNEL_ID** (morning briefing unchanged)."
        )


class ScheduledTasksCog(commands.Cog):
    """Scheduled: 52-week scan, weekly analysis, morning briefing (9 AM local clock)."""

    def __init__(self, bot: StockMonitorBot):
        self.bot = bot
        self._warned_missing_event_channel = False

    async def cog_load(self) -> None:
        if not self.daily_check.is_running():
            self.daily_check.start()
        if not self.weekly_analysis.is_running():
            self.weekly_analysis.start()
        if not self.morning_briefing_loop.is_running():
            self.morning_briefing_loop.start()
        if not self.event_alerts_loop.is_running():
            self.event_alerts_loop.start()

    async def cog_unload(self) -> None:
        for t in (
            self.daily_check,
            self.weekly_analysis,
            self.morning_briefing_loop,
            self.event_alerts_loop,
        ):
            if t.is_running():
                t.cancel()

    @tasks.loop(time=time(hour=16, minute=0))  # 4 PM daily (after market close)
    async def daily_check(self):
        """Daily check for 52-week signals."""
        from bot.error_notifications import notify_job_failure

        try:
            if not self.bot.alert_channel_id:
                return

            channel = self.bot.get_channel(self.bot.alert_channel_id)
            if not channel:
                return

            signals = self.bot.fetcher.check_signals(threshold_percent=self.bot.alert_threshold)

            if signals:
                embed = discord.Embed(
                    title="🔔 Daily Alert - 52-Week Signals",
                    color=discord.Color.orange(),
                    timestamp=datetime.now(),
                )

                for signal in signals:
                    emoji = "🚀" if signal.signal_type == "52_WEEK_HIGH" else "📉"
                    embed.add_field(
                        name=f"{emoji} {signal.ticker}",
                        value=str(signal),
                        inline=False,
                    )

                await channel.send(embed=embed)
        except Exception as exc:
            await notify_job_failure(self.bot, "daily_check", exc)

    @tasks.loop(time=time(hour=9, minute=0))  # 9 AM — Mondays only inside task
    async def weekly_analysis(self):
        """Weekly regression analysis report (Mondays)."""
        from bot.error_notifications import notify_job_failure

        try:
            if datetime.now().weekday() != 0:
                return

            if not self.bot.alert_channel_id:
                return

            channel = self.bot.get_channel(self.bot.alert_channel_id)
            if not channel:
                return

            await channel.send("📊 **Weekly Analysis Report**\nAnalyzing watchlist trends...")

            for ticker in self.bot.watchlist[:5]:
                df = self.bot.fetcher.fetch_52_week_data(ticker)
                if df is None:
                    continue

                analysis = self.bot.analyzer.predict_weekly_trend(df, ticker)
                if not analysis:
                    continue

                prediction = {
                    "direction": analysis.trend_direction,
                    "change": analysis.predicted_change_percent,
                    "recommendation": analysis.recommendation,
                }
                chart_buffer = self.bot.chart_gen.create_chart_bytes(
                    df, ticker, prediction=prediction
                )

                embed = discord.Embed(
                    title=f"📈 {ticker}",
                    description=f"**{analysis.recommendation}** - {analysis.reasoning}",
                    color=(
                        discord.Color.green()
                        if analysis.recommendation == "BUY"
                        else discord.Color.red()
                        if analysis.recommendation == "SELL"
                        else discord.Color.gold()
                    ),
                )
                embed.add_field(name="Trend", value=analysis.trend_direction, inline=True)
                embed.add_field(
                    name="Forecast",
                    value=f"{analysis.predicted_change_percent:+.1f}%",
                    inline=True,
                )
                embed.add_field(
                    name="Confidence", value=f"{analysis.confidence:.0%}", inline=True
                )

                file = discord.File(chart_buffer, filename=f"{ticker}.png")
                embed.set_image(url=f"attachment://{ticker}.png")

                await channel.send(embed=embed, file=file)
                await asyncio.sleep(1)
        except Exception as exc:
            await notify_job_failure(self.bot, "weekly_analysis", exc)

    @tasks.loop(time=time(hour=9, minute=0))
    async def morning_briefing_loop(self):
        """Weekday morning briefing; dedup via alerts_sent. Set TZ=America/New_York for 9 AM ET."""
        from bot.error_notifications import notify_job_failure

        try:
            if datetime.now().weekday() >= 5:
                return
            if self.bot.store is None:
                return
            _, status = await deliver_morning_briefing(self.bot, skip_dedup=False)
            if status not in ("ok", "already_sent", "no_channel"):
                pass
        except Exception as exc:
            await notify_job_failure(self.bot, "morning_briefing_loop", exc)

    @tasks.loop(minutes=15)
    async def event_alerts_loop(self):
        """Every 15m; session gating + NYSE calendar live in run_event_alert_job."""
        if self.bot.store is None:
            return
        cid = int(self.bot.alert_channel_id or 0)
        if not cid:
            if not self._warned_missing_event_channel:
                print(
                    "warning: ALERT_CHANNEL_ID not set or 0; event alerts disabled "
                    "(does not fall back to BRIEFING_CHANNEL_ID)."
                )
                self._warned_missing_event_channel = True
            return
        token = (os.getenv("DISCORD_TOKEN") or "").strip()
        if not token:
            return
        try:
            await run_event_alert_job(
                self.bot.store,
                self.bot.fetcher,
                set(self.bot.watchlist),
                channel_id=cid,
                bot_token=token,
            )
        except Exception as exc:
            from bot.error_notifications import notify_job_failure

            await notify_job_failure(self.bot, "event_alerts_loop", exc)


# ============== Main ==============

def main() -> None:
    import asyncio

    from bot.health import start_health_server
    from core.logging import configure_structlog

    load_dotenv()
    configure_structlog()
    token = os.getenv("DISCORD_TOKEN")

    if not token:
        print("❌ Error: DISCORD_TOKEN not found in environment!")
        print("Create a .env file with:")
        print("  DISCORD_TOKEN=your_bot_token_here")
        print("  ALERT_CHANNEL_ID=your_channel_id (optional)")
        return

    async def runner() -> None:
        bot = StockMonitorBot()
        await start_health_server(bot)
        async with bot:
            await bot.start(token)

    asyncio.run(runner())


if __name__ == "__main__":
    main()
