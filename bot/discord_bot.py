"""
Stock Monitor Discord Bot
Main bot file with commands and scheduled alerts.
"""

import discord
from discord.ext import commands, tasks
from discord import app_commands
import asyncio
from datetime import datetime, time
from typing import Optional
import os
from dotenv import load_dotenv

# Import our modules
from core.stock_data import StockDataFetcher, StockSignal
from core.regression import RegressionAnalyzer, TrendAnalysis
from core.visualizer import ChartGenerator

load_dotenv()


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
        
        # Alert channel (set via command or env)
        self.alert_channel_id = int(os.getenv("ALERT_CHANNEL_ID", "0"))
        
        # Signal threshold (how close to 52w high/low to alert)
        self.alert_threshold = 3.0  # percent
        
    async def setup_hook(self):
        """Called when bot is ready, before on_ready."""
        # Add cog with slash commands
        await self.add_cog(StockCommands(self))
        
        # Sync commands
        await self.tree.sync()
        print("Slash commands synced!")
        
        # Start scheduled tasks
        self.daily_check.start()
        self.weekly_analysis.start()
        
    async def on_ready(self):
        print(f"✅ {self.user} is online!")
        print(f"📊 Monitoring: {', '.join(self.watchlist)}")
        print(f"🔔 Alert threshold: {self.alert_threshold}%")


class StockCommands(commands.Cog):
    """Slash commands for stock monitoring."""
    
    def __init__(self, bot: StockMonitorBot):
        self.bot = bot
    
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
        action="add, remove, or list",
        ticker="Stock symbol (for add/remove)"
    )
    async def watchlist(
        self, 
        interaction: discord.Interaction, 
        action: str = "list",
        ticker: Optional[str] = None
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
                await interaction.response.send_message(f"✅ Added **{ticker}** to watchlist")
            else:
                await interaction.response.send_message(f"⚠️ **{ticker}** already in watchlist")
                
        elif action == "remove" and ticker:
            ticker = ticker.upper()
            if ticker in self.bot.watchlist:
                self.bot.watchlist.remove(ticker)
                self.bot.fetcher = StockDataFetcher(self.bot.watchlist)
                await interaction.response.send_message(f"✅ Removed **{ticker}** from watchlist")
            else:
                await interaction.response.send_message(f"⚠️ **{ticker}** not in watchlist")
        else:
            await interaction.response.send_message("Usage: `/watchlist list` | `/watchlist add TICKER` | `/watchlist remove TICKER`")
    
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


# ============== Scheduled Tasks ==============

@tasks.loop(time=time(hour=16, minute=0))  # 4 PM daily (after market close)
async def daily_check(self):
    """Daily check for 52-week signals."""
    if not self.alert_channel_id:
        return
        
    channel = self.get_channel(self.alert_channel_id)
    if not channel:
        return
    
    signals = self.fetcher.check_signals(threshold_percent=self.alert_threshold)
    
    if signals:
        embed = discord.Embed(
            title="🔔 Daily Alert - 52-Week Signals",
            color=discord.Color.orange(),
            timestamp=datetime.now()
        )
        
        for signal in signals:
            emoji = "🚀" if signal.signal_type == "52_WEEK_HIGH" else "📉"
            embed.add_field(
                name=f"{emoji} {signal.ticker}",
                value=str(signal),
                inline=False
            )
        
        await channel.send(embed=embed)

# Bind task to bot
StockMonitorBot.daily_check = daily_check


@tasks.loop(time=time(hour=9, minute=0))  # 9 AM on Mondays
async def weekly_analysis(self):
    """Weekly regression analysis report (Mondays)."""
    # Only run on Monday
    if datetime.now().weekday() != 0:
        return
        
    if not self.alert_channel_id:
        return
        
    channel = self.get_channel(self.alert_channel_id)
    if not channel:
        return
    
    await channel.send("📊 **Weekly Analysis Report**\nAnalyzing watchlist trends...")
    
    for ticker in self.watchlist[:5]:  # Limit to avoid spam
        df = self.fetcher.fetch_52_week_data(ticker)
        if df is None:
            continue
            
        analysis = self.analyzer.predict_weekly_trend(df, ticker)
        if not analysis:
            continue
        
        prediction = {
            'direction': analysis.trend_direction,
            'change': analysis.predicted_change_percent,
            'recommendation': analysis.recommendation
        }
        chart_buffer = self.chart_gen.create_chart_bytes(df, ticker, prediction=prediction)
        
        embed = discord.Embed(
            title=f"📈 {ticker}",
            description=f"**{analysis.recommendation}** - {analysis.reasoning}",
            color=discord.Color.green() if analysis.recommendation == "BUY" else 
                  discord.Color.red() if analysis.recommendation == "SELL" else
                  discord.Color.gold()
        )
        embed.add_field(name="Trend", value=analysis.trend_direction, inline=True)
        embed.add_field(name="Forecast", value=f"{analysis.predicted_change_percent:+.1f}%", inline=True)
        embed.add_field(name="Confidence", value=f"{analysis.confidence:.0%}", inline=True)
        
        file = discord.File(chart_buffer, filename=f"{ticker}.png")
        embed.set_image(url=f"attachment://{ticker}.png")
        
        await channel.send(embed=embed, file=file)
        await asyncio.sleep(1)  # Rate limiting

# Bind task to bot
StockMonitorBot.weekly_analysis = weekly_analysis


# ============== Main ==============

def main():
    token = os.getenv("DISCORD_TOKEN")
    
    if not token:
        print("❌ Error: DISCORD_TOKEN not found in environment!")
        print("Create a .env file with:")
        print("  DISCORD_TOKEN=your_bot_token_here")
        print("  ALERT_CHANNEL_ID=your_channel_id (optional)")
        return
    
    bot = StockMonitorBot()
    bot.run(token)


if __name__ == "__main__":
    main()
