# 📈 Stock Monitor Bot

A Discord bot that monitors stocks for 52-week highs/lows and provides weekly regression-based trend analysis.

## Features

- **52-Week Signal Detection**: Alerts when stocks approach or hit 52-week highs/lows
- **Regression Analysis**: Weekly Mon-Thu trend predictions using machine learning
- **Visual Charts**: 52-week charts with support/resistance levels
- **Automated Alerts**: Daily scans and weekly analysis reports
- **Watchlist Management**: Add/remove stocks to monitor

## Quick Start

### 1. Create Discord Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" → Name it "Stock Monitor"
3. Go to "Bot" tab → Click "Add Bot"
4. Copy the **Token** (keep this secret!)
5. Enable these Privileged Gateway Intents:
   - MESSAGE CONTENT INTENT
6. Go to OAuth2 → URL Generator:
   - Scopes: `bot`, `applications.commands`
   - Bot Permissions: `Send Messages`, `Embed Links`, `Attach Files`
7. Copy the generated URL and open it to invite bot to your server

### 2. Setup Project

```bash
# Clone/download the project
cd stock_monitor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env and add your DISCORD_TOKEN
```

### 3. Run the Bot

```bash
python -m bot.discord_bot
```

## Commands

| Command | Description |
|---------|-------------|
| `/check TICKER` | Check 52-week status of a stock |
| `/analyze TICKER` | Full trend analysis with chart |
| `/scan` | Scan entire watchlist for signals |
| `/summary` | Summary table of all stocks |
| `/watchlist list` | View current watchlist |
| `/watchlist add TICKER` | Add stock to watchlist |
| `/watchlist remove TICKER` | Remove stock from watchlist |
| `/setchannel` | Set current channel for alerts |

## Automated Alerts

- **Daily (4 PM)**: Scans watchlist for 52-week high/low signals
- **Weekly (Monday 9 AM)**: Full regression analysis report for top 5 stocks

To enable: Run `/setchannel` in your preferred alerts channel.

## Project Structure

```
stock_monitor/
├── bot/
│   ├── __init__.py
│   └── discord_bot.py      # Main bot with commands
├── core/
│   ├── __init__.py
│   ├── stock_data.py       # Data fetching & 52-week detection
│   ├── regression.py       # Trend analysis & prediction
│   └── visualizer.py       # Chart generation
├── charts/                 # Generated charts (auto-created)
├── .env                    # Your secrets (create from .env.example)
├── .env.example
├── requirements.txt
└── README.md
```

## Customization

### Default Watchlist
Edit `bot/discord_bot.py`:
```python
self.watchlist = ["AAPL", "GOOGL", "MSFT", ...]  # Your stocks here
```

### Alert Threshold
How close to 52-week high/low to trigger alerts (default 3%):
```python
self.alert_threshold = 3.0  # Change this value
```

### Scheduled Times
Modify the `@tasks.loop(time=...)` decorators in `discord_bot.py`

## How It Works

### 52-Week Detection
Fetches 1 year of historical data and compares current price to the highest high and lowest low. Alerts when within the threshold percentage.

### Regression Analysis
Uses Ridge regression with features including:
- Price momentum (5, 10, 20 day)
- Moving average deviations
- Volatility measures
- RSI
- Volume ratios

Predicts expected 4-day (Mon-Thu) return and generates BUY/SELL/HOLD signals.

## Notes

- Stock data from Yahoo Finance (free, no API key needed)
- Predictions are statistical estimates, not financial advice
- Bot needs to be running continuously for scheduled alerts (consider hosting on a VPS)

## License

MIT - Use at your own risk. This is not financial advice.
# stock_monitor
