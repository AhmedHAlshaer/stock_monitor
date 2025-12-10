# 📈 Stock Monitor Bot

> **An AI-powered Discord bot for intelligent stock analysis combining machine learning ensemble models, LLM-based sentiment analysis, and multi-factor signal generation.**

---

## 🎯 Project Overview

Stock Monitor Bot is a sophisticated Discord-based stock analysis platform that aggregates **7 different signal sources** into a unified trading recommendation. Unlike simple price alert bots, this system employs a weighted ensemble approach combining:

- **Deep Learning** (LSTM neural networks)
- **Gradient Boosting** (XGBoost + LightGBM)
- **Natural Language Processing** (LLM-powered news sentiment)
- **Quantitative Analysis** (technical indicators, seasonality patterns)
- **Fundamental Data** (analyst ratings, insider activity, earnings history)

The bot delivers real-time analysis through Discord slash commands with rich embeds and auto-generated charts.

---

## ✨ Key Features

### 🤖 ML Ensemble Prediction Engine
- **LSTM Neural Network**: Captures temporal dependencies in price sequences using 60-day lookback windows
- **XGBoost Classifier**: Gradient boosting for tabular feature analysis with feature importance ranking
- **LightGBM Classifier**: Fast, efficient tree-based learning for real-time predictions
- **Weighted Ensemble**: Combines model outputs (35% LSTM, 35% XGBoost, 30% LightGBM) for robust predictions

### 📰 LLM-Powered Sentiment Analysis
- Integrates with **DeepSeek API** for intelligent news headline analysis
- Extracts bullish/bearish factors from recent articles
- Falls back to keyword-based analysis when API unavailable
- Processes up to 15 articles per stock with 7-day lookback

### 📊 Multi-Factor Signal Generation
| Factor | Weight | Data Source |
|--------|--------|-------------|
| ML Ensemble | 25% | LSTM + XGBoost + LightGBM |
| News Sentiment | 20% | DeepSeek LLM / Keyword Analysis |
| Analyst Ratings | 15% | Yahoo Finance |
| Earnings History | 15% | Historical beat/miss records |
| Insider Activity | 10% | SEC filings via yfinance |
| Seasonality | 10% | 5-year monthly patterns |
| Technical Score | 5% | RSI, SMA crossovers, momentum |

### 📈 52-Week Signal Detection
- Real-time monitoring for stocks approaching 52-week highs/lows
- Configurable threshold alerts (default: 3%)
- Automated daily scans at market close

### 📉 Visual Chart Generation
- Dark-themed 52-week price charts with volume
- Support/resistance level markers
- Moving average overlays (20 SMA, 50 SMA)
- Prediction annotations with trend direction

---

## 🏗️ Architecture

```
stock_monitor/
├── bot/
│   └── discord_bot.py          # Discord interface & command handlers
├── core/
│   ├── ml_models.py            # LSTM, XGBoost, LightGBM ensemble
│   ├── news_sentiment.py       # LLM integration & sentiment analysis
│   ├── historical_patterns.py  # Earnings & seasonality analysis
│   ├── regression.py           # Ridge regression trend prediction
│   ├── stock_data.py           # Data fetching & 52-week metrics
│   ├── ultimate_signal.py      # Multi-factor signal aggregation
│   └── visualizer.py           # Chart generation with matplotlib
├── charts/                     # Generated chart images
├── requirements.txt
└── README.md
```

### Data Flow
```
User Command → Discord Bot → Ultimate Signal Generator
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ML Ensemble    News Analyzer    Market Data
              (LSTM/XGB/LGB)  (DeepSeek)      (yfinance)
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                           Weighted Score Calculation
                                    │
                                    ▼
                    Signal: STRONG BUY / BUY / HOLD / SELL / STRONG SELL
```

---

## 🛠️ Technical Implementation

### Feature Engineering (70+ Features)
The ML models consume a comprehensive feature set including:

**Price-Based Features**
- Multi-period returns (2, 3, 5, 10, 20 days)
- Momentum indicators across timeframes
- Log returns for normalization

**Technical Indicators**
- RSI (7, 14, 21 periods)
- MACD with signal line crossovers
- Bollinger Bands (position & width)
- Stochastic Oscillator
- ADX trend strength

**Volume Analysis**
- On-Balance Volume (OBV)
- Volume momentum ratios
- Price-volume trend correlation

**Volatility Measures**
- Average True Range (ATR)
- Historical volatility (5, 10, 20 days)
- Bollinger Band width

### Model Training Pipeline
```python
# Ensemble trains on 2 years of historical data
# 80/20 train-test split with time-series awareness
# Early stopping to prevent overfitting
# Automatic feature scaling with MinMaxScaler
```

### Signal Score Calculation
```python
combined_score = (
    0.25 * ml_score +           # ML ensemble prediction
    0.20 * sentiment_score +     # News sentiment (-1 to 1)
    0.15 * analyst_score +       # Analyst consensus
    0.15 * earnings_score +      # Earnings track record
    0.10 * insider_score +       # Insider buying/selling
    0.10 * seasonality_score +   # Monthly patterns
    0.05 * technical_score       # Technical indicators
)

# Signal thresholds
STRONG_BUY:  score >= 0.7
BUY:         score >= 0.3
HOLD:        -0.3 < score < 0.3
SELL:        score <= -0.3
STRONG_SELL: score <= -0.7
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Discord Bot Token ([Get one here](https://discord.com/developers/applications))
- DeepSeek API Key (optional, for enhanced sentiment analysis)

### Installation

```bash
# Clone the repository
git clone https://github.com/AhmedHAlshaer/stock_monitor.git
cd stock_monitor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your tokens
```

### Environment Variables
```env
DISCORD_TOKEN=your_discord_bot_token
ALERT_CHANNEL_ID=your_channel_id        # Optional
DEEPSEEK_API_KEY=your_deepseek_key      # Optional
```

### Run the Bot
```bash
python -m bot.discord_bot
```

---

## 💬 Discord Commands

| Command | Description |
|---------|-------------|
| `/signal TICKER` | **Ultimate signal** - Full ML + sentiment + analyst analysis |
| `/check TICKER` | Quick 52-week high/low status check |
| `/analyze TICKER` | Trend analysis with generated chart |
| `/news TICKER` | Latest news headlines with sentiment |
| `/scan` | Scan entire watchlist for 52-week signals |
| `/summary` | Summary table of all monitored stocks |
| `/watchlist list\|add\|remove` | Manage monitored stocks |
| `/setchannel` | Configure alert channel |

### Example Output: `/signal NVDA`
```
🚀 NVDA - STRONG BUY
Score: +0.85 | Confidence: 78%

🤖 ML Models:     +0.72 (UP)
📰 Sentiment:     +0.45
🎯 Analysts:      +0.85 (BUY)
👔 Insiders:      +0.20
📊 Earnings:      +0.64 (89% beat rate)
📅 Seasonal:      +0.31

🟢 Bullish Factors:
• ML models predict UP (72% confidence)
• Analysts bullish (target $180)
• Strong earnings track record
• 🔥 6 quarter beat streak

⚠️ Warnings:
• Earnings in 12 days
```

---

## 📅 Automated Alerts

The bot runs scheduled tasks for proactive monitoring:

- **Daily (4 PM EST)**: Scans watchlist for 52-week high/low signals
- **Weekly (Monday 9 AM)**: Generates trend analysis reports for top stocks

Enable by running `/setchannel` in your preferred alerts channel.

---

## 🧪 Testing Individual Modules

```bash
# Test ML ensemble
python -m core.ml_models

# Test sentiment analysis
python -m core.news_sentiment

# Test historical patterns
python -m core.historical_patterns

# Test chart generation
python -m core.visualizer
```

---

## 📦 Dependencies

### Core
- `discord.py` - Discord API wrapper
- `yfinance` - Yahoo Finance market data
- `pandas` / `numpy` - Data manipulation
- `scikit-learn` - ML utilities

### Machine Learning
- `tensorflow` - LSTM neural networks
- `xgboost` - Gradient boosting
- `lightgbm` - Fast gradient boosting

### Visualization
- `matplotlib` - Chart generation

### NLP
- `openai` - DeepSeek API client (OpenAI-compatible)

---

## 🔮 Future Enhancements

- [ ] Options flow analysis integration
- [ ] Social media sentiment (Reddit, Twitter/X)
- [ ] Backtesting framework with historical performance
- [ ] Portfolio tracking and P&L reporting
- [ ] Web dashboard with React frontend
- [ ] Real-time WebSocket price updates
- [ ] Multi-timeframe analysis (intraday, weekly, monthly)

---

## ⚠️ Disclaimer

This bot is for **educational and informational purposes only**. It does not constitute financial advice. Stock predictions are statistical estimates based on historical data and should not be the sole basis for investment decisions. Always do your own research and consult with qualified financial advisors.

---

## 📄 License

MIT License 

---

## 👤 Author

**Ahmed Alshaer**  
Computer Science @ Indiana University Bloomington  
[GitHub](https://github.com/AhmedHAlshaer) • [LinkedIn](https://linkedin.com/in/ahmealsh)

---

<p align="center">
  <i>Built with ☕ and a passion for quantitative finance</i>
</p>
