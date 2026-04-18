# TQQQ MA200 Strategy Backtest

A comprehensive backtesting framework for the TQQQ MA200 trading strategy with a Dash web dashboard for daily monitoring.

## Strategy Overview

The strategy uses QQQ (Nasdaq-100 ETF) as a signal indicator to trade TQQQ (3x leveraged Nasdaq-100 ETF).

### Trading Rules

| Action | Condition |
|--------|-----------|
| **BUY TQQQ** | QQQ > MA200 × 1.04 **AND** QQQ daily loss >= 1% |
| **SELL TQQQ** | QQQ < MA200 × 0.97 |

### Rationale

- **Buy on dips in uptrends**: Only buy when QQQ is well above its 200-day moving average (strong uptrend) AND has a down day (better entry price)
- **Sell on trend breakdown**: Exit when QQQ falls below a key support level relative to MA200
- **Risk management**: The 3% buffer on the sell side prevents whipsaws during normal volatility

## Project Structure

```
T3_Strategy_Backtest/
├── app.py                          # Dash web dashboard (port 8050)
├── start_dashboard.command         # Double-click to launch dashboard (macOS)
├── api/                            # FastAPI REST API layer
│   ├── __init__.py
│   ├── main.py                     # FastAPI app & endpoints (port 8000)
│   └── models.py                   # Pydantic request/response models
├── strategies/                     # Python modules
│   ├── __init__.py
│   ├── tqqq_ma200_strategy.py      # Core backtesting engine
│   ├── signal_checker.py           # Daily signal checker
│   ├── signal_engine.py            # Cached signal service for API
│   ├── backtest_runner.py          # Structured backtest interface for API
│   ├── feature_store.py            # Parquet-based feature history store
│   ├── signal_scorecard.py         # Signal accuracy evaluation
│   ├── experiment_log.py           # Backtest experiment tracking
│   ├── leveraged_etf_comparison.py # Compare multiple ETFs
│   ├── liquidity_analysis.py       # Market liquidity analysis
│   ├── alerts.py                   # Email/Discord alerts
│   ├── position_sizing.py          # Kelly, volatility-adjusted sizing
│   └── optimizer.py                # Grid search & walk-forward optimization
├── tqqq_ma200_strategy_notebook.ipynb      # Backtest notebook
├── tqqq_signal_checker.ipynb               # Daily signal notebook
├── leveraged_etf_comparison_notebook.ipynb # ETF comparison notebook
├── us_market_liquidity_analysis.ipynb      # Liquidity analysis notebook
├── Dockerfile                      # Container image definition
├── docker-compose.yml              # Multi-service orchestration
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/z26qin/T3_Strategy_Backtest.git
cd T3_Strategy_Backtest
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install yfinance pandas numpy plotly dash dash-bootstrap-components scipy fastapi "uvicorn[standard]" "pydantic>=2.0"
```

## Usage

### Option 1: Web Dashboard (Recommended)

Launch the interactive dashboard:

```bash
python app.py
```

Then open http://127.0.0.1:8050 in your browser.

**On macOS**: Double-click `start_dashboard.command` to launch.

### Option 2: Jupyter Notebooks

```bash
jupyter notebook
```

Open any of the `.ipynb` files for detailed analysis.

### Option 3: Python Scripts

```python
from strategies import TQQQMA200Strategy, StrategyParams

# Run backtest with custom parameters
params = StrategyParams(
    start_date="2018-01-01",
    initial_capital=100000,
    buy_threshold=1.04,
    sell_threshold=0.97
)

strategy = TQQQMA200Strategy(params)
results = strategy.run_full_analysis()

print(f"Total Return: {results['metrics']['strategy']['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['metrics']['strategy']['sharpe']:.2f}")
```

## Dashboard Features

The web dashboard has 6 tabs:

### 1. Daily Signal

Check today's market conditions for BUY/SELL/HOLD signals.

- Current signal status (BUY/SELL/HOLD)
- Market data (QQQ close, daily change, TQQQ price)
- Condition checklist (which conditions are met)
- 60-day price chart with MA200 and buy/sell levels
- Recent price history table

### 2. TQQQ Backtest

Run backtests with customizable parameters.

- Adjustable start date, initial capital, buy/sell thresholds
- Performance metrics comparison (Strategy vs Buy & Hold)
- Trade log with all buy/sell actions
- Portfolio value chart (log scale)
- Drawdown chart

### 3. Position Sizing (NEW)

Advanced backtesting with dynamic position sizing.

- **Full Position**: 100% in or out
- **Kelly Criterion**: Optimal sizing based on win rate and win/loss ratio
- **Volatility-Adjusted**: Size inversely proportional to volatility, with VIX adjustment
- **Scale In/Out**: Gradually build position as price drops from entry

### 4. Optimization (NEW)

Find optimal strategy parameters.

- **Grid Search**: Test all parameter combinations
- **Walk-Forward**: Train on historical data, test on out-of-sample (prevents overfitting)
- Heatmaps showing Sharpe ratio by parameters
- Performance degradation analysis

### 5. ETF Comparison

Compare different ETFs using the same QQQ-based signals.

- Long-term analysis (2018-present): TQQQ, NVDA, TSLA
- Recent analysis (2023-present): TQQQ, NVDA, TSLA, NVDL, TSLL
- Side-by-side performance metrics
- Strategy performance chart
- Buy & Hold comparison chart

### 6. Liquidity Analysis

Monitor US market liquidity conditions.

- Composite Liquidity Index (0-100 scale)
- Volume metrics (SPY volume, dollar volume)
- Volatility metrics (VIX, realized volatility, VIX-RV spread)
- 95% confidence intervals for all metrics
- Liquidity regime classification (High/Normal/Low/Crisis)

### 7. API Docs

Embedded Swagger UI for the FastAPI backend. Test API calls directly from the dashboard or open in a new browser tab.

## REST API

The project includes a FastAPI backend that exposes the strategy as REST endpoints.

### Quick Start

```bash
# Start the API server
uvicorn api.main:app --reload --port 8000

# Open Swagger docs in browser
open http://localhost:8000/docs
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/signal/current` | Current trading signal (BUY/SELL/HOLD) |
| `GET` | `/backtest/run` | Run backtest with custom parameters |
| `GET` | `/backtest/trades` | List all round-trip trade records |
| `POST` | `/alert/subscribe` | Subscribe to signal alerts |
| `GET` | `/features/history` | Query feature history (Parquet-backed) |
| `GET` | `/features/scorecard` | Signal accuracy report with forward returns |
| `GET` | `/experiments` | Backtest experiment log |
| `GET` | `/health` | Health check & service status |

### curl Examples

**Get current signal:**

```bash
curl http://localhost:8000/signal/current | python -m json.tool
```

**Run backtest with custom parameters:**

```bash
curl "http://localhost:8000/backtest/run?start_date=2020-01-01&end_date=2024-12-31&initial_capital=50000" \
  | python -m json.tool
```

**Get trade history:**

```bash
curl "http://localhost:8000/backtest/trades?start_date=2020-01-01" \
  | python -m json.tool
```

**Subscribe to alerts:**

```bash
curl -X POST http://localhost:8000/alert/subscribe \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "signal_types": ["BUY", "SELL"]}' \
  | python -m json.tool
```

**Health check:**

```bash
curl http://localhost:8000/health | python -m json.tool
```

**Query feature history (with column pruning):**

```bash
curl "http://localhost:8000/features/history?start=2025-01-01&columns=ma200,signal" \
  | python -m json.tool
```

**Signal accuracy scorecard:**

```bash
curl http://localhost:8000/features/scorecard | python -m json.tool
```

**Backtest experiment log:**

```bash
curl "http://localhost:8000/experiments?recent=10" | python -m json.tool
```

### API Authentication

API Key authentication is optional and controlled by the `API_KEY` environment variable:

- **Not set** (default): No auth required — ideal for local development
- **Set**: All endpoints (except `/health`) require the `X-API-Key` header

```bash
# Enable API Key auth
export API_KEY=your-secret-key

# Call with auth header
curl -H "X-API-Key: your-secret-key" http://localhost:8000/signal/current
```

## Docker Deployment

Run both the Dash dashboard and FastAPI backend with a single command:

```bash
# Build and start both services
docker-compose up --build

# Or run in background
docker-compose up --build -d
```

| Service | URL | Description |
|---------|-----|-------------|
| Dashboard | http://localhost:8050 | Dash web UI |
| API | http://localhost:8000 | FastAPI REST API |
| API Docs | http://localhost:8000/docs | Swagger UI |

To set an API Key in Docker:

```bash
API_KEY=your-secret-key docker-compose up --build
```

## MLOps: Feature History & Experiment Tracking

The project includes a lightweight MLOps layer for recording, auditing, and evaluating strategy behavior over time. All data is stored as Parquet files in `data/` (git-ignored).

### Feature Store

Every time the signal engine computes a new signal, it automatically records a snapshot of all features (QQQ price, MA200, signal, strength, etc.) to `data/features/daily_features.parquet`.

```python
from strategies.feature_store import FeatureStore

store = FeatureStore()

# Read all history
df = store.read()

# Column pruning — only load what you need (Parquet advantage)
df = store.read(columns=["ma200", "signal"], start="2025-01-01")

print(f"Total rows: {store.count()}")
print(f"Latest date: {store.latest_date()}")
```

### Signal Scorecard

Evaluates the accuracy of historical BUY/SELL signals by checking what actually happened 5, 10, and 20 trading days after each signal.

```python
from strategies.signal_scorecard import SignalScorecard

scorecard = SignalScorecard()
report = scorecard.full_report()

# Example output:
# BUY signals: 12 total
#   5-day avg return: +2.1%, win rate: 72%
#   10-day avg return: +3.5%, win rate: 68%
#   20-day avg return: +1.8%, win rate: 55%
```

### Experiment Log

Every backtest run is automatically logged to `data/experiments/backtest_log.parquet` with full parameters and results.

```python
from strategies.experiment_log import ExperimentLog

log = ExperimentLog()

# View recent experiments
print(log.recent(5))

# Find the best parameter combination
best = log.best_by("sharpe_ratio")
print(f"Best Sharpe: {best['sharpe_ratio']} (start: {best['start_date']})")
```

### Storage Layout

```
data/
├── features/
│   └── daily_features.parquet       # Daily feature snapshots (auto-recorded)
└── experiments/
    └── backtest_log.parquet         # Backtest experiment log (auto-recorded)
```

## Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start_date` | 2015-01-01 | Backtest start date |
| `initial_capital` | $100,000 | Starting portfolio value |
| `ma_period` | 200 | Moving average period (days) |
| `buy_threshold` | 1.04 | Buy when QQQ > MA200 × this value |
| `sell_threshold` | 0.97 | Sell when QQQ < MA200 × this value |
| `daily_loss_threshold` | -0.01 | Required daily loss to trigger buy (-1%) |

## Performance Metrics

The backtester calculates:

- **Total Return**: Cumulative return over the period
- **Annualized Return**: CAGR (Compound Annual Growth Rate)
- **Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return (assuming 0% risk-free rate)
- **Max Drawdown**: Largest peak-to-trough decline
- **Time in Market**: Percentage of days holding TQQQ
- **Number of Trades**: Total buy/sell transactions

## Liquidity Index Methodology

The Composite Liquidity Index combines:

| Component | Weight | Description |
|-----------|--------|-------------|
| Volume Score | 40% | SPY volume relative to 20-day average |
| VIX Score | 40% | Inverse VIX percentile (lower VIX = higher liquidity) |
| Range Score | 20% | Inverse of daily price range (lower volatility = higher liquidity) |

**Regime Classification**:
- GREEN (80-100): High liquidity
- YELLOW (60-80): Normal liquidity
- ORANGE (40-60): Low liquidity
- RED (0-40): Liquidity crisis

## Data Sources

All market data is fetched from Yahoo Finance via the `yfinance` library:

- **QQQ**: Invesco QQQ Trust (Nasdaq-100 ETF)
- **TQQQ**: ProShares UltraPro QQQ (3x leveraged)
- **SPY**: SPDR S&P 500 ETF
- **^VIX**: CBOE Volatility Index
- **TLT**: iShares 20+ Year Treasury Bond ETF
- **HYG**: iShares iBoxx High Yield Corporate Bond ETF

## Notebooks

### tqqq_ma200_strategy_notebook.ipynb
Full backtest analysis with detailed charts and metrics.

### tqqq_signal_checker.ipynb
Check today's signal and recent market conditions.

### leveraged_etf_comparison_notebook.ipynb
Compare strategy performance across different leveraged ETFs and stocks.

### us_market_liquidity_analysis.ipynb
Comprehensive liquidity analysis with 95% confidence intervals using:
- Parametric confidence intervals (t-distribution)
- Bootstrap confidence intervals (10,000 iterations)
- Rolling confidence intervals

## Automated Alerts

Get notified when BUY/SELL signals trigger via Email or Discord.

### Setup

1. **Email (Gmail)**:
   ```bash
   export ALERT_EMAIL_ENABLED=true
   export ALERT_EMAIL_SENDER=your.email@gmail.com
   export ALERT_EMAIL_PASSWORD=your_app_password  # Use App Password, not regular password
   # Default recipient: z26qin@uwaterloo.ca
   ```

2. **Discord**:
   ```bash
   # Create webhook in Server Settings > Integrations > Webhooks
   export ALERT_DISCORD_ENABLED=true
   export ALERT_DISCORD_WEBHOOK_URL=your_webhook_url
   ```

### Usage

```python
from strategies import AlertManager, load_config_from_env, SignalChecker

# Load config from environment
config = load_config_from_env()
alerts = AlertManager(config)

# Check signal and send alert
checker = SignalChecker()
summary = checker.get_summary_dict()

if summary['signal'] in ['BUY', 'SELL']:
    alerts.send_alert(summary['signal'], summary)
```

## Position Sizing Methods

### Kelly Criterion

Optimal position sizing based on historical performance:

```
Kelly % = W - [(1-W) / R]
```
Where W = win rate, R = average win / average loss

Recommended: Use half-Kelly (0.5 fraction) for safety.

### Volatility-Adjusted

Position size inversely proportional to current volatility:

```
Position Size = Target Vol / Current Vol
```

Also adjusts based on VIX levels:
- VIX < 15: Full position
- VIX > 30: 25% position
- Linear interpolation between

### Scale In/Out

Gradually build position as price drops from entry:
- Level 1: 33% at entry
- Level 2: 66% after 2% drop
- Level 3: 100% after 4% drop

## AWS Deployment

### Option 1: EC2 (Recommended for Always-On)

**Instance**: t3.micro (free tier eligible for 12 months)

| Component | Cost/Month |
|-----------|------------|
| t3.micro instance | $0 (free tier) or ~$8.50 |
| EBS storage (8GB) | ~$0.80 |
| Data transfer | ~$0-1 |
| **Total** | **$0-10/month** |

**Setup**:
```bash
# SSH into EC2
ssh -i your-key.pem ec2-user@your-ec2-ip

# Install dependencies
sudo yum update -y
sudo yum install python3 python3-pip git -y

# Clone and setup
git clone https://github.com/z26qin/T3_Strategy_Backtest.git
cd T3_Strategy_Backtest
pip3 install -r requirements.txt

# Run with screen (keeps running after disconnect)
screen -S dashboard
python3 app.py --host 0.0.0.0
# Ctrl+A, D to detach
```

### Option 2: AWS Lambda + EventBridge (Cheapest for Daily Alerts)

For daily signal checking and alerts only (no dashboard):

| Component | Cost/Month |
|-----------|------------|
| Lambda (1 run/day) | ~$0.01 |
| EventBridge | Free |
| **Total** | **~$0.01/month** |

### Option 3: AWS Lightsail (Simplest)

**Instance**: $3.50/month (512MB RAM, 1 vCPU)

| Component | Cost/Month |
|-----------|------------|
| Lightsail instance | $3.50 |
| Static IP | Free |
| **Total** | **$3.50/month** |

### Option 4: ECS Fargate (Scalable)

For production deployments with auto-scaling:

| Component | Cost/Month |
|-----------|------------|
| Fargate (0.25 vCPU, 0.5GB) | ~$9 |
| Load Balancer | ~$16 |
| **Total** | **~$25/month** |

### Recommendation

| Use Case | Best Option | Cost |
|----------|-------------|------|
| Personal daily monitoring | EC2 t3.micro | $0-10/mo |
| Alerts only (no dashboard) | Lambda | $0.01/mo |
| Simple always-on | Lightsail | $3.50/mo |
| Production/scaling | ECS Fargate | $25+/mo |

## License

MIT License

## Disclaimer

This project is for educational and research purposes only. Past performance does not guarantee future results. Trading leveraged ETFs involves significant risk of loss. Always do your own research and consult a financial advisor before making investment decisions.
