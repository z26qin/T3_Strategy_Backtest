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
├── app.py                          # Dash web dashboard
├── start_dashboard.command         # Double-click to launch dashboard (macOS)
├── strategies/                     # Python modules
│   ├── __init__.py
│   ├── tqqq_ma200_strategy.py      # Core backtesting engine
│   ├── signal_checker.py           # Daily signal checker
│   ├── leveraged_etf_comparison.py # Compare multiple ETFs
│   ├── liquidity_analysis.py       # Market liquidity analysis
│   ├── alerts.py                   # Email/Discord alerts
│   ├── position_sizing.py          # Kelly, volatility-adjusted sizing
│   └── optimizer.py                # Grid search & walk-forward optimization
├── tqqq_ma200_strategy_notebook.ipynb      # Backtest notebook
├── tqqq_signal_checker.ipynb               # Daily signal notebook
├── leveraged_etf_comparison_notebook.ipynb # ETF comparison notebook
├── us_market_liquidity_analysis.ipynb      # Liquidity analysis notebook
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
pip install yfinance pandas numpy plotly dash dash-bootstrap-components scipy
```

Or install all at once:

```bash
pip install yfinance pandas numpy plotly dash dash-bootstrap-components scipy jupyter
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
