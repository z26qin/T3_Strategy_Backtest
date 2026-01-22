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
│   └── liquidity_analysis.py       # Market liquidity analysis
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

The web dashboard has 4 tabs:

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

### 3. ETF Comparison

Compare different ETFs using the same QQQ-based signals.

- Long-term analysis (2018-present): TQQQ, NVDA, TSLA
- Recent analysis (2023-present): TQQQ, NVDA, TSLA, NVDL, TSLL
- Side-by-side performance metrics
- Strategy performance chart
- Buy & Hold comparison chart

### 4. Liquidity Analysis

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

## License

MIT License

## Disclaimer

This project is for educational and research purposes only. Past performance does not guarantee future results. Trading leveraged ETFs involves significant risk of loss. Always do your own research and consult a financial advisor before making investment decisions.
