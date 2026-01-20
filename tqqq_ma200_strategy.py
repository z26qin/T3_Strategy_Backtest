"""
TQQQ MA200 Backtesting Strategy

Strategy Rules:
- BUY TQQQ when: QQQ > MA200 * 1.04 AND QQQ drops >= 1% intraday (Low vs prev Close)
- SELL when: QQQ < MA200 * 0.97
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class TQQQ_MA200_Strategy:
    def __init__(self, start_date: str, end_date: str, initial_capital: float = 100000):
        """
        Initialize the backtesting strategy.

        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Starting capital in USD
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.ma_period = 200
        self.buy_threshold = 1.04  # QQQ > MA200 * 1.04
        self.sell_threshold = 0.97  # QQQ < MA200 * 0.97
        self.daily_loss_threshold = -0.01  # QQQ daily loss >= 1%

        self.qqq_data = None
        self.tqqq_data = None
        self.signals = None
        self.portfolio = None

    def fetch_data(self):
        """Fetch QQQ and TQQQ historical data."""
        # Fetch extra data for MA200 calculation
        extended_start = pd.to_datetime(self.start_date) - pd.Timedelta(days=300)

        print(f"Fetching QQQ data from {extended_start.date()} to {self.end_date}...")
        self.qqq_data = yf.download('QQQ', start=extended_start, end=self.end_date, progress=False)

        print(f"Fetching TQQQ data from {extended_start.date()} to {self.end_date}...")
        self.tqqq_data = yf.download('TQQQ', start=extended_start, end=self.end_date, progress=False)

        # Handle multi-level columns from yfinance
        if isinstance(self.qqq_data.columns, pd.MultiIndex):
            self.qqq_data.columns = self.qqq_data.columns.get_level_values(0)
        if isinstance(self.tqqq_data.columns, pd.MultiIndex):
            self.tqqq_data.columns = self.tqqq_data.columns.get_level_values(0)

        print(f"QQQ data shape: {self.qqq_data.shape}")
        print(f"TQQQ data shape: {self.tqqq_data.shape}")

    def calculate_indicators(self):
        """Calculate MA200 and intraday drop for QQQ."""
        # Calculate 200-day moving average
        self.qqq_data['MA200'] = self.qqq_data['Close'].rolling(window=self.ma_period).mean()

        # Calculate intraday drop: (Low - Previous Close) / Previous Close
        self.qqq_data['Prev_Close'] = self.qqq_data['Close'].shift(1)
        self.qqq_data['Intraday_Drop'] = (self.qqq_data['Low'] - self.qqq_data['Prev_Close']) / self.qqq_data['Prev_Close']

        # Calculate thresholds
        self.qqq_data['Buy_Level'] = self.qqq_data['MA200'] * self.buy_threshold
        self.qqq_data['Sell_Level'] = self.qqq_data['MA200'] * self.sell_threshold

    def generate_signals(self):
        """Generate buy/sell signals based on strategy rules."""
        # Filter to backtest period
        self.qqq_data = self.qqq_data[self.qqq_data.index >= self.start_date].copy()
        self.tqqq_data = self.tqqq_data[self.tqqq_data.index >= self.start_date].copy()

        # Align TQQQ data with QQQ data
        common_dates = self.qqq_data.index.intersection(self.tqqq_data.index)
        self.qqq_data = self.qqq_data.loc[common_dates]
        self.tqqq_data = self.tqqq_data.loc[common_dates]

        # Initialize signals DataFrame
        self.signals = pd.DataFrame(index=self.qqq_data.index)
        self.signals['QQQ_Close'] = self.qqq_data['Close']
        self.signals['QQQ_Low'] = self.qqq_data['Low']
        self.signals['TQQQ_Close'] = self.tqqq_data['Close']
        self.signals['MA200'] = self.qqq_data['MA200']
        self.signals['Intraday_Drop'] = self.qqq_data['Intraday_Drop']
        self.signals['Buy_Level'] = self.qqq_data['Buy_Level']
        self.signals['Sell_Level'] = self.qqq_data['Sell_Level']

        # Buy condition: QQQ > MA200 * 1.04 AND intraday drop >= 1%
        self.signals['Buy_Condition'] = (
            (self.signals['QQQ_Close'] > self.signals['Buy_Level']) &
            (self.signals['Intraday_Drop'] <= self.daily_loss_threshold)
        )

        # Sell condition: QQQ < MA200 * 0.97
        self.signals['Sell_Condition'] = self.signals['QQQ_Close'] < self.signals['Sell_Level']

        # Generate position signals (1 = long, 0 = cash)
        self.signals['Signal'] = 0
        position = 0

        for i in range(len(self.signals)):
            if position == 0 and self.signals['Buy_Condition'].iloc[i]:
                position = 1
                self.signals.iloc[i, self.signals.columns.get_loc('Signal')] = 1
            elif position == 1 and self.signals['Sell_Condition'].iloc[i]:
                position = 0
                self.signals.iloc[i, self.signals.columns.get_loc('Signal')] = 0
            else:
                self.signals.iloc[i, self.signals.columns.get_loc('Signal')] = position

        # Track position changes for trade logging
        self.signals['Position_Change'] = self.signals['Signal'].diff().fillna(0)

    def run_backtest(self):
        """Run the backtest and calculate portfolio value."""
        self.portfolio = pd.DataFrame(index=self.signals.index)

        # Calculate TQQQ daily returns
        tqqq_returns = self.tqqq_data['Close'].pct_change().fillna(0)

        # Calculate strategy returns (TQQQ returns when in position, 0 when in cash)
        # Use shifted signal to avoid look-ahead bias (buy at next day's open)
        self.portfolio['Position'] = self.signals['Signal'].shift(1).fillna(0)
        self.portfolio['TQQQ_Return'] = tqqq_returns
        self.portfolio['Strategy_Return'] = self.portfolio['Position'] * self.portfolio['TQQQ_Return']

        # Calculate cumulative returns
        self.portfolio['Cumulative_TQQQ'] = (1 + self.portfolio['TQQQ_Return']).cumprod()
        self.portfolio['Cumulative_Strategy'] = (1 + self.portfolio['Strategy_Return']).cumprod()

        # Calculate portfolio value
        self.portfolio['TQQQ_Value'] = self.initial_capital * self.portfolio['Cumulative_TQQQ']
        self.portfolio['Strategy_Value'] = self.initial_capital * self.portfolio['Cumulative_Strategy']

    def calculate_metrics(self) -> dict:
        """Calculate performance metrics."""
        strategy_returns = self.portfolio['Strategy_Return']
        tqqq_returns = self.portfolio['TQQQ_Return']

        # Total return
        total_strategy_return = (self.portfolio['Strategy_Value'].iloc[-1] / self.initial_capital - 1) * 100
        total_tqqq_return = (self.portfolio['TQQQ_Value'].iloc[-1] / self.initial_capital - 1) * 100

        # Annualized return
        trading_days = len(self.portfolio)
        years = trading_days / 252
        annualized_strategy = ((1 + total_strategy_return / 100) ** (1 / years) - 1) * 100
        annualized_tqqq = ((1 + total_tqqq_return / 100) ** (1 / years) - 1) * 100

        # Volatility (annualized)
        strategy_vol = strategy_returns.std() * np.sqrt(252) * 100
        tqqq_vol = tqqq_returns.std() * np.sqrt(252) * 100

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_strategy = annualized_strategy / strategy_vol if strategy_vol > 0 else 0
        sharpe_tqqq = annualized_tqqq / tqqq_vol if tqqq_vol > 0 else 0

        # Maximum drawdown
        strategy_cummax = self.portfolio['Strategy_Value'].cummax()
        strategy_drawdown = (self.portfolio['Strategy_Value'] - strategy_cummax) / strategy_cummax
        max_drawdown_strategy = strategy_drawdown.min() * 100

        tqqq_cummax = self.portfolio['TQQQ_Value'].cummax()
        tqqq_drawdown = (self.portfolio['TQQQ_Value'] - tqqq_cummax) / tqqq_cummax
        max_drawdown_tqqq = tqqq_drawdown.min() * 100

        # Trade statistics
        trades = self.signals[self.signals['Position_Change'] != 0]
        num_trades = len(trades[trades['Position_Change'] == 1])

        # Time in market
        time_in_market = self.portfolio['Position'].mean() * 100

        return {
            'Strategy Total Return (%)': round(total_strategy_return, 2),
            'Buy & Hold TQQQ Return (%)': round(total_tqqq_return, 2),
            'Strategy Annualized Return (%)': round(annualized_strategy, 2),
            'Buy & Hold Annualized Return (%)': round(annualized_tqqq, 2),
            'Strategy Volatility (%)': round(strategy_vol, 2),
            'Buy & Hold Volatility (%)': round(tqqq_vol, 2),
            'Strategy Sharpe Ratio': round(sharpe_strategy, 2),
            'Buy & Hold Sharpe Ratio': round(sharpe_tqqq, 2),
            'Strategy Max Drawdown (%)': round(max_drawdown_strategy, 2),
            'Buy & Hold Max Drawdown (%)': round(max_drawdown_tqqq, 2),
            'Number of Trades': num_trades,
            'Time in Market (%)': round(time_in_market, 2),
            'Final Strategy Value ($)': round(self.portfolio['Strategy_Value'].iloc[-1], 2),
            'Final Buy & Hold Value ($)': round(self.portfolio['TQQQ_Value'].iloc[-1], 2),
        }

    def get_trade_log(self) -> pd.DataFrame:
        """Get a log of all trades."""
        trade_signals = self.signals[self.signals['Position_Change'] != 0].copy()
        trade_log = []

        for date, row in trade_signals.iterrows():
            action = 'BUY' if row['Position_Change'] == 1 else 'SELL'
            trade_log.append({
                'Date': date,
                'Action': action,
                'QQQ_Price': round(row['QQQ_Close'], 2),
                'QQQ_Low': round(row['QQQ_Low'], 2),
                'TQQQ_Price': round(row['TQQQ_Close'], 2),
                'MA200': round(row['MA200'], 2),
                'Intraday_Drop': f"{row['Intraday_Drop']*100:.2f}%"
            })

        return pd.DataFrame(trade_log)

    def plot_results(self, save_path: str = None):
        """Plot backtest results."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Plot 1: Portfolio value comparison
        ax1 = axes[0]
        ax1.plot(self.portfolio.index, self.portfolio['Strategy_Value'],
                label='Strategy', linewidth=1.5, color='blue')
        ax1.plot(self.portfolio.index, self.portfolio['TQQQ_Value'],
                label='Buy & Hold TQQQ', linewidth=1.5, color='orange', alpha=0.7)
        ax1.set_title('Portfolio Value: Strategy vs Buy & Hold TQQQ')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Plot 2: QQQ with MA200 and buy/sell signals
        ax2 = axes[1]
        ax2.plot(self.signals.index, self.signals['QQQ_Close'],
                label='QQQ', linewidth=1, color='black')
        ax2.plot(self.signals.index, self.signals['MA200'],
                label='MA200', linewidth=1, color='blue', alpha=0.7)
        ax2.plot(self.signals.index, self.signals['Buy_Level'],
                label='Buy Level (MA200×1.04)', linewidth=1, linestyle='--', color='green', alpha=0.5)
        ax2.plot(self.signals.index, self.signals['Sell_Level'],
                label='Sell Level (MA200×0.97)', linewidth=1, linestyle='--', color='red', alpha=0.5)

        # Mark buy/sell points
        buys = self.signals[self.signals['Position_Change'] == 1]
        sells = self.signals[self.signals['Position_Change'] == -1]
        ax2.scatter(buys.index, buys['QQQ_Close'], marker='^', color='green', s=100, label='Buy', zorder=5)
        ax2.scatter(sells.index, sells['QQQ_Close'], marker='v', color='red', s=100, label='Sell', zorder=5)

        ax2.set_title('QQQ with MA200 and Trading Signals')
        ax2.set_ylabel('QQQ Price ($)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Position over time
        ax3 = axes[2]
        ax3.fill_between(self.portfolio.index, self.portfolio['Position'],
                        step='post', alpha=0.5, color='blue', label='In Position')
        ax3.set_title('Position Over Time (1 = Long TQQQ, 0 = Cash)')
        ax3.set_ylabel('Position')
        ax3.set_xlabel('Date')
        ax3.set_ylim(-0.1, 1.1)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def run(self, plot: bool = True, save_plot: str = None):
        """Run the complete backtest."""
        print("=" * 60)
        print("TQQQ MA200 Strategy Backtest")
        print("=" * 60)
        print(f"\nStrategy Rules:")
        print(f"  BUY TQQQ when: QQQ > MA200 × {self.buy_threshold} AND QQQ intraday drop >= 1%")
        print(f"  SELL when: QQQ < MA200 × {self.sell_threshold}")
        print(f"\nBacktest Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print("-" * 60)

        # Execute backtest steps
        print("\n[1/4] Fetching data...")
        self.fetch_data()

        print("\n[2/4] Calculating indicators...")
        self.calculate_indicators()

        print("\n[3/4] Generating signals...")
        self.generate_signals()

        print("\n[4/4] Running backtest...")
        self.run_backtest()

        # Display results
        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS")
        print("=" * 60)

        metrics = self.calculate_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 60)
        print("TRADE LOG")
        print("=" * 60)

        trade_log = self.get_trade_log()
        if len(trade_log) > 0:
            print(trade_log.to_string(index=False))
        else:
            print("  No trades executed during backtest period.")

        if plot:
            print("\n" + "=" * 60)
            print("Generating plots...")
            self.plot_results(save_path=save_plot)

        return metrics, trade_log


def main():
    """Main entry point for running the backtest."""
    # Default parameters
    START_DATE = "2015-01-01"
    END_DATE = "2025-01-17"
    INITIAL_CAPITAL = 100000

    # Create and run strategy
    strategy = TQQQ_MA200_Strategy(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL
    )

    metrics, trades = strategy.run(plot=True, save_plot="backtest_results.png")

    return strategy, metrics, trades


if __name__ == "__main__":
    strategy, metrics, trades = main()
