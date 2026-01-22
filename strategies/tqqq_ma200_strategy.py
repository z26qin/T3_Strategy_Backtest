"""
TQQQ MA200 Backtesting Strategy

Strategy Rules:
- BUY TQQQ when: QQQ > MA200 × 1.04 AND QQQ daily loss >= 1%
- SELL when: QQQ < MA200 × 0.97
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class StrategyParams:
    """Strategy parameters."""
    start_date: str = "2015-01-01"
    end_date: str = None
    initial_capital: float = 100000
    ma_period: int = 200
    buy_threshold: float = 1.04
    sell_threshold: float = 0.97
    daily_loss_threshold: float = -0.01

    def __post_init__(self):
        if self.end_date is None:
            self.end_date = datetime.now().strftime("%Y-%m-%d")


class TQQQMA200Strategy:
    """TQQQ MA200 backtesting strategy."""

    def __init__(self, params: Optional[StrategyParams] = None):
        self.params = params or StrategyParams()
        self.qqq_data = None
        self.tqqq_data = None
        self.signals = None
        self.portfolio = None
        self.metrics = None
        self.trade_log = None

    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch QQQ and TQQQ data."""
        extended_start = pd.to_datetime(self.params.start_date) - pd.Timedelta(days=300)

        qqq_data = yf.download('QQQ', start=extended_start, end=self.params.end_date, progress=False)
        tqqq_data = yf.download('TQQQ', start=extended_start, end=self.params.end_date, progress=False)

        # Handle multi-level columns
        if isinstance(qqq_data.columns, pd.MultiIndex):
            qqq_data.columns = qqq_data.columns.get_level_values(0)
        if isinstance(tqqq_data.columns, pd.MultiIndex):
            tqqq_data.columns = tqqq_data.columns.get_level_values(0)

        self.qqq_data = qqq_data
        self.tqqq_data = tqqq_data

        return qqq_data, tqqq_data

    def calculate_indicators(self) -> pd.DataFrame:
        """Calculate technical indicators."""
        if self.qqq_data is None:
            self.fetch_data()

        df = self.qqq_data.copy()
        df['MA200'] = df['Close'].rolling(window=self.params.ma_period).mean()
        df['Daily_Return'] = df['Close'].pct_change()
        df['Buy_Level'] = df['MA200'] * self.params.buy_threshold
        df['Sell_Level'] = df['MA200'] * self.params.sell_threshold

        return df

    def generate_signals(self) -> pd.DataFrame:
        """Generate trading signals."""
        qqq_data = self.calculate_indicators()

        # Filter to backtest period
        qqq_data = qqq_data[qqq_data.index >= self.params.start_date].copy()
        tqqq_data = self.tqqq_data[self.tqqq_data.index >= self.params.start_date].copy()

        # Align data
        common_dates = qqq_data.index.intersection(tqqq_data.index)
        qqq_data = qqq_data.loc[common_dates]
        tqqq_data = tqqq_data.loc[common_dates]

        # Create signals DataFrame
        signals = pd.DataFrame(index=qqq_data.index)
        signals['QQQ_Close'] = qqq_data['Close']
        signals['TQQQ_Close'] = tqqq_data['Close']
        signals['MA200'] = qqq_data['MA200']
        signals['Daily_Return'] = qqq_data['Daily_Return']
        signals['Buy_Level'] = qqq_data['Buy_Level']
        signals['Sell_Level'] = qqq_data['Sell_Level']

        # Buy/Sell conditions
        signals['Buy_Condition'] = (
            (signals['QQQ_Close'] > signals['Buy_Level']) &
            (signals['Daily_Return'] <= self.params.daily_loss_threshold)
        )
        signals['Sell_Condition'] = signals['QQQ_Close'] < signals['Sell_Level']

        # Generate position signals
        signals['Signal'] = 0
        position = 0
        for i in range(len(signals)):
            if position == 0 and signals['Buy_Condition'].iloc[i]:
                position = 1
                signals.iloc[i, signals.columns.get_loc('Signal')] = 1
            elif position == 1 and signals['Sell_Condition'].iloc[i]:
                position = 0
            else:
                signals.iloc[i, signals.columns.get_loc('Signal')] = position

        signals['Position_Change'] = signals['Signal'].diff().fillna(0)

        self.signals = signals
        self.tqqq_data = tqqq_data

        return signals

    def run_backtest(self) -> pd.DataFrame:
        """Run backtest simulation."""
        if self.signals is None:
            self.generate_signals()

        tqqq_returns = self.tqqq_data['Close'].pct_change().fillna(0)

        portfolio = pd.DataFrame(index=self.signals.index)
        portfolio['Position'] = self.signals['Signal'].shift(1).fillna(0)
        portfolio['TQQQ_Return'] = tqqq_returns
        portfolio['Strategy_Return'] = portfolio['Position'] * portfolio['TQQQ_Return']
        portfolio['Cumulative_TQQQ'] = (1 + portfolio['TQQQ_Return']).cumprod()
        portfolio['Cumulative_Strategy'] = (1 + portfolio['Strategy_Return']).cumprod()
        portfolio['TQQQ_Value'] = self.params.initial_capital * portfolio['Cumulative_TQQQ']
        portfolio['Strategy_Value'] = self.params.initial_capital * portfolio['Cumulative_Strategy']
        portfolio['Strategy_Drawdown'] = (portfolio['Strategy_Value'] / portfolio['Strategy_Value'].cummax() - 1) * 100
        portfolio['TQQQ_Drawdown'] = (portfolio['TQQQ_Value'] / portfolio['TQQQ_Value'].cummax() - 1) * 100

        self.portfolio = portfolio

        return portfolio

    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if self.portfolio is None:
            self.run_backtest()

        portfolio = self.portfolio
        initial_capital = self.params.initial_capital

        strategy_returns = portfolio['Strategy_Return']
        tqqq_returns = portfolio['TQQQ_Return']

        # Total return
        total_strategy = (portfolio['Strategy_Value'].iloc[-1] / initial_capital - 1) * 100
        total_tqqq = (portfolio['TQQQ_Value'].iloc[-1] / initial_capital - 1) * 100

        # Annualized return
        years = len(portfolio) / 252
        ann_strategy = ((1 + total_strategy / 100) ** (1 / years) - 1) * 100
        ann_tqqq = ((1 + total_tqqq / 100) ** (1 / years) - 1) * 100

        # Volatility
        strategy_vol = strategy_returns.std() * np.sqrt(252) * 100
        tqqq_vol = tqqq_returns.std() * np.sqrt(252) * 100

        # Sharpe ratio
        sharpe_strategy = ann_strategy / strategy_vol if strategy_vol > 0 else 0
        sharpe_tqqq = ann_tqqq / tqqq_vol if tqqq_vol > 0 else 0

        # Max drawdown
        max_dd_strategy = portfolio['Strategy_Drawdown'].min()
        max_dd_tqqq = portfolio['TQQQ_Drawdown'].min()

        # Time in market
        time_in_market = portfolio['Position'].mean() * 100

        # Trade count
        num_trades = (self.signals['Position_Change'] == 1).sum()

        self.metrics = {
            'strategy': {
                'total_return': total_strategy,
                'ann_return': ann_strategy,
                'volatility': strategy_vol,
                'sharpe': sharpe_strategy,
                'max_drawdown': max_dd_strategy,
                'final_value': portfolio['Strategy_Value'].iloc[-1],
            },
            'buy_hold': {
                'total_return': total_tqqq,
                'ann_return': ann_tqqq,
                'volatility': tqqq_vol,
                'sharpe': sharpe_tqqq,
                'max_drawdown': max_dd_tqqq,
                'final_value': portfolio['TQQQ_Value'].iloc[-1],
            },
            'time_in_market': time_in_market,
            'num_trades': num_trades,
        }

        return self.metrics

    def get_trade_log(self) -> pd.DataFrame:
        """Get trade log."""
        if self.signals is None:
            self.generate_signals()

        trades = self.signals[self.signals['Position_Change'] != 0].copy()
        trades['Action'] = trades['Position_Change'].apply(lambda x: 'BUY' if x == 1 else 'SELL')

        self.trade_log = trades[['Action', 'QQQ_Close', 'TQQQ_Close', 'MA200', 'Daily_Return']].copy()
        self.trade_log.columns = ['Action', 'QQQ_Price', 'TQQQ_Price', 'MA200', 'QQQ_Daily_Return']

        return self.trade_log

    def create_portfolio_chart(self) -> go.Figure:
        """Create interactive portfolio chart."""
        if self.portfolio is None:
            self.run_backtest()

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Portfolio Value', 'QQQ with Signals', 'Position'),
            row_heights=[0.4, 0.4, 0.2]
        )

        # Portfolio value
        fig.add_trace(
            go.Scatter(x=self.portfolio.index, y=self.portfolio['Strategy_Value'],
                      name='Strategy', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.portfolio.index, y=self.portfolio['TQQQ_Value'],
                      name='Buy & Hold TQQQ', line=dict(color='orange', width=2)),
            row=1, col=1
        )

        # QQQ with signals
        fig.add_trace(
            go.Scatter(x=self.signals.index, y=self.signals['QQQ_Close'],
                      name='QQQ', line=dict(color='black', width=1)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.signals.index, y=self.signals['MA200'],
                      name='MA200', line=dict(color='blue', width=1)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.signals.index, y=self.signals['Buy_Level'],
                      name='Buy Level', line=dict(color='green', width=1, dash='dash')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.signals.index, y=self.signals['Sell_Level'],
                      name='Sell Level', line=dict(color='red', width=1, dash='dash')),
            row=2, col=1
        )

        # Buy/Sell markers
        buys = self.signals[self.signals['Position_Change'] == 1]
        sells = self.signals[self.signals['Position_Change'] == -1]

        fig.add_trace(
            go.Scatter(x=buys.index, y=buys['QQQ_Close'], mode='markers',
                      name='Buy', marker=dict(color='green', size=12, symbol='triangle-up')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=sells.index, y=sells['QQQ_Close'], mode='markers',
                      name='Sell', marker=dict(color='red', size=12, symbol='triangle-down')),
            row=2, col=1
        )

        # Position
        fig.add_trace(
            go.Scatter(x=self.portfolio.index, y=self.portfolio['Position'],
                      fill='tozeroy', name='Position', line=dict(color='blue')),
            row=3, col=1
        )

        fig.update_layout(
            height=900,
            title_text="TQQQ MA200 Strategy Backtest",
            showlegend=True,
            hovermode='x unified'
        )

        fig.update_yaxes(type='log', row=1, col=1)
        fig.update_yaxes(title_text='Portfolio ($)', row=1, col=1)
        fig.update_yaxes(title_text='QQQ ($)', row=2, col=1)
        fig.update_yaxes(title_text='Position', row=3, col=1, range=[-0.1, 1.1])

        return fig

    def create_drawdown_chart(self) -> go.Figure:
        """Create drawdown comparison chart."""
        if self.portfolio is None:
            self.run_backtest()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=self.portfolio.index, y=self.portfolio['Strategy_Drawdown'],
                      fill='tozeroy', name='Strategy', line=dict(color='blue'))
        )
        fig.add_trace(
            go.Scatter(x=self.portfolio.index, y=self.portfolio['TQQQ_Drawdown'],
                      fill='tozeroy', name='Buy & Hold TQQQ', line=dict(color='orange'))
        )

        fig.update_layout(
            title='Drawdown Comparison',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            height=400,
            hovermode='x unified'
        )

        return fig

    def run_full_analysis(self) -> Dict:
        """Run full analysis and return all results."""
        self.fetch_data()
        self.generate_signals()
        self.run_backtest()
        self.calculate_metrics()
        self.get_trade_log()

        return {
            'signals': self.signals,
            'portfolio': self.portfolio,
            'metrics': self.metrics,
            'trade_log': self.trade_log,
            'params': self.params,
        }


if __name__ == "__main__":
    # Example usage
    strategy = TQQQMA200Strategy()
    results = strategy.run_full_analysis()

    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)

    metrics = results['metrics']
    print(f"\nStrategy:")
    print(f"  Total Return: {metrics['strategy']['total_return']:.2f}%")
    print(f"  Annualized Return: {metrics['strategy']['ann_return']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['strategy']['sharpe']:.2f}")
    print(f"  Max Drawdown: {metrics['strategy']['max_drawdown']:.2f}%")
    print(f"  Final Value: ${metrics['strategy']['final_value']:,.2f}")

    print(f"\nBuy & Hold TQQQ:")
    print(f"  Total Return: {metrics['buy_hold']['total_return']:.2f}%")
    print(f"  Annualized Return: {metrics['buy_hold']['ann_return']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['buy_hold']['sharpe']:.2f}")
    print(f"  Max Drawdown: {metrics['buy_hold']['max_drawdown']:.2f}%")
    print(f"  Final Value: ${metrics['buy_hold']['final_value']:,.2f}")

    print(f"\nTime in Market: {metrics['time_in_market']:.2f}%")
    print(f"Number of Trades: {metrics['num_trades']}")
