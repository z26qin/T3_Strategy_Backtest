"""
Leveraged ETF Comparison - Using QQQ Entry Signals

Compares buying different ETFs/stocks using the same QQQ-based entry/exit signals.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class ComparisonParams:
    """Comparison parameters."""
    start_date: str = "2018-01-01"
    end_date: str = None
    initial_capital: float = 100000
    ma_period: int = 200
    buy_threshold: float = 1.04
    sell_threshold: float = 0.97
    daily_loss_threshold: float = -0.01

    def __post_init__(self):
        if self.end_date is None:
            self.end_date = datetime.now().strftime("%Y-%m-%d")


class LeveragedETFComparison:
    """Compare multiple ETFs/stocks using QQQ-based signals."""

    DEFAULT_ASSETS_LONG = ['TQQQ', 'NVDA', 'TSLA']
    DEFAULT_ASSETS_RECENT = ['TQQQ', 'NVDA', 'TSLA', 'NVDL', 'TSLL']
    COLORS = {'TQQQ': 'blue', 'NVDA': 'green', 'TSLA': 'red', 'TSLL': 'darkred', 'NVDL': 'darkgreen'}

    def __init__(self, params: Optional[ComparisonParams] = None):
        self.params = params or ComparisonParams()
        self.data = {}
        self.signals = None
        self.portfolios = {}
        self.metrics = {}
        self.trade_log = None

    def fetch_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers."""
        extended_start = pd.to_datetime(self.params.start_date) - pd.Timedelta(days=300)

        for ticker in ['QQQ'] + tickers:
            if ticker not in self.data:
                df = yf.download(ticker, start=extended_start, end=self.params.end_date, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                self.data[ticker] = df

        return self.data

    def calculate_signals(self) -> pd.DataFrame:
        """Calculate QQQ-based trading signals."""
        if 'QQQ' not in self.data:
            self.fetch_data([])

        df = self.data['QQQ'].copy()

        # Calculate indicators
        df['MA200'] = df['Close'].rolling(window=self.params.ma_period).mean()
        df['Daily_Return'] = df['Close'].pct_change()
        df['Buy_Level'] = df['MA200'] * self.params.buy_threshold
        df['Sell_Level'] = df['MA200'] * self.params.sell_threshold

        # Filter to backtest period
        df = df[df.index >= self.params.start_date].copy()

        # Buy/Sell conditions
        df['Buy_Condition'] = (
            (df['Close'] > df['Buy_Level']) &
            (df['Daily_Return'] <= self.params.daily_loss_threshold)
        )
        df['Sell_Condition'] = df['Close'] < df['Sell_Level']

        # Generate position signals
        df['Signal'] = 0
        position = 0
        for i in range(len(df)):
            if position == 0 and df['Buy_Condition'].iloc[i]:
                position = 1
                df.iloc[i, df.columns.get_loc('Signal')] = 1
            elif position == 1 and df['Sell_Condition'].iloc[i]:
                position = 0
            else:
                df.iloc[i, df.columns.get_loc('Signal')] = position

        df['Position_Change'] = df['Signal'].diff().fillna(0)

        self.signals = df
        return df

    def run_backtest(self, asset: str) -> pd.DataFrame:
        """Run backtest for a single asset using QQQ signals."""
        if self.signals is None:
            self.calculate_signals()

        if asset not in self.data:
            self.fetch_data([asset])

        asset_data = self.data[asset][self.data[asset].index >= self.params.start_date].copy()

        # Align dates
        common_dates = self.signals.index.intersection(asset_data.index)
        signals_aligned = self.signals.loc[common_dates]
        asset_aligned = asset_data.loc[common_dates]

        # Calculate returns
        asset_returns = asset_aligned['Close'].pct_change().fillna(0)

        portfolio = pd.DataFrame(index=common_dates)
        portfolio['Position'] = signals_aligned['Signal'].shift(1).fillna(0)
        portfolio['Asset_Return'] = asset_returns
        portfolio['Strategy_Return'] = portfolio['Position'] * portfolio['Asset_Return']
        portfolio['Cumulative_Asset'] = (1 + portfolio['Asset_Return']).cumprod()
        portfolio['Cumulative_Strategy'] = (1 + portfolio['Strategy_Return']).cumprod()
        portfolio['Asset_Value'] = self.params.initial_capital * portfolio['Cumulative_Asset']
        portfolio['Strategy_Value'] = self.params.initial_capital * portfolio['Cumulative_Strategy']
        portfolio['Strategy_Drawdown'] = (portfolio['Strategy_Value'] / portfolio['Strategy_Value'].cummax() - 1) * 100
        portfolio['Asset_Drawdown'] = (portfolio['Asset_Value'] / portfolio['Asset_Value'].cummax() - 1) * 100

        self.portfolios[asset] = portfolio
        return portfolio

    def calculate_metrics(self, asset: str) -> Dict:
        """Calculate performance metrics for an asset."""
        if asset not in self.portfolios:
            self.run_backtest(asset)

        portfolio = self.portfolios[asset]
        years = len(portfolio) / 252

        # Strategy metrics
        total_return = (portfolio['Strategy_Value'].iloc[-1] / self.params.initial_capital - 1) * 100
        ann_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
        vol = portfolio['Strategy_Return'].std() * np.sqrt(252) * 100
        sharpe = ann_return / vol if vol > 0 else 0
        max_dd = portfolio['Strategy_Drawdown'].min()

        # Buy & Hold metrics
        bh_total = (portfolio['Asset_Value'].iloc[-1] / self.params.initial_capital - 1) * 100
        bh_ann = ((1 + bh_total / 100) ** (1 / years) - 1) * 100
        bh_vol = portfolio['Asset_Return'].std() * np.sqrt(252) * 100
        bh_sharpe = bh_ann / bh_vol if bh_vol > 0 else 0
        bh_max_dd = portfolio['Asset_Drawdown'].min()

        metrics = {
            'strategy': {
                'total_return': total_return,
                'ann_return': ann_return,
                'volatility': vol,
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'final_value': portfolio['Strategy_Value'].iloc[-1],
            },
            'buy_hold': {
                'total_return': bh_total,
                'ann_return': bh_ann,
                'volatility': bh_vol,
                'sharpe': bh_sharpe,
                'max_drawdown': bh_max_dd,
                'final_value': portfolio['Asset_Value'].iloc[-1],
            },
            'time_in_market': portfolio['Position'].mean() * 100,
        }

        self.metrics[asset] = metrics
        return metrics

    def run_comparison(self, assets: List[str]) -> Dict:
        """Run comparison for multiple assets."""
        self.fetch_data(assets)
        self.calculate_signals()

        for asset in assets:
            self.run_backtest(asset)
            self.calculate_metrics(asset)

        # Get trade log
        trades = self.signals[self.signals['Position_Change'] != 0].copy()
        trades['Action'] = trades['Position_Change'].apply(lambda x: 'BUY' if x == 1 else 'SELL')
        self.trade_log = trades

        return {
            'signals': self.signals,
            'portfolios': self.portfolios,
            'metrics': self.metrics,
            'trade_log': self.trade_log,
        }

    def get_comparison_table(self, assets: List[str]) -> pd.DataFrame:
        """Get comparison table as DataFrame."""
        if not all(asset in self.metrics for asset in assets):
            self.run_comparison(assets)

        rows = []
        for asset in assets:
            m = self.metrics[asset]
            rows.append({
                'Asset': asset,
                'Strategy Return (%)': m['strategy']['total_return'],
                'Strategy Ann Return (%)': m['strategy']['ann_return'],
                'Strategy Volatility (%)': m['strategy']['volatility'],
                'Strategy Sharpe': m['strategy']['sharpe'],
                'Strategy Max DD (%)': m['strategy']['max_drawdown'],
                'Strategy Final ($)': m['strategy']['final_value'],
                'Buy&Hold Return (%)': m['buy_hold']['total_return'],
                'Buy&Hold Ann (%)': m['buy_hold']['ann_return'],
                'Buy&Hold Max DD (%)': m['buy_hold']['max_drawdown'],
            })

        return pd.DataFrame(rows)

    def create_performance_chart(self, assets: List[str]) -> go.Figure:
        """Create performance comparison chart."""
        if not all(asset in self.portfolios for asset in assets):
            self.run_comparison(assets)

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Strategy Performance', 'Drawdowns'),
            row_heights=[0.6, 0.4]
        )

        for asset in assets:
            portfolio = self.portfolios[asset]
            color = self.COLORS.get(asset, 'gray')

            # Strategy performance
            fig.add_trace(
                go.Scatter(x=portfolio.index, y=portfolio['Strategy_Value'],
                          name=asset, line=dict(color=color, width=2)),
                row=1, col=1
            )

            # Drawdown
            fig.add_trace(
                go.Scatter(x=portfolio.index, y=portfolio['Strategy_Drawdown'],
                          name=f'{asset} DD', line=dict(color=color, width=1),
                          fill='tozeroy', showlegend=False),
                row=2, col=1
            )

        fig.update_layout(
            height=700,
            title_text=f"ETF Comparison: Strategy Performance ({self.params.start_date} to {self.params.end_date})",
            hovermode='x unified'
        )

        fig.update_yaxes(type='log', title_text='Portfolio ($)', row=1, col=1)
        fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)

        return fig

    def create_buy_hold_chart(self, assets: List[str]) -> go.Figure:
        """Create buy & hold comparison chart."""
        if not all(asset in self.portfolios for asset in assets):
            self.run_comparison(assets)

        fig = go.Figure()

        for asset in assets:
            portfolio = self.portfolios[asset]
            color = self.COLORS.get(asset, 'gray')

            fig.add_trace(
                go.Scatter(x=portfolio.index, y=portfolio['Asset_Value'],
                          name=f'{asset} B&H', line=dict(color=color, width=2))
            )

        fig.update_layout(
            height=500,
            title_text='Buy & Hold Performance Comparison',
            xaxis_title='Date',
            yaxis_title='Portfolio ($)',
            yaxis_type='log',
            hovermode='x unified'
        )

        return fig

    def run_long_term_analysis(self) -> Dict:
        """Run long-term analysis (2018-present)."""
        self.params.start_date = "2018-01-01"
        return self.run_comparison(self.DEFAULT_ASSETS_LONG)

    def run_recent_analysis(self) -> Dict:
        """Run recent analysis (2023-present) including NVDL, TSLL."""
        self.params.start_date = "2023-01-01"
        return self.run_comparison(self.DEFAULT_ASSETS_RECENT)


if __name__ == "__main__":
    # Long-term comparison
    print("="*80)
    print("LONG-TERM COMPARISON (2018-present)")
    print("="*80)

    comparison = LeveragedETFComparison()
    comparison.run_long_term_analysis()

    table = comparison.get_comparison_table(comparison.DEFAULT_ASSETS_LONG)
    print("\nStrategy Performance:")
    print(table[['Asset', 'Strategy Return (%)', 'Strategy Sharpe', 'Strategy Max DD (%)']].to_string(index=False))

    # Recent comparison
    print("\n" + "="*80)
    print("RECENT COMPARISON (2023-present)")
    print("="*80)

    comparison2 = LeveragedETFComparison()
    comparison2.run_recent_analysis()

    table2 = comparison2.get_comparison_table(comparison2.DEFAULT_ASSETS_RECENT)
    print("\nStrategy Performance:")
    print(table2[['Asset', 'Strategy Return (%)', 'Strategy Sharpe', 'Strategy Max DD (%)']].to_string(index=False))
