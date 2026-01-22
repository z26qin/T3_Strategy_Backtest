"""
Parameter Optimization Module

Implements:
- Grid Search optimization
- Walk-Forward optimization
- Visualization of optimization results
"""

import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import product
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OptimizationParams:
    """Optimization parameters."""
    # Parameter ranges for grid search
    buy_threshold_range: Tuple[float, float, float] = (1.02, 1.08, 0.01)  # (min, max, step)
    sell_threshold_range: Tuple[float, float, float] = (0.94, 0.99, 0.01)
    daily_loss_range: Tuple[float, float, float] = (-0.02, -0.005, 0.005)

    # Walk-forward settings
    train_period_months: int = 24  # Training window
    test_period_months: int = 6    # Testing window
    step_months: int = 3           # Step between windows

    # Optimization target
    target_metric: str = "sharpe"  # sharpe, total_return, max_drawdown


class StrategyOptimizer:
    """Optimize strategy parameters."""

    def __init__(self, params: Optional[OptimizationParams] = None):
        self.params = params or OptimizationParams()
        self.qqq_data = None
        self.tqqq_data = None
        self.grid_results = None
        self.walk_forward_results = None

    def fetch_data(self, start_date: str = "2012-01-01", end_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch historical data."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=300)

        qqq = yf.download('QQQ', start=extended_start, end=end_date, progress=False)
        tqqq = yf.download('TQQQ', start=extended_start, end=end_date, progress=False)

        if isinstance(qqq.columns, pd.MultiIndex):
            qqq.columns = qqq.columns.get_level_values(0)
        if isinstance(tqqq.columns, pd.MultiIndex):
            tqqq.columns = tqqq.columns.get_level_values(0)

        self.qqq_data = qqq
        self.tqqq_data = tqqq

        return qqq, tqqq

    def _run_single_backtest(self, qqq_df: pd.DataFrame, tqqq_df: pd.DataFrame,
                             buy_threshold: float, sell_threshold: float,
                             daily_loss_threshold: float, ma_period: int = 200) -> Dict:
        """Run a single backtest with given parameters."""
        df = qqq_df.copy()

        # Calculate indicators
        df['MA200'] = df['Close'].rolling(window=ma_period).mean()
        df['Daily_Return'] = df['Close'].pct_change()
        df['Buy_Level'] = df['MA200'] * buy_threshold
        df['Sell_Level'] = df['MA200'] * sell_threshold

        # Generate signals
        df['Signal'] = 0
        position = 0
        for i in range(len(df)):
            if pd.isna(df['Buy_Level'].iloc[i]):
                continue
            buy_cond = (df['Close'].iloc[i] > df['Buy_Level'].iloc[i]) and \
                       (df['Daily_Return'].iloc[i] <= daily_loss_threshold)
            sell_cond = df['Close'].iloc[i] < df['Sell_Level'].iloc[i]

            if position == 0 and buy_cond:
                position = 1
            elif position == 1 and sell_cond:
                position = 0
            df.iloc[i, df.columns.get_loc('Signal')] = position

        # Align with TQQQ
        common_dates = df.index.intersection(tqqq_df.index)
        df = df.loc[common_dates]
        tqqq = tqqq_df.loc[common_dates]

        # Calculate returns
        tqqq_returns = tqqq['Close'].pct_change().fillna(0)
        strategy_returns = df['Signal'].shift(1).fillna(0) * tqqq_returns

        # Calculate metrics
        cumulative = (1 + strategy_returns).cumprod()
        years = len(strategy_returns) / 252

        if years <= 0 or cumulative.iloc[-1] <= 0:
            return {
                'total_return': -100,
                'ann_return': -100,
                'sharpe': -10,
                'max_drawdown': -100,
                'volatility': 100,
                'num_trades': 0,
            }

        total_return = (cumulative.iloc[-1] - 1) * 100
        ann_return = ((cumulative.iloc[-1]) ** (1 / years) - 1) * 100
        vol = strategy_returns.std() * np.sqrt(252) * 100
        sharpe = ann_return / vol if vol > 0 else 0
        drawdown = (cumulative / cumulative.cummax() - 1) * 100
        max_dd = drawdown.min()
        num_trades = (df['Signal'].diff().fillna(0) != 0).sum()

        return {
            'total_return': total_return,
            'ann_return': ann_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'volatility': vol,
            'num_trades': num_trades,
        }

    def grid_search(self, start_date: str = "2015-01-01", end_date: str = None) -> pd.DataFrame:
        """
        Run grid search over parameter combinations.

        Returns DataFrame with results for each combination.
        """
        if self.qqq_data is None:
            self.fetch_data(start_date, end_date)

        # Filter data to date range
        qqq = self.qqq_data[self.qqq_data.index >= start_date].copy()
        tqqq = self.tqqq_data[self.tqqq_data.index >= start_date].copy()

        if end_date:
            qqq = qqq[qqq.index <= end_date]
            tqqq = tqqq[tqqq.index <= end_date]

        # Generate parameter combinations
        buy_thresholds = np.arange(*self.params.buy_threshold_range)
        sell_thresholds = np.arange(*self.params.sell_threshold_range)
        daily_losses = np.arange(*self.params.daily_loss_range)

        results = []
        total_combinations = len(buy_thresholds) * len(sell_thresholds) * len(daily_losses)
        print(f"Running grid search over {total_combinations} combinations...")

        for i, (buy_t, sell_t, daily_l) in enumerate(product(buy_thresholds, sell_thresholds, daily_losses)):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{total_combinations}")

            metrics = self._run_single_backtest(qqq, tqqq, buy_t, sell_t, daily_l)

            results.append({
                'buy_threshold': buy_t,
                'sell_threshold': sell_t,
                'daily_loss_threshold': daily_l,
                **metrics
            })

        self.grid_results = pd.DataFrame(results)
        return self.grid_results

    def get_best_params(self, metric: str = None) -> Dict:
        """Get best parameters based on target metric."""
        if self.grid_results is None:
            raise ValueError("Run grid_search first")

        metric = metric or self.params.target_metric

        if metric == "max_drawdown":
            # Higher (less negative) is better for drawdown
            best_idx = self.grid_results[metric].idxmax()
        else:
            best_idx = self.grid_results[metric].idxmax()

        best_row = self.grid_results.loc[best_idx]

        return {
            'buy_threshold': best_row['buy_threshold'],
            'sell_threshold': best_row['sell_threshold'],
            'daily_loss_threshold': best_row['daily_loss_threshold'],
            'metrics': {
                'total_return': best_row['total_return'],
                'ann_return': best_row['ann_return'],
                'sharpe': best_row['sharpe'],
                'max_drawdown': best_row['max_drawdown'],
            }
        }

    def walk_forward_optimization(self, start_date: str = "2015-01-01",
                                   end_date: str = None) -> pd.DataFrame:
        """
        Perform walk-forward optimization.

        This prevents overfitting by:
        1. Training on historical data
        2. Testing on out-of-sample data
        3. Rolling forward and repeating
        """
        if self.qqq_data is None:
            self.fetch_data(start_date, end_date)

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        train_months = self.params.train_period_months
        test_months = self.params.test_period_months
        step_months = self.params.step_months

        results = []
        window_start = start

        print("Running walk-forward optimization...")

        while True:
            train_end = window_start + pd.DateOffset(months=train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)

            if test_end > end:
                break

            print(f"  Window: Train {window_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}, "
                  f"Test {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}")

            # Get training data
            qqq_train = self.qqq_data[(self.qqq_data.index >= window_start) &
                                       (self.qqq_data.index < train_end)].copy()
            tqqq_train = self.tqqq_data[(self.tqqq_data.index >= window_start) &
                                         (self.tqqq_data.index < train_end)].copy()

            # Grid search on training data
            best_sharpe = -np.inf
            best_params = None

            buy_thresholds = np.arange(*self.params.buy_threshold_range)
            sell_thresholds = np.arange(*self.params.sell_threshold_range)
            daily_losses = np.arange(*self.params.daily_loss_range)

            for buy_t, sell_t, daily_l in product(buy_thresholds, sell_thresholds, daily_losses):
                metrics = self._run_single_backtest(qqq_train, tqqq_train, buy_t, sell_t, daily_l)
                if metrics['sharpe'] > best_sharpe:
                    best_sharpe = metrics['sharpe']
                    best_params = (buy_t, sell_t, daily_l)

            # Test on out-of-sample data
            qqq_test = self.qqq_data[(self.qqq_data.index >= test_start) &
                                      (self.qqq_data.index < test_end)].copy()
            tqqq_test = self.tqqq_data[(self.tqqq_data.index >= test_start) &
                                        (self.tqqq_data.index < test_end)].copy()

            test_metrics = self._run_single_backtest(qqq_test, tqqq_test, *best_params)

            results.append({
                'train_start': window_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
                'best_buy_threshold': best_params[0],
                'best_sell_threshold': best_params[1],
                'best_daily_loss': best_params[2],
                'train_sharpe': best_sharpe,
                'test_sharpe': test_metrics['sharpe'],
                'test_return': test_metrics['total_return'],
                'test_max_dd': test_metrics['max_drawdown'],
            })

            window_start += pd.DateOffset(months=step_months)

        self.walk_forward_results = pd.DataFrame(results)
        return self.walk_forward_results

    def create_heatmap(self, x_param: str = 'buy_threshold',
                       y_param: str = 'sell_threshold',
                       metric: str = 'sharpe') -> go.Figure:
        """Create heatmap of optimization results."""
        if self.grid_results is None:
            raise ValueError("Run grid_search first")

        # Pivot data for heatmap
        pivot = self.grid_results.pivot_table(
            values=metric,
            index=y_param,
            columns=x_param,
            aggfunc='mean'
        )

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[f"{x:.2f}" for x in pivot.columns],
            y=[f"{y:.2f}" for y in pivot.index],
            colorscale='RdYlGn',
            colorbar_title=metric.replace('_', ' ').title()
        ))

        fig.update_layout(
            title=f'Parameter Optimization: {metric.replace("_", " ").title()}',
            xaxis_title=x_param.replace('_', ' ').title(),
            yaxis_title=y_param.replace('_', ' ').title(),
            height=500
        )

        return fig

    def create_optimization_dashboard(self) -> go.Figure:
        """Create comprehensive optimization dashboard."""
        if self.grid_results is None:
            raise ValueError("Run grid_search first")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Sharpe Ratio by Parameters',
                'Total Return by Parameters',
                'Max Drawdown by Parameters',
                'Parameter Distribution'
            )
        )

        # Sharpe heatmap
        pivot_sharpe = self.grid_results.pivot_table(
            values='sharpe', index='sell_threshold', columns='buy_threshold', aggfunc='mean'
        )
        fig.add_trace(
            go.Heatmap(z=pivot_sharpe.values, x=pivot_sharpe.columns, y=pivot_sharpe.index,
                      colorscale='RdYlGn', showscale=False),
            row=1, col=1
        )

        # Return heatmap
        pivot_return = self.grid_results.pivot_table(
            values='total_return', index='sell_threshold', columns='buy_threshold', aggfunc='mean'
        )
        fig.add_trace(
            go.Heatmap(z=pivot_return.values, x=pivot_return.columns, y=pivot_return.index,
                      colorscale='RdYlGn', showscale=False),
            row=1, col=2
        )

        # Drawdown heatmap
        pivot_dd = self.grid_results.pivot_table(
            values='max_drawdown', index='sell_threshold', columns='buy_threshold', aggfunc='mean'
        )
        fig.add_trace(
            go.Heatmap(z=pivot_dd.values, x=pivot_dd.columns, y=pivot_dd.index,
                      colorscale='RdYlGn', showscale=False),
            row=2, col=1
        )

        # Parameter distribution (scatter)
        fig.add_trace(
            go.Scatter(
                x=self.grid_results['buy_threshold'],
                y=self.grid_results['sharpe'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=self.grid_results['total_return'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title='Return %', x=1.02)
                ),
                name='Combinations'
            ),
            row=2, col=2
        )

        fig.update_layout(height=800, title_text="Parameter Optimization Dashboard")
        return fig

    def create_walk_forward_chart(self) -> go.Figure:
        """Create walk-forward optimization results chart."""
        if self.walk_forward_results is None:
            raise ValueError("Run walk_forward_optimization first")

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Train vs Test Sharpe Ratio', 'Optimal Parameters Over Time'),
            vertical_spacing=0.15
        )

        df = self.walk_forward_results

        # Sharpe comparison
        fig.add_trace(
            go.Bar(x=df['test_start'], y=df['train_sharpe'], name='Train Sharpe',
                  marker_color='blue', opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=df['test_start'], y=df['test_sharpe'], name='Test Sharpe',
                  marker_color='green', opacity=0.7),
            row=1, col=1
        )

        # Parameters over time
        fig.add_trace(
            go.Scatter(x=df['test_start'], y=df['best_buy_threshold'],
                      name='Buy Threshold', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['test_start'], y=df['best_sell_threshold'],
                      name='Sell Threshold', line=dict(color='red')),
            row=2, col=1
        )

        fig.update_layout(
            height=700,
            title_text="Walk-Forward Optimization Results",
            barmode='group'
        )

        return fig

    def get_optimization_summary(self) -> str:
        """Get text summary of optimization results."""
        summary = []
        summary.append("=" * 60)
        summary.append("OPTIMIZATION SUMMARY")
        summary.append("=" * 60)

        if self.grid_results is not None:
            best = self.get_best_params()
            summary.append("\nGRID SEARCH RESULTS:")
            summary.append(f"  Best Buy Threshold: {best['buy_threshold']:.2f}")
            summary.append(f"  Best Sell Threshold: {best['sell_threshold']:.2f}")
            summary.append(f"  Best Daily Loss Threshold: {best['daily_loss_threshold']:.3f}")
            summary.append(f"  Sharpe Ratio: {best['metrics']['sharpe']:.2f}")
            summary.append(f"  Total Return: {best['metrics']['total_return']:.2f}%")
            summary.append(f"  Max Drawdown: {best['metrics']['max_drawdown']:.2f}%")

        if self.walk_forward_results is not None:
            summary.append("\nWALK-FORWARD RESULTS:")
            avg_train = self.walk_forward_results['train_sharpe'].mean()
            avg_test = self.walk_forward_results['test_sharpe'].mean()
            degradation = (avg_train - avg_test) / avg_train * 100 if avg_train > 0 else 0

            summary.append(f"  Avg Train Sharpe: {avg_train:.2f}")
            summary.append(f"  Avg Test Sharpe: {avg_test:.2f}")
            summary.append(f"  Performance Degradation: {degradation:.1f}%")
            summary.append(f"  Number of Windows: {len(self.walk_forward_results)}")

            if degradation > 30:
                summary.append("\n  WARNING: High degradation suggests overfitting!")
            elif degradation < 10:
                summary.append("\n  GOOD: Parameters appear robust.")

        return "\n".join(summary)


if __name__ == "__main__":
    # Example usage
    optimizer = StrategyOptimizer()

    # Run grid search
    print("Starting Grid Search...")
    results = optimizer.grid_search(start_date="2018-01-01")

    print("\nBest Parameters:")
    best = optimizer.get_best_params()
    print(f"  Buy Threshold: {best['buy_threshold']:.2f}")
    print(f"  Sell Threshold: {best['sell_threshold']:.2f}")
    print(f"  Daily Loss: {best['daily_loss_threshold']:.3f}")
    print(f"  Sharpe: {best['metrics']['sharpe']:.2f}")

    # Run walk-forward optimization
    print("\nStarting Walk-Forward Optimization...")
    wf_results = optimizer.walk_forward_optimization(start_date="2016-01-01")

    print(optimizer.get_optimization_summary())
