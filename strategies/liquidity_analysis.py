"""
US Market Daily Liquidity Analysis

Analyzes daily liquidity conditions in US equity markets with confidence intervals.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class LiquidityParams:
    """Liquidity analysis parameters."""
    lookback_days: int = 730  # 2 years
    confidence_level: float = 0.95
    rolling_window: int = 60


class LiquidityAnalysis:
    """US market liquidity analysis with confidence intervals."""

    TICKERS = ['SPY', 'QQQ', 'IWM', '^VIX', 'TLT', 'HYG']

    def __init__(self, params: Optional[LiquidityParams] = None):
        self.params = params or LiquidityParams()
        self.data = {}
        self.liquidity_df = None
        self.ci_results = {}
        self.regime_ci_results = {}

    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch market data."""
        end_date = datetime.today()
        start_date = end_date - timedelta(days=self.params.lookback_days)

        for ticker in self.TICKERS:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            self.data[ticker] = df

        return self.data

    def calculate_liquidity_metrics(self) -> pd.DataFrame:
        """Calculate all liquidity metrics."""
        if not self.data:
            self.fetch_data()

        df = pd.DataFrame(index=self.data['SPY'].index)

        # Price and volume data
        df['SPY_Close'] = self.data['SPY']['Close']
        df['SPY_Volume'] = self.data['SPY']['Volume']
        df['QQQ_Volume'] = self.data['QQQ']['Volume']
        df['IWM_Volume'] = self.data['IWM']['Volume']
        df['VIX'] = self.data['^VIX']['Close']

        # Derived metrics
        df['SPY_Range'] = (self.data['SPY']['High'] - self.data['SPY']['Low']) / self.data['SPY']['Close'] * 100
        df['SPY_Dollar_Volume'] = self.data['SPY']['Volume'] * self.data['SPY']['Close']
        df['SPY_Return'] = self.data['SPY']['Close'].pct_change() * 100

        df.dropna(inplace=True)

        # Volume metrics
        df['SPY_Volume_MA20'] = df['SPY_Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['SPY_Volume'] / df['SPY_Volume_MA20']
        df['Dollar_Volume_MA20'] = df['SPY_Dollar_Volume'].rolling(20).mean()

        vol_mean = df['SPY_Volume'].rolling(50).mean()
        vol_std = df['SPY_Volume'].rolling(50).std()
        df['Volume_ZScore'] = (df['SPY_Volume'] - vol_mean) / vol_std

        # VIX metrics
        df['VIX_MA20'] = df['VIX'].rolling(20).mean()
        df['VIX_Percentile'] = df['VIX'].rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        df['Realized_Vol'] = df['SPY_Return'].rolling(20).std() * np.sqrt(252)
        df['VIX_RV_Spread'] = df['VIX'] - df['Realized_Vol']

        # Composite index components
        df['Volume_Score'] = self._normalize_to_percentile(df['SPY_Volume'])
        df['VIX_Score'] = 100 - self._normalize_to_percentile(df['VIX'])
        df['Range_Score'] = 100 - self._normalize_to_percentile(df['SPY_Range'])

        # Composite Liquidity Index
        df['Liquidity_Index'] = (
            df['Volume_Score'] * 0.4 +
            df['VIX_Score'] * 0.4 +
            df['Range_Score'] * 0.2
        )
        df['Liquidity_Index_MA5'] = df['Liquidity_Index'].rolling(5).mean()

        # Forward returns
        df['Fwd_Return_1d'] = df['SPY_Return'].shift(-1)
        df['Fwd_Return_5d'] = df['SPY_Close'].pct_change(5).shift(-5) * 100
        df['Fwd_Return_20d'] = df['SPY_Close'].pct_change(20).shift(-20) * 100

        # Regime classification
        df['Regime'] = df['Liquidity_Index_MA5'].apply(self._classify_regime)

        self.liquidity_df = df
        return df

    def _normalize_to_percentile(self, series: pd.Series, window: int = 252) -> pd.Series:
        """Normalize series to percentile."""
        return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]) * 100

    def _classify_regime(self, idx: float) -> str:
        """Classify liquidity regime."""
        if pd.isna(idx):
            return 'Unknown'
        elif idx >= 60:
            return 'High'
        elif idx >= 40:
            return 'Normal'
        else:
            return 'Low'

    def calculate_confidence_intervals(self) -> Dict:
        """Calculate confidence intervals for all metrics."""
        if self.liquidity_df is None:
            self.calculate_liquidity_metrics()

        metrics = {
            'SPY Volume (M)': self.liquidity_df['SPY_Volume'] / 1e6,
            'Dollar Volume ($B)': self.liquidity_df['SPY_Dollar_Volume'] / 1e9,
            'Volume Ratio': self.liquidity_df['Volume_Ratio'],
            'VIX': self.liquidity_df['VIX'],
            'Realized Volatility (%)': self.liquidity_df['Realized_Vol'],
            'VIX-RV Spread': self.liquidity_df['VIX_RV_Spread'],
            'SPY Range (%)': self.liquidity_df['SPY_Range'],
            'Liquidity Index': self.liquidity_df['Liquidity_Index'],
            'SPY Daily Return (%)': self.liquidity_df['SPY_Return'],
        }

        for name, data in metrics.items():
            data_clean = data.dropna()
            n = len(data_clean)
            mean = data_clean.mean()
            std = data_clean.std()
            se = stats.sem(data_clean)
            ci_low, ci_high = stats.t.interval(self.params.confidence_level, n-1, loc=mean, scale=se)

            self.ci_results[name] = {
                'mean': mean,
                'std': std,
                'se': se,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'n': n,
            }

        return self.ci_results

    def calculate_regime_ci(self) -> Dict:
        """Calculate confidence intervals for returns by regime."""
        if self.liquidity_df is None:
            self.calculate_liquidity_metrics()

        return_cols = ['Fwd_Return_1d', 'Fwd_Return_5d', 'Fwd_Return_20d']
        return_names = ['1-Day Forward', '5-Day Forward', '20-Day Forward']

        for ret_col, ret_name in zip(return_cols, return_names):
            self.regime_ci_results[ret_name] = {}

            for regime in ['High', 'Normal', 'Low']:
                regime_data = self.liquidity_df[self.liquidity_df['Regime'] == regime][ret_col].dropna()

                if len(regime_data) > 1:
                    n = len(regime_data)
                    mean = regime_data.mean()
                    std = regime_data.std()
                    se = stats.sem(regime_data)
                    ci_low, ci_high = stats.t.interval(self.params.confidence_level, n-1, loc=mean, scale=se)

                    self.regime_ci_results[ret_name][regime] = {
                        'n': n,
                        'mean': mean,
                        'std': std,
                        'ci_low': ci_low,
                        'ci_high': ci_high,
                    }

        return self.regime_ci_results

    def get_current_status(self) -> Dict:
        """Get current liquidity status."""
        if self.liquidity_df is None:
            self.calculate_liquidity_metrics()

        if not self.ci_results:
            self.calculate_confidence_intervals()

        latest = self.liquidity_df.iloc[-1]
        date = self.liquidity_df.index[-1]

        # Determine status vs CI
        status_checks = []
        metrics_to_check = [
            ('SPY Volume (M)', latest['SPY_Volume'] / 1e6),
            ('Dollar Volume ($B)', latest['SPY_Dollar_Volume'] / 1e9),
            ('Volume Ratio', latest['Volume_Ratio']),
            ('VIX', latest['VIX']),
            ('Realized Volatility (%)', latest['Realized_Vol']),
            ('VIX-RV Spread', latest['VIX_RV_Spread']),
            ('SPY Range (%)', latest['SPY_Range']),
            ('Liquidity Index', latest['Liquidity_Index']),
        ]

        for name, current in metrics_to_check:
            ci = self.ci_results[name]
            if current < ci['ci_low']:
                status = 'BELOW CI'
            elif current > ci['ci_high']:
                status = 'ABOVE CI'
            else:
                status = 'NORMAL'

            status_checks.append({
                'metric': name,
                'current': current,
                'mean': ci['mean'],
                'ci_low': ci['ci_low'],
                'ci_high': ci['ci_high'],
                'status': status,
            })

        # Overall assessment
        liq_idx = latest['Liquidity_Index_MA5']
        if liq_idx >= 65:
            assessment = "HIGH LIQUIDITY - Favorable for large trades"
            color = "GREEN"
        elif liq_idx >= 50:
            assessment = "NORMAL LIQUIDITY - Standard market conditions"
            color = "YELLOW"
        elif liq_idx >= 35:
            assessment = "BELOW AVERAGE - Use caution with size"
            color = "ORANGE"
        else:
            assessment = "LOW LIQUIDITY - Market stress conditions"
            color = "RED"

        return {
            'date': date.strftime('%Y-%m-%d'),
            'spy_volume': latest['SPY_Volume'] / 1e6,
            'dollar_volume': latest['SPY_Dollar_Volume'] / 1e9,
            'volume_ratio': latest['Volume_Ratio'],
            'vix': latest['VIX'],
            'vix_percentile': latest['VIX_Percentile'] * 100 if pd.notna(latest['VIX_Percentile']) else None,
            'realized_vol': latest['Realized_Vol'],
            'vix_rv_spread': latest['VIX_RV_Spread'],
            'liquidity_index': latest['Liquidity_Index_MA5'],
            'regime': self._classify_regime(latest['Liquidity_Index_MA5']),
            'assessment': assessment,
            'color': color,
            'status_vs_ci': status_checks,
        }

    def create_dashboard_chart(self) -> go.Figure:
        """Create main liquidity dashboard chart."""
        if self.liquidity_df is None:
            self.calculate_liquidity_metrics()

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('SPY Price', 'Volume Ratio', 'VIX', 'Liquidity Index'),
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )

        df = self.liquidity_df

        # SPY Price
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SPY_Close'], name='SPY', line=dict(color='blue', width=1.5)),
            row=1, col=1
        )

        # Volume Ratio
        colors = ['green' if x > 1 else 'red' for x in df['Volume_Ratio']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume_Ratio'], name='Volume Ratio',
                  marker_color=colors, opacity=0.6),
            row=2, col=1
        )
        fig.add_hline(y=1, line_dash='dash', line_color='black', row=2, col=1)

        # VIX
        fig.add_trace(
            go.Scatter(x=df.index, y=df['VIX'], name='VIX', line=dict(color='purple', width=1.5)),
            row=3, col=1
        )
        fig.add_hline(y=20, line_dash='dash', line_color='green', row=3, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='red', row=3, col=1)

        # Liquidity Index
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Liquidity_Index_MA5'], name='Liquidity Index',
                      line=dict(color='teal', width=2)),
            row=4, col=1
        )
        fig.add_hline(y=60, line_dash='dash', line_color='green', row=4, col=1)
        fig.add_hline(y=40, line_dash='dash', line_color='red', row=4, col=1)

        fig.update_layout(
            height=900,
            title_text='US Market Liquidity Dashboard',
            showlegend=False,
            hovermode='x unified'
        )

        return fig

    def create_ci_chart(self) -> go.Figure:
        """Create confidence interval visualization."""
        if not self.ci_results:
            self.calculate_confidence_intervals()

        if not self.regime_ci_results:
            self.calculate_regime_ci()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Volume Metrics', 'Volatility Metrics',
                          'Forward Returns by Regime', 'Liquidity Index Distribution'),
            specs=[[{}, {}], [{}, {}]]
        )

        # Volume metrics
        vol_metrics = ['SPY Volume (M)', 'Dollar Volume ($B)']
        y_pos = list(range(len(vol_metrics)))
        means = [self.ci_results[m]['mean'] for m in vol_metrics]
        errors = [(self.ci_results[m]['ci_high'] - self.ci_results[m]['ci_low']) / 2 for m in vol_metrics]

        fig.add_trace(
            go.Bar(y=vol_metrics, x=means, orientation='h',
                  error_x=dict(type='data', array=errors),
                  marker_color=['steelblue', 'coral'], name='Volume'),
            row=1, col=1
        )

        # Volatility metrics
        vix_metrics = ['VIX', 'Realized Volatility (%)', 'VIX-RV Spread']
        means = [self.ci_results[m]['mean'] for m in vix_metrics]
        errors = [(self.ci_results[m]['ci_high'] - self.ci_results[m]['ci_low']) / 2 for m in vix_metrics]

        fig.add_trace(
            go.Bar(y=vix_metrics, x=means, orientation='h',
                  error_x=dict(type='data', array=errors),
                  marker_color=['purple', 'blue', 'green'], name='Volatility'),
            row=1, col=2
        )

        # Forward returns by regime
        regimes = ['High', 'Normal', 'Low']
        for i, (ret_name, color) in enumerate([('1-Day Forward', 'green'),
                                                ('5-Day Forward', 'blue'),
                                                ('20-Day Forward', 'orange')]):
            if ret_name in self.regime_ci_results:
                means = [self.regime_ci_results[ret_name].get(r, {}).get('mean', 0) for r in regimes]
                errors = [(self.regime_ci_results[ret_name].get(r, {}).get('ci_high', 0) -
                          self.regime_ci_results[ret_name].get(r, {}).get('ci_low', 0)) / 2 for r in regimes]

                fig.add_trace(
                    go.Bar(x=regimes, y=means, name=ret_name,
                          error_y=dict(type='data', array=errors),
                          marker_color=color, opacity=0.7),
                    row=2, col=1
                )

        # Liquidity index distribution
        liq_data = self.liquidity_df['Liquidity_Index'].dropna()
        fig.add_trace(
            go.Histogram(x=liq_data, nbinsx=30, name='Distribution',
                        marker_color='teal', opacity=0.6),
            row=2, col=2
        )

        # Add CI lines
        mean_val = self.ci_results['Liquidity Index']['mean']
        ci_low = self.ci_results['Liquidity Index']['ci_low']
        ci_high = self.ci_results['Liquidity Index']['ci_high']

        fig.add_vline(x=mean_val, line_dash='solid', line_color='red', row=2, col=2)
        fig.add_vline(x=ci_low, line_dash='dash', line_color='red', row=2, col=2)
        fig.add_vline(x=ci_high, line_dash='dash', line_color='red', row=2, col=2)

        fig.update_layout(
            height=700,
            title_text='95% Confidence Intervals for Liquidity Metrics',
            showlegend=True,
            barmode='group'
        )

        return fig

    def run_full_analysis(self) -> Dict:
        """Run full liquidity analysis."""
        self.fetch_data()
        self.calculate_liquidity_metrics()
        self.calculate_confidence_intervals()
        self.calculate_regime_ci()

        return {
            'liquidity_df': self.liquidity_df,
            'ci_results': self.ci_results,
            'regime_ci_results': self.regime_ci_results,
            'current_status': self.get_current_status(),
        }


if __name__ == "__main__":
    analysis = LiquidityAnalysis()
    results = analysis.run_full_analysis()

    status = results['current_status']

    print("="*70)
    print("US MARKET LIQUIDITY DASHBOARD")
    print("="*70)
    print(f"Date: {status['date']}")

    print("\n[VOLUME METRICS]")
    print(f"  SPY Volume:           {status['spy_volume']:.1f}M shares")
    print(f"  Volume Ratio:         {status['volume_ratio']:.2f}x")
    print(f"  Dollar Volume:        ${status['dollar_volume']:.2f}B")

    print("\n[VOLATILITY METRICS]")
    print(f"  VIX Level:            {status['vix']:.2f}")
    print(f"  VIX Percentile:       {status['vix_percentile']:.1f}%")
    print(f"  Realized Vol (20d):   {status['realized_vol']:.1f}%")
    print(f"  VIX-RV Spread:        {status['vix_rv_spread']:.1f}")

    print("\n[LIQUIDITY INDEX]")
    print(f"  Current Score:        {status['liquidity_index']:.1f}/100")
    print(f"  Regime:               {status['regime']}")

    print("\n" + "="*70)
    print(f"OVERALL ASSESSMENT: [{status['color']}]")
    print(f"  {status['assessment']}")
    print("="*70)

    print("\n[CURRENT VALUES VS 95% CI]")
    for check in status['status_vs_ci']:
        print(f"  {check['metric']:<25}: {check['current']:>10.2f} [{check['ci_low']:.2f}, {check['ci_high']:.2f}] -> {check['status']}")
