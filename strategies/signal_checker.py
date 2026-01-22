"""
TQQQ MA200 Strategy - Daily Signal Checker

Checks today's market conditions to determine if it's a BUY or SELL signal.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import plotly.graph_objects as go


@dataclass
class SignalStatus:
    """Current signal status."""
    date: str
    signal: str  # 'BUY', 'SELL', or 'HOLD'
    current_position: int  # 1 = Long TQQQ, 0 = Cash
    qqq_close: float
    qqq_daily_return: float
    ma200: float
    buy_level: float
    sell_level: float
    tqqq_close: float
    qqq_above_buy_level: bool
    qqq_daily_loss_met: bool
    qqq_below_sell_level: bool
    last_action: Optional[str]
    last_action_date: Optional[str]
    last_action_price: Optional[float]


class SignalChecker:
    """Daily signal checker for TQQQ MA200 strategy."""

    MA_PERIOD = 200
    BUY_THRESHOLD = 1.04
    SELL_THRESHOLD = 0.97
    DAILY_LOSS_THRESHOLD = -0.01

    def __init__(self):
        self.qqq_data = None
        self.tqqq_data = None
        self.signal_status = None

    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch latest market data."""
        start_date = datetime.now() - timedelta(days=365)

        qqq = yf.download('QQQ', start=start_date, progress=False)
        tqqq = yf.download('TQQQ', start=start_date, progress=False)

        if isinstance(qqq.columns, pd.MultiIndex):
            qqq.columns = qqq.columns.get_level_values(0)
        if isinstance(tqqq.columns, pd.MultiIndex):
            tqqq.columns = tqqq.columns.get_level_values(0)

        self.qqq_data = qqq
        self.tqqq_data = tqqq

        return qqq, tqqq

    def calculate_indicators(self) -> pd.DataFrame:
        """Calculate technical indicators."""
        if self.qqq_data is None:
            self.fetch_data()

        df = self.qqq_data.copy()
        df['MA200'] = df['Close'].rolling(window=self.MA_PERIOD).mean()
        df['Daily_Return'] = df['Close'].pct_change()
        df['Buy_Level'] = df['MA200'] * self.BUY_THRESHOLD
        df['Sell_Level'] = df['MA200'] * self.SELL_THRESHOLD

        self.qqq_data = df
        return df

    def get_current_position(self) -> Tuple[int, Optional[str], Optional[datetime], Optional[float]]:
        """Determine current position based on historical signals."""
        if self.qqq_data is None or 'Buy_Level' not in self.qqq_data.columns:
            self.calculate_indicators()

        position = 0
        last_action = None
        last_action_date = None
        last_action_price = None

        for i in range(len(self.qqq_data)):
            row = self.qqq_data.iloc[i]
            date = self.qqq_data.index[i]

            if pd.isna(row['Buy_Level']) or pd.isna(row['Daily_Return']):
                continue

            buy_cond = (row['Close'] > row['Buy_Level']) and (row['Daily_Return'] <= self.DAILY_LOSS_THRESHOLD)
            sell_cond = row['Close'] < row['Sell_Level']

            if position == 0 and buy_cond:
                position = 1
                last_action = 'BUY'
                last_action_date = date
                last_action_price = row['Close']
            elif position == 1 and sell_cond:
                position = 0
                last_action = 'SELL'
                last_action_date = date
                last_action_price = row['Close']

        return position, last_action, last_action_date, last_action_price

    def check_signal(self) -> SignalStatus:
        """Check today's signal."""
        if self.qqq_data is None or 'Buy_Level' not in self.qqq_data.columns:
            self.calculate_indicators()

        today = self.qqq_data.iloc[-1]
        latest_date = self.qqq_data.index[-1]

        # Check conditions
        qqq_above_buy_level = today['Close'] > today['Buy_Level']
        qqq_daily_loss_met = today['Daily_Return'] <= self.DAILY_LOSS_THRESHOLD
        qqq_below_sell_level = today['Close'] < today['Sell_Level']

        buy_signal = qqq_above_buy_level and qqq_daily_loss_met
        sell_signal = qqq_below_sell_level

        # Determine signal
        if buy_signal:
            signal = 'BUY'
        elif sell_signal:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        # Get current position
        position, last_action, last_date, last_price = self.get_current_position()

        self.signal_status = SignalStatus(
            date=latest_date.strftime('%Y-%m-%d'),
            signal=signal,
            current_position=position,
            qqq_close=float(today['Close']),
            qqq_daily_return=float(today['Daily_Return']),
            ma200=float(today['MA200']),
            buy_level=float(today['Buy_Level']),
            sell_level=float(today['Sell_Level']),
            tqqq_close=float(self.tqqq_data['Close'].iloc[-1]),
            qqq_above_buy_level=qqq_above_buy_level,
            qqq_daily_loss_met=qqq_daily_loss_met,
            qqq_below_sell_level=qqq_below_sell_level,
            last_action=last_action,
            last_action_date=last_date.strftime('%Y-%m-%d') if last_date else None,
            last_action_price=float(last_price) if last_price else None,
        )

        return self.signal_status

    def get_recent_history(self, days: int = 10) -> pd.DataFrame:
        """Get recent price history."""
        if self.qqq_data is None or 'Buy_Level' not in self.qqq_data.columns:
            self.calculate_indicators()

        recent = self.qqq_data[['Close', 'Daily_Return', 'MA200', 'Buy_Level', 'Sell_Level']].tail(days).copy()
        recent.columns = ['QQQ_Close', 'Daily_Chg', 'MA200', 'Buy_Level', 'Sell_Level']

        return recent

    def create_chart(self, days: int = 60) -> go.Figure:
        """Create interactive chart."""
        if self.qqq_data is None or 'Buy_Level' not in self.qqq_data.columns:
            self.calculate_indicators()

        plot_data = self.qqq_data.tail(days)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=plot_data.index, y=plot_data['Close'],
            name='QQQ', line=dict(color='black', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=plot_data.index, y=plot_data['MA200'],
            name='MA200', line=dict(color='blue', width=1.5)
        ))

        fig.add_trace(go.Scatter(
            x=plot_data.index, y=plot_data['Buy_Level'],
            name='Buy Level', line=dict(color='green', width=1, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=plot_data.index, y=plot_data['Sell_Level'],
            name='Sell Level', line=dict(color='red', width=1, dash='dash')
        ))

        # Mark today
        fig.add_trace(go.Scatter(
            x=[plot_data.index[-1]], y=[plot_data['Close'].iloc[-1]],
            mode='markers', name='Today',
            marker=dict(color='purple', size=15, symbol='circle')
        ))

        fig.update_layout(
            title=f"QQQ - Last {days} Days ({self.qqq_data.index[-1].strftime('%Y-%m-%d')})",
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            hovermode='x unified'
        )

        return fig

    def get_summary_dict(self) -> Dict:
        """Get summary as dictionary for Dash display."""
        if self.signal_status is None:
            self.check_signal()

        status = self.signal_status

        # Calculate distances
        buy_distance = status.buy_level - status.qqq_close
        buy_distance_pct = (buy_distance / status.qqq_close) * 100
        sell_distance = status.qqq_close - status.sell_level
        sell_distance_pct = (sell_distance / status.qqq_close) * 100

        return {
            'date': status.date,
            'signal': status.signal,
            'position': 'LONG TQQQ' if status.current_position == 1 else 'CASH',
            'qqq_close': status.qqq_close,
            'qqq_daily_return': status.qqq_daily_return * 100,
            'ma200': status.ma200,
            'buy_level': status.buy_level,
            'sell_level': status.sell_level,
            'tqqq_close': status.tqqq_close,
            'conditions': {
                'above_buy_level': status.qqq_above_buy_level,
                'daily_loss_met': status.qqq_daily_loss_met,
                'below_sell_level': status.qqq_below_sell_level,
            },
            'distances': {
                'to_buy_level': buy_distance,
                'to_buy_level_pct': buy_distance_pct,
                'to_sell_level': sell_distance,
                'to_sell_level_pct': sell_distance_pct,
            },
            'last_action': status.last_action,
            'last_action_date': status.last_action_date,
            'last_action_price': status.last_action_price,
        }


if __name__ == "__main__":
    checker = SignalChecker()
    status = checker.check_signal()

    print("="*60)
    print(f"TODAY'S MARKET DATA ({status.date})")
    print("="*60)
    print(f"\nQQQ Close:         ${status.qqq_close:.2f}")
    print(f"QQQ Daily Change:  {status.qqq_daily_return*100:+.2f}%")
    print(f"\nMA200:             ${status.ma200:.2f}")
    print(f"Buy Level:         ${status.buy_level:.2f}")
    print(f"Sell Level:        ${status.sell_level:.2f}")
    print(f"\nTQQQ Close:        ${status.tqqq_close:.2f}")

    print("\n" + "="*60)
    print("CONDITION CHECK")
    print("="*60)
    print(f"\nBUY CONDITIONS:")
    print(f"  [{'✓' if status.qqq_above_buy_level else '✗'}] QQQ > Buy Level")
    print(f"  [{'✓' if status.qqq_daily_loss_met else '✗'}] Daily loss >= 1%")
    print(f"\nSELL CONDITION:")
    print(f"  [{'✓' if status.qqq_below_sell_level else '✗'}] QQQ < Sell Level")

    print("\n" + "="*60)
    print(f"TODAY'S SIGNAL: {status.signal}")
    print("="*60)

    print(f"\nCurrent Position: {'LONG TQQQ' if status.current_position == 1 else 'CASH'}")
    if status.last_action:
        print(f"Last Action: {status.last_action} on {status.last_action_date}")
