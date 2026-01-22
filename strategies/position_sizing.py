"""
Position Sizing Module

Implements various position sizing strategies:
- Kelly Criterion
- Volatility-adjusted sizing
- Scale in/out
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SizingMethod(Enum):
    """Position sizing methods."""
    FULL = "full"                    # 100% in or out
    KELLY = "kelly"                  # Kelly Criterion
    VOLATILITY = "volatility"        # Volatility-adjusted
    SCALE = "scale"                  # Scale in/out


@dataclass
class PositionSizingParams:
    """Position sizing parameters."""
    method: SizingMethod = SizingMethod.FULL
    max_position: float = 1.0        # Maximum position size (1.0 = 100%)
    min_position: float = 0.0        # Minimum position size
    kelly_fraction: float = 0.5      # Fraction of Kelly to use (half-Kelly recommended)
    vol_target: float = 0.20         # Target annual volatility (20%)
    vol_lookback: int = 20           # Lookback period for volatility calculation
    vix_threshold_low: float = 15    # VIX below this = full position
    vix_threshold_high: float = 30   # VIX above this = minimum position
    scale_levels: int = 3            # Number of scale-in levels
    scale_threshold: float = 0.02    # Price drop between scale levels (2%)


class PositionSizer:
    """Calculate position sizes based on various methods."""

    def __init__(self, params: Optional[PositionSizingParams] = None):
        self.params = params or PositionSizingParams()

    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion fraction.

        Kelly % = W - [(1-W) / R]
        Where:
        - W = Win rate
        - R = Win/Loss ratio (avg_win / avg_loss)
        """
        if avg_loss == 0:
            return 0

        win_loss_ratio = abs(avg_win / avg_loss)
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Apply fraction (half-Kelly is common for safety)
        kelly = kelly * self.params.kelly_fraction

        # Clamp to valid range
        return max(self.params.min_position, min(self.params.max_position, kelly))

    def calculate_kelly_from_returns(self, returns: pd.Series) -> float:
        """Calculate Kelly fraction from a series of trade returns."""
        if len(returns) < 10:
            return self.params.max_position  # Not enough data

        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return self.params.max_position

        win_rate = len(wins) / len(returns)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())

        return self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)

    def calculate_volatility_adjusted_size(self, current_vol: float, vix: Optional[float] = None) -> float:
        """
        Calculate position size based on volatility targeting.

        Position Size = Target Vol / Current Vol

        Also adjusts based on VIX if provided.
        """
        if current_vol <= 0:
            return self.params.max_position

        # Base calculation: vol targeting
        size = self.params.vol_target / current_vol
        size = max(self.params.min_position, min(self.params.max_position, size))

        # Adjust based on VIX if provided
        if vix is not None:
            vix_multiplier = self._get_vix_multiplier(vix)
            size *= vix_multiplier

        return max(self.params.min_position, min(self.params.max_position, size))

    def _get_vix_multiplier(self, vix: float) -> float:
        """Get position multiplier based on VIX level."""
        if vix <= self.params.vix_threshold_low:
            return 1.0
        elif vix >= self.params.vix_threshold_high:
            return 0.25
        else:
            # Linear interpolation between thresholds
            range_vix = self.params.vix_threshold_high - self.params.vix_threshold_low
            position_in_range = (vix - self.params.vix_threshold_low) / range_vix
            return 1.0 - (0.75 * position_in_range)

    def calculate_scale_in_levels(self, entry_price: float, current_price: float,
                                   signal_strength: int = 1) -> Tuple[float, int]:
        """
        Calculate position size for scale-in strategy.

        Returns (position_size, level)
        """
        if signal_strength <= 0:
            return 0.0, 0

        # Calculate how many levels of drop from entry
        if entry_price > 0:
            drop_pct = (entry_price - current_price) / entry_price
            levels_triggered = min(
                int(drop_pct / self.params.scale_threshold) + 1,
                self.params.scale_levels
            )
        else:
            levels_triggered = 1

        # Position size increases with each level
        position_per_level = self.params.max_position / self.params.scale_levels
        total_position = min(levels_triggered * position_per_level, self.params.max_position)

        return total_position, levels_triggered

    def get_position_size(self, signal: int, data: Dict) -> float:
        """
        Get position size based on configured method.

        Args:
            signal: 1 for buy, 0 for no position
            data: Dict containing relevant data (returns, vol, vix, prices, etc.)

        Returns:
            Position size as fraction (0.0 to 1.0)
        """
        if signal == 0:
            return 0.0

        if self.params.method == SizingMethod.FULL:
            return self.params.max_position

        elif self.params.method == SizingMethod.KELLY:
            returns = data.get('trade_returns', pd.Series())
            return self.calculate_kelly_from_returns(returns)

        elif self.params.method == SizingMethod.VOLATILITY:
            current_vol = data.get('realized_vol', 0.20)
            vix = data.get('vix', None)
            return self.calculate_volatility_adjusted_size(current_vol, vix)

        elif self.params.method == SizingMethod.SCALE:
            entry_price = data.get('entry_price', 0)
            current_price = data.get('current_price', 0)
            position, _ = self.calculate_scale_in_levels(entry_price, current_price, signal)
            return position

        return self.params.max_position


class AdvancedBacktester:
    """Backtester with position sizing support."""

    def __init__(self, position_sizer: Optional[PositionSizer] = None):
        self.position_sizer = position_sizer or PositionSizer()
        self.trade_returns = []

    def run_backtest_with_sizing(self, signals_df: pd.DataFrame, asset_df: pd.DataFrame,
                                  vix_df: Optional[pd.DataFrame] = None,
                                  initial_capital: float = 100000) -> pd.DataFrame:
        """
        Run backtest with dynamic position sizing.

        Args:
            signals_df: DataFrame with 'Signal' column (1 = long, 0 = cash)
            asset_df: DataFrame with 'Close' column for the asset to trade
            vix_df: Optional DataFrame with VIX data
            initial_capital: Starting capital

        Returns:
            Portfolio DataFrame
        """
        # Align data
        common_dates = signals_df.index.intersection(asset_df.index)
        signals = signals_df.loc[common_dates]
        asset = asset_df.loc[common_dates]

        if vix_df is not None:
            vix_aligned = vix_df.reindex(common_dates)
        else:
            vix_aligned = None

        # Calculate returns
        asset_returns = asset['Close'].pct_change().fillna(0)

        # Calculate rolling volatility
        rolling_vol = asset_returns.rolling(self.position_sizer.params.vol_lookback).std() * np.sqrt(252)

        # Initialize portfolio
        portfolio = pd.DataFrame(index=common_dates)
        portfolio['Signal'] = signals['Signal'].shift(1).fillna(0)
        portfolio['Asset_Return'] = asset_returns
        portfolio['Position_Size'] = 0.0
        portfolio['Strategy_Return'] = 0.0

        cash = initial_capital
        position_value = 0
        entry_price = None

        for i in range(len(portfolio)):
            date = portfolio.index[i]
            signal = portfolio['Signal'].iloc[i]

            # Gather data for position sizing
            sizing_data = {
                'trade_returns': pd.Series(self.trade_returns) if self.trade_returns else pd.Series(),
                'realized_vol': rolling_vol.iloc[i] if not pd.isna(rolling_vol.iloc[i]) else 0.20,
                'vix': vix_aligned['Close'].iloc[i] if vix_aligned is not None and not pd.isna(vix_aligned['Close'].iloc[i]) else None,
                'entry_price': entry_price or asset['Close'].iloc[i],
                'current_price': asset['Close'].iloc[i],
            }

            # Calculate position size
            position_size = self.position_sizer.get_position_size(int(signal), sizing_data)
            portfolio.iloc[i, portfolio.columns.get_loc('Position_Size')] = position_size

            # Calculate return
            if i > 0:
                prev_position_size = portfolio['Position_Size'].iloc[i-1]
                asset_ret = portfolio['Asset_Return'].iloc[i]
                strategy_ret = prev_position_size * asset_ret
                portfolio.iloc[i, portfolio.columns.get_loc('Strategy_Return')] = strategy_ret

                # Track trade returns for Kelly calculation
                if prev_position_size > 0 and signal == 0:
                    # Closing a trade
                    self.trade_returns.append(strategy_ret)

            # Track entry price for scale-in
            if signal == 1 and (i == 0 or portfolio['Signal'].iloc[i-1] == 0):
                entry_price = asset['Close'].iloc[i]
            elif signal == 0:
                entry_price = None

        # Calculate cumulative values
        portfolio['Cumulative_Asset'] = (1 + portfolio['Asset_Return']).cumprod()
        portfolio['Cumulative_Strategy'] = (1 + portfolio['Strategy_Return']).cumprod()
        portfolio['Asset_Value'] = initial_capital * portfolio['Cumulative_Asset']
        portfolio['Strategy_Value'] = initial_capital * portfolio['Cumulative_Strategy']
        portfolio['Strategy_Drawdown'] = (portfolio['Strategy_Value'] / portfolio['Strategy_Value'].cummax() - 1) * 100
        portfolio['Asset_Drawdown'] = (portfolio['Asset_Value'] / portfolio['Asset_Value'].cummax() - 1) * 100

        return portfolio

    def calculate_metrics(self, portfolio: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """Calculate performance metrics."""
        years = len(portfolio) / 252

        # Strategy metrics
        total_return = (portfolio['Strategy_Value'].iloc[-1] / initial_capital - 1) * 100
        ann_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
        vol = portfolio['Strategy_Return'].std() * np.sqrt(252) * 100
        sharpe = ann_return / vol if vol > 0 else 0
        max_dd = portfolio['Strategy_Drawdown'].min()

        # Buy & Hold metrics
        bh_total = (portfolio['Asset_Value'].iloc[-1] / initial_capital - 1) * 100
        bh_ann = ((1 + bh_total / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
        bh_vol = portfolio['Asset_Return'].std() * np.sqrt(252) * 100
        bh_sharpe = bh_ann / bh_vol if bh_vol > 0 else 0
        bh_max_dd = portfolio['Asset_Drawdown'].min()

        # Average position size
        avg_position = portfolio[portfolio['Position_Size'] > 0]['Position_Size'].mean() * 100

        return {
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
            'avg_position_size': avg_position,
            'num_trades': len(self.trade_returns),
        }


if __name__ == "__main__":
    # Example usage
    print("Position Sizing Examples")
    print("=" * 50)

    # Kelly Criterion example
    sizer = PositionSizer(PositionSizingParams(method=SizingMethod.KELLY))
    kelly = sizer.calculate_kelly_fraction(win_rate=0.55, avg_win=0.08, avg_loss=0.05)
    print(f"\nKelly Fraction (55% win rate, 8% avg win, 5% avg loss): {kelly:.2%}")

    # Volatility-adjusted example
    sizer = PositionSizer(PositionSizingParams(method=SizingMethod.VOLATILITY))
    vol_size = sizer.calculate_volatility_adjusted_size(current_vol=0.30, vix=25)
    print(f"Volatility-adjusted size (30% vol, VIX=25): {vol_size:.2%}")

    # Scale-in example
    sizer = PositionSizer(PositionSizingParams(method=SizingMethod.SCALE, scale_levels=4))
    scale_size, level = sizer.calculate_scale_in_levels(entry_price=100, current_price=95)
    print(f"Scale-in size (entry=100, current=95): {scale_size:.2%} (level {level})")
