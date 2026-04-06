"""
Backtest Runner — Structured backtest interface for the API layer.

Wraps the existing TQQQMA200Strategy with:
- Simple function interface: run_backtest(start, end, capital) -> BacktestResult
- Round-trip trade pairing (BUY + SELL = one complete trade)
- Win rate calculation
- All results returned as Pydantic models
"""

from typing import List, Tuple

from strategies.tqqq_ma200_strategy import TQQQMA200Strategy, StrategyParams
from api.models import BacktestResult, TradeRecord, TradeListResponse


def run_backtest(
    start_date: str = "2015-01-01",
    end_date: str | None = None,
    initial_capital: float = 100000,
) -> BacktestResult:
    """
    Execute a full backtest and return structured results.

    Args:
        start_date: Backtest start date (YYYY-MM-DD).
        end_date: Backtest end date (YYYY-MM-DD). Defaults to today.
        initial_capital: Starting capital in USD.

    Returns:
        BacktestResult with all key performance metrics.
    """
    # 1. Configure and run the existing strategy engine
    params = StrategyParams(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )
    strategy = TQQQMA200Strategy(params)
    strategy.run_full_analysis()

    metrics = strategy.metrics
    sm = metrics["strategy"]
    bh = metrics["buy_hold"]

    # 2. Calculate win rate from round-trip trades
    trades = _pair_trades(strategy)
    closed_trades = [t for t in trades if not t.is_open]
    winning = [t for t in closed_trades if t.return_pct is not None and t.return_pct > 0]
    win_rate = (len(winning) / len(closed_trades) * 100) if closed_trades else 0.0

    # 3. Build response
    return BacktestResult(
        start_date=params.start_date,
        end_date=params.end_date,
        initial_capital=initial_capital,
        final_value=round(sm["final_value"], 2),
        total_return_pct=round(sm["total_return"], 2),
        annualized_return_pct=round(sm["ann_return"], 2),
        sharpe_ratio=round(sm["sharpe"], 2),
        max_drawdown_pct=round(sm["max_drawdown"], 2),
        volatility_pct=round(sm["volatility"], 2),
        total_trades=metrics["num_trades"],
        win_rate_pct=round(win_rate, 2),
        time_in_market_pct=round(metrics["time_in_market"], 2),
        buy_hold_return_pct=round(bh["total_return"], 2),
        buy_hold_max_drawdown_pct=round(bh["max_drawdown"], 2),
    )


def get_trades(
    start_date: str = "2015-01-01",
    end_date: str | None = None,
    initial_capital: float = 100000,
) -> TradeListResponse:
    """
    Execute backtest and return all round-trip trade records.

    Args:
        start_date: Backtest start date (YYYY-MM-DD).
        end_date: Backtest end date (YYYY-MM-DD). Defaults to today.
        initial_capital: Starting capital in USD.

    Returns:
        TradeListResponse with paired trade records and summary stats.
    """
    params = StrategyParams(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )
    strategy = TQQQMA200Strategy(params)
    strategy.run_full_analysis()

    trades = _pair_trades(strategy)
    closed = [t for t in trades if not t.is_open]
    winning = [t for t in closed if t.return_pct is not None and t.return_pct > 0]
    losing = [t for t in closed if t.return_pct is not None and t.return_pct <= 0]

    return TradeListResponse(
        total_trades=len(trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        trades=trades,
    )


def _pair_trades(strategy: TQQQMA200Strategy) -> List[TradeRecord]:
    """
    Pair BUY and SELL actions into round-trip TradeRecord objects.

    Logic:
        - Walk through signals where Position_Change != 0
        - Position_Change == 1  → open a new trade (BUY)
        - Position_Change == -1 → close the current trade (SELL)
        - If the last trade has no matching SELL, mark it as is_open=True

    Args:
        strategy: A fully-run TQQQMA200Strategy instance.

    Returns:
        List of TradeRecord, each representing one round-trip trade.
    """
    signals = strategy.signals
    trade_changes = signals[signals["Position_Change"] != 0].copy()

    trades: List[TradeRecord] = []
    trade_id = 1
    pending_open = None  # Holds the BUY row while waiting for matching SELL

    for date, row in trade_changes.iterrows():
        change = row["Position_Change"]
        date_str = date.strftime("%Y-%m-%d")

        if change == 1:
            # Opening a new position
            pending_open = {
                "date": date_str,
                "qqq_price": float(row["QQQ_Close"]),
                "tqqq_price": float(row["TQQQ_Close"]),
            }

        elif change == -1 and pending_open is not None:
            # Closing the position — pair with the pending open
            close_tqqq = float(row["TQQQ_Close"])
            open_tqqq = pending_open["tqqq_price"]
            return_pct = ((close_tqqq - open_tqqq) / open_tqqq) * 100

            open_date = pending_open["date"]
            holding_days = (date - _parse_date(open_date)).days

            trades.append(TradeRecord(
                trade_id=trade_id,
                open_date=open_date,
                close_date=date_str,
                open_qqq_price=round(pending_open["qqq_price"], 2),
                close_qqq_price=round(float(row["QQQ_Close"]), 2),
                open_tqqq_price=round(open_tqqq, 2),
                close_tqqq_price=round(close_tqqq, 2),
                return_pct=round(return_pct, 2),
                holding_days=holding_days,
                is_open=False,
            ))
            trade_id += 1
            pending_open = None

    # If there's an unmatched BUY (still holding), add it as open trade
    if pending_open is not None:
        trades.append(TradeRecord(
            trade_id=trade_id,
            open_date=pending_open["date"],
            close_date=None,
            open_qqq_price=round(pending_open["qqq_price"], 2),
            close_qqq_price=None,
            open_tqqq_price=round(pending_open["tqqq_price"], 2),
            close_tqqq_price=None,
            return_pct=None,
            holding_days=None,
            is_open=True,
        ))

    return trades


def _parse_date(date_str: str):
    """Parse YYYY-MM-DD string to pandas Timestamp for date arithmetic."""
    import pandas as pd
    return pd.Timestamp(date_str)
