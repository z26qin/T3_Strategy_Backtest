"""
Experiment Log — Parquet-based storage for backtest experiment records.

Every time a backtest runs, a row is appended with the parameters used
and the results obtained. This enables:
  - "What parameters did I use last time?"
  - "Which parameter combo gave the best Sharpe?"
  - "Is strategy performance degrading over time?"

Storage layout:
    data/experiments/backtest_log.parquet
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path(__file__).parent.parent / "data" / "experiments"
_DEFAULT_PATH = _DEFAULT_DIR / "backtest_log.parquet"


class ExperimentLog:
    """
    Append-only log of backtest experiments.

    Each row records:
      - When the backtest was run (run_timestamp)
      - Input parameters (start_date, end_date, initial_capital)
      - Output metrics (return, sharpe, drawdown, win_rate, etc.)

    Usage:
        log = ExperimentLog()
        log.record(params_dict, results_dict)
        df = log.read()
        best = log.best_by("sharpe_ratio")
    """

    REQUIRED_COLUMNS = [
        "run_timestamp",
        "start_date",
        "end_date",
        "initial_capital",
        "total_return_pct",
        "annualized_return_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "win_rate_pct",
        "total_trades",
        "time_in_market_pct",
    ]

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else _DEFAULT_PATH

    def record(self, params: dict, results: dict) -> None:
        """
        Append one experiment record.

        Args:
            params: Backtest input parameters.
                    Expected keys: start_date, end_date, initial_capital
            results: Backtest output metrics.
                    Expected keys: total_return_pct, annualized_return_pct,
                    sharpe_ratio, max_drawdown_pct, win_rate_pct,
                    total_trades, time_in_market_pct, final_value,
                    buy_hold_return_pct, buy_hold_max_drawdown_pct
        """
        row = pd.DataFrame([{
            "run_timestamp": datetime.now().isoformat(),
            "start_date": params.get("start_date"),
            "end_date": params.get("end_date"),
            "initial_capital": params.get("initial_capital"),
            "total_return_pct": results.get("total_return_pct"),
            "annualized_return_pct": results.get("annualized_return_pct"),
            "sharpe_ratio": results.get("sharpe_ratio"),
            "max_drawdown_pct": results.get("max_drawdown_pct"),
            "volatility_pct": results.get("volatility_pct"),
            "win_rate_pct": results.get("win_rate_pct"),
            "total_trades": results.get("total_trades"),
            "time_in_market_pct": results.get("time_in_market_pct"),
            "final_value": results.get("final_value"),
            "buy_hold_return_pct": results.get("buy_hold_return_pct"),
            "buy_hold_max_drawdown_pct": results.get("buy_hold_max_drawdown_pct"),
        }])

        if self.path.exists():
            existing = pd.read_parquet(self.path)
            combined = pd.concat([existing, row], ignore_index=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            combined = row

        combined.to_parquet(self.path, index=False, engine="pyarrow")
        logger.info(
            f"Experiment logged: {params.get('start_date')} → {params.get('end_date')} | "
            f"Return: {results.get('total_return_pct')}% | Sharpe: {results.get('sharpe_ratio')}"
        )

    def read(self) -> pd.DataFrame:
        """Read all experiment records."""
        if not self.path.exists():
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        return pd.read_parquet(self.path, engine="pyarrow")

    def count(self) -> int:
        """Total number of experiments logged."""
        if not self.path.exists():
            return 0
        return len(pd.read_parquet(self.path, columns=["run_timestamp"], engine="pyarrow"))

    def best_by(self, metric: str = "sharpe_ratio") -> Optional[pd.Series]:
        """
        Return the experiment with the highest value for the given metric.

        Args:
            metric: Column name to rank by (e.g. "sharpe_ratio", "total_return_pct").

        Returns:
            pd.Series of the best experiment, or None if no data.
        """
        df = self.read()
        if df.empty or metric not in df.columns:
            return None
        return df.loc[df[metric].idxmax()]

    def recent(self, n: int = 10) -> pd.DataFrame:
        """Return the N most recent experiments."""
        df = self.read()
        if df.empty:
            return df
        return df.tail(n).reset_index(drop=True)
