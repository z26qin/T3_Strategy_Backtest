"""
Feature Store — Parquet-based storage for daily feature history.

Persists computed features (QQQ price, MA200, signals, etc.) to disk
so they can be queried later for auditing, signal evaluation, and ML training.

Storage layout:
    data/features/daily_features.parquet   — single file, appended daily

Design principles:
    - Idempotent: writing the same date twice won't create duplicates
    - Column-pruning friendly: read only the columns you need
    - Zero external dependencies beyond pyarrow (ships with pandas)
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


# Default storage path (relative to project root)
_DEFAULT_DIR = Path(__file__).parent.parent / "data" / "features"
_DEFAULT_PATH = _DEFAULT_DIR / "daily_features.parquet"


class FeatureStore:
    """
    Read/write daily feature snapshots to Parquet.

    Usage:
        store = FeatureStore()
        store.write(feature_df)              # Append new rows
        df = store.read()                    # Read all history
        df = store.read(columns=["ma200"])   # Column pruning
        df = store.read(start="2024-01-01")  # Time range filter
    """

    # Schema: every row must have these columns (enforced on write)
    REQUIRED_COLUMNS = [
        "date",           # str YYYY-MM-DD — the trading date
        "qqq_close",      # float
        "ma200",          # float
        "buy_level",      # float
        "sell_level",     # float
        "daily_return",   # float — QQQ daily return (decimal, e.g. -0.012)
        "tqqq_close",     # float
        "signal",         # str — BUY / SELL / HOLD
        "signal_strength", # str — STRONG / MODERATE / WEAK
        "current_position", # str — LONG TQQQ / CASH
    ]

    def __init__(self, path: Optional[str | Path] = None):
        """
        Args:
            path: Path to the Parquet file. Defaults to data/features/daily_features.parquet.
        """
        self.path = Path(path) if path else _DEFAULT_PATH

    def write(self, df: pd.DataFrame) -> int:
        """
        Append feature rows to the store. Idempotent — duplicate dates are skipped.

        Args:
            df: DataFrame with at least the REQUIRED_COLUMNS.

        Returns:
            Number of new rows actually written.
        """
        self._validate_schema(df)

        new_df = df.copy()
        new_df["date"] = new_df["date"].astype(str)
        new_df["recorded_at"] = datetime.now().isoformat()

        if self.path.exists():
            existing = pd.read_parquet(self.path)
            existing_dates = set(existing["date"].astype(str))
            new_df = new_df[~new_df["date"].isin(existing_dates)]

            if new_df.empty:
                return 0

            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            combined = new_df

        combined = combined.sort_values("date").reset_index(drop=True)
        combined.to_parquet(self.path, index=False, engine="pyarrow")

        return len(new_df)

    def read(
        self,
        columns: Optional[list[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Read feature history with optional column pruning and time range filter.

        Args:
            columns: List of columns to read. None = all columns.
                     This is where Parquet shines — only these columns are
                     loaded from disk, skipping the rest entirely.
            start: Start date inclusive (YYYY-MM-DD). None = from beginning.
            end: End date inclusive (YYYY-MM-DD). None = to latest.

        Returns:
            DataFrame with requested features. Empty DataFrame if no data.
        """
        if not self.path.exists():
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        # Column pruning: Parquet only reads requested columns from disk
        read_cols = columns
        if read_cols and "date" not in read_cols:
            read_cols = ["date"] + read_cols  # Need date for filtering

        df = pd.read_parquet(self.path, columns=read_cols, engine="pyarrow")

        # Time range filter
        if start:
            df = df[df["date"] >= start]
        if end:
            df = df[df["date"] <= end]

        # If caller didn't ask for date but we added it for filtering, drop it
        if columns and "date" not in columns:
            df = df.drop(columns=["date"])

        return df.reset_index(drop=True)

    def latest_date(self) -> Optional[str]:
        """Return the most recent date in the store, or None if empty."""
        if not self.path.exists():
            return None
        df = pd.read_parquet(self.path, columns=["date"], engine="pyarrow")
        if df.empty:
            return None
        return str(df["date"].max())

    def count(self) -> int:
        """Return total number of rows in the store."""
        if not self.path.exists():
            return 0
        df = pd.read_parquet(self.path, columns=["date"], engine="pyarrow")
        return len(df)

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Ensure DataFrame has all required columns."""
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {missing}. "
                f"Required: {self.REQUIRED_COLUMNS}"
            )
