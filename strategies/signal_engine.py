"""
Signal Engine — Cached real-time signal service for the API layer.

Wraps the existing SignalChecker with:
- TTL-based caching (default 15 minutes) to avoid excessive yfinance calls
- Conversion to Pydantic SignalResponse model
- Signal strength and human-readable trigger explanation
"""

import logging
import time
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

from strategies.signal_checker import SignalChecker, SignalStatus
from strategies.feature_store import FeatureStore
from api.models import SignalResponse, SignalConditions

logger = logging.getLogger(__name__)


class SignalCache:
    """Simple TTL cache for signal data."""

    def __init__(self, ttl_seconds: int = 900):
        """
        Args:
            ttl_seconds: Cache time-to-live in seconds. Default 900 = 15 minutes.
        """
        self.ttl_seconds = ttl_seconds
        self._cached_status: Optional[SignalStatus] = None
        self._cached_at: float = 0.0  # Unix timestamp

    def is_valid(self) -> bool:
        """Check if cache is still within TTL."""
        if self._cached_status is None:
            return False
        return (time.time() - self._cached_at) < self.ttl_seconds

    def get(self) -> Optional[SignalStatus]:
        """Return cached status if valid, else None."""
        if self.is_valid():
            return self._cached_status
        return None

    def set(self, status: SignalStatus) -> None:
        """Store a new status in cache."""
        self._cached_status = status
        self._cached_at = time.time()

    @property
    def last_updated_iso(self) -> Optional[str]:
        """Return last cache update time as ISO string."""
        if self._cached_at == 0.0:
            return None
        return datetime.fromtimestamp(self._cached_at).isoformat()


class SignalEngine:
    """
    High-level signal service used by the API.

    Usage:
        engine = SignalEngine()
        response = engine.get_current_signal()  # Returns SignalResponse
    """

    def __init__(self, cache_ttl: int = 900, feature_store: Optional[FeatureStore] = None):
        """
        Args:
            cache_ttl: Cache TTL in seconds. Default 900 = 15 minutes.
            feature_store: Optional FeatureStore instance. Defaults to standard path.
        """
        self._cache = SignalCache(ttl_seconds=cache_ttl)
        self._checker = SignalChecker()
        self._feature_store = feature_store or FeatureStore()

    def get_current_signal(self, force_refresh: bool = False) -> SignalResponse:
        """
        Get the current trading signal, using cache when possible.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data.

        Returns:
            SignalResponse pydantic model ready for API serialization.
        """
        # 1. Try cache first
        if not force_refresh:
            cached = self._cache.get()
            if cached is not None:
                return self._status_to_response(cached)

        # 2. Cache miss — fetch fresh data via SignalChecker
        self._checker = SignalChecker()  # Reset to clear stale internal state
        status = self._checker.check_signal()

        # 3. Store in cache
        self._cache.set(status)

        # 4. Persist to feature store (non-blocking: failure won't break the API)
        self._record_features(status)

        return self._status_to_response(status)

    @property
    def data_last_updated(self) -> Optional[str]:
        """ISO timestamp of the last data fetch."""
        return self._cache.last_updated_iso

    def _status_to_response(self, status: SignalStatus) -> SignalResponse:
        """
        Convert the internal SignalStatus dataclass to a Pydantic SignalResponse.

        This is the bridge between the existing strategy code and the API layer.
        """
        strength = self._calc_signal_strength(status)
        explanation = self._build_explanation(status)

        return SignalResponse(
            date=status.date,
            signal=status.signal,
            signal_strength=strength,
            current_position="LONG TQQQ" if status.current_position == 1 else "CASH",
            qqq_close=round(status.qqq_close, 2),
            qqq_daily_return_pct=round(status.qqq_daily_return * 100, 2),
            tqqq_close=round(status.tqqq_close, 2),
            ma200=round(status.ma200, 2),
            buy_level=round(status.buy_level, 2),
            sell_level=round(status.sell_level, 2),
            conditions=SignalConditions(
                qqq_above_buy_level=status.qqq_above_buy_level,
                qqq_daily_loss_met=status.qqq_daily_loss_met,
                qqq_below_sell_level=status.qqq_below_sell_level,
            ),
            trigger_explanation=explanation,
            last_action=status.last_action,
            last_action_date=status.last_action_date,
            last_action_price=round(status.last_action_price, 2) if status.last_action_price else None,
            data_updated_at=self._cache.last_updated_iso or datetime.now().isoformat(),
        )

    @staticmethod
    def _calc_signal_strength(status: SignalStatus) -> str:
        """
        Calculate signal strength based on how far QQQ is from key levels.

        - STRONG: both buy conditions met (BUY), or price well below sell level (SELL)
        - MODERATE: one condition met, or price near a trigger level
        - WEAK: no conditions close to triggering
        """
        if status.signal == "BUY":
            # Both conditions are already met for a BUY signal
            return "STRONG"

        if status.signal == "SELL":
            # How far below the sell level?
            pct_below = (status.sell_level - status.qqq_close) / status.qqq_close
            return "STRONG" if pct_below > 0.02 else "MODERATE"

        # HOLD — check proximity to trigger levels
        conditions_met = sum([
            status.qqq_above_buy_level,
            status.qqq_daily_loss_met,
            status.qqq_below_sell_level,
        ])
        if conditions_met >= 1:
            return "MODERATE"
        return "WEAK"

    @staticmethod
    def _build_explanation(status: SignalStatus) -> str:
        """Build a human-readable explanation of why this signal was generated."""
        if status.signal == "BUY":
            return (
                f"BUY signal triggered: QQQ (${status.qqq_close:.2f}) is above the buy level "
                f"(${status.buy_level:.2f} = MA200 x 1.04) and today's drop "
                f"({status.qqq_daily_return * 100:+.2f}%) meets the >= 1% loss threshold."
            )

        if status.signal == "SELL":
            return (
                f"SELL signal triggered: QQQ (${status.qqq_close:.2f}) has fallen below "
                f"the sell level (${status.sell_level:.2f} = MA200 x 0.97)."
            )

        # HOLD — describe what's missing
        parts = []
        if status.qqq_above_buy_level:
            parts.append("QQQ is above buy level, but no daily dip >= 1% yet")
        elif status.qqq_daily_loss_met:
            parts.append("Daily loss threshold met, but QQQ is below the buy level")
        else:
            gap_to_buy = ((status.buy_level - status.qqq_close) / status.qqq_close) * 100
            gap_to_sell = ((status.qqq_close - status.sell_level) / status.qqq_close) * 100
            parts.append(
                f"QQQ is {gap_to_buy:+.1f}% from buy level and "
                f"{gap_to_sell:+.1f}% above sell level"
            )

        return f"HOLD — no action needed. {'; '.join(parts)}."

    def _record_features(self, status: SignalStatus) -> None:
        """
        Persist today's features to the Parquet feature store.

        Called after every fresh data fetch (cache miss). Safe to call
        multiple times — FeatureStore.write() is idempotent by date.

        Wrapped in try/except so a feature store failure never breaks the API.
        """
        try:
            strength = self._calc_signal_strength(status)
            position = "LONG TQQQ" if status.current_position == 1 else "CASH"

            row = pd.DataFrame([{
                "date": status.date,
                "qqq_close": round(status.qqq_close, 2),
                "ma200": round(status.ma200, 2),
                "buy_level": round(status.buy_level, 2),
                "sell_level": round(status.sell_level, 2),
                "daily_return": round(status.qqq_daily_return, 6),
                "tqqq_close": round(status.tqqq_close, 2),
                "signal": status.signal,
                "signal_strength": strength,
                "current_position": position,
            }])

            written = self._feature_store.write(row)
            if written > 0:
                logger.info(f"Feature store: recorded {status.date} ({status.signal})")
        except Exception as e:
            logger.warning(f"Feature store write failed (non-fatal): {e}")
