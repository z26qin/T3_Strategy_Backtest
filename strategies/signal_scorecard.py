"""
Signal Scorecard — Evaluate the accuracy of historical trading signals.

For each BUY/SELL signal in the feature store, looks at what actually
happened N days later (forward return) to answer:
  - "How accurate are our BUY signals?"
  - "How much downside did SELL signals help us avoid?"
  - "Is signal quality stable over time?"

Requires:
  - Feature store with historical signal records
  - yfinance for forward price lookups
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf

from strategies.feature_store import FeatureStore

logger = logging.getLogger(__name__)

# Default lookahead windows (trading days)
DEFAULT_WINDOWS = [5, 10, 20]


@dataclass
class SignalScore:
    """Score for a single signal event."""
    date: str
    signal: str                          # BUY or SELL
    signal_strength: str
    tqqq_price_at_signal: float
    forward_returns: dict[int, float]    # {5: 0.032, 10: 0.051, 20: -0.018}
    forward_prices: dict[int, float]     # {5: 56.76, 10: 57.81, 20: 54.01}


@dataclass
class ScorecardSummary:
    """Aggregated scorecard for a signal type."""
    signal_type: str                     # BUY or SELL
    total_signals: int
    avg_forward_return: dict[int, float] # {5: 0.021, 10: 0.035, 20: 0.018}
    win_rate: dict[int, float]           # {5: 0.72, 10: 0.68, 20: 0.55}
    best_signal: Optional[SignalScore]
    worst_signal: Optional[SignalScore]


class SignalScorecard:
    """
    Evaluate historical signal accuracy using forward returns.

    Usage:
        scorecard = SignalScorecard()
        scores = scorecard.score_all_signals()         # Score every signal
        summary = scorecard.summarize(signal_type="BUY")  # Aggregate stats
        report = scorecard.full_report()               # Complete report dict
    """

    def __init__(
        self,
        feature_store: Optional[FeatureStore] = None,
        windows: Optional[list[int]] = None,
    ):
        """
        Args:
            feature_store: FeatureStore instance. Defaults to standard path.
            windows: Lookahead windows in trading days. Default [5, 10, 20].
        """
        self._store = feature_store or FeatureStore()
        self._windows = windows or DEFAULT_WINDOWS
        self._tqqq_data: Optional[pd.DataFrame] = None

    def score_all_signals(self) -> list[SignalScore]:
        """
        Score every BUY and SELL signal in the feature store.

        Returns:
            List of SignalScore, one per signal event. Signals too recent
            to have full forward returns are still included (with partial data).
        """
        features = self._store.read()
        if features.empty:
            logger.info("Feature store is empty — no signals to score")
            return []

        # Filter to actionable signals only (BUY and SELL, not HOLD)
        signals = features[features["signal"].isin(["BUY", "SELL"])].copy()
        if signals.empty:
            logger.info("No BUY/SELL signals found in feature store")
            return []

        # Fetch TQQQ price history for forward return calculation
        self._fetch_tqqq_data(features["date"].min())

        scores = []
        for _, row in signals.iterrows():
            score = self._score_single_signal(row)
            if score is not None:
                scores.append(score)

        return scores

    def summarize(self, signal_type: str = "BUY") -> Optional[ScorecardSummary]:
        """
        Aggregate statistics for a specific signal type.

        Args:
            signal_type: "BUY" or "SELL"

        Returns:
            ScorecardSummary with average returns, win rates, best/worst signals.
            None if no signals of this type exist.
        """
        all_scores = self.score_all_signals()
        typed_scores = [s for s in all_scores if s.signal == signal_type]

        if not typed_scores:
            return None

        # Calculate averages and win rates per window
        avg_return = {}
        win_rate = {}

        for w in self._windows:
            returns = [
                s.forward_returns[w]
                for s in typed_scores
                if w in s.forward_returns
            ]
            if returns:
                avg_return[w] = round(sum(returns) / len(returns), 4)
                # For BUY: "win" = positive return
                # For SELL: "win" = negative return (avoided the drop)
                if signal_type == "BUY":
                    wins = sum(1 for r in returns if r > 0)
                else:
                    wins = sum(1 for r in returns if r < 0)
                win_rate[w] = round(wins / len(returns), 4)
            else:
                avg_return[w] = 0.0
                win_rate[w] = 0.0

        # Find best and worst by 10-day return (middle window)
        mid_window = self._windows[1] if len(self._windows) > 1 else self._windows[0]
        scored_with_mid = [
            s for s in typed_scores if mid_window in s.forward_returns
        ]

        best = max(scored_with_mid, key=lambda s: s.forward_returns[mid_window]) if scored_with_mid else None
        worst = min(scored_with_mid, key=lambda s: s.forward_returns[mid_window]) if scored_with_mid else None

        return ScorecardSummary(
            signal_type=signal_type,
            total_signals=len(typed_scores),
            avg_forward_return=avg_return,
            win_rate=win_rate,
            best_signal=best,
            worst_signal=worst,
        )

    def full_report(self) -> dict:
        """
        Generate a complete scorecard report for all signal types.

        Returns:
            Dict with keys 'buy_summary', 'sell_summary', 'all_scores',
            'total_signals', 'data_range'.
        """
        features = self._store.read(columns=["date"])

        all_scores = self.score_all_signals()
        buy_summary = self.summarize("BUY")
        sell_summary = self.summarize("SELL")

        return {
            "data_range": {
                "start": features["date"].min() if not features.empty else None,
                "end": features["date"].max() if not features.empty else None,
                "total_days": len(features),
            },
            "total_signals": len(all_scores),
            "buy_summary": self._summary_to_dict(buy_summary) if buy_summary else None,
            "sell_summary": self._summary_to_dict(sell_summary) if sell_summary else None,
            "all_scores": [self._score_to_dict(s) for s in all_scores],
        }

    def _fetch_tqqq_data(self, start_date: str) -> None:
        """Fetch TQQQ daily close prices from the earliest feature date."""
        if self._tqqq_data is not None:
            return

        try:
            data = yf.download("TQQQ", start=start_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            # Normalize index to date strings for easy lookup
            data.index = data.index.strftime("%Y-%m-%d")
            self._tqqq_data = data
        except Exception as e:
            logger.warning(f"Failed to fetch TQQQ data: {e}")
            self._tqqq_data = pd.DataFrame()

    def _score_single_signal(self, row: pd.Series) -> Optional[SignalScore]:
        """
        Calculate forward returns for a single signal event.

        Looks up TQQQ price on signal date, then checks price N days later
        for each window in self._windows.
        """
        if self._tqqq_data is None or self._tqqq_data.empty:
            return None

        signal_date = str(row["date"])

        if signal_date not in self._tqqq_data.index:
            return None

        signal_idx = self._tqqq_data.index.get_loc(signal_date)
        signal_price = float(self._tqqq_data["Close"].iloc[signal_idx])

        forward_returns = {}
        forward_prices = {}

        for w in self._windows:
            future_idx = signal_idx + w
            if future_idx < len(self._tqqq_data):
                future_price = float(self._tqqq_data["Close"].iloc[future_idx])
                forward_prices[w] = round(future_price, 2)
                forward_returns[w] = round(
                    (future_price - signal_price) / signal_price, 4
                )

        # Skip if we couldn't compute any forward returns
        if not forward_returns:
            return None

        return SignalScore(
            date=signal_date,
            signal=str(row["signal"]),
            signal_strength=str(row.get("signal_strength", "UNKNOWN")),
            tqqq_price_at_signal=round(signal_price, 2),
            forward_returns=forward_returns,
            forward_prices=forward_prices,
        )

    @staticmethod
    def _summary_to_dict(summary: ScorecardSummary) -> dict:
        """Convert ScorecardSummary to a JSON-serializable dict."""
        return {
            "signal_type": summary.signal_type,
            "total_signals": summary.total_signals,
            "avg_forward_return_pct": {
                f"{k}d": round(v * 100, 2) for k, v in summary.avg_forward_return.items()
            },
            "win_rate_pct": {
                f"{k}d": round(v * 100, 1) for k, v in summary.win_rate.items()
            },
            "best_signal": {
                "date": summary.best_signal.date,
                "return_pct": {
                    f"{k}d": round(v * 100, 2)
                    for k, v in summary.best_signal.forward_returns.items()
                },
            } if summary.best_signal else None,
            "worst_signal": {
                "date": summary.worst_signal.date,
                "return_pct": {
                    f"{k}d": round(v * 100, 2)
                    for k, v in summary.worst_signal.forward_returns.items()
                },
            } if summary.worst_signal else None,
        }

    @staticmethod
    def _score_to_dict(score: SignalScore) -> dict:
        """Convert SignalScore to a JSON-serializable dict."""
        return {
            "date": score.date,
            "signal": score.signal,
            "signal_strength": score.signal_strength,
            "tqqq_price": score.tqqq_price_at_signal,
            "forward_returns_pct": {
                f"{k}d": round(v * 100, 2) for k, v in score.forward_returns.items()
            },
        }
