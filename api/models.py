"""
Pydantic data models for the TQQQ Strategy API.

Defines strict type models for all API request/response schemas.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr


# =====================================================
# Signal Models
# =====================================================

class SignalConditions(BaseModel):
    """Breakdown of each trading condition."""
    qqq_above_buy_level: bool = Field(description="QQQ > MA200 x 1.04")
    qqq_daily_loss_met: bool = Field(description="QQQ daily loss >= 1%")
    qqq_below_sell_level: bool = Field(description="QQQ < MA200 x 0.97")


class SignalResponse(BaseModel):
    """Response model for GET /signal/current."""
    date: str = Field(description="Trading date (YYYY-MM-DD)")
    signal: str = Field(description="Current signal: BUY, SELL, or HOLD")
    signal_strength: str = Field(description="Signal strength: STRONG, MODERATE, or WEAK")
    current_position: str = Field(description="Current position: LONG TQQQ or CASH")
    qqq_close: float = Field(description="QQQ closing price")
    qqq_daily_return_pct: float = Field(description="QQQ daily return in percent")
    tqqq_close: float = Field(description="TQQQ closing price")
    ma200: float = Field(description="QQQ 200-day moving average")
    buy_level: float = Field(description="Buy threshold (MA200 x 1.04)")
    sell_level: float = Field(description="Sell threshold (MA200 x 0.97)")
    conditions: SignalConditions
    trigger_explanation: str = Field(description="Human-readable explanation of the signal")
    last_action: Optional[str] = Field(default=None, description="Last trade action")
    last_action_date: Optional[str] = Field(default=None, description="Date of last trade")
    last_action_price: Optional[float] = Field(default=None, description="Price at last trade")
    data_updated_at: str = Field(description="Timestamp when data was last fetched")


# =====================================================
# Backtest Models
# =====================================================

class BacktestResult(BaseModel):
    """Response model for GET /backtest/run."""
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return_pct: float = Field(description="Total return in percent")
    annualized_return_pct: float = Field(description="Annualized return in percent")
    sharpe_ratio: float
    max_drawdown_pct: float = Field(description="Maximum drawdown in percent")
    volatility_pct: float = Field(description="Annualized volatility in percent")
    total_trades: int
    win_rate_pct: float = Field(description="Winning trade percentage")
    time_in_market_pct: float = Field(description="Percentage of time holding position")
    buy_hold_return_pct: float = Field(description="Buy & hold TQQQ return for comparison")
    buy_hold_max_drawdown_pct: float


class TradeRecord(BaseModel):
    """A single round-trip trade (open -> close)."""
    trade_id: int
    open_date: str
    close_date: Optional[str] = Field(default=None, description="None if trade is still open")
    open_qqq_price: float
    close_qqq_price: Optional[float] = None
    open_tqqq_price: float
    close_tqqq_price: Optional[float] = None
    return_pct: Optional[float] = Field(default=None, description="Trade return in percent")
    holding_days: Optional[int] = None
    is_open: bool = Field(default=False, description="True if trade is still active")


class TradeListResponse(BaseModel):
    """Response model for GET /backtest/trades."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    trades: List[TradeRecord]


# =====================================================
# Alert Subscription Models
# =====================================================

class AlertSubscription(BaseModel):
    """Request model for POST /alert/subscribe."""
    email: Optional[str] = Field(default=None, description="Email address for alerts")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL for alerts")
    signal_types: List[str] = Field(
        default=["BUY", "SELL"],
        description="Which signals to subscribe to"
    )


class AlertSubscriptionResponse(BaseModel):
    """Response model for POST /alert/subscribe."""
    success: bool
    message: str
    subscription_id: str


# =====================================================
# Feature History Models
# =====================================================

class FeatureRow(BaseModel):
    """A single day's feature snapshot."""
    date: str
    qqq_close: float
    ma200: float
    buy_level: float
    sell_level: float
    daily_return: float
    tqqq_close: float
    signal: str
    signal_strength: str
    current_position: str
    recorded_at: Optional[str] = None


class FeatureHistoryResponse(BaseModel):
    """Response model for GET /features/history."""
    total_rows: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    features: List[FeatureRow]


# =====================================================
# Signal Scorecard Models
# =====================================================

class SignalScoreItem(BaseModel):
    """Score for a single signal event."""
    date: str
    signal: str
    signal_strength: str
    tqqq_price: float
    forward_returns_pct: dict = Field(description="e.g. {'5d': 3.2, '10d': 5.1, '20d': -1.8}")


class SignalTypeSummary(BaseModel):
    """Aggregated stats for one signal type (BUY or SELL)."""
    signal_type: str
    total_signals: int
    avg_forward_return_pct: dict = Field(description="e.g. {'5d': 2.1, '10d': 3.5}")
    win_rate_pct: dict = Field(description="e.g. {'5d': 72.0, '10d': 68.0}")
    best_signal: Optional[dict] = None
    worst_signal: Optional[dict] = None


class ScorecardResponse(BaseModel):
    """Response model for GET /features/scorecard."""
    data_range: dict
    total_signals: int
    buy_summary: Optional[SignalTypeSummary] = None
    sell_summary: Optional[SignalTypeSummary] = None
    all_scores: List[SignalScoreItem]


# =====================================================
# Experiment Log Models
# =====================================================

class ExperimentRecord(BaseModel):
    """A single backtest experiment record."""
    run_timestamp: str
    start_date: str
    end_date: Optional[str] = None
    initial_capital: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    volatility_pct: Optional[float] = None
    win_rate_pct: float
    total_trades: int
    time_in_market_pct: float
    final_value: Optional[float] = None
    buy_hold_return_pct: Optional[float] = None
    buy_hold_max_drawdown_pct: Optional[float] = None


class ExperimentListResponse(BaseModel):
    """Response model for GET /experiments."""
    total_experiments: int
    best_sharpe: Optional[ExperimentRecord] = None
    experiments: List[ExperimentRecord]


# =====================================================
# Health Check Model
# =====================================================

class HealthResponse(BaseModel):
    """Response model for GET /health."""
    status: str = Field(description="Service status: healthy or degraded")
    version: str
    uptime_seconds: float
    data_last_updated: Optional[str] = Field(
        default=None,
        description="Last time market data was fetched"
    )
    components: dict = Field(
        default_factory=dict,
        description="Status of sub-components (yfinance, cache, etc.)"
    )
