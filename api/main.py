"""
FastAPI application for the TQQQ MA200 Strategy API.

Endpoints:
    GET  /signal/current    — Current trading signal with real-time data
    GET  /backtest/run      — Run backtest with custom parameters
    GET  /backtest/trades   — List all round-trip trade records
    POST /alert/subscribe   — Subscribe to signal alerts
    GET  /features/history  — Query feature history from Parquet store
    GET  /features/scorecard — Signal accuracy report
    GET  /experiments       — Backtest experiment log
    GET  /health            — Health check

Run with:
    uvicorn api.main:app --reload --port 8000
"""

import os
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Query, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    SignalResponse,
    BacktestResult,
    TradeListResponse,
    AlertSubscription,
    AlertSubscriptionResponse,
    HealthResponse,
    FeatureRow,
    FeatureHistoryResponse,
    ScorecardResponse,
    SignalScoreItem,
    SignalTypeSummary,
    ExperimentRecord,
    ExperimentListResponse,
)
from strategies.signal_engine import SignalEngine
from strategies.backtest_runner import run_backtest, get_trades
from strategies.feature_store import FeatureStore
from strategies.signal_scorecard import SignalScorecard
from strategies.experiment_log import ExperimentLog


# =====================================================
# App Initialization
# =====================================================

app = FastAPI(
    title="TQQQ MA200 Strategy API",
    description=(
        "REST API for the TQQQ MA200 trading strategy. "
        "Provides real-time signals, backtesting, and trade history."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Record startup time for health check uptime calculation
_startup_time = time.time()

# Global signal engine (singleton with 15-min cache)
_signal_engine = SignalEngine(cache_ttl=900)

# In-memory alert subscriptions (simple storage for now)
_subscriptions: list[dict] = []

# MLOps components (feature store + experiment log)
_feature_store = FeatureStore()
_experiment_log = ExperimentLog()


# =====================================================
# CORS Middleware
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# API Key Authentication
# =====================================================

def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    Validate the X-API-Key header against the API_KEY environment variable.

    If API_KEY is not set in the environment, authentication is skipped
    (development mode). This makes local development frictionless while
    still supporting auth in production.
    """
    expected_key = os.getenv("API_KEY")

    # If no API_KEY is configured, skip auth (dev mode)
    if expected_key is None:
        return

    # If API_KEY is configured, enforce it
    if x_api_key is None or x_api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide X-API-Key header.",
        )


# =====================================================
# Endpoints
# =====================================================

@app.get(
    "/signal/current",
    response_model=SignalResponse,
    summary="Get current trading signal",
    tags=["Signal"],
)
def get_current_signal(
    force_refresh: bool = Query(
        default=False,
        description="Bypass cache and fetch fresh data from yfinance",
    ),
    _: None = Depends(verify_api_key),
) -> SignalResponse:
    """
    Returns the current trading signal (BUY / SELL / HOLD) with full context:
    QQQ price, MA200 value, signal strength, trigger conditions, and explanation.

    Data is cached for 15 minutes to avoid excessive yfinance requests.
    Use `force_refresh=true` to bypass cache.
    """
    try:
        return _signal_engine.get_current_signal(force_refresh=force_refresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch signal: {e}")


@app.get(
    "/backtest/run",
    response_model=BacktestResult,
    summary="Run strategy backtest",
    tags=["Backtest"],
)
def run_backtest_endpoint(
    start_date: str = Query(
        default="2015-01-01",
        description="Backtest start date (YYYY-MM-DD)",
    ),
    end_date: Optional[str] = Query(
        default=None,
        description="Backtest end date (YYYY-MM-DD). Defaults to today.",
    ),
    initial_capital: float = Query(
        default=100000,
        ge=1000,
        description="Starting capital in USD (minimum $1,000)",
    ),
    _: None = Depends(verify_api_key),
) -> BacktestResult:
    """
    Execute a full backtest with custom parameters and return summary metrics:
    total return, annualized return, Sharpe ratio, max drawdown, win rate, etc.

    Note: This endpoint fetches data from yfinance each time, so it may take
    a few seconds to respond.
    """
    try:
        return run_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")


@app.get(
    "/backtest/trades",
    response_model=TradeListResponse,
    summary="Get trade history",
    tags=["Backtest"],
)
def get_trades_endpoint(
    start_date: str = Query(
        default="2015-01-01",
        description="Backtest start date (YYYY-MM-DD)",
    ),
    end_date: Optional[str] = Query(
        default=None,
        description="Backtest end date (YYYY-MM-DD). Defaults to today.",
    ),
    initial_capital: float = Query(
        default=100000,
        ge=1000,
        description="Starting capital in USD",
    ),
    _: None = Depends(verify_api_key),
) -> TradeListResponse:
    """
    Execute backtest and return all round-trip trade records.
    Each trade includes: open/close dates, entry/exit prices, P&L, holding days.
    Open trades (still holding) are marked with `is_open: true`.
    """
    try:
        return get_trades(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trades: {e}")


@app.post(
    "/alert/subscribe",
    response_model=AlertSubscriptionResponse,
    summary="Subscribe to signal alerts",
    tags=["Alerts"],
)
def subscribe_alert(
    subscription: AlertSubscription,
    _: None = Depends(verify_api_key),
) -> AlertSubscriptionResponse:
    """
    Register for signal alert notifications via email or webhook.
    At least one of `email` or `webhook_url` must be provided.
    """
    if not subscription.email and not subscription.webhook_url:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'email' or 'webhook_url' must be provided.",
        )

    subscription_id = str(uuid.uuid4())[:8]

    _subscriptions.append({
        "id": subscription_id,
        "email": subscription.email,
        "webhook_url": subscription.webhook_url,
        "signal_types": subscription.signal_types,
        "created_at": time.time(),
    })

    return AlertSubscriptionResponse(
        success=True,
        message=f"Subscribed successfully. Monitoring signals: {subscription.signal_types}",
        subscription_id=subscription_id,
    )


# =====================================================
# MLOps Endpoints
# =====================================================

@app.get(
    "/features/history",
    response_model=FeatureHistoryResponse,
    summary="Query feature history",
    tags=["MLOps"],
)
def get_feature_history(
    start: Optional[str] = Query(
        default=None,
        description="Start date inclusive (YYYY-MM-DD)",
    ),
    end: Optional[str] = Query(
        default=None,
        description="End date inclusive (YYYY-MM-DD)",
    ),
    columns: Optional[str] = Query(
        default=None,
        description="Comma-separated column names to return (e.g. 'ma200,signal'). "
                    "Omit for all columns.",
    ),
    _: None = Depends(verify_api_key),
) -> FeatureHistoryResponse:
    """
    Query the feature history store (Parquet-backed).

    Returns daily feature snapshots recorded each time a signal is computed.
    Supports time range filtering and column pruning for efficient queries.
    """
    try:
        col_list = [c.strip() for c in columns.split(",")] if columns else None
        df = _feature_store.read(columns=col_list, start=start, end=end)

        features = [FeatureRow(**row) for row in df.to_dict("records")]

        return FeatureHistoryResponse(
            total_rows=len(features),
            start_date=df["date"].min() if not df.empty and "date" in df.columns else None,
            end_date=df["date"].max() if not df.empty and "date" in df.columns else None,
            features=features,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read features: {e}")


@app.get(
    "/features/scorecard",
    response_model=ScorecardResponse,
    summary="Signal accuracy report",
    tags=["MLOps"],
)
def get_signal_scorecard(
    _: None = Depends(verify_api_key),
) -> ScorecardResponse:
    """
    Evaluate historical signal accuracy using forward returns.

    For each BUY/SELL signal in the feature store, computes what actually
    happened 5, 10, and 20 trading days later. Returns per-signal scores
    and aggregated win rates.

    Note: Requires feature history data. Returns empty results if the
    feature store is new.
    """
    try:
        scorecard = SignalScorecard(feature_store=_feature_store)
        report = scorecard.full_report()

        # Convert nested dicts to Pydantic models
        all_scores = [SignalScoreItem(**s) for s in report["all_scores"]]

        buy_summary = None
        if report["buy_summary"]:
            buy_summary = SignalTypeSummary(**report["buy_summary"])

        sell_summary = None
        if report["sell_summary"]:
            sell_summary = SignalTypeSummary(**report["sell_summary"])

        return ScorecardResponse(
            data_range=report["data_range"],
            total_signals=report["total_signals"],
            buy_summary=buy_summary,
            sell_summary=sell_summary,
            all_scores=all_scores,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scorecard failed: {e}")


@app.get(
    "/experiments",
    response_model=ExperimentListResponse,
    summary="Backtest experiment log",
    tags=["MLOps"],
)
def get_experiments(
    recent: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Number of most recent experiments to return",
    ),
    _: None = Depends(verify_api_key),
) -> ExperimentListResponse:
    """
    List backtest experiment records (auto-logged each time /backtest/run is called).

    Shows parameters used and results obtained for each run. Useful for
    comparing different parameter combinations and tracking strategy performance.
    """
    try:
        df = _experiment_log.recent(n=recent)
        experiments = [ExperimentRecord(**row) for row in df.to_dict("records")]

        # Find best Sharpe across all experiments
        best_series = _experiment_log.best_by("sharpe_ratio")
        best_sharpe = ExperimentRecord(**best_series.to_dict()) if best_series is not None else None

        return ExperimentListResponse(
            total_experiments=_experiment_log.count(),
            best_sharpe=best_sharpe,
            experiments=experiments,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read experiments: {e}")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health_check() -> HealthResponse:
    """
    Returns service status, uptime, and data freshness.
    No authentication required.
    """
    # Check yfinance connectivity
    yfinance_ok = True
    try:
        import yfinance as yf
        yf.Ticker("QQQ").info
    except Exception:
        yfinance_ok = False

    return HealthResponse(
        status="healthy" if yfinance_ok else "degraded",
        version="1.0.0",
        uptime_seconds=round(time.time() - _startup_time, 1),
        data_last_updated=_signal_engine.data_last_updated,
        components={
            "yfinance": "ok" if yfinance_ok else "unreachable",
            "cache_ttl_seconds": 900,
            "active_subscriptions": len(_subscriptions),
            "feature_store_rows": _feature_store.count(),
            "feature_store_latest": _feature_store.latest_date(),
            "experiment_log_count": _experiment_log.count(),
        },
    )
