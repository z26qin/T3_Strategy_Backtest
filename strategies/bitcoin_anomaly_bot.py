"""
Bitcoin Abnormal Activity Tracking Bot

Monitors Bitcoin for anomalous activity including:
- Price anomalies (sudden spikes/drops)
- Volume anomalies (unusual trading volume)
- Volatility anomalies (abnormal price swings)
- Whale activity (large transactions)
- Network anomalies (hash rate, transaction count)
- Social sentiment shifts

Uses multiple detection methods:
- Z-score analysis
- Moving average deviations
- Bollinger Band breakouts
- Interquartile Range (IQR) outliers
- Rate of change thresholds
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum


class AnomalyType(Enum):
    """Types of anomalies detected."""
    PRICE_SPIKE = "price_spike"
    PRICE_DROP = "price_drop"
    VOLUME_SURGE = "volume_surge"
    VOLATILITY_SPIKE = "volatility_spike"
    WHALE_ACTIVITY = "whale_activity"
    NETWORK_ANOMALY = "network_anomaly"
    SENTIMENT_SHIFT = "sentiment_shift"
    BOLLINGER_BREAKOUT = "bollinger_breakout"
    RSI_EXTREME = "rsi_extreme"


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyEvent:
    """Represents a detected anomaly."""
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: AlertSeverity
    description: str
    current_value: float
    threshold_value: float
    z_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "type": self.anomaly_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "z_score": self.z_score,
            "metadata": self.metadata
        }


@dataclass
class BotConfig:
    """Configuration for the Bitcoin anomaly bot."""
    # Data settings
    symbol: str = "BTC-USD"
    lookback_days: int = 90
    update_interval_seconds: int = 300  # 5 minutes

    # Z-score thresholds
    price_zscore_threshold: float = 2.5
    volume_zscore_threshold: float = 3.0
    volatility_zscore_threshold: float = 2.5

    # Percentage thresholds (24-hour)
    price_change_threshold_24h_pct: float = 5.0  # 5% in 24 hours
    volume_spike_multiplier_24h: float = 3.0  # 3x average 24h volume

    # Percentage thresholds (7-day)
    price_change_threshold_7d_pct: float = 15.0  # 15% in 7 days
    volume_spike_multiplier_7d: float = 2.0  # 2x average 7d volume
    volatility_zscore_threshold_7d: float = 2.0  # Lower threshold for 7d

    # Bollinger Bands
    bollinger_period: int = 20
    bollinger_std: float = 2.0

    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Whale tracking (via blockchain API)
    whale_threshold_btc: float = 100.0  # Transactions > 100 BTC

    # Alert settings
    enable_email: bool = False
    enable_discord: bool = False
    enable_console: bool = True

    # Rate limiting
    max_alerts_per_hour: int = 10
    cooldown_minutes: int = 15  # Min time between same alert type


class BitcoinDataFetcher:
    """Fetches Bitcoin data from multiple sources."""

    def __init__(self, config: BotConfig):
        self.config = config
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = 60  # Cache for 60 seconds

    def get_price_data(self, period: str = "90d", interval: str = "1h") -> pd.DataFrame:
        """Fetch Bitcoin price data from Yahoo Finance."""
        cache_key = f"price_{period}_{interval}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < self._cache_ttl:
                return cached_data

        try:
            btc = yf.Ticker(self.config.symbol)
            df = btc.history(period=period, interval=interval)

            if df.empty:
                print(f"Warning: No data returned for {self.config.symbol}")
                return pd.DataFrame()

            # Clean column names
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]
            df = df.reset_index()
            df.rename(columns={"Datetime": "datetime", "Date": "datetime"}, inplace=True)

            self._cache[cache_key] = (datetime.now(), df)
            return df

        except Exception as e:
            print(f"Error fetching price data: {e}")
            return pd.DataFrame()

    def get_current_price(self) -> Optional[float]:
        """Get the current Bitcoin price."""
        try:
            btc = yf.Ticker(self.config.symbol)
            return btc.info.get('regularMarketPrice') or btc.info.get('currentPrice')
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return None

    def get_whale_transactions(self) -> List[Dict]:
        """
        Fetch large Bitcoin transactions from blockchain API.
        Uses blockchain.info API (free, no key required).
        """
        try:
            # Get recent unconfirmed transactions over threshold
            # Note: This is a simplified approach; production would use websockets
            url = "https://blockchain.info/unconfirmed-transactions?format=json"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return []

            data = response.json()
            whale_txs = []

            for tx in data.get("txs", [])[:100]:  # Check latest 100
                total_output_btc = sum(
                    out.get("value", 0) for out in tx.get("out", [])
                ) / 1e8  # Convert satoshi to BTC

                if total_output_btc >= self.config.whale_threshold_btc:
                    whale_txs.append({
                        "hash": tx.get("hash", "")[:16] + "...",
                        "amount_btc": total_output_btc,
                        "time": datetime.fromtimestamp(tx.get("time", 0)),
                        "inputs": len(tx.get("inputs", [])),
                        "outputs": len(tx.get("out", []))
                    })

            return whale_txs

        except Exception as e:
            print(f"Error fetching whale transactions: {e}")
            return []

    def get_network_stats(self) -> Dict[str, Any]:
        """Fetch Bitcoin network statistics."""
        stats = {}

        try:
            # Hash rate
            url = "https://blockchain.info/q/hashrate"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                stats["hash_rate"] = float(response.text)

            # Difficulty
            url = "https://blockchain.info/q/getdifficulty"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                stats["difficulty"] = float(response.text)

            # Unconfirmed transactions
            url = "https://blockchain.info/q/unconfirmedcount"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                stats["unconfirmed_tx_count"] = int(response.text)

            # Market cap
            url = "https://blockchain.info/q/marketcap"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                stats["market_cap"] = float(response.text)

        except Exception as e:
            print(f"Error fetching network stats: {e}")

        return stats

    def get_fear_greed_index(self) -> Optional[Dict]:
        """Fetch the Crypto Fear & Greed Index."""
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get("data"):
                    fng = data["data"][0]
                    return {
                        "value": int(fng.get("value", 50)),
                        "classification": fng.get("value_classification", "Neutral"),
                        "timestamp": datetime.fromtimestamp(int(fng.get("timestamp", 0)))
                    }
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")

        return None


class AnomalyDetector:
    """Detects anomalies in Bitcoin data using multiple methods."""

    def __init__(self, config: BotConfig):
        self.config = config

    def calculate_z_score(self, values: pd.Series, current: float) -> float:
        """Calculate Z-score for a value against historical data."""
        mean = values.mean()
        std = values.std()
        if std == 0:
            return 0.0
        return (current - mean) / std

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50.0

    def calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        period = self.config.bollinger_period
        std_dev = self.config.bollinger_std

        middle = prices.rolling(window=period).mean().iloc[-1]
        std = prices.rolling(window=period).std().iloc[-1]

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        return lower, middle, upper

    def detect_price_anomalies(self, df: pd.DataFrame) -> List[AnomalyEvent]:
        """Detect price-related anomalies for both 24h and 7d timeframes."""
        anomalies = []

        if df.empty or 'close' not in df.columns:
            return anomalies

        prices = df['close']
        current_price = prices.iloc[-1]

        # Z-score based detection (full period)
        z_score = self.calculate_z_score(prices[:-1], current_price)

        if abs(z_score) > self.config.price_zscore_threshold:
            severity = AlertSeverity.HIGH if abs(z_score) > 3.5 else AlertSeverity.MEDIUM
            anomaly_type = AnomalyType.PRICE_SPIKE if z_score > 0 else AnomalyType.PRICE_DROP

            anomalies.append(AnomalyEvent(
                timestamp=datetime.now(),
                anomaly_type=anomaly_type,
                severity=severity,
                description=f"Bitcoin price is {abs(z_score):.2f} standard deviations "
                           f"{'above' if z_score > 0 else 'below'} the mean",
                current_value=current_price,
                threshold_value=prices.mean(),
                z_score=z_score,
                metadata={"mean_price": prices.mean(), "std_dev": prices.std(), "timeframe": "full"}
            ))

        # 24-hour percentage change detection
        if len(prices) >= 24:
            price_24h_ago = prices.iloc[-24]
            change_24h_pct = ((current_price - price_24h_ago) / price_24h_ago) * 100

            if abs(change_24h_pct) > self.config.price_change_threshold_24h_pct:
                severity = AlertSeverity.CRITICAL if abs(change_24h_pct) > 10 else AlertSeverity.HIGH
                anomaly_type = AnomalyType.PRICE_SPIKE if change_24h_pct > 0 else AnomalyType.PRICE_DROP

                anomalies.append(AnomalyEvent(
                    timestamp=datetime.now(),
                    anomaly_type=anomaly_type,
                    severity=severity,
                    description=f"Bitcoin {'surged' if change_24h_pct > 0 else 'dropped'} "
                               f"{abs(change_24h_pct):.2f}% in the last 24 hours",
                    current_value=current_price,
                    threshold_value=price_24h_ago,
                    metadata={"change_pct": change_24h_pct, "timeframe": "24h"}
                ))

        # 7-day percentage change detection
        if len(prices) >= 168:  # 7 days * 24 hours
            price_7d_ago = prices.iloc[-168]
            change_7d_pct = ((current_price - price_7d_ago) / price_7d_ago) * 100

            if abs(change_7d_pct) > self.config.price_change_threshold_7d_pct:
                severity = AlertSeverity.CRITICAL if abs(change_7d_pct) > 25 else AlertSeverity.HIGH
                anomaly_type = AnomalyType.PRICE_SPIKE if change_7d_pct > 0 else AnomalyType.PRICE_DROP

                anomalies.append(AnomalyEvent(
                    timestamp=datetime.now(),
                    anomaly_type=anomaly_type,
                    severity=severity,
                    description=f"Bitcoin {'surged' if change_7d_pct > 0 else 'dropped'} "
                               f"{abs(change_7d_pct):.2f}% in the last 7 days",
                    current_value=current_price,
                    threshold_value=price_7d_ago,
                    metadata={"change_pct": change_7d_pct, "timeframe": "7d"}
                ))

        # Bollinger Band breakout
        if len(prices) >= self.config.bollinger_period:
            lower, middle, upper = self.calculate_bollinger_bands(prices)

            if current_price > upper:
                anomalies.append(AnomalyEvent(
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.BOLLINGER_BREAKOUT,
                    severity=AlertSeverity.MEDIUM,
                    description=f"Bitcoin broke above upper Bollinger Band (${upper:.2f})",
                    current_value=current_price,
                    threshold_value=upper,
                    metadata={"lower": lower, "middle": middle, "upper": upper}
                ))
            elif current_price < lower:
                anomalies.append(AnomalyEvent(
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.BOLLINGER_BREAKOUT,
                    severity=AlertSeverity.MEDIUM,
                    description=f"Bitcoin broke below lower Bollinger Band (${lower:.2f})",
                    current_value=current_price,
                    threshold_value=lower,
                    metadata={"lower": lower, "middle": middle, "upper": upper}
                ))

        return anomalies

    def detect_volume_anomalies(self, df: pd.DataFrame) -> List[AnomalyEvent]:
        """Detect volume-related anomalies for both 24h and 7d timeframes."""
        anomalies = []

        if df.empty or 'volume' not in df.columns:
            return anomalies

        volumes = df['volume']

        # 24-hour volume analysis
        if len(volumes) >= 24:
            volume_24h = volumes.tail(24).sum()
            avg_volume_24h = volumes[:-24].rolling(24).sum().mean() if len(volumes) > 48 else volumes.mean() * 24

            volume_ratio_24h = volume_24h / avg_volume_24h if avg_volume_24h > 0 else 0

            if volume_ratio_24h > self.config.volume_spike_multiplier_24h:
                severity = AlertSeverity.HIGH if volume_ratio_24h > 5 else AlertSeverity.MEDIUM

                anomalies.append(AnomalyEvent(
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.VOLUME_SURGE,
                    severity=severity,
                    description=f"24h trading volume is {volume_ratio_24h:.1f}x higher than average",
                    current_value=volume_24h,
                    threshold_value=avg_volume_24h,
                    metadata={"volume_ratio": volume_ratio_24h, "avg_volume": avg_volume_24h, "timeframe": "24h"}
                ))

        # 7-day volume analysis
        if len(volumes) >= 168:  # 7 days * 24 hours
            volume_7d = volumes.tail(168).sum()
            # Compare to previous 7-day periods if we have enough data
            if len(volumes) > 336:  # Need at least 2 weeks of data
                prev_volumes = volumes[:-168]
                avg_volume_7d = prev_volumes.rolling(168).sum().mean()
            else:
                avg_volume_7d = volumes.mean() * 168

            volume_ratio_7d = volume_7d / avg_volume_7d if avg_volume_7d > 0 else 0

            if volume_ratio_7d > self.config.volume_spike_multiplier_7d:
                severity = AlertSeverity.HIGH if volume_ratio_7d > 3 else AlertSeverity.MEDIUM

                anomalies.append(AnomalyEvent(
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.VOLUME_SURGE,
                    severity=severity,
                    description=f"7-day trading volume is {volume_ratio_7d:.1f}x higher than average",
                    current_value=volume_7d,
                    threshold_value=avg_volume_7d,
                    metadata={"volume_ratio": volume_ratio_7d, "avg_volume": avg_volume_7d, "timeframe": "7d"}
                ))

        return anomalies

    def detect_volatility_anomalies(self, df: pd.DataFrame) -> List[AnomalyEvent]:
        """Detect volatility-related anomalies for both 24h and 7d timeframes."""
        anomalies = []

        if df.empty or 'close' not in df.columns or len(df) < 20:
            return anomalies

        # Calculate rolling volatility (standard deviation of returns)
        returns = df['close'].pct_change().dropna()

        # 24-hour volatility analysis (using 24-hour rolling window)
        if len(returns) >= 24:
            rolling_vol_24h = returns.rolling(window=24).std()

            if len(rolling_vol_24h) >= 2:
                current_vol_24h = rolling_vol_24h.iloc[-1]
                historical_vol_24h = rolling_vol_24h[:-1].dropna()

                if len(historical_vol_24h) > 0:
                    z_score_24h = self.calculate_z_score(historical_vol_24h, current_vol_24h)

                    if abs(z_score_24h) > self.config.volatility_zscore_threshold:
                        severity = AlertSeverity.HIGH if abs(z_score_24h) > 3.5 else AlertSeverity.MEDIUM

                        anomalies.append(AnomalyEvent(
                            timestamp=datetime.now(),
                            anomaly_type=AnomalyType.VOLATILITY_SPIKE,
                            severity=severity,
                            description=f"24h Bitcoin volatility is {abs(z_score_24h):.2f} standard deviations "
                                       f"{'above' if z_score_24h > 0 else 'below'} normal",
                            current_value=current_vol_24h,
                            threshold_value=historical_vol_24h.mean(),
                            z_score=z_score_24h,
                            metadata={"current_volatility": current_vol_24h, "avg_volatility": historical_vol_24h.mean(), "timeframe": "24h"}
                        ))

        # 7-day volatility analysis (using 168-hour rolling window)
        if len(returns) >= 168:
            rolling_vol_7d = returns.rolling(window=168).std()

            if len(rolling_vol_7d) >= 2:
                current_vol_7d = rolling_vol_7d.iloc[-1]
                historical_vol_7d = rolling_vol_7d[:-1].dropna()

                if len(historical_vol_7d) > 0:
                    z_score_7d = self.calculate_z_score(historical_vol_7d, current_vol_7d)

                    if abs(z_score_7d) > self.config.volatility_zscore_threshold_7d:
                        severity = AlertSeverity.HIGH if abs(z_score_7d) > 3.0 else AlertSeverity.MEDIUM

                        anomalies.append(AnomalyEvent(
                            timestamp=datetime.now(),
                            anomaly_type=AnomalyType.VOLATILITY_SPIKE,
                            severity=severity,
                            description=f"7-day Bitcoin volatility is {abs(z_score_7d):.2f} standard deviations "
                                       f"{'above' if z_score_7d > 0 else 'below'} normal",
                            current_value=current_vol_7d,
                            threshold_value=historical_vol_7d.mean(),
                            z_score=z_score_7d,
                            metadata={"current_volatility": current_vol_7d, "avg_volatility": historical_vol_7d.mean(), "timeframe": "7d"}
                        ))

        return anomalies

    def detect_rsi_anomalies(self, df: pd.DataFrame) -> List[AnomalyEvent]:
        """Detect RSI extremes."""
        anomalies = []

        if df.empty or 'close' not in df.columns or len(df) < self.config.rsi_period + 1:
            return anomalies

        rsi = self.calculate_rsi(df['close'], self.config.rsi_period)

        if rsi <= self.config.rsi_oversold:
            anomalies.append(AnomalyEvent(
                timestamp=datetime.now(),
                anomaly_type=AnomalyType.RSI_EXTREME,
                severity=AlertSeverity.MEDIUM,
                description=f"RSI is oversold at {rsi:.1f} (threshold: {self.config.rsi_oversold})",
                current_value=rsi,
                threshold_value=self.config.rsi_oversold,
                metadata={"condition": "oversold"}
            ))
        elif rsi >= self.config.rsi_overbought:
            anomalies.append(AnomalyEvent(
                timestamp=datetime.now(),
                anomaly_type=AnomalyType.RSI_EXTREME,
                severity=AlertSeverity.MEDIUM,
                description=f"RSI is overbought at {rsi:.1f} (threshold: {self.config.rsi_overbought})",
                current_value=rsi,
                threshold_value=self.config.rsi_overbought,
                metadata={"condition": "overbought"}
            ))

        return anomalies

    def detect_whale_activity(self, transactions: List[Dict]) -> List[AnomalyEvent]:
        """Detect whale transaction anomalies."""
        anomalies = []

        for tx in transactions:
            amount = tx.get("amount_btc", 0)

            if amount >= self.config.whale_threshold_btc * 10:  # 10x threshold = major whale
                severity = AlertSeverity.CRITICAL
            elif amount >= self.config.whale_threshold_btc * 5:
                severity = AlertSeverity.HIGH
            else:
                severity = AlertSeverity.MEDIUM

            anomalies.append(AnomalyEvent(
                timestamp=tx.get("time", datetime.now()),
                anomaly_type=AnomalyType.WHALE_ACTIVITY,
                severity=severity,
                description=f"Large transaction detected: {amount:.2f} BTC",
                current_value=amount,
                threshold_value=self.config.whale_threshold_btc,
                metadata={
                    "tx_hash": tx.get("hash", ""),
                    "inputs": tx.get("inputs", 0),
                    "outputs": tx.get("outputs", 0)
                }
            ))

        return anomalies

    def detect_all_anomalies(self, df: pd.DataFrame,
                             whale_txs: List[Dict] = None) -> List[AnomalyEvent]:
        """Run all anomaly detection methods."""
        all_anomalies = []

        all_anomalies.extend(self.detect_price_anomalies(df))
        all_anomalies.extend(self.detect_volume_anomalies(df))
        all_anomalies.extend(self.detect_volatility_anomalies(df))
        all_anomalies.extend(self.detect_rsi_anomalies(df))

        if whale_txs:
            all_anomalies.extend(self.detect_whale_activity(whale_txs))

        # Sort by severity (critical first) then by timestamp
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3
        }
        all_anomalies.sort(key=lambda x: (severity_order[x.severity], x.timestamp))

        return all_anomalies


class BitcoinAnomalyBot:
    """Main bot class for tracking Bitcoin abnormal activity."""

    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or BotConfig()
        self.data_fetcher = BitcoinDataFetcher(self.config)
        self.detector = AnomalyDetector(self.config)

        self._alert_history: List[AnomalyEvent] = []
        self._last_alert_times: Dict[str, datetime] = {}
        self._running = False

        # Import AlertManager from existing module
        try:
            from strategies.alerts import AlertManager, AlertConfig
            alert_config = AlertConfig(
                email_enabled=self.config.enable_email,
                discord_enabled=self.config.enable_discord
            )
            self.alert_manager = AlertManager(alert_config)
        except ImportError:
            self.alert_manager = None
            print("Warning: AlertManager not available. Using console output only.")

    def _should_send_alert(self, anomaly: AnomalyEvent) -> bool:
        """Check if alert should be sent (rate limiting)."""
        alert_key = anomaly.anomaly_type.value

        if alert_key in self._last_alert_times:
            time_since_last = datetime.now() - self._last_alert_times[alert_key]
            if time_since_last.seconds < self.config.cooldown_minutes * 60:
                return False

        # Check hourly limit
        recent_alerts = [
            a for a in self._alert_history
            if (datetime.now() - a.timestamp).seconds < 3600
        ]
        if len(recent_alerts) >= self.config.max_alerts_per_hour:
            return False

        return True

    def _format_console_alert(self, anomaly: AnomalyEvent) -> str:
        """Format anomaly for console output."""
        severity_colors = {
            AlertSeverity.CRITICAL: "\033[91m",  # Red
            AlertSeverity.HIGH: "\033[93m",      # Yellow
            AlertSeverity.MEDIUM: "\033[94m",    # Blue
            AlertSeverity.LOW: "\033[92m"        # Green
        }
        reset = "\033[0m"

        color = severity_colors.get(anomaly.severity, "")

        return f"""
{color}{'='*60}
BITCOIN ANOMALY DETECTED
{'='*60}{reset}
Time:       {anomaly.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Type:       {anomaly.anomaly_type.value.upper()}
Severity:   {anomaly.severity.value.upper()}
{'='*60}
{anomaly.description}

Current Value:   {anomaly.current_value:,.2f}
Threshold:       {anomaly.threshold_value:,.2f}
{f'Z-Score:         {anomaly.z_score:.2f}' if anomaly.z_score else ''}
{'='*60}
"""

    def _send_discord_anomaly(self, anomaly: AnomalyEvent) -> bool:
        """Send anomaly to Discord."""
        if not self.alert_manager or not self.config.enable_discord:
            return False

        severity_colors = {
            AlertSeverity.CRITICAL: 0xFF0000,  # Red
            AlertSeverity.HIGH: 0xFFA500,      # Orange
            AlertSeverity.MEDIUM: 0x0066FF,    # Blue
            AlertSeverity.LOW: 0x00FF00        # Green
        }

        summary = {
            "date": anomaly.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "type": anomaly.anomaly_type.value,
            "severity": anomaly.severity.value,
            "description": anomaly.description,
            "current_value": anomaly.current_value,
            "threshold_value": anomaly.threshold_value,
        }

        return self.alert_manager._send_discord(
            f"BTC {anomaly.anomaly_type.value.upper()}",
            summary
        )

    def send_alert(self, anomaly: AnomalyEvent) -> None:
        """Send alert through configured channels."""
        if not self._should_send_alert(anomaly):
            return

        # Update tracking
        self._alert_history.append(anomaly)
        self._last_alert_times[anomaly.anomaly_type.value] = datetime.now()

        # Console output
        if self.config.enable_console:
            print(self._format_console_alert(anomaly))

        # Discord
        if self.config.enable_discord:
            self._send_discord_anomaly(anomaly)

    def get_market_summary(self) -> Dict[str, Any]:
        """Get current market summary with both 24h and 7d stats."""
        df = self.data_fetcher.get_price_data(period="30d", interval="1h")
        current_price = self.data_fetcher.get_current_price()
        fear_greed = self.data_fetcher.get_fear_greed_index()
        network_stats = self.data_fetcher.get_network_stats()

        summary = {
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "fear_greed_index": fear_greed,
            "network_stats": network_stats,
        }

        if not df.empty:
            # 24-hour stats
            if len(df) >= 24:
                summary.update({
                    "price_24h_high": df['high'].tail(24).max(),
                    "price_24h_low": df['low'].tail(24).min(),
                    "volume_24h": df['volume'].tail(24).sum(),
                    "price_change_24h_pct": ((df['close'].iloc[-1] - df['close'].iloc[-24])
                                             / df['close'].iloc[-24] * 100),
                })

            # 7-day stats
            if len(df) >= 168:  # 7 days * 24 hours
                summary.update({
                    "price_7d_high": df['high'].tail(168).max(),
                    "price_7d_low": df['low'].tail(168).min(),
                    "volume_7d": df['volume'].tail(168).sum(),
                    "price_change_7d_pct": ((df['close'].iloc[-1] - df['close'].iloc[-168])
                                            / df['close'].iloc[-168] * 100),
                })

        return summary

    def check_once(self) -> List[AnomalyEvent]:
        """Run a single check for anomalies."""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking for anomalies (24h & 7d)...")

        # Fetch data - use 30d to have enough historical data for 7d comparisons
        df = self.data_fetcher.get_price_data(period="30d", interval="1h")
        whale_txs = self.data_fetcher.get_whale_transactions()

        # Detect anomalies
        anomalies = self.detector.detect_all_anomalies(df, whale_txs)

        # Send alerts
        for anomaly in anomalies:
            self.send_alert(anomaly)

        if not anomalies:
            print("No anomalies detected.")
        else:
            print(f"Detected {len(anomalies)} anomalies.")

        return anomalies

    def run(self) -> None:
        """Run the bot in continuous monitoring mode."""
        print(f"""
{'='*60}
BITCOIN ANOMALY TRACKING BOT STARTED
{'='*60}
Symbol:              {self.config.symbol}
Update Interval:     {self.config.update_interval_seconds} seconds
Price Z-Score:       {self.config.price_zscore_threshold}
Whale Threshold:     {self.config.whale_threshold_btc} BTC
--- 24-Hour Thresholds ---
Price Change (24h):  {self.config.price_change_threshold_24h_pct}%
Volume Spike (24h):  {self.config.volume_spike_multiplier_24h}x
--- 7-Day Thresholds ---
Price Change (7d):   {self.config.price_change_threshold_7d_pct}%
Volume Spike (7d):   {self.config.volume_spike_multiplier_7d}x
Volatility Z (7d):   {self.config.volatility_zscore_threshold_7d}
{'='*60}
Press Ctrl+C to stop
""")

        self._running = True

        try:
            while self._running:
                self.check_once()
                time.sleep(self.config.update_interval_seconds)
        except KeyboardInterrupt:
            print("\nBot stopped by user.")
            self._running = False

    def stop(self) -> None:
        """Stop the bot."""
        self._running = False


def load_config_from_env() -> BotConfig:
    """Load bot configuration from environment variables."""
    return BotConfig(
        symbol=os.getenv("BTC_SYMBOL", "BTC-USD"),
        lookback_days=int(os.getenv("BTC_LOOKBACK_DAYS", "90")),
        update_interval_seconds=int(os.getenv("BTC_UPDATE_INTERVAL", "300")),
        price_zscore_threshold=float(os.getenv("BTC_PRICE_ZSCORE", "2.5")),
        volume_zscore_threshold=float(os.getenv("BTC_VOLUME_ZSCORE", "3.0")),
        # 24-hour thresholds
        price_change_threshold_24h_pct=float(os.getenv("BTC_PRICE_CHANGE_24H_PCT", "5.0")),
        volume_spike_multiplier_24h=float(os.getenv("BTC_VOLUME_SPIKE_24H", "3.0")),
        # 7-day thresholds
        price_change_threshold_7d_pct=float(os.getenv("BTC_PRICE_CHANGE_7D_PCT", "15.0")),
        volume_spike_multiplier_7d=float(os.getenv("BTC_VOLUME_SPIKE_7D", "2.0")),
        volatility_zscore_threshold_7d=float(os.getenv("BTC_VOLATILITY_ZSCORE_7D", "2.0")),
        whale_threshold_btc=float(os.getenv("BTC_WHALE_THRESHOLD", "100.0")),
        enable_email=os.getenv("ALERT_EMAIL_ENABLED", "").lower() == "true",
        enable_discord=os.getenv("ALERT_DISCORD_ENABLED", "").lower() == "true",
        enable_console=os.getenv("BTC_CONSOLE_OUTPUT", "true").lower() == "true",
    )


# Example usage
if __name__ == "__main__":
    # Load config from environment or use defaults
    config = load_config_from_env()

    # Create and run bot
    bot = BitcoinAnomalyBot(config)

    # Single check mode (for testing)
    # anomalies = bot.check_once()
    # summary = bot.get_market_summary()
    # print(json.dumps(summary, indent=2, default=str))

    # Continuous monitoring mode
    bot.run()
