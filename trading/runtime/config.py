"""
Trading Runtime Configuration & Specification.

Enforces:
1. Strict Enum-based configuration (MARKET=US, TRADING_MODE=paper|live, DATA_MODE=live|historical|replay).
2. Hard defaults: TRADING_MODE=paper, LAYA_MODE=shadow, LIVE_TRADING_ENABLED=false.
3. Prohibits ambiguous boolean flags (USE_REAL, PAPER=false).
4. Validation of legal combinations (fails closed on invalid combinations).
"""

import os
import sys
import json
import hashlib
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

load_dotenv(override=True)


class MarketType(str, Enum):
    US = "US"


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class DataMode(str, Enum):
    LIVE = "live"
    HISTORICAL = "historical"
    REPLAY = "replay"


class LayaMode(str, Enum):
    SHADOW = "shadow"
    ADVISORY = "advisory"


class BrokerProvider(str, Enum):
    PAPER = "paper"
    ALPACA = "alpaca"
    IBKR = "ibkr"
    MOCK_LIVE = "mock_live"


@dataclass(frozen=True)
class RuntimeConfig:
    market: MarketType = MarketType.US
    trading_mode: TradingMode = TradingMode.PAPER
    data_mode: DataMode = DataMode.LIVE
    laya_model: str = "laya-financial-us-v2"
    laya_mode: LayaMode = LayaMode.SHADOW
    broker_provider: BrokerProvider = BrokerProvider.PAPER
    paper_initial_capital: float = 100000.0
    live_trading_enabled: bool = False
    live_confirmation_required: bool = True
    live_confirmation_token: Optional[str] = None
    max_data_age_seconds: int = 60
    max_daily_loss: float = 2000.0
    max_per_order: float = 5000.0
    max_positions: int = 10
    stop_loss_pct: float = 2.0
    target_pct: float = 4.0
    config_version: str = "runtime_us_v2.0"

    def validate(self) -> None:
        """Enforces legal configuration combinations. Fails closed."""
        if self.market != MarketType.US:
            raise ValueError(f"Invalid MARKET: {self.market}. Only US market is supported in this release.")

        if self.trading_mode == TradingMode.LIVE:
            if not self.live_trading_enabled:
                raise ValueError("LIVE trading requested but LIVE_TRADING_ENABLED is False. Execution blocked.")
            if self.data_mode != DataMode.LIVE:
                raise ValueError(f"Illegal configuration: LIVE trading requires DATA_MODE=live, got {self.data_mode}.")
            if self.broker_provider == BrokerProvider.PAPER:
                raise ValueError("Illegal configuration: LIVE trading cannot use BrokerProvider.PAPER.")

        if self.trading_mode == TradingMode.PAPER:
            if self.broker_provider not in (BrokerProvider.PAPER, BrokerProvider.MOCK_LIVE):
                raise ValueError(f"Illegal configuration: PAPER trading cannot use live broker {self.broker_provider}.")

        if self.data_mode == DataMode.REPLAY and self.trading_mode == TradingMode.LIVE:
            raise ValueError("Illegal configuration: REPLAY data mode cannot be paired with LIVE trading.")

        if self.laya_model != "laya-financial-us-v2":
            raise ValueError(f"Unknown or unsupported Laya specialist model: {self.laya_model}. Expected 'laya-financial-us-v2'.")

    def compute_hash(self) -> str:
        """Produces deterministic SHA256 configuration hash."""
        d = asdict(self)
        # Exclude secrets/transient tokens
        d.pop("live_confirmation_token", None)
        serialized = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """Loads configuration strictly from environment with safe defaults."""
        m_str = os.getenv("MARKET", "US").strip().upper()
        t_str = os.getenv("TRADING_MODE", "paper").strip().lower()
        d_str = os.getenv("DATA_MODE", "live").strip().lower()
        lm_str = os.getenv("LAYA_MODE", "shadow").strip().lower()
        b_str = os.getenv("BROKER_PROVIDER", "paper").strip().lower()

        # Parse enums safely
        try:
            market = MarketType(m_str)
        except ValueError:
            raise ValueError(f"Invalid MARKET '{m_str}'. Must be 'US'.")

        try:
            trading_mode = TradingMode(t_str)
        except ValueError:
            raise ValueError(f"Invalid TRADING_MODE '{t_str}'. Must be 'paper' or 'live'.")

        try:
            data_mode = DataMode(d_str)
        except ValueError:
            raise ValueError(f"Invalid DATA_MODE '{d_str}'. Must be 'live', 'historical', or 'replay'.")

        try:
            laya_mode = LayaMode(lm_str)
        except ValueError:
            raise ValueError(f"Invalid LAYA_MODE '{lm_str}'. Must be 'shadow' or 'advisory'.")

        try:
            broker_provider = BrokerProvider(b_str)
        except ValueError:
            raise ValueError(f"Invalid BROKER_PROVIDER '{b_str}'.")

        live_enabled = os.getenv("LIVE_TRADING_ENABLED", "false").strip().lower() in ("true", "1", "yes")
        live_confirm = os.getenv("LIVE_CONFIRMATION_REQUIRED", "true").strip().lower() in ("true", "1", "yes")
        live_token = os.getenv("LIVE_CONFIRMATION_TOKEN", None)
        laya_model = os.getenv("LAYA_MODEL", "laya-financial-us-v2").strip()

        cfg = cls(
            market=market,
            trading_mode=trading_mode,
            data_mode=data_mode,
            laya_model=laya_model,
            laya_mode=laya_mode,
            broker_provider=broker_provider,
            paper_initial_capital=float(os.getenv("PAPER_INITIAL_CAPITAL", "100000.0")),
            live_trading_enabled=live_enabled,
            live_confirmation_required=live_confirm,
            live_confirmation_token=live_token,
            max_data_age_seconds=int(os.getenv("MAX_DATA_AGE_SECONDS", "60")),
            max_daily_loss=float(os.getenv("TRADE_MAX_DAILY_LOSS", "2000.0")),
            max_per_order=float(os.getenv("TRADE_MAX_PER_ORDER", "5000.0")),
            max_positions=int(os.getenv("TRADE_MAX_POSITIONS", "10")),
            stop_loss_pct=float(os.getenv("TRADE_STOP_LOSS_PCT", "2.0")),
            target_pct=float(os.getenv("TRADE_TARGET_PCT", "4.0")),
        )
        cfg.validate()
        return cfg
