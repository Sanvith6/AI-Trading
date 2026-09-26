"""
Laya Feature Formatter & Causal State Serialization.

Enforces Directives 4, 8, and 23:
- Strictly point-in-time causal feature formatting. Zero future leakage.
- Transforms raw candidate bar, alpha signals, ML probability, regime, fundamentals,
  and portfolio risk state into a strongly typed `LayaMarketState`.
- Computes SHA256 `state_hash` for deterministic caching and replay.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .schemas import LayaMarketState

logger = logging.getLogger(__name__)


class LayaFeatureFormatter:
    """Transforms point-in-time candidate state into standardized LayaMarketState."""

    @classmethod
    def format_state(
        cls,
        symbol: str,
        timestamp: str,
        market: str = "US",
        exchange: str = "NASDAQ",
        bar: Optional[Dict[str, float]] = None,
        technical_indicators: Optional[Dict[str, float]] = None,
        alpha_signals: Optional[Dict[str, float]] = None,
        ml_prediction: Optional[Dict[str, float]] = None,
        regime_info: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None,
        portfolio_state: Optional[Dict[str, Any]] = None,
        risk_state: Optional[Dict[str, Any]] = None,
        event_state: Optional[Dict[str, Any]] = None,
        fundamental_state: Optional[Dict[str, Any]] = None,
    ) -> LayaMarketState:
        """
        Constructs a sanitized, point-in-time causal LayaMarketState.
        All fields are validated for types and defaults provided for missing data.
        """
        b = bar or {}
        price_state = {
            "open": float(b.get("open", 0.0)),
            "high": float(b.get("high", 0.0)),
            "low": float(b.get("low", 0.0)),
            "close": float(b.get("close", 0.0)),
            "volume": float(b.get("volume", 0.0)),
        }

        tech = technical_indicators or {}
        technical_state = {
            "ret_1d": float(tech.get("ret_1d", tech.get("ret_1", 0.0))),
            "ret_5d": float(tech.get("ret_5d", tech.get("ret_5", 0.0))),
            "ret_20d": float(tech.get("ret_20d", tech.get("ret_20", 0.0))),
            "rsi_14": float(tech.get("rsi_14", 50.0)),
            "adx_14": float(tech.get("adx_14", 20.0)),
            "vol_20d": float(tech.get("vol_20d", 0.20)),
            "atr_pct": float(tech.get("atr_pct", 0.02)),
        }

        alphas = alpha_signals or {}
        alpha_state = {k: round(float(v), 4) for k, v in alphas.items()}

        ml = ml_prediction or {}
        ml_state = {
            "ml_prob_up": float(ml.get("ml_prob_up", ml.get("prob_up", 0.50))),
            "ml_confidence": float(ml.get("ml_confidence", ml.get("confidence", 0.50))),
            "ml_signal": float(ml.get("ml_signal", ml.get("signal", 0.0))),
        }

        reg = regime_info or {}
        regime_state = {
            "current_regime": str(reg.get("current_regime", reg.get("regime", "SIDEWAYS"))),
            "confidence": float(reg.get("confidence", 0.70)),
            "regime_id": int(reg.get("regime_id", 0)),
        }

        ctx = market_context or {}
        sanitized_context = {
            "market_trend": str(ctx.get("market_trend", "NEUTRAL")),
            "vix_level": float(ctx.get("vix_level", ctx.get("vix", 16.5))),
            "risk_on_off_score": float(ctx.get("risk_on_off_score", 0.0)),
            "spy_ret_20d": float(ctx.get("spy_ret_20d", 0.01)),
        }

        port = portfolio_state or {}
        sanitized_port = {
            "exposure_pct": float(port.get("exposure_pct", 0.0)),
            "drawdown_pct": float(port.get("drawdown_pct", 0.0)),
            "num_positions": int(port.get("num_positions", 0)),
            "available_cash_pct": float(port.get("available_cash_pct", 1.0)),
        }

        r_state = risk_state or {}
        sanitized_risk = {
            "var_95": float(r_state.get("var_95", 0.02)),
            "max_drawdown_limit": float(r_state.get("max_drawdown_limit", 0.15)),
            "margin_usage": float(r_state.get("margin_usage", 0.0)),
        }

        ev = event_state or {}
        sanitized_event = {
            "days_to_earnings": int(ev.get("days_to_earnings", 45)),
            "macro_event_flag": bool(ev.get("macro_event_flag", False)),
        }

        f = fundamental_state or {}
        sanitized_fundamental = {
            "f_quality_score": float(f.get("f_quality_score", 0.60)),
            "f_operating_margin": float(f.get("f_operating_margin", 0.18)),
            "pe_ratio": float(f.get("pe_ratio", 22.0)),
        }

        return LayaMarketState(
            market=market,
            exchange=exchange,
            symbol=symbol,
            timestamp=str(timestamp),
            price_state=price_state,
            technical_state=technical_state,
            alpha_state=alpha_state,
            ml_state=ml_state,
            regime_state=regime_state,
            market_context=sanitized_context,
            portfolio_state=sanitized_port,
            risk_state=sanitized_risk,
            event_state=sanitized_event,
            fundamental_state=sanitized_fundamental,
            schema_version="1.0",
        )
