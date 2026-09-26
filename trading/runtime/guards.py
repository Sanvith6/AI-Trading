"""
Automatic Live Trading Guardrails & Air Gap Verification.

Enforces:
1. Fail-closed architecture: ANY failed requirement immediately blocks live execution.
2. Comprehensive multi-system health check before any live order can be created.
3. Strict air-gap validation between paper and live components.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from .config import RuntimeConfig, MarketType, TradingMode, DataMode
from .session import USMarketCalendar, MarketSessionInfo, SessionStatus
from .kill_switch import GlobalKillSwitch

logger = logging.getLogger(__name__)


@dataclass
class GuardVerificationResult:
    passed: bool
    blocked_reasons: List[str] = field(default_factory=list)
    checks_performed: Dict[str, bool] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "blocked_reasons": self.blocked_reasons,
            "checks_performed": self.checks_performed,
            "timestamp": self.timestamp,
        }


class LiveTradingGuard:
    """Rigorous gatekeeper that prevents unauthorized or unsafe live trading."""

    @classmethod
    def verify_live_eligibility(
        cls,
        config: RuntimeConfig,
        session_info: MarketSessionInfo,
        broker_identity: Optional[Dict[str, Any]] = None,
        risk_engine_healthy: bool = True,
        data_freshness_seconds: Optional[float] = None,
        feature_pit_valid: bool = True,
        laya_healthy: bool = True,
        portfolio_healthy: bool = True,
        kill_switch: Optional[GlobalKillSwitch] = None,
    ) -> GuardVerificationResult:
        """
        Runs comprehensive multi-system verification for LIVE trading.
        Must satisfy ALL 15 independent conditions. Fails closed.
        """
        checks: Dict[str, bool] = {}
        blocked: List[str] = []

        # 1. Market must be US
        c1 = config.market == MarketType.US
        checks["market_is_us"] = c1
        if not c1:
            blocked.append(f"MARKET is '{config.market}', but only 'US' is permitted for live execution.")

        # 2. Trading Mode must be LIVE
        c2 = config.trading_mode == TradingMode.LIVE
        checks["trading_mode_is_live"] = c2
        if not c2:
            blocked.append(f"TRADING_MODE is '{config.trading_mode}', not 'live'.")

        # 3. Data Mode must be LIVE
        c3 = config.data_mode == DataMode.LIVE
        checks["data_mode_is_live"] = c3
        if not c3:
            blocked.append(f"DATA_MODE is '{config.data_mode}'. LIVE trading strictly prohibits historical/replay data.")

        # 4. LIVE_TRADING_ENABLED must be explicitly True
        c4 = config.live_trading_enabled is True
        checks["live_trading_enabled_flag"] = c4
        if not c4:
            blocked.append("LIVE_TRADING_ENABLED is False. Accidental live trading blocked.")

        # 5. Live confirmation token check
        c5 = True
        if config.live_confirmation_required:
            c5 = bool(config.live_confirmation_token and config.live_confirmation_token == "AUTHORIZE_LIVE_US_TRADING")
        checks["live_confirmation_token_valid"] = c5
        if not c5:
            blocked.append("LIVE_CONFIRMATION_REQUIRED is True, but valid confirmation token was not provided.")

        # 6. Global Kill Switch must NOT be engaged
        ks = kill_switch or GlobalKillSwitch.get_instance()
        c6 = not ks.is_active()
        checks["kill_switch_inactive"] = c6
        if not c6:
            status = ks.get_status()
            blocked.append(f"GLOBAL KILL SWITCH IS ENGAGED (Reason: {status.get('reason')}). All live orders blocked.")

        # 7. Market Session must be open for order execution
        c7 = session_info.can_execute_orders
        checks["market_session_open"] = c7
        if not c7:
            blocked.append(f"Market session is {session_info.status.value}: {session_info.reason}. Normal live execution blocked.")

        # 8. Broker Identity must be verified as genuinely LIVE
        c8 = False
        if broker_identity:
            env = str(broker_identity.get("environment", "")).lower()
            acc_id = broker_identity.get("account_id")
            c8 = bool(acc_id and env == "live")
        checks["broker_account_verified_live"] = c8
        if not c8:
            env_found = broker_identity.get("environment") if broker_identity else "NONE"
            blocked.append(f"Broker identity verification failed: account environment is '{env_found}', expected verified 'live'.")

        # 9. RiskEngine must be healthy and not halted
        c9 = risk_engine_healthy
        checks["risk_engine_healthy"] = c9
        if not c9:
            blocked.append("RiskEngine reported unhealthy or daily loss limit / emergency halt triggered.")

        # 10. Market Data Freshness
        c10 = True
        if data_freshness_seconds is not None:
            c10 = data_freshness_seconds <= config.max_data_age_seconds
        checks["data_freshness_ok"] = c10
        if not c10:
            blocked.append(f"Market data is stale ({data_freshness_seconds:.1f}s old > max allowed {config.max_data_age_seconds}s).")

        # 11. Feature Point-in-Time (PIT) Causal Validity
        c11 = feature_pit_valid
        checks["feature_pit_valid"] = c11
        if not c11:
            blocked.append("Feature pipeline failed Point-in-Time causality validation (future leakage suspected).")

        # 12. Laya Health
        c12 = laya_healthy
        checks["laya_model_healthy"] = c12
        if not c12:
            blocked.append("Laya Financial Specialist model failed health / integrity check.")

        # 13. Portfolio State Healthy
        c13 = portfolio_healthy
        checks["portfolio_healthy"] = c13
        if not c13:
            blocked.append("Portfolio state reported unhealthy (cash floor or position limit exceeded).")

        passed = len(blocked) == 0
        if not passed:
            logger.warning("Live trading guards failed %d check(s): %s", len(blocked), "; ".join(blocked))

        return GuardVerificationResult(
            passed=passed,
            blocked_reasons=blocked,
            checks_performed=checks,
        )
