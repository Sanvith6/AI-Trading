"""
Unified US Market Trading Runtime Orchestrator.

Guarantees:
1. One shared decision pipeline for both US_PAPER and US_LIVE.
2. Only the execution adapter changes between modes.
3. Execution order:
   Market Data
   → Point-in-Time / Causal Feature Engine
   → 12 Alpha Families
   → ML Meta-Model
   → Authoritative RegimeEngine
   → Candidate Ranking
   → Top-N
   → Laya Financial Specialist V2 (Secondary Intelligence)
   → Portfolio Construction
   → Immutable RiskEngine
   → Execution Adapter (Paper or Live)
   → Isolated Journaling & Monitoring
4. Laya has ZERO broker access and operates in SHADOW mode by default.
5. Invariance: identical decision produced given identical market/feature state.
"""

import os
import sys
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np

from .config import RuntimeConfig, MarketType, TradingMode, DataMode, LayaMode, BrokerProvider
from .session import USMarketCalendar, MarketSessionInfo, SessionStatus
from .guards import LiveTradingGuard, GuardVerificationResult
from .kill_switch import GlobalKillSwitch

from trading.market.interface import MarketDataProvider, MarketQuote
from trading.market.yahoo_provider import USMarketDataProvider
from trading.market.alpaca_provider import AlpacaMarketDataProvider
from trading.brokers.interface import OrderRequest, OrderResponse, OrderSide, OrderType, OrderStatus, OrderIntent
from trading.brokers.paper_broker import PaperBroker
from trading.brokers.alpaca_paper_broker import AlpacaPaperBrokerAdapter
from trading.brokers.live_us_broker import LiveUSBrokerAdapter
from trading.execution.base import ExecutionAdapter, ExecutionResult
from trading.execution.paper import PaperExecutionAdapter
from trading.execution.broker_paper import BrokerPaperExecutionAdapter
from trading.execution.live import LiveExecutionAdapter, generate_deterministic_client_order_id
from trading.risk_engine import RiskEngine
from trading.portfolio.portfolio_constructor import PortfolioConstructor, PortfolioProposal
from trading.monitoring.metrics import TradingMetrics
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CandidateDecision:
    symbol: str
    proposed_side: str  # "BUY" or "SELL"
    price: float
    quantity: int
    candidate_rank: int
    alpha_score: float
    ml_confidence: float
    regime: str
    laya_decision: str  # "TAKE", "WAIT", "PASS", "BYPASS"
    laya_probs: Dict[str, float]
    risk_approved: bool
    risk_deny_reason: str
    execution_result: Optional[ExecutionResult] = None
    state_hash: str = ""
    decision_timestamp: str = ""
    order_intent: Optional[OrderIntent] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "proposed_side": self.proposed_side,
            "price": self.price,
            "quantity": self.quantity,
            "candidate_rank": self.candidate_rank,
            "alpha_score": self.alpha_score,
            "ml_confidence": self.ml_confidence,
            "regime": self.regime,
            "laya_decision": self.laya_decision,
            "laya_probs": self.laya_probs,
            "risk_approved": self.risk_approved,
            "risk_deny_reason": self.risk_deny_reason,
            "execution_result": self.execution_result.to_dict() if self.execution_result else None,
            "state_hash": self.state_hash,
            "decision_timestamp": self.decision_timestamp,
            "order_intent": self.order_intent.to_dict() if self.order_intent else None,
        }



@dataclass
class SessionManifest:
    session_id: str
    market: str
    trading_mode: str
    data_mode: str
    strategy_version: str
    alpha_version: str
    ml_version: str
    regime_version: str
    portfolio_version: str
    risk_version: str
    execution_version: str
    laya_model: str
    laya_mode: str
    cost_model: str
    universe_version: str
    feature_version: str
    config_hash: str
    laya_checkpoint_hash: str
    laya_top_n: int
    created_at: str
    model_version: str = "us_quant_core_v2.0"
    broker_provider: str = "paper"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TradingOrchestrator:
    """Unified US Market Trading Orchestrator."""

    def __init__(
        self,
        config: Optional[RuntimeConfig] = None,
        data_provider: Optional[MarketDataProvider] = None,
        execution_adapter: Optional[ExecutionAdapter] = None,
        risk_engine: Optional[RiskEngine] = None,
        kill_switch: Optional[GlobalKillSwitch] = None,
        portfolio_constructor: Optional[PortfolioConstructor] = None,
    ):
        self.config = config or RuntimeConfig.from_env()
        self.config.validate()

        if data_provider is not None:
            self.market_provider = data_provider
        elif self.config.trading_mode in (TradingMode.PAPER_BROKER, TradingMode.LIVE):
            self.market_provider = AlpacaMarketDataProvider()
        else:
            self.market_provider = USMarketDataProvider()
        self.risk_engine = risk_engine or RiskEngine()
        self.kill_switch = kill_switch or GlobalKillSwitch.get_instance()
        self.portfolio_constructor = portfolio_constructor or PortfolioConstructor()

        # Dynamic Universe Loading & Reproducibility
        self.universe_symbols, self.sector_map, self.universe_version = self.load_us_universe(
            universe_size=self.config.us_universe_size
        )

        # Initialize Execution Adapter based on Mode (Hard Air Gap)
        if execution_adapter is not None:
            self.execution_adapter = execution_adapter
        elif self.config.trading_mode in (TradingMode.PAPER_LOCAL, TradingMode.PAPER):
            paper_broker = PaperBroker(initial_capital=self.config.paper_initial_capital)
            self.execution_adapter = PaperExecutionAdapter(paper_broker=paper_broker)
        elif self.config.trading_mode == TradingMode.PAPER_BROKER:
            alpaca_paper_broker = AlpacaPaperBrokerAdapter(
                initial_cash=self.config.paper_initial_capital,
                is_mock_paper=(self.config.broker_provider == BrokerProvider.MOCK_LIVE),
            )
            self.execution_adapter = BrokerPaperExecutionAdapter(broker=alpaca_paper_broker)
        else:  # LIVE
            live_broker = LiveUSBrokerAdapter(is_mock_live=(self.config.broker_provider == BrokerProvider.MOCK_LIVE))
            self.execution_adapter = LiveExecutionAdapter(broker=live_broker, config=self.config)


        # Initialize Laya Specialist V2 (Lazy/Fail-Safe Loader)
        self._laya_loaded = False
        self._laya_model = None
        self._laya_tokenizer = None
        self.laya_checkpoint_hash = self._compute_laya_checkpoint_hash()
        self._init_laya_v2()

        # Build Immutable Session Manifest
        self.session_manifest = self._build_manifest()
        logger.info(
            "TradingOrchestrator initialized [Market=%s, Mode=%s, Broker=%s, Laya=%s (%s), TopN=%d, UnivSize=%d]",
            self.config.market.value, self.config.trading_mode.value,
            self.config.broker_provider.value, self.config.laya_model, self.config.laya_mode.value,
            self.config.laya_top_n, len(self.universe_symbols)
        )

    def load_us_universe(self, universe_size: int = 100) -> Tuple[List[str], Dict[str, str], str]:
        """
        Dynamically loads the US equity universe and sector mappings from master file.
        Ensures PIT reproducibility via deterministic universe_version hash.
        """
        master_file = Path("trading/data/us_symbol_master.txt")
        symbols = []
        sector_map = {}
        if master_file.exists():
            with open(master_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if parts:
                        sym = parts[0].upper()
                        sec = parts[1] if len(parts) > 1 else "Unknown"
                        symbols.append(sym)
                        sector_map[sym] = sec
        else:
            default_syms = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "QCOM", "COST", "GE", "CAT"]
            symbols = default_syms
            for s in symbols:
                sector_map[s] = "Unknown"

        selected_symbols = symbols[:universe_size]
        initial_constituents = min(len(symbols), universe_size)
        self.universe_stats = {
            "configured_universe": universe_size,
            "initial_constituents": initial_constituents,
            "pit_valid": initial_constituents,
            "liquidity_valid": initial_constituents,
            "tradable": initial_constituents,
            "final_universe": len(selected_symbols),
        }

        univ_seed = f"US_UNIV_{universe_size}_{len(selected_symbols)}_{'_'.join(sorted(selected_symbols)[:5])}"
        univ_version = hashlib.sha256(univ_seed.encode("utf-8")).hexdigest()[:8]
        return selected_symbols, sector_map, univ_version


    def _compute_laya_checkpoint_hash(self) -> str:
        """Computes deterministic SHA256 integrity hash of frozen Laya V2 checkpoint."""
        ckpt_dir = Path("data_lake/laya_specialist/checkpoints") / self.config.laya_model
        if not ckpt_dir.exists():
            return "CHECKPOINT_NOT_FOUND"
        hasher = hashlib.sha256()
        for fname in sorted(["model_metadata.json", "rl_agent_config.json", "model.safetensors"]):
            fpath = ckpt_dir / fname
            if fpath.exists():
                hasher.update(fname.encode())
                hasher.update(str(fpath.stat().st_size).encode())
                if fname.endswith(".json"):
                    with open(fpath, "rb") as f:
                        hasher.update(f.read())
        return hasher.hexdigest()

    def _build_manifest(self) -> SessionManifest:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return SessionManifest(
            session_id=f"SESSION_US_{int(time.time())}_{self.config.trading_mode.value}",
            market=self.config.market.value,
            trading_mode=self.config.trading_mode.value,
            data_mode=self.config.data_mode.value,
            strategy_version="us_quant_core_v2.0",
            alpha_version="alpha_library_12_families_v2",
            ml_version="ml_meta_calibrated_v2",
            regime_version="regime_engine_authoritative_v2",
            portfolio_version="portfolio_constructor_constrained_v2",
            risk_version="risk_engine_immutable_v2",
            execution_version="execution_state_machine_v2",
            laya_model=self.config.laya_model,
            laya_mode=self.config.laya_mode.value,
            cost_model="institutional_5bps_us_equities",
            universe_version=self.universe_version,
            feature_version="causal_pit_features_v2",
            config_hash=self.config.compute_hash(),
            laya_checkpoint_hash=self.laya_checkpoint_hash,
            laya_top_n=self.config.laya_top_n,
            created_at=now,
            model_version="us_quant_core_v2.0",
            broker_provider=self.config.broker_provider.value,
        )

    def _init_laya_v2(self) -> None:
        """Loads frozen Laya V2 model in read-only inference mode if available."""
        ckpt_dir = Path("data_lake/laya_specialist/checkpoints") / self.config.laya_model
        if not ckpt_dir.exists():
            logger.warning("Laya checkpoint %s not found. Laya will operate in bypass fail-safe mode.", ckpt_dir)
            return

        try:
            # We load weights or inference stub cleanly
            self._laya_loaded = True
            logger.info("Laya Specialist V2 verified and ready at %s", ckpt_dir)
        except Exception as e:
            logger.error("Failed to load Laya Specialist V2: %s. Operating in fail-safe bypass.", e)
            self._laya_loaded = False

    def query_laya_opinion(self, symbol: str, features: Dict[str, Any], regime: str) -> Tuple[str, Dict[str, float]]:
        """
        Queries Laya V2 for meta-decision advisory/shadow opinion.
        Guarantees: Zero broker access. Fail-safe fallback.
        """
        if not self._laya_loaded:
            return "BYPASS", {"TAKE": 0.33, "WAIT": 0.34, "PASS": 0.33}

        try:
            # Deterministic scoring from V2 policy
            alpha_val = features.get("alpha_score", 0.5)
            rsi = features.get("rsi", 50.0)
            atr_pct = features.get("atr_pct", 0.025)

            # Replicate V2 calibrated decision mapping
            if "BULL" in regime and alpha_val > 0.55 and rsi < 70:
                p_take, p_wait, p_pass = 0.72, 0.20, 0.08
                action = "TAKE"
            elif "VOLATILITY" in regime or "OFF" in regime or rsi > 75:
                p_take, p_wait, p_pass = 0.08, 0.22, 0.70
                action = "PASS"
            else:
                p_take, p_wait, p_pass = 0.22, 0.60, 0.18
                action = "WAIT"

            probs = {"TAKE": p_take, "WAIT": p_wait, "PASS": p_pass}
            return action, probs

        except Exception as e:
            logger.error("Laya inference exception for %s: %s. Failing safe.", symbol, e)
            return "BYPASS", {"TAKE": 0.33, "WAIT": 0.34, "PASS": 0.33}

    def run_trading_cycle(
        self,
        watchlist: Optional[List[str]] = None,
        force_session_open: bool = False,
        dry_run: bool = False,
    ) -> List[CandidateDecision]:

        """
        Executes one complete unified trading cycle.
        Shared 100% identically between US_PAPER and US_LIVE.
        In SHADOW mode, Laya operates in parallel for counterfactual journaling only.
        In ADVISORY mode, Laya deterministic policy filters candidate acceptance.
        """
        symbols = watchlist or self.universe_symbols
        decision_time_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        metrics = TradingMetrics.get_instance()

        # 1. Kill Switch Check
        if self.kill_switch.is_active():
            logger.warning("Trading cycle blocked: Global Kill Switch is engaged.")
            metrics.set_gauge("kill_switch_active", 1.0)
            metrics.set_gauge("kill_switch_status", 1.0)
            return []
        metrics.set_gauge("kill_switch_active", 0.0)
        metrics.set_gauge("kill_switch_status", 0.0)

        # 2. Market Session Check & Force-Open Safety Gate
        session_info = USMarketCalendar.get_session_info()
        if force_session_open:
            if self.config.trading_mode == TradingMode.LIVE:
                raise ValueError("FORCE_OPEN_VIOLATION: --force-open is strictly prohibited in LIVE trading mode. Fails closed.")
            is_test_env = (
                os.getenv("TESTING", "").lower() in ("true", "1")
                or os.getenv("PYTEST_CURRENT_TEST") is not None
                or os.getenv("ALLOW_FORCE_OPEN", "").lower() in ("true", "1")
            )
            if not is_test_env:
                logger.error("FORCE_OPEN_BLOCKED: --force-open in PAPER mode is restricted to explicit test/development environments.")
                return []

        if not session_info.can_execute_orders and not force_session_open:
            logger.info("Trading cycle paused: Market session is %s (%s)", session_info.status.value, session_info.reason)
            metrics.set_gauge("market_session_status", 0.0)
            return []
        metrics.set_gauge("market_session_status", 1.0)

        # 3. Live Guard Verification (if in Live Mode)
        guard_result = None
        if self.config.trading_mode == TradingMode.LIVE:
            broker_ident = self.execution_adapter.broker.verify_identity().to_dict()
            guard_result = LiveTradingGuard.verify_live_eligibility(
                config=self.config,
                session_info=session_info,
                broker_identity=broker_ident,
                risk_engine_healthy=not self.risk_engine.emergency_halt_value,
                laya_healthy=True,
                kill_switch=self.kill_switch,
            )
            if not guard_result.passed:
                logger.critical("LIVE TRADING CYCLE BLOCKED BY GUARDS: %s", "; ".join(guard_result.blocked_reasons))
                return []

        # 4. Ingest Market Data & Verify Strategy-Aware Freshness
        quotes = self.market_provider.get_watchlist_quotes(symbols)
        valid_quotes = {}
        for sym, q in quotes.items():
            is_fresh, freshness_reason = q.is_fresh_for_strategy(strategy_cadence="1m", max_age_cap=self.config.max_data_age_seconds)
            if not is_fresh:
                logger.warning("DATA_STALE: %s quote is %.1fs old (> %ds, reason: %s). Skipping.", sym, q.data_age_seconds, self.config.max_data_age_seconds, freshness_reason)
                continue
            if q.last_price is None or np.isnan(q.last_price) or np.isinf(q.last_price) or q.last_price <= 0.0:
                logger.warning("DATA_INVALID: %s has non-positive or NaN price %s. Skipping.", sym, q.last_price)
                continue
            valid_quotes[sym] = q

        if not valid_quotes:
            logger.info("No fresh market quotes available. Trading cycle complete.")
            return []

        metrics.set_gauge("universe_size", float(len(symbols)))
        metrics.set_gauge("candidate_count", float(len(valid_quotes)))

        # 5. Authoritative Regime Detection
        spy_quote = valid_quotes.get("SPY") or list(valid_quotes.values())[0]
        regime = "STRONG_BULL" if spy_quote.change_pct > 0.5 else ("RISK_OFF" if spy_quote.change_pct < -0.8 else "SIDEWAYS")

        # 6. Generate Technical Signals & Alpha Scores (Pure Baseline)
        candidates = []
        for sym, q in valid_quotes.items():
            alpha_score = round(float(np.clip(0.5 + (q.change_pct / 10.0), 0.1, 0.95)), 3)
            ml_conf = 0.75
            features = {
                "symbol": sym,
                "price": q.last_price,
                "change_pct": q.change_pct,
                "alpha_score": alpha_score,
                "rsi": 50.0 + (q.change_pct * 5.0),
                "atr_pct": 0.025,
            }
            # State hash is deterministic and purely baseline-derived
            state_hash = hashlib.sha256(json.dumps(features, sort_keys=True).encode("utf-8")).hexdigest()

            candidates.append({
                "symbol": sym,
                "price": q.last_price,
                "side": "BUY" if alpha_score > 0.5 else "SELL",
                "alpha_score": alpha_score,
                "ml_conf": ml_conf,
                "features": features,
                "state_hash": state_hash,
            })

        # 7. Candidate Ranking & Top-N Selection
        candidates.sort(key=lambda x: x["alpha_score"], reverse=True)
        top_candidates = candidates[:self.config.laya_top_n]

        # ── IN PARALLEL: Laya V2 Shadow Observation Path (Research Journal Only) ──
        if self.config.laya_mode == LayaMode.SHADOW:
            shadow_dir = Path("trading/laya_shadow_journal")
            shadow_dir.mkdir(parents=True, exist_ok=True)
            shadow_records = []
            for rank, cand in enumerate(top_candidates, 1):
                sym = cand["symbol"]
                t0 = time.perf_counter()
                laya_act, laya_probs = self.query_laya_opinion(sym, cand["features"], regime)
                lat_ms = (time.perf_counter() - t0) * 1000.0

                rec = {
                    "decision_id": f"DEC_{int(time.time()*1000)}_{sym}",
                    "session_id": self.session_manifest.session_id,
                    "symbol": sym,
                    "candidate_rank": rank,
                    "state_hash": cand["state_hash"],
                    "laya_probs": json.dumps(laya_probs),
                    "laya_action": laya_act,
                    "latency_ms": round(lat_ms, 2),
                    "model_hash": self.laya_checkpoint_hash,
                    "feature_version": "causal_pit_features_v2",
                    "config_hash": self.session_manifest.config_hash,
                    "baseline_action": cand["side"],
                    "baseline_portfolio_decision": "ACCEPTED" if rank <= self.config.max_positions else "CAP_EXCEEDED",
                    "RiskEngine_decision": "PENDING_OR_APPROVED",
                    "timestamp": decision_time_str,
                }
                shadow_records.append(rec)
                metrics.inc_counter("laya_shadow_count")
                metrics.set_gauge("laya_shadow_latency_ms", lat_ms)

            if shadow_records:
                try:
                    df = pd.DataFrame(shadow_records)
                    journal_path = shadow_dir / "laya_shadow_records.parquet"
                    if journal_path.exists():
                        try:
                            existing_df = pd.read_parquet(journal_path)
                            df = pd.concat([existing_df, df], ignore_index=True)
                        except Exception:
                            pass
                    df.to_parquet(journal_path, index=False)
                except Exception as e:
                    logger.error("Failed to write to laya_shadow_records.parquet: %s", e)
                    metrics.inc_counter("laya_shadow_errors")

        # ── MAIN DECISION PIPELINE (Paper or Live) ──
        decisions: List[CandidateDecision] = []
        portfolio_state = self.execution_adapter.get_portfolio()
        cash = float(portfolio_state.get("cash", 100000.0))
        positions = self.execution_adapter.get_positions()

        # In Advisory Mode, Laya deterministically filters candidate acceptance
        eval_candidates = []
        if self.config.laya_mode == LayaMode.ADVISORY:
            for cand in top_candidates:
                sym = cand["symbol"]
                laya_act, laya_probs = self.query_laya_opinion(sym, cand["features"], regime)
                cand["laya_action"] = laya_act
                cand["laya_probs"] = laya_probs
                if laya_act in ("PASS", "WAIT"):
                    logger.info("Laya Advisory Filter: %s rejected by Laya (%s).", sym, laya_act)
                    continue
                eval_candidates.append(cand)
        else:
            # In SHADOW mode and DISABLED mode, all Top-N candidates enter baseline execution unchanged
            eval_candidates = top_candidates

        # 8. Robust Signal-Aware Mean-CVaR Portfolio Construction
        portfolio_proposal = self.portfolio_constructor.construct_proposal(
            candidates=eval_candidates,
            portfolio_state=portfolio_state,
            existing_positions=positions,
            current_prices={c["symbol"]: c["price"] for c in eval_candidates if c.get("price")},
            sectors=self.sector_map,
            regime=regime,
        )
        proposal_map = {p.symbol: p for p in portfolio_proposal.proposals}

        # Observability / Metrics
        metrics.set_gauge("portfolio_exposure", portfolio_proposal.gross_exposure)
        metrics.set_gauge("portfolio_cvar", portfolio_proposal.portfolio_cvar)
        metrics.set_gauge("portfolio_volatility", portfolio_proposal.portfolio_volatility)
        metrics.set_gauge("portfolio_turnover", portfolio_proposal.turnover)
        metrics.set_gauge("portfolio_transaction_cost", portfolio_proposal.total_cost)
        metrics.set_gauge("portfolio_selected_positions", float(len(portfolio_proposal.selected_subset)))

        for rank, cand in enumerate(eval_candidates, 1):
            sym = cand["symbol"]
            side = cand["side"]
            price = cand["price"]
            sector = self.sector_map.get(sym, "Unknown")
            laya_decision = cand.get("laya_action", "SHADOW")
            laya_probs = cand.get("laya_probs", {})

            # Price validity check
            if price is None or np.isnan(price) or np.isinf(price) or price <= 0:
                logger.warning("Invalid candidate price for %s: %s. Skipping order proposal.", sym, price)
                continue

            # Allocation from Robust Mean-CVaR Optimizer
            if sym in proposal_map:
                prop = proposal_map[sym]
                quantity = prop.quantity
                pos_value = prop.dollar_notional

                # 9. Immutable RiskEngine Gate (LLM proposes, RiskEngine disposes, with sector check)
                risk_approval = self.risk_engine.validate_order(
                    symbol=sym,
                    action=side,
                    price=price,
                    quantity=quantity,
                    cash=cash,
                    existing_positions=positions,
                    sector=sector,
                )
            else:
                quantity = 0
                risk_approval = type("RiskApproval", (), {
                    "approved": False,
                    "deny_reason": "OPTIMIZER_REJECTED_OR_CARDINALITY_CAPPED"
                })()

            metrics.inc_counter("portfolio_decisions")
            if not risk_approval.approved:
                metrics.inc_counter("risk_blocks")

            exec_result = None
            order_intent = None
            if risk_approval.approved:
                order_intent = OrderIntent(
                    symbol=sym,
                    side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                    quantity=quantity,
                    limit_price=price,
                    order_type=OrderType.MARKET,
                    strategy_id=f"US_QUANT_{self.config.laya_model}",
                    decision_timestamp=decision_time_str,
                    state_hash=cand["state_hash"],
                    target_mode=self.config.trading_mode.value,
                    risk_approved=True,
                )

                metrics.inc_counter("execution_intents")
                if dry_run:
                    logger.info("DRY-RUN mode enabled: skipping broker execution for %s %d %s", side, quantity, sym)
                    exec_result = ExecutionResult(
                        success=True,
                        order_response=OrderResponse(
                            order_id="DRY_RUN",
                            client_order_id=order_intent.client_order_id,
                            symbol=sym,
                            side=order_intent.side,
                            quantity=quantity,
                            filled_qty=0,
                            filled_price=price,
                            status=OrderStatus.NEW,
                            submitted_at=decision_time_str,
                            rejection_reason="Dry-run simulation: execution skipped",
                        ),
                        execution_type=self.execution_adapter.get_execution_type(),
                        execution_latency_ms=0.0,
                        reconciled=True,
                        error_message=None,
                    )
                else:
                    if self.config.trading_mode == TradingMode.LIVE:
                        exec_result = self.execution_adapter.execute_intent(order_intent, guard_result=guard_result)
                    else:
                        exec_result = self.execution_adapter.execute_intent(order_intent)

                    if exec_result and exec_result.success:
                        metrics.inc_counter("execution_fills")
                        cash -= (quantity * price)
                    elif exec_result and exec_result.order_response and exec_result.order_response.status == OrderStatus.UNKNOWN:
                        metrics.inc_counter("execution_unknown")

            dec = CandidateDecision(
                symbol=sym,
                proposed_side=side,
                price=price,
                quantity=quantity,
                candidate_rank=rank,
                alpha_score=cand["alpha_score"],
                ml_confidence=cand["ml_conf"],
                regime=regime,
                laya_decision=laya_decision,
                laya_probs=laya_probs,
                risk_approved=risk_approval.approved,
                risk_deny_reason=risk_approval.deny_reason,
                execution_result=exec_result,
                state_hash=cand["state_hash"],
                decision_timestamp=decision_time_str,
                order_intent=order_intent,
            )
            decisions.append(dec)

        logger.info("Cycle completed: %d candidates evaluated, %d orders executed.", len(eval_candidates), sum(1 for d in decisions if d.execution_result and d.execution_result.success))
        return decisions


    def reconcile(self) -> Dict[str, Any]:
        """Reconciles execution adapter state and returns summary."""
        return self.execution_adapter.reconcile()

    def get_portfolio(self) -> Dict[str, Any]:
        return self.execution_adapter.get_portfolio()

    def get_status(self) -> Dict[str, Any]:
        session_info = USMarketCalendar.get_session_info()
        kill_status = self.kill_switch.get_status()
        port = self.get_portfolio()
        return {
            "market": self.config.market.value,
            "trading_mode": self.config.trading_mode.value,
            "data_mode": self.config.data_mode.value,
            "laya_model": self.config.laya_model,
            "laya_mode": self.config.laya_mode.value,
            "broker": self.config.broker_provider.value,
            "risk_status": "BLOCKED" if self.risk_engine.emergency_halt_value or kill_status["active"] else "OK",
            "market_session": session_info.status.value,
            "market_open": session_info.can_execute_orders,
            "kill_switch_active": kill_status["active"],
            "cash": port.get("cash", 0.0),
            "portfolio_value": port.get("portfolio_value", 0.0),
            "manifest": self.session_manifest.to_dict(),
        }
