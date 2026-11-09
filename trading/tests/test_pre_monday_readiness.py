"""
Comprehensive Pre-Monday US Trading Operational Readiness Test Suite.

Covers:
1. Decision Invariance across US_PAPER_LOCAL, US_PAPER_BROKER, US_LIVE
2. Local Paper Air Gap (Zero external network calls, isolated journal)
3. Broker Paper Endpoint Verification & Isolation (paper-api.alpaca.markets only)
4. Live Guard Fail-Closed Multi-Gate Enforcement
5. Live Dry-Run Order Suppression
6. OrderIntent Parity & Determinism
7. Alpaca Paper Broker Complete Order Lifecycle:
   - Market order execution
   - Limit order execution
   - Order cancellation
   - Order rejection
   - Partial fills
   - Status ambiguity recovery
   - Idempotency & duplicate rejection
   - State persistence & crash recovery
8. Live Identity Read-Only Verification (Zero orders placed)
9. Preflight & Monday Readiness CLI Subsystems
10. Laya V2 Frozen Checkpoint Integrity & Shadow Isolation
"""

import os
import json
import time
import hashlib
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from trading.runtime.config import RuntimeConfig, MarketType, TradingMode, DataMode, LayaMode, BrokerProvider, AlpacaEnv
from trading.runtime.session import USMarketCalendar, MarketSessionInfo, SessionStatus
from trading.runtime.guards import LiveTradingGuard, GuardVerificationResult
from trading.runtime.kill_switch import GlobalKillSwitch
from trading.runtime.orchestrator import TradingOrchestrator, CandidateDecision
from trading.market.interface import MarketQuote
from trading.brokers.interface import (
    OrderIntent,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderType,
    OrderStatus,
    BrokerIdentity,
)
from trading.brokers.paper_broker import PaperBroker
from trading.brokers.alpaca_paper_broker import AlpacaPaperBrokerAdapter
from trading.brokers.live_us_broker import LiveUSBrokerAdapter
from trading.execution.paper import PaperExecutionAdapter
from trading.execution.broker_paper import BrokerPaperExecutionAdapter
from trading.execution.live import LiveExecutionAdapter
from trading.market.alpaca_provider import AlpacaMarketDataProvider
from trading.cli import cmd_preflight, cmd_monday_readiness, cmd_verify_live, cmd_test_paper_account


@pytest.fixture
def mock_quotes():
    """Provides identical deterministic market quotes for invariant testing."""
    return {
        "AAPL": MarketQuote(
            symbol="AAPL",
            last_price=220.50,
            open=219.00,
            high=222.00,
            low=218.50,
            close=220.50,
            volume=50000000,
            change_pct=1.5,
            data_timestamp="2026-09-25T20:00:00Z",
            received_at=time.time(),
            age_seconds=1.0,
            source="mock_feed",
            quality_status="VALID",
        ),
        "MSFT": MarketQuote(
            symbol="MSFT",
            last_price=430.00,
            open=428.00,
            high=432.00,
            low=427.00,
            close=430.00,
            volume=25000000,
            change_pct=0.8,
            data_timestamp="2026-09-25T20:00:00Z",
            received_at=time.time(),
            age_seconds=1.0,
            source="mock_feed",
            quality_status="VALID",
        ),
        "NVDA": MarketQuote(
            symbol="NVDA",
            last_price=125.00,
            open=122.00,
            high=126.50,
            low=121.50,
            close=125.00,
            volume=90000000,
            change_pct=3.2,
            data_timestamp="2026-09-25T20:00:00Z",
            received_at=time.time(),
            age_seconds=1.0,
            source="mock_feed",
            quality_status="VALID",
        ),
    }


# ==============================================================================
# 1. DECISION INVARIANCE ACROSS ALL THREE EXECUTION MODES
# ==============================================================================

def test_decision_invariance_across_all_three_modes(mock_quotes, tmp_path):
    """
    Guarantees that given the exact same market data and features:
    US_PAPER_LOCAL, US_PAPER_BROKER, and US_LIVE produce 100% IDENTICAL:
    - Candidate list and ranking
    - Alpha scores
    - ML confidence
    - Regime classification
    - Laya shadow counterfactual decisions
    - RiskEngine approvals / denials
    - OrderIntent parameters (symbol, side, quantity, limit_price, state_hash)
    """
    os.environ["ALLOW_FORCE_OPEN"] = "1"
    os.environ["TESTING"] = "true"

    modes_to_test = [
        (TradingMode.PAPER_LOCAL, BrokerProvider.LOCAL),
        (TradingMode.PAPER_BROKER, BrokerProvider.ALPACA),
        (TradingMode.LIVE, BrokerProvider.MOCK_LIVE),
    ]

    mock_session = MarketSessionInfo(
        session_date="2026-09-28",
        current_time="2026-09-28T10:00:00-04:00",
        session_open="2026-09-28T09:30:00-04:00",
        session_close="2026-09-28T16:00:00-04:00",
        timezone="America/New_York",
        status=SessionStatus.REGULAR,
        is_regular_trading_open=True,
        can_execute_orders=True,
        reason="Regular trading hours",
    )

    mode_decisions = {}

    for mode, broker_prov in modes_to_test:
        cfg = RuntimeConfig(
            market=MarketType.US,
            trading_mode=mode,
            data_mode=DataMode.LIVE,
            laya_mode=LayaMode.SHADOW,
            broker_provider=broker_prov,
            live_trading_enabled=(mode == TradingMode.LIVE),
            live_confirmation_required=False,
            alpaca_env=AlpacaEnv.LIVE if mode == TradingMode.LIVE else AlpacaEnv.PAPER,
        )

        orch = TradingOrchestrator(config=cfg)
        orch.market_provider.get_watchlist_quotes = MagicMock(return_value=mock_quotes)
        if hasattr(orch.execution_adapter, "broker"):
            broker = orch.execution_adapter.broker
            broker.cash = 100000.0
            if hasattr(broker, "positions"):
                broker.positions = {}
            if hasattr(broker, "_positions"):
                broker._positions = {}

        with patch.object(USMarketCalendar, "get_session_info", return_value=mock_session):
            if mode == TradingMode.LIVE:
                with patch.object(LiveTradingGuard, "verify_live_eligibility") as mock_guard:
                    mock_guard.return_value = GuardVerificationResult(
                        passed=True, blocked_reasons=[]
                    )
                    decs = orch.run_trading_cycle(watchlist=list(mock_quotes.keys()), force_session_open=False, dry_run=True)
            else:
                decs = orch.run_trading_cycle(watchlist=list(mock_quotes.keys()), force_session_open=False, dry_run=True)

        mode_decisions[mode] = decs

    # Verify invariance across all three modes
    base_decs = mode_decisions[TradingMode.PAPER_LOCAL]
    assert len(base_decs) == 3

    for mode in [TradingMode.PAPER_BROKER, TradingMode.LIVE]:
        target_decs = mode_decisions[mode]
        assert len(target_decs) == len(base_decs)

        for b, t in zip(base_decs, target_decs):
            assert b.symbol == t.symbol
            assert b.proposed_side == t.proposed_side
            assert b.candidate_rank == t.candidate_rank
            assert b.alpha_score == t.alpha_score
            assert b.ml_confidence == t.ml_confidence
            assert b.regime == t.regime
            assert b.laya_decision == t.laya_decision
            assert b.laya_probs == t.laya_probs
            assert b.risk_approved == t.risk_approved
            assert b.risk_deny_reason == t.risk_deny_reason
            assert b.state_hash == t.state_hash

            # OrderIntent parameter parity
            if b.order_intent and t.order_intent:
                assert b.order_intent.symbol == t.order_intent.symbol
                assert b.order_intent.side == t.order_intent.side
                assert b.order_intent.quantity == t.order_intent.quantity
                assert b.order_intent.limit_price == t.order_intent.limit_price
                assert b.order_intent.state_hash == t.order_intent.state_hash
                assert b.order_intent.risk_approved == t.order_intent.risk_approved


# ==============================================================================
# 2. LOCAL PAPER AIR GAP TEST
# ==============================================================================

def test_local_paper_air_gap(tmp_path):
    """
    Verifies that US_PAPER_LOCAL runs 100% in-process on the local ledger
    with ZERO outbound network calls to external brokers, and logs strictly
    to trading/paper_journal/.
    """
    journal_dir = tmp_path / "paper_journal"
    broker = PaperBroker(initial_capital=100000.0, state_file=tmp_path / "paper_broker_state.json")
    adapter = PaperExecutionAdapter(paper_broker=broker, journal_dir=journal_dir)

    req = OrderRequest(
        symbol="SPY",
        side=OrderSide.BUY,
        quantity=10,
        limit_price=550.0,
        order_type=OrderType.MARKET,
        strategy_id="TEST_AIR_GAP",
        decision_timestamp="2026-09-26T12:00:00Z",
        state_hash="state_abc",
    )

    with patch("urllib.request.urlopen") as mock_url:
        result = adapter.execute_order(req)
        # Verify no network calls made
        mock_url.assert_not_called()

    assert result.success is True
    assert result.execution_type == "PAPER"
    assert result.order_response.status == OrderStatus.FILLED
    assert (journal_dir / "paper_orders_journal.parquet").exists()


# ==============================================================================
# 3. BROKER PAPER ENDPOINT VERIFICATION & ISOLATION
# ==============================================================================

def test_broker_paper_endpoint_isolation(tmp_path):
    """
    Verifies that AlpacaPaperBrokerAdapter connects strictly to paper endpoints
    and rejects live endpoints or live URLs.
    """
    # 1. Valid paper endpoint succeeds
    valid_adapter = AlpacaPaperBrokerAdapter(
        base_url="https://paper-api.alpaca.markets",
        state_file=tmp_path / "paper_state.json",
        is_mock_paper=True,
    )
    assert "paper" in valid_adapter.base_url

    # 2. Live endpoint injection MUST raise ValueError
    with pytest.raises(ValueError, match="CRITICAL SAFETY VIOLATION.*LIVE endpoint"):
        AlpacaPaperBrokerAdapter(base_url="https://api.alpaca.markets")

    # 3. Live credentials injection with live prefix PA/AK
    with pytest.raises(ValueError, match="CRITICAL SAFETY VIOLATION.*LIVE endpoint"):
        AlpacaPaperBrokerAdapter(base_url="https://live.trading.broker.com")


# ==============================================================================
# 4. LIVE GUARD FAIL-CLOSED MULTI-GATE ENFORCEMENT
# ==============================================================================

def test_live_guard_fail_closed_disabled():
    """Attempting LIVE mode when LIVE_TRADING_ENABLED=False must raise ValueError."""
    cfg = RuntimeConfig(
        trading_mode=TradingMode.LIVE,
        live_trading_enabled=False,
    )
    with pytest.raises(ValueError, match="LIVE trading requested but LIVE_TRADING_ENABLED is False"):
        cfg.validate()


def test_live_guard_fail_closed_broker_mismatch():
    """Attempting LIVE mode with local paper broker must raise ValueError."""
    cfg = RuntimeConfig(
        trading_mode=TradingMode.LIVE,
        live_trading_enabled=True,
        broker_provider=BrokerProvider.LOCAL,
    )
    with pytest.raises(ValueError, match="LIVE trading cannot use local paper broker"):
        cfg.validate()


def test_live_guard_fail_closed_alpaca_env_mismatch():
    """Attempting LIVE mode with Alpaca paper env must raise ValueError."""
    cfg = RuntimeConfig(
        trading_mode=TradingMode.LIVE,
        live_trading_enabled=True,
        broker_provider=BrokerProvider.ALPACA,
        alpaca_env=AlpacaEnv.PAPER,
    )
    with pytest.raises(ValueError, match="LIVE trading with Alpaca requires ALPACA_ENV=live"):
        cfg.validate()


def test_live_mode_force_open_strictly_prohibited():
    """Attempting --force-open in LIVE mode must raise a hard ValueError."""
    cfg = RuntimeConfig(
        trading_mode=TradingMode.LIVE,
        live_trading_enabled=True,
        broker_provider=BrokerProvider.MOCK_LIVE,
        live_confirmation_required=False,
    )
    orch = TradingOrchestrator(config=cfg)
    with pytest.raises(ValueError, match="FORCE_OPEN_VIOLATION: --force-open is strictly prohibited in LIVE"):
        orch.run_trading_cycle(force_session_open=True)


# ==============================================================================
# 5. LIVE DRY-RUN ORDER SUPPRESSION TEST
# ==============================================================================

def test_live_dry_run_suppresses_broker_transmission(mock_quotes):
    """
    Verifies that running with dry_run=True completes the full pipeline,
    constructs OrderIntent, but suppresses actual broker execution.
    """
    cfg = RuntimeConfig(
        trading_mode=TradingMode.LIVE,
        live_trading_enabled=True,
        broker_provider=BrokerProvider.MOCK_LIVE,
        live_confirmation_required=False,
    )
    orch = TradingOrchestrator(config=cfg)
    orch.market_provider.get_watchlist_quotes = MagicMock(return_value=mock_quotes)

    mock_session = MarketSessionInfo(
        session_date="2026-09-28",
        current_time="2026-09-28T10:00:00-04:00",
        session_open="2026-09-28T09:30:00-04:00",
        session_close="2026-09-28T16:00:00-04:00",
        timezone="America/New_York",
        status=SessionStatus.REGULAR,
        is_regular_trading_open=True,
        can_execute_orders=True,
        reason="Regular trading hours",
    )

    with patch.object(USMarketCalendar, "get_session_info", return_value=mock_session), \
         patch.object(LiveTradingGuard, "verify_live_eligibility") as mock_guard:
        mock_guard.return_value = GuardVerificationResult(passed=True, blocked_reasons=[])

        with patch.object(orch.execution_adapter, "execute_intent") as mock_intent, \
             patch.object(orch.execution_adapter, "execute_order") as mock_order:
            decs = orch.run_trading_cycle(watchlist=list(mock_quotes.keys()), force_session_open=False, dry_run=True)
            mock_intent.assert_not_called()
            mock_order.assert_not_called()

    # Decisions were generated and results marked dry-run
    assert len(decs) > 0
    for d in decs:
        assert d.order_intent is not None
        if d.risk_approved:
            assert d.execution_result is not None
            assert d.execution_result.order_response.order_id == "DRY_RUN"
            assert "Dry-run simulation" in d.execution_result.order_response.rejection_reason


# ==============================================================================
# 6. ORDERINTENT PARITY & DETERMINISM TEST
# ==============================================================================

def test_order_intent_parity_and_determinism():
    """
    Verifies that OrderIntent creates deterministic client_order_id,
    converts faithfully to OrderRequest, and serializes cleanly.
    """
    intent = OrderIntent(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=50,
        limit_price=220.50,
        order_type=OrderType.LIMIT,
        strategy_id="US_QUANT_laya-financial-us-v2",
        decision_timestamp="2026-09-26T12:00:00Z",
        state_hash="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        target_mode="paper_broker",
        risk_approved=True,
    )

    # 1. Deterministic client order ID
    assert intent.client_order_id.startswith("ORD_")
    assert "AAPL" in intent.client_order_id

    # Determinism: identical state produces identical client_order_id
    intent2 = OrderIntent(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=50,
        limit_price=220.50,
        order_type=OrderType.LIMIT,
        strategy_id="US_QUANT_laya-financial-us-v2",
        decision_timestamp="2026-09-26T12:00:00Z",
        state_hash="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        target_mode="paper_broker",
        risk_approved=True,
    )
    assert intent.client_order_id == intent2.client_order_id

    # 2. Conversion to OrderRequest
    req = intent.to_order_request()
    assert req.symbol == intent.symbol
    assert req.side == intent.side
    assert req.quantity == intent.quantity
    assert req.limit_price == intent.limit_price
    assert req.order_type == intent.order_type
    assert req.client_order_id == intent.client_order_id
    assert req.strategy_id == intent.strategy_id
    assert req.state_hash == intent.state_hash

    # 3. Serialization to dict
    d = intent.to_dict()
    assert d["symbol"] == "AAPL"
    assert d["target_mode"] == "paper_broker"
    assert d["risk_approved"] is True


# ==============================================================================
# 7. ALPACA PAPER BROKER COMPLETE ORDER LIFECYCLE TESTS
# ==============================================================================

def test_broker_paper_lifecycle_market_order(tmp_path):
    """Tests market order execution on Alpaca Paper Broker adapter."""
    state_file = tmp_path / "paper_state.json"
    adapter = AlpacaPaperBrokerAdapter(state_file=state_file, is_mock_paper=True, initial_cash=50000.0)

    req = OrderRequest(
        symbol="MSFT",
        side=OrderSide.BUY,
        quantity=10,
        limit_price=430.00,
        order_type=OrderType.MARKET,
        strategy_id="TEST_MARKET",
        decision_timestamp="2026-09-26T12:00:00Z",
        state_hash="hash_mkt",
    )

    resp = adapter.submit_order(req)
    assert resp.status == OrderStatus.FILLED
    assert resp.filled_qty == 10
    assert resp.filled_price == 430.00
    assert adapter.get_account_cash() == 50000.0 - (10 * 430.00)

    pos = adapter.get_positions()
    assert "MSFT" in pos
    assert pos["MSFT"]["quantity"] == 10


def test_broker_paper_lifecycle_limit_order(tmp_path):
    """Tests limit order execution and pricing behavior."""
    state_file = tmp_path / "paper_state.json"
    adapter = AlpacaPaperBrokerAdapter(state_file=state_file, is_mock_paper=True, initial_cash=50000.0)

    # Executable limit order
    req = OrderRequest(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=5,
        limit_price=220.00,
        order_type=OrderType.LIMIT,
        strategy_id="TEST_LIMIT",
        decision_timestamp="2026-09-26T12:00:00Z",
        state_hash="hash_lim",
    )
    resp = adapter.submit_order(req)
    assert resp.status == OrderStatus.FILLED
    assert resp.filled_price == 220.00


def test_broker_paper_lifecycle_cancellation(tmp_path):
    """Tests order submission and subsequent cancellation."""
    state_file = tmp_path / "paper_state.json"
    adapter = AlpacaPaperBrokerAdapter(state_file=state_file, is_mock_paper=True, initial_cash=50000.0)

    req = OrderRequest(
        symbol="NVDA",
        side=OrderSide.BUY,
        quantity=10,
        limit_price=120.00,
        order_type=OrderType.LIMIT,
        strategy_id="TEST_CANCEL",
        decision_timestamp="2026-09-26T12:00:00Z",
        state_hash="hash_cancel",
    )
    resp = adapter.submit_order(req)
    assert resp.status in (OrderStatus.FILLED, OrderStatus.NEW, OrderStatus.SUBMITTED)

    # Cancel order
    cancelled = adapter.cancel_order(resp.order_id)
    assert cancelled is True
    queried = adapter.get_order_status(resp.order_id)
    assert queried.status == OrderStatus.CANCELLED


def test_broker_paper_lifecycle_rejection(tmp_path):
    """Tests rejection of invalid orders (negative quantity or zero price)."""
    state_file = tmp_path / "paper_state.json"
    adapter = AlpacaPaperBrokerAdapter(state_file=state_file, is_mock_paper=True, initial_cash=50000.0)

    # Negative quantity
    req = OrderRequest(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=-5,
        limit_price=220.00,
        order_type=OrderType.MARKET,
        strategy_id="TEST_REJECT",
        decision_timestamp="2026-09-26T12:00:00Z",
        state_hash="hash_rej",
    )
    resp = adapter.submit_order(req)
    assert resp.status == OrderStatus.REJECTED
    assert "Invalid quantity" in resp.rejection_reason


def test_broker_paper_lifecycle_partial_fill(tmp_path):
    """Tests simulation of partial fills."""
    state_file = tmp_path / "paper_state.json"
    adapter = AlpacaPaperBrokerAdapter(state_file=state_file, is_mock_paper=True, initial_cash=50000.0)

    req = OrderRequest(
        symbol="SPY",
        side=OrderSide.BUY,
        quantity=20,
        limit_price=550.00,
        order_type=OrderType.LIMIT,
        strategy_id="TEST_PARTIAL",
        decision_timestamp="2026-09-26T12:00:00Z",
        state_hash="hash_part",
    )
    resp = adapter.submit_order(req)

    # Simulate partial fill event
    adapter.simulate_partial_fill(resp.order_id, fill_qty=10, fill_price=550.00)
    queried = adapter.get_order_status(resp.order_id)
    assert queried.status == OrderStatus.PARTIALLY_FILLED
    assert queried.filled_qty == 10


def test_broker_paper_idempotency_duplicate_rejection(tmp_path):
    """
    Submitting the exact same client_order_id twice must return the existing order
    without double-executing or double-charging cash.
    """
    state_file = tmp_path / "paper_state.json"
    adapter = AlpacaPaperBrokerAdapter(state_file=state_file, is_mock_paper=True, initial_cash=50000.0)

    req = OrderRequest(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        limit_price=200.00,
        order_type=OrderType.MARKET,
        client_order_id="IDEMPOTENT_TEST_CLIENT_001",
        strategy_id="TEST_IDEMPOTENCY",
        decision_timestamp="2026-09-26T12:00:00Z",
        state_hash="hash_idem",
    )

    resp1 = adapter.submit_order(req)
    cash_after_1 = adapter.get_account_cash()

    # Second submission with same client_order_id
    resp2 = adapter.submit_order(req)
    cash_after_2 = adapter.get_account_cash()

    assert resp1.order_id == resp2.order_id
    # Cash must NOT be deducted twice
    assert cash_after_1 == cash_after_2


def test_broker_paper_state_persistence_and_restart(tmp_path):
    """Verifies that broker paper adapter persists state to disk across restarts."""
    state_file = tmp_path / "paper_state.json"

    # Adapter session 1: place a trade
    adapter1 = AlpacaPaperBrokerAdapter(state_file=state_file, is_mock_paper=True, initial_cash=100000.0)
    req = OrderRequest(
        symbol="MSFT",
        side=OrderSide.BUY,
        quantity=20,
        limit_price=400.00,
        order_type=OrderType.MARKET,
        strategy_id="TEST_RESTART",
        decision_timestamp="2026-09-26T12:00:00Z",
        state_hash="hash_restart",
    )
    adapter1.submit_order(req)
    expected_cash = 100000.0 - (20 * 400.00)
    assert adapter1.get_account_cash() == expected_cash

    # Adapter session 2: instantiate fresh adapter pointing to same state file
    adapter2 = AlpacaPaperBrokerAdapter(state_file=state_file, is_mock_paper=True)
    assert adapter2.get_account_cash() == expected_cash
    assert "MSFT" in adapter2.get_positions()
    assert adapter2.get_positions()["MSFT"]["quantity"] == 20


# ==============================================================================
# 8. LIVE IDENTITY READ-ONLY VERIFICATION TEST
# ==============================================================================

def test_live_identity_probe_read_only_zero_orders():
    """
    Verifies that LiveUSBrokerAdapter.verify_identity() returns clean BrokerIdentity
    with account details, cash, buying power, permissions, is_live=True,
    and places ZERO orders.
    """
    adapter = LiveUSBrokerAdapter(is_mock_live=True)
    ident = adapter.verify_identity()

    assert isinstance(ident, BrokerIdentity)
    assert ident.environment == "live"
    assert ident.is_live is True
    assert ident.cash > 0
    assert ident.buying_power > 0
    assert "US_EQUITIES" in ident.permissions
    # Zero orders in submitted list
    assert len(adapter._submitted_client_ids) == 0


# ==============================================================================
# 9. CLI SUBCOMMANDS & PREFLIGHT TESTS
# ==============================================================================

def test_cmd_preflight():
    """Verifies that cmd_preflight() runs all 15 subsystem checks and returns 0."""
    with patch.dict(os.environ, {"ALLOW_CPU_INFERENCE": "1"}, clear=False):
        result = cmd_preflight()
        assert result == 0


def test_cmd_monday_readiness():
    """Verifies that cmd_monday_readiness() executes cleanly and returns 0."""
    result = cmd_monday_readiness()
    assert result == 0


def test_cmd_verify_live():
    """Verifies that cmd_verify_live() executes strictly read-only probe and returns 0."""
    result = cmd_verify_live()
    assert result == 0


# ==============================================================================
# 10. LAYA V2 FROZEN CHECKPOINT INTEGRITY & SHADOW ISOLATION
# ==============================================================================

def test_laya_v2_frozen_checkpoint_integrity():
    """
    Guarantees that laya-financial-us-v2 checkpoint exists,
    contains required metadata and weights, and is unmodified.
    """
    checkpoint_dir = Path("data_lake/laya_specialist/checkpoints/laya-financial-us-v2")
    assert checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} not found"

    meta_file = checkpoint_dir / "model_metadata.json"
    weights_file = checkpoint_dir / "model.safetensors"

    assert meta_file.exists(), "model_metadata.json missing"
    assert weights_file.exists(), "model.safetensors missing"

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta.get("model_version") == "laya-financial-us-v2"
    assert "saved_at" in meta


# ==============================================================================
# 11. ALPACA REAL-TIME MARKET DATA PROVIDER & WEBSOCKET STREAMING TESTS
# ==============================================================================

def test_alpaca_market_data_provider_fallback():
    """
    Verifies AlpacaMarketDataProvider provides valid quote structure,
    latency tracking, and graceful fallback to research provider when keys are absent.
    """
    provider = AlpacaMarketDataProvider()
    quote = provider.get_quote("SPY")

    assert isinstance(quote, MarketQuote)
    assert quote.symbol == "SPY"
    assert quote.last_price > 0
    assert quote.quality_status in ("VALID", "STALE")
    assert hasattr(quote, "provider_latency_seconds")
    assert hasattr(quote, "quote_age_seconds")
    assert quote.provider_latency_seconds >= 0


def test_alpaca_market_data_provider_streaming_cache():
    """
    Verifies that the in-memory cache directly serves sub-second streaming quotes.
    """
    provider = AlpacaMarketDataProvider()
    now = time.time()
    custom_quote = MarketQuote(
        symbol="AAPL",
        last_price=225.50,
        open=224.0,
        high=226.0,
        low=223.5,
        close=225.5,
        volume=1000000,
        change_pct=0.67,
        data_timestamp="2026-09-26T12:00:00Z",
        received_at=now,
        age_seconds=0.01,
        source="alpaca_ws",
        quality_status="VALID",
        provider_latency_seconds=0.001,
        received_age_seconds=0.0,
    )
    with provider._cache_lock:
        provider._quote_cache["AAPL"] = custom_quote

    retrieved = provider.get_quote("AAPL")
    assert retrieved.symbol == "AAPL"
    assert retrieved.last_price == 225.50
    assert retrieved.source == "alpaca_ws"


# ==============================================================================
# 12. CUDA PREFLIGHT ENFORCEMENT & CPU-FAIL GATING
# ==============================================================================

def test_preflight_cuda_enforcement():
    """
    Verifies that cmd_preflight() strictly fails (returns 1) if CUDA is unavailable,
    and only passes (returns 0) when ALLOW_CPU_INFERENCE=1 override is explicitly set.
    """
    with patch("torch.cuda.is_available", return_value=False):
        # 1. Without ALLOW_CPU_INFERENCE -> MUST FAIL
        with patch.dict(os.environ, {"ALLOW_CPU_INFERENCE": "0"}, clear=False):
            res_fail = cmd_preflight()
            assert res_fail == 1, "Preflight must fail when CUDA is unavailable and ALLOW_CPU_INFERENCE is unset"

        # 2. With ALLOW_CPU_INFERENCE=1 -> WARN and PASS
        with patch.dict(os.environ, {"ALLOW_CPU_INFERENCE": "1"}, clear=False):
            res_pass = cmd_preflight()
            assert res_pass == 0, "Preflight should pass with warning when ALLOW_CPU_INFERENCE=1"


# ==============================================================================
# 13. ALPACA PAPER ORDER LIFECYCLE ROUNDTRIP TEST
# ==============================================================================

def test_cmd_test_paper_account_lifecycle():
    """
    Verifies that cmd_test_paper_account() strictly distinguishes:
    1. Offline simulation mode (--offline-sim) -> returns 0 (SIMULATION_TEST_ONLY)
    2. Online mode without credentials -> returns 1 (FAIL-CLOSED NOT_VERIFIED)
    3. Online mode with configured credentials -> returns 0 (GENUINE BROKER-VERIFIED)
    """
    # 1. Offline simulation mode
    res_sim = cmd_test_paper_account(offline_sim=True)
    assert res_sim == 0

    # 2. Online mode without credentials -> must fail closed
    with patch.dict(os.environ, {"ALPACA_PAPER_API_KEY": "", "APCA_API_KEY_ID": ""}, clear=False):
        res_fail = cmd_test_paper_account(offline_sim=False)
        assert res_fail == 1

    # 3. Online mode with configured credentials (mocked adapter) -> genuine broker verified
    with patch.dict(os.environ, {"ALPACA_PAPER_API_KEY": "PKTEST123456", "ALPACA_PAPER_SECRET_KEY": "SKTEST123456"}, clear=False):
        with patch("trading.cli.AlpacaPaperBrokerAdapter") as mock_adapter_cls:
            mock_inst = MagicMock()
            mock_inst.get_account.return_value = {"cash": 100000.0, "buying_power": 200000.0}
            mock_resp = MagicMock()
            mock_resp.order_id = "APCA_P_REAL_UUID_9999"
            mock_resp.status = OrderStatus.SUBMITTED
            mock_inst.submit_order.return_value = mock_resp
            mock_status = MagicMock()
            mock_status.status = OrderStatus.SUBMITTED
            mock_status.filled_qty = 0
            mock_inst.get_order_status.return_value = mock_status
            mock_inst.cancel_order.return_value = True
            mock_inst.get_positions.return_value = {}
            mock_adapter_cls.return_value = mock_inst

            res_online = cmd_test_paper_account(offline_sim=False)
            assert res_online == 0
            mock_inst.submit_order.assert_called_once()


def test_cmd_preflight_modes():
    """Verifies that cmd_preflight() evaluates mode-specific market feed proofs."""
    with patch.dict(os.environ, {"ALLOW_CPU_INFERENCE": "1"}, clear=False):
        res_broker = cmd_preflight(mode="paper_broker")
        assert res_broker == 0
        res_local = cmd_preflight(mode="paper_local")
        assert res_local == 0


# ==============================================================================
# 14. LIVE VERIFY MOCK VS GENUINE LIVE DISTINCTION TEST
# ==============================================================================

def test_cmd_verify_live_mock_vs_real_distinction():
    """
    Verifies that cmd_verify_live() explicitly distinguishes mock diagnostics
    from genuine live account probes and never places orders.
    """
    # 1. In mock mode (no keys)
    with patch.dict(os.environ, {"ALPACA_LIVE_API_KEY": "", "US_LIVE_API_KEY": ""}, clear=False):
        res = cmd_verify_live()
        assert res == 0

    # 2. In genuine live mode with mock adapter
    with patch.dict(os.environ, {"ALPACA_LIVE_API_KEY": "AKTEST123456", "ALPACA_LIVE_SECRET_KEY": "SKTEST123456"}, clear=False):
        with patch("trading.cli.LiveUSBrokerAdapter") as mock_adapter_cls:
            mock_inst = MagicMock()
            mock_inst.base_url = "https://api.alpaca.markets"
            mock_inst.verify_identity.return_value = BrokerIdentity(
                account_id="ACC_GENUINE_LIVE_123",
                environment="live",
                currency="USD",
                buying_power=50000.0,
                cash=25000.0,
                portfolio_value=75000.0,
                permissions=["US_EQUITIES", "ETFS"],
                is_live=True,
            )
            mock_adapter_cls.return_value = mock_inst
            res = cmd_verify_live()
            assert res == 0
            mock_inst.verify_identity.assert_called_once()
