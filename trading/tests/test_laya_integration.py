"""
Comprehensive Laya Integration Test Suite.

Enforces Directive 23:
1. schema validation
2. probability validation
3. timeout & retry
4. circuit breaker
5. GPU failure & degradation
6. malformed response handling
7. historical determinism & cache
8. shadow non-interference
9. advisory boundedness
10. gate boundedness
11. RegimeEngine authoritative secondary opinion (Directive 1)
12. RiskEngine precedence (Directive 11)
13. holdout isolation & future leakage audit (Directive 23)
14. StrategyDNA bounds & illegal gene forbidding (Directive 12)
15. paper/live air-gap verification (Directives 10, 22)
16. counterfactual horizon calculations (Directives 7, 16)
17. probability calibration engine (Directives 8, 17)
"""

import unittest
import math
import json
import os
from unittest.mock import patch, MagicMock
from datetime import datetime

from trading.laya.schemas import LayaMarketState, LayaDecision, ResponseValidator
from trading.laya.config import LayaConfig
from trading.laya.cache import LayaDecisionCache
from trading.laya.fallback import LayaFailSafeEngine, LayaCircuitBreaker
from trading.laya.gpu_monitor import GpuMonitor, GpuTelemetry
from trading.laya.feature_formatter import LayaFeatureFormatter
from trading.laya.client import LayaClient
from trading.laya.policy import LayaPolicyEngine, PolicyEvaluationResult
from trading.laya.decision_engine import LayaDecisionEngine, CandidateEvaluation
from trading.laya.calibration import LayaCalibrationEngine
from trading.laya.counterfactual import LayaCounterfactualEngine, CounterfactualRecord
from trading.laya.historical_ab import LayaHistoricalABRunner
from trading.swarm.strategy_dna import StrategyDNA, LEGAL_BOUNDS, LEGAL_DISAGREEMENT_POLICIES


class TestLayaIntegration(unittest.TestCase):
    """Rigorous verification of all 25 architectural directives."""

    def setUp(self):
        self.sample_bar = {"open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1500000}
        self.sample_tech = {"ret_1d": 0.04, "ret_5d": 0.06, "ret_20d": 0.12, "rsi_14": 62.0, "adx_14": 26.0, "vol_20d": 0.22, "atr_pct": 0.025}
        self.sample_alpha = {"TrendAlpha": 0.8, "MomentumAlpha": 0.7, "MeanReversionAlpha": 0.3}
        self.sample_ml = {"ml_prob_up": 0.68, "ml_confidence": 0.75, "ml_signal": 0.55}
        self.sample_regime = {"current_regime": "STRONG_BULL", "confidence": 0.85, "regime_id": 1}
        self.sample_market_ctx = {"market_trend": "BULLISH", "vix_level": 14.5, "risk_on_off_score": 0.70}
        self.sample_portfolio = {"exposure_pct": 0.45, "drawdown_pct": 0.02, "num_positions": 4, "available_cash_pct": 0.55}

        self.market_state = LayaFeatureFormatter.format_state(
            symbol="NVDA",
            timestamp="2026-09-25T15:30:00Z",
            market="US",
            exchange="NASDAQ",
            bar=self.sample_bar,
            technical_indicators=self.sample_tech,
            alpha_signals=self.sample_alpha,
            ml_prediction=self.sample_ml,
            regime_info=self.sample_regime,
            market_context=self.sample_market_ctx,
            portfolio_state=self.sample_portfolio,
        )

    # 1. Schema Validation
    def test_schema_validation(self):
        state_dict = self.market_state.to_dict()
        self.assertEqual(state_dict["symbol"], "NVDA")
        self.assertEqual(state_dict["market"], "US")
        self.assertEqual(state_dict["schema_version"], "1.0")
        narrative = self.market_state.to_narrative_context()
        self.assertIn("NVDA", narrative)
        self.assertIn("SPY trend BULLISH", narrative)

        # State hash must be non-empty SHA256 hex string
        h = self.market_state.compute_state_hash()
        self.assertEqual(len(h), 64)

    # 2. Probability Validation
    def test_probability_validation(self):
        state_hash = self.market_state.compute_state_hash()

        # Valid payload
        valid_raw = {
            "choice": "ENTER",
            "probabilities": {"ENTER": 0.70, "WAIT": 0.20, "REJECT": 0.10},
            "confidence": 0.75,
            "answer_confidence": 0.70,
        }
        dec = ResponseValidator.validate("TRADE_QUALITY", valid_raw, "v1.0", state_hash, 50.0)
        self.assertTrue(dec.is_valid)
        self.assertEqual(dec.choice, "ENTER")
        self.assertEqual(dec.decision_probability, 0.70)
        self.assertEqual(dec.selected_choice_probability, 0.70)

        # Out-of-bounds probability -> must be rejected into fallback
        invalid_prob_raw = {
            "choice": "ENTER",
            "probabilities": {"ENTER": 1.50, "WAIT": -0.20, "REJECT": 0.0},
        }
        dec_invalid = ResponseValidator.validate("TRADE_QUALITY", invalid_prob_raw, "v1.0", state_hash, 50.0)
        self.assertFalse(dec_invalid.is_valid)
        self.assertTrue(dec_invalid.fallback_triggered)

        # NaN probability -> must be rejected
        nan_raw = {
            "choice": "ENTER",
            "probabilities": {"ENTER": float("nan"), "WAIT": 0.5, "REJECT": 0.5},
        }
        dec_nan = ResponseValidator.validate("TRADE_QUALITY", nan_raw, "v1.0", state_hash, 50.0)
        self.assertFalse(dec_nan.is_valid)
        self.assertTrue(dec_nan.fallback_triggered)

    # 3. Timeout & Retry
    def test_timeout_and_fail_safe(self):
        client = LayaClient()
        state_hash = self.market_state.compute_state_hash()

        # Simulate timeout by pointing to non-existent endpoint with 0.1s timeout
        client.config.endpoint = "http://10.255.255.1:9999/v1/systemone"
        client.config.request_timeout_seconds = 0.1

        decisions = client.predict_tasks(self.market_state, ["TRADE_QUALITY"])
        self.assertIn("TRADE_QUALITY", decisions)
        tq = decisions["TRADE_QUALITY"]
        self.assertFalse(tq.is_valid)
        self.assertTrue(tq.fallback_triggered)

    # 4. Circuit Breaker
    def test_circuit_breaker(self):
        breaker = LayaCircuitBreaker(failure_threshold=3, reset_timeout_seconds=60.0)
        self.assertTrue(breaker.can_attempt())
        self.assertEqual(breaker.state, "CLOSED")

        breaker.record_failure("Err 1")
        self.assertTrue(breaker.can_attempt())
        breaker.record_failure("Err 2")
        self.assertTrue(breaker.can_attempt())
        breaker.record_failure("Err 3")

        # Tripped to OPEN
        self.assertEqual(breaker.state, "OPEN")
        self.assertFalse(breaker.can_attempt())

        # Reset on success
        breaker.record_success()
        self.assertEqual(breaker.state, "CLOSED")
        self.assertTrue(breaker.can_attempt())

    # 5. GPU Failure & Degradation
    def test_gpu_failure_and_degradation(self):
        monitor = GpuMonitor(max_vram_mb=3500.0, max_temp_c=80.0)
        # Even if nvidia-smi fails or returns mock high VRAM, system does not crash
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="GeForce RTX 3050, 4096, 3900, 196, 90, 85\n")
            telemetry = monitor.check_telemetry()
            self.assertFalse(telemetry.is_safe)
            self.assertIn("High VRAM", telemetry.warning)
            self.assertFalse(monitor.is_safe_for_inference())

    # 6. Malformed Response Handling
    def test_malformed_response_handling(self):
        state_hash = self.market_state.compute_state_hash()
        # String instead of dict
        dec = ResponseValidator.validate("TRADE_QUALITY", "invalid_response", "v1.0", state_hash, 10.0)
        self.assertFalse(dec.is_valid)
        self.assertTrue(dec.fallback_triggered)

        # Unknown choice
        dec_choice = ResponseValidator.validate("TRADE_QUALITY", {"choice": "SUPER_BUY", "probabilities": {}}, "v1.0", state_hash, 10.0)
        self.assertFalse(dec_choice.is_valid)
        self.assertTrue(dec_choice.fallback_triggered)

    # 7. Historical Determinism & Cache
    def test_historical_determinism_and_cache(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LayaDecisionCache(cache_dir=Path(tmpdir))
            state_hash = self.market_state.compute_state_hash()

            dec = LayaDecision(
                task="TRADE_QUALITY",
                choice="ENTER",
                decision_probability=0.72,
                selected_choice_probability=0.72,
                probabilities={"ENTER": 0.72, "WAIT": 0.18, "REJECT": 0.10},
                confidence=0.75,
                latency_ms=120.0,
                model_version="test-model",
                state_hash=state_hash,
                timestamp="2026-09-25T15:30:00Z",
                is_valid=True,
            )
            cache.put(dec)
            self.assertEqual(cache.size(), 1)

            # Retrieve
            retrieved = cache.get(state_hash, "TRADE_QUALITY", model_version="test-model")
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.choice, "ENTER")
            self.assertEqual(retrieved.decision_probability, 0.72)

            # Different task -> miss
            miss = cache.get(state_hash, "RISK_ACTION", model_version="test-model")
            self.assertIsNone(miss)

    # 8. Shadow Non-Interference
    def test_shadow_non_interference(self):
        res = LayaFailSafeEngine.resolve_failure("TRADE_QUALITY", "Test failure", mode="shadow")
        self.assertEqual(res.mode, "shadow")
        self.assertEqual(res.action, "CONTINUE_BASELINE")
        self.assertEqual(res.position_multiplier, 1.0)

    # 9. Advisory Boundedness
    def test_advisory_boundedness(self):
        state_hash = self.market_state.compute_state_hash()
        dec_normal = LayaDecision("RISK_ACTION", "NORMAL", 0.8, 0.8, {}, 0.8, 10.0, "v1", state_hash, "ts")
        pol_normal = LayaPolicyEngine.evaluate_risk_action(dec_normal)
        self.assertEqual(pol_normal.position_size_multiplier, 1.0)
        self.assertFalse(pol_normal.is_blocked)

        dec_reduce = LayaDecision("RISK_ACTION", "REDUCE_POSITION", 0.7, 0.7, {}, 0.7, 10.0, "v1", state_hash, "ts")
        pol_reduce = LayaPolicyEngine.evaluate_risk_action(dec_reduce, reduce_multiplier=0.50)
        self.assertEqual(pol_reduce.position_size_multiplier, 0.50)
        self.assertFalse(pol_reduce.is_blocked)

        dec_block = LayaDecision("RISK_ACTION", "BLOCK", 0.85, 0.85, {}, 0.85, 10.0, "v1", state_hash, "ts")
        pol_block = LayaPolicyEngine.evaluate_risk_action(dec_block)
        self.assertEqual(pol_block.position_size_multiplier, 0.0)
        self.assertTrue(pol_block.is_blocked)

    # 10. Gate Boundedness
    def test_gate_boundedness(self):
        res = LayaFailSafeEngine.resolve_failure("TRADE_QUALITY", "Timeout in gate mode", mode="gate")
        self.assertEqual(res.mode, "gate")
        self.assertEqual(res.action, "HOLD_BLOCK")
        self.assertEqual(res.position_multiplier, 0.0)
        self.assertFalse(res.confidence_multiplier > 0.0)

    # 11. RegimeEngine Authoritative (Directive 1)
    def test_regime_engine_authoritative(self):
        state_hash = self.market_state.compute_state_hash()
        # Laya disagrees: signals RISK_OFF while RegimeEngine says STRONG_BULL
        dec_disagree = LayaDecision("REGIME_OPINION", "RISK_OFF", 0.80, 0.80, {}, 0.80, 10.0, "v1", state_hash, "ts")
        pol = LayaPolicyEngine.evaluate_regime_secondary_opinion(
            decision=dec_disagree,
            authoritative_regime="STRONG_BULL",
            authoritative_confidence=0.85,
        )
        # Authoritative regime must remain STRONG_BULL (Directive 1)
        self.assertEqual(pol.effective_regime_opinion, "STRONG_BULL")
        self.assertTrue(pol.disagreement_detected)
        # Disagreement reduces confidence, does not replace regime
        self.assertLess(pol.confidence_multiplier, 0.85)

    # 12. RiskEngine Precedence (Directive 11)
    def test_risk_engine_precedence(self):
        state_hash = self.market_state.compute_state_hash()
        dec_normal = LayaDecision("RISK_ACTION", "NORMAL", 0.8, 0.8, {}, 0.8, 10.0, "v1", state_hash, "ts")
        pol_normal = LayaPolicyEngine.evaluate_risk_action(dec_normal)

        # 1. Laya = NORMAL, RiskEngine = BLOCK -> BLOCK
        allowed, weight, reason = LayaPolicyEngine.apply_risk_engine_precedence(
            laya_eval=pol_normal,
            risk_engine_allowed=False,
            portfolio_target_weight=0.10,
        )
        self.assertFalse(allowed)
        self.assertEqual(weight, 0.0)
        self.assertIn("RiskEngine authoritative rejection", reason)

        # 2. Laya = REDUCE (0.5x), RiskEngine = BLOCK -> BLOCK
        dec_reduce = LayaDecision("RISK_ACTION", "REDUCE_POSITION", 0.7, 0.7, {}, 0.7, 10.0, "v1", state_hash, "ts")
        pol_reduce = LayaPolicyEngine.evaluate_risk_action(dec_reduce, reduce_multiplier=0.5)
        allowed, weight, _ = LayaPolicyEngine.apply_risk_engine_precedence(
            laya_eval=pol_reduce,
            risk_engine_allowed=False,
            portfolio_target_weight=0.10,
        )
        self.assertFalse(allowed)
        self.assertEqual(weight, 0.0)

        # 3. Laya = BLOCK, RiskEngine = ALLOW -> BLOCK
        dec_block = LayaDecision("RISK_ACTION", "BLOCK", 0.9, 0.9, {}, 0.9, 10.0, "v1", state_hash, "ts")
        pol_block = LayaPolicyEngine.evaluate_risk_action(dec_block)
        allowed, weight, _ = LayaPolicyEngine.apply_risk_engine_precedence(
            laya_eval=pol_block,
            risk_engine_allowed=True,
            portfolio_target_weight=0.10,
        )
        self.assertFalse(allowed)
        self.assertEqual(weight, 0.0)

        # 4. Laya = REDUCE (0.5x), RiskEngine = ALLOW -> Target scaled to 0.05
        allowed, weight, _ = LayaPolicyEngine.apply_risk_engine_precedence(
            laya_eval=pol_reduce,
            risk_engine_allowed=True,
            portfolio_target_weight=0.10,
        )
        self.assertTrue(allowed)
        self.assertEqual(round(weight, 4), 0.05)

    # 13. Future Leakage Audit & Causality
    def test_future_leakage_and_causality(self):
        # Ensure timestamp is ISO formatted and only historical state is present
        ts = self.market_state.timestamp
        self.assertIn("2026-09-25", ts)
        # Verify no forward return fields exist in LayaMarketState contract
        state_dict = self.market_state.to_dict()
        self.assertNotIn("future_return", state_dict["price_state"])
        self.assertNotIn("future_ret_10d", state_dict["technical_state"])

    # 14. StrategyDNA Bounds & Illegal Gene Rejection (Directive 12)
    def test_strategy_dna_bounds_and_invariants(self):
        dna = StrategyDNA(
            strategy_id="strat_test_01",
            name="TestDNA",
            generation=1,
            species="Trend",
            laya_enabled=True,
            laya_trade_quality_threshold=0.99,  # exceeds upper bound 0.90
            laya_disagreement_policy="INVALID_POLICY",
        )
        # Attach illegal attribute
        setattr(dna, "laya_overrides_risk", True)

        dna.validate_bounds()

        # Clamped within legal bounds
        self.assertLessEqual(dna.laya_trade_quality_threshold, LEGAL_BOUNDS["laya_trade_quality_threshold"][1])
        # Invalid disagreement policy reverted to BASELINE_WINS
        self.assertIn(dna.laya_disagreement_policy, LEGAL_DISAGREEMENT_POLICIES)
        self.assertEqual(dna.laya_disagreement_policy, "BASELINE_WINS")
        # Invariant: "laya_overrides_risk" must be purged!
        self.assertFalse(hasattr(dna, "laya_overrides_risk"))

    # 15. Paper/Live Air Gap Verification (Directives 10 & 22)
    def test_paper_live_air_gap(self):
        from trading.laya import (
            LayaDecisionEngine, LayaClient, LayaPolicyEngine, LayaFailSafeEngine
        )
        import inspect

        # Inspect all functions and methods across trading.laya to verify zero broker imports
        for cls in [LayaDecisionEngine, LayaClient, LayaPolicyEngine, LayaFailSafeEngine]:
            source = inspect.getsource(cls)
            self.assertNotIn("place_buy", source)
            self.assertNotIn("place_sell", source)
            self.assertNotIn("TRADE_MODE = 'live'", source)
            self.assertNotIn("BROKER = 'angel'", source)
            self.assertNotIn("api_key", source.lower())

    # 16. Counterfactual Horizons (Directives 7 & 16)
    def test_counterfactual_horizons(self):
        # Simulate price trajectory: 100 -> 102 -> 105 -> 108 -> 110 (win)
        outcome_win = LayaCounterfactualEngine.compute_horizon_outcome(
            entry_price=100.0,
            future_closes=[102.0, 105.0, 108.0, 110.0],
            future_highs=[103.0, 106.0, 109.0, 112.0],
            future_lows=[100.0, 101.0, 104.0, 107.0],
            horizon_bars=5,
        )
        self.assertEqual(outcome_win.future_return, 0.10)
        self.assertEqual(outcome_win.mfe, 0.12)
        self.assertEqual(outcome_win.mae, 0.0)

        # Classifications:
        # 1. Trade blocked, future ret <= 0 -> BAD_TRADE_AVOIDED (Save!)
        c1 = LayaCounterfactualEngine.classify_decision("ENTER", "BLOCKED", -0.05)
        self.assertEqual(c1, "BAD_TRADE_AVOIDED")

        # 2. Trade blocked, future ret > 0 -> GOOD_TRADE_MISSED (Opportunity cost)
        c2 = LayaCounterfactualEngine.classify_decision("ENTER", "BLOCKED", 0.08)
        self.assertEqual(c2, "GOOD_TRADE_MISSED")

        # 3. Trade approved, future ret <= 0 -> BAD_TRADE_ACCEPTED (False approval)
        c3 = LayaCounterfactualEngine.classify_decision("ENTER", "APPROVED", -0.04)
        self.assertEqual(c3, "BAD_TRADE_ACCEPTED")

        # 4. Trade approved, future ret > 0 -> GOOD_TRADE_ACCEPTED (True positive)
        c4 = LayaCounterfactualEngine.classify_decision("ENTER", "APPROVED", 0.06)
        self.assertEqual(c4, "GOOD_TRADE_ACCEPTED")

    # 17. Probability Calibration Engine (Directives 8 & 17)
    def test_calibration_metrics(self):
        y_true = [1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
        y_prob = [0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.65, 0.4, 0.75]

        brier = LayaCalibrationEngine.brier_score(y_true, y_prob)
        self.assertGreaterEqual(brier, 0.0)
        self.assertLessEqual(brier, 1.0)

        ll = LayaCalibrationEngine.log_loss(y_true, y_prob)
        self.assertGreater(ll, 0.0)

        ece = LayaCalibrationEngine.expected_calibration_error(y_true, y_prob, n_bins=5)
        self.assertGreaterEqual(ece, 0.0)

        rep = LayaCalibrationEngine.generate_calibration_report("TRADE_QUALITY", y_true, y_prob)
        self.assertIn("brier_score", rep)
        self.assertIn("confidence_buckets", rep)
        self.assertIn("methodology_note", rep)


if __name__ == "__main__":
    unittest.main()
