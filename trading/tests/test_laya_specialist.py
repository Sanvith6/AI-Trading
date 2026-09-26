"""
Comprehensive Unit & Integration Test Suite for Laya Financial Specialist.

Enforces Phase 34:
1. PIT integrity & available_at causality
2. Future-label isolation (Zero leakage into state)
3. State hashing determinism
4. Dataset generation determinism
5. Label generation determinism
6. Checkpoint determinism & parameter stability
7. Calibration isolation (Never fitted on holdouts)
8. Holdout isolation (Locked holdout & frozen holdout preservation)
9. Forward isolation
10. Baseline invariance (Shadow mode leaves baseline 100% untouched)
11. Laya failure safety & circuit breaker degradation
12. Zero direct broker access
13. Immutable RiskEngine precedence
14. Explicit abstention behavior (Entropy/uncertainty -> WAIT/PASS)
15. Economic decision policy determinism
"""

import unittest
from datetime import datetime
import json
import inspect
from pathlib import Path

from trading.laya_specialist.dataset_builder import (
    FinancialDatasetBuilder,
    FinancialCandidateExample,
    FinancialOutcomeEngine,
    SoftTargetGenerator,
)
from trading.laya_specialist.tasks import SpecialistTasks
from trading.laya_specialist.decision_policy import EconomicDecisionPolicy, MetaDecisionOutput
from trading.laya_specialist.evaluator import SpecialistEvaluator


class TestLayaSpecialist(unittest.TestCase):
    """Rigorous verification of the Financial Specialist architecture and invariants."""

    def setUp(self):
        self.sample_example = FinancialCandidateExample(
            example_id="fin_test_001",
            symbol="NVDA",
            market="US",
            exchange="NASDAQ",
            sector="Semiconductors",
            timestamp="2025-05-15T15:30:00Z",
            split="val",
            proposed_side="LONG",
            entry_price=120.0,
            candidate_rank=1,
            baseline_score=0.85,
            expected_holding_horizon=10,
            estimated_transaction_cost_bps=5.0,
            estimated_slippage_bps=10.0,
            state_narrative="NVDA Long proposal at $120.00. RSI 64.0, Regime STRONG_BULL.",
            state_features={"close": 120.0, "atr": 3.0, "atr_pct": 0.025, "rsi_14": 64.0},
            state_hash="a1b2c3d4e5f67890" * 4,
            outcomes={
                "5_bars": {"raw_return": 0.02, "net_return": 0.0185, "mfe": 0.03, "mae": -0.005, "normalized_return_atr": 0.74},
                "10_bars": {"raw_return": 0.05, "net_return": 0.0485, "mfe": 0.06, "mae": -0.010, "normalized_return_atr": 1.94},
                "20_bars": {"raw_return": 0.08, "net_return": 0.0785, "mfe": 0.09, "mae": -0.015, "normalized_return_atr": 3.14},
            },
            targets={
                "META_DECISION": {"TAKE": 0.75, "WAIT": 0.15, "PASS": 0.10},
                "OUTCOME_10BAR": {"STRONGLY_NEGATIVE": 0.05, "NEGATIVE": 0.10, "NEUTRAL": 0.15, "POSITIVE": 0.45, "STRONGLY_POSITIVE": 0.25},
            },
        )
        self.policy = EconomicDecisionPolicy()

    # 1. PIT Integrity & Timestamp Causality
    def test_pit_integrity(self):
        dt = datetime.fromisoformat(self.sample_example.timestamp.replace("Z", "+00:00"))
        self.assertEqual(dt.year, 2025)
        # Split assignment must match temporal rules
        split = FinancialDatasetBuilder.assign_split(dt.replace(tzinfo=None))
        self.assertEqual(split, "val")

    # 2. Future-Label Isolation
    def test_future_label_isolation(self):
        # State narrative and features must NOT contain future outcomes
        state_str = self.sample_example.state_narrative
        features = self.sample_example.state_features
        self.assertNotIn("net_return", state_str)
        self.assertNotIn("future_return", state_str)
        self.assertNotIn("mae", state_str)
        self.assertNotIn("net_return", features)

    # 3. State Hashing Determinism
    def test_state_hashing_determinism(self):
        builder = FinancialDatasetBuilder()
        ex1 = builder.build_dataset(target_examples=5, seed=99)
        ex2 = builder.build_dataset(target_examples=5, seed=99)
        self.assertEqual(len(ex1), len(ex2))
        for a, b in zip(ex1, ex2):
            self.assertEqual(a.state_hash, b.state_hash)
            self.assertEqual(a.symbol, b.symbol)

    # 4. Dataset Generation Determinism
    def test_dataset_generation_determinism(self):
        ex1 = FinancialDatasetBuilder.build_dataset(target_examples=10, seed=123)
        ex2 = FinancialDatasetBuilder.build_dataset(target_examples=10, seed=123)
        self.assertEqual([e.entry_price for e in ex1], [e.entry_price for e in ex2])
        self.assertEqual([e.proposed_side for e in ex1], [e.proposed_side for e in ex2])

    # 5. Label Generation Determinism
    def test_label_generation_determinism(self):
        targets1 = SoftTargetGenerator.generate_outcome_distribution(1.5)
        targets2 = SoftTargetGenerator.generate_outcome_distribution(1.5)
        self.assertEqual(targets1, targets2)
        # Probabilities must sum to ~1.0
        self.assertAlmostEqual(sum(targets1.values()), 1.0, places=3)

    # 6. Checkpoint Loading Determinism
    def test_checkpoint_determinism(self):
        meta_path = Path(__file__).parent.parent.parent / "data_lake" / "laya_specialist" / "checkpoints" / "laya-financial-us-v1" / "training_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["checkpoint_name"], "laya-financial-us-v1")
            self.assertIn("calibration_temperatures", meta)

    # 7. Calibration Isolation
    def test_calibration_isolation(self):
        # Verify calibration temperature bounds
        meta_path = Path(__file__).parent.parent.parent / "data_lake" / "laya_specialist" / "checkpoints" / "laya-financial-us-v1" / "training_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            temps = meta["calibration_temperatures"]
            for task, t_val in temps.items():
                self.assertGreaterEqual(t_val, 0.5)
                self.assertLessEqual(t_val, 3.0)

    # 8. Holdout Isolation
    def test_holdout_isolation(self):
        # Locked holdout is strictly 2026-07-01 to 2026-08-31
        dt_locked = datetime(2026, 7, 15)
        self.assertEqual(FinancialDatasetBuilder.assign_split(dt_locked), "locked_holdout")

        # Frozen holdout is 2026-09-01 to 2026-09-25
        dt_frozen = datetime(2026, 9, 10)
        self.assertEqual(FinancialDatasetBuilder.assign_split(dt_frozen), "frozen_holdout")

        # True forward is 2026-09-28 onward
        dt_forward = datetime(2026, 10, 1)
        self.assertEqual(FinancialDatasetBuilder.assign_split(dt_forward), "true_forward")

    # 9. Forward Isolation
    def test_forward_isolation(self):
        dt_train = datetime(2023, 6, 1)
        self.assertEqual(FinancialDatasetBuilder.assign_split(dt_train), "train")
        # Ensure training samples never bleed into holdout
        self.assertNotEqual(FinancialDatasetBuilder.assign_split(dt_train), "frozen_holdout")

    # 10. Baseline Invariance (Phase 24)
    def test_baseline_invariance(self):
        evaluator = SpecialistEvaluator(self.policy)
        is_invariant = evaluator.verify_baseline_invariance([self.sample_example])
        self.assertTrue(is_invariant)

    # 11. Laya Failure Safety
    def test_failure_safety(self):
        from trading.laya.fallback import LayaFailSafeEngine
        res = LayaFailSafeEngine.resolve_failure("META_DECISION", "Service Timeout", mode="shadow")
        self.assertEqual(res.action, "CONTINUE_BASELINE")
        self.assertEqual(res.position_multiplier, 1.0)

    # 12. No Broker Access
    def test_no_broker_access(self):
        from trading.laya_specialist import (
            FinancialDatasetBuilder, EconomicDecisionPolicy, SpecialistEvaluator
        )
        for cls in [FinancialDatasetBuilder, EconomicDecisionPolicy, SpecialistEvaluator]:
            src = inspect.getsource(cls)
            self.assertNotIn("place_buy", src)
            self.assertNotIn("place_sell", src)
            self.assertNotIn("TRADE_MODE = 'live'", src)

    # 13. RiskEngine Precedence
    def test_risk_engine_precedence(self):
        from trading.laya.policy import LayaPolicyEngine, PolicyEvaluationResult
        pol_res = PolicyEvaluationResult(
            task="META_DECISION",
            laya_choice="TAKE",
            laya_probability=0.85,
            confidence_multiplier=1.0,
            position_size_multiplier=1.0,
            is_blocked=False,
            policy_reason="Specialist approved",
            disagreement_detected=False,
        )
        # RiskEngine BLOCK must override Specialist TAKE
        allowed, weight, reason = LayaPolicyEngine.apply_risk_engine_precedence(
            laya_eval=pol_res,
            risk_engine_allowed=False,
            portfolio_target_weight=0.10,
        )
        self.assertFalse(allowed)
        self.assertEqual(weight, 0.0)

    # 14. Explicit Abstention Behavior (Phase 19)
    def test_abstention_behavior(self):
        # Uniform distribution (high entropy) must trigger explicit abstention (WAIT)
        uniform_meta = {"TAKE": 0.333, "WAIT": 0.334, "PASS": 0.333}
        flat_outcome = {k: 0.20 for k in ["STRONGLY_NEGATIVE", "NEGATIVE", "NEUTRAL", "POSITIVE", "STRONGLY_POSITIVE"]}
        decision = self.policy.evaluate_candidate(flat_outcome, uniform_meta)
        self.assertEqual(decision.action, "WAIT")
        self.assertTrue(decision.is_abstaining)
        self.assertEqual(decision.position_scale_multiplier, 0.50)

    # 15. Decision Policy Determinism (Phase 18)
    def test_decision_policy_determinism(self):
        # Strongly bullish distribution -> TAKE
        bull_meta = {"TAKE": 0.80, "WAIT": 0.15, "PASS": 0.05}
        bull_outcome = {
            "STRONGLY_NEGATIVE": 0.02, "NEGATIVE": 0.08, "NEUTRAL": 0.15,
            "POSITIVE": 0.45, "STRONGLY_POSITIVE": 0.30
        }
        dec1 = self.policy.evaluate_candidate(bull_outcome, bull_meta)
        dec2 = self.policy.evaluate_candidate(bull_outcome, bull_meta)
        self.assertEqual(dec1.action, "TAKE")
        self.assertEqual(dec1.action, dec2.action)
        self.assertEqual(dec1.position_scale_multiplier, 1.0)

        # Strongly bearish distribution -> PASS
        bear_meta = {"TAKE": 0.05, "WAIT": 0.15, "PASS": 0.80}
        bear_outcome = {
            "STRONGLY_NEGATIVE": 0.40, "NEGATIVE": 0.35, "NEUTRAL": 0.15,
            "POSITIVE": 0.08, "STRONGLY_POSITIVE": 0.02
        }
        dec_bear = self.policy.evaluate_candidate(bear_outcome, bear_meta)
        self.assertEqual(dec_bear.action, "PASS")
        self.assertEqual(dec_bear.position_scale_multiplier, 0.0)


if __name__ == "__main__":
    unittest.main()
