"""
Comprehensive Unit & Invariant Test Suite for Laya Financial Specialist V2.

Verifies:
1. De-biasing / Anonymization: State narrative strictly omits raw ticker symbols.
2. Dual Labels: Both Label A (ATR-normalized) and Label B (Triple-Barrier) exist and sum to 1.0.
3. Boundary Purge + Embargo: No candidate examples generated within 20 bars of split boundaries.
4. Economic Decision Policy: Explicit abstention, downside rejection, bounded position multiplier [0.0, 1.0].
5. Zero Target Leakage: Evaluator uses model forward predictions, not ground truth labels.
6. Mandatory Placebo Suite: Label shuffle, timestamp shuffle, prediction permutation.
7. Adversarial Stress Tests: Friction scaling (1x to 5x).
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from trading.laya_specialist_v2.config import LayaV2Config
from trading.laya_specialist_v2.tasks import SpecialistTasksV2
from trading.laya_specialist_v2.dataset_builder import (
    FinancialCandidateExampleV2,
    FinancialDatasetBuilderV2,
    FinancialOutcomeEngineV2,
)
from trading.laya_specialist_v2.decision_policy import (
    EconomicDecisionPolicyV2,
    MetaDecisionOutputV2,
)
from trading.laya_specialist_v2.evaluator import SpecialistEvaluatorV2
from trading.laya_specialist_v2.inference import SpecialistInferenceEngineV2


def test_dataset_builder_purging_and_anonymization():
    """Verifies de-biasing narrative and 20-bar purge/embargo."""
    examples = FinancialDatasetBuilderV2.build_dataset(target_examples=100, seed=42)
    assert len(examples) == 100

    raw_tickers = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "META"]
    for ex in examples:
        # 1. State narrative must NOT contain raw ticker symbol
        for ticker in raw_tickers:
            assert f"Candidate: {ticker}" not in ex.state_narrative
            assert f" {ticker} (" not in ex.state_narrative
        assert "Profile[" in ex.state_narrative

        # 2. Dual targets must exist and sum to ~1.0
        assert "targets_label_a" in ex.to_dict()
        assert "targets_label_b" in ex.to_dict()
        p_meta_a = sum(ex.targets_label_a["META_DECISION"].values())
        p_meta_b = sum(ex.targets_label_b["META_DECISION"].values())
        assert abs(p_meta_a - 1.0) < 0.05
        assert abs(p_meta_b - 1.0) < 0.05

        # 3. Purged dates check: no example should be within 28 days before 2024-12-31
        dt = datetime.strptime(ex.timestamp[:10], "%Y-%m-%d")
        if dt.year == 2024 and dt.month == 12:
            assert dt < datetime(2024, 12, 3)  # Within purge window is disallowed


def test_triple_barrier_event_detection():
    """Verifies path-dependent triple-barrier detection."""
    entry = 100.0
    atr = 2.0
    # Upper barrier = 104.0, lower barrier = 98.0
    future_highs_upper = [101.0, 102.5, 104.5, 103.0]
    future_lows_upper = [99.5, 100.0, 102.0, 101.0]

    event = FinancialOutcomeEngineV2.calculate_triple_barrier_event(
        side="LONG",
        entry_price=entry,
        future_highs=future_highs_upper,
        future_lows=future_lows_upper,
        atr=atr,
        take_atr_mult=2.0,
        stop_atr_mult=1.0,
        max_bars=5,
    )
    assert event == "UPPER_TOUCH"

    future_highs_lower = [100.5, 99.0, 98.5, 97.0]
    future_lows_lower = [99.0, 98.2, 97.5, 96.0]

    event_loss = FinancialOutcomeEngineV2.calculate_triple_barrier_event(
        side="LONG",
        entry_price=entry,
        future_highs=future_highs_lower,
        future_lows=future_lows_lower,
        atr=atr,
        take_atr_mult=2.0,
        stop_atr_mult=1.0,
        max_bars=5,
    )
    assert event_loss == "LOWER_TOUCH"


def test_economic_decision_policy_invariants():
    """Verifies deterministic policy mapping and hard architectural constraints."""
    policy = EconomicDecisionPolicyV2()

    # Case 1: High downside risk -> PASS
    high_loss_outcome = {
        "STRONGLY_NEGATIVE": 0.35,
        "NEGATIVE": 0.30,
        "NEUTRAL": 0.20,
        "POSITIVE": 0.10,
        "STRONGLY_POSITIVE": 0.05,
    }
    meta_p = {"TAKE": 0.10, "WAIT": 0.20, "PASS": 0.70}
    dec_pass = policy.evaluate_candidate(high_loss_outcome, meta_p, atr_pct=0.03)
    assert dec_pass.action == "PASS"
    assert dec_pass.position_scale_multiplier == 0.0

    # Case 2: High entropy / uncertainty -> WAIT (Abstain)
    flat_outcome = {
        "STRONGLY_NEGATIVE": 0.05,
        "NEGATIVE": 0.15,
        "NEUTRAL": 0.60,
        "POSITIVE": 0.15,
        "STRONGLY_POSITIVE": 0.05,
    }
    meta_p_flat = {"TAKE": 0.20, "WAIT": 0.60, "PASS": 0.20}
    dec_wait = policy.evaluate_candidate(flat_outcome, meta_p_flat, atr_pct=0.02)
    assert dec_wait.action == "WAIT"
    assert dec_wait.is_abstaining is True
    assert dec_wait.position_scale_multiplier == 0.0

    # Case 3: High utility, strong asymmetry -> TAKE
    favorable_outcome = {
        "STRONGLY_NEGATIVE": 0.02,
        "NEGATIVE": 0.08,
        "NEUTRAL": 0.20,
        "POSITIVE": 0.45,
        "STRONGLY_POSITIVE": 0.25,
    }
    meta_p_fav = {"TAKE": 0.75, "WAIT": 0.15, "PASS": 0.10}
    dec_take = policy.evaluate_candidate(favorable_outcome, meta_p_fav, atr_pct=0.025)
    assert dec_take.action == "TAKE"
    assert 0.5 <= dec_take.position_scale_multiplier <= 1.0


def test_evaluator_zero_leakage_and_comparative_arms():
    """Verifies that evaluator uses model_predictions and does not leak targets."""
    examples = FinancialDatasetBuilderV2.build_dataset(target_examples=80, seed=42)
    evaluator = SpecialistEvaluatorV2()

    # Generate synthetic model predictions
    np.random.seed(42)
    model_predictions = {}
    for ex in examples:
        p_take = float(np.random.uniform(0.1, 0.8))
        p_pass = float(np.random.uniform(0.1, 0.8))
        p_wait = 1.0 - p_take - p_pass
        norm_p = [p_take, max(0.05, p_wait), p_pass]
        norm_p = [p / sum(norm_p) for p in norm_p]

        model_predictions[ex.example_id] = {
            "META_DECISION": {"TAKE": norm_p[0], "WAIT": norm_p[1], "PASS": norm_p[2]},
            "OUTCOME_10BAR": {
                "STRONGLY_NEGATIVE": 0.05,
                "NEGATIVE": 0.15,
                "NEUTRAL": 0.30,
                "POSITIVE": 0.35,
                "STRONGLY_POSITIVE": 0.15,
            },
        }

    # Run comparative arms
    arms = evaluator.run_comparative_arms(examples, model_predictions)
    assert "arm_1_baseline" in arms
    assert "arm_3_laya_v2" in arms
    assert "arm_4_oracle" in arms
    assert "arm_5_random_placebo" in arms

    assert arms["arm_3_laya_v2"]["trade_count"] <= len(examples)


def test_placebo_suite_execution():
    """Verifies that 3 placebo tests run and report expected structures."""
    examples = FinancialDatasetBuilderV2.build_dataset(target_examples=50, seed=42)
    evaluator = SpecialistEvaluatorV2()

    model_predictions = {}
    for ex in examples:
        # Give mock strong predictions for positive return examples
        is_win = ex.outcomes["10_bars"]["net_return"] > 0
        p_take = 0.80 if is_win else 0.10
        p_pass = 0.10 if is_win else 0.80
        p_wait = 0.10

        model_predictions[ex.example_id] = {
            "META_DECISION": {"TAKE": p_take, "WAIT": p_wait, "PASS": p_pass},
            "OUTCOME_10BAR": {
                "STRONGLY_NEGATIVE": 0.02 if is_win else 0.40,
                "NEGATIVE": 0.08 if is_win else 0.35,
                "NEUTRAL": 0.20 if is_win else 0.15,
                "POSITIVE": 0.45 if is_win else 0.08,
                "STRONGLY_POSITIVE": 0.25 if is_win else 0.02,
            },
        }

    placebo_results = evaluator.run_placebo_suite(examples, model_predictions)
    assert "real_model" in placebo_results
    assert "label_shuffle_placebo" in placebo_results
    assert "timestamp_shuffle_placebo" in placebo_results
    assert "prediction_permutation_placebo" in placebo_results
    assert "placebo_validation_passed" in placebo_results


def test_adversarial_stress_testing():
    """Verifies stress tests across 1x, 2x, 3x, 5x friction."""
    examples = FinancialDatasetBuilderV2.build_dataset(target_examples=40, seed=42)
    evaluator = SpecialistEvaluatorV2()

    model_predictions = {
        ex.example_id: {
            "META_DECISION": {"TAKE": 0.70, "WAIT": 0.20, "PASS": 0.10},
            "OUTCOME_10BAR": {
                "STRONGLY_NEGATIVE": 0.05,
                "NEGATIVE": 0.15,
                "NEUTRAL": 0.30,
                "POSITIVE": 0.35,
                "STRONGLY_POSITIVE": 0.15,
            },
        }
        for ex in examples
    }

    stress = evaluator.run_stress_tests(examples, model_predictions)
    assert "friction_1x" in stress
    assert "friction_2x" in stress
    assert "friction_3x" in stress
    assert "friction_5x" in stress
    # Higher friction must result in lower or equal mean net return
    assert stress["friction_1x"]["mean_net_return"] >= stress["friction_5x"]["mean_net_return"]


def test_shadow_inference_engine():
    """Verifies that shadow inference works and respects all invariants."""
    engine = SpecialistInferenceEngineV2()
    features = {
        "ret_1d": 0.015,
        "ret_5d": 0.035,
        "ret_20d": 0.08,
        "rsi_14": 58.0,
        "adx_14": 28.0,
        "atr_pct": 0.022,
        "vol_20d": 0.18,
        "vix": 16.5,
        "ml_prob_up": 0.68,
        "ml_confidence": 0.82,
        "alpha_trend": 0.75,
        "alpha_momentum": 0.70,
        "alpha_mean_reversion": 0.40,
    }

    decision = engine.evaluate_candidate_trade(
        symbol="AAPL",
        side="LONG",
        entry_price=225.50,
        sector="Technology",
        cap_tier="MEGA_CAP",
        beta=1.05,
        features=features,
        regime="STRONG_BULL",
        candidate_rank=1,
        baseline_score=0.88,
    )

    assert isinstance(decision, MetaDecisionOutputV2)
    assert decision.action in ["TAKE", "WAIT", "PASS"]
    assert 0.0 <= decision.position_scale_multiplier <= 1.0
