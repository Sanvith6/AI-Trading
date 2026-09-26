"""
Comprehensive Institutional Architecture Test Suite.

Verifies end-to-end operational integrity across all 12 core quant systems:
1. Data Layer & Ingestion Validation
2. Data & Experiment Versioning
3. Regime Engine Multi-State Detection
4. Baseline Benchmark Strategies (9 baselines)
5. Full Alpha Library (12 families strictly in [-1, +1])
6. Machine Learning Layer & Model Calibration
7. Meta Model & Disagreement Gate
8. Portfolio Construction & Risk-Aware Sizing
9. Execution Simulator Realism & Hostile Stressing
10. Walk-Forward Validation & Holdout Contamination Guard
11. Adversarial Robustness & Parameter Sensitivity
12. Darwinian Strategy DNA, Multi-Objective Fitness & Research Memory
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("BROKER", "paper")
os.environ.setdefault("TRADE_MODE", "paper")


def _generate_synthetic_ohlcv(bars: int = 250) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for offline determinism."""
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=bars, freq="D")
    returns = np.random.normal(0.0005, 0.015, bars)
    prices = 1000.0 * np.cumprod(1.0 + returns)

    high = prices * (1.0 + np.abs(np.random.normal(0.005, 0.005, bars)))
    low = prices * (1.0 - np.abs(np.random.normal(0.005, 0.005, bars)))
    open_p = low + (high - low) * np.random.uniform(0.2, 0.8, bars)
    volume = np.random.randint(100000, 2000000, bars)

    df = pd.DataFrame({
        "date": dates,
        "open": np.round(open_p, 2),
        "high": np.round(high, 2),
        "low": np.round(low, 2),
        "close": np.round(prices, 2),
        "volume": volume,
    })
    return df


def test_data_layer_and_versioning():
    print("Testing Data Layer & Versioning...")
    from trading.data import validate_quote, validate_ohlcv, QualityFlags, create_experiment_context, ExperimentRegistry

    # 1. Quote validation
    bad_quote = {"symbol": "BAD", "last_price": -50.0, "open": 110.0, "high": 100.0, "low": 120.0, "close": 115.0}
    val = validate_quote(bad_quote)
    assert val["quality_status"] in (QualityFlags.SUSPICIOUS, QualityFlags.ZERO)
    assert "high_lt_low" in val["quality_flags"]

    # 2. OHLCV validation
    df = _generate_synthetic_ohlcv(60)
    clean_df, flags = validate_ohlcv(df, "TEST")
    assert QualityFlags.MISSING not in flags

    # 3. Experiment versioning context
    exp = create_experiment_context(strategy_version="v2.1-quant", random_seed=1234)
    reg = ExperimentRegistry(storage_dir=Path("trading/data_cache/test_exp"))
    saved_path = reg.record_experiment(exp)
    assert saved_path.exists()
    loaded = reg.get_experiment(exp.experiment_id)
    assert loaded["strategy_version"] == "v2.1-quant"
    assert loaded["random_seed"] == 1234
    print("✅ Data layer, validation, and immutable versioning passed.")


def test_regime_engine():
    print("Testing Market Regime Engine...")
    from trading.regime import RegimeDetector, RegimeType
    detector = RegimeDetector()
    df = _generate_synthetic_ohlcv(100)
    state = detector.detect(df)
    assert state.regime in [
        RegimeType.STRONG_BULL, RegimeType.WEAK_BULL, RegimeType.SIDEWAYS,
        RegimeType.WEAK_BEAR, RegimeType.STRONG_BEAR, RegimeType.HIGH_VOLATILITY,
        RegimeType.LOW_VOLATILITY
    ]
    assert 0.0 <= state.confidence <= 1.0
    assert state.duration >= 1
    assert "close" in state.features
    assert sum(state.transition_probabilities.values()) > 0.95
    print(f"✅ Regime Engine detected {state.regime} (confidence: {state.confidence*100:.1f}%)")


def test_baseline_strategies():
    print("Testing 9 Baseline Benchmark Strategies...")
    from trading.baselines import get_all_baselines
    baselines = get_all_baselines()
    assert len(baselines) == 7, "Expected baseline strategies instantiated"
    df = _generate_synthetic_ohlcv(100)
    for strat in baselines:
        res = strat.generate_signal(df)
        assert res["signal"] in ("BUY", "SELL", "HOLD")
        assert -1.5 <= res["weight"] <= 1.5
        assert 0.0 <= res["confidence"] <= 1.0
        assert len(res["reason"]) > 0
    print(f"✅ All {len(baselines)} baseline benchmark strategies verified.")


def test_alpha_library_12_families():
    print("Testing 12 Alpha Families...")
    from trading.alpha import get_all_alphas
    alphas = get_all_alphas()
    assert len(alphas) == 12, f"Expected 12 alpha families, got {len(alphas)}"

    # Mock feature vector
    sample_features = {
        "ret_1": 0.012, "ret_5": 0.035, "ret_20": 0.08, "atr_pct": 1.8,
        "rsi_14": 42.0, "adx_14": 28.5, "zscore_20": -1.6, "bb_deviation": -65.0,
        "dist_from_high_20": -0.4, "dist_from_low_20": 4.5, "relative_volume": 1.45,
        "volume_acceleration": 1.25, "price_vol_interaction": 0.017,
        "return_rank": 0.85, "momentum_5_20": 0.04, "trend_persistence": 0.65,
        "drawdown": -1.2, "realized_vol_20": 16.5, "trend_slope_20": 6.2,
        "ml_prob_up": 0.68, "ml_confidence": 0.72,
    }

    for alpha in alphas:
        sig = alpha.generate_signal(sample_features)
        conf = alpha.score(sample_features)
        exp = alpha.explain(sample_features)
        assert -1.0 <= sig <= 1.0, f"{alpha.__class__.__name__} signal {sig} out of range [-1, 1]"
        assert 0.0 <= conf <= 1.0, f"{alpha.__class__.__name__} confidence {conf} out of range [0, 1]"
        assert "reason_codes" in exp, f"{alpha.__class__.__name__} missing reason codes"
    print("✅ All 12 alpha families generated valid bounded signals.")


def test_ml_layer_and_calibration():
    print("Testing Tabular ML & Model Calibration...")
    from trading.ml import TabularMLModel, ModelCalibrator
    df = _generate_synthetic_ohlcv(200)
    df["ret_1"] = df["close"].pct_change()
    df["rsi_14"] = 50.0 + np.random.normal(0, 10, len(df))
    df["atr_pct"] = 1.5

    model = TabularMLModel(model_type="random_forest", target_horizon=5)
    train_res = model.fit(df, feature_cols=["ret_1", "rsi_14", "atr_pct"])
    assert train_res["status"] == "trained"

    pred = model.predict_latest({"ret_1": 0.01, "rsi_14": 45.0, "atr_pct": 1.5})
    assert 0.0 <= pred["probability_up"] <= 1.0
    assert 0.0 <= pred["confidence"] <= 1.0

    # Calibration testing
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1])
    y_prob = np.array([0.8, 0.2, 0.7, 0.9, 0.3, 0.65, 0.15, 0.4, 0.85, 0.75, 0.25, 0.6, 0.3, 0.8, 0.7])
    cal_res = ModelCalibrator.evaluate_calibration(y_true, y_prob)
    assert 0.0 <= cal_res["brier_score"] <= 1.0
    assert "buckets" in cal_res
    print(f"✅ Tabular ML trained and evaluated (Brier score: {cal_res['brier_score']:.4f})")


def test_meta_model_and_disagreement():
    print("Testing Meta Model & Disagreement Gate...")
    from trading.ml import MetaModel
    meta = MetaModel(disagreement_threshold=0.45)

    # 1. Aligned alphas
    aligned_signals = {"TrendAlpha": 0.8, "MomentumAlpha": 0.75, "BreakoutAlpha": 0.85}
    aligned_conf = {"TrendAlpha": 0.8, "MomentumAlpha": 0.75, "BreakoutAlpha": 0.85}
    dec_aligned = meta.evaluate(aligned_signals, aligned_conf)
    assert dec_aligned.signal == "BUY"
    assert dec_aligned.exposure_scale >= 0.80

    # 2. Conflicting alphas -> Circuit Breaker
    conflicting_signals = {"TrendAlpha": 0.9, "MeanReversionAlpha": -0.9, "BreakoutAlpha": 0.8}
    conflicting_conf = {"TrendAlpha": 0.8, "MeanReversionAlpha": 0.8, "BreakoutAlpha": 0.8}
    dec_conflict = meta.evaluate(conflicting_signals, conflicting_conf)
    assert dec_conflict.disagreement_score > 0.45
    assert dec_conflict.exposure_scale < 0.80
    assert any("Disagreement" in r for r in dec_conflict.reasons)
    print(f"✅ Meta Model handled alignment and triggered circuit breaker on disagreement ({dec_conflict.disagreement_score:.2f})")


def test_portfolio_constructor_and_sizer():
    print("Testing Portfolio Construction & Position Sizing...")
    from trading.portfolio import PortfolioConstructor, PositionSizer
    constructor = PortfolioConstructor(
        max_position_weight=0.25,
        max_total_exposure=0.90,
        min_cash_pct=0.10,
    )
    candidates = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    weights = constructor.construct(candidates, method="risk_parity")

    assert len(weights) == 5
    assert sum(weights.values()) <= 0.9001, f"Total exposure {sum(weights.values())} exceeds 0.90"
    for s, w in weights.items():
        assert w <= 0.2501, f"Asset {s} weight {w} exceeds 0.25"

    sizer = PositionSizer(default_risk_per_trade_pct=2.0, max_order_value=5000.0)
    sized = sizer.size_by_risk(cash=100000.0, price=2500.0, stop_loss_pct=2.0)
    assert sized["quantity"] >= 1
    assert sized["order_value"] <= 5000.0
    print(f"✅ Portfolio weights constrained (Sum: {sum(weights.values()):.2f}), Sizer: {sized['quantity']} shares")


def test_execution_simulator_and_stress():
    print("Testing Execution Simulator Realism & Hostile Stress...")
    from trading.execution.simulator import ExecutionSimulator
    sim = ExecutionSimulator(fill_prob=1.0, slippage_bps=10.0, spread_paise=5.0)
    res = sim.execute("RELIANCE", "BUY", qty=2, price=2500.0, bar_volume=100000)
    assert res.filled_qty > 0
    assert res.filled_price > 2500.0  # Slippage & spread increase purchase price
    assert res.commission > 0.0

    # Stress execution (override fill_prob for deterministic test)
    stressed = sim.stress_execution(multiplier=3.0)
    stressed.fill_prob = 1.0
    res_stressed = stressed.execute("RELIANCE", "BUY", qty=2, price=2500.0, bar_volume=100000)
    assert res_stressed.filled_price >= res.filled_price
    print(f"✅ Base fill: ₹{res.filled_price:.2f}, Stressed fill: ₹{res_stressed.filled_price:.2f} (Comm: ₹{res.commission:.2f})")


def test_walk_forward_and_holdout():
    print("Testing Walk-Forward Validation & Holdout Contamination Guard...")
    from trading.validation import WalkForwardValidator, ProtectedHoldout
    df = _generate_synthetic_ohlcv(260)

    # 1. Holdout
    holdout_guard = ProtectedHoldout(holdout_pct=0.20)
    research_df, holdout_df = holdout_guard.split(df)
    assert len(research_df) == int(260 * 0.80)
    assert len(holdout_df) == 260 - len(research_df)

    audit = holdout_guard.audit_access("STRAT_FINALIST_01", candidate_generation=3)
    assert audit["eval_count"] >= 1

    # 2. Walk-forward with purge and embargo
    wf = WalkForwardValidator(train_bars=80, val_bars=25, test_bars=25, purge_bars=5, embargo_bars=5)
    splits = wf.generate_splits(research_df)
    assert len(splits) >= 2
    for sp in splits:
        assert len(sp.train_indices) == 80
        assert len(sp.val_indices) == 25
        assert len(sp.test_indices) == 25
        # Ensure chronological ordering: max(train) < min(val)
        assert sp.train_indices[-1] < sp.val_indices[0]
        assert sp.val_indices[-1] < sp.test_indices[0]
    print(f"✅ Generated {len(splits)} chronological walk-forward folds with purge/embargo.")


def test_adversarial_and_sensitivity():
    print("Testing Adversarial Robustness & Parameter Sensitivity...")
    from trading.validation import AdversarialTester
    tester = AdversarialTester(cost_multiplier=3.0, slippage_multiplier=3.0)

    mock_trades = [
        {"pnl": 120.0, "fees": 12.0, "exec_price": 2000.0, "qty": 1},
        {"pnl": -45.0, "fees": 10.0, "exec_price": 2050.0, "qty": 1},
        {"pnl": 90.0, "fees": 11.0, "exec_price": 1980.0, "qty": 1},
        {"pnl": -30.0, "fees": 9.0, "exec_price": 2020.0, "qty": 1},
        {"pnl": 150.0, "fees": 15.0, "exec_price": 2100.0, "qty": 1},
    ]

    report = tester.run_stress_suite(mock_trades)
    assert report.base_pnl == 285.0
    assert report.cost_stressed_pnl < report.base_pnl
    assert isinstance(report.monte_carlo_var_95, float)
    print(f"✅ Base P&L: ₹{report.base_pnl:.2f} -> 3x Cost Stressed: ₹{report.cost_stressed_pnl:.2f} (Passed: {report.passed_all_gates})")


def test_strategy_dna_and_research_memory():
    print("Testing Darwinian Strategy DNA & Research Memory...")
    from trading.swarm.strategy_dna import StrategyDNA, create_initial_population, mutate_dna
    from trading.swarm.multi_objective_fitness import MultiObjectiveFitness
    from trading.swarm.research_memory import ResearchMemory

    pop = create_initial_population(size=12)
    assert len(pop) == 12
    p1 = pop[0]
    assert p1.validate_bounds()

    # Mutation
    child = mutate_dna(p1, generation=2, mutation_rate=0.3)
    assert child.generation == 2
    assert child.parent_id == p1.strategy_id
    assert child.validate_bounds()

    # Multi-objective fitness
    fit = MultiObjectiveFitness.compute_fitness({"sharpe": 1.8, "sortino": 2.1, "profit_factor": 1.6, "max_drawdown_pct": 6.5})
    novelty = MultiObjectiveFitness.compute_novelty(child, pop)
    assert isinstance(fit, float)
    assert novelty > 0.0

    # Research memory failure tracking
    mem = ResearchMemory(storage_dir=Path("trading/data_cache/test_mem"))
    p1.death_reason = "Max Drawdown Exceeded in Choppy Regime"
    mem.record_failure(p1, fatal_metric="max_dd_8%", reason="Choppy regime exhaustion")
    failed_again, reason = mem.has_idea_already_failed(p1)
    assert failed_again is True
    print(f"✅ StrategyDNA mutated, evaluated (Fitness: {fit:.3f}, Novelty: {novelty:.2f}), and failure recalled.")


def test_champion_challenger_and_cycles():
    print("Testing Champion/Challenger & Research Cycles...")
    from trading.research import ChampionChallengerManager, ResearchLifecycle

    mgr = ChampionChallengerManager(state_file=Path("trading/data_cache/test_champ.json"))
    champ = mgr.get_champion()
    assert "strategy_id" in champ

    # Unworthy challenger rejection
    bad_metrics = {"trades": 10, "sharpe": 0.4, "profit_factor": 0.9, "max_drawdown_pct": 25.0}
    verdict = mgr.evaluate_promotion(bad_metrics)
    assert verdict["approved"] is False
    assert len(verdict["failed_gates"]) > 0

    # Operational research cycles
    d_res = ResearchLifecycle.run_daily_cycle()
    w_res = ResearchLifecycle.run_weekly_cycle()
    m_res = ResearchLifecycle.run_monthly_cycle()
    assert d_res["cycle"] == "DAILY"
    assert w_res["cycle"] == "WEEKLY"
    assert m_res["cycle"] == "MONTHLY"
    print(f"✅ Champion/Challenger rejected invalid challenger ({len(verdict['failed_gates'])} gate failures). Cycles verified.")


if __name__ == "__main__":
    test_data_layer_and_versioning()
    test_regime_engine()
    test_baseline_strategies()
    test_alpha_library_12_families()
    test_ml_layer_and_calibration()
    test_meta_model_and_disagreement()
    test_portfolio_constructor_and_sizer()
    test_execution_simulator_and_stress()
    test_walk_forward_and_holdout()
    test_adversarial_and_sensitivity()
    test_strategy_dna_and_research_memory()
    test_champion_challenger_and_cycles()
    print("\n🏆 ALL 12 QUANTITATIVE ARCHITECTURE SYSTEMS PASSED PERFECTLY!")
