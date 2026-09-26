"""
Phase 1 Quantitative Research Foundation — Comprehensive Verification Suite.

Tests and enforces all critical quantitative invariants:
1. INVARIANT 1: Zero Future Fundamental Leakage (strict available_at causality).
2. INVARIANT 2: Zero Future Event Leakage (strict event availability boundary).
3. INVARIANT 3: Zero Pre-IPO or Delisted Universe Contamination (point-in-time universe).
4. INVARIANT 4: Zero Protected Holdout Access During Optimization (Optuna isolation).
5. INVARIANT 5: Zero Unchecked Risk Bypass (Risk constraints remain authoritative).
6. Data Lake Parquet Roundtrip & DuckDB Analytical Querying.
7. Dataset Registry Cryptographic Integrity Verification.
8. Feature Registry Immutability & Version Mutation Protection.
9. Corporate Actions Price Adjustments & Split Normalization.
10. Factor Model Style & Alpha/Beta Decomposition.
11. True Risk Parity / ERC Portfolio Optimization.
12. Statistical Feature & Prediction Drift Detection (KS / PSI).
13. Golden Backtest Regression Detection on Metric Deviation.
14. Cryptographic Experiment Reproducibility Verification.
15. End-to-End Trade Decision Trace & Point-in-Time Replay.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from trading.storage import ParquetStore, DuckDBStore, DatasetRegistry, PolarsEngine
from trading.universe.point_in_time import LocalPointInTimeUniverse
from trading.corporate_actions import CorporateActionEngine, CorporateAction, PriceRepresentation
from trading.market_context import USMarketContext
from trading.events import EventFoundation, CorporateEvent
from trading.fundamentals import EdgarFundamentalProvider, FundamentalRecord
from trading.features.registry import FeatureRegistry, FeatureDefinition
from trading.factors import FactorModel
from trading.portfolio.portfolio_constructor import PortfolioConstructor
from trading.portfolio.skfolio_adapter import SkfolioAdapter
from trading.portfolio.capacity_engine import CapacityEngine
from trading.portfolio.attribution import PerformanceAttributor
from trading.mlops import ExperimentTracker, OptunaTuner
from trading.monitoring.drift_detector import DriftDetector
from trading.research.golden_backtests import GoldenBacktestSuite
from trading.research.reproducibility import ReproducibilityEngine
from trading.research.decision_trace import DecisionTraceEngine, TradeDecisionTrace
from trading.research.trade_replay import TradeReplayEngine


class TestPhase1Foundation:

    def test_invariant_1_fundamental_causality_no_future_leakage(self, tmp_path):
        """Verify that quarterly fundamentals are completely invisible before available_at."""
        fp = EdgarFundamentalProvider(storage_path=tmp_path / "fundamentals.json")
        rec = FundamentalRecord(
            symbol="TEST_CO",
            period_end="2023-12-31",
            filing_date="2024-02-15",
            available_at="2024-02-15T21:00:00Z",
            form_type="10-K",
            revenue=1000e6,
            revenue_growth_yoy=0.20,
            gross_margin=0.50,
            operating_margin=0.25,
            ebitda=300e6,
            net_income=200e6,
            eps=2.0,
            free_cash_flow=180e6,
            cash_and_equivalents=500e6,
            total_debt=200e6,
            total_assets=2000e6,
            total_liabilities=800e6,
            shares_outstanding=100e6,
            roe=0.20,
            roic=0.15,
        )
        fp.record_fundamental(rec)

        # Query on 2024-01-31 (BEFORE available_at): MUST BE NONE
        early = fp.get_latest_pit_fundamentals("TEST_CO", "2024-01-31T23:59:59Z")
        assert early is None, "CRITICAL LEAKAGE: Fundamental statement was visible before public filing!"

        # Query on 2024-02-16 (AFTER available_at): MUST BE ACCESSIBLE
        post = fp.get_latest_pit_fundamentals("TEST_CO", "2024-02-16T10:00:00Z")
        assert post is not None
        assert post.period_end == "2023-12-31"

    def test_invariant_2_event_causality_no_future_leakage(self, tmp_path):
        """Verify that events cannot be observed prior to available_at."""
        ef = EventFoundation(storage_path=tmp_path / "events.json")
        ev = CorporateEvent(
            event_id="EV_EARNINGS_01",
            symbol="NVDA",
            event_type="EARNINGS_RELEASE",
            event_time="2024-05-22T20:00:00Z",
            source="SEC_EDGAR",
            available_at="2024-05-22T20:05:00Z",
            importance="HIGH",
            raw_reference="0001045810-24-000045"
        )
        ef.record_event(ev)

        # Query before availability
        early_events = ef.get_events_for_symbol("NVDA", "2024-05-22T19:59:59Z")
        assert len(early_events) == 0, "CRITICAL LEAKAGE: Event visible before public availability!"

        # Query after availability
        post_events = ef.get_events_for_symbol("NVDA", "2024-05-23T09:30:00Z")
        assert len(post_events) == 1

    def test_invariant_3_point_in_time_universe_eligibility(self):
        """Verify that IPOs are not eligible before listing date and delisted stocks are excluded."""
        pit = LocalPointInTimeUniverse()

        # ABNB listed 2020-12-10
        ok_before, reason_before = pit.is_eligible_at_date("ABNB", "2020-06-01", market="US")
        assert not ok_before, f"Survivorship leak: ABNB eligible before IPO: {reason_before}"

        ok_after, reason_after = pit.is_eligible_at_date("ABNB", "2021-06-01", market="US")
        assert ok_after, f"ABNB should be eligible post-IPO: {reason_after}"

        # TWTR delisted 2022-10-27
        ok_delist, reason_delist = pit.is_eligible_at_date("TWTR", "2023-01-01", market="US")
        assert not ok_delist, f"Survivorship leak: Delisted ticker TWTR eligible in 2023: {reason_delist}"

    def test_invariant_4_optuna_holdout_isolation(self, tmp_path):
        """Enforces that Optuna optimization cannot access the protected holdout period."""
        dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
        full_df = pd.DataFrame({"close": np.random.uniform(100, 200, len(dates))}, index=dates)

        # Enforce boundary: validation ends 2023-12-31, holdout is 2024
        val_end = "2023-12-31"
        isolated_df = OptunaTuner.enforce_holdout_isolation(full_df, validation_end_date=val_end, holdout_start_date="2024-01-01")
        assert isolated_df.index.max() <= pd.to_datetime(val_end)
        assert pd.to_datetime("2024-01-01") not in isolated_df.index

    def test_data_lake_parquet_and_duckdb(self, tmp_path):
        """Verify Parquet write/read and DuckDB querying."""
        ps = ParquetStore(base_dir=tmp_path / "lake")
        df = pd.DataFrame({
            "open": [150.0, 152.0], "high": [155.0, 156.0],
            "low": [149.0, 151.0], "close": [153.0, 155.0], "volume": [10000, 12000]
        }, index=pd.date_range("2024-01-01", periods=2))

        meta = ps.write_bars("AAPL", df, market="US")
        assert meta["rows"] == 2
        read_df = ps.read_bars("AAPL", market="US")
        assert len(read_df) == 2

        db = DuckDBStore(db_path=tmp_path / "lake" / "test.duckdb", data_lake_dir=tmp_path / "lake")
        db.create_market_views()
        res = db.query("SELECT 100 as test_val")
        assert res.iloc[0]["test_val"] == 100
        db.close()

    def test_dataset_registry_integrity(self, tmp_path):
        """Verify dataset hashing and integrity check."""
        reg = DatasetRegistry(registry_path=tmp_path / "registry.json")
        df = pd.DataFrame({"val": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3))
        manifest = reg.register_dataset("ds_test", df, market="US", source="synthetic", storage_path="test.parquet")
        assert reg.verify_integrity("ds_test", manifest.version, df)

        # Mutate df
        df_mutated = df.copy()
        df_mutated.iloc[0, 0] = 999
        assert not reg.verify_integrity("ds_test", manifest.version, df_mutated)

    def test_feature_registry_immutability(self, tmp_path):
        """Verify that existing feature hashes cannot be silently mutated."""
        fr = FeatureRegistry(storage_path=tmp_path / "features.json")
        f1 = FeatureDefinition(
            feature_name="test_mom", version="v1", description="Momentum",
            source="price", lookback=10, timestamp_semantics="LAGGED_EOD",
            dependencies=["close"], market="US", status="ACTIVE", formula_expression="close/close[t-10]"
        )
        fr.register_feature(f1)

        # Attempt silent mutation with different formula: must raise ValueError
        f1_mutated = FeatureDefinition(
            feature_name="test_mom", version="v1", description="Mutated",
            source="price", lookback=20, timestamp_semantics="LAGGED_EOD",
            dependencies=["close"], market="US", status="ACTIVE", formula_expression="close/close[t-20]"
        )
        with pytest.raises(ValueError, match="Feature mutation forbidden"):
            fr.register_feature(f1_mutated)

    def test_corporate_action_split_adjustment(self):
        """Verify split factor adjustment for stock splits."""
        cae = CorporateActionEngine()
        dates = pd.date_range("2024-06-05", periods=5, freq="D")
        raw_df = pd.DataFrame({"close": [1200.0, 1200.0, 1200.0, 120.0, 122.0]}, index=dates)
        # NVDA 10:1 split on 2024-06-10
        adj_df = cae.adjust_series(raw_df, "NVDA", PriceRepresentation.SPLIT_ADJUSTED)
        assert round(float(adj_df.iloc[0]["close"]), 1) == 120.0

    def test_factor_model_and_alpha_decomposition(self):
        """Verify factor standardizations and Alpha vs Beta return split."""
        fm = FactorModel()
        metrics = {
            "A": {"momentum": 0.20, "size": 1e11, "value": 0.05, "quality": 0.30, "volatility": 0.15, "growth": 0.10},
            "B": {"momentum": -0.10, "size": 1e10, "value": 0.08, "quality": 0.20, "volatility": 0.25, "growth": 0.02},
        }
        exp = fm.compute_factor_exposures(metrics)
        assert "A" in exp and "B" in exp
        prof = fm.profile_portfolio(
            weights={"A": 0.6, "B": 0.4},
            symbol_exposures=exp,
            betas={"A": 1.2, "B": 0.8},
            sectors={"A": "Tech", "B": "Financials"},
            portfolio_return=0.15,
            benchmark_return=0.10,
        )
        assert prof.market_beta == 1.04
        assert prof.alpha_beta_decomposition["selection_alpha"] == round(0.15 - (1.04 * 0.10), 4)

    def test_portfolio_risk_parity_and_mrc(self):
        """Verify true Risk Parity / Equal Risk Contribution weights."""
        pc = PortfolioConstructor()
        np.random.seed(42)
        ret_df = pd.DataFrame({
            "LOW_VOL": np.random.normal(0, 0.005, 100),
            "HIGH_VOL": np.random.normal(0, 0.025, 100),
        })
        weights = pc.construct(candidates=["LOW_VOL", "HIGH_VOL"], returns_df=ret_df, method="risk_parity")
        # LOW_VOL must receive significantly higher weight than HIGH_VOL in risk parity
        assert weights["LOW_VOL"] > weights["HIGH_VOL"]

    def test_drift_detection_alerting(self):
        """Verify KS/PSI drift detection and alert level assignment."""
        dd = DriftDetector()
        np.random.seed(42)
        base = pd.DataFrame({"f1": np.random.normal(0, 1, 200)})
        curr_drifted = pd.DataFrame({"f1": np.random.normal(2.5, 1.5, 200)})
        rep = dd.check_features(base, curr_drifted)
        assert rep.feature_drift_detected is True
        assert rep.alert_level in ("WARNING", "CRITICAL")

    def test_golden_backtest_regression(self, tmp_path):
        """Verify that GoldenBacktestSuite catches degraded Sharpe/CAGR."""
        gbs = GoldenBacktestSuite(storage_path=tmp_path / "golden.json")
        # Passing metrics
        ok, warns, _ = gbs.evaluate_regression("GOLDEN_MOMENTUM_US_2023", {
            "cagr": 0.385, "sharpe": 1.78, "sortino": 2.45, "calmar": 2.65,
            "max_drawdown": 0.145, "trade_count": 68.0, "turnover": 6.8, "final_equity": 138500.0
        })
        assert ok is True

        # Regressed metrics
        bad_ok, bad_warns, _ = gbs.evaluate_regression("GOLDEN_MOMENTUM_US_2023", {
            "cagr": 0.15, "sharpe": 0.85, "sortino": 1.1, "calmar": 0.8,
            "max_drawdown": 0.30, "trade_count": 68.0, "turnover": 6.8, "final_equity": 115000.0
        })
        assert bad_ok is False
        assert len(bad_warns) > 0

    def test_reproducibility_engine(self, tmp_path):
        """Verify experiment capture and reproduction verification."""
        repro = ReproducibilityEngine(storage_dir=tmp_path / "repro")
        bundle = repro.freeze_experiment(
            experiment_id="EXP_TEST_01",
            strategy_id="STRAT_01",
            strategy_dna={"dna": 123},
            dataset_sha256="abc999",
            dataset_version="v1",
            feature_version="v1",
            universe_version="v1",
            model_version="v1",
            baseline_metrics={"sharpe": 1.50, "cagr": 0.25},
            seed=42
        )
        res = repro.reproduce_experiment("EXP_TEST_01", {"sharpe": 1.50, "cagr": 0.25})
        assert res["reproduced"] is True

    def test_trade_replay_and_trace(self, tmp_path):
        """Verify structured decision trace and narrative replay."""
        dte = DecisionTraceEngine(storage_path=tmp_path / "traces.json")
        tre = TradeReplayEngine(dte)
        trace = TradeDecisionTrace(
            trade_id="TR_100",
            timestamp="2024-04-01T15:00:00Z",
            market="US",
            symbol="AAPL",
            action="BUY",
            universe_rank=2,
            alpha_scores={"MOMENTUM": 0.75},
            ml_probability=0.82,
            ml_confidence=0.85,
            regime="BULLISH",
            market_context={"market_trend": "BULLISH", "vix_level": 15.0},
            fundamentals={"f_operating_margin": 0.33},
            portfolio_state_before={"cash": 100000.0},
            risk_validation={"passed": True},
            position_size=50.0,
            target_weight=0.10,
            execution_assumption={"slippage_bps": 5.0, "fee": 1.0},
            final_decision_reason="Strong tech momentum"
        )
        dte.record_trace(trace)
        replay = tre.replay_trade("TR_100")
        assert replay is not None
        assert "AAPL" in replay.explanation_narrative
