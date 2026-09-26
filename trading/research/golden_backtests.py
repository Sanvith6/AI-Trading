"""
Golden Backtest Regression Suite & Integrity Guard.

Maintains an immutable repository of quantitative golden baselines across fixed:
- Instruments (e.g., SPY, AAPL, NVDA, MSFT)
- Historical Periods (e.g., 2022 Bear Market, 2023 Bull Market, 2024 Expansion)
- Strategy DNA configurations
- Execution Seeds

For every code change, compares newly computed metrics (CAGR, Sharpe, Sortino, Calmar,
Max Drawdown, Trade Count, Turnover) against frozen golden benchmarks.
Triggers REGRESSION_WARNING if any variation exceeds statistical tolerance bounds.
Prevents future refactors from silently degrading trading performance.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN_BASELINES_PATH = Path(__file__).parent.parent.parent / "data_lake" / "golden_baselines.json"


@dataclass
class GoldenBenchmark:
    benchmark_id: str
    strategy_id: str
    symbols: List[str]
    period: Dict[str, str]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    seed: int
    expected_metrics: Dict[str, float]
    metric_tolerances_pct: Dict[str, float]  # e.g., {"sharpe": 0.05, "cagr": 0.05, "max_drawdown": 0.05}
    last_frozen_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RegressionReport:
    all_passed: bool
    benchmarks_evaluated: int
    failed_benchmarks: List[str]
    regression_warnings: List[Dict[str, Any]]
    detailed_comparisons: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GoldenBacktestSuite:
    """
    Manages golden backtest regression verification.
    """

    # Pre-calibrated standard institutional benchmark baselines
    CORE_BASELINES: List[GoldenBenchmark] = [
        GoldenBenchmark(
            benchmark_id="GOLDEN_MOMENTUM_US_2023",
            strategy_id="CROSS_SECTIONAL_MOMENTUM",
            symbols=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"],
            period={"start": "2023-01-01", "end": "2023-12-31"},
            seed=42,
            expected_metrics={
                "cagr": 0.385,
                "sharpe": 1.78,
                "sortino": 2.45,
                "calmar": 2.65,
                "max_drawdown": 0.145,
                "trade_count": 68.0,
                "turnover": 6.8,
                "final_equity": 138500.0,
            },
            metric_tolerances_pct={
                "sharpe": 0.05,
                "cagr": 0.05,
                "max_drawdown": 0.05,
                "trade_count": 0.02,
                "turnover": 0.05,
            },
            last_frozen_at="2024-01-15T00:00:00Z"
        ),
        GoldenBenchmark(
            benchmark_id="GOLDEN_VOL_ARBITRAGE_US_2022",
            strategy_id="DEFENSIVE_VOLATILITY",
            symbols=["SPY", "QQQ", "IWM", "XLP", "XLU"],
            period={"start": "2022-01-01", "end": "2022-12-31"},
            seed=42,
            expected_metrics={
                "cagr": 0.085,
                "sharpe": 0.95,
                "sortino": 1.35,
                "calmar": 1.05,
                "max_drawdown": 0.081,
                "trade_count": 42.0,
                "turnover": 4.2,
                "final_equity": 108500.0,
            },
            metric_tolerances_pct={
                "sharpe": 0.05,
                "cagr": 0.05,
                "max_drawdown": 0.05,
                "trade_count": 0.05,
            },
            last_frozen_at="2023-01-15T00:00:00Z"
        ),
    ]

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or DEFAULT_GOLDEN_BASELINES_PATH
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.baselines: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.storage_path.exists():
            try:
                return json.loads(self.storage_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Failed to load golden baselines: {e}")

        # Seed defaults
        data = {b.benchmark_id: b.to_dict() for b in self.CORE_BASELINES}
        self._save(data)
        return data

    def _save(self, data: Dict[str, Dict[str, Any]]) -> None:
        try:
            self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save golden baselines: {e}")

    def freeze_baseline(self, benchmark: GoldenBenchmark) -> None:
        """Register or freeze a new quantitative golden benchmark."""
        self.baselines[benchmark.benchmark_id] = benchmark.to_dict()
        self._save(self.baselines)
        logger.info(f"Frozen golden baseline: {benchmark.benchmark_id}")

    def evaluate_regression(
        self,
        benchmark_id: str,
        current_metrics: Dict[str, float],
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Evaluate current metrics against a frozen benchmark.
        Returns: (passed, regression_warnings, comparison_dict)
        """
        if benchmark_id not in self.baselines:
            raise KeyError(f"Unknown golden benchmark: {benchmark_id}")

        bm = GoldenBenchmark(**self.baselines[benchmark_id])
        expected = bm.expected_metrics
        tolerances = bm.metric_tolerances_pct
        default_tol = 0.05

        warnings = []
        comparisons = {}
        all_ok = True

        for metric, exp_val in expected.items():
            curr_val = current_metrics.get(metric)
            if curr_val is None:
                warnings.append(f"Metric '{metric}' missing in current run!")
                all_ok = False
                continue

            tol = tolerances.get(metric, default_tol)
            diff = curr_val - exp_val
            pct_diff = abs(diff) / max(abs(exp_val), 1e-4)

            # Check directional degradation:
            # For Sharpe, CAGR, Sortino, Calmar: lower is bad
            # For Max Drawdown: higher is bad
            is_regression = False
            if metric in ("sharpe", "cagr", "sortino", "calmar", "final_equity") and diff < 0 and pct_diff > tol:
                is_regression = True
            elif metric == "max_drawdown" and diff > 0 and pct_diff > tol:
                is_regression = True
            elif metric in ("trade_count", "turnover") and pct_diff > (tol * 2):
                is_regression = True

            if is_regression:
                all_ok = False
                msg = f"REGRESSION WARNING: {metric} deviated by {pct_diff*100:.1f}% (Expected: {exp_val}, Actual: {curr_val}, Tol: {tol*100:.1f}%)"
                warnings.append(msg)

            comparisons[metric] = {
                "expected": exp_val,
                "actual": round(float(curr_val), 4),
                "diff": round(float(diff), 4),
                "pct_diff": round(float(pct_diff * 100.0), 2),
                "tolerance_pct": round(float(tol * 100.0), 1),
                "passed": not is_regression,
            }

        return all_ok, warnings, comparisons

    def run_suite(self, results_by_benchmark: Dict[str, Dict[str, float]]) -> RegressionReport:
        """Run complete golden backtest regression verification across all submitted results."""
        failed = []
        all_warnings = []
        details = {}

        for b_id, metrics in results_by_benchmark.items():
            if b_id in self.baselines:
                ok, warns, comp = self.evaluate_regression(b_id, metrics)
                details[b_id] = comp
                if not ok:
                    failed.append(b_id)
                    all_warnings.extend([f"[{b_id}] {w}" for w in warns])

        return RegressionReport(
            all_passed=len(failed) == 0,
            benchmarks_evaluated=len(results_by_benchmark),
            failed_benchmarks=failed,
            regression_warnings=all_warnings,
            detailed_comparisons=details,
        )
