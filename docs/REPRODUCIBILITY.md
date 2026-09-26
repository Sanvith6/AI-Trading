# Deterministic Quantitative Reproducibility Protocol

## Provenance Tuple
Every published research experiment produces an immutable bundle containing:
$$\langle \text{Git Commit SHA}, \text{Dataset SHA256}, \text{Feature Registry Hash}, \text{Seed}, \text{Package Manifest} \rangle$$

## Verification via CLI
```python
from trading.research.reproducibility import ReproducibilityEngine

repro = ReproducibilityEngine()
result = repro.reproduce_experiment(
    experiment_id="EXP_MOMENTUM_2024",
    reproduced_metrics={"sharpe": 1.78, "cagr": 0.385},
    tolerance_pct=0.02
)
assert result["reproduced"] is True
```
If code changes alter backtest math beyond $2\%$ tolerance, reproduction fails with an explicit diff report.
