# MLOps Experiment Tracking & Model Governance

## Lineage Ledger
Every quantitative experiment is recorded in `data_lake/mlops/experiment_runs.json` (and mirrored to local MLflow when available) with:
- `experiment_id` & `run_id`
- `strategy_id` & `strategy_dna`
- `model_name` & `parameters`
- `dataset_version` & `feature_version`
- `git_commit_sha` & `seed`
- `training_period` & `validation_period`
- `metrics` (Sharpe, CAGR, Sortino, Max Drawdown)
- `promotion_state`: `CANDIDATE` -> `CHALLENGER` -> `CHAMPION` -> `RETIRED`

## Holdout Protection Rule
Optuna Bayesian hyperparameter search is programmatically restricted to training and validation splits. Any script passing holdout dates to `OptunaTuner` triggers an immediate assertion error.
