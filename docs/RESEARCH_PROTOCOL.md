# RESEARCH PROTOCOL — How This Platform Conducts Research

> Every trading strategy treated as a scientific experiment.
> No quiet changes after seeing results. Every experiment versioned.

## 1. Core Discipline

```
HYPOTHESIS → EXPERIMENT → RESULTS → REVIEW → NEXT HYPOTHESIS
```

**No strategy starts without a hypothesis.**
**No hypothesis changes after seeing results.**
**If anything changes, create a new experiment version.**

## 2. Hypothesis Template

Every experiment begins with:

```markdown
HYPOTHESIS: "MEDIUM-TERM cross-sectional momentum MAY PERFORM BETTER
after volatility-normalization DURING moderate-trend regimes."

expected mechanism: Why this should work (economic rationale)
features: Which features this uses
entry logic: When to enter
exit logic: When to exit
risk logic: How risk is controlled
failure conditions: When to stop testing this
test period: Data window
baseline: Which baseline strategy to compare (SMA_TREND, BUY_AND_HOLD, etc.)
```

The ResearchAgent (autonomous LLM agent, runs daily/weekly/monthly)
generates new hypotheses by inspecting results, detecting anomalies,
reviewing failures, and recommending experiments.

## 3. Research Data Splits (Strict)

```
RAW DATA ──► TRAIN ──► VALIDATE ──► TEST ──► HOLDOUT ──► PAPER
```

**Strict chronological ordering. NEVER shuffle.**

- **RAW**: All data, untouched
- **TRAIN**: Strategy/parameter discovery
- **VALIDATE**: Signal threshold tuning, feature selection
- **TEST**: Final performance estimate (strict OOS)
- **HOLDOUT**: Protected permanently. Evolutionary system NEVER optimizes against it.
  - Only finalists tested against holdout
  - If holdout used repeatedly → contamination flag
  - LLM must NOT inspect holdout details during training cycles
- **PAPER**: Real market, no real money, simulated fills

## 4. Experiment Versioning

Every experiment records (REQUIRED):

```python
experiment = {
    "experiment_id": "uuid",
    "git_commit": "hash",
    "code_version": "version",
    "config": {...},  # complete snapshot
    "environment": {...},  # Python version, packages
    "random_seed": 42,
    "dataset_version": "hash_of_data",
    "universe_version": "v1",
    "feature_version": "v3",
    "strategy_version": "v2",
    "model_version": "v1",
    "hypothesis": "...",
    "baseline": "SMA_TREND",
    "results": {...},
    "status": "active | superseded | rejected | promoted"
}
```

## 5. Research Cadence

### DAILY
- Data validation (freshness, gaps, staleness)
- Paper performance summary
- Risk monitoring (drawdown, exposure, daily loss)
- Drift monitoring (feature, prediction, signal, regime)
- Anomaly detection (ResearchAgent)

### WEEKLY
- Strategy diagnostics (is each strategy still functioning?)
- Error analysis (why did losses occur?)
- Regime analysis (which regimes hurt?)
- Correlation analysis (are strategies overlapping?)
- ResearchAgent report generation

### MONTHLY
- Full evolutionary research cycle (new generation)
- Retraining (if ML models exist)
- New feature/alpha experiments
- Walk-forward evaluation
- Promotion review (paper → shadow candidates)
- ResearchAgent synthesis report

## 6. Research Agent (Autonomous)

The ResearchAgent is an LLM agent with a specific personality: **skeptical senior quant**.

Responsibilities:
1. Inspect results for anomalies
2. Detect statistical tricks (data peeking, cherry-picking)
3. Generate falsifiable hypotheses
4. Propose new experiments (maximize information gain)
5. Review strategy failures for root causes
6. Identify potential data leakage
7. Recommend new features/alphas to test
8. Produce weekly/monthly research reports
9. Check whether the system is optimizing against the holdout
10. Flag any experiment that shows suspicious returns

ResearchAgent memory (structured, not chat history):
- What worked, under which regime, with which data
- What failed, why, when, under which assumptions
- Which features are informative, which are noise
- Which strategies are correlated, which are diverse

## 7. Experiment Prioritization

Not all experiments are equal. Prioritize by expected information gain:

**HIGH priority** (run first):
- High-uncertainty questions (untested regimes)
- Important failure modes (cost stress, gap stress)
- Unexplored strategy families (new alpha type)
- Useful feature combinations (orthogonal hypotheses)

**LOW priority** (avoid):
- Repeatedly testing tiny parameter changes
- Experiments likely to confirm existing knowledge
- Testing without a falsifiable hypothesis

## 8. Robustness Validation Schedule

For every candidate strategy, run:

**Every experiment:**
- Walk-forward (rolling + expanding)
- Parameter sensitivity (±20% on each parameter)
- Cost stress (2× default costs)
- Minimum trade count check (≥30)

**Every weekly review:**
- Regime isolation (bull/bear/sideways separately)
- Correlation analysis against existing strategies
- Drift detection check

**Every monthly review:**
- Monte Carlo (1000 shuffled sequences)
- Crash period analysis (2020, 2022 bear)
- Adversarial stress (3× worse execution)
- Permanent holdout test (if finalist)
- Complexity audit (is a simpler model almost as good?)

## 9. Reporting Requirements

After every experiment: `experiments/<experiment_id>.md`

After every generation: `reports/generation_<N>.md`

Generation report includes:
- Population summary (species distribution)
- Survivors (who passed, who died)
- Dead strategies (with failure reasons)
- Champion (with full metrics)
- Parameter sensitivity scores
- Correlation matrix summary
- Novelty scores
- Paper performance if available
- WHY strategies survived/died
- WHAT the system learned
- WHAT should be tested next

## 10. What Never Changes Mid-Experiment

The following CANNOT be changed after an experiment starts seeing results:
- Data source/snapshot
- Features (without new version)
- Cost model assumptions
- Evaluation period
- Benchmark comparison
- Risk limits
- Random seed (if reproducible results desired)

If ANYTHING changes → new experiment version, documented hypothesis.

## 11. Evidence Standard

A strategy must demonstrate:
1. **Performance**: Positive OOS returns with 30+ trades
2. **Stability**: Rolling Sharpe doesn't drop below 0
3. **Robustness**: Survives all stress tests
4. **Diversification**: Adds uncorrelated return stream
5. **Calibration**: Confidence matches hit rate
6. **Reproducibility**: Same seed → same results
7. **Reasonable complexity**: Simpler model not almost as good

**"Survived hostile evaluation"** is the promotion criterion.
**"Made the most money"** is NOT.

---

*The ResearchAgent should continually ask: "What evidence do we have
that this works, and what evidence would prove it wrong?"*
