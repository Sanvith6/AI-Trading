You are taking over the existing `ai-trading/` repository as the PRINCIPAL ARCHITECT of an institutional-grade quantitative research and autonomous paper-trading platform.

Your mission is to transform the repository from a capable AI trading project into a deeply engineered, research-driven, continuously improving quantitative trading laboratory with strict safety controls.

Do NOT optimize for a visually impressive demo.

Do NOT optimize for the highest historical return.

Do NOT assume that a profitable backtest represents a profitable real-world strategy.

Your primary objective is:

BUILD THE MOST ROBUST, GENERALIZABLE, DIVERSIFIED, RISK-CONTROLLED, DATA-DRIVEN, SELF-TESTING AND SELF-IMPROVING QUANTITATIVE TRADING SYSTEM POSSIBLE.

The system must attempt to discover strategies that survive hostile evaluation.

Think like:

* Quant researcher
* Portfolio manager
* ML researcher
* Statistical researcher
* Market microstructure researcher
* Risk manager
* Execution engineer
* MLOps engineer
* Software architect
* Security engineer
* Research auditor
* Evolutionary optimization scientist

The LLM is NOT the trader.

The LLM is the RESEARCH ORCHESTRATOR.

Numerical models, portfolio mathematics and hard risk constraints determine actual decisions.

============================================================
0. CORE PHILOSOPHY
==================

The system must follow these principles:

1. GENERALIZATION > BACKTEST PERFORMANCE
2. RISK-ADJUSTED RETURN > RAW RETURN
3. ROBUSTNESS > COMPLEXITY
4. DIVERSIFICATION > CONCENTRATION
5. OUT-OF-SAMPLE PERFORMANCE > IN-SAMPLE PERFORMANCE
6. REALISTIC EXECUTION > PERFECT EXECUTION ASSUMPTIONS
7. REPRODUCIBILITY > AD-HOC EXPERIMENTS
8. EVIDENCE > LLM OPINION
9. CAPITAL PRESERVATION > TRADING FREQUENCY
10. SIMPLE STRATEGIES MUST BEAT COMPLEX STRATEGIES ON EVIDENCE
11. EVERY STRATEGY MUST HAVE A FAILURE MODEL
12. EVERY MODEL MUST BE AUDITABLE
13. EVERY EVOLUTIONARY MUTATION MUST BE TRACEABLE
14. EVERY PROMOTION MUST BE REVERSIBLE
15. LIVE TRADING MUST NEVER BE THE DEFAULT

Never claim:

"guaranteed profit"
"100% accuracy"
"perfect strategy"
"risk-free"
"certain returns"

The system is a research platform, not a profit oracle.

============================================================

1. FIRST TASK — DEEPLY AUDIT THE EXISTING REPOSITORY
   ============================================================

Before changing architecture, inspect the entire repository.

Read:

README.md
TECHNOLOGIES.md
PROJECT_WALKTHROUGH.md
requirements.txt
.env.example

Inspect all modules.

Build a dependency map.

Understand:

Agent
TradingAgent
tools
data_provider
market_data
technical_analysis
order_manager
risk_manager
backtest
swarm
arena_manager
evolution_engine
genetic_ops
graveyard
quant_metrics
trading_app

Identify:

* architectural weaknesses
* duplicated logic
* hidden bugs
* data leakage
* look-ahead bias
* survivorship bias
* unrealistic execution assumptions
* overfitting risks
* weak statistical assumptions
* inadequate tests
* hard-coded limits
* brittle APIs
* excessive coupling
* unsafe execution pathways
* race conditions
* reproducibility problems
* model-version problems
* data-version problems

Do not rewrite everything blindly.

Create:

`docs/ARCHITECTURE_AUDIT.md`

containing:

CURRENT STATE
PROBLEMS
RISKS
PRIORITY
PROPOSED SOLUTION
MIGRATION PLAN

Classify every discovered issue:

CRITICAL
HIGH
MEDIUM
LOW

Fix CRITICAL issues first.

============================================================
2. TARGET SYSTEM
================

Build this architecture:

DATA SOURCES
↓
DATA INGESTION
↓
DATA VALIDATION
↓
DATA VERSIONING
↓
DYNAMIC MARKET UNIVERSE
↓
FEATURE ENGINE
↓
ALPHA LIBRARY
↓
REGIME ENGINE
↓
ML MODELS
↓
META MODEL
↓
ENSEMBLE
↓
PORTFOLIO CONSTRUCTION
↓
RISK ENGINE
↓
EXECUTION SIMULATOR
↓
PAPER BROKER
↓
PERFORMANCE ENGINE
↓
RESEARCH REGISTRY
↓
DARWINIAN EVOLUTION
↓
ROBUSTNESS GATES
↓
PAPER DEPLOYMENT
↓
SHADOW LIVE
↓
MANUAL PROMOTION GATE
↓
BROKER ADAPTER

The LLM sits ABOVE this architecture as the research orchestrator.

It does not bypass lower layers.

============================================================
3. DYNAMIC MARKET UNIVERSE
==========================

Destroy the assumption that the system should permanently watch:

RELIANCE
TCS
INFY

Those may remain examples.

The platform must dynamically discover eligible instruments.

Create:

`trading/universe/`

Support:

* equities
* ETFs
* indices
* sectors
* future asset-class expansion

Universe selection must be configurable.

Filters should include:

* instrument validity
* exchange
* price
* liquidity
* average traded value
* historical availability
* data completeness
* volume
* volatility
* corporate-action consistency
* stale-price detection
* suspicious data
* trading halts where detectable
* excessive illiquidity
* penny-stock suppression
* configurable exclusion lists

Support:

50
100
250
500
1000+

without changing strategy logic.

Use dynamic ranking rather than hard-coded watchlists.

Maintain:

`universe_version`

for reproducibility.

============================================================
4. DATA ARCHITECTURE
====================

Create:

`trading/data/`

Architecture:

BaseDataProvider
YahooDataProvider
AngelDataProvider
FutureProductionProvider

Every record must contain:

symbol
exchange
timestamp
timezone
source
interval
retrieval_timestamp
quality_status

Build:

data validator
data cache
data normalizer
data freshness checker
duplicate detector
missing-bar detector
OHLC sanity checker

Detect:

high < low
close outside expected ranges
zero/negative values
timestamp anomalies
duplicate candles
missing intervals
unexpected gaps
stale data
future timestamps
timezone errors

Never silently consume suspicious data.

============================================================
5. DATA VERSIONING
==================

Implement immutable experiment references.

Every experiment must identify:

dataset_version
universe_version
feature_version
strategy_version
model_version
code_version
random_seed

A backtest must be reproducible from recorded metadata.

============================================================
6. FEATURE ENGINE
=================

Create:

`trading/features/`

Build modular features.

PRICE:

returns
log returns
rolling returns
gap returns
range
ATR
realized volatility
drawdown
distance from highs/lows

TREND:

EMA
SMA
ADX
trend slope
trend persistence

MOMENTUM:

ROC
RSI
multi-horizon momentum
relative momentum
cross-sectional rank

MEAN REVERSION:

z-score
distance from moving average
Bollinger deviation
short-term reversal
residual reversion

VOLUME:

relative volume
volume acceleration
volume surprise
price-volume interaction

VOLATILITY:

realized volatility
ATR ratio
volatility compression
volatility expansion

MARKET:

index return
index volatility
breadth
sector momentum
market regime

CROSS-SECTIONAL:

rank
percentile
relative strength
sector-adjusted momentum
market-adjusted momentum

STATISTICAL:

beta
rolling correlation
residual return
correlation regime
volatility ratio

Every feature MUST be time-safe.

No feature may use information unavailable at decision time.

============================================================
7. ALPHA FACTORY
================

Create:

`trading/alpha/`

Implement independent alpha families:

Trend
Momentum
Mean Reversion
Breakout
Volatility
Volume/Flow
Cross-Sectional
Statistical
Relative Strength
Defensive
Market Regime
ML Hybrid

Every alpha must use a common interface:

generate_signal()
score()
explain()
required_features()
parameters()
validate()

Output:

signal ∈ [-1,+1]

plus:

confidence
expected_horizon
expected_return
expected_risk
regime_compatibility
reason_codes

Do not force BUY/SELL.

HOLD is a valid and often preferred decision.

============================================================
8. BASELINE STRATEGIES
======================

Before allowing AI evolution, create strong baselines:

BUY_AND_HOLD
SMA_TREND
EMA_TREND
MOMENTUM
MEAN_REVERSION
BREAKOUT
VOLATILITY_TARGET
EQUAL_WEIGHT
RISK_PARITY

The AI system must prove that complexity adds value over these baselines.

============================================================
9. REGIME ENGINE
================

Create:

`trading/regime/`

Detect:

strong bull
weak bull
sideways
weak bear
strong bear
high volatility
low volatility
risk-on
risk-off

Start with a transparent rule-based baseline.

Then optionally evaluate:

clustering
HMM-like approaches
probabilistic models
supervised regime classifier

Output:

regime
confidence
features
duration
transition_probability

Never assume a single strategy works in every regime.

============================================================
10. MACHINE LEARNING LAYER
==========================

Create:

`trading/ml/`

Start with robust tabular ML.

Evaluate:

logistic regression
random forest
gradient boosting
LightGBM/XGBoost-compatible models
calibrated classifiers
regression models

Predict:

probability_up
probability_down
expected_return
expected_volatility
confidence

The ML model must NOT directly control execution.

It generates information.

The portfolio/risk layer makes the final controlled decision.

============================================================
11. META MODEL
==============

Build an ensemble/meta-model combining:

technical features
alpha signals
ML predictions
regime information
cross-sectional ranking
risk estimates

Potential output:

expected_return
expected_risk
probability_positive
signal_strength
confidence

Measure model disagreement.

When models strongly disagree:

REDUCE EXPOSURE

or

HOLD.

Do not manufacture confidence.

============================================================
12. PORTFOLIO CONSTRUCTION
==========================

Create:

`trading/portfolio/`

The system must stop thinking:

"What stock should I buy?"

and start thinking:

"What portfolio has the best expected risk-adjusted opportunity?"

Support:

equal weight
score weight
confidence weight
inverse volatility
risk parity
volatility targeting

Constraints:

max position weight
max sector weight
max total exposure
max number positions
minimum cash
max correlated exposure
max turnover

Cluster positions by correlation.

Do not treat 20 highly correlated stocks as 20 independent bets.

============================================================
13. RISK ENGINE — IMMUTABLE SECURITY BOUNDARY
=============================================

The risk engine is stronger than the AI.

The AI CANNOT override:

max daily loss
max drawdown
max position size
max portfolio exposure
max sector exposure
max correlation exposure
max turnover
max order value
max number of trades
max leverage
minimum liquidity
stop-loss rules
emergency halt

Build independent validation:

AI decision
→ portfolio validation
→ risk validation
→ execution validation
→ broker adapter

The LLM is never allowed to skip this chain.

============================================================
14. POSITION SIZING
===================

Do not size positions simply as:

cash / number_of_stocks

Use risk-aware sizing.

Possible model:

position_size =
risk_budget / stop_distance

then enforce:

max_position_size
liquidity_limit
portfolio_limit
sector_limit
correlation_limit

Support:

fixed fractional risk
volatility targeting
inverse volatility
Kelly fraction with extremely conservative caps

Never use raw Kelly without conservative constraints.

============================================================
15. EXECUTION SIMULATOR
=======================

Create:

`trading/execution/`

Backtests must not assume perfect execution.

Model:

commission
fees
taxes
slippage
spread
latency
partial fill
price gaps
volume limits
execution delay
stop execution
market impact approximation

Support:

market
limit
stop
stop-limit

Allow pessimistic execution assumptions.

Run stress tests with worsened assumptions.

============================================================
16. REALISTIC BACKTESTING
=========================

The backtester becomes a first-class research engine.

Support:

multi-asset portfolios
position sizing
portfolio constraints
transaction costs
slippage
stop-loss
take-profit
trailing stops
capital allocation
rebalancing
benchmark comparison

Never rely on single-stock isolated backtests.

Evaluate:

portfolio-level behavior.

============================================================
17. WALK-FORWARD VALIDATION
===========================

Implement:

rolling walk-forward
expanding walk-forward

Never randomly shuffle time series.

Every experiment must separate:

TRAIN
VALIDATE
TEST

Use strict chronological ordering.

Where labels overlap, apply appropriate purge/embargo logic.

============================================================
18. PERMANENT HOLDOUT
=====================

Create a protected final holdout.

The evolutionary system MUST NEVER optimize against it.

The LLM MUST NOT inspect detailed holdout outcomes during training cycles.

Only finalists may be tested against the holdout.

If the holdout is repeatedly used for optimization:

MARK IT CONTAMINATED.

Create an audit flag.

============================================================
19. ADVERSARIAL BACKTESTING
===========================

Treat every profitable strategy as guilty until proven robust.

For every finalist run:

transaction cost stress
slippage stress
execution delay
entry delay
exit delay
parameter perturbation
universe perturbation
missing-data stress
noise injection
trade-order reshuffling
bootstrap
Monte Carlo
regime isolation
crash-period analysis
bull-period analysis
bear-period analysis
sideways-period analysis

Ask:

"Does the strategy still function when reality is worse than the historical simulation?"

============================================================
20. STRATEGY SENSITIVITY
========================

Detect fragile parameters.

Example:

Strategy works at:

RSI = 37

but fails at:

36
38
35
39

This is suspicious.

A robust strategy should generally survive reasonable parameter perturbations.

Create a sensitivity score.

Penalize fragile strategies.

============================================================
21. DARWINIAN ARENA — NEXT GENERATION
=====================================

KEEP THE DARWINIAN IDEA.

But evolve:

STRATEGY DNA

not merely agent prompts.

Each StrategyDNA must contain:

universe rules
alpha families
feature subset
lookbacks
thresholds
signal weights
regime rules
holding period
rebalance frequency
stop model
target model
trailing logic
position sizing
portfolio constraints
execution assumptions

All values have hard legal bounds.

============================================================
22. POPULATION
==============

Support:

32
64
128
256

research genomes.

The dashboard may display only the leading 4–8.

Maintain species.

Species examples:

Trend
Momentum
Mean Reversion
Breakout
Cross-sectional
Volatility
Statistical
ML Hybrid
Defensive

Prevent population collapse.

Prevent every strategy from converging into the same parameter region.

============================================================
23. EVOLUTION
=============

Use:

tournament selection
elitism
bounded mutation
crossover
fitness sharing
species preservation
novelty pressure
diversity preservation

Mutations must be schema validated.

The LLM cannot generate arbitrary executable code as a mutation.

============================================================
24. LLM MUTATION
================

The LLM may propose:

feature changes
parameter changes
strategy combinations
new hypotheses
new experiments

It must receive:

current DNA
performance
regime performance
failure modes
correlation
graveyard lessons
available features
allowed parameter bounds

It must return STRUCTURED proposals.

Every proposal is validated before execution.

============================================================
25. MULTI-OBJECTIVE FITNESS
===========================

Never:

fitness = total_return

Instead evaluate:

CAGR
Sharpe
Sortino
Calmar
max drawdown
profit factor
expectancy
trade consistency
downside deviation
tail loss
turnover
cost burden
slippage burden
regime robustness
parameter robustness
out-of-sample degradation
diversification contribution
correlation penalty

Use Pareto-style selection where practical.

Reward strategies that provide different useful return streams.

============================================================
26. NOVELTY
===========

Add novelty pressure.

A strategy should not survive solely because it resembles another winner.

Measure:

feature similarity
signal similarity
return correlation
trade overlap
parameter similarity

Reward useful novelty.

============================================================
27. GRAVEYARD
=============

Preserve and greatly expand the Graveyard.

Every dead strategy must store:

DNA
generation
parent
mutation
metrics
drawdowns
regime behavior
failure reason
paper results
sensitivity
correlations
lessons

Build a machine-readable failure database.

Before creating new mutations, ask:

"Has this idea already failed?"

============================================================
28. RESEARCH MEMORY
===================

Build a strategy knowledge base.

Store:

successful experiments
failed experiments
features
regimes
mutation history
failure patterns

Use it to prevent repeating low-value experiments.

The LLM should learn from historical failures.

============================================================
29. STRATEGY CORRELATION
========================

Build:

strategy return matrix
strategy correlation matrix
trade overlap matrix

Cluster strategies.

Portfolio allocation must account for strategy correlation.

Twenty strategies that all lose during the same event are one risk cluster.

============================================================
30. PORTFOLIO LEVEL FITNESS
===========================

Evaluate individual strategies AND combinations of strategies.

Measure:

diversification benefit
portfolio drawdown
correlation
risk concentration
capital efficiency

A mediocre strategy may become valuable if it diversifies stronger strategies.

Do not evaluate every strategy in isolation only.

============================================================
31. PAPER TRADING
=================

Paper trading is the main operational environment.

Keep:

BROKER=paper

by default.

Record:

timestamp
symbol
generation
strategy
signal
confidence
regime
expected_return
expected_risk
quantity
price
execution
fees
slippage
P&L
portfolio_state

Also record WHY the decision occurred.

============================================================
32. PAPER TRADING AS A RESEARCH EXPERIMENT
==========================================

Do not treat paper trading simply as a simulated wallet.

Compare:

predicted price
actual price

predicted return
realized return

predicted volatility
realized volatility

predicted confidence
actual hit rate

predicted fill
simulated fill

Analyze calibration.

If confidence = 80%, determine whether roughly 80% of predictions behave accordingly over sufficiently large samples.

============================================================
33. MODEL CALIBRATION
=====================

Evaluate:

Brier score
log loss
calibration curve
precision by confidence bucket
returns by confidence bucket

If the model is overconfident:

PENALIZE IT.

============================================================
34. DRIFT DETECTION
===================

Monitor:

feature drift
prediction drift
signal drift
regime drift
return degradation
turnover drift
execution drift
slippage drift

Detect when a once-good strategy becomes unreliable.

============================================================
35. AUTOMATIC RESEARCH CYCLES
=============================

Implement research lifecycle:

DAILY:

data validation
paper performance
risk monitoring
drift monitoring

WEEKLY:

strategy diagnostics
error analysis
regime analysis
correlation analysis

MONTHLY:

full evolutionary research cycle
retraining
new feature search
new alpha experiments
walk-forward evaluation
promotion review

Never automatically deploy to real money merely because a research cycle succeeds.

============================================================
36. CHALLENGER / CHAMPION SYSTEM
================================

Implement:

CHAMPION
CHALLENGER

The current production candidate remains frozen.

New strategies are challengers.

A challenger cannot replace a champion until it passes all gates.

Maintain rollback capability.

============================================================
37. SHADOW MODE
===============

Before live money:

run strategies in SHADOW mode.

They produce orders but do not execute them.

Compare:

would-be execution
actual market movement
expected slippage
actual price path

Measure paper/live divergence.

============================================================
38. PROMOTION GATES
===================

Require configurable minimum standards for:

sample size
drawdown
Sharpe
Sortino
Calmar
profit factor
out-of-sample performance
walk-forward stability
parameter robustness
transaction-cost robustness
regime robustness
paper consistency
model calibration
data integrity

A strategy failing ANY mandatory hard gate is rejected.

Do not average away hard failures.

============================================================
39. PRODUCTION SAFETY
=====================

Maintain:

paper mode by default

Live execution must require:

BROKER=angel
AND
TRADE_MODE=live

Do not alter these defaults.

Create an independent execution gateway.

The LLM cannot directly invoke arbitrary broker endpoints.

All live requests must pass:

schema validation
risk validation
position validation
capital validation
market-open validation
duplicate-order validation
rate-limit validation
emergency-stop validation

============================================================
40. SECURITY
============

Protect:

API keys
broker credentials
TOTP secrets
tokens
session files

Never print secrets.

Never place secrets in logs.

Do not return secrets to the LLM.

Implement safe logging.

============================================================
41. OBSERVABILITY
=================

Implement structured logs.

Every event should be traceable by:

request_id
experiment_id
strategy_id
generation
trade_id
model_id

Create metrics for:

latency
errors
signals
orders
rejections
drawdown
exposure
turnover
model confidence
strategy health

============================================================
42. RESEARCH DASHBOARD
======================

Upgrade the Streamlit dashboard.

Create views:

MARKET

UNIVERSE

SIGNALS

PORTFOLIO

RISK

ARENA

GENETICS

GRAVEYARD

EXPERIMENTS

MODELS

PAPER TRADING

ROBUSTNESS

REGIMES

CORRELATIONS

PROMOTION

AUDIT LOG

The dashboard should make it possible to understand:

WHAT happened
WHY it happened
WHICH strategy caused it
WHICH model produced it
WHICH data was used
HOW risky it was
WHETHER it survived validation

============================================================
43. GENERATION REPORT
=====================

After every Darwinian generation create:

`reports/generation_<N>.md`

Include:

population
survivors
dead strategies
champions
species distribution
fitness
drawdown
Sharpe
Sortino
Calmar
profit factor
turnover
cost impact
slippage impact
regime performance
parameter sensitivity
correlation
novelty
paper performance

Then provide:

WHY STRATEGIES SURVIVED

WHY STRATEGIES DIED

WHAT THE SYSTEM LEARNED

WHAT SHOULD BE TESTED NEXT

============================================================
44. AUTOPSY ENGINE
==================

When a strategy fails:

DO NOT simply mark it "bad."

Investigate:

bad regime?
bad execution?
bad data?
bad feature?
bad assumptions?
overfitting?
too much turnover?
too concentrated?
wrong holding period?
wrong risk sizing?
poor calibration?

Generate structured failure hypotheses.

============================================================
45. EXPERIMENT PRIORITIZATION
=============================

Do not run thousands of random experiments.

Use experiment prioritization.

Select experiments that maximize expected information gain.

Prefer:

high-uncertainty questions
important failure modes
unexplored strategy families
useful feature combinations
orthogonal hypotheses

Avoid repeatedly testing tiny parameter changes without evidence.

============================================================
46. HYPOTHESIS SYSTEM
=====================

Every new strategy must start with:

HYPOTHESIS

Example:

"Medium-term cross-sectional momentum may perform better after volatility-normalization during moderate trend regimes."

Then define:

expected mechanism
features
entry logic
exit logic
risk logic
failure conditions
test period
baseline

The experiment must test that hypothesis.

============================================================
47. SCIENTIFIC RESEARCH DISCIPLINE
==================================

Treat every trading strategy like a scientific experiment.

Never quietly change:

data
features
cost model
evaluation period
benchmark
risk limits

after seeing the result.

If anything changes, create a new experiment version.

============================================================
48. REPRODUCIBILITY
===================

Every experiment must be reproducible.

Store:

git commit
configuration
environment
seed
dataset version
universe version
feature version
model version
strategy DNA

============================================================
49. PERFORMANCE ATTRIBUTION
===========================

For every portfolio:

attribute returns to:

strategy
sector
symbol
regime
factor
market direction
position sizing
entry
exit
execution
fees

Answer:

"Where did the money come from?"

and:

"Where did the losses come from?"

============================================================
50. REGIME ATTRIBUTION
======================

Calculate performance separately across:

bull
bear
sideways
high volatility
low volatility
crisis-like periods

A strategy that works only in one regime must be identified as such.

============================================================
51. FAILURE SIMULATION
======================

Simulate:

missing data
API outage
stale prices
duplicate orders
partial fills
high slippage
market gaps
sudden volatility
strategy disagreement
model outage
broker outage
network failure

The system must fail safely.

============================================================
52. CHAOS TESTING
=================

Inject controlled failures into:

data layer
model layer
execution
broker adapter
risk manager

The platform must demonstrate graceful degradation.

============================================================
53. TESTING
===========

Add:

unit tests
integration tests
regression tests
property tests
data validation tests
evolution tests
backtest tests
risk tests
execution tests
security tests

Important invariants:

risk limits cannot be bypassed
holdout cannot influence training
future data cannot enter features
mutation cannot exceed bounds
negative quantities cannot appear
duplicate orders are rejected
paper mode cannot call live broker
invalid symbols cannot crash the system

============================================================
54. PROPERTY-BASED TESTING
==========================

Create invariants such as:

portfolio exposure <= configured maximum

position <= configured maximum

daily loss <= allowed threshold unless emergency state

mutation parameter ∈ allowed bounds

cash >= allowed minimum

orders cannot duplicate

future timestamps are rejected

missing features cannot silently become future data

============================================================
55. LLM SAFETY
==============

The LLM is untrusted input to the deterministic trading engine.

It can propose.

It cannot override.

It cannot:

change risk limits
unlock live trading
read secrets
bypass validation
write arbitrary broker requests
modify historical results
modify evaluation code during an experiment
access protected holdout results

============================================================
56. CODE QUALITY
================

Maintain:

clean interfaces
type hints
docstrings
small modules
dependency injection
configuration management
testability
logging
error handling

Avoid giant classes.

Avoid hidden globals.

Avoid circular dependencies.

============================================================
57. CONFIGURATION
=================

Move key research parameters into structured configuration.

Examples:

population_size
mutation_rate
crossover_rate
elite_fraction
max_position_weight
max_sector_weight
max_drawdown
risk_per_trade
transaction_cost
slippage
walk_forward_window
holdout_window
minimum_trades
universe_size
regime_thresholds

Never scatter constants across code.

============================================================
58. PERFORMANCE ENGINE
======================

Compute:

CAGR
Sharpe
Sortino
Calmar
max drawdown
profit factor
expectancy
win rate
average win
average loss
payoff ratio
volatility
downside deviation
turnover
cost ratio
tail risk
VaR
CVaR
recovery time
drawdown duration
rolling Sharpe
rolling drawdown

Also analyze:

monthly returns
weekly returns
daily returns
return distribution

============================================================
59. BENCHMARKING
================

Every strategy and portfolio must be compared with:

cash
buy-and-hold
market index
simple trend
simple momentum
simple mean reversion
equal weight
risk parity

A complex AI strategy must demonstrate incremental value.

============================================================
60. MODEL ENSEMBLE SELECTION
============================

Do not assume the biggest model is best.

Compare:

simple baseline
medium model
complex model

Use complexity penalties.

If the simpler model performs similarly out-of-sample:

prefer the simpler model.

============================================================
61. ONLINE ADAPTATION
=====================

Do not blindly retrain every day.

Use:

scheduled retraining
drift-triggered investigation
champion/challenger evaluation

Every retraining cycle must preserve historical comparison.

============================================================
62. ACTIVE RESEARCH AGENT
=========================

Create a dedicated ResearchAgent.

Its responsibilities:

inspect results
detect anomalies
generate hypotheses
propose experiments
review strategy failures
identify potential leakage
recommend new experiments
produce reports

It should act like a skeptical senior quant.

============================================================
63. STRATEGY MEMORY
===================

The ResearchAgent should remember:

what worked
what failed
under which regime
with which data
with which assumptions

This memory must be structured, not merely an LLM chat history.

============================================================
64. EVOLUTIONARY RESEARCH OBJECTIVE
===================================

The arena is NOT:

"Who made the most money?"

It is:

"Which strategy exhibits the strongest evidence of persistent, robust, risk-adjusted alpha under hostile evaluation?"

A winner must therefore demonstrate:

performance
stability
robustness
diversification
calibration
reproducibility
reasonable complexity

============================================================
65. NO SINGLE STRATEGY DOMINANCE
================================

Do not allow one strategy to consume all capital merely because it has a high score.

Portfolio allocation must consider:

risk
correlation
regime
confidence
diversification
drawdown

============================================================
66. META-OPTIMIZATION WARNING
=============================

Be extremely careful when optimizing the evolutionary engine itself.

Otherwise:

Strategy optimizer
optimizes strategies
which optimize against historical data
which optimizes against test assumptions

This creates meta-overfitting.

Keep evolutionary configuration conservative.

============================================================
67. RESEARCH DATA SPLITS
========================

Maintain:

RAW DATA
TRAIN DATA
VALIDATION DATA
TEST DATA
HOLDOUT DATA
PAPER DATA

Never mix them casually.

============================================================
68. LIVE READINESS
==================

DO NOT enable live trading as part of this implementation.

Instead implement a readiness report:

LIVE_READINESS.md

with:

technical readiness
data readiness
risk readiness
model readiness
operational readiness
broker readiness
monitoring readiness
regulatory checklist placeholder
failure recovery readiness

The final decision to use real money is manual.

============================================================
69. FINAL ACCEPTANCE CRITERIA
=============================

The new platform is considered successful only when it can answer:

1. What instruments are currently eligible?
2. Why was each instrument selected?
3. Which strategies are active?
4. Why is each strategy active?
5. What regime are we in?
6. How confident is the system?
7. What is the expected risk?
8. What is the position size?
9. What hard limits apply?
10. What would cause the strategy to stop?
11. How did it perform out-of-sample?
12. How sensitive is it to parameters?
13. How does it behave under worse execution?
14. Which strategies are correlated?
15. Which strategies failed?
16. Why did they fail?
17. What did the system learn?
18. What is being tested next?
19. Can every result be reproduced?
20. Can every decision be audited?

============================================================
70. FINAL OUTPUT REQUIREMENT
============================

At the end of the implementation produce:

`docs/GOD_LEVEL_ARCHITECTURE.md`

`docs/ARCHITECTURE_AUDIT.md`

`docs/RISK_MODEL.md`

`docs/BACKTEST_METHODOLOGY.md`

`docs/EVOLUTION_METHODOLOGY.md`

`docs/PAPER_TRADING_PROTOCOL.md`

`docs/LIVE_READINESS.md`

`docs/RESEARCH_PROTOCOL.md`

Also produce generation reports.

============================================================
71. EXECUTION RULE
==================

DO NOT stop after describing architecture.

Inspect the repository.

Implement the improvements.

Run tests.

Fix failures.

Run backtests.

Run walk-forward evaluation.

Run robustness tests.

Run the Darwinian arena.

Inspect the generated strategies.

Inspect the Graveyard.

Inspect the portfolio.

Inspect paper trading.

Document everything.

Do not fabricate results.

If a metric cannot be calculated because the available data is insufficient, explicitly report that.

============================================================
72. FINAL MENTAL MODEL
======================

Build the system as:

QUANT RESEARCH LAB
+
AI RESEARCHER
+
MULTI-ALPHA ENGINE
+
ML LABORATORY
+
REGIME ENGINE
+
PORTFOLIO MANAGER
+
HARD RISK GATE
+
REALISTIC EXECUTION SIMULATOR
+
DARWINIAN EVOLUTION
+
PAPER MARKET
+
RESEARCH MEMORY
+
AUTOMATED AUDITOR

The LLM proposes.

The mathematics evaluates.

The data decides.

The risk engine constrains.

The execution engine simulates.

The arena evolves.

The graveyard remembers.

The paper market validates.

Only evidence earns promotion.

Never optimize for the appearance of intelligence.

Optimize for robustness.

Never optimize for the highest backtest return.

Optimize for evidence of generalizable risk-adjusted performance.

The ultimate objective is not to create an AI that "predicts the market."

The objective is to create a system that can continuously discover, test, reject, improve, diversify and safely operate quantitative strategies while remaining skeptical of its own results.
