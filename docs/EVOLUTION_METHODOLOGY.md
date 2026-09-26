# EVOLUTION METHODOLOGY — Darwinian Swarm

> Evolve STRATEGY DNA, not agent prompts.
> Only strategies with strongest evidence of robust, risk-adjusted alpha survive.

## 1. What Evolves

Each strategy genome (`StrategyDNA`) encodes the COMPLETE strategy definition:

```python
@dataclass
class StrategyDNA:
    # Identity
    strategy_id: str
    name: str
    species: str            # Trend / Momentum / MeanReversion / etc.
    generation: int
    parent_id: Optional[str]
    
    # Universe & Data
    universe_rules: Dict[str, Any]      # min_price, max_price, min_vol, exchange, sector
    feature_subset: List[str]           # Which features this strategy uses
    lookbacks: Dict[str, int]           # RSI 14? 21? 50?
    
    # Signal Logic
    alpha_families: List[str]           # Trend, Momentum, ... combinations
    signal_weights: Dict[str, float]    # How each alpha contributes
    thresholds: Dict[str, float]        # Entry/exit thresholds
    
    # Position & Risk
    position_sizing: str                # fixed_fractional / inverse_vol / kelly / risk_parity
    risk_per_trade_pct: float           # % of capital per trade
    stop_model: Dict[str, float]        # stop loss type + %
    target_model: Dict[str, float]      # take profit type + %
    
    # Portfolio
    portfolio_constraints: Dict[str, Any]  # max positions, max exposure, max sector
    
    # Execution Assumptions
    execution_model: Dict[str, Any]     # slippage, spread, delay assumptions
    
    # Validation (immutable)
    creation_timestamp: str
    code_version: str
    experiment_id: str
    last_promotion_gate_result: Optional[str]
```

**Hard bounds exist on every numeric parameter.** No parameter can exceed:
- `stop_loss_pct`: [1.0%, 5.0%]
- `take_profit_pct`: [2.0%, 10.0%]
- `risk_per_trade_pct`: [0.5%, 10.0%]
- `min_confidence`: [50%, 90%]
- All lookbacks: [5, 200] bars
- All weights: [0.0, 1.0], normalized to sum 1.0

**Mutations cannot exceed bounds.** Schema validation is mandatory.

## 2. Population

Initial population: 4 distinct archetypes (Gen 1)
- Breakout Hunter (aggressive momentum)
- Mean Reversion Scalper (statistical)
- Trend Follower (disciplined momentum)
- Contrarian Sniper (patience-based)

Target population scales with config:
- Research small: 32 genomes
- Research medium: 64
- Research large: 128-256
- Dashboard displays top 4-8, full population maintained internally

**Species preservation**: 4 species minimum maintained at all times:
- Trend
- Momentum
- Mean Reversion
- Breakout/Cross-Sectional

No species can be eliminated until 30+ generations have passed.

## 3. Fitness Function (Multi-Objective)

**NEVER** `fitness = total_return`. Instead:

```
fitness = (CAGR_score × 2)
        + (Sharpe_score × 3)
        + (Sortino_score × 2)
        + (profit_factor_score × 1)
        + (stability_score × 1)         # rolling Sharpe consistency
        + (complexity_penalty × -1)     # fewer params = higher score
        + (diversification_score × 1)   # provides unique return stream
        - (drawdown_penalty × 2)        # severe penalty for big DD
```

Where:
- `CAGR_score` = normalized CAGR (0-100)
- `Sharpe_score` = Sharpe ratio clamped to 0-100
- `profit_factor_score` = min(PF, 3.0) / 3.0 × 100
- `stability_score` = Sharpe on out-of-sample / Sharpe on in-sample × 100
- `complexity_penalty` = fewer features/parameters = higher score (max 20)
- `diversification_score` = 100 if strategy adds uncorrelated returns, else lower
- `drawdown_penalty` = max(0, (DD/max_DD)^2) × 100

**Key insight**: A strategy with 15% CAGR and Sharpe 2.0 beats 30% CAGR and Sharpe 0.5.

## 4. Selection (Tournament)

```python
parent = tournament_selection(population, fitness_map, k=3)
```

- Select 3 random strategies from population
- Choose the fittest
- Repeat for second parent (tournament selection, not elitist-only)

## 5. Crossover (BLX-α)

Numeric parameters crossed via Blended Crossover:
```
child_param = uniform(min(P1,P2) - α×|P1-P2|, max(P1,P2) + α×|P1-P2|)
α = 0.3
Then clipped to hard bounds
```

## 6. Mutation (Adaptive Gaussian)

```
if random() < mutation_rate (15%):
    param += gaussian(0, sigma × (max - min))
    param = clip to bounds
```

## 7. LLM Mutation (Controlled)

The LLM proposes STRUCTURED proposals, never arbitrary code:

```
Input to LLM:
  - Champion DNA
  - Performance (not returns alone — Sharpe, drawdown, stability)
  - Regime performance breakdown
  - Failure modes (from graveyard)
  - Correlation analysis
  - Available features (full list)
  - Allowed parameter bounds (hard limits)
  - Graveyard lessons

Output from LLM (must be JSON):
  - name: evolved strategy name
  - archetype: category
  - mutation_notes: 1 sentence
  - system_prompt: short research hypothesis
  - parameters: ONLY within hard bounds

Validation: EVERY LLM proposal validated against bounds before becoming DNA.
Temperature for LLM mutation calls: ZERO (deterministic).
```

## 8. Novelty Pressure

A strategy surviving solely on similarity to a winner is penalized:

- **Feature similarity**: cosine distance on feature weight vectors
- **Signal similarity**: correlation of generated signals over 60 days
- **Return correlation**: how correlated returns are with champion
- **Trade overlap**: what % of trades overlap with champion

Novelty bonus added to fitness if strategy provides a different return stream.
Prevents every strategy from collapsing into one that worked in the last cycle.

## 9. Graveyard

Every dead strategy stores COMPLETE forensic record:
- Full DNA
- Generation, parent, mutations received
- All metrics at death
- Drawdown history
- Regime performance (was it in/out of its comfort regime?)
- Sensitivity analysis (did it fail on parameter perturbation?)
- Correlation with other strategies
- Failure reason (auto-generated by LLM + validated)
- Lessons (directive for future generations)

Before creating any mutation: check "Has this idea already failed?"
(Graveyard lookup by strategy name and parameter region.)

## 10. Evolution Cycle

```
Every generation (1-2 weeks in production, configurable):

1. Run market scan round for each agent
2. Record all trades, P&L, drawdowns
3. Check health: mark dead strategies
4. Bury dead in graveyard with autopsies
5. Calculate fitness (multi-objective)
6. Tournament select champions (top 2)
7. BLX crossover + Gaussian mutation → offspring
8. LLM proposes variations → validate → merge
9. Speciation: enforce 4 species minimum
10. Reconcile treasury (mark-to-market)
11. Persist generation report to reports/generation_N.md

Evolution can be triggered manually via "Evolve Gen" button.
NEVER automatic without human review of generation report.
```

## 11. What Makes a Winner

A strategy is promoted only after demonstrating (ALL of):
- Out-of-sample Sharpe > 0.5
- Max drawdown < 15%
- Profit factor > 1.2
- ≥30 trades minimum
- Parameter sensitivity < 50%
- Survives 3× cost stress test
- Survives regime isolation (all regimes tested)
- Positive returns in at least 2 of 3 regimes

The arena is NOT "who made the most money."
The arena is "which strategy exhibits the strongest evidence of
persistent, robust, risk-adjusted alpha under hostile evaluation?"

---

*Evolution is a research tool, not an autopilot. Every generation report
is reviewed by a human before the next generation starts.*
