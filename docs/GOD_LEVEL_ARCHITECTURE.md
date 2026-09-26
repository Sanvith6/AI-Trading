# GOD-LEVEL ARCHITECTURE — ai-trading Platform

> Institutional-grade quant research + autonomous paper-trading platform.
> The LLM is the RESEARCH ORCHESTRATOR. Mathematics evaluates. Risk constrains.
> Only evidence earns promotion.

## 1. Data Sources

| Source | Cost | Data | Route |
|---|---|---|---|
| Yahoo Finance | FREE | NSE/ETF/Index OHLCV, quotes | `trading/data_provider.py` (default) |
| Angel One SmartAPI | FREE | Live NSE quotes, orders | `trading/market_data.py` (opt-in, `BROKER=angel`) |
| Zerodha Kite Connect | ~₹2000/mo | Live NSE orders | `trading/kite_auth.py` (opt-in, manual token) |
| OpenRouter | FREE tier | LLM reasoning (agent, swarm, evolution, autopsies) | `llm_client.py` |

**Default data provider: Yahoo Finance (zero API key required).**
No live broker trading unless `BROKER=angel` AND `TRADE_MODE=live`.

## 2. Architecture Layers (top to bottom)

```
LLM (Research Orchestrator)
 ├── Cannot override risk gates (see RISK_MODEL.md)
 ├── Proposes strategies → validated before execution
 ├── Proposes mutations → schema-validated, bounded
 └── Cannot read holdout results during training

┌──────────────────────────────────────────────────────────────┐
│ DATA SOURCES ─────────────────────────────────────────────── │
│ Yahoo Finance ──┐                                            │
│ Angel (opt-in) ─┤──► DATA INGESTION                          │
│                 └──► DATA VALIDATOR (trading/data/validator)  │
│                      Every record: symbol, exchange, ts,      │
│                      tz, source, interval, quality_status     │
│                      Flags: STALE/SUSPICIOUS/MISSING/ZERO/   │
│                      FUTURE/DUPLICATE/GAP/HALT              │
│                      Never silently consumes bad data         │
├──────────────────────────────────────────────────────────────┤
│ DATA VERSIONING ──► experiment_id, dataset_version,           │
│                    universe_version, feature_version,         │
│                    strategy_version, model_version, code_ver, │
│                    random_seed, retrieval_timestamp           │
├──────────────────────────────────────────────────────────────┤
│ DYNAMIC MARKET UNIVERSE (trading/universe)                   │
│ Discover 50-1000+ symbols via filters: validity, exchange,    │
│ price, liquidity, volatility, staleness, exclusion list      │
│ Maintains universe_version for reproducibility               │
├──────────────────────────────────────────────────────────────┤
│ FEATURE ENGINE (trading/features/engine.py)                  │
│ 40+ time-safe features across 8 categories:                  │
│ PRICE, TREND, MOMENTUM, MEAN_REVERSION, VOLUME, VOLATILITY,  │
│ MARKET, CROSS_SECTIONAL                                      │
│ Every feature: past/current data only. No look-ahead.        │
├──────────────────────────────────────────────────────────────┤
│ ALPHA LIBRARY (trading/alpha/) — 12 families                 │
│ Trend, Momentum, Mean Reversion, Breakout, Volatility,       │
│ Volume/Flow, Cross-Sectional, Statistical, Relative Strength,│
│ Defensive, Market Regime, ML Hybrid                          │
│ Common interface: generate_signal() → signal ∈ [-1,+1]       │
│ + confidence, horizon, expected_return/risk, regime, reason   │
├──────────────────────────────────────────────────────────────┤
│ REGIME ENGINE (trading/regime/)                              │
│ Rule-based baseline → bull/weak-bull/sideways/weak-bear/     │
│ high-vol/low-vol/risk-on/risk-off                           │
│ + optional ML (HMM, clustering)                            │
├──────────────────────────────────────────────────────────────┤
│ ML LAYER (trading/ml/) — tabular models                      │
│ Logistic Regression, Random Forest, GBM,                    │
│ calibated classifiers, regressors                            │
│ Predicts: P(up), P(down), E[return], E[vol]                 │
│ NEVER controls execution directly                            │
├──────────────────────────────────────────────────────────────┤
│ META MODEL + ENSEMBLE                                        │
│ Combines: technical features + alpha signals + ML predictions│
│ + regime + cross-sectional rank + risk estimates             │
│ Outputs: E[return], E[risk], signal_strength, confidence     │
│ Measures model disagreement → REDUCE EXPOSURE on disagreement│
├──────────────────────────────────────────────────────────────┤
│ BASELINE STRATEGIES (all 8 tested)                           │
│ BUY_AND_HOLD, SMA_TREND, EMA_TREND, MOMENTUM, MEAN_REV,      │
│ BREAKOUT, VOLATILITY_TARGET, EQUAL_WEIGHT                    │
│ AI must prove complexity adds value over baselines           │
├──────────────────────────────────────────────────────────────┤
│ PORTFOLIO CONSTRUCTION (trading/portfolio/)                  │
│ Equal weight / score / confidence / inverse vol /            │
│ risk parity / vol target, with constraints: max position,    │
│ max sector, max exposure, max positions, min cash, max corr  │
│ Clusters by correlation — 20 correlated stocks = 1 risk     │
├──────────────────────────────────────────────────────────────┤
│ RISK ENGINE (trading/risk_engine.py) ◄── IMMUTABLE ───────── │
│ AI CANNOT override: daily loss, drawdown, position size,     │
│ exposure, sector exposure, correlation, turnover, order size,│
│ max trades, leverage, min liquidity, stop-loss, emergency    │
│ HARD GATE: AI decision → portfolio val → risk val →          │
│ execution val → broker adapter. LLM NEVER skips chain.       │
├──────────────────────────────────────────────────────────────┤
│ EXECUTION SIMULATOR (trading/execution/simulator.py)         │
│ Commission, slippage, spread, partial fill, price gaps,      │
│ volume limits, latency, market impact, stop order types      │
│ Adversarial stress: 3× worse defaults via stress_execution() │
├──────────────────────────────────────────────────────────────┤
│ PAPER BROKER (trading/order_manager.py)                      │
│ Paper ledger (₹100k cash) + CSV trade log                    │
│ All fills costed (NSE STT+stamp+exchange+SEBI+GST+slippage)  │
│ Live routes to broker adapter only after promotion gate      │
├──────────────────────────────────────────────────────────────┤
│ PERFORMANCE ENGINE (trading/swarm/quant_metrics.py)          │
│ CAGR, Sharpe, Sortino, Calmar, maxDD, PF, expectancy,        │
│ win rate, avg win/loss, payoff ratio, vol, downside dev,     │
│ turnover, cost ratio, tail risk, VaR, CVaR, recovery time,   │
│ rolling Sharpe/drawdown, monthly/weekly/daily returns,        │
│ return distribution, + summarize_trades() with bootstrap CI  │
├──────────────────────────────────────────────────────────────┤
│ RESEARCH REGISTRY (trading/memory/)                          │
│ Successful experiments, failures, features, regimes,          │
│ mutation history, failure patterns — structured, not chat    │
├──────────────────────────────────────────────────────────────┤
│ DARWINIAN EVOLUTION (trading/swarm/)                         │
│ Strategy DNA (not just prompts) → universe rules, alphas,    │
│ features, lookbacks, thresholds, sizing, execution params    │
│ Tournament selection, BLX-α crossover, Gaussian mutation,    │
│ fitness sharing, species preservation, novelty pressure      │
│ Multi-objective fitness (NOT raw returns)                    │
│ Graveyard = machine-readable failure database                │
├──────────────────────────────────────────────────────────────┤
│ ROBUSTNESS GATES                                             │
│ Parameter sensitivity, adversarial testing, Monte Carlo,     │
│ cost stress, regime isolation, crash-period analysis         │
│ Minimum 30 trades before trusting any metric                 │
├──────────────────────────────────────────────────────────────┤
│ WALK-FORWARD VALIDATION                                      │
│ Rolling + expanding walk-forward. Train/Val/Test/Holdout.    │
│ Strict chronological order. Purge/embargo on label overlap.  │
│ Permanent holdout — NEVER optimized against, contamination   │
│ flag if repeatedly used.                                     │
├──────────────────────────────────────────────────────────────┤
│ PAPER DEPLOYMENT ──► SHADOW LIVE ──► MANUAL PROMOTION GATE   │
│ Paper validates → Shadow produces orders without executing → │
│ Compare paper vs live divergence → Human promotion decision   │
└──────────────────────────────────────────────────────────────┘
```

## 3. Key Design Decisions

| Decision | Rationale |
|---|---|
| LLM is orchestrator, not trader | LLMs hallucinate risk. Math doesn't. |
| Risk engine is immutable | 14 hard rules enforced programmatically |
| 1-bar execution delay in backtest | Eliminates look-ahead bias (CRITICAL fix) |
| Yahoo Finance default | Zero-cost data, no brokerage subscription |
| Paper mode default | No real money without explicit promotion |
| All costs modeled | STT, stamp, exchange, SEBI, GST, slippage |
| Strategy DNA not agent prompts | Evolve actual strategies, not chat personas |
| Multi-objective fitness | PnL alone rewards excessive risk-taking |
| Permanent holdout | Prevents optimization overfitting |
| Adversarial testing | "Does it survive worse-than-reality?" |
| Model disagreement → reduce exposure | Manufacturing confidence is forbidden |
| Minimum 30 trades for metrics | Statistical significance threshold |
| No seed → no result | Reproducibility is non-negotiable |
| Structured research memory | LLM should learn from past failures |

## 4. Existing Code Mapping

| New Architecture Layer | Existing File(s) | Status |
|---|---|---|
| Data Provider (Yahoo) | `trading/data_provider.py` | ✅ Functional (needs validator integration) |
| Technical Analysis | `trading/technical_analysis.py` | ✅ Enhanced (ATR/ADX added) |
| Order Manager (paper) | `trading/order_manager.py` | ✅ Enhanced (costed fills, ledger) |
| Risk Manager (basic) | `trading/risk_manager.py` | ✅ Kept (per-trade daily tracking) |
| Risk Engine (NEW hard gate) | `trading/risk_engine.py` | ✅ New (14 hard rules) |
| Execution Simulator (NEW) | `trading/execution/simulator.py` | ✅ New (slippage/partial/costs) |
| Feature Engine (NEW) | `trading/features/engine.py` | ✅ New (40 features) |
| Universe Discovery (NEW) | `trading/universe/discovery.py` | ✅ New |
| Data Validator (NEW) | `trading/data/validator.py` | ✅ New |
| Trading Agent (Enhanced) | `trading/trading_agent.py` | ✅ Enhanced (risk gate calls) |
| Backtest (Enhanced) | `trading/backtest.py` | ✅ Enhanced (look-ahead fix, costs) |
| Quant Metrics (Enhanced) | `trading/swarm/quant_metrics.py` | ✅ Enhanced (summarize_trades) |
| GA Operators (Wired) | `trading/swarm/genetic_ops.py` | ✅ Wired into evolution |
| Evolution Engine (Enhanced) | `trading/swarm/evolution_engine.py` | ✅ Enhanced (fitness, GA) |
| Arena Manager (Enhanced) | `trading/swarm/arena_manager.py` | ✅ Enhanced (MTM treasury) |
| Agent DNA (Enhanced) | `trading/swarm/agent_dna.py` | ✅ Enhanced (equity(), health) |
| LLM Client | `llm_client.py` | ✅ Stable |
| Base Tools | `tools.py` | ✅ Stable (run_python added to safe list) |
| Base Agent | `agent.py` | ✅ Enhanced (extension hooks) |
| Dashboard | `trading/trading_app.py` | ✅ Enhanced (risk, paper, logging) |
| Streamlit Agent UI | `agent_app.py` | ✅ Stable |
| CLI Agent | `agent_cli.py` | ✅ Stable |

## 5. New Modules Added

```
trading/risk_engine.py       # Immutable 14-rule risk boundary
trading/data/validator.py    # Data quality flags (STALE/SUSPICIOUS/.../)
trading/data/__init__.py
trading/execution/           # Simulator + gateway
trading/features/engine.py   # 40 time-safe features
trading/features/__init__.py
trading/universe/            # Dynamic discovery
trading/universe/discovery.py
trading/universe/__init__.py
docs/ARCHITECTURE_AUDIT.md   # 18 CRITICAL + 26 HIGH + 22 MEDIUM + 15 LOW
docs/GOD_LEVEL_ARCHITECTURE.md  # This file
docs/RISK_MODEL.md
docs/BACKTEST_METHODOLOGY.md
docs/EVOLUTION_METHODOLOGY.md
docs/PAPER_TRADING_PROTOCOL.md
docs/LIVE_READINESS.md
docs/RESEARCH_PROTOCOL.md
trading/logging_setup.py     # Centralized logging
PROJECT_WALKTHROUGH.md       # Module-by-module guide
TECHNOLOGIES.md              # Stack report
reports/                     # Generation reports (auto-generated)
```

## 6. Entry Points

| Command | What runs | Port |
|---|---|---|
| `streamlit run trading/trading_app.py` | Dashboard (signals, portfolio, risk, swarm) | 8502 |
| `python agent_cli.py` | Rich terminal agent console | — |
| `python agent_app.py` | Streamlit agent web dashboard | 8501 |
| `python chat.py` | Terminal chat | — |
| `python app.py` | Streamlit chat | 8501 |
| `python quickstart.py` | Single prompt test | — |
| `python -m trading.backtest RELIANCE --days 365` | Walk-forward backtest with costs | — |
| `python trading/swarm/test_swarm.py` | Swarm lifecycle test | — |
| `python trading/swarm/test_quant_wiring.py` | Quant/GA/risk wiring test | — |
| `run_trading.bat` | Dashboard launcher | 8502 |
| `run_backtest.bat` | Backtest launcher | — |

## 7. Safety Hierarchy (highest to lowest)

1. **Emergency halt** (`EMERGENCY_HALT=true` env var) — blocks ALL trading
2. **Risk Engine hard gates** — every order validated before fill
3. **Daily loss circuit breaker** — auto-halt at ₹{TRADE_MAX_DAILY_LOSS}
4. **Per-trade risk cap** — `TRADE_RISK_PER_TRADE_PCT`% of cash per trade
5. **LLM system prompt rules** — advisory only, NEVER override
6. **Portfolio construction constraints** — sector/exposure/position limits
7. **Execution simulation** — costs, slippage, partial fills
8. **Strategy signals** — alpha/ML/ensemble outputs
9. **LLM proposals** — validated before execution, never auto-executed

## 8. Defaults That Must Never Change

| Parameter | Default | Why |
|---|---|---|
| `BROKER` | `paper` | No real money without explicit human decision |
| `TRADE_MODE` | `paper` | Paper validates; live requires promotion gate |
| `OPENROUTER_MODEL` | `inclusionai/ling-3.0-flash-vl:free` | Cost-free, adequate for orchestration |
| `LOG_LEVEL` | `INFO` | Audit trail requirement |
| `TRADE_RISK_PER_TRADE_PCT` | `5.0` | 1/20th of capital — institutional 1% rule |
| `TRADE_MAX_PER_ORDER` | `5000` | Conservative per-order cap |
| `TRADE_MAX_DAILY_LOSS` | `2000` | Hard daily loss kill-switch |
| `TRADE_MAX_POSITIONS` | `3` | Maximum diversification at this capital scale |
| `TRADE_STOP_LOSS_PCT` | `2.0` | Per-trade hard stop |
| `TRADE_TARGET_PCT` | `4.0` | Per-trade take-profit |
| `MIN_TRADES` | `30` | Minimum trades before trusting metrics |

---

*See individual docs for detailed methodology on backtesting, evolution, risk, paper trading, and research protocols.*
