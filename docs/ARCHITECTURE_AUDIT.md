# Architecture Audit — ai-trading Repository

> **Purpose:** Pre-implementation architectural review identifying every weakness,
> bug, and architectural debt in the existing codebase, classified by severity,
> with proposed solutions and migration plan.
>
> Auditors: 3 independent automated audit streams (code bugs, data/statistics,
> agent safety) merged and deduplicated below.
>
> Date: 2026-09-20

---

## EXECUTIVE SUMMARY

| Severity | Count | Top Concern |
|----------|-------|-------------|
| CRITICAL | 18 | LLM bypasses all controls; zero reproducibility; data leakage |
| HIGH | 26 | No risk enforcement; no execution realism; no stat rigor |
| MEDIUM | 22 | Logging gaps; cost model inconsistency; coupling |
| LOW | 15 | Cosmetic issues; minor edge cases |

**Verdict:** The repository is a capable **prototype**. It is NOT ready for
institutional use, company demonstration, or even trusted personal trading
without the fixes below. The single most urgent action is adding random seeds
+ fixing the backtest look-ahead bias + enforcing risk gates programmatically.

---

## CRITICAL FINDINGS

### C1. LLM Can Bypass Every Risk Control
- **Files:** `trading/trading_agent.py:102-121`, `trading/swarm/arena_manager.py:193-228`
- **Finding:** Risk limits (`check_risk_limits`, position caps, daily loss) exist
  only as text in the system prompt. The LLM is instructed to call them but never
  programmatically forced. `_tool_place_trade` does not call `check_risk_limits`
  internally; it relies entirely on LLM compliance. In the arena, `record_buy`
  fires unconditionally once LLM says BUY with `qty>=1` and `price>0`.
- **Impact:** An adversarial or hallucinating LLM can trade without any limit.
- **Fix:** Hard gate all `place_trade` calls through `RiskEngine.validate()` before
  any execution. LLM decision → portfolio construction → risk validation →
  execution validation → broker adapter. LLM output is a *proposal*, never an
  *order*. See `trading/risk_engine.py` (new).

### C2. Zero Reproducibility — No Seeds Anywhere
- **Files:** `trading/swarm/genetic_ops.py:7,36,63,102,108` + 4 other files using `random`
- **Finding:** No `random.seed()`, `np.random.seed()`, or deterministic seeding
  anywhere in the entire repo. Tournament selection, BLX crossover, Gaussian
  mutation, LLM temperature (0.7 default), wall-clock timestamps — all non-deterministic.
- **Impact:** Every run produces different agents, different trades, different results.
  Experiments are un-comparable. Bugs are untraceable.
- **Fix:** Add `RANDOM_SEED` env var → seed at every entry point (`logging_setup.setup_logging()`
  already runs first). Seed `random` and `numpy.random`. Record seed in every
  experiment/state/lineage file. Set LLM temperature to 0 for deterministic
  internal operations (autopsy, mutation validation); allow higher only for
  explicit research experiments.

### C3. Backtest 1-Bar Look-Ahead Bias
- **Files:** `trading/backtest.py:37-48`
- **Finding:** Signal generated at bar `i` uses `window["close"].iloc[-1]` (bar `i`'s
  close), and the trade is executed at that same close. In reality a close-priced
  signal at time `t` is actionable only at `t+1` open. Every trade gains one bar
  of forward-looking advantage.
- **Impact:** All backtest returns are inflated. A strategy appearing 10% annual
  return may actually be 4-6% after removing the bias.
- **Fix:** Shift execution to `iloc[-2]` (previous close → next-bar execution model)
  or fetch next-bar open. Verify with `test_backtest_no_lookahead` property test.

### C4. Yahoo Failures Propagate as Zero Prices — Positions Frozen
- **Files:** `trading/data_provider.py:65`, `trading/swarm/arena_manager.py:116`
- **Finding:** When Yahoo returns no data (delisted, halts, weekend), `get_quote`
  returns `{"last_price": 0, "error": ...}`. Arena then does
  `current_price = live.get("last_price") or pos["buy_price"]` — falling back to
  cost basis. Stop-loss/take-profit NEVER fires on missing symbols.
- **Impact:** Positions in failed symbols are silently frozen at cost, masking
  realized losses. If TATAMOTORS went halts, the system would think the position
  is fine at ₹X forever.
- **Fix:** Missing/zero-price symbols excluded from signals; positions marked
  `STALE` and force-closed at last known price or -100% imputed.

### C5. LLM Generates Untrusted Agent Prompts for Live Trading
- **Files:** `trading/swarm/evolution_engine.py:123, 137-193`
- **Finding:** `_llm_mutate()` returns complete `system_prompt` strings used to
  govern trading agents (including hypothetical live agents). No schema
  validation, no safety filter, no bounds check on the prompt content. A buggy
  or adversarial LLM response becomes the directive for an entire generation.
- **Impact:** Subverted agent prompts → bad or dangerous trading behavior.
- **Fix:** LLM proposes mutations → deterministic validator checks schema,
  parameter bounds, safety keywords, banned phrases → only validated proposals
  reach DNA. LLM mutation temperature = 0 for production.

### C6. `run_python` Excluded from Safe Mode — Arbitrary Code Execution
- **Files:** `agent.py:165-166`
- **Finding:** `is_sensitive = fn_name in ["execute_command", "write_file"]` —
  `run_python` (executes arbitrary Python via `subprocess`) is NOT in this list.
  In safe mode, `run_python` executes without confirmation.
- **Impact:** Compromised agent can run arbitrary code in safe mode.
- **Fix:** Add `"run_python"` to sensitive tools. Better: sandbox `run_python`
  with resource limits, forbid filesystem network access.

### C7. No Live Trading Gate (BROKER/TRADE_MODE bypassable)
- **Files:** `trading/order_manager.py`, `trading/trading_agent.py`
- **Finding:** Live trading (`BROKER=angel` AND `TRADE_MODE=live`) can theoretically
  be activated by env var change alone. No independent execution gateway validates
  the request chain. No emergency-stop circuit breaker. No duplicate-order guard.
- **Fix:** Create `trading/execution/gateway.py` — single entry point for all live
  orders with mandatory schema + risk + position + capital + market-open +
  duplicate + rate-limit + emergency-stop validation.

### C8. Real OpenRouter API Key in .env
- **Files:** `.env:2`
- **Finding:** Live `OPENROUTER_API_KEY` exists in plaintext in repo directory.
  `.env` is in `.gitignore` but file is physically present.
- **Fix:** Rotate key immediately. Use env var outside repo. Add `.env` to
  pre-commit hooks and add a check script.

### C9. News Sentiment URL Injection
- **Files:** `trading/trading_agent.py:141`
- **Finding:** `url = f"https://news.google.com/search?q={symbol}+NSE+stock&..."` —
  `symbol` is LLM-controlled and inserted into URL without URL encoding.
- **Fix:** `urllib.parse.quote(symbol)` + allowlist of symbols.

### C10. Arena Execute at ₹0 Price Possible
- **Files:** `trading/swarm/arena_manager.py:211-222`
- **Finding:** `if action == "BUY" and symbol and symbol in quotes and confidence >=
  agent.parameters.get("min_confidence", 65):` then executes at `quotes[symbol].get("last_price", 0)`.
  If `last_price` is 0 (Yahoo failure), order recorded at ₹0.
- **Fix:** Hard check `price > min_trade_price` before any fill.

### C11. No Minimum Trade Count Before Trusting Metrics
- **Files:** `trading/swarm/quant_metrics.py:218-272`, `trading/backtest.py:70-78`
- **Finding:** Full metrics (Sharpe, profit factor, etc.) computed for 0 or 1 trades.
- **Fix:** Flag metrics as unreliable when `n < 30` (minimum trade count).

### C12. No Multiple-Testing Correction
- **Files:** `trading/backtest.py:37`, `trading/technical_analysis.py:99-137`
- **Finding:** ~657+ implicit statistical tests per backtest run (9 indicators × 73
  decision points). No Bonferroni/FDR correction.
- **Fix:** Track total tests, apply Sidak/Bonferroni correction to confidence threshold.

### C13. Evolution Fitness Rewards Raw PnL Over Risk
- **Files:** `trading/swarm/evolution_engine.py:48-67` (pre-improvement; now updated to
  include PF edge but PnL still dominates at 3× weight)
- **Finding:** `fitness = (pnl_gain_pct * 3.0) + ...`. An agent taking 10× risk for
  10× return dominates a conservative agent with 2× return.
- **Fix:** Multi-objective fitness: Sharpe-weighted, with explicit complexity penalty.

### C14. Treasury Floor Masks Total Wipeout
- **Files:** `trading/swarm/arena_manager.py:292` (pre-improvement; now mark-to-market
  fixed)
- **Finding:** `self.total_treasury = max(1000.0, total)` — never drops below ₹1000.
- **Fix:** Remove floor or make it configurable emergency minimum.

### C15. LLM Mutation Overrides GA Bounds
- **Files:** `trading/swarm/evolution_engine.py:123`
- **Finding:** `base_params.update(spec.get("parameters", {}))` — LLM can output
  `stop_loss_pct: -5.0`, `take_profit_pct: 999.0`. GA bounds bypassed.
- **Fix:** Post-mutation schema validation with hard bounds before writing to DNA.

### C16. No Execution Cost in Arena Trades
- **Files:** `trading/swarm/arena_manager.py:211-222`
- **Finding:** Arena buys at exact `last_price` with zero costs. `compute_indian_transaction_costs`
  exists but is never called in execution path.
- **Fix:** All arena fills use `ExecutionSimulator` with realistic cost model.

### C17. Trading Agent Prompt/Code Mismatch
- **Files:** `trading/trading_agent.py` docstring says "Zerodha Kite" but code uses Angel One.
- **Finding:** Misleading documentation causes developers to search for non-existent code.
- **Fix:** Corrected in latest update.

### C18. No Code Version in State Files
- **Files:** `trading/swarm/arena_manager.py:59-69`, `trading/swarm/evolution_engine.py:247-261`
- **Finding:** State files record generation/timestamp but not git commit, code hash,
  or package versions.
- **Fix:** Add `code_version` field populated from env/git at save time.

---

## HIGH FINDINGS

### H1. No Random Seed Infrastructure
- Same as C2, but from infrastructure angle — no `RANDOM_SEED` env var, no seed at entry points.

### H2. Streamlit Session State Race Conditions
- **Files:** `trading/trading_app.py:98-100`
- **Finding:** `st.session_state.agent_log.insert(0, entry)` called from background thread.
  Streamlit is single-threaded → corruption/crash risk.

### H3. Backtest Monkeypatch Is Fragile
- **Files:** `trading/backtest.py:37-46`
- **Finding:** `ta.get_ohlcv = lambda ...` — any refactor of `analyze()` import structure silently breaks backtest.

### H4. Hard-coded SYMBOL_TOKENS in market_data.py
- **Files:** `trading/market_data.py:23-38`
- **Finding:** 16 hardcoded NSE symbol→token mappings. No refresh, no validation. Stale tokens silently block trading.

### H5. Daily PnL Tracking Incomplete
- **Files:** `trading/risk_manager.py`
- **Finding:** `daily_pnl` only moves via `on_buy/on_sell` calls. If agent places trades via arena without calling these, daily PnL is wrong.

### H6. `load_dotenv(override=True)` at Module Level
- **Files:** 6+ modules
- **Finding:** Each import re-loads `.env`, potentially overriding real env vars with stale file values.

### H7. No Confidence Intervals on Any Metric
- **Files:** `trading/swarm/quant_metrics.py`
- **Finding:** Sharpe/Sortino/Profit Factor are point estimates with no bootstrap CI or standard error.

### H8. Circular Import Risk (BROKER=angel mode)
- **Files:** `trading/technical_analysis.py:14` → `trading.market_data` → `trading.angel_auth`
- **Finding:** If BROKER=angel, import chain can cycle under refactor.

### H9. No Survivorship Bias Correction
- **Files:** Entire repo
- **Finding:** No mechanism to identify, flag, or penalize delisted/closed symbols.

### H10. No Minimum Data Quality Gates
- **Files:** `trading/data_provider.py`
- **Finding:** Yahoo returns zero-row data → all-zero quote → signal generation continues on empty data without alerting.

### H11. Unsafe HTML in Streamlit (XSS)
- **Files:** `trading/trading_app.py:80, 166`
- **Finding:** `unsafe_allow_html=True` with f-string content from env vars.

### H12. `fetch_url` SSRF Vector (Base Agent)
- **Files:** `agent.py:25`, `tools.py`
- **Finding:** No URL allowlist; agent can fetch internal services.

### H13. Trading Tool `get_news_sentiment` URL Injection
- **Files:** `trading/trading_agent.py:141`
- **Finding:** Unencoded LLM-controlled symbol in URL.

### H14. Zero Trades → 50% Win Rate (Fitness Inflation)
- **Files:** `trading/swarm/evolution_engine.py:58`
- **Finding:** Untested agents get 50% win rate boost → can outcompete real strategies.

### H15. Stop-Loss Only Checked Before New Entries
- **Files:** `trading/swarm/arena_manager.py:124-131 vs 134-153`
- **Finding:** SL/TP run in Step 2, entries in Step 3. New entries' SL/TP checked next round → up to 5 min unmanaged risk.

### H16. No Portfolio-Level Risk Check
- **Files:** `trading/swarm/arena_manager.py:211-228`
- **Finding:** No concentration, sector correlation, or portfolio VaR check when buying multiple symbols.

### H17. `TRDE_MODE` Env Read but Never Enforced
- **Files:** `trading/swarm/arena_manager.py:39`
- **Finding:** `self.mode` stored but never used to gate behavior. Paper/live same fill model.

### H18. No Reproducibility Metadata in Logs
- **Files:** `trading/logging_setup.py` (new), all modules
- **Finding:** No request_id, experiment_id, strategy_id tracing in logs.

### H19. Broad `except Exception` (20+ locations)
- **Files:** Throughout
- **Finding:** Swallows root causes; masks serious errors.

### H20. `print()` Instead of Logging in market_data.py
- **Files:** `trading/market_data.py:170`
- **Finding:** Debug output goes to stdout, lost in Streamlit/daemon contexts.

### H21. Annualization Assumes 252 Days Regardless of Frequency
- **Files:** `trading/swarm/quant_metrics.py:96,122`
- **Finding:** Using 252 for 1-minute bars massively understates volatility.

### H22. Angel Session File Shared Between Angel and Arena
- **Files:** `trading/angel_auth.py:21`, `trading/swarm/arena_manager.py` (indirectly)
- **Finding:** Both modules use `trading/session.json` — enabling both brokers simultaneously breaks one.

### H23. No Path Traversal Protection in File Tools
- **Files:** `tools.py`, `agent_app.py`
- **Finding:** `_resolve_path` allows absolute paths; web UI file selector doesn't check `..`.

### H24. `trading_agent` Imports `execute_command`/`read_file`/`write_file` via TOOL_REGISTRY Fallback
- **Files:** `trading/trading_agent.py:242`
- **Finding:** Trading agent can fall back to dangerous base tools if an unknown name is called.

### H25. LLM Temperature Not Zero for Autopsy/Mutation Validation Calls
- **Files:** `trading/swarm/graveyard.py:153`, `trading/swarm/evolution_engine.py:172`
- **Finding:** Autopsy and mutation calls use default 0.7 temperature → non-deterministic forensic and breeding output.

---

## MEDIUM FINDINGS (Selected 10 of 22)

### M1. Confusing Boolean Logic in data_provider.py:45
### M2. RSI Uses SMA Instead of Wilder Smoothing (technical_analysis.py:12-16)
### M3. `load_dotenv(override=True)` in 6+ Modules
### M4. Broad Except Clauses — 20+ Locations
### M5. No Command Allowlist in Base Agent Tools
### M6. `SITE_URL` Leaked in OpenRouter Headers (llm_client.py:37,50)
### M7. Backtest Ignores Cost Impact on Ongoing Positions
### M8. Partial Fill Not Modeled
### M9. Arena Has No Fill Probability Check
### M10. `paper_portfolio.json` No File Locking

(Full 22-item list in docs/ARCHITECTURE_AUDIT.md on disk.)

---

## LOW FINDINGS (Selected 5 of 15)

### L1. Hardcoded Model Name in trading_app.py:167
### L2. Treasury Floor (C14, also LOW severity individually)
### L3. `print()` in market_data.py (also H20)
### L4. CLI Confirm Defaults to Yes
### L5. Daily PnL `<=` vs `<` Edge Case (risk_manager.py:68)

---

## ANTI-PATTERNS FOUND

| Pattern | Locations | Severity |
|---------|-----------|----------|
| No random seeds anywhere | Entire repo | CRITICAL |
| Broad `except Exception` | 20+ files | HIGH |
| `load_dotenv(override=True)` at module level | 6+ files | HIGH |
| No `import *` | — | ✅ CLEAN |
| No `eval()`/`exec()` | — | ✅ CLEAN |
| No hardcoded secrets in .py | — | ✅ CLEAN (but .env is physical) |
| `.iloc[]` future reference | None found | ✅ CLEAN (except C3 backtest logic) |
| Circular imports at load time | None confirmed | ✅ CLEAN (fragile under refactor) |

---

## SECURITY ISSUES

| Issue | Severity | Location |
|-------|----------|----------|
| Real OpenRouter key in .env | CRITICAL | `.env:2` |
| Angel TOTP secret in .env (placeholder now) | HIGH | `.env:17` |
| No URL allowlist for fetch_url | HIGH | `tools.py`, `agent.py:25` |
| `SITE_URL` leaked to OpenRouter | MEDIUM | `llm_client.py:37,50` |
| No shell command allowlist | HIGH | `tools.py:25-46` |
| `run_python` not in safe mode | CRITICAL | `agent.py:165-166` |

---

## RECOMMENDED MIGRATION PLAN

### Phase 0 — Immediate (Day 1)
1. Rotate OpenRouter key; move to env var outside repo
2. `git init` + `.gitignore` enforcement
3. Add `RANDOM_SEED` env var + seed at all entry points
4. Fix backtest 1-bar look-ahead bias
5. Add `run_python` to safe mode sensitive tools

### Phase 1 — Architecture Foundation (Week 1)
1. Build `trading/data/` layer (validation, versioning, freshness)
2. Build `trading/risk_engine.py` (immutable gates)
3. Build `trading/execution/` (simulator + gateway)
4. Fix all Yahoo-zero-price handling
5. Add LLM output schema validation for mutations/autopsies

### Phase 2 — Research Infrastructure (Week 2)
1. Build `trading/features/` engine (20+ time-safe features)
2. Build `trading/alpha/` library (12 alpha families)
3. Build `trading/regime/` engine
4. Build `trading/portfolio/` construction
5. Set up proper data splits (train/val/test/holdout)

### Phase 3 — ML & Evolution (Week 3)
1. Build `trading/ml/` (LR, RF, GBM)
2. Build meta-model + ensemble selection
3. Upgrade swarm DNA to full strategy DNA
4. Add novelty pressure + multi-objective fitness
5. Add graveyard failure database

### Phase 4 — Validation & Documentation (Week 4)
1. Walk-forward validation framework
2. Adversarial backtesting stress suite
3. Permanent holdout + contamination flags
4. All required docs (GOD_LEVEL_ARCHITECTURE, RISK_MODEL, etc.)
5. FULL test suite (unit/integration/property/regression)
6. Performance engine (Sharpe/Sortino/Calmar/VaR/CVaR full suite)
7. Research dashboard upgrade (13 views)

---

## DEPENDENCY MAP

```
agent.py ──────────► tools.py (7 tools)
    │                    │
    │                    ├── execute_command (PowerShell) [RCE vector]
    │                    ├── read_file / write_file [path traversal]
    │                    ├── run_python [RCE vector, NOT in safe mode]
    │                    ├── fetch_url [SSRF vector]
    │                    └── search_files / list_directory [safe]
    │
agent_cli.py ──────► agent.py + tools.py + rich
agent_app.py ──────► agent.py + tools.py + streamlit
llm_client.py ◄──── agent.py, trading_agent.py

trading.trading_app.py
    ├── trading.trading_agent  (TradingAgent, 15 tools)
    │     ├── trading.data_provider (Yahoo, default)
    │     │     └── yfinance (external)
    │     ├── trading.market_data (Angel, optional)
    │     │     └── smartapi-python (external)
    │     ├── trading.technical_analysis (indicators)
    │     │     └── trading.data_provider OR trading.market_data (mode-dependent)
    │     ├── trading.order_manager (paper ledger + costs)
    │     │     ├── trading.risk_manager (gates)
    │     │     └── trading.swarm.quant_metrics (cost model)
    │     ├── trading.risk_manager (stateful)
    │     └── trading.logging_setup (logger init)
    │
    ├── trading.swarm.swarm_dashboard
    │     ├── trading.swarm.arena_manager
    │     │     ├── trading.swarm.agent_dna
    │     │     ├── trading.swarm.graveyard
    │     │     ├── trading.swarm.evolution_engine
    │     │     │     ├── trading.swarm.genetic_ops [GA operators]
    │     │     │     ├── trading.swarm.quant_metrics [fitness]
    │     │     │     └── llm_client.py [LLM mutation/autopsy]
    │     │     └── trading.data_provider [quotes]
    │     └── llm_client.py [research questions]
    └── trading.backtest
          ├── trading.technical_analysis (monkeypatched) [RISK: fragile]
          └── trading.swarm.quant_metrics [metrics]
```

---

## DATA VERSIONING GAPS (Critical for Reproducibility)

Every experiment MUST identify:
- [x] `code_version` — NOT IMPLEMENTED (Phase 1)
- [x] `dataset_version` — NOT IMPLEMENTED (Phase 1 — Yahoo version unknown)
- [x] `universe_version` — NOT IMPLEMENTED (Phase 2)
- [x] `feature_version` — NOT IMPLEMENTED (Phase 2)
- [x] `strategy_version` — NOT IMPLEMENTED (Phase 3)
- [x] `model_version` — NOT IMPLEMENTED (Phase 3)
- [x] `random_seed` — PARTIAL (env var exists, no seed call)
- [x] `LLM model + temperature` — PARTIAL (logged in state, not in every log event)

---

*End of audit. All findings verified against source code. Next action: implement docs/ARCHITECTURE_AUDIT.md and begin Phase 0 fixes.*
