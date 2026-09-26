# PROJECT WALKTHROUGH — AI Trading Platform (`ai-trading/`)
Machine-readable complete analysis for an AI agent taking over this repo.
Human companion docs: `README.md` (quickstart), `TECHNOLOGIES.md` (stack report).

## 1. What this project is
Two systems in one repo, sharing the OpenRouter LLM client:
- **A. General-purpose autonomous agent** (root files): ReAct loop with 7 system
  tools (shell, files, Python, web). Terminal + web UIs.
- **B. AI stock-trading platform** (`trading/` package): paper-trading by default
  on FREE Yahoo Finance data; optional live brokers (Angel One free API,
  Zerodha paid API). Single `TradingAgent` + 4-agent Darwinian `Swarm Arena` +
  Streamlit dashboard + backtester. **No real money moves unless
  `BROKER=angel` AND `TRADE_MODE=live`.**

## 2. File inventory (every file, one line)
```
.env                    # live config (gitignored). OPENROUTER_API_KEY + BROKER=paper + watchlist + risk
.env.example            # template incl. BROKER, Angel/Zerodha keys, TRADE_RISK_PER_TRADE_PCT, LOG_LEVEL
.gitignore              # ignores .env, __pycache__, venv, .streamlit/secrets.toml
README.md / TECHNOLOGIES.md / PROJECT_WALKTHROUGH.md (this file)
requirements.txt        # openai, dotenv, streamlit, rich, yfinance, pandas, numpy, plotly,
                        # requests, smartapi-python, kiteconnect, logzero, pyotp
llm_client.py           # get_client(), ask(), ChatSession; DEFAULT_MODEL from env
tools.py                # 7 base tools + TOOL_DEFINITIONS/REGISTRY/call_tool; WORKSPACE_DIR
agent.py                # AgentEvent + Agent (ReAct). Extension hooks: get_tool_defs(), call_tool_fn()
agent_cli.py            # rich terminal console (/tools /safe /clear /model /help /exit)
agent_app.py            # Streamlit generic-agent dashboard (port 8501 typical)
app.py / chat.py / quickstart.py   # plain LLM chat (web / terminal / one-shot), no tools
run_*.bat               # Windows launchers (see §9)
trading/__init__.py / trading/swarm/__init__.py  # package markers
trading/logging_setup.py# setup_logging() → console + rotating trading/agent.log (1MB×3)
trading/data_provider.py# DEFAULT market data: Yahoo Finance quotes/OHLCV (.NS suffix map)
trading/market_data.py  # OPTIONAL live data via Angel SmartAPI (BROKER=angel only)
trading/angel_auth.py   # Angel TOTP session, daily cache in trading/session.json
trading/kite_auth.py    # Zerodha manual-token module (paid API, no auto-login)
trading/technical_analysis.py  # indicators → BUY/SELL/HOLD + confidence + ATR/ADX
trading/order_manager.py# paper ledger (₹100k) + costed fills; live route only if angel+live
trading/risk_manager.py # kill-switches + position tracking (trading/risk_state.json)
trading/trading_agent.py# TradingAgent(Agent): 8 trading tools + prompt + market loop
trading/backtest.py     # walk-forward backtester on free history, with costs + quant metrics
trading/trading_app.py  # Streamlit dashboard :8502 — standard view + swarm view
trading/swarm/agent_dna.py      # AgentDNA genome/health/ledger + Gen-1 cohort factory
trading/swarm/arena_manager.py  # tournament rounds, treasury, evolution trigger
trading/swarm/evolution_engine.py # fitness, tournament+BLX breeding, LLM prompt mutation
trading/swarm/genetic_ops.py    # tournament_selection, blx_alpha_crossover,
                                # recombine_parameters (bounded), enforce_speciation
trading/swarm/graveyard.py      # burial, liquidation, LLM autopsy, lessons bank
trading/swarm/quant_metrics.py  # NSE costs, Sharpe/Sortino/maxDD/Calmar/PF/Kelly, summarize_trades()
trading/swarm/swarm_dashboard.py# 5-tab arena UI rendered inside trading_app.py
trading/swarm/test_swarm.py / test_quant_wiring.py  # test suites (both passing)
```

## 3. Architecture
```
Yahoo Finance (free) ──► data_provider ──┬──► technical_analysis ──► signals
Angel SmartAPI (opt.) ─► market_data ────┘                              │
Paper ledger ◄── order_manager (costs) ◄── TradingAgent (15 tools, LLM) │
      ▲                         ▲              │ check_risk_limits      │
      │                    risk_manager ◄──────┘                        │
      ▼                                                                ▼
trade_log.csv / paper_portfolio.json / risk_state.json        trading_app.py :8502
Swarm: 4×AgentDNA → arena rounds (SL/TP, LLM votes) → graveyard → evolution → Gen N+1
```

## 4. Module deep dive
### 4.1 `llm_client.py` — OpenRouter access
- `get_client()` OpenAI SDK pointed at `https://openrouter.ai/api/v1` + headers.
- `ask(prompt, system_prompt, model, stream, temperature)` — `max_tokens=2048` default.
- `ChatSession` — multi-turn history with streaming support.
- `DEFAULT_MODEL` ← `OPENROUTER_MODEL`/`MODEL` env (default `inclusionai/ling-3.0-flash-vl:free`).

### 4.2 `tools.py` — 7 base tools
`execute_command` (PowerShell, cwd=repo), `read_file` (line ranges), `write_file`
(overwrite/append), `list_directory`, `search_files` (regex, 30-hit cap),
`run_python` (subprocess), `fetch_url` (HTML-stripped, 4000 chars).
`call_tool(name, args)` dispatches via `TOOL_REGISTRY`; unknown → error string
(never raises — agent loop depends on this).

### 4.3 `agent.py` — ReAct loop (THE extension contract)
- `Agent(model, system_prompt, max_steps=15, safe_mode, confirm_callback, temperature=0.4)`.
- **`get_tool_defs()` → tool schemas; `call_tool_fn(name, args)` → executes.**
  Subclasses override these (TradingAgent does). `step_stream()` calls
  `self.get_tool_defs()` / `self.call_tool_fn()` — never the globals directly.
  ⚠️ Do NOT reintroduce direct `tools.TOOL_DEFINITIONS`/`tools.call_tool` calls
  in `step_stream` (that was a real past bug: trading tools silently never fired).
- `step_stream(user_instruction)` yields `AgentEvent(user_message|step_start|
  thought|tool_start|tool_end|tool_denied|error|final_answer|max_steps_reached)`.
  `run()` drains it synchronously. Sensitive tools (`execute_command`,
  `write_file`) gated by `safe_mode + confirm_callback`.

### 4.4 `trading/data_provider.py` — default market data (no keys)
- `BROKER` env (default `paper`); Yahoo symbols via `_to_yahoo_symbol` (+`.NS`).
- `get_quote(symbol)` → `{symbol,last_price,open,high,low,close,volume,change,
  change_pct,source}`; error dict with `last_price: 0` + `error` on failure.
- `get_ohlcv(symbol, interval, days)` normalizes `day/daily/D/1D→1d` etc.;
  intraday ranges clamped to Yahoo limits; flattens MultiIndex columns.
- `get_watchlist_quotes(watchlist=None)` (defaults to `TRADE_WATCHLIST` or
  `RELIANCE,TCS,INFY`), `quotes_to_map()`, `get_ltp()`.
- Dead symbols (e.g. ex-`TATAMOTORS`) return error dicts — callers must handle
  `last_price == 0` (all current callers do).

### 4.5 `trading/market_data.py` — Angel live module (dormant unless BROKER=angel)
Static `SYMBOL_TOKENS` + `searchScrip` fallback in `_get_token`; `get_quote`
tries `ltpData` then FULL `getMarketData`; `get_ohlcv` alias-normalized;
`get_portfolio` (`holding`/`position`/`rmsLimit`), `get_orders` (`orderBook`).

### 4.6 Auth modules
- `angel_auth.get_smartapi_client(force_refresh)` → `(obj, auth_token, feed_token)`,
  TOTP via `pyotp`, daily cache. `get_client()` convenience wrapper.
- `kite_auth` → `get_kite_client`, `manual_token_login` (Zerodha paid API).
- ⚠️ GOTCHA: both use `trading/session.json` as `SESSION_FILE`. Never enable
  both brokers simultaneously without renaming one file.

### 4.7 `trading/technical_analysis.py` — signal engine
Indicators: RSI14, MACD(12,26,9)+histogram, Bollinger(20,2σ), EMA9/21, SMA50/200,
volume spike 1.5×20d avg, **ATR14** (2×ATR stop / 3×ATR target guides),
**ADX14** regime (`TRENDING≥20`, `CHOPPY<15`). Scoring: RSI/MACD/BB/EMA/SMA/volume
vote buy vs sell → confidence = majority share; `<60%` → HOLD; chop + `<75%` →
HOLD. `analyze()` returns `{symbol,signal,confidence,buy/sell scores,last_price,
indicators{...},reasons[]}`; `generate_signal()` formats it for LLM prompts.
Needs ≥30 bars or HOLD with reason.

### 4.8 `trading/order_manager.py` — execution + ledger
- Paper state `trading/paper_portfolio.json` `{cash: 100000.0, positions:
  {SYM: {quantity, avg_price}}}`. Every fill appended to `trading/trade_log.csv`.
- `place_buy`: LTP default → qty auto (`MAX_PER_ORDER/price`) → **risk cap
  shrinks qty to `RISK_PER_TRADE_PCT`% of cash** → deducts cost+round-trip fees →
  avg-price update. Insufficient cash → `{"status":"error"}` (no exception).
- `place_sell`: sells min(held, qty); deducts sell-leg fees from proceeds.
- Angel live path executes ONLY if `TRADE_MODE==live` AND `BROKER==angel`.
- `exit_position` (paper: full position at 0.995×LTP), `cancel_order`
  (paper: instant pseudo-cancel), `get_paper_portfolio()` (positions @ Yahoo LTP
  with per-line P&L), `get_trade_log()`.

### 4.9 `trading/risk_manager.py`
`RiskManager` over `trading/risk_state.json` `{date, daily_pnl, open_positions}`
(auto-resets daily). `is_trade_allowed(symbol, order_value, action)` blocks on:
daily-loss breach, max-positions (BUY only), order-size cap. `on_buy/on_sell`
track P&L/positions; `should_stop_loss/should_take_profit`; `get_status()` adds
`trading_halted`. Note: day-P&L only moves via `on_buy/on_sell` calls.

### 4.10 `trading/trading_agent.py` — the autonomous trader
8 trading tools (see §6) + 7 base = **15 tool schemas**. Paper-mode system prompt
with watchlist + ₹/position/SL/TP rules injected from env. `TradingAgent(Agent)`:
`max_steps=20`, `temperature=0.3`, overrides `get_tool_defs/call_tool_fn`.
`_tool_get_portfolio_status` serves paper portfolio (Angel snapshot only if
enabled). `run_market_scan(on_event)` = one instructed cycle; 
...[truncated 6157 chars]