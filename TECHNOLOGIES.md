# Inevitable Tools & Technologies — AI Trading Platform (company-grade report)

Stack principle: **paper-first, free data, real accounting**. Every rupee of cost,
every risk metric and every evolutionary operator below is wired into live code
paths — not demo stubs. Verified end-to-end (see Tests).

## 1. Runtime & Framework

| Technology | Version | Why inevitable | Status |
|---|---|---|---|
| Python | 3.12 | Entire stack language | ✅ |
| Streamlit | ≥1.30 | Dashboard for signals, portfolio, swarm arena, agent controls (`trading/trading_app.py`) | ✅ |
| pandas / numpy | ≥2.0 / ≥1.24 | OHLCV frames, indicator math | ✅ |
| plotly | ≥5.0 | Charts | ✅ |
| rich | ≥13 | Terminal agent console (`agent_cli.py`) | ✅ |
| python-dotenv | ≥1.0 | `.env` config (`BROKER`, risk limits, keys) | ✅ |
| Rotating file logging | stdlib | `trading/logging_setup.py` → console + `trading/agent.log` (1 MB × 3). Previously loggers existed but logged nowhere | ✅ NEW |

## 2. Market Data (paper default: FREE, no keys)

| Technology | Why inevitable | Status |
|---|---|---|
| Yahoo Finance (`yfinance`) | Free NSE quotes + history (`RELIANCE.NS`); only keyless legal source good enough for paper trading | ✅ default (`trading/data_provider.py`) |
| Angel One SmartAPI (`smartapi-python`) | Free official broker API with TOTP session; optional live route (`BROKER=angel`) | ✅ optional |
| Zerodha Kite Connect (`kiteconnect`) | Industry-standard broker API, but ~₹2000/mo + daily manual login; kept as manual-token module only | ✅ optional |

## 3. Strategy Algorithms (best-in-class for retail momentum/mean-reversion)

| Algorithm | Role | Where |
|---|---|---|
| RSI(14) | Overbought/oversold scoring | `technical_analysis.py` |
| MACD(12,26,9) crossover + histogram | Momentum regime | same |
| Bollinger Bands(20, 2σ) | Overextension entries | same |
| EMA 9/21 cross + SMA 50/200 | Trend filter | same |
| Volume spike (1.5× 20-day avg) | Confirmation weight | same |
| **ATR(14)** | Volatility ruler; emits 2×ATR stop / 3×ATR target guides | ✅ NEW |
| **ADX(14) regime filter** | Chop (ADX<15) downgrades signals to HOLD — trend edges die in chop | ✅ NEW |
| Composite BUY/SELL/HOLD vote with 60% confidence gate | Single explainable decision per symbol | ✅ |

## 4. Risk Framework (the part companies audit first)

| Control | Rule | Where |
|---|---|---|
| Per-order ₹ cap | `TRADE_MAX_PER_ORDER` (₹5000) | `risk_manager.py` + `order_manager.py` |
| **Per-trade % of equity cap** | `TRADE_RISK_PER_TRADE_PCT` (5% of paper cash), auto-shrinks qty | ✅ NEW (`order_manager.py`) |
| Daily loss kill-switch | Halt at `TRADE_MAX_DAILY_LOSS` (₹2000), persisted in `risk_state.json` | ✅ |
| Max concurrent positions | `TRADE_MAX_POSITIONS` (3) | ✅ |
| Stop-loss / take-profit exits | Per-agent genome %, checked every arena round | ✅ |
| Pre-trade gate | Agent **must** call `check_risk_limits` before `place_trade` | ✅ (`trading_agent.py`) |
| Real NSE cost model | STT + stamp + exchange + SEBI + GST + slippage deducted on **both** legs | ✅ NEW wired into fills |

## 5. Execution & Accounting

| Piece | Detail |
|---|---|
| Paper ledger | `trading/paper_portfolio.json` — cash, avg-price positions, insufficient-funds guard |
| Trade audit log | `trading/trade_log.csv` — every fill with cost notes |
| Cost engine | `compute_indian_transaction_costs` (CNC delivery: 0.1% STT both legs, 0.015% stamp buy, NSE 0.00345%, SEBI, 18% GST, 5 bps slippage) |
| Backtester | `trading/backtest.py` — walk-forward signals on free history **with costs**, outputs Sharpe/Sortino/maxDD/Calmar/profit-factor/Kelly |

## 6. Quant Measurement (previously dead code — now wired)

`trading/swarm/quant_metrics.py` + `summarize_trades()` feeds the backtester,
evolution fitness and (via logs) the dashboard: Sharpe, Sortino, max drawdown,
Calmar, profit factor, win rate, expectancy, half-Kelly size.

## 7. Darwinian Swarm (evolutionary framework)

| Operator | Algorithm | Status |
|---|---|---|
| Selection | Tournament selection (k=3) on fitness | ✅ NEWLY WIRED (`genetic_ops.py` → `evolution_engine.py`) |
| Crossover | BLX-α blended crossover on numeric genome (bounded) | ✅ NEWLY WIRED |
| Mutation | Adaptive Gaussian + LLM prompt mutation with graveyard-immunity directives | ✅ |
| Speciation | 4 enforced niches so one regime can't kill the whole swarm | ✅ NEWLY WIRED |
| Fitness | gain%×3 + win-rate×0.5 + health×0.5 **+ profit-factor edge** | ✅ upgraded |
| Memory | Graveyard autopsies → `lessons_learned.json` → injected into offspring prompts | ✅ |
| Accounting | Mark-to-market `equity()`; treasury reconciled on live quotes, not stale cost | ✅ NEW |

## 8. AI Reasoning

| Piece | Detail |
|---|---|
| OpenRouter (OpenAI-compatible) | `inclusionai/ling-3.0-flash-vl:free` default, swappable via `.env` |
| ReAct agent loop | `agent.py` — thought → tool calls → verification, 15–20 steps, safe mode |
| Trading tools (15 total) | 8 trading (`get_market_data`, `get_technical_signal`, `get_portfolio_status`, `check_risk_limits`, `place_trade`, `get_watchlist_scan`, `get_news_sentiment`, `get_trade_history`) + 7 system |
| Swarm reasoning | Per-agent archetype prompts, strict-JSON decisions, confidence gates |

## 9. Daily Operating Flow

1. `run_trading.bat` → dashboard `:8502` (PAPER badge). **Sunday/after-hours: agent idles** (market gate 09:15–15:20 IST, Mon–Fri).
2. **Refresh Signals** → vote + confidence + ATR/ADX regime per watchlist symbol.
3. **Run Single Scan Now** → agent reasons with tools, respects risk gates, paper-fills post costs.
4. **Start Autonomous Agent** → scan every 5 min in hours; activity log streams.
5. **Swarm Arena** → `Scan Round` (trade + SL/TP + HP) → `Evolve Gen` (tournament → BLX breed → speciate).
6. **Backtest** (`run_backtest.bat`) before trusting any parameter change.
7. Review `trade_log.csv`, `trading/agent.log`, quant summaries.

## 10. Tests (all passing)

- `trading/swarm/test_swarm.py` — arena lifecycle, death, burial, Gen-2 breeding
- `trading/swarm/test_quant_wiring.py` — quant summary, GA bounds, MTM, ATR/ADX, costed fills ✅ NEW
