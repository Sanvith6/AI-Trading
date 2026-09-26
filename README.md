# 📈 AI Trading Platform — NSE/BSE India (Paper-Trading, FREE data)

Autonomous AI trading stack: technical analysis (RSI/MACD/Bollinger/EMA/volume),
paper order manager with real ledger, risk manager, single trading agent,
Darwinian swarm arena, Streamlit dashboard, and backtester — all on **FREE Yahoo
Finance data by default. No broker keys needed.**

## Broker API decision (checked properly)

| Option | Cost | Auto-login | Verdict |
|---|---|---|---|
| Angel One SmartAPI | FREE | Official TOTP `generateSession` — stable | Good **optional** live broker |
| Zerodha Kite Connect | ~₹2000/mo + GST (+₹500/mo historical) | No official auto-login; daily manual login ~7:30am; TOTP-screen-scrape breaks on every login-flow change | Keep as **optional manual-token** module only |

**This project now defaults to `BROKER=paper`**: Yahoo Finance quotes/OHLCV +
simulated fills in `trading/paper_portfolio.json` + CSV log in
`trading/trade_log.csv`. Set `BROKER=angel` (+ Angel keys in `.env`) to route
quotes/orders to Angel One; `trading/kite_auth.py` remains for Zerodha manual-token use.

## ⚡ Quick Start (paper, no keys)

```bash
pip install -r requirements.txt
cp .env.example .env   # only OPENROUTER_API_KEY needed for AI reasoning
# Dashboard:
streamlit run trading/trading_app.py --server.port 8502
# or double-click run_trading.bat
# Backtest (no keys):
python -m trading.backtest RELIANCE --days 365
# or double-click run_backtest.bat
```

## File Structure (trading)

```
├── trading/
│   ├── data_provider.py   # FREE Yahoo Finance quotes/OHLCV (default source)
│   ├── technical_analysis.py  # RSI/MACD/BB/EMA/SMA/volume → BUY/SELL/HOLD
│   ├── order_manager.py   # Paper ledger + optional Angel live routing
│   ├── risk_manager.py    # Per-order / daily-loss / position-count guards
│   ├── market_data.py     # Angel One live module (BROKER=angel only)
│   ├── angel_auth.py      # Angel TOTP session (optional)
│   ├── kite_auth.py       # Zerodha manual-token module (optional, paid API)
│   ├── trading_agent.py   # AI agent (15 tools: 8 trading + 7 system)
│   ├── backtest.py        # Walk-forward backtester on free history
│   ├── trading_app.py     # Streamlit dashboard (signals/portfolio/swarm)
│   ├── paper_portfolio.json / trade_log.csv  # Paper state (auto-created)
│   └── swarm/             # Darwinian arena (DNA/evolution/graveyard/dash)
```

## 🤖 OpenRouter Autonomous AI Agent & LLM Suite

> Company-grade stack report: see **[TECHNOLOGIES.md](TECHNOLOGIES.md)** —
> every inevitable tool/algorithm (ATR/ADX, Kelly, Sharpe, BLX evolution,
> NSE cost model, risk gates) and the daily operating flow.

This workspace contains a full-featured **Autonomous AI Agent** that can execute real tasks on your system using OpenRouter, configured with `inclusionai/ling-3.0-flash-vl:free`.

---

## ⚡ Quick Start Options

You can interact with your Agent or Model in three ways:

### 1. Rich Terminal Agent Console (Interactive CLI)
```bash
python agent_cli.py
```
*(Or double-click [`run_agent.bat`](file:///c:/GRAG/AI/run_agent.bat))*

- **Live animated spinners** and real-time execution steps
- **Tool call panels** displaying exact parameters and outputs
- **Safe Mode**: Toggle with `/safe` to prompt for confirmation before commands or file edits run
- **Commands**: `/tools`, `/safe`, `/clear`, `/model`, `/help`, `/exit`

---

### 2. Streamlit Web Dashboard (Browser)
```bash
streamlit run agent_app.py
```
*(Or double-click [`run_agent_web.bat`](file:///c:/GRAG/AI/run_agent_web.bat))*

- **Collapsible tool execution accordions** showing full arguments & return values
- **Real-time reasoning status box** (Step-by-step trace)
- **Built-in Workspace File Inspector** in sidebar
- **One-click quick task buttons**

---

### 3. Lightweight Chat & Quickstart
- **Simple 1-query test**: `python quickstart.py`
- **Pure text terminal chat**: `python chat.py` (or double-click [`run_chat.bat`](file:///c:/GRAG/AI/run_chat.bat))
- **Pure text web chat**: `streamlit run app.py` (or double-click [`run_web.bat`](file:///c:/GRAG/AI/run_web.bat))

---

## 🛠️ Built-in Agent Tools

The agent has access to 7 built-in tools defined in [`tools.py`](file:///c:/GRAG/AI/tools.py):

| Tool | Purpose |
|---|---|
| `execute_command` | Runs PowerShell commands on your system and returns terminal stdout/stderr |
| `read_file` | Inspects contents of files (supports line ranges) |
| `write_file` | Creates or modifies files |
| `list_directory` | Explores directories and lists file sizes |
| `search_files` | Searches workspace files with regex or text patterns |
| `run_python` | Runs Python code snippets in an isolated subprocess |
| `fetch_url` | Fetches webpage text or REST API data from the web |

---

## 🐍 Using the Agent in Python Code

```python
from agent import Agent

# Create an agent instance
agent = Agent()

# Run a task autonomously
task = "Search the project for where API keys are loaded, then create a summary file."
result = agent.run(task)
print(result)
```

### Real-Time Event Streaming:
```python
from agent import Agent

agent = Agent()
for event in agent.step_stream("Calculate 999 * 888 using python"):
    if event.type == "tool_start":
        print(f"Tool called: {event.data['tool_name']}")
    elif event.type == "tool_end":
        print(f"Result: {event.data['result']}")
    elif event.type == "final_answer":
        print(f"Answer: {event.data['content']}")
```

---

## ⚙️ Configuration & Model Switching

All settings are configured in [`.env`](file:///c:/GRAG/AI/.env):

```env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=inclusionai/ling-3.0-flash-vl:free
```

You can change to any OpenRouter model (e.g., `deepseek/deepseek-r1:free`, `meta-llama/llama-3.3-70b-instruct:free`, `anthropic/claude-3.5-sonnet`) simply by editing `OPENROUTER_MODEL`.

---

## 📁 File Structure

```
├── .env                 # OpenRouter API Key & configuration (keep private)
├── .env.example         # Template for environment setup
├── .gitignore           # Prevents committing .env and cache files
├── README.md            # Documentation & guide
├── requirements.txt     # Python dependencies
│
├── tools.py             # Tool definitions & OpenAI function schemas
├── agent.py             # Core autonomous ReAct agent loop
│
├── agent_cli.py         # Rich interactive terminal agent console
├── agent_app.py         # Full-featured Streamlit agent web dashboard
│
├── llm_client.py        # OpenAI SDK wrapper for OpenRouter
├── quickstart.py        # Single-prompt test script
├── chat.py              # Terminal text chat
├── app.py               # Streamlit text chat
│
├── run_agent.bat        # Double-click launcher for Terminal Agent
├── run_agent_web.bat    # Double-click launcher for Web Agent Dashboard
├── run_chat.bat         # Double-click launcher for Terminal Chat
└── run_web.bat          # Double-click launcher for Web Chat
```
