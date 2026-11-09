# Alpaca MCP Server V2 Setup & Security Architecture Guide

## 1. Overview & Purpose

The **Alpaca Model Context Protocol (MCP) Server V2** provides Antigravity AI agents with a strictly controlled, read-only diagnostic and research interface into the Alpaca paper trading ecosystem.

It allows conversational agents to query real-time market data, inspect paper account balances, read corporate actions, and analyze market news directly within the IDE, **without** providing any capability to submit, modify, or cancel orders.

> [!IMPORTANT]
> **MCP is NOT in the automated trade execution path.**
> All trading decisions and order executions in this repository are handled deterministically by the internal trading runtime (`TradingOrchestrator` &rarr; `RiskEngine` &rarr; `ExecutionAdapter`). MCP exists solely as an external, read-only research and telemetry interface for the human operator and assistant.

---

## 2. Architectural Separation & Invariants

```
+-----------------------------------------------------------------------------------+
|                           EXTERNAL RESEARCH INTERFACE                             |
|                                                                                   |
|   Antigravity Agent  <--[JSON-RPC stdio]-->  Alpaca MCP Server V2 (uvx)           |
|                                                       |                           |
|                        Read-Only Tools Only:          | (HTTPS Read)              |
|                        - get_account_info             v                           |
|                        - get_stock_bars        Alpaca Paper API                   |
|                        - get_news              (paper-api.alpaca.markets)         |
|                        - get_corporate_actions                                    |
+-----------------------------------------------------------------------------------+
                                          |
                                    STRICT AIR-GAP
                                          |
+-----------------------------------------------------------------------------------+
|                        DETERMINISTIC TRADING PIPELINE                             |
|                                                                                   |
|   Market Data & Providers                                                         |
|             |                                                                     |
|             v                                                                     |
|   Feature Engineering & Alpha Models (12 Families)                                |
|             |                                                                     |
|             v                                                                     |
|   Meta-Model + Laya V2 Specialist (Shadow Mode, Frozen Weights, Zero Broker Access)|
|             |                                                                     |
|             v                                                                     |
|   Regime Detection & Candidate Ranking                                            |
|             |                                                                     |
|             v                                                                     |
|   Portfolio Constructor                                                           |
|             |                                                                     |
|             v                                                                     |
|   RiskEngine (Pre-Trade Invariants, Drawdown Limits, Position Sizing)             |
|             |                                                                     |
|             v                                                                     |
|   OrderIntent                                                                     |
|             |                                                                     |
|             v                                                                     |
|   ExecutionAdapter (Local Paper / Alpaca Paper Broker / Guarded Live)             |
+-----------------------------------------------------------------------------------+
```

### Core Invariants
1. **Zero Trade Authority**: The MCP server is configured to exclude the `trading` and `watchlists` toolsets. The agent has zero tools to place, cancel, modify, or exercise orders.
2. **Deterministic Runtime Isolation**: `TradingOrchestrator`, `RiskEngine`, `AlpacaPaperBrokerAdapter`, and `LiveTradingGuard` contain zero MCP dependencies or imports.
3. **Laya V2 Specialist Isolation**: The 421M parameter `laya-financial-us-v2` checkpoint remains completely frozen, shadow-isolated (`LAYA_MODE=shadow`), and has zero access to broker credentials or MCP servers.
4. **Paper-Only Guarantee**: `ALPACA_PAPER_TRADE` is hard-locked to `"true"`, enforcing connections to `https://paper-api.alpaca.markets`. No live endpoints or credentials are ever provided to MCP.

---

## 3. Installation Prerequisites

The MCP server runs via `uvx` (part of the Astral `uv` toolsuite), which downloads and executes `alpaca-mcp-server` V2 in an isolated ephemeral environment without polluting your Python runtime.

### Checking uv & uvx

#### On Windows (PowerShell):
```powershell
uv --version
uvx --version
```

#### On Linux / WSL2:
```bash
uv --version
uvx --version
```

If `uv` / `uvx` is not installed, install it using the official standalone installer:
- **Windows**: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **Linux/WSL**: `curl -LsSf https://astral.sh/uv/install.sh | sh`

---

## 4. Antigravity MCP Workspace Configuration

The configuration file is located at [`.agents/mcp_config.json`](file:///.agents/mcp_config.json) in the workspace root.

```json
{
  "mcpServers": {
    "alpaca-paper-readonly": {
      "command": "uvx",
      "args": [
        "alpaca-mcp-server"
      ],
      "env": {
        "ALPACA_API_KEY": "$ALPACA_PAPER_API_KEY",
        "ALPACA_SECRET_KEY": "$ALPACA_PAPER_SECRET_KEY",
        "ALPACA_PAPER_TRADE": "true",
        "ALPACA_TOOLSETS": "account,stock-data,assets,news,corporate-actions"
      },
      "excludeTools": [
        "update_account_config"
      ],
      "trust": false
    }
  }
}
```

### Configuration Parameters

| Parameter | Value | Description |
|---|---|---|
| **Server Name** | `alpaca-paper-readonly` | Unique identifier in Antigravity |
| **Command** | `uvx` | Astral uv tool runner |
| **Args** | `["alpaca-mcp-server"]` | Resolves official Alpaca MCP Server V2 (v2.3.2+) |
| **ALPACA_API_KEY** | `$ALPACA_PAPER_API_KEY` | Environment variable expansion (no hardcoded secret) |
| **ALPACA_SECRET_KEY** | `$ALPACA_PAPER_SECRET_KEY` | Environment variable expansion (no hardcoded secret) |
| **ALPACA_PAPER_TRADE** | `"true"` | Hardcoded safety flag forcing paper trading endpoints |
| **ALPACA_TOOLSETS** | `account,stock-data,assets,news,corporate-actions` | Server-side toolset filter excluding trading and watchlists |
| **excludeTools** | `["update_account_config"]` | Client-side exclusion stripping account mutation/write capabilities |
| **trust** | `false` | Enforces Antigravity confirmation prompts; prevents bypassing tool-call confirmations |

---

## 5. Toolset Configuration & Defense-in-Depth Architecture

To guarantee a strictly read-only, non-mutating interface, the system implements **defense-in-depth across two layers**:

```
                       Alpaca MCP Server V2
                                │
                                ▼
         Layer 1: Server-Side Filtering (ALPACA_TOOLSETS)
       "account,stock-data,assets,news,corporate-actions"
    [33 initial tools: eliminates trading & watchlists toolsets]
                                │
                                ▼
         Layer 2: Client-Side Exclusion (excludeTools)
                  ["update_account_config"]
          [Strips account configuration write capability]
                                │
                                ▼
      32 Explicitly Approved Read-Only Research & Telemetry Tools
       [Zero order tools, zero mutation tools, zero write access]
```

### The 32 Explicitly Approved Read-Only Tools

1. **Account & Portfolio Telemetry (5 tools)**:
   - `get_account_info` (cash, buying power, equity, portfolio values)
   - `get_account_activities` (historical transactions, dividends, transfers)
   - `get_account_activities_by_type` (filtered activity audit)
   - `get_account_config` (read-only view of account settings)
   - `get_portfolio_history` (historical equity curve and PnL)

2. **Market Data & Historical Prints (13 tools)**:
   - `get_stock_bars` (historical OHLCV bars)
   - `get_stock_latest_bar` (latest 1-min/daily bar)
   - `get_stock_quotes` (historical bid/ask quotes)
   - `get_stock_latest_quote` (current NBBO bid/ask)
   - `get_stock_snapshot` (full ticker snapshot)
   - `get_stock_trades` (historical public tape trades)
   - `get_stock_latest_trade` (latest executed trade print on tape)
   - `get_crypto_bars` (crypto historical bars)
   - `get_crypto_quotes` (crypto quotes)
   - `get_crypto_trades` (crypto public trade prints)
   - `get_market_movers` (top gainers/losers)
   - `get_most_active_stocks` (volume leaders)
   - `get_news` (market news sentiment & articles)

3. **Asset & Option Metadata (5 tools)**:
   - `get_all_assets` (tradeable assets directory)
   - `get_asset` (individual asset metadata and status)
   - `get_option_contract` (option contract details)
   - `get_option_contracts` (option chain inquiry)
   - `get_clock` / `get_calendar` (NYSE market hours & settlement calendar)

4. **Corporate Actions & Documentation (9 tools)**:
   - `get_corporate_actions` (dividends, splits, mergers)
   - `get_corporate_action_announcement` (specific action notice)
   - `get_corporate_action_announcements` (historical announcement records)
   - `search_alpaca_docs` / `fetch_alpaca_doc` (API doc search)
   - `search_alpaca_api_specs` / `list_alpaca_api_endpoints` / `get_alpaca_endpoint_docs` (endpoint specifications)

### Excluded & Prohibited Operations (Zero Exposure)
- **Order Placement**: `place_stock_order`, `place_crypto_order`, `place_option_order` &rarr; **BLOCKED**
- **Order Modification**: `replace_order_by_id`, `exercise_options_position`, `do_not_exercise_options_position` &rarr; **BLOCKED**
- **Order Cancellation**: `cancel_all_orders`, `cancel_order_by_id` &rarr; **BLOCKED**
- **Position Liquidations**: `close_all_positions`, `close_position` &rarr; **BLOCKED**
- **Account Configuration Mutation**: `update_account_config` &rarr; **BLOCKED**

**Result**: 32 explicitly approved read-only tools; all trading/order tools and account-configuration mutation tools are excluded.

---

## 6. Environment Variables & Credential Handling

### Security Rules
- **NEVER** commit API keys or secret keys to source control.
- **NEVER** write literal secrets into `.agents/mcp_config.json`.
- The MCP configuration uses variable substitution syntax (`$ALPACA_PAPER_API_KEY`), which reads from the host operating system's environment.

### Setting Credentials on the Host

#### Windows PowerShell:
```powershell
$env:ALPACA_PAPER_API_KEY = "PK..."
$env:ALPACA_PAPER_SECRET_KEY = "..."
```

#### Linux / macOS / WSL2:
```bash
export ALPACA_PAPER_API_KEY="PK..."
export ALPACA_PAPER_SECRET_KEY="..."
```

#### Verification:
Verify the presence of keys without printing them:
```powershell
if ($env:ALPACA_PAPER_API_KEY) { Write-Output "ALPACA_PAPER_API_KEY is SET" } else { Write-Output "NOT SET" }
if ($env:ALPACA_PAPER_SECRET_KEY) { Write-Output "ALPACA_PAPER_SECRET_KEY is SET" } else { Write-Output "NOT SET" }
```

---

## 7. Reloading & Verifying the MCP Server

### Reloading Antigravity MCP Client
1. In the Antigravity IDE, open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`).
2. Search for `Antigravity: Reload MCP Servers` or restart the IDE session.
3. Check the MCP status tray or server log to verify that `alpaca-paper-readonly` is connected.

### Verifying Tool Discovery & Toolset Constraints
Run the repository test suite to verify configuration validity, tool restrictions, and zero order-placement tools:

```bash
# In WSL2 CUDA environment:
PYTHONPATH=. pytest trading/tests/test_alpaca_mcp_config.py -v

# Or on Windows:
pytest trading/tests/test_alpaca_mcp_config.py -v
```

---

## 8. Relationship to Trading Runtime & Laya V2

### Deterministic Trading Pipeline
The trading runtime does NOT interact with MCP. When running paper or live trading:
- Market data is collected via `AlpacaMarketDataProvider` or `YahooMarketDataProvider`.
- Alpha signals are produced deterministically across 12 feature families.
- Portfolio construction and sizing are strictly governed by `RiskEngine`.
- Order submission is executed through `AlpacaPaperBrokerAdapter` or `PaperExecutionAdapter`.

### Laya V2 Specialist
- Checkpoint: `data_lake/laya_specialist/checkpoints/laya-financial-us-v2`
- Checkpoint SHA256 integrity is verified on every test run.
- Laya executes exclusively in **shadow mode** (`LAYA_MODE=shadow`).
- Laya outputs logged to `trading/laya_shadow_journal/`.
- Laya possesses zero network access, zero broker credentials, and zero MCP access.

---

## 9. Troubleshooting

### Issue: `uvx` command not found
- Ensure `uv` is installed and that the directory containing `uvx` (typically `~/.local/bin` on Linux or `%USERPROFILE%\.cargo\bin` on Windows) is in your system `PATH`.

### Issue: MCP Server fails with "Missing credentials"
- Confirm that `ALPACA_PAPER_API_KEY` and `ALPACA_PAPER_SECRET_KEY` are exported in the environment where Antigravity was launched.
- Remember to restart Antigravity after setting new system environment variables.

### Issue: Forbidden trading tools are visible
- Verify that `ALPACA_TOOLSETS` in `.agents/mcp_config.json` is set exactly to:
  `account,stock-data,assets,news,corporate-actions`
- Ensure that `trading` is NOT present in the comma-delimited string.
