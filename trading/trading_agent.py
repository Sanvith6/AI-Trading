"""
AI Trading Agent
Extends the base Agent with trading-specific tools.
Runs an autonomous market scanning loop every N minutes during market hours.
Now with mandatory RiskEngine validation — LLM proposes, RiskEngine disposes.
"""

import os
import json
import time
import logging
from datetime import datetime, time as dtime
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from agent import Agent
from tools import TOOL_DEFINITIONS, TOOL_REGISTRY, call_tool
from trading.logging_setup import setup_logging
from trading.risk_engine.validators import RiskValidator
from trading.data_provider import get_quote, get_watchlist_quotes, get_ltp
from trading.technical_analysis import generate_signal, analyze, batch_analyze
from trading.risk_manager import RiskManager
from trading.order_manager import place_buy, place_sell, exit_position, get_trade_log, get_paper_portfolio

load_dotenv(override=True)

logger = logging.getLogger(__name__)

WATCHLIST = [s.strip().upper() for s in os.getenv("TRADE_WATCHLIST", "SPY,AAPL,NVDA,MSFT,TSLA").split(",") if s.strip()]
SCAN_INTERVAL_MIN = int(os.getenv("SCAN_INTERVAL_MINUTES", 5))
MARKET = os.getenv("MARKET", "IN").strip().upper()  # US | IN
TRADE_MODE = os.getenv("TRADE_MODE", "paper")


def _market_open_close():
    """Return (open_time, close_time) as dtime objects in local wall-clock time.
    For US markets, NYSE 09:30-16:00 EST is converted to IST (add 10h30m).
    For IN markets, NSE 09:15-15:20 IST is used directly."""
    if MARKET == "US":
        # NYSE 09:30 EST = 20:00 IST, 16:00 EST = 02:30 IST next day
        # We compare in IST since datetime.now() is local (IST on this machine)
        return dtime(20, 0), dtime(2, 30)
    return dtime(9, 15), dtime(15, 20)


MARKET_OPEN, MARKET_CLOSE = _market_open_close()

# Initialize RiskEngine as immutable security boundary
# The LLM proposes → RiskEngine validates → execution proceeds only if approved
risk_validator = RiskValidator()
risk_validator.initialize({
    "TRADE_MAX_PER_ORDER": float(os.getenv("TRADE_MAX_PER_ORDER", 5000)),
    "TRADE_MAX_DAILY_LOSS": float(os.getenv("TRADE_MAX_DAILY_LOSS", 2000)),
    "TRADE_MAX_POSITIONS": int(os.getenv("TRADE_MAX_POSITIONS", 10)),
    "TRADE_MAX_EXPOSURE": 0.95,
    "TRADE_MAX_SECTOR_WEIGHT": 0.40,
    "TRADE_MAX_CORRELATED": 0.60,
    "TRADE_MIN_CASH": 1000,
    "TRADE_RISK_PER_TRADE_PCT": 5.0,
    "TRADE_STOP_LOSS_PCT": float(os.getenv("TRADE_STOP_LOSS_PCT", 2.0)),
    "TRADE_TARGET_PCT": float(os.getenv("TRADE_TARGET_PCT", 4.0)),
    "TRADE_MIN_TRADES": 30,
})

# Trading tool definitions (must be defined before TradingAgent class)
TRADING_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_market_data",
            "description": "Fetch live quote data for a stock: LTP, open, high, low, close, volume, change%",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string", "description": "US stock ticker e.g. AAPL, NVDA, SPY"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_signal",
            "description": "Run full technical analysis (RSI, MACD, Bollinger Bands, EMA crossover, volume) and get BUY/SELL/HOLD signal with confidence score and reasons.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string", "description": "US stock ticker e.g. AAPL, NVDA"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_status",
            "description": "Get current holdings, open positions, today's orders, available margin, and daily P&L risk status.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_risk_limits",
            "description": "ALWAYS call this before placing any trade. Checks if the trade is allowed given daily loss limits, max position count, and order size limits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "order_value": {"type": "number", "description": "Expected ₹ value of the order"},
                    "action": {"type": "string", "enum": ["BUY", "SELL"]},
                },
                "required": ["symbol", "order_value", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_trade",
            "description": "Place a PAPER (simulated) BUY or SELL order on NYSE/NASDAQ at Yahoo Finance LTP. Always check_risk_limits first. No real money moves.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "US stock ticker e.g. AAPL, NVDA, SPY"},
                    "action": {"type": "string", "enum": ["BUY", "SELL"]},
                    "quantity": {"type": "integer", "description": "Number of shares (optional, auto-calculated if not given)"},
                    "price": {"type": "number", "description": "Limit price (optional, uses LTP if not given)"},
                },
                "required": ["symbol", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_watchlist_scan",
            "description": "Scan all configured watchlist stocks at once and return a summary table of signals, prices, RSI, and MACD.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_sentiment",
            "description": "Fetch recent financial news headlines about a stock to assess market sentiment.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trade_history",
            "description": "Return the full trade log of all orders placed today (CSV format).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TRADING_TOOL_REGISTRY = {
    "get_market_data": lambda s: json.dumps(get_quote(s), indent=2, default=str),
    "get_technical_signal": lambda s: generate_signal(s),
    "get_portfolio_status": lambda: json.dumps(
        {"portfolio": get_paper_portfolio(), "risk_status": RiskManager().get_status()}, indent=2, default=str
    ),
    "check_risk_limits": lambda s, ov, a: json.dumps(
        risk_validator.validate_order(symbol=s, action=a, price=ov // int(ov**0.5) if ov > 0 else 0, quantity=int(ov**0.5) if ov > 0 else 0)
    ),
    "get_watchlist_scan": lambda: json.dumps(
        batch_analyze(WATCHLIST), indent=2, default=str
    ),
    "get_news_sentiment": lambda s: f"News snippets for {s}: [fetched]",
    "get_trade_history": lambda: get_trade_log(),
}


def _call_trading_tool(name: str, arguments: dict) -> str:
    if name in TRADING_TOOL_REGISTRY:
        try:
            return str(TRADING_TOOL_REGISTRY[name](**arguments))
        except Exception as e:
            return f"Error in tool '{name}': {e}"
    # Fallback to base tools
    return call_tool(name, arguments)


# ── Trading Agent Class ───────────────────────────────────────────────────────

class TradingAgent(Agent):
    """
    Autonomous AI agent specialized for NSE stock trading via FREE Yahoo Finance data.
    Extends the base Agent with market-specific tools and a timed scan loop.
    Critical: Every trade must pass RiskEngine validation before execution.
    """

    def __init__(self, **kwargs):
        super().__init__(
            system_prompt=self._build_system_prompt(),
            max_steps=20,
            temperature=0.3,
            **kwargs,
        )
        # Trading tools + base file/system tools (wired via base-class hooks
        # so step_stream actually calls them — previously ignored).
        import tools as base_tools
        self._all_tool_defs = TRADING_TOOL_DEFINITIONS + base_tools.TOOL_DEFINITIONS

    def _build_system_prompt(self) -> str:
        """Build the system prompt with risk rules embedded."""
        cash_env = os.getenv("TRADE_INITIAL_CASH", "100000")
        market_label = "US NYSE/NASDAQ" if MARKET == "US" else "Indian NSE"
        currency = "$" if MARKET == "US" else "\u20b9"
        tz_label = "EST" if MARKET == "US" else "IST"
        symbols_summary = ", ".join(WATCHLIST[:12]) + (f" and {len(WATCHLIST)-12} more" if len(WATCHLIST) > 12 else "")
        return f"""You are an autonomous AI stock trading agent operating on {market_label} markets in PAPER-TRADING mode.

Data source: FREE Yahoo Finance quotes + OHLCV (no broker login required). Orders are SIMULATED fills logged to trade_log.csv — no real money moves.

Your role:
1. Scan all {len(WATCHLIST)} watchlist stocks: {symbols_summary}
2. Analyze each stock using technical indicators (RSI, MACD, Bollinger Bands, EMA crossovers, volume)
3. Fetch recent financial news or sentiment if available
4. Check the portfolio and open positions
5. **MANDATORY: Check risk limits before any trade** — call check_risk_limits tool
6. Make a well-reasoned decision: BUY, SELL, or HOLD for each stock
7. Place orders for strong signals (confidence > 65%)
8. Monitor open positions for stop-loss and take-profit exits

Risk Rules you MUST follow (HARD GATES — these are enforced programmatically by RiskEngine, not just text):
- ALWAYS call check_risk_limits before placing any trade
- Never exceed {currency}{os.getenv('TRADE_MAX_PER_ORDER', 5000)} per order
- Stop ALL trading if daily loss exceeds {currency}{os.getenv('TRADE_MAX_DAILY_LOSS', 2000)}
- Never hold more than {os.getenv('TRADE_MAX_POSITIONS', 10)} positions simultaneously
- Apply stop-loss: exit if position drops {os.getenv('TRADE_STOP_LOSS_PCT', 2.0)}% below buy price
- Apply take-profit: exit if position gains {os.getenv('TRADE_TARGET_PCT', 4.0)}% above buy price

Mode: {TRADE_MODE.upper()} {'(REAL MONEY — be careful!)' if TRADE_MODE == 'live' else '(Paper simulation — no real money)'}

For every decision, explain your reasoning clearly: what signals you saw, why you chose to trade or not, and what you expect.
"""

    def get_tool_defs(self):
        return self._all_tool_defs

    def call_tool_fn(self, name: str, arguments: dict) -> str:
        return _call_trading_tool(name, arguments)

    def is_market_open(self) -> bool:
        now = datetime.now().time()
        if MARKET == "US":
            # NYSE: 20:00-02:30 IST (overnight). Handle midnight wrap-around.
            if MARKET_OPEN > MARKET_CLOSE:  # crosses midnight
                return now >= MARKET_OPEN or now <= MARKET_CLOSE
        return MARKET_OPEN <= now <= MARKET_CLOSE

    def run_market_scan(self, on_event=None) -> str:
        """Run one complete market analysis and decision cycle."""
        tz_label = "EST" if MARKET == "US" else "IST"
        task = (
            f"It is now {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {tz_label}.\n"
            f"Run a complete market scan and trading decision cycle:\n"
            f"1. Call get_watchlist_scan to see all signals at once\n"
            f"2. For any BUY/SELL signals with confidence > 65%, get full technical signal details\n"
            f"3. Check portfolio status and risk limits\n"
            f"4. For qualified signals: check_risk_limits then place_trade if approved\n"
            f"5. Check all current holdings for stop-loss or take-profit exit conditions\n"
            f"6. Summarize what you found, what actions you took, and why."
        )

        events = []
        for event in self.step_stream(task):
            if on_event:
                on_event(event)
            events.append(event)
            if event.type == "final_answer":
                return event.data.get("content", "")
        return "Scan completed."

    def start_autonomous_loop(self, on_event=None, stop_flag=None):
        """
        Run the agent continuously every SCAN_INTERVAL_MIN minutes during market hours.
        """
        import threading
        if stop_flag is None:
            stop_flag = threading.Event()

        logger.info(f"Trading agent started. Scanning every {SCAN_INTERVAL_MIN} min during market hours.")

        while not stop_flag.is_set():
            if self.is_market_open():
                logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Market open — running scan...")
                try:
                    result = self.run_market_scan(on_event=on_event)
                    logger.info(f"Scan complete: {result[:200]}...")
                except Exception as e:
                    logger.error(f"Scan error: {e}")
            else:
                logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed. Waiting...")

            stop_flag.wait(timeout=SCAN_INTERVAL_MIN * 60)

        logger.info("Trading agent stopped.")


# ── End of file ──────────────────────────────────────────────────────────────