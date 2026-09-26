"""
AI Trading Agent — Streamlit Dashboard
Full visual interface with live charts, signals, portfolio, order book, and agent controls.
Run with: streamlit run trading/trading_app.py
"""

import os
import sys
# Ensure project root is on sys.path so `import trading...` works no matter
# how Streamlit is launched (root or trading/ as CWD, file or package mode).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import threading
import time
from trading.logging_setup import setup_logging

setup_logging()
from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv(override=True)

TRADE_MODE = os.getenv("TRADE_MODE", "paper")
MARKET = os.getenv("MARKET", "IN").strip().upper()  # US | IN
WATCHLIST  = [s.strip() for s in os.getenv("TRADE_WATCHLIST", "SPY,AAPL,NVDA,MSFT,TSLA").split(",")]
CURRENCY   = "$" if MARKET == "US" else "\u20b9"
EXCHANGE   = "NYSE/NASDAQ" if MARKET == "US" else "NSE/BSE India"

st.set_page_config(
    page_title="AI Trading Agent — NYSE/NASDAQ" if MARKET == "US" else "AI Trading Agent — NSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
.trade-header {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #10b981 0%, #3b82f6 60%, #8b5cf6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-card {
    background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2);
    border-radius: 12px; padding: 1rem; margin-bottom: 0.5rem;
}
.signal-buy  { color: #10b981; font-weight: 700; }
.signal-sell { color: #ef4444; font-weight: 700; }
.signal-hold { color: #f59e0b; font-weight: 700; }
.live-badge  {
    background: #10b981; color: white;
    padding: 2px 10px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 700;
}
.paper-badge {
    background: #f59e0b; color: white;
    padding: 2px 10px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────

if "agent_running"  not in st.session_state: st.session_state.agent_running  = False
if "agent_log"      not in st.session_state: st.session_state.agent_log      = []
if "stop_flag"      not in st.session_state: st.session_state.stop_flag      = None
if "agent_thread"   not in st.session_state: st.session_state.agent_thread   = None
if "scan_results"   not in st.session_state: st.session_state.scan_results   = []
if "manual_trade_result" not in st.session_state: st.session_state.manual_trade_result = None

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    mode_badge = f'<span class="{"live-badge" if TRADE_MODE == "live" else "paper-badge"}">{"🔴 LIVE" if TRADE_MODE == "live" else "🟡 PAPER"}</span>'
    st.markdown(f"### ⚙️ Agent Control  {mode_badge}", unsafe_allow_html=True)

    if TRADE_MODE == "live":
        st.error("⚠️ **LIVE MODE** — Real money orders will be placed!", icon="🚨")

    # Agent start/stop
    if not st.session_state.agent_running:
        if st.button("▶️ Start Autonomous Agent", use_container_width=True, type="primary"):
            try:
                from trading.trading_agent import TradingAgent

                def on_event(event):
                    entry = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": event.type,
                        "data": event.data,
                    }
                    st.session_state.agent_log.insert(0, entry)
                    if len(st.session_state.agent_log) > 100:
                        st.session_state.agent_log = st.session_state.agent_log[:100]

                stop_flag = threading.Event()
                agent = TradingAgent()
                thread = threading.Thread(
                    target=agent.start_autonomous_loop,
                    kwargs={"on_event": on_event, "stop_flag": stop_flag},
                    daemon=True,
                )
                thread.start()
                st.session_state.agent_running = True
                st.session_state.stop_flag = stop_flag
                st.session_state.agent_thread = thread
                st.success("Agent started!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start agent: {e}")
    else:
        st.success("🟢 Agent is running...")
        if st.button("⏹ Stop Agent", use_container_width=True):
            if st.session_state.stop_flag:
                st.session_state.stop_flag.set()
            st.session_state.agent_running = False
            st.warning("Agent stopped.")
            st.rerun()

    st.markdown("---")
    st.markdown(f"### 📊 Watchlist ({len(WATCHLIST)} stocks)")
    with st.expander("View All Symbols", expanded=False):
        st.write(", ".join(WATCHLIST))

    st.markdown("---")
    st.markdown("### ⚠️ Risk Limits")
    st.markdown(f"""
    - Max/order: **{CURRENCY}{os.getenv('TRADE_MAX_PER_ORDER', 5000)}**
    - Daily loss limit: **{CURRENCY}{os.getenv('TRADE_MAX_DAILY_LOSS', 2000)}**
    - Max positions: **{os.getenv('TRADE_MAX_POSITIONS', 10)}**
    - Stop-loss: **{os.getenv('TRADE_STOP_LOSS_PCT', 2.0)}%**
    - Take-profit: **{os.getenv('TRADE_TARGET_PCT', 4.0)}%**
    """)

    st.markdown("---")
    # Manual single scan trigger
    if st.button("🔍 Run Single Scan Now", use_container_width=True):
        try:
            from trading.trading_agent import TradingAgent
            agent = TradingAgent()
            events = []
            with st.spinner("Running market scan..."):
                for ev in agent.step_stream(
                    "Do a quick watchlist scan with get_watchlist_scan, then summarize the signals."
                ):
                    events.append(ev)
            for ev in events:
                if ev.type == "final_answer":
                    st.session_state.agent_log.insert(0, {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": "manual_scan_result",
                        "data": {"content": ev.data.get("content", "")},
                    })
            st.success("Scan done! See Activity Log.")
            st.rerun()
        except Exception as e:
            st.error(f"Scan error: {e}")

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown(f'<div class="trade-header">📈 AI Trading Platform — {EXCHANGE}</div>', unsafe_allow_html=True)
st.caption(f"Powered by **{os.getenv('OPENROUTER_MODEL', 'inclusionai/ling-3.0-flash-vl:free')}** via OpenRouter • 📝 PAPER mode (Yahoo Finance FREE data, simulated fills) • {datetime.now().strftime('%d %b %Y')}")

view_mode = st.radio(
    "Select Operating Mode:",
    [
        "⚔️ Darwinian Swarm Arena (Survival of the Fittest)",
        "🔬 Institutional Research Dashboard (13+ Views)",
        "🏛️ Standard Trading Agent",
    ],
    horizontal=True,
    index=0
)

def render_standard_dashboard():
    # ── Live Market Signal Scanner ─────────────────────────────────────────────────
    st.markdown(f"### 📡 Live Technical Signals ({len(WATCHLIST)} Stocks)")

    if st.button("🔄 Refresh Signals", key="refresh_signals"):
        with st.spinner(f"Analyzing all {len(WATCHLIST)} stocks concurrently..."):
            try:
                from trading.technical_analysis import batch_analyze
                results = batch_analyze(WATCHLIST)
                rows = []
                for result in results:
                    sym = result.get("symbol", "")
                    ind = result.get("indicators", {})
                    rows.append({
                        "Symbol": sym,
                        "Signal": result.get("signal", "HOLD"),
                        "Confidence": f"{result.get('confidence', 0)}%",
                        "Last Price": f"{CURRENCY}{result.get('last_price', 'N/A')}",
                        "RSI": ind.get("rsi", "N/A"),
                        "MACD Hist": ind.get("macd_histogram", "N/A"),
                        "EMA9 > EMA21": "✅" if (ind.get("ema9", 0) or 0) > (ind.get("ema21", 0) or 0) else "❌",
                        "Vol Spike": "🔥" if ind.get("volume_spike") else "—",
                    })
                st.session_state.scan_results = rows
            except Exception as e:
                st.error(f"Could not fetch signals: {e}")

    if st.session_state.scan_results:
        df = pd.DataFrame(st.session_state.scan_results)

        def color_signal(val):
            if val == "BUY":  return "color: #10b981; font-weight: bold"
            if val == "SELL": return "color: #ef4444; font-weight: bold"
            return "color: #f59e0b"

        st.dataframe(
            df.style.applymap(color_signal, subset=["Signal"]),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Click **Refresh Signals** to scan all watchlist stocks.")

    # ── Portfolio / Risk Status ────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💼 Portfolio & Risk Status")
        if st.button("🔄 Refresh Portfolio", key="refresh_portfolio"):
            try:
                from trading.risk_manager import RiskManager
                rm = RiskManager()
                st.json(rm.get_status())

                from trading.order_manager import get_paper_portfolio
                portfolio = get_paper_portfolio()
                st.metric("Paper Cash", f"{CURRENCY}{portfolio.get('cash', 0):,.2f}")
                holdings = portfolio.get("holdings", [])
                if holdings:
                    st.markdown(f"**Paper Holdings (Yahoo LTP):**")
                    hdf = pd.DataFrame(holdings)
                    cols = [c for c in ["tradingsymbol", "quantity", "average_price", "last_price", "pnl"] if c in hdf.columns]
                    hdf = hdf[cols]
                    hdf.columns = ["Symbol", "Qty", "Avg Price", "LTP", "P&L"][:len(cols)]
                    st.dataframe(hdf, use_container_width=True, hide_index=True)
                else:
                    st.info("No paper positions yet. Place a manual trade below.")
            except Exception as e:
                st.error(f"Portfolio fetch error: {e}")

    with col2:
        st.markdown("### 📋 Trade Log (today)")
        if st.button("🔄 Refresh Orders", key="refresh_orders"):
            try:
                from trading.order_manager import get_trade_log
                st.code(get_trade_log()[:4000], language="text")
            except Exception as e:
                st.error(f"Order fetch error: {e}")

    # ── Manual Trade ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🖐️ Manual Trade")

    with st.form("manual_trade_form"):
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1: symbol_input = st.selectbox("Symbol", WATCHLIST)
        with tc2: action_input = st.selectbox("Action", ["BUY", "SELL"])
        with tc3: price_input  = st.number_input(f"Price ({CURRENCY}, 0=auto LTP)", min_value=0.0, value=0.0, step=0.5)
        with tc4: qty_input    = st.number_input("Quantity (0=auto)", min_value=0, value=0, step=1)

        submitted = st.form_submit_button("Place Manual Trade", type="primary")
        if submitted:
            try:
                from trading.risk_manager import RiskManager
                from trading.order_manager import place_buy, place_sell
                from trading.data_provider import get_quote

                rm = RiskManager()
                live = get_quote(symbol_input).get("last_price", 0) or 0
                est_price = price_input if price_input > 0 else (live or 0)
                if not est_price:
                    st.error(f"Could not fetch LTP for {symbol_input} (market closed?). Enter a manual price.")
                else:
                    est_qty = qty_input if qty_input > 0 else max(1, int(float(os.getenv("TRADE_MAX_PER_ORDER", 5000)) // est_price))
                    order_val = est_price * est_qty

                allowed, reason = rm.is_trade_allowed(symbol_input, order_val, action_input)
                if not allowed:
                    st.error(f"❌ Trade blocked by risk manager: {reason}")
                else:
                    if action_input == "BUY":
                        result = place_buy(
                            symbol_input,
                            price=price_input if price_input > 0 else None,
                            quantity=qty_input if qty_input > 0 else None,
                        )
                    else:
                        result = place_sell(
                            symbol_input,
                            quantity=qty_input if qty_input > 0 else 1,
                            price=price_input if price_input > 0 else None,
                        )
                    st.session_state.manual_trade_result = result
            except Exception as e:
                st.error(f"Trade error: {e}")

    if st.session_state.manual_trade_result:
        r = st.session_state.manual_trade_result
        if r.get("status") in ("placed", "paper_executed"):
            st.success(f"✅ Order {r['status'].upper()}: {r.get('symbol')} × {r.get('quantity')} @ {CURRENCY}{r.get('price')} | Order ID: {r.get('order_id')}")
        else:
            st.error(f"❌ {r.get('message', 'Unknown error')}")

    # ── Agent Activity Log ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Agent Activity Log")

    if not st.session_state.agent_log:
        st.info("No agent activity yet. Start the agent or run a manual scan.")
    else:
        for entry in st.session_state.agent_log[:30]:
            t    = entry.get("time", "")
            etype = entry.get("type", "")
            data = entry.get("data", {})

            if etype == "tool_start":
                with st.expander(f"[{t}] ⚡ Tool: `{data.get('tool_name')}` called", expanded=False):
                    st.json(data.get("arguments", {}))
            elif etype == "tool_end":
                with st.expander(f"[{t}] ✅ Tool: `{data.get('tool_name')}` completed", expanded=False):
                    st.code(str(data.get("result", ""))[:1000], language="text")
            elif etype in ("final_answer", "manual_scan_result"):
                with st.expander(f"[{t}] 🤖 Agent Decision", expanded=True):
                    st.markdown(data.get("content", ""))
            elif etype == "thought":
                st.caption(f"[{t}] 💭 {data.get('content', '')[:200]}")
            elif etype == "error":
                st.error(f"[{t}] ❌ {data.get('message', '')}")

    # ── Trade History Log ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📜 Trade History Log")

    trade_log_path = Path(__file__).parent / "trade_log.csv"
    if trade_log_path.exists():
        try:
            tdf = pd.read_csv(trade_log_path)
            st.dataframe(tdf[::-1], use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not read trade log: {e}")
    else:
        st.info("No trades have been placed yet. The trade log will appear here once orders are placed.")

# ── Route View Mode ────────────────────────────────────────────────────────────
if view_mode == "⚔️ Darwinian Swarm Arena (Survival of the Fittest)":
    from trading.swarm.swarm_dashboard import render_swarm_dashboard
    render_swarm_dashboard()
elif view_mode == "🔬 Institutional Research Dashboard (13+ Views)":
    from trading.research_dashboard import render_institutional_research_dashboard
    render_institutional_research_dashboard()
else:
    render_standard_dashboard()

# Auto-refresh when agent is running
if st.session_state.agent_running:
    time.sleep(15)
    st.rerun()
