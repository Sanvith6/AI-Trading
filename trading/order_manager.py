"""
Order Manager — Paper-trading by default (FREE Yahoo Finance LTP, simulated fills).
Optional live mode: BROKER=angel routes orders to Angel One SmartAPI.
All orders (paper + live) are logged to trade_log.csv.
 paper"""

import os
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

load_dotenv(override=True)

BROKER = os.getenv("BROKER", "paper").strip().lower()
MAX_PER_ORDER  = float(os.getenv("TRADE_MAX_PER_ORDER", 5000))
RISK_PER_TRADE_PCT = float(os.getenv("TRADE_RISK_PER_TRADE_PCT", 5.0))  # % of paper cash per trade
TRADE_PRODUCT  = os.getenv("TRADE_PRODUCT", "DELIVERY")   # DELIVERY or INTRADAY
TRADE_MODE     = os.getenv("TRADE_MODE", "paper")
TRADE_LOG      = Path(__file__).parent / "trade_log.csv"
PAPER_STATE    = Path(__file__).parent / "paper_portfolio.json"


def _round_trip_cost(price: float, qty: int) -> float:
    """Conservative NSE delivery cost estimate (buy + sell legs + slippage)."""
    try:
        from trading.swarm.quant_metrics import compute_indian_transaction_costs
        buy = compute_indian_transaction_costs(price, qty, action="BUY", product="CNC")
        sell = compute_indian_transaction_costs(price, qty, action="SELL", product="CNC")
        return round(buy.total_cost + sell.total_cost, 2)
    except Exception:
        return round(price * qty * 0.003, 2)  # ~0.3% fallback


def _get_client():
    from trading.angel_auth import get_smartapi_client
    obj, _, _ = get_smartapi_client()
    return obj


def _ltp(symbol: str) -> float:
    """LTP from free provider by default; Angel only when BROKER=angel."""
    if BROKER == "angel":
        try:
            from trading.market_data import get_ltp as _angel_ltp
            return (_angel_ltp([symbol]) or {}).get(symbol, 0) or 0
        except Exception:
            pass
    from trading.data_provider import get_ltp as _free_ltp
    return (_free_ltp([symbol]) or {}).get(symbol, 0) or 0


def _load_paper() -> Dict[str, Any]:
    if PAPER_STATE.exists():
        try:
            return json.loads(PAPER_STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"cash": 100000.0, "positions": {}}


def _save_paper(state: Dict[str, Any]) -> None:
    PAPER_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_paper_portfolio() -> Dict[str, Any]:
    """Paper holdings valued at live Yahoo LTP (no broker login)."""
    from trading.data_provider import get_quote
    state = _load_paper()
    positions: List[Dict[str, Any]] = []
    for sym, pos in state.get("positions", {}).items():
        q = get_quote(sym)
        ltp = q.get("last_price", 0) or pos.get("avg_price", 0)
        qty = pos.get("quantity", 0)
        avg = pos.get("avg_price", 0)
        pnl = round((ltp - avg) * qty, 2)
        positions.append({"tradingsymbol": sym, "quantity": qty,
                          "average_price": avg, "last_price": ltp, "pnl": pnl})
    return {"holdings": positions, "positions": positions,
            "cash": round(state.get("cash", 0.0), 2),
            "available_margin": round(state.get("cash", 0.0), 2)}


def _log_trade(record: Dict[str, Any]) -> None:
    fieldnames = ["timestamp", "mode", "action", "symbol", "quantity",
                  "price", "order_value", "order_id", "status", "notes"]
    write_header = not TRADE_LOG.exists()
    with open(TRADE_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {k: record.get(k, "") for k in fieldnames}
        row.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        row.setdefault("mode", TRADE_MODE)
        writer.writerow(row)


def _calc_qty(price: float) -> int:
    if price <= 0:
        return 1
    return max(1, math.floor(MAX_PER_ORDER / price))


def place_buy(symbol: str, price: Optional[float] = None,
              quantity: Optional[int] = None) -> Dict[str, Any]:
    """Paper BUY (simulated fill at LTP) — routes to Angel only when BROKER=angel + TRADE_MODE=live."""
    symbol = symbol.strip().upper()
    try:
        if price is None:
            price = _ltp(symbol)
        if not price:
            return {"status": "error", "message": f"Could not get LTP for {symbol} (market closed or bad symbol?)"}

        qty = quantity or _calc_qty(price)

        if TRADE_MODE != "live" or BROKER != "angel":
            order_id = f"PAPER-BUY-{symbol}-{datetime.now().strftime('%H%M%S')}"
            # Update paper ledger
            state = _load_paper()
            # Per-trade risk cap: min(₹ cap, % of paper cash) — institutional 1%-rule style guard
            cash_cap = round(state.get("cash", 0) * RISK_PER_TRADE_PCT / 100, 2)
            if price * qty > cash_cap:
                qty = max(1, math.floor(cash_cap / price))
            cost = price * qty
            fees = _round_trip_cost(price, qty)
            if state["cash"] < cost + fees:
                return {"status": "error", "message": f"Insufficient paper cash (need Rs.{cost + fees:,.0f} incl. ~Rs.{fees:,.0f} costs, have Rs.{state['cash']:,.0f})"}
            state["cash"] = round(state["cash"] - cost - fees, 2)
            pos = state["positions"].get(symbol, {"quantity": 0, "avg_price": 0.0})
            new_qty = pos["quantity"] + qty
            pos["avg_price"] = round((pos["avg_price"] * pos["quantity"] + cost) / new_qty, 2)
            pos["quantity"] = new_qty
            state["positions"][symbol] = pos
            _save_paper(state)
            _log_trade({"action": "BUY", "symbol": symbol, "quantity": qty,
                        "price": price, "order_value": price * qty,
                        "order_id": order_id, "status": "PAPER_EXECUTED",
                        "notes": f"Paper fill (Yahoo LTP) incl. ~Rs.{fees:.2f} NSE costs"})
            return {"status": "paper_executed", "order_id": order_id,
                    "symbol": symbol, "quantity": qty, "price": price,
                    "order_value": price * qty}

        from trading.market_data import _get_token, EXCHANGE_NSE
        token = _get_token(symbol)
        if not token:
            return {"status": "error", "message": f"Unknown symbol: {symbol}"}
        obj = _get_client()
        order_params = {
            "variety":         "NORMAL",
            "tradingsymbol":   symbol,
            "symboltoken":     token,
            "transactiontype": "BUY",
            "exchange":        EXCHANGE_NSE,
            "ordertype":       "LIMIT",
            "producttype":     TRADE_PRODUCT,
            "duration":        "DAY",
            "price":           str(round(price, 2)),
            "squareoff":       "0",
            "stoploss":        "0",
            "quantity":        str(qty),
        }
        result = obj.placeOrder(order_params)
        order_id = result.get("data", {}).get("orderid", "UNKNOWN")

        _log_trade({"action": "BUY", "symbol": symbol, "quantity": qty,
                    "price": price, "order_value": price * qty,
                    "order_id": order_id, "status": "PLACED"})

        return {"status": "placed", "order_id": order_id,
                "symbol": symbol, "quantity": qty, "price": price,
                "order_value": price * qty}

    except Exception as e:
        _log_trade({"action": "BUY", "symbol": symbol, "price": price,
                    "quantity": quantity, "status": "ERROR", "notes": str(e)})
        return {"status": "error", "message": str(e)}


def place_sell(symbol: str, quantity: int,
               price: Optional[float] = None) -> Dict[str, Any]:
    """Paper SELL (simulated fill) — Angel only when BROKER=angel + TRADE_MODE=live."""
    symbol = symbol.strip().upper()
    try:
        if price is None:
            price = _ltp(symbol)

        if TRADE_MODE != "live" or BROKER != "angel":
            order_id = f"PAPER-SELL-{symbol}-{datetime.now().strftime('%H%M%S')}"
            state = _load_paper()
            pos = state["positions"].get(symbol)
            if pos and quantity >= pos["quantity"]:
                quantity = pos["quantity"]  # sell what we hold
                del state["positions"][symbol]
            elif pos:
                pos["quantity"] -= quantity
            from trading.swarm.quant_metrics import compute_indian_transaction_costs as _costs
            try:
                sell_fees = _costs(price, quantity, action="SELL", product="CNC").total_cost
            except Exception:
                sell_fees = round(price * quantity * 0.0015, 2)
            state["cash"] = round(state["cash"] + price * quantity - sell_fees, 2)
            _save_paper(state)
            _log_trade({"action": "SELL", "symbol": symbol, "quantity": quantity,
                        "price": price, "order_value": price * quantity,
                        "order_id": order_id, "status": "PAPER_EXECUTED",
                        "notes": f"Paper fill (Yahoo LTP) net of ~Rs.{sell_fees:.2f} NSE costs"})
            return {"status": "paper_executed", "order_id": order_id,
                    "symbol": symbol, "quantity": quantity, "price": price}

        from trading.market_data import _get_token, EXCHANGE_NSE
        token = _get_token(symbol)
        if not token:
            return {"status": "error", "message": f"Unknown symbol: {symbol}"}
        obj = _get_client()
        order_params = {
            "variety":         "NORMAL",
            "tradingsymbol":   symbol,
            "symboltoken":     token,
            "transactiontype": "SELL",
            "exchange":        EXCHANGE_NSE,
            "ordertype":       "LIMIT",
            "producttype":     TRADE_PRODUCT,
            "duration":        "DAY",
            "price":           str(round(price, 2)),
            "squareoff":       "0",
            "stoploss":        "0",
            "quantity":        str(quantity),
        }
        result = obj.placeOrder(order_params)
        order_id = result.get("data", {}).get("orderid", "UNKNOWN")

        _log_trade({"action": "SELL", "symbol": symbol, "quantity": quantity,
                    "price": price, "order_value": price * quantity,
                    "order_id": order_id, "status": "PLACED"})

        return {"status": "placed", "order_id": order_id,
                "symbol": symbol, "quantity": quantity, "price": price}

    except Exception as e:
        _log_trade({"action": "SELL", "symbol": symbol, "quantity": quantity,
                    "price": price, "status": "ERROR", "notes": str(e)})
        return {"status": "error", "message": str(e)}


def cancel_order(order_id: str, variety: str = "NORMAL") -> Dict[str, Any]:
    """Cancel a pending order by its order ID."""
    try:
        if TRADE_MODE == "paper":
            return {"status": "paper_cancelled", "order_id": order_id}
        obj = _get_client()
        obj.cancelOrder(order_id, variety)
        return {"status": "cancelled", "order_id": order_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def exit_position(symbol: str) -> Dict[str, Any]:
    """Exit the full open paper position in a symbol at ~LTP."""
    try:
        symbol = symbol.strip().upper()
        if TRADE_MODE != "live" or BROKER != "angel":
            state = _load_paper()
            pos = state["positions"].get(symbol)
            if not pos:
                return {"status": "no_position", "symbol": symbol,
                        "message": f"No paper position found for {symbol}"}
            price = round(_ltp(symbol) * 0.995, 2) or pos["avg_price"]
            return place_sell(symbol, int(pos["quantity"]), price=price)
        from trading.market_data import get_portfolio
        portfolio = get_portfolio()
        holdings  = portfolio.get("holdings", [])
        holding   = next((h for h in holdings if h.get("tradingsymbol") == symbol), None)

        if not holding:
            return {"status": "no_position", "symbol": symbol,
                    "message": f"No holding found for {symbol}"}

        qty   = int(holding.get("quantity", 0))
        price = round(_ltp(symbol) * 0.995, 2)  # 0.5% below LTP for quick fill

        return place_sell(symbol, qty, price=price)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_trade_log() -> str:
    """Return the full trade log CSV contents as a string."""
    if not TRADE_LOG.exists():
        return "No trades logged yet."
    with open(TRADE_LOG) as f:
        return f.read()
