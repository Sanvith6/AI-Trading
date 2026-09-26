"""
Backtest Engine — Paper strategy validation on FREE Yahoo Finance history.
No broker keys needed. Usage:
    python -m trading.backtest RELIANCE --days 365
"""

import argparse
import pandas as pd
from typing import Dict, Any

from trading.data_provider import get_ohlcv
from trading.technical_analysis import analyze


def backtest(symbol: str, days: int = 365, initial_cash: float = 100000.0) -> Dict[str, Any]:
    """Walk-forward backtest with realistic execution and NO look-ahead.

    Signals at bar i are executed at bar i+1's open (1-bar delay).
    This is the only realistic execution model for daily data.
    """
    from trading.swarm.quant_metrics import (
        compute_indian_transaction_costs, summarize_trades,
    )
    df = get_ohlcv(symbol, interval="ONE_DAY", days=days)
    if df.empty or len(df) < 60:
        return {"symbol": symbol, "error": "Insufficient history for backtest."}

    cash = initial_cash
    qty = 0
    entry_price = 0.0
    trades = []
    closed = []
    total_costs = 0.0
    from trading.execution.simulator import ExecutionSimulator
    simulator = ExecutionSimulator()

    def _leg_cost(price: float, q: int, action: str) -> float:
        try:
            return compute_indian_transaction_costs(price, q, action=action, product="CNC").total_cost
        except Exception:
            return round(price * q * 0.0015, 2)

    # Walk-forward: signal at bar i, EXECUTE at bar i+1's open (1-bar delay)
    # This eliminates look-ahead bias — trades never use the close they were
    # generated from.
    for i in range(50, len(df) - 1, 1):  # stop 1 bar early: execution at i+1
        window = df.iloc[: i + 1]
        import trading.technical_analysis as ta
        orig = ta.get_ohlcv
        ta.get_ohlcv = lambda s, interval="ONE_DAY", days=120, _w=window: _w
        try:
            res = ta.analyze(symbol)
        finally:
            ta.get_ohlcv = orig

        signal_price = float(window["close"].iloc[-1])  # signal price at bar i close
        next_bar = df.iloc[i + 1]
        exec_price = float(next_bar["open"])  # execute at bar i+1 open
        exec_date = str(next_bar["date"])
        signal_date = str(window["date"].iloc[-1])
        action = res["signal"]
        confidence = res["confidence"]

        # Estimate volatility for execution gap check
        if len(window) >= 14:
            recent = window.tail(14)
            vol_pct = float((recent["high"].max() - recent["low"].min()) /
                            recent["close"].mean() * 100)
        else:
            vol_pct = 0.0

        if action == "BUY" and qty == 0 and confidence >= 65:
            q = max(1, int(5000 // exec_price))
            fee = _leg_cost(exec_price, q, "BUY")
            cost = q * exec_price + fee
            sim = simulator.execute(symbol, "BUY", q, exec_price,
                                    bar_volume=int(window["volume"].iloc[-1]),
                                    volatility_pct=vol_pct)
            if sim.filled_qty > 0 and sim.filled_price > 0:
                actual_qty = sim.filled_qty
                actual_price = sim.filled_price
                total_qty_cost = actual_qty * actual_price + sim.commission
                if total_qty_cost <= cash:
                    cash -= total_qty_cost
                    total_costs += sim.commission
                    qty, entry_price = actual_qty, actual_price
                    trades.append({"signal_date": signal_date, "exec_date": exec_date,
                                   "action": "BUY", "signal_price": signal_price,
                                   "exec_price": actual_price, "qty": actual_qty,
                                   "fees": round(sim.commission, 2), "slippage_bps": sim.slippage_bps})
        elif action == "SELL" and qty > 0:
            sim = simulator.execute(symbol, "SELL", qty, exec_price,
                                    bar_volume=int(window["volume"].iloc[-1]),
                                    volatility_pct=vol_pct)
            if sim.filled_qty > 0 and sim.filled_price > 0:
                fee = _leg_cost(exec_price, qty, "SELL")
                proceeds = sim.filled_qty * sim.filled_price - sim.commission
                cash += proceeds
                total_costs += sim.commission
                pnl = round(proceeds - qty * entry_price, 2)
                closed.append({"signal_date": signal_date, "exec_date": exec_date,
                               "action": "SELL", "signal_price": signal_price,
                               "exec_price": sim.filled_price, "qty": qty,
                               "pnl": pnl, "fees": round(sim.commission, 2),
                               "slippage_bps": sim.slippage_bps})
                qty, entry_price = 0, 0.0

    final_row = df.iloc[-1]
    final_price = float(final_row["close"])
    equity = cash + qty * final_price
    metrics = summarize_trades(closed, initial_capital=initial_cash, final_equity=equity)

    # Walk-forward integrity check
    metrics["look_ahead_bias"] = "none (1-bar delay, exec at next-bar open)"
    metrics.update({
        "symbol": symbol, "days": days, "bars": len(df),
        "initial_cash": initial_cash,
        "open_qty": qty, "open_value": round(qty * final_price, 2),
        "total_costs": round(total_costs, 2),
        "lookahead_bars": 1,
        "legs": len(trades),
        "history": trades[-20:],
    })
    return metrics


from trading.portfolio.portfolio_backtest import run_portfolio_backtest


def main():
    ap = argparse.ArgumentParser(description="Backtest TA strategy on free Yahoo data")
    ap.add_argument("symbol", nargs="?", default="RELIANCE")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--portfolio", action="store_true", help="Run multi-asset portfolio backtest")
    args = ap.parse_args()
    import json
    if args.portfolio:
        syms = [s.strip().upper() for s in args.symbol.split(",")]
        print(json.dumps(run_portfolio_backtest(symbols=syms, days=args.days), indent=2, default=str))
    else:
        print(json.dumps(backtest(args.symbol.upper(), args.days), indent=2, default=str))


if __name__ == "__main__":
    main()
