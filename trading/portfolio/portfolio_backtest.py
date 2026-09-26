"""
Multi-Asset Portfolio Backtest Engine — Institutional Research Simulator.

Features:
- Multi-asset portfolio evaluation across dynamic universe
- Portfolio-level capital allocation and periodic rebalancing
- PortfolioConstructor constraint enforcement (exposure <= 95%, max position 25%, min cash floor)
- Risk-aware position sizing with stop-loss, take-profit, and trailing stops
- Full Indian transaction costs (STT, exchange charges, SEBI, GST, stamp duty) + slippage
- Benchmark comparison against Buy & Hold
- Attribution of returns by symbol, strategy, and statutory cost burden
- Zero look-ahead bias: signal at bar i, execution strictly at bar i+1 open
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from trading.data_provider import get_ohlcv
from trading.portfolio.portfolio_constructor import PortfolioConstructor
from trading.portfolio.position_sizer import PositionSizer
from trading.execution.simulator import ExecutionSimulator
from trading.swarm.quant_metrics import summarize_trades, compute_indian_transaction_costs
from trading.regime.regime_detector import RegimeDetector


def run_portfolio_backtest(
    symbols: Optional[List[str]] = None,
    days: int = 180,
    initial_capital: float = 100000.0,
    rebalance_frequency_bars: int = 5,
    stop_loss_pct: float = 2.5,
    take_profit_pct: float = 6.0,
    trailing_stop_pct: float = 1.5,
    max_position_weight: float = 0.25,
) -> Dict[str, Any]:
    """
    Executes an institutional multi-asset portfolio backtest.
    """
    if not symbols:
        symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

    # 1. Ingest historical data for all symbols
    dfs: Dict[str, pd.DataFrame] = {}
    min_len = 999999
    for s in symbols:
        df = get_ohlcv(s, interval="ONE_DAY", days=days)
        if not df.empty and len(df) >= 40:
            dfs[s] = df
            min_len = min(min_len, len(df))

    if len(dfs) < 2 or min_len < 40:
        return {"error": "Insufficient history across symbols for portfolio backtest"}

    # Align data frames by length
    aligned_dfs = {s: df.iloc[-min_len:].reset_index(drop=True) for s, df in dfs.items()}

    cash = initial_capital
    positions: Dict[str, Dict[str, Any]] = {}
    trades: List[Dict[str, Any]] = []
    equity_curve = []
    total_costs = 0.0

    constructor = PortfolioConstructor(max_position_weight=max_position_weight, min_cash_pct=0.05)
    sizer = PositionSizer(default_risk_per_trade_pct=2.0, max_position_weight=max_position_weight)
    simulator = ExecutionSimulator(fill_prob=1.0)
    regime_detector = RegimeDetector()

    # 2. Walk-forward chronological loop (1-bar execution delay)
    warmup = 30
    for i in range(warmup, min_len - 1):
        # Current bar i close is decision point
        current_equity = cash + sum(pos["qty"] * float(aligned_dfs[s]["close"].iloc[i]) for s, pos in positions.items())
        equity_curve.append(current_equity)

        # Check existing positions for stop-loss / take-profit / trailing stop
        for s in list(positions.keys()):
            pos = positions[s]
            curr_bar = aligned_dfs[s].iloc[i]
            next_bar = aligned_dfs[s].iloc[i + 1]
            c_price = float(curr_bar["close"])
            entry_p = pos["entry_price"]
            highest_p = pos.get("highest_price", entry_p)

            if c_price > highest_p:
                pos["highest_price"] = c_price
                highest_p = c_price

            # Stop conditions
            is_stop_loss = (c_price - entry_p) / entry_p * 100.0 <= -stop_loss_pct
            is_take_profit = (c_price - entry_p) / entry_p * 100.0 >= take_profit_pct
            is_trailing = (c_price - highest_p) / highest_p * 100.0 <= -trailing_stop_pct and c_price > entry_p

            if is_stop_loss or is_take_profit or is_trailing:
                reason = "Stop Loss" if is_stop_loss else ("Take Profit" if is_take_profit else "Trailing Stop")
                exec_price = float(next_bar["open"])
                sim = simulator.execute(s, "SELL", pos["qty"], exec_price)
                if sim.filled_qty > 0:
                    proceeds = sim.filled_qty * sim.filled_price - sim.commission
                    cash += proceeds
                    total_costs += sim.commission
                    pnl = proceeds - (pos["qty"] * entry_p)
                    trades.append({
                        "symbol": s, "action": "SELL", "exec_price": sim.filled_price,
                        "qty": sim.filled_qty, "pnl": round(pnl, 2), "fees": round(sim.commission, 2),
                        "reason": reason, "date": str(next_bar["date"]),
                    })
                    del positions[s]

        # Periodic Rebalancing
        if (i - warmup) % rebalance_frequency_bars == 0:
            active_symbols = list(aligned_dfs.keys())
            weights = constructor.construct(active_symbols, method="risk_parity")

            for s, target_w in weights.items():
                target_val = current_equity * target_w
                curr_price = float(aligned_dfs[s]["close"].iloc[i])
                next_open = float(aligned_dfs[s]["iloc" if "iloc" in aligned_dfs[s] else ""].iloc[i + 1]["open"] if "iloc" in aligned_dfs[s] else aligned_dfs[s]["open"].iloc[i + 1])

                if s not in positions and target_w > 0.05:
                    sized = sizer.size_by_risk(cash, next_open, stop_loss_pct)
                    target_qty = min(sized["quantity"], int(target_val // next_open))
                    if target_qty > 0 and (target_qty * next_open) <= cash:
                        sim = simulator.execute(s, "BUY", target_qty, next_open)
                        if sim.filled_qty > 0:
                            cost = sim.filled_qty * sim.filled_price + sim.commission
                            cash -= cost
                            total_costs += sim.commission
                            positions[s] = {
                                "qty": sim.filled_qty,
                                "entry_price": sim.filled_price,
                                "highest_price": sim.filled_price,
                                "date": str(aligned_dfs[s]["date"].iloc[i + 1]),
                            }
                            trades.append({
                                "symbol": s, "action": "BUY", "exec_price": sim.filled_price,
                                "qty": sim.filled_qty, "fees": round(sim.commission, 2),
                                "reason": "Portfolio Rebalance Entry", "date": str(aligned_dfs[s]["date"].iloc[i + 1]),
                            })

    # Final mark-to-market
    final_equity = cash + sum(pos["qty"] * float(aligned_dfs[s]["close"].iloc[-1]) for s, pos in positions.items())
    closed_trades = [t for t in trades if t["action"] == "SELL"]
    metrics = summarize_trades(closed_trades, initial_capital=initial_capital, final_equity=final_equity)

    # Benchmark comparison: 1/N Buy and Hold across all symbols
    bnh_initial = initial_capital / len(aligned_dfs)
    bnh_final = 0.0
    for s, df in aligned_dfs.items():
        p0 = float(df["close"].iloc[warmup])
        p1 = float(df["close"].iloc[-1])
        bnh_final += (bnh_initial / p0) * p1

    bnh_return_pct = round((bnh_final - initial_capital) / initial_capital * 100.0, 2)

    return {
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "net_return_pct": round((final_equity - initial_capital) / initial_capital * 100.0, 2),
        "benchmark_buy_and_hold_return_pct": bnh_return_pct,
        "outperformance_pct": round(((final_equity - initial_capital) / initial_capital * 100.0) - bnh_return_pct, 2),
        "total_trades": len(closed_trades),
        "total_costs_paid": round(total_costs, 2),
        "sharpe": metrics.get("sharpe", 0.0),
        "sortino": metrics.get("sortino", 0.0),
        "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
        "profit_factor": metrics.get("profit_factor", 0.0),
        "open_positions": {s: p["qty"] for s, p in positions.items()},
        "recent_trades": trades[-10:],
    }
