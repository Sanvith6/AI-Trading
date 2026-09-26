"""
Quantitative Performance Metrics & Indian Market Microstructure Model
Institutional-grade risk-adjusted return calculations (Sharpe, Sortino, Calmar),
Fractional Kelly Criterion sizing, and NSE/BSE transaction fee & slippage modeling.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class TransactionCost:
    brokerage: float
    stt: float                  # Securities Transaction Tax
    exchange_charges: float     # NSE / BSE transaction charges
    sebi_charges: float
    stamp_duty: float
    gst: float                  # 18% on (brokerage + exchange + SEBI)
    slippage: float
    total_cost: float


def compute_indian_transaction_costs(
    price: float,
    quantity: int,
    action: str = "BUY",
    product: str = "CNC",        # CNC (Delivery) or MIS (Intraday)
    broker_fixed_fee: float = 20.0,
    slippage_bps: float = 5.0    # 5 basis points (0.05%) default execution slippage
) -> TransactionCost:
    """
    Computes exact transaction costs for Indian equities (NSE/BSE).
    
    Fee Structure (Standard Indian Discount Broker, e.g. Zerodha / Angel One):
    - Brokerage: Zero for equity delivery (CNC) or flat ₹20/order for intraday (MIS).
    - STT (Securities Transaction Tax):
      - CNC Delivery: 0.1% on both Buy and Sell turnover.
      - MIS Intraday: 0.025% on Sell turnover only.
    - Exchange Turnover Charges: 0.00345% (NSE).
    - SEBI Charges: ₹10 per crore (0.0001%).
    - Stamp Duty: 0.015% on Buy turnover only.
    - GST: 18% on (Brokerage + Exchange Charges + SEBI Charges).
    - Slippage: Modeled in basis points (1 bp = 0.01%).
    """
    turnover = price * quantity
    action = action.upper()
    product = product.upper()

    # 1. Brokerage
    if product == "CNC":
        brokerage = 0.0  # Free equity delivery
    else:
        brokerage = min(broker_fixed_fee, turnover * 0.0003)

    # 2. STT
    if product == "CNC":
        stt = turnover * 0.001  # 0.1% on both buy and sell
    else:
        stt = (turnover * 0.00025) if action == "SELL" else 0.0

    # 3. Exchange Charges (NSE equity: 0.00345%)
    exchange_charges = turnover * 0.0000345

    # 4. SEBI Turnover Charges (₹10 / crore = 0.0001%)
    sebi_charges = turnover * 0.000001

    # 5. Stamp Duty (0.015% on buy only)
    stamp_duty = (turnover * 0.00015) if action == "BUY" else 0.0

    # 6. GST (18% on Brokerage + Exchange Charges + SEBI Charges)
    gst = (brokerage + exchange_charges + sebi_charges) * 0.18

    # 7. Slippage Model
    slippage = turnover * (slippage_bps / 10000.0)

    total_cost = brokerage + stt + exchange_charges + sebi_charges + stamp_duty + gst + slippage

    return TransactionCost(
        brokerage=round(brokerage, 2),
        stt=round(stt, 2),
        exchange_charges=round(exchange_charges, 2),
        sebi_charges=round(sebi_charges, 2),
        stamp_duty=round(stamp_duty, 2),
        gst=round(gst, 2),
        slippage=round(slippage, 2),
        total_cost=round(total_cost, 2),
    )


# ── Risk-Adjusted Quant Metrics ───────────────────────────────────────────────

def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.065,  # 6.5% Indian 10-year G-Sec yield
    periods_per_year: int = 252
) -> float:
    """
    Calculates annualized Sharpe Ratio:
    Sharpe = (E[R] - Rf) / sigma
    """
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0

    if std_dev == 0.0:
        return 0.0

    # Annualize
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_return = mean_return - rf_per_period
    annualized_sharpe = (excess_return / std_dev) * math.sqrt(periods_per_year)
    return round(annualized_sharpe, 2)


def calculate_sortino_ratio(
    returns: List[float],
    target_return: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculates annualized Sortino Ratio:
    Penalizes ONLY downside volatility below target return.
    Sortino = (E[R] - Target) / Downside_Deviation
    """
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    downside_diffs = [min(0.0, r - target_return) ** 2 for r in returns]
    downside_variance = sum(downside_diffs) / len(returns)
    downside_dev = math.sqrt(downside_variance)

    if downside_dev == 0.0:
        return 0.0

    annualized_sortino = ((mean_return - target_return) / downside_dev) * math.sqrt(periods_per_year)
    return round(annualized_sortino, 2)


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculates Maximum Peak-to-Trough Drawdown (%) from an equity curve series.
    """
    if equity_curve is None or len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak * 100.0
            if dd > max_dd:
                max_dd = dd

    return round(max_dd, 2)


def calculate_calmar_ratio(annualized_return: float, max_drawdown_pct: float) -> float:
    """
    Calmar Ratio = Annualized Return / Maximum Drawdown.
    Measures return per unit of worst-case drawdown.
    """
    if max_drawdown_pct <= 0:
        return 0.0
    return round(annualized_return / max_drawdown_pct, 2)


def calculate_profit_factor(trades: List[Dict[str, Any]]) -> float:
    """
    Profit Factor = Gross Profits / Gross Losses.
    Values > 1.5 indicate robust statistical edge; > 2.0 is institutional grade.
    """
    gross_profits = sum(t["pnl"] for t in trades if t.get("pnl", 0) > 0)
    gross_losses = abs(sum(t["pnl"] for t in trades if t.get("pnl", 0) < 0))

    if gross_losses == 0:
        return round(gross_profits, 2) if gross_profits > 0 else 1.0
    return round(gross_profits / gross_losses, 2)


def calculate_kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.5   # Half-Kelly (industry standard to prevent over-betting)
) -> float:
    """
    Calculates Fractional Kelly Criterion for optimal position sizing:
    f* = (p * b - q) / b
    where:
    p = win rate (0.0 to 1.0)
    q = loss rate (1 - p)
    b = payoff ratio (avg_win / avg_loss)

    Returns optimal capital allocation fraction (0.0 to 1.0).
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1.0:
        return 0.1  # Fallback to 10% safe sizing

    b = avg_win / avg_loss
    p = win_rate
    q = 1.0 - p

    kelly = (p * b - q) / b
    fractional_kelly = max(0.02, min(0.35, kelly * fraction))  # Bound between 2% and 35%
    return round(fractional_kelly, 4)


# ── Unified Trade Summary (used by backtester, evolution fitness & dashboard) ──

def summarize_trades(
    trades: List[Dict[str, Any]],
    initial_capital: float = 100000.0,
    final_equity: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build an institutional-grade performance report from a closed-trade list.
    Each trade dict needs at least a numeric "pnl" key.
    Never raises — returns zeros when there is nothing to measure.
    """
    pnls = [float(t.get("pnl", 0) or 0) for t in trades]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    # Per-trade return series (relative to starting capital) for Sharpe/Sortino
    returns = [p / initial_capital for p in pnls] if initial_capital > 0 else []

    # Equity curve from cumulative P&L
    equity_curve: List[float] = []
    running = initial_capital
    for p in pnls:
        running += p
        equity_curve.append(round(running, 2))
    if final_equity is None:
        final_equity = equity_curve[-1] if equity_curve else initial_capital

    total_pnl = round(sum(pnls), 2)
    max_dd = calculate_max_drawdown(equity_curve)
    total_return_pct = round((final_equity - initial_capital) / initial_capital * 100, 2) if initial_capital else 0.0
    # Annualize by trade count heuristic: assume ~252 trading days => scale by sqrt coverage
    calmar = calculate_calmar_ratio(abs(total_return_pct), max_dd) if total_pnl >= 0 else -calculate_calmar_ratio(abs(total_return_pct), max_dd)

    avg_win = round(sum(wins) / len(wins), 2) if wins else 0.0
    avg_loss = round(abs(sum(losses) / len(losses)), 2) if losses else 0.0
    win_rate = round(len(wins) / n, 4) if n else 0.0
    kelly = calculate_kelly_criterion(win_rate, avg_win, avg_loss) if n >= 5 else 0.1

    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate * 100, 1),
        "total_pnl": total_pnl,
        "final_equity": round(final_equity, 2),
        "return_pct": total_return_pct,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": calculate_profit_factor(trades),
        "sharpe": calculate_sharpe_ratio(returns),
        "sortino": calculate_sortino_ratio(returns),
        "max_drawdown_pct": max_dd,
        "calmar": calmar,
        "half_kelly_frac": kelly,
    }
