"""
Phase 3 & Phase 4: Sharpe Calculation Audit & True Unit of Observation Analysis.

Audits:
1. Return frequency and observation frequency.
2. Annualization factor (sqrt(252) vs sqrt(25.2) vs trade cycle).
3. Active-days only vs All-calendar business days (cash drag).
4. Trade concentration: 10-bar trade returns recorded on entry date vs multi-bar accretion.
5. True Unit of Observation: Candidates vs Trades vs Sessions vs Portfolio-Days.
6. Exact computation of Sharpe, Sortino, Calmar, and Drawdown under each definition.
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

repo_root = Path("/mnt/c/GRAG/AI/ai-trading" if os.path.exists("/mnt/c/GRAG/AI/ai-trading") else "C:/GRAG/AI/ai-trading")
audit_out_dir = repo_root / "research" / "laya_specialist_v2_audit"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SharpeAudit")


def run_sharpe_audit():
    trades_path = audit_out_dir / "v2_reproduced_trades.parquet"
    equity_path = audit_out_dir / "v2_equity_curve.parquet"

    df_trades = pd.read_parquet(trades_path)
    df_equity = pd.read_parquet(equity_path)

    # 1. Trade-level series
    rets = df_trades["net_return"].values
    mean_trade_ret = float(np.mean(rets))
    std_trade_ret = float(np.std(rets)) + 1e-8
    # If 10-bar holding period, approx 25.2 cycles per year
    trade_sharpe_25 = (mean_trade_ret / std_trade_ret) * np.sqrt(25.2)
    # If incorrectly annualized as daily trades:
    trade_sharpe_252 = (mean_trade_ret / std_trade_ret) * np.sqrt(252.0)

    # 2. Concentrated active entry dates (V2 evaluator original methodology)
    active_days = df_equity[df_equity["active_trades"] > 0]["daily_return"].values
    mean_active = float(np.mean(active_days))
    std_active = float(np.std(active_days)) + 1e-8
    sharpe_active = (mean_active / std_active) * np.sqrt(252.0)

    # 3. All calendar business days (including cash drag on 0-return days)
    all_days = df_equity["daily_return"].values
    mean_all = float(np.mean(all_days))
    std_all = float(np.std(all_days)) + 1e-8
    sharpe_all_calendar = (mean_all / std_all) * np.sqrt(252.0)

    # 4. Sortino across all days
    neg_days = all_days[all_days < 0]
    downside_dev = float(np.std(neg_days)) + 1e-8 if len(neg_days) > 0 else 1e-6
    sortino_all = (mean_all / downside_dev) * np.sqrt(252.0)

    # 5. Calmar Ratio
    max_dd = float(df_equity["drawdown"].min())
    total_cum_ret = float(df_equity["cum_equity"].iloc[-1] - 1.0)
    # Annualized return over the holdout window (43 business days = 43/252 = 0.1706 years)
    years = len(all_days) / 252.0
    ann_return = ((1.0 + total_cum_ret) ** (1.0 / max(1e-4, years))) - 1.0
    calmar_ann = (ann_return / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0

    # 6. Unit of Observation Breakdown (Phase 4)
    unique_candidates = 371
    unique_approved = len(df_trades)
    unique_sessions = df_trades["date"].nunique()
    total_holdout_sessions = len(df_equity)

    # Cross-sectional clustering (multiple candidates on same day)
    trades_per_day = df_trades.groupby("date").size()
    max_trades_single_day = int(trades_per_day.max())
    mean_trades_active_day = float(trades_per_day.mean())

    audit_results = {
        "unit_of_observation": {
            "total_candidates": unique_candidates,
            "approved_candidates": unique_approved,
            "unique_entry_trading_sessions": unique_sessions,
            "total_calendar_business_days": total_holdout_sessions,
            "non_invested_days": total_holdout_sessions - unique_sessions,
            "max_trades_in_single_session": max_trades_single_day,
            "mean_trades_per_active_session": round(mean_trades_active_day, 2),
            "effective_independent_portfolio_observations": total_holdout_sessions,
        },
        "sharpe_methodology_comparison": {
            "method_1_v2_published_evaluator": {
                "annualization_factor": "sqrt(252)",
                "observation_basis": "Active entry days only (19 days)",
                "cash_days_included": False,
                "sharpe": 8.32,
                "notes": "Flawed: Excludes 24 cash days and concentrates 10-bar holding return onto single entry date.",
            },
            "method_2_active_days_exact": {
                "annualization_factor": "sqrt(252)",
                "observation_basis": "Active entry days (19 days)",
                "cash_days_included": False,
                "mean_return": round(mean_active, 5),
                "std_return": round(std_active, 5),
                "sharpe": round(sharpe_active, 2),
            },
            "method_3_all_calendar_portfolio_days": {
                "annualization_factor": "sqrt(252)",
                "observation_basis": "All business days in holdout (43 days)",
                "cash_days_included": True,
                "mean_daily_return": round(mean_all, 5),
                "std_daily_return": round(std_all, 5),
                "sharpe": round(sharpe_all_calendar, 2),
                "sortino": round(sortino_all, 2),
                "notes": "True daily portfolio performance including cash drag on uninvested days.",
            },
            "method_4_trade_cycle_annualized": {
                "annualization_factor": "sqrt(25.2)",
                "observation_basis": "28 approved trades (10-bar holding horizon)",
                "cash_days_included": "N/A",
                "mean_trade_net_return": round(mean_trade_ret, 5),
                "std_trade_net_return": round(std_trade_ret, 5),
                "sharpe": round(trade_sharpe_25, 2),
                "notes": "True per-trade annualized Sharpe assuming 25.2 non-overlapping 10-bar cycles/year.",
            },
            "method_5_naive_trade_annualized": {
                "annualization_factor": "sqrt(252)",
                "observation_basis": "28 approved trades treated as 1-day trades",
                "sharpe": round(trade_sharpe_252, 2),
                "notes": "Severely inflated: treats 10-day trade as 1-day trade.",
            },
        },
        "risk_adjusted_metrics_all_days": {
            "daily_sharpe": round(sharpe_all_calendar, 2),
            "daily_sortino": round(sortino_all, 2),
            "max_drawdown": round(max_dd, 4),
            "calmar_ratio_annualized": round(calmar_ann, 2),
            "cumulative_return": round(total_cum_ret, 4),
        }
    }

    out_file = audit_out_dir / "sharpe_audit_results.json"
    with open(out_file, "w") as f:
        json.dump(audit_results, f, indent=2)
    logger.info(f"Saved Sharpe audit results to {out_file}")

    print("\n" + "=" * 70)
    print("PHASE 3 & 4: SHARPE CALCULATION & OBSERVATION AUDIT RESULTS")
    print("=" * 70)
    for m_id, m_data in audit_results["sharpe_methodology_comparison"].items():
        print(f"  {m_id}: Sharpe = {m_data['sharpe']} ({m_data['observation_basis']}, {m_data['annualization_factor']})")
    print("-" * 70)
    print(f"  True All-Calendar Days Sharpe:  {audit_results['risk_adjusted_metrics_all_days']['daily_sharpe']}")
    print(f"  True Trade-Cycle (25.2) Sharpe: {audit_results['sharpe_methodology_comparison']['method_4_trade_cycle_annualized']['sharpe']}")
    print(f"  True Sortino (All Days):        {audit_results['risk_adjusted_metrics_all_days']['daily_sortino']}")
    print(f"  Max Drawdown:                   {audit_results['risk_adjusted_metrics_all_days']['max_drawdown']}")
    print("=" * 70)

    return audit_results


if __name__ == "__main__":
    run_sharpe_audit()
