"""
Phase 6: 1,000+ Prediction Permutation Null Distribution Audit.

Executes 1,000 independent prediction permutations on the 371 holdout candidates.
For every permutation, calculates:
- Profit Factor
- Daily All-Days Sharpe
- Holding-Cycle Sharpe
- Win Rate
- Mean Net Return
- Maximum Drawdown

Builds the empirical null distribution and calculates exact empirical p-values for Laya V2.
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

repo_root = Path("/mnt/c/GRAG/AI/ai-trading" if os.path.exists("/mnt/c/GRAG/AI/ai-trading") else "C:/GRAG/AI/ai-trading")
audit_out_dir = repo_root / "research" / "laya_specialist_v2_audit"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PermutationAudit")


def run_permutation_audit(num_permutations: int = 1000):
    dataset_path = repo_root / "data_lake" / "laya_specialist" / "financial_specialist_v2_dataset.json"
    with open(dataset_path) as f:
        all_data = json.load(f)
    holdout_data = [d for d in all_data if d["split"] == "dev_holdout"]

    all_dates = pd.date_range("2026-07-01", "2026-08-28", freq="B").strftime("%Y-%m-%d").tolist()

    all_recs = []
    for d in holdout_data:
        out_10 = d["outcomes"]["10_bars"]
        raw_r = out_10["raw_return"]
        friction = (d["estimated_transaction_cost_bps"] + d["estimated_slippage_bps"]) / 10000.0
        net_r = raw_r - friction
        all_recs.append({
            "example_id": d["example_id"],
            "date": d["timestamp"][:10],
            "net_return": net_r,
        })
    df_all = pd.DataFrame(all_recs)

    # Observed Laya V2 Real Metrics
    laya_trades_path = audit_out_dir / "v2_reproduced_trades.parquet"
    df_laya = pd.read_parquet(laya_trades_path)
    
    rets_laya = df_laya["net_return"].values
    wins_l = rets_laya[rets_laya > 0]
    losses_l = rets_laya[rets_laya < 0]
    obs_pf = float(np.sum(wins_l) / max(1e-4, np.sum(np.abs(losses_l))))
    obs_wr = float(len(wins_l) / len(rets_laya))
    obs_mean_ret = float(np.mean(rets_laya))

    # Daily aggregation for Laya observed Sharpe
    daily_pnl_l = defaultdict(list)
    for _, r in df_laya.iterrows():
        daily_pnl_l[r["date"]].append(r["net_return"] * r.get("scale_mult", 1.0))
    daily_rets_l = [float(np.mean(daily_pnl_l.get(d, [0.0]))) if daily_pnl_l.get(d) else 0.0 for d in all_dates]
    obs_sharpe_daily = float((np.mean(daily_rets_l) / (np.std(daily_rets_l) + 1e-8)) * np.sqrt(252.0))
    obs_sharpe_cycle = float((np.mean(rets_laya) / (np.std(rets_laya) + 1e-8)) * np.sqrt(25.2))

    logger.info(f"Observed Laya V2: PF={obs_pf:.2f}, WR={obs_wr:.1%}, DailySharpe={obs_sharpe_daily:.2f}, CycleSharpe={obs_sharpe_cycle:.2f}, MeanRet={obs_mean_ret*100:+.2f}%")
    logger.info(f"Running {num_permutations} independent prediction permutations...")

    np.random.seed(42)
    null_pf = []
    null_sharpe_daily = []
    null_sharpe_cycle = []
    null_wr = []
    null_mean_ret = []
    null_max_dd = []

    for i in range(num_permutations):
        # Sample exactly 28 trades at random
        sampled = df_all.sample(n=28, replace=False)
        r_arr = sampled["net_return"].values
        w = r_arr[r_arr > 0]
        l = r_arr[r_arr < 0]

        pf = float(np.sum(w) / max(1e-4, np.sum(np.abs(l))))
        wr = float(len(w) / len(r_arr))
        m_ret = float(np.mean(r_arr))
        s_ret = float(np.std(r_arr)) + 1e-8
        c_sharpe = (m_ret / s_ret) * np.sqrt(25.2)

        # Daily equity
        d_pnl = defaultdict(list)
        for _, row in sampled.iterrows():
            d_pnl[row["date"]].append(row["net_return"])
        
        c_eq = 1.0
        peak = 1.0
        d_rets = []
        dds = []
        for d in all_dates:
            day_val = float(np.mean(d_pnl.get(d, [0.0]))) if d_pnl.get(d) else 0.0
            c_eq *= (1.0 + day_val)
            if c_eq > peak:
                peak = c_eq
            d_rets.append(day_val)
            dds.append((c_eq - peak) / peak)

        d_arr = np.array(d_rets)
        d_sharpe = float((np.mean(d_arr) / (np.std(d_arr) + 1e-8)) * np.sqrt(252.0))

        null_pf.append(pf)
        null_sharpe_daily.append(d_sharpe)
        null_sharpe_cycle.append(c_sharpe)
        null_wr.append(wr)
        null_mean_ret.append(m_ret)
        null_max_dd.append(float(np.min(dds)))

    null_pf = np.array(null_pf)
    null_sharpe_daily = np.array(null_sharpe_daily)
    null_sharpe_cycle = np.array(null_sharpe_cycle)
    null_wr = np.array(null_wr)
    null_mean_ret = np.array(null_mean_ret)
    null_max_dd = np.array(null_max_dd)

    # Empirical p-values: P(Null >= Observed)
    p_val_pf = float(np.mean(null_pf >= obs_pf))
    p_val_sharpe_daily = float(np.mean(null_sharpe_daily >= obs_sharpe_daily))
    p_val_sharpe_cycle = float(np.mean(null_sharpe_cycle >= obs_sharpe_cycle))
    p_val_wr = float(np.mean(null_wr >= obs_wr))
    p_val_mean_ret = float(np.mean(null_mean_ret >= obs_mean_ret))

    def get_stats(arr):
        return {
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "p95": round(float(np.percentile(arr, 95.0)), 4),
            "p99": round(float(np.percentile(arr, 99.0)), 4),
            "p99_5": round(float(np.percentile(arr, 99.5)), 4),
            "p99_9": round(float(np.percentile(arr, 99.9)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    null_summary = {
        "num_permutations": num_permutations,
        "observed_laya_v2": {
            "profit_factor": round(obs_pf, 2),
            "daily_all_sharpe": round(obs_sharpe_daily, 2),
            "cycle_sharpe": round(obs_sharpe_cycle, 2),
            "win_rate": round(obs_wr, 4),
            "mean_net_return": round(obs_mean_ret, 5),
        },
        "empirical_p_values": {
            "profit_factor_p_value": p_val_pf,
            "daily_sharpe_p_value": p_val_sharpe_daily,
            "cycle_sharpe_p_value": p_val_sharpe_cycle,
            "win_rate_p_value": p_val_wr,
            "mean_net_return_p_value": p_val_mean_ret,
        },
        "null_distributions": {
            "profit_factor": get_stats(null_pf),
            "daily_all_sharpe": get_stats(null_sharpe_daily),
            "cycle_sharpe": get_stats(null_sharpe_cycle),
            "win_rate": get_stats(null_wr),
            "mean_net_return": get_stats(null_mean_ret),
            "max_drawdown": get_stats(null_max_dd),
        }
    }

    out_file = audit_out_dir / "permutation_null_results.json"
    with open(out_file, "w") as f:
        json.dump(null_summary, f, indent=2)

    df_null_runs = pd.DataFrame({
        "profit_factor": null_pf,
        "daily_sharpe": null_sharpe_daily,
        "cycle_sharpe": null_sharpe_cycle,
        "win_rate": null_wr,
        "mean_net_return": null_mean_ret,
        "max_drawdown": null_max_dd,
    })
    df_null_runs.to_parquet(audit_out_dir / "permutation_null_runs.parquet", index=False)

    print("\n" + "=" * 80)
    print(f"PHASE 6: 1,000 PREDICTION PERMUTATION NULL DISTRIBUTION RESULTS")
    print("=" * 80)
    print(f"Metric             Observed Laya V2   Null Mean (Std)       Null 99th Pct    Empirical p-value")
    print(f"--------------------------------------------------------------------------------")
    pf_st = null_summary["null_distributions"]["profit_factor"]
    print(f"Profit Factor      {obs_pf:<18.2f} {pf_st['mean']:<6.2f} (+/- {pf_st['std']:<4.2f})    {pf_st['p99']:<16.2f} {p_val_pf:<10.4f} {'***' if p_val_pf < 0.01 else ''}")
    
    ds_st = null_summary["null_distributions"]["daily_all_sharpe"]
    print(f"Daily Sharpe       {obs_sharpe_daily:<18.2f} {ds_st['mean']:<6.2f} (+/- {ds_st['std']:<4.2f})    {ds_st['p99']:<16.2f} {p_val_sharpe_daily:<10.4f} {'***' if p_val_sharpe_daily < 0.01 else ''}")
    
    cs_st = null_summary["null_distributions"]["cycle_sharpe"]
    print(f"Cycle Sharpe       {obs_sharpe_cycle:<18.2f} {cs_st['mean']:<6.2f} (+/- {cs_st['std']:<4.2f})    {cs_st['p99']:<16.2f} {p_val_sharpe_cycle:<10.4f} {'***' if p_val_sharpe_cycle < 0.01 else ''}")
    
    wr_st = null_summary["null_distributions"]["win_rate"]
    print(f"Win Rate           {obs_wr*100:<17.1f}% {wr_st['mean']*100:<5.1f}% (+/- {wr_st['std']*100:<3.1f}%)   {wr_st['p99']*100:<15.1f}% {p_val_wr:<10.4f} {'***' if p_val_wr < 0.01 else ''}")
    
    mr_st = null_summary["null_distributions"]["mean_net_return"]
    print(f"Mean Net Return    {obs_mean_ret*100:<17.2f}% {mr_st['mean']*100:<5.2f}% (+/- {mr_st['std']*100:<3.2f}%)   {mr_st['p99']*100:<15.2f}% {p_val_mean_ret:<10.4f} {'***' if p_val_mean_ret < 0.01 else ''}")
    print("=" * 80)

    return null_summary


if __name__ == "__main__":
    run_permutation_audit(1000)
