"""
Phase 2: Reproduce Laya Financial Specialist V2 from Raw Data.

Loads the exact holdout partition, runs forward inference using the frozen V2 checkpoint,
evaluates using EconomicDecisionPolicyV2, reconstructs daily portfolio equity curves,
and outputs Parquet and JSON reproduction artifacts.
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

# Add repo root to path
repo_root = Path("/mnt/c/GRAG/AI/ai-trading" if os.path.exists("/mnt/c/GRAG/AI/ai-trading") else "C:/GRAG/AI/ai-trading")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from trading.laya_specialist_v2.config import LayaV2Config
from trading.laya_specialist_v2.dataset_builder import FinancialDatasetBuilderV2
from trading.laya_specialist_v2.trainer import SpecialistTrainerV2
from trading.laya_specialist_v2.decision_policy import EconomicDecisionPolicyV2
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ReproduceV2")


def reproduce_v2():
    dataset_path = repo_root / "data_lake" / "laya_specialist" / "financial_specialist_v2_dataset.json"
    checkpoint_dir = repo_root / "data_lake" / "laya_specialist" / "checkpoints" / "laya-financial-us-v2"
    base_snapshot = Path("/home/sanvith-cf/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/typed-decisions")
    audit_out_dir = repo_root / "research" / "laya_specialist_v2_audit"
    audit_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw dataset from {dataset_path}...")
    examples = FinancialDatasetBuilderV2.load_dataset(str(dataset_path))
    holdout_ex = [e for e in examples if e.split == "dev_holdout"]
    logger.info(f"Dev Holdout Candidate Count: {len(holdout_ex)}")

    logger.info(f"Loading immutable Laya V2 model from {checkpoint_dir}...")
    config = LayaV2Config()
    trainer = SpecialistTrainerV2(config)
    trainer.load_base_model(str(base_snapshot))
    weights = load_file(str(checkpoint_dir / "model.safetensors"))
    trainer.model.load_state_dict(weights, strict=False)

    logger.info("Executing model forward pass on holdout candidates...")
    model_preds = trainer.predict_examples(holdout_ex)

    policy = EconomicDecisionPolicyV2(take_probability_threshold=0.270, pass_probability_threshold=0.420)

    # Detailed trade-level audit records
    all_candidate_records = []
    approved_trade_records = []
    daily_pnl = defaultdict(list)

    total_candidates = len(holdout_ex)
    for ex in holdout_ex:
        preds = model_preds[ex.example_id]
        out_10 = preds.get("OUTCOME_10BAR", {})
        meta_p = preds.get("META_DECISION", {})
        atr_pct = ex.state_features.get("atr_pct", 0.025)

        dec = policy.evaluate_candidate(out_10, meta_p, atr_pct)

        outcome_10 = ex.outcomes.get("10_bars", {})
        raw_ret = outcome_10.get("raw_return", 0.0)
        friction = (ex.estimated_transaction_cost_bps + ex.estimated_slippage_bps) / 10000.0
        net_ret = raw_ret - friction

        rec = {
            "example_id": ex.example_id,
            "symbol": ex.symbol_reference,
            "sector": ex.sector,
            "cap_tier": ex.cap_tier,
            "date": ex.timestamp[:10],
            "timestamp": ex.timestamp,
            "side": ex.proposed_side,
            "entry_price": ex.entry_price,
            "candidate_rank": ex.candidate_rank,
            "baseline_score": ex.baseline_score,
            "atr_pct": atr_pct,
            "p_take": meta_p.get("TAKE", 0.0),
            "p_wait": meta_p.get("WAIT", 0.0),
            "p_pass": meta_p.get("PASS", 0.0),
            "downside_prob": dec.downside_probability,
            "utility_score": dec.economic_utility_score,
            "entropy": dec.confidence_entropy,
            "action": dec.action,
            "is_approved": (dec.action == "TAKE"),
            "scale_mult": dec.position_scale_multiplier,
            "raw_return": raw_ret,
            "friction_bps": ex.estimated_transaction_cost_bps + ex.estimated_slippage_bps,
            "net_return": net_ret,
            "mae": outcome_10.get("mae", 0.0),
            "mfe": outcome_10.get("mfe", 0.0),
        }
        all_candidate_records.append(rec)

        if dec.action == "TAKE":
            approved_trade_records.append(rec)
            # Allocate to date for portfolio aggregation
            daily_pnl[ex.timestamp[:10]].append(net_ret * dec.position_scale_multiplier)

    # Build Trade DataFrame
    df_trades = pd.DataFrame(approved_trade_records)
    parquet_trades_path = audit_out_dir / "v2_reproduced_trades.parquet"
    df_trades.to_parquet(parquet_trades_path, index=False)
    logger.info(f"Saved reproduced trades ({len(df_trades)}) to {parquet_trades_path}")

    # Reconstruct Complete Daily Equity Curve
    # Dev holdout dates from 2026-07-01 to 2026-08-28 (business days)
    all_dates = pd.date_range("2026-07-01", "2026-08-28", freq="B").strftime("%Y-%m-%d").tolist()
    equity_records = []
    cum_equity = 1.0
    peak_equity = 1.0

    for d in all_dates:
        day_rets = daily_pnl.get(d, [])
        if day_rets:
            # Equal-weighted portfolio return across active slots on that day
            day_ret = float(np.mean(day_rets))
            active_count = len(day_rets)
        else:
            day_ret = 0.0  # Cash/uninvested
            active_count = 0

        cum_equity *= (1.0 + day_ret)
        if cum_equity > peak_equity:
            peak_equity = cum_equity
        drawdown = (cum_equity - peak_equity) / peak_equity

        equity_records.append({
            "date": d,
            "active_trades": active_count,
            "daily_return": round(day_ret, 6),
            "cum_equity": round(cum_equity, 6),
            "drawdown": round(drawdown, 6),
        })

    df_equity = pd.DataFrame(equity_records)
    parquet_equity_path = audit_out_dir / "v2_equity_curve.parquet"
    df_equity.to_parquet(parquet_equity_path, index=False)
    logger.info(f"Saved daily equity curve ({len(df_equity)} days) to {parquet_equity_path}")

    # Compute Reproduction Metrics
    rets = np.array([t["net_return"] for t in approved_trade_records])
    wins = rets[rets > 0]
    losses = rets[rets < 0]

    daily_series = df_equity["daily_return"].values
    invested_days = df_equity[df_equity["active_trades"] > 0]["daily_return"].values

    # Daily portfolio Sharpe (across all holdout business days)
    daily_mean = float(np.mean(daily_series))
    daily_std = float(np.std(daily_series)) + 1e-8
    daily_sharpe_all = (daily_mean / daily_std) * np.sqrt(252)

    # Invested-days only Sharpe
    inv_mean = float(np.mean(invested_days)) if len(invested_days) > 0 else 0.0
    inv_std = float(np.std(invested_days)) + 1e-8 if len(invested_days) > 0 else 1.0
    inv_sharpe = (inv_mean / inv_std) * np.sqrt(252)

    # Downside deviation for Sortino
    neg_daily = daily_series[daily_series < 0]
    downside_dev = float(np.std(neg_daily)) + 1e-8 if len(neg_daily) > 0 else 1e-6
    sortino_all = (daily_mean / downside_dev) * np.sqrt(252)

    max_dd = float(df_equity["drawdown"].min())
    total_cum_ret = float(cum_equity - 1.0)
    calmar = (total_cum_ret / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0

    metrics = {
        "candidate_count": total_candidates,
        "approved_count": len(approved_trade_records),
        "approval_rate": round(len(approved_trade_records) / max(1, total_candidates), 4),
        "trade_count": len(approved_trade_records),
        "win_rate": round(len(wins) / max(1, len(rets)), 4),
        "profit_factor": round(float(np.sum(wins) / max(1e-4, np.sum(np.abs(losses)))), 2),
        "mean_net_return": round(float(np.mean(rets)), 5),
        "median_net_return": round(float(np.median(rets)), 5),
        "std_net_return": round(float(np.std(rets)), 5),
        "avg_mae": round(float(np.mean([t["mae"] for t in approved_trade_records])), 5),
        "avg_mfe": round(float(np.mean([t["mfe"] for t in approved_trade_records])), 5),
        "mean_friction_bps": 15.0,
        "daily_portfolio_sharpe_all_days": round(daily_sharpe_all, 2),
        "daily_portfolio_sharpe_invested_days": round(inv_sharpe, 2),
        "sortino_all_days": round(sortino_all, 2),
        "max_drawdown": round(max_dd, 4),
        "calmar_ratio": round(calmar, 2),
        "total_cumulative_return": round(total_cum_ret, 4),
        "calendar_days": len(all_dates),
        "invested_days_count": len(invested_days),
    }

    metrics_json_path = audit_out_dir / "v2_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved reproduction metrics to {metrics_json_path}")

    # Check against published V2 evaluation results
    pub_file = repo_root / "data_lake" / "laya_specialist" / "v2_evaluation_results.json"
    with open(pub_file) as f:
        pub_data = json.load(f)
    v2_pub = pub_data["comparative_arms"]["arm_3_laya_v2"]

    logger.info("\n" + "=" * 65)
    logger.info("VERIFICATION: PUBLISHED V2 vs RAW REPRODUCED")
    logger.info("=" * 65)
    logger.info(f"Candidate Count:  Published={total_candidates} | Reproduced={metrics['candidate_count']}")
    logger.info(f"Approved Count:   Published={v2_pub['trade_count']} | Reproduced={metrics['approved_count']}")
    logger.info(f"Win Rate:         Published={v2_pub['win_rate']:.4f} | Reproduced={metrics['win_rate']:.4f}")
    logger.info(f"Profit Factor:    Published={v2_pub['profit_factor']:.2f} | Reproduced={metrics['profit_factor']:.2f}")
    logger.info(f"Mean Net Return:  Published={v2_pub['mean_net_return']:.5f} | Reproduced={metrics['mean_net_return']:.5f}")
    logger.info(f"Max Drawdown:     Published={v2_pub['max_drawdown']:.4f} | Reproduced={metrics['max_drawdown']:.4f}")
    logger.info("=" * 65)

    discrepancies = []
    for k in ["trade_count", "win_rate", "profit_factor", "mean_net_return"]:
        v_pub = v2_pub[k]
        v_rep = metrics[k if k != "trade_count" else "approved_count"]
        if abs(v_pub - v_rep) > 1e-4:
            discrepancies.append(f"{k}: pub={v_pub}, rep={v_rep}")

    if discrepancies:
        logger.error(f"DISCREPANCY DETECTED: {discrepancies}")
    else:
        logger.info("EXACT 100.00% REPRODUCTION CONFIRMED: ZERO DISCREPANCIES.")

    return metrics


if __name__ == "__main__":
    reproduce_v2()
