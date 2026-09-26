"""
Re-evaluates V2.1 with un-flattened temperature (T=1.0) and proper threshold sweep.
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

repo_root = Path("/mnt/c/GRAG/AI/ai-trading" if os.path.exists("/mnt/c/GRAG/AI/ai-trading") else "C:/GRAG/AI/ai-trading")
sys.path.insert(0, str(repo_root))

from trading.laya_specialist_v2.config import LayaV2Config
from trading.laya_specialist_v2.dataset_builder import FinancialDatasetBuilderV2
from trading.laya_specialist_v2.tasks import SpecialistTasksV2
from trading.laya_specialist_v2.decision_policy import EconomicDecisionPolicyV2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("V2_1_Sweep")


def evaluate_v2_1_thresholds():
    config = LayaV2Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_file = repo_root / "data_lake" / "laya_specialist" / "financial_specialist_v2_1_dataset.json"
    examples = FinancialDatasetBuilderV2.load_dataset(str(dataset_file))
    holdout_ex = [e for e in examples if e.split == "dev_holdout"]

    checkpoint_dir = repo_root / "data_lake" / "laya_specialist" / "checkpoints" / "laya-financial-us-v2.1"
    base_dir = "/home/sanvith-cf/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/typed-decisions"

    from laya.common import build_model, build_sequence
    with open(os.path.join(base_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir / "tokenizer"))
    enc_dir = os.path.join(base_dir, "encoder")
    model = build_model(cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None, pretrained=False)
    weights = load_file(str(checkpoint_dir / "model.safetensors"))
    model.load_state_dict(weights, strict=False)
    model.to(device)
    model.eval()

    task_meta = SpecialistTasksV2.get_task_dict("META_DECISION")
    task_out = SpecialistTasksV2.get_task_dict("OUTCOME_10BAR")
    meta_choices = list(task_meta["crit"].keys())
    out_choices = list(task_out["crit"].keys())

    all_dates = sorted(list(set([d.timestamp[:10] for d in holdout_ex])))

    # Compute raw logits and probs at T=1.0
    preds_t1 = []
    with torch.no_grad():
        for ex in holdout_ex:
            narrative = ex.state_narrative
            input_ids_m, markers_m = build_sequence(tokenizer, narrative, task_meta, max_len=config.max_length)
            input_ids_tm = torch.tensor([input_ids_m], device=device)
            markers_tm = torch.tensor([markers_m], device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                logits_m, _ = model(input_ids_tm, torch.ones_like(input_ids_tm), markers_tm, torch.ones_like(markers_tm, dtype=torch.bool), torch.tensor([0], device=device))
                probs_m = torch.softmax(logits_m, dim=-1).cpu().numpy()[0]  # T = 1.0
            meta_p = {c: float(p) for c, p in zip(meta_choices, probs_m)}

            input_ids_o, markers_o = build_sequence(tokenizer, narrative, task_out, max_len=config.max_length)
            input_ids_to = torch.tensor([input_ids_o], device=device)
            markers_to = torch.tensor([markers_o], device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                logits_o, _ = model(input_ids_to, torch.ones_like(input_ids_to), markers_to, torch.ones_like(markers_to, dtype=torch.bool), torch.tensor([0], device=device))
                probs_o = torch.softmax(logits_o, dim=-1).cpu().numpy()[0]  # T = 1.0
            out_10 = {c: float(p) for c, p in zip(out_choices, probs_o)}

            out_10b = ex.outcomes["10_bars"]
            raw_ret = out_10b["raw_return"]
            friction = (ex.estimated_transaction_cost_bps + ex.estimated_slippage_bps) / 10000.0
            net_ret = raw_ret - friction

            preds_t1.append({
                "example": ex,
                "meta_p": meta_p,
                "out_10": out_10,
                "net_ret": net_ret,
                "raw_ret": raw_ret,
            })

    # Sweep thresholds at T=1.0
    sweep = []
    for th in [0.25, 0.27, 0.30, 0.33, 0.35, 0.38, 0.40]:
        pol = EconomicDecisionPolicyV2(take_probability_threshold=th, pass_probability_threshold=0.420)
        app_trades = []
        d_pnl = {d: [] for d in all_dates}

        for p in preds_t1:
            ex = p["example"]
            atr_pct = ex.state_features.get("atr_pct", 0.025)
            dec = pol.evaluate_candidate(p["out_10"], p["meta_p"], atr_pct)
            if dec.action == "TAKE":
                app_trades.append(p["net_ret"])
                d_pnl[ex.timestamp[:10]].append(p["net_ret"] * dec.position_scale_multiplier)

        cnt = len(app_trades)
        if cnt > 0:
            arr = np.array(app_trades)
            w = arr[arr > 0]
            l = arr[arr < 0]
            wr = float(len(w) / len(arr))
            pf = float(np.sum(w) / max(1e-4, np.sum(np.abs(l))))
            m_ret = float(np.mean(arr))
            d_arr = np.array([float(np.mean(d_pnl[dt])) if len(d_pnl[dt]) > 0 else 0.0 for dt in all_dates])
            sh = float((np.mean(d_arr) / (np.std(d_arr) + 1e-8)) * np.sqrt(252.0))
        else:
            wr, pf, m_ret, sh = 0.0, 0.0, 0.0, 0.0

        sweep.append({
            "threshold": th,
            "trade_count": cnt,
            "selectivity_pct": round(cnt / len(holdout_ex) * 100, 2),
            "win_rate": round(wr, 4),
            "profit_factor": round(pf, 2),
            "mean_net_return": round(m_ret, 5),
            "daily_portfolio_sharpe": round(sh, 2),
        })

    print("\n" + "=" * 80)
    print("V2.1 THRESHOLD SWEEP (T=1.0)")
    print("=" * 80)
    print(f"{'Theta':<8} | {'Trades':<8} | {'Selectivity':<12} | {'Win Rate':<10} | {'PF':<6} | {'Daily Sharpe'}")
    print("-" * 80)
    for r in sweep:
        print(f"{r['threshold']:<8.2f} | {r['trade_count']:<8} | {r['selectivity_pct']:>10.1f}% | {r['win_rate']*100:>8.1f}% | {r['profit_factor']:>6.2f} | {r['daily_portfolio_sharpe']:>12.2f}")
    print("=" * 80)

    # Save updated metadata
    meta = {
        "model_version": "laya-financial-us-v2.1",
        "frozen": True,
        "base_model": "convaiinnovations/laya-typed-decisions",
        "temperature": 1.0,
        "recommended_threshold": 0.35,
        "threshold_sweep": sweep,
    }
    with open(checkpoint_dir / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return sweep


if __name__ == "__main__":
    evaluate_v2_1_thresholds()
