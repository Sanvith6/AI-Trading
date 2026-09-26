"""
Generates and caches exact model forward prediction probabilities for all 371 holdout candidates.
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
from trading.laya_specialist_v2.tasks import SpecialistTasksV2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CachePredictions")


def cache_holdout_predictions():
    config = LayaV2Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    dataset_file = repo_root / "data_lake" / "laya_specialist" / "financial_specialist_v2_dataset.json"
    with open(dataset_file) as f:
        all_data = json.load(f)
    holdout_dicts = [d for d in all_data if d["split"] == "dev_holdout"]
    logger.info(f"Loaded {len(holdout_dicts)} Dev Holdout candidates.")

    base_dir = "/home/sanvith-cf/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/typed-decisions"
    checkpoint_dir = repo_root / "data_lake" / "laya_specialist" / "checkpoints" / "laya-financial-us-v2"

    from laya.common import build_model, build_sequence
    with open(os.path.join(base_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(base_dir, "tokenizer"))
    enc_dir = os.path.join(base_dir, "encoder")
    model = build_model(cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None, pretrained=False)
    weights = load_file(str(checkpoint_dir / "model.safetensors"))
    model.load_state_dict(weights, strict=False)
    model.to(device)
    model.eval()

    with open(checkpoint_dir / "model_metadata.json") as f:
        meta = json.load(f)
    temperature = meta.get("temperature", 1.0)

    task_meta = SpecialistTasksV2.get_task_dict("META_DECISION")
    task_out = SpecialistTasksV2.get_task_dict("OUTCOME_10BAR")
    meta_choices = list(task_meta["crit"].keys())
    out_choices = list(task_out["crit"].keys())

    records = []
    logger.info("Running forward inference across all 371 candidates...")

    for ex in holdout_dicts:
        narrative = ex["state_narrative"]

        # META_DECISION
        input_ids_m, markers_m = build_sequence(tokenizer, narrative, task_meta, max_len=config.max_length)
        input_ids_tm = torch.tensor([input_ids_m], device=device)
        markers_tm = torch.tensor([markers_m], device=device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                logits_m, _ = model(input_ids_tm, torch.ones_like(input_ids_tm), markers_tm, torch.ones_like(markers_tm, dtype=torch.bool), torch.tensor([0], device=device))
                probs_m = torch.softmax(logits_m / temperature, dim=-1).cpu().numpy()[0]

        meta_p = {c: float(p) for c, p in zip(meta_choices, probs_m)}

        # OUTCOME_10BAR
        input_ids_o, markers_o = build_sequence(tokenizer, narrative, task_out, max_len=config.max_length)
        input_ids_to = torch.tensor([input_ids_o], device=device)
        markers_to = torch.tensor([markers_o], device=device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                logits_o, _ = model(input_ids_to, torch.ones_like(input_ids_to), markers_to, torch.ones_like(markers_to, dtype=torch.bool), torch.tensor([0], device=device))
                probs_o = torch.softmax(logits_o / temperature, dim=-1).cpu().numpy()[0]

        out_10 = {c: float(p) for c, p in zip(out_choices, probs_o)}

        records.append({
            "example_id": ex["example_id"],
            "prob_take": meta_p.get("TAKE", 0.0),
            "prob_wait": meta_p.get("WAIT", 0.0),
            "prob_pass": meta_p.get("PASS", 0.0),
            "prob_pos": out_10.get("POSITIVE", 0.0) + out_10.get("STRONGLY_POSITIVE", 0.0),
            "prob_neg": out_10.get("NEGATIVE", 0.0) + out_10.get("STRONGLY_NEGATIVE", 0.0),
            "meta_p_json": json.dumps(meta_p),
            "out_10_json": json.dumps(out_10),
        })

    df = pd.DataFrame(records)
    out_file = repo_root / "research" / "laya_specialist_v2_audit" / "v2_holdout_model_predictions.parquet"
    df.to_parquet(out_file)
    logger.info(f"Model predictions cached to {out_file}")


if __name__ == "__main__":
    cache_holdout_predictions()
