"""
Financial Specialist V2 Trainer & Calibrator.

Hardware Target: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
Backbone: ModernBERT-large (421M parameters) + Typed Decision Head
Base Snapshot: convaiinnovations/laya-typed-decisions (Arm B winner)
Optimizer: Adafactor (O(1) memory optimizer state ~25 MB)
Precision: Mixed Precision (fp16/bf16 autocast) + Gradient Checkpointing
Batching: Micro-batch 1, Gradient Accumulation 32 (effective batch 32)
"""

import os
import sys
import json
import time
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.optimization import Adafactor

from .config import LayaV2Config
from .tasks import SpecialistTasksV2
from .dataset_builder import FinancialCandidateExampleV2, FinancialDatasetBuilderV2

logger = logging.getLogger(__name__)


class SpecialistTrainerV2:
    """End-to-end trainer and calibrator for Laya Financial Specialist V2."""

    def __init__(self, config: Optional[LayaV2Config] = None):
        self.config = config or LayaV2Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.temperature: float = 1.0

    def load_base_model(self, snapshot_path: Optional[str] = None):
        """Loads Arm B winning base (laya-typed-decisions) with gradient checkpointing."""
        from laya.common import build_model
        from safetensors.torch import load_file

        base_dir = snapshot_path or self.config.base_snapshot_wsl
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Base model snapshot not found at: {base_dir}")

        logger.info(f"Loading Laya Specialist V2 base from {base_dir}...")
        with open(os.path.join(base_dir, "rl_agent_config.json")) as f:
            cfg = json.load(f)
        cfg["gradient_checkpointing"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(base_dir, "tokenizer"))
        enc_dir = os.path.join(base_dir, "encoder")
        self.model = build_model(cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None, pretrained=False)

        weights_file = os.path.join(base_dir, "model.safetensors")
        weights = load_file(weights_file)
        self.model.load_state_dict(weights, strict=False)

        if hasattr(self.model.encoder, "gradient_checkpointing_enable"):
            self.model.encoder.gradient_checkpointing_enable()

        self.model.to(self.device)
        logger.info(f"Base model loaded successfully onto {self.device}.")

    def train_epoch(
        self,
        train_examples: List[FinancialCandidateExampleV2],
        val_examples: List[FinancialCandidateExampleV2],
        max_steps: int = 500,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Executes high-efficiency Adafactor fine-tuning across 6 typed choice tasks.
        """
        from laya.common import build_sequence

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model.train()

        # Adafactor: maintains factorized second moments (~25 MB optimizer states)
        optimizer = Adafactor(
            self.model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
        )

        task_keys = ["META_DECISION", "OUTCOME_10BAR", "TRADE_QUALITY", "RISK_CONDITION"]
        task_dicts = {k: SpecialistTasksV2.get_task_dict(k) for k in task_keys}

        train_losses = []
        step_count = 0
        accum_steps = self.config.gradient_accumulation_steps
        optimizer.zero_grad(set_to_none=True)

        logger.info(f"Starting V2 training: {max_steps} steps, effective batch {accum_steps}, seed {seed}...")
        t_start = time.perf_counter()

        for step in range(max_steps):
            ex = train_examples[step % len(train_examples)]
            t_key = task_keys[step % len(task_keys)]
            q_dict = task_dicts[t_key]
            target_probs = list(ex.targets[t_key].values())

            input_ids, markers = build_sequence(
                self.tokenizer,
                ex.state_narrative,
                q_dict,
                max_len=self.config.max_length,
            )

            input_ids_t = torch.tensor([input_ids], device=self.device)
            attention_mask_t = torch.ones_like(input_ids_t, device=self.device)
            marker_pos_t = torch.tensor([markers], device=self.device)
            marker_mask_t = torch.ones_like(marker_pos_t, dtype=torch.bool, device=self.device)
            qtype_t = torch.tensor([0], device=self.device)
            target_t = torch.tensor([target_probs], dtype=torch.float32, device=self.device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                logits, _ = self.model(input_ids_t, attention_mask_t, marker_pos_t, marker_mask_t, qtype_t)
                loss = -(target_t * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                loss = loss / accum_steps

            loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == max_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1

            train_losses.append(loss.item() * accum_steps)

            if (step + 1) % 100 == 0:
                elapsed = time.perf_counter() - t_start
                logger.info(f"Step {step+1}/{max_steps} | Loss: {np.mean(train_losses[-50:]):.4f} | Time: {elapsed:.1f}s")

        # Validation evaluation
        val_loss, val_brier = self.evaluate_loss(val_examples)

        return {
            "seed": seed,
            "total_steps": max_steps,
            "train_loss": round(float(np.mean(train_losses)), 4),
            "val_loss": round(float(val_loss), 4),
            "val_brier": round(float(val_brier), 4),
            "duration_sec": round(time.perf_counter() - t_start, 1),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        }

    def evaluate_loss(self, examples: List[FinancialCandidateExampleV2], limit: int = 100) -> Tuple[float, float]:
        """Evaluates cross-entropy loss and Brier score on a dataset split."""
        from laya.common import build_sequence

        self.model.eval()
        task_dict = SpecialistTasksV2.get_task_dict("META_DECISION")
        losses, briers = [], []

        eval_subset = examples[:limit]
        with torch.no_grad():
            for ex in eval_subset:
                target_probs = np.array(list(ex.targets["META_DECISION"].values()), dtype=np.float32)
                input_ids, markers = build_sequence(
                    self.tokenizer,
                    ex.state_narrative,
                    task_dict,
                    max_len=self.config.max_length,
                )
                input_ids_t = torch.tensor([input_ids], device=self.device)
                attention_mask_t = torch.ones_like(input_ids_t, device=self.device)
                marker_pos_t = torch.tensor([markers], device=self.device)
                marker_mask_t = torch.ones_like(marker_pos_t, dtype=torch.bool, device=self.device)
                qtype_t = torch.tensor([0], device=self.device)
                target_t = torch.tensor([target_probs], dtype=torch.float32, device=self.device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    logits, _ = self.model(input_ids_t, attention_mask_t, marker_pos_t, marker_mask_t, qtype_t)
                    loss = -(target_t * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

                losses.append(loss.item())
                briers.append(np.mean((probs - target_probs) ** 2))

        return float(np.mean(losses)), float(np.mean(briers))

    def calibrate_temperature(self, cal_examples: List[FinancialCandidateExampleV2]) -> float:
        """
        Derives scalar temperature scaling parameter T strictly on the Calibration partition.
        """
        from laya.common import build_sequence
        self.model.eval()
        task_dict = SpecialistTasksV2.get_task_dict("META_DECISION")

        all_logits = []
        all_targets = []

        with torch.no_grad():
            for ex in cal_examples[:150]:
                target_probs = list(ex.targets["META_DECISION"].values())
                input_ids, markers = build_sequence(
                    self.tokenizer, ex.state_narrative, task_dict, max_len=self.config.max_length
                )
                input_ids_t = torch.tensor([input_ids], device=self.device)
                attention_mask_t = torch.ones_like(input_ids_t, device=self.device)
                marker_pos_t = torch.tensor([markers], device=self.device)
                marker_mask_t = torch.ones_like(marker_pos_t, dtype=torch.bool, device=self.device)
                qtype_t = torch.tensor([0], device=self.device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    logits, _ = self.model(input_ids_t, attention_mask_t, marker_pos_t, marker_mask_t, qtype_t)

                all_logits.append(logits.float().cpu())
                all_targets.append(torch.tensor([target_probs], dtype=torch.float32))

        logits_cat = torch.cat(all_logits, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)

        # Optimize temperature T via grid search over [0.5, 3.0]
        best_t = 1.0
        best_nll = float("inf")
        for t_val in np.linspace(0.5, 3.0, 51):
            scaled_log_probs = torch.log_softmax(logits_cat / t_val, dim=-1)
            nll = -(targets_cat * scaled_log_probs).sum(dim=-1).mean().item()
            if nll < best_nll:
                best_nll = nll
                best_t = float(t_val)

        self.temperature = best_t
        logger.info(f"Optimal calibration temperature derived on Cal set: T = {self.temperature:.3f} (NLL: {best_nll:.4f})")
        return self.temperature

    def predict_examples(
        self,
        examples: List[FinancialCandidateExampleV2],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Runs true forward pass on examples to produce predicted probability distributions.
        GUARANTEE: NO GROUND-TRUTH LEAKAGE! Output comes purely from model weights.
        """
        from laya.common import build_sequence
        self.model.eval()

        predictions = {}
        task_names = ["META_DECISION", "OUTCOME_10BAR"]
        task_dicts = {k: SpecialistTasksV2.get_task_dict(k) for k in task_names}
        choices_map = {k: list(task_dicts[k]["crit"].keys()) for k in task_names}

        with torch.no_grad():
            for ex in examples:
                ex_preds = {}
                for t_name in task_names:
                    q_dict = task_dicts[t_name]
                    choices = choices_map[t_name]

                    input_ids, markers = build_sequence(
                        self.tokenizer,
                        ex.state_narrative,
                        q_dict,
                        max_len=self.config.max_length,
                    )
                    input_ids_t = torch.tensor([input_ids], device=self.device)
                    attention_mask_t = torch.ones_like(input_ids_t, device=self.device)
                    marker_pos_t = torch.tensor([markers], device=self.device)
                    marker_mask_t = torch.ones_like(marker_pos_t, dtype=torch.bool, device=self.device)
                    qtype_t = torch.tensor([0], device=self.device)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                        logits, _ = self.model(input_ids_t, attention_mask_t, marker_pos_t, marker_mask_t, qtype_t)
                        # Apply calibrated temperature scaling
                        probs = torch.softmax(logits / self.temperature, dim=-1).cpu().numpy()[0]

                    ex_preds[t_name] = {c: round(float(p), 4) for c, p in zip(choices, probs)}

                predictions[ex.example_id] = ex_preds

        return predictions

    def save_checkpoint(self, output_dir: Optional[str] = None):
        """Saves immutable model checkpoint, tokenizer, and config to data lake."""
        from safetensors.torch import save_file

        out_path = Path(output_dir or self.config.checkpoint_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving immutable Laya V2 checkpoint to {out_path}...")
        # Save weights
        save_file(self.model.state_dict(), str(out_path / "model.safetensors"))
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(str(out_path / "tokenizer"))
        # Save metadata
        meta = {
            "model_version": "laya-financial-us-v2",
            "base_model": self.config.base_model_hf_id,
            "temperature": self.temperature,
            "max_length": self.config.max_length,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(out_path / "model_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Checkpoint successfully saved to {out_path}.")
