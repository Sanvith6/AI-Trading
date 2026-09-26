"""
Financial Specialist Training Engine & Multi-Seed RLCD Fine-Tuning.

Enforces Phases 2, 11, 12, 13, 14, 15, 16, 17, 29:
- Evaluates Arm A (convaiinnovations/laya) vs Arm B (convaiinnovations/laya-typed-decisions).
- Builds 5,000 causal, multi-horizon financial decision training samples.
- Strictly chronological train/val/cal/holdout splits with 20-bar purge+embargo.
- Full fine-tuning on RTX 3050 GPU (micro_batch=1, grad_accum=16, grad_checkpointing, max_len=256).
- Proper scoring rules & soft cross-entropy target optimization.
- Multi-seed training (seeds 42, 123, 2026).
- Post-training temperature calibration on dedicated calibration split.
- Exports immutable checkpoint to `data_lake/laya_specialist/checkpoints/laya-financial-us-v1/`.
"""

import os
import sys
import json
import time
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add workspace to path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
from transformers import AutoTokenizer

from trading.laya_specialist.dataset_builder import FinancialDatasetBuilder, FinancialCandidateExample
from trading.laya_specialist.tasks import SpecialistTasks
from trading.laya_specialist.decision_policy import EconomicDecisionPolicy

# Laya core package imports
from laya.common import build_model, build_sequence, proper_reward, QTYPES

logger = logging = __import__("logging").getLogger("laya_specialist_trainer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class LayaFinancialTrainer:
    """Orchestrates end-to-end training and evaluation of the Financial Specialist."""

    BASE_MODELS = {
        "ARM_A_BASE": "convaiinnovations/laya",
        "ARM_B_TYPED": "convaiinnovations/laya-typed-decisions",
    }

    CHECKPOINT_CACHE_DIR = Path("/home/sanvith-cf/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851")

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        max_length: int = 256,
        learning_rate: float = 2e-5,
        grad_accum_steps: int = 16,
    ):
        self.output_dir = output_dir or (root_dir / "data_lake" / "laya_specialist" / "checkpoints" / "laya-financial-us-v1")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.grad_accum_steps = grad_accum_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_base_model_and_tokenizer(self, arm: str = "ARM_B_TYPED") -> Tuple[Any, Any, Dict[str, Any]]:
        """Loads model weights, tokenizer, and config from the local cache."""
        subfolder = "typed-decisions" if arm == "ARM_B_TYPED" else ""
        model_dir = self.CHECKPOINT_CACHE_DIR / subfolder if subfolder else self.CHECKPOINT_CACHE_DIR

        cfg_path = model_dir / "rl_agent_config.json"
        with open(cfg_path) as f:
            cfg = json.load(f)

        tok_dir = model_dir / "tokenizer"
        enc_dir = model_dir / "encoder"
        tok = AutoTokenizer.from_pretrained(str(tok_dir))

        cfg["gradient_checkpointing"] = True
        model = build_model(cfg, encoder_dir=str(enc_dir) if enc_dir.exists() else None, pretrained=False)

        from safetensors.torch import load_file
        weights_path = model_dir / "model.safetensors"
        weights = load_file(str(weights_path))
        model.load_state_dict(weights, strict=True)

        if hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable()

        model.to(self.device)
        return model, tok, cfg

    def prepare_batch_item(
        self,
        tok: Any,
        example: FinancialCandidateExample,
        task_name: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Encodes state narrative and question choices into tensor sequence."""
        q_dict = SpecialistTasks.get_task_dict(task_name)
        soft_targets = example.targets.get(task_name, {})
        if not soft_targets:
            return None

        # Build sequence using official Laya sequence builder
        input_ids, markers = build_sequence(tok, example.state_narrative, q_dict, max_len=self.max_length)
        if len(markers) != len(soft_targets):
            return None

        input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention_mask_t = torch.ones_like(input_ids_t, device=self.device)
        marker_pos_t = torch.tensor([markers], dtype=torch.long, device=self.device)
        marker_mask_t = torch.ones_like(marker_pos_t, dtype=torch.bool, device=self.device)
        qtype_t = torch.tensor([0], dtype=torch.long, device=self.device)

        # Soft target tensor
        target_vals = list(soft_targets.values())
        target_t = torch.tensor([target_vals], dtype=torch.float32, device=self.device)

        return {
            "input_ids": input_ids_t,
            "attention_mask": attention_mask_t,
            "marker_pos": marker_pos_t,
            "marker_mask": marker_mask_t,
            "qtype": qtype_t,
            "target": target_t,
        }

    def train_seed(
        self,
        seed: int,
        train_examples: List[FinancialCandidateExample],
        val_examples: List[FinancialCandidateExample],
        num_epochs: int = 1,
        max_train_steps: int = 400,
        arm: str = "ARM_B_TYPED",
    ) -> Dict[str, Any]:
        """Executes full-parameter fine-tuning for a single seed (Phase 15)."""
        logger.info(f"--- Training Specialist on {arm} | Seed={seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model, tok, cfg = self.load_base_model_and_tokenizer(arm)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)

        tasks = ["TRADE_QUALITY", "OUTCOME_10BAR", "META_DECISION", "RISK_CONDITION"]
        total_loss = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        peak_vram_recorded = 0.0

        t_start = time.perf_counter()

        for epoch in range(num_epochs):
            indices = np.random.permutation(len(train_examples))
            for idx in indices:
                ex = train_examples[idx]
                for task in tasks:
                    batch = self.prepare_batch_item(tok, ex, task)
                    if not batch:
                        continue

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                        logits, act_logits = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            marker_pos=batch["marker_pos"],
                            marker_mask=batch["marker_mask"],
                            qtype=batch["qtype"],
                        )
                        # Soft cross-entropy loss against entropy-regularized soft target distribution
                        log_probs = torch.log_softmax(logits, dim=-1)
                        loss = -(batch["target"] * log_probs).sum(dim=-1).mean() / self.grad_accum_steps

                    loss.backward()
                    total_loss += loss.item() * self.grad_accum_steps
                    steps += 1

                    if steps % self.grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        vram = torch.cuda.max_memory_allocated() / (1024**2)
                        if vram > peak_vram_recorded:
                            peak_vram_recorded = vram

                    if steps >= max_train_steps:
                        break
                if steps >= max_train_steps:
                    break

        duration_sec = time.perf_counter() - t_start
        avg_train_loss = total_loss / max(1, steps)

        # Validation evaluation
        val_loss, val_brier, val_ece = self.evaluate_dataset(model, tok, val_examples, tasks)
        logger.info(
            f"Seed {seed} Completed: Train Loss={avg_train_loss:.4f} | Val Loss={val_loss:.4f} | "
            f"Val Brier={val_brier:.4f} | Val ECE={val_ece:.4f} | Peak VRAM={peak_vram_recorded:.1f}MB | Time={duration_sec:.1f}s"
        )

        return {
            "seed": seed,
            "arm": arm,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_brier": round(val_brier, 4),
            "val_ece": round(val_ece, 4),
            "peak_vram_mb": round(peak_vram_recorded, 1),
            "duration_sec": round(duration_sec, 1),
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()} if seed == 42 else None,
            "tok": tok,
            "cfg": cfg,
        }

    def evaluate_dataset(
        self,
        model: Any,
        tok: Any,
        examples: List[FinancialCandidateExample],
        tasks: List[str],
    ) -> Tuple[float, float, float]:
        """Evaluates model loss, Brier score, and Expected Calibration Error (ECE)."""
        model.eval()
        total_loss = 0.0
        n_evals = 0
        brier_diffs = []
        ece_diffs = []

        with torch.no_grad():
            for ex in examples:
                for task in tasks:
                    batch = self.prepare_batch_item(tok, ex, task)
                    if not batch:
                        continue

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                        logits, _ = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            marker_pos=batch["marker_pos"],
                            marker_mask=batch["marker_mask"],
                            qtype=batch["qtype"],
                        )
                        log_probs = torch.log_softmax(logits, dim=-1)
                        probs = torch.softmax(logits, dim=-1)
                        loss = -(batch["target"] * log_probs).sum(dim=-1).mean()

                    total_loss += loss.item()
                    # Brier score calculation vs target distribution
                    p_np = probs.cpu().numpy()[0]
                    t_np = batch["target"].cpu().numpy()[0]
                    brier_diffs.append(np.mean((p_np - t_np) ** 2))
                    ece_diffs.append(np.abs(np.max(p_np) - np.max(t_np)))
                    n_evals += 1

        avg_loss = total_loss / max(1, n_evals)
        avg_brier = float(np.mean(brier_diffs)) if brier_diffs else 0.0
        avg_ece = float(np.mean(ece_diffs)) if ece_diffs else 0.0
        return avg_loss, avg_brier, avg_ece

    def fit_calibration_temperatures(
        self,
        model: Any,
        tok: Any,
        cal_examples: List[FinancialCandidateExample],
    ) -> Dict[str, float]:
        """
        Fits temperature scaling on the dedicated calibration split (Phase 17).
        Never fits on validation, locked holdout, or true forward.
        """
        logger.info("Fitting temperature scaling on dedicated calibration split...")
        tasks = ["TRADE_QUALITY", "OUTCOME_10BAR", "META_DECISION", "RISK_CONDITION"]
        cal_temps = {}

        for task in tasks:
            best_temp = 1.0
            lowest_brier = 1e9

            # Grid search temperature in [0.5, 2.5]
            for t_candidate in np.linspace(0.6, 2.2, 17):
                briers = []
                for ex in cal_examples:
                    batch = self.prepare_batch_item(tok, ex, task)
                    if not batch:
                        continue
                    with torch.no_grad():
                        logits, _ = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            marker_pos=batch["marker_pos"],
                            marker_mask=batch["marker_mask"],
                            qtype=batch["qtype"],
                        )
                        scaled_probs = torch.softmax(logits / t_candidate, dim=-1).cpu().numpy()[0]
                        target_np = batch["target"].cpu().numpy()[0]
                        briers.append(np.mean((scaled_probs - target_np) ** 2))

                mean_brier = np.mean(briers) if briers else 1.0
                if mean_brier < lowest_brier:
                    lowest_brier = mean_brier
                    best_temp = float(t_candidate)

            cal_temps[task] = round(best_temp, 3)
            logger.info(f"Fitted temperature for {task}: {best_temp:.3f} (Brier={lowest_brier:.4f})")

        return cal_temps

    def save_specialist_checkpoint(
        self,
        model_state: Dict[str, Any],
        tok: Any,
        cfg: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Path:
        """
        Saves the new immutable checkpoint: laya-financial-us-v1 (Phase 29).
        Never overwrites the pretrained checkpoint.
        """
        logger.info(f"Saving immutable checkpoint to {self.output_dir}...")
        from safetensors.torch import save_file

        # 1. Save model.safetensors
        weights_path = self.output_dir / "model.safetensors"
        save_file(model_state, str(weights_path))

        # 2. Save config & metadata
        cfg_path = self.output_dir / "rl_agent_config.json"
        cfg["model_name"] = "laya-financial-us-v1"
        cfg["fine_tuned_for"] = "financial_meta_decision"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        meta_path = self.output_dir / "training_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 3. Copy encoder and tokenizer directories
        subfolder = "typed-decisions"
        src_base = self.CHECKPOINT_CACHE_DIR / subfolder
        dst_tok = self.output_dir / "tokenizer"
        dst_enc = self.output_dir / "encoder"

        if (src_base / "tokenizer").exists() and not dst_tok.exists():
            shutil.copytree(src_base / "tokenizer", dst_tok)
        if (src_base / "encoder").exists() and not dst_enc.exists():
            shutil.copytree(src_base / "encoder", dst_enc)

        logger.info(f"Checkpoint successfully written to: {self.output_dir}")
        return self.output_dir
