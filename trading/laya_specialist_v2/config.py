"""
Configuration for Laya Financial Specialist V2.

Hardware target: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
Backbone: ModernBERT-large (421M params) + Typed Decision Head
Optimizer: Adafactor (O(1) memory optimizer state ~25 MB vs AdamW 3.37 GB)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class LayaV2Config:
    # Model & Weights
    model_name: str = "laya-financial-us-v2"
    base_model_hf_id: str = "convaiinnovations/laya-typed-decisions"
    base_snapshot_wsl: str = (
        "/home/sanvith-cf/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/typed-decisions"
    )
    checkpoint_dir: str = "data_lake/laya_specialist/checkpoints/laya-financial-us-v2"
    dataset_path: str = "data_lake/laya_specialist/financial_specialist_v2_dataset.json"
    
    # Sequence & Hardware Constraints
    max_length: int = 256
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    gradient_checkpointing: bool = True
    use_adafactor: bool = True
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    
    # Dataset & Partitioning
    num_dataset_examples: int = 12000
    purge_embargo_bars: int = 20
    split_dates: Dict[str, str] = field(default_factory=lambda: {
        "train_end": "2024-12-31",
        "val_end": "2025-12-31",
        "cal_end": "2026-06-30",
        "dev_holdout_end": "2026-09-25",
        "true_forward_start": "2026-09-28",
    })
    
    # Labeling Strategies
    primary_label_type: str = "label_b_barrier"  # "label_a_atr", "label_b_barrier", or "hybrid"
    barrier_take_atr_mult: float = 2.0
    barrier_stop_atr_mult: float = 1.0
    holding_horizon_bars: int = 10
    
    # Economic Decision Policy
    min_expected_utility: float = 0.005  # +0.50% expected return threshold
    max_downside_prob: float = 0.38     # Max 38% loss probability allowed
    min_risk_asymmetry: float = 1.35    # Upside / Downside probability ratio
    max_entropy_for_take: float = 0.88  # Shannon entropy threshold for explicit WAIT
    default_friction_bps: float = 15.0  # 15 bps round-trip friction
    
    # Multi-Seed Training
    seeds: List[int] = field(default_factory=lambda: [42, 123, 2026])
    primary_seed: int = 42
