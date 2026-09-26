# Laya Financial Specialist V2.1 Pipeline Report & Model Champion Decision

**Model Target**: `laya-financial-us-v2.1` (`data_lake/laya_specialist/checkpoints/laya-financial-us-v2.1/`)  
**Base Model**: `convaiinnovations/laya-typed-decisions` (ModernBERT-large 421M + Typed Decision Head)  
**Optimizer**: Adafactor (O(1) memory state ~25MB)  
**Training Device**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)  
**Date**: September 26, 2026  
**Auditor**: Antigravity Autonomous Quantitative Auditing System  

---

## 1. Context & Motivation for V2.1

During the adversarial audit of Laya Financial Specialist V2, several methodological and evaluation weaknesses were identified:
1. **Evaluation Annualization Bug**: V2's evaluator annualized 10-bar returns over 19 active days ($\sqrt{252}$) rather than all 43 calendar trading days and non-overlapping cycles.
2. **Noise-Based Ablations in V2 Scratch Scripts**: Feature ablations were simulated via noise injection rather than true prompt masking.
3. **Selectivity Cliff**: V2 had a steep transition between $\theta=0.27$ (28 trades) and $\theta=0.30$ (3 trades).

To test whether an expanded, re-calibrated model could achieve superior or more stable performance under the corrected evaluation metrics, we built **Laya Financial Specialist V2.1**.

---

## 2. V2.1 Dataset & Training Architecture

### Dataset Expansion (15,000 Candidates)
- **Total Point-in-Time Candidates**: 15,000 across 2021-01 to 2026-09.
- **Strict 20-Bar Purge & Embargo**: 28 calendar days purged at all 4 partition boundaries.
- **Partitions**:
  - `train`: 10,908 candidates (2021–2024)
  - `val`: 2,501 candidates (2025)
  - `cal`: 1,179 candidates (2026-H1)
  - `dev_holdout`: 412 candidates (2026-Q3)
- **Dual Label Soft Targets**: Triple-barrier path excursion (Label B, primary) + continuous ATR-normalized returns (Label A).

### Multi-Seed Training Details
- **Primary Seed 42**: 350 steps, gradient accumulation 32 (effective batch size 32 = 11,200 candidate exposures). Duration: 322.4 seconds (~5.4 minutes). Final train loss: 1.4361, validation loss: 1.0949. Peak VRAM: 3,994.5 MB.
- **Pilot Convergence Verification (Seeds 123 & 2026)**:
  - Seed 123: 60 steps, duration: 60.1s, val loss: 1.1411.
  - Seed 2026: 60 steps, duration: 93.6s, val loss: 1.1426.

---

## 3. V2.1 Evaluation & Threshold Sweep

### Temperature Over-Smoothing Artifact
On the `cal` partition, temperature grid search converged to the upper bound ($T^* = 3.000$, NLL: 1.0994). This divided all logits by 3.0, collapsing output probability dispersion toward uniform ($1/3$). At a fixed threshold $\theta = 0.27$, the model approved 100% of candidates (412 out of 412).

### Threshold Sweep at Un-Flattened Temperature ($T = 1.0$)
To test whether V2.1 possessed latent selective edge, we evaluated forward inference at $T = 1.0$ across a selectivity threshold sweep on the 412 holdout candidates:

| Theta ($\theta$) | Approved Trades | Selectivity | Win Rate | Profit Factor | Mean Net Return | Daily Portfolio Sharpe | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.25** | 412 | 100.0% | 51.5% | 1.23 | +0.60% | 5.26 | Baseline universe (all trades) |
| **0.27** | 412 | 100.0% | 51.5% | 1.23 | +0.60% | 5.25 | All trades approved |
| **0.30** | 407 | 98.8% | 50.9% | 1.21 | +0.57% | 4.90 | Marginal unselective filter |
| **0.33** | 102 | 24.8% | 48.0% | 1.03 | +0.10% | -0.09 | Edge collapses below 50% WR |
| **0.35** | 8 | 1.9% | 37.5% | **0.90** | **-0.35%** | **-0.21** | **Losing system** |
| **0.38** | 0 | 0.0% | 0.0% | 0.00 | 0.00% | 0.00 | Full pass |

---

## 4. Head-to-Head Comparison: V2 vs V2.1

| Dimension | Laya Specialist V2 (Champion) | Laya Specialist V2.1 (Challenger) | Winner |
| :--- | :---: | :---: | :---: |
| **Dev Holdout Candidates** | 371 | 412 | — |
| **Approved Trades** | 28 | 8 (at $\theta=0.35$) / 412 (at $\theta=0.27$) | **V2** |
| **Operating Win Rate** | **71.43%** | 37.50% / 51.46% | **V2 (by +19.9%)** |
| **Profit Factor** | **3.01** | 0.90 / 1.23 | **V2 (by +1.78)** |
| **Mean Net Return** | **+3.555%** | -0.35% / +0.60% | **V2 (by +2.95%)** |
| **Holding-Cycle Sharpe** | **2.09** | -0.21 / 0.39 | **V2** |
| **Permutation Null $p$-value** | **$p = 0.036$ (Significant)** | $p = 1.000$ (Not Significant) | **V2** |
| **Matched Control $p$-value** | **$p = 0.032$ (Significant)** | Not Significant | **V2** |
| **Downside Defense** | **95.4% Toxic Rejection** | 0.0% to 5.0% Rejection | **V2** |

---

## 5. Champion Determination & Checkpoint Status

### Why V2.1 Failed to Surpass V2
1. **Probability Dispersion Collapse**: Training with 350 steps on a 15,000-candidate dataset caused the classification head to converge toward safer, more entropy-regularized predictions. While validation cross-entropy remained good (1.0949), the model lost the sharp tail-discrimination capability that enabled V2 to isolate winning candidates.
2. **Loss of High-Conviction Thresholding**: At tight thresholds ($\theta \ge 0.35$), V2.1's approved trades had a lower win rate (37.5%) than the background candidate universe (51.5%).

### Formal Champion Decision
- **`laya-financial-us-v2` is confirmed and retained as the authoritative champion model.**
- **`laya-financial-us-v2.1` is archived as an experimental artifact and frozen** at `data_lake/laya_specialist/checkpoints/laya-financial-us-v2.1/`.
- Neither V1 nor V2 were modified or overwritten.
- **Both checkpoints remain immutable.**
- **Laya V2 remains strictly in SHADOW ONLY mode.** Zero live order execution authorized.
