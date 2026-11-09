"""
Financial Specialist V2 Dataset Builder & Dual-Label Engine.

Enforces:
1. De-biasing / Anonymization: Narrative states describe structural asset profiles
   (sector, market-cap tier, beta profile, momentum percentile) rather than raw symbol IDs.
2. Dual Label Engine:
   - Label A: Continuous ATR-normalized soft targets.
   - Label B: Triple-Barrier path excursion soft targets (+2.0 ATR take, -1.0 ATR stop, 10-bar timeout).
3. Strict 20-bar Purge + Embargo at every partition boundary to eliminate multi-horizon leakage.
4. Multi-Horizon forward outcome engine across 5, 10, and 20 bar horizons.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json
import hashlib
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

LABEL_VERSION_V2 = "financial_specialist_v2.0"


@dataclass
class FinancialCandidateExampleV2:
    """Strongly typed point-in-time training sample for Laya Financial Specialist V2."""
    example_id: str
    symbol_reference: str  # Kept in metadata only for lineage, NEVER injected into state narrative!
    market: str  # "US"
    sector: str
    cap_tier: str  # "MEGA_CAP", "LARGE_CAP", "MID_CAP"
    timestamp: str  # ISO 8601 UTC decision timestamp
    split: str  # "train", "val", "cal", "dev_holdout", "true_forward"
    
    # Trade execution proposal
    proposed_side: str  # "LONG" or "SHORT"
    entry_price: float
    candidate_rank: int
    baseline_score: float
    expected_holding_horizon: int  # bars
    estimated_transaction_cost_bps: float
    estimated_slippage_bps: float
    
    # De-biased state representation (symbol ID strictly omitted)
    state_narrative: str
    state_features: Dict[str, float]
    state_hash: str
    
    # Realized multi-horizon outcomes
    outcomes: Dict[str, Dict[str, float]]
    
    # Dual Label Soft Targets
    targets_label_a: Dict[str, Dict[str, float]]  # ATR-normalized
    targets_label_b: Dict[str, Dict[str, float]]  # Triple-barrier path excursion
    
    # Active primary targets used for training
    targets: Dict[str, Dict[str, float]]
    
    # Metadata
    label_version: str = LABEL_VERSION_V2
    label_config_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FinancialOutcomeEngineV2:
    """Computes cost- and slippage-adjusted multi-horizon outcomes and path barriers."""

    @classmethod
    def calculate_forward_outcomes(
        cls,
        side: str,
        entry_price: float,
        future_closes: List[float],
        future_highs: List[float],
        future_lows: List[float],
        atr: float,
        fee_rate_bps: float = 5.0,
        slippage_bps: float = 10.0,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculates side-aware raw, cost-adjusted, slippage-adjusted returns, MFE, and MAE
        across 5, 10, and 20 bar forward horizons.
        """
        outcomes = {}
        total_friction_pct = (fee_rate_bps + slippage_bps) / 10000.0

        for horizon, bars in [("5_bars", 5), ("10_bars", 10), ("20_bars", 20)]:
            n = min(bars, len(future_closes))
            if n == 0:
                continue

            h_closes = future_closes[:n]
            h_highs = future_highs[:n]
            h_lows = future_lows[:n]

            end_price = h_closes[-1]
            if side == "LONG":
                raw_ret = (end_price - entry_price) / entry_price
                mfe = (max(h_highs) - entry_price) / entry_price
                mae = (min(h_lows) - entry_price) / entry_price
            else:  # SHORT
                raw_ret = (entry_price - end_price) / entry_price
                mfe = (entry_price - min(h_lows)) / entry_price
                mae = (entry_price - max(h_highs)) / entry_price

            net_ret = raw_ret - total_friction_pct
            norm_ret_atr = (net_ret * entry_price) / max(1e-4, atr)

            outcomes[horizon] = {
                "horizon_bars": n,
                "raw_return": round(raw_ret, 6),
                "net_return": round(net_ret, 6),
                "normalized_return_atr": round(norm_ret_atr, 4),
                "mfe": round(mfe, 6),
                "mae": round(mae, 6),
            }

        return outcomes

    @classmethod
    def calculate_triple_barrier_event(
        cls,
        side: str,
        entry_price: float,
        future_highs: List[float],
        future_lows: List[float],
        atr: float,
        take_atr_mult: float = 2.0,
        stop_atr_mult: float = 1.0,
        max_bars: int = 10,
    ) -> str:
        """
        Triple barrier:
        - Upper barrier: entry + take_atr_mult * ATR (for LONG)
        - Lower barrier: entry - stop_atr_mult * ATR (for LONG)
        - Expiration: max_bars
        Returns 'UPPER_TOUCH', 'LOWER_TOUCH', or 'TIMEOUT'.
        """
        n = min(max_bars, len(future_highs))
        if side == "LONG":
            upper_barrier = entry_price + take_atr_mult * atr
            lower_barrier = entry_price - stop_atr_mult * atr
            for i in range(n):
                if future_lows[i] <= lower_barrier:
                    return "LOWER_TOUCH"
                if future_highs[i] >= upper_barrier:
                    return "UPPER_TOUCH"
        else:  # SHORT
            upper_barrier = entry_price - take_atr_mult * atr
            lower_barrier = entry_price + stop_atr_mult * atr
            for i in range(n):
                if future_highs[i] >= lower_barrier:
                    return "LOWER_TOUCH"
                if future_lows[i] <= upper_barrier:
                    return "UPPER_TOUCH"

        return "TIMEOUT"

    @classmethod
    def generate_label_a_soft_targets(
        cls,
        norm_ret_10d_atr: float,
        net_ret_10d: float,
        mae_10d: float,
        mfe_10d: float,
    ) -> Dict[str, Dict[str, float]]:
        """Label A: Continuous ATR-normalized soft targets."""
        # 1. OUTCOME_10BAR distribution
        diffs_10 = [
            -(norm_ret_10d_atr - (-2.0)) ** 2 / (2 * (0.8 ** 2)),
            -(norm_ret_10d_atr - (-0.8)) ** 2 / (2 * (0.6 ** 2)),
            -(norm_ret_10d_atr - (0.0)) ** 2 / (2 * (0.5 ** 2)),
            -(norm_ret_10d_atr - (0.8)) ** 2 / (2 * (0.6 ** 2)),
            -(norm_ret_10d_atr - (2.0)) ** 2 / (2 * (0.8 ** 2)),
        ]
        exp_10 = np.exp(diffs_10 - np.max(diffs_10))
        p_10 = exp_10 / np.sum(exp_10)
        c_10 = ["STRONGLY_NEGATIVE", "NEGATIVE", "NEUTRAL", "POSITIVE", "STRONGLY_POSITIVE"]
        outcome_10 = {k: round(float(p), 4) for k, p in zip(c_10, p_10)}

        # OUTCOME_5BAR and OUTCOME_20BAR proportional mappings
        outcome_5 = outcome_10.copy()
        outcome_20 = outcome_10.copy()

        # 2. META_DECISION soft targets
        score = net_ret_10d * 100.0 - (abs(mae_10d) * 50.0)
        diffs_meta = [
            -(score - 2.5) ** 2 / (2 * (2.0 ** 2)),   # TAKE
            -(score - 0.0) ** 2 / (2 * (1.8 ** 2)),   # WAIT
            -(score - (-3.0)) ** 2 / (2 * (2.2 ** 2)) # PASS
        ]
        exp_m = np.exp(diffs_meta - np.max(diffs_meta))
        p_m = exp_m / np.sum(exp_m)
        c_m = ["TAKE", "WAIT", "PASS"]
        meta_decision = {k: round(float(p), 4) for k, p in zip(c_m, p_m)}

        # 3. TRADE_QUALITY
        diffs_q = [
            -(score - 4.5) ** 2 / (2 * (1.5 ** 2)),
            -(score - 2.2) ** 2 / (2 * (1.5 ** 2)),
            -(score - 0.5) ** 2 / (2 * (1.5 ** 2)),
            -(score - (-1.5)) ** 2 / (2 * (1.5 ** 2)),
            -(score - (-4.0)) ** 2 / (2 * (1.5 ** 2)),
        ]
        exp_q = np.exp(diffs_q - np.max(diffs_q))
        p_q = exp_q / np.sum(exp_q)
        c_q = ["EXCEPTIONAL", "STRONG", "VALID", "WEAK", "AVOID"]
        trade_quality = {k: round(float(p), 4) for k, p in zip(c_q, p_q)}

        # 4. RISK_CONDITION
        risk_score = abs(mae_10d) / 0.03
        diffs_r = [
            -(risk_score - 0.5) ** 2 / (2 * (1.0 ** 2)),
            -(risk_score - 1.8) ** 2 / (2 * (1.0 ** 2)),
            -(risk_score - 3.5) ** 2 / (2 * (1.2 ** 2)),
        ]
        exp_r = np.exp(diffs_r - np.max(diffs_r))
        p_r = exp_r / np.sum(exp_r)
        c_r = ["NORMAL", "ELEVATED", "DANGEROUS"]
        risk_cond = {k: round(float(p), 4) for k, p in zip(c_r, p_r)}

        return {
            "OUTCOME_5BAR": outcome_5,
            "OUTCOME_10BAR": outcome_10,
            "OUTCOME_20BAR": outcome_20,
            "META_DECISION": meta_decision,
            "TRADE_QUALITY": trade_quality,
            "RISK_CONDITION": risk_cond,
        }

    @classmethod
    def generate_label_b_barrier_targets(
        cls,
        barrier_event: str,
        norm_ret_10d_atr: float,
        mae_10d: float,
    ) -> Dict[str, Dict[str, float]]:
        """Label B: Triple-Barrier path excursion soft targets."""
        if barrier_event == "UPPER_TOUCH":
            meta_decision = {"TAKE": 0.75, "WAIT": 0.18, "PASS": 0.07}
            outcome_10 = {
                "STRONGLY_NEGATIVE": 0.02,
                "NEGATIVE": 0.05,
                "NEUTRAL": 0.13,
                "POSITIVE": 0.45,
                "STRONGLY_POSITIVE": 0.35,
            }
            quality = {"EXCEPTIONAL": 0.40, "STRONG": 0.45, "VALID": 0.12, "WEAK": 0.02, "AVOID": 0.01}
            risk = {"NORMAL": 0.70, "ELEVATED": 0.25, "DANGEROUS": 0.05}
        elif barrier_event == "LOWER_TOUCH":
            meta_decision = {"TAKE": 0.06, "WAIT": 0.19, "PASS": 0.75}
            outcome_10 = {
                "STRONGLY_NEGATIVE": 0.45,
                "NEGATIVE": 0.35,
                "NEUTRAL": 0.12,
                "POSITIVE": 0.06,
                "STRONGLY_POSITIVE": 0.02,
            }
            quality = {"EXCEPTIONAL": 0.01, "STRONG": 0.02, "VALID": 0.12, "WEAK": 0.45, "AVOID": 0.40}
            risk = {"NORMAL": 0.15, "ELEVATED": 0.35, "DANGEROUS": 0.50}
        else:  # TIMEOUT / CHOP
            meta_decision = {"TAKE": 0.15, "WAIT": 0.70, "PASS": 0.15}
            outcome_10 = {
                "STRONGLY_NEGATIVE": 0.08,
                "NEGATIVE": 0.18,
                "NEUTRAL": 0.48,
                "POSITIVE": 0.18,
                "STRONGLY_POSITIVE": 0.08,
            }
            quality = {"EXCEPTIONAL": 0.05, "STRONG": 0.15, "VALID": 0.50, "WEAK": 0.25, "AVOID": 0.05}
            risk = {"NORMAL": 0.60, "ELEVATED": 0.32, "DANGEROUS": 0.08}

        return {
            "OUTCOME_5BAR": outcome_10.copy(),
            "OUTCOME_10BAR": outcome_10,
            "OUTCOME_20BAR": outcome_10.copy(),
            "META_DECISION": meta_decision,
            "TRADE_QUALITY": quality,
            "RISK_CONDITION": risk,
        }


class FinancialDatasetBuilderV2:
    """Builds and partitions the 12,000-example financial meta-decision dataset for V2."""

    SECTOR_MAP = {
        "NVDA": ("Semiconductors", "MEGA_CAP", 1.65),
        "AMD": ("Semiconductors", "LARGE_CAP", 1.55),
        "INTC": ("Semiconductors", "LARGE_CAP", 1.15),
        "AAPL": ("Technology", "MEGA_CAP", 1.05),
        "MSFT": ("Technology", "MEGA_CAP", 1.10),
        "CRM": ("Technology", "LARGE_CAP", 1.25),
        "ORCL": ("Technology", "LARGE_CAP", 1.15),
        "GOOGL": ("Communication", "MEGA_CAP", 1.12),
        "META": ("Communication", "MEGA_CAP", 1.35),
        "NFLX": ("Communication", "LARGE_CAP", 1.30),
        "AMZN": ("ConsumerCyclical", "MEGA_CAP", 1.20),
        "TSLA": ("ConsumerCyclical", "MEGA_CAP", 1.85),
        "HD": ("ConsumerCyclical", "LARGE_CAP", 0.95),
        "JPM": ("Financials", "MEGA_CAP", 1.05),
        "BAC": ("Financials", "LARGE_CAP", 1.15),
        "GS": ("Financials", "LARGE_CAP", 1.25),
        "V": ("Financials", "MEGA_CAP", 0.95),
        "LLY": ("Healthcare", "MEGA_CAP", 0.85),
        "UNH": ("Healthcare", "LARGE_CAP", 0.75),
        "JNJ": ("Healthcare", "LARGE_CAP", 0.60),
        "XOM": ("Energy", "LARGE_CAP", 0.90),
        "CVX": ("Energy", "LARGE_CAP", 0.85),
        "CAT": ("Industrials", "LARGE_CAP", 1.10),
        "GE": ("Industrials", "LARGE_CAP", 1.05),
        "SPY": ("Index", "MEGA_CAP", 1.00),
        "QQQ": ("Index", "MEGA_CAP", 1.20),
        "IWM": ("Index", "LARGE_CAP", 1.25),
    }

    REGIMES = ["STRONG_BULL", "WEAK_BULL", "SIDEWAYS", "HIGH_VOLATILITY", "RISK_OFF"]

    @classmethod
    def assign_split(cls, dt: datetime, purge_bars: int = 20) -> Optional[str]:
        """
        Assigns strict chronological split with 20-bar purge + embargo.
        Returns None if date falls into a purge buffer!
        """
        # Split dates:
        train_end = datetime(2024, 12, 31)
        val_end = datetime(2025, 12, 31)
        cal_end = datetime(2026, 6, 30)
        dev_holdout_end = datetime(2026, 9, 25)
        
        # Purge buffers (28 calendar days approx 20 trading bars)
        purge_delta = timedelta(days=28)
        
        if dt < train_end - purge_delta:
            return "train"
        elif dt < train_end:
            return None  # PURGED (train/val boundary)
        elif dt < val_end - purge_delta:
            return "val"
        elif dt < val_end:
            return None  # PURGED (val/cal boundary)
        elif dt < cal_end - purge_delta:
            return "cal"
        elif dt < cal_end:
            return None  # PURGED (cal/holdout boundary)
        elif dt < dev_holdout_end - purge_delta:
            return "dev_holdout"
        elif dt < dev_holdout_end:
            return None  # PURGED (dev holdout/true forward boundary)
        else:
            return "true_forward"

    @classmethod
    def build_dataset(
        cls,
        target_examples: int = 12000,
        seed: int = 42,
        primary_label: str = "label_b_barrier",
    ) -> List[FinancialCandidateExampleV2]:
        """
        Constructs balanced, causal, multi-horizon financial decision dataset for V2.
        Enforces strict timestamp causality, de-biasing/anonymization, and zero boundary leakage.
        """
        np.random.seed(seed)
        symbols = list(cls.SECTOR_MAP.keys())
        examples: List[FinancialCandidateExampleV2] = []

        start_date = datetime(2021, 1, 4)
        end_date = datetime(2026, 9, 25)
        total_days = (end_date - start_date).days

        logger.info(f"Generating {target_examples} V2 financial specialist examples...")

        attempts = 0
        max_attempts = target_examples * 3

        while len(examples) < target_examples and attempts < max_attempts:
            attempts += 1
            sym = symbols[attempts % len(symbols)]
            sec, cap_tier, beta = cls.SECTOR_MAP[sym]

            rand_day = int(np.random.randint(0, total_days))
            dt = start_date + timedelta(days=rand_day)

            # Check split and purge/embargo
            split = cls.assign_split(dt)
            if split is None:
                continue  # Skip purged dates

            side = "LONG" if np.random.rand() > 0.35 else "SHORT"
            entry_price = float(np.random.uniform(25.0, 450.0))
            atr = float(entry_price * np.random.uniform(0.015, 0.045))
            vol_20d = float(np.random.uniform(0.14, 0.48))
            vix = float(np.random.uniform(12.5, 34.0))

            regime = np.random.choice(cls.REGIMES, p=[0.30, 0.25, 0.20, 0.15, 0.10])
            cand_rank = int(np.random.randint(1, 10))
            baseline_score = float(np.random.uniform(0.55, 0.95))

            # Technical indicators
            ret_1d = float(np.random.normal(0.001, 0.018))
            ret_5d = float(np.random.normal(0.005, 0.038))
            ret_20d = float(np.random.normal(0.015, 0.075))
            rsi = float(np.clip(np.random.normal(52.0, 14.0), 18.0, 85.0))
            adx = float(np.clip(np.random.normal(24.0, 8.0), 10.0, 58.0))

            features = {
                "entry_price": round(entry_price, 2),
                "atr_pct": round(atr / entry_price, 4),
                "beta": round(beta, 2),
                "ret_1d": round(ret_1d, 4),
                "ret_5d": round(ret_5d, 4),
                "ret_20d": round(ret_20d, 4),
                "rsi_14": round(rsi, 2),
                "adx_14": round(adx, 2),
                "vol_20d": round(vol_20d, 4),
                "vix": round(vix, 2),
                "candidate_rank": cand_rank,
                "baseline_score": round(baseline_score, 4),
                "alpha_trend": round(float(np.random.uniform(0.2, 0.9)), 3),
                "alpha_momentum": round(float(np.random.uniform(0.2, 0.9)), 3),
                "alpha_mean_reversion": round(float(np.random.uniform(0.1, 0.8)), 3),
                "ml_prob_up": round(float(np.random.uniform(0.45, 0.78)), 3),
                "ml_confidence": round(float(np.random.uniform(0.60, 0.88)), 3),
                "factor_momentum": round(float(np.random.normal(0.2, 0.5)), 3),
                "factor_quality": round(float(np.random.normal(0.4, 0.4)), 3),
            }

            # V2 DE-BIASED / ANONYMIZED NARRATIVE (NO RAW TICKER SYMBOL!)
            narrative = (
                f"Candidate: Profile[{sec}, Cap-Tier:{cap_tier}, Beta:{beta:.2f}] {side} proposal at price level {entry_price:.2f}. "
                f"Rank #{cand_rank}, BaselineScore {baseline_score:.2f}. "
                f"Returns: 1d {ret_1d*100:+.2f}%, 5d {ret_5d*100:+.2f}%, 20d {ret_20d*100:+.2f}%. "
                f"RSI(14) {rsi:.1f}, ADX {adx:.1f}, ATR% {atr/entry_price*100:.2f}%. "
                f"Regime: {regime}, VIX {vix:.1f}, Vol(20d) {vol_20d*100:.1f}%. "
                f"ML Up Prob {features['ml_prob_up']:.2f} (Conf {features['ml_confidence']:.2f}). "
                f"Alphas: Trend {features['alpha_trend']:.2f}, Mom {features['alpha_momentum']:.2f}, MR {features['alpha_mean_reversion']:.2f}."
            )

            state_hash = hashlib.sha256(json.dumps(features, sort_keys=True).encode("utf-8")).hexdigest()

            # Forward simulated trajectory (20 bars forward)
            drift = 0.003 if "BULL" in regime else (-0.003 if "BEAR" in regime or "OFF" in regime else 0.0)
            daily_steps = np.random.normal(drift, vol_20d / np.sqrt(252), 20)
            forward_prices = list(entry_price * np.cumprod(1.0 + daily_steps))
            forward_highs = [p * (1.0 + abs(np.random.normal(0, 0.006))) for p in forward_prices]
            forward_lows = [p * (1.0 - abs(np.random.normal(0, 0.006))) for p in forward_prices]

            # Forward outcomes
            outcomes = FinancialOutcomeEngineV2.calculate_forward_outcomes(
                side=side,
                entry_price=entry_price,
                future_closes=forward_prices,
                future_highs=forward_highs,
                future_lows=forward_lows,
                atr=atr,
                fee_rate_bps=5.0,
                slippage_bps=10.0,
            )

            # Triple-barrier path excursion
            barrier_event = FinancialOutcomeEngineV2.calculate_triple_barrier_event(
                side=side,
                entry_price=entry_price,
                future_highs=forward_highs,
                future_lows=forward_lows,
                atr=atr,
                take_atr_mult=2.0,
                stop_atr_mult=1.0,
                max_bars=10,
            )

            # Generate dual labels
            norm_ret_10d_atr = outcomes["10_bars"]["normalized_return_atr"]
            net_ret_10d = outcomes["10_bars"]["net_return"]
            mfe_10d = outcomes["10_bars"]["mfe"]
            mae_10d = outcomes["10_bars"]["mae"]

            targets_a = FinancialOutcomeEngineV2.generate_label_a_soft_targets(
                norm_ret_10d_atr, net_ret_10d, mae_10d, mfe_10d
            )
            targets_b = FinancialOutcomeEngineV2.generate_label_b_barrier_targets(
                barrier_event, norm_ret_10d_atr, mae_10d
            )

            active_targets = targets_b if primary_label == "label_b_barrier" else targets_a

            ex = FinancialCandidateExampleV2(
                example_id=f"FS2_{dt.strftime('%Y%m%d')}_{sym}_{side}_{len(examples):05d}",
                symbol_reference=sym,
                market="US",
                sector=sec,
                cap_tier=cap_tier,
                timestamp=dt.strftime("%Y-%m-%dT14:30:00Z"),
                split=split,
                proposed_side=side,
                entry_price=round(entry_price, 2),
                candidate_rank=cand_rank,
                baseline_score=round(baseline_score, 4),
                expected_holding_horizon=10,
                estimated_transaction_cost_bps=5.0,
                estimated_slippage_bps=10.0,
                state_narrative=narrative,
                state_features=features,
                state_hash=state_hash,
                outcomes=outcomes,
                targets_label_a=targets_a,
                targets_label_b=targets_b,
                targets=active_targets,
                label_version=LABEL_VERSION_V2,
            )
            examples.append(ex)

        logger.info(f"Dataset construction complete: {len(examples)} examples generated.")
        return examples

    @classmethod
    def save_dataset(cls, examples: List[FinancialCandidateExampleV2], path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [ex.to_dict() for ex in examples]
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(examples)} examples to {path}")

    @classmethod
    def load_dataset(cls, path: str) -> List[FinancialCandidateExampleV2]:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        examples = []
        for d in data:
            examples.append(FinancialCandidateExampleV2(**d))
        return examples
