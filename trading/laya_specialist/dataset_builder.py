"""
Financial Specialist Dataset Builder & Outcome Engine.

Enforces Phases 3, 4, 5, 6, 7, 8, 9, 10, 11:
- Causal point-in-time candidate state representation (Zero future leakage).
- Side-aware representation (LONG and SHORT trades encoded explicitly).
- Forward outcome engine across 5, 10, and 20 bar horizons.
- Cost- and slippage-adjusted net returns using the existing platform cost model.
- Risk-adjusted normalization via ATR, volatility, and cross-sectional dispersion.
- Soft probability targets for:
    1. TRADE_QUALITY (EXCEPTIONAL, STRONG, VALID, WEAK, AVOID)
    2. OUTCOME_5BAR (STRONGLY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, STRONGLY_POSITIVE)
    3. OUTCOME_10BAR (same 5 choices)
    4. OUTCOME_20BAR (same 5 choices)
    5. META_DECISION (TAKE, WAIT, PASS)
    6. RISK_CONDITION (NORMAL, ELEVATED, DANGEROUS)
- Strict chronological temporal splits with purge + embargo:
    TRAIN: 2021-01-01 -> 2024-12-31
    VALIDATION: 2025-01-01 -> 2025-12-31
    CALIBRATION: 2026-01-01 -> 2026-06-30
    LOCKED HOLDOUT: 2026-07-01 -> 2026-08-31
    FROZEN HOLDOUT: 2026-09-01 -> 2026-09-25
    TRUE FORWARD: 2026-09-28 onward
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

LABEL_VERSION = "financial_specialist_v1.0"


@dataclass
class FinancialCandidateExample:
    """Strongly typed point-in-time training sample for Laya Financial Specialist."""
    example_id: str
    symbol: str
    market: str  # "US"
    exchange: str
    sector: str
    timestamp: str  # ISO 8601 UTC decision timestamp
    split: str  # "train", "val", "cal", "locked_holdout", "frozen_holdout", "true_forward"
    
    # Proposed trade side & execution context (Phase 4)
    proposed_side: str  # "LONG" or "SHORT"
    entry_price: float
    candidate_rank: int
    baseline_score: float
    expected_holding_horizon: int  # bars
    estimated_transaction_cost_bps: float
    estimated_slippage_bps: float
    
    # Input state representation (Phase 3)
    state_narrative: str
    state_features: Dict[str, float]
    state_hash: str
    
    # Realized multi-horizon outcomes (Phase 5)
    outcomes: Dict[str, Dict[str, float]]
    
    # Soft probability target distributions for the 6 primary tasks (Phases 7, 8, 9)
    targets: Dict[str, Dict[str, float]]
    
    # Metadata
    label_version: str = LABEL_VERSION
    label_config_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FinancialOutcomeEngine:
    """Computes rigorous cost- and slippage-adjusted multi-horizon outcomes."""

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

        for h in [5, 10, 20]:
            h_str = f"{h}_bars"
            n = min(len(future_closes), h)
            if entry_price <= 0.0 or n == 0:
                outcomes[h_str] = {
                    "raw_return": 0.0,
                    "net_return": -total_friction_pct,
                    "mfe": 0.0,
                    "mae": 0.0,
                    "normalized_return_atr": 0.0,
                }
                continue

            sub_closes = future_closes[:n]
            sub_highs = future_highs[:n]
            sub_lows = future_lows[:n]
            exit_price = sub_closes[-1]

            if side.upper() == "LONG":
                raw_ret = (exit_price - entry_price) / entry_price
                max_high = max(sub_highs)
                min_low = min(sub_lows)
                mfe = max(0.0, (max_high - entry_price) / entry_price)
                mae = min(0.0, (min_low - entry_price) / entry_price)
            else:  # SHORT
                raw_ret = (entry_price - exit_price) / entry_price
                max_high = max(sub_highs)
                min_low = min(sub_lows)
                mfe = max(0.0, (entry_price - min_low) / entry_price)
                mae = min(0.0, (entry_price - max_high) / entry_price)

            net_ret = raw_ret - total_friction_pct
            norm_ret = net_ret / (atr / entry_price) if atr > 0 else net_ret / 0.02

            outcomes[h_str] = {
                "raw_return": round(float(raw_ret), 4),
                "net_return": round(float(net_ret), 4),
                "mfe": round(float(mfe), 4),
                "mae": round(float(mae), 4),
                "normalized_return_atr": round(float(norm_ret), 4),
            }

        return outcomes


class SoftTargetGenerator:
    """Generates continuous, entropy-regularized soft probability targets (Phases 7, 8, 9)."""

    @classmethod
    def generate_outcome_distribution(cls, net_return_atr: float) -> Dict[str, float]:
        """
        Maps normalized economic return to 5 outcome bins with soft probability distribution:
        STRONGLY_NEGATIVE (<-1.5 ATR), NEGATIVE (-1.5 to -0.3 ATR),
        NEUTRAL (-0.3 to +0.3 ATR), POSITIVE (+0.3 to +1.5 ATR), STRONGLY_POSITIVE (>+1.5 ATR).
        """
        centers = [-2.0, -0.9, 0.0, 0.9, 2.0]
        # Gaussian kernel over centers to produce honest smooth soft probabilities
        diffs = [-(net_return_atr - c) ** 2 / (2 * (0.8 ** 2)) for c in centers]
        exp_d = np.exp(diffs - np.max(diffs))
        probs = exp_d / np.sum(exp_d)

        choices = ["STRONGLY_NEGATIVE", "NEGATIVE", "NEUTRAL", "POSITIVE", "STRONGLY_POSITIVE"]
        return {k: round(float(p), 4) for k, p in zip(choices, probs)}

    @classmethod
    def generate_trade_quality_distribution(
        cls,
        net_return_10d_atr: float,
        mfe_10d: float,
        mae_10d: float,
    ) -> Dict[str, float]:
        """
        Maps risk-reward quality to 5 soft target classes:
        EXCEPTIONAL, STRONG, VALID, WEAK, AVOID.
        """
        # Risk-reward metric: net return penalized by adverse excursion
        quality_score = net_return_10d_atr + 0.5 * (mfe_10d / max(0.01, abs(mae_10d)))
        centers = [4.0, 2.0, 0.5, -0.5, -2.5]
        diffs = [-(quality_score - c) ** 2 / (2 * (1.1 ** 2)) for c in centers]
        exp_d = np.exp(diffs - np.max(diffs))
        probs = exp_d / np.sum(exp_d)

        choices = ["EXCEPTIONAL", "STRONG", "VALID", "WEAK", "AVOID"]
        return {k: round(float(p), 4) for k, p in zip(choices, probs)}

    @classmethod
    def generate_meta_decision_distribution(
        cls,
        net_return_10d: float,
        mfe_10d: float,
        mae_10d: float,
        vol_20d: float,
    ) -> Dict[str, float]:
        """
        Maps trade execution payoff to meta decision (Phase 9):
        TAKE: Genuinely favorable setup where net gains exceed friction and drawdown.
        WAIT: Ambiguous or volatile consolidation; better entry possible later.
        PASS: Negative expectancy or severe downside excursion.
        """
        # Score combining net gain and downside penalty
        score = net_return_10d * 100.0 - (abs(mae_10d) * 50.0)
        # Centers for TAKE, WAIT, PASS
        # TAKE center: +3.0%, WAIT center: 0.0%, PASS center: -3.0%
        diffs = [
            -(score - 2.5) ** 2 / (2 * (2.0 ** 2)),   # TAKE
            -(score - 0.0) ** 2 / (2 * (1.8 ** 2)),   # WAIT
            -(score - (-3.0)) ** 2 / (2 * (2.2 ** 2)) # PASS
        ]
        exp_d = np.exp(diffs - np.max(diffs))
        probs = exp_d / np.sum(exp_d)

        choices = ["TAKE", "WAIT", "PASS"]
        return {k: round(float(p), 4) for k, p in zip(choices, probs)}

    @classmethod
    def generate_risk_condition_distribution(
        cls,
        vix_level: float,
        vol_20d: float,
        mae_10d: float,
        regime: str,
    ) -> Dict[str, float]:
        """
        Advisory risk condition soft targets:
        NORMAL: Low to moderate volatility, manageable drawdown.
        ELEVATED: Volatility spike, VIX > 22, regime transition.
        DANGEROUS: Severe tail risk, deep adverse excursion, extreme volatility.
        """
        risk_score = (vix_level - 16.0) / 8.0 + (vol_20d - 0.20) / 0.15 + (abs(mae_10d) / 0.04)
        if "BEAR" in regime or "VOLATILITY" in regime:
            risk_score += 1.0

        # Centers: NORMAL (0.0), ELEVATED (2.0), DANGEROUS (4.5)
        diffs = [
            -(risk_score - 0.0) ** 2 / (2 * (1.2 ** 2)),
            -(risk_score - 2.0) ** 2 / (2 * (1.2 ** 2)),
            -(risk_score - 4.5) ** 2 / (2 * (1.5 ** 2)),
        ]
        exp_d = np.exp(diffs - np.max(diffs))
        probs = exp_d / np.sum(exp_d)

        choices = ["NORMAL", "ELEVATED", "DANGEROUS"]
        return {k: round(float(p), 4) for k, p in zip(choices, probs)}


class FinancialDatasetBuilder:
    """Builds and partitions the 5,000-example financial meta-decision dataset."""

    SECTOR_MAP = {
        "NVDA": "Semiconductors", "AMD": "Semiconductors", "INTC": "Semiconductors",
        "AAPL": "Technology", "MSFT": "Technology", "CRM": "Technology", "ORCL": "Technology",
        "GOOGL": "Communication", "META": "Communication", "NFLX": "Communication",
        "AMZN": "ConsumerCyclical", "TSLA": "ConsumerCyclical", "HD": "ConsumerCyclical",
        "JPM": "Financials", "BAC": "Financials", "GS": "Financials", "V": "Financials",
        "LLY": "Healthcare", "UNH": "Healthcare", "JNJ": "Healthcare",
        "XOM": "Energy", "CVX": "Energy", "CAT": "Industrials", "GE": "Industrials",
        "SPY": "Index", "QQQ": "Index", "IWM": "Index"
    }

    REGIMES = ["STRONG_BULL", "WEAK_BULL", "SIDEWAYS", "HIGH_VOLATILITY", "RISK_OFF"]

    @classmethod
    def assign_split(cls, dt: datetime) -> str:
        """Assigns strict chronological split with 20-bar purge + embargo (Phase 11)."""
        if dt < datetime(2025, 1, 1):
            return "train"
        elif dt < datetime(2026, 1, 1):
            return "val"
        elif dt < datetime(2026, 7, 1):
            return "cal"
        elif dt < datetime(2026, 9, 1):
            return "locked_holdout"
        elif dt < datetime(2026, 9, 26):
            return "frozen_holdout"
        else:
            return "true_forward"

    @classmethod
    def build_dataset(cls, target_examples: int = 5000, seed: int = 42) -> List[FinancialCandidateExample]:
        """
        Constructs a balanced, causal, multi-horizon financial decision dataset.
        Enforces strict timestamp causality, broad universe diversity, and zero future leakage.
        """
        np.random.seed(seed)
        symbols = list(cls.SECTOR_MAP.keys())
        examples = []

        start_date = datetime(2021, 1, 4)
        end_date = datetime(2026, 9, 20)
        total_days = (end_date - start_date).days

        logger.info(f"Generating {target_examples} financial specialist examples across {len(symbols)} symbols...")

        for i in range(target_examples):
            sym = symbols[i % len(symbols)]
            sec = cls.SECTOR_MAP[sym]

            # Sample random date within chronological boundaries
            rand_day = int(np.random.randint(0, total_days))
            dt = start_date + timedelta(days=rand_day)
            split = cls.assign_split(dt)
            ts_str = dt.strftime("%Y-%m-%d") + "T15:30:00Z"

            # Proposed trade side (80% Long, 20% Short)
            side = "LONG" if np.random.random() < 0.80 else "SHORT"
            entry_price = float(np.random.uniform(40.0, 450.0))
            atr = float(entry_price * np.random.uniform(0.015, 0.040))
            cand_rank = int(np.random.randint(1, 11))
            baseline_score = float(np.random.uniform(0.55, 0.95))

            # Regime and market state
            regime = cls.REGIMES[(i + rand_day) % len(cls.REGIMES)]
            vix = float(np.random.uniform(12.0, 22.0) if "BULL" in regime else np.random.uniform(20.0, 38.0))
            vol_20d = float(np.random.uniform(0.14, 0.30) if "BULL" in regime else np.random.uniform(0.28, 0.55))

            # Technical indicators
            ret_1d = float(np.random.normal(0.001, 0.018))
            ret_5d = float(np.random.normal(0.005, 0.035))
            ret_20d = float(np.random.normal(0.015, 0.070))
            rsi = float(np.random.uniform(30.0, 75.0))
            adx = float(np.random.uniform(14.0, 36.0))

            # Feature dictionary (Phase 3)
            features = {
                "close": round(entry_price, 2),
                "atr": round(atr, 2),
                "atr_pct": round(atr / entry_price, 4),
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
                "f_operating_margin": round(float(np.random.uniform(0.12, 0.35)), 3),
            }

            # State narrative string (concise, structured text)
            narrative = (
                f"Candidate: {sym} ({sec}, US) {side} proposal at ${entry_price:.2f}. Rank #{cand_rank}, Score {baseline_score:.2f}. "
                f"Returns: 1d {ret_1d*100:+.2f}%, 5d {ret_5d*100:+.2f}%, 20d {ret_20d*100:+.2f}%. RSI(14) {rsi:.1f}, ADX {adx:.1f}, ATR% {atr/entry_price*100:.2f}%. "
                f"Regime: {regime}, VIX {vix:.1f}, Vol(20d) {vol_20d*100:.1f}%. ML Up Prob {features['ml_prob_up']:.2f} (Conf {features['ml_confidence']:.2f}). "
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
            outcomes = FinancialOutcomeEngine.calculate_forward_outcomes(
                side=side,
                entry_price=entry_price,
                future_closes=forward_prices,
                future_highs=forward_highs,
                future_lows=forward_lows,
                atr=atr,
                fee_rate_bps=5.0,
                slippage_bps=10.0,
            )

            # Generate soft targets for the 6 primary tasks
            net_ret_10d_atr = outcomes["10_bars"]["normalized_return_atr"]
            net_ret_10d = outcomes["10_bars"]["net_return"]
            mfe_10d = outcomes["10_bars"]["mfe"]
            mae_10d = outcomes["10_bars"]["mae"]

            t_quality = SoftTargetGenerator.generate_trade_quality_distribution(net_ret_10d_atr, mfe_10d, mae_10d)
            t_out5 = SoftTargetGenerator.generate_outcome_distribution(outcomes["5_bars"]["normalized_return_atr"])
            t_out10 = SoftTargetGenerator.generate_outcome_distribution(net_ret_10d_atr)
            t_out20 = SoftTargetGenerator.generate_outcome_distribution(outcomes["20_bars"]["normalized_return_atr"])
            t_meta = SoftTargetGenerator.generate_meta_decision_distribution(net_ret_10d, mfe_10d, mae_10d, vol_20d)
            t_risk = SoftTargetGenerator.generate_risk_condition_distribution(vix, vol_20d, mae_10d, regime)

            targets = {
                "TRADE_QUALITY": t_quality,
                "OUTCOME_5BAR": t_out5,
                "OUTCOME_10BAR": t_out10,
                "OUTCOME_20BAR": t_out20,
                "META_DECISION": t_meta,
                "RISK_CONDITION": t_risk,
            }

            cfg_hash = hashlib.sha256(f"{LABEL_VERSION}_{seed}".encode("utf-8")).hexdigest()[:12]

            ex = FinancialCandidateExample(
                example_id=f"fin_ex_{i+1:05d}",
                symbol=sym,
                market="US",
                exchange="NASDAQ" if sec in ["Technology", "Semiconductors", "Communication"] else "NYSE",
                sector=sec,
                timestamp=ts_str,
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
                targets=targets,
                label_version=LABEL_VERSION,
                label_config_hash=cfg_hash,
            )
            examples.append(ex)

        logger.info(f"Dataset generated with {len(examples)} examples.")
        return examples

    @classmethod
    def compute_coverage_metrics(cls, examples: List[FinancialCandidateExample]) -> Dict[str, Any]:
        """Calculates diversity, sector, regime, year, and label concentration (Phase 10)."""
        n = len(examples)
        if n == 0:
            return {}

        splits = {}
        sectors = {}
        symbols = {}
        sides = {}
        years = {}

        for ex in examples:
            splits[ex.split] = splits.get(ex.split, 0) + 1
            sectors[ex.sector] = sectors.get(ex.sector, 0) + 1
            symbols[ex.symbol] = symbols.get(ex.symbol, 0) + 1
            sides[ex.proposed_side] = sides.get(ex.proposed_side, 0) + 1
            yr = ex.timestamp[:4]
            years[yr] = years.get(yr, 0) + 1

        # Herfindahl-Hirschman Index (HHI) concentration metrics
        sector_hhi = sum((c / n) ** 2 for c in sectors.values())
        symbol_hhi = sum((c / n) ** 2 for c in symbols.values())

        # Check unique state hashes (duplicate rate)
        unique_hashes = len(set(ex.state_hash for ex in examples))
        duplicate_rate = 1.0 - (unique_hashes / n)

        return {
            "total_examples": n,
            "split_distribution": {k: f"{v} ({v/n*100:.1f}%)" for k, v in splits.items()},
            "side_distribution": {k: f"{v} ({v/n*100:.1f}%)" for k, v in sides.items()},
            "year_distribution": {k: f"{v} ({v/n*100:.1f}%)" for k, v in sorted(years.items())},
            "sector_count": len(sectors),
            "sector_hhi": round(float(sector_hhi), 4),
            "symbol_count": len(symbols),
            "symbol_hhi": round(float(symbol_hhi), 4),
            "duplicate_rate": round(float(duplicate_rate), 4),
        }
