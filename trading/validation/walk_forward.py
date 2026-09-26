"""
Walk-Forward Validation Engine — Institutional time-series split with purge and embargo.

Never randomly shuffles time series.
Enforces strict chronological sequence:
TRAIN -> PURGE -> VALIDATE -> EMBARGO -> TEST

Supports:
1. Rolling Walk-Forward (fixed lookback window sliding forward)
2. Expanding Walk-Forward (anchored start, expanding training window)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Generator
from dataclasses import dataclass


@dataclass
class WalkForwardSplit:
    fold: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


class WalkForwardValidator:
    """Generates strictly chronological train/validate/test folds with purge and embargo."""

    def __init__(
        self,
        train_bars: int = 150,
        val_bars: int = 40,
        test_bars: int = 40,
        purge_bars: int = 5,    # Discards bars with forward-label overlap
        embargo_bars: int = 5,  # Prevents leakage from post-trade autocorrelation
        mode: str = "rolling",  # "rolling" or "expanding"
    ):
        self.train_bars = train_bars
        self.val_bars = val_bars
        self.test_bars = test_bars
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
        self.mode = mode.lower()

    def generate_splits(self, df: pd.DataFrame) -> List[WalkForwardSplit]:
        """
        Generate time-series split intervals.
        """
        n_bars = len(df)
        splits: List[WalkForwardSplit] = []

        total_fold_span = self.train_bars + self.purge_bars + self.val_bars + self.embargo_bars + self.test_bars
        if n_bars < total_fold_span:
            # Fallback single chronological partition
            p_train = int(n_bars * 0.6)
            p_val = int(n_bars * 0.8)
            return [
                WalkForwardSplit(
                    fold=0,
                    train_start=str(df["date"].iloc[0]),
                    train_end=str(df["date"].iloc[p_train - 1]),
                    val_start=str(df["date"].iloc[p_train]),
                    val_end=str(df["date"].iloc[p_val - 1]),
                    test_start=str(df["date"].iloc[p_val]),
                    test_end=str(df["date"].iloc[-1]),
                    train_indices=np.arange(0, p_train),
                    val_indices=np.arange(p_train, p_val),
                    test_indices=np.arange(p_val, n_bars),
                )
            ]

        step_size = self.test_bars
        current_start = 0
        fold_idx = 0

        while True:
            train_start = 0 if self.mode == "expanding" else current_start
            train_end = current_start + self.train_bars

            # Purge after train
            val_start = train_end + self.purge_bars
            val_end = val_start + self.val_bars

            # Embargo after validation
            test_start = val_end + self.embargo_bars
            test_end = test_start + self.test_bars

            if test_end > n_bars:
                break

            splits.append(
                WalkForwardSplit(
                    fold=fold_idx,
                    train_start=str(df["date"].iloc[train_start]),
                    train_end=str(df["date"].iloc[train_end - 1]),
                    val_start=str(df["date"].iloc[val_start]),
                    val_end=str(df["date"].iloc[val_end - 1]),
                    test_start=str(df["date"].iloc[test_start]),
                    test_end=str(df["date"].iloc[test_end - 1]),
                    train_indices=np.arange(train_start, train_end),
                    val_indices=np.arange(val_start, val_end),
                    test_indices=np.arange(test_start, test_end),
                )
            )

            current_start += step_size
            fold_idx += 1

        return splits
