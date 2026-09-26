"""
Polars High-Performance Data Processing Engine.

Provides blazingly fast multi-threaded operations for:
- Batch market-data normalization
- Rolling indicator and feature vectorization
- Cross-sectional universe filtering
- High-throughput dataset transformations

Only utilized where measurable performance benefits over pandas exist.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    logger.warning("Polars is not available. Falling back to Pandas implementations.")


class PolarsEngine:
    """
    High-performance data transformation and feature pipeline using Polars.
    Preserves Pandas compatibility at external module boundaries.
    """

    @staticmethod
    def is_available() -> bool:
        return POLARS_AVAILABLE

    @staticmethod
    def compute_technical_features(df_pandas: pd.DataFrame) -> pd.DataFrame:
        """
        Compute high-throughput technical features (Returns, SMA, EMA, Volatility, ATR)
        using Polars columnar multi-threading.
        Returns a Pandas DataFrame for downstream compatibility.
        """
        if not POLARS_AVAILABLE or df_pandas.empty:
            return PolarsEngine._compute_features_pandas_fallback(df_pandas)

        try:
            # Convert pandas to Polars
            df_in = df_pandas.copy()
            date_col = "date"
            if not isinstance(df_in.index, pd.RangeIndex):
                df_in = df_in.reset_index()
                if "index" in df_in.columns:
                    date_col = "index"

            pl_df = pl.from_pandas(df_in)

            # Ensure lowercase column names
            cols = [c.lower() for c in pl_df.columns]
            pl_df.columns = cols

            # Multi-threaded Polars expression evaluation
            exprs = [
                (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret_1d"),
                (pl.col("close") / pl.col("close").shift(5) - 1.0).alias("ret_5d"),
                (pl.col("close") / pl.col("close").shift(20) - 1.0).alias("ret_20d"),
                pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
                pl.col("close").rolling_mean(window_size=50).alias("sma_50"),
                pl.col("close").rolling_std(window_size=20).alias("vol_20"),
            ]

            if "high" in pl_df.columns and "low" in pl_df.columns:
                tr = pl.max_horizontal(
                    pl.col("high") - pl.col("low"),
                    (pl.col("high") - pl.col("close").shift(1)).abs(),
                    (pl.col("low") - pl.col("close").shift(1)).abs()
                )
                exprs.append(tr.rolling_mean(window_size=14).alias("atr_14"))

            pl_result = pl_df.with_columns(exprs)
            df_out = pl_result.to_pandas()

            if date_col in df_out.columns:
                df_out = df_out.set_index(date_col)

            return df_out
        except Exception as e:
            logger.error(f"Polars feature computation failed: {e}. Falling back to Pandas.")
            return PolarsEngine._compute_features_pandas_fallback(df_pandas)

    @staticmethod
    def _compute_features_pandas_fallback(df: pd.DataFrame) -> pd.DataFrame:
        """Pandas fallback for environments without Polars."""
        df_out = df.copy()
        close = df_out["close"] if "close" in df_out.columns else df_out["Close"]
        df_out["ret_1d"] = close.pct_change(1)
        df_out["ret_5d"] = close.pct_change(5)
        df_out["ret_20d"] = close.pct_change(20)
        df_out["sma_20"] = close.rolling(20).mean()
        df_out["sma_50"] = close.rolling(50).mean()
        df_out["vol_20"] = close.rolling(20).std()
        return df_out

    @staticmethod
    def cross_sectional_rank(panel_df: pd.DataFrame, score_col: str = "ret_20d", date_col: str = "date") -> pd.DataFrame:
        """
        Cross-sectional percentile ranking across universe for each date.
        Uses Polars window function over (partition_by=date).
        """
        if not POLARS_AVAILABLE or panel_df.empty:
            df = panel_df.copy()
            df["rank_pct"] = df.groupby(date_col)[score_col].rank(pct=True)
            return df

        try:
            pl_df = pl.from_pandas(panel_df)
            pl_ranked = pl_df.with_columns(
                (pl.col(score_col).rank() / pl.count().over(date_col)).alias("rank_pct")
            )
            return pl_ranked.to_pandas()
        except Exception as e:
            logger.error(f"Polars cross-sectional rank failed: {e}")
            df = panel_df.copy()
            df["rank_pct"] = df.groupby(date_col)[score_col].rank(pct=True)
            return df

    @staticmethod
    def benchmark_pandas_vs_polars(n_rows: int = 250_000) -> Dict[str, Any]:
        """
        Benchmark feature computation speed between Pandas and Polars on synthetic dataset.
        Demonstrates measurable acceleration.
        """
        dates = pd.date_range("2020-01-01", periods=n_rows, freq="1min")
        synthetic_df = pd.DataFrame({
            "open": np.random.uniform(100, 200, n_rows),
            "high": np.random.uniform(200, 210, n_rows),
            "low": np.random.uniform(90, 100, n_rows),
            "close": np.random.uniform(100, 200, n_rows),
            "volume": np.random.randint(1000, 50000, n_rows),
        }, index=dates)

        # 1. Pandas Benchmark
        t0 = time.perf_counter()
        _ = PolarsEngine._compute_features_pandas_fallback(synthetic_df)
        pandas_time = time.perf_counter() - t0

        # 2. Polars Benchmark
        if POLARS_AVAILABLE:
            t1 = time.perf_counter()
            _ = PolarsEngine.compute_technical_features(synthetic_df)
            polars_time = time.perf_counter() - t1
            speedup = pandas_time / polars_time if polars_time > 0 else 1.0
        else:
            polars_time = None
            speedup = 1.0

        return {
            "rows_processed": n_rows,
            "pandas_seconds": round(pandas_time, 4),
            "polars_seconds": round(polars_time, 4) if polars_time else "N/A",
            "speedup_factor": round(speedup, 2),
            "recommendation": "Use Polars for batch ETL & features; keep Pandas for scikit-learn models."
        }
