"""
Parquet Data Lake Storage Engine.

Provides high-performance, columnar read/write access for market bars, quotes,
and computed feature matrices using PyArrow and Parquet format.
Organizes data partitioned by market, frequency, and symbol.
"""

import os
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DATA_LAKE_DIR = Path(__file__).parent.parent.parent / "data_lake"


class ParquetStore:
    """
    Manages persistent Parquet dataset files for research and backtesting.
    Enforces standardized schemas, date ranges, and metadata headers.
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_DATA_LAKE_DIR
        self.market_data_dir = self.base_dir / "market_data"
        self.features_dir = self.base_dir / "features"
        self.datasets_dir = self.base_dir / "datasets"

        for d in [self.market_data_dir, self.features_dir, self.datasets_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _get_market_path(self, market: str, symbol: str, timeframe: str = "1d") -> Path:
        """Construct partitioned directory path: market_data/{market}/{timeframe}/{symbol}.parquet"""
        market_clean = market.upper()
        dir_path = self.market_data_dir / market_clean / timeframe
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{symbol.upper()}.parquet"

    def write_bars(
        self,
        symbol: str,
        df: pd.DataFrame,
        market: str = "US",
        timeframe: str = "1d",
        compression: str = "snappy",
    ) -> Dict[str, Any]:
        """
        Write or overwrite OHLCV market bars for a symbol in Parquet format.
        Ensures datetime index or date column is strictly formatted.
        """
        if df.empty:
            logger.warning(f"Empty dataframe provided for {symbol}; skipping write.")
            return {"status": "SKIPPED", "symbol": symbol, "rows": 0}

        df_out = df.copy()
        if not isinstance(df_out.index, pd.DatetimeIndex) and "date" in df_out.columns:
            df_out["date"] = pd.to_datetime(df_out["date"])
            df_out = df_out.set_index("date")
        elif isinstance(df_out.index, pd.DatetimeIndex):
            df_out.index.name = "date"

        df_out = df_out.sort_index()

        target_file = self._get_market_path(market, symbol, timeframe)
        # Write to parquet
        df_out.to_parquet(target_file, engine="pyarrow", compression=compression)

        file_bytes = target_file.stat().st_size
        sha256 = self._compute_file_hash(target_file)

        metadata = {
            "symbol": symbol.upper(),
            "market": market.upper(),
            "timeframe": timeframe,
            "rows": len(df_out),
            "start_date": str(df_out.index.min()),
            "end_date": str(df_out.index.max()),
            "file_path": str(target_file),
            "size_bytes": file_bytes,
            "sha256": sha256,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }

        # Write sidecar metadata json
        meta_file = target_file.with_suffix(".meta.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Wrote {len(df_out)} bars to Parquet for {symbol} ({market}): {target_file}")
        return metadata

    def read_bars(
        self,
        symbol: str,
        market: str = "US",
        timeframe: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Read historical bars for a symbol from Parquet with optional date range and column filtering.
        """
        target_file = self._get_market_path(market, symbol, timeframe)
        if not target_file.exists():
            logger.debug(f"Parquet file not found: {target_file}")
            return pd.DataFrame()

        try:
            df = pd.read_parquet(target_file, columns=columns, engine="pyarrow")
            if not isinstance(df.index, pd.DatetimeIndex) and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]

            return df
        except Exception as e:
            logger.error(f"Error reading Parquet for {symbol}: {e}")
            return pd.DataFrame()

    def list_symbols(self, market: str = "US", timeframe: str = "1d") -> List[str]:
        """List all symbols stored in the Parquet store for a given market and timeframe."""
        dir_path = self.market_data_dir / market.upper() / timeframe
        if not dir_path.exists():
            return []
        return sorted([f.stem for f in dir_path.glob("*.parquet")])

    def write_features(
        self,
        dataset_name: str,
        df: pd.DataFrame,
        feature_version: str = "v1",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Store an engineered feature matrix as Parquet."""
        target_file = self.features_dir / f"{dataset_name}_{feature_version}.parquet"
        df.to_parquet(target_file, engine="pyarrow", compression="snappy")

        meta = {
            "dataset_name": dataset_name,
            "feature_version": feature_version,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "written_at": datetime.now(timezone.utc).isoformat(),
            "sha256": self._compute_file_hash(target_file),
            **(metadata or {}),
        }
        with open(target_file.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return target_file

    def read_features(self, dataset_name: str, feature_version: str = "v1") -> pd.DataFrame:
        """Read engineered feature matrix."""
        target_file = self.features_dir / f"{dataset_name}_{feature_version}.parquet"
        if not target_file.exists():
            return pd.DataFrame()
        return pd.read_parquet(target_file, engine="pyarrow")

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Calculate SHA256 hash of a file for integrity tracking."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
