"""
Data Cache Layer — Local disk & memory caching with TTL for reproducibility and resilience.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.getenv("DATA_CACHE_DIR", "trading/data_cache"))


class DataCache:
    """Manages disk and memory caching of market data."""

    def __init__(self, cache_dir: Optional[Path] = None, default_ttl_seconds: int = 300):
        self.cache_dir = cache_dir or CACHE_DIR
        self.default_ttl = default_ttl_seconds
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached quote if not expired."""
        clean_sym = symbol.strip().upper()
        cache_key = f"quote_{clean_sym}"
        now = time.time()

        # Check in-memory first
        if cache_key in self._memory_cache:
            item = self._memory_cache[cache_key]
            if now - item["timestamp"] < self.default_ttl:
                return item["data"]

        # Check disk
        file_path = self.cache_dir / f"{self._hash_key(cache_key)}.json"
        if file_path.exists():
            try:
                content = json.loads(file_path.read_text(encoding="utf-8"))
                if now - content.get("timestamp", 0) < self.default_ttl:
                    data = content.get("data")
                    self._memory_cache[cache_key] = {"timestamp": content["timestamp"], "data": data}
                    return data
            except Exception as e:
                logger.debug("Error reading quote cache for %s: %s", clean_sym, e)

        return None

    def set_quote(self, symbol: str, quote: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Cache quote in memory and disk."""
        clean_sym = symbol.strip().upper()
        cache_key = f"quote_{clean_sym}"
        now = time.time()
        payload = {"timestamp": now, "symbol": clean_sym, "data": quote}
        self._memory_cache[cache_key] = payload

        file_path = self.cache_dir / f"{self._hash_key(cache_key)}.json"
        try:
            file_path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        except Exception as e:
            logger.debug("Error writing quote cache for %s: %s", clean_sym, e)

    def get_ohlcv(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        """Retrieve cached OHLCV dataframe if valid."""
        clean_sym = symbol.strip().upper()
        cache_key = f"ohlcv_{clean_sym}_{interval}_{days}"
        now = time.time()
        ttl = self.default_ttl * 4  # OHLCV has longer TTL (20 min)

        file_path = self.cache_dir / f"{self._hash_key(cache_key)}.parquet"
        meta_path = self.cache_dir / f"{self._hash_key(cache_key)}_meta.json"

        if file_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if now - meta.get("timestamp", 0) < ttl:
                    df = pd.read_parquet(file_path)
                    return df
            except Exception:
                # Fallback to csv if parquet engine not available
                pass

        csv_path = self.cache_dir / f"{self._hash_key(cache_key)}.csv"
        if csv_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if now - meta.get("timestamp", 0) < ttl:
                    df = pd.read_csv(csv_path)
                    return df
            except Exception as e:
                logger.debug("Error reading OHLCV CSV cache for %s: %s", clean_sym, e)

        return None

    def set_ohlcv(self, symbol: str, interval: str, days: int, df: pd.DataFrame) -> None:
        """Cache OHLCV dataframe to disk."""
        if df is None or df.empty:
            return
        clean_sym = symbol.strip().upper()
        cache_key = f"ohlcv_{clean_sym}_{interval}_{days}"
        now = time.time()

        meta_path = self.cache_dir / f"{self._hash_key(cache_key)}_meta.json"
        csv_path = self.cache_dir / f"{self._hash_key(cache_key)}.csv"
        try:
            meta_path.write_text(json.dumps({"timestamp": now, "symbol": clean_sym, "rows": len(df)}), encoding="utf-8")
            df.to_csv(csv_path, index=False)
        except Exception as e:
            logger.debug("Error caching OHLCV for %s: %s", clean_sym, e)

    def clear(self) -> None:
        """Clear cache files."""
        self._memory_cache.clear()
        for p in self.cache_dir.glob("*"):
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass
