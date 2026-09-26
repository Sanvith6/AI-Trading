"""
DuckDB Analytical Query Engine for Data Lake.

Provides fast, in-process, zero-copy SQL analytics over columnar Parquet files
and embedded tabular structures. Allows cross-sectional screening, multi-asset aggregation,
and research slice generation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.warning("duckdb is not installed; DuckDBStore will operate in limited fallback mode.")


class DuckDBStore:
    """
    Manages DuckDB embedded analytics database over Parquet files.
    Enables blazing fast SQL queries across all market data and features.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None, data_lake_dir: Optional[Union[str, Path]] = None):
        self.data_lake_dir = Path(data_lake_dir) if data_lake_dir else Path(__file__).parent.parent.parent / "data_lake"
        self.db_path = Path(db_path) if db_path else self.data_lake_dir / "research.duckdb"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._con = None
        if DUCKDB_AVAILABLE:
            try:
                self._con = duckdb.connect(database=str(self.db_path), read_only=False)
                logger.info(f"Connected to DuckDB database at: {self.db_path}")
            except Exception as e:
                logger.warning(f"Could not open persistent DuckDB, falling back to in-memory: {e}")
                self._con = duckdb.connect(database=":memory:")

    def query(self, sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """Execute arbitrary SQL query and return result as pandas DataFrame."""
        if not DUCKDB_AVAILABLE or self._con is None:
            raise RuntimeError("DuckDB is not available in the environment.")

        try:
            if params:
                return self._con.execute(sql, params).df()
            else:
                return self._con.execute(sql).df()
        except Exception as e:
            logger.error(f"DuckDB query failed: {sql} | Error: {e}")
            raise

    def register_parquet_view(self, view_name: str, glob_pattern: Union[str, Path]) -> None:
        """
        Create or replace a DuckDB SQL view pointing directly to one or more Parquet files.
        Zero-copy, streaming execution.
        """
        if not DUCKDB_AVAILABLE or self._con is None:
            return

        pattern_str = str(glob_pattern).replace("\\", "/")
        sql = f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{pattern_str}', filename=true);"
        self._con.execute(sql)
        logger.info(f"Registered DuckDB view '{view_name}' -> {pattern_str}")

    def create_market_views(self) -> None:
        """Automatically create views for US and IN daily market data."""
        if not DUCKDB_AVAILABLE or self._con is None:
            return

        us_pattern = self.data_lake_dir / "market_data" / "US" / "1d" / "*.parquet"
        in_pattern = self.data_lake_dir / "market_data" / "IN" / "1d" / "*.parquet"

        if any(self.data_lake_dir.glob("market_data/US/1d/*.parquet")):
            self.register_parquet_view("us_bars_1d", us_pattern)
        if any(self.data_lake_dir.glob("market_data/IN/1d/*.parquet")):
            self.register_parquet_view("in_bars_1d", in_pattern)

    def get_cross_sectional_momentum(self, as_of_date: str, lookback_days: int = 20, market: str = "US") -> pd.DataFrame:
        """
        Compute cross-sectional returns and momentum ranks across all available symbols
        on a given date using analytical SQL.
        """
        if not DUCKDB_AVAILABLE or self._con is None:
            return pd.DataFrame()

        view_name = f"{market.lower()}_bars_1d"
        sql = f"""
        WITH ranked_bars AS (
            SELECT 
                regexp_extract(filename, '([^/\\\\]+)\\.parquet$', 1) as symbol,
                date,
                close,
                ROW_NUMBER() OVER (PARTITION BY filename ORDER BY date DESC) as rn
            FROM {view_name}
            WHERE date <= ?
        ),
        endpoints AS (
            SELECT 
                symbol,
                MAX(CASE WHEN rn = 1 THEN close END) as latest_close,
                MAX(CASE WHEN rn = ? THEN close END) as lookback_close
            FROM ranked_bars
            WHERE rn <= ?
            GROUP BY symbol
        )
        SELECT 
            symbol,
            latest_close,
            lookback_close,
            (latest_close - lookback_close) / NULLIF(lookback_close, 0) as momentum_ret,
            PERCENT_RANK() OVER (ORDER BY (latest_close - lookback_close) / NULLIF(lookback_close, 0)) as momentum_percentile
        FROM endpoints
        WHERE latest_close IS NOT NULL AND lookback_close IS NOT NULL
        ORDER BY momentum_ret DESC;
        """
        return self.query(sql, [as_of_date, lookback_days, lookback_days])

    def close(self) -> None:
        """Close connection to database."""
        if self._con:
            try:
                self._con.close()
            except Exception:
                pass
            self._con = None
