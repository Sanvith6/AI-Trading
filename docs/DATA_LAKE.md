# Research Data Lake — Storage Architecture & Operational Guide

## Overview
The Research Data Lake replaces flat, unstructured files with an institutional columnar Lakehouse architecture:
`Raw Market Data -> PyArrow Parquet -> DuckDB Views -> Dataset Registry`.

## Storage Organization
```
data_lake/
├── market_data/
│   ├── US/
│   │   └── 1d/
│   │       ├── AAPL.parquet
│   │       ├── AAPL.meta.json
│   │       └── NVDA.parquet
│   └── IN/
│       └── 1d/
├── features/
│   ├── momentum_v1.parquet
│   └── momentum_v1.meta.json
├── dataset_registry.json
├── research.duckdb
└── events.json
```

## Technologies & Separation of Responsibilities
| Component | Technology | Responsibility |
|---|---|---|
| **Columnar Storage** | `pyarrow` (Parquet) | High-compression, partition-aware binary persistence |
| **Analytical Querying** | `duckdb` | Zero-copy SQL aggregations across multi-symbol Parquet files |
| **High-Volume Preprocessing** | `polars` | Multi-threaded rolling indicators and cross-sectional ranks |
| **ML Model Compatibility** | `pandas` | Scikit-learn / LightGBM model ingestion |
| **Dataset Provenance** | `DatasetRegistry` | Cryptographic SHA256 integrity verification |

## Zero-Copy DuckDB Usage
```python
from trading.storage import DuckDBStore

db = DuckDBStore()
db.create_market_views()
# Direct analytical SQL query across all US daily bars
df_mom = db.get_cross_sectional_momentum(as_of_date="2024-03-01", lookback_days=20)
```
