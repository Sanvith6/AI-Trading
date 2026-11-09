import sys
import importlib

packages = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("duckdb", "duckdb"),
    ("scipy", "scipy"),
    ("sklearn", "sklearn"),
    ("torch", "torch"),
    ("fastapi", "fastapi"),
    ("websockets", "websockets"),
    ("pyarrow", "pyarrow"),
    ("safetensors", "safetensors"),
    ("laya", "laya"),
]

print("Python Version:", sys.version.split()[0])
all_ok = True
for name, mod_path in packages:
    try:
        mod = importlib.import_module(mod_path)
        ver = getattr(mod, "__version__", "installed")
        print(f"  [PASS] {name:<20}: {ver}")
    except Exception as e:
        print(f"  [FAIL] {name:<20}: {e}")
        all_ok = False

# Native Architectural Adapters
try:
    from trading.brokers.alpaca_paper_broker import AlpacaPaperBrokerAdapter
    from trading.market.alpaca_provider import AlpacaMarketDataProvider
    print(f"  [PASS] {'alpaca_client':<20}: Native REST + WebSocket Streaming Adapter")
except Exception as e:
    print(f"  [FAIL] {'alpaca_client':<20}: {e}")
    all_ok = False

try:
    from trading.monitoring.metrics import TradingMetrics
    tm = TradingMetrics.get_instance()
    print(f"  [PASS] {'prometheus_client':<20}: Native TradingMetrics Exposition Engine")
except Exception as e:
    print(f"  [FAIL] {'prometheus_client':<20}: {e}")
    all_ok = False

try:
    from trading.storage.polars_engine import PolarsEngine
    status = "Accelerated (Polars)" if PolarsEngine.is_available() else "Operational (Pandas/PyArrow Fallback)"
    print(f"  [PASS] {'polars_engine':<20}: {status}")
except Exception as e:
    print(f"  [FAIL] {'polars_engine':<20}: {e}")
    all_ok = False

if all_ok:
    print("\nPYTHON_ENVIRONMENT: PASS")
else:
    print("\nPYTHON_ENVIRONMENT: FAIL")
