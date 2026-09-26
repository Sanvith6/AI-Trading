"""
Dynamic Market Universe

The platform must dynamically discover eligible instruments.

Supports:
- equities
- ETFs
- indices
- sectors
- future asset-class expansion

Universe selection must be configurable.

Filters should include:
- instrument validity
- exchange
- price
- liquidity
- average traded value
- historical availability
- data completeness
- volume
- volatility
- corporate-action consistency
- stale-price detection
- suspicious data
- trading halts where detectable
- excessive illiquidity
- penny-stock suppression
- configurable exclusion lists

Support: 50, 100, 250, 500, 1000+ without changing strategy logic.
Use dynamic ranking rather than hard-coded watchlists.

Maintain universe_version for reproducibility.
"""

from trading.universe.universe_manager import UniverseManager

__all__ = ["UniverseManager", "get_universe", "get_eligible_instruments"]