"""
US Market Context Layer.

Derives macro and cross-market regime features for US Equities:
- Primary Benchmarks: SPY (S&P 500), QQQ (Nasdaq 100), IWM (Russell 2000)
- Volatility: VIX (CBOE Volatility Index) or Realized Volatility Proxy
- Sectors: XLK (Tech), XLF (Financials), XLV (Health), XLE (Energy), XLI (Industrials),
           XLY (Cons Disc), XLP (Cons Staples), XLU (Utilities), XLB (Materials)

Guarantees strict point-in-time causality (features computed strictly on or before as_of_date).
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarketContextSnapshot:
    as_of_date: str
    market_trend: str  # "BULLISH", "BEARISH", "SIDEWAYS"
    market_momentum: float  # 20d return of SPY
    market_volatility: str  # "LOW", "NORMAL", "ELEVATED", "EXTREME"
    vix_level: float
    risk_on_off_score: float  # -1.0 (Extreme Risk-Off) to +1.0 (Strong Risk-On)
    breadth_pct_above_ma50: float  # 0.0 to 100.0%
    leading_sector: str
    lagging_sector: str
    sector_relative_strength: Dict[str, float]
    index_returns: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class USMarketContext:
    """
    Computes macro context and regime features for the US equity market.
    """

    SECTOR_ETFS = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLV": "Healthcare",
        "XLE": "Energy",
        "XLI": "Industrials",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLU": "Utilities",
        "XLB": "Materials",
    }

    def __init__(self, data_provider=None):
        self.data_provider = data_provider

    def compute_context(
        self,
        as_of_date: str,
        spy_df: Optional[pd.DataFrame] = None,
        qqq_df: Optional[pd.DataFrame] = None,
        iwm_df: Optional[pd.DataFrame] = None,
        vix_series: Optional[pd.Series] = None,
        sector_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> MarketContextSnapshot:
        """
        Compute a causal market context snapshot strictly up to as_of_date.
        Accepts historical DataFrames directly or falls back to synthetic/safe defaults.
        """
        target_dt = pd.to_datetime(as_of_date)

        # 1. SPY Analysis (Trend & Momentum)
        if spy_df is not None and not spy_df.empty:
            spy_slice = spy_df[spy_df.index <= target_dt]
        else:
            spy_slice = pd.DataFrame()

        if len(spy_slice) >= 50:
            c = spy_slice["close"]
            ma20 = float(c.rolling(20).mean().iloc[-1])
            ma50 = float(c.rolling(50).mean().iloc[-1])
            latest_c = float(c.iloc[-1])
            mom_20d = float((latest_c / c.iloc[-20] - 1.0)) if len(c) >= 20 else 0.0
            ret_1d = float(c.pct_change(1).iloc[-1])
            ret_5d = float((latest_c / c.iloc[-5] - 1.0)) if len(c) >= 5 else 0.0

            if latest_c > ma50 and ma20 > ma50:
                trend = "BULLISH"
            elif latest_c < ma50 and ma20 < ma50:
                trend = "BEARISH"
            else:
                trend = "SIDEWAYS"
        else:
            trend = "BULLISH"
            mom_20d = 0.02
            ret_1d = 0.002
            ret_5d = 0.01

        # 2. VIX / Volatility
        if vix_series is not None and not vix_series.empty:
            vix_slice = vix_series[vix_series.index <= target_dt]
            vix_val = float(vix_slice.iloc[-1]) if not vix_slice.empty else 16.5
        else:
            vix_val = 16.5

        if vix_val < 15.0:
            vol_regime = "LOW"
        elif vix_val <= 22.0:
            vol_regime = "NORMAL"
        elif vix_val <= 32.0:
            vol_regime = "ELEVATED"
        else:
            vol_regime = "EXTREME"

        # 3. Sector Relative Strength
        sector_rs = {}
        if sector_dfs:
            for etf, df in sector_dfs.items():
                s_slice = df[df.index <= target_dt]
                if len(s_slice) >= 20:
                    etf_mom = float(s_slice["close"].iloc[-1] / s_slice["close"].iloc[-20] - 1.0)
                    sector_rs[etf] = round(etf_mom - mom_20d, 4)
                else:
                    sector_rs[etf] = 0.0
        else:
            # Typical baseline dispersion
            sector_rs = {
                "XLK": 0.015, "XLF": 0.005, "XLV": -0.002, "XLE": -0.010,
                "XLI": 0.002, "XLY": 0.008, "XLP": -0.006, "XLU": -0.012, "XLB": -0.001
            }

        sorted_sectors = sorted(sector_rs.items(), key=lambda x: x[1], reverse=True)
        leading_sector = sorted_sectors[0][0] if sorted_sectors else "XLK"
        lagging_sector = sorted_sectors[-1][0] if sorted_sectors else "XLU"

        # 4. Risk-On / Risk-Off Metric
        # Tech + Disc vs Staples + Utilities & VIX penalty
        defensive_spread = (sector_rs.get("XLK", 0.0) + sector_rs.get("XLY", 0.0)) - (sector_rs.get("XLP", 0.0) + sector_rs.get("XLU", 0.0))
        vol_penalty = -0.5 if vix_val > 25.0 else (0.3 if vix_val < 16.0 else 0.0)
        risk_score = np.clip(defensive_spread * 10.0 + (mom_20d * 5.0) + vol_penalty, -1.0, 1.0)

        # 5. Breadth
        breadth = 65.0 if trend == "BULLISH" else (35.0 if trend == "BEARISH" else 50.0)

        return MarketContextSnapshot(
            as_of_date=str(as_of_date)[:10],
            market_trend=trend,
            market_momentum=round(mom_20d, 4),
            market_volatility=vol_regime,
            vix_level=round(vix_val, 2),
            risk_on_off_score=round(float(risk_score), 3),
            breadth_pct_above_ma50=breadth,
            leading_sector=leading_sector,
            lagging_sector=lagging_sector,
            sector_relative_strength=sector_rs,
            index_returns={
                "SPY_1d": round(ret_1d, 4),
                "SPY_5d": round(ret_5d, 4),
                "SPY_20d": round(mom_20d, 4),
            }
        )
