"""
Corporate Action Engine & Multi-Tier Price Representation.

Distinguishes between:
1. RAW PRICE: The exact unadjusted traded price on execution date (used for slippage, order sizing, limit checks).
2. SPLIT-ADJUSTED PRICE: Adjusted strictly for splits & reverse splits (used for chart volume & price continuity).
3. TOTAL RETURN ADJUSTED PRICE: Adjusted for splits and cash/special dividends (used for alpha features & Sharpe calculations).

Prevents artificial return spikes (e.g., 80% false crash on 5:1 stock split) and lookahead contamination.
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PriceRepresentation(str, Enum):
    RAW = "RAW"
    SPLIT_ADJUSTED = "SPLIT_ADJUSTED"
    TOTAL_RETURN = "TOTAL_RETURN"


@dataclass
class CorporateAction:
    symbol: str
    action_type: str  # "SPLIT", "REVERSE_SPLIT", "DIVIDEND", "SPECIAL_DIVIDEND", "SYMBOL_CHANGE"
    effective_date: str  # "YYYY-MM-DD"
    ratio: float = 1.0  # e.g., 4.0 for 4-for-1 split; 0.1 for 1-for-10 reverse split
    cash_amount: float = 0.0  # dividend cash per share
    new_symbol: Optional[str] = None
    notes: str = ""


class CorporateActionEngine:
    """
    Manages corporate action adjustments and provides clean price representations.
    """

    # Verified historical corporate actions for testing and baseline normalization
    CURATED_ACTIONS: List[CorporateAction] = [
        # US Splits
        CorporateAction(symbol="NVDA", action_type="SPLIT", effective_date="2024-06-10", ratio=10.0, notes="10-for-1 forward split"),
        CorporateAction(symbol="TSLA", action_type="SPLIT", effective_date="2022-08-25", ratio=3.0, notes="3-for-1 forward split"),
        CorporateAction(symbol="AAPL", action_type="SPLIT", effective_date="2020-08-31", ratio=4.0, notes="4-for-1 forward split"),
        CorporateAction(symbol="AMZN", action_type="SPLIT", effective_date="2022-06-06", ratio=20.0, notes="20-for-1 forward split"),
        CorporateAction(symbol="GOOGL", action_type="SPLIT", effective_date="2022-07-18", ratio=20.0, notes="20-for-1 forward split"),
        # Symbol Changes
        CorporateAction(symbol="FB", action_type="SYMBOL_CHANGE", effective_date="2022-06-09", new_symbol="META", notes="Ticker changed to META"),
    ]

    def __init__(self, actions: Optional[List[CorporateAction]] = None):
        self.actions = actions if actions is not None else list(self.CURATED_ACTIONS)

    def register_action(self, action: CorporateAction) -> None:
        """Register a new corporate action."""
        self.actions.append(action)

    def get_actions_for_symbol(self, symbol: str) -> List[CorporateAction]:
        """Fetch all registered corporate actions for a symbol sorted chronologically."""
        clean_sym = symbol.upper().replace(".NS", "").replace(".BO", "")
        matching = [a for a in self.actions if a.symbol.upper() == clean_sym]
        return sorted(matching, key=lambda x: x.effective_date)

    def adjust_series(
        self,
        df: pd.DataFrame,
        symbol: str,
        target_repr: PriceRepresentation = PriceRepresentation.SPLIT_ADJUSTED,
    ) -> pd.DataFrame:
        """
        Adjust an OHLCV dataframe according to the requested PriceRepresentation.
        Assumes df contains ['open', 'high', 'low', 'close', 'volume'].
        """
        if df.empty:
            return df

        df_out = df.copy()
        actions = self.get_actions_for_symbol(symbol)
        if not actions or target_repr == PriceRepresentation.RAW:
            return df_out

        # Ensure datetime index
        if not isinstance(df_out.index, pd.DatetimeIndex) and "date" in df_out.columns:
            df_out["date"] = pd.to_datetime(df_out["date"])
            df_out = df_out.set_index("date")

        split_factor = pd.Series(1.0, index=df_out.index)
        dividend_factor = pd.Series(1.0, index=df_out.index)

        for act in actions:
            act_date = pd.to_datetime(act.effective_date)
            # Apply backward split adjustment
            if act.action_type in ("SPLIT", "REVERSE_SPLIT") and act.ratio > 0:
                mask = df_out.index < act_date
                split_factor.loc[mask] *= (1.0 / act.ratio)

            # Apply backward dividend adjustment for Total Return
            if target_repr == PriceRepresentation.TOTAL_RETURN and act.action_type in ("DIVIDEND", "SPECIAL_DIVIDEND") and act.cash_amount > 0:
                post_action_bars = df_out[df_out.index >= act_date]
                if not post_action_bars.empty:
                    close_on_eff = post_action_bars.iloc[0]["close"]
                    div_ratio = 1.0 - (act.cash_amount / max(close_on_eff, 0.01))
                    mask = df_out.index < act_date
                    dividend_factor.loc[mask] *= max(div_ratio, 0.01)

        total_price_multiplier = split_factor
        if target_repr == PriceRepresentation.TOTAL_RETURN:
            total_price_multiplier = split_factor * dividend_factor

        for col in ["open", "high", "low", "close"]:
            if col in df_out.columns:
                df_out[col] = df_out[col] * total_price_multiplier

        # Volume adjusts inversely to price on splits
        if "volume" in df_out.columns and target_repr in (PriceRepresentation.SPLIT_ADJUSTED, PriceRepresentation.TOTAL_RETURN):
            df_out["volume"] = df_out["volume"] / split_factor

        return df_out

    def detect_unadjusted_split_anomaly(self, df: pd.DataFrame, threshold_drop: float = 0.40) -> List[Dict[str, Any]]:
        """
        Scans a price series for abnormal overnight single-day price drops that strongly resemble
        unadjusted corporate actions (e.g. 50%, 66%, 75%, 90% drops with sudden volume surge).
        """
        if df.empty or "close" not in df.columns:
            return []

        anomalies = []
        ret = df["close"].pct_change()
        for dt, r in ret.items():
            if r <= -threshold_drop:
                anomalies.append({
                    "date": str(dt),
                    "return": round(float(r), 4),
                    "suspected_action": "UNADJUSTED_STOCK_SPLIT",
                    "severity": "CRITICAL_LOOKAHEAD_OR_ADJUSTMENT_RISK"
                })
        return anomalies
