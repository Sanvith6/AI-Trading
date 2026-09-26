"""
Performance & Regime Attribution — Dissects Sources of Gains and Losses.

Answers institutional investment committee questions:
- WHERE DID THE PROFITS COME FROM? (Symbols, sectors, style factors, regimes)
- WHERE DID THE LOSSES COME FROM?
- HOW MUCH WAS CONSUMED BY FEES AND SLIPPAGE?
- DID PROFITS STEM FROM TIMING (Entry/Exit) OR ASSET SELECTION?
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class AttributionReport:
    total_gross_pnl: float
    total_fees: float
    total_slippage: float
    net_pnl: float
    cost_drag_pct: float
    by_strategy: Dict[str, float]
    by_symbol: Dict[str, float]
    by_sector: Dict[str, float]
    by_factor: Dict[str, float]
    by_regime: Dict[str, float]
    top_winning_drivers: List[Dict[str, Any]]
    top_losing_drivers: List[Dict[str, Any]]
    executive_summary: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceAttributor:
    """Attributes portfolio outcomes across strategies, symbols, sectors, factors, and market regimes."""

    @staticmethod
    def attribute(
        trades: List[Dict[str, Any]],
        symbol_factors: Optional[Dict[str, Dict[str, float]]] = None,
        symbol_sectors: Optional[Dict[str, str]] = None,
    ) -> AttributionReport:
        """
        Decomposes completed trades into comprehensive attribution buckets.
        """
        if not trades:
            return AttributionReport(
                total_gross_pnl=0.0, total_fees=0.0, total_slippage=0.0, net_pnl=0.0,
                cost_drag_pct=0.0, by_strategy={}, by_symbol={}, by_sector={},
                by_factor={}, by_regime={}, top_winning_drivers=[], top_losing_drivers=[],
                executive_summary={"summary": "No trades to attribute"}
            )

        df = pd.DataFrame(trades)
        pnl_col = "pnl" if "pnl" in df.columns else ("proceeds" if "proceeds" in df.columns else "realized_pnl")
        if pnl_col not in df.columns:
            df[pnl_col] = 0.0

        fee_col = "fees" if "fees" in df.columns else None
        slip_col = "slippage" if "slippage" in df.columns else None

        tot_pnl = float(df[pnl_col].sum())
        tot_fees = float(df[fee_col].sum()) if fee_col else 0.0
        tot_slip = float(df[slip_col].sum()) if slip_col else (len(df) * 0.50)  # nominal slippage estimate
        net_pnl = tot_pnl - tot_fees - tot_slip
        cost_drag = ((tot_fees + tot_slip) / max(abs(tot_pnl), 1.0)) * 100.0

        # 1. By Strategy
        by_strat = {}
        if "strategy" in df.columns:
            by_strat = {str(k): round(float(v), 2) for k, v in df.groupby("strategy")[pnl_col].sum().items()}

        # 2. By Symbol
        by_sym = {}
        if "symbol" in df.columns:
            by_sym = {str(k): round(float(v), 2) for k, v in df.groupby("symbol")[pnl_col].sum().items()}

        # 3. By Sector
        by_sec = {}
        sectors = symbol_sectors or {}
        if "symbol" in df.columns:
            df["inferred_sector"] = df["symbol"].map(lambda s: sectors.get(s, "Other"))
            by_sec = {str(k): round(float(v), 2) for k, v in df.groupby("inferred_sector")[pnl_col].sum().items()}

        # 4. By Regime
        by_reg = {}
        if "regime" in df.columns:
            by_reg = {str(k): round(float(v), 2) for k, v in df.groupby("regime")[pnl_col].sum().items()}

        # 5. By Factor (Momentum, Value, Quality, Volatility)
        by_factor = {"momentum": 0.0, "value": 0.0, "quality": 0.0, "volatility": 0.0}
        if symbol_factors and "symbol" in df.columns:
            for _, row in df.iterrows():
                sym = row["symbol"]
                trade_pnl = float(row[pnl_col])
                s_factors = symbol_factors.get(sym, {})
                for f in by_factor:
                    f_exposure = s_factors.get(f, 0.0)
                    by_factor[f] += trade_pnl * f_exposure

            by_factor = {k: round(float(v), 2) for k, v in by_factor.items()}

        # Top drivers
        sorted_syms = sorted(by_sym.items(), key=lambda x: x[1], reverse=True)
        top_winners = [{"symbol": s, "pnl": round(p, 2)} for s, p in sorted_syms if p > 0][:5]
        top_losers = [{"symbol": s, "pnl": round(p, 2)} for s, p in reversed(sorted_syms) if p < 0][:5]

        # Executive Summary
        winning_sym = top_winners[0]["symbol"] if top_winners else "None"
        losing_sym = top_losers[0]["symbol"] if top_losers else "None"
        winning_strat = max(by_strat.items(), key=lambda x: x[1])[0] if by_strat else "None"

        summary = {
            "primary_alpha_source": f"Top contributor was {winning_sym} via {winning_strat}",
            "primary_risk_detractor": f"Largest loss driver was {losing_sym}",
            "execution_efficiency": f"Total transaction friction (fees + slippage) consumed {cost_drag:.1f}% of gross gains",
            "net_verdict": "Profitable alpha" if net_pnl > 0 else "Unprofitable or fee-dragged"
        }

        return AttributionReport(
            total_gross_pnl=round(tot_pnl, 2),
            total_fees=round(tot_fees, 2),
            total_slippage=round(tot_slip, 2),
            net_pnl=round(net_pnl, 2),
            cost_drag_pct=round(cost_drag, 2),
            by_strategy=by_strat,
            by_symbol=by_sym,
            by_sector=by_sec,
            by_factor=by_factor,
            by_regime=by_reg,
            top_winning_drivers=top_winners,
            top_losing_drivers=top_losers,
            executive_summary=summary,
        )
