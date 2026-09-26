"""
Persistent Forward Paper-Trading Journal & Calibration Ledger.

Maintains an immutable, append-only historical ledger of all paper-trading decisions,
simulated execution fills, statutory fees, and daily/weekly/monthly performance rollups.

Ensures real-world tracking error, prediction calibration, and forward evidence
are preserved permanently across system restarts.
"""

import os
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_JOURNAL_DIR = Path("trading/paper/journal_data")


class ForwardPaperJournal:
    """Persistent ledger for live and forward paper-trading execution logs."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or DEFAULT_JOURNAL_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.storage_dir / "paper_orders_ledger.jsonl"
        self.daily_file = self.storage_dir / "daily_rollups.json"
        self._ensure_files()

    def _ensure_files(self):
        if not self.log_file.exists():
            self.log_file.touch()
        if not self.daily_file.exists():
            self.daily_file.write_text("{}", encoding="utf-8")

    def record_decision(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        expected_return_pct: float,
        expected_vol_pct: float,
        position_size: int,
        execution_price: float,
        slippage_bps: float,
        fees_statutory: float,
        realized_pnl: float,
        strategy_version: str = "v2.0-DNA",
        universe_version: str = "20260920-v1",
        generation: int = 10,
        model_version: str = "MetaModel-v1",
        regime: str = "SIDEWAYS",
        reason: str = "Signal alignment",
        risk_checks_passed: bool = True,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Append a forward paper trading decision to the persistent ledger.
        """
        ts = timestamp or datetime.now().isoformat()
        trade_id = f"FWD-{int(datetime.now().timestamp()*1000)}"

        entry = {
            "trade_id": trade_id,
            "timestamp": ts,
            "date": ts[:10],
            "symbol": symbol.upper(),
            "signal": signal,
            "confidence": round(float(confidence), 3),
            "expected_return_pct": round(float(expected_return_pct), 2),
            "expected_vol_pct": round(float(expected_vol_pct), 2),
            "position_size": int(position_size),
            "execution_price": round(float(execution_price), 2),
            "slippage_bps": round(float(slippage_bps), 1),
            "fees_statutory": round(float(fees_statutory), 2),
            "realized_pnl": round(float(realized_pnl), 2),
            "strategy_version": strategy_version,
            "universe_version": universe_version,
            "generation": generation,
            "model_version": model_version,
            "regime": regime,
            "reason": reason,
            "risk_checks_passed": risk_checks_passed,
        }

        # Write to JSONL
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        self._update_daily_rollup(entry)
        return entry

    def _update_daily_rollup(self, entry: Dict[str, Any]):
        """Maintain daily performance metrics in aggregated registry."""
        dt = entry["date"]
        try:
            rollups = json.loads(self.daily_file.read_text(encoding="utf-8"))
        except Exception:
            rollups = {}

        if dt not in rollups:
            rollups[dt] = {
                "date": dt,
                "trades_count": 0,
                "gross_pnl": 0.0,
                "fees_total": 0.0,
                "net_pnl": 0.0,
                "win_count": 0,
                "symbols_traded": [],
            }

        day = rollups[dt]
        day["trades_count"] += 1
        day["fees_total"] = round(day["fees_total"] + entry["fees_statutory"], 2)
        day["net_pnl"] = round(day["net_pnl"] + entry["realized_pnl"], 2)
        day["gross_pnl"] = round(day["net_pnl"] + day["fees_total"], 2)
        if entry["realized_pnl"] > 0:
            day["win_count"] += 1
        if entry["symbol"] not in day["symbols_traded"]:
            day["symbols_traded"].append(entry["symbol"])

        self.daily_file.write_text(json.dumps(rollups, indent=2), encoding="utf-8")

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Retrieve complete history of paper trade orders."""
        records = []
        if self.log_file.exists():
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            pass
        return records

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Compute aggregate forward paper trading performance metrics."""
        records = self.get_all_records()
        if not records:
            return {
                "total_trades": 0,
                "active_days": 0,
                "net_pnl": 0.0,
                "total_fees": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "forward_validation_status": "PENDING (0 forward sessions logged)",
            }

        pnls = [r["realized_pnl"] for r in records]
        fees = [r["fees_statutory"] for r in records]
        dates = list(set([r["date"] for r in records]))
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        gross_win = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1e-6
        pf = round(gross_win / gross_loss, 2)
        win_rate = round(len(wins) / len(pnls) * 100.0, 1)

        status = f"PENDING ({len(dates)} / 30 required forward calendar sessions logged)"
        if len(dates) >= 30 and len(records) >= 50:
            status = "FORWARD VALIDATION COMPLETED"

        return {
            "total_trades": len(records),
            "active_days": len(dates),
            "net_pnl": round(sum(pnls), 2),
            "total_fees": round(sum(fees), 2),
            "win_rate_pct": win_rate,
            "profit_factor": pf,
            "forward_validation_status": status,
        }


# Singleton instance
paper_journal = ForwardPaperJournal()
