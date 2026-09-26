"""
SEC EDGAR / XBRL Fundamental Data Engine & Causal Point-in-Time Store.

Tracks quarterly financial health metrics with strict public availability timestamps:
- period_end: Financial quarter end date (e.g., 2023-12-31)
- filing_date: When the 10-Q/10-K was formally accepted by SEC (e.g., 2024-02-01)
- available_at: Exact UTC dissemination timestamp (e.g., 2024-02-01T21:30:00Z)

CRITICAL INVARIANT:
A model running on 2024-01-15 CANNOT see Q4 2023 fundamentals, because they were
not publicly filed until 2024-02-01. Violating this rule introduces massive lookahead bias.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FUNDAMENTALS_PATH = Path(__file__).parent.parent.parent / "data_lake" / "fundamentals.json"


@dataclass
class FundamentalRecord:
    symbol: str
    period_end: str  # "YYYY-MM-DD"
    filing_date: str  # "YYYY-MM-DD"
    available_at: str  # "YYYY-MM-DDTHH:MM:SSZ" (Strict Causality Boundary)
    form_type: str  # "10-Q" or "10-K"
    revenue: float
    revenue_growth_yoy: float
    gross_margin: float
    operating_margin: float
    ebitda: float
    net_income: float
    eps: float
    free_cash_flow: float
    cash_and_equivalents: float
    total_debt: float
    total_assets: float
    total_liabilities: float
    shares_outstanding: float
    roe: float
    roic: float
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    accession_number: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EdgarFundamentalProvider:
    """
    Manages point-in-time fundamental statements for research and live trading.
    Guarantees zero future data leakage.
    """

    # Baseline curated fundamentals for primary active US watchlist
    CURATED_RECORDS = [
        # AAPL Q4 2023 (Ended Sep 2023, Filed Nov 2023)
        FundamentalRecord(
            symbol="AAPL", period_end="2023-09-30", filing_date="2023-11-03", available_at="2023-11-03T21:30:00Z",
            form_type="10-K", revenue=89498e6, revenue_growth_yoy=-0.007, gross_margin=0.452, operating_margin=0.301,
            ebitda=32000e6, net_income=22956e6, eps=1.46, free_cash_flow=20700e6, cash_and_equivalents=29965e6,
            total_debt=111088e6, total_assets=352583e6, total_liabilities=290437e6, shares_outstanding=15550e6,
            roe=1.47, roic=0.45, accession_number="0000320193-23-000106"
        ),
        # AAPL Q1 2024 (Ended Dec 2023, Filed Feb 2024)
        FundamentalRecord(
            symbol="AAPL", period_end="2023-12-30", filing_date="2024-02-02", available_at="2024-02-02T21:30:00Z",
            form_type="10-Q", revenue=119575e6, revenue_growth_yoy=0.021, gross_margin=0.459, operating_margin=0.338,
            ebitda=42500e6, net_income=33916e6, eps=2.18, free_cash_flow=37500e6, cash_and_equivalents=40760e6,
            total_debt=108000e6, total_assets=353514e6, total_liabilities=279414e6, shares_outstanding=15440e6,
            roe=1.54, roic=0.48, accession_number="0000320193-24-000006"
        ),
        # NVDA Q3 FY24 (Ended Oct 2023, Filed Nov 2023)
        FundamentalRecord(
            symbol="NVDA", period_end="2023-10-29", filing_date="2023-11-21", available_at="2023-11-21T21:30:00Z",
            form_type="10-Q", revenue=18120e6, revenue_growth_yoy=2.06, gross_margin=0.740, operating_margin=0.575,
            ebitda=11000e6, net_income=9243e6, eps=3.71, free_cash_flow=7042e6, cash_and_equivalents=18281e6,
            total_debt=9706e6, total_assets=54150e6, total_liabilities=20819e6, shares_outstanding=2470e6,
            roe=0.88, roic=0.55, accession_number="0001045810-23-000068"
        ),
        # NVDA Q4 FY24 (Ended Jan 2024, Filed Feb 2024)
        FundamentalRecord(
            symbol="NVDA", period_end="2024-01-28", filing_date="2024-02-21", available_at="2024-02-21T21:30:00Z",
            form_type="10-K", revenue=22103e6, revenue_growth_yoy=2.65, gross_margin=0.760, operating_margin=0.616,
            ebitda=14500e6, net_income=12285e6, eps=4.93, free_cash_flow=11217e6, cash_and_equivalents=25984e6,
            total_debt=9706e6, total_assets=65728e6, total_liabilities=22750e6, shares_outstanding=2465e6,
            roe=1.15, roic=0.68, accession_number="0001045810-24-000029"
        ),
        # MSFT Q2 FY24 (Ended Dec 2023, Filed Jan 2024)
        FundamentalRecord(
            symbol="MSFT", period_end="2023-12-31", filing_date="2024-01-30", available_at="2024-01-30T21:30:00Z",
            form_type="10-Q", revenue=62020e6, revenue_growth_yoy=0.176, gross_margin=0.684, operating_margin=0.436,
            ebitda=30000e6, net_income=21870e6, eps=2.93, free_cash_flow=9115e6, cash_and_equivalents=88000e6,
            total_debt=75000e6, total_assets=470558e6, total_liabilities=232316e6, shares_outstanding=7430e6,
            roe=0.39, roic=0.28, accession_number="0000950170-24-008762"
        ),
    ]

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or DEFAULT_FUNDAMENTALS_PATH
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[FundamentalRecord] = self._load()

    def _load(self) -> List[FundamentalRecord]:
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                return [FundamentalRecord(**item) for item in data]
            except Exception as e:
                logger.error(f"Failed to read fundamentals storage: {e}")

        # Seed defaults
        self._save(self.CURATED_RECORDS)
        return list(self.CURATED_RECORDS)

    def _save(self, records: List[FundamentalRecord]) -> None:
        try:
            data = [r.to_dict() for r in records]
            self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to write fundamentals storage: {e}")

    def record_fundamental(self, rec: FundamentalRecord) -> None:
        """Register a new verified fundamental record."""
        # Replace if duplicate symbol + period_end
        self.records = [r for r in self.records if not (r.symbol == rec.symbol and r.period_end == rec.period_end)]
        self.records.append(rec)
        self._save(self.records)

    def get_latest_pit_fundamentals(self, symbol: str, as_of_date: str) -> Optional[FundamentalRecord]:
        """
        Fetch the latest legally accessible fundamental statement for a symbol on as_of_date.
        Strict Causality: filters available_at <= as_of_date.
        """
        clean_sym = symbol.strip().upper()
        # Parse as_of_date to isoformat with time if needed
        as_of_dt_str = as_of_date if "T" in as_of_date else f"{as_of_date}T23:59:59Z"
        as_of_dt = datetime.fromisoformat(as_of_dt_str.replace("Z", "+00:00"))

        matching = []
        for r in self.records:
            if r.symbol.upper() == clean_sym:
                avail_dt = datetime.fromisoformat(r.available_at.replace("Z", "+00:00"))
                # CAUSALITY GUARANTEE: Must be filed before or at as_of_dt
                if avail_dt <= as_of_dt:
                    matching.append(r)

        if not matching:
            return None

        # Return latest filed statement
        return sorted(matching, key=lambda x: x.available_at)[-1]

    def get_feature_vector(self, symbol: str, as_of_date: str, current_price: float) -> Dict[str, float]:
        """
        Generate fundamental features safely usable by ML/alpha models as of historical date.
        """
        rec = self.get_latest_pit_fundamentals(symbol, as_of_date)
        if not rec:
            return {
                "f_revenue_growth_yoy": 0.0,
                "f_gross_margin": 0.40,
                "f_operating_margin": 0.20,
                "f_roe": 0.15,
                "f_roic": 0.12,
                "f_pe_ratio": 25.0,
                "f_quality_score": 0.50,
            }

        market_cap = current_price * rec.shares_outstanding if rec.shares_outstanding > 0 else 1.0
        pe = (market_cap / rec.net_income) if rec.net_income > 0 else 50.0

        # Quality score composite
        quality = np.clip(
            (rec.operating_margin * 2.0) + (rec.roe * 0.5) + (rec.free_cash_flow / max(rec.revenue, 1.0)),
            0.0, 1.0
        )

        return {
            "f_revenue_growth_yoy": round(float(rec.revenue_growth_yoy), 4),
            "f_gross_margin": round(float(rec.gross_margin), 4),
            "f_operating_margin": round(float(rec.operating_margin), 4),
            "f_roe": round(float(rec.roe), 4),
            "f_roic": round(float(rec.roic), 4),
            "f_pe_ratio": round(float(pe), 2),
            "f_quality_score": round(float(quality), 4),
        }
