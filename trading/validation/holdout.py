"""
Permanent Protected Holdout — Protected out-of-sample data with contamination auditing.

The evolutionary system MUST NEVER optimize against the holdout set.
Only finalists that passed all walk-forward and adversarial gates are evaluated.
If evaluated more than MAX_ALLOWED_EVALUATIONS times, marks data as CONTAMINATED.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

AUDIT_LOG_FILE = Path("trading/validation/holdout_audit.json")
MAX_ALLOWED_EVALUATIONS = 5  # Maximum number of finalist checks before marking contaminated


class ProtectedHoldout:
    """Manages the final holdout split and audits access."""

    def __init__(self, holdout_pct: float = 0.20, audit_file: Optional[Path] = None):
        self.holdout_pct = holdout_pct
        self.audit_file = audit_file or AUDIT_LOG_FILE
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Partition dataset into (research_df, protected_holdout_df).
        Research df contains 80% chronological history, holdout contains recent 20%.
        """
        n = len(df)
        split_idx = int(n * (1.0 - self.holdout_pct))
        research_df = df.iloc[:split_idx].copy()
        holdout_df = df.iloc[split_idx:].copy()
        return research_df, holdout_df

    def audit_access(self, strategy_id: str, candidate_generation: int) -> Dict[str, Any]:
        """
        Log access to the holdout set and check for dataset contamination.
        """
        records = []
        if self.audit_file.exists():
            try:
                records = json.loads(self.audit_file.read_text(encoding="utf-8"))
            except Exception:
                records = []

        # Count previous accesses
        access_count = len(records)
        is_contaminated = access_count >= MAX_ALLOWED_EVALUATIONS

        entry = {
            "access_id": access_count + 1,
            "strategy_id": strategy_id,
            "generation": candidate_generation,
            "contaminated": is_contaminated,
            "total_accesses": access_count + 1,
        }
        records.append(entry)
        self.audit_file.write_text(json.dumps(records, indent=2), encoding="utf-8")

        if is_contaminated:
            logger.error("🚨 CRITICAL AUDIT FLAG: Protected holdout set is CONTAMINATED (%d evaluations)", access_count + 1)

        return {
            "strategy_id": strategy_id,
            "eval_count": access_count + 1,
            "is_contaminated": is_contaminated,
            "allowed": not is_contaminated,
            "warning": "Holdout set has reached maximum reuse threshold. Results may be overfitted." if is_contaminated else "Holdout access logged cleanly.",
        }
