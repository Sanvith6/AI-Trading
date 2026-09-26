"""
Research Memory & Failure Knowledge Base — Prevents repeating failed quantitative ideas.

Maintains:
1. Machine-readable failure database of every dead genome
2. Pre-mutation query: `has_idea_already_failed(candidate_dna)`
3. Structured registry of successful vs failed experiments across regimes
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from trading.swarm.strategy_dna import StrategyDNA

MEMORY_DIR = Path("trading/swarm/research_memory")


class ResearchMemory:
    """Stores quantitative knowledge gained across all evolutionary generations."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or MEMORY_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.failures_file = self.storage_dir / "failed_hypotheses.json"
        self.insights_file = self.storage_dir / "regime_insights.json"
        self._ensure_files()

    def _ensure_files(self):
        if not self.failures_file.exists():
            self.failures_file.write_text("[]", encoding="utf-8")
        if not self.insights_file.exists():
            self.insights_file.write_text("{}", encoding="utf-8")

    def record_failure(
        self,
        dna: StrategyDNA,
        fatal_metric: str,
        reason: str,
        regime_behavior: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Archive a terminated strategy with autopsy cause."""
        failures = self.get_all_failures()
        record = {
            "strategy_id": dna.strategy_id,
            "name": dna.name,
            "generation": dna.generation,
            "species": dna.species,
            "alphas": dna.active_alphas,
            "stop_loss_pct": dna.stop_loss_pct,
            "take_profit_pct": dna.take_profit_pct,
            "fatal_metric": fatal_metric,
            "reason": reason,
            "regime_behavior": regime_behavior or {},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        failures.append(record)
        self.failures_file.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    def get_all_failures(self) -> List[Dict[str, Any]]:
        """Retrieve all recorded failure instances."""
        try:
            return json.loads(self.failures_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def has_idea_already_failed(self, candidate: StrategyDNA, tolerance: float = 0.15) -> Tuple[bool, Optional[str]]:
        """
        Queries whether this parameter and alpha combination closely resembles a prior failure.
        """
        failures = self.get_all_failures()
        cand_alphas = set(candidate.active_alphas)

        for f in failures:
            f_alphas = set(f.get("alphas", []))
            # If same alpha combination
            if cand_alphas == f_alphas and f.get("species") == candidate.species:
                # Check parameter proximity
                sl_diff = abs(candidate.stop_loss_pct - f.get("stop_loss_pct", 0)) / max(0.1, candidate.stop_loss_pct)
                tp_diff = abs(candidate.take_profit_pct - f.get("take_profit_pct", 0)) / max(0.1, candidate.take_profit_pct)
                if sl_diff < tolerance and tp_diff < tolerance:
                    reason = f"Idea failed in Gen {f.get('generation')}: {f.get('reason')} ({f.get('fatal_metric')})"
                    return True, reason

        return False, None

    def record_regime_insight(self, regime: str, winning_species: str, lessons: str) -> None:
        """Record what works well in a specific market regime."""
        try:
            data = json.loads(self.insights_file.read_text(encoding="utf-8"))
        except Exception:
            data = {}

        if regime not in data:
            data[regime] = []
        data[regime].append({
            "winning_species": winning_species,
            "lessons": lessons,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        self.insights_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
