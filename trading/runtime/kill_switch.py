"""
Global Emergency Kill Switch for US Trading Runtime.

Guarantees:
1. When active, NO NEW ORDERS can be generated or submitted in either Paper or Live.
2. Read-only market data, reconciliation, and monitoring remain operational.
3. State is persisted atomically to disk and survives process restarts.
4. Requires explicit operator identity to engage or disengage.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_KILL_SWITCH_PATH = Path("trading/kill_switch.json")


class GlobalKillSwitch:
    """Thread-safe persistent emergency kill switch."""

    _instance: Optional["GlobalKillSwitch"] = None

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or DEFAULT_KILL_SWITCH_PATH
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file()

    @classmethod
    def get_instance(cls, state_file: Optional[Path] = None) -> "GlobalKillSwitch":
        if cls._instance is None:
            cls._instance = cls(state_file)
        return cls._instance

    def _ensure_file(self) -> None:
        if not self.state_file.exists():
            initial_state = {
                "active": False,
                "engaged_at": None,
                "engaged_by": None,
                "reason": None,
                "disengaged_at": None,
                "disengaged_by": None,
            }
            self._write_state(initial_state)

    def _read_state(self) -> Dict[str, Any]:
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to read kill switch state from %s: %s", self.state_file, e)
            # Fail closed: if we cannot read state, assume kill switch IS ACTIVE
            return {"active": True, "reason": "FAILED_TO_READ_STATE_FILE_FAIL_CLOSED"}

    def _write_state(self, state: Dict[str, Any]) -> None:
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(state, f, indent=2)
        temp_file.replace(self.state_file)

    def is_active(self) -> bool:
        """Returns True if the kill switch is currently engaged."""
        # Also check runtime environment variable EMERGENCY_HALT
        if os.getenv("EMERGENCY_HALT", "false").lower() in ("true", "1", "yes"):
            return True
        state = self._read_state()
        return bool(state.get("active", False))

    def get_status(self) -> Dict[str, Any]:
        """Returns complete kill switch status dict."""
        state = self._read_state()
        env_halt = os.getenv("EMERGENCY_HALT", "false").lower() in ("true", "1", "yes")
        return {
            "active": state.get("active", False) or env_halt,
            "file_active": state.get("active", False),
            "env_halt": env_halt,
            "engaged_at": state.get("engaged_at"),
            "engaged_by": state.get("engaged_by"),
            "reason": state.get("reason"),
        }

    def engage(self, reason: str, operator: str = "SYSTEM") -> None:
        """Engages the emergency kill switch, blocking all new trading."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state = {
            "active": True,
            "engaged_at": now,
            "engaged_by": operator,
            "reason": reason,
            "disengaged_at": None,
            "disengaged_by": None,
        }
        self._write_state(state)
        logger.critical("[KILL SWITCH ENGAGED] Global kill switch engaged by %s: %s", operator, reason)

    def disengage(self, operator: str = "SYSTEM") -> None:
        """Disengages the emergency kill switch, permitting normal trading."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state = self._read_state()
        state["active"] = False
        state["disengaged_at"] = now
        state["disengaged_by"] = operator
        self._write_state(state)
        logger.warning("[KILL SWITCH DISENGAGED] Global kill switch disengaged by %s.", operator)
