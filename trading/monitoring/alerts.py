"""
Alerting Engine for US Market Trading Runtime.

Monitors operational thresholds and emits structured alerts for:
- data stale
- broker unavailable
- laya unavailable
- risk engine block
- daily loss threshold
- large drawdown
- reconciliation mismatch
- duplicate order attempt
- unexpected order / position
- model / config mismatch
"""

import time
import logging
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    alert_type: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class AlertManager:
    """Centralized trading alert registry."""

    _instance: Optional["AlertManager"] = None

    def __init__(self):
        self.alerts: List[Alert] = []

    @classmethod
    def get_instance(cls) -> "AlertManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def emit(self, alert_type: str, severity: AlertSeverity, message: str, details: Optional[Dict[str, Any]] = None) -> Alert:
        al = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details or {},
        )
        self.alerts.append(al)

        log_fn = logger.info
        if severity == AlertSeverity.WARNING:
            log_fn = logger.warning
        elif severity == AlertSeverity.ERROR:
            log_fn = logger.error
        elif severity == AlertSeverity.CRITICAL:
            log_fn = logger.critical

        log_fn("[%s ALERT] %s: %s | %s", severity.value, alert_type, message, details or "")
        return al

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.alerts[-limit:]]
