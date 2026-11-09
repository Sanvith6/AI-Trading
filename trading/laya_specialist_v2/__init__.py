"""
Laya Financial Specialist V2 Package.

Autonomous Financial Meta-Decision Specialist (Generation 2).
Operates as a secondary advisory intelligence evaluating candidate trades proposed
by the 12-alpha baseline ensemble and ML meta-model.
"""

from .config import LayaV2Config
from .tasks import SpecialistTasksV2
from .dataset_builder import (
    FinancialCandidateExampleV2,
    FinancialDatasetBuilderV2,
    FinancialOutcomeEngineV2,
)
from .decision_policy import EconomicDecisionPolicyV2, MetaDecisionOutputV2
from .evaluator import SpecialistEvaluatorV2

__all__ = [
    "LayaV2Config",
    "SpecialistTasksV2",
    "FinancialCandidateExampleV2",
    "FinancialDatasetBuilderV2",
    "FinancialOutcomeEngineV2",
    "EconomicDecisionPolicyV2",
    "MetaDecisionOutputV2",
    "SpecialistEvaluatorV2",
]
