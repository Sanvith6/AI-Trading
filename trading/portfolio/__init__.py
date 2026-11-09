"""
Portfolio Package — Construction, sizing, correlation clustering, and attribution.
"""

from trading.portfolio.portfolio_constructor import (
    PortfolioConstructor,
    PortfolioProposal,
    PositionProposal,
)
from trading.portfolio.optimizer import RobustMeanCVaROptimizer, OptimizerResult
from trading.portfolio.covariance import ShrinkageCovarianceEstimator
from trading.portfolio.risk_models import SignalRiskModel
from trading.portfolio.kelly import FractionalKellyCeiling
from trading.portfolio.cvar import CVaRModel
from trading.portfolio.transaction_costs import TransactionCostModel
from trading.portfolio.position_sizer import PositionSizer
from trading.portfolio.correlation_matrix import StrategyCorrelationMatrix
from trading.portfolio.attribution import PerformanceAttributor, AttributionReport
from trading.portfolio.legacy_constructor import LegacyPortfolioConstructor

__all__ = [
    "PortfolioConstructor",
    "PortfolioProposal",
    "PositionProposal",
    "RobustMeanCVaROptimizer",
    "OptimizerResult",
    "ShrinkageCovarianceEstimator",
    "SignalRiskModel",
    "FractionalKellyCeiling",
    "CVaRModel",
    "TransactionCostModel",
    "PositionSizer",
    "StrategyCorrelationMatrix",
    "PerformanceAttributor",
    "AttributionReport",
    "LegacyPortfolioConstructor",
]