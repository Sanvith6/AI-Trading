"""
Research Operations Layer — Champion/Challenger framework, shadow execution, and research cycles.
"""

from trading.research.champion_challenger import ChampionChallengerManager, PromotionGateThresholds
from trading.research.cycles import ResearchLifecycle

__all__ = [
    "ChampionChallengerManager",
    "PromotionGateThresholds",
    "ResearchLifecycle",
]
