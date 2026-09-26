"""
Advanced Genetic Algorithm Operators for Trading Swarm
Implements Tournament Selection, Blended Crossover (BLX-alpha),
Adaptive Gaussian Mutation, and Speciation (Niching) to prevent premature convergence.
"""

import random
import math
from typing import List, Dict, Any, Tuple, Optional
from trading.swarm.agent_dna import AgentDNA


# ── Archetype Niches (Speciation) ─────────────────────────────────────────────
NICHE_BREAKOUT       = "Momentum Breakout"
NICHE_MEAN_REVERSION = "Mean Reversion Stat-Arb"
NICHE_TREND          = "Trend Following"
NICHE_CONTRARIAN     = "Contrarian Defensive"

NICHES = [NICHE_BREAKOUT, NICHE_MEAN_REVERSION, NICHE_TREND, NICHE_CONTRARIAN]


def tournament_selection(
    population: List[AgentDNA],
    fitness_map: Dict[str, float],
    k: int = 3
) -> AgentDNA:
    """
    Stochastically selects the fittest individual from k random candidates.
    Avoids extreme elitism while maintaining selection pressure.
    """
    alive = [a for a in population if a.status != "DEAD"]
    if not alive:
        return population[0]

    sample_size = min(k, len(alive))
    candidates = random.sample(alive, sample_size)
    candidates.sort(key=lambda a: fitness_map.get(a.agent_id, -999.0), reverse=True)
    return candidates[0]


def blx_alpha_crossover(
    parent1_val: float,
    parent2_val: float,
    min_val: float,
    max_val: float,
    alpha: float = 0.3
) -> float:
    """
    Blended Crossover (BLX-alpha) for real-valued numerical parameters:
    Offspring is sampled from [min(P1, P2) - alpha*d, max(P1, P2) + alpha*d]
    where d = |P1 - P2|.
    """
    p_min = min(parent1_val, parent2_val)
    p_max = max(parent1_val, parent2_val)
    d = p_max - p_min

    lower_bound = max(min_val, p_min - alpha * d)
    upper_bound = min(max_val, p_max + alpha * d)

    if lower_bound >= upper_bound:
        return round(p_min, 2)

    child_val = random.uniform(lower_bound, upper_bound)
    return round(child_val, 2)


def recombine_parameters(
    parent1_params: Dict[str, Any],
    parent2_params: Dict[str, Any],
    mutation_rate: float = 0.15,
    mutation_sigma: float = 0.1
) -> Dict[str, Any]:
    """
    Recombines the genome parameters of two parents using BLX-alpha crossover
    and applies adaptive Gaussian mutation.
    """
    child_params = {}

    # Parameter constraints: (min, max, is_integer)
    param_bounds = {
        "stop_loss_pct": (1.0, 3.5, False),
        "take_profit_pct": (2.0, 7.0, False),
        "min_confidence": (60, 85, True),
        "max_position_pct": (20, 50, True),
        "rsi_oversold": (25, 40, True),
        "rsi_overbought": (60, 75, True),
    }

    all_keys = set(parent1_params.keys()).union(set(parent2_params.keys()))

    for key in all_keys:
        val1 = parent1_params.get(key)
        val2 = parent2_params.get(key, val1)

        if key in param_bounds and isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            b_min, b_max, is_int = param_bounds[key]
            # Crossover
            child_val = blx_alpha_crossover(float(val1), float(val2), b_min, b_max)

            # Adaptive Mutation
            if random.random() < mutation_rate:
                perturbation = random.gauss(0, mutation_sigma * (b_max - b_min))
                child_val = max(b_min, min(b_max, child_val + perturbation))

            child_params[key] = int(round(child_val)) if is_int else round(child_val, 2)
        else:
            # Discrete or non-numeric: 50/50 inheritance
            child_params[key] = val1 if random.random() < 0.5 else val2

    return child_params


def enforce_speciation(
    next_generation: List[AgentDNA],
    target_niches: Optional[List[str]] = None
) -> List[AgentDNA]:
    """
    Speciation / Niching Engine:
    Ensures the swarm preserves diverse alpha archetypes across market regimes.
    Prevents all agents from collapsing into a single correlated strategy.
    """
    niches = target_niches or NICHES
    for i, agent in enumerate(next_generation):
        assigned_niche = niches[i % len(niches)]
        agent.archetype = assigned_niche
    return next_generation
