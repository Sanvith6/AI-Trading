"""
Strategy DNA — Next-Generation Institutional Genome Model.

Evolves structured mathematical and operational DNA rather than arbitrary agent text prompts.
Every field is typed, schema-validated, and bound within hard legal quantitative limits.
"""

import uuid
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


LEGAL_BOUNDS = {
    "stop_loss_pct": (0.5, 6.0),
    "take_profit_pct": (1.0, 15.0),
    "trailing_stop_pct": (0.5, 5.0),
    "max_position_weight": (0.05, 0.35),
    "risk_per_trade_pct": (0.5, 4.0),
    "rebalance_bars": (1, 30),
    "max_holding_bars": (2, 90),
    "adx_threshold": (12.0, 40.0),
    "rsi_oversold": (20.0, 42.0),
    "rsi_overbought": (58.0, 80.0),
    "zscore_threshold": (1.0, 3.0),
    "min_confidence": (0.50, 0.85),
    "turnover_penalty_weight": (0.01, 0.20),
    # Laya Task-Specific Threshold Bounds (Directives 3 & 12)
    "laya_trade_quality_threshold": (0.40, 0.90),
    "laya_risk_threshold": (0.45, 0.90),
    "laya_regime_threshold": (0.35, 0.85),
}

LEGAL_DISAGREEMENT_POLICIES = ["BASELINE_WINS", "REDUCE_CONFIDENCE", "HALVE_SIZE"]

AVAILABLE_ALPHAS = [
    "TrendAlpha", "MomentumAlpha", "MeanReversionAlpha", "BreakoutAlpha",
    "VolatilityAlpha", "VolumeFlowAlpha", "CrossSectionalAlpha", "StatisticalAlpha",
    "RelativeStrengthAlpha", "DefensiveAlpha", "RegimeAlpha", "MLHybridAlpha"
]

AVAILABLE_REGIMES = [
    "STRONG_BULL", "WEAK_BULL", "SIDEWAYS", "WEAK_BEAR", "STRONG_BEAR",
    "HIGH_VOLATILITY", "LOW_VOLATILITY", "RISK_ON", "RISK_OFF"
]


@dataclass
class StrategyDNA:
    """Institutional Strategy Genome with 15+ typed fields."""

    strategy_id: str
    name: str
    generation: int
    species: str  # e.g., "Trend", "Momentum", "MeanReversion", "Breakout", "Defensive", "Hybrid"
    parent_id: Optional[str] = None

    # 1. Universe rules
    universe_size: int = 50
    min_volume: int = 100000

    # 2. Alpha families and signal weights
    active_alphas: List[str] = field(default_factory=lambda: ["TrendAlpha", "MomentumAlpha"])
    alpha_weights: Dict[str, float] = field(default_factory=lambda: {"TrendAlpha": 0.5, "MomentumAlpha": 0.5})

    # 3. Feature subset
    feature_subset: List[str] = field(default_factory=lambda: ["ret_1", "ret_20", "rsi_14", "adx_14", "atr_pct"])

    # 4. Lookbacks & Thresholds
    lookback_bars: int = 20
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 5.0
    trailing_stop_pct: float = 1.5
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    zscore_threshold: float = 2.0
    min_confidence: float = 0.65

    # 5. Regime rules (permitted regimes to deploy capital)
    allowed_regimes: List[str] = field(default_factory=lambda: ["STRONG_BULL", "WEAK_BULL", "SIDEWAYS"])

    # 6. Holding period & rebalancing
    rebalance_bars: int = 5
    max_holding_bars: int = 20

    # 7. Position sizing & portfolio constraints
    position_sizing_method: str = "risk_budget"  # "risk_budget" | "inverse_vol" | "half_kelly"
    risk_per_trade_pct: float = 2.0
    max_position_weight: float = 0.20
    max_correlated_exposure: float = 0.40

    # 8. Execution assumptions
    execution_order_type: str = "market"  # "market" | "limit"
    assumed_slippage_bps: float = 10.0

    # 9. Fitness & Novelty tracking
    fitness: float = 0.0
    novelty_score: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    is_alive: bool = True
    death_reason: Optional[str] = None

    # 10. Laya Challenger Intelligence Genes (Directives 3 & 12)
    laya_enabled: bool = False
    laya_trade_quality_enabled: bool = False
    laya_risk_action_enabled: bool = False
    laya_regime_opinion_enabled: bool = False

    laya_trade_quality_threshold: float = 0.60
    laya_risk_threshold: float = 0.65
    laya_regime_threshold: float = 0.50

    laya_disagreement_policy: str = "BASELINE_WINS"  # BASELINE_WINS | REDUCE_CONFIDENCE | HALVE_SIZE

    def validate_bounds(self) -> bool:
        """Enforces that all numeric parameters remain strictly within legal limits."""
        for param, (low, high) in LEGAL_BOUNDS.items():
            if hasattr(self, param):
                val = getattr(self, param)
                if not (low <= val <= high):
                    setattr(self, param, max(low, min(high, val)))

        # Directive 12: Ensure legal disagreement policy
        if self.laya_disagreement_policy not in LEGAL_DISAGREEMENT_POLICIES:
            self.laya_disagreement_policy = "BASELINE_WINS"

        # Directive 12 Invariant: "Laya overrides risk" is ILLEGAL under all circumstances
        if hasattr(self, "laya_overrides_risk"):
            delattr(self, "laya_overrides_risk")

        return True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def mutate(self, mutation_rate: float = 0.20) -> "StrategyDNA":
        """Creates a schema-bounded mutation of this StrategyDNA."""
        return mutate_dna(self, self.generation + 1, mutation_rate)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyDNA":
        valid_keys = cls.__dataclass_fields__.keys()
        clean_data = {k: v for k, v in data.items() if k in valid_keys}
        dna = cls(**clean_data)
        dna.validate_bounds()
        return dna


def mutate_dna(parent: StrategyDNA, generation: int, mutation_rate: float = 0.20) -> StrategyDNA:
    """
    Creates a schema-bounded mutation of a parent StrategyDNA.
    No arbitrary prompt or executable code is generated.
    """
    child_data = parent.to_dict()
    child_data["strategy_id"] = f"strat_{generation}_{uuid.uuid4().hex[:6]}"
    child_data["parent_id"] = parent.strategy_id
    child_data["generation"] = generation
    child_data["fitness"] = 0.0
    child_data["metrics"] = {}
    child_data["is_alive"] = True
    child_data["death_reason"] = None

    # Mutate numeric parameters within bounds
    for param, (low, high) in LEGAL_BOUNDS.items():
        if random.random() < mutation_rate and param in child_data:
            current_val = child_data[param]
            delta = (high - low) * random.uniform(-0.15, 0.15)
            child_data[param] = round(max(low, min(high, current_val + delta)), 3)

    # Mutate alpha library composition
    if random.random() < mutation_rate:
        k = random.randint(1, 3)
        alphas = random.sample(AVAILABLE_ALPHAS, k)
        child_data["active_alphas"] = alphas
        weights = [random.random() for _ in range(k)]
        tot = sum(weights)
        child_data["alpha_weights"] = {a: round(w / tot, 2) for a, w in zip(alphas, weights)}

    # Mutate Laya genes (Directive 12)
    if random.random() < mutation_rate:
        child_data["laya_enabled"] = not child_data.get("laya_enabled", False)
    if random.random() < mutation_rate:
        child_data["laya_trade_quality_enabled"] = not child_data.get("laya_trade_quality_enabled", False)
    if random.random() < mutation_rate:
        child_data["laya_risk_action_enabled"] = not child_data.get("laya_risk_action_enabled", False)
    if random.random() < mutation_rate:
        child_data["laya_regime_opinion_enabled"] = not child_data.get("laya_regime_opinion_enabled", False)
    if random.random() < mutation_rate:
        child_data["laya_disagreement_policy"] = random.choice(LEGAL_DISAGREEMENT_POLICIES)

    child = StrategyDNA.from_dict(child_data)
    child.validate_bounds()
    return child


def create_initial_population(size: int = 32) -> List[StrategyDNA]:
    """Generates initial diverse population with species preservation."""
    species_list = ["Trend", "Momentum", "MeanReversion", "Breakout", "Defensive", "Hybrid"]
    population = []

    for i in range(size):
        spec = species_list[i % len(species_list)]
        strat_id = f"strat_1_{spec.lower()}_{i+1:02d}"

        if spec == "Trend":
            alphas = ["TrendAlpha", "RegimeAlpha"]
            weights = {"TrendAlpha": 0.7, "RegimeAlpha": 0.3}
            stop_loss = 2.5
            take_profit = 6.0
        elif spec == "Momentum":
            alphas = ["MomentumAlpha", "RelativeStrengthAlpha"]
            weights = {"MomentumAlpha": 0.6, "RelativeStrengthAlpha": 0.4}
            stop_loss = 2.0
            take_profit = 5.0
        elif spec == "MeanReversion":
            alphas = ["MeanReversionAlpha", "StatisticalAlpha"]
            weights = {"MeanReversionAlpha": 0.7, "StatisticalAlpha": 0.3}
            stop_loss = 1.5
            take_profit = 3.5
        elif spec == "Breakout":
            alphas = ["BreakoutAlpha", "VolumeFlowAlpha"]
            weights = {"BreakoutAlpha": 0.6, "VolumeFlowAlpha": 0.4}
            stop_loss = 2.0
            take_profit = 7.0
        elif spec == "Defensive":
            alphas = ["DefensiveAlpha", "RegimeAlpha"]
            weights = {"DefensiveAlpha": 0.8, "RegimeAlpha": 0.2}
            stop_loss = 1.2
            take_profit = 3.0
        else:
            alphas = ["TrendAlpha", "MLHybridAlpha"]
            weights = {"TrendAlpha": 0.5, "MLHybridAlpha": 0.5}
            stop_loss = 2.0
            take_profit = 5.0

        dna = StrategyDNA(
            strategy_id=strat_id,
            name=f"{spec}Gladiator_{i+1}",
            generation=1,
            species=spec,
            active_alphas=alphas,
            alpha_weights=weights,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
        )
        dna.validate_bounds()
        population.append(dna)

    return population
