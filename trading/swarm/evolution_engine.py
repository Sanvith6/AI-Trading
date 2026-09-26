"""
Evolutionary Engine — Genetic Selection, Inheritance & Mutation
Evaluates agent fitness, reallocates capital to surviving champions,
and uses LLM prompt mutation + parameter tweaking to breed Generation N+1.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from trading.swarm.agent_dna import AgentDNA
from trading.swarm.graveyard import Graveyard
from trading.swarm.genetic_ops import (
    tournament_selection,
    recombine_parameters,
    enforce_speciation,
)
from trading.swarm.quant_metrics import calculate_profit_factor
from llm_client import ask

logger = logging.getLogger(__name__)

LINEAGE_FILE = Path(__file__).parent / "lineage.json"


class EvolutionEngine:
    """
    Orchestrates the Darwinian selection, capital inheritance, and LLM-driven mutation.
    """

    def __init__(self, graveyard: Optional[Graveyard] = None):
        self.graveyard = graveyard or Graveyard()
        self._ensure_files()

    def _ensure_files(self):
        if not LINEAGE_FILE.exists():
            LINEAGE_FILE.write_text("[]", encoding="utf-8")

    def get_lineage(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(LINEAGE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []

    def calculate_fitness(self, agent: AgentDNA) -> float:
        """
        Holistic fitness:
          gain% × 3  +  win-rate × 0.5  +  health × 0.5  +  profit-factor edge
        Profit factor (gross wins / gross losses) rewards real statistical edge,
        not just one lucky trade; agents with <3 trades get no PF bonus.
        """
        if agent.status == "DEAD":
            return -999.0

        win_rate = (agent.wins_count / agent.trades_count * 100.0) if agent.trades_count > 0 else 50.0
        pnl_gain_pct = ((agent.current_capital - agent.initial_capital) / agent.initial_capital) * 100.0
        health_bonus = agent.health * 0.5
        pf_edge = 0.0
        if agent.trades_count >= 3 and agent.trade_history:
            pf = calculate_profit_factor(agent.trade_history)
            pf_edge = max(-10.0, min(20.0, (pf - 1.0) * 10.0))

        fitness = (pnl_gain_pct * 3.0) + (win_rate * 0.5) + health_bonus + pf_edge
        return round(fitness, 2)

    def select_champions(self, agents: List[AgentDNA], top_n: int = 2) -> List[AgentDNA]:
        """Rank active surviving agents by fitness score."""
        alive_agents = [a for a in agents if a.status != "DEAD"]
        if not alive_agents:
            return []
        alive_agents.sort(key=self.calculate_fitness, reverse=True)
        return alive_agents[:top_n]

    def breed_next_generation(
        self,
        current_cohort: List[AgentDNA],
        total_treasury: float,
        target_swarm_size: int = 4,
    ) -> List[AgentDNA]:
        """
        Takes surviving champions, incorporates autopsy lessons from fallen agents,
        and breeds the next generation with mutated prompts and inherited capital.
        """
        champions = self.select_champions(current_cohort, top_n=2)

        # If all died, resurrect a new Gen 1 with fresh lessons
        if not champions:
            logger.warning("All agents in current generation perished! Spawning adaptive reboot generation.")
            return self._spawn_reboot_cohort(total_treasury, target_swarm_size)

        fitness_map = {a.agent_id: self.calculate_fitness(a) for a in current_cohort}
        primary_champ = champions[0]
        secondary_champ = champions[1] if len(champions) > 1 else primary_champ
        next_gen_num = primary_champ.generation + 1
        capital_per_agent = round(total_treasury / target_swarm_size, 2)

        # Retrieve negative lessons from the graveyard
        lessons = self.graveyard.get_lessons()
        recent_lessons = lessons[-4:] if lessons else []
        lessons_text = "\n".join([f"- {l.get('agent_name')} failed: {l.get('fatal_mistake')}. Directive: {l.get('directive')}" for l in recent_lessons]) if recent_lessons else "None yet."

        logger.info(f"🧬 Breeding Generation {next_gen_num} from Champion: {primary_champ.name} (Fitness: {self.calculate_fitness(primary_champ)})")

        # Mutate prompts & parameters via LLM
        mutated_specs = self._llm_mutate(primary_champ, secondary_champ, lessons_text, count=target_swarm_size)

        next_gen: List[AgentDNA] = []
        for i, spec in enumerate(mutated_specs):
            agent_id = f"gen{next_gen_num}_{spec.get('archetype_slug', 'trader')}_{i+1:02d}"

            # Numeric genome: BLX-alpha crossover of two tournament-selected
            # parents (+ Gaussian mutation), then LLM deltas blended on top.
            # Pure LLM output used to overwrite the whole genome — now the GA
            # owns numbers, the LLM owns words.
            parent_b = tournament_selection(current_cohort, fitness_map, k=3)
            base_params = recombine_parameters(
                primary_champ.parameters, parent_b.parameters,
                mutation_rate=0.15, mutation_sigma=0.1,
            )
            base_params.update(spec.get("parameters", {}))
            
            offspring = AgentDNA(
                agent_id=agent_id,
                name=spec.get("name", f"Gen{next_gen_num} Trader #{i+1}"),
                generation=next_gen_num,
                archetype=spec.get("archetype", primary_champ.archetype),
                initial_capital=capital_per_agent,
                current_capital=capital_per_agent,
                cash=capital_per_agent,
                peak_capital=capital_per_agent,
                max_drawdown_pct=spec.get("max_drawdown_pct", 3.0),
                max_loss_amount=round(capital_per_agent * 0.12, 2),
                parameters=base_params,
                system_prompt=spec.get("system_prompt", primary_champ.system_prompt),
                parent_id=primary_champ.agent_id,
                mutation_notes=spec.get("mutation_notes", "Inherited from champion with LLM optimization."),
            )
            next_gen.append(offspring)

        # Speciation: keep niches diverse so the swarm never collapses
        # into one correlated strategy that dies together in one regime.
        next_gen = enforce_speciation(next_gen)

        # Record generational lineage
        self._record_lineage(primary_champ, next_gen, total_treasury)
        return next_gen

    def _llm_mutate(
        self,
        champ1: AgentDNA,
        champ2: AgentDNA,
        lessons_text: str,
        count: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Uses LLM to synthesize the winning champion prompts and inject graveyard immunities.
        """
        prompt = f"""You are the Genetic Mutation Engine of an evolutionary AI stock trading swarm.
A trading generation has completed.

CHAMPION AGENT:
- Name: {champ1.name}
- Archetype: {champ1.archetype}
- Generation: {champ1.generation}
- Total P&L: ₹{champ1.total_pnl} (Wins: {champ1.wins_count}, Losses: {champ1.losses_count})
- Core Strategy: "{champ1.system_prompt[:250]}..."

RECENT DEATHS & FATAL LESSONS FROM FALLEN AGENTS:
{lessons_text}

YOUR TASK:
Breed {count} NEW, evolved offspring agents for Generation {champ1.generation + 1}.
Each offspring must:
1. Retain the winning instincts of Champion {champ1.name}.
2. Strictly integrate the directives learned from fallen agents so it avoids their fatal mistakes.
3. Specialize into a slightly different evolutionary variation (e.g. Offspring 1: aggressive continuation, Offspring 2: conservative confirmation, Offspring 3: adaptive volatility, Offspring 4: risk-hardened guardian).

Return a JSON array containing exactly {count} agent specification objects with keys:
- "name": Evolved name (e.g. "Apex Viper II (Hardened Breakout)")
- "archetype": Short archetype title
- "archetype_slug": Lowercase slug (e.g. "breakout", "trend")
- "mutation_notes": 1 sentence explaining what was evolved/improved
- "system_prompt": Complete, concise system prompt for this agent (120-180 words)
- "parameters": Object with:
    - "stop_loss_pct": float (1.0 to 3.0)
    - "take_profit_pct": float (2.5 to 6.0)
    - "min_confidence": int (60 to 80)
- "max_drawdown_pct": float (typically 2.5 to 3.5)

Respond ONLY with the JSON array. No markdown code blocks, no other text."""

        try:
            raw = ask(prompt, system_prompt="You are an expert quantitative trader and genetic algorithm architect. Output valid JSON only.")
            cleaned = raw.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            parsed = json.loads(cleaned.strip())
            if isinstance(parsed, list) and len(parsed) >= count:
                return parsed[:count]
        except Exception as e:
            logger.warning(f"Mutation LLM error: {e}. Using deterministic mutation fallback.")

        # Fallback mutation if LLM fails or hits rate limits
        fallbacks = []
        archetypes = [
            ("Apex", "Enhanced Breakout", 1.8, 4.0, 72),
            ("Aegis", "Shielded Trend", 2.2, 4.8, 68),
            ("Pulse", "Adaptive Momentum", 1.6, 3.8, 70),
            ("Sentinel", "Risk-Hardened Scalper", 1.5, 3.2, 75),
        ]
        for i, (prefix, arch, sl, tp, conf) in enumerate(archetypes[:count]):
            fallbacks.append({
                "name": f"{prefix} {champ1.name.split(' ')[0]} (Gen {champ1.generation+1})",
                "archetype": arch,
                "archetype_slug": arch.lower().split()[0],
                "mutation_notes": f"Tuned stop loss to {sl}%, target {tp}%, confidence {conf}% based on graveyard learnings.",
                "system_prompt": f"{champ1.system_prompt}\nCRITICAL DIRECTIVE: Strictly respect stop loss at {sl}% and skip trades without strong volume.",
                "parameters": {
                    "stop_loss_pct": sl,
                    "take_profit_pct": tp,
                    "min_confidence": conf,
                    "max_position_pct": 50,
                },
                "max_drawdown_pct": 3.0,
            })
        return fallbacks

    def _spawn_reboot_cohort(self, total_treasury: float, count: int) -> List[AgentDNA]:
        """If all died, spawn an adaptive reset cohort with lessons embedded."""
        from trading.swarm.agent_dna import create_generation_1_cohort
        capital = round(total_treasury / count, 2)
        cohort = create_generation_1_cohort(capital)
        for a in cohort:
            a.name = f"Phoenix {a.name}"
            a.mutation_notes = "Spawned after full generation wipeout with tightened risk controls."
            a.parameters["stop_loss_pct"] = max(1.2, a.parameters.get("stop_loss_pct", 2.0) * 0.8)
            a.parameters["min_confidence"] = min(78, a.parameters.get("min_confidence", 65) + 5)
        return cohort

    def _record_lineage(self, champion: AgentDNA, next_gen: List[AgentDNA], treasury: float):
        """Append to generation lineage history."""
        lineage = self.get_lineage()
        entry = {
            "generation": champion.generation,
            "next_generation": champion.generation + 1,
            "champion_id": champion.agent_id,
            "champion_name": champion.name,
            "champion_pnl": champion.total_pnl,
            "offspring": [o.name for o in next_gen],
            "treasury": treasury,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        lineage.append(entry)
        LINEAGE_FILE.write_text(json.dumps(lineage, indent=2), encoding="utf-8")
