"""
Agent Graveyard & Autopsy System
Handles self-termination, emergency position liquidation, post-mortem autopsies,
and archives fatal mistakes into the negative memory bank for future generations.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from trading.swarm.agent_dna import AgentDNA
from llm_client import ask

logger = logging.getLogger(__name__)

GRAVEYARD_FILE = Path(__file__).parent / "graveyard.json"
LESSONS_FILE   = Path(__file__).parent / "lessons_learned.json"


class Graveyard:
    """
    Manages the cemetery of fallen agents and extracts lessons from their deaths.
    """

    def __init__(self):
        self._ensure_files()

    def _ensure_files(self):
        if not GRAVEYARD_FILE.exists():
            GRAVEYARD_FILE.write_text("[]", encoding="utf-8")
        if not LESSONS_FILE.exists():
            LESSONS_FILE.write_text("[]", encoding="utf-8")

    def get_dead_agents(self) -> List[Dict[str, Any]]:
        """Retrieve all buried agents."""
        try:
            return json.loads(GRAVEYARD_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []

    def get_lessons(self) -> List[Dict[str, Any]]:
        """Retrieve all accumulated lessons from fallen agents."""
        try:
            return json.loads(LESSONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []

    def bury_agent(self, agent: AgentDNA, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Officially terminates an agent:
        1. Marks as DEAD
        2. Closes any lingering open positions
        3. Generates an AI Autopsy / Post-Mortem
        4. Archives lessons to lessons_learned.json
        5. Saves to graveyard.json
        Returns the autopsy record and remaining cash to return to treasury.
        """
        agent.status = "DEAD"
        agent.health = 0.0
        if reason:
            agent.death_reason = reason
        if not agent.time_of_death:
            agent.time_of_death = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.warning(f"⚰️ AGENT TERMINATED: {agent.name} (Gen {agent.generation}) | Reason: {agent.death_reason}")

        # Emergency liquidation of any remaining open positions
        liquidated_trades = []
        for symbol, pos in list(agent.open_positions.items()):
            # Use buy price with an assumed slippage/market exit if needed
            exit_price = pos["buy_price"] * (1 - (agent.parameters.get("stop_loss_pct", 2.0) / 100.0))
            record = agent.record_sell(symbol, exit_price, pos["quantity"], reason="Emergency Graveyard Liquidation")
            liquidated_trades.append(record)

        # Generate AI Autopsy
        autopsy = self._generate_autopsy(agent)

        # Package the tombstone record
        tombstone = {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "generation": agent.generation,
            "archetype": agent.archetype,
            "initial_capital": agent.initial_capital,
            "final_capital": round(agent.current_capital, 2),
            "remaining_cash": round(agent.cash, 2),
            "total_loss": round(agent.initial_capital - agent.current_capital, 2),
            "total_pnl": round(agent.total_pnl, 2),
            "trades_count": agent.trades_count,
            "wins_count": agent.wins_count,
            "losses_count": agent.losses_count,
            "death_reason": agent.death_reason,
            "time_of_death": agent.time_of_death,
            "parent_id": agent.parent_id,
            "autopsy": autopsy,
            "trade_history": agent.trade_history,
        }

        # Save to Graveyard
        dead_agents = self.get_dead_agents()
        dead_agents.append(tombstone)
        GRAVEYARD_FILE.write_text(json.dumps(dead_agents, indent=2), encoding="utf-8")

        # Save lesson
        if autopsy.get("fatal_mistake"):
            lessons = self.get_lessons()
            lessons.append({
                "agent_name": agent.name,
                "generation": agent.generation,
                "archetype": agent.archetype,
                "fatal_mistake": autopsy.get("fatal_mistake"),
                "directive": autopsy.get("directive"),
                "timestamp": agent.time_of_death,
            })
            LESSONS_FILE.write_text(json.dumps(lessons, indent=2), encoding="utf-8")

        return tombstone

    def _generate_autopsy(self, agent: AgentDNA) -> Dict[str, str]:
        """
        Calls the LLM to perform an autopsy on the agent's trades and generate actionable lessons.
        """
        history_summary = json.dumps(agent.trade_history[-5:], indent=1) if agent.trade_history else "No closed trades recorded."
        
        prompt = f"""You are the Chief Forensic Analyst for a Darwinian AI Trading Swarm.
An AI trading agent has DIED and failed its performance criteria.

Agent Details:
- Name: {agent.name}
- Generation: {agent.generation}
- Archetype: {agent.archetype}
- Cause of Death: {agent.death_reason}
- Initial Capital: ₹{agent.initial_capital}
- Final Capital: ₹{agent.current_capital} (P&L: ₹{agent.total_pnl})
- Recent Trades:
{history_summary}

Original Agent Strategy Prompt:
"{agent.system_prompt[:300]}..."

TASK:
Analyze why this agent failed. Provide your diagnosis in strict JSON format with exactly three fields:
1. "fatal_mistake": A 1-2 sentence forensic summary of the exact flaw in its logic, timing, or risk management.
2. "market_condition": What kind of market environment trapped and eliminated this agent.
3. "directive": A clear, enforceable rule that future generations must inherit so they never repeat this death.

Respond ONLY with valid JSON. No markdown, no preface."""

        try:
            raw_response = ask(prompt, system_prompt="You are a strict financial forensic investigator. Respond in valid JSON only.")
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            data = json.loads(cleaned.strip())
            return {
                "fatal_mistake": data.get("fatal_mistake", "Excessive drawdown due to repeated counter-trend entries."),
                "market_condition": data.get("market_condition", "Adverse volatility or choppy sideways chop."),
                "directive": data.get("directive", "Tighter stop losses and mandatory confirmation before entry."),
            }
        except Exception as e:
            logger.warning(f"Autopsy LLM error: {e}. Using fallback autopsy.")
            return {
                "fatal_mistake": f"Agent failed due to: {agent.death_reason}. Losses compounded beyond allowable threshold.",
                "market_condition": "Adverse market momentum with lack of trend follow-through.",
                "directive": "Enforce stricter confirmation signals and immediate stop-loss exit.",
            }

    def clear(self):
        """Reset the graveyard (e.g. For starting a fresh tournament)."""
        GRAVEYARD_FILE.write_text("[]", encoding="utf-8")
        LESSONS_FILE.write_text("[]", encoding="utf-8")
