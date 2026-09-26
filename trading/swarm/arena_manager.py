"""
Arena Manager — Darwinian Multi-Agent Tournament Coordinator
Manages live agent execution, evaluation rounds, capital pools, self-termination triggers,
and generational progression.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from trading.swarm.agent_dna import AgentDNA, create_generation_1_cohort
from trading.swarm.graveyard import Graveyard
from trading.swarm.evolution_engine import EvolutionEngine
from trading.data_provider import get_watchlist_quotes, quotes_to_map
from trading.technical_analysis import generate_signal
from llm_client import ask

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).parent / "swarm_state.json"
DEFAULT_TREASURY = float(os.getenv("SWARM_TOTAL_TREASURY", 10000.0))


class ArenaManager:
    """
    Coordinates the Darwinian trading arena.
    """

    def __init__(self, treasury: float = DEFAULT_TREASURY):
        self.graveyard = Graveyard()
        self.evolution_engine = EvolutionEngine(self.graveyard)
        self.total_treasury = treasury
        self.current_generation = 1
        self.round_number = 0
        self.active_agents: List[AgentDNA] = []
        self.mode = os.getenv("TRADE_MODE", "paper")
        self.load_or_init()

    def load_or_init(self) -> None:
        """Load persistent swarm state or initialize Generation 1."""
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                self.current_generation = data.get("current_generation", 1)
                self.round_number = data.get("round_number", 0)
                self.total_treasury = data.get("total_treasury", DEFAULT_TREASURY)
                self.active_agents = [AgentDNA.from_dict(a) for a in data.get("active_agents", [])]
                if self.active_agents:
                    return
            except Exception as e:
                logger.error(f"Failed to load swarm state: {e}. Reinitializing.")

        # Fresh Generation 1 initialization
        self.reset_tournament(self.total_treasury)

    def save_state(self) -> None:
        """Persist current arena state to disk."""
        data = {
            "current_generation": self.current_generation,
            "round_number": self.round_number,
            "total_treasury": round(self.total_treasury, 2),
            "mode": self.mode,
            "active_agents": [a.to_dict() for a in self.active_agents],
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def reset_tournament(self, treasury: float = DEFAULT_TREASURY) -> None:
        """Reset the tournament completely to Generation 1."""
        self.total_treasury = treasury
        self.current_generation = 1
        self.round_number = 0
        capital_per_agent = round(treasury / 4.0, 2)
        self.active_agents = create_generation_1_cohort(capital_per_agent)
        self.graveyard.clear()
        self.save_state()
        logger.info(f"🏟️ Arena Reset: Generation 1 spawned with 4 agents, ₹{capital_per_agent} each.")

    def run_market_scan_round(self, watchlist: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Executes one battle round:
        1. Fetches real market prices and technical signals for watchlist
        2. Monitors all open positions for stop-loss and take-profit hits
        3. Each living agent evaluates signals and decides BUY / SELL / HOLD
        4. Rechecks health & executes self-termination / graveyard burials
        5. Updates treasury & persists state
        """
        if watchlist is None:
            wl_env = os.getenv("TRADE_WATCHLIST", "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK")
            watchlist = [s.strip() for s in wl_env.split(",") if s.strip()]

        self.round_number += 1
        round_log = {
            "round": self.round_number,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "decisions": [],
            "exits": [],
            "deaths": [],
        }

        # Step 1: Fetch quotes & technical analysis (free provider; dict keyed by symbol)
        quotes_list = get_watchlist_quotes(watchlist)
        quotes = quotes_to_map(quotes_list)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(watchlist), 16)) as executor:
            sig_list = list(executor.map(generate_signal, watchlist))
        signals = dict(zip(watchlist, sig_list))

        # Step 2: Check open positions for exits across all active agents
        for agent in self.active_agents:
            if agent.status == "DEAD":
                continue

            for symbol, pos in list(agent.open_positions.items()):
                live = quotes.get(symbol, {}) or {}
                current_price = live.get("last_price") or pos["buy_price"]
                buy_price = pos["buy_price"]
                change_pct = ((current_price - buy_price) / buy_price) * 100.0

                sl_pct = agent.parameters.get("stop_loss_pct", 2.0)
                tp_pct = agent.parameters.get("take_profit_pct", 4.0)

                # Stop loss exit
                if change_pct <= -sl_pct:
                    record = agent.record_sell(symbol, current_price, pos["quantity"], reason=f"Stop-Loss Hit ({change_pct:.2f}%)")
                    round_log["exits"].append({"agent": agent.name, "trade": record, "type": "STOP_LOSS"})

                # Take profit exit
                elif change_pct >= tp_pct:
                    record = agent.record_sell(symbol, current_price, pos["quantity"], reason=f"Take-Profit Hit (+{change_pct:.2f}%)")
                    round_log["exits"].append({"agent": agent.name, "trade": record, "type": "TAKE_PROFIT"})

        # Step 3: Each living agent reasons and executes new entries
        for agent in self.active_agents:
            if agent.status == "DEAD":
                continue

            # Update health based on current equity
            agent.update_health()
            if agent.status == "DEAD":
                tombstone = self.graveyard.bury_agent(agent)
                round_log["deaths"].append(tombstone)
                continue

            # Let agent decide on opportunities
            decision = self._agent_reason_and_act(agent, quotes, signals)
            round_log["decisions"].append(decision)

            # Re-check health after trade action
            agent.update_health()
            if agent.status == "DEAD":
                tombstone = self.graveyard.bury_agent(agent)
                round_log["deaths"].append(tombstone)

        # Step 4: Reconcile Treasury (mark-to-market on this round's quotes)
        live_px = {s: (q.get("last_price") or 0) for s, q in quotes.items()}
        self._reconcile_treasury(live_px)
        self.save_state()
        return round_log

    def _agent_reason_and_act(
        self,
        agent: AgentDNA,
        quotes: Dict[str, Any],
        signals: Dict[str, str]
    ) -> Dict[str, Any]:
        """Runs the LLM reasoning process for a single agent."""
        market_overview = []
        for sym, q in quotes.items():
            market_overview.append(f"{sym}: ₹{q.get('last_price', 'N/A')} ({q.get('change_pct', 0):+.2f}%) | Signal: {signals.get(sym, 'N/A')[:60]}...")
        market_text = "\n".join(market_overview)

        prompt = f"""You are AI Trader: {agent.name}
Archetype: {agent.archetype}
Current Health: {agent.health}% | Cash Available: ₹{agent.cash:.2f} | P&L: ₹{agent.total_pnl:+.2f}
Open Positions: {list(agent.open_positions.keys()) if agent.open_positions else 'None'}
Risk Parameters: StopLoss={agent.parameters.get('stop_loss_pct')}% | TakeProfit={agent.parameters.get('take_profit_pct')}% | MinConfidence={agent.parameters.get('min_confidence')}%

LIVE MARKET OVERVIEW:
{market_text}

YOUR MISSION:
Review the market based on your unique archetype rules. Decide whether to BUY ONE STOCK, SELL an open position, or HOLD.

Format your decision in STRICT JSON with keys:
- "action": "BUY", "SELL", or "HOLD"
- "symbol": Stock symbol (or null if HOLD)
- "confidence": Integer 0 to 100
- "reasoning": 1-2 sentence justification matching your archetype

Respond ONLY with the JSON object."""

        try:
            raw = ask(prompt, system_prompt=agent.system_prompt)
            cleaned = raw.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            res = json.loads(cleaned.strip())
        except Exception as e:
            res = {"action": "HOLD", "symbol": None, "confidence": 0, "reasoning": f"Observation cycle. Error: {str(e)[:50]}"}

        action = res.get("action", "HOLD").upper()
        symbol = res.get("symbol")
        confidence = res.get("confidence", 0)

        # Execute if valid BUY
        if action == "BUY" and symbol and symbol in quotes and confidence >= agent.parameters.get("min_confidence", 65):
            price = quotes[symbol].get("last_price", 0)
            if price > 0 and symbol not in agent.open_positions:
                max_pos_val = agent.cash * (agent.parameters.get("max_position_pct", 50) / 100.0)
                qty = int(max_pos_val // price)
                if qty >= 1:
                    order_id = f"SWARM_{agent.agent_id}_{int(datetime.now().timestamp())}"
                    agent.record_buy(symbol, price, qty, order_id)
                    res["executed"] = True
                    res["quantity"] = qty
                    res["price"] = price
                else:
                    res["executed"] = False
                    res["reasoning"] += " (Insufficient cash for 1 share)"
            else:
                res["executed"] = False
        else:
            res["executed"] = False

        res["agent_id"] = agent.agent_id
        res["agent_name"] = agent.name
        return res

    def trigger_evolution(self) -> Dict[str, Any]:
        """
        Manually or automatically ends the current generation round:
        1. Selects surviving champion
        2. Kills/retires non-qualifying agents
        3. Inherits all accumulated capital
        4. Breeds Generation N+1
        """
        champions = self.evolution_engine.select_champions(self.active_agents, top_n=2)
        if not champions:
            # If all are dead, reboot with fresh generation
            self.active_agents = self.evolution_engine.breed_next_generation(
                self.active_agents, self.total_treasury, target_swarm_size=4
            )
            self.current_generation += 1
            self.save_state()
            return {"status": "rebooted", "generation": self.current_generation, "champions": []}

        champ = champions[0]
        logger.info(f"🏆 Champion Crowned: {champ.name} (P&L: ₹{champ.total_pnl:+.2f})")

        # Terminate or harvest non-champions that underperformed
        for a in self.active_agents:
            if a.agent_id != champ.agent_id and a.status != "DEAD":
                if a.total_pnl < 0:
                    self.graveyard.bury_agent(a, reason="Eliminated in tournament selection (negative fitness)")

        # Re-pool all remaining cash & profits into treasury
        self._reconcile_treasury()

        # Breed next generation
        next_cohort = self.evolution_engine.breed_next_generation(
            self.active_agents, self.total_treasury, target_swarm_size=4
        )

        self.current_generation += 1
        self.active_agents = next_cohort
        self.save_state()

        return {
            "status": "evolved",
            "generation": self.current_generation,
            "champion": champ.name,
            "champion_pnl": champ.total_pnl,
            "new_treasury": self.total_treasury,
            "offspring": [a.name for a in next_cohort],
        }

    def _reconcile_treasury(self, live_prices: Optional[Dict[str, float]] = None):
        """Net worth of the swarm, marked to market (not stale cost basis)."""
        if live_prices is None:
            try:
                live_prices = {q.get("symbol", "").upper(): q.get("last_price", 0)
                               for q in get_watchlist_quotes()}
            except Exception:
                live_prices = {}
        total = round(sum(a.equity(live_prices) for a in self.active_agents), 2)
        # Also add unallocated cash
        self.total_treasury = max(1000.0, total)
