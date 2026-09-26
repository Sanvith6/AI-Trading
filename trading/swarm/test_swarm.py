"""
Automated End-to-End Test for Darwinian Swarm System
Tests:
1. Arena initialization & Cohort creation
2. Health deduction & Self-termination condition
3. Graveyard burial & Autopsy generation
4. Negative knowledge memory saving
5. Champion selection & Generation 2 breeding
"""

import os
import sys
from pathlib import Path

# Fix Windows console UTF-8 printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading.swarm.agent_dna import create_generation_1_cohort, AgentDNA
from trading.swarm.graveyard import Graveyard
from trading.swarm.evolution_engine import EvolutionEngine
from trading.swarm.arena_manager import ArenaManager


def test_darwinian_swarm():
    print("=== STEP 1: Initializing Arena & Generation 1 Cohort ===")
    arena = ArenaManager(treasury=10000.0)
    arena.reset_tournament(10000.0)
    
    assert len(arena.active_agents) == 4, "Expected 4 agents in Generation 1"
    for a in arena.active_agents:
        assert a.health == 100.0, f"Agent {a.name} should start at 100% HP"
        assert a.current_capital == 2500.0, f"Agent {a.name} should start with ₹2,500"
    print(f"✅ Generation 1 successfully spawned with 4 gladiators. Total Treasury: ₹{arena.total_treasury}")

    print("\n=== STEP 2: Testing Trades & Drawdown Health Tracking ===")
    victim = arena.active_agents[0]
    print(f"Testing on gladiator: {victim.name}")
    
    # Simulate a buy order
    victim.record_buy("RELIANCE", 2500.0, 1, "TEST_ORDER_01")
    assert "RELIANCE" in victim.open_positions
    print(f"Simulated BUY: 1 RELIANCE @ ₹2500. Cash remaining: ₹{victim.cash}")

    # Simulate a losing exit: price drops to 2100 (-16% loss, ₹400 loss, breaching ₹300 limit)
    trade = victim.record_sell("RELIANCE", 2100.0, 1, reason="Simulated Stop-Loss")
    print(f"Simulated SELL: P&L = ₹{trade['pnl']:.2f}")
    assert victim.status == "DEAD", "Agent should have died after breaching ₹300 loss limit"
    assert victim.health == 0.0, "Dead agent should have 0% HP"
    print(f"✅ Self-termination triggered: {victim.name} is {victim.status}. Cause: {victim.death_reason}")

    print("\n=== STEP 3: Testing Graveyard & Autopsy ===")
    tombstone = arena.graveyard.bury_agent(victim)
    dead_agents = arena.graveyard.get_dead_agents()
    assert len(dead_agents) >= 1, "Graveyard should contain at least 1 dead agent"
    print(f"✅ Agent buried in Graveyard. Tombstone: {tombstone['name']}")
    print(f"   Fatal Mistake: {tombstone['autopsy'].get('fatal_mistake')}")
    print(f"   Directive: {tombstone['autopsy'].get('directive')}")

    lessons = arena.graveyard.get_lessons()
    assert len(lessons) >= 1, "Lessons learned bank should be populated"
    print(f"✅ Knowledge Bank updated with {len(lessons)} permanent directive(s).")

    print("\n=== STEP 4: Simulating a Winning Agent ===")
    winner = arena.active_agents[1]
    winner.record_buy("TCS", 3500.0, 1, "WIN_ORDER_01")
    winner.record_sell("TCS", 3850.0, 1, reason="Take Profit Hit (+10%)")
    print(f"Gladiator {winner.name} won +₹{winner.total_pnl:.2f}. Capital: ₹{winner.current_capital}")
    assert winner.total_pnl > 0
    assert winner.health == 100.0

    print("\n=== STEP 5: Testing Evolution Engine (Breeding Gen 2) ===")
    evo_result = arena.trigger_evolution()
    print(f"✅ Generational Evolution complete!")
    print(f"   Status: {evo_result.get('status')}")
    print(f"   Crowned Champion: {evo_result.get('champion')} (+₹{evo_result.get('champion_pnl'):.2f})")
    print(f"   Current Generation: Gen {arena.current_generation}")
    print(f"   Offspring ({len(arena.active_agents)} agents):")
    for a in arena.active_agents:
        print(f"     - {a.name} | Gen {a.generation} | Capital: ₹{a.current_capital} | StopLoss: {a.parameters.get('stop_loss_pct')}%")
        assert a.generation == 2, "All offspring must be Generation 2"
        assert a.parent_id is not None, "Offspring should have parent ID set"

    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY! The Darwinian Swarm is fully operational.")


if __name__ == "__main__":
    test_darwinian_swarm()
