"""
Darwinian Swarm Dashboard
Interactive Streamlit interface for the Survival of the Fittest AI Agent Arena.
Displays living gladiator agents, health bars, the Graveyard of fallen agents, autopsies,
and generational lineage.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List

from trading.swarm.arena_manager import ArenaManager
from trading.swarm.agent_dna import AgentDNA
from trading.swarm.graveyard import Graveyard
from trading.swarm.evolution_engine import EvolutionEngine


def render_swarm_dashboard():
    st.markdown("## ⚔️ Darwinian AI Agent Arena — Survival of the Fittest")
    st.caption("Multiple AI trading agents compete in generational tournaments. Failing agents self-terminate into the Graveyard; surviving champions inherit the capital and mutate the next generation.")

    # Initialize Arena Manager in session state if not present
    if "arena" not in st.session_state:
        st.session_state.arena = ArenaManager()

    arena: ArenaManager = st.session_state.arena

    # ── Top Control Bar & High-Level Metrics ──────────────────────────────────────
    alive_count = len([a for a in arena.active_agents if a.status != "DEAD"])
    dead_count = len([a for a in arena.active_agents if a.status == "DEAD"])
    total_count = len(arena.active_agents)

    col1, col2, col3, col4, col5 = st.columns([1.5, 1.5, 1.5, 1.5, 2])
    with col1:
        st.metric("🧬 Current Generation", f"Gen {arena.current_generation}")
    with col2:
        st.metric("💰 Master Treasury", f"₹{arena.total_treasury:,.2f}")
    with col3:
        st.metric("🛡️ Living Gladiators", f"{alive_count} / {total_count}")
    with col4:
        st.metric("⚔️ Battle Round", f"#{arena.round_number}")
    with col5:
        st.markdown("**Arena Controls**")
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("⚔️ Scan Round", use_container_width=True, help="Run 1 battle cycle: quotes, signals, AI trades, health check"):
                with st.spinner("Gladiators scanning market and executing trades..."):
                    result = arena.run_market_scan_round()
                    st.session_state.last_round_result = result
                    st.rerun()
        with btn_col2:
            if st.button("🧬 Evolve Gen", use_container_width=True, help="Crown champion, inherit capital, breed next generation"):
                with st.spinner("Evolution engine breeding next generation..."):
                    evo_res = arena.trigger_evolution()
                    st.session_state.last_evo_result = evo_res
                    st.rerun()

    st.markdown("---")

    # Display notice if last round had deaths or exits
    if "last_round_result" in st.session_state:
        res = st.session_state.last_round_result
        if res.get("deaths"):
            for d in res["deaths"]:
                st.error(f"💀 **AGENT CASUALTY**: {d['name']} (Gen {d['generation']}) has perished! Cause: {d['death_reason']}. Moved to Graveyard.")
        if res.get("exits"):
            for ex in res["exits"]:
                st.info(f"🔄 **Position Closed**: {ex['agent']} triggered {ex['type']} on {ex['trade'].get('symbol')} (P&L: ₹{ex['trade'].get('pnl'):+.2f})")

    if "last_evo_result" in st.session_state:
        evo = st.session_state.last_evo_result
        if evo.get("champion"):
            st.success(f"🏆 **Generation Evolved**: Champion **{evo['champion']}** crowned (+₹{evo['champion_pnl']:+.2f})! Inherited Treasury: ₹{evo['new_treasury']:,.2f}. Evolved {len(evo.get('offspring', []))} offspring for Gen {evo['generation']}.")

    # ── Arena Navigation Tabs ───────────────────────────────────────────────────
    tabs = st.tabs([
        "🛡️ Active Gladiators",
        "⚰️ The Graveyard & Autopsies",
        "🌳 Generational Lineage",
        "📜 Negative Knowledge Bank",
        "⚙️ Tournament Settings"
    ])

    # ── TAB 1: Active Gladiators ────────────────────────────────────────────────
    with tabs[0]:
        st.markdown(f"### Generation {arena.current_generation} Cohort")
        if not arena.active_agents:
            st.info("No active agents. Click 'Reset Arena' below to spawn Generation 1.")
        else:
            cols = st.columns(min(4, len(arena.active_agents)))
            for i, agent in enumerate(arena.active_agents):
                with cols[i % len(cols)]:
                    _render_agent_card(agent)

    # ── TAB 2: The Graveyard ────────────────────────────────────────────────────
    with tabs[1]:
        graveyard = arena.graveyard
        dead_agents = graveyard.get_dead_agents()
        st.markdown(f"### ⚰️ Hall of Fallen Agents ({len(dead_agents)} Buried)")
        st.caption("When agents violate drawdown or loss limits, they liquidate positions, commit suicide, and generate a post-mortem autopsy to teach future generations.")

        if not dead_agents:
            st.info("The Graveyard is empty! All gladiators are currently surviving.")
        else:
            for tomb in reversed(dead_agents):
                with st.expander(f"💀 {tomb.get('name')} (Gen {tomb.get('generation')}) — {tomb.get('death_reason')}", expanded=False):
                    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
                    with tcol1:
                        st.metric("Initial Capital", f"₹{tomb.get('initial_capital', 0):,.2f}")
                    with tcol2:
                        st.metric("Final Capital", f"₹{tomb.get('final_capital', 0):,.2f}")
                    with tcol3:
                        st.metric("Total Loss", f"-₹{tomb.get('total_loss', 0):,.2f}", delta_color="inverse")
                    with tcol4:
                        st.caption(f"**Time of Death:**\n{tomb.get('time_of_death')}")

                    st.markdown("#### 🔬 AI Forensic Autopsy")
                    autopsy = tomb.get("autopsy", {})
                    st.markdown(f"**🔴 Fatal Mistake:** {autopsy.get('fatal_mistake', 'N/A')}")
                    st.markdown(f"**🌧️ Hostile Market Conditions:** {autopsy.get('market_condition', 'N/A')}")
                    st.markdown(f"**📜 Permanent Directive for Future Generations:**\n> *\"{autopsy.get('directive', 'N/A')}\"*")

                    trades = tomb.get("trade_history", [])
                    if trades:
                        st.markdown("**Closed Trades Before Death:**")
                        st.dataframe(pd.DataFrame(trades), use_container_width=True)

    # ── TAB 3: Generational Lineage ─────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### 🌳 Darwinian Lineage Tree")
        st.caption("Tracks how champion agents pass their traits, prompts, and capital down through generations.")
        lineage = arena.evolution_engine.get_lineage()
        if not lineage:
            st.info(f"You are currently on **Generation 1**. Once Gen 1 completes and you click 'Evolve Gen', the lineage tree will begin recording ancestors and offspring.")
        else:
            for entry in reversed(lineage):
                st.markdown(f"""
                #### 🧬 Generation {entry.get('generation')} ➔ Generation {entry.get('next_generation')}
                - **🏆 Crowned Champion:** `{entry.get('champion_name')}` (Profit: ₹{entry.get('champion_pnl'):+.2f})
                - **💰 Inherited Treasury:** ₹{entry.get('treasury', 0):,.2f}
                - **👶 Offspring Spawned:** {', '.join(entry.get('offspring', []))}
                - **📅 Date:** {entry.get('timestamp')}
                ---
                """)

    # ── TAB 4: Negative Knowledge Bank ──────────────────────────────────────────
    with tabs[3]:
        st.markdown("### 📜 Shared Memory: Lessons & Immunities")
        st.caption("These lessons are extracted from fallen agents and automatically injected into new agent prompts so history never repeats itself.")
        lessons = arena.graveyard.get_lessons()
        if not lessons:
            st.info("No lessons recorded yet. As agents make mistakes and die, their forensic autopsies populate this memory bank.")
        else:
            st.dataframe(pd.DataFrame(lessons), use_container_width=True)

    # ── TAB 5: Tournament Settings ──────────────────────────────────────────────
    with tabs[4]:
        st.markdown("### ⚙️ Tournament Configuration")
        with st.form("tournament_reset_form"):
            new_treasury = st.number_input("Reset Master Treasury (₹)", min_value=1000.0, max_value=500000.0, value=float(arena.total_treasury), step=1000.0)
            submitted = st.form_submit_button("🚨 Reset Arena to Generation 1")
            if submitted:
                arena.reset_tournament(new_treasury)
                st.success("Tournament completely reset to Generation 1!")
                st.rerun()


def _render_agent_card(agent: AgentDNA):
    """Renders a single gladiator agent card with health bar, P&L, and status."""
    is_dead = agent.status == "DEAD"
    card_border = "border: 1px solid #ff4b4b; background: rgba(255, 75, 75, 0.05);" if is_dead else "border: 1px solid rgba(255, 255, 255, 0.15);"
    
    with st.container():
        # Status icon
        status_icon = "💀 DEAD" if is_dead else ("👑 ALIVE" if agent.total_pnl > 0 else "🛡️ ALIVE")
        st.markdown(f"#### {agent.name}")
        st.caption(f"**{agent.archetype}** • Gen {agent.generation} • `{status_icon}`")

        # Health bar
        hp = agent.health
        health_color = "green" if hp > 60 else ("orange" if hp > 25 else "red")
        st.progress(max(0.0, min(1.0, hp / 100.0)), text=f"HP: {hp}% (Max Drawdown: -{agent.max_drawdown_pct}%)")

        # Financial metrics
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Capital", f"₹{agent.current_capital:,.1f}", delta=f"{agent.total_pnl:+.1f}")
        with c2:
            st.metric("Cash", f"₹{agent.cash:,.1f}")

        # Win/Loss record
        win_rate = (agent.wins_count / agent.trades_count * 100) if agent.trades_count > 0 else 0
        st.caption(f"Trades: {agent.trades_count} | Wins: {agent.wins_count} | Win Rate: {win_rate:.0f}%")

        # Open positions
        if agent.open_positions:
            st.markdown("**Open Positions:**")
            for sym, p in agent.open_positions.items():
                st.code(f"{sym}: {p['quantity']} qty @ ₹{p['buy_price']:.1f}", language="text")
        else:
            st.caption("*No open positions*")

        # Strategy & DNA expander
        with st.expander("🧬 Strategy Genome & Prompt"):
            st.markdown(f"**Stop-Loss:** {agent.parameters.get('stop_loss_pct')}% | **Take-Profit:** {agent.parameters.get('take_profit_pct')}%")
            st.markdown(f"**Min Confidence:** {agent.parameters.get('min_confidence')}%")
            if agent.mutation_notes:
                st.info(f"**Mutation:** {agent.mutation_notes}")
            st.text(agent.system_prompt)

        st.markdown("---")
