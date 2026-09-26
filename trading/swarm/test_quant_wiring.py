"""
Production wiring tests — quant metrics, genetic ops, MTM treasury,
ATR/ADX signals, costed paper fills. Run: python trading/swarm/test_quant_wiring.py
"""

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

os.environ.setdefault("BROKER", "paper")
os.environ.setdefault("TRADE_MODE", "paper")


def test_quant_summary():
    from trading.swarm.quant_metrics import summarize_trades
    trades = [{"pnl": 100}, {"pnl": 50}, {"pnl": -40}, {"pnl": 80}, {"pnl": -30}]
    s = summarize_trades(trades, initial_capital=10000.0)
    assert s["trades"] == 5 and s["wins"] == 3, s
    assert s["profit_factor"] > 1.0, s
    assert isinstance(s["sharpe"], float) and isinstance(s["max_drawdown_pct"], (int, float)), s
    empty = summarize_trades([], initial_capital=10000.0)
    assert empty["trades"] == 0 and empty["sharpe"] == 0.0, empty
    print("✅ summarize_trades:", {k: s[k] for k in ("profit_factor", "sharpe", "sortino", "max_drawdown_pct", "half_kelly_frac")})


def test_genetic_ops_bounds():
    from trading.swarm.genetic_ops import recombine_parameters, enforce_speciation, tournament_selection
    from trading.swarm.agent_dna import create_generation_1_cohort
    p1 = {"stop_loss_pct": 1.5, "take_profit_pct": 3.5, "min_confidence": 70}
    p2 = {"stop_loss_pct": 2.5, "take_profit_pct": 5.0, "min_confidence": 65}
    for _ in range(20):
        child = recombine_parameters(p1, p2)
        assert 1.0 <= child["stop_loss_pct"] <= 3.5, child
        assert 2.0 <= child["take_profit_pct"] <= 7.0, child
        assert 60 <= child["min_confidence"] <= 85, child
    cohort = create_generation_1_cohort(2500.0)
    fmap = {a.agent_id: float(i) for i, a in enumerate(cohort)}
    winner = tournament_selection(cohort, fmap, k=3)
    assert winner is not None
    spec = enforce_speciation(list(cohort))
    assert len({a.archetype for a in spec}) == 4, "speciation must keep 4 niches"
    print("✅ genetic ops: BLX bounds hold, tournament + speciation OK")


def test_mtm_equity():
    from trading.swarm.agent_dna import AgentDNA, create_generation_1_cohort
    a = create_generation_1_cohort(2500.0)[0]
    a.record_buy("RELIANCE", 1000.0, 1, "T1")
    assert a.equity({"RELIANCE": 1100.0}) == round(a.cash + 1100.0, 2)
    assert a.equity({}) == round(a.cash + 1000.0, 2), "fallback to cost basis"
    print("✅ mark-to-market equity OK")


def test_ta_regime():
    from trading.technical_analysis import analyze
    r = analyze("RELIANCE")
    ind = r["indicators"]
    assert "atr" in ind and "adx" in ind and "regime" in ind, ind.keys()
    assert ind["regime"] in ("TRENDING", "WEAK_TREND", "CHOPPY"), ind
    assert "atr_stop_guide" in ind and "atr_target_guide" in ind, ind
    print(f"✅ TA regime: {r['signal']} {r['confidence']}% regime={ind['regime']} ADX={ind['adx']} ATR={ind['atr_pct']}%")


def test_costed_paper_fill():
    from trading.order_manager import place_buy, place_sell, get_paper_portfolio, PAPER_STATE, _round_trip_cost
    if PAPER_STATE.exists():
        PAPER_STATE.unlink()
    assert _round_trip_cost(1000.0, 4) > 0
    b = place_buy("RELIANCE", price=1200.0, quantity=2)
    assert b["status"] == "paper_executed", b
    pf = get_paper_portfolio()
    assert pf["holdings"] and pf["cash"] < 100000.0 - 2400.0, pf  # costs deducted
    s = place_sell("RELIANCE", 2, price=1210.0)
    assert s["status"] == "paper_executed", s
    if PAPER_STATE.exists():
        PAPER_STATE.unlink()
    print("✅ costed paper fills (buy+ sell legs charged)")


if __name__ == "__main__":
    test_quant_summary()
    test_genetic_ops_bounds()
    test_mtm_equity()
    test_ta_regime()
    test_costed_paper_fill()
    print("\n🎉 ALL WIRING TESTS PASSED")
