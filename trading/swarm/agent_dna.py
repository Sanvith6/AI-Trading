"""
Agent DNA & Lifecycle Model
Defines the genome, archetype, health, and state of an evolutionary AI trading agent.
"""

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class AgentDNA:
    agent_id: str
    name: str
    generation: int
    archetype: str
    system_prompt: str
    
    # Capital & Risk
    initial_capital: float = 2500.0
    current_capital: float = 2500.0
    cash: float = 2500.0
    peak_capital: float = 2500.0
    max_drawdown_pct: float = 3.0      # Dies if current capital drops > 3% from peak
    max_loss_amount: float = 300.0     # Dies if total loss exceeds ₹300
    
    # Strategy Parameters (Genome)
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "stop_loss_pct": 2.0,
        "take_profit_pct": 4.0,
        "min_confidence": 65,
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "max_position_pct": 50,         # Max 50% of agent's capital per single trade
    })
    
    # Health & Lifecycle
    health: float = 100.0              # 100.0 down to 0.0
    status: str = "ALIVE"              # ALIVE, DEAD, PROMOTED, RETIRED
    death_reason: Optional[str] = None
    time_of_death: Optional[str] = None
    parent_id: Optional[str] = None
    mutation_notes: Optional[str] = None
    
    # Performance Track Record
    trades_count: int = 0
    wins_count: int = 0
    losses_count: int = 0
    total_pnl: float = 0.0
    consecutive_losses: int = 0
    open_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def update_health(self) -> float:
        """
        Calculates health (0% to 100%) based on drawdown and max loss limits.
        Health = 0 means dead.
        """
        if self.status == "DEAD":
            self.health = 0.0
            return 0.0

        # Drawdown from peak
        if self.peak_capital > 0:
            dd_pct = ((self.peak_capital - self.current_capital) / self.peak_capital) * 100.0
        else:
            dd_pct = 0.0

        # Absolute loss
        loss_amt = max(0.0, self.initial_capital - self.current_capital)

        # Fraction of limit reached
        dd_fraction = min(1.0, max(0.0, dd_pct / self.max_drawdown_pct)) if self.max_drawdown_pct > 0 else 0
        loss_fraction = min(1.0, max(0.0, loss_amt / self.max_loss_amount)) if self.max_loss_amount > 0 else 0
        
        worst_fraction = max(dd_fraction, loss_fraction)
        self.health = round(max(0.0, (1.0 - worst_fraction) * 100.0), 1)

        # Check death conditions
        if self.health <= 0.0 or dd_pct >= self.max_drawdown_pct or loss_amt >= self.max_loss_amount:
            self.health = 0.0
            self.status = "DEAD"
            if dd_pct >= self.max_drawdown_pct:
                self.death_reason = f"Max Drawdown Breached: -{dd_pct:.2f}% (limit: -{self.max_drawdown_pct}%)"
            else:
                self.death_reason = f"Max Loss Breached: -₹{loss_amt:.2f} (limit: -₹{self.max_loss_amount})"
            self.time_of_death = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return self.health

    def record_buy(self, symbol: str, price: float, quantity: int, order_id: str) -> None:
        """Record open position and debit cash."""
        trade_cost = price * quantity
        self.cash -= trade_cost
        self.open_positions[symbol] = {
            "symbol": symbol,
            "quantity": quantity,
            "buy_price": price,
            "cost": trade_cost,
            "order_id": order_id,
            "bought_at": datetime.now().strftime("%H:%M:%S"),
        }
        self.trades_count += 1

    def record_sell(self, symbol: str, sell_price: float, quantity: int, reason: str = "Exit") -> Dict[str, Any]:
        """Record closed position, compute P&L, update capital & health."""
        pos = self.open_positions.pop(symbol, None)
        if not pos:
            return {}

        buy_price = pos["buy_price"]
        proceeds = sell_price * quantity
        cost = buy_price * quantity
        pnl = proceeds - cost
        pnl_pct = (pnl / cost) * 100.0 if cost > 0 else 0.0

        self.cash += proceeds
        self.current_capital += pnl
        self.total_pnl += pnl

        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        if pnl > 0:
            self.wins_count += 1
            self.consecutive_losses = 0
        else:
            self.losses_count += 1
            self.consecutive_losses += 1

        trade_record = {
            "symbol": symbol,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "quantity": quantity,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "reason": reason,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.trade_history.append(trade_record)

        # Re-evaluate health and death status
        self.update_health()
        return trade_record

    def equity(self, live_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Mark-to-market net worth = cash + Σ(open_qty × live_price).
        Falls back to cost basis when no live price is available, so the
        number is never worse than stale — treasury reconciliation uses this.
        """
        total = self.cash
        for sym, pos in (self.open_positions or {}).items():
            px = (live_prices or {}).get(sym) or pos.get("buy_price", 0)
            total += pos.get("quantity", 0) * px
        return round(total, 2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentDNA":
        return cls(**data)


# ── Archetype Factory Functions ───────────────────────────────────────────────

def create_generation_1_cohort(capital_per_agent: float = 2500.0) -> List[AgentDNA]:
    """
    Spawns the initial Generation 1 cohort of 4 distinct AI trader archetypes.
    """
    cohort = [
        # Agent 1: Breakout Hunter
        AgentDNA(
            agent_id="gen1_breakout_01",
            name="Viper (Breakout Hunter)",
            generation=1,
            archetype="Breakout Hunter",
            initial_capital=capital_per_agent,
            current_capital=capital_per_agent,
            cash=capital_per_agent,
            peak_capital=capital_per_agent,
            max_drawdown_pct=3.0,
            max_loss_amount=capital_per_agent * 0.12,  # ₹300 on ₹2500
            parameters={
                "stop_loss_pct": 1.5,
                "take_profit_pct": 3.5,
                "min_confidence": 70,
                "volume_multiplier": 1.3,
                "max_position_pct": 50,
            },
            system_prompt="""You are 'Viper', an aggressive Breakout Hunter AI trader.
YOUR GOAL: Find stocks consolidating near 20-day highs or breaking above resistance with above-average volume.
RULES:
- Enter ONLY when RSI > 55, MACD histogram is expanding, and volume is strong.
- Cut losses fast: Stop loss is strict 1.5%.
- Take profit quickly at 3.5%.
- Do NOT buy if market is falling or stock is choppy without momentum.
Reason aggressively but protect capital: any trade with confidence under 70% must be skipped."""
        ),

        # Agent 2: Mean Reverter
        AgentDNA(
            agent_id="gen1_meanrev_02",
            name="Kite (Mean Reverter)",
            generation=1,
            archetype="Mean Reversion Scalper",
            initial_capital=capital_per_agent,
            current_capital=capital_per_agent,
            cash=capital_per_agent,
            peak_capital=capital_per_agent,
            max_drawdown_pct=3.0,
            max_loss_amount=capital_per_agent * 0.12,
            parameters={
                "stop_loss_pct": 2.0,
                "take_profit_pct": 3.0,
                "min_confidence": 65,
                "rsi_oversold": 35,
                "rsi_overbought": 68,
                "max_position_pct": 50,
            },
            system_prompt="""You are 'Kite', a statistical Mean Reversion Scalper AI trader.
YOUR GOAL: Profit from short-term overextensions and rubber-band snaps back to the mean (20-period SMA).
RULES:
- Buy oversold dips when RSI is below 35 and price touches or dips below the lower Bollinger Band.
- Exit or avoid when RSI > 68 (overbought).
- Stop loss: 2.0%. Profit target: 3.0% (at 20-SMA or mid-band).
- Do NOT catch falling knives on massive catastrophic news. Check if volume is exhausting."""
        ),

        # Agent 3: Trend Follower
        AgentDNA(
            agent_id="gen1_trend_03",
            name="Titan (Trend Follower)",
            generation=1,
            archetype="Trend Follower",
            initial_capital=capital_per_agent,
            current_capital=capital_per_agent,
            cash=capital_per_agent,
            peak_capital=capital_per_agent,
            max_drawdown_pct=3.0,
            max_loss_amount=capital_per_agent * 0.12,
            parameters={
                "stop_loss_pct": 2.5,
                "take_profit_pct": 5.0,
                "min_confidence": 65,
                "ema_short": 9,
                "ema_long": 21,
                "max_position_pct": 50,
            },
            system_prompt="""You are 'Titan', a disciplined Trend Follower AI trader.
YOUR GOAL: Ride strong established trends with high probability. "The trend is your friend."
RULES:
- Enter when EMA 9 is above EMA 21 and price is making higher highs and higher lows.
- Let winners run: Take profit target is 5.0%, with a trailing stop-loss of 2.5%.
- Never trade against the primary trend.
- If the market is sideways or directionless, HOLD and preserve cash."""
        ),

        # Agent 4: Contrarian Sniper
        AgentDNA(
            agent_id="gen1_contrarian_04",
            name="Shadow (Contrarian Sniper)",
            generation=1,
            archetype="Contrarian Sniper",
            initial_capital=capital_per_agent,
            current_capital=capital_per_agent,
            cash=capital_per_agent,
            peak_capital=capital_per_agent,
            max_drawdown_pct=3.0,
            max_loss_amount=capital_per_agent * 0.12,
            parameters={
                "stop_loss_pct": 1.8,
                "take_profit_pct": 4.0,
                "min_confidence": 68,
                "divergence_required": True,
                "max_position_pct": 50,
            },
            system_prompt="""You are 'Shadow', a tactical Contrarian Sniper AI trader.
YOUR GOAL: Exploit crowd panic and euphoric FOMO traps.
RULES:
- Look for bullish divergences (price makes lower low, but RSI makes higher low).
- Fade irrational panic selling when stocks hit major support levels.
- Stop loss: 1.8% (exit immediately if support shatters).
- Take profit: 4.0% bounce.
- Patience is your weapon: if there is no clear trap, do NOT trade."""
        ),
    ]
    return cohort
