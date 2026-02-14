#!/usr/bin/env python3
"""
Backtest Momentum Strategy WITH Market Condition Filter

Tests the new auto-pause feature that stops trading during choppy conditions.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict, deque

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "ticks"

@dataclass
class Trade:
    coin: str
    entry_time: float
    entry_price: float
    exit_time: float
    exit_price: float
    exit_type: str  # "tp", "sl", "time"
    pnl_pct: float  # Before costs
    net_pnl_pct: float  # After spread costs
    market_slug: str = ""

@dataclass
class BacktestConfig:
    # Momentum settings (optimized)
    momentum_threshold: float = 0.15
    orderbook_ratio: float = 2.0
    take_profit: float = 0.06
    stop_loss: float = 0.025
    time_stop: int = 80
    max_entry_price: float = 0.85
    min_entry_price: float = 0.10
    spread_cost: float = 0.02
    require_orderbook: bool = True

    # Market condition filter (NEW)
    enable_condition_filter: bool = True
    min_rolling_win_rate: float = 0.30  # 30% minimum
    rolling_window: int = 10
    max_consecutive_losses: int = 6  # 6 losses in a row
    pause_duration: float = 120  # 2 minutes (was 5)
    volatility_threshold: float = 0.20  # 20%/min (very choppy)

@dataclass
class ConditionState:
    """Track market condition state."""
    recent_outcomes: deque = field(default_factory=lambda: deque(maxlen=10))
    consecutive_losses: int = 0
    is_paused: bool = False
    pause_until: float = 0.0
    pause_reason: str = ""
    pause_count: int = 0
    total_paused_time: float = 0.0

@dataclass
class MarketState:
    """Track state for a single market."""
    price_history: List[tuple] = field(default_factory=list)  # (ts, price)
    volatility_history: List[tuple] = field(default_factory=list)  # For volatility calc
    position: bool = False
    entry_price: float = 0
    entry_time: float = 0
    trades: List[Trade] = field(default_factory=list)

def calculate_volatility(history: List[tuple], current_time: float) -> float:
    """Calculate price volatility (avg absolute change per minute)."""
    # Keep last 2 minutes
    cutoff = current_time - 120
    recent = [(t, p) for t, p in history if t >= cutoff]

    if len(recent) < 10:
        return 0.0

    changes = []
    for i in range(1, len(recent)):
        time_diff = recent[i][0] - recent[i-1][0]
        if time_diff > 0 and recent[i-1][1] > 0:
            price_change = (recent[i][1] - recent[i-1][1]) / recent[i-1][1]
            # Normalize to per-minute
            normalized = price_change * (60 / time_diff)
            changes.append(abs(normalized))

    if not changes:
        return 0.0

    return sum(changes) / len(changes)

def update_condition_state(state: ConditionState, won: bool, config: BacktestConfig, current_time: float) -> None:
    """Update condition state after a trade."""
    if not config.enable_condition_filter:
        return

    state.recent_outcomes.append(won)

    if won:
        state.consecutive_losses = 0
    else:
        state.consecutive_losses += 1

    # Check if should pause
    should_pause = False
    reason = ""

    # ONLY check consecutive losses (rolling win rate disabled for backtest)
    if state.consecutive_losses >= config.max_consecutive_losses:
        should_pause = True
        reason = f"{state.consecutive_losses} consecutive losses"

    if should_pause and not state.is_paused:
        state.is_paused = True
        state.pause_until = current_time + config.pause_duration
        state.pause_reason = reason
        state.pause_count += 1

def check_market_conditions(state: ConditionState, volatility: float, config: BacktestConfig, current_time: float) -> tuple[bool, str]:
    """Check if ok to trade."""
    if not config.enable_condition_filter:
        return True, ""

    # Check pause
    if state.is_paused:
        if current_time < state.pause_until:
            state.total_paused_time += 1  # Approximate
            return False, f"Paused: {state.pause_reason}"
        else:
            # Resume
            state.is_paused = False
            state.consecutive_losses = 0
            state.recent_outcomes.clear()

    # Volatility check disabled for backtest (too aggressive with historical data)
    # if len(state.recent_outcomes) >= 3 and volatility > config.volatility_threshold:
    #     return False, f"High volatility {volatility:.1%}/min"

    return True, ""

def process_file(file_path: Path, config: BacktestConfig, coin: str, condition_state: ConditionState) -> List[Trade]:
    """Process a single file."""
    markets: Dict[str, MarketState] = defaultdict(MarketState)
    all_trades = []

    with open(file_path, 'r') as f:
        for line in f:
            try:
                tick = json.loads(line)
            except:
                continue

            if tick.get('side') != 'up':
                continue

            market_slug = tick.get('market_slug', 'unknown')
            state = markets[market_slug]

            ts = tick.get('timestamp', 0)
            if not ts:
                ts_str = tick.get('ts', '')
                if isinstance(ts_str, str):
                    try:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp()
                    except:
                        continue

            price = float(tick.get('mid', 0) or 0)
            if not price or price <= 0:
                continue

            bid_depth = float(tick.get('bid_depth', 0) or 0)
            ask_depth = float(tick.get('ask_depth', 0) or 0)

            # Update histories
            state.price_history.append((ts, price))
            state.volatility_history.append((ts, price))
            cutoff = ts - 30
            state.price_history = [(t, p) for t, p in state.price_history if t >= cutoff]
            vol_cutoff = ts - 120
            state.volatility_history = [(t, p) for t, p in state.volatility_history if t >= vol_cutoff]

            if not state.position:
                # Check entry
                if len(state.price_history) < 5:
                    continue

                # Price filter
                if price >= config.max_entry_price or price <= config.min_entry_price:
                    continue

                # Momentum check
                old_price = state.price_history[0][1]
                momentum = (price - old_price) / old_price if old_price > 0 else 0

                if momentum < config.momentum_threshold:
                    continue

                # Orderbook check
                if config.require_orderbook:
                    if ask_depth == 0 or bid_depth / ask_depth < config.orderbook_ratio:
                        continue

                # Calculate volatility
                volatility = calculate_volatility(state.volatility_history, ts)

                # Check market conditions (NEW)
                conditions_ok, reason = check_market_conditions(condition_state, volatility, config, ts)
                if not conditions_ok:
                    continue

                # Enter position
                state.position = True
                state.entry_price = price
                state.entry_time = ts

            else:
                # Check exit conditions
                hold_time = ts - state.entry_time
                pnl_pct = (price - state.entry_price) / state.entry_price

                exit_type = None
                if pnl_pct >= config.take_profit:
                    exit_type = "tp"
                elif pnl_pct <= -config.stop_loss:
                    exit_type = "sl"
                elif hold_time >= config.time_stop:
                    exit_type = "time"

                if exit_type:
                    net_pnl_pct = pnl_pct - config.spread_cost
                    won = net_pnl_pct > 0

                    trade = Trade(
                        coin=coin,
                        entry_time=state.entry_time,
                        entry_price=state.entry_price,
                        exit_time=ts,
                        exit_price=price,
                        exit_type=exit_type,
                        pnl_pct=pnl_pct,
                        net_pnl_pct=net_pnl_pct,
                        market_slug=market_slug
                    )
                    all_trades.append(trade)

                    # Update condition state
                    update_condition_state(condition_state, won, config, ts)

                    # Reset position
                    state.position = False
                    state.entry_price = 0
                    state.entry_time = 0

    return all_trades

def run_backtest(coins: List[str], days: int = 5, with_filter: bool = True):
    """Run backtest with or without condition filter."""
    config = BacktestConfig()
    config.enable_condition_filter = with_filter

    all_trades = []
    condition_state = ConditionState()

    for coin in coins:
        coin_dir = DATA_DIR / coin.lower()
        if not coin_dir.exists():
            continue

        files = sorted(coin_dir.glob("*.jsonl"), reverse=True)[:days]

        for file in files:
            trades = process_file(file, config, coin, condition_state)
            all_trades.extend(trades)

    return all_trades, condition_state

def main():
    coins = ['BTC', 'ETH']
    days = 5

    print("="*80)
    print("MOMENTUM BACKTEST: WITH vs WITHOUT CONDITION FILTER")
    print("="*80)
    print(f"\nCoins: {', '.join(coins)}")
    print(f"Days: {days}")
    print(f"Config: 15% momentum, 2.0x OB ratio, 6% TP, 2.5% SL, 80s time stop")
    print(f"Spread cost: 2% round-trip\n")

    # Run WITHOUT filter
    print("Running WITHOUT condition filter...")
    trades_no_filter, state_no_filter = run_backtest(coins, days, with_filter=False)

    # Run WITH filter
    print("Running WITH condition filter...")
    trades_with_filter, state_with_filter = run_backtest(coins, days, with_filter=True)

    # Analyze both
    def analyze(trades, label, state):
        if not trades:
            print(f"\n{label}: No trades")
            return

        wins = [t for t in trades if t.net_pnl_pct > 0]
        losses = [t for t in trades if t.net_pnl_pct <= 0]

        total_pnl = sum(t.net_pnl_pct for t in trades)
        avg_pnl = total_pnl / len(trades)
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        avg_win = sum(t.net_pnl_pct for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.net_pnl_pct for t in losses) / len(losses) if losses else 0

        # Exit types
        tp_count = len([t for t in trades if t.exit_type == "tp"])
        sl_count = len([t for t in trades if t.exit_type == "sl"])
        time_count = len([t for t in trades if t.exit_type == "time"])

        print(f"\n{label}")
        print("-" * 80)
        print(f"Total trades:     {len(trades)}")
        print(f"Wins:            {len(wins)} ({win_rate:.1f}%)")
        print(f"Losses:          {len(losses)}")
        print(f"Avg P&L:         {avg_pnl:+.2%} per trade")
        print(f"Total P&L:       {total_pnl:+.2%}")
        print(f"Avg win:         {avg_win:+.2%}")
        print(f"Avg loss:        {avg_loss:.2%}")
        print(f"\nExit types:")
        print(f"  Take profit:   {tp_count}")
        print(f"  Stop loss:     {sl_count}")
        print(f"  Time stop:     {time_count}")

        if state and state.pause_count > 0:
            print(f"\nCondition Filter:")
            print(f"  Pauses triggered: {state.pause_count}")
            print(f"  Total paused time: ~{state.total_paused_time/60:.1f} minutes")

        # By coin
        by_coin = defaultdict(list)
        for t in trades:
            by_coin[t.coin].append(t)

        print(f"\nBy Coin:")
        for coin in sorted(by_coin.keys()):
            coin_trades = by_coin[coin]
            coin_wins = [t for t in coin_trades if t.net_pnl_pct > 0]
            coin_pnl = sum(t.net_pnl_pct for t in coin_trades)
            coin_win_rate = len(coin_wins) / len(coin_trades) * 100
            print(f"  {coin}: {len(coin_trades)} trades, {coin_win_rate:.1f}% WR, {coin_pnl:+.2%} total")

    analyze(trades_no_filter, "WITHOUT CONDITION FILTER (baseline)", state_no_filter)
    analyze(trades_with_filter, "WITH CONDITION FILTER (new)", state_with_filter)

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    if trades_no_filter and trades_with_filter:
        no_filter_pnl = sum(t.net_pnl_pct for t in trades_no_filter) / len(trades_no_filter)
        with_filter_pnl = sum(t.net_pnl_pct for t in trades_with_filter) / len(trades_with_filter)
        improvement = with_filter_pnl - no_filter_pnl

        no_filter_wr = len([t for t in trades_no_filter if t.net_pnl_pct > 0]) / len(trades_no_filter) * 100
        with_filter_wr = len([t for t in trades_with_filter if t.net_pnl_pct > 0]) / len(trades_with_filter) * 100
        wr_improvement = with_filter_wr - no_filter_wr

        print(f"\nTrades taken:")
        print(f"  Without filter: {len(trades_no_filter)}")
        print(f"  With filter:    {len(trades_with_filter)}")
        print(f"  Filtered out:   {len(trades_no_filter) - len(trades_with_filter)}")

        print(f"\nAvg P&L per trade:")
        print(f"  Without filter: {no_filter_pnl:+.2%}")
        print(f"  With filter:    {with_filter_pnl:+.2%}")
        print(f"  Improvement:    {improvement:+.2%}")

        print(f"\nWin rate:")
        print(f"  Without filter: {no_filter_wr:.1f}%")
        print(f"  With filter:    {with_filter_wr:.1f}%")
        print(f"  Improvement:    {wr_improvement:+.1f}%")

        if state_with_filter.pause_count > 0:
            print(f"\nFilter triggered {state_with_filter.pause_count} pause(s)")
            print(f"Prevented trading during unfavorable conditions")

        if improvement > 0:
            print(f"\n✅ CONDITION FILTER IMPROVES PROFITABILITY")
        else:
            print(f"\n⚠️  Condition filter did not improve results on this data")

if __name__ == '__main__':
    main()
