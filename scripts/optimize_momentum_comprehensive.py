#!/usr/bin/env python3
"""
COMPREHENSIVE Momentum Strategy Optimization

Tests ALL dimensions:
1. Entry timing (early momentum vs late momentum)
2. Multi-timeframe confirmation
3. Exit optimization (TP/SL/time stop combinations)
4. Price zone filtering (50% vs 75% has different profit potential)
5. Momentum quality (smooth vs choppy)
6. Spread filtering
7. Time-of-market filtering (early vs late in 15min window)

Goal: Find a TRULY profitable configuration
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime
from collections import defaultdict
import itertools

DATA_DIR = Path(__file__).parent.parent / "data" / "ticks"

@dataclass
class Config:
    # Momentum detection
    momentum_threshold: float = 0.15
    momentum_window: int = 30

    # Multi-timeframe (optional)
    use_multi_timeframe: bool = False
    short_window: int = 10
    short_threshold: float = 0.08

    # Entry filters
    min_entry_price: float = 0.10
    max_entry_price: float = 0.85
    orderbook_ratio: float = 2.0
    max_spread: float = 0.03

    # Price zone optimization
    use_zone_tp: bool = False  # Adjust TP based on entry price

    # Exit parameters
    take_profit: float = 0.06
    stop_loss: float = 0.025
    time_stop: int = 80
    use_trailing_stop: bool = False

    # Momentum quality
    check_momentum_quality: bool = False
    max_volatility: float = 0.05  # Max variance in momentum

    # Time-in-market filter
    min_time_remaining: int = 120  # Don't enter with <2min left
    max_time_in_window: int = 720  # Don't enter in first 12min of 15min market

    # Spread cost
    spread_cost: float = 0.02

@dataclass
class Trade:
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    pnl_pct: float
    net_pnl_pct: float
    exit_type: str
    market_slug: str = ""

def calculate_momentum_quality(prices: List[Tuple[float, float]]) -> float:
    """Calculate how smooth/choppy the momentum is. Lower = smoother."""
    if len(prices) < 3:
        return 0.0

    changes = []
    for i in range(1, len(prices)):
        pct_change = (prices[i][1] - prices[i-1][1]) / prices[i-1][1]
        changes.append(pct_change)

    # Standard deviation of changes
    if not changes:
        return 0.0

    mean = sum(changes) / len(changes)
    variance = sum((x - mean) ** 2 for x in changes) / len(changes)
    return variance ** 0.5

def get_dynamic_tp(entry_price: float, base_tp: float) -> float:
    """Adjust TP based on how much room price has to run."""
    if entry_price <= 0.30:
        # Lots of room - can target higher
        return base_tp * 1.5  # 9% instead of 6%
    elif entry_price >= 0.70:
        # Less room - target lower
        return base_tp * 0.67  # 4% instead of 6%
    else:
        return base_tp

def backtest_config(config: Config, coin: str, days: int = 5) -> Dict:
    """Backtest a single configuration."""
    coin_dir = DATA_DIR / coin.lower()
    if not coin_dir.exists():
        return None

    files = sorted(coin_dir.glob("*.jsonl"), reverse=True)[:days]

    markets = defaultdict(lambda: {
        'prices': [],
        'position': None,
        'trades': [],
        'market_start_time': None,
        'market_end_time': None,
    })

    for file in files:
        with open(file, 'r') as f:
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
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp()
                        except:
                            continue

                price = float(tick.get('mid', 0) or 0)
                if not price or price <= 0:
                    continue

                time_remaining = tick.get('time_remaining', 0)
                bid_depth = float(tick.get('bid_depth', 0) or 0)
                ask_depth = float(tick.get('ask_depth', 0) or 0)
                spread = tick.get('spread', 0) or 0

                # Track market timing
                if state['market_start_time'] is None:
                    state['market_start_time'] = ts
                    state['market_end_time'] = ts + 900  # 15min market

                # Update price history
                state['prices'].append((ts, price))
                cutoff = ts - max(config.momentum_window, config.short_window if config.use_multi_timeframe else 0)
                state['prices'] = [(t, p) for t, p in state['prices'] if t >= cutoff]

                # ENTRY LOGIC
                if not state['position']:
                    if len(state['prices']) < 5:
                        continue

                    # Time-in-market filter
                    if time_remaining < config.min_time_remaining:
                        continue

                    time_in_market = ts - state['market_start_time']
                    if time_in_market > config.max_time_in_window:
                        continue

                    # Price filter
                    if price < config.min_entry_price or price > config.max_entry_price:
                        continue

                    # Spread filter
                    if spread > config.max_spread:
                        continue

                    # Main momentum check
                    prices_window = [(t, p) for t, p in state['prices'] if ts - t <= config.momentum_window]
                    if len(prices_window) < 5:
                        continue

                    old_price = prices_window[0][1]
                    momentum = (price - old_price) / old_price if old_price > 0 else 0

                    if momentum < config.momentum_threshold:
                        continue

                    # Multi-timeframe check
                    if config.use_multi_timeframe:
                        short_prices = [(t, p) for t, p in state['prices'] if ts - t <= config.short_window]
                        if len(short_prices) >= 3:
                            short_old = short_prices[0][1]
                            short_momentum = (price - short_old) / short_old if short_old > 0 else 0
                            if short_momentum < config.short_threshold:
                                continue

                    # Momentum quality check
                    if config.check_momentum_quality:
                        quality = calculate_momentum_quality(prices_window)
                        if quality > config.max_volatility:
                            continue

                    # Orderbook check
                    if ask_depth == 0 or bid_depth / ask_depth < config.orderbook_ratio:
                        continue

                    # ENTER
                    tp = config.take_profit
                    if config.use_zone_tp:
                        tp = get_dynamic_tp(price, config.take_profit)

                    state['position'] = {
                        'entry_price': price,
                        'entry_time': ts,
                        'tp': tp,
                        'highest_price': price,  # For trailing stop
                    }

                # EXIT LOGIC
                elif state['position']:
                    entry_price = state['position']['entry_price']
                    entry_time = state['position']['entry_time']
                    tp = state['position']['tp']
                    hold_time = ts - entry_time

                    # Update highest price for trailing stop
                    if price > state['position']['highest_price']:
                        state['position']['highest_price'] = price

                    pnl_pct = (price - entry_price) / entry_price

                    # Trailing stop logic
                    if config.use_trailing_stop:
                        highest = state['position']['highest_price']
                        drawdown = (highest - price) / highest
                        if drawdown > config.stop_loss:
                            exit_type = "trailing_sl"
                            net_pnl_pct = pnl_pct - config.spread_cost
                            state['trades'].append(Trade(
                                entry_price=entry_price,
                                exit_price=price,
                                entry_time=entry_time,
                                exit_time=ts,
                                pnl_pct=pnl_pct,
                                net_pnl_pct=net_pnl_pct,
                                exit_type=exit_type,
                                market_slug=market_slug
                            ))
                            state['position'] = None
                            continue

                    # Standard exits
                    exit_type = None
                    if pnl_pct >= tp:
                        exit_type = "tp"
                    elif pnl_pct <= -config.stop_loss:
                        exit_type = "sl"
                    elif hold_time >= config.time_stop:
                        exit_type = "time"

                    if exit_type:
                        net_pnl_pct = pnl_pct - config.spread_cost
                        state['trades'].append(Trade(
                            entry_price=entry_price,
                            exit_price=price,
                            entry_time=entry_time,
                            exit_time=ts,
                            pnl_pct=pnl_pct,
                            net_pnl_pct=net_pnl_pct,
                            exit_type=exit_type,
                            market_slug=market_slug
                        ))
                        state['position'] = None

    # Aggregate results
    all_trades = []
    for state in markets.values():
        all_trades.extend(state['trades'])

    if not all_trades:
        return None

    wins = [t for t in all_trades if t.net_pnl_pct > 0]
    losses = [t for t in all_trades if t.net_pnl_pct <= 0]

    return {
        'trades': len(all_trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(all_trades) * 100,
        'avg_pnl': sum(t.net_pnl_pct for t in all_trades) / len(all_trades),
        'total_pnl': sum(t.net_pnl_pct for t in all_trades),
        'avg_win': sum(t.net_pnl_pct for t in wins) / len(wins) if wins else 0,
        'avg_loss': sum(t.net_pnl_pct for t in losses) / len(losses) if losses else 0,
    }

def main():
    print("="*80)
    print("COMPREHENSIVE MOMENTUM STRATEGY OPTIMIZATION")
    print("="*80)
    print("\nTesting BTC (5 days of data)")
    print("This will take a few minutes...\n")

    # Define parameter space to test
    test_configs = []

    # 1. Baseline (current)
    test_configs.append(("Baseline", Config()))

    # 2. Tighter time stop
    for time_stop in [40, 50, 60]:
        cfg = Config(time_stop=time_stop)
        test_configs.append((f"TimeStop={time_stop}s", cfg))

    # 3. Different TP/SL combinations
    for tp, sl in [(0.04, 0.02), (0.05, 0.02), (0.08, 0.03)]:
        cfg = Config(take_profit=tp, stop_loss=sl)
        test_configs.append((f"TP={tp:.0%}_SL={sl:.1%}", cfg))

    # 4. Multi-timeframe confirmation
    cfg = Config(use_multi_timeframe=True, short_window=10, short_threshold=0.08)
    test_configs.append(("Multi-timeframe", cfg))

    # 5. Momentum quality filter
    cfg = Config(check_momentum_quality=True, max_volatility=0.03)
    test_configs.append(("MomentumQuality", cfg))

    # 6. Dynamic TP by price zone
    cfg = Config(use_zone_tp=True)
    test_configs.append(("DynamicTP", cfg))

    # 7. Trailing stop
    cfg = Config(use_trailing_stop=True, stop_loss=0.03)
    test_configs.append(("TrailingStop", cfg))

    # 8. Don't enter late in market window
    cfg = Config(max_time_in_window=600)  # Only first 10min of 15min
    test_configs.append(("EarlyEntry", cfg))

    # 9. Lower momentum threshold (enter earlier)
    cfg = Config(momentum_threshold=0.12, momentum_window=20)
    test_configs.append(("EarlyMomentum_12%", cfg))

    # 10. Higher momentum threshold (enter on stronger signals)
    cfg = Config(momentum_threshold=0.18, momentum_window=30)
    test_configs.append(("LateMomentum_18%", cfg))

    # 11. Combination: Early entry + tighter stops + dynamic TP
    cfg = Config(
        momentum_threshold=0.12,
        momentum_window=20,
        time_stop=50,
        take_profit=0.05,
        stop_loss=0.02,
        use_zone_tp=True,
        max_time_in_window=600
    )
    test_configs.append(("Aggressive_Combo", cfg))

    # 12. Combination: Quality filters + trailing stop
    cfg = Config(
        check_momentum_quality=True,
        max_volatility=0.03,
        use_trailing_stop=True,
        stop_loss=0.03,
        use_multi_timeframe=True
    )
    test_configs.append(("Quality_Combo", cfg))

    # Run all tests
    results = []
    for name, config in test_configs:
        print(f"Testing: {name}...", end=" ")
        result = backtest_config(config, 'BTC', days=5)
        if result:
            result['name'] = name
            result['config'] = config
            results.append(result)
            print(f"{result['trades']} trades, {result['avg_pnl']:+.2%}/trade, {result['win_rate']:.1f}% WR")
        else:
            print("No trades")

    # Sort by avg PnL
    results.sort(key=lambda x: x['avg_pnl'], reverse=True)

    print("\n" + "="*80)
    print("RESULTS (Ranked by Avg P&L)")
    print("="*80)
    print(f"{'Rank':<6}{'Strategy':<25}{'Trades':<10}{'WinRate':<10}{'Avg P&L':<12}{'Total P&L'}")
    print("-"*80)

    for i, r in enumerate(results, 1):
        print(f"{i:<6}{r['name']:<25}{r['trades']:<10}{r['win_rate']:<9.1f}%{r['avg_pnl']:<11.2%}{r['total_pnl']:+.1%}")

    # Best strategy
    if results:
        best = results[0]
        print("\n" + "="*80)
        print("🏆 BEST STRATEGY")
        print("="*80)
        print(f"Name: {best['name']}")
        print(f"\nPerformance:")
        print(f"  Trades: {best['trades']}")
        print(f"  Win rate: {best['win_rate']:.1f}%")
        print(f"  Avg P&L: {best['avg_pnl']:+.2%} per trade")
        print(f"  Total P&L: {best['total_pnl']:+.1%}")
        print(f"  Avg win: {best['avg_win']:+.2%}")
        print(f"  Avg loss: {best['avg_loss']:.2%}")

        cfg = best['config']
        print(f"\nConfiguration:")
        print(f"  Momentum: {cfg.momentum_threshold:.0%} in {cfg.momentum_window}s")
        print(f"  TP/SL: {cfg.take_profit:.1%} / {cfg.stop_loss:.1%}")
        print(f"  Time stop: {cfg.time_stop}s")
        if cfg.use_multi_timeframe:
            print(f"  Multi-timeframe: {cfg.short_threshold:.0%} in {cfg.short_window}s")
        if cfg.check_momentum_quality:
            print(f"  Quality filter: max volatility {cfg.max_volatility:.1%}")
        if cfg.use_zone_tp:
            print(f"  Dynamic TP: enabled")
        if cfg.use_trailing_stop:
            print(f"  Trailing stop: enabled")
        if cfg.max_time_in_window < 720:
            print(f"  Entry window: first {cfg.max_time_in_window}s only")

        improvement = best['avg_pnl'] - results[-1]['avg_pnl']
        print(f"\nImprovement over baseline: {improvement:+.2%} per trade")

if __name__ == '__main__':
    main()
