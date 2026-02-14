#!/usr/bin/env python3
"""
Realistic Momentum Backtest (Memory Efficient)

Accounts for:
- 2% round-trip spread (1% buy, 1% sell)
- Entry price filters (avoid extremes)
- Orderbook imbalance requirements
- Processes data streaming to avoid OOM
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "ticks"

@dataclass
class Trade:
    coin: str
    entry_price: float
    exit_price: float
    exit_type: str  # "tp", "sl", "time"
    pnl_pct: float  # Before costs
    net_pnl_pct: float  # After spread costs

@dataclass
class BacktestConfig:
    momentum_threshold: float = 0.15
    orderbook_ratio: float = 1.5
    take_profit: float = 0.04
    stop_loss: float = 0.03
    time_stop: int = 70
    max_entry_price: float = 0.90
    min_entry_price: float = 0.10
    spread_cost: float = 0.02
    require_orderbook: bool = True

@dataclass
class MarketState:
    """Track state for a single market."""
    price_history: List[tuple] = field(default_factory=list)  # (ts, price)
    position: bool = False
    entry_price: float = 0
    entry_time: float = 0
    trades: List[Trade] = field(default_factory=list)

def process_file(file_path: Path, config: BacktestConfig, coin: str) -> List[Trade]:
    """Process a single file, streaming line by line."""
    markets: Dict[str, MarketState] = defaultdict(MarketState)
    all_trades = []

    with open(file_path, 'r') as f:
        for line in f:
            try:
                tick = json.loads(line)
            except:
                continue

            # Only trade "up" side
            if tick.get('side') != 'up':
                continue

            market_slug = tick.get('market_slug', 'unknown')
            state = markets[market_slug]

            # Parse timestamp
            ts = tick.get('ts', '')
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                except:
                    continue

            # Get price
            price = tick.get('mid', 0)
            if not price or price <= 0:
                continue
            price = float(price)

            # Get orderbook
            bid_depth = float(tick.get('bid_depth', 0) or 0)
            ask_depth = float(tick.get('ask_depth', 0) or 0)

            # Update price history (keep last 30 seconds)
            state.price_history.append((ts, price))
            cutoff = ts - 30
            state.price_history = [(t, p) for t, p in state.price_history if t >= cutoff]

            if not state.position:
                # Check entry conditions
                if len(state.price_history) < 5:
                    continue

                # Price filter
                if price >= config.max_entry_price or price <= config.min_entry_price:
                    continue

                # Momentum
                old_price = state.price_history[0][1]
                if old_price <= 0:
                    continue
                momentum = (price - old_price) / old_price

                if momentum < config.momentum_threshold:
                    continue

                # Orderbook filter
                if config.require_orderbook:
                    if bid_depth <= 0 or ask_depth <= 0:
                        continue
                    ratio = bid_depth / ask_depth
                    if ratio < config.orderbook_ratio:
                        continue

                # Enter
                state.position = True
                state.entry_price = price
                state.entry_time = ts

            else:
                # Check exit
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
                    net_pnl = pnl_pct - config.spread_cost
                    state.trades.append(Trade(
                        coin=coin,
                        entry_price=state.entry_price,
                        exit_price=price,
                        exit_type=exit_type,
                        pnl_pct=pnl_pct * 100,
                        net_pnl_pct=net_pnl * 100
                    ))
                    state.position = False
                    state.price_history = []

    # Collect trades from all markets
    for state in markets.values():
        all_trades.extend(state.trades)

    return all_trades

def run_backtest(config: BacktestConfig, coins: List[str] = None, days: int = 3) -> Dict:
    """Run backtest across coins."""
    if coins is None:
        coins = ['BTC', 'ETH', 'SOL', 'XRP']

    all_trades = []
    coin_stats = {}

    for coin in coins:
        coin_dir = DATA_DIR / coin.lower()
        if not coin_dir.exists():
            continue

        files = sorted(coin_dir.glob("*.jsonl"))[-days:]
        print(f"  {coin}: processing {len(files)} files...", end=" ", flush=True)

        coin_trades = []
        for file_path in files:
            trades = process_file(file_path, config, coin)
            coin_trades.extend(trades)

        print(f"{len(coin_trades)} trades")

        if coin_trades:
            wins = [t for t in coin_trades if t.net_pnl_pct > 0]
            win_rate = len(wins) / len(coin_trades) * 100
            avg_pnl = sum(t.net_pnl_pct for t in coin_trades) / len(coin_trades)

            coin_stats[coin] = {
                'trades': len(coin_trades),
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': sum(t.net_pnl_pct for t in coin_trades),
                'tp': len([t for t in coin_trades if t.exit_type == 'tp']),
                'sl': len([t for t in coin_trades if t.exit_type == 'sl']),
                'time': len([t for t in coin_trades if t.exit_type == 'time']),
            }

        all_trades.extend(coin_trades)

    # Overall stats
    if all_trades:
        wins = [t for t in all_trades if t.net_pnl_pct > 0]
        total_stats = {
            'trades': len(all_trades),
            'win_rate': len(wins) / len(all_trades) * 100,
            'avg_pnl': sum(t.net_pnl_pct for t in all_trades) / len(all_trades),
            'total_pnl': sum(t.net_pnl_pct for t in all_trades),
            'tp': len([t for t in all_trades if t.exit_type == 'tp']),
            'sl': len([t for t in all_trades if t.exit_type == 'sl']),
            'time': len([t for t in all_trades if t.exit_type == 'time']),
        }
    else:
        total_stats = {'trades': 0, 'win_rate': 0, 'avg_pnl': 0, 'total_pnl': 0, 'tp': 0, 'sl': 0, 'time': 0}

    return {
        'config': config,
        'coin_stats': coin_stats,
        'total': total_stats,
    }

def print_results(results: Dict, label: str = "") -> float:
    """Print backtest results."""
    config = results['config']
    total = results['total']
    coin_stats = results['coin_stats']

    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    print(f"  Momentum: {config.momentum_threshold:.0%}, OB: {config.orderbook_ratio}, "
          f"TP: {config.take_profit:.0%}, SL: {config.stop_loss:.0%}, Time: {config.time_stop}s")
    print(f"  Price: {config.min_entry_price:.0%}-{config.max_entry_price:.0%}, Spread: {config.spread_cost:.0%}")
    print()

    for coin, stats in coin_stats.items():
        status = "PROFIT" if stats['avg_pnl'] > 0 else "LOSS"
        print(f"  {coin}: {stats['trades']:4d} trades | {stats['win_rate']:.1f}% win | "
              f"{stats['avg_pnl']:+.2f}%/trade | {status}")

    print()
    if total['trades'] > 0:
        print(f"  TOTAL: {total['trades']} trades, {total['win_rate']:.1f}% win rate")
        print(f"  Net P&L: {total['avg_pnl']:+.2f}% per trade (after 2% spread)")
        print(f"  Exits: TP={total['tp']}, SL={total['sl']}, Time={total['time']}")

        # Simulated profit
        sim_profit = total['total_pnl'] / 100 * 5.0
        print(f"  Simulated ($5 trades): ${sim_profit:+.2f}")

    return total['avg_pnl'] if total['trades'] > 0 else -999

def main():
    print("="*70)
    print("REALISTIC MOMENTUM BACKTEST")
    print("2% spread cost | 3 days data | Memory efficient")
    print("="*70)

    configs = [
        ("1. Current Settings", BacktestConfig(
            momentum_threshold=0.12, orderbook_ratio=1.5,
            take_profit=0.04, stop_loss=0.03, time_stop=70,
            max_entry_price=0.90, spread_cost=0.02
        )),
        ("2. Higher Momentum (18%)", BacktestConfig(
            momentum_threshold=0.18, orderbook_ratio=1.5,
            take_profit=0.05, stop_loss=0.03, time_stop=70,
            max_entry_price=0.90, spread_cost=0.02
        )),
        ("3. Strict OB (2.0 ratio)", BacktestConfig(
            momentum_threshold=0.15, orderbook_ratio=2.0,
            take_profit=0.05, stop_loss=0.03, time_stop=70,
            max_entry_price=0.90, spread_cost=0.02
        )),
        ("4. Wide TP (6%), Tight SL (2%)", BacktestConfig(
            momentum_threshold=0.15, orderbook_ratio=1.5,
            take_profit=0.06, stop_loss=0.02, time_stop=90,
            max_entry_price=0.90, spread_cost=0.02
        )),
        ("5. Conservative Price (<80%)", BacktestConfig(
            momentum_threshold=0.15, orderbook_ratio=1.5,
            take_profit=0.05, stop_loss=0.03, time_stop=70,
            max_entry_price=0.80, spread_cost=0.02
        )),
        ("6. Aggressive (10% mom, no OB)", BacktestConfig(
            momentum_threshold=0.10, orderbook_ratio=0,
            take_profit=0.04, stop_loss=0.03, time_stop=70,
            max_entry_price=0.90, spread_cost=0.02,
            require_orderbook=False
        )),
        ("7. Best Combo Attempt", BacktestConfig(
            momentum_threshold=0.15, orderbook_ratio=2.0,
            take_profit=0.06, stop_loss=0.025, time_stop=80,
            max_entry_price=0.85, spread_cost=0.02
        )),
        ("8. Very High Mom (25%)", BacktestConfig(
            momentum_threshold=0.25, orderbook_ratio=1.5,
            take_profit=0.06, stop_loss=0.03, time_stop=90,
            max_entry_price=0.90, spread_cost=0.02
        )),
    ]

    results_list = []
    for label, config in configs:
        results = run_backtest(config, days=3)
        avg_pnl = print_results(results, label)
        results_list.append((label, avg_pnl, results))

    # Find best
    profitable = [(l, p, r) for l, p, r in results_list if p > 0 and r['total']['trades'] >= 50]

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    if profitable:
        profitable.sort(key=lambda x: x[1], reverse=True)
        best_label, best_pnl, best_results = profitable[0]
        config = best_results['config']

        print(f"\nBEST: {best_label}")
        print(f"  Net P&L: {best_pnl:+.2f}% per trade after spread")
        print(f"  Trades: {best_results['total']['trades']}")
        print(f"\nRecommended settings:")
        print(f"  --momentum {config.momentum_threshold}")
        print(f"  --tp {config.take_profit}")
        print(f"  --sl {config.stop_loss}")
        print(f"  --time-stop {config.time_stop}")
        print(f"  (also set max_entry_price={config.max_entry_price}, ob_ratio={config.orderbook_ratio})")
    else:
        print("\nNo profitable configuration found.")
        print("Consider:")
        print("  - The strategy may not be viable with current market conditions")
        print("  - Try a different approach (mean reversion, settlement plays)")

if __name__ == "__main__":
    main()
