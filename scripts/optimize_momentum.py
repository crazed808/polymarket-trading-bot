#!/usr/bin/env python3
"""
Momentum Strategy Parameter Optimizer

Runs a grid search over key parameters to find profitable configurations.
Uses recorded tick data with realistic bid/ask pricing.

Parameters optimized:
- TP/SL ratios (overcome spread costs)
- Spread filters (only trade quality entries)
- Depth filters (only liquid markets)
- Momentum thresholds (signal quality)

Usage:
    python scripts/optimize_momentum.py
    python scripts/optimize_momentum.py --coin ETH  # Single coin
    python scripts/optimize_momentum.py --top 20    # Show top 20 configs
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_momentum_ticks import (
    load_tick_data,
    backtest_coin,
    analyze_results,
    Trade,
)


@dataclass
class OptimizationResult:
    """Result of a single parameter configuration test."""
    params: dict
    trades: int
    win_rate: float
    avg_pnl_pct: float
    total_pnl_pct: float
    avg_spread: float
    avg_depth: float
    by_coin: Dict[str, dict]


def run_single_config(
    all_ticks: Dict[str, List[dict]],
    params: dict,
    coins: Optional[List[str]] = None,
) -> OptimizationResult:
    """Run backtest with a single parameter configuration."""
    all_trades = []
    by_coin = {}

    for coin, ticks in all_ticks.items():
        if coins and coin not in coins:
            continue

        trades = backtest_coin(ticks, coin, **params)
        all_trades.extend(trades)

        if trades:
            results = analyze_results(trades)
            by_coin[coin] = {
                "trades": results["trades"],
                "win_rate": results["win_rate"],
                "avg_pnl": results["avg_per_trade"],
            }

    if not all_trades:
        return OptimizationResult(
            params=params,
            trades=0,
            win_rate=0,
            avg_pnl_pct=0,
            total_pnl_pct=0,
            avg_spread=0,
            avg_depth=0,
            by_coin={},
        )

    results = analyze_results(all_trades)

    return OptimizationResult(
        params=params,
        trades=results["trades"],
        win_rate=results["win_rate"],
        avg_pnl_pct=results["avg_per_trade"] * 100,
        total_pnl_pct=results["total_pnl"] * 100,
        avg_spread=results.get("avg_spread", 0) * 100,
        avg_depth=results.get("avg_depth", 0),
        by_coin=by_coin,
    )


def generate_param_grid() -> List[dict]:
    """Generate parameter grid for optimization."""

    # Base parameters (fixed)
    base = {
        "momentum_window_ticks": 100,
        "time_stop_ticks": 167,
        "cooldown_ticks": 50,
        "min_time_remaining": 120,
        "use_orderbook": True,
        "depth_ratio_threshold": 1.5,
    }

    # Grid parameters
    grid = {
        "take_profit_pct": [0.05, 0.06, 0.08, 0.10, 0.12],
        "stop_loss_pct": [0.02, 0.03, 0.04, 0.05],
        "max_spread": [0.02, 0.03, 0.04, 0.05, 1.0],  # 1.0 = no filter
        "min_depth": [0, 100, 500, 1000, 2000],
        "momentum_threshold": [0.08, 0.10, 0.12, 0.15, 0.20],
    }

    # Generate all combinations
    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    param_list = []
    for combo in product(*values):
        params = base.copy()
        for i, key in enumerate(keys):
            params[key] = combo[i]
        param_list.append(params)

    return param_list


def main():
    parser = argparse.ArgumentParser(description="Optimize momentum strategy parameters")
    parser.add_argument("--coin", type=str, help="Test single coin (BTC, ETH, SOL, XRP)")
    parser.add_argument("--top", type=int, default=10, help="Number of top configs to show")
    parser.add_argument("--min-trades", type=int, default=10, help="Minimum trades required")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    print("=" * 80)
    print("MOMENTUM STRATEGY PARAMETER OPTIMIZATION")
    print("=" * 80)
    print()

    # Load tick data
    print("Loading tick data...")
    all_ticks = load_tick_data()

    if not all_ticks:
        print("ERROR: No tick data found in data/ticks/")
        print("Run scripts/tick_recorder.py first to collect data.")
        sys.exit(1)

    total_ticks = sum(len(ticks) for ticks in all_ticks.values())
    print(f"Loaded {total_ticks:,} ticks across {len(all_ticks)} coins")
    for coin, ticks in all_ticks.items():
        print(f"  {coin}: {len(ticks):,} ticks")
    print()

    # Filter coins if specified
    coins = [args.coin.upper()] if args.coin else None
    if coins:
        print(f"Testing single coin: {coins[0]}")

    # Generate parameter grid
    param_list = generate_param_grid()
    print(f"Testing {len(param_list):,} parameter combinations...")
    print()

    # Run optimization
    results: List[OptimizationResult] = []
    completed = 0

    for params in param_list:
        result = run_single_config(all_ticks, params, coins)
        results.append(result)
        completed += 1

        if completed % 100 == 0:
            print(f"  Progress: {completed}/{len(param_list)} ({completed*100/len(param_list):.0f}%)")

    print(f"  Completed: {len(results)} configurations tested")
    print()

    # Filter and sort results
    valid_results = [r for r in results if r.trades >= args.min_trades]
    print(f"Configurations with {args.min_trades}+ trades: {len(valid_results)}")

    # Sort by average P&L per trade
    valid_results.sort(key=lambda x: x.avg_pnl_pct, reverse=True)

    # Show top results
    print()
    print("=" * 80)
    print(f"TOP {args.top} PROFITABLE CONFIGURATIONS")
    print("=" * 80)

    profitable_count = sum(1 for r in valid_results if r.avg_pnl_pct > 0)
    print(f"\nProfitable configs: {profitable_count} / {len(valid_results)} ({profitable_count*100/len(valid_results):.1f}%)")
    print()

    for i, result in enumerate(valid_results[:args.top]):
        status = "✓ PROFITABLE" if result.avg_pnl_pct > 0 else "✗ LOSING"
        print(f"\n{'─'*80}")
        print(f"Rank #{i+1}: {status}")
        print(f"{'─'*80}")

        print(f"  Trades: {result.trades:4} | Win Rate: {result.win_rate:5.1f}% | "
              f"Avg P&L: {result.avg_pnl_pct:+6.2f}%/trade")
        print(f"  Total P&L: {result.total_pnl_pct:+.1f}% | "
              f"Avg Spread: {result.avg_spread:.2f}% | Avg Depth: {result.avg_depth:.0f}")

        print(f"\n  Parameters:")
        print(f"    TP: {result.params['take_profit_pct']*100:.0f}% | "
              f"SL: {result.params['stop_loss_pct']*100:.0f}% | "
              f"Momentum: {result.params['momentum_threshold']*100:.0f}%")

        max_spread = result.params['max_spread']
        spread_str = f"{max_spread*100:.1f}%" if max_spread < 1.0 else "No filter"
        min_depth = result.params['min_depth']
        depth_str = f"{min_depth:.0f}" if min_depth > 0 else "No filter"
        print(f"    Max Spread: {spread_str} | Min Depth: {depth_str}")

        if result.by_coin:
            print(f"\n  By Coin:")
            for coin, stats in sorted(result.by_coin.items()):
                avg_pnl = stats['avg_pnl'] * 100
                status = "+" if avg_pnl > 0 else ""
                print(f"    {coin}: {stats['trades']:3} trades, "
                      f"{stats['win_rate']:5.1f}% win, {status}{avg_pnl:.2f}%/trade")

    # Summary statistics
    print()
    print("=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    if valid_results:
        best = valid_results[0]
        print(f"\nBest configuration achieves {best.avg_pnl_pct:+.2f}% per trade")

        if best.avg_pnl_pct > 0:
            # Calculate expected daily profit
            # Assume 3 days of data, estimate trades per day
            trades_per_day = best.trades / 3
            daily_pnl_pct = best.avg_pnl_pct * trades_per_day

            print(f"\nProjected performance (based on {best.trades} trades over ~3 days):")
            print(f"  Trades per day: ~{trades_per_day:.0f}")
            print(f"  Daily P&L: ~{daily_pnl_pct:+.1f}%")
            print(f"  With $100 position size: ~${daily_pnl_pct:.2f}/day")

            print(f"\nRecommended config for strategies/momentum.py:")
            print(f"```python")
            print(f"take_profit: float = {best.params['take_profit_pct']:.2f}")
            print(f"stop_loss: float = {best.params['stop_loss_pct']:.2f}")
            print(f"momentum_threshold: float = {best.params['momentum_threshold']:.2f}")
            max_spread = best.params['max_spread']
            if max_spread < 1.0:
                print(f"max_entry_spread: float = {max_spread:.3f}")
            min_depth = best.params['min_depth']
            if min_depth > 0:
                print(f"min_orderbook_depth: float = {min_depth:.0f}")
            print(f"```")

    # Save results if requested
    if args.output:
        output_data = {
            "total_configs": len(param_list),
            "valid_configs": len(valid_results),
            "profitable_configs": profitable_count,
            "top_results": [
                {
                    "rank": i + 1,
                    "params": r.params,
                    "trades": r.trades,
                    "win_rate": r.win_rate,
                    "avg_pnl_pct": r.avg_pnl_pct,
                    "total_pnl_pct": r.total_pnl_pct,
                    "by_coin": r.by_coin,
                }
                for i, r in enumerate(valid_results[:50])  # Save top 50
            ],
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print()


if __name__ == "__main__":
    main()
