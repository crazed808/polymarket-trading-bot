"""
Analyze tick data to find optimal momentum thresholds for sniper bot.

This script backtests different momentum filter settings to find the sweet spot
that maximizes win rate while avoiding reversals.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, deque
from typing import List, Tuple, Dict
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_tick_data(data_dir: str, coin: str = None, days: int = 7):
    """
    Load tick data from JSONL files.

    Args:
        data_dir: Path to data/ticks directory
        coin: Filter by coin (BTC, ETH, SOL, XRP) or None for all
        days: Number of recent days to load
    """
    data_path = Path(data_dir)
    all_ticks = []

    coins = [coin.lower()] if coin else ['btc', 'eth', 'sol', 'xrp']

    for coin_name in coins:
        coin_dir = data_path / coin_name
        if not coin_dir.exists():
            continue

        # Get recent JSONL files
        files = sorted(coin_dir.glob("*.jsonl"), reverse=True)[:days]

        for file in files:
            print(f"Loading {file.name} ({coin_name.upper()})...", end=" ")
            count = 0
            with open(file, 'r') as f:
                for line in f:
                    try:
                        tick = json.loads(line)
                        # Only include ticks in snipe window
                        if 5 <= tick.get('time_remaining', 0) <= 30:
                            tick['coin'] = coin_name.upper()
                            all_ticks.append(tick)
                            count += 1
                    except json.JSONDecodeError:
                        continue
            print(f"{count} ticks")

    return all_ticks


def analyze_snipe_opportunities(ticks: List[Dict]):
    """
    Analyze historical snipe opportunities and outcomes.
    """
    # Group by market
    markets = defaultdict(list)
    for tick in ticks:
        market_slug = tick.get('market_slug', '')
        if market_slug:
            markets[market_slug].append(tick)

    # Sort each market by timestamp
    for market_slug in markets:
        markets[market_slug].sort(key=lambda t: t.get('timestamp', 0))

    # Analyze each market's snipe window
    results = []

    for market_slug, market_ticks in markets.items():
        if len(market_ticks) < 5:
            continue

        # Get final outcome (last tick)
        final_tick = market_ticks[-1]
        final_price = final_tick.get('mid', 0)

        # Determine if market resolved UP (>0.95) or DOWN (<0.05)
        if final_price >= 0.95:
            outcome = 'UP'
            winning_side = 'up'
        elif final_price <= 0.05:
            outcome = 'DOWN'
            winning_side = 'down'
        else:
            continue  # Skip unclear outcomes

        # Find potential snipe entries (95-97% range)
        for i, tick in enumerate(market_ticks):
            price = tick.get('mid', 0)
            time_remaining = tick.get('time_remaining', 0)
            timestamp = tick.get('timestamp', 0)

            # Check if this is a valid snipe opportunity (UP side)
            if winning_side == 'up' and 0.95 <= price <= 0.97:
                # Calculate momentum over last 10 seconds
                lookback_ticks = [t for t in market_ticks[:i+1]
                                 if timestamp - t.get('timestamp', 0) <= 10]

                if len(lookback_ticks) >= 2:
                    old_price = lookback_ticks[0].get('mid', 0)
                    momentum = (price - old_price) / old_price if old_price > 0 else 0

                    # Calculate orderbook ratio
                    bid_depth = tick.get('bid_depth', 0) or 0
                    ask_depth = tick.get('ask_depth', 0) or 0
                    ob_ratio = bid_depth / ask_depth if ask_depth > 0 else 999

                    # Would this trade win?
                    won = outcome == 'UP'
                    profit = (1.0 - price) if won else -price

                    results.append({
                        'coin': tick.get('coin', ''),
                        'market': market_slug,
                        'entry_price': price,
                        'time_remaining': time_remaining,
                        'momentum': momentum,
                        'ob_ratio': ob_ratio,
                        'won': won,
                        'profit': profit,
                        'final_price': final_price
                    })

            # Check DOWN side opportunities
            elif winning_side == 'down' and 0.03 <= price <= 0.05:
                lookback_ticks = [t for t in market_ticks[:i+1]
                                 if timestamp - t.get('timestamp', 0) <= 10]

                if len(lookback_ticks) >= 2:
                    old_price = lookback_ticks[0].get('mid', 0)
                    momentum = (price - old_price) / old_price if old_price > 0 else 0

                    bid_depth = tick.get('bid_depth', 0) or 0
                    ask_depth = tick.get('ask_depth', 0) or 0
                    ob_ratio = ask_depth / bid_depth if bid_depth > 0 else 999  # Inverted for DOWN

                    won = outcome == 'DOWN'
                    profit = (1.0 - price) if won else -price

                    results.append({
                        'coin': tick.get('coin', ''),
                        'market': market_slug,
                        'entry_price': price,
                        'time_remaining': time_remaining,
                        'momentum': momentum,
                        'ob_ratio': ob_ratio,
                        'won': won,
                        'profit': profit,
                        'final_price': final_price
                    })

    return results


def test_momentum_thresholds(results: List[Dict],
                             momentum_thresholds: List[float],
                             ob_ratio_thresholds: List[float]):
    """
    Test different momentum threshold combinations.

    Returns best configuration.
    """
    print("\n" + "="*80)
    print("MOMENTUM THRESHOLD OPTIMIZATION")
    print("="*80 + "\n")

    best_config = None
    best_score = 0

    print(f"Testing {len(momentum_thresholds)} momentum × {len(ob_ratio_thresholds)} orderbook thresholds...")
    print(f"Total opportunities: {len(results)}\n")

    for mom_thresh in momentum_thresholds:
        for ob_thresh in ob_ratio_thresholds:
            # Filter trades that pass this threshold
            filtered = [r for r in results
                       if r['momentum'] >= mom_thresh and r['ob_ratio'] >= ob_thresh]

            if len(filtered) < 10:
                continue

            wins = sum(1 for r in filtered if r['won'])
            losses = len(filtered) - wins
            win_rate = wins / len(filtered) * 100 if filtered else 0

            total_profit = sum(r['profit'] for r in filtered)
            avg_profit = total_profit / len(filtered) if filtered else 0

            # Score: prioritize win rate but consider volume
            score = win_rate * len(filtered) * avg_profit

            if score > best_score and win_rate >= 95:
                best_score = score
                best_config = {
                    'momentum_threshold': mom_thresh,
                    'ob_ratio_threshold': ob_thresh,
                    'trades': len(filtered),
                    'wins': wins,
                    'losses': losses,
                    'win_rate': win_rate,
                    'total_profit': total_profit,
                    'avg_profit': avg_profit
                }

    return best_config, results


def main():
    """Run momentum optimization analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Optimize sniper momentum thresholds')
    parser.add_argument('--data', default='data/ticks', help='Path to tick data directory')
    parser.add_argument('--coin', default=None, help='Filter by coin (BTC, ETH, SOL, XRP)')
    parser.add_argument('--days', type=int, default=7, help='Days of data to analyze')
    args = parser.parse_args()

    print(f"Loading tick data from: {args.data}")
    if args.coin:
        print(f"Filtering for: {args.coin}")
    print(f"Loading last {args.days} days\n")

    # Load data
    ticks = load_tick_data(args.data, args.coin, args.days)

    if not ticks:
        print("\nNo tick data found!")
        print("Run tick recorder first: python scripts/tick_recorder.py start")
        return

    print(f"\nLoaded {len(ticks)} ticks in snipe window (5-30s)")

    # Analyze opportunities
    print("\nAnalyzing snipe opportunities...")
    results = analyze_snipe_opportunities(ticks)

    if not results:
        print("No snipe opportunities found in data!")
        return

    print(f"Found {len(results)} potential snipe entries")

    # Calculate baseline (no filter)
    baseline_wins = sum(1 for r in results if r['won'])
    baseline_win_rate = baseline_wins / len(results) * 100
    baseline_profit = sum(r['profit'] for r in results)

    print(f"\nBaseline (no momentum filter):")
    print(f"  Win rate:      {baseline_win_rate:.1f}% ({baseline_wins}/{len(results)})")
    print(f"  Total profit:  ${baseline_profit:.2f}")
    print(f"  Avg profit:    ${baseline_profit/len(results):.3f}/trade")

    # Test thresholds
    momentum_thresholds = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01]
    ob_ratio_thresholds = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

    best_config, all_results = test_momentum_thresholds(
        results, momentum_thresholds, ob_ratio_thresholds
    )

    if best_config:
        print("\n" + "="*80)
        print("OPTIMAL CONFIGURATION FOUND")
        print("="*80)
        print(f"\nMomentum threshold: {best_config['momentum_threshold']:.2%}")
        print(f"Orderbook ratio:    {best_config['ob_ratio_threshold']:.2f}x")
        print(f"\nPerformance:")
        print(f"  Total trades:   {best_config['trades']}")
        print(f"  Wins:          {best_config['wins']}")
        print(f"  Losses:        {best_config['losses']}")
        print(f"  Win rate:      {best_config['win_rate']:.1f}%")
        print(f"  Total profit:  ${best_config['total_profit']:.2f}")
        print(f"  Avg profit:    ${best_config['avg_profit']:.3f}/trade")

        improvement = best_config['win_rate'] - baseline_win_rate
        print(f"\n  Win rate improvement: +{improvement:.1f}%")
        trades_filtered = len(results) - best_config['trades']
        print(f"  Trades filtered out:  {trades_filtered} ({trades_filtered/len(results)*100:.1f}%)")

        print("\n" + "="*80)
        print("RECOMMENDED CONFIG")
        print("="*80)
        print(f"""
config = SnipeConfig(
    check_momentum=True,
    max_negative_momentum={best_config['momentum_threshold']:.2f},
    min_orderbook_ratio={best_config['ob_ratio_threshold']:.1f},
    # ... other settings
)
        """)
    else:
        print("\nNo configuration improved win rate above 95%")
        print("Current baseline is already strong!")

    # Show breakdown by coin
    print("\n" + "="*80)
    print("BREAKDOWN BY COIN")
    print("="*80 + "\n")

    by_coin = defaultdict(list)
    for r in results:
        by_coin[r['coin']].append(r)

    for coin, coin_results in sorted(by_coin.items()):
        wins = sum(1 for r in coin_results if r['won'])
        total = len(coin_results)
        win_rate = wins / total * 100 if total else 0
        profit = sum(r['profit'] for r in coin_results)

        print(f"{coin}:")
        print(f"  Opportunities: {total}")
        print(f"  Win rate:      {win_rate:.1f}% ({wins}/{total})")
        print(f"  Total profit:  ${profit:.2f}")
        print(f"  Avg profit:    ${profit/total:.3f}/trade")
        print()


if __name__ == '__main__':
    main()
