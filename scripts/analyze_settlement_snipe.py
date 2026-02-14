#!/usr/bin/env python3
"""
Analyze "Settlement Snipe" Strategy

Theory: Buy at 98-99% probability seconds before resolution.
- Win: +1-2% (price goes to $1.00)
- Lose: -98-99% (price goes to $0.00)

Question: What's the actual win rate at extreme prices near settlement?
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data" / "ticks"

def analyze_settlements(coin: str, days: int = 7):
    """Analyze what happens to extreme prices near settlement."""
    coin_dir = DATA_DIR / coin.lower()
    if not coin_dir.exists():
        return None

    files = sorted(coin_dir.glob("*.jsonl"))[-days:]

    # Track markets and their final states
    markets = defaultdict(list)

    for file_path in files:
        print(f"  Processing {file_path.name}...", end=" ", flush=True)
        count = 0
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    tick = json.loads(line)
                    if tick.get('side') != 'up':
                        continue

                    market_slug = tick.get('market_slug', 'unknown')
                    time_remaining = tick.get('time_remaining', 999)
                    mid = tick.get('mid', 0)

                    if mid and time_remaining is not None:
                        markets[market_slug].append({
                            'time_remaining': time_remaining,
                            'mid': float(mid),
                            'ts': tick.get('ts', '')
                        })
                        count += 1
                except:
                    continue
        print(f"{count} ticks")

    return markets

def calculate_outcomes(markets: dict):
    """
    For each market, find:
    1. Price at various time points before settlement
    2. Final outcome (did it settle at 1.00 or 0.00?)
    """
    results = {
        'extreme_high': [],  # Price >= 0.95 near end
        'extreme_low': [],   # Price <= 0.05 near end
        'moderate': []       # Everything else
    }

    # Time windows to analyze (seconds before settlement)
    time_windows = [5, 10, 15, 30, 60]

    window_results = {w: {'high_wins': 0, 'high_losses': 0, 'low_wins': 0, 'low_losses': 0}
                      for w in time_windows}

    for market_slug, ticks in markets.items():
        if len(ticks) < 10:
            continue

        # Sort by time remaining (descending - higher = earlier)
        ticks.sort(key=lambda x: x['time_remaining'], reverse=True)

        # Find the final price (lowest time_remaining)
        final_ticks = [t for t in ticks if t['time_remaining'] <= 5]
        if not final_ticks:
            continue

        final_price = final_ticks[-1]['mid']

        # Determine outcome: settled high (>=0.95) or low (<=0.05)?
        if final_price >= 0.95:
            outcome = 'high'
        elif final_price <= 0.05:
            outcome = 'low'
        else:
            outcome = 'uncertain'
            continue  # Skip markets that didn't clearly resolve

        # For each time window, find the price and check if prediction was correct
        for window in time_windows:
            window_ticks = [t for t in ticks if window - 3 <= t['time_remaining'] <= window + 3]
            if not window_ticks:
                continue

            # Get price at this window
            price_at_window = window_ticks[0]['mid']

            # If price was high (>=0.95), did it stay high?
            if price_at_window >= 0.95:
                if outcome == 'high':
                    window_results[window]['high_wins'] += 1
                else:
                    window_results[window]['high_losses'] += 1

            # If price was extreme high (>=0.98)
            if price_at_window >= 0.98:
                results['extreme_high'].append({
                    'market': market_slug,
                    'price': price_at_window,
                    'window': window,
                    'outcome': outcome,
                    'won': outcome == 'high'
                })

            # If price was low (<=0.05), did it stay low?
            if price_at_window <= 0.05:
                if outcome == 'low':
                    window_results[window]['low_wins'] += 1
                else:
                    window_results[window]['low_losses'] += 1

            if price_at_window <= 0.02:
                results['extreme_low'].append({
                    'market': market_slug,
                    'price': price_at_window,
                    'window': window,
                    'outcome': outcome,
                    'won': outcome == 'low'
                })

    return results, window_results

def simulate_strategy(results: dict, buy_price: float = 0.98):
    """Simulate P&L for settlement snipe strategy."""
    trades = [r for r in results['extreme_high'] if r['price'] >= buy_price]

    if not trades:
        return None

    wins = [t for t in trades if t['won']]
    losses = [t for t in trades if not t['won']]

    win_rate = len(wins) / len(trades) * 100

    # P&L calculation
    # Win: pay buy_price, receive 1.00 → profit = 1.00 - buy_price
    # Lose: pay buy_price, receive 0.00 → loss = buy_price
    profit_per_win = 1.00 - buy_price
    loss_per_loss = buy_price

    total_profit = len(wins) * profit_per_win - len(losses) * loss_per_loss
    avg_per_trade = total_profit / len(trades)

    # Account for ~1% buy spread
    spread_cost = 0.01
    net_per_trade = avg_per_trade - spread_cost

    return {
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'gross_per_trade': avg_per_trade,
        'net_per_trade': net_per_trade,
        'total_profit': total_profit,
        'break_even_rate': buy_price / 1.00 * 100
    }

def main():
    print("="*70)
    print("SETTLEMENT SNIPE STRATEGY ANALYSIS")
    print("="*70)
    print()

    all_results = {'extreme_high': [], 'extreme_low': []}
    all_window_results = {}

    for coin in ['BTC', 'ETH', 'SOL', 'XRP']:
        print(f"\nAnalyzing {coin}...")
        markets = analyze_settlements(coin, days=5)

        if not markets:
            print(f"  No data for {coin}")
            continue

        print(f"  Found {len(markets)} markets")

        results, window_results = calculate_outcomes(markets)
        all_results['extreme_high'].extend(results['extreme_high'])
        all_results['extreme_low'].extend(results['extreme_low'])

        for w, r in window_results.items():
            if w not in all_window_results:
                all_window_results[w] = {'high_wins': 0, 'high_losses': 0, 'low_wins': 0, 'low_losses': 0}
            for k, v in r.items():
                all_window_results[w][k] += v

    print("\n" + "="*70)
    print("WIN RATES BY TIME WINDOW (price >= 0.95)")
    print("="*70)

    for window in sorted(all_window_results.keys()):
        r = all_window_results[window]
        total_high = r['high_wins'] + r['high_losses']
        if total_high > 0:
            win_rate = r['high_wins'] / total_high * 100
            print(f"  {window:2d}s before settlement: {win_rate:.1f}% win rate ({r['high_wins']}/{total_high} trades)")

    print("\n" + "="*70)
    print("STRATEGY SIMULATION - BUY AT EXTREME PRICES")
    print("="*70)

    for buy_threshold in [0.95, 0.96, 0.97, 0.98, 0.99]:
        sim = simulate_strategy(all_results, buy_threshold)
        if sim:
            status = "PROFITABLE" if sim['net_per_trade'] > 0 else "LOSING"
            print(f"\n  Buy at >= {buy_threshold:.0%}:")
            print(f"    Trades: {sim['trades']}")
            print(f"    Win rate: {sim['win_rate']:.1f}% (need {sim['break_even_rate']:.1f}% to break even)")
            print(f"    Gross P&L: {sim['gross_per_trade']*100:+.2f}% per trade")
            print(f"    Net P&L (after 1% spread): {sim['net_per_trade']*100:+.2f}% per trade")
            print(f"    Status: {status}")

    # Detailed analysis of losses
    print("\n" + "="*70)
    print("LOSS ANALYSIS - WHEN 98%+ BETS FAIL")
    print("="*70)

    high_losses = [r for r in all_results['extreme_high'] if not r['won'] and r['price'] >= 0.98]
    if high_losses:
        print(f"\n  Found {len(high_losses)} cases where price >= 98% but outcome was LOW:")
        for loss in high_losses[:10]:  # Show first 10
            print(f"    - {loss['market']}: price={loss['price']:.2%} at {loss['window']}s, resolved LOW")
    else:
        print("\n  No losses found at 98%+ prices (but sample may be small)")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    sim_98 = simulate_strategy(all_results, 0.98)
    if sim_98:
        if sim_98['net_per_trade'] > 0:
            print(f"\n  Strategy appears PROFITABLE at 98%+ entries")
            print(f"  Expected: {sim_98['net_per_trade']*100:+.2f}% per trade after spread")
            print(f"  Risk: Each loss costs ~98 cents per $1 bet")
        else:
            print(f"\n  Strategy is NOT profitable at current win rates")
            print(f"  Win rate {sim_98['win_rate']:.1f}% is below break-even {sim_98['break_even_rate']:.1f}%")
    else:
        print("\n  Insufficient data to evaluate strategy")

if __name__ == "__main__":
    main()
