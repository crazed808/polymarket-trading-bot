#!/usr/bin/env python3
"""
Analyze how often a 99%+ price moves to 98% or lower WITHIN the same market.

Key insight: When we see price go from 99% to <50%, that's a market reset,
not a reversal. We need to look at movements that happen QUICKLY.

Strategy: Buy at 1%, sell at 2% = 100% profit
Question: How often does this work before market ends?
"""

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PricePoint:
    timestamp: datetime
    mid: float


@dataclass
class ExtremeWindow:
    """A window where price was at 99%+."""
    start_time: datetime
    end_time: datetime
    start_price: float
    min_price_during: float
    reached_98: bool
    time_to_98: Optional[float]  # seconds
    ended_at_99_plus: bool  # Did market end while still 99%+?


def load_market_data(filepath: Path) -> List[PricePoint]:
    """Load price data from a JSONL file."""
    points = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                ts_str = data['ts']
                # Handle various timestamp formats
                if '.' in ts_str:
                    ts = datetime.fromisoformat(ts_str.split('+')[0])
                else:
                    ts = datetime.fromisoformat(ts_str.replace('Z', ''))
                points.append(PricePoint(timestamp=ts, mid=data.get('mid', 0)))
            except Exception:
                continue
    return sorted(points, key=lambda p: p.timestamp)


def find_extreme_windows(points: List[PricePoint], threshold: float = 0.99) -> List[ExtremeWindow]:
    """
    Find continuous windows where price is at threshold or above.

    Track what happens: does it dip to 98% before market resets?
    """
    windows = []
    i = 0

    while i < len(points):
        # Look for start of extreme window
        if points[i].mid >= threshold:
            start_idx = i
            start_time = points[i].timestamp
            start_price = points[i].mid
            min_price = points[i].mid
            reached_98 = False
            time_to_98 = None

            # Track through the extreme period
            j = i + 1
            while j < len(points):
                curr = points[j]
                time_diff = (curr.timestamp - start_time).total_seconds()

                # If we see a big gap (>60s), market probably reset
                if j > start_idx and time_diff > 0:
                    prev_time = points[j-1].timestamp
                    gap = (curr.timestamp - prev_time).total_seconds()
                    if gap > 60:  # Market reset detected
                        break

                # If price drops below 80%, definitely market reset
                if curr.mid < 0.80:
                    break

                # Track minimum price
                if curr.mid < min_price:
                    min_price = curr.mid

                # Check if reached 98%
                if curr.mid <= 0.98 and not reached_98:
                    reached_98 = True
                    time_to_98 = time_diff

                # If price goes back above threshold after dipping, window ends
                if curr.mid >= threshold and min_price < threshold:
                    break

                j += 1

            # Determine if market ended while still at 99%+
            ended_at_99_plus = min_price >= threshold

            windows.append(ExtremeWindow(
                start_time=start_time,
                end_time=points[min(j, len(points)-1)].timestamp,
                start_price=start_price,
                min_price_during=min_price,
                reached_98=reached_98,
                time_to_98=time_to_98,
                ended_at_99_plus=ended_at_99_plus,
            ))

            i = j
        else:
            i += 1

    return windows


def analyze_for_quick_flip(points: List[PricePoint], threshold: float = 0.99, max_time: float = 120) -> Dict:
    """
    Specifically analyze: if we enter at 99%+, can we exit at 98% within max_time seconds?

    This is the "quick flip" strategy analysis.
    """
    results = {
        'total_entries': 0,
        'exits_at_98': 0,
        'exits_at_97': 0,
        'stayed_99_plus': 0,  # Market ended at 99%+
        'times_to_98': [],
        'times_to_97': [],
    }

    i = 0
    while i < len(points):
        if points[i].mid >= threshold:
            entry_time = points[i].timestamp
            entry_price = points[i].mid
            results['total_entries'] += 1

            found_98 = False
            found_97 = False

            # Look ahead for exit opportunity
            j = i + 1
            while j < len(points):
                curr = points[j]
                elapsed = (curr.timestamp - entry_time).total_seconds()

                # Only look within our time window
                if elapsed > max_time:
                    break

                # Market reset detection
                if curr.mid < 0.80:
                    break

                # Check exit points
                if curr.mid <= 0.98 and not found_98:
                    found_98 = True
                    results['exits_at_98'] += 1
                    results['times_to_98'].append(elapsed)

                if curr.mid <= 0.97 and not found_97:
                    found_97 = True
                    results['exits_at_97'] += 1
                    results['times_to_97'].append(elapsed)

                if found_98:
                    break

                j += 1

            if not found_98 and not found_97:
                # Check if stayed at 99%+
                end_price = points[min(j-1, len(points)-1)].mid if j > i else entry_price
                if end_price >= threshold:
                    results['stayed_99_plus'] += 1

            # Skip past this extreme period
            while j < len(points) and points[j].mid >= 0.95:
                j += 1
            i = j
        else:
            i += 1

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--max-time", type=float, default=120, help="Max seconds to wait for exit")
    parser.add_argument("--coin", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data" / "recordings"

    coins = ["btc", "eth", "sol", "xrp"]
    if args.coin:
        coins = [args.coin.lower()]

    print("=" * 70)
    print("QUICK FLIP STRATEGY ANALYSIS")
    print("=" * 70)
    print(f"Strategy: Buy at 1% when opposite hits 99%, sell at 2%")
    print(f"Time window: {args.max_time}s max wait for exit")
    print()

    combined = {
        'total_entries': 0,
        'exits_at_98': 0,
        'exits_at_97': 0,
        'stayed_99_plus': 0,
        'times_to_98': [],
    }

    for coin in coins:
        coin_dir = data_dir / coin
        if not coin_dir.exists():
            continue

        coin_results = {
            'total_entries': 0,
            'exits_at_98': 0,
            'exits_at_97': 0,
            'stayed_99_plus': 0,
            'times_to_98': [],
        }

        for filepath in sorted(coin_dir.glob("*.jsonl")):
            points = load_market_data(filepath)
            if not points:
                continue

            results = analyze_for_quick_flip(points, args.threshold, args.max_time)

            coin_results['total_entries'] += results['total_entries']
            coin_results['exits_at_98'] += results['exits_at_98']
            coin_results['exits_at_97'] += results['exits_at_97']
            coin_results['stayed_99_plus'] += results['stayed_99_plus']
            coin_results['times_to_98'].extend(results['times_to_98'])

        # Print coin results
        total = coin_results['total_entries']
        if total == 0:
            continue

        exits_98 = coin_results['exits_at_98']
        stayed = coin_results['stayed_99_plus']

        print(f"{coin.upper()}:")
        print(f"  Total 99%+ entries: {total}")
        print(f"  Exited at 98% within {args.max_time}s: {exits_98} ({100*exits_98/total:.1f}%)")
        print(f"  Stayed at 99%+ (no exit): {stayed} ({100*stayed/total:.1f}%)")

        if coin_results['times_to_98']:
            avg_time = sum(coin_results['times_to_98']) / len(coin_results['times_to_98'])
            print(f"  Avg time to 98%: {avg_time:.1f}s")
        print()

        # Add to combined
        for key in combined:
            if isinstance(combined[key], list):
                combined[key].extend(coin_results[key])
            else:
                combined[key] += coin_results[key]

    # Combined summary
    print("=" * 70)
    print("COMBINED RESULTS")
    print("=" * 70)

    total = combined['total_entries']
    if total > 0:
        exits_98 = combined['exits_at_98']
        stayed = combined['stayed_99_plus']
        other = total - exits_98 - stayed

        success_rate = exits_98 / total

        print(f"Total 99%+ entries: {total}")
        print(f"Successfully exited at 98%: {exits_98} ({100*success_rate:.1f}%)")
        print(f"Stayed at 99%+ (loss): {stayed} ({100*stayed/total:.1f}%)")
        print(f"Other (market movement): {other} ({100*other/total:.1f}%)")

        if combined['times_to_98']:
            times = combined['times_to_98']
            print(f"\nTime to reach 98%:")
            print(f"  Average: {sum(times)/len(times):.1f}s")
            print(f"  Median: {sorted(times)[len(times)//2]:.1f}s")
            print(f"  Min: {min(times):.1f}s")
            print(f"  Max: {max(times):.1f}s")

        # Calculate EV
        # Win: doubled money (100% profit)
        # Loss: lose everything (-100%)
        # "Other" outcomes - assume we break even or small loss

        print(f"\n{'='*70}")
        print("EXPECTED VALUE CALCULATION")
        print(f"{'='*70}")
        print(f"If we buy at 1% and sell at 2%:")
        print(f"  Win probability: {100*success_rate:.1f}%")
        print(f"  Win payout: +100% (doubled)")
        print(f"  Loss probability: {100*(1-success_rate):.1f}%")
        print(f"  Loss amount: -100% (total loss)")

        ev = success_rate * 1.0 - (1 - success_rate) * 1.0
        print(f"\n  Expected Value: {ev*100:+.1f}% per trade")

        if ev > 0:
            print(f"\n  VERDICT: PROFITABLE strategy with +{ev*100:.1f}% expected ROI")
        else:
            print(f"\n  VERDICT: NOT profitable (negative EV)")

        # More realistic calculation accounting for spreads
        print(f"\n{'='*70}")
        print("REALISTIC SCENARIO (accounting for spreads)")
        print(f"{'='*70}")
        print("Reality: You don't buy at exactly 1% or sell at exactly 2%")
        print("  - Buy price: ~1.5-2% (paying the ask)")
        print("  - Sell price: ~1.8-2% (hitting the bid)")
        print("  - Actual profit on win: ~30-50% (not 100%)")
        print("  - Loss on failure: still ~100%")

        realistic_win = 0.40  # 40% profit instead of 100%
        realistic_ev = success_rate * realistic_win - (1 - success_rate) * 1.0
        print(f"\n  Realistic EV (40% win, 100% loss): {realistic_ev*100:+.1f}%")

        if realistic_ev > 0:
            print(f"  Still profitable!")
        else:
            print(f"  NOT profitable with realistic spreads")


if __name__ == "__main__":
    main()
