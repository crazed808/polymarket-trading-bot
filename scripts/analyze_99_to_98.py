#!/usr/bin/env python3
"""
Analyze how often a 99%+ price moves to 98% or lower.

This helps evaluate a "quick flip" strategy:
- Buy at 1% (when opposite is 99%)
- Sell at 2% (when opposite drops to 98%)
- 100% profit per trade

The question: How often does 99% drop to at least 98%?
"""

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PricePoint:
    """A single price observation."""
    timestamp: datetime
    mid: float
    best_bid: float
    best_ask: float


@dataclass
class ExtremeEvent:
    """An extreme price event and its aftermath."""
    start_time: datetime
    start_price: float  # The 99%+ price
    min_price_after: float  # Lowest price it dropped to
    dropped_to_98: bool  # Did it reach 98% or lower?
    dropped_to_97: bool
    dropped_to_95: bool
    time_to_98: Optional[float]  # Seconds to reach 98% (if it did)
    recovery_time: Optional[float]  # Seconds until it went back above 99%


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
                ts = datetime.fromisoformat(data['ts'].replace('Z', '+00:00').replace('+00:00', ''))
                points.append(PricePoint(
                    timestamp=ts,
                    mid=data.get('mid', 0),
                    best_bid=data.get('best_bid', 0),
                    best_ask=data.get('best_ask', 0),
                ))
            except Exception:
                continue
    return points


def analyze_extreme_events(points: List[PricePoint], threshold: float = 0.99) -> List[ExtremeEvent]:
    """
    Find all 99%+ events and track what happened after.

    Returns list of events with analysis of price movement after.
    """
    events = []
    i = 0

    while i < len(points):
        point = points[i]

        # Check if this is an extreme point (99%+)
        if point.mid >= threshold:
            start_time = point.timestamp
            start_price = point.mid

            # Track what happens after
            min_price_after = point.mid
            time_to_98 = None
            recovery_time = None

            j = i + 1
            while j < len(points):
                next_point = points[j]

                # Update minimum price seen
                if next_point.mid < min_price_after:
                    min_price_after = next_point.mid

                # Check if dropped to 98%
                if next_point.mid <= 0.98 and time_to_98 is None:
                    time_to_98 = (next_point.timestamp - start_time).total_seconds()

                # Check if recovered back to 99%+
                if next_point.mid >= threshold and min_price_after < threshold:
                    recovery_time = (next_point.timestamp - start_time).total_seconds()
                    break

                # Stop tracking after the extreme period ends and we've moved on
                if next_point.mid < 0.95:
                    break

                j += 1

            events.append(ExtremeEvent(
                start_time=start_time,
                start_price=start_price,
                min_price_after=min_price_after,
                dropped_to_98=min_price_after <= 0.98,
                dropped_to_97=min_price_after <= 0.97,
                dropped_to_95=min_price_after <= 0.95,
                time_to_98=time_to_98,
                recovery_time=recovery_time,
            ))

            # Skip past this extreme period
            i = j
        else:
            i += 1

    return events


def analyze_directory(dirpath: Path, threshold: float = 0.99) -> Dict[str, List[ExtremeEvent]]:
    """Analyze all markets in a directory."""
    all_events = {}

    for filepath in sorted(dirpath.glob("*.jsonl")):
        points = load_market_data(filepath)
        if not points:
            continue

        events = analyze_extreme_events(points, threshold)
        if events:
            all_events[filepath.name] = events

    return all_events


def print_report(events_by_market: Dict[str, List[ExtremeEvent]], coin: str):
    """Print analysis report."""
    all_events = []
    for events in events_by_market.values():
        all_events.extend(events)

    if not all_events:
        print(f"\n{coin}: No 99%+ events found")
        return

    total = len(all_events)
    dropped_98 = sum(1 for e in all_events if e.dropped_to_98)
    dropped_97 = sum(1 for e in all_events if e.dropped_to_97)
    dropped_95 = sum(1 for e in all_events if e.dropped_to_95)

    print(f"\n{'='*60}")
    print(f"{coin} - 99%+ PRICE MOVEMENT ANALYSIS")
    print(f"{'='*60}")
    print(f"Total 99%+ events: {total}")
    print()
    print(f"Dropped to 98% or lower: {dropped_98}/{total} ({100*dropped_98/total:.1f}%)")
    print(f"Dropped to 97% or lower: {dropped_97}/{total} ({100*dropped_97/total:.1f}%)")
    print(f"Dropped to 95% or lower: {dropped_95}/{total} ({100*dropped_95/total:.1f}%)")

    # Time to reach 98%
    times_to_98 = [e.time_to_98 for e in all_events if e.time_to_98 is not None]
    if times_to_98:
        avg_time = sum(times_to_98) / len(times_to_98)
        min_time = min(times_to_98)
        max_time = max(times_to_98)
        print()
        print(f"When it dropped to 98%:")
        print(f"  Average time: {avg_time:.1f}s")
        print(f"  Fastest: {min_time:.1f}s")
        print(f"  Slowest: {max_time:.1f}s")

    # Distribution of minimum prices
    print()
    print("Minimum price distribution after 99%+ event:")
    buckets = defaultdict(int)
    for e in all_events:
        if e.min_price_after >= 0.99:
            buckets["99%+"] += 1
        elif e.min_price_after >= 0.98:
            buckets["98-99%"] += 1
        elif e.min_price_after >= 0.97:
            buckets["97-98%"] += 1
        elif e.min_price_after >= 0.95:
            buckets["95-97%"] += 1
        elif e.min_price_after >= 0.90:
            buckets["90-95%"] += 1
        else:
            buckets["<90%"] += 1

    for bucket in ["99%+", "98-99%", "97-98%", "95-97%", "90-95%", "<90%"]:
        count = buckets[bucket]
        pct = 100 * count / total
        bar = "#" * int(pct / 2)
        print(f"  {bucket:>8}: {count:>4} ({pct:>5.1f}%) {bar}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze 99% to 98% probability movements")
    parser.add_argument("--threshold", type=float, default=0.99, help="Extreme threshold (default: 0.99)")
    parser.add_argument("--coin", type=str, default=None, help="Specific coin to analyze")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data" / "recordings"

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    coins = ["btc", "eth", "sol", "xrp"]
    if args.coin:
        coins = [args.coin.lower()]

    print("="*60)
    print("99% -> 98% MOVEMENT ANALYSIS")
    print("="*60)
    print(f"Question: How often does a 99%+ price drop to 98% or lower?")
    print(f"Strategy implication: Buy at 1%, sell at 2% = 100% profit")
    print()

    all_coin_events = {}

    for coin in coins:
        coin_dir = data_dir / coin
        if not coin_dir.exists():
            continue

        events = analyze_directory(coin_dir, args.threshold)
        all_coin_events[coin.upper()] = events
        print_report(events, coin.upper())

    # Combined summary
    print()
    print("="*60)
    print("COMBINED SUMMARY")
    print("="*60)

    total_events = 0
    total_dropped_98 = 0
    total_dropped_97 = 0

    for coin, events_by_market in all_coin_events.items():
        for events in events_by_market.values():
            for e in events:
                total_events += 1
                if e.dropped_to_98:
                    total_dropped_98 += 1
                if e.dropped_to_97:
                    total_dropped_97 += 1

    if total_events > 0:
        print(f"Total 99%+ events across all coins: {total_events}")
        print(f"Dropped to 98%: {total_dropped_98}/{total_events} ({100*total_dropped_98/total_events:.1f}%)")
        print(f"Dropped to 97%: {total_dropped_97}/{total_events} ({100*total_dropped_97/total_events:.1f}%)")
        print()
        print("STRATEGY VIABILITY:")
        print(f"  If you buy at 1% and aim to sell at 2%:")
        print(f"  - Success rate: {100*total_dropped_98/total_events:.1f}%")
        print(f"  - Profit per success: 100%")
        print(f"  - Loss on failure: ~100% (market ends at 99%+)")

        # Calculate expected value
        success_rate = total_dropped_98 / total_events
        ev = success_rate * 1.0 - (1 - success_rate) * 1.0  # Win 100%, lose 100%
        print(f"  - Expected value per $1 bet: ${ev:+.2f}")

        if ev > 0:
            print(f"\n  VERDICT: Strategy is PROFITABLE with {ev*100:+.1f}% expected ROI")
        else:
            print(f"\n  VERDICT: Strategy is NOT profitable (negative EV)")

    print()


if __name__ == "__main__":
    main()
