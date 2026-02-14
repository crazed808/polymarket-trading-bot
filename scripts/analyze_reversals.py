#!/usr/bin/env python3
"""
Reversal Analysis Script

Analyzes recorded reversal data to answer questions like:
- How often does a 99% probability with 2+ min left get reversed?
- What's the reversal rate at different time thresholds?
- Which coins have higher reversal rates?

Usage:
    python scripts/analyze_reversals.py data/reversals/
    python scripts/analyze_reversals.py data/historical/
    python scripts/analyze_reversals.py data/reversals/ --min-time 120
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ExtremeEvent:
    """An extreme probability event."""
    timestamp: str
    side: str
    probability: float
    time_remaining_seconds: int


@dataclass
class MarketRecord:
    """A recorded market with extreme events."""
    slug: str
    coin: str
    end_time: str
    winner: Optional[str]
    extreme_events: List[ExtremeEvent]
    reversed: bool


class ReversalAnalyzer:
    """Analyzes reversal data from recorded files."""

    # Time buckets for analysis
    TIME_BUCKETS = [
        ("5min+", 300, float('inf')),
        ("3-5min", 180, 300),
        ("2-3min", 120, 180),
        ("1-2min", 60, 120),
        ("30s-1min", 30, 60),
        ("<30s", 0, 30),
    ]

    # Probability buckets
    PROB_BUCKETS = [
        ("99.5%+", 0.995, 1.0),
        ("99-99.5%", 0.99, 0.995),
        ("98-99%", 0.98, 0.99),
        ("95-98%", 0.95, 0.98),
    ]

    def __init__(self):
        self.records: List[MarketRecord] = []

    def load_file(self, filepath: Path) -> int:
        """Load records from a JSONL file."""
        count = 0
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    record = self._parse_record(data)
                    if record:
                        self.records.append(record)
                        count += 1
                except Exception as e:
                    print(f"Warning: Failed to parse line: {e}")
        return count

    def _parse_record(self, data: Dict[str, Any]) -> Optional[MarketRecord]:
        """Parse a JSON record into MarketRecord."""
        events = []
        for e in data.get("extreme_events", []):
            events.append(ExtremeEvent(
                timestamp=e.get("timestamp", ""),
                side=e.get("side", ""),
                probability=float(e.get("probability", 0)),
                time_remaining_seconds=int(e.get("time_remaining_seconds", 0))
            ))

        return MarketRecord(
            slug=data.get("slug") or data.get("market_slug", ""),
            coin=data.get("coin", ""),
            end_time=data.get("end_time", ""),
            winner=data.get("winner"),
            extreme_events=events,
            reversed=data.get("reversed", False)
        )

    def load_directory(self, dirpath: Path) -> int:
        """Load all JSONL files from a directory."""
        total = 0
        for filepath in dirpath.glob("*.jsonl"):
            count = self.load_file(filepath)
            print(f"Loaded {count} records from {filepath.name}")
            total += count
        return total

    def _get_time_bucket(self, seconds: int) -> str:
        """Get the time bucket name for a given number of seconds."""
        for name, min_s, max_s in self.TIME_BUCKETS:
            if min_s <= seconds < max_s:
                return name
        return "<30s"

    def _get_prob_bucket(self, prob: float) -> str:
        """Get the probability bucket name."""
        for name, min_p, max_p in self.PROB_BUCKETS:
            if min_p <= prob < max_p:
                return name
        return "95-98%"

    def analyze_by_time(
        self,
        min_prob: float = 0.99,
        coin_filter: Optional[str] = None
    ) -> Dict[str, Dict[str, int]]:
        """
        Analyze reversal rates by time remaining.

        Args:
            min_prob: Minimum probability to consider as "extreme"
            coin_filter: Optional coin to filter by

        Returns:
            Dictionary mapping time bucket to stats
        """
        stats = {name: {"extremes": 0, "reversals": 0} for name, _, _ in self.TIME_BUCKETS}

        for record in self.records:
            if coin_filter and record.coin != coin_filter:
                continue

            if not record.winner:
                continue

            for event in record.extreme_events:
                if event.probability < min_prob:
                    continue

                bucket = self._get_time_bucket(event.time_remaining_seconds)
                stats[bucket]["extremes"] += 1

                # Check if this specific event was a reversal
                if event.side != record.winner:
                    stats[bucket]["reversals"] += 1

        return stats

    def analyze_by_probability(
        self,
        min_time: int = 0,
        coin_filter: Optional[str] = None
    ) -> Dict[str, Dict[str, int]]:
        """
        Analyze reversal rates by probability level.

        Args:
            min_time: Minimum time remaining to consider
            coin_filter: Optional coin to filter by

        Returns:
            Dictionary mapping probability bucket to stats
        """
        stats = {name: {"extremes": 0, "reversals": 0} for name, _, _ in self.PROB_BUCKETS}

        for record in self.records:
            if coin_filter and record.coin != coin_filter:
                continue

            if not record.winner:
                continue

            for event in record.extreme_events:
                if event.time_remaining_seconds < min_time:
                    continue

                bucket = self._get_prob_bucket(event.probability)
                stats[bucket]["extremes"] += 1

                if event.side != record.winner:
                    stats[bucket]["reversals"] += 1

        return stats

    def analyze_by_coin(self, min_prob: float = 0.99, min_time: int = 0) -> Dict[str, Dict[str, int]]:
        """
        Analyze reversal rates by coin.

        Args:
            min_prob: Minimum probability threshold
            min_time: Minimum time remaining

        Returns:
            Dictionary mapping coin to stats
        """
        stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"markets": 0, "extremes": 0, "reversals": 0})

        for record in self.records:
            if not record.winner:
                continue

            coin = record.coin or "UNKNOWN"
            stats[coin]["markets"] += 1

            for event in record.extreme_events:
                if event.probability < min_prob:
                    continue
                if event.time_remaining_seconds < min_time:
                    continue

                stats[coin]["extremes"] += 1

                if event.side != record.winner:
                    stats[coin]["reversals"] += 1

        return dict(stats)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get overall summary statistics."""
        total_markets = len(self.records)
        markets_with_winner = sum(1 for r in self.records if r.winner)
        markets_with_extremes = sum(1 for r in self.records if r.extreme_events)
        markets_reversed = sum(1 for r in self.records if r.reversed)

        total_events = sum(len(r.extreme_events) for r in self.records)

        return {
            "total_markets": total_markets,
            "markets_with_winner": markets_with_winner,
            "markets_with_extremes": markets_with_extremes,
            "markets_reversed": markets_reversed,
            "total_extreme_events": total_events
        }

    def print_report(
        self,
        min_prob: float = 0.99,
        min_time: int = 0,
        coin_filter: Optional[str] = None
    ):
        """Print a comprehensive analysis report."""
        summary = self.get_summary_stats()

        print("=" * 70)
        print("REVERSAL ANALYSIS REPORT")
        print("=" * 70)
        print()
        print("SUMMARY")
        print("-" * 40)
        print(f"Total markets analyzed:      {summary['total_markets']}")
        print(f"Markets with known winner:   {summary['markets_with_winner']}")
        print(f"Markets with extreme events: {summary['markets_with_extremes']}")
        print(f"Markets with reversals:      {summary['markets_reversed']}")
        print(f"Total extreme events:        {summary['total_extreme_events']}")

        if summary['markets_with_extremes'] > 0:
            rev_rate = 100 * summary['markets_reversed'] / summary['markets_with_extremes']
            print(f"Overall reversal rate:       {rev_rate:.1f}%")

        # Analysis by time
        print()
        print("REVERSAL RATES BY TIME REMAINING")
        print("-" * 40)
        print(f"(Minimum probability: {min_prob*100:.1f}%)")
        print()
        print(f"{'Time Bucket':<12} {'Extremes':>10} {'Reversals':>10} {'Rate':>10}")
        print("-" * 42)

        time_stats = self.analyze_by_time(min_prob=min_prob, coin_filter=coin_filter)
        for name, _, _ in self.TIME_BUCKETS:
            data = time_stats[name]
            extremes = data["extremes"]
            reversals = data["reversals"]
            rate = f"{100*reversals/extremes:.1f}%" if extremes > 0 else "N/A"
            print(f"{name:<12} {extremes:>10} {reversals:>10} {rate:>10}")

        # Analysis by probability
        print()
        print("REVERSAL RATES BY PROBABILITY LEVEL")
        print("-" * 40)
        print(f"(Minimum time remaining: {min_time}s)")
        print()
        print(f"{'Prob Level':<12} {'Extremes':>10} {'Reversals':>10} {'Rate':>10}")
        print("-" * 42)

        prob_stats = self.analyze_by_probability(min_time=min_time, coin_filter=coin_filter)
        for name, _, _ in self.PROB_BUCKETS:
            data = prob_stats[name]
            extremes = data["extremes"]
            reversals = data["reversals"]
            rate = f"{100*reversals/extremes:.1f}%" if extremes > 0 else "N/A"
            print(f"{name:<12} {extremes:>10} {reversals:>10} {rate:>10}")

        # Analysis by coin
        print()
        print("REVERSAL RATES BY COIN")
        print("-" * 40)
        print(f"(Min prob: {min_prob*100:.1f}%, Min time: {min_time}s)")
        print()
        print(f"{'Coin':<8} {'Markets':>10} {'Extremes':>10} {'Reversals':>10} {'Rate':>10}")
        print("-" * 48)

        coin_stats = self.analyze_by_coin(min_prob=min_prob, min_time=min_time)
        for coin in sorted(coin_stats.keys()):
            data = coin_stats[coin]
            markets = data["markets"]
            extremes = data["extremes"]
            reversals = data["reversals"]
            rate = f"{100*reversals/extremes:.1f}%" if extremes > 0 else "N/A"
            print(f"{coin:<8} {markets:>10} {extremes:>10} {reversals:>10} {rate:>10}")

        # Key insights
        print()
        print("KEY INSIGHTS")
        print("-" * 40)

        # Find most dangerous time bucket
        max_rate = 0
        max_bucket = None
        for name, data in time_stats.items():
            if data["extremes"] >= 5:  # Minimum sample size
                rate = data["reversals"] / data["extremes"]
                if rate > max_rate:
                    max_rate = rate
                    max_bucket = name

        if max_bucket:
            print(f"- Highest reversal rate at: {max_bucket} ({max_rate*100:.1f}%)")

        # Check 2+ minute reversal rate
        two_min_plus = {"extremes": 0, "reversals": 0}
        for name, min_s, _ in self.TIME_BUCKETS:
            if min_s >= 120:
                data = time_stats[name]
                two_min_plus["extremes"] += data["extremes"]
                two_min_plus["reversals"] += data["reversals"]

        if two_min_plus["extremes"] > 0:
            rate = 100 * two_min_plus["reversals"] / two_min_plus["extremes"]
            print(
                f"- 99%+ with 2+ min remaining: "
                f"{two_min_plus['reversals']}/{two_min_plus['extremes']} reversed ({rate:.1f}%)"
            )

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze reversal data from recorded files"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to JSONL file or directory containing JSONL files"
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.99,
        help="Minimum probability to consider extreme (default: 0.99)"
    )
    parser.add_argument(
        "--min-time",
        type=int,
        default=0,
        help="Minimum time remaining in seconds (default: 0)"
    )
    parser.add_argument(
        "--coin",
        type=str,
        default=None,
        help="Filter by coin (BTC, ETH, SOL, XRP)"
    )

    args = parser.parse_args()

    analyzer = ReversalAnalyzer()
    path = Path(args.path)

    if path.is_file():
        count = analyzer.load_file(path)
        print(f"Loaded {count} records from {path}")
    elif path.is_dir():
        count = analyzer.load_directory(path)
        print(f"Loaded {count} total records from directory")
    else:
        print(f"Error: Path not found: {path}")
        sys.exit(1)

    if not analyzer.records:
        print("No records to analyze")
        sys.exit(0)

    print()
    analyzer.print_report(
        min_prob=args.min_prob,
        min_time=args.min_time,
        coin_filter=args.coin.upper() if args.coin else None
    )


if __name__ == "__main__":
    main()
