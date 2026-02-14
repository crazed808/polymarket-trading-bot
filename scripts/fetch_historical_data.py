#!/usr/bin/env python3
"""
Historical Data Fetcher for 15-Minute Market Reversals

Fetches historical price data for closed 15-minute markets and analyzes
how often 99%+ probabilities get reversed with time remaining.

Usage:
    python scripts/fetch_historical_data.py --coin BTC --days 7
    python scripts/fetch_historical_data.py --coin ALL --days 3 --output data/reversals/
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gamma_client import GammaClient
from src.client import ClobClient


@dataclass
class PricePoint:
    """A single price point in the history."""
    timestamp: int  # Epoch seconds
    price: float


@dataclass
class ExtremeEvent:
    """An event where probability reached an extreme (>=99% or <=1%)."""
    timestamp: str  # ISO format
    side: str  # "up" or "down"
    probability: float
    time_remaining_seconds: int


@dataclass
class MarketAnalysis:
    """Analysis result for a single market."""
    slug: str
    coin: str
    end_time: str
    winner: str  # "up" or "down"
    extreme_events: List[ExtremeEvent]
    reversed: bool  # True if any 99%+ side ended up losing


class HistoricalDataFetcher:
    """Fetches and analyzes historical 15-minute market data."""

    COINS = ["BTC", "ETH", "SOL", "XRP"]

    def __init__(self, output_dir: str = "data/historical"):
        self.gamma = GammaClient()
        self.clob = ClobClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def discover_past_markets(
        self,
        coin: str,
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Discover closed 15-minute markets for a coin.

        Args:
            coin: Coin symbol (BTC, ETH, SOL, XRP)
            days_back: Number of days to look back

        Returns:
            List of market info dictionaries
        """
        print(f"Discovering closed {coin} markets from last {days_back} days...")
        markets = self.gamma.get_closed_markets(coin, days_back=days_back)
        print(f"  Found {len(markets)} markets")
        return markets

    def fetch_price_history(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None
    ) -> List[PricePoint]:
        """
        Fetch price history for a token.

        Args:
            token_id: Token ID to fetch history for
            start_ts: Start timestamp (optional)
            end_ts: End timestamp (optional)

        Returns:
            List of PricePoint objects
        """
        try:
            data = self.clob.get_price_history(
                token_id=token_id,
                fidelity=1,  # 1-minute resolution
                start_ts=start_ts,
                end_ts=end_ts
            )

            points = []
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        ts = entry.get("t", 0)
                        price = float(entry.get("p", 0.5))
                        points.append(PricePoint(timestamp=ts, price=price))

            return points
        except Exception as e:
            print(f"    Error fetching price history: {e}")
            return []

    def find_extreme_events(
        self,
        price_history: List[PricePoint],
        side: str,
        end_timestamp: int,
        threshold: float = 0.99
    ) -> List[ExtremeEvent]:
        """
        Find extreme probability events in price history.

        Args:
            price_history: List of price points
            side: "up" or "down"
            end_timestamp: Market end timestamp
            threshold: Probability threshold for extreme events

        Returns:
            List of ExtremeEvent objects
        """
        events = []

        for point in price_history:
            time_remaining = end_timestamp - point.timestamp
            if time_remaining < 0:
                continue  # Skip points after market end

            # Check for extreme probability
            is_extreme = point.price >= threshold

            if is_extreme and time_remaining > 0:
                events.append(ExtremeEvent(
                    timestamp=datetime.fromtimestamp(
                        point.timestamp, tz=timezone.utc
                    ).isoformat(),
                    side=side,
                    probability=point.price,
                    time_remaining_seconds=time_remaining
                ))

        return events

    def analyze_market(
        self,
        market: Dict[str, Any],
        coin: str
    ) -> Optional[MarketAnalysis]:
        """
        Analyze a single market for reversal events.

        Args:
            market: Market data from Gamma API
            coin: Coin symbol

        Returns:
            MarketAnalysis or None if analysis failed
        """
        slug = market.get("slug", "")
        winner = market.get("winner")
        end_date_str = market.get("end_date")
        token_ids = market.get("token_ids", {})

        if not winner or not end_date_str:
            return None

        # Parse end timestamp
        try:
            end_time = datetime.fromisoformat(
                end_date_str.replace('Z', '+00:00')
            )
            end_ts = int(end_time.timestamp())
        except Exception:
            return None

        # Calculate start time (15 minutes before end)
        start_ts = end_ts - 900  # 15 minutes

        extreme_events = []

        # Fetch price history for both sides
        for side in ["up", "down"]:
            token_id = token_ids.get(side)
            if not token_id:
                continue

            history = self.fetch_price_history(token_id, start_ts, end_ts)
            if not history:
                continue

            events = self.find_extreme_events(history, side, end_ts)
            extreme_events.extend(events)

        # Determine if any reversal occurred
        reversed_flag = False
        for event in extreme_events:
            # A reversal occurs if a side at 99%+ ended up losing
            if event.probability >= 0.99 and event.side != winner:
                reversed_flag = True
                break

        return MarketAnalysis(
            slug=slug,
            coin=coin,
            end_time=end_date_str,
            winner=winner,
            extreme_events=extreme_events,
            reversed=reversed_flag
        )

    def fetch_and_analyze(
        self,
        coin: str,
        days_back: int = 7,
        save_results: bool = True
    ) -> List[MarketAnalysis]:
        """
        Fetch historical data and analyze for reversals.

        Args:
            coin: Coin symbol
            days_back: Days to look back
            save_results: Whether to save results to file

        Returns:
            List of MarketAnalysis results
        """
        markets = self.discover_past_markets(coin, days_back)

        results = []
        for i, market in enumerate(markets):
            slug = market.get("slug", "unknown")
            print(f"  [{i+1}/{len(markets)}] Analyzing {slug}...", end="")

            analysis = self.analyze_market(market, coin)
            if analysis:
                results.append(analysis)
                status = "REVERSED" if analysis.reversed else "normal"
                events = len(analysis.extreme_events)
                print(f" {events} extreme events, {status}")
            else:
                print(" skipped")

            # Rate limiting
            time.sleep(0.2)

        if save_results and results:
            output_file = self.output_dir / f"{coin.lower()}_historical.jsonl"
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(asdict(result)) + '\n')
            print(f"  Saved to {output_file}")

        return results

    def print_summary(self, results: List[MarketAnalysis], coin: str):
        """Print analysis summary statistics."""
        if not results:
            print(f"\nNo results for {coin}")
            return

        total = len(results)
        with_extremes = sum(1 for r in results if r.extreme_events)
        reversals = sum(1 for r in results if r.reversed)

        # Group by time remaining buckets
        buckets = {
            "2min+": {"extremes": 0, "reversals": 0},
            "1-2min": {"extremes": 0, "reversals": 0},
            "<1min": {"extremes": 0, "reversals": 0},
        }

        for result in results:
            for event in result.extreme_events:
                secs = event.time_remaining_seconds
                if secs >= 120:
                    bucket = "2min+"
                elif secs >= 60:
                    bucket = "1-2min"
                else:
                    bucket = "<1min"

                buckets[bucket]["extremes"] += 1
                if event.probability >= 0.99 and event.side != result.winner:
                    buckets[bucket]["reversals"] += 1

        print(f"\n{'='*60}")
        print(f"SUMMARY: {coin} ({total} markets analyzed)")
        print(f"{'='*60}")
        print(f"Markets with extreme events (99%+): {with_extremes} ({100*with_extremes/total:.1f}%)")
        print(f"Markets with reversals:             {reversals} ({100*reversals/total:.1f}%)")
        print()
        print("Reversal rates by time remaining:")
        print("-" * 40)

        for bucket, data in buckets.items():
            extremes = data["extremes"]
            revs = data["reversals"]
            rate = f"{100*revs/extremes:.1f}%" if extremes > 0 else "N/A"
            print(f"  {bucket:8s}: {revs}/{extremes} reversals ({rate})")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and analyze historical 15-minute market data"
    )
    parser.add_argument(
        "--coin",
        type=str,
        default="BTC",
        help="Coin to analyze (BTC, ETH, SOL, XRP, or ALL)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to look back"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/historical",
        help="Output directory for results"
    )

    args = parser.parse_args()

    fetcher = HistoricalDataFetcher(output_dir=args.output)

    coins = (
        HistoricalDataFetcher.COINS
        if args.coin.upper() == "ALL"
        else [args.coin.upper()]
    )

    print(f"Fetching historical data for: {', '.join(coins)}")
    print(f"Days back: {args.days}")
    print(f"Output: {args.output}")
    print()

    all_results = {}

    for coin in coins:
        print(f"\n{'='*60}")
        print(f"Processing {coin}...")
        print(f"{'='*60}")

        results = fetcher.fetch_and_analyze(coin, days_back=args.days)
        all_results[coin] = results
        fetcher.print_summary(results, coin)

    # Combined summary if multiple coins
    if len(coins) > 1:
        combined = []
        for results in all_results.values():
            combined.extend(results)
        fetcher.print_summary(combined, "ALL COINS")


if __name__ == "__main__":
    main()
