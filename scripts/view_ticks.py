#!/usr/bin/env python3
"""
View recorded tick data.

Usage:
    # View latest ticks for a coin
    python scripts/view_ticks.py --coin BTC --last 10

    # View extreme events
    python scripts/view_ticks.py --extremes --last 20

    # View data stats
    python scripts/view_ticks.py --stats

    # Tail live (like tail -f)
    python scripts/view_ticks.py --coin BTC --follow
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = Path(__file__).parent.parent / "data" / "ticks"
COINS = ["BTC", "ETH", "SOL", "XRP"]


def get_latest_file(coin: str) -> Path:
    """Get the most recent data file for a coin."""
    coin_dir = DATA_DIR / coin.lower()
    if not coin_dir.exists():
        return None
    files = sorted(coin_dir.glob("*.jsonl"), reverse=True)
    return files[0] if files else None


def get_latest_extreme_file() -> Path:
    """Get the most recent extremes file."""
    extremes_dir = DATA_DIR / "extremes"
    if not extremes_dir.exists():
        return None
    files = sorted(extremes_dir.glob("*.jsonl"), reverse=True)
    return files[0] if files else None


def view_last_ticks(coin: str, count: int):
    """View last N ticks for a coin."""
    filepath = get_latest_file(coin)
    if not filepath:
        print(f"No data found for {coin}")
        return

    # Read last N lines
    with open(filepath) as f:
        lines = f.readlines()

    print(f"Last {count} ticks for {coin} ({filepath.name}):")
    print("-" * 100)
    print(f"{'Time':>12} {'Side':>6} {'Mid':>8} {'Spread':>8} {'Bid Depth':>10} {'Ask Depth':>10} {'Time Left':>10}")
    print("-" * 100)

    for line in lines[-count:]:
        try:
            data = json.loads(line)
            ts = data['ts'].split('T')[1][:12]
            print(
                f"{ts:>12} {data['side']:>6} {data['mid']:>8.4f} "
                f"{data['spread']:>8.4f} {data['bid_depth']:>10.0f} "
                f"{data['ask_depth']:>10.0f} {data['time_remaining']:>10}s"
            )
        except:
            pass


def view_extremes(count: int):
    """View recent extreme events."""
    filepath = get_latest_extreme_file()
    if not filepath:
        print("No extreme events recorded")
        return

    with open(filepath) as f:
        lines = f.readlines()

    print(f"Last {count} extreme events ({filepath.name}):")
    print("-" * 110)
    print(f"{'Time':>12} {'Coin':>5} {'Side':>6} {'Price':>8} {'Opposite':>8} {'D/Ratio':>8} {'Time Left':>10} {'Market'}")
    print("-" * 110)

    for line in lines[-count:]:
        try:
            data = json.loads(line)
            ts = data['ts'].split('T')[1][:12]
            print(
                f"{ts:>12} {data['coin']:>5} {data['side']:>6} "
                f"{data['price']:>8.4f} {data['opposite_price']:>8.4f} "
                f"{data['depth_ratio']:>8.1f} {data['time_remaining']:>10}s "
                f"{data['market_slug'][-20:]}"
            )
        except:
            pass


def view_stats():
    """View data statistics."""
    print("=" * 60)
    print("TICK DATA STATISTICS")
    print("=" * 60)

    total_ticks = 0
    total_size = 0

    for coin in COINS:
        coin_dir = DATA_DIR / coin.lower()
        if not coin_dir.exists():
            continue

        files = list(coin_dir.glob("*.jsonl"))
        if not files:
            continue

        # Count ticks and size
        ticks = 0
        size = 0
        oldest = None
        newest = None

        for f in files:
            size += f.stat().st_size
            with open(f) as fp:
                lines = fp.readlines()
                ticks += len(lines)
                if lines:
                    try:
                        if oldest is None:
                            oldest = json.loads(lines[0])['ts']
                        newest = json.loads(lines[-1])['ts']
                    except:
                        pass

        total_ticks += ticks
        total_size += size

        print(f"\n{coin}:")
        print(f"  Files: {len(files)}")
        print(f"  Ticks: {ticks:,}")
        print(f"  Size: {size/1024/1024:.1f} MB")
        if oldest and newest:
            print(f"  Range: {oldest[:10]} to {newest[:10]}")

    # Extremes
    extremes_dir = DATA_DIR / "extremes"
    if extremes_dir.exists():
        files = list(extremes_dir.glob("*.jsonl"))
        extremes = 0
        for f in files:
            with open(f) as fp:
                extremes += len(fp.readlines())
        print(f"\nExtreme Events: {extremes:,}")

    print(f"\n{'='*60}")
    print(f"Total: {total_ticks:,} ticks, {total_size/1024/1024:.1f} MB")


def follow_ticks(coin: str):
    """Follow ticks in real-time (like tail -f)."""
    filepath = get_latest_file(coin)
    if not filepath:
        print(f"No data found for {coin}")
        return

    print(f"Following {coin} ticks (Ctrl+C to stop)...")
    print("-" * 80)

    # Seek to end
    with open(filepath) as f:
        f.seek(0, 2)  # End of file

        while True:
            line = f.readline()
            if line:
                try:
                    data = json.loads(line)
                    ts = data['ts'].split('T')[1][:12]

                    # Color extreme prices
                    mid = data['mid']
                    if mid >= 0.98 or mid <= 0.02:
                        color = "\033[91m"  # Red
                    elif mid >= 0.95 or mid <= 0.05:
                        color = "\033[93m"  # Yellow
                    else:
                        color = ""
                    reset = "\033[0m" if color else ""

                    print(
                        f"{ts} {data['side']:>4} {color}{mid:>7.4f}{reset} "
                        f"spread={data['spread']:.4f} "
                        f"depth={data['bid_depth']:.0f}/{data['ask_depth']:.0f} "
                        f"time={data['time_remaining']}s"
                    )
                except:
                    pass
            else:
                time.sleep(0.1)

                # Check if file rotated
                new_file = get_latest_file(coin)
                if new_file and new_file != filepath:
                    filepath = new_file
                    f.close()
                    f = open(filepath)
                    print(f"\n--- Rotated to {filepath.name} ---\n")


def main():
    parser = argparse.ArgumentParser(description="View recorded tick data")
    parser.add_argument("--coin", type=str, help="Coin to view (BTC, ETH, SOL, XRP)")
    parser.add_argument("--last", type=int, default=20, help="Number of records to show")
    parser.add_argument("--extremes", action="store_true", help="View extreme events")
    parser.add_argument("--stats", action="store_true", help="View statistics")
    parser.add_argument("--follow", action="store_true", help="Follow in real-time")

    args = parser.parse_args()

    if args.stats:
        view_stats()
    elif args.extremes:
        view_extremes(args.last)
    elif args.coin:
        if args.follow:
            try:
                follow_ticks(args.coin.upper())
            except KeyboardInterrupt:
                print("\nStopped")
        else:
            view_last_ticks(args.coin.upper(), args.last)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
