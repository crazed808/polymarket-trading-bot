#!/usr/bin/env python3
"""
Reversal Recorder - Live recording of extreme probability events

Records orderbook data with focus on capturing extreme probability events
(>=99% or <=1%) and tracking whether they reverse before market end.

Usage:
    python scripts/reversal_recorder.py --coin BTC
    python scripts/reversal_recorder.py --coin ALL
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.market_manager import MarketManager, MarketInfo
from src.websocket_client import OrderbookSnapshot


@dataclass
class ExtremeEvent:
    """An event where probability reached an extreme (>=99% or <=1%)."""
    timestamp: str  # ISO format
    side: str  # "up" or "down"
    probability: float
    time_remaining_seconds: int
    best_bid: float
    best_ask: float


@dataclass
class MarketOutcome:
    """Complete record of a market's extreme events and outcome."""
    market_slug: str
    coin: str
    end_time: str
    winner: Optional[str]  # "up" or "down" (None until determined)
    extreme_events: List[ExtremeEvent] = field(default_factory=list)
    reversed: bool = False  # True if 99%+ side lost


class ReversalRecorder:
    """Records extreme probability events and market outcomes."""

    EXTREME_THRESHOLD = 0.99
    MIN_TIME_REMAINING = 5  # Minimum seconds to consider an event

    def __init__(
        self,
        coin: str,
        output_dir: str = "data/reversals",
        threshold: float = 0.99
    ):
        self.coin = coin.upper()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold

        self.manager: Optional[MarketManager] = None
        self.current_outcome: Optional[MarketOutcome] = None
        self.completed_markets: List[MarketOutcome] = []

        self.output_file = self.output_dir / f"{coin.lower()}_reversals.jsonl"
        self.snapshot_count = 0
        self.extreme_count = 0

    def _get_time_remaining(self, market: MarketInfo) -> int:
        """Get seconds remaining until market end."""
        mins, secs = market.get_countdown()
        if mins < 0:
            return 0
        return mins * 60 + secs

    def _determine_winner(self, up_price: float, down_price: float) -> Optional[str]:
        """Determine winner based on final prices."""
        if up_price >= 0.95:
            return "up"
        elif down_price >= 0.95:
            return "down"
        return None

    def _check_reversal(self, outcome: MarketOutcome) -> bool:
        """Check if any extreme event was reversed."""
        if not outcome.winner:
            return False

        for event in outcome.extreme_events:
            # Reversal: side was at 99%+ but ended up losing
            if event.probability >= 0.99 and event.side != outcome.winner:
                return True
        return False

    async def _handle_book_update(self, snapshot: OrderbookSnapshot):
        """Handle orderbook update and check for extreme events."""
        self.snapshot_count += 1

        if not self.manager or not self.manager.current_market:
            return

        market = self.manager.current_market
        time_remaining = self._get_time_remaining(market)

        # Skip if market is ending
        if time_remaining < self.MIN_TIME_REMAINING:
            return

        # Determine which side this is
        side = None
        if snapshot.asset_id == market.up_token:
            side = "up"
        elif snapshot.asset_id == market.down_token:
            side = "down"
        else:
            return

        mid_price = snapshot.mid_price
        if not mid_price:
            return

        # Check for extreme probability
        is_extreme_high = mid_price >= self.threshold
        is_extreme_low = mid_price <= (1 - self.threshold)

        if is_extreme_high or is_extreme_low:
            # Record the extreme event
            event = ExtremeEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                side=side,
                probability=mid_price,
                time_remaining_seconds=time_remaining,
                best_bid=snapshot.best_bid or 0.0,
                best_ask=snapshot.best_ask or 1.0
            )

            if self.current_outcome:
                self.current_outcome.extreme_events.append(event)
                self.extreme_count += 1

                # Log extreme event
                status = f"99%+" if is_extreme_high else "1%-"
                print(
                    f"[{self.coin}] EXTREME: {side.upper()} @ {mid_price:.4f} "
                    f"({status}), {time_remaining}s remaining"
                )

    def _handle_market_change(self, old_slug: str, new_slug: str):
        """Handle market change - finalize old market and start new one."""
        print(f"[{self.coin}] Market changed: {old_slug} -> {new_slug}")

        # Finalize previous market
        if self.current_outcome and old_slug:
            self._finalize_market()

        # Start tracking new market
        if self.manager and self.manager.current_market:
            market = self.manager.current_market
            self.current_outcome = MarketOutcome(
                market_slug=new_slug,
                coin=self.coin,
                end_time=market.end_date,
                winner=None,
                extreme_events=[],
                reversed=False
            )

    def _finalize_market(self):
        """Finalize current market outcome and save."""
        if not self.current_outcome:
            return

        # Try to determine winner from final prices
        if self.manager:
            up_ob = self.manager.get_orderbook("up")
            down_ob = self.manager.get_orderbook("down")

            up_price = up_ob.mid_price if up_ob else 0.5
            down_price = down_ob.mid_price if down_ob else 0.5

            self.current_outcome.winner = self._determine_winner(up_price, down_price)
            self.current_outcome.reversed = self._check_reversal(self.current_outcome)

        # Save to file
        self._save_outcome(self.current_outcome)
        self.completed_markets.append(self.current_outcome)

        # Log completion
        events = len(self.current_outcome.extreme_events)
        status = "REVERSED" if self.current_outcome.reversed else "normal"
        print(
            f"[{self.coin}] Market complete: {self.current_outcome.market_slug} "
            f"- {events} extreme events, winner={self.current_outcome.winner}, {status}"
        )

    def _save_outcome(self, outcome: MarketOutcome):
        """Append outcome to JSONL file."""
        with open(self.output_file, 'a') as f:
            f.write(json.dumps(asdict(outcome)) + '\n')

    async def run(self):
        """Run the reversal recorder."""
        print(f"[{self.coin}] Starting reversal recorder")
        print(f"[{self.coin}] Threshold: {self.threshold}")
        print(f"[{self.coin}] Output: {self.output_file}")
        print()

        while True:
            try:
                # Create market manager
                self.manager = MarketManager(
                    coin=self.coin,
                    auto_switch_market=True,
                    market_check_interval=30.0
                )

                # Register callbacks
                @self.manager.on_book_update
                async def handle_book(snapshot: OrderbookSnapshot):
                    await self._handle_book_update(snapshot)

                @self.manager.on_market_change
                def handle_change(old_slug: str, new_slug: str):
                    self._handle_market_change(old_slug, new_slug)

                # Start manager
                if not await self.manager.start():
                    print(f"[{self.coin}] Failed to start manager, retrying...")
                    await asyncio.sleep(30)
                    continue

                # Initialize first market
                if self.manager.current_market:
                    market = self.manager.current_market
                    self.current_outcome = MarketOutcome(
                        market_slug=market.slug,
                        coin=self.coin,
                        end_time=market.end_date,
                        winner=None,
                        extreme_events=[],
                        reversed=False
                    )
                    print(f"[{self.coin}] Recording: {market.slug}")

                # Run indefinitely with status updates
                while True:
                    await asyncio.sleep(300)  # 5 minute status

                    print(
                        f"[{self.coin}] Status: {self.snapshot_count} snapshots, "
                        f"{self.extreme_count} extreme events, "
                        f"{len(self.completed_markets)} markets completed"
                    )

            except KeyboardInterrupt:
                print(f"\n[{self.coin}] Stopped by user")
                break
            except Exception as e:
                print(f"[{self.coin}] Error: {e}, restarting...")
                await asyncio.sleep(30)
            finally:
                if self.manager:
                    await self.manager.stop()

        # Finalize on exit
        if self.current_outcome:
            self._finalize_market()


async def run_all_coins(coins: List[str], output_dir: str, threshold: float):
    """Run recorders for multiple coins simultaneously."""
    recorders = [
        ReversalRecorder(coin, output_dir=output_dir, threshold=threshold)
        for coin in coins
    ]

    tasks = [recorder.run() for recorder in recorders]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\n\nShutting down all recorders...")


def main():
    parser = argparse.ArgumentParser(
        description="Record extreme probability events and reversals"
    )
    parser.add_argument(
        "--coin",
        type=str,
        default="BTC",
        help="Coin to record (BTC, ETH, SOL, XRP, or ALL)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/reversals",
        help="Output directory"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Extreme probability threshold (default: 0.99)"
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    coins = (
        ["BTC", "ETH", "SOL", "XRP"]
        if args.coin.upper() == "ALL"
        else [args.coin.upper()]
    )

    print("=" * 60)
    print("REVERSAL RECORDER")
    print("=" * 60)
    print(f"Coins: {', '.join(coins)}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {args.output}")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    asyncio.run(run_all_coins(coins, args.output, args.threshold))


if __name__ == "__main__":
    main()
