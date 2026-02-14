#!/usr/bin/env python3
"""
Tick-by-Tick Market Data Recorder

Records every orderbook update from all coin markets with:
- Full orderbook depth (top 10 levels)
- Timestamps, spreads, mid prices
- Extreme event detection and logging
- Automatic 30-day data expiration

Designed to run silently in the background.

Usage:
    # Start recording (background)
    python scripts/tick_recorder.py start

    # Stop recording
    python scripts/tick_recorder.py stop

    # Check status
    python scripts/tick_recorder.py status

    # Run in foreground (for testing)
    python scripts/tick_recorder.py run

Data stored in: data/ticks/{coin}/YYYYMMDD.jsonl
Extreme events: data/ticks/extremes/YYYYMMDD.jsonl
"""

import asyncio
import json
import os
import sys
import signal
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Set
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from lib.market_manager import MarketManager
from src.websocket_client import OrderbookSnapshot


# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "ticks"
PID_FILE = Path(__file__).parent.parent / ".tick_recorder.pid"
LOG_FILE = Path(__file__).parent.parent / "logs" / "tick_recorder.log"
RETENTION_DAYS = 30
COINS = ["BTC", "ETH", "SOL", "XRP"]
EXTREME_THRESHOLD = 0.98  # Log when price >= 98% or <= 2%


@dataclass
class TickRecord:
    """A single tick record."""
    ts: str
    coin: str
    side: str  # "up" or "down"
    asset_id: str
    best_bid: float
    best_ask: float
    mid: float
    spread: float
    bid_depth: float  # Total size top 10
    ask_depth: float
    bids: list  # Top 10 [[price, size], ...]
    asks: list
    time_remaining: int  # Seconds until market end
    market_slug: str


@dataclass
class ExtremeEvent:
    """An extreme price event."""
    ts: str
    coin: str
    side: str
    price: float
    opposite_price: float
    time_remaining: int
    market_slug: str
    bid_depth: float
    ask_depth: float
    depth_ratio: float


class TickRecorder:
    """Records tick-by-tick market data."""

    def __init__(self):
        self.markets: Dict[str, MarketManager] = {}
        self.running = False
        self._file_handles: Dict[str, any] = {}
        self._current_date: Optional[str] = None
        self._extreme_prices: Dict[str, float] = {}  # Track last extreme to avoid spam
        self._tick_counts: Dict[str, int] = {coin: 0 for coin in COINS}
        self._extreme_counts: Dict[str, int] = {coin: 0 for coin in COINS}

        # Ensure directories exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "extremes").mkdir(exist_ok=True)
        for coin in COINS:
            (DATA_DIR / coin.lower()).mkdir(exist_ok=True)
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, msg: str, level: str = "INFO"):
        """Log to file."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {msg}\n"
        with open(LOG_FILE, "a") as f:
            f.write(line)

    def _get_file(self, coin: str, date_str: str):
        """Get or create file handle for coin/date."""
        key = f"{coin}_{date_str}"
        if key not in self._file_handles:
            filepath = DATA_DIR / coin.lower() / f"{date_str}.jsonl"
            self._file_handles[key] = open(filepath, "a", buffering=1)  # Line buffered
        return self._file_handles[key]

    def _get_extreme_file(self, date_str: str):
        """Get file handle for extreme events."""
        key = f"extreme_{date_str}"
        if key not in self._file_handles:
            filepath = DATA_DIR / "extremes" / f"{date_str}.jsonl"
            self._file_handles[key] = open(filepath, "a", buffering=1)
        return self._file_handles[key]

    def _rotate_files_if_needed(self):
        """Close and reopen files if date changed."""
        today = datetime.now().strftime("%Y%m%d")
        if self._current_date != today:
            # Close old handles
            for handle in self._file_handles.values():
                handle.close()
            self._file_handles.clear()
            self._current_date = today
            self._log(f"Rotated to new date: {today}")

    def _cleanup_old_data(self):
        """Delete data files older than RETENTION_DAYS."""
        cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)
        cutoff_str = cutoff.strftime("%Y%m%d")
        deleted = 0

        for subdir in DATA_DIR.iterdir():
            if not subdir.is_dir():
                continue
            for filepath in subdir.glob("*.jsonl"):
                # Extract date from filename (YYYYMMDD.jsonl)
                try:
                    file_date = filepath.stem
                    if file_date < cutoff_str:
                        filepath.unlink()
                        deleted += 1
                except Exception:
                    pass

        if deleted > 0:
            self._log(f"Cleaned up {deleted} files older than {RETENTION_DAYS} days")

    def _record_tick(self, coin: str, side: str, snapshot: OrderbookSnapshot,
                     market_slug: str, time_remaining: int):
        """Record a single tick."""
        self._rotate_files_if_needed()

        # Build record
        bids = [[level.price, level.size] for level in snapshot.bids[:10]]
        asks = [[level.price, level.size] for level in snapshot.asks[:10]]

        bid_depth = sum(level.size for level in snapshot.bids[:10])
        ask_depth = sum(level.size for level in snapshot.asks[:10])

        best_bid = snapshot.bids[0].price if snapshot.bids else 0
        best_ask = snapshot.asks[0].price if snapshot.asks else 1
        spread = best_ask - best_bid

        record = TickRecord(
            ts=datetime.now(timezone.utc).isoformat(),
            coin=coin,
            side=side,
            asset_id=snapshot.asset_id,
            best_bid=best_bid,
            best_ask=best_ask,
            mid=snapshot.mid_price,
            spread=spread,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            bids=bids,
            asks=asks,
            time_remaining=time_remaining,
            market_slug=market_slug,
        )

        # Write to file
        f = self._get_file(coin, self._current_date)
        f.write(json.dumps(asdict(record)) + "\n")
        self._tick_counts[coin] += 1

        # Check for extreme
        self._check_extreme(coin, side, snapshot, market_slug, time_remaining,
                           bid_depth, ask_depth)

    def _check_extreme(self, coin: str, side: str, snapshot: OrderbookSnapshot,
                       market_slug: str, time_remaining: int,
                       bid_depth: float, ask_depth: float):
        """Check and log extreme price events."""
        price = snapshot.mid_price
        key = f"{coin}_{side}"

        # Is this an extreme?
        is_extreme = price >= EXTREME_THRESHOLD or price <= (1 - EXTREME_THRESHOLD)

        if not is_extreme:
            # Clear tracking when not extreme
            if key in self._extreme_prices:
                del self._extreme_prices[key]
            return

        # Avoid logging same extreme repeatedly (only log on significant change)
        last_extreme = self._extreme_prices.get(key, 0)
        if abs(price - last_extreme) < 0.005:  # Within 0.5%
            return

        self._extreme_prices[key] = price

        # Calculate opposite price
        opposite_price = 1 - price

        # Depth ratio
        depth_ratio = bid_depth / ask_depth if ask_depth > 0 else 0

        event = ExtremeEvent(
            ts=datetime.now(timezone.utc).isoformat() + "Z",
            coin=coin,
            side=side,
            price=price,
            opposite_price=opposite_price,
            time_remaining=time_remaining,
            market_slug=market_slug,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            depth_ratio=depth_ratio,
        )

        # Write to extremes file
        f = self._get_extreme_file(self._current_date)
        f.write(json.dumps(asdict(event)) + "\n")
        self._extreme_counts[coin] += 1

    async def _run_coin(self, coin: str):
        """Run recorder for a single coin."""
        market = MarketManager(coin=coin)
        self.markets[coin] = market

        @market.on_book_update
        async def handle_book(snapshot: OrderbookSnapshot):
            try:
                # Determine which side this is
                for side, token_id in market.token_ids.items():
                    if token_id == snapshot.asset_id:
                        # Get time remaining
                        time_remaining = 0
                        if market.current_market:
                            mins, secs = market.current_market.get_countdown()
                            time_remaining = max(0, mins * 60 + secs)

                        market_slug = market.current_market.slug if market.current_market else ""
                        self._record_tick(coin, side, snapshot, market_slug, time_remaining)
                        break
            except Exception as e:
                self._log(f"Error recording {coin} tick: {e}", "ERROR")

        if not await market.start():
            self._log(f"Failed to start {coin} market", "ERROR")
            return

        self._log(f"Started recording {coin}")

        # Keep running
        while self.running:
            await asyncio.sleep(1)

        await market.stop()
        self._log(f"Stopped recording {coin}")

    async def run(self):
        """Main run loop."""
        self.running = True
        self._current_date = datetime.now().strftime("%Y%m%d")

        self._log("=" * 50)
        self._log("Tick Recorder Started")
        self._log(f"Recording: {', '.join(COINS)}")
        self._log(f"Data dir: {DATA_DIR}")
        self._log(f"Retention: {RETENTION_DAYS} days")
        self._log("=" * 50)

        # Initial cleanup
        self._cleanup_old_data()

        # Schedule daily cleanup
        async def cleanup_loop():
            while self.running:
                await asyncio.sleep(3600)  # Every hour
                self._cleanup_old_data()

        # Start all coin recorders
        tasks = [asyncio.create_task(self._run_coin(coin)) for coin in COINS]
        tasks.append(asyncio.create_task(cleanup_loop()))

        # Status logging every 5 minutes
        async def status_loop():
            while self.running:
                await asyncio.sleep(300)
                total_ticks = sum(self._tick_counts.values())
                total_extremes = sum(self._extreme_counts.values())
                self._log(f"Status: {total_ticks} ticks, {total_extremes} extremes recorded")

        tasks.append(asyncio.create_task(status_loop()))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            # Close all file handles
            for handle in self._file_handles.values():
                handle.close()
            self._file_handles.clear()

            total_ticks = sum(self._tick_counts.values())
            total_extremes = sum(self._extreme_counts.values())
            self._log(f"Recorder stopped. Total: {total_ticks} ticks, {total_extremes} extremes")

    def stop(self):
        """Signal stop."""
        self.running = False
        for market in self.markets.values():
            asyncio.create_task(market.stop())


def write_pid():
    """Write current PID to file."""
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def read_pid() -> Optional[int]:
    """Read PID from file."""
    if not PID_FILE.exists():
        return None
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except:
        return None


def is_running(pid: int) -> bool:
    """Check if process with PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cmd_start():
    """Start recorder in background."""
    pid = read_pid()
    if pid and is_running(pid):
        print(f"Recorder already running (PID {pid})")
        return

    # Fork to background
    if os.fork() > 0:
        print("Starting tick recorder in background...")
        time.sleep(1)
        pid = read_pid()
        if pid and is_running(pid):
            print(f"Recorder started (PID {pid})")
            print(f"Log file: {LOG_FILE}")
            print(f"Data dir: {DATA_DIR}")
        else:
            print("Failed to start recorder. Check logs.")
        return

    # Child process
    os.setsid()

    # Second fork
    if os.fork() > 0:
        os._exit(0)

    # Redirect stdout/stderr to log
    sys.stdout = open(LOG_FILE, "a")
    sys.stderr = sys.stdout

    write_pid()

    recorder = TickRecorder()

    def handle_signal(sig, frame):
        recorder.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        asyncio.run(recorder.run())
    finally:
        if PID_FILE.exists():
            PID_FILE.unlink()


def cmd_stop():
    """Stop recorder."""
    pid = read_pid()
    if not pid:
        print("Recorder not running (no PID file)")
        return

    if not is_running(pid):
        print("Recorder not running (stale PID file)")
        PID_FILE.unlink()
        return

    print(f"Stopping recorder (PID {pid})...")
    os.kill(pid, signal.SIGTERM)

    # Wait for stop
    for _ in range(10):
        time.sleep(0.5)
        if not is_running(pid):
            print("Recorder stopped")
            return

    print("Recorder did not stop gracefully, sending SIGKILL...")
    os.kill(pid, signal.SIGKILL)
    time.sleep(0.5)
    if PID_FILE.exists():
        PID_FILE.unlink()
    print("Recorder killed")


def cmd_status():
    """Show recorder status."""
    pid = read_pid()

    print("=" * 50)
    print("TICK RECORDER STATUS")
    print("=" * 50)

    if pid and is_running(pid):
        print(f"Status: RUNNING (PID {pid})")
    else:
        print("Status: STOPPED")

    print(f"Data dir: {DATA_DIR}")
    print(f"Log file: {LOG_FILE}")
    print()

    # Show data stats
    print("Data Statistics:")
    total_size = 0
    for coin in COINS:
        coin_dir = DATA_DIR / coin.lower()
        if coin_dir.exists():
            files = list(coin_dir.glob("*.jsonl"))
            size = sum(f.stat().st_size for f in files)
            total_size += size
            print(f"  {coin}: {len(files)} files, {size/1024/1024:.1f} MB")

    extremes_dir = DATA_DIR / "extremes"
    if extremes_dir.exists():
        files = list(extremes_dir.glob("*.jsonl"))
        size = sum(f.stat().st_size for f in files)
        total_size += size
        print(f"  Extremes: {len(files)} files, {size/1024:.1f} KB")

    print(f"  Total: {total_size/1024/1024:.1f} MB")

    # Show recent log entries
    if LOG_FILE.exists():
        print()
        print("Recent Log Entries:")
        with open(LOG_FILE) as f:
            lines = f.readlines()
            for line in lines[-5:]:
                print(f"  {line.rstrip()}")


def cmd_run():
    """Run recorder in foreground (for testing)."""
    print("Running tick recorder in foreground (Ctrl+C to stop)...")
    print(f"Data dir: {DATA_DIR}")
    print()

    recorder = TickRecorder()

    def handle_signal(sig, frame):
        print("\nStopping...")
        recorder.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    asyncio.run(recorder.run())


def main():
    if len(sys.argv) < 2:
        print("Usage: python tick_recorder.py <command>")
        print()
        print("Commands:")
        print("  start   - Start recorder in background")
        print("  stop    - Stop recorder")
        print("  status  - Show recorder status")
        print("  run     - Run in foreground (for testing)")
        return

    cmd = sys.argv[1].lower()

    if cmd == "start":
        cmd_start()
    elif cmd == "stop":
        cmd_stop()
    elif cmd == "status":
        cmd_status()
    elif cmd == "run":
        cmd_run()
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
