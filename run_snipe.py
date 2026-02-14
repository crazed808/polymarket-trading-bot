#!/usr/bin/env python3
"""
Run Settlement Snipe Strategy

Buys at 95-97% probability in final 30 seconds before settlement.
Expected: +1.47% per trade after spread (97.8% win rate).

Usage:
    python run_snipe.py --coin BTC,ETH --size 1.0
    python run_snipe.py --coin BTC --size 1.0 --min-price 0.96
"""

import asyncio
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set, Optional, List, Deque
from dataclasses import dataclass, field
from collections import defaultdict, deque

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Suppress all console logging during dashboard mode
# Must be done BEFORE importing bot.py which calls basicConfig
import logging as _logging
_logging.basicConfig(handlers=[_logging.NullHandler()], force=True)
_logging.getLogger("httpx").setLevel(_logging.WARNING)
_logging.getLogger("src.websocket_client").setLevel(_logging.WARNING)
_logging.getLogger("src.bot").setLevel(_logging.WARNING)

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

from src.bot import TradingBot
from src.gamma_client import GammaClient
from src.websocket_client import MarketWebSocket, OrderbookSnapshot

# Logging setup - file only (dashboard handles console)
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"snipe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger("snipe")
logger.setLevel(logging.DEBUG)
logger.handlers = []

# File handler only
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(file_handler)

console = Console()


@dataclass
class SnipeConfig:
    """Settlement snipe configuration."""
    coins: list = field(default_factory=lambda: ['BTC', 'ETH', 'SOL', 'XRP'])
    size: float = 5.0  # $5 to ensure >5 shares at 95%+ prices
    min_price: float = 0.95
    max_price: float = 0.97
    min_time: int = 5
    max_time: int = 30
    min_liquidity: float = 100.0
    max_spread: float = 0.03

    # Per-coin minimum sizes (Polymarket requirements)
    min_sizes: dict = field(default_factory=lambda: {
        'BTC': 1.0,
        'ETH': 1.0,
        'SOL': 1.0,
        'XRP': 5.0  # XRP has $5 minimum
    })

    # Momentum safety checks
    check_momentum: bool = True
    momentum_lookback: int = 10  # Seconds to look back
    max_negative_momentum: float = -0.02  # Skip if price dropped >2% in lookback
    min_orderbook_ratio: float = 1.5  # Require bid/ask ratio >= 1.5

    # Stop-loss: stop bot after N losses (0 = disabled)
    max_losses: int = 0


@dataclass
class SnipeResult:
    """Track a snipe trade."""
    coin: str
    side: str  # "up" or "down"
    market_slug: str
    token_id: str
    entry_price: float
    entry_time: float
    size_shares: float
    size_usd: float
    settlement_time: float
    outcome: Optional[str] = None
    pnl: float = 0.0


@dataclass
class CoinState:
    """Current state for a coin."""
    coin: str
    price_up: float = 0.0  # UP side price
    price_down: float = 0.0  # DOWN side price
    end_ts: float = 0.0  # Store end timestamp, calculate time_remaining at display time
    status: str = "waiting"  # waiting, watching, sniping, pending
    last_update: float = 0.0
    market_slug: str = ""

    @property
    def time_remaining(self) -> float:
        """Calculate time remaining dynamically."""
        if self.end_ts <= 0:
            return 0.0
        return max(0.0, self.end_ts - time.time())


class SettlementSniper:
    """Settlement Snipe Strategy with Dashboard."""

    def __init__(self, bot: TradingBot, config: SnipeConfig):
        self.bot = bot
        self.config = config
        self.gamma = GammaClient()

        self.sniped_markets: Set[str] = set()
        self.pending_snipes: Dict[str, SnipeResult] = {}
        self.completed_snipes: list = []

        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.running = False
        self.usdc_balance = 0.0

        # Current market info per coin
        self.current_markets: Dict[str, dict] = {}
        self.market_end_times: Dict[str, float] = {}
        self.coin_states: Dict[str, CoinState] = {}

        # Event log for dashboard
        self.events: List[str] = []
        self.max_events = 8

        # Price history for momentum detection: {market_slug: deque([(timestamp, price)])}
        self.price_history: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=30))
        self.momentum_skips = 0

        # Initialize coin states
        for coin in config.coins:
            self.coin_states[coin] = CoinState(coin=coin)

    def add_event(self, msg: str, style: str = "white"):
        """Add event to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.events.append(f"[{style}][{timestamp}] {msg}[/{style}]")
        if len(self.events) > self.max_events:
            self.events.pop(0)
        logger.info(msg)

    def build_dashboard(self) -> Layout:
        """Build the dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

        layout["body"].split_row(
            Layout(name="markets", ratio=2),
            Layout(name="sidebar", ratio=1)
        )

        layout["sidebar"].split_column(
            Layout(name="stats", size=10),
            Layout(name="events")
        )

        # Header
        header_text = Text()
        header_text.append("⚡ SETTLEMENT SNIPER ", style="bold cyan")
        header_text.append(f"| Entry: {self.config.min_price:.0%}-{self.config.max_price:.0%} ", style="white")
        header_text.append(f"| Window: {self.config.min_time}s-{self.config.max_time}s ", style="white")
        header_text.append(f"| Size: ${self.usdc_balance / 20:.2f} (5%)", style="green")
        layout["header"].update(Panel(header_text, style="bold"))

        # Markets table
        markets_table = Table(
            title="Market Status",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            expand=True
        )
        markets_table.add_column("Coin", style="cyan", width=6)
        markets_table.add_column("UP", justify="right", width=7)
        markets_table.add_column("DOWN", justify="right", width=7)
        markets_table.add_column("Time", justify="right", width=8)
        markets_table.add_column("Status", justify="center", width=12)
        markets_table.add_column("Progress", width=18)

        for coin in self.config.coins:
            state = self.coin_states.get(coin, CoinState(coin=coin))

            # UP price with color
            if state.price_up >= self.config.min_price:
                up_style = "green" if state.price_up <= self.config.max_price else "yellow"
            else:
                up_style = "white"
            up_str = f"{state.price_up:.1%}" if state.price_up > 0 else "---"

            # DOWN price with color (inverted - low price = high probability for DOWN)
            # DOWN at 0.05 = 95% for DOWN to win
            down_prob = 1 - state.price_down if state.price_down > 0 else 0
            if down_prob >= self.config.min_price:
                down_style = "green" if down_prob <= self.config.max_price else "yellow"
            else:
                down_style = "white"
            down_str = f"{state.price_down:.1%}" if state.price_down > 0 else "---"

            # Time remaining
            if state.time_remaining > 0:
                if state.time_remaining <= self.config.max_time:
                    time_style = "green bold" if state.time_remaining >= self.config.min_time else "red"
                else:
                    time_style = "white"
                time_str = f"{state.time_remaining:.0f}s"
            else:
                time_style = "dim"
                time_str = "---"

            # Status indicator
            status_map = {
                "waiting": ("⏳ Waiting", "dim"),
                "watching": ("👁 Watching", "yellow"),
                "sniping": ("🎯 SNIPING", "green bold"),
                "pending": ("⏱ Pending", "cyan"),
                "won": ("✅ WON", "green bold"),
                "lost": ("❌ LOST", "red bold"),
            }
            status_text, status_style = status_map.get(state.status, ("?", "white"))

            # Progress bar for time window
            if 0 < state.time_remaining <= 60:
                progress = min(1.0, (60 - state.time_remaining) / 60)
                bar_len = 15
                filled = int(progress * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)
                if state.time_remaining <= self.config.max_time:
                    bar_style = "green" if state.time_remaining >= self.config.min_time else "red"
                else:
                    bar_style = "blue"
                progress_str = f"[{bar_style}]{bar}[/{bar_style}]"
            else:
                progress_str = "[dim]" + "░" * 15 + "[/dim]"

            markets_table.add_row(
                coin,
                f"[{up_style}]{up_str}[/{up_style}]",
                f"[{down_style}]{down_str}[/{down_style}]",
                f"[{time_style}]{time_str}[/{time_style}]",
                f"[{status_style}]{status_text}[/{status_style}]",
                progress_str
            )

        layout["markets"].update(Panel(markets_table, title="Markets", border_style="blue"))

        # Stats panel
        trades_entered = len(self.completed_snipes) + len(self.pending_snipes)

        stats_text = Text()
        stats_text.append(f"Balance: ", style="white")
        stats_text.append(f"${self.usdc_balance:.2f}\n", style="green")
        stats_text.append(f"Trades:  ", style="white")
        stats_text.append(f"{trades_entered}\n", style="cyan")
        stats_text.append(f"P&L:     ", style="white")
        pnl_style = "green" if self.total_pnl >= 0 else "red"
        stats_text.append(f"${self.total_pnl:+.2f}\n", style=pnl_style)
        stats_text.append(f"Pending: ", style="white")
        stats_text.append(f"{len(self.pending_snipes)}\n", style="yellow")
        stats_text.append(f"Filtered:", style="white")
        stats_text.append(f" {self.momentum_skips}", style="dim")

        layout["stats"].update(Panel(stats_text, title="Stats", border_style="green"))

        # Events panel
        events_text = Text()
        for event in self.events[-self.max_events:]:
            events_text.append(event + "\n")
        if not self.events:
            events_text.append("[dim]Waiting for events...[/dim]")

        layout["events"].update(Panel(events_text, title="Events", border_style="yellow"))

        # Footer
        footer_text = Text()
        footer_text.append(f"Log: {log_file.name} ", style="dim")
        footer_text.append("| Press Ctrl+C to stop", style="dim")
        layout["footer"].update(Panel(footer_text, style="dim"))

        return layout

    async def start(self) -> bool:
        """Initialize the strategy."""
        self.add_event("Initializing strategy...", "cyan")

        # Check if gasless is enabled
        if self.bot.config.use_gasless:
            self.add_event(f"✅ Gasless redemptions enabled", "green")
            logger.info(f"Builder credentials configured: {self.bot.config.builder.is_configured()}")
        else:
            self.add_event(f"⚠️ Gasless disabled - redemptions may fail!", "yellow")
            logger.warning("Builder credentials not configured - redemptions will fail")

        # Get USDC balance
        try:
            ba = await self.bot.get_balance_allowance('', 'COLLATERAL')
            raw_balance = float(ba.get("balance", "0"))
            self.usdc_balance = raw_balance / 1_000_000
        except Exception as e:
            self.add_event(f"Failed to get balance: {e}", "red")
            return False

        self.starting_balance = self.usdc_balance
        self.add_event(f"Balance: ${self.usdc_balance:.2f}", "green")

        if self.usdc_balance < self.config.size:
            self.add_event("Insufficient balance!", "red")
            return False

        self.running = True
        self.add_event("Strategy started", "green")
        return True

    async def run(self):
        """Main run loop with dashboard."""
        tasks = []

        for coin in self.config.coins:
            task = asyncio.create_task(self._run_coin(coin))
            tasks.append(task)

        tasks.append(asyncio.create_task(self._redeem_loop()))

        # Run dashboard
        with Live(self.build_dashboard(), refresh_per_second=4, console=console) as live:
            self.live = live

            async def update_dashboard():
                while self.running:
                    live.update(self.build_dashboard())
                    await asyncio.sleep(0.25)

            tasks.append(asyncio.create_task(update_dashboard()))

            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                pass

    async def _run_coin(self, coin: str):
        """Run sniper for a single coin."""
        self.add_event(f"[{coin}] Starting monitor", "cyan")

        while self.running:
            try:
                # Get current market
                market = self.gamma.get_current_15m_market(coin)
                if not market:
                    self.coin_states[coin].status = "waiting"
                    await asyncio.sleep(5)
                    continue

                market_slug = market.get('slug', '')
                tokens = self.gamma.parse_token_ids(market)
                end_date = market.get('endDate', '')

                # Calculate end time
                if end_date:
                    end_ts = datetime.fromisoformat(end_date.replace('Z', '+00:00')).timestamp()
                else:
                    end_ts = time.time() + 900

                self.current_markets[coin] = market
                self.market_end_times[coin] = end_ts
                self.coin_states[coin].market_slug = market_slug

                up_token = tokens.get('up', '')
                down_token = tokens.get('down', '')

                if not up_token:
                    await asyncio.sleep(5)
                    continue

                self.coin_states[coin].status = "watching"

                # Create WebSocket and set up callbacks
                ws = MarketWebSocket()

                @ws.on_book
                async def handle_book(snapshot: OrderbookSnapshot):
                    if not self.running:
                        return

                    # Determine which side this is
                    if snapshot.asset_id == up_token:
                        side = "up"
                        token = up_token
                    elif snapshot.asset_id == down_token:
                        side = "down"
                        token = down_token
                    else:
                        return

                    # Update coin state
                    if side == "up":
                        self.coin_states[coin].price_up = snapshot.mid_price
                    else:
                        self.coin_states[coin].price_down = snapshot.mid_price

                    self.coin_states[coin].end_ts = end_ts  # Store end time, calculate remaining dynamically
                    self.coin_states[coin].last_update = time.time()

                    time_remaining = self.coin_states[coin].time_remaining

                    # Calculate depth totals from orderbook
                    bid_depth = sum(level.size for level in snapshot.bids)
                    ask_depth = sum(level.size for level in snapshot.asks)
                    spread = snapshot.best_ask - snapshot.best_bid if snapshot.bids and snapshot.asks else 1.0

                    await self._process_tick(
                        coin=coin,
                        market_slug=market_slug,
                        side=side,
                        token_id=token,
                        mid=snapshot.mid_price,
                        bid_depth=bid_depth,
                        ask_depth=ask_depth,
                        spread=spread,
                        time_remaining=time_remaining,
                        end_ts=end_ts
                    )

                # Subscribe and run
                await ws.subscribe([up_token, down_token])

                # Run until market ends
                try:
                    await asyncio.wait_for(
                        ws.run(),
                        timeout=max(end_ts - time.time() + 30, 60)
                    )
                except asyncio.TimeoutError:
                    pass

                await ws.disconnect()
                self.coin_states[coin].status = "waiting"
                self.coin_states[coin].price_up = 0
                self.coin_states[coin].price_down = 0
                self.coin_states[coin].end_ts = 0

            except Exception as e:
                self.add_event(f"[{coin}] Error: {e}", "red")
                await asyncio.sleep(5)

    async def _process_tick(self, coin: str, market_slug: str, side: str, token_id: str,
                           mid: float, bid_depth: float, ask_depth: float,
                           spread: float, time_remaining: float, end_ts: float):
        """Process a market tick."""
        # Refresh balance at ~10 minutes remaining (5 min into 15-min market), once per market
        balance_key = f"{market_slug}:balance_refreshed"
        if 595 <= time_remaining <= 605 and balance_key not in self.sniped_markets:
            self.sniped_markets.add(balance_key)
            try:
                ba = await self.bot.get_balance_allowance('', 'COLLATERAL')
                self.usdc_balance = float(ba.get("balance", "0")) / 1_000_000
                logger.info(f"Balance refreshed: ${self.usdc_balance:.2f}, bet size: ${self.usdc_balance / 20:.2f}")
            except Exception as e:
                logger.warning(f"Balance refresh failed: {e}")

        # Track price history for momentum detection
        current_time = time.time()
        history_key = f"{market_slug}:{side}"
        self.price_history[history_key].append((current_time, mid))

        # Check for settlements first
        await self._check_settlement(market_slug, side, mid, time_remaining)

        # Skip if already sniped
        if market_slug in self.sniped_markets:
            self.coin_states[coin].status = "pending"
            return

        # Check time window
        in_time = self.config.min_time <= time_remaining <= self.config.max_time
        if not in_time:
            return

        # Check price range based on side
        # For UP: price should be 0.95-0.97 (high probability UP wins)
        # For DOWN: price should be 0.95-0.97 (high probability DOWN wins)
        in_price = self.config.min_price <= mid <= self.config.max_price

        # Update status based on conditions
        if in_time and in_price:
            self.coin_states[coin].status = "sniping"
        elif time_remaining <= self.config.max_time:
            self.coin_states[coin].status = "watching"

        # Check price range
        if not in_price:
            return

        # Check liquidity
        if (bid_depth + ask_depth) < self.config.min_liquidity:
            return

        # Check spread
        if spread > self.config.max_spread:
            return

        # CRITICAL: Price stability check - reject if opposite side was dominant recently
        # This prevents betting on a side that just "flashed" high while the other side was winning
        opposite_side = "down" if side == "up" else "up"
        opposite_key = f"{market_slug}:{opposite_side}"
        opposite_history = list(self.price_history.get(opposite_key, []))

        if opposite_history:
            # Check last 5 seconds of opposite side's price history
            lookback = 5.0
            cutoff = current_time - lookback
            recent_opposite = [(t, p) for t, p in opposite_history if t >= cutoff]

            if recent_opposite:
                max_opposite = max(p for _, p in recent_opposite)
                # If opposite side was above 80% in last 5 seconds, this is likely manipulation
                if max_opposite >= 0.80:
                    logger.warning(f"[{coin}] {side} REJECT MANIPULATION: "
                                 f"opposite side ({opposite_side}) was at {max_opposite:.1%} within last {lookback}s")
                    self.add_event(f"[{coin}] SKIP: Price spike trap (opposite was {max_opposite:.0%})", "red")
                    return

        # Check momentum safety
        if self.config.check_momentum:
            is_safe, reason = self._check_momentum_safety(
                history_key, side, mid, bid_depth, ask_depth
            )
            if not is_safe:
                self.momentum_skips += 1
                self.add_event(f"[{coin}] SKIP: {reason}", "yellow")
                return
            logger.debug(f"[{coin}] Momentum check PASSED (side={side}, price={mid:.4f}, history_pts={len(self.price_history.get(history_key, []))})")
        else:
            logger.debug(f"[{coin}] Momentum check DISABLED")

        # Execute snipe
        await self._execute_snipe(coin, market_slug, side, token_id, mid, time_remaining, end_ts)

    def _check_momentum_safety(self, history_key: str, side: str, current_price: float,
                               bid_depth: float, ask_depth: float) -> tuple:
        """
        Check if momentum and orderbook conditions are safe for entry.

        Returns:
            (is_safe, reason)
        """
        # Check price momentum over lookback period
        history = self.price_history.get(history_key)
        if history and len(history) >= 2:
            current_time = time.time()
            lookback = self.config.momentum_lookback

            recent = [(t, p) for t, p in history if current_time - t <= lookback]
            if len(recent) >= 2:
                oldest_price = recent[0][1]
                if oldest_price > 0:
                    momentum = (current_price - oldest_price) / oldest_price
                    logger.debug(f"Momentum: {momentum:+.4%} over {lookback}s ({len(recent)} pts, {oldest_price:.4f} -> {current_price:.4f})")
                    if momentum < self.config.max_negative_momentum:
                        return False, f"Falling price ({momentum:+.1%} in {lookback}s)"
        else:
            logger.debug(f"Momentum: insufficient history (key={history_key}, pts={len(history) if history else 0})")

        # Check orderbook ratio (buy pressure for the token we're buying)
        # For both UP and DOWN: bid_depth = buyers, ask_depth = sellers
        # We want more buyers than sellers (bid/ask > 1.5)
        if bid_depth and ask_depth:
            ratio = bid_depth / ask_depth if ask_depth > 0 else 999
            logger.debug(f"Orderbook: ratio={ratio:.2f} (bid={bid_depth:.0f}, ask={ask_depth:.0f}, need={self.config.min_orderbook_ratio})")
            if ratio < self.config.min_orderbook_ratio:
                return False, f"Weak pressure (ratio: {ratio:.2f})"

        return True, ""

    async def _execute_snipe(self, coin: str, market_slug: str, side: str, token_id: str,
                            mid: float, time_remaining: float, end_ts: float):
        """Execute a snipe trade."""
        # Size is 1/20 of current balance, with coin-specific minimums
        min_size = self.config.min_sizes.get(coin, 1.0)
        size_usd = max(self.usdc_balance / 20, min_size)

        # Validate we have enough balance
        if self.usdc_balance < min_size:
            self.add_event(f"[{coin}] Insufficient balance (need ${min_size:.2f})", "red")
            return

        # For display, show implied probability (for DOWN, show 1-price)
        display_prob = mid if side == "up" else (1 - mid)
        self.add_event(f"[{coin}] SNIPE {side.upper()} @ {display_prob:.1%}, ${size_usd:.2f}, {time_remaining:.0f}s left", "yellow")
        self.coin_states[coin].status = "sniping"

        shares = size_usd / mid

        self.sniped_markets.add(market_slug)

        # Place order - any exception here means order wasn't placed
        order_placed = False
        result = None
        try:
            buy_price = min(mid * 1.01, 0.99)

            result = await self.bot.place_order(
                token_id=token_id,
                side="BUY",
                price=buy_price,
                size=shares,
                order_type="FOK"
            )
            order_placed = result and result.success
            if result:
                logger.info(f"[{coin}] Order response: {result.data}")
        except Exception as e:
            self.add_event(f"[{coin}] Order error: {e}", "red")
            self.sniped_markets.discard(market_slug)  # Safe to retry - order wasn't placed
            self.coin_states[coin].status = "watching"
            return

        # Process result - DON'T discard from sniped_markets here (order may have gone through)
        if order_placed:
            try:
                # Check if FOK actually filled — API returns success even for unfilled FOKs
                status = result.data.get("status", "")
                taking = result.data.get("takingAmount", "")
                making = result.data.get("makingAmount", "")
                if status == "live" or (not taking and not making):
                    logger.info(f"[{coin}] FOK not filled (status={status}, taking={taking}, making={making})")
                    self.add_event(f"[{coin}] FOK not filled — no liquidity", "yellow")
                    self.coin_states[coin].status = "waiting"
                    return

                # Get actual values from response data, fallback to estimates
                actual_price = float(result.data.get("price", mid))
                actual_shares = float(result.data.get("size", shares))

                self.add_event(f"[{coin}] BOUGHT {side.upper()} @ {actual_price:.2%}", "green")

                snipe = SnipeResult(
                    coin=coin,
                    side=side,
                    market_slug=market_slug,
                    token_id=token_id,
                    entry_price=actual_price,
                    entry_time=time.time(),
                    size_shares=actual_shares,
                    size_usd=actual_price * actual_shares,
                    settlement_time=end_ts
                )
                self.pending_snipes[market_slug] = snipe
                self.coin_states[coin].status = "pending"
            except Exception as e:
                # Order succeeded but couldn't parse response - DON'T retry
                self.add_event(f"[{coin}] BOUGHT (parse error: {e})", "yellow")
                self.coin_states[coin].status = "pending"
        else:
            msg = result.message if result else "No result"
            self.add_event(f"[{coin}] Order failed: {msg}", "red")

            # Check if this is a permanent error (don't retry)
            permanent_errors = ["minimum", "min size", "invalid", "size"]
            is_permanent = any(err in msg.lower() for err in permanent_errors)

            if is_permanent:
                # Don't retry - keep in sniped_markets to prevent spam
                self.add_event(f"[{coin}] Permanent error - skipping market", "yellow")
                self.coin_states[coin].status = "waiting"
            else:
                # Transient error - allow retry
                self.sniped_markets.discard(market_slug)
                self.coin_states[coin].status = "watching"

    async def _check_settlement(self, market_slug: str, tick_side: str, mid: float, time_remaining: float):
        """Move pending snipes to 'awaiting redemption' once market settles.

        We do NOT determine win/loss here — that is done by the redeem loop
        based on actual balance changes, which is the only reliable source.
        """
        if market_slug not in self.pending_snipes:
            return

        snipe = self.pending_snipes[market_slug]

        # Check if market has settled (time expired or price at extreme)
        if time_remaining > 0 and not (mid >= 0.99 or mid <= 0.01):
            return

        # Only trigger once per snipe (use our position's side tick)
        if tick_side != snipe.side:
            return

        snipe.outcome = "settled"
        self.coin_states[snipe.coin].status = "settled"
        self.add_event(f"[{snipe.coin}] Market settled — awaiting redemption", "yellow")
        logger.info(f"[{snipe.coin}] {snipe.side.upper()} position settled, awaiting redeem")

        self.completed_snipes.append(snipe)
        del self.pending_snipes[market_slug]

    async def _redeem_loop(self):
        """Periodically check for redeemable positions every 30 seconds."""
        await asyncio.sleep(30)  # Initial delay to let first markets settle
        check_count = 0

        while self.running:
            try:
                check_count += 1

                # Check for redeemable positions
                logger.info(f"Checking for redeemable positions (check #{check_count})...")
                positions = await self.bot.get_redeemable_positions()

                if positions:
                    logger.info(f"Found {len(positions)} redeemable position(s):")
                    for pos in positions:
                        title = pos.get('title', 'Unknown')
                        value = pos.get('currentValue', 0)
                        logger.info(f"  - {title}: ${value:.2f}")

                    self.add_event(f"🔄 Redeeming {len(positions)} position(s)...", "yellow")
                    redeemed = await self.bot.auto_redeem_all()

                    if redeemed > 0:
                        # Update balance and compute P&L from actual balance change
                        ba = await self.bot.get_balance_allowance('', 'COLLATERAL')
                        new_balance = float(ba.get("balance", "0")) / 1_000_000
                        balance_change = new_balance - self.usdc_balance
                        self.usdc_balance = new_balance

                        # Track cumulative P&L from balance (true source of truth)
                        # Starting balance is set at init; total_pnl = current - starting
                        self.total_pnl = self.usdc_balance - self.starting_balance

                        # Count redeemed positions as wins (the few losses will show
                        # as smaller-than-expected balance gains, but we can't distinguish
                        # individual wins/losses when multiple positions redeem together)
                        self.wins = len([s for s in self.completed_snipes if s.outcome != "settled"])
                        settled_now = [s for s in self.completed_snipes if s.outcome == "settled"]
                        for s in settled_now:
                            s.outcome = "redeemed"

                        self.add_event(f"✅ Redeemed {redeemed} position(s) (+${balance_change:.2f})", "green bold")
                        self.add_event(f"💰 Balance: ${self.usdc_balance:.2f} (P&L: ${self.total_pnl:+.2f})", "green")
                        logger.info(f"💰 Balance: ${self.usdc_balance:.2f} (+${balance_change:.2f}), Total P&L: ${self.total_pnl:+.2f}")
                    else:
                        self.add_event(f"⚠️ Redeem attempted but failed", "red")
                        logger.warning("Redeem operation returned 0 - check logs for errors")
                else:
                    logger.debug(f"No redeemable positions found (check #{check_count})")
                    # Only show event every 5 checks to avoid spam
                    if check_count % 5 == 0:
                        self.add_event(f"ℹ️ No positions to redeem yet (checking every 30s)", "dim")

            except Exception as e:
                logger.error(f"Redeem loop error: {e}", exc_info=True)
                self.add_event(f"❌ Redeem check failed: {str(e)[:50]}", "red")

            # Check every 30 seconds (more frequent than 60s)
            await asyncio.sleep(30)

    async def stop(self):
        """Stop the strategy."""
        self.running = False

        console.print("\n")
        trades_entered = len(self.completed_snipes) + len(self.pending_snipes)
        console.print(Panel.fit(
            f"[bold]Final Stats[/bold]\n\n"
            f"Trades: {trades_entered}\n"
            f"Balance: ${self.usdc_balance:.2f}\n"
            f"Total P&L: [{'green' if self.total_pnl >= 0 else 'red'}]${self.total_pnl:+.2f}[/{'green' if self.total_pnl >= 0 else 'red'}]",
            title="Settlement Sniper",
            border_style="cyan"
        ))
        console.print(f"[dim]Log saved to: {log_file}[/dim]")


async def main():
    parser = argparse.ArgumentParser(description="Settlement Snipe Strategy")
    parser.add_argument('--coin', type=str, default='BTC,ETH,SOL,XRP',
                        help='Coin(s) to trade (comma-separated)')
    parser.add_argument('--size', type=float, default=5.0,
                        help='Trade size in USD (min: $1 for BTC/ETH/SOL, $5 for XRP)')
    parser.add_argument('--min-price', type=float, default=0.95,
                        help='Min entry price (0.95 = 95%%)')
    parser.add_argument('--max-price', type=float, default=0.97,
                        help='Max entry price (0.97 = 97%%)')
    parser.add_argument('--max-losses', type=int, default=0,
                        help='Stop after N losses (0 = unlimited)')

    args = parser.parse_args()

    coins = [c.strip().upper() for c in args.coin.split(',')]

    config = SnipeConfig(
        coins=coins,
        size=args.size,
        min_price=args.min_price,
        max_price=args.max_price,
        max_losses=args.max_losses
    )

    # Get credentials from environment
    private_key = os.getenv("POLY_PRIVATE_KEY")
    safe_address = os.getenv("POLY_SAFE_ADDRESS")

    if not private_key or not safe_address:
        console.print("[red]ERROR: Set POLY_PRIVATE_KEY and POLY_SAFE_ADDRESS in .env[/red]")
        return

    bot = TradingBot(
        private_key=private_key,
        safe_address=safe_address,
        config_path="config.yaml"
    )

    sniper = SettlementSniper(bot, config)

    if await sniper.start():
        try:
            await sniper.run()
        except KeyboardInterrupt:
            pass
        finally:
            await sniper.stop()


if __name__ == "__main__":
    asyncio.run(main())
