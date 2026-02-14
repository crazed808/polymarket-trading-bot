#!/usr/bin/env python3
"""
5-Minute Micro-Snipe Strategy

Buys at high probability near settlement in 5-minute BTC Up/Down markets.
Same logic as the 15-minute micro sniper, adapted for 5-minute windows.

BTC only (only coin with 5-minute markets).

Usage:
    python run_micro_snipe_5m.py --coin BTC --size 5.0
    python run_micro_snipe_5m.py --coin BTC --size 5.0 --max-price 0.985
"""

import asyncio
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Set, Optional, List, Deque
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

from lib.dashboard import ModernDashboard

from src.bot import TradingBot
from src.gamma_client import GammaClient
from src.websocket_client import MarketWebSocket, OrderbookSnapshot

# Logging setup - file only (dashboard handles console)
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"micro5m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger("micro5m")
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
    """Settlement snipe configuration for 5-minute markets."""
    coins: list = field(default_factory=lambda: ['BTC'])
    size: float = 5.0  # Fallback size when orderbook is empty; normally sizes to full available liquidity
    min_price: float = 0.95
    max_price: float = 0.995
    min_time: int = 1
    max_time: int = 3
    min_liquidity: float = 50.0  # Lower than 15m (5m markets have ~$1,100 total liquidity)
    max_spread: float = 0.05
    balance_fraction: float = 1.0  # Fraction of balance to use per trade (0.1 = 10%)

    # Per-coin minimum sizes (5m markets: orderMinSize=5)
    min_sizes: dict = field(default_factory=lambda: {
        'BTC': 5.0,
    })

    # Safety checks
    check_momentum: bool = False
    momentum_lookback: int = 10
    max_negative_momentum: float = -0.02
    min_orderbook_ratio: float = 1.5

    # Probability stability check: reject entries where probability just spiked
    # into the valid range (e.g. 83% -> 95% in 300ms = boundary trade, high reversal risk)
    stability_check: bool = True
    stability_lookback: float = 2.0   # Seconds to look back
    stability_min_price: float = 0.90  # Side must have been >= this for the entire lookback

    # Stop-loss: stop bot after N losses (0 = disabled)
    max_losses: int = 0


@dataclass
class SnipeResult:
    """Track a snipe trade."""
    coin: str
    side: str
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
class SnipeCandidate:
    """A candidate for sniping, used for prioritization."""
    coin: str
    side: str
    market_slug: str
    token_id: str
    mid: float
    best_ask: float
    time_remaining: float
    end_ts: float
    ask_levels: list
    profit_potential: float
    timestamp: float = field(default_factory=time.time)
    skip_until: float = 0.0


@dataclass
class CoinState:
    """Current state for a coin."""
    coin: str
    price_up: float = 0.0
    price_down: float = 0.0
    end_ts: float = 0.0
    status: str = "waiting"
    last_update: float = 0.0
    market_slug: str = ""

    @property
    def time_remaining(self) -> float:
        if self.end_ts <= 0:
            return 0.0
        return max(0.0, self.end_ts - time.time())


class SettlementSniper5m:
    """5-Minute Settlement Snipe Strategy with Dashboard."""

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

        self.current_markets: Dict[str, dict] = {}
        self.market_end_times: Dict[str, float] = {}
        self.coin_states: Dict[str, CoinState] = {}

        self.events: List[str] = []
        self.max_events = 8

        self.candidates: Dict[str, SnipeCandidate] = {}
        self.candidate_lock = asyncio.Lock()

        self.price_history: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=30))
        self.momentum_skips = 0

        # Pre-signed orders: token_id -> signed order object
        self.pre_signed: Dict[str, Any] = {}
        self.pre_signed_markets: Set[str] = set()  # market slugs already pre-signed

        for coin in config.coins:
            self.coin_states[coin] = CoinState(coin=coin)

        self.dashboard = ModernDashboard(
            coins=config.coins,
            min_price=config.min_price,
            max_price=config.max_price,
            min_time=config.min_time,
            max_time=config.max_time,
        )
        self.starting_balance = 0.0

    def add_event(self, msg: str, style: str = "white"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.events.append(f"[{style}][{timestamp}] {msg}[/{style}]")
        if len(self.events) > self.max_events:
            self.events.pop(0)
        logger.info(msg)
        self.dashboard.add_event(msg, style)

    def build_dashboard(self) -> Layout:
        for coin in self.config.coins:
            state = self.coin_states.get(coin, CoinState(coin=coin))
            self.dashboard.update_coin_state(
                coin=coin,
                price_up=state.price_up,
                price_down=state.price_down,
                time_remaining=state.time_remaining,
                status=state.status,
            )

        trades_count = len(self.completed_snipes) + len(self.pending_snipes)
        self.dashboard.update_stats(
            balance=self.usdc_balance,
            total_pnl=self.total_pnl,
            trades_count=trades_count,
            wins=self.wins,
            losses=self.losses,
            pending_count=len(self.pending_snipes),
            filtered_count=self.momentum_skips,
        )

        return self.dashboard.build(log_file=log_file.name)

    async def start(self) -> bool:
        self.add_event("Initializing 5m strategy...", "cyan")

        if self.bot.config.use_gasless:
            self.add_event(f"Gasless redemptions enabled", "green")
            logger.info(f"Builder credentials configured: {self.bot.config.builder.is_configured()}")
        else:
            self.add_event(f"Gasless disabled - redemptions may fail!", "yellow")
            logger.warning("Builder credentials not configured - redemptions will fail")

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
        self.add_event("5m Micro Sniper started", "green")
        return True

    async def run(self):
        tasks = []

        for coin in self.config.coins:
            task = asyncio.create_task(self._run_coin(coin))
            tasks.append(task)

        tasks.append(asyncio.create_task(self._redeem_loop()))
        tasks.append(asyncio.create_task(self._candidate_processor()))
        tasks.append(asyncio.create_task(self._keepalive_loop()))

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
        """Run sniper for a single coin on 5-minute markets."""
        self.add_event(f"[{coin}] Starting 5m monitor", "cyan")

        while self.running:
            try:
                # Get current 5-minute market
                market = self.gamma.get_current_5m_market(coin)
                if not market:
                    self.coin_states[coin].status = "waiting"
                    logger.debug(f"[{coin}] No active 5m market found, retrying in 5s")
                    await asyncio.sleep(5)
                    continue

                market_slug = market.get('slug', '')
                tokens = self.gamma.parse_token_ids(market)
                end_date = market.get('endDate', '')

                if end_date:
                    end_ts = datetime.fromisoformat(end_date.replace('Z', '+00:00')).timestamp()
                else:
                    end_ts = time.time() + 300  # 5 minutes fallback

                self.current_markets[coin] = market
                self.market_end_times[coin] = end_ts
                self.coin_states[coin].market_slug = market_slug

                up_token = tokens.get('up', '')
                down_token = tokens.get('down', '')

                if not up_token:
                    await asyncio.sleep(5)
                    continue

                # Pre-warm SDK caches so order signing doesn't hit the network
                await self.bot.pre_warm_token(up_token)
                await self.bot.pre_warm_token(down_token)

                self.coin_states[coin].status = "watching"

                ws = MarketWebSocket()

                @ws.on_book
                async def handle_book(snapshot: OrderbookSnapshot):
                    if not self.running:
                        return

                    if snapshot.asset_id == up_token:
                        side = "up"
                        token = up_token
                    elif snapshot.asset_id == down_token:
                        side = "down"
                        token = down_token
                    else:
                        return

                    if side == "up":
                        self.coin_states[coin].price_up = snapshot.mid_price
                    else:
                        self.coin_states[coin].price_down = snapshot.mid_price

                    self.coin_states[coin].end_ts = end_ts
                    self.coin_states[coin].last_update = time.time()

                    time_remaining = self.coin_states[coin].time_remaining

                    bid_depth = sum(level.size for level in snapshot.bids)
                    ask_depth = sum(level.size for level in snapshot.asks)
                    spread = snapshot.best_ask - snapshot.best_bid if snapshot.bids and snapshot.asks else 0.0

                    best_ask = snapshot.best_ask if snapshot.asks else snapshot.mid_price

                    await self._process_tick(
                        coin=coin,
                        market_slug=market_slug,
                        side=side,
                        token_id=token,
                        mid=snapshot.mid_price,
                        best_ask=best_ask,
                        bid_depth=bid_depth,
                        ask_depth=ask_depth,
                        spread=spread,
                        time_remaining=time_remaining,
                        end_ts=end_ts,
                        ask_levels=snapshot.asks
                    )

                await ws.subscribe([up_token, down_token])

                try:
                    await asyncio.wait_for(
                        ws.run(),
                        timeout=max(end_ts - time.time() + 30, 60)
                    )
                except asyncio.TimeoutError:
                    pass

                await ws.disconnect()

                # Clear pre-signed orders for expired tokens
                self.pre_signed.pop(up_token, None)
                self.pre_signed.pop(down_token, None)

                self.coin_states[coin].status = "waiting"
                self.coin_states[coin].price_up = 0
                self.coin_states[coin].price_down = 0
                self.coin_states[coin].end_ts = 0

            except Exception as e:
                self.add_event(f"[{coin}] Error: {e}", "red")
                await asyncio.sleep(5)

    async def _process_tick(self, coin: str, market_slug: str, side: str, token_id: str,
                           mid: float, best_ask: float, bid_depth: float, ask_depth: float,
                           spread: float, time_remaining: float, end_ts: float,
                           ask_levels: list = None):
        """Process a market tick."""
        current_time = time.time()
        history_key = f"{market_slug}:{side}"
        self.price_history[history_key].append((current_time, mid))

        await self._check_settlement(market_slug, side, mid, time_remaining)

        if market_slug in self.sniped_markets:
            self.coin_states[coin].status = "pending"
            return

        in_time = self.config.min_time <= time_remaining <= self.config.max_time

        # Back off if a 15-minute market is near settlement (last 10s).
        # The 15m micro sniper uses full balance, so we must not compete.
        if in_time:
            now = datetime.now(timezone.utc)
            seconds_into_15m = (now.minute % 15) * 60 + now.second
            seconds_left_15m = 900 - seconds_into_15m
            if seconds_left_15m <= 10:
                logger.info(f"[{coin}] DEFER: 15m market settling in {seconds_left_15m}s — backing off")
                return

        # Refresh balance at ~3.5s
        balance_key = f"{market_slug}:balance_refreshed"
        if 3.2 <= time_remaining <= 3.8 and balance_key not in self.sniped_markets:
            self.sniped_markets.add(balance_key)
            try:
                ba = await self.bot.get_balance_allowance('', 'COLLATERAL')
                self.usdc_balance = float(ba.get("balance", "0")) / 1_000_000
                logger.info(f"[{coin}] Balance refreshed: ${self.usdc_balance:.2f}")
            except Exception as e:
                logger.warning(f"[{coin}] Balance refresh failed: {e}")

        # Pre-sign orders at 5-7s remaining (removes ~100ms from critical path)
        if 5.0 <= time_remaining <= 7.0 and market_slug not in self.pre_signed_markets:
            self.pre_signed_markets.add(market_slug)
            try:
                market_info = self.current_markets.get(coin, {})
                tokens = self.gamma.parse_token_ids(market_info)
                up_token = tokens.get('up', '')
                down_token = tokens.get('down', '')
                fraction = self.config.balance_fraction
                max_p = self.config.max_price
                shares = int(self.usdc_balance * fraction / max_p)
                if shares > 0 and up_token and down_token:
                    signed_up, signed_down = await asyncio.gather(
                        self.bot.create_signed_order(
                            token_id=up_token, side="BUY", price=max_p, size=shares,
                        ),
                        self.bot.create_signed_order(
                            token_id=down_token, side="BUY", price=max_p, size=shares,
                        ),
                    )
                    self.pre_signed[up_token] = signed_up
                    self.pre_signed[down_token] = signed_down
                    logger.info(f"[{coin}] [PRE-SIGN] Both sides signed: {shares} shares @ {max_p} "
                               f"(balance=${self.usdc_balance:.2f}, fraction={fraction})")
                else:
                    logger.warning(f"[{coin}] [PRE-SIGN] Skipped: shares={shares}, up={bool(up_token)}, down={bool(down_token)}")
            except Exception as e:
                logger.warning(f"[{coin}] [PRE-SIGN] Failed: {e}")

        if time_remaining <= 5:
            logger.debug(f"[{coin}] {side} tr={time_remaining:.1f}s mid={mid:.3f} ask={best_ask:.3f} in_time={in_time}")

        if not in_time:
            return

        in_price = self.config.min_price <= mid <= self.config.max_price

        if best_ask > self.config.max_price:
            in_price = False
        if best_ask < self.config.min_price:
            in_price = False
            logger.debug(f"[{coin}] {side} REJECT: best_ask {best_ask:.3f} below min_price {self.config.min_price}")

        if in_time and in_price:
            self.coin_states[coin].status = "sniping"
        elif time_remaining <= self.config.max_time:
            self.coin_states[coin].status = "watching"

        if not in_price:
            logger.debug(f"[{coin}] {side} REJECT price: mid={mid:.3f} ask={best_ask:.3f} range=[{self.config.min_price}-{self.config.max_price}]")
            return

        # Manipulation check
        opposite_side = "down" if side == "up" else "up"
        opposite_key = f"{market_slug}:{opposite_side}"
        opposite_history = list(self.price_history.get(opposite_key, []))

        if opposite_history:
            lookback = 5.0
            cutoff = current_time - lookback
            recent_opposite = [(t, p) for t, p in opposite_history if t >= cutoff]

            if recent_opposite:
                max_opposite = max(p for _, p in recent_opposite)
                if max_opposite >= 0.80:
                    logger.warning(f"[{coin}] {side} REJECT MANIPULATION: "
                                 f"opposite side ({opposite_side}) was at {max_opposite:.1%} within last {lookback}s")
                    self.add_event(f"[{coin}] SKIP: Price spike trap (opposite was {max_opposite:.0%})", "red")
                    return

        # Probability stability check: reject if this side's probability just spiked
        # into the valid range. A rapid spike (e.g. 83% -> 95% in 300ms) means the
        # underlying price is right at the boundary and can easily flip back.
        if self.config.stability_check:
            history_key_self = f"{market_slug}:{side}"
            side_history = list(self.price_history.get(history_key_self, []))
            if side_history:
                lookback = self.config.stability_lookback
                cutoff = current_time - lookback
                recent_prices = [(t, p) for t, p in side_history if t >= cutoff]

                if recent_prices:
                    min_recent = min(p for _, p in recent_prices)
                    if min_recent < self.config.stability_min_price:
                        logger.warning(f"[{coin}] {side} REJECT STABILITY: "
                                     f"min price in last {lookback}s was {min_recent:.1%} "
                                     f"(need >= {self.config.stability_min_price:.0%}), "
                                     f"current={mid:.1%}")
                        self.momentum_skips += 1
                        self.add_event(f"[{coin}] SKIP: Probability spike ({min_recent:.0%} -> {mid:.0%} in {lookback}s)", "yellow")
                        return

        logger.info(f"[{coin}] {side} CANDIDATE: mid={mid:.3f} ask={best_ask:.3f} tr={time_remaining:.1f}s spread={spread:.4f} bids={bid_depth:.0f} asks={ask_depth:.0f}")

        if (bid_depth + ask_depth) < self.config.min_liquidity:
            logger.debug(f"[{coin}] REJECT liquidity: {bid_depth+ask_depth:.0f} < {self.config.min_liquidity}")
            return

        if spread > self.config.max_spread:
            logger.debug(f"[{coin}] REJECT spread: {spread:.4f} > {self.config.max_spread}")
            return

        if self.config.check_momentum:
            is_safe, reason = self._check_momentum_safety(
                history_key, side, mid, bid_depth, ask_depth
            )
            if not is_safe:
                self.momentum_skips += 1
                self.add_event(f"[{coin}] SKIP: {reason}", "yellow")
                return

        if not self.config.check_momentum and bid_depth and ask_depth:
            ratio = bid_depth / ask_depth if ask_depth > 0 else 999
            if ratio < self.config.min_orderbook_ratio:
                self.momentum_skips += 1
                self.add_event(f"[{coin}] SKIP: Weak pressure ({ratio:.2f})", "yellow")
                return

        # Fast-execute bypass: skip candidate queue for high-confidence trades
        # with pre-signed orders (saves 10-25ms queue polling delay)
        if mid >= 0.98 and token_id in self.pre_signed:
            logger.info(f"[{coin}] [FAST-EXEC] High-confidence bypass: mid={mid:.3f}, pre-signed available")
            await self._fast_execute(
                coin=coin, market_slug=market_slug, side=side, token_id=token_id,
                mid=mid, best_ask=best_ask, time_remaining=time_remaining, end_ts=end_ts,
                ask_levels=ask_levels
            )
            return

        profit_potential = 1.0 - mid
        candidate = SnipeCandidate(
            coin=coin,
            side=side,
            market_slug=market_slug,
            token_id=token_id,
            mid=mid,
            best_ask=best_ask,
            time_remaining=time_remaining,
            end_ts=end_ts,
            ask_levels=ask_levels,
            profit_potential=profit_potential
        )

        async with self.candidate_lock:
            if market_slug not in self.sniped_markets:
                existing = self.candidates.get(market_slug)
                if not existing or candidate.profit_potential >= existing.profit_potential:
                    self.candidates[market_slug] = candidate
                    logger.debug(f"[{coin}] Queued candidate: profit_potential={profit_potential:.2%}")

    def _check_momentum_safety(self, history_key: str, side: str, current_price: float,
                               bid_depth: float, ask_depth: float) -> tuple:
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

        if bid_depth and ask_depth:
            ratio = bid_depth / ask_depth if ask_depth > 0 else 999
            logger.debug(f"Orderbook: ratio={ratio:.2f} (bid={bid_depth:.0f}, ask={ask_depth:.0f}, need={self.config.min_orderbook_ratio})")
            if ratio < self.config.min_orderbook_ratio:
                return False, f"Weak pressure (ratio: {ratio:.2f})"

        return True, ""

    async def _candidate_processor(self):
        """Process queued candidates in priority order."""
        while self.running:
            try:
                candidates_to_execute = []

                async with self.candidate_lock:
                    if self.candidates:
                        current_time = time.time()
                        ready_candidates = []
                        cooling_candidates = []

                        for candidate in self.candidates.values():
                            if candidate.market_slug in self.sniped_markets:
                                continue
                            current_tr = candidate.end_ts - current_time
                            is_thin_margin = candidate.profit_potential < 0.02
                            if is_thin_margin:
                                if current_tr < self.config.min_time:
                                    continue
                            else:
                                if current_time - candidate.timestamp > 0.5:
                                    continue

                            if candidate.skip_until <= current_time:
                                ready_candidates.append(candidate)
                            else:
                                cooling_candidates.append(candidate)

                        ready_candidates.sort(key=lambda c: c.profit_potential, reverse=True)

                        if ready_candidates:
                            candidates_to_execute = ready_candidates
                        elif cooling_candidates:
                            cooling_candidates.sort(key=lambda c: (c.skip_until, -c.profit_potential))
                            candidates_to_execute = [cooling_candidates[0]]
                            logger.debug(f"[{cooling_candidates[0].coin}] Retrying cooled candidate")

                        stale = []
                        for k, v in self.candidates.items():
                            if k in self.sniped_markets:
                                stale.append(k)
                            elif v.profit_potential < 0.02:
                                if v.end_ts - time.time() < self.config.min_time:
                                    stale.append(k)
                            else:
                                if time.time() - v.timestamp > 0.5:
                                    stale.append(k)
                        for k in stale:
                            self.candidates.pop(k, None)

                for candidate in candidates_to_execute:
                    if candidate.market_slug in self.sniped_markets:
                        continue

                    # 5m markets: shorter thin-margin delay than 15m (2.0s vs 1.5s)
                    # to balance accuracy vs being beaten by competitors
                    current_time_remaining = candidate.end_ts - time.time()
                    if candidate.profit_potential < 0.02 and current_time_remaining > 2.0:
                        logger.debug(f"[{candidate.coin}] Delaying thin-margin candidate "
                                    f"({candidate.profit_potential:.2%}) - waiting for better price")
                        continue

                    current_state = self.coin_states.get(candidate.coin)
                    if current_state:
                        current_price = current_state.price_up if candidate.side == "up" else current_state.price_down
                        if current_price > 0:
                            price_drift = abs(current_price - candidate.mid)
                            if price_drift > 0.05:
                                logger.warning(f"[{candidate.coin}] STALE candidate rejected: "
                                             f"candidate mid={candidate.mid:.3f} but current={current_price:.3f}")
                                async with self.candidate_lock:
                                    self.candidates.pop(candidate.market_slug, None)
                                continue
                            if not (self.config.min_price <= current_price <= self.config.max_price):
                                logger.warning(f"[{candidate.coin}] Price out of range: current={current_price:.3f}")
                                async with self.candidate_lock:
                                    self.candidates.pop(candidate.market_slug, None)
                                continue

                    logger.info(f"[{candidate.coin}] Executing priority candidate: "
                               f"profit_potential={candidate.profit_potential:.2%}, mid={candidate.mid:.3f}")

                    await self._execute_snipe(
                        candidate.coin, candidate.market_slug, candidate.side, candidate.token_id,
                        candidate.mid, candidate.best_ask, candidate.time_remaining, candidate.end_ts,
                        candidate.ask_levels
                    )
                    break

            except Exception as e:
                logger.error(f"Candidate processor error: {e}", exc_info=True)

            await asyncio.sleep(0.01)  # 10ms loop — faster reaction to candidates

    async def _fast_execute(self, coin: str, market_slug: str, side: str, token_id: str,
                           mid: float, best_ask: float, time_remaining: float, end_ts: float,
                           ask_levels: list = None):
        """Fast-path execution for high-confidence trades with pre-signed orders.

        Bypasses candidate queue entirely. Self-contained to avoid function-call
        overhead on the hot path.
        """
        if market_slug in self.sniped_markets:
            return

        # Validate orderbook before committing the pre-signed order.
        # The pre-signed order is sized to full balance at max_price — if the
        # orderbook has cheap stale asks below min_price, FAK would sweep them.
        # Require that most available liquidity is within the valid price range.
        if ask_levels:
            valid_usd = sum(l.size * l.price for l in ask_levels
                           if self.config.min_price <= l.price <= self.config.max_price)
            invalid_usd = sum(l.size * l.price for l in ask_levels
                             if l.price < self.config.min_price)
            if invalid_usd > valid_usd * 0.1:
                logger.warning(f"[{coin}] [FAST-EXEC] ABORT: ${invalid_usd:.2f} in asks below "
                             f"{self.config.min_price} (valid=${valid_usd:.2f}) — risk of deep sweep")
                return
            if valid_usd < self.config.min_sizes.get(coin, 5.0):
                logger.warning(f"[{coin}] [FAST-EXEC] ABORT: only ${valid_usd:.2f} valid liquidity")
                return

        # Pop pre-signed order
        signed_order = self.pre_signed.pop(token_id, None)
        if not signed_order:
            logger.warning(f"[{coin}] [FAST-EXEC] Pre-signed order disappeared, falling back to queue")
            return

        # Margin check
        exec_price = min(best_ask, self.config.max_price) if best_ask > 0 else mid
        profit_margin = 1.0 - exec_price
        if profit_margin < 0.005:
            self.add_event(f"[{coin}] SKIP: Margin too thin ({profit_margin:.2%})", "yellow")
            return

        # Size check
        min_size = self.config.min_sizes.get(coin, 5.0)
        max_usd = self.usdc_balance * self.config.balance_fraction
        if max_usd < min_size or self.usdc_balance < min_size:
            self.add_event(f"[{coin}] Insufficient balance (need ${min_size:.2f})", "red")
            return

        size_usd = round(min(max_usd, self.usdc_balance), 2)

        display_prob = mid if side == "up" else (1 - mid)
        self.add_event(f"[{coin}] [FAST] 5m MICRO {side.upper()} @ {display_prob:.1%}, ${size_usd:.2f}, {time_remaining:.0f}s left", "yellow")

        # Mark sniped and pre-deduct balance
        self.sniped_markets.add(market_slug)
        self.usdc_balance = max(0, self.usdc_balance - size_usd)
        self.coin_states[coin].status = "sniping"

        logger.info(f"[{coin}] [FAST-EXEC] Posting pre-signed order @ limit {self.config.max_price}")

        result = None
        try:
            result = await self.bot.post_signed_order(signed_order, order_type="FAK")
            if result:
                logger.info(f"[{coin}] [FAST-EXEC] Order response: {result.data}")
        except Exception as e:
            self.add_event(f"[{coin}] Order error: {e}", "red")
            self.sniped_markets.discard(market_slug)
            self.usdc_balance += size_usd
            self.coin_states[coin].status = "watching"
            return

        order_placed = result and result.success

        # FAK no-match: API returns 400 "no orders found to match" — treat as no-fill
        if not order_placed and result and "no orders found to match" in (result.message or "").lower():
            logger.info(f"[{coin}] [FAST-EXEC] FAK no match — no liquidity available")
            self.add_event(f"[{coin}] FAK not filled — no matching orders", "yellow")
            self.usdc_balance += size_usd
            self.sniped_markets.discard(market_slug)
            self.coin_states[coin].status = "watching"
            return

        if order_placed:
            try:
                status = result.data.get("status", "")
                taking = result.data.get("takingAmount", "")
                making = result.data.get("makingAmount", "")

                if status == "live" or (not taking and not making):
                    logger.info(f"[{coin}] [FAST-EXEC] IOC not filled (status={status})")
                    self.add_event(f"[{coin}] IOC not filled — no liquidity", "yellow")
                    self.usdc_balance += size_usd
                    self.coin_states[coin].status = "waiting"
                    return

                if taking and making:
                    actual_shares = float(taking)
                    actual_usd = float(making)
                    actual_price = actual_usd / actual_shares if actual_shares > 0 else mid

                    if actual_price < self.config.min_price:
                        logger.error(f"[{coin}] [FAST-EXEC] BAD FILL: {actual_shares:.2f} shares at {actual_price:.2%}")
                        self.add_event(f"[{coin}] WARNING: Filled at {actual_price:.0%} (below {self.config.min_price:.0%})", "red")
                else:
                    actual_shares = float(result.data.get("size", 0))
                    actual_price = mid
                    actual_usd = actual_shares * actual_price

                self.add_event(f"[{coin}] BOUGHT {side.upper()} @ {actual_price:.2%} (${actual_usd:.2f})", "green")

                balance_diff = size_usd - actual_usd
                if abs(balance_diff) > 0.01:
                    self.usdc_balance += balance_diff

                snipe = SnipeResult(
                    coin=coin, side=side, market_slug=market_slug, token_id=token_id,
                    entry_price=actual_price, entry_time=time.time(),
                    size_shares=actual_shares, size_usd=actual_usd, settlement_time=end_ts,
                )
                self.pending_snipes[market_slug] = snipe
                self.coin_states[coin].status = "pending"
            except Exception as e:
                self.add_event(f"[{coin}] BOUGHT (parse error: {e})", "yellow")
                self.coin_states[coin].status = "pending"
        else:
            msg = result.message if result else "No result"
            self.add_event(f"[{coin}] Order failed: {msg}", "red")
            permanent_errors = ["minimum", "min size", "invalid", "size"]
            is_permanent = any(err in msg.lower() for err in permanent_errors)
            if is_permanent:
                self.coin_states[coin].status = "waiting"
            else:
                self.sniped_markets.discard(market_slug)
                self.coin_states[coin].status = "watching"

    async def _execute_snipe(self, coin: str, market_slug: str, side: str, token_id: str,
                            mid: float, best_ask: float, time_remaining: float, end_ts: float,
                            ask_levels: list = None):
        """Execute a micro-snipe trade, taking all available liquidity up to balance."""
        available_shares = 0.0
        available_usd = 0.0
        worst_price = 0.0
        filtered_out_usd = 0.0
        if ask_levels:
            if len(ask_levels) > 0:
                all_prices = [l.price for l in ask_levels]
                total_raw = sum(l.size * l.price for l in ask_levels)
                logger.info(f"[{coin}] Orderbook: {len(ask_levels)} ask levels, "
                           f"prices: {[f'{p:.4f}' for p in all_prices[:5]]}, "
                           f"total_raw=${total_raw:.2f}, max_price_filter={self.config.max_price}")
            filtered_below_min = 0.0
            for level in ask_levels:
                if level.price >= self.config.min_price and level.price <= self.config.max_price:
                    available_shares += level.size
                    available_usd += level.size * level.price
                    worst_price = max(worst_price, level.price)
                elif level.price > self.config.max_price:
                    filtered_out_usd += level.size * level.price
                else:
                    filtered_below_min += level.size * level.price
            if filtered_out_usd > 0:
                logger.info(f"[{coin}] Filtered out ${filtered_out_usd:.2f} above {self.config.max_price} price cap")
            if filtered_below_min > 0:
                logger.info(f"[{coin}] Filtered out ${filtered_below_min:.2f} below {self.config.min_price} price floor")
        else:
            logger.info(f"[{coin}] ask_levels is empty or None (no asks in orderbook)")

        # Cap execution price at best_ask to prevent deep sweeps (BAD FILL protection)
        exec_price = min(best_ask, self.config.max_price) if best_ask > 0 else worst_price

        profit_margin = 1.0 - exec_price
        if profit_margin < 0.005:
            async with self.candidate_lock:
                if market_slug in self.candidates:
                    self.candidates[market_slug].skip_until = time.time() + 0.2
            self.add_event(f"[{coin}] SKIP: Margin too thin ({profit_margin:.2%})", "yellow")
            return

        min_size = self.config.min_sizes.get(coin, 5.0)
        max_usd = self.usdc_balance * self.config.balance_fraction
        if available_usd > 0:
            size_usd = min(available_usd, max_usd)
        else:
            async with self.candidate_lock:
                if market_slug in self.candidates:
                    self.candidates[market_slug].skip_until = time.time() + 0.2
            logger.warning(f"[{coin}] ABORT: No liquidity in price range [{self.config.min_price}-{self.config.max_price}]")
            self.add_event(f"[{coin}] SKIP: No liquidity at valid prices", "yellow")
            return

        size_usd = max(size_usd, min_size)

        if self.usdc_balance < min_size:
            self.add_event(f"[{coin}] Insufficient balance (need ${min_size:.2f})", "red")
            return

        size_usd = min(size_usd, max_usd)

        display_prob = mid if side == "up" else (1 - mid)
        self.add_event(f"[{coin}] 5m MICRO {side.upper()} @ {display_prob:.1%}, ${size_usd:.2f}, {time_remaining:.0f}s left", "yellow")
        self.coin_states[coin].status = "sniping"

        # FAK (Fill-And-Kill) orders require maker_amount with max 2 decimals.
        # The SDK computes maker = price * shares.  Using integer shares guarantees
        # price(2dp) * shares(0dp) = product(≤2dp).
        size_usd = round(size_usd, 2)
        shares = int(size_usd / exec_price)

        is_fallback = available_usd == 0
        logger.info(f"[{coin}] Sizing: available_liq=${available_usd:.2f} ({available_shares:.1f} shares), "
                     f"using=${size_usd:.2f}, limit_price={exec_price:.4f} (best_ask={best_ask:.4f}), balance=${self.usdc_balance:.2f}"
                     f"{' [FALLBACK - no tradeable asks]' if is_fallback else ''}")

        self.sniped_markets.add(market_slug)

        self.usdc_balance = max(0, self.usdc_balance - size_usd)
        logger.debug(f"[{coin}] Pre-deducted ${size_usd:.2f}, remaining balance: ${self.usdc_balance:.2f}")

        order_placed = False
        result = None
        try:
            buy_price = exec_price

            signed_order = await self.bot.create_signed_order(
                token_id=token_id,
                side="BUY",
                price=buy_price,
                size=shares,
            )

            result = await self.bot.post_signed_order(signed_order, order_type="FAK")
            order_placed = result and result.success
            if result:
                logger.info(f"[{coin}] Order response: {result.data}")
        except Exception as e:
            self.add_event(f"[{coin}] Order error: {e}", "red")
            self.sniped_markets.discard(market_slug)
            self.usdc_balance += size_usd
            logger.debug(f"[{coin}] Restored ${size_usd:.2f} after error, balance: ${self.usdc_balance:.2f}")
            self.coin_states[coin].status = "watching"
            return

        # FAK no-match: API returns 400 "no orders found to match" — treat as no-fill
        if not order_placed and result and "no orders found to match" in (result.message or "").lower():
            logger.info(f"[{coin}] FAK no match — no liquidity available")
            self.add_event(f"[{coin}] FAK not filled — no matching orders", "yellow")
            self.usdc_balance += size_usd
            self.sniped_markets.discard(market_slug)
            self.coin_states[coin].status = "watching"
            return

        if order_placed:
            try:
                status = result.data.get("status", "")
                taking = result.data.get("takingAmount", "")
                making = result.data.get("makingAmount", "")

                if status == "live" or (not taking and not making):
                    logger.info(f"[{coin}] IOC not filled (status={status}, taking={taking}, making={making})")
                    self.add_event(f"[{coin}] IOC not filled — no liquidity", "yellow")
                    self.usdc_balance += size_usd
                    logger.debug(f"[{coin}] Restored ${size_usd:.2f} after no fill, balance: ${self.usdc_balance:.2f}")
                    self.coin_states[coin].status = "waiting"
                    return

                if taking and making:
                    actual_shares = float(taking)
                    actual_usd = float(making)
                    actual_price = actual_usd / actual_shares if actual_shares > 0 else mid

                    if actual_price < self.config.min_price:
                        logger.error(f"[{coin}] BAD FILL: Got {actual_shares:.2f} shares at {actual_price:.2%} "
                                   f"(min={self.config.min_price:.0%}) - paid ${actual_usd:.2f}")
                        self.add_event(f"[{coin}] WARNING: Filled at {actual_price:.0%} (below {self.config.min_price:.0%})", "red")
                else:
                    actual_shares = float(result.data.get("size", shares))
                    actual_price = mid
                    actual_usd = actual_shares * actual_price

                self.add_event(f"[{coin}] BOUGHT {side.upper()} @ {actual_price:.2%} (${actual_usd:.2f})", "green")

                balance_diff = size_usd - actual_usd
                if abs(balance_diff) > 0.01:
                    self.usdc_balance += balance_diff
                    logger.debug(f"[{coin}] Adjusted balance by ${balance_diff:.2f}, now: ${self.usdc_balance:.2f}")

                snipe = SnipeResult(
                    coin=coin,
                    side=side,
                    market_slug=market_slug,
                    token_id=token_id,
                    entry_price=actual_price,
                    entry_time=time.time(),
                    size_shares=actual_shares,
                    size_usd=actual_usd,
                    settlement_time=end_ts
                )
                self.pending_snipes[market_slug] = snipe
                self.coin_states[coin].status = "pending"
            except Exception as e:
                self.add_event(f"[{coin}] BOUGHT (parse error: {e})", "yellow")
                self.coin_states[coin].status = "pending"
        else:
            msg = result.message if result else "No result"
            self.add_event(f"[{coin}] Order failed: {msg}", "red")

            permanent_errors = ["minimum", "min size", "invalid", "size"]
            is_permanent = any(err in msg.lower() for err in permanent_errors)

            if is_permanent:
                self.add_event(f"[{coin}] Permanent error - skipping market", "yellow")
                self.coin_states[coin].status = "waiting"
            else:
                self.sniped_markets.discard(market_slug)
                self.coin_states[coin].status = "watching"

    async def _check_settlement(self, market_slug: str, tick_side: str, mid: float, time_remaining: float):
        if market_slug not in self.pending_snipes:
            return

        snipe = self.pending_snipes[market_slug]

        if time_remaining > 0 and not (mid >= 0.99 or mid <= 0.01):
            return

        if tick_side != snipe.side:
            return

        snipe.outcome = "settled"
        self.coin_states[snipe.coin].status = "settled"
        self.add_event(f"[{snipe.coin}] Market settled — awaiting redemption", "yellow")
        logger.info(f"[{snipe.coin}] {snipe.side.upper()} position settled, awaiting redeem")

        self.completed_snipes.append(snipe)
        del self.pending_snipes[market_slug]

    async def _keepalive_loop(self):
        """Keep HTTP connection warm so the critical POST doesn't hit a cold connection."""
        while self.running:
            try:
                await asyncio.sleep(45)
                await self.bot.get_balance_allowance('', 'COLLATERAL')
            except Exception:
                pass

    async def _redeem_loop(self):
        """Periodically check for redeemable positions every 30 seconds."""
        await asyncio.sleep(30)
        check_count = 0

        while self.running:
            try:
                check_count += 1

                logger.info(f"Checking for redeemable positions (check #{check_count})...")
                positions = await self.bot.get_redeemable_positions()

                if positions:
                    logger.info(f"Found {len(positions)} redeemable position(s):")
                    for pos in positions:
                        title = pos.get('title', 'Unknown')
                        value = pos.get('currentValue', 0)
                        logger.info(f"  - {title}: ${value:.2f}")

                    self.add_event(f"Redeeming {len(positions)} position(s)...", "yellow")
                    redeemed = await self.bot.auto_redeem_all()

                    if redeemed > 0:
                        ba = await self.bot.get_balance_allowance('', 'COLLATERAL')
                        new_balance = float(ba.get("balance", "0")) / 1_000_000
                        balance_change = new_balance - self.usdc_balance
                        self.usdc_balance = new_balance
                        self.total_pnl = self.usdc_balance - self.starting_balance

                        for s in self.completed_snipes:
                            if s.outcome == "settled":
                                s.outcome = "redeemed"

                        self.add_event(f"Redeemed {redeemed} position(s) (+${balance_change:.2f})", "green bold")
                        self.add_event(f"Balance: ${self.usdc_balance:.2f} (P&L: ${self.total_pnl:+.2f})", "green")
                        logger.info(f"Balance: ${self.usdc_balance:.2f} (+${balance_change:.2f}), Total P&L: ${self.total_pnl:+.2f}")
                    else:
                        self.add_event(f"Redeem attempted but failed", "red")
                        logger.warning("Redeem operation returned 0 - check logs for errors")
                else:
                    logger.debug(f"No redeemable positions found (check #{check_count})")
                    if check_count % 5 == 0:
                        self.add_event(f"No positions to redeem yet (checking every 30s)", "dim")

                # Periodic balance refresh every 5 minutes
                if check_count % 10 == 0:
                    try:
                        ba = await self.bot.get_balance_allowance('', 'COLLATERAL')
                        new_balance = float(ba.get("balance", "0")) / 1_000_000
                        if abs(new_balance - self.usdc_balance) > 0.01:
                            logger.info(f"Balance sync: ${self.usdc_balance:.2f} -> ${new_balance:.2f}")
                            self.usdc_balance = new_balance
                            self.total_pnl = self.usdc_balance - self.starting_balance
                    except Exception as e:
                        logger.debug(f"Balance refresh failed: {e}")

            except Exception as e:
                logger.error(f"Redeem loop error: {e}", exc_info=True)
                self.add_event(f"Redeem check failed: {str(e)[:50]}", "red")

            await asyncio.sleep(30)

    async def stop(self):
        self.running = False

        console.print("\n")
        console.print(Panel.fit(
            f"[bold]Final Stats[/bold]\n\n"
            f"Trades: {len(self.completed_snipes) + len(self.pending_snipes)}\n"
            f"Balance: ${self.usdc_balance:.2f}\n"
            f"Total P&L: [{'green' if self.total_pnl >= 0 else 'red'}]${self.total_pnl:+.2f}[/{'green' if self.total_pnl >= 0 else 'red'}]",
            title="5m Micro Sniper",
            border_style="cyan"
        ))
        console.print(f"[dim]Log saved to: {log_file}[/dim]")


async def main():
    parser = argparse.ArgumentParser(description="5-Minute Micro-Snipe Strategy")
    parser.add_argument('--coin', type=str, default='BTC',
                        help='Coin(s) to trade (BTC only for now)')
    parser.add_argument('--size', type=float, default=5.0,
                        help='Trade size in USD (min $5 for 5m markets)')
    parser.add_argument('--min-price', type=float, default=0.95,
                        help='Min entry price (0.95 = 95%%)')
    parser.add_argument('--max-price', type=float, default=0.995,
                        help='Max entry price (0.995 = 99.5%%)')
    parser.add_argument('--balance-fraction', type=float, default=1.0,
                        help='Fraction of balance to use per trade (0.1 = 10%%)')
    parser.add_argument('--max-losses', type=int, default=0,
                        help='Stop after N losses (0 = unlimited)')
    parser.add_argument('--no-stability-check', action='store_true',
                        help='Disable probability stability check')
    parser.add_argument('--stability-lookback', type=float, default=2.0,
                        help='Stability check lookback in seconds (default: 2.0)')
    parser.add_argument('--stability-min', type=float, default=0.90,
                        help='Min probability over lookback period (default: 0.90 = 90%%)')

    args = parser.parse_args()

    coins = [c.strip().upper() for c in args.coin.split(',')]

    # Validate only BTC is used
    for coin in coins:
        if coin not in GammaClient.COIN_SLUGS_5M:
            console.print(f"[red]ERROR: {coin} not available for 5-minute markets. Only BTC is supported.[/red]")
            return

    config = SnipeConfig(
        coins=coins,
        size=args.size,
        min_price=args.min_price,
        max_price=args.max_price,
        balance_fraction=args.balance_fraction,
        max_losses=args.max_losses,
        stability_check=not args.no_stability_check,
        stability_lookback=args.stability_lookback,
        stability_min_price=args.stability_min,
    )

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

    sniper = SettlementSniper5m(bot, config)

    if await sniper.start():
        try:
            await sniper.run()
        except KeyboardInterrupt:
            pass
        finally:
            await sniper.stop()


if __name__ == "__main__":
    asyncio.run(main())
