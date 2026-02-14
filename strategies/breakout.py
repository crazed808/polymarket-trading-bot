"""
Consolidation Breakout Strategy

Trades breakouts from tight price ranges. Based on backtesting:
- 63.6% win rate
- +0.53% per trade after spread costs
- Works best on BTC (65% win rate)

Strategy Logic:
1. Monitor for price consolidation (range < 2% over ~30 seconds)
2. When price breaks out of the range, enter in breakout direction
3. Take profit at +4%, stop loss at -3%

Usage:
    from strategies.breakout import BreakoutStrategy, BreakoutConfig

    config = BreakoutConfig(coin="BTC", size=2.0)
    strategy = BreakoutStrategy(bot, config)
    await strategy.run()
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from collections import deque

from lib.console import Colors, format_countdown
from strategies.base import BaseStrategy, StrategyConfig
from src.bot import TradingBot
from src.websocket_client import OrderbookSnapshot


@dataclass
class BreakoutConfig(StrategyConfig):
    """Breakout strategy configuration."""

    # Consolidation detection
    consolidation_window: int = 30  # Seconds to measure consolidation
    max_consolidation_range: float = 0.02  # Max range to be "consolidating" (2%)
    min_consolidation_range: float = 0.005  # Min range (not completely dead)

    # Breakout detection
    breakout_threshold: float = 0.005  # How far outside range = breakout (0.5%)

    # Entry conditions
    min_time_remaining: int = 180  # Need 3+ minutes remaining

    # Position management
    take_profit: float = 0.04  # +4% take profit
    stop_loss: float = 0.03  # -3% stop loss
    time_stop: int = 90  # Exit if no movement after 90 seconds

    # Trade sizing
    size: float = 2.0  # $2 per trade
    max_positions: int = 1  # One position at a time

    # Cooldown
    trade_cooldown: int = 30  # Seconds between trades


@dataclass
class PricePoint:
    """Single price observation."""
    timestamp: float
    price: float


@dataclass
class ConsolidationRange:
    """Detected consolidation range."""
    high: float
    low: float
    mid: float

    @property
    def range_size(self) -> float:
        return self.high - self.low


class BreakoutStrategy(BaseStrategy):
    """
    Consolidation Breakout Strategy.

    Monitors for tight price ranges and trades breakouts.
    """

    def __init__(self, bot: TradingBot, config: BreakoutConfig):
        """Initialize breakout strategy."""
        super().__init__(bot, config)
        self.breakout_config = config

        # Price history for consolidation detection
        self._price_history: Dict[str, deque] = {
            "up": deque(maxlen=500),
            "down": deque(maxlen=500)
        }

        # Current consolidation state
        self._consolidation: Dict[str, Optional[ConsolidationRange]] = {
            "up": None,
            "down": None
        }

        # Trading state
        self._last_trade_time: float = 0
        self._in_consolidation: Dict[str, bool] = {"up": False, "down": False}

        # Stats
        self.breakouts_detected = 0
        self.trades_entered = 0
        self.wins = 0
        self.losses = 0

    def _record_price(self, side: str, price: float) -> None:
        """Record a price observation."""
        self._price_history[side].append(PricePoint(
            timestamp=time.time(),
            price=price
        ))

    def _get_recent_prices(self, side: str, seconds: int) -> List[float]:
        """Get prices from the last N seconds."""
        cutoff = time.time() - seconds
        return [p.price for p in self._price_history[side] if p.timestamp >= cutoff]

    def _detect_consolidation(self, side: str) -> Optional[ConsolidationRange]:
        """Check if price is consolidating."""
        prices = self._get_recent_prices(side, self.breakout_config.consolidation_window)

        if len(prices) < 10:  # Need enough data
            return None

        high = max(prices)
        low = min(prices)
        range_size = high - low

        # Check if in consolidation range
        if (self.breakout_config.min_consolidation_range <= range_size <=
            self.breakout_config.max_consolidation_range):
            return ConsolidationRange(high=high, low=low, mid=(high + low) / 2)

        return None

    def _check_breakout(self, side: str, current_price: float) -> Optional[str]:
        """
        Check if price broke out of consolidation.

        Returns:
            "up" or "down" for breakout direction, None if no breakout
        """
        consolidation = self._consolidation[side]
        if not consolidation:
            return None

        threshold = self.breakout_config.breakout_threshold

        if current_price > consolidation.high + threshold:
            return "up"
        elif current_price < consolidation.low - threshold:
            return "down"

        return None

    def _get_time_remaining_seconds(self) -> int:
        """Get seconds remaining in current market."""
        market = self.current_market
        if not market:
            return 0

        mins, secs = market.get_countdown()
        if mins < 0:
            return 0
        return mins * 60 + secs

    async def on_book_update(self, snapshot: OrderbookSnapshot) -> None:
        """Handle orderbook update - record price."""
        for side, token_id in self.token_ids.items():
            if token_id == snapshot.asset_id:
                self._record_price(side, snapshot.mid_price)
                break

    async def on_tick(self, prices: Dict[str, float]) -> None:
        """Main strategy tick - check for breakouts."""
        time_remaining = self._get_time_remaining_seconds()

        # Check minimum time
        if time_remaining < self.breakout_config.min_time_remaining:
            return

        # Check cooldown
        if time.time() - self._last_trade_time < self.breakout_config.trade_cooldown:
            return

        # Can we open a position?
        if not self.positions.can_open_position:
            return

        # Check each side for breakout opportunities
        for side in ["up", "down"]:
            price = prices.get(side, 0.5)

            # Update consolidation detection
            new_consolidation = self._detect_consolidation(side)

            if new_consolidation:
                # We're in consolidation
                if not self._in_consolidation[side]:
                    self._in_consolidation[side] = True
                    self._consolidation[side] = new_consolidation
                    self.log(f"{side.upper()} consolidating: {new_consolidation.low:.4f}-{new_consolidation.high:.4f}", "info")
            else:
                # Check for breakout from previous consolidation
                if self._in_consolidation[side] and self._consolidation[side]:
                    breakout_dir = self._check_breakout(side, price)

                    if breakout_dir:
                        self.breakouts_detected += 1
                        self.log(
                            f"BREAKOUT {breakout_dir.upper()}! {side} @ {price:.4f} "
                            f"(range was {self._consolidation[side].low:.4f}-{self._consolidation[side].high:.4f})",
                            "trade"
                        )

                        # Execute trade in breakout direction
                        # If UP side breaks up -> buy UP
                        # If UP side breaks down -> buy DOWN (opposite)
                        if side == "up":
                            trade_side = "up" if breakout_dir == "up" else "down"
                        else:
                            trade_side = "down" if breakout_dir == "up" else "up"

                        trade_price = prices.get(trade_side, 0.5)
                        success = await self._execute_breakout_trade(trade_side, trade_price)

                        if success:
                            self.trades_entered += 1
                            self._last_trade_time = time.time()

                        # Reset consolidation state
                        self._in_consolidation[side] = False
                        self._consolidation[side] = None

                # Clear consolidation if no longer in range
                self._in_consolidation[side] = False

    async def _execute_breakout_trade(self, side: str, current_price: float) -> bool:
        """Execute breakout trade with TP/SL."""
        token_id = self.token_ids.get(side)
        if not token_id:
            self.log(f"No token ID for {side}", "error")
            return False

        # Calculate entry price (aggressive to ensure fill)
        buy_price = min(current_price + 0.01, 0.99)
        size = self.breakout_config.size / buy_price

        self.log(
            f"BUY {side.upper()} @ {current_price:.4f} (limit {buy_price:.4f}) "
            f"TP: {current_price + self.breakout_config.take_profit:.4f} "
            f"SL: {current_price - self.breakout_config.stop_loss:.4f}",
            "trade"
        )

        result = await self.bot.place_order(
            token_id=token_id,
            price=buy_price,
            size=size,
            side="BUY"
        )

        if result.success:
            self.log(f"Order placed: {result.order_id}", "success")

            # Open position with custom TP/SL
            self.positions.open_position(
                side=side,
                token_id=token_id,
                entry_price=current_price,
                size=size,
                order_id=result.order_id,
                take_profit=self.breakout_config.take_profit,
                stop_loss=self.breakout_config.stop_loss,
            )
            return True
        else:
            self.log(f"Order failed: {result.message}", "error")
            return False

    async def _check_exits(self, prices: Dict[str, float]) -> None:
        """Check exits including time stop."""
        # First check normal TP/SL
        await super()._check_exits(prices)

        # Then check time stop
        for position in self.positions.get_all_positions():
            hold_time = position.get_hold_time()

            if hold_time >= self.breakout_config.time_stop:
                current_price = prices.get(position.side, 0)
                pnl = position.get_pnl(current_price)

                self.log(
                    f"TIME STOP: {position.side.upper()} held {hold_time:.0f}s, PnL: ${pnl:+.2f}",
                    "warning"
                )

                await self.execute_sell(position, current_price)

                if pnl > 0:
                    self.wins += 1
                else:
                    self.losses += 1

    def render_status(self, prices: Dict[str, float]) -> None:
        """Render TUI status display."""
        lines = []

        # Header
        ws_status = f"{Colors.GREEN}WS{Colors.RESET}" if self.is_connected else f"{Colors.RED}REST{Colors.RESET}"
        market = self.current_market
        countdown = market.get_countdown_str() if market else "--:--"
        time_remaining = self._get_time_remaining_seconds()

        lines.append(f"{Colors.BOLD}{'='*85}{Colors.RESET}")
        lines.append(
            f"{Colors.CYAN}BREAKOUT{Colors.RESET} [{Colors.YELLOW}{self.config.coin}{Colors.RESET}] [{ws_status}] "
            f"Ends: {countdown} | "
            f"Breakouts: {self.breakouts_detected} | Trades: {self.trades_entered} | "
            f"W/L: {self.wins}/{self.losses}"
        )
        lines.append(f"{Colors.BOLD}{'='*85}{Colors.RESET}")

        # Current prices and consolidation status
        up_price = prices.get("up", 0.5)
        down_price = prices.get("down", 0.5)

        # Consolidation info
        up_cons = self._consolidation.get("up")
        down_cons = self._consolidation.get("down")

        up_status = f"Range: {up_cons.low:.3f}-{up_cons.high:.3f}" if up_cons else "Watching..."
        down_status = f"Range: {down_cons.low:.3f}-{down_cons.high:.3f}" if down_cons else "Watching..."

        lines.append(f"UP:   {Colors.GREEN}{up_price:.4f}{Colors.RESET}  |  {up_status}")
        lines.append(f"DOWN: {Colors.RED}{down_price:.4f}{Colors.RESET}  |  {down_status}")

        # Trading conditions
        can_trade = (
            time_remaining >= self.breakout_config.min_time_remaining and
            self.positions.can_open_position and
            time.time() - self._last_trade_time >= self.breakout_config.trade_cooldown
        )

        if can_trade:
            if self._in_consolidation["up"] or self._in_consolidation["down"]:
                lines.append(f"{Colors.YELLOW}>>> CONSOLIDATION DETECTED - WATCHING FOR BREAKOUT <<<{Colors.RESET}")
            else:
                lines.append(f"{Colors.DIM}Monitoring for consolidation...{Colors.RESET}")
        else:
            reasons = []
            if time_remaining < self.breakout_config.min_time_remaining:
                reasons.append(f"time<{self.breakout_config.min_time_remaining}s")
            if not self.positions.can_open_position:
                reasons.append("position open")
            if time.time() - self._last_trade_time < self.breakout_config.trade_cooldown:
                reasons.append("cooldown")
            lines.append(f"{Colors.DIM}Not trading: {', '.join(reasons)}{Colors.RESET}")

        lines.append("-" * 85)

        # Positions
        lines.append(f"{Colors.BOLD}Positions:{Colors.RESET}")
        all_positions = self.positions.get_all_positions()

        if all_positions:
            for pos in all_positions:
                current = prices.get(pos.side, 0)
                pnl = pos.get_pnl(current)
                pnl_pct = pos.get_pnl_percent(current)
                hold_time = pos.get_hold_time()
                color = Colors.GREEN if pnl >= 0 else Colors.RED

                # Progress to TP/SL (use position's actual delta values)
                if current > pos.entry_price:
                    progress = (current - pos.entry_price) / pos.take_profit_delta if pos.take_profit_delta > 0 else 0
                else:
                    progress = -(pos.entry_price - current) / pos.stop_loss_delta if pos.stop_loss_delta > 0 else 0

                bar_len = 10
                if progress >= 0:
                    filled = int(min(progress, 1.0) * bar_len)
                    bar = f"{Colors.GREEN}{'█' * filled}{'░' * (bar_len - filled)}{Colors.RESET}"
                else:
                    filled = int(min(abs(progress), 1.0) * bar_len)
                    bar = f"{Colors.RED}{'█' * filled}{'░' * (bar_len - filled)}{Colors.RESET}"

                lines.append(
                    f"  {Colors.BOLD}{pos.side.upper():4}{Colors.RESET} "
                    f"Entry: {pos.entry_price:.4f} | Current: {current:.4f} | "
                    f"PnL: {color}${pnl:+.2f} ({pnl_pct:+.1f}%){Colors.RESET} | "
                    f"[{bar}] | Hold: {hold_time:.0f}s/{self.breakout_config.time_stop}s"
                )
        else:
            lines.append(f"  {Colors.CYAN}(no positions){Colors.RESET}")

        # Recent logs
        if self._log_buffer.messages:
            lines.append("-" * 85)
            lines.append(f"{Colors.BOLD}Events:{Colors.RESET}")
            for msg in self._log_buffer.get_messages():
                lines.append(f"  {msg}")

        lines.append(f"{Colors.BOLD}{'='*85}{Colors.RESET}")

        # Render
        print("\033[H\033[J" + "\n".join(lines), flush=True)

    def on_market_change(self, old_slug: str, new_slug: str) -> None:
        """Handle market change - reset state."""
        super().on_market_change(old_slug, new_slug)

        # Clear price history and consolidation state
        for side in ["up", "down"]:
            self._price_history[side].clear()
            self._consolidation[side] = None
            self._in_consolidation[side] = False
