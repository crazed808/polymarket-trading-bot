"""
Longshot Strategy - Betting on 99% Reversals

This strategy monitors 15-minute Up/Down markets for extreme probabilities
(99%+) with 2+ minutes remaining and bets on the opposite side (the ~1% longshot).

Based on analysis:
- ~2% of the time, a 99%+ side with 2+ min remaining reverses
- Entry at ~1-2% means potential 50-100x returns on wins
- Expected ROI: ~31% per trade

Strategy Logic:
1. Monitor orderbook for both UP and DOWN sides
2. When either side hits 99%+ with 2+ minutes remaining:
   - Buy the OTHER side (the 1% longshot)
3. Exit conditions:
   - Take profit: Position reaches target (default 20%)
   - Market end: Hold until resolution (win big or lose small)

Usage:
    from strategies.longshot import LongshotStrategy, LongshotConfig

    strategy = LongshotStrategy(bot, config)
    await strategy.run()
"""

from dataclasses import dataclass
from typing import Dict, Optional

from lib.console import Colors, format_countdown
from strategies.base import BaseStrategy, StrategyConfig
from src.bot import TradingBot
from src.websocket_client import OrderbookSnapshot


@dataclass
class LongshotConfig(StrategyConfig):
    """Longshot strategy configuration."""

    # Entry conditions
    extreme_threshold: float = 0.99  # Trigger when opposite side >= this
    max_entry_price: float = 0.02  # Only buy if longshot side <= this (2%)
    min_time_remaining: int = 120  # Minimum seconds remaining (2 min)

    # Take profit at 20% (buying at 1-2%, selling at 20%+ = 10-20x return)
    take_profit: float = 0.18  # Exit when position price rises by this much

    # No stop loss by default (we're already at 1-2%)
    stop_loss: float = 0.02  # Small stop loss to prevent total loss

    # Smaller position size for high-risk trades
    size: float = 1.0  # $1 per trade

    # Allow multiple positions (one per side possible)
    max_positions: int = 2


class LongshotStrategy(BaseStrategy):
    """
    Longshot Reversal Strategy.

    Monitors for extreme probabilities and bets on reversals.
    High risk, high reward - most trades lose, but winners are big.
    """

    def __init__(self, bot: TradingBot, config: LongshotConfig):
        """Initialize longshot strategy."""
        super().__init__(bot, config)
        self.longshot_config = config

        # Track opportunities to avoid duplicate entries
        self._last_entry_side: Optional[str] = None
        self._last_entry_time: float = 0

        # Stats
        self.opportunities_seen = 0
        self.entries_made = 0

    def _get_time_remaining_seconds(self) -> int:
        """Get seconds remaining in current market."""
        market = self.current_market
        if not market:
            return 0

        mins, secs = market.get_countdown()
        if mins < 0:
            return 0
        return mins * 60 + secs

    def _check_extreme_condition(self, prices: Dict[str, float]) -> Optional[str]:
        """
        Check if extreme condition is met.

        Returns:
            Side to BUY (the longshot) if conditions met, None otherwise
        """
        time_remaining = self._get_time_remaining_seconds()

        # Need minimum time remaining
        if time_remaining < self.longshot_config.min_time_remaining:
            return None

        up_price = prices.get("up", 0.5)
        down_price = prices.get("down", 0.5)

        # If UP is at 99%+, consider buying DOWN (the longshot)
        if up_price >= self.longshot_config.extreme_threshold:
            # Only buy if DOWN is cheap enough (<= max_entry_price)
            if down_price <= self.longshot_config.max_entry_price:
                return "down"

        # If DOWN is at 99%+, consider buying UP (the longshot)
        if down_price >= self.longshot_config.extreme_threshold:
            # Only buy if UP is cheap enough (<= max_entry_price)
            if up_price <= self.longshot_config.max_entry_price:
                return "up"

        return None

    async def on_book_update(self, snapshot: OrderbookSnapshot) -> None:
        """Handle orderbook update."""
        pass  # Price recording done in base class

    async def on_tick(self, prices: Dict[str, float]) -> None:
        """Check for longshot entry opportunities."""
        # Check if we can open a position
        if not self.positions.can_open_position:
            return

        # Check for extreme condition
        buy_side = self._check_extreme_condition(prices)
        if not buy_side:
            return

        # Don't re-enter same side too quickly
        if buy_side == self._last_entry_side:
            import time
            if time.time() - self._last_entry_time < 60:  # 1 minute cooldown
                return

        # Already have position on this side?
        if self.positions.has_position(buy_side):
            return

        self.opportunities_seen += 1

        # Get the longshot price (should be ~1-2%)
        longshot_price = prices.get(buy_side, 0)
        opposite_price = prices.get("up" if buy_side == "down" else "down", 0)

        # Log the opportunity
        time_remaining = self._get_time_remaining_seconds()
        self.log(
            f"LONGSHOT OPPORTUNITY: {buy_side.upper()} @ {longshot_price:.4f} "
            f"(opposite @ {opposite_price:.4f}, {time_remaining}s remaining)",
            "trade"
        )

        # Execute buy
        success = await self.execute_longshot_buy(buy_side, longshot_price)
        if success:
            import time
            self._last_entry_side = buy_side
            self._last_entry_time = time.time()
            self.entries_made += 1

    async def execute_longshot_buy(self, side: str, current_price: float) -> bool:
        """
        Execute longshot buy order.

        Overrides base class to use more aggressive pricing for fills.
        """
        token_id = self.token_ids.get(side)
        if not token_id:
            self.log(f"No token ID for {side}", "error")
            return False

        # Use aggressive buy price to ensure fill (willing to pay up to 5%)
        buy_price = min(current_price + 0.01, 0.05)

        # Calculate size based on USDC amount at buy_price (not current_price)
        size = self.config.size / buy_price

        self.log(
            f"BUY LONGSHOT {side.upper()} @ {current_price:.4f} "
            f"(limit {buy_price:.4f}) size={size:.2f}",
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
            self.positions.open_position(
                side=side,
                token_id=token_id,
                entry_price=current_price,
                size=size,
                order_id=result.order_id,
            )
            return True
        else:
            self.log(f"Order failed: {result.message}", "error")
            return False

    def render_status(self, prices: Dict[str, float]) -> None:
        """Render TUI status display."""
        lines = []

        # Header
        ws_status = f"{Colors.GREEN}WS{Colors.RESET}" if self.is_connected else f"{Colors.RED}REST{Colors.RESET}"
        countdown = self._get_countdown_str()
        time_remaining = self._get_time_remaining_seconds()
        stats = self.positions.get_stats()

        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
        lines.append(
            f"{Colors.MAGENTA}LONGSHOT{Colors.RESET} [{Colors.CYAN}{self.config.coin}{Colors.RESET}] [{ws_status}] "
            f"Ends: {countdown} | Trades: {stats['trades_closed']} | PnL: ${stats['total_pnl']:+.2f}"
        )
        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")

        # Strategy status
        up_price = prices.get("up", 0)
        down_price = prices.get("down", 0)

        # Highlight extreme prices
        up_color = Colors.RED if up_price >= self.longshot_config.extreme_threshold else Colors.GREEN
        down_color = Colors.RED if down_price >= self.longshot_config.extreme_threshold else Colors.RED

        # Check if conditions are met
        ready_to_trade = time_remaining >= self.longshot_config.min_time_remaining
        up_extreme = up_price >= self.longshot_config.extreme_threshold
        down_extreme = down_price >= self.longshot_config.extreme_threshold

        status_line = f"UP: {up_color}{up_price:.4f}{Colors.RESET}  |  DOWN: {down_color}{down_price:.4f}{Colors.RESET}"

        # Check entry conditions
        max_entry = self.longshot_config.max_entry_price

        if ready_to_trade and up_extreme and down_price <= max_entry:
            status_line += f"  |  {Colors.YELLOW}>>> BUY DOWN @ {down_price:.4f} <<<{Colors.RESET}"
        elif ready_to_trade and down_extreme and up_price <= max_entry:
            status_line += f"  |  {Colors.YELLOW}>>> BUY UP @ {up_price:.4f} <<<{Colors.RESET}"
        elif ready_to_trade and up_extreme and down_price > max_entry:
            status_line += f"  |  {Colors.DIM}(DOWN too expensive: {down_price:.4f} > {max_entry:.2f}){Colors.RESET}"
        elif ready_to_trade and down_extreme and up_price > max_entry:
            status_line += f"  |  {Colors.DIM}(UP too expensive: {up_price:.4f} > {max_entry:.2f}){Colors.RESET}"
        elif not ready_to_trade:
            status_line += f"  |  {Colors.DIM}(waiting for 2min+ remaining){Colors.RESET}"
        else:
            status_line += f"  |  {Colors.DIM}(waiting for 99%+){Colors.RESET}"

        lines.append(status_line)
        lines.append(f"Threshold: {self.longshot_config.extreme_threshold:.2f} | Max entry: {max_entry:.2f} | Min time: {self.longshot_config.min_time_remaining}s")
        lines.append("-" * 80)

        # Orderbook display
        up_ob = self.market.get_orderbook("up")
        down_ob = self.market.get_orderbook("down")

        lines.append(f"{Colors.GREEN}{'UP':^39}{Colors.RESET}|{Colors.RED}{'DOWN':^39}{Colors.RESET}")
        lines.append(f"{'Bid':>9} {'Size':>9} | {'Ask':>9} {'Size':>9}|{'Bid':>9} {'Size':>9} | {'Ask':>9} {'Size':>9}")
        lines.append("-" * 80)

        # Get 5 levels
        up_bids = up_ob.bids[:5] if up_ob else []
        up_asks = up_ob.asks[:5] if up_ob else []
        down_bids = down_ob.bids[:5] if down_ob else []
        down_asks = down_ob.asks[:5] if down_ob else []

        for i in range(5):
            up_bid = f"{up_bids[i].price:>9.4f} {up_bids[i].size:>9.1f}" if i < len(up_bids) else f"{'--':>9} {'--':>9}"
            up_ask = f"{up_asks[i].price:>9.4f} {up_asks[i].size:>9.1f}" if i < len(up_asks) else f"{'--':>9} {'--':>9}"
            down_bid = f"{down_bids[i].price:>9.4f} {down_bids[i].size:>9.1f}" if i < len(down_bids) else f"{'--':>9} {'--':>9}"
            down_ask = f"{down_asks[i].price:>9.4f} {down_asks[i].size:>9.1f}" if i < len(down_asks) else f"{'--':>9} {'--':>9}"
            lines.append(f"{up_bid} | {up_ask}|{down_bid} | {down_ask}")

        lines.append("-" * 80)

        # Stats
        lines.append(
            f"Opportunities: {self.opportunities_seen} | "
            f"Entries: {self.entries_made} | "
            f"Win rate: {stats['win_rate']:.1f}%"
        )
        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")

        # Open positions
        lines.append(f"{Colors.BOLD}Positions:{Colors.RESET}")
        all_positions = self.positions.get_all_positions()
        if all_positions:
            for pos in all_positions:
                current = prices.get(pos.side, 0)
                pnl = pos.get_pnl(current)
                pnl_pct = pos.get_pnl_percent(current)
                hold_time = pos.get_hold_time()
                color = Colors.GREEN if pnl >= 0 else Colors.RED

                # Calculate potential return
                potential_return = (1.0 - pos.entry_price) * pos.size if current > 0 else 0

                lines.append(
                    f"  {Colors.BOLD}{pos.side.upper():4}{Colors.RESET} "
                    f"Entry: {pos.entry_price:.4f} | Current: {current:.4f} | "
                    f"Size: ${pos.size * pos.entry_price:.2f} | "
                    f"PnL: {color}${pnl:+.2f} ({pnl_pct:+.1f}%){Colors.RESET} | "
                    f"Hold: {hold_time:.0f}s"
                )
                lines.append(
                    f"       TP: {pos.take_profit_price:.4f} | "
                    f"Max potential: ${potential_return:.2f}"
                )
        else:
            lines.append(f"  {Colors.CYAN}(no open positions - waiting for 99%+ opportunity){Colors.RESET}")

        # Recent logs
        if self._log_buffer.messages:
            lines.append("-" * 80)
            lines.append(f"{Colors.BOLD}Recent Events:{Colors.RESET}")
            for msg in self._log_buffer.get_messages():
                lines.append(f"  {msg}")

        # Render
        output = "\033[H\033[J" + "\n".join(lines)
        print(output, flush=True)

    def _get_countdown_str(self) -> str:
        """Get formatted countdown string."""
        market = self.current_market
        if not market:
            return "--:--"

        mins, secs = market.get_countdown()
        return format_countdown(mins, secs)

    def on_market_change(self, old_slug: str, new_slug: str) -> None:
        """Handle market change."""
        self.prices.clear()
        self._last_entry_side = None

        # Close any open positions (market ended)
        positions = self.positions.get_all_positions()
        if positions:
            self.log(f"Market ended - closing {len(positions)} positions")
            for pos in positions:
                # Market ended - determine final PnL
                # If we held a longshot and market ended, we either won big or lost
                self.positions.close_position(pos.id)
                self.log(f"Closed {pos.side} position (market ended)")
