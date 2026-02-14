"""
Strategy Base Class - Foundation for Trading Strategies

Provides:
- Base class for all trading strategies
- Common lifecycle methods (start, stop, run)
- Integration with lib components (MarketManager, PriceTracker, PositionManager)
- Logging and status display utilities

Usage:
    from strategies.base import BaseStrategy, StrategyConfig

    class MyStrategy(BaseStrategy):
        async def on_book_update(self, snapshot):
            # Handle orderbook updates
            pass

        async def on_tick(self, prices):
            # Called each strategy tick
            pass
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List

from lib.console import LogBuffer, log
from lib.market_manager import MarketManager, MarketInfo
from lib.price_tracker import PriceTracker
from lib.position_manager import PositionManager, Position
from src.bot import TradingBot
from src.gamma_client import GammaClient
from src.websocket_client import OrderbookSnapshot


@dataclass
class StrategyConfig:
    """Base strategy configuration."""

    coin: str = "ETH"
    size: float = 5.0  # USDC size per trade
    max_positions: int = 1
    take_profit: float = 0.10
    stop_loss: float = 0.05

    # Market settings
    market_check_interval: float = 30.0
    auto_switch_market: bool = True

    # Price tracking
    price_lookback_seconds: int = 10
    price_history_size: int = 100

    # Display settings
    update_interval: float = 0.1
    order_refresh_interval: float = 30.0  # Seconds between order refreshes

    # Auto-redeem settings
    auto_redeem: bool = True  # Automatically redeem resolved positions
    redeem_check_interval: float = 60.0  # Seconds between redeem checks


class BaseStrategy(ABC):
    """
    Base class for trading strategies.

    Provides common infrastructure:
    - MarketManager for WebSocket and market discovery
    - PriceTracker for price history
    - PositionManager for positions and TP/SL
    - Logging and status display
    """

    def __init__(self, bot: TradingBot, config: StrategyConfig):
        """
        Initialize base strategy.

        Args:
            bot: TradingBot instance for order execution
            config: Strategy configuration
        """
        self.bot = bot
        self.config = config

        # Core components
        self.market = MarketManager(
            coin=config.coin,
            market_check_interval=config.market_check_interval,
            auto_switch_market=config.auto_switch_market,
        )

        self.prices = PriceTracker(
            lookback_seconds=config.price_lookback_seconds,
            max_history=config.price_history_size,
        )

        self.positions = PositionManager(
            take_profit=config.take_profit,
            stop_loss=config.stop_loss,
            max_positions=config.max_positions,
        )

        # State
        self.running = False
        self._status_mode = False

        # Logging
        self._log_buffer = LogBuffer(max_size=5)

        # Open orders cache (refreshed in background)
        self._cached_orders: List[dict] = []
        self._last_order_refresh: float = 0
        self._order_refresh_task: Optional[asyncio.Task] = None

        # Auto-redeem tracking
        self._last_redeem_check: float = 0
        self._total_redeemed: float = 0.0

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.market.is_connected

    @property
    def current_market(self) -> Optional[MarketInfo]:
        """Get current market info."""
        return self.market.current_market

    @property
    def token_ids(self) -> Dict[str, str]:
        """Get current token IDs."""
        return self.market.token_ids

    @property
    def open_orders(self) -> List[dict]:
        """Get cached open orders."""
        return self._cached_orders

    def _refresh_orders_sync(self) -> List[dict]:
        """Refresh open orders synchronously (called via to_thread)."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.bot.get_open_orders())
            finally:
                loop.close()
        except Exception:
            return []

    async def _do_order_refresh(self) -> None:
        """Background task to refresh orders without blocking."""
        try:
            orders = await asyncio.to_thread(self._refresh_orders_sync)
            self._cached_orders = orders
        except Exception:
            pass
        finally:
            self._order_refresh_task = None

    def _maybe_refresh_orders(self) -> None:
        """Schedule order refresh if interval has passed (fire-and-forget)."""
        now = time.time()
        if now - self._last_order_refresh > self.config.order_refresh_interval:
            # Don't start new refresh if one is already running
            if self._order_refresh_task is not None and not self._order_refresh_task.done():
                return
            self._last_order_refresh = now
            # Fire and forget - doesn't block main loop
            self._order_refresh_task = asyncio.create_task(self._do_order_refresh())

    async def _maybe_redeem(self) -> None:
        """Check and redeem resolved positions if enabled."""
        if not self.config.auto_redeem:
            return

        now = time.time()
        if now - self._last_redeem_check < self.config.redeem_check_interval:
            return

        self._last_redeem_check = now

        try:
            positions = await self.bot.get_redeemable_positions()
            if positions:
                total_value = sum(p.get("currentValue", 0) for p in positions)
                self.log(f"Found {len(positions)} redeemable positions (${total_value:.2f})", "info")

                redeemed = await self.bot.auto_redeem_all()
                if redeemed > 0:
                    self._total_redeemed += total_value
                    self.log(f"Redeemed {redeemed} positions!", "success")
        except Exception as e:
            self.log(f"Redeem check failed: {e}", "error")

    def log(self, msg: str, level: str = "info") -> None:
        """
        Log a message.

        Args:
            msg: Message to log
            level: Log level (info, success, warning, error, trade)
        """
        if self._status_mode:
            self._log_buffer.add(msg, level)
        else:
            log(msg, level)

    async def start(self) -> bool:
        """
        Start the strategy.

        Returns:
            True if started successfully
        """
        self.running = True

        # Register callbacks on market manager
        @self.market.on_book_update
        async def handle_book(snapshot: OrderbookSnapshot):  # pyright: ignore[reportUnusedFunction]
            # Record price
            for side, token_id in self.token_ids.items():
                if token_id == snapshot.asset_id:
                    self.prices.record(side, snapshot.mid_price)
                    break

            # Delegate to subclass
            await self.on_book_update(snapshot)

        @self.market.on_market_change
        def handle_market_change(old_slug: str, new_slug: str):  # pyright: ignore[reportUnusedFunction]
            self.log(f"Market changed: {old_slug} -> {new_slug}", "warning")
            self.prices.clear()
            self.on_market_change(old_slug, new_slug)

        @self.market.on_connect
        def handle_connect():  # pyright: ignore[reportUnusedFunction]
            self.log("WebSocket connected", "success")
            self.on_connect()

        @self.market.on_disconnect
        def handle_disconnect():  # pyright: ignore[reportUnusedFunction]
            self.log("WebSocket disconnected", "warning")
            self.on_disconnect()

        # Start market manager
        if not await self.market.start():
            self.running = False
            return False

        # Wait for initial data
        if not await self.market.wait_for_data(timeout=5.0):
            self.log("Timeout waiting for market data", "warning")

        return True

    async def stop(self) -> None:
        """Stop the strategy."""
        self.running = False

        # Cancel order refresh task if running
        if self._order_refresh_task is not None:
            self._order_refresh_task.cancel()
            try:
                await self._order_refresh_task
            except asyncio.CancelledError:
                pass
            self._order_refresh_task = None

        await self.market.stop()

    async def run(self) -> None:
        """Main strategy loop."""
        try:
            if not await self.start():
                self.log("Failed to start strategy", "error")
                return

            self._status_mode = True

            while self.running:
                # Get current prices
                prices = self._get_current_prices()

                # Call tick handler
                await self.on_tick(prices)

                # Check position exits
                await self._check_exits(prices)

                # Refresh orders in background (fire-and-forget)
                self._maybe_refresh_orders()

                # Auto-redeem resolved positions
                await self._maybe_redeem()

                # Update display
                self.render_status(prices)

                await asyncio.sleep(self.config.update_interval)

        except KeyboardInterrupt:
            self.log("Strategy stopped by user")
        finally:
            await self.stop()
            self._print_summary()

    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices from market manager."""
        prices = {}
        for side in ["up", "down"]:
            price = self.market.get_mid_price(side)
            if price > 0:
                prices[side] = price
        return prices

    async def _check_exits(self, prices: Dict[str, float]) -> None:
        """Check and execute exits for all positions."""
        exits = self.positions.check_all_exits(prices)

        for position, exit_type, pnl in exits:
            if exit_type == "take_profit":
                self.log(
                    f"TAKE PROFIT: {position.side.upper()} PnL: +${pnl:.2f}",
                    "success"
                )
            elif exit_type == "stop_loss":
                self.log(
                    f"STOP LOSS: {position.side.upper()} PnL: ${pnl:.2f}",
                    "warning"
                )

            # Execute sell
            self.log(f"Executing sell for {position.side.upper()}...", "info")
            try:
                await self.execute_sell(position, prices.get(position.side, 0))
            except Exception as e:
                self.log(f"Sell failed with error: {e}", "error")

    async def execute_buy(self, side: str, current_price: float) -> bool:
        """
        Execute market buy order using FOK (Fill Or Kill).

        Args:
            side: "up" or "down"
            current_price: Current market price

        Returns:
            True if order filled successfully
        """
        token_id = self.token_ids.get(side)
        if not token_id:
            self.log(f"No token ID for {side}", "error")
            return False

        # Use market order with FOK for guaranteed fill or fail
        order_value = max(self.config.size, 1.00)  # Polymarket $1 minimum
        worst_price = min(current_price + 0.03, 0.99)  # Allow 3% slippage

        self.log(f"BUY {side.upper()} @ market (~{current_price:.4f}) ${order_value:.2f}", "trade")

        result = await self.bot.place_market_order(
            token_id=token_id,
            amount=order_value,  # USD amount
            side="BUY",
            worst_price=worst_price,
        )

        if result.success:
            # Calculate size and USDC spent from response
            # Note: takingAmount = shares received, makingAmount = USDC spent
            taking_amount = result.data.get("takingAmount", "0")
            making_amount = result.data.get("makingAmount", "0")

            try:
                size = float(taking_amount) if taking_amount and taking_amount != "0" else order_value / current_price
                usdc_spent = float(making_amount) if making_amount and making_amount != "0" else order_value
            except (ValueError, TypeError):
                size = order_value / current_price
                usdc_spent = order_value

            self.log(f"Order filled: {result.order_id} size={size:.2f} spent=${usdc_spent:.2f}", "success")
            self.positions.open_position(
                side=side,
                token_id=token_id,
                entry_price=current_price,
                size=size,
                order_id=result.order_id,
                usdc_spent=usdc_spent,
            )
            return True
        else:
            self.log(f"Order failed: {result.message}", "error")
            return False

    async def execute_sell(self, position: Position, current_price: float, retries: int = 5) -> bool:
        """
        Execute sell order to close position with aggressive retry logic.

        Strategy: Try full amount first, progressively reduce size and increase
        slippage tolerance on each retry. More retries than before to handle
        low liquidity situations.

        Args:
            position: Position to close
            current_price: Current price
            retries: Number of retry attempts (default 5 for better fill rate)

        Returns:
            True if order filled successfully
        """
        pnl = position.get_pnl(current_price)

        # Add sync delay to allow blockchain state to settle
        await asyncio.sleep(0.5)

        # Check actual token balance with retry for sync delay
        actual_balance = 0.0
        balance_retries = 3
        for balance_attempt in range(balance_retries):
            try:
                ba = await self.bot.get_balance_allowance(position.token_id, "CONDITIONAL")
                raw_balance = float(ba.get("balance", "0"))
                actual_balance = raw_balance / 1_000_000  # Convert from base units

                self.log(f"Balance check: raw={raw_balance:.0f} tokens={actual_balance:.4f} tracked={position.size:.4f}", "info")

                if actual_balance >= position.size * 0.5:  # Have at least 50% of expected tokens
                    break
                elif balance_attempt < balance_retries - 1:
                    # Balance too low, might be sync delay - wait and retry
                    self.log(f"Balance low ({actual_balance:.4f}), waiting for sync (attempt {balance_attempt + 1}/{balance_retries})...", "warning")
                    await asyncio.sleep(1.0)
            except Exception as e:
                self.log(f"Balance check failed: {e}", "warning")
                if balance_attempt < balance_retries - 1:
                    await asyncio.sleep(0.5)

        if actual_balance <= 0.001:  # Still essentially zero after retries
            # Check if position is very new (< 5 seconds) - likely sync delay
            hold_time = position.get_hold_time()
            if hold_time < 5.0:
                self.log(f"Position only {hold_time:.1f}s old and balance=0 - likely sync delay, skipping sell", "warning")
                return False  # Don't close position, try again next tick

            # Tokens are gone - check market settlement to determine actual PnL
            settlement_pnl = await self.get_settlement_pnl(position)
            self.log(f"No tokens to sell (balance={actual_balance}) - settlement PnL: ${settlement_pnl:+.2f}", "warning")
            self.positions.close_position(position.id, realized_pnl=settlement_pnl)
            return True

        # Use actual balance - NO 99% buffer on first attempt
        base_sell_size = min(actual_balance, position.size)
        self.log(f"Selling {base_sell_size:.4f} tokens (balance={actual_balance:.4f})", "info")

        # More aggressive progressive buffers for better fill rate in low liquidity
        buffer_factors = [1.0, 0.99, 0.97, 0.95, 0.90]  # 100%, 99%, 97%, 95%, 90%

        for attempt in range(retries):
            # More aggressive slippage progression: 5%, 10%, 15%, 20%, 25%
            slippage_pct = 0.05 + (attempt * 0.05)
            worst_price = max(current_price * (1 - slippage_pct), 0.01)

            # Apply buffer factor based on attempt number
            buffer = buffer_factors[min(attempt, len(buffer_factors) - 1)]
            sell_size = base_sell_size * buffer

            if attempt > 0:
                self.log(f"SELL retry {attempt + 1}/{retries}: {sell_size:.2f} shares ({buffer*100:.0f}%) @ worst {worst_price:.4f} (slippage {slippage_pct*100:.0f}%)", "warning")
                await asyncio.sleep(0.3)  # Shorter delay for faster retries
            else:
                self.log(f"SELL {position.side.upper()} {sell_size:.2f} shares (100%) @ worst {worst_price:.4f}", "trade")

            result = await self.bot.place_market_order(
                token_id=position.token_id,
                amount=sell_size,
                side="SELL",
                worst_price=worst_price,
            )

            if result.success:
                # Calculate REAL P&L from actual USDC received vs spent
                usdc_received = 0.0
                try:
                    # For SELL: takingAmount = USDC received
                    taking_amount = result.data.get("takingAmount", "0")
                    usdc_received = float(taking_amount) if taking_amount else 0
                except (ValueError, TypeError):
                    usdc_received = 0

                # Real P&L = USDC received - USDC spent
                if usdc_received > 0 and position.usdc_spent > 0:
                    real_pnl = usdc_received - position.usdc_spent
                    self.log(f"Sell filled: {result.order_id} | Spent: ${position.usdc_spent:.2f} Received: ${usdc_received:.2f} | Real PnL: ${real_pnl:+.2f}", "success")
                else:
                    # Fallback to estimated PnL if we don't have actual amounts
                    real_pnl = pnl
                    self.log(f"Sell filled: {result.order_id} | Est PnL: ${pnl:+.2f}", "success")

                self.positions.close_position(position.id, realized_pnl=real_pnl)
                return True
            else:
                self.log(f"Sell attempt {attempt + 1} failed: {result.message}", "error")

        # All retries failed - DO NOT remove from tracking, position still exists!
        self.log(f"SELL FAILED after {retries} attempts - position still open!", "error")
        return False

    def _check_settlement_sync(self, position: Position) -> float:
        """
        Synchronous version of settlement check for use in sync callbacks.

        Args:
            position: Position to check settlement for

        Returns:
            Actual PnL based on market settlement
        """
        if not position.market_slug:
            return -position.usdc_spent if position.usdc_spent > 0 else 0

        try:
            gamma = GammaClient()
            market = gamma.get_market_by_slug(position.market_slug)

            if not market:
                return -position.usdc_spent if position.usdc_spent > 0 else 0

            prices = gamma.parse_prices(market)
            our_side_price = prices.get(position.side, 0.5)

            if our_side_price >= 0.9:
                # We won
                return position.size - position.usdc_spent
            elif our_side_price <= 0.1:
                # We lost
                return -position.usdc_spent
            else:
                # Unclear - assume loss
                return -position.usdc_spent if position.usdc_spent > 0 else 0

        except Exception:
            return -position.usdc_spent if position.usdc_spent > 0 else 0

    async def get_settlement_pnl(self, position: Position) -> float:
        """
        Calculate accurate PnL by checking market settlement outcome.

        Uses Gamma API to check if the market settled in our favor.

        Args:
            position: Position to check settlement for

        Returns:
            Actual PnL: positive if we won, negative (usdc_spent) if we lost
        """
        if not position.market_slug:
            # No market slug, can't look up settlement - assume loss
            self.log(f"No market slug for position, assuming loss", "warning")
            return -position.usdc_spent if position.usdc_spent > 0 else 0

        try:
            gamma = GammaClient()
            market = gamma.get_market_by_slug(position.market_slug)

            if not market:
                self.log(f"Could not find market {position.market_slug}", "warning")
                return -position.usdc_spent if position.usdc_spent > 0 else 0

            # Get final prices to determine winner
            prices = gamma.parse_prices(market)
            our_side_price = prices.get(position.side, 0.5)

            # For binary markets: price > 0.5 = that outcome won
            # Use 0.95/0.05 for clear settlement, 0.5 for close calls
            if our_side_price >= 0.95:
                # Clear win! Settlement = $1 per share
                pnl = position.size - position.usdc_spent
                self.log(f"Settlement: {position.side.upper()} WON (price={our_side_price:.2f}), PnL: ${pnl:+.2f}", "success")
                return pnl
            elif our_side_price <= 0.05:
                # Clear loss - tokens worth $0
                pnl = -position.usdc_spent
                self.log(f"Settlement: {position.side.upper()} LOST (price={our_side_price:.2f}), PnL: ${pnl:+.2f}", "warning")
                return pnl
            elif our_side_price > 0.5:
                # Likely won - price favors our side
                pnl = position.size - position.usdc_spent
                self.log(f"Settlement: {position.side.upper()} likely WON (price={our_side_price:.2f}), PnL: ${pnl:+.2f}", "success")
                return pnl
            else:
                # Price <= 0.5 - our side lost
                pnl = -position.usdc_spent
                self.log(f"Settlement: {position.side.upper()} LOST (price={our_side_price:.2f}), PnL: ${pnl:+.2f}", "warning")
                return pnl

        except Exception as e:
            self.log(f"Failed to check settlement: {e}", "error")
            return -position.usdc_spent if position.usdc_spent > 0 else 0

    def _print_summary(self) -> None:
        """Print session summary."""
        self._status_mode = False
        print()
        stats = self.positions.get_stats()
        self.log("Session Summary:")
        self.log(f"  Trades: {stats['trades_closed']}")
        self.log(f"  Total PnL: ${stats['total_pnl']:+.2f}")
        self.log(f"  Win rate: {stats['win_rate']:.1f}%")

    # Abstract methods to implement in subclasses

    @abstractmethod
    async def on_book_update(self, snapshot: OrderbookSnapshot) -> None:
        """
        Handle orderbook update.

        Called when new orderbook data is received.

        Args:
            snapshot: OrderbookSnapshot from WebSocket
        """
        pass

    @abstractmethod
    async def on_tick(self, prices: Dict[str, float]) -> None:
        """
        Handle strategy tick.

        Called on each iteration of the main loop.

        Args:
            prices: Current prices {side: price}
        """
        pass

    @abstractmethod
    def render_status(self, prices: Dict[str, float]) -> None:
        """
        Render status display.

        Called on each tick to update the display.

        Args:
            prices: Current prices
        """
        pass

    # Optional hooks (override as needed)

    def on_market_change(self, old_slug: str, new_slug: str) -> None:
        """Called when market changes - force close any open positions."""
        # Close all positions before market switch (can't trade on old market)
        positions = self.positions.get_all_positions()
        if positions:
            self.log(f"Market changed! Force closing {len(positions)} position(s)", "warning")
            for pos in positions:
                try:
                    # Check market settlement to determine actual PnL
                    settlement_pnl = self._check_settlement_sync(pos)
                    self.positions.close_position(pos.id, realized_pnl=settlement_pnl)
                    result = "WON" if settlement_pnl > 0 else "LOST"
                    self.log(f"Market ended: {pos.side.upper()} {result}, PnL: ${settlement_pnl:+.2f}", "warning")
                except Exception as e:
                    self.log(f"Failed to close {pos.side}: {e}", "error")

    def on_connect(self) -> None:
        """Called when WebSocket connects."""
        pass

    def on_disconnect(self) -> None:
        """Called when WebSocket disconnects."""
        pass
