#!/usr/bin/env python3
"""
Runner for Longshot Strategy

Runs the longshot reversal strategy on one or all coins.
Bets on 99%+ reversals with 2+ minutes remaining.

Usage:
    # Single coin
    python run_longshot.py --coin BTC

    # All coins (runs 4 strategies in parallel with unified dashboard)
    python run_longshot.py --coin ALL

    # Custom settings
    python run_longshot.py --coin ETH --size 2.0 --threshold 0.99 --min-time 120
"""
import asyncio
import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

from lib.console import Colors, LogBuffer, format_countdown
from strategies.longshot import LongshotStrategy, LongshotConfig
from src.bot import TradingBot


class MultiCoinDashboard:
    """Unified dashboard for displaying all coins at once."""

    COINS = ["BTC", "ETH", "SOL", "XRP"]

    def __init__(self, config: LongshotConfig):
        self.config = config
        self.strategies: Dict[str, LongshotStrategy] = {}
        self.log_buffer = LogBuffer(max_size=8)
        self.running = False

        # Stats
        self.total_opportunities = 0
        self.total_entries = 0
        self.total_wins = 0
        self.total_losses = 0
        self.total_pnl = 0.0

    def add_strategy(self, coin: str, strategy: LongshotStrategy):
        """Register a strategy for a coin."""
        self.strategies[coin] = strategy

    def log(self, msg: str, level: str = "info"):
        """Add a log message."""
        self.log_buffer.add(msg, level)

    def _get_coin_status(self, coin: str) -> Dict:
        """Get status info for a coin."""
        strategy = self.strategies.get(coin)
        if not strategy:
            return {
                "up": 0.5, "down": 0.5, "time": "--:--",
                "status": "NOT STARTED", "connected": False,
                "positions": [], "time_secs": 0
            }

        prices = strategy._get_current_prices()
        up_price = prices.get("up", 0.5)
        down_price = prices.get("down", 0.5)

        # Get time remaining
        time_secs = strategy._get_time_remaining_seconds()
        mins, secs = divmod(time_secs, 60)
        time_str = format_countdown(mins, secs)

        # Determine status
        ready = time_secs >= self.config.min_time_remaining
        up_extreme = up_price >= self.config.extreme_threshold
        down_extreme = down_price >= self.config.extreme_threshold
        max_entry = self.config.max_entry_price

        if ready and up_extreme and down_price <= max_entry:
            status = f"{Colors.YELLOW}>>> BUY DOWN @ {down_price:.4f} <<<{Colors.RESET}"
        elif ready and down_extreme and up_price <= max_entry:
            status = f"{Colors.YELLOW}>>> BUY UP @ {up_price:.4f} <<<{Colors.RESET}"
        elif ready and up_extreme and down_price > max_entry:
            status = f"{Colors.DIM}DOWN too expensive ({down_price:.4f}){Colors.RESET}"
        elif ready and down_extreme and up_price > max_entry:
            status = f"{Colors.DIM}UP too expensive ({up_price:.4f}){Colors.RESET}"
        elif not ready and time_secs > 0:
            status = f"{Colors.DIM}Waiting for 2min+ remaining{Colors.RESET}"
        elif up_extreme or down_extreme:
            status = f"{Colors.DIM}Extreme but <2min left{Colors.RESET}"
        else:
            status = f"{Colors.DIM}Monitoring...{Colors.RESET}"

        # Get positions
        positions = strategy.positions.get_all_positions()

        return {
            "up": up_price,
            "down": down_price,
            "time": time_str,
            "time_secs": time_secs,
            "status": status,
            "connected": strategy.is_connected,
            "positions": positions,
            "opportunities": strategy.opportunities_seen,
            "entries": strategy.entries_made,
        }

    def render(self):
        """Render the unified dashboard."""
        lines = []
        width = 100

        # Header
        lines.append(f"{Colors.BOLD}{'=' * width}{Colors.RESET}")
        lines.append(
            f"{Colors.MAGENTA}{Colors.BOLD}LONGSHOT STRATEGY{Colors.RESET} - ALL COINS"
            f"  |  Threshold: {self.config.extreme_threshold:.0%}"
            f"  |  Max Entry: {self.config.max_entry_price:.0%}"
            f"  |  Min Time: {self.config.min_time_remaining}s"
        )
        lines.append(f"{Colors.BOLD}{'=' * width}{Colors.RESET}")

        # Coin table header
        lines.append(
            f"{Colors.BOLD}{'COIN':<6}{'':3}{'UP':>8}{'':3}{'DOWN':>8}{'':3}"
            f"{'TIME':>7}{'':3}{'STATUS':<35}{'':2}{'WS':<4}{Colors.RESET}"
        )
        lines.append("-" * width)

        # Coin rows
        all_positions = []
        total_opps = 0
        total_entries = 0

        for coin in self.COINS:
            data = self._get_coin_status(coin)

            # Color prices based on extremity
            up_price = data["up"]
            down_price = data["down"]

            if up_price >= self.config.extreme_threshold:
                up_str = f"{Colors.RED}{up_price:>8.4f}{Colors.RESET}"
            elif up_price <= self.config.max_entry_price:
                up_str = f"{Colors.GREEN}{up_price:>8.4f}{Colors.RESET}"
            else:
                up_str = f"{up_price:>8.4f}"

            if down_price >= self.config.extreme_threshold:
                down_str = f"{Colors.RED}{down_price:>8.4f}{Colors.RESET}"
            elif down_price <= self.config.max_entry_price:
                down_str = f"{Colors.GREEN}{down_price:>8.4f}{Colors.RESET}"
            else:
                down_str = f"{down_price:>8.4f}"

            ws_status = f"{Colors.GREEN}●{Colors.RESET}" if data["connected"] else f"{Colors.RED}○{Colors.RESET}"

            lines.append(
                f"{Colors.CYAN}{coin:<6}{Colors.RESET}{'':3}{up_str}{'':3}{down_str}{'':3}"
                f"{data['time']:>7}{'':3}{data['status']:<35}{'':2}{ws_status:<4}"
            )

            # Collect positions
            for pos in data["positions"]:
                all_positions.append((coin, pos, data["up"] if pos.side == "up" else data["down"]))

            total_opps += data.get("opportunities", 0)
            total_entries += data.get("entries", 0)

        lines.append("-" * width)

        # Positions section
        lines.append(f"{Colors.BOLD}OPEN POSITIONS{Colors.RESET}")
        lines.append("-" * width)

        if all_positions:
            lines.append(
                f"{'COIN':<6}{'SIDE':<6}{'ENTRY':>8}{'CURRENT':>10}{'SIZE':>10}"
                f"{'PNL':>12}{'HOLD TIME':>12}{'TP TARGET':>12}"
            )
            lines.append("-" * width)

            total_pnl = 0
            for coin, pos, current_price in all_positions:
                pnl = pos.get_pnl(current_price)
                pnl_pct = pos.get_pnl_percent(current_price)
                hold_time = pos.get_hold_time()
                total_pnl += pnl

                pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED
                side_color = Colors.GREEN if pos.side == "up" else Colors.RED

                # Format hold time
                if hold_time < 60:
                    hold_str = f"{hold_time:.0f}s"
                else:
                    hold_str = f"{hold_time/60:.1f}m"

                lines.append(
                    f"{Colors.CYAN}{coin:<6}{Colors.RESET}"
                    f"{side_color}{pos.side.upper():<6}{Colors.RESET}"
                    f"{pos.entry_price:>8.4f}"
                    f"{current_price:>10.4f}"
                    f"${pos.size * pos.entry_price:>9.2f}"
                    f"{pnl_color}${pnl:>+10.2f}{Colors.RESET}"
                    f"{hold_str:>12}"
                    f"{pos.take_profit_price:>12.4f}"
                )

            lines.append("-" * width)
            pnl_color = Colors.GREEN if total_pnl >= 0 else Colors.RED
            lines.append(f"{'':50}{Colors.BOLD}Total PnL:{Colors.RESET} {pnl_color}${total_pnl:>+.2f}{Colors.RESET}")
        else:
            lines.append(f"{Colors.DIM}  No open positions - waiting for 99%+ opportunities with <2% entry{Colors.RESET}")

        lines.append("")

        # Recent events
        lines.append(f"{Colors.BOLD}RECENT EVENTS{Colors.RESET}")
        lines.append("-" * width)

        if self.log_buffer.messages:
            for msg in self.log_buffer.get_messages():
                lines.append(f"  {msg}")
        else:
            lines.append(f"{Colors.DIM}  No events yet...{Colors.RESET}")

        lines.append("")

        # Footer stats
        lines.append(f"{Colors.BOLD}{'=' * width}{Colors.RESET}")
        lines.append(
            f"Opportunities: {total_opps}  |  "
            f"Entries: {total_entries}  |  "
            f"Time: {datetime.now().strftime('%H:%M:%S')}  |  "
            f"Press Ctrl+C to stop"
        )
        lines.append(f"{Colors.BOLD}{'=' * width}{Colors.RESET}")

        # Clear and print
        output = "\033[H\033[J" + "\n".join(lines)
        print(output, flush=True)


async def run_single_coin(coin: str, config: LongshotConfig, bot: TradingBot):
    """Run strategy for a single coin with its own TUI."""
    config.coin = coin
    strategy = LongshotStrategy(bot, config)

    try:
        await strategy.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[{coin}] Error: {e}")
    finally:
        await strategy.stop()


async def run_all_coins(config: LongshotConfig, bot: TradingBot):
    """Run strategy for all 4 coins with unified dashboard."""
    dashboard = MultiCoinDashboard(config)

    print("Initializing strategies for all coins...")

    # Create strategies for each coin
    strategies: List[LongshotStrategy] = []
    for coin in MultiCoinDashboard.COINS:
        coin_config = LongshotConfig(
            coin=coin,
            size=config.size,
            take_profit=config.take_profit,
            stop_loss=config.stop_loss,
            max_positions=config.max_positions,
            extreme_threshold=config.extreme_threshold,
            max_entry_price=config.max_entry_price,
            min_time_remaining=config.min_time_remaining,
        )
        strategy = LongshotStrategy(bot, coin_config)

        # Override strategy's log method to use dashboard
        original_log = strategy.log
        def make_logger(c):
            def log_wrapper(msg, level="info"):
                dashboard.log(f"[{c}] {msg}", level)
            return log_wrapper
        strategy.log = make_logger(coin)

        strategies.append(strategy)
        dashboard.add_strategy(coin, strategy)

    # Start all strategies
    for strategy in strategies:
        if not await strategy.start():
            dashboard.log(f"[{strategy.config.coin}] Failed to start", "error")
        else:
            dashboard.log(f"[{strategy.config.coin}] Started monitoring", "success")

    dashboard.running = True

    async def run_strategy_loop(strategy: LongshotStrategy):
        """Run the strategy tick loop."""
        while dashboard.running and strategy.running:
            try:
                prices = strategy._get_current_prices()
                await strategy.on_tick(prices)
                await strategy._check_exits(prices)
                await asyncio.sleep(0.5)
            except Exception as e:
                dashboard.log(f"[{strategy.config.coin}] Error: {e}", "error")
                await asyncio.sleep(1)

    async def render_loop():
        """Render dashboard periodically."""
        while dashboard.running:
            dashboard.render()
            await asyncio.sleep(0.3)

    # Run all strategy loops + render loop
    try:
        tasks = [asyncio.create_task(run_strategy_loop(s)) for s in strategies]
        tasks.append(asyncio.create_task(render_loop()))
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.running = False
        print("\n\nShutting down all strategies...")
        for strategy in strategies:
            await strategy.stop()
        print("All strategies stopped.")


async def main():
    parser = argparse.ArgumentParser(
        description='Longshot Reversal Strategy - Bet on 99% reversals'
    )
    parser.add_argument(
        '--coin',
        type=str,
        required=True,
        help='Coin symbol (BTC, ETH, SOL, XRP) or ALL'
    )
    parser.add_argument(
        '--size',
        type=float,
        default=1.0,
        help='Position size in USD (default: 1.0)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.99,
        help='Extreme probability threshold (default: 0.99)'
    )
    parser.add_argument(
        '--max-entry',
        type=float,
        default=0.02,
        help='Max entry price for longshot (default: 0.02)'
    )
    parser.add_argument(
        '--min-time',
        type=int,
        default=120,
        help='Minimum seconds remaining (default: 120)'
    )
    parser.add_argument(
        '--take-profit',
        type=float,
        default=0.18,
        help='Take profit delta (default: 0.18)'
    )
    parser.add_argument(
        '--stop-loss',
        type=float,
        default=0.02,
        help='Stop loss delta (default: 0.02)'
    )
    parser.add_argument(
        '--max-positions',
        type=int,
        default=2,
        help='Max concurrent positions per coin (default: 2)'
    )

    args = parser.parse_args()

    # Get credentials
    private_key = os.getenv("POLY_PRIVATE_KEY")
    safe_address = os.getenv("POLY_SAFE_ADDRESS")

    if not private_key or not safe_address:
        print("ERROR: Please set POLY_PRIVATE_KEY and POLY_SAFE_ADDRESS in .env file")
        return

    # Initialize bot
    print("Initializing trading bot...")
    bot = TradingBot(
        private_key=private_key,
        safe_address=safe_address,
        config_path="config.yaml"
    )

    # Create strategy config
    config = LongshotConfig(
        coin=args.coin,
        size=args.size,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        max_positions=args.max_positions,
        extreme_threshold=args.threshold,
        max_entry_price=args.max_entry,
        min_time_remaining=args.min_time,
    )

    if args.coin.upper() == "ALL":
        await run_all_coins(config, bot)
    else:
        print()
        print("=" * 60)
        print(f"LONGSHOT STRATEGY - {args.coin.upper()}")
        print("=" * 60)
        print(f"Threshold: {args.threshold} (buy opposite when one side >= this)")
        print(f"Max entry: {args.max_entry} (only buy if <= this price)")
        print(f"Min time: {args.min_time}s remaining")
        print(f"Position size: ${args.size}")
        print(f"Take profit: +${args.take_profit}")
        print(f"Stop loss: -${args.stop_loss}")
        print()
        print("WARNING: This will place REAL trades with REAL money!")
        print("Press Ctrl+C to stop at any time.")
        print("=" * 60)
        print()

        await run_single_coin(args.coin.upper(), config, bot)


if __name__ == "__main__":
    asyncio.run(main())
