#!/usr/bin/env python3
"""
Runner for Consolidation Breakout Strategy

A strategy that trades breakouts from tight price consolidation ranges.

Backtested results:
- 63.6% win rate
- +0.53% per trade after spread costs
- Works best on BTC (65% win rate)

Strategy Logic:
1. Detect when price consolidates (range < 2% over 30 seconds)
2. When price breaks out of the range, trade in breakout direction
3. Take profit at +4%, stop loss at -3%, time stop at 90 seconds

Usage:
    # Single coin
    python run_breakout.py --coin BTC

    # All coins
    python run_breakout.py --coin ALL

    # Custom settings
    python run_breakout.py --coin ETH --size 5.0 --tp 0.05
"""
import asyncio
import argparse
import os
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

from lib.console import Colors, LogBuffer, format_countdown
from strategies.breakout import BreakoutStrategy, BreakoutConfig
from src.bot import TradingBot


class BreakoutMultiCoinDashboard:
    """Unified dashboard for breakout strategy across all coins."""

    COINS = ["BTC", "ETH", "SOL", "XRP"]

    def __init__(self, config: BreakoutConfig):
        self.config = config
        self.strategies: Dict[str, BreakoutStrategy] = {}
        self.log_buffer = LogBuffer(max_size=10)
        self.running = False

    def add_strategy(self, coin: str, strategy: BreakoutStrategy):
        self.strategies[coin] = strategy

    def log(self, msg: str, level: str = "info"):
        self.log_buffer.add(msg, level)

    def _get_coin_data(self, coin: str) -> Dict:
        strategy = self.strategies.get(coin)
        if not strategy:
            return {
                "up": 0.5, "down": 0.5, "time_secs": 0,
                "up_cons": None, "down_cons": None,
                "connected": False, "positions": [],
                "breakouts": 0, "trades": 0, "wins": 0, "losses": 0,
            }

        prices = strategy._get_current_prices()
        time_secs = strategy._get_time_remaining_seconds()

        return {
            "up": prices.get("up", 0.5),
            "down": prices.get("down", 0.5),
            "time_secs": time_secs,
            "up_cons": strategy._consolidation.get("up"),
            "down_cons": strategy._consolidation.get("down"),
            "up_in_cons": strategy._in_consolidation.get("up", False),
            "down_in_cons": strategy._in_consolidation.get("down", False),
            "connected": strategy.is_connected,
            "positions": strategy.positions.get_all_positions(),
            "breakouts": strategy.breakouts_detected,
            "trades": strategy.trades_entered,
            "wins": strategy.wins,
            "losses": strategy.losses,
        }

    def render(self):
        lines = []
        width = 110

        # Header
        lines.append(f"{Colors.BOLD}{'='*width}{Colors.RESET}")
        lines.append(
            f"{Colors.CYAN}{Colors.BOLD}BREAKOUT STRATEGY{Colors.RESET} - ALL COINS  |  "
            f"TP: +{self.config.take_profit*100:.0f}%  |  "
            f"SL: -{self.config.stop_loss*100:.0f}%  |  "
            f"Time Stop: {self.config.time_stop}s"
        )
        lines.append(f"{Colors.BOLD}{'='*width}{Colors.RESET}")

        # Column headers
        lines.append(
            f"{Colors.BOLD}{'COIN':<6}{'UP':>8}{'DOWN':>8}{'TIME':>8}"
            f"{'CONSOLIDATION':<30}{'STATUS':<20}{'W/L':>8}{'WS':>4}{Colors.RESET}"
        )
        lines.append("-" * width)

        # Coin rows
        all_positions = []
        total_breakouts = 0
        total_trades = 0
        total_wins = 0
        total_losses = 0

        for coin in self.COINS:
            data = self._get_coin_data(coin)

            # Prices
            up_price = data["up"]
            down_price = data["down"]

            # Time
            mins, secs = divmod(data["time_secs"], 60)
            time_str = format_countdown(mins, secs)

            # Consolidation status
            up_cons = data.get("up_cons")
            down_cons = data.get("down_cons")

            if data.get("up_in_cons") and up_cons:
                cons_str = f"{Colors.YELLOW}UP: {up_cons.low:.3f}-{up_cons.high:.3f}{Colors.RESET}"
            elif data.get("down_in_cons") and down_cons:
                cons_str = f"{Colors.YELLOW}DN: {down_cons.low:.3f}-{down_cons.high:.3f}{Colors.RESET}"
            else:
                cons_str = f"{Colors.DIM}Monitoring...{Colors.RESET}"

            # Status
            if data.get("up_in_cons") or data.get("down_in_cons"):
                status = f"{Colors.YELLOW}CONSOLIDATING{Colors.RESET}"
            elif data["time_secs"] < self.config.min_time_remaining:
                status = f"{Colors.DIM}Wait for time{Colors.RESET}"
            else:
                status = f"{Colors.DIM}Watching{Colors.RESET}"

            ws = f"{Colors.GREEN}●{Colors.RESET}" if data["connected"] else f"{Colors.RED}○{Colors.RESET}"
            wl = f"{data['wins']}/{data['losses']}"

            lines.append(
                f"{Colors.CYAN}{coin:<6}{Colors.RESET}"
                f"{Colors.GREEN}{up_price:>8.4f}{Colors.RESET}"
                f"{Colors.RED}{down_price:>8.4f}{Colors.RESET}"
                f"{time_str:>8}"
                f"{cons_str:<30}"
                f"{status:<20}"
                f"{wl:>8}"
                f"{ws:>4}"
            )

            for pos in data["positions"]:
                curr = data["up"] if pos.side == "up" else data["down"]
                all_positions.append((coin, pos, curr))

            total_breakouts += data["breakouts"]
            total_trades += data["trades"]
            total_wins += data["wins"]
            total_losses += data["losses"]

        lines.append("-" * width)

        # Positions
        lines.append(f"{Colors.BOLD}POSITIONS{Colors.RESET}")
        lines.append("-" * width)

        if all_positions:
            lines.append(
                f"{'COIN':<6}{'SIDE':<6}{'ENTRY':>8}{'CURRENT':>9}{'PNL':>12}"
                f"{'PROGRESS':<20}{'HOLD':>8}"
            )

            total_pnl = 0
            for coin, pos, curr in all_positions:
                pnl = pos.get_pnl(curr)
                hold = pos.get_hold_time()
                total_pnl += pnl

                # Progress bar
                if curr > pos.entry_price:
                    progress = (curr - pos.entry_price) / self.config.take_profit
                    bar = f"{Colors.GREEN}{'█' * int(min(progress, 1) * 10)}{'░' * (10 - int(min(progress, 1) * 10))}{Colors.RESET}"
                else:
                    progress = (pos.entry_price - curr) / self.config.stop_loss
                    bar = f"{Colors.RED}{'█' * int(min(progress, 1) * 10)}{'░' * (10 - int(min(progress, 1) * 10))}{Colors.RESET}"

                pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED
                side_color = Colors.GREEN if pos.side == "up" else Colors.RED

                lines.append(
                    f"{Colors.CYAN}{coin:<6}{Colors.RESET}"
                    f"{side_color}{pos.side.upper():<6}{Colors.RESET}"
                    f"{pos.entry_price:>8.4f}"
                    f"{curr:>9.4f}"
                    f"{pnl_color}${pnl:>+10.2f}{Colors.RESET}"
                    f" [{bar}]"
                    f"{hold:>7.0f}s"
                )

            lines.append("-" * width)
            pnl_color = Colors.GREEN if total_pnl >= 0 else Colors.RED
            lines.append(f"{'':60}{Colors.BOLD}Total:{Colors.RESET} {pnl_color}${total_pnl:>+.2f}{Colors.RESET}")
        else:
            lines.append(f"{Colors.DIM}  Waiting for consolidation breakout...{Colors.RESET}")

        lines.append("")

        # Events
        lines.append(f"{Colors.BOLD}EVENTS{Colors.RESET}")
        lines.append("-" * width)
        if self.log_buffer.messages:
            for msg in self.log_buffer.get_messages():
                lines.append(f"  {msg}")
        else:
            lines.append(f"{Colors.DIM}  No events yet{Colors.RESET}")

        # Footer
        lines.append("")
        lines.append(f"{Colors.BOLD}{'='*width}{Colors.RESET}")
        win_rate = 100 * total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0
        lines.append(
            f"Breakouts: {total_breakouts}  |  Trades: {total_trades}  |  "
            f"Win Rate: {win_rate:.1f}%  |  "
            f"Time: {datetime.now().strftime('%H:%M:%S')}  |  Ctrl+C to stop"
        )
        lines.append(f"{Colors.BOLD}{'='*width}{Colors.RESET}")

        print("\033[H\033[J" + "\n".join(lines), flush=True)


async def run_single(coin: str, config: BreakoutConfig, bot: TradingBot):
    """Run strategy for single coin."""
    config.coin = coin
    strategy = BreakoutStrategy(bot, config)

    try:
        await strategy.run()
    except KeyboardInterrupt:
        pass
    finally:
        await strategy.stop()


async def run_all(config: BreakoutConfig, bot: TradingBot):
    """Run strategy for all coins with unified dashboard."""
    dashboard = BreakoutMultiCoinDashboard(config)

    print("Initializing breakout strategies...")

    strategies: List[BreakoutStrategy] = []
    for coin in BreakoutMultiCoinDashboard.COINS:
        coin_config = BreakoutConfig(
            coin=coin,
            size=config.size,
            consolidation_window=config.consolidation_window,
            max_consolidation_range=config.max_consolidation_range,
            min_consolidation_range=config.min_consolidation_range,
            breakout_threshold=config.breakout_threshold,
            min_time_remaining=config.min_time_remaining,
            take_profit=config.take_profit,
            stop_loss=config.stop_loss,
            time_stop=config.time_stop,
            trade_cooldown=config.trade_cooldown,
            max_positions=config.max_positions,
        )
        strategy = BreakoutStrategy(bot, coin_config)

        # Redirect logs to dashboard
        def make_logger(c):
            def log_wrapper(msg, level="info"):
                dashboard.log(f"[{c}] {msg}", level)
            return log_wrapper
        strategy.log = make_logger(coin)

        strategies.append(strategy)
        dashboard.add_strategy(coin, strategy)

    # Try to start all
    for strategy in strategies:
        if await strategy.start():
            dashboard.log(f"[{strategy.config.coin}] Started", "success")
        else:
            dashboard.log(f"[{strategy.config.coin}] No market yet - will retry", "warning")

    dashboard.running = True

    async def strategy_loop(strategy):
        """Main loop for a single strategy with auto-restart."""
        retry_interval = 15

        while dashboard.running:
            if not strategy.running:
                if await strategy.start():
                    dashboard.log(f"[{strategy.config.coin}] Connected to market", "success")
                else:
                    await asyncio.sleep(retry_interval)
                    continue

            try:
                prices = strategy._get_current_prices()
                await strategy.on_tick(prices)
                await strategy._check_exits(prices)
                await asyncio.sleep(0.5)
            except Exception as e:
                dashboard.log(f"[{strategy.config.coin}] {e}", "error")
                await asyncio.sleep(1)

    async def render_loop():
        while dashboard.running:
            dashboard.render()
            await asyncio.sleep(0.3)

    try:
        tasks = [asyncio.create_task(strategy_loop(s)) for s in strategies]
        tasks.append(asyncio.create_task(render_loop()))
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.running = False
        print("\n\nShutting down...")
        for s in strategies:
            await s.stop()
        print("Done.")


async def main():
    parser = argparse.ArgumentParser(
        description='Breakout Strategy - Trade consolidation breakouts'
    )
    parser.add_argument('--coin', type=str, required=True, help='BTC, ETH, SOL, XRP, or ALL')
    parser.add_argument('--size', type=float, default=2.0, help='Position size USD (default: 2.0)')
    parser.add_argument('--tp', type=float, default=0.04, help='Take profit (default: 0.04 = 4%)')
    parser.add_argument('--sl', type=float, default=0.03, help='Stop loss (default: 0.03 = 3%)')
    parser.add_argument('--time-stop', type=int, default=90, help='Time stop seconds (default: 90)')
    parser.add_argument('--min-time', type=int, default=180, help='Min time remaining (default: 180)')
    parser.add_argument('--cooldown', type=int, default=30, help='Trade cooldown seconds (default: 30)')

    args = parser.parse_args()

    private_key = os.getenv("POLY_PRIVATE_KEY")
    safe_address = os.getenv("POLY_SAFE_ADDRESS")

    if not private_key or not safe_address:
        print("ERROR: Set POLY_PRIVATE_KEY and POLY_SAFE_ADDRESS in .env")
        return

    print("Initializing trading bot...")
    bot = TradingBot(
        private_key=private_key,
        safe_address=safe_address,
        config_path="config.yaml"
    )

    config = BreakoutConfig(
        coin=args.coin,
        size=args.size,
        take_profit=args.tp,
        stop_loss=args.sl,
        time_stop=args.time_stop,
        min_time_remaining=args.min_time,
        trade_cooldown=args.cooldown,
    )

    if args.coin.upper() == "ALL":
        await run_all(config, bot)
    else:
        print()
        print("=" * 70)
        print(f"BREAKOUT STRATEGY - {args.coin.upper()}")
        print("=" * 70)
        print(f"Take Profit: +{args.tp*100:.0f}%  |  Stop Loss: -{args.sl*100:.0f}%")
        print(f"Time Stop: {args.time_stop}s  |  Min Time: {args.min_time}s")
        print(f"Size: ${args.size} per trade")
        print()
        print("Based on backtesting: 63.6% win rate, +0.53% per trade")
        print()
        print("WARNING: REAL trades with REAL money!")
        print("=" * 70)
        print()

        await run_single(args.coin.upper(), config, bot)


if __name__ == "__main__":
    asyncio.run(main())
