#!/usr/bin/env python3
"""
Runner for OPTIMIZED Smart Longshot Strategy

Based on 18+ hours of tick data analysis (1.1M+ ticks, 280 markets):

KEY FINDING: Timing is everything!
- Entry window: 45-120 seconds remaining
- Win rate: 4.6% (vs 1.4% breakeven needed)
- ROI: +169% on backtested trades

IMPORTANT - Only trade profitable coins:
- BTC: +282% ROI (6.2% win rate)
- SOL: +269% ROI (7.0% win rate)
- ETH: AVOID (-100% ROI in testing)
- XRP: AVOID (-100% ROI in testing)

Usage:
    # Recommended: BTC and SOL only
    python run_smart_longshot.py --coin ALL

    # Single coin
    python run_smart_longshot.py --coin BTC

    # Custom settings
    python run_smart_longshot.py --coin SOL --size 2.0
"""
import asyncio
import argparse
import os
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

from lib.console import Colors, LogBuffer, format_countdown
from strategies.smart_longshot import SmartLongshotStrategy, SmartLongshotConfig
from src.bot import TradingBot


class SmartMultiCoinDashboard:
    """Unified dashboard for smart longshot across profitable coins only."""

    # OPTIMIZED: Only BTC and SOL are profitable based on backtesting
    # ETH and XRP had 0% win rate in 18 hours of data
    COINS = ["BTC", "SOL"]

    def __init__(self, config: SmartLongshotConfig):
        self.config = config
        self.strategies: Dict[str, SmartLongshotStrategy] = {}
        self.log_buffer = LogBuffer(max_size=10)
        self.running = False

    def add_strategy(self, coin: str, strategy: SmartLongshotStrategy):
        self.strategies[coin] = strategy

    def log(self, msg: str, level: str = "info"):
        self.log_buffer.add(msg, level)

    def _get_coin_data(self, coin: str) -> Dict:
        strategy = self.strategies.get(coin)
        if not strategy:
            return {
                "up": 0.5, "down": 0.5, "time_secs": 0,
                "up_signal": 0, "down_signal": 0,
                "connected": False, "positions": [],
                "signals": 0, "entries": 0, "filtered": 0,
            }

        prices = strategy._get_current_prices()
        time_secs = strategy._get_time_remaining_seconds()
        up_metrics = strategy._metrics["up"]
        down_metrics = strategy._metrics["down"]

        return {
            "up": prices.get("up", 0.5),
            "down": prices.get("down", 0.5),
            "time_secs": time_secs,
            "up_signal": up_metrics.signal_strength,
            "down_signal": down_metrics.signal_strength,
            "up_depth_ratio": up_metrics.depth_ratio,
            "down_depth_ratio": down_metrics.depth_ratio,
            "up_vol": up_metrics.volatility,
            "down_vol": down_metrics.volatility,
            "connected": strategy.is_connected,
            "positions": strategy.positions.get_all_positions(),
            "signals": strategy.signals_seen,
            "entries": strategy.entries_made,
            "filtered": strategy.filtered_out,
        }

    def render(self):
        lines = []
        width = 110

        # Header
        lines.append(f"{Colors.BOLD}{'='*width}{Colors.RESET}")
        lines.append(
            f"{Colors.MAGENTA}{Colors.BOLD}OPTIMIZED LONGSHOT{Colors.RESET} - BTC/SOL  |  "
            f"Window: {self.config.min_time_remaining}-{self.config.max_time_remaining}s  |  "
            f"Entry: <={self.config.max_entry_price*100:.1f}%  |  "
            f"Expected: ~4.6% wins @ 40-60x"
        )
        lines.append(f"{Colors.BOLD}{'='*width}{Colors.RESET}")

        # Column headers
        lines.append(
            f"{Colors.BOLD}{'COIN':<6}{'UP':>8}{'DOWN':>8}{'TIME':>8}"
            f"{'UP Sig':>8}{'DN Sig':>8}{'UP D/R':>8}{'DN D/R':>8}"
            f"{'STATUS':<25}{'WS':>4}{Colors.RESET}"
        )
        lines.append("-" * width)

        # Coin rows
        all_positions = []
        total_signals = 0
        total_entries = 0
        total_filtered = 0

        for coin in self.COINS:
            data = self._get_coin_data(coin)

            # Format prices
            up_price = data["up"]
            down_price = data["down"]

            up_color = Colors.GREEN if up_price <= self.config.max_entry_price else ""
            down_color = Colors.GREEN if down_price <= self.config.max_entry_price else ""

            # Time
            mins, secs = divmod(data["time_secs"], 60)
            time_str = format_countdown(mins, secs)

            # Signal strength colors
            def sig_color(s):
                if s >= 3:
                    return Colors.GREEN
                elif s >= 2:
                    return Colors.YELLOW
                return Colors.DIM

            up_sig_color = sig_color(data["up_signal"])
            down_sig_color = sig_color(data["down_signal"])

            # Status - OPTIMIZED time window logic
            time_secs = data["time_secs"]
            in_window = self.config.min_time_remaining <= time_secs <= self.config.max_time_remaining
            too_early = time_secs > self.config.max_time_remaining
            too_late = time_secs < self.config.min_time_remaining

            if too_late:
                status = f"{Colors.RED}Too late{Colors.RESET}"
            elif too_early:
                wait = time_secs - self.config.max_time_remaining
                status = f"{Colors.DIM}Wait {wait}s{Colors.RESET}"
            elif in_window and up_price <= self.config.max_entry_price:
                status = f"{Colors.YELLOW}>>> BUY UP{Colors.RESET}"
            elif in_window and down_price <= self.config.max_entry_price:
                status = f"{Colors.YELLOW}>>> BUY DOWN{Colors.RESET}"
            elif in_window:
                status = f"{Colors.GREEN}IN WINDOW{Colors.RESET}"
            else:
                status = f"{Colors.DIM}Monitoring{Colors.RESET}"

            ws = f"{Colors.GREEN}●{Colors.RESET}" if data["connected"] else f"{Colors.RED}○{Colors.RESET}"

            lines.append(
                f"{Colors.CYAN}{coin:<6}{Colors.RESET}"
                f"{up_color}{up_price:>8.4f}{Colors.RESET}"
                f"{down_color}{down_price:>8.4f}{Colors.RESET}"
                f"{time_str:>8}"
                f"{up_sig_color}{data['up_signal']:>7}/4{Colors.RESET}"
                f"{down_sig_color}{data['down_signal']:>7}/4{Colors.RESET}"
                f"{data.get('up_depth_ratio', 0):>8.1f}"
                f"{data.get('down_depth_ratio', 0):>8.1f}"
                f"{status:<25}"
                f"{ws:>4}"
            )

            for pos in data["positions"]:
                curr = data["up"] if pos.side == "up" else data["down"]
                all_positions.append((coin, pos, curr))

            total_signals += data["signals"]
            total_entries += data["entries"]
            total_filtered += data["filtered"]

        lines.append("-" * width)

        # Positions
        lines.append(f"{Colors.BOLD}POSITIONS{Colors.RESET}")
        lines.append("-" * width)

        if all_positions:
            lines.append(
                f"{'COIN':<6}{'SIDE':<6}{'ENTRY':>8}{'CURRENT':>9}{'TARGET':>9}"
                f"{'PNL':>12}{'PROGRESS':<15}{'HOLD':>8}"
            )

            total_pnl = 0
            for coin, pos, curr in all_positions:
                pnl = pos.get_pnl(curr)
                pnl_pct = pos.get_pnl_percent(curr)
                hold = pos.get_hold_time()
                total_pnl += pnl

                # Progress bar
                if pos.take_profit_price > pos.entry_price:
                    prog = (curr - pos.entry_price) / (pos.take_profit_price - pos.entry_price)
                    prog = max(0, min(1, prog))
                    bar = "█" * int(prog * 10) + "░" * (10 - int(prog * 10))
                else:
                    bar = "░" * 10

                pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED
                side_color = Colors.GREEN if pos.side == "up" else Colors.RED

                lines.append(
                    f"{Colors.CYAN}{coin:<6}{Colors.RESET}"
                    f"{side_color}{pos.side.upper():<6}{Colors.RESET}"
                    f"{pos.entry_price:>8.4f}"
                    f"{curr:>9.4f}"
                    f"{pos.take_profit_price:>9.4f}"
                    f"{pnl_color}${pnl:>+10.2f}{Colors.RESET}"
                    f" [{bar}]"
                    f"{hold:>7.0f}s"
                )

            lines.append("-" * width)
            pnl_color = Colors.GREEN if total_pnl >= 0 else Colors.RED
            lines.append(f"{'':60}{Colors.BOLD}Total:{Colors.RESET} {pnl_color}${total_pnl:>+.2f}{Colors.RESET}")
        else:
            lines.append(f"{Colors.DIM}  Waiting for high-signal opportunities (3+/4 signals required){Colors.RESET}")

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
        lines.append(
            f"Signals: {total_signals}  |  Filtered: {total_filtered}  |  "
            f"Entries: {total_entries}  |  "
            f"Time: {datetime.now().strftime('%H:%M:%S')}  |  Ctrl+C to stop"
        )
        lines.append(f"{Colors.BOLD}{'='*width}{Colors.RESET}")

        print("\033[H\033[J" + "\n".join(lines), flush=True)


async def run_single(coin: str, config: SmartLongshotConfig, bot: TradingBot):
    """Run strategy for single coin."""
    config.coin = coin
    strategy = SmartLongshotStrategy(bot, config)

    try:
        await strategy.run()
    except KeyboardInterrupt:
        pass
    finally:
        await strategy.stop()


async def run_all(config: SmartLongshotConfig, bot: TradingBot):
    """Run strategy for all coins with unified dashboard."""
    dashboard = SmartMultiCoinDashboard(config)

    print("Initializing smart longshot strategies...")

    strategies: List[SmartLongshotStrategy] = []
    for coin in SmartMultiCoinDashboard.COINS:
        coin_config = SmartLongshotConfig(
            coin=coin,
            size=config.size,
            max_entry_price=config.max_entry_price,
            min_time_remaining=config.min_time_remaining,
            max_time_remaining=config.max_time_remaining,
            depth_ratio_threshold=config.depth_ratio_threshold,
            volatility_threshold=config.volatility_threshold,
            max_ask_depth=config.max_ask_depth,
            use_orderbook_filter=config.use_orderbook_filter,
            target_multiple=config.target_multiple,
            stop_loss_pct=config.stop_loss_pct,
            max_positions=config.max_positions,
        )
        strategy = SmartLongshotStrategy(bot, coin_config)

        # Redirect logs to dashboard
        def make_logger(c):
            def log_wrapper(msg, level="info"):
                dashboard.log(f"[{c}] {msg}", level)
            return log_wrapper
        strategy.log = make_logger(coin)

        strategies.append(strategy)
        dashboard.add_strategy(coin, strategy)

    # Try to start all (some may fail if no market available)
    for strategy in strategies:
        if await strategy.start():
            dashboard.log(f"[{strategy.config.coin}] Started", "success")
        else:
            dashboard.log(f"[{strategy.config.coin}] No market yet - will retry", "warning")

    dashboard.running = True

    async def strategy_loop(strategy):
        """Main loop for a single strategy with auto-restart."""
        retry_interval = 15  # seconds between retry attempts

        while dashboard.running:
            # If not running, try to start
            if not strategy.running:
                if await strategy.start():
                    dashboard.log(f"[{strategy.config.coin}] Connected to market", "success")
                else:
                    # No market available, wait and retry
                    await asyncio.sleep(retry_interval)
                    continue

            # Normal tick loop
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
        description='OPTIMIZED Longshot Strategy - Late-game reversal betting (45-120s window)'
    )
    parser.add_argument('--coin', type=str, required=True,
                        help='BTC, SOL, or ALL (recommended). ETH/XRP not profitable.')
    parser.add_argument('--size', type=float, default=1.0,
                        help='Position size USD (default: 1.0)')
    parser.add_argument('--max-entry', type=float, default=0.025,
                        help='Max entry price (default: 0.025 = 2.5%%)')
    parser.add_argument('--min-time', type=int, default=45,
                        help='Min seconds remaining (default: 45)')
    parser.add_argument('--max-time', type=int, default=120,
                        help='Max seconds remaining (default: 120)')
    parser.add_argument('--use-filters', action='store_true',
                        help='Enable orderbook filters (optional, timing is key)')
    parser.add_argument('--stop-loss', type=float, default=0.80,
                        help='Stop loss percent (default: 0.80 = 80%%)')
    parser.add_argument('--max-positions', type=int, default=1,
                        help='Max positions per coin (default: 1)')

    args = parser.parse_args()

    # Warn about unprofitable coins
    if args.coin.upper() in ['ETH', 'XRP']:
        print(f"WARNING: {args.coin.upper()} had 0% win rate in backtesting!")
        print("Recommended: Use BTC, SOL, or ALL (which uses BTC+SOL only)")
        print()

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

    config = SmartLongshotConfig(
        coin=args.coin,
        size=args.size,
        max_entry_price=args.max_entry,
        min_time_remaining=args.min_time,
        max_time_remaining=args.max_time,
        use_orderbook_filter=args.use_filters,
        stop_loss_pct=args.stop_loss,
        max_positions=args.max_positions,
    )

    if args.coin.upper() == "ALL":
        await run_all(config, bot)
    else:
        print()
        print("=" * 70)
        print(f"OPTIMIZED LONGSHOT - {args.coin.upper()}")
        print("=" * 70)
        print(f"Entry: <= {args.max_entry*100:.1f}%  |  Window: {args.min_time}-{args.max_time}s")
        print(f"Strategy: Hold to market end (binary outcome)")
        print(f"Size: ${args.size} max per trade")
        print()
        print("Based on backtesting: 4.6% win rate @ ~50x payout = +169% ROI")
        print()
        print("WARNING: REAL trades with REAL money!")
        print("=" * 70)
        print()

        await run_single(args.coin.upper(), config, bot)


if __name__ == "__main__":
    asyncio.run(main())
