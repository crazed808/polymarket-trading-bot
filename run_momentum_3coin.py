#!/usr/bin/env python3
"""
Runner for Momentum Quality_Combo Strategy (OPTIMIZED)

Based on comprehensive optimization testing (12+ strategy variants):
- BTC: 61.0% win rate, +1.57% per trade
- ETH: 60.4% win rate, +1.54% per trade
- SOL: 54.9% win rate, +0.52% per trade (EXCLUDED - marginal)
- XRP: EXCLUDED (unprofitable)

Combined BTC+ETH: 60.8% win rate, +1.56% avg profit per trade

Quality_Combo Strategy:
1. Main momentum: 15% price move in 30 seconds
2. Multi-timeframe confirmation: 8% move in 10 seconds
3. Momentum quality filter: < 3% volatility (smooth signals only)
4. Orderbook confirmation: depth ratio > 2.0
5. Trailing stop: Exit on 3% drawdown from peak
6. TP: +6%, SL: -3% (wider = better), Time stop: 80s

Improvement: 2.6x better than baseline (+0.57%/trade)

Usage:
    # Both coins (recommended - best combined performance)
    python run_momentum_3coin.py --coin ALL

    # Single coin
    python run_momentum_3coin.py --coin BTC

    # Custom settings (not recommended - defaults are optimized)
    python run_momentum_3coin.py --coin ETH --size 5.0 --tp 0.05

    # With logging to file (recommended)
    python run_momentum_3coin.py --coin ALL --log
"""
import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

# Setup file logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def setup_logging(enable_file_log: bool = False) -> logging.Logger:
    """Setup logging to both console and file."""
    logger = logging.getLogger("momentum_3coin")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers = []

    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    if enable_file_log:
        # File handler (DEBUG level - captures everything)
        log_file = LOG_DIR / f"momentum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
        print(f"Logging to: {log_file}")

    return logger


# Global logger
logger = setup_logging()

from lib.console import Colors, LogBuffer, format_countdown
from strategies.momentum import MomentumStrategy, MomentumConfig
from src.bot import TradingBot


class MomentumMultiCoinDashboard:
    """Unified dashboard for momentum strategy across profitable coins only."""

    # Only highly profitable coins based on Quality_Combo optimization
    # BTC: +1.57%/trade, 61.0% WR
    # ETH: +1.54%/trade, 60.4% WR
    # SOL: +0.52%/trade (excluded - marginal)
    COINS = ["BTC", "ETH"]

    def __init__(self, config: MomentumConfig):
        self.config = config
        self.strategies: Dict[str, MomentumStrategy] = {}
        self.log_buffer = LogBuffer(max_size=10)
        self.running = False
        self._render_lock = asyncio.Lock()  # Prevent display corruption

    def add_strategy(self, coin: str, strategy: MomentumStrategy):
        self.strategies[coin] = strategy

    def log(self, msg: str, level: str = "info"):
        self.log_buffer.add(msg, level)
        # Also log to file
        if level == "error":
            logger.error(msg)
        elif level == "warning":
            logger.warning(msg)
        elif level == "success":
            logger.info(f"✓ {msg}")
        elif level == "trade":
            logger.info(f"$ {msg}")
        else:
            logger.debug(msg)

    def _get_coin_data(self, coin: str) -> Dict:
        strategy = self.strategies.get(coin)
        if not strategy:
            return {
                "up": 0.5, "down": 0.5, "time_secs": 0,
                "up_change": None, "down_change": None,
                "up_depth": 1.0, "down_depth": 1.0,
                "connected": False, "positions": [],
                "signals": 0, "trades": 0, "wins": 0, "losses": 0,
            }

        prices = strategy._get_current_prices()
        time_secs = strategy._get_time_remaining_seconds()

        up_change = strategy._get_price_change("up", strategy.momentum_config.momentum_window)
        down_change = strategy._get_price_change("down", strategy.momentum_config.momentum_window)

        # Calculate unrealized P&L from open positions
        unrealized_pnl = 0.0
        for pos in strategy.positions.get_all_positions():
            price = prices.get(pos.side, 0)
            if price > 0:
                unrealized_pnl += pos.get_pnl(price)

        return {
            "up": prices.get("up", 0.5),
            "down": prices.get("down", 0.5),
            "time_secs": time_secs,
            "up_change": up_change,
            "down_change": down_change,
            "up_depth": strategy._orderbook["up"].depth_ratio,
            "down_depth": strategy._orderbook["down"].depth_ratio,
            "connected": strategy.is_connected,
            "positions": strategy.positions.get_all_positions(),
            "signals": strategy.signals_detected,
            "trades": strategy.trades_entered,
            "wins": strategy.wins,
            "losses": strategy.losses,
            "realized_pnl": strategy.positions.total_pnl,
            "unrealized_pnl": unrealized_pnl,
        }

    def render(self):
        lines = []
        width = 115

        # Header
        lines.append(f"{Colors.BOLD}{'='*width}{Colors.RESET}")
        # Show enabled optimizations
        optimizations = []
        if self.config.use_multi_timeframe:
            optimizations.append("Multi-TF")
        if self.config.check_momentum_quality:
            optimizations.append("Quality")
        if self.config.use_trailing_stop:
            optimizations.append("Trailing")
        opt_str = f" | {'+'.join(optimizations)}" if optimizations else ""

        lines.append(
            f"{Colors.CYAN}{Colors.BOLD}MOMENTUM QUALITY_COMBO{Colors.RESET} - BTC/ETH{opt_str}  |  "
            f"TP: +{self.config.take_profit*100:.0f}%  |  "
            f"SL: -{self.config.stop_loss*100:.0f}%  |  "
            f"Momentum: {self.config.momentum_threshold*100:.0f}%/{self.config.momentum_window}s"
        )
        lines.append(f"{Colors.BOLD}{'='*width}{Colors.RESET}")

        # Column headers
        lines.append(
            f"{Colors.BOLD}{'COIN':<6}{'UP':>8}{'DOWN':>8}{'TIME':>8}"
            f"{'UP Δ':>10}{'DN Δ':>10}{'UP D/R':>8}{'DN D/R':>8}"
            f"{'STATUS':<20}{'W/L':>8}{'WS':>4}{Colors.RESET}"
        )
        lines.append("-" * width)

        # Coin rows
        all_positions = []
        total_signals = 0
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

            # Momentum changes
            def change_str(change):
                if change is None:
                    return f"{Colors.DIM}  N/A{Colors.RESET}"
                color = Colors.GREEN if change >= 0.10 else Colors.RED if change <= -0.10 else Colors.DIM
                return f"{color}{change*100:+6.1f}%{Colors.RESET}"

            up_change_str = change_str(data["up_change"])
            down_change_str = change_str(data["down_change"])

            # Depth ratios
            def depth_color(ratio):
                if ratio >= 1.5:
                    return Colors.GREEN
                elif ratio <= 0.67:
                    return Colors.RED
                return Colors.DIM

            # Status
            has_momentum = (
                (data["up_change"] is not None and abs(data["up_change"]) >= self.config.momentum_threshold) or
                (data["down_change"] is not None and abs(data["down_change"]) >= self.config.momentum_threshold)
            )

            if has_momentum and data["time_secs"] >= self.config.min_time_remaining:
                status = f"{Colors.YELLOW}>>> SIGNAL{Colors.RESET}"
            elif data["time_secs"] < self.config.min_time_remaining:
                status = f"{Colors.DIM}Wait time{Colors.RESET}"
            else:
                status = f"{Colors.DIM}Monitoring{Colors.RESET}"

            ws = f"{Colors.GREEN}●{Colors.RESET}" if data["connected"] else f"{Colors.RED}○{Colors.RESET}"
            wl = f"{data['wins']}/{data['losses']}"

            lines.append(
                f"{Colors.CYAN}{coin:<6}{Colors.RESET}"
                f"{Colors.GREEN}{up_price:>8.4f}{Colors.RESET}"
                f"{Colors.RED}{down_price:>8.4f}{Colors.RESET}"
                f"{time_str:>8}"
                f"{up_change_str:>10}"
                f"{down_change_str:>10}"
                f"{depth_color(data['up_depth'])}{data['up_depth']:>8.2f}{Colors.RESET}"
                f"{depth_color(data['down_depth'])}{data['down_depth']:>8.2f}{Colors.RESET}"
                f"{status:<20}"
                f"{wl:>8}"
                f"{ws:>4}"
            )

            for pos in data["positions"]:
                curr = data["up"] if pos.side == "up" else data["down"]
                all_positions.append((coin, pos, curr))

            total_signals += data["signals"]
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

                # Progress bar - use position's actual delta, not config percentage
                if curr > pos.entry_price:
                    progress = (curr - pos.entry_price) / pos.take_profit_delta if pos.take_profit_delta > 0 else 0
                    bar = f"{Colors.GREEN}{'█' * int(min(progress, 1) * 10)}{'░' * (10 - int(min(progress, 1) * 10))}{Colors.RESET}"
                else:
                    progress = (pos.entry_price - curr) / pos.stop_loss_delta if pos.stop_loss_delta > 0 else 0
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
            lines.append(f"{Colors.DIM}  Waiting for momentum signal with orderbook confirmation...{Colors.RESET}")

        lines.append("")

        # Session P&L Summary
        lines.append(f"{Colors.BOLD}SESSION P&L{Colors.RESET}")
        lines.append("-" * width)

        # Collect P&L data per coin
        session_total_realized = 0.0
        session_total_unrealized = 0.0
        coin_pnl_data = []

        for coin in self.COINS:
            data = self._get_coin_data(coin)
            realized = data.get("realized_pnl", 0.0)
            unrealized = data.get("unrealized_pnl", 0.0)
            total = realized + unrealized
            trades = data.get("trades", 0)
            wins = data.get("wins", 0)
            losses = data.get("losses", 0)

            session_total_realized += realized
            session_total_unrealized += unrealized
            coin_pnl_data.append((coin, realized, unrealized, total, trades, wins, losses))

        # Header
        lines.append(
            f"{'COIN':<6}{'REALIZED':>12}{'UNREALIZED':>12}{'TOTAL':>12}"
            f"{'TRADES':>10}{'W/L':>10}"
        )

        # Per-coin rows
        for coin, realized, unrealized, total, trades, wins, losses in coin_pnl_data:
            r_color = Colors.GREEN if realized >= 0 else Colors.RED
            u_color = Colors.GREEN if unrealized >= 0 else Colors.RED
            t_color = Colors.GREEN if total >= 0 else Colors.RED

            lines.append(
                f"{Colors.CYAN}{coin:<6}{Colors.RESET}"
                f"{r_color}${realized:>+10.2f}{Colors.RESET}"
                f"{u_color}${unrealized:>+10.2f}{Colors.RESET}"
                f"{t_color}${total:>+10.2f}{Colors.RESET}"
                f"{trades:>10}"
                f"{wins}/{losses:>8}"
            )

        # Session totals
        session_total = session_total_realized + session_total_unrealized
        lines.append("-" * width)
        r_color = Colors.GREEN if session_total_realized >= 0 else Colors.RED
        u_color = Colors.GREEN if session_total_unrealized >= 0 else Colors.RED
        t_color = Colors.GREEN if session_total >= 0 else Colors.RED

        lines.append(
            f"{Colors.BOLD}{'TOTAL':<6}{Colors.RESET}"
            f"{r_color}${session_total_realized:>+10.2f}{Colors.RESET}"
            f"{u_color}${session_total_unrealized:>+10.2f}{Colors.RESET}"
            f"{t_color}{Colors.BOLD}${session_total:>+10.2f}{Colors.RESET}"
            f"{total_trades:>10}"
            f"{total_wins}/{total_losses:>8}"
        )

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
            f"Signals: {total_signals}  |  Trades: {total_trades}  |  "
            f"Win Rate: {win_rate:.1f}%  |  "
            f"Time: {datetime.now().strftime('%H:%M:%S')}  |  Ctrl+C to stop"
        )
        lines.append(f"{Colors.BOLD}{'='*width}{Colors.RESET}")

        # Use lock to prevent display corruption from multiple async tasks
        try:
            print("\033[H\033[J" + "\n".join(lines), flush=True)
        except Exception:
            pass  # Ignore display errors during shutdown


async def run_single(coin: str, config: MomentumConfig, bot: TradingBot):
    """Run strategy for single coin."""
    config.coin = coin
    strategy = MomentumStrategy(bot, config)

    try:
        await strategy.run()
    except KeyboardInterrupt:
        pass
    finally:
        await strategy.stop()


async def run_all(config: MomentumConfig, bot: TradingBot):
    """Run strategy for all profitable coins with unified dashboard."""
    dashboard = MomentumMultiCoinDashboard(config)

    print("Initializing Quality_Combo momentum strategies for BTC, ETH...")
    print(f"Expected: +1.56%/trade, 60.8% win rate (combined)")

    strategies: List[MomentumStrategy] = []
    for coin in MomentumMultiCoinDashboard.COINS:
        coin_config = MomentumConfig(
            coin=coin,
            size=config.size,
            momentum_window=config.momentum_window,
            momentum_threshold=config.momentum_threshold,
            depth_ratio_threshold=config.depth_ratio_threshold,
            use_orderbook_filter=config.use_orderbook_filter,
            min_time_remaining=config.min_time_remaining,
            take_profit=config.take_profit,
            stop_loss=config.stop_loss,
            time_stop=config.time_stop,
            trade_cooldown=config.trade_cooldown,
            max_positions=config.max_positions,
            test_run=config.test_run,
            # Quality_Combo optimizations (inherited from config defaults)
            use_multi_timeframe=config.use_multi_timeframe,
            short_window=config.short_window,
            short_threshold=config.short_threshold,
            check_momentum_quality=config.check_momentum_quality,
            max_volatility=config.max_volatility,
            use_trailing_stop=config.use_trailing_stop,
        )
        strategy = MomentumStrategy(bot, coin_config)

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
                await strategy._maybe_redeem()  # Auto-redeem resolved positions
                await asyncio.sleep(0.5)
            except Exception as e:
                dashboard.log(f"[{strategy.config.coin}] {e}", "error")
                await asyncio.sleep(1)

    async def render_loop():
        while dashboard.running:
            async with dashboard._render_lock:
                dashboard.render()
            await asyncio.sleep(0.5)  # Slightly slower to reduce flicker

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
        description='Momentum Quality_Combo Strategy - BTC, ETH (optimized for +1.56%%/trade)'
    )
    parser.add_argument('--coin', type=str, required=True, help='BTC, ETH, or ALL')
    parser.add_argument('--size', type=float, default=5.0, help='Position size USD (default: 5.0)')
    parser.add_argument('--momentum', type=float, default=0.15, help='Momentum threshold (default: 0.15 = 15%%)')
    parser.add_argument('--window', type=int, default=30, help='Momentum window seconds (default: 30)')
    parser.add_argument('--depth-ratio', type=float, default=2.0, help='Depth ratio threshold (default: 2.0)')
    parser.add_argument('--no-orderbook', action='store_true', help='Disable orderbook confirmation')
    parser.add_argument('--tp', type=float, default=0.06, help='Take profit (default: 0.06 = 6%%)')
    parser.add_argument('--sl', type=float, default=0.03, help='Stop loss (default: 0.03 = 3%% - Quality_Combo)')
    parser.add_argument('--time-stop', type=int, default=80, help='Time stop seconds (default: 80)')
    parser.add_argument('--min-time', type=int, default=120, help='Min time remaining (default: 120)')
    parser.add_argument('--cooldown', type=int, default=15, help='Trade cooldown seconds (default: 15)')
    parser.add_argument('--test-run', action='store_true', help='Stop after one complete trade (entry + exit)')
    parser.add_argument('--log', action='store_true', help='Enable file logging to logs/ directory')

    args = parser.parse_args()

    # Validate coin selection
    valid_coins = ["BTC", "ETH", "ALL"]
    if args.coin.upper() not in valid_coins:
        print(f"ERROR: Invalid coin '{args.coin}'. Must be one of: {', '.join(valid_coins)}")
        print("Note: SOL (+0.52%%/trade - marginal), XRP (-1.48%%/trade) excluded")
        return

    # Setup logging to file if requested
    global logger
    logger = setup_logging(enable_file_log=args.log)

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

    config = MomentumConfig(
        coin=args.coin,
        size=args.size,
        momentum_threshold=args.momentum,
        momentum_window=args.window,
        depth_ratio_threshold=args.depth_ratio,
        use_orderbook_filter=not args.no_orderbook,
        take_profit=args.tp,
        stop_loss=args.sl,
        time_stop=args.time_stop,
        min_time_remaining=args.min_time,
        trade_cooldown=args.cooldown,
        test_run=args.test_run,
    )

    if args.coin.upper() == "ALL":
        await run_all(config, bot)
    else:
        print()
        print("=" * 70)
        print(f"MOMENTUM + ORDERBOOK STRATEGY - {args.coin.upper()}")
        print("=" * 70)
        print(f"Momentum: {args.momentum*100:.0f}% in {args.window}s  |  Depth Ratio: {args.depth_ratio}")
        print(f"Take Profit: +{args.tp*100:.0f}%  |  Stop Loss: -{args.sl*100:.0f}%")
        print(f"Time Stop: {args.time_stop}s  |  Min Time: {args.min_time}s")
        print(f"Size: ${args.size} per trade")
        print()
        print("Based on backtesting: BTC 52.4%, ETH 43.2%, SOL 37.8% win rate")
        print()
        print("WARNING: REAL trades with REAL money!")
        print("=" * 70)
        print()

        await run_single(args.coin.upper(), config, bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # Auto-diagnostics: dump error info to file
        import traceback
        from datetime import datetime
        error_file = f"logs/error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(error_file, 'w') as f:
            f.write(f"ERROR: {e}\n\n")
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
            # Add account state
            try:
                from dotenv import dotenv_values
                config = dotenv_values('.env')
                f.write(f"\nAccount: {config.get('POLY_SAFE_ADDRESS')}\n")
            except:
                pass
        print(f"\n Error logged to: {error_file}")
        raise
