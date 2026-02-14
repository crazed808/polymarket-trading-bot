#!/usr/bin/env python3
"""
Quick momentum backtest - processes one coin at a time with streaming.
Memory efficient for large datasets.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque, defaultdict
from pathlib import Path


@dataclass
class PricePoint:
    ts: str
    price: float


@dataclass
class OrderbookState:
    mid_price: float
    bid_depth: float
    ask_depth: float
    best_bid: float = 0.0
    best_ask: float = 0.0

    @property
    def depth_ratio(self) -> float:
        if self.ask_depth > 0:
            return self.bid_depth / self.ask_depth
        return 1.0

    @property
    def buy_price(self) -> float:
        return self.best_ask if self.best_ask > 0 else self.mid_price

    @property
    def sell_price(self) -> float:
        return self.best_bid if self.best_bid > 0 else self.mid_price


@dataclass
class Position:
    side: str
    entry_price: float
    entry_ts: str
    entry_idx: int
    take_profit_price: float
    stop_loss_price: float
    market_slug: str
    entry_spread: float = 0.0
    entry_depth: float = 0.0

    def check_exit(self, current_price: float) -> Optional[str]:
        if current_price >= self.take_profit_price:
            return "tp"
        if current_price <= self.stop_loss_price:
            return "sl"
        return None


@dataclass
class Trade:
    coin: str
    side: str
    entry_price: float
    exit_price: float
    exit_type: str
    pnl_pct: float
    hold_ticks: int
    market_slug: str
    entry_spread: float = 0.0
    entry_depth: float = 0.0


@dataclass
class MarketState:
    slug: str
    price_history: Dict[str, deque] = field(default_factory=lambda: {
        "up": deque(maxlen=500),
        "down": deque(maxlen=500)
    })
    orderbook: Dict[str, OrderbookState] = field(default_factory=lambda: {
        "up": OrderbookState(0.5, 0, 0),
        "down": OrderbookState(0.5, 0, 0)
    })
    tick_count: int = 0


def backtest_coin_streaming(
    coin: str,
    data_dir: str = "data/ticks",
    momentum_threshold: float = 0.12,
    momentum_window_ticks: int = 100,
    depth_ratio_threshold: float = 1.5,
    take_profit_pct: float = 0.08,
    stop_loss_pct: float = 0.03,
    time_stop_ticks: int = 167,
    cooldown_ticks: int = 50,
    min_time_remaining: int = 120,
    use_orderbook: bool = True,
    max_spread: float = 0.03,
    min_depth: float = 500,
) -> List[Trade]:
    """Backtest by streaming through files - memory efficient."""
    trades = []
    markets: Dict[str, MarketState] = {}
    position: Optional[Position] = None
    last_trade_idx = -cooldown_ticks
    tick_count = 0

    coin_dir = Path(data_dir) / coin.lower()
    if not coin_dir.exists():
        print(f"  No data for {coin}")
        return []

    files = sorted(coin_dir.glob("*.jsonl"))
    print(f"  Processing {len(files)} files for {coin}...")

    for file_path in files:
        with open(file_path) as f:
            for line in f:
                try:
                    tick = json.loads(line.strip())
                except:
                    continue

                side = tick.get("side", "")
                if side not in ["up", "down"]:
                    continue

                market_slug = tick.get("market_slug", "unknown")
                mid_price = tick.get("mid", 0.5)
                time_remaining = tick.get("time_remaining", 0)
                bid_depth = tick.get("bid_depth", 0)
                ask_depth = tick.get("ask_depth", 0)

                if market_slug not in markets:
                    markets[market_slug] = MarketState(slug=market_slug)

                market = markets[market_slug]
                market.tick_count += 1

                best_bid = tick.get("best_bid", mid_price)
                best_ask = tick.get("best_ask", mid_price)

                market.price_history[side].append(PricePoint(ts=tick.get("ts", ""), price=mid_price))
                market.orderbook[side] = OrderbookState(mid_price, bid_depth, ask_depth, best_bid, best_ask)

                # Check position exit
                if position and position.market_slug == market_slug:
                    ob = market.orderbook[position.side]
                    exit_price = ob.sell_price
                    exit_type = position.check_exit(exit_price)

                    hold_ticks = tick_count - position.entry_idx
                    if hold_ticks >= time_stop_ticks:
                        exit_type = "time"

                    if time_remaining <= 10 and time_remaining > 0:
                        exit_type = "market_end"

                    if exit_type:
                        pnl_pct = (exit_price - position.entry_price) / position.entry_price

                        trades.append(Trade(
                            coin=coin,
                            side=position.side,
                            entry_price=position.entry_price,
                            exit_price=exit_price,
                            exit_type=exit_type,
                            pnl_pct=pnl_pct,
                            hold_ticks=hold_ticks,
                            market_slug=market_slug,
                            entry_spread=position.entry_spread,
                            entry_depth=position.entry_depth,
                        ))

                        position = None
                        last_trade_idx = tick_count
                        tick_count += 1
                        continue

                # Check for new entry
                if position is None and (tick_count - last_trade_idx) >= cooldown_ticks:
                    if time_remaining < min_time_remaining:
                        tick_count += 1
                        continue

                    history = market.price_history[side]
                    if len(history) >= momentum_window_ticks:
                        old_price = history[-momentum_window_ticks].price
                        new_price = mid_price

                        if old_price > 0:
                            price_change = (new_price - old_price) / old_price

                            momentum_dir = None
                            if price_change >= momentum_threshold:
                                momentum_dir = "up"
                            elif price_change <= -momentum_threshold:
                                momentum_dir = "down"

                            if momentum_dir:
                                if side == "up":
                                    trade_side = "up" if momentum_dir == "up" else "down"
                                else:
                                    trade_side = "down" if momentum_dir == "up" else "up"

                                ob_confirmed = True
                                if use_orderbook:
                                    ob = market.orderbook[trade_side]
                                    ob_confirmed = ob.depth_ratio >= depth_ratio_threshold

                                entry_ob = market.orderbook[trade_side]
                                spread = 0.0
                                if entry_ob.best_ask > 0 and entry_ob.best_bid > 0:
                                    spread = (entry_ob.best_ask - entry_ob.best_bid) / entry_ob.mid_price
                                spread_ok = spread <= max_spread

                                total_depth = entry_ob.bid_depth + entry_ob.ask_depth
                                depth_ok = total_depth >= min_depth

                                if ob_confirmed and spread_ok and depth_ok:
                                    entry_price = entry_ob.buy_price

                                    if entry_price <= 0.02 or entry_price >= 0.98:
                                        tick_count += 1
                                        continue

                                    tp_price = entry_price * (1 + take_profit_pct)
                                    sl_price = entry_price * (1 - stop_loss_pct)

                                    position = Position(
                                        side=trade_side,
                                        entry_price=entry_price,
                                        entry_ts=tick.get("ts", ""),
                                        entry_idx=tick_count,
                                        take_profit_price=tp_price,
                                        stop_loss_price=sl_price,
                                        market_slug=market_slug,
                                        entry_spread=spread,
                                        entry_depth=total_depth,
                                    )

                tick_count += 1

                if tick_count % 500000 == 0:
                    print(f"    {tick_count:,} ticks, {len(trades)} trades")

    print(f"  Total: {tick_count:,} ticks processed")
    return trades


def analyze_results(trades: List[Trade]) -> dict:
    if not trades:
        return {"trades": 0}

    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    by_exit = defaultdict(int)
    for t in trades:
        by_exit[t.exit_type] += 1

    total_pnl = sum(t.pnl_pct for t in trades)

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100,
        "avg_win": sum(t.pnl_pct for t in wins) / len(wins) if wins else 0,
        "avg_loss": sum(t.pnl_pct for t in losses) / len(losses) if losses else 0,
        "total_pnl": total_pnl,
        "avg_per_trade": total_pnl / len(trades),
        "by_exit": dict(by_exit),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Quick momentum backtest')
    parser.add_argument('--coin', type=str, default='all',
                        help='Coin to backtest (btc/eth/sol/xrp/all)')
    parser.add_argument('--momentum', type=float, default=0.12,
                        help='Momentum threshold (default: 0.12)')
    parser.add_argument('--tp', type=float, default=0.10,
                        help='Take profit %% (default: 0.10)')
    parser.add_argument('--sl', type=float, default=0.03,
                        help='Stop loss %% (default: 0.03)')
    parser.add_argument('--max-spread', type=float, default=0.03,
                        help='Max entry spread (default: 0.03)')
    parser.add_argument('--min-depth', type=float, default=500,
                        help='Min orderbook depth (default: 500)')
    args = parser.parse_args()

    print("=" * 70)
    print("MOMENTUM STRATEGY BACKTEST (Streaming)")
    print("=" * 70)
    print(f"Parameters: momentum={args.momentum}, tp={args.tp}, sl={args.sl}")
    print(f"Filters: max_spread={args.max_spread}, min_depth={args.min_depth}")
    print()

    coins = ['xrp', 'sol', 'eth', 'btc'] if args.coin == 'all' else [args.coin.lower()]
    all_trades = []

    for coin in coins:
        print(f"\n{coin.upper()}:")
        trades = backtest_coin_streaming(
            coin=coin,
            momentum_threshold=args.momentum,
            take_profit_pct=args.tp,
            stop_loss_pct=args.sl,
            max_spread=args.max_spread,
            min_depth=args.min_depth,
        )
        all_trades.extend(trades)

        if trades:
            results = analyze_results(trades)
            avg_pct = results['avg_per_trade'] * 100
            status = "✓" if avg_pct > 0 else "✗"
            print(f"  Results: {results['trades']:4} trades, "
                  f"{results['win_rate']:5.1f}% win, "
                  f"avg {avg_pct:+6.2f}%/trade {status}")
            print(f"  Exits: ", end="")
            for exit_type, count in sorted(results['by_exit'].items()):
                pct = count / results['trades'] * 100
                print(f"{exit_type}={count}({pct:.0f}%) ", end="")
            print()

    if all_trades and len(coins) > 1:
        print("\n" + "=" * 70)
        print("COMBINED RESULTS")
        print("=" * 70)
        results = analyze_results(all_trades)
        avg_pct = results['avg_per_trade'] * 100
        status = "✓ PROFITABLE" if avg_pct > 0 else "✗ LOSING"

        print(f"Total: {results['trades']:4} trades, "
              f"{results['win_rate']:5.1f}% win rate, "
              f"avg {avg_pct:+6.2f}%/trade {status}")

        if avg_pct > 0:
            days_of_data = 5  # approximate
            trades_per_day = results['trades'] / days_of_data
            daily_profit_pct = avg_pct * trades_per_day
            print(f"Projected: ~{trades_per_day:.0f} trades/day, ~{daily_profit_pct:.1f}% daily return")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
