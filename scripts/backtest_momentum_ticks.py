#!/usr/bin/env python3
"""
Backtest Momentum + Orderbook Strategy with recorded tick data.

Uses data from data/ticks/{coin}/*.jsonl format:
- Already has side (up/down)
- Has time_remaining for entry conditions
- Has market_slug for proper market segmentation
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
from pathlib import Path
from datetime import datetime


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
        """Price you pay to BUY (ask price)."""
        return self.best_ask if self.best_ask > 0 else self.mid_price

    @property
    def sell_price(self) -> float:
        """Price you receive to SELL (bid price)."""
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
    entry_spread: float = 0.0  # Spread at entry as decimal
    entry_depth: float = 0.0   # Total depth at entry


@dataclass
class MarketState:
    """State for a single 15-minute market."""
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


def load_tick_data(data_dir: str = "data/ticks") -> Dict[str, List[dict]]:
    """Load tick data from JSONL files."""
    all_ticks = {}

    for coin in ["btc", "eth", "sol", "xrp"]:
        coin_dir = Path(data_dir) / coin
        if not coin_dir.exists():
            continue

        ticks = []
        for jsonl_file in sorted(coin_dir.glob("*.jsonl")):
            with open(jsonl_file) as f:
                for line in f:
                    try:
                        tick = json.loads(line.strip())
                        ticks.append(tick)
                    except:
                        continue

        if ticks:
            # Sort by timestamp
            ticks.sort(key=lambda x: x.get("ts", ""))
            all_ticks[coin.upper()] = ticks

    return all_ticks


def backtest_coin(
    ticks: List[dict],
    coin: str,
    momentum_threshold: float = 0.15,
    momentum_window_ticks: int = 100,  # ~30 seconds at 300ms intervals
    depth_ratio_threshold: float = 1.5,
    take_profit_pct: float = 0.04,
    stop_loss_pct: float = 0.03,
    time_stop_ticks: int = 167,  # ~50 seconds
    cooldown_ticks: int = 50,  # ~15 seconds
    min_time_remaining: int = 120,  # 2 minutes
    use_orderbook: bool = True,
    max_spread: float = 1.0,  # Max bid/ask spread as decimal (1.0 = no filter)
    min_depth: float = 0.0,  # Minimum total orderbook depth (bid + ask)
) -> List[Trade]:
    """Backtest momentum strategy on a single coin."""
    trades = []

    # State per market
    markets: Dict[str, MarketState] = {}

    # Current position (one at a time)
    position: Optional[Position] = None
    last_trade_idx = -cooldown_ticks

    for i, tick in enumerate(ticks):
        side = tick.get("side", "")
        if side not in ["up", "down"]:
            continue

        market_slug = tick.get("market_slug", "unknown")
        mid_price = tick.get("mid", 0.5)
        time_remaining = tick.get("time_remaining", 0)
        bid_depth = tick.get("bid_depth", 0)
        ask_depth = tick.get("ask_depth", 0)

        # Get or create market state
        if market_slug not in markets:
            markets[market_slug] = MarketState(slug=market_slug)

        market = markets[market_slug]
        market.tick_count += 1

        # Get best bid/ask for realistic pricing
        best_bid = tick.get("best_bid", mid_price)
        best_ask = tick.get("best_ask", mid_price)

        # Update price history and orderbook
        market.price_history[side].append(PricePoint(ts=tick.get("ts", ""), price=mid_price))
        market.orderbook[side] = OrderbookState(mid_price, bid_depth, ask_depth, best_bid, best_ask)

        # Check position exit
        if position and position.market_slug == market_slug:
            # Use SELL price (best_bid) for realistic exit
            ob = market.orderbook[position.side]
            exit_price = ob.sell_price  # Realistic: you get bid price when selling
            exit_type = position.check_exit(exit_price)

            hold_ticks = i - position.entry_idx
            if hold_ticks >= time_stop_ticks:
                exit_type = "time"

            # Also exit if market is about to end
            if time_remaining <= 10 and time_remaining > 0:
                exit_type = "market_end"

            if exit_type:
                # Real P&L: what you receive (sell_price) vs what you paid (entry_price)
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
                last_trade_idx = i
                continue

        # Check for new entry (only if no position and cooldown passed)
        if position is None and (i - last_trade_idx) >= cooldown_ticks:
            # Check minimum time remaining
            if time_remaining < min_time_remaining:
                continue

            # Check momentum
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
                        # Determine trade side
                        if side == "up":
                            trade_side = "up" if momentum_dir == "up" else "down"
                        else:
                            trade_side = "down" if momentum_dir == "up" else "up"

                        # Orderbook confirmation
                        ob_confirmed = True
                        if use_orderbook:
                            ob = market.orderbook[trade_side]
                            ob_confirmed = ob.depth_ratio >= depth_ratio_threshold

                        # Spread filter: skip if spread is too wide
                        entry_ob = market.orderbook[trade_side]
                        spread = 0.0
                        if entry_ob.best_ask > 0 and entry_ob.best_bid > 0:
                            spread = (entry_ob.best_ask - entry_ob.best_bid) / entry_ob.mid_price
                        spread_ok = spread <= max_spread

                        # Depth filter: skip if orderbook too thin
                        total_depth = entry_ob.bid_depth + entry_ob.ask_depth
                        depth_ok = total_depth >= min_depth

                        if ob_confirmed and spread_ok and depth_ok:
                            # Use BUY price (best_ask) for realistic entry
                            entry_ob = market.orderbook[trade_side]
                            entry_price = entry_ob.buy_price  # Realistic: you pay ask price when buying

                            # Skip if price is at extreme (likely won't fill well)
                            if entry_price <= 0.02 or entry_price >= 0.98:
                                continue

                            # TP/SL as percentage of entry price
                            tp_price = entry_price * (1 + take_profit_pct)
                            sl_price = entry_price * (1 - stop_loss_pct)

                            position = Position(
                                side=trade_side,
                                entry_price=entry_price,
                                entry_ts=tick.get("ts", ""),
                                entry_idx=i,
                                take_profit_price=tp_price,
                                stop_loss_price=sl_price,
                                market_slug=market_slug,
                                entry_spread=spread,
                                entry_depth=total_depth,
                            )

    return trades


def analyze_results(trades: List[Trade]) -> dict:
    """Analyze backtest results with realistic bid/ask pricing (no additional spread needed)."""
    if not trades:
        return {"trades": 0}

    # Wins = positive P&L (already accounts for spread via bid/ask)
    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    by_exit = defaultdict(int)
    for t in trades:
        by_exit[t.exit_type] += 1

    total_pnl = sum(t.pnl_pct for t in trades)

    # Spread statistics
    spreads = [t.entry_spread for t in trades if t.entry_spread > 0]
    avg_spread = sum(spreads) / len(spreads) if spreads else 0

    # Depth statistics
    depths = [t.entry_depth for t in trades if t.entry_depth > 0]
    avg_depth = sum(depths) / len(depths) if depths else 0

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
        "avg_spread": avg_spread,
        "avg_depth": avg_depth,
    }


def main():
    print("=" * 70)
    print("MOMENTUM STRATEGY BACKTEST - NEW TICK DATA")
    print("=" * 70)
    print()

    print("Loading tick data from data/ticks/...")
    all_ticks = load_tick_data()

    total_ticks = sum(len(ticks) for ticks in all_ticks.values())
    print(f"Loaded {total_ticks:,} ticks across {len(all_ticks)} coins")
    for coin, ticks in all_ticks.items():
        print(f"  {coin}: {len(ticks):,} ticks")
    print()

    # Test multiple parameter sets
    param_sets = [
        # Current "optimized" params (with orderbook)
        {"name": "Baseline (15%/1.5)", "momentum_threshold": 0.15, "depth_ratio_threshold": 1.5, "use_orderbook": True},
        # With spread filter
        {"name": "Spread<1% filter", "momentum_threshold": 0.15, "use_orderbook": True, "max_spread": 0.01},
        # With spread filter + higher TP
        {"name": "Spread<1%, TP8/SL3", "momentum_threshold": 0.15, "take_profit_pct": 0.08, "stop_loss_pct": 0.03, "use_orderbook": True, "max_spread": 0.01},
        # With depth filter
        {"name": "Depth>500 filter", "momentum_threshold": 0.15, "use_orderbook": True, "min_depth": 500},
        # Combined: spread + depth + higher TP
        {"name": "Spread<1%+Depth>500+TP8", "momentum_threshold": 0.12, "take_profit_pct": 0.08, "stop_loss_pct": 0.03, "use_orderbook": True, "max_spread": 0.01, "min_depth": 500},
    ]

    base_params = {
        "momentum_window_ticks": 100,
        "take_profit_pct": 0.04,
        "stop_loss_pct": 0.03,
        "time_stop_ticks": 167,
        "cooldown_ticks": 50,
        "min_time_remaining": 120,
    }

    for params in param_sets:
        name = params.pop("name")
        test_params = {**base_params, **params}

        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print("=" * 70)

        all_trades = []

        for coin, ticks in all_ticks.items():
            trades = backtest_coin(ticks, coin, **test_params)
            all_trades.extend(trades)

            if trades:
                results = analyze_results(trades)
                avg_pct = results['avg_per_trade'] * 100
                status = "✓" if avg_pct > 0 else "✗"
                print(f"  {coin}: {results['trades']:4} trades, "
                      f"{results['win_rate']:5.1f}% win, "
                      f"avg {avg_pct:+6.2f}%/trade {status}")

        if all_trades:
            results = analyze_results(all_trades)
            avg_pct = results['avg_per_trade'] * 100
            status = "✓ PROFITABLE" if avg_pct > 0 else "✗ LOSING"

            print(f"\n  {'─'*50}")
            print(f"  TOTAL: {results['trades']:4} trades, "
                  f"{results['win_rate']:5.1f}% win rate, "
                  f"avg {avg_pct:+6.2f}%/trade {status}")

            print(f"  Exit breakdown: ", end="")
            for exit_type, count in sorted(results['by_exit'].items()):
                pct = count / results['trades'] * 100
                print(f"{exit_type}={count}({pct:.0f}%) ", end="")
            print()

            if avg_pct > 0:
                trades_per_day = results['trades'] / 3  # 3 days of data
                daily_profit = avg_pct * trades_per_day * 5 / 100  # $5 per trade
                print(f"  Projected: {trades_per_day:.0f} trades/day, ${daily_profit:.2f}/day ($5 size)")

        # Restore name for next iteration
        params["name"] = name

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
