#!/usr/bin/env python3
"""
Backtest Momentum + Orderbook Strategy with CORRECTED TP/SL logic.

Properly segments data by market (each 15-min window has unique asset IDs).
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import deque, defaultdict
from pathlib import Path
from datetime import datetime


@dataclass
class PricePoint:
    tick_idx: int
    price: float


@dataclass
class OrderbookState:
    mid_price: float
    bid_depth: float
    ask_depth: float

    @property
    def depth_ratio(self) -> float:
        if self.ask_depth > 0:
            return self.bid_depth / self.ask_depth
        return 1.0


@dataclass
class Position:
    side: str
    entry_price: float
    entry_tick: int
    take_profit_price: float
    stop_loss_price: float

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


@dataclass
class MarketData:
    """Data for a single 15-minute market."""
    up_asset: str
    down_asset: str
    ticks: List[dict] = field(default_factory=list)


def load_and_segment_data(data_dir: str = "data/recordings") -> Dict[str, List[MarketData]]:
    """Load tick data and segment by market."""
    all_markets = {}

    for coin in ["BTC", "ETH", "SOL", "XRP"]:
        coin_dir = Path(data_dir) / coin.lower()
        if not coin_dir.exists():
            continue

        markets = []

        # Each JSONL file is typically one market
        for jsonl_file in sorted(coin_dir.glob("*.jsonl")):
            ticks = []
            asset_ids = set()

            with open(jsonl_file) as f:
                for line in f:
                    try:
                        tick = json.loads(line.strip())
                        ticks.append(tick)
                        asset_ids.add(tick.get("asset_id", ""))
                    except:
                        continue

            if len(ticks) < 100 or len(asset_ids) < 2:
                continue

            # Identify UP vs DOWN by checking which has higher mid price
            asset_prices = defaultdict(list)
            for tick in ticks[:100]:
                aid = tick.get("asset_id", "")
                mid = tick.get("mid", 0.5)
                asset_prices[aid].append(mid)

            # Sort by average price - higher is UP
            asset_avgs = [(aid, sum(prices)/len(prices)) for aid, prices in asset_prices.items() if prices]
            asset_avgs.sort(key=lambda x: -x[1])

            if len(asset_avgs) >= 2:
                up_asset = asset_avgs[0][0]
                down_asset = asset_avgs[1][0]

                markets.append(MarketData(
                    up_asset=up_asset,
                    down_asset=down_asset,
                    ticks=ticks
                ))

        if markets:
            all_markets[coin] = markets

    return all_markets


def backtest_market(
    market: MarketData,
    coin: str,
    momentum_threshold: float = 0.15,
    momentum_window_ticks: int = 100,
    depth_ratio_threshold: float = 1.5,
    take_profit_pct: float = 0.04,
    stop_loss_pct: float = 0.03,
    time_stop_ticks: int = 167,
    cooldown_ticks: int = 50,
    use_orderbook: bool = True,
) -> List[Trade]:
    """Backtest momentum strategy on a single market."""
    trades = []

    # Price history per side
    price_history: Dict[str, deque] = {
        "up": deque(maxlen=momentum_window_ticks * 2),
        "down": deque(maxlen=momentum_window_ticks * 2)
    }

    # Current orderbook state
    orderbook: Dict[str, OrderbookState] = {
        "up": OrderbookState(0.5, 0, 0),
        "down": OrderbookState(0.5, 0, 0)
    }

    # Trading state
    position: Optional[Position] = None
    last_trade_tick = -cooldown_ticks

    for i, tick in enumerate(market.ticks):
        asset_id = tick.get("asset_id", "")
        mid_price = tick.get("mid", 0.5)

        # Determine side
        if asset_id == market.up_asset:
            side = "up"
        elif asset_id == market.down_asset:
            side = "down"
        else:
            continue

        # Calculate depth
        bids = tick.get("bids", [])
        asks = tick.get("asks", [])
        total_bid = sum(float(b[1]) for b in bids if isinstance(b, list) and len(b) >= 2)
        total_ask = sum(float(a[1]) for a in asks if isinstance(a, list) and len(a) >= 2)

        # Update state
        price_history[side].append(PricePoint(tick_idx=i, price=mid_price))
        orderbook[side] = OrderbookState(mid_price, total_bid, total_ask)

        # Check position exit
        if position:
            current_price = orderbook[position.side].mid_price
            exit_type = position.check_exit(current_price)

            hold_ticks = i - position.entry_tick
            if hold_ticks >= time_stop_ticks:
                exit_type = "time"

            if exit_type:
                pnl_pct = (current_price - position.entry_price) / position.entry_price

                trades.append(Trade(
                    coin=coin,
                    side=position.side,
                    entry_price=position.entry_price,
                    exit_price=current_price,
                    exit_type=exit_type,
                    pnl_pct=pnl_pct,
                    hold_ticks=hold_ticks
                ))

                position = None
                last_trade_tick = i
                continue

        # Check for new entry
        if position is None and (i - last_trade_tick) >= cooldown_ticks:
            if len(price_history[side]) >= momentum_window_ticks:
                old_price = price_history[side][-momentum_window_ticks].price
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
                            ob = orderbook[trade_side]
                            ob_confirmed = ob.depth_ratio >= depth_ratio_threshold

                        if ob_confirmed:
                            entry_price = orderbook[trade_side].mid_price

                            # CORRECT: TP/SL as percentage of entry
                            tp_price = entry_price * (1 + take_profit_pct)
                            sl_price = entry_price * (1 - stop_loss_pct)

                            position = Position(
                                side=trade_side,
                                entry_price=entry_price,
                                entry_tick=i,
                                take_profit_price=tp_price,
                                stop_loss_price=sl_price
                            )

    return trades


def analyze_results(trades: List[Trade], spread_cost: float = 0.01) -> dict:
    """Analyze backtest results."""
    if not trades:
        return {"trades": 0}

    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    net_pnls = [t.pnl_pct - spread_cost for t in trades]

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100,
        "avg_win": sum(t.pnl_pct for t in wins) / len(wins) if wins else 0,
        "avg_loss": sum(t.pnl_pct for t in losses) / len(losses) if losses else 0,
        "gross_pnl": sum(t.pnl_pct for t in trades),
        "net_pnl": sum(net_pnls),
        "net_per_trade": sum(net_pnls) / len(trades),
        "by_exit": {
            "tp": len([t for t in trades if t.exit_type == "tp"]),
            "sl": len([t for t in trades if t.exit_type == "sl"]),
            "time": len([t for t in trades if t.exit_type == "time"]),
        }
    }


def main():
    print("=" * 70)
    print("MOMENTUM STRATEGY BACKTEST - CORRECTED TP/SL LOGIC")
    print("=" * 70)
    print()

    print("Loading and segmenting tick data by market...")
    all_markets = load_and_segment_data()

    total_markets = sum(len(markets) for markets in all_markets.values())
    total_ticks = sum(sum(len(m.ticks) for m in markets) for markets in all_markets.values())
    print(f"Loaded {total_ticks:,} ticks across {total_markets} markets ({len(all_markets)} coins)")
    print()

    # Test multiple parameter sets
    param_sets = [
        # Original "optimized" params
        {"name": "Optimized (15%/1.5)", "momentum_threshold": 0.15, "depth_ratio_threshold": 1.5},
        # More relaxed params
        {"name": "Relaxed (10%/1.5)", "momentum_threshold": 0.10, "depth_ratio_threshold": 1.5},
        # Without orderbook filter
        {"name": "No OB (10%)", "momentum_threshold": 0.10, "use_orderbook": False},
    ]

    base_params = {
        "momentum_window_ticks": 100,
        "take_profit_pct": 0.04,
        "stop_loss_pct": 0.03,
        "time_stop_ticks": 167,
        "cooldown_ticks": 50,
    }

    for params in param_sets:
        name = params.pop("name")
        test_params = {**base_params, **params}

        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print("=" * 70)

        all_trades = []
        coin_results = {}

        for coin, markets in all_markets.items():
            coin_trades = []
            for market in markets:
                trades = backtest_market(market, coin, **test_params)
                coin_trades.extend(trades)
            all_trades.extend(coin_trades)

            if coin_trades:
                results = analyze_results(coin_trades)
                coin_results[coin] = results
                net_pct = results['net_per_trade'] * 100
                status = "✓" if net_pct > 0 else "✗"
                print(f"  {coin}: {results['trades']:3} trades, "
                      f"{results['win_rate']:.1f}% win, "
                      f"net {net_pct:+.2f}%/trade {status}")

        if all_trades:
            results = analyze_results(all_trades)
            print(f"\n  TOTAL: {results['trades']} trades, "
                  f"{results['win_rate']:.1f}% win, "
                  f"net {results['net_per_trade']*100:+.2f}%/trade")
            print(f"  Exit types: TP={results['by_exit']['tp']}, "
                  f"SL={results['by_exit']['sl']}, Time={results['by_exit']['time']}")

            if results['net_per_trade'] > 0:
                profit_5 = results['net_per_trade'] * 5 * results['trades']
                print(f"  Simulated profit ($5 trades): ${profit_5:.2f}")

        # Restore name for next iteration
        params["name"] = name

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
