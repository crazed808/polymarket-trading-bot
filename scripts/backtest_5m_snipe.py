#!/usr/bin/env python3
"""
Backtest: 5-Minute Micro Snipe Strategy

Uses existing 15-minute BTC tick data to determine optimal parameters
for the 5-minute micro sniper. The last N seconds of any market behave
similarly regardless of market duration — we have tick-level time_remaining
data for every market.

Usage:
    python scripts/backtest_5m_snipe.py
    python scripts/backtest_5m_snipe.py --data-dir data/ticks/btc
    python scripts/backtest_5m_snipe.py --verbose
"""

import json
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class Tick:
    """A single orderbook tick."""
    ts: str
    coin: str
    side: str  # "up" or "down"
    mid: float
    best_bid: float
    best_ask: float
    spread: float
    bid_depth: float
    ask_depth: float
    time_remaining: float
    market_slug: str
    bids: list = field(default_factory=list)
    asks: list = field(default_factory=list)


@dataclass
class SimTrade:
    """A simulated trade."""
    market_slug: str
    side: str
    entry_price: float
    entry_time_remaining: float
    outcome: str  # "win", "loss"
    pnl_pct: float  # percent P&L (e.g., 0.03 = +3%)
    ask_depth_at_entry: float
    bid_depth_at_entry: float


@dataclass
class ParamCombo:
    """A parameter combination to test."""
    min_time: float
    max_time: float
    min_price: float
    max_price: float
    min_liquidity: float
    min_orderbook_ratio: float = 1.5

    def label(self) -> str:
        return (f"t={self.min_time}-{self.max_time}s "
                f"p={self.min_price:.0%}-{self.max_price:.1%} "
                f"liq>={self.min_liquidity:.0f}")


def load_ticks(data_dir: str, max_time_remaining: float = 15.0) -> Dict[str, List[Tick]]:
    """
    Load tick data near settlement, grouped by market_slug.

    Only loads ticks with time_remaining <= max_time_remaining to avoid
    loading the full 15GB dataset into memory. We only need the final
    seconds for the micro snipe backtest.
    """
    markets: Dict[str, List[Tick]] = defaultdict(list)
    files = sorted(Path(data_dir).glob("*.jsonl"))

    if not files:
        print(f"ERROR: No .jsonl files found in {data_dir}")
        sys.exit(1)

    total_ticks = 0
    skipped_ticks = 0
    for i, f in enumerate(files):
        print(f"  Loading {f.name} ({i+1}/{len(files)})...", end="", flush=True)
        file_ticks = 0
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue

                # Fast pre-filter: check time_remaining before full parse
                # Look for "time_remaining": N pattern in the raw JSON
                tr_idx = line.find('"time_remaining":')
                if tr_idx != -1:
                    tr_start = tr_idx + 18  # len('"time_remaining": ')
                    # Skip whitespace
                    while tr_start < len(line) and line[tr_start] == ' ':
                        tr_start += 1
                    tr_end = tr_start
                    while tr_end < len(line) and line[tr_end] not in (',', '}', ' '):
                        tr_end += 1
                    try:
                        tr_val = float(line[tr_start:tr_end])
                        if tr_val > max_time_remaining:
                            skipped_ticks += 1
                            continue
                    except ValueError:
                        pass

                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                tr = d.get("time_remaining", 999)
                if tr > max_time_remaining:
                    skipped_ticks += 1
                    continue

                tick = Tick(
                    ts=d.get("ts", ""),
                    coin=d.get("coin", "BTC"),
                    side=d.get("side", ""),
                    mid=d.get("mid", 0.0),
                    best_bid=d.get("best_bid", 0.0),
                    best_ask=d.get("best_ask", 0.0),
                    spread=d.get("spread", 0.0),
                    bid_depth=d.get("bid_depth", 0.0),
                    ask_depth=d.get("ask_depth", 0.0),
                    time_remaining=d.get("time_remaining", 0.0),
                    market_slug=d.get("market_slug", ""),
                    bids=d.get("bids", []),
                    asks=d.get("asks", []),
                )
                markets[tick.market_slug].append(tick)
                total_ticks += 1
                file_ticks += 1
        print(f" {file_ticks:,} ticks")

    print(f"\nLoaded {total_ticks:,} ticks (skipped {skipped_ticks:,}) across {len(markets)} markets from {len(files)} files")
    return markets


def determine_outcome(ticks: List[Tick]) -> Optional[str]:
    """
    Determine market outcome from final ticks.
    Returns "up" or "down", or None if indeterminate.
    """
    # Get the last few ticks for each side
    final_up = [t for t in ticks if t.side == "up" and t.time_remaining <= 1.0]
    final_down = [t for t in ticks if t.side == "down" and t.time_remaining <= 1.0]

    # Also check ticks at time_remaining == 0 (settlement)
    settled_up = [t for t in ticks if t.side == "up" and t.time_remaining == 0]
    settled_down = [t for t in ticks if t.side == "down" and t.time_remaining == 0]

    # Prefer settled ticks
    if settled_up:
        up_final = settled_up[-1].mid
    elif final_up:
        up_final = final_up[-1].mid
    else:
        # Fall back to last tick for up side
        up_ticks = [t for t in ticks if t.side == "up"]
        if not up_ticks:
            return None
        up_final = up_ticks[-1].mid

    if settled_down:
        down_final = settled_down[-1].mid
    elif final_down:
        down_final = final_down[-1].mid
    else:
        down_ticks = [t for t in ticks if t.side == "down"]
        if not down_ticks:
            return None
        down_final = down_ticks[-1].mid

    if up_final >= 0.95:
        return "up"
    elif down_final >= 0.95:
        return "down"
    elif up_final > down_final:
        return "up"
    elif down_final > up_final:
        return "down"

    return None


def check_manipulation(ticks: List[Tick], side: str, entry_tick: Tick) -> bool:
    """
    Check if opposite side was dominant recently (manipulation check).
    Returns True if entry should be REJECTED.
    """
    opposite = "down" if side == "up" else "up"
    # Look at ticks from the last 5 seconds (by time_remaining)
    lookback_tr = entry_tick.time_remaining + 5.0

    opposite_recent = [
        t for t in ticks
        if t.side == opposite
        and entry_tick.time_remaining <= t.time_remaining <= lookback_tr
    ]

    if opposite_recent:
        max_opposite = max(t.mid for t in opposite_recent)
        if max_opposite >= 0.80:
            return True  # Reject — likely manipulation
    return False


def simulate_market(
    market_slug: str,
    ticks: List[Tick],
    outcome: str,
    params: ParamCombo,
    verbose: bool = False
) -> List[SimTrade]:
    """Simulate micro snipe entries for a single market."""
    trades = []
    sniped = False  # Only one entry per market

    # Separate ticks by side
    up_ticks = [t for t in ticks if t.side == "up"]
    down_ticks = [t for t in ticks if t.side == "down"]

    # Check each side
    for side, side_ticks in [("up", up_ticks), ("down", down_ticks)]:
        if sniped:
            break

        for tick in side_ticks:
            if sniped:
                break

            # Check time window
            if not (params.min_time <= tick.time_remaining <= params.max_time):
                continue

            # Check mid price range
            if not (params.min_price <= tick.mid <= params.max_price):
                continue

            # Check best_ask in range (this is what we'd actually pay)
            if tick.best_ask > params.max_price or tick.best_ask < params.min_price:
                continue

            # Check liquidity
            total_liq = tick.bid_depth + tick.ask_depth
            if total_liq < params.min_liquidity:
                continue

            # Check orderbook ratio
            if tick.ask_depth > 0:
                ratio = tick.bid_depth / tick.ask_depth
                if ratio < params.min_orderbook_ratio:
                    continue
            # If ask_depth is 0, ratio is infinite (good)

            # Manipulation check
            if check_manipulation(ticks, side, tick):
                if verbose:
                    print(f"  [{market_slug}] SKIP {side} tr={tick.time_remaining:.1f}s — manipulation detected")
                continue

            # Calculate available liquidity from asks within price range
            entry_price = tick.best_ask
            available_usd = 0.0
            for ask_price, ask_size in tick.asks:
                if params.min_price <= ask_price <= params.max_price:
                    available_usd += ask_size * ask_price

            if available_usd < 5.0:  # Minimum $5 order for 5m markets
                continue

            # We would enter here — determine outcome
            won = (side == outcome)
            if won:
                pnl_pct = 1.0 - entry_price  # e.g., buy at 0.97, win = 3%
            else:
                pnl_pct = -entry_price  # lose entire cost basis

            trade = SimTrade(
                market_slug=market_slug,
                side=side,
                entry_price=entry_price,
                entry_time_remaining=tick.time_remaining,
                outcome="win" if won else "loss",
                pnl_pct=pnl_pct,
                ask_depth_at_entry=tick.ask_depth,
                bid_depth_at_entry=tick.bid_depth,
            )
            trades.append(trade)
            sniped = True

            if verbose:
                marker = "WIN" if won else "LOSS"
                print(f"  [{market_slug}] {marker} {side} @ {entry_price:.3f} tr={tick.time_remaining:.1f}s "
                      f"pnl={pnl_pct:+.2%} liq=${available_usd:.0f}")

    return trades


def run_backtest(
    markets: Dict[str, List[Tick]],
    params: ParamCombo,
    verbose: bool = False
) -> Tuple[List[SimTrade], Dict]:
    """Run backtest with given parameters."""
    all_trades = []
    skipped_no_outcome = 0

    for slug, ticks in markets.items():
        outcome = determine_outcome(ticks)
        if outcome is None:
            skipped_no_outcome += 1
            continue

        trades = simulate_market(slug, ticks, outcome, params, verbose)
        all_trades.extend(trades)

    # Compute stats
    wins = [t for t in all_trades if t.outcome == "win"]
    losses = [t for t in all_trades if t.outcome == "loss"]

    win_rate = len(wins) / len(all_trades) if all_trades else 0.0
    avg_pnl = sum(t.pnl_pct for t in all_trades) / len(all_trades) if all_trades else 0.0
    total_pnl = sum(t.pnl_pct for t in all_trades)
    avg_entry = sum(t.entry_price for t in all_trades) / len(all_trades) if all_trades else 0.0

    stats = {
        "trades": len(all_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "avg_pnl_pct": avg_pnl,
        "total_pnl_pct": total_pnl,
        "avg_entry_price": avg_entry,
        "skipped_no_outcome": skipped_no_outcome,
        "total_markets": len(markets),
    }

    return all_trades, stats


def main():
    parser = argparse.ArgumentParser(description="Backtest 5-Minute Micro Snipe Strategy")
    parser.add_argument("--data-dir", type=str, default="data/ticks/btc",
                        help="Directory containing tick data JSONL files")
    parser.add_argument("--verbose", action="store_true", help="Show individual trades")
    args = parser.parse_args()

    # Resolve data dir relative to project root
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    print("=" * 80)
    print("  5-Minute Micro Snipe Backtest")
    print("  Using 15-minute tick data (last N seconds are market-duration agnostic)")
    print("=" * 80)
    print()

    markets = load_ticks(str(data_dir))

    # Parameter grid
    time_windows = [
        (1, 3),   # Current 15m default
        (1, 5),   # Wider window
        (2, 5),   # Skip first second
        (3, 7),   # Later window
        (5, 10),  # Early window
    ]
    min_prices = [0.90, 0.93, 0.95]
    max_prices = [0.99, 0.995]
    min_liquidities = [50, 100, 200]

    combos = []
    for min_t, max_t in time_windows:
        for min_p in min_prices:
            for max_p in max_prices:
                for min_l in min_liquidities:
                    combos.append(ParamCombo(
                        min_time=min_t,
                        max_time=max_t,
                        min_price=min_p,
                        max_price=max_p,
                        min_liquidity=min_l,
                    ))

    print(f"Testing {len(combos)} parameter combinations...\n")

    # Run all backtests
    results = []
    for combo in combos:
        trades, stats = run_backtest(markets, combo, verbose=args.verbose)
        results.append((combo, trades, stats))

    # Sort by total P&L descending
    results.sort(key=lambda r: r[2]["total_pnl_pct"], reverse=True)

    # Print results table
    print()
    print("=" * 120)
    print(f"{'Parameters':<42} {'Trades':>6} {'Wins':>5} {'Losses':>6} {'Win%':>7} {'Avg PnL':>9} {'Tot PnL':>9} {'Avg Entry':>10}")
    print("-" * 120)

    for combo, trades, stats in results:
        if stats["trades"] == 0:
            continue
        print(f"{combo.label():<42} "
              f"{stats['trades']:>6} "
              f"{stats['wins']:>5} "
              f"{stats['losses']:>6} "
              f"{stats['win_rate']:>6.1%} "
              f"{stats['avg_pnl_pct']:>+8.2%} "
              f"{stats['total_pnl_pct']:>+8.2%} "
              f"{stats['avg_entry_price']:>9.3f}")

    # Print top 5 recommendations
    print()
    print("=" * 80)
    print("  TOP 5 PARAMETER COMBOS (by total P&L)")
    print("=" * 80)

    top5 = [r for r in results if r[2]["trades"] >= 5][:5]
    for i, (combo, trades, stats) in enumerate(top5, 1):
        print(f"\n  #{i}: {combo.label()}")
        print(f"      Trades: {stats['trades']}, Win rate: {stats['win_rate']:.1%}, "
              f"Avg PnL: {stats['avg_pnl_pct']:+.2%}/trade, Total PnL: {stats['total_pnl_pct']:+.2%}")
        print(f"      Avg entry: {stats['avg_entry_price']:.3f}")

    # 5-minute frequency extrapolation
    if top5:
        best_combo, best_trades, best_stats = top5[0]
        num_15m_markets = best_stats["total_markets"] - best_stats["skipped_no_outcome"]
        trades_per_market = best_stats["trades"] / num_15m_markets if num_15m_markets > 0 else 0

        print()
        print("=" * 80)
        print("  5-MINUTE FREQUENCY EXTRAPOLATION")
        print("=" * 80)
        print()
        print(f"  Data: {best_stats['total_markets']} total markets over {len(list(Path(str(data_dir)).glob('*.jsonl')))} days")
        print(f"  15m trade rate: {trades_per_market:.3f} trades/market")
        print(f"  5m markets settle 3x more often than 15m markets")
        print(f"  Estimated 5m opportunities: ~{best_stats['trades'] * 3:.0f} trades over same period")
        print(f"  (Note: 5m markets have ~$1,100 liquidity vs ~$4,000+ for 15m — actual fill rates may differ)")

        # Print recommended config
        print()
        print("=" * 80)
        print("  RECOMMENDED CONFIG for run_micro_snipe_5m.py")
        print("=" * 80)
        print()
        print(f"  min_time: {best_combo.min_time}")
        print(f"  max_time: {best_combo.max_time}")
        print(f"  min_price: {best_combo.min_price}")
        print(f"  max_price: {best_combo.max_price}")
        print(f"  min_liquidity: {best_combo.min_liquidity}")

    print()


if __name__ == "__main__":
    main()
