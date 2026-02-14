#!/usr/bin/env python3
"""
Backtest Settlement Snipe Strategy

Simulates the snipe strategy on recorded tick data with and without
momentum/orderbook safety filters.

Usage:
    python scripts/backtest_snipe.py                    # All coins, 9 days
    python scripts/backtest_snipe.py --coin BTC,SOL     # Specific coins
    python scripts/backtest_snipe.py --days 3            # Last 3 days
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone

DATA_DIR = Path(__file__).parent.parent / "data" / "ticks"

# ── Config ──────────────────────────────────────────────────────────────

@dataclass
class SnipeBacktestConfig:
    min_price: float = 0.95
    max_price: float = 0.97
    min_time: int = 5
    max_time: int = 30
    min_liquidity: float = 100.0
    max_spread: float = 0.03
    size_usd: float = 5.0
    buy_slippage: float = 0.01  # 1% slippage on entry

    # Safety filters
    check_momentum: bool = True
    momentum_lookback_secs: float = 10.0  # Seconds to look back (matches live bot)
    max_negative_momentum: float = -0.02
    min_orderbook_ratio: float = 1.5

    label: str = ""


# ── Trade result ────────────────────────────────────────────────────────

@dataclass
class SnipeTrade:
    coin: str
    side: str
    market_slug: str
    entry_price: float
    entry_time: str
    time_remaining: float
    size_shares: float
    size_usd: float
    won: bool = False
    pnl: float = 0.0
    momentum: Optional[float] = None
    ob_ratio: Optional[float] = None
    skip_reason: str = ""


# ── Backtest engine ─────────────────────────────────────────────────────

class SnipeBacktester:
    def __init__(self, config: SnipeBacktestConfig):
        self.config = config
        self.trades: list[SnipeTrade] = []
        self.skipped: list[SnipeTrade] = []
        self.sniped_markets: set[str] = set()

        # Per-market state: track ticks to determine outcome
        # {market_slug: {"ticks": [...], "final_price_up": float}}
        self.market_data: dict = defaultdict(lambda: {
            "ticks_up": [],
            "ticks_down": [],
        })

        # Price history for momentum: {market_slug:side -> deque of (epoch_secs, price)}
        # ~7 ticks/sec * 10s lookback = ~70 ticks needed, use 200 for safety
        self.price_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))

    def process_file(self, file_path: Path, coin_filter: str = None):
        """Process a JSONL tick file."""
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    tick = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue

                coin = tick.get('coin', '')
                if coin_filter and coin.upper() != coin_filter.upper():
                    continue

                self._process_tick(tick)

    def _process_tick(self, tick: dict):
        coin = tick.get('coin', '')
        side = tick.get('side', '')
        market_slug = tick.get('market_slug', '')
        mid = tick.get('mid', 0)
        time_remaining = tick.get('time_remaining', 999)
        bid_depth = tick.get('bid_depth', 0) or 0
        ask_depth = tick.get('ask_depth', 0) or 0
        spread = tick.get('spread', 1) or 1
        ts = tick.get('ts', '')

        if not all([coin, side, market_slug, mid]):
            return

        mid = float(mid)
        time_remaining = float(time_remaining)
        bid_depth = float(bid_depth)
        ask_depth = float(ask_depth)
        spread = float(spread)

        # Parse timestamp to epoch seconds for time-based momentum
        try:
            epoch = datetime.fromisoformat(ts).timestamp()
        except (ValueError, TypeError):
            epoch = 0

        # Track price history for momentum
        history_key = f"{market_slug}:{side}"
        self.price_history[history_key].append((epoch, mid))

        # Store tick data for outcome determination
        if side == "up":
            self.market_data[market_slug]["ticks_up"].append((time_remaining, mid))
        else:
            self.market_data[market_slug]["ticks_down"].append((time_remaining, mid))

        # Skip if already sniped this market+side
        snipe_key = f"{market_slug}:{side}"
        if snipe_key in self.sniped_markets:
            return

        # Check time window
        if not (self.config.min_time <= time_remaining <= self.config.max_time):
            return

        # Check price range (both UP and DOWN use same range after fix)
        if not (self.config.min_price <= mid <= self.config.max_price):
            return

        # Check liquidity
        if (bid_depth + ask_depth) < self.config.min_liquidity:
            return

        # Check spread
        if spread > self.config.max_spread:
            return

        # Calculate momentum and orderbook ratio for logging
        momentum = self._calc_momentum(history_key, mid)
        ob_ratio = self._calc_ob_ratio(side, bid_depth, ask_depth)

        # Check safety filters
        if self.config.check_momentum:
            if momentum is not None and momentum < self.config.max_negative_momentum:
                self.skipped.append(SnipeTrade(
                    coin=coin, side=side, market_slug=market_slug,
                    entry_price=mid, entry_time=ts, time_remaining=time_remaining,
                    size_shares=0, size_usd=0,
                    momentum=momentum, ob_ratio=ob_ratio,
                    skip_reason=f"Falling price ({momentum:+.1%})"
                ))
                return

            if ob_ratio is not None and ob_ratio < self.config.min_orderbook_ratio:
                self.skipped.append(SnipeTrade(
                    coin=coin, side=side, market_slug=market_slug,
                    entry_price=mid, entry_time=ts, time_remaining=time_remaining,
                    size_shares=0, size_usd=0,
                    momentum=momentum, ob_ratio=ob_ratio,
                    skip_reason=f"Weak pressure ({ob_ratio:.2f})"
                ))
                return

        # Entry!
        entry_price = min(mid * (1 + self.config.buy_slippage), 0.99)
        shares = self.config.size_usd / entry_price
        self.sniped_markets.add(snipe_key)

        self.trades.append(SnipeTrade(
            coin=coin, side=side, market_slug=market_slug,
            entry_price=entry_price, entry_time=ts, time_remaining=time_remaining,
            size_shares=shares, size_usd=self.config.size_usd,
            momentum=momentum, ob_ratio=ob_ratio,
        ))

    def _calc_momentum(self, history_key: str, current_price: float) -> Optional[float]:
        history = self.price_history.get(history_key)
        if not history or len(history) < 3:
            return None

        current_time = history[-1][0]  # epoch seconds
        lookback_secs = self.config.momentum_lookback_secs

        # Find the oldest price within the lookback window
        oldest_price = None
        for t, p in history:
            if current_time - t <= lookback_secs:
                oldest_price = p
                break

        if oldest_price is None or oldest_price <= 0:
            return None

        return (current_price - oldest_price) / oldest_price

    def _calc_ob_ratio(self, side: str, bid_depth: float, ask_depth: float) -> Optional[float]:
        if not bid_depth or not ask_depth:
            return None
        # For both sides: bid/ask measures buy pressure for the token
        return bid_depth / ask_depth if ask_depth > 0 else 999

    def resolve_outcomes(self):
        """Determine win/loss for each trade based on final market prices.

        Uses BOTH sides to determine the true market winner:
        - If UP final >= 0.95 → UP won
        - If DOWN final >= 0.95 → DOWN won
        - Otherwise → unknown (exclude from results)
        """
        resolved = []
        self.unknown_outcomes = 0

        for trade in self.trades:
            market = self.market_data.get(trade.market_slug, {})

            # Get final prices for BOTH sides
            up_ticks = market.get("ticks_up", [])
            down_ticks = market.get("ticks_down", [])

            # Find lowest time_remaining for each side
            up_final = None
            if up_ticks:
                up_ticks.sort(key=lambda x: x[0])
                if up_ticks[0][0] <= 5:  # Must have data near settlement
                    up_final = up_ticks[0][1]

            down_final = None
            if down_ticks:
                down_ticks.sort(key=lambda x: x[0])
                if down_ticks[0][0] <= 5:
                    down_final = down_ticks[0][1]

            # Determine market winner
            winner = None
            if up_final is not None and up_final >= 0.95:
                winner = "up"
            elif down_final is not None and down_final >= 0.95:
                winner = "down"
            elif up_final is not None and up_final <= 0.05:
                winner = "down"
            elif down_final is not None and down_final <= 0.05:
                winner = "up"

            if winner is None:
                # Can't determine outcome — exclude
                self.unknown_outcomes += 1
                continue

            trade.won = (trade.side == winner)

            if trade.won:
                trade.pnl = (1.0 - trade.entry_price) * trade.size_shares
            else:
                trade.pnl = -trade.entry_price * trade.size_shares

            resolved.append(trade)

        self.trades = resolved

    def report(self) -> dict:
        """Generate summary report."""
        if not self.trades:
            return {"trades": 0}

        wins = [t for t in self.trades if t.won]
        losses = [t for t in self.trades if not t.won]
        total_pnl = sum(t.pnl for t in self.trades)
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        avg_pnl = total_pnl / len(self.trades) if self.trades else 0

        # Per-coin breakdown
        coins = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
        for t in self.trades:
            if t.won:
                coins[t.coin]["wins"] += 1
            else:
                coins[t.coin]["losses"] += 1
            coins[t.coin]["pnl"] += t.pnl

        return {
            "trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "skipped": len(self.skipped),
            "unknown": getattr(self, 'unknown_outcomes', 0),
            "coins": dict(coins),
        }


# ── Main ────────────────────────────────────────────────────────────────

def run_backtest(coins: list[str], days: int, config: SnipeBacktestConfig) -> dict:
    bt = SnipeBacktester(config)

    for coin in coins:
        coin_dir = DATA_DIR / coin.lower()
        if not coin_dir.exists():
            print(f"  No data for {coin}")
            continue

        files = sorted(coin_dir.glob("*.jsonl"))[-days:]
        for f in files:
            print(f"  {coin} {f.name}...", end=" ", flush=True)
            bt.process_file(f, coin_filter=coin)
            print(f"({len(bt.trades)} trades so far)")

    bt.resolve_outcomes()
    return bt.report(), bt


def print_report(label: str, report: dict, bt: SnipeBacktester):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if report["trades"] == 0:
        print("  No trades found.")
        return

    print(f"  Trades:     {report['trades']}")
    print(f"  Wins:       {report['wins']}")
    print(f"  Losses:     {report['losses']}")
    print(f"  Win Rate:   {report['win_rate']:.1f}%")
    print(f"  Total P&L:  ${report['total_pnl']:+.2f}")
    print(f"  Avg P&L:    ${report['avg_pnl']:+.4f}/trade")
    print(f"  Skipped:    {report['skipped']} (safety filter)")
    print(f"  Unknown:    {report.get('unknown', 0)} (no settlement data, excluded)")

    if report.get("coins"):
        print(f"\n  Per-coin breakdown:")
        for coin, data in sorted(report["coins"].items()):
            total = data["wins"] + data["losses"]
            wr = data["wins"] / total * 100 if total else 0
            print(f"    {coin:4s}: {data['wins']}W/{data['losses']}L ({wr:.0f}%) P&L: ${data['pnl']:+.2f}")

    # Show losses detail
    losses = [t for t in bt.trades if not t.won]
    if losses:
        print(f"\n  Loss details:")
        for t in losses[:15]:
            mom_str = f"mom={t.momentum:+.2%}" if t.momentum is not None else "mom=N/A"
            ob_str = f"ob={t.ob_ratio:.1f}" if t.ob_ratio is not None else "ob=N/A"
            print(f"    {t.coin:4s} {t.side:4s} @ {t.entry_price:.2%} | {t.time_remaining:.0f}s left | {mom_str} | {ob_str} | ${t.pnl:+.2f}")

    # Show skipped trades that would have lost
    if bt.skipped:
        # Check if any skipped trades would have been losses
        skip_slugs = {s.market_slug for s in bt.skipped}
        would_have_lost = 0
        would_have_won = 0
        for s in bt.skipped:
            market = bt.market_data.get(s.market_slug, {})
            ticks = market.get(f"ticks_{s.side}", [])
            if ticks:
                ticks_sorted = sorted(ticks, key=lambda x: x[0])
                final = ticks_sorted[0][1]
                if final >= 0.95:
                    would_have_won += 1
                else:
                    would_have_lost += 1
        print(f"\n  Safety filter analysis:")
        print(f"    Blocked:         {len(bt.skipped)} entries")
        print(f"    Would have won:  {would_have_won}")
        print(f"    Would have lost: {would_have_lost}")
        if would_have_lost > 0:
            saved = would_have_lost * bt.config.size_usd * 0.95  # approximate loss per trade
            print(f"    Estimated saved: ~${saved:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Backtest Settlement Snipe Strategy")
    parser.add_argument('--coin', type=str, default='BTC,ETH,SOL,XRP')
    parser.add_argument('--days', type=int, default=10)
    args = parser.parse_args()

    coins = [c.strip().upper() for c in args.coin.split(',')]

    print("="*60)
    print("  SETTLEMENT SNIPE BACKTEST — MOMENTUM THRESHOLD COMPARISON")
    print(f"  Coins: {', '.join(coins)} | Days: {args.days}")
    print(f"  Entry: 95-97% | Window: 5-30s | Size: $6")
    print(f"  Momentum: time-based (10s lookback)")
    print("="*60)

    results = []

    # ── Run 1: NO safety filters (baseline) ──
    print("\n\n[1/4] Running WITHOUT safety filters...")
    config_nosafety = SnipeBacktestConfig(
        check_momentum=False,
        size_usd=6.0,
        label="No Safety Filters"
    )
    report1, bt1 = run_backtest(coins, args.days, config_nosafety)
    print_report("WITHOUT Safety Filters (baseline)", report1, bt1)
    results.append(("No Safety", report1))

    # ── Run 2: Safety with -2% momentum threshold (current) ──
    print("\n\n[2/4] Running with -2% momentum threshold (current)...")
    config_2pct = SnipeBacktestConfig(
        check_momentum=True,
        max_negative_momentum=-0.02,
        size_usd=6.0,
        label="Safety -2% mom"
    )
    report2, bt2 = run_backtest(coins, args.days, config_2pct)
    print_report("Safety with -2% momentum (current)", report2, bt2)
    results.append(("Safety -2% mom", report2))

    # ── Run 3: Safety with -1% momentum threshold (tighter) ──
    print("\n\n[3/4] Running with -1% momentum threshold (tighter)...")
    config_1pct = SnipeBacktestConfig(
        check_momentum=True,
        max_negative_momentum=-0.01,
        size_usd=6.0,
        label="Safety -1% mom"
    )
    report3, bt3 = run_backtest(coins, args.days, config_1pct)
    print_report("Safety with -1% momentum (tighter)", report3, bt3)
    results.append(("Safety -1% mom", report3))

    # ── Run 4: Safety with -0.5% momentum threshold (strictest) ──
    print("\n\n[4/4] Running with -0.5% momentum threshold (strictest)...")
    config_half = SnipeBacktestConfig(
        check_momentum=True,
        max_negative_momentum=-0.005,
        size_usd=6.0,
        label="Safety -0.5% mom"
    )
    report4, bt4 = run_backtest(coins, args.days, config_half)
    print_report("Safety with -0.5% momentum (strictest)", report4, bt4)
    results.append(("Safety -0.5% mom", report4))

    # ── Comparison ──
    print("\n\n" + "="*60)
    print("  COMPARISON SUMMARY")
    print("="*60)
    for label, r in results:
        if r["trades"] > 0:
            print(f"  {label:22s}: {r['trades']:4d} trades | {r['win_rate']:5.1f}% WR | ${r['total_pnl']:+8.2f} P&L | ${r['avg_pnl']:+.4f}/trade | {r.get('unknown',0)} unknown")
        else:
            print(f"  {label:22s}: No trades")


if __name__ == "__main__":
    main()
