#!/usr/bin/env python3
"""
Analyze reversals in tick data to find counter-betting signals.

A "reversal" = leader (>90% at 30s remaining) ends up LOSING.
Tests price-drop signals at various thresholds and calculates EV.

Usage:
    python scripts/analyze_reversal_signals.py
    python scripts/analyze_reversal_signals.py --coin BTC,SOL --days 12
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data" / "ticks"


@dataclass
class MarketWindow:
    coin: str
    market_slug: str
    ticks: dict = field(default_factory=lambda: {"up": [], "down": []})

    def get_price_at(self, side: str, target_time: float, tolerance: float = 3.0) -> Optional[float]:
        candidates = [(t, mid) for t, mid, *_ in self.ticks.get(side, [])
                       if abs(t - target_time) <= tolerance]
        if not candidates:
            return None
        candidates.sort(key=lambda x: abs(x[0] - target_time))
        return candidates[0][1]

    def get_leader_at(self, target_time: float, min_prob: float = 0.90) -> Optional[tuple]:
        up_price = self.get_price_at("up", target_time)
        down_price = self.get_price_at("down", target_time)
        if up_price is not None and up_price >= min_prob:
            return ("up", up_price)
        if down_price is not None and down_price >= min_prob:
            return ("down", down_price)
        return None

    def get_winner(self) -> Optional[str]:
        up_final = self.get_price_at("up", 0, tolerance=5)
        down_final = self.get_price_at("down", 0, tolerance=5)
        if up_final is not None and up_final >= 0.95:
            return "up"
        if down_final is not None and down_final >= 0.95:
            return "down"
        if up_final is not None and up_final <= 0.05:
            return "down"
        if down_final is not None and down_final <= 0.05:
            return "up"
        return None

    def get_momentum(self, side: str, from_time: float, to_time: float) -> Optional[float]:
        p_from = self.get_price_at(side, from_time, tolerance=3)
        p_to = self.get_price_at(side, to_time, tolerance=3)
        if p_from is None or p_to is None or p_from <= 0:
            return None
        return (p_to - p_from) / p_from

    def get_volatility(self, side: str, from_time: float, to_time: float) -> Optional[float]:
        ticks_in_range = [mid for t, mid, *_ in self.ticks.get(side, [])
                          if from_time >= t >= to_time]
        if len(ticks_in_range) < 3:
            return None
        return (max(ticks_in_range) - min(ticks_in_range)) / min(ticks_in_range)


def load_markets(coins: list, days: int) -> list:
    markets = {}
    for coin in coins:
        coin_dir = DATA_DIR / coin.lower()
        if not coin_dir.exists():
            continue
        files = sorted(coin_dir.glob("*.jsonl"))[-days:]
        for f in files:
            with open(f, 'r') as fh:
                for line in fh:
                    try:
                        tick = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    slug = tick.get('market_slug', '')
                    side = tick.get('side', '')
                    mid = tick.get('mid', 0)
                    if not slug or not side or not mid:
                        continue
                    if slug not in markets:
                        markets[slug] = MarketWindow(coin=tick.get('coin', ''), market_slug=slug)
                    markets[slug].ticks[side].append((
                        float(tick.get('time_remaining', 999)), float(mid),
                        float(tick.get('bid_depth', 0) or 0),
                        float(tick.get('ask_depth', 0) or 0),
                        tick.get('ts', '')
                    ))
    return list(markets.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coin', type=str, default='BTC,ETH,SOL,XRP')
    parser.add_argument('--days', type=int, default=12)
    args = parser.parse_args()
    coins = [c.strip().upper() for c in args.coin.split(',')]

    print(f"Loading tick data for {', '.join(coins)} ({args.days} days)...")
    markets = load_markets(coins, args.days)
    print(f"Found {len(markets)} market windows\n")

    # Find markets with a clear leader at 30s
    leader_markets = []
    for m in markets:
        leader = m.get_leader_at(30, min_prob=0.90)
        winner = m.get_winner()
        if leader is None or winner is None:
            continue
        leader_side, leader_price = leader
        reversed_market = (leader_side != winner)
        leader_markets.append((m, leader_side, leader_price, winner, reversed_market))

    reversals = [x for x in leader_markets if x[4]]
    normals = [x for x in leader_markets if not x[4]]

    print(f"Markets with leader >90% at 30s: {len(leader_markets)}")
    print(f"  Normal (leader held): {len(normals)}")
    print(f"  Reversals (leader lost): {len(reversals)}")
    if leader_markets:
        print(f"  Reversal rate: {len(reversals)/len(leader_markets)*100:.1f}%")

    for min_p in [0.93, 0.95, 0.97]:
        subset = [x for x in leader_markets if x[2] >= min_p]
        rev_count = sum(1 for x in subset if x[4])
        if subset:
            print(f"  Leader >={min_p:.0%} at 30s: {len(subset)} markets, {rev_count} reversals ({rev_count/len(subset)*100:.1f}%)")

    # ── Reversal details ──
    print(f"\n{'='*80}")
    print("REVERSAL DETAILS")
    print(f"{'='*80}")
    for m, leader_side, leader_price, winner, _ in reversals:
        underdog_side = "down" if leader_side == "up" else "up"
        p60 = m.get_price_at(leader_side, 60)
        p45 = m.get_price_at(leader_side, 45)
        p30 = m.get_price_at(leader_side, 30)
        p15 = m.get_price_at(leader_side, 15)
        p5 = m.get_price_at(leader_side, 5)
        u30 = m.get_price_at(underdog_side, 30)
        u15 = m.get_price_at(underdog_side, 15)
        drop_30_15 = m.get_momentum(leader_side, 30, 15)

        prices = f"60s={p60 or 0:.1%} 45s={p45 or 0:.1%} 30s={p30 or 0:.1%} 15s={p15 or 0:.1%} 5s={p5 or 0:.1%}"
        print(f"\n  {m.coin:4s} {leader_side:4s} leader @ 30s={leader_price:.1%} | Winner: {winner}")
        print(f"    Leader prices:  {prices}")
        print(f"    Underdog: @30s={u30 or 0:.1%} @15s={u15 or 0:.1%}")
        if drop_30_15 is not None:
            print(f"    Drop 30s->15s: {drop_30_15:+.2%}")

    # ── Collect signals ──
    signals = []
    for m, leader_side, leader_price, winner, reversed_mkt in leader_markets:
        drop = m.get_momentum(leader_side, 30, 15)
        if drop is None:
            continue
        underdog_side = "down" if leader_side == "up" else "up"
        signals.append({
            "coin": m.coin,
            "slug": m.market_slug,
            "leader_side": leader_side,
            "leader_price_30": leader_price,
            "drop_30_15": drop,
            "reversed": reversed_mkt,
            "underdog_price_15": m.get_price_at(underdog_side, 15),
            "underdog_price_30": m.get_price_at(underdog_side, 30),
            "vol_60_30": m.get_volatility(leader_side, 60, 30),
        })

    if not signals:
        print("No signals found!")
        return

    total_reversals = sum(1 for s in signals if s["reversed"])

    # ── Distribution ──
    rev_drops = sorted([s["drop_30_15"] for s in signals if s["reversed"]])
    norm_drops = sorted([s["drop_30_15"] for s in signals if not s["reversed"]])

    print(f"\n{'='*80}")
    print("SIGNAL: Price drop 30s -> 15s (leader side)")
    print(f"{'='*80}")
    if rev_drops:
        print(f"\n  Reversals (n={len(rev_drops)}):")
        print(f"    Min={min(rev_drops):+.2%}  Median={rev_drops[len(rev_drops)//2]:+.2%}  Max={max(rev_drops):+.2%}")
        for s in signals:
            if s["reversed"]:
                u = s["underdog_price_15"] or 0
                print(f"    {s['coin']:4s} drop={s['drop_30_15']:+.2%} underdog@15s={u:.1%} underdog@30s={s['underdog_price_30'] or 0:.1%}")

    if norm_drops:
        print(f"\n  Normal (n={len(norm_drops)}):")
        print(f"    Min={min(norm_drops):+.2%}  Median={norm_drops[len(norm_drops)//2]:+.2%}  Max={max(norm_drops):+.2%}")

    # ── Threshold testing ──
    print(f"\n{'='*80}")
    print("THRESHOLD TESTING (buy underdog at 15s when leader drops)")
    print(f"{'='*80}")
    print(f"\n  {'Threshold':<18s} | {'Trig':>5s} | {'Rev':>4s} | {'FP':>4s} | {'Prec':>6s} | {'Recall':>7s} | {'AvgUdog':>8s} | {'EV/$1':>8s}")
    print(f"  {'-'*18}-+-{'-'*5}-+-{'-'*4}-+-{'-'*4}-+-{'-'*6}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")

    for threshold in [-0.01, -0.02, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.40, -0.50]:
        triggered = [s for s in signals if s["drop_30_15"] <= threshold]
        if not triggered:
            continue

        true_pos = [s for s in triggered if s["reversed"]]
        false_pos = [s for s in triggered if not s["reversed"]]
        precision = len(true_pos) / len(triggered)
        recall = len(true_pos) / total_reversals if total_reversals > 0 else 0

        # Average underdog price at 15s for ALL triggered (what we'd pay)
        udog_prices = [s["underdog_price_15"] for s in triggered if s["underdog_price_15"] is not None and s["underdog_price_15"] > 0]
        avg_udog = sum(udog_prices) / len(udog_prices) if udog_prices else 0.10

        # EV: with $1 bet, buy shares at avg_udog price
        # Win: each share pays $1, so profit = $1/avg_udog * (1-avg_udog) = (1-avg_udog)/avg_udog
        # But we spent $1, so net = (1/avg_udog) - 1
        # Loss: lose $1
        win_payout = (1.0 / avg_udog) - 1.0 if avg_udog > 0 else 0
        ev = precision * win_payout - (1 - precision) * 1.0

        label = f"Drop <= {threshold:+.0%}"
        print(f"  {label:<18s} | {len(triggered):>5d} | {len(true_pos):>4d} | {len(false_pos):>4d} | {precision:>5.1%} | {recall:>6.1%} | {avg_udog:>7.1%} | ${ev:>+7.2f}")

        # Show individual triggered markets
        if len(triggered) <= 25:
            for s in triggered:
                tag = "REV" if s["reversed"] else "held"
                u = s["underdog_price_15"] or 0
                print(f"      {tag:4s} {s['coin']:4s} drop={s['drop_30_15']:+.2%} udog@15s={u:.1%}")

    # ── Sniper loss cross-check ──
    print(f"\n{'='*80}")
    print("CROSS-CHECK: Sniper losses (entries at 95-97% that reversed)")
    print(f"{'='*80}")
    sniper_losses = []
    for m, leader_side, leader_price, winner, reversed_mkt in leader_markets:
        if not reversed_mkt:
            continue
        # Would sniper have entered? Check 95-97% range in 5-30s window
        for t_check in range(30, 4, -1):
            p = m.get_price_at(leader_side, t_check, tolerance=1)
            if p is not None and 0.95 <= p <= 0.97:
                drop = m.get_momentum(leader_side, 30, 15)
                underdog_side = "down" if leader_side == "up" else "up"
                u15 = m.get_price_at(underdog_side, 15)
                sniper_losses.append((m, leader_side, t_check, p, drop, u15))
                break

    print(f"\n  Reversals where sniper would have entered: {len(sniper_losses)}")
    for m, ls, entry_t, entry_p, drop, u15 in sniper_losses:
        drop_str = f"drop30→15={drop:+.2%}" if drop is not None else "drop=N/A"
        print(f"    {m.coin:4s} {ls:4s} entry@{entry_t}s={entry_p:.1%} | {drop_str}")
        if u15 and u15 > 0:
            counter_win = (1.0 / u15) - 1.0
            print(f"      → Counter-bet @15s: buy underdog at {u15:.1%} → win ${counter_win:.2f} per $1")
            if drop is not None:
                would_trigger_20 = drop <= -0.20
                would_trigger_10 = drop <= -0.10
                print(f"      → Would trigger at -20%? {'YES' if would_trigger_20 else 'NO'} | at -10%? {'YES' if would_trigger_10 else 'NO'}")


if __name__ == "__main__":
    main()
