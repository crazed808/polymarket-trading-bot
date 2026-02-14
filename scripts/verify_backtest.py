#!/usr/bin/env python3
"""
Manual verification of backtest results.
Simple, transparent calculation without condition filter complexity.
"""

import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data" / "ticks"

def simple_backtest(coin: str, days: int = 5):
    """Ultra-simple backtest - just raw numbers."""
    coin_dir = DATA_DIR / coin.lower()
    if not coin_dir.exists():
        return None

    files = sorted(coin_dir.glob("*.jsonl"), reverse=True)[:days]

    print(f"\n{'='*80}")
    print(f"VERIFYING {coin} BACKTEST")
    print(f"{'='*80}")
    print(f"Files: {len(files)}")

    markets = defaultdict(lambda: {
        'prices': [],
        'position': None,
        'trades': []
    })

    total_ticks = 0

    for file in files:
        print(f"\nProcessing {file.name}...")
        file_ticks = 0

        with open(file, 'r') as f:
            for line in f:
                try:
                    tick = json.loads(line)
                except:
                    continue

                if tick.get('side') != 'up':
                    continue

                file_ticks += 1
                total_ticks += 1

                market_slug = tick.get('market_slug', 'unknown')
                state = markets[market_slug]

                # Get timestamp
                ts = tick.get('timestamp', 0)
                if not ts:
                    ts_str = tick.get('ts', '')
                    if ts_str:
                        from datetime import datetime
                        try:
                            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp()
                        except:
                            continue

                price = float(tick.get('mid', 0) or 0)
                if not price or price <= 0:
                    continue

                bid_depth = float(tick.get('bid_depth', 0) or 0)
                ask_depth = float(tick.get('ask_depth', 0) or 0)

                # Track prices
                state['prices'].append((ts, price))
                # Keep last 30 seconds
                cutoff = ts - 30
                state['prices'] = [(t, p) for t, p in state['prices'] if t >= cutoff]

                # Entry logic
                if not state['position']:
                    if len(state['prices']) < 5:
                        continue

                    # Price filter
                    if price >= 0.85 or price <= 0.10:
                        continue

                    # Momentum check (15% in 30s)
                    old_price = state['prices'][0][1]
                    momentum = (price - old_price) / old_price if old_price > 0 else 0

                    if momentum < 0.15:
                        continue

                    # Orderbook check (2.0x ratio)
                    if ask_depth == 0 or bid_depth / ask_depth < 2.0:
                        continue

                    # Enter
                    state['position'] = {
                        'entry_price': price,
                        'entry_time': ts
                    }

                # Exit logic
                elif state['position']:
                    entry_price = state['position']['entry_price']
                    entry_time = state['position']['entry_time']
                    hold_time = ts - entry_time

                    pnl_pct = (price - entry_price) / entry_price

                    exit_type = None
                    if pnl_pct >= 0.06:  # 6% TP
                        exit_type = "tp"
                    elif pnl_pct <= -0.025:  # 2.5% SL
                        exit_type = "sl"
                    elif hold_time >= 80:  # 80s time stop
                        exit_type = "time"

                    if exit_type:
                        net_pnl_pct = pnl_pct - 0.02  # 2% spread cost

                        state['trades'].append({
                            'entry': entry_price,
                            'exit': price,
                            'pnl': pnl_pct,
                            'net_pnl': net_pnl_pct,
                            'exit_type': exit_type
                        })

                        state['position'] = None

        print(f"  Ticks processed: {file_ticks}")

    print(f"\nTotal ticks: {total_ticks}")
    print(f"Markets traded: {len(markets)}")

    # Aggregate trades
    all_trades = []
    for market_slug, state in markets.items():
        all_trades.extend(state['trades'])

    if not all_trades:
        print("\n❌ NO TRADES FOUND")
        return None

    wins = [t for t in all_trades if t['net_pnl'] > 0]
    losses = [t for t in all_trades if t['net_pnl'] <= 0]

    total_pnl = sum(t['net_pnl'] for t in all_trades)
    avg_pnl = total_pnl / len(all_trades)
    win_rate = len(wins) / len(all_trades) * 100

    tp_count = len([t for t in all_trades if t['exit_type'] == 'tp'])
    sl_count = len([t for t in all_trades if t['exit_type'] == 'sl'])
    time_count = len([t for t in all_trades if t['exit_type'] == 'time'])

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Total trades:     {len(all_trades)}")
    print(f"Wins:            {len(wins)} ({win_rate:.1f}%)")
    print(f"Losses:          {len(losses)}")
    print(f"")
    print(f"Avg P&L:         {avg_pnl:+.2%} per trade")
    print(f"Total P&L:       {total_pnl:+.2%}")
    print(f"")
    print(f"Exit breakdown:")
    print(f"  Take profit:   {tp_count}")
    print(f"  Stop loss:     {sl_count}")
    print(f"  Time stop:     {time_count}")

    if wins:
        avg_win = sum(t['net_pnl'] for t in wins) / len(wins)
        print(f"")
        print(f"Avg win:         {avg_win:+.2%}")

    if losses:
        avg_loss = sum(t['net_pnl'] for t in losses) / len(losses)
        print(f"Avg loss:        {avg_loss:.2%}")

    return {
        'coin': coin,
        'trades': len(all_trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'total_pnl': total_pnl
    }

def main():
    print("="*80)
    print("MANUAL BACKTEST VERIFICATION")
    print("="*80)
    print("\nConfig:")
    print("  Momentum: 15% in 30s")
    print("  Orderbook: 2.0x bid/ask ratio")
    print("  Entry: 10-85% price range")
    print("  TP: +6%, SL: -2.5%, Time: 80s")
    print("  Spread: 2% round-trip cost")

    results = {}

    for coin in ['BTC', 'ETH']:
        result = simple_backtest(coin, days=5)
        if result:
            results[coin] = result

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for coin, data in results.items():
        print(f"\n{coin}:")
        print(f"  Trades: {data['trades']}")
        print(f"  Win rate: {data['win_rate']:.1f}%")
        print(f"  Avg P&L: {data['avg_pnl']:+.2%}/trade")
        print(f"  Total P&L: {data['total_pnl']:+.2%}")

        if data['avg_pnl'] > 0:
            print(f"  ✅ PROFITABLE")
        else:
            print(f"  ❌ UNPROFITABLE")

    # Combined
    if len(results) == 2:
        total_trades = results['BTC']['trades'] + results['ETH']['trades']
        total_wins = results['BTC']['wins'] + results['ETH']['wins']
        combined_wr = total_wins / total_trades * 100
        combined_pnl = (results['BTC']['total_pnl'] + results['ETH']['total_pnl']) / total_trades

        print(f"\nCOMBINED (BTC + ETH):")
        print(f"  Total trades: {total_trades}")
        print(f"  Win rate: {combined_wr:.1f}%")
        print(f"  Avg P&L: {combined_pnl:+.2%}/trade")

if __name__ == '__main__':
    main()
