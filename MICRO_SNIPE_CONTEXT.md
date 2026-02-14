# Micro Snipe Bot - Development Context

This file tracks key decisions, edits, and known issues for the micro sniper bot (`run_micro_snipe.py`). Read this before making changes to understand the reasoning behind the current design.

## What the Bot Does

Buys outcome tokens at 95-99.5% probability in the final 1-3 seconds before 15-minute market settlement. High win rate, thin margins. Uses IOC (Immediate-Or-Cancel) orders to sweep available liquidity, sizing up to the full USDC balance.

## How We Run It

```bash
python run_micro_snipe.py --coin BTC,ETH,SOL,XRP &
```

- All 4 coins, default settings, backgrounded
- `--size` flag is just a fallback when orderbook is empty; the bot sizes to full available liquidity by default
- The recorder runs separately in a screen session:
  ```bash
  screen -dmS polymarket-recorder bash -c 'cd ~/polymarket-trading-bot && source venv/bin/activate && PYTHONPATH=. python3 continuous_recorder.py'
  ```

## Key Design Decisions

### 1. Full Balance Sizing (not fixed size)
The bot uses `min(available_liquidity, usdc_balance)` for order sizing. This maximizes profit on high-confidence opportunities rather than placing small fixed-size bets.

### 2. IOC Orders with Worst-Price Limit
Orders are submitted as IOC at `worst_price` (highest ask in the valid price range, typically 0.99). This sweeps all available liquidity up to that limit in a single order. The tradeoff is that fills can occur at any price below the limit.

### 3. Price Volatility Below min_price is Normal
In the final 1-5 seconds before settlement, mid prices frequently oscillate below the 95% min_price threshold before snapping to the correct outcome. This is normal market behavior and is where the opportunity comes from. **Do not add filters that reject entries just because the price was recently below min_price** — this was tested and would have blocked 2 out of 3 winning trades.

### 4. Manipulation Check (Opposite Side)
The bot checks if the opposite side's price was above 80% in the last 5 seconds. If so, the current side's high price is likely a momentary spike/trap and the entry is rejected.

### 5. Minimum 0.5% Profit Margin
If the worst fill price leaves less than 0.5% profit potential, the candidate is put on a 200ms cooldown to let better opportunities surface.

## Known Issues & Watch Items

### BAD FILL at 89% (Feb 12, 2026) - MONITORING
**Status: Watch — first and only occurrence**

The bot entered BTC UP when mid spiked from 89% to 96% in ~1 second. The IOC order filled at the actual market level of 88.66%, well below the 95% minimum. The `BAD FILL` detection logged the error but couldn't prevent it (order already filled).

**Why we didn't fix it:** A price stability check (requiring min price above 95% for 5 seconds) was prototyped but reverted because it would have blocked 2 out of 3 historical winning trades. The volatility is part of the opportunity.

**If this happens again:** Consider a narrower fix such as:
- Shorter lookback window (2s instead of 5s)
- Require a minimum number of consecutive ticks above min_price (e.g., 3 ticks)
- Cap the IOC limit price to `best_ask` instead of `worst_price` to reduce how far the order sweeps
- Post-fill validation that immediately sells if average fill is too far below min_price

### Balance Pre-Deduction
The bot pre-deducts the expected order cost from the cached balance to prevent multiple coins from double-spending the same funds. If an order fails or doesn't fill, the balance is restored. Watch for edge cases where balance gets out of sync.

### Per-Coin Minimums
XRP has a $5 minimum order size. BTC/ETH/SOL have $1 minimums. These are Polymarket requirements.

## 5-Minute Micro Sniper (`run_micro_snipe_5m.py`)

### Overview

Same micro snipe strategy applied to 5-minute BTC Up/Down markets (`btc-updown-5m-{timestamp}`). Settles 3x more often than 15-minute markets, giving more trading opportunities.

### How to Run

```bash
# Run the 5-minute sniper (BTC only)
python run_micro_snipe_5m.py --coin BTC --size 5.0

# With custom price range
python run_micro_snipe_5m.py --coin BTC --min-price 0.93 --max-price 0.99
```

### Run the Backtest First

```bash
python scripts/backtest_5m_snipe.py
python scripts/backtest_5m_snipe.py --verbose  # Show individual trades
```

The backtest uses existing 15-minute tick data (the last N seconds of any market behave similarly regardless of duration). It tests a grid of parameters and recommends optimal settings.

### Differences from 15-Minute Bot

| Setting | 15m Bot | 5m Bot | Why |
|---------|---------|--------|-----|
| Market discovery | `get_current_15m_market()` | `get_current_5m_market()` | 300s intervals instead of 900s |
| Coins | BTC, ETH, SOL, XRP | BTC only | Only BTC has 5m markets |
| Min liquidity | $100 | $50 | 5m markets have ~$1,100 total (vs ~$4,000+ for 15m) |
| Min order size | $1 (BTC/ETH/SOL), $5 (XRP) | $5 (BTC) | Polymarket `orderMinSize=5` for 5m |
| Log prefix | `logs/micro_*.log` | `logs/micro5m_*.log` | Distinguish log files |
| Dashboard title | "Micro Sniper" | "5m Micro Sniper" | Visual distinction |

### Known Limitations

- **BTC only**: Polymarket currently only offers 5-minute markets for BTC
- **Lower liquidity**: ~$1,100 total vs ~$4,000+ for 15-minute markets, so fills may be smaller
- **Tick size**: 0.01 (same as 15m)
- **No tick recording yet**: The continuous_recorder.py only records 15m markets. To record 5m ticks, the recorder would need to be extended.

## Edit History

| Date | Change | Reason |
|------|--------|--------|
| Feb 12, 2026 | Investigated BAD FILL at 89% | Price spike from 89%->96% caused fill at actual market (89%). Decided not to fix — first occurrence, and price stability filter would block winning trades. |
| Feb 12, 2026 | Prototyped + reverted price stability check | 5-second lookback requiring all prices above min_price. Reverted because it would reject 2/3 of historical winning trades. |
| Feb 12, 2026 | Added 5-minute micro sniper | New `run_micro_snipe_5m.py` for BTC 5-minute markets. Added `get_current_5m_market()` to GammaClient. Created `scripts/backtest_5m_snipe.py` for parameter optimization. |
