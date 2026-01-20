#!/usr/bin/env python3
"""
Longshot Strategy: Bet on extreme underdogs (99-1 odds or higher)
Theory: Market overreacts near end, small chance of reversal
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
from dataclasses import dataclass, field
from collections import deque

@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    coin: str
    trades: List[Dict] = field(default_factory=list)
    total_pnl: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

class MarketState:
    """Market state with timing info"""
    def __init__(self, timestamp, up_price, down_price):
        self.timestamp = timestamp
        self.up_price = up_price
        self.down_price = down_price

class LongshotBacktester:
    """Bet on extreme underdogs"""
    
    def __init__(self,
                 underdog_threshold: float = 0.01,  # 1% = 99-1 odds
                 min_time_remaining: int = 120,      # 2 minutes in seconds
                 position_size: float = 1.0,
                 fee_rate: float = 0.02,
                 slippage: float = 0.01):
        self.underdog_threshold = underdog_threshold
        self.min_time_remaining = min_time_remaining
        self.position_size = position_size
        self.fee_rate = fee_rate
        self.slippage = slippage
        
        self.position = None
        self.trades = []
        self.balance = 0.0
        self.market_end_time = None
        
    def should_enter(self, market: MarketState) -> Dict:
        """Check if we should bet on an underdog"""
        if self.position:
            return None  # Already in a position
        
        # Check if we have market end time (set when we see first data)
        if not self.market_end_time:
            # Assume 15-min markets end at next 15-min mark
            current = market.timestamp
            # Round down to current minute
            base = current.replace(second=0, microsecond=0)
            # Calculate minutes until next 15-min boundary
            minutes_into_period = base.minute % 15
            minutes_to_add = 15 - minutes_into_period if minutes_into_period > 0 else 15
            # Handle edge case: if we're exactly at a boundary with no seconds, stay there
            if minutes_into_period == 0 and current.second == 0 and current.microsecond == 0:
                minutes_to_add = 0
            self.market_end_time = base + timedelta(minutes=minutes_to_add)
        
        # Calculate time remaining
        time_remaining = (self.market_end_time - market.timestamp).total_seconds()
        
        # Not enough time left
        if time_remaining < self.min_time_remaining:
            return None
        
        # Check if UP is underdog
        if market.up_price <= self.underdog_threshold:
            return {
                'side': 'UP',
                'entry_price': market.up_price,
                'time_remaining': time_remaining,
                'odds': f"{(1-market.up_price)*100:.0f}-{market.up_price*100:.0f}"
            }
        
        # Check if DOWN is underdog
        if market.down_price <= self.underdog_threshold:
            return {
                'side': 'DOWN',
                'entry_price': market.down_price,
                'time_remaining': time_remaining,
                'odds': f"{(1-market.down_price)*100:.0f}-{market.down_price*100:.0f}"
            }
        
        return None
    
    def check_exit(self, market: MarketState) -> Dict:
        """Check if position should be closed"""
        if not self.position:
            return None
        
        current_price = market.up_price if self.position['side'] == 'UP' else market.down_price
        entry_price = self.position['entry_price']
        
        # Check if market ended (time's up)
        time_remaining = (self.market_end_time - market.timestamp).total_seconds()
        if time_remaining <= 0:
            # Market ended - final settlement
            return self.close_position(current_price, market.timestamp, 'MARKET_END')
        
        # Optional: Take profit if price recovers significantly
        # (e.g., if 1% becomes 20%, take profit)
        if current_price >= 0.20:
            return self.close_position(current_price, market.timestamp, 'TAKE_PROFIT')
        
        return None
    
    def close_position(self, exit_price: float, timestamp: datetime, reason: str) -> Dict:
        """Close position and calculate PnL"""
        if not self.position:
            return None
        
        entry_price = self.position['entry_price']
        
        # Add slippage
        exit_price_with_slippage = exit_price * (1 - self.slippage)
        
        # Calculate PnL
        # If we bought at 0.01 and it wins (pays 1.0), we make 0.99
        # If we bought at 0.01 and it loses (pays 0.0), we lose 0.01
        if reason == 'MARKET_END':
            # Simplified: assume current price is settlement price
            # In reality, it settles to 1.0 (winner) or 0.0 (loser)
            # We'll approximate: if price > 0.5, assume it won (1.0), else lost (0.0)
            if exit_price >= 0.5:
                settlement_price = 1.0
            else:
                settlement_price = 0.0
            gross_pnl = settlement_price - entry_price
        else:
            gross_pnl = exit_price_with_slippage - entry_price
        
        # Subtract fees
        fees = entry_price * self.fee_rate + exit_price_with_slippage * self.fee_rate
        net_pnl = (gross_pnl - fees) * self.position_size
        
        trade = {
            'entry_time': self.position['entry_time'],
            'entry_price': self.position['entry_price'],
            'side': self.position['side'],
            'entry_odds': self.position['odds'],
            'exit_time': timestamp,
            'exit_price': exit_price,
            'pnl': net_pnl,
            'reason': reason,
            'hold_time': (timestamp - self.position['entry_time']).total_seconds()
        }
        
        self.trades.append(trade)
        self.balance += net_pnl
        self.position = None
        
        return trade
    
    def process_market_state(self, market: MarketState):
        """Process market state"""
        # Check for exit
        if self.position:
            exit_event = self.check_exit(market)
            if exit_event:
                return exit_event
        
        # Check for entry
        if not self.position:
            signal = self.should_enter(market)
            if signal:
                entry_price_with_slippage = signal['entry_price'] * (1 + self.slippage)
                
                self.position = {
                    'entry_time': market.timestamp,
                    'entry_price': entry_price_with_slippage,
                    'side': signal['side'],
                    'odds': signal['odds'],
                    'size': self.position_size
                }
                return {'type': 'OPEN', 'signal': signal}
        
        return None

    def run_backtest(self, recording_files: List[Path]) -> BacktestResult:
        """Run backtest on recorded data"""
        print(f"Processing {len(recording_files)} files...")

        asset_prices = {}
        market_states = []
        current_asset_ids = set()

        for file_path in sorted(recording_files):
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        ts = datetime.fromisoformat(data['ts'])
                        asset_id = data['asset_id']
                        mid = data['mid']

                        # Detect market change: if we have 2 asset_ids and see a new one
                        if len(current_asset_ids) == 2 and asset_id not in current_asset_ids:
                            # New market detected - reset state
                            asset_prices = {}
                            current_asset_ids = set()
                            # Mark that we need to reset market end time
                            market_states.append(('MARKET_RESET', ts))

                        current_asset_ids.add(asset_id)
                        asset_prices[asset_id] = {'price': mid, 'ts': ts}

                        if len(asset_prices) == 2:
                            prices = list(asset_prices.values())
                            p1, p2 = prices[0]['price'], prices[1]['price']

                            up, down = (p1, p2) if p1 > p2 else (p2, p1)
                            market = MarketState(ts, up, down)
                            market_states.append(market)

                    except (json.JSONDecodeError, KeyError):
                        continue

        print(f"Created {len([m for m in market_states if isinstance(m, MarketState)])} market states")
        print(f"Processing strategy...")

        # Reset market end time for each new market
        last_market_time = None

        for i, market in enumerate(market_states):
            # Handle market reset markers
            if isinstance(market, tuple) and market[0] == 'MARKET_RESET':
                if self.position and last_market_time:
                    # Close position from old market
                    self.close_position(0.0, last_market_time, 'MARKET_CHANGED')
                self.market_end_time = None
                continue

            # Detect new market via time gaps (backup detection)
            if last_market_time:
                time_diff = (market.timestamp - last_market_time).total_seconds()
                if time_diff > 1000:  # Gap indicates new market
                    self.market_end_time = None
                    if self.position:
                        # Close position from old market
                        self.close_position(0.0, last_market_time, 'MARKET_CHANGED')

            self.process_market_state(market)
            last_market_time = market.timestamp
            
            if i % 10000 == 0 and i > 0:
                print(f"  {i}/{len(market_states)} - Trades: {len(self.trades)}, PnL: ${self.balance:+.2f}")
        
        # Close any open position
        if self.position and market_states:
            last = market_states[-1]
            price = last.up_price if self.position['side'] == 'UP' else last.down_price
            self.close_position(price, last.timestamp, 'END_OF_DATA')
        
        winning = [t for t in self.trades if t['pnl'] > 0]
        losing = [t for t in self.trades if t['pnl'] <= 0]
        
        coin = recording_files[0].parent.name.upper()
        
        return BacktestResult(
            strategy_name=f"Longshot ({int((1-self.underdog_threshold)*100)}-{int(self.underdog_threshold*100)} odds)",
            coin=coin,
            trades=self.trades,
            total_pnl=self.balance,
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning)/len(self.trades)*100 if self.trades else 0
        )

def print_results(result: BacktestResult):
    """Print backtest results"""
    print("\n" + "="*70)
    print(f"BACKTEST RESULTS - {result.coin} - {result.strategy_name}")
    print("="*70)
    print(f"\nTotal Trades: {result.total_trades}")
    print(f"Winning: {result.winning_trades} ({result.win_rate:.1f}%)")
    print(f"Losing: {result.losing_trades}")
    print(f"\nTotal PnL: ${result.total_pnl:+.2f}")
    
    if result.trades:
        avg_pnl = result.total_pnl / len(result.trades)
        print(f"Average PnL per trade: ${avg_pnl:+.2f}")
        
        winning = [t for t in result.trades if t['pnl'] > 0]
        losing = [t for t in result.trades if t['pnl'] <= 0]
        
        if winning:
            avg_win = sum(t['pnl'] for t in winning) / len(winning)
            max_win = max(t['pnl'] for t in winning)
            print(f"Average win: ${avg_win:+.2f}")
            print(f"Max win: ${max_win:+.2f}")
        
        if losing:
            avg_loss = sum(t['pnl'] for t in losing) / len(losing)
            max_loss = min(t['pnl'] for t in losing)
            print(f"Average loss: ${avg_loss:+.2f}")
            print(f"Max loss: ${max_loss:+.2f}")
        
        print(f"\nSample trades (first 10):")
        for trade in result.trades[:10]:
            print(f"  {trade['side']} ({trade['entry_odds']}) @{trade['entry_price']:.3f} -> {trade['exit_price']:.3f}: "
                  f"${trade['pnl']:+.2f} ({trade['reason']}, {trade['hold_time']:.0f}s)")
    else:
        print("\n⚠️  No longshot opportunities found.")
        print(f"   No extreme longshot opportunities found in this data")
    
    print("="*70)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest longshot strategy')
    parser.add_argument('--coin', type=str, required=True, choices=['BTC', 'ETH', 'SOL', 'XRP'])
    parser.add_argument('--odds', type=float, default=0.01, 
                        help='Underdog threshold (0.01 = 99-1 odds)')
    parser.add_argument('--min-time', type=int, default=120,
                        help='Minimum time remaining in seconds (default: 120)')
    
    args = parser.parse_args()
    
    coin_dir = Path('data/recordings') / args.coin.lower()
    files = sorted(coin_dir.glob("*.jsonl"))
    
    if not files:
        print(f"No data for {args.coin}")
        return
    
    print(f"Testing longshot strategy on {args.coin}")
    print(f"  Betting on {int((1-args.odds)*100)}-{int(args.odds*100)} underdogs")
    print(f"  Minimum time remaining: {args.min_time}s")
    
    backtester = LongshotBacktester(
        underdog_threshold=args.odds,
        min_time_remaining=args.min_time
    )
    
    result = backtester.run_backtest(files)
    print_results(result)

if __name__ == "__main__":
    main()
