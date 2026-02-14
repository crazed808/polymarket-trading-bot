#!/usr/bin/env python3
"""
Fixed backtester - treats UP/DOWN as a paired market
"""
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, field
from collections import deque

@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    coin: str
    start_time: str
    end_time: str
    trades: List[Dict] = field(default_factory=list)
    total_pnl: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

class MarketState:
    """Tracks both UP and DOWN tokens for a market"""
    def __init__(self, timestamp, up_price, down_price):
        self.timestamp = timestamp
        self.up_price = up_price
        self.down_price = down_price
        
class FlashCrashBacktester:
    """Fixed backtester with proper market pairing"""
    
    def __init__(self, 
                 drop_threshold: float = 0.30,
                 take_profit: float = 0.10,
                 stop_loss: float = 0.05,
                 position_size: float = 1.0,
                 lookback_window: int = 100,
                 fee_rate: float = 0.02,
                 slippage: float = 0.01):
        self.drop_threshold = drop_threshold
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.position_size = position_size
        self.lookback_window = lookback_window
        self.fee_rate = fee_rate
        self.slippage = slippage
        
        self.up_history = deque(maxlen=lookback_window)
        self.down_history = deque(maxlen=lookback_window)
        
        self.position = None
        self.trades = []
        self.balance = 0.0
        
    def detect_flash_crash(self, market: MarketState) -> Dict[str, Any]:
        """Detect flash crash in either UP or DOWN"""
        if len(self.up_history) < 10:
            return None
        
        signals = []
        
        # Check UP for crash
        recent_max_up = max(self.up_history)
        drop_pct_up = (recent_max_up - market.up_price) / recent_max_up if recent_max_up > 0 else 0
        
        if drop_pct_up >= self.drop_threshold:
            signals.append({
                'side': 'UP',
                'entry_price': market.up_price,
                'drop_pct': drop_pct_up
            })
        
        # Check DOWN for crash
        recent_max_down = max(self.down_history)
        drop_pct_down = (recent_max_down - market.down_price) / recent_max_down if recent_max_down > 0 else 0
        
        if drop_pct_down >= self.drop_threshold:
            signals.append({
                'side': 'DOWN',
                'entry_price': market.down_price,
                'drop_pct': drop_pct_down
            })
        
        return max(signals, key=lambda s: s['drop_pct']) if signals else None
    
    def update_position(self, market: MarketState) -> Dict:
        """Check if position should be closed"""
        if not self.position:
            return None
        
        current_price = market.up_price if self.position['side'] == 'UP' else market.down_price
        entry = self.position['entry_price']
        
        exit_price_with_slippage = current_price * (1 - self.slippage)
        gross_pnl = exit_price_with_slippage - entry
        fees = entry * self.fee_rate + exit_price_with_slippage * self.fee_rate
        net_pnl = (gross_pnl - fees) * self.position_size
        
        if gross_pnl >= self.take_profit:
            return self.close_position(exit_price_with_slippage, market.timestamp, 'TAKE_PROFIT', net_pnl)
        
        if gross_pnl <= -self.stop_loss:
            return self.close_position(exit_price_with_slippage, market.timestamp, 'STOP_LOSS', net_pnl)
        
        return None
    
    def close_position(self, exit_price: float, timestamp: datetime, reason: str, pnl: float) -> Dict:
        if not self.position:
            return None
        
        trade = {
            'entry_time': self.position['entry_time'],
            'entry_price': self.position['entry_price'],
            'side': self.position['side'],
            'exit_time': timestamp,
            'exit_price': exit_price,
            'pnl': pnl,
            'reason': reason,
            'hold_time': (timestamp - self.position['entry_time']).total_seconds()
        }
        
        self.trades.append(trade)
        self.balance += pnl
        self.position = None
        return trade
    
    def process_market_state(self, market: MarketState):
        """Process a market state update"""
        self.up_history.append(market.up_price)
        self.down_history.append(market.down_price)
        
        if self.position:
            trade = self.update_position(market)
            if trade:
                return {'type': 'CLOSE', 'trade': trade}
        
        if not self.position:
            signal = self.detect_flash_crash(market)
            if signal:
                entry_price_with_slippage = signal['entry_price'] * (1 + self.slippage)
                
                self.position = {
                    'entry_time': market.timestamp,
                    'entry_price': entry_price_with_slippage,
                    'side': signal['side'],
                    'size': self.position_size
                }
                return {'type': 'OPEN', 'signal': signal}
        
        return None

    def run_backtest(self, recording_files: List[Path]) -> 'BacktestResult':
        """Run backtest - process each snapshot directly"""
        print(f"Processing {len(recording_files)} recording files...")
        
        # Track asset IDs to determine which is UP/DOWN
        asset_prices = {}  # asset_id -> latest price
        
        # Just process snapshots sequentially
        # When we see both asset IDs, create market states
        market_states = []
        snapshot_count = 0
        
        for file_path in sorted(recording_files):
            print(f"  Reading {file_path.name}...")
            
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        ts = datetime.fromisoformat(data['ts'])
                        asset_id = data['asset_id']
                        mid = data['mid']
                        
                        asset_prices[asset_id] = {'price': mid, 'ts': ts}
                        snapshot_count += 1
                        
                        # When we have 2 assets, create market state
                        if len(asset_prices) == 2:
                            prices = list(asset_prices.values())
                            p1, p2 = prices[0]['price'], prices[1]['price']
                            
                            # Higher price is UP (prices sum to ~1.0)
                            if p1 > p2:
                                up, down = p1, p2
                            else:
                                up, down = p2, p1
                            
                            market = MarketState(ts, up, down)
                            market_states.append(market)
                            
                    except (json.JSONDecodeError, KeyError) as e:
                        continue
        
        print(f"\n  Total snapshots: {snapshot_count}")
        print(f"  Created {len(market_states)} market states")
        print(f"  Processing strategy...")
        
        if not market_states:
            print("  ERROR: No market states created!")
            return BacktestResult("Flash Crash", "UNKNOWN", "", "", [])
        
        start_time = market_states[0].timestamp
        end_time = market_states[-1].timestamp
        
        for i, market in enumerate(market_states):
            self.process_market_state(market)
            
            if i % 10000 == 0 and i > 0:
                print(f"    {i}/{len(market_states)} - Trades: {len(self.trades)}, PnL: ${self.balance:+.2f}")
        
        # Close any open position
        if self.position and market_states:
            last = market_states[-1]
            price = last.up_price if self.position['side'] == 'UP' else last.down_price
            self.close_position(price, last.timestamp, 'END_OF_DATA', 0)
        
        winning = [t for t in self.trades if t['pnl'] > 0]
        losing = [t for t in self.trades if t['pnl'] <= 0]
        
        coin = recording_files[0].parent.name.upper()
        
        return BacktestResult(
            strategy_name="Flash Crash (Fixed)",
            coin=coin,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
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
    print(f"Period: {result.start_time} to {result.end_time}")
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
            print(f"Average win: ${avg_win:+.2f}")
        
        if losing:
            avg_loss = sum(t['pnl'] for t in losing) / len(losing)
            print(f"Average loss: ${avg_loss:+.2f}")
        
        print(f"\nSample trades:")
        for trade in result.trades[:10]:
            print(f"  {trade['side']} @{trade['entry_price']:.3f} -> {trade['exit_price']:.3f}: "
                  f"${trade['pnl']:+.2f} ({trade['reason']}, {trade['hold_time']:.0f}s)")
    else:
        print("\n⚠️  No trades executed. Strategy didn't find any flash crashes.")
        print("   Try adjusting --drop-threshold to a lower value (e.g. 0.20)")
    
    print("="*70)

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--coin', type=str, required=True, choices=['BTC', 'ETH', 'SOL', 'XRP'])
    parser.add_argument('--drop-threshold', type=float, default=0.30)
    parser.add_argument('--take-profit', type=float, default=0.10)
    parser.add_argument('--stop-loss', type=float, default=0.05)
    
    args = parser.parse_args()
    
    coin_dir = Path('data/recordings') / args.coin.lower()
    files = sorted(coin_dir.glob("*.jsonl"))
    
    if not files:
        print(f"No data for {args.coin}")
        return
    
    backtester = FlashCrashBacktester(
        drop_threshold=args.drop_threshold,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss
    )
    
    result = backtester.run_backtest(files)
    print_results(result)

if __name__ == "__main__":
    main()
