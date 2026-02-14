#!/usr/bin/env python3
"""
Backtest trading strategies using recorded orderbook data
Replays historical snapshots to simulate live trading
"""
import json
import asyncio
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
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

class OrderbookSnapshot:
    """Recreate snapshot from recorded data"""
    def __init__(self, data: Dict):
        self.timestamp = datetime.fromisoformat(data['ts'])
        self.asset_id = data['asset_id']
        self.bids = [(p, s) for p, s in data['bids']]
        self.asks = [(p, s) for p, s in data['asks']]
        self.best_bid = data.get('best_bid', self.bids[0][0] if self.bids else 0.0)
        self.best_ask = data.get('best_ask', self.asks[0][0] if self.asks else 1.0)
        self.mid = data.get('mid', (self.best_bid + self.best_ask) / 2)

class FlashCrashBacktester:
    """Backtest flash crash strategy on recorded data"""
    
    def __init__(self, 
                 drop_threshold: float = 0.30,
                 take_profit: float = 0.10,
                 stop_loss: float = 0.05,
                 position_size: float = 1.0,
                 lookback_window: int = 100):
        self.drop_threshold = drop_threshold
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.position_size = position_size
        self.lookback_window = lookback_window
        
        # Strategy state
        self.price_history = deque(maxlen=lookback_window)
        self.position = None
        self.trades = []
        self.balance = 0.0
        
    def detect_flash_crash(self, current_price: float) -> Dict[str, Any]:
        """Detect if there's a flash crash"""
        if len(self.price_history) < 10:
            return None
        
        # Check for sudden drop
        recent_max = max(self.price_history)
        drop = recent_max - current_price
        drop_pct = drop / recent_max if recent_max > 0 else 0
        
        if drop_pct >= self.drop_threshold:
            return {
                'side': 'BUY',
                'entry_price': current_price,
                'recent_max': recent_max,
                'drop_pct': drop_pct
            }
        
        return None
    
    def update_position(self, current_price: float, timestamp: datetime) -> Dict:
        """Check if position should be closed"""
        if not self.position:
            return None
        
        entry = self.position['entry_price']
        pnl = current_price - entry
        pnl_pct = pnl / entry if entry > 0 else 0
        
        # Take profit
        if pnl >= self.take_profit:
            return self.close_position(current_price, timestamp, 'TAKE_PROFIT')
        
        # Stop loss
        if pnl <= -self.stop_loss:
            return self.close_position(current_price, timestamp, 'STOP_LOSS')
        
        return None
    
    def close_position(self, exit_price: float, timestamp: datetime, reason: str) -> Dict:
        """Close current position"""
        if not self.position:
            return None
        
        pnl = (exit_price - self.position['entry_price']) * self.position_size
        
        trade = {
            'entry_time': self.position['entry_time'],
            'entry_price': self.position['entry_price'],
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
    
    def process_snapshot(self, snapshot: OrderbookSnapshot):
        """Process a single orderbook snapshot"""
        # Use mid price for signal
        current_price = snapshot.mid
        self.price_history.append(current_price)
        
        # Update existing position
        if self.position:
            trade = self.update_position(current_price, snapshot.timestamp)
            if trade:
                return {'type': 'CLOSE', 'trade': trade}
        
        # Look for entry signal (only if no position)
        if not self.position:
            signal = self.detect_flash_crash(current_price)
            if signal:
                self.position = {
                    'entry_time': snapshot.timestamp,
                    'entry_price': signal['entry_price'],
                    'size': self.position_size
                }
                return {'type': 'OPEN', 'signal': signal}
        
        return None

    def run_backtest(self, recording_files: List[Path]) -> BacktestResult:
        """Run backtest on recorded data"""
        print(f"Processing {len(recording_files)} recording files...")
        
        snapshot_count = 0
        start_time = None
        end_time = None
        
        for file_path in sorted(recording_files):
            print(f"  Reading {file_path.name}...")
            
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        snapshot = OrderbookSnapshot(data)
                        
                        if start_time is None:
                            start_time = snapshot.timestamp
                        end_time = snapshot.timestamp
                        
                        # Process snapshot
                        event = self.process_snapshot(snapshot)
                        
                        snapshot_count += 1
                        
                        if snapshot_count % 10000 == 0:
                            print(f"    Processed {snapshot_count} snapshots, "
                                  f"Trades: {len(self.trades)}, "
                                  f"Balance: ${self.balance:+.2f}")
                        
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"    Error processing snapshot: {e}")
                        continue
        
        # Close any open position at end
        if self.position and end_time:
            self.close_position(self.price_history[-1], end_time, 'END_OF_DATA')
        
        # Calculate results
        winning = [t for t in self.trades if t['pnl'] > 0]
        losing = [t for t in self.trades if t['pnl'] <= 0]
        
        result = BacktestResult(
            strategy_name="Flash Crash",
            coin=recording_files[0].parent.name.upper(),
            start_time=start_time.isoformat() if start_time else "",
            end_time=end_time.isoformat() if end_time else "",
            trades=self.trades,
            total_pnl=self.balance,
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning)/len(self.trades)*100 if self.trades else 0
        )
        
        return result

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
        
        # Show some recent trades
        print(f"\nRecent trades (last 5):")
        for trade in result.trades[-5:]:
            print(f"  {trade['entry_time'].strftime('%H:%M:%S')} -> "
                  f"{trade['exit_time'].strftime('%H:%M:%S')}: "
                  f"${trade['pnl']:+.2f} ({trade['reason']})")
    
    print("="*70)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest with recorded data')
    parser.add_argument('--coin', type=str, required=True, choices=['BTC', 'ETH', 'SOL', 'XRP'],
                        help='Coin to backtest')
    parser.add_argument('--drop-threshold', type=float, default=0.30,
                        help='Flash crash drop threshold (default: 0.30)')
    parser.add_argument('--take-profit', type=float, default=0.10,
                        help='Take profit amount (default: 0.10)')
    parser.add_argument('--stop-loss', type=float, default=0.05,
                        help='Stop loss amount (default: 0.05)')
    parser.add_argument('--data-dir', type=str, default='data/recordings',
                        help='Directory with recorded data')
    
    args = parser.parse_args()
    
    # Find recording files
    coin_dir = Path(args.data_dir) / args.coin.lower()
    if not coin_dir.exists():
        print(f"Error: No data found for {args.coin} in {coin_dir}")
        print(f"Start recording first: python3 continuous_recorder.py")
        return
    
    recording_files = sorted(coin_dir.glob("*.jsonl"))
    if not recording_files:
        print(f"Error: No recording files found in {coin_dir}")
        return
    
    print(f"Found {len(recording_files)} recording files for {args.coin}")
    
    # Run backtest
    backtester = FlashCrashBacktester(
        drop_threshold=args.drop_threshold,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        position_size=1.0
    )
    
    result = backtester.run_backtest(recording_files)
    print_results(result)

if __name__ == "__main__":
    main()
