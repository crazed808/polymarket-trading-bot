#!/usr/bin/env python3
"""
Continuous orderbook recorder with automatic 30-day data retention
Records BTC, ETH, SOL, XRP simultaneously
Automatically deletes data older than 30 days
"""
import asyncio
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from lib.market_manager import MarketManager
from src.websocket_client import OrderbookSnapshot

class ContinuousRecorder:
    def __init__(self, coin: str, output_dir: str = "data/recordings", retention_days: int = 30):
        self.coin = coin
        self.output_dir = Path(output_dir) / coin.lower()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.current_market_file = None
        self.snapshot_count = 0
        self.start_time = None
        self.cleanup_interval = 3600  # Clean up every hour
        self.last_cleanup = None
        
    async def record_forever(self):
        """Record orderbook data continuously"""
        print(f"[{self.coin}] Starting continuous recorder")
        print(f"[{self.coin}] Output: {self.output_dir}")
        print(f"[{self.coin}] Retention: {self.retention_days} days\n")
        
        self.start_time = datetime.now()
        self.last_cleanup = self.start_time
        
        while True:  # Run forever
            try:
                # Create market manager
                mm = MarketManager(
                    coin=self.coin,
                    auto_switch_market=True,
                    market_check_interval=30.0
                )
                
                # Set up callbacks
                @mm.on_book_update
                async def handle_orderbook(snapshot: OrderbookSnapshot):
                    await self.save_snapshot(snapshot)
                    
                    # Periodic cleanup
                    if (datetime.now() - self.last_cleanup).total_seconds() > self.cleanup_interval:
                        self.cleanup_old_data()
                        self.last_cleanup = datetime.now()
                
                @mm.on_market_change
                def handle_market_change(old_slug: str, new_slug: str):
                    print(f"[{self.coin}] [{datetime.now().strftime('%H:%M:%S')}] Market: {new_slug}")
                    self.current_market_file = None
                
                # Start recording
                if not await mm.start():
                    print(f"[{self.coin}] Failed to start, retrying in 60s...")
                    await asyncio.sleep(60)
                    continue
                
                print(f"[{self.coin}] ✓ Recording started: {mm.current_market.slug}")
                
                # Run indefinitely
                while True:
                    await asyncio.sleep(300)  # Status update every 5 min
                    
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    disk_usage = self.get_disk_usage()
                    
                    print(f"[{self.coin}] [{datetime.now().strftime('%H:%M:%S')}] "
                          f"Snapshots: {self.snapshot_count} | "
                          f"Uptime: {elapsed/3600:.1f}h | "
                          f"Storage: {disk_usage:.1f} MB")
                
            except KeyboardInterrupt:
                print(f"\n[{self.coin}] Stopped by user")
                break
            except Exception as e:
                print(f"[{self.coin}] Error: {e}, restarting in 60s...")
                await asyncio.sleep(60)
    
    async def save_snapshot(self, snapshot: OrderbookSnapshot):
        """Save orderbook snapshot to disk"""
        try:
            # Create new file for each day
            date_str = datetime.now().strftime("%Y%m%d")
            if self.current_market_file is None or not self.current_market_file.name.startswith(date_str):
                filename = f"{date_str}_{snapshot.asset_id[:8]}.jsonl"
                self.current_market_file = self.output_dir / filename
            
            # Snapshot data - use built-in attributes
            data = {
                "ts": datetime.now().isoformat(),
                "asset_id": snapshot.asset_id,
                "bids": [[float(level.price), float(level.size)] for level in snapshot.bids[:10]],
                "asks": [[float(level.price), float(level.size)] for level in snapshot.asks[:10]],
                "best_bid": float(snapshot.best_bid) if snapshot.best_bid else 0.0,
                "best_ask": float(snapshot.best_ask) if snapshot.best_ask else 0.0,
                "mid": float(snapshot.mid_price) if snapshot.mid_price else 0.0
            }
            
            # Append to file
            with open(self.current_market_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
            
            self.snapshot_count += 1
            
        except Exception as e:
            print(f"[{self.coin}] Error saving snapshot: {e}")
    
    def cleanup_old_data(self):
        """Delete data older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            deleted_count = 0
            deleted_size = 0
            
            for file_path in self.output_dir.glob("*.jsonl"):
                # Parse date from filename (format: YYYYMMDD_...)
                try:
                    date_str = file_path.name.split('_')[0]
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    
                    if file_date < cutoff_date:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        deleted_count += 1
                        deleted_size += file_size
                        
                except Exception:
                    continue  # Skip files with unexpected names
            
            if deleted_count > 0:
                print(f"[{self.coin}] 🗑️  Cleaned up {deleted_count} files "
                      f"({deleted_size/1024/1024:.1f} MB) older than {self.retention_days} days")
                
        except Exception as e:
            print(f"[{self.coin}] Error during cleanup: {e}")
    
    def get_disk_usage(self) -> float:
        """Get total disk usage in MB"""
        total = 0
        for file_path in self.output_dir.glob("*.jsonl"):
            total += file_path.stat().st_size
        return total / 1024 / 1024

async def record_all_coins():
    """Record all 4 coins simultaneously with auto-cleanup"""
    coins = ['BTC', 'ETH', 'SOL', 'XRP']
    
    print("="*70)
    print("CONTINUOUS ORDERBOOK RECORDER")
    print("="*70)
    print(f"Coins: {', '.join(coins)}")
    print(f"Retention: 30 days")
    print(f"Output: data/recordings/")
    print(f"\nPress Ctrl+C to stop")
    print("="*70 + "\n")
    
    # Create recorders
    recorders = [ContinuousRecorder(coin, retention_days=30) for coin in coins]
    
    # Run all concurrently
    tasks = [recorder.record_forever() for recorder in recorders]
    
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\n\nShutting down all recorders...")

def main():
    """Main entry point"""
    # Create data directory
    Path("data/recordings").mkdir(parents=True, exist_ok=True)
    
    # Run continuous recording
    asyncio.run(record_all_coins())

if __name__ == "__main__":
    main()
