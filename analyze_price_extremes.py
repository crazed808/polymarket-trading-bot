#!/usr/bin/env python3
"""Analyze price extremes in recorded data"""
import json
from pathlib import Path
from collections import defaultdict

def analyze_coin(coin):
    coin_dir = Path('data/recordings') / coin.lower()
    files = sorted(coin_dir.glob("*.jsonl"))
    
    if not files:
        return
    
    prices = []
    
    for file_path in files:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    prices.append(data['mid'])
                except:
                    continue
    
    if prices:
        min_price = min(prices)
        max_price = max(prices)
        
        print(f"\n{coin}:")
        print(f"  Lowest price seen: {min_price:.4f} ({(1-min_price)*100:.1f}-{min_price*100:.1f} odds)")
        print(f"  Highest price seen: {max_price:.4f}")
        print(f"  Total data points: {len(prices)}")
        
        # Count how many times it got below certain thresholds
        below_05 = sum(1 for p in prices if p <= 0.05)
        below_10 = sum(1 for p in prices if p <= 0.10)
        below_20 = sum(1 for p in prices if p <= 0.20)
        
        print(f"  Times below 0.05 (95-5): {below_05}")
        print(f"  Times below 0.10 (90-10): {below_10}")
        print(f"  Times below 0.20 (80-20): {below_20}")

print("PRICE EXTREMES IN RECORDED DATA")
print("="*60)

for coin in ['BTC', 'ETH', 'SOL', 'XRP']:
    analyze_coin(coin)
