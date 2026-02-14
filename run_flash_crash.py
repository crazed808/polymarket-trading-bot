#!/usr/bin/env python3
"""
Runner for Flash Crash Strategy
"""
import asyncio
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

from strategies.flash_crash import FlashCrashStrategy, FlashCrashConfig
from src.bot import TradingBot

async def main():
    parser = argparse.ArgumentParser(description='Flash Crash Trading Strategy')
    parser.add_argument('--coin', type=str, required=True, help='Coin symbol (BTC, ETH, SOL, etc.)')
    parser.add_argument('--drop-threshold', type=float, default=0.30, help='Absolute probability drop threshold (default: 0.30)')
    parser.add_argument('--size', type=float, default=5.0, help='Position size in USD (default: 5.0)')
    parser.add_argument('--take-profit', type=float, default=0.10, help='Take profit in USD (default: 0.10)')
    parser.add_argument('--stop-loss', type=float, default=0.05, help='Stop loss in USD (default: 0.05)')
    parser.add_argument('--max-positions', type=int, default=1, help='Max concurrent positions (default: 1)')
    
    args = parser.parse_args()
    
    # Get credentials
    private_key = os.getenv("POLY_PRIVATE_KEY")
    safe_address = os.getenv("POLY_SAFE_ADDRESS")
    
    if not private_key or not safe_address:
        print("ERROR: Please set POLY_PRIVATE_KEY and POLY_SAFE_ADDRESS in .env file")
        return
    
    # Initialize bot
    print(f"Initializing bot for {args.coin}...")
    bot = TradingBot(
        private_key=private_key,
        safe_address=safe_address,
        config_path="config.yaml"
    )
    
    # Create strategy config
    config = FlashCrashConfig(
        coin=args.coin,
        size=args.size,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        max_positions=args.max_positions,
        drop_threshold=args.drop_threshold
    )
    
    # Create and run strategy
    print(f"\nStarting Flash Crash Strategy:")
    print(f"  Coin: {args.coin}")
    print(f"  Drop Threshold: {args.drop_threshold} (absolute probability)")
    print(f"  Position Size: ${args.size}")
    print(f"  Take Profit: ${args.take_profit}")
    print(f"  Stop Loss: ${args.stop_loss}")
    print(f"  Max Positions: {args.max_positions}")
    print()
    print("⚠️  WARNING: This will place REAL trades with REAL money!")
    print("Press Ctrl+C to stop at any time.")
    print()
    
    strategy = FlashCrashStrategy(bot, config)
    
    try:
        await strategy.run()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        await strategy.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
