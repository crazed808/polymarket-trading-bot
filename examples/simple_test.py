#!/usr/bin/env python3
"""
Simple test to verify bot is working
"""
import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.bot import TradingBot

async def main():
    print("=" * 50)
    print("Simple Bot Test")
    print("=" * 50)
    
    # Get credentials from environment
    private_key = os.getenv("POLY_PRIVATE_KEY")
    safe_address = os.getenv("POLY_SAFE_ADDRESS")
    
    if not private_key or not safe_address:
        print("ERROR: Please set POLY_PRIVATE_KEY and POLY_SAFE_ADDRESS in .env file")
        return
    
    # Initialize bot with direct credentials
    bot = TradingBot(
        private_key=private_key,
        safe_address=safe_address,
        config_path="config.yaml"
    )
    
    # Get account info
    print(f"\n✓ Safe Address: {bot.config.safe_address}")
    print(f"✓ Gasless Mode: {bot.config.use_gasless}")
    
    # Get open orders
    print("\n--- Checking Open Orders ---")
    orders = await bot.get_open_orders()
    print(f"You have {len(orders)} open orders")
    
    if orders:
        for order in orders[:3]:  # Show first 3
            print(f"  - Order: {order}")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(main())
