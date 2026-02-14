#!/usr/bin/env python3
"""
Example: How to place an order on Polymarket

BEFORE RUNNING THIS:
1. Go to https://polymarket.com
2. Find a market you want to trade
3. Click on the outcome you want (Yes/No)
4. Look at the URL or inspect element to find the token ID
5. Replace TOKEN_ID below with the actual token ID

This is a DEMO - it won't actually place orders unless you uncomment the execute line.
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
    print("=" * 60)
    print("Polymarket Order Placement Example")
    print("=" * 60)
    
    # Get credentials
    private_key = os.getenv("POLY_PRIVATE_KEY")
    safe_address = os.getenv("POLY_SAFE_ADDRESS")
    
    # Initialize bot
    bot = TradingBot(
        private_key=private_key,
        safe_address=safe_address,
        config_path="config.yaml"
    )
    
    print(f"\n✓ Bot initialized")
    print(f"✓ Safe Address: {safe_address}")
    print(f"✓ Balance: Check on polymarket.com - you have $65")
    
    # Example order parameters
    TOKEN_ID = "YOUR_TOKEN_ID_HERE"  # Replace with actual token ID
    PRICE = 0.65      # Price you want to buy/sell at (0.01 to 0.99)
    SIZE = 1.0        # Amount in USD to trade
    SIDE = "BUY"      # "BUY" or "SELL"
    
    print("\n" + "=" * 60)
    print("EXAMPLE ORDER (NOT EXECUTED)")
    print("=" * 60)
    print(f"Token ID: {TOKEN_ID}")
    print(f"Side: {SIDE}")
    print(f"Price: ${PRICE} (meaning {PRICE*100:.0f}% probability)")
    print(f"Size: ${SIZE}")
    print(f"Total Cost: ${PRICE * SIZE:.2f}")
    
    print("\n" + "=" * 60)
    print("HOW TO ACTUALLY TRADE:")
    print("=" * 60)
    print("""
1. Find a market on polymarket.com
2. Get the token ID for the outcome you want
3. Update the TOKEN_ID variable above
4. Review the price and size
5. Uncomment the line below to execute:

   # result = await bot.place_order(
   #     token_id=TOKEN_ID,
   #     price=PRICE,
   #     size=SIZE,
   #     side=SIDE
   # )
   # print(f"Order result: {result}")

WARNING: Real trades use real money!
Start with small amounts ($1-5) to test.
""")
    
    print("\n" + "=" * 60)
    print("CHECKING YOUR CURRENT ORDERS:")
    print("=" * 60)
    
    orders = await bot.get_open_orders()
    print(f"\nYou currently have {len(orders)} open orders")
    
    if orders:
        for order in orders:
            print(f"\nOrder ID: {order.get('id')}")
            print(f"  Side: {order.get('side')}")
            print(f"  Price: {order.get('price')}")
            print(f"  Size: {order.get('size')}")

if __name__ == "__main__":
    asyncio.run(main())
