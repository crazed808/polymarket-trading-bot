#!/usr/bin/env python3
"""
View popular markets and their prices
"""
import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.bot import TradingBot

async def main():
    print("=" * 60)
    print("Polymarket Markets Viewer")
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
    
    # Fetch CURRENT markets using the newer API endpoint
    print("\nFetching current markets...")
    
    # Try the newer Polymarket API
    try:
        # This endpoint should have current markets
        response = requests.get(
            "https://clob.polymarket.com/markets",
            params={
                "limit": 10,
                "active": "true"
            }
        )
        
        if response.status_code == 200:
            markets_data = response.json()
            markets = markets_data if isinstance(markets_data, list) else markets_data.get('data', [])
        else:
            print(f"API returned status {response.status_code}")
            print("Trying alternative endpoint...")
            
            # Alternative: scrape from web or use different endpoint
            response = requests.get("https://strapi-matic.poly.market/markets?_limit=10&active=true&closed=false&_sort=volume:desc")
            markets = response.json()
    except Exception as e:
        print(f"Error fetching markets: {e}")
        markets = []
    
    if not markets:
        print("\n⚠️  Could not fetch current markets")
        print("\nYou can find markets manually at: https://polymarket.com")
        print("\nTo trade, you need:")
        print("1. Go to polymarket.com and find a market you like")
        print("2. Copy the token ID from the URL or market page")
        print("3. Use that token ID with the bot's trading functions")
        return
    
    print(f"\n📊 Found {len(markets)} Markets:\n")
    
    for i, market in enumerate(markets[:10], 1):
        title = market.get('question') or market.get('title') or market.get('description', 'Unknown')
        print(f"{i}. {title[:80]}")
        
        # Try to show any available data
        for key in ['volume', 'liquidity', 'volumeNum']:
            if key in market:
                val = market[key]
                try:
                    print(f"   {key.title()}: ${float(val):,.0f}")
                    break
                except:
                    pass
        
        print()
    
    print("=" * 60)
    print("\n💡 To trade on a specific market:")
    print("   1. Visit https://polymarket.com")
    print("   2. Find a market and copy its token ID")
    print("   3. Use bot.place_order(token_id, price, size, side)")

if __name__ == "__main__":
    asyncio.run(main())
