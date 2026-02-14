#!/usr/bin/env python3
"""Quick account status check - no bot initialization overhead."""
import asyncio
import sys
from dotenv import dotenv_values
import os
for k,v in dotenv_values('.env').items(): os.environ[k]=v

from src.utils import create_bot_from_env
import httpx

async def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "status":
        bot = create_bot_from_env()
        config = dotenv_values('.env')
        safe = config.get('POLY_SAFE_ADDRESS')

        # USDC balance
        ba = await bot.get_balance_allowance('', 'COLLATERAL')
        usdc = float(ba.get("balance", 0)) / 1_000_000
        print(f"USDC: ${usdc:.2f}")

        # Positions
        resp = httpx.get(f'https://data-api.polymarket.com/positions?user={safe}', timeout=10)
        positions = [p for p in resp.json() if isinstance(p, dict) and float(p.get('size', 0)) > 0]
        print(f"Positions: {len(positions)}")
        for p in positions:
            print(f"  {p.get('outcome')}: {float(p.get('size',0)):.2f} @ {float(p.get('avgPrice',0)):.2f}")

        # Open orders
        orders = await bot.get_open_orders()
        print(f"Open orders: {len(orders)}")

    elif cmd == "activity":
        config = dotenv_values('.env')
        safe = config.get('POLY_SAFE_ADDRESS')
        resp = httpx.get(f'https://data-api.polymarket.com/activity?user={safe}&limit=10', timeout=10)
        for a in resp.json()[:10]:
            if isinstance(a, dict):
                print(f"{a.get('type'):6} {a.get('outcome',''):4} {float(a.get('size',0) or 0):>6.2f} @ {float(a.get('price',0) or 0):.2f}")

if __name__ == "__main__":
    asyncio.run(main())
