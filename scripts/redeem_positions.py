#!/usr/bin/env python3
"""
Redeem all resolved positions from Polymarket.

This script finds any positions from resolved markets and redeems them
to convert winning shares back to USDC.

Usage:
    python scripts/redeem_positions.py          # Check and redeem
    python scripts/redeem_positions.py --dry-run # Just show redeemable positions
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.bot import TradingBot


async def main():
    parser = argparse.ArgumentParser(description='Redeem resolved Polymarket positions')
    parser.add_argument('--dry-run', action='store_true', help='Only show redeemable positions, do not redeem')
    args = parser.parse_args()

    private_key = os.getenv("POLY_PRIVATE_KEY")
    safe_address = os.getenv("POLY_SAFE_ADDRESS")

    if not private_key or not safe_address:
        print("ERROR: Set POLY_PRIVATE_KEY and POLY_SAFE_ADDRESS in .env")
        return

    print(f"Safe address: {safe_address[:10]}...{safe_address[-6:]}")
    print()

    # Initialize bot
    print("Initializing bot...")
    bot = TradingBot(
        private_key=private_key,
        safe_address=safe_address,
        config_path="config.yaml"
    )

    # Check gasless mode
    if not bot.config.use_gasless:
        print()
        print("WARNING: Gasless mode is NOT enabled!")
        print("Redemption requires Builder credentials for gasless transactions.")
        print()
        print("Make sure these are set in your .env file:")
        print("  POLY_BUILDER_API_KEY=...")
        print("  POLY_BUILDER_API_SECRET=...")
        print("  POLY_BUILDER_API_PASSPHRASE=...")
        print()
        return

    print(f"Gasless mode: {bot.config.use_gasless}")
    print()

    # Get redeemable positions
    print("Checking for redeemable positions...")
    positions = await bot.get_redeemable_positions()

    if not positions:
        print("No redeemable positions found.")
        print()
        print("Tips:")
        print("  - Positions are only redeemable after a market resolves")
        print("  - Markets resolve at their end time when the outcome is determined")
        print("  - Your bot trades minute-by-minute markets, which resolve quickly")
        return

    # Display redeemable positions
    print(f"\nFound {len(positions)} redeemable position(s):")
    print("-" * 70)

    total_value = 0.0
    for pos in positions:
        title = pos.get("title", "Unknown")[:50]
        condition_id = pos.get("conditionId", "")[:16]
        value = pos.get("currentValue", 0)
        outcome = pos.get("outcome", "?")
        total_value += value

        print(f"  {title}")
        print(f"    Condition: {condition_id}...")
        print(f"    Outcome: {outcome}")
        print(f"    Value: ${value:.4f}")
        print()

    print("-" * 70)
    print(f"Total redeemable value: ${total_value:.4f}")
    print()

    if args.dry_run:
        print("(Dry run - not redeeming)")
        return

    # Redeem all
    print("Redeeming positions...")
    redeemed = await bot.auto_redeem_all()

    print()
    if redeemed > 0:
        print(f"Successfully redeemed {redeemed} position(s)!")
        print("USDC should be added to your balance shortly.")
    else:
        print("No positions were redeemed. Check the logs for errors.")


if __name__ == "__main__":
    asyncio.run(main())
