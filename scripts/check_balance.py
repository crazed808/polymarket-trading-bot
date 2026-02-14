#!/usr/bin/env python3
"""Check USDC balance and allowance on Polymarket."""

import os
from dotenv import load_dotenv

load_dotenv()

from py_clob_client.client import ClobClient

def main():
    private_key = os.getenv("POLY_PRIVATE_KEY")
    safe_address = os.getenv("POLY_SAFE_ADDRESS")

    if not private_key or not safe_address:
        print("ERROR: Set POLY_PRIVATE_KEY and POLY_SAFE_ADDRESS in .env")
        return

    print(f"Safe address: {safe_address}")
    print()

    # Initialize client
    client = ClobClient(
        host="https://clob.polymarket.com",
        key=private_key,
        chain_id=137,
        signature_type=2,
        funder=safe_address,
    )

    # Derive API creds
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)

    print("Checking USDC (collateral) balance and allowance...")
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        result = client.get_balance_allowance(params)
        print(f"  Raw result: {result}")

        # Parse if it's a dict
        if isinstance(result, dict):
            balance = float(result.get("balance", 0)) / 1e6  # USDC has 6 decimals
            print(f"  Balance:   ${balance:.2f} USDC")

            # Allowances can be singular or plural (dict of contract addresses)
            allowances = result.get("allowances", {})
            if isinstance(allowances, dict) and allowances:
                print(f"  Allowances approved for {len(allowances)} contract(s):")
                for contract, amount in allowances.items():
                    amount_val = float(amount) / 1e6
                    if amount_val > 1e12:  # Effectively unlimited
                        print(f"    {contract[:10]}...{contract[-6:]}: unlimited")
                    else:
                        print(f"    {contract[:10]}...{contract[-6:]}: ${amount_val:.2f}")
            else:
                # Fallback to singular allowance
                allowance = float(result.get("allowance", 0)) / 1e6
                print(f"  Allowance: ${allowance:.2f}")
                if allowance < 1000:
                    print()
                    print("⚠️  Allowance is low! You may need to approve the exchange contract.")
                    print("   This is usually done automatically when you trade on polymarket.com")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("Checking open orders...")
    try:
        orders = client.get_orders()
        if orders:
            print(f"  Found {len(orders)} open orders:")
            for order in orders[:5]:
                print(f"    - {order.get('side')} @ {order.get('price')} size={order.get('original_size')}")
            if len(orders) > 5:
                print(f"    ... and {len(orders) - 5} more")
            print()
            print("⚠️  Open orders reserve your balance!")
        else:
            print("  No open orders")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    main()
