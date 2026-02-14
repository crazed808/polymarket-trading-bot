#!/usr/bin/env python3
"""Check actual token positions on Polymarket."""

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

    client = ClobClient(
        host="https://clob.polymarket.com",
        key=private_key,
        chain_id=137,
        signature_type=2,
        funder=safe_address,
    )
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)

    print("=== CHECKING OPEN POSITIONS ===")
    print()

    # Try to get positions from the API
    try:
        # The client doesn't have a direct get_positions method
        # Let's check balance for specific tokens from recent trades

        # First, get recent trades to find token IDs
        trades = client.get_trades()
        print(f"Found {len(trades) if trades else 0} recent trades")

        if trades:
            # Get unique token IDs
            token_ids = set()
            for trade in trades[:20]:  # Check last 20 trades
                asset_id = trade.get("asset_id")
                if asset_id:
                    token_ids.add(asset_id)

            print(f"Checking balances for {len(token_ids)} tokens...")
            print()

            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            for token_id in token_ids:
                try:
                    params = BalanceAllowanceParams(
                        asset_type=AssetType.CONDITIONAL,
                        token_id=token_id
                    )
                    result = client.get_balance_allowance(params)
                    balance = float(result.get("balance", 0))

                    if balance > 0.001:  # Only show non-zero balances
                        print(f"Token: {token_id[:30]}...")
                        print(f"  Balance: {balance:.4f} shares")
                        print()
                except Exception as e:
                    pass  # Token might not exist anymore

    except Exception as e:
        print(f"Error getting trades: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=== RECENT TRADES (last 10) ===")
    try:
        trades = client.get_trades()
        for trade in (trades or [])[:10]:
            side = trade.get("side", "?")
            price = trade.get("price", 0)
            size = trade.get("size", 0)
            status = trade.get("status", "?")
            ts = trade.get("created_at", "")[:19]
            asset_id = trade.get("asset_id", "")[:20]
            print(f"  {ts} | {side:4} {float(size):.2f} @ {float(price):.4f} | {status} | {asset_id}...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
