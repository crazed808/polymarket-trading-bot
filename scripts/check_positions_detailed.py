#!/usr/bin/env python3
"""Check actual token positions with market details."""

import os
from dotenv import load_dotenv

load_dotenv()

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

def main():
    private_key = os.getenv("POLY_PRIVATE_KEY")
    safe_address = os.getenv("POLY_SAFE_ADDRESS")

    client = ClobClient(
        host="https://clob.polymarket.com",
        key=private_key,
        chain_id=137,
        signature_type=2,
        funder=safe_address,
    )
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)

    print("=== DETAILED POSITION CHECK ===")
    print()

    # Get recent trades to find token IDs
    trades = client.get_trades()

    # Get unique token IDs
    token_ids = set()
    for trade in (trades or [])[:50]:
        asset_id = trade.get("asset_id")
        if asset_id:
            token_ids.add(asset_id)

    print(f"Checking {len(token_ids)} tokens from recent trades...")
    print()

    for token_id in token_ids:
        try:
            params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=token_id
            )
            result = client.get_balance_allowance(params)
            raw_balance = result.get("balance", "0")

            # Try different decimal interpretations
            balance_raw = float(raw_balance)
            balance_6dec = balance_raw / 1e6  # USDC-style 6 decimals
            balance_18dec = balance_raw / 1e18  # ETH-style 18 decimals

            if balance_raw > 0.001:
                print(f"Token: {token_id}")
                print(f"  Raw balance:   {balance_raw}")
                print(f"  If 6 decimals: {balance_6dec:.6f}")
                print(f"  If 18 decimals: {balance_18dec:.18f}")

                # Try to get market info for this token
                try:
                    market = client.get_market(token_id)
                    if market:
                        print(f"  Market: {market.get('question', 'Unknown')[:60]}...")
                        print(f"  End date: {market.get('end_date_iso', 'Unknown')}")
                except:
                    pass

                # Get last price from orderbook
                try:
                    book = client.get_order_book(token_id)
                    if book:
                        bids = book.get("bids", [])
                        asks = book.get("asks", [])
                        best_bid = float(bids[0]["price"]) if bids else 0
                        best_ask = float(asks[0]["price"]) if asks else 0
                        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
                        print(f"  Current price: bid={best_bid:.4f} ask={best_ask:.4f} mid={mid:.4f}")

                        # Calculate value
                        if balance_6dec > 0 and mid > 0:
                            value_6dec = balance_6dec * mid
                            print(f"  Estimated value (6 dec): ${value_6dec:.2f}")
                except Exception as e:
                    print(f"  Could not get orderbook: {e}")

                print()
        except Exception as e:
            pass

if __name__ == "__main__":
    main()
