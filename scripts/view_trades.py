#!/usr/bin/env python3
"""View recent trade history from Polymarket."""

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import TradeParams


def format_timestamp(ts: str) -> str:
    """Convert timestamp to readable format."""
    try:
        # Handle both epoch seconds and ISO format
        if ts.isdigit():
            dt = datetime.fromtimestamp(int(ts))
        else:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


def format_price(price: str) -> str:
    """Format price as percentage."""
    try:
        p = float(price)
        return f"{p*100:.1f}%"
    except Exception:
        return price


def format_size(size: str) -> str:
    """Format size in USDC."""
    try:
        s = float(size)
        return f"${s:.2f}"
    except Exception:
        return size


def main():
    private_key = os.getenv("POLY_PRIVATE_KEY")
    safe_address = os.getenv("POLY_SAFE_ADDRESS")

    if not private_key or not safe_address:
        print("ERROR: Set POLY_PRIVATE_KEY and POLY_SAFE_ADDRESS in .env")
        return

    print(f"Fetching trades for: {safe_address[:10]}...{safe_address[-6:]}")
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
    try:
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
    except Exception as e:
        print(f"ERROR: Failed to derive API credentials: {e}")
        return

    # Fetch trades
    print("Recent Trades:")
    print("-" * 80)

    try:
        # Get trades - filter by maker_address to get our trades
        params = TradeParams(maker_address=safe_address)
        result = client.get_trades(params)

        trades = result if isinstance(result, list) else result.get("data", [])

        if not trades:
            print("No trades found.")
            print()
            print("Tips:")
            print("  - Trades may take a few seconds to appear after execution")
            print("  - Check that your safe address is correct")
            print("  - Verify your bot has successfully placed orders")
            return

        for trade in trades[:20]:  # Show last 20 trades
            ts = trade.get("match_time") or trade.get("created_at") or trade.get("timestamp", "")
            trade_id = trade.get("id", "")[:8] or trade.get("trade_id", "")[:8]
            side = trade.get("side", "?").upper()
            price = format_price(trade.get("price", "0"))
            size = format_size(trade.get("size", "0"))
            status = trade.get("status", "FILLED")
            asset_id = trade.get("asset_id", "")
            market = trade.get("market", "")

            # Color-code side
            side_display = f"{'BUY ' if side == 'BUY' else 'SELL'}"

            # Truncate asset_id for display
            asset_short = asset_id[:12] + "..." if len(asset_id) > 15 else asset_id

            print(f"{format_timestamp(ts)} | {trade_id} | {side_display} | {price:>6} | {size:>10} | {status}")
            if market:
                print(f"                        Market: {market[:60]}")
            if asset_id:
                print(f"                        Token:  {asset_short}")
            print()

        print("-" * 80)
        print(f"Showing {min(len(trades), 20)} of {len(trades)} trades")

    except Exception as e:
        print(f"ERROR: Failed to fetch trades: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
