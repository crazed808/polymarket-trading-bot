#!/usr/bin/env python3
"""Debug why market isn't switching"""
from lib.market_manager import MarketManager
import time

mm = MarketManager(coin="BTC", auto_switch_market=True, market_check_interval=5.0)

# Get current market
current = mm.discover_market()
if current:
    print(f"Current market: {current.slug}")
    print(f"  End date: {current.end_date}")
    print(f"  Timestamp: {mm._market_sort_key(current)}")
    print(f"  Tokens: {current.token_ids}")
    print(f"  Accepting orders: {current.accepting_orders}")
    mins, secs = current.get_countdown()
    print(f"  Countdown: {mins}m {secs}s")

print("\nWaiting 10 seconds to discover again...")
time.sleep(10)

# Discover again
new = mm.discover_market(update_state=False)
if new:
    print(f"\nNew market discovered: {new.slug}")
    print(f"  End date: {new.end_date}")
    print(f"  Timestamp: {mm._market_sort_key(new)}")
    print(f"  Tokens: {new.token_ids}")
    print(f"  Accepting orders: {new.accepting_orders}")
    mins, secs = new.get_countdown()
    print(f"  Countdown: {mins}m {secs}s")
    
    if current and new:
        print(f"\nComparison:")
        print(f"  Same slug? {current.slug == new.slug}")
        print(f"  Same tokens? {set(current.token_ids.values()) == set(new.token_ids.values())}")
        print(f"  Current timestamp: {mm._market_sort_key(current)}")
        print(f"  New timestamp: {mm._market_sort_key(new)}")
        print(f"  New > Current? {mm._market_sort_key(new) > mm._market_sort_key(current)}")
        print(f"  Would switch? {mm._should_switch_market(current, new)}")
else:
    print("\nNo new market found! This is the problem.")
    print("The market discovery isn't finding the next 15-min market.")
