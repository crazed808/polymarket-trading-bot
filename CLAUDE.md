# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A beginner-friendly Python trading bot for Polymarket with gasless transactions via Builder Program. Uses EIP-712 signing for orders, encrypted private key storage, and supports both the CLOB API and Relayer API.

## Common Commands

```bash
# Setup (first time)
pip install -r requirements.txt
cp .env.example .env  # Edit with your credentials
source .env

# Run quickstart example
python examples/quickstart.py

# Run full integration test
python scripts/full_test.py

# Run the bot
python scripts/run_bot.py              # Quick demo
python scripts/run_bot.py --interactive # Interactive mode

# Testing
pytest tests/ -v                        # Run all tests (89 tests)
pytest tests/test_utils.py -v           # Test utility functions
pytest tests/test_bot.py -v             # Test bot module
pytest tests/test_crypto.py -v          # Test encryption
pytest tests/test_signer.py -v          # Test EIP-712 signing
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         TradingBot                          │
│                        (bot.py)                             │
│  - High-level trading interface                             │
│  - Async order operations                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
┌─────────────┐ ┌───────────┐ ┌───────────────┐
│ OrderSigner │ │ ClobClient│ │ RelayerClient │
│ (signer.py) │ │(client.py)│ │ (client.py)   │
│             │ │           │ │               │
│ EIP-712     │ │ Order     │ │ Gasless       │
│ signatures  │ │ submission│ │ transactions  │
└──────┬──────┘ └─────┬─────┘ └───────────────┘
       │              │
       ▼              ▼
┌─────────────┐ ┌───────────┐
│ KeyManager  │ │  Config   │
│ (crypto.py) │ │(config.py)│
│             │ │           │
│ PBKDF2 +    │ │ YAML/ENV  │
│ Fernet      │ │ loading   │
└─────────────┘ └───────────┘
```

### Module Responsibilities

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `bot.py` | Main trading interface | `TradingBot`, `OrderResult` |
| `client.py` | API communication | `ClobClient`, `RelayerClient` |
| `signer.py` | EIP-712 signing | `OrderSigner`, `Order` |
| `crypto.py` | Key encryption | `KeyManager` |
| `config.py` | Configuration | `Config`, `BuilderConfig` |
| `utils.py` | Helper functions | `create_bot_from_env`, `validate_address` |

### Data Flow

1. `TradingBot.place_order()` creates an `Order` dataclass
2. `OrderSigner.sign_order()` produces EIP-712 signature
3. `ClobClient.post_order()` submits to CLOB with Builder HMAC auth headers
4. If gasless enabled, `RelayerClient` handles Safe deployment/approvals

## Key Patterns

- **Async methods**: All trading operations (`place_order`, `cancel_order`, `get_trades`) are async
- **Config precedence**: Environment vars > YAML file > defaults
- **Builder HMAC auth**: Timestamp + method + path + body signed with api_secret
- **Signature type 2**: Gnosis Safe signatures for Polymarket

## Configuration

Config loads from `config.yaml` or environment variables:

```python
# From environment
config = Config.from_env()

# From YAML
config = Config.load("config.yaml")

# With env overrides
config = Config.load_with_env("config.yaml")
```

Key fields:
- `safe_address`: Your Polymarket proxy wallet address
- `builder.api_key/api_secret/api_passphrase`: For gasless trading
- `clob.chain_id`: 137 (Polygon mainnet)

## Testing Notes

- Tests use `pytest` with `pytest-asyncio` for async
- Mock external API calls; never hit real Polymarket APIs in tests
- Test private key: `"0x" + "a" * 64`
- Test safe address: `"0x" + "b" * 40`
- YAML config values starting with `0x` must be quoted to avoid integer parsing

## Dependencies

- `eth-account>=0.13.0`: Uses new `encode_typed_data` API
- `web3>=6.0.0`: Polygon RPC interactions
- `cryptography`: Fernet encryption for private keys
- `pyyaml`: YAML config file support
- `python-dotenv`: .env file loading

## Polymarket API Context

- CLOB API: `https://clob.polymarket.com` - order submission/cancellation
- Relayer API: `https://relayer-v2.polymarket.com` - gasless transactions
- Token IDs are ERC-1155 identifiers for market outcomes
- Prices are 0-1 (probability percentages)
- USDC has 6 decimal places

**Important**: The `docs/` directory contains official Polymarket documentation. When implementing or debugging API features, always reference:
- `docs/developers/CLOB/` - CLOB API endpoints, authentication, orders
- `docs/developers/builders/` - Builder Program, Relayer, gasless transactions
- `docs/api-reference/` - REST API endpoint specifications

## For Beginners

Start with these files in order:
1. `examples/quickstart.py` - Simplest possible example
2. `examples/basic_trading.py` - Common operations
3. `src/bot.py` - Read the TradingBot class
4. `examples/strategy_example.py` - Custom strategy framework

## Quick Commands

```bash
# Check account status (USDC, positions, orders)
python scripts/quick_check.py status

# Check recent activity
python scripts/quick_check.py activity

# Run momentum strategy (recommended - BTC only)
python run_momentum_3coin.py --coin BTC --size 1.0 --log

# Run momentum strategy (BTC + ETH)
python run_momentum_3coin.py --coin BTC,ETH --size 1.0 --log

# Check tick recorder status
python scripts/tick_recorder.py status

# Run realistic backtest
python scripts/backtest_realistic.py
```

## Optimized Momentum Strategy - QUALITY_COMBO (Jan 2025)

**🏆 Comprehensively optimized via testing 12+ strategy variations**

**Quality_Combo Settings:**

| Setting | Value | Notes |
|---------|-------|-------|
| Momentum threshold | 15% in 30s | Main momentum window |
| Multi-timeframe | 8% in 10s | Short-term confirmation (CRITICAL) |
| Momentum quality | Max 3% volatility | Filters choppy signals (CRITICAL) |
| Trailing stop | Yes | Locks in profits from peak |
| Orderbook ratio | 2.0x | Strict depth filtering |
| Take profit | 6% | Achievable target |
| Stop loss | 3% | Wider = better (counterintuitive!) |
| Time stop | 80s | Exit if not hit TP/SL |
| Max entry spread | 3% | Ensures good execution |

**Performance (5-day BTC backtest, 2% spread cost):**
- **Quality_Combo: +1.50%/trade, 60.5% win rate** ⭐
- Multi-timeframe only: +1.29%/trade, 60.9% win rate
- Baseline (old): +0.57%/trade, 55.8% win rate

**Improvement: 2.6x better than baseline!**

**Key Insights:**
- Multi-timeframe confirmation is CRITICAL (+126% improvement)
- Quality filtering removes unreliable signals
- Trailing stops > fixed stops
- Wider SL (3% vs 2.5%) paradoxically improves results

**IMPORTANT:** Only run on BTC for best results. Strategy is now enabled by default in momentum.py.

## Momentum Strategy Notes

- Balances are in base units (6 decimals) - divide by 1,000,000
- `--test-run` flag stops after one complete trade cycle
- `--log` flag saves logs to `logs/momentum_*.log`
- Safety features: balance sync retry, position age protection, cooldowns

**Known issues fixed:**
- Balance=0 bug: Added retry logic for blockchain sync delay
- TP too high: Removed spread adjustment from TP calculation
- Spread costs: Strategy accounts for ~2% round-trip spread

## Strategy Context Files

When working on a specific strategy, **always read its context file first** for design decisions, known issues, and edit history:

- **Micro Snipe Bot**: [`MICRO_SNIPE_CONTEXT.md`](MICRO_SNIPE_CONTEXT.md) — covers `run_micro_snipe.py`, known issues (BAD FILL monitoring), and why certain fixes were rejected.

## Claude Code Preferences

- **Use Explore agent for searches**: When searching the codebase (finding implementations, understanding patterns, locating files), use the Task tool with `subagent_type=Explore` instead of multiple Glob/Grep calls. This is more efficient.
- **Haiku for simple tasks**: Use `model: "haiku"` for straightforward agent tasks to reduce cost/latency.
