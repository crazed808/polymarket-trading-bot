#!/bin/bash
# Polymarket Trading Bot Aliases
# Usage: source ~/polymarket-trading-bot/aliases.sh
# Or add to ~/.bashrc: source ~/polymarket-trading-bot/aliases.sh

# Change to bot directory
cd ~/polymarket-trading-bot 2>/dev/null || true

# === MAIN BOTS ===
# Sniper Bot - 100% win rate on all coins, buy at 95-97% in final 30s
alias sniper='python run_snipe.py --coin BTC,ETH,SOL,XRP --size 5.0'

# Momentum Bot - Quality_Combo optimized (+1.56%/trade), BTC+ETH, $1 size
alias momentum='python run_momentum_3coin.py --coin ALL --size 1.0 --log'

# === TEST RUNS (one trade cycle) ===
alias sniper-test='python run_snipe.py --coin BTC,ETH --size 5.0'
alias momentum-test='python run_momentum_3coin.py --coin ALL --size 1.0 --log --test-run'

# === ACCOUNT MANAGEMENT ===
alias balance='python scripts/quick_check.py status'
alias activity='python scripts/quick_check.py activity'
alias positions='python scripts/check_positions_detailed.py'
alias redeem='python scripts/redeem_positions.py'

# === LOGS ===
alias logs='ls -lth logs/ | head -20'
alias sniper-log='tail -f logs/snipe_*.log | tail -50'
alias momentum-log='tail -f logs/momentum_*.log | tail -50'

# === BACKTESTING ===
alias backtest='python scripts/backtest_realistic.py'
alias optimize='python scripts/optimize_momentum_comprehensive.py'
alias verify='python scripts/verify_backtest.py'

echo ""
echo "✅ Polymarket Bot Aliases Loaded!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🤖 BOTS:"
echo "  sniper          - Run sniper bot (ALL 4 coins, $5 each)"
echo "  momentum        - Run momentum bot (BTC+ETH, $1 each, +1.56%/trade)"
echo ""
echo "🧪 TESTING:"
echo "  sniper-test     - Test sniper (BTC, ETH only)"
echo "  momentum-test   - Test momentum (one trade cycle)"
echo ""
echo "💰 ACCOUNT:"
echo "  balance         - Check USDC balance"
echo "  positions       - Check open positions"
echo "  redeem          - Redeem all winning positions"
echo ""
echo "📊 LOGS:"
echo "  logs            - List recent log files"
echo "  sniper-log      - Tail sniper bot log"
echo "  momentum-log    - Tail momentum bot log"
echo ""
echo "📈 ANALYSIS:"
echo "  backtest        - Run backtest"
echo "  optimize        - Run optimization"
echo "  verify          - Verify backtest results"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
