#!/bin/bash
cd ~/polymarket-trading-bot
mkdir -p logs

tmux new-session -d -s polymarket
tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v

# Run with tee to log each coin
tmux select-pane -t 0
tmux send-keys "cd ~/polymarket-trading-bot && source venv/bin/activate && PYTHONPATH=. python3 run_flash_crash.py --coin BTC --size 1.0 2>&1 | tee logs/btc.log" C-m

tmux select-pane -t 1
tmux send-keys "cd ~/polymarket-trading-bot && source venv/bin/activate && PYTHONPATH=. python3 run_flash_crash.py --coin ETH --size 1.0 2>&1 | tee logs/eth.log" C-m

tmux select-pane -t 2
tmux send-keys "cd ~/polymarket-trading-bot && source venv/bin/activate && PYTHONPATH=. python3 run_flash_crash.py --coin SOL --size 1.0 2>&1 | tee logs/sol.log" C-m

tmux select-pane -t 3
tmux send-keys "cd ~/polymarket-trading-bot && source venv/bin/activate && PYTHONPATH=. python3 run_flash_crash.py --coin XRP --size 1.0 2>&1 | tee logs/xrp.log" C-m

tmux attach-session -t polymarket
