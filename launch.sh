#!/bin/bash
# Launch or resume the FlowMatch-PdM experiment inside tmux.
# Usage: bash launch.sh
# To watch: tmux attach -t flowmatch

set -euo pipefail

SESSION="flowmatch"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
mkdir -p logs results

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Attaching..."
  tmux attach -t "$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -x 220 -y 50

# Send the RUL pivot execution into the tmux pane
tmux send-keys -t "$SESSION" \
  "cd $SCRIPT_DIR && source /home/buddhiw/miniconda3/etc/profile.d/conda.sh && conda activate flowmatch_pdm && bash final_execution.sh 2>&1 | tee logs/final_execution_\$(date +%Y%m%d_%H%M%S).log" \
  Enter

echo "Launched in tmux session '$SESSION'."
echo "Attach with:  tmux attach -t $SESSION"
echo "Detach with:  Ctrl-B then D"
