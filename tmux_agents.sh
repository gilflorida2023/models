#!/usr/bin/env bash
# Launch the two-LLM Writer+Reviewer tmux harness for the simplesieve project.
#
# Layout (tmux session "simplesieve_pair"):
#   pane 0  writer    - Writer agent (edits Go code)
#   pane 1  reviewer  - Reviewer agent (critiques)
#   pane 2  monitor   - tails the latest reviewer notes
#
# Attach anytime with:  tmux attach -t simplesieve_pair
#
# Usage:
#   ./tmux_agents.sh                       # defaults writer=qwen2.5-coder:7b reviewer=qwen3:8b
#   ./tmux_agents.sh --writer MODEL --reviewer MODEL
set -euo pipefail
cd "$(dirname "$0")"

source venv/bin/activate 2>/dev/null || true

# Defaults
WRITER="qwen2.5-coder:7b"
REVIEWER="qwen3:8b"

# Parse simple --writer/--reviewer passthrough (everything else goes to agent_pair.py).
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --writer) WRITER="$2"; shift 2;;
    --reviewer) REVIEWER="$2"; shift 2;;
    *) EXTRA+=("$1"); shift;;
  esac
done

echo "Launching Writer+Reviewer harness (writer=$WRITER reviewer=$REVIEWER)"
exec python agent_pair.py --writer "$WRITER" --reviewer "$REVIEWER" "${EXTRA[@]}"
