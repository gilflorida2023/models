


#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/models

# === Stop any previous benchmark instance (fresh run supersedes old) ===
# A lingering tool_benchmark.py holds the Qdrant lock and may still be writing
# into results/; terminate it before we wipe results and start a clean run.
# NOTE: this kills the OLD run, not this script (firstrun.sh's own cmdline
# does not contain "tool_benchmark.py", so pkill -f won't match itself).
prev_pids=$(pgrep -f "tool_benchmark.py" || true)
if [ -n "$prev_pids" ]; then
    echo "Stopping previous benchmark instance(s): $prev_pids"
    pkill -TERM -f "tool_benchmark.py" || true
    # Wait up to ~10s for a clean exit (releases the Qdrant lock).
    for _ in $(seq 1 20); do
        pgrep -f "tool_benchmark.py" >/dev/null || break
        sleep 0.5
    done
    # Force-kill only if still alive.
    if pgrep -f "tool_benchmark.py" >/dev/null; then
        pkill -KILL -f "tool_benchmark.py" || true
        sleep 1
    fi
fi

# === Clear a stale Qdrant lock left by a crashed/killed instance ===
# Safe only because no benchmark process can be alive here (we just stopped
# them above and haven't launched the new one yet). If a live instance is ever
# detected, we leave the lock alone to avoid corrupting it.
if [ -f qdrant_data/.lock ] && ! pgrep -f "tool_benchmark.py" >/dev/null; then
    echo "Removing stale Qdrant lock (qdrant_data/.lock)."
    rm -f qdrant_data/.lock
fi

# Wipe results entirely for a fresh run (removes ALL per-model timestamp dirs,
# not just dotfiles). Guard so we never rm outside of results/.
if [ -d results ]; then
    echo " results exists — removing all contents for a clean start."
    find results -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    ls -l results
else
    mkdir -p results
fi

# Set up venv if missing.
if ! [ -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt

# Run the benchmark (no need for a separate --clean since we wiped results above).
python tool_benchmark.py  2>&1 | tee tool_benchmark.log

