


#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/models

# Wipe results entirely for a fresh run (removes ALL per-model timestamp dirs,
# not just dotfiles). Guard so we never rm outside of results/.
if [ -d results ]; then
    echo " results exists — removing all contents for a clean start."
    find results -mindepth 1 -maxdepth 1 -exec rm -rf {} +
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

