


#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/models

# === Lint tooling (Checkstyle + PMD) — auto-fetched, user-only ===
# These are downloaded on demand so the repo stays lean (tools/ is gitignored).
# They are wired into tool_benchmark.py but results are NEVER sent to the model;
# they only enrich the user-facing VIOL column in the summary report.
CHECKSTYLE_VER="13.8.0"
PMD_VER="7.26.0"
CHECKSTYLE_JAR_URL="https://github.com/checkstyle/checkstyle/releases/download/checkstyle-${CHECKSTYLE_VER}/checkstyle-${CHECKSTYLE_VER}-all.jar"
CHECKSTYLE_CFG_URL="https://raw.githubusercontent.com/checkstyle/checkstyle/checkstyle-${CHECKSTYLE_VER}/src/main/resources/google_checks.xml"
PMD_ZIP_URL="https://github.com/pmd/pmd/releases/download/pmd_releases%2F${PMD_VER}/pmd-dist-${PMD_VER}-bin.zip"

setup_tools() {
    mkdir -p tools
    local ok=1

    # --- Checkstyle jar ---
    if [ ! -f tools/checkstyle.jar ]; then
        echo "Fetching checkstyle ${CHECKSTYLE_VER}..."
        if curl -fsSL "$CHECKSTYLE_JAR_URL" -o tools/checkstyle.jar && [ -s tools/checkstyle.jar ]; then
            echo "  checkstyle.jar downloaded."
        else
            echo "  WARNING: checkstyle download failed; lint will be javac-only."
            rm -f tools/checkstyle.jar
            ok=0
        fi
    fi

    # --- Checkstyle config (Google style) ---
    if [ ! -f tools/checkstyle_config.xml ] && [ -f tools/checkstyle.jar ]; then
        echo "Fetching checkstyle google_checks config..."
        if curl -fsSL "$CHECKSTYLE_CFG_URL" -o tools/checkstyle_config.xml && [ -s tools/checkstyle_config.xml ]; then
            echo "  checkstyle_config.xml downloaded."
        else
            echo "  WARNING: checkstyle config download failed."
            rm -f tools/checkstyle_config.xml
            ok=0
        fi
    fi

    # --- PMD distribution ---
    if [ ! -x tools/pmd/bin/pmd ]; then
        echo "Fetching PMD ${PMD_VER}..."
        if curl -fsSL "$PMD_ZIP_URL" -o tools/pmd-dist.zip && [ -s tools/pmd-dist.zip ]; then
            rm -rf tools/pmd
            unzip -q -o tools/pmd-dist.zip -d tools/
            # The zip contains a single top-level dir like pmd-bin-7.26.0; rename
            # it to the stable 'pmd' so PMD_BIN (tools/pmd/bin/pmd) resolves.
            local pmd_top
            pmd_top=$(find tools -maxdepth 1 -type d -name 'pmd-bin-*' | head -1)
            if [ -n "$pmd_top" ]; then
                mv "$pmd_top" tools/pmd
            fi
            rm -f tools/pmd-dist.zip
            if [ -x tools/pmd/bin/pmd ]; then
                echo "  PMD installed at tools/pmd/bin/pmd."
            else
                echo "  WARNING: PMD install layout unexpected; lint will be javac-only."
                ok=0
            fi
        else
            echo "  WARNING: PMD download failed; lint will be javac-only."
            rm -f tools/pmd-dist.zip
            ok=0
        fi
    fi

    if [ "$ok" -eq 1 ]; then
        echo "Lint tooling ready: checkstyle + PMD."
    else
        echo "Lint tooling partially unavailable — benchmark will run with javac-only lint."
    fi
}

setup_tools

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

