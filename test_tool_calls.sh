#!/usr/bin/env bash
# test_tool_calls.sh — benchmark models on ##mcp_tool format compliance.
# Tests whether each model can output properly formatted tool calls.
# Usage: bash test_tool_calls.sh [model_name]
#   Without args, tests all installed Ollama models.
#   With a model name, tests only that model.

set -euo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SELF_DIR"

MODELS=()
if [ $# -gt 0 ]; then
    MODELS=("$1")
else
    readarray -t MODELS < <(ollama list | awk 'NR>1 {print $1}')
fi

# The prompt: asks for a simple two-step tool sequence
PROMPT=$(cat << 'PROMPT_EOF'
You have access to tools via lines starting with ##mcp_tool.

## Instructions
1. Read the file "calc.py" from the workspace using workspace.read.
2. Write a new file "result.txt" with the content "test passed" using workspace.write.

Output each tool call on its own line starting with ##mcp_tool.
Do NOT explain — just output the tool calls.
PROMPT_EOF
)

PASS=0
FAIL=0
TOTAL=0
RESULTS_FILE="tool_call_results.txt"
> "$RESULTS_FILE"

echo "=================================================================="
echo "  Tool Call Format Benchmark"
echo "  Prompt: read calc.py, write result.txt"
echo "  Expected format: ##mcp_tool workspace.read {...}"
echo "                   ##mcp_tool workspace.write {...}"
echo "=================================================================="
echo ""

for model in "${MODELS[@]}"; do
    TOTAL=$((TOTAL + 1))
    name="${model//:/_}"
    outfile="tool_${name}.txt"

    echo "--- Testing: $model ---"

    start=$(date +%s.%N)

    # Run model with prompt (non-streaming, no thinking)
    ollama run "$model" --think=false <<< "$PROMPT" 2>/dev/null > "$outfile"

    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc -l)
    minutes=$(echo "$elapsed / 60" | bc)
    seconds=$(echo "$elapsed % 60" | bc)
    if (( minutes > 0 )); then
        time_str="${minutes}m $(printf '%.2f' $seconds)s"
    else
        time_str="$(printf '%.2f' $seconds)s"
    fi

    # Analyze output
    output=$(cat "$outfile")

    # Count ##mcp_tool lines
    tool_lines=$(echo "$output" | grep -c '##mcp_tool' 2>/dev/null || echo 0)

    # Check for read tool
    has_read=false
    echo "$output" | grep -q '##mcp_tool workspace.read' && has_read=true

    # Check for write tool
    has_write=false
    echo "$output" | grep -q '##mcp_tool workspace.write' && has_write=true

    # Check for valid JSON after tool name
    valid_json=true
    while IFS= read -r line; do
        if [[ "$line" =~ ^##mcp_tool\ ([a-z._]+)\ (.*) ]]; then
            args="${BASH_REMATCH[2]}"
            if ! echo "$args" | python3 -c "import json,sys; json.loads(sys.stdin.read())" 2>/dev/null; then
                valid_json=false
            fi
        fi
    done <<< "$output"

    # Check for refusal keywords
    refusal=false
    for word in "sorry" "can't" "cannot" "unable" "refuse" "I'm"; do
        if echo "$output" | grep -qi "$word"; then
            refusal=true
            break
        fi
    done

    # Check for instruction-giving (talks about what to do instead of doing it)
    instruction_only=false
    if [ "$tool_lines" -eq 0 ] && echo "$output" | grep -qiE "you (should|need to|can |could |would )"; then
        instruction_only=true
    fi

    # Score
    score="FAIL"
    reasons=()
    if [ "$refusal" = true ]; then
        reasons+=("REFUSED")
    fi
    if [ "$instruction_only" = true ]; then
        reasons+=("INSTRUCTIONS_ONLY")
    fi
    if [ "$has_read" = false ]; then
        reasons+=("missing_read")
    fi
    if [ "$has_write" = false ]; then
        reasons+=("missing_write")
    fi
    if [ "$valid_json" = false ]; then
        reasons+=("bad_json")
    fi
    if [ "$tool_lines" -eq 0 ]; then
        reasons+=("no_tool_calls")
    fi

    if [ ${#reasons[@]} -eq 0 ]; then
        score="PASS"
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi

    # Print summary
    IFS=", " 
    reason_str="${reasons[*]}"
    unset IFS
    printf "  %-30s %s  tools=%d  %s  %s\n" "$model" "$score" "$tool_lines" "$time_str" "${reason_str:+($reason_str)}"

    # Record to results file
    echo "$model | $score | tools=$tool_lines | read=$has_read | write=$has_write | json_ok=$valid_json | refusal=$refusal | time=$time_str | ${reason_str:-ok}" >> "$RESULTS_FILE"

    # Save first 5 lines of output for inspection
    {
        echo "=== Model: $model ==="
        echo "=== Time: $time_str ==="
        echo "=== Score: $score ==="
        echo "=== Output ==="
        head -20 "$outfile"
        echo ""
        echo "---"
    } > "tool_${name}_summary.txt"
done

echo ""
echo "=================================================================="
echo "  Results: $PASS pass / $FAIL fail / $TOTAL total"
echo "  Detailed: $RESULTS_FILE"
echo "  Summaries: tool_*_summary.txt"
echo "  Raw output: tool_*.txt"
echo "=================================================================="

# Print summary table
echo ""
echo "=== Summary Table ==="
echo "Model | Result | Tool Calls | Read | Write | Valid JSON | Refusal | Time"
echo "------|--------|------------|------|-------|------------|---------|-----"
column -t -s '|' "$RESULTS_FILE" 2>/dev/null || cat "$RESULTS_FILE"
