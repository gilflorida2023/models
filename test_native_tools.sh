#!/usr/bin/env bash
# test_native_tools.sh — benchmark models on Ollama native tool calling API.
# Tests whether each model returns structured tool_calls for read + write.
# Usage: bash test_native_tools.sh [model_name]

set -euo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SELF_DIR"

MODELS=()
if [ $# -gt 0 ]; then
    MODELS=("$1")
else
    readarray -t MODELS < <(ollama list | awk 'NR>1 {print $1}')
fi

PROMPT='Read the file "calc.py" from the workspace. Then write "result.txt" with content "test passed".'

PASS=0
FAIL=0
TOTAL=0
RESULTS_FILE="native_tool_results.txt"
> "$RESULTS_FILE"

echo "=================================================================="
echo "  Native Tool Calling API Benchmark"
echo "  Prompt: read calc.py, write result.txt"
echo "  Expected: structured tool_calls with workspace.read + workspace.write"
echo "=================================================================="
echo ""

for model in "${MODELS[@]}"; do
    TOTAL=$((TOTAL + 1))
    name="${model//:/_}"
    outfile="native_${name}.txt"

    echo "--- Testing: $model ---"

    start=$(date +%s.%N)

    # POST to /api/chat with tools (same API ralph_agent.py uses)
    payload=$(jq -n \
        --arg model "$model" \
        --arg prompt "$PROMPT" \
        '{
            model: $model,
            messages: [{role: "user", content: $prompt}],
            tools: [
                {type: "function", function: {name: "workspace.read", description: "Read a file from the workspace", parameters: {type: "object", properties: {path: {type: "string"}}, required: ["path"]}}},
                {type: "function", function: {name: "workspace.write", description: "Write a file to the workspace", parameters: {type: "object", properties: {path: {type: "string"}, content: {type: "string"}}, required: ["path", "content"]}}}
            ],
            stream: false,
            options: {temperature: 0.3, num_ctx: 8192}
        }')

    curl -s -X POST http://localhost:11434/api/chat \
        -H "Content-Type: application/json" \
        -d "$payload" > "$outfile" 2>/dev/null || {
        echo "  $model FAIL (curl error)" >> "$RESULTS_FILE"
        FAIL=$((FAIL + 1))
        continue
    }

    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc -l)
    minutes=$(echo "$elapsed / 60" | bc)
    seconds=$(echo "$elapsed % 60" | bc)
    if (( minutes > 0 )); then
        time_str="${minutes}m $(printf '%.2f' $seconds)s"
    else
        time_str="$(printf '%.2f' $seconds)s"
    fi

    # Parse response
    msg=$(jq -r '.message // empty' "$outfile" 2>/dev/null || echo "")
    tool_calls=$(jq -r '.message.tool_calls // empty' "$outfile" 2>/dev/null || echo "")
    content=$(jq -r '.message.content // ""' "$outfile" 2>/dev/null || echo "")

    error=$(jq -r '.error // empty' "$outfile" 2>/dev/null || echo "")
    if [ -n "$error" ]; then
        echo "  $model FAIL (api_error: $error)" >> "$RESULTS_FILE"
        FAIL=$((FAIL + 1))
        continue
    fi

    # Count tool calls
    tc_count=$(echo "$tool_calls" | jq 'length' 2>/dev/null)
    tc_count=${tc_count:-0}

    # Check for workspace.read
    has_read=false
    echo "$tool_calls" | jq -e '.[] | select(.function.name == "workspace.read")' >/dev/null 2>&1 && has_read=true

    # Check for workspace.write
    has_write=false
    echo "$tool_calls" | jq -e '.[] | select(.function.name == "workspace.write")' >/dev/null 2>&1 && has_write=true

    # Check valid JSON arguments
    valid_json=true
    while IFS= read -r tc; do
        [ -z "$tc" ] && continue
        args=$(echo "$tc" | jq -r '.function.arguments // empty' 2>/dev/null || echo "")
        if [ -z "$args" ] || [ "$args" = "null" ]; then
            valid_json=false
            break
        fi
        if ! echo "$args" | jq '.' >/dev/null 2>&1; then
            valid_json=false
            break
        fi
    done < <(echo "$tool_calls" | jq -c '.[]' 2>/dev/null || echo "")

    # Check for refusal keywords (only if no tool calls)
    refusal=false
    if [ "$tc_count" -eq 0 ]; then
        for word in "sorry" "can't" "cannot" "unable" "refuse"; do
            if echo "$content" | grep -qi "$word"; then
                refusal=true
                break
            fi
        done
    fi

    # Score
    score="FAIL"
    reasons=()
    if [ "$refusal" = true ]; then
        reasons+=("REFUSED")
    fi
    if [ "$tc_count" -eq 0 ]; then
        reasons+=("no_tool_calls")
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

    if [ ${#reasons[@]} -eq 0 ]; then
        score="PASS"
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi

    IFS=", "
    reason_str="${reasons[*]}"
    unset IFS
    printf "  %-30s %s  tools=%d  %s  %s\n" "$model" "$score" "$tc_count" "$time_str" "${reason_str:+($reason_str)}"

    echo "$model | $score | tools=$tc_count | read=$has_read | write=$has_write | json_ok=$valid_json | refusal=$refusal | time=$time_str | ${reason_str:-ok}" >> "$RESULTS_FILE"

    # Save first 50 lines of output for inspection
    {
        echo "=== Model: $model ==="
        echo "=== Time: $time_str ==="
        echo "=== Score: $score ==="
        echo "=== tool_calls ==="
        echo "$tool_calls" | jq '.' 2>/dev/null || echo "(empty)"
        echo "=== content ==="
        echo "$content" | head -10
        echo ""
        echo "---"
    } > "native_${name}_summary.txt"
done

echo ""
echo "=================================================================="
echo "  Results: $PASS pass / $FAIL fail / $TOTAL total"
echo "  Detailed: $RESULTS_FILE"
echo "  Summaries: native_*_summary.txt"
echo "  Raw output: native_*.txt"
echo "=================================================================="

echo ""
echo "=== Summary Table ==="
echo "Model | Result | Tool Calls | Read | Write | Valid JSON | Refusal | Time"
echo "------|--------|------------|------|-------|------------|---------|-----"
column -t -s '|' "$RESULTS_FILE" 2>/dev/null || cat "$RESULTS_FILE"
