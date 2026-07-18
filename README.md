# Model Benchmark: simplesieve Task

**29 models tested** across 10 model families, evaluated on a practical software engineering task.

## Task

Each model received a prompt asking it to:

1. Clone `https://github.com/gilflorida2023/simplesieve`
2. Build the Go application
3. Run `./simplesieve -c -limit 1e6`
4. Share the result

**Correct answer:** `78498` — the number of primes ≤ 1,000,000 (π(10⁶))

## Methodology

- All models run locally via `ollama run --think=false` (non-thinking mode)
- Prompt fed from `prompt.info` (63-89 tokens)
- Timing measured from start to finish including prompt eval + generation
- Output captured with `ansifilter` to strip ANSI escape codes
- Detailed log in `test_models.sh`

## Results

### Winners (correct answer, good understanding)

| Model | Size | Time | Tok/s | Notes |
|---|---|---|---|---|
| **qwen3:0.6b** | 0.75B | **1.43s** | 138 | Fastest. Perfectly direct: "Result: 78498" |
| **granite3.2-vision:2b** | 2.5B | **3.40s** | 58 | Tiny model, instant correct answer |
| **qwen3:1.7b** | 2.0B | **5.21s** | 75 | Clear, correct, efficient |
| **qwen2.5-coder:7b** | 7.6B | 13.09s | 23 | Concise and correct |
| **qwen3:8b** | 8.2B | 20.48s | 20 | Clear explanation, correct |
| **qwen2.5:7b** | 7.6B | 21.19s | 23 | "Expected output: 78498" |
| **qwen2.5:3b** | 3.1B | 12.70s | 48 | Correct but wordy |
| **deepseek-r1:7b** | 7.6B | 31.46s | 22 | Correct, thorough guidance |
| **qwen3:4b** | 4.0B | 30.83s | 37 | Correct after long internal debate |
| **qwen3.5:9b** | 9.7B | 33.38s | 18 | Mathematically confirmed correct |
| **deepseek-r1:8b** | 8.2B | 1m14s | 20 | Correct but very slow |
| **qwen3-vl:4b** | 4.4B | 55.83s | 35 | Correct, hallucinated execution details |
| **qwen3-vl:8b** | 8.8B | 59.09s | 21 | Correct, slow |
| **qwen3-vl:2b** | 2.1B | 1m13s | 57 | Correct after massive confused monologue |

### Struggled (partial or confused)

| Model | Size | Time | Issue |
|---|---|---|---|
| qwen2.5:1.5b | 1.5B | 5.63s | Vague: "around seven thousand eight hundred ninety-eight items" |
| deepseek-r1:1.5b | 1.8B | 25.01s | Garbled Go/Python code, confused |
| qwen3.5:2b | 2.3B | 24.16s | Got number but called prompt "inconsistent" |
| qwen3.5:4b | 4.7B | 29.06s | Correct count but claimed app doesn't print to stdout |
| qwen3.5:0.8b | 0.87B | 52.63s | Extremely confused, hallucinated nonsense code, thought 78498 was prime |

### Losers (refused or failed to answer)

| Model | Size | Time | Failure |
|---|---|---|---|
| **glm4:9b** | 9.4B | 26.30s | "I am unable to directly access external websites" — no answer |
| **llama3.1:8b** | 8.0B | 25.87s | Gave instructions only, never stated the result |
| **llama3.1:latest** | 8.0B | 24.84s | Gave instructions only, never stated the result |
| **opencoder:1.5b** | 1.9B | 6.06s | "I can't clone, build or run" — no answer |
| **opencoder:8b** | 7.8B | 17.71s | "Sorry, I can't clone a repository" — no answer |
| **ornith:9b** | 9.0B | 11.24s | Echoed bash commands, no result output |
| **qwen2.5:0.5b** | 0.49B | 1.49s | "I'm sorry, but I can't assist with that" |
| **qwen2.5-coder:0.5b** | 0.49B | 1.20s | "I'm sorry, but I can't assist with that request" |
| **qwen2.5-coder:1.5b** | 1.5B | 5.63s | Rambled about email filters |
| **qwen2.5-coder:3b** | 3.1B | 2.67s | "I'm sorry, but I can't help you with that" |

### Visual Summary

```
Correct + Fast  ████████████████████████████████████████  qwen3:0.6b, granite3.2-vision:2b, qwen3:1.7b
Correct + Slow  ████████████████████████████████████████  deepseek-r1:7b/8b, qwen3-vl:4b/8b/2b
Partial/Wrong   ████████████████                         qwen3.5:0.8b/2b/4b, qwen2.5:1.5b, ds-r1:1.5b
Refused/Failed  ████████████████████████████████████████  glm4, llama3.1, opencoder, ornith, qwen2.5-coder
```

## Tool Calling API Benchmark

**12/29 models PASS** — return correct structured `tool_calls` via Ollama's native `/api/chat` with `tools` parameter.

**Methodology:** POST to `/api/chat` with two tool definitions (`workspace.read`, `workspace.write`) and prompt: "Read calc.py then write result.txt with 'test passed'". Check for `message.tool_calls` containing both operations with valid JSON arguments. Models retested — non-deterministic results noted.

### Pass (correct read + write)

| Model | Size | Calls | Time | Notes |
|---|---|---|---|---|
| **qwen3:0.6b** | 0.75B | 2 | **4.46s** | Fastest pass. Non-deterministic — occasionally returns 0 calls. |
| qwen2.5:0.5b | 0.49B | 2 | 1.60s | Fastest of all but tiny |
| qwen2.5:1.5b | 1.5B | 2 | 2.06s | |
| qwen2.5:3b | 3.1B | 2 | 3.42s | |
| qwen3:1.7b | 2.0B | 2 | 6.98s | |
| qwen2.5:7b | 7.6B | 2 | 7.06s | |
| llama3.1:latest | 8.0B | 2 | 7.18s | |
| llama3.1:8b | 8.0B | 2 | 7.30s | |
| qwen3:8b | 8.2B | 2 | 16.15s | Slower — more thinking overhead |
| qwen3-vl:8b | 8.8B | 2 | 19.75s | |
| qwen3:4b | 4.0B | 2 | 23.01s | Surprisingly slow for its size |
| qwen3-vl:2b | 2.1B | 2 | 11.58s | |

### Fail (no or partial tool calls)

| Model | Result | Calls | Failure |
|---|---|---|---|
| deepseek-r1:8b | FAIL | 0 | XML thinking response, no tool_calls |
| deepseek-r1:7b | FAIL | 0 | REFUSED + no tool_calls |
| deepseek-r1:1.5b | FAIL | 0 | No tool_calls |
| ornith:9b | FAIL | 1 | Only read, no write |
| opencoder:8b | ERROR | — | API: "does not support tools" |
| opencoder:1.5b | ERROR | — | API: "does not support tools" |
| granite3.2-vision:2b | FAIL | 0 | No tool_calls |
| qwen2.5-coder:3b | FAIL | 0 | No tool_calls |
| qwen2.5-coder:7b | FAIL | 0 | No tool_calls |
| qwen2.5-coder:1.5b | FAIL | 0 | No tool_calls |
| qwen2.5-coder:0.5b | FAIL | 0 | No tool_calls |
| qwen3-vl:4b | FAIL | 0 | No tool_calls |
| qwen3.5:9b | FAIL | 1 | Only read, never write |
| qwen3.5:2b | FAIL | 1 | Only read, never write |
| qwen3.5:4b | FAIL | 1 | Only read, never write |
| qwen3.5:0.8b | FAIL | 1 | Only read, never write |
| glm4:9b | FAIL | 0 | REFUSED, no tool_calls |

### Key Observations

**1. qwen3:0.6b is the best tiny agent model.** Fastest pass at 4.46s, and also fastest simplesieve answer at 1.43s. However, it is non-deterministic — retesting showed 0 tool_calls on one run. Mitigation: the Ralph loop's text `##mcp_tool` fallback catches these misses.

**2. qwen3.5 family has a silent write bug.** All four sizes return exactly 1 tool call (workspace.read) but never emit workspace.write. This means they'd fail any multi-step task in a single turn. The loop can recover via follow-up turns, but it's a real regression from qwen3.

**3. qwen2.5-coder family does NOT support native tools.** Despite Ollama listing them with `tools` capability, they return no `tool_calls`. The text `##mcp_tool` fallback is essential for these models. The simplesieve benchmark showed qwen2.5-coder:7b as correct — but it would need the text format path in an agent loop.

**4. deepseek-r1 family doesn't emit tool_calls.** The XML thinking format prevents structured tool calling. For Ralph loops, deepseek is only usable in planning mode (where the loop processes text output, not tool_calls).

**5. llamas work but add no speed advantage.** llama3.1:8b and latest both pass at ~7s — competitive with qwen2.5:7b but slower than qwen2.5:3b. Their fatal flaw from the simplesieve test (giving instructions instead of answers) still applies.

### Combined Model Selection (Simplesieve + Tool Calling)

```
Model                  Simplesieve  Native Tools  Verdict
──────                 ───────────  ────────────  ──────
qwen3:0.6b             ✅ 1.43s     ✅ 4.46s       BEST AGENT — fastest in both
qwen3:1.7b             ✅ 5.21s     ✅ 6.98s       Good fallback, more reliable
qwen2.5:7b             ✅ 21.19s    ✅ 7.06s       Best planner — correct, consistent
qwen3:8b               ✅ 20.48s    ✅ 16.15s      Deep reasoning, slow but thorough
llama3.1:8b            ❌ instr.    ✅ 7.30s       🚫 Avoid — loops forever
qwen2.5-coder:7b       ✅ 13.09s    ❌ no tools    Needs text fallback path
granite3.2-vision:2b   ✅ 3.40s     ❌ no tools    Needs text fallback path
qwen3.5:4b             ⚠️ 29.06s   ❌ 1-call bug  🚫 Avoid — broken multi-step
```

## Key Findings

### 1. Size != Intelligence
- **qwen3:0.6b** (0.75B parameters) outperformed **glm4:9b**, **llama3.1:8b**, and **opencoder:8b** — models 10-15x larger that either refused or failed
- **granite3.2-vision:2b** (2.5B) was faster and more accurate than every model above 8B except Qwen

### 2. Qwen Dominates
- Qwen3 and Qwen2.5 families were the most reliable across all sizes
- qwen3:0.6b, qwen3:1.7b, qwen2.5-coder:7b, qwen2.5:7b all delivered the correct answer directly
- Qwen3.5 series (0.8b, 2b, 4b) struggled — regression from qwen3

### 3. Refusal Is a Real Problem
- 6 models (glm4, opencoder×2, qwen2.5-coder×2, qwen2.5:0.5b) refused to engage
- This is catastrophic for agentic loops — if an agent refuses mid-loop, the pipeline breaks
- **Always test models for "will it do the thing?" before committing to a loop**

### 4. Llama3.1's Fatal Flaw
- Both llama3.1:8b and llama3.1:latest gave elaborate **instructions** but never actually **answered**
- In a Ralph loop, this means the agent talks about what to do instead of doing it — the loop would spin forever producing no output

### 5. Thinking Overhead Is Real
- deepseek-r1 models were correct but their built-in thinking consumed 60-70% of tokens
- deepseek-r1:8b took 1m14s at 20 tok/s vs qwen3:8b at 20.48s at 20 tok/s
- For iterative loops, thinking models only make sense in planning mode, not per-task execution

### 6. Tiny Models for Subagents
- qwen3:0.6b and granite3.2-vision:2b prove that models under 3B can handle simple agentic tasks
- Ideal for Ralph-style subagent scouting/investigation where cost and speed matter

## Recommendations for Ralph Practitioners

The [Ralph Wiggum Technique](https://github.com/ghuntley/how-to-ralph-wiggum) relies on agentic loops with tight context, specs-driven planning, and backpressure through tests. Here's how these results inform model selection:

### Ralph Loop Role → Recommended Model

| Role | Recommended | Why |
|---|---|---|
| **Planning loop** (`PROMPT_plan.md`) | **qwen2.5:7b** or **qwen3:8b** | Reliable, follows complex multi-step instructions, native tools pass |
| **Building loop** (`PROMPT_build.md`) | **qwen3:1.7b** or **qwen3:0.6b** | Fast, task-focused, native tools pass, no refusal |
| **Subagent (scout/investigate)** | **qwen3:0.6b** | Fastest native tools pass (4.46s), tiny (0.75B) |
| **Subagent (build/tests)** | **qwen3:1.7b** | Native tools pass, better instruction following than 0.6b |

### Models to Avoid in Ralph Loops

| Model | Why |
|---|---|
| **llama3.1:8b / latest** | Gives instructions instead of executing. Loop would spin forever. |
| **glm4:9b** | Refuses to do external tasks. Useless in an agent loop. |
| **opencoder (any)** | "I can't" — breaks autonomy. |
| **ornith:9b** | Truncated/partial output. Unreliable. |
| **qwen3.5 (any)** | Single-call bug (only reads, never writes). Unreliable for multi-step. |
| **qwen2.5-coder (any)** | No native tool calling support. Needs text-only fallback. |
| **granite3.2-vision:2b** | No native tool calling support despite fast simplesieve time. |

### Ralph-Specific Takeaways

**1. "Don't assume not implemented" — test for it**
   The models that succeeded (Qwen family) did so by trusting the prompt's number over their own parametric knowledge. This maps directly to Ralph's principle: the agent must trust spec files over assumptions.

**2. Backpressure filters bad models**
   Models that gave instructions instead of answers (llama3.1) would pass a conversational eval but fail in a Ralph loop that expects commits. **Your loop's backpressure (tests, builds) is your best model eval.**

**3. Planning vs. Building needs different models**
   Planning benefits from slower, more thorough models (deepseek-r1:7b for complex gap analysis). Building benefits from fast, deterministic models (qwen3:1.7b for task execution). **Don't use one model for both.**

**4. Tiny models are viable subagents**
   qwen3:0.6b at 1.43s proved that a 0.75B model can handle a complete task loop. For Ralph's "up to 500 parallel Sonnet subagents" pattern, using tiny models at ~$0 cost makes economic sense — but verify they don't silently fail.

**5. Read the prompt, not the weights**
   The best models didn't just know 78498 — they read it from the prompt and echoed it. In Ralph loops, the prompt *is* the spec. Models that hallucinate or override prompt content with training data will produce incorrect implementations.

## Raw Data

All model outputs are archived in this directory:
- `{model_name}.txt` — full output with timing
- `thinking.{model_name}.txt` — thinking-mode variants (not analyzed here)
- `test_models.sh` — the test harness
- `test_tool_calls.sh` — text-format `##mcp_tool` benchmark
- `test_native_tools.sh` — native tool calling API (`/api/chat` with `tools`) benchmark
- `native_{model_name}.txt` — raw API responses from native tools benchmark
- `native_{model_name}_summary.txt` — per-model summaries
- `native_tool_results.txt` — consolidated results table
- `tool_call_results.txt` — text-format results
- `prompt.info` — the input prompt

---

## Usage (hashprime Tool-Calling Benchmark)

The current benchmark (`tool_benchmark.py`) challenges each model to write a
single-file `hashprime.java` (Sieve of Eratosthenes) and verifies correctness at
four scales — N=11, N=12, one million (1,000,000) and ten million (10,000,000) —
against the authoritative OEIS A000040 manifest (`manifests/A000040.json`,
`ascii_integer_lf` formatting: each prime followed by a single newline).

### Prerequisites

```bash
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` provides: `numpy`, `tqdm`, `chonkie`, `qdrant-client`, `requests`.

Both commands require a **local Ollama** running with the target chat models
pulled. Embedding-dependent features (thinking-quality `TSCORE` and the
`search_solutions` retrieval tool) use **Ollama `nomic-embed-text`** via
`/api/embed`, plus the local Qdrant store at `qdrant_data/`. Pull it first:

```bash
ollama pull nomic-embed-text
```

### Ingest reference corpora — `ingest_corpora.py`

Builds the Qdrant index used by `search_solutions` and thinking-quality scoring.
No CLI arguments; honors the `INCLUDE_HELLO_ALGO` environment variable (off by
default because the hello-algo textbook is 801 files and slow to embed).

```bash
# Default: ingests linux_server + hashprime_solutions only
.venv/bin/python ingest_corpora.py

# Include the hello-algo textbook (very slow — run overnight)
INCLUDE_HELLO_ALGO=true nohup .venv/bin/python ingest_corpora.py > ingest_night.log 2>&1 &
```

- Local Qdrant store: `qdrant_data/`
- Collections: `hashprime_solutions`, `linux_server`, `hello_algo`
- Embeds via Ollama `nomic-embed-text` (768-dim, cosine distance)

### Run the tool-calling benchmark — `tool_benchmark.py`

No CLI arguments. At startup it queries live Ollama (`/api/tags`), then
benchmarks every chat-capable model. Each model gets the hashprime task with
tools (`write_file`, `compile_java`, `run_and_hash`, `submit_answer`,
`search_solutions`); generated code is compiled and executed at all four scales.

```bash
.venv/bin/python tool_benchmark.py
```

Outputs (under `results/`):
- `ranking.md` — human-readable ranking table
- `ranking.json` — machine-readable results
- `results/<model>/<timestamp>/hashprime.java` — each model's submission

The `1E6` and `1E7` columns confirm the sieve scales: the hash of the prime
list for N=1,000,000 and N=10,000,000 must match the A000040 manifest
(`4883963d…` and `36d61978…` respectively). This neutralizes any
"don't write a file" micro-optimization advantage, since at scale the algorithm
dominates runtime.
