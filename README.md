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
| **Planning loop** (`PROMPT_plan.md`) | **qwen3:8b** or **qwen2.5:7b** | Reliable, follows complex multi-step instructions, won't hallucinate or refuse |
| **Building loop** (`PROMPT_build.md`) | **qwen2.5-coder:7b** or **qwen3:1.7b** | Fast, task-focused, respects backpressure |
| **Subagent (scout/investigate)** | **qwen3:0.6b** or **granite3.2-vision:2b** | Cheap, fast, surprisingly capable for narrow tasks |
| **Subagent (build/tests)** | **qwen2.5-coder:7b** | Code-oriented, understands testing idioms |

### Models to Avoid in Ralph Loops

| Model | Why |
|---|---|
| **llama3.1:8b / latest** | Gives instructions instead of executing. Loop would spin forever. |
| **glm4:9b** | Refuses to do external tasks. Useless in an agent loop. |
| **opencoder (any)** | "I can't" — breaks autonomy. |
| **ornith:9b** | Truncated/partial output. Unreliable. |
| **qwen3.5 (any)** | Regression from qwen3. Confused, slow, hallucinates. |

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
- `prompt.info` — the input prompt
