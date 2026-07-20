#!/usr/bin/env python3
"""
Two-LLM collaboration harness (Writer + Reviewer) on a shared coding project,
managed with tmux.

Topology: a WRITER agent edits the code; a REVIEWER agent critiques it. The
orchestrator (this process) hands the reviewer's notes back to the writer each
round. tmux is used as the agent runtime/manager: each agent runs as its own
long-lived process in its own tmux pane, so a crash in one agent does not take
down the other, and `tmux capture-pane` gives a live, attachable view of every
agent (the canonical "spawn and manage multiple agents" pattern).

Agents talk to the orchestrator through a file mailbox in the shared workspace:
  workspace/mailbox/<role>_task.json    -> orchestrator -> agent
  workspace/mailbox/<role>_resp.json    -> agent -> orchestrator
  workspace/mailbox/review.json         -> latest reviewer critique (for the writer)

The shared project is cloned into workspace/<project>/ (gitignored). For the
default target (gilflorida2023/simplesieve) correctness is anchored on:
  go build && ./simplesieve --limit 1e6 -c  ->  78498

LLM calls go through the existing Ollama /api/chat endpoint (reused from
tool_benchmark.call_ollama) with a Go-appropriate tool schema
(write_file / run_shell / submit). tmux only manages the agent *processes*.

Usage:
  python agent_pair.py                       # defaults: writer=qwen2.5-coder:7b, reviewer=qwen3:8b
  python agent_pair.py --writer MODEL --reviewer MODEL
  python agent_pair.py --project git@...:user/repo.git
  python agent_pair.py --no-tmux             # run agents in-process (no tmux panes)
  python agent_pair.py --attach              # after launch, print how to tmux attach
"""

import os
import sys
import re
import json
import time
import shutil
import subprocess
import argparse
import signal

# Reuse the existing Ollama client + capability detection from the benchmark.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tool_benchmark as tb  # call_ollama, get_model_capabilities

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.join(BASE_DIR, "workspace")
MAILBOX = os.path.join(WORKSPACE, "mailbox")

DEFAULT_PROJECT_URL = "https://github.com/gilflorida2023/simplesieve"
DEFAULT_PROJECT_NAME = "simplesieve"

# Canonical correctness anchor for the default project.
CORRECT_COUNT_1E6 = "78498"

TMUX_SESSION = "simplesieve_pair"
MAX_TURNS = 12
TURN_POLL = 0.5          # seconds between mailbox polls
AGENT_TIMEOUT = 600      # per-agent call timeout (s)
HANDSHAKE_TIMEOUT = 30   # wait for an agent pane to come up

# Set by main(); False means agents run in-process (no tmux panes).
USE_TMUX = True


# === Go-appropriate tool schema (reused pattern from tool_benchmark.TOOL_SCHEMA) ===
GO_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write (or overwrite) a file in the project working directory with "
                "the given content. Use this to create or edit Go source files "
                "(e.g. 'internal/sieve/sieve.go', 'main.go'). The path is relative "
                "to the project root."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                             "description": "Relative path, e.g. 'internal/sieve/sieve.go'"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command in the project directory (e.g. 'go build ./...', "
                "'go vet ./...', './simplesieve --limit 1e6 -c'). Returns stdout+stderr. "
                "Use it to compile, test, and measure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Submit the final solution. Call this only when the project builds and "
                "produces correct output (./simplesieve --limit 1e6 -c prints 78498)."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

KNOWN_TOOLS = {"write_file", "run_shell", "submit"}


def extract_text_tool_calls(text):
    """Some models (e.g. small coders) emit tool calls as JSON inside a markdown
    fence or as multiple top-level JSON objects, instead of using the native
    tool_calls API. Extract a list shaped like Ollama tool_calls:
      [{"function": {"name": ..., "arguments": {...}}}, ...]
    Tolerant of: a single fenced object, several fenced objects, or a JSON array
    of objects. Skips anything whose 'name' is not a known tool. Returns [].
    """
    if not text:
        return []
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()

    # If the whole thing is a JSON array, parse and flatten.
    try:
        arr = json.loads(cleaned)
        if isinstance(arr, list):
            objs = arr
        else:
            objs = [arr]
    except Exception:
        # Not a single JSON value: brace-match each top-level {...} object.
        objs = []
        i = 0
        n = len(cleaned)
        while i < n:
            if cleaned[i] == "{":
                depth = 0
                in_str = False
                j = i
                while j < n:
                    c = cleaned[j]
                    if c == "\\" and in_str:
                        j += 2
                        continue
                    if c == '"':
                        in_str = not in_str
                    elif not in_str:
                        if c == "{":
                            depth += 1
                        elif c == "}":
                            depth -= 1
                            if depth == 0:
                                j += 1
                                break
                    j += 1
                frag = cleaned[i:j]
                # Models often emit real (unescaped) newlines/tabs inside JSON
                # string values, which is invalid JSON. Repair by escaping bare
                # control chars within this single-object fragment. (Each frag
                # here is exactly one object, so any bare newline is inside a
                # string and safe to escape.)
                frag = re.sub(r'[\r\t]', lambda m: {"\r": "\\r", "\t": "\\t"}[m.group(0)], frag)
                # Note: use a lambda so re.sub does NOT reinterpret the
                # replacement as an escape sequence (it would turn '\\n' back
                # into a real newline). The lambda returns a literal backslash+n.
                frag = re.sub(r'(?<!\\)\n', lambda m: "\\n", frag)
                try:
                    objs.append(json.loads(frag))
                except Exception:
                    pass
                i = j
            else:
                i += 1

    calls = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        name = o.get("name") or (o.get("function") or {}).get("name")
        if name not in KNOWN_TOOLS:
            continue
        args = o.get("arguments") or o.get("parameters") or {}
        if not isinstance(args, dict):
            args = {}
        calls.append({"function": {"name": name, "arguments": args}})
    return calls


# === Per-agent driver (runs inside its own tmux pane) ===
def _agent_loop(role, model):
    """Long-lived agent process. Reads a task file, calls the model, writes a
    response file. Loops until told to stop. This is what tmux spawns/manages."""
    import requests  # local import so the driver is self-contained

    os.makedirs(MAILBOX, exist_ok=True)
    task_path = os.path.join(MAILBOX, f"{role}_task.json")
    resp_path = os.path.join(MAILBOX, f"{role}_resp.json")
    stop_path = os.path.join(MAILBOX, f"{role}_stop")

    print(f"[{role}] agent up (model={model}) pid={os.getpid()}", flush=True)
    last_seq = -1
    while True:
        if os.path.exists(stop_path):
            print(f"[{role}] stop signal received", flush=True)
            break
        if not os.path.exists(task_path):
            time.sleep(TURN_POLL)
            continue
        try:
            with open(task_path) as f:
                task = json.load(f)
        except Exception:
            time.sleep(TURN_POLL)
            continue
        seq = task.get("seq", -1)
        if seq == last_seq:
            time.sleep(TURN_POLL)
            continue
        last_seq = seq

        messages = task.get("messages", [])
        # The orchestrator decides which tool schema each role gets.
        schema = task.get("schema", GO_TOOL_SCHEMA)
        print(f"[{role}] task seq={seq} turns={len(messages)} -> calling model", flush=True)
        try:
            out = tb.call_ollama(model, messages, timeout=AGENT_TIMEOUT)
            resp = {
                "seq": seq,
                "content": out.get("message", {}).get("content", ""),
                "tool_calls": out.get("tool_calls", []),
                "thinking": out.get("thinking", ""),
                "ok": True,
            }
        except Exception as e:
            resp = {"seq": seq, "content": "", "tool_calls": [],
                    "thinking": "", "ok": False, "error": str(e)}
            print(f"[{role}] model call failed: {e}", flush=True)
        with open(resp_path, "w") as f:
            json.dump(resp, f)
        print(f"[{role}] responded seq={seq}", flush=True)


def _write_task(role, messages, schema=GO_TOOL_SCHEMA):
    os.makedirs(MAILBOX, exist_ok=True)
    seq = int(time.time() * 1000)
    with open(os.path.join(MAILBOX, f"{role}_task.json"), "w") as f:
        json.dump({"seq": seq, "messages": messages, "schema": schema}, f)
    return seq


def _read_resp(role, seq, timeout=AGENT_TIMEOUT):
    resp_path = os.path.join(MAILBOX, f"{role}_resp.json")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(resp_path):
            try:
                with open(resp_path) as f:
                    r = json.load(f)
                if r.get("seq") == seq:
                    return r
            except Exception:
                pass
        time.sleep(TURN_POLL)
    return {"seq": seq, "content": "", "tool_calls": [],
            "thinking": "", "ok": False, "error": "agent timeout"}


# === Tool dispatch for the Writer (operates on the shared project dir) ===
def _dispatch_writer_tool(name, args, project_dir):
    if name == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        # Prevent escaping the project dir.
        full = os.path.normpath(os.path.join(project_dir, path))
        if not (full == project_dir or full.startswith(project_dir + os.sep)):
            return {"ok": False, "error": f"path escapes project dir: {path}"}
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
        # Invalidate the reviewer's cached view by bumping a "changed" marker.
        with open(os.path.join(MAILBOX, "writer_changed"), "w") as f:
            f.write(str(time.time()))
        return {"ok": True, "path": path, "size": len(content)}
    if name == "run_shell":
        cmd = args.get("command", "")
        # Sandbox a little: only allow go / make / the built binary / common utils.
        if any(token in cmd for token in (";", "&&", "||", "|", "rm ", "sudo", "curl", "wget", ">")):
            return {"ok": False, "error": "command rejected by sandbox (no chains/redirects)"}
        try:
            r = subprocess.run(cmd, shell=True, cwd=project_dir,
                               capture_output=True, text=True, timeout=120)
            return {"ok": r.returncode == 0, "returncode": r.returncode,
                    "stdout": r.stdout[-4000:], "stderr": r.stderr[-4000:]}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "command timed out"}
    if name == "submit":
        return {"ok": True, "submitted": True}
    return {"ok": False, "error": f"unknown tool {name}"}


# === Reviewer agent content ===
REVIEWER_SYSTEM = (
    "You are a senior Go code reviewer. You are paired with a Writer agent that is "
    "improving a prime-sieve program (simplesieve). Your job is to READ the current "
    "source and the latest build/test output, then give concise, actionable review "
    "notes: correctness risks, Go idioms, clarity/streamlining opportunities, and "
    "performance. Do NOT edit files. Be specific (cite function names / line ranges). "
    "If the code is correct and clean, say so briefly. Keep it under 250 words."
)


def _trigger_review(reviewer_model, project_dir):
    """Build a reviewer prompt from the current project files + last build output,
    send it to the reviewer agent, return the critique text."""
    files_of_interest = ["main.go", "internal/sieve/sieve.go"]
    snippets = []
    for rel in files_of_interest:
        p = os.path.join(project_dir, rel)
        if os.path.exists(p):
            try:
                txt = open(p, encoding="utf-8").read()
            except Exception:
                txt = ""
            snippets.append(f"--- {rel} ---\n{txt[:6000]}\n")
    build_log = ""
    blp = os.path.join(MAILBOX, "last_build.txt")
    if os.path.exists(blp):
        build_log = open(blp).read()[-3000:]

    messages = [
        {"role": "system", "content": REVIEWER_SYSTEM},
        {"role": "user", "content": (
            "Review the current simplesieve code.\n\n"
            + "\n".join(snippets)
            + f"\n--- last build/test output ---\n{build_log}\n"
        )},
    ]
    seq = _write_task("reviewer", messages, schema=[])  # reviewer has no tools
    try:
        if USE_TMUX:
            resp = _read_resp("reviewer", seq)
        else:
            out = tb.call_ollama(reviewer_model, messages, timeout=AGENT_TIMEOUT)
            resp = {"seq": seq, "content": out.get("message", {}).get("content", ""),
                    "tool_calls": [], "thinking": out.get("thinking", ""), "ok": True}
        review = resp.get("content", "").strip()
    except Exception as e:
        review = f"[reviewer call failed: {e}]"
        print(f"  ⚠ reviewer error: {e}", flush=True)
    with open(os.path.join(MAILBOX, "review.json"), "w") as f:
        json.dump({"review": review, "seq": seq}, f)
    return review


# === Orchestrator ===
WRITER_SYSTEM = (
    "You are a Go engineer paired with a Reviewer agent. Your task: improve and "
    "streamline the 'simplesieve' prime-sieve program (a bit-packed, wheel-based, "
    "segmented, parallel Eratosthenes sieve in Go) WITHOUT changing its observable "
    "behavior. Correctness anchor: `go build` must succeed and "
    "`./simplesieve --limit 1e6 -c` must print exactly 78498, and the SHA-256 hash "
    "of the prime list must stay stable. Streamline/clarify the code, remove "
    "redundancy, improve naming and comments, and keep it idiomatic Go. Use the "
    "tools: write_file to EDIT THE EXISTING FILES (main.go and internal/sieve/sieve.go) "
    "in place — do NOT create a new top-level file; preserve the package layout and "
    "the `internal/sieve` API (NewBitPackedEratosthenes, ForEachPrime, StreamHasher). "
    "run_shell to build/vet/run, and submit when done. Fix issues the Reviewer raises. "
    "Be careful with the parallel sieve — preserve correctness."
)


def _verify(project_dir):
    """Build and run the canonical correctness check. Returns (passed, detail)."""
    try:
        b = subprocess.run(["go", "build", "-o", "simplesieve", "."],
                           cwd=project_dir, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return False, "build timed out"
    if b.returncode != 0:
        return False, "BUILD FAILED:\n" + (b.stderr or b.stdout)[-3000:]
    try:
        r = subprocess.run(["./simplesieve", "--limit", "1e6", "-c"],
                           cwd=project_dir, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "run timed out"
    out = (r.stderr or "").strip().splitlines()
    count = out[-1] if out else ""
    passed = (count == CORRECT_COUNT_1E6)
    return passed, f"count={count} (expected {CORRECT_COUNT_1E6}) -> {'PASS' if passed else 'FAIL'}"


def run_orchestrator(writer_model, reviewer_model, project_dir, use_tmux):
    messages = [{"role": "system", "content": WRITER_SYSTEM}]
    conversation = []
    review = ""
    os.makedirs(MAILBOX, exist_ok=True)

    print("=" * 64)
    print(f"Orchestrating: writer={writer_model}  reviewer={reviewer_model}")
    print(f"Project: {project_dir}")
    print("=" * 64)

    submitted = False
    for turn in range(1, MAX_TURNS + 1):
        try:
            # Inject the latest review as context for the writer.
            writer_msgs = list(messages)
            if review:
                writer_msgs.append({
                    "role": "user",
                    "content": (
                        "REVIEWER NOTES (address these in your next edit):\n" + review
                    ),
                })
            writer_msgs.append({
                "role": "user",
                "content": "Continue improving simplesieve. Use tools, then submit when correct & clean.",
            })

            if use_tmux:
                seq = _write_task("writer", writer_msgs, schema=GO_TOOL_SCHEMA)
                resp = _read_resp("writer", seq)
            else:
                out = tb.call_ollama(writer_model, writer_msgs,
                                     timeout=AGENT_TIMEOUT)
                resp = {
                    "seq": 0,
                    "content": out.get("message", {}).get("content", ""),
                    "tool_calls": out.get("tool_calls", []),
                    "thinking": out.get("thinking", ""),
                    "ok": True,
                }
        except Exception as e:
            print(f"  ⚠ writer call failed (turn {turn}): {e}", flush=True)
            conversation.append({"turn": turn, "role": "error", "content": str(e)})
            break
        content = resp.get("content", "")
        tool_calls = resp.get("tool_calls", []) or []
        # Some models emit tool calls as fenced/multi-object JSON text instead
        # of native tool_calls — recover them so dispatch still works.
        if not tool_calls and content:
            tool_calls = extract_text_tool_calls(content)
        conversation.append({"turn": turn, "role": "writer",
                             "content": content, "tool_calls": tool_calls,
                             "thinking": resp.get("thinking", "")})
        print(f"\n--- Writer turn {turn} ---")
        if content:
            print(content[:280])

        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                result = _dispatch_writer_tool(name, args, project_dir)
                print(f"  tool {name}: {json.dumps(result)[:200]}")
                # Mirror the tool result back to the writer like a normal tool turn.
                messages.append({"role": "assistant", "content": content,
                                 "tool_calls": tool_calls})
                messages.append({
                    "role": "user",
                    "content": json.dumps({"name": name, "result": result}),
                })
                if name == "run_shell":
                    with open(os.path.join(MAILBOX, "last_build.txt"), "w") as f:
                        f.write(json.dumps(result, indent=2))
                if name == "submit" and result.get("ok"):
                    submitted = True
        else:
            messages.append({"role": "assistant", "content": content})

        # After the writer acts, get a fresh review.
        print("  -> requesting reviewer critique...")
        review = _trigger_review(reviewer_model, project_dir)
        print(f"  reviewer: {review[:200]}")

        if submitted:
            print("\nWriter submitted. Final verification...")
            break

    passed, detail = _verify(project_dir)
    print("\n=== VERIFICATION ===")
    print(detail)
    # Persist the run report.
    report = {
        "writer": writer_model,
        "reviewer": reviewer_model,
        "submitted": submitted,
        "verification_passed": passed,
        "verification_detail": detail,
        "conversation": conversation,
        "final_review": review,
    }
    out_path = os.path.join(WORKSPACE, "agent_pair_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {out_path}")
    return report


# === tmux management ===
def _tmux(*args):
    return subprocess.run(["tmux"] + list(args), capture_output=True, text=True)


def launch_tmux_agents(writer_model, reviewer_model):
    """Create a tmux session with writer + reviewer + monitor panes, each running
    the agent driver loop. Returns nothing; panes keep running."""
    _tmux("kill-session", "-t", TMUX_SESSION)  # idempotent-ish; ignore errors
    r = _tmux("new-session", "-d", "-s", TMUX_SESSION, "-n", "writer",
              "bash -c 'cd %s && exec python3 agent_pair.py --_agent writer --model %s; exec bash'"
              % (BASE_DIR, writer_model))
    if r.returncode != 0:
        raise RuntimeError("tmux new-session failed: " + r.stderr)
    _tmux("split-window", "-t", f"{TMUX_SESSION}:writer", "-h",
          "bash -c 'cd %s && exec python3 agent_pair.py --_agent reviewer --model %s; exec bash'"
          % (BASE_DIR, reviewer_model))
    _tmux("split-window", "-t", f"{TMUX_SESSION}:writer", "-v",
          "bash -c 'cd %s && echo MONITOR: writer=writer pane, reviewer=reviewer pane; "
          "tail -f mailbox/review.json 2>/dev/null; exec bash'" % BASE_DIR)
    _tmux("select-layout", "-t", TMUX_SESSION, "tiled")
    print(f"tmux session '{TMUX_SESSION}' up. Panes: writer | reviewer | monitor.")
    print(f"  Attach with:  tmux attach -t {TMUX_SESSION}")


def stop_tmux_agents():
    for role in ("writer", "reviewer"):
        sp = os.path.join(MAILBOX, f"{role}_stop")
        try:
            open(sp, "w").close()
        except Exception:
            pass
    time.sleep(1)
    _tmux("kill-session", "-t", TMUX_SESSION)


# === project setup ===
def setup_project(project_url, project_name):
    proj_dir = os.path.join(WORKSPACE, project_name)
    if not os.path.isdir(proj_dir):
        print(f"Cloning {project_url} -> {proj_dir}")
        subprocess.run(["git", "clone", "--depth", "1", project_url, proj_dir],
                       check=True)
    else:
        print(f"Project already at {proj_dir} (skip clone).")
    return proj_dir


def main():
    ap = argparse.ArgumentParser(description="Two-LLM Writer+Reviewer tmux harness")
    ap.add_argument("--writer", default="qwen2.5-coder:7b")
    ap.add_argument("--reviewer", default="qwen3:8b")
    ap.add_argument("--project", default=DEFAULT_PROJECT_URL)
    ap.add_argument("--project-name", default=DEFAULT_PROJECT_NAME)
    ap.add_argument("--no-tmux", action="store_true",
                    help="Run agents in-process (no tmux panes)")
    ap.add_argument("--attach", action="store_true",
                    help="Print tmux attach command and exit after launch")
    ap.add_argument("--_agent", help="INTERNAL: run an agent driver loop (writer|reviewer)")
    ap.add_argument("--model", help="INTERNAL: model for --_agent")
    args = ap.parse_args()

    if args._agent:
        # Agent driver mode (invoked inside a tmux pane).
        _agent_loop(args._agent, args.model or "qwen2.5-coder:7b")
        return

    writer_model = args.writer
    reviewer_model = args.reviewer
    project_dir = setup_project(args.project, args.project_name)

    # Warm up both models (not timed critically; keeps first call fast).
    for m in (writer_model, reviewer_model):
        try:
            tb.warmup_model(m)
        except Exception as e:
            print(f"warmup warning for {m}: {e}")

    use_tmux = not args.no_tmux
    global USE_TMUX
    USE_TMUX = use_tmux
    if use_tmux:
        launch_tmux_agents(writer_model, reviewer_model)
        if args.attach:
            print(f"\nRun:  tmux attach -t {TMUX_SESSION}")
            return
        # Give panes a moment to import and print "agent up".
        time.sleep(HANDSHAKE_TIMEOUT)
        # Confirm both agents are alive.
        try:
            wp = os.path.join(MAILBOX, "writer_task.json")
        except Exception:
            pass

    try:
        run_orchestrator(writer_model, reviewer_model, project_dir, use_tmux)
    finally:
        if use_tmux:
            stop_tmux_agents()


if __name__ == "__main__":
    main()
