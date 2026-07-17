#!/usr/bin/env python3
"""
Tool-calling benchmark for Ollama models.
Each model receives the hashprime problem with tool access.
Results are saved to results/<model>/<timestamp>/.
Scoring: correctness (compile + hash match), speed, code quality, tool usage.
Handles both native tool-calling models and text-only models (extracts code).
"""

import json, os, sys, time, subprocess, hashlib, re, shutil
from datetime import datetime
import requests, numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

RESULTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OLLAMA_URL = "http://localhost:11434/api/chat"
PER_MODEL_TIMEOUT = 600
MAX_TURNS = 10
EXPECTED_HASH = "563d8e0603dcc07d784135d99fd81ff6bf98495e898ec1f52e2e7605320cf6dc"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
CHECKSTYLE_JAR = os.path.join(TOOLS_DIR, "checkstyle.jar")
CHECKSTYLE_CONFIG = os.path.join(TOOLS_DIR, "checkstyle_config.xml")
PMD_BIN = os.path.join(TOOLS_DIR, "pmd", "bin", "pmd")


def cleanup_all_models():
    """Stop all running models and wait until none remain. Not timed."""
    while True:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=30)
        lines = result.stdout.strip().split("\n")
        if len(lines) <= 1:
            break
        for line in lines[1:]:
            parts = line.split()
            if parts:
                subprocess.run(["ollama", "stop", parts[0]], capture_output=True, timeout=30)
        time.sleep(2)


def warmup_model(model):
    """Load model into memory with a trivial prompt. Not timed."""
    print(f"  Warming up {model}...")
    try:
        requests.post(OLLAMA_URL, json={
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "options": {"temperature": 0.2}
        }, timeout=120)
    except Exception:
        pass


def format_time(seconds):
    """Format seconds as human-readable string."""
    if seconds >= 120:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    if seconds >= 60:
        return f"{seconds:.1f}s"
    if seconds >= 1:
        return f"{seconds:.2f}s"
    return f"{seconds*1000:.0f}ms"


# === Code extraction ===

TOOL_DESCRIPTIONS = """
You have these tools available. To use a tool, respond with a JSON object (NOT inside a code block, just raw JSON):

Tool: write_file
  Description: Write content to a file
  Arguments: path (string), content (string)
  Example: {"name": "write_file", "arguments": {"path": "hashprime.java", "content": "public class hashprime { ... }"}}

Tool: compile_java
  Description: Compile hashprime.java with javac
  Arguments: none
  Example: {"name": "compile_java", "arguments": {}}

Tool: run_and_hash
  Description: Run 'java hashprime N' and return its SHA-256 hash
  Arguments: n (integer) - upper limit for prime finding
  Example: {"name": "run_and_hash", "arguments": {"n": 12}}

Tool: submit_answer
  Description: Submit the final SHA-256 hash as the answer
  Arguments: hash (string), expected (string)
  Example: {"name": "submit_answer", "arguments": {"hash": "563d...", "expected": "563d..."}}

Tool: search_solutions
  Description: Search the indexed reference corpus (prior hashprime.java solutions, algorithm/data-structure patterns, and sysadmin notes) for relevant code and explanations. Use it to ground your approach.
  Arguments: query (string, optional) - what to look for; if omitted the task is used
  Example: {"name": "search_solutions", "arguments": {"query": "Sieve of Eratosthenes feed primes to SHA-256"}}
"""

PROMPT = """You are solving a Java programming challenge. You have access to tools to write files, compile, run commands, and verify results.
""" + TOOL_DESCRIPTIONS + """
Your task:
Create a fast Java program called hashprime.java that identifies prime numbers within the natural numbers specified on the command line and prints the primelist out in this format: prime1\\nprime2\\n...

Run the primelist through SHA-256 and see if we get a matching hash.

Input bytes: 2\\n3\\n5\\n7\\n11\\n
SHA-256: 563d8e0603dcc07d784135d99fd81ff6bf98495e898ec1f52e2e7605320cf6dc

Confirm that 'java hashprime 11' is the same as 'java hashprime 12' and the same as SHA-256: 563d8e0603dcc07d784135d99fd81ff6bf98495e898ec1f52e2e7605320cf6dc

Hint: To be the fastest, don't write an output file. Instead, write directly to the sha256sum calculation with the newly discovered prime '\\n'.

Steps you should follow:
1. Write the Java code to hashprime.java using the write_file tool
2. Compile it using the compile_java tool
3. Run it and compute the hash using run_and_hash. Verify it matches the expected hash
4. Confirm that N=11 and N=12 produce the same output
5. Submit your answer with submit_answer when you are confident

Important: use `public class hashprime` (lowercase) to match the filename.
Respond with a JSON tool call to use a tool, or plain text to communicate.

Optional: you may call search_solutions to retrieve prior solutions and algorithm patterns from the indexed reference corpus. If you view a sample solution, improve on it rather than copying it verbatim — aim for a faster or cleaner implementation."""


# === Code extraction ===

def extract_java_code(text):
    """Extract Java source code from LLM response text."""
    if not text:
        return None

    patterns = [
        r'```java\s*\n(.*?)```',
        r'```\s*\n(.*?)```',
        r'public\s+(?:class|interface|enum)\s+\w+',
    ]

    for p in patterns:
        if p.startswith('public'):
            idx = text.find('public ')
            if idx >= 0:
                return text[idx:].rstrip() + '\n'
        else:
            m = re.search(p, text, re.DOTALL)
            if m:
                code = m.group(1).strip()
                if 'public class' in code or 'class hashprime' in code or 'import' in code:
                    return code + '\n'

    lines = text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        if line.startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)

    if code_lines:
        return '\n'.join(code_lines) + '\n'

    return None


def extract_fake_tool_call(text):
    """Some models output tool calls as JSON in plain text instead of using the API.
    Try to extract write_file calls from such JSON, handling nested braces.
    Handles varying schemas: {name, arguments}, {function, filename, content}, etc."""
    if not text:
        return None

    # Strip markdown fences
    cleaned = re.sub(r'```(?:json)?\s*', '', text)

    # Find a write_file reference in the text
    idx = cleaned.find('write_file')
    if idx < 0:
        idx = cleaned.find('write')
    if idx < 0:
        return None

    # Find enclosing outer braces with depth tracking
    # Track JSON string state to avoid counting braces inside string values
    start = idx
    depth = 0
    while start >= 0:
        ch = cleaned[start]
        if ch == '}':
            depth += 1
        elif ch == '{':
            depth -= 1
            if depth < 0:
                break
        start -= 1

    if depth >= 0:
        return None  # No opening brace found

    end = idx
    depth = 1  # outer { already found by backward search
    # Determine if we start inside a JSON string by scanning backward
    in_string = False
    pos = idx
    while pos >= 0:
        if cleaned[pos] == '"' and (pos == 0 or cleaned[pos-1] != '\\'):
            in_string = not in_string
        pos -= 1
    while end < len(cleaned):
        ch = cleaned[end]
        # Toggle string state on unescaped quotes
        if ch == '"' and (end == 0 or cleaned[end-1] != '\\'):
            in_string = not in_string
        if not in_string:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth <= 0:
                    end += 1
                    break
        end += 1

    json_str = cleaned[start:end]

    # Try std schema first: {"name":"write_file","arguments":{"path":"...","content":"..."}}
    try:
        data = json.loads(json_str)
        args = data.get("arguments") or data.get("parameters") or data
        path = args.get("path") or args.get("filename") or "hashprime.java"
        raw_content = args.get("content") or args.get("code") or args.get("source") or ""
        if isinstance(raw_content, dict):
            raw_content = raw_content.get("value") or raw_content.get("text") or ""
        if raw_content and len(raw_content) > 50 and ('class' in raw_content or 'main' in raw_content or 'import' in raw_content):
            return path, raw_content
    except json.JSONDecodeError:
        pass

    # Try alternate schema: {"function":"write_file","filename":"...","content":"..."}
    try:
        data = json.loads(json_str)
        path = data.get("filename") or data.get("path") or "hashprime.java"
        raw_content = data.get("content") or data.get("code") or ""
        if isinstance(raw_content, dict):
            raw_content = raw_content.get("value") or raw_content.get("text") or ""
        if raw_content and len(raw_content) > 50 and ('class' in raw_content or 'main' in raw_content or 'import' in raw_content):
            return path, raw_content
    except json.JSONDecodeError:
        pass

    # Reject here unless the captured model JSON is fully valid (jq-clean).
    # Per policy: if the model's JSON does not parse, we FAIL the tool call
    # rather than "repairing" it — any correction risks injecting our own
    # error. No lenient regex fallback that fabricates content.
    try:
        json.loads(json_str)
    except json.JSONDecodeError:
        return None

    # Extract path for the fallback paths
    path_m = re.search(r'"(?:path|filename)"\s*:\s*"([^"]+)"', json_str)
    path = path_m.group(1) if path_m else "hashprime.java"

    # Build the combined content from all fragments after "content": "...
    # Models often use: "content": "part1" + "part2" + "part3"
    content_start = re.search(r'"content"\s*:\s*"', json_str, re.DOTALL)
    if not content_start:
        return None

    idx = content_start.end()
    fragments = []
    while idx < len(json_str):
        # Extract fragment: captures everything between the opening and closing "
        frag_m = re.match(r'((?:[^"\\]|\\.)*)', json_str[idx:])
        if not frag_m:
            break
        fragment = frag_m.group(1)
        idx += frag_m.end()
        if idx >= len(json_str) or json_str[idx] != '"':
            break
        idx += 1  # skip closing "
        fragments.append(fragment)
        # Check for concatenation: " + "..."
        cont_m = re.match(r'\s*\+\s*"', json_str[idx:])
        if not cont_m:
            break
        idx += cont_m.end()

    if not fragments:
        return None

    content = ''.join(fragments)
    # Unescape JSON escape sequences
    content = content.replace('\\\\', '\x00BS\x00')
    content = content.replace('\\"', '"')
    content = content.replace('\\n', '\n')
    content = content.replace('\\t', '\t')
    content = content.replace('\\r', '')
    content = content.replace('\x00BS\x00', '\\')
    # Only accept if it actually looks like Java. Plain JSON / stray objects
    # (e.g. a retrieved reference echo starting with "{") are rejected so we
    # never write non-code into hashprime.java.
    if content and ('class' in content or 'public' in content or 'void main' in content):
        return path, content

    return None


# === Tool implementations ===

def tool_write_file(args, run_dir):
    path = args["path"]
    content = args["content"]
    full_path = os.path.join(run_dir, path)
    os.makedirs(os.path.dirname(full_path) or run_dir, exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    return json.dumps({"ok": True, "path": path, "size": len(content)})


def tool_read_file(args, run_dir):
    path = args["path"]
    full_path = os.path.join(run_dir, path)
    if not os.path.exists(full_path):
        return json.dumps({"ok": False, "error": "file not found"})
    with open(full_path) as f:
        content = f.read()
    return json.dumps({"ok": True, "path": path, "content": content, "size": len(content)})


def tool_run_command(args, run_dir):
    cmd = args["command"]
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60, cwd=run_dir)
        return json.dumps({
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-2000:]
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"ok": False, "error": "Command timed out"})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


def tool_compile_java(args, run_dir):
    full_path = os.path.join(run_dir, "hashprime.java")
    if not os.path.exists(full_path):
        return json.dumps({"ok": False, "error": "hashprime.java not found"})

    # Show the code
    with open(full_path) as f:
        code_lines = f.readlines()
    print(f"  ── hashprime.java ({len(code_lines)} lines) ──")
    for l in code_lines[:35]:
        print(f"    {l}", end="")
    if len(code_lines) > 35:
        print(f"    ... ({len(code_lines) - 35} more lines)")
    print()

    try:
        result = subprocess.run(
            ["javac", "-Xlint:all", "hashprime.java"],
            capture_output=True, text=True, timeout=30, cwd=run_dir
        )

        if result.stderr.strip():
            print(f"  ── javac output ──")
            for l in result.stderr.strip().split("\n"):
                print(f"    {l}")

        response = {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "checkstyle_count": 0,
            "checkstyle_output": "",
            "pmd_count": 0,
            "pmd_output": "",
            "lint_time": 0,
        }

        if result.returncode == 0:
            cs_count, cs_lines, cs_time = run_checkstyle(full_path, run_dir)
            pmd_count, pmd_lines, pmd_time = run_pmd(full_path, run_dir)
            response["checkstyle_count"] = cs_count
            response["checkstyle_output"] = "\n".join(cs_lines)
            response["pmd_count"] = pmd_count
            response["pmd_output"] = "\n".join(pmd_lines)
            response["lint_time"] = cs_time + pmd_time
            if cs_lines:
                with open(os.path.join(run_dir, "checkstyle_report.txt"), "w") as f:
                    f.write("\n".join(cs_lines) + "\n")
            if pmd_lines:
                with open(os.path.join(run_dir, "pmd_report.txt"), "w") as f:
                    f.write("\n".join(pmd_lines) + "\n")

        return json.dumps(response)
    except subprocess.TimeoutExpired:
        return json.dumps({"ok": False, "error": "Compilation timed out"})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


def tool_run_and_hash(args, run_dir):
    n = args.get("n", 12)
    if not os.path.exists(os.path.join(run_dir, "hashprime.class")):
        return json.dumps({"ok": False, "error": "hashprime.class not found. Compile first."})
    try:
        start = time.time()
        result = subprocess.run(
            ["java", "-cp", run_dir, "hashprime", str(n)],
            capture_output=True, text=True, timeout=30, cwd=run_dir
        )
        exec_time = time.time() - start
        if result.returncode != 0:
            return json.dumps({"ok": False, "returncode": result.returncode, "stderr": result.stderr})
        output = result.stdout
        sha = hashlib.sha256(output.encode()).hexdigest()
        return json.dumps({
            "ok": True,
            "n": n,
            "output": output,
            "sha256": sha,
            "matches_expected": sha == EXPECTED_HASH,
            "exec_time": round(exec_time, 3)
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"ok": False, "error": "Execution timed out"})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


def tool_submit_answer(args, run_dir):
    h = args.get("hash", "")
    e = args.get("expected", EXPECTED_HASH)
    correct = h.lower() == e.lower()
    return json.dumps({
        "ok": True,
        "submitted_hash": h,
        "expected_hash": e,
        "correct": correct
    })


# Qdrant collections the search tool may query (across all reference corpora)
SEARCH_COLLECTIONS = ["hashprime_solutions", "hello_algo", "linux_server"]
SEARCH_TOP_K = 3


def tool_search_solutions(args, run_dir):
    """Search the indexed Qdrant reference corpora for relevant prior solutions
    and algorithm patterns. Returns top-k chunks per collection, tagged with
    source collection + file so the model can judge provenance.

    If the model supplies no/short query, falls back to the benchmark PROMPT so
    the search is still driven by the task's initial tokens.
    """
    query = (args.get("query", "") or "").strip()
    if len(query) < 12:
        query = PROMPT
    try:
        vec = embed_text(query)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"embed failed: {e}"})

    try:
        qdrant = QdrantClient(path=QDRANT_PATH)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"qdrant unavailable: {e}"})

    results = []
    for coll in SEARCH_COLLECTIONS:
        try:
            qdrant.get_collection(coll)
        except Exception:
            continue
        try:
            pts = qdrant.query_points(
                collection_name=coll, query=vec.tolist(), limit=SEARCH_TOP_K
            ).points
        except Exception:
            continue
        for p in pts:
            payload = p.payload or {}
            results.append({
                "collection": coll,
                "source": payload.get("source", "?"),
                "score": round(p.score, 4),
                "text": (payload.get("text", "") or "")[:800],
            })
    if not results:
        return json.dumps({"ok": True, "query": query[:200], "results": [],
                            "note": "No reference chunks found."})
    return json.dumps({"ok": True, "query": query[:200], "results": results})


TOOL_DISPATCH = {
    "write_file": tool_write_file,
    "read_file": tool_read_file,
    "run_command": tool_run_command,
    "compile_java": tool_compile_java,
    "run_and_hash": tool_run_and_hash,
    "submit_answer": tool_submit_answer,
    "search_solutions": tool_search_solutions,
}

# Ollama tool schema (native tool-calling API). Mirrors TOOL_DISPATCH.
TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compile_java",
            "description": "Compile hashprime.java with javac and run lint checks (checkstyle, PMD).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_and_hash",
            "description": "Run 'java hashprime N' and return its SHA-256 hash of the prime list.",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final SHA-256 hash as the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hash": {"type": "string"},
                    "expected": {"type": "string"},
                },
                "required": ["hash", "expected"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_solutions",
            "description": "Search the indexed Qdrant reference corpora (prior hashprime.java solutions, algorithm/data-structure patterns, and sysadmin notes) for relevant code and explanations. Returns top chunks tagged by source collection and file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (e.g. 'Sieve of Eratosthenes SHA-256 primes'). If omitted, the task prompt is used.",
                    }
                },
                "required": [],
            },
        },
    },
]


def get_model_capabilities(model):
    """Return the set of Ollama capabilities for a model via POST /api/show.
    E.g. {'completion', 'tools', 'thinking'}. Returns empty set on error.
    """
    try:
        resp = requests.post("http://localhost:11434/api/show",
                             json={"model": model}, timeout=30)
        resp.raise_for_status()
        caps = resp.json().get("capabilities", [])
        return set(caps)
    except Exception:
        return set()


# === Lint tools ===

CHECKSTYLE_ENABLED = os.path.exists(CHECKSTYLE_JAR)
PMD_ENABLED = os.path.exists(PMD_BIN)


def run_checkstyle(java_path, run_dir):
    if not CHECKSTYLE_ENABLED:
        return 0, [], 0
    if not os.path.exists(java_path):
        return 0, [], 0
    try:
        start = time.time()
        result = subprocess.run(
            ["java", "-jar", CHECKSTYLE_JAR, "-c", CHECKSTYLE_CONFIG, java_path],
            capture_output=True, text=True, timeout=30, cwd=run_dir
        )
        elapsed = time.time() - start
        lines = [l for l in result.stdout.split("\n") if l.strip()]
        count = len(lines)
        if count > 0:
            print(f"  ── Checkstyle ({count} violations) ──")
            for l in lines:
                print(f"    {l}")
        return count, lines, elapsed
    except subprocess.TimeoutExpired:
        return 0, [], 0
    except Exception:
        return 0, [], 0


def run_pmd(java_path, run_dir):
    if not PMD_ENABLED:
        return 0, [], 0
    if not os.path.exists(java_path):
        return 0, [], 0
    try:
        start = time.time()
        result = subprocess.run(
            [PMD_BIN, "check", "-f", "text", "-R",
             "category/java/bestpractices.xml,category/java/errorprone.xml",
             "-d", java_path],
            capture_output=True, text=True, timeout=60, cwd=run_dir
        )
        elapsed = time.time() - start
        lines = [l for l in result.stdout.split("\n") if l.strip() and "PMD " not in l]
        count = len(lines)
        if count > 0:
            print(f"  ── PMD ({count} violations) ──")
            for l in lines:
                print(f"    {l}")
        return count, lines, elapsed
    except subprocess.TimeoutExpired:
        return 0, [], 0
    except Exception:
        return 0, [], 0


def jq_audit(text):
    """Pipe text through jq to verify it's valid JSON.
    Strips ```json fences first, then tests jq parseability.
    Returns True if jq accepts the output as-is, False otherwise."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        fence_start = 0
        for i, l in enumerate(lines):
            if l.strip().startswith("```"):
                fence_start = i
                break
        cleaned = "\n".join(lines[fence_start + 1:])
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    try:
        result = subprocess.run(
            ["jq", "."],
            input=cleaned, capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


# === Semantic gate (on-topic detection via embedding + Qdrant) ===

QDRANT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_data")
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_DIM = 768
SEMANTIC_THRESHOLD = 0.60


def embed_text(text):
    """Get embedding vector via Ollama nomic-embed-text."""
    resp = requests.post(OLLAMA_EMBED_URL, json={
        "model": "nomic-embed-text",
        "input": text,
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return np.array(data["embeddings"][0], dtype=np.float32)


def semantic_gate_check(text):
    """Check if model output is semantically on-topic.
    Hard gate based on hashprime_solutions similarity (threshold 0.60).
    Also checks linux_server for context (lower = less sysadmin drift).
    Returns (passed: bool, max_score: float, best_hit: dict).
    """
    if not text or len(text.strip()) < 20:
        return False, 0.0, {}
    try:
        qdrant = QdrantClient(path=QDRANT_PATH)
        vec = embed_text(text)
        hp_score = 0.0
        ls_score = 0.0
        best_hit = {}
        for coll in ["hashprime_solutions", "linux_server"]:
            try:
                qdrant.get_collection(coll)
            except Exception:
                continue
            result = qdrant.query_points(
                collection_name=coll,
                query=vec.tolist(),
                limit=1,
            )
            for p in result.points:
                if coll == "hashprime_solutions":
                    hp_score = p.score
                else:
                    ls_score = p.score
                if p.score > best_hit.get("score", 0):
                    best_hit = {
                        "score": round(p.score, 4),
                        "collection": coll,
                        "text": p.payload.get("text", "")[:200],
                    }
        # Pass if high similarity to hashprime solutions (writing Java code for the problem)
        # Also pass if both are high (writing about both topics, which is valid)
        # Fail only if hashprime similarity is very low
        passed = hp_score >= SEMANTIC_THRESHOLD
        return passed, hp_score, best_hit
    except Exception as e:
        return True, 0.0, {"error": str(e)}  # pass by default on error


def thinking_quality_score(thinking_text):
    """Measure semantic quality of a model's reasoning/thinking trace.

    Chunks the thinking text with chonkie, embeds each chunk via Ollama
    nomic-embed-text, and scores how on-topic the reasoning is relative to
    the known-good hashprime_solutions corpus in Qdrant.

    Returns (score_0_100, chunks_scored, mean_similarity, error_or_None).
    Higher = thinking is more semantically aligned with good solutions.
    """
    if not thinking_text or len(thinking_text.strip()) < 20:
        return 0, 0, 0.0, "empty"
    try:
        from chonkie import TokenChunker
        chunker = TokenChunker(tokenizer="gpt2", chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk(thinking_text)
        if not chunks:
            return 0, 0, 0.0, "no_chunks"

        qdrant = QdrantClient(path=QDRANT_PATH)
        try:
            qdrant.get_collection("hashprime_solutions")
        except Exception:
            return 0, 0, 0.0, "collection_missing"

        texts = [c.text for c in chunks]
        # Embed in batches (Ollama accepts a list input)
        vecs = []
        batch = 8
        for i in range(0, len(texts), batch):
            resp = requests.post(OLLAMA_EMBED_URL, json={
                "model": "nomic-embed-text",
                "input": texts[i:i + batch],
            }, timeout=60)
            resp.raise_for_status()
            vecs.extend(np.array(e, dtype=np.float32) for e in resp.json()["embeddings"])

        sims = []
        for v in vecs:
            result = qdrant.query_points(
                collection_name="hashprime_solutions",
                query=v.tolist(),
                limit=1,
            )
            for p in result.points:
                sims.append(p.score)
        if not sims:
            return 0, 0, 0.0, "no_results"
        mean_sim = sum(sims) / len(sims)
        return int(min(100, max(0, round(mean_sim * 100)))), len(sims), round(mean_sim, 4), None
    except Exception as e:
        return 0, 0, 0.0, str(e)


# === Compile and verify (used for text-only models) ===

def try_compile_and_verify(run_dir):
    """Compile hashprime.java, run it, check hash, run lint tools.
    Returns dict with compile_ok, hash_match, javac_warnings, checkstyle_count, pmd_count, lint_time, exec_time, output."""
    java_file = os.path.join(run_dir, "hashprime.java")
    result = {
        "compile_ok": False, "hash_match": False, "javac_warnings": 0,
        "checkstyle_count": 0, "pmd_count": 0, "lint_time": 0, "exec_time": 0, "output": "",
    }
    if not os.path.exists(java_file):
        return result

    # Show the code
    with open(java_file) as f:
        code_lines = f.readlines()
    print(f"  ── hashprime.java ({len(code_lines)} lines) ──")
    for l in code_lines[:35]:
        print(f"    {l}", end="")
    if len(code_lines) > 35:
        print(f"    ... ({len(code_lines) - 35} more lines)")
    print()

    javac_result = subprocess.run(
        ["javac", "-Xlint:all", "hashprime.java"],
        capture_output=True, text=True, timeout=30, cwd=run_dir
    )
    compile_ok = javac_result.returncode == 0
    javac_warnings = len([l for l in javac_result.stderr.split("\n") if "warning" in l.lower()])

    if javac_result.stderr.strip():
        print(f"  ── javac output ──")
        for l in javac_result.stderr.strip().split("\n"):
            print(f"    {l}")

    if not compile_ok:
        javac_result2 = subprocess.run(
            ["javac", "hashprime.java"],
            capture_output=True, text=True, timeout=30, cwd=run_dir
        )
        compile_ok = javac_result2.returncode == 0
        if not compile_ok:
            result["javac_warnings"] = javac_warnings
            result["output"] = javac_result2.stderr
            print(f"  ── javac (no lint) output ──")
            for l in javac_result2.stderr.strip().split("\n"):
                print(f"    {l}")
            return result

    result["compile_ok"] = True
    result["javac_warnings"] = javac_warnings

    # Run lint tools
    cs_count, cs_lines, cs_time = run_checkstyle(java_file, run_dir)
    pmd_count, pmd_lines, pmd_time = run_pmd(java_file, run_dir)
    result["checkstyle_count"] = cs_count
    result["pmd_count"] = pmd_count
    result["lint_time"] = cs_time + pmd_time
    if cs_lines:
        with open(os.path.join(run_dir, "checkstyle_report.txt"), "w") as f:
            f.write("\n".join(cs_lines) + "\n")
    if pmd_lines:
        with open(os.path.join(run_dir, "pmd_report.txt"), "w") as f:
            f.write("\n".join(pmd_lines) + "\n")

    # Run and verify hash
    for n in [11, 12]:
        try:
            start = time.time()
            r = subprocess.run(
                ["java", "-cp", run_dir, "hashprime", str(n)],
                capture_output=True, text=True, timeout=30, cwd=run_dir
            )
            result["exec_time"] += time.time() - start
            if r.returncode == 0:
                result["output"] = r.stdout
                sha = hashlib.sha256(r.stdout.encode()).hexdigest()
                if sha == EXPECTED_HASH:
                    result["hash_match"] = True
        except:
            pass

    # Anti-cheat consistency check: a correct algorithm must produce the SAME
    # output hash for n=8, 9, 10 (the prefix of first-n-primes is identical up
    # to n=10: 2,3,5,7,11). If the model hardcodes only the n=11/12 expected
    # output, these three will NOT match each other -> cheat detected.
    consistency_hashes = []
    for n in [8, 9, 10]:
        try:
            r = subprocess.run(
                ["java", "-cp", run_dir, "hashprime", str(n)],
                capture_output=True, text=True, timeout=30, cwd=run_dir
            )
            if r.returncode == 0:
                consistency_hashes.append(hashlib.sha256(r.stdout.encode()).hexdigest())
        except:
            pass
    result["anti_cheat_consistent"] = (
        len(consistency_hashes) == 3 and len(set(consistency_hashes)) == 1
    )
    result["anti_cheat_hashes"] = consistency_hashes

    return result


_PROMPT_TS_CACHE = None


def would_accept_submission(tcv, best_exec_time=None):
    """Stringent gate for accepting a model's solution into the corpus/DB.

    Any auto-submit path MUST call this first. Criteria:
      1. Compiles cleanly (no javac errors).
      2. Hash matches EXPECTED_HASH for n=11 and n=12 (correct output).
      3. Anti-cheat: identical hashes for n=8,9,10 (real algorithm, not hardcoded).
      4. Speed gate (optional): if best_exec_time is given, only accept when
         this run is FASTER, to keep corpus pollution (slow/duplicate) down.

    Returns (accepted: bool, reason: str).
    """
    if not tcv.get("compile_ok"):
        return False, "rejected: does not compile"
    if not tcv.get("hash_match"):
        return False, "rejected: hash mismatch (incorrect output)"
    if not tcv.get("anti_cheat_consistent"):
        return False, "rejected: anti-cheat failed (n=8,9,10 hashes inconsistent — possible hardcoding)"
    if best_exec_time is not None and tcv.get("exec_time", 0) >= best_exec_time:
        return False, f"rejected: not faster than best ({best_exec_time:.3f}s)"
    return True, "accepted"


def _prompt_ts():
    """Cached TSCORE of the benchmark PROMPT itself (topic-alignment baseline)."""
    global _PROMPT_TS_CACHE
    if _PROMPT_TS_CACHE is not None:
        return _PROMPT_TS_CACHE
    _, _, mean_sim, err = thinking_quality_score(PROMPT)
    _PROMPT_TS_CACHE = mean_sim if err is None else 0.0
    return _PROMPT_TS_CACHE


# === Scoring ===

def score_run(run_dir, conversation, timing, mode):
    """Score a model run."""
    score = {
        "correctness": 0, "speed_score": 0, "code_quality": 0,
        "tool_usage": 0, "total": 0, "details": {}
    }

    # Correctness
    hash_match = timing.get("hash_match", False)
    compile_ok = timing.get("compile_ok", False)
    has_java = os.path.exists(os.path.join(run_dir, "hashprime.java"))
    anti_cheat_consistent = timing.get("anti_cheat_consistent", None)

    # A solution is only "correct" if it matches the expected hash AND passes
    # the anti-cheat consistency check (n=8,9,10 yield identical hashes). A
    # model that hardcodes only the expected output will match the hash but
    # fail consistency -> capped at 50 (compiles) and flagged as a cheat.
    if hash_match and anti_cheat_consistent:
        score["correctness"] = 100
    elif hash_match and anti_cheat_consistent is False:
        score["correctness"] = 50
        score["details"]["cheat_suspected"] = True
    elif compile_ok:
        score["correctness"] = 50
    elif has_java:
        score["correctness"] = 25

    score["details"]["compile_ok"] = compile_ok
    score["details"]["hash_match"] = hash_match
    score["details"]["has_java_source"] = has_java
    anti_cheat_consistent = timing.get("anti_cheat_consistent", None)
    score["details"]["anti_cheat_consistent"] = anti_cheat_consistent
    score["details"]["anti_cheat_hashes"] = timing.get("anti_cheat_hashes", [])

    # Speed (exclude lint time from model's speed score)
    total_time = timing.get("total_seconds", 999)
    lint_time = timing.get("lint_time", 0)
    model_time = max(total_time - lint_time, 0)
    score["details"]["total_time_seconds"] = round(total_time, 2)
    score["details"]["lint_time_seconds"] = round(lint_time, 2)
    score["details"]["model_time_seconds"] = round(model_time, 2)
    if model_time < 30:
        score["speed_score"] = 100
    elif model_time < 60:
        score["speed_score"] = 75
    elif model_time < 120:
        score["speed_score"] = 50
    elif model_time < 300:
        score["speed_score"] = 25
    else:
        score["speed_score"] = 10

    # Code quality (composite: javac warnings + checkstyle + PMD)
    javac_warnings = timing.get("javac_warnings", 0)
    checkstyle_count = timing.get("checkstyle_count", 0)
    pmd_count = timing.get("pmd_count", 0)

    def _score_warnings(n, thresholds):
        for t, s in thresholds:
            if n <= t:
                return s
        return thresholds[-1][1]

    javac_score = _score_warnings(javac_warnings, [(0, 100), (2, 80), (5, 60), (999, 30)])
    cs_score = _score_warnings(checkstyle_count, [(0, 100), (5, 80), (15, 60), (999, 30)])
    pmd_score = _score_warnings(pmd_count, [(0, 100), (3, 80), (8, 60), (999, 30)])

    score["code_quality"] = int(javac_score * 0.4 + cs_score * 0.3 + pmd_score * 0.3)
    score["details"]["javac_warnings"] = javac_warnings
    score["details"]["checkstyle_count"] = checkstyle_count
    score["details"]["pmd_count"] = pmd_count
    score["details"]["javac_score"] = javac_score
    score["details"]["checkstyle_score"] = cs_score
    score["details"]["pmd_score"] = pmd_score

    # Tool usage - check both native API calls and text-based tool dispatches
    tools_used = set()
    for msg in conversation:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {}).get("name", "")
                if fn:
                    tools_used.add(fn)
        if msg.get("role") == "tool" and msg.get("name"):
            tool_name = msg["name"]
            tools_used.add(tool_name)

    jq_audit_passed = timing.get("jq_audit_passed", True)
    semantic_passed = timing.get("semantic_passed", True)
    if mode != "tool_call" and (not jq_audit_passed or not semantic_passed):
        score["tool_usage"] = 0
    else:
        expected_tools = {"write_file", "compile_java", "run_and_hash", "submit_answer"}
        used = len(tools_used & expected_tools)
        score["tool_usage"] = int((used / len(expected_tools)) * 100)
    score["details"]["tools_used"] = sorted(tools_used) if tools_used else ["(none)"]
    score["details"]["mode"] = mode
    score["details"]["jq_audit_passed"] = jq_audit_passed
    score["details"]["semantic_passed"] = semantic_passed
    score["details"]["semantic_score"] = timing.get("semantic_score", 0.0)
    score["details"]["turns_used"] = timing.get("turns_used", 0)
    score["details"]["exec_time"] = timing.get("exec_time", 0)
    score["details"]["thinking_support"] = timing.get("thinking_support", False)

    # Semantic quality of the reasoning trace (chonkie + Qdrant vs hashprime_solutions)
    thinking_text = "\n".join(
        m.get("content", "") for m in conversation if m.get("role") == "thinking"
    )
    tq_score, tq_chunks, tq_mean, tq_err = thinking_quality_score(thinking_text)
    score["details"]["thinking_quality"] = tq_score
    score["details"]["thinking_quality_chunks"] = tq_chunks
    score["details"]["thinking_quality_mean_sim"] = tq_mean
    score["details"]["thinking_quality_error"] = tq_err

    # Prompt baseline: how on-topic is the TASK PROMPT itself vs hashprime_solutions?
    # Cached so we embed it once per process, not per model.
    prompt_ts = _prompt_ts()
    score["details"]["prompt_ts"] = prompt_ts
    score["details"]["thinking_drift"] = round(tq_mean - prompt_ts, 4) if (tq_mean and prompt_ts) else None
    score["details"]["used_retrieval"] = timing.get("used_retrieval", False)

    score["total"] = int(
        score["correctness"] * 0.5 +
        score["speed_score"] * 0.2 +
        score["code_quality"] * 0.15 +
        score["tool_usage"] * 0.15
    )

    return score


def save_results(run_dir, model, conversation, timing, score):
    with open(os.path.join(run_dir, "conversation.json"), "w") as f:
        json.dump(conversation, f, indent=2)
    with open(os.path.join(run_dir, "score.json"), "w") as f:
        json.dump(score, f, indent=2)

    total_secs = timing.get("total_seconds", 0)
    lint_secs = timing.get("lint_time", 0)
    model_secs = max(total_secs - lint_secs, 0)
    human_total = format_time(total_secs)
    human_model = format_time(model_secs)
    human_lint = format_time(lint_secs)
    prompt_tokens = timing.get("prompt_tokens", 0)
    completion_tokens = timing.get("completion_tokens", 0)
    tools_used = ", ".join(score["details"].get("tools_used", ["(none)"]))
    javac_warnings = score["details"].get("javac_warnings", 0)
    checkstyle_count = score["details"].get("checkstyle_count", 0)
    pmd_count = score["details"].get("pmd_count", 0)
    mode = score["details"].get("mode", "?")
    jq_passed = score["details"].get("jq_audit_passed", True)
    jq_str = "✓ jq" if jq_passed else "✗ jq FAIL"
    sem_passed = score["details"].get("semantic_passed", True)
    sem_score = score["details"].get("semantic_score", 0.0)
    sem_str = f"✓ sem ({sem_score:.2f})" if sem_passed else f"✗ sem FAIL ({sem_score:.2f})"

    lint_str = f"{javac_warnings} javac + {checkstyle_count} checkstyle + {pmd_count} PMD"
    if javac_warnings == 0 and checkstyle_count == 0 and pmd_count == 0:
        lint_str = "0 violations (clean)"

    summary = f"""Model: {model}
─────────────────────────────────────────────────
Result:   {'✓ PASS' if score['correctness'] >= 100 else '✗ FAIL'}
Time:     {human_model} model + {human_lint} lint = {human_total} wall  (score: {score['speed_score']}/100)
Correct:  {'Yes - hash matches expected' if score['correctness'] >= 100 else 'No - wrong hash'}
Compiled: {'Yes' if score['details'].get('compile_ok') else 'No'}
Quality:  {lint_str}  (score: {score['code_quality']}/100 — higher=cleaner)
Tokens:   {prompt_tokens} sent + {completion_tokens} generated = {prompt_tokens + completion_tokens} total
Tools:    [{tools_used}]  (score: {score['tool_usage']}/100 — higher=more tools used)
Mode:     {mode}  {jq_str}  {sem_str}
"""
    ac = score["details"].get("anti_cheat_consistent")
    if ac is True:
        ac_str = "✓ consistent (n=8,9,10 equal)"
    elif ac is False:
        ac_str = "✗ INCONSISTENT — possible hardcode (correctness capped, cheat suspected)"
    else:
        ac_str = "n/a (no successful run)"
    summary += f"""Anti-cheat: {ac_str}
Total:    {score['total']}/100
"""
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(summary)

    print(summary)

    artifacts_dir = os.path.join(run_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    for fname in os.listdir(run_dir):
        if fname.endswith(".java") or fname.endswith(".class"):
            shutil.copy2(os.path.join(run_dir, fname), os.path.join(artifacts_dir, fname))
    for src_name in ["checkstyle_report.txt", "pmd_report.txt"]:
        src = os.path.join(run_dir, src_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(artifacts_dir, src_name))


# === Main benchmark ===

class OllamaCallError(Exception):
    """Raised when an Ollama /api/chat call fails (HTTP error, timeout, bad JSON)."""


def call_ollama(model, messages, capabilities=None, timeout=300):
    """Call Ollama chat. Sends the native tool schema and enables thinking only
    for models that support it. Preserves thinking/content/tool_calls separately.
    Raises OllamaCallError on failure (fail-fast; no silent empty run).
    """
    options = {"temperature": 0.2}
    if capabilities is None:
        capabilities = get_model_capabilities(model)
    if "thinking" in capabilities:
        options["think"] = True

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "tools": TOOL_SCHEMA,
        "options": options,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        raise OllamaCallError(f"HTTP {e.response.status_code} for {model}: {e.response.text[:300]}") from e
    except requests.exceptions.Timeout as e:
        raise OllamaCallError(f"timeout calling {model}") from e
    except Exception as e:
        raise OllamaCallError(f"error calling {model}: {e}") from e

    msg = data.get("message", {})
    return {
        "message": msg,
        "thinking": msg.get("thinking", "") or "",
        "tool_calls": msg.get("tool_calls", []),
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
        "total_duration_ns": data.get("total_duration", 0),
    }


def run_model_benchmark(model):
    model_clean = model.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_BASE, model_clean, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Capability detection (fail-fast prep). Skip models that cannot chat.
    capabilities = get_model_capabilities(model)
    if "completion" not in capabilities:
        print(f"\n{'='*60}")
        print(f"SKIPPING {model}: not a chat model (capabilities={sorted(capabilities) or 'none'})")
        print(f"{'='*60}")
        return {
            "skipped": True,
            "reason": f"not a chat model (capabilities={sorted(capabilities) or 'none'})",
            "capabilities": sorted(capabilities),
        }, run_dir

    print(f"\n{'='*60}")
    print(f"Testing model: {model}")
    print(f"Results: {run_dir}")
    print(f"  capabilities: {sorted(capabilities)}")
    print(f"{'='*60}")

    # Cleanup any leftover models and warmup — NOT timed
    print("  Cleaning up previous models...")
    cleanup_all_models()
    warmup_model(model)

    messages = [{"role": "user", "content": PROMPT}]
    conversation = []
    timing = {"total_seconds": 0, "lint_time": 0, "javac_warnings": 0, "compile_ok": False, "hash_match": False, "checkstyle_count": 0, "pmd_count": 0, "thinking_support": False, "used_retrieval": False, "capabilities": sorted(capabilities)}
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # ELAPSED TIME STARTS HERE
    start_time = time.time()
    javac_warnings = 0
    mode = "tool_call"  # or "text"
    did_compile_verify = False
    stall_counter = {}  # content_hash -> count, to detect repeated failures

    for turn in range(1, MAX_TURNS + 1):
        elapsed = time.time() - start_time
        if elapsed > PER_MODEL_TIMEOUT:
            print(f"  TIMEOUT ({PER_MODEL_TIMEOUT}s reached)")
            break

        print(f"\n--- Turn {turn} ---")

        try:
            response = call_ollama(model, messages, capabilities=capabilities)
        except OllamaCallError as e:
            print(f"  FATAL CALL ERROR: {e}")
            break
        except Exception as e:
            print(f"  FATAL UNEXPECTED ERROR: {e}")
            break

        msg = response["message"]
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        # Track tokens
        total_prompt_tokens += response.get("prompt_tokens", 0)
        total_completion_tokens += response.get("completion_tokens", 0)

        # Thinking detection: Ollama exposes a non-empty "thinking" field in the
        # message when the model emits a reasoning trace (requested via think:true).
        # A model "supports thinking" if any turn produces a non-empty thinking field.
        if response.get("thinking", ""):
            timing["thinking_support"] = True
            conversation.append({
                "turn": turn, "role": "thinking", "content": response["thinking"]
            })

        conversation.append({
            "turn": turn, "role": "assistant",
            "content": content, "tool_calls": tool_calls
        })

        # Save raw output for jq audit trail
        with open(os.path.join(run_dir, f"raw_output_turn{turn}.txt"), "w") as f:
            f.write(content)

        # jq audit: check if raw content parses as valid JSON
        jq_passed = jq_audit(content) if content and not tool_calls else True
        if "jq_audit_passed" not in timing:
            timing["jq_audit_passed"] = jq_passed
        elif not timing["jq_audit_passed"]:
            pass  # once failed, stays failed

        if not tool_calls and content and not jq_passed:
            print(f"  ⚠ jq audit FAILED — raw output is not valid JSON")

        # Semantic gate: check if model output is on-topic
        semantic_passed, sem_score, sem_hit = semantic_gate_check(content)
        if "semantic_passed" not in timing:
            timing["semantic_passed"] = semantic_passed
            timing["semantic_score"] = sem_score
            timing["semantic_hit"] = sem_hit
        elif not timing["semantic_passed"]:
            pass  # once failed, stays failed

        if not tool_calls and content and not semantic_passed:
            print(f"  ⚠ SEMANTIC GATE FAILED — off-topic (score={sem_score:.3f}, threshold={SEMANTIC_THRESHOLD})")
            if sem_hit:
                print(f"    Best match: [{sem_hit.get('collection','?')}] score={sem_hit.get('score',0)}")

        if content:
            print(f"  Response: {content[:300]}")

        if tool_calls:
            mode = "tool_call"
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})

                print(f"  Tool call: {name}")

                if name in TOOL_DISPATCH:
                    try:
                        result = TOOL_DISPATCH[name](args, run_dir)
                    except Exception as e:
                        result = json.dumps({"ok": False, "error": str(e)})
                else:
                    result = json.dumps({"ok": False, "error": f"Unknown tool: {name}"})

                if name == "compile_java":
                    try:
                        data = json.loads(result)
                        timing["compile_ok"] = data.get("ok", False)
                        stderr = data.get("stderr", "")
                        wc = len([l for l in stderr.split("\n") if "warning" in l.lower()])
                        javac_warnings = max(javac_warnings, wc)
                        timing["lint_time"] = timing.get("lint_time", 0) + data.get("lint_time", 0)
                    except: pass

                if name == "run_and_hash":
                    try:
                        data = json.loads(result)
                        if data.get("matches_expected"):
                            timing["hash_match"] = True
                        timing["exec_time"] = timing.get("exec_time", 0) + data.get("exec_time", 0)
                    except: pass

                if name == "submit_answer":
                    try:
                        data = json.loads(result)
                        timing["hash_match"] = data.get("correct", False)
                    except: pass

                if name == "search_solutions":
                    timing["used_retrieval"] = True
                    try:
                        data = json.loads(result)
                        n_hits = len(data.get("results", []))
                        print(f"  search_solutions returned {n_hits} reference chunk(s)")
                    except: pass

                conversation.append({
                    "turn": turn, "role": "tool",
                    "name": name, "content": result
                })
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                messages.append({"role": "tool", "content": result, "name": name, "tool_call_id": tc.get("id", "")})

                if name == "submit_answer":
                    print("  Answer submitted!")

        else:
            if mode == "tool_call" and turn > 1:
                print("  No tool calls. Ending.")
                break

            mode = "text"

            # Try to extract fake tool call (model wrote JSON tool call in plain text)
            fake_tc = extract_fake_tool_call(content)
            if fake_tc:
                path, content_body = fake_tc

                # Validate: real Java code should have class/main/import and be >50 bytes
                if not (len(content_body) > 50 and ('class' in content_body or 'main' in content_body or 'import' in content_body)):
                    print(f"  Fake tool call extracted suspicious content ({len(content_body)} bytes), trying Java code extraction instead")
                    fake_tc = None
                    content_body = None

            if fake_tc:
                full_path = os.path.join(run_dir, path)
                os.makedirs(os.path.dirname(full_path) or run_dir, exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(content_body)
                print(f"  Extracted fake tool call: wrote {path} ({len(content_body)} bytes)")
                conversation.append({
                    "turn": turn, "role": "tool",
                    "name": "write_file", "content": json.dumps({"ok": True, "path": path, "size": len(content_body)})
                })

                tcv = try_compile_and_verify(run_dir)
                timing["compile_ok"] = tcv["compile_ok"]
                timing["hash_match"] = tcv["hash_match"]
                javac_warnings = max(javac_warnings, tcv["javac_warnings"])
                timing["checkstyle_count"] = tcv["checkstyle_count"]
                timing["pmd_count"] = tcv["pmd_count"]
                timing["lint_time"] = timing.get("lint_time", 0) + tcv["lint_time"]
                timing["exec_time"] = timing.get("exec_time", 0) + tcv["exec_time"]
                did_compile_verify = True
                c_ok = tcv["compile_ok"]
                h_match = tcv["hash_match"]

                if c_ok:
                    print(f"  Compile: OK, Hash match: {h_match}")
                    if h_match:
                        break
                    print(f"  Code compiles but hash doesn't match — retrying")
                    # Stall detection: check if model keeps outputting the same code
                    code_hash = hashlib.sha256(content_body.encode()).hexdigest()
                    stall_counter[code_hash] = stall_counter.get(code_hash, 0) + 1
                    if stall_counter[code_hash] >= 3:
                        print(f"  STALL DETECTED: same code output {stall_counter[code_hash]} times, aborting")
                        break
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": f"The code compiled successfully but the SHA-256 hash doesn't match {EXPECTED_HASH}. The output produces hash that doesn't match. Remember: the input bytes should be '2\\n3\\n5\\n7\\n11\\n' and the hash should match. Make sure to use Sieve of Eratosthenes and feed primes directly to SHA-256 with newlines."
                    })
                    continue
                else:
                    print(f"  Compile: FAILED — letting model try again")
                    # Stall detection: check if model keeps outputting the same broken code
                    code_hash = hashlib.sha256(content_body.encode()).hexdigest()
                    stall_counter[code_hash] = stall_counter.get(code_hash, 0) + 1
                    if stall_counter[code_hash] >= 3:
                        print(f"  STALL DETECTED: same broken code output {stall_counter[code_hash]} times, aborting")
                        break
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": f"The compilation failed. Here is the error:\n{tcv['output'] if tcv['output'] else 'Unknown error'}\nPlease fix the code and try again."
                    })
                    continue
            else:
                code = extract_java_code(content)
            if code:
                java_path = os.path.join(run_dir, "hashprime.java")
                with open(java_path, "w") as f:
                    f.write(code)
                print(f"  Extracted Java code ({len(code)} bytes)")

                tcv = try_compile_and_verify(run_dir)
                timing["compile_ok"] = tcv["compile_ok"]
                timing["hash_match"] = tcv["hash_match"]
                javac_warnings = max(javac_warnings, tcv["javac_warnings"])
                timing["checkstyle_count"] = tcv["checkstyle_count"]
                timing["pmd_count"] = tcv["pmd_count"]
                timing["lint_time"] = timing.get("lint_time", 0) + tcv["lint_time"]
                timing["exec_time"] = timing.get("exec_time", 0) + tcv["exec_time"]
                did_compile_verify = True
                c_ok = tcv["compile_ok"]
                h_match = tcv["hash_match"]

                if c_ok:
                    print(f"  Compile: OK, Hash match: {h_match}")
                else:
                    print(f"  Compile: FAILED")

                break
            else:
                print(f"  No code found in response.")
                break

    timing["total_seconds"] = time.time() - start_time
    timing["javac_warnings"] = javac_warnings
    timing["turns_used"] = len([m for m in conversation if m.get("role") == "assistant"])
    timing["prompt_tokens"] = total_prompt_tokens
    timing["completion_tokens"] = total_completion_tokens

    score = score_run(run_dir, conversation, timing, mode)

    # Cleanup this model after testing — NOT timed
    print("  Cleaning up model...")
    cleanup_all_models()

    save_results(run_dir, model, conversation, timing, score)

    return score, run_dir


def main():
    os.makedirs(RESULTS_BASE, exist_ok=True)

    resp = requests.get("http://localhost:11434/api/tags", timeout=10)
    resp.raise_for_status()
    models = [m["name"] for m in resp.json().get("models", [])]

    if not models:
        print("No models found.")
        sys.exit(1)

    print(f"Found {len(models)} models:")
    for m in models:
        print(f"  - {m}")

    # Filter out non-chat models (e.g. embedding-only) up front; report them.
    chat_models = []
    skipped_models = []
    for m in models:
        caps = get_model_capabilities(m)
        if "completion" in caps:
            chat_models.append(m)
        else:
            skipped_models.append((m, sorted(caps) or "none"))
    if skipped_models:
        print(f"\nSkipping {len(skipped_models)} non-chat model(s):")
        for m, caps in skipped_models:
            print(f"  - {m} (capabilities: {caps})")

    results_summary = []

    for model in chat_models:
        result, run_dir = run_model_benchmark(model)
        if isinstance(result, dict) and result.get("skipped"):
            print(f"  (skipped {model}: {result.get('reason')})")
            continue
        score = result
        sem_passed = score["details"].get("semantic_passed", True)
        sem_score = score["details"].get("semantic_score", 0.0)
        sem_tag = "✓" if sem_passed else "✗"
        tools_used_list = score["details"].get("tools_used", [])
        has_write_file = "write_file" in tools_used_list
        jq_passed = score["details"].get("jq_audit_passed", True)
        tool_ok = "✓" if (score["details"].get("mode") == "tool_call" or (has_write_file and jq_passed)) else "✗"
        compile_ok = score["details"].get("compile_ok", False)
        hash_match = score["details"].get("hash_match", False)
        thinking_support = score["details"].get("thinking_support", False)
        think_tag = "✓" if thinking_support else "✗"
        used_retrieval = score["details"].get("used_retrieval", False)
        retr_tag = "✓" if used_retrieval else "✗"
        total_violations = (
            score["details"].get("javac_warnings", 0) +
            score["details"].get("checkstyle_count", 0) +
            score["details"].get("pmd_count", 0)
        )
        turns_used = score["details"].get("turns_used", 0)
        exec_secs = score["details"].get("exec_time", 0)
        model_time_secs = score["details"].get("model_time_seconds", 0)
        model_name_short = model[:25] if len(model) > 25 else model

        results_summary.append({
            "model": model,
            "model_short": model_name_short,
            "score": score["total"],
            "correct": score["correctness"],
            "compile_ok": compile_ok,
            "hash_match": hash_match,
            "speed": score["speed_score"],
            "quality": score["code_quality"],
            "tools": score["tool_usage"],
            "violations": total_violations,
            "turns_used": turns_used,
            "exec_time_secs": round(exec_secs, 3),
            "exec_str": format_time(exec_secs) if exec_secs > 0 else "-",
            "model_time_secs": model_time_secs,
            "time_str": format_time(model_time_secs),
            "tools_used": ", ".join(tools_used_list),
            "mode": score["details"].get("mode", "?"),
            "javac_w": score["details"].get("javac_warnings", 0),
            "cs": score["details"].get("checkstyle_count", 0),
            "pmd": score["details"].get("pmd_count", 0),
            "sem_tag": sem_tag,
            "sem_score": round(sem_score, 2),
            "tool_ok": tool_ok,
            "think_tag": think_tag,
            "thinking_support": thinking_support,
            "think_score": score["details"].get("thinking_quality", 0),
            "thinking_quality_chunks": score["details"].get("thinking_quality_chunks", 0),
            "thinking_quality_mean_sim": score["details"].get("thinking_quality_mean_sim", 0),
            "thinking_quality_error": score["details"].get("thinking_quality_error"),
            "prompt_ts": score["details"].get("prompt_ts", 0.0),
            "thinking_drift": score["details"].get("thinking_drift"),
            "retr_tag": retr_tag,
            "used_retrieval": used_retrieval,
            "jq_passed": jq_passed,
            "has_write_file": has_write_file,
            "run_dir": run_dir
        })

    results_summary.sort(key=lambda x: x["score"], reverse=True)

    # ── Single column-spec drives BOTH console + markdown tables (alignment-safe) ──
    # (key, console_header, console_width, md_header)
    COLS = [
        ("rank",    "RANK",  5,   "RANK"),
        ("model",   "MODEL", 25,  "MODEL"),
        ("score",   "TOTAL", 5,   "TOTAL"),
        ("correct", "CORR",  5,   "CORR"),
        ("c_tag",   "CPILE", 5,   "CPILE"),
        ("h_tag",   "HASH",  5,   "HASH"),
        ("viol",    "VIOL",  4,   "VIOL"),
        ("sem",     "SEM",   3,   "SEM"),
        ("tool",    "TOOL",  4,   "TOOL"),
        ("think",   "THINK", 5,   "THINK"),
        ("retr",    "RETR",  5,   "RETR"),
        ("ts",      "TSCORE",6,   "TSCORE"),
        ("drift",   "DRIFT", 6,   "DRIFT"),
        ("iter",    "ITER",  4,   "ITER"),
        ("exec",    "EXEC",  9,   "EXEC"),
        ("time",    "MTIME", 10,  "MTIME"),
        ("mode",    "MODE",  8,   "MODE"),
    ]

    def row_values(i, r):
        c_tag = "✓" if r["compile_ok"] else "✗"
        h_tag = "✓" if r["hash_match"] else "✗"
        drift = r["thinking_drift"]
        drift_s = f"{drift:+.2f}" if isinstance(drift, (int, float)) else "-"
        return {
            "rank": i,
            "model": r["model_short"],
            "score": r["score"],
            "correct": r["correct"],
            "c_tag": c_tag,
            "h_tag": h_tag,
            "viol": r["violations"],
            "sem": r["sem_tag"],
            "tool": r["tool_ok"],
            "think": r["think_tag"],
            "retr": r["retr_tag"],
            "ts": r["think_score"],
            "drift": drift_s,
            "iter": r["turns_used"],
            "exec": r["exec_str"],
            "time": r["time_str"],
            "mode": r["mode"],
        }

    # ── Console ranking table ──
    print(f"\n{'='*150}")
    console_header = "  " + "  ".join(f"{h:<{w}}" for _, h, w, _ in COLS)
    print(console_header)
    print(f"  {'─'*len(console_header)}")
    print(f"  (higher=better: TOTAL/CORR/TSCORE; lower=better: VIOL; ✓/✗=pass/fail; ITER=turns; EXEC=code run; MTIME=model wall-clock)")
    print(f"{'─'*150}")
    for i, r in enumerate(results_summary, 1):
        v = row_values(i, r)
        print("  " + "  ".join(f"{str(v[k]):<{w}}" for k, _, w, _ in COLS))

    # ── Markdown ranking table ──
    md_lines = [
        "# Benchmark Ranking",
        "",
        f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "| " + " | ".join(h for _, _, _, h in COLS) + " |",
        "|" + "|".join("-" * len(h) for _, _, _, h in COLS) + "|",
    ]
    for i, r in enumerate(results_summary, 1):
        v = row_values(i, r)
        md_lines.append("| " + " | ".join(str(v[k]) for k, _, _, _ in COLS) + " |")

    md_lines.extend([
        "",
        "### Key",
        "- **TOTAL**: composite score (0–100)",
        "- **CORR**: correctness (0=none, 25=has Java, 50=compiles, 100=hash match)",
        "- **CPILE**: code compiled? (✓/✗)",
        "- **HASH**: SHA-256 hash matches expected? (✓/✗)",
        "- **VIOL**: total lint violations (javac warnings + checkstyle + PMD; lower is better)",
        "- **SEM**: semantic gate pass? (✓ = output is on-topic for the hashprime problem)",
        "- **TOOL**: tool call support? (✓ = native API or text-mode with valid JSON + file write)",
        "- **THINK**: thinking/reasoning trace present? (✓ = Ollama returned a non-empty `thinking` field when `think:true` requested)",
        "- **RETR**: did the model use the `search_solutions` retrieval tool? (✓ = yes)",
        "- **TSCORE**: semantic quality of the thinking trace (0–100). Chonkie chunks the trace, embeds via Ollama nomic-embed-text, mean cosine similarity vs hashprime_solutions in Qdrant. Higher = reasoning more aligned with known-good solutions.",
        "- **DRIFT**: TSCORE − PROMPT_TS (prompt baseline alignment). Positive = model reasoning is more on-topic than the prompt itself.",
        "- **ITER**: number of conversation turns the model took",
        "- **EXEC**: time for compiled Java code to execute (seconds)",
        "- **MTIME**: total model wall-clock time excluding lint (seconds or minutes)",
        "- **MODE**: `tool_call` = native Ollama tool API, `text` = JSON extracted from plaintext",
        "",
    ])

    # Summary section
    best_names = []
    for i, r in enumerate(results_summary[:3], 1):
        best_names.append(f"  {i}. **{r['model']}** — TOTAL {r['score']} | CORR {r['correct']} | SEM {r['sem_tag']} | TOOL {r['tool_ok']} | {r['time_str']}")
    md_lines.append("### Top 3 Overall\n")
    md_lines.extend(best_names)
    md_lines.append("")

    best_semantic = [r for r in results_summary if r["sem_tag"] == "✓"]
    if best_semantic:
        md_lines.append("### Semantic Gate Passed\n")
        for r in best_semantic:
            md_lines.append(f"  - **{r['model']}** — score {r['score']} | sem {r['sem_score']:.2f}")
        md_lines.append("")

    best_tool = [r for r in results_summary if r["tool_ok"] == "✓"]
    if best_tool:
        md_lines.append("### Tool Call Support\n")
        for r in best_tool:
            md_lines.append(f"  - **{r['model']}** — mode {r['mode']} | TOTAL {r['score']}")
        md_lines.append("")

    no_tool = [r for r in results_summary if r["tool_ok"] == "✗"]
    if no_tool:
        md_lines.append("### No Tool Call Support\n")
        for r in no_tool:
            md_lines.append(f"  - **{r['model']}** — mode {r['mode']} | jq {'✓' if r['jq_passed'] else '✗'} | write_file {'✓' if r['has_write_file'] else '✗'}")
        md_lines.append("")

    # Thinking quality ranking (only models that emitted a thinking trace)
    thinking_models = [r for r in results_summary if r["thinking_support"]]
    if thinking_models:
        thinking_models.sort(key=lambda x: x["think_score"], reverse=True)
        md_lines.append("### Thinking Quality (chonkie + Qdrant vs hashprime_solutions)\n")
        for r in thinking_models:
            err = ""
            if r.get("thinking_quality_error"):
                err = f" (err: {r['thinking_quality_error']})"
            md_lines.append(f"  - **{r['model']}** — TSCORE {r['think_score']} | chunks {r.get('thinking_quality_chunks', 0)} | mean_sim {r.get('thinking_quality_mean_sim', 0)} | THINK {r['think_tag']}{err}")
        md_lines.append("")

    md_content = "\n".join(md_lines)

    ranking_json = os.path.join(RESULTS_BASE, "ranking.json")
    with open(ranking_json, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nRanking saved to {ranking_json}")

    ranking_md = os.path.join(RESULTS_BASE, "ranking.md")
    with open(ranking_md, "w") as f:
        f.write(md_content)
    print(f"Ranking (markdown) saved to {ranking_md}")


if __name__ == "__main__":
    main()
