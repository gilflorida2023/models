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

# Heavy/optional deps are imported lazily so the module still loads (e.g. for
# standalone Java verification) on a machine without numpy / qdrant / chonkie.
try:
    import requests  # noqa: F401
except ImportError:
    requests = None
try:
    import numpy as np
except ImportError:
    np = None
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams
except ImportError:
    QdrantClient = None
    Distance = VectorParams = None

RESULTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OLLAMA_URL = "http://localhost:11434/api/chat"
PER_MODEL_TIMEOUT = 600
MAX_TURNS = 20
EXPECTED_HASH = "563d8e0603dcc07d784135d99fd81ff6bf98495e898ec1f52e2e7605320cf6dc"

# Large-N validation: confirm the sieve actually scales (and isn't a hardcoded
# small-N trick) by checking the prime list against the authoritative OEIS
# A000040 manifest (ascii_integer_lf formatting) at two checkpoints.
MANIFEST_PATH = "/home/scout/projects/math-kat/manifests/A000040.json"
LARGE_CHECKPOINTS = {
    1_000_000: {"hash": "4883963dd4510a29d6df2ffe4dd11e4e1a910e815c7810b200c77b3357f22a28", "count": 78498},
    10_000_000: {"hash": "36d6197802bc3b635b43b31cd6a2583f7cf8f5badff7992f3693c5102beefd14", "count": 664579},
}


def load_manifest_checkpoints():
    """Load checkpoint hashes/counts from the A000040 manifest; fall back to the
    hardcoded constants above if the manifest is missing/unreadable."""
    out = {k: dict(v) for k, v in LARGE_CHECKPOINTS.items()}
    try:
        with open(MANIFEST_PATH) as f:
            data = json.load(f)
        for n, chk in data.get("checkpoint_hashes", {}).items():
            try:
                ni = int(n)
            except ValueError:
                continue
            if ni in out:
                out[ni]["hash"] = chk.get("hash", out[ni]["hash"])
                out[ni]["count"] = chk.get("count", out[ni]["count"])
    except Exception:
        pass
    return out


MANIFEST_CP = load_manifest_checkpoints()
LARGE_N = 1_000_000
LARGE_N2 = 10_000_000
LARGE_N_EXPECTED_HASH = MANIFEST_CP[LARGE_N]["hash"]
LARGE_N_EXPECTED_COUNT = MANIFEST_CP[LARGE_N]["count"]
LARGE_N2_EXPECTED_HASH = MANIFEST_CP[LARGE_N2]["hash"]
LARGE_N2_EXPECTED_COUNT = MANIFEST_CP[LARGE_N2]["count"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
CHECKSTYLE_JAR = os.path.join(TOOLS_DIR, "checkstyle.jar")
CHECKSTYLE_CONFIG = os.path.join(TOOLS_DIR, "checkstyle_config.xml")
PMD_BIN = os.path.join(TOOLS_DIR, "pmd", "bin", "pmd")
JUNIT_JAR = os.path.join(BASE_DIR, "junit", "junit-platform-console-standalone.jar")
JUNIT_SRC = os.path.join(BASE_DIR, "junit", "HashprimeVerificationTest.java")
JUNIT_OUT = os.path.join(BASE_DIR, "junit", "out")


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

Tool: read_file
  Description: Read the full contents of a file in the working directory and return them as text. Use it to inspect source you wrote with write_file (e.g. hashprime.java), to re-read a file before editing it, or to confirm what a prior step produced. Returns {ok:true, path, content, size} on success, or {ok:false, error:"file not found"} if the path does not exist. Paths are relative to the working directory.
  Arguments: path (string) - file to read, relative to the working directory
  Example: {"name": "read_file", "arguments": {"path": "hashprime.java"}}

Tool: compile_java
  Description: Compile hashprime.java with javac
  Arguments: none
  Example: {"name": "compile_java", "arguments": {}}

Tool: run
  Description: Run 'java hashprime N' and return its SHA-256 hash
  Arguments: n (integer) - upper limit for prime finding
  Example: {"name": "run", "arguments": {"n": 12}}

Tool: submit_answer
  Description: Submit the final SHA-256 hash as the answer
  Arguments: hash (string), expected (string)
  Example: {"name": "submit_answer", "arguments": {"hash": "563d...", "expected": "563d..."}}

Tool: search_solutions
  Description: Search the indexed reference corpus (prior hashprime.java solutions, algorithm/data-structure patterns, and sysadmin notes) for relevant code and explanations. Use it to ground your approach.
  Arguments: query (string, optional) - what to look for; if omitted the task is used
  Example: {"name": "search_solutions", "arguments": {"query": "Sieve of Eratosthenes feed primes to SHA-256"}}
"""

def _load_prompt_body():
    """Load the task-specific prompt body from prompt.hashprime.txt so the
    prompt can be edited on disk without touching this script."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.hashprime.txt")
    with open(path) as f:
        return f.read()


PROMPT = "You are solving a Java programming challenge. You have access to tools to write files, compile, run commands, and verify results.\n" + TOOL_DESCRIPTIONS + "\n" + _load_prompt_body()


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
        }

        if result.returncode == 0:
            cs_count, cs_lines, cs_time = run_checkstyle(full_path, run_dir)
            pmd_count, pmd_lines, pmd_time = run_pmd(full_path, run_dir)
            response["checkstyle_count"] = cs_count
            response["checkstyle_output"] = "\n".join(cs_lines)
            response["pmd_count"] = pmd_count
            response["pmd_output"] = "\n".join(pmd_lines)
            if cs_lines:
                with open(os.path.join(run_dir, "checkstyle_report.txt"), "w") as f:
                    f.write("\n".join(cs_lines) + "\n")
            if pmd_lines:
                with open(os.path.join(run_dir, "pmd_report.txt"), "w") as f:
                    f.write("\n".join(pmd_lines) + "\n")
            # LINT IS USER-ONLY: never sent to the model. The model only ever
            # sees the javac output (ok/returncode/stdout/stderr) so it can fix
            # its own code; checkstyle/PMD violations are for the user's report.
            print(f"  ── LINT (user-only, NOT sent to model) ── "
                  f"checkstyle={cs_count}, pmd={pmd_count}")
        else:
            print(f"  ── LINT skipped (compile failed) ──")

        # Build the result string FED TO THE MODEL: javac output only, lint
        # content (checkstyle/PMD) stripped. lint is USER-ONLY, never sent to
        # the model. (Full response w/ lint is kept for the user in score.json.)
        model_response = {
            "ok": response["ok"],
            "returncode": response["returncode"],
            "stdout": response["stdout"],
            "stderr": response["stderr"],
        }
        return json.dumps(model_response)
    except subprocess.TimeoutExpired:
        return json.dumps({"ok": False, "error": "Compilation timed out"})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


def tool_run(args, run_dir):
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
        # The program prints ONLY the SHA-256 hash (lowercase hex) to stdout.
        # Validate by comparing that printed hash against the EXPECTED digest for
        # the requested N: small-N (n in {11,12}) uses EXPECTED_HASH; the large-N
        # checkpoints use their manifest digests. For any other N we can't verify
        # against a known digest, so matches_expected is None (unknown) rather
        # than a misleading false.
        actual = output.strip().lower()
        sha = actual
        if n in (11, 12):
            expected = EXPECTED_HASH
        elif n == LARGE_N:
            expected = LARGE_N_EXPECTED_HASH
        elif n == LARGE_N2:
            expected = LARGE_N2_EXPECTED_HASH
        else:
            expected = None
        matches = (sha == expected) if expected is not None else None
        return json.dumps({
            "ok": True,
            "n": n,
            "output": output,
            "sha256": sha,
            "expected": expected,
            "matches_expected": matches,
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
    collections_checked = 0
    for coll in SEARCH_COLLECTIONS:
        try:
            qdrant.get_collection(coll)
        except Exception:
            # Collection doesn't exist in this store — skip, but record it.
            continue
        collections_checked += 1
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
                "verified": bool(payload.get("verified", False)),
                "text": (payload.get("text", "") or "")[:800],
            })
    if not results:
        # Distinguish "store is empty/unavailable" from "no relevant match" so
        # the model isn't misled into thinking retrieval is unsupported.
        note = ("No reference chunks found for this query."
                if collections_checked
                else "Reference corpus unavailable/empty (no collections loaded).")
        return json.dumps({"ok": True, "query": query[:200], "results": [],
                            "collections_checked": collections_checked,
                            "note": note})
    # Surface verified (certified-correct) solutions first so a model that
    # chooses to consult prior art sees known-good code ahead of unverified.
    results.sort(key=lambda r: (not r.get("verified", False), -r["score"]))
    return json.dumps({"ok": True, "query": query[:200],
                        "verified_count": sum(1 for r in results if r.get("verified")),
                        "results": results})


def _required_args_for(name):
    """Return the list of required argument names for a tool from TOOL_SCHEMA,
    used to build model-actionable error hints for missing arguments."""
    for entry in TOOL_SCHEMA:
        fn = entry.get("function", {})
        if fn.get("name") == name:
            return list(fn.get("parameters", {}).get("required", []))
    return []


def _run_junit(run_dir):
    """Compile + run the standalone JUnit verification (HashprimeVerificationTest)
    against the model's hashprime.class. Returns (passed, total, failed, error).

    This is an ALTERNATIVE scale-proof to the large-N manifest hash: a full
    JUnit sweep over many N (small/large) with zero failures proves the sieve
    is correct AND scales, independent of any hardcoded single-hash trick.
    """
    if not os.path.exists(JUNIT_JAR):
        return 0, 0, 0, "junit standalone jar missing"
    if not os.path.exists(os.path.join(run_dir, "hashprime.class")):
        return 0, 0, 0, "hashprime.class not found (compile first)"
    try:
        os.makedirs(JUNIT_OUT, exist_ok=True)
        # The JUnit test runs `java -cp hashprime.dir hashprime N` itself, so it
        # needs the compiled class in run_dir (already there) and the test class
        # on the launcher classpath. Compile ONLY the test into JUNIT_OUT.
        cf = subprocess.run(
            ["javac", "-cp", JUNIT_JAR, "-d", JUNIT_OUT, JUNIT_SRC],
            capture_output=True, text=True, timeout=60, cwd=run_dir
        )
        if cf.returncode != 0:
            return 0, 0, 0, "junit compile failed: " + cf.stderr[:200]
        # Point the test at the model's run_dir (hashes its class) + the manifest.
        props = [
            f"-Dhashprime.dir={os.path.abspath(run_dir)}",
            f"-Dmanifest.file={MANIFEST_PATH}",
            f"-Dmax.n=1000000",
        ]
        r = subprocess.run(
            ["java", *props, "-jar", JUNIT_JAR, "--classpath", JUNIT_OUT,
             "--select-class", "HashprimeVerificationTest", "--details=tree"],
            capture_output=True, text=True, timeout=600, cwd=run_dir
        )
        out = (r.stdout or "") + (r.stderr or "")
        total = failed = 0
        m_total = re.search(r"(\d+)\s+tests found", out)
        m_fail = re.search(r"(\d+)\s+tests failed", out)
        if m_total:
            total = int(m_total.group(1))
        if m_fail:
            failed = int(m_fail.group(1))
        passed = max(total - failed, 0)
        err = "" if failed == 0 else out[-300:]
        return passed, total, failed, err
    except subprocess.TimeoutExpired:
        return 0, 0, 0, "junit run timed out"
    except Exception as e:
        return 0, 0, 0, f"junit error: {e}"


def tool_run_junit_verification(args, run_dir):
    """Run the JUnit verification suite over the model's hashprime class.
    Returns pass/fail counts and the scale-proof result. Lint/user-only
    semantics do not apply; this is a correctness harness fed back to the model
    so it can see exactly which N failed."""
    passed, total, failed, err = _run_junit(run_dir)
    return json.dumps({
        "ok": failed == 0 and total > 0,
        "passed": passed,
        "total": total,
        "failed": failed,
        "error": err,
    })


TOOL_DISPATCH = {
    "write_file": tool_write_file,
    "read_file": tool_read_file,
    "run_command": tool_run_command,
    "compile_java": tool_compile_java,
    "run": tool_run,
    "submit_answer": tool_submit_answer,
    "search_solutions": tool_search_solutions,
    "run_junit_verification": tool_run_junit_verification,
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
            "name": "read_file",
            "description": (
                "Read the full contents of a file in the working directory and "
                "return them as text. Use it to inspect source you previously "
                "wrote with write_file (e.g. hashprime.java), or to re-read a "
                "file before editing it, or to confirm what a prior step produced. "
                "Returns {ok:true, path, content, size} on success, or "
                "{ok:false, error:'file not found'} if the path does not exist. "
                "Paths are relative to the working directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file to read, relative to the working directory (e.g. 'hashprime.java')."},
                },
                "required": ["path"],
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
            "name": "run",
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
        {
            "type": "function",
            "function": {
                "name": "run_junit_verification",
                "description": "Run the JUnit verification suite (HashprimeVerificationTest) over the compiled hashprime class across many N values. Provides an independent scale-proof (correct at small AND large N) beyond the single-hash manifest check.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]


# Models that are NOT chat models (embedding-only, etc.). Used to invert the
# capability gate: we skip ONLY these, and default-allow everything else so a
# transient empty/missing `capabilities` list never wrongly skips a capable
# chat model (e.g. qwen3.5:*, glm4:9b, llama3.1:8b).
EMBEDDING_CAPS = {"embedding"}
EMBEDDING_FAMILIES = {"nomic-bert", "bert", "e5", "bge", "gte", "minilm"}

_CAP_CACHE = {}


def get_model_capabilities(model):
    """Return the set of Ollama capabilities for a model via POST /api/show.
    E.g. {'completion', 'tools', 'thinking'}.

    Returns an empty set on error/transient failure rather than raising. The
    caller should use is_chat_model() to decide skipping, which treats an
    empty/absent capability list as CHAT-CAPABLE (we only exclude models that
    explicitly report embedding capability), so transient empty responses do
    not wrongly skip capable models.
    """
    if model in _CAP_CACHE:
        return _CAP_CACHE[model]
    caps = set()
    if requests is None:
        return caps
    for attempt in range(2):  # one retry for transient Ollama blips
        try:
            resp = requests.post("http://localhost:11434/api/show",
                                 json={"model": model}, timeout=30)
            resp.raise_for_status()
            caps = set(resp.json().get("capabilities", []) or [])
            break
        except Exception:
            if attempt == 0:
                time.sleep(0.5)
            continue
    _CAP_CACHE[model] = caps
    return caps


def get_model_family(model):
    """Return the Ollama model 'family' from /api/show details, or '' on error.
    Used to identify embedding-only models by family when capabilities is empty.
    """
    if requests is None:
        return ""
    try:
        resp = requests.post("http://localhost:11434/api/show",
                             json={"model": model}, timeout=30)
        resp.raise_for_status()
        return (resp.json().get("details", {}) or {}).get("family", "") or ""
    except Exception:
        return ""


def is_chat_model(model):
    """Decide whether a model should be benchmarked.

    We skip ONLY models that are explicitly embedding-only:
      - capabilities contains 'embedding', OR
      - details.family is a known embedding family.
    Everything else is treated as chat-capable, even if Ollama returns an empty
    capability list (transient/newer-tag models like qwen3.5). This prevents
    capable models from being wrongly skipped as 'not a chat model'.
    """
    caps = get_model_capabilities(model)
    if caps & EMBEDDING_CAPS:
        return False
    fam = get_model_family(model)
    if fam and fam.lower() in EMBEDDING_FAMILIES:
        return False
    return True


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


def _pretty_json(obj, max_chars=200000):
    """Pretty-print anything as indented JSON for the log. Falls back to a
    UTF-8-safe, head+tail-truncated repr if the value isn't JSON-serializable
    or is pathologically large — so logging never raises and never floods."""
    try:
        text = json.dumps(obj, indent=2)
    except Exception:
        text = repr(obj)
    # Sanitize to valid UTF-8 first so binary/garbage can't corrupt the stream.
    text = text.encode("utf-8", "replace").decode("utf-8")
    if len(text) > max_chars:
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2 :]
        text = f"{head}\n… (+{len(text) - max_chars} more chars truncated) …\n{tail}"
    return text


def _safe_log(text, limit=2000):
    """Return a log-safe version of arbitrary text: valid UTF-8, head+tail
    truncated at `limit` chars, wrapped so callers can never raise on logging.
    Use this for malformed/oversized tool calls / raw model output so a bad
    payload is recorded but can never kill or corrupt the log stream."""
    try:
        s = str(text).encode("utf-8", "replace").decode("utf-8")
    except Exception:
        s = "<unloggable content>"
    if len(s) > limit:
        s = s[: limit // 2] + f"\n… (+{len(s) - limit} more chars truncated) …\n" + s[-limit // 2 :]
    return s


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
    Returns (passed, max_score, best_hit):
      - passed=True  -> on-topic (measured)
      - passed=False -> off-topic (measured, genuinely fails the gate)
      - passed=None  -> UNAVAILABLE: Qdrant/embedding down or collection missing,
                         so on-topic-ness could NOT be measured (do NOT treat as fail)
    """
    if not text or len(text.strip()) < 20:
        return False, 0.0, {}
    try:
        qdrant = QdrantClient(path=QDRANT_PATH)
        # Probe whether the scoring collection exists up front. If it's missing,
        # report UNAVAILABLE (None) rather than a misleading off-topic FAIL.
        try:
            qdrant.get_collection("hashprime_solutions")
        except Exception:
            return None, 0.0, {"error": "collection_missing"}
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
        # Embedding/Qdrant infra error -> UNAVAILABLE, not a model failure.
        return None, 0.0, {"error": str(e)}


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
            return 0, 0, 0.0, "UNAVAILABLE"

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
        # Embedding/Qdrant infra error -> UNAVAILABLE, not a model failure.
        return 0, 0, 0.0, "UNAVAILABLE"


# === Compile and verify (used for text-only models) ===

def _verify_large_n(run_dir, n, exp_hash, exp_count, prefix):
    """Run hashprime at N=n, hash the output, compare to the manifest.
    Returns (match, count_match, exec_time, hash, error).

    Retries transient java failures (intermittent ClassNotFoundException /
    classpath races seen under subprocess with an absolute -cp), and uses
    `cwd=run_dir` with `-cp .` to match the working interactive invocation.
    """
    rd_abs = os.path.abspath(run_dir)
    last_err = ""
    for attempt in range(3):
        start = time.time()
        try:
            r = subprocess.run(
                ["java", "-cp", ".", "hashprime", str(n)],
                capture_output=True, text=True, timeout=600, cwd=rd_abs
            )
            exec_time = time.time() - start
            if r.returncode == 0:
                # The program prints ONLY the SHA-256 hash (lowercase hex) to
                # stdout. Compare that printed hash to the manifest digest.
                h = r.stdout.strip().lower()
                if not h:
                    last_err = "program printed no hash to stdout"
                    if attempt < 2:
                        time.sleep(0.5)
                        continue
                    return False, False, exec_time, "", last_err
                match = (h == exp_hash)
                count_match = match  # hash match implies the prime set is correct
                print(f"  ── large-N ({prefix}) ── hash_match={match} "
                      f"({exec_time:.2f}s)")
                return match, count_match, exec_time, h, ""
            last_err = f"java rc={r.returncode}: {r.stderr[:200]}"
            if attempt < 2:
                time.sleep(0.5)
                continue
            return False, False, exec_time, "", last_err
        except Exception as e:
            last_err = str(e)
            if attempt < 2:
                time.sleep(0.5)
                continue
            return False, False, time.time() - start, "", last_err
    return False, False, 0.0, "", last_err


def try_compile_and_verify(run_dir):
    """Compile hashprime.java, run it, check hash, run lint tools.
    Returns dict with compile_ok, hash_match, javac_warnings, checkstyle_count, pmd_count, exec_time, output."""
    java_file = os.path.join(run_dir, "hashprime.java")
    result = {
        "compile_ok": False, "hash_match": False, "javac_warnings": 0,
        "checkstyle_count": 0, "pmd_count": 0, "exec_time": 0, "output": "",
        "large_n_match": False, "large_n_count_match": False,
        "large_n_exec_time": 0, "large_n_hash": "", "large_n_error": "",
        "large_n2_match": False, "large_n2_count_match": False,
        "large_n2_exec_time": 0, "large_n2_hash": "", "large_n2_error": "",
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
    if cs_lines:
        with open(os.path.join(run_dir, "checkstyle_report.txt"), "w") as f:
            f.write("\n".join(cs_lines) + "\n")
    if pmd_lines:
        with open(os.path.join(run_dir, "pmd_report.txt"), "w") as f:
            f.write("\n".join(pmd_lines) + "\n")

    # Run and verify hash
    for n in [11, 12]:
        matched = False
        for attempt in range(3):
            try:
                start = time.time()
                r = subprocess.run(
                    ["java", "-cp", ".", "hashprime", str(n)],
                    capture_output=True, text=True, timeout=30, cwd=run_dir
                )
                result["exec_time"] += time.time() - start
                if r.returncode == 0:
                    result["output"] = r.stdout
                    # The program prints ONLY the SHA-256 hash to stdout. Compare
                    # the printed hash to the expected digest for n=11/12.
                    actual = r.stdout.strip().lower()
                    if actual == EXPECTED_HASH:
                        matched = True
                        break
                    else:
                        if attempt < 2:
                            time.sleep(0.5)
                            continue
                else:
                    if attempt < 2:
                        time.sleep(0.5)
                        continue
            except Exception:
                if attempt < 2:
                    time.sleep(0.5)
                    continue
        if matched:
            result["hash_match"] = True

    # Large-N validation against the A000040 manifest at two checkpoints.
    # Proves the sieve scales and isn't a hardcoded small-N trick.
    m, cm, et, h, err = _verify_large_n(run_dir, LARGE_N, LARGE_N_EXPECTED_HASH, LARGE_N_EXPECTED_COUNT, "1e6")
    result["large_n_match"], result["large_n_count_match"] = m, cm
    result["large_n_exec_time"], result["large_n_hash"], result["large_n_error"] = et, h, err

    m2, cm2, et2, h2, err2 = _verify_large_n(run_dir, LARGE_N2, LARGE_N2_EXPECTED_HASH, LARGE_N2_EXPECTED_COUNT, "1e7")
    result["large_n2_match"], result["large_n2_count_match"] = m2, cm2
    result["large_n2_exec_time"], result["large_n2_hash"], result["large_n2_error"] = et2, h2, err2

    return result


def _copy_large_n(timing, tcv):
    """Copy the 1e6 validation results from a try_compile_and_verify result
    into the run timing dict (text-mode paths otherwise drop them)."""
    for k in ("large_n_match", "large_n_count_match", "large_n_exec_time",
              "large_n_hash", "large_n_error", "large_n_count",
              "large_n2_match", "large_n2_count_match", "large_n2_exec_time",
              "large_n2_hash", "large_n2_error"):
        timing[k] = tcv.get(k)


_PROMPT_TS_CACHE = None


def would_accept_submission(tcv, best_exec_time=None):
    """Stringent gate for accepting a model's solution into the corpus/DB.

    Any auto-submit path MUST call this first. Criteria:
      1. Compiles cleanly (no javac errors).
      2. Hash matches EXPECTED_HASH for n=11 and n=12 (correct output) — the
         natural validation that the program is written correctly.
      3. Speed gate (optional): if best_exec_time is given, only accept when
         this run is FASTER, to keep corpus pollution (slow/duplicate) down.

    Returns (accepted: bool, reason: str).
    """
    if not tcv.get("compile_ok"):
        return False, "rejected: does not compile"
    if not tcv.get("hash_match"):
        return False, "rejected: hash mismatch (incorrect output)"
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

    # Correctness: a correct solution compiles and produces the expected hash.
    # The n=7..10 hash equality is a natural property of any correct program
    # (same prime prefix) and needs no special enforcement — if it's written
    # right, it just works.
    hash_match = timing.get("hash_match", False)
    compile_ok = timing.get("compile_ok", False)
    large_n_match = timing.get("large_n_match", False)
    junit_passed = timing.get("junit_passed", 0)
    junit_total = timing.get("junit_total", 0)
    junit_failed = timing.get("junit_failed", 0)
    # Full large-N correctness = the sieve proven to SCALE (1e6/1e7 manifest
    # match OR a full JUnit manifest sweep with zero failures). Small-N hash
    # alone only proves the code is locally right, NOT that it scales — so it
    # caps below 100 (catching hardcoded/naive small-N tricks).
    scales = large_n_match or (junit_total > 0 and junit_failed == 0 and junit_passed > 0)
    has_java = os.path.exists(os.path.join(run_dir, "hashprime.java"))

    if scales:
        # Proven to scale: full correctness.
        score["correctness"] = 100
    elif hash_match:
        # Correct small-N output but not proven to scale: partial.
        score["correctness"] = 75
    elif compile_ok:
        score["correctness"] = 50
    elif has_java:
        score["correctness"] = 25
    else:
        score["correctness"] = 0

    score["details"]["compile_ok"] = compile_ok
    score["details"]["hash_match"] = hash_match
    score["details"]["has_java_source"] = has_java
    score["details"]["large_n_match"] = timing.get("large_n_match", False)
    score["details"]["large_n_count_match"] = timing.get("large_n_count_match", False)
    score["details"]["large_n_exec_time"] = timing.get("large_n_exec_time", 0)
    score["details"]["large_n_hash"] = timing.get("large_n_hash", "")
    score["details"]["large_n_error"] = timing.get("large_n_error", "")
    score["details"]["large_n2_match"] = timing.get("large_n2_match", False)
    score["details"]["large_n2_count_match"] = timing.get("large_n2_count_match", False)
    score["details"]["large_n2_exec_time"] = timing.get("large_n2_exec_time", 0)
    score["details"]["large_n2_hash"] = timing.get("large_n2_hash", "")
    score["details"]["large_n2_error"] = timing.get("large_n2_error", "")
    score["details"]["junit_passed"] = timing.get("junit_passed", 0)
    score["details"]["junit_total"] = timing.get("junit_total", 0)
    score["details"]["junit_failed"] = timing.get("junit_failed", 0)

    # Speed (model wall-clock only; lint time is not separately tracked)
    total_time = timing.get("total_seconds", 999)
    model_time = total_time
    score["details"]["total_time_seconds"] = round(total_time, 2)
    score["details"]["model_time_seconds"] = round(model_time, 2)

    # Token throughput (generated tokens per second of model wall-clock)
    _pt_total = timing.get("prompt_tokens", 0)
    _ct_total = timing.get("completion_tokens", 0)
    _gen_tok_s = round(_ct_total / model_time, 2) if model_time > 0 else 0.0
    _prompt_tok_s = round(_pt_total / model_time, 2) if model_time > 0 else 0.0
    score["details"]["prompt_tokens_total"] = _pt_total
    score["details"]["completion_tokens_total"] = _ct_total
    score["details"]["gen_tok_s"] = _gen_tok_s
    score["details"]["prompt_tok_s"] = _prompt_tok_s
    score["details"]["token_log"] = timing.get("token_log", [])

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
        expected_tools = {"write_file", "compile_java", "run", "submit_answer"}
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
    model_secs = total_secs
    human_total = format_time(total_secs)
    human_model = format_time(model_secs)
    # Lint is user-only and not separately timed; keep the line honest.
    human_lint = format_time(0)
    prompt_tokens = timing.get("prompt_tokens", 0)
    completion_tokens = timing.get("completion_tokens", 0)
    tools_used = ", ".join(score["details"].get("tools_used", ["(none)"]))
    javac_warnings = score["details"].get("javac_warnings", 0)
    checkstyle_count = score["details"].get("checkstyle_count", 0)
    pmd_count = score["details"].get("pmd_count", 0)
    mode = score["details"].get("mode", "?")
    jq_passed = score["details"].get("jq_audit_passed", True)
    jq_str = "PASS jq" if jq_passed else "FAIL jq"
    sem_passed = score["details"].get("semantic_passed", True)
    sem_score = score["details"].get("semantic_score", 0.0)
    sem_str = f"PASS sem ({sem_score:.2f})" if sem_passed else f"FAIL sem ({sem_score:.2f})"

    lint_str = f"{javac_warnings} javac + {checkstyle_count} checkstyle + {pmd_count} PMD"
    if javac_warnings == 0 and checkstyle_count == 0 and pmd_count == 0:
        lint_str = "0 violations (clean)"

    gen_tok_s = score["details"].get("gen_tok_s", 0.0)
    prompt_tok_s = score["details"].get("prompt_tok_s", 0.0)
    token_log = score["details"].get("token_log", [])
    if token_log:
        per_turn = "\n".join(
            f"    turn {t['turn']:<3} sent={t['prompt_tokens']:<6} gen={t['completion_tokens']:<5} {t['duration_s']:.2f}s"
            for t in token_log
        )
        token_block = f"\n  Per-turn:\n{per_turn}"
    else:
        token_block = ""

    summary = f"""Model: {model}
─────────────────────────────────────────────────
Result:   {'PASS' if score['correctness'] >= 100 else 'FAIL'}
Time:     {human_model} model = {human_total} wall  (score: {score['speed_score']}/100)
Correct:  {'Yes - hash matches expected' if score['correctness'] >= 100 else 'No - wrong hash'}
Compiled: {'Yes' if score['details'].get('compile_ok') else 'No'}
Large-N (1e6):  {'Yes - hash matches A000040 manifest' if score['details'].get('large_n_match') else ('ERR - check failed: ' + str(score['details'].get('large_n_error', ''))[:60] if score['details'].get('large_n_error') else 'No - 1e6 hash mismatch')} ({score['details'].get('large_n_exec_time', 0):.2f}s)
Large-N (1e7):  {'Yes - hash matches A000040 manifest' if score['details'].get('large_n2_match') else ('ERR - check failed: ' + str(score['details'].get('large_n2_error', ''))[:60] if score['details'].get('large_n2_error') else 'No - 1e7 hash mismatch')} ({score['details'].get('large_n2_exec_time', 0):.2f}s)
Quality:  {lint_str}  (score: {score['code_quality']}/100 — higher=cleaner)
Tokens:   {prompt_tokens} sent + {completion_tokens} generated = {prompt_tokens + completion_tokens} total  (gen {gen_tok_s:.1f} tok/s, prompt {prompt_tok_s:.1f} tok/s){token_block}
Tools:    [{tools_used}]  (score: {score['tool_usage']}/100 — higher=more tools used)
Mode:     {mode}  {jq_str}  {sem_str}
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

    # Capability detection (fail-fast prep). Skip ONLY embedding-only models;
    # an empty/absent capability list is treated as chat-capable (is_chat_model)
    # so transient/newer-tag models are not wrongly skipped.
    capabilities = get_model_capabilities(model)
    if not is_chat_model(model):
        print(f"\n{'='*60}")
        print(f"SKIPPING {model}: not a chat model (capabilities={sorted(capabilities) or 'embedding'})")
        print(f"{'='*60}")
        return {
            "skipped": True,
            "reason": f"not a chat model (capabilities={sorted(capabilities) or 'embedding'})",
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
    timing = {"total_seconds": 0, "javac_warnings": 0, "compile_ok": False, "hash_match": False, "checkstyle_count": 0, "pmd_count": 0, "thinking_support": False, "used_retrieval": False, "capabilities": sorted(capabilities),
              "large_n_match": False, "large_n_count_match": False, "large_n_exec_time": 0, "large_n_hash": "", "large_n_error": "", "large_n_count": None,
              "large_n2_match": False, "large_n2_count_match": False, "large_n2_exec_time": 0, "large_n2_hash": "", "large_n2_error": ""}
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
            _call_start = time.time()
            response = call_ollama(model, messages, capabilities=capabilities)
            _call_dur = time.time() - _call_start
        except OllamaCallError as e:
            print(f"  FATAL CALL ERROR: {e}")
            break
        except Exception as e:
            print(f"  FATAL UNEXPECTED ERROR: {e}")
            break

        msg = response["message"]
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        # Track tokens (per-turn log + run totals)
        _pt = response.get("prompt_tokens", 0)
        _ct = response.get("completion_tokens", 0)
        total_prompt_tokens += _pt
        total_completion_tokens += _ct
        timing.setdefault("token_log", []).append({
            "turn": turn, "prompt_tokens": _pt, "completion_tokens": _ct,
            "duration_s": round(_call_dur, 3),
        })

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
            # Persist the failed stream for the user to inspect: a consolidated,
            # clearly-labeled log of every turn whose raw output failed the jq
            # JSON-validity audit (mirrors the per-turn raw_output_turnN.txt but
            # with a header so the failure is obvious).
            with open(os.path.join(run_dir, "failed_streams.log"), "a") as _fs:
                _fs.write(f"\n=== Turn {turn} — jq audit FAILED (raw output is not valid JSON) ===\n")
                _fs.write(_safe_log(content, limit=200000) + "\n")
            # Record the failure in the persisted conversation transcript so it
            # is visible in conversation.json (not just in failed_streams.log).
            conversation.append({
                "turn": turn, "role": "jq_audit_failed", "content": content
            })
            # Send the error back to the model as a normal turn so it can
            # self-correct — same convention as the compile/hash-failure paths.
            # Tool-call-specific: tell it to emit write_file JSON, raw (no fences).
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON, so the tool call "
                    "could not be parsed. Respond with RAW JSON only (no markdown "
                    "fences, no prose) using the write_file tool, e.g. "
                    '{"name": "write_file", "arguments": {"path": "hashprime.java", '
                    '"content": "public class hashprime { ... }"}}.'
                ),
                "fed_to_model": True,
            })

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
            # jq-audit-first: validate the raw assistant text as JSON BEFORE we
            # trust any tool-call extraction. If the model emitted raw JSON it
            # should at least be well-formed; log the verdict transparently so
            # the user can audit what the model actually produced each turn.
            _audit = jq_audit(content) if content else True
            print(f"  ── TOOL REQUEST ── turn={turn} n_calls={len(tool_calls)} "
                  f"jq_audit={'PASS' if _audit else 'FAIL'}")
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})

                # ── TOOL REQUEST (sent): pretty-print the call + args ──
                print(f"  ── TOOL REQUEST ── {name}")
                try:
                    print(_pretty_json(args))
                except Exception:
                    print(_safe_log(args))

                if name in TOOL_DISPATCH:
                    try:
                        result = TOOL_DISPATCH[name](args, run_dir)
                    except Exception as e:
                        result = json.dumps({"ok": False, "error": str(e)})
                else:
                    # Model-actionable guidance: tell the model the exact valid
                    # tool names so it can self-correct instead of stalling.
                    valid = ", ".join(sorted(TOOL_DISPATCH.keys()))
                    # Log the bad call safely (truncated/sanitized) so it can't
                    # kill or corrupt the log stream.
                    print(f"  ── BAD TOOL CALL (safe-logged) ── name={_safe_log(name)} "
                          f"args={_safe_log(args)}")
                    result = json.dumps({
                        "ok": False,
                        "error": f"Unknown tool: '{name}'. Valid tools are: {valid}.",
                    })

                # Model-actionable guidance for missing/invalid arguments:
                # surface the required schema fields so the model can retry.
                # Guard the result parse so a malformed (non-JSON) result is
                # logged safely instead of crashing the harness.
                try:
                    _rj = json.loads(result)
                    _malformed = False
                except Exception:
                    _rj = None
                    _malformed = True
                if _malformed:
                    print(f"  ── MALFORMED TOOL RESULT (safe-logged) ── {_safe_log(result)}")
                elif name in TOOL_DISPATCH and _rj.get("ok") is False and "error" in _rj:
                    _req = _required_args_for(name)
                    if _req:
                        _rj["hint"] = (
                            f"Required arguments for '{name}': "
                            + ", ".join(_req)
                        )
                        result = json.dumps(_rj)

                fed_to_model = True
                if name == "compile_java":
                    try:
                        data = json.loads(result)
                        timing["compile_ok"] = data.get("ok", False)
                        stderr = data.get("stderr", "")
                        wc = len([l for l in stderr.split("\n") if "warning" in l.lower()])
                        javac_warnings = max(javac_warnings, wc)
                    except: pass

                if name == "run":
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

                if name == "run_junit_verification":
                    try:
                        data = json.loads(result)
                        timing["junit_passed"] = data.get("passed", 0)
                        timing["junit_total"] = data.get("total", 0)
                        timing["junit_failed"] = data.get("failed", 0)
                        print(f"  JUnit: passed={timing['junit_passed']} "
                              f"total={timing['junit_total']} "
                              f"failed={timing['junit_failed']}")
                        if data.get("error"):
                            print(f"  JUnit error: {_safe_log(data['error'])}")
                    except: pass

                if name == "write_file":
                    # Pretty-print the file the model wrote (full source).
                    _p = args.get("path", "?")
                    _c = args.get("content", "")
                    print(f"  ── SOURCE WRITTEN ── {_p} ({len(_c)} bytes)")
                    print(_pretty_json({"path": _p, "content": _c}))

                if name == "read_file":
                    # Pretty-print the read request + returned content (full).
                    _p = args.get("path", "?")
                    print(f"  ── FILE READ ── {_p}")
                    try:
                        _rd = json.loads(result)
                        if _rd.get("ok"):
                            print(_pretty_json({"path": _rd.get("path"),
                                                "content": _rd.get("content", ""),
                                                "size": _rd.get("size")}))
                        else:
                            print(f"  read_file error: {_safe_log(_rd.get('error'))}")
                    except Exception:
                        print(f"  ── MALFORMED read_file RESULT (safe-logged) ── {_safe_log(result)}")

                # ── FED TO MODEL (received): pretty-print the result ──
                print(f"  ── FED TO MODEL ── {name}")
                try:
                    print(_pretty_json(result))
                except Exception:
                    print(_safe_log(result))
                conversation.append({
                    "turn": turn, "role": "tool",
                    "name": name, "content": result,
                    "fed_to_model": fed_to_model,
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
                timing["exec_time"] = timing.get("exec_time", 0) + tcv["exec_time"]
                _copy_large_n(timing, tcv)
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
                        "content": f"The code compiled successfully but the SHA-256 hash it printed to stdout doesn't match {EXPECTED_HASH}. Print ONLY the lowercase-hex SHA-256 of the prime bytes (each prime on its own line followed by '\\n') to stdout — no other output, no file. The digest for N=11 (primes 2,3,5,7,11) must be {EXPECTED_HASH}."
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
                timing["exec_time"] = timing.get("exec_time", 0) + tcv["exec_time"]
                _copy_large_n(timing, tcv)
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

    # Post-loop large-N recompute. The per-turn loop may hit PER_MODEL_TIMEOUT
    # (or MAX_TURNS) before the in-loop try_compile_and_verify finishes the
    # 1E6/1E7 sieves, leaving large_n_* empty/False even though the compiled
    # class is correct. If the class compiled but large-N was not verified
    # (empty hash), re-run try_compile_and_verify now (on its own budget,
    # untimed) so a correct-but-slow model is not recorded as a false negative.
    # NOTE: fires for BOTH tool_call and text modes — do NOT gate on
    # did_compile_verify, which is only set in text-mode paths; tool-call
    # models (e.g. satgeze) would otherwise never get their false negatives
    # corrected.
    if (timing.get("compile_ok")
            and os.path.exists(os.path.join(run_dir, "hashprime.java"))
            and not timing.get("large_n_match")
            and not timing.get("large_n2_match")):
        try:
            print("  Re-running large-N verification (post-loop, untimed)...")
            tcv = try_compile_and_verify(run_dir)
            _copy_large_n(timing, tcv)
            # Keep the better of the two hash_match observations.
            timing["hash_match"] = timing.get("hash_match") or tcv.get("hash_match", False)
        except Exception as e:
            print(f"  Post-loop large-N verify failed: {e}")

    score = score_run(run_dir, conversation, timing, mode)

    # Cleanup this model after testing — NOT timed
    print("  Cleaning up model...")
    cleanup_all_models()

    save_results(run_dir, model, conversation, timing, score)

    return score, run_dir


def score_to_row(model, run_dir, score):
    """Convert a finished score dict into a ranking summary row."""
    sem_passed = score["details"].get("semantic_passed", True)
    sem_score = score["details"].get("semantic_score", 0.0)
    # sem_passed is None when scoring was UNAVAILABLE (Qdrant/collection missing)
    # — render as N/A so it is never mistaken for a genuine off-topic FAIL.
    sem_tag = "N/A" if sem_passed is None else ("PASS" if sem_passed else "FAIL")
    tools_used_list = score["details"].get("tools_used", [])
    has_write_file = "write_file" in tools_used_list
    jq_passed = score["details"].get("jq_audit_passed", True)
    tool_ok = "PASS" if (score["details"].get("mode") == "tool_call" or (has_write_file and jq_passed)) else "FAIL"
    compile_ok = score["details"].get("compile_ok", False)
    hash_match = score["details"].get("hash_match", False)
    thinking_support = score["details"].get("thinking_support", False)
    think_tag = "PASS" if thinking_support else "FAIL"
    used_retrieval = score["details"].get("used_retrieval", False)
    retr_tag = "PASS" if used_retrieval else "FAIL"
    # Thinking-quality scoring: N/A when the embedding/scoring infra was down,
    # so a 0 is not mistaken for "reasoning was poor".
    tq_err = score["details"].get("thinking_quality_error")
    tq_unavailable = (tq_err == "UNAVAILABLE")
    large_n_match = score["details"].get("large_n_match", False)
    large_n_err = score["details"].get("large_n_error", "")
    large_n_tag = "PASS" if large_n_match else ("ERR" if large_n_err else "FAIL")
    large_n2_match = score["details"].get("large_n2_match", False)
    large_n2_err = score["details"].get("large_n2_error", "")
    large_n2_tag = "PASS" if large_n2_match else ("ERR" if large_n2_err else "FAIL")
    total_violations = (
        score["details"].get("javac_warnings", 0) +
        score["details"].get("checkstyle_count", 0) +
        score["details"].get("pmd_count", 0)
    )
    turns_used = score["details"].get("turns_used", 0)
    exec_secs = score["details"].get("exec_time", 0)
    model_time_secs = score["details"].get("model_time_seconds", 0)
    model_name_short = model[:25] if len(model) > 25 else model

    return {
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
        "gen_tok_s": score["details"].get("gen_tok_s", 0.0),
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
        "tq_unavailable": tq_unavailable,
        "retr_tag": retr_tag,
        "large_n_tag": large_n_tag,
        "large_n2_tag": large_n2_tag,
        "used_retrieval": used_retrieval,
        "jq_passed": jq_passed,
        "has_write_file": has_write_file,
        "run_dir": run_dir,
    }


def regenerate_ranking():
    """Rebuild ranking.md / ranking.json from all existing results/*/*/score.json
    on disk. Lets a re-verification pass (which patches score.json in place)
    refresh the tables without re-running the whole benchmark."""
    import glob as _glob

    results_summary = []
    for sp in _glob.glob(os.path.join(RESULTS_BASE, "*", "*", "score.json")):
        try:
            with open(sp) as f:
                score = json.load(f)
        except Exception:
            continue
        run_dir = os.path.dirname(sp)
        # Recover the model name from the run_dir layout: results/<model>/<ts>/
        model = os.path.basename(os.path.dirname(run_dir))
        results_summary.append(score_to_row(model, run_dir, score))

    if not results_summary:
        print("No score.json files found to rank.")
        return results_summary

    results_summary.sort(key=lambda x: x["score"], reverse=True)
    _write_ranking(results_summary)
    return results_summary


def _write_ranking(results_summary):
    """Render the console + markdown ranking tables (shared by main() and
    regenerate_ranking())."""
    # ── Single column-spec drives BOTH console + markdown tables (alignment-safe) ──
    COLS = [
        ("rank",    "RANK",  5,   "RANK"),
        ("model",   "MODEL", 25,  "MODEL"),
        ("score",   "TOTAL", 5,   "TOTAL"),
        ("correct", "CORR",  5,   "CORR"),
        ("c_tag",   "CPILE", 5,   "CPILE"),
        ("h_tag",   "HASH",  5,   "HASH"),
        ("viol",    "VIOL",  4,   "VIOL"),
        ("sem",     "SEM",   4,   "SEM"),
        ("tool",    "TOOL",  4,   "TOOL"),
        ("think",   "THINK", 5,   "THINK"),
        ("retr",    "RETR",  5,   "RETR"),
        ("ts",      "TSCORE",6,   "TSCORE"),
        ("drift",   "DRIFT", 6,   "DRIFT"),
        ("iter",    "ITER",  4,   "ITER"),
        ("exec",    "EXEC",  9,   "EXEC"),
        ("time",    "MTIME", 10,  "MTIME"),
        ("tok_s",   "TOK/s", 7,   "TOK/s"),
        ("large",   "1E6",   5,   "1E6"),
        ("large2",  "1E7",   5,   "1E7"),
        ("mode",    "MODE",  8,   "MODE"),
    ]

    def row_values(i, r):
        c_tag = "PASS" if r["compile_ok"] else "FAIL"
        h_tag = "PASS" if r["hash_match"] else "FAIL"
        drift = r["thinking_drift"]
        drift_s = f"{drift:+.2f}" if isinstance(drift, (int, float)) else "-"
        # When thinking-quality scoring was UNAVAILABLE, render TSCORE/DRIFT as N/A
        # instead of a misleading 0 / "-", so infra outage is never seen as a model flaw.
        ts_val = r["think_score"]
        if r.get("tq_unavailable"):
            ts_val = "N/A"
            drift_s = "N/A"
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
            "tok_s": f"{r['gen_tok_s']:.1f}",
            "large": r["large_n_tag"],
            "large2": r["large_n2_tag"],
            "mode": r["mode"],
        }

    # Compact per-column legend shown right under the table (console + markdown).
    COL_LEGEND = {
        "rank":   "rank / position",
        "model":  "model name",
        "score":  "composite score (0-100)",
        "correct":"correctness (0=none, 25=has Java, 50=compiles, 75=small-N hash match but not proven to scale, 100=hash match AND proven to scale)",
        "c_tag":  "compiled? (PASS/FAIL)",
        "h_tag":  "SHA-256 hash matches expected? (PASS/FAIL)",
        "viol":   "lint violations (javac+checkstyle+PMD; lower better)",
        "sem":    "semantic gate (PASS=on-topic, FAIL=off-topic, N/A=scoring unavailable / Qdrant down — NOT a model failure)",
        "tool":   "tool-call support? (PASS/FAIL)",
        "think":  "thinking/reasoning trace present? (PASS/FAIL)",
        "retr":   "used search_solutions retrieval? (PASS/FAIL)",
        "ts":     "thinking quality (0-100, vs hashprime_solutions; N/A=scoring unavailable)",
        "drift":  "TSCORE − prompt baseline (positive=more on-topic; N/A=scoring unavailable)",
        "iter":   "conversation turns used",
        "exec":   "compiled Java execution time",
        "time":   "model wall-clock time (excl. lint)",
        "tok_s":  "generation throughput (tokens/sec)",
        "large":  "1E6 sieve hash == manifest? (PASS/FAIL/ERR)",
        "large2": "1E7 sieve hash == manifest? (PASS/FAIL/ERR)",
        "mode":   "tool_call = native API, text = JSON-in-plaintext",
    }

    # ── Console ranking table ──
    print(f"\n{'='*150}")
    console_header = "  " + "  ".join(f"{h:<{w}}" for _, h, w, _ in COLS)
    print(console_header)
    print(f"  {'─'*len(console_header)}")
    print(f"  (higher=better: TOTAL/CORR/TSCORE; lower=better: VIOL; PASS/FAIL=pass/fail; ITER=turns; EXEC=code run; MTIME=model wall-clock)")
    print(f"{'─'*150}")
    for i, r in enumerate(results_summary, 1):
        v = row_values(i, r)
        print("  " + "  ".join(f"{str(v[k]):<{w}}" for k, _, w, _ in COLS))

    # Column legend directly under the console table.
    print(f"\n  Column legend:")
    for key, _, _, h in COLS:
        print(f"    {h:<7} = {COL_LEGEND.get(key, '')}")

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
        "### Column legend",
    ])
    for key, _, _, h in COLS:
        md_lines.append(f"- **{h}**: {COL_LEGEND.get(key, '')}")

    md_lines.extend([
        "",
        "### Key",
        "- **TOTAL**: composite score (0–100)",
        "- **CORR**: correctness (0=none, 25=has Java, 50=compiles, 75=small-N hash match but not proven to scale, 100=hash match AND proven to scale via 1E6/1E7 manifest or full JUnit sweep)",
        "- **CPILE**: code compiled? (PASS/FAIL)",
        "- **HASH**: SHA-256 hash matches expected? (PASS/FAIL)",
        "- **VIOL**: total lint violations (javac warnings + checkstyle + PMD; lower is better)",
        "- **SEM**: semantic gate pass? (PASS = output is on-topic for the hashprime problem)",
        "- **TOOL**: tool call support? (PASS = native API or text-mode with valid JSON + file write)",
        "- **THINK**: thinking/reasoning trace present? (PASS = Ollama returned a non-empty `thinking` field when `think:true` requested)",
        "- **RETR**: did the model use the `search_solutions` retrieval tool? (PASS = yes)",
        "- **TSCORE**: semantic quality of the thinking trace (0–100). Chonkie chunks the trace, embeds via Ollama nomic-embed-text, mean cosine similarity vs hashprime_solutions in Qdrant. Higher = reasoning more aligned with known-good solutions.",
        "- **DRIFT**: TSCORE − PROMPT_TS (prompt baseline alignment). Positive = model reasoning is more on-topic than the prompt itself.",
        "- **ITER**: number of conversation turns the model took",
        "- **EXEC**: time for compiled Java code to execute (seconds)",
        "- **MTIME**: total model wall-clock time excluding lint (seconds or minutes)",
        "- **TOK/s**: generation throughput — completion tokens per second of model wall-clock time (higher = faster generation)",
        "- **1E6**: does the sieve scale? The SHA-256 hash the program prints to stdout for N=1,000,000 matches the authoritative OEIS A000040 manifest (ascii_integer_lf). PASS = matches, FAIL = hash mismatch (wrong sieve), ERR = the 1e6 check itself errored/timed out (indeterminate — not a model fail). At 1e6 the algorithm dominates runtime, so this neutralizes any 'don't write a file' micro-optimization advantage.",
        "- **1E7**: does the faster-than-naive sieve scale to N=10,000,000? Same manifest hash/check (664579 primes). PASS = matches, FAIL = hash mismatch, ERR = check errored/timed out. Confirms the optimized algorithm (segmented / bit-packed / odds-only) produces identical output at scale and runs faster than the naive boolean[] sieve.",
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

    best_semantic = [r for r in results_summary if r["sem_tag"] == "PASS"]
    if best_semantic:
        md_lines.append("### Semantic Gate Passed\n")
        for r in best_semantic:
            md_lines.append(f"  - **{r['model']}** — score {r['score']} | sem {r['sem_score']:.2f}")
        md_lines.append("")

    best_tool = [r for r in results_summary if r["tool_ok"] == "PASS"]
    if best_tool:
        md_lines.append("### Tool Call Support\n")
        for r in best_tool:
            md_lines.append(f"  - **{r['model']}** — mode {r['mode']} | TOTAL {r['score']}")
        md_lines.append("")

    no_tool = [r for r in results_summary if r["tool_ok"] == "FAIL"]
    if no_tool:
        md_lines.append("### No Tool Call Support\n")
        for r in no_tool:
            md_lines.append(f"  - **{r['model']}** — mode {r['mode']} | jq {'PASS' if r['jq_passed'] else 'FAIL'} | write_file {'PASS' if r['has_write_file'] else 'FAIL'}")
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


def _dedup_rows(rows):
    """Collapse duplicate rows (same model) keeping the highest-scoring entry.
    Models can appear twice if a re-run created a second timestamped dir; the
    ranking should show each model once, at its best."""
    best = {}
    for r in rows:
        k = r.get("model") or r.get("model_short")
        if k not in best or r.get("score", 0) > best[k].get("score", 0):
            best[k] = r
    return list(best.values())


def clean_results(regen=True, purge=False, dry_run=False):
    """Tidy the results/ tree.

    - dedup: keep only the best score per model in ranking.md/ranking.json
      (rebuilt from on-disk score.json, dropping duplicate low runs).
    - purge: also DELETE the losing run directories from disk (the per-model
      timestamped subdirs not represented in the deduped ranking).
    - regen: rebuild ranking tables from the surviving score.json files.
    - dry_run: report what WOULD change, delete nothing, write nothing.
    """
    import glob as _glob
    import shutil as _shutil

    rows = []
    all_scores = {}  # run_dir -> (model, score)
    for sp in _glob.glob(os.path.join(RESULTS_BASE, "*", "*", "score.json")):
        try:
            with open(sp) as f:
                score = json.load(f)
        except Exception:
            continue
        run_dir = os.path.dirname(sp)
        model = os.path.basename(os.path.dirname(run_dir))
        all_scores[run_dir] = (model, score)
        rows.append(score_to_row(model, run_dir, score))

    deduped = _dedup_rows(rows)
    keep_dirs = set()
    for r in deduped:
        rd = r.get("run_dir")
        if rd:
            keep_dirs.add(os.path.abspath(rd))

    losers = [d for d in all_scores if os.path.abspath(d) not in keep_dirs]

    print(f"[clean] {len(all_scores)} total runs, {len(deduped)} models after dedup, "
          f"{len(losers)} run dir(s) would be removed.")
    for d in sorted(losers):
        print(f"  REMOVE {d}")

    if dry_run:
        print("[clean] dry-run: no changes made.")
        return deduped

    if purge:
        for d in losers:
            try:
                _shutil.rmtree(d)
                print(f"  deleted {d}")
            except Exception as e:
                print(f"  FAILED to delete {d}: {e}")

    if regen:
        deduped.sort(key=lambda x: x["score"], reverse=True)
        _write_ranking(deduped)
        print(f"[clean] ranking regenerated for {len(deduped)} model(s).")
    return deduped


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

    # Filter out non-chat models (embedding-only) up front; report them.
    # We skip ONLY models that explicitly report embedding capability; an empty
    # capability list is treated as chat-capable (see is_chat_model).
    chat_models = []
    skipped_models = []
    for m in models:
        if is_chat_model(m):
            chat_models.append(m)
        else:
            caps = get_model_capabilities(m)
            skipped_models.append((m, sorted(caps) or "embedding"))
    if skipped_models:
        print(f"\nSkipping {len(skipped_models)} non-chat model(s):")
        for m, caps in skipped_models:
            print(f"  - {m} (capabilities: {caps})")

    # Probe Qdrant semantic-scoring availability up front so a missing
    # collection (data loss) is reported clearly rather than silently turning
    # every model's SEM/TSCORE/DRIFT into a misleading FAIL/0.
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(path=QDRANT_PATH)
        qc.get_collection("hashprime_solutions")
        print("Semantic scoring: Qdrant collection 'hashprime_solutions' available.")
    except Exception as e:
        print("=" * 70)
        print("⚠  SEMANTIC / THINKING SCORING UNAVAILABLE")
        print("   Qdrant collection 'hashprime_solutions' is missing or Qdrant")
        print("   is unreachable. SEM, TSCORE and DRIFT will show N/A (not a")
        print(f"   model failure). Cause: {e}")
        print("   To restore: python ingest_corpora.py")
        print("=" * 70)

    results_summary = []

    for model in chat_models:
        try:
            result, run_dir = run_model_benchmark(model)
        except Exception as e:
            # A single model's failure (e.g. transient Ollama connection drop)
            # must not abort the whole benchmark. Record it and continue.
            print(f"  FATAL: model {model} crashed the run: {e}")
            results_summary.append({
                "model": model,
                "model_short": model[:25],
                "score": 0, "correct": 0, "compile_ok": False, "hash_match": False,
                "speed": 0, "quality": 0, "tools": 0, "violations": 0,
                "turns_used": 0, "exec_time_secs": 0, "exec_str": "-",
                "model_time_secs": 0, "time_str": "-", "gen_tok_s": 0.0,
                "tools_used": "", "mode": "?", "javac_w": 0, "cs": 0, "pmd": 0,
                "sem_tag": "FAIL", "sem_score": 0.0, "tool_ok": "FAIL", "think_tag": "FAIL",
                "thinking_support": False, "think_score": 0, "thinking_quality_chunks": 0,
                "thinking_quality_mean_sim": 0.0, "thinking_quality_error": str(e)[:200],
                "prompt_ts": 0.0, "thinking_drift": None, "retr_tag": "FAIL",
                "large_n_tag": "FAIL", "large_n2_tag": "FAIL", "used_retrieval": False,
                "jq_passed": False, "has_write_file": False, "run_dir": "",
            })
            continue
        if isinstance(result, dict) and result.get("skipped"):
            print(f"  (skipped {model}: {result.get('reason')})")
            continue
        score = result
        results_summary.append(score_to_row(model, run_dir, score))

        # Flush stdout so tool_benchmark.log keeps pace with Ollama as we move
        # from one model to the next (the pipe to `tee` is fully buffered
        # otherwise and lags far behind the live terminal).
        sys.stdout.flush()

    results_summary.sort(key=lambda x: x["score"], reverse=True)
    _write_ranking(results_summary)

    # Completion sentinel — lets the supervisor know all models finished.
    try:
        with open(os.path.join(RESULTS_BASE, "benchmark_ALLDONE.sentinel"), "w") as f:
            f.write(f"done {datetime.now().isoformat()}\n")
    except Exception:
        pass


if __name__ == "__main__":
    import sys as _sys
    # Line-buffer stdout so every print flushes immediately through the `tee`
    # pipe — keeps tool_benchmark.log in step with the live Ollama run instead
    # of lagging many models behind. (Python >=3.7.)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = set(_sys.argv[1:])
    if "--clean" in args:
        # Deduplicate ranking + delete losing run dirs, then regen tables.
        clean_results(regen=True, purge=True, dry_run=False)
    elif "--clean-regen" in args:
        # Deduplicate + regen tables, but keep all run dirs on disk.
        clean_results(regen=True, purge=False, dry_run=False)
    elif "--clean-purge" in args:
        # Delete losing run dirs WITHOUT rewriting the ranking tables.
        clean_results(regen=False, purge=True, dry_run=False)
    elif "--dry-run" in args:
        # Show what --clean would remove, change nothing.
        clean_results(regen=True, purge=True, dry_run=True)
    elif "--regen" in args:
        regenerate_ranking()
    else:
        main()
