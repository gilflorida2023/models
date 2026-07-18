#!/usr/bin/env python3
"""
Ingest three corpora into Qdrant for semantic gating of benchmark results.
Corpora:
  1. hello-algo          — DSA textbook (multilingual: zh, en, ja, etc.)
  2. Linux-Server        — Linux sysadmin guide (English + Hindi)
  3. hashprime_solutions — previously submitted hashprime.java solutions

Uses Qdrant in local persistent mode (no server binary needed).
Uses Ollama's nomic-embed-text for fast, local embeddings.
Uses Ollama nomic-embed-text via /api/embed (local, no external API).
"""

import os, hashlib, re, subprocess, json
from pathlib import Path

# Heavy / optional deps are imported lazily so ingestion still runs (and the
# nightly scheduled run doesn't die) on a host missing requests/tqdm/chonkie/
# qdrant_client. Each is imported where it is actually first used.
def _tqdm(x, *a, **k):
    try:
        from tqdm import tqdm
        return tqdm(x, *a, **k)
    except ImportError:
        return x

QDRANT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_data")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

REPOS = {
    "hello_algo": {
        "url": "https://github.com/krahets/hello-algo.git",
        "local": os.path.join(BASE_DIR, "corpora", "hello-algo"),
        "collection": "hello_algo",
    },
    "linux_server": {
        "url": "https://github.com/nikhilpatidar01/Linux-Server.git",
        "local": os.path.join(BASE_DIR, "corpora", "linux-server"),
        "collection": "linux_server",
    },
}

# Skip hello-algo by default (801 files, very slow to embed)
INCLUDE_HELLO_ALGO = os.environ.get("INCLUDE_HELLO_ALGO", "").lower() in ("1", "true", "yes")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 180
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
SIMILARITY_THRESHOLD = 0.60  # hard gate threshold (matches tool_benchmark.py)


def get_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class EmbeddingEngine:
    """Singleton-like wrapper for the embedding model + Qdrant client."""

    def __init__(self, qdrant_path: str = QDRANT_PATH):
        os.makedirs(qdrant_path, exist_ok=True)
        from qdrant_client import QdrantClient
        try:
            self.qdrant = QdrantClient(path=qdrant_path)
        except Exception as e:
            if "already accessed by another instance" in str(e):
                raise RuntimeError(
                    "Qdrant store is locked by another process. Stop it first "
                    "(e.g. `pkill -f ingest_corpora.py` and/or `pkill -f tool_benchmark.py`) "
                    "then retry."
                ) from e
            raise
        self.ollama_url = "http://localhost:11434/api/embed"
        self.dimension = 768  # nomic-embed-text
        self._init_collections()

    def _init_collections(self):
        for repo in REPOS.values():
            coll = repo["collection"]
            self._ensure_collection(coll)
        self._ensure_collection("hashprime_solutions")

    def _ensure_collection(self, name: str):
        if not self.qdrant.collection_exists(name):
            from qdrant_client.http.models import VectorParams, Distance
            self.qdrant.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.dimension, distance=Distance.COSINE
                ),
            )

    def embed(self, texts):
        import requests
        resp = requests.post(self.ollama_url, json={
            "model": "nomic-embed-text",
            "input": texts,
        }, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        import numpy as np
        return [np.array(e, dtype=np.float32) for e in data.get("embeddings", [])]

    def upsert(self, collection: str, points: list):
        if points:
            self.qdrant.upsert(
                collection_name=collection, points=points, wait=True
            )

    def search(self, collection: str, vector, limit: int = 3):
        result = self.qdrant.query_points(
            collection_name=collection,
            query=vector.tolist(),
            limit=limit,
        )
        return result.points

    def search_all(self, vector, limit: int = 3):
        """Search all corpora and return flat results sorted by score."""
        all_points = []
        for repo in REPOS.values():
            coll = repo["collection"]
            try:
                self.qdrant.get_collection(coll)
            except Exception:
                continue
            result = self.qdrant.query_points(
                collection_name=coll,
                query=vector.tolist(),
                limit=limit,
            )
            for p in result.points:
                setattr(p, "_collection", coll)
                all_points.append(p)
        # Also search hashprime solutions
        try:
            self.qdrant.get_collection("hashprime_solutions")
            result = self.qdrant.query_points(
                collection_name="hashprime_solutions",
                query=vector.tolist(),
                limit=limit,
            )
            for p in result.points:
                setattr(p, "_collection", "hashprime_solutions")
                all_points.append(p)
        except Exception:
            pass
        all_points.sort(key=lambda p: p.score, reverse=True)
        return all_points


def clone_or_update(url: str, local_path: str):
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        subprocess.run(
            ["git", "clone", url, local_path],
            capture_output=True, text=True, check=True,
        )
    else:
        subprocess.run(
            ["git", "-C", local_path, "pull"],
            capture_output=True, text=True,
        )


def get_chunker():
    from chonkie import TokenChunker
    return TokenChunker(tokenizer="gpt2", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def ingest_repo(engine: EmbeddingEngine, repo_cfg: dict):
    local = repo_cfg["local"]
    coll = repo_cfg["collection"]

    print(f"\n{'='*60}")
    print(f"  Ingesting: {repo_cfg['url']}")
    print(f"  Local:     {local}")
    print(f"  Qdrant:    {coll}")
    print(f"{'='*60}")

    chunker = get_chunker()
    md_files = list(Path(local).rglob("*.md")) + list(Path(local).rglob("*.mdx"))

    BATCH_SIZE = 32  # small batches to avoid OOM
    text_buffer = []
    meta_buffer = []
    points = []

    def flush():
        nonlocal text_buffer, meta_buffer, points
        if not text_buffer:
            return
        from qdrant_client.http.models import PointStruct
        embeddings = list(engine.embed(text_buffer))
        for (chunk_text, meta), embedding in zip(zip(text_buffer, meta_buffer), embeddings):
            points.append(PointStruct(
                id=get_md5(f"{meta['source']}_{meta['chunk_index']}"),
                vector=embedding.tolist(),
                payload={"text": chunk_text, **meta},
            ))
        text_buffer = []
        meta_buffer = []
        if len(points) >= 512:
            engine.upsert(coll, points)
            points = []

    for file_path in _tqdm(md_files, desc=f"  Ingesting {coll}"):
        try:
            relative = str(file_path.relative_to(Path(local)))
            content = file_path.read_text(encoding="utf-8", errors="replace")
            if len(content.strip()) < 100:
                continue
            chunks = chunker.chunk(content)
            for i, chunk in enumerate(chunks):
                text_buffer.append(chunk.text)
                meta_buffer.append({
                    "source": relative,
                    "file_name": file_path.name,
                    "chunk_index": i,
                })
                if len(text_buffer) >= BATCH_SIZE:
                    flush()
        except Exception as e:
            print(f"  Warning: {file_path}: {e}")

    flush()
    if points:
        engine.upsert(coll, points)

    coll_info = engine.qdrant.get_collection(coll)
    total_in_coll = coll_info.points_count
    print(f"  Uploaded {total_in_coll} chunks to '{coll}'")
    return total_in_coll


def ingest_hashprime_solutions(engine: EmbeddingEngine):
    coll = "hashprime_solutions"
    print(f"\n{'='*60}")
    print(f"  Ingesting: hashprime solutions from {RESULTS_DIR}")
    print(f"  Qdrant:    {coll}")
    print(f"{'='*60}")

    if not os.path.exists(RESULTS_DIR):
        print("  No results directory found.")
        return 0

    # Collect every hashprime.java together with a "verified" flag derived from
    # its run's score.json (large_n_match / large_n2_match / hash_match). Verified
    # solutions are processed FIRST so that when two runs share identical source
    # (dedup by md5), the verified copy wins the slot and gets tagged verified.
    candidates = []
    for root, dirs, files in os.walk(RESULTS_DIR):
        if "hashprime.java" not in files:
            continue
        path = os.path.join(root, "hashprime.java")
        run_dir = root
        verified = False
        score_path = os.path.join(run_dir, "score.json")
        if os.path.exists(score_path):
            try:
                sc = json.load(open(score_path))
                d = sc.get("details", {})
                verified = bool(
                    d.get("large_n_match") or d.get("large_n2_match") or d.get("hash_match")
                )
            except Exception:
                verified = False
        candidates.append((verified, path))
    # Verified first; stable order within each group.
    candidates.sort(key=lambda x: (not x[0], x[1]))

    chunker = get_chunker()
    seen = set()
    text_buffer = []
    meta_buffer = []
    points = []
    BATCH_SIZE = 32

    def flush():
        nonlocal text_buffer, meta_buffer, points
        if not text_buffer:
            return
        from qdrant_client.http.models import PointStruct
        embeddings = list(engine.embed(text_buffer))
        for (chunk_text, meta), embedding in zip(zip(text_buffer, meta_buffer), embeddings):
            points.append(PointStruct(
                id=get_md5(f"solution_{meta['solution_hash']}_{meta['chunk_index']}"),
                vector=embedding.tolist(),
                payload={"text": chunk_text, **meta},
            ))
        text_buffer = []
        meta_buffer = []
        if len(points) >= 512:
            engine.upsert(coll, points)
            points = []

    for verified, path in candidates:
        try:
            code = open(path, encoding="utf-8").read()
            if len(code.strip()) < 50:
                continue
            dedup = hashlib.md5(code.encode()).hexdigest()
            if dedup in seen:
                continue
            seen.add(dedup)
            chunks = chunker.chunk(code)
            for i, chunk in enumerate(chunks):
                text_buffer.append(chunk.text)
                meta_buffer.append({
                    "source": "hashprime.java",
                    "file_name": "hashprime.java",
                    "chunk_index": i,
                    "solution_hash": dedup,
                    "verified": verified,
                })
                if len(text_buffer) >= BATCH_SIZE:
                    flush()
        except Exception as e:
                    print(f"  Warning: {path}: {e}")

    flush()
    if points:
        engine.upsert(coll, points)

    coll_info = engine.qdrant.get_collection(coll)
    total = coll_info.points_count
    print(f"  Uploaded {total} unique chunks to '{coll}'")
    return total


def main():
    engine = EmbeddingEngine()

    # Clone / update repos and ingest
    total = 0
    for name, cfg in REPOS.items():
        if name == "hello_algo" and not INCLUDE_HELLO_ALGO:
            print(f"\n  Skipping hello_algo (set INCLUDE_HELLO_ALGO=true to include)")
            continue
        print(f"\n  Cloning/updating {name}...")
        clone_or_update(cfg["url"], cfg["local"])
        total += ingest_repo(engine, cfg)

    total += ingest_hashprime_solutions(engine)

    count = 0
    for repo in REPOS.values():
        try:
            info = engine.qdrant.get_collection(repo["collection"])
            count += info.points_count
        except Exception:
            pass
    try:
        info = engine.qdrant.get_collection("hashprime_solutions")
        count += info.points_count
    except Exception:
        pass

    print(f"\n{'='*60}")
    print(f"  Ingestion complete!")
    print(f"  Total chunks ingested: {total}")
    print(f"  Total points in Qdrant: {count}")
    print(f"  Qdrant path: {QDRANT_PATH}")
    print(f"  Embedding model: {EMBED_MODEL}")
    print(f"  Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"{'='*60}")

    # Snapshot the store so it can be restored if the live copy is ever lost or
    # corrupted. Close the client first so the lock is released and the copy is
    # clean (no stale .lock in the backup). Never fail the ingest on backup error.
    try:
        engine.qdrant.close()
    except Exception:
        pass
    try:
        backup_path = QDRANT_PATH + ".bak"
        if os.path.exists(backup_path):
            import shutil as _shutil
            _shutil.rmtree(backup_path)
        import shutil as _shutil
        _shutil.copytree(QDRANT_PATH, backup_path)
        # Drop the lock file from the backup so a restore opens cleanly.
        lock_file = os.path.join(backup_path, ".lock")
        if os.path.exists(lock_file):
            os.remove(lock_file)
        print(f"  Backup saved to {backup_path}")
    except Exception as e:
        print(f"  Warning: backup failed: {e}")


def restore_qdrant():
    """Restore qdrant_data from the latest .bak snapshot (no lock in backup)."""
    import shutil as _shutil
    backup_path = QDRANT_PATH + ".bak"
    if not os.path.exists(backup_path):
        print(f"  No backup found at {backup_path}; nothing to restore.")
        return
    if os.path.exists(QDRANT_PATH):
        _shutil.rmtree(QDRANT_PATH)
    _shutil.copytree(backup_path, QDRANT_PATH)
    print(f"  Restored {QDRANT_PATH} from {backup_path}")


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "--restore":
        restore_qdrant()
    else:
        main()
