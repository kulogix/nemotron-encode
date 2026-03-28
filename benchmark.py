#!/usr/bin/env python3
"""
Nemotron Benchmark Tool
========================
Comprehensive benchmarking for multimodal embedding and reranking endpoints.
Works with any OpenAI-compatible embedding/reranking server.

Measures:
  - Single text latency (p50, p95, p99)
  - Batch throughput (texts/sec, embeddings/sec)
  - Batch scaling (1, 4, 8, 16, 32 texts)
  - Concurrent request throughput
  - Text length scaling (100 to 16K chars with diverse text)
  - Max input size discovery (binary search)
  - Rerank latency (per-document cost)
  - Rerank document-count scaling (1 to 32 docs)
  - Image embedding latency (if multimodal)
  - Similarity quality (semantic pair accuracy)

Uses diverse multi-domain vocabulary (IT, manufacturing, marketing, philosophy,
science) to accurately represent real tokenization costs.

Usage:
  python benchmark.py --url http://localhost:8020
  python benchmark.py --url http://localhost:8020 --all
  python benchmark.py --url http://localhost:8025 --rerank --all
  python benchmark.py --url http://localhost:8020 --max-input
  python benchmark.py --url http://localhost:8020 --corpus book_chapter.txt --all
  python benchmark.py --url http://localhost:8020 --output results.json
"""

__version__ = "1.1.0"

import argparse
import base64
import io
import json
import math
import os
import statistics
import sys
import textwrap
import time
import concurrent.futures
from pathlib import Path
from typing import Optional

import random as _random
import string as _string

try:
    import requests
except ImportError:
    print("  Missing 'requests' package. Install: pip install requests")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic Data Generators (like vllm bench random / random-mm)
# ═══════════════════════════════════════════════════════════════════════════════

# Diverse vocabulary across multiple domains — NOT just IT terms.
# Matches the diverse probe text in _probe_model_limits() to accurately
# represent real tokenization costs (repeated domain words under-count tokens).
_VOCAB = (
    # IT/Infrastructure
    "server deployment kubernetes docker container orchestration pipeline "
    "database replication failover monitoring metrics latency throughput "
    "authentication authorization security encryption certificate "
    "microservice gateway load balancer health check endpoint "
    # Manufacturing/Lean
    "Toyota Production System jidoka automation kanban scheduling kaizen "
    "continuous improvement just-in-time delivery bottleneck throughput "
    "Goldratt Theory Constraints focusing steps gemba muda muri mura "
    # Marketing/Business
    "segmentation positioning differentiation brand equity customer "
    "competitive advantage market share revenue pricing strategy "
    "Porter five forces supplier buyer substitution rivalry entrants "
    # Philosophy/Literature
    "dharma artha kama moksha Mahabharata Kurukshetra Pandavas Kauravas "
    "narrative philosophical discourse Vedic traditions ethical principles "
    "dharmasutras societal organization legal frameworks ritual procedures "
    # Science/Technical
    "algorithm optimization gradient descent convergence neural network "
    "transformer attention mechanism embedding vector dimensionality "
    "hypothesis experiment observation measurement statistical significance "
    # General
    "collaboration responsibility infrastructure communication framework "
    "implementation architecture documentation specification requirement "
    "analysis evaluation synthesis recommendation comprehensive overview "
).split()


def generate_random_text(length_chars: int, seed: int = None) -> str:
    """Generate random text of approximately `length_chars` characters using
    a realistic vocabulary. No external datasets needed."""
    rng = _random.Random(seed)
    words = []
    total = 0
    while total < length_chars:
        word = rng.choice(_VOCAB)
        words.append(word)
        total += len(word) + 1
    return " ".join(words)[:length_chars]


def generate_random_image(width: int = 64, height: int = 64, seed: int = None) -> str:
    """Generate a random RGB image as a base64 data URL. No external datasets needed."""
    rng = _random.Random(seed)
    # Create random pixel data
    pixels = bytes(rng.randint(0, 255) for _ in range(width * height * 3))
    try:
        from PIL import Image
        img = Image.frombytes("RGB", (width, height), pixels)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"
    except ImportError:
        # Fallback: return raw base64 (server may reject without PIL header)
        b64 = base64.b64encode(pixels).decode()
        return f"data:image/png;base64,{b64}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test Data
# ═══════════════════════════════════════════════════════════════════════════════

# Diverse text samples for embedding benchmarks
BENCHMARK_TEXTS = [
    "Blue-green deployment enables zero-downtime releases by maintaining two identical environments.",
    "The disaster recovery plan specifies an RTO of 4 hours and RPO of 1 hour.",
    "PostgreSQL 15 supports logical replication and improved partitioning.",
    "Kubernetes pod autoscaling adjusts replica count based on CPU utilization.",
    "The CI/CD pipeline runs unit tests, integration tests, and security scans.",
    "Load balancing distributes incoming traffic across multiple server instances.",
    "Container orchestration manages the lifecycle of containerized applications.",
    "Database sharding horizontally partitions data across multiple servers.",
    "Service mesh provides observability, traffic management, and security for microservices.",
    "Infrastructure as code defines cloud resources through declarative configuration files.",
    "The SBOM lists all software components including Python 3.11 and Django 5.0.",
    "Prometheus collects time-series metrics from instrumented applications.",
    "GraphQL provides a flexible query language for API consumers.",
    "Redis serves as both a cache layer and message broker for the application.",
    "Terraform manages cloud infrastructure through declarative HCL configuration.",
    "The monitoring dashboard tracks request latency, error rates, and throughput.",
    "OAuth 2.0 authorization flows secure API access with bearer tokens.",
    "WebSocket connections enable real-time bidirectional communication.",
    "Elasticsearch provides full-text search with inverted index data structures.",
    "gRPC uses protocol buffers for efficient binary serialization between services.",
    "The team roster includes Alice Chen (Tech Lead) and Bob Kim (Senior SRE).",
    "Nginx reverse proxy handles TLS termination and static file serving.",
    "Apache Kafka provides distributed event streaming with topic partitioning.",
    "The budget report shows $2.4M total engineering spend for Q1 2026.",
    "Docker Compose defines multi-container applications with YAML configuration.",
    "Horizontal pod autoscaler targets 70% CPU utilization across the deployment.",
    "The server inventory lists web-prod-01 at 10.0.1.10 with 32GB RAM.",
    "Ansible playbooks automate server provisioning and configuration management.",
    "Vault securely stores and manages secrets, encryption keys, and certificates.",
    "The CDN distributes static assets across global edge locations for low latency.",
    "Circuit breakers prevent cascading failures in distributed microservice architectures.",
    "Feature flags enable gradual rollout and A/B testing of new functionality.",
]

# Semantic similarity test pairs: (text_a, text_b, expected_high_similarity)
SIMILARITY_PAIRS = [
    ("zero-downtime releases", "blue-green deployment strategy", True),
    ("RTO", "recovery time objective", True),
    ("SBOM", "software bill of materials", True),
    ("CI/CD pipeline", "continuous integration and deployment", True),
    ("Kubernetes pod autoscaling", "container orchestration scaling", True),
    ("server inventory", "hardware asset list", True),
    ("deployment strategy", "cafeteria lunch menu", False),
    ("database replication", "office parking policy", False),
    ("server inventory", "employee birthday party", False),
]

# Rerank test cases: (query, documents, expected_best_index)
RERANK_TESTS = [
    (
        "deployment strategy for production",
        [
            "Blue-green deployment enables zero-downtime releases",
            "The cafeteria menu includes pasta on Thursdays",
            "Rolling deployments gradually replace old instances",
            "The office dress code is business casual",
        ],
        0,  # Blue-green should rank highest
    ),
    (
        "what Python version do we use",
        [
            "Kubernetes 1.29 manages container orchestration",
            "The SBOM lists Python 3.11.8, Django 5.0.2, PostgreSQL 15.4",
            "Server web-prod-01 has 32GB RAM and 8 cores",
            "The team uses Slack for communication",
        ],
        1,  # SBOM with Python version should rank highest
    ),
    (
        "database failover procedure",
        [
            "The team celebration is scheduled for Friday",
            "DR plan: failover to us-west-2 with synchronous database replication",
            "The parking lot has 200 spaces available",
            "PostgreSQL primary replicates to standby with automatic failover",
        ],
        1,  # DR plan should rank highest
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint Discovery
# ═══════════════════════════════════════════════════════════════════════════════

def discover_endpoint(base_url: str, headers: dict, kind: str = "embed") -> Optional[str]:
    """Auto-discover the working endpoint for embed or rerank.
    Handles servers at any path depth (e.g. /v1/embeddings, /embeddings,
    /some/long/path/v1/rerank, etc.).

    Strategy:
    1. If base_url already ends with /embeddings or /rerank, use it as-is
    2. Otherwise try appending common suffixes in order
    """
    base = base_url.rstrip("/")

    if kind == "embed":
        # If URL already ends with an embeddings path, try it directly
        if base.endswith("/embeddings"):
            try:
                r = requests.post(base, headers=headers, json={"input": ["test"]}, timeout=10)
                if r.status_code in (200, 401):
                    return base
            except Exception:
                pass

        # Try common suffixes
        suffixes = ["/v1/embeddings", "/embeddings"]
        test_payload = {"input": ["test"]}
    else:
        if base.endswith("/rerank"):
            try:
                r = requests.post(base, headers=headers,
                                  json={"query": "test", "documents": ["doc"]}, timeout=10)
                if r.status_code in (200, 401):
                    return base
            except Exception:
                pass

        suffixes = ["/v1/rerank", "/rerank"]
        test_payload = {"query": "test", "documents": ["doc"]}

    for suffix in suffixes:
        url = base + suffix
        try:
            r = requests.post(url, headers=headers, json=test_payload, timeout=10)
            if r.status_code in (200, 401, 400):
                return url
        except Exception:
            continue

    return None


def discover_health(base_url: str) -> dict:
    """Try common health/info endpoints and return whatever we find."""
    base = base_url.rstrip("/")
    for path in ["/health", "/info", "/v1/models", "/models", "/"]:
        try:
            r = requests.get(base + path, timeout=5)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return {"status": "ok", "_raw": r.text[:200]}
        except Exception:
            continue
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Core
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkResults:
    """Collects and formats benchmark results."""

    def __init__(self):
        self.results = {}
        self.server_info = {}

    def add(self, name: str, data: dict):
        self.results[name] = data

    def print_summary(self):
        print("\n" + "=" * 72)
        print("  BENCHMARK RESULTS")
        print("=" * 72)

        if self.server_info:
            print(f"\n  Server: {self.server_info.get('model', 'unknown')}")
            print(f"  Device: {self.server_info.get('device', 'unknown')} ({self.server_info.get('dtype', '')})")
            print(f"  Dims:   {self.server_info.get('embedding_dims', 'N/A')}")

        for name, data in self.results.items():
            print(f"\n  {name}")
            print(f"  {'-' * len(name)}")
            for key, value in data.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, float):
                    print(f"    {key:30s}  {value:>10.2f}")
                elif isinstance(value, list):
                    print(f"    {key:30s}  {value}")
                else:
                    print(f"    {key:30s}  {value}")
        print("\n" + "=" * 72)

    def to_json(self) -> dict:
        return {
            "version": __version__,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "server_info": self.server_info,
            "results": self.results,
        }


def percentile(data: list, p: float) -> float:
    """Calculate percentile from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def latency_stats(times_ms: list) -> dict:
    """Compute latency statistics from a list of times in ms."""
    if not times_ms:
        return {}
    return {
        "mean_ms": round(statistics.mean(times_ms), 2),
        "median_ms": round(statistics.median(times_ms), 2),
        "p95_ms": round(percentile(times_ms, 95), 2),
        "p99_ms": round(percentile(times_ms, 99), 2),
        "min_ms": round(min(times_ms), 2),
        "max_ms": round(max(times_ms), 2),
        "stdev_ms": round(statistics.stdev(times_ms), 2) if len(times_ms) > 1 else 0.0,
        "samples": len(times_ms),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Tests
# ═══════════════════════════════════════════════════════════════════════════════

def bench_single_latency(endpoint: str, headers: dict, warmup: int = 3, iterations: int = 20) -> dict:
    """Measure single-text embedding latency."""
    print("  Running single text latency benchmark...")
    text = BENCHMARK_TEXTS[0]

    # Warmup
    for _ in range(warmup):
        requests.post(endpoint, headers=headers, json={"input": [text]})

    # Measure
    times = []
    for i in range(iterations):
        t0 = time.perf_counter()
        r = requests.post(endpoint, headers=headers, json={"input": [text]})
        elapsed = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            times.append(elapsed)
        else:
            print(f"    WARNING: request {i} failed: {r.status_code}")

    stats = latency_stats(times)
    stats["texts_per_sec"] = round(1000.0 / stats["mean_ms"], 1) if stats.get("mean_ms") else 0
    return stats


def bench_batch_scaling(endpoint: str, headers: dict, warmup: int = 2) -> dict:
    """Measure how latency scales with batch size."""
    print("  Running batch scaling benchmark...")
    batch_sizes = [1, 4, 8, 16, 32]
    results = {}

    for size in batch_sizes:
        texts = BENCHMARK_TEXTS[:size] if size <= len(BENCHMARK_TEXTS) else (
            BENCHMARK_TEXTS * (size // len(BENCHMARK_TEXTS) + 1)
        )[:size]

        # Warmup
        for _ in range(warmup):
            requests.post(endpoint, headers=headers, json={"input": texts})

        # Measure (3 runs)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = requests.post(endpoint, headers=headers, json={"input": texts})
            elapsed = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                times.append(elapsed)

        if times:
            avg = statistics.mean(times)
            per_text = avg / size
            results[f"batch_{size}"] = {
                "total_ms": round(avg, 1),
                "per_text_ms": round(per_text, 1),
                "texts_per_sec": round(1000.0 / per_text, 1),
            }
            print(f"    batch={size:3d}: {avg:7.1f}ms total, {per_text:6.1f}ms/text, "
                  f"{1000/per_text:6.1f} texts/sec")

    return results


def bench_concurrent(endpoint: str, headers: dict, concurrency: int = 4, total_requests: int = 20) -> dict:
    """Measure throughput under concurrent load."""
    print(f"  Running concurrent benchmark (concurrency={concurrency})...")
    text = BENCHMARK_TEXTS[0]

    def do_request(_):
        t0 = time.perf_counter()
        r = requests.post(endpoint, headers=headers, json={"input": [text]})
        elapsed = (time.perf_counter() - t0) * 1000
        return elapsed if r.status_code == 200 else None

    t0_total = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(do_request, range(total_requests)))
    total_time = (time.perf_counter() - t0_total) * 1000

    times = [t for t in results if t is not None]
    errors = sum(1 for t in results if t is None)

    stats = latency_stats(times)
    stats["concurrency"] = concurrency
    stats["total_requests"] = total_requests
    stats["errors"] = errors
    stats["wall_time_ms"] = round(total_time, 1)
    stats["requests_per_sec"] = round(total_requests / (total_time / 1000), 1)
    return stats


def bench_rerank(endpoint: str, headers: dict, warmup: int = 2, iterations: int = 10) -> dict:
    """Measure reranking latency."""
    print("  Running rerank latency benchmark...")
    query = RERANK_TESTS[0][0]
    docs = RERANK_TESTS[0][1]

    # Warmup
    for _ in range(warmup):
        requests.post(endpoint, headers=headers, json={"query": query, "documents": docs})

    # Measure
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        r = requests.post(endpoint, headers=headers, json={"query": query, "documents": docs})
        elapsed = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            times.append(elapsed)

    stats = latency_stats(times)
    stats["documents_per_query"] = len(docs)
    if stats.get("mean_ms"):
        stats["per_doc_ms"] = round(stats["mean_ms"] / len(docs), 2)
    return stats


def bench_rerank_accuracy(endpoint: str, headers: dict) -> dict:
    """Test reranking quality on known test cases."""
    print("  Running rerank accuracy test...")
    correct = 0
    total = len(RERANK_TESTS)

    for query, docs, expected_idx in RERANK_TESTS:
        r = requests.post(endpoint, headers=headers, json={"query": query, "documents": docs})
        if r.status_code != 200:
            continue
        results = r.json().get("results", [])
        if results:
            best = max(results, key=lambda x: x.get("relevance_score", 0))
            if best["index"] == expected_idx:
                correct += 1
                print(f"    PASS: \"{query[:50]}\" -> idx {best['index']} (score={best['relevance_score']:.4f})")
            else:
                print(f"    FAIL: \"{query[:50]}\" -> idx {best['index']} (expected {expected_idx})")

    return {
        "accuracy": f"{correct}/{total}",
        "accuracy_pct": round(correct / total * 100, 1) if total else 0,
    }


def bench_similarity_quality(endpoint: str, headers: dict) -> dict:
    """Test embedding quality using semantic similarity pairs."""
    print("  Running similarity quality test...")

    correct = 0
    total = len(SIMILARITY_PAIRS)
    details = []

    for text_a, text_b, should_be_high in SIMILARITY_PAIRS:
        r = requests.post(endpoint, headers=headers, json={"input": [text_a, text_b]})
        if r.status_code != 200:
            continue

        data = r.json()
        emb_a = data["data"][0]["embedding"]
        emb_b = data["data"][1]["embedding"]

        # Cosine similarity
        dot = sum(a * b for a, b in zip(emb_a, emb_b))
        norm_a = math.sqrt(sum(a * a for a in emb_a))
        norm_b = math.sqrt(sum(b * b for b in emb_b))
        sim = dot / (norm_a * norm_b) if norm_a and norm_b else 0

        threshold = 0.75
        is_high = sim > threshold
        is_correct = is_high == should_be_high
        if is_correct:
            correct += 1

        label = "PASS" if is_correct else "FAIL"
        expected = "high" if should_be_high else "low"
        print(f"    {label}: {sim:.4f} ({expected:4s}) \"{text_a}\" vs \"{text_b}\"")
        details.append({
            "text_a": text_a, "text_b": text_b,
            "similarity": round(sim, 4), "expected_high": should_be_high,
            "correct": is_correct,
        })

    return {
        "accuracy": f"{correct}/{total}",
        "accuracy_pct": round(correct / total * 100, 1) if total else 0,
        "_details": details,
    }


def bench_long_text(endpoint: str, headers: dict, corpus_text: str = "") -> dict:
    """Measure latency vs text length. Uses diverse text to accurately
    represent real tokenization costs (repeated text under-counts tokens)."""
    print("  Running text length scaling benchmark...")
    # Use corpus text if available, otherwise generate diverse synthetic text
    if corpus_text and len(corpus_text) >= 16000:
        base_text = corpus_text
    else:
        base_text = generate_random_text(20000, seed=42)

    lengths = [100, 500, 1000, 2000, 4000, 8000, 16000]
    results = {}

    for target_len in lengths:
        # Use different offsets for each length to avoid repeating the same prefix
        offset = (target_len * 7) % max(1, len(base_text) - target_len)
        text = base_text[offset:offset + target_len]
        if len(text) < target_len:
            text = base_text[:target_len]

        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = requests.post(endpoint, headers=headers, json={"input": [text]})
            elapsed = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                times.append(elapsed)
            elif r.status_code in (413, 422, 400):
                # Input too long for model — record as max
                print(f"    {target_len:5d} chars: REJECTED (status {r.status_code})")
                results[f"len_{target_len}"] = {"chars": target_len, "status": "rejected"}
                break

        if times:
            avg = statistics.mean(times)
            results[f"len_{target_len}"] = {"chars": target_len, "mean_ms": round(avg, 1)}
            print(f"    {target_len:5d} chars: {avg:7.1f}ms")

    return results


def bench_max_input(endpoint: str, headers: dict, mode: str = "embed") -> dict:
    """Discover maximum input size by binary search. Reports the largest
    text length (in chars) that the model accepts without error."""
    print(f"  Probing max input size ({mode})...")

    # Generate diverse text for probing
    probe_text = generate_random_text(100000, seed=99)

    def try_size(n: int) -> bool:
        text = probe_text[:n]
        try:
            if mode == "embed":
                r = requests.post(endpoint, headers=headers,
                                  json={"input": [text]}, timeout=30)
            else:
                r = requests.post(endpoint, headers=headers,
                                  json={"query": "test query", "documents": [text]}, timeout=30)
            return r.status_code == 200
        except Exception:
            return False

    # Coarse scan: find approximate ceiling
    ceiling = 1000
    for size in [2000, 4000, 8000, 16000, 32000, 64000]:
        if try_size(size):
            ceiling = size
            print(f"    {size:6d} chars: OK")
        else:
            print(f"    {size:6d} chars: REJECTED")
            break

    # Binary search between last-good and first-bad
    lo, hi = ceiling, ceiling * 2
    while hi - lo > 500:
        mid = (lo + hi) // 2
        if try_size(mid):
            lo = mid
        else:
            hi = mid

    print(f"    Max input: ~{lo:,} chars")
    return {"max_input_chars": lo, "mode": mode}


def bench_rerank_scaling(endpoint: str, headers: dict, warmup: int = 2) -> dict:
    """Measure how rerank latency scales with document count."""
    print("  Running rerank document-count scaling...")
    query = "deployment strategy for production systems"
    base_docs = BENCHMARK_TEXTS[:32]
    doc_counts = [1, 4, 8, 16, 32]
    results = {}

    for count in doc_counts:
        docs = base_docs[:count] if count <= len(base_docs) else (
            base_docs * (count // len(base_docs) + 1))[:count]

        # Warmup
        for _ in range(warmup):
            requests.post(endpoint, headers=headers, json={"query": query, "documents": docs})

        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = requests.post(endpoint, headers=headers, json={"query": query, "documents": docs})
            elapsed = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                times.append(elapsed)

        if times:
            avg = statistics.mean(times)
            per_doc = avg / count
            results[f"docs_{count}"] = {
                "total_ms": round(avg, 1),
                "per_doc_ms": round(per_doc, 1),
                "docs_per_sec": round(1000.0 / per_doc, 1),
            }
            print(f"    docs={count:3d}: {avg:7.1f}ms total, {per_doc:6.1f}ms/doc")

    return results


def bench_random(endpoint: str, headers: dict, num_prompts: int = 50,
                 input_len: int = 128, batch_size: int = 1) -> dict:
    """Pure throughput benchmark with synthetic random text.
    Similar to vllm bench random — no accuracy testing, just raw speed.
    No external datasets needed."""
    print(f"  Running random text benchmark (n={num_prompts}, len={input_len}, batch={batch_size})...")

    # Generate all texts upfront
    texts = [generate_random_text(input_len, seed=i) for i in range(num_prompts)]

    # Warmup
    requests.post(endpoint, headers=headers, json={"input": texts[:1]})

    # Process in batches, measure total wall time
    t0_wall = time.perf_counter()
    total_tokens = 0
    total_embeddings = 0
    request_times = []

    for batch_start in range(0, num_prompts, batch_size):
        batch = texts[batch_start:batch_start + batch_size]
        t0 = time.perf_counter()
        r = requests.post(endpoint, headers=headers, json={"input": batch})
        elapsed = (time.perf_counter() - t0) * 1000
        request_times.append(elapsed)
        if r.status_code == 200:
            total_embeddings += len(batch)
            total_tokens += sum(len(t.split()) for t in batch)

    wall_time = (time.perf_counter() - t0_wall) * 1000

    stats = latency_stats(request_times)
    stats["total_prompts"] = num_prompts
    stats["input_len_chars"] = input_len
    stats["batch_size"] = batch_size
    stats["total_embeddings"] = total_embeddings
    stats["wall_time_ms"] = round(wall_time, 1)
    stats["embeddings_per_sec"] = round(total_embeddings / (wall_time / 1000), 1) if wall_time else 0
    stats["tokens_per_sec"] = round(total_tokens / (wall_time / 1000), 1) if wall_time else 0

    print(f"    {total_embeddings} embeddings in {wall_time:.0f}ms "
          f"= {stats['embeddings_per_sec']:.1f} emb/s, {stats['tokens_per_sec']:.0f} tok/s")
    return stats


def bench_random_mm(endpoint: str, headers: dict, num_prompts: int = 10,
                    input_len: int = 64, image_size: int = 64) -> dict:
    """Multimodal throughput benchmark with synthetic random text + images.
    Similar to vllm bench random-mm — generates random images inline,
    no external datasets needed."""
    print(f"  Running random multimodal benchmark (n={num_prompts}, text_len={input_len}, "
          f"img={image_size}x{image_size})...")

    # Generate inputs: alternate text and image
    times_text = []
    times_image = []
    times_combined = []

    for i in range(num_prompts):
        text = generate_random_text(input_len, seed=i)

        # Text-only
        t0 = time.perf_counter()
        r = requests.post(endpoint, headers=headers, json={"input": [text]})
        times_text.append((time.perf_counter() - t0) * 1000)

        # Image-only
        img_data = generate_random_image(image_size, image_size, seed=i)
        t0 = time.perf_counter()
        r = requests.post(endpoint, headers=headers, json={"input": [img_data]})
        elapsed = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            times_image.append(elapsed)
        else:
            print(f"    Image embed failed: {r.status_code}")

        # Combined text+image
        t0 = time.perf_counter()
        r = requests.post(endpoint, headers=headers, json={"input": [text, img_data]})
        elapsed = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            times_combined.append(elapsed)

    results = {}
    if times_text:
        results["text_only"] = {
            "mean_ms": round(statistics.mean(times_text), 1),
            "p95_ms": round(percentile(times_text, 95), 1),
        }
    if times_image:
        results["image_only"] = {
            "mean_ms": round(statistics.mean(times_image), 1),
            "p95_ms": round(percentile(times_image, 95), 1),
        }
    if times_combined:
        results["text_plus_image"] = {
            "mean_ms": round(statistics.mean(times_combined), 1),
            "p95_ms": round(percentile(times_combined, 95), 1),
        }
    results["num_prompts"] = num_prompts
    results["image_size"] = f"{image_size}x{image_size}"

    for mode, data in results.items():
        if isinstance(data, dict) and "mean_ms" in data:
            print(f"    {mode:20s}: mean={data['mean_ms']:.1f}ms  p95={data['p95_ms']:.1f}ms")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Nemotron Benchmark Tool — multimodal embedding & reranking benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        examples:
          %(prog)s --url http://localhost:8020 --api-key my_secret_key
          %(prog)s --url http://localhost:8020 --all
          %(prog)s --url http://localhost:8025 --rerank --api-key my_secret_key
          %(prog)s --url http://localhost:8020 --output results.json
        """),
    )
    parser.add_argument("--url", required=True, help="Server base URL (e.g. http://localhost:8020)")
    parser.add_argument("--api-key", default="", help="API key for authentication")
    parser.add_argument("--output", "-o", default="", help="Save results to JSON file")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    parser.add_argument("--iterations", "-n", type=int, default=20, help="Benchmark iterations (default: 20)")
    parser.add_argument("--concurrency", "-c", type=int, default=4, help="Concurrent requests (default: 4)")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--embed", action="store_true", help="Run embedding benchmarks")
    parser.add_argument("--rerank", action="store_true", help="Run rerank benchmarks")
    parser.add_argument("--quality", action="store_true", help="Run quality/accuracy benchmarks")
    parser.add_argument("--random", action="store_true",
                        help="Pure throughput: random text, no accuracy (like vllm bench random)")
    parser.add_argument("--random-mm", action="store_true",
                        help="Multimodal throughput: random text+images (like vllm bench random-mm)")
    parser.add_argument("--max-input", action="store_true",
                        help="Discover maximum input size via binary search")
    parser.add_argument("--corpus", default="",
                        help="Path to a text file for realistic long-text benchmarks (e.g. a book chapter)")
    parser.add_argument("--num-prompts", type=int, default=50,
                        help="Number of prompts for random/random-mm benchmarks (default: 50)")
    parser.add_argument("--input-len", type=int, default=128,
                        help="Input text length in chars for random benchmarks (default: 128)")
    parser.add_argument("-b", "--batch-size", type=int, default=1,
                        help="Batch size for random benchmarks (default: 1)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    url = args.url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    # Check server health
    print(f"\nConnecting to {url}...")
    info = discover_health(url)
    if not info:
        print(f"  WARNING: No health endpoint found at {url}, continuing anyway...")
        info = {}
    else:
        print(f"  Model: {info.get('model', 'unknown')}")
        print(f"  Type:  {info.get('model_type', 'unknown')}")
        if info.get('device'):
            print(f"  Device: {info.get('device', 'unknown')} ({info.get('dtype', '')})")

    results = BenchmarkResults()
    results.server_info = info

    # Load corpus text if provided
    corpus_text = ""
    if args.corpus:
        try:
            corpus_text = Path(args.corpus).read_text(encoding="utf-8", errors="ignore")
            print(f"  Corpus loaded: {len(corpus_text):,} chars from {args.corpus}")
        except Exception as e:
            print(f"  WARNING: Could not load corpus: {e}")

    # Determine what to run
    has_specific = (args.embed or args.rerank or args.quality or args.random
                    or getattr(args, 'random_mm', False) or args.max_input)
    run_embed = args.embed or args.all or (not has_specific)
    run_rerank = args.rerank or args.all
    run_quality = args.quality or args.all
    run_random = args.random or args.all
    run_random_mm = getattr(args, 'random_mm', False) or args.all
    run_max_input = args.max_input or args.all

    # Auto-detect model type from health info, or from what endpoints exist
    model_type = info.get("model_type", "")
    if not model_type:
        # Try to discover: if user asked for --rerank, assume rerank; otherwise probe
        if args.rerank and not args.embed:
            model_type = "rerank"
        elif args.embed and not args.rerank:
            model_type = "embed"
        else:
            # Probe both
            if discover_endpoint(url, headers, "embed"):
                model_type = "embed"
            elif discover_endpoint(url, headers, "rerank"):
                model_type = "rerank"
            else:
                print("  ERROR: Could not find any embedding or rerank endpoint.")
                sys.exit(1)
        print(f"  Auto-detected type: {model_type}")

    # Discover endpoints
    embed_endpoint = None
    rerank_endpoint = None

    need_embed = (run_embed or run_random or run_random_mm or
                  (run_quality and model_type == "embed") or
                  (run_max_input and model_type == "embed"))
    need_rerank = (run_rerank or
                   (run_quality and model_type == "rerank") or
                   (run_max_input and model_type == "rerank"))

    if need_embed:
        embed_endpoint = discover_endpoint(url, headers, "embed")
        if embed_endpoint:
            print(f"  Embed endpoint: {embed_endpoint}")
        elif model_type == "embed":
            print("  ERROR: Could not find embedding endpoint")
            sys.exit(1)

    if need_rerank:
        rerank_endpoint = discover_endpoint(url, headers, "rerank")
        if rerank_endpoint:
            print(f"  Rerank endpoint: {rerank_endpoint}")
        elif model_type == "rerank":
            print("  ERROR: Could not find rerank endpoint")
            sys.exit(1)

    if run_max_input:
        print("\n--- Max Input Size Discovery ---")
        if model_type == "rerank" and rerank_endpoint:
            results.add("Max Input Size", bench_max_input(rerank_endpoint, headers, mode="rerank"))
        elif embed_endpoint:
            results.add("Max Input Size", bench_max_input(embed_endpoint, headers, mode="embed"))

    if run_embed and model_type == "embed" and embed_endpoint:
        print("\n--- Embedding Benchmarks ---")
        results.add("Single Text Latency", bench_single_latency(embed_endpoint, headers, args.warmup, args.iterations))
        results.add("Batch Scaling", bench_batch_scaling(embed_endpoint, headers, args.warmup))
        results.add("Concurrent Requests", bench_concurrent(embed_endpoint, headers, args.concurrency))
        results.add("Text Length Scaling", bench_long_text(embed_endpoint, headers, corpus_text=corpus_text))

    if run_rerank and model_type == "rerank" and rerank_endpoint:
        print("\n--- Rerank Benchmarks ---")
        results.add("Rerank Latency", bench_rerank(rerank_endpoint, headers, args.warmup, args.iterations))
        results.add("Rerank Document Scaling", bench_rerank_scaling(rerank_endpoint, headers, args.warmup))

    if run_quality:
        print("\n--- Quality Benchmarks ---")
        if model_type == "embed" and embed_endpoint:
            results.add("Semantic Similarity Quality", bench_similarity_quality(embed_endpoint, headers))
        if model_type == "rerank" and rerank_endpoint:
            results.add("Rerank Accuracy", bench_rerank_accuracy(rerank_endpoint, headers))

    if run_random and model_type == "embed" and embed_endpoint:
        print("\n--- Random Text Throughput (no accuracy, pure speed) ---")
        results.add("Random Text Throughput", bench_random(
            embed_endpoint, headers, num_prompts=args.num_prompts,
            input_len=args.input_len, batch_size=args.batch_size,
        ))

    if run_random_mm and model_type == "embed" and embed_endpoint:
        print("\n--- Random Multimodal Throughput (text + synthetic images) ---")
        results.add("Random Multimodal Throughput", bench_random_mm(
            embed_endpoint, headers, num_prompts=min(args.num_prompts, 10),
            input_len=args.input_len,
        ))

    results.print_summary()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results.to_json(), f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
