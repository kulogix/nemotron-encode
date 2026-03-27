# nemotron-encode

High-performance inference server for NVIDIA Nemotron VL embedding and reranking models, with comprehensive benchmarking tools.

Serves [llama-nemotron-embed-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) and [llama-nemotron-rerank-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2) via an OpenAI-compatible API with automatic model type detection, cross-platform device selection, Prometheus metrics, and a built-in test UI.

## Features

- **OpenAI-compatible API** — Drop-in replacement for `/v1/embeddings` and `/v1/rerank` endpoints
- **Multimodal** — Text and image embedding in the same 2048-dimensional vector space
- **Auto-detection** — Model type (embed vs rerank) detected from `config.json`, no manual configuration
- **Cross-platform** — NVIDIA CUDA, AMD ROCm, Intel XPU, Apple Silicon, x86/ARM CPUs
- **Production-ready** — Prometheus `/metrics`, `/health` readiness probe, graceful shutdown, API key auth
- **Built-in test UI** — Interactive browser interface at `/` for quick testing
- **Comprehensive benchmarks** — Latency, throughput, batch scaling, concurrent load, semantic quality tests, and synthetic random workloads (like `vllm bench random`)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download models (see "Obtaining Models" below)

# 3. Start embedding server
python server.py --model-dir ./models/nvidia_llama-nemotron-embed-vl-1b-v2

# 4. Start reranking server (in another terminal)
python server.py --model-dir ./models/nvidia_llama-nemotron-rerank-vl-1b-v2 --port 8025

# 5. Test it
curl http://localhost:8020/health
curl -X POST http://localhost:8020/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["hello world"]}'
```

Open http://localhost:8020 in your browser for the interactive test UI.

## Obtaining Models

### Option A: Hugging Face CLI (recommended)

```bash
pip install huggingface-hub

# Embedding model (~3.2 GB)
huggingface-cli download nvidia/llama-nemotron-embed-vl-1b-v2 \
  --local-dir ./models/nvidia_llama-nemotron-embed-vl-1b-v2

# Reranking model (~3.1 GB)
huggingface-cli download nvidia/llama-nemotron-rerank-vl-1b-v2 \
  --local-dir ./models/nvidia_llama-nemotron-rerank-vl-1b-v2
```

### Option B: Git LFS

```bash
git lfs install

git clone https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2 \
  ./models/nvidia_llama-nemotron-embed-vl-1b-v2

git clone https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2 \
  ./models/nvidia_llama-nemotron-rerank-vl-1b-v2
```

> **Note:** Both methods require ~6.3 GB total disk space. No Hugging Face account or token is needed — these models are publicly available.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/embeddings` | POST | OpenAI-compatible text/image embedding |
| `/v1/rerank` | POST | Cross-encoder document reranking |
| `/rerank` | POST | Alias for `/v1/rerank` |
| `/v1/similarity` | POST | Pairwise cosine similarity |
| `/tokenize` | POST | Token count estimation |
| `/v1/models` | GET | Model information |
| `/health` | GET | Readiness probe (no auth required) |
| `/metrics` | GET | Prometheus-compatible metrics |
| `/docs` | GET | OpenAPI/Swagger documentation |
| `/` | GET | Interactive test UI |

### Embedding Request

```bash
curl -X POST http://localhost:8020/v1/embeddings \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Blue-green deployment enables zero-downtime releases."]
  }'
```

Supports batch input (list of strings) and image input (base64 data URLs or OpenAI-style `image_url` objects).

### Rerank Request

```bash
curl -X POST http://localhost:8025/v1/rerank \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deployment strategy",
    "documents": ["Blue-green deployment...", "Cafeteria menu..."],
    "top_n": 3
  }'
```

## Server Options

```
python server.py --help
```

### Model & Server

| Flag | Default | Description |
|---|---|---|
| `--model-dir` | (required) | Path to HuggingFace model directory |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8020` | Listen port |
| `--api-key` | (none) | Bearer token for authentication |
| `--auto-port` | off | Find next free port if default is in use |
| `--workers` | `1` | Uvicorn worker processes |
| `--log-level` | `info` | `debug`, `info`, `warning`, `error` |

### Performance Tuning

| Flag | Default | Description |
|---|---|---|
| `-t, --threads` | auto | CPU inference threads (see [Thread Tuning](#thread-tuning)) |
| `-b, --batch-size` | `32` | Maximum batch size per request |
| `-c, --ctx-size` | model config | Override context length (tokens) |
| `--dtype` | `auto` | `float32`, `float16`, `bfloat16`, or `auto` |
| `--device` | auto | Force `cpu`, `cuda`, `mps`, or `xpu` |
| `--warmup` | enabled | Run probe inference after load to warm caches |
| `--no-warmup` | — | Skip warmup (faster startup) |
| `--timeout` | `120` | Request timeout in seconds |
| `--max-concurrent-requests` | `64` | Backpressure limit |

## Performance Guide

### Memory Requirements

Both Nemotron VL 1B v2 models have ~1.7B parameters:

| Configuration | Embed Model | Rerank Model | Both Models |
|---|---|---|---|
| float32 | ~6.8 GB | ~6.4 GB | ~13.2 GB |
| bfloat16 (default) | ~3.4 GB | ~3.2 GB | ~6.6 GB |
| float16 | ~3.4 GB | ~3.2 GB | ~6.6 GB |

**Formula:** `memory_gb = parameters × bytes_per_param / 1e9`
- float32: `1.7B × 4 = 6.8 GB`
- bfloat16/float16: `1.7B × 2 = 3.4 GB`

Add ~0.5-1 GB overhead for activation memory during inference (varies with batch size and sequence length).

**For systems with limited memory:** Use `--dtype bfloat16` (default) and run only one model at a time. Both models fit in 8 GB of RAM.

### Thread Tuning

CPU inference performance is highly sensitive to thread count. The optimal value depends on your hardware:

| Platform | Recommended | Why |
|---|---|---|
| Apple Silicon (M1-M4) | `-t 8` | Performance cores only; hyperthreading overhead on efficiency cores |
| Intel (no HT) | `-t <physical_cores>` | One thread per physical core |
| Intel (with HT) | `-t <physical_cores>` | Hyperthreading adds overhead for inference |
| AMD Ryzen | `-t <physical_cores>` | CCX topology makes more threads counterproductive |

The server auto-detects `min(cpu_count, 8)` by default. Override with `-t` if benchmarks show a different optimum.

**Finding your optimal thread count:**
```bash
# Quick sweep — try 1, 2, 4, 8, 16 threads and compare
for t in 1 2 4 8 16; do
  echo "=== threads=$t ==="
  python server.py --model-dir ./models/nvidia_llama-nemotron-embed-vl-1b-v2 \
    -t $t --no-warmup --port 0 &  # port 0 = don't actually serve
  # (or use the benchmark tool)
done
```

### Device Selection

The server automatically selects the best available device:

| Priority | Device | When Used | Performance |
|---|---|---|---|
| 1 | CUDA | NVIDIA GPU detected | Fastest (with flash_attention_2) |
| 2 | Intel XPU | Intel discrete GPU | Fast |
| 3 | CPU (bfloat16) | Apple Silicon detected | Good — see note below |
| 4 | CPU (float32) | Fallback | Baseline |

#### Apple Silicon Note

While Apple Silicon supports MPS (Metal Performance Shaders), the Nemotron VL architecture uses custom operations (bidirectional attention, pixel shuffle, dynamic image tiling) that trigger MPS-to-CPU fallbacks. In testing, **MPS is 13x slower than CPU+bfloat16** for this specific model:

| Device | Single Text Latency | Notes |
|---|---|---|
| CPU + bfloat16 | 142ms | Default on Apple Silicon |
| MPS + float32 | 1,850ms | Fallback ops dominate |

The server automatically selects CPU on Apple Silicon. You can force MPS with `--device mps` for experimentation, but it's not recommended.

### GGUF / llama.cpp Compatibility

GGUF conversion is **not currently feasible** for these models because:

1. **Bidirectional attention** — Nemotron VL uses non-causal (bidirectional) self-attention, while `convert_hf_to_gguf.py` assumes causal/decoder-only architectures
2. **Custom vision encoder** — SigLip2 vision encoder requires architecture-specific support not yet in llama.cpp
3. **No community conversions** — As of March 2026, no GGUF versions of any Nemotron embed/rerank model exist on Hugging Face
4. **Embedding model bug** — llama.cpp has a known issue (#14459) with embedding-only model conversion

The PyTorch inference path with CPU+bfloat16 and thread tuning remains the recommended approach.

## Benchmarks

All benchmarks collected on **MacBook Pro M3 Max** (16-core CPU, 128 GB unified memory), PyTorch 2.11.0, bfloat16, 8 threads.

### NVIDIA Baseline vs nemotron-encode Server

Comparing raw `transformers` library usage (NVIDIA's demo code) against the optimized server:

| Metric | NVIDIA Demo (raw) | nemotron-encode | Change |
|---|---|---|---|
| Model load time | 1.3s | 3.2s (includes warmup) | +1.9s startup |
| Single text mean latency | 198.1ms | 176.9ms | **-10.7% faster** |
| Single text p95 latency | 203.7ms | 182.1ms | **-10.6% faster** |
| Batch 8 per-text latency | 158.2ms | 162.7ms | +2.8% (HTTP overhead) |
| First request after cold start | ~220ms | ~180ms | **-18% (warmup effect)** |
| Latency std deviation | 8.6ms | 3.7ms | **57% more consistent** |

The server adds ~3ms HTTP overhead per request but provides 10-18% faster inference due to thread tuning and warmup. Latency consistency improves significantly.

### Throughput Benchmarks

```
$ python benchmark.py --url http://localhost:8020 --api-key KEY --random --num-prompts 50

Random Text Throughput:  5.1 embeddings/sec, 77 tokens/sec
```

### Batch Scaling

| Batch Size | Total (ms) | Per Text (ms) | Texts/sec |
|---|---|---|---|
| 1 | 182 | 182 | 5.5 |
| 4 | 831 | 208 | 4.8 |
| 8 | 1,703 | 213 | 4.7 |
| 16 | 3,405 | 213 | 4.7 |
| 32 | 8,069 | 252 | 4.0 |

The model processes texts sequentially internally, so batch size doesn't improve per-text throughput. Batch size 1-8 is optimal for latency.

### Reranking Performance

| Metric | Value |
|---|---|
| 4 documents | 662ms mean |
| Per-document cost | 166ms |
| 10 documents | 1,252ms mean |

### Multimodal Performance

| Input Type | Mean Latency |
|---|---|
| Text only (short) | 137ms |
| Image only (64x64) | 23,400ms |
| Text + Image | 23,700ms |

Image embedding is compute-intensive due to the SigLip2 vision encoder + pixel shuffle + MLP projection. On CUDA with flash_attention_2, expect 5-10x faster image processing.

### Semantic Quality

```
$ python benchmark.py --url http://localhost:8020 --api-key KEY --quality

Semantic Similarity Quality: 88.9% (8/9 pairs correct)
  0.8148  "zero-downtime releases" vs "blue-green deployment strategy"
  0.8072  "SBOM" vs "software bill of materials"
  0.8887  "CI/CD pipeline" vs "continuous integration and deployment"
  0.7158  "deployment strategy" vs "cafeteria lunch menu" (correctly low)
```

## Benchmark Tool

The benchmark tool supports multiple modes similar to `vllm bench`:

```bash
# Full suite (latency + throughput + quality)
python benchmark.py --url http://localhost:8020 --api-key KEY --all

# Pure throughput with synthetic random text (no external datasets needed)
python benchmark.py --url http://localhost:8020 --random --num-prompts 100 -b 8

# Multimodal throughput with synthetic random images
python benchmark.py --url http://localhost:8020 --random-mm --num-prompts 10

# Reranking benchmarks
python benchmark.py --url http://localhost:8025 --rerank --api-key KEY

# Save results to JSON
python benchmark.py --url http://localhost:8020 --all -o results.json
```

### Benchmark Options

| Flag | Default | Description |
|---|---|---|
| `--url` | (required) | Server URL |
| `--api-key` | (none) | Bearer token |
| `--all` | off | Run all benchmarks |
| `--embed` | off | Embedding latency/throughput |
| `--rerank` | off | Reranking latency |
| `--quality` | off | Semantic accuracy tests |
| `--random` | off | Pure throughput with synthetic text |
| `--random-mm` | off | Multimodal throughput |
| `--num-prompts` | `50` | Prompts for random benchmarks |
| `--input-len` | `128` | Text length (chars) for random |
| `-b, --batch-size` | `1` | Batch size for random |
| `-n, --iterations` | `20` | Iterations per test |
| `-c, --concurrency` | `4` | Concurrent request threads |
| `-o, --output` | (none) | Save results as JSON |

## Architecture

```
nemotron-encode/
  server.py          # Inference server (FastAPI + PyTorch)
  benchmark.py       # Benchmarking tool
  requirements.txt   # Python dependencies
  README.md          # This file
  models/            # Model weights (not included, see "Obtaining Models")
    nvidia_llama-nemotron-embed-vl-1b-v2/
    nvidia_llama-nemotron-rerank-vl-1b-v2/
```

The server auto-detects the model type from `config.json`:
- If `model_type` contains "rerank", architectures contain "SequenceClassification", or `num_labels` is present → **rerank mode** (serves `/v1/rerank`)
- Otherwise → **embed mode** (serves `/v1/embeddings`)

This means you run the same `server.py` for both models — just point `--model-dir` at the appropriate directory.

## Running Both Models

A common setup runs the embedding and reranking models on separate ports:

```bash
# Terminal 1: Embedding server
python server.py \
  --model-dir ./models/nvidia_llama-nemotron-embed-vl-1b-v2 \
  --port 8020 \
  --api-key my_secret_key \
  -t 8

# Terminal 2: Reranking server
python server.py \
  --model-dir ./models/nvidia_llama-nemotron-rerank-vl-1b-v2 \
  --port 8025 \
  --api-key my_secret_key \
  -t 8
```

**Memory requirement:** ~6.6 GB total for both models in bfloat16.

## Integration

### Open WebUI

Configure as external embedding and reranking endpoints:

```bash
# In your Open WebUI startup script:
export RAG_EMBEDDING_ENGINE=openai
export RAG_EMBEDDING_MODEL="nvidia_llama-nemotron-embed-vl-1b-v2"
export RAG_OPENAI_API_BASE_URL="http://localhost:8020"
export RAG_OPENAI_API_KEY="my_secret_key"

export RAG_RERANKING_ENGINE=external
export RAG_RERANKING_MODEL="nvidia_llama-nemotron-rerank-vl-1b-v2"
export RAG_EXTERNAL_RERANKER_URL="http://localhost:8025/v1/rerank"
export RAG_EXTERNAL_RERANKER_API_KEY="my_secret_key"
```

### LangChain

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="nvidia_llama-nemotron-embed-vl-1b-v2",
    openai_api_base="http://localhost:8020/v1",
    openai_api_key="my_secret_key",
)

vectors = embeddings.embed_documents(["hello world"])
```

### Direct API (Python)

```python
import requests

response = requests.post(
    "http://localhost:8020/v1/embeddings",
    headers={"Authorization": "Bearer my_secret_key"},
    json={"input": ["hello world"]},
)
embedding = response.json()["data"][0]["embedding"]  # 2048-dim vector
```

## About the Models

| | Embedding Model | Reranking Model |
|---|---|---|
| Name | llama-nemotron-embed-vl-1b-v2 | llama-nemotron-rerank-vl-1b-v2 |
| Parameters | ~1.7B | ~1.7B |
| Output | 2048-dim embedding vector | Relevance score (0-1) |
| Architecture | SigLip2 + Bidirectional LLaMA | SigLip2 + Bidirectional LLaMA + Classification Head |
| Modalities | Text, Image, Text+Image | Text, Image, Text+Image |
| Context | 8,192 tokens (text), 6 tiles (image) | 8,192 tokens |
| License | [cc-by-4.0](https://creativecommons.org/licenses/by/4.0/) | [cc-by-4.0](https://creativecommons.org/licenses/by/4.0/) |

## License

This server and benchmark code is released under the MIT License.

The Nemotron VL models themselves are licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) by NVIDIA.
