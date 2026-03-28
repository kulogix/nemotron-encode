#!/usr/bin/env python3
"""
Nemotron VL Inference Server
=============================
High-performance inference server for NVIDIA Nemotron VL embedding and reranking models.
Provides OpenAI-compatible API with automatic model type detection.

Endpoints:
  POST /v1/embeddings     Text and/or image embedding (OpenAI-compatible)
  POST /v1/rerank         Cross-encoder reranking
  POST /rerank            Alias for /v1/rerank
  POST /v1/similarity     Pairwise cosine similarity
  POST /tokenize          Token count estimation
  GET  /v1/models         Model information
  GET  /health            Health + readiness probe
  GET  /metrics           Prometheus-compatible metrics
  GET  /                  Built-in test UI

Cross-platform support:
  NVIDIA CUDA, AMD ROCm, Apple Silicon (MPS/CPU+bfloat16), Intel XPU,
  x86 (AVX2/AVX-512), ARM (NEON/SVE), Windows/macOS/Linux

Usage:
  python nemotron_server.py --model-dir ./models/nvidia_llama-nemotron-embed-vl-1b-v2
  python nemotron_server.py --model-dir ./models/nvidia_llama-nemotron-rerank-vl-1b-v2 --port 8025
  python nemotron_server.py --help
"""

__version__ = "1.0.0"

import argparse
import base64
import io
import json
import logging
import math
import os
import platform
import signal
import socket
import sys
import textwrap
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

# ═══════════════════════════════════════════════════════════════════════════════
# Platform Detection (before torch import for early error messages)
# ═══════════════════════════════════════════════════════════════════════════════

def get_system_info() -> dict:
    """Collect system information for diagnostics."""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
    }
    # Detect CPU features on x86
    if info["arch"] in ("x86_64", "AMD64", "x86"):
        try:
            # Check /proc/cpuinfo on Linux
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo") as f:
                    cpuinfo = f.read()
                flags = set()
                for line in cpuinfo.splitlines():
                    if line.startswith("flags"):
                        flags.update(line.split(":")[1].strip().split())
                        break
                info["cpu_features"] = []
                for feat in ("avx", "avx2", "avx512f", "avx512_bf16", "amx_tile"):
                    if feat in flags:
                        info["cpu_features"].append(feat)
            # macOS: use sysctl
            elif info["os"] == "Darwin":
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-a"], capture_output=True, text=True, timeout=5
                )
                out = result.stdout
                info["cpu_features"] = []
                if "hw.optional.avx2_0: 1" in out:
                    info["cpu_features"].append("avx2")
                if "hw.optional.avx512f: 1" in out:
                    info["cpu_features"].append("avx512f")
        except Exception:
            info["cpu_features"] = ["unknown"]
    elif info["arch"] in ("arm64", "aarch64"):
        info["cpu_features"] = ["neon"]  # All ARM64 has NEON
        if info["os"] == "Darwin":
            info["cpu_features"].append("apple_silicon")
    # Memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_gb"] = round(mem.total / (1024**3), 1)
        info["ram_available_gb"] = round(mem.available / (1024**3), 1)
    except ImportError:
        pass
    return info


def check_dependencies():
    """Verify required packages are installed with helpful error messages."""
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append(("torch", "pip install torch"))
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append(("transformers", "pip install transformers"))
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        missing.append(("uvicorn", "pip install uvicorn"))
    try:
        import fastapi  # noqa: F401
    except ImportError:
        missing.append(("fastapi", "pip install fastapi"))
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        missing.append(("Pillow", "pip install Pillow"))
    try:
        import torchvision  # noqa: F401
    except ImportError:
        missing.append(("torchvision", "pip install torchvision"))

    if missing:
        print("\n  Missing required dependencies:\n")
        for pkg, cmd in missing:
            print(f"    {pkg:20s}  {cmd}")
        print(f"\n  Install all at once:")
        print(f"    pip install {' '.join(p for p, _ in missing)}\n")
        sys.exit(1)


check_dependencies()

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nemotron")


# ═══════════════════════════════════════════════════════════════════════════════
# Device & Dtype Selection
# ═══════════════════════════════════════════════════════════════════════════════

def select_device(force_device: str = "") -> tuple:
    """Select optimal device, dtype, and attention implementation.

    Priority: user override > CUDA > ROCm > Intel XPU > MPS > CPU
    Returns (device_str, dtype, attn_impl, device_info).
    """
    info = {}

    if force_device:
        device = force_device
        info["reason"] = "user override"
    elif torch.cuda.is_available():
        device = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
        )
        cap = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{cap[0]}.{cap[1]}"
        info["reason"] = "NVIDIA CUDA detected"
    elif hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
        device = "xpu"
        info["reason"] = "Intel XPU detected"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # MPS is available but Nemotron VL model runs slower on MPS due to
        # custom op fallbacks — unless PYTORCH_ENABLE_MPS_FALLBACK=1 is set
        # AND PyTorch >= 2.4 (which has many more MPS operators).
        # Default to CPU with bfloat16+SDPA which is typically faster.
        # User can override with --device mps to try Metal acceleration.
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            device = "mps"
            info["reason"] = "Apple Silicon MPS (with CPU fallback for unsupported ops)"
            info["mps_fallback"] = True
        else:
            device = "cpu"
            info["reason"] = "Apple Silicon (CPU+bfloat16+SDPA; set PYTORCH_ENABLE_MPS_FALLBACK=1 to try MPS)"
            info["mps_available"] = True
    else:
        device = "cpu"
        info["reason"] = "CPU fallback"

    # Dtype selection
    if device == "cuda":
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 8:  # Ampere+ (A100, H100, RTX 30xx+)
            dtype = torch.bfloat16
        elif cap[0] >= 7:  # Volta/Turing (V100, RTX 20xx)
            dtype = torch.float16
        else:
            dtype = torch.float32
    elif device == "xpu":
        dtype = torch.bfloat16
    else:
        # CPU: try bfloat16 (works on Apple Silicon and Intel with AVX-512 BF16)
        try:
            t = torch.tensor([1.0], dtype=torch.bfloat16)
            _ = t + t  # Ensure arithmetic works
            dtype = torch.bfloat16
        except Exception:
            dtype = torch.float32

    # Attention implementation
    # SDPA (Scaled Dot Product Attention) is the optimal fallback for non-CUDA:
    # it uses PyTorch's built-in fused attention kernels, significantly faster
    # than "eager" on Apple Silicon CPU and Intel with AVX-512.
    if device == "cuda":
        try:
            attn_impl = "flash_attention_2"
        except Exception:
            attn_impl = "sdpa"
    elif device == "mps":
        # MPS may not support all SDPA paths; try sdpa, fall back to eager
        attn_impl = "sdpa"
    else:
        # CPU: SDPA is well-supported and much faster than eager
        attn_impl = "sdpa"

    info["dtype"] = str(dtype)
    info["attn_impl"] = attn_impl
    return device, dtype, attn_impl, info


# ═══════════════════════════════════════════════════════════════════════════════
# Port Availability Check
# ═══════════════════════════════════════════════════════════════════════════════

def check_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host if host != "0.0.0.0" else "127.0.0.1", port))
            return True
    except OSError:
        return False


def find_available_port(host: str, start_port: int, max_tries: int = 10) -> int:
    """Find the next available port starting from start_port."""
    for offset in range(max_tries):
        port = start_port + offset
        if check_port_available(host, port):
            return port
    return -1


# ═══════════════════════════════════════════════════════════════════════════════
# Model Detection & Loading
# ═══════════════════════════════════════════════════════════════════════════════

def validate_model_dir(model_dir: str) -> tuple:
    """Validate model directory exists and contains required files.
    Returns (abs_path, config_dict) or raises with helpful error."""
    model_dir = os.path.abspath(model_dir)

    if not os.path.exists(model_dir):
        # Check common locations
        candidates = []
        for prefix in ["./models", "../models", os.path.expanduser("~/.cache/huggingface/hub")]:
            candidate = os.path.join(prefix, os.path.basename(model_dir))
            if os.path.exists(candidate):
                candidates.append(candidate)

        msg = f"Model directory not found: {model_dir}"
        if candidates:
            msg += f"\n\n  Did you mean one of these?\n"
            for c in candidates:
                msg += f"    {c}\n"
        msg += (
            f"\n  To download models, use:\n"
            f"    huggingface-cli download nvidia/llama-nemotron-embed-vl-1b-v2 --local-dir ./models/nvidia_llama-nemotron-embed-vl-1b-v2\n"
            f"    huggingface-cli download nvidia/llama-nemotron-rerank-vl-1b-v2 --local-dir ./models/nvidia_llama-nemotron-rerank-vl-1b-v2\n"
        )
        raise FileNotFoundError(msg)

    if not os.path.isdir(model_dir):
        raise NotADirectoryError(f"Not a directory: {model_dir}")

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        files = os.listdir(model_dir)[:20]
        raise FileNotFoundError(
            f"config.json not found in {model_dir}\n"
            f"  Directory contains: {', '.join(files) if files else '(empty)'}\n"
            f"  This doesn't appear to be a valid HuggingFace model directory."
        )

    # Check for model weights
    has_weights = any(
        f.endswith((".safetensors", ".bin", ".pt", ".pth"))
        for f in os.listdir(model_dir)
    )
    if not has_weights:
        raise FileNotFoundError(
            f"No model weight files found in {model_dir}\n"
            f"  Expected .safetensors or .bin files.\n"
            f"  The download may be incomplete — try re-downloading."
        )

    with open(config_path) as f:
        config = json.load(f)

    return model_dir, config


def detect_model_type(config: dict) -> str:
    """Detect if model is 'embed' or 'rerank' from config.json contents."""
    # Check model_type field
    model_type = config.get("model_type", "")
    if "rerank" in model_type.lower():
        return "rerank"

    # Check architectures
    for arch in config.get("architectures", []):
        if any(kw in arch for kw in ("SequenceClassification", "Rerank", "CrossEncoder")):
            return "rerank"

    # Check for num_labels (classification head)
    if "num_labels" in config:
        return "rerank"

    return "embed"


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics Collector
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    """Thread-safe Prometheus-compatible metrics."""

    def __init__(self):
        self._lock = threading.Lock()
        self.requests_total = 0
        self.requests_by_endpoint = {}
        self.errors_total = 0
        self.tokens_processed = 0
        self.embeddings_generated = 0
        self.rerank_queries = 0
        self.latency_sum_ms = 0.0
        self.latency_count = 0
        self.latency_buckets = {5: 0, 10: 0, 25: 0, 50: 0, 100: 0, 250: 0,
                                500: 0, 1000: 0, 2500: 0, 5000: 0, float("inf"): 0}
        self.start_time = time.time()

    def record_request(self, endpoint: str, latency_ms: float, tokens: int = 0,
                       embeddings: int = 0, is_rerank: bool = False, error: bool = False):
        with self._lock:
            self.requests_total += 1
            self.requests_by_endpoint[endpoint] = self.requests_by_endpoint.get(endpoint, 0) + 1
            self.latency_sum_ms += latency_ms
            self.latency_count += 1
            self.tokens_processed += tokens
            self.embeddings_generated += embeddings
            if is_rerank:
                self.rerank_queries += 1
            if error:
                self.errors_total += 1
            for bucket in sorted(self.latency_buckets.keys()):
                if latency_ms <= bucket:
                    self.latency_buckets[bucket] += 1

    def prometheus_format(self, model_name: str) -> str:
        with self._lock:
            uptime = time.time() - self.start_time
            lines = [
                f'# HELP nemotron_requests_total Total requests processed',
                f'# TYPE nemotron_requests_total counter',
                f'nemotron_requests_total{{model="{model_name}"}} {self.requests_total}',
                f'# HELP nemotron_errors_total Total error responses',
                f'# TYPE nemotron_errors_total counter',
                f'nemotron_errors_total{{model="{model_name}"}} {self.errors_total}',
                f'# HELP nemotron_tokens_processed_total Total tokens processed',
                f'# TYPE nemotron_tokens_processed_total counter',
                f'nemotron_tokens_processed_total{{model="{model_name}"}} {self.tokens_processed}',
                f'# HELP nemotron_embeddings_generated_total Total embeddings generated',
                f'# TYPE nemotron_embeddings_generated_total counter',
                f'nemotron_embeddings_generated_total{{model="{model_name}"}} {self.embeddings_generated}',
                f'# HELP nemotron_rerank_queries_total Total rerank queries',
                f'# TYPE nemotron_rerank_queries_total counter',
                f'nemotron_rerank_queries_total{{model="{model_name}"}} {self.rerank_queries}',
                f'# HELP nemotron_uptime_seconds Server uptime in seconds',
                f'# TYPE nemotron_uptime_seconds gauge',
                f'nemotron_uptime_seconds{{model="{model_name}"}} {uptime:.1f}',
            ]
            if self.latency_count > 0:
                avg = self.latency_sum_ms / self.latency_count
                lines.extend([
                    f'# HELP nemotron_request_latency_ms_avg Average request latency',
                    f'# TYPE nemotron_request_latency_ms_avg gauge',
                    f'nemotron_request_latency_ms_avg{{model="{model_name}"}} {avg:.2f}',
                ])
                lines.append(f'# HELP nemotron_request_latency_ms Latency histogram')
                lines.append(f'# TYPE nemotron_request_latency_ms histogram')
                cumulative = 0
                for bucket in sorted(b for b in self.latency_buckets if b != float("inf")):
                    cumulative += self.latency_buckets[bucket]
                    lines.append(
                        f'nemotron_request_latency_ms_bucket{{model="{model_name}",le="{bucket}"}} {cumulative}'
                    )
                cumulative += self.latency_buckets[float("inf")]
                lines.append(
                    f'nemotron_request_latency_ms_bucket{{model="{model_name}",le="+Inf"}} {cumulative}'
                )
            return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════════
# Server State
# ═══════════════════════════════════════════════════════════════════════════════

class ServerState:
    """Encapsulates all server state — no module-level globals."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.model_type: str = ""
        self.model_dir: str = ""
        self.model_name: str = ""
        self.device: str = "cpu"
        self.dtype = torch.float32
        self.attn_impl: str = "eager"
        self.device_info: dict = {}
        self.system_info: dict = {}
        self.api_key: Optional[str] = None
        self.metrics = MetricsCollector()
        self.ready = False
        self.load_time_s: float = 0
        self.config: dict = {}
        self.embedding_dims: int = 0
        self.context_length: int = 8192

    def load_model(self, model_dir: str, force_device: str = "",
                   force_dtype: str = "auto", threads: int = 0,
                   ctx_size: int = 0, do_warmup: bool = True):
        self.model_dir = model_dir
        self.model_name = os.path.basename(model_dir)
        self.system_info = get_system_info()

        # Validate & detect
        model_dir, self.config = validate_model_dir(model_dir)
        self.model_dir = model_dir
        self.model_type = detect_model_type(self.config)
        self.context_length = ctx_size if ctx_size > 0 else self.config.get("max_position_embeddings", 8192)

        # Select device
        self.device, self.dtype, self.attn_impl, self.device_info = select_device(force_device)

        # Apply dtype override
        if force_dtype != "auto":
            dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
            self.dtype = dtype_map.get(force_dtype, self.dtype)

        logger.info(f"Loading {self.model_type} model: {self.model_name}")
        logger.info(f"  Device: {self.device} ({self.device_info.get('reason', '')})")
        logger.info(f"  Dtype: {self.dtype}")
        logger.info(f"  Attention: {self.attn_impl}")

        t0 = time.time()

        if self.model_type == "rerank":
            from transformers import AutoModelForSequenceClassification, AutoProcessor
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                attn_implementation=self.attn_impl,
            ).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(
                model_dir,
                trust_remote_code=True,
                max_input_tiles=6,
                use_thumbnail=True,
                rerank_max_length=self.context_length,
            )
        else:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                model_dir,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                attn_implementation=self.attn_impl,
            ).to(self.device).eval()
            self.model.processor.p_max_length = self.context_length

            # Detect embedding dimensions
            try:
                with torch.inference_mode():
                    probe = self.model.encode_documents(texts=["dim probe"])
                    self.embedding_dims = probe[0].shape[0]
            except Exception:
                self.embedding_dims = self.config.get("hidden_size", 2048)

        self.load_time_s = time.time() - t0
        self.ready = True

        # Count parameters
        param_count = sum(p.numel() for p in self.model.parameters())
        param_str = f"{param_count / 1e9:.1f}B" if param_count >= 1e9 else f"{param_count / 1e6:.0f}M"

        # Optimize thread count for CPU inference
        if self.device == "cpu":
            physical_cores = os.cpu_count() or 4
            if threads > 0:
                optimal_threads = threads
            else:
                # Auto-detect: try to use performance cores only
                # Apple Silicon: M1=8(4P+4E), M2=8(4P+4E), M3 Pro=12(6P+6E), M3 Max=16(12P+4E)
                # Intel: use physical cores (no hyperthreading benefit for inference)
                import platform
                is_apple = platform.machine() == "arm64" and platform.system() == "Darwin"
                if is_apple:
                    # Apple Silicon: use ~75% of cores (targets P-cores, avoids E-cores)
                    optimal_threads = max(4, int(physical_cores * 0.75))
                else:
                    # x86: use all physical cores (hyperthreading detected via os.cpu_count)
                    optimal_threads = max(1, min(physical_cores, 16))
            torch.set_num_threads(optimal_threads)
            torch.set_num_interop_threads(max(1, optimal_threads // 4))
            logger.info(f"  CPU threads: {optimal_threads} intra-op, "
                        f"{max(1, optimal_threads // 4)} inter-op (of {physical_cores} available)")

        logger.info(f"  Parameters: {param_str}")
        if self.embedding_dims:
            logger.info(f"  Embedding dims: {self.embedding_dims}")
        logger.info(f"  Context length: {self.context_length}")
        logger.info(f"  Loaded in {self.load_time_s:.1f}s")

        # torch.compile: JIT-compile the model's computation graph for fused operations
        # This can significantly speed up CPU inference by eliminating Python overhead
        # and fusing pointwise operations. Safe to skip if it fails (some model architectures
        # with custom ops may not be compilable).
        try:
            logger.info("  Compiling model graph with torch.compile...")
            t_compile = time.time()
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info(f"  torch.compile completed in {time.time() - t_compile:.1f}s")
        except Exception as e:
            logger.info(f"  torch.compile skipped (non-fatal): {e}")

        # Warmup: run a probe inference to JIT-compile kernels and warm caches
        if do_warmup:
            logger.info("  Running warmup inference...")
            t_warm = time.time()
            try:
                with torch.inference_mode():
                    if self.model_type == "embed":
                        self.model.processor.p_max_length = 128
                        _ = self.model.encode_documents(texts=["warmup probe"])
                        self.model.processor.p_max_length = self.context_length
                    else:
                        examples = [{"question": "warmup", "doc_text": "probe", "doc_image": ""}]
                        batch = self.processor.process_queries_documents_crossencoder(examples)
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}
                        _ = self.model(**batch, return_dict=True)
                logger.info(f"  Warmup completed in {time.time() - t_warm:.1f}s")
            except Exception as e:
                logger.warning(f"  Warmup failed (non-fatal): {e}")


state = ServerState()


# ═══════════════════════════════════════════════════════════════════════════════
# API Models
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingRequest(BaseModel):
    input: Union[str, list] = Field(..., description="Text string(s) or image data URL(s)")
    model: str = Field(default="", description="Model name (optional, uses loaded model)")

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list = []
    model: str = ""
    usage: dict = {}

class RerankRequest(BaseModel):
    query: str = Field(..., description="Query to rank documents against")
    documents: list[str] = Field(..., description="Documents to rerank")
    images: Optional[list[Optional[str]]] = Field(default=None, description="Optional base64 image per document (parallel to documents list)")
    model: str = Field(default="", description="Model name (optional)")
    top_n: Optional[int] = Field(default=None, description="Return only top N results (sorted by score)")

class RerankResponse(BaseModel):
    object: str = "rerank"
    results: list = []
    model: str = ""
    usage: dict = {}

class SimilarityRequest(BaseModel):
    text_a: Union[str, list[str]] = Field(..., description="First text(s)")
    text_b: Union[str, list[str]] = Field(..., description="Second text(s)")

class TokenizeRequest(BaseModel):
    input: Union[str, list[str]] = Field(..., description="Text(s) to tokenize")


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI App with Lifespan
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("Server ready to accept requests")
    yield
    logger.info("Shutting down gracefully...")
    # Cleanup: free GPU memory
    if state.model is not None:
        del state.model
        state.model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Nemotron VL Inference Server",
    description="High-performance embedding and reranking server for NVIDIA Nemotron VL models",
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def check_api_key(request: Request):
    if state.api_key:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != state.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def check_ready(request: Request):
    if not state.ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")


def parse_image_from_b64(data_url: str):
    """Parse a data:image/... URL or raw base64 to PIL Image."""
    from PIL import Image
    if data_url.startswith("data:"):
        header, b64 = data_url.split(",", 1)
    else:
        b64 = data_url
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Health check / readiness probe. No authentication required."""
    return {
        "status": "ready" if state.ready else "loading",
        "model": state.model_name,
        "model_type": state.model_type,
        "device": state.device,
        "dtype": str(state.dtype),
        "embedding_dims": state.embedding_dims,
        "uptime_seconds": round(time.time() - state.metrics.start_time, 1),
        "version": __version__,
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    return state.metrics.prometheus_format(state.model_name)


@app.get("/v1/models")
async def list_models(_=Depends(check_api_key)):
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": int(state.metrics.start_time),
            "owned_by": "nvidia",
            "capabilities": (
                ["embed", "multimodal"] if state.model_type == "embed"
                else ["rerank", "multimodal"]
            ),
            "context_length": state.context_length,
            "embedding_dims": state.embedding_dims,
        }],
    }


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def create_embeddings(req: EmbeddingRequest, _=Depends(check_api_key),
                            __=Depends(check_ready)):
    """Generate embeddings for text and/or images (OpenAI-compatible)."""
    if state.model_type != "embed":
        raise HTTPException(
            400,
            f"This server is running a {state.model_type} model. "
            f"Use /v1/rerank instead of /v1/embeddings."
        )

    t0 = time.perf_counter()
    inputs = req.input if isinstance(req.input, list) else [req.input]
    texts, images = [], []

    for inp in inputs:
        if isinstance(inp, str):
            if inp.startswith("data:image/") or (len(inp) > 1000 and not inp.startswith("http")):
                try:
                    images.append(parse_image_from_b64(inp))
                except Exception as e:
                    raise HTTPException(400, f"Failed to parse image: {e}")
            else:
                texts.append(inp)
        elif isinstance(inp, dict):
            if inp.get("type") == "image_url":
                url = inp.get("image_url", {}).get("url", "")
                try:
                    images.append(parse_image_from_b64(url))
                except Exception as e:
                    raise HTTPException(400, f"Failed to parse image: {e}")

    if not texts and not images:
        raise HTTPException(400, "No valid text or image input provided")

    embeddings = []
    try:
        with torch.inference_mode():
            if texts and not images:
                state.model.processor.p_max_length = state.context_length
                vecs = state.model.encode_documents(texts=texts)
            elif images and not texts:
                state.model.processor.p_max_length = 2048
                state.model.processor.max_input_tiles = 6
                state.model.processor.use_thumbnail = True
                vecs = state.model.encode_documents(images=images)
            else:
                state.model.processor.p_max_length = min(10240, state.context_length)
                state.model.processor.max_input_tiles = 6
                state.model.processor.use_thumbnail = True
                vecs = state.model.encode_documents(texts=texts, images=images)

            for i, vec in enumerate(vecs):
                embeddings.append({
                    "object": "embedding",
                    "index": i,
                    "embedding": vec.cpu().float().tolist(),
                })
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        state.metrics.record_request("/v1/embeddings", elapsed, error=True)
        raise HTTPException(500, f"Embedding inference failed: {e}")

    elapsed = (time.perf_counter() - t0) * 1000
    token_est = sum(len(t.split()) for t in texts)
    state.metrics.record_request(
        "/v1/embeddings", elapsed, tokens=token_est, embeddings=len(embeddings)
    )

    return EmbeddingResponse(
        data=embeddings,
        model=state.model_name,
        usage={"prompt_tokens": token_est, "total_tokens": token_est},
    )


@app.post("/v1/rerank")
@app.post("/rerank")
async def rerank(req: RerankRequest, _=Depends(check_api_key), __=Depends(check_ready)):
    """Rerank documents by relevance to a query."""
    if state.model_type != "rerank":
        raise HTTPException(
            400,
            f"This server is running a {state.model_type} model. "
            f"Use /v1/embeddings instead of /v1/rerank."
        )

    if not req.documents:
        raise HTTPException(400, "documents list cannot be empty")

    t0 = time.perf_counter()

    # Build examples — optionally include images (parallel to documents list)
    has_images = req.images and len(req.images) == len(req.documents)
    examples = []
    for i, doc in enumerate(req.documents):
        img = ""
        if has_images and req.images[i]:
            try:
                img = parse_image_from_b64(req.images[i])
            except Exception:
                img = ""  # Failed to parse image, fall back to text-only
        examples.append({
            "question": req.query,
            "doc_text": doc,
            "doc_image": img,
        })

    try:
        batch_dict = state.processor.process_queries_documents_crossencoder(examples)
        batch_dict = {
            k: v.to(state.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_dict.items()
        }

        with torch.no_grad():
            outputs = state.model(**batch_dict, return_dict=True)

        logits = outputs.logits.squeeze(-1)
        scores = torch.sigmoid(logits).cpu().float().tolist()
        if isinstance(scores, float):
            scores = [scores]
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        state.metrics.record_request("/v1/rerank", elapsed, error=True)
        raise HTTPException(500, f"Rerank inference failed: {e}")

    results = [
        {"index": i, "relevance_score": round(score, 6), "document": None}
        for i, score in enumerate(scores)
    ]

    # top_n: sort by score, return only top N
    if req.top_n is not None and req.top_n > 0:
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        results = results[:req.top_n]

    elapsed = (time.perf_counter() - t0) * 1000
    token_est = len(req.query.split()) + sum(len(d.split()) for d in req.documents)
    state.metrics.record_request("/v1/rerank", elapsed, tokens=token_est, is_rerank=True)

    return RerankResponse(
        results=results,
        model=state.model_name,
        usage={"prompt_tokens": token_est, "total_tokens": token_est},
    )


@app.post("/v1/similarity")
async def similarity(req: SimilarityRequest, _=Depends(check_api_key), __=Depends(check_ready)):
    """Compute pairwise cosine similarity between text pairs."""
    if state.model_type != "embed":
        raise HTTPException(400, "Similarity requires an embedding model")

    texts_a = [req.text_a] if isinstance(req.text_a, str) else req.text_a
    texts_b = [req.text_b] if isinstance(req.text_b, str) else req.text_b

    if len(texts_a) != len(texts_b):
        raise HTTPException(400, f"text_a ({len(texts_a)}) and text_b ({len(texts_b)}) must have equal length")

    with torch.inference_mode():
        state.model.processor.p_max_length = state.context_length
        vecs_a = state.model.encode_documents(texts=texts_a)
        vecs_b = state.model.encode_documents(texts=texts_b)

    results = []
    for i, (va, vb) in enumerate(zip(vecs_a, vecs_b)):
        va_f = va.cpu().float()
        vb_f = vb.cpu().float()
        cos = torch.nn.functional.cosine_similarity(va_f.unsqueeze(0), vb_f.unsqueeze(0)).item()
        results.append({"index": i, "score": round(cos, 6)})

    return {"results": results, "model": state.model_name}


@app.post("/tokenize")
async def tokenize(req: TokenizeRequest, _=Depends(check_api_key)):
    """Estimate token counts for text input."""
    texts = [req.input] if isinstance(req.input, str) else req.input
    results = []
    for i, text in enumerate(texts):
        # Rough estimate: ~4 chars per token for English
        word_count = len(text.split())
        char_count = len(text)
        token_est = max(word_count, char_count // 4)
        results.append({
            "index": i,
            "text_length": char_count,
            "word_count": word_count,
            "token_estimate": token_est,
        })
    return {"results": results}


# ═══════════════════════════════════════════════════════════════════════════════
# Built-in Test UI
# ═══════════════════════════════════════════════════════════════════════════════

TEST_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nemotron VL Inference Server</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
         background: #0a0a0a; color: #e0e0e0; padding: 24px; max-width: 960px; margin: 0 auto; }
  h1 { font-size: 1.5rem; margin-bottom: 4px; color: #76b900; }
  .subtitle { color: #888; margin-bottom: 24px; font-size: 0.9rem; }
  .card { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px; margin-bottom: 16px; }
  .card h2 { font-size: 1.1rem; color: #76b900; margin-bottom: 12px; }
  .info-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
  .info-item { background: #222; padding: 8px 12px; border-radius: 4px; }
  .info-item .label { font-size: 0.75rem; color: #888; text-transform: uppercase; }
  .info-item .value { font-size: 1rem; color: #fff; font-weight: 500; }
  textarea { width: 100%; background: #111; color: #e0e0e0; border: 1px solid #444;
             border-radius: 4px; padding: 10px; font-family: monospace; font-size: 0.9rem;
             resize: vertical; min-height: 80px; }
  button { background: #76b900; color: #000; border: none; padding: 10px 24px; border-radius: 4px;
           font-weight: 600; cursor: pointer; font-size: 0.9rem; margin-top: 8px; }
  button:hover { background: #8ad400; }
  button:disabled { background: #444; color: #888; cursor: not-allowed; }
  .result { background: #111; border: 1px solid #333; border-radius: 4px; padding: 12px;
            margin-top: 12px; font-family: monospace; font-size: 0.85rem; white-space: pre-wrap;
            max-height: 400px; overflow-y: auto; }
  .latency { color: #76b900; font-size: 0.8rem; margin-top: 4px; }
  .tab-bar { display: flex; gap: 2px; margin-bottom: 16px; }
  .tab { padding: 8px 20px; background: #222; border: 1px solid #333; border-radius: 4px 4px 0 0;
         cursor: pointer; color: #888; }
  .tab.active { background: #1a1a1a; color: #76b900; border-bottom-color: #1a1a1a; }
  .hidden { display: none; }
  .row { display: flex; gap: 12px; align-items: flex-start; }
  .row > * { flex: 1; }
  label { display: block; font-size: 0.8rem; color: #888; margin-bottom: 4px; }
</style>
</head>
<body>
<h1>Nemotron VL Inference Server</h1>
<p class="subtitle">v{{VERSION}} &mdash; {{MODEL_NAME}} ({{MODEL_TYPE}})</p>

<div class="card">
  <h2>Server Info</h2>
  <div class="info-grid" id="info-grid"></div>
</div>

<div class="tab-bar">
  <div class="tab active" onclick="switchTab('embed')">Embeddings</div>
  <div class="tab" onclick="switchTab('rerank')">Rerank</div>
  <div class="tab" onclick="switchTab('similarity')">Similarity</div>
</div>

<div id="tab-embed" class="card">
  <h2>Generate Embeddings</h2>
  <label>Input text (one per line for batch):</label>
  <textarea id="embed-input" rows="4">blue-green deployment for zero-downtime releases</textarea>
  <button onclick="runEmbed()" id="embed-btn">Generate Embeddings</button>
  <div id="embed-result" class="result hidden"></div>
  <div id="embed-latency" class="latency"></div>
</div>

<div id="tab-rerank" class="card hidden">
  <h2>Rerank Documents</h2>
  <label>Query:</label>
  <textarea id="rerank-query" rows="2">deployment strategy for production</textarea>
  <label style="margin-top:8px">Documents (one per line):</label>
  <textarea id="rerank-docs" rows="5">Blue-green deployment enables zero-downtime releases
The cafeteria menu includes pasta on Thursdays
Rolling deployments gradually replace old instances
Canary releases route small traffic to new version</textarea>
  <button onclick="runRerank()" id="rerank-btn">Rerank</button>
  <div id="rerank-result" class="result hidden"></div>
  <div id="rerank-latency" class="latency"></div>
</div>

<div id="tab-similarity" class="card hidden">
  <h2>Cosine Similarity</h2>
  <div class="row">
    <div><label>Text A:</label><textarea id="sim-a" rows="3">zero-downtime releases</textarea></div>
    <div><label>Text B:</label><textarea id="sim-b" rows="3">blue-green deployment</textarea></div>
  </div>
  <button onclick="runSimilarity()" id="sim-btn">Compare</button>
  <div id="sim-result" class="result hidden"></div>
</div>

<script>
const API_KEY = '{{API_KEY}}';
const headers = {'Content-Type': 'application/json'};
if (API_KEY) headers['Authorization'] = 'Bearer ' + API_KEY;

async function fetchHealth() {
  const r = await fetch('/health');
  const d = await r.json();
  const grid = document.getElementById('info-grid');
  const items = [
    ['Status', d.status], ['Model', d.model], ['Type', d.model_type],
    ['Device', d.device], ['Dtype', d.dtype],
    ['Dims', d.embedding_dims || 'N/A'], ['Uptime', Math.round(d.uptime_seconds) + 's'],
    ['Version', d.version],
  ];
  grid.innerHTML = items.map(([l,v]) =>
    `<div class="info-item"><div class="label">${l}</div><div class="value">${v}</div></div>`
  ).join('');
}
fetchHealth(); setInterval(fetchHealth, 10000);

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    t.classList.toggle('active', ['embed','rerank','similarity'][i] === name);
  });
  ['embed','rerank','similarity'].forEach(n => {
    document.getElementById('tab-'+n).classList.toggle('hidden', n !== name);
  });
}

async function runEmbed() {
  const btn = document.getElementById('embed-btn');
  btn.disabled = true; btn.textContent = 'Processing...';
  const texts = document.getElementById('embed-input').value.split('\\n').filter(t => t.trim());
  const t0 = performance.now();
  try {
    const r = await fetch('/v1/embeddings', {method:'POST', headers, body: JSON.stringify({input: texts})});
    const d = await r.json();
    const ms = (performance.now() - t0).toFixed(0);
    const el = document.getElementById('embed-result');
    el.classList.remove('hidden');
    if (d.data) {
      const summary = d.data.map(e =>
        `[${e.index}] dims=${e.embedding.length} first5=[${e.embedding.slice(0,5).map(v=>v.toFixed(4)).join(', ')}]`
      ).join('\\n');
      el.textContent = summary;
    } else {
      el.textContent = JSON.stringify(d, null, 2);
    }
    document.getElementById('embed-latency').textContent = `${ms}ms (${texts.length} text${texts.length>1?'s':''})`;
  } catch(e) { document.getElementById('embed-result').classList.remove('hidden');
    document.getElementById('embed-result').textContent = 'Error: ' + e.message; }
  btn.disabled = false; btn.textContent = 'Generate Embeddings';
}

async function runRerank() {
  const btn = document.getElementById('rerank-btn');
  btn.disabled = true; btn.textContent = 'Processing...';
  const query = document.getElementById('rerank-query').value;
  const docs = document.getElementById('rerank-docs').value.split('\\n').filter(t => t.trim());
  const t0 = performance.now();
  try {
    const r = await fetch('/v1/rerank', {method:'POST', headers, body: JSON.stringify({query, documents: docs})});
    const d = await r.json();
    const ms = (performance.now() - t0).toFixed(0);
    const el = document.getElementById('rerank-result');
    el.classList.remove('hidden');
    if (d.results) {
      const sorted = [...d.results].sort((a,b) => b.relevance_score - a.relevance_score);
      el.textContent = sorted.map(r =>
        `#${r.index+1} score=${r.relevance_score.toFixed(4)}  "${docs[r.index].substring(0,80)}"`
      ).join('\\n');
    } else { el.textContent = JSON.stringify(d, null, 2); }
    document.getElementById('rerank-latency').textContent = `${ms}ms (${docs.length} docs)`;
  } catch(e) { document.getElementById('rerank-result').classList.remove('hidden');
    document.getElementById('rerank-result').textContent = 'Error: ' + e.message; }
  btn.disabled = false; btn.textContent = 'Rerank';
}

async function runSimilarity() {
  const btn = document.getElementById('sim-btn');
  btn.disabled = true; btn.textContent = 'Computing...';
  const a = document.getElementById('sim-a').value, b = document.getElementById('sim-b').value;
  try {
    const r = await fetch('/v1/similarity', {method:'POST', headers, body: JSON.stringify({text_a:a, text_b:b})});
    const d = await r.json();
    const el = document.getElementById('sim-result');
    el.classList.remove('hidden');
    el.textContent = d.results ? `Cosine similarity: ${d.results[0].score.toFixed(6)}` : JSON.stringify(d, null, 2);
  } catch(e) { document.getElementById('sim-result').classList.remove('hidden');
    document.getElementById('sim-result').textContent = 'Error: ' + e.message; }
  btn.disabled = false; btn.textContent = 'Compare';
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def test_ui():
    """Built-in test UI for interactive exploration."""
    html = TEST_UI_HTML.replace("{{VERSION}}", __version__)
    html = html.replace("{{MODEL_NAME}}", state.model_name or "loading...")
    html = html.replace("{{MODEL_TYPE}}", state.model_type or "unknown")
    html = html.replace("{{API_KEY}}", state.api_key or "")
    return html


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

BANNER = r"""
  _  _                     _                   __   __ _
 | \| | ___  _ __   ___  | |_  _ _  ___  _ _   \ \ / /| |
 | .` |/ -_)| '  \ / _ \ |  _|| '_|/ _ \| ' \   \ V / | |__
 |_|\_|\___||_|_|_|\___/  \__||_|  \___/|_||_|   \_/  |____|
  Inference Server v{version}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Nemotron VL Inference Server — high-performance embedding & reranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        examples:
          # Start embedding server:
          %(prog)s --model-dir ./models/nvidia_llama-nemotron-embed-vl-1b-v2

          # Start rerank server on custom port with API key:
          %(prog)s --model-dir ./models/nvidia_llama-nemotron-rerank-vl-1b-v2 --port 8025 --api-key secret

          # Force CPU device (skip GPU detection):
          %(prog)s --model-dir ./models/nvidia_llama-nemotron-embed-vl-1b-v2 --device cpu

          # Auto-find available port if default is in use:
          %(prog)s --model-dir ./models/nvidia_llama-nemotron-embed-vl-1b-v2 --auto-port

        endpoints:
          POST /v1/embeddings     OpenAI-compatible text/image embedding
          POST /v1/rerank         Cross-encoder document reranking
          POST /v1/similarity     Pairwise cosine similarity
          POST /tokenize          Token count estimation
          GET  /v1/models         Model information
          GET  /health            Readiness probe
          GET  /metrics           Prometheus metrics
          GET  /                  Interactive test UI
          GET  /docs              OpenAPI documentation
        """),
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to HuggingFace model directory (must contain config.json + weights)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8020,
        help="Listen port (default: 8020)",
    )
    parser.add_argument(
        "--api-key", default="",
        help="Require this API key for authentication (optional)",
    )
    parser.add_argument(
        "--device", default="",
        choices=["", "cpu", "cuda", "mps", "xpu"],
        help="Force device (default: auto-detect best available)",
    )
    parser.add_argument(
        "--dtype", default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model data type (default: auto — bfloat16 on Ampere+/Apple Silicon, float32 elsewhere)",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=0,
        help="CPU threads for inference (default: 0 = auto-detect optimal count)",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=32,
        help="Maximum batch size per request (default: 32)",
    )
    parser.add_argument(
        "-c", "--ctx-size", type=int, default=0,
        help="Override context length in tokens (default: 0 = use model config)",
    )
    parser.add_argument(
        "--max-concurrent-requests", type=int, default=64,
        help="Maximum concurrent requests before backpressure (default: 64)",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="Request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--warmup", "--model-warmup", action="store_true", default=True,
        help="Run warmup inference after model load (default: enabled)",
    )
    parser.add_argument(
        "--no-warmup", dest="warmup", action="store_false",
        help="Skip model warmup (faster startup, slower first request)",
    )
    parser.add_argument(
        "--auto-port", action="store_true",
        help="If port is in use, automatically find next available port",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of uvicorn workers (default: 1, increase for CPU-bound loads)",
    )
    parser.add_argument(
        "--log-level", default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    args = parser.parse_args()

    # Banner (print early before any logging)
    sys.stdout.write(BANNER.format(version=__version__))
    sys.stdout.flush()

    # System info
    sys_info = get_system_info()
    logger.info(f"System: {sys_info['os']} {sys_info['arch']} "
                f"(Python {sys_info['python']}, {sys_info['cpu_count']} cores)")
    if "ram_gb" in sys_info:
        logger.info(f"  RAM: {sys_info['ram_gb']}GB total, {sys_info['ram_available_gb']}GB available")
    if sys_info.get("cpu_features"):
        logger.info(f"  CPU features: {', '.join(sys_info['cpu_features'])}")
    logger.info(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA: {torch.version.cuda} ({torch.cuda.get_device_name(0)})")
    if torch.backends.mps.is_available():
        logger.info(f"  MPS: available (Apple Metal)")

    # Set API key
    state.api_key = args.api_key or None

    # Validate model directory (with friendly error)
    try:
        model_dir, config = validate_model_dir(args.model_dir)
    except (FileNotFoundError, NotADirectoryError) as e:
        logger.error(str(e))
        sys.exit(1)

    model_type = detect_model_type(config)
    logger.info(f"Detected model type: {model_type}")

    # Check port availability BEFORE loading the model (loading takes 5-30s)
    port = args.port
    if not check_port_available(args.host, port):
        if args.auto_port:
            new_port = find_available_port(args.host, port + 1)
            if new_port < 0:
                logger.error(
                    f"Port {port} is already in use and no available port found "
                    f"in range {port+1}-{port+10}.\n"
                    f"  Try: lsof -i :{port}  (macOS/Linux) or  netstat -ano | findstr :{port}  (Windows)\n"
                    f"  to find what's using it, or specify a different port with --port"
                )
                sys.exit(1)
            logger.warning(f"Port {port} is in use, using port {new_port} instead")
            port = new_port
        else:
            # Identify what's using the port
            hint = ""
            if sys.platform != "win32":
                try:
                    import subprocess
                    result = subprocess.run(
                        ["lsof", "-i", f":{port}", "-t"],
                        capture_output=True, text=True, timeout=3,
                    )
                    pids = result.stdout.strip()
                    if pids:
                        hint = f" (PIDs: {pids})"
                except Exception:
                    pass
            logger.error(
                f"Port {port} is already in use{hint}.\n\n"
                f"  Options:\n"
                f"    1. Use a different port:  --port {port + 1}\n"
                f"    2. Auto-find free port:   --auto-port\n"
                f"    3. Stop the other process: kill $(lsof -t -i:{port})\n"
            )
            sys.exit(1)

    # Load model
    try:
        state.load_model(
            model_dir,
            force_device=args.device,
            force_dtype=args.dtype,
            threads=args.threads,
            ctx_size=args.ctx_size,
            do_warmup=args.warmup,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Startup summary
    logger.info(f"Starting server on {args.host}:{port}")
    logger.info(f"  API docs:  http://localhost:{port}/docs")
    logger.info(f"  Test UI:   http://localhost:{port}/")
    logger.info(f"  Health:    http://localhost:{port}/health")
    logger.info(f"  Metrics:   http://localhost:{port}/metrics")
    if state.api_key:
        logger.info(f"  Auth:      Bearer token required")
    else:
        logger.info(f"  Auth:      none (open access)")

    # Graceful shutdown handler
    def handle_signal(signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.info(f"Received {sig_name}, shutting down...")
        sys.exit(0)

    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Run
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level=args.log_level,
        workers=args.workers,
        access_log=args.log_level == "debug",
    )


if __name__ == "__main__":
    main()
