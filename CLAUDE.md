# LLM Inference Benchmarking — Project Guide

## Project Goal

Benchmark LLM inference across four stages, isolating one bottleneck per stage:
1. Naive HuggingFace serving (baseline)
2. vLLM (runtime/serving optimization)
3. vLLM + AWQ 4-bit quantization (memory optimization)
4. vLLM + GPTQ 4-bit quantization (alternative quantization — AWQ vs GPTQ comparison)

Model: `mistralai/Mistral-7B-Instruct-v0.1` on RunPod RTX 4090 (24GB VRAM)

## Collaboration Rules

- NEVER generate large amounts of code independently — ask questions first, build one piece at a time
- Always ensure the user understands WHY before moving to the next piece
- NEVER include "Co-Authored-By: Claude" or any mention of Claude in git commits

## File Structure

```
app_hf.py        # Stage 1 — naive HuggingFace + FastAPI
app_vllm.py      # Stage 2 — vLLM server
app_gptq.py      # Stage 4 — vLLM + GPTQ quantized server
benchmark.py     # concurrent benchmark harness using asyncio + aiohttp
requirements.txt
devlogs.md       # running dev log — bugs, decisions, observations
README.md        # project overview
```

## Current Status

- [x] Stage 1 server (`app_hf.py`) — HF model loaded, FastAPI endpoint working, threading.Lock added
- [x] Benchmark harness (`benchmark.py`) — asyncio + aiohttp, p50/p99, tokens/sec, req/sec, GPU memory
- [x] Stage 2 — vLLM via `vllm serve`, benchmarked at c=1,10,50,100
- [x] Stage 3 — AWQ via `vllm serve`, benchmarked at c=1,10,50,100
- [x] Results tables
- [ ] Stage 4 — GPTQ (`app_gptq.py`), benchmark at c=1,10,50,100

## Key Design Decisions

- Separate `app_hf.py`, `app_vllm.py`, `app_gptq.py` — never overwrite, always compare
- `benchmark.py` is backend-agnostic — same script for all three stages
- `max_new_tokens` is configurable and must stay constant across benchmark runs
- Results saved to JSON with backend + concurrency in filename
- HuggingFace cache at `/workspace/hf-cache` on RunPod network volume

## RunPod Setup (every pod restart)

**Stage 1 — HF:**
```bash
cd /workspace/llm-inference-benchmarking
git pull
export HF_HOME=/workspace/hf-cache
uvicorn app_hf:app --host 0.0.0.0 --port 8000
```

**Stage 2 — vLLM:**
```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.1 --host 0.0.0.0 --port 8000 --tokenizer-mode mistral --download-dir /workspace/hf-cache/hub
```

**Stage 3 — AWQ:**
```bash
vllm serve TheBloke/Mistral-7B-Instruct-v0.1-AWQ --host 0.0.0.0 --port 8000 --quantization awq --dtype float16 --download-dir /workspace/hf-cache/hub
```

**Stage 4 — GPTQ:**
```bash
vllm serve TheBloke/Mistral-7B-Instruct-v0.1-GPTQ --host 0.0.0.0 --port 8000 --quantization gptq --dtype float16 --download-dir /workspace/hf-cache/hub
```

Note: always use `--download-dir /workspace/hf-cache/hub` (not `/workspace/hf-cache`) — HF puts models in a `hub/` subdirectory, vLLM doesn't know this unless told explicitly.

SSH config (update IP/port after each restart):
```
Host runpod
    HostName <pod-ip>
    User root
    Port <pod-port>
    IdentityFile ~/.ssh/id_ed25519
```

## Benchmark Command

```bash
python benchmark.py --backend hf --concurrency 1 --num_requests 20 --max_new_tokens 200
```

## Metrics

- `latency_p50_s` — median request latency
- `latency_p99_s` — 99th percentile latency (worst-case user experience)
- `tokens_per_sec` — total tokens generated / total wall time
- `requests_per_sec` — requests completed / total wall time
