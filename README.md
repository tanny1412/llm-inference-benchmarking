# LLM Inference Benchmarking

An end-to-end LLM inference benchmarking project comparing naive HuggingFace serving, vLLM with PagedAttention and continuous batching, and AWQ 4-bit quantization — measuring throughput, latency, and GPU memory across concurrency levels.

## Project Story

Each stage isolates one bottleneck and fixes it:

| Stage | Server | Precision | What it shows |
|-------|--------|-----------|---------------|
| 1 | Naive HuggingFace + FastAPI | FP16 | Baseline — naive serving limitations, sequential request handling |
| 2 | vLLM | FP16 | Runtime optimization — continuous batching, PagedAttention, concurrency scaling |
| 3 | vLLM + AWQ | 4-bit | Memory optimization — quantization effect on VRAM, throughput, quality |

## Model

`mistralai/Mistral-7B-Instruct-v0.1` — decoder-only transformer with GQA and SWA, instruction-tuned

## Metrics

| Metric | What it measures |
|--------|-----------------|
| Latency p50 (s) | Typical user wait time |
| Latency p99 (s) | Worst-case user wait time |
| Tokens/sec | Model generation speed |
| Requests/sec | Serving throughput |
| GPU memory (MiB) | VRAM usage |

## Benchmark Matrix

| Backend | Precision | Concurrency |
|---------|-----------|-------------|
| HF naive | FP16 | 1, 5, 10 |
| vLLM | FP16 | 1, 10, 50, 100 |
| vLLM + AWQ | 4-bit | 1, 10, 50, 100 |

## File Structure

```
app_hf.py        # Stage 1 — naive HuggingFace + FastAPI server
app_vllm.py      # Stage 2 — vLLM server
app_awq.py       # Stage 3 — vLLM + AWQ quantized server
benchmark.py     # benchmark harness (same for all stages)
requirements.txt
devlogs.md       # detailed dev notes, bugs, observations
```

## Setup (RunPod)

```bash
git clone https://github.com/tanny1412/llm-inference-benchmarking.git
cd llm-inference-benchmarking
pip install -r requirements.txt
export HF_HOME=/workspace/hf-cache
```

## Running Each Stage

**Stage 1 — HF Baseline:**
```bash
uvicorn app_hf:app --host 0.0.0.0 --port 8000
```

**Stage 2 — vLLM:**
```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.1 --host 0.0.0.0 --port 8000 --tokenizer-mode mistral
```

**Stage 3 — AWQ:**
```bash
uvicorn app_awq:app --host 0.0.0.0 --port 8000
```

## Running Benchmarks

```bash
python benchmark.py --backend hf --concurrency 1 --num_requests 20 --max_new_tokens 200
python benchmark.py --backend vllm --concurrency 10 --num_requests 50 --max_new_tokens 200
python benchmark.py --backend awq --concurrency 10 --num_requests 50 --max_new_tokens 200
```

Results saved to `results_<backend>_c<concurrency>_<timestamp>.json`

## Results

### Stage 1 vs Stage 2

| Backend | Concurrency | p50 latency | Tokens/sec | Req/sec |
|---------|-------------|-------------|------------|---------|
| HF naive | 1 | 5.148s | 32.55 | 0.19 |
| HF naive | 5 | 25.792s | 32.76 | 0.19 |
| vLLM | 1 | 3.304s | 49.60 | 0.30 |
| vLLM | 10 | 3.611s | 444.60 | 2.76 |
| vLLM | 50 | 4.747s | 1714.23 | 10.56 |
| vLLM | 100 | 6.241s | 2596.58 | 16.01 |

**80x throughput improvement** at concurrency=100. HF flat-lines; vLLM scales. p50 latency at 100 concurrent users (6.2s) is only 20% worse than HF at 1 user (5.1s).

## Key Concepts

- **Continuous batching** — vLLM fills freed batch slots immediately, keeping GPU saturated
- **PagedAttention** — KV cache stored in fixed-size pages, eliminates fragmentation
- **AWQ** — activation-aware weight quantization, 4-bit weights with minimal quality loss
- **KV cache** — stores previous K/V vectors so decode steps don't recompute attention
- **Decode is memory-bandwidth bound** — GPU-Util looks low but HBM is busy streaming KV cache
