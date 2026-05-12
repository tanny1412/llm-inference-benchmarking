# LLM Inference Benchmarking

An end-to-end LLM inference benchmarking project that isolates one bottleneck per stage — from naive HuggingFace serving to optimized quantized kernels — measuring throughput, latency, and GPU memory across concurrency levels on a single RTX 4090.

## The Story

Five stages. Each one asks: what is the bottleneck right now, and what fixes it?

| Stage | Backend | Precision | The question it answers |
|-------|---------|-----------|------------------------|
| 1 | Naive HuggingFace + FastAPI | FP16 | What does naive serving actually cost? |
| 2 | vLLM | FP16 | What does a real inference engine buy you? |
| 3 | vLLM + AWQ | 4-bit | What does quantization buy you? |
| 4 | vLLM + GPTQ | 4-bit | Does the quantization algorithm matter? |
| 5 | vLLM + GPTQ Marlin | 4-bit | Does the kernel implementation matter? |

The final answer: **kernel implementation quality can dominate quantization algorithm choice.**

## Model

`mistralai/Mistral-7B-Instruct-v0.1` — 7B parameter decoder-only transformer, instruction-tuned. FP16 = 14GB VRAM. 4-bit quantized = ~4GB VRAM.

Hardware: RunPod RTX 4090 (24GB VRAM)

---

## Stage 1 — Naive HuggingFace Serving

Load the model with `transformers`, serve with FastAPI. No batching, no optimization.

**The problem:** FastAPI runs sync endpoints in a thread pool. Multiple threads call `model.generate()` simultaneously → CUDA errors. Fix: add `threading.Lock()`. Now the model processes exactly one request at a time regardless of concurrency.

**What this reveals:** throughput is completely flat across concurrency levels. Adding users doesn't increase GPU utilization — it just makes everyone wait longer in the queue. p50 latency at concurrency=5 is 5× worse than concurrency=1 (25.8s vs 5.1s). The GPU is barely used.

**Why:** `nvidia-smi` shows near-0% GPU-Util during generation. This is because LLM decode is memory-bandwidth bound, not compute bound. The GPU spends most of its time streaming KV cache from HBM, not running tensor cores. And with only one request at a time, even that bandwidth is barely used.

---

## Stage 2 — vLLM (Continuous Batching + PagedAttention)

Replace the HF serving loop with vLLM's `AsyncLLMEngine`. Same model, completely different serving runtime.

**What vLLM does:**

*Continuous batching:* Instead of processing one request to completion before starting the next, vLLM batches decode steps across concurrent requests. Multiple sequences are stacked into one matrix multiply: `[total_tokens, d_model] × W`. One large matmul is far more efficient than many small sequential ones — tensor cores stay saturated.

*PagedAttention:* KV cache is stored in fixed-size pages instead of contiguous buffers. Each sequence gets a block table mapping logical pages → physical GPU memory pages. This eliminates fragmentation and lets vLLM pack more sequences into the same VRAM.

**Result:** At concurrency=100, throughput goes from 32 to 2,596 tokens/sec — **80x improvement**. p50 latency at 100 concurrent users (6.2s) is only 20% worse than HF at 1 user (5.1s). The GPU is now actually doing useful work at scale.

---

## Stage 3 — AWQ 4-bit Quantization

Switch to `TheBloke/Mistral-7B-Instruct-v0.1-AWQ`. Same vLLM engine, 4-bit weights instead of FP16.

**What AWQ does:** Activation-aware Weight Quantization analyzes which weights get multiplied by large activations during inference. Those weights have the most impact on output quality — AWQ protects them by scaling before quantization. Everything else gets aggressively quantized to 4-bit. Result: 14GB → ~4GB weight footprint with minimal quality loss.

**Why this matters for serving:** LLM decode is memory-bandwidth bound. The GPU streams weight tensors from HBM on every decode step. Smaller weights = less data streamed = faster token generation. At c=1, AWQ produces 117 tok/s vs 49 tok/s for vLLM FP16 — **2.4x faster** with no concurrency involved, purely from reduced HBM traffic.

**VRAM split:** vLLM pre-allocates 90% of VRAM regardless of model size. FP16 uses 14GB for weights + ~5GB for KV cache. AWQ uses 4GB for weights + ~15GB for KV cache. More KV cache = more concurrent sequences fit in memory.

---

## Stage 4 — GPTQ 4-bit Quantization

Switch to `TheBloke/Mistral-7B-Instruct-v0.1-GPTQ`. Same 4-bit idea, different algorithm.

**What GPTQ does:** Uses the Hessian (second-order gradient information) to measure how sensitive each weight is to perturbation. Sensitive weights are quantized more carefully. Crucially, GPTQ actively compensates for quantization error — when it quantizes a weight and introduces error, it adjusts neighboring weights to cancel it out. Layer by layer.

**Observation:** GPTQ edges AWQ at c=1 (140 vs 117 tok/s) but collapses at scale. At c=50, AWQ is 2× faster (2,157 vs 1,014 tok/s).

**Why:** vLLM itself warns at startup — *"gptq quantization is not fully optimized yet."* The GPTQ CUDA kernel in vLLM doesn't scale with batch size. At high concurrency, the batch is large and the GPU wants continuous tensor core work. The GPTQ kernel can't feed tensor cores efficiently at scale. Throughput collapses.

This raised a question: **is this a problem with GPTQ the algorithm, or GPTQ the kernel implementation?**

---

## Stage 5 — GPTQ Marlin (Optimized Kernel)

Same `TheBloke/Mistral-7B-Instruct-v0.1-GPTQ` weights. Change only the kernel: `--quantization gptq_marlin`.

Marlin is a CUDA kernel written specifically for GPTQ on Ampere/Ada GPUs (RTX 4090 is Ada). Better tiling, better batched dequantization, better warp scheduling, better tensor core utilization. The quantization math is identical. The weights are identical. Only the kernel changes.

**Result at c=100:**
- GPTQ (weak kernel): 1,718 tok/s
- GPTQ Marlin (optimized kernel): 3,023 tok/s
- **Same weights. 1.76x throughput. Pure kernel engineering.**

Marlin also beats AWQ (3,023 vs 2,511 tok/s at c=100). AWQ's apparent advantage over GPTQ was never about AWQ being a better quantization algorithm — it was about AWQ having a better-engineered kernel in vLLM. Once GPTQ got an equivalently optimized kernel, it beat AWQ.

---

## Results

| Backend | Concurrency | p50 | p99 | Tokens/sec | Req/sec | GPU Mem |
|---------|-------------|-----|-----|------------|---------|---------|
| HF naive | 1 | 5.148s | 5.148s | 32.55 | 0.19 | 14,466 MiB |
| HF naive | 5 | 25.792s | 25.792s | 32.76 | 0.19 | 14,466 MiB |
| vLLM FP16 | 1 | 3.304s | 3.330s | 49.60 | 0.30 | 19,472 MiB |
| vLLM FP16 | 10 | 3.611s | 3.632s | 444.60 | 2.76 | 19,472 MiB |
| vLLM FP16 | 50 | 4.747s | 4.753s | 1,714.23 | 10.56 | 19,472 MiB |
| vLLM FP16 | 100 | 6.241s | 6.246s | 2,596.58 | 16.01 | 19,472 MiB |
| AWQ 4-bit | 1 | 1.358s | 1.421s | 117.93 | 0.73 | 19,262 MiB |
| AWQ 4-bit | 10 | 1.773s | 1.816s | 899.53 | 5.62 | 19,342 MiB |
| AWQ 4-bit | 50 | 3.793s | 3.844s | 2,157.86 | 13.31 | 19,566 MiB |
| AWQ 4-bit | 100 | 6.485s | 6.489s | 2,511.52 | 15.41 | 19,636 MiB |
| GPTQ 4-bit | 1 | 1.160s | 1.198s | 140.30 | 0.86 | 19,538 MiB |
| GPTQ 4-bit | 10 | 1.986s | 2.013s | 807.18 | 5.02 | 19,540 MiB |
| GPTQ 4-bit | 50 | 8.181s | 8.463s | 1,014.76 | 6.31 | 19,764 MiB |
| GPTQ 4-bit | 100 | 9.341s | 9.587s | 1,718.31 | 10.57 | 19,764 MiB |
| GPTQ Marlin | 1 | 1.284s | 1.333s | 125.81 | 0.77 | 19,032 MiB |
| GPTQ Marlin | 10 | 1.515s | 1.578s | 1,068.90 | 6.54 | 19,032 MiB |
| GPTQ Marlin | 50 | 3.362s | 3.403s | 2,537.04 | 15.57 | 19,162 MiB |
| GPTQ Marlin | 100 | 5.273s | 5.404s | 3,023.39 | 18.73 | 19,358 MiB |

**Throughput at concurrency=100 (highest load):**

```
GPTQ Marlin   ████████████████████████  3,023 tok/s
AWQ           ████████████████████      2,511 tok/s
vLLM FP16     █████████████████████     2,596 tok/s
GPTQ          █████████████            1,718 tok/s
HF naive      ▎                           32 tok/s
```

---

## Key Concepts

**Continuous batching** — vLLM batches decode steps across concurrent requests into one large matmul instead of processing requests sequentially. Keeps tensor cores saturated.

**PagedAttention** — KV cache stored in fixed-size pages mapped via per-sequence block tables. Eliminates fragmentation, allows more sequences per VRAM.

**Decode is memory-bandwidth bound** — GPU-Util looks low during generation but HBM is busy streaming KV cache and weights. The bottleneck is HBM bandwidth, not tensor cores.

**AWQ** — Activation-aware Weight Quantization. Protects weights that get multiplied by large activations. 4-bit with minimal quality loss.

**GPTQ** — Hessian-based quantization with error compensation. Theoretically rigorous but the vLLM kernel implementation is less optimized for large batches.

**GPTQ Marlin** — Same GPTQ weights, purpose-built CUDA kernel for Ampere/Ada GPUs. Better tensor core scheduling at large batch sizes. Beats AWQ at scale.

**The key lesson** — In production LLM serving, kernel implementation quality can outweigh quantization algorithm choice. GPTQ vs GPTQ Marlin is the same 4-bit math with a 1.76x throughput difference. Real-world performance is an engineering problem, not just an algorithms problem.

---

## File Structure

```
app_hf.py        # Stage 1 — naive HuggingFace + FastAPI
app_vllm.py      # Stage 2 — vLLM AsyncLLMEngine wrapper
app_gptq.py      # Stage 4 — vLLM + GPTQ (FastAPI wrapper)
benchmark.py     # async benchmark harness — works for all backends
requirements.txt
devlogs.md       # full dev log — bugs, decisions, observations
```

## Setup (RunPod)

```bash
git clone https://github.com/tanny1412/llm-inference-benchmarking.git
cd llm-inference-benchmarking
pip install -r requirements.txt
export HF_HOME=/workspace/hf-cache
```

## Running Each Stage

**Stage 1 — HF:**
```bash
uvicorn app_hf:app --host 0.0.0.0 --port 8000
```

**Stage 2 — vLLM FP16:**
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

**Stage 5 — GPTQ Marlin:**
```bash
vllm serve TheBloke/Mistral-7B-Instruct-v0.1-GPTQ --host 0.0.0.0 --port 8000 --quantization gptq_marlin --dtype float16 --download-dir /workspace/hf-cache/hub
```

## Running Benchmarks

```bash
python benchmark.py --backend hf --concurrency 1 --num_requests 20 --max_new_tokens 200
python benchmark.py --backend vllm --concurrency 10 --num_requests 50 --max_new_tokens 200
python benchmark.py --backend awq --concurrency 10 --num_requests 50 --max_new_tokens 200
python benchmark.py --backend gptq --concurrency 10 --num_requests 50 --max_new_tokens 200
python benchmark.py --backend gptq_marlin --concurrency 10 --num_requests 50 --max_new_tokens 200
```

Results saved to `results_<backend>_c<concurrency>_<timestamp>.json`
