# Dev Logs — LLM Inference Benchmarking

---

## Stage 1 — Naive HF Baseline

### Questions & Doubts

**Q: Why load the model on GPU and not CPU?**
Mistral 7B in FP16 = ~14GB. CPU doesn't have that memory headroom and has no tensor cores — matrix multiplications that take milliseconds on GPU take seconds on CPU.

**Q: Why FP16 and not FP32?**
FP32 = 4 bytes/weight → ~28GB for 7B params → OOM on most single GPUs. FP16 = 2 bytes/weight → ~14GB → fits on A100 40GB. FP16 is the standard inference baseline before quantization.

**Q: Why `AutoModelForCausalLM` and not `AutoModel`?**
`AutoModel` is generic. `AutoModelForCausalLM` is specific to decoder-only causal language models (like Mistral) and gives us the `.generate()` method needed for text generation.

**Q: Why declare `model = None` and `tokenizer = None` at module level?**
Two reasons: (1) Python's `global` keyword requires the variable to exist at module scope before reassigning inside a function. (2) Both the lifespan loader and the `/generate` endpoint need to access them — they must be at module scope for sharing.

**Q: Why `device_map="cuda"` instead of `.to("cuda")`?**
`device_map` comes from the `accelerate` library and is smarter — it can split the model across multiple GPUs or CPU+GPU if needed. `.to("cuda")` blindly moves everything to one GPU and OOMs if it doesn't fit. That's why `accelerate` is in requirements.txt.

**Q: What is `@asynccontextmanager` and why use it for lifespan?**
`contextmanager` lets a function with a `yield` behave as a resource manager — before yield is setup, after yield is teardown. `async` is needed because FastAPI is an async framework. The decorator is essentially boilerplate FastAPI requires for lifespan — the real logic is just what's before and after the `yield`.

**Q: Why is lifespan loading done once outside the endpoint and not inside?**
Loading a 14GB model on every request would be catastrophically slow. Load once at startup, reuse across all requests. The `/generate` endpoint and lifespan both access the same module-level `model` and `tokenizer` objects.

**Q: Why does `max_new_tokens=200` need to stay constant across all benchmark stages?**
Every new token = one decode step. If Stage 1 generates 200 tokens and Stage 2 generates 500, you're not comparing the same workload — Stage 2 did more work. Same prompt + same `max_new_tokens` = fair comparison.

**Q: Why is `accelerate` in requirements.txt?**
HuggingFace uses it under the hood when `device_map` is passed to `from_pretrained`. Without it, `device_map="cuda"` fails.

---

## Observations

**GPU-Util during inference**
- Single request showed 0% GPU-Util in `nvidia-smi`
- Longer prompt briefly spiked to 70-80% then dropped back to near 0%
- Why: the spike is the **prefill phase** — full prompt processed in parallel, compute-heavy
- The drop is the **decode phase** — one token at a time, memory-bandwidth bound not compute-bound
- `nvidia-smi` GPU-Util measures compute utilization, not memory bandwidth — so decode looks idle even though HBM is busy streaming KV cache
- Lesson: GPU-Util alone is a misleading metric for LLM inference. A GPU can be memory-bandwidth saturated while showing low compute utilization.

---

## RunPod SSH + VS Code Remote Setup

**Problem:** RunPod only injects SSH keys at pod creation time. If pod is already running, you must manually add the key.

**Steps to connect VS Code to RunPod via SSH:**

1. Get your local public key:
```bash
cat ~/.ssh/id_ed25519.pub
```

2. In RunPod Settings → SSH Public Keys → paste key → click **Update public key**

3. Manually inject key into running pod via web terminal:
```bash
echo "your-public-key" >> ~/.ssh/authorized_keys
```

4. Add RunPod to local SSH config (`~/.ssh/config`):
```
Host runpod
    HostName 209.170.80.132
    User root
    Port 12649
    IdentityFile ~/.ssh/id_ed25519
```

5. In VS Code → Remote Explorer → SSH → click arrow next to `runpod` to connect

6. Open `/workspace/llm-inference-benchmarking` as the working folder

**Note:** Pod IP and port change every time you restart the pod — update `~/.ssh/config` accordingly.

**SSH issue — port 22 blocked:**
- Proxied SSH (`ssh.runpod.io`) uses port 22 which is blocked on some home/ISP networks
- Direct TCP SSH (e.g. `ssh root@209.170.80.132 -p 12602`) uses a non-standard port that isn't blocked
- Fix: always use direct TCP SSH and update IP/port in `~/.ssh/config` after each pod restart
- To check: `ssh -v` shows "Connecting to ssh.runpod.io port 22" and hangs = port 22 blocked

---

## Concurrency Bug — Naive HF Server

**Bug:** Running benchmark with concurrency > 1 causes `CUDA error: device-side assert triggered`

**Why:** FastAPI runs synchronous `def` endpoints in a thread pool. With concurrency=5, five threads simultaneously call `model.generate()` on the same PyTorch model. PyTorch model inference is NOT thread-safe — concurrent GPU operations corrupt each other's state.

**What this reveals:** This is the fundamental problem with naive HF serving. No request queuing, no concurrency control at the model level. The model can only safely handle one request at a time.

**Fix:** Add a `threading.Lock()` so only one request can use the model at a time — making the sequential behavior explicit and safe.

**Key lesson:** This is exactly WHY vLLM exists. vLLM handles concurrency properly with a scheduler that queues requests and batches them efficiently. Naive HF serving can't do this safely.

---

## Bugs & Fixes

**Bug 1: `transformers 5.8.0` incompatible with `torch 2.4.0` on RunPod**
- Error: crash in `transformers/integrations/moe.py` — newer transformers used a PyTorch API that didn't exist in 2.4.0
- Fix: pinned `transformers==4.40.0` in requirements.txt — stable, fully supports Mistral 7B, compatible with PyTorch 2.4.0

**Bug 2: `dtype` vs `torch_dtype`**
- The deprecation warning `torch_dtype is deprecated, use dtype instead` came from the system-installed newer transformers, not our pinned 4.40.0
- In transformers 4.40.0, the correct argument is `torch_dtype` — using `dtype` caused `TypeError: unexpected keyword argument`
- Fix: reverted back to `torch_dtype=torch.float16`

**Lesson:** Always check version compatibility between PyTorch and transformers. RunPod templates ship with a fixed PyTorch version — pip installing the latest transformers can break things.

---

## Benchmark Results — Stage 1 (Naive HF)

| Concurrency | p50 latency | p99 latency | Tokens/sec | Req/sec |
|-------------|-------------|-------------|------------|---------|
| 1           | 5.148s      | 5.148s      | 32.55      | 0.19    |
| 5           | 25.792s     | 25.792s     | 32.76      | 0.19    |

**Why tokens/sec and req/sec stay flat as concurrency increases:**

With `threading.Lock()`, the GPU runs exactly one request at a time regardless of the concurrency setting. The semaphore in benchmark.py lets 5 requests start — but 4 immediately block on the lock. The GPU never sees 5 parallel requests. It sees a sequential queue.

- `tokens_per_sec = total_tokens / wall_time` — wall time and tokens both grow proportionally → ratio constant
- `req/sec = num_requests / wall_time` — same reason → ratio constant

Only p50 latency explodes: each user now waits for everyone ahead of them in the queue. At concurrency=5, p50 went from 5.1s → 25.8s (≈5x), which is exactly what you'd expect from a perfect queue with sequential execution.

**The lesson:** The naive server cannot scale. Adding concurrent users increases queue wait time, but does nothing to throughput. It's a single-lane road — more cars don't make the lane faster, they just make the queue longer. This is exactly what vLLM fixes with continuous batching.

---

## vLLM — How Batching Works

**Why naive HF can't batch:** The threading.Lock() makes requests sequential. Removing the lock causes CUDA crashes because PyTorch model inference is not thread-safe.

**What vLLM does instead — batching at the matmul level:**

Multiple requests are stacked into a single input matrix before the forward pass:

```
[seq_len_1, d_model]
[seq_len_2, d_model]   →  [total_tokens, d_model]
[seq_len_3, d_model]
```

This becomes one big matrix multiply: `[total_tokens, d_model] × W`. GPUs are designed to do large matmuls efficiently — tensor cores stay saturated. One big matmul is much faster than 10 small sequential ones.

**Why KV caches don't contaminate each other:**

The weight matrix `W` is shared across all requests (same model). The batched matmul only uses `W` — it doesn't touch any KV cache.

The KV cache is only involved during the attention step, which happens after the projections. Attention is computed independently per sequence. Each token carries metadata about which sequence it belongs to, and vLLM's scheduler maintains a **block table** per sequence: a mapping from logical page index → physical page in GPU memory (PagedAttention).

So during attention:
- Token from request A → look up request A's block table → fetch request A's K/V pages only
- Token from request B → look up request B's block table → fetch request B's K/V pages only

**Mental model:**
```
Batch of tokens → big matmul with W (shared weights, no KV cache)
                         ↓
             Split back into per-request sequences
                         ↓
             Attention computed independently per request
             (each using only its own KV cache via block table)
```

Batching = shared matmul. Isolation = per-request block tables.

---

## Why We Keep the Same FastAPI Wrapper Across All Stages

`benchmark.py` hits `POST /generate` with `{"prompt": ..., "max_new_tokens": ...}` for every backend. vLLM's built-in server exposes OpenAI-compatible endpoints (`/v1/completions`) — different format, which would require changing the benchmark script per stage.

Same FastAPI wrapper → same benchmark script → fair comparison.

**The general principle: control everything except the variable you're testing.**

```
same API
same request format
same benchmark script
same hardware
same workload
same FastAPI layer

ONLY inference engine changes
```

If numbers improve, you can confidently say the improvement came from vLLM itself — not from a different server, endpoint, networking stack, or serialization format.

**One-line interview answer:**
We kept the same FastAPI wrapper across backends to ensure apples-to-apples benchmarking and isolate inference engine performance differences.

---

## Bug: vLLM 0.20.2 CUDA Driver Incompatibility

**Root cause of all sequential behavior:**

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12040)
```

vLLM 0.20.2 uses the V1 engine internally which requires CUDA 12.6+. The RunPod RTX 4090 pod has CUDA 12.4 (driver version 12040). The V1 engine crashed on initialization.

**Why our FastAPI wrapper appeared to work but didn't batch:**

When `AsyncLLMEngine.from_engine_args()` was called, it tried to initialize the V1 engine under the hood — which crashed on the CUDA driver check. The crash was silent — our FastAPI server still responded to requests, but the actual vLLM engine never started. Requests were processed sequentially as a fallback, which is why we saw exactly 34 tokens/sec regardless of concurrency. The engine wasn't running at all.

**This was not a FastAPI wrapping problem.** The wrapper code was correct. The engine underneath it never initialized.

**Fix:** Pin vLLM to a version compatible with CUDA 12.4:
```bash
pip install vllm==0.6.6
```

vLLM 0.6.x uses the V0 engine which works with CUDA 12.1+.

**Secondary bug: transformers 5.8.0 incompatible with vLLM 0.6.6**

After downgrading vLLM, the next error was:
```
AttributeError: LlamaTokenizer has no attribute all_special_tokens_extended
```

vLLM 0.6.6 was built against transformers 4.x. The `all_special_tokens_extended` attribute was removed in transformers 5.x. The system had 5.8.0 installed.

Fix: downgrade transformers to match vLLM 0.6.6:
```bash
pip install transformers==4.45.2
```

`transformers==4.45.2` still uses `torch_dtype` (not `dtype`), so `app_hf.py` remains compatible. Updated `requirements.txt` to pin both `transformers==4.45.2` and `vllm==0.6.6`.

---

## Benchmark Results — Stage 3 (AWQ)

| Concurrency | p50 latency | p99 latency | Tokens/sec | Req/sec | GPU Memory |
|-------------|-------------|-------------|------------|---------|------------|
| 1           | 1.358s      | 1.421s      | 117.93     | 0.73    | 19,262 MiB |
| 10          | 1.773s      | 1.816s      | 899.53     | 5.62    | 19,342 MiB |
| 50          | 3.793s      | 3.844s      | 2157.86    | 13.31   | 19,566 MiB |
| 100         | 6.485s      | 6.489s      | 2511.52    | 15.41   | 19,636 MiB |

**Model weight footprint (disk = VRAM footprint for weights):**

| Model | Size |
|-------|------|
| Mistral 7B FP16 | 14 GB |
| Mistral 7B AWQ 4-bit | 3.9 GB |

3.6x smaller. 16-bit weights compressed to 4-bit. The freed VRAM goes to KV cache, allowing vLLM to hold more concurrent sequences.

`nvidia-smi` total usage looks the same (~19GB) for both because vLLM pre-allocates 90% of VRAM regardless of model size. The difference is in how that VRAM is split: FP16 uses 14GB for weights + 5GB for KV cache, AWQ uses 4GB for weights + 15GB for KV cache. More KV cache = larger batches at high concurrency.

**Concurrency=1 comparison across all stages:**

| Backend | p50 | Tokens/sec | Model size |
|---------|-----|------------|------------|
| HF FP16 | 5.148s | 32.55 | 14 GB |
| vLLM FP16 | 3.304s | 49.60 | 14 GB |
| AWQ 4-bit | 1.358s | 117.93 | 3.9 GB |

AWQ is 2.4x faster than vLLM FP16 at single request — 4-bit weights are smaller → less data streamed from HBM per decode step → faster token generation even before batching kicks in.

**Full three-stage comparison at all concurrency levels:**

| Backend | Concurrency | p50 | p99 | Tokens/sec | Req/sec | GPU Memory |
|---------|-------------|-----|-----|------------|---------|------------|
| HF FP16 | 1 | 5.148s | 5.148s | 32.55 | 0.19 | 14,466 MiB |
| HF FP16 | 5 | 25.792s | 25.792s | 32.76 | 0.19 | 14,466 MiB |
| vLLM FP16 | 1 | 3.304s | 3.330s | 49.60 | 0.30 | 19,472 MiB |
| vLLM FP16 | 10 | 3.611s | 3.632s | 444.60 | 2.76 | 19,472 MiB |
| vLLM FP16 | 50 | 4.747s | 4.753s | 1,714.23 | 10.56 | 19,472 MiB |
| vLLM FP16 | 100 | 6.241s | 6.246s | 2,596.58 | 16.01 | 19,472 MiB |
| AWQ 4-bit | 1 | 1.358s | 1.421s | 117.93 | 0.73 | 19,262 MiB |
| AWQ 4-bit | 10 | 1.773s | 1.816s | 899.53 | 5.62 | 19,342 MiB |
| AWQ 4-bit | 50 | 3.793s | 3.844s | 2,157.86 | 13.31 | 19,566 MiB |
| AWQ 4-bit | 100 | 6.485s | 6.489s | 2,511.52 | 15.41 | 19,636 MiB |

**AWQ vs vLLM FP16 — key observations:**

At low concurrency (c=1, c=10), AWQ wins clearly:
- c=1: AWQ 117.93 tokens/sec vs vLLM 49.60 — **2.4x faster**
- c=10: AWQ 899.53 tokens/sec vs vLLM 444.60 — **2.0x faster**

Why: AWQ 4-bit weights are 3.5x smaller → HBM streams less data per decode step. Decode is memory-bandwidth bound, so weight size directly determines token generation speed. Single-request performance is a pure test of per-step HBM bandwidth efficiency.

At high concurrency (c=50, c=100), the gap closes:
- c=50: AWQ 2,157 vs vLLM 1,714 — **1.26x faster**
- c=100: AWQ 2,511 vs vLLM 2,596 — **vLLM slightly ahead**

Why: at saturation, dequantization overhead starts to matter. AWQ stores weights at 4-bit but must dequantize them to float16 before each matmul — this is a small overhead per step. At c=1, this overhead is negligible compared to the bandwidth savings. At c=100, the batch size is large enough that the dequantization cost becomes meaningful, and both backends are hitting the same fundamental GPU compute ceiling. The GPU is fully saturated with work regardless — AWQ's bandwidth advantage shrinks as compute becomes the bottleneck instead of bandwidth.

**Key takeaway per stage:**
- Stage 1 → Stage 2: throughput 80x at high concurrency. Continuous batching and PagedAttention let the GPU handle concurrent requests together instead of sequentially. Latency barely changes (6.2s vs 5.1s) despite 100x the load.
- Stage 2 → Stage 3: bandwidth efficiency 2–2.4x at low concurrency. 4-bit weights stream faster from HBM. The benefit is most visible where a single request has full GPU attention — with no batching to amortize costs, every saved HBM read directly speeds up the user.
- At saturation: both vLLM FP16 and AWQ converge toward the same GPU compute ceiling. Quantization is a memory trick, not a compute trick — once you're compute-bound, it stops helping.

---

## Benchmark Results — Stage 2 (vLLM)

| Concurrency | p50 latency | p99 latency | Tokens/sec | Req/sec |
|-------------|-------------|-------------|------------|---------|
| 1           | 3.304s      | 3.330s      | 49.60      | 0.30    |
| 10          | 3.611s      | 3.632s      | 444.60     | 2.76    |
| 50          | 4.747s      | 4.753s      | 1714.23    | 10.56   |
| 100         | 6.241s      | 6.246s      | 2596.58    | 16.01   |

**Concurrency=1 comparison — HF vs vLLM (correct results after fixing engine):**

| | HF | vLLM |
|--|--|--|
| p50 | 5.148s | 3.304s |
| Tokens/sec | 32.55 | 49.60 |
| Req/sec | 0.19 | 0.30 |

Even at concurrency=1, vLLM is 1.5x faster. This is because vLLM uses kernel fusion, FlashAttention, and compiled CUDA graphs even for a single request — the engine itself is more efficient than raw HuggingFace regardless of batching.

The earlier "identical" numbers (34 tokens/sec) were from the broken AsyncLLMEngine setup where the V1 engine never initialized. These numbers are the real baseline.

**Full comparison — HF naive vs vLLM:**

| Backend | Concurrency | p50 | Tokens/sec | Req/sec |
|---------|-------------|-----|------------|---------|
| HF naive | 1 | 5.148s | 32.55 | 0.19 |
| HF naive | 5 | 25.792s | 32.76 | 0.19 |
| vLLM | 1 | 3.304s | 49.60 | 0.30 |
| vLLM | 10 | 3.611s | 444.60 | 2.76 |
| vLLM | 50 | 4.747s | 1714.23 | 10.56 |
| vLLM | 100 | 6.241s | 2596.58 | 16.01 |

**GPU Memory comparison:**

| Stage | GPU Memory |
|-------|------------|
| HF FP16 | 14,466 MiB |
| vLLM FP16 | 19,472 MiB |

vLLM uses more memory than HF because it pre-allocates KV cache pages upfront (`gpu_memory_utilization=0.9` × 24GB = ~21GB total, model takes 14GB, rest goes to pre-allocated KV cache). HF only allocates memory per request on demand.

`gpu_memory_utilization=0.9` is vLLM's default — we never set it explicitly. Running `vllm serve` without specifying it automatically reserves 90% of GPU VRAM for model weights + pre-allocated KV cache pages.

**Key takeaways:**
- HF flat-lines: adding concurrency kills latency, throughput never improves
- vLLM scales: 100 concurrent users, 80x throughput improvement (32 → 2596 tokens/sec), 84x more req/sec (0.19 → 16)
- vLLM p50 latency at 100 concurrent users (6.2s) is only 20% worse than HF at 1 user (5.1s) — despite 100x the load
- This is continuous batching + PagedAttention + kernel fusion working together

The real improvement shows at higher concurrency (10, 50, 100).

**Bug: LLM (sync) vs AsyncLLMEngine — why vLLM showed no improvement at concurrency=10:**

First attempt used `LLM` (synchronous class). Each FastAPI request called `llm.generate([prompt], params)` independently with one prompt. vLLM never saw 10 requests at once — it saw 10 separate `generate()` calls one after another. No batching happened. Results were identical to naive HF.

Results with sync `LLM` at concurrency=10:
- p50: 49.969s (exactly 10× the concurrency=1 latency — pure sequential queuing)
- Tokens/sec: 34.18 (identical to concurrency=1 — no throughput gain)
- Req/sec: 0.20 (identical — same ceiling as HF)

**Fix: switch to `AsyncLLMEngine` with correct imports and batching config**

With `LLM` (sync): FastAPI runs endpoint in a thread pool. 10 threads each independently call `llm.generate()` and block. Engine never sees them together.

With `AsyncLLMEngine` (async): FastAPI runs endpoint in the event loop. 10 coroutines each call `await engine.generate()` and yield control back. The engine collects all pending requests and batches them together in one forward pass.

`AsyncLLMEngine.generate()` is a streaming API — it yields tokens as they're generated. The endpoint consumes the stream and returns the final output.

Key change: endpoint becomes `async def`, uses `await engine.generate()`, collects streamed output.

**Additional fixes for vLLM 0.20.x:**
- `AsyncEngineArgs` must be imported directly from `vllm`, not from `vllm.engine.arg_utils` — wrong import path causes silent fallback to sequential processing
- `max_num_seqs=256` — explicitly tells vLLM to batch up to 256 concurrent sequences (default is too conservative)
- `max_num_batched_tokens=8192` — allows larger batches per forward pass

**Why `final_output = None` before the async for loop:**

`engine.generate()` is a streaming API — it yields a partial output object after every new token generated, not the full response at the end. Each yielded object contains only the tokens generated so far.

```python
final_output = None
async for output in engine.generate(prompt, params, request_id):
    final_output = output  # overwrites on every token
response = final_output.outputs[0].text  # last output = full completed text
```

We initialize to `None` so Python doesn't throw `NameError` when accessing `final_output` after the loop. We keep overwriting it so that when the loop ends (all 200 tokens done), we have the final output object with the complete text.

**Why `None` specifically and not just leaving it uninitialized:**

Python only knows a variable exists if it was assigned before you use it. If the `async for` loop ran zero iterations (empty generator), `final_output` would never get assigned — then `final_output.outputs[0].text` would reference a variable that doesn't exist → `NameError`. In practice the generator always yields at least one output, but Python doesn't know that at parse time. Initializing to `None` guarantees the variable always exists after the loop.

**General Python rule:** If a variable is assigned inside a loop and used after the loop, initialize it beforehand.

---

**Why each vLLM feature does nothing at concurrency=1:**

- **Continuous batching** — fills freed batch slots with waiting requests. With one request, there are no waiting requests. Nothing to batch.
- **PagedAttention** — eliminates KV cache fragmentation when multiple sequences compete for memory. With one request, there's no competition. Fragmentation isn't a problem when only one sequence uses the cache.
- **Scheduler** — manages which requests run each forward pass, preempts when memory is tight. With one request, every decision is trivial — run the one request, nothing to preempt, nothing to prioritize.

All three are solutions to problems that only appear under concurrent load. Single request = no concurrency problems = no benefit from any of these systems.

---

## nvidia-smi — Empty Processes Section in Containers

In RunPod (containerized environment), the Processes section in `nvidia-smi` always appears empty even when the GPU is actively running. This is a permission restriction — the container can't see OS-level process info.

The GPU utilization metrics are still accurate and reliable:
- **GPU-Util %** — compute utilization
- **Memory-Usage MiB** — VRAM used
- **Pwr:Usage/Cap W** — power draw

If these numbers are non-zero, the GPU is working. The empty process list is not an error.

---

## Bug: Duplicate Model Download — HF Cache Path Mismatch

Two different tools use different cache path conventions:

- `app_hf.py` with `HF_HOME=/workspace/hf-cache` → HuggingFace puts model at:
  `/workspace/hf-cache/hub/models--mistralai--Mistral-7B-Instruct-v0.1/`

- `vllm serve --download-dir /workspace/hf-cache` → vLLM looks for model at:
  `/workspace/hf-cache/models--mistralai--Mistral-7B-Instruct-v0.1/`

vLLM didn't find the model at its expected path, started downloading a second copy, hit disk quota partway through, and left an incomplete ~5GB copy wasting space.

**Fix:** Delete the incomplete copy and point vLLM to the correct hub/ path:
```bash
rm -rf /workspace/hf-cache/models--mistralai--Mistral-7B-Instruct-v0.1
vllm serve mistralai/Mistral-7B-Instruct-v0.1 --host 0.0.0.0 --port 8000 --tokenizer-mode mistral --download-dir /workspace/hf-cache/hub
```

**General lesson:** When using multiple tools with the same model, always verify they're reading from the same cache location — otherwise you end up with duplicate downloads and wasted disk space.

---

## Bug: AWQ requires float16, not bfloat16

```
ValueError: torch.bfloat16 is not supported for quantization method awq. Supported dtypes: [torch.float16]
```

vLLM defaulted to bfloat16 because `TheBloke/Mistral-7B-Instruct-v0.1-AWQ`'s `config.json` specifies `torch_dtype: bfloat16`. vLLM reads that and uses it as the default when `--dtype` isn't passed explicitly.

AWQ's kernel implementation only supports float16 — the mismatch causes the crash.

**Fix:** add `--dtype float16` to the serve command:
```bash
vllm serve TheBloke/Mistral-7B-Instruct-v0.1-AWQ --host 0.0.0.0 --port 8000 --tokenizer-mode mistral --quantization awq --dtype float16 --download-dir /workspace/hf-cache/hub
```

**Note:** vLLM also detected the model can run with `awq_marlin` — a more optimized AWQ kernel for modern GPUs. We use `awq` first for the baseline, then can try `awq_marlin` for better performance.

**Secondary bug: `--tokenizer-mode mistral` incompatible with TheBloke AWQ model**

```
OSError: Found 0 files matching the pattern: tokenizer.model.v.*|tekken.json
```

`--tokenizer-mode mistral` expects the newer Mistral tokenizer format (`tokenizer.model.v3` or `tekken.json`). But `TheBloke/Mistral-7B-Instruct-v0.1-AWQ` ships with the standard HuggingFace tokenizer (`tokenizer.model`) — the older format. The newer `--tokenizer-mode mistral` flag only works with official Mistral models that use their proprietary tokenizer format.

**Fix:** remove `--tokenizer-mode mistral`. Final working command:
```bash
vllm serve TheBloke/Mistral-7B-Instruct-v0.1-AWQ --host 0.0.0.0 --port 8000 --quantization awq --dtype float16 --download-dir /workspace/hf-cache/hub
```

---

## Stage 3 — AWQ Quantization

**What AWQ does:**

AWQ (Activation-aware Weight Quantization) looks at which weights actually matter by analyzing activation magnitudes during calibration. Weights that get multiplied by large activations are kept at higher precision — they have the most impact on output quality. Weights that barely affect the output get quantized aggressively to 4-bit.

Result: 4-bit weights instead of 16-bit. Mistral 7B in FP16 = ~14GB VRAM. In AWQ 4-bit = ~4GB VRAM. Same model, 3.5x less memory.

**Why this matters for serving:**
- More VRAM headroom → larger KV cache → more concurrent sequences fit in memory
- 4-bit weights are smaller → faster to stream from HBM during decode → higher tokens/sec
- Minimal quality loss because AWQ protects the weights that matter most

---

## Key Decisions

- Model: `mistralai/Mistral-7B-Instruct-v0.1` (instruction-tuned, not base — responds coherently without fine-tuning)
- Precision: FP16 (`torch.float16`)
- Server: FastAPI + uvicorn
- Model loaded once at startup via FastAPI lifespan event
