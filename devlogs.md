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

## Key Decisions

- Model: `mistralai/Mistral-7B-Instruct-v0.1` (instruction-tuned, not base — responds coherently without fine-tuning)
- Precision: FP16 (`torch.float16`)
- Server: FastAPI + uvicorn
- Model loaded once at startup via FastAPI lifespan event
