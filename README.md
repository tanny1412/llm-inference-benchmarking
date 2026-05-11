# LLM Inference Benchmarking

An end-to-end LLM inference benchmarking project comparing naive HuggingFace serving, vLLM with PagedAttention and continuous batching, and AWQ 4-bit quantization — measuring throughput, latency, and GPU memory across concurrency levels.

## Stages

| Stage | Backend | Precision |
|-------|---------|-----------|
| 1 | Naive HuggingFace + FastAPI | FP16 |
| 2 | vLLM | FP16 |
| 3 | vLLM + AWQ | 4-bit |

## Model

`mistralai/Mistral-7B-Instruct-v0.1`

## Metrics

- TTFT (Time To First Token) — p50 and p99
- Throughput (tokens/sec)
- GPU memory usage
- Concurrency scaling

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
