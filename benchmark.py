import asyncio
import aiohttp
import argparse
import json
import time
import statistics
import subprocess
from datetime import datetime

PROMPT = "Explain the difference between machine learning and deep learning in detail."
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"


async def send_request(session, url, max_new_tokens, backend):
    if backend == "vllm":
        payload = {"model": MODEL_NAME, "prompt": PROMPT, "max_tokens": max_new_tokens}
    else:
        payload = {"prompt": PROMPT, "max_new_tokens": max_new_tokens}

    start = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        result = await resp.json()
    latency = time.perf_counter() - start

    if backend == "vllm":
        tokens = len(result["choices"][0]["text"].split())
    else:
        tokens = len(result.get("response", "").split())

    return latency, tokens


async def run_benchmark(url, concurrency, num_requests, max_new_tokens, backend):

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(session):
        async with semaphore:
            return await send_request(session, url, max_new_tokens, backend)

    async with aiohttp.ClientSession() as session:
        start_time = time.perf_counter()
        tasks = [bounded_request(session) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

    latencies = sorted([r[0] for r in results])
    total_tokens = sum(r[1] for r in results)

    p50 = latencies[int(len(latencies) * 0.50)]
    p99 = latencies[int(len(latencies) * 0.99)]

    return {
        "latency_p50_s": round(p50, 3),
        "latency_p99_s": round(p99, 3),
        "tokens_per_sec": round(total_tokens / total_time, 2),
        "requests_per_sec": round(num_requests / total_time, 2),
        "total_time_s": round(total_time, 2),
        "num_requests": num_requests,
        "concurrency": concurrency,
        "max_new_tokens": max_new_tokens,
    }


def get_gpu_memory_mib():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/generate")
    parser.add_argument("--backend", default="hf", help="hf | vllm | awq")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--num_requests", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    print(f"\nBenchmarking: backend={args.backend}, concurrency={args.concurrency}, num_requests={args.num_requests}, max_new_tokens={args.max_new_tokens}")

    if args.backend == "vllm":
        args.url = "http://localhost:8000/v1/completions"

    results = asyncio.run(run_benchmark(args.url, args.concurrency, args.num_requests, args.max_new_tokens, args.backend))

    gpu_mem = get_gpu_memory_mib()
    results["gpu_memory_mib"] = gpu_mem

    print(f"\n--- Results ---")
    print(f"Latency p50:   {results['latency_p50_s']}s")
    print(f"Latency p99:   {results['latency_p99_s']}s")
    print(f"Tokens/sec:    {results['tokens_per_sec']}")
    print(f"Requests/sec:  {results['requests_per_sec']}")
    print(f"Total time:    {results['total_time_s']}s")
    print(f"GPU memory:    {gpu_mem} MiB")

    filename = f"results_{args.backend}_c{args.concurrency}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump({**results, "backend": args.backend}, f, indent=2)
    print(f"\nSaved to {filename}")


if __name__ == "__main__":
    main()
