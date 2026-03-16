#!/usr/bin/env python 3

import argparse
import time
import json
import subprocess
import threading
import os
import sys
from urllib import request, error
from statistics import mean
from typing import List, Any

DEFAULT_MODEL_NAME = "some model"
DEFAULT_PROMPT_FILE = "prompts"
DEFAULT_MAX_TOKENS = 256
DEFAULT_GPU_SAMPLING_INTERVAL_S = 0.5
DEFAULT_BASE_URL = os.getenv("RHEL_AI_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_REQUEST_TIMEOUT = 300


def sample_gpu_utilization(
        samples: List[int], 
        stop_event: threading.Event, 
        interval: float
) -> None:
    while not stop_event.is_set():
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi", 
                    "--query-gpu=utilization.gpu", 
                    "--format=csv,noheader,nounits"
                ]
            )
            samples.append(int(output.decode("utf-8").strip()))
        except Exception:
            pass
        time.sleep(interval)


def load_prompts(prompt_file: str) -> List[str]:
    with open(prompt_file, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def run_inference(
        prompt: str, 
        model_name: str, 
        max_tokens: int, 
        base_url: str, 
        request_timeout: int
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "message": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }

    req = request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST"
    )

    with request.urlopen(req, timeout=request_timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# Run benchmark prompts
def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_prompts(args.prompt_file)
    if not prompts:
        raise ValueError(f"No prompts file found in {args.prompt_file}")

    gpu_samples: List[int] = []
    stop_event = threading.Event()
    gpu_thread = threading.Thread(
        target = sample_gpu_utilization,
        args=(gpu_samples, stop_event, args.gpu_sampling_interval),
        daemon=True
    )
    gpu_thread.start()
    start_time = time.time()
    responses: List[dict[str, Any]]= []
    failures: List[dict[str, Any]]= []
    for idx, prompt in enumerate(prompts, start=1):
        try:
            responses.append(run_inference(
                prompt=prompt,
                model_name=args.model,
                max_tokens=args.max_tokens,
                base_url=args.base_url,
                request_timeout=args.request_timeout
                ))
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            failures.append({"index": idx, "type": "http", "status": e.code, "error": body})
        except Exception as e:
            failures.append({"index": idx, "type": "exception", "status": str(e)})
    duration = time.time() - start_time
    stop_event.set()
    gpu_thread.join()

    # Calculate results

    total_generated_tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r in responses)
    tokens_per_sec = total_generated_tokens / duration if duration else 0
    tokens_per_hour = tokens_per_sec * 3600
    avg_gpu_util = mean(gpu_samples) if gpu_samples else 0

    return {
        "model": args.model,
        "num_prompts": len(prompts),
        "total_generated_tokens": total_generated_tokens,
        "duration": duration,
        "tokens_per_sec": tokens_per_sec,
        "tokens_per_hour": tokens_per_hour,
        "avg_gpu_utilization_percent": avg_gpu_util,
        "successful_requests": len(responses),
        "failed_requests": len(failures),
        "errors": failures
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark prompts against a local RHEL AI inference server."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Name of the huggingface model served by vLLM")
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE, help="Path to file with prompt workload. Each line is turned into a prompt.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max number of completion tokens per prompt.")
    parser.add_argument("--gpu-interval", type=float, default=DEFAULT_GPU_SAMPLING_INTERVAL_S, help="Interval in which to sa,ple GPU utilization (seconds).")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL for RHEL AI inference server.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT, help="Specify request timeout for prompting.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        results = run_benchmark(args)
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
