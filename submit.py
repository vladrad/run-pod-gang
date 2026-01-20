#!/usr/bin/env python3
"""
Submit kernel to RunPod serverless endpoint for NVFP4 Group GEMM benchmarking.

Usage:
    python submit.py my_kernel.py
    python submit.py my_kernel.py --endpoint YOUR_ENDPOINT_ID --api-key YOUR_API_KEY
    python submit.py my_kernel.py --no-check  # skip correctness check (faster)
"""
import argparse
import requests
import json
import time
import os
from pathlib import Path

# Load .env file if present
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def submit_kernel(
    kernel_file: str,
    endpoint_id: str,
    api_key: str,
    check_correctness: bool = True,
    warmup: int = 5,
    runs: int = 20,
):
    """Submit kernel code and wait for results."""

    with open(kernel_file, "r") as f:
        kernel_code = f.read()

    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Default test cases matching competition
    payload = {
        "input": {
            "kernel_code": kernel_code,
            "test_cases": [
                {"m": [7168], "n": [7168], "k": [16384], "g": 1},
                {"m": [4096], "n": [4096], "k": [7168], "g": 1},
                {"m": [7168], "n": [7168], "k": [2048], "g": 1},
            ],
            "warmup": warmup,
            "runs": runs,
            "check_correctness": check_correctness,
            "seed": 42
        }
    }

    print(f"Submitting {kernel_file}...")
    print(f"Endpoint: {endpoint_id}")
    print(f"Correctness check: {check_correctness}")
    print()

    start = time.time()

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=600)
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out (10 min limit)")
        return
    except requests.exceptions.RequestException as e:
        print(f"ERROR: {e}")
        return

    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s\n")

    if "output" in result:
        output = result["output"]

        if "error" in output:
            print(f"ERROR: {output['error']}")
            if "traceback" in output:
                print("\nTraceback:")
                print(output["traceback"])
            return

        gpu = output.get("gpu", {})
        print(f"GPU: {gpu.get('name', 'unknown')}")
        print(f"CUDA: {gpu.get('cuda', 'unknown')}")
        print(f"PyTorch: {gpu.get('pytorch', 'unknown')}")
        print()

        print("Results:")
        print("=" * 70)
        print(f"{'M':>8} {'N':>8} {'K':>8} {'G':>4} {'Mean (ms)':>12} {'Min (ms)':>12} {'Correct':>10}")
        print("-" * 70)

        all_correct = True
        for r in output.get("results", []):
            m_str = str(r['m'][0]) if len(r['m']) == 1 else str(r['m'])
            n_str = str(r['n'][0]) if len(r['n']) == 1 else str(r['n'])
            k_str = str(r['k'][0]) if len(r['k']) == 1 else str(r['k'])

            correct = r.get('correctness', {})
            correct_str = "PASS" if correct.get('passed', True) else "FAIL"
            if not correct.get('passed', True):
                all_correct = False

            print(f"{m_str:>8} {n_str:>8} {k_str:>8} {r['g']:>4} "
                  f"{r['mean_ms']:>12.3f} {r['min_ms']:>12.3f} {correct_str:>10}")

        print("=" * 70)
        print(f"Geometric Mean: {output.get('geomean_ms', 0):.3f} ms")

        if not all_correct:
            print("\nWARNING: Some test cases failed correctness check!")

    elif "error" in result:
        print(f"RunPod Error: {result.get('error')}")
    else:
        print("Unexpected response:")
        print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Submit NVFP4 Group GEMM kernel to RunPod B200"
    )
    parser.add_argument("kernel_file", help="Python file with custom_kernel(data) function")
    parser.add_argument("--endpoint", default=os.environ.get("RUNPOD_ENDPOINT_ID"),
                        help="RunPod endpoint ID (or RUNPOD_ENDPOINT_ID env)")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"),
                        help="RunPod API key (or RUNPOD_API_KEY env)")
    parser.add_argument("--no-check", action="store_true",
                        help="Skip correctness check (faster)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations (default: 5)")
    parser.add_argument("--runs", type=int, default=20,
                        help="Benchmark iterations (default: 20)")

    args = parser.parse_args()

    if not args.endpoint or not args.api_key:
        print("Error: Need --endpoint and --api-key")
        print("  Or set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY environment variables")
        print()
        print("Example:")
        print("  export RUNPOD_ENDPOINT_ID=abc123")
        print("  export RUNPOD_API_KEY=your_key_here")
        print("  python submit.py my_kernel.py")
        return 1

    submit_kernel(
        args.kernel_file,
        args.endpoint,
        args.api_key,
        check_correctness=not args.no_check,
        warmup=args.warmup,
        runs=args.runs,
    )
    return 0


if __name__ == "__main__":
    exit(main())
