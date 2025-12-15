"""
Comparative Grammar Backend Tests

Principled comparison of:
1. Unconstrained generation (no grammar backend)
2. Ananke constrained generation

Tests measure:
- Syntactic validity of generated code
- Generation latency
- Token efficiency

This file is fully self-contained - all configuration and image building is inline.
"""

import os
import time
from typing import Optional

import modal

# =============================================================================
# Configuration (self-contained, no imports from other modules)
# =============================================================================

APP_NAME = "grammar-backend-tests"
MODEL_PATH = "unsloth/Qwen3-Coder-30B-A3B-Instruct"

# MoE model optimizations
MEM_FRACTION_STATIC = 0.85
MAX_RUNNING_REQUESTS = 32
CHUNKED_PREFILL_SIZE = 8192

# Ananke configuration
ANANKE_LANGUAGE = "python"
ANANKE_MAX_ROLLBACK_TOKENS = 200

# Timeouts
MODEL_LOAD_TIMEOUT = 600  # 10 minutes for MoE model

# =============================================================================
# Pre-built Image with SGLang + Ananke
# =============================================================================

# Get the repo root (deploy/modal/tests/test_grammar_backends.py -> repo root)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CUDA_VERSION = "12.4"

# Build the image with SGLang from local repo
test_image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    # System dependencies
    .apt_install(
        "git",
        "curl",
        "wget",
        "build-essential",
        "cmake",
        "libopenmpi-dev",
        "libnuma-dev",
        "patchelf",
    )
    # Upgrade pip and install build tools
    .run_commands(
        "pip install --upgrade pip setuptools wheel",
    )
    # Copy local sglang repo - need python/ dir and README.md for pyproject.toml
    .add_local_dir(
        os.path.join(REPO_ROOT, "python"),
        remote_path="/sglang/python",
        copy=True,
    )
    .add_local_file(
        os.path.join(REPO_ROOT, "README.md"),
        remote_path="/sglang/python/README.md",
        copy=True,
    )
    # Install sglang from local repo
    .run_commands(
        "cd /sglang/python && pip install -e '.[srt]'",
    )
    # Install Ananke dependencies
    .pip_install(
        "z3-solver>=4.12.0",
        "tree-sitter>=0.22.0",
        "immutables>=0.20",
    )
    .env({
        "SGLANG_GRAMMAR_BACKEND": "ananke",
        "ANANKE_LANGUAGE": ANANKE_LANGUAGE,
        "ANANKE_MAX_ROLLBACK_TOKENS": str(ANANKE_MAX_ROLLBACK_TOKENS),
        "SGLANG_ALLOW_OVERWRITE": "1",
        "CUDA_HOME": "/usr/local/cuda",
    })
)

# Create volume for model caching
model_volume = modal.Volume.from_name(
    "qwen3-coder-model-cache",
    create_if_missing=True
)

# Create test app
app = modal.App(APP_NAME)

# =============================================================================
# Test Configuration - Token budgets based on real code analysis
# =============================================================================
# Token budget guidelines (validated against real code samples):
#   - Simple completion (1-3 lines):   50-100 tokens
#   - Function body (5-15 lines):      150-300 tokens
#   - Full class/complex logic:        400-600 tokens
#
# The key insight: prompts must be designed so the model CAN complete
# a syntactically valid unit within the token budget.
# =============================================================================

TEST_PROMPTS = [
    # --- TIER 1: Simple completions (100 tokens) ---
    # These should complete a single logical unit easily
    {
        "name": "fibonacci_recursive",
        "prompt": "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n    if n <= 1:\n        return n\n    return",
        "description": "Complete recursive return - trivial",
        "max_tokens": 100,
        "tier": "simple",
    },
    {
        "name": "list_comprehension",
        "prompt": "numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n# Get squares of even numbers\nresult = [x**2 for x in numbers if",
        "description": "Complete list comprehension condition",
        "max_tokens": 100,
        "tier": "simple",
    },

    # --- TIER 2: Function bodies (500 tokens) ---
    # Increased token budget to allow natural completion
    {
        "name": "is_prime_function",
        "prompt": "def is_prime(n: int) -> bool:\n    \"\"\"Check if n is a prime number.\"\"\"",
        "description": "Complete full prime check function",
        "max_tokens": 500,
        "tier": "function",
    },
    {
        "name": "binary_search",
        "prompt": "def binary_search(arr: list[int], target: int) -> int:\n    \"\"\"Return index of target in sorted array, or -1 if not found.\"\"\"",
        "description": "Complete binary search implementation",
        "max_tokens": 500,
        "tier": "function",
    },
    {
        "name": "merge_sorted_lists",
        "prompt": "def merge_sorted(list1: list[int], list2: list[int]) -> list[int]:\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"",
        "description": "Complete merge function",
        "max_tokens": 500,
        "tier": "function",
    },

    # --- TIER 3: Complex structures (600 tokens) ---
    # Classes and multi-function completions - increased budget
    {
        "name": "stack_class",
        "prompt": "class Stack:\n    \"\"\"A simple stack implementation.\"\"\"\n    \n    def __init__(self):",
        "description": "Complete Stack class with push/pop/peek",
        "max_tokens": 600,
        "tier": "complex",
    },
    {
        "name": "linked_list_node",
        "prompt": "class ListNode:\n    \"\"\"Node for a singly linked list.\"\"\"\n    \n    def __init__(self, val: int = 0, next: 'ListNode' = None):\n        self.val = val\n        self.next = next\n\ndef reverse_list(head: ListNode) -> ListNode:\n    \"\"\"Reverse a singly linked list.\"\"\"",
        "description": "Complete linked list reversal",
        "max_tokens": 600,
        "tier": "complex",
    },

    # --- TIER 4: Stress tests (500 tokens) ---
    # These test grammar enforcement under uncertainty
    {
        "name": "nested_control_flow",
        "prompt": "def process_matrix(matrix: list[list[int]]) -> int:\n    \"\"\"Sum all positive numbers in a 2D matrix.\"\"\"",
        "description": "Nested loops - syntax sensitive",
        "max_tokens": 500,
        "tier": "stress",
    },
]


def check_python_syntax(code: str) -> tuple[bool, Optional[str]]:
    """Check if code is syntactically valid Python."""
    try:
        compile(code, "<string>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


@app.cls(
    gpu="A100-80GB",
    image=test_image,
    volumes={"/models": model_volume},
    scaledown_window=600,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MODEL_LOAD_TIMEOUT,
)
@modal.concurrent(max_inputs=100)
class GrammarBackendTester:
    """Test harness for comparing grammar backends."""

    server_process = None
    server_url: str = "http://localhost:30000"

    @modal.enter()
    def start_server(self):
        """Start the SGLang server."""
        import subprocess
        import requests

        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", MODEL_PATH,
            "--host", "0.0.0.0",
            "--port", "30000",
            "--grammar-backend", "ananke",
            "--ananke-language", ANANKE_LANGUAGE,
            "--ananke-max-rollback-tokens", str(ANANKE_MAX_ROLLBACK_TOKENS),
            "--mem-fraction-static", str(MEM_FRACTION_STATIC),
            "--max-running-requests", str(MAX_RUNNING_REQUESTS),
            "--chunked-prefill-size", str(CHUNKED_PREFILL_SIZE),
            "--trust-remote-code",
        ]

        env = os.environ.copy()
        env["HF_HOME"] = "/models"
        env["TRANSFORMERS_CACHE"] = "/models"
        env["HF_HUB_CACHE"] = "/models/hub"

        print(f"Starting SGLang server with model: {MODEL_PATH}")
        self.server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        print(f"Waiting up to {MODEL_LOAD_TIMEOUT}s for model to load...")
        start_time = time.time()

        while time.time() - start_time < MODEL_LOAD_TIMEOUT:
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    print(f"Server ready in {elapsed:.1f}s")
                    return
            except requests.exceptions.RequestException:
                pass

            if self.server_process.poll() is not None:
                stdout, _ = self.server_process.communicate()
                raise RuntimeError(f"Server died. Output:\n{stdout.decode()[:2000]}")

            time.sleep(10)

        raise RuntimeError(f"Server failed to start within {MODEL_LOAD_TIMEOUT}s")

    @modal.exit()
    def stop_server(self):
        """Stop the server."""
        if self.server_process:
            print("Stopping SGLang server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.server_process.kill()

    @modal.method()
    def generate_unconstrained(self, prompt: str, max_tokens: int = 200) -> dict:
        """Generate without grammar constraints."""
        import requests

        start = time.time()
        payload = {
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.9,
        }

        response = requests.post(
            f"{self.server_url}/v1/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        result = response.json()
        elapsed_ms = (time.time() - start) * 1000

        text = result["choices"][0]["text"]
        full_code = prompt + text
        is_valid, error = check_python_syntax(full_code)

        return {
            "text": text,
            "latency_ms": elapsed_ms,
            "tokens": result.get("usage", {}).get("completion_tokens", 0),
            "is_valid": is_valid,
            "error": error,
        }

    @modal.method()
    def generate_ananke(self, prompt: str, max_tokens: int = 200) -> dict:
        """Generate with Ananke syntax constraints."""
        import requests

        start = time.time()
        # constraint_spec is a top-level field in SGLang's OpenAI-compatible API
        payload = {
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.9,
            "constraint_spec": {
                "language": "python",
                "domains": ["syntax"],
            },
        }

        response = requests.post(
            f"{self.server_url}/v1/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        result = response.json()
        elapsed_ms = (time.time() - start) * 1000

        text = result["choices"][0]["text"]
        full_code = prompt + text
        is_valid, error = check_python_syntax(full_code)

        return {
            "text": text,
            "latency_ms": elapsed_ms,
            "tokens": result.get("usage", {}).get("completion_tokens", 0),
            "is_valid": is_valid,
            "error": error,
        }

    @modal.method()
    def stress_test(self, prompt: str, num_samples: int = 10, temperature: float = 0.8, max_tokens: int = 30) -> dict:
        """Run stress test comparing validity rates."""
        import requests

        results = {
            "unconstrained": {"valid": 0, "invalid": 0},
            "ananke": {"valid": 0, "invalid": 0},
        }

        for backend in ["unconstrained", "ananke"]:
            for _ in range(num_samples):
                try:
                    payload = {
                        "model": "default",
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": 0.95,
                    }

                    if backend == "ananke":
                        # constraint_spec is a top-level field
                        payload["constraint_spec"] = {
                            "language": "python",
                            "domains": ["syntax"],
                        }

                    response = requests.post(
                        f"{self.server_url}/v1/completions",
                        json=payload,
                        timeout=60,
                    )
                    response.raise_for_status()

                    text = response.json()["choices"][0]["text"]
                    is_valid, _ = check_python_syntax(prompt + text)

                    if is_valid:
                        results[backend]["valid"] += 1
                    else:
                        results[backend]["invalid"] += 1

                except Exception:
                    results[backend]["invalid"] += 1

        return results


@app.local_entrypoint()
def main():
    """Run comparative tests."""
    print("\n" + "=" * 70)
    print("GRAMMAR BACKEND COMPARISON: Unconstrained vs Ananke")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Test cases: {len(TEST_PROMPTS)}")
    print("Token budgets: 100 (simple), 500 (function), 600 (complex)")
    print("=" * 70)

    tester = GrammarBackendTester()

    # Track results by tier
    results_by_tier = {}
    unconstrained_latency = []
    ananke_latency = []

    # Phase 1: Comparative generation tests
    print("\n[PHASE 1] Tiered Comparative Generation Tests")
    print("-" * 50)

    for i, test in enumerate(TEST_PROMPTS, 1):
        max_tokens = test.get("max_tokens", 200)
        tier = test.get("tier", "unknown")

        if tier not in results_by_tier:
            results_by_tier[tier] = {"unconstrained": 0, "ananke": 0, "total": 0}
        results_by_tier[tier]["total"] += 1

        print(f"\n[{i}/{len(TEST_PROMPTS)}] {test['name']}")
        print(f"    Tier: {tier} | Max tokens: {max_tokens}")
        print(f"    Prompt: {test['prompt'][:50]}...")

        # Unconstrained
        result_u = tester.generate_unconstrained.remote(test["prompt"], max_tokens)
        status_u = "✓" if result_u["is_valid"] else "✗"
        print(f"    Unconstrained: {status_u} ({result_u['latency_ms']:.0f}ms, {result_u['tokens']} tokens)")
        if not result_u["is_valid"]:
            print(f"      Error: {result_u['error']}")
        else:
            results_by_tier[tier]["unconstrained"] += 1
        unconstrained_latency.append(result_u["latency_ms"])

        # Ananke
        result_a = tester.generate_ananke.remote(test["prompt"], max_tokens)
        status_a = "✓" if result_a["is_valid"] else "✗"
        print(f"    Ananke:        {status_a} ({result_a['latency_ms']:.0f}ms, {result_a['tokens']} tokens)")
        if not result_a["is_valid"]:
            print(f"      Error: {result_a['error']}")
        else:
            results_by_tier[tier]["ananke"] += 1
        ananke_latency.append(result_a["latency_ms"])

        # Show generated code (first 150 chars)
        print(f"\n    --- Unconstrained output ---")
        for line in result_u['text'][:200].split('\n')[:6]:
            print(f"      {line}")
        print(f"\n    --- Ananke output ---")
        for line in result_a['text'][:200].split('\n')[:6]:
            print(f"      {line}")

    # Phase 2: Stress test at high temperature
    print("\n\n[PHASE 2] High-Temperature Stress Test")
    print("-" * 50)
    print("Testing grammar enforcement under high uncertainty (temp=0.9)")
    print("Prompt: Complete a function with nested control flow")
    print("Max tokens: 200 (realistic function completion)")

    stress_result = tester.stress_test.remote(
        prompt="def find_max_subarray(nums: list[int]) -> int:\n    \"\"\"Find maximum sum of contiguous subarray (Kadane's algorithm).\"\"\"",
        num_samples=10,
        temperature=0.9,
        max_tokens=200,
    )

    print(f"\n    Unconstrained: {stress_result['unconstrained']['valid']}/10 valid ({stress_result['unconstrained']['valid']*10}%)")
    print(f"    Ananke:        {stress_result['ananke']['valid']}/10 valid ({stress_result['ananke']['valid']*10}%)")

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n  Results by Tier:")
    for tier, data in sorted(results_by_tier.items()):
        total = data["total"]
        u_pct = data["unconstrained"] / total * 100 if total > 0 else 0
        a_pct = data["ananke"] / total * 100 if total > 0 else 0
        print(f"    {tier:12s}: Unconstrained {data['unconstrained']}/{total} ({u_pct:.0f}%) | Ananke {data['ananke']}/{total} ({a_pct:.0f}%)")

    total_tests = len(TEST_PROMPTS)
    total_u = sum(d["unconstrained"] for d in results_by_tier.values())
    total_a = sum(d["ananke"] for d in results_by_tier.values())

    print(f"\n  Overall Syntax Validity:")
    print(f"    Unconstrained: {total_u}/{total_tests} ({total_u/total_tests*100:.0f}%)")
    print(f"    Ananke:        {total_a}/{total_tests} ({total_a/total_tests*100:.0f}%)")

    print(f"\n  Stress Test (temp=0.9, 200 tokens):")
    stress_u = stress_result['unconstrained']
    stress_a = stress_result['ananke']
    print(f"    Unconstrained: {stress_u['valid']}/10 ({stress_u['valid']*10}%)")
    print(f"    Ananke:        {stress_a['valid']}/10 ({stress_a['valid']*10}%)")

    print(f"\n  Latency Analysis:")
    print(f"    Unconstrained avg: {sum(unconstrained_latency)/len(unconstrained_latency):.0f}ms")
    print(f"    Ananke avg:        {sum(ananke_latency)/len(ananke_latency):.0f}ms")
    if sum(unconstrained_latency) > 0:
        overhead = (sum(ananke_latency) - sum(unconstrained_latency)) / sum(unconstrained_latency) * 100
        print(f"    Ananke overhead:   {overhead:+.1f}%")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
