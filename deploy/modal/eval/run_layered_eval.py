"""Run Layered Evaluation of Ananke Constraint System on Modal.

This script runs the principled two-layer evaluation:
- Layer 1: Mechanism verification (do constraints work?)
- Layer 2: Value measurement (do constraints add value?)

Usage:
    # Quick mechanism check only
    modal run deploy/modal/eval/run_layered_eval.py --layer1-only

    # Full evaluation
    modal run deploy/modal/eval/run_layered_eval.py

    # Full evaluation with specific category
    modal run deploy/modal/eval/run_layered_eval.py --category json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

import modal

# Add repo root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# =============================================================================
# Modal App Setup
# =============================================================================

app = modal.App("ananke-layered-eval")

# Warmup parameters
WARMUP_TIMEOUT = 1500  # 25 minutes - matches MODEL_LOAD_TIMEOUT
WARMUP_INITIAL_DELAY = 10
WARMUP_MAX_DELAY = 60


def warm_up_server(server, timeout: int = WARMUP_TIMEOUT) -> bool:
    """Wait for server to be ready with exponential backoff.

    Args:
        server: Modal server instance
        timeout: Maximum wait time in seconds

    Returns:
        True if server is ready, raises RuntimeError if timeout
    """
    print(f"\nWarming up server (timeout: {timeout}s)...")
    print("Note: First cold start may take 15-20 minutes for 60GB model")

    start_time = time.time()
    delay = WARMUP_INITIAL_DELAY
    attempt = 0

    while time.time() - start_time < timeout:
        attempt += 1
        elapsed = time.time() - start_time

        try:
            print(f"  [{elapsed:.0f}s] Attempt {attempt}: checking health...")
            result = server.health.remote()

            if result.get("status") == "healthy":
                print(f"  [{elapsed:.0f}s] Server healthy! Checking readiness...")

                # Try a simple generation to verify full readiness
                ready = server.health_generate.remote()
                if ready.get("ready"):
                    print(f"  [{elapsed:.0f}s] Server ready for generation!")
                    return True
                else:
                    print(f"  [{elapsed:.0f}s] Not ready yet: {ready}")

            elif not result.get("process_alive", True):
                print(f"\n  SERVER PROCESS DIED!")
                print(f"  Recent logs: {result.get('recent_logs', 'N/A')}")
                raise RuntimeError("Server process died during warm-up")

        except Exception as e:
            error_msg = str(e)[:100]
            print(f"  [{elapsed:.0f}s] Connection error (expected during cold start): {error_msg}")

        print(f"  [{elapsed:.0f}s] Waiting {delay}s before next attempt...")
        time.sleep(delay)
        delay = min(delay * 1.5, WARMUP_MAX_DELAY)

    raise RuntimeError(f"Server failed to become ready within {timeout}s")


def create_generate_fn(server):
    """Create a generate function that wraps the Modal server.

    Returns a function with signature:
        generate(prompt, constraint_spec, max_tokens, temperature) -> str
    """
    def generate_fn(
        prompt: str,
        constraint_spec: dict | None,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate text with optional constraints."""
        if constraint_spec is None:
            # Unconstrained generation
            return server.generate.remote(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            # Constrained generation
            result = server.generate_constrained.remote(
                prompt=prompt,
                constraint_spec=constraint_spec,
                max_tokens=max_tokens,
                temperature=temperature
            )
            # Extract text from result dict
            if isinstance(result, dict):
                return result.get("text", "")
            return str(result)

    return generate_fn


# =============================================================================
# Evaluation Entry Points
# =============================================================================

@app.local_entrypoint()
def main(
    layer1_only: bool = False,
    category: Optional[str] = None,
    verbose: bool = True,
    output_file: Optional[str] = None,
):
    """Run layered evaluation.

    Args:
        layer1_only: Only run Layer 1 mechanism tests
        category: Filter tests by category (json, regex, domain, code, security, multilang)
        verbose: Print progress
        output_file: Output file path (auto-generated if not specified)
    """
    # Flush output immediately
    import sys
    sys.stdout.flush()

    print("Starting layered evaluation...", flush=True)

    # Import here to avoid Modal serialization issues
    print("Importing statistics...", flush=True)
    from deploy.modal.eval.statistics import (
        wilson_score_interval, cohens_h, compare_conditions,
        evaluate_test, ConfidenceInterval
    )
    print("  OK", flush=True)

    print("Importing layer1 tests...", flush=True)
    from deploy.modal.eval.tests.layer1_mechanism_tests import (
        MechanismTest, get_layer1_tests
    )
    print("  OK", flush=True)

    print("Importing layer2 tests...", flush=True)
    from deploy.modal.eval.tests.layer2_value_tests import (
        ValueTest, get_layer2_tests
    )
    print("  OK", flush=True)

    print("Importing evaluator...", flush=True)
    from deploy.modal.eval.layered_eval import LayeredEvaluator, save_results
    print("  OK", flush=True)

    print("All imports successful!", flush=True)

    print("=" * 70)
    print("ANANKE LAYERED EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Mode: {'Layer 1 only' if layer1_only else 'Full evaluation'}")
    if category:
        print(f"Category filter: {category}")

    # Connect to deployed server
    print("\nConnecting to deployed model...")
    Qwen3CoderAnanke = modal.Cls.from_name("qwen3-coder-ananke", "Qwen3CoderAnanke")
    server = Qwen3CoderAnanke()

    # Warm up server
    warm_up_server(server)

    # Create generate function wrapper
    generate_fn = create_generate_fn(server)

    # Create evaluator
    evaluator = LayeredEvaluator(generate_fn, verbose=verbose)

    # Run evaluation
    if layer1_only:
        # Layer 1 only
        layer1_results = evaluator.run_layer1(category=category)

        results = {
            "timestamp": datetime.now().isoformat(),
            "mode": "layer1_only",
            "category": category,
            "layer1": layer1_results.to_dict(),
            "summary": {
                "layer1_passed": layer1_results.all_passed,
                "json_passed": layer1_results.json_passed,
                "regex_passed": layer1_results.regex_passed,
                "domain_passed": layer1_results.domain_passed,
            }
        }

        print("\n" + "=" * 70)
        print("LAYER 1 RESULTS")
        print("=" * 70)
        print(f"JSON Schema: {'PASS' if layer1_results.json_passed else 'FAIL'}")
        print(f"Regex: {'PASS' if layer1_results.regex_passed else 'FAIL'}")
        print(f"Domain: {'PASS' if layer1_results.domain_passed else 'FAIL'}")
        print(f"Overall: {'PASS' if layer1_results.all_passed else 'FAIL'}")

    else:
        # Full evaluation
        results = evaluator.run_full(skip_layer2_on_layer1_fail=True)

        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"Layer 1: {'PASS' if results['summary']['layer1_passed'] else 'FAIL'}")
        if results['summary']['layer2_ran']:
            value_str = 'YES' if results['summary']['layer2_value_demonstrated'] else 'NO'
            print(f"Layer 2 Value Demonstrated: {value_str}")
        print(f"Overall Success: {'YES' if results['summary']['overall_success'] else 'NO'}")

    # Save results
    if output_file is None:
        timestamp = int(time.time())
        output_file = f"layered_eval_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Return exit code based on results
    if layer1_only:
        return 0 if results['summary']['layer1_passed'] else 1
    else:
        return 0 if results['summary']['overall_success'] else 1


@app.function()
def run_layer1_remote(category: Optional[str] = None) -> dict:
    """Run Layer 1 evaluation remotely on Modal.

    This function runs the evaluation in the Modal cloud,
    avoiding local dependency on the server.
    """
    from deploy.modal.eval.layered_eval import LayeredEvaluator

    # Connect to server within Modal
    Qwen3CoderAnanke = modal.Cls.from_name("qwen3-coder-ananke", "Qwen3CoderAnanke")
    server = Qwen3CoderAnanke()

    # Warm up
    warm_up_server(server)

    # Create evaluator
    generate_fn = create_generate_fn(server)
    evaluator = LayeredEvaluator(generate_fn, verbose=True)

    # Run Layer 1
    results = evaluator.run_layer1(category=category)
    return results.to_dict()


@app.function()
def run_full_remote() -> dict:
    """Run full evaluation remotely on Modal."""
    from deploy.modal.eval.layered_eval import LayeredEvaluator

    # Connect to server within Modal
    Qwen3CoderAnanke = modal.Cls.from_name("qwen3-coder-ananke", "Qwen3CoderAnanke")
    server = Qwen3CoderAnanke()

    # Warm up
    warm_up_server(server)

    # Create evaluator
    generate_fn = create_generate_fn(server)
    evaluator = LayeredEvaluator(generate_fn, verbose=True)

    # Run full evaluation
    return evaluator.run_full()


if __name__ == "__main__":
    # For local testing with argparse
    parser = argparse.ArgumentParser(description="Run Ananke Layered Evaluation")
    parser.add_argument("--layer1-only", action="store_true", help="Only run Layer 1")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")

    args = parser.parse_args()

    # This would need Modal's local entrypoint mechanism
    print("Use: modal run deploy/modal/eval/run_layered_eval.py")
    print("  --layer1-only    Only run Layer 1 mechanism tests")
    print("  --category X     Filter by category (json, regex, domain, code, security)")
