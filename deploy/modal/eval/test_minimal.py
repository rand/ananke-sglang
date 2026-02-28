"""Minimal test to debug Modal local_entrypoint import issue."""

import os
import sys

import modal

app = modal.App("test-minimal-eval")

@app.local_entrypoint()
def main():
    print("Step 1: In local_entrypoint", flush=True)

    # Add repo root to path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    print(f"Step 2: Added {REPO_ROOT} to sys.path", flush=True)

    print("Step 3: About to import statistics...", flush=True)
    try:
        from deploy.modal.eval.statistics import wilson_score_interval
        print("Step 4: statistics imported OK", flush=True)
    except Exception as e:
        print(f"Step 4: FAILED: {e}", flush=True)
        return

    print("Step 5: About to import layer1...", flush=True)
    try:
        from deploy.modal.eval.tests.layer1_mechanism_tests import get_layer1_tests
        print("Step 6: layer1 imported OK", flush=True)
    except Exception as e:
        print(f"Step 6: FAILED: {e}", flush=True)
        return

    print("Step 7: About to import layered_eval...", flush=True)
    try:
        from deploy.modal.eval.layered_eval import LayeredEvaluator
        print("Step 8: layered_eval imported OK", flush=True)
    except Exception as e:
        print(f"Step 8: FAILED: {e}", flush=True)
        return

    print("Step 9: All imports successful!", flush=True)

    # Try connecting to the server
    print("Step 10: Connecting to server...", flush=True)
    try:
        Qwen3CoderAnanke = modal.Cls.from_name("qwen3-coder-ananke", "Qwen3CoderAnanke")
        server = Qwen3CoderAnanke()
        print("Step 11: Server connection created", flush=True)
    except Exception as e:
        print(f"Step 11: FAILED: {e}", flush=True)
        return

    print("Step 12: Checking server health...", flush=True)
    try:
        health = server.health.remote()
        print(f"Step 13: Health response: {health}", flush=True)
    except Exception as e:
        print(f"Step 13: FAILED: {e}", flush=True)
        return

    print("All steps completed successfully!", flush=True)


if __name__ == "__main__":
    print("Use: modal run deploy/modal/eval/test_minimal.py")
