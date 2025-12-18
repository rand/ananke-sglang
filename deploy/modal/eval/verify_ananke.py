"""Verify Ananke backend is actually being used."""

import modal

DEPLOYED_APP = "qwen3-coder-ananke"
DEPLOYED_CLASS = "Qwen3CoderAnanke"

app = modal.App("verify-ananke")


@app.function(timeout=120)
def check_server_config():
    """Check the server configuration from inside a Modal container."""
    import requests

    # Connect to qwen3-coder-ananke
    Qwen3CoderAnanke = modal.Cls.from_name(DEPLOYED_APP, DEPLOYED_CLASS)
    server = Qwen3CoderAnanke()

    # Get server info
    print("Checking server configuration...")

    # Try to get server info endpoint
    try:
        result = server.health.remote()
        print(f"Health: {result}")
    except Exception as e:
        print(f"Health error: {e}")

    return "Done"


@app.function(
    image=modal.Image.debian_slim().pip_install("requests"),
    timeout=300,
)
def inspect_deployed_server():
    """Inspect the deployed server directly."""
    import subprocess
    import requests

    # Connect to the deployed class
    Qwen3CoderAnanke = modal.Cls.from_name(DEPLOYED_APP, DEPLOYED_CLASS)
    server = Qwen3CoderAnanke()

    print("="*60)
    print("VERIFYING ANANKE DEPLOYMENT")
    print("="*60)

    # 1. Check if Ananke modules are importable on the server
    print("\n1. Checking Ananke module availability...")
    try:
        # This runs a method that will show us the server state
        result = server.completions.remote({
            "model": "default",
            "prompt": "# Test",
            "max_tokens": 1,
        })
        print(f"   Basic completion works: {bool(result)}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # 2. Test with constraint_spec and check if it's processed
    print("\n2. Testing constraint_spec processing...")
    try:
        # With constraint_spec
        result_constrained = server.completions.remote({
            "model": "default",
            "prompt": "def test():",
            "max_tokens": 50,
            "constraint_spec": {
                "language": "python",
            },
        })
        print(f"   Constrained response keys: {result_constrained.keys()}")
        text = result_constrained.get("choices", [{}])[0].get("text", "")
        print(f"   Generated text: {text[:100]}...")
    except Exception as e:
        print(f"   ERROR with constraint_spec: {e}")

    # 3. Test with INVALID constraint to see if it's validated
    print("\n3. Testing with invalid constraint (should error if Ananke validates)...")
    try:
        result_invalid = server.completions.remote({
            "model": "default",
            "prompt": "def test():",
            "max_tokens": 50,
            "constraint_spec": {
                "language": "not_a_real_language_xyz",
            },
        })
        print(f"   WARNING: Invalid language was accepted - Ananke may not be active")
        print(f"   Response: {result_invalid}")
    except Exception as e:
        print(f"   Good - invalid constraint rejected: {str(e)[:100]}")

    # 4. Check server startup command via environment
    print("\n4. Checking environment variables on server...")
    # We can't directly access the server's env, but we can check what was configured

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

    return "Done"


@app.local_entrypoint()
def main():
    result = inspect_deployed_server.remote()
    print(f"\nResult: {result}")
