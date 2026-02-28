"""Simple test for constraint_spec."""

import modal

DEPLOYED_APP = "qwen3-coder-ananke"
DEPLOYED_CLASS = "Qwen3CoderAnanke"

app = modal.App("test-constraint-spec")

image = modal.Image.debian_slim().pip_install("requests")


@app.function(timeout=300, image=image)
def test():
    """Test constraint_spec."""
    import requests
    import json

    # Get the deployed server URL
    Qwen3CoderAnanke = modal.Cls.from_name(DEPLOYED_APP, DEPLOYED_CLASS)
    server = Qwen3CoderAnanke()

    # Test health first
    health = server.health.remote()
    print(f"Health: {health}")

    # Test 1: Direct regex (known working)
    print("\n=== Test 1: Direct regex ===")
    try:
        result = server.completions.remote({
            "model": "default",
            "prompt": "Number: ",
            "max_tokens": 5,
            "regex": "[0-9]+",
        })
        print(f"Result: {result['choices'][0]['text']}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: constraint_spec with regex
    print("\n=== Test 2: constraint_spec with regex ===")
    try:
        result = server.completions.remote({
            "model": "default",
            "prompt": "Number: ",
            "max_tokens": 5,
            "constraint_spec": {
                "regex": "[0-9]+",
            },
        })
        print(f"Result: {result['choices'][0]['text']}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: constraint_spec with full spec (version, language)
    print("\n=== Test 3: constraint_spec with full spec ===")
    try:
        result = server.completions.remote({
            "model": "default",
            "prompt": "Number: ",
            "max_tokens": 5,
            "constraint_spec": {
                "version": "1.0",
                "regex": "[0-9]+",
                "language": "python",
            },
        })
        print(f"Result: {result['choices'][0]['text']}")
    except Exception as e:
        print(f"Error: {e}")

    return "Done"


@app.local_entrypoint()
def main():
    result = test.remote()
    print(f"\n{result}")
