"""Diagnose constraint enforcement.

Quick test to verify:
1. Does json_schema work when passed directly?
2. Does regex work when passed directly?
3. Does constraint_spec work with json_schema inside?
"""

import json
import modal

DEPLOYED_APP = "qwen3-coder-ananke"
DEPLOYED_CLASS = "Qwen3CoderAnanke"

app = modal.App("diagnose-constraints")

image = modal.Image.debian_slim().pip_install("requests")


@app.function(timeout=300, image=image)
def diagnose():
    """Run diagnostic tests."""
    print("=" * 60)
    print("CONSTRAINT DIAGNOSIS")
    print("=" * 60)

    # Connect to deployed model
    Qwen3CoderAnanke = modal.Cls.from_name(DEPLOYED_APP, DEPLOYED_CLASS)
    server = Qwen3CoderAnanke()

    health = server.health.remote()
    print(f"Server healthy: {health}")

    # Test 1: Direct regex parameter (standard SGLang)
    print("\n" + "-" * 60)
    print("TEST 1: Direct regex parameter (standard SGLang)")
    print("-" * 60)

    result = server.completions.remote({
        "model": "default",
        "prompt": "What is 5+5?",
        "max_tokens": 10,
        "regex": "[0-9]+",
    })
    text = result.get("choices", [{}])[0].get("text", "")
    print(f"Prompt: 'What is 5+5?'")
    print(f"Regex: '[0-9]+'")
    print(f"Output: '{text}'")
    print(f"Valid (digits only): {text.strip().isdigit()}")

    # Test 2: Direct json_schema parameter (standard SGLang)
    print("\n" + "-" * 60)
    print("TEST 2: Direct json_schema parameter (standard SGLang)")
    print("-" * 60)

    schema = json.dumps({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    })

    result = server.completions.remote({
        "model": "default",
        "prompt": "Generate a JSON user:",
        "max_tokens": 100,
        "json_schema": schema,
    })
    text = result.get("choices", [{}])[0].get("text", "")
    print(f"Prompt: 'Generate a JSON user:'")
    print(f"Output: '{text[:150]}'")
    try:
        obj = json.loads(text.strip())
        valid = "name" in obj and "age" in obj
        print(f"Valid JSON with fields: {valid}")
    except:
        print(f"Valid JSON: False")

    # Test 3: constraint_spec with regex
    print("\n" + "-" * 60)
    print("TEST 3: constraint_spec with regex")
    print("-" * 60)

    result = server.completions.remote({
        "model": "default",
        "prompt": "What is 7+7?",
        "max_tokens": 10,
        "constraint_spec": {
            "regex": "[0-9]+",
        },
    })
    text = result.get("choices", [{}])[0].get("text", "")
    print(f"Prompt: 'What is 7+7?'")
    print(f"constraint_spec.regex: '[0-9]+'")
    print(f"Output: '{text}'")
    print(f"Valid (digits only): {text.strip().isdigit()}")

    # Test 4: constraint_spec with json_schema
    print("\n" + "-" * 60)
    print("TEST 4: constraint_spec with json_schema")
    print("-" * 60)

    result = server.completions.remote({
        "model": "default",
        "prompt": "Generate a JSON user:",
        "max_tokens": 100,
        "constraint_spec": {
            "json_schema": schema,
        },
    })
    text = result.get("choices", [{}])[0].get("text", "")
    print(f"Prompt: 'Generate a JSON user:'")
    print(f"Output: '{text[:150]}'")
    try:
        obj = json.loads(text.strip())
        valid = "name" in obj and "age" in obj
        print(f"Valid JSON with fields: {valid}")
    except:
        print(f"Valid JSON: False")

    # Test 5: Unconstrained (baseline)
    print("\n" + "-" * 60)
    print("TEST 5: Unconstrained baseline")
    print("-" * 60)

    result = server.completions.remote({
        "model": "default",
        "prompt": "What is 5+5?",
        "max_tokens": 10,
    })
    text = result.get("choices", [{}])[0].get("text", "")
    print(f"Prompt: 'What is 5+5?'")
    print(f"Output (no constraint): '{text}'")
    print(f"Is digits only: {text.strip().isdigit()}")

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)

    return "Done"


@app.local_entrypoint()
def main():
    result = diagnose.remote()
    print(f"\n{result}")
