"""Quick diagnostic to see what outputs failing tests generate."""

import os
import sys
import modal

app = modal.App("diagnose-layer1")

@app.local_entrypoint()
def main():
    # Add repo root to path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    # Connect to server
    print("Connecting to server...")
    Qwen3CoderAnanke = modal.Cls.from_name("qwen3-coder-ananke", "Qwen3CoderAnanke")
    server = Qwen3CoderAnanke()

    # Quick health check
    health = server.health.remote()
    print(f"Server status: {health['status']}")

    # Test 1: Email Regex
    print("\n" + "=" * 60)
    print("TEST 1: Email Regex")
    print("=" * 60)
    print("Constraint: regex: [a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}")
    print("Prompt: 'Generate a valid email address:'")

    for i in range(3):
        result = server.generate_constrained.remote(
            prompt="Generate a valid email address:",
            constraint_spec={"regex": r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"},
            max_tokens=50,
            temperature=0.3
        )
        output = result.get("text", "") if isinstance(result, dict) else str(result)
        print(f"  Output {i+1}: '{output[:100]}'")

    # Test 2: Type Domain Blocking
    print("\n" + "=" * 60)
    print("TEST 2: Type Domain Blocking")
    print("=" * 60)
    print("Constraint: regex + type_bindings + expected_type=int")
    print("Prompt: 'x: int = '")

    for i in range(3):
        result = server.generate_constrained.remote(
            prompt="x: int = ",
            constraint_spec={
                "language": "python",
                "regex": r"[a-z_][a-z0-9_]*",
                "type_bindings": [
                    {"name": "x", "type_expr": "int", "scope": "local"},
                    {"name": "count", "type_expr": "int", "scope": "local"},
                    {"name": "name", "type_expr": "str", "scope": "local"},
                ],
                "expected_type": "int"
            },
            max_tokens=20,
            temperature=0.3
        )
        output = result.get("text", "") if isinstance(result, dict) else str(result)
        print(f"  Output {i+1}: '{output}'")

    # Test 3: Type Domain Control (no type constraint)
    print("\n" + "=" * 60)
    print("TEST 3: Type Domain Control (no TypeDomain)")
    print("=" * 60)
    print("Constraint: regex only")
    print("Prompt: 'x: int = '")

    for i in range(3):
        result = server.generate_constrained.remote(
            prompt="x: int = ",
            constraint_spec={
                "language": "python",
                "regex": r"[a-z_][a-z0-9_]*",
            },
            max_tokens=20,
            temperature=0.3
        )
        output = result.get("text", "") if isinstance(result, dict) else str(result)
        print(f"  Output {i+1}: '{output}'")

    # Test 4: Import Domain Blocking
    print("\n" + "=" * 60)
    print("TEST 4: Import Domain Blocking")
    print("=" * 60)
    print("Constraint: regex + forbidden_imports")
    print("Prompt: 'import '")

    for i in range(3):
        result = server.generate_constrained.remote(
            prompt="import ",
            constraint_spec={
                "language": "python",
                "regex": r"[a-z_][a-z0-9_]*",
                "forbidden_imports": ["os", "subprocess", "sys"],
            },
            max_tokens=20,
            temperature=0.3
        )
        output = result.get("text", "") if isinstance(result, dict) else str(result)
        print(f"  Output {i+1}: '{output}'")

    # Test 5: Import Domain Control (no import constraint)
    print("\n" + "=" * 60)
    print("TEST 5: Import Domain Control")
    print("=" * 60)
    print("Constraint: regex only")
    print("Prompt: 'import '")

    for i in range(3):
        result = server.generate_constrained.remote(
            prompt="import ",
            constraint_spec={
                "language": "python",
                "regex": r"[a-z_][a-z0-9_]*",
            },
            max_tokens=20,
            temperature=0.3
        )
        output = result.get("text", "") if isinstance(result, dict) else str(result)
        print(f"  Output {i+1}: '{output}'")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
