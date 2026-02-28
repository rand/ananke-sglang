"""
Example: Constrained Code Generation with SGLang Ananke on Modal

This example demonstrates how to use the Ananke constrained generation
backend to generate syntactically and semantically valid code.

Usage:
    python constrained_generation.py
"""

import modal

# Connect to the deployed app
app = modal.App.lookup("sglang-ananke")
SGLangAnanke = modal.Cls.lookup("sglang-ananke", "SGLangAnanke")


def generate_python_function():
    """Generate a Python function with syntax and type constraints."""
    sglang = SGLangAnanke()

    result = sglang.generate_constrained.remote(
        prompt='''def calculate_average(numbers: list[float]) -> float:
    """Calculate the average of a list of numbers."""
''',
        constraint_spec={
            "language": "python",
            "domains": ["syntax", "types"],
            "context": {
                "imports": ["typing"],
                "type_context": {
                    "numbers": "list[float]",
                    "return": "float",
                },
            },
        },
        max_tokens=150,
        temperature=0.7,
    )

    print("Generated Python function:")
    print(result)
    return result


def generate_typescript_interface():
    """Generate a TypeScript interface with type constraints."""
    sglang = SGLangAnanke()

    result = sglang.generate_constrained.remote(
        prompt='''interface UserProfile {
    id: string;
    email: string;
''',
        constraint_spec={
            "language": "typescript",
            "domains": ["syntax", "types"],
        },
        max_tokens=200,
        temperature=0.5,
    )

    print("Generated TypeScript interface:")
    print(result)
    return result


def generate_with_imports():
    """Generate code with import resolution constraints."""
    sglang = SGLangAnanke()

    result = sglang.generate_constrained.remote(
        prompt='''from dataclasses import dataclass
from typing import Optional, List

@dataclass
class User:
''',
        constraint_spec={
            "language": "python",
            "domains": ["syntax", "types", "imports"],
            "context": {
                "imports": [
                    {"module": "dataclasses", "names": ["dataclass"]},
                    {"module": "typing", "names": ["Optional", "List"]},
                ],
            },
        },
        max_tokens=200,
        temperature=0.6,
    )

    print("Generated dataclass with imports:")
    print(result)
    return result


def batch_generation():
    """Batch multiple constrained generations."""
    sglang = SGLangAnanke()

    prompts = [
        ("def add(a: int, b: int) -> int:", {"language": "python", "domains": ["syntax", "types"]}),
        ("def multiply(x: float, y: float) -> float:", {"language": "python", "domains": ["syntax", "types"]}),
        ("def greet(name: str) -> str:", {"language": "python", "domains": ["syntax", "types"]}),
    ]

    results = []
    for prompt, spec in prompts:
        result = sglang.generate_constrained.remote(
            prompt=prompt,
            constraint_spec=spec,
            max_tokens=100,
        )
        results.append(result)

    print("Batch generation results:")
    for i, (prompt, result) in enumerate(zip([p[0] for p in prompts], results)):
        print(f"\n--- Function {i+1} ---")
        print(prompt + result)

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("SGLang Ananke Constrained Generation Examples")
    print("=" * 60)

    print("\n1. Python Function Generation")
    print("-" * 40)
    generate_python_function()

    print("\n2. TypeScript Interface Generation")
    print("-" * 40)
    generate_typescript_interface()

    print("\n3. Python with Import Resolution")
    print("-" * 40)
    generate_with_imports()

    print("\n4. Batch Generation")
    print("-" * 40)
    batch_generation()
