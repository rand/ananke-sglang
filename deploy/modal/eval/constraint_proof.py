"""Constraint Proof - Definitively test if Ananke constraints work.

This test proves constraint enforcement by:
1. Using regex that FORCES specific output format
2. Comparing constrained vs unconstrained on same prompt
3. Using a prompt where the model would naturally NOT produce the constrained format

If constraints work: constrained output matches regex, unconstrained doesn't
If constraints don't work: both outputs are similar (model's natural output)

Run:
    modal run deploy/modal/eval/constraint_proof.py
"""

import json
import re
import time
from typing import Any

import modal

DEPLOYED_APP = "qwen3-coder-ananke"
DEPLOYED_CLASS = "Qwen3CoderAnanke"

app = modal.App("constraint-proof")


@app.function(timeout=600)
def prove_constraints() -> dict:
    """Prove that Ananke constraints are being enforced."""
    print("=" * 70)
    print("CONSTRAINT PROOF TEST")
    print("=" * 70)

    # Connect to deployed model
    print("\nConnecting to deployed model...")
    Qwen3CoderAnanke = modal.Cls.from_name(DEPLOYED_APP, DEPLOYED_CLASS)
    server = Qwen3CoderAnanke()

    health = server.health.remote()
    print(f"Connected: {health}")

    results = []

    # ==========================================================================
    # TEST 1: JSON Schema forces structured output
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: JSON Schema Constraint")
    print("=" * 70)

    prompt1 = "Tell me about the weather today."
    schema1 = json.dumps({
        "type": "object",
        "properties": {
            "temperature": {"type": "number"},
            "conditions": {"type": "string"},
            "humidity": {"type": "number"}
        },
        "required": ["temperature", "conditions", "humidity"]
    })

    print(f"Prompt: {prompt1}")
    print(f"Schema forces: JSON with temperature, conditions, humidity fields")

    # Constrained
    print("\nConstrained output:")
    try:
        t0 = time.time()
        resp = server.generate_constrained.remote(
            prompt=prompt1,
            constraint_spec={
                "json_schema": schema1,
            },
            max_tokens=200,
            temperature=0.7,
        )
        constrained_out = resp.get("text", "")
        t1 = time.time()
        print(f"  Output: {constrained_out[:200]}")
        print(f"  Time: {t1-t0:.2f}s")

        # Validate JSON
        try:
            obj = json.loads(constrained_out.strip())
            has_fields = all(k in obj for k in ["temperature", "conditions", "humidity"])
            print(f"  Valid JSON: True, Has required fields: {has_fields}")
            constrained_valid = has_fields
        except:
            print(f"  Valid JSON: False")
            constrained_valid = False
    except Exception as e:
        print(f"  ERROR: {e}")
        constrained_out = ""
        constrained_valid = False

    # Unconstrained
    print("\nUnconstrained output:")
    try:
        t0 = time.time()
        unconstrained_out = server.generate.remote(
            prompt=prompt1,
            max_tokens=200,
            temperature=0.7,
        )
        t1 = time.time()
        print(f"  Output: {unconstrained_out[:200]}")
        print(f"  Time: {t1-t0:.2f}s")

        # Validate JSON
        try:
            obj = json.loads(unconstrained_out.strip())
            has_fields = all(k in obj for k in ["temperature", "conditions", "humidity"])
            print(f"  Valid JSON: True, Has required fields: {has_fields}")
            unconstrained_valid = has_fields
        except:
            print(f"  Valid JSON: False (expected - natural language response)")
            unconstrained_valid = False
    except Exception as e:
        print(f"  ERROR: {e}")
        unconstrained_out = ""
        unconstrained_valid = False

    results.append({
        "test": "json_schema",
        "constrained_valid": constrained_valid,
        "unconstrained_valid": unconstrained_valid,
        "proves_constraint": constrained_valid and not unconstrained_valid,
    })

    # ==========================================================================
    # TEST 2: Regex forces digit-only output
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Regex Constraint (digits only)")
    print("=" * 70)

    prompt2 = "What is 2+2?"
    regex2 = r"[0-9]+"

    print(f"Prompt: {prompt2}")
    print(f"Regex forces: Only digits (no text like 'four' or '2+2=4')")

    # Constrained
    print("\nConstrained output:")
    try:
        t0 = time.time()
        resp = server.generate_constrained.remote(
            prompt=prompt2,
            constraint_spec={
                "regex": regex2,
            },
            max_tokens=10,
            temperature=0.7,
        )
        constrained_out = resp.get("text", "")
        t1 = time.time()
        print(f"  Output: '{constrained_out}'")
        print(f"  Time: {t1-t0:.2f}s")
        constrained_valid = bool(re.fullmatch(regex2, constrained_out.strip()))
        print(f"  Matches regex: {constrained_valid}")
    except Exception as e:
        print(f"  ERROR: {e}")
        constrained_out = ""
        constrained_valid = False

    # Unconstrained
    print("\nUnconstrained output:")
    try:
        t0 = time.time()
        unconstrained_out = server.generate.remote(
            prompt=prompt2,
            max_tokens=10,
            temperature=0.7,
        )
        t1 = time.time()
        print(f"  Output: '{unconstrained_out}'")
        print(f"  Time: {t1-t0:.2f}s")
        unconstrained_valid = bool(re.fullmatch(regex2, unconstrained_out.strip()))
        print(f"  Matches regex: {unconstrained_valid} (expected False - natural response)")
    except Exception as e:
        print(f"  ERROR: {e}")
        unconstrained_out = ""
        unconstrained_valid = False

    results.append({
        "test": "regex_digits",
        "constrained_valid": constrained_valid,
        "unconstrained_valid": unconstrained_valid,
        "proves_constraint": constrained_valid and not unconstrained_valid,
    })

    # ==========================================================================
    # TEST 3: Type bindings with type checking
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Type Bindings")
    print("=" * 70)

    prompt3 = "# Complete: return the sum\ndef add(a, b):\n    return "

    print(f"Prompt: {prompt3}")
    print(f"Type bindings: a:int, b:int, expected return: int")

    # Constrained with type info
    print("\nConstrained output (with type context):")
    try:
        t0 = time.time()
        resp = server.generate_constrained.remote(
            prompt=prompt3,
            constraint_spec={
                "language": "python",
                "type_bindings": [
                    {"name": "a", "type_expr": "int"},
                    {"name": "b", "type_expr": "int"},
                ],
                "expected_type": "int",
            },
            max_tokens=50,
            temperature=0.7,
        )
        constrained_out = resp.get("text", "")
        t1 = time.time()
        print(f"  Output: '{constrained_out}'")
        print(f"  Time: {t1-t0:.2f}s")
    except Exception as e:
        print(f"  ERROR: {e}")
        constrained_out = ""

    # Unconstrained
    print("\nUnconstrained output:")
    try:
        t0 = time.time()
        unconstrained_out = server.generate.remote(
            prompt=prompt3,
            max_tokens=50,
            temperature=0.7,
        )
        t1 = time.time()
        print(f"  Output: '{unconstrained_out}'")
        print(f"  Time: {t1-t0:.2f}s")
    except Exception as e:
        print(f"  ERROR: {e}")
        unconstrained_out = ""

    # Both should produce "a + b" - this test shows type context is provided
    # but doesn't prove enforcement without syntax constraints
    results.append({
        "test": "type_bindings",
        "constrained_output": constrained_out[:50],
        "unconstrained_output": unconstrained_out[:50],
        "note": "Type bindings provide context; enforcement requires syntax constraint",
    })

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in results:
        print(f"\n{r['test']}:")
        if "proves_constraint" in r:
            if r["proves_constraint"]:
                print(f"  PROVED: Constraint enforced (constrained={r['constrained_valid']}, unconstrained={r['unconstrained_valid']})")
            else:
                print(f"  NOT PROVED: constrained={r.get('constrained_valid')}, unconstrained={r.get('unconstrained_valid')}")
        else:
            print(f"  {r.get('note', 'No validation')}")

    proved_count = sum(1 for r in results if r.get("proves_constraint", False))
    print(f"\n{'='*70}")
    print(f"CONSTRAINTS PROVED WORKING: {proved_count}/{len([r for r in results if 'proves_constraint' in r])}")
    print(f"{'='*70}")

    return {"results": results, "proved": proved_count}


@app.local_entrypoint()
def main():
    result = prove_constraints.remote()
    print(f"\nResult: {result}")
