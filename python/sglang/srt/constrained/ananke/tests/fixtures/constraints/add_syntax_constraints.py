#!/usr/bin/env python3
"""Script to add syntax constraints to all domain-only examples.

This script systematically adds regex, json_schema, or ebnf constraints
to examples that currently only have domain context (type_bindings, etc.)
but no syntax constraints.

The backend requires syntax constraints for grammar creation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.fixtures.constraints import get_all_examples
from tests.fixtures.constraints.base import ConstraintExample


def analyze_outputs(valid: List[str], invalid: List[str]) -> Dict[str, str]:
    """Analyze valid/invalid outputs to determine appropriate constraint type."""
    info = {
        "constraint_type": "regex",  # Default
        "pattern": "",
        "notes": [],
    }

    if not valid:
        info["notes"].append("No valid outputs to analyze")
        return info

    first_valid = valid[0]

    # Detect import statements
    if re.match(r'^(import|from)\s+', first_valid):
        # Extract allowed modules from valid outputs
        allowed = set()
        for v in valid:
            m = re.match(r'^(?:import|from)\s+(\w+(?:\.\w+)*)', v)
            if m:
                allowed.add(m.group(1).split('.')[0])

        if allowed:
            pattern = r'^(?:import|from)\s+(' + '|'.join(sorted(allowed)) + r')(?:\s|\.|$)'
            info["pattern"] = pattern
            info["notes"].append(f"Import constraint: {allowed}")
        return info

    # Detect function definitions
    if first_valid.startswith('func ') or first_valid.startswith('def '):
        info["constraint_type"] = "ebnf"
        # Extract function signature pattern
        if 'def ' in first_valid:
            # Python function
            m = re.match(r'def\s+(\w+)\s*\(([^)]*)\)', first_valid)
            if m:
                func_name = m.group(1)
                info["pattern"] = f'root ::= "def {func_name}(" params ")" rest'
        elif 'func ' in first_valid:
            # Go function
            m = re.match(r'func\s+(?:\([^)]+\)\s+)?(\w+)', first_valid)
            if m:
                func_name = m.group(1)
                info["pattern"] = f'root ::= "func " [receiver] "{func_name}" "(" params ")" returns body'
        return info

    # Detect return statements
    if first_valid.startswith('return '):
        # Check what's being returned
        if '[' in first_valid and ']' in first_valid:
            info["pattern"] = r'^return\s+\[.*\]$'
            info["notes"].append("List return")
        elif first_valid.startswith('return {'):
            info["pattern"] = r'^return\s+\{.*\}$'
            info["notes"].append("Dict/set return")
        else:
            # Generic return pattern
            info["pattern"] = r'^return\s+.+$'
        return info

    # Detect if statements
    if first_valid.startswith('if '):
        info["constraint_type"] = "ebnf"
        info["pattern"] = 'root ::= "if " condition ":" body'
        return info

    # Detect method calls with channels (Go)
    if '<-' in first_valid:
        info["pattern"] = r'^.*<-.*$'
        info["notes"].append("Channel operation")
        return info

    # Detect scope/launch patterns (Kotlin coroutines)
    if 'scope.launch' in first_valid or 'launch {' in first_valid:
        info["pattern"] = r'.*(?:scope\.launch|launch\s*\{).*'
        info["notes"].append("Coroutine launch")
        return info

    # Default: create pattern from first valid output structure
    # Escape special regex chars and create a loose pattern
    first_words = first_valid.split()[:3]
    if first_words:
        pattern_start = r'^\s*' + re.escape(first_words[0])
        info["pattern"] = pattern_start + r'.*'

    return info


def generate_constraint_for_example(example: ConstraintExample) -> Optional[Tuple[str, str]]:
    """Generate appropriate syntax constraint for an example.

    Returns:
        Tuple of (constraint_type, constraint_value) or None if already has constraint
    """
    spec = example.spec

    # Skip if already has syntax constraint
    if spec.json_schema or spec.regex or spec.ebnf:
        return None

    analysis = analyze_outputs(example.valid_outputs, example.invalid_outputs)

    if not analysis["pattern"]:
        return None

    return (analysis["constraint_type"], analysis["pattern"])


def main():
    """Analyze all examples and report needed constraints."""
    examples = get_all_examples()

    domain_only = []
    with_syntax = []

    for e in examples:
        if e.spec.json_schema or e.spec.regex or e.spec.ebnf:
            with_syntax.append(e)
        else:
            domain_only.append(e)

    print(f"Total examples: {len(examples)}")
    print(f"With syntax constraint: {len(with_syntax)}")
    print(f"Domain-only (need constraint): {len(domain_only)}")
    print()

    # Group by file
    by_file: Dict[str, List[Tuple[ConstraintExample, Optional[Tuple[str, str]]]]] = {}

    for e in domain_only:
        # Determine file path
        lang = e.language
        domain = e.domain

        # Handle special domains
        if domain in ("coroutines", "comptime"):
            file_key = f"{lang}/{domain}"
        else:
            file_key = f"{lang}/{domain}"

        if file_key not in by_file:
            by_file[file_key] = []

        constraint = generate_constraint_for_example(e)
        by_file[file_key].append((e, constraint))

    # Print summary by file
    print("=" * 60)
    print("CONSTRAINTS NEEDED BY FILE")
    print("=" * 60)

    for file_key, items in sorted(by_file.items()):
        print(f"\n{file_key}.py ({len(items)} examples):")
        for e, constraint in items:
            if constraint:
                ctype, cval = constraint
                cval_short = cval[:50] + "..." if len(cval) > 50 else cval
                print(f"  {e.id}: {ctype} = {cval_short}")
            else:
                print(f"  {e.id}: NEEDS MANUAL CONSTRAINT")


if __name__ == "__main__":
    main()
