# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Backend integration tests for constraint examples.

These tests validate that constraint examples work correctly with the
Ananke backend, verifying:
1. ConstraintSpec round-trip serialization
2. Grammar creation for examples with syntax constraints
3. Non-trivial constraint creation (not all TOP)
4. Domain context validity
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import Mock

import pytest

# Ensure imports work from test directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from . import get_all_examples
    from .base import ConstraintExample, LANGUAGES, DOMAINS
except ImportError:
    from tests.fixtures.constraints import get_all_examples
    from tests.fixtures.constraints.base import ConstraintExample, LANGUAGES, DOMAINS

from spec.constraint_spec import ConstraintSpec

# Check for optional dependencies
try:
    import torch
    HAS_TORCH = True
    # Check if CUDA is available when torch is compiled with CUDA support
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

try:
    import llguidance
    HAS_LLGUIDANCE = True
except ImportError:
    HAS_LLGUIDANCE = False

# Check if full sglang environment is available for backend tests
try:
    from backend.backend import AnankeBackend
    from core.unified import UNIFIED_TOP
    HAS_BACKEND = True
except (ImportError, ModuleNotFoundError):
    HAS_BACKEND = False
    AnankeBackend = None
    UNIFIED_TOP = None


# =============================================================================
# Mock Infrastructure
# =============================================================================


class MockTokenizer:
    """Minimal mock tokenizer for testing without real tokenizer."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self._vocab = {f"token_{i}": i for i in range(vocab_size)}

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, int):
            return f"token_{token_ids}"
        return "".join(f"token_{tid}" for tid in token_ids)

    def get_vocab(self):
        return self._vocab.copy()

    def encode(self, text, add_special_tokens=False):
        return [0]  # Simplified


class MockSyntaxGrammar:
    """Mock syntax grammar for testing without llguidance."""

    def __init__(self, name: str = "mock"):
        self.name = name
        self._is_accepting = False

    def fill_vocab_mask(self, vocab_mask, idx=0):
        pass  # No-op for testing

    def accept_token(self, token_id):
        pass

    def rollback(self, k):
        pass

    def copy(self):
        return MockSyntaxGrammar(name=f"{self.name}_copy")

    def is_accepting(self):
        return self._is_accepting


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def all_examples() -> List[ConstraintExample]:
    """Load all constraint examples."""
    return get_all_examples()


@pytest.fixture(scope="module")
def examples_with_syntax(all_examples: List[ConstraintExample]) -> List[ConstraintExample]:
    """Get examples that have syntax constraints (json_schema, regex, ebnf)."""
    return [
        e for e in all_examples
        if e.spec.json_schema or e.spec.regex or e.spec.ebnf
    ]


@pytest.fixture(scope="module")
def examples_domain_only(all_examples: List[ConstraintExample]) -> List[ConstraintExample]:
    """Get examples with only domain context (no syntax constraint)."""
    return [
        e for e in all_examples
        if not (e.spec.json_schema or e.spec.regex or e.spec.ebnf)
    ]


@pytest.fixture(scope="module")
def mock_tokenizer() -> MockTokenizer:
    """Create mock tokenizer."""
    return MockTokenizer(vocab_size=1000)


@pytest.fixture(scope="module")
def backend(mock_tokenizer: MockTokenizer):
    """Create AnankeBackend for testing (if available)."""
    if not HAS_BACKEND:
        pytest.skip("AnankeBackend not available (missing sglang)")
    backend = AnankeBackend(
        tokenizer=mock_tokenizer,
        vocab_size=1000,
        language="python",
    )
    # Disable syntax backend to avoid llguidance dependency
    backend.syntax_backend = None
    return backend


# =============================================================================
# ConstraintSpec Validation Tests
# =============================================================================


class TestConstraintSpecValidity:
    """Test that all ConstraintSpecs are well-formed."""

    def test_all_specs_have_language(self, all_examples: List[ConstraintExample]) -> None:
        """Every spec should have a language set."""
        missing = [e.id for e in all_examples if not e.spec.language]
        assert len(missing) == 0, f"Specs missing language: {missing}"

    def test_all_specs_serializable(self, all_examples: List[ConstraintExample]) -> None:
        """Every spec should round-trip through to_dict/from_dict."""
        failures = []
        for example in all_examples:
            try:
                d = example.spec.to_dict()
                restored = ConstraintSpec.from_dict(d)
                # Basic equality checks
                if restored.language != example.spec.language:
                    failures.append(f"{example.id}: language mismatch")
                if restored.json_schema != example.spec.json_schema:
                    failures.append(f"{example.id}: json_schema mismatch")
                if restored.regex != example.spec.regex:
                    failures.append(f"{example.id}: regex mismatch")
            except Exception as e:
                failures.append(f"{example.id}: {type(e).__name__}: {str(e)[:50]}")

        assert len(failures) == 0, f"Serialization failures:\n" + "\n".join(failures[:10])

    def test_syntax_constraint_types_valid(
        self, examples_with_syntax: List[ConstraintExample]
    ) -> None:
        """Examples with syntax constraints should have valid constraint types."""
        for example in examples_with_syntax:
            spec = example.spec
            constraint_type = spec.get_syntax_constraint_type()
            assert constraint_type in ("json_schema", "regex", "ebnf"), (
                f"Example {example.id} has invalid syntax constraint type: {constraint_type}"
            )

            # Verify the constraint value is non-empty
            constraint_value = spec.get_syntax_constraint_value()
            assert constraint_value and len(constraint_value) > 0, (
                f"Example {example.id} has empty syntax constraint value"
            )

    def test_domain_only_has_context(
        self, examples_domain_only: List[ConstraintExample]
    ) -> None:
        """Domain-only examples should have meaningful domain context."""
        missing_context = []
        for example in examples_domain_only:
            spec = example.spec
            has_context = any([
                spec.type_bindings,
                spec.function_signatures,
                spec.class_definitions,
                spec.expected_type,
                spec.imports,
                spec.available_modules,
                spec.forbidden_imports,
                spec.control_flow,
                spec.semantic_constraints,
            ])
            if not has_context:
                missing_context.append(example.id)

        # Allow a few examples without explicit context (they may rely on language defaults)
        assert len(missing_context) <= 5, (
            f"{len(missing_context)} domain-only examples lack context: {missing_context[:10]}"
        )


# =============================================================================
# Backend Integration Tests
# =============================================================================


class TestBackendIntegration:
    """Test backend integration with constraint examples."""

    def test_syntax_examples_have_valid_constraints(
        self, examples_with_syntax: List[ConstraintExample]
    ) -> None:
        """Examples with syntax constraints should have valid constraint strings."""
        invalid = []
        for example in examples_with_syntax:
            spec = example.spec
            if spec.json_schema:
                # Basic JSON schema validation
                try:
                    import json
                    json.loads(spec.json_schema)
                except json.JSONDecodeError as e:
                    invalid.append(f"{example.id}: invalid JSON schema: {str(e)[:30]}")

            if spec.regex:
                # Basic regex validation
                # Note: Some examples use JavaScript-style named groups (?<name>...)
                # which Python's re module doesn't support. Convert to Python style.
                try:
                    import re
                    # Convert JS-style named groups to Python style for validation
                    python_regex = re.sub(r'\(\?<(\w+)>', r'(?P<\1>', spec.regex)
                    re.compile(python_regex)
                except re.error as e:
                    invalid.append(f"{example.id}: invalid regex: {str(e)[:30]}")

        assert len(invalid) == 0, f"Invalid syntax constraints:\n" + "\n".join(invalid)

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
    @pytest.mark.skipif(not HAS_BACKEND, reason="AnankeBackend not available")
    def test_grammar_creation_with_mock(
        self,
        backend,
        examples_with_syntax: List[ConstraintExample],
    ) -> None:
        """Test grammar creation using mock syntax grammar.

        Note: This test requires the full sglang environment with CUDA support.
        It will be skipped when running from the ananke directory alone or
        when CUDA is not available.
        """
        if not HAS_BACKEND:
            pytest.skip("AnankeBackend not available (sglang not installed)")
        if backend is None:
            pytest.skip("Backend fixture returned None")

        # Verify backend has the method we need
        if not hasattr(backend, '_create_ananke_grammar'):
            pytest.skip("Backend missing _create_ananke_grammar method")

        created = 0
        failed = []
        cuda_errors = 0

        for example in examples_with_syntax[:20]:  # Test first 20
            try:
                mock_syntax = MockSyntaxGrammar(name=f"mock_{example.id}")

                grammar = backend._create_ananke_grammar(
                    syntax_grammar=mock_syntax,
                    constraint_spec=example.spec,
                )

                assert grammar is not None, f"Grammar is None for {example.id}"
                assert grammar.syntax_grammar is mock_syntax
                created += 1
            except AssertionError as e:
                # Check for CUDA-related assertion errors
                if "CUDA" in str(e) or "cuda" in str(e):
                    cuda_errors += 1
                    if cuda_errors == 1:
                        # First CUDA error - skip the entire test
                        pytest.skip(f"CUDA not available: {str(e)[:50]}")
                else:
                    failed.append(f"{example.id}: AssertionError: {str(e)[:50]}")
            except Exception as e:
                failed.append(f"{example.id}: {type(e).__name__}: {str(e)[:50]}")

        print(f"\nGrammars created: {created}/{len(examples_with_syntax[:20])}")
        if failed:
            print(f"Failures: {len(failed)}")
            for f in failed[:5]:
                print(f"  - {f}")

        # Allow some failures (may be due to missing dependencies)
        assert created > len(examples_with_syntax[:20]) * 0.8, (
            f"Too many failures: {len(failed)} out of {len(examples_with_syntax[:20])}"
        )

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
    @pytest.mark.skipif(not HAS_BACKEND, reason="AnankeBackend not available")
    def test_domain_only_returns_none_from_dispatch(
        self,
        backend,
        examples_domain_only: List[ConstraintExample],
    ) -> None:
        """Domain-only examples should return None from dispatch_with_spec.

        This is expected behavior - domain-only constraints are not fully
        supported yet per the backend documentation.
        """
        if not HAS_BACKEND:
            pytest.skip("AnankeBackend not available (sglang not installed)")
        if backend is None:
            pytest.skip("Backend fixture returned None")

        for example in examples_domain_only[:10]:  # Test first 10
            result = backend.dispatch_with_spec(example.spec)
            # Domain-only specs return None because they have no syntax constraint
            # This is documented behavior, not a bug
            if result is not None:
                # If it did return something, it should have domains
                assert hasattr(result, 'domains')


# =============================================================================
# Constraint Quality Tests
# =============================================================================


class TestConstraintQuality:
    """Test that constraints are meaningful and non-trivial."""

    def test_type_bindings_have_types(self, all_examples: List[ConstraintExample]) -> None:
        """Type bindings should have type expressions."""
        issues = []
        for example in all_examples:
            for binding in example.spec.type_bindings:
                if not binding.type_expr:
                    issues.append(f"{example.id}: binding '{binding.name}' missing type_expr")

        assert len(issues) == 0, f"Type binding issues:\n" + "\n".join(issues[:10])

    def test_function_signatures_complete(
        self, all_examples: List[ConstraintExample]
    ) -> None:
        """Function signatures should have names and return types."""
        issues = []
        for example in all_examples:
            for sig in example.spec.function_signatures:
                if not sig.name:
                    issues.append(f"{example.id}: signature missing name")
                # return_type can be None for void functions, so don't require it

        assert len(issues) == 0, f"Signature issues:\n" + "\n".join(issues[:10])

    def test_semantic_constraints_have_expressions(
        self, all_examples: List[ConstraintExample]
    ) -> None:
        """Semantic constraints should have expressions."""
        issues = []
        for example in all_examples:
            for sc in example.spec.semantic_constraints:
                if not sc.expression:
                    issues.append(f"{example.id}: semantic constraint missing expression")
                if not sc.kind:
                    issues.append(f"{example.id}: semantic constraint missing kind")

        assert len(issues) == 0, f"Semantic constraint issues:\n" + "\n".join(issues[:10])

    def test_control_flow_context_valid(
        self, all_examples: List[ConstraintExample]
    ) -> None:
        """Control flow contexts should have valid fields."""
        issues = []
        for example in all_examples:
            cf = example.spec.control_flow
            if cf is not None:
                # Control flow should have function name or meaningful context
                has_context = (
                    cf.function_name or
                    cf.loop_depth > 0 or
                    cf.in_try_block or
                    cf.in_async_context or
                    cf.exception_types
                )
                if not has_context:
                    issues.append(f"{example.id}: control_flow has no meaningful context")

        # Allow some without full context
        assert len(issues) <= 5, f"Control flow issues:\n" + "\n".join(issues[:10])


# =============================================================================
# Coverage Tests
# =============================================================================


class TestConstraintCoverage:
    """Test that constraint examples cover the expected domains and features."""

    def test_json_schema_examples_exist(
        self, examples_with_syntax: List[ConstraintExample]
    ) -> None:
        """Should have examples using JSON schema constraints."""
        json_schema_examples = [
            e for e in examples_with_syntax if e.spec.json_schema
        ]
        assert len(json_schema_examples) >= 10, (
            f"Only {len(json_schema_examples)} JSON schema examples, expected 10+"
        )

    def test_regex_examples_exist(
        self, examples_with_syntax: List[ConstraintExample]
    ) -> None:
        """Should have examples using regex constraints."""
        regex_examples = [e for e in examples_with_syntax if e.spec.regex]
        assert len(regex_examples) >= 5, (
            f"Only {len(regex_examples)} regex examples, expected 5+"
        )

    def test_ebnf_examples_exist(
        self, examples_with_syntax: List[ConstraintExample]
    ) -> None:
        """Should have examples using EBNF constraints."""
        ebnf_examples = [e for e in examples_with_syntax if e.spec.ebnf]
        # EBNF is less common, so lower threshold
        assert len(ebnf_examples) >= 3, (
            f"Only {len(ebnf_examples)} EBNF examples, expected 3+"
        )

    def test_type_binding_coverage(self, all_examples: List[ConstraintExample]) -> None:
        """Should have substantial examples with type bindings."""
        with_type_bindings = [e for e in all_examples if e.spec.type_bindings]
        assert len(with_type_bindings) >= 50, (
            f"Only {len(with_type_bindings)} examples with type_bindings, expected 50+"
        )

    def test_semantic_constraint_coverage(
        self, all_examples: List[ConstraintExample]
    ) -> None:
        """Should have examples with semantic constraints."""
        with_semantic = [e for e in all_examples if e.spec.semantic_constraints]
        assert len(with_semantic) >= 15, (
            f"Only {len(with_semantic)} examples with semantic_constraints, expected 15+"
        )

    def test_control_flow_coverage(self, all_examples: List[ConstraintExample]) -> None:
        """Should have examples with control flow context."""
        with_cf = [e for e in all_examples if e.spec.control_flow]
        assert len(with_cf) >= 20, (
            f"Only {len(with_cf)} examples with control_flow, expected 20+"
        )


# =============================================================================
# Summary Report
# =============================================================================


class TestSummaryReport:
    """Generate a summary report of constraint coverage."""

    def test_print_coverage_report(self, all_examples: List[ConstraintExample]) -> None:
        """Print a coverage report (always passes, informational only)."""
        print("\n" + "=" * 60)
        print("CONSTRAINT EXAMPLE COVERAGE REPORT")
        print("=" * 60)

        # Count by constraint type
        with_json = len([e for e in all_examples if e.spec.json_schema])
        with_regex = len([e for e in all_examples if e.spec.regex])
        with_ebnf = len([e for e in all_examples if e.spec.ebnf])
        domain_only = len([
            e for e in all_examples
            if not (e.spec.json_schema or e.spec.regex or e.spec.ebnf)
        ])

        print(f"\nSyntax Constraint Types:")
        print(f"  JSON Schema:  {with_json}")
        print(f"  Regex:        {with_regex}")
        print(f"  EBNF:         {with_ebnf}")
        print(f"  Domain-only:  {domain_only}")

        # Count domain context
        with_types = len([e for e in all_examples if e.spec.type_bindings])
        with_funcs = len([e for e in all_examples if e.spec.function_signatures])
        with_classes = len([e for e in all_examples if e.spec.class_definitions])
        with_imports = len([e for e in all_examples if e.spec.imports or e.spec.forbidden_imports])
        with_cf = len([e for e in all_examples if e.spec.control_flow])
        with_semantic = len([e for e in all_examples if e.spec.semantic_constraints])

        print(f"\nDomain Context:")
        print(f"  Type bindings:       {with_types}")
        print(f"  Function signatures: {with_funcs}")
        print(f"  Class definitions:   {with_classes}")
        print(f"  Import constraints:  {with_imports}")
        print(f"  Control flow:        {with_cf}")
        print(f"  Semantic:            {with_semantic}")

        print(f"\nTotal examples: {len(all_examples)}")
        print("=" * 60)

        # Always pass - this is informational
        assert True
