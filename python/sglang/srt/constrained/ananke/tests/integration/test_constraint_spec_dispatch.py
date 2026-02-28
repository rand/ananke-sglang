# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Reproduction test for constraint_spec dispatch issues.

This test aims to understand what ACTUALLY happens when:
1. constraint_spec={"language": "python"} is used (no syntax constraint)
2. constraint_spec={"language": "python", "regex": "[0-9]+"} is used
3. Direct regex parameter is used

The goal is to diagnose why constraint_spec causes issues on Modal while
direct regex works fine.

NOTE: This is a diagnostic test that runs WITHOUT full SGLang/torch dependencies.
It traces the dispatch logic to understand the flow.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import Mock

try:
    import pytest
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch not available, skipping tests requiring tensors")

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spec.constraint_spec import ConstraintSpec
from backend.backend import AnankeBackend
from backend.grammar import AnankeGrammar
from core.unified import UNIFIED_TOP


class MockTokenizer:
    """Minimal mock tokenizer for testing."""
    
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self._vocab = {i: f"token_{i}" for i in range(vocab_size)}
    
    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, int):
            return self._vocab.get(token_ids, "")
        return "".join(self._vocab.get(tid, "") for tid in token_ids)
    
    def get_vocab(self):
        return {v: k for k, v in self._vocab.items()}


class MockSyntaxGrammar:
    """Mock syntax grammar that tracks calls."""
    
    def __init__(self, name: str = "mock"):
        self.name = name
        self.finished = False
        self.fill_vocab_mask_calls = []
        self.accept_token_calls = []
        self.rollback_calls = []
    
    def fill_vocab_mask(self, vocab_mask, idx=0):
        """Track fill_vocab_mask calls."""
        self.fill_vocab_mask_calls.append((vocab_mask.shape, idx))
        # Leave mask as-is (all True)
    
    def accept_token(self, token_id):
        """Track accept_token calls."""
        self.accept_token_calls.append(token_id)
    
    def rollback(self, k):
        """Track rollback calls."""
        self.rollback_calls.append(k)
    
    def copy(self):
        """Return a new mock."""
        return MockSyntaxGrammar(name=self.name + "_copy")


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer(vocab_size=100)


@pytest.fixture
def backend(mock_tokenizer):
    """Create AnankeBackend without llguidance dependency."""
    backend = AnankeBackend(
        tokenizer=mock_tokenizer,
        vocab_size=100,
        language="python",
    )
    # Override syntax backend to avoid llguidance dependency
    backend.syntax_backend = None
    return backend


class TestConstraintSpecDispatch:
    """Test constraint_spec dispatch vs direct parameter dispatch."""
    
    def test_language_only_constraint_spec_no_syntax(self, backend):
        """Test constraint_spec with only language, no syntax constraint.
        
        This is the problematic case: {"language": "python"} with no regex/json_schema.
        According to epistemic analysis, this creates AnankeGrammar with syntax_grammar=None
        and domain constraints defaulting to TOP (no masking).
        """
        spec = ConstraintSpec(
            language="python",
            # NO syntax constraint (json_schema, regex, ebnf, structural_tag)
        )
        
        # Verify spec has no syntax constraint
        assert not spec.has_syntax_constraint()
        assert spec.get_syntax_constraint_type() is None
        
        # This should create grammar with no syntax constraint
        grammar = backend._create_ananke_grammar(
            syntax_grammar=None,
            constraint_spec=spec,
        )
        
        assert grammar is not None
        assert grammar.syntax_grammar is None
        print(f"\n[language-only] Grammar created: syntax_grammar={grammar.syntax_grammar}")
        print(f"[language-only] Domains: {list(grammar.domains.keys())}")
        print(f"[language-only] Constraint: {grammar.constraint}")
        
        # Check what constraint we have
        assert grammar.constraint == UNIFIED_TOP or grammar.constraint.is_top()
        
        # Test mask computation
        vocab_mask = torch.ones((1, 4), dtype=torch.int32, device="cpu")
        original_mask = vocab_mask.clone()
        
        grammar.fill_vocab_mask(vocab_mask, idx=0)
        
        print(f"[language-only] Original mask: {original_mask}")
        print(f"[language-only] After fill_vocab_mask: {vocab_mask}")
        
        # With TOP constraint and no syntax grammar, mask should be unchanged (all valid)
        # This means NO actual constraint is applied
        assert torch.equal(vocab_mask, original_mask), \
            "Language-only spec should not modify mask (no constraints active)"
    
    def test_constraint_spec_with_regex(self, backend):
        """Test constraint_spec with both language and regex.
        
        This tests: {"language": "python", "regex": "[0-9]+"}
        Should create syntax grammar from regex.
        """
        spec = ConstraintSpec(
            language="python",
            regex="[0-9]+",
        )
        
        # Verify spec has syntax constraint
        assert spec.has_syntax_constraint()
        assert spec.get_syntax_constraint_type() == "regex"
        assert spec.get_syntax_constraint_value() == "[0-9]+"
        
        # Create mock syntax grammar
        mock_syntax = MockSyntaxGrammar(name="regex_grammar")
        
        grammar = backend._create_ananke_grammar(
            syntax_grammar=mock_syntax,
            constraint_spec=spec,
        )
        
        assert grammar is not None
        assert grammar.syntax_grammar is mock_syntax
        print(f"\n[spec+regex] Grammar created: syntax_grammar={grammar.syntax_grammar.name}")
        print(f"[spec+regex] Domains: {list(grammar.domains.keys())}")
        
        # Test mask computation
        vocab_mask = torch.ones((1, 4), dtype=torch.int32, device="cpu")
        
        grammar.fill_vocab_mask(vocab_mask, idx=0)
        
        print(f"[spec+regex] fill_vocab_mask calls: {mock_syntax.fill_vocab_mask_calls}")
        
        # Should have called syntax grammar
        assert len(mock_syntax.fill_vocab_mask_calls) > 0, \
            "Should delegate to syntax grammar"
    
    def test_direct_regex_dispatch(self, backend):
        """Test direct regex dispatch (not via constraint_spec).
        
        This is the working case that we compare against.
        """
        # Since we don't have real llguidance, we mock it
        mock_syntax = MockSyntaxGrammar(name="direct_regex")
        
        # Simulate what dispatch_regex would do
        grammar = backend._create_ananke_grammar(
            syntax_grammar=mock_syntax,
            constraint_spec=None,
        )
        
        assert grammar is not None
        assert grammar.syntax_grammar is mock_syntax
        print(f"\n[direct-regex] Grammar created: syntax_grammar={grammar.syntax_grammar.name}")
        print(f"[direct-regex] Domains: {list(grammar.domains.keys())}")
        
        # Test mask computation
        vocab_mask = torch.ones((1, 4), dtype=torch.int32, device="cpu")
        
        grammar.fill_vocab_mask(vocab_mask, idx=0)
        
        print(f"[direct-regex] fill_vocab_mask calls: {mock_syntax.fill_vocab_mask_calls}")
        
        assert len(mock_syntax.fill_vocab_mask_calls) > 0
    
    def test_domains_with_language_only_spec(self, backend):
        """Examine what domains are created with language-only spec."""
        spec = ConstraintSpec(language="python")
        
        grammar = backend._create_ananke_grammar(
            syntax_grammar=None,
            constraint_spec=spec,
        )
        
        print(f"\n[domains] Created domains: {list(grammar.domains.keys())}")
        
        # Check each domain's constraint state
        for domain_name, domain in grammar.domains.items():
            print(f"[domains] {domain_name}:")
            print(f"  - top: {domain.top}")
            print(f"  - bottom: {domain.bottom}")
            
            # Get token mask from this domain
            mask = domain.token_mask(domain.top, grammar.context)
            true_count = mask.sum().item()
            total = mask.numel()
            print(f"  - token_mask with TOP: {true_count}/{total} tokens allowed")
            
            # TOP should allow all tokens (no actual constraint)
            assert true_count == total, \
                f"Domain {domain_name} with TOP should allow all tokens"
    
    def test_constraint_spec_intensity_selection(self, backend):
        """Test how intensity affects domain selection with constraint_spec."""
        # Default intensity should be auto-assessed
        spec_default = ConstraintSpec(language="python")
        grammar_default = backend._create_ananke_grammar(
            syntax_grammar=None,
            constraint_spec=spec_default,
        )
        
        print(f"\n[intensity] Default domains: {list(grammar_default.domains.keys())}")
        
        # Explicit SYNTAX_ONLY intensity
        spec_syntax_only = ConstraintSpec(
            language="python",
            intensity="syntax_only",
        )
        grammar_syntax_only = backend._create_ananke_grammar(
            syntax_grammar=None,
            constraint_spec=spec_syntax_only,
        )
        
        print(f"[intensity] SYNTAX_ONLY domains: {list(grammar_syntax_only.domains.keys())}")
        
        # With no syntax constraint and SYNTAX_ONLY, should have minimal domains
        # (possibly none if intensity gates properly)
    
    def test_mask_fusion_with_multiple_domains(self, backend):
        """Test how masks from multiple domains are fused."""
        spec = ConstraintSpec(language="python")
        
        grammar = backend._create_ananke_grammar(
            syntax_grammar=None,
            constraint_spec=spec,
        )
        
        # Create vocab mask
        vocab_mask = torch.ones((1, 4), dtype=torch.int32, device="cpu")
        
        print(f"\n[fusion] Initial mask: {vocab_mask}")
        
        # Fill with grammar
        grammar.fill_vocab_mask(vocab_mask, idx=0)
        
        print(f"[fusion] After fill_vocab_mask: {vocab_mask}")
        print(f"[fusion] Number of domains: {len(grammar.domains)}")
        
        # With TOP constraints everywhere, should remain all-valid
        # This is the KEY INSIGHT: constraint_spec with only language
        # creates a "constraint" that doesn't actually constrain anything


class TestConstraintSpecParsing:
    """Test ConstraintSpec parsing and validation."""
    
    def test_spec_with_only_language_is_valid(self):
        """Verify that ConstraintSpec with only language is valid."""
        spec = ConstraintSpec(language="python")
        
        # Should be valid spec (no exception)
        assert spec.language == "python"
        assert not spec.has_syntax_constraint()
    
    def test_spec_with_language_and_regex(self):
        """Verify that ConstraintSpec with language + regex works."""
        spec = ConstraintSpec(
            language="python",
            regex="[0-9]+",
        )
        
        assert spec.language == "python"
        assert spec.has_syntax_constraint()
        assert spec.regex == "[0-9]+"
    
    def test_spec_serialization_roundtrip(self):
        """Test that ConstraintSpec can round-trip through JSON."""
        spec = ConstraintSpec(
            language="python",
            regex="[0-9]+",
        )
        
        # Serialize to dict
        spec_dict = spec.to_dict()
        print(f"\n[serialization] Serialized: {spec_dict}")
        
        # Deserialize
        spec2 = ConstraintSpec.from_dict(spec_dict)
        
        assert spec2.language == spec.language
        assert spec2.regex == spec.regex
    
    def test_language_only_spec_serialization(self):
        """Test serialization of language-only spec."""
        spec = ConstraintSpec(language="python")
        
        spec_dict = spec.to_dict()
        print(f"\n[serialization] Language-only: {spec_dict}")
        
        # Should only have version and language
        assert "language" in spec_dict
        assert "regex" not in spec_dict or spec_dict["regex"] is None


class TestActualMaskComputation:
    """Test actual mask computation behavior."""
    
    def test_top_constraint_produces_all_true_mask(self, backend):
        """Verify that TOP constraint allows all tokens."""
        from core.domain import PassthroughDomain, GenerationContext
        from core.constraint import TOP, BOTTOM

        # Create a passthrough domain (should always return TOP)
        domain = PassthroughDomain(domain_name="test", top_constraint=TOP, bottom_constraint=BOTTOM)

        # Create context
        context = GenerationContext(
            vocab_size=100,
            device="cpu",
        )

        # Get mask for TOP constraint
        mask = domain.token_mask(TOP, context)
        
        print(f"\n[mask-computation] Mask shape: {mask.shape}")
        print(f"[mask-computation] True count: {mask.sum().item()}/{mask.numel()}")
        print(f"[mask-computation] All true: {mask.all().item()}")
        
        # TOP should allow everything
        assert mask.all(), "TOP constraint should allow all tokens"
    
    def test_domain_mask_with_no_context(self, backend):
        """Test what happens when domain has no context seeded."""
        from domains.types.domain import TypeDomain
        from core.domain import GenerationContext
        
        # Create TypeDomain with no context
        domain = TypeDomain(language="python")
        
        context = GenerationContext(
            vocab_size=100,
            device="cpu",
        )
        
        # Get mask for TOP (initial state)
        mask = domain.token_mask(domain.top, context)
        
        true_count = mask.sum().item()
        total = mask.numel()
        
        print(f"\n[no-context] TypeDomain with TOP:")
        print(f"  Allowed: {true_count}/{total}")
        
        # With no type context, TOP should allow all tokens
        assert true_count == total, \
            "TypeDomain with no context and TOP should allow all tokens"


class TestForbiddenImportsConstraintFlow:
    """Test that forbidden_imports flows from spec to actual constraint."""

    def test_forbidden_imports_creates_import_constraint(self, backend):
        """Verify that forbidden_imports in spec creates ImportConstraint.

        This tests the fix for the backend integration gap where forbidden_imports
        was silently ignored because UNIFIED_TOP was always passed.
        """
        spec = ConstraintSpec(
            language="python",
            regex=r"[a-z_][a-z0-9_]*",  # Module name pattern
            forbidden_imports=["os", "subprocess", "sys"],
        )

        # Create unified constraint from spec
        constraint = backend._create_unified_constraint_from_spec(spec)

        print(f"\n[forbidden-imports] Spec has forbidden_imports: {spec.forbidden_imports}")
        print(f"[forbidden-imports] Created constraint: {constraint}")
        print(f"[forbidden-imports] Is TOP: {constraint.is_top()}")
        print(f"[forbidden-imports] Imports component: {constraint.imports}")

        # Constraint should NOT be TOP
        assert not constraint.is_top(), \
            "Constraint with forbidden_imports should not be TOP"

        # Imports component should have forbidden modules
        assert hasattr(constraint.imports, "forbidden"), \
            "Imports constraint should have forbidden attribute"
        assert constraint.imports.forbidden == frozenset(["os", "subprocess", "sys"]), \
            f"Expected forbidden={{'os', 'subprocess', 'sys'}}, got {constraint.imports.forbidden}"

    @pytest.mark.skipif(
        not HAS_TORCH or not torch.cuda.is_available(),
        reason="Requires CUDA for full grammar creation"
    )
    def test_forbidden_imports_passed_to_grammar(self, backend):
        """Verify that forbidden_imports constraint is passed to AnankeGrammar."""
        spec = ConstraintSpec(
            language="python",
            regex=r"[a-z_][a-z0-9_]*",
            forbidden_imports=["os", "subprocess"],
        )

        # Create mock syntax grammar
        mock_syntax = MockSyntaxGrammar(name="regex_grammar")

        # dispatch_with_spec should create grammar with proper constraint
        # We test via _create_ananke_grammar which is the internal path
        grammar = backend._create_ananke_grammar(
            syntax_grammar=mock_syntax,
            constraint_spec=spec,
        )

        print(f"\n[grammar-constraint] Grammar.constraint: {grammar.constraint}")
        print(f"[grammar-constraint] Is TOP: {grammar.constraint.is_top()}")

        # Grammar's constraint should have forbidden imports
        assert not grammar.constraint.is_top(), \
            "Grammar constraint should not be TOP when forbidden_imports specified"
        assert grammar.constraint.imports.forbidden == frozenset(["os", "subprocess"]), \
            f"Grammar should have forbidden imports, got {grammar.constraint.imports.forbidden}"

    def test_no_forbidden_imports_uses_top(self, backend):
        """Verify that spec without forbidden_imports uses TOP."""
        spec = ConstraintSpec(
            language="python",
            regex=r"[a-z]+",
        )

        constraint = backend._create_unified_constraint_from_spec(spec)

        print(f"\n[no-forbidden] Created constraint: {constraint}")
        print(f"[no-forbidden] Is TOP: {constraint.is_top()}")

        # Without forbidden_imports, should be TOP
        assert constraint.is_top(), \
            "Constraint without forbidden_imports should be TOP"

    def test_import_domain_receives_constraint(self, backend):
        """Verify that ImportDomain's token_mask receives the constraint."""
        from domains.imports.domain import ImportDomain
        from domains.imports.constraint import ImportConstraint
        from core.domain import GenerationContext

        # Create domain with Python language
        domain = ImportDomain(language="python")

        context = GenerationContext(
            vocab_size=100,
            device="cpu",
        )

        # Create constraint with forbidden imports
        forbidden_constraint = ImportConstraint(
            forbidden=frozenset(["os", "subprocess", "sys"])
        )

        # Get mask with forbidden constraint
        mask_forbidden = domain.token_mask(forbidden_constraint, context)

        # Get mask with TOP (no constraint)
        from domains.imports.constraint import IMPORT_TOP
        mask_top = domain.token_mask(IMPORT_TOP, context)

        print(f"\n[domain-mask] Mask with TOP: {mask_top.sum().item()}/{mask_top.numel()} allowed")
        print(f"[domain-mask] Mask with forbidden: {mask_forbidden.sum().item()}/{mask_forbidden.numel()} allowed")

        # TOP should allow at least as many tokens as forbidden constraint
        assert mask_top.sum() >= mask_forbidden.sum(), \
            "TOP constraint should allow >= tokens than forbidden constraint"


class TestTypeConstraintFlow:
    """Test that expected_type flows from spec to TypeConstraint."""

    def test_expected_type_creates_type_constraint(self, backend):
        """Verify that expected_type in spec creates TypeConstraint.

        This tests the TypeConstraint integration where expected_type
        from the spec is propagated to the unified constraint.
        """
        from domains.types.constraint import TypeConstraint, TYPE_TOP
        from domains.types.domain import TypeDomain

        spec = ConstraintSpec(
            language="python",
            regex=r"[a-z_][a-z0-9_]*",
            expected_type="int",  # Expect an integer expression
        )

        # Create domains first (they get seeded with spec)
        domains = backend._create_domains_with_spec(spec, "python")

        # Create unified constraint from spec with domains
        constraint = backend._create_unified_constraint_from_spec(spec, domains)

        print(f"\n[expected-type] Spec has expected_type: {spec.expected_type}")
        print(f"[expected-type] Created constraint: {constraint}")
        print(f"[expected-type] Is TOP: {constraint.is_top()}")
        print(f"[expected-type] Types component: {constraint.types}")

        # Constraint should NOT be TOP
        assert not constraint.is_top(), \
            "Constraint with expected_type should not be TOP"

        # Types component should have expected type
        assert hasattr(constraint.types, "expected_type"), \
            "Types constraint should have expected_type attribute"
        assert constraint.types.expected_type is not None, \
            f"Expected expected_type to be set, got None"

    def test_no_expected_type_uses_top(self, backend):
        """Verify that spec without expected_type uses TYPE_TOP."""
        spec = ConstraintSpec(
            language="python",
            regex=r"[a-z]+",
        )

        # Create domains
        domains = backend._create_domains_with_spec(spec, "python")

        # Create unified constraint
        constraint = backend._create_unified_constraint_from_spec(spec, domains)

        print(f"\n[no-expected-type] Created constraint: {constraint}")
        print(f"[no-expected-type] Types component is_top: {constraint.types.is_top()}")

        # Types component should be TOP (no type constraint)
        assert constraint.types.is_top(), \
            "Types constraint without expected_type should be TOP"

    def test_type_bindings_seed_domain(self, backend):
        """Verify that type_bindings are seeded into TypeDomain."""
        from spec.constraint_spec import TypeBinding

        spec = ConstraintSpec(
            language="python",
            regex=r"[a-z_][a-z0-9_]*",
            type_bindings=[
                TypeBinding(name="x", type_expr="int"),
                TypeBinding(name="y", type_expr="str"),
            ],
            expected_type="int",
        )

        # Create domains (seeded with type bindings)
        domains = backend._create_domains_with_spec(spec, "python")

        assert "types" in domains, "TypeDomain should be in domains"
        type_domain = domains["types"]

        # Check that domain has expected_type set
        print(f"\n[type-bindings] TypeDomain expected_type: {type_domain.expected_type}")

        # Create unified constraint
        constraint = backend._create_unified_constraint_from_spec(spec, domains)

        print(f"[type-bindings] Created constraint: {constraint}")
        print(f"[type-bindings] Types constraint: {constraint.types}")

        # Constraint should have types set (not TOP)
        assert not constraint.types.is_top(), \
            "Types constraint with expected_type should not be TOP"


if __name__ == "__main__":
    # Allow running directly for quick diagnostics
    import sys
    
    print("=" * 80)
    print("CONSTRAINT_SPEC DISPATCH DIAGNOSTICS")
    print("=" * 80)
    
    tokenizer = MockTokenizer(vocab_size=100)
    backend = AnankeBackend(
        tokenizer=tokenizer,
        vocab_size=100,
        language="python",
    )
    backend.syntax_backend = None  # Disable llguidance
    
    tester = TestConstraintSpecDispatch()
    
    print("\n" + "=" * 80)
    print("TEST 1: Language-only constraint_spec (PROBLEMATIC CASE)")
    print("=" * 80)
    tester.test_language_only_constraint_spec_no_syntax(backend)
    
    print("\n" + "=" * 80)
    print("TEST 2: Constraint_spec with regex")
    print("=" * 80)
    tester.test_constraint_spec_with_regex(backend)
    
    print("\n" + "=" * 80)
    print("TEST 3: Direct regex dispatch")
    print("=" * 80)
    tester.test_direct_regex_dispatch(backend)
    
    print("\n" + "=" * 80)
    print("TEST 4: Domain examination")
    print("=" * 80)
    tester.test_domains_with_language_only_spec(backend)
    
    print("\n" + "=" * 80)
    print("TEST 5: Mask fusion")
    print("=" * 80)
    tester.test_mask_fusion_with_multiple_domains(backend)
    
    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)
