# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for mask relaxation module."""

import pytest
import torch
from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock

import sys
from pathlib import Path

# Add parent directories to path for standalone testing
ananke_dir = Path(__file__).parent.parent.parent
if str(ananke_dir) not in sys.path:
    sys.path.insert(0, str(ananke_dir))

from masks.relaxation import (
    MaskRelaxation,
    RelaxationPolicy,
    RelaxationResult,
    RelaxationAwareEvaluator,
    compute_mask_with_relaxation,
    RELAXATION_ORDER,
    NEVER_RELAX,
    create_relaxation_policy,
)


# =============================================================================
# Test Data and Fixtures
# =============================================================================


@dataclass
class MockContext:
    """Mock generation context for testing."""
    vocab_size: int = 1000
    device: str = "cpu"


@pytest.fixture
def vocab_size() -> int:
    """Standard vocabulary size for tests."""
    return 1000


@pytest.fixture
def context(vocab_size: int) -> MockContext:
    """Create mock context."""
    return MockContext(vocab_size=vocab_size, device="cpu")


@pytest.fixture
def all_true_mask(vocab_size: int) -> torch.Tensor:
    """Create a mask with all tokens allowed."""
    return torch.ones(vocab_size, dtype=torch.bool)


@pytest.fixture
def half_true_mask(vocab_size: int) -> torch.Tensor:
    """Create a mask with half tokens allowed."""
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[:vocab_size // 2] = True
    return mask


@pytest.fixture
def sparse_mask(vocab_size: int) -> torch.Tensor:
    """Create a mask with only 20 tokens allowed."""
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[:20] = True
    return mask


@pytest.fixture
def very_sparse_mask(vocab_size: int) -> torch.Tensor:
    """Create a mask with only 5 tokens allowed."""
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[:5] = True
    return mask


# =============================================================================
# Test MaskRelaxation Enum
# =============================================================================


class TestMaskRelaxation:
    """Tests for MaskRelaxation enum."""

    def test_values(self):
        """Test enum values."""
        assert MaskRelaxation.NONE.value == 1
        assert MaskRelaxation.PARTIAL.value == 2
        assert MaskRelaxation.SYNTAX_ONLY.value == 3
        assert MaskRelaxation.FULL.value == 4

    def test_ordering(self):
        """Test enum ordering by value."""
        assert MaskRelaxation.NONE.value < MaskRelaxation.PARTIAL.value
        assert MaskRelaxation.PARTIAL.value < MaskRelaxation.SYNTAX_ONLY.value
        assert MaskRelaxation.SYNTAX_ONLY.value < MaskRelaxation.FULL.value


# =============================================================================
# Test RelaxationPolicy
# =============================================================================


class TestRelaxationPolicy:
    """Tests for RelaxationPolicy dataclass."""

    def test_default_values(self):
        """Test default policy configuration."""
        policy = RelaxationPolicy()
        assert policy.enabled is True
        assert policy.threshold == 10
        assert policy.min_threshold == 1
        assert policy.max_relaxation_level == MaskRelaxation.SYNTAX_ONLY
        assert policy.domains_to_relax is None
        assert policy.log_relaxation is True

    def test_custom_values(self):
        """Test custom policy configuration."""
        policy = RelaxationPolicy(
            enabled=False,
            threshold=50,
            min_threshold=5,
            max_relaxation_level=MaskRelaxation.PARTIAL,
            domains_to_relax=["types", "imports"],
            log_relaxation=False,
        )
        assert policy.enabled is False
        assert policy.threshold == 50
        assert policy.min_threshold == 5
        assert policy.max_relaxation_level == MaskRelaxation.PARTIAL
        assert policy.domains_to_relax == ["types", "imports"]
        assert policy.log_relaxation is False

    def test_can_relax_domain_default(self):
        """Test can_relax_domain with default policy."""
        policy = RelaxationPolicy()

        # Domains in RELAXATION_ORDER can be relaxed
        for domain in RELAXATION_ORDER:
            assert policy.can_relax_domain(domain) is True

        # Syntax can never be relaxed
        assert policy.can_relax_domain("syntax") is False

        # Unknown domains can be relaxed if in RELAXATION_ORDER
        assert policy.can_relax_domain("unknown") is False

    def test_can_relax_domain_restricted(self):
        """Test can_relax_domain with restricted domains."""
        policy = RelaxationPolicy(domains_to_relax=["types", "imports"])

        assert policy.can_relax_domain("types") is True
        assert policy.can_relax_domain("imports") is True
        assert policy.can_relax_domain("semantics") is False
        assert policy.can_relax_domain("controlflow") is False
        assert policy.can_relax_domain("syntax") is False  # Never relaxed

    def test_get_relaxation_order_default(self):
        """Test default relaxation order."""
        policy = RelaxationPolicy()
        order = policy.get_relaxation_order()
        assert order == RELAXATION_ORDER

    def test_get_relaxation_order_restricted(self):
        """Test restricted relaxation order preserves RELAXATION_ORDER."""
        policy = RelaxationPolicy(domains_to_relax=["imports", "types"])
        order = policy.get_relaxation_order()
        # Should maintain RELAXATION_ORDER order for specified domains
        # RELAXATION_ORDER = ["semantics", "controlflow", "imports", "types"]
        # So imports comes before types
        assert order == ["imports", "types"]


# =============================================================================
# Test RelaxationResult
# =============================================================================


class TestRelaxationResult:
    """Tests for RelaxationResult dataclass."""

    def test_default_values(self, all_true_mask):
        """Test default result values."""
        result = RelaxationResult(fused_mask=all_true_mask)
        assert result.relaxation_level == MaskRelaxation.NONE
        assert result.domains_applied == []
        assert result.domains_relaxed == []
        assert result.final_popcount == 0
        assert result.initial_popcount == 0
        assert result.popcount_history == {}

    def test_with_relaxation(self, sparse_mask):
        """Test result with relaxation."""
        result = RelaxationResult(
            fused_mask=sparse_mask,
            relaxation_level=MaskRelaxation.PARTIAL,
            domains_applied=["types", "imports"],
            domains_relaxed=["semantics"],
            final_popcount=20,
            initial_popcount=500,
            popcount_history={"types": 100, "imports": 20, "semantics_relaxed": 5},
        )
        assert result.relaxation_level == MaskRelaxation.PARTIAL
        assert result.domains_applied == ["types", "imports"]
        assert result.domains_relaxed == ["semantics"]
        assert result.final_popcount == 20
        assert result.initial_popcount == 500


# =============================================================================
# Test compute_mask_with_relaxation
# =============================================================================


class TestComputeMaskWithRelaxation:
    """Tests for compute_mask_with_relaxation function."""

    def test_no_domain_masks(self, all_true_mask, context):
        """Test with only syntax mask, no domain masks."""
        policy = RelaxationPolicy()
        result = compute_mask_with_relaxation(
            syntax_mask=all_true_mask,
            domain_masks={},
            context=context,
            policy=policy,
        )
        assert result.relaxation_level == MaskRelaxation.NONE
        assert result.domains_applied == []
        assert result.domains_relaxed == []
        assert result.final_popcount == 1000

    def test_all_masks_applied(self, vocab_size, context):
        """Test when all domain masks can be applied safely."""
        syntax_mask = torch.ones(vocab_size, dtype=torch.bool)

        # Domain masks that still leave plenty of tokens
        types_mask = torch.ones(vocab_size, dtype=torch.bool)
        types_mask[500:] = False  # 500 allowed

        imports_mask = torch.ones(vocab_size, dtype=torch.bool)
        imports_mask[250:] = False  # 250 allowed

        policy = RelaxationPolicy(threshold=10)
        result = compute_mask_with_relaxation(
            syntax_mask=syntax_mask,
            domain_masks={
                "types": types_mask,
                "imports": imports_mask,
            },
            context=context,
            policy=policy,
        )

        assert result.relaxation_level == MaskRelaxation.NONE
        assert set(result.domains_applied) == {"types", "imports"}
        assert result.domains_relaxed == []
        # Final mask should be intersection: first 250 tokens
        assert result.final_popcount == 250

    def test_partial_relaxation(self, vocab_size, context):
        """Test when some domains need to be relaxed."""
        syntax_mask = torch.zeros(vocab_size, dtype=torch.bool)
        syntax_mask[:100] = True  # 100 allowed initially

        # Types mask keeps 50 tokens
        types_mask = torch.ones(vocab_size, dtype=torch.bool)
        types_mask[50:] = False

        # Semantics mask would drop to 5 tokens (below threshold)
        semantics_mask = torch.ones(vocab_size, dtype=torch.bool)
        semantics_mask[5:] = False

        policy = RelaxationPolicy(threshold=10)
        result = compute_mask_with_relaxation(
            syntax_mask=syntax_mask,
            domain_masks={
                "types": types_mask,
                "semantics": semantics_mask,
            },
            context=context,
            policy=policy,
        )

        assert result.relaxation_level == MaskRelaxation.PARTIAL
        assert "types" in result.domains_applied
        assert "semantics" in result.domains_relaxed
        # Should only apply types, not semantics
        assert result.final_popcount == 50

    def test_syntax_only_relaxation(self, vocab_size, context):
        """Test when all domains need to be relaxed (syntax only)."""
        syntax_mask = torch.zeros(vocab_size, dtype=torch.bool)
        syntax_mask[:20] = True  # 20 allowed initially

        # All domain masks would drop below threshold
        types_mask = torch.zeros(vocab_size, dtype=torch.bool)
        types_mask[:5] = True  # Would drop to 5

        imports_mask = torch.zeros(vocab_size, dtype=torch.bool)
        imports_mask[:3] = True  # Would drop to 3

        policy = RelaxationPolicy(threshold=10)
        result = compute_mask_with_relaxation(
            syntax_mask=syntax_mask,
            domain_masks={
                "types": types_mask,
                "imports": imports_mask,
            },
            context=context,
            policy=policy,
        )

        assert result.relaxation_level == MaskRelaxation.SYNTAX_ONLY
        assert result.domains_applied == []
        assert set(result.domains_relaxed) == {"types", "imports"}
        # Should keep syntax-only mask
        assert result.final_popcount == 20

    def test_relaxation_disabled(self, vocab_size, context):
        """Test when relaxation is disabled."""
        syntax_mask = torch.zeros(vocab_size, dtype=torch.bool)
        syntax_mask[:100] = True

        # Semantics mask would drop to 5 tokens
        semantics_mask = torch.ones(vocab_size, dtype=torch.bool)
        semantics_mask[5:] = False

        policy = RelaxationPolicy(enabled=False, threshold=10)
        result = compute_mask_with_relaxation(
            syntax_mask=syntax_mask,
            domain_masks={"semantics": semantics_mask},
            context=context,
            policy=policy,
        )

        # Relaxation disabled - should apply all masks regardless of popcount
        assert result.relaxation_level == MaskRelaxation.NONE
        assert "semantics" in result.domains_applied
        assert result.final_popcount == 5  # Applied despite low popcount

    def test_application_order(self, vocab_size, context):
        """Test that domains are applied in reverse relaxation order."""
        # Application order should be: types -> imports -> controlflow -> semantics
        # (reverse of RELAXATION_ORDER)

        syntax_mask = torch.ones(vocab_size, dtype=torch.bool)

        # Track application order by mask contents
        types_mask = torch.ones(vocab_size, dtype=torch.bool)
        types_mask[800:] = False  # Tokens 0-799

        controlflow_mask = torch.ones(vocab_size, dtype=torch.bool)
        controlflow_mask[600:] = False  # Tokens 0-599

        semantics_mask = torch.ones(vocab_size, dtype=torch.bool)
        semantics_mask[400:] = False  # Tokens 0-399

        policy = RelaxationPolicy(threshold=10)
        result = compute_mask_with_relaxation(
            syntax_mask=syntax_mask,
            domain_masks={
                "types": types_mask,
                "controlflow": controlflow_mask,
                "semantics": semantics_mask,
            },
            context=context,
            policy=policy,
        )

        # All should be applied since popcounts stay above threshold
        assert "types" in result.domains_applied
        assert "controlflow" in result.domains_applied
        assert "semantics" in result.domains_applied
        # Final intersection should be 0-399
        assert result.final_popcount == 400


# =============================================================================
# Test RelaxationAwareEvaluator
# =============================================================================


class TestRelaxationAwareEvaluator:
    """Tests for RelaxationAwareEvaluator class."""

    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = RelaxationAwareEvaluator()
        assert evaluator.policy.enabled is True
        assert evaluator.policy.threshold == 10

    def test_custom_policy(self):
        """Test evaluator with custom policy."""
        policy = RelaxationPolicy(threshold=50, enabled=False)
        evaluator = RelaxationAwareEvaluator(policy=policy)
        assert evaluator.policy.threshold == 50
        assert evaluator.policy.enabled is False

    def test_register_domain(self):
        """Test domain registration."""
        evaluator = RelaxationAwareEvaluator()

        def mock_evaluator(constraint, context):
            return torch.ones(100, dtype=torch.bool)

        evaluator.register("types", mock_evaluator)
        assert "types" in evaluator._domains

    def test_unregister_domain(self):
        """Test domain unregistration."""
        evaluator = RelaxationAwareEvaluator()

        def mock_evaluator(constraint, context):
            return torch.ones(100, dtype=torch.bool)

        evaluator.register("types", mock_evaluator)
        evaluator.unregister("types")
        assert "types" not in evaluator._domains

    def test_evaluate_empty(self, all_true_mask, context):
        """Test evaluation with no constraints."""
        evaluator = RelaxationAwareEvaluator()

        result = evaluator.evaluate(
            constraints={},
            context=context,
            syntax_mask=all_true_mask,
        )

        assert result.relaxation_level == MaskRelaxation.NONE
        assert result.final_popcount == 1000

    def test_evaluate_with_domains(self, vocab_size, context):
        """Test evaluation with registered domains."""
        evaluator = RelaxationAwareEvaluator(policy=RelaxationPolicy(threshold=10))

        def types_evaluator(constraint, ctx):
            mask = torch.ones(vocab_size, dtype=torch.bool)
            mask[500:] = False
            return mask

        evaluator.register("types", types_evaluator)

        syntax_mask = torch.ones(vocab_size, dtype=torch.bool)
        result = evaluator.evaluate(
            constraints={"types": "int"},
            context=context,
            syntax_mask=syntax_mask,
        )

        assert "types" in result.domains_applied
        assert result.final_popcount == 500

    def test_statistics(self, vocab_size, context):
        """Test statistics tracking."""
        evaluator = RelaxationAwareEvaluator(policy=RelaxationPolicy(threshold=100))

        def types_evaluator(constraint, ctx):
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            mask[:50] = True  # Below threshold
            return mask

        evaluator.register("types", types_evaluator)

        syntax_mask = torch.ones(vocab_size, dtype=torch.bool)

        # Run evaluation that will trigger relaxation
        evaluator.evaluate(
            constraints={"types": "int"},
            context=context,
            syntax_mask=syntax_mask,
        )

        stats = evaluator.get_statistics()
        assert stats["total_evaluations"] == 1
        assert stats["relaxation_count"] == 1
        assert stats["relaxation_rate"] == 1.0

    def test_reset_statistics(self, vocab_size, context):
        """Test statistics reset."""
        evaluator = RelaxationAwareEvaluator()

        def types_evaluator(constraint, ctx):
            return torch.ones(vocab_size, dtype=torch.bool)

        evaluator.register("types", types_evaluator)

        syntax_mask = torch.ones(vocab_size, dtype=torch.bool)
        evaluator.evaluate(
            constraints={"types": "int"},
            context=context,
            syntax_mask=syntax_mask,
        )

        evaluator.reset_statistics()
        stats = evaluator.get_statistics()
        assert stats["total_evaluations"] == 0
        assert stats["relaxation_count"] == 0


# =============================================================================
# Test create_relaxation_policy Factory
# =============================================================================


class TestCreateRelaxationPolicy:
    """Tests for create_relaxation_policy factory function."""

    def test_default_values(self):
        """Test factory with default values."""
        policy = create_relaxation_policy()
        assert policy.enabled is True
        assert policy.threshold == 10
        assert policy.domains_to_relax is None

    def test_custom_values(self):
        """Test factory with custom values."""
        policy = create_relaxation_policy(
            threshold=50,
            enabled=False,
            domains=["types", "imports"],
        )
        assert policy.enabled is False
        assert policy.threshold == 50
        assert policy.domains_to_relax == ["types", "imports"]


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_relaxation_order(self):
        """Test RELAXATION_ORDER constant."""
        assert RELAXATION_ORDER == ["semantics", "controlflow", "imports", "types"]
        assert "syntax" not in RELAXATION_ORDER

    def test_never_relax(self):
        """Test NEVER_RELAX constant."""
        assert "syntax" in NEVER_RELAX
        for domain in RELAXATION_ORDER:
            assert domain not in NEVER_RELAX


# =============================================================================
# Integration Tests
# =============================================================================


class TestRelaxationIntegration:
    """Integration tests for relaxation system."""

    def test_progressive_relaxation_scenario(self, vocab_size, context):
        """Test a realistic progressive relaxation scenario."""
        # Scenario: Code generation where semantic constraints are too strict
        syntax_mask = torch.zeros(vocab_size, dtype=torch.bool)
        syntax_mask[:200] = True  # Syntax allows 200 tokens

        # Types mask: allows first 100 tokens
        types_mask = torch.zeros(vocab_size, dtype=torch.bool)
        types_mask[:100] = True

        # Imports mask: allows first 80 tokens
        imports_mask = torch.zeros(vocab_size, dtype=torch.bool)
        imports_mask[:80] = True

        # Controlflow mask: allows first 50 tokens
        controlflow_mask = torch.zeros(vocab_size, dtype=torch.bool)
        controlflow_mask[:50] = True

        # Semantics mask: would drop to 5 tokens (too restrictive!)
        semantics_mask = torch.zeros(vocab_size, dtype=torch.bool)
        semantics_mask[:5] = True

        policy = RelaxationPolicy(threshold=10)
        result = compute_mask_with_relaxation(
            syntax_mask=syntax_mask,
            domain_masks={
                "types": types_mask,
                "imports": imports_mask,
                "controlflow": controlflow_mask,
                "semantics": semantics_mask,
            },
            context=context,
            policy=policy,
        )

        # Types, imports, controlflow should be applied
        # Semantics should be relaxed (would drop to 5 < threshold 10)
        assert "types" in result.domains_applied
        assert "imports" in result.domains_applied
        assert "controlflow" in result.domains_applied
        assert "semantics" in result.domains_relaxed
        assert result.relaxation_level == MaskRelaxation.PARTIAL
        assert result.final_popcount == 50  # controlflow intersection

    def test_evaluator_with_mock_domains(self, vocab_size, context):
        """Test evaluator with mock domain implementations."""
        evaluator = RelaxationAwareEvaluator(policy=RelaxationPolicy(threshold=15))

        # Register domain evaluators
        def types_fn(constraint, ctx):
            mask = torch.ones(ctx.vocab_size, dtype=torch.bool)
            mask[100:] = False
            return mask

        def imports_fn(constraint, ctx):
            mask = torch.ones(ctx.vocab_size, dtype=torch.bool)
            mask[50:] = False
            return mask

        def semantics_fn(constraint, ctx):
            # Very restrictive
            mask = torch.zeros(ctx.vocab_size, dtype=torch.bool)
            mask[:10] = True
            return mask

        evaluator.register("types", types_fn)
        evaluator.register("imports", imports_fn)
        evaluator.register("semantics", semantics_fn)

        syntax_mask = torch.ones(vocab_size, dtype=torch.bool)
        result = evaluator.evaluate(
            constraints={
                "types": {"expected": "int"},
                "imports": {"module": "os"},
                "semantics": {"invariant": "x > 0"},
            },
            context=context,
            syntax_mask=syntax_mask,
        )

        # Should apply types and imports, relax semantics
        assert "types" in result.domains_applied
        assert "imports" in result.domains_applied
        assert "semantics" in result.domains_relaxed
        assert result.relaxation_level == MaskRelaxation.PARTIAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
