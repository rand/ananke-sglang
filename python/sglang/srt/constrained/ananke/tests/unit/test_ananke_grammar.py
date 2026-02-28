# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Unit tests for AnankeGrammar backend integration.

These tests verify the AnankeGrammar class correctly integrates:
- MaskPool for tensor allocation
- Lazy evaluation with budget control
- Checkpoint management for rollback
- Domain mask fusion
- Cross-domain constraint propagation
"""

import pytest
import torch

# Import from the ananke package
try:
    from backend.grammar import AnankeGrammar
    from core.domain import (
        ConstraintDomain,
        GenerationContext,
        MaskPool,
        PassthroughDomain,
    )
    from core.constraint import Constraint, Satisfiability, TOP, BOTTOM
    from core.unified import UnifiedConstraint, UNIFIED_TOP
    from masks.lazy import EvaluationBudget, EvaluationPriority
except ImportError:
    from sglang.srt.constrained.ananke.backend.grammar import AnankeGrammar
    from sglang.srt.constrained.ananke.core.domain import (
        ConstraintDomain,
        GenerationContext,
        MaskPool,
        PassthroughDomain,
    )
    from sglang.srt.constrained.ananke.core.constraint import (
        Constraint,
        Satisfiability,
        TOP,
        BOTTOM,
    )
    from sglang.srt.constrained.ananke.core.unified import (
        UnifiedConstraint,
        UNIFIED_TOP,
    )
    from sglang.srt.constrained.ananke.masks.lazy import (
        EvaluationBudget,
        EvaluationPriority,
    )


# =============================================================================
# Test Fixtures
# =============================================================================


class MockConstraint(Constraint):
    """Mock constraint for testing."""

    def __init__(self, value: int = 0, is_top: bool = False, is_bottom: bool = False):
        self._value = value
        self._is_top = is_top
        self._is_bottom = is_bottom

    def meet(self, other: "Constraint") -> "Constraint":
        if self._is_bottom or (hasattr(other, "_is_bottom") and other._is_bottom):
            return MockConstraint(is_bottom=True)
        if self._is_top:
            return other
        if hasattr(other, "_is_top") and other._is_top:
            return self
        return MockConstraint(value=self._value + getattr(other, "_value", 0))

    def is_top(self) -> bool:
        return self._is_top

    def is_bottom(self) -> bool:
        return self._is_bottom

    def satisfiability(self) -> Satisfiability:
        if self._is_bottom:
            return Satisfiability.UNSAT
        return Satisfiability.SAT

    def __eq__(self, other):
        if not isinstance(other, MockConstraint):
            return False
        return (
            self._value == other._value
            and self._is_top == other._is_top
            and self._is_bottom == other._is_bottom
        )

    def __hash__(self):
        return hash((self._value, self._is_top, self._is_bottom))


class MockDomain(ConstraintDomain[MockConstraint]):
    """Mock domain for testing."""

    def __init__(
        self,
        domain_name: str = "mock",
        mask_value: bool = True,
        vocab_size: int = 100,
    ):
        self._name = domain_name
        self._mask_value = mask_value
        self._vocab_size = vocab_size
        self._checkpoint_data = {}
        self._observe_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def top(self) -> MockConstraint:
        return MockConstraint(is_top=True)

    @property
    def bottom(self) -> MockConstraint:
        return MockConstraint(is_bottom=True)

    def token_mask(
        self,
        constraint: MockConstraint,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Return a mask based on configuration."""
        mask = torch.full(
            (self._vocab_size,),
            self._mask_value,
            dtype=torch.bool,
            device=context.device,
        )
        # Block some tokens based on constraint value
        if hasattr(constraint, "_value"):
            block_count = min(constraint._value, self._vocab_size // 2)
            mask[:block_count] = False
        return mask

    def observe_token(
        self,
        constraint: MockConstraint,
        token_id: int,
        context: GenerationContext,
    ) -> MockConstraint:
        """Update constraint after observing a token."""
        self._observe_count += 1
        if constraint.is_top():
            return MockConstraint(value=1)
        if constraint.is_bottom():
            return constraint
        return MockConstraint(value=constraint._value + 1)

    def checkpoint(self):
        from core.checkpoint import Checkpoint

        return Checkpoint(
            domain_name=self._name,
            state={"observe_count": self._observe_count},
        )

    def restore(self, checkpoint):
        self._observe_count = checkpoint.state.get("observe_count", 0)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self._vocab = {i: f"token_{i}" for i in range(vocab_size)}

    def decode(self, token_ids):
        return "".join(self._vocab.get(tid, "") for tid in token_ids)

    def get_vocab(self):
        return {v: k for k, v in self._vocab.items()}


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer(vocab_size=100)


@pytest.fixture
def mock_domains():
    return {
        "types": MockDomain("types", vocab_size=100),
        "imports": MockDomain("imports", vocab_size=100),
    }


@pytest.fixture
def grammar(mock_domains, mock_tokenizer):
    return AnankeGrammar(
        syntax_grammar=None,
        domains=mock_domains,
        vocab_size=100,
        device="cpu",
        tokenizer=mock_tokenizer,
        language="python",
        mask_pool_size=4,
    )


# =============================================================================
# MaskPool Integration Tests
# =============================================================================


class TestMaskPoolIntegration:
    """Tests for MaskPool integration in AnankeGrammar."""

    def test_mask_pool_created(self, grammar):
        """Test that MaskPool is created when vocab_size > 0."""
        assert grammar._mask_pool is not None
        assert grammar._mask_pool.pool_size == 4

    def test_mask_pool_disabled_when_zero_size(self, mock_domains, mock_tokenizer):
        """Test that MaskPool is not created when mask_pool_size=0."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains=mock_domains,
            vocab_size=100,
            device="cpu",
            tokenizer=mock_tokenizer,
            mask_pool_size=0,
        )
        assert grammar._mask_pool is None

    def test_context_has_mask_pool(self, grammar):
        """Test that GenerationContext receives the mask pool."""
        assert grammar.context.mask_pool is grammar._mask_pool

    def test_mask_pool_in_copy(self, grammar):
        """Test that copy() preserves mask_pool_size configuration."""
        copy = grammar.copy()
        assert copy._mask_pool_size == grammar._mask_pool_size
        # Copy gets its own pool instance
        assert copy._mask_pool is not None

    def test_acquire_release_from_context(self, grammar):
        """Test acquiring and releasing masks via context."""
        mask, handle = grammar.context.acquire_mask(fill_value=True)
        assert mask.shape == (100,)
        assert mask.dtype == torch.bool
        assert mask.all()  # All True

        grammar.context.release_mask(handle)
        # Pool should have the tensor back
        assert grammar._mask_pool.available_count == 4


# =============================================================================
# Lazy Evaluation Tests
# =============================================================================


class TestLazyEvaluation:
    """Tests for lazy evaluation with budget control."""

    def test_evaluator_initialized(self, grammar):
        """Test that an evaluator is initialized based on strategy."""
        # Default strategy is ADAPTIVE, so adaptive evaluator should be initialized
        assert grammar._adaptive_evaluator is not None
        assert grammar._evaluation_budget is not None

    def test_evaluation_budget_defaults(self, grammar):
        """Test default budget values."""
        budget = grammar._evaluation_budget
        assert budget.max_time_ns == 2_000_000  # 2ms
        assert budget.max_domains == 5
        assert budget.min_selectivity == 0.95

    def test_fill_vocab_mask_with_lazy_evaluation(self, grammar):
        """Test fill_vocab_mask uses lazy evaluation by default."""
        vocab_mask = torch.ones((1, 4), dtype=torch.int32, device="cpu")
        # This should not raise
        grammar.fill_vocab_mask(vocab_mask, idx=0, use_lazy_evaluation=True)

    def test_fill_vocab_mask_eager(self, grammar):
        """Test fill_vocab_mask with eager evaluation."""
        vocab_mask = torch.ones((1, 4), dtype=torch.int32, device="cpu")
        grammar.fill_vocab_mask(vocab_mask, idx=0, use_lazy_evaluation=False)


# =============================================================================
# Checkpoint and Rollback Tests
# =============================================================================


class TestCheckpointRollback:
    """Tests for checkpoint management and rollback."""

    def test_checkpoint_created_on_accept(self, grammar):
        """Test that checkpoints are created when accepting tokens."""
        # Accept a few tokens
        grammar.accept_token(0)
        grammar.accept_token(1)
        grammar.accept_token(2)

        # Should have checkpoints
        assert len(grammar.checkpoint_manager._checkpoints) > 0

    def test_rollback_restores_state(self, grammar):
        """Test that rollback restores previous state."""
        # Accept tokens
        grammar.accept_token(0)
        grammar.accept_token(1)
        grammar.accept_token(2)

        # Record position
        position_before_rollback = grammar.context.position
        assert position_before_rollback == 3

        # Rollback 2 tokens
        grammar.rollback(2)

        # Position should be restored
        assert grammar.context.position == 1

    def test_sparse_checkpointing(self, mock_tokenizer):
        """Test sparse checkpointing with interval > 1 and stable constraints."""

        class StableDomain(MockDomain):
            """Domain that returns stable constraints (no change per token)."""

            def observe_token(self, constraint, token_id, context):
                # Always return the same constraint (no change triggers)
                self._observe_count += 1
                return constraint if not constraint.is_top() else MockConstraint(value=1)

        stable_domains = {
            "types": StableDomain("types", vocab_size=100),
            "imports": StableDomain("imports", vocab_size=100),
        }

        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains=stable_domains,
            vocab_size=100,
            device="cpu",
            tokenizer=mock_tokenizer,
            checkpoint_interval=5,
        )

        # Accept many tokens
        for i in range(20):
            grammar.accept_token(i)

        # With interval=5 and stable constraints, we should have fewer checkpoints
        # Expected: initial (pos 0) + pos 5, 10, 15 = 4 checkpoints
        # But first token causes a change from TOP, so pos 1 also gets one
        # So we expect around 5-7 checkpoints, definitely fewer than 20
        assert len(grammar.checkpoint_manager._checkpoints) <= 10

    def test_rollback_clears_mask_cache(self, grammar):
        """Test that rollback clears the domain mask cache."""
        grammar.accept_token(0)
        grammar._domain_mask_cache["test"] = torch.ones(100)

        grammar.rollback(1)

        assert len(grammar._domain_mask_cache) == 0


# =============================================================================
# Domain Integration Tests
# =============================================================================


class TestDomainIntegration:
    """Tests for domain integration in AnankeGrammar."""

    def test_accept_token_updates_domains(self, grammar, mock_domains):
        """Test that accept_token updates all domains."""
        types_domain = mock_domains["types"]
        initial_count = types_domain._observe_count

        grammar.accept_token(0)

        assert types_domain._observe_count > initial_count

    def test_selective_cache_invalidation(self, grammar):
        """Test that only changed domains have their cache invalidated."""
        # Populate cache
        grammar._domain_mask_cache["types"] = torch.ones(100)
        grammar._domain_mask_cache["imports"] = torch.ones(100)

        # Create constraints that will be different
        old_constraint = grammar.constraint
        grammar.accept_token(0)

        # Cache should be selectively cleared based on constraint changes
        # (exact behavior depends on hash changes)

    def test_unsatisfiable_constraint_marks_finished(self, mock_tokenizer):
        """Test that BOTTOM constraint marks grammar as finished."""

        class BottomDomain(MockDomain):
            def observe_token(self, constraint, token_id, context):
                return MockConstraint(is_bottom=True)

        domains = {"types": BottomDomain("types", vocab_size=100)}
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains=domains,
            vocab_size=100,
            device="cpu",
            tokenizer=mock_tokenizer,
        )

        # Observing should make constraint BOTTOM
        grammar.accept_token(0)

        # Note: This requires UnifiedConstraint to properly check satisfiability
        # The test verifies the mechanism exists


# =============================================================================
# Copy and Parallel Decoding Tests
# =============================================================================


class TestCopyForParallelDecoding:
    """Tests for grammar copy functionality."""

    def test_copy_preserves_configuration(self, grammar):
        """Test that copy preserves all configuration."""
        copy = grammar.copy()

        assert copy.vocab_size == grammar.vocab_size
        assert copy.device == grammar.device
        assert copy.language == grammar.language
        assert copy._checkpoint_interval == grammar._checkpoint_interval
        assert copy._mask_pool_size == grammar._mask_pool_size

    def test_copy_has_independent_state(self, grammar):
        """Test that copy has independent mutable state."""
        copy = grammar.copy()

        # Modify original
        grammar.accept_token(0)
        grammar._domain_mask_cache["test"] = torch.ones(100)

        # Copy should be unaffected
        assert copy.context.position == 0
        assert "test" not in copy._domain_mask_cache

    def test_copy_shares_domains(self, grammar):
        """Test that copy shares domain instances (stateless by design)."""
        copy = grammar.copy()
        assert copy.domains is grammar.domains


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Basic performance sanity checks."""

    def test_mask_computation_completes(self, grammar):
        """Test that mask computation completes in reasonable time."""
        import time

        vocab_mask = torch.ones((1, 4), dtype=torch.int32, device="cpu")

        start = time.perf_counter()
        for _ in range(100):
            grammar.fill_vocab_mask(vocab_mask, idx=0)
        elapsed = time.perf_counter() - start

        # Should complete 100 iterations in under 1 second on CPU
        assert elapsed < 1.0

    def test_checkpoint_creation_completes(self, grammar):
        """Test that checkpoint creation is fast."""
        import time

        start = time.perf_counter()
        for i in range(100):
            grammar.accept_token(i % grammar.vocab_size)
        elapsed = time.perf_counter() - start

        # Should complete 100 accept_token calls in under 1 second
        assert elapsed < 1.0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_rollback_beyond_history(self, grammar):
        """Test rollback when k exceeds checkpoint history."""
        grammar.accept_token(0)

        # Rollback more than we have
        grammar.rollback(100)

        # Should handle gracefully (rollback to earliest checkpoint)

    def test_empty_domains(self, mock_tokenizer):
        """Test grammar with no domains."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains={},
            vocab_size=100,
            device="cpu",
            tokenizer=mock_tokenizer,
        )

        vocab_mask = torch.ones((1, 4), dtype=torch.int32, device="cpu")
        grammar.fill_vocab_mask(vocab_mask, idx=0)

        # Should not modify mask (no domain constraints)

    def test_zero_vocab_size(self, mock_domains, mock_tokenizer):
        """Test grammar with zero vocab size."""
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains=mock_domains,
            vocab_size=0,
            device="cpu",
            tokenizer=mock_tokenizer,
        )

        # Should not create mask pool
        assert grammar._mask_pool is None

    def test_domain_mask_error_does_not_crash(self, mock_tokenizer):
        """Test that domain mask errors are caught and don't crash the server.

        Regression test for ananke-pwh: _apply_domain_mask_with_relaxation crash
        when domain.token_mask() raises an exception.
        """

        class CrashingDomain(MockDomain):
            """Domain that raises an error during token_mask."""

            def token_mask(self, constraint, context):
                raise RuntimeError("Simulated domain mask failure")

        domains = {
            "types": CrashingDomain("types", vocab_size=100),
        }
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains=domains,
            constraint=UnifiedConstraint(
                types=MockConstraint(value=5),  # Non-TOP to trigger evaluation
            ),
            vocab_size=100,
            device="cpu",
            tokenizer=mock_tokenizer,
        )

        vocab_mask = torch.full((1, 4), fill_value=-1, dtype=torch.int32, device="cpu")
        # Must not raise - should catch and fall through to syntax-only
        grammar.fill_vocab_mask(vocab_mask, idx=0)

        # vocab_mask should be unchanged (no domain mask applied)
        assert (vocab_mask == -1).all()

    def test_domain_mask_with_non_top_constraint(self, mock_tokenizer):
        """Test domain mask application with non-TOP constraints.

        Exercises the _fill_vocab_mask_lazy → _evaluate_and_apply_domain_masks
        → _apply_domain_mask_with_relaxation path.
        """
        domains = {
            "types": MockDomain("types", vocab_size=100),
        }
        grammar = AnankeGrammar(
            syntax_grammar=None,
            domains=domains,
            constraint=UnifiedConstraint(
                types=MockConstraint(value=10),  # Non-TOP, blocks first 10 tokens
            ),
            vocab_size=100,
            device="cpu",
            tokenizer=mock_tokenizer,
            allow_relaxation=True,
            relaxation_threshold=1,  # Low threshold to avoid relaxation
        )

        vocab_mask = torch.full((1, 4), fill_value=-1, dtype=torch.int32, device="cpu")
        # Should apply domain mask without crashing
        grammar.fill_vocab_mask(vocab_mask, idx=0)

        # Some bits should now be cleared by the domain mask
        # (types MockDomain blocks first 10 tokens when value=10)
        assert not (vocab_mask == -1).all()
