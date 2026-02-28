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
"""Unit tests for domain abstraction and GenerationContext."""

import pytest
import torch

from core.domain import (
    ConstraintDomain,
    GenerationContext,
    PassthroughDomain,
)
from core.constraint import TOP, BOTTOM, Satisfiability
from core.checkpoint import Checkpoint


class TestGenerationContext:
    """Tests for GenerationContext dataclass."""

    def test_default_context(self):
        """Default context has sensible defaults."""
        ctx = GenerationContext()
        assert ctx.generated_text == ""
        assert ctx.generated_tokens == []
        assert ctx.position == 0
        assert ctx.vocab_size == 0
        assert ctx.device == "cuda"
        assert ctx.language == "python"
        assert ctx.tokenizer is None
        assert ctx.metadata == {}

    def test_context_with_values(self):
        """Can create context with specific values."""
        ctx = GenerationContext(
            generated_text="def foo():",
            generated_tokens=[1, 2, 3],
            position=3,
            vocab_size=50000,
            device="cpu",
            language="rust",
            metadata={"file": "main.rs"},
        )

        assert ctx.generated_text == "def foo():"
        assert ctx.generated_tokens == [1, 2, 3]
        assert ctx.position == 3
        assert ctx.vocab_size == 50000
        assert ctx.device == "cpu"
        assert ctx.language == "rust"
        assert ctx.metadata["file"] == "main.rs"

    def test_extend_creates_new_context(self):
        """extend() returns new context without modifying original."""
        ctx1 = GenerationContext(
            generated_text="hello",
            generated_tokens=[1, 2],
            position=2,
            vocab_size=1000,
        )

        ctx2 = ctx1.extend(token_id=3, token_text=" world")

        # Original unchanged
        assert ctx1.generated_text == "hello"
        assert ctx1.generated_tokens == [1, 2]
        assert ctx1.position == 2

        # New context extended
        assert ctx2.generated_text == "hello world"
        assert ctx2.generated_tokens == [1, 2, 3]
        assert ctx2.position == 3

        # Properties preserved
        assert ctx2.vocab_size == ctx1.vocab_size

    def test_extend_copies_metadata(self):
        """extend() creates independent copy of metadata."""
        ctx1 = GenerationContext(metadata={"key": "value"})
        ctx2 = ctx1.extend(token_id=1, token_text="x")

        # Modify ctx2's metadata
        ctx2.metadata["new_key"] = "new_value"

        # ctx1's metadata unchanged
        assert "new_key" not in ctx1.metadata


class TestMaskPoolMethods:
    """Tests for MaskPool integration in GenerationContext."""

    def test_create_mask_without_pool(self):
        """create_mask works without pool (fallback allocation)."""
        ctx = GenerationContext(vocab_size=100, device="cpu")
        mask = ctx.create_mask(fill_value=True)

        assert mask.shape == (100,)
        assert mask.dtype == torch.bool
        assert mask.all()

    def test_create_mask_fill_false(self):
        """create_mask with fill_value=False creates all-False mask."""
        ctx = GenerationContext(vocab_size=100, device="cpu")
        mask = ctx.create_mask(fill_value=False)

        assert mask.shape == (100,)
        assert not mask.any()

    def test_borrowed_mask_without_pool(self):
        """borrowed_mask works without pool (fallback allocation)."""
        ctx = GenerationContext(vocab_size=100, device="cpu")

        with ctx.borrowed_mask(fill_value=True) as mask:
            assert mask.shape == (100,)
            assert mask.all()
            # Modify mask inside context
            mask[0] = False
            assert not mask[0]

    def test_borrowed_mask_with_pool(self):
        """borrowed_mask properly returns mask to pool."""
        from core.domain import MaskPool

        pool = MaskPool(vocab_size=100, device="cpu", pool_size=2)
        ctx = GenerationContext(vocab_size=100, device="cpu", mask_pool=pool)

        # Initially pool has 2 tensors
        assert pool.available_count == 2

        with ctx.borrowed_mask() as mask:
            # During borrow, one tensor is taken
            assert pool.available_count == 1
            assert mask.shape == (100,)

        # After context exit, tensor returned to pool
        assert pool.available_count == 2

    def test_borrowed_mask_releases_on_exception(self):
        """borrowed_mask returns mask to pool even on exception."""
        from core.domain import MaskPool

        pool = MaskPool(vocab_size=100, device="cpu", pool_size=2)
        ctx = GenerationContext(vocab_size=100, device="cpu", mask_pool=pool)

        try:
            with ctx.borrowed_mask() as mask:
                assert pool.available_count == 1
                raise ValueError("test exception")
        except ValueError:
            pass

        # Tensor returned despite exception
        assert pool.available_count == 2

    def test_acquire_release_mask(self):
        """acquire_mask and release_mask work correctly."""
        from core.domain import MaskPool

        pool = MaskPool(vocab_size=50, device="cpu", pool_size=3)
        ctx = GenerationContext(vocab_size=50, device="cpu", mask_pool=pool)

        mask1, handle1 = ctx.acquire_mask(fill_value=True)
        assert pool.available_count == 2
        assert mask1.all()

        mask2, handle2 = ctx.acquire_mask(fill_value=False)
        assert pool.available_count == 1
        assert not mask2.any()

        ctx.release_mask(handle1)
        assert pool.available_count == 2

        ctx.release_mask(handle2)
        assert pool.available_count == 3

    def test_extend_preserves_mask_pool(self):
        """extend() preserves mask_pool reference."""
        from core.domain import MaskPool

        pool = MaskPool(vocab_size=100, device="cpu", pool_size=4)
        ctx1 = GenerationContext(vocab_size=100, device="cpu", mask_pool=pool)

        ctx2 = ctx1.extend(token_id=1, token_text="x")

        # Same pool is shared
        assert ctx2.mask_pool is ctx1.mask_pool


class TestPassthroughDomain:
    """Tests for PassthroughDomain (placeholder implementation)."""

    def test_create_passthrough_domain(self):
        """Can create passthrough domain with name and constraints."""
        domain = PassthroughDomain(
            domain_name="types",
            top_constraint=TOP,
            bottom_constraint=BOTTOM,
        )

        assert domain.name == "types"
        assert domain.top == TOP
        assert domain.bottom == BOTTOM

    def test_token_mask_all_true(self):
        """Passthrough returns all-True mask."""
        domain = PassthroughDomain(
            domain_name="test",
            top_constraint=TOP,
            bottom_constraint=BOTTOM,
        )

        ctx = GenerationContext(vocab_size=100, device="cpu")
        mask = domain.token_mask(TOP, ctx)

        assert mask.shape == (100,)
        assert mask.dtype == torch.bool
        assert mask.all()

    def test_observe_token_unchanged(self):
        """Passthrough returns constraint unchanged."""
        domain = PassthroughDomain(
            domain_name="test",
            top_constraint=TOP,
            bottom_constraint=BOTTOM,
        )

        ctx = GenerationContext(vocab_size=100)
        result = domain.observe_token(TOP, token_id=42, context=ctx)

        assert result == TOP

    def test_checkpoint_restore(self):
        """Passthrough checkpoint/restore are no-ops."""
        domain = PassthroughDomain(
            domain_name="test",
            top_constraint=TOP,
            bottom_constraint=BOTTOM,
        )

        cp = domain.checkpoint()
        assert isinstance(cp, Checkpoint)
        assert cp.domain_name == "test"
        assert cp.state == {}

        # restore() should not raise
        domain.restore(cp)

    def test_satisfiability_delegates(self):
        """satisfiability() delegates to constraint."""
        domain = PassthroughDomain(
            domain_name="test",
            top_constraint=TOP,
            bottom_constraint=BOTTOM,
        )

        assert domain.satisfiability(TOP) == Satisfiability.SAT
        assert domain.satisfiability(BOTTOM) == Satisfiability.UNSAT

    def test_propagate_from_unchanged(self):
        """propagate_from() returns current constraint unchanged."""
        domain = PassthroughDomain(
            domain_name="test",
            top_constraint=TOP,
            bottom_constraint=BOTTOM,
        )

        ctx = GenerationContext()
        result = domain.propagate_from(
            source_domain="other",
            source_constraint=BOTTOM,
            current_constraint=TOP,
            context=ctx,
        )

        assert result == TOP


class TestConstraintDomainInterface:
    """Tests verifying ConstraintDomain ABC interface."""

    def test_passthrough_is_constraint_domain(self):
        """PassthroughDomain implements ConstraintDomain."""
        domain = PassthroughDomain(
            domain_name="test",
            top_constraint=TOP,
            bottom_constraint=BOTTOM,
        )

        assert isinstance(domain, ConstraintDomain)

    def test_abstract_methods_present(self):
        """ConstraintDomain has required abstract methods."""
        # These should all exist as abstract methods
        assert hasattr(ConstraintDomain, "name")
        assert hasattr(ConstraintDomain, "top")
        assert hasattr(ConstraintDomain, "bottom")
        assert hasattr(ConstraintDomain, "token_mask")
        assert hasattr(ConstraintDomain, "observe_token")
        assert hasattr(ConstraintDomain, "checkpoint")
        assert hasattr(ConstraintDomain, "restore")


class TestMultipleDomains:
    """Tests for using multiple domains together."""

    def test_create_multiple_domains(self):
        """Can create multiple independent domains."""
        types_domain = PassthroughDomain(
            domain_name="types",
            top_constraint=TOP,
            bottom_constraint=BOTTOM,
        )
        imports_domain = PassthroughDomain(
            domain_name="imports",
            top_constraint=TOP,
            bottom_constraint=BOTTOM,
        )

        assert types_domain.name == "types"
        assert imports_domain.name == "imports"

    def test_domains_independent_checkpoints(self):
        """Domain checkpoints are independent."""
        types_domain = PassthroughDomain(
            domain_name="types",
            top_constraint=TOP,
            bottom_constraint=BOTTOM,
        )
        imports_domain = PassthroughDomain(
            domain_name="imports",
            top_constraint=TOP,
            bottom_constraint=BOTTOM,
        )

        types_cp = types_domain.checkpoint()
        imports_cp = imports_domain.checkpoint()

        assert types_cp.domain_name == "types"
        assert imports_cp.domain_name == "imports"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
