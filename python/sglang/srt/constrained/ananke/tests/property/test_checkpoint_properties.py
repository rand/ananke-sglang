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
"""Property tests for checkpoint round-trip correctness.

These tests verify that checkpoint-modify-restore preserves all state correctly,
ensuring rollback support maintains system invariants.
"""

import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st

from core.checkpoint import (
    Checkpoint,
    CheckpointManager,
    UnifiedCheckpoint,
    create_context_snapshot,
)
from core.constraint import TOP, BOTTOM
from core.unified import UnifiedConstraint, UNIFIED_TOP, UNIFIED_BOTTOM
from domains.types.constraint import TypeConstraint, TYPE_TOP, TYPE_BOTTOM, INT, STR
from domains.imports.constraint import ImportConstraint, IMPORT_TOP


# =============================================================================
# Strategy Definitions
# =============================================================================


@st.composite
def domain_state_strategy(draw):
    """Generate random domain state dictionary."""
    num_keys = draw(st.integers(min_value=0, max_value=5))
    state = {}
    for i in range(num_keys):
        key = draw(st.sampled_from(["parser_state", "env", "cache_hash", "position", f"key_{i}"]))
        value = draw(st.one_of(
            st.integers(min_value=0, max_value=1000),
            st.text(min_size=0, max_size=20, alphabet="abcdef"),
            st.booleans(),
        ))
        state[key] = value
    return state


@st.composite
def checkpoint_strategy(draw):
    """Generate random Checkpoint."""
    domain_name = draw(st.sampled_from(["types", "imports", "syntax", "controlflow", "semantics"]))
    state = draw(domain_state_strategy())
    position = draw(st.integers(min_value=0, max_value=1000))
    constraint_hash = draw(st.integers(min_value=0, max_value=2**32 - 1))
    return Checkpoint(
        domain_name=domain_name,
        state=state,
        position=position,
        constraint_hash=constraint_hash,
    )


@st.composite
def unified_constraint_strategy(draw):
    """Generate random UnifiedConstraint."""
    choice = draw(st.integers(min_value=0, max_value=9))
    if choice == 0:
        return UNIFIED_TOP
    elif choice == 1:
        return UNIFIED_BOTTOM
    else:
        # Generate with some type constraint
        type_choice = draw(st.integers(min_value=0, max_value=2))
        if type_choice == 0:
            types = TYPE_TOP
        elif type_choice == 1:
            types = TYPE_BOTTOM
        else:
            expected = draw(st.sampled_from([INT, STR, None]))
            types = TypeConstraint(expected_type=expected)
        return UnifiedConstraint(types=types)


@st.composite
def context_snapshot_strategy(draw):
    """Generate random context snapshot."""
    text_len = draw(st.integers(min_value=0, max_value=50))
    text = draw(st.text(min_size=text_len, max_size=text_len, alphabet="abcdefghij "))
    num_tokens = draw(st.integers(min_value=0, max_value=20))
    tokens = draw(st.lists(st.integers(min_value=0, max_value=32000), min_size=num_tokens, max_size=num_tokens))
    position = num_tokens
    return create_context_snapshot(
        generated_text=text,
        generated_tokens=tokens,
        position=position,
        metadata={"lang": "python"},
    )


@st.composite
def unified_checkpoint_strategy(draw):
    """Generate random UnifiedCheckpoint."""
    position = draw(st.integers(min_value=0, max_value=100))
    unified_constraint = draw(unified_constraint_strategy())

    # Generate domain checkpoints
    domain_names = ["types", "imports", "syntax", "controlflow", "semantics"]
    num_domains = draw(st.integers(min_value=0, max_value=5))
    selected_domains = draw(st.lists(
        st.sampled_from(domain_names),
        min_size=num_domains,
        max_size=num_domains,
        unique=True,
    ))

    domain_checkpoints = {}
    for domain_name in selected_domains:
        state = draw(domain_state_strategy())
        domain_checkpoints[domain_name] = Checkpoint(
            domain_name=domain_name,
            state=state,
            position=position,
        )

    context = draw(context_snapshot_strategy())

    return UnifiedCheckpoint(
        position=position,
        unified_constraint=unified_constraint,
        domain_checkpoints=domain_checkpoints,
        context_snapshot=context,
    )


# =============================================================================
# Property Tests
# =============================================================================


class TestCheckpointRoundTrip:
    """Property tests for checkpoint round-trip correctness."""

    @given(cp=checkpoint_strategy())
    def test_checkpoint_preserves_domain_name(self, cp: Checkpoint):
        """Checkpoint preserves domain name."""
        # Create manager and store checkpoint
        mgr = CheckpointManager(max_checkpoints=10)
        mgr.create_checkpoint(
            position=cp.position,
            unified_constraint=UNIFIED_TOP,
            domain_checkpoints={cp.domain_name: cp},
        )

        # Retrieve and verify
        retrieved = mgr.get_checkpoint_at(cp.position)
        assert retrieved is not None
        domain_cp = retrieved.get_domain_checkpoint(cp.domain_name)
        assert domain_cp is not None
        assert domain_cp.domain_name == cp.domain_name

    @given(cp=checkpoint_strategy())
    def test_checkpoint_preserves_state(self, cp: Checkpoint):
        """Checkpoint preserves domain state dictionary."""
        mgr = CheckpointManager(max_checkpoints=10)
        mgr.create_checkpoint(
            position=cp.position,
            unified_constraint=UNIFIED_TOP,
            domain_checkpoints={cp.domain_name: cp},
        )

        retrieved = mgr.get_checkpoint_at(cp.position)
        assert retrieved is not None
        domain_cp = retrieved.get_domain_checkpoint(cp.domain_name)
        assert domain_cp is not None
        assert domain_cp.state == cp.state

    @given(ucp=unified_checkpoint_strategy())
    def test_unified_checkpoint_preserves_constraint(self, ucp: UnifiedCheckpoint):
        """UnifiedCheckpoint preserves unified constraint."""
        mgr = CheckpointManager(max_checkpoints=10)
        mgr.create_checkpoint(
            position=ucp.position,
            unified_constraint=ucp.unified_constraint,
            domain_checkpoints=ucp.domain_checkpoints,
            context_snapshot=ucp.context_snapshot,
        )

        retrieved = mgr.get_checkpoint_at(ucp.position)
        assert retrieved is not None
        assert retrieved.unified_constraint == ucp.unified_constraint

    @given(ucp=unified_checkpoint_strategy())
    def test_unified_checkpoint_preserves_all_domain_checkpoints(self, ucp: UnifiedCheckpoint):
        """UnifiedCheckpoint preserves all domain checkpoints."""
        mgr = CheckpointManager(max_checkpoints=10)
        mgr.create_checkpoint(
            position=ucp.position,
            unified_constraint=ucp.unified_constraint,
            domain_checkpoints=ucp.domain_checkpoints,
            context_snapshot=ucp.context_snapshot,
        )

        retrieved = mgr.get_checkpoint_at(ucp.position)
        assert retrieved is not None

        for domain_name, domain_cp in ucp.domain_checkpoints.items():
            retrieved_cp = retrieved.get_domain_checkpoint(domain_name)
            assert retrieved_cp is not None, f"Missing checkpoint for {domain_name}"
            assert retrieved_cp.domain_name == domain_cp.domain_name
            assert retrieved_cp.state == domain_cp.state

    @given(ucp=unified_checkpoint_strategy())
    def test_unified_checkpoint_preserves_context(self, ucp: UnifiedCheckpoint):
        """UnifiedCheckpoint preserves context snapshot."""
        mgr = CheckpointManager(max_checkpoints=10)
        mgr.create_checkpoint(
            position=ucp.position,
            unified_constraint=ucp.unified_constraint,
            domain_checkpoints=ucp.domain_checkpoints,
            context_snapshot=ucp.context_snapshot,
        )

        retrieved = mgr.get_checkpoint_at(ucp.position)
        assert retrieved is not None
        assert retrieved.context_snapshot == ucp.context_snapshot


class TestCheckpointManagerRoundTrip:
    """Property tests for CheckpointManager rollback correctness."""

    @given(
        positions=st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=2,
            max_size=10,
            unique=True,
        ).map(sorted),  # Ensure positions are sorted
        rollback_idx=st.integers(min_value=0, max_value=9),
    )
    def test_rollback_preserves_earlier_checkpoints(self, positions, rollback_idx):
        """Rollback to position N preserves all checkpoints at positions <= N."""
        assume(len(positions) >= 2)
        assume(rollback_idx < len(positions))

        mgr = CheckpointManager(max_checkpoints=20)

        # Create checkpoints at each position
        for pos in positions:
            mgr.create_checkpoint(
                position=pos,
                unified_constraint=UNIFIED_TOP,
                domain_checkpoints={},
            )

        rollback_position = positions[rollback_idx]
        mgr.rollback_to(rollback_position)

        # All positions <= rollback_position should still be accessible
        for pos in positions[:rollback_idx + 1]:
            cp = mgr.get_checkpoint_at(pos)
            assert cp is not None, f"Checkpoint at {pos} lost after rollback to {rollback_position}"
            assert cp.position == pos

    @given(
        positions=st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=2,
            max_size=10,
            unique=True,
        ).map(sorted),
        rollback_idx=st.integers(min_value=0, max_value=8),
    )
    def test_rollback_removes_later_checkpoints(self, positions, rollback_idx):
        """Rollback to position N removes all checkpoints at positions > N."""
        assume(len(positions) >= 2)
        assume(rollback_idx < len(positions) - 1)  # Ensure there are checkpoints after rollback point

        mgr = CheckpointManager(max_checkpoints=20)

        for pos in positions:
            mgr.create_checkpoint(
                position=pos,
                unified_constraint=UNIFIED_TOP,
                domain_checkpoints={},
            )

        rollback_position = positions[rollback_idx]
        mgr.rollback_to(rollback_position)

        # All positions > rollback_position should be removed
        for pos in positions[rollback_idx + 1:]:
            cp = mgr.get_checkpoint_at(pos)
            assert cp is None, f"Checkpoint at {pos} should be removed after rollback to {rollback_position}"

    @given(constraint=unified_constraint_strategy())
    def test_rollback_restores_exact_constraint(self, constraint: UnifiedConstraint):
        """Rollback restores the exact constraint from checkpoint."""
        mgr = CheckpointManager(max_checkpoints=10)

        # Create checkpoint with specific constraint
        mgr.create_checkpoint(
            position=0,
            unified_constraint=constraint,
            domain_checkpoints={},
        )

        # Create more checkpoints
        mgr.create_checkpoint(
            position=5,
            unified_constraint=UNIFIED_TOP,  # Different constraint
            domain_checkpoints={},
        )

        # Rollback to original
        cp = mgr.rollback_to(0)
        assert cp is not None
        assert cp.unified_constraint == constraint


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
