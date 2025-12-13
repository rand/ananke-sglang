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
"""Unit tests for checkpoint management."""

import pytest

from core.checkpoint import (
    Checkpoint,
    CheckpointManager,
    UnifiedCheckpoint,
    create_context_snapshot,
)
from core.constraint import TOP, BOTTOM
from core.unified import UnifiedConstraint, UNIFIED_TOP


class TestCheckpoint:
    """Tests for single-domain Checkpoint."""

    def test_create_checkpoint(self):
        """Can create a checkpoint with domain state."""
        cp = Checkpoint(
            domain_name="types",
            state={"parser_state": 42, "env": {"x": "int"}},
            position=10,
            constraint_hash=12345,
        )
        assert cp.domain_name == "types"
        assert cp.state["parser_state"] == 42
        assert cp.position == 10
        assert cp.constraint_hash == 12345

    def test_checkpoint_immutable(self):
        """Checkpoint is immutable (frozen dataclass)."""
        cp = Checkpoint(domain_name="types", state={})
        with pytest.raises(AttributeError):
            cp.domain_name = "other"

    def test_checkpoint_validate(self):
        """Checkpoint validation checks position."""
        cp = Checkpoint(domain_name="types", state={}, position=5)

        # Current position >= checkpoint position is valid
        assert cp.validate(5, TOP) is True
        assert cp.validate(10, TOP) is True

        # Current position < checkpoint position is invalid
        assert cp.validate(3, TOP) is False


class TestUnifiedCheckpoint:
    """Tests for UnifiedCheckpoint combining all domains."""

    def test_create_unified_checkpoint(self):
        """Can create unified checkpoint with all components."""
        domain_cps = {
            "types": Checkpoint(domain_name="types", state={"a": 1}),
            "imports": Checkpoint(domain_name="imports", state={"b": 2}),
        }

        ucp = UnifiedCheckpoint(
            position=5,
            unified_constraint=UNIFIED_TOP,
            domain_checkpoints=domain_cps,
            context_snapshot={"text": "hello"},
        )

        assert ucp.position == 5
        assert ucp.unified_constraint == UNIFIED_TOP
        assert len(ucp.domain_checkpoints) == 2

    def test_get_domain_checkpoint(self):
        """Can retrieve domain checkpoints by name."""
        types_cp = Checkpoint(domain_name="types", state={"x": 1})
        ucp = UnifiedCheckpoint(
            position=0,
            unified_constraint=UNIFIED_TOP,
            domain_checkpoints={"types": types_cp},
        )

        assert ucp.get_domain_checkpoint("types") == types_cp
        assert ucp.get_domain_checkpoint("nonexistent") is None


class TestCheckpointManager:
    """Tests for CheckpointManager with bounded history."""

    def test_create_manager(self):
        """Can create checkpoint manager with limits."""
        mgr = CheckpointManager(max_checkpoints=100, checkpoint_interval=1)
        assert mgr.max_checkpoints == 100
        assert mgr.checkpoint_count == 0

    def test_create_checkpoint(self):
        """Can create and store checkpoints."""
        mgr = CheckpointManager(max_checkpoints=10)

        cp = mgr.create_checkpoint(
            position=0,
            unified_constraint=UNIFIED_TOP,
            domain_checkpoints={},
        )

        assert mgr.checkpoint_count == 1
        assert cp.position == 0

    def test_get_checkpoint_at(self):
        """Can retrieve checkpoint at exact position."""
        mgr = CheckpointManager(max_checkpoints=10)

        mgr.create_checkpoint(position=0, unified_constraint=UNIFIED_TOP, domain_checkpoints={})
        mgr.create_checkpoint(position=5, unified_constraint=UNIFIED_TOP, domain_checkpoints={})
        mgr.create_checkpoint(position=10, unified_constraint=UNIFIED_TOP, domain_checkpoints={})

        assert mgr.get_checkpoint_at(5).position == 5
        assert mgr.get_checkpoint_at(7) is None  # No checkpoint at position 7

    def test_get_checkpoint_before(self):
        """Can retrieve checkpoint at or before position."""
        mgr = CheckpointManager(max_checkpoints=10)

        mgr.create_checkpoint(position=0, unified_constraint=UNIFIED_TOP, domain_checkpoints={})
        mgr.create_checkpoint(position=5, unified_constraint=UNIFIED_TOP, domain_checkpoints={})
        mgr.create_checkpoint(position=10, unified_constraint=UNIFIED_TOP, domain_checkpoints={})

        # Exact match
        assert mgr.get_checkpoint_before(5).position == 5

        # Between checkpoints - returns earlier one
        assert mgr.get_checkpoint_before(7).position == 5

        # After all checkpoints
        assert mgr.get_checkpoint_before(15).position == 10

        # Before first checkpoint
        assert mgr.get_checkpoint_before(-1) is None

    def test_rollback_to(self):
        """Rollback removes newer checkpoints."""
        mgr = CheckpointManager(max_checkpoints=10)

        mgr.create_checkpoint(position=0, unified_constraint=UNIFIED_TOP, domain_checkpoints={})
        mgr.create_checkpoint(position=5, unified_constraint=UNIFIED_TOP, domain_checkpoints={})
        mgr.create_checkpoint(position=10, unified_constraint=UNIFIED_TOP, domain_checkpoints={})

        assert mgr.checkpoint_count == 3

        # Rollback to position 5
        cp = mgr.rollback_to(5)
        assert cp.position == 5
        assert mgr.checkpoint_count == 2  # Checkpoint at 10 removed
        assert mgr.newest_position == 5

    def test_prune_old_checkpoints(self):
        """Old checkpoints are pruned when exceeding limit."""
        mgr = CheckpointManager(max_checkpoints=3)

        for i in range(5):
            mgr.create_checkpoint(
                position=i,
                unified_constraint=UNIFIED_TOP,
                domain_checkpoints={},
            )

        # Only 3 most recent should remain
        assert mgr.checkpoint_count == 3
        assert mgr.oldest_position == 2
        assert mgr.newest_position == 4

    def test_clear(self):
        """Can clear all checkpoints."""
        mgr = CheckpointManager(max_checkpoints=10)

        mgr.create_checkpoint(position=0, unified_constraint=UNIFIED_TOP, domain_checkpoints={})
        mgr.create_checkpoint(position=5, unified_constraint=UNIFIED_TOP, domain_checkpoints={})

        assert mgr.checkpoint_count == 2

        mgr.clear()

        assert mgr.checkpoint_count == 0
        assert mgr.oldest_position is None
        assert mgr.newest_position is None


class TestCreateContextSnapshot:
    """Tests for context snapshot creation."""

    def test_create_snapshot(self):
        """Can create context snapshot."""
        snapshot = create_context_snapshot(
            generated_text="hello world",
            generated_tokens=[1, 2, 3],
            position=3,
            metadata={"lang": "python"},
        )

        assert snapshot["generated_text"] == "hello world"
        assert snapshot["generated_tokens"] == [1, 2, 3]
        assert snapshot["position"] == 3
        assert snapshot["metadata"]["lang"] == "python"

    def test_snapshot_creates_copies(self):
        """Snapshot creates copies of mutable data."""
        tokens = [1, 2, 3]
        metadata = {"key": "value"}

        snapshot = create_context_snapshot(
            generated_text="test",
            generated_tokens=tokens,
            position=3,
            metadata=metadata,
        )

        # Modify originals
        tokens.append(4)
        metadata["new_key"] = "new_value"

        # Snapshot should be unchanged
        assert snapshot["generated_tokens"] == [1, 2, 3]
        assert "new_key" not in snapshot["metadata"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
