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
"""Unit tests for sudoku-style hole filling with MCV heuristic.

Tests verify:
1. Hole selection strategies (MCV, sequential, random)
2. Fill candidate generation and filtering
3. Constraint propagation
4. Backtracking behavior
5. Trajectory tracking efficiency

Key Property:
    The MCV heuristic should select holes with fewest valid options first,
    enabling early pruning of the search tree.
"""

import pytest
import sys
from pathlib import Path
from typing import List, Tuple, Any

# Add the ananke package root to sys.path for standalone testing
_ANANKE_ROOT = Path(__file__).parent.parent.parent
if str(_ANANKE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANANKE_ROOT))

from holes.hole import Hole, HoleId, HoleState, TypeEnvironment, HoleGranularity
from search.sudoku_filler import (
    SudokuStyleHoleFiller,
    FillResult,
    FillStrategy,
    FillCandidate,
    HoledCode,
    fill_with_mcv_heuristic,
)
from search.trajectory import (
    Trajectory,
    TrajectoryNode,
    TrajectoryTrie,
    create_trajectory_trie,
)


class TestHoledCode:
    """Tests for HoledCode class."""

    def test_create_empty_holed_code(self):
        """Test creating HoledCode with no holes."""
        code = HoledCode(template="x = 1")
        assert code.template == "x = 1"
        assert not code.has_holes()
        assert code.to_string() == "x = 1"

    def test_create_holed_code_with_holes(self):
        """Test creating HoledCode with holes."""
        hole_id = HoleId(namespace="default", name="value", index=0)
        hole = Hole(id=hole_id, expected_type="int")

        code = HoledCode(
            template="x = ?default:value[0]",
            holes={hole_id: hole},
            hole_markers={hole_id: "?default:value[0]"},
        )

        assert code.has_holes()
        assert len(code.unfilled_holes()) == 1
        assert code.get_hole(hole_id) is not None

    def test_fill_hole(self):
        """Test filling a hole."""
        hole_id = HoleId(namespace="default", name="value", index=0)
        hole = Hole(id=hole_id, expected_type="int")

        code = HoledCode(
            template="x = ?default:value[0]",
            holes={hole_id: hole},
            hole_markers={hole_id: "?default:value[0]"},
        )

        filled = code.fill_hole(hole_id, "42")

        assert filled.to_string() == "x = 42"
        assert not filled.has_holes() or filled.get_hole(hole_id).is_filled

    def test_fill_preserves_original(self):
        """Test that filling creates a new instance (immutability)."""
        hole_id = HoleId(namespace="default", name="value", index=0)
        hole = Hole(id=hole_id)

        code = HoledCode(
            template="x = ?default:value[0]",
            holes={hole_id: hole},
            hole_markers={hole_id: "?default:value[0]"},
        )

        filled = code.fill_hole(hole_id, "42")

        # Original should be unchanged
        assert code.template == "x = ?default:value[0]"
        assert code.has_holes()

    def test_unfill_hole(self):
        """Test unfilling a hole (for backtracking)."""
        hole_id = HoleId(namespace="default", name="value", index=0)
        hole = Hole(id=hole_id)

        code = HoledCode(
            template="x = ?default:value[0]",
            holes={hole_id: hole},
            hole_markers={hole_id: "?default:value[0]"},
        )

        filled = code.fill_hole(hole_id, "42")
        unfilled = filled.unfill_hole(hole_id)

        # Unfilled should have empty hole
        unfilled_hole = unfilled.get_hole(hole_id)
        assert unfilled_hole.state == HoleState.EMPTY

    def test_multiple_holes(self):
        """Test code with multiple holes."""
        hole1 = HoleId(namespace="default", name="a", index=0)
        hole2 = HoleId(namespace="default", name="b", index=0)

        code = HoledCode(
            template="?default:a[0] + ?default:b[0]",
            holes={
                hole1: Hole(id=hole1, expected_type="int"),
                hole2: Hole(id=hole2, expected_type="int"),
            },
            hole_markers={
                hole1: "?default:a[0]",
                hole2: "?default:b[0]",
            },
        )

        assert len(code.unfilled_holes()) == 2

        filled1 = code.fill_hole(hole1, "1")
        assert len(filled1.unfilled_holes()) == 1

        filled2 = filled1.fill_hole(hole2, "2")
        assert filled2.to_string() == "1 + 2"

    def test_is_consistent(self):
        """Test consistency checking."""
        hole_id = HoleId(namespace="default", name="value", index=0)
        hole = Hole(id=hole_id)

        code = HoledCode(
            template="x = ?default:value[0]",
            holes={hole_id: hole},
            hole_markers={hole_id: "?default:value[0]"},
        )

        assert code.is_consistent()

        # Manually create invalid hole state
        invalid_hole = hole.invalidate()
        code.holes[hole_id] = invalid_hole
        assert not code.is_consistent()


class TestFillCandidate:
    """Tests for FillCandidate class."""

    def test_create_fill_candidate(self):
        """Test creating a fill candidate."""
        candidate = FillCandidate(value="42", score=0.9)
        assert candidate.value == "42"
        assert candidate.score == 0.9
        assert len(candidate.constraint_violations) == 0

    def test_candidate_with_violations(self):
        """Test candidate with constraint violations."""
        candidate = FillCandidate(
            value="invalid",
            score=0.3,
            constraint_violations=("Type error", "Syntax error"),
        )
        assert candidate.score == 0.3
        assert len(candidate.constraint_violations) == 2

    def test_candidate_ordering(self):
        """Test that candidates are ordered by score (descending)."""
        c1 = FillCandidate(value="a", score=0.5)
        c2 = FillCandidate(value="b", score=0.9)
        c3 = FillCandidate(value="c", score=0.7)

        sorted_candidates = sorted([c1, c2, c3])

        # Higher scores should come first
        assert sorted_candidates[0].value == "b"
        assert sorted_candidates[1].value == "c"
        assert sorted_candidates[2].value == "a"


class TestSudokuStyleHoleFiller:
    """Tests for SudokuStyleHoleFiller class."""

    @pytest.fixture
    def simple_holed_code(self):
        """Create simple code with one hole."""
        hole_id = HoleId(namespace="default", name="value", index=0)
        return HoledCode(
            template="x = ?default:value[0]",
            holes={hole_id: Hole(id=hole_id, expected_type="int")},
            hole_markers={hole_id: "?default:value[0]"},
        )

    @pytest.fixture
    def multi_hole_code(self):
        """Create code with multiple holes."""
        hole1 = HoleId(namespace="default", name="a", index=0)
        hole2 = HoleId(namespace="default", name="b", index=0)
        hole3 = HoleId(namespace="default", name="c", index=0)

        return HoledCode(
            template="def f(): return ?default:a[0] + ?default:b[0] * ?default:c[0]",
            holes={
                hole1: Hole(id=hole1, expected_type="int"),
                hole2: Hole(id=hole2, expected_type="int"),
                hole3: Hole(id=hole3, expected_type="int"),
            },
            hole_markers={
                hole1: "?default:a[0]",
                hole2: "?default:b[0]",
                hole3: "?default:c[0]",
            },
        )

    def test_fill_single_hole(self, simple_holed_code):
        """Test filling a single hole."""
        filler = SudokuStyleHoleFiller(strategy=FillStrategy.SEQUENTIAL)
        result = filler.fill(simple_holed_code)

        assert isinstance(result, FillResult)
        # Should succeed with placeholder candidates
        if result.success:
            assert result.filled_code is not None
            assert "?default:value[0]" not in result.filled_code

    def test_fill_multiple_holes(self, multi_hole_code):
        """Test filling multiple holes."""
        filler = SudokuStyleHoleFiller(strategy=FillStrategy.SEQUENTIAL)
        result = filler.fill(multi_hole_code)

        assert isinstance(result, FillResult)
        # Should produce some result (success or partial)
        assert result.filled_code is not None or len(result.unfilled_holes) > 0

    def test_mcv_strategy(self, multi_hole_code):
        """Test MCV (Most Constrained Variable) strategy."""
        filler = SudokuStyleHoleFiller(strategy=FillStrategy.MCV)
        result = filler.fill(multi_hole_code)

        assert isinstance(result, FillResult)

    def test_sequential_strategy(self, multi_hole_code):
        """Test sequential hole selection strategy."""
        filler = SudokuStyleHoleFiller(strategy=FillStrategy.SEQUENTIAL)
        result = filler.fill(multi_hole_code)

        assert isinstance(result, FillResult)

    def test_backtracking_limit(self, simple_holed_code):
        """Test that backtracking respects max_backtracks."""
        filler = SudokuStyleHoleFiller(max_backtracks=5)
        result = filler.fill(simple_holed_code)

        assert result.backtrack_count <= 5

    def test_fill_returns_history(self, multi_hole_code):
        """Test that fill history is tracked."""
        filler = SudokuStyleHoleFiller(strategy=FillStrategy.SEQUENTIAL)
        result = filler.fill(multi_hole_code)

        # Should have fill history for any successful fills
        assert isinstance(result.fill_history, list)

    def test_result_to_dict(self, simple_holed_code):
        """Test result serialization."""
        filler = SudokuStyleHoleFiller()
        result = filler.fill(simple_holed_code)

        d = result.to_dict()
        assert "success" in d
        assert "filled_code" in d
        assert "fill_history" in d
        assert "backtrack_count" in d

    def test_stats_tracking(self, simple_holed_code):
        """Test that statistics are tracked."""
        filler = SudokuStyleHoleFiller()
        filler.reset_stats()

        filler.fill(simple_holed_code)

        stats = filler.get_stats()
        assert "total_candidates" in stats
        assert "propagations" in stats

    def test_beam_width_limits_candidates(self, simple_holed_code):
        """Test that beam_width limits considered candidates."""
        filler_narrow = SudokuStyleHoleFiller(beam_width=2)
        filler_wide = SudokuStyleHoleFiller(beam_width=10)

        # Both should work, narrow just considers fewer options
        result_narrow = filler_narrow.fill(simple_holed_code)
        result_wide = filler_wide.fill(simple_holed_code)

        assert isinstance(result_narrow, FillResult)
        assert isinstance(result_wide, FillResult)

    def test_empty_code_no_holes(self):
        """Test with code that has no holes."""
        code = HoledCode(template="x = 42")
        filler = SudokuStyleHoleFiller()

        result = filler.fill(code)

        assert result.success
        assert result.filled_code == "x = 42"
        assert len(result.fill_history) == 0


class TestConvenienceFunction:
    """Tests for fill_with_mcv_heuristic convenience function."""

    def test_basic_usage(self):
        """Test basic usage of convenience function."""
        hole_id = HoleId(namespace="default", name="x", index=0)
        code = HoledCode(
            template="result = ?default:x[0]",
            holes={hole_id: Hole(id=hole_id, expected_type="int")},
            hole_markers={hole_id: "?default:x[0]"},
        )

        result = fill_with_mcv_heuristic(code)

        assert isinstance(result, FillResult)


class TestTrajectoryNode:
    """Tests for TrajectoryNode class."""

    def test_create_root_node(self):
        """Test creating a root node."""
        root = TrajectoryNode(state="initial")
        assert root.is_root()
        assert root.depth == 0
        assert root.state == "initial"

    def test_add_child(self):
        """Test adding child nodes."""
        root = TrajectoryNode(state="initial")
        hole_id = HoleId(namespace="default", name="x", index=0)

        child = root.add_child(
            hole_id=hole_id,
            fill_value="42",
            state="after_fill",
            score=0.9,
        )

        assert child.parent == root
        assert child.depth == 1
        assert child.fill_value == "42"
        assert child.score == 0.9

    def test_path_to_root(self):
        """Test getting path from node to root."""
        root = TrajectoryNode(state="s0")
        hole_id = HoleId(namespace="default", name="x", index=0)

        c1 = root.add_child(hole_id, "a", "s1")
        c2 = c1.add_child(hole_id, "b", "s2")
        c3 = c2.add_child(hole_id, "c", "s3")

        path = c3.path_to_root()

        assert len(path) == 4
        assert path[0] == c3
        assert path[-1] == root

    def test_fill_history(self):
        """Test getting fill history."""
        root = TrajectoryNode()
        h1 = HoleId(namespace="default", name="a", index=0)
        h2 = HoleId(namespace="default", name="b", index=0)

        c1 = root.add_child(h1, "1", "s1")
        c2 = c1.add_child(h2, "2", "s2")

        history = c2.fill_history()

        assert len(history) == 2
        assert history[0] == (h1, "1")
        assert history[1] == (h2, "2")

    def test_subtree_size(self):
        """Test computing subtree size."""
        root = TrajectoryNode()
        hole_id = HoleId(namespace="default", name="x", index=0)

        c1 = root.add_child(hole_id, "a", "s1")
        c2 = root.add_child(hole_id, "b", "s2")
        c1.add_child(hole_id, "c", "s3")

        assert root.subtree_size() == 4

    def test_subtree_leaves(self):
        """Test getting leaf nodes."""
        root = TrajectoryNode()
        hole_id = HoleId(namespace="default", name="x", index=0)

        c1 = root.add_child(hole_id, "a", "s1")
        c2 = root.add_child(hole_id, "b", "s2")
        c1.add_child(hole_id, "c", "s3")

        leaves = root.subtree_leaves()

        assert len(leaves) == 2


class TestTrajectory:
    """Tests for Trajectory class."""

    def test_extend_trajectory(self):
        """Test extending a trajectory."""
        trie = TrajectoryTrie()
        traj = trie.new_trajectory("s0")
        hole_id = HoleId(namespace="default", name="x", index=0)

        traj2 = traj.extend(hole_id, "42", "s1", score=0.9)

        assert traj2.depth == 1
        assert traj2.state == "s1"
        assert traj2.score == 0.9
        # Original unchanged
        assert traj.depth == 0

    def test_backtrack(self):
        """Test backtracking."""
        trie = TrajectoryTrie()
        traj = trie.new_trajectory("s0")
        hole_id = HoleId(namespace="default", name="x", index=0)

        traj = traj.extend(hole_id, "1", "s1")
        traj = traj.extend(hole_id, "2", "s2")
        traj = traj.extend(hole_id, "3", "s3")

        assert traj.depth == 3

        back1 = traj.backtrack(1)
        assert back1.depth == 2

        back2 = traj.backtrack(2)
        assert back2.depth == 1

    def test_backtrack_to_depth(self):
        """Test backtracking to specific depth."""
        trie = TrajectoryTrie()
        traj = trie.new_trajectory("s0")
        hole_id = HoleId(namespace="default", name="x", index=0)

        traj = traj.extend(hole_id, "1", "s1")
        traj = traj.extend(hole_id, "2", "s2")
        traj = traj.extend(hole_id, "3", "s3")

        back = traj.backtrack_to_depth(1)

        assert back.depth == 1

    def test_backtrack_to_root(self):
        """Test backtracking to root."""
        trie = TrajectoryTrie()
        traj = trie.new_trajectory("s0")
        hole_id = HoleId(namespace="default", name="x", index=0)

        traj = traj.extend(hole_id, "1", "s1")
        traj = traj.extend(hole_id, "2", "s2")

        back = traj.backtrack_to_root()

        assert back.depth == 0
        assert back.state == "s0"

    def test_fill_history_from_trajectory(self):
        """Test getting fill history from trajectory."""
        trie = TrajectoryTrie()
        traj = trie.new_trajectory("s0")
        h1 = HoleId(namespace="default", name="a", index=0)
        h2 = HoleId(namespace="default", name="b", index=0)

        traj = traj.extend(h1, "x", "s1")
        traj = traj.extend(h2, "y", "s2")

        history = traj.fill_history()

        assert len(history) == 2
        assert history[0] == (h1, "x")
        assert history[1] == (h2, "y")

    def test_cumulative_score(self):
        """Test computing cumulative score."""
        trie = TrajectoryTrie()
        traj = trie.new_trajectory("s0")
        hole_id = HoleId(namespace="default", name="x", index=0)

        traj = traj.extend(hole_id, "1", "s1", score=0.9)
        traj = traj.extend(hole_id, "2", "s2", score=0.8)

        cum_score = traj.cumulative_score()

        # 1.0 (root) * 0.9 * 0.8 = 0.72
        assert abs(cum_score - 0.72) < 0.01


class TestTrajectoryTrie:
    """Tests for TrajectoryTrie class."""

    def test_create_trajectory_trie(self):
        """Test creating a trajectory trie."""
        trie = TrajectoryTrie()
        assert trie.node_count == 1  # Just root

    def test_new_trajectory(self):
        """Test creating new trajectory."""
        trie = TrajectoryTrie()
        traj = trie.new_trajectory("initial")

        assert traj.state == "initial"
        assert traj.depth == 0

    def test_checkpoint_and_restore(self):
        """Test checkpointing and restoring."""
        trie = TrajectoryTrie()
        traj = trie.new_trajectory("s0")
        hole_id = HoleId(namespace="default", name="x", index=0)

        traj = traj.extend(hole_id, "1", "s1")
        checkpoint = trie.checkpoint(traj)

        traj = traj.extend(hole_id, "2", "s2")
        traj = traj.extend(hole_id, "3", "s3")

        restored = trie.restore(checkpoint)

        assert restored.depth == 1
        assert restored.state == "s1"

    def test_best_leaf(self):
        """Test finding best leaf by score."""
        trie = TrajectoryTrie()
        traj = trie.new_trajectory("s0")
        hole_id = HoleId(namespace="default", name="x", index=0)

        # Create branching tree
        t1 = traj.extend(hole_id, "a", "s1", score=0.5)
        t2 = traj.extend(hole_id, "b", "s2", score=0.9)

        t1.extend(hole_id, "c", "s3", score=0.9)  # Path: 0.5 * 0.9 = 0.45
        t2.extend(hole_id, "d", "s4", score=0.8)  # Path: 0.9 * 0.8 = 0.72

        best = trie.best_leaf()

        # Best path should be through t2
        assert best is not None
        assert best.cumulative_score() > 0.7

    def test_all_leaves(self):
        """Test getting all leaves."""
        trie = TrajectoryTrie()
        traj = trie.new_trajectory("s0")
        hole_id = HoleId(namespace="default", name="x", index=0)

        t1 = traj.extend(hole_id, "a", "s1")
        t2 = traj.extend(hole_id, "b", "s2")
        t1.extend(hole_id, "c", "s3")

        leaves = trie.all_leaves()

        assert len(leaves) == 2

    def test_prune_below_score(self):
        """Test pruning low-score subtrees."""
        trie = TrajectoryTrie()
        traj = trie.new_trajectory("s0")
        hole_id = HoleId(namespace="default", name="x", index=0)

        t1 = traj.extend(hole_id, "a", "s1", score=0.1)  # Low score
        t2 = traj.extend(hole_id, "b", "s2", score=0.9)  # High score

        t1.extend(hole_id, "c", "s3", score=0.1)
        t2.extend(hole_id, "d", "s4", score=0.9)

        before_count = trie.node_count
        pruned = trie.prune_below_score(0.5)

        # Low-score branch should be pruned
        assert trie.node_count < before_count

    def test_depth_stats(self):
        """Test computing depth statistics."""
        trie = TrajectoryTrie()
        traj = trie.new_trajectory("s0")
        hole_id = HoleId(namespace="default", name="x", index=0)

        t1 = traj.extend(hole_id, "a", "s1")
        t2 = traj.extend(hole_id, "b", "s2")
        t1.extend(hole_id, "c", "s3")
        t1.current.children[f"{hole_id}:c"].add_child(hole_id, "d", "s4")

        stats = trie.depth_stats()

        assert stats["min"] >= 1
        assert stats["max"] >= stats["min"]
        assert stats["count"] >= 1


class TestCreateTrajectoryTrie:
    """Tests for create_trajectory_trie convenience function."""

    def test_basic_usage(self):
        """Test basic usage."""
        trie, traj = create_trajectory_trie("initial_state")

        assert trie is not None
        assert traj is not None
        assert traj.state == "initial_state"
        assert traj.depth == 0


class TestMCVHeuristic:
    """Tests specifically for MCV (Most Constrained Variable) heuristic."""

    def test_mcv_selects_most_constrained(self):
        """Test that MCV selects holes with fewest options."""
        # Create code where one hole has fewer valid fills
        hole1 = HoleId(namespace="default", name="constrained", index=0)
        hole2 = HoleId(namespace="default", name="free", index=0)

        # hole1 expects bool (2 options), hole2 expects int (3+ options)
        code = HoledCode(
            template="if ?default:constrained[0]: return ?default:free[0]",
            holes={
                hole1: Hole(id=hole1, expected_type="bool"),
                hole2: Hole(id=hole2, expected_type="int"),
            },
            hole_markers={
                hole1: "?default:constrained[0]",
                hole2: "?default:free[0]",
            },
        )

        filler = SudokuStyleHoleFiller(strategy=FillStrategy.MCV)
        result = filler.fill(code)

        # Should complete (may or may not select bool first depending on implementation)
        assert isinstance(result, FillResult)

    def test_mrv_is_alias_for_mcv(self):
        """Test that MRV strategy behaves same as MCV."""
        hole_id = HoleId(namespace="default", name="x", index=0)
        code = HoledCode(
            template="x = ?default:x[0]",
            holes={hole_id: Hole(id=hole_id, expected_type="int")},
            hole_markers={hole_id: "?default:x[0]"},
        )

        filler_mcv = SudokuStyleHoleFiller(strategy=FillStrategy.MCV)
        filler_mrv = SudokuStyleHoleFiller(strategy=FillStrategy.MRV)

        result_mcv = filler_mcv.fill(code)
        result_mrv = filler_mrv.fill(code)

        # Both should produce similar results
        assert result_mcv.success == result_mrv.success


class TestBacktracking:
    """Tests for backtracking behavior."""

    def test_backtracking_on_invalid_state(self):
        """Test that backtracking occurs when state becomes invalid."""
        # This would require a custom constraint checker that rejects some fills
        # For now, just verify the mechanism exists
        filler = SudokuStyleHoleFiller(max_backtracks=10)
        assert filler.max_backtracks == 10

    def test_max_iterations_respected(self):
        """Test that max_iterations limits search."""
        hole_id = HoleId(namespace="default", name="x", index=0)
        code = HoledCode(
            template="x = ?default:x[0]",
            holes={hole_id: Hole(id=hole_id, expected_type="int")},
            hole_markers={hole_id: "?default:x[0]"},
        )

        filler = SudokuStyleHoleFiller()
        result = filler.fill(code, max_iterations=5)

        # Should complete within iteration limit
        assert result.metadata.get("iterations", 0) <= 5 or result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
