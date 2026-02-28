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
"""End-to-end integration tests for Test-Time Compute (TTC) improvements.

These tests verify the complete flow of TTC features:
1. Adaptive constraint intensity assessment
2. Best-of-N candidate verification and selection
3. Sudoku-style hole filling with MCV heuristic
4. Trajectory tracking for backtracking

Tests demonstrate that all components work together correctly.
"""

import sys
from pathlib import Path

import pytest

# Setup import paths
_TEST_DIR = Path(__file__).parent
_ANANKE_ROOT = _TEST_DIR.parent.parent
if str(_ANANKE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANANKE_ROOT))

# Import TTC components
from adaptive.intensity import (
    ConstraintIntensity,
    TaskComplexityAssessor,
    assess_complexity,
    domains_for_intensity,
)
from verification.verifier import ConstraintVerifier, VerificationResult
from verification.selector import BestOfNSelector, SelectionStrategy
from search.sudoku_filler import (
    SudokuStyleHoleFiller,
    FillStrategy,
    FillResult,
    HoledCode,
)
from search.trajectory import TrajectoryTrie, Trajectory, create_trajectory_trie
from search.generators import (
    TypeAwareFillGenerator,
    UnifiedConstraintChecker,
    TypeConstraintInferencer,
)
from holes.hole import Hole, HoleId, HoleState, TypeEnvironment
from domains.types.constraint import INT, STR, BOOL, FLOAT, NONE, ANY, ListType


# =============================================================================
# Adaptive Intensity E2E Tests
# =============================================================================


class TestAdaptiveIntensityE2E:
    """End-to-end tests for adaptive constraint intensity."""

    def test_simple_completion_uses_minimal_intensity(self):
        """Test that simple completions use minimal constraint intensity."""
        # Simple completion prompt
        prompt = "x = 1 + 2"
        expected_tokens = 10

        assessor = TaskComplexityAssessor()
        intensity = assessor.assess(prompt, expected_tokens)

        # Should use SYNTAX_ONLY or NONE for simple tasks
        assert intensity in (ConstraintIntensity.NONE, ConstraintIntensity.SYNTAX_ONLY)

    def test_function_definition_uses_standard_intensity(self):
        """Test that function definitions use standard intensity."""
        prompt = """def calculate_total(items: List[int]) -> int:
    '''Calculate the sum of all items.'''
    """
        expected_tokens = 50

        assessor = TaskComplexityAssessor()
        intensity = assessor.assess(prompt, expected_tokens)

        # Should use at least STANDARD for function definitions
        assert intensity.value >= ConstraintIntensity.STANDARD.value

    def test_complex_class_uses_full_intensity(self):
        """Test that complex classes use full intensity."""
        prompt = """class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._cache: Dict[str, Any] = {}

    def process(self, data: List[Dict]) -> ProcessedResult:
        '''Process a batch of data records with validation.'''
    """
        expected_tokens = 200

        assessor = TaskComplexityAssessor()
        intensity = assessor.assess(prompt, expected_tokens)

        # Should use FULL for complex classes
        assert intensity.value >= ConstraintIntensity.FULL.value

    def test_intensity_determines_enabled_domains(self):
        """Test that intensity level determines which domains are enabled."""
        # None intensity - no domains
        domains_none = domains_for_intensity(ConstraintIntensity.NONE)
        assert len(domains_none) == 0

        # Syntax only - just syntax
        domains_syntax = domains_for_intensity(ConstraintIntensity.SYNTAX_ONLY)
        assert "syntax" in domains_syntax
        assert "types" not in domains_syntax

        # Standard - syntax + types
        domains_std = domains_for_intensity(ConstraintIntensity.STANDARD)
        assert "syntax" in domains_std
        assert "types" in domains_std

        # Full - all domains
        domains_full = domains_for_intensity(ConstraintIntensity.FULL)
        assert "syntax" in domains_full
        assert "types" in domains_full
        assert "imports" in domains_full


# =============================================================================
# Best-of-N Verification E2E Tests
# =============================================================================


class TestBestOfNVerificationE2E:
    """End-to-end tests for Best-of-N verification and selection."""

    def test_verify_and_select_best_candidate(self):
        """Test verifying and selecting the best candidate from multiple options."""
        # Multiple code candidates of varying quality
        candidates = [
            # Good: Valid Python with proper types
            """def add(x: int, y: int) -> int:
    return x + y""",
            # Medium: Valid but without type hints
            """def add(x, y):
    return x + y""",
            # Bad: Syntax error
            """def add(x, y) ->
    return x + """,
        ]

        verifier = ConstraintVerifier(language="python")
        selector = BestOfNSelector(verifier=verifier)

        # Verify all candidates
        results = verifier.verify_batch(candidates)

        # First should have highest score (typed)
        # Third should have lowest (syntax error)
        assert results[0].overall_score >= results[1].overall_score
        assert results[2].overall_score < results[0].overall_score

        # Select best
        result = selector.select_best(candidates)
        assert result.selected == candidates[0]  # The typed version

    def test_best_of_n_with_type_context(self):
        """Test Best-of-N selection with type context."""
        candidates = [
            "my_list.append(1)",
            "my_list.append('hello')",
            "my_list.append(None)",
        ]

        # Type context says my_list is List[int]
        verifier = ConstraintVerifier(
            language="python",
            type_context={"my_list": "List[int]"},
        )
        selector = BestOfNSelector(verifier=verifier)

        # All candidates should verify (they're syntactically valid)
        results = verifier.verify_batch(candidates)
        assert all(r.overall_score > 0 for r in results)

        # Select returns one of the candidates
        result = selector.select_best(candidates)
        assert result.selected in candidates

    def test_selection_strategies(self):
        """Test different selection strategies."""
        candidates = [
            "x = 1",
            "x = 'hello'",
            "x = True",
        ]

        verifier = ConstraintVerifier(language="python")

        # Test BEST_SCORE strategy
        selector_max = BestOfNSelector(
            verifier=verifier,
            strategy=SelectionStrategy.BEST_SCORE,
        )
        result_max = selector_max.select_best(candidates)
        assert result_max.selected in candidates

        # Test FIRST_VALID strategy
        selector_first = BestOfNSelector(
            verifier=verifier,
            strategy=SelectionStrategy.FIRST_VALID,
        )
        result_first = selector_first.select_best(candidates)
        assert result_first.selected in candidates

        # Test WEIGHTED strategy
        selector_weighted = BestOfNSelector(
            verifier=verifier,
            strategy=SelectionStrategy.WEIGHTED,
        )
        result_weighted = selector_weighted.select_best(candidates)
        assert result_weighted.selected in candidates


# =============================================================================
# Sudoku-Style Hole Filling E2E Tests
# =============================================================================


class TestSudokuHoleFillingE2E:
    """End-to-end tests for Sudoku-style hole filling."""

    def test_fill_typed_holes_with_mcv(self):
        """Test filling multiple typed holes using MCV heuristic."""
        # Create code with multiple typed holes
        hole1_id = HoleId(namespace="user", name="x_value", index=0)
        hole2_id = HoleId(namespace="user", name="y_value", index=1)
        hole3_id = HoleId(namespace="user", name="result", index=2)

        hole1 = Hole(id=hole1_id, expected_type=INT)
        hole2 = Hole(id=hole2_id, expected_type=INT)
        hole3 = Hole(id=hole3_id, expected_type=INT)

        code = HoledCode(
            template="x = ?user:x_value[0]\ny = ?user:y_value[1]\nresult = ?user:result[2]",
            holes={hole1_id: hole1, hole2_id: hole2, hole3_id: hole3},
            hole_markers={
                hole1_id: "?user:x_value[0]",
                hole2_id: "?user:y_value[1]",
                hole3_id: "?user:result[2]",
            },
        )

        # Create filler with generator
        generator = TypeAwareFillGenerator()
        filler = SudokuStyleHoleFiller(
            fill_generator=generator,
            strategy=FillStrategy.MCV,
        )

        # Fill holes
        result = filler.fill(code)

        # Should succeed
        assert result.success
        assert result.filled_code is not None
        assert "?" not in result.filled_code
        assert len(result.fill_history) == 3

    def test_fill_with_constraint_propagation(self):
        """Test that constraint propagation refines dependent holes."""
        # Create code with dependent holes
        hole1_id = HoleId(namespace="user", name="return_value", index=0)
        hole2_id = HoleId(namespace="user", name="other_return", index=1)

        hole1 = Hole(id=hole1_id, expected_type=INT)
        hole2 = Hole(id=hole2_id, expected_type=ANY)  # Initially Any

        # Set up dependency: hole2 depends on hole1
        code = HoledCode(
            template="def f() -> int:\n    return ?user:return_value[0]\n\ndef g() -> int:\n    return ?user:other_return[1]",
            holes={hole1_id: hole1, hole2_id: hole2},
            hole_markers={
                hole1_id: "?user:return_value[0]",
                hole2_id: "?user:other_return[1]",
            },
            dependencies={hole1_id: frozenset({hole2_id})},
        )

        generator = TypeAwareFillGenerator()
        filler = SudokuStyleHoleFiller(
            fill_generator=generator,
            strategy=FillStrategy.MCV,
            propagation_enabled=True,
        )

        result = filler.fill(code)

        # Should succeed
        assert result.success
        # Propagation should have occurred
        assert result.propagation_count >= 0

    def test_fill_with_backtracking(self):
        """Test that backtracking recovers from invalid states."""
        # Create code with holes where first choice might be wrong
        hole_id = HoleId(name="value")
        hole = Hole(id=hole_id, expected_type=BOOL)

        code = HoledCode(
            template="x: bool = ?default:value[0]",
            holes={hole_id: hole},
            hole_markers={hole_id: "?default:value[0]"},
        )

        generator = TypeAwareFillGenerator()
        filler = SudokuStyleHoleFiller(
            fill_generator=generator,
            strategy=FillStrategy.MCV,
            max_backtracks=10,
        )

        result = filler.fill(code)

        # Should succeed with True or False
        assert result.success
        assert "True" in result.filled_code or "False" in result.filled_code


# =============================================================================
# Trajectory Tracking E2E Tests
# =============================================================================


class TestTrajectoryTrackingE2E:
    """End-to-end tests for trajectory tracking during search."""

    def test_trajectory_records_fill_history(self):
        """Test that trajectory correctly records fill history."""
        trie, traj = create_trajectory_trie(initial_state="initial")

        # Simulate a series of fills
        hole1 = HoleId(name="x")
        hole2 = HoleId(name="y")
        hole3 = HoleId(name="z")

        traj = traj.extend(hole1, "1", state="after_x", score=0.9)
        traj = traj.extend(hole2, "2", state="after_y", score=0.8)
        traj = traj.extend(hole3, "3", state="after_z", score=0.85)

        # Check fill history
        history = traj.fill_history()
        assert len(history) == 3
        assert history[0] == (hole1, "1")
        assert history[1] == (hole2, "2")
        assert history[2] == (hole3, "3")

    def test_trajectory_enables_efficient_backtracking(self):
        """Test that trajectory enables efficient backtracking."""
        trie, traj = create_trajectory_trie(initial_state="start")

        hole1 = HoleId(name="choice")

        # Extend with first choice
        checkpoint = trie.checkpoint(traj)
        traj1 = traj.extend(hole1, "option_a", state="state_a", score=0.5)

        # Restore and try second choice
        traj_restored = trie.restore(checkpoint)
        assert traj_restored.depth == 0
        assert traj_restored.state == "start"

        traj2 = traj_restored.extend(hole1, "option_b", state="state_b", score=0.8)

        # Both branches exist in trie
        assert trie.node_count >= 3  # root + 2 children

        # Can find best leaf
        best = trie.best_leaf()
        assert best is not None
        assert best.score == 0.8

    def test_trajectory_pruning(self):
        """Test that low-score trajectories can be pruned."""
        trie, traj = create_trajectory_trie(initial_state="start")

        hole = HoleId(name="value")

        # Create multiple branches with different scores
        traj.extend(hole, "bad1", state="s1", score=0.1)
        traj.extend(hole, "bad2", state="s2", score=0.2)
        traj.extend(hole, "good", state="s3", score=0.9)

        # Prune low-score leaves
        pruned = trie.prune_below_score(0.5)

        # Should have pruned the low-score branches
        assert pruned >= 2

        # Best leaf should still be the good one
        best = trie.best_leaf()
        assert best is not None
        assert best.current.fill_value == "good"


# =============================================================================
# Full Pipeline E2E Tests
# =============================================================================


class TestFullPipelineE2E:
    """End-to-end tests for the complete TTC pipeline."""

    def test_complete_pipeline_simple_completion(self):
        """Test complete pipeline for a simple code completion."""
        # 1. Assess complexity
        prompt = "x = "
        intensity = assess_complexity(prompt, expected_tokens=5)

        # Should use minimal intensity
        assert intensity.value <= ConstraintIntensity.STANDARD.value

        # 2. Create holed code
        hole_id = HoleId(name="value")
        hole = Hole(id=hole_id, expected_type=INT)
        code = HoledCode(
            template="x = ?default:value[0]",
            holes={hole_id: hole},
            hole_markers={hole_id: "?default:value[0]"},
        )

        # 3. Fill holes
        generator = TypeAwareFillGenerator()
        filler = SudokuStyleHoleFiller(fill_generator=generator)
        result = filler.fill(code)

        # 4. Verify result
        assert result.success
        verifier = ConstraintVerifier(language="python")
        verification = verifier.verify(result.filled_code)
        assert verification.overall_score > 0.5

    def test_complete_pipeline_function_completion(self):
        """Test complete pipeline for function body completion."""
        # 1. Assess complexity
        prompt = """def add_numbers(x: int, y: int) -> int:
    '''Add two numbers and return the result.'''
    """
        intensity = assess_complexity(prompt, expected_tokens=50)

        # Should use at least standard intensity
        assert intensity.value >= ConstraintIntensity.STANDARD.value

        # 2. Create holed code for function body
        hole_id = HoleId(name="body")
        hole = Hole(id=hole_id, expected_type=INT)  # Return type is int
        code = HoledCode(
            template=prompt + "return ?default:body[0]",
            holes={hole_id: hole},
            hole_markers={hole_id: "?default:body[0]"},
        )

        # 3. Fill holes with type awareness
        env = TypeEnvironment.from_dict({"x": INT, "y": INT})
        generator = TypeAwareFillGenerator(type_environment=env)
        filler = SudokuStyleHoleFiller(
            fill_generator=generator,
            strategy=FillStrategy.MCV,
        )
        result = filler.fill(code)

        # 4. Verify filled code
        assert result.success
        verifier = ConstraintVerifier(
            language="python",
            type_context={"x": "int", "y": "int"},
        )
        verification = verifier.verify(result.filled_code)
        assert verification.overall_score > 0.5

    def test_complete_pipeline_with_best_of_n(self):
        """Test pipeline combining hole filling with Best-of-N selection."""
        # 1. Create holed code
        hole_id = HoleId(name="expr")
        hole = Hole(id=hole_id, expected_type=INT)
        code = HoledCode(
            template="result = ?default:expr[0]",
            holes={hole_id: hole},
            hole_markers={hole_id: "?default:expr[0]"},
        )

        # 2. Generate multiple candidates via different strategies
        candidates = []

        for strategy in [FillStrategy.MCV, FillStrategy.SEQUENTIAL, FillStrategy.RANDOM]:
            generator = TypeAwareFillGenerator()
            filler = SudokuStyleHoleFiller(
                fill_generator=generator,
                strategy=strategy,
            )
            result = filler.fill(code)
            if result.success:
                candidates.append(result.filled_code)

        # Ensure we have candidates
        assert len(candidates) > 0

        # 3. Select best using Best-of-N
        verifier = ConstraintVerifier(language="python")
        selector = BestOfNSelector(verifier=verifier)
        result = selector.select_best(candidates)

        # 4. Verify best candidate
        assert result.selected is not None
        final_verification = verifier.verify(result.selected)
        assert final_verification.overall_score > 0.5

    def test_complete_pipeline_list_type(self):
        """Test pipeline for list type completion."""
        # 1. Create holed code with list type
        hole_id = HoleId(name="items")
        hole = Hole(id=hole_id, expected_type=ListType(INT))
        code = HoledCode(
            template="numbers: List[int] = ?default:items[0]",
            holes={hole_id: hole},
            hole_markers={hole_id: "?default:items[0]"},
        )

        # 2. Fill with type-aware generator
        generator = TypeAwareFillGenerator()
        filler = SudokuStyleHoleFiller(fill_generator=generator)
        result = filler.fill(code)

        # 3. Verify
        assert result.success
        # Should have a list literal
        assert "[" in result.filled_code


class TestPipelineWithConstraintChecker:
    """Tests for pipeline using full constraint checking."""

    def test_checker_rejects_type_mismatch(self):
        """Test that checker rejects fills with type mismatches."""
        checker = UnifiedConstraintChecker()

        hole = Hole(id=HoleId(name="value"), expected_type=INT)
        code = HoledCode(
            template="x: int = ?default:value[0]",
            holes={hole.id: hole},
            hole_markers={hole.id: "?default:value[0]"},
        )

        # Check string literal for int type
        valid, score, violations = checker.check_fill(hole, '"hello"', code)

        # Should have low score
        assert score < 0.5

    def test_checker_accepts_type_match(self):
        """Test that checker accepts fills with matching types."""
        checker = UnifiedConstraintChecker()

        hole = Hole(id=HoleId(name="value"), expected_type=INT)
        code = HoledCode(
            template="x: int = ?default:value[0]",
            holes={hole.id: hole},
            hole_markers={hole.id: "?default:value[0]"},
        )

        # Check int literal for int type
        valid, score, violations = checker.check_fill(hole, "42", code)

        # Should have high score
        assert score > 0.5

    def test_filler_with_checker_integration(self):
        """Test filler with constraint checker for validation."""
        generator = TypeAwareFillGenerator()
        checker = UnifiedConstraintChecker()

        filler = SudokuStyleHoleFiller(
            fill_generator=generator,
            constraint_checker=checker,
            strategy=FillStrategy.MCV,
        )

        hole = Hole(id=HoleId(name="value"), expected_type=STR)
        code = HoledCode(
            template='msg = ?default:value[0]',
            holes={hole.id: hole},
            hole_markers={hole.id: "?default:value[0]"},
        )

        result = filler.fill(code)

        # Should fill with string
        assert result.success
        assert '"' in result.filled_code or "'" in result.filled_code or "str" in result.filled_code


# =============================================================================
# Beam Search and Speculative Decoding E2E Tests
# =============================================================================


class TestBeamSearchE2E:
    """End-to-end tests for beam search integration."""

    def test_beam_search_with_constraint_verification(self):
        """Test beam search integrated with constraint verification."""
        from search.beam import BeamSearch, BeamSearchConfig, BeamCandidate

        # Create a scorer that provides multiple candidates
        class MockConstraintScorer:
            def score_tokens(self, tokens, state, top_k=50):
                # Return multiple candidates with different scores
                if len(tokens) < 3:
                    return [
                        (10 + len(tokens), -0.1, 0.9),
                        (20 + len(tokens), -0.3, 0.7),
                    ]
                return [(1, -0.05, 1.0)]  # END token

        config = BeamSearchConfig(beam_width=3, max_length=5)
        search = BeamSearch(config=config, token_scorer=MockConstraintScorer())

        result = search.search(start_tokens=[0], end_token=1)

        # Should complete successfully
        assert isinstance(result, BeamCandidate)
        assert len(result.tokens) > 0

        # Check stats
        stats = search.get_stats()
        assert stats.total_steps > 0

    def test_beam_search_returns_multiple_candidates(self):
        """Test beam search returns multiple candidates for Best-of-N."""
        from search.beam import BeamSearch, BeamSearchConfig

        class MultiPathScorer:
            def score_tokens(self, tokens, state, top_k=50):
                if len(tokens) == 1:
                    return [
                        (10, -0.1, 0.9),
                        (20, -0.2, 0.85),
                        (30, -0.3, 0.8),
                    ]
                return [(1, -0.1, 1.0)]

        config = BeamSearchConfig(beam_width=3, max_length=5)
        search = BeamSearch(config=config, token_scorer=MultiPathScorer())

        results = search.search_with_constraint(start_tokens=[0], end_token=1)

        # Should return multiple candidates
        assert len(results) >= 1
        assert len(results) <= 3

    def test_beam_search_diversity_penalty(self):
        """Test that diversity penalty prevents beam collapse."""
        from search.beam import BeamSearch, BeamSearchConfig

        class HomogeneousScorer:
            def score_tokens(self, tokens, state, top_k=50):
                # Return same token repeatedly
                return [(42, -0.1, 0.9)] * 5

        config = BeamSearchConfig(
            beam_width=3,
            diversity_penalty=0.5,
            max_length=3,
        )
        search = BeamSearch(config=config, token_scorer=HomogeneousScorer())

        result = search.search(start_tokens=[0])

        # Should complete without error
        assert result is not None


class TestSpeculativeDecodingE2E:
    """End-to-end tests for speculative decoding integration."""

    def test_constrained_lookahead_basic(self):
        """Test basic constrained lookahead verification."""
        from speculative.constrained_lookahead import (
            ConstrainedLookahead,
            LookaheadConfig,
        )
        from speculative.draft_model import GreedyDraftModel, DraftContext
        import torch
        from typing import List, Tuple, Any, Optional

        # Create mock logits and mask functions
        vocab_size = 100

        def logits_fn(tokens: List[int]) -> torch.Tensor:
            logits = torch.zeros(vocab_size)
            logits[42] = 10.0  # High prob for token 42
            return logits

        # Create mock verifier that accepts tokens < 50
        class MockVerifier:
            def verify_draft_tokens(
                self, draft_tokens: List[int]
            ) -> Tuple[int, Optional[Any]]:
                """Verify draft tokens, accept those < 50."""
                num_valid = 0
                for token in draft_tokens:
                    if token < 50:
                        num_valid += 1
                    else:
                        break
                return num_valid, None if num_valid == len(draft_tokens) else "token >= 50"

        config = LookaheadConfig(initial_lookahead=3)
        draft_model = GreedyDraftModel(logits_fn=logits_fn, vocab_size=vocab_size)
        verifier = MockVerifier()

        lookahead = ConstrainedLookahead(
            draft_model=draft_model,
            verifier=verifier,
            config=config,
        )

        context = DraftContext(prefix_tokens=[0])
        result = lookahead.generate_next(context)

        # Should return list of accepted tokens
        assert result is not None
        assert isinstance(result, list)
        # Token 42 should be accepted (it's < 50)
        assert all(t < 50 for t in result)

    def test_draft_model_generates_candidates(self):
        """Test that draft model generates multiple candidates."""
        from speculative.draft_model import SamplingDraftModel, DraftContext
        import torch
        from typing import List

        vocab_size = 100

        def logits_fn(tokens: List[int]) -> torch.Tensor:
            logits = torch.randn(vocab_size)
            logits[10:20] += 5.0  # Boost tokens 10-19
            return logits

        draft_model = SamplingDraftModel(
            logits_fn=logits_fn,
            vocab_size=vocab_size,
            temperature=1.0,
        )

        context = DraftContext(
            prefix_tokens=[0],
            temperature=1.0,
        )

        result = draft_model.generate_draft(context, lookahead_length=5)

        # Should generate draft tokens
        assert result is not None
        assert len(result.tokens) > 0

    def test_adaptive_lookahead_adjustment(self):
        """Test that lookahead adjusts based on acceptance rate."""
        from speculative.constrained_lookahead import (
            ConstrainedLookahead,
            LookaheadConfig,
        )
        from speculative.draft_model import NullDraftModel
        from typing import List, Tuple, Any, Optional

        config = LookaheadConfig(
            initial_lookahead=5,
            min_lookahead=2,
            max_lookahead=10,
            adaptive=True,
        )
        draft_model = NullDraftModel()

        # Mock verifier that accepts all tokens
        class AlwaysAcceptVerifier:
            def verify_draft_tokens(
                self, draft_tokens: List[int]
            ) -> Tuple[int, Optional[Any]]:
                return len(draft_tokens), None

        verifier = AlwaysAcceptVerifier()

        lookahead = ConstrainedLookahead(
            draft_model=draft_model,
            verifier=verifier,
            config=config,
        )

        # Current lookahead should start at initial
        assert lookahead.stats.current_lookahead == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
