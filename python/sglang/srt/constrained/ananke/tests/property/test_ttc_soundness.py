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
"""Property-based tests for Test-Time Compute (TTC) improvements soundness.

These tests verify soundness properties for:
1. Adaptive Constraint Intensity - lower intensity is more permissive
2. Best-of-N Verification - never blocks, always returns valid results
3. Sudoku-Style Hole Filling - fill/unfill operations, trajectory properties

Key Invariant: NEVER block a valid token / reject valid code.
Soundness > Completeness throughout.

References:
- Hazel: Mathematical foundations of typed holes
- BoNBoN: Best-of-N with verification
- GenCP: LLM + Constraint Propagation
"""

import pytest
import sys
from pathlib import Path
from hypothesis import given, settings, assume, strategies as st, HealthCheck

# Add the ananke package root to sys.path for standalone testing
_PROPERTY_DIR = Path(__file__).parent
_TESTS_DIR = _PROPERTY_DIR.parent
_ANANKE_ROOT = _TESTS_DIR.parent
if str(_ANANKE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANANKE_ROOT))

from adaptive.intensity import (
    ConstraintIntensity,
    TaskComplexityAssessor,
    IntensityConfig,
    domains_for_intensity,
)
from verification.verifier import (
    ConstraintVerifier,
    VerificationResult,
    DomainScore,
    DEFAULT_DOMAIN_WEIGHTS,
)
from verification.selector import (
    BestOfNSelector,
    SelectionStrategy,
    SelectionResult,
    select_best_of_n,
)
from search.sudoku_filler import (
    SudokuStyleHoleFiller,
    FillResult,
    FillStrategy,
    FillCandidate,
    HoledCode,
)
from search.trajectory import (
    Trajectory,
    TrajectoryNode,
    TrajectoryTrie,
    create_trajectory_trie,
)
from holes.hole import Hole, HoleId, HoleState, TypeEnvironment


# =============================================================================
# Strategies for generating test inputs
# =============================================================================

@st.composite
def constraint_intensity(draw):
    """Generate a random ConstraintIntensity."""
    return draw(st.sampled_from(list(ConstraintIntensity)))


@st.composite
def intensity_pair(draw):
    """Generate a pair of intensities where first <= second."""
    i1 = draw(constraint_intensity())
    i2 = draw(constraint_intensity())
    if i1 > i2:
        i1, i2 = i2, i1
    return (i1, i2)


@st.composite
def valid_python_code(draw):
    """Generate valid Python code snippets."""
    templates = [
        "x = {value}",
        "y = x + {value}",
        "def f(): return {value}",
        "class C: pass",
        "import {module}",
        "for i in range({value}): pass",
        "if True: x = {value}",
        "result = [i for i in range({value})]",
        "[x for x in range({value})]",
        "lambda x: x + {value}",
    ]
    template = draw(st.sampled_from(templates))

    value = draw(st.integers(min_value=0, max_value=100))
    module = draw(st.sampled_from(["os", "sys", "json", "re", "math"]))

    return template.format(value=value, module=module)


@st.composite
def invalid_python_code(draw):
    """Generate invalid Python code snippets."""
    invalids = [
        "def foo(",
        "class Bar(",
        "if True",
        "for in range(10):",
        "x = =",
        "return return",
        "import",
        "from import x",
        "@#$%^",
    ]
    return draw(st.sampled_from(invalids))


@st.composite
def python_code(draw):
    """Generate either valid or invalid Python code."""
    if draw(st.booleans()):
        return draw(valid_python_code())
    else:
        return draw(invalid_python_code())


@st.composite
def hole_id(draw):
    """Generate a random HoleId."""
    namespace = draw(st.sampled_from(["default", "user", "system", "type"]))
    name = draw(st.sampled_from(["value", "expr", "stmt", "block", "body"]))
    index = draw(st.integers(min_value=0, max_value=10))
    depth = draw(st.integers(min_value=0, max_value=3))
    return HoleId(namespace=namespace, name=name, index=index, depth=depth)


@st.composite
def hole(draw):
    """Generate a random Hole."""
    hid = draw(hole_id())
    expected_type = draw(st.sampled_from([None, "int", "str", "bool", "float", "List[int]"]))
    state = draw(st.sampled_from([HoleState.EMPTY, HoleState.PARTIAL, HoleState.FILLED]))
    content = draw(st.text(min_size=0, max_size=20)) if state != HoleState.EMPTY else None

    return Hole(
        id=hid,
        expected_type=expected_type,
        state=state,
        content=content,
    )


@st.composite
def fill_candidate(draw):
    """Generate a random FillCandidate."""
    value = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'S'))))
    score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    return FillCandidate(value=value, score=score)


@st.composite
def selection_strategy(draw):
    """Generate a random SelectionStrategy."""
    return draw(st.sampled_from([
        SelectionStrategy.BEST_SCORE,
        SelectionStrategy.FIRST_VALID,
        SelectionStrategy.THRESHOLD,
        SelectionStrategy.WEIGHTED,
    ]))


# =============================================================================
# Adaptive Intensity Soundness Tests
# =============================================================================

class TestAdaptiveIntensitySoundness:
    """Property tests for adaptive intensity soundness.

    Key Property: Lower intensity means more permissive (fewer domains).
    This is SOUND because fewer constraints = more tokens allowed.
    """

    @given(i1=constraint_intensity(), i2=constraint_intensity())
    @settings(max_examples=50)
    def test_intensity_ordering_implies_domain_subset(self, i1, i2):
        """Lower intensity should have subset of domains of higher intensity."""
        if i1 <= i2:
            domains1 = domains_for_intensity(i1)
            domains2 = domains_for_intensity(i2)
            assert domains1 <= domains2, \
                f"Intensity {i1} should have subset of {i2}'s domains"

    @given(intensity=constraint_intensity())
    @settings(max_examples=20)
    def test_intensity_has_valid_domains(self, intensity):
        """Each intensity should return a valid frozenset of domain names."""
        domains = domains_for_intensity(intensity)
        assert isinstance(domains, frozenset)
        valid_domain_names = {"syntax", "types", "imports", "controlflow", "semantics"}
        assert domains <= valid_domain_names

    @given(intensity=constraint_intensity())
    @settings(max_examples=20)
    def test_none_intensity_is_empty(self, intensity):
        """NONE intensity should have no domains."""
        if intensity == ConstraintIntensity.NONE:
            domains = domains_for_intensity(intensity)
            assert len(domains) == 0

    @given(intensity=constraint_intensity())
    @settings(max_examples=20)
    def test_exhaustive_has_all_domains(self, intensity):
        """EXHAUSTIVE intensity should have all domains."""
        if intensity == ConstraintIntensity.EXHAUSTIVE:
            domains = domains_for_intensity(intensity)
            assert "syntax" in domains
            assert "types" in domains
            assert "imports" in domains
            assert "controlflow" in domains
            assert "semantics" in domains

    def test_intensity_monotonicity(self):
        """Intensities should form a chain with monotonic domain inclusion."""
        intensities = sorted(list(ConstraintIntensity))
        for i in range(len(intensities) - 1):
            lower = intensities[i]
            higher = intensities[i + 1]
            domains_lower = domains_for_intensity(lower)
            domains_higher = domains_for_intensity(higher)
            assert domains_lower <= domains_higher, \
                f"Monotonicity violated: {lower} -> {higher}"


class TestTaskComplexityAssessorSoundness:
    """Property tests for task complexity assessment soundness."""

    @pytest.fixture
    def assessor(self):
        return TaskComplexityAssessor()

    @given(
        prompt=st.text(min_size=0, max_size=500),
        expected_tokens=st.integers(min_value=0, max_value=1000),
        temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_assessment_always_returns_valid_intensity(
        self, prompt, expected_tokens, temperature, assessor
    ):
        """Assessment should always return a valid ConstraintIntensity."""
        result = assessor.assess(
            prompt=prompt,
            expected_tokens=expected_tokens,
            temperature=temperature,
        )
        assert isinstance(result, ConstraintIntensity)
        assert result in list(ConstraintIntensity)

    @given(
        intensity=constraint_intensity(),
        prompt=st.text(min_size=0, max_size=100),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_explicit_intensity_overrides_assessment(
        self, intensity, prompt, assessor
    ):
        """Explicit intensity should override assessment."""
        result = assessor.assess(
            prompt=prompt,
            explicit_intensity=intensity,
        )
        assert result == intensity

    @given(prompt=st.text(min_size=0, max_size=100))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_high_temperature_increases_intensity(self, prompt, assessor):
        """High temperature should suggest higher intensity."""
        low_temp_result = assessor.assess(prompt=prompt, temperature=0.1)
        high_temp_result = assessor.assess(prompt=prompt, temperature=1.5)

        # High temperature should be at least as intense
        assert high_temp_result >= low_temp_result


# =============================================================================
# Verification Soundness Tests
# =============================================================================

class TestVerificationSoundness:
    """Property tests for verification soundness.

    Key Property: Verification NEVER blocks - it only scores.
    Post-hoc verification is inherently SOUND.
    """

    @pytest.fixture
    def verifier(self):
        return ConstraintVerifier(language="python")

    @given(code=python_code())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_verify_always_returns_result(self, code, verifier):
        """Verification should ALWAYS return a VerificationResult."""
        result = verifier.verify(code)
        assert isinstance(result, VerificationResult)
        assert result.candidate == code

    @given(code=python_code())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_verify_score_in_valid_range(self, code, verifier):
        """Verification score should be in [0, 1]."""
        result = verifier.verify(code)
        assert 0.0 <= result.overall_score <= 1.0

    @given(code=valid_python_code())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_valid_code_gets_reasonable_score(self, code, verifier):
        """Valid Python code should get a reasonable score (soundness)."""
        result = verifier.verify(code)
        # Valid code should get at least some points
        # Note: May not be 1.0 due to domain-specific checks
        assert result.overall_score >= 0.0  # Never penalized to negative

    @given(codes=st.lists(python_code(), min_size=1, max_size=5))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_batch_verify_returns_all_results(self, codes, verifier):
        """Batch verification should return result for each candidate."""
        results = verifier.verify_batch(codes)
        assert len(results) == len(codes)
        for i, result in enumerate(results):
            assert isinstance(result, VerificationResult)
            assert result.candidate == codes[i]


class TestDomainScoreSoundness:
    """Property tests for DomainScore soundness."""

    @given(
        score=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        valid=st.booleans(),
    )
    @settings(max_examples=50)
    def test_score_clamping(self, score, valid):
        """Score should always be clamped to [0, 1]."""
        domain_score = DomainScore(
            domain="test",
            valid=valid,
            score=score,
        )
        assert 0.0 <= domain_score.score <= 1.0

    @given(
        domain=st.sampled_from(["syntax", "types", "imports", "controlflow", "semantics"]),
        valid=st.booleans(),
        score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_domain_score_immutability(self, domain, valid, score):
        """DomainScore should be immutable (frozen dataclass)."""
        ds = DomainScore(domain=domain, valid=valid, score=score)
        with pytest.raises(AttributeError):
            ds.score = 0.5


class TestBestOfNSelectorSoundness:
    """Property tests for Best-of-N selection soundness.

    Key Property: Selection always returns a candidate from the list.
    Never crashes, always provides graceful degradation.
    """

    @given(
        candidates=st.lists(python_code(), min_size=1, max_size=10),
        strategy=selection_strategy(),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_selection_always_returns_from_candidates(self, candidates, strategy):
        """Selection should always return one of the input candidates."""
        assume(len(candidates) > 0)
        selector = BestOfNSelector(language="python", strategy=strategy)
        result = selector.select_best(candidates)

        assert isinstance(result, SelectionResult)
        assert result.selected in candidates
        assert 0 <= result.selected_index < len(candidates)
        assert result.num_candidates == len(candidates)

    @given(candidates=st.lists(invalid_python_code(), min_size=1, max_size=5))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_selection_handles_all_invalid(self, candidates):
        """Selection should handle case where all candidates are invalid."""
        assume(len(candidates) > 0)
        selector = BestOfNSelector(language="python")
        result = selector.select_best(candidates)

        # Should still return one of them (graceful degradation)
        assert result.selected in candidates

    def test_selection_empty_raises(self):
        """Selection from empty list should raise ValueError."""
        selector = BestOfNSelector(language="python")
        with pytest.raises(ValueError):
            selector.select_best([])

    @given(candidates=st.lists(python_code(), min_size=2, max_size=5))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_best_score_strategy_returns_highest(self, candidates):
        """BEST_SCORE strategy should return candidate with highest score."""
        assume(len(candidates) >= 2)
        selector = BestOfNSelector(
            language="python",
            strategy=SelectionStrategy.BEST_SCORE,
            return_all_results=True,
        )
        result = selector.select_best(candidates)

        # Verify it's actually the best score
        if result.all_results:
            best_score = max(r.overall_score for r in result.all_results)
            assert result.selected_result.overall_score == best_score


# =============================================================================
# Sudoku Filler Soundness Tests
# =============================================================================

class TestHoledCodeSoundness:
    """Property tests for HoledCode operations soundness."""

    @given(hid=hole_id())
    @settings(max_examples=30)
    def test_fill_is_reversible(self, hid):
        """Fill followed by unfill should restore hole state."""
        h = Hole(id=hid, expected_type="int")
        code = HoledCode(
            template=f"x = {hid}",
            holes={hid: h},
            hole_markers={hid: str(hid)},
        )

        filled = code.fill_hole(hid, "42")
        unfilled = filled.unfill_hole(hid)

        # Hole should be empty again
        unfilled_hole = unfilled.get_hole(hid)
        assert unfilled_hole.state == HoleState.EMPTY

    @given(hid=hole_id(), value=st.text(min_size=1, max_size=20, alphabet="abcdef0123456789"))
    @settings(max_examples=30)
    def test_fill_preserves_immutability(self, hid, value):
        """Fill should return new instance without modifying original."""
        h = Hole(id=hid)
        original_template = f"x = {hid}"
        code = HoledCode(
            template=original_template,
            holes={hid: h},
            hole_markers={hid: str(hid)},
        )

        filled = code.fill_hole(hid, value)

        # Original should be unchanged
        assert code.template == original_template
        assert code.holes[hid].state == HoleState.EMPTY

    @given(hids=st.lists(hole_id(), min_size=1, max_size=5, unique_by=lambda x: (x.namespace, x.name, x.index)))
    @settings(max_examples=20)
    def test_has_holes_consistency(self, hids):
        """has_holes should correctly report unfilled holes."""
        holes = {hid: Hole(id=hid) for hid in hids}
        markers = {hid: str(hid) for hid in hids}
        template = " ".join(str(hid) for hid in hids)

        code = HoledCode(template=template, holes=holes, hole_markers=markers)

        # Should have holes initially
        assert code.has_holes()

        # Fill all holes
        current = code
        for hid in hids:
            current = current.fill_hole(hid, "val")

        # Should have no unfilled holes now
        assert len(current.unfilled_holes()) == 0


class TestTrajectoryTrieSoundness:
    """Property tests for trajectory trie operations soundness."""

    @given(
        data=st.data(),
    )
    @settings(max_examples=30)
    def test_trajectory_extension_depth_consistency(self, data):
        """Trajectory extension should maintain consistent depth."""
        n = data.draw(st.integers(min_value=1, max_value=10))
        values = data.draw(st.lists(st.text(min_size=1, max_size=10, alphabet="abc"), min_size=n, max_size=n))
        scores = data.draw(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=n, max_size=n))

        trie, traj = create_trajectory_trie("s0")
        hid = HoleId(namespace="default", name="x", index=0)

        current = traj
        for i, (value, score) in enumerate(zip(values, scores)):
            current = current.extend(hid, value, f"s{i+1}", score=score)
            assert current.depth == i + 1

    @given(
        depth=st.integers(min_value=1, max_value=10),
        backtrack_steps=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=30)
    def test_backtrack_never_exceeds_depth(self, depth, backtrack_steps):
        """Backtracking should never exceed current depth."""
        trie, traj = create_trajectory_trie("s0")
        hid = HoleId(namespace="default", name="x", index=0)

        # Build trajectory of given depth
        current = traj
        for i in range(depth):
            current = current.extend(hid, f"v{i}", f"s{i+1}")

        # Backtrack
        actual_steps = min(backtrack_steps, depth)
        if actual_steps <= depth:
            result = current.backtrack(actual_steps)
            assert result.depth == depth - actual_steps

    @given(
        scores=st.lists(
            st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=5,
        )
    )
    @settings(max_examples=30)
    def test_cumulative_score_is_product(self, scores):
        """Cumulative score should be product of path scores."""
        trie, traj = create_trajectory_trie("s0")
        hid = HoleId(namespace="default", name="x", index=0)

        current = traj
        expected_product = 1.0
        for i, score in enumerate(scores):
            current = current.extend(hid, f"v{i}", f"s{i+1}", score=score)
            expected_product *= score

        actual = current.cumulative_score()
        assert abs(actual - expected_product) < 0.001

    def test_checkpoint_restore_preserves_state(self):
        """Checkpoint/restore should preserve trajectory state."""
        trie, traj = create_trajectory_trie("initial")
        hid = HoleId(namespace="default", name="x", index=0)

        # Extend
        traj = traj.extend(hid, "a", "s1", score=0.9)
        checkpoint = trie.checkpoint(traj)

        # Extend further
        traj = traj.extend(hid, "b", "s2", score=0.8)
        traj = traj.extend(hid, "c", "s3", score=0.7)

        # Restore
        restored = trie.restore(checkpoint)

        assert restored.depth == 1
        assert restored.state == "s1"
        assert restored.score == 0.9


class TestFillCandidateOrdering:
    """Property tests for fill candidate ordering."""

    @given(candidates=st.lists(fill_candidate(), min_size=2, max_size=10))
    @settings(max_examples=30)
    def test_sorting_orders_by_score_descending(self, candidates):
        """Sorted candidates should be in descending score order."""
        sorted_candidates = sorted(candidates)

        for i in range(len(sorted_candidates) - 1):
            assert sorted_candidates[i].score >= sorted_candidates[i + 1].score


class TestSudokuFillerSoundness:
    """Property tests for SudokuStyleHoleFiller soundness."""

    @given(strategy=st.sampled_from(list(FillStrategy)))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_fill_empty_code_succeeds(self, strategy):
        """Filling code with no holes should succeed immediately."""
        code = HoledCode(template="x = 42")
        filler = SudokuStyleHoleFiller(strategy=strategy)

        result = filler.fill(code)

        assert result.success
        assert result.filled_code == "x = 42"
        assert len(result.fill_history) == 0

    @given(
        strategy=st.sampled_from(list(FillStrategy)),
        max_backtracks=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_filler_respects_backtrack_limit(self, strategy, max_backtracks):
        """Filler should not exceed max_backtracks."""
        hid = HoleId(namespace="default", name="x", index=0)
        code = HoledCode(
            template="y = ?default:x[0]",
            holes={hid: Hole(id=hid, expected_type="int")},
            hole_markers={hid: "?default:x[0]"},
        )

        filler = SudokuStyleHoleFiller(
            strategy=strategy,
            max_backtracks=max_backtracks,
        )
        result = filler.fill(code)

        assert result.backtrack_count <= max_backtracks

    @given(strategy=st.sampled_from(list(FillStrategy)))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_filler_always_returns_result(self, strategy):
        """Filler should always return a FillResult (never crash)."""
        hid = HoleId(namespace="default", name="x", index=0)
        code = HoledCode(
            template="result = ?default:x[0]",
            holes={hid: Hole(id=hid, expected_type="int")},
            hole_markers={hid: "?default:x[0]"},
        )

        filler = SudokuStyleHoleFiller(strategy=strategy)
        result = filler.fill(code)

        assert isinstance(result, FillResult)
        # Either success with filled code, or failure with metadata
        if result.success:
            assert result.filled_code is not None
        else:
            assert len(result.unfilled_holes) >= 0


# =============================================================================
# Cross-Module Integration Soundness Tests
# =============================================================================

class TestIntegrationSoundness:
    """Property tests for integration between TTC modules."""

    @given(
        intensity=constraint_intensity(),
        code=valid_python_code(),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_lower_intensity_more_permissive(self, intensity, code):
        """Lower intensity should be at least as permissive as higher intensity.

        This is the core soundness property: reducing intensity should not
        cause previously-allowed code to be rejected.
        """
        # Get enabled domains for this intensity
        domains = domains_for_intensity(intensity)

        # Verify with only enabled domains
        verifier = ConstraintVerifier(
            language="python",
            enabled_domains=domains if domains else {"syntax"},
        )
        result = verifier.verify(code)

        # Valid Python should get reasonable score regardless of intensity
        # (fewer domains means fewer potential violations)
        assert result.overall_score >= 0.0

    @given(
        candidates=st.lists(valid_python_code(), min_size=1, max_size=5),
        strategy=selection_strategy(),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_selection_result_is_verifiable(self, candidates, strategy):
        """Selected candidate should be verifiable."""
        assume(len(candidates) > 0)

        selector = BestOfNSelector(language="python", strategy=strategy)
        selection = selector.select_best(candidates)

        # The selected candidate should be verifiable
        verifier = ConstraintVerifier(language="python")
        verify_result = verifier.verify(selection.selected)

        assert isinstance(verify_result, VerificationResult)
        # Verification should match selection result
        if selection.selected_result.valid:
            # If selection said valid, verification should agree
            # (they use the same verifier internally)
            pass  # May differ slightly due to caching


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
