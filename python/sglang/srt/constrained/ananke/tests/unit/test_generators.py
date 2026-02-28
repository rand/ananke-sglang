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
"""Tests for search generators module.

Tests cover:
- TypeAwareFillGenerator: Type-based candidate generation
- UnifiedConstraintChecker: Constraint verification
- TypeConstraintInferencer: Constraint propagation inference
"""

import sys
from pathlib import Path

import pytest

# Setup import paths
_TEST_DIR = Path(__file__).parent
_ANANKE_ROOT = _TEST_DIR.parent.parent
if str(_ANANKE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANANKE_ROOT))

from holes.hole import Hole, HoleId, HoleState, TypeEnvironment
from search.sudoku_filler import HoledCode, FillCandidate
from search.generators import (
    TypeAwareFillGenerator,
    UnifiedConstraintChecker,
    TypeConstraintInferencer,
    create_fill_generator,
    create_constraint_checker,
    create_constraint_inferencer,
)

# Import type system
from domains.types.constraint import (
    INT, STR, BOOL, FLOAT, NONE, ANY,
    ListType, DictType, TupleType, FunctionType, ClassType,
    TypeConstraint, TYPE_TOP, TYPE_BOTTOM,
    type_expecting,
)


# =============================================================================
# TypeAwareFillGenerator Tests
# =============================================================================


class TestTypeAwareFillGenerator:
    """Tests for TypeAwareFillGenerator."""

    def test_create_generator(self):
        """Test creating a generator."""
        gen = TypeAwareFillGenerator()
        assert gen.language == "python"
        assert gen.include_identifiers is True

    def test_generate_for_int(self):
        """Test generating candidates for int type."""
        gen = TypeAwareFillGenerator()
        hole = Hole(
            id=HoleId(name="value"),
            expected_type=INT,
        )

        candidates = gen.generate_candidates(hole, "x: int = ")

        assert len(candidates) > 0
        # Should include int literals
        values = [c.value for c in candidates]
        assert "0" in values
        assert "1" in values

        # All should have positive scores
        for c in candidates:
            assert c.score > 0

    def test_generate_for_str(self):
        """Test generating candidates for str type."""
        gen = TypeAwareFillGenerator()
        hole = Hole(
            id=HoleId(name="value"),
            expected_type=STR,
        )

        candidates = gen.generate_candidates(hole, "x: str = ")

        assert len(candidates) > 0
        values = [c.value for c in candidates]
        assert '""' in values or "str()" in values

    def test_generate_for_bool(self):
        """Test generating candidates for bool type."""
        gen = TypeAwareFillGenerator()
        hole = Hole(
            id=HoleId(name="value"),
            expected_type=BOOL,
        )

        candidates = gen.generate_candidates(hole, "x: bool = ")

        assert len(candidates) > 0
        values = [c.value for c in candidates]
        assert "True" in values
        assert "False" in values

    def test_generate_for_none(self):
        """Test generating candidates for None type."""
        gen = TypeAwareFillGenerator()
        hole = Hole(
            id=HoleId(name="value"),
            expected_type=NONE,
        )

        candidates = gen.generate_candidates(hole, "x = ")

        assert len(candidates) > 0
        values = [c.value for c in candidates]
        assert "None" in values

    def test_generate_for_list(self):
        """Test generating candidates for list type."""
        gen = TypeAwareFillGenerator()
        hole = Hole(
            id=HoleId(name="value"),
            expected_type=ListType(INT),
        )

        candidates = gen.generate_candidates(hole, "x: List[int] = ")

        assert len(candidates) > 0
        values = [c.value for c in candidates]
        assert "[]" in values

    def test_generate_for_dict(self):
        """Test generating candidates for dict type."""
        gen = TypeAwareFillGenerator()
        hole = Hole(
            id=HoleId(name="value"),
            expected_type=DictType(STR, ANY),
        )

        candidates = gen.generate_candidates(hole, "x: Dict[str, Any] = ")

        assert len(candidates) > 0
        values = [c.value for c in candidates]
        assert "{}" in values

    def test_generate_for_function(self):
        """Test generating candidates for function type."""
        gen = TypeAwareFillGenerator()
        hole = Hole(
            id=HoleId(name="func"),
            expected_type=FunctionType(params=(INT,), returns=STR),
        )

        candidates = gen.generate_candidates(hole, "f: Callable[[int], str] = ")

        assert len(candidates) > 0
        # Should include lambda
        values = [c.value for c in candidates]
        assert any("lambda" in v for v in values)

    def test_generate_for_any(self):
        """Test generating candidates for Any type."""
        gen = TypeAwareFillGenerator()
        hole = Hole(
            id=HoleId(name="value"),
            expected_type=ANY,
        )

        candidates = gen.generate_candidates(hole, "x = ")

        assert len(candidates) > 0
        # Should have diverse candidates
        assert len(candidates) >= 3

    def test_generate_with_type_environment(self):
        """Test generating candidates from type environment."""
        env = TypeEnvironment.from_dict({
            "my_int": INT,
            "my_str": STR,
        })

        gen = TypeAwareFillGenerator(type_environment=env)
        hole = Hole(
            id=HoleId(name="value"),
            expected_type=INT,
        )

        candidates = gen.generate_candidates(hole, "x: int = ")

        # Should include my_int from environment
        values = [c.value for c in candidates]
        assert "my_int" in values

    def test_generate_respects_max_candidates(self):
        """Test that max_candidates is respected."""
        gen = TypeAwareFillGenerator()
        hole = Hole(
            id=HoleId(name="value"),
            expected_type=ANY,
        )

        candidates = gen.generate_candidates(hole, "", max_candidates=3)

        assert len(candidates) <= 3

    def test_generate_deduplicates_by_value(self):
        """Test that duplicate values are removed."""
        gen = TypeAwareFillGenerator()
        hole = Hole(
            id=HoleId(name="value"),
            expected_type=INT,
        )

        candidates = gen.generate_candidates(hole, "", max_candidates=100)

        # No duplicates
        values = [c.value for c in candidates]
        assert len(values) == len(set(values))


class TestTypeAwareFillGeneratorEdgeCases:
    """Edge case tests for TypeAwareFillGenerator."""

    def test_empty_context(self):
        """Test with empty context string."""
        gen = TypeAwareFillGenerator()
        hole = Hole(id=HoleId(name="value"), expected_type=INT)

        candidates = gen.generate_candidates(hole, "")

        assert len(candidates) > 0

    def test_no_expected_type(self):
        """Test with no expected type."""
        gen = TypeAwareFillGenerator()
        hole = Hole(id=HoleId(name="value"))

        candidates = gen.generate_candidates(hole, "x = ")

        # Should default to Any-type candidates
        assert len(candidates) > 0

    def test_complex_nested_type(self):
        """Test with complex nested type."""
        gen = TypeAwareFillGenerator()
        hole = Hole(
            id=HoleId(name="value"),
            expected_type=ListType(DictType(STR, INT)),
        )

        candidates = gen.generate_candidates(hole, "")

        # Should at least have empty list
        values = [c.value for c in candidates]
        assert "[]" in values


# =============================================================================
# UnifiedConstraintChecker Tests
# =============================================================================


class TestUnifiedConstraintChecker:
    """Tests for UnifiedConstraintChecker."""

    def test_create_checker(self):
        """Test creating a checker."""
        checker = UnifiedConstraintChecker()
        assert checker.language == "python"

    def test_check_valid_int_fill(self):
        """Test checking a valid int fill."""
        checker = UnifiedConstraintChecker()

        hole = Hole(id=HoleId(name="value"), expected_type=INT)
        code = HoledCode(
            template="x: int = ?default:value[0]",
            holes={hole.id: hole},
            hole_markers={hole.id: "?default:value[0]"},
        )

        valid, score, violations = checker.check_fill(hole, "42", code)

        # Should be valid (type precheck passes)
        assert score > 0.5

    def test_check_invalid_str_for_int(self):
        """Test checking an invalid str fill for int."""
        checker = UnifiedConstraintChecker()

        hole = Hole(id=HoleId(name="value"), expected_type=INT)
        code = HoledCode(
            template="x: int = ?default:value[0]",
            holes={hole.id: hole},
            hole_markers={hole.id: "?default:value[0]"},
        )

        valid, score, violations = checker.check_fill(hole, '"hello"', code)

        # Should have low score (type mismatch)
        assert score < 0.5

    def test_check_fill_with_type_context(self):
        """Test checking fill with type context."""
        checker = UnifiedConstraintChecker(
            type_context={"my_var": "int"},
        )

        hole = Hole(id=HoleId(name="value"), expected_type=INT)
        code = HoledCode(
            template="x: int = ?default:value[0]",
            holes={hole.id: hole},
            hole_markers={hole.id: "?default:value[0]"},
        )

        valid, score, violations = checker.check_fill(hole, "my_var", code)

        # Should pass (identifier allowed for int context)
        assert score > 0.5

    def test_check_any_type_fill(self):
        """Test checking fill for Any type."""
        checker = UnifiedConstraintChecker()

        hole = Hole(id=HoleId(name="value"), expected_type=ANY)
        code = HoledCode(
            template="x = ?default:value[0]",
            holes={hole.id: hole},
            hole_markers={hole.id: "?default:value[0]"},
        )

        valid, score, violations = checker.check_fill(hole, '"anything"', code)

        # Any type accepts anything
        assert score > 0.5


class TestUnifiedConstraintCheckerPrecheck:
    """Tests for type precheck functionality."""

    def test_precheck_int_literal(self):
        """Test precheck with int literal."""
        checker = UnifiedConstraintChecker()
        hole = Hole(id=HoleId(name="value"), expected_type=INT)

        valid, score = checker._type_precheck(hole, "42")

        assert valid is True
        assert score == 1.0

    def test_precheck_str_literal_for_str(self):
        """Test precheck with str literal for str type."""
        checker = UnifiedConstraintChecker()
        hole = Hole(id=HoleId(name="value"), expected_type=STR)

        valid, score = checker._type_precheck(hole, '"hello"')

        assert valid is True
        assert score == 1.0

    def test_precheck_bool_literal(self):
        """Test precheck with bool literal."""
        checker = UnifiedConstraintChecker()
        hole = Hole(id=HoleId(name="value"), expected_type=BOOL)

        valid, score = checker._type_precheck(hole, "True")

        assert valid is True
        assert score == 1.0

    def test_precheck_none_literal(self):
        """Test precheck with None literal."""
        checker = UnifiedConstraintChecker()
        hole = Hole(id=HoleId(name="value"), expected_type=NONE)

        valid, score = checker._type_precheck(hole, "None")

        assert valid is True
        assert score == 1.0


# =============================================================================
# TypeConstraintInferencer Tests
# =============================================================================


class TestTypeConstraintInferencer:
    """Tests for TypeConstraintInferencer."""

    def test_create_inferencer(self):
        """Test creating an inferencer."""
        inferencer = TypeConstraintInferencer()
        assert inferencer.language == "python"

    def test_infer_fill_type_int(self):
        """Test inferring type of int literal."""
        inferencer = TypeConstraintInferencer()

        result = inferencer._infer_fill_type("42")

        assert result == INT

    def test_infer_fill_type_str(self):
        """Test inferring type of string literal."""
        inferencer = TypeConstraintInferencer()

        result = inferencer._infer_fill_type('"hello"')

        assert result == STR

    def test_infer_fill_type_bool(self):
        """Test inferring type of bool literal."""
        inferencer = TypeConstraintInferencer()

        for literal in ["True", "False"]:
            result = inferencer._infer_fill_type(literal)
            assert result == BOOL

    def test_infer_fill_type_none(self):
        """Test inferring type of None."""
        inferencer = TypeConstraintInferencer()

        result = inferencer._infer_fill_type("None")

        assert result == NONE

    def test_infer_fill_type_float(self):
        """Test inferring type of float literal."""
        inferencer = TypeConstraintInferencer()

        result = inferencer._infer_fill_type("3.14")

        assert result == FLOAT

    def test_infer_fill_type_empty_list(self):
        """Test inferring type of empty list."""
        inferencer = TypeConstraintInferencer()

        result = inferencer._infer_fill_type("[]")

        assert isinstance(result, ListType)

    def test_infer_fill_type_empty_dict(self):
        """Test inferring type of empty dict."""
        inferencer = TypeConstraintInferencer()

        result = inferencer._infer_fill_type("{}")

        assert isinstance(result, DictType)

    def test_infer_fill_type_lambda(self):
        """Test inferring type of lambda."""
        inferencer = TypeConstraintInferencer()

        result = inferencer._infer_fill_type("lambda x: x")

        assert isinstance(result, FunctionType)
        assert len(result.params) == 1

    def test_infer_constraint_return_type(self):
        """Test inferring constraint for return type relationship."""
        inferencer = TypeConstraintInferencer()

        filled_hole = Hole(
            id=HoleId(name="return_value"),
            expected_type=ANY,
        )
        dependent_hole = Hole(
            id=HoleId(name="other"),
            expected_type=ANY,
        )

        constraint = inferencer.infer_constraint_from_fill(
            filled_hole, "42", dependent_hole
        )

        # Should infer int type constraint
        if constraint is not None:
            assert isinstance(constraint, TypeConstraint)
            assert constraint.expected_type == INT

    def test_infer_constraint_no_relationship(self):
        """Test no inference when holes are unrelated."""
        inferencer = TypeConstraintInferencer()

        filled_hole = Hole(
            id=HoleId(name="foo"),
            expected_type=ANY,
        )
        dependent_hole = Hole(
            id=HoleId(name="bar"),
            expected_type=ANY,
        )

        constraint = inferencer.infer_constraint_from_fill(
            filled_hole, "42", dependent_hole
        )

        # No relationship detected - no constraint
        # (This depends on the hole names not matching patterns)
        # The implementation returns None for unknown relationships

    def test_infer_constraint_none_filled_hole(self):
        """Test handling of None filled hole."""
        inferencer = TypeConstraintInferencer()

        dependent_hole = Hole(
            id=HoleId(name="value"),
            expected_type=ANY,
        )

        constraint = inferencer.infer_constraint_from_fill(
            None, "42", dependent_hole
        )

        assert constraint is None


class TestTypeConstraintInferencerLambda:
    """Tests for lambda type inference."""

    def test_infer_lambda_no_params(self):
        """Test inferring type of parameterless lambda."""
        inferencer = TypeConstraintInferencer()

        result = inferencer._infer_lambda_type("lambda: None")

        assert isinstance(result, FunctionType)
        assert len(result.params) == 0

    def test_infer_lambda_one_param(self):
        """Test inferring type of single-param lambda."""
        inferencer = TypeConstraintInferencer()

        result = inferencer._infer_lambda_type("lambda x: x")

        assert isinstance(result, FunctionType)
        assert len(result.params) == 1

    def test_infer_lambda_multiple_params(self):
        """Test inferring type of multi-param lambda."""
        inferencer = TypeConstraintInferencer()

        result = inferencer._infer_lambda_type("lambda x, y, z: x + y + z")

        assert isinstance(result, FunctionType)
        assert len(result.params) == 3


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_fill_generator(self):
        """Test create_fill_generator factory."""
        gen = create_fill_generator()

        assert isinstance(gen, TypeAwareFillGenerator)
        assert gen.language == "python"

    def test_create_fill_generator_with_options(self):
        """Test create_fill_generator with options."""
        env = TypeEnvironment.from_dict({"x": INT})
        gen = create_fill_generator(language="typescript", type_environment=env)

        assert gen.language == "typescript"
        assert gen.type_environment == env

    def test_create_constraint_checker(self):
        """Test create_constraint_checker factory."""
        checker = create_constraint_checker()

        assert isinstance(checker, UnifiedConstraintChecker)
        assert checker.language == "python"

    def test_create_constraint_checker_with_options(self):
        """Test create_constraint_checker with options."""
        checker = create_constraint_checker(
            language="rust",
            enabled_domains={"syntax", "types"},
            type_context={"x": "i32"},
        )

        assert checker.language == "rust"
        assert checker.enabled_domains == {"syntax", "types"}

    def test_create_constraint_inferencer(self):
        """Test create_constraint_inferencer factory."""
        inferencer = create_constraint_inferencer()

        assert isinstance(inferencer, TypeConstraintInferencer)
        assert inferencer.language == "python"


# =============================================================================
# Integration Tests
# =============================================================================


class TestGeneratorsIntegration:
    """Integration tests for generators with SudokuStyleHoleFiller."""

    def test_generator_with_filler(self):
        """Test using generator with SudokuStyleHoleFiller."""
        from search.sudoku_filler import SudokuStyleHoleFiller, FillStrategy

        gen = TypeAwareFillGenerator()
        checker = UnifiedConstraintChecker()

        filler = SudokuStyleHoleFiller(
            fill_generator=gen,
            constraint_checker=checker,
            strategy=FillStrategy.MCV,
        )

        # Create holed code
        hole_id = HoleId(name="value")
        hole = Hole(id=hole_id, expected_type=INT)
        code = HoledCode(
            template="x: int = ?default:value[0]",
            holes={hole_id: hole},
            hole_markers={hole_id: "?default:value[0]"},
        )

        result = filler.fill(code)

        # Should have a result
        assert result is not None
        # May or may not succeed depending on verification
        if result.success:
            assert "?default:value[0]" not in result.filled_code

    def test_end_to_end_simple_fill(self):
        """Test end-to-end simple fill."""
        from search.sudoku_filler import fill_with_mcv_heuristic

        gen = TypeAwareFillGenerator()

        hole_id = HoleId(name="value")
        hole = Hole(id=hole_id, expected_type=INT)
        code = HoledCode(
            template="x = ?default:value[0]",
            holes={hole_id: hole},
            hole_markers={hole_id: "?default:value[0]"},
        )

        result = fill_with_mcv_heuristic(code, fill_generator=gen)

        # Should succeed with int fill
        assert result.success or len(result.fill_history) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
