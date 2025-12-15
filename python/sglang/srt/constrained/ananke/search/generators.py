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
"""Concrete implementations of FillGenerator and ConstraintChecker protocols.

This module provides production-ready implementations that integrate with
Ananke's type system and verification infrastructure for sudoku-style hole filling.

Key Components:
1. TypeAwareFillGenerator: Generates type-compatible fill candidates
2. UnifiedConstraintChecker: Verifies fills against all constraint domains
3. TypeConstraintInferencer: Infers constraints from fills for propagation

References:
- GenCP: LLM Meets Constraint Propagation (arXiv 2024)
- Hazel: Typed Holes for Code Generation (OOPSLA 2024)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, TYPE_CHECKING

# Handle imports for both package and standalone usage
try:
    from ..holes.hole import Hole, HoleId, HoleState, TypeEnvironment
    from ..verification.verifier import ConstraintVerifier, VerificationResult
except ImportError:
    _SEARCH_DIR = Path(__file__).parent
    _ANANKE_ROOT = _SEARCH_DIR.parent
    if str(_ANANKE_ROOT) not in sys.path:
        sys.path.insert(0, str(_ANANKE_ROOT))
    from holes.hole import Hole, HoleId, HoleState, TypeEnvironment
    from verification.verifier import ConstraintVerifier, VerificationResult

# Import type-related components
try:
    from ..domains.types.constraint import (
        Type,
        TypeVar,
        TypeConstraint,
        TYPE_TOP,
        TYPE_BOTTOM,
        ANY,
        NEVER,
        INT,
        STR,
        BOOL,
        FLOAT,
        NONE,
        AnyType,
        NeverType,
        HoleType,
        FunctionType,
        ListType,
        DictType,
        TupleType,
        ClassType,
        UnionType,
        PrimitiveType,
        type_expecting,
    )
except ImportError:
    from domains.types.constraint import (
        Type,
        TypeVar,
        TypeConstraint,
        TYPE_TOP,
        TYPE_BOTTOM,
        ANY,
        NEVER,
        INT,
        STR,
        BOOL,
        FLOAT,
        NONE,
        AnyType,
        NeverType,
        HoleType,
        FunctionType,
        ListType,
        DictType,
        TupleType,
        ClassType,
        UnionType,
        PrimitiveType,
        type_expecting,
    )

# Import from sudoku_filler (local import)
from .sudoku_filler import FillCandidate, HoledCode

logger = logging.getLogger(__name__)


# =============================================================================
# Type-Aware Fill Generator
# =============================================================================


@dataclass
class TypeAwareFillGenerator:
    """Generates fill candidates based on type constraints.

    Uses the expected type from a hole's constraint to generate
    type-compatible fill candidates. This is a deterministic generator
    that produces candidates based on type analysis, not LLM inference.

    For LLM-based generation, wrap this with an LLM generator that uses
    the type information to guide prompts.

    Attributes:
        language: Target programming language
        type_environment: Type bindings for identifier lookups
        include_identifiers: Whether to include environment identifiers
        max_literal_candidates: Maximum literal candidates per type
    """

    language: str = "python"
    type_environment: TypeEnvironment = field(default_factory=TypeEnvironment.empty)
    include_identifiers: bool = True
    max_literal_candidates: int = 5

    def generate_candidates(
        self,
        hole: Hole[Any],
        context: str,
        max_candidates: int = 10,
    ) -> List[FillCandidate]:
        """Generate type-aware fill candidates for a hole.

        Uses the hole's expected type and constraint to generate candidates
        that are likely to be type-correct.

        Args:
            hole: The hole to generate fills for
            context: Surrounding code context (for identifier extraction)
            max_candidates: Maximum number of candidates to return

        Returns:
            List of FillCandidate ordered by type compatibility score
        """
        candidates: List[FillCandidate] = []

        # Get expected type from hole
        expected_type = hole.expected_type
        if expected_type is None:
            # No type constraint - use Any
            expected_type = ANY

        # Generate type-specific candidates
        type_candidates = self._generate_for_type(expected_type)
        candidates.extend(type_candidates)

        # Add identifiers from environment that match the type
        if self.include_identifiers:
            env_candidates = self._generate_from_environment(expected_type, hole)
            candidates.extend(env_candidates)

        # Add identifiers from hole's captured environment
        if hole.environment:
            captured_candidates = self._generate_from_captured_env(expected_type, hole)
            candidates.extend(captured_candidates)

        # Deduplicate by value, keeping highest score
        seen: Dict[str, FillCandidate] = {}
        for c in candidates:
            if c.value not in seen or c.score > seen[c.value].score:
                seen[c.value] = c

        # Sort by score and limit
        result = sorted(seen.values(), key=lambda x: -x.score)[:max_candidates]
        return result

    def _generate_for_type(self, expected: Type) -> List[FillCandidate]:
        """Generate candidates for a specific type."""
        candidates: List[FillCandidate] = []

        # Handle Any type - generate diverse candidates
        if isinstance(expected, AnyType):
            return self._generate_any_candidates()

        # Handle Never type - no valid fills
        if isinstance(expected, NeverType):
            return []

        # Handle Hole type - use expected type inside hole
        if isinstance(expected, HoleType):
            if expected.expected:
                return self._generate_for_type(expected.expected)
            return self._generate_any_candidates()

        # Handle primitive types
        if isinstance(expected, PrimitiveType):
            return self._generate_primitive_candidates(expected)

        # Handle function types
        if isinstance(expected, FunctionType):
            return self._generate_function_candidates(expected)

        # Handle list types
        if isinstance(expected, ListType):
            return self._generate_list_candidates(expected)

        # Handle dict types
        if isinstance(expected, DictType):
            return self._generate_dict_candidates(expected)

        # Handle tuple types
        if isinstance(expected, TupleType):
            return self._generate_tuple_candidates(expected)

        # Handle union types
        if isinstance(expected, UnionType):
            return self._generate_union_candidates(expected)

        # Handle class types
        if isinstance(expected, ClassType):
            return self._generate_class_candidates(expected)

        # Handle type variables - conservative
        if isinstance(expected, TypeVar):
            return self._generate_any_candidates()

        # Fallback
        return [FillCandidate("...", score=0.3, metadata={"fallback": True})]

    def _generate_any_candidates(self) -> List[FillCandidate]:
        """Generate candidates for Any type."""
        return [
            FillCandidate("None", score=0.8),
            FillCandidate("0", score=0.7),
            FillCandidate('""', score=0.7),
            FillCandidate("[]", score=0.6),
            FillCandidate("{}", score=0.6),
            FillCandidate("True", score=0.5),
            FillCandidate("...", score=0.4),
        ]

    def _generate_primitive_candidates(self, ptype: PrimitiveType) -> List[FillCandidate]:
        """Generate candidates for primitive types."""
        name = ptype.name.lower()

        if name == "int":
            return [
                FillCandidate("0", score=0.95, metadata={"type": "int"}),
                FillCandidate("1", score=0.9, metadata={"type": "int"}),
                FillCandidate("-1", score=0.85, metadata={"type": "int"}),
                FillCandidate("42", score=0.7, metadata={"type": "int"}),
                FillCandidate("len([])", score=0.6, metadata={"type": "int", "expr": True}),
            ]

        if name == "str":
            return [
                FillCandidate('""', score=0.95, metadata={"type": "str"}),
                FillCandidate('"placeholder"', score=0.7, metadata={"type": "str"}),
                FillCandidate("str()", score=0.6, metadata={"type": "str", "expr": True}),
                FillCandidate('f""', score=0.5, metadata={"type": "str", "fstring": True}),
            ]

        if name == "bool":
            return [
                FillCandidate("True", score=0.95, metadata={"type": "bool"}),
                FillCandidate("False", score=0.95, metadata={"type": "bool"}),
                FillCandidate("bool()", score=0.6, metadata={"type": "bool", "expr": True}),
            ]

        if name == "float":
            return [
                FillCandidate("0.0", score=0.95, metadata={"type": "float"}),
                FillCandidate("1.0", score=0.9, metadata={"type": "float"}),
                FillCandidate("-1.0", score=0.85, metadata={"type": "float"}),
                FillCandidate("float('inf')", score=0.5, metadata={"type": "float"}),
            ]

        if name == "none":
            return [
                FillCandidate("None", score=1.0, metadata={"type": "None"}),
            ]

        if name == "bytes":
            return [
                FillCandidate('b""', score=0.95, metadata={"type": "bytes"}),
                FillCandidate("bytes()", score=0.8, metadata={"type": "bytes"}),
            ]

        # Unknown primitive
        return [FillCandidate(f"{name}()", score=0.5)]

    def _generate_function_candidates(self, ftype: FunctionType) -> List[FillCandidate]:
        """Generate candidates for function types."""
        # Generate lambda with appropriate param count
        param_count = len(ftype.params)
        if param_count == 0:
            return [
                FillCandidate("lambda: None", score=0.9),
                FillCandidate("lambda: ...", score=0.7),
            ]
        elif param_count == 1:
            return [
                FillCandidate("lambda x: x", score=0.9),
                FillCandidate("lambda x: None", score=0.8),
                FillCandidate("lambda _: None", score=0.7),
            ]
        elif param_count == 2:
            return [
                FillCandidate("lambda x, y: x", score=0.85),
                FillCandidate("lambda x, y: None", score=0.8),
            ]
        else:
            params = ", ".join(f"x{i}" for i in range(param_count))
            return [
                FillCandidate(f"lambda {params}: None", score=0.8),
            ]

    def _generate_list_candidates(self, ltype: ListType) -> List[FillCandidate]:
        """Generate candidates for list types."""
        candidates = [
            FillCandidate("[]", score=0.95, metadata={"type": "list", "empty": True}),
        ]

        # Add typed list literals based on element type
        elem_type = ltype.element
        if isinstance(elem_type, PrimitiveType):
            elem_name = elem_type.name.lower()
            if elem_name == "int":
                candidates.append(FillCandidate("[0]", score=0.85, metadata={"type": "list"}))
                candidates.append(FillCandidate("[1, 2, 3]", score=0.7, metadata={"type": "list"}))
            elif elem_name == "str":
                candidates.append(FillCandidate('[""]', score=0.85, metadata={"type": "list"}))
            elif elem_name == "bool":
                candidates.append(FillCandidate("[True]", score=0.85, metadata={"type": "list"}))

        candidates.append(FillCandidate("list()", score=0.7, metadata={"type": "list"}))
        return candidates

    def _generate_dict_candidates(self, dtype: DictType) -> List[FillCandidate]:
        """Generate candidates for dict types."""
        candidates = [
            FillCandidate("{}", score=0.95, metadata={"type": "dict", "empty": True}),
        ]

        # Add typed dict literals based on key/value types
        key_type = dtype.key
        if isinstance(key_type, PrimitiveType) and key_type.name.lower() == "str":
            candidates.append(FillCandidate('{"key": None}', score=0.7, metadata={"type": "dict"}))

        candidates.append(FillCandidate("dict()", score=0.7, metadata={"type": "dict"}))
        return candidates

    def _generate_tuple_candidates(self, ttype: TupleType) -> List[FillCandidate]:
        """Generate candidates for tuple types."""
        if not ttype.elements:
            return [FillCandidate("()", score=0.95, metadata={"type": "tuple"})]

        # Generate tuple with appropriate number of elements
        elem_count = len(ttype.elements)
        if elem_count == 1:
            return [
                FillCandidate("(None,)", score=0.9, metadata={"type": "tuple"}),
            ]
        else:
            elems = ", ".join(["None"] * elem_count)
            return [
                FillCandidate(f"({elems})", score=0.9, metadata={"type": "tuple"}),
            ]

    def _generate_union_candidates(self, utype: UnionType) -> List[FillCandidate]:
        """Generate candidates for union types."""
        # Generate candidates for each member type
        all_candidates: List[FillCandidate] = []
        for member in utype.members:
            member_candidates = self._generate_for_type(member)
            # Slightly lower scores for union members
            for c in member_candidates:
                c.score *= 0.95
            all_candidates.extend(member_candidates)

        return all_candidates

    def _generate_class_candidates(self, ctype: ClassType) -> List[FillCandidate]:
        """Generate candidates for class types."""
        class_name = ctype.name

        # Handle common types
        common_types = {
            "object": [FillCandidate("object()", score=0.9)],
            "Exception": [
                FillCandidate('Exception("")', score=0.9),
                FillCandidate("Exception()", score=0.85),
            ],
            "ValueError": [FillCandidate('ValueError("")', score=0.9)],
            "TypeError": [FillCandidate('TypeError("")', score=0.9)],
            "Path": [
                FillCandidate('Path(".")', score=0.9),
                FillCandidate("Path()", score=0.8),
            ],
        }

        if class_name in common_types:
            return common_types[class_name]

        # Default: constructor call
        return [
            FillCandidate(f"{class_name}()", score=0.8, metadata={"type": "class"}),
        ]

    def _generate_from_environment(
        self,
        expected: Type,
        hole: Hole[Any],
    ) -> List[FillCandidate]:
        """Generate candidates from type environment."""
        candidates: List[FillCandidate] = []

        # Get all bindings from the type environment
        all_bindings = self.type_environment.all_bindings()

        for name, var_type in all_bindings.items():
            # Check if the variable type is compatible with expected type
            if self._types_compatible(var_type, expected):
                score = self._compute_type_match_score(var_type, expected)
                candidates.append(
                    FillCandidate(
                        name,
                        score=score,
                        metadata={"source": "environment", "type": str(var_type)},
                    )
                )

        return candidates

    def _generate_from_captured_env(
        self,
        expected: Type,
        hole: Hole[Any],
    ) -> List[FillCandidate]:
        """Generate candidates from hole's captured environment."""
        candidates: List[FillCandidate] = []

        if not hole.environment:
            return candidates

        all_bindings = hole.environment.all_bindings()

        for name, var_type in all_bindings.items():
            if self._types_compatible(var_type, expected):
                score = self._compute_type_match_score(var_type, expected)
                candidates.append(
                    FillCandidate(
                        name,
                        score=score * 1.1,  # Prefer captured environment
                        metadata={"source": "captured_env", "type": str(var_type)},
                    )
                )

        return candidates

    def _types_compatible(self, actual: Type, expected: Type) -> bool:
        """Check if actual type is compatible with expected type."""
        # Any is compatible with everything
        if isinstance(expected, AnyType) or isinstance(actual, AnyType):
            return True

        # Holes are compatible with anything
        if isinstance(expected, HoleType) or isinstance(actual, HoleType):
            return True

        # Same type
        if actual == expected:
            return True

        # int is compatible with float (numeric promotion)
        if actual == INT and expected == FLOAT:
            return True

        # Union type compatibility
        if isinstance(expected, UnionType):
            return any(self._types_compatible(actual, m) for m in expected.members)

        # TypeVar - assume compatible (conservative)
        if isinstance(expected, TypeVar) or isinstance(actual, TypeVar):
            return True

        return False

    def _compute_type_match_score(self, actual: Type, expected: Type) -> float:
        """Compute a score for how well actual matches expected."""
        # Exact match
        if actual == expected:
            return 1.0

        # Any matches
        if isinstance(expected, AnyType) or isinstance(actual, AnyType):
            return 0.8

        # Numeric promotion
        if actual == INT and expected == FLOAT:
            return 0.9

        # Union member
        if isinstance(expected, UnionType):
            if any(actual == m for m in expected.members):
                return 0.95

        # Compatible but not exact
        return 0.7


# =============================================================================
# Unified Constraint Checker
# =============================================================================


@dataclass
class UnifiedConstraintChecker:
    """Checks fill candidates against all Ananke constraint domains.

    Uses the ConstraintVerifier to perform comprehensive checking of
    fills against syntax, types, imports, control flow, and semantics.

    Attributes:
        language: Target programming language
        enabled_domains: Set of domains to check (None = all)
        type_context: Type bindings for the type domain
        import_context: Available imports
        verifier: Internal ConstraintVerifier instance
    """

    language: str = "python"
    enabled_domains: Optional[Set[str]] = None
    type_context: Optional[Dict[str, str]] = None
    import_context: Optional[List[str]] = None
    _verifier: Optional[ConstraintVerifier] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize the internal verifier."""
        if self._verifier is None:
            self._verifier = ConstraintVerifier(
                language=self.language,
                enabled_domains=self.enabled_domains,
                type_context=self.type_context or {},
                import_context=self.import_context or [],
            )

    def check_fill(
        self,
        hole: Hole[Any],
        fill: str,
        context: HoledCode,
    ) -> Tuple[bool, float, Tuple[str, ...]]:
        """Check if a fill satisfies constraints.

        Creates a version of the code with the hole filled and verifies
        it against all enabled constraint domains.

        Args:
            hole: The hole being filled
            fill: The fill value
            context: Current state of holed code

        Returns:
            (valid, score, violations) tuple:
            - valid: True if all domains report validity
            - score: Weighted score from 0.0 to 1.0
            - violations: Tuple of violation messages
        """
        # Create the filled code
        try:
            filled_code = context.fill_hole(hole.id, fill)
            code_str = filled_code.to_string()
        except Exception as e:
            return False, 0.0, (f"Fill failed: {str(e)}",)

        # Type-level pre-check (fast path)
        type_valid, type_score = self._type_precheck(hole, fill)
        if not type_valid and type_score < 0.3:
            return False, type_score, ("Type mismatch",)

        # Full verification
        result = self._verifier.verify(code_str)

        # Collect violations
        violations = tuple(result.get_errors())

        return result.valid, result.overall_score, violations

    def _type_precheck(self, hole: Hole[Any], fill: str) -> Tuple[bool, float]:
        """Fast type-level pre-check before full verification.

        Performs a quick check if the fill is obviously type-incompatible.
        This avoids expensive full verification for clearly invalid fills.

        Args:
            hole: The hole being filled
            fill: The fill value

        Returns:
            (valid, score) tuple
        """
        expected_type = hole.expected_type
        if expected_type is None:
            return True, 1.0

        # Handle Any type
        if isinstance(expected_type, AnyType):
            return True, 1.0

        # Quick literal type check
        fill_stripped = fill.strip()

        # Check numeric literals
        if expected_type == INT:
            try:
                int(fill_stripped)
                return True, 1.0
            except ValueError:
                if fill_stripped.isidentifier():
                    return True, 0.8  # Identifier - might be int
                if fill_stripped.startswith('"') or fill_stripped.startswith("'"):
                    return False, 0.1  # String literal
                return True, 0.5  # Expression

        if expected_type == STR:
            if fill_stripped.startswith('"') or fill_stripped.startswith("'"):
                return True, 1.0
            if fill_stripped.startswith('f"') or fill_stripped.startswith("f'"):
                return True, 1.0
            try:
                int(fill_stripped)
                return False, 0.2  # Bare int literal
            except ValueError:
                pass
            return True, 0.7  # Might be string expression

        if expected_type == BOOL:
            if fill_stripped in ("True", "False"):
                return True, 1.0
            return True, 0.7  # Might be bool expression

        if expected_type == FLOAT:
            try:
                float(fill_stripped)
                return True, 1.0
            except ValueError:
                pass
            return True, 0.7

        if expected_type == NONE:
            if fill_stripped == "None":
                return True, 1.0
            return True, 0.5

        # For complex types, assume OK at precheck level
        return True, 0.8


# =============================================================================
# Type Constraint Inferencer
# =============================================================================


@dataclass
class TypeConstraintInferencer:
    """Infers type constraints from fills for constraint propagation.

    Implements the constraint inference logic for sudoku-style hole filling.
    When a hole is filled, this class determines how that fill affects
    the constraints on dependent holes.

    Key Insight (from GenCP):
        Constraint propagation is bidirectional - filling a hole can:
        1. Constrain dependent holes (forward propagation)
        2. Validate the fill against hole constraints (backward checking)

    Attributes:
        language: Target programming language
    """

    language: str = "python"

    def infer_constraint_from_fill(
        self,
        filled_hole: Optional[Hole[Any]],
        fill_value: str,
        dependent_hole: Hole[Any],
    ) -> Optional[TypeConstraint]:
        """Infer constraint for dependent hole from a fill.

        Analyzes the relationship between the filled hole and dependent hole
        to determine how the fill constrains the dependent hole's valid values.

        Args:
            filled_hole: The hole that was filled
            fill_value: The fill value
            dependent_hole: The dependent hole to constrain

        Returns:
            TypeConstraint to apply to dependent_hole, or None if no constraint inferred
        """
        if filled_hole is None:
            return None

        # Get the type of the filled value
        fill_type = self._infer_fill_type(fill_value)
        if fill_type is None:
            return None

        # Determine relationship between holes
        relationship = self._determine_relationship(filled_hole, dependent_hole)

        # Infer constraint based on relationship
        if relationship == "return_type":
            # Filled hole is in return position - dependent hole must match return type
            return self._infer_return_type_constraint(fill_type, dependent_hole)

        elif relationship == "argument":
            # Filled hole is an argument - dependent hole might be constrained by param type
            return self._infer_argument_constraint(filled_hole, fill_type, dependent_hole)

        elif relationship == "assignment":
            # Filled hole is assignment target - dependent hole is the value
            return self._infer_assignment_constraint(fill_type, dependent_hole)

        elif relationship == "element":
            # Filled hole is container element - dependent hole must match element type
            return self._infer_element_constraint(fill_type, dependent_hole)

        elif relationship == "key" or relationship == "value":
            # Filled hole is dict key/value
            return self._infer_dict_constraint(fill_type, relationship, dependent_hole)

        # No constraint inference possible
        return None

    def _infer_fill_type(self, fill_value: str) -> Optional[Type]:
        """Infer the type of a fill value."""
        fill_stripped = fill_value.strip()

        if not fill_stripped:
            return None

        # None literal
        if fill_stripped == "None":
            return NONE

        # Boolean literals
        if fill_stripped in ("True", "False"):
            return BOOL

        # Integer literals
        try:
            int(fill_stripped)
            return INT
        except ValueError:
            pass

        # Float literals
        try:
            float(fill_stripped)
            return FLOAT
        except ValueError:
            pass

        # String literals
        if (fill_stripped.startswith('"') and fill_stripped.endswith('"')) or \
           (fill_stripped.startswith("'") and fill_stripped.endswith("'")):
            return STR

        # F-string
        if fill_stripped.startswith('f"') or fill_stripped.startswith("f'"):
            return STR

        # Empty list
        if fill_stripped == "[]":
            return ListType(ANY)

        # Empty dict
        if fill_stripped == "{}":
            return DictType(ANY, ANY)

        # Empty tuple
        if fill_stripped == "()":
            return TupleType(())

        # Lambda expressions
        if fill_stripped.startswith("lambda"):
            return self._infer_lambda_type(fill_stripped)

        # Can't determine type
        return None

    def _infer_lambda_type(self, lambda_str: str) -> Optional[FunctionType]:
        """Infer the type of a lambda expression."""
        # Count parameters by parsing lambda header
        try:
            # Extract parameter part: "lambda x, y: ..." -> "x, y"
            colon_idx = lambda_str.index(":")
            param_part = lambda_str[6:colon_idx].strip()

            if not param_part:
                return FunctionType(params=(), returns=ANY)

            params = [p.strip() for p in param_part.split(",")]
            param_types = tuple(ANY for _ in params)
            return FunctionType(params=param_types, returns=ANY)

        except (ValueError, IndexError):
            return FunctionType(params=(), returns=ANY)

    def _determine_relationship(
        self,
        filled_hole: Hole[Any],
        dependent_hole: Hole[Any],
    ) -> Optional[str]:
        """Determine the relationship between two holes.

        Relationships are inferred from hole names and metadata.
        """
        filled_name = filled_hole.id.name.lower()
        dep_name = dependent_hole.id.name.lower()

        # Return type relationship
        if "return" in filled_name or "result" in filled_name:
            return "return_type"

        # Argument relationship
        if "arg" in filled_name or "param" in filled_name:
            return "argument"

        # Assignment relationship
        if "target" in filled_name or "lhs" in filled_name:
            if "value" in dep_name or "rhs" in dep_name:
                return "assignment"

        if "value" in filled_name or "rhs" in filled_name:
            if "target" in dep_name or "lhs" in dep_name:
                return "assignment"

        # Element relationship
        if "element" in filled_name or "item" in filled_name:
            return "element"

        if "key" in filled_name:
            return "key"

        if "value" in filled_name and "dict" in dep_name:
            return "value"

        return None

    def _infer_return_type_constraint(
        self,
        fill_type: Type,
        dependent_hole: Hole[Any],
    ) -> Optional[TypeConstraint]:
        """Infer constraint when fill is in return position."""
        # If filled with a type T, dependent holes in same function context
        # should be consistent with T as the return type
        return type_expecting(fill_type)

    def _infer_argument_constraint(
        self,
        filled_hole: Hole[Any],
        fill_type: Type,
        dependent_hole: Hole[Any],
    ) -> Optional[TypeConstraint]:
        """Infer constraint when fill is an argument."""
        # Check if dependent hole is the function being called
        if "func" in dependent_hole.id.name.lower() or "call" in dependent_hole.id.name.lower():
            # Dependent hole should be a function that accepts fill_type
            return type_expecting(FunctionType(params=(fill_type,), returns=ANY))
        return None

    def _infer_assignment_constraint(
        self,
        fill_type: Type,
        dependent_hole: Hole[Any],
    ) -> Optional[TypeConstraint]:
        """Infer constraint when fill is assignment target."""
        # Value must match target type
        return type_expecting(fill_type)

    def _infer_element_constraint(
        self,
        fill_type: Type,
        dependent_hole: Hole[Any],
    ) -> Optional[TypeConstraint]:
        """Infer constraint when fill is a container element."""
        # Other elements should match
        return type_expecting(fill_type)

    def _infer_dict_constraint(
        self,
        fill_type: Type,
        relationship: str,
        dependent_hole: Hole[Any],
    ) -> Optional[TypeConstraint]:
        """Infer constraint for dict key/value fills."""
        if relationship == "key":
            # Other keys should match
            return type_expecting(fill_type)
        else:
            # Other values should match
            return type_expecting(fill_type)


# =============================================================================
# Factory Functions
# =============================================================================


def create_fill_generator(
    language: str = "python",
    type_environment: Optional[TypeEnvironment] = None,
) -> TypeAwareFillGenerator:
    """Create a TypeAwareFillGenerator with configuration.

    Args:
        language: Target programming language
        type_environment: Type bindings for identifier lookups

    Returns:
        Configured TypeAwareFillGenerator
    """
    return TypeAwareFillGenerator(
        language=language,
        type_environment=type_environment or TypeEnvironment.empty(),
    )


def create_constraint_checker(
    language: str = "python",
    enabled_domains: Optional[Set[str]] = None,
    type_context: Optional[Dict[str, str]] = None,
) -> UnifiedConstraintChecker:
    """Create a UnifiedConstraintChecker with configuration.

    Args:
        language: Target programming language
        enabled_domains: Domains to verify against
        type_context: Type bindings for the type domain

    Returns:
        Configured UnifiedConstraintChecker
    """
    return UnifiedConstraintChecker(
        language=language,
        enabled_domains=enabled_domains,
        type_context=type_context,
    )


def create_constraint_inferencer(
    language: str = "python",
) -> TypeConstraintInferencer:
    """Create a TypeConstraintInferencer.

    Args:
        language: Target programming language

    Returns:
        Configured TypeConstraintInferencer
    """
    return TypeConstraintInferencer(language=language)
