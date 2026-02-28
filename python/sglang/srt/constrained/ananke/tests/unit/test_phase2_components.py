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
"""Unit tests for Phase 2 components.

Tests for:
- Phase 2.1: Cross-domain constraint propagation edges
- Phase 2.2: LSP confidence-based hard blocking
- Phase 2.3: Enhanced generic type inference
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

# Import Phase 2.1 components (propagation edges)
try:
    from propagation.edges import (
        TypesToSyntaxEdge,
        ImportsToTypesEdge,
        ControlFlowToSemanticsEdge,
        SyntaxToTypesEdge,
    )
    from core.domain import GenerationContext
    from core.constraint import Constraint
except ImportError:
    from sglang.srt.constrained.ananke.propagation.edges import (
        TypesToSyntaxEdge,
        ImportsToTypesEdge,
        ControlFlowToSemanticsEdge,
        SyntaxToTypesEdge,
    )
    from sglang.srt.constrained.ananke.core.domain import GenerationContext
    from sglang.srt.constrained.ananke.core.constraint import Constraint

# Import Phase 2.2 components (LSP)
try:
    from lsp.integration import (
        DiagnosticConfidence,
        LSPDiagnosticProvider,
    )
    from lsp.protocol import (
        Diagnostic,
        DiagnosticSeverity,
        Position,
        Range,
    )
except ImportError:
    from sglang.srt.constrained.ananke.lsp.integration import (
        DiagnosticConfidence,
        LSPDiagnosticProvider,
    )
    from sglang.srt.constrained.ananke.lsp.protocol import (
        Diagnostic,
        DiagnosticSeverity,
        Position,
        Range,
    )

# Import Phase 2.3 components (type inference)
try:
    from domains.types.inference import (
        BoundedTypeVar,
        TypeParameterInference,
        RecursiveTypeHandler,
        create_bounded_var,
        infer_type_params,
        unify_with_bounds,
    )
    from domains.types.constraint import (
        INT, STR, FLOAT, BOOL, ANY, NEVER,
        TypeVar, ListType, FunctionType, ClassType, DictType, TupleType,
        Variance,
    )
except ImportError:
    from sglang.srt.constrained.ananke.domains.types.inference import (
        BoundedTypeVar,
        TypeParameterInference,
        RecursiveTypeHandler,
        create_bounded_var,
        infer_type_params,
        unify_with_bounds,
    )
    from sglang.srt.constrained.ananke.domains.types.constraint import (
        INT, STR, FLOAT, BOOL, ANY, NEVER,
        TypeVar, ListType, FunctionType, ClassType, DictType, TupleType,
        Variance,
    )


# =============================================================================
# Mock Constraints for Testing
# =============================================================================


@dataclass
class MockTypeConstraint:
    """Mock type constraint for testing edges."""
    expected: Optional[str] = None
    _is_bottom: bool = False

    def is_bottom(self) -> bool:
        return self._is_bottom

    def with_metadata(self, metadata: dict) -> "MockTypeConstraint":
        return MockTypeConstraint(expected=self.expected)


@dataclass
class MockSyntaxConstraint:
    """Mock syntax constraint for testing edges."""
    position: Optional[str] = None
    current_rule: str = ""
    in_function_call: bool = False
    in_return: bool = False
    _is_bottom: bool = False

    def is_bottom(self) -> bool:
        return self._is_bottom

    def with_metadata(self, metadata: dict) -> "MockSyntaxConstraint":
        return MockSyntaxConstraint(position=self.position, current_rule=self.current_rule)


@dataclass
class MockImportConstraint:
    """Mock import constraint for testing edges."""
    available: list = field(default_factory=list)
    required: list = field(default_factory=list)
    _is_bottom: bool = False

    def is_bottom(self) -> bool:
        return self._is_bottom

    def with_metadata(self, metadata: dict) -> "MockImportConstraint":
        return MockImportConstraint(available=self.available)


@dataclass
class MockModuleSpec:
    """Mock module spec for import testing."""
    name: str


@dataclass
class MockControlFlowConstraint:
    """Mock control flow constraint for testing edges."""
    is_reachable: bool = True
    in_loop: bool = False
    loop_depth: int = 0
    in_branch: bool = False
    _is_bottom: bool = False

    def is_bottom(self) -> bool:
        return self._is_bottom


@dataclass
class MockSemanticConstraint:
    """Mock semantic constraint for testing edges."""
    _is_bottom: bool = False
    _metadata: dict = field(default_factory=dict)

    def is_bottom(self) -> bool:
        return self._is_bottom

    def with_metadata(self, metadata: dict) -> "MockSemanticConstraint":
        new = MockSemanticConstraint()
        new._metadata = metadata
        return new


# =============================================================================
# Phase 2.1: Propagation Edge Tests
# =============================================================================


class TestTypesToSyntaxEdge:
    """Tests for TypesToSyntaxEdge."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context = GenerationContext(vocab_size=100, device="cpu")
        self.edge = TypesToSyntaxEdge()

    def test_extracts_expected_type(self):
        """Edge extracts expected type from constraint."""
        source = MockTypeConstraint(expected="int")
        target = MockSyntaxConstraint()

        result = self.edge.propagate(source, target, self.context)
        # Should return target (possibly with metadata)
        assert result is not None

    def test_numeric_type_hints(self):
        """Numeric types generate appropriate hints."""
        source = MockTypeConstraint(expected="int")
        target = MockSyntaxConstraint()

        hints = self.edge._type_to_syntax_hints("int", self.context)

        assert hints.get("allow_numeric_literals") is True
        assert hints.get("block_string_literals") is True

    def test_string_type_hints(self):
        """String types generate appropriate hints."""
        hints = self.edge._type_to_syntax_hints("str", self.context)

        assert hints.get("allow_string_literals") is True
        assert hints.get("block_numeric_literals") is True

    def test_callable_type_hints(self):
        """Callable types generate appropriate hints."""
        hints = self.edge._type_to_syntax_hints("Callable[[int], str]", self.context)

        assert hints.get("block_non_callable_literals") is True
        assert hints.get("allow_lambda") is True

    def test_collection_type_hints(self):
        """Collection types generate appropriate hints."""
        hints = self.edge._type_to_syntax_hints("List[int]", self.context)
        assert hints.get("allow_list_literal") is True

        hints = self.edge._type_to_syntax_hints("Dict[str, int]", self.context)
        assert hints.get("allow_dict_literal") is True

    def test_handles_bottom_source(self):
        """Edge handles BOTTOM source constraint."""
        source = MockTypeConstraint(_is_bottom=True)
        target = MockSyntaxConstraint()

        result = self.edge.propagate(source, target, self.context)
        assert result == target

    def test_handles_bottom_target(self):
        """Edge handles BOTTOM target constraint."""
        source = MockTypeConstraint(expected="int")
        target = MockSyntaxConstraint(_is_bottom=True)

        result = self.edge.propagate(source, target, self.context)
        assert result == target


class TestSyntaxToTypesEdge:
    """Tests for SyntaxToTypesEdge."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context = GenerationContext(vocab_size=100, device="cpu")
        self.edge = SyntaxToTypesEdge()

    def test_function_call_hints(self):
        """Function call context generates callable hint."""
        source = MockSyntaxConstraint(in_function_call=True)
        target = MockTypeConstraint()

        hints = self.edge._syntax_to_type_hints({"in_call": True})
        assert hints.get("expect_callable") is True
        assert hints.get("position") == "function"

    def test_return_context_hints(self):
        """Return context generates return type hint."""
        hints = self.edge._syntax_to_type_hints({"in_return": True})
        assert hints.get("position") == "return_value"
        assert hints.get("expect_return_compatible") is True

    def test_collection_context_hints(self):
        """Collection context generates appropriate hints."""
        hints = self.edge._syntax_to_type_hints({"in_list": True})
        assert hints.get("position") == "list_element"
        assert hints.get("in_collection") is True


class TestImportsToTypesEdge:
    """Tests for ImportsToTypesEdge."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context = GenerationContext(vocab_size=100, device="cpu")
        self.edge = ImportsToTypesEdge()

    def test_extracts_available_modules(self):
        """Edge extracts modules from import constraint."""
        source = MockImportConstraint(
            available=[MockModuleSpec("typing"), MockModuleSpec("numpy")]
        )

        modules = self.edge._extract_available_modules(source)
        assert "typing" in modules
        assert "numpy" in modules

    def test_derives_typing_types(self):
        """Typing module provides typing types."""
        accessible = self.edge._derive_accessible_types({"typing"})

        assert "List" in accessible
        assert "Dict" in accessible
        assert "Optional" in accessible

    def test_derives_numpy_types(self):
        """Numpy module provides numpy types."""
        accessible = self.edge._derive_accessible_types({"numpy"})

        assert "ndarray" in accessible
        assert "dtype" in accessible

    def test_handles_empty_modules(self):
        """Edge handles empty module set."""
        source = MockImportConstraint(available=[])
        target = MockTypeConstraint()

        result = self.edge.propagate(source, target, self.context)
        assert result == target


class TestControlFlowToSemanticsEdge:
    """Tests for ControlFlowToSemanticsEdge."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context = GenerationContext(vocab_size=100, device="cpu")
        self.edge = ControlFlowToSemanticsEdge()

    def test_extracts_unreachable_info(self):
        """Edge extracts reachability info."""
        source = MockControlFlowConstraint(is_reachable=False)

        info = self.edge._extract_cf_info(source)
        assert info.get("unreachable") is True

    def test_extracts_loop_info(self):
        """Edge extracts loop context."""
        source = MockControlFlowConstraint(in_loop=True, loop_depth=2)

        info = self.edge._extract_cf_info(source)
        assert info.get("in_loop") is True
        assert info.get("loop_depth") == 2

    def test_extracts_branch_info(self):
        """Edge extracts branch context."""
        source = MockControlFlowConstraint(in_branch=True)

        info = self.edge._extract_cf_info(source)
        assert info.get("in_branch") is True


# =============================================================================
# Phase 2.2: LSP Diagnostic Confidence Tests
# =============================================================================


class TestDiagnosticConfidence:
    """Tests for DiagnosticConfidence scoring."""

    def test_dataclass_fields(self):
        """DiagnosticConfidence has expected fields."""
        conf = DiagnosticConfidence(
            confidence=0.95,
            is_type_error=True,
            source="pyright",
            can_hard_block=True,
        )

        assert conf.confidence == 0.95
        assert conf.is_type_error is True
        assert conf.source == "pyright"
        assert conf.can_hard_block is True


class TestLSPDiagnosticProvider:
    """Tests for LSPDiagnosticProvider with confidence scoring."""

    def setup_method(self):
        """Set up test fixtures."""
        self.provider = LSPDiagnosticProvider(
            hard_block_threshold=0.95,
            enable_hard_blocking=True,
        )

    def test_high_confidence_pyright_error(self):
        """Pyright type error gets high confidence."""
        diagnostic = Diagnostic(
            range=Range(Position(0, 0), Position(0, 10)),
            message="Type 'str' is not assignable to type 'int'",
            severity=DiagnosticSeverity.Error,
            source="pyright",
            code="2322",
        )

        conf = self.provider._compute_confidence(diagnostic)

        # pyright (+0.3) + type error code (+0.15) + message keywords (+0.1) + error (+0.1)
        assert conf.confidence >= 0.95
        assert conf.is_type_error is True
        assert conf.can_hard_block is True

    def test_low_confidence_warning(self):
        """Warnings get lower confidence."""
        diagnostic = Diagnostic(
            range=Range(Position(0, 0), Position(0, 10)),
            message="Variable may be undefined",
            severity=DiagnosticSeverity.Warning,
            source="unknown",
        )

        conf = self.provider._compute_confidence(diagnostic)

        assert conf.confidence < 0.95
        assert conf.can_hard_block is False

    def test_should_hard_block_with_high_confidence(self):
        """should_hard_block returns True for high-confidence type errors."""
        diagnostic = Diagnostic(
            range=Range(Position(0, 0), Position(0, 10)),
            message="Type error: expected int",
            severity=DiagnosticSeverity.Error,
            source="pyright",
            code="reportArgumentType",
        )

        self.provider.update_diagnostics("file:///test.py", [diagnostic])

        assert self.provider.should_hard_block("file:///test.py") is True

    def test_should_not_hard_block_low_confidence(self):
        """should_hard_block returns False for low-confidence errors."""
        diagnostic = Diagnostic(
            range=Range(Position(0, 0), Position(0, 10)),
            message="Some generic warning",
            severity=DiagnosticSeverity.Warning,
            source="unknown",
        )

        self.provider.update_diagnostics("file:///test.py", [diagnostic])

        assert self.provider.should_hard_block("file:///test.py") is False

    def test_clear_diagnostics_clears_confidence(self):
        """clear_diagnostics also clears confidence cache."""
        diagnostic = Diagnostic(
            range=Range(Position(0, 0), Position(0, 10)),
            message="Type error",
            severity=DiagnosticSeverity.Error,
            source="pyright",
        )

        self.provider.update_diagnostics("file:///test.py", [diagnostic])
        assert self.provider.should_hard_block("file:///test.py")

        self.provider.clear_diagnostics("file:///test.py")
        assert not self.provider.should_hard_block("file:///test.py")

    def test_get_high_confidence_type_errors(self):
        """get_high_confidence_type_errors filters correctly."""
        diag_high = Diagnostic(
            range=Range(Position(0, 0), Position(0, 10)),
            message="Type 'str' is not assignable to type 'int'",
            severity=DiagnosticSeverity.Error,
            source="pyright",
            code="2322",
        )
        diag_low = Diagnostic(
            range=Range(Position(1, 0), Position(1, 10)),
            message="Some hint",
            severity=DiagnosticSeverity.Hint,
            source="unknown",
        )

        self.provider.update_diagnostics("file:///test.py", [diag_high, diag_low])

        errors = self.provider.get_high_confidence_type_errors("file:///test.py")
        assert len(errors) == 1
        assert errors[0][0] == diag_high

    def test_disabled_hard_blocking(self):
        """Hard blocking can be disabled."""
        provider = LSPDiagnosticProvider(
            hard_block_threshold=0.95,
            enable_hard_blocking=False,
        )

        diagnostic = Diagnostic(
            range=Range(Position(0, 0), Position(0, 10)),
            message="Type error",
            severity=DiagnosticSeverity.Error,
            source="pyright",
            code="2322",
        )

        provider.update_diagnostics("file:///test.py", [diagnostic])

        # Hard blocking disabled - should not hard block
        assert not provider.should_hard_block("file:///test.py")


# =============================================================================
# Phase 2.3: Type Inference Tests
# =============================================================================


class TestBoundedTypeVar:
    """Tests for BoundedTypeVar."""

    def test_create_basic(self):
        """Create a basic bounded type var."""
        T = create_bounded_var("T")

        assert T.name == "T"
        assert T.upper_bound == ANY
        assert T.lower_bound == NEVER

    def test_create_with_upper_bound(self):
        """Create bounded var with upper bound."""
        T = create_bounded_var("T", upper=FLOAT)

        assert T.upper_bound == FLOAT
        assert T.is_within_bounds(INT)  # int <: float
        assert T.is_within_bounds(FLOAT)
        assert not T.is_within_bounds(STR)  # str NOT <: float

    def test_create_with_variance(self):
        """Create bounded var with variance."""
        T = create_bounded_var("T", variance=Variance.COVARIANT)

        assert T.variance == Variance.COVARIANT

    def test_free_type_vars(self):
        """free_type_vars includes self and bounds."""
        T = create_bounded_var("T")
        assert "T" in T.free_type_vars()

    def test_substitute(self):
        """Substitution replaces variable."""
        T = create_bounded_var("T")
        result = T.substitute({"T": INT})

        assert result == INT

    def test_substitute_preserves_unmatched(self):
        """Substitution preserves unmatched variables."""
        T = create_bounded_var("T")
        result = T.substitute({"U": INT})

        assert result == T

    def test_repr(self):
        """repr is readable."""
        T = create_bounded_var("T", upper=FLOAT)
        assert "T" in repr(T)
        assert "float" in repr(T)


class TestTypeParameterInference:
    """Tests for TypeParameterInference."""

    def setup_method(self):
        """Set up test fixtures."""
        self.inference = TypeParameterInference()

    def test_infer_simple_type_var(self):
        """Infer type variable from argument."""
        T = TypeVar("T")

        self.inference.infer_from_argument(T, INT)
        result = self.inference.solve()

        assert result.is_success
        assert result.substitution.mapping.get("T") == INT

    def test_infer_from_list(self):
        """Infer type variable from list type."""
        T = TypeVar("T")
        list_T = ListType(element=T)
        list_int = ListType(element=INT)

        self.inference.infer_from_argument(list_T, list_int)
        result = self.inference.solve()

        assert result.is_success
        assert result.substitution.mapping.get("T") == INT

    def test_infer_from_dict(self):
        """Infer type variables from dict type."""
        K = TypeVar("K")
        V = TypeVar("V")
        dict_KV = DictType(key=K, value=V)
        dict_str_int = DictType(key=STR, value=INT)

        self.inference.infer_from_argument(dict_KV, dict_str_int)
        result = self.inference.solve()

        assert result.is_success
        assert result.substitution.mapping.get("K") == STR
        assert result.substitution.mapping.get("V") == INT

    def test_infer_from_tuple(self):
        """Infer type variables from tuple type."""
        T = TypeVar("T")
        U = TypeVar("U")
        tuple_TU = TupleType(elements=(T, U))
        tuple_int_str = TupleType(elements=(INT, STR))

        self.inference.infer_from_argument(tuple_TU, tuple_int_str)
        result = self.inference.solve()

        assert result.is_success
        assert result.substitution.mapping.get("T") == INT
        assert result.substitution.mapping.get("U") == STR

    def test_infer_from_function(self):
        """Infer type variables from function type."""
        T = TypeVar("T")
        fn_T = FunctionType(params=(T,), returns=T)
        fn_int = FunctionType(params=(INT,), returns=INT)

        self.inference.infer_from_argument(fn_T, fn_int)
        result = self.inference.solve()

        assert result.is_success
        assert result.substitution.mapping.get("T") == INT

    def test_infer_from_class_type(self):
        """Infer type variables from class type."""
        T = TypeVar("T")
        class_T = ClassType(name="Box", type_args=(T,))
        class_int = ClassType(name="Box", type_args=(INT,))

        self.inference.infer_from_argument(class_T, class_int)
        result = self.inference.solve()

        assert result.is_success
        assert result.substitution.mapping.get("T") == INT

    def test_add_constraint_exact(self):
        """Add exact constraint."""
        self.inference.add_constraint("T", INT, "exact", "test")

        result = self.inference.solve()
        assert result.is_success
        assert result.substitution.mapping.get("T") == INT

    def test_multiple_constraints_same_var(self):
        """Multiple constraints on same variable resolve."""
        self.inference.add_constraint("T", INT, "exact", "test1")
        self.inference.add_constraint("T", INT, "exact", "test2")

        result = self.inference.solve()
        assert result.is_success
        assert result.substitution.mapping.get("T") == INT


class TestRecursiveTypeHandler:
    """Tests for RecursiveTypeHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = RecursiveTypeHandler()

    def test_simple_type_not_recursive(self):
        """Simple types are not recursive."""
        assert not self.handler.is_recursive(INT)
        assert not self.handler.is_recursive(STR)
        assert not self.handler.is_recursive(ListType(element=INT))

    def test_enter_exit_cycle(self):
        """Enter/exit cycle detection works."""
        ty = ListType(element=INT)

        assert self.handler.enter(ty)
        assert not self.handler.enter(ty)  # Already processing
        self.handler.exit(ty)
        assert self.handler.enter(ty)  # Can enter again

    def test_depth_limit(self):
        """Depth limit prevents infinite recursion."""
        handler = RecursiveTypeHandler(max_depth=2)

        ty1 = INT
        ty2 = STR
        ty3 = FLOAT

        assert handler.enter(ty1)
        assert handler.enter(ty2)
        assert not handler.enter(ty3)  # Depth limit reached


class TestInferTypeParams:
    """Tests for infer_type_params convenience function."""

    def test_infer_list_element(self):
        """Infer list element type."""
        T = TypeVar("T")
        generic = ListType(element=T)
        concrete = ListType(element=INT)

        subst = infer_type_params(generic, concrete)

        assert subst is not None
        assert subst.mapping.get("T") == INT

    def test_infer_returns_none_on_mismatch(self):
        """Returns None when types don't match."""
        T = TypeVar("T")
        generic = ListType(element=T)
        concrete = INT  # Not a list

        subst = infer_type_params(generic, concrete)

        assert subst is None


class TestUnifyWithBounds:
    """Tests for unify_with_bounds."""

    def test_unify_bounded_var_in_bounds(self):
        """Unify bounded var with type within bounds."""
        T = create_bounded_var("T", upper=FLOAT)

        result = unify_with_bounds(T, INT)

        assert result.is_success
        assert result.substitution.mapping.get("T") == INT

    def test_unify_bounded_var_out_of_bounds(self):
        """Unify bounded var with type outside bounds fails."""
        T = create_bounded_var("T", upper=FLOAT)

        result = unify_with_bounds(T, STR)

        assert result.is_failure

    def test_unify_regular_types(self):
        """Falls back to regular unify for non-bounded types."""
        result = unify_with_bounds(INT, INT)

        assert result.is_success
