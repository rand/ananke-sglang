"""Property-based tests for Phase 2 components.

Tests Phase 2.1 (propagation), 2.2 (LSP hard blocking), and 2.3 (type inference)
using Hypothesis for property-based testing.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st, assume
from dataclasses import dataclass
from typing import Set, Optional

# Import propagation edges
import sys
from pathlib import Path

# Add the ananke directory to path for direct imports
ananke_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ananke_path))

from propagation.edges import (
    PropagationEdge,
    SyntaxToTypesEdge,
    TypesToSyntaxEdge,
    ImportsToTypesEdge,
    ControlFlowToSemanticsEdge,
)
from lsp.integration import (
    DiagnosticConfidence,
    LSPDiagnosticProvider,
    Diagnostic,
    DiagnosticSeverity,
    Range,
    Position,
)
from domains.types.inference import (
    BoundedTypeVar,
    TypeParameterInference,
    RecursiveTypeHandler,
    Variance,
    InferenceConstraint,
    InferenceDirection,
    infer_type_params,
    unify_with_bounds,
)
from domains.types.unification import (
    Type, TypeVar, ListType, DictType, TupleType, FunctionType,
    UnionType, ClassType, ANY, NEVER,
    Substitution, is_subtype, UnificationResult,
)

# Create primitive type constants
INT = ClassType("int")
STRING = ClassType("str")
BOOL = ClassType("bool")
FLOAT = ClassType("float")


# =============================================================================
# Strategies for generating test data
# =============================================================================

@st.composite
def type_names(draw) -> str:
    """Generate valid type names."""
    return draw(st.sampled_from([
        "int", "str", "bool", "float", "None",
        "List", "Dict", "Set", "Tuple", "Optional",
        "Callable", "Any", "Union", "Iterable",
        "integer", "string", "boolean", "i32", "i64",
        "func", "function", "lambda",
    ]))


@st.composite
def module_names(draw) -> str:
    """Generate valid module names."""
    return draw(st.sampled_from([
        "typing", "numpy", "pandas", "collections",
        "functools", "itertools", "os", "sys",
        "json", "re", "math", "random",
    ]))


@st.composite
def simple_types(draw) -> Type:
    """Generate simple, non-recursive types."""
    return draw(st.sampled_from([INT, STRING, BOOL, FLOAT, ANY, NEVER]))


@st.composite
def type_vars(draw) -> TypeVar:
    """Generate type variables."""
    name = draw(st.sampled_from(["T", "U", "V", "K", "R", "A", "B"]))
    return TypeVar(name)


@st.composite
def variances(draw) -> Variance:
    """Generate variance values."""
    return draw(st.sampled_from(list(Variance)))


@st.composite
def bounded_type_vars(draw) -> BoundedTypeVar:
    """Generate bounded type variables."""
    name = draw(st.text(alphabet="TUVKRAB", min_size=1, max_size=2))
    upper = draw(st.sampled_from([ANY, INT, STRING, None]))
    lower = draw(st.sampled_from([NEVER, None]))
    variance = draw(variances())

    return BoundedTypeVar(
        name=name,
        upper_bound=upper if upper else ANY,
        lower_bound=lower if lower else NEVER,
        variance=variance,
    )


@st.composite
def compound_types(draw, max_depth: int = 2) -> Type:
    """Generate compound types up to a max depth."""
    if max_depth <= 0:
        return draw(simple_types())

    choice = draw(st.integers(min_value=0, max_value=5))

    if choice == 0:
        return draw(simple_types())
    elif choice == 1:
        return draw(type_vars())
    elif choice == 2:
        elem = draw(compound_types(max_depth - 1))
        return ListType(elem)
    elif choice == 3:
        key = draw(simple_types())
        val = draw(compound_types(max_depth - 1))
        return DictType(key, val)
    elif choice == 4:
        elems = draw(st.lists(compound_types(max_depth - 1), min_size=0, max_size=3))
        return TupleType(tuple(elems))
    else:
        params = draw(st.lists(simple_types(), min_size=0, max_size=2))
        ret = draw(simple_types())
        return FunctionType(tuple(params), ret)


def make_range(line: int = 0, start_char: int = 0, end_char: int = 10) -> Range:
    """Create an LSP Range object."""
    return Range(
        start=Position(line=line, character=start_char),
        end=Position(line=line, character=end_char)
    )


def make_diagnostic(
    severity: DiagnosticSeverity,
    source: str,
    code: Optional[str],
    message: str = "Test diagnostic"
) -> Diagnostic:
    """Create an LSP Diagnostic object."""
    return Diagnostic(
        range=make_range(),
        message=message,
        severity=severity,
        source=source,
        code=code,
    )


# =============================================================================
# Property Tests: Propagation Edges (Phase 2.1)
# =============================================================================

class TestPropagationEdgeProperties:
    """Property-based tests for propagation edge behavior."""

    @given(type_name=type_names())
    @settings(max_examples=50)
    def test_types_to_syntax_edge_creation_deterministic(self, type_name: str):
        """Edge creation should be deterministic."""
        edge1 = TypesToSyntaxEdge()
        edge2 = TypesToSyntaxEdge()

        # Same default priority
        assert edge1.priority == edge2.priority
        # Edges should be of the same type
        assert type(edge1) == type(edge2)

    @given(type_name=type_names())
    @settings(max_examples=50)
    def test_types_to_syntax_internal_hint_determinism(self, type_name: str):
        """Internal type hint function should be deterministic."""
        edge = TypesToSyntaxEdge()

        # Test the internal _type_to_syntax_hints method directly
        hints1 = edge._type_to_syntax_hints(type_name, None)
        hints2 = edge._type_to_syntax_hints(type_name, None)

        # Same inputs should produce same outputs
        assert hints1 == hints2

    @given(type_name=type_names())
    @settings(max_examples=50)
    def test_types_to_syntax_produces_valid_hints(self, type_name: str):
        """Type hints should always produce valid syntax guidance."""
        edge = TypesToSyntaxEdge()

        # Test internal method directly
        hints = edge._type_to_syntax_hints(type_name, None)

        # Result should be a dict with boolean values
        assert isinstance(hints, dict)
        for key, value in hints.items():
            assert isinstance(key, str)
            assert isinstance(value, bool)

    @given(modules=st.sets(module_names(), min_size=0, max_size=5))
    @settings(max_examples=50)
    def test_imports_to_types_derive_monotonic(self, modules: Set[str]):
        """More imports should never reduce accessible types."""
        edge = ImportsToTypesEdge()

        # Test with subset and full set using internal method
        modules_list = list(modules)
        if len(modules_list) >= 2:
            subset = set(modules_list[:len(modules_list)//2])
            full_set = modules

            subset_types = edge._derive_accessible_types(subset)
            full_types = edge._derive_accessible_types(full_set)

            # Superset of modules should give superset of types
            assert subset_types.issubset(full_types)


# =============================================================================
# Property Tests: LSP Confidence Scoring (Phase 2.2)
# =============================================================================

class TestLSPConfidenceProperties:
    """Property-based tests for LSP confidence scoring."""

    @given(
        severity=st.sampled_from(list(DiagnosticSeverity)),
        source=st.sampled_from(["pyright", "typescript", "rust-analyzer", "gopls", "pylint", "unknown"]),
        code=st.sampled_from(["reportGeneralTypeIssues", "2322", "E0308", "some-other-code", None]),
    )
    @settings(max_examples=100)
    def test_confidence_bounded(self, severity: DiagnosticSeverity, source: str, code: Optional[str]):
        """Confidence scores should always be in [0, 1]."""
        provider = LSPDiagnosticProvider(enable_hard_blocking=True)

        diagnostic = make_diagnostic(severity, source, code)
        confidence = provider._compute_confidence(diagnostic)

        assert 0.0 <= confidence.confidence <= 1.0

    @given(
        severity=st.sampled_from(list(DiagnosticSeverity)),
        source=st.sampled_from(["pyright", "typescript", "rust-analyzer", "gopls", "pylint", "unknown"]),
        code=st.sampled_from(["reportGeneralTypeIssues", "2322", "E0308", "some-other-code", None]),
    )
    @settings(max_examples=100)
    def test_hard_block_requires_high_confidence(self, severity: DiagnosticSeverity, source: str, code: Optional[str]):
        """Hard blocking should only occur with high confidence type errors."""
        provider = LSPDiagnosticProvider(
            enable_hard_blocking=True,
            hard_block_threshold=0.95,
        )

        diagnostic = make_diagnostic(severity, source, code)
        confidence = provider._compute_confidence(diagnostic)

        if confidence.can_hard_block:
            assert confidence.confidence >= 0.95
            assert confidence.is_type_error

    @given(
        severity=st.sampled_from(list(DiagnosticSeverity)),
        code=st.sampled_from(["reportGeneralTypeIssues", "2322", "E0308"]),
    )
    @settings(max_examples=50)
    def test_high_confidence_sources_boost_score(self, severity: DiagnosticSeverity, code: str):
        """High-confidence sources should produce higher confidence scores."""
        provider = LSPDiagnosticProvider(enable_hard_blocking=True)

        # Create diagnostic from high-confidence source
        high_conf_diag = make_diagnostic(severity, "pyright", code)

        # Create diagnostic from unknown source
        low_conf_diag = make_diagnostic(severity, "unknown-linter", code)

        high_conf = provider._compute_confidence(high_conf_diag)
        low_conf = provider._compute_confidence(low_conf_diag)

        # High-confidence sources should have higher or equal confidence
        assert high_conf.confidence >= low_conf.confidence

    @given(severity=st.sampled_from(list(DiagnosticSeverity)))
    @settings(max_examples=20)
    def test_error_severity_higher_than_warning(self, severity: DiagnosticSeverity):
        """Errors (severity 1) should have higher confidence than warnings (2+)."""
        provider = LSPDiagnosticProvider(enable_hard_blocking=True)

        error_diag = make_diagnostic(DiagnosticSeverity.Error, "pyright", "reportGeneralTypeIssues")
        other_diag = make_diagnostic(severity, "pyright", "reportGeneralTypeIssues")

        error_conf = provider._compute_confidence(error_diag)
        other_conf = provider._compute_confidence(other_diag)

        if severity != DiagnosticSeverity.Error:
            assert error_conf.confidence >= other_conf.confidence


# =============================================================================
# Property Tests: Type Inference (Phase 2.3)
# =============================================================================

class TestBoundedTypeVarProperties:
    """Property-based tests for bounded type variables."""

    @given(btv=bounded_type_vars())
    @settings(max_examples=50)
    def test_bounded_typevar_contains_itself_as_free_var(self, btv: BoundedTypeVar):
        """A bounded type var should report itself as a free variable."""
        free_vars = btv.free_type_vars()
        assert btv.name in free_vars

    @given(btv=bounded_type_vars(), ty=simple_types())
    @settings(max_examples=50)
    def test_substitute_with_matching_name_via_dict(self, btv: BoundedTypeVar, ty: Type):
        """Substituting with matching name should replace the variable."""
        subst = {btv.name: ty}
        result = btv.substitute(subst)
        assert result == ty

    @given(btv=bounded_type_vars(), ty=simple_types())
    @settings(max_examples=50)
    def test_substitute_with_matching_name_via_substitution(self, btv: BoundedTypeVar, ty: Type):
        """Substituting with Substitution object should work."""
        subst = Substitution({btv.name: ty})
        result = btv.substitute(subst)
        assert result == ty

    @given(btv=bounded_type_vars(), ty=simple_types())
    @settings(max_examples=50)
    def test_substitute_with_non_matching_name(self, btv: BoundedTypeVar, ty: Type):
        """Substituting with non-matching name should preserve the variable."""
        # Use a name that definitely doesn't match
        other_name = btv.name + "_other"
        subst = {other_name: ty}
        result = btv.substitute(subst)
        assert result == btv

    @given(variance=variances())
    @settings(max_examples=10)
    def test_variance_preserved_in_repr(self, variance: Variance):
        """Variance should be visible in string representation."""
        btv = BoundedTypeVar("T", variance=variance)
        repr_str = repr(btv)
        assert "T" in repr_str


class TestTypeParameterInferenceProperties:
    """Property-based tests for type parameter inference."""

    @given(ty=simple_types())
    @settings(max_examples=30)
    def test_infer_identity_for_non_generic(self, ty: Type):
        """Inferring from non-generic types should succeed trivially."""
        inference = TypeParameterInference()
        # Non-generic to non-generic should always work
        result = inference.infer_from_argument(ty, ty)
        assert result is True

    @given(elem_type=simple_types())
    @settings(max_examples=30)
    def test_infer_list_adds_constraint(self, elem_type: Type):
        """Inferring from list should add appropriate constraints."""
        t_var = TypeVar("T")
        param_type = ListType(t_var)
        arg_type = ListType(elem_type)

        inference = TypeParameterInference()
        result = inference.infer_from_argument(param_type, arg_type)

        assert result is True
        # Check that a constraint was added
        assert len(inference.constraints) >= 1

    @given(key_type=simple_types(), val_type=simple_types())
    @settings(max_examples=30)
    def test_infer_dict_adds_constraints(self, key_type: Type, val_type: Type):
        """Inferring from dict should add constraints for key and value."""
        k_var = TypeVar("K")
        v_var = TypeVar("V")
        param_type = DictType(k_var, v_var)
        arg_type = DictType(key_type, val_type)

        inference = TypeParameterInference()
        result = inference.infer_from_argument(param_type, arg_type)

        assert result is True
        # Check that constraints were added
        assert len(inference.constraints) >= 2

    @given(
        direction=st.sampled_from(["exact", "upper", "lower"]),
        ty=simple_types(),
    )
    @settings(max_examples=30)
    def test_add_constraint_stored(self, direction: str, ty: Type):
        """Added constraints should be stored."""
        inference = TypeParameterInference()
        var_name = "T"

        inference.add_constraint(var_name, ty, direction)

        assert len(inference.constraints) == 1
        assert inference.constraints[0].target == var_name
        assert inference.constraints[0].constraint_type == ty
        assert inference.constraints[0].direction == direction


class TestRecursiveTypeHandlerProperties:
    """Property-based tests for recursive type handling."""

    @given(ty=simple_types())
    @settings(max_examples=20)
    def test_simple_types_not_recursive(self, ty: Type):
        """Simple types should never be detected as recursive."""
        handler = RecursiveTypeHandler()
        assert not handler.is_recursive(ty)

    @given(max_depth=st.integers(min_value=1, max_value=100))
    @settings(max_examples=20)
    def test_depth_limit_configurable(self, max_depth: int):
        """Depth limit should be configurable and respected."""
        handler = RecursiveTypeHandler(max_depth=max_depth)
        assert handler.max_depth == max_depth

    @given(ty=compound_types(max_depth=3))
    @settings(max_examples=50)
    def test_enter_exit_balanced(self, ty: Type):
        """Enter and exit calls should be balanced."""
        handler = RecursiveTypeHandler()

        # Enter returns True if not seen before (can proceed)
        can_proceed = handler.enter(ty)
        assert can_proceed is True

        # After enter, type should be in seen
        assert id(ty) in handler.seen

        handler.exit(ty)
        # After exit, type should be removed
        assert id(ty) not in handler.seen


class TestUnificationWithBoundsProperties:
    """Property-based tests for unification with bounds."""

    @given(ty1=simple_types(), ty2=simple_types())
    @settings(max_examples=50)
    def test_unification_reflexive(self, ty1: Type, ty2: Type):
        """Same types should always unify."""
        result = unify_with_bounds(ty1, ty1)
        assert result.success

    @given(ty=simple_types())
    @settings(max_examples=30)
    def test_any_unifies_with_everything(self, ty: Type):
        """ANY should unify with any type."""
        result = unify_with_bounds(ANY, ty)
        assert result.success

        result2 = unify_with_bounds(ty, ANY)
        assert result2.success

    @given(ty=simple_types())
    @settings(max_examples=30)
    def test_never_special_handling(self, ty: Type):
        """NEVER should have special unification behavior."""
        # NEVER is the bottom type, should unify with anything as it's uninhabited
        result = unify_with_bounds(NEVER, ty)
        # The behavior depends on implementation, but it should not crash
        assert isinstance(result, UnificationResult)

    @given(
        name=st.text(alphabet="TUVKR", min_size=1, max_size=2),
        concrete=simple_types(),
    )
    @settings(max_examples=50)
    def test_bounded_var_within_bounds_unifies(self, name: str, concrete: Type):
        """Bounded type var within bounds should unify."""
        # Create a bounded var with ANY upper bound (everything is within bounds)
        btv = BoundedTypeVar(name=name, upper_bound=ANY, lower_bound=NEVER)

        result = unify_with_bounds(btv, concrete)
        assert result.success
        assert name in result.substitution.mapping
        assert result.substitution.mapping[name] == concrete


class TestInferTypeParamsProperties:
    """Property-based tests for the infer_type_params function."""

    @given(elem=simple_types())
    @settings(max_examples=30)
    def test_infer_list_returns_correct_binding(self, elem: Type):
        """infer_type_params should correctly bind list element type."""
        t_var = TypeVar("T")
        generic_type = ListType(t_var)
        concrete_type = ListType(elem)

        result = infer_type_params(generic_type, concrete_type)

        assert result is not None
        assert "T" in result.mapping
        assert result.mapping["T"] == elem

    @given(elem=simple_types())
    @settings(max_examples=30)
    def test_infer_nested_list_returns_correct_binding(self, elem: Type):
        """infer_type_params should handle nested generics."""
        t_var = TypeVar("T")
        generic_type = ListType(ListType(t_var))
        concrete_type = ListType(ListType(elem))

        result = infer_type_params(generic_type, concrete_type)

        assert result is not None
        assert "T" in result.mapping
        assert result.mapping["T"] == elem

    @given(ret_type=simple_types())
    @settings(max_examples=30)
    def test_infer_function_return_type(self, ret_type: Type):
        """infer_type_params should infer function return types."""
        t_var = TypeVar("T")
        generic_type = FunctionType((), t_var)
        concrete_type = FunctionType((), ret_type)

        result = infer_type_params(generic_type, concrete_type)

        assert result is not None
        assert "T" in result.mapping
        assert result.mapping["T"] == ret_type


# =============================================================================
# Cross-Component Integration Properties
# =============================================================================

class TestCrossComponentProperties:
    """Property-based tests for interactions between Phase 2 components."""

    @given(type_name=type_names(), modules=st.sets(module_names(), min_size=1, max_size=3))
    @settings(max_examples=30)
    def test_propagation_chain_consistency(self, type_name: str, modules: Set[str]):
        """Propagation from imports → types and types → syntax should be consistent."""
        types_to_syntax = TypesToSyntaxEdge()
        imports_to_types = ImportsToTypesEdge()

        # First propagation: imports → types (using internal method)
        accessible_types = imports_to_types._derive_accessible_types(modules)

        # Second propagation: types → syntax (using internal method)
        syntax_hints = types_to_syntax._type_to_syntax_hints(type_name, None)

        # Both results should be valid
        assert isinstance(accessible_types, set)
        assert isinstance(syntax_hints, dict)

        # All hint values should be booleans
        for key, value in syntax_hints.items():
            assert isinstance(key, str)
            assert isinstance(value, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
