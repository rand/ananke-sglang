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
"""Propagation edges for cross-domain constraint flow.

Each edge defines how constraints flow from a source domain to a target domain.
Edges have:
- Source and target domain names
- A propagation function that computes the new target constraint
- A priority (lower = higher priority)

Standard edges implement common propagation patterns:
- syntax_to_types: Syntactic structure informs type expectations
- types_to_syntax: Type expectations restrict valid syntax
- types_to_imports: Type usage implies required imports
- imports_to_types: Available imports affect type environment
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.constraint import Constraint
from core.domain import GenerationContext


@dataclass
class PropagationEdge(ABC):
    """Base class for propagation edges.

    An edge connects a source domain to a target domain and defines
    how constraints propagate between them.

    Attributes:
        source: Name of the source domain
        target: Name of the target domain
        priority: Priority (lower = higher priority, default 100)
        enabled: Whether this edge is active
    """

    source: str
    target: str
    priority: int = 100
    enabled: bool = True

    @abstractmethod
    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Compute the new target constraint after propagation.

        Given the source domain's constraint and the target's current
        constraint, compute what the target's new constraint should be.

        The result should be at least as restrictive as target_constraint
        (monotonicity requirement).

        Args:
            source_constraint: Constraint from the source domain
            target_constraint: Current constraint of the target domain
            context: The generation context

        Returns:
            New constraint for the target domain
        """
        pass

    def __lt__(self, other: PropagationEdge) -> bool:
        """Compare by priority for sorting."""
        return self.priority < other.priority


@dataclass
class FunctionEdge(PropagationEdge):
    """Edge using a custom propagation function.

    This allows defining edge behavior with a simple callable.

    WARNING: MONOTONICITY REQUIREMENT
    ---------------------------------
    The propagate_fn MUST maintain monotonicity: the returned constraint
    must be at least as restrictive as target_constraint. Formally:

        target_constraint.meet(result) == result

    Violating monotonicity can cause:
    - Non-convergence of the propagation network
    - Incorrect constraint states
    - Unbounded iteration in the worklist algorithm

    Safe patterns that guarantee monotonicity:
    - Return target_constraint unchanged (identity)
    - Return target_constraint.meet(something) (refinement)
    - Return BOTTOM when detecting unsatisfiability

    Unsafe patterns that may violate monotonicity:
    - Returning a completely different constraint
    - Loosening restrictions based on source
    - Conditionally returning TOP

    The PropagationNetwork will log warnings but not prevent non-monotonic
    updates. It is the caller's responsibility to ensure monotonicity.

    Attributes:
        propagate_fn: Function to compute new target constraint
    """

    propagate_fn: Callable[
        [Constraint, Constraint, GenerationContext],
        Constraint
    ] = field(default=lambda s, t, c: t)

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Apply the custom propagation function."""
        return self.propagate_fn(source_constraint, target_constraint, context)


@dataclass
class IdentityEdge(PropagationEdge):
    """Edge that passes constraint unchanged.

    Useful for simple dependency tracking without constraint modification.
    """

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Return target constraint unchanged."""
        return target_constraint


@dataclass
class MeetEdge(PropagationEdge):
    """Edge that meets source and target constraints.

    The result is the meet (conjunction) of both constraints,
    making the target at least as restrictive as both.
    """

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Return meet of source and target."""
        return target_constraint.meet(source_constraint)


class SyntaxToTypesEdge(PropagationEdge):
    """Edge from syntax domain to types domain.

    Syntactic structure provides type expectations. For example:
    - Function call implies callable type at function position
    - Assignment implies type compatibility between LHS and RHS
    - Return statement implies function return type

    This edge reads syntactic context and refines type expectations.
    """

    def __init__(self, priority: int = 50):
        """Initialize syntax-to-types edge.

        Args:
            priority: Edge priority (default 50)
        """
        super().__init__(
            source="syntax",
            target="types",
            priority=priority,
        )

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Propagate syntactic context to type expectations.

        Analyzes the current syntax position to derive type expectations:
        - Function call position: expect callable
        - Assignment RHS: match LHS type
        - Return statement: match function return type
        - List literal: expect compatible element types

        Args:
            source_constraint: SyntaxConstraint with position info
            target_constraint: TypeConstraint to refine
            context: Generation context

        Returns:
            Refined type constraint
        """
        # Handle TOP/BOTTOM
        if source_constraint.is_bottom():
            return target_constraint
        if target_constraint.is_bottom():
            return target_constraint

        # Extract syntax context
        syntax_context = self._extract_syntax_context(source_constraint, context)
        if not syntax_context:
            return target_constraint

        # Derive type expectations from syntax
        type_hints = self._syntax_to_type_hints(syntax_context)
        if not type_hints:
            return target_constraint

        # Apply type hints to constraint
        if hasattr(target_constraint, "with_syntax_hints"):
            return target_constraint.with_syntax_hints(type_hints)

        # Fallback: use metadata annotation
        if hasattr(target_constraint, "with_metadata"):
            return target_constraint.with_metadata({
                "syntax_hints": type_hints,
                "syntax_context": syntax_context.get("position"),
            })

        return target_constraint

    def _extract_syntax_context(
        self,
        constraint: Constraint,
        context: GenerationContext,
    ) -> dict:
        """Extract syntax context information.

        Args:
            constraint: The syntax constraint
            context: Generation context

        Returns:
            Dictionary of syntax context info
        """
        ctx = {}

        # Check for position info from syntax constraint
        if hasattr(constraint, "position"):
            ctx["position"] = constraint.position
        if hasattr(constraint, "current_rule"):
            ctx["rule"] = constraint.current_rule
        if hasattr(constraint, "parent_rule"):
            ctx["parent_rule"] = constraint.parent_rule

        # Check for expression context
        if hasattr(constraint, "in_function_call"):
            ctx["in_call"] = constraint.in_function_call
        if hasattr(constraint, "in_assignment"):
            ctx["in_assignment"] = constraint.in_assignment
        if hasattr(constraint, "in_return"):
            ctx["in_return"] = constraint.in_return

        # Check for literal context
        if hasattr(constraint, "in_list_literal"):
            ctx["in_list"] = constraint.in_list_literal
        if hasattr(constraint, "in_dict_literal"):
            ctx["in_dict"] = constraint.in_dict_literal

        # Try to get position from context
        if hasattr(context, "position"):
            ctx.setdefault("position", context.position)
        if hasattr(context, "current_node"):
            ctx["node_type"] = type(context.current_node).__name__

        return ctx

    def _syntax_to_type_hints(self, syntax_ctx: dict) -> dict:
        """Convert syntax context to type hints.

        Args:
            syntax_ctx: Syntax context dictionary

        Returns:
            Dictionary of type hints for the TypeDomain
        """
        hints = {}

        # Function call position: expect callable
        if syntax_ctx.get("in_call"):
            hints["expect_callable"] = True
            hints["position"] = "function"

        # Assignment RHS: get LHS type if available
        if syntax_ctx.get("in_assignment"):
            hints["position"] = "assignment_rhs"
            # LHS type would come from context if available

        # Return statement: match function return type
        if syntax_ctx.get("in_return"):
            hints["position"] = "return_value"
            hints["expect_return_compatible"] = True

        # List literal: element context
        if syntax_ctx.get("in_list"):
            hints["position"] = "list_element"
            hints["in_collection"] = True

        # Dict literal: key/value context
        if syntax_ctx.get("in_dict"):
            hints["position"] = "dict_element"
            hints["in_collection"] = True

        # Grammar rule hints
        rule = syntax_ctx.get("rule", "")
        if "expr" in rule.lower():
            hints["expect_expression"] = True
        if "arg" in rule.lower():
            hints["expect_argument"] = True
        if "param" in rule.lower():
            hints["expect_parameter"] = True

        return hints


class TypesToSyntaxEdge(PropagationEdge):
    """Edge from types domain to syntax domain.

    Type expectations can restrict valid syntax. For example:
    - If expecting int, string literals may be invalid
    - If expecting callable, literals are invalid
    - If expecting List[T], only list syntax valid

    This edge reads type context and refines syntax grammar.
    """

    def __init__(self, priority: int = 50):
        """Initialize types-to-syntax edge.

        Args:
            priority: Edge priority (default 50)
        """
        super().__init__(
            source="types",
            target="syntax",
            priority=priority,
        )

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Propagate type expectations to syntax restrictions.

        Analyzes the expected type and restricts syntax accordingly:
        - Numeric types: block string literals
        - String types: block numeric literals (without conversion)
        - Callable types: block non-callable literals
        - List/Dict/Tuple: allow constructor syntax

        Args:
            source_constraint: TypeConstraint with expected type
            target_constraint: SyntaxConstraint to refine
            context: Generation context

        Returns:
            Refined syntax constraint
        """
        # Handle TOP/BOTTOM
        if source_constraint.is_bottom():
            return target_constraint
        if target_constraint.is_bottom():
            return target_constraint

        # Try to extract expected type from TypeConstraint
        expected_type = self._extract_expected_type(source_constraint)
        if expected_type is None:
            return target_constraint

        # Derive syntax restrictions from type
        syntax_hints = self._type_to_syntax_hints(expected_type, context)
        if not syntax_hints:
            return target_constraint

        # Apply hints to target constraint if it supports metadata
        if hasattr(target_constraint, "with_metadata"):
            return target_constraint.with_metadata({
                "type_hints": syntax_hints,
                "expected_type": str(expected_type),
            })

        return target_constraint

    def _extract_expected_type(self, constraint: Constraint) -> Optional[str]:
        """Extract the expected type from a type constraint.

        Args:
            constraint: The type constraint

        Returns:
            String representation of expected type, or None
        """
        # TypeConstraint has 'expected' field
        if hasattr(constraint, "expected"):
            expected = constraint.expected
            if expected is not None:
                return str(expected)

        # Check for expected_type field
        if hasattr(constraint, "expected_type"):
            return str(constraint.expected_type)

        return None

    def _type_to_syntax_hints(
        self,
        expected_type: str,
        context: GenerationContext,
    ) -> dict:
        """Convert expected type to syntax hints.

        Args:
            expected_type: String representation of expected type
            context: Generation context

        Returns:
            Dictionary of syntax hints
        """
        hints = {}
        type_lower = expected_type.lower()

        # Numeric types
        if type_lower in ("int", "integer", "i32", "i64", "isize"):
            hints["allow_numeric_literals"] = True
            hints["block_string_literals"] = True
            hints["prefer_numeric_context"] = True

        # Float types
        elif type_lower in ("float", "double", "f32", "f64"):
            hints["allow_numeric_literals"] = True
            hints["allow_float_literals"] = True
            hints["block_string_literals"] = True

        # String types
        elif type_lower in ("str", "string", "&str"):
            hints["allow_string_literals"] = True
            hints["block_numeric_literals"] = True
            hints["prefer_string_context"] = True

        # Boolean types
        elif type_lower in ("bool", "boolean"):
            hints["allow_boolean_literals"] = True
            hints["prefer_boolean_context"] = True

        # Callable types
        elif "callable" in type_lower or "fn" in type_lower or "function" in type_lower:
            hints["block_non_callable_literals"] = True
            hints["allow_lambda"] = True
            hints["allow_function_refs"] = True

        # Collection types
        elif "list" in type_lower or "vec" in type_lower:
            hints["allow_list_literal"] = True
            hints["prefer_collection_context"] = True

        elif "dict" in type_lower or "map" in type_lower:
            hints["allow_dict_literal"] = True
            hints["prefer_collection_context"] = True

        return hints


class TypesToImportsEdge(PropagationEdge):
    """Edge from types domain to imports domain.

    Type usage implies required imports. For example:
    - Using List[T] requires 'from typing import List'
    - Using numpy.ndarray requires 'import numpy'
    - Using custom class requires its import

    This edge reads type usage and derives import requirements.
    """

    def __init__(self, priority: int = 75):
        """Initialize types-to-imports edge.

        Args:
            priority: Edge priority (default 75)
        """
        super().__init__(
            source="types",
            target="imports",
            priority=priority,
        )

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Propagate type usage to import requirements.

        Currently returns target unchanged - full implementation would
        analyze types and derive required imports.
        """
        if source_constraint.is_bottom():
            return target_constraint
        if target_constraint.is_bottom():
            return target_constraint

        return target_constraint


class ImportsToTypesEdge(PropagationEdge):
    """Edge from imports domain to types domain.

    Available imports affect the type environment. For example:
    - Imported modules provide type bindings
    - Import errors can make types unavailable
    - Version constraints affect available types

    This edge reads import state and updates type environment.
    """

    # Common typing module exports
    TYPING_TYPES = frozenset({
        "List", "Dict", "Set", "Tuple", "Optional", "Union", "Any",
        "Callable", "Type", "TypeVar", "Generic", "Protocol",
        "Sequence", "Mapping", "Iterable", "Iterator", "Generator",
        "Awaitable", "Coroutine", "AsyncIterator", "AsyncGenerator",
        "ClassVar", "Final", "Literal", "TypedDict",
    })

    # Module to types mapping
    MODULE_TYPES = {
        "typing": TYPING_TYPES,
        "collections.abc": frozenset({
            "Sequence", "Mapping", "MutableMapping", "Set", "MutableSet",
            "Iterable", "Iterator", "Callable", "Hashable", "Sized",
        }),
        "numpy": frozenset({"ndarray", "dtype", "int32", "int64", "float32", "float64"}),
        "pandas": frozenset({"DataFrame", "Series", "Index"}),
        "torch": frozenset({"Tensor", "nn", "optim"}),
    }

    def __init__(self, priority: int = 25):
        """Initialize imports-to-types edge.

        Args:
            priority: Edge priority (default 25)
        """
        super().__init__(
            source="imports",
            target="types",
            priority=priority,
        )

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Propagate import availability to type environment.

        Extracts available modules from ImportConstraint and derives
        which types are accessible in the type environment.

        Args:
            source_constraint: ImportConstraint with available modules
            target_constraint: TypeConstraint to update
            context: Generation context

        Returns:
            Updated type constraint with available types info
        """
        if source_constraint.is_bottom():
            return target_constraint
        if target_constraint.is_bottom():
            return target_constraint

        # Extract available modules from import constraint
        available_modules = self._extract_available_modules(source_constraint)
        if not available_modules:
            return target_constraint

        # Determine accessible types
        accessible_types = self._derive_accessible_types(available_modules)

        # Update type constraint with import context
        if hasattr(target_constraint, "with_import_context"):
            return target_constraint.with_import_context(
                available_modules=available_modules,
                accessible_types=accessible_types,
            )

        # Fallback: annotate with metadata
        if hasattr(target_constraint, "with_metadata"):
            return target_constraint.with_metadata({
                "import_modules": list(available_modules),
                "accessible_types": list(accessible_types),
            })

        return target_constraint

    def _extract_available_modules(self, constraint: Constraint) -> set:
        """Extract available module names from an import constraint.

        Args:
            constraint: The import constraint

        Returns:
            Set of available module names
        """
        available = set()

        # ImportConstraint has 'available' field with ModuleSpec objects
        if hasattr(constraint, "available"):
            for spec in constraint.available:
                if hasattr(spec, "name"):
                    available.add(spec.name)
                elif isinstance(spec, str):
                    available.add(spec)

        # Also check required modules (they should be available)
        if hasattr(constraint, "required"):
            for spec in constraint.required:
                if hasattr(spec, "name"):
                    available.add(spec.name)
                elif isinstance(spec, str):
                    available.add(spec)

        return available

    def _derive_accessible_types(self, available_modules: set) -> set:
        """Derive accessible types from available modules.

        Args:
            available_modules: Set of available module names

        Returns:
            Set of accessible type names
        """
        accessible = set()

        for module in available_modules:
            # Check exact matches
            if module in self.MODULE_TYPES:
                accessible.update(self.MODULE_TYPES[module])

            # Check prefix matches (e.g., "typing.List" imports)
            for known_module, types in self.MODULE_TYPES.items():
                if module.startswith(known_module):
                    accessible.update(types)

        return accessible


class ControlFlowToSemanticsEdge(PropagationEdge):
    """Edge from control flow domain to semantics domain.

    Control flow affects semantic constraints. For example:
    - Unreachable code has no semantic effect
    - Loop invariants must hold on each iteration
    - Branches create conditional semantic constraints

    This edge reads CFG and derives semantic conditions.
    """

    def __init__(self, priority: int = 100):
        """Initialize control-flow-to-semantics edge.

        Args:
            priority: Edge priority (default 100)
        """
        super().__init__(
            source="controlflow",
            target="semantics",
            priority=priority,
        )

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Propagate control flow to semantic constraints.

        Analyzes the CFG constraint to determine:
        - Reachability: unreachable code relaxes semantic constraints
        - Loop context: derives loop invariants
        - Branch context: conditional constraints

        Args:
            source_constraint: ControlFlowConstraint with CFG state
            target_constraint: SemanticConstraint to update
            context: Generation context

        Returns:
            Updated semantic constraint
        """
        if source_constraint.is_bottom():
            return target_constraint
        if target_constraint.is_bottom():
            return target_constraint

        # Extract control flow information
        cf_info = self._extract_cf_info(source_constraint)
        if not cf_info:
            return target_constraint

        # Handle unreachable code
        if cf_info.get("unreachable"):
            # Unreachable code: semantic constraints are vacuously satisfied
            if hasattr(target_constraint, "relax_for_unreachable"):
                return target_constraint.relax_for_unreachable()

        # Handle loop context
        if cf_info.get("in_loop"):
            # In loop: may need to track loop iterations for termination
            if hasattr(target_constraint, "with_loop_context"):
                return target_constraint.with_loop_context(
                    loop_depth=cf_info.get("loop_depth", 1),
                    may_break=cf_info.get("may_break", False),
                    may_continue=cf_info.get("may_continue", False),
                )

        # Handle branch context
        if cf_info.get("in_branch"):
            # In conditional: constraints may be conditional
            if hasattr(target_constraint, "with_condition"):
                return target_constraint.with_condition(
                    condition=cf_info.get("branch_condition"),
                    is_true_branch=cf_info.get("is_true_branch", True),
                )

        # Annotate with metadata as fallback
        if hasattr(target_constraint, "with_metadata") and cf_info:
            return target_constraint.with_metadata({
                "controlflow": cf_info,
            })

        return target_constraint

    def _extract_cf_info(self, constraint: Constraint) -> dict:
        """Extract control flow information from constraint.

        Args:
            constraint: The control flow constraint

        Returns:
            Dictionary of control flow information
        """
        info = {}

        # Check for reachability
        if hasattr(constraint, "is_reachable"):
            info["unreachable"] = not constraint.is_reachable
        elif hasattr(constraint, "reachable"):
            info["unreachable"] = not constraint.reachable

        # Check for loop context
        if hasattr(constraint, "loop_depth"):
            info["in_loop"] = constraint.loop_depth > 0
            info["loop_depth"] = constraint.loop_depth
        if hasattr(constraint, "in_loop"):
            info["in_loop"] = constraint.in_loop

        # Check for branch context
        if hasattr(constraint, "branch_depth"):
            info["in_branch"] = constraint.branch_depth > 0
        if hasattr(constraint, "in_branch"):
            info["in_branch"] = constraint.in_branch
        if hasattr(constraint, "branch_condition"):
            info["branch_condition"] = constraint.branch_condition
        if hasattr(constraint, "is_true_branch"):
            info["is_true_branch"] = constraint.is_true_branch

        # Check for return/break/continue
        if hasattr(constraint, "after_return"):
            info["after_return"] = constraint.after_return
        if hasattr(constraint, "may_break"):
            info["may_break"] = constraint.may_break
        if hasattr(constraint, "may_continue"):
            info["may_continue"] = constraint.may_continue

        return info


def create_standard_edges() -> list[PropagationEdge]:
    """Create the standard set of propagation edges.

    Returns a list of edges implementing common propagation patterns:
    - syntax <-> types
    - types <-> imports
    - controlflow -> semantics

    Returns:
        List of standard propagation edges
    """
    return [
        SyntaxToTypesEdge(),
        TypesToSyntaxEdge(),
        TypesToImportsEdge(),
        ImportsToTypesEdge(),
        ControlFlowToSemanticsEdge(),
    ]
