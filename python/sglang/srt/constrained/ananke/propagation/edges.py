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
from typing import Callable, Dict, Optional, Set

try:
    from core.constraint import Constraint
    from core.domain import GenerationContext
except ImportError:
    from ..core.constraint import Constraint
    from ..core.domain import GenerationContext


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
    - List literal implies sequence type
    - Dict literal implies mapping type

    This edge reads syntactic context and refines type expectations.
    """

    # Patterns that suggest callable types
    _CALLABLE_PATTERNS = frozenset([
        "(",   # Function call: foo(
        "def", # Function definition
        "lambda",
    ])

    # Patterns that suggest sequence types
    _SEQUENCE_PATTERNS = frozenset([
        "[",   # List/subscript
        "for", # For loop (iterator expected)
        "in",  # Membership test
    ])

    # Patterns that suggest mapping types
    _MAPPING_PATTERNS = frozenset([
        "{",   # Dict literal
        ":",   # Dict key-value separator
    ])

    # Patterns that suggest boolean types
    _BOOLEAN_PATTERNS = frozenset([
        "if",
        "elif",
        "while",
        "and",
        "or",
        "not",
    ])

    # Patterns that suggest numeric types
    _NUMERIC_PATTERNS = frozenset([
        "+", "-", "*", "/", "//", "%", "**",
        "<", ">", "<=", ">=",
    ])

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

        Analyzes the generated text to infer type expectations based
        on syntactic patterns.

        Args:
            source_constraint: SyntaxConstraint from syntax domain
            target_constraint: TypeConstraint from types domain
            context: Generation context

        Returns:
            Updated TypeConstraint with inferred type expectations
        """
        if source_constraint.is_bottom():
            return target_constraint
        if target_constraint.is_bottom():
            return target_constraint

        # Skip if target already has a concrete expected type
        existing_expected = getattr(target_constraint, "expected_type", None)
        if existing_expected is not None:
            # Only skip if it's not Any
            type_name = type(existing_expected).__name__
            if type_name not in ("AnyType", "type"):
                return target_constraint

        try:
            # Analyze generation context for syntactic patterns
            text = getattr(context, "generated_text", "")
            if not isinstance(text, str) or not text:
                return target_constraint

            # Get the last few tokens for pattern matching
            # Use a simple split - full implementation would use tokenization
            last_tokens = self._get_last_tokens(text, count=3)
            if not last_tokens:
                return target_constraint

            # Detect type expectations based on patterns
            inferred_type_name = self._infer_type_from_patterns(last_tokens, text)
            if inferred_type_name is None:
                return target_constraint

            # Update environment hash to signal type context change
            # The actual type binding happens in the types domain
            # Use abs() to ensure non-negative hash (required for semilattice max() semantics)
            current_hash = getattr(target_constraint, "environment_hash", 0)
            pattern_hash = abs(hash((inferred_type_name, tuple(last_tokens))))
            new_hash = abs(hash((current_hash, pattern_hash)))

            if new_hash == current_hash:
                return target_constraint

            # Return updated type constraint
            if hasattr(target_constraint, "with_environment_hash"):
                return target_constraint.with_environment_hash(new_hash)

            return target_constraint

        except (AttributeError, TypeError):
            return target_constraint

    def _get_last_tokens(self, text: str, count: int = 3) -> list[str]:
        """Extract the last N meaningful tokens from text.

        Args:
            text: Generated text
            count: Number of tokens to extract

        Returns:
            List of last tokens
        """
        # Simple whitespace split - full implementation would use proper tokenization
        tokens = text.split()
        if not tokens:
            return []

        # Take last N tokens, also consider operators attached to words
        result = []
        for token in tokens[-count*2:]:
            # Split on common operators
            for op in ["(", ")", "[", "]", "{", "}", ":", ",", "=", "."]:
                if op in token and len(token) > 1:
                    parts = token.split(op)
                    for p in parts:
                        if p:
                            result.append(p)
                        result.append(op)
                    break
            else:
                result.append(token)

        return result[-count:]

    def _infer_type_from_patterns(
        self, last_tokens: list[str], full_text: str
    ) -> Optional[str]:
        """Infer expected type from syntactic patterns.

        Args:
            last_tokens: Recent tokens
            full_text: Full generated text

        Returns:
            Inferred type name or None
        """
        for token in last_tokens:
            # Check for callable patterns
            if token in self._CALLABLE_PATTERNS:
                return "Callable"

            # Check for sequence patterns
            if token in self._SEQUENCE_PATTERNS:
                return "Sequence"

            # Check for mapping patterns
            if token in self._MAPPING_PATTERNS:
                return "Mapping"

            # Check for boolean patterns
            if token in self._BOOLEAN_PATTERNS:
                return "bool"

            # Check for numeric patterns
            if token in self._NUMERIC_PATTERNS:
                # Context-dependent: if we're after an operator, expect numeric
                return "numeric"

        # Check for assignment context (= at the end)
        if last_tokens and last_tokens[-1] == "=":
            # After assignment, any type is possible
            return None

        # Check for return context
        if "return" in last_tokens:
            return "ReturnType"

        return None


class TypesToSyntaxEdge(PropagationEdge):
    """Edge from types domain to syntax domain.

    Type expectations can restrict valid syntax. For example:
    - If expecting int, string literals may be invalid
    - If expecting callable, literals are invalid
    - If expecting List[T], only list syntax valid

    This edge reads type context and refines syntax grammar.
    """

    # Type names that suggest specific syntax restrictions
    _CALLABLE_TYPES = frozenset([
        "FunctionType", "CallableType", "Callable", "MethodType",
    ])

    _SEQUENCE_TYPES = frozenset([
        "ListType", "TupleType", "List", "Tuple", "Sequence",
    ])

    _MAPPING_TYPES = frozenset([
        "DictType", "Dict", "Mapping", "MutableMapping",
    ])

    _NUMERIC_TYPES = frozenset([
        "IntType", "FloatType", "int", "float", "Int", "Float",
        "NumberType", "Complex",
    ])

    _STRING_TYPES = frozenset([
        "StringType", "str", "String", "StrType",
    ])

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

        Analyzes type expectations to inform syntax-level constraints.
        For example, if a callable type is expected, this can be used
        to prioritize function call or lambda syntax.

        Args:
            source_constraint: TypeConstraint from types domain
            target_constraint: SyntaxConstraint from syntax domain
            context: Generation context

        Returns:
            Updated SyntaxConstraint with type-derived hints
        """
        # Handle TOP/BOTTOM
        if source_constraint.is_bottom():
            return target_constraint
        if target_constraint.is_bottom():
            return target_constraint

        try:
            # Get expected type from source constraint
            expected_type = getattr(source_constraint, "expected_type", None)
            if expected_type is None:
                return target_constraint

            # Determine syntax category from type
            type_name = type(expected_type).__name__
            syntax_hint = self._type_to_syntax_hint(type_name)

            if syntax_hint is None:
                return target_constraint

            # Update target constraint's metadata to include hint
            # This is used by the syntax domain to prioritize certain productions
            if hasattr(target_constraint, "with_hint"):
                return target_constraint.with_hint(syntax_hint)

            # Alternative: update via metadata if available
            if hasattr(target_constraint, "metadata"):
                metadata = dict(target_constraint.metadata or {})
                metadata["type_hint"] = syntax_hint
                if hasattr(target_constraint, "with_metadata"):
                    return target_constraint.with_metadata(metadata)

            return target_constraint

        except (AttributeError, TypeError):
            return target_constraint

    def _type_to_syntax_hint(self, type_name: str) -> Optional[str]:
        """Convert type name to syntax production hint.

        Args:
            type_name: Name of the expected type

        Returns:
            Syntax hint string or None
        """
        if type_name in self._CALLABLE_TYPES:
            return "callable"
        elif type_name in self._SEQUENCE_TYPES:
            return "sequence"
        elif type_name in self._MAPPING_TYPES:
            return "mapping"
        elif type_name in self._NUMERIC_TYPES:
            return "numeric"
        elif type_name in self._STRING_TYPES:
            return "string"
        elif type_name == "BoolType" or type_name == "bool":
            return "boolean"
        return None


class TypesToImportsEdge(PropagationEdge):
    """Edge from types domain to imports domain.

    Type usage implies required imports. For example:
    - Using List[T] requires 'from typing import List'
    - Using numpy.ndarray requires 'import numpy'
    - Using custom class requires its import

    This edge reads type usage and derives import requirements.
    """

    # Map type names to required module imports
    _TYPE_TO_MODULE: Dict[str, str] = {
        # typing module types
        "List": "typing",
        "Dict": "typing",
        "Set": "typing",
        "Tuple": "typing",
        "Optional": "typing",
        "Union": "typing",
        "Callable": "typing",
        "Any": "typing",
        "Sequence": "typing",
        "Iterable": "typing",
        "Iterator": "typing",
        "Generator": "typing",
        "TypeVar": "typing",
        "Generic": "typing",
        "Protocol": "typing",
        # dataclass
        "dataclass": "dataclasses",
        # collections
        "deque": "collections",
        "defaultdict": "collections",
        "Counter": "collections",
        # enum
        "Enum": "enum",
        "IntEnum": "enum",
        # abc
        "ABC": "abc",
        "abstractmethod": "abc",
        # pathlib
        "Path": "pathlib",
        # datetime
        "datetime": "datetime",
        "date": "datetime",
        "timedelta": "datetime",
    }

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

        Analyzes types used in the source constraint and infers
        what modules need to be imported for those types.

        Args:
            source_constraint: TypeConstraint from types domain
            target_constraint: ImportConstraint from imports domain
            context: Generation context

        Returns:
            Updated ImportConstraint with required module info
        """
        if source_constraint.is_bottom():
            return target_constraint
        if target_constraint.is_bottom():
            return target_constraint

        try:
            # Get types referenced in the constraint
            # Look for expected_type and environment bindings
            referenced_types = set()

            expected_type = getattr(source_constraint, "expected_type", None)
            if expected_type is not None:
                type_name = type(expected_type).__name__
                if type_name in self._TYPE_TO_MODULE:
                    referenced_types.add(type_name)

            # Also check type environment if available
            env = getattr(source_constraint, "environment", None)
            if env is not None and hasattr(env, "bindings"):
                for name, ty in getattr(env, "bindings", {}).items():
                    if ty is not None:
                        type_name = type(ty).__name__
                        if type_name in self._TYPE_TO_MODULE:
                            referenced_types.add(type_name)

            if not referenced_types:
                return target_constraint

            # Find modules needed for these types
            required_modules = set()
            for type_name in referenced_types:
                module = self._TYPE_TO_MODULE.get(type_name)
                if module:
                    required_modules.add(module)

            if not required_modules:
                return target_constraint

            # Update import constraint if it has a requires method
            # This is informational - actual import tracking is in ImportDomain
            if hasattr(target_constraint, "metadata"):
                metadata = dict(target_constraint.metadata or {})
                existing = metadata.get("suggested_imports", set())
                metadata["suggested_imports"] = existing | required_modules
                if hasattr(target_constraint, "with_metadata"):
                    return target_constraint.with_metadata(metadata)

            return target_constraint

        except (AttributeError, TypeError):
            return target_constraint

    def get_required_module(self, type_name: str) -> Optional[str]:
        """Get the module required for a given type.

        Args:
            type_name: Name of the type

        Returns:
            Module name or None
        """
        return self._TYPE_TO_MODULE.get(type_name)


class ImportsToTypesEdge(PropagationEdge):
    """Edge from imports domain to types domain.

    Available imports affect the type environment. For example:
    - Imported modules provide type bindings
    - Import errors can make types unavailable
    - Version constraints affect available types

    This edge reads import state and updates type environment.
    """

    # Type stubs for common Python modules
    # Maps module name -> set of exported type names
    _MODULE_TYPE_STUBS: Dict[str, Set[str]] = {
        # typing module - core type annotations
        "typing": {
            "List", "Dict", "Set", "Tuple", "Optional", "Union", "Any",
            "Callable", "Sequence", "Iterable", "Iterator", "Generator",
            "TypeVar", "Generic", "Protocol", "Final", "Literal",
            "ClassVar", "cast", "overload", "TYPE_CHECKING",
        },
        "typing_extensions": {
            "TypedDict", "Protocol", "Final", "Literal", "Self",
            "ParamSpec", "Concatenate", "TypeAlias", "NotRequired",
        },
        # collections - data structures
        "collections": {
            "deque", "defaultdict", "OrderedDict", "Counter", "ChainMap",
            "namedtuple",
        },
        "collections.abc": {
            "Iterable", "Iterator", "Sequence", "MutableSequence",
            "Set", "MutableSet", "Mapping", "MutableMapping",
            "Callable", "Hashable", "Awaitable", "Coroutine",
        },
        # dataclasses
        "dataclasses": {"dataclass", "field", "Field", "FrozenInstanceError"},
        # enum
        "enum": {"Enum", "IntEnum", "Flag", "IntFlag", "auto"},
        # abc
        "abc": {"ABC", "ABCMeta", "abstractmethod", "abstractproperty"},
        # functools
        "functools": {
            "wraps", "partial", "reduce", "lru_cache", "cache",
            "cached_property", "total_ordering",
        },
        # pathlib
        "pathlib": {"Path", "PurePath", "PurePosixPath", "PureWindowsPath"},
        # datetime
        "datetime": {"datetime", "date", "time", "timedelta", "timezone"},
        # json
        "json": {"dumps", "loads", "dump", "load", "JSONEncoder", "JSONDecoder"},
        # os
        "os": {"path", "environ", "getcwd", "listdir", "makedirs", "remove"},
        "os.path": {"join", "exists", "isfile", "isdir", "dirname", "basename"},
        # sys
        "sys": {"argv", "exit", "stdin", "stdout", "stderr", "path", "modules"},
        # re
        "re": {"compile", "match", "search", "findall", "sub", "Pattern", "Match"},
        # math
        "math": {"sqrt", "sin", "cos", "tan", "pi", "e", "log", "exp", "floor", "ceil"},
        # random
        "random": {"random", "randint", "choice", "shuffle", "sample", "seed"},
        # itertools
        "itertools": {
            "chain", "combinations", "permutations", "product", "repeat",
            "islice", "takewhile", "dropwhile", "groupby", "count", "cycle",
        },
        # contextlib
        "contextlib": {
            "contextmanager", "asynccontextmanager", "suppress",
            "redirect_stdout", "redirect_stderr", "ExitStack",
        },
        # io
        "io": {"StringIO", "BytesIO", "TextIOWrapper", "BufferedReader"},
        # logging
        "logging": {
            "Logger", "Handler", "Formatter", "getLogger", "basicConfig",
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
        },
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

        When modules are imported, this edge updates the type constraint's
        environment hash to indicate that new bindings are available.
        The actual bindings are managed by the TypeDomain based on context.

        Args:
            source_constraint: ImportConstraint from imports domain
            target_constraint: TypeConstraint from types domain
            context: Generation context

        Returns:
            Updated TypeConstraint with new environment hash if imports changed
        """
        if source_constraint.is_bottom():
            return target_constraint
        if target_constraint.is_bottom():
            return target_constraint

        # Try to access ImportConstraint.available
        try:
            available = getattr(source_constraint, "available", None)
            if available is None or not available:
                return target_constraint

            # Compute a hash based on available imports
            # This signals to the types domain that new bindings may be available
            import_names = frozenset(m.name for m in available)

            # Count how many type stubs we have for available imports
            type_binding_count = 0
            for module_name in import_names:
                if module_name in self._MODULE_TYPE_STUBS:
                    type_binding_count += len(self._MODULE_TYPE_STUBS[module_name])

            if type_binding_count == 0:
                return target_constraint

            # Create updated environment hash that incorporates import info
            # Use abs() to ensure non-negative hash (required for semilattice max() semantics)
            current_hash = getattr(target_constraint, "environment_hash", 0)
            import_hash = abs(hash(import_names))
            new_hash = abs(hash((current_hash, import_hash, type_binding_count)))

            # Only update if hash actually changed
            if new_hash == current_hash:
                return target_constraint

            # Return updated type constraint with new environment hash
            if hasattr(target_constraint, "with_environment_hash"):
                return target_constraint.with_environment_hash(new_hash)

            return target_constraint

        except (AttributeError, TypeError):
            # Source constraint doesn't have expected attributes
            return target_constraint

    def get_type_exports(self, module_name: str) -> Set[str]:
        """Get known type exports for a module.

        Args:
            module_name: Name of the module

        Returns:
            Set of exported type/function names
        """
        return self._MODULE_TYPE_STUBS.get(module_name, set())


class TypesToControlFlowEdge(PropagationEdge):
    """Edge from types domain to control flow domain.

    Type information affects control flow constraints. For example:
    - A function with return type T (non-void) must have return statements
    - A function with return type Never cannot have normal returns
    - Return type expectations inform termination requirements

    This edge reads type context and updates control flow requirements.
    """

    def __init__(self, priority: int = 50):
        """Initialize types-to-controlflow edge.

        Args:
            priority: Edge priority (default 50)
        """
        super().__init__(
            source="types",
            target="controlflow",
            priority=priority,
        )

    def propagate(
        self,
        source_constraint: Constraint,
        target_constraint: Constraint,
        context: GenerationContext,
    ) -> Constraint:
        """Propagate type information to control flow requirements.

        When a function has a non-void return type, this edge signals
        that the function must terminate and return a value.

        Args:
            source_constraint: TypeConstraint from types domain
            target_constraint: ControlFlowConstraint from controlflow domain
            context: Generation context

        Returns:
            Updated ControlFlowConstraint with termination requirements
        """
        if source_constraint.is_bottom():
            return target_constraint
        if target_constraint.is_bottom():
            return target_constraint

        try:
            # Get expected type from TypeConstraint
            expected_type = getattr(source_constraint, "expected_type", None)
            if expected_type is None:
                return target_constraint

            # Check if we're in a function context (context may have function info)
            in_function_context = self._is_function_context(context)
            if not in_function_context:
                return target_constraint

            # Check the type class name to determine termination requirements
            type_name = type(expected_type).__name__

            # If return type is Never (noreturn), allow non-termination
            if type_name == "NeverType":
                if hasattr(target_constraint, "allow_non_termination"):
                    return target_constraint.allow_non_termination()
                return target_constraint

            # If we have a concrete expected type (not Any, not None for void),
            # then the function should terminate with a return
            if type_name not in ("AnyType", "NoneType") and expected_type is not None:
                if hasattr(target_constraint, "termination"):
                    # Only set termination requirement if currently unknown
                    from_termination = getattr(target_constraint, "termination", None)
                    if from_termination is not None:
                        termination_name = getattr(from_termination, "name", "")
                        if termination_name == "UNKNOWN":
                            if hasattr(target_constraint, "require_termination"):
                                return target_constraint.require_termination()

            return target_constraint

        except (AttributeError, TypeError):
            # Source constraint doesn't have expected attributes
            return target_constraint

    def _is_function_context(self, context: GenerationContext) -> bool:
        """Check if we're inside a function context.

        Args:
            context: Generation context

        Returns:
            True if we're generating inside a function
        """
        if context is None:
            return False

        # Check if context has function-related metadata
        metadata = getattr(context, "metadata", {})
        if isinstance(metadata, dict):
            if metadata.get("in_function", False):
                return True
            if metadata.get("function_name"):
                return True

        # Check generated text for function definition markers
        text = getattr(context, "generated_text", "")
        if isinstance(text, str):
            # Simple heuristic: check for def/fn patterns
            if "def " in text or "fn " in text or "function " in text:
                return True

        return False


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

        Analyzes the control flow graph to determine semantic implications.
        For example, unreachable code paths should not contribute semantic
        constraints, and loop conditions may establish invariants.

        Args:
            source_constraint: ControlFlowConstraint from controlflow domain
            target_constraint: SemanticConstraint from semantics domain
            context: Generation context

        Returns:
            Updated SemanticConstraint with control-flow-derived information
        """
        if source_constraint.is_bottom():
            return target_constraint
        if target_constraint.is_bottom():
            return target_constraint

        try:
            # Check for dead code detection from CFG
            # Dead code should not contribute semantic constraints
            dead_blocks = getattr(source_constraint, "dead_blocks", None)
            if dead_blocks and hasattr(target_constraint, "mark_unreachable"):
                # If we're in a dead block, mark semantics as unreachable
                current_block = getattr(context, "current_block", None)
                if current_block and current_block in dead_blocks:
                    return target_constraint.mark_unreachable()

            # Check for loop context - may affect semantic reasoning mode
            in_loop = getattr(source_constraint, "in_loop", False)
            if in_loop and hasattr(target_constraint, "set_loop_context"):
                target_constraint = target_constraint.set_loop_context(True)

            # Check for conditional branches - may split semantic reasoning
            branch_condition = getattr(source_constraint, "current_condition", None)
            if branch_condition and hasattr(target_constraint, "add_path_condition"):
                target_constraint = target_constraint.add_path_condition(branch_condition)

            # Check termination status - affects semantic validity
            must_terminate = getattr(source_constraint, "must_terminate", None)
            if must_terminate and hasattr(target_constraint, "assume_termination"):
                target_constraint = target_constraint.assume_termination()

            return target_constraint

        except (AttributeError, TypeError):
            return target_constraint


def create_standard_edges() -> list[PropagationEdge]:
    """Create the standard set of propagation edges.

    Returns a list of edges implementing common propagation patterns:
    - syntax <-> types
    - types <-> imports
    - types -> controlflow
    - controlflow -> semantics

    Returns:
        List of standard propagation edges
    """
    return [
        SyntaxToTypesEdge(),
        TypesToSyntaxEdge(),
        TypesToImportsEdge(),
        ImportsToTypesEdge(),
        TypesToControlFlowEdge(),
        ControlFlowToSemanticsEdge(),
    ]
