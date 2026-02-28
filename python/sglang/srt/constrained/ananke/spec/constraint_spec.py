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
"""Rich constraint specification dataclasses for Ananke.

This module defines the core data structures for passing constraint context
to the Ananke constrained generation system, including:

- ConstraintSpec: The unified internal representation
- Type context structures (TypeBinding, FunctionSignature, ClassDefinition)
- Import context structures (ImportBinding, ModuleStub)
- Control flow context (ControlFlowContext)
- Semantic constraints (SemanticConstraint)
- Language support (LanguageFrame, LanguageDetection)
- Cache control (CacheScope)

Design Principles:
1. Immutable where possible (frozen dataclasses)
2. Hashable for caching
3. Serializable to/from JSON and binary formats
4. Backward compatible with legacy constraint APIs
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from ..adaptive.intensity import ConstraintIntensity


# =============================================================================
# Enums
# =============================================================================


class LanguageDetection(Enum):
    """Language detection strategy for constraint specification.

    AUTO: Tree-sitter based detection from generated text (default)
    EXPLICIT: Use the language field only, no detection
    STACK: Use language_stack for polyglot generation
    """

    AUTO = "auto"
    EXPLICIT = "explicit"
    STACK = "stack"

    def __str__(self) -> str:
        return self.value


class CacheScope(Enum):
    """What context is included in cache key - ordered by reuse potential.

    SYNTAX_ONLY: Maximum cache reuse. Key = (constraint_type, constraint_content).
        Use when type context doesn't affect syntax grammar.

    SYNTAX_AND_LANG: Include language in cache key.
        Use when same constraint is used with different languages.

    FULL_CONTEXT: Include all context in cache key.
        Use when context affects grammar adaptation (e.g., custom types).
    """

    SYNTAX_ONLY = "syntax_only"
    SYNTAX_AND_LANG = "syntax_and_lang"
    FULL_CONTEXT = "full_context"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, s: str) -> "CacheScope":
        """Parse CacheScope from string."""
        try:
            return cls(s)
        except ValueError:
            return cls.SYNTAX_ONLY


class ConstraintSource(Enum):
    """How the constraint specification was provided.

    INLINE: JSON dict in request body (constraint_spec field)
    URI: External reference (constraint_uri field)
    BINARY: Dense encoding from ananke tool (constraint_bytes field)
    LEGACY: From legacy parameters (json_schema, regex, ebnf, structural_tag)
    """

    INLINE = "inline"
    URI = "uri"
    BINARY = "binary"
    LEGACY = "legacy"

    def __str__(self) -> str:
        return self.value


# =============================================================================
# Language Support
# =============================================================================


@dataclass(frozen=True)
class LanguageFrame:
    """A frame in the language stack for polyglot generation.

    Used to track language context changes within a single generation,
    supporting scenarios like TypeScript containing Python code blocks.

    Attributes:
        language: Programming language identifier
        start_position: Character position where this frame begins
        delimiter: Token that triggered this language switch (optional)
        end_delimiter: Token that ends this language context (optional)

    Example:
        >>> # TypeScript with embedded Python
        >>> stack = [
        ...     LanguageFrame("typescript", 0),
        ...     LanguageFrame("python", 150, delimiter="'''", end_delimiter="'''"),
        ... ]
    """

    language: str
    start_position: int
    delimiter: Optional[str] = None
    end_delimiter: Optional[str] = None

    def contains(self, position: int) -> bool:
        """Check if a position is within this language frame."""
        return position >= self.start_position

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: Dict[str, Any] = {
            "language": self.language,
            "start_position": self.start_position,
        }
        if self.delimiter is not None:
            d["delimiter"] = self.delimiter
        if self.end_delimiter is not None:
            d["end_delimiter"] = self.end_delimiter
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LanguageFrame":
        """Create from dictionary."""
        return cls(
            language=d["language"],
            start_position=d["start_position"],
            delimiter=d.get("delimiter"),
            end_delimiter=d.get("end_delimiter"),
        )


# =============================================================================
# Type Context
# =============================================================================


@dataclass(frozen=True)
class TypeBinding:
    """Variable-to-type binding for type context.

    Represents a variable in scope with its type annotation, used to seed
    the TypeDomain with external type information.

    Attributes:
        name: Variable name
        type_expr: Type expression as string (e.g., "List[int]", "Dict[str, User]")
        scope: Scope identifier ("local", "parameter", "global", "class:ClassName")
        mutable: Whether the binding is mutable
        origin: Description of where this binding came from

    Example:
        >>> binding = TypeBinding(name="users", type_expr="List[User]", scope="local")
    """

    name: str
    type_expr: str
    scope: Optional[str] = None
    mutable: bool = True
    origin: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: Dict[str, Any] = {"name": self.name, "type_expr": self.type_expr}
        if self.scope is not None:
            d["scope"] = self.scope
        if not self.mutable:
            d["mutable"] = self.mutable
        if self.origin is not None:
            d["origin"] = self.origin
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TypeBinding":
        """Create from dictionary."""
        return cls(
            name=d["name"],
            type_expr=d["type_expr"],
            scope=d.get("scope"),
            mutable=d.get("mutable", True),
            origin=d.get("origin"),
        )


@dataclass(frozen=True)
class FunctionSignature:
    """Function type signature for type context.

    Represents a function's complete type signature including parameters,
    return type, and modifiers.

    Attributes:
        name: Function name
        params: Parameter bindings as tuple
        return_type: Return type expression
        type_params: Generic type parameters (e.g., ("T", "U"))
        decorators: Applied decorators
        is_async: Whether function is async
        is_generator: Whether function is a generator

    Example:
        >>> sig = FunctionSignature(
        ...     name="process",
        ...     params=(TypeBinding("x", "int"), TypeBinding("y", "str")),
        ...     return_type="bool",
        ...     is_async=True,
        ... )
    """

    name: str
    params: Tuple[TypeBinding, ...]
    return_type: str
    type_params: Tuple[str, ...] = ()
    decorators: Tuple[str, ...] = ()
    is_async: bool = False
    is_generator: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: Dict[str, Any] = {
            "name": self.name,
            "params": [p.to_dict() for p in self.params],
            "return_type": self.return_type,
        }
        if self.type_params:
            d["type_params"] = list(self.type_params)
        if self.decorators:
            d["decorators"] = list(self.decorators)
        if self.is_async:
            d["is_async"] = self.is_async
        if self.is_generator:
            d["is_generator"] = self.is_generator
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FunctionSignature":
        """Create from dictionary."""
        return cls(
            name=d["name"],
            params=tuple(TypeBinding.from_dict(p) for p in d.get("params", [])),
            return_type=d["return_type"],
            type_params=tuple(d.get("type_params", [])),
            decorators=tuple(d.get("decorators", [])),
            is_async=d.get("is_async", False),
            is_generator=d.get("is_generator", False),
        )


@dataclass(frozen=True)
class ClassDefinition:
    """Class definition for type context.

    Represents a class with its complete type structure including
    inheritance, methods, and attributes.

    Attributes:
        name: Class name
        bases: Base class names
        type_params: Generic type parameters
        methods: Method signatures
        class_vars: Class-level variable bindings
        instance_vars: Instance variable bindings

    Example:
        >>> cls_def = ClassDefinition(
        ...     name="User",
        ...     bases=("BaseModel",),
        ...     instance_vars=(
        ...         TypeBinding("id", "int"),
        ...         TypeBinding("name", "str"),
        ...     ),
        ... )
    """

    name: str
    bases: Tuple[str, ...] = ()
    type_params: Tuple[str, ...] = ()
    methods: Tuple[FunctionSignature, ...] = ()
    class_vars: Tuple[TypeBinding, ...] = ()
    instance_vars: Tuple[TypeBinding, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: Dict[str, Any] = {"name": self.name}
        if self.bases:
            d["bases"] = list(self.bases)
        if self.type_params:
            d["type_params"] = list(self.type_params)
        if self.methods:
            d["methods"] = [m.to_dict() for m in self.methods]
        if self.class_vars:
            d["class_vars"] = [v.to_dict() for v in self.class_vars]
        if self.instance_vars:
            d["instance_vars"] = [v.to_dict() for v in self.instance_vars]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClassDefinition":
        """Create from dictionary."""
        return cls(
            name=d["name"],
            bases=tuple(d.get("bases", [])),
            type_params=tuple(d.get("type_params", [])),
            methods=tuple(
                FunctionSignature.from_dict(m) for m in d.get("methods", [])
            ),
            class_vars=tuple(
                TypeBinding.from_dict(v) for v in d.get("class_vars", [])
            ),
            instance_vars=tuple(
                TypeBinding.from_dict(v) for v in d.get("instance_vars", [])
            ),
        )


# =============================================================================
# Import Context
# =============================================================================


@dataclass(frozen=True)
class ImportBinding:
    """Import statement representation.

    Represents a single import for seeding the ImportDomain.

    Attributes:
        module: Module path (e.g., "typing", "collections.abc")
        name: Imported name for 'from x import name'
        alias: Alias for 'import x as alias' or 'from x import y as alias'
        is_wildcard: True for 'from x import *'

    Examples:
        >>> ImportBinding(module="typing", name="List")  # from typing import List
        >>> ImportBinding(module="numpy", alias="np")    # import numpy as np
        >>> ImportBinding(module="os.path")              # import os.path
    """

    module: str
    name: Optional[str] = None
    alias: Optional[str] = None
    is_wildcard: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: Dict[str, Any] = {"module": self.module}
        if self.name is not None:
            d["name"] = self.name
        if self.alias is not None:
            d["alias"] = self.alias
        if self.is_wildcard:
            d["is_wildcard"] = self.is_wildcard
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImportBinding":
        """Create from dictionary."""
        return cls(
            module=d["module"],
            name=d.get("name"),
            alias=d.get("alias"),
            is_wildcard=d.get("is_wildcard", False),
        )


@dataclass(frozen=True)
class ModuleStub:
    """Type stub for a module.

    Provides type information for module exports, similar to .pyi stub files.

    Attributes:
        module_name: Full module path
        exports: Mapping of exported names to their type expressions
        submodules: Names of submodules

    Example:
        >>> stub = ModuleStub(
        ...     module_name="mymodule",
        ...     exports={"MyClass": "Type[MyClass]", "helper": "Callable[[int], str]"},
        ...     submodules=("utils", "core"),
        ... )
    """

    module_name: str
    exports: Dict[str, str]
    submodules: Tuple[str, ...] = ()

    def __hash__(self) -> int:
        # Make exports hashable by converting to tuple of items
        return hash(
            (self.module_name, tuple(sorted(self.exports.items())), self.submodules)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: Dict[str, Any] = {
            "module_name": self.module_name,
            "exports": self.exports,
        }
        if self.submodules:
            d["submodules"] = list(self.submodules)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModuleStub":
        """Create from dictionary."""
        return cls(
            module_name=d["module_name"],
            exports=d.get("exports", {}),
            submodules=tuple(d.get("submodules", [])),
        )


# =============================================================================
# Control Flow Context
# =============================================================================


@dataclass(frozen=True)
class ControlFlowContext:
    """Control flow context at the generation point.

    Provides information about the control flow state where code will be
    generated, enabling the ControlFlowDomain to make informed decisions.

    Attributes:
        function_name: Name of containing function (if any)
        function_signature: Full signature of containing function
        expected_return_type: Expected return type for this context
        loop_depth: Nesting depth of loops
        loop_variables: Variables defined by enclosing loops
        in_try_block: Whether inside a try block
        exception_types: Exception types being caught
        in_async_context: Whether in async function/block
        in_generator: Whether in generator function
        reachable: Whether this code point is reachable
        dominators: Dominating code points (for SSA-like analysis)

    Example:
        >>> ctx = ControlFlowContext(
        ...     function_name="process_items",
        ...     expected_return_type="List[Result]",
        ...     loop_depth=1,
        ...     loop_variables=("item",),
        ...     in_async_context=True,
        ... )
    """

    function_name: Optional[str] = None
    function_signature: Optional[FunctionSignature] = None
    expected_return_type: Optional[str] = None
    loop_depth: int = 0
    loop_variables: Tuple[str, ...] = ()
    in_try_block: bool = False
    exception_types: Tuple[str, ...] = ()
    in_async_context: bool = False
    in_generator: bool = False
    reachable: bool = True
    dominators: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: Dict[str, Any] = {}
        if self.function_name is not None:
            d["function_name"] = self.function_name
        if self.function_signature is not None:
            d["function_signature"] = self.function_signature.to_dict()
        if self.expected_return_type is not None:
            d["expected_return_type"] = self.expected_return_type
        if self.loop_depth != 0:
            d["loop_depth"] = self.loop_depth
        if self.loop_variables:
            d["loop_variables"] = list(self.loop_variables)
        if self.in_try_block:
            d["in_try_block"] = self.in_try_block
        if self.exception_types:
            d["exception_types"] = list(self.exception_types)
        if self.in_async_context:
            d["in_async_context"] = self.in_async_context
        if self.in_generator:
            d["in_generator"] = self.in_generator
        if not self.reachable:
            d["reachable"] = self.reachable
        if self.dominators:
            d["dominators"] = list(self.dominators)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ControlFlowContext":
        """Create from dictionary."""
        sig = d.get("function_signature")
        return cls(
            function_name=d.get("function_name"),
            function_signature=(
                FunctionSignature.from_dict(sig) if sig is not None else None
            ),
            expected_return_type=d.get("expected_return_type"),
            loop_depth=d.get("loop_depth", 0),
            loop_variables=tuple(d.get("loop_variables", [])),
            in_try_block=d.get("in_try_block", False),
            exception_types=tuple(d.get("exception_types", [])),
            in_async_context=d.get("in_async_context", False),
            in_generator=d.get("in_generator", False),
            reachable=d.get("reachable", True),
            dominators=tuple(d.get("dominators", [])),
        )


# =============================================================================
# Semantic Constraints
# =============================================================================


@dataclass(frozen=True)
class SemanticConstraint:
    """Semantic constraint for SMT-based checking.

    Represents a logical constraint that must hold during or after
    code generation, checked by the SemanticDomain.

    Attributes:
        kind: Constraint kind ("precondition", "postcondition", "invariant",
              "assertion", "assume")
        expression: Boolean expression in the target language
        scope: Where this constraint applies (function name, block, etc.)
        variables: Free variables referenced in the expression

    Example:
        >>> constraint = SemanticConstraint(
        ...     kind="postcondition",
        ...     expression="result >= 0",
        ...     scope="abs",
        ...     variables=("result",),
        ... )
    """

    kind: str
    expression: str
    scope: Optional[str] = None
    variables: Tuple[str, ...] = ()

    VALID_KINDS = frozenset(
        {"precondition", "postcondition", "invariant", "assertion", "assume"}
    )

    def __post_init__(self) -> None:
        if self.kind not in self.VALID_KINDS:
            # Can't raise in frozen dataclass, but validation happens in from_dict
            pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: Dict[str, Any] = {"kind": self.kind, "expression": self.expression}
        if self.scope is not None:
            d["scope"] = self.scope
        if self.variables:
            d["variables"] = list(self.variables)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SemanticConstraint":
        """Create from dictionary."""
        kind = d["kind"]
        if kind not in cls.VALID_KINDS:
            raise ValueError(
                f"Invalid constraint kind: {kind}. Must be one of {cls.VALID_KINDS}"
            )
        return cls(
            kind=kind,
            expression=d["expression"],
            scope=d.get("scope"),
            variables=tuple(d.get("variables", [])),
        )


# =============================================================================
# Main Constraint Specification
# =============================================================================


@dataclass
class ConstraintSpec:
    """Rich constraint specification - the unified internal representation.

    ConstraintSpec is the core data structure for passing constraint context
    to the Ananke constrained generation system. It supports three input formats:
    1. JSON dict in request body (constraint_spec field)
    2. External reference URI (constraint_uri field)
    3. Dense binary encoding (constraint_bytes field, from ananke tool)

    The specification includes:
    - Core syntax constraint (exactly one of json_schema, regex, ebnf, structural_tag)
    - Language configuration with auto-detection support
    - Type context (bindings, signatures, class definitions)
    - Import context (imports, available modules, forbidden imports)
    - Control flow context
    - Semantic constraints (pre/postconditions, invariants)
    - Domain configuration
    - Cache control

    Attributes:
        version: Specification version for compatibility
        json_schema: JSON schema string (mutually exclusive with other constraints)
        regex: Regular expression pattern (mutually exclusive)
        ebnf: EBNF grammar string (mutually exclusive)
        structural_tag: Structural tag identifier (mutually exclusive)
        language: Explicit language override
        language_detection: Detection strategy (AUTO, EXPLICIT, STACK)
        language_stack: Language frames for polyglot generation
        type_bindings: Variable type bindings
        function_signatures: Function signatures in scope
        class_definitions: Class definitions in scope
        expected_type: Expected type of generated expression
        type_aliases: Type alias mappings
        imports: Import bindings
        available_modules: Set of available module names
        forbidden_imports: Set of forbidden module names
        module_stubs: Type stubs for modules
        control_flow: Control flow context
        semantic_constraints: Semantic constraints
        enabled_domains: Domains to enable (overrides backend default)
        disabled_domains: Domains to disable
        domain_configs: Per-domain configuration
        allow_relaxation: Enable progressive domain relaxation
        relaxation_threshold: Minimum popcount before relaxation triggers
        relaxation_domains: Domains that can be relaxed (default: all except syntax)
        enable_early_termination: Stop generation when regex satisfied at boundary
        max_tokens: Per-constraint token limit (overrides request-level if set)
        cache_scope: What to include in cache key
        context_hash: Pre-computed context hash for efficiency
        source: How the specification was provided
        source_uri: URI for external references

    Example:
        >>> spec = ConstraintSpec(
        ...     json_schema='{"type": "object", "properties": {"name": {"type": "string"}}}',
        ...     language="python",
        ...     type_bindings=[TypeBinding("x", "int"), TypeBinding("y", "str")],
        ...     expected_type="Dict[str, Any]",
        ...     cache_scope=CacheScope.SYNTAX_AND_LANG,
        ... )
    """

    # === Version ===
    version: str = "1.0"

    # === Core Syntax Constraint (exactly one required for grammar) ===
    json_schema: Optional[str] = None
    regex: Optional[str] = None
    negative_regex: Optional[str] = None  # Pattern that output must NOT match
    ebnf: Optional[str] = None
    structural_tag: Optional[str] = None

    # === Language Configuration ===
    language: Optional[str] = None
    language_detection: LanguageDetection = LanguageDetection.AUTO
    language_stack: List[LanguageFrame] = field(default_factory=list)

    # === Type Context ===
    type_bindings: List[TypeBinding] = field(default_factory=list)
    function_signatures: List[FunctionSignature] = field(default_factory=list)
    class_definitions: List[ClassDefinition] = field(default_factory=list)
    expected_type: Optional[str] = None
    type_aliases: Dict[str, str] = field(default_factory=dict)

    # === Import Context ===
    imports: List[ImportBinding] = field(default_factory=list)
    available_modules: Set[str] = field(default_factory=set)
    forbidden_imports: Set[str] = field(default_factory=set)
    module_stubs: Dict[str, ModuleStub] = field(default_factory=dict)

    # === Control Flow Context ===
    control_flow: Optional[ControlFlowContext] = None

    # === Semantic Constraints ===
    semantic_constraints: List[SemanticConstraint] = field(default_factory=list)

    # === Domain Configuration ===
    enabled_domains: Optional[Set[str]] = None
    disabled_domains: Optional[Set[str]] = None
    domain_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # === Adaptive Intensity ===
    # Constraint intensity level (NONE, SYNTAX_ONLY, STANDARD, FULL, EXHAUSTIVE)
    # When set to "auto", intensity is determined by TaskComplexityAssessor
    intensity: Optional[str] = None  # String to avoid circular import; parsed at runtime
    intensity_config: Dict[str, Any] = field(default_factory=dict)  # IntensityConfig overrides

    # === Mask Relaxation ===
    # When enabled, domain constraints are applied progressively and skipped if
    # they would reduce the token mask popcount below the threshold
    allow_relaxation: bool = True
    relaxation_threshold: int = 10  # Minimum popcount before relaxation triggers
    relaxation_domains: Optional[List[str]] = None  # Domains that can be relaxed (default: all except syntax)

    # === Early Termination ===
    # Stop generation when the regex constraint is satisfied at a natural code boundary
    enable_early_termination: bool = True

    # === Generation Limits ===
    # Per-constraint token limit; overrides request-level max_tokens if set
    max_tokens: Optional[int] = None

    # === Cache Control ===
    cache_scope: CacheScope = CacheScope.SYNTAX_ONLY
    context_hash: Optional[str] = None

    # === Source Tracking ===
    source: ConstraintSource = ConstraintSource.INLINE
    source_uri: Optional[str] = None

    def has_syntax_constraint(self) -> bool:
        """Check if a syntax constraint is specified."""
        return any(
            [self.json_schema, self.regex, self.ebnf, self.structural_tag]
        )

    def has_domain_context(self) -> bool:
        """Check if any domain context is specified.

        Domain context includes type bindings, function signatures, imports,
        control flow context, or semantic constraints that would enable
        non-trivial domain-level masking.

        Returns:
            True if any domain context is present
        """
        return bool(
            self.type_bindings
            or self.function_signatures
            or self.class_definitions
            or self.expected_type
            or self.type_aliases
            or self.imports
            or self.available_modules
            or self.forbidden_imports
            or self.control_flow
            or self.semantic_constraints
        )

    def has_any_constraint(self) -> bool:
        """Check if the spec has any meaningful constraint.

        A constraint_spec should have either:
        - A syntax constraint (json_schema, regex, ebnf, structural_tag)
        - Domain context (type bindings, imports, etc.)

        A spec with only 'language' but no syntax or domain context is
        semantically empty and will not apply any actual constraints.

        Returns:
            True if the spec has meaningful constraints
        """
        return self.has_syntax_constraint() or self.has_domain_context()

    def get_intensity(self) -> Optional["ConstraintIntensity"]:
        """Get parsed ConstraintIntensity from intensity string.

        Returns:
            Parsed ConstraintIntensity or None if not set or "auto"
        """
        if self.intensity is None or self.intensity.lower() == "auto":
            return None
        try:
            from ..adaptive.intensity import ConstraintIntensity
            return ConstraintIntensity.from_string(self.intensity)
        except ImportError:
            return None

    def is_auto_intensity(self) -> bool:
        """Check if intensity should be auto-determined."""
        return self.intensity is None or self.intensity.lower() == "auto"

    def get_syntax_constraint_type(self) -> Optional[str]:
        """Get the type of syntax constraint specified."""
        if self.json_schema is not None:
            return "json_schema"
        if self.regex is not None:
            return "regex"
        if self.ebnf is not None:
            return "ebnf"
        if self.structural_tag is not None:
            return "structural_tag"
        return None

    def get_syntax_constraint_value(self) -> Optional[str]:
        """Get the value of the syntax constraint."""
        return self.json_schema or self.regex or self.ebnf or self.structural_tag

    def compute_cache_key(self) -> str:
        """Compute cache key based on cache_scope setting.

        Returns:
            Stable hash string suitable for cache lookup
        """
        parts: List[Tuple[str, str]] = []

        # Always include core constraint
        constraint_type = self.get_syntax_constraint_type()
        constraint_value = self.get_syntax_constraint_value()
        if constraint_type and constraint_value:
            parts.append((constraint_type, _hash_content(constraint_value)))

        # Include language if scope >= SYNTAX_AND_LANG
        if self.cache_scope in (CacheScope.SYNTAX_AND_LANG, CacheScope.FULL_CONTEXT):
            if self.language:
                parts.append(("lang", self.language))

        # Include context hash if scope == FULL_CONTEXT
        if self.cache_scope == CacheScope.FULL_CONTEXT:
            ctx_hash = self.context_hash or self._compute_context_hash()
            parts.append(("ctx", ctx_hash))

        return _stable_hash(parts)

    def _compute_context_hash(self) -> str:
        """Compute hash of all context (type, import, control flow, semantic)."""
        ctx_parts = []

        # Type context
        if self.type_bindings:
            ctx_parts.append(("type_bindings", str(self.type_bindings)))
        if self.function_signatures:
            ctx_parts.append(("function_signatures", str(self.function_signatures)))
        if self.class_definitions:
            ctx_parts.append(("class_definitions", str(self.class_definitions)))
        if self.expected_type:
            ctx_parts.append(("expected_type", self.expected_type))
        if self.type_aliases:
            ctx_parts.append(("type_aliases", str(sorted(self.type_aliases.items()))))

        # Import context
        if self.imports:
            ctx_parts.append(("imports", str(self.imports)))
        if self.available_modules:
            ctx_parts.append(("available_modules", str(sorted(self.available_modules))))
        if self.forbidden_imports:
            ctx_parts.append(("forbidden_imports", str(sorted(self.forbidden_imports))))

        # Control flow context
        if self.control_flow:
            ctx_parts.append(("control_flow", str(self.control_flow)))

        # Semantic constraints
        if self.semantic_constraints:
            ctx_parts.append(("semantic_constraints", str(self.semantic_constraints)))

        return _stable_hash(ctx_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: Dict[str, Any] = {"version": self.version}

        # Core syntax constraint
        if self.json_schema is not None:
            d["json_schema"] = self.json_schema
        if self.regex is not None:
            d["regex"] = self.regex
        if self.negative_regex is not None:
            d["negative_regex"] = self.negative_regex
        if self.ebnf is not None:
            d["ebnf"] = self.ebnf
        if self.structural_tag is not None:
            d["structural_tag"] = self.structural_tag

        # Language configuration
        if self.language is not None:
            d["language"] = self.language
        if self.language_detection != LanguageDetection.AUTO:
            d["language_detection"] = str(self.language_detection)
        if self.language_stack:
            d["language_stack"] = [f.to_dict() for f in self.language_stack]

        # Type context
        if self.type_bindings:
            d["type_bindings"] = [b.to_dict() for b in self.type_bindings]
        if self.function_signatures:
            d["function_signatures"] = [s.to_dict() for s in self.function_signatures]
        if self.class_definitions:
            d["class_definitions"] = [c.to_dict() for c in self.class_definitions]
        if self.expected_type is not None:
            d["expected_type"] = self.expected_type
        if self.type_aliases:
            d["type_aliases"] = self.type_aliases

        # Import context
        if self.imports:
            d["imports"] = [i.to_dict() for i in self.imports]
        if self.available_modules:
            d["available_modules"] = sorted(self.available_modules)
        if self.forbidden_imports:
            d["forbidden_imports"] = sorted(self.forbidden_imports)
        if self.module_stubs:
            d["module_stubs"] = {k: v.to_dict() for k, v in self.module_stubs.items()}

        # Control flow context
        if self.control_flow is not None:
            d["control_flow"] = self.control_flow.to_dict()

        # Semantic constraints
        if self.semantic_constraints:
            d["semantic_constraints"] = [c.to_dict() for c in self.semantic_constraints]

        # Domain configuration
        if self.enabled_domains is not None:
            d["enabled_domains"] = sorted(self.enabled_domains)
        if self.disabled_domains is not None:
            d["disabled_domains"] = sorted(self.disabled_domains)
        if self.domain_configs:
            d["domain_configs"] = self.domain_configs

        # Adaptive intensity
        if self.intensity is not None:
            d["intensity"] = self.intensity
        if self.intensity_config:
            d["intensity_config"] = self.intensity_config

        # Cache control
        if self.cache_scope != CacheScope.SYNTAX_ONLY:
            d["cache_scope"] = str(self.cache_scope)
        if self.context_hash is not None:
            d["context_hash"] = self.context_hash

        # Source tracking
        if self.source != ConstraintSource.INLINE:
            d["source"] = str(self.source)
        if self.source_uri is not None:
            d["source_uri"] = self.source_uri

        # Relaxation config (only include non-defaults)
        if not self.allow_relaxation:
            d["allow_relaxation"] = self.allow_relaxation
        if self.relaxation_threshold != 10:
            d["relaxation_threshold"] = self.relaxation_threshold

        # Early termination (only include if disabled)
        if not self.enable_early_termination:
            d["enable_early_termination"] = self.enable_early_termination

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConstraintSpec":
        """Create from dictionary."""
        return cls(
            version=d.get("version", "1.0"),
            # Core syntax constraint
            json_schema=d.get("json_schema"),
            regex=d.get("regex"),
            negative_regex=d.get("negative_regex"),
            ebnf=d.get("ebnf"),
            structural_tag=d.get("structural_tag"),
            # Language configuration
            language=d.get("language"),
            language_detection=LanguageDetection(
                d.get("language_detection", "auto")
            ),
            language_stack=[
                LanguageFrame.from_dict(f) for f in d.get("language_stack", [])
            ],
            # Type context
            type_bindings=[
                TypeBinding.from_dict(b) for b in d.get("type_bindings", [])
            ],
            function_signatures=[
                FunctionSignature.from_dict(s)
                for s in d.get("function_signatures", [])
            ],
            class_definitions=[
                ClassDefinition.from_dict(c) for c in d.get("class_definitions", [])
            ],
            expected_type=d.get("expected_type"),
            type_aliases=d.get("type_aliases", {}),
            # Import context
            imports=[ImportBinding.from_dict(i) for i in d.get("imports", [])],
            available_modules=set(d.get("available_modules", [])),
            forbidden_imports=set(d.get("forbidden_imports", [])),
            module_stubs={
                k: ModuleStub.from_dict(v)
                for k, v in d.get("module_stubs", {}).items()
            },
            # Control flow context
            control_flow=(
                ControlFlowContext.from_dict(d["control_flow"])
                if "control_flow" in d
                else None
            ),
            # Semantic constraints
            semantic_constraints=[
                SemanticConstraint.from_dict(c)
                for c in d.get("semantic_constraints", [])
            ],
            # Domain configuration
            # Accept "domains" as shorthand alias for "enabled_domains"
            enabled_domains=(
                set(d["enabled_domains"])
                if "enabled_domains" in d
                else (set(d["domains"]) if "domains" in d else None)
            ),
            disabled_domains=(
                set(d["disabled_domains"]) if "disabled_domains" in d else None
            ),
            domain_configs=d.get("domain_configs", {}),
            # Adaptive intensity
            intensity=d.get("intensity"),
            intensity_config=d.get("intensity_config", {}),
            # Cache control
            cache_scope=CacheScope.from_string(d.get("cache_scope", "syntax_only")),
            context_hash=d.get("context_hash"),
            # Source tracking
            source=ConstraintSource(d.get("source", "inline")),
            source_uri=d.get("source_uri"),
            # Relaxation config
            allow_relaxation=d.get("allow_relaxation", True),
            relaxation_threshold=d.get("relaxation_threshold", 10),
            # Early termination
            enable_early_termination=d.get("enable_early_termination", True),
        )

    @classmethod
    def from_legacy(
        cls,
        json_schema: Optional[str] = None,
        regex: Optional[str] = None,
        ebnf: Optional[str] = None,
        structural_tag: Optional[str] = None,
    ) -> "ConstraintSpec":
        """Create from legacy constraint parameters.

        Provides backward compatibility with existing SGLang constraint APIs.

        Args:
            json_schema: JSON schema string
            regex: Regular expression pattern
            ebnf: EBNF grammar string
            structural_tag: Structural tag identifier

        Returns:
            ConstraintSpec with legacy source tracking
        """
        return cls(
            json_schema=json_schema,
            regex=regex,
            ebnf=ebnf,
            structural_tag=structural_tag,
            source=ConstraintSource.LEGACY,
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_json(cls, json_str: str) -> "ConstraintSpec":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def copy(self, **updates: Any) -> "ConstraintSpec":
        """Create a copy with optional field updates."""
        d = self.to_dict()
        d.update(updates)
        return self.from_dict(d)

    # =========================================================================
    # Factory Methods for Ergonomic Construction
    # =========================================================================

    @classmethod
    def from_regex(
        cls,
        pattern: str,
        language: Optional[str] = None,
        expected_type: Optional[str] = None,
    ) -> "ConstraintSpec":
        """Create a regex constraint specification.

        Factory method for simple regex-based constrained generation.

        Args:
            pattern: Regular expression pattern that output must match
            language: Optional language hint (python, rust, typescript, etc.)
            expected_type: Optional expected type of the generated expression

        Returns:
            ConstraintSpec configured with the regex pattern

        Example:
            >>> # Simple regex constraint
            >>> spec = ConstraintSpec.from_regex(r'^\\s+return\\s+')
            >>>
            >>> # With language hint
            >>> spec = ConstraintSpec.from_regex(
            ...     r'^\\s+if n <= 1:',
            ...     language="python",
            ... )
            >>>
            >>> # With expected type
            >>> spec = ConstraintSpec.from_regex(
            ...     r'^return\\s+\\[',
            ...     language="python",
            ...     expected_type="List[int]",
            ... )
        """
        return cls(
            regex=pattern,
            language=language,
            expected_type=expected_type,
        )

    @classmethod
    def from_json_schema(
        cls,
        schema: Union[str, Dict[str, Any]],
        language: Optional[str] = None,
    ) -> "ConstraintSpec":
        """Create a JSON schema constraint specification.

        Factory method for JSON schema-based constrained generation.

        Args:
            schema: JSON schema as string or dict
            language: Optional language hint

        Returns:
            ConstraintSpec configured with the JSON schema

        Example:
            >>> spec = ConstraintSpec.from_json_schema({
            ...     "type": "object",
            ...     "properties": {
            ...         "name": {"type": "string"},
            ...         "age": {"type": "integer"},
            ...     },
            ...     "required": ["name", "age"],
            ... })
        """
        if isinstance(schema, dict):
            schema = json.dumps(schema)
        return cls(
            json_schema=schema,
            language=language,
        )

    @classmethod
    def from_ebnf(
        cls,
        grammar: str,
        language: Optional[str] = None,
    ) -> "ConstraintSpec":
        """Create an EBNF grammar constraint specification.

        Factory method for EBNF-based constrained generation.

        Args:
            grammar: EBNF grammar string
            language: Optional language hint

        Returns:
            ConstraintSpec configured with the EBNF grammar

        Example:
            >>> spec = ConstraintSpec.from_ebnf('''
            ...     root ::= greeting name "!"
            ...     greeting ::= "Hello, " | "Hi, "
            ...     name ::= [A-Za-z]+
            ... ''')
        """
        return cls(
            ebnf=grammar,
            language=language,
        )

    @classmethod
    def from_structural_tag(
        cls,
        tag: str,
        language: Optional[str] = None,
    ) -> "ConstraintSpec":
        """Create a structural tag constraint specification.

        Factory method for structural tag-based constrained generation.

        Args:
            tag: Structural tag identifier
            language: Optional language hint

        Returns:
            ConstraintSpec configured with the structural tag
        """
        return cls(
            structural_tag=tag,
            language=language,
        )

    @classmethod
    def for_completion(
        cls,
        language: str,
        expected_type: Optional[str] = None,
        in_function: Optional[str] = None,
        return_type: Optional[str] = None,
        type_bindings: Optional[List["TypeBinding"]] = None,
    ) -> "ConstraintSpec":
        """Create a constraint specification for code completion.

        Factory method for code completion scenarios with type context.

        Args:
            language: Programming language
            expected_type: Expected type of the generated expression
            in_function: Name of containing function (if any)
            return_type: Expected return type (if in function)
            type_bindings: Variable type bindings in scope

        Returns:
            ConstraintSpec configured for code completion

        Example:
            >>> spec = ConstraintSpec.for_completion(
            ...     language="python",
            ...     expected_type="int",
            ...     in_function="fibonacci",
            ...     return_type="int",
            ...     type_bindings=[
            ...         TypeBinding("n", "int", scope="parameter"),
            ...     ],
            ... )
        """
        control_flow = None
        if in_function or return_type:
            control_flow = ControlFlowContext(
                function_name=in_function,
                expected_return_type=return_type,
            )

        return cls(
            language=language,
            language_detection=LanguageDetection.EXPLICIT,
            expected_type=expected_type,
            type_bindings=type_bindings or [],
            control_flow=control_flow,
        )

    @classmethod
    def builder(cls) -> "ConstraintBuilder":
        """Get a fluent builder for constructing ConstraintSpec.

        Returns a ConstraintBuilder instance for ergonomic construction
        of complex constraint specifications.

        Returns:
            New ConstraintBuilder instance

        Example:
            >>> spec = (
            ...     ConstraintSpec.builder()
            ...     .language("python")
            ...     .regex(r"^return\\s+")
            ...     .type_binding("x", "int")
            ...     .expected_type("int")
            ...     .build()
            ... )
        """
        from ..client import ConstraintBuilder
        return ConstraintBuilder()


# Type alias for use in factory method
ConstraintBuilder = Any  # Forward reference, actual import done at runtime


# =============================================================================
# Helper Functions
# =============================================================================


def _hash_content(content: str) -> str:
    """Hash string content using SHA-256."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _stable_hash(parts: List[Tuple[str, str]]) -> str:
    """Create stable hash from list of (key, value) pairs."""
    combined = "|".join(f"{k}:{v}" for k, v in sorted(parts))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:32]
