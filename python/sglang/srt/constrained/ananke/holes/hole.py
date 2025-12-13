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
"""Typed holes following the Hazel research program.

Holes represent incomplete portions of code during generation.
From Hazel (Cyrus Omar et al.):
- "Holes serve as membranes around missing code and type inconsistencies"
- Every partial program with holes has a well-defined type
- Holes capture their typing environment for fill-and-resume semantics

References:
- Live Functional Programming with Typed Holes (ICFP 2019)
- Total Type Error Localization and Recovery with Holes (POPL 2024)
- Statically Contextualizing LLMs with Typed Holes (OOPSLA 2024)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, Generic, Optional, TypeVar

# Type variable for constraint types
C = TypeVar("C")


class HoleGranularity(Enum):
    """Granularity level of a hole.

    Holes can represent missing code at different granularities:
    - TOKEN: A single token placeholder
    - EXPRESSION: A complete expression (e.g., function argument)
    - STATEMENT: A complete statement
    - BLOCK: A block of code (e.g., loop body)
    - FUNCTION: An entire function body
    - MODULE: An entire module
    - SYSTEM: System-level placeholder
    """

    TOKEN = auto()       # Single token
    EXPRESSION = auto()  # Expression (e.g., x + ?)
    STATEMENT = auto()   # Statement (e.g., if ?:)
    BLOCK = auto()       # Block of statements
    FUNCTION = auto()    # Function body
    MODULE = auto()      # Module content
    SYSTEM = auto()      # System-level


@dataclass(frozen=True)
class SourceLocation:
    """Source code location for a hole.

    Attributes:
        file: Source file path (optional)
        line: Line number (1-indexed)
        column: Column number (0-indexed)
        offset: Byte offset from start of file
        length: Length in bytes
    """

    line: int = 0
    column: int = 0
    offset: int = 0
    length: int = 0
    file: Optional[str] = None

    def __str__(self) -> str:
        if self.file:
            return f"{self.file}:{self.line}:{self.column}"
        return f"{self.line}:{self.column}"

    def contains(self, offset: int) -> bool:
        """Check if an offset is within this location."""
        return self.offset <= offset < self.offset + self.length

    def overlaps(self, other: SourceLocation) -> bool:
        """Check if this location overlaps with another."""
        return (
            self.offset < other.offset + other.length and
            other.offset < self.offset + self.length
        )


@dataclass(frozen=True)
class HoleId:
    """Unique identifier for a hole.

    Holes are identified by:
    - namespace: Organizational namespace (e.g., "user", "system", "type")
    - name: Human-readable name (e.g., "function_body", "return_expr")
    - index: Unique index within namespace/name combination
    - depth: Nesting depth for hierarchical holes

    Attributes:
        namespace: Namespace for organization
        name: Descriptive name
        index: Unique index
        depth: Nesting depth (0 = top-level)
    """

    namespace: str = "default"
    name: str = "hole"
    index: int = 0
    depth: int = 0

    def __str__(self) -> str:
        return f"?{self.namespace}:{self.name}[{self.index}]"

    def __repr__(self) -> str:
        return f"HoleId({self.namespace!r}, {self.name!r}, {self.index}, {self.depth})"

    def with_depth(self, depth: int) -> HoleId:
        """Return a new HoleId with different depth."""
        return HoleId(
            namespace=self.namespace,
            name=self.name,
            index=self.index,
            depth=depth,
        )

    def child(self, name: str, index: int = 0) -> HoleId:
        """Create a child hole ID."""
        return HoleId(
            namespace=self.namespace,
            name=name,
            index=index,
            depth=self.depth + 1,
        )

    @classmethod
    def create(
        cls,
        name: str = "hole",
        *,
        namespace: str = "default",
        index: int = 0,
        depth: int = 0,
    ) -> HoleId:
        """Factory method to create a HoleId."""
        return cls(
            namespace=namespace,
            name=name,
            index=index,
            depth=depth,
        )


class HoleState(Enum):
    """State of a hole during generation.

    States:
    - EMPTY: Hole has no content (awaiting fill)
    - PARTIAL: Hole has partial content (still being filled)
    - FILLED: Hole has complete content
    - VALIDATED: Hole content has been type-checked
    - INVALID: Hole content failed type checking
    """

    EMPTY = auto()
    PARTIAL = auto()
    FILLED = auto()
    VALIDATED = auto()
    INVALID = auto()


@dataclass(frozen=True)
class TypeEnvironment:
    """Captured typing environment at a hole site.

    Stores all type bindings visible at the hole position.
    This enables fill-and-resume semantics where filled code
    has access to the same type context as the original hole.

    Attributes:
        bindings: Map from name to type
        parent: Parent environment (for lexical scoping)
    """

    bindings: FrozenSet[tuple[str, Any]] = field(default_factory=frozenset)
    parent: Optional[TypeEnvironment] = None

    def lookup(self, name: str) -> Optional[Any]:
        """Look up a type binding by name."""
        for n, t in self.bindings:
            if n == name:
                return t
        if self.parent:
            return self.parent.lookup(name)
        return None

    def bind(self, name: str, typ: Any) -> TypeEnvironment:
        """Create a new environment with an additional binding."""
        new_bindings = frozenset(self.bindings | {(name, typ)})
        return TypeEnvironment(bindings=new_bindings, parent=self.parent)

    def all_bindings(self) -> Dict[str, Any]:
        """Get all bindings including from parent environments."""
        result: Dict[str, Any] = {}
        if self.parent:
            result.update(self.parent.all_bindings())
        for name, typ in self.bindings:
            result[name] = typ
        return result

    def names(self) -> FrozenSet[str]:
        """Get all bound names."""
        local_names = frozenset(n for n, _ in self.bindings)
        if self.parent:
            return local_names | self.parent.names()
        return local_names

    @classmethod
    def empty(cls) -> TypeEnvironment:
        """Create an empty environment."""
        return cls()

    @classmethod
    def from_dict(cls, bindings: Dict[str, Any]) -> TypeEnvironment:
        """Create an environment from a dictionary."""
        return cls(bindings=frozenset(bindings.items()))


@dataclass
class Hole(Generic[C]):
    """A typed hole representing incomplete code.

    Following Hazel, holes are first-class citizens in the type system.
    Each hole has:
    - A unique identifier
    - An expected type (what type of code should fill it)
    - A captured environment (available bindings)
    - An optional constraint (domain-specific restrictions)
    - Optional content (for partial fills)

    Type Parameters:
        C: The constraint type for this hole

    Attributes:
        id: Unique hole identifier
        expected_type: The type expected for the filled code
        environment: Captured typing environment
        constraint: Domain-specific constraint
        granularity: How much code this hole represents
        location: Source location
        parent: Parent hole (for nested holes)
        content: Current content (if partially filled)
        state: Current hole state
        metadata: Additional metadata
    """

    id: HoleId
    expected_type: Optional[Any] = None
    environment: TypeEnvironment = field(default_factory=TypeEnvironment.empty)
    constraint: Optional[C] = None
    granularity: HoleGranularity = HoleGranularity.EXPRESSION
    location: Optional[SourceLocation] = None
    parent: Optional[HoleId] = None
    content: Optional[str] = None
    state: HoleState = HoleState.EMPTY
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate hole state after initialization."""
        if self.content is not None and self.state == HoleState.EMPTY:
            # Content implies at least PARTIAL state
            object.__setattr__(self, "state", HoleState.PARTIAL)

    @property
    def is_empty(self) -> bool:
        """Check if the hole has no content."""
        return self.state == HoleState.EMPTY

    @property
    def is_filled(self) -> bool:
        """Check if the hole is completely filled."""
        return self.state in (HoleState.FILLED, HoleState.VALIDATED)

    @property
    def is_valid(self) -> bool:
        """Check if the hole content is valid."""
        return self.state == HoleState.VALIDATED

    @property
    def is_nested(self) -> bool:
        """Check if this is a nested hole."""
        return self.parent is not None

    @property
    def depth(self) -> int:
        """Get the nesting depth of this hole."""
        return self.id.depth

    def fill(self, content: str) -> Hole[C]:
        """Create a new hole with the given content.

        Does not modify this hole - returns a new one.
        """
        return Hole(
            id=self.id,
            expected_type=self.expected_type,
            environment=self.environment,
            constraint=self.constraint,
            granularity=self.granularity,
            location=self.location,
            parent=self.parent,
            content=content,
            state=HoleState.FILLED,
            metadata=self.metadata.copy(),
        )

    def unfill(self) -> Hole[C]:
        """Create a new empty hole from this one.

        Used for backtracking.
        """
        return Hole(
            id=self.id,
            expected_type=self.expected_type,
            environment=self.environment,
            constraint=self.constraint,
            granularity=self.granularity,
            location=self.location,
            parent=self.parent,
            content=None,
            state=HoleState.EMPTY,
            metadata=self.metadata.copy(),
        )

    def with_constraint(self, constraint: C) -> Hole[C]:
        """Create a new hole with updated constraint."""
        return Hole(
            id=self.id,
            expected_type=self.expected_type,
            environment=self.environment,
            constraint=constraint,
            granularity=self.granularity,
            location=self.location,
            parent=self.parent,
            content=self.content,
            state=self.state,
            metadata=self.metadata.copy(),
        )

    def with_type(self, expected_type: Any) -> Hole[C]:
        """Create a new hole with updated expected type."""
        return Hole(
            id=self.id,
            expected_type=expected_type,
            environment=self.environment,
            constraint=self.constraint,
            granularity=self.granularity,
            location=self.location,
            parent=self.parent,
            content=self.content,
            state=self.state,
            metadata=self.metadata.copy(),
        )

    def with_environment(self, environment: TypeEnvironment) -> Hole[C]:
        """Create a new hole with updated environment."""
        return Hole(
            id=self.id,
            expected_type=self.expected_type,
            environment=environment,
            constraint=self.constraint,
            granularity=self.granularity,
            location=self.location,
            parent=self.parent,
            content=self.content,
            state=self.state,
            metadata=self.metadata.copy(),
        )

    def validate(self) -> Hole[C]:
        """Mark the hole as validated."""
        return Hole(
            id=self.id,
            expected_type=self.expected_type,
            environment=self.environment,
            constraint=self.constraint,
            granularity=self.granularity,
            location=self.location,
            parent=self.parent,
            content=self.content,
            state=HoleState.VALIDATED,
            metadata=self.metadata.copy(),
        )

    def invalidate(self) -> Hole[C]:
        """Mark the hole as invalid."""
        return Hole(
            id=self.id,
            expected_type=self.expected_type,
            environment=self.environment,
            constraint=self.constraint,
            granularity=self.granularity,
            location=self.location,
            parent=self.parent,
            content=self.content,
            state=HoleState.INVALID,
            metadata=self.metadata.copy(),
        )

    def child_hole(self, name: str, index: int = 0) -> Hole[C]:
        """Create a child hole nested within this one."""
        child_id = self.id.child(name, index)
        return Hole(
            id=child_id,
            expected_type=None,
            environment=self.environment,
            constraint=None,
            granularity=self.granularity,
            location=None,
            parent=self.id,
            content=None,
            state=HoleState.EMPTY,
            metadata={},
        )

    def __str__(self) -> str:
        type_str = str(self.expected_type) if self.expected_type else "?"
        return f"Hole({self.id}: {type_str})"

    def __repr__(self) -> str:
        return (
            f"Hole(id={self.id!r}, "
            f"expected_type={self.expected_type!r}, "
            f"state={self.state.name})"
        )


def create_hole(
    name: str = "hole",
    *,
    namespace: str = "default",
    index: int = 0,
    expected_type: Optional[Any] = None,
    environment: Optional[TypeEnvironment] = None,
    granularity: HoleGranularity = HoleGranularity.EXPRESSION,
) -> Hole[Any]:
    """Factory function to create a hole.

    Args:
        name: Descriptive name for the hole
        namespace: Organizational namespace
        index: Unique index
        expected_type: Expected type for filled code
        environment: Captured typing environment
        granularity: Code granularity level

    Returns:
        New Hole instance
    """
    hole_id = HoleId.create(name, namespace=namespace, index=index)
    return Hole(
        id=hole_id,
        expected_type=expected_type,
        environment=environment or TypeEnvironment.empty(),
        granularity=granularity,
    )
