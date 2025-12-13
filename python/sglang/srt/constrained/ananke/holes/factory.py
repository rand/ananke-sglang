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
"""Factory for creating typed holes from various sources.

Creates holes from:
- AST gaps (parsing incomplete code)
- Type errors (converting inconsistencies to holes)
- Incomplete expressions
- User annotations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar

from .hole import (
    Hole,
    HoleId,
    HoleGranularity,
    HoleState,
    SourceLocation,
    TypeEnvironment,
)
from .registry import HoleRegistry
from .environment_capture import EnvironmentCapture, capture_environment

# Type variable for constraint types
C = TypeVar("C")


@dataclass
class HoleSpec:
    """Specification for creating a hole.

    Attributes:
        name: Descriptive name
        namespace: Organizational namespace
        expected_type: Expected type for the hole
        granularity: Code granularity
        location: Source location
        parent: Parent hole ID
        metadata: Additional metadata
    """

    name: str = "hole"
    namespace: str = "default"
    expected_type: Optional[Any] = None
    granularity: HoleGranularity = HoleGranularity.EXPRESSION
    location: Optional[SourceLocation] = None
    parent: Optional[HoleId] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class HoleFactory(Generic[C]):
    """Factory for creating typed holes.

    Provides methods to create holes from various sources:
    - Direct specification
    - AST gaps during parsing
    - Type errors that need filling
    - Template placeholders

    All holes are automatically registered with the registry.

    Example:
        >>> factory = HoleFactory(registry)
        >>> hole = factory.from_spec(HoleSpec(
        ...     name="function_body",
        ...     expected_type="int",
        ...     granularity=HoleGranularity.BLOCK,
        ... ))
    """

    def __init__(
        self,
        registry: HoleRegistry[C],
        default_namespace: str = "default",
    ) -> None:
        """Initialize the factory.

        Args:
            registry: Hole registry to use
            default_namespace: Default namespace for new holes
        """
        self._registry = registry
        self._default_namespace = default_namespace
        self._hole_counter: Dict[str, int] = {}

    @property
    def registry(self) -> HoleRegistry[C]:
        """Get the hole registry."""
        return self._registry

    def _next_index(self, namespace: str, name: str) -> int:
        """Get the next index for a hole name.

        Args:
            namespace: Hole namespace
            name: Hole name

        Returns:
            Next unique index
        """
        key = f"{namespace}:{name}"
        if key not in self._hole_counter:
            self._hole_counter[key] = 0
        index = self._hole_counter[key]
        self._hole_counter[key] += 1
        return index

    def from_spec(
        self,
        spec: HoleSpec,
        environment: Optional[TypeEnvironment] = None,
        constraint: Optional[C] = None,
    ) -> Hole[C]:
        """Create a hole from a specification.

        Args:
            spec: Hole specification
            environment: Typing environment (optional)
            constraint: Domain constraint (optional)

        Returns:
            Created and registered hole
        """
        return self._registry.create(
            name=spec.name,
            namespace=spec.namespace or self._default_namespace,
            expected_type=spec.expected_type,
            environment=environment,
            constraint=constraint,
            granularity=spec.granularity,
            parent=spec.parent,
        )

    def from_ast_gap(
        self,
        location: SourceLocation,
        *,
        name: str = "gap",
        expected_type: Optional[Any] = None,
        granularity: HoleGranularity = HoleGranularity.EXPRESSION,
        environment: Optional[TypeEnvironment] = None,
        parent: Optional[HoleId] = None,
    ) -> Hole[C]:
        """Create a hole from an AST gap.

        Used when parsing incomplete code - creates a hole
        to represent the missing portion.

        Args:
            location: Source location of the gap
            name: Descriptive name
            expected_type: Expected type
            granularity: Code granularity
            environment: Typing environment
            parent: Parent hole ID

        Returns:
            Created hole
        """
        hole = self._registry.create(
            name=name,
            namespace="ast",
            expected_type=expected_type,
            environment=environment,
            granularity=granularity,
            parent=parent,
        )

        # Update with location
        updated = Hole(
            id=hole.id,
            expected_type=hole.expected_type,
            environment=hole.environment,
            constraint=hole.constraint,
            granularity=hole.granularity,
            location=location,
            parent=hole.parent,
            content=hole.content,
            state=hole.state,
            metadata={"source": "ast_gap"},
        )

        # Re-register with location
        self._registry._holes[hole.id] = updated
        return updated

    def from_type_error(
        self,
        location: SourceLocation,
        synthesized_type: Any,
        expected_type: Any,
        *,
        name: str = "type_error",
        environment: Optional[TypeEnvironment] = None,
    ) -> Hole[C]:
        """Create a hole from a type error.

        Converts a type inconsistency into a hole that can be filled
        with correctly typed code. This is the "error as data" approach
        from Hazel.

        Args:
            location: Error location
            synthesized_type: Type that was synthesized
            expected_type: Type that was expected
            name: Descriptive name
            environment: Typing environment

        Returns:
            Created hole
        """
        hole = self._registry.create(
            name=name,
            namespace="type_error",
            expected_type=expected_type,
            environment=environment,
            granularity=HoleGranularity.EXPRESSION,
        )

        # Update with error metadata
        updated = Hole(
            id=hole.id,
            expected_type=hole.expected_type,
            environment=hole.environment,
            constraint=hole.constraint,
            granularity=hole.granularity,
            location=location,
            parent=hole.parent,
            content=hole.content,
            state=hole.state,
            metadata={
                "source": "type_error",
                "synthesized_type": synthesized_type,
                "expected_type": expected_type,
            },
        )

        self._registry._holes[hole.id] = updated
        return updated

    def from_incomplete_expression(
        self,
        partial_code: str,
        location: SourceLocation,
        *,
        expected_type: Optional[Any] = None,
        environment: Optional[TypeEnvironment] = None,
    ) -> Hole[C]:
        """Create a hole from an incomplete expression.

        Used when code generation produces an incomplete expression
        that needs completion.

        Args:
            partial_code: The partial code so far
            location: Source location
            expected_type: Expected type
            environment: Typing environment

        Returns:
            Created hole with partial content
        """
        hole = self._registry.create(
            name="incomplete",
            namespace="expression",
            expected_type=expected_type,
            environment=environment,
            granularity=HoleGranularity.EXPRESSION,
        )

        # Update with partial content
        updated = Hole(
            id=hole.id,
            expected_type=hole.expected_type,
            environment=hole.environment,
            constraint=hole.constraint,
            granularity=hole.granularity,
            location=location,
            parent=hole.parent,
            content=partial_code,
            state=HoleState.PARTIAL,
            metadata={"source": "incomplete_expression"},
        )

        self._registry._holes[hole.id] = updated
        return updated

    def from_placeholder(
        self,
        placeholder: str,
        location: SourceLocation,
        *,
        expected_type: Optional[Any] = None,
        environment: Optional[TypeEnvironment] = None,
    ) -> Hole[C]:
        """Create a hole from a placeholder marker in code.

        Handles common placeholder patterns like:
        - ??? or ... (Python ellipsis)
        - TODO: ... comments
        - _hole_, __HOLE__, etc.

        Args:
            placeholder: The placeholder text
            location: Source location
            expected_type: Expected type
            environment: Typing environment

        Returns:
            Created hole
        """
        # Determine name from placeholder
        if placeholder in ("???", "...", "pass"):
            name = "placeholder"
        elif "TODO" in placeholder.upper():
            name = "todo"
        elif "HOLE" in placeholder.upper():
            name = "hole"
        else:
            name = "placeholder"

        hole = self._registry.create(
            name=name,
            namespace="placeholder",
            expected_type=expected_type,
            environment=environment,
            granularity=HoleGranularity.EXPRESSION,
        )

        # Update with placeholder metadata
        updated = Hole(
            id=hole.id,
            expected_type=hole.expected_type,
            environment=hole.environment,
            constraint=hole.constraint,
            granularity=hole.granularity,
            location=location,
            parent=hole.parent,
            content=None,
            state=HoleState.EMPTY,
            metadata={
                "source": "placeholder",
                "placeholder_text": placeholder,
            },
        )

        self._registry._holes[hole.id] = updated
        return updated

    def create_nested(
        self,
        parent_id: HoleId,
        name: str,
        *,
        expected_type: Optional[Any] = None,
        granularity: HoleGranularity = HoleGranularity.EXPRESSION,
    ) -> Hole[C]:
        """Create a nested hole within another hole.

        Args:
            parent_id: Parent hole ID
            name: Name for the nested hole
            expected_type: Expected type
            granularity: Code granularity

        Returns:
            Created nested hole
        """
        parent = self._registry.lookup(parent_id)
        if parent is None:
            raise ValueError(f"Parent hole not found: {parent_id}")

        return self._registry.create(
            name=name,
            namespace=parent.id.namespace,
            expected_type=expected_type,
            environment=parent.environment,
            granularity=granularity,
            parent=parent_id,
        )

    def batch_create(self, specs: List[HoleSpec]) -> List[Hole[C]]:
        """Create multiple holes from specifications.

        Args:
            specs: List of hole specifications

        Returns:
            List of created holes
        """
        return [self.from_spec(spec) for spec in specs]

    def reset_counters(self) -> None:
        """Reset the hole index counters."""
        self._hole_counter.clear()


def create_factory(
    registry: Optional[HoleRegistry[Any]] = None,
) -> HoleFactory[Any]:
    """Factory function to create a HoleFactory.

    Args:
        registry: Optional registry (creates new if None)

    Returns:
        New HoleFactory
    """
    reg = registry or HoleRegistry()
    return HoleFactory(reg)
