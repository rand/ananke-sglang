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
"""Import constraint for module/package availability.

The ImportConstraint tracks which modules/packages are:
- Required: Must be imported for the code to work
- Forbidden: Cannot be used (e.g., security policy)
- Available: Currently imported and usable

Import constraints form a semilattice where:
- TOP: No import restrictions
- BOTTOM: Conflicting requirements (required ∩ forbidden ≠ ∅)
- meet(): Union of requirements, intersection of allowed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional

try:
    from ...core.constraint import Constraint, Satisfiability
except ImportError:
    from core.constraint import Constraint, Satisfiability


@dataclass(frozen=True, slots=True)
class ModuleSpec:
    """Specification for a module/package.

    Attributes:
        name: Module name (e.g., "numpy", "typing")
        version: Optional version constraint (e.g., ">=1.0.0")
        alias: Optional import alias (e.g., "np" for numpy)
    """

    name: str
    version: Optional[str] = None
    alias: Optional[str] = None

    def __repr__(self) -> str:
        parts = [self.name]
        if self.version:
            parts.append(f"({self.version})")
        if self.alias:
            parts.append(f"as {self.alias}")
        return " ".join(parts)


@dataclass(frozen=True)
class ImportConstraint(Constraint["ImportConstraint"]):
    """Constraint on module/package imports.

    Tracks three sets of modules:
    - required: Modules that MUST be imported
    - forbidden: Modules that CANNOT be imported
    - available: Modules currently imported and usable

    The constraint is satisfiable iff required ∩ forbidden = ∅.

    Attributes:
        required: Set of required module specifications
        forbidden: Set of forbidden module names
        available: Set of currently available module specs
        _is_top: True if this is TOP (no restrictions)
        _is_bottom: True if this is BOTTOM (unsatisfiable)
    """

    required: FrozenSet[ModuleSpec] = field(default_factory=frozenset)
    forbidden: FrozenSet[str] = field(default_factory=frozenset)
    available: FrozenSet[ModuleSpec] = field(default_factory=frozenset)
    _is_top: bool = False
    _is_bottom: bool = False

    def meet(self, other: ImportConstraint) -> ImportConstraint:
        """Compute the meet (conjunction) of two import constraints.

        Meet combines constraints:
        - required = union of both required sets
        - forbidden = union of both forbidden sets
        - available = intersection of both available sets

        If required ∩ forbidden ≠ ∅ after meet, return BOTTOM.

        Args:
            other: The constraint to meet with

        Returns:
            The combined constraint
        """
        if self._is_bottom or other._is_bottom:
            return IMPORT_BOTTOM

        if self._is_top:
            return other
        if other._is_top:
            return self

        # Combine required sets
        combined_required = self.required | other.required
        # Combine forbidden sets
        combined_forbidden = self.forbidden | other.forbidden
        # Intersect available sets
        combined_available = self.available & other.available

        # Check for conflict: any required module is forbidden?
        required_names = {m.name for m in combined_required}
        if required_names & combined_forbidden:
            return IMPORT_BOTTOM

        return ImportConstraint(
            required=combined_required,
            forbidden=combined_forbidden,
            available=combined_available,
        )

    def is_top(self) -> bool:
        """Check if this is the TOP constraint (no restrictions)."""
        return self._is_top

    def is_bottom(self) -> bool:
        """Check if this is the BOTTOM constraint (unsatisfiable)."""
        return self._is_bottom

    def satisfiability(self) -> Satisfiability:
        """Check satisfiability of the import constraint.

        Returns:
            UNSAT if required ∩ forbidden ≠ ∅
            SAT otherwise
        """
        if self._is_bottom:
            return Satisfiability.UNSAT

        # Check for conflict
        required_names = {m.name for m in self.required}
        if required_names & self.forbidden:
            return Satisfiability.UNSAT

        return Satisfiability.SAT

    def requires(self, module: ModuleSpec) -> ImportConstraint:
        """Create a new constraint requiring an additional module.

        Args:
            module: The module to require

        Returns:
            New constraint with module required
        """
        if self._is_bottom:
            return self
        if module.name in self.forbidden:
            return IMPORT_BOTTOM

        return ImportConstraint(
            required=self.required | {module},
            forbidden=self.forbidden,
            available=self.available,
        )

    def forbids(self, module_name: str) -> ImportConstraint:
        """Create a new constraint forbidding a module.

        Args:
            module_name: Name of the module to forbid

        Returns:
            New constraint with module forbidden
        """
        if self._is_bottom:
            return self

        # Check if already required
        required_names = {m.name for m in self.required}
        if module_name in required_names:
            return IMPORT_BOTTOM

        return ImportConstraint(
            required=self.required,
            forbidden=self.forbidden | {module_name},
            available=self.available,
        )

    def with_available(self, module: ModuleSpec) -> ImportConstraint:
        """Create a new constraint with a module marked as available.

        Args:
            module: The module that is now available

        Returns:
            New constraint with module available
        """
        if self._is_bottom:
            return self

        return ImportConstraint(
            required=self.required,
            forbidden=self.forbidden,
            available=self.available | {module},
        )

    def is_required(self, module_name: str) -> bool:
        """Check if a module is required.

        Args:
            module_name: Name to check

        Returns:
            True if the module is required
        """
        return any(m.name == module_name for m in self.required)

    def is_forbidden(self, module_name: str) -> bool:
        """Check if a module is forbidden.

        Args:
            module_name: Name to check

        Returns:
            True if the module is forbidden
        """
        return module_name in self.forbidden

    def is_available(self, module_name: str) -> bool:
        """Check if a module is available.

        Args:
            module_name: Name to check

        Returns:
            True if the module is available
        """
        return any(m.name == module_name for m in self.available)

    def get_available(self, module_name: str) -> Optional[ModuleSpec]:
        """Get the spec for an available module.

        Args:
            module_name: Name to look up

        Returns:
            ModuleSpec if available, None otherwise
        """
        for m in self.available:
            if m.name == module_name:
                return m
        return None

    def missing_requirements(self) -> FrozenSet[ModuleSpec]:
        """Get required modules that are not yet available.

        Returns:
            Set of required but unavailable modules
        """
        available_names = {m.name for m in self.available}
        return frozenset(m for m in self.required if m.name not in available_names)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImportConstraint):
            return NotImplemented
        return (
            self._is_top == other._is_top
            and self._is_bottom == other._is_bottom
            and self.required == other.required
            and self.forbidden == other.forbidden
            and self.available == other.available
        )

    def __hash__(self) -> int:
        if self._is_top:
            return hash("IMPORT_TOP")
        if self._is_bottom:
            return hash("IMPORT_BOTTOM")
        return hash((self.required, self.forbidden, self.available))

    def __repr__(self) -> str:
        if self._is_top:
            return "IMPORT_TOP"
        if self._is_bottom:
            return "IMPORT_BOTTOM"
        parts = []
        if self.required:
            parts.append(f"required={len(self.required)}")
        if self.forbidden:
            parts.append(f"forbidden={len(self.forbidden)}")
        if self.available:
            parts.append(f"available={len(self.available)}")
        return f"ImportConstraint({', '.join(parts)})"


# Singleton instances
IMPORT_TOP = ImportConstraint(_is_top=True)
IMPORT_BOTTOM = ImportConstraint(_is_bottom=True)


# Factory functions
def import_requiring(*modules: str) -> ImportConstraint:
    """Create a constraint requiring the given modules.

    Args:
        *modules: Module names to require

    Returns:
        ImportConstraint requiring those modules
    """
    specs = frozenset(ModuleSpec(name=m) for m in modules)
    return ImportConstraint(required=specs)


def import_forbidding(*modules: str) -> ImportConstraint:
    """Create a constraint forbidding the given modules.

    Args:
        *modules: Module names to forbid

    Returns:
        ImportConstraint forbidding those modules
    """
    return ImportConstraint(forbidden=frozenset(modules))
