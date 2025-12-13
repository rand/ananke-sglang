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
"""Import domain for tracking module/package constraints.

The ImportDomain tracks:
- Which modules have been imported in generated code
- Which modules are required but not yet imported
- Which modules are forbidden

It can detect import statements in the token stream and update
the constraint accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import torch

try:
    from ...core.constraint import Satisfiability
    from ...core.domain import ConstraintDomain, GenerationContext
    from .constraint import (
        IMPORT_TOP,
        IMPORT_BOTTOM,
        ImportConstraint,
        ModuleSpec,
    )
except ImportError:
    from core.constraint import Satisfiability
    from core.domain import ConstraintDomain, GenerationContext
    from domains.imports.constraint import (
        IMPORT_TOP,
        IMPORT_BOTTOM,
        ImportConstraint,
        ModuleSpec,
    )


@dataclass
class ImportDomainCheckpoint:
    """Checkpoint for ImportDomain state.

    Attributes:
        imported_modules: Set of imported module names
        import_buffer: Current import statement being built
        state_counter: State counter value
    """

    imported_modules: Set[str]
    import_buffer: str
    state_counter: int


class ImportDomain(ConstraintDomain[ImportConstraint]):
    """Import domain for module/package constraint tracking.

    The import domain:
    1. Detects import statements in generated code
    2. Tracks which modules are available
    3. Provides token masks based on import constraints
    4. Updates constraints when imports are detected

    Example:
        >>> domain = ImportDomain(language="python")
        >>> constraint = domain.create_constraint(required=["numpy"])
        >>> # As code is generated, constraint updates with available imports
    """

    def __init__(self, language: str = "python"):
        """Initialize the import domain.

        Args:
            language: Programming language (affects import detection)
        """
        self._language = language
        self._imported_modules: Set[str] = set()
        self._import_buffer: str = ""
        self._state_counter = 0

    @property
    def name(self) -> str:
        """Return the domain name."""
        return "imports"

    @property
    def top(self) -> ImportConstraint:
        """Return the TOP constraint (no restrictions)."""
        return IMPORT_TOP

    @property
    def bottom(self) -> ImportConstraint:
        """Return the BOTTOM constraint (unsatisfiable)."""
        return IMPORT_BOTTOM

    @property
    def language(self) -> str:
        """Return the target language."""
        return self._language

    @property
    def imported_modules(self) -> Set[str]:
        """Return set of currently imported modules."""
        return self._imported_modules.copy()

    def token_mask(
        self,
        constraint: ImportConstraint,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Compute a token mask based on import constraints.

        If a module is forbidden, tokens that would complete an import
        of that module may be blocked.

        Current implementation returns all True (conservative).
        Full implementation would analyze partial import statements.

        Args:
            constraint: Current import constraint
            context: Generation context

        Returns:
            Boolean tensor of valid tokens
        """
        # Handle TOP/BOTTOM
        if constraint.is_top():
            return torch.ones(context.vocab_size, dtype=torch.bool, device=context.device)
        if constraint.is_bottom():
            return torch.zeros(context.vocab_size, dtype=torch.bool, device=context.device)

        # Conservative: allow all tokens
        # Full implementation would block tokens that would import forbidden modules
        return torch.ones(context.vocab_size, dtype=torch.bool, device=context.device)

    def observe_token(
        self,
        constraint: ImportConstraint,
        token_id: int,
        context: GenerationContext,
    ) -> ImportConstraint:
        """Update the import constraint after observing a token.

        Detects import statements and updates the constraint with
        newly imported modules.

        Args:
            constraint: Current import constraint
            token_id: The generated token
            context: Generation context

        Returns:
            Updated import constraint
        """
        if constraint.is_top() or constraint.is_bottom():
            return constraint

        self._state_counter += 1

        # Get token text
        token_text = ""
        if context.tokenizer is not None:
            try:
                token_text = context.tokenizer.decode([token_id])
            except Exception:
                pass

        # Try to detect imports
        updated_constraint = self._detect_imports(constraint, token_text, context)

        return updated_constraint

    def _detect_imports(
        self,
        constraint: ImportConstraint,
        token_text: str,
        context: GenerationContext,
    ) -> ImportConstraint:
        """Detect import statements and update constraint.

        Args:
            constraint: Current constraint
            token_text: New token text
            context: Generation context

        Returns:
            Updated constraint if import detected
        """
        # Accumulate tokens in buffer for import detection
        self._import_buffer += token_text

        # Check for complete import statement
        if self._language == "python":
            constraint = self._detect_python_imports(constraint)
        elif self._language == "typescript":
            constraint = self._detect_typescript_imports(constraint)
        # Other languages can be added

        return constraint

    def _detect_python_imports(self, constraint: ImportConstraint) -> ImportConstraint:
        """Detect Python import statements.

        Handles:
        - import module
        - import module as alias
        - from module import name
        - from module import name as alias

        Args:
            constraint: Current constraint

        Returns:
            Updated constraint if import detected
        """
        # Check for newline/semicolon before stripping (they indicate statement end)
        has_terminator = "\n" in self._import_buffer or ";" in self._import_buffer
        buffer = self._import_buffer.strip()

        # Check for simple "import X" pattern
        if buffer.startswith("import "):
            # Look for newline or semicolon to indicate end
            if has_terminator:
                # Extract module name
                import_part = buffer.split("\n")[0].split(";")[0]
                import_part = import_part[7:].strip()  # Remove "import "

                # Handle "import X as Y"
                parts = import_part.split(" as ")
                module_name = parts[0].strip()
                alias = parts[1].strip() if len(parts) > 1 else None

                # Handle multi-import "import X, Y, Z"
                for name in module_name.split(","):
                    name = name.strip()
                    if name:
                        self._imported_modules.add(name)
                        spec = ModuleSpec(name=name, alias=alias)
                        constraint = constraint.with_available(spec)

                # Clear buffer after processing
                self._import_buffer = ""

        # Check for "from X import Y" pattern
        elif buffer.startswith("from "):
            if has_terminator:
                # Extract module name
                import_part = buffer.split("\n")[0].split(";")[0]
                # from X import Y
                if " import " in import_part:
                    parts = import_part.split(" import ")
                    module_name = parts[0][5:].strip()  # Remove "from "

                    self._imported_modules.add(module_name)
                    spec = ModuleSpec(name=module_name)
                    constraint = constraint.with_available(spec)

                # Clear buffer
                self._import_buffer = ""

        # Limit buffer size to prevent memory issues
        if len(self._import_buffer) > 1000:
            self._import_buffer = self._import_buffer[-500:]

        return constraint

    def _detect_typescript_imports(self, constraint: ImportConstraint) -> ImportConstraint:
        """Detect TypeScript/JavaScript import statements.

        Handles:
        - import X from 'module'
        - import { X } from 'module'
        - import * as X from 'module'

        Args:
            constraint: Current constraint

        Returns:
            Updated constraint if import detected
        """
        # Check for newline/semicolon before stripping (they indicate statement end)
        has_terminator = "\n" in self._import_buffer or ";" in self._import_buffer
        buffer = self._import_buffer.strip()

        # Check for "import ... from 'module'" pattern
        if buffer.startswith("import ") and " from " in buffer:
            if has_terminator:
                # Extract module name from quotes
                import_part = buffer.split("\n")[0].split(";")[0]

                # Find the 'module' or "module" part
                for quote in ["'", '"']:
                    if quote in import_part:
                        parts = import_part.split(quote)
                        if len(parts) >= 2:
                            module_name = parts[1]
                            self._imported_modules.add(module_name)
                            spec = ModuleSpec(name=module_name)
                            constraint = constraint.with_available(spec)
                            break

                self._import_buffer = ""

        if len(self._import_buffer) > 1000:
            self._import_buffer = self._import_buffer[-500:]

        return constraint

    def checkpoint(self) -> ImportDomainCheckpoint:
        """Create a checkpoint of the current state.

        Returns:
            Checkpoint for restoration
        """
        return ImportDomainCheckpoint(
            imported_modules=self._imported_modules.copy(),
            import_buffer=self._import_buffer,
            state_counter=self._state_counter,
        )

    def restore(self, checkpoint: Any) -> None:
        """Restore state from a checkpoint.

        Args:
            checkpoint: Previously created checkpoint
        """
        if not isinstance(checkpoint, ImportDomainCheckpoint):
            raise TypeError(
                f"Expected ImportDomainCheckpoint, got {type(checkpoint).__name__}"
            )
        self._imported_modules = checkpoint.imported_modules.copy()
        self._import_buffer = checkpoint.import_buffer
        self._state_counter = checkpoint.state_counter

    def satisfiability(self, constraint: ImportConstraint) -> Satisfiability:
        """Check satisfiability of an import constraint.

        Args:
            constraint: The constraint to check

        Returns:
            Satisfiability status
        """
        return constraint.satisfiability()

    def create_constraint(
        self,
        required: Optional[List[str]] = None,
        forbidden: Optional[List[str]] = None,
    ) -> ImportConstraint:
        """Create an import constraint.

        Args:
            required: List of required module names
            forbidden: List of forbidden module names

        Returns:
            New ImportConstraint
        """
        if required is None and forbidden is None:
            return IMPORT_TOP

        req_specs = frozenset(
            ModuleSpec(name=m) for m in (required or [])
        )
        forb = frozenset(forbidden or [])

        # Check for conflict
        req_names = {m.name for m in req_specs}
        if req_names & forb:
            return IMPORT_BOTTOM

        return ImportConstraint(
            required=req_specs,
            forbidden=forb,
        )

    def add_available(self, module: str, alias: Optional[str] = None) -> None:
        """Mark a module as available (imported).

        Args:
            module: Module name
            alias: Optional alias
        """
        self._imported_modules.add(module)

    def is_imported(self, module: str) -> bool:
        """Check if a module has been imported.

        Args:
            module: Module name

        Returns:
            True if imported
        """
        return module in self._imported_modules
