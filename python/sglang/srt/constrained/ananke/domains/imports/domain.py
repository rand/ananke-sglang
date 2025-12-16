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
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

import torch

try:
    from ...core.constraint import Satisfiability
    from ...core.domain import ConstraintDomain, GenerationContext
    from ...core.token_classifier import (
        TokenClassifier,
        TokenCategory,
        get_or_create_classifier,
        PYTHON_IMPORT_KEYWORDS,
    )
    from .constraint import (
        IMPORT_TOP,
        IMPORT_BOTTOM,
        ImportConstraint,
        ModuleSpec,
    )
except ImportError:
    from core.constraint import Satisfiability
    from core.domain import ConstraintDomain, GenerationContext
    from core.token_classifier import (
        TokenClassifier,
        TokenCategory,
        get_or_create_classifier,
        PYTHON_IMPORT_KEYWORDS,
    )
    from domains.imports.constraint import (
        IMPORT_TOP,
        IMPORT_BOTTOM,
        ImportConstraint,
        ModuleSpec,
    )


class ImportContext(Enum):
    """Current position within an import statement.

    Tracks the state machine for import detection:
    - NONE: Not in an import statement
    - IMPORT_KEYWORD: Just saw 'import' keyword
    - FROM_KEYWORD: Just saw 'from' keyword
    - MODULE_NAME: In module name position (after import/from)
    - IMPORT_AFTER_FROM: After 'from X import' keyword
    """

    NONE = auto()
    IMPORT_KEYWORD = auto()
    FROM_KEYWORD = auto()
    MODULE_NAME = auto()
    IMPORT_AFTER_FROM = auto()


@dataclass
class ImportDomainCheckpoint:
    """Checkpoint for ImportDomain state.

    Attributes:
        imported_modules: Set of imported module names
        import_buffer: Current import statement being built
        state_counter: State counter value
        import_context: Current import context state
        partial_module: Partial module name being typed
    """

    imported_modules: Set[str]
    import_buffer: str
    state_counter: int
    import_context: ImportContext = ImportContext.NONE
    partial_module: str = ""


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

    def __init__(self, language: str = "python", tokenizer: Optional[Any] = None):
        """Initialize the import domain.

        Args:
            language: Programming language (affects import detection)
            tokenizer: Optional tokenizer for precise masking
        """
        self._language = language
        self._imported_modules: Set[str] = set()
        self._import_buffer: str = ""
        self._state_counter = 0

        # Import context state machine
        self._import_context = ImportContext.NONE
        self._partial_module = ""

        # Lazy-initialized classifier
        self._tokenizer = tokenizer
        self._classifier: Optional[TokenClassifier] = None

        # Precomputed token sets for common modules (populated on init)
        self._module_name_tokens: Dict[str, Set[int]] = {}

    def _ensure_classifier_initialized(self, context: GenerationContext) -> None:
        """Ensure classifier is initialized.

        Args:
            context: Generation context with tokenizer
        """
        tokenizer = context.tokenizer or self._tokenizer
        if tokenizer is None:
            return

        if self._classifier is None:
            self._classifier = get_or_create_classifier(tokenizer, self._language)

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

        If a module is forbidden and we're in an import context,
        tokens that would complete an import of that module are blocked.

        The mask is only restrictive when:
        1. We're in an import statement (after 'import' or 'from')
        2. The constraint has forbidden modules
        3. The partial module name could match a forbidden module

        Performance target: <50μs typical, <200μs worst case.

        Args:
            constraint: Current import constraint
            context: Generation context

        Returns:
            Boolean tensor of valid tokens
        """
        # Handle TOP/BOTTOM
        if constraint.is_top():
            return context.create_mask(fill_value=True)
        if constraint.is_bottom():
            return context.create_mask(fill_value=False)

        # If no forbidden modules, allow all
        if not constraint.forbidden:
            return context.create_mask(fill_value=True)

        # If not in import context, allow all
        if self._import_context == ImportContext.NONE:
            return context.create_mask(fill_value=True)

        # Ensure classifier is initialized
        self._ensure_classifier_initialized(context)

        # Create base mask (all True)
        mask = context.create_mask(fill_value=True)

        # Only apply blocking if we're in module name position
        if self._import_context in (ImportContext.IMPORT_KEYWORD, ImportContext.FROM_KEYWORD, ImportContext.MODULE_NAME):
            mask = self._apply_forbidden_module_blocking(mask, constraint, context)

        return mask

    def _apply_forbidden_module_blocking(
        self,
        mask: torch.Tensor,
        constraint: ImportConstraint,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Block tokens that would complete a forbidden import.

        Args:
            mask: Current mask to modify
            constraint: Constraint with forbidden modules
            context: Generation context

        Returns:
            Modified mask
        """
        if self._classifier is None:
            return mask

        # For each forbidden module, check if partial matches
        for forbidden in constraint.forbidden:
            # Check if current partial could lead to this forbidden module
            if self._could_match_forbidden(forbidden):
                # Block tokens that would continue toward this module
                self._block_module_completion_tokens(mask, forbidden, context)

        return mask

    def _could_match_forbidden(self, forbidden: str) -> bool:
        """Check if current partial module name could match forbidden module.

        Args:
            forbidden: Forbidden module name (e.g., "subprocess")

        Returns:
            True if partial could lead to forbidden
        """
        if not self._partial_module:
            # Empty partial - any forbidden module could match
            return True

        # Check if forbidden module starts with partial
        return forbidden.startswith(self._partial_module)

    def _block_module_completion_tokens(
        self,
        mask: torch.Tensor,
        forbidden: str,
        context: GenerationContext,
    ) -> None:
        """Block tokens that would complete import of forbidden module.

        Args:
            mask: Mask to modify (in place)
            forbidden: Forbidden module name
            context: Generation context
        """
        if self._classifier is None:
            return

        # Calculate what continuation would complete the forbidden import
        if self._partial_module:
            # Need to complete from partial to forbidden
            if forbidden.startswith(self._partial_module):
                continuation = forbidden[len(self._partial_module):]
            else:
                return  # Partial doesn't match, nothing to block
        else:
            # Need full module name
            continuation = forbidden

        # Block tokens that are the forbidden module name or start it
        # Check all identifier tokens
        for token_id in self._classifier.by_category(TokenCategory.IDENTIFIER):
            if token_id >= context.vocab_size:
                continue

            tc = self._classifier.get_classification(token_id)
            token_text = tc.text.strip()

            # Block if token would complete or continue toward forbidden
            if continuation.startswith(token_text) or token_text.startswith(continuation):
                # This token would help complete the forbidden import
                mask[token_id] = False

            # Also block exact match
            if token_text == forbidden or token_text == continuation:
                mask[token_id] = False

    def _update_import_context(self, token_text: str) -> None:
        """Update the import context state machine.

        Args:
            token_text: The token just observed
        """
        stripped = token_text.strip()

        if self._import_context == ImportContext.NONE:
            if stripped == "import":
                self._import_context = ImportContext.IMPORT_KEYWORD
                self._partial_module = ""
            elif stripped == "from":
                self._import_context = ImportContext.FROM_KEYWORD
                self._partial_module = ""

        elif self._import_context == ImportContext.IMPORT_KEYWORD:
            if stripped and stripped.isidentifier():
                # This is a module name
                self._partial_module += stripped
                self._import_context = ImportContext.MODULE_NAME
            elif stripped == ".":
                # Dot in module path
                self._partial_module += "."
            elif stripped in ("\n", ";", ","):
                # End of import or next module
                self._import_context = ImportContext.NONE
                self._partial_module = ""

        elif self._import_context == ImportContext.FROM_KEYWORD:
            if stripped and (stripped.isidentifier() or stripped.startswith(".")):
                # Module name after 'from'
                self._partial_module += stripped
                self._import_context = ImportContext.MODULE_NAME
            elif stripped == "import":
                # 'from X import' - now in import names
                self._import_context = ImportContext.IMPORT_AFTER_FROM
            elif stripped in ("\n", ";"):
                self._import_context = ImportContext.NONE
                self._partial_module = ""

        elif self._import_context == ImportContext.MODULE_NAME:
            if stripped == ".":
                # Continuing module path
                self._partial_module += "."
            elif stripped and stripped.isidentifier():
                # More of the module name
                self._partial_module += stripped
            elif stripped == "import":
                # 'from X import' transition
                self._import_context = ImportContext.IMPORT_AFTER_FROM
            elif stripped in ("\n", ";", ",", "as"):
                # End of this import
                self._import_context = ImportContext.NONE
                self._partial_module = ""

        elif self._import_context == ImportContext.IMPORT_AFTER_FROM:
            if stripped in ("\n", ";"):
                self._import_context = ImportContext.NONE
                self._partial_module = ""

    def observe_token(
        self,
        constraint: ImportConstraint,
        token_id: int,
        context: GenerationContext,
    ) -> ImportConstraint:
        """Update the import constraint after observing a token.

        Detects import statements and updates the constraint with
        newly imported modules. Also updates the import context
        state machine for precise masking.

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

        # Update import context state machine
        self._update_import_context(token_text)

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
            import_context=self._import_context,
            partial_module=self._partial_module,
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
        self._import_context = checkpoint.import_context
        self._partial_module = checkpoint.partial_module

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

    def add_import(
        self,
        module: str,
        name: Optional[str] = None,
        alias: Optional[str] = None,
        is_wildcard: bool = False,
    ) -> None:
        """Add an import to the available imports.

        Args:
            module: Module name (e.g., "numpy", "typing")
            name: Specific name imported (e.g., "List" from "typing")
            alias: Alias for the import (e.g., "np" for "numpy")
            is_wildcard: Whether this is a wildcard import
        """
        self._imported_modules.add(module)
        if name:
            # Track specific imports like "from typing import List"
            self._imported_modules.add(f"{module}.{name}")

    def set_available_modules(self, modules: Set[str]) -> None:
        """Set the complete set of available modules.

        This replaces any existing available modules.

        Args:
            modules: Set of module names that are available
        """
        self._imported_modules = modules.copy()

    def set_forbidden_imports(self, forbidden: Set[str]) -> ImportConstraint:
        """Create a constraint with forbidden imports.

        Args:
            forbidden: Set of module names that are forbidden

        Returns:
            ImportConstraint with the forbidden modules
        """
        return self.create_constraint(forbidden=list(forbidden))

    def inject_context(self, spec: Any) -> None:
        """Inject context from a ConstraintSpec.

        Called when a cached grammar object needs fresh context.
        This re-seeds the import state with data from the spec.

        Args:
            spec: A ConstraintSpec object (typed as Any to avoid circular import)
        """
        # Import locally to avoid circular dependency
        # Try relative import first, fall back to absolute
        try:
            from ...spec.constraint_spec import ConstraintSpec
        except ImportError:
            try:
                from spec.constraint_spec import ConstraintSpec
            except ImportError:
                # If we can't import, check by class name
                if spec.__class__.__name__ != "ConstraintSpec":
                    return
                ConstraintSpec = spec.__class__

        if not isinstance(spec, ConstraintSpec):
            return

        # Clear existing imports
        self._imported_modules.clear()

        # Add imports from spec
        for import_binding in spec.imports:
            self.add_import(
                module=import_binding.module,
                name=import_binding.name,
                alias=import_binding.alias,
                is_wildcard=import_binding.is_wildcard,
            )

        # Add available modules
        for module in spec.available_modules:
            self._imported_modules.add(module)

    def seed_imports(
        self,
        imports: List[Tuple[str, Optional[str], Optional[str], bool]],
    ) -> None:
        """Seed the import state from a list of import tuples.

        This is the primary method for initializing import context from
        a ConstraintSpec's imports.

        Args:
            imports: List of (module, name, alias, is_wildcard) tuples
        """
        for module, name, alias, is_wildcard in imports:
            self.add_import(
                module=module,
                name=name,
                alias=alias,
                is_wildcard=is_wildcard,
            )
