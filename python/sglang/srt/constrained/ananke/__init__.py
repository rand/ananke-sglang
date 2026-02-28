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
"""Ananke: A compositional constraint system for verified code generation.

Ananke extends SGLang's constrained decoding to support multi-domain constraint
fusion across syntax, types, imports, control flow, and semantics. The system
treats code generation as a constraint satisfaction problem where each token
must satisfy all active constraints.

Key Components:
    - core: Constraint algebra, domains, and unified constraints
    - domains: Syntax, types, imports, control flow, semantics
    - propagation: Cross-domain constraint flow
    - holes: Typed hole management with fill-and-resume
    - masks: Token mask computation and fusion
    - parsing: Incremental parsing per language
    - backend: SGLang integration (AnankeBackend, AnankeGrammar)

Usage:
    Start SGLang server with --grammar-backend=ananke to enable Ananke.

References:
    - Hazel: "Statically Contextualizing LLMs with Typed Holes" (OOPSLA 2024)
    - Hazel: "Total Type Error Localization and Recovery with Holes" (POPL 2024)
    - Hazel: "Incremental Bidirectional Typing via Order Maintenance" (OOPSLA 2025)
"""

# Use lazy imports to allow standalone testing of submodules
# Full imports are done on first access via __getattr__


def __getattr__(name: str):
    """Lazy import of module attributes."""
    # Core constraint algebra
    if name in (
        "BOTTOM",
        "TOP",
        "BottomConstraint",
        "Constraint",
        "Satisfiability",
        "TopConstraint",
        "verify_semilattice_laws",
    ):
        from .core.constraint import (
            BOTTOM,
            TOP,
            BottomConstraint,
            Constraint,
            Satisfiability,
            TopConstraint,
            verify_semilattice_laws,
        )

        return locals()[name]

    # Domain abstractions
    if name in ("ConstraintDomain", "GenerationContext", "PassthroughDomain"):
        from .core.domain import (
            ConstraintDomain,
            GenerationContext,
            PassthroughDomain,
        )

        return locals()[name]

    # Unified constraint
    if name in ("UNIFIED_BOTTOM", "UNIFIED_TOP", "UnifiedConstraint"):
        from .core.unified import (
            UNIFIED_BOTTOM,
            UNIFIED_TOP,
            UnifiedConstraint,
        )

        return locals()[name]

    # Checkpointing
    if name in ("Checkpoint", "CheckpointManager", "UnifiedCheckpoint"):
        from .core.checkpoint import (
            Checkpoint,
            CheckpointManager,
            UnifiedCheckpoint,
        )

        return locals()[name]

    # Backend (requires full sglang)
    if name == "AnankeGrammar":
        from .backend.grammar import AnankeGrammar

        return AnankeGrammar

    if name in ("AnankeBackend", "create_ananke_backend"):
        from .backend.backend import AnankeBackend, create_ananke_backend

        return locals()[name]

    # Syntax domain
    if name in (
        "SYNTAX_BOTTOM",
        "SYNTAX_TOP",
        "GrammarType",
        "SyntaxConstraint",
        "SyntaxDomain",
        "syntax_from_ebnf",
        "syntax_from_json_schema",
        "syntax_from_regex",
        "syntax_from_structural_tag",
    ):
        from .domains.syntax import (
            SYNTAX_BOTTOM,
            SYNTAX_TOP,
            GrammarType,
            SyntaxConstraint,
            SyntaxDomain,
            syntax_from_ebnf,
            syntax_from_json_schema,
            syntax_from_regex,
            syntax_from_structural_tag,
        )

        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Constraint algebra
    "Constraint",
    "Satisfiability",
    "TopConstraint",
    "BottomConstraint",
    "TOP",
    "BOTTOM",
    "verify_semilattice_laws",
    # Domain abstractions
    "ConstraintDomain",
    "GenerationContext",
    "PassthroughDomain",
    # Unified constraint
    "UnifiedConstraint",
    "UNIFIED_TOP",
    "UNIFIED_BOTTOM",
    # Checkpointing
    "Checkpoint",
    "UnifiedCheckpoint",
    "CheckpointManager",
    # Backend
    "AnankeGrammar",
    "AnankeBackend",
    "create_ananke_backend",
    # Syntax domain
    "GrammarType",
    "SyntaxConstraint",
    "SYNTAX_TOP",
    "SYNTAX_BOTTOM",
    "syntax_from_json_schema",
    "syntax_from_regex",
    "syntax_from_ebnf",
    "syntax_from_structural_tag",
    "SyntaxDomain",
]

__version__ = "0.1.0"
