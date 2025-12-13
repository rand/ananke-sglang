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
"""Core constraint algebra and domain abstractions for Ananke."""

from .constraint import (
    BOTTOM,
    TOP,
    BottomConstraint,
    Constraint,
    Satisfiability,
    TopConstraint,
    verify_semilattice_laws,
)
from .domain import (
    ConstraintDomain,
    GenerationContext,
    PassthroughDomain,
)
from .unified import (
    UNIFIED_BOTTOM,
    UNIFIED_TOP,
    UnifiedConstraint,
    unified_from_controlflow,
    unified_from_imports,
    unified_from_semantics,
    unified_from_syntax,
    unified_from_types,
)
from .checkpoint import (
    Checkpoint,
    CheckpointManager,
    UnifiedCheckpoint,
    create_context_snapshot,
)

__all__ = [
    # Constraint algebra
    "Constraint",
    "Satisfiability",
    "TopConstraint",
    "BottomConstraint",
    "TOP",
    "BOTTOM",
    "verify_semilattice_laws",
    # Domains
    "ConstraintDomain",
    "GenerationContext",
    "PassthroughDomain",
    # Unified constraint
    "UnifiedConstraint",
    "UNIFIED_TOP",
    "UNIFIED_BOTTOM",
    "unified_from_syntax",
    "unified_from_types",
    "unified_from_imports",
    "unified_from_controlflow",
    "unified_from_semantics",
    # Checkpointing
    "Checkpoint",
    "UnifiedCheckpoint",
    "CheckpointManager",
    "create_context_snapshot",
]
