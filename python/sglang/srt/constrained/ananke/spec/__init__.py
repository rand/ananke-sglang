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
"""Constraint specification package for Ananke.

This package provides rich constraint specification support for the Ananke
constrained generation system, enabling:
- Per-request language configuration with auto-detection
- External constraint data (type environments, imports, control flow, semantic)
- Multiple input formats (JSON, URI, binary)
- Efficient caching with explicit scope control

Key Components:
    ConstraintSpec: Unified internal representation for constraint specifications
    ConstraintSpecParser: Multi-format parser supporting JSON, URI, and binary
    LanguageDetector: Tree-sitter based language detection for polyglot support

Example:
    >>> from sglang.srt.constrained.ananke.spec import ConstraintSpec, CacheScope
    >>> spec = ConstraintSpec(
    ...     json_schema='{"type": "object"}',
    ...     language="python",
    ...     type_bindings=[TypeBinding(name="x", type_expr="int")],
    ...     cache_scope=CacheScope.SYNTAX_AND_LANG,
    ... )
"""

from __future__ import annotations

from .constraint_spec import (
    # Enums
    CacheScope,
    ConstraintSource,
    LanguageDetection,
    # Core specification
    ConstraintSpec,
    # Type context
    TypeBinding,
    FunctionSignature,
    ClassDefinition,
    # Import context
    ImportBinding,
    ModuleStub,
    # Control flow context
    ControlFlowContext,
    # Semantic constraints
    SemanticConstraint,
    # Language support
    LanguageFrame,
)

from .parser import ConstraintSpecParser

from .language_detector import (
    DetectionResult,
    LanguageDetector,
    LanguageStackManager,
)

__all__ = [
    # Enums
    "CacheScope",
    "ConstraintSource",
    "LanguageDetection",
    # Core specification
    "ConstraintSpec",
    # Type context
    "TypeBinding",
    "FunctionSignature",
    "ClassDefinition",
    # Import context
    "ImportBinding",
    "ModuleStub",
    # Control flow context
    "ControlFlowContext",
    # Semantic constraints
    "SemanticConstraint",
    # Language support
    "LanguageFrame",
    # Parser
    "ConstraintSpecParser",
    # Language detection
    "DetectionResult",
    "LanguageDetector",
    "LanguageStackManager",
]
