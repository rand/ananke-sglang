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
"""Marked lambda calculus implementation (POPL 2024).

This module implements the marking system from "Total Type Error Localization
and Recovery with Holes" (POPL 2024). Marks localize type errors to specific
locations while allowing type checking to continue.

Key concepts:
- HoleMark: Empty hole awaiting a term
- InconsistentMark: Type mismatch between synthesized and expected
- Provenance: Traces errors back to their source
- MarkedAST: AST with marks at each node
- Totalization: Assigns types to ALL partial programs

References:
    - POPL 2024: "Total Type Error Localization and Recovery with Holes"
    - Hazel: https://hazel.org
"""

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from .marks import (
        HoleMark,
        InconsistentMark,
        Mark,
        NonEmptyHoleMark,
    )
    from .provenance import (
        Provenance,
        SourceSpan,
    )
    from .marked_ast import (
        MarkedAST,
        MarkedASTNode,
    )
    from .totalization import (
        totalize,
        TotalizationResult,
    )
except ImportError:
    from domains.types.marking.marks import (
        HoleMark,
        InconsistentMark,
        Mark,
        NonEmptyHoleMark,
    )
    from domains.types.marking.provenance import (
        Provenance,
        SourceSpan,
    )
    from domains.types.marking.marked_ast import (
        MarkedAST,
        MarkedASTNode,
    )
    from domains.types.marking.totalization import (
        totalize,
        TotalizationResult,
    )

__all__ = [
    # Marks
    "Mark",
    "HoleMark",
    "InconsistentMark",
    "NonEmptyHoleMark",
    # Provenance
    "Provenance",
    "SourceSpan",
    # Marked AST
    "MarkedAST",
    "MarkedASTNode",
    # Totalization
    "totalize",
    "TotalizationResult",
]
