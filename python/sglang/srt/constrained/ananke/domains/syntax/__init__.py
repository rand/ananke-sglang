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
"""Syntax domain for grammar-based structural constraints.

This domain wraps llguidance for efficient CFG parsing and constraint checking.
"""

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from .constraint import (
        SYNTAX_BOTTOM,
        SYNTAX_TOP,
        GrammarType,
        SyntaxConstraint,
        syntax_from_ebnf,
        syntax_from_json_schema,
        syntax_from_regex,
        syntax_from_structural_tag,
    )
    from .domain import SyntaxDomain
except ImportError:
    from domains.syntax.constraint import (
        SYNTAX_BOTTOM,
        SYNTAX_TOP,
        GrammarType,
        SyntaxConstraint,
        syntax_from_ebnf,
        syntax_from_json_schema,
        syntax_from_regex,
        syntax_from_structural_tag,
    )
    from domains.syntax.domain import SyntaxDomain

__all__ = [
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
