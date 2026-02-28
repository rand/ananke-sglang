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
"""Import domain for module/package availability constraints.

This module provides:
- ImportConstraint: Constraint tracking required/forbidden/available modules
- ImportDomain: Domain for detecting and validating imports
- ModuleSpec: Specification for a module (name, version, alias)
- Import resolvers for language-specific module resolution
"""

from __future__ import annotations

from .constraint import (
    ImportConstraint,
    ModuleSpec,
    IMPORT_TOP,
    IMPORT_BOTTOM,
    import_requiring,
    import_forbidding,
)
from .domain import ImportDomain, ImportDomainCheckpoint

__all__ = [
    # Constraint types
    "ImportConstraint",
    "ModuleSpec",
    "IMPORT_TOP",
    "IMPORT_BOTTOM",
    # Factory functions
    "import_requiring",
    "import_forbidding",
    # Domain
    "ImportDomain",
    "ImportDomainCheckpoint",
]
