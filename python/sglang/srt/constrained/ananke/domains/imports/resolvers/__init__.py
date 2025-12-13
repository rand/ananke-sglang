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
"""Language-specific import resolvers for Python, TypeScript, Rust, Zig, Go.

This module provides import resolution for various languages:
- PythonImportResolver: Python stdlib and pip packages
- PassthroughResolver: Testing resolver that always succeeds
- DenyListResolver: Security-focused resolver that blocks specific modules
"""

from __future__ import annotations

from .base import (
    ImportResolver,
    ImportResolution,
    ResolvedModule,
    PassthroughResolver,
    DenyListResolver,
)
from .python import (
    PythonImportResolver,
    PYTHON_STDLIB,
    create_python_resolver,
)

__all__ = [
    # Base types
    "ImportResolver",
    "ImportResolution",
    "ResolvedModule",
    # Utility resolvers
    "PassthroughResolver",
    "DenyListResolver",
    # Python resolver
    "PythonImportResolver",
    "PYTHON_STDLIB",
    "create_python_resolver",
]
