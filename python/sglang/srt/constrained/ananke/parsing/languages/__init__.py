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
"""Language-specific incremental parsers.

This package provides incremental parser implementations for each
supported programming language.

Currently Supported:
    - Python (PythonIncrementalParser)
    - Zig (ZigIncrementalParser) - with comptime and error union support
    - Rust (RustIncrementalParser) - with lifetime and ownership tracking

Planned:
    - TypeScript
    - Go
"""

from parsing.languages.python import (
    PythonIncrementalParser,
    create_python_parser,
)
from parsing.languages.zig import (
    ZigIncrementalParser,
    create_zig_parser,
)
from parsing.languages.rust import (
    RustIncrementalParser,
    create_rust_parser,
)


__all__ = [
    # Python
    "PythonIncrementalParser",
    "create_python_parser",
    # Zig
    "ZigIncrementalParser",
    "create_zig_parser",
    # Rust
    "RustIncrementalParser",
    "create_rust_parser",
]
