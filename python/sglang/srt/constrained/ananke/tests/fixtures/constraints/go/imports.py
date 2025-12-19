# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Go import constraint examples.

Demonstrates import-level constraints specific to Go:
- Internal package path restrictions
- Unsafe package blocking for safe code
- Standard library restrictions for embedded/constrained environments
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        ImportBinding,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        ImportBinding,
    )


GO_IMPORT_EXAMPLES: List[ConstraintExample] = [
    ConstraintExample(
        id="go-imports-001",
        name="Internal Package Path Restrictions",
        description="Enforce Go's internal package visibility rules",
        scenario=(
            "Developer importing packages in a Go project. Go's 'internal' package "
            "convention restricts imports: only code in the parent tree of the "
            "internal directory can import it. E.g., pkg/internal/util can only be "
            "imported by pkg/* packages, not by other top-level packages."
        ),
        prompt="""Write import statements for a Go project. Only import your own internal packages
or public packages from external repos. Don't import other projects' internal packages.

""",
        spec=ConstraintSpec(
            language="go",
            # Regex enforces valid import statements for own internal packages
            regex=r'^import\s+(?:\(\s*)?(?:"github\.com/myapp/|"github\.com/other/project/pkg/public")',
            ebnf=r'''
root ::= single_internal | grouped_internal | external_public
single_internal ::= "import \"github.com/myapp/pkg/internal/util\""
grouped_internal ::= "import (" nl "\t\"github.com/myapp/pkg/handlers\"" nl "\t\"github.com/myapp/pkg/internal/util\"" nl ")"
external_public ::= "import \"github.com/other/project/pkg/public\""
nl ::= "\n"
''',
            forbidden_imports={
                "github.com/other/project/internal/secrets",
                "github.com/other/project/pkg/internal/util",
            },
            available_modules={
                "github.com/myapp/pkg/internal/util",  # Our internal package
                "github.com/myapp/pkg/handlers",
                "github.com/other/project/pkg/public",
            },
        ),
        expected_effect=(
            "Masks import statements for external 'internal' packages. "
            "Allows imports from same-tree internal packages. "
            "Enforces Go's internal package visibility invariant."
        ),
        valid_outputs=[
            'import "github.com/myapp/pkg/internal/util"',
            'import (\n\t"github.com/myapp/pkg/handlers"\n\t"github.com/myapp/pkg/internal/util"\n)',
            'import "github.com/other/project/pkg/public"',
        ],
        invalid_outputs=[
            'import "github.com/other/project/internal/secrets"',  # External internal package
            'import "github.com/other/project/pkg/internal/util"',  # External pkg/internal
            'import (\n\t"fmt"\n\t"github.com/other/project/internal/secrets"\n)',  # Blocked import in group
        ],
        tags=["imports", "internal", "visibility", "packages"],
        language="go",
        domain="imports",
    ),
    ConstraintExample(
        id="go-imports-002",
        name="Unsafe Package Blocking",
        description="Block unsafe package for memory-safe code generation",
        scenario=(
            "Developer writing production Go code where the 'unsafe' package is "
            "forbidden by policy. The unsafe package allows arbitrary pointer "
            "manipulation and breaks Go's type safety and memory safety guarantees. "
            "Code must not import or use unsafe."
        ),
        prompt="""Write import statements for a memory-safe Go project. No unsafe package allowed.
Use only safe standard library packages like fmt, io, context, sync.

""",
        spec=ConstraintSpec(
            language="go",
            # Regex enforces only safe standard library imports (no unsafe)
            regex=r'^import\s+(?:\(\s*)?(?:"fmt"|"io"|"context"|"sync"|"encoding/)',
            ebnf=r'''
root ::= single_fmt | grouped_safe | grouped_io
single_fmt ::= "import \"fmt\""
grouped_safe ::= "import (" nl "\t\"context\"" nl "\t\"encoding/json\"" nl "\t\"sync\"" nl ")"
grouped_io ::= "import (" nl "\t\"io\"" nl "\t\"fmt\"" nl ")"
nl ::= "\n"
''',
            forbidden_imports={"unsafe"},
            available_modules={
                "fmt",
                "io",
                "context",
                "sync",
                "encoding/json",
            },
        ),
        expected_effect=(
            "Masks any import statement containing 'unsafe'. "
            "Blocks unsafe.Pointer, unsafe.Sizeof, and other unsafe operations. "
            "Ensures memory-safe code generation."
        ),
        valid_outputs=[
            'import "fmt"',
            'import (\n\t"context"\n\t"encoding/json"\n\t"sync"\n)',
            'import (\n\t"io"\n\t"fmt"\n)',
        ],
        invalid_outputs=[
            'import "unsafe"',
            'import (\n\t"fmt"\n\t"unsafe"\n)',
            'import u "unsafe"',  # Aliased import
            'var _ = unsafe.Pointer(nil)',  # Direct usage
        ],
        tags=["imports", "unsafe", "safety", "policy"],
        language="go",
        domain="imports",
    ),
    ConstraintExample(
        id="go-imports-003",
        name="Standard Library Restrictions for Embedded",
        description="Restrict standard library imports for embedded/TinyGo targets",
        scenario=(
            "Developer writing Go code for TinyGo (embedded systems). "
            "Many standard library packages aren't available: net/http, "
            "os/exec, database/sql. Only core packages like fmt, sync, "
            "time (partial) are available."
        ),
        prompt="""Write import statements for TinyGo (embedded). Only core packages work:
fmt, sync, time, machine (TinyGo-specific). No net/http, database/sql, os/exec.

""",
        spec=ConstraintSpec(
            language="go",
            # Regex enforces TinyGo-compatible imports
            regex=r'^import\s+(?:\(\s*)?(?:"fmt"|"sync"|"time"|"machine"|"runtime/volatile")',
            ebnf=r'''
root ::= single_fmt | single_machine | grouped_tinygo | single_volatile
single_fmt ::= "import \"fmt\""
single_machine ::= "import \"machine\""
grouped_tinygo ::= "import (" nl "\t\"fmt\"" nl "\t\"sync\"" nl "\t\"machine\"" nl ")"
single_volatile ::= "import \"runtime/volatile\""
nl ::= "\n"
''',
            forbidden_imports={
                "net/http",
                "net/http/httptest",
                "os/exec",
                "database/sql",
                "reflect",  # Limited in TinyGo
                "plugin",
            },
            available_modules={
                "fmt",
                "sync",
                "time",
                "machine",  # TinyGo-specific
                "runtime/volatile",  # TinyGo-specific
            },
        ),
        expected_effect=(
            "Masks imports of heavy standard library packages unavailable in "
            "embedded contexts. Allows core packages and TinyGo-specific packages. "
            "Enforces embedded environment constraints."
        ),
        valid_outputs=[
            'import "fmt"',
            'import "machine"',
            'import (\n\t"fmt"\n\t"sync"\n\t"machine"\n)',
            'import "runtime/volatile"',
        ],
        invalid_outputs=[
            'import "net/http"',
            'import "database/sql"',
            'import (\n\t"fmt"\n\t"net/http"\n)',  # Blocked package in group
            'import "os/exec"',
        ],
        tags=["imports", "embedded", "tinygo", "constraints"],
        language="go",
        domain="imports",
    ),
]


__all__ = ["GO_IMPORT_EXAMPLES"]
