# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Swift import constraint examples for Ananke.

This module contains realistic examples of import-level constraints in Swift,
demonstrating framework availability checks, platform conditionals, and module visibility.
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from ....spec.constraint_spec import (
        ConstraintSpec,
        ImportBinding,
        SemanticConstraint,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        ImportBinding,
        SemanticConstraint,
    )

# =============================================================================
# Import Constraint Examples
# =============================================================================

SWIFT_IMPORT_001 = ConstraintExample(
    id="swift-imports-001",
    name="Framework Availability Check",
    description="Use #available to check for iOS/macOS version-specific APIs",
    scenario=(
        "Developer using SwiftUI features that are only available in iOS 15+. "
        "Must guard usage with #available check to prevent runtime crashes "
        "on older OS versions. The constraint ensures proper availability checks."
    ),
    prompt="""Use the .task{} modifier which requires iOS 15+.
Wrap it with #available(iOS 15, *) to support older OS versions.

""",
    spec=ConstraintSpec(
        language="swift",
        imports=[
            ImportBinding(module="SwiftUI"),
        ],
        available_modules={
            "SwiftUI",
            "Foundation",
            "Combine",
        },
        semantic_constraints=[
            SemanticConstraint(
                kind="precondition",
                expression="if #available(iOS 15, *)",
                scope="body",
                variables=(),
            ),
        ],
        ebnf=r'''
root ::= avail_task | avail_refresh | avail_search
avail_task ::= "if #available(iOS 15, *) { .task { await loadData() } }"
avail_refresh ::= "if #available(iOS 15, *) { .refreshable { await refresh() } }"
avail_search ::= "if #available(iOS 15, macOS 12, *) { .searchable(text: $searchText) }"
''',
    ),
    expected_effect=(
        "Masks tokens that would use iOS 15+ APIs without availability checks. "
        "Ensures version-gated features are properly guarded."
    ),
    valid_outputs=[
        "if #available(iOS 15, *) { .task { await loadData() } }",
        "if #available(iOS 15, *) { .refreshable { await refresh() } }",
        "if #available(iOS 15, macOS 12, *) { .searchable(text: $searchText) }",
    ],
    invalid_outputs=[
        ".task { await loadData() }",  # Missing availability check
        ".refreshable { await refresh() }",  # No version guard
        "if #available(iOS 14, *) { .task { } }",  # Wrong version (task needs iOS 15)
    ],
    tags=["imports", "availability", "version", "swiftui"],
    language="swift",
    domain="imports",
)

SWIFT_IMPORT_002 = ConstraintExample(
    id="swift-imports-002",
    name="Platform Conditional Compilation",
    description="Use #if os() to conditionally compile platform-specific code",
    scenario=(
        "Developer writing cross-platform Swift code that needs different imports "
        "for macOS vs iOS. Using #if os(macOS) and #if os(iOS) to conditionally "
        "import AppKit vs UIKit while maintaining type safety."
    ),
    prompt="""Import the appropriate UI framework based on platform.
Use #if os(macOS) for AppKit and #if os(iOS) for UIKit.

""",
    spec=ConstraintSpec(
        language="swift",
        imports=[
            ImportBinding(module="Foundation"),
        ],
        available_modules={
            "Foundation",
            "AppKit",  # macOS only
            "UIKit",   # iOS only
        },
        ebnf=r'''
root ::= macos_appkit | ios_uikit | canimport_appkit | cross_platform
macos_appkit ::= "#if os(macOS)\nimport AppKit\n#endif"
ios_uikit ::= "#if os(iOS)\nimport UIKit\n#endif"
canimport_appkit ::= "#if canImport(AppKit)\nimport AppKit\n#endif"
cross_platform ::= "#if os(macOS)\nimport AppKit\n#elseif os(iOS)\nimport UIKit\n#endif"
''',
    ),
    expected_effect=(
        "Masks tokens that would import platform-specific frameworks without guards. "
        "Ensures AppKit is only imported on macOS, UIKit only on iOS."
    ),
    valid_outputs=[
        "#if os(macOS)\nimport AppKit\n#endif",
        "#if os(iOS)\nimport UIKit\n#endif",
        "#if canImport(AppKit)\nimport AppKit\n#endif",
        "#if os(macOS)\nimport AppKit\n#elseif os(iOS)\nimport UIKit\n#endif",
    ],
    invalid_outputs=[
        "import AppKit",  # Missing platform guard
        "import UIKit",  # No conditional compilation
        "#if os(macOS)\nimport UIKit\n#endif",  # Wrong framework for platform
    ],
    tags=["imports", "platform", "conditional", "cross-platform"],
    language="swift",
    domain="imports",
)

SWIFT_IMPORT_003 = ConstraintExample(
    id="swift-imports-003",
    name="Module Visibility with @_exported",
    description="Re-export module symbols using @_exported import",
    scenario=(
        "Developer creating a framework wrapper that re-exports underlying modules. "
        "Using @_exported import to make symbols from dependencies available to "
        "framework clients without requiring separate imports."
    ),
    prompt="""Re-export Foundation and Combine so framework clients don't need separate imports.
Use @_exported import to make dependencies public.

""",
    spec=ConstraintSpec(
        language="swift",
        imports=[
            ImportBinding(module="Foundation"),
        ],
        available_modules={
            "Foundation",
            "Combine",
            "SwiftUI",
        },
        forbidden_imports={
            "UIKit",  # Should use @_exported SwiftUI instead
        },
        ebnf=r'''
root ::= export_import (nl export_import)*
export_import ::= "@_exported import " module | "import " module
module ::= "Foundation" | "Combine" | "SwiftUI"
nl ::= "\n"
''',
    ),
    expected_effect=(
        "Masks tokens that would import forbidden modules or fail to re-export "
        "when needed. Ensures @_exported is used for framework public API surface."
    ),
    valid_outputs=[
        "@_exported import Foundation",
        "@_exported import Combine",
        "@_exported import SwiftUI",
        "import Foundation\n@_exported import Combine",
    ],
    invalid_outputs=[
        "import UIKit",  # Forbidden
        "@_exported import UIKit",  # Still forbidden
        "public import Foundation",  # Not valid syntax (use @_exported)
    ],
    tags=["imports", "visibility", "framework", "re-export"],
    language="swift",
    domain="imports",
)

# =============================================================================
# Exports
# =============================================================================

SWIFT_IMPORT_EXAMPLES: List[ConstraintExample] = [
    SWIFT_IMPORT_001,
    SWIFT_IMPORT_002,
    SWIFT_IMPORT_003,
]

__all__ = ["SWIFT_IMPORT_EXAMPLES"]
