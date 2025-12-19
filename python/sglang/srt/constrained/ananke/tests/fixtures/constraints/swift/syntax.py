# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Swift syntax constraint examples for Ananke.

This module contains realistic examples of syntax-level constraints in Swift,
demonstrating Codable struct schemas, bundle ID patterns, and SwiftUI DSL structure.
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from ....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
    )

# =============================================================================
# Syntax Constraint Examples
# =============================================================================

SWIFT_SYNTAX_001 = ConstraintExample(
    id="swift-syntax-001",
    name="Codable Struct Schema for JSON",
    description="Generate Codable struct matching JSON schema",
    scenario=(
        "Developer creating a Swift struct to decode JSON API responses. "
        "The struct must conform to Codable protocol with property names "
        "and types matching the JSON structure, using CodingKeys for snake_case."
    ),
    prompt="""Define a User struct that decodes JSON with snake_case keys (user_id, display_name).
Use CodingKeys enum to map camelCase Swift properties to snake_case JSON.

""",
    spec=ConstraintSpec(
        language="swift",
        json_schema="""{
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "display_name": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
                "is_verified": {"type": "boolean"}
            },
            "required": ["user_id", "display_name"]
        }""",
        type_aliases={
            "Codable": "typealias Codable = Decodable & Encodable",
        },
        ebnf=r'''
root ::= "struct User: Codable {" nl properties coding_keys? "}"
properties ::= property+
property ::= "    let " identifier ": " typename nl
coding_keys ::= "    enum CodingKeys: String, CodingKey {" nl cases "    }" nl
cases ::= coding_case+
coding_case ::= "        case " identifier " = " qstring nl
identifier ::= [a-z][a-zA-Z0-9]*
typename ::= "String" | "Bool" | "Date" | "Int"
qstring ::= "\"" [a-z_]+ "\""
nl ::= "\n"
''',
    ),
    expected_effect=(
        "Masks tokens that would create non-Codable properties or missing required fields. "
        "Ensures CodingKeys enum maps Swift camelCase to JSON snake_case."
    ),
    valid_outputs=[
        """struct User: Codable {
    let userId: String
    let displayName: String
    let createdAt: Date
    let isVerified: Bool
    enum CodingKeys: String, CodingKey {
        case userId = "user_id"
        case displayName = "display_name"
        case createdAt = "created_at"
        case isVerified = "is_verified"
    }
}""",
        """struct User: Codable {
    let userId: String
    let displayName: String
}""",
    ],
    invalid_outputs=[
        "struct User { let userId: String }",  # Missing Codable conformance
        "struct User: Codable { var userId: String }",  # var instead of let for immutable JSON
        "struct User: Codable { let user_id: String }",  # Using snake_case directly
    ],
    tags=["syntax", "codable", "json", "schema"],
    language="swift",
    domain="syntax",
)

SWIFT_SYNTAX_002 = ConstraintExample(
    id="swift-syntax-002",
    name="Bundle ID Regex Pattern",
    description="Validate iOS bundle identifier format",
    scenario=(
        "Developer configuring an iOS app bundle identifier. "
        "Must follow reverse DNS notation with lowercase alphanumeric "
        "segments separated by dots (e.g., com.company.appname)."
    ),
    prompt="""Write an iOS bundle identifier in reverse DNS format.
Use lowercase letters, numbers, and dots only (e.g., com.example.myapp).

""",
    spec=ConstraintSpec(
        language="swift",
        regex=r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9]*)+$",
        ebnf="""
        root ::= segment ("." segment)+
        segment ::= [a-z] [a-z0-9]*
        """,
    ),
    expected_effect=(
        "Masks tokens that would create invalid bundle IDs. Ensures lowercase, "
        "alphanumeric segments in reverse DNS format."
    ),
    valid_outputs=[
        "com.example.myapp",
        "com.company.productname",
        "io.github.username.project",
        "org.openai.chatapp",
    ],
    invalid_outputs=[
        "MyApp",  # Not reverse DNS
        "com.Example.App",  # Contains uppercase
        "com.company",  # Only two segments (need 3+)
        "com.company.my-app",  # Contains hyphen
        "com.company.my_app",  # Contains underscore
    ],
    tags=["syntax", "regex", "bundle-id", "validation"],
    language="swift",
    domain="syntax",
)

SWIFT_SYNTAX_003 = ConstraintExample(
    id="swift-syntax-003",
    name="SwiftUI DSL Structure",
    description="Enforce SwiftUI view hierarchy DSL syntax",
    scenario=(
        "Developer building SwiftUI views with proper DSL structure. "
        "Views must use containers (VStack, HStack, ZStack) with view builders, "
        "and modifiers must be chained in the correct order (frame before padding)."
    ),
    prompt="""Create a VStack with Text and Image views using SwiftUI DSL syntax.
Use view builder closure syntax - no array literals or explicit returns.

""",
    spec=ConstraintSpec(
        language="swift",
        type_bindings=[
            TypeBinding(
                name="self",
                type_expr="some View",
                scope="local",
            ),
        ],
        type_aliases={
            "View": "protocol { var body: some View }",
        },
        ebnf=r'''
root ::= vstack_simple | hstack_spacing | zstack_color
vstack_simple ::= "VStack {\n    Text(\"Hello\")\n    Image(systemName: \"star\")\n}"
hstack_spacing ::= "HStack(spacing: 10) {\n    Text(\"Left\").padding()\n    Text(\"Right\").padding()\n}.frame(width: 200)"
zstack_color ::= "ZStack {\n    Text(\"Background\")\n    Text(\"Foreground\").foregroundColor(.blue)\n}"
''',
    ),
    expected_effect=(
        "Masks tokens that would violate SwiftUI DSL structure. Ensures proper "
        "container usage, view builder syntax, and modifier ordering."
    ),
    valid_outputs=[
        """VStack {
    Text("Hello")
    Image(systemName: "star")
}""",
        """HStack(spacing: 10) {
    Text("Left").padding()
    Text("Right").padding()
}.frame(width: 200)""",
        """ZStack {
    Text("Background")
    Text("Foreground").foregroundColor(.blue)
}""",
    ],
    invalid_outputs=[
        "VStack { print('debug'); Text('Hello') }",  # Non-view statement
        "Text('Hello') { Text('Child') }",  # Leaf views can't have children
        "HStack([Text('A'), Text('B')])",  # Array syntax instead of builder
    ],
    tags=["syntax", "swiftui", "dsl", "view-builder"],
    language="swift",
    domain="syntax",
)

# =============================================================================
# Exports
# =============================================================================

SWIFT_SYNTAX_EXAMPLES: List[ConstraintExample] = [
    SWIFT_SYNTAX_001,
    SWIFT_SYNTAX_002,
    SWIFT_SYNTAX_003,
]

__all__ = ["SWIFT_SYNTAX_EXAMPLES"]
