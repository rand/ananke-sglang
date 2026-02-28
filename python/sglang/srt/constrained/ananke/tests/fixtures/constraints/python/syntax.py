# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Syntax constraint examples for Python.

This module contains realistic examples of syntax-level constraints that
demonstrate how Ananke's SyntaxDomain uses JSON schemas, regexes, and EBNF
grammars to enforce structural patterns.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import ConstraintSpec
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import ConstraintSpec


PYTHON_SYNTAX_EXAMPLES = [
    ConstraintExample(
        id="py-syntax-001",
        name="JSON API Response Schema",
        description="Generate JSON response matching API schema",
        scenario=(
            "Developer generating a JSON API response that must match the expected "
            'schema: {"status": "ok" | "error", "data": {...}, "message": string?}. '
            "The response must be valid JSON and include required fields."
        ),
        prompt="""Generate a JSON API response with status ("ok" or "error"), required data object, and optional message.
Must match schema: {"status": enum, "data": object, "message"?: string}

""",
        spec=ConstraintSpec(
            language="python",
            json_schema="""{
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["ok", "error"]
        },
        "data": {
            "type": "object"
        },
        "message": {
            "type": "string"
        }
    },
    "required": ["status", "data"],
    "additionalProperties": false
}""",
        ),
        expected_effect=(
            "Masks tokens that would produce invalid JSON or schema violations. "
            "Ensures 'status' is exactly 'ok' or 'error', 'data' is an object, "
            "and no additional fields beyond status/data/message appear."
        ),
        valid_outputs=[
            '{"status": "ok", "data": {"items": [1, 2, 3]}}',
            '{"status": "error", "data": {}, "message": "Not found"}',
            '{"status": "ok", "data": {"user": {"id": 123, "name": "Alice"}}}',
        ],
        invalid_outputs=[
            '{"status": "success", "data": {}}',  # Invalid status value
            '{"status": "ok"}',  # Missing required 'data'
            '{"status": "ok", "data": [], "extra": "field"}',  # data is array, extra field
            '{"status": "ok", "data": null}',  # data is null, not object
        ],
        tags=["syntax", "json-schema", "api", "validation"],
        language="python",
        domain="syntax",
    ),
    ConstraintExample(
        id="py-syntax-002",
        name="Email Validation Regex",
        description="Generate email address matching RFC-like pattern",
        scenario=(
            "Developer generating email addresses for testing that must match a "
            "realistic email pattern: username@domain.tld where username is "
            "alphanumeric with dots/underscores, domain is alphanumeric with hyphens, "
            "and TLD is 2-6 letters."
        ),
        prompt="""Generate a valid email address for testing. Format: username@domain.tld
Username can have letters, numbers, dots, underscores. TLD should be 2-6 letters.

""",
        spec=ConstraintSpec(
            language="python",
            regex=r"^[a-zA-Z0-9._+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}$",
        ),
        expected_effect=(
            "Masks tokens that would produce invalid email formats. Ensures proper "
            "structure with @ separator, valid characters in username and domain, "
            "and TLD of appropriate length."
        ),
        valid_outputs=[
            "alice@example.com",
            "bob.smith@company.co.uk",
            "user_123@sub-domain.example.org",
            "test+tag@domain.io",
        ],
        invalid_outputs=[
            "notanemail",  # No @ symbol
            "@example.com",  # Missing username
            "user@",  # Missing domain
            "user@domain",  # Missing TLD
            "user@domain.c",  # TLD too short
            "user@domain.toolongtld",  # TLD too long
            "user name@domain.com",  # Space in username
        ],
        tags=["syntax", "regex", "validation", "email"],
        language="python",
        domain="syntax",
    ),
    ConstraintExample(
        id="py-syntax-003",
        name="SQL SELECT Statement EBNF",
        description="Generate SQL SELECT statement following DSL grammar",
        scenario=(
            "Developer generating SQL SELECT queries for a query builder DSL. "
            "The grammar enforces structure: SELECT columns FROM table [WHERE condition] "
            "[ORDER BY column]. Must be valid according to simplified SQL EBNF."
        ),
        prompt="""Generate a SQL SELECT query. Format: SELECT columns FROM table [WHERE condition] [ORDER BY column].
Only use basic SELECT/FROM/WHERE/ORDER BY - no JOINs, LIMIT, or subqueries.

""",
        spec=ConstraintSpec(
            language="python",
            ebnf=r'''
root ::= "SELECT " columns " FROM " identifier where_clause? order_clause?
columns ::= "*" | identifier (", " identifier)*
where_clause ::= " WHERE " condition
order_clause ::= " ORDER BY " identifier (" ASC" | " DESC")?
condition ::= identifier " " operator " " value
operator ::= ">" | "<" | "=" | ">=" | "<=" | "<>"
identifier ::= [a-zA-Z_]+
value ::= [0-9]+ | "'" [^']+ "'"
''',
        ),
        expected_effect=(
            "Masks tokens that would produce invalid SQL according to the grammar. "
            "Enforces correct keyword order (SELECT before FROM), proper column/table "
            "identifiers, and optional WHERE/ORDER BY clauses in correct positions."
        ),
        valid_outputs=[
            "SELECT * FROM users",
            "SELECT id, name FROM users WHERE age > 18",
            "SELECT name FROM products ORDER BY price DESC",
            "SELECT username, email FROM accounts WHERE active = 1 ORDER BY created_at ASC",
        ],
        invalid_outputs=[
            "SELECT FROM users",  # Missing columns
            "FROM users SELECT *",  # Wrong order
            "SELECT * FROM",  # Missing table
            "SELECT * FROM users WHERE",  # Incomplete WHERE
            "SELECT * FROM users ORDER",  # Incomplete ORDER BY
            "SELECT * FROM users LIMIT 10",  # LIMIT not in grammar
        ],
        tags=["syntax", "ebnf", "dsl", "sql"],
        language="python",
        domain="syntax",
    ),
]
