# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Syntax constraint examples for TypeScript.

This module contains realistic examples of syntax-level constraints for
TypeScript including OpenAPI/Swagger schemas, URL pattern regex with capture
groups, and GraphQL query string schemas.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import ConstraintSpec
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import ConstraintSpec


TYPESCRIPT_SYNTAX_EXAMPLES = [
    ConstraintExample(
        id="ts-syntax-001",
        name="OpenAPI Schema Definition",
        description="Generate TypeScript types from OpenAPI/Swagger schema",
        scenario=(
            "Developer working with OpenAPI specification and generating TypeScript "
            "types for API request/response objects. The schema defines a User object "
            "with required and optional fields, string formats, and nested objects. "
            "Generated types must match the JSON Schema exactly."
        ),
        prompt="""Write a TypeScript interface for a User object based on this JSON Schema:
required fields are id, email, createdAt (all strings). Optional fields are name (string)
and profile (nested object with optional bio and avatarUrl). Use ? for optional properties.

""",
        spec=ConstraintSpec(
            language="typescript",
            json_schema="""{
  "type": "object",
  "required": ["id", "email", "createdAt"],
  "properties": {
    "id": {
      "type": "string",
      "format": "uuid"
    },
    "email": {
      "type": "string",
      "format": "email"
    },
    "name": {
      "type": "string",
      "minLength": 1,
      "maxLength": 100
    },
    "createdAt": {
      "type": "string",
      "format": "date-time"
    },
    "profile": {
      "type": "object",
      "properties": {
        "bio": { "type": "string" },
        "avatarUrl": { "type": "string", "format": "uri" }
      }
    }
  },
  "additionalProperties": false
}""",
        ),
        expected_effect=(
            "Masks tokens that would create type definitions not matching the JSON "
            "Schema. Ensures required fields (id, email, createdAt) are non-optional, "
            "optional fields (name, profile) use ? modifier. Blocks additional "
            "properties due to additionalProperties: false. Enforces string types "
            "for all fields."
        ),
        valid_outputs=[
            """interface User {
  id: string;
  email: string;
  name?: string;
  createdAt: string;
  profile?: {
    bio?: string;
    avatarUrl?: string;
  };
}""",
            """type User = {
  id: string;
  email: string;
  name?: string;
  createdAt: string;
  profile?: {
    bio?: string;
    avatarUrl?: string;
  };
};""",
        ],
        invalid_outputs=[
            """interface User {
  id: string;
  email: string;
  name: string;
  createdAt: string;
}""",  # name should be optional
            """interface User {
  id: string;
  email?: string;
  name?: string;
  createdAt: string;
}""",  # email should be required
            """interface User {
  id: string;
  email: string;
  name?: string;
  createdAt: string;
  profile?: object;
  extraField?: string;
}""",  # additionalProperties not allowed
        ],
        tags=["syntax", "json-schema", "openapi", "swagger"],
        language="typescript",
        domain="syntax",
    ),
    ConstraintExample(
        id="ts-syntax-002",
        name="URL Pattern with Capture Groups",
        description="Parse URL with regex and extract route parameters",
        scenario=(
            "Developer implementing client-side routing with URL pattern matching. "
            "Must parse URLs like '/users/123/posts/456' and extract userId and "
            "postId. Uses regex with named capture groups to extract parameters "
            "in a type-safe way."
        ),
        prompt="""Write a regex pattern that matches "/users/:userId/posts/:postId" URLs and extracts
the IDs using named capture groups. Use (?<userId>\\d+) syntax. Include ^ and $ anchors.
Then use pattern.exec(url) or url.match(pattern) to extract match.groups.

""",
        spec=ConstraintSpec(
            language="typescript",
            # Regex validates that output contains named capture group pattern for URL routing
            regex=r"const\s+pattern\s*=\s*/[\s\S]*\(\?<userId>[\s\S]*\(\?<postId>",
            ebnf=r'''
root ::= exec_pattern | match_pattern
exec_pattern ::= pattern_decl nl "const match = pattern.exec(url);" nl "if (match?.groups) {" nl "  const { userId, postId } = match.groups;" nl "  return { userId, postId };" nl "}"
match_pattern ::= pattern_decl nl "const result = url.match(pattern);" nl "return result?.groups as { userId: string; postId: string } | undefined;"
pattern_decl ::= "const pattern = /^\\/users\\/(?<userId>\\d+)\\/posts\\/(?<postId>\\d+)$/;"
nl ::= "\n"
''',
        ),
        expected_effect=(
            "Masks tokens that would create regex patterns not matching the URL "
            "structure. Ensures the pattern captures userId and postId as numeric "
            "strings. Blocks patterns without named capture groups or with incorrect "
            "anchoring (^/$). Enforces digit-only capture groups (\\d+)."
        ),
        valid_outputs=[
            r"""const pattern = /^\/users\/(?<userId>\d+)\/posts\/(?<postId>\d+)$/;
const match = pattern.exec(url);
if (match?.groups) {
  const { userId, postId } = match.groups;
  return { userId, postId };
}""",
            r"""const pattern = /^\/users\/(?<userId>\d+)\/posts\/(?<postId>\d+)$/;
const result = url.match(pattern);
return result?.groups as { userId: string; postId: string } | undefined;""",
        ],
        invalid_outputs=[
            r"""const pattern = /\/users\/(\d+)\/posts\/(\d+)/;
const match = pattern.exec(url);
return match ? { userId: match[1], postId: match[2] } : null;""",  # No named groups
            r"""const pattern = /users\/(?<userId>\d+)\/posts\/(?<postId>\d+)/;""",  # Missing ^ anchor
            r"""const pattern = /^\/users\/(?<userId>\w+)\/posts\/(?<postId>\w+)$/;""",  # \w+ instead of \d+
        ],
        tags=["syntax", "regex", "url-routing", "capture-groups"],
        language="typescript",
        domain="syntax",
    ),
    ConstraintExample(
        id="ts-syntax-003",
        name="GraphQL Query Schema",
        description="Define TypeScript types for GraphQL query results",
        scenario=(
            "Developer working with GraphQL API and generating TypeScript types "
            "for query results. The query fetches user data with nested posts and "
            "comments. Generated types must match the GraphQL schema structure "
            "including nullable fields and nested relationships."
        ),
        prompt="""Write TypeScript interfaces for a GraphQL query result: QueryResult has user which
can be User | null (GraphQL nullable). User has id, username (required), email (optional),
and posts: Post[]. Post has id, title, content?, and comments: Comment[].

""",
        spec=ConstraintSpec(
            language="typescript",
            json_schema="""{
  "type": "object",
  "required": ["user"],
  "properties": {
    "user": {
      "oneOf": [
        {
          "type": "object",
          "required": ["id", "username", "posts"],
          "properties": {
            "id": { "type": "string" },
            "username": { "type": "string" },
            "email": { "type": "string" },
            "posts": {
              "type": "array",
              "items": {
                "type": "object",
                "required": ["id", "title", "comments"],
                "properties": {
                  "id": { "type": "string" },
                  "title": { "type": "string" },
                  "content": { "type": "string" },
                  "comments": {
                    "type": "array",
                    "items": {
                      "type": "object",
                      "required": ["id", "text"],
                      "properties": {
                        "id": { "type": "string" },
                        "text": { "type": "string" }
                      }
                    }
                  }
                }
              }
            }
          }
        },
        { "type": "null" }
      ]
    }
  }
}""",
        ),
        expected_effect=(
            "Masks tokens that would create types not matching the GraphQL schema. "
            "Ensures user can be null (nullable in GraphQL), nested arrays are "
            "properly typed, and required fields are non-optional. Blocks missing "
            "nested types (Post, Comment) or incorrect nullability."
        ),
        valid_outputs=[
            """interface Comment {
  id: string;
  text: string;
}

interface Post {
  id: string;
  title: string;
  content?: string;
  comments: Comment[];
}

interface User {
  id: string;
  username: string;
  email?: string;
  posts: Post[];
}

interface QueryResult {
  user: User | null;
}""",
            """type Comment = {
  id: string;
  text: string;
};

type Post = {
  id: string;
  title: string;
  content?: string;
  comments: Comment[];
};

type User = {
  id: string;
  username: string;
  email?: string;
  posts: Post[];
};

type QueryResult = {
  user: User | null;
};""",
        ],
        invalid_outputs=[
            """interface QueryResult {
  user: User;
}""",  # user should be nullable (User | null)
            """interface User {
  id: string;
  username: string;
  posts: any[];
}""",  # posts should be Post[], not any[]
            """interface QueryResult {
  user?: User;
}""",  # Should be | null, not optional (?)
        ],
        tags=["syntax", "graphql", "schema", "nested-types"],
        language="typescript",
        domain="syntax",
    ),
]
