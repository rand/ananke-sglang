# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Go syntax constraint examples.

Demonstrates syntax-level constraints specific to Go:
- Protobuf struct tags schema validation
- Import path regex validation
- Template syntax for text/template and html/template
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
    )


GO_SYNTAX_EXAMPLES: List[ConstraintExample] = [
    ConstraintExample(
        id="go-syntax-001",
        name="Protobuf Struct Tags Schema",
        description="Validate protobuf struct tag syntax and field numbering",
        scenario=(
            "Developer writing Go structs with protobuf tags for code generation. "
            "Tags must follow exact syntax: `protobuf:\"type,number,opt,name=fieldname\"`. "
            "Field numbers must be unique and in range 1-536870911 (excluding 19000-19999)."
        ),
        prompt="""Write a Go struct with protobuf struct tags. Format is:
`protobuf:"type,number,label,name=fieldname"`. Use varint for ints, bytes for strings.

""",
        spec=ConstraintSpec(
            language="go",
            regex=r'`protobuf:"[a-z0-9]+,[1-9][0-9]*,(opt|req|rep)(,name=[a-zA-Z_][a-zA-Z0-9_]*)?(,json=[a-zA-Z_][a-zA-Z0-9_]*)?"`',
            ebnf=r'''
root ::= user_struct | request_struct | message_struct
user_struct ::= "type User struct {" nl "\tID int64 `protobuf:\"varint,1,opt,name=id\"`" nl "\tName string `protobuf:\"bytes,2,opt,name=name\"`" nl "}"
request_struct ::= "type Request struct {" nl "\tUserIDs []int64 `protobuf:\"varint,1,rep,name=user_ids,json=userIds\"`" nl "}"
message_struct ::= "type Message struct {" nl "\tPayload []byte `protobuf:\"bytes,5,req,name=payload\"`" nl "}"
nl ::= "\n"
''',
        ),
        expected_effect=(
            "Masks struct tags not matching protobuf format. "
            "Ensures proper type,number,label format. "
            "Validates field numbers and options."
        ),
        valid_outputs=[
            'type User struct {\n\tID int64 `protobuf:"varint,1,opt,name=id"`\n\tName string `protobuf:"bytes,2,opt,name=name"`\n}',
            'type Request struct {\n\tUserIDs []int64 `protobuf:"varint,1,rep,name=user_ids,json=userIds"`\n}',
            'type Message struct {\n\tPayload []byte `protobuf:"bytes,5,req,name=payload"`\n}',
        ],
        invalid_outputs=[
            'type User struct {\n\tID int64 `protobuf:"varint,0,opt"`\n}',  # Field number 0 invalid
            'type User struct {\n\tID int64 `protobuf:"varint 1 opt"`\n}',  # Wrong separator
            'type User struct {\n\tName string `protobuf:"bytes,2"`\n}',  # Missing label
            'type User struct {\n\tID int64 `proto:"varint,1,opt"`\n}',  # Wrong tag name
        ],
        tags=["syntax", "protobuf", "struct-tags", "validation"],
        language="go",
        domain="syntax",
    ),
    ConstraintExample(
        id="go-syntax-002",
        name="Import Path Regex Validation",
        description="Validate Go import path format and constraints",
        scenario=(
            "Developer writing import statements. Go import paths must be "
            "valid: no uppercase in domain, alphanumeric with limited special chars, "
            "proper domain format for remote packages. Examples: "
            "\"fmt\", \"encoding/json\", \"github.com/user/repo/pkg\"."
        ),
        prompt="""Generate a valid Go import path string. Lowercase domain, no spaces,
forward slashes for paths. Examples: "fmt", "encoding/json", "github.com/user/repo".

""",
        spec=ConstraintSpec(
            language="go",
            regex=r'^"([a-z0-9]+(/[a-z0-9_-]+)*|[a-z0-9.-]+\.[a-z]{2,}/[a-zA-Z0-9_/-]+)"$',
            ebnf=r'''
root ::= fmt_import | encoding_import | github_import | internal_import | golang_import | google_import
fmt_import ::= "\"fmt\""
encoding_import ::= "\"encoding/json\""
github_import ::= "\"github.com/gorilla/mux\""
internal_import ::= "\"github.com/user/repo/internal/pkg\""
golang_import ::= "\"golang.org/x/net/context\""
google_import ::= "\"google.golang.org/grpc\""
''',
        ),
        expected_effect=(
            "Masks import paths with invalid characters or format. "
            "Ensures lowercase domains, valid path separators. "
            "Blocks malformed or non-standard import paths."
        ),
        valid_outputs=[
            '"fmt"',
            '"encoding/json"',
            '"github.com/gorilla/mux"',
            '"github.com/user/repo/internal/pkg"',
            '"golang.org/x/net/context"',
            '"google.golang.org/grpc"',
        ],
        invalid_outputs=[
            '"GitHub.com/user/repo"',  # Uppercase in domain
            '"fmt "',  # Trailing space
            '"encoding\\json"',  # Wrong path separator
            '"my package"',  # Space in path
            '"../relative/path"',  # Relative path
            '"C:\\windows\\path"',  # Windows path
        ],
        tags=["syntax", "imports", "regex", "validation"],
        language="go",
        domain="syntax",
    ),
    ConstraintExample(
        id="go-syntax-003",
        name="Template Syntax for text/template",
        description="Validate Go template action syntax",
        scenario=(
            "Developer writing Go templates using text/template or html/template. "
            "Template actions use {{...}} delimiters with specific syntax: "
            "{{.Field}}, {{if .Cond}}...{{end}}, {{range .Items}}...{{end}}, "
            "{{template \"name\" .}}."
        ),
        prompt="""Assign a Go text/template string to tmpl. Use {{.Field}} for data, {{if}}...{{end}} for conditionals.

tmpl := `""",
        spec=ConstraintSpec(
            language="go",
            # Regex matches Go template syntax: {{.Field}}, {{if}}, {{range}}, {{define}}, etc.
            regex=r'\{\{-?\s*(?:\.\w+|if|else|end|range|template|define|block|with)\b',
            ebnf=r'''
root ::= simple_tmpl | if_tmpl | range_tmpl | nested_tmpl | trim_tmpl
simple_tmpl ::= "tmpl := `Hello {{.Name}}!`"
if_tmpl ::= "tmpl := `{{if .Active}}Enabled{{else}}Disabled{{end}}`"
range_tmpl ::= "tmpl := `{{range .Items}}{{.Title}} {{end}}`"
nested_tmpl ::= "tmpl := `{{template \"header\" .}} {{.Content}}`"
trim_tmpl ::= "tmpl := `{{- .Field -}}`"
''',
        ),
        expected_effect=(
            "Masks template syntax errors in Go templates. "
            "Ensures proper {{action}} delimiter usage. "
            "Validates control structures and variable references."
        ),
        valid_outputs=[
            'tmpl := `Hello {{.Name}}!`',
            'tmpl := `{{if .Active}}Enabled{{else}}Disabled{{end}}`',
            'tmpl := `{{range .Items}}{{.Title}} {{end}}`',
            'tmpl := `{{template "header" .}} {{.Content}}`',
            'tmpl := `{{- .Field -}}`',  # Whitespace trimming
        ],
        invalid_outputs=[
            'tmpl := `Hello {.Name}!`',  # Single braces
            'tmpl := `{{if .Active}}`',  # Missing {{end}}
            'tmpl := `{{ .Invalid Field }}`',  # Space in field name
            'tmpl := `{{.}}`',  # Bare dot (valid but discouraged)
            'tmpl := `{{.Field}`',  # Missing closing braces
        ],
        tags=["syntax", "templates", "regex", "text-template"],
        language="go",
        domain="syntax",
    ),
]


__all__ = ["GO_SYNTAX_EXAMPLES"]
