#!/usr/bin/env python3
"""Script to update all domain-only examples with syntax constraints.

This script reads all example files and adds appropriate regex or EBNF
constraints based on the valid/invalid outputs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Mapping of example IDs to their constraints
# Format: {example_id: (constraint_type, constraint_value)}
# constraint_type is one of: "regex", "ebnf", "json_schema"

CONSTRAINTS: Dict[str, tuple] = {
    # ============================================================
    # GO EXAMPLES
    # ============================================================

    # Go imports - need import patterns
    "go-imports-001": ("regex", r'^import\s+"github\.com/myapp/'),
    "go-imports-002": ("regex", r'^import\s+"(?!unsafe)[^"]+"\s*$'),
    "go-imports-003": ("regex", r'^import\s+"(?!net/http|os)[^"]+"\s*$'),

    # Go control flow
    "go-controlflow-001": ("ebnf", r'''
        root ::= func_def
        func_def ::= "func " name "(" params ")" return_type " {" body "}"
        name ::= [a-zA-Z_][a-zA-Z0-9_]*
        params ::= param ("," param)*
        param ::= name " " type
        type ::= "*"? name
        return_type ::= " " type | " (" type "," " error)"
        body ::= statement+
        statement ::= defer_stmt | if_stmt | assign_stmt | return_stmt | [^\n]+
        defer_stmt ::= ws "defer " expr
        ws ::= [ \t\n]*
        expr ::= [^\n]+
    '''),
    "go-controlflow-002": ("ebnf", r'''
        root ::= func_def
        func_def ::= "func " name "(" params ")" " {" body "}"
        body ::= statement+
        statement ::= select_stmt | for_stmt | [^\n]+
        select_stmt ::= ws "select {" case+ ws "}"
        case ::= ws "case " expr ":" ws statement
        for_stmt ::= ws "for " expr " {" statement+ ws "}"
        ws ::= [ \t\n]*
        expr ::= [^\n]+
    '''),
    "go-controlflow-003": ("ebnf", r'''
        root ::= func_def
        func_def ::= "func " name "(" params ")" " (" type "," " error)" " {" body "}"
        body ::= statement+
        statement ::= if_err_stmt | return_stmt | assign_stmt | [^\n]+
        if_err_stmt ::= ws "if err != nil {" ws return_stmt ws "}"
        return_stmt ::= ws "return " expr
        assign_stmt ::= ws name ("," name)* " " (":=" | "=") " " expr
        ws ::= [ \t\n]*
        expr ::= [^\n]+
    '''),

    # Go types
    "go-types-001": ("ebnf", r'''
        root ::= "func (r *CustomReader) Read(p []byte) (" return_type ") {" body "}"
        return_type ::= "n int, err error" | "int, error"
        body ::= statement+
        statement ::= assign_stmt | if_stmt | return_stmt | [^\n]+
        assign_stmt ::= ws name " " (":=" | "=") " " expr
        if_stmt ::= ws "if " condition " {" statement+ ws "}"
        return_stmt ::= ws "return" expr?
        condition ::= [^{]+
        ws ::= [ \t\n]*
        expr ::= [^\n]*
    '''),
    "go-types-002": ("ebnf", r'''
        root ::= producer_func | consumer_func
        producer_func ::= "func producer(out chan<- int) {" body "}"
        consumer_func ::= "func consumer(in <-chan int) {" body "}"
        body ::= statement+
        statement ::= for_stmt | select_stmt | send_stmt | [^\n]+
        for_stmt ::= ws "for " expr " {" statement+ ws "}"
        select_stmt ::= ws "select {" case+ ws "}"
        case ::= ws "case " expr ":" statement
        send_stmt ::= ws "out <- " expr
        ws ::= [ \t\n]*
        expr ::= [^\n]+
    '''),
    "go-types-003": ("ebnf", r'''
        root ::= "func Contains[T comparable](slice []T, target T) bool {" body "}"
        body ::= statement+
        statement ::= for_stmt | if_stmt | return_stmt | [^\n]+
        for_stmt ::= ws "for " expr " {" statement+ ws "}"
        if_stmt ::= ws "if " compare_expr " {" statement+ ws "}"
        compare_expr ::= expr " == " expr
        return_stmt ::= ws "return " bool
        bool ::= "true" | "false"
        ws ::= [ \t\n]*
        expr ::= [^\n]+
    '''),
    "go-interfaces-001": ("regex", r'^type\s+\w+\s+struct\s*\{'),
    "go-interfaces-002": ("ebnf", r'''
        root ::= func_def
        func_def ::= "func " name "(" params ")" return_type " {" body "}"
        body ::= switch_stmt | [^\n]+
        switch_stmt ::= ws "switch " expr " {" case+ ws "}"
        case ::= ws "case " type ":" statement
        ws ::= [ \t\n]*
    '''),
    "go-interfaces-003": ("regex", r'^type\s+\w+\s+interface\s*\{'),
    "go-interfaces-004": ("regex", r'^type\s+\w+\[\w+\s+\w+\]\s+struct'),
    "go-interfaces-005": ("ebnf", r'''
        root ::= func_def
        func_def ::= "func " name "(" params ")" return_type " {" body "}"
        return_type ::= " error"?
        body ::= statement+
        statement ::= if_stmt | return_stmt | [^\n]+
        if_stmt ::= ws "if err != nil {" ws statement+ ws "}"
        return_stmt ::= ws "return " expr
        ws ::= [ \t\n]*
        expr ::= [^\n]+
    '''),

    # Go semantics
    "go-semantics-001": ("ebnf", r'''
        root ::= func_def
        func_def ::= "func " name "(" params ")" return_type " {" body "}"
        return_type ::= " error"
        body ::= nil_check rest
        nil_check ::= ws "if " name " == nil {" ws return_stmt ws "}"
        return_stmt ::= ws "return " expr
        rest ::= statement+
        statement ::= [^\n]+
        ws ::= [ \t\n]*
        expr ::= [^\n]+
    '''),
    "go-semantics-002": ("ebnf", r'''
        root ::= func_def
        func_def ::= "func " name "(ctx context.Context," rest_params ")" return_type " {" body "}"
        rest_params ::= " " param ("," " " param)*
        param ::= name " " type
        return_type ::= " (" type "," " error)"
        body ::= statement+
        ws ::= [ \t\n]*
    '''),
    "go-semantics-003": ("ebnf", r'''
        root ::= func_def
        func_def ::= "func " name "()" return_type " {" body "}"
        return_type ::= " error"
        body ::= lock_stmt defer_unlock rest
        lock_stmt ::= ws "mu.Lock()"
        defer_unlock ::= ws "defer mu.Unlock()"
        rest ::= statement+
        statement ::= [^\n]+
        ws ::= [ \t\n]*
    '''),

    # Cross-language error examples
    "cross-error-go": ("regex", r'^(data|result|value)\s*,\s*err\s*:='),
    "cross-error-kotlin": ("regex", r'^return\s+Result\.(success|failure)\('),
    "cross-error-python": ("regex", r'^with\s+\w+\s+as\s+\w+:'),
    "cross-error-rust": ("regex", r'^let\s+\w+\s*=\s*\w+\?;'),
    "cross-error-swift": ("regex", r'^(try\s+\w+|return\s+\.success\()'),
    "cross-error-typescript": ("regex", r'^try\s*\{'),
    "cross-error-zig": ("regex", r'^const\s+\w+\s*=\s*\w+\s+catch'),

    # ============================================================
    # KOTLIN EXAMPLES
    # ============================================================

    # Kotlin control flow
    "kt-cf-001": ("ebnf", r'''
        root ::= when_expr
        when_expr ::= "when (state) {" cases "}"
        cases ::= case+
        case ::= ws "is " type " -> " expr
        type ::= "Loading" | "Success" | "Error"
        ws ::= [ \t\n]*
        expr ::= [^\n]+
    '''),
    "kt-cf-002": ("regex", r'(scope\.launch|launch)\s*\{'),
    "kt-cf-003": ("regex", r'\w+\?\.let\s*\{|\w+\?\.\w+\s*\?:'),

    # Kotlin coroutines
    "kt-coro-001": ("regex", r'\.(map|filter|collect)\s*\{'),
    "kt-coro-002": ("regex", r'^val\s+\w+\s*=\s*(fetch|async|await|coroutineScope)'),
    "kt-coro-003": ("regex", r'CoroutineScope\(.*Dispatchers\.'),
    "kt-coro-004": ("regex", r'\.flatMap(Latest|Concat|Merge)?\s*\{'),
    "kt-coro-005": ("regex", r'(MutableStateFlow|StateFlow|Flow)<'),

    # Kotlin imports
    "kt-imports-001": ("regex", r'^import\s+(java\.|kotlin\.)'),
    "kt-imports-002": ("regex", r'^import\s+kotlinx\.coroutines\.'),
    "kt-imports-003": ("regex", r'^import\s+kotlinx\.serialization\.'),

    # Kotlin semantics
    "kt-sem-001": ("regex", r'(requireNotNull|checkNotNull|require)\s*\('),
    "kt-sem-002": ("regex", r'\.(getOrNull|getOrElse|getOrDefault)\s*\('),
    "kt-sem-003": ("regex", r'\.map\s*\{.*\.also\s*\{'),

    # Kotlin syntax
    "kt-syn-001": ("regex", r'@Serializable'),
    "kt-syn-002": ("regex", r'(html|div|span|p|a)\s*\{'),
    "kt-syn-003": ("regex", r'@(RestController|GetMapping|PostMapping|RequestMapping)'),

    # Kotlin types
    "kt-types-001": ("regex", r'\w+\?\.\w+'),
    "kt-types-002": ("ebnf", r'''
        root ::= when_expr
        when_expr ::= "when (result) {" cases "}"
        cases ::= case+
        case ::= ws "is " type " -> " expr
        type ::= "Success" | "Error" | "Loading"
        ws ::= [ \t\n]*
        expr ::= [^\n]+
    '''),
    "kt-types-003": ("regex", r'^return\s+fetch\w+\s*\('),

    # ============================================================
    # PYTHON EXAMPLES
    # ============================================================

    # Python control flow
    "py-controlflow-001": ("regex", r'^(data|result)\s*=\s*await\s+'),
    "py-controlflow-002": ("regex", r'^if\s+\w+\s*==\s*.+:\s*(continue|break)'),
    "py-controlflow-003": ("regex", r'^except\s+(IOError|ValueError)\s+(as\s+\w+)?:'),

    # Python imports
    "py-imports-001": ("regex", r'^(import|from)\s+(json|math|datetime|re|typing|dataclasses)\b'),
    "py-imports-002": ("regex", r'^(import|from)\s+(numpy|scipy|sklearn|pandas|torch|matplotlib)\b'),
    "py-imports-003": ("regex", r'^if\s+TYPE_CHECKING:\s*$'),

    # Python semantics
    "py-semantics-001": ("regex", r'^return\s+len\s*\(\s*\w+\s*\)\s*>\s*\d+'),
    "py-semantics-002": ("regex", r'^if\s+\w+\s*(>|<|>=|<=|==)\s*\d+:'),
    "py-semantics-003": ("regex", r'^def\s+withdraw\s*\(self,\s*amount:\s*(int|float)\)'),

    # Python types (deep example)
    "py-deep-001": ("regex", r'^@dataclass\s*$'),

    # ============================================================
    # RUST EXAMPLES (already have some EBNF, just cross-error)
    # ============================================================

    # ============================================================
    # SWIFT EXAMPLES
    # ============================================================

    # ============================================================
    # TYPESCRIPT EXAMPLES
    # ============================================================

    # TypeScript control flow
    "ts-controlflow-001": ("ebnf", r'''
        root ::= switch_stmt
        switch_stmt ::= "switch (" expr ") {" cases "}"
        cases ::= case+ default?
        case ::= ws "case " literal ":" statement
        default ::= ws "default:" statement
        literal ::= '"' [^"]+ '"' | "'" [^']+ "'" | [0-9]+
        ws ::= [ \t\n]*
        statement ::= [^\n]+
        expr ::= [^\n)]+
    '''),
    "ts-controlflow-002": ("regex", r'^async\s+function\s+\w+\s*\('),
    "ts-controlflow-003": ("regex", r'^function\s+\w+\s*\([^)]*\):\s*\w+\s+is\s+\w+'),

    # TypeScript imports
    "ts-imports-001": ("regex", r'^import\s+(\{[^}]+\}|type\s+\{[^}]+\}|\*\s+as\s+\w+|\w+)\s+from\s+[\'"]'),
    "ts-imports-002": ("regex", r'^import\s+(React|type\s+\{[^}]+\})\s+from\s+[\'"]react[\'"]'),
    "ts-imports-003": ("regex", r'^import\s+type\s+\{'),

    # TypeScript semantics
    "ts-semantics-001": ("regex", r'^const\s+\w+\s*=\s*Object\.freeze\s*\('),
    "ts-semantics-002": ("regex", r'^function\s+assert\w*\s*\([^)]+\):\s*asserts\s+'),
    "ts-semantics-003": ("regex", r'^function\s+\w+\s*\([^)]*\):\s*\w+(\[\]|<[^>]+>)?\s*\{'),

    # TypeScript types
    "ts-types-001": ("regex", r'^type\s+\w+<[^>]+>\s*='),
    "ts-types-002": ("regex", r'^type\s+\w+\s*=\s*\{[^}]+\}'),
    "ts-types-003": ("regex", r'^type\s+\w+<[^>]+>\s*='),
    "ts-conditional-001": ("regex", r'^type\s+\w+<[^>]+>\s*=\s*\w+\s+extends\s+'),
    "ts-conditional-002": ("regex", r'^type\s+\w+<[^>]+>\s*=\s*\w+\s+extends\s+'),
    "ts-conditional-003": ("regex", r'^type\s+\w+<[^>]+>\s*=\s*\w+\s+extends\s+'),
    "ts-conditional-004": ("regex", r'^type\s+\w+<[^>]+>\s*=\s*\w+\s+extends\s+'),
    "ts-conditional-005": ("regex", r'^type\s+\w+<[^>]+>\s*=\s*\{[^}]+\}\s+extends\s+'),

    # ============================================================
    # ZIG EXAMPLES
    # ============================================================

    # Zig comptime
    "zig-comptime-001": ("regex", r'^pub\s+fn\s+\w+\s*\(comptime\s+'),
    "zig-comptime-002": ("regex", r'^const\s+\w+\s*=\s*comptime\s+'),
    "zig-comptime-003": ("regex", r'^const\s+\w+\s*:\s*type\s*='),

    # Zig control flow
    "zig-controlflow-001": ("regex", r'^pub\s+fn\s+\w+\s*\([^)]*\)\s*!'),
    "zig-controlflow-002": ("regex", r'^pub\s+fn\s+\w+\s*\([^)]*\)\s*\?'),
    "zig-controlflow-003": ("regex", r'^const\s+result\s*=\s*\w+\s+catch\s+'),

    # Zig imports
    "zig-imports-001": ("regex", r'^const\s+std\s*=\s*@import\s*\(\s*"std"\s*\)'),
    "zig-imports-002": ("regex", r'^const\s+\w+\s*=\s*@import\s*\(\s*"[^"]+"\s*\)'),
    "zig-imports-003": ("regex", r'^const\s+c\s*=\s*@cImport\s*\('),

    # Zig semantics
    "zig-semantics-001": ("regex", r'^pub\s+fn\s+\w+\s*\([^)]*\)\s*!'),
    "zig-semantics-002": ("regex", r'^const\s+slice\s*=\s*\w+\[\d+\.\.'),
    "zig-semantics-003": ("regex", r'^pub\s+const\s+\w+\s*:\s*\w+\s*='),

    # Zig syntax
    "zig-syntax-001": ("regex", r'^const\s+\w+\s*=\s*struct\s*\{'),
    "zig-syntax-003": ("regex", r'^std\.debug\.print\s*\('),

    # Zig types
    "zig-types-001": ("regex", r'^fn\s+\w+\s*\(.*anytype'),
    "zig-types-002": ("regex", r'^fn\s+\w+\s*\([^)]*\)\s*\w+\s*\{'),
    "zig-types-003": ("regex", r'^fn\s+\w+\s*\([^)]*\)\s*\*'),
}


def update_file_with_constraints(file_path: Path, constraints: Dict[str, tuple]) -> int:
    """Update a single file with constraints for matching example IDs.

    Returns:
        Number of examples updated
    """
    content = file_path.read_text()
    updated_count = 0

    for example_id, (constraint_type, constraint_value) in constraints.items():
        # Check if this example is in this file
        if f'id="{example_id}"' not in content:
            continue

        # Check if already has constraint
        # Find the spec for this example and check
        pattern = rf'(id="{example_id}".*?spec=ConstraintSpec\([^)]*?)(,?\s*\))'
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            continue

        spec_content = match.group(1)

        # Skip if already has a syntax constraint
        if 'json_schema=' in spec_content or 'regex=' in spec_content or 'ebnf=' in spec_content:
            continue

        # Add the constraint
        if constraint_type == "regex":
            # Escape the constraint value for insertion
            escaped_value = constraint_value.replace('\\', '\\\\').replace('"', '\\"')
            constraint_str = f'\n            regex=r"{constraint_value}",'
        elif constraint_type == "ebnf":
            constraint_str = f'\n            ebnf=r"""{constraint_value}""",'
        else:
            continue

        # Find where to insert (after language= line typically)
        # Look for the pattern in the spec and insert after language
        lang_pattern = rf'(id="{example_id}".*?language="[^"]+")(,)'
        lang_match = re.search(lang_pattern, content, re.DOTALL)

        if lang_match:
            new_content = content[:lang_match.end(1)] + constraint_str + content[lang_match.end(1):]
            content = new_content
            updated_count += 1

    if updated_count > 0:
        file_path.write_text(content)

    return updated_count


def main():
    """Update all example files with constraints."""
    fixtures_dir = Path(__file__).parent

    # Find all Python files in language subdirectories
    updated_total = 0
    files_updated = []

    for lang_dir in fixtures_dir.iterdir():
        if not lang_dir.is_dir() or lang_dir.name.startswith('_'):
            continue

        for py_file in lang_dir.glob("*.py"):
            if py_file.name.startswith('_'):
                continue

            count = update_file_with_constraints(py_file, CONSTRAINTS)
            if count > 0:
                updated_total += count
                files_updated.append((py_file, count))

    print(f"Updated {updated_total} examples across {len(files_updated)} files:")
    for f, count in files_updated:
        print(f"  {f.relative_to(fixtures_dir)}: {count} examples")


if __name__ == "__main__":
    main()
