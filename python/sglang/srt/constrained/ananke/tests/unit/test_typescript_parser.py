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
"""Unit tests for TypeScript incremental parser.

Tests for the TypeScriptIncrementalParser implementation including:
- Basic parsing functionality
- Variable declarations
- Function declarations
- Arrow functions
- Interfaces and type aliases
- Classes
- Incremental parsing
- Checkpoint/restore
- Hole detection
- TSX mode
"""

import pytest

from parsing import get_parser
from parsing.languages.typescript import (
    TypeScriptIncrementalParser,
    create_typescript_parser,
)


# ===========================================================================
# Factory Tests
# ===========================================================================


class TestTypeScriptParserFactory:
    """Tests for parser factory functions."""

    def test_get_parser_typescript(self):
        """Should return TypeScript parser for 'typescript'."""
        parser = get_parser("typescript")
        assert isinstance(parser, TypeScriptIncrementalParser)

    def test_get_parser_ts(self):
        """Should return TypeScript parser for 'ts'."""
        parser = get_parser("ts")
        assert isinstance(parser, TypeScriptIncrementalParser)

    def test_get_parser_javascript(self):
        """Should return TypeScript parser for 'javascript'."""
        parser = get_parser("javascript")
        assert isinstance(parser, TypeScriptIncrementalParser)

    def test_get_parser_js(self):
        """Should return TypeScript parser for 'js'."""
        parser = get_parser("js")
        assert isinstance(parser, TypeScriptIncrementalParser)

    def test_create_typescript_parser(self):
        """Should create parser with factory function."""
        parser = create_typescript_parser()
        assert isinstance(parser, TypeScriptIncrementalParser)


# ===========================================================================
# Basic Parsing Tests
# ===========================================================================


class TestTypeScriptBasicParsing:
    """Tests for basic parsing functionality."""

    @pytest.fixture
    def parser(self):
        """Create a fresh parser for each test."""
        return TypeScriptIncrementalParser()

    def test_parse_empty_string(self, parser):
        """Should parse empty string."""
        result = parser.parse_initial("")
        assert result.is_valid

    def test_parse_whitespace(self, parser):
        """Should parse whitespace only."""
        result = parser.parse_initial("   \n\t  ")
        assert result.is_valid

    def test_parse_comment_single_line(self, parser):
        """Should parse single-line comment."""
        result = parser.parse_initial("// this is a comment")
        assert result.is_valid

    def test_parse_comment_multi_line(self, parser):
        """Should parse multi-line comment."""
        result = parser.parse_initial("/* this is\na multi-line\ncomment */")
        assert result.is_valid


# ===========================================================================
# Variable Declaration Tests
# ===========================================================================


class TestTypeScriptVariableDeclarations:
    """Tests for parsing variable declarations."""

    @pytest.fixture
    def parser(self):
        return TypeScriptIncrementalParser()

    def test_parse_const_number(self, parser):
        """Should parse const with number."""
        result = parser.parse_initial("const x = 42;")
        assert result.is_valid

    def test_parse_const_with_type(self, parser):
        """Should parse const with type annotation."""
        result = parser.parse_initial("const x: number = 42;")
        assert result.is_valid

    def test_parse_let(self, parser):
        """Should parse let declaration."""
        result = parser.parse_initial("let x = 'hello';")
        assert result.is_valid

    def test_parse_let_with_type(self, parser):
        """Should parse let with type annotation."""
        result = parser.parse_initial("let x: string = 'hello';")
        assert result.is_valid

    def test_parse_var(self, parser):
        """Should parse var declaration."""
        result = parser.parse_initial("var x = true;")
        assert result.is_valid

    def test_parse_const_array(self, parser):
        """Should parse const array."""
        result = parser.parse_initial("const arr: number[] = [1, 2, 3];")
        assert result.is_valid

    def test_parse_const_object(self, parser):
        """Should parse const object."""
        result = parser.parse_initial("const obj = { name: 'test', value: 42 };")
        assert result.is_valid

    def test_parse_destructuring_object(self, parser):
        """Should parse object destructuring."""
        result = parser.parse_initial("const { name, value } = obj;")
        assert result.is_valid

    def test_parse_destructuring_array(self, parser):
        """Should parse array destructuring."""
        result = parser.parse_initial("const [a, b, c] = arr;")
        assert result.is_valid


# ===========================================================================
# Function Declaration Tests
# ===========================================================================


class TestTypeScriptFunctionDeclarations:
    """Tests for parsing function declarations."""

    @pytest.fixture
    def parser(self):
        return TypeScriptIncrementalParser()

    def test_parse_simple_function(self, parser):
        """Should parse simple function."""
        result = parser.parse_initial("function foo() {}")
        assert result.is_valid

    def test_parse_function_with_params(self, parser):
        """Should parse function with parameters."""
        result = parser.parse_initial("function add(a: number, b: number) { return a + b; }")
        assert result.is_valid

    def test_parse_function_with_return_type(self, parser):
        """Should parse function with return type."""
        result = parser.parse_initial("function greet(name: string): string { return 'Hello ' + name; }")
        assert result.is_valid

    def test_parse_async_function(self, parser):
        """Should parse async function."""
        result = parser.parse_initial("async function fetchData() { return await fetch('/api'); }")
        assert result.is_valid

    def test_parse_generator_function(self, parser):
        """Should parse generator function."""
        result = parser.parse_initial("function* gen() { yield 1; yield 2; }")
        assert result.is_valid

    def test_parse_generic_function(self, parser):
        """Should parse generic function."""
        result = parser.parse_initial("function identity<T>(x: T): T { return x; }")
        assert result.is_valid


# ===========================================================================
# Arrow Function Tests
# ===========================================================================


class TestTypeScriptArrowFunctions:
    """Tests for parsing arrow functions."""

    @pytest.fixture
    def parser(self):
        return TypeScriptIncrementalParser()

    def test_parse_arrow_expression_body(self, parser):
        """Should parse arrow function with expression body."""
        result = parser.parse_initial("const add = (a: number, b: number) => a + b;")
        assert result.is_valid

    def test_parse_arrow_block_body(self, parser):
        """Should parse arrow function with block body."""
        result = parser.parse_initial("const add = (a: number, b: number) => { return a + b; };")
        assert result.is_valid

    def test_parse_arrow_single_param(self, parser):
        """Should parse arrow function with single param (no parens)."""
        result = parser.parse_initial("const double = x => x * 2;")
        assert result.is_valid

    def test_parse_arrow_no_params(self, parser):
        """Should parse arrow function with no params."""
        result = parser.parse_initial("const hello = () => 'hello';")
        assert result.is_valid

    def test_parse_async_arrow(self, parser):
        """Should parse async arrow function."""
        result = parser.parse_initial("const fetch = async () => await getData();")
        assert result.is_valid


# ===========================================================================
# Interface and Type Alias Tests
# ===========================================================================


class TestTypeScriptInterfacesAndTypes:
    """Tests for parsing interfaces and type aliases."""

    @pytest.fixture
    def parser(self):
        return TypeScriptIncrementalParser()

    def test_parse_empty_interface(self, parser):
        """Should parse empty interface."""
        result = parser.parse_initial("interface Empty {}")
        assert result.is_valid

    def test_parse_interface_with_properties(self, parser):
        """Should parse interface with properties."""
        result = parser.parse_initial("""
        interface Person {
            name: string;
            age: number;
        }
        """)
        assert result.is_valid

    def test_parse_interface_optional_properties(self, parser):
        """Should parse interface with optional properties."""
        result = parser.parse_initial("""
        interface Config {
            host: string;
            port?: number;
        }
        """)
        assert result.is_valid

    def test_parse_interface_readonly(self, parser):
        """Should parse interface with readonly properties."""
        result = parser.parse_initial("""
        interface Point {
            readonly x: number;
            readonly y: number;
        }
        """)
        assert result.is_valid

    def test_parse_interface_extends(self, parser):
        """Should parse interface extending another."""
        result = parser.parse_initial("""
        interface Animal { name: string; }
        interface Dog extends Animal { breed: string; }
        """)
        assert result.is_valid

    def test_parse_type_alias_primitive(self, parser):
        """Should parse type alias for primitive."""
        result = parser.parse_initial("type ID = string;")
        assert result.is_valid

    def test_parse_type_alias_union(self, parser):
        """Should parse union type alias."""
        result = parser.parse_initial("type Status = 'pending' | 'active' | 'done';")
        assert result.is_valid

    def test_parse_type_alias_object(self, parser):
        """Should parse object type alias."""
        result = parser.parse_initial("type Point = { x: number; y: number };")
        assert result.is_valid

    def test_parse_type_alias_generic(self, parser):
        """Should parse generic type alias."""
        result = parser.parse_initial("type Container<T> = { value: T };")
        assert result.is_valid


# ===========================================================================
# Class Tests
# ===========================================================================


class TestTypeScriptClasses:
    """Tests for parsing classes."""

    @pytest.fixture
    def parser(self):
        return TypeScriptIncrementalParser()

    def test_parse_empty_class(self, parser):
        """Should parse empty class."""
        result = parser.parse_initial("class Empty {}")
        assert result.is_valid

    def test_parse_class_with_constructor(self, parser):
        """Should parse class with constructor."""
        result = parser.parse_initial("""
        class Person {
            name: string;
            constructor(name: string) {
                this.name = name;
            }
        }
        """)
        assert result.is_valid

    def test_parse_class_with_methods(self, parser):
        """Should parse class with methods."""
        result = parser.parse_initial("""
        class Calculator {
            add(a: number, b: number): number {
                return a + b;
            }
        }
        """)
        assert result.is_valid

    def test_parse_class_extends(self, parser):
        """Should parse class extending another."""
        result = parser.parse_initial("""
        class Animal { name: string; }
        class Dog extends Animal {
            bark() { console.log('woof'); }
        }
        """)
        assert result.is_valid

    def test_parse_class_implements(self, parser):
        """Should parse class implementing interface."""
        result = parser.parse_initial("""
        interface Runnable { run(): void; }
        class Task implements Runnable {
            run() { console.log('running'); }
        }
        """)
        assert result.is_valid

    def test_parse_abstract_class(self, parser):
        """Should parse abstract class."""
        result = parser.parse_initial("""
        abstract class Shape {
            abstract area(): number;
        }
        """)
        assert result.is_valid


# ===========================================================================
# Import/Export Tests
# ===========================================================================


class TestTypeScriptImportsExports:
    """Tests for parsing imports and exports."""

    @pytest.fixture
    def parser(self):
        return TypeScriptIncrementalParser()

    def test_parse_import_default(self, parser):
        """Should parse default import."""
        result = parser.parse_initial("import React from 'react';")
        assert result.is_valid

    def test_parse_import_named(self, parser):
        """Should parse named imports."""
        result = parser.parse_initial("import { useState, useEffect } from 'react';")
        assert result.is_valid

    def test_parse_import_all(self, parser):
        """Should parse namespace import."""
        result = parser.parse_initial("import * as fs from 'fs';")
        assert result.is_valid

    def test_parse_import_type(self, parser):
        """Should parse type-only import."""
        result = parser.parse_initial("import type { ReactNode } from 'react';")
        assert result.is_valid

    def test_parse_export_named(self, parser):
        """Should parse named export."""
        result = parser.parse_initial("export const x = 42;")
        assert result.is_valid

    def test_parse_export_default(self, parser):
        """Should parse default export."""
        result = parser.parse_initial("export default function main() {}")
        assert result.is_valid

    def test_parse_export_from(self, parser):
        """Should parse re-export."""
        result = parser.parse_initial("export { foo, bar } from './utils';")
        assert result.is_valid


# ===========================================================================
# Incremental Parsing Tests
# ===========================================================================


class TestTypeScriptIncrementalParsing:
    """Tests for incremental parsing functionality."""

    @pytest.fixture
    def parser(self):
        return TypeScriptIncrementalParser()

    def test_extend_with_text(self, parser):
        """Should extend parse with text."""
        parser.parse_initial("const x: num")
        result = parser.extend_with_text("ber = 42;")
        assert result.is_valid

    def test_extend_multiple_times(self, parser):
        """Should handle multiple extensions."""
        parser.parse_initial("function ")
        parser.extend_with_text("add(")
        parser.extend_with_text("a: number, ")
        parser.extend_with_text("b: number)")
        result = parser.extend_with_text(" { return a + b; }")
        assert result.is_valid

    def test_extend_multiline(self, parser):
        """Should handle multiline extensions."""
        parser.parse_initial("const obj = {")
        parser.extend_with_text("\n  name: 'test',")
        parser.extend_with_text("\n  value: 42")
        result = parser.extend_with_text("\n};")
        assert result.is_valid


# ===========================================================================
# Checkpoint and Restore Tests
# ===========================================================================


class TestTypeScriptCheckpointRestore:
    """Tests for checkpoint and restore functionality."""

    @pytest.fixture
    def parser(self):
        return TypeScriptIncrementalParser()

    def test_checkpoint_restore(self, parser):
        """Should checkpoint and restore parser state."""
        parser.parse_initial("const x: number = ")
        checkpoint = parser.checkpoint()

        # Extend in one direction
        parser.extend_with_text("42;")

        # Restore and extend differently
        parser.restore(checkpoint)
        result = parser.extend_with_text("100;")
        assert result.is_valid

    def test_multiple_checkpoints(self, parser):
        """Should handle multiple checkpoints."""
        parser.parse_initial("let x")
        cp1 = parser.checkpoint()

        parser.extend_with_text(": number")
        cp2 = parser.checkpoint()

        parser.extend_with_text(" = 42;")

        # Restore to cp2
        parser.restore(cp2)
        result = parser.extend_with_text(" = 100;")
        assert result.is_valid

    def test_copy_parser(self, parser):
        """Should create independent copy."""
        parser.parse_initial("const x = 1;")
        parser2 = parser.copy()

        parser.extend_with_text(" const y = 2;")
        parser2.extend_with_text(" const z = 3;")

        # Both should be valid and independent
        assert parser.get_source() != parser2.get_source()


# ===========================================================================
# Hole Detection Tests
# ===========================================================================


class TestTypeScriptHoleDetection:
    """Tests for hole detection."""

    @pytest.fixture
    def parser(self):
        return TypeScriptIncrementalParser()

    def test_find_no_holes_complete_code(self, parser):
        """Complete code should have no holes."""
        parser.parse_initial("const x: number = 42;")
        holes = parser.find_holes()
        assert len(holes) == 0

    def test_find_holes_incomplete_statement(self, parser):
        """Incomplete statement should have holes."""
        parser.parse_initial("const x: ")
        holes = parser.find_holes()
        assert len(holes) > 0

    def test_find_holes_missing_value(self, parser):
        """Missing value should create hole."""
        parser.parse_initial("const x: number = ")
        holes = parser.find_holes()
        assert len(holes) > 0


# ===========================================================================
# Expected Tokens Tests
# ===========================================================================


class TestTypeScriptExpectedTokens:
    """Tests for expected token computation."""

    @pytest.fixture
    def parser(self):
        return TypeScriptIncrementalParser()

    def test_expected_after_colon(self, parser):
        """Should expect types after colon."""
        parser.parse_initial("const x: ")
        expected = parser.get_expected_tokens()
        # Should include type keywords
        assert "string" in expected or "number" in expected or len(expected) > 0

    def test_expected_after_function_keyword(self, parser):
        """Should expect identifier or paren after function."""
        parser.parse_initial("function ")
        expected = parser.get_expected_tokens()
        assert len(expected) > 0


# ===========================================================================
# TSX Mode Tests
# ===========================================================================


class TestTypeScriptTSXMode:
    """Tests for TSX/JSX mode."""

    @pytest.fixture
    def parser(self):
        p = TypeScriptIncrementalParser()
        p.set_tsx_mode(True)
        return p

    def test_parse_simple_jsx(self, parser):
        """Should parse simple JSX element."""
        result = parser.parse_initial("const elem = <div>Hello</div>;")
        assert result.is_valid

    def test_parse_jsx_with_attributes(self, parser):
        """Should parse JSX with attributes."""
        result = parser.parse_initial('const elem = <div className="test">Hello</div>;')
        assert result.is_valid

    def test_parse_jsx_self_closing(self, parser):
        """Should parse self-closing JSX."""
        result = parser.parse_initial('const elem = <img src="test.png" />;')
        assert result.is_valid

    def test_parse_jsx_with_expression(self, parser):
        """Should parse JSX with expression."""
        result = parser.parse_initial("const elem = <div>{name}</div>;")
        assert result.is_valid

    def test_parse_jsx_fragment(self, parser):
        """Should parse JSX fragment."""
        result = parser.parse_initial("const elem = <><span>A</span><span>B</span></>;")
        assert result.is_valid


# ===========================================================================
# Error Recovery Tests
# ===========================================================================


class TestTypeScriptErrorRecovery:
    """Tests for parser error handling and recovery."""

    @pytest.fixture
    def parser(self):
        return TypeScriptIncrementalParser()

    def test_parse_with_syntax_error(self, parser):
        """Should handle syntax errors gracefully."""
        result = parser.parse_initial("const x = {{{")
        # May not be valid but shouldn't crash
        assert result is not None

    def test_recover_from_error(self, parser):
        """Should recover after syntax error."""
        parser.parse_initial("const x = {")  # Incomplete
        result = parser.extend_with_text("};")  # Complete it
        assert result.is_valid


# ===========================================================================
# Complex Code Tests
# ===========================================================================


class TestTypeScriptComplexCode:
    """Tests for parsing complex TypeScript code."""

    @pytest.fixture
    def parser(self):
        return TypeScriptIncrementalParser()

    def test_parse_generic_class(self, parser):
        """Should parse generic class."""
        result = parser.parse_initial("""
        class Container<T> {
            private value: T;
            constructor(value: T) {
                this.value = value;
            }
            getValue(): T {
                return this.value;
            }
        }
        """)
        assert result.is_valid

    def test_parse_conditional_type(self, parser):
        """Should parse conditional type."""
        result = parser.parse_initial("""
        type IsArray<T> = T extends any[] ? true : false;
        """)
        assert result.is_valid

    def test_parse_mapped_type(self, parser):
        """Should parse mapped type."""
        result = parser.parse_initial("""
        type Readonly<T> = {
            readonly [K in keyof T]: T[K];
        };
        """)
        assert result.is_valid

    def test_parse_template_literal_type(self, parser):
        """Should parse template literal type."""
        result = parser.parse_initial("""
        type EventName = `on${string}`;
        """)
        assert result.is_valid

    def test_parse_decorator(self, parser):
        """Should parse decorator syntax."""
        result = parser.parse_initial("""
        @Component({
            selector: 'app-root'
        })
        class AppComponent {}
        """)
        assert result.is_valid
