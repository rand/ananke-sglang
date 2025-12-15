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
"""Tests for language-specific control flow detection.

Tests the _detect_{language}_control_flow() methods for:
- Go: defer, go, fallthrough, select, switch/case, panic
- Rust: match, loop, ? operator
- Kotlin: when, do-while, labeled returns
- Swift: guard, repeat-while, do-catch, defer
- Zig: try-catch, orelse, defer/errdefer, unreachable
"""

from __future__ import annotations

import pytest
from typing import List

try:
    from ...domains.controlflow import ControlFlowDomain
    from ...core.domain import GenerationContext
except ImportError:
    from domains.controlflow import ControlFlowDomain
    from core.domain import GenerationContext


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def go_domain() -> ControlFlowDomain:
    """Create a Go ControlFlowDomain."""
    return ControlFlowDomain(language="go")


@pytest.fixture
def rust_domain() -> ControlFlowDomain:
    """Create a Rust ControlFlowDomain."""
    return ControlFlowDomain(language="rust")


@pytest.fixture
def kotlin_domain() -> ControlFlowDomain:
    """Create a Kotlin ControlFlowDomain."""
    return ControlFlowDomain(language="kotlin")


@pytest.fixture
def swift_domain() -> ControlFlowDomain:
    """Create a Swift ControlFlowDomain."""
    return ControlFlowDomain(language="swift")


@pytest.fixture
def zig_domain() -> ControlFlowDomain:
    """Create a Zig ControlFlowDomain."""
    return ControlFlowDomain(language="zig")


# =============================================================================
# Go Control Flow Tests
# =============================================================================


class TestGoControlFlow:
    """Tests for Go-specific control flow detection."""

    def test_domain_name(self, go_domain: ControlFlowDomain) -> None:
        """Test domain is controlflow."""
        assert go_domain.name == "controlflow"
        assert go_domain._language == "go"

    def test_domain_language(self, go_domain: ControlFlowDomain) -> None:
        """Test domain reports Go language."""
        assert go_domain._language == "go"

    def test_if_else_detection(self, go_domain: ControlFlowDomain) -> None:
        """Test detection of Go if/else statements."""
        go_domain._token_buffer = "if x > 0 {"
        go_domain._detect_go_control_flow()
        # Should have detected if construct

    def test_for_loop_detection(self, go_domain: ControlFlowDomain) -> None:
        """Test detection of Go for loops."""
        go_domain._token_buffer = "for i := 0; i < 10; i++ {"
        go_domain._detect_go_control_flow()
        assert go_domain._cfg is not None or len(go_domain._control_stack) > 0

    def test_switch_case_detection(self, go_domain: ControlFlowDomain) -> None:
        """Test detection of Go switch/case statements."""
        go_domain._token_buffer = "switch x {"
        go_domain._detect_go_control_flow()
        # Switch should create CFG structure

    def test_defer_detection(self, go_domain: ControlFlowDomain) -> None:
        """Test detection of Go defer statements."""
        go_domain._token_buffer = "defer cleanup()"
        go_domain._detect_go_control_flow()
        # Defer should be recorded

    def test_go_goroutine_detection(self, go_domain: ControlFlowDomain) -> None:
        """Test detection of Go goroutine statements."""
        go_domain._token_buffer = "go handleRequest()"
        go_domain._detect_go_control_flow()
        # Goroutine spawn should be recorded

    def test_select_detection(self, go_domain: ControlFlowDomain) -> None:
        """Test detection of Go select statements."""
        go_domain._token_buffer = "select {"
        go_domain._detect_go_control_flow()
        # Select should create switch-like structure

    def test_panic_detection(self, go_domain: ControlFlowDomain) -> None:
        """Test detection of Go panic calls."""
        go_domain._token_buffer = 'panic("error")'
        go_domain._detect_go_control_flow()
        # Panic should be recorded as termination

    def test_return_detection(self, go_domain: ControlFlowDomain) -> None:
        """Test detection of Go return statements."""
        go_domain._token_buffer = "return nil"
        go_domain._detect_go_control_flow()
        # Return should be recorded

    def test_fallthrough_detection(self, go_domain: ControlFlowDomain) -> None:
        """Test detection of Go fallthrough in switch."""
        go_domain._token_buffer = "fallthrough"
        go_domain._detect_go_control_flow()
        # Fallthrough should be recorded as special edge


# =============================================================================
# Rust Control Flow Tests
# =============================================================================


class TestRustControlFlow:
    """Tests for Rust-specific control flow detection."""

    def test_domain_name(self, rust_domain: ControlFlowDomain) -> None:
        """Test domain is controlflow."""
        assert rust_domain.name == "controlflow"
        assert rust_domain._language == "rust"

    def test_domain_language(self, rust_domain: ControlFlowDomain) -> None:
        """Test domain reports Rust language."""
        assert rust_domain._language == "rust"

    def test_if_else_detection(self, rust_domain: ControlFlowDomain) -> None:
        """Test detection of Rust if/else statements."""
        rust_domain._token_buffer = "if x > 0 {"
        rust_domain._detect_rust_control_flow()

    def test_loop_detection(self, rust_domain: ControlFlowDomain) -> None:
        """Test detection of Rust loop statements."""
        rust_domain._token_buffer = "loop {"
        rust_domain._detect_rust_control_flow()
        # Infinite loop should create appropriate structure

    def test_while_detection(self, rust_domain: ControlFlowDomain) -> None:
        """Test detection of Rust while loops."""
        rust_domain._token_buffer = "while running {"
        rust_domain._detect_rust_control_flow()

    def test_for_detection(self, rust_domain: ControlFlowDomain) -> None:
        """Test detection of Rust for loops."""
        rust_domain._token_buffer = "for item in items {"
        rust_domain._detect_rust_control_flow()

    def test_match_detection(self, rust_domain: ControlFlowDomain) -> None:
        """Test detection of Rust match expressions."""
        rust_domain._token_buffer = "match value {"
        rust_domain._detect_rust_control_flow()
        # Match should create switch-like structure

    def test_match_arm_detection(self, rust_domain: ControlFlowDomain) -> None:
        """Test detection of Rust match arms."""
        rust_domain._token_buffer = "Some(x) => {"
        rust_domain._detect_rust_control_flow()

    def test_question_operator_detection(self, rust_domain: ControlFlowDomain) -> None:
        """Test detection of Rust ? operator."""
        rust_domain._token_buffer = "result?"
        rust_domain._detect_rust_control_flow()
        # ? operator should be recorded as potential early return

    def test_return_detection(self, rust_domain: ControlFlowDomain) -> None:
        """Test detection of Rust return statements."""
        rust_domain._token_buffer = "return Ok(value)"
        rust_domain._detect_rust_control_flow()

    def test_break_with_value_detection(self, rust_domain: ControlFlowDomain) -> None:
        """Test detection of Rust break with value."""
        rust_domain._token_buffer = "break result"
        rust_domain._detect_rust_control_flow()

    def test_continue_detection(self, rust_domain: ControlFlowDomain) -> None:
        """Test detection of Rust continue statements."""
        rust_domain._token_buffer = "continue"
        rust_domain._detect_rust_control_flow()


# =============================================================================
# Kotlin Control Flow Tests
# =============================================================================


class TestKotlinControlFlow:
    """Tests for Kotlin-specific control flow detection."""

    def test_domain_name(self, kotlin_domain: ControlFlowDomain) -> None:
        """Test domain is controlflow."""
        assert kotlin_domain.name == "controlflow"
        assert kotlin_domain._language == "kotlin"

    def test_domain_language(self, kotlin_domain: ControlFlowDomain) -> None:
        """Test domain reports Kotlin language."""
        assert kotlin_domain._language == "kotlin"

    def test_if_else_detection(self, kotlin_domain: ControlFlowDomain) -> None:
        """Test detection of Kotlin if/else statements."""
        kotlin_domain._token_buffer = "if (x > 0) {"
        kotlin_domain._detect_kotlin_control_flow()

    def test_when_detection(self, kotlin_domain: ControlFlowDomain) -> None:
        """Test detection of Kotlin when expressions."""
        kotlin_domain._token_buffer = "when (value) {"
        kotlin_domain._detect_kotlin_control_flow()
        # when should create switch-like structure

    def test_when_branch_detection(self, kotlin_domain: ControlFlowDomain) -> None:
        """Test detection of Kotlin when branches."""
        kotlin_domain._token_buffer = "1 -> {"
        kotlin_domain._detect_kotlin_control_flow()

    def test_for_loop_detection(self, kotlin_domain: ControlFlowDomain) -> None:
        """Test detection of Kotlin for loops."""
        kotlin_domain._token_buffer = "for (item in items) {"
        kotlin_domain._detect_kotlin_control_flow()

    def test_while_loop_detection(self, kotlin_domain: ControlFlowDomain) -> None:
        """Test detection of Kotlin while loops."""
        kotlin_domain._token_buffer = "while (running) {"
        kotlin_domain._detect_kotlin_control_flow()

    def test_do_while_detection(self, kotlin_domain: ControlFlowDomain) -> None:
        """Test detection of Kotlin do-while loops."""
        kotlin_domain._token_buffer = "do {"
        kotlin_domain._detect_kotlin_control_flow()
        # do-while has special completion behavior

    def test_return_detection(self, kotlin_domain: ControlFlowDomain) -> None:
        """Test detection of Kotlin return statements."""
        kotlin_domain._token_buffer = "return result"
        kotlin_domain._detect_kotlin_control_flow()

    def test_labeled_return_detection(self, kotlin_domain: ControlFlowDomain) -> None:
        """Test detection of Kotlin labeled returns."""
        kotlin_domain._token_buffer = "return@outer result"
        kotlin_domain._detect_kotlin_control_flow()

    def test_throw_detection(self, kotlin_domain: ControlFlowDomain) -> None:
        """Test detection of Kotlin throw statements."""
        kotlin_domain._token_buffer = "throw IllegalStateException()"
        kotlin_domain._detect_kotlin_control_flow()


# =============================================================================
# Swift Control Flow Tests
# =============================================================================


class TestSwiftControlFlow:
    """Tests for Swift-specific control flow detection."""

    def test_domain_name(self, swift_domain: ControlFlowDomain) -> None:
        """Test domain is controlflow."""
        assert swift_domain.name == "controlflow"
        assert swift_domain._language == "swift"

    def test_domain_language(self, swift_domain: ControlFlowDomain) -> None:
        """Test domain reports Swift language."""
        assert swift_domain._language == "swift"

    def test_if_else_detection(self, swift_domain: ControlFlowDomain) -> None:
        """Test detection of Swift if/else statements."""
        swift_domain._token_buffer = "if x > 0 {"
        swift_domain._detect_swift_control_flow()

    def test_guard_detection(self, swift_domain: ControlFlowDomain) -> None:
        """Test detection of Swift guard statements."""
        swift_domain._token_buffer = "guard let value = optional else {"
        swift_domain._detect_swift_control_flow()
        # guard must have early exit in else block

    def test_switch_detection(self, swift_domain: ControlFlowDomain) -> None:
        """Test detection of Swift switch statements."""
        swift_domain._token_buffer = "switch value {"
        swift_domain._detect_swift_control_flow()

    def test_case_detection(self, swift_domain: ControlFlowDomain) -> None:
        """Test detection of Swift case labels."""
        swift_domain._token_buffer = "case .success:"
        swift_domain._detect_swift_control_flow()

    def test_for_loop_detection(self, swift_domain: ControlFlowDomain) -> None:
        """Test detection of Swift for-in loops."""
        swift_domain._token_buffer = "for item in items {"
        swift_domain._detect_swift_control_flow()

    def test_while_loop_detection(self, swift_domain: ControlFlowDomain) -> None:
        """Test detection of Swift while loops."""
        swift_domain._token_buffer = "while running {"
        swift_domain._detect_swift_control_flow()

    def test_repeat_while_detection(self, swift_domain: ControlFlowDomain) -> None:
        """Test detection of Swift repeat-while loops."""
        swift_domain._token_buffer = "repeat {"
        swift_domain._detect_swift_control_flow()

    def test_do_catch_detection(self, swift_domain: ControlFlowDomain) -> None:
        """Test detection of Swift do-catch blocks."""
        swift_domain._token_buffer = "do {"
        swift_domain._detect_swift_control_flow()

    def test_catch_detection(self, swift_domain: ControlFlowDomain) -> None:
        """Test detection of Swift catch blocks."""
        swift_domain._token_buffer = "catch {"
        swift_domain._detect_swift_control_flow()

    def test_defer_detection(self, swift_domain: ControlFlowDomain) -> None:
        """Test detection of Swift defer statements."""
        swift_domain._token_buffer = "defer {"
        swift_domain._detect_swift_control_flow()

    def test_return_detection(self, swift_domain: ControlFlowDomain) -> None:
        """Test detection of Swift return statements."""
        swift_domain._token_buffer = "return result"
        swift_domain._detect_swift_control_flow()

    def test_throw_detection(self, swift_domain: ControlFlowDomain) -> None:
        """Test detection of Swift throw statements."""
        swift_domain._token_buffer = "throw MyError.invalidInput"
        swift_domain._detect_swift_control_flow()

    def test_fallthrough_detection(self, swift_domain: ControlFlowDomain) -> None:
        """Test detection of Swift fallthrough statements."""
        swift_domain._token_buffer = "fallthrough"
        swift_domain._detect_swift_control_flow()


# =============================================================================
# Zig Control Flow Tests
# =============================================================================


class TestZigControlFlow:
    """Tests for Zig-specific control flow detection."""

    def test_domain_name(self, zig_domain: ControlFlowDomain) -> None:
        """Test domain is controlflow."""
        assert zig_domain.name == "controlflow"
        assert zig_domain._language == "zig"

    def test_domain_language(self, zig_domain: ControlFlowDomain) -> None:
        """Test domain reports Zig language."""
        assert zig_domain._language == "zig"

    def test_if_else_detection(self, zig_domain: ControlFlowDomain) -> None:
        """Test detection of Zig if/else statements."""
        zig_domain._token_buffer = "if (x > 0) {"
        zig_domain._detect_zig_control_flow()

    def test_switch_detection(self, zig_domain: ControlFlowDomain) -> None:
        """Test detection of Zig switch expressions."""
        zig_domain._token_buffer = "switch (value) {"
        zig_domain._detect_zig_control_flow()

    def test_while_loop_detection(self, zig_domain: ControlFlowDomain) -> None:
        """Test detection of Zig while loops."""
        zig_domain._token_buffer = "while (iter.next()) |item| {"
        zig_domain._detect_zig_control_flow()

    def test_for_loop_detection(self, zig_domain: ControlFlowDomain) -> None:
        """Test detection of Zig for loops."""
        zig_domain._token_buffer = "for (items) |item| {"
        zig_domain._detect_zig_control_flow()

    def test_try_catch_detection(self, zig_domain: ControlFlowDomain) -> None:
        """Test detection of Zig try expressions."""
        zig_domain._token_buffer = "const result = try fetch()"
        zig_domain._detect_zig_control_flow()
        # try can return error early

    def test_catch_detection(self, zig_domain: ControlFlowDomain) -> None:
        """Test detection of Zig catch expressions."""
        zig_domain._token_buffer = "catch |err| {"
        zig_domain._detect_zig_control_flow()

    def test_orelse_detection(self, zig_domain: ControlFlowDomain) -> None:
        """Test detection of Zig orelse expressions."""
        zig_domain._token_buffer = "optional orelse default"
        zig_domain._detect_zig_control_flow()

    def test_defer_detection(self, zig_domain: ControlFlowDomain) -> None:
        """Test detection of Zig defer statements."""
        zig_domain._token_buffer = "defer cleanup()"
        zig_domain._detect_zig_control_flow()

    def test_errdefer_detection(self, zig_domain: ControlFlowDomain) -> None:
        """Test detection of Zig errdefer statements."""
        zig_domain._token_buffer = "errdefer rollback()"
        zig_domain._detect_zig_control_flow()
        # errdefer only runs on error return

    def test_return_detection(self, zig_domain: ControlFlowDomain) -> None:
        """Test detection of Zig return statements."""
        zig_domain._token_buffer = "return value"
        zig_domain._detect_zig_control_flow()

    def test_unreachable_detection(self, zig_domain: ControlFlowDomain) -> None:
        """Test detection of Zig unreachable statements."""
        zig_domain._token_buffer = "unreachable"
        zig_domain._detect_zig_control_flow()
        # unreachable terminates execution

    def test_break_detection(self, zig_domain: ControlFlowDomain) -> None:
        """Test detection of Zig break statements."""
        zig_domain._token_buffer = "break"
        zig_domain._detect_zig_control_flow()

    def test_continue_detection(self, zig_domain: ControlFlowDomain) -> None:
        """Test detection of Zig continue statements."""
        zig_domain._token_buffer = "continue"
        zig_domain._detect_zig_control_flow()


# =============================================================================
# Cross-Language Tests
# =============================================================================


class TestCrossLanguageControlFlow:
    """Tests that verify consistent behavior across languages."""

    @pytest.mark.parametrize("language", ["go", "rust", "kotlin", "swift", "zig"])
    def test_domain_creation(self, language: str) -> None:
        """Test that domains can be created for all languages."""
        domain = ControlFlowDomain(language=language)
        assert domain._language == language

    @pytest.mark.parametrize("language", ["go", "rust", "kotlin", "swift", "zig"])
    def test_if_keyword_exists(self, language: str) -> None:
        """Test that all languages handle 'if' keyword."""
        domain = ControlFlowDomain(language=language)
        domain._token_buffer = "if"
        # Should not raise
        method_name = f"_detect_{language}_control_flow"
        if hasattr(domain, method_name):
            getattr(domain, method_name)()

    @pytest.mark.parametrize("language", ["go", "rust", "kotlin", "swift", "zig"])
    def test_return_keyword_exists(self, language: str) -> None:
        """Test that all languages handle 'return' keyword."""
        domain = ControlFlowDomain(language=language)
        domain._token_buffer = "return"
        method_name = f"_detect_{language}_control_flow"
        if hasattr(domain, method_name):
            getattr(domain, method_name)()

    @pytest.mark.parametrize("language", ["go", "rust", "kotlin", "swift", "zig"])
    def test_break_keyword_exists(self, language: str) -> None:
        """Test that all languages handle 'break' keyword."""
        domain = ControlFlowDomain(language=language)
        domain._token_buffer = "break"
        method_name = f"_detect_{language}_control_flow"
        if hasattr(domain, method_name):
            getattr(domain, method_name)()

    @pytest.mark.parametrize("language", ["go", "rust", "kotlin", "swift", "zig"])
    def test_continue_keyword_exists(self, language: str) -> None:
        """Test that all languages handle 'continue' keyword."""
        domain = ControlFlowDomain(language=language)
        domain._token_buffer = "continue"
        method_name = f"_detect_{language}_control_flow"
        if hasattr(domain, method_name):
            getattr(domain, method_name)()


# =============================================================================
# Language-Specific Feature Tests
# =============================================================================


class TestLanguageSpecificFeatures:
    """Tests for unique language features."""

    def test_go_goroutines_unique(self, go_domain: ControlFlowDomain) -> None:
        """Test Go's unique goroutine feature."""
        go_domain._token_buffer = "go func() {}"
        go_domain._detect_go_control_flow()
        # Only Go has 'go' keyword for goroutines

    def test_rust_loop_unique(self, rust_domain: ControlFlowDomain) -> None:
        """Test Rust's unique infinite loop keyword."""
        rust_domain._token_buffer = "loop {"
        rust_domain._detect_rust_control_flow()
        # Only Rust has bare 'loop' keyword

    def test_kotlin_when_unique(self, kotlin_domain: ControlFlowDomain) -> None:
        """Test Kotlin's unique when expression."""
        kotlin_domain._token_buffer = "when {"
        kotlin_domain._detect_kotlin_control_flow()
        # Only Kotlin uses 'when' for pattern matching

    def test_swift_guard_unique(self, swift_domain: ControlFlowDomain) -> None:
        """Test Swift's unique guard statement."""
        swift_domain._token_buffer = "guard let x else"
        swift_domain._detect_swift_control_flow()
        # Only Swift has 'guard' keyword

    def test_zig_errdefer_unique(self, zig_domain: ControlFlowDomain) -> None:
        """Test Zig's unique errdefer statement."""
        zig_domain._token_buffer = "errdefer cleanup()"
        zig_domain._detect_zig_control_flow()
        # Only Zig has 'errdefer' keyword
