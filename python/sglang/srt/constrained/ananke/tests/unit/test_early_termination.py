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
"""Tests for early termination functionality."""

import pytest
from dataclasses import dataclass, field
from typing import Optional, List
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path

# Add parent directories to path for standalone testing
ananke_dir = Path(__file__).parent.parent.parent
if str(ananke_dir) not in sys.path:
    sys.path.insert(0, str(ananke_dir))


# =============================================================================
# Mock Classes for Testing
# =============================================================================


@dataclass
class MockConstraintSpec:
    """Mock constraint specification."""
    regex: Optional[str] = None
    language: Optional[str] = None
    enable_early_termination: bool = True


@dataclass
class MockGenerationContext:
    """Mock generation context."""
    generated_text: str = ""
    generated_tokens: List[int] = field(default_factory=list)
    position: int = 0
    vocab_size: int = 32000
    device: str = "cpu"

    def extend(self, token: int, token_text: str) -> "MockGenerationContext":
        """Extend context with new token."""
        return MockGenerationContext(
            generated_text=self.generated_text + token_text,
            generated_tokens=self.generated_tokens + [token],
            position=self.position + 1,
            vocab_size=self.vocab_size,
            device=self.device,
        )


class MockAnankeGrammar:
    """Mock AnankeGrammar for testing early termination methods."""

    def __init__(
        self,
        language: str = "python",
        constraint_spec: Optional[MockConstraintSpec] = None,
    ):
        self.language = language
        self.constraint_spec = constraint_spec
        self.context = MockGenerationContext()
        self.finished = False
        self._enable_early_termination = True
        self._early_termination_triggered = False

    def _check_regex_satisfied(self) -> bool:
        """Check if the current output fully satisfies the regex constraint."""
        if self.constraint_spec is None or not self.constraint_spec.regex:
            return False

        generated_text = self.context.generated_text
        if not generated_text:
            return False

        if len(generated_text) < 10:
            return False

        regex_pattern = self.constraint_spec.regex

        import re
        try:
            if regex_pattern.startswith("^") and regex_pattern.endswith("$"):
                match = re.fullmatch(regex_pattern, generated_text)
            elif regex_pattern.startswith("^"):
                match = re.match(regex_pattern, generated_text)
            else:
                match = re.search(regex_pattern, generated_text)

            return match is not None

        except re.error:
            return False

    def _is_natural_boundary(self) -> bool:
        """Check if the current position is a natural code boundary."""
        generated_text = self.context.generated_text
        if not generated_text:
            return False

        text = generated_text.rstrip()
        if not text:
            return False

        boundaries = {
            "python": ["\n\n", ":\n", "\n    pass\n", "\nreturn ", "\n    return "],
            "rust": ["}\n", ";\n", "\n}\n"],
            "go": ["}\n", ";\n", "\n}\n"],
            "typescript": ["}\n", ";\n", "\n}\n"],
            "kotlin": ["}\n", "\n}\n"],
            "swift": ["}\n", "\n}\n"],
            "zig": ["}\n", ";\n", "\n}\n"],
        }

        lang_boundaries = boundaries.get(self.language, ["\n\n", "}\n", ";\n"])

        for boundary in lang_boundaries:
            boundary_stripped = boundary.rstrip()
            # Only check non-empty stripped boundaries (empty string matches everything)
            if boundary_stripped and text.endswith(boundary_stripped):
                return True

        last_line = text.split("\n")[-1] if "\n" in text else text
        last_line_stripped = last_line.strip()  # Remove leading and trailing whitespace

        # Endings that the line must END WITH
        complete_endings = {
            "python": [":", "pass"],
            "rust": [";", "}", "),", ");"],
            "go": ["}", ";"],
            "typescript": [";", "}", "),", ");"],
            "kotlin": ["}", "),", ");"],
            "swift": ["}", "),", ");"],
            "zig": [";", "}", "),", ");"],
        }

        lang_endings = complete_endings.get(self.language, [";", "}"])

        for ending in lang_endings:
            if last_line_stripped.endswith(ending):
                return True

        # Keywords that the line must START WITH (Python-specific)
        if self.language == "python":
            statement_keywords = ["return", "break", "continue", "raise"]
            for keyword in statement_keywords:
                if last_line_stripped.startswith(keyword):
                    return True

        return False


# =============================================================================
# Test _check_regex_satisfied
# =============================================================================


class TestCheckRegexSatisfied:
    """Tests for _check_regex_satisfied method."""

    def test_no_constraint_spec(self):
        """Test with no constraint spec."""
        grammar = MockAnankeGrammar()
        grammar.constraint_spec = None
        assert grammar._check_regex_satisfied() is False

    def test_no_regex(self):
        """Test with constraint spec but no regex."""
        grammar = MockAnankeGrammar(
            constraint_spec=MockConstraintSpec(regex=None)
        )
        assert grammar._check_regex_satisfied() is False

    def test_empty_generated_text(self):
        """Test with empty generated text."""
        grammar = MockAnankeGrammar(
            constraint_spec=MockConstraintSpec(regex=r"def \w+\(")
        )
        grammar.context = MockGenerationContext(generated_text="")
        assert grammar._check_regex_satisfied() is False

    def test_short_generated_text(self):
        """Test with text shorter than 10 characters."""
        grammar = MockAnankeGrammar(
            constraint_spec=MockConstraintSpec(regex=r"def")
        )
        grammar.context = MockGenerationContext(generated_text="def foo")
        assert grammar._check_regex_satisfied() is False

    def test_anchored_pattern_matches(self):
        """Test with anchored pattern that matches."""
        grammar = MockAnankeGrammar(
            constraint_spec=MockConstraintSpec(regex=r"^def \w+\(.*\):")
        )
        grammar.context = MockGenerationContext(
            generated_text="def fibonacci(n):"
        )
        assert grammar._check_regex_satisfied() is True

    def test_anchored_pattern_no_match(self):
        """Test with anchored pattern that doesn't match."""
        grammar = MockAnankeGrammar(
            constraint_spec=MockConstraintSpec(regex=r"^class \w+:")
        )
        grammar.context = MockGenerationContext(
            generated_text="def fibonacci(n):"
        )
        assert grammar._check_regex_satisfied() is False

    def test_fully_anchored_pattern(self):
        """Test with pattern anchored at both ends."""
        grammar = MockAnankeGrammar(
            constraint_spec=MockConstraintSpec(regex=r"^def \w+\(\):\s*pass$")
        )
        grammar.context = MockGenerationContext(
            generated_text="def foo(): pass"
        )
        assert grammar._check_regex_satisfied() is True

    def test_fully_anchored_partial_match(self):
        """Test fully anchored pattern with partial match."""
        grammar = MockAnankeGrammar(
            constraint_spec=MockConstraintSpec(regex=r"^def \w+\(\)$")
        )
        grammar.context = MockGenerationContext(
            generated_text="def foo(): pass"  # Has extra content
        )
        assert grammar._check_regex_satisfied() is False

    def test_unanchored_pattern(self):
        """Test with unanchored pattern (search anywhere)."""
        grammar = MockAnankeGrammar(
            constraint_spec=MockConstraintSpec(regex=r"return \d+")
        )
        grammar.context = MockGenerationContext(
            generated_text="def foo():\n    return 42"
        )
        assert grammar._check_regex_satisfied() is True

    def test_invalid_regex(self):
        """Test with invalid regex pattern."""
        grammar = MockAnankeGrammar(
            constraint_spec=MockConstraintSpec(regex=r"[invalid(")
        )
        grammar.context = MockGenerationContext(
            generated_text="some generated text"
        )
        assert grammar._check_regex_satisfied() is False


# =============================================================================
# Test _is_natural_boundary
# =============================================================================


class TestIsNaturalBoundary:
    """Tests for _is_natural_boundary method."""

    def test_empty_text(self):
        """Test with empty generated text."""
        grammar = MockAnankeGrammar()
        grammar.context = MockGenerationContext(generated_text="")
        assert grammar._is_natural_boundary() is False

    def test_whitespace_only(self):
        """Test with whitespace-only text."""
        grammar = MockAnankeGrammar()
        grammar.context = MockGenerationContext(generated_text="   \n\n  ")
        assert grammar._is_natural_boundary() is False

    # Python boundaries
    def test_python_colon_newline(self):
        """Test Python colon boundary."""
        grammar = MockAnankeGrammar(language="python")
        grammar.context = MockGenerationContext(
            generated_text="def foo():"
        )
        assert grammar._is_natural_boundary() is True

    def test_python_double_newline(self):
        """Test Python double newline boundary."""
        grammar = MockAnankeGrammar(language="python")
        grammar.context = MockGenerationContext(
            generated_text="def foo():\n    pass\n"
        )
        # Ends with \n after stripping, check for pass ending
        assert grammar._is_natural_boundary() is True

    def test_python_pass(self):
        """Test Python pass statement."""
        grammar = MockAnankeGrammar(language="python")
        grammar.context = MockGenerationContext(
            generated_text="class Empty:\n    pass"
        )
        assert grammar._is_natural_boundary() is True

    def test_python_return(self):
        """Test Python return statement."""
        grammar = MockAnankeGrammar(language="python")
        grammar.context = MockGenerationContext(
            generated_text="def foo():\n    return"
        )
        assert grammar._is_natural_boundary() is True

    # Rust boundaries
    def test_rust_closing_brace(self):
        """Test Rust closing brace boundary."""
        grammar = MockAnankeGrammar(language="rust")
        grammar.context = MockGenerationContext(
            generated_text="fn foo() {\n    println!(\"hello\");\n}"
        )
        assert grammar._is_natural_boundary() is True

    def test_rust_semicolon(self):
        """Test Rust semicolon boundary."""
        grammar = MockAnankeGrammar(language="rust")
        grammar.context = MockGenerationContext(
            generated_text="let x = 42;"
        )
        assert grammar._is_natural_boundary() is True

    # Go boundaries
    def test_go_closing_brace(self):
        """Test Go closing brace boundary."""
        grammar = MockAnankeGrammar(language="go")
        grammar.context = MockGenerationContext(
            generated_text="func foo() {\n    return\n}"
        )
        assert grammar._is_natural_boundary() is True

    # TypeScript boundaries
    def test_typescript_semicolon(self):
        """Test TypeScript semicolon boundary."""
        grammar = MockAnankeGrammar(language="typescript")
        grammar.context = MockGenerationContext(
            generated_text="const x: number = 42;"
        )
        assert grammar._is_natural_boundary() is True

    def test_typescript_closing_paren_comma(self):
        """Test TypeScript ),  pattern."""
        grammar = MockAnankeGrammar(language="typescript")
        grammar.context = MockGenerationContext(
            generated_text="foo(1, 2),"
        )
        assert grammar._is_natural_boundary() is True

    # Zig boundaries
    def test_zig_closing_brace(self):
        """Test Zig closing brace boundary."""
        grammar = MockAnankeGrammar(language="zig")
        grammar.context = MockGenerationContext(
            generated_text="fn foo() void {\n    return;\n}"
        )
        assert grammar._is_natural_boundary() is True

    # Non-boundary cases
    def test_incomplete_statement(self):
        """Test incomplete statement (not at boundary)."""
        grammar = MockAnankeGrammar(language="python")
        grammar.context = MockGenerationContext(
            generated_text="def foo(x, y"  # Mid-parameter list
        )
        assert grammar._is_natural_boundary() is False

    def test_mid_expression(self):
        """Test mid-expression (not at boundary)."""
        grammar = MockAnankeGrammar(language="rust")
        grammar.context = MockGenerationContext(
            generated_text="let x = 1 +"  # Mid-expression
        )
        assert grammar._is_natural_boundary() is False

    # Default language fallback
    def test_unknown_language_default(self):
        """Test unknown language uses default boundaries."""
        grammar = MockAnankeGrammar(language="unknown_lang")
        grammar.context = MockGenerationContext(
            generated_text="some code}"
        )
        assert grammar._is_natural_boundary() is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestEarlyTerminationIntegration:
    """Integration tests for early termination."""

    def test_python_function_complete(self):
        """Test early termination for complete Python function."""
        grammar = MockAnankeGrammar(
            language="python",
            constraint_spec=MockConstraintSpec(regex=r"^def \w+\(.*\):\s*\n\s+.*")
        )
        grammar.context = MockGenerationContext(
            generated_text="def fibonacci(n):\n    if n <= 1:\n        return n"
        )

        # Both conditions should be met
        assert grammar._check_regex_satisfied() is True
        assert grammar._is_natural_boundary() is True

    def test_rust_function_complete(self):
        """Test early termination for complete Rust function."""
        grammar = MockAnankeGrammar(
            language="rust",
            constraint_spec=MockConstraintSpec(regex=r"fn \w+\(.*\).*\{")
        )
        grammar.context = MockGenerationContext(
            generated_text='fn greet(name: &str) {\n    println!("Hello, {}!", name);\n}'
        )

        assert grammar._check_regex_satisfied() is True
        assert grammar._is_natural_boundary() is True

    def test_typescript_statement_complete(self):
        """Test early termination for TypeScript statement."""
        grammar = MockAnankeGrammar(
            language="typescript",
            constraint_spec=MockConstraintSpec(regex=r"const \w+:\s*\w+\s*=")
        )
        grammar.context = MockGenerationContext(
            generated_text="const count: number = 42;"
        )

        assert grammar._check_regex_satisfied() is True
        assert grammar._is_natural_boundary() is True

    def test_regex_satisfied_not_at_boundary(self):
        """Test regex satisfied but not at natural boundary."""
        grammar = MockAnankeGrammar(
            language="python",
            constraint_spec=MockConstraintSpec(regex=r"def \w+\(")
        )
        grammar.context = MockGenerationContext(
            generated_text="def fibonacci("  # Mid-signature, not a boundary
        )

        assert grammar._check_regex_satisfied() is True
        assert grammar._is_natural_boundary() is False

    def test_at_boundary_not_regex_satisfied(self):
        """Test at natural boundary but regex not satisfied."""
        grammar = MockAnankeGrammar(
            language="python",
            constraint_spec=MockConstraintSpec(regex=r"^class \w+:")
        )
        grammar.context = MockGenerationContext(
            generated_text="def foo():"  # Boundary but wrong pattern
        )

        assert grammar._check_regex_satisfied() is False
        assert grammar._is_natural_boundary() is True


# =============================================================================
# ConstraintSpec Tests
# =============================================================================


class TestConstraintSpecEarlyTermination:
    """Tests for early termination in ConstraintSpec."""

    def test_default_enabled(self):
        """Test early termination is enabled by default."""
        from spec.constraint_spec import ConstraintSpec

        spec = ConstraintSpec()
        assert spec.enable_early_termination is True

    def test_can_disable(self):
        """Test early termination can be disabled."""
        from spec.constraint_spec import ConstraintSpec

        spec = ConstraintSpec(enable_early_termination=False)
        assert spec.enable_early_termination is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
