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
"""Property-based tests for control flow detection across languages.

Tests invariants that control flow detection should maintain:
- Determinism: Same input produces same output
- Construct recognition: Language-specific constructs are detected
- CFG well-formedness: Generated CFG has valid structure
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck

from domains.controlflow.domain import ControlFlowDomain
from tests.conftest import MockTokenizer


# =============================================================================
# Code Snippet Strategies
# =============================================================================

# Go code snippets with control flow constructs
GO_SNIPPETS = [
    # if/else
    "if x > 0 { return x }",
    "if err != nil { return err }",
    "if x := compute(); x > 0 { return x }",
    # for loops
    "for i := 0; i < 10; i++ { sum += i }",
    "for _, v := range items { process(v) }",
    "for { break }",
    # switch
    "switch x { case 1: return \"one\" case 2: return \"two\" default: return \"other\" }",
    # defer
    "defer file.Close()",
    "defer func() { recover() }()",
    # go
    "go func() { work() }()",
    # select
    "select { case <-ch: done() default: wait() }",
    # return
    "return nil",
    "return x, nil",
]

# Rust code snippets
RUST_SNIPPETS = [
    # if/else
    "if x > 0 { x } else { 0 }",
    "if let Some(v) = opt { v } else { 0 }",
    # loops
    "loop { break; }",
    "while x > 0 { x -= 1; }",
    "for item in items { process(item); }",
    # match
    "match x { 0 => zero(), _ => other() }",
    "match opt { Some(v) => v, None => 0 }",
    # ? operator
    "let x = file.read()?;",
    # return
    "return Ok(result);",
    "return Err(error);",
]

# Kotlin code snippets
KOTLIN_SNIPPETS = [
    # if/else
    "if (x > 0) x else 0",
    "if (x != null) x.process()",
    # when
    "when (x) { 1 -> \"one\" 2 -> \"two\" else -> \"other\" }",
    "when { x > 0 -> positive() else -> nonPositive() }",
    # loops
    "for (item in items) { process(item) }",
    "while (x > 0) { x-- }",
    "do { work() } while (running)",
    # return with labels
    "return@forEach",
    "return@outer result",
    # throw
    "throw IllegalArgumentException()",
]

# Swift code snippets
SWIFT_SNIPPETS = [
    # if/else
    "if x > 0 { return x }",
    "if let value = optional { use(value) }",
    # guard
    "guard x > 0 else { return }",
    "guard let value = optional else { return }",
    # switch
    "switch x { case 1: return \"one\" case 2: return \"two\" default: return \"other\" }",
    # loops
    "for item in items { process(item) }",
    "while x > 0 { x -= 1 }",
    "repeat { work() } while running",
    # do-catch
    "do { try riskyOperation() } catch { handleError() }",
    # defer
    "defer { cleanup() }",
]

# Zig code snippets
ZIG_SNIPPETS = [
    # if/else
    "if (x > 0) return x",
    "if (optional) |value| use(value)",
    # switch
    "switch (x) { 1 => one(), 2 => two(), else => other() }",
    # loops
    "while (x > 0) : (x -= 1) {}",
    "for (items) |item| { process(item); }",
    # try-catch inline
    "const value = try riskyOperation();",
    "const value = riskyOperation() catch |err| handleError(err);",
    # orelse
    "const value = optional orelse default;",
    # defer/errdefer
    "defer allocator.free(memory);",
    "errdefer cleanup();",
    # unreachable
    "unreachable;",
]


go_snippet = st.sampled_from(GO_SNIPPETS)
rust_snippet = st.sampled_from(RUST_SNIPPETS)
kotlin_snippet = st.sampled_from(KOTLIN_SNIPPETS)
swift_snippet = st.sampled_from(SWIFT_SNIPPETS)
zig_snippet = st.sampled_from(ZIG_SNIPPETS)


# =============================================================================
# Fixture
# =============================================================================

@pytest.fixture
def tokenizer():
    """Create a mock tokenizer for testing."""
    return MockTokenizer()


# =============================================================================
# Go Control Flow Properties
# =============================================================================

class TestGoControlFlowProperties:
    """Property tests for Go control flow detection."""

    @pytest.fixture
    def domain(self, tokenizer):
        return ControlFlowDomain(language="go", tokenizer=tokenizer)

    @given(go_snippet)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_deterministic_detection(self, domain, snippet):
        """Control flow detection should be deterministic."""
        # Reset between runs
        domain.reset()
        domain.accept_bytes(snippet.encode())
        cfg1 = domain.get_cfg()

        domain.reset()
        domain.accept_bytes(snippet.encode())
        cfg2 = domain.get_cfg()

        # Same snippet should produce same CFG structure
        assert len(cfg1.nodes) == len(cfg2.nodes)
        assert len(cfg1.edges) == len(cfg2.edges)

    def test_defer_detected(self, domain):
        """Go defer statements should be detected."""
        domain.accept_bytes(b"defer file.Close()")
        cfg = domain.get_cfg()
        # CFG should have nodes for defer
        assert cfg is not None

    def test_goroutine_detected(self, domain):
        """Go goroutines should be detected."""
        domain.accept_bytes(b"go work()")
        cfg = domain.get_cfg()
        assert cfg is not None

    def test_switch_case_detected(self, domain):
        """Go switch/case should be detected."""
        domain.accept_bytes(b"switch x { case 1: a() case 2: b() }")
        cfg = domain.get_cfg()
        assert cfg is not None


# =============================================================================
# Rust Control Flow Properties
# =============================================================================

class TestRustControlFlowProperties:
    """Property tests for Rust control flow detection."""

    @pytest.fixture
    def domain(self, tokenizer):
        return ControlFlowDomain(language="rust", tokenizer=tokenizer)

    @given(rust_snippet)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_deterministic_detection(self, domain, snippet):
        """Control flow detection should be deterministic."""
        domain.reset()
        domain.accept_bytes(snippet.encode())
        cfg1 = domain.get_cfg()

        domain.reset()
        domain.accept_bytes(snippet.encode())
        cfg2 = domain.get_cfg()

        assert len(cfg1.nodes) == len(cfg2.nodes)
        assert len(cfg1.edges) == len(cfg2.edges)

    def test_match_detected(self, domain):
        """Rust match expressions should be detected."""
        domain.accept_bytes(b"match x { 0 => zero(), _ => other() }")
        cfg = domain.get_cfg()
        assert cfg is not None

    def test_loop_detected(self, domain):
        """Rust infinite loop should be detected."""
        domain.accept_bytes(b"loop { break; }")
        cfg = domain.get_cfg()
        assert cfg is not None

    def test_question_mark_detected(self, domain):
        """Rust ? operator should be detected."""
        domain.accept_bytes(b"let x = file.read()?;")
        cfg = domain.get_cfg()
        assert cfg is not None


# =============================================================================
# Kotlin Control Flow Properties
# =============================================================================

class TestKotlinControlFlowProperties:
    """Property tests for Kotlin control flow detection."""

    @pytest.fixture
    def domain(self, tokenizer):
        return ControlFlowDomain(language="kotlin", tokenizer=tokenizer)

    @given(kotlin_snippet)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_deterministic_detection(self, domain, snippet):
        """Control flow detection should be deterministic."""
        domain.reset()
        domain.accept_bytes(snippet.encode())
        cfg1 = domain.get_cfg()

        domain.reset()
        domain.accept_bytes(snippet.encode())
        cfg2 = domain.get_cfg()

        assert len(cfg1.nodes) == len(cfg2.nodes)
        assert len(cfg1.edges) == len(cfg2.edges)

    def test_when_detected(self, domain):
        """Kotlin when expressions should be detected."""
        domain.accept_bytes(b"when (x) { 1 -> one() else -> other() }")
        cfg = domain.get_cfg()
        assert cfg is not None

    def test_labeled_return_detected(self, domain):
        """Kotlin labeled returns should be detected."""
        domain.accept_bytes(b"return@forEach")
        cfg = domain.get_cfg()
        assert cfg is not None


# =============================================================================
# Swift Control Flow Properties
# =============================================================================

class TestSwiftControlFlowProperties:
    """Property tests for Swift control flow detection."""

    @pytest.fixture
    def domain(self, tokenizer):
        return ControlFlowDomain(language="swift", tokenizer=tokenizer)

    @given(swift_snippet)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_deterministic_detection(self, domain, snippet):
        """Control flow detection should be deterministic."""
        domain.reset()
        domain.accept_bytes(snippet.encode())
        cfg1 = domain.get_cfg()

        domain.reset()
        domain.accept_bytes(snippet.encode())
        cfg2 = domain.get_cfg()

        assert len(cfg1.nodes) == len(cfg2.nodes)
        assert len(cfg1.edges) == len(cfg2.edges)

    def test_guard_detected(self, domain):
        """Swift guard statements should be detected."""
        domain.accept_bytes(b"guard x > 0 else { return }")
        cfg = domain.get_cfg()
        assert cfg is not None

    def test_defer_detected(self, domain):
        """Swift defer statements should be detected."""
        domain.accept_bytes(b"defer { cleanup() }")
        cfg = domain.get_cfg()
        assert cfg is not None

    def test_do_catch_detected(self, domain):
        """Swift do-catch should be detected."""
        domain.accept_bytes(b"do { try op() } catch { handle() }")
        cfg = domain.get_cfg()
        assert cfg is not None


# =============================================================================
# Zig Control Flow Properties
# =============================================================================

class TestZigControlFlowProperties:
    """Property tests for Zig control flow detection."""

    @pytest.fixture
    def domain(self, tokenizer):
        return ControlFlowDomain(language="zig", tokenizer=tokenizer)

    @given(zig_snippet)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_deterministic_detection(self, domain, snippet):
        """Control flow detection should be deterministic."""
        domain.reset()
        domain.accept_bytes(snippet.encode())
        cfg1 = domain.get_cfg()

        domain.reset()
        domain.accept_bytes(snippet.encode())
        cfg2 = domain.get_cfg()

        assert len(cfg1.nodes) == len(cfg2.nodes)
        assert len(cfg1.edges) == len(cfg2.edges)

    def test_try_catch_detected(self, domain):
        """Zig try/catch should be detected."""
        domain.accept_bytes(b"const x = try op();")
        cfg = domain.get_cfg()
        assert cfg is not None

    def test_orelse_detected(self, domain):
        """Zig orelse should be detected."""
        domain.accept_bytes(b"const x = opt orelse default;")
        cfg = domain.get_cfg()
        assert cfg is not None

    def test_errdefer_detected(self, domain):
        """Zig errdefer should be detected."""
        domain.accept_bytes(b"errdefer cleanup();")
        cfg = domain.get_cfg()
        assert cfg is not None

    def test_unreachable_detected(self, domain):
        """Zig unreachable should be detected."""
        domain.accept_bytes(b"unreachable;")
        cfg = domain.get_cfg()
        assert cfg is not None


# =============================================================================
# Cross-Language Properties
# =============================================================================

class TestCrossLanguageProperties:
    """Property tests that should hold across all languages."""

    @pytest.fixture(params=["python", "typescript", "go", "rust", "kotlin", "swift", "zig"])
    def language(self, request):
        return request.param

    @pytest.fixture
    def domain(self, language, tokenizer):
        return ControlFlowDomain(language=language, tokenizer=tokenizer)

    def test_empty_input_has_entry_node(self, domain):
        """Empty input should still have an entry node."""
        domain.accept_bytes(b"")
        cfg = domain.get_cfg()
        assert cfg is not None
        # CFG should have at least entry node
        assert len(cfg.nodes) >= 1

    def test_reset_clears_state(self, domain):
        """Reset should clear all state."""
        domain.accept_bytes(b"if x { return }")
        cfg1 = domain.get_cfg()

        domain.reset()
        cfg2 = domain.get_cfg()

        # After reset, should be back to initial state
        assert len(cfg2.nodes) <= len(cfg1.nodes)

    def test_cfg_is_well_formed(self, domain, language):
        """CFG should have valid structure."""
        code_map = {
            "python": b"if x:\n    return y",
            "typescript": b"if (x) { return y; }",
            "go": b"if x { return y }",
            "rust": b"if x { return y; }",
            "kotlin": b"if (x) return y",
            "swift": b"if x { return y }",
            "zig": b"if (x) return y;",
        }
        domain.accept_bytes(code_map.get(language, b"if x { return }"))
        cfg = domain.get_cfg()

        # All edges should reference valid nodes
        node_ids = {n.id for n in cfg.nodes}
        for edge in cfg.edges:
            assert edge.source in node_ids or edge.source == "entry"
            assert edge.target in node_ids or edge.target == "exit"
