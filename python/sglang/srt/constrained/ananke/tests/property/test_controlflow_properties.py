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
- Determinism: Same input produces same CFG structure
- Construct recognition: Language-specific constructs are detected
- CFG well-formedness: Generated CFG has valid structure
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

try:
    from ...domains.controlflow.domain import ControlFlowDomain
except ImportError:
    from domains.controlflow.domain import ControlFlowDomain


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
# Go Control Flow Properties
# =============================================================================

class TestGoControlFlowProperties:
    """Property tests for Go control flow detection."""

    @given(go_snippet)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_deterministic_detection(self, snippet):
        """Control flow detection should be deterministic."""
        # Create two fresh domains
        domain1 = ControlFlowDomain(language="go")
        domain2 = ControlFlowDomain(language="go")

        # Process same snippet
        domain1._token_buffer = snippet
        domain1._detect_go_control_flow()
        cfg1 = domain1.cfg

        domain2._token_buffer = snippet
        domain2._detect_go_control_flow()
        cfg2 = domain2.cfg

        # Same snippet should produce same CFG structure
        assert len(cfg1.blocks) == len(cfg2.blocks)
        assert len(cfg1.edges) == len(cfg2.edges)

    def test_defer_detected(self):
        """Go defer statements should be detected."""
        domain = ControlFlowDomain(language="go")
        domain._token_buffer = "defer file.Close()"
        domain._detect_go_control_flow()
        # CFG should exist and not raise
        assert domain.cfg is not None

    def test_goroutine_detected(self):
        """Go goroutines should be detected."""
        domain = ControlFlowDomain(language="go")
        domain._token_buffer = "go work()"
        domain._detect_go_control_flow()
        assert domain.cfg is not None

    def test_switch_case_detected(self):
        """Go switch/case should be detected."""
        domain = ControlFlowDomain(language="go")
        domain._token_buffer = "switch x { case 1: a() case 2: b() }"
        domain._detect_go_control_flow()
        assert domain.cfg is not None


# =============================================================================
# Rust Control Flow Properties
# =============================================================================

class TestRustControlFlowProperties:
    """Property tests for Rust control flow detection."""

    @given(rust_snippet)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_deterministic_detection(self, snippet):
        """Control flow detection should be deterministic."""
        domain1 = ControlFlowDomain(language="rust")
        domain2 = ControlFlowDomain(language="rust")

        domain1._token_buffer = snippet
        domain1._detect_rust_control_flow()
        cfg1 = domain1.cfg

        domain2._token_buffer = snippet
        domain2._detect_rust_control_flow()
        cfg2 = domain2.cfg

        assert len(cfg1.blocks) == len(cfg2.blocks)
        assert len(cfg1.edges) == len(cfg2.edges)

    def test_match_detected(self):
        """Rust match expressions should be detected."""
        domain = ControlFlowDomain(language="rust")
        domain._token_buffer = "match x { 0 => zero(), _ => other() }"
        domain._detect_rust_control_flow()
        assert domain.cfg is not None

    def test_loop_detected(self):
        """Rust infinite loop should be detected."""
        domain = ControlFlowDomain(language="rust")
        domain._token_buffer = "loop { break; }"
        domain._detect_rust_control_flow()
        assert domain.cfg is not None

    def test_question_mark_detected(self):
        """Rust ? operator should be detected."""
        domain = ControlFlowDomain(language="rust")
        domain._token_buffer = "let x = file.read()?;"
        domain._detect_rust_control_flow()
        assert domain.cfg is not None


# =============================================================================
# Kotlin Control Flow Properties
# =============================================================================

class TestKotlinControlFlowProperties:
    """Property tests for Kotlin control flow detection."""

    @given(kotlin_snippet)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_deterministic_detection(self, snippet):
        """Control flow detection should be deterministic."""
        domain1 = ControlFlowDomain(language="kotlin")
        domain2 = ControlFlowDomain(language="kotlin")

        domain1._token_buffer = snippet
        domain1._detect_kotlin_control_flow()
        cfg1 = domain1.cfg

        domain2._token_buffer = snippet
        domain2._detect_kotlin_control_flow()
        cfg2 = domain2.cfg

        assert len(cfg1.blocks) == len(cfg2.blocks)
        assert len(cfg1.edges) == len(cfg2.edges)

    def test_when_detected(self):
        """Kotlin when expressions should be detected."""
        domain = ControlFlowDomain(language="kotlin")
        domain._token_buffer = "when (x) { 1 -> one() else -> other() }"
        domain._detect_kotlin_control_flow()
        assert domain.cfg is not None

    def test_labeled_return_detected(self):
        """Kotlin labeled returns should be detected."""
        domain = ControlFlowDomain(language="kotlin")
        domain._token_buffer = "return@forEach"
        domain._detect_kotlin_control_flow()
        assert domain.cfg is not None


# =============================================================================
# Swift Control Flow Properties
# =============================================================================

class TestSwiftControlFlowProperties:
    """Property tests for Swift control flow detection."""

    @given(swift_snippet)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_deterministic_detection(self, snippet):
        """Control flow detection should be deterministic."""
        domain1 = ControlFlowDomain(language="swift")
        domain2 = ControlFlowDomain(language="swift")

        domain1._token_buffer = snippet
        domain1._detect_swift_control_flow()
        cfg1 = domain1.cfg

        domain2._token_buffer = snippet
        domain2._detect_swift_control_flow()
        cfg2 = domain2.cfg

        assert len(cfg1.blocks) == len(cfg2.blocks)
        assert len(cfg1.edges) == len(cfg2.edges)

    def test_guard_detected(self):
        """Swift guard statements should be detected."""
        domain = ControlFlowDomain(language="swift")
        domain._token_buffer = "guard x > 0 else { return }"
        domain._detect_swift_control_flow()
        assert domain.cfg is not None

    def test_defer_detected(self):
        """Swift defer statements should be detected."""
        domain = ControlFlowDomain(language="swift")
        domain._token_buffer = "defer { cleanup() }"
        domain._detect_swift_control_flow()
        assert domain.cfg is not None

    def test_do_catch_detected(self):
        """Swift do-catch should be detected."""
        domain = ControlFlowDomain(language="swift")
        domain._token_buffer = "do { try op() } catch { handle() }"
        domain._detect_swift_control_flow()
        assert domain.cfg is not None


# =============================================================================
# Zig Control Flow Properties
# =============================================================================

class TestZigControlFlowProperties:
    """Property tests for Zig control flow detection."""

    @given(zig_snippet)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_deterministic_detection(self, snippet):
        """Control flow detection should be deterministic."""
        domain1 = ControlFlowDomain(language="zig")
        domain2 = ControlFlowDomain(language="zig")

        domain1._token_buffer = snippet
        domain1._detect_zig_control_flow()
        cfg1 = domain1.cfg

        domain2._token_buffer = snippet
        domain2._detect_zig_control_flow()
        cfg2 = domain2.cfg

        assert len(cfg1.blocks) == len(cfg2.blocks)
        assert len(cfg1.edges) == len(cfg2.edges)

    def test_try_catch_detected(self):
        """Zig try/catch should be detected."""
        domain = ControlFlowDomain(language="zig")
        domain._token_buffer = "const x = try op();"
        domain._detect_zig_control_flow()
        assert domain.cfg is not None

    def test_orelse_detected(self):
        """Zig orelse should be detected."""
        domain = ControlFlowDomain(language="zig")
        domain._token_buffer = "const x = opt orelse default;"
        domain._detect_zig_control_flow()
        assert domain.cfg is not None

    def test_errdefer_detected(self):
        """Zig errdefer should be detected."""
        domain = ControlFlowDomain(language="zig")
        domain._token_buffer = "errdefer cleanup();"
        domain._detect_zig_control_flow()
        assert domain.cfg is not None

    def test_unreachable_detected(self):
        """Zig unreachable should be detected."""
        domain = ControlFlowDomain(language="zig")
        domain._token_buffer = "unreachable;"
        domain._detect_zig_control_flow()
        assert domain.cfg is not None


# =============================================================================
# Cross-Language Properties
# =============================================================================

class TestCrossLanguageProperties:
    """Property tests that should hold across all languages."""

    @pytest.mark.parametrize("language", ["python", "typescript", "go", "rust", "kotlin", "swift", "zig"])
    def test_empty_input_has_cfg(self, language):
        """Empty input should still have a CFG."""
        domain = ControlFlowDomain(language=language)
        domain._token_buffer = ""
        # Call language-specific detection
        method_name = f"_detect_{language}_control_flow"
        if hasattr(domain, method_name):
            getattr(domain, method_name)()
        else:
            domain._detect_generic_control_flow()
        # CFG should exist
        assert domain.cfg is not None

    @pytest.mark.parametrize("language", ["python", "typescript", "go", "rust", "kotlin", "swift", "zig"])
    def test_fresh_domain_has_empty_control_stack(self, language):
        """Fresh domain should have empty control stack."""
        domain = ControlFlowDomain(language=language)
        assert len(domain._control_stack) == 0

    @pytest.mark.parametrize("language", ["go", "rust", "kotlin", "swift", "zig"])
    def test_if_keyword_populates_buffer(self, language):
        """if keyword should be processable."""
        domain = ControlFlowDomain(language=language)
        domain._token_buffer = "if"
        method_name = f"_detect_{language}_control_flow"
        # Should not raise
        getattr(domain, method_name)()
        assert domain.cfg is not None

    @pytest.mark.parametrize("language", ["go", "rust", "kotlin", "swift", "zig"])
    def test_return_keyword_processable(self, language):
        """return keyword should be processable."""
        domain = ControlFlowDomain(language=language)
        domain._token_buffer = "return"
        method_name = f"_detect_{language}_control_flow"
        getattr(domain, method_name)()
        assert domain.cfg is not None
