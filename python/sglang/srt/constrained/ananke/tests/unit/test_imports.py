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
"""Tests for the Import Domain.

Tests cover:
- ImportConstraint semilattice laws
- ModuleSpec creation and properties
- ImportDomain functionality
- Python import detection
- TypeScript import detection
- Import resolvers
"""

from __future__ import annotations

import pytest
import torch
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

try:
    from ...domains.imports import (
        ImportConstraint,
        ModuleSpec,
        IMPORT_TOP,
        IMPORT_BOTTOM,
        import_requiring,
        import_forbidding,
        ImportDomain,
        ImportDomainCheckpoint,
    )
    from ...domains.imports.resolvers import (
        ImportResolver,
        ImportResolution,
        ResolvedModule,
        PassthroughResolver,
        DenyListResolver,
        PythonImportResolver,
        PYTHON_STDLIB,
        create_python_resolver,
    )
    from ...core.constraint import Satisfiability
    from ...core.domain import GenerationContext
except ImportError:
    from domains.imports import (
        ImportConstraint,
        ModuleSpec,
        IMPORT_TOP,
        IMPORT_BOTTOM,
        import_requiring,
        import_forbidding,
        ImportDomain,
        ImportDomainCheckpoint,
    )
    from domains.imports.resolvers import (
        ImportResolver,
        ImportResolution,
        ResolvedModule,
        PassthroughResolver,
        DenyListResolver,
        PythonImportResolver,
        PYTHON_STDLIB,
        create_python_resolver,
    )
    from core.constraint import Satisfiability
    from core.domain import GenerationContext


# =============================================================================
# Mock Tokenizer
# =============================================================================


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab: Optional[Dict[int, str]] = None):
        """Initialize with optional vocabulary mapping."""
        self._vocab = vocab or {}

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return "".join(self._vocab.get(tid, f"<{tid}>") for tid in token_ids)


class SequentialMockTokenizer:
    """Mock tokenizer that returns tokens sequentially."""

    def __init__(self, tokens: List[str]):
        """Initialize with list of tokens to return."""
        self._tokens = tokens
        self._idx = 0

    def decode(self, token_ids: List[int]) -> str:
        """Return next token in sequence."""
        if self._idx < len(self._tokens):
            result = self._tokens[self._idx]
            self._idx += 1
            return result
        return ""


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_tokenizer() -> MockTokenizer:
    """Create a simple tokenizer for testing."""
    return MockTokenizer()


@pytest.fixture
def generation_context(simple_tokenizer: MockTokenizer) -> GenerationContext:
    """Create a generation context for testing."""
    return GenerationContext(
        vocab_size=100,
        device=torch.device("cpu"),
        tokenizer=simple_tokenizer,
        generated_text="",
        position=0,
    )


@pytest.fixture
def python_domain() -> ImportDomain:
    """Create a Python import domain."""
    return ImportDomain(language="python")


@pytest.fixture
def typescript_domain() -> ImportDomain:
    """Create a TypeScript import domain."""
    return ImportDomain(language="typescript")


@pytest.fixture
def python_resolver() -> PythonImportResolver:
    """Create a Python import resolver."""
    return create_python_resolver(check_installed=True)


# =============================================================================
# ModuleSpec Tests
# =============================================================================


class TestModuleSpec:
    """Tests for ModuleSpec dataclass."""

    def test_create_simple(self) -> None:
        """Test creating a simple ModuleSpec."""
        spec = ModuleSpec(name="numpy")
        assert spec.name == "numpy"
        assert spec.version is None
        assert spec.alias is None

    def test_create_with_version(self) -> None:
        """Test creating ModuleSpec with version."""
        spec = ModuleSpec(name="numpy", version=">=1.20.0")
        assert spec.name == "numpy"
        assert spec.version == ">=1.20.0"
        assert spec.alias is None

    def test_create_with_alias(self) -> None:
        """Test creating ModuleSpec with alias."""
        spec = ModuleSpec(name="numpy", alias="np")
        assert spec.name == "numpy"
        assert spec.version is None
        assert spec.alias == "np"

    def test_create_full(self) -> None:
        """Test creating ModuleSpec with all fields."""
        spec = ModuleSpec(name="numpy", version=">=1.20.0", alias="np")
        assert spec.name == "numpy"
        assert spec.version == ">=1.20.0"
        assert spec.alias == "np"

    def test_equality(self) -> None:
        """Test ModuleSpec equality."""
        spec1 = ModuleSpec(name="numpy", alias="np")
        spec2 = ModuleSpec(name="numpy", alias="np")
        spec3 = ModuleSpec(name="numpy", alias="npy")

        assert spec1 == spec2
        assert spec1 != spec3

    def test_hashable(self) -> None:
        """Test that ModuleSpec is hashable."""
        spec = ModuleSpec(name="numpy")
        # Should be usable in sets
        s = {spec}
        assert spec in s

    def test_repr_simple(self) -> None:
        """Test string representation."""
        spec = ModuleSpec(name="numpy")
        assert "numpy" in repr(spec)

    def test_repr_with_alias(self) -> None:
        """Test string representation with alias."""
        spec = ModuleSpec(name="numpy", alias="np")
        assert "numpy" in repr(spec)
        assert "np" in repr(spec)

    def test_repr_with_version(self) -> None:
        """Test string representation with version."""
        spec = ModuleSpec(name="numpy", version=">=1.20")
        assert "numpy" in repr(spec)
        assert ">=1.20" in repr(spec)


# =============================================================================
# ImportConstraint Tests
# =============================================================================


class TestImportConstraint:
    """Tests for ImportConstraint."""

    def test_top_is_top(self) -> None:
        """Test TOP constraint is_top."""
        assert IMPORT_TOP.is_top()
        assert not IMPORT_TOP.is_bottom()

    def test_bottom_is_bottom(self) -> None:
        """Test BOTTOM constraint is_bottom."""
        assert IMPORT_BOTTOM.is_bottom()
        assert not IMPORT_BOTTOM.is_top()

    def test_empty_constraint_satisfiable(self) -> None:
        """Test empty constraint is satisfiable."""
        c = ImportConstraint()
        assert c.satisfiability() == Satisfiability.SAT
        assert not c.is_top()
        assert not c.is_bottom()

    def test_required_modules(self) -> None:
        """Test constraint with required modules."""
        c = import_requiring("numpy", "pandas")
        assert c.is_required("numpy")
        assert c.is_required("pandas")
        assert not c.is_required("scipy")

    def test_forbidden_modules(self) -> None:
        """Test constraint with forbidden modules."""
        c = import_forbidding("os", "subprocess")
        assert c.is_forbidden("os")
        assert c.is_forbidden("subprocess")
        assert not c.is_forbidden("math")

    def test_conflict_is_bottom(self) -> None:
        """Test that required + forbidden same module = BOTTOM."""
        c1 = import_requiring("numpy")
        c2 = import_forbidding("numpy")
        result = c1.meet(c2)
        assert result.is_bottom()

    def test_meet_with_top(self) -> None:
        """Test meet with TOP returns other constraint."""
        c = import_requiring("numpy")
        assert c.meet(IMPORT_TOP) == c
        assert IMPORT_TOP.meet(c) == c

    def test_meet_with_bottom(self) -> None:
        """Test meet with BOTTOM returns BOTTOM."""
        c = import_requiring("numpy")
        assert c.meet(IMPORT_BOTTOM).is_bottom()
        assert IMPORT_BOTTOM.meet(c).is_bottom()

    def test_meet_combines_required(self) -> None:
        """Test meet combines required sets."""
        c1 = import_requiring("numpy")
        c2 = import_requiring("pandas")
        result = c1.meet(c2)

        assert result.is_required("numpy")
        assert result.is_required("pandas")

    def test_meet_combines_forbidden(self) -> None:
        """Test meet combines forbidden sets."""
        c1 = import_forbidding("os")
        c2 = import_forbidding("subprocess")
        result = c1.meet(c2)

        assert result.is_forbidden("os")
        assert result.is_forbidden("subprocess")

    def test_requires_method(self) -> None:
        """Test adding required module."""
        c = ImportConstraint()
        spec = ModuleSpec(name="numpy")
        c2 = c.requires(spec)

        assert c2.is_required("numpy")
        assert not c.is_required("numpy")  # Original unchanged

    def test_forbids_method(self) -> None:
        """Test adding forbidden module."""
        c = ImportConstraint()
        c2 = c.forbids("os")

        assert c2.is_forbidden("os")
        assert not c.is_forbidden("os")  # Original unchanged

    def test_with_available(self) -> None:
        """Test marking module as available."""
        c = ImportConstraint()
        spec = ModuleSpec(name="numpy", alias="np")
        c2 = c.with_available(spec)

        assert c2.is_available("numpy")
        assert c2.get_available("numpy") == spec

    def test_missing_requirements(self) -> None:
        """Test finding missing requirements."""
        c = import_requiring("numpy", "pandas")
        spec = ModuleSpec(name="numpy")
        c = c.with_available(spec)

        missing = c.missing_requirements()
        assert len(missing) == 1
        assert any(m.name == "pandas" for m in missing)

    def test_requires_forbidden_returns_bottom(self) -> None:
        """Test requiring a forbidden module returns BOTTOM."""
        c = import_forbidding("os")
        spec = ModuleSpec(name="os")
        result = c.requires(spec)

        assert result.is_bottom()

    def test_forbids_required_returns_bottom(self) -> None:
        """Test forbidding a required module returns BOTTOM."""
        c = import_requiring("numpy")
        result = c.forbids("numpy")

        assert result.is_bottom()


# =============================================================================
# ImportConstraint Semilattice Laws
# =============================================================================


class TestImportConstraintSemilattice:
    """Verify semilattice laws for ImportConstraint."""

    def test_idempotent(self) -> None:
        """Test c.meet(c) == c."""
        c = import_requiring("numpy").meet(import_forbidding("os"))
        assert c.meet(c) == c

    def test_commutative(self) -> None:
        """Test c1.meet(c2) == c2.meet(c1)."""
        c1 = import_requiring("numpy")
        c2 = import_forbidding("os")

        # Note: We compare by checking equivalent properties since
        # the exact frozenset order might differ
        r1 = c1.meet(c2)
        r2 = c2.meet(c1)

        assert r1.required == r2.required
        assert r1.forbidden == r2.forbidden

    def test_associative(self) -> None:
        """Test (c1.meet(c2)).meet(c3) == c1.meet(c2.meet(c3))."""
        c1 = import_requiring("numpy")
        c2 = import_requiring("pandas")
        c3 = import_forbidding("os")

        left = (c1.meet(c2)).meet(c3)
        right = c1.meet(c2.meet(c3))

        assert left.required == right.required
        assert left.forbidden == right.forbidden

    def test_top_identity(self) -> None:
        """Test c.meet(TOP) == c."""
        c = import_requiring("numpy")
        assert c.meet(IMPORT_TOP) == c

    def test_bottom_absorbing(self) -> None:
        """Test c.meet(BOTTOM) == BOTTOM."""
        c = import_requiring("numpy")
        assert c.meet(IMPORT_BOTTOM).is_bottom()


# =============================================================================
# ImportDomain Tests
# =============================================================================


class TestImportDomain:
    """Tests for ImportDomain."""

    def test_domain_name(self, python_domain: ImportDomain) -> None:
        """Test domain name is 'imports'."""
        assert python_domain.name == "imports"

    def test_domain_language(self, python_domain: ImportDomain) -> None:
        """Test domain language."""
        assert python_domain.language == "python"

    def test_top_and_bottom(self, python_domain: ImportDomain) -> None:
        """Test domain provides TOP and BOTTOM."""
        assert python_domain.top.is_top()
        assert python_domain.bottom.is_bottom()

    def test_create_constraint_empty(self, python_domain: ImportDomain) -> None:
        """Test creating empty constraint returns TOP."""
        c = python_domain.create_constraint()
        assert c.is_top()

    def test_create_constraint_required(self, python_domain: ImportDomain) -> None:
        """Test creating constraint with required modules."""
        c = python_domain.create_constraint(required=["numpy", "pandas"])
        assert c.is_required("numpy")
        assert c.is_required("pandas")

    def test_create_constraint_forbidden(self, python_domain: ImportDomain) -> None:
        """Test creating constraint with forbidden modules."""
        c = python_domain.create_constraint(forbidden=["os", "subprocess"])
        assert c.is_forbidden("os")
        assert c.is_forbidden("subprocess")

    def test_create_constraint_conflict(self, python_domain: ImportDomain) -> None:
        """Test creating conflicting constraint returns BOTTOM."""
        c = python_domain.create_constraint(
            required=["numpy"],
            forbidden=["numpy"]
        )
        assert c.is_bottom()

    def test_add_available(self, python_domain: ImportDomain) -> None:
        """Test adding available module."""
        python_domain.add_available("numpy", alias="np")
        assert python_domain.is_imported("numpy")

    def test_is_imported_initially_false(self, python_domain: ImportDomain) -> None:
        """Test modules are not imported initially."""
        assert not python_domain.is_imported("numpy")

    def test_imported_modules_property(self, python_domain: ImportDomain) -> None:
        """Test imported_modules returns copy."""
        python_domain.add_available("numpy")
        modules = python_domain.imported_modules

        assert "numpy" in modules
        # Modifying returned set shouldn't affect domain
        modules.add("fake")
        assert "fake" not in python_domain.imported_modules


class TestImportDomainTokenMask:
    """Tests for ImportDomain.token_mask()."""

    def test_top_allows_all(
        self,
        python_domain: ImportDomain,
        generation_context: GenerationContext,
    ) -> None:
        """Test TOP constraint allows all tokens."""
        mask = python_domain.token_mask(IMPORT_TOP, generation_context)
        assert mask.all()

    def test_bottom_blocks_all(
        self,
        python_domain: ImportDomain,
        generation_context: GenerationContext,
    ) -> None:
        """Test BOTTOM constraint blocks all tokens."""
        mask = python_domain.token_mask(IMPORT_BOTTOM, generation_context)
        assert not mask.any()

    def test_regular_constraint_conservative(
        self,
        python_domain: ImportDomain,
        generation_context: GenerationContext,
    ) -> None:
        """Test regular constraint currently allows all (conservative)."""
        c = import_forbidding("os")
        mask = python_domain.token_mask(c, generation_context)
        # Current implementation is conservative
        assert mask.all()


class TestImportDomainObserveToken:
    """Tests for ImportDomain.observe_token()."""

    def test_observe_token_top_unchanged(
        self,
        python_domain: ImportDomain,
        generation_context: GenerationContext,
    ) -> None:
        """Test observing token with TOP constraint."""
        result = python_domain.observe_token(IMPORT_TOP, 0, generation_context)
        assert result.is_top()

    def test_observe_token_bottom_unchanged(
        self,
        python_domain: ImportDomain,
        generation_context: GenerationContext,
    ) -> None:
        """Test observing token with BOTTOM constraint."""
        result = python_domain.observe_token(IMPORT_BOTTOM, 0, generation_context)
        assert result.is_bottom()


class TestImportDomainPythonDetection:
    """Tests for Python import statement detection."""

    def test_detect_simple_import(self, python_domain: ImportDomain) -> None:
        """Test detecting simple 'import numpy' statement."""
        # Simulate tokens for "import numpy\n"
        tokens = ["import", " ", "numpy", "\n"]

        c = python_domain.create_constraint(required=["numpy"])

        tokenizer = SequentialMockTokenizer(tokens)
        ctx = GenerationContext(
            vocab_size=100,
            device=torch.device("cpu"),
            tokenizer=tokenizer,
            generated_text="",
            position=0,
        )

        # Observe each "token"
        for i, _ in enumerate(tokens):
            c = python_domain.observe_token(c, i, ctx)

        # After newline, import should be detected
        assert python_domain.is_imported("numpy")
        assert c.is_available("numpy")

    def test_detect_import_as(self, python_domain: ImportDomain) -> None:
        """Test detecting 'import numpy as np' statement."""
        tokens = ["import", " ", "numpy", " ", "as", " ", "np", "\n"]

        c = python_domain.create_constraint(required=["numpy"])

        tokenizer = SequentialMockTokenizer(tokens)
        ctx = GenerationContext(
            vocab_size=100,
            device=torch.device("cpu"),
            tokenizer=tokenizer,
            generated_text="",
            position=0,
        )

        for i, _ in enumerate(tokens):
            c = python_domain.observe_token(c, i, ctx)

        assert python_domain.is_imported("numpy")

    def test_detect_from_import(self, python_domain: ImportDomain) -> None:
        """Test detecting 'from os import path' statement."""
        tokens = ["from", " ", "os", " ", "import", " ", "path", "\n"]

        c = ImportConstraint()

        tokenizer = SequentialMockTokenizer(tokens)
        ctx = GenerationContext(
            vocab_size=100,
            device=torch.device("cpu"),
            tokenizer=tokenizer,
            generated_text="",
            position=0,
        )

        for i, _ in enumerate(tokens):
            c = python_domain.observe_token(c, i, ctx)

        assert python_domain.is_imported("os")


class TestImportDomainTypeScriptDetection:
    """Tests for TypeScript import statement detection."""

    def test_detect_ts_import(self, typescript_domain: ImportDomain) -> None:
        """Test detecting TypeScript import statement."""
        tokens = ["import", " ", "React", " ", "from", " ", "'", "react", "'", "\n"]

        c = ImportConstraint()

        tokenizer = SequentialMockTokenizer(tokens)
        ctx = GenerationContext(
            vocab_size=100,
            device=torch.device("cpu"),
            tokenizer=tokenizer,
            generated_text="",
            position=0,
        )

        for i, _ in enumerate(tokens):
            c = typescript_domain.observe_token(c, i, ctx)

        assert typescript_domain.is_imported("react")


class TestImportDomainCheckpoint:
    """Tests for ImportDomain checkpoint/restore."""

    def test_checkpoint_preserves_state(self, python_domain: ImportDomain) -> None:
        """Test checkpoint preserves domain state."""
        python_domain.add_available("numpy")
        python_domain.add_available("pandas")

        checkpoint = python_domain.checkpoint()

        assert "numpy" in checkpoint.imported_modules
        assert "pandas" in checkpoint.imported_modules

    def test_restore_reverts_state(self, python_domain: ImportDomain) -> None:
        """Test restore reverts domain state."""
        python_domain.add_available("numpy")
        checkpoint = python_domain.checkpoint()

        python_domain.add_available("pandas")
        assert python_domain.is_imported("pandas")

        python_domain.restore(checkpoint)

        assert python_domain.is_imported("numpy")
        assert not python_domain.is_imported("pandas")

    def test_restore_wrong_type_raises(self, python_domain: ImportDomain) -> None:
        """Test restore with wrong type raises TypeError."""
        with pytest.raises(TypeError):
            python_domain.restore("not a checkpoint")  # type: ignore

    def test_satisfiability_check(self, python_domain: ImportDomain) -> None:
        """Test satisfiability returns constraint's satisfiability."""
        c = import_requiring("numpy")
        assert python_domain.satisfiability(c) == Satisfiability.SAT

        assert python_domain.satisfiability(IMPORT_BOTTOM) == Satisfiability.UNSAT


# =============================================================================
# ImportResolver Tests
# =============================================================================


class TestResolvedModule:
    """Tests for ResolvedModule dataclass."""

    def test_create_simple(self) -> None:
        """Test creating simple ResolvedModule."""
        m = ResolvedModule(name="numpy")
        assert m.name == "numpy"
        assert m.version is None
        assert m.path is None
        assert m.is_available

    def test_create_full(self) -> None:
        """Test creating ResolvedModule with all fields."""
        m = ResolvedModule(
            name="numpy",
            version="1.24.0",
            path="/path/to/numpy",
            is_builtin=False,
            is_available=True,
        )
        assert m.name == "numpy"
        assert m.version == "1.24.0"
        assert m.path == "/path/to/numpy"

    def test_exports_default_empty(self) -> None:
        """Test exports defaults to empty set."""
        m = ResolvedModule(name="numpy")
        assert m.exports == set()


class TestImportResolution:
    """Tests for ImportResolution dataclass."""

    def test_success_resolution(self) -> None:
        """Test successful resolution."""
        module = ResolvedModule(name="numpy", version="1.24.0")
        resolution = ImportResolution(success=True, module=module)

        assert resolution.success
        assert resolution.module is not None
        assert resolution.module.name == "numpy"

    def test_failed_resolution(self) -> None:
        """Test failed resolution."""
        resolution = ImportResolution(
            success=False,
            error="Module not found",
            alternatives=["numpy-stubs"],
        )

        assert not resolution.success
        assert resolution.error == "Module not found"
        assert "numpy-stubs" in resolution.alternatives


class TestPassthroughResolver:
    """Tests for PassthroughResolver."""

    def test_always_available(self) -> None:
        """Test passthrough always returns available."""
        resolver = PassthroughResolver()

        assert resolver.is_available("anything")
        assert resolver.is_available("made_up_module")

    def test_resolve_always_succeeds(self) -> None:
        """Test resolve always succeeds."""
        resolver = PassthroughResolver()

        result = resolver.resolve("anything")
        assert result.success
        assert result.module is not None

    def test_language_property(self) -> None:
        """Test language property."""
        resolver = PassthroughResolver(language="rust")
        assert resolver.language == "rust"

    def test_version_returns_none(self) -> None:
        """Test get_version returns None."""
        resolver = PassthroughResolver()
        assert resolver.get_version("anything") is None


class TestDenyListResolver:
    """Tests for DenyListResolver."""

    def test_denied_module_unavailable(self) -> None:
        """Test denied modules are unavailable."""
        resolver = DenyListResolver(denied={"os", "subprocess"})

        assert not resolver.is_available("os")
        assert not resolver.is_available("subprocess")

    def test_allowed_module_available(self) -> None:
        """Test non-denied modules use fallback."""
        resolver = DenyListResolver(denied={"os"})

        assert resolver.is_available("numpy")

    def test_resolve_denied_fails(self) -> None:
        """Test resolving denied module fails."""
        resolver = DenyListResolver(denied={"os"})

        result = resolver.resolve("os")
        assert not result.success
        assert "not allowed" in result.error

    def test_resolve_allowed_succeeds(self) -> None:
        """Test resolving allowed module succeeds."""
        resolver = DenyListResolver(denied={"os"})

        result = resolver.resolve("numpy")
        assert result.success

    def test_custom_fallback(self) -> None:
        """Test custom fallback resolver."""
        class CustomResolver(ImportResolver):
            @property
            def language(self) -> str:
                return "test"

            def resolve(self, module_name: str) -> ImportResolution:
                return ImportResolution(
                    success=True,
                    module=ResolvedModule(name=module_name, version="custom"),
                )

            def is_available(self, module_name: str) -> bool:
                return True

            def get_version(self, module_name: str) -> Optional[str]:
                return "custom"

        custom = CustomResolver()
        resolver = DenyListResolver(denied={"os"}, fallback=custom)

        result = resolver.resolve("numpy")
        assert result.success
        assert result.module.version == "custom"


# =============================================================================
# PythonImportResolver Tests
# =============================================================================


class TestPythonImportResolver:
    """Tests for PythonImportResolver."""

    def test_language_is_python(self, python_resolver: PythonImportResolver) -> None:
        """Test language is 'python'."""
        assert python_resolver.language == "python"

    def test_stdlib_modules_available(self, python_resolver: PythonImportResolver) -> None:
        """Test stdlib modules are available."""
        assert python_resolver.is_available("os")
        assert python_resolver.is_available("sys")
        assert python_resolver.is_available("json")
        assert python_resolver.is_available("typing")

    def test_builtin_modules_available(self, python_resolver: PythonImportResolver) -> None:
        """Test builtin modules are available."""
        assert python_resolver.is_available("builtins")

    def test_is_stdlib(self, python_resolver: PythonImportResolver) -> None:
        """Test is_stdlib method."""
        assert python_resolver.is_stdlib("os")
        assert python_resolver.is_stdlib("sys")
        assert python_resolver.is_stdlib("json")
        assert not python_resolver.is_stdlib("nonexistent_module_xyz")

    def test_submodule_resolution(self, python_resolver: PythonImportResolver) -> None:
        """Test submodule resolution."""
        assert python_resolver.is_available("os.path")
        assert python_resolver.is_available("collections.abc")

    def test_resolve_stdlib_success(self, python_resolver: PythonImportResolver) -> None:
        """Test resolving stdlib module succeeds."""
        result = python_resolver.resolve("os")

        assert result.success
        assert result.module is not None
        assert result.module.name == "os"
        assert result.module.is_available

    def test_resolve_nonexistent_fails(self, python_resolver: PythonImportResolver) -> None:
        """Test resolving nonexistent module fails."""
        result = python_resolver.resolve("nonexistent_module_xyz_123")

        assert not result.success
        assert result.error is not None
        assert "not found" in result.error

    def test_suggest_alternatives(self, python_resolver: PythonImportResolver) -> None:
        """Test suggest_alternatives method."""
        # Common aliases should suggest alternatives
        suggestions = python_resolver.suggest_alternatives("np")
        assert "numpy" in suggestions

        suggestions = python_resolver.suggest_alternatives("pd")
        assert "pandas" in suggestions

    def test_caching(self, python_resolver: PythonImportResolver) -> None:
        """Test results are cached."""
        # First call
        result1 = python_resolver.resolve("os")
        # Second call should return same object (cached)
        result2 = python_resolver.resolve("os")

        assert result1 is result2

    def test_factory_function(self) -> None:
        """Test create_python_resolver factory."""
        resolver = create_python_resolver(check_installed=False)
        assert resolver.language == "python"


class TestPythonStdlib:
    """Tests for PYTHON_STDLIB constant."""

    def test_common_modules_present(self) -> None:
        """Test common stdlib modules are in the set."""
        assert "os" in PYTHON_STDLIB
        assert "sys" in PYTHON_STDLIB
        assert "json" in PYTHON_STDLIB
        assert "typing" in PYTHON_STDLIB
        assert "collections" in PYTHON_STDLIB
        assert "functools" in PYTHON_STDLIB
        assert "itertools" in PYTHON_STDLIB

    def test_common_submodules_present(self) -> None:
        """Test common submodules are in the set."""
        assert "collections.abc" in PYTHON_STDLIB
        assert "os.path" in PYTHON_STDLIB
        assert "typing_extensions" in PYTHON_STDLIB


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestImportConstraintProperties:
    """Property-based tests for ImportConstraint."""

    def test_meet_preserves_required(self) -> None:
        """Test meet preserves all required modules."""
        c1 = import_requiring("numpy", "pandas")
        c2 = import_requiring("scipy")

        result = c1.meet(c2)

        assert result.is_required("numpy")
        assert result.is_required("pandas")
        assert result.is_required("scipy")

    def test_meet_preserves_forbidden(self) -> None:
        """Test meet preserves all forbidden modules."""
        c1 = import_forbidding("os")
        c2 = import_forbidding("subprocess")

        result = c1.meet(c2)

        assert result.is_forbidden("os")
        assert result.is_forbidden("subprocess")

    def test_no_false_conflicts(self) -> None:
        """Test no false conflicts between unrelated constraints."""
        c1 = import_requiring("numpy").meet(import_forbidding("os"))
        c2 = import_requiring("pandas").meet(import_forbidding("subprocess"))

        result = c1.meet(c2)

        # Should not be bottom
        assert not result.is_bottom()
        # Should have all requirements
        assert result.is_required("numpy")
        assert result.is_required("pandas")
        # Should have all forbiddens
        assert result.is_forbidden("os")
        assert result.is_forbidden("subprocess")


# =============================================================================
# Integration Tests
# =============================================================================


class TestImportDomainIntegration:
    """Integration tests for Import Domain."""

    def test_full_import_flow(self, python_domain: ImportDomain) -> None:
        """Test full flow of import constraint + detection."""
        # Create constraint requiring numpy
        c = python_domain.create_constraint(required=["numpy"])

        # Mark as available (simulating import detection)
        spec = ModuleSpec(name="numpy", alias="np")
        c = c.with_available(spec)

        # Check requirements met
        assert c.missing_requirements() == frozenset()

    def test_checkpoint_restore_cycle(self, python_domain: ImportDomain) -> None:
        """Test multiple checkpoint/restore cycles."""
        python_domain.add_available("numpy")
        cp1 = python_domain.checkpoint()

        python_domain.add_available("pandas")
        cp2 = python_domain.checkpoint()

        python_domain.add_available("scipy")

        # Restore to cp2
        python_domain.restore(cp2)
        assert python_domain.is_imported("numpy")
        assert python_domain.is_imported("pandas")
        assert not python_domain.is_imported("scipy")

        # Restore to cp1
        python_domain.restore(cp1)
        assert python_domain.is_imported("numpy")
        assert not python_domain.is_imported("pandas")
        assert not python_domain.is_imported("scipy")

    def test_resolver_with_domain(
        self,
        python_domain: ImportDomain,
        python_resolver: PythonImportResolver,
    ) -> None:
        """Test using resolver to validate domain constraints."""
        c = python_domain.create_constraint(required=["os", "json"])

        # Validate all required modules exist
        for spec in c.required:
            resolution = python_resolver.resolve(spec.name)
            assert resolution.success, f"Required module {spec.name} not found"
