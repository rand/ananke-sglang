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
"""Property-based tests for ConstraintSpec using Hypothesis.

These tests verify:
- Roundtrip serialization (to_dict/from_dict, to_json/from_json)
- Cache key computation is deterministic
- ConstraintSpec invariants are maintained
- Parser handles edge cases correctly
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spec.constraint_spec import (
    CacheScope,
    ClassDefinition,
    ConstraintSource,
    ConstraintSpec,
    ControlFlowContext,
    FunctionSignature,
    ImportBinding,
    LanguageDetection,
    LanguageFrame,
    ModuleStub,
    SemanticConstraint,
    TypeBinding,
)
from spec.parser import ConstraintSpecParser, apply_convenience_overrides
from spec.language_detector import (
    DetectionResult,
    LanguageDetector,
    LanguageStackManager,
)


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Strategy for valid Python identifiers
identifier_strategy = st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,20}", fullmatch=True)

# Strategy for type expressions
simple_type_strategy = st.sampled_from(
    ["int", "str", "bool", "float", "None", "Any", "object"]
)

compound_type_strategy = st.one_of(
    st.builds(lambda t: f"List[{t}]", simple_type_strategy),
    st.builds(lambda t: f"Optional[{t}]", simple_type_strategy),
    st.builds(lambda k, v: f"Dict[{k}, {v}]", simple_type_strategy, simple_type_strategy),
)

type_expr_strategy = st.one_of(simple_type_strategy, compound_type_strategy)

# Strategy for languages
language_strategy = st.sampled_from(
    ["python", "typescript", "go", "rust", "kotlin", "swift", "zig"]
)

# Strategy for TypeBinding
type_binding_strategy = st.builds(
    TypeBinding,
    name=identifier_strategy,
    type_expr=type_expr_strategy,
    scope=st.one_of(st.none(), st.sampled_from(["local", "parameter", "global"])),
    mutable=st.booleans(),
    origin=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
)

# Strategy for FunctionSignature
function_signature_strategy = st.builds(
    FunctionSignature,
    name=identifier_strategy,
    params=st.lists(type_binding_strategy, max_size=5).map(tuple),
    return_type=type_expr_strategy,
    type_params=st.lists(identifier_strategy, max_size=3).map(tuple),
    decorators=st.lists(identifier_strategy, max_size=2).map(tuple),
    is_async=st.booleans(),
    is_generator=st.booleans(),
)

# Strategy for ClassDefinition
class_definition_strategy = st.builds(
    ClassDefinition,
    name=identifier_strategy,
    bases=st.lists(identifier_strategy, max_size=3).map(tuple),
    type_params=st.lists(identifier_strategy, max_size=2).map(tuple),
    methods=st.lists(function_signature_strategy, max_size=3).map(tuple),
    class_vars=st.lists(type_binding_strategy, max_size=2).map(tuple),
    instance_vars=st.lists(type_binding_strategy, max_size=2).map(tuple),
)

# Strategy for ImportBinding
import_binding_strategy = st.builds(
    ImportBinding,
    module=identifier_strategy,
    name=st.one_of(st.none(), identifier_strategy),
    alias=st.one_of(st.none(), identifier_strategy),
    is_wildcard=st.booleans(),
)

# Strategy for ModuleStub
module_stub_strategy = st.builds(
    ModuleStub,
    module_name=identifier_strategy,
    exports=st.dictionaries(identifier_strategy, type_expr_strategy, max_size=5),
    submodules=st.lists(identifier_strategy, max_size=3).map(tuple),
)

# Strategy for LanguageFrame
language_frame_strategy = st.builds(
    LanguageFrame,
    language=language_strategy,
    start_position=st.integers(min_value=0, max_value=10000),
    delimiter=st.one_of(st.none(), st.sampled_from(["'''", '"""', "`", "```"])),
    end_delimiter=st.one_of(st.none(), st.sampled_from(["'''", '"""', "`", "```"])),
)

# Strategy for ControlFlowContext
control_flow_context_strategy = st.builds(
    ControlFlowContext,
    function_name=st.one_of(st.none(), identifier_strategy),
    function_signature=st.one_of(st.none(), function_signature_strategy),
    expected_return_type=st.one_of(st.none(), type_expr_strategy),
    loop_depth=st.integers(min_value=0, max_value=5),
    loop_variables=st.lists(identifier_strategy, max_size=3).map(tuple),
    in_try_block=st.booleans(),
    exception_types=st.lists(identifier_strategy, max_size=2).map(tuple),
    in_async_context=st.booleans(),
    in_generator=st.booleans(),
    reachable=st.booleans(),
    dominators=st.lists(identifier_strategy, max_size=3).map(tuple),
)

# Strategy for SemanticConstraint
semantic_constraint_strategy = st.builds(
    SemanticConstraint,
    kind=st.sampled_from(["precondition", "postcondition", "invariant", "assertion", "assume"]),
    expression=st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]* [><=!]+ \d+", fullmatch=True),
    scope=st.one_of(st.none(), identifier_strategy),
    variables=st.lists(identifier_strategy, max_size=3).map(tuple),
)

# Strategy for CacheScope
cache_scope_strategy = st.sampled_from(list(CacheScope))

# Strategy for ConstraintSource
constraint_source_strategy = st.sampled_from(list(ConstraintSource))

# Strategy for LanguageDetection
language_detection_strategy = st.sampled_from(list(LanguageDetection))


# Strategy for valid JSON schema strings
json_schema_strategy = st.sampled_from([
    '{"type": "object"}',
    '{"type": "string"}',
    '{"type": "integer"}',
    '{"type": "array", "items": {"type": "string"}}',
    '{"type": "object", "properties": {"name": {"type": "string"}}}',
])

# Strategy for valid regex patterns
regex_strategy = st.sampled_from([
    r"[a-z]+",
    r"\d{3}-\d{4}",
    r"[A-Za-z_][A-Za-z0-9_]*",
    r"(foo|bar|baz)",
])


# Strategy for ConstraintSpec
@st.composite
def constraint_spec_strategy(draw):
    """Generate valid ConstraintSpec instances."""
    # Choose one core constraint type
    constraint_type = draw(st.sampled_from(["json_schema", "regex", "ebnf", "structural_tag", "none"]))

    json_schema = draw(json_schema_strategy) if constraint_type == "json_schema" else None
    regex = draw(regex_strategy) if constraint_type == "regex" else None
    ebnf = None  # EBNF is complex, skip for now
    structural_tag = None  # Skip for simplicity

    return ConstraintSpec(
        version=draw(st.sampled_from(["1.0", "1.1"])),
        json_schema=json_schema,
        regex=regex,
        ebnf=ebnf,
        structural_tag=structural_tag,
        language=draw(st.one_of(st.none(), language_strategy)),
        language_detection=draw(language_detection_strategy),
        language_stack=draw(st.lists(language_frame_strategy, max_size=3)),
        type_bindings=draw(st.lists(type_binding_strategy, max_size=5)),
        function_signatures=draw(st.lists(function_signature_strategy, max_size=3)),
        class_definitions=draw(st.lists(class_definition_strategy, max_size=2)),
        expected_type=draw(st.one_of(st.none(), type_expr_strategy)),
        type_aliases=draw(st.dictionaries(identifier_strategy, type_expr_strategy, max_size=3)),
        imports=draw(st.lists(import_binding_strategy, max_size=5)),
        available_modules=draw(st.frozensets(identifier_strategy, max_size=5)),
        forbidden_imports=draw(st.frozensets(identifier_strategy, max_size=3)),
        module_stubs=draw(st.dictionaries(identifier_strategy, module_stub_strategy, max_size=2)),
        control_flow=draw(st.one_of(st.none(), control_flow_context_strategy)),
        semantic_constraints=draw(st.lists(semantic_constraint_strategy, max_size=3)),
        enabled_domains=draw(st.one_of(st.none(), st.frozensets(st.sampled_from(["types", "imports", "controlflow", "semantics"]), max_size=4))),
        disabled_domains=draw(st.one_of(st.none(), st.frozensets(st.sampled_from(["types", "imports", "controlflow", "semantics"]), max_size=2))),
        domain_configs=draw(st.dictionaries(identifier_strategy, st.dictionaries(st.text(max_size=10), st.integers()), max_size=2)),
        cache_scope=draw(cache_scope_strategy),
        context_hash=draw(st.one_of(st.none(), st.text(min_size=8, max_size=32))),
        source=draw(constraint_source_strategy),
        source_uri=draw(st.one_of(st.none(), st.from_regex(r"https?://[a-z]+\.[a-z]+", fullmatch=True))),
    )


# =============================================================================
# Property Tests: TypeBinding
# =============================================================================

class TestTypeBindingProperties:
    """Property tests for TypeBinding dataclass."""

    @given(type_binding_strategy)
    def test_roundtrip_serialization(self, binding: TypeBinding) -> None:
        """TypeBinding should survive roundtrip through dict serialization."""
        d = binding.to_dict()
        reconstructed = TypeBinding.from_dict(d)
        assert reconstructed == binding

    @given(type_binding_strategy)
    def test_dict_has_required_fields(self, binding: TypeBinding) -> None:
        """TypeBinding dict should always have name and type_expr."""
        d = binding.to_dict()
        assert "name" in d
        assert "type_expr" in d

    @given(type_binding_strategy)
    def test_frozen(self, binding: TypeBinding) -> None:
        """TypeBinding should be immutable (frozen)."""
        with pytest.raises(AttributeError):
            binding.name = "new_name"


# =============================================================================
# Property Tests: FunctionSignature
# =============================================================================

class TestFunctionSignatureProperties:
    """Property tests for FunctionSignature dataclass."""

    @given(function_signature_strategy)
    def test_roundtrip_serialization(self, sig: FunctionSignature) -> None:
        """FunctionSignature should survive roundtrip through dict serialization."""
        d = sig.to_dict()
        reconstructed = FunctionSignature.from_dict(d)
        assert reconstructed == sig

    @given(function_signature_strategy)
    def test_params_are_tuple(self, sig: FunctionSignature) -> None:
        """params should be a tuple for immutability."""
        assert isinstance(sig.params, tuple)


# =============================================================================
# Property Tests: ImportBinding
# =============================================================================

class TestImportBindingProperties:
    """Property tests for ImportBinding dataclass."""

    @given(import_binding_strategy)
    def test_roundtrip_serialization(self, binding: ImportBinding) -> None:
        """ImportBinding should survive roundtrip through dict serialization."""
        d = binding.to_dict()
        reconstructed = ImportBinding.from_dict(d)
        assert reconstructed == binding


# =============================================================================
# Property Tests: LanguageFrame
# =============================================================================

class TestLanguageFrameProperties:
    """Property tests for LanguageFrame dataclass."""

    @given(language_frame_strategy)
    def test_roundtrip_serialization(self, frame: LanguageFrame) -> None:
        """LanguageFrame should survive roundtrip through dict serialization."""
        d = frame.to_dict()
        reconstructed = LanguageFrame.from_dict(d)
        assert reconstructed == frame

    @given(language_frame_strategy, st.integers(min_value=0, max_value=20000))
    def test_contains_consistency(self, frame: LanguageFrame, position: int) -> None:
        """contains() should be consistent with start_position."""
        if position >= frame.start_position:
            assert frame.contains(position)
        else:
            assert not frame.contains(position)


# =============================================================================
# Property Tests: ControlFlowContext
# =============================================================================

class TestControlFlowContextProperties:
    """Property tests for ControlFlowContext dataclass."""

    @given(control_flow_context_strategy)
    def test_roundtrip_serialization(self, ctx: ControlFlowContext) -> None:
        """ControlFlowContext should survive roundtrip through dict serialization."""
        d = ctx.to_dict()
        reconstructed = ControlFlowContext.from_dict(d)
        assert reconstructed == ctx


# =============================================================================
# Property Tests: SemanticConstraint
# =============================================================================

class TestSemanticConstraintProperties:
    """Property tests for SemanticConstraint dataclass."""

    @given(semantic_constraint_strategy)
    def test_roundtrip_serialization(self, constraint: SemanticConstraint) -> None:
        """SemanticConstraint should survive roundtrip through dict serialization."""
        d = constraint.to_dict()
        reconstructed = SemanticConstraint.from_dict(d)
        assert reconstructed == constraint

    @given(semantic_constraint_strategy)
    def test_kind_is_valid(self, constraint: SemanticConstraint) -> None:
        """kind should be one of the valid values."""
        assert constraint.kind in SemanticConstraint.VALID_KINDS


# =============================================================================
# Property Tests: ConstraintSpec
# =============================================================================

class TestConstraintSpecProperties:
    """Property tests for ConstraintSpec dataclass."""

    @given(constraint_spec_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_roundtrip_dict_serialization(self, spec: ConstraintSpec) -> None:
        """ConstraintSpec should survive roundtrip through dict serialization."""
        d = spec.to_dict()
        reconstructed = ConstraintSpec.from_dict(d)

        # Compare key fields (some fields may be normalized)
        assert reconstructed.version == spec.version
        assert reconstructed.json_schema == spec.json_schema
        assert reconstructed.regex == spec.regex
        assert reconstructed.language == spec.language
        assert reconstructed.cache_scope == spec.cache_scope

    @given(constraint_spec_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_roundtrip_json_serialization(self, spec: ConstraintSpec) -> None:
        """ConstraintSpec should survive roundtrip through JSON serialization."""
        json_str = spec.to_json()
        # Verify it's valid JSON
        parsed = json.loads(json_str)
        reconstructed = ConstraintSpec.from_json(json_str)

        # Compare key fields
        assert reconstructed.version == spec.version
        assert reconstructed.json_schema == spec.json_schema
        assert reconstructed.regex == spec.regex

    @given(constraint_spec_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_cache_key_deterministic(self, spec: ConstraintSpec) -> None:
        """compute_cache_key should be deterministic."""
        key1 = spec.compute_cache_key()
        key2 = spec.compute_cache_key()
        assert key1 == key2

    @given(constraint_spec_strategy(), constraint_spec_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_different_specs_different_keys(self, spec1: ConstraintSpec, spec2: ConstraintSpec) -> None:
        """Different specs should (usually) have different cache keys."""
        # This is a probabilistic property - not always true but usually
        if spec1.json_schema != spec2.json_schema or spec1.regex != spec2.regex:
            key1 = spec1.compute_cache_key()
            key2 = spec2.compute_cache_key()
            # Different core constraints should yield different keys
            assert key1 != key2

    @given(constraint_spec_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_dict_keys_are_strings(self, spec: ConstraintSpec) -> None:
        """All dict keys should be strings for JSON compatibility."""
        d = spec.to_dict()

        def check_keys(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    assert isinstance(k, str), f"Non-string key at {path}: {k}"
                    check_keys(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_keys(item, f"{path}[{i}]")

        check_keys(d)


# =============================================================================
# Property Tests: ConstraintSpecParser
# =============================================================================

class TestConstraintSpecParserProperties:
    """Property tests for ConstraintSpecParser."""

    @given(json_schema_strategy)
    def test_parse_json_schema(self, schema: str) -> None:
        """Parser should successfully parse valid JSON schemas."""
        parser = ConstraintSpecParser()
        spec = parser.parse(json_schema=schema)
        assert spec is not None
        assert spec.json_schema == schema

    @given(regex_strategy)
    def test_parse_regex(self, regex: str) -> None:
        """Parser should successfully parse valid regex patterns."""
        parser = ConstraintSpecParser()
        spec = parser.parse(regex=regex)
        assert spec is not None
        assert spec.regex == regex

    @given(constraint_spec_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_parse_inline_roundtrip(self, spec: ConstraintSpec) -> None:
        """Parser should roundtrip through inline dict format."""
        parser = ConstraintSpecParser()
        d = spec.to_dict()
        parsed = parser.parse(constraint_spec=d)

        assert parsed is not None
        assert parsed.json_schema == spec.json_schema
        assert parsed.regex == spec.regex


# =============================================================================
# Property Tests: LanguageDetector
# =============================================================================

class TestLanguageDetectorProperties:
    """Property tests for LanguageDetector."""

    @given(st.text(min_size=0, max_size=100))
    def test_detect_returns_valid_language(self, text: str) -> None:
        """detect() should always return a valid language string."""
        detector = LanguageDetector(use_tree_sitter=False, default_language="python")
        result = detector.detect(text)
        assert isinstance(result, str)
        assert len(result) > 0

    @given(st.text(min_size=0, max_size=100))
    def test_detect_with_confidence_returns_result(self, text: str) -> None:
        """detect_with_confidence() should return DetectionResult."""
        detector = LanguageDetector(use_tree_sitter=False, default_language="python")
        result = detector.detect_with_confidence(text)
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.confidence <= 1.0

    @given(language_strategy)
    def test_default_language_used_for_empty(self, default: str) -> None:
        """Empty text should return default language."""
        detector = LanguageDetector(use_tree_sitter=False, default_language=default)
        result = detector.detect("")
        assert result == default


# =============================================================================
# Property Tests: LanguageStackManager
# =============================================================================

class TestLanguageStackManagerProperties:
    """Property tests for LanguageStackManager."""

    @given(language_strategy)
    def test_initial_language(self, language: str) -> None:
        """Initial language should be preserved."""
        manager = LanguageStackManager(language)
        assert manager.current_language == language

    @given(language_strategy, st.lists(st.tuples(language_strategy, st.integers(min_value=0, max_value=1000)), max_size=5))
    def test_push_pop_preserves_stack(self, base_lang: str, frames: List[Tuple[str, int]]) -> None:
        """Push then pop should restore previous language."""
        manager = LanguageStackManager(base_lang)

        for lang, pos in frames:
            manager.push(lang, pos)

        for _ in frames:
            manager.pop()

        assert manager.current_language == base_lang

    @given(language_strategy)
    def test_cannot_pop_base(self, language: str) -> None:
        """Cannot pop the base language frame."""
        manager = LanguageStackManager(language)
        result = manager.pop()
        assert result is None
        assert manager.current_language == language


# =============================================================================
# Property Tests: Convenience Overrides
# =============================================================================

class TestConvenienceOverridesProperties:
    """Property tests for apply_convenience_overrides."""

    @given(constraint_spec_strategy(), language_strategy)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_language_override(self, spec: ConstraintSpec, new_lang: str) -> None:
        """Language override should set the language."""
        result = apply_convenience_overrides(spec, constraint_language=new_lang)
        assert result.language == new_lang

    @given(constraint_spec_strategy(), type_expr_strategy)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_expected_type_override(self, spec: ConstraintSpec, expected: str) -> None:
        """Expected type override should set expected_type."""
        result = apply_convenience_overrides(spec, expected_type=expected)
        assert result.expected_type == expected

    @given(constraint_spec_strategy(), cache_scope_strategy)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_cache_scope_override(self, spec: ConstraintSpec, scope: CacheScope) -> None:
        """Cache scope override should set cache_scope."""
        result = apply_convenience_overrides(spec, cache_scope=str(scope))
        assert result.cache_scope == scope


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
