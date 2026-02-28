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
"""Unit tests for ConstraintSpec and related dataclasses."""

from __future__ import annotations

import json
import pytest
import sys
from pathlib import Path

# Add parent to path for imports
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


class TestTypeBinding:
    """Tests for TypeBinding dataclass."""

    def test_basic_construction(self) -> None:
        binding = TypeBinding(name="x", type_expr="int")
        assert binding.name == "x"
        assert binding.type_expr == "int"
        assert binding.scope is None
        assert binding.mutable is True
        assert binding.origin is None

    def test_full_construction(self) -> None:
        binding = TypeBinding(
            name="items",
            type_expr="List[str]",
            scope="local",
            mutable=False,
            origin="parameter",
        )
        assert binding.name == "items"
        assert binding.scope == "local"
        assert binding.mutable is False
        assert binding.origin == "parameter"

    def test_to_dict(self) -> None:
        binding = TypeBinding(name="x", type_expr="int", scope="local")
        d = binding.to_dict()
        assert d["name"] == "x"
        assert d["type_expr"] == "int"
        assert d["scope"] == "local"
        assert "mutable" not in d  # Only included if False

    def test_from_dict(self) -> None:
        d = {"name": "y", "type_expr": "str", "scope": "global", "mutable": False}
        binding = TypeBinding.from_dict(d)
        assert binding.name == "y"
        assert binding.type_expr == "str"
        assert binding.scope == "global"
        assert binding.mutable is False

    def test_frozen(self) -> None:
        binding = TypeBinding(name="x", type_expr="int")
        with pytest.raises(AttributeError):
            binding.name = "y"  # type: ignore


class TestFunctionSignature:
    """Tests for FunctionSignature dataclass."""

    def test_basic_construction(self) -> None:
        sig = FunctionSignature(
            name="foo",
            params=(TypeBinding("x", "int"),),
            return_type="str",
        )
        assert sig.name == "foo"
        assert len(sig.params) == 1
        assert sig.return_type == "str"
        assert sig.is_async is False
        assert sig.is_generator is False

    def test_async_generator(self) -> None:
        sig = FunctionSignature(
            name="gen",
            params=(),
            return_type="AsyncIterator[int]",
            is_async=True,
            is_generator=True,
        )
        assert sig.is_async is True
        assert sig.is_generator is True

    def test_roundtrip(self) -> None:
        sig = FunctionSignature(
            name="process",
            params=(TypeBinding("x", "int"), TypeBinding("y", "str")),
            return_type="bool",
            type_params=("T",),
            decorators=("staticmethod",),
        )
        d = sig.to_dict()
        sig2 = FunctionSignature.from_dict(d)
        assert sig2.name == sig.name
        assert len(sig2.params) == len(sig.params)
        assert sig2.type_params == sig.type_params
        assert sig2.decorators == sig.decorators


class TestClassDefinition:
    """Tests for ClassDefinition dataclass."""

    def test_basic_construction(self) -> None:
        cls_def = ClassDefinition(name="MyClass")
        assert cls_def.name == "MyClass"
        assert cls_def.bases == ()
        assert cls_def.methods == ()

    def test_full_construction(self) -> None:
        cls_def = ClassDefinition(
            name="User",
            bases=("BaseModel",),
            type_params=("T",),
            instance_vars=(
                TypeBinding("id", "int"),
                TypeBinding("name", "str"),
            ),
        )
        assert cls_def.bases == ("BaseModel",)
        assert len(cls_def.instance_vars) == 2

    def test_roundtrip(self) -> None:
        cls_def = ClassDefinition(
            name="Test",
            bases=("Base1", "Base2"),
            methods=(
                FunctionSignature("method", (), "None"),
            ),
        )
        d = cls_def.to_dict()
        cls_def2 = ClassDefinition.from_dict(d)
        assert cls_def2.name == cls_def.name
        assert cls_def2.bases == cls_def.bases
        assert len(cls_def2.methods) == len(cls_def.methods)


class TestImportBinding:
    """Tests for ImportBinding dataclass."""

    def test_module_import(self) -> None:
        imp = ImportBinding(module="os")
        assert imp.module == "os"
        assert imp.name is None
        assert imp.alias is None

    def test_from_import(self) -> None:
        imp = ImportBinding(module="typing", name="List")
        assert imp.module == "typing"
        assert imp.name == "List"

    def test_aliased_import(self) -> None:
        imp = ImportBinding(module="numpy", alias="np")
        assert imp.alias == "np"

    def test_wildcard_import(self) -> None:
        imp = ImportBinding(module="typing", is_wildcard=True)
        assert imp.is_wildcard is True

    def test_roundtrip(self) -> None:
        imp = ImportBinding(module="collections", name="OrderedDict", alias="OD")
        d = imp.to_dict()
        imp2 = ImportBinding.from_dict(d)
        assert imp2.module == imp.module
        assert imp2.name == imp.name
        assert imp2.alias == imp.alias


class TestLanguageFrame:
    """Tests for LanguageFrame dataclass."""

    def test_basic_construction(self) -> None:
        frame = LanguageFrame(language="python", start_position=0)
        assert frame.language == "python"
        assert frame.start_position == 0
        assert frame.delimiter is None

    def test_contains(self) -> None:
        frame = LanguageFrame(language="python", start_position=100)
        assert frame.contains(100) is True
        assert frame.contains(150) is True
        assert frame.contains(50) is False

    def test_with_delimiters(self) -> None:
        frame = LanguageFrame(
            language="sql",
            start_position=50,
            delimiter="'''",
            end_delimiter="'''",
        )
        assert frame.delimiter == "'''"
        assert frame.end_delimiter == "'''"


class TestControlFlowContext:
    """Tests for ControlFlowContext dataclass."""

    def test_default_values(self) -> None:
        ctx = ControlFlowContext()
        assert ctx.function_name is None
        assert ctx.loop_depth == 0
        assert ctx.in_try_block is False
        assert ctx.reachable is True

    def test_full_context(self) -> None:
        ctx = ControlFlowContext(
            function_name="process",
            expected_return_type="bool",
            loop_depth=2,
            loop_variables=("i", "j"),
            in_async_context=True,
        )
        assert ctx.function_name == "process"
        assert ctx.loop_depth == 2
        assert ctx.loop_variables == ("i", "j")

    def test_roundtrip(self) -> None:
        ctx = ControlFlowContext(
            function_name="test",
            in_try_block=True,
            exception_types=("ValueError", "TypeError"),
        )
        d = ctx.to_dict()
        ctx2 = ControlFlowContext.from_dict(d)
        assert ctx2.function_name == ctx.function_name
        assert ctx2.exception_types == ctx.exception_types


class TestSemanticConstraint:
    """Tests for SemanticConstraint dataclass."""

    def test_basic_construction(self) -> None:
        constraint = SemanticConstraint(
            kind="precondition",
            expression="x > 0",
        )
        assert constraint.kind == "precondition"
        assert constraint.expression == "x > 0"

    def test_with_variables(self) -> None:
        constraint = SemanticConstraint(
            kind="postcondition",
            expression="result == x + y",
            scope="add",
            variables=("result", "x", "y"),
        )
        assert constraint.variables == ("result", "x", "y")
        assert constraint.scope == "add"

    def test_invalid_kind(self) -> None:
        with pytest.raises(ValueError):
            SemanticConstraint.from_dict({
                "kind": "invalid_kind",
                "expression": "true",
            })


class TestCacheScope:
    """Tests for CacheScope enum."""

    def test_values(self) -> None:
        assert CacheScope.SYNTAX_ONLY.value == "syntax_only"
        assert CacheScope.SYNTAX_AND_LANG.value == "syntax_and_lang"
        assert CacheScope.FULL_CONTEXT.value == "full_context"

    def test_from_string(self) -> None:
        assert CacheScope.from_string("syntax_only") == CacheScope.SYNTAX_ONLY
        assert CacheScope.from_string("full_context") == CacheScope.FULL_CONTEXT
        assert CacheScope.from_string("invalid") == CacheScope.SYNTAX_ONLY  # default


class TestConstraintSpec:
    """Tests for ConstraintSpec dataclass."""

    def test_basic_construction(self) -> None:
        spec = ConstraintSpec(json_schema='{"type": "object"}')
        assert spec.json_schema == '{"type": "object"}'
        assert spec.version == "1.0"
        assert spec.cache_scope == CacheScope.SYNTAX_ONLY

    def test_has_syntax_constraint(self) -> None:
        spec_json = ConstraintSpec(json_schema="{}")
        assert spec_json.has_syntax_constraint() is True

        spec_regex = ConstraintSpec(regex=r"\d+")
        assert spec_regex.has_syntax_constraint() is True

        spec_empty = ConstraintSpec()
        assert spec_empty.has_syntax_constraint() is False

    def test_get_syntax_constraint_type(self) -> None:
        spec = ConstraintSpec(json_schema="{}")
        assert spec.get_syntax_constraint_type() == "json_schema"

        spec = ConstraintSpec(regex=r".*")
        assert spec.get_syntax_constraint_type() == "regex"

        spec = ConstraintSpec(ebnf="root ::= 'a'")
        assert spec.get_syntax_constraint_type() == "ebnf"

    def test_compute_cache_key_syntax_only(self) -> None:
        spec1 = ConstraintSpec(
            json_schema='{"type": "object"}',
            language="python",
            cache_scope=CacheScope.SYNTAX_ONLY,
        )
        spec2 = ConstraintSpec(
            json_schema='{"type": "object"}',
            language="typescript",  # Different language
            cache_scope=CacheScope.SYNTAX_ONLY,
        )
        # Same cache key because language is not included
        assert spec1.compute_cache_key() == spec2.compute_cache_key()

    def test_compute_cache_key_syntax_and_lang(self) -> None:
        spec1 = ConstraintSpec(
            json_schema='{"type": "object"}',
            language="python",
            cache_scope=CacheScope.SYNTAX_AND_LANG,
        )
        spec2 = ConstraintSpec(
            json_schema='{"type": "object"}',
            language="typescript",
            cache_scope=CacheScope.SYNTAX_AND_LANG,
        )
        # Different cache keys because language is included
        assert spec1.compute_cache_key() != spec2.compute_cache_key()

    def test_compute_cache_key_full_context(self) -> None:
        spec1 = ConstraintSpec(
            json_schema='{"type": "object"}',
            language="python",
            type_bindings=[TypeBinding("x", "int")],
            cache_scope=CacheScope.FULL_CONTEXT,
        )
        spec2 = ConstraintSpec(
            json_schema='{"type": "object"}',
            language="python",
            type_bindings=[TypeBinding("y", "str")],  # Different context
            cache_scope=CacheScope.FULL_CONTEXT,
        )
        # Different cache keys because context is included
        assert spec1.compute_cache_key() != spec2.compute_cache_key()

    def test_to_dict_minimal(self) -> None:
        spec = ConstraintSpec(regex=r"\d+")
        d = spec.to_dict()
        assert d["version"] == "1.0"
        assert d["regex"] == r"\d+"
        assert "json_schema" not in d  # Not set
        assert "cache_scope" not in d  # Default value

    def test_to_dict_full(self) -> None:
        spec = ConstraintSpec(
            json_schema='{"type": "object"}',
            language="python",
            language_detection=LanguageDetection.EXPLICIT,
            type_bindings=[TypeBinding("x", "int")],
            imports=[ImportBinding("typing", "List")],
            control_flow=ControlFlowContext(function_name="test"),
            semantic_constraints=[
                SemanticConstraint("precondition", "x > 0")
            ],
            cache_scope=CacheScope.FULL_CONTEXT,
        )
        d = spec.to_dict()
        assert "type_bindings" in d
        assert "imports" in d
        assert "control_flow" in d
        assert "semantic_constraints" in d
        assert d["cache_scope"] == "full_context"

    def test_roundtrip(self) -> None:
        spec = ConstraintSpec(
            json_schema='{"type": "string"}',
            language="python",
            type_bindings=[
                TypeBinding("x", "int"),
                TypeBinding("y", "str", scope="local"),
            ],
            function_signatures=[
                FunctionSignature("foo", (TypeBinding("a", "int"),), "bool")
            ],
            imports=[
                ImportBinding("typing", "Optional"),
            ],
            available_modules={"typing", "collections"},
            control_flow=ControlFlowContext(loop_depth=1),
            semantic_constraints=[
                SemanticConstraint("invariant", "len(items) > 0")
            ],
            cache_scope=CacheScope.SYNTAX_AND_LANG,
        )
        d = spec.to_dict()
        spec2 = ConstraintSpec.from_dict(d)

        assert spec2.json_schema == spec.json_schema
        assert spec2.language == spec.language
        assert len(spec2.type_bindings) == len(spec.type_bindings)
        assert len(spec2.function_signatures) == len(spec.function_signatures)
        assert len(spec2.imports) == len(spec.imports)
        assert spec2.available_modules == spec.available_modules
        assert spec2.control_flow is not None
        assert spec2.control_flow.loop_depth == 1
        assert len(spec2.semantic_constraints) == 1
        assert spec2.cache_scope == spec.cache_scope

    def test_json_serialization(self) -> None:
        spec = ConstraintSpec(
            regex=r"def \w+",
            language="python",
        )
        json_str = spec.to_json()
        spec2 = ConstraintSpec.from_json(json_str)
        assert spec2.regex == spec.regex
        assert spec2.language == spec.language

    def test_from_legacy(self) -> None:
        spec = ConstraintSpec.from_legacy(json_schema='{"type": "array"}')
        assert spec.json_schema == '{"type": "array"}'
        assert spec.source == ConstraintSource.LEGACY

    def test_copy_with_updates(self) -> None:
        spec = ConstraintSpec(
            json_schema='{"type": "object"}',
            language="python",
        )
        spec2 = spec.copy(language="typescript")
        assert spec2.language == "typescript"
        assert spec2.json_schema == spec.json_schema

    def test_from_dict_domains_alias(self) -> None:
        """Test that 'domains' is accepted as alias for 'enabled_domains'."""
        # Using "domains" shorthand
        d = {"regex": r"\d+", "language": "python", "domains": ["types", "imports"]}
        spec = ConstraintSpec.from_dict(d)
        assert spec.enabled_domains == {"types", "imports"}

    def test_from_dict_enabled_domains_preferred(self) -> None:
        """Test that 'enabled_domains' takes priority over 'domains'."""
        # When both are present, enabled_domains should take priority
        d = {
            "regex": r"\d+",
            "language": "python",
            "enabled_domains": ["types"],
            "domains": ["imports"],
        }
        spec = ConstraintSpec.from_dict(d)
        assert spec.enabled_domains == {"types"}

    def test_from_dict_no_domains(self) -> None:
        """Test that absent domains results in None (backend defaults)."""
        d = {"regex": r"\d+", "language": "python"}
        spec = ConstraintSpec.from_dict(d)
        assert spec.enabled_domains is None

    def test_from_dict_domains_single(self) -> None:
        """Test 'domains' with a single domain."""
        d = {"regex": r"\d+", "domains": ["syntax"]}
        spec = ConstraintSpec.from_dict(d)
        assert spec.enabled_domains == {"syntax"}

    def test_from_dict_domains_all(self) -> None:
        """Test 'domains' with all domains."""
        all_domains = ["syntax", "types", "imports", "controlflow", "semantics"]
        d = {"regex": r"\d+", "domains": all_domains}
        spec = ConstraintSpec.from_dict(d)
        assert spec.enabled_domains == set(all_domains)


class TestConstraintSpecParser:
    """Tests for ConstraintSpecParser."""

    def test_parse_json_dict(self) -> None:
        parser = ConstraintSpecParser()
        spec = parser.parse(constraint_spec={
            "json_schema": '{"type": "object"}',
            "language": "python",
        })
        assert spec is not None
        assert spec.json_schema == '{"type": "object"}'
        assert spec.language == "python"

    def test_parse_json_string(self) -> None:
        parser = ConstraintSpecParser()
        spec = parser.parse(constraint_spec='{"regex": "\\\\d+"}')
        assert spec is not None
        assert spec.regex == r"\d+"

    def test_parse_legacy_json_schema(self) -> None:
        parser = ConstraintSpecParser()
        spec = parser.parse(json_schema='{"type": "string"}')
        assert spec is not None
        assert spec.json_schema == '{"type": "string"}'
        assert spec.source == ConstraintSource.LEGACY

    def test_parse_legacy_regex(self) -> None:
        parser = ConstraintSpecParser()
        spec = parser.parse(regex=r"[a-z]+")
        assert spec is not None
        assert spec.regex == r"[a-z]+"

    def test_parse_legacy_ebnf(self) -> None:
        parser = ConstraintSpecParser()
        spec = parser.parse(ebnf="root ::= 'hello'")
        assert spec is not None
        assert spec.ebnf == "root ::= 'hello'"

    def test_parse_nothing(self) -> None:
        parser = ConstraintSpecParser()
        spec = parser.parse()
        assert spec is None

    def test_parse_priority(self) -> None:
        parser = ConstraintSpecParser()
        # constraint_spec should take priority over legacy
        spec = parser.parse(
            constraint_spec={"regex": r"\d+"},
            json_schema='{"type": "object"}',  # Should be ignored
        )
        assert spec is not None
        assert spec.regex == r"\d+"
        assert spec.json_schema is None

    def test_strict_mode(self) -> None:
        parser = ConstraintSpecParser(strict=True)
        with pytest.raises(ValueError):
            parser.parse(constraint_spec="invalid json{")


class TestApplyConvenienceOverrides:
    """Tests for apply_convenience_overrides function."""

    def test_override_language(self) -> None:
        spec = ConstraintSpec(json_schema="{}")
        spec2 = apply_convenience_overrides(spec, constraint_language="typescript")
        assert spec2.language == "typescript"
        assert spec2.language_detection == LanguageDetection.EXPLICIT

    def test_override_type_bindings(self) -> None:
        spec = ConstraintSpec(
            json_schema="{}",
            type_bindings=[TypeBinding("existing", "int")],
        )
        spec2 = apply_convenience_overrides(
            spec,
            type_bindings={"x": "str", "y": "bool"},
        )
        assert len(spec2.type_bindings) == 3  # 1 existing + 2 new

    def test_override_expected_type(self) -> None:
        spec = ConstraintSpec(json_schema="{}")
        spec2 = apply_convenience_overrides(spec, expected_type="List[int]")
        assert spec2.expected_type == "List[int]"

    def test_override_cache_scope(self) -> None:
        spec = ConstraintSpec(json_schema="{}")
        spec2 = apply_convenience_overrides(spec, cache_scope="full_context")
        assert spec2.cache_scope == CacheScope.FULL_CONTEXT

    def test_no_overrides(self) -> None:
        spec = ConstraintSpec(json_schema="{}", language="python")
        spec2 = apply_convenience_overrides(spec)
        assert spec2 is spec  # Same object when no changes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
