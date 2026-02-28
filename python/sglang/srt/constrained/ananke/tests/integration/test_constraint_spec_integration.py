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
"""Integration tests for ConstraintSpec end-to-end flow.

These tests verify the complete flow from:
1. ConstraintSpec creation and parsing
2. Domain context seeding
3. Grammar object creation via dispatch_with_spec
4. Context injection into cached grammars
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spec.constraint_spec import (
    CacheScope,
    ClassDefinition,
    ConstraintSpec,
    ControlFlowContext,
    FunctionSignature,
    ImportBinding,
    LanguageDetection,
    LanguageFrame,
    SemanticConstraint,
    TypeBinding,
)
from spec.parser import ConstraintSpecParser
from spec.language_detector import LanguageDetector, LanguageStackManager

from domains.types.domain import TypeDomain
from domains.imports.domain import ImportDomain
from domains.controlflow.domain import ControlFlowDomain
from domains.semantics.domain import SemanticDomain


class TestTypeDomainIntegration:
    """Integration tests for TypeDomain context seeding."""

    def test_inject_context_with_type_bindings(self) -> None:
        """TypeDomain should correctly inject type bindings from ConstraintSpec."""
        spec = ConstraintSpec(
            type_bindings=[
                TypeBinding(name="x", type_expr="int"),
                TypeBinding(name="y", type_expr="str"),
                TypeBinding(name="z", type_expr="List[int]"),
            ],
            expected_type="bool",
        )

        domain = TypeDomain(language="python")
        domain.inject_context(spec)

        # Verify bindings were added
        assert domain.lookup_variable("x") is not None
        assert domain.lookup_variable("y") is not None
        assert domain.lookup_variable("z") is not None
        assert domain.lookup_variable("nonexistent") is None

    def test_inject_context_with_type_aliases(self) -> None:
        """TypeDomain should correctly inject type aliases from ConstraintSpec."""
        spec = ConstraintSpec(
            type_aliases={
                "UserId": "int",
                "UserName": "str",
            },
        )

        domain = TypeDomain(language="python")
        domain.inject_context(spec)

        # Verify aliases were added as bindings
        assert domain.lookup_variable("UserId") is not None
        assert domain.lookup_variable("UserName") is not None

    def test_inject_context_with_expected_type(self) -> None:
        """TypeDomain should correctly set expected type from ConstraintSpec."""
        spec = ConstraintSpec(
            expected_type="Dict[str, int]",
        )

        domain = TypeDomain(language="python")
        domain.inject_context(spec)

        assert domain.expected_type is not None

    def test_register_function(self) -> None:
        """TypeDomain should correctly register function signatures."""
        domain = TypeDomain(language="python")

        domain.register_function(
            name="add",
            params=[("a", domain._parse_type_expr("int")), ("b", domain._parse_type_expr("int"))],
            return_type=domain._parse_type_expr("int"),
        )

        # Function should be in environment
        func_type = domain.lookup_variable("add")
        assert func_type is not None

    def test_register_class(self) -> None:
        """TypeDomain should correctly register class definitions."""
        domain = TypeDomain(language="python")

        domain.register_class(
            name="MyClass",
            bases=["BaseClass"],
            type_params=["T"],
        )

        # Class should be in environment
        class_type = domain.lookup_variable("MyClass")
        assert class_type is not None


class TestImportDomainIntegration:
    """Integration tests for ImportDomain context seeding."""

    def test_inject_context_with_imports(self) -> None:
        """ImportDomain should correctly inject imports from ConstraintSpec."""
        spec = ConstraintSpec(
            imports=[
                ImportBinding(module="numpy", alias="np"),
                ImportBinding(module="typing", name="List"),
                ImportBinding(module="os"),
            ],
            available_modules=frozenset({"json", "sys"}),
        )

        domain = ImportDomain(language="python")
        domain.inject_context(spec)

        # Verify imports were added
        assert domain.is_imported("numpy")
        assert domain.is_imported("typing")
        assert domain.is_imported("os")
        assert domain.is_imported("json")
        assert domain.is_imported("sys")

    def test_add_import_with_name(self) -> None:
        """ImportDomain should track specific name imports."""
        domain = ImportDomain(language="python")

        domain.add_import(module="typing", name="List")

        assert domain.is_imported("typing")
        assert domain.is_imported("typing.List")

    def test_set_available_modules(self) -> None:
        """ImportDomain should correctly set available modules."""
        domain = ImportDomain(language="python")

        domain.set_available_modules({"numpy", "pandas", "torch"})

        assert domain.is_imported("numpy")
        assert domain.is_imported("pandas")
        assert domain.is_imported("torch")


class TestControlFlowDomainIntegration:
    """Integration tests for ControlFlowDomain context seeding."""

    def test_inject_context_with_function(self) -> None:
        """ControlFlowDomain should correctly set function context."""
        spec = ConstraintSpec(
            control_flow=ControlFlowContext(
                function_name="my_function",
                expected_return_type="int",
                in_async_context=False,
                in_generator=False,
            ),
        )

        domain = ControlFlowDomain(language="python")
        domain.inject_context(spec)

        # Verify function context was set
        assert domain._in_function_context()

    def test_inject_context_with_loop(self) -> None:
        """ControlFlowDomain should correctly set loop context."""
        spec = ConstraintSpec(
            control_flow=ControlFlowContext(
                function_name="process",
                loop_depth=2,
                loop_variables=("i", "j"),
            ),
        )

        domain = ControlFlowDomain(language="python")
        domain.inject_context(spec)

        # Verify loop context was set
        assert domain._in_loop_context()

    def test_inject_context_with_try_block(self) -> None:
        """ControlFlowDomain should correctly set try/except context."""
        spec = ConstraintSpec(
            control_flow=ControlFlowContext(
                function_name="safe_divide",
                in_try_block=True,
                exception_types=("ValueError", "ZeroDivisionError"),
            ),
        )

        domain = ControlFlowDomain(language="python")
        domain.inject_context(spec)

        # Verify try context was set (checking control stack)
        found_try = any(ctx.get("type") == "try" for ctx in domain._control_stack)
        assert found_try

    def test_set_function_context_directly(self) -> None:
        """ControlFlowDomain should correctly set function context directly."""
        domain = ControlFlowDomain(language="python")

        domain.set_function_context(
            function_name="calculate",
            expected_return_type="float",
            is_async=True,
            is_generator=False,
        )

        assert domain._in_function_context()
        # Check that async context was recorded
        func_ctx = next((ctx for ctx in domain._control_stack if ctx.get("type") == "function"), None)
        assert func_ctx is not None
        assert func_ctx.get("is_async") is True


class TestSemanticDomainIntegration:
    """Integration tests for SemanticDomain context seeding."""

    def test_inject_context_with_constraints(self) -> None:
        """SemanticDomain should correctly inject semantic constraints."""
        spec = ConstraintSpec(
            semantic_constraints=[
                SemanticConstraint(
                    kind="precondition",
                    expression="x > 0",
                    variables=("x",),
                ),
                SemanticConstraint(
                    kind="postcondition",
                    expression="result >= 0",
                    variables=("result",),
                ),
            ],
        )

        domain = SemanticDomain(language="python")
        domain.inject_context(spec)

        # Verify constraints were added
        assert domain.formula_count == 2

    def test_set_bounds_directly(self) -> None:
        """SemanticDomain should correctly set variable bounds."""
        domain = SemanticDomain(language="python")

        domain.set_bounds("x", lower=0, upper=100, is_float=False)

        # Verify bounds were set
        assert "x" in domain._variable_bounds
        bounds = domain._variable_bounds["x"]
        assert bounds.lower == 0
        assert bounds.upper == 100

    def test_add_semantic_constraint(self) -> None:
        """SemanticDomain should correctly add individual constraints."""
        domain = SemanticDomain(language="python")

        constraint = domain.add_semantic_constraint(
            kind="invariant",
            expression="len(items) <= max_size",
            scope="class:Container",
            variables=["items", "max_size"],
        )

        # The returned constraint should have the formula
        assert constraint.formula_count() > 0


class TestConstraintSpecParserIntegration:
    """Integration tests for ConstraintSpecParser."""

    def test_parse_full_spec(self) -> None:
        """Parser should correctly parse a full constraint spec."""
        spec_dict = {
            "version": "1.0",
            "json_schema": '{"type": "object", "properties": {"name": {"type": "string"}}}',
            "language": "python",
            "type_bindings": [
                {"name": "x", "type_expr": "int"},
                {"name": "y", "type_expr": "str"},
            ],
            "imports": [
                {"module": "typing", "name": "List"},
            ],
            "expected_type": "Dict[str, Any]",
            "cache_scope": "syntax_and_lang",
        }

        parser = ConstraintSpecParser()
        spec = parser.parse(constraint_spec=spec_dict)

        assert spec is not None
        assert spec.version == "1.0"
        assert spec.json_schema is not None
        assert spec.language == "python"
        assert len(spec.type_bindings) == 2
        assert len(spec.imports) == 1
        assert spec.expected_type == "Dict[str, Any]"
        assert spec.cache_scope == CacheScope.SYNTAX_AND_LANG

    def test_parse_legacy_params(self) -> None:
        """Parser should correctly parse legacy JSON/regex parameters."""
        parser = ConstraintSpecParser()

        # Legacy JSON schema
        spec = parser.parse(json_schema='{"type": "string"}')
        assert spec is not None
        assert spec.json_schema == '{"type": "string"}'

        # Legacy regex
        spec = parser.parse(regex=r"[a-z]+")
        assert spec is not None
        assert spec.regex == r"[a-z]+"

    def test_parse_json_string(self) -> None:
        """Parser should correctly parse JSON string spec."""
        json_str = '{"version": "1.0", "json_schema": "{\\"type\\": \\"object\\"}"}'

        parser = ConstraintSpecParser()
        spec = parser.parse(constraint_spec=json_str)

        assert spec is not None
        assert spec.json_schema is not None


class TestLanguageDetectorIntegration:
    """Integration tests for LanguageDetector."""

    def test_detect_python_code(self) -> None:
        """LanguageDetector should correctly identify Python code."""
        detector = LanguageDetector(use_tree_sitter=False)

        python_code = """
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

class Calculator:
    def __init__(self):
        self.value = 0
"""

        result = detector.detect_with_confidence(python_code)
        assert result.language == "python"
        assert result.confidence > 0.5

    def test_detect_typescript_code(self) -> None:
        """LanguageDetector should correctly identify TypeScript code."""
        detector = LanguageDetector(use_tree_sitter=False)

        typescript_code = """
interface User {
    name: string;
    age: number;
}

const greet = (user: User): string => {
    return `Hello, ${user.name}`;
};

class UserService {
    private users: User[] = [];
}
"""

        result = detector.detect_with_confidence(typescript_code)
        assert result.language == "typescript"
        assert result.confidence > 0.5

    def test_detect_go_code(self) -> None:
        """LanguageDetector should correctly identify Go code."""
        detector = LanguageDetector(use_tree_sitter=False)

        go_code = """
package main

import "fmt"

func main() {
    x := 42
    fmt.Println(x)
}

type User struct {
    Name string
    Age  int
}
"""

        result = detector.detect_with_confidence(go_code)
        assert result.language == "go"
        assert result.confidence > 0.5

    def test_detect_with_candidates(self) -> None:
        """LanguageDetector should respect candidate set."""
        detector = LanguageDetector(use_tree_sitter=False)

        # Code with Go-specific syntax
        go_code = """
package main

func main() {
    x := 42
}
"""

        # Restrict to only Rust and Go - should identify as Go
        result = detector.detect_with_confidence(go_code, candidates={"rust", "go"})
        assert result.language == "go"


class TestLanguageStackManagerIntegration:
    """Integration tests for LanguageStackManager."""

    def test_polyglot_context(self) -> None:
        """LanguageStackManager should correctly track polyglot contexts."""
        manager = LanguageStackManager("python")

        # Start in Python
        assert manager.current_language == "python"

        # Enter SQL context (e.g., in a docstring or f-string)
        manager.push("sql", 100, delimiter="'''")
        assert manager.current_language == "sql"
        assert len(manager.stack) == 2

        # Enter JavaScript context (nested)
        manager.push("javascript", 200, delimiter="`")
        assert manager.current_language == "javascript"
        assert len(manager.stack) == 3

        # Exit JavaScript
        manager.pop()
        assert manager.current_language == "sql"

        # Exit SQL
        manager.pop()
        assert manager.current_language == "python"

    def test_language_at_position(self) -> None:
        """LanguageStackManager should correctly report language at position."""
        manager = LanguageStackManager("python")
        manager.push("sql", 100)
        manager.push("javascript", 200)

        assert manager.language_at_position(50) == "python"
        assert manager.language_at_position(150) == "sql"
        assert manager.language_at_position(250) == "javascript"


class TestCacheScopeIntegration:
    """Integration tests for cache scope behavior."""

    def test_syntax_only_cache_key(self) -> None:
        """SYNTAX_ONLY should produce same key regardless of context."""
        spec1 = ConstraintSpec(
            json_schema='{"type": "object"}',
            language="python",
            type_bindings=[TypeBinding(name="x", type_expr="int")],
            cache_scope=CacheScope.SYNTAX_ONLY,
        )

        spec2 = ConstraintSpec(
            json_schema='{"type": "object"}',
            language="typescript",  # Different language
            type_bindings=[TypeBinding(name="y", type_expr="str")],  # Different bindings
            cache_scope=CacheScope.SYNTAX_ONLY,
        )

        # Same JSON schema, same cache key
        assert spec1.compute_cache_key() == spec2.compute_cache_key()

    def test_syntax_and_lang_cache_key(self) -> None:
        """SYNTAX_AND_LANG should include language in cache key."""
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

        # Different language, different cache key
        assert spec1.compute_cache_key() != spec2.compute_cache_key()

    def test_full_context_cache_key(self) -> None:
        """FULL_CONTEXT should include all context in cache key."""
        spec1 = ConstraintSpec(
            json_schema='{"type": "object"}',
            language="python",
            type_bindings=[TypeBinding(name="x", type_expr="int")],
            cache_scope=CacheScope.FULL_CONTEXT,
        )

        spec2 = ConstraintSpec(
            json_schema='{"type": "object"}',
            language="python",
            type_bindings=[TypeBinding(name="y", type_expr="str")],  # Different bindings
            cache_scope=CacheScope.FULL_CONTEXT,
        )

        # Different context, different cache key
        assert spec1.compute_cache_key() != spec2.compute_cache_key()


class TestEndToEndFlow:
    """End-to-end integration tests for the complete flow."""

    def test_full_flow_with_type_context(self) -> None:
        """Test complete flow from spec creation to domain seeding."""
        # 1. Create a rich constraint spec
        spec = ConstraintSpec(
            json_schema='{"type": "object", "properties": {"value": {"type": "integer"}}}',
            language="python",
            language_detection=LanguageDetection.EXPLICIT,
            type_bindings=[
                TypeBinding(name="user_id", type_expr="int"),
                TypeBinding(name="user_name", type_expr="str"),
            ],
            function_signatures=[
                FunctionSignature(
                    name="get_user",
                    params=(TypeBinding(name="id", type_expr="int"),),
                    return_type="Optional[User]",
                ),
            ],
            expected_type="Dict[str, Any]",
            imports=[
                ImportBinding(module="typing", name="Dict"),
                ImportBinding(module="typing", name="Any"),
                ImportBinding(module="typing", name="Optional"),
            ],
            cache_scope=CacheScope.SYNTAX_AND_LANG,
        )

        # 2. Create domains with context
        type_domain = TypeDomain(language="python")
        type_domain.inject_context(spec)

        import_domain = ImportDomain(language="python")
        import_domain.inject_context(spec)

        # 3. Verify context was properly seeded
        assert type_domain.lookup_variable("user_id") is not None
        assert type_domain.lookup_variable("user_name") is not None
        assert import_domain.is_imported("typing")

        # 4. Verify cache key is deterministic
        key1 = spec.compute_cache_key()
        key2 = spec.compute_cache_key()
        assert key1 == key2

        # 5. Verify serialization roundtrip
        spec_dict = spec.to_dict()
        spec_restored = ConstraintSpec.from_dict(spec_dict)
        assert spec_restored.json_schema == spec.json_schema
        assert spec_restored.language == spec.language
        assert len(spec_restored.type_bindings) == len(spec.type_bindings)

    def test_full_flow_with_control_flow_context(self) -> None:
        """Test complete flow with control flow context."""
        # 1. Create spec with control flow context
        spec = ConstraintSpec(
            regex=r"return \d+",
            language="python",
            control_flow=ControlFlowContext(
                function_name="calculate_total",
                expected_return_type="int",
                loop_depth=1,
                loop_variables=("item",),
                in_try_block=True,
                exception_types=("ValueError",),
            ),
        )

        # 2. Create and seed control flow domain
        cf_domain = ControlFlowDomain(language="python")
        cf_domain.inject_context(spec)

        # 3. Verify context was properly seeded
        assert cf_domain._in_function_context()
        assert cf_domain._in_loop_context()

        # 4. Verify we can create constraints
        constraint = cf_domain.create_constraint(
            must_reach=["function_exit"],
            requires_termination=True,
        )
        assert constraint is not None

    def test_full_flow_with_semantic_constraints(self) -> None:
        """Test complete flow with semantic constraints."""
        # 1. Create spec with semantic constraints
        spec = ConstraintSpec(
            json_schema='{"type": "integer"}',
            language="python",
            semantic_constraints=[
                SemanticConstraint(
                    kind="precondition",
                    expression="n > 0",
                    scope="factorial",
                    variables=("n",),
                ),
                SemanticConstraint(
                    kind="postcondition",
                    expression="result >= 1",
                    scope="factorial",
                    variables=("result",),
                ),
                SemanticConstraint(
                    kind="invariant",
                    expression="acc >= 1",
                    variables=("acc",),
                ),
            ],
        )

        # 2. Create and seed semantic domain
        sem_domain = SemanticDomain(language="python")
        sem_domain.inject_context(spec)

        # 3. Verify constraints were added
        assert sem_domain.formula_count == 3

        # 4. Verify bounds were extracted
        # "n > 0" should create a lower bound for n
        assert "n" in sem_domain._variable_bounds
        bounds = sem_domain._variable_bounds["n"]
        assert bounds.lower is not None
        assert bounds.lower >= 1  # n > 0 means n >= 1 for integers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
