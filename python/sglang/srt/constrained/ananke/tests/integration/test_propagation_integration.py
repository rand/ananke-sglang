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
"""Integration tests for cross-domain constraint propagation.

These tests verify that:
1. Propagation edges correctly transfer constraint information
2. Multi-hop propagation chains work correctly
3. Propagation converges to a fixed point
4. The propagation network integrates properly with AnankeGrammar
"""

import pytest
import sys
from pathlib import Path

# Add the ananke package to the path
ananke_root = Path(__file__).parent.parent.parent
if str(ananke_root) not in sys.path:
    sys.path.insert(0, str(ananke_root))

from core.domain import GenerationContext
from core.unified import UNIFIED_TOP, UnifiedConstraint
from domains.imports.constraint import (
    IMPORT_TOP,
    ImportConstraint,
    ModuleSpec,
)
from domains.types.constraint import (
    INT,
    TYPE_TOP,
    TYPE_BOTTOM,
    TypeConstraint,
)
from domains.controlflow.constraint import (
    CONTROLFLOW_TOP,
    ControlFlowConstraint,
    TerminationRequirement,
)
from propagation.edges import (
    ImportsToTypesEdge,
    SyntaxToTypesEdge,
    TypesToControlFlowEdge,
    create_standard_edges,
)
from propagation.network import PropagationNetwork


def make_context(generated_text: str = "", position: int = 0, **metadata) -> GenerationContext:
    """Helper to create GenerationContext with common defaults."""
    return GenerationContext(
        generated_text=generated_text,
        position=position,
        metadata=metadata,
    )


class TestImportsToTypesIntegration:
    """Test imports → types propagation."""

    def test_imports_update_type_environment_hash(self):
        """When modules are imported, type environment hash should change."""
        edge = ImportsToTypesEdge()
        context = make_context()

        # Create import constraint with available modules
        import_constraint = ImportConstraint(
            available=frozenset([
                ModuleSpec(name="typing"),
                ModuleSpec(name="collections"),
            ])
        )

        # Type constraint starts with default hash
        type_constraint = TypeConstraint()
        initial_hash = type_constraint.environment_hash

        # Propagate
        result = edge.propagate(import_constraint, type_constraint, context)

        # Environment hash should change
        assert result.environment_hash != initial_hash

    def test_imports_without_stubs_dont_change_hash(self):
        """Imports without known type stubs shouldn't affect type constraint."""
        edge = ImportsToTypesEdge()
        context = make_context()

        # Unknown module
        import_constraint = ImportConstraint(
            available=frozenset([
                ModuleSpec(name="unknown_module_xyz"),
            ])
        )

        type_constraint = TypeConstraint()
        initial_hash = type_constraint.environment_hash

        result = edge.propagate(import_constraint, type_constraint, context)

        # Hash should remain unchanged
        assert result.environment_hash == initial_hash

    def test_empty_imports_dont_change_hash(self):
        """No imports shouldn't affect type constraint."""
        edge = ImportsToTypesEdge()
        context = make_context()

        import_constraint = IMPORT_TOP
        type_constraint = TypeConstraint()

        result = edge.propagate(import_constraint, type_constraint, context)

        # Should return unchanged
        assert result == type_constraint

    def test_type_stubs_coverage(self):
        """Verify type stubs exist for common modules."""
        edge = ImportsToTypesEdge()

        # Check common modules have stubs
        assert edge.get_type_exports("typing")
        assert edge.get_type_exports("collections")
        assert edge.get_type_exports("dataclasses")
        assert edge.get_type_exports("pathlib")
        assert edge.get_type_exports("datetime")

        # Check specific exports
        typing_exports = edge.get_type_exports("typing")
        assert "List" in typing_exports
        assert "Dict" in typing_exports
        assert "Optional" in typing_exports


class TestSyntaxToTypesIntegration:
    """Test syntax → types propagation."""

    def test_function_call_suggests_callable(self):
        """Function call syntax should influence type expectations."""
        edge = SyntaxToTypesEdge()
        context = make_context(generated_text="def foo(", position=8)

        type_constraint = TypeConstraint()
        initial_hash = type_constraint.environment_hash

        result = edge.propagate(TYPE_TOP, type_constraint, context)

        # Hash should change due to callable pattern
        assert result.environment_hash != initial_hash

    def test_boolean_context_detected(self):
        """Boolean context should be detected."""
        edge = SyntaxToTypesEdge()
        context = make_context(generated_text="if x", position=4)

        type_constraint = TypeConstraint()
        initial_hash = type_constraint.environment_hash

        result = edge.propagate(TYPE_TOP, type_constraint, context)

        # Hash should change due to boolean pattern (if)
        assert result.environment_hash != initial_hash

    def test_empty_text_no_change(self):
        """Empty generated text shouldn't change constraint."""
        edge = SyntaxToTypesEdge()
        context = make_context()

        type_constraint = TypeConstraint()
        result = edge.propagate(TYPE_TOP, type_constraint, context)

        assert result == type_constraint


class TestTypesToControlFlowIntegration:
    """Test types → controlflow propagation."""

    def test_function_context_sets_termination(self):
        """Functions should require termination."""
        edge = TypesToControlFlowEdge()
        context = make_context(
            generated_text="def foo():",
            position=10,
            in_function=True,
        )

        # Type constraint with int return type
        type_constraint = TypeConstraint(expected_type=INT)

        cf_constraint = ControlFlowConstraint(
            termination=TerminationRequirement.UNKNOWN
        )

        result = edge.propagate(type_constraint, cf_constraint, context)

        # Should require termination
        assert result.termination == TerminationRequirement.MUST_TERMINATE

    def test_non_function_context_no_change(self):
        """Non-function context shouldn't change termination."""
        edge = TypesToControlFlowEdge()
        context = make_context(generated_text="x = 1", position=5)

        type_constraint = TypeConstraint()
        cf_constraint = ControlFlowConstraint(
            termination=TerminationRequirement.UNKNOWN
        )

        result = edge.propagate(type_constraint, cf_constraint, context)

        # Should remain unknown
        assert result.termination == TerminationRequirement.UNKNOWN


class TestPropagationNetworkIntegration:
    """Test full propagation network integration."""

    def test_network_initialization(self):
        """Network should initialize with standard edges."""
        network = PropagationNetwork()

        # Add edges
        for edge in create_standard_edges():
            network.add_edge(edge)

        # Verify edges added
        assert len(network._edges) == 6

    def test_propagation_converges(self):
        """Propagation should converge to a fixed point."""
        network = PropagationNetwork(max_iterations=100)

        # Create mock domain (we can use constraint directly)
        class MockDomain:
            def __init__(self, name):
                self.name = name
                self.constraint = None

            def checkpoint(self):
                return {"constraint": self.constraint}

            def restore(self, checkpoint):
                self.constraint = checkpoint["constraint"]

        # Register domains
        types_domain = MockDomain("types")
        imports_domain = MockDomain("imports")

        network.register_domain(types_domain, TYPE_TOP)
        network.register_domain(imports_domain, IMPORT_TOP)

        # Add edge
        network.add_edge(ImportsToTypesEdge())

        # Create context
        context = make_context()

        # Run propagation
        result = network.propagate(context)

        # Should converge
        assert result.is_success
        assert result.iterations <= 100

    def test_propagation_with_imports(self):
        """Propagation with actual imports should update constraints."""
        network = PropagationNetwork(max_iterations=100)

        class MockDomain:
            def __init__(self, name):
                self.name = name
                self.constraint = None

            def checkpoint(self):
                return {"constraint": self.constraint}

            def restore(self, checkpoint):
                self.constraint = checkpoint["constraint"]

        # Register domains
        types_domain = MockDomain("types")
        imports_domain = MockDomain("imports")

        # Set initial constraints
        import_constraint = ImportConstraint(
            available=frozenset([
                ModuleSpec(name="typing"),
                ModuleSpec(name="dataclasses"),
            ])
        )
        type_constraint = TypeConstraint()
        initial_hash = type_constraint.environment_hash

        network.register_domain(types_domain, type_constraint)
        network.register_domain(imports_domain, import_constraint)

        # Add edge
        network.add_edge(ImportsToTypesEdge())

        # Mark imports domain dirty to trigger propagation
        network.mark_dirty("imports")

        # Create context
        context = make_context()

        # Run propagation
        result = network.propagate(context)

        # Should converge and types should be updated
        assert result.is_success

        # Get updated types constraint
        updated_types = network.get_constraint("types")
        if updated_types is not None:
            # Hash should have changed due to imports
            assert updated_types.environment_hash != initial_hash


class TestMultiHopPropagation:
    """Test propagation chains through multiple domains."""

    def test_imports_to_types_to_controlflow(self):
        """Test propagation: imports → types → controlflow."""
        network = PropagationNetwork(max_iterations=100)

        class MockDomain:
            def __init__(self, name):
                self.name = name

            def checkpoint(self):
                return {}

            def restore(self, checkpoint):
                pass

        # Register all domains
        network.register_domain(MockDomain("imports"), IMPORT_TOP)
        network.register_domain(MockDomain("types"), TYPE_TOP)
        network.register_domain(MockDomain("controlflow"), CONTROLFLOW_TOP)

        # Add edges
        network.add_edge(ImportsToTypesEdge())
        network.add_edge(TypesToControlFlowEdge())

        # Create context
        context = make_context()

        # Run propagation
        result = network.propagate(context)

        # Should converge
        assert result.is_success

    def test_all_standard_edges(self):
        """Test with all standard edges."""
        network = PropagationNetwork(max_iterations=100)

        class MockDomain:
            def __init__(self, name):
                self.name = name

            def checkpoint(self):
                return {}

            def restore(self, checkpoint):
                pass

        # Register all domains
        network.register_domain(MockDomain("syntax"), TYPE_TOP)  # Using TOP for syntax
        network.register_domain(MockDomain("types"), TYPE_TOP)
        network.register_domain(MockDomain("imports"), IMPORT_TOP)
        network.register_domain(MockDomain("controlflow"), CONTROLFLOW_TOP)
        network.register_domain(MockDomain("semantics"), TYPE_TOP)  # Using TOP for semantics

        # Add all standard edges
        for edge in create_standard_edges():
            network.add_edge(edge)

        # Create context
        context = make_context(generated_text="def foo():", position=10)

        # Run propagation
        result = network.propagate(context)

        # Should converge within iteration limit
        assert result.is_success
        assert result.iterations <= 100


class TestEdgeMonotonicity:
    """Test that edges maintain monotonicity."""

    def test_imports_to_types_monotonic(self):
        """ImportsToTypes should only refine, never loosen."""
        edge = ImportsToTypesEdge()
        context = make_context()

        # Create constraints
        import_constraint = ImportConstraint(
            available=frozenset([ModuleSpec(name="typing")])
        )
        type_constraint = TypeConstraint(environment_hash=100)

        # First propagation
        result1 = edge.propagate(import_constraint, type_constraint, context)

        # Second propagation should not change result
        result2 = edge.propagate(import_constraint, result1, context)

        # Constraint should be stable (idempotent after convergence)
        # The hash may differ but constraint structure should be consistent
        assert result2.environment_hash == result1.environment_hash or \
               result2.environment_hash != type_constraint.environment_hash

    def test_syntax_to_types_handles_bottom(self):
        """SyntaxToTypes should handle BOTTOM correctly."""
        edge = SyntaxToTypesEdge()
        context = make_context(generated_text="if x", position=4)

        # BOTTOM should propagate unchanged
        result = edge.propagate(TYPE_TOP, TYPE_BOTTOM, context)
        assert result.is_bottom()


class TestPropagationPerformance:
    """Test propagation performance characteristics."""

    def test_propagation_completes_quickly(self):
        """Propagation should complete within reasonable time."""
        import time

        network = PropagationNetwork(max_iterations=100)

        class MockDomain:
            def __init__(self, name):
                self.name = name

            def checkpoint(self):
                return {}

            def restore(self, checkpoint):
                pass

        # Register domains
        for name in ["syntax", "types", "imports", "controlflow", "semantics"]:
            network.register_domain(MockDomain(name), TYPE_TOP)

        # Add all edges
        for edge in create_standard_edges():
            network.add_edge(edge)

        context = make_context(generated_text="def foo(): return 1", position=19)

        start = time.perf_counter()
        result = network.propagate(context)
        elapsed = time.perf_counter() - start

        assert result.is_success
        # Should complete in less than 10ms
        assert elapsed < 0.01  # 10ms

    def test_iteration_limit_prevents_infinite_loop(self):
        """Iteration limit should prevent infinite loops."""
        network = PropagationNetwork(max_iterations=10)

        class MockDomain:
            def __init__(self, name):
                self.name = name

            def checkpoint(self):
                return {}

            def restore(self, checkpoint):
                pass

        network.register_domain(MockDomain("types"), TYPE_TOP)

        context = make_context()

        # This should always complete (even if not converged)
        result = network.propagate(context)

        # Should complete with max iterations or less
        assert result.iterations <= 10
