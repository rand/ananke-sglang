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
"""Benchmarks for type checking operations.

Performance targets (from implementation plan):
- Type mask: <500μs
- Single-token type check: <500μs (after warm-up)
- Incremental type check: O(k) where k = affected nodes

Run with: pytest tests/benchmark/bench_type_checking.py -v -s
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pytest

from domains.types.constraint import (
    TypeConstraint,
    TYPE_TOP,
    TYPE_BOTTOM,
    INT,
    STR,
    BOOL,
    FLOAT,
    ANY,
    NONE,
    TypeVar,
    FunctionType,
    ListType,
    DictType,
    UnionType,
    type_expecting,
)
from domains.types.unification import (
    Substitution,
    TypeEquation,
    unify,
    solve_equations,
    unify_types,
    occurs_check,
)
from domains.types.environment import TypeEnvironment
from domains.types.marking.marks import HoleMark, InconsistentMark, NonEmptyHoleMark
from domains.types.marking.provenance import Provenance, SourceSpan
from domains.types.incremental.order_maintenance import OrderMaintenanceList
from domains.types.incremental.dependency_graph import DependencyGraph, DependencyKind
from domains.types.bidirectional.synthesis import synthesize, SynthesisResult
from domains.types.bidirectional.analysis import analyze, AnalysisResult
from domains.types.bidirectional.subsumption import subsumes, check_subsumption, is_assignable
from domains.types.marking.marked_ast import MarkedASTNode, ASTNodeKind
from domains.types.marking.provenance import SourceSpan


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    operation: str
    iterations: int
    total_time_ns: int
    mean_time_ns: float
    min_time_ns: int
    max_time_ns: int

    @property
    def mean_time_us(self) -> float:
        """Mean time in microseconds."""
        return self.mean_time_ns / 1000

    @property
    def mean_time_ms(self) -> float:
        """Mean time in milliseconds."""
        return self.mean_time_ns / 1_000_000


def run_benchmark(
    operation: str,
    func,
    iterations: int = 1000,
) -> BenchmarkResult:
    """Run a benchmark for a function."""
    times: List[int] = []

    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)

    return BenchmarkResult(
        operation=operation,
        iterations=iterations,
        total_time_ns=sum(times),
        mean_time_ns=sum(times) / len(times),
        min_time_ns=min(times),
        max_time_ns=max(times),
    )


class TestUnificationBenchmarks:
    """Benchmarks for type unification."""

    def test_simple_unification(self):
        """Benchmark simple type unification."""
        t1 = TypeVar("T")
        t2 = INT

        result = run_benchmark(
            "simple_unification",
            lambda: unify(t1, t2),
            iterations=10000,
        )

        print(f"\nSimple unification: {result.mean_time_ns:.0f}ns ({result.mean_time_us:.2f}μs)")
        assert result.mean_time_us < 10, "Simple unification should be <10μs"

    def test_function_type_unification(self):
        """Benchmark function type unification."""
        t1 = FunctionType(params=[TypeVar("A"), TypeVar("B")], returns=TypeVar("R"))
        t2 = FunctionType(params=[INT, STR], returns=BOOL)

        result = run_benchmark(
            "function_unification",
            lambda: unify(t1, t2),
            iterations=5000,
        )

        print(f"\nFunction type unification: {result.mean_time_us:.2f}μs")
        assert result.mean_time_us < 50, "Function unification should be <50μs"

    def test_nested_type_unification(self):
        """Benchmark nested type unification."""
        # List[Dict[str, List[T]]] with List[Dict[str, List[int]]]
        t1 = ListType(DictType(STR, ListType(TypeVar("T"))))
        t2 = ListType(DictType(STR, ListType(INT)))

        result = run_benchmark(
            "nested_unification",
            lambda: unify(t1, t2),
            iterations=2000,
        )

        print(f"\nNested type unification: {result.mean_time_us:.2f}μs")
        assert result.mean_time_us < 100, "Nested unification should be <100μs"

    def test_multiple_equations(self):
        """Benchmark solving multiple type equations."""
        equations = [
            TypeEquation(TypeVar("A"), INT),
            TypeEquation(TypeVar("B"), STR),
            TypeEquation(TypeVar("C"), FunctionType(params=[TypeVar("A")], returns=TypeVar("B"))),
            TypeEquation(TypeVar("D"), ListType(TypeVar("A"))),
        ]

        result = run_benchmark(
            "solve_4_equations",
            lambda: solve_equations(equations),
            iterations=2000,
        )

        print(f"\nSolve 4 equations: {result.mean_time_us:.2f}μs")
        assert result.mean_time_us < 100, "Solving 4 equations should be <100μs"

    def test_occurs_check(self):
        """Benchmark occurs check."""
        var = TypeVar("T")
        # Deep nested type
        nested = ListType(ListType(ListType(ListType(var))))

        result = run_benchmark(
            "occurs_check_deep",
            lambda: occurs_check(var, nested),
            iterations=10000,
        )

        print(f"\nOccurs check (4 levels): {result.mean_time_ns:.0f}ns")
        assert result.mean_time_us < 5, "Occurs check should be <5μs"


class TestTypeEnvironmentBenchmarks:
    """Benchmarks for TypeEnvironment operations."""

    def test_environment_lookup(self):
        """Benchmark environment lookup."""
        env = TypeEnvironment()
        for i in range(100):
            env = env.bind(f"var_{i}", INT)

        result = run_benchmark(
            "env_lookup",
            lambda: env.lookup("var_50"),
            iterations=10000,
        )

        print(f"\nEnvironment lookup (100 bindings): {result.mean_time_ns:.0f}ns")
        assert result.mean_time_us < 1, "Lookup should be <1μs"

    def test_environment_bind(self):
        """Benchmark environment binding."""
        env = TypeEnvironment()

        def bind_and_create():
            nonlocal env
            env = env.bind(f"var_{time.perf_counter_ns()}", INT)

        result = run_benchmark(
            "env_bind",
            bind_and_create,
            iterations=1000,
        )

        print(f"\nEnvironment bind: {result.mean_time_us:.2f}μs")
        assert result.mean_time_us < 50, "Bind should be <50μs"

    def test_environment_snapshot(self):
        """Benchmark environment snapshot."""
        env = TypeEnvironment()
        for i in range(100):
            env = env.bind(f"var_{i}", INT)

        result = run_benchmark(
            "env_snapshot",
            lambda: env.snapshot(),
            iterations=5000,
        )

        print(f"\nEnvironment snapshot: {result.mean_time_us:.2f}μs")
        assert result.mean_time_us < 20, "Snapshot should be <20μs"


class TestOrderMaintenanceBenchmarks:
    """Benchmarks for order maintenance data structure."""

    def test_insert_sequential(self):
        """Benchmark sequential insertion."""
        oml = OrderMaintenanceList()
        elements = []

        def insert_elem():
            if not elements:
                elem = oml.insert_first(len(elements))
            else:
                elem = oml.insert_after(elements[-1], len(elements))
            elements.append(elem)

        result = run_benchmark(
            "oml_insert_seq",
            insert_elem,
            iterations=1000,
        )

        print(f"\nOML insert (sequential): {result.mean_time_us:.2f}μs")
        assert result.mean_time_us < 50, "Sequential insert should be <50μs"

    def test_order_query(self):
        """Benchmark order query."""
        oml = OrderMaintenanceList()
        elements = []

        # Build list
        for i in range(100):
            if not elements:
                elem = oml.insert_first(i)
            else:
                elem = oml.insert_after(elements[-1], i)
            elements.append(elem)

        # Query middle elements
        elem_a = elements[25]
        elem_b = elements[75]

        result = run_benchmark(
            "oml_query",
            lambda: oml.order(elem_a, elem_b),
            iterations=10000,
        )

        print(f"\nOML order query: {result.mean_time_ns:.0f}ns")
        assert result.mean_time_us < 1, "Query should be <1μs (O(1))"


class TestDependencyGraphBenchmarks:
    """Benchmarks for dependency graph operations."""

    def test_add_dependency(self):
        """Benchmark adding dependencies."""
        graph = DependencyGraph()
        counter = [0]

        def add_dep():
            graph.add_dependency(
                f"node_{counter[0]}",
                f"node_{counter[0] + 1}",
                DependencyKind.SYNTHESIS,
            )
            counter[0] += 1

        result = run_benchmark(
            "dep_add",
            add_dep,
            iterations=1000,
        )

        print(f"\nAdd dependency: {result.mean_time_us:.2f}μs")
        assert result.mean_time_us < 20, "Add dependency should be <20μs"

    def test_get_dependents(self):
        """Benchmark getting dependents."""
        graph = DependencyGraph()

        # Build a tree-like dependency structure
        for i in range(10):
            for j in range(10):
                graph.add_dependency(f"node_{i}", f"child_{i}_{j}", DependencyKind.SYNTHESIS)

        result = run_benchmark(
            "get_dependents",
            lambda: graph.dependents_of(f"node_5"),
            iterations=5000,
        )

        print(f"\nGet dependents (10 children): {result.mean_time_us:.2f}μs")
        assert result.mean_time_us < 10, "Get dependents should be <10μs"


class TestSubsumptionBenchmarks:
    """Benchmarks for subsumption checking."""

    def test_primitive_subsumption(self):
        """Benchmark primitive type subsumption."""
        result = run_benchmark(
            "subsume_primitive",
            lambda: subsumes(INT, INT),
            iterations=10000,
        )

        print(f"\nSubsume primitive: {result.mean_time_ns:.0f}ns")
        assert result.mean_time_us < 5, "Primitive subsumption should be <5μs"

    def test_function_subsumption(self):
        """Benchmark function type subsumption."""
        sub = FunctionType(params=[ANY], returns=INT)
        sup = FunctionType(params=[INT], returns=ANY)

        result = run_benchmark(
            "subsume_function",
            lambda: subsumes(sub, sup),
            iterations=5000,
        )

        print(f"\nSubsume function: {result.mean_time_us:.2f}μs")
        assert result.mean_time_us < 20, "Function subsumption should be <20μs"

    def test_union_subsumption(self):
        """Benchmark union type subsumption."""
        sub = INT
        sup = UnionType(members=frozenset([INT, STR, BOOL]))

        result = run_benchmark(
            "subsume_union",
            lambda: subsumes(sub, sup),
            iterations=5000,
        )

        print(f"\nSubsume into union: {result.mean_time_us:.2f}μs")
        assert result.mean_time_us < 20, "Union subsumption should be <20μs"


class TestTypeConstraintBenchmarks:
    """Benchmarks for TypeConstraint operations."""

    def test_constraint_meet(self):
        """Benchmark constraint meet operation."""
        c1 = type_expecting(INT)
        c2 = type_expecting(INT)

        result = run_benchmark(
            "constraint_meet",
            lambda: c1.meet(c2),
            iterations=10000,
        )

        print(f"\nConstraint meet: {result.mean_time_us:.2f}μs")
        assert result.mean_time_us < 20, "Constraint meet should be <20μs"

    def test_constraint_satisfiability(self):
        """Benchmark satisfiability check."""
        c = type_expecting(INT)

        result = run_benchmark(
            "satisfiability",
            lambda: c.satisfiability(),
            iterations=10000,
        )

        print(f"\nSatisfiability check: {result.mean_time_ns:.0f}ns")
        assert result.mean_time_us < 5, "Satisfiability should be <5μs"


class TestSynthesizerBenchmarks:
    """Benchmarks for type synthesis."""

    @pytest.fixture
    def env(self):
        env = TypeEnvironment()
        env = env.bind("x", INT)
        env = env.bind("y", STR)
        env = env.bind("f", FunctionType(params=[INT], returns=STR))
        return env

    def test_synthesize_literal(self, env):
        """Benchmark literal synthesis."""
        node = MarkedASTNode(
            kind=ASTNodeKind.LITERAL,
            span=SourceSpan(0, 2),
            data={"value": 42},
        )

        result = run_benchmark(
            "synth_literal",
            lambda: synthesize(node, env),
            iterations=10000,
        )

        print(f"\nSynthesize literal: {result.mean_time_ns:.0f}ns")
        assert result.mean_time_us < 50, "Literal synthesis should be <50μs"

    def test_synthesize_variable(self, env):
        """Benchmark variable synthesis."""
        node = MarkedASTNode(
            kind=ASTNodeKind.VARIABLE,
            span=SourceSpan(0, 1),
            data={"name": "x"},
        )

        result = run_benchmark(
            "synth_variable",
            lambda: synthesize(node, env),
            iterations=10000,
        )

        print(f"\nSynthesize variable: {result.mean_time_us:.2f}μs")
        assert result.mean_time_us < 50, "Variable synthesis should be <50μs"


class TestEndToEndTypeBenchmarks:
    """End-to-end type checking benchmarks."""

    def test_type_mask_target(self):
        """Test that type mask computation meets <500μs target."""
        # Simulate type mask computation workflow
        env = TypeEnvironment()
        for i in range(20):
            env = env.bind(f"var_{i}", INT if i % 2 == 0 else STR)

        expected = INT

        def compute_type_mask():
            # Check 100 tokens for type compatibility
            allowed = []
            for token in range(100):
                # Simulate inferring token type
                inferred = INT if token % 3 == 0 else STR
                if subsumes(inferred, expected):
                    allowed.append(token)
            return allowed

        result = run_benchmark(
            "type_mask_100_tokens",
            compute_type_mask,
            iterations=100,
        )

        print(f"\nType mask (100 tokens): {result.mean_time_us:.0f}μs")
        # This tests infrastructure, not full implementation
        assert result.mean_time_us < 500, "Type mask should be <500μs"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
