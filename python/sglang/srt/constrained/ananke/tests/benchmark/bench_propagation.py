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
"""Benchmarks for constraint propagation.

Performance targets (from implementation plan):
- Constraint propagation: <3ms (full network)
- Token observation: <5ms (including AST update)

Run with: pytest tests/benchmark/bench_propagation.py -v -s
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from core.constraint import Constraint, TOP, BOTTOM, Satisfiability
from core.domain import GenerationContext, PassthroughDomain
from propagation.network import PropagationNetwork
from propagation.edges import PropagationEdge, FunctionEdge
from propagation.worklist import PriorityWorklist


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
    iterations: int = 100,
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


class MockConstraint(Constraint):
    """Mock constraint for benchmarking."""

    def __init__(self, value: int = 0, is_bottom_flag: bool = False):
        self._value = value
        self._is_bottom = is_bottom_flag

    def meet(self, other: "Constraint") -> "Constraint":
        if isinstance(other, MockConstraint):
            if self._is_bottom or other._is_bottom:
                return MockConstraint(0, is_bottom_flag=True)
            return MockConstraint(max(self._value, other._value))
        return other.meet(self)

    def is_top(self) -> bool:
        return self._value == 0 and not self._is_bottom

    def is_bottom(self) -> bool:
        return self._is_bottom

    def satisfiability(self) -> Satisfiability:
        if self._is_bottom:
            return Satisfiability.UNSAT
        return Satisfiability.SAT

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MockConstraint):
            return self._value == other._value and self._is_bottom == other._is_bottom
        return False

    def __hash__(self) -> int:
        return hash((self._value, self._is_bottom))


# Singleton instances for MockConstraint
MOCK_TOP = MockConstraint(0)
MOCK_BOTTOM = MockConstraint(0, is_bottom_flag=True)


class TestWorklistBenchmarks:
    """Benchmarks for worklist operations."""

    def test_worklist_add(self):
        """Benchmark worklist add operation."""
        worklist = PriorityWorklist()
        counter = [0]

        def add_item():
            worklist.add(f"item_{counter[0]}", priority=50)
            counter[0] += 1

        result = run_benchmark("worklist_add", add_item, iterations=10000)

        print(f"\nWorklist add: {result.mean_time_ns:.0f}ns")
        assert result.mean_time_us < 10, "Worklist add should be <10μs"

    def test_worklist_pop(self):
        """Benchmark worklist pop operation."""
        worklist = PriorityWorklist()

        # Fill worklist
        for i in range(10000):
            worklist.add(f"item_{i}", priority=50)

        result = run_benchmark(
            "worklist_pop",
            lambda: worklist.pop() if not worklist.is_empty() else None,
            iterations=10000,
        )

        print(f"\nWorklist pop: {result.mean_time_ns:.0f}ns")
        assert result.mean_time_us < 10, "Worklist pop should be <10μs"

    def test_worklist_priority_ordering(self):
        """Benchmark worklist with mixed priorities."""
        worklist = PriorityWorklist()
        counter = [0]

        def push_mixed():
            base = counter[0]
            worklist.add(f"crit_{base}", priority=10)
            worklist.add(f"high_{base}", priority=25)
            worklist.add(f"norm_{base}", priority=50)
            worklist.add(f"low_{base}", priority=100)
            counter[0] += 1
            # Pop 4 items
            for _ in range(4):
                if not worklist.is_empty():
                    worklist.pop()

        result = run_benchmark("worklist_priority", push_mixed, iterations=1000)

        print(f"\nWorklist push+pop 4 items (mixed priority): {result.mean_time_us:.2f}μs")
        assert result.mean_time_us < 100, "Priority operations should be <100μs"


class TestPropagationEdgeBenchmarks:
    """Benchmarks for propagation edge operations."""

    def test_edge_application(self):
        """Benchmark edge propagation function."""

        def simple_propagate(source: Constraint, target: Constraint, ctx: GenerationContext) -> Constraint:
            if source.is_top():
                return target
            return target.meet(MockConstraint(1))

        edge = FunctionEdge(
            source="syntax",
            target="types",
            priority=50,
            propagate_fn=simple_propagate,
        )

        source_constraint = MockConstraint(5)
        target_constraint = MockConstraint(0)
        context = GenerationContext(vocab_size=32000, position=0, device="cpu")

        result = run_benchmark(
            "edge_apply",
            lambda: edge.propagate(source_constraint, target_constraint, context),
            iterations=10000,
        )

        print(f"\nEdge application: {result.mean_time_ns:.0f}ns")
        assert result.mean_time_us < 10, "Edge application should be <10μs"


class TestPropagationNetworkBenchmarks:
    """Benchmarks for PropagationNetwork."""

    @pytest.fixture
    def context(self):
        return GenerationContext(vocab_size=32000, position=0, device="cpu")

    @pytest.fixture
    def network(self):
        """Create network with 5 domains and standard edges."""
        network = PropagationNetwork()

        # Register 5 domains with passthrough domains
        for domain_name in ["syntax", "types", "imports", "controlflow", "semantics"]:
            domain = PassthroughDomain(domain_name, MOCK_TOP, MOCK_BOTTOM)
            network.register_domain(domain, MockConstraint(0))

        # Add some edges
        def make_propagator(delta: int):
            def propagate(source: Constraint, target: Constraint, ctx: GenerationContext) -> Constraint:
                if isinstance(source, MockConstraint) and not source.is_top():
                    return target.meet(MockConstraint(source._value + delta))
                return target

            return propagate

        # syntax -> types
        network.add_edge(
            FunctionEdge("syntax", "types", priority=25, propagate_fn=make_propagator(1))
        )
        # types -> imports
        network.add_edge(
            FunctionEdge("types", "imports", priority=50, propagate_fn=make_propagator(1))
        )
        # imports -> controlflow
        network.add_edge(
            FunctionEdge("imports", "controlflow", priority=50, propagate_fn=make_propagator(1))
        )
        # controlflow -> semantics
        network.add_edge(
            FunctionEdge("controlflow", "semantics", priority=100, propagate_fn=make_propagator(1))
        )

        return network

    def test_propagation_single_change(self, network, context):
        """Benchmark propagation from single constraint change."""
        counter = [0]

        def propagate_from_syntax():
            network.set_constraint("syntax", MockConstraint(counter[0] % 100))
            counter[0] += 1
            network.propagate(context)

        result = run_benchmark("propagate_single", propagate_from_syntax, iterations=500)

        print(f"\nPropagate from single change: {result.mean_time_us:.0f}μs")
        assert result.mean_time_us < 1000, "Single propagation should be <1ms"

    def test_propagation_multiple_changes(self, network, context):
        """Benchmark propagation from multiple constraint changes."""
        counter = [0]

        def propagate_multiple():
            t = counter[0]
            network.set_constraint("syntax", MockConstraint(t % 50))
            network.set_constraint("types", MockConstraint((t + 1) % 50))
            network.set_constraint("imports", MockConstraint((t + 2) % 50))
            counter[0] += 1
            network.propagate(context)

        result = run_benchmark("propagate_multiple", propagate_multiple, iterations=500)

        print(f"\nPropagate from 3 changes: {result.mean_time_us:.0f}μs")
        assert result.mean_time_us < 2000, "Multiple propagation should be <2ms"

    def test_propagation_fixpoint_convergence(self, network, context):
        """Benchmark time to reach fixpoint."""
        counter = [0]

        # Add cyclic edge to create potential for multiple iterations
        def cycle_propagate(source: Constraint, target: Constraint, ctx: GenerationContext) -> Constraint:
            if isinstance(source, MockConstraint) and source._value < 10:
                return target.meet(MockConstraint(source._value + 1))
            return target

        network.add_edge(
            FunctionEdge("semantics", "syntax", priority=100, propagate_fn=cycle_propagate)
        )

        def propagate_to_fixpoint():
            network.set_constraint("syntax", MockConstraint(counter[0] % 10))
            counter[0] += 1
            network.propagate(context)

        result = run_benchmark("propagate_fixpoint", propagate_to_fixpoint, iterations=100)

        print(f"\nPropagate to fixpoint: {result.mean_time_us:.0f}μs")
        assert result.mean_time_ms < 10, "Fixpoint convergence should be <10ms"


class TestNetworkScalabilityBenchmarks:
    """Benchmarks for network scalability."""

    def test_large_network(self):
        """Benchmark propagation in a large network."""
        network = PropagationNetwork()
        context = GenerationContext(vocab_size=32000, position=0, device="cpu")

        # Create 20 domains
        domain_count = 20
        for i in range(domain_count):
            domain = PassthroughDomain(f"domain_{i}", MOCK_TOP, MOCK_BOTTOM)
            network.register_domain(domain, MockConstraint(0))

        # Create chain of edges with proper function signature
        def make_chain_propagator():
            def propagate(source: Constraint, target: Constraint, ctx: GenerationContext) -> Constraint:
                if not source.is_top():
                    return target.meet(MockConstraint(1))
                return target
            return propagate

        for i in range(domain_count - 1):
            network.add_edge(
                FunctionEdge(
                    f"domain_{i}",
                    f"domain_{i+1}",
                    priority=50,
                    propagate_fn=make_chain_propagator(),
                )
            )

        def propagate_chain():
            network.set_constraint("domain_0", MockConstraint(1))
            network.propagate(context)

        result = run_benchmark("propagate_20_domains", propagate_chain, iterations=100)

        print(f"\nPropagate through 20 domains: {result.mean_time_us:.0f}μs")
        assert result.mean_time_ms < 3, "Large network propagation should be <3ms"

    def test_dense_network(self):
        """Benchmark propagation in a densely connected network."""
        network = PropagationNetwork()
        context = GenerationContext(vocab_size=32000, position=0, device="cpu")

        # Create 10 domains with dense connections
        domain_count = 10
        for i in range(domain_count):
            domain = PassthroughDomain(f"domain_{i}", MOCK_TOP, MOCK_BOTTOM)
            network.register_domain(domain, MockConstraint(0))

        # No-op propagator
        def noop_propagate(source: Constraint, target: Constraint, ctx: GenerationContext) -> Constraint:
            return target

        # Connect each domain to every other domain
        edge_count = 0
        for i in range(domain_count):
            for j in range(domain_count):
                if i != j:
                    network.add_edge(
                        FunctionEdge(
                            f"domain_{i}",
                            f"domain_{j}",
                            priority=50,
                            propagate_fn=noop_propagate,
                        )
                    )
                    edge_count += 1

        print(f"\nDense network: {domain_count} domains, {edge_count} edges")

        def propagate_dense():
            network.set_constraint("domain_0", MockConstraint(1))
            network.propagate(context)

        result = run_benchmark("propagate_dense", propagate_dense, iterations=100)

        print(f"Propagate in dense network: {result.mean_time_us:.0f}μs")
        assert result.mean_time_ms < 5, "Dense network propagation should be <5ms"


class TestEndToEndPropagationBenchmarks:
    """End-to-end propagation benchmarks."""

    def test_full_propagation_target(self):
        """Test that full propagation meets <3ms target."""
        network = PropagationNetwork()
        context = GenerationContext(vocab_size=32000, position=0, device="cpu")

        # Standard 5-domain setup
        domain_names = ["syntax", "types", "imports", "controlflow", "semantics"]
        for domain_name in domain_names:
            domain = PassthroughDomain(domain_name, MOCK_TOP, MOCK_BOTTOM)
            network.register_domain(domain, MockConstraint(0))

        # Standard edges
        def make_propagator():
            def propagate(source: Constraint, target: Constraint, ctx: GenerationContext) -> Constraint:
                if not source.is_top():
                    return target.meet(MockConstraint(1))
                return target

            return propagate

        edges = [
            ("syntax", "types"),
            ("types", "syntax"),
            ("types", "imports"),
            ("imports", "types"),
            ("types", "controlflow"),
            ("controlflow", "types"),
            ("controlflow", "semantics"),
            ("semantics", "controlflow"),
        ]

        for src, tgt in edges:
            network.add_edge(
                FunctionEdge(src, tgt, priority=50, propagate_fn=make_propagator())
            )

        def full_propagation():
            # Simulate token observation triggering propagation
            network.set_constraint("syntax", MockConstraint(time.perf_counter_ns() % 100))
            network.propagate(context)

        result = run_benchmark("full_propagation", full_propagation, iterations=100)

        print(f"\nFull propagation (5 domains, 8 edges): {result.mean_time_us:.0f}μs")
        print(f"  Target: <3ms")
        assert result.mean_time_ms < 3, "Full propagation should be <3ms"

    def test_token_observation_workflow(self):
        """Benchmark complete token observation workflow."""
        network = PropagationNetwork()
        context = GenerationContext(vocab_size=32000, position=0, device="cpu")

        # Setup network
        for domain_name in ["syntax", "types", "imports"]:
            domain = PassthroughDomain(domain_name, MOCK_TOP, MOCK_BOTTOM)
            network.register_domain(domain, MockConstraint(0))

        def syntax_to_types(source: Constraint, target: Constraint, ctx: GenerationContext) -> Constraint:
            if not source.is_top():
                return target.meet(MockConstraint(1))
            return target

        def types_to_imports(source: Constraint, target: Constraint, ctx: GenerationContext) -> Constraint:
            if not source.is_top():
                return target.meet(MockConstraint(1))
            return target

        network.add_edge(
            FunctionEdge(
                "syntax",
                "types",
                priority=25,
                propagate_fn=syntax_to_types,
            )
        )
        network.add_edge(
            FunctionEdge(
                "types",
                "imports",
                priority=50,
                propagate_fn=types_to_imports,
            )
        )

        def token_observation():
            # 1. Update syntax constraint (simulated)
            network.set_constraint("syntax", MockConstraint(1))

            # 2. Propagate
            network.propagate(context)

            # 3. Get all constraints (simulated mask computation would follow)
            for domain in ["syntax", "types", "imports"]:
                _ = network.get_constraint(domain)

        result = run_benchmark("token_observation", token_observation, iterations=500)

        print(f"\nToken observation workflow: {result.mean_time_us:.0f}μs")
        print(f"  Target: <5ms (including AST update)")
        # This tests propagation only, not AST update
        assert result.mean_time_us < 1000, "Token observation should be <1ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
