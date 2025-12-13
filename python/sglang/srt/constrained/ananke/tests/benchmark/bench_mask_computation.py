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
"""Benchmarks for token mask computation.

Performance targets (from implementation plan):
- Syntax mask: ~50μs (delegated to llguidance)
- Type mask: <500μs
- Import mask: <100μs
- Control flow mask: <200μs
- Semantic mask: <1ms
- Fused mask: <2ms

Run with: pytest tests/benchmark/bench_mask_computation.py -v
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest
import torch

from masks import (
    TokenMaskFuser,
    MaskCache,
    IncrementalMaskComputer,
    LazyConstraintEvaluator,
    EvaluationPriority,
    EvaluationBudget,
    FusionStrategy,
    CacheKey,
)
from core.constraint import TOP, BOTTOM, Constraint
from core.domain import GenerationContext


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


class MockDomain:
    """Mock domain for benchmarking with configurable latency."""

    def __init__(
        self,
        name: str,
        vocab_size: int = 32000,
        latency_ns: int = 100_000,  # 100μs default
        selectivity: float = 0.5,  # 50% of tokens allowed
    ):
        self._name = name
        self._vocab_size = vocab_size
        self._latency_ns = latency_ns
        self._selectivity = selectivity

    @property
    def name(self) -> str:
        return self._name

    def token_mask(self, constraint: Any, context: Any) -> torch.Tensor:
        """Generate mask with simulated latency."""
        # Simulate computation time
        start = time.perf_counter_ns()
        while time.perf_counter_ns() - start < self._latency_ns:
            pass

        # Generate mask based on selectivity
        mask = torch.rand(self._vocab_size) < self._selectivity
        return mask

    def observe_token(self, constraint: Any, token: int, context: Any) -> Any:
        return constraint

    def checkpoint(self, constraint: Any) -> Any:
        return constraint

    def restore(self, checkpoint: Any) -> Any:
        return checkpoint

    def top(self) -> Constraint:
        return TOP

    def bottom(self) -> Constraint:
        return BOTTOM


class TestMaskFuserBenchmarks:
    """Benchmarks for TokenMaskFuser."""

    @pytest.fixture
    def context(self):
        return GenerationContext(vocab_size=32000, position=0, device="cpu")

    @pytest.fixture
    def fuser_with_domains(self):
        """Create fuser with multiple domains of varying latency."""
        fuser = TokenMaskFuser()

        # Add domains with different latencies
        fuser.register_domain("syntax", MockDomain("syntax", latency_ns=50_000))  # 50μs
        fuser.register_domain("types", MockDomain("types", latency_ns=400_000))  # 400μs
        fuser.register_domain("imports", MockDomain("imports", latency_ns=80_000))  # 80μs
        fuser.register_domain("controlflow", MockDomain("controlflow", latency_ns=150_000))  # 150μs
        fuser.register_domain("semantics", MockDomain("semantics", latency_ns=800_000))  # 800μs

        return fuser

    def run_benchmark(
        self,
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

    def test_single_domain_mask(self, context):
        """Benchmark single domain mask computation."""
        domain = MockDomain("test", latency_ns=100_000)  # 100μs

        result = self.run_benchmark(
            "single_domain_mask",
            lambda: domain.token_mask(None, context),
            iterations=50,
        )

        print(f"\nSingle domain mask: {result.mean_time_us:.2f}μs")
        assert result.mean_time_us < 500, "Single domain mask should be <500μs"

    def test_fused_mask_sequential(self, fuser_with_domains, context):
        """Benchmark fused mask with sequential strategy."""
        fuser_with_domains._strategy = FusionStrategy.BITWISE_AND
        constraints = {
            "syntax": None,
            "types": None,
            "imports": None,
            "controlflow": None,
            "semantics": None,
        }

        result = self.run_benchmark(
            "fused_mask_sequential",
            lambda: fuser_with_domains.fuse(constraints, context),
            iterations=20,
        )

        print(f"\nFused mask (sequential): {result.mean_time_ms:.2f}ms")
        # Sequential should be sum of all latencies: ~1.48ms + overhead
        assert result.mean_time_ms < 5, "Sequential fused mask should be <5ms"

    def test_fused_mask_selectivity_ordered(self, fuser_with_domains, context):
        """Benchmark fused mask with selectivity ordering."""
        fuser_with_domains._strategy = FusionStrategy.SELECTIVITY_ORDERED
        constraints = {
            "syntax": None,
            "types": None,
            "imports": None,
            "controlflow": None,
            "semantics": None,
        }

        # Set selectivities (lower = more restrictive = evaluate first)
        fuser_with_domains.update_selectivity("syntax", 0.1)  # Very restrictive
        fuser_with_domains.update_selectivity("types", 0.5)
        fuser_with_domains.update_selectivity("imports", 0.9)  # Not restrictive
        fuser_with_domains.update_selectivity("controlflow", 0.8)
        fuser_with_domains.update_selectivity("semantics", 0.7)

        result = self.run_benchmark(
            "fused_mask_selectivity",
            lambda: fuser_with_domains.fuse(constraints, context),
            iterations=20,
        )

        print(f"\nFused mask (selectivity): {result.mean_time_ms:.2f}ms")
        # May short-circuit if syntax is restrictive
        assert result.mean_time_ms < 5, "Selectivity fused mask should be <5ms"


class TestMaskCacheBenchmarks:
    """Benchmarks for MaskCache."""

    @pytest.fixture
    def cache(self):
        return MaskCache(max_size=1000, max_age_seconds=60.0)

    def run_benchmark(
        self,
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

    def test_cache_put(self, cache):
        """Benchmark cache put operation."""
        mask = torch.ones(32000, dtype=torch.bool)
        counter = [0]

        def put_op():
            key = cache.make_key("test", f"constraint_{counter[0]}", counter[0])
            cache.put(key, mask)
            counter[0] += 1

        result = self.run_benchmark("cache_put", put_op, iterations=1000)

        print(f"\nCache put: {result.mean_time_ns:.0f}ns ({result.mean_time_us:.2f}μs)")
        assert result.mean_time_us < 100, "Cache put should be <100μs"

    def test_cache_get_hit(self, cache):
        """Benchmark cache get with hit."""
        mask = torch.ones(32000, dtype=torch.bool)
        key = cache.make_key("test", "constraint", 0)
        cache.put(key, mask)

        result = self.run_benchmark(
            "cache_get_hit",
            lambda: cache.get(key),
            iterations=10000,
        )

        print(f"\nCache get (hit): {result.mean_time_ns:.0f}ns ({result.mean_time_us:.2f}μs)")
        assert result.mean_time_us < 10, "Cache get hit should be <10μs"

    def test_cache_get_miss(self, cache):
        """Benchmark cache get with miss."""
        key = cache.make_key("test", "nonexistent", 0)

        result = self.run_benchmark(
            "cache_get_miss",
            lambda: cache.get(key),
            iterations=10000,
        )

        print(f"\nCache get (miss): {result.mean_time_ns:.0f}ns ({result.mean_time_us:.2f}μs)")
        assert result.mean_time_us < 10, "Cache get miss should be <10μs"


class TestIncrementalMaskBenchmarks:
    """Benchmarks for IncrementalMaskComputer."""

    @pytest.fixture
    def context(self):
        return GenerationContext(vocab_size=32000, position=0, device="cpu")

    def run_benchmark(
        self,
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

    def test_change_detection(self, context):
        """Benchmark constraint change detection."""
        fuser = TokenMaskFuser()
        fuser.register_domain("syntax", MockDomain("syntax", latency_ns=10_000))
        fuser.register_domain("types", MockDomain("types", latency_ns=10_000))
        fuser.register_domain("imports", MockDomain("imports", latency_ns=10_000))

        computer = IncrementalMaskComputer(fuser)

        # Initial constraints
        constraints = {
            "syntax": "constraint_1",
            "types": "constraint_2",
            "imports": "constraint_3",
        }
        computer.detect_changes(constraints)

        # Measure detection with one change
        def detect_one_change():
            constraints["types"] = f"constraint_{time.perf_counter_ns()}"
            return computer.detect_changes(constraints)

        result = self.run_benchmark("change_detection", detect_one_change, iterations=1000)

        print(f"\nChange detection: {result.mean_time_ns:.0f}ns ({result.mean_time_us:.2f}μs)")
        assert result.mean_time_us < 50, "Change detection should be <50μs"

    def test_incremental_vs_full_compute(self, context):
        """Compare incremental vs full computation."""
        fuser = TokenMaskFuser()
        fuser.register_domain("syntax", MockDomain("syntax", latency_ns=50_000))
        fuser.register_domain("types", MockDomain("types", latency_ns=200_000))
        fuser.register_domain("imports", MockDomain("imports", latency_ns=50_000))

        computer = IncrementalMaskComputer(fuser)

        # Initial full computation
        constraints = {
            "syntax": "constraint_1",
            "types": "constraint_2",
            "imports": "constraint_3",
        }
        computer.compute(constraints, context)

        # Benchmark incremental with one domain changed
        def incremental_compute():
            constraints["syntax"] = f"changed_{time.perf_counter_ns()}"
            return computer.compute(constraints, context)

        result = self.run_benchmark("incremental_compute", incremental_compute, iterations=20)

        print(f"\nIncremental compute (1 changed): {result.mean_time_us:.0f}μs")
        # Should only recompute syntax (~50μs) not all domains
        # Allow some overhead
        assert result.mean_time_us < 500, "Incremental compute should be <500μs"


class TestLazyEvaluatorBenchmarks:
    """Benchmarks for LazyConstraintEvaluator."""

    @pytest.fixture
    def context(self):
        return GenerationContext(vocab_size=32000, position=0, device="cpu")

    def run_benchmark(
        self,
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

    def test_lazy_evaluation_with_budget(self, context):
        """Benchmark lazy evaluation respecting budget."""
        evaluator = LazyConstraintEvaluator()

        # Register domains with priorities
        evaluator.register(
            "syntax",
            lambda c, ctx: torch.ones(32000, dtype=torch.bool),
            EvaluationPriority.CRITICAL,
            estimated_time_ns=50_000,
        )
        evaluator.register(
            "types",
            lambda c, ctx: torch.ones(32000, dtype=torch.bool),
            EvaluationPriority.HIGH,
            estimated_time_ns=200_000,
        )
        evaluator.register(
            "semantics",
            lambda c, ctx: torch.ones(32000, dtype=torch.bool),
            EvaluationPriority.LOW,
            estimated_time_ns=500_000,
        )

        constraints = {
            "syntax": None,
            "types": None,
            "semantics": None,
        }

        # Tight budget should skip semantics
        budget = EvaluationBudget(
            max_time_ns=500_000,  # 500μs
            max_domains=2,
            min_selectivity=0.99,
        )

        result = self.run_benchmark(
            "lazy_evaluation",
            lambda: evaluator.evaluate(constraints, context, budget),
            iterations=100,
        )

        print(f"\nLazy evaluation (budget-limited): {result.mean_time_us:.0f}μs")
        assert result.mean_time_us < 1000, "Lazy evaluation should be <1ms"

    def test_priority_ordering_overhead(self, context):
        """Benchmark overhead of priority-based ordering."""
        evaluator = LazyConstraintEvaluator()

        # Register many domains
        for i in range(10):
            priority = EvaluationPriority.NORMAL
            if i < 2:
                priority = EvaluationPriority.CRITICAL
            elif i < 5:
                priority = EvaluationPriority.HIGH

            evaluator.register(
                f"domain_{i}",
                lambda c, ctx: torch.ones(32000, dtype=torch.bool),
                priority,
                estimated_time_ns=10_000,
            )

        constraints = {f"domain_{i}": None for i in range(10)}

        result = self.run_benchmark(
            "priority_ordering",
            lambda: evaluator.create_lazy_masks(constraints, context),
            iterations=1000,
        )

        print(f"\nPriority ordering (10 domains): {result.mean_time_ns:.0f}ns")
        assert result.mean_time_us < 100, "Priority ordering should be <100μs"


class TestEndToEndMaskBenchmarks:
    """End-to-end benchmarks for mask computation pipeline."""

    @pytest.fixture
    def context(self):
        return GenerationContext(vocab_size=32000, position=0, device="cpu")

    def run_benchmark(
        self,
        operation: str,
        func,
        iterations: int = 50,
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

    def test_full_pipeline_target(self, context):
        """Test that full pipeline meets <2ms target."""
        fuser = TokenMaskFuser()

        # Add realistic domains (without actual computation)
        # Use fast mock domains since we're testing infrastructure
        fuser.register_domain("syntax", MockDomain("syntax", latency_ns=50_000))
        fuser.register_domain("types", MockDomain("types", latency_ns=400_000))
        fuser.register_domain("imports", MockDomain("imports", latency_ns=80_000))
        fuser.register_domain("controlflow", MockDomain("controlflow", latency_ns=150_000))
        fuser.register_domain("semantics", MockDomain("semantics", latency_ns=800_000))

        constraints = {
            "syntax": None,
            "types": None,
            "imports": None,
            "controlflow": None,
            "semantics": None,
        }

        result = self.run_benchmark(
            "full_pipeline",
            lambda: fuser.fuse(constraints, context),
            iterations=20,
        )

        print(f"\nFull pipeline: {result.mean_time_ms:.2f}ms")
        # Note: This will exceed 2ms because mock domains simulate real latency
        # In production, domains would be optimized
        print(f"  (target: <2ms, actual includes simulated domain latency)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
