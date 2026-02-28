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
"""Property-based tests for propagation monotonicity using Hypothesis.

These tests verify that constraint propagation only refines constraints,
never loosens them. This property is critical for:

1. Fixpoint Convergence: If propagation can both refine and loosen,
   the fixpoint algorithm may oscillate indefinitely.

2. Soundness: Once a token is ruled out, it stays ruled out.

3. Progress Guarantee: Propagation always moves toward more precise constraints.

Monotonicity Property:
    After propagation:
        ∀ domain d: d.constraint ⊑ d.constraint_before

Where c₁ ⊑ c₂ means c₁.meet(c₂) == c₁ (c₁ is more restrictive than c₂)

References:
    - Abstract Interpretation: Cousot & Cousot
    - Dataflow Analysis: Kildall's algorithm convergence proof
"""

import pytest
from hypothesis import given, settings, assume, strategies as st
from typing import Dict, List, Tuple

from core.constraint import Constraint, TOP, BOTTOM, Satisfiability
from core.domain import GenerationContext, PassthroughDomain
from propagation.network import PropagationNetwork
from propagation.edges import FunctionEdge


# ============================================================================
# Test Constraint Implementation
# ============================================================================


class MonotonicConstraint(Constraint):
    """A constraint with a numeric level for testing monotonicity.

    Higher levels are MORE restrictive (lower in the lattice).
    Level 0 = TOP, Level 100 = effective BOTTOM.

    The meet operation takes the maximum level (more restrictive).
    """

    def __init__(self, level: int, is_bottom: bool = False):
        self._level = max(0, min(100, level))  # Clamp to 0-100
        self._is_bottom = is_bottom

    @property
    def level(self) -> int:
        return self._level

    def meet(self, other: "Constraint") -> "Constraint":
        if isinstance(other, MonotonicConstraint):
            # If either is bottom, result is bottom
            if self._is_bottom or other._is_bottom:
                return MonotonicConstraint(100, is_bottom=True)
            # Meet takes the more restrictive (higher) level
            return MonotonicConstraint(max(self._level, other._level))
        return other.meet(self)

    def is_top(self) -> bool:
        return self._level == 0 and not self._is_bottom

    def is_bottom(self) -> bool:
        return self._is_bottom

    def satisfiability(self) -> Satisfiability:
        if self._is_bottom:
            return Satisfiability.UNSAT
        return Satisfiability.SAT

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MonotonicConstraint):
            return self._level == other._level and self._is_bottom == other._is_bottom
        return False

    def __hash__(self) -> int:
        return hash((self._level, self._is_bottom))

    def __repr__(self) -> str:
        if self._is_bottom:
            return "MonotonicConstraint(BOTTOM)"
        return f"MonotonicConstraint(level={self._level})"

    def is_more_restrictive_than(self, other: "MonotonicConstraint") -> bool:
        """Check if self is more (or equally) restrictive than other."""
        if self._is_bottom:
            return True  # Bottom is most restrictive
        if other._is_bottom:
            return self._is_bottom  # Only bottom is as restrictive as bottom
        return self._level >= other._level


# Singleton instances
MONO_TOP = MonotonicConstraint(0)
MONO_BOTTOM = MonotonicConstraint(100, is_bottom=True)


# ============================================================================
# Hypothesis Strategies
# ============================================================================


@st.composite
def monotonic_constraint_strategy(draw):
    """Generate random MonotonicConstraint."""
    is_bottom = draw(st.booleans())
    if is_bottom:
        return MonotonicConstraint(100, is_bottom=True)
    level = draw(st.integers(min_value=0, max_value=100))
    return MonotonicConstraint(level)


@st.composite
def propagation_delta_strategy(draw):
    """Generate a propagation delta (amount to increase constraint level)."""
    # Delta should be non-negative for monotonic propagation
    return draw(st.integers(min_value=0, max_value=50))


@st.composite
def domain_names_strategy(draw):
    """Generate a list of domain names."""
    count = draw(st.integers(min_value=2, max_value=6))
    return [f"domain_{i}" for i in range(count)]


@st.composite
def edge_topology_strategy(draw, domain_names: List[str]):
    """Generate edges between domains."""
    edges = []
    # Generate some random edges
    num_edges = draw(st.integers(min_value=1, max_value=len(domain_names) * 2))
    for _ in range(num_edges):
        src = draw(st.sampled_from(domain_names))
        tgt = draw(st.sampled_from(domain_names))
        if src != tgt:
            delta = draw(propagation_delta_strategy())
            edges.append((src, tgt, delta))
    return edges


# ============================================================================
# Property Tests: Basic Monotonicity
# ============================================================================


class TestConstraintMonotonicity:
    """Tests for basic constraint monotonicity properties."""

    @given(c1=monotonic_constraint_strategy(), c2=monotonic_constraint_strategy())
    @settings(max_examples=200)
    def test_meet_is_monotonic(self, c1: MonotonicConstraint, c2: MonotonicConstraint):
        """Property: c1.meet(c2) is at least as restrictive as c1 and c2.

        This ensures that combining constraints only adds restrictions.
        """
        result = c1.meet(c2)
        assert isinstance(result, MonotonicConstraint)

        # Result should be at least as restrictive as both inputs
        assert result.is_more_restrictive_than(c1)
        assert result.is_more_restrictive_than(c2)

    @given(c=monotonic_constraint_strategy())
    @settings(max_examples=200)
    def test_meet_with_top_identity(self, c: MonotonicConstraint):
        """Property: c.meet(TOP) == c

        Meeting with TOP should not change the constraint.
        """
        result = c.meet(MONO_TOP)
        assert result == c

    @given(c=monotonic_constraint_strategy())
    @settings(max_examples=200)
    def test_meet_with_bottom_absorbing(self, c: MonotonicConstraint):
        """Property: c.meet(BOTTOM) == BOTTOM

        Meeting with BOTTOM should always produce BOTTOM.
        """
        result = c.meet(MONO_BOTTOM)
        assert result.is_bottom()

    @given(c=monotonic_constraint_strategy())
    @settings(max_examples=200)
    def test_meet_idempotent(self, c: MonotonicConstraint):
        """Property: c.meet(c) == c

        Meeting a constraint with itself should not change it.
        """
        result = c.meet(c)
        assert result == c


# ============================================================================
# Property Tests: Propagation Monotonicity
# ============================================================================


class TestPropagationMonotonicity:
    """Tests for propagation network monotonicity."""

    def create_monotonic_propagator(self, delta: int):
        """Create a propagator that increases constraint level by delta."""
        def propagate(
            source: Constraint,
            target: Constraint,
            ctx: GenerationContext
        ) -> Constraint:
            if not isinstance(source, MonotonicConstraint):
                return target
            if source.is_top():
                return target

            # Propagate: increase target's level based on source
            new_level = source.level + delta
            new_constraint = MonotonicConstraint(new_level)

            # Return meet of target and new constraint (more restrictive)
            return target.meet(new_constraint)

        return propagate

    @given(
        initial_levels=st.lists(
            st.integers(min_value=0, max_value=50),
            min_size=3,
            max_size=6
        ),
        propagation_deltas=st.lists(
            st.integers(min_value=0, max_value=20),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100)
    def test_propagation_only_refines(
        self,
        initial_levels: List[int],
        propagation_deltas: List[int]
    ):
        """Property: After propagation, all constraints are at least as restrictive.

        This is the core monotonicity property:
            ∀ domain d: d.constraint_after ⊒ d.constraint_before
        """
        # Setup network
        network = PropagationNetwork()
        context = GenerationContext(vocab_size=1000, position=0, device="cpu")

        domain_names = [f"domain_{i}" for i in range(len(initial_levels))]

        # Register domains with initial constraints
        for i, (name, level) in enumerate(zip(domain_names, initial_levels)):
            domain = PassthroughDomain(name, MONO_TOP, MONO_BOTTOM)
            network.register_domain(domain, MonotonicConstraint(level))

        # Add edges (chain topology for simplicity)
        for i in range(len(domain_names) - 1):
            delta = propagation_deltas[i % len(propagation_deltas)]
            edge = FunctionEdge(
                source=domain_names[i],
                target=domain_names[i + 1],
                priority=50,
                propagate_fn=self.create_monotonic_propagator(delta)
            )
            network.add_edge(edge)

        # Record constraints before propagation
        before = {}
        for name in domain_names:
            before[name] = network.get_constraint(name)

        # Propagate
        network.propagate(context)

        # Check monotonicity: all constraints should be at least as restrictive
        for name in domain_names:
            after = network.get_constraint(name)
            assert isinstance(after, MonotonicConstraint)
            before_c = before[name]
            assert isinstance(before_c, MonotonicConstraint)

            # After should be more (or equally) restrictive than before
            assert after.is_more_restrictive_than(before_c), (
                f"Monotonicity violated for {name}: "
                f"before={before_c}, after={after}"
            )

    @given(
        trigger_domain=st.integers(min_value=0, max_value=4),
        trigger_level=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100)
    def test_propagation_from_single_change(
        self,
        trigger_domain: int,
        trigger_level: int
    ):
        """Property: Updating one domain propagates monotonically to others."""
        # Setup 5-domain network
        network = PropagationNetwork()
        context = GenerationContext(vocab_size=1000, position=0, device="cpu")

        domain_names = ["syntax", "types", "imports", "controlflow", "semantics"]
        trigger_domain = trigger_domain % len(domain_names)

        # Register all domains with TOP
        for name in domain_names:
            domain = PassthroughDomain(name, MONO_TOP, MONO_BOTTOM)
            network.register_domain(domain, MONO_TOP)

        # Add chain of edges
        for i in range(len(domain_names) - 1):
            edge = FunctionEdge(
                source=domain_names[i],
                target=domain_names[i + 1],
                priority=50,
                propagate_fn=self.create_monotonic_propagator(1)
            )
            network.add_edge(edge)

        # Record initial state
        before = {name: network.get_constraint(name) for name in domain_names}

        # Trigger propagation by updating one domain
        network.set_constraint(
            domain_names[trigger_domain],
            MonotonicConstraint(trigger_level)
        )
        network.propagate(context)

        # Check monotonicity
        for name in domain_names:
            after = network.get_constraint(name)
            before_c = before[name]

            assert isinstance(after, MonotonicConstraint)
            assert isinstance(before_c, MonotonicConstraint)
            assert after.is_more_restrictive_than(before_c), (
                f"Monotonicity violated for {name}"
            )


# ============================================================================
# Property Tests: Fixpoint Properties
# ============================================================================


class TestFixpointProperties:
    """Tests for fixpoint convergence properties."""

    @given(iterations=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50)
    def test_repeated_propagation_converges(self, iterations: int):
        """Property: Multiple propagation calls should converge to same result.

        After the first propagation reaches fixpoint, subsequent calls
        should not change constraints.
        """
        # Setup network with cyclic topology
        network = PropagationNetwork()
        context = GenerationContext(vocab_size=1000, position=0, device="cpu")

        domain_names = ["a", "b", "c"]

        for name in domain_names:
            domain = PassthroughDomain(name, MONO_TOP, MONO_BOTTOM)
            network.register_domain(domain, MonotonicConstraint(10))

        # Create cycle: a -> b -> c -> a
        def bounded_propagator(
            source: Constraint,
            target: Constraint,
            ctx: GenerationContext
        ) -> Constraint:
            if not isinstance(source, MonotonicConstraint):
                return target
            if source.is_top() or source.level >= 50:
                return target
            return target.meet(MonotonicConstraint(source.level + 5))

        for i, src in enumerate(domain_names):
            tgt = domain_names[(i + 1) % len(domain_names)]
            edge = FunctionEdge(
                source=src,
                target=tgt,
                priority=50,
                propagate_fn=bounded_propagator
            )
            network.add_edge(edge)

        # First propagation
        network.propagate(context)
        after_first = {name: network.get_constraint(name) for name in domain_names}

        # Additional propagations
        for _ in range(iterations):
            network.propagate(context)

        # Should be same as after first propagation (fixpoint)
        for name in domain_names:
            current = network.get_constraint(name)
            first = after_first[name]
            assert current == first, (
                f"Fixpoint unstable for {name}: "
                f"after_first={first}, current={current}"
            )

    @given(st.data())
    @settings(max_examples=50)
    def test_propagation_terminates(self, data):
        """Property: Propagation should always terminate.

        With proper iteration limits and monotonicity, propagation
        should not run forever.
        """
        # Setup network
        network = PropagationNetwork()
        context = GenerationContext(vocab_size=1000, position=0, device="cpu")

        num_domains = data.draw(st.integers(min_value=3, max_value=8))
        domain_names = [f"d{i}" for i in range(num_domains)]

        for name in domain_names:
            domain = PassthroughDomain(name, MONO_TOP, MONO_BOTTOM)
            initial = MonotonicConstraint(data.draw(st.integers(0, 20)))
            network.register_domain(domain, initial)

        # Add random edges
        num_edges = data.draw(st.integers(1, num_domains * 2))
        for _ in range(num_edges):
            src = data.draw(st.sampled_from(domain_names))
            tgt = data.draw(st.sampled_from(domain_names))
            if src != tgt:
                delta = data.draw(st.integers(0, 10))

                def make_prop(d):
                    def prop(s, t, c):
                        if isinstance(s, MonotonicConstraint) and not s.is_top():
                            return t.meet(MonotonicConstraint(s.level + d))
                        return t
                    return prop

                edge = FunctionEdge(
                    source=src,
                    target=tgt,
                    priority=50,
                    propagate_fn=make_prop(delta)
                )
                network.add_edge(edge)

        # This should complete without hanging
        # The network has iteration limits built in
        network.propagate(context)

        # If we get here, propagation terminated
        assert True


# ============================================================================
# Property Tests: Ordering Properties
# ============================================================================


class TestOrderingProperties:
    """Tests for constraint ordering properties."""

    @given(
        c1=monotonic_constraint_strategy(),
        c2=monotonic_constraint_strategy(),
        c3=monotonic_constraint_strategy()
    )
    @settings(max_examples=200)
    def test_meet_associativity(
        self,
        c1: MonotonicConstraint,
        c2: MonotonicConstraint,
        c3: MonotonicConstraint
    ):
        """Property: (c1 ⊓ c2) ⊓ c3 = c1 ⊓ (c2 ⊓ c3)

        Meet is associative.
        """
        left = c1.meet(c2).meet(c3)
        right = c1.meet(c2.meet(c3))

        assert left == right

    @given(
        c1=monotonic_constraint_strategy(),
        c2=monotonic_constraint_strategy()
    )
    @settings(max_examples=200)
    def test_meet_commutativity(
        self,
        c1: MonotonicConstraint,
        c2: MonotonicConstraint
    ):
        """Property: c1 ⊓ c2 = c2 ⊓ c1

        Meet is commutative.
        """
        assert c1.meet(c2) == c2.meet(c1)

    @given(
        c1=monotonic_constraint_strategy(),
        c2=monotonic_constraint_strategy()
    )
    @settings(max_examples=200)
    def test_ordering_antisymmetric(
        self,
        c1: MonotonicConstraint,
        c2: MonotonicConstraint
    ):
        """Property: If c1 ⊑ c2 and c2 ⊑ c1, then c1 = c2

        The constraint ordering is antisymmetric.
        """
        # c1 ⊑ c2 means c1.meet(c2) == c1
        c1_leq_c2 = c1.meet(c2) == c1
        c2_leq_c1 = c2.meet(c1) == c2

        if c1_leq_c2 and c2_leq_c1:
            assert c1 == c2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
