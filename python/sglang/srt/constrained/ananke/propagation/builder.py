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
"""Builder for standard propagation networks.

Provides factory functions to create pre-configured propagation networks
with standard domains and edges.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from core.constraint import Constraint
from core.domain import ConstraintDomain

from .network import PropagationNetwork
from .edges import (
    PropagationEdge,
    create_standard_edges,
    SyntaxToTypesEdge,
    TypesToSyntaxEdge,
    TypesToImportsEdge,
    ImportsToTypesEdge,
)


class PropagationNetworkBuilder:
    """Builder for constructing propagation networks.

    Provides a fluent interface for configuring networks:

    Example:
        >>> network = (
        ...     PropagationNetworkBuilder()
        ...     .with_domain(syntax_domain)
        ...     .with_domain(type_domain)
        ...     .with_standard_edges()
        ...     .with_max_iterations(50)
        ...     .build()
        ... )
    """

    def __init__(self):
        """Initialize the builder."""
        self._domains: Dict[str, ConstraintDomain] = {}
        self._initial_constraints: Dict[str, Constraint] = {}
        self._edges: List[PropagationEdge] = []
        self._max_iterations: int = PropagationNetwork.DEFAULT_MAX_ITERATIONS

    def with_domain(
        self,
        domain: ConstraintDomain,
        initial_constraint: Optional[Constraint] = None,
    ) -> PropagationNetworkBuilder:
        """Add a domain to the network.

        Args:
            domain: The domain to add
            initial_constraint: Optional initial constraint

        Returns:
            Self for chaining
        """
        self._domains[domain.name] = domain
        if initial_constraint is not None:
            self._initial_constraints[domain.name] = initial_constraint
        return self

    def with_edge(self, edge: PropagationEdge) -> PropagationNetworkBuilder:
        """Add a propagation edge.

        Args:
            edge: The edge to add

        Returns:
            Self for chaining
        """
        self._edges.append(edge)
        return self

    def with_edges(self, edges: List[PropagationEdge]) -> PropagationNetworkBuilder:
        """Add multiple propagation edges.

        Args:
            edges: List of edges to add

        Returns:
            Self for chaining
        """
        self._edges.extend(edges)
        return self

    def with_standard_edges(self) -> PropagationNetworkBuilder:
        """Add the standard set of propagation edges.

        Returns:
            Self for chaining
        """
        self._edges.extend(create_standard_edges())
        return self

    def with_max_iterations(self, max_iterations: int) -> PropagationNetworkBuilder:
        """Set the maximum iteration limit.

        Args:
            max_iterations: Maximum iterations for propagation

        Returns:
            Self for chaining
        """
        self._max_iterations = max_iterations
        return self

    def build(self) -> PropagationNetwork:
        """Build the configured propagation network.

        Returns:
            The configured PropagationNetwork
        """
        network = PropagationNetwork(max_iterations=self._max_iterations)

        # Register domains
        for name, domain in self._domains.items():
            initial = self._initial_constraints.get(name)
            network.register_domain(domain, initial)

        # Add edges (filtering for registered domains)
        registered = set(self._domains.keys())
        for edge in self._edges:
            if edge.source in registered and edge.target in registered:
                network.add_edge(edge)

        return network


def build_standard_propagation_network(
    domains: List[ConstraintDomain],
    max_iterations: int = 100,
) -> PropagationNetwork:
    """Build a standard propagation network with given domains.

    Creates a network with standard edges connecting:
    - syntax <-> types
    - types <-> imports
    - controlflow -> semantics

    Only edges between registered domains are added.

    Args:
        domains: List of domains to include
        max_iterations: Maximum propagation iterations

    Returns:
        Configured PropagationNetwork
    """
    builder = PropagationNetworkBuilder()
    builder.with_max_iterations(max_iterations)
    builder.with_standard_edges()

    for domain in domains:
        builder.with_domain(domain)

    return builder.build()


def build_minimal_network(
    syntax_domain: Optional[ConstraintDomain] = None,
    type_domain: Optional[ConstraintDomain] = None,
    max_iterations: int = 100,
) -> PropagationNetwork:
    """Build a minimal network with just syntax and types.

    This is the most common configuration for basic
    type-constrained code generation.

    Args:
        syntax_domain: Optional syntax domain
        type_domain: Optional type domain
        max_iterations: Maximum propagation iterations

    Returns:
        Configured PropagationNetwork
    """
    builder = PropagationNetworkBuilder()
    builder.with_max_iterations(max_iterations)

    if syntax_domain is not None:
        builder.with_domain(syntax_domain)

    if type_domain is not None:
        builder.with_domain(type_domain)

    # Add bidirectional edges if both domains present
    if syntax_domain is not None and type_domain is not None:
        builder.with_edge(SyntaxToTypesEdge())
        builder.with_edge(TypesToSyntaxEdge())

    return builder.build()
