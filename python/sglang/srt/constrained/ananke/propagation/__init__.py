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
"""Cross-domain constraint propagation network.

This package provides the infrastructure for propagating constraints
between domains (syntax, types, imports, control flow, semantics).

Key components:
- PropagationNetwork: Manages domain registration and constraint flow
- PropagationEdge: Defines how constraints propagate between domains
- PriorityWorklist: Schedules domain processing order
- PropagationNetworkBuilder: Fluent builder for networks
"""

from .network import PropagationNetwork, PropagationResult, DomainState
from .edges import (
    PropagationEdge,
    FunctionEdge,
    IdentityEdge,
    MeetEdge,
    SyntaxToTypesEdge,
    TypesToSyntaxEdge,
    TypesToImportsEdge,
    ImportsToTypesEdge,
    ControlFlowToSemanticsEdge,
    create_standard_edges,
)
from .worklist import (
    PriorityWorklist,
    FIFOWorklist,
    IterationLimiter,
    FixpointResult,
)
from .builder import (
    PropagationNetworkBuilder,
    build_standard_propagation_network,
    build_minimal_network,
)


__all__ = [
    # Network
    "PropagationNetwork",
    "PropagationResult",
    "DomainState",
    # Edges
    "PropagationEdge",
    "FunctionEdge",
    "IdentityEdge",
    "MeetEdge",
    "SyntaxToTypesEdge",
    "TypesToSyntaxEdge",
    "TypesToImportsEdge",
    "ImportsToTypesEdge",
    "ControlFlowToSemanticsEdge",
    "create_standard_edges",
    # Worklist
    "PriorityWorklist",
    "FIFOWorklist",
    "IterationLimiter",
    "FixpointResult",
    # Builder
    "PropagationNetworkBuilder",
    "build_standard_propagation_network",
    "build_minimal_network",
]
