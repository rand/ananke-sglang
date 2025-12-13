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
"""Control flow domain for CFG-based reachability constraints.

This module provides:
- ControlFlowConstraint: Constraint on reachability and termination
- ControlFlowDomain: Domain for CFG-based constraint tracking
- CFGSketch: Lightweight control flow graph representation
- Reachability analysis utilities
"""

from __future__ import annotations

from .constraint import (
    ControlFlowConstraint,
    CodePoint,
    ReachabilityKind,
    TerminationRequirement,
    ReachabilityConstraintItem,
    CONTROLFLOW_TOP,
    CONTROLFLOW_BOTTOM,
    controlflow_requiring_reach,
    controlflow_forbidding_reach,
    controlflow_requiring_termination,
)
from .cfg import (
    BasicBlock,
    CFGEdge,
    CFGSketch,
    CFGBuilder,
    EdgeKind,
)
from .domain import (
    ControlFlowDomain,
    ControlFlowDomainCheckpoint,
)
from .reachability import (
    ReachabilityAnalyzer,
    ReachabilityResult,
    compute_dominators,
    compute_post_dominators,
    find_loop_bodies,
)

__all__ = [
    # Constraint types
    "ControlFlowConstraint",
    "CodePoint",
    "ReachabilityKind",
    "TerminationRequirement",
    "ReachabilityConstraintItem",
    "CONTROLFLOW_TOP",
    "CONTROLFLOW_BOTTOM",
    # Factory functions
    "controlflow_requiring_reach",
    "controlflow_forbidding_reach",
    "controlflow_requiring_termination",
    # CFG types
    "BasicBlock",
    "CFGEdge",
    "CFGSketch",
    "CFGBuilder",
    "EdgeKind",
    # Domain
    "ControlFlowDomain",
    "ControlFlowDomainCheckpoint",
    # Reachability analysis
    "ReachabilityAnalyzer",
    "ReachabilityResult",
    "compute_dominators",
    "compute_post_dominators",
    "find_loop_bodies",
]
