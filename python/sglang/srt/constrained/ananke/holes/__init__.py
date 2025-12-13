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
"""Typed hole management with Hazel-style fill-and-resume evaluation.

This module provides typed holes following the Hazel research program:
- Holes represent incomplete portions of code during generation
- Every partial program with holes has a well-defined type
- Fill-and-resume enables backtracking and exploration

References:
- Live Functional Programming with Typed Holes (ICFP 2019)
- Total Type Error Localization and Recovery with Holes (POPL 2024)
- Statically Contextualizing LLMs with Typed Holes (OOPSLA 2024)
"""

from __future__ import annotations

from .hole import (
    Hole,
    HoleId,
    HoleGranularity,
    HoleState,
    SourceLocation,
    TypeEnvironment,
    create_hole,
)
from .registry import (
    HoleRegistry,
    RegistryCheckpoint,
    create_registry,
)
from .closure import (
    HoleClosure,
    FilledClosure,
    Continuation,
    ContinuationKind,
    ClosureManager,
    create_closure,
)
from .environment_capture import (
    CapturedBinding,
    ScopeInfo,
    EnvironmentCapture,
    EnvironmentCapturer,
    capture_environment,
    merge_captures,
)
from .factory import (
    HoleSpec,
    HoleFactory,
    create_factory,
)
from .fill_resume import (
    EvaluationState,
    EvaluationResult,
    FillResult,
    FillAndResumeEngine,
    create_fill_resume_engine,
)
from .strategy import (
    HoleSelectionStrategy,
    DepthFirstStrategy,
    BreadthFirstStrategy,
    SourceOrderStrategy,
    TypeGuidedStrategy,
    PriorityStrategy,
    GranularityStrategy,
    CompositeStrategy,
    depth_first,
    breadth_first,
    source_order,
    type_guided,
    priority_based,
    granularity_based,
    default_strategy,
)

__all__ = [
    # Core hole types
    "Hole",
    "HoleId",
    "HoleGranularity",
    "HoleState",
    "SourceLocation",
    "TypeEnvironment",
    "create_hole",
    # Registry
    "HoleRegistry",
    "RegistryCheckpoint",
    "create_registry",
    # Closures
    "HoleClosure",
    "FilledClosure",
    "Continuation",
    "ContinuationKind",
    "ClosureManager",
    "create_closure",
    # Environment capture
    "CapturedBinding",
    "ScopeInfo",
    "EnvironmentCapture",
    "EnvironmentCapturer",
    "capture_environment",
    "merge_captures",
    # Factory
    "HoleSpec",
    "HoleFactory",
    "create_factory",
    # Fill-and-resume
    "EvaluationState",
    "EvaluationResult",
    "FillResult",
    "FillAndResumeEngine",
    "create_fill_resume_engine",
    # Strategies
    "HoleSelectionStrategy",
    "DepthFirstStrategy",
    "BreadthFirstStrategy",
    "SourceOrderStrategy",
    "TypeGuidedStrategy",
    "PriorityStrategy",
    "GranularityStrategy",
    "CompositeStrategy",
    "depth_first",
    "breadth_first",
    "source_order",
    "type_guided",
    "priority_based",
    "granularity_based",
    "default_strategy",
]
