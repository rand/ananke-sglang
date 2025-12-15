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
"""Adaptive constraint intensity module.

This module provides adaptive constraint intensity selection based on task
complexity assessment, optimizing the trade-off between constraint overhead
and code quality improvement.

Key insight: Simple tasks (~90% valid unconstrained) don't benefit from
full constraint checking (~2.3ms/token overhead), while complex tasks
(functions, classes, high-temp) see significant quality gains.

The solution is adaptive intensity:
- NONE: No constraints for trivial completions
- SYNTAX_ONLY: Just CFG (~50μs) for simple expressions
- STANDARD: Syntax + Types (~550μs) for most tasks
- FULL: All domains (~2.3ms) for complex code
- EXHAUSTIVE: Full + SMT semantics (~3ms+) for verified code
"""

from .intensity import (
    ConstraintIntensity,
    TaskComplexityAssessor,
    IntensityConfig,
    assess_complexity,
    domains_for_intensity,
)

__all__ = [
    "ConstraintIntensity",
    "TaskComplexityAssessor",
    "IntensityConfig",
    "assess_complexity",
    "domains_for_intensity",
]
