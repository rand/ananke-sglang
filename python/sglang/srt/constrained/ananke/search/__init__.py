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
"""Search strategies for code generation with constraints.

This module provides search algorithms that leverage Ananke constraints
for test-time compute optimization:

1. SudokuStyleHoleFiller: Fill typed holes using constraint propagation
   and the Most Constrained Variable (MCV) heuristic, with backtracking.

2. Trajectory tracking for efficient rollback during search.

Key Insight (from CSP research):
    Code generation with typed holes is a Constraint Satisfaction Problem.
    Apply Sudoku-solving techniques:
    - Most Constrained First: Fill holes with fewest valid options first
    - Constraint Propagation: After each fill, propagate to other holes
    - Backtracking: If stuck, backtrack and try alternatives

References:
- GenCP: LLM Meets Constraint Propagation (arXiv 2024)
- ROCODE: Backtracking for Code Generation (EMNLP 2024)
- AdapTrack: Dynamic Backtracking (arXiv 2024)
"""

import sys
from pathlib import Path

# Handle imports for both package and standalone usage
_SEARCH_DIR = Path(__file__).parent
_ANANKE_ROOT = _SEARCH_DIR.parent
if str(_ANANKE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANANKE_ROOT))

from search.sudoku_filler import (
    SudokuStyleHoleFiller,
    FillResult,
    FillStrategy,
    FillCandidate,
    HoledCode,
    FillGenerator,
    ConstraintChecker,
    fill_with_mcv_heuristic,
)
from search.trajectory import (
    Trajectory,
    TrajectoryNode,
    TrajectoryTrie,
    create_trajectory_trie,
)
from search.generators import (
    TypeAwareFillGenerator,
    UnifiedConstraintChecker,
    TypeConstraintInferencer,
    create_fill_generator,
    create_constraint_checker,
    create_constraint_inferencer,
)
from search.beam import (
    BeamCandidate,
    BeamSearchConfig,
    BeamSearchStats,
    TokenScorer,
    ConstraintScorer,
    BeamSearch,
    SimpleTokenScorer,
    create_beam_search,
)

__all__ = [
    # Sudoku-style filler
    "SudokuStyleHoleFiller",
    "FillResult",
    "FillStrategy",
    "FillCandidate",
    "HoledCode",
    "FillGenerator",
    "ConstraintChecker",
    "fill_with_mcv_heuristic",
    # Trajectory tracking
    "Trajectory",
    "TrajectoryNode",
    "TrajectoryTrie",
    "create_trajectory_trie",
    # Concrete implementations
    "TypeAwareFillGenerator",
    "UnifiedConstraintChecker",
    "TypeConstraintInferencer",
    "create_fill_generator",
    "create_constraint_checker",
    "create_constraint_inferencer",
    # Beam search
    "BeamCandidate",
    "BeamSearchConfig",
    "BeamSearchStats",
    "TokenScorer",
    "ConstraintScorer",
    "BeamSearch",
    "SimpleTokenScorer",
    "create_beam_search",
]
