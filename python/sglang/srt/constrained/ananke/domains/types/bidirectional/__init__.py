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
"""Bidirectional type checking with synthesis and analysis.

Bidirectional typing combines two directions of type information flow:
- Synthesis (↑): Types flow up from leaves to root
- Analysis (↓): Types flow down from context to expression

This enables:
- More expressions to type check (e.g., lambdas need context)
- Better error messages (context provides expected types)
- Type-directed completion (holes know their expected type)

References:
    - Pierce & Turner (2000). "Local Type Inference"
    - Dunfield & Krishnaswami (2019). "Bidirectional Typing"
    - POPL 2024: "Total Type Error Localization and Recovery with Holes"
"""

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from .synthesis import (
        synthesize,
        SynthesisResult,
    )
    from .analysis import (
        analyze,
        AnalysisResult,
    )
    from .subsumption import (
        subsumes,
        check_subsumption,
    )
except ImportError:
    from domains.types.bidirectional.synthesis import (
        synthesize,
        SynthesisResult,
    )
    from domains.types.bidirectional.analysis import (
        analyze,
        AnalysisResult,
    )
    from domains.types.bidirectional.subsumption import (
        subsumes,
        check_subsumption,
    )

__all__ = [
    # Synthesis
    "synthesize",
    "SynthesisResult",
    # Analysis
    "analyze",
    "AnalysisResult",
    # Subsumption
    "subsumes",
    "check_subsumption",
]
