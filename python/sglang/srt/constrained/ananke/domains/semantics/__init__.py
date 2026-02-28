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
"""Semantics domain for SMT-based semantic constraints via Z3.

This module provides:
- SemanticConstraint: Constraint on semantic properties
- SemanticDomain: Domain for SMT-based constraint tracking
- SMT solver integration (Z3 when available)
- Formula extraction from code (assertions, contracts)
"""

from __future__ import annotations

from .constraint import (
    SemanticConstraint,
    SMTFormula,
    FormulaKind,
    SEMANTIC_TOP,
    SEMANTIC_BOTTOM,
    semantic_assertion,
    semantic_precondition,
    semantic_postcondition,
)
from .domain import (
    SemanticDomain,
    SemanticDomainCheckpoint,
    VariableBounds,
    ExpressionState,
    ExpressionContext,
    ContextConfidence,
    BoundsConfidence,
    BlockingLevel,
)
from .smt import (
    SMTSolver,
    SimpleSMTSolver,
    IncrementalSMTSolver,
    SMTResult,
    SMTModel,
    SMTCheckResult,
    is_z3_available,
    create_smt_solver,
)
from .extractors import (
    FormulaExtractor,
    PythonAssertExtractor,
    DocstringContractExtractor,
    TypeAnnotationExtractor,
    CompositeExtractor,
    ExtractionResult,
    extract_formulas,
    extract_assertions,
    extract_contracts,
)

__all__ = [
    # Constraint types
    "SemanticConstraint",
    "SMTFormula",
    "FormulaKind",
    "SEMANTIC_TOP",
    "SEMANTIC_BOTTOM",
    # Factory functions
    "semantic_assertion",
    "semantic_precondition",
    "semantic_postcondition",
    # Domain
    "SemanticDomain",
    "SemanticDomainCheckpoint",
    "VariableBounds",
    "ExpressionState",
    "ExpressionContext",
    "ContextConfidence",
    "BoundsConfidence",
    "BlockingLevel",
    # SMT solver
    "SMTSolver",
    "SimpleSMTSolver",
    "IncrementalSMTSolver",
    "SMTResult",
    "SMTModel",
    "SMTCheckResult",
    "is_z3_available",
    "create_smt_solver",
    # Extractors
    "FormulaExtractor",
    "PythonAssertExtractor",
    "DocstringContractExtractor",
    "TypeAnnotationExtractor",
    "CompositeExtractor",
    "ExtractionResult",
    "extract_formulas",
    "extract_assertions",
    "extract_contracts",
]
