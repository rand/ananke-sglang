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
"""Verification module for Best-of-N sampling with Ananke constraints.

This module provides post-hoc verification of generated code using Ananke's
constraint domains as verifiers instead of per-token masking.

Key Insight:
    Instead of constraining every token (~2.3ms/token overhead), we can:
    1. Generate N candidates unconstrained (faster)
    2. Verify each candidate against all constraints
    3. Select the best candidate by constraint satisfaction score

Benefits:
    - Zero per-token overhead for generation
    - Better quality than unconstrained (verification catches errors)
    - Scalable compute: use more candidates for harder tasks
    - Graceful degradation: return best even if none perfect

Soundness:
    Post-hoc verification is inherently SOUND since we never block tokens
    during generation. We only score and rank candidates after generation.
"""

from .verifier import (
    ConstraintVerifier,
    VerificationResult,
    DomainScore,
)
from .selector import (
    BestOfNSelector,
    SelectionStrategy,
    SelectionResult,
)
from .soft_verifier import (
    SoftVerifier,
    SoftScore,
    GenerationContext,
    NullSoftVerifier,
    EnsembleSoftVerifier,
    HeuristicSoftVerifier,
    CombinedVerificationResult,
    create_soft_verifier,
)
from .execution import (
    ExecutionVerifier,
    ExecutionResult,
    ExecutionScore,
    TestCase,
    StaticExecutionVerifier,
    RestrictedExecutionVerifier,
    create_execution_verifier,
    generate_test_cases_from_signature,
)
from .collaborative import (
    CollaborativeVerifier,
    CollaborativeResult,
    VerificationStrategy,
    StrategyScore,
    collaborative_verify,
)

__all__ = [
    # Rule-based verification
    "ConstraintVerifier",
    "VerificationResult",
    "DomainScore",
    # Best-of-N selection
    "BestOfNSelector",
    "SelectionStrategy",
    "SelectionResult",
    # Soft verification (PRMs)
    "SoftVerifier",
    "SoftScore",
    "GenerationContext",
    "NullSoftVerifier",
    "EnsembleSoftVerifier",
    "HeuristicSoftVerifier",
    "CombinedVerificationResult",
    "create_soft_verifier",
    # Execution verification (PoT)
    "ExecutionVerifier",
    "ExecutionResult",
    "ExecutionScore",
    "TestCase",
    "StaticExecutionVerifier",
    "RestrictedExecutionVerifier",
    "create_execution_verifier",
    "generate_test_cases_from_signature",
    # Collaborative verification (CoT + PoT)
    "CollaborativeVerifier",
    "CollaborativeResult",
    "VerificationStrategy",
    "StrategyScore",
    "collaborative_verify",
]
