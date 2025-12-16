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
"""Speculative decoding module for CDSL-style constrained generation.

This module provides CDSL-style (Constrained Decoding with Speculative
Lookaheads) speculative decoding for efficient constrained generation.

Key Components:
    - DraftModel: Protocol for generating draft token sequences
    - ConstrainedLookahead: CDSL algorithm for efficient generation
    - ParallelConstrainedLookahead: Multi-sequence speculation

Algorithm Overview:
    1. Generate K draft tokens with relaxed constraints (syntax-only)
    2. Verify draft sequence against full domain stack
    3. Accept longest valid prefix
    4. Adapt lookahead based on acceptance rate

Expected Speedup: 2-4x for typical code generation

References:
    - CDSL: Constrained Decoding with Speculative Lookaheads (NAACL 2025)
      https://arxiv.org/abs/2412.10418
    - Test-Time Compute Scaling (DeepMind, 2024)
"""

from .draft_model import (
    DraftContext,
    DraftResult,
    DraftModel,
    NullDraftModel,
    GreedyDraftModel,
    SamplingDraftModel,
    CachedDraftModel,
    create_draft_model,
)
from .constrained_lookahead import (
    LookaheadConfig,
    VerificationResult,
    LookaheadStats,
    ConstraintVerifier,
    ConstrainedLookahead,
    ParallelConstrainedLookahead,
    create_constrained_lookahead,
)

__all__ = [
    # Draft model
    "DraftContext",
    "DraftResult",
    "DraftModel",
    "NullDraftModel",
    "GreedyDraftModel",
    "SamplingDraftModel",
    "CachedDraftModel",
    "create_draft_model",
    # Constrained lookahead
    "LookaheadConfig",
    "VerificationResult",
    "LookaheadStats",
    "ConstraintVerifier",
    "ConstrainedLookahead",
    "ParallelConstrainedLookahead",
    "create_constrained_lookahead",
]
