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
"""Incremental mask computation.

Tracks which domain constraints have changed and only
recomputes affected masks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar
import time

import torch

from .fuser import FusionResult, DomainMaskInfo, TokenMaskFuser
from .cache import MaskCache, CacheKey

# Type variable for constraint types
C = TypeVar("C")


class ChangeKind(Enum):
    """Kind of constraint change.

    Changes:
    - NONE: No change
    - MODIFIED: Constraint was modified
    - ADDED: New constraint added
    - REMOVED: Constraint removed
    """

    NONE = auto()
    MODIFIED = auto()
    ADDED = auto()
    REMOVED = auto()


@dataclass
class ConstraintChange:
    """Record of a constraint change.

    Attributes:
        domain: Domain name
        kind: Type of change
        old_hash: Hash of old constraint (if any)
        new_hash: Hash of new constraint (if any)
    """

    domain: str
    kind: ChangeKind
    old_hash: Optional[int] = None
    new_hash: Optional[int] = None


@dataclass
class ComputationResult:
    """Result of incremental mask computation.

    Attributes:
        fused_mask: The final fused mask
        recomputed_domains: Domains that were recomputed
        cached_domains: Domains that used cached masks
        total_time_ns: Total computation time
    """

    fused_mask: torch.Tensor
    recomputed_domains: List[str] = field(default_factory=list)
    cached_domains: List[str] = field(default_factory=list)
    total_time_ns: int = 0

    @property
    def recompute_ratio(self) -> float:
        """Get ratio of domains that were recomputed."""
        total = len(self.recomputed_domains) + len(self.cached_domains)
        return len(self.recomputed_domains) / total if total > 0 else 0.0


class IncrementalMaskComputer:
    """Computes masks incrementally.

    Tracks constraint changes and only recomputes masks for
    domains whose constraints have changed.

    Key features:
    - Change detection via constraint hashing
    - Mask caching per domain
    - Dirty flag tracking

    Example:
        >>> computer = IncrementalMaskComputer(fuser)
        >>> result1 = computer.compute(constraints, context)
        >>> # Later, only type constraint changed
        >>> constraints["types"] = new_type_constraint
        >>> result2 = computer.compute(constraints, context)
        >>> # Only types domain mask recomputed
    """

    def __init__(
        self,
        fuser: TokenMaskFuser,
        cache: Optional[MaskCache] = None,
    ) -> None:
        """Initialize the incremental computer.

        Args:
            fuser: TokenMaskFuser to use for actual computation
            cache: Optional mask cache
        """
        self._fuser = fuser
        self._cache = cache or MaskCache()
        self._constraint_hashes: Dict[str, int] = {}
        self._dirty_domains: Set[str] = set()
        self._cached_masks: Dict[str, torch.Tensor] = {}
        self._position: int = 0

    @property
    def position(self) -> int:
        """Get current position."""
        return self._position

    def advance_position(self) -> None:
        """Advance to next position.

        Invalidates position-dependent caches.
        """
        self._position += 1
        # Mark all domains as dirty since position changed
        self._dirty_domains = set(self._fuser.domain_names)

    def detect_changes(
        self,
        constraints: Dict[str, Any],
    ) -> List[ConstraintChange]:
        """Detect which constraints have changed.

        Args:
            constraints: Current constraints

        Returns:
            List of detected changes
        """
        changes: List[ConstraintChange] = []

        # Check for modified/added constraints
        for domain, constraint in constraints.items():
            try:
                new_hash = hash(constraint)
            except TypeError:
                new_hash = id(constraint)

            old_hash = self._constraint_hashes.get(domain)

            if old_hash is None:
                # New constraint
                changes.append(ConstraintChange(
                    domain=domain,
                    kind=ChangeKind.ADDED,
                    new_hash=new_hash,
                ))
                self._dirty_domains.add(domain)
            elif old_hash != new_hash:
                # Modified constraint
                changes.append(ConstraintChange(
                    domain=domain,
                    kind=ChangeKind.MODIFIED,
                    old_hash=old_hash,
                    new_hash=new_hash,
                ))
                self._dirty_domains.add(domain)

            self._constraint_hashes[domain] = new_hash

        # Check for removed constraints
        for domain in list(self._constraint_hashes.keys()):
            if domain not in constraints:
                changes.append(ConstraintChange(
                    domain=domain,
                    kind=ChangeKind.REMOVED,
                    old_hash=self._constraint_hashes[domain],
                ))
                self._dirty_domains.add(domain)
                del self._constraint_hashes[domain]
                self._cached_masks.pop(domain, None)

        return changes

    def compute(
        self,
        constraints: Dict[str, Any],
        context: Any,
    ) -> ComputationResult:
        """Compute fused mask incrementally.

        Args:
            constraints: Map from domain name to constraint
            context: Generation context

        Returns:
            ComputationResult with fused mask and statistics
        """
        import time

        start_ns = time.perf_counter_ns()

        # Detect changes
        self.detect_changes(constraints)

        recomputed: List[str] = []
        cached: List[str] = []
        domain_masks: Dict[str, torch.Tensor] = {}

        # Process each domain
        for domain in self._fuser.domain_names:
            constraint = constraints.get(domain)
            if constraint is None:
                continue

            if domain in self._dirty_domains:
                # Need to recompute
                domain_obj = self._fuser._domains.get(domain)
                if domain_obj:
                    mask = domain_obj.token_mask(constraint, context)
                    domain_masks[domain] = mask
                    self._cached_masks[domain] = mask
                    recomputed.append(domain)
            else:
                # Can use cached
                if domain in self._cached_masks:
                    domain_masks[domain] = self._cached_masks[domain]
                    cached.append(domain)

        # Clear dirty flags
        self._dirty_domains.clear()

        # Fuse masks
        masks = list(domain_masks.values())
        if masks:
            fused_mask = self._fuser.fuse_masks(masks)
        else:
            vocab_size = getattr(context, "vocab_size", 32000)
            device = getattr(context, "device", "cpu")
            fused_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)

        end_ns = time.perf_counter_ns()

        return ComputationResult(
            fused_mask=fused_mask,
            recomputed_domains=recomputed,
            cached_domains=cached,
            total_time_ns=end_ns - start_ns,
        )

    def invalidate_domain(self, domain: str) -> None:
        """Mark a domain as needing recomputation.

        Args:
            domain: Domain name to invalidate
        """
        self._dirty_domains.add(domain)
        self._cached_masks.pop(domain, None)

    def invalidate_all(self) -> None:
        """Mark all domains as needing recomputation."""
        self._dirty_domains = set(self._fuser.domain_names)
        self._cached_masks.clear()

    def reset(self) -> None:
        """Reset all state."""
        self._constraint_hashes.clear()
        self._dirty_domains.clear()
        self._cached_masks.clear()
        self._position = 0


class PositionAwareMaskComputer:
    """Mask computer aware of generation position.

    Tracks masks per position for backtracking support.
    """

    def __init__(
        self,
        fuser: TokenMaskFuser,
        max_history: int = 100,
    ) -> None:
        """Initialize position-aware computer.

        Args:
            fuser: TokenMaskFuser to use
            max_history: Maximum positions to track
        """
        self._fuser = fuser
        self._max_history = max_history
        self._position_masks: Dict[int, Dict[str, torch.Tensor]] = {}
        self._position_constraints: Dict[int, Dict[str, int]] = {}
        self._current_position: int = 0

    @property
    def current_position(self) -> int:
        """Get current position."""
        return self._current_position

    def compute_at_position(
        self,
        position: int,
        constraints: Dict[str, Any],
        context: Any,
    ) -> torch.Tensor:
        """Compute mask at a specific position.

        Args:
            position: Position to compute for
            constraints: Current constraints
            context: Generation context

        Returns:
            Fused mask
        """
        self._current_position = position

        # Check if we have cached masks for this position
        constraint_hashes = {
            domain: self._safe_hash(constraint)
            for domain, constraint in constraints.items()
        }

        cached_hashes = self._position_constraints.get(position, {})
        if constraint_hashes == cached_hashes and position in self._position_masks:
            # Can reuse cached masks
            masks = list(self._position_masks[position].values())
            if masks:
                return self._fuser.fuse_masks(masks)

        # Need to compute
        result = self._fuser.fuse(constraints, context)

        # Cache results
        self._position_masks[position] = {
            info.domain_name: info.mask
            for info in result.domain_masks
        }
        self._position_constraints[position] = constraint_hashes

        # Evict old entries
        self._evict_old()

        return result.fused_mask

    def _safe_hash(self, obj: Any) -> int:
        """Safely hash an object."""
        try:
            return hash(obj)
        except TypeError:
            return id(obj)

    def _evict_old(self) -> None:
        """Evict old position entries."""
        if len(self._position_masks) > self._max_history:
            # Keep most recent positions
            positions = sorted(self._position_masks.keys())
            to_remove = positions[:-self._max_history]
            for pos in to_remove:
                del self._position_masks[pos]
                self._position_constraints.pop(pos, None)

    def rollback_to(self, position: int) -> bool:
        """Roll back to a previous position.

        Args:
            position: Position to roll back to

        Returns:
            True if rollback successful
        """
        if position > self._current_position:
            return False

        # Remove positions after target
        for pos in list(self._position_masks.keys()):
            if pos > position:
                del self._position_masks[pos]
                self._position_constraints.pop(pos, None)

        self._current_position = position
        return True

    def reset(self) -> None:
        """Reset all state."""
        self._position_masks.clear()
        self._position_constraints.clear()
        self._current_position = 0


def create_incremental_computer(
    fuser: TokenMaskFuser,
) -> IncrementalMaskComputer:
    """Factory function to create an IncrementalMaskComputer.

    Args:
        fuser: TokenMaskFuser to use

    Returns:
        New IncrementalMaskComputer instance
    """
    return IncrementalMaskComputer(fuser)
