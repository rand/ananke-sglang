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
"""Priority worklist for propagation scheduling.

The worklist manages the order in which domains are processed during
constraint propagation. It supports:
- Priority-based ordering (lower priority = process first)
- Deduplication (each domain appears at most once)
- Fixpoint detection (empty worklist = fixed point)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import Dict, List, Optional, Set, Tuple


@dataclass(order=True)
class WorklistItem:
    """An item in the priority worklist.

    Attributes:
        priority: Lower values are processed first
        domain_name: Name of the domain to process
        sequence: Insertion order for stable sorting
    """

    priority: int
    sequence: int
    domain_name: str = field(compare=False)


class PriorityWorklist:
    """Priority queue-based worklist for domain scheduling.

    Domains are processed in priority order (lower = first).
    Each domain appears at most once in the worklist.

    Example:
        >>> worklist = PriorityWorklist()
        >>> worklist.add("types", priority=50)
        >>> worklist.add("syntax", priority=25)
        >>> worklist.pop()  # Returns "syntax" (lower priority)
        'syntax'
    """

    def __init__(self):
        """Initialize an empty worklist."""
        self._heap: List[WorklistItem] = []
        self._in_worklist: Set[str] = set()
        self._sequence = 0

    def add(self, domain_name: str, priority: int = 100) -> bool:
        """Add a domain to the worklist.

        If the domain is already in the worklist, it is not added again.

        Args:
            domain_name: Name of the domain
            priority: Processing priority (lower = first)

        Returns:
            True if the domain was added, False if already present
        """
        if domain_name in self._in_worklist:
            return False

        item = WorklistItem(
            priority=priority,
            sequence=self._sequence,
            domain_name=domain_name,
        )
        self._sequence += 1
        heappush(self._heap, item)
        self._in_worklist.add(domain_name)
        return True

    def pop(self) -> Optional[str]:
        """Remove and return the highest-priority domain.

        Returns:
            Domain name, or None if worklist is empty
        """
        while self._heap:
            item = heappop(self._heap)
            # Skip if domain was removed (shouldn't happen with dedup)
            if item.domain_name in self._in_worklist:
                self._in_worklist.remove(item.domain_name)
                return item.domain_name
        return None

    def peek(self) -> Optional[str]:
        """View the highest-priority domain without removing it.

        Returns:
            Domain name, or None if worklist is empty
        """
        while self._heap:
            if self._heap[0].domain_name in self._in_worklist:
                return self._heap[0].domain_name
            heappop(self._heap)
        return None

    def contains(self, domain_name: str) -> bool:
        """Check if a domain is in the worklist.

        Args:
            domain_name: Name to check

        Returns:
            True if domain is in worklist
        """
        return domain_name in self._in_worklist

    def remove(self, domain_name: str) -> bool:
        """Remove a domain from the worklist.

        Args:
            domain_name: Name to remove

        Returns:
            True if domain was removed, False if not present
        """
        if domain_name in self._in_worklist:
            self._in_worklist.remove(domain_name)
            return True
        return False

    def clear(self) -> None:
        """Clear all domains from the worklist."""
        self._heap = []
        self._in_worklist = set()

    def is_empty(self) -> bool:
        """Check if the worklist is empty.

        Returns:
            True if no domains in worklist
        """
        return len(self._in_worklist) == 0

    def __len__(self) -> int:
        """Return number of domains in worklist."""
        return len(self._in_worklist)

    def __bool__(self) -> bool:
        """True if worklist is non-empty."""
        return not self.is_empty()

    def domains(self) -> Set[str]:
        """Get set of all domains in worklist.

        Returns:
            Set of domain names
        """
        return self._in_worklist.copy()


class FIFOWorklist:
    """First-in-first-out worklist for domain scheduling.

    Simple queue-based worklist without priorities.
    Each domain appears at most once.

    Example:
        >>> worklist = FIFOWorklist()
        >>> worklist.add("types")
        >>> worklist.add("syntax")
        >>> worklist.pop()  # Returns "types" (first added)
        'types'
    """

    def __init__(self):
        """Initialize an empty worklist."""
        self._queue: List[str] = []
        self._in_worklist: Set[str] = set()

    def add(self, domain_name: str) -> bool:
        """Add a domain to the worklist.

        Args:
            domain_name: Name of the domain

        Returns:
            True if added, False if already present
        """
        if domain_name in self._in_worklist:
            return False

        self._queue.append(domain_name)
        self._in_worklist.add(domain_name)
        return True

    def pop(self) -> Optional[str]:
        """Remove and return the first domain.

        Returns:
            Domain name, or None if empty
        """
        while self._queue:
            domain = self._queue.pop(0)
            if domain in self._in_worklist:
                self._in_worklist.remove(domain)
                return domain
        return None

    def is_empty(self) -> bool:
        """Check if worklist is empty."""
        return len(self._in_worklist) == 0

    def __len__(self) -> int:
        """Return number of domains."""
        return len(self._in_worklist)

    def __bool__(self) -> bool:
        """True if non-empty."""
        return not self.is_empty()

    def clear(self) -> None:
        """Clear the worklist."""
        self._queue = []
        self._in_worklist = set()


class IterationLimiter:
    """Utility to limit iterations and detect infinite loops.

    Tracks iteration count and can enforce a maximum.

    Example:
        >>> limiter = IterationLimiter(max_iterations=100)
        >>> while not limiter.is_exhausted():
        ...     limiter.increment()
        ...     # do work
    """

    def __init__(self, max_iterations: int = 100):
        """Initialize the limiter.

        Args:
            max_iterations: Maximum allowed iterations
        """
        self._max = max_iterations
        self._count = 0

    @property
    def count(self) -> int:
        """Get current iteration count."""
        return self._count

    @property
    def max_iterations(self) -> int:
        """Get maximum iterations."""
        return self._max

    @property
    def remaining(self) -> int:
        """Get remaining iterations."""
        return max(0, self._max - self._count)

    def increment(self) -> bool:
        """Increment counter and check limit.

        Returns:
            True if under limit, False if exhausted
        """
        self._count += 1
        return self._count <= self._max

    def is_exhausted(self) -> bool:
        """Check if limit has been reached.

        Returns:
            True if at or over limit
        """
        return self._count >= self._max

    def reset(self) -> None:
        """Reset counter to zero."""
        self._count = 0


@dataclass
class FixpointResult:
    """Result of a fixpoint computation.

    Attributes:
        converged: True if fixpoint was reached
        iterations: Number of iterations performed
        processed: Set of domains that were processed
    """

    converged: bool
    iterations: int
    processed: Set[str] = field(default_factory=set)

    @property
    def is_success(self) -> bool:
        """True if converged successfully."""
        return self.converged
