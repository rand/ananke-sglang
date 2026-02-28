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
"""Order maintenance data structure for incremental bidirectional typing.

This implements the Dietz & Sleator algorithm for maintaining a total order
over elements with O(1) amortized insert and query operations.

The key insight is that we can maintain a labeling of elements such that:
- order(a, b) can be answered by comparing labels
- insert_after(a, b) creates a new label between a and its successor
- When labels get too dense, we relabel a portion of the list

This enables O(1) amortized type dependency tracking during incremental
type checking, providing ~275x speedup over naive reanalysis.

References:
    - Dietz & Sleator (1987). "Two algorithms for maintaining order in a list"
    - OOPSLA 2025: "Incremental Bidirectional Typing via Order Maintenance"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generic, Iterator, List, Optional, TypeVar

# Maximum label value (using 64-bit integers for large capacity)
MAX_LABEL = 2**62


T = TypeVar("T")


@dataclass
class OrderedElement(Generic[T]):
    """An element in the order maintenance list.

    Each element has a label used for O(1) order comparisons.
    Elements also form a doubly-linked list for efficient traversal.

    Attributes:
        value: The actual value stored in this element
        label: Integer label for order comparison
        prev: Previous element in the list
        next: Next element in the list
    """

    value: T
    label: int = 0
    prev: Optional[OrderedElement[T]] = None
    next: Optional[OrderedElement[T]] = None

    def __lt__(self, other: OrderedElement[T]) -> bool:
        """Compare elements by their labels."""
        return self.label < other.label

    def __le__(self, other: OrderedElement[T]) -> bool:
        return self.label <= other.label

    def __gt__(self, other: OrderedElement[T]) -> bool:
        return self.label > other.label

    def __ge__(self, other: OrderedElement[T]) -> bool:
        return self.label >= other.label

    def __repr__(self) -> str:
        return f"OrderedElement({self.value!r}, label={self.label})"


class OrderMaintenanceList(Generic[T]):
    """Order maintenance data structure with O(1) amortized operations.

    Maintains a total order over elements with efficient:
    - order(a, b): O(1) - compare if a comes before b
    - insert_after(a, elem): O(1) amortized - insert elem after a
    - insert_before(a, elem): O(1) amortized - insert elem before a
    - delete(elem): O(1) - remove element from list

    The implementation uses the Dietz & Sleator algorithm:
    - Each element has an integer label
    - order(a, b) compares labels directly
    - insert assigns a label between neighbors
    - When labels get too dense, relabel a region

    Attributes:
        head: First element in the list
        tail: Last element in the list
        size: Number of elements
    """

    def __init__(self, initial_spacing: int = MAX_LABEL // 2):
        """Initialize an empty order maintenance list.

        Args:
            initial_spacing: Initial gap between labels
        """
        self._head: Optional[OrderedElement[T]] = None
        self._tail: Optional[OrderedElement[T]] = None
        self._size: int = 0
        self._initial_spacing = initial_spacing
        self._elements: Dict[int, OrderedElement[T]] = {}  # id(value) -> element

    @property
    def head(self) -> Optional[OrderedElement[T]]:
        """Get the first element."""
        return self._head

    @property
    def tail(self) -> Optional[OrderedElement[T]]:
        """Get the last element."""
        return self._tail

    @property
    def size(self) -> int:
        """Get the number of elements."""
        return self._size

    def is_empty(self) -> bool:
        """Check if the list is empty."""
        return self._size == 0

    def order(self, a: OrderedElement[T], b: OrderedElement[T]) -> int:
        """Compare the order of two elements.

        Args:
            a: First element
            b: Second element

        Returns:
            -1 if a < b, 0 if a == b, 1 if a > b
        """
        if a.label < b.label:
            return -1
        elif a.label > b.label:
            return 1
        return 0

    def comes_before(self, a: OrderedElement[T], b: OrderedElement[T]) -> bool:
        """Check if a comes before b in the total order.

        Args:
            a: First element
            b: Second element

        Returns:
            True if a < b
        """
        return a.label < b.label

    def insert_first(self, value: T) -> OrderedElement[T]:
        """Insert a value at the beginning of the list.

        Args:
            value: The value to insert

        Returns:
            The new OrderedElement
        """
        elem = OrderedElement(value=value)

        if self._head is None:
            # Empty list - first element gets middle label
            elem.label = self._initial_spacing
            self._head = elem
            self._tail = elem
        else:
            # Insert before current head
            new_label = self._head.label // 2
            if new_label <= 0:
                # Need to relabel to make space
                self._relabel_from(self._head)
                new_label = self._head.label // 2

            elem.label = new_label
            elem.next = self._head
            self._head.prev = elem
            self._head = elem

        self._size += 1
        self._elements[id(value)] = elem
        return elem

    def insert_last(self, value: T) -> OrderedElement[T]:
        """Insert a value at the end of the list.

        Args:
            value: The value to insert

        Returns:
            The new OrderedElement
        """
        elem = OrderedElement(value=value)

        if self._tail is None:
            # Empty list - first element gets middle label
            elem.label = self._initial_spacing
            self._head = elem
            self._tail = elem
        else:
            # Insert after current tail
            new_label = self._tail.label + (MAX_LABEL - self._tail.label) // 2
            if new_label >= MAX_LABEL or new_label <= self._tail.label:
                # Need to relabel to make space
                self._relabel_region_around(self._tail)
                new_label = self._tail.label + (MAX_LABEL - self._tail.label) // 2

            elem.label = new_label
            elem.prev = self._tail
            self._tail.next = elem
            self._tail = elem

        self._size += 1
        self._elements[id(value)] = elem
        return elem

    def insert_after(
        self, after: OrderedElement[T], value: T
    ) -> OrderedElement[T]:
        """Insert a value after an existing element.

        Args:
            after: The element to insert after
            value: The value to insert

        Returns:
            The new OrderedElement
        """
        if after.next is None:
            # Inserting at end
            return self.insert_last(value)

        elem = OrderedElement(value=value)

        # Calculate new label between after and after.next
        gap = after.next.label - after.label
        if gap <= 1:
            # No space - need to relabel
            self._relabel_region_around(after)
            gap = after.next.label - after.label

        new_label = after.label + gap // 2
        elem.label = new_label

        # Link into list
        elem.prev = after
        elem.next = after.next
        after.next.prev = elem
        after.next = elem

        self._size += 1
        self._elements[id(value)] = elem
        return elem

    def insert_before(
        self, before: OrderedElement[T], value: T
    ) -> OrderedElement[T]:
        """Insert a value before an existing element.

        Args:
            before: The element to insert before
            value: The value to insert

        Returns:
            The new OrderedElement
        """
        if before.prev is None:
            # Inserting at beginning
            return self.insert_first(value)

        return self.insert_after(before.prev, value)

    def delete(self, elem: OrderedElement[T]) -> None:
        """Delete an element from the list.

        Args:
            elem: The element to delete
        """
        # Update links
        if elem.prev is not None:
            elem.prev.next = elem.next
        else:
            self._head = elem.next

        if elem.next is not None:
            elem.next.prev = elem.prev
        else:
            self._tail = elem.prev

        # Clear element's links
        elem.prev = None
        elem.next = None

        self._size -= 1
        self._elements.pop(id(elem.value), None)

    def find(self, value: T) -> Optional[OrderedElement[T]]:
        """Find the element containing a value.

        Args:
            value: The value to find

        Returns:
            The element, or None if not found
        """
        return self._elements.get(id(value))

    def _relabel_from(self, start: OrderedElement[T]) -> None:
        """Relabel a region starting from an element.

        This spreads out labels to make room for insertions.

        Args:
            start: Starting element of region to relabel
        """
        # Count elements that need relabeling
        count = 0
        current = start
        target_density = MAX_LABEL // (self._size + 10)  # Target spacing

        # Find region that's too dense
        while current is not None and count < self._size:
            count += 1
            if current.next is None:
                break
            gap = current.next.label - current.label
            if gap >= target_density:
                break
            current = current.next

        # Relabel the region with even spacing
        spacing = target_density
        label = start.label if start.prev is None else start.prev.label + spacing

        current = start
        for _ in range(count):
            if current is None:
                break
            current.label = label
            label += spacing
            current = current.next

    def _relabel_region_around(self, center: OrderedElement[T]) -> None:
        """Relabel a region around an element.

        Args:
            center: Center element of region to relabel
        """
        # For small lists, do a full relabel for simplicity
        if self._size <= 32:
            self._relabel_all()
            return

        # Find region boundaries (look at surrounding elements)
        region_size = min(32, self._size)  # Relabel up to 32 elements

        # Find start of region
        start = center
        for _ in range(region_size // 2):
            if start.prev is None:
                break
            start = start.prev

        # Count actual region size and find end
        count = 0
        end = start
        while end is not None and count < region_size:
            count += 1
            if end.next is None:
                break
            end = end.next

        # Determine label range for the region
        start_label = 0 if start.prev is None else start.prev.label + 1
        end_label = MAX_LABEL if end.next is None else end.next.label - 1

        # Calculate spacing
        available_range = end_label - start_label
        if count > 0:
            spacing = available_range // (count + 1)
            if spacing < 2:
                # Not enough space in local region - do a full relabel
                self._relabel_all()
                return

            # Relabel elements in the region
            current = start
            label = start_label + spacing
            while current is not None and current != end.next:
                current.label = label
                label += spacing
                current = current.next

    def _relabel_all(self) -> None:
        """Relabel all elements with even spacing.

        This is a fallback when local relabeling isn't sufficient.
        """
        if self._size == 0:
            return

        # Calculate even spacing across full label range
        spacing = MAX_LABEL // (self._size + 1)
        if spacing < 1:
            spacing = 1

        # Relabel all elements
        current = self._head
        label = spacing
        while current is not None:
            current.label = label
            label += spacing
            current = current.next

    def __iter__(self) -> Iterator[T]:
        """Iterate over values in order."""
        current = self._head
        while current is not None:
            yield current.value
            current = current.next

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        values = list(self)
        return f"OrderMaintenanceList({values})"


def create_order_list() -> OrderMaintenanceList:
    """Create a new order maintenance list.

    Returns:
        An empty OrderMaintenanceList
    """
    return OrderMaintenanceList()
