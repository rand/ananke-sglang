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
"""Pre-allocated mask buffers for zero-copy operations.

This module provides MaskPool for efficient mask allocation without
triggering garbage collection in hot paths.

Key Features:
- Pre-allocated tensor pool
- Context manager for automatic return
- Thread-safe borrowing
- Configurable pool size and tensor size
- Device-aware allocation (CPU/CUDA)

Performance Target: 20-30% latency reduction by eliminating GC pressure.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, List, Optional

import torch


@dataclass
class MaskPoolStats:
    """Statistics for mask pool usage.

    Attributes:
        total_borrows: Total number of borrow operations
        total_returns: Total number of return operations
        pool_misses: Number of borrows when pool was empty
        peak_borrowed: Maximum concurrent borrows
        current_borrowed: Currently borrowed count
    """

    total_borrows: int = 0
    total_returns: int = 0
    pool_misses: int = 0
    peak_borrowed: int = 0
    current_borrowed: int = 0


class MaskPool:
    """Pre-allocated mask buffers to avoid allocation in hot paths.

    This pool pre-allocates a set of boolean mask tensors that can be
    borrowed and returned efficiently without triggering garbage collection.

    Thread Safety:
        All operations are thread-safe using a lock.

    Example:
        >>> pool = MaskPool(vocab_size=32000, pool_size=8)
        >>> with pool.borrow() as mask:
        ...     mask.fill_(False)  # Reset
        ...     mask[valid_tokens] = True
        ...     result = apply_mask(logits, mask)
        >>> # Mask is automatically returned to pool
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        pool_size: int = 16,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bool,
    ) -> None:
        """Initialize the mask pool.

        Args:
            vocab_size: Size of each mask tensor
            pool_size: Number of pre-allocated masks
            device: Device for tensors (default: CPU)
            dtype: Data type for masks (default: bool)
        """
        self._vocab_size = vocab_size
        self._pool_size = pool_size
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._lock = threading.RLock()
        self._stats = MaskPoolStats()

        # Pre-allocate masks
        self._pool: List[torch.Tensor] = [
            torch.zeros(vocab_size, dtype=dtype, device=self._device)
            for _ in range(pool_size)
        ]
        self._available: List[int] = list(range(pool_size))
        self._borrowed: set[int] = set()

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._vocab_size

    @property
    def pool_size(self) -> int:
        """Get pool size."""
        return self._pool_size

    @property
    def device(self) -> torch.device:
        """Get device."""
        return self._device

    @property
    def available_count(self) -> int:
        """Get number of available masks."""
        with self._lock:
            return len(self._available)

    @property
    def borrowed_count(self) -> int:
        """Get number of borrowed masks."""
        with self._lock:
            return len(self._borrowed)

    def get_stats(self) -> MaskPoolStats:
        """Get pool statistics.

        Returns:
            Current pool statistics
        """
        with self._lock:
            return MaskPoolStats(
                total_borrows=self._stats.total_borrows,
                total_returns=self._stats.total_returns,
                pool_misses=self._stats.pool_misses,
                peak_borrowed=self._stats.peak_borrowed,
                current_borrowed=len(self._borrowed),
            )

    @contextmanager
    def borrow(self) -> Generator[torch.Tensor, None, None]:
        """Borrow a mask from the pool.

        The mask is automatically returned when the context exits.
        If the pool is empty, a new mask is allocated (pool miss).

        Yields:
            A boolean mask tensor initialized to all True (all valid)
        """
        idx: Optional[int] = None
        mask: torch.Tensor

        with self._lock:
            self._stats.total_borrows += 1

            if self._available:
                idx = self._available.pop()
                self._borrowed.add(idx)
                mask = self._pool[idx]
            else:
                # Pool miss - allocate new mask
                self._stats.pool_misses += 1
                mask = torch.zeros(
                    self._vocab_size,
                    dtype=self._dtype,
                    device=self._device,
                )

            # Update peak
            current = len(self._borrowed)
            if current > self._stats.peak_borrowed:
                self._stats.peak_borrowed = current

        # Reset mask to all True (all valid)
        mask.fill_(True)

        try:
            yield mask
        finally:
            if idx is not None:
                with self._lock:
                    self._stats.total_returns += 1
                    self._borrowed.discard(idx)
                    self._available.append(idx)

    def borrow_raw(self) -> tuple[torch.Tensor, Optional[int]]:
        """Borrow a mask without context manager.

        Caller is responsible for calling return_mask().

        Returns:
            Tuple of (mask tensor, pool index or None if allocated)
        """
        with self._lock:
            self._stats.total_borrows += 1

            if self._available:
                idx = self._available.pop()
                self._borrowed.add(idx)
                mask = self._pool[idx]

                # Update peak
                current = len(self._borrowed)
                if current > self._stats.peak_borrowed:
                    self._stats.peak_borrowed = current

                mask.fill_(True)
                return mask, idx
            else:
                # Pool miss
                self._stats.pool_misses += 1
                mask = torch.ones(
                    self._vocab_size,
                    dtype=self._dtype,
                    device=self._device,
                )
                return mask, None

    def return_mask(self, idx: Optional[int]) -> None:
        """Return a borrowed mask to the pool.

        Args:
            idx: Pool index from borrow_raw, or None if was allocated
        """
        if idx is None:
            return

        with self._lock:
            self._stats.total_returns += 1
            self._borrowed.discard(idx)
            if idx not in self._available:
                self._available.append(idx)

    def resize(self, new_vocab_size: int) -> None:
        """Resize all masks in the pool.

        Note: This invalidates all currently borrowed masks!

        Args:
            new_vocab_size: New vocabulary size
        """
        with self._lock:
            if self._borrowed:
                raise RuntimeError(
                    f"Cannot resize pool while {len(self._borrowed)} masks are borrowed"
                )

            self._vocab_size = new_vocab_size
            self._pool = [
                torch.zeros(new_vocab_size, dtype=self._dtype, device=self._device)
                for _ in range(self._pool_size)
            ]
            self._available = list(range(self._pool_size))

    def to_device(self, device: torch.device) -> None:
        """Move all masks to a new device.

        Note: This invalidates all currently borrowed masks!

        Args:
            device: Target device
        """
        with self._lock:
            if self._borrowed:
                raise RuntimeError(
                    f"Cannot move pool while {len(self._borrowed)} masks are borrowed"
                )

            self._device = device
            self._pool = [mask.to(device) for mask in self._pool]

    def reset_stats(self) -> None:
        """Reset pool statistics."""
        with self._lock:
            self._stats = MaskPoolStats()


class MultiDeviceMaskPool:
    """Mask pool that manages pools across multiple devices.

    This is useful when constrained generation may happen on
    different GPUs in a multi-GPU setup.

    Example:
        >>> pool = MultiDeviceMaskPool(vocab_size=32000)
        >>> with pool.borrow(device=torch.device("cuda:0")) as mask:
        ...     # Use mask on GPU 0
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        pool_size_per_device: int = 8,
        dtype: torch.dtype = torch.bool,
    ) -> None:
        """Initialize multi-device pool.

        Args:
            vocab_size: Size of each mask
            pool_size_per_device: Masks per device
            dtype: Data type for masks
        """
        self._vocab_size = vocab_size
        self._pool_size = pool_size_per_device
        self._dtype = dtype
        self._pools: dict[torch.device, MaskPool] = {}
        self._lock = threading.Lock()

    def _get_pool(self, device: torch.device) -> MaskPool:
        """Get or create pool for device."""
        with self._lock:
            if device not in self._pools:
                self._pools[device] = MaskPool(
                    vocab_size=self._vocab_size,
                    pool_size=self._pool_size,
                    device=device,
                    dtype=self._dtype,
                )
            return self._pools[device]

    @contextmanager
    def borrow(
        self,
        device: Optional[torch.device] = None,
    ) -> Generator[torch.Tensor, None, None]:
        """Borrow a mask from the appropriate pool.

        Args:
            device: Target device (default: CPU)

        Yields:
            Borrowed mask tensor
        """
        device = device or torch.device("cpu")
        pool = self._get_pool(device)
        with pool.borrow() as mask:
            yield mask

    def get_stats(self) -> dict[str, MaskPoolStats]:
        """Get statistics for all device pools.

        Returns:
            Dict mapping device name to stats
        """
        with self._lock:
            return {
                str(device): pool.get_stats()
                for device, pool in self._pools.items()
            }


def create_mask_pool(
    vocab_size: int = 32000,
    pool_size: int = 16,
    device: Optional[torch.device] = None,
) -> MaskPool:
    """Factory function to create a mask pool.

    Args:
        vocab_size: Size of each mask
        pool_size: Number of pre-allocated masks
        device: Target device

    Returns:
        New MaskPool instance
    """
    return MaskPool(
        vocab_size=vocab_size,
        pool_size=pool_size,
        device=device,
    )


def create_multi_device_pool(
    vocab_size: int = 32000,
    pool_size_per_device: int = 8,
) -> MultiDeviceMaskPool:
    """Factory function to create a multi-device mask pool.

    Args:
        vocab_size: Size of each mask
        pool_size_per_device: Masks per device

    Returns:
        New MultiDeviceMaskPool instance
    """
    return MultiDeviceMaskPool(
        vocab_size=vocab_size,
        pool_size_per_device=pool_size_per_device,
    )
