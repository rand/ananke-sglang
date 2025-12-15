# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Python FFI wrapper for Zig SIMD mask fusion.

This module provides a Python interface to the Zig-implemented SIMD
mask fusion functions. Falls back to pure Python if the native library
is not available.

Usage:
    from ananke.zig.ffi import fuse_masks, fuse_masks_selective

    # Fuse multiple masks
    result, popcount = fuse_masks([mask1, mask2, mask3])

    # Fuse with early termination
    result, popcount = fuse_masks_selective(
        masks=[mask1, mask2, mask3],
        selectivities=[100, 200, 50],
        early_stop_threshold=10,
    )
"""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# Optional numpy import (needed for native library but not for fallback)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

logger = logging.getLogger(__name__)

# Library loading
_lib: Optional[ctypes.CDLL] = None
_lib_loaded = False


def _get_lib_path() -> Optional[Path]:
    """Get path to the native library.

    Searches in order:
    1. Same directory as this module
    2. zig-out/lib subdirectory (build output)
    3. System library paths

    Returns:
        Path to library or None if not found
    """
    import platform

    module_dir = Path(__file__).parent

    # Library name varies by platform
    system = platform.system()
    if system == "Darwin":
        lib_names = ["libananke_native.dylib", "libananke_native.so"]
    elif system == "Linux":
        lib_names = ["libananke_native.so"]
    elif system == "Windows":
        lib_names = ["ananke_native.dll"]
    else:
        lib_names = ["libananke_native.so"]

    # Search paths
    search_paths = [
        module_dir,
        module_dir / "zig-out" / "lib",
        Path("/usr/local/lib"),
    ]

    for search_path in search_paths:
        for lib_name in lib_names:
            lib_path = search_path / lib_name
            if lib_path.exists():
                return lib_path

    return None


def _load_library() -> Optional[ctypes.CDLL]:
    """Load the native library.

    Returns:
        Loaded library or None if not available
    """
    global _lib, _lib_loaded

    if _lib_loaded:
        return _lib

    _lib_loaded = True

    # Require numpy for native library
    if not HAS_NUMPY:
        logger.info("NumPy not available, using pure Python fallback")
        return None

    lib_path = _get_lib_path()
    if lib_path is None:
        logger.info("Ananke native library not found, using pure Python fallback")
        return None

    try:
        lib = ctypes.CDLL(str(lib_path))

        # Configure function signatures
        lib.ananke_fuse_masks.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.ananke_fuse_masks.restype = ctypes.c_uint64

        lib.ananke_fuse_masks_selective.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint64,
        ]
        lib.ananke_fuse_masks_selective.restype = ctypes.c_uint64

        lib.ananke_popcount.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
        ]
        lib.ananke_popcount.restype = ctypes.c_uint64

        lib.ananke_native_version.argtypes = []
        lib.ananke_native_version.restype = ctypes.c_uint32

        # Verify library is functional
        version = lib.ananke_native_version()
        logger.info(f"Loaded Ananke native library v{version >> 24}.{(version >> 16) & 0xFF}.{(version >> 8) & 0xFF}")

        _lib = lib
        return lib

    except Exception as e:
        logger.warning(f"Failed to load Ananke native library: {e}")
        return None


def is_native_available() -> bool:
    """Check if native library is available.

    Returns:
        True if native SIMD functions are available
    """
    return _load_library() is not None


def fuse_masks(
    masks: List[torch.Tensor],
    device: str = "cpu",
) -> Tuple[torch.Tensor, int]:
    """Fuse multiple boolean masks using SIMD-accelerated bitwise AND.

    Args:
        masks: List of boolean tensors of shape (vocab_size,)
        device: Target device for result

    Returns:
        Tuple of (fused_mask, popcount)
    """
    if not masks:
        raise ValueError("At least one mask required")

    vocab_size = masks[0].shape[0]
    mask_size = (vocab_size + 31) // 32  # Round up to u32 boundary

    lib = _load_library()
    if lib is None:
        # Pure Python fallback
        result = masks[0].clone()
        for mask in masks[1:]:
            result &= mask
        return result, int(result.sum().item())

    # Convert boolean tensors to packed u32 arrays
    mask_arrays = []
    mask_ptrs = []

    for mask in masks:
        # Ensure on CPU and convert to numpy
        mask_np = mask.cpu().numpy().astype(np.uint8)

        # Pad to u32 boundary
        padded_size = mask_size * 32
        if len(mask_np) < padded_size:
            mask_np = np.pad(mask_np, (0, padded_size - len(mask_np)))

        # Pack bits into u32s
        packed = np.packbits(mask_np, bitorder="little").view(np.uint32)

        mask_arrays.append(packed)
        mask_ptrs.append(packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)))

    # Create output buffer
    result_array = np.zeros(mask_size, dtype=np.uint32)

    # Create pointer array
    ptr_array = (ctypes.POINTER(ctypes.c_uint32) * len(masks))(*mask_ptrs)

    # Call native function
    popcount = lib.ananke_fuse_masks(
        ptr_array,
        len(masks),
        mask_size,
        result_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
    )

    # Unpack result to boolean tensor
    result_bits = np.unpackbits(result_array.view(np.uint8), bitorder="little")
    result_tensor = torch.from_numpy(result_bits[:vocab_size].astype(bool)).to(device)

    return result_tensor, int(popcount)


def fuse_masks_selective(
    masks: List[torch.Tensor],
    selectivities: Optional[List[int]] = None,
    early_stop_threshold: int = 0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, int]:
    """Fuse masks in selectivity order with early termination.

    Processes masks from most selective (fewest 1-bits) to least selective.
    Stops early if popcount drops below threshold.

    Args:
        masks: List of boolean tensors
        selectivities: Pre-computed popcounts (computed if not provided)
        early_stop_threshold: Stop if popcount <= this value
        device: Target device for result

    Returns:
        Tuple of (fused_mask, final_popcount)
    """
    if not masks:
        raise ValueError("At least one mask required")

    lib = _load_library()
    if lib is None:
        # Pure Python fallback with selectivity ordering
        if selectivities is None:
            selectivities = [int(m.sum().item()) for m in masks]

        # Sort by selectivity (most selective = lowest popcount first)
        sorted_indices = sorted(range(len(masks)), key=lambda i: selectivities[i])
        sorted_masks = [masks[i] for i in sorted_indices]

        result = sorted_masks[0].clone()
        for mask in sorted_masks[1:]:
            result &= mask
            if result.sum().item() <= early_stop_threshold:
                break

        return result, int(result.sum().item())

    # Compute selectivities if not provided
    if selectivities is None:
        selectivities = [int(m.sum().item()) for m in masks]

    vocab_size = masks[0].shape[0]
    mask_size = (vocab_size + 31) // 32

    # Convert to packed format (same as fuse_masks)
    mask_arrays = []
    mask_ptrs = []

    for mask in masks:
        mask_np = mask.cpu().numpy().astype(np.uint8)
        padded_size = mask_size * 32
        if len(mask_np) < padded_size:
            mask_np = np.pad(mask_np, (0, padded_size - len(mask_np)))
        packed = np.packbits(mask_np, bitorder="little").view(np.uint32)
        mask_arrays.append(packed)
        mask_ptrs.append(packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)))

    result_array = np.zeros(mask_size, dtype=np.uint32)
    ptr_array = (ctypes.POINTER(ctypes.c_uint32) * len(masks))(*mask_ptrs)
    sel_array = (ctypes.c_uint64 * len(masks))(*selectivities)

    popcount = lib.ananke_fuse_masks_selective(
        ptr_array,
        sel_array,
        len(masks),
        mask_size,
        result_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        early_stop_threshold,
    )

    result_bits = np.unpackbits(result_array.view(np.uint8), bitorder="little")
    result_tensor = torch.from_numpy(result_bits[:vocab_size].astype(bool)).to(device)

    return result_tensor, int(popcount)
