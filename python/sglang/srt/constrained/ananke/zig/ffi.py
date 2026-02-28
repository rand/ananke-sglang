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


# =============================================================================
# VocabPartition - Full Vocabulary Classification in Zig
# =============================================================================


# Language enum mapping (matches classifier.zig Language enum)
LANGUAGE_PYTHON = 0
LANGUAGE_TYPESCRIPT = 1
LANGUAGE_RUST = 2
LANGUAGE_GO = 3
LANGUAGE_KOTLIN = 4
LANGUAGE_ZIG = 5
LANGUAGE_SWIFT = 6

# TypeCategory enum mapping (matches vocab_partition.zig TypeCategory)
TYPE_CATEGORY_INTEGER = 0
TYPE_CATEGORY_FLOAT = 1
TYPE_CATEGORY_STRING = 2
TYPE_CATEGORY_BOOLEAN = 3
TYPE_CATEGORY_NONE_NULL = 4
TYPE_CATEGORY_IDENTIFIER = 10
TYPE_CATEGORY_TYPE_NAME = 11
TYPE_CATEGORY_KEYWORD = 12
TYPE_CATEGORY_BUILTIN = 13
TYPE_CATEGORY_ARITHMETIC_OP = 20
TYPE_CATEGORY_COMPARISON_OP = 21
TYPE_CATEGORY_LOGICAL_OP = 22
TYPE_CATEGORY_BITWISE_OP = 23
TYPE_CATEGORY_STRING_OP = 24
TYPE_CATEGORY_BRACKET_OPEN = 30
TYPE_CATEGORY_BRACKET_CLOSE = 31
TYPE_CATEGORY_DELIMITER = 32
TYPE_CATEGORY_WHITESPACE = 33
TYPE_CATEGORY_ANY = 254
TYPE_CATEGORY_UNKNOWN = 255


def _language_to_int(language: str) -> int:
    """Convert language string to enum int."""
    lang_map = {
        "python": LANGUAGE_PYTHON,
        "py": LANGUAGE_PYTHON,
        "typescript": LANGUAGE_TYPESCRIPT,
        "ts": LANGUAGE_TYPESCRIPT,
        "javascript": LANGUAGE_TYPESCRIPT,  # Use TS classifier
        "js": LANGUAGE_TYPESCRIPT,
        "rust": LANGUAGE_RUST,
        "rs": LANGUAGE_RUST,
        "go": LANGUAGE_GO,
        "golang": LANGUAGE_GO,
        "kotlin": LANGUAGE_KOTLIN,
        "kt": LANGUAGE_KOTLIN,
        "zig": LANGUAGE_ZIG,
        "swift": LANGUAGE_SWIFT,
    }
    return lang_map.get(language.lower(), LANGUAGE_PYTHON)


class VocabPartition:
    """Python wrapper for Zig VocabPartition.

    Provides O(1) token category lookup after one-time vocabulary classification.
    Falls back to pure Python if native library is not available.

    Usage:
        partition = VocabPartition(vocab_size=32000, language="python")
        partition.classify_vocabulary(token_strings)

        # O(1) lookups
        category = partition.get_category(token_id)
        mask = partition.get_category_mask(TYPE_CATEGORY_INTEGER)
    """

    def __init__(self, vocab_size: int, language: str = "python"):
        """Initialize a vocabulary partition.

        Args:
            vocab_size: Number of tokens in vocabulary
            language: Programming language for classification
        """
        self._vocab_size = vocab_size
        self._language = language
        self._language_int = _language_to_int(language)
        self._handle: Optional[ctypes.c_void_p] = None
        self._mask_size = (vocab_size + 31) // 32

        # Try to use native library
        lib = _load_library()
        if lib is not None and HAS_NUMPY:
            self._lib = lib
            self._setup_native()
        else:
            self._lib = None
            # Fallback: store categories in numpy array
            if HAS_NUMPY:
                self._categories = np.full(vocab_size, TYPE_CATEGORY_UNKNOWN, dtype=np.uint8)
            else:
                self._categories = [TYPE_CATEGORY_UNKNOWN] * vocab_size

    def _setup_native(self) -> None:
        """Set up native library function signatures."""
        lib = self._lib
        if lib is None:
            return

        # Configure function signatures if not already done
        if not hasattr(lib, "_vocab_partition_configured"):
            lib.ananke_vocab_partition_create.argtypes = [ctypes.c_size_t]
            lib.ananke_vocab_partition_create.restype = ctypes.c_void_p

            lib.ananke_vocab_partition_destroy.argtypes = [ctypes.c_void_p]
            lib.ananke_vocab_partition_destroy.restype = None

            lib.ananke_vocab_partition_classify_all.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.c_uint8,
            ]
            lib.ananke_vocab_partition_classify_all.restype = None

            lib.ananke_vocab_partition_get_category_mask.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint8,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
            ]
            lib.ananke_vocab_partition_get_category_mask.restype = ctypes.c_uint64

            lib.ananke_vocab_partition_compute_type_mask.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.c_bool,
            ]
            lib.ananke_vocab_partition_compute_type_mask.restype = ctypes.c_uint64

            lib.ananke_vocab_partition_get_vocab_size.argtypes = [ctypes.c_void_p]
            lib.ananke_vocab_partition_get_vocab_size.restype = ctypes.c_size_t

            lib.ananke_vocab_partition_get_category.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            lib.ananke_vocab_partition_get_category.restype = ctypes.c_uint8

            lib.ananke_vocab_partition_is_in_category.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint8,
            ]
            lib.ananke_vocab_partition_is_in_category.restype = ctypes.c_bool

            lib._vocab_partition_configured = True

        # Create the native partition
        self._handle = lib.ananke_vocab_partition_create(self._vocab_size)
        if self._handle is None:
            logger.warning("Failed to create native VocabPartition, falling back to Python")
            self._lib = None
            if HAS_NUMPY:
                self._categories = np.full(self._vocab_size, TYPE_CATEGORY_UNKNOWN, dtype=np.uint8)
            else:
                self._categories = [TYPE_CATEGORY_UNKNOWN] * self._vocab_size

    def __del__(self):
        """Clean up native resources."""
        if self._lib is not None and self._handle is not None:
            self._lib.ananke_vocab_partition_destroy(self._handle)
            self._handle = None

    @property
    def is_native(self) -> bool:
        """Check if using native Zig implementation."""
        return self._lib is not None and self._handle is not None

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._vocab_size

    def classify_vocabulary(self, token_strings: List[str]) -> None:
        """Classify all tokens in the vocabulary.

        This should be called once after initialization with all decoded
        token strings from the tokenizer.

        Args:
            token_strings: List of decoded token strings, indexed by token_id
        """
        if len(token_strings) != self._vocab_size:
            raise ValueError(
                f"Expected {self._vocab_size} tokens, got {len(token_strings)}"
            )

        if self.is_native:
            self._classify_vocabulary_native(token_strings)
        else:
            self._classify_vocabulary_python(token_strings)

    def _classify_vocabulary_native(self, token_strings: List[str]) -> None:
        """Classify vocabulary using native Zig implementation."""
        # Encode strings to bytes
        encoded = [s.encode("utf-8") for s in token_strings]
        lengths = [len(s) for s in encoded]

        # Create ctypes arrays
        token_ptrs = (ctypes.c_char_p * len(encoded))(*encoded)
        length_array = (ctypes.c_size_t * len(lengths))(*lengths)

        # Call native function
        self._lib.ananke_vocab_partition_classify_all(
            self._handle,
            token_ptrs,
            length_array,
            len(token_strings),
            self._language_int,
        )

    def _classify_vocabulary_python(self, token_strings: List[str]) -> None:
        """Classify vocabulary using pure Python (fallback)."""
        # Import the Python classifier
        try:
            from sglang.srt.constrained.ananke.core.token_classifier import (
                classify_token_for_language,
                TokenCategory,
            )
        except ImportError:
            from core.token_classifier import classify_token_for_language, TokenCategory

        # Map Python TokenCategory to our TypeCategory
        category_map = {
            TokenCategory.INT_LITERAL: TYPE_CATEGORY_INTEGER,
            TokenCategory.FLOAT_LITERAL: TYPE_CATEGORY_FLOAT,
            TokenCategory.STRING_LITERAL: TYPE_CATEGORY_STRING,
            TokenCategory.BOOL_LITERAL: TYPE_CATEGORY_BOOLEAN,
            TokenCategory.NONE_LITERAL: TYPE_CATEGORY_NONE_NULL,
            TokenCategory.IDENTIFIER: TYPE_CATEGORY_IDENTIFIER,
            TokenCategory.KEYWORD: TYPE_CATEGORY_KEYWORD,
            TokenCategory.BUILTIN: TYPE_CATEGORY_BUILTIN,
            TokenCategory.OPERATOR: TYPE_CATEGORY_ARITHMETIC_OP,
            TokenCategory.DELIMITER: TYPE_CATEGORY_DELIMITER,
            TokenCategory.WHITESPACE: TYPE_CATEGORY_WHITESPACE,
            TokenCategory.COMMENT: TYPE_CATEGORY_WHITESPACE,
            TokenCategory.MIXED: TYPE_CATEGORY_UNKNOWN,
            TokenCategory.UNKNOWN: TYPE_CATEGORY_UNKNOWN,
        }

        for token_id, text in enumerate(token_strings):
            cat, _, _ = classify_token_for_language(text, self._language)
            type_cat = category_map.get(cat, TYPE_CATEGORY_UNKNOWN)
            if HAS_NUMPY:
                self._categories[token_id] = type_cat
            else:
                self._categories[token_id] = type_cat

    def get_category(self, token_id: int) -> int:
        """Get category for a single token (O(1))."""
        if self.is_native:
            return self._lib.ananke_vocab_partition_get_category(
                self._handle, token_id
            )
        else:
            if 0 <= token_id < self._vocab_size:
                return self._categories[token_id]
            return TYPE_CATEGORY_UNKNOWN

    def is_in_category(self, token_id: int, category: int) -> bool:
        """Check if token belongs to category (O(1))."""
        if self.is_native:
            return self._lib.ananke_vocab_partition_is_in_category(
                self._handle, token_id, category
            )
        else:
            return self.get_category(token_id) == category

    def get_category_mask(self, category: int) -> Tuple[np.ndarray, int]:
        """Get precomputed mask for a category.

        Returns:
            Tuple of (mask as packed u32 array, popcount)
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy required for mask operations")

        result = np.zeros(self._mask_size, dtype=np.uint32)

        if self.is_native:
            popcount = self._lib.ananke_vocab_partition_get_category_mask(
                self._handle,
                category,
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                self._mask_size,
            )
        else:
            # Python fallback: build mask manually
            popcount = 0
            for token_id in range(self._vocab_size):
                if self._categories[token_id] == category:
                    word_idx = token_id // 32
                    bit_idx = token_id % 32
                    result[word_idx] |= 1 << bit_idx
                    popcount += 1

        return result, int(popcount)

    def compute_type_mask(
        self,
        categories: List[int],
        use_simd: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """Compute mask for multiple categories (OR combination).

        Args:
            categories: List of TypeCategory values to include
            use_simd: Whether to use SIMD acceleration (native only)

        Returns:
            Tuple of (mask as packed u32 array, popcount)
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy required for mask operations")

        result = np.zeros(self._mask_size, dtype=np.uint32)

        if not categories:
            return result, 0

        if self.is_native:
            cat_array = (ctypes.c_uint8 * len(categories))(*categories)
            popcount = self._lib.ananke_vocab_partition_compute_type_mask(
                self._handle,
                cat_array,
                len(categories),
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                self._mask_size,
                use_simd,
            )
        else:
            # Python fallback: OR together category masks
            popcount = 0
            for token_id in range(self._vocab_size):
                if self._categories[token_id] in categories:
                    word_idx = token_id // 32
                    bit_idx = token_id % 32
                    result[word_idx] |= 1 << bit_idx
                    popcount += 1

        return result, int(popcount)

    def to_bool_tensor(
        self,
        packed_mask: np.ndarray,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Convert packed u32 mask to boolean tensor.

        Args:
            packed_mask: Packed u32 array from get_category_mask/compute_type_mask
            device: Target PyTorch device

        Returns:
            Boolean tensor of shape (vocab_size,)
        """
        # Unpack bits
        unpacked = np.unpackbits(packed_mask.view(np.uint8), bitorder="little")
        return torch.from_numpy(unpacked[: self._vocab_size].astype(bool)).to(device)
