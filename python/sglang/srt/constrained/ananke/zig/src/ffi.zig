// Copyright Rand Arete @ Ananke 2025
// FFI exports for Python ctypes integration
//
// This module exposes the SIMD mask fusion functions to Python via ctypes.
// All exported functions use C calling convention and opaque pointers.

const std = @import("std");
const mask_fusion = @import("mask_fusion.zig");

/// Export: Fuse multiple masks using SIMD-accelerated bitwise AND
///
/// Args:
///     masks: Pointer to array of mask pointers (each pointing to u32 array)
///     mask_count: Number of masks to fuse
///     mask_size: Size of each mask in u32 elements
///     result: Output buffer (must be pre-allocated with mask_size u32s)
///
/// Returns: Number of 1-bits in result (popcount)
export fn ananke_fuse_masks(
    masks: [*]const [*]const u32,
    mask_count: usize,
    mask_size: usize,
    result: [*]u32,
) callconv(.C) u64 {
    return mask_fusion.fuseMasks(masks, mask_count, mask_size, result);
}

/// Export: Fuse masks in selectivity order with early termination
///
/// Args:
///     masks: Pointer to array of mask pointers
///     selectivities: Pre-computed popcount for each mask
///     mask_count: Number of masks
///     mask_size: Size of each mask in u32 elements
///     result: Output buffer
///     early_stop_threshold: Stop if popcount drops below this
///
/// Returns: Final popcount
export fn ananke_fuse_masks_selective(
    masks: [*]const [*]const u32,
    selectivities: [*]const u64,
    mask_count: usize,
    mask_size: usize,
    result: [*]u32,
    early_stop_threshold: u64,
) callconv(.C) u64 {
    return mask_fusion.fuseMasksSelectivityOrdered(
        masks,
        selectivities,
        mask_count,
        mask_size,
        result,
        early_stop_threshold,
    );
}

/// Export: Count 1-bits in a mask (popcount)
///
/// Args:
///     mask: Pointer to mask array
///     mask_size: Size of mask in u32 elements
///
/// Returns: Number of 1-bits
export fn ananke_popcount(
    mask: [*]const u32,
    mask_size: usize,
) callconv(.C) u64 {
    var count: u64 = 0;
    var i: usize = 0;
    while (i < mask_size) : (i += 1) {
        count += @popCount(mask[i]);
    }
    return count;
}

/// Export: Get library version
export fn ananke_native_version() callconv(.C) u32 {
    return 0x00_01_00_00; // 0.1.0.0
}
