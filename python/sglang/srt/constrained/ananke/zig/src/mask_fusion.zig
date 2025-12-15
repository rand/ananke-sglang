// Copyright Rand Arete @ Ananke 2025
// SIMD-accelerated mask fusion for constrained decoding
//
// Performance target: 50x faster than Python vectorized implementation
// Uses AVX2 on x86-64, NEON on ARM64

const std = @import("std");

/// Number of elements processed per SIMD iteration
const SIMD_WIDTH = 8; // 256 bits / 32 bits per int = 8 elements

/// Fuse multiple boolean masks using bitwise AND.
///
/// Each mask is a packed u32 array where each bit represents a token's validity.
/// The result contains only tokens that are valid in ALL input masks.
///
/// Performance:
/// - Uses SIMD (AVX2/NEON) for bulk operations
/// - Processes 8 u32s (256 bits / 256 tokens) per iteration
/// - Falls back to scalar for remainder
///
/// Args:
///     masks: Array of pointers to mask arrays (each u32* is a mask)
///     mask_count: Number of masks to fuse
///     mask_size: Size of each mask in u32 elements (vocab_size / 32)
///     result: Output buffer for fused mask (must be pre-allocated)
///
/// Returns: Number of 1-bits in result (popcount), useful for selectivity
pub fn fuseMasks(
    masks: [*]const [*]const u32,
    mask_count: usize,
    mask_size: usize,
    result: [*]u32,
) u64 {
    if (mask_count == 0) {
        // No masks = all tokens valid
        @memset(result[0..mask_size], 0xFFFFFFFF);
        return @as(u64, mask_size) * 32;
    }

    if (mask_count == 1) {
        // Single mask = just copy
        @memcpy(result[0..mask_size], masks[0][0..mask_size]);
        return countBits(result, mask_size);
    }

    // Initialize result with first mask
    @memcpy(result[0..mask_size], masks[0][0..mask_size]);

    // Fuse remaining masks
    var i: usize = 1;
    while (i < mask_count) : (i += 1) {
        fuseTwoMasks(result, masks[i], mask_size);
    }

    return countBits(result, mask_size);
}

/// Fuse two masks in-place (result &= mask)
inline fn fuseTwoMasks(result: [*]u32, mask: [*]const u32, mask_size: usize) void {
    // SIMD vectorized portion
    const simd_iterations = mask_size / SIMD_WIDTH;
    var j: usize = 0;

    // Process 8 u32s at a time using SIMD
    while (j < simd_iterations) : (j += 1) {
        const offset = j * SIMD_WIDTH;
        const result_vec: @Vector(SIMD_WIDTH, u32) = result[offset..][0..SIMD_WIDTH].*;
        const mask_vec: @Vector(SIMD_WIDTH, u32) = mask[offset..][0..SIMD_WIDTH].*;
        result[offset..][0..SIMD_WIDTH].* = result_vec & mask_vec;
    }

    // Scalar remainder
    const remainder_start = simd_iterations * SIMD_WIDTH;
    var k: usize = remainder_start;
    while (k < mask_size) : (k += 1) {
        result[k] &= mask[k];
    }
}

/// Count number of 1-bits in mask (popcount)
fn countBits(mask: [*]const u32, mask_size: usize) u64 {
    var count: u64 = 0;
    var i: usize = 0;
    while (i < mask_size) : (i += 1) {
        count += @popCount(mask[i]);
    }
    return count;
}

/// Fuse masks in selectivity order (most selective first).
///
/// This optimization can enable early termination when the result
/// becomes highly constrained.
///
/// Args:
///     masks: Array of mask pointers
///     selectivities: Pre-computed selectivity (popcount) for each mask
///     mask_count: Number of masks
///     mask_size: Size of each mask
///     result: Output buffer
///     early_stop_threshold: Stop if popcount drops below this
///
/// Returns: Final popcount
pub fn fuseMasksSelectivityOrdered(
    masks: [*]const [*]const u32,
    selectivities: [*]const u64,
    mask_count: usize,
    mask_size: usize,
    result: [*]u32,
    early_stop_threshold: u64,
) u64 {
    if (mask_count == 0) {
        @memset(result[0..mask_size], 0xFFFFFFFF);
        return @as(u64, mask_size) * 32;
    }

    // Find most selective mask (lowest popcount)
    var min_idx: usize = 0;
    var min_sel = selectivities[0];
    var i: usize = 1;
    while (i < mask_count) : (i += 1) {
        if (selectivities[i] < min_sel) {
            min_sel = selectivities[i];
            min_idx = i;
        }
    }

    // Start with most selective mask
    @memcpy(result[0..mask_size], masks[min_idx][0..mask_size]);
    var current_count = min_sel;

    if (current_count <= early_stop_threshold) {
        return current_count;
    }

    // Fuse remaining masks in selectivity order
    // (Simple greedy - could be optimized with sorting)
    var processed = [_]bool{false} ** 256; // Assume max 256 masks
    processed[min_idx] = true;

    var remaining = mask_count - 1;
    while (remaining > 0) : (remaining -= 1) {
        // Find next most selective unprocessed mask
        var next_idx: usize = 0;
        var next_sel: u64 = std.math.maxInt(u64);
        var j: usize = 0;
        while (j < mask_count) : (j += 1) {
            if (!processed[j] and selectivities[j] < next_sel) {
                next_sel = selectivities[j];
                next_idx = j;
            }
        }

        processed[next_idx] = true;
        fuseTwoMasks(result, masks[next_idx], mask_size);

        // Check for early termination
        current_count = countBits(result, mask_size);
        if (current_count <= early_stop_threshold) {
            return current_count;
        }
    }

    return current_count;
}

// =============================================================================
// Tests
// =============================================================================

test "fuse empty masks returns all ones" {
    var result: [4]u32 = undefined;
    const masks: [0][*]const u32 = undefined;

    const count = fuseMasks(&masks, 0, 4, &result);

    try std.testing.expectEqual(@as(u32, 0xFFFFFFFF), result[0]);
    try std.testing.expectEqual(@as(u32, 0xFFFFFFFF), result[1]);
    try std.testing.expectEqual(@as(u64, 128), count); // 4 * 32 bits
}

test "fuse single mask copies it" {
    const mask1 = [_]u32{ 0xAAAAAAAA, 0x55555555, 0xFFFF0000, 0x0000FFFF };
    const masks = [_][*]const u32{&mask1};
    var result: [4]u32 = undefined;

    const count = fuseMasks(&masks, 1, 4, &result);

    try std.testing.expectEqual(@as(u32, 0xAAAAAAAA), result[0]);
    try std.testing.expectEqual(@as(u32, 0x55555555), result[1]);
    try std.testing.expectEqual(@as(u64, 64), count); // Half bits set
}

test "fuse two masks performs AND" {
    const mask1 = [_]u32{ 0xFFFFFFFF, 0x00000000, 0xAAAAAAAA, 0x55555555 };
    const mask2 = [_]u32{ 0x00000000, 0xFFFFFFFF, 0x55555555, 0x55555555 };
    const masks = [_][*]const u32{ &mask1, &mask2 };
    var result: [4]u32 = undefined;

    _ = fuseMasks(&masks, 2, 4, &result);

    try std.testing.expectEqual(@as(u32, 0x00000000), result[0]); // FF & 00 = 00
    try std.testing.expectEqual(@as(u32, 0x00000000), result[1]); // 00 & FF = 00
    try std.testing.expectEqual(@as(u32, 0x00000000), result[2]); // AA & 55 = 00
    try std.testing.expectEqual(@as(u32, 0x55555555), result[3]); // 55 & 55 = 55
}

test "fuse three masks" {
    const mask1 = [_]u32{ 0xFFFFFFFF, 0xFFFFFFFF };
    const mask2 = [_]u32{ 0xFF00FF00, 0x00FF00FF };
    const mask3 = [_]u32{ 0xF0F0F0F0, 0x0F0F0F0F };
    const masks = [_][*]const u32{ &mask1, &mask2, &mask3 };
    var result: [2]u32 = undefined;

    _ = fuseMasks(&masks, 3, 2, &result);

    // FF & FF00FF00 & F0F0F0F0 = F000F000
    try std.testing.expectEqual(@as(u32, 0xF000F000), result[0]);
    // FF & 00FF00FF & 0F0F0F0F = 000F000F
    try std.testing.expectEqual(@as(u32, 0x000F000F), result[1]);
}

test "popcount is correct" {
    const mask = [_]u32{ 0x00000001, 0x00000003, 0x00000007, 0x0000000F };
    const count = countBits(&mask, 4);
    try std.testing.expectEqual(@as(u64, 1 + 2 + 3 + 4), count);
}

test "simd alignment handling" {
    // Test with size not aligned to SIMD_WIDTH
    var mask1: [17]u32 = undefined;
    var mask2: [17]u32 = undefined;
    var result: [17]u32 = undefined;

    for (&mask1, 0..) |*m, i| {
        m.* = @truncate(i * 0x11111111);
    }
    for (&mask2, 0..) |*m, i| {
        m.* = @truncate((16 - i) * 0x11111111);
    }

    const masks = [_][*]const u32{ &mask1, &mask2 };
    _ = fuseMasks(&masks, 2, 17, &result);

    // Verify AND operation worked
    for (result, 0..) |r, i| {
        const expected = mask1[i] & mask2[i];
        try std.testing.expectEqual(expected, r);
    }
}
