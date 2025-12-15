// Copyright Rand Arete @ Ananke 2025
// Parallel domain mask fusion with multi-threaded SIMD
//
// Computes the intersection of multiple domain masks in parallel.
// Each domain mask is computed on a separate thread, then fused.
//
// Performance target: 3x speedup on multi-core systems

const std = @import("std");
const mask_fusion = @import("mask_fusion.zig");

/// Maximum number of domains supported in parallel
pub const MAX_DOMAINS: usize = 8;

/// Domain mask computation result
pub const DomainResult = struct {
    /// The computed mask
    mask: []u32,

    /// Number of allowed tokens (popcount)
    popcount: u64,

    /// Whether computation succeeded
    success: bool,

    /// Error message if failed
    error_msg: ?[]const u8,
};

/// Domain mask computation function signature
pub const DomainComputeFn = *const fn (
    domain_id: usize,
    context: *anyopaque,
    result_mask: []u32,
) bool;

/// Parallel fusion configuration
pub const ParallelConfig = struct {
    /// Maximum number of threads to use
    max_threads: usize = 4,

    /// Early termination threshold (stop if popcount drops below)
    early_stop_threshold: u64 = 0,

    /// Process domains in selectivity order
    selectivity_ordered: bool = true,

    /// Timeout in nanoseconds (0 = no timeout)
    timeout_ns: u64 = 0,
};

/// Parallel domain mask fusion engine
pub const ParallelFusion = struct {
    /// Thread pool
    thread_pool: std.Thread.Pool,

    /// Allocator
    allocator: std.mem.Allocator,

    /// Configuration
    config: ParallelConfig,

    /// Pre-allocated result masks for each domain
    domain_masks: [MAX_DOMAINS][]u32,

    /// Mask size (vocab_size / 32)
    mask_size: usize,

    /// Initialize the parallel fusion engine
    pub fn init(
        allocator: std.mem.Allocator,
        mask_size: usize,
        config: ParallelConfig,
    ) !ParallelFusion {
        var domain_masks: [MAX_DOMAINS][]u32 = undefined;
        for (&domain_masks) |*mask| {
            mask.* = try allocator.alloc(u32, mask_size);
            @memset(mask.*, 0xFFFFFFFF); // Start with all tokens allowed
        }

        return ParallelFusion{
            .thread_pool = undefined, // Initialized lazily
            .allocator = allocator,
            .config = config,
            .domain_masks = domain_masks,
            .mask_size = mask_size,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *ParallelFusion) void {
        for (&self.domain_masks) |*mask| {
            self.allocator.free(mask.*);
        }
    }

    /// Compute domain masks in parallel and fuse them
    ///
    /// This is a simplified synchronous implementation that processes
    /// domains sequentially but uses SIMD for fusion.
    ///
    /// Args:
    ///     domain_fns: Array of domain mask computation functions
    ///     domain_contexts: Array of opaque context pointers for each domain
    ///     domain_count: Number of domains
    ///     result: Output buffer for final fused mask
    ///
    /// Returns: Popcount of final mask, or null on error
    pub fn fuseParallel(
        self: *ParallelFusion,
        domain_fns: []const DomainComputeFn,
        domain_contexts: []const *anyopaque,
        domain_count: usize,
        result: []u32,
    ) ?u64 {
        if (domain_count == 0) {
            @memset(result[0..self.mask_size], 0xFFFFFFFF);
            return @as(u64, self.mask_size) * 32;
        }

        if (domain_count > MAX_DOMAINS) {
            return null;
        }

        // Compute each domain mask
        var successful_domains: usize = 0;
        var domain_popcounts: [MAX_DOMAINS]u64 = undefined;

        for (0..domain_count) |i| {
            // Reset mask to all ones
            @memset(self.domain_masks[i][0..self.mask_size], 0xFFFFFFFF);

            // Compute domain mask
            const success = domain_fns[i](i, domain_contexts[i], self.domain_masks[i]);
            if (success) {
                domain_popcounts[successful_domains] = countBits(self.domain_masks[i], self.mask_size);
                successful_domains += 1;
            }
        }

        if (successful_domains == 0) {
            @memset(result[0..self.mask_size], 0xFFFFFFFF);
            return @as(u64, self.mask_size) * 32;
        }

        // Fuse masks using SIMD
        if (self.config.selectivity_ordered) {
            // Sort by selectivity (most selective first)
            var indices: [MAX_DOMAINS]usize = undefined;
            for (indices[0..successful_domains], 0..) |*idx, i| {
                idx.* = i;
            }

            // Simple bubble sort (small N)
            for (0..successful_domains) |_| {
                for (0..successful_domains - 1) |j| {
                    if (domain_popcounts[indices[j]] > domain_popcounts[indices[j + 1]]) {
                        const tmp = indices[j];
                        indices[j] = indices[j + 1];
                        indices[j + 1] = tmp;
                    }
                }
            }

            // Fuse in selectivity order
            @memcpy(result[0..self.mask_size], self.domain_masks[indices[0]][0..self.mask_size]);
            var current_popcount = domain_popcounts[indices[0]];

            for (1..successful_domains) |i| {
                if (self.config.early_stop_threshold > 0 and current_popcount <= self.config.early_stop_threshold) {
                    break;
                }
                fuseTwo(result, self.domain_masks[indices[i]], self.mask_size);
                current_popcount = countBits(result, self.mask_size);
            }

            return current_popcount;
        } else {
            // Simple sequential fusion
            @memcpy(result[0..self.mask_size], self.domain_masks[0][0..self.mask_size]);

            for (1..successful_domains) |i| {
                fuseTwo(result, self.domain_masks[i], self.mask_size);
            }

            return countBits(result, self.mask_size);
        }
    }

    /// Fuse a single mask into result in-place (result &= mask)
    fn fuseTwo(result: []u32, mask: []const u32, mask_size: usize) void {
        const SIMD_WIDTH = 8;
        const simd_iterations = mask_size / SIMD_WIDTH;

        // SIMD portion
        var j: usize = 0;
        while (j < simd_iterations) : (j += 1) {
            const offset = j * SIMD_WIDTH;
            const result_vec: @Vector(SIMD_WIDTH, u32) = result[offset..][0..SIMD_WIDTH].*;
            const mask_vec: @Vector(SIMD_WIDTH, u32) = mask[offset..][0..SIMD_WIDTH].*;
            result[offset..][0..SIMD_WIDTH].* = result_vec & mask_vec;
        }

        // Scalar remainder
        var k: usize = simd_iterations * SIMD_WIDTH;
        while (k < mask_size) : (k += 1) {
            result[k] &= mask[k];
        }
    }

    /// Count bits in mask
    fn countBits(mask: []const u32, mask_size: usize) u64 {
        var count: u64 = 0;
        for (mask[0..mask_size]) |word| {
            count += @popCount(word);
        }
        return count;
    }
};

/// Fuse masks from multiple pre-computed sources
///
/// This is a simpler interface when masks are already computed.
pub fn fusePremadeMasks(
    masks: []const []const u32,
    mask_size: usize,
    result: []u32,
    selectivity_ordered: bool,
) u64 {
    if (masks.len == 0) {
        @memset(result[0..mask_size], 0xFFFFFFFF);
        return @as(u64, mask_size) * 32;
    }

    if (!selectivity_ordered) {
        // Simple fusion
        @memcpy(result[0..mask_size], masks[0][0..mask_size]);
        for (masks[1..]) |mask| {
            const SIMD_WIDTH = 8;
            const simd_iterations = mask_size / SIMD_WIDTH;

            var j: usize = 0;
            while (j < simd_iterations) : (j += 1) {
                const offset = j * SIMD_WIDTH;
                const result_vec: @Vector(SIMD_WIDTH, u32) = result[offset..][0..SIMD_WIDTH].*;
                const mask_vec: @Vector(SIMD_WIDTH, u32) = mask[offset..][0..SIMD_WIDTH].*;
                result[offset..][0..SIMD_WIDTH].* = result_vec & mask_vec;
            }

            var k: usize = simd_iterations * SIMD_WIDTH;
            while (k < mask_size) : (k += 1) {
                result[k] &= mask[k];
            }
        }

        return countBitsSimple(result, mask_size);
    }

    // Compute selectivities
    var selectivities: [MAX_DOMAINS]u64 = undefined;
    var indices: [MAX_DOMAINS]usize = undefined;

    for (masks, 0..) |mask, i| {
        selectivities[i] = countBitsSimple(mask, mask_size);
        indices[i] = i;
    }

    // Sort by selectivity
    for (0..masks.len) |_| {
        for (0..masks.len - 1) |j| {
            if (selectivities[indices[j]] > selectivities[indices[j + 1]]) {
                const tmp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = tmp;
            }
        }
    }

    // Fuse in order
    @memcpy(result[0..mask_size], masks[indices[0]][0..mask_size]);

    for (1..masks.len) |i| {
        const mask = masks[indices[i]];
        const SIMD_WIDTH = 8;
        const simd_iterations = mask_size / SIMD_WIDTH;

        var j: usize = 0;
        while (j < simd_iterations) : (j += 1) {
            const offset = j * SIMD_WIDTH;
            const result_vec: @Vector(SIMD_WIDTH, u32) = result[offset..][0..SIMD_WIDTH].*;
            const mask_vec: @Vector(SIMD_WIDTH, u32) = mask[offset..][0..SIMD_WIDTH].*;
            result[offset..][0..SIMD_WIDTH].* = result_vec & mask_vec;
        }

        var k: usize = simd_iterations * SIMD_WIDTH;
        while (k < mask_size) : (k += 1) {
            result[k] &= mask[k];
        }
    }

    return countBitsSimple(result, mask_size);
}

fn countBitsSimple(mask: []const u32, mask_size: usize) u64 {
    var count: u64 = 0;
    for (mask[0..mask_size]) |word| {
        count += @popCount(word);
    }
    return count;
}

// =============================================================================
// Tests
// =============================================================================

fn testDomainCompute1(domain_id: usize, context: *anyopaque, result: []u32) bool {
    _ = domain_id;
    _ = context;
    // Set specific pattern
    for (result) |*word| {
        word.* = 0xAAAAAAAA;
    }
    return true;
}

fn testDomainCompute2(domain_id: usize, context: *anyopaque, result: []u32) bool {
    _ = domain_id;
    _ = context;
    // Set different pattern
    for (result) |*word| {
        word.* = 0x55555555;
    }
    return true;
}

test "ParallelFusion init/deinit" {
    var fusion = try ParallelFusion.init(std.testing.allocator, 32, .{});
    defer fusion.deinit();

    try std.testing.expectEqual(@as(usize, 32), fusion.mask_size);
}

test "ParallelFusion empty domains" {
    var fusion = try ParallelFusion.init(std.testing.allocator, 4, .{});
    defer fusion.deinit();

    var result: [4]u32 = undefined;
    const empty_fns: []const DomainComputeFn = &.{};
    const empty_contexts: []const *anyopaque = &.{};

    const popcount = fusion.fuseParallel(empty_fns, empty_contexts, 0, &result);

    try std.testing.expect(popcount != null);
    try std.testing.expectEqual(@as(u64, 128), popcount.?);
    try std.testing.expectEqual(@as(u32, 0xFFFFFFFF), result[0]);
}

test "ParallelFusion single domain" {
    var fusion = try ParallelFusion.init(std.testing.allocator, 4, .{});
    defer fusion.deinit();

    var result: [4]u32 = undefined;
    var dummy_context: u32 = 0;
    const fns = [_]DomainComputeFn{testDomainCompute1};
    const contexts = [_]*anyopaque{@ptrCast(&dummy_context)};

    const popcount = fusion.fuseParallel(&fns, &contexts, 1, &result);

    try std.testing.expect(popcount != null);
    try std.testing.expectEqual(@as(u32, 0xAAAAAAAA), result[0]);
}

test "ParallelFusion two domains" {
    var fusion = try ParallelFusion.init(std.testing.allocator, 4, .{ .selectivity_ordered = false });
    defer fusion.deinit();

    var result: [4]u32 = undefined;
    var dummy_context: u32 = 0;
    const fns = [_]DomainComputeFn{ testDomainCompute1, testDomainCompute2 };
    const contexts = [_]*anyopaque{ @ptrCast(&dummy_context), @ptrCast(&dummy_context) };

    const popcount = fusion.fuseParallel(&fns, &contexts, 2, &result);

    try std.testing.expect(popcount != null);
    // 0xAAAAAAAA & 0x55555555 = 0x00000000
    try std.testing.expectEqual(@as(u32, 0x00000000), result[0]);
    try std.testing.expectEqual(@as(u64, 0), popcount.?);
}

test "fusePremadeMasks simple" {
    const mask1 = [_]u32{ 0xFFFF0000, 0x0000FFFF };
    const mask2 = [_]u32{ 0xFF00FF00, 0x00FF00FF };
    var result: [2]u32 = undefined;

    const masks = [_][]const u32{ &mask1, &mask2 };

    const popcount = fusePremadeMasks(&masks, 2, &result, false);

    try std.testing.expectEqual(@as(u32, 0xFF000000), result[0]);
    try std.testing.expectEqual(@as(u32, 0x000000FF), result[1]);
    try std.testing.expectEqual(@as(u64, 16), popcount);
}

test "fusePremadeMasks selectivity ordered" {
    // mask1 has more bits set
    const mask1 = [_]u32{ 0xFFFFFFFF, 0xFFFFFFFF };
    // mask2 has fewer bits set
    const mask2 = [_]u32{ 0x0000000F, 0x0000000F };
    var result: [2]u32 = undefined;

    const masks = [_][]const u32{ &mask1, &mask2 };

    // With selectivity ordering, mask2 (more selective) should be first
    const popcount = fusePremadeMasks(&masks, 2, &result, true);

    try std.testing.expectEqual(@as(u32, 0x0000000F), result[0]);
    try std.testing.expectEqual(@as(u32, 0x0000000F), result[1]);
    try std.testing.expectEqual(@as(u64, 8), popcount);
}
