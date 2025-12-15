// Copyright Rand Arete @ Ananke 2025
// Vocabulary partitioning for O(1) type category lookup
//
// Implements XGrammar-style vocabulary partitioning where tokens are
// pre-classified into semantic categories at initialization time.
// This enables O(1) lookup of whether a token belongs to a category,
// vs O(n) iteration through token lists.
//
// Performance target: O(1) category lookup vs O(n) iteration

const std = @import("std");

/// Token categories for type-aware masking
pub const TypeCategory = enum(u8) {
    // Literal categories
    INTEGER = 0,
    FLOAT = 1,
    STRING = 2,
    BOOLEAN = 3,
    NONE_NULL = 4,

    // Structural categories
    IDENTIFIER = 10,
    TYPE_NAME = 11,
    KEYWORD = 12,
    BUILTIN = 13,

    // Operators that produce specific types
    ARITHMETIC_OP = 20, // +, -, *, / -> numeric
    COMPARISON_OP = 21, // ==, <, > -> bool
    LOGICAL_OP = 22, // and, or, not -> bool
    BITWISE_OP = 23, // &, |, ^ -> int
    STRING_OP = 24, // + on strings, f-strings

    // Structural tokens
    BRACKET_OPEN = 30,
    BRACKET_CLOSE = 31,
    DELIMITER = 32,
    WHITESPACE = 33,

    // Special
    ANY = 254, // Matches any type
    UNKNOWN = 255,
};

/// Maximum vocabulary size supported
pub const MAX_VOCAB_SIZE: usize = 256 * 1024; // 256K tokens

/// Vocabulary partition structure
/// Pre-computed at initialization time, O(1) lookup at runtime
pub const VocabPartition = struct {
    /// Token -> category mapping (direct lookup table)
    token_categories: []TypeCategory,

    /// Category -> token mask (precomputed masks per category)
    /// Each mask is a bitset where bit i is set if token i belongs to category
    category_masks: [@typeInfo(TypeCategory).@"enum".fields.len][]u32,

    /// Vocabulary size
    vocab_size: usize,

    /// Allocator for cleanup
    allocator: std.mem.Allocator,

    /// Initialize a vocabulary partition
    pub fn init(allocator: std.mem.Allocator, vocab_size: usize) !VocabPartition {
        if (vocab_size > MAX_VOCAB_SIZE) {
            return error.VocabTooLarge;
        }

        // Allocate token -> category lookup table
        const token_categories = try allocator.alloc(TypeCategory, vocab_size);
        @memset(token_categories, .UNKNOWN);

        // Allocate category masks
        const mask_size = (vocab_size + 31) / 32; // Round up to u32 boundary
        var category_masks: [@typeInfo(TypeCategory).@"enum".fields.len][]u32 = undefined;

        for (&category_masks) |*mask| {
            mask.* = try allocator.alloc(u32, mask_size);
            @memset(mask.*, 0);
        }

        return VocabPartition{
            .token_categories = token_categories,
            .category_masks = category_masks,
            .vocab_size = vocab_size,
            .allocator = allocator,
        };
    }

    /// Free all allocated memory
    pub fn deinit(self: *VocabPartition) void {
        self.allocator.free(self.token_categories);
        for (&self.category_masks) |*mask| {
            self.allocator.free(mask.*);
        }
    }

    /// Set the category for a token (called during initialization)
    pub fn setTokenCategory(self: *VocabPartition, token_id: usize, category: TypeCategory) void {
        if (token_id >= self.vocab_size) return;

        // Update lookup table
        self.token_categories[token_id] = category;

        // Update category mask
        const word_idx = token_id / 32;
        const bit_idx: u5 = @truncate(token_id % 32);
        self.category_masks[@intFromEnum(category)][word_idx] |= (@as(u32, 1) << bit_idx);
    }

    /// O(1) lookup: get category for a token
    pub fn getCategory(self: *const VocabPartition, token_id: usize) TypeCategory {
        if (token_id >= self.vocab_size) return .UNKNOWN;
        return self.token_categories[token_id];
    }

    /// O(1) lookup: check if token belongs to category
    pub fn isInCategory(self: *const VocabPartition, token_id: usize, category: TypeCategory) bool {
        if (token_id >= self.vocab_size) return false;
        return self.token_categories[token_id] == category;
    }

    /// Get the precomputed mask for a category
    /// This mask has bit i set if token i belongs to the category
    pub fn getCategoryMask(self: *const VocabPartition, category: TypeCategory) []const u32 {
        return self.category_masks[@intFromEnum(category)];
    }

    /// Compute a type mask by OR'ing together multiple category masks
    /// This is the core operation for type-based token masking
    pub fn computeTypeMask(
        self: *const VocabPartition,
        categories: []const TypeCategory,
        result: []u32,
    ) void {
        const mask_size = (self.vocab_size + 31) / 32;
        if (result.len < mask_size) return;

        // Start with zeros
        @memset(result[0..mask_size], 0);

        // OR in each category's mask
        for (categories) |cat| {
            const cat_mask = self.category_masks[@intFromEnum(cat)];
            var i: usize = 0;
            while (i < mask_size) : (i += 1) {
                result[i] |= cat_mask[i];
            }
        }
    }

    /// SIMD-accelerated type mask computation
    pub fn computeTypeMaskSIMD(
        self: *const VocabPartition,
        categories: []const TypeCategory,
        result: []u32,
    ) void {
        const mask_size = (self.vocab_size + 31) / 32;
        if (result.len < mask_size) return;

        // Start with zeros
        @memset(result[0..mask_size], 0);

        const SIMD_WIDTH = 8; // Process 8 u32s at a time
        const simd_iterations = mask_size / SIMD_WIDTH;

        // OR in each category's mask using SIMD
        for (categories) |cat| {
            const cat_mask = self.category_masks[@intFromEnum(cat)];

            // SIMD portion
            var j: usize = 0;
            while (j < simd_iterations) : (j += 1) {
                const offset = j * SIMD_WIDTH;
                const result_vec: @Vector(SIMD_WIDTH, u32) = result[offset..][0..SIMD_WIDTH].*;
                const mask_vec: @Vector(SIMD_WIDTH, u32) = cat_mask[offset..][0..SIMD_WIDTH].*;
                result[offset..][0..SIMD_WIDTH].* = result_vec | mask_vec;
            }

            // Scalar remainder
            var k: usize = simd_iterations * SIMD_WIDTH;
            while (k < mask_size) : (k += 1) {
                result[k] |= cat_mask[k];
            }
        }
    }
};

/// Pre-defined category sets for common types
pub const TYPE_INT_CATEGORIES = [_]TypeCategory{ .INTEGER, .IDENTIFIER, .BUILTIN };
pub const TYPE_FLOAT_CATEGORIES = [_]TypeCategory{ .FLOAT, .INTEGER, .IDENTIFIER, .BUILTIN };
pub const TYPE_STR_CATEGORIES = [_]TypeCategory{ .STRING, .IDENTIFIER, .BUILTIN };
pub const TYPE_BOOL_CATEGORIES = [_]TypeCategory{ .BOOLEAN, .IDENTIFIER, .BUILTIN, .COMPARISON_OP, .LOGICAL_OP };
pub const TYPE_NONE_CATEGORIES = [_]TypeCategory{ .NONE_NULL, .IDENTIFIER };
pub const TYPE_ANY_CATEGORIES = [_]TypeCategory{.ANY};

// =============================================================================
// Tests
// =============================================================================

test "VocabPartition basic operations" {
    var partition = try VocabPartition.init(std.testing.allocator, 1000);
    defer partition.deinit();

    // Set some categories
    partition.setTokenCategory(0, .INTEGER);
    partition.setTokenCategory(1, .FLOAT);
    partition.setTokenCategory(2, .STRING);
    partition.setTokenCategory(100, .IDENTIFIER);

    // Test lookup
    try std.testing.expectEqual(TypeCategory.INTEGER, partition.getCategory(0));
    try std.testing.expectEqual(TypeCategory.FLOAT, partition.getCategory(1));
    try std.testing.expectEqual(TypeCategory.STRING, partition.getCategory(2));
    try std.testing.expectEqual(TypeCategory.IDENTIFIER, partition.getCategory(100));
    try std.testing.expectEqual(TypeCategory.UNKNOWN, partition.getCategory(999));
}

test "VocabPartition category membership" {
    var partition = try VocabPartition.init(std.testing.allocator, 100);
    defer partition.deinit();

    partition.setTokenCategory(42, .INTEGER);

    try std.testing.expect(partition.isInCategory(42, .INTEGER));
    try std.testing.expect(!partition.isInCategory(42, .FLOAT));
    try std.testing.expect(!partition.isInCategory(0, .INTEGER));
}

test "VocabPartition category mask" {
    var partition = try VocabPartition.init(std.testing.allocator, 100);
    defer partition.deinit();

    partition.setTokenCategory(0, .INTEGER);
    partition.setTokenCategory(32, .INTEGER);
    partition.setTokenCategory(64, .INTEGER);

    const mask = partition.getCategoryMask(.INTEGER);

    // Check bits are set
    try std.testing.expect((mask[0] & 1) != 0); // Token 0
    try std.testing.expect((mask[1] & 1) != 0); // Token 32
    try std.testing.expect((mask[2] & 1) != 0); // Token 64
}

test "VocabPartition compute type mask" {
    var partition = try VocabPartition.init(std.testing.allocator, 100);
    defer partition.deinit();

    partition.setTokenCategory(10, .INTEGER);
    partition.setTokenCategory(20, .FLOAT);
    partition.setTokenCategory(30, .IDENTIFIER);

    var result: [4]u32 = undefined;
    const categories = [_]TypeCategory{ .INTEGER, .FLOAT };
    partition.computeTypeMask(&categories, &result);

    // Check tokens 10 and 20 are in result
    try std.testing.expect((result[0] & (@as(u32, 1) << 10)) != 0);
    try std.testing.expect((result[0] & (@as(u32, 1) << 20)) != 0);
    // Token 30 should NOT be in result (IDENTIFIER not requested)
    try std.testing.expect((result[0] & (@as(u32, 1) << 30)) == 0);
}

test "VocabPartition SIMD type mask" {
    var partition = try VocabPartition.init(std.testing.allocator, 1000);
    defer partition.deinit();

    // Set some categories across multiple SIMD lanes
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        partition.setTokenCategory(i * 10, .INTEGER);
    }

    var result: [32]u32 = undefined;
    const categories = [_]TypeCategory{.INTEGER};
    partition.computeTypeMaskSIMD(&categories, &result);

    // Verify some expected bits
    try std.testing.expect((result[0] & 1) != 0); // Token 0
    try std.testing.expect((result[0] & (@as(u32, 1) << 10)) != 0); // Token 10
    try std.testing.expect((result[0] & (@as(u32, 1) << 20)) != 0); // Token 20
}

test "VocabPartition out of bounds" {
    var partition = try VocabPartition.init(std.testing.allocator, 100);
    defer partition.deinit();

    // Should not crash on out-of-bounds
    partition.setTokenCategory(1000, .INTEGER); // Ignored
    try std.testing.expectEqual(TypeCategory.UNKNOWN, partition.getCategory(1000));
    try std.testing.expect(!partition.isInCategory(1000, .INTEGER));
}
