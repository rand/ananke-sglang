// Copyright Rand Arete @ Ananke 2025
// Type-aware mask computation with SIMD optimization
//
// Computes token masks based on expected types using vocabulary partitions.
// Uses SIMD (AVX2/NEON) for bulk mask operations.
//
// Performance target: 10-20x faster than Python implementation

const std = @import("std");
const vocab_partition = @import("vocab_partition.zig");

const TypeCategory = vocab_partition.TypeCategory;
const VocabPartition = vocab_partition.VocabPartition;

/// SIMD width for bulk operations
const SIMD_WIDTH = 8;

/// Type mask configuration
pub const TypeMaskConfig = struct {
    /// Whether to allow coercion (e.g., int -> float)
    allow_coercion: bool = true,

    /// Whether to include None/null for optional types
    include_none: bool = false,

    /// Whether to allow identifiers (variables that could have the type)
    allow_identifiers: bool = true,

    /// Whether to allow builtins (functions returning the type)
    allow_builtins: bool = true,
};

/// Type specifier for mask computation
pub const TypeSpec = enum(u8) {
    // Primitive types
    INT = 0,
    FLOAT = 1,
    STR = 2,
    BOOL = 3,
    NONE = 4,

    // Container types (allow multiple categories)
    LIST = 10,
    DICT = 11,
    SET = 12,
    TUPLE = 13,

    // Special types
    ANY = 254,
    UNKNOWN = 255,
};

/// Compute a type-aware token mask
///
/// Given an expected type, computes a mask of all tokens that could
/// produce a value of that type.
///
/// Args:
///     partition: Pre-computed vocabulary partition
///     type_spec: Expected type
///     config: Configuration options
///     result: Output buffer for mask (must be vocab_size/32 u32s)
///
/// Performance: O(n) where n = vocab_size/32, SIMD-accelerated
pub fn computeTypeMask(
    partition: *const VocabPartition,
    type_spec: TypeSpec,
    config: TypeMaskConfig,
    result: []u32,
) void {
    const categories = getCategoriesForType(type_spec, config);
    partition.computeTypeMaskSIMD(categories, result);
}

/// Get the categories that can produce a given type
fn getCategoriesForType(type_spec: TypeSpec, config: TypeMaskConfig) []const TypeCategory {
    return switch (type_spec) {
        .INT => if (config.allow_coercion)
            &[_]TypeCategory{ .INTEGER, .IDENTIFIER, .BUILTIN, .ARITHMETIC_OP, .BITWISE_OP }
        else
            &[_]TypeCategory{ .INTEGER, .IDENTIFIER },

        .FLOAT => if (config.allow_coercion)
            &[_]TypeCategory{ .FLOAT, .INTEGER, .IDENTIFIER, .BUILTIN, .ARITHMETIC_OP }
        else
            &[_]TypeCategory{ .FLOAT, .IDENTIFIER },

        .STR => &[_]TypeCategory{ .STRING, .IDENTIFIER, .BUILTIN, .STRING_OP },

        .BOOL => &[_]TypeCategory{ .BOOLEAN, .IDENTIFIER, .COMPARISON_OP, .LOGICAL_OP },

        .NONE => &[_]TypeCategory{ .NONE_NULL, .IDENTIFIER },

        .LIST, .DICT, .SET, .TUPLE => &[_]TypeCategory{ .BRACKET_OPEN, .IDENTIFIER, .BUILTIN },

        .ANY => &[_]TypeCategory{.ANY},

        .UNKNOWN => &[_]TypeCategory{.UNKNOWN},
    };
}

/// Compute intersection of two type masks (AND operation)
///
/// result = mask1 AND mask2
///
/// Performance: SIMD-accelerated, O(n/8) operations where n = mask_size
pub fn intersectTypeMasks(
    mask1: []const u32,
    mask2: []const u32,
    mask_size: usize,
    result: []u32,
) void {
    const simd_iterations = mask_size / SIMD_WIDTH;

    // SIMD portion
    var j: usize = 0;
    while (j < simd_iterations) : (j += 1) {
        const offset = j * SIMD_WIDTH;
        const vec1: @Vector(SIMD_WIDTH, u32) = mask1[offset..][0..SIMD_WIDTH].*;
        const vec2: @Vector(SIMD_WIDTH, u32) = mask2[offset..][0..SIMD_WIDTH].*;
        result[offset..][0..SIMD_WIDTH].* = vec1 & vec2;
    }

    // Scalar remainder
    var k: usize = simd_iterations * SIMD_WIDTH;
    while (k < mask_size) : (k += 1) {
        result[k] = mask1[k] & mask2[k];
    }
}

/// Compute union of two type masks (OR operation)
///
/// result = mask1 OR mask2
///
/// Performance: SIMD-accelerated
pub fn unionTypeMasks(
    mask1: []const u32,
    mask2: []const u32,
    mask_size: usize,
    result: []u32,
) void {
    const simd_iterations = mask_size / SIMD_WIDTH;

    // SIMD portion
    var j: usize = 0;
    while (j < simd_iterations) : (j += 1) {
        const offset = j * SIMD_WIDTH;
        const vec1: @Vector(SIMD_WIDTH, u32) = mask1[offset..][0..SIMD_WIDTH].*;
        const vec2: @Vector(SIMD_WIDTH, u32) = mask2[offset..][0..SIMD_WIDTH].*;
        result[offset..][0..SIMD_WIDTH].* = vec1 | vec2;
    }

    // Scalar remainder
    var k: usize = simd_iterations * SIMD_WIDTH;
    while (k < mask_size) : (k += 1) {
        result[k] = mask1[k] | mask2[k];
    }
}

/// Invert a type mask (NOT operation)
///
/// result = NOT mask
pub fn invertTypeMask(
    mask: []const u32,
    mask_size: usize,
    result: []u32,
) void {
    const simd_iterations = mask_size / SIMD_WIDTH;

    // SIMD portion
    var j: usize = 0;
    while (j < simd_iterations) : (j += 1) {
        const offset = j * SIMD_WIDTH;
        const vec: @Vector(SIMD_WIDTH, u32) = mask[offset..][0..SIMD_WIDTH].*;
        result[offset..][0..SIMD_WIDTH].* = ~vec;
    }

    // Scalar remainder
    var k: usize = simd_iterations * SIMD_WIDTH;
    while (k < mask_size) : (k += 1) {
        result[k] = ~mask[k];
    }
}

/// Count tokens allowed by a mask (popcount)
pub fn countAllowedTokens(mask: []const u32, mask_size: usize) u64 {
    var count: u64 = 0;
    var i: usize = 0;
    while (i < mask_size) : (i += 1) {
        count += @popCount(mask[i]);
    }
    return count;
}

/// Check if a specific token is allowed by a mask
pub fn isTokenAllowed(mask: []const u32, token_id: usize) bool {
    const word_idx = token_id / 32;
    const bit_idx: u5 = @truncate(token_id % 32);
    if (word_idx >= mask.len) return false;
    return (mask[word_idx] & (@as(u32, 1) << bit_idx)) != 0;
}

/// Set a token as allowed in a mask
pub fn allowToken(mask: []u32, token_id: usize) void {
    const word_idx = token_id / 32;
    const bit_idx: u5 = @truncate(token_id % 32);
    if (word_idx >= mask.len) return;
    mask[word_idx] |= (@as(u32, 1) << bit_idx);
}

/// Set a token as disallowed in a mask
pub fn disallowToken(mask: []u32, token_id: usize) void {
    const word_idx = token_id / 32;
    const bit_idx: u5 = @truncate(token_id % 32);
    if (word_idx >= mask.len) return;
    mask[word_idx] &= ~(@as(u32, 1) << bit_idx);
}

// =============================================================================
// Type Compatibility Functions
// =============================================================================

/// Check if source_type is assignable to target_type
pub fn isAssignable(source_type: TypeSpec, target_type: TypeSpec) bool {
    if (target_type == .ANY) return true;
    if (source_type == target_type) return true;

    // Coercion rules
    return switch (target_type) {
        .FLOAT => source_type == .INT, // int -> float
        .STR => false, // No implicit string conversion
        .BOOL => false, // No implicit bool conversion
        else => false,
    };
}

/// Get the meet (intersection) of two types
pub fn typeMeet(t1: TypeSpec, t2: TypeSpec) TypeSpec {
    if (t1 == .ANY) return t2;
    if (t2 == .ANY) return t1;
    if (t1 == t2) return t1;

    // Handle coercion
    if (t1 == .INT and t2 == .FLOAT) return .INT;
    if (t1 == .FLOAT and t2 == .INT) return .INT;

    return .UNKNOWN; // No common type
}

/// Get the join (union) of two types
pub fn typeJoin(t1: TypeSpec, t2: TypeSpec) TypeSpec {
    if (t1 == .UNKNOWN) return t2;
    if (t2 == .UNKNOWN) return t1;
    if (t1 == t2) return t1;

    // Handle widening
    if ((t1 == .INT and t2 == .FLOAT) or (t1 == .FLOAT and t2 == .INT)) {
        return .FLOAT;
    }

    return .ANY; // Least upper bound
}

// =============================================================================
// Tests
// =============================================================================

test "isTokenAllowed and allowToken" {
    var mask = [_]u32{ 0, 0, 0, 0 };

    try std.testing.expect(!isTokenAllowed(&mask, 0));
    try std.testing.expect(!isTokenAllowed(&mask, 42));

    allowToken(&mask, 0);
    allowToken(&mask, 42);
    allowToken(&mask, 64);

    try std.testing.expect(isTokenAllowed(&mask, 0));
    try std.testing.expect(isTokenAllowed(&mask, 42));
    try std.testing.expect(isTokenAllowed(&mask, 64));
    try std.testing.expect(!isTokenAllowed(&mask, 1));
}

test "disallowToken" {
    var mask = [_]u32{ 0xFFFFFFFF, 0xFFFFFFFF };

    try std.testing.expect(isTokenAllowed(&mask, 10));
    disallowToken(&mask, 10);
    try std.testing.expect(!isTokenAllowed(&mask, 10));
}

test "intersectTypeMasks" {
    const mask1 = [_]u32{ 0xFFFF0000, 0x0000FFFF };
    const mask2 = [_]u32{ 0xFF00FF00, 0x00FF00FF };
    var result: [2]u32 = undefined;

    intersectTypeMasks(&mask1, &mask2, 2, &result);

    try std.testing.expectEqual(@as(u32, 0xFF000000), result[0]);
    try std.testing.expectEqual(@as(u32, 0x000000FF), result[1]);
}

test "unionTypeMasks" {
    const mask1 = [_]u32{ 0xF0000000, 0x0000000F };
    const mask2 = [_]u32{ 0x0F000000, 0x000000F0 };
    var result: [2]u32 = undefined;

    unionTypeMasks(&mask1, &mask2, 2, &result);

    try std.testing.expectEqual(@as(u32, 0xFF000000), result[0]);
    try std.testing.expectEqual(@as(u32, 0x000000FF), result[1]);
}

test "invertTypeMask" {
    const mask = [_]u32{ 0xF0F0F0F0, 0x0F0F0F0F };
    var result: [2]u32 = undefined;

    invertTypeMask(&mask, 2, &result);

    try std.testing.expectEqual(@as(u32, 0x0F0F0F0F), result[0]);
    try std.testing.expectEqual(@as(u32, 0xF0F0F0F0), result[1]);
}

test "countAllowedTokens" {
    const mask = [_]u32{ 0x00000001, 0x00000003, 0x00000007 };
    const count = countAllowedTokens(&mask, 3);
    try std.testing.expectEqual(@as(u64, 1 + 2 + 3), count);
}

test "isAssignable" {
    // Same type
    try std.testing.expect(isAssignable(.INT, .INT));
    try std.testing.expect(isAssignable(.STR, .STR));

    // ANY accepts all
    try std.testing.expect(isAssignable(.INT, .ANY));
    try std.testing.expect(isAssignable(.STR, .ANY));

    // Coercion
    try std.testing.expect(isAssignable(.INT, .FLOAT));
    try std.testing.expect(!isAssignable(.FLOAT, .INT));
    try std.testing.expect(!isAssignable(.INT, .STR));
}

test "typeMeet" {
    try std.testing.expectEqual(TypeSpec.INT, typeMeet(.INT, .INT));
    try std.testing.expectEqual(TypeSpec.INT, typeMeet(.INT, .ANY));
    try std.testing.expectEqual(TypeSpec.INT, typeMeet(.ANY, .INT));
    try std.testing.expectEqual(TypeSpec.INT, typeMeet(.INT, .FLOAT));
    try std.testing.expectEqual(TypeSpec.UNKNOWN, typeMeet(.INT, .STR));
}

test "typeJoin" {
    try std.testing.expectEqual(TypeSpec.INT, typeJoin(.INT, .INT));
    try std.testing.expectEqual(TypeSpec.INT, typeJoin(.INT, .UNKNOWN));
    try std.testing.expectEqual(TypeSpec.FLOAT, typeJoin(.INT, .FLOAT));
    try std.testing.expectEqual(TypeSpec.ANY, typeJoin(.INT, .STR));
}
