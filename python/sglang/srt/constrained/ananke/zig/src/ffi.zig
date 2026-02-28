// Copyright Rand Arete @ Ananke 2025
// FFI exports for Python ctypes integration
//
// This module exposes the SIMD mask fusion functions to Python via ctypes.
// All exported functions use C calling convention and opaque pointers.

const std = @import("std");
const mask_fusion = @import("mask_fusion.zig");
const classifier = @import("classifier.zig");
const vocab_partition = @import("vocab_partition.zig");
const type_mask = @import("type_mask.zig");
const constraint_prop = @import("constraint_prop.zig");
const parallel_fusion = @import("parallel_fusion.zig");

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
) callconv(.c) u64 {
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
) callconv(.c) u64 {
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
) callconv(.c) u64 {
    var count: u64 = 0;
    var i: usize = 0;
    while (i < mask_size) : (i += 1) {
        count += @popCount(mask[i]);
    }
    return count;
}

/// Export: Get library version
export fn ananke_native_version() callconv(.c) u32 {
    return 0x00_01_00_00; // 0.1.0.0
}

// =============================================================================
// Token Classifier FFI
// =============================================================================

/// Export: Classify a single token
///
/// Args:
///     str: Pointer to token string (not null-terminated)
///     len: Length of token string
///     lang: Language enum value (0=Python, 1=TypeScript, etc.)
///
/// Returns: TokenCategory enum value
export fn ananke_classify_token(
    str: [*]const u8,
    len: usize,
    lang: u8,
) callconv(.c) u8 {
    const language = @as(classifier.Language, @enumFromInt(lang));
    const category = classifier.classifyToken(str[0..len], language);
    return @intFromEnum(category);
}

/// Export: Batch classify multiple tokens
///
/// Args:
///     tokens: Array of token string pointers
///     lengths: Array of token lengths
///     count: Number of tokens
///     lang: Language enum value
///     results: Output array for categories (must be pre-allocated)
export fn ananke_classify_tokens_batch(
    tokens: [*]const [*]const u8,
    lengths: [*]const usize,
    count: usize,
    lang: u8,
    results: [*]u8,
) callconv(.c) void {
    const language = @as(classifier.Language, @enumFromInt(lang));
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const category = classifier.classifyToken(tokens[i][0..lengths[i]], language);
        results[i] = @intFromEnum(category);
    }
}

/// Export: Check if token is a Python keyword
export fn ananke_is_python_keyword(
    str: [*]const u8,
    len: usize,
) callconv(.c) bool {
    return classifier.isPythonKeyword(str[0..len]);
}

/// Export: Check if token is a Python builtin
export fn ananke_is_python_builtin(
    str: [*]const u8,
    len: usize,
) callconv(.c) bool {
    return classifier.isPythonBuiltin(str[0..len]);
}

/// Export: Check if token is a TypeScript keyword
export fn ananke_is_typescript_keyword(
    str: [*]const u8,
    len: usize,
) callconv(.c) bool {
    return classifier.isTypeScriptKeyword(str[0..len]);
}

/// Export: Check if token is a Go keyword
export fn ananke_is_go_keyword(
    str: [*]const u8,
    len: usize,
) callconv(.c) bool {
    return classifier.isGoKeyword(str[0..len]);
}

/// Export: Check if token is a Rust keyword
export fn ananke_is_rust_keyword(
    str: [*]const u8,
    len: usize,
) callconv(.c) bool {
    return classifier.isRustKeyword(str[0..len]);
}

/// Export: Check if token is a Kotlin keyword
export fn ananke_is_kotlin_keyword(
    str: [*]const u8,
    len: usize,
) callconv(.c) bool {
    return classifier.isKotlinKeyword(str[0..len]);
}

/// Export: Check if token is a Swift keyword
export fn ananke_is_swift_keyword(
    str: [*]const u8,
    len: usize,
) callconv(.c) bool {
    return classifier.isSwiftKeyword(str[0..len]);
}

/// Export: Check if token is a Zig keyword
export fn ananke_is_zig_keyword(
    str: [*]const u8,
    len: usize,
) callconv(.c) bool {
    return classifier.isZigKeyword(str[0..len]);
}

// =============================================================================
// Type Mask FFI
// =============================================================================

/// Export: Compute type mask intersection
export fn ananke_intersect_type_masks(
    mask1: [*]const u32,
    mask2: [*]const u32,
    mask_size: usize,
    result: [*]u32,
) callconv(.c) void {
    type_mask.intersectTypeMasks(mask1[0..mask_size], mask2[0..mask_size], mask_size, result[0..mask_size]);
}

/// Export: Compute type mask union
export fn ananke_union_type_masks(
    mask1: [*]const u32,
    mask2: [*]const u32,
    mask_size: usize,
    result: [*]u32,
) callconv(.c) void {
    type_mask.unionTypeMasks(mask1[0..mask_size], mask2[0..mask_size], mask_size, result[0..mask_size]);
}

/// Export: Invert type mask
export fn ananke_invert_type_mask(
    mask: [*]const u32,
    mask_size: usize,
    result: [*]u32,
) callconv(.c) void {
    type_mask.invertTypeMask(mask[0..mask_size], mask_size, result[0..mask_size]);
}

/// Export: Count allowed tokens in mask
export fn ananke_count_allowed_tokens(
    mask: [*]const u32,
    mask_size: usize,
) callconv(.c) u64 {
    return type_mask.countAllowedTokens(mask[0..mask_size], mask_size);
}

/// Export: Check if token is allowed in mask
export fn ananke_is_token_allowed(
    mask: [*]const u32,
    mask_size: usize,
    token_id: usize,
) callconv(.c) bool {
    if (token_id / 32 >= mask_size) return false;
    return type_mask.isTokenAllowed(mask[0..mask_size], token_id);
}

/// Export: Allow a token in mask
export fn ananke_allow_token(
    mask: [*]u32,
    mask_size: usize,
    token_id: usize,
) callconv(.c) void {
    if (token_id / 32 >= mask_size) return;
    type_mask.allowToken(mask[0..mask_size], token_id);
}

/// Export: Disallow a token in mask
export fn ananke_disallow_token(
    mask: [*]u32,
    mask_size: usize,
    token_id: usize,
) callconv(.c) void {
    if (token_id / 32 >= mask_size) return;
    type_mask.disallowToken(mask[0..mask_size], token_id);
}

/// Export: Check type assignability
export fn ananke_is_type_assignable(
    source_type: u8,
    target_type: u8,
) callconv(.c) bool {
    return type_mask.isAssignable(
        @as(type_mask.TypeSpec, @enumFromInt(source_type)),
        @as(type_mask.TypeSpec, @enumFromInt(target_type)),
    );
}

/// Export: Compute type meet
export fn ananke_type_meet(
    t1: u8,
    t2: u8,
) callconv(.c) u8 {
    return @intFromEnum(type_mask.typeMeet(
        @as(type_mask.TypeSpec, @enumFromInt(t1)),
        @as(type_mask.TypeSpec, @enumFromInt(t2)),
    ));
}

/// Export: Compute type join
export fn ananke_type_join(
    t1: u8,
    t2: u8,
) callconv(.c) u8 {
    return @intFromEnum(type_mask.typeJoin(
        @as(type_mask.TypeSpec, @enumFromInt(t1)),
        @as(type_mask.TypeSpec, @enumFromInt(t2)),
    ));
}

// =============================================================================
// Parallel Fusion FFI
// =============================================================================

/// Export: Fuse pre-computed masks with optional selectivity ordering
export fn ananke_fuse_premade_masks(
    masks: [*]const [*]const u32,
    mask_count: usize,
    mask_size: usize,
    result: [*]u32,
    selectivity_ordered: bool,
) callconv(.c) u64 {
    if (mask_count == 0) {
        @memset(result[0..mask_size], 0xFFFFFFFF);
        return @as(u64, mask_size) * 32;
    }

    // Convert to slice of slices
    var mask_slices: [parallel_fusion.MAX_DOMAINS][]const u32 = undefined;
    const count = @min(mask_count, parallel_fusion.MAX_DOMAINS);
    for (0..count) |i| {
        mask_slices[i] = masks[i][0..mask_size];
    }

    return parallel_fusion.fusePremadeMasks(
        mask_slices[0..count],
        mask_size,
        result[0..mask_size],
        selectivity_ordered,
    );
}

// =============================================================================
// VocabPartition FFI - Full vocabulary classification
// =============================================================================

/// Opaque handle to VocabPartition
const VocabPartitionHandle = *vocab_partition.VocabPartition;

/// Global allocator for FFI allocations
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const ffi_allocator = gpa.allocator();

/// Export: Create a new vocabulary partition
///
/// Args:
///     vocab_size: Number of tokens in vocabulary
///
/// Returns: Opaque handle to VocabPartition, or null on failure
export fn ananke_vocab_partition_create(
    vocab_size: usize,
) callconv(.c) ?*anyopaque {
    const partition = ffi_allocator.create(vocab_partition.VocabPartition) catch return null;
    partition.* = vocab_partition.VocabPartition.init(ffi_allocator, vocab_size) catch {
        ffi_allocator.destroy(partition);
        return null;
    };
    return @ptrCast(partition);
}

/// Export: Destroy a vocabulary partition and free memory
///
/// Args:
///     handle: Opaque handle from ananke_vocab_partition_create
export fn ananke_vocab_partition_destroy(
    handle: ?*anyopaque,
) callconv(.c) void {
    if (handle) |h| {
        const partition: *vocab_partition.VocabPartition = @ptrCast(@alignCast(h));
        partition.deinit();
        ffi_allocator.destroy(partition);
    }
}

/// Export: Set the category for a single token
///
/// Args:
///     handle: VocabPartition handle
///     token_id: Token index
///     category: Category enum value (from classifier.TokenCategory)
export fn ananke_vocab_partition_set_category(
    handle: ?*anyopaque,
    token_id: usize,
    category: u8,
) callconv(.c) void {
    if (handle) |h| {
        const partition: *vocab_partition.VocabPartition = @ptrCast(@alignCast(h));
        // Map classifier.TokenCategory to vocab_partition.TypeCategory
        // They have different enum values, so we need to translate
        const type_cat = mapClassifierToTypeCategory(category);
        partition.setTokenCategory(token_id, type_cat);
    }
}

/// Map classifier.TokenCategory (u8) to vocab_partition.TypeCategory
fn mapClassifierToTypeCategory(cat: u8) vocab_partition.TypeCategory {
    return switch (cat) {
        // Literals
        0 => .INTEGER, // INT_LITERAL
        1 => .FLOAT, // FLOAT_LITERAL
        2 => .STRING, // STRING_LITERAL
        3 => .BOOLEAN, // BOOL_LITERAL
        4 => .NONE_NULL, // NONE_LITERAL
        // Names and keywords
        10 => .IDENTIFIER, // IDENTIFIER
        11 => .KEYWORD, // KEYWORD
        12 => .BUILTIN, // BUILTIN
        13 => .TYPE_NAME, // TYPE_NAME
        // Operators and punctuation
        20 => .ARITHMETIC_OP, // OPERATOR
        21 => .DELIMITER, // DELIMITER
        22 => .BRACKET_OPEN, // BRACKET (we'll use BRACKET_OPEN)
        // Whitespace and structure
        30 => .WHITESPACE, // WHITESPACE
        31 => .WHITESPACE, // NEWLINE -> WHITESPACE
        34 => .WHITESPACE, // COMMENT -> WHITESPACE (excluded from most masks)
        else => .UNKNOWN,
    };
}

/// Export: Classify a batch of tokens and populate the partition
///
/// This is the main entry point for vocabulary-wide classification.
/// Call this once with all decoded token strings from the tokenizer.
///
/// Args:
///     handle: VocabPartition handle
///     token_ptrs: Array of pointers to token strings (not null-terminated)
///     token_lengths: Array of token string lengths
///     token_count: Number of tokens
///     language: Language enum value (0=Python, 1=TypeScript, etc.)
export fn ananke_vocab_partition_classify_all(
    handle: ?*anyopaque,
    token_ptrs: [*]const [*]const u8,
    token_lengths: [*]const usize,
    token_count: usize,
    language: u8,
) callconv(.c) void {
    if (handle == null) return;

    const partition: *vocab_partition.VocabPartition = @ptrCast(@alignCast(handle.?));
    const lang = @as(classifier.Language, @enumFromInt(language));

    // Classify each token and set its category
    var i: usize = 0;
    while (i < token_count) : (i += 1) {
        const token_str = token_ptrs[i][0..token_lengths[i]];
        const cat = classifier.classifyToken(token_str, lang);
        const type_cat = mapClassifierToTypeCategory(@intFromEnum(cat));
        partition.setTokenCategory(i, type_cat);
    }
}

/// Export: Get the precomputed mask for a category
///
/// Returns a bitset where bit i is 1 if token i belongs to the category.
///
/// Args:
///     handle: VocabPartition handle
///     category: TypeCategory enum value (from vocab_partition.TypeCategory)
///     result: Output buffer (must be pre-allocated with ceiling(vocab_size/32) u32s)
///     result_size: Size of result buffer in u32 elements
///
/// Returns: Number of tokens in category (popcount)
export fn ananke_vocab_partition_get_category_mask(
    handle: ?*anyopaque,
    category: u8,
    result: [*]u32,
    result_size: usize,
) callconv(.c) u64 {
    if (handle == null) return 0;

    const partition: *vocab_partition.VocabPartition = @ptrCast(@alignCast(handle.?));
    const type_cat = @as(vocab_partition.TypeCategory, @enumFromInt(category));
    const mask = partition.getCategoryMask(type_cat);

    // Copy mask to result buffer
    const copy_size = @min(mask.len, result_size);
    @memcpy(result[0..copy_size], mask[0..copy_size]);

    // Count set bits
    var count: u64 = 0;
    for (mask) |word| {
        count += @popCount(word);
    }
    return count;
}

/// Export: Compute a type mask for multiple categories combined
///
/// This is the main entry point for computing token masks from type constraints.
/// Combines multiple categories with OR to create the allowed token set.
///
/// Args:
///     handle: VocabPartition handle
///     categories: Array of TypeCategory enum values
///     category_count: Number of categories
///     result: Output buffer
///     result_size: Size of result buffer in u32 elements
///     use_simd: Whether to use SIMD acceleration
///
/// Returns: Number of allowed tokens (popcount)
export fn ananke_vocab_partition_compute_type_mask(
    handle: ?*anyopaque,
    categories: [*]const u8,
    category_count: usize,
    result: [*]u32,
    result_size: usize,
    use_simd: bool,
) callconv(.c) u64 {
    if (handle == null or category_count == 0) {
        // No categories = allow nothing
        @memset(result[0..result_size], 0);
        return 0;
    }

    const partition: *vocab_partition.VocabPartition = @ptrCast(@alignCast(handle.?));

    // Convert u8 array to TypeCategory array
    var type_cats: [32]vocab_partition.TypeCategory = undefined;
    const count = @min(category_count, 32);
    for (0..count) |i| {
        type_cats[i] = @as(vocab_partition.TypeCategory, @enumFromInt(categories[i]));
    }

    // Compute combined mask
    if (use_simd) {
        partition.computeTypeMaskSIMD(type_cats[0..count], result[0..result_size]);
    } else {
        partition.computeTypeMask(type_cats[0..count], result[0..result_size]);
    }

    // Count set bits
    var popcount: u64 = 0;
    for (result[0..result_size]) |word| {
        popcount += @popCount(word);
    }
    return popcount;
}

/// Export: Get vocabulary size of partition
export fn ananke_vocab_partition_get_vocab_size(
    handle: ?*anyopaque,
) callconv(.c) usize {
    if (handle == null) return 0;
    const partition: *vocab_partition.VocabPartition = @ptrCast(@alignCast(handle.?));
    return partition.vocab_size;
}

/// Export: Get category for a single token
export fn ananke_vocab_partition_get_category(
    handle: ?*anyopaque,
    token_id: usize,
) callconv(.c) u8 {
    if (handle == null) return 255; // UNKNOWN
    const partition: *vocab_partition.VocabPartition = @ptrCast(@alignCast(handle.?));
    return @intFromEnum(partition.getCategory(token_id));
}

/// Export: Check if a token belongs to a category
export fn ananke_vocab_partition_is_in_category(
    handle: ?*anyopaque,
    token_id: usize,
    category: u8,
) callconv(.c) bool {
    if (handle == null) return false;
    const partition: *vocab_partition.VocabPartition = @ptrCast(@alignCast(handle.?));
    const type_cat = @as(vocab_partition.TypeCategory, @enumFromInt(category));
    return partition.isInCategory(token_id, type_cat);
}
