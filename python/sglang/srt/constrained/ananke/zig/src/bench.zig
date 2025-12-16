// Copyright Rand Arete @ Ananke 2025
// Benchmark for mask fusion

const std = @import("std");
const mask_fusion = @import("mask_fusion.zig");

const VOCAB_SIZE: usize = 32768; // Typical LLM vocab size
const MASK_SIZE: usize = VOCAB_SIZE / 32; // 1024 u32s per mask
const NUM_MASKS: usize = 5; // Typical number of domain masks
const ITERATIONS: usize = 10000;

pub fn main() !void {
    // Zig 0.15 unbuffered output for simplicity
    var w = std.fs.File.stdout().writer(&.{});
    const writer = &w.interface;

    // Allocate masks
    var masks: [NUM_MASKS][MASK_SIZE]u32 = undefined;
    var mask_ptrs: [NUM_MASKS][*]const u32 = undefined;

    // Initialize with random-ish data
    var prng = std.Random.DefaultPrng.init(42);
    for (&masks, 0..) |*mask, i| {
        for (mask) |*word| {
            word.* = prng.random().int(u32);
            // Make some masks more selective
            if (i > 2) {
                word.* &= prng.random().int(u32); // More zeros
            }
        }
        mask_ptrs[i] = mask;
    }

    var result: [MASK_SIZE]u32 = undefined;

    // Warm up
    for (0..100) |_| {
        _ = mask_fusion.fuseMasks(&mask_ptrs, NUM_MASKS, MASK_SIZE, &result);
    }

    // Benchmark
    var timer = std.time.Timer.start() catch unreachable;

    for (0..ITERATIONS) |_| {
        _ = mask_fusion.fuseMasks(&mask_ptrs, NUM_MASKS, MASK_SIZE, &result);
    }

    const elapsed_ns = timer.read();
    const avg_ns = elapsed_ns / ITERATIONS;
    const avg_us = @as(f64, @floatFromInt(avg_ns)) / 1000.0;

    try writer.print("\n=== Ananke Zig SIMD Mask Fusion Benchmark ===\n", .{});
    try writer.print("Vocab size: {d}\n", .{VOCAB_SIZE});
    try writer.print("Mask size: {d} u32s ({d} KB)\n", .{ MASK_SIZE, MASK_SIZE * 4 / 1024 });
    try writer.print("Number of masks: {d}\n", .{NUM_MASKS});
    try writer.print("Iterations: {d}\n\n", .{ITERATIONS});

    try writer.print("Total time: {d:.2} ms\n", .{@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0});
    try writer.print("Average per fusion: {d:.2} us\n", .{avg_us});
    try writer.print("Throughput: {d:.0} fusions/sec\n\n", .{1_000_000_000.0 / @as(f64, @floatFromInt(avg_ns))});

    // Final popcount for verification
    const popcount = mask_fusion.fuseMasks(&mask_ptrs, NUM_MASKS, MASK_SIZE, &result);
    try writer.print("Final popcount: {d} / {d} ({d:.1}% selective)\n", .{
        popcount,
        VOCAB_SIZE,
        100.0 - @as(f64, @floatFromInt(popcount)) / @as(f64, @floatFromInt(VOCAB_SIZE)) * 100.0,
    });
}
