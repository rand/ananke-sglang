const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main library for mask fusion
    const lib = b.addSharedLibrary(.{
        .name = "ananke_native",
        .root_source_file = .{ .cwd_relative = "src/ffi.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Enable SIMD optimizations
    lib.root_module.addCMacro("__AVX2__", "1");

    // Install the library
    b.installArtifact(lib);

    // Unit tests
    const unit_tests = b.addTest(.{
        .root_source_file = .{ .cwd_relative = "src/mask_fusion.zig" },
        .target = target,
        .optimize = optimize,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Benchmark
    const bench = b.addExecutable(.{
        .name = "bench",
        .root_source_file = .{ .cwd_relative = "src/bench.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });

    b.installArtifact(bench);

    const run_bench = b.addRunArtifact(bench);
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&run_bench.step);
}
