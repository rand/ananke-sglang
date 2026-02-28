const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main library for mask fusion (dynamic linking)
    const lib = b.addLibrary(.{
        .name = "ananke_native",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ffi.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Install the library
    b.installArtifact(lib);

    // Unit tests for all modules
    const test_modules = [_][]const u8{
        "src/mask_fusion.zig",
        "src/classifier.zig",
        "src/vocab_partition.zig",
        "src/type_mask.zig",
        "src/constraint_prop.zig",
        "src/parallel_fusion.zig",
    };

    const test_step = b.step("test", "Run unit tests");

    for (test_modules) |module| {
        const unit_tests = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(module),
                .target = target,
                .optimize = optimize,
            }),
        });
        const run_unit_tests = b.addRunArtifact(unit_tests);
        test_step.dependOn(&run_unit_tests.step);
    }

    // Benchmark
    const bench = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });

    b.installArtifact(bench);

    const run_bench = b.addRunArtifact(bench);
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&run_bench.step);
}
