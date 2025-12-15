// Copyright Rand Arete @ Ananke 2025
// Fast constraint propagation using worklist algorithm
//
// Implements efficient constraint propagation for Sudoku-style hole filling.
// Uses a priority queue (worklist) to process constraints in optimal order.
//
// Performance target: 5x faster than Python dict-based implementation

const std = @import("std");

/// Maximum number of holes supported
pub const MAX_HOLES: usize = 256;

/// Constraint identifier
pub const ConstraintId = u32;

/// Hole identifier (index into hole array)
pub const HoleId = u16;

/// Constraint state
pub const ConstraintState = enum(u8) {
    /// Constraint has not been evaluated
    PENDING = 0,

    /// Constraint is satisfied
    SATISFIED = 1,

    /// Constraint is violated (conflict detected)
    VIOLATED = 2,

    /// Constraint is being processed
    PROCESSING = 3,
};

/// Priority level for worklist ordering
pub const Priority = enum(u8) {
    /// Immediate: constraint affects soundness
    IMMEDIATE = 0,

    /// High: constraint is highly selective
    HIGH = 1,

    /// Normal: standard constraint
    NORMAL = 2,

    /// Low: constraint is permissive
    LOW = 3,

    /// Deferred: process last
    DEFERRED = 4,
};

/// Dependency edge in the constraint graph
pub const Dependency = struct {
    /// Source hole (when this changes...)
    source: HoleId,

    /// Target constraint (...this constraint needs re-evaluation)
    target_constraint: ConstraintId,

    /// Target hole (...this hole may be affected)
    target_hole: HoleId,
};

/// Constraint descriptor
pub const Constraint = struct {
    /// Unique identifier
    id: ConstraintId,

    /// Holes involved in this constraint
    holes: [4]HoleId,
    hole_count: u8,

    /// Current state
    state: ConstraintState,

    /// Priority for worklist ordering
    priority: Priority,

    /// Number of remaining valid candidates for most constrained hole
    remaining_candidates: u32,
};

/// Worklist entry
const WorklistEntry = struct {
    constraint_id: ConstraintId,
    priority: Priority,
    remaining: u32, // For MCV ordering within priority
};

/// Compare worklist entries (lower priority first, then fewer remaining candidates)
fn compareWorklist(context: void, a: WorklistEntry, b: WorklistEntry) std.math.Order {
    _ = context;
    // First by priority (lower = more urgent)
    if (@intFromEnum(a.priority) < @intFromEnum(b.priority)) return .lt;
    if (@intFromEnum(a.priority) > @intFromEnum(b.priority)) return .gt;

    // Then by remaining candidates (fewer = more constrained)
    if (a.remaining < b.remaining) return .lt;
    if (a.remaining > b.remaining) return .gt;

    return .eq;
}

/// Maximum dependencies per hole
const MAX_DEPS_PER_HOLE: usize = 32;

/// Simple dependency list (fixed-size)
const DependencyList = struct {
    items: [MAX_DEPS_PER_HOLE]Dependency,
    len: usize,

    pub fn init() DependencyList {
        return .{
            .items = undefined,
            .len = 0,
        };
    }

    pub fn append(self: *DependencyList, dep: Dependency) void {
        if (self.len < MAX_DEPS_PER_HOLE) {
            self.items[self.len] = dep;
            self.len += 1;
        }
    }

    pub fn slice(self: *const DependencyList) []const Dependency {
        return self.items[0..self.len];
    }
};

/// Constraint propagation engine
pub const PropagationEngine = struct {
    /// All constraints
    constraints: []Constraint,
    constraint_count: usize,

    /// Dependency graph: source_hole -> list of dependencies (fixed-size)
    dependencies: []DependencyList,

    /// Worklist (priority queue)
    worklist: std.PriorityQueue(WorklistEntry, void, compareWorklist),

    /// Holes that have been filled
    filled_holes: std.bit_set.IntegerBitSet(MAX_HOLES),

    /// Values assigned to holes (indices into candidate arrays)
    hole_values: [MAX_HOLES]u32,

    /// Number of remaining candidates per hole
    remaining_per_hole: [MAX_HOLES]u32,

    /// Allocator
    allocator: std.mem.Allocator,

    /// Statistics
    propagations: u64,
    backtracks: u64,

    /// Initialize the propagation engine
    pub fn init(allocator: std.mem.Allocator, max_constraints: usize, num_holes: usize) !PropagationEngine {
        const constraints = try allocator.alloc(Constraint, max_constraints);
        @memset(constraints, Constraint{
            .id = 0,
            .holes = .{ 0, 0, 0, 0 },
            .hole_count = 0,
            .state = .PENDING,
            .priority = .NORMAL,
            .remaining_candidates = 0,
        });

        const dependencies = try allocator.alloc(DependencyList, num_holes);
        for (dependencies) |*dep_list| {
            dep_list.* = DependencyList.init();
        }

        return PropagationEngine{
            .constraints = constraints,
            .constraint_count = 0,
            .dependencies = dependencies,
            .worklist = std.PriorityQueue(WorklistEntry, void, compareWorklist).init(allocator, {}),
            .filled_holes = std.bit_set.IntegerBitSet(MAX_HOLES).initEmpty(),
            .hole_values = [_]u32{0} ** MAX_HOLES,
            .remaining_per_hole = [_]u32{std.math.maxInt(u32)} ** MAX_HOLES,
            .allocator = allocator,
            .propagations = 0,
            .backtracks = 0,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *PropagationEngine) void {
        self.allocator.free(self.constraints);
        self.allocator.free(self.dependencies);
        self.worklist.deinit();
    }

    /// Add a constraint to the engine
    pub fn addConstraint(
        self: *PropagationEngine,
        holes: []const HoleId,
        priority: Priority,
    ) ConstraintId {
        const id = @as(ConstraintId, @truncate(self.constraint_count));

        var constraint = Constraint{
            .id = id,
            .holes = .{ 0, 0, 0, 0 },
            .hole_count = @intCast(@min(holes.len, 4)),
            .state = .PENDING,
            .priority = priority,
            .remaining_candidates = std.math.maxInt(u32),
        };

        for (holes[0..constraint.hole_count], 0..) |hole, i| {
            constraint.holes[i] = hole;
        }

        self.constraints[self.constraint_count] = constraint;
        self.constraint_count += 1;

        return id;
    }

    /// Add a dependency edge
    pub fn addDependency(
        self: *PropagationEngine,
        source_hole: HoleId,
        target_constraint: ConstraintId,
        target_hole: HoleId,
    ) !void {
        if (source_hole >= self.dependencies.len) return;

        self.dependencies[source_hole].append(Dependency{
            .source = source_hole,
            .target_constraint = target_constraint,
            .target_hole = target_hole,
        });
    }

    /// Initialize remaining candidates for a hole
    pub fn setRemainingCandidates(self: *PropagationEngine, hole: HoleId, count: u32) void {
        if (hole >= MAX_HOLES) return;
        self.remaining_per_hole[hole] = count;
    }

    /// Fill a hole with a value (triggers propagation)
    pub fn fillHole(self: *PropagationEngine, hole: HoleId, value: u32) !void {
        if (hole >= MAX_HOLES) return;

        self.filled_holes.set(hole);
        self.hole_values[hole] = value;
        self.remaining_per_hole[hole] = 1;

        // Add affected constraints to worklist
        for (self.dependencies[hole].slice()) |dep| {
            const constraint = &self.constraints[dep.target_constraint];
            if (constraint.state != .VIOLATED) {
                try self.worklist.add(WorklistEntry{
                    .constraint_id = dep.target_constraint,
                    .priority = constraint.priority,
                    .remaining = constraint.remaining_candidates,
                });
            }
        }
    }

    /// Get the most constrained unfilled hole (MCV heuristic)
    pub fn getMostConstrainedHole(self: *PropagationEngine) ?HoleId {
        var min_remaining: u32 = std.math.maxInt(u32);
        var best_hole: ?HoleId = null;

        for (self.remaining_per_hole, 0..) |remaining, i| {
            if (!self.filled_holes.isSet(i) and remaining > 0 and remaining < min_remaining) {
                min_remaining = remaining;
                best_hole = @truncate(i);
            }
        }

        return best_hole;
    }

    /// Process constraints in the worklist
    ///
    /// Calls the evaluator function for each constraint to be processed.
    /// The evaluator returns the new remaining candidate count, or null if violated.
    ///
    /// Returns: true if propagation succeeded, false if conflict detected
    pub fn propagate(
        self: *PropagationEngine,
        evaluator: *const fn (ConstraintId, *PropagationEngine) ?u32,
    ) bool {
        while (self.worklist.removeOrNull()) |entry| {
            self.propagations += 1;

            const constraint = &self.constraints[entry.constraint_id];
            if (constraint.state == .VIOLATED) continue;

            constraint.state = .PROCESSING;

            // Call evaluator to check constraint
            if (evaluator(entry.constraint_id, self)) |new_remaining| {
                constraint.remaining_candidates = new_remaining;
                constraint.state = if (new_remaining > 0) .SATISFIED else .VIOLATED;

                // Update remaining counts for involved holes
                for (constraint.holes[0..constraint.hole_count]) |hole| {
                    if (!self.filled_holes.isSet(hole)) {
                        self.remaining_per_hole[hole] = @min(
                            self.remaining_per_hole[hole],
                            new_remaining,
                        );
                    }
                }

                if (new_remaining == 0) {
                    return false; // Conflict!
                }
            } else {
                constraint.state = .VIOLATED;
                return false; // Conflict!
            }
        }

        return true;
    }

    /// Create a checkpoint for backtracking
    pub fn checkpoint(self: *PropagationEngine) Checkpoint {
        return Checkpoint{
            .filled_holes = self.filled_holes,
            .hole_values = self.hole_values,
            .remaining_per_hole = self.remaining_per_hole,
            .propagations = self.propagations,
        };
    }

    /// Restore from a checkpoint
    pub fn restore(self: *PropagationEngine, cp: Checkpoint) void {
        self.filled_holes = cp.filled_holes;
        self.hole_values = cp.hole_values;
        self.remaining_per_hole = cp.remaining_per_hole;
        self.backtracks += 1;

        // Reset constraint states
        for (self.constraints[0..self.constraint_count]) |*constraint| {
            constraint.state = .PENDING;
        }

        // Clear worklist
        while (self.worklist.removeOrNull()) |_| {}
    }

    /// Get statistics
    pub fn getStats(self: *const PropagationEngine) struct { propagations: u64, backtracks: u64 } {
        return .{
            .propagations = self.propagations,
            .backtracks = self.backtracks,
        };
    }
};

/// Checkpoint for backtracking
pub const Checkpoint = struct {
    filled_holes: std.bit_set.IntegerBitSet(MAX_HOLES),
    hole_values: [MAX_HOLES]u32,
    remaining_per_hole: [MAX_HOLES]u32,
    propagations: u64,
};

// =============================================================================
// Tests
// =============================================================================

fn testEvaluator(constraint_id: ConstraintId, engine: *PropagationEngine) ?u32 {
    _ = constraint_id;
    _ = engine;
    return 5; // Always return 5 remaining candidates
}

test "PropagationEngine basic" {
    var engine = try PropagationEngine.init(std.testing.allocator, 100, 10);
    defer engine.deinit();

    // Add a constraint involving holes 0 and 1
    const c1 = engine.addConstraint(&[_]HoleId{ 0, 1 }, .NORMAL);
    try std.testing.expectEqual(@as(ConstraintId, 0), c1);

    // Set remaining candidates
    engine.setRemainingCandidates(0, 10);
    engine.setRemainingCandidates(1, 5);

    // Get most constrained hole
    const mcv = engine.getMostConstrainedHole();
    try std.testing.expectEqual(@as(?HoleId, 1), mcv);
}

test "PropagationEngine fill and propagate" {
    var engine = try PropagationEngine.init(std.testing.allocator, 100, 10);
    defer engine.deinit();

    // Add constraint
    const c1 = engine.addConstraint(&[_]HoleId{ 0, 1 }, .NORMAL);

    // Add dependency: when hole 0 changes, re-evaluate c1
    try engine.addDependency(0, c1, 1);

    // Set remaining candidates
    engine.setRemainingCandidates(0, 10);
    engine.setRemainingCandidates(1, 10);

    // Fill hole 0
    try engine.fillHole(0, 42);

    // Propagate
    const success = engine.propagate(testEvaluator);
    try std.testing.expect(success);

    // Check that propagation happened
    const stats = engine.getStats();
    try std.testing.expect(stats.propagations > 0);
}

test "PropagationEngine checkpoint/restore" {
    var engine = try PropagationEngine.init(std.testing.allocator, 100, 10);
    defer engine.deinit();

    engine.setRemainingCandidates(0, 10);

    // Create checkpoint
    const cp = engine.checkpoint();

    // Fill a hole
    try engine.fillHole(0, 42);
    try std.testing.expect(engine.filled_holes.isSet(0));
    try std.testing.expectEqual(@as(u32, 42), engine.hole_values[0]);

    // Restore
    engine.restore(cp);
    try std.testing.expect(!engine.filled_holes.isSet(0));
    try std.testing.expectEqual(@as(u32, 10), engine.remaining_per_hole[0]);
}

test "PropagationEngine MCV ordering" {
    var engine = try PropagationEngine.init(std.testing.allocator, 100, 10);
    defer engine.deinit();

    engine.setRemainingCandidates(0, 100);
    engine.setRemainingCandidates(1, 5);
    engine.setRemainingCandidates(2, 50);
    engine.setRemainingCandidates(3, 1);

    // Most constrained should be hole 3 (only 1 candidate)
    try std.testing.expectEqual(@as(?HoleId, 3), engine.getMostConstrainedHole());

    // Fill it
    try engine.fillHole(3, 0);

    // Now most constrained should be hole 1
    try std.testing.expectEqual(@as(?HoleId, 1), engine.getMostConstrainedHole());
}
