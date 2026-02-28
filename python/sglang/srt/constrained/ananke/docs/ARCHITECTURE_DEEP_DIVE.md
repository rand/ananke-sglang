# Ananke Architecture Deep Dive

This document covers the mathematical foundations, constraint algebra, and implementation details of the Ananke system.

For a high-level overview, see [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## Table of Contents

1. [Mathematical Foundations](#mathematical-foundations)
2. [Constraint Algebra](#constraint-algebra)
3. [Domain Implementations](#domain-implementations)
4. [Cross-Domain Propagation](#cross-domain-propagation)
5. [Typed Holes System](#typed-holes-system)
6. [Incremental Type Checking](#incremental-type-checking)
7. [Zig SIMD Implementation](#zig-simd-implementation)
8. [Performance Analysis](#performance-analysis)

---

## Mathematical Foundations

### The Core Insight

Constrained code generation is fundamentally a **constraint satisfaction problem with incremental observation**. Each generated token is both:

1. An **observation** that narrows the solution space
2. A **decision** that must satisfy all active constraints

This dual nature suggests modeling the system as a constraint propagation network where token generation interleaves with constraint solving.

### Bounded Meet-Semilattice

Constraints are modeled as elements of a **bounded meet-semilattice** ⟨C, ⊓, ⊤, ⊥⟩:

- **C** is the set of constraints
- **⊓** (meet) is constraint conjunction: `c₁ ⊓ c₂` means "both c₁ and c₂ must hold"
- **⊤** (top) is the trivial constraint: always satisfied
- **⊥** (bottom) is the absurd constraint: never satisfied

**Algebraic Properties:**

```
c ⊓ ⊤ = c                        (identity)
c ⊓ ⊥ = ⊥                        (annihilation)
c ⊓ c = c                        (idempotence)
c₁ ⊓ c₂ = c₂ ⊓ c₁                (commutativity)
(c₁ ⊓ c₂) ⊓ c₃ = c₁ ⊓ (c₂ ⊓ c₃)  (associativity)
```

The partial order `c₁ ⊑ c₂` ("c₁ is at least as constraining as c₂") is defined as `c₁ ⊓ c₂ = c₁`.

### Why a Semilattice?

The semilattice structure provides critical guarantees:

| Property | Meaning | Benefit |
|----------|---------|---------|
| Identity | Adding ⊤ is a no-op | Unconstrained domains don't interfere |
| Annihilation | Contradiction always propagates | Errors detected immediately |
| Idempotence | Re-applying constraint is safe | Enables caching |
| Commutativity | Order doesn't matter | Parallel evaluation |
| Associativity | Grouping doesn't matter | Incremental combination |

### Constraint Domains as Functors

Each constraint domain D (syntax, types, imports, control flow, semantics) is modeled as a **functor** from program contexts to constraint semilattices:

```
D : Context → ConstraintSemilattice
```

This captures that the same constraint may have different meanings in different contexts.

### Cross-Domain Morphisms

Domains are connected by **constraint morphisms** that translate constraints between domains:

```
φ : D₁(Γ) → D₂(Γ)
```

These morphisms must be:
- **Monotonic**: `c₁ ⊑ c₂ ⟹ φ(c₁) ⊑ φ(c₂)` (preserves partial order)
- **⊥-preserving**: `φ(⊥) = ⊥` (contradictions propagate)

---

## Constraint Algebra

### Base Constraint Interface

```python
class Constraint(ABC, Generic[C]):
    """Abstract base for bounded meet-semilattice constraints."""

    @abstractmethod
    def meet(self, other: C) -> C:
        """Compute the greatest lower bound (conjunction)."""
        ...

    @abstractmethod
    def satisfiability(self) -> Satisfiability:
        """Check if this constraint is satisfiable."""
        ...

    @abstractmethod
    def is_top(self) -> bool:
        """Is this the trivial (unconstrained) element?"""
        ...

    @abstractmethod
    def is_bottom(self) -> bool:
        """Is this the absurd (unsatisfiable) element?"""
        ...

    def __le__(self, other: Constraint) -> bool:
        """Partial order: self ⊑ other iff self ⊓ other = self"""
        return self.meet(other) == self
```

### Satisfiability Trichotomy

```python
class Satisfiability(Enum):
    SAT = auto()      # Definitely satisfiable (at least one solution)
    UNSAT = auto()    # Definitely unsatisfiable (no solution)
    UNKNOWN = auto()  # Undecidable or timeout
```

Conservative composition:
- `SAT & UNSAT = UNSAT` (conjunction is conservative)
- `UNKNOWN & SAT = UNKNOWN` (uncertainty propagates)

### Product Domain

The unified constraint system is the **product** of individual domains:

```
Ω(Γ) = Syntax(Γ) × Types(Γ) × Imports(Γ) × ControlFlow(Γ) × Semantics(Γ)
```

With component-wise meet:

```python
@dataclass(frozen=True, slots=True)
class UnifiedConstraint(Constraint["UnifiedConstraint"]):
    syntax: Constraint = TOP
    types: Constraint = TOP
    imports: Constraint = TOP
    controlflow: Constraint = TOP
    semantics: Constraint = TOP

    def meet(self, other: UnifiedConstraint) -> UnifiedConstraint:
        return UnifiedConstraint(
            syntax=self.syntax.meet(other.syntax),
            types=self.types.meet(other.types),
            imports=self.imports.meet(other.imports),
            controlflow=self.controlflow.meet(other.controlflow),
            semantics=self.semantics.meet(other.semantics),
        )
```

---

## Domain Implementations

### Domain Interface

```python
class ConstraintDomain(ABC, Generic[C]):
    """A constraint domain with its own semilattice structure."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique domain name."""

    @property
    @abstractmethod
    def top(self) -> C:
        """The trivial constraint (everything allowed)."""

    @property
    @abstractmethod
    def bottom(self) -> C:
        """The absurd constraint (nothing allowed)."""

    @abstractmethod
    def token_mask(
        self,
        constraint: C,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Project constraint to a token mask.

        Returns a boolean tensor (vocab_size,) where True = allowed.
        """

    @abstractmethod
    def observe_token(
        self,
        constraint: C,
        token_id: int,
        context: GenerationContext,
    ) -> C:
        """Update constraint after observing a generated token."""

    @abstractmethod
    def checkpoint(self) -> Checkpoint:
        """Save current state for backtracking."""

    @abstractmethod
    def restore(self, checkpoint: Checkpoint) -> None:
        """Restore to a previous state."""
```

### Type Domain: Bidirectional Type Checking

The type domain implements bidirectional type checking based on the Hazel research.

**Type Synthesis (↑)**: Infer the type of an expression
```
Γ ⊢ e ⇒ τ    (synthesize type τ from expression e in context Γ)
```

**Type Analysis (↓)**: Check an expression against an expected type
```
Γ ⊢ e ⇐ τ    (analyze expression e against expected type τ in context Γ)
```

**Key insight**: Bidirectional typing reduces the search space by propagating type information both up and down the AST.

### Type Constraint Structure

```python
@dataclass(frozen=True)
class TypeConstraint(Constraint):
    expected: Type              # What type should this have?
    environment: TypeEnvironment  # What types are in scope?
    unification: FrozenSet[TypeEquation]  # What equalities must hold?

    def meet(self, other: TypeConstraint) -> TypeConstraint:
        # 1. Merge environments (later shadows earlier)
        merged_env = self.environment.merge(other.environment)

        # 2. Unify expected types
        unified_expected, new_equations = unify(self.expected, other.expected)
        if unified_expected is None:
            return TYPE_BOTTOM

        # 3. Combine all unification constraints
        all_equations = self.unification | other.unification | new_equations

        # 4. Check satisfiability via unification
        solution = solve_unification(all_equations)
        if solution is None:
            return TYPE_BOTTOM

        return TypeConstraint(
            expected=apply_substitution(unified_expected, solution),
            environment=apply_substitution(merged_env, solution),
            unification=all_equations
        )
```

### Control Flow Domain: CFG Analysis

The control flow domain builds and analyzes control flow graphs incrementally.

**Immutable CFG with Structural Sharing**:

```python
@dataclass(frozen=True)
class CFGSketch:
    """Immutable CFG representation with structural sharing."""
    blocks: Tuple[BasicBlock, ...]
    current_block: int
    entry_block: int

    def extend(self, statement: Statement) -> CFGSketch:
        """Return new CFG with statement added (O(1) with sharing)."""
        # Only copy the modified block, share the rest
        new_blocks = list(self.blocks)
        current = self.blocks[self.current_block]
        new_blocks[self.current_block] = current.with_statement(statement)
        return CFGSketch(
            blocks=tuple(new_blocks),
            current_block=self.current_block,
            entry_block=self.entry_block,
        )
```

**Reachability Analysis**:

The domain tracks:
- Whether code is reachable from function entry
- Whether all paths terminate (for return statements)
- Loop depth for break/continue validity

---

## Cross-Domain Propagation

### Propagation Network

```python
class PropagationNetwork:
    """Worklist algorithm for cross-domain constraint propagation."""

    def __init__(self, domains: Dict[str, ConstraintDomain]):
        self.domains = domains
        self.edges: List[PropagationEdge] = []

    def propagate(
        self,
        initial: UnifiedConstraint,
        changed_domain: str,
    ) -> UnifiedConstraint:
        """Fixed-point propagation using worklist algorithm."""
        worklist = [changed_domain]
        current = initial

        while worklist:
            source = worklist.pop()
            for edge in self.edges:
                if edge.source == source:
                    # Apply morphism to get constraint in target domain
                    source_constraint = getattr(current, source)
                    propagated = edge.morphism(source_constraint)

                    # Meet with existing target constraint
                    target_constraint = getattr(current, edge.target)
                    new_target = target_constraint.meet(propagated)

                    if new_target != target_constraint:
                        # Constraint changed, add to worklist
                        current = current.with_domain(edge.target, new_target)
                        worklist.append(edge.target)

        return current
```

### Propagation Edges

| Source | Target | Effect |
|--------|--------|--------|
| types | imports | Type annotations require certain imports |
| imports | types | Available types depend on imports |
| controlflow | types | Unreachable code has no type constraints |
| types | semantics | Type constraints become SMT formulas |

---

## Typed Holes System

### Hazel Integration

Ananke's typed holes are based on the [Hazel](https://hazel.org) research, specifically "Statically Contextualizing LLMs with Typed Holes" (OOPSLA 2024).

### Key Principles

1. **Totality**: Every partial program state has a well-defined meaning
2. **Environment Capture**: Holes carry their typing context
3. **Hierarchical Refinement**: Coarse holes constrain finer holes

### Hole Granularities

```python
class HoleGranularity(Enum):
    """Hierarchy of hole granularities."""
    TOKEN = 0       # Single token
    EXPRESSION = 1  # Complete expression
    STATEMENT = 2   # Statement
    BLOCK = 3       # Block of statements
    FUNCTION = 4    # Function body
    MODULE = 5      # Module
    SYSTEM = 6      # Multi-module system
```

Coarser holes constrain finer holes: `SYSTEM > MODULE > FUNCTION > BLOCK > STATEMENT > EXPRESSION > TOKEN`

### Hole Structure

```python
@dataclass
class Hole:
    """A typed hole representing an unknown program fragment."""

    name: str
    expected_type: Optional[Type] = None
    environment: TypeEnvironment = field(default_factory=TypeEnvironment)
    constraint: Constraint = TOP
    state: HoleState = HoleState.EMPTY
    fill: Optional[str] = None

    def with_fill(self, fill: str) -> Hole:
        """Return hole with fill applied."""
        return Hole(
            name=self.name,
            expected_type=self.expected_type,
            environment=self.environment,
            constraint=self.constraint,
            state=HoleState.FILLED,
            fill=fill,
        )
```

### Fill-and-Resume Semantics

When a hole is filled:
1. Validate fill against hole's constraint
2. Type-check fill in hole's captured environment
3. Propagate any new constraints to dependent holes
4. Continue generation from hole boundary

---

## Incremental Type Checking

### Delta Typing

Instead of re-checking the entire program on each token, we use **delta typing**:

```python
class IncrementalTypeChecker:
    """Incremental bidirectional type checker."""

    def __init__(self):
        self.dependency_graph = DependencyGraph()
        self.cached_types: Dict[NodeId, Type] = {}

    def check_delta(
        self,
        new_node: ASTNode,
        context: GenerationContext,
    ) -> TypeCheckResult:
        """Check only the affected nodes."""
        # 1. Find affected nodes
        affected = self.dependency_graph.get_affected(new_node)

        # 2. Invalidate their cached types
        for node_id in affected:
            self.cached_types.pop(node_id, None)

        # 3. Re-check only affected nodes
        for node_id in self.dependency_graph.topological_order(affected):
            node = context.get_node(node_id)
            result = self._check_node(node, context)
            if result.is_error:
                return result
            self.cached_types[node_id] = result.type

        return TypeCheckResult.success()
```

### Dependency Graph

```
        ┌─────────────────────┐
        │   Function Def      │
        └──────────┬──────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    ┌───────┐  ┌───────┐  ┌───────┐
    │ Param │  │ Param │  │ Body  │
    │   a   │  │   b   │  │       │
    └───────┘  └───────┘  └───┬───┘
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
              ┌─────────┐          ┌─────────┐
              │ Return  │          │  Expr   │
              │   Stmt  │          │   a+b   │
              └─────────┘          └─────────┘
```

When `b`'s type changes, only dependent nodes (Body, Return, Expr) are re-checked.

---

## Zig SIMD Implementation

### Why Zig?

- **Compile-time target detection**: Automatically selects AVX2/NEON
- **Zero-overhead FFI**: C ABI without wrapper cost
- **Cross-platform**: Same code works on x86 and ARM
- **Graceful degradation**: Falls back to scalar code

### Mask Fusion with SIMD

```zig
const SIMD_WIDTH = 8;  // 8 × u32 = 256 bits

pub fn fuseTwoMasks(
    result: [*]u32,
    mask: [*]const u32,
    mask_size: usize,
) void {
    const simd_iterations = mask_size / SIMD_WIDTH;

    var j: usize = 0;
    while (j < simd_iterations) : (j += 1) {
        const offset = j * SIMD_WIDTH;

        // Load 8 u32 values from each mask
        const result_vec: @Vector(8, u32) = result[offset..][0..8].*;
        const mask_vec: @Vector(8, u32) = mask[offset..][0..8].*;

        // SIMD AND operation (compiled to VPAND on AVX2)
        result[offset..][0..8].* = result_vec & mask_vec;
    }

    // Scalar remainder
    const remainder_start = simd_iterations * SIMD_WIDTH;
    for (remainder_start..mask_size) |i| {
        result[i] &= mask[i];
    }
}
```

### Python FFI Integration

```python
from ctypes import CDLL, c_uint32, POINTER, c_size_t

class ZigMaskFusion:
    def __init__(self):
        self._lib = CDLL("libananke_simd.so")
        self._lib.fuse_masks.argtypes = [
            POINTER(c_uint32),  # result
            POINTER(c_uint32),  # mask
            c_size_t,           # mask_size
        ]

    def fuse(self, result: np.ndarray, mask: np.ndarray) -> None:
        result_ptr = result.ctypes.data_as(POINTER(c_uint32))
        mask_ptr = mask.ctypes.data_as(POINTER(c_uint32))
        self._lib.fuse_masks(result_ptr, mask_ptr, len(result))
```

---

## Performance Analysis

### Complexity Analysis

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Token classification | O(1) | Precomputed lookup table |
| Syntax mask | O(vocab_size) | llguidance linear scan |
| Type mask | O(k × vocab_size) | k = affected nodes |
| Mask fusion | O(vocab_size / SIMD_WIDTH) | SIMD parallelism |
| Checkpoint | O(1) | Structural sharing |
| Rollback | O(k) | k = tokens rolled back |

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| Mask pool | 8 × vocab_size × 4 bytes | Pre-allocated |
| CFG | O(statements) | Structural sharing |
| Type cache | O(unique_types) | LRU bounded |
| Checkpoints | O(max_rollback) | Ring buffer |

### Bottleneck Analysis

```
┌────────────────────────────────────────────────────────────┐
│                   Per-Token Latency                         │
├────────────────────────────────────────────────────────────┤
│ Syntax (llguidance)  ██████████                    50μs    │
│ Types                ██████████████████████████   500μs    │
│ Imports              ████████                     200μs    │
│ ControlFlow          ██████                       100μs    │
│ Semantics            ██████████████████████      1000μs    │
│ Mask Fusion          █                            10μs     │
├────────────────────────────────────────────────────────────┤
│ Total (worst case)   ████████████████████████████ ~2ms     │
│ Total (typical)      ██████████████               ~500μs   │
└────────────────────────────────────────────────────────────┘
```

### Optimization Strategies

1. **Lazy Evaluation**: Only evaluate domains within time budget
2. **Selectivity Ordering**: Most selective domains first (early termination)
3. **Caching**: Reuse masks when constraints unchanged
4. **Sparse Checkpointing**: Checkpoint every N tokens, not every token
5. **SIMD Acceleration**: Zig native code for mask operations

---

## References

### Academic Papers

- [Hazel: Statically Contextualizing LLMs with Typed Holes](https://hazel.org) (OOPSLA 2024)
- [Bidirectional Typing](https://arxiv.org/abs/1908.05839) (Dunfield & Krishnaswami)
- [Order-Maintenance Problem](https://doi.org/10.1137/S0097539795287316) (Dietz & Sleator)

### Related Systems

- [llguidance](https://github.com/microsoft/llguidance) - Dynamic token masking
- [XGrammar](https://arxiv.org/abs/2404.14366) - Grammar-constrained decoding
- [Outlines](https://github.com/outlines-dev/outlines) - Structured generation

---

## See Also

- [ARCHITECTURE.md](./ARCHITECTURE.md) - High-level overview
- [REFERENCE.md](./REFERENCE.md) - Complete API reference
- [CONTRIBUTING.md](./CONTRIBUTING.md) - How to extend Ananke
