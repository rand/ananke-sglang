# Ananke System Architecture

This document provides an overview of the Ananke multi-domain constrained generation system.

For detailed mathematical foundations and implementation specifics, see [ARCHITECTURE_DEEP_DIVE.md](./ARCHITECTURE_DEEP_DIVE.md).

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [Domain Interactions](#domain-interactions)
6. [Performance Characteristics](#performance-characteristics)
7. [Key Design Decisions](#key-design-decisions)

---

## System Overview

Ananke is a compositional constraint system for verified code generation. It combines multiple constraint domains (syntax, types, imports, control flow, semantics) into a unified framework that guides LLM token generation.

### Core Principles

1. **Algebraic Compositionality**: Constraints form a bounded meet-semilattice with well-defined operations
2. **Domain Independence**: Each domain operates independently; the propagation mechanism is agnostic to specific domains
3. **Incremental Computation**: All operations are efficient under single-token updates
4. **Backtracking Support**: Checkpoint/restore enables speculative decoding

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                  AnankeGrammar                                   │
│                                                                                  │
│  ┌─────────────────────┐    ┌──────────────────────────────────────────────────┐ │
│  │   Syntax Domain     │    │              Constraint Domains                  │ │
│  │    (llguidance)     │    │                                                  │ │
│  │                     │    │  ┌────────────┐ ┌────────────┐ ┌──────────────┐  │ │
│  │  ┌───────────────┐  │    │  │   Types    │ │  Imports   │ │ ControlFlow  │  │ │
│  │  │ CFG Parsing   │  │    │  │            │ │            │ │              │  │ │
│  │  │               │  │    │  │ Bidirec-   │ │  Module    │ │ CFG + reach- │  │ │
│  │  │ JSON Schema   │  │    │  │ tional     │ │  tracking  │ │ ability      │  │ │
│  │  │ Regex         │  │    │  │ inference  │ │            │ │ analysis     │  │ │
│  │  │ EBNF          │  │    │  │            │ │            │ │              │  │ │
│  │  │               │  │    │  │  <500μs    │ │  <200μs    │ │   <100μs     │  │ │
│  │  │   ~50μs       │  │    │  └─────┬──────┘ └─────┬──────┘ └───────┬──────┘  │ │
│  │  └───────┬───────┘  │    │        │              │                │         │ │
│  │          │          │    │        └──────────────┴────────────────┘         │ │
│  └──────────┼──────────┘    │                       │                          │ │
│             │               │        ┌──────────────▼─────────────┐            │ │
│             │               │        │    Semantics Domain        │            │ │
│             │               │        │   (SMT/Z3 - optional)      │            │ │
│             │               │        │        <1ms                │            │ │
│             │               │        └─────────────┬──────────────┘            │ │
│             │               └──────────────────────┼───────────────────────────┘ │
│             │                                      │                             │
│             │          ┌───────────────────────────▼───────────────────────────┐ │
│             └──────────►                  Mask Fusion                          │ │
│                        │                                                       │ │
│                        │   • Selectivity-ordered evaluation                    │ │
│                        │   • Early termination on highly selective masks       │ │
│                        │   • Zig SIMD acceleration (~50x faster)               │ │
│                        │   • Lazy evaluation with budget control               │ │
│                        └───────────────────────────┬───────────────────────────┘ │
│                                                    │                             │
│  ┌─────────────────────────┐    ┌──────────────────▼───────────────────────────┐ │
│  │   Checkpoint System     │    │              Final Token Mask                │ │
│  │                         │    │                                              │ │
│  │  • Sparse checkpoints   │    │   Boolean tensor (vocab_size,) indicating    │ │
│  │  • Structural sharing   │    │   which tokens are valid for generation      │ │
│  │  • O(1) checkpoint ops  │    │                                              │ │
│  │  • Max 200 rollback     │    └──────────────────────────────────────────────┘ │
│  └─────────────────────────┘                                                     │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                         Typed Holes System (Hazel)                          │ │
│  │                                                                             │ │
│  │   • Environment capture: Holes carry their typing context                   │ │
│  │   • Fill-and-resume: Generate code incrementally                            │ │
│  │   • Granularities: TOKEN → EXPRESSION → STATEMENT → BLOCK → FUNCTION        │ │
│  │   • Totality: Every partial program has a well-defined constraint           │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Syntax Domain (llguidance)

The syntax domain delegates to Microsoft's llguidance library for CFG-based token masking.

| Feature | Description |
|---------|-------------|
| JSON Schema | Constrain output to valid JSON matching a schema |
| Regex | Constrain output to match regex patterns |
| EBNF | Constrain output to match EBNF grammar |
| Latency | ~50μs per token |

### Types Domain

Implements incremental bidirectional type checking based on Hazel research.

| Feature | Description |
|---------|-------------|
| Type Synthesis | Infer types from expressions (↑) |
| Type Analysis | Check expressions against expected types (↓) |
| Incremental | Delta typing with dependency graphs |
| Languages | All 7 languages with full type systems |

### Imports Domain

Tracks module dependencies and validates import availability.

| Feature | Description |
|---------|-------------|
| Detection | Recognizes import statements in generated code |
| Resolution | Language-specific module resolvers |
| Validation | Ensures imported modules exist |

### ControlFlow Domain

Builds and analyzes control flow graphs incrementally.

| Feature | Description |
|---------|-------------|
| CFG Construction | Incremental with structural sharing |
| Reachability | Validates code is reachable |
| Termination | Ensures functions terminate properly |

### Semantics Domain

Optional SMT-based constraint checking using Z3.

| Feature | Description |
|---------|-------------|
| Formula Extraction | Extracts assertions from code |
| SMT Solving | Z3 integration for satisfiability |
| Graceful Degradation | Works without Z3 installed |

---

## Data Flow

### Token Generation Cycle

```
1. Token Generated
        │
        ▼
2. Token Classified (O(1) lookup)
        │
        ▼
3. For each domain:
   compute token_mask(constraint, context)
        │
        ▼
4. Fuse all masks via Zig SIMD
        │
        ▼
5. Use fused mask for sampling
        │
        ▼
6. observe_token() in all domains
        │
        ▼
7. Propagate constraints via worklist
        │
        ▼
8. Checkpoint (sparse: every N tokens)
        │
        ▼
9. Return to step 1
```

### Constraint Propagation

```
              ┌─────────────┐
              │   Types     │
              └──────┬──────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
   ┌─────────┐ ┌──────────┐ ┌─────────────┐
   │ Imports │ │ Syntax   │ │ ControlFlow │
   └────┬────┘ └────┬─────┘ └──────┬──────┘
        │           │              │
        └───────────┴──────────────┘
                    │
                    ▼
             ┌────────────┐
             │ Semantics  │
             └────────────┘
```

---

## Domain Interactions

### Cross-Domain Constraint Flow

| From | To | Effect |
|------|-----|--------|
| Types | Imports | Type annotations require imports |
| Imports | Types | Available types depend on imports |
| ControlFlow | Types | Reachability affects type checking |
| Types | Semantics | Type constraints become SMT formulas |

### Unified Constraint (Product Type)

```python
@dataclass(frozen=True)
class UnifiedConstraint(Constraint):
    syntax: Constraint = TOP
    types: Constraint = TOP
    imports: Constraint = TOP
    controlflow: Constraint = TOP
    semantics: Constraint = TOP
```

Meet operation is component-wise:
```
(s₁, t₁, i₁, cf₁, sem₁) ⊓ (s₂, t₂, i₂, cf₂, sem₂) =
    (s₁ ⊓ s₂, t₁ ⊓ t₂, i₁ ⊓ i₂, cf₁ ⊓ cf₂, sem₁ ⊓ sem₂)
```

---

## Performance Characteristics

### Per-Token Latency Targets

| Component | Target | Notes |
|-----------|--------|-------|
| Syntax (llguidance) | ~50μs | Delegated to llguidance |
| Types | <500μs | Incremental checking |
| Imports | <200μs | Statement detection |
| ControlFlow | <100μs | CFG analysis |
| Semantics | <1ms | Z3 dependent |
| Mask Fusion (SIMD) | ~10μs | With Zig native |
| **Total** | **<2-3ms** | Real-time performance |

### Optimizations Implemented

| Optimization | Speedup | Description |
|--------------|---------|-------------|
| Vectorized mask application | 455x | NumPy vectorization |
| TokenClassifier disk cache | 38x | Warm start acceleration |
| Pre-allocated mask pool | ~100μs/token | Avoid allocation overhead |
| Incremental constraints | 50%+ cache hits | Selective invalidation |
| Lazy evaluation | Bounded latency | Budget control |
| Immutable CFG | O(1) checkpoint | Structural sharing |
| Sparse checkpointing | 10x fewer ops | Checkpoint every N tokens |
| Zig SIMD mask fusion | ~50x | Native SIMD instructions |

---

## Key Design Decisions

### Why a Semilattice?

The bounded meet-semilattice structure provides:
- **Identity**: `c ⊓ ⊤ = c` (TOP doesn't change constraints)
- **Annihilation**: `c ⊓ ⊥ = ⊥` (contradictions propagate)
- **Idempotence**: `c ⊓ c = c` (re-applying is safe)
- **Commutativity**: `c₁ ⊓ c₂ = c₂ ⊓ c₁` (order independence)
- **Associativity**: Allows arbitrary grouping

### Why Hazel-Style Typed Holes?

Adapted from [Hazel](https://hazel.org) (OOPSLA 2024):
- Every partial program has a meaning (totality)
- Holes capture their typing environment
- Enables type-directed code completion
- Supports incremental refinement

### Why Zig for SIMD?

- Compile-time target feature detection
- Zero-overhead FFI via C ABI
- Cross-platform SIMD (AVX2/NEON)
- Graceful fallback to pure Python

---

## See Also

- [ARCHITECTURE_DEEP_DIVE.md](./ARCHITECTURE_DEEP_DIVE.md) - Mathematical foundations
- [REFERENCE.md](./REFERENCE.md) - Complete API reference
- [CONTRIBUTING.md](./CONTRIBUTING.md) - How to extend Ananke
- [GETTING_STARTED.md](./GETTING_STARTED.md) - Usage guide
