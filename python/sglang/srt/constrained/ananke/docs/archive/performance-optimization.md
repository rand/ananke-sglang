> ---
> **STATUS: HISTORICAL DOCUMENT - PARTIALLY IMPLEMENTED**
> 
> This is the performance optimization roadmap for the Ananke system.
> 
> **Completed Phases:**
> - Phase 1: Pure Python Optimizations (vectorized masks, disk cache, mask pool)
> - Phase 2: Data Structure Improvements (immutable CFG, sparse checkpointing)
> - Phase 3: Zig Native Rewrites (SIMD mask fusion)
> 
> **Implementation Status:**
> - 455x speedup via vectorized mask application: COMPLETE
> - 38x faster warm start via TokenClassifier disk cache: COMPLETE
> - ~100μs/token savings via pre-allocated mask pool: COMPLETE
> - O(1) checkpoint via immutable CFG with structural sharing: COMPLETE
> - 10x fewer checkpoint ops via sparse checkpointing: COMPLETE
> - ~50x speedup via Zig SIMD mask fusion: COMPLETE
> 
> For current architecture documentation, see:
> - [ARCHITECTURE.md](../ARCHITECTURE.md) - System overview
> - [ARCHITECTURE_DEEP_DIVE.md](../ARCHITECTURE_DEEP_DIVE.md) - Performance analysis
> 
> This document is preserved for historical reference and tracking.
> 
> ---
> 
# Ananke Performance Optimization Plan

> **Status**: Ready for Implementation
> **Last Updated**: December 2024
> **Prerequisite**: Multi-language support complete (7 languages implemented)
> **Branch Strategy**: Use git worktree for isolated development

---

## Table of Contents

1. [Current Project State](#current-project-state)
2. [Git Workflow](#git-workflow)
3. [Executive Summary](#executive-summary)
4. [Architecture Overview](#architecture-overview)
5. [Performance Baselines](#performance-baselines)
6. [Critical Findings](#critical-findings)
7. [Phase 1: Pure Python Optimizations](#phase-1-pure-python-optimizations)
8. [Phase 2: Data Structure Improvements](#phase-2-data-structure-improvements)
9. [Phase 3: Zig Native Rewrites](#phase-3-zig-native-rewrites)
10. [Deployment Targets](#deployment-targets)
11. [Benchmarking Strategy](#benchmarking-strategy)
12. [Risk Assessment](#risk-assessment)
13. [Appendix: Implementation Details](#appendix-implementation-details)

---

## Current Project State

### Codebase Summary

The Ananke constrained generation system is **production-ready** with comprehensive multi-language support:

| Metric | Value |
|--------|-------|
| Total Lines of Code | ~82,000 |
| Source Files | 167 |
| Test Functions | 2,537 |
| Test Lines | ~34,500 |
| Supported Languages | 7 (Python, TypeScript, Rust, Go, Kotlin, Swift, Zig) |

### Implementation Completeness

All 5 constraint domains are **fully implemented**:

| Domain | Status | Key Methods | Lines |
|--------|--------|-------------|-------|
| Syntax | Complete | `token_mask`, `observe_token`, `checkpoint`, `restore` | ~270 |
| Types | Complete | Bidirectional + incremental typing | ~958 |
| Imports | Complete | 7 language resolvers | ~646 |
| ControlFlow | Complete | CFG + reachability analysis | ~753 |
| Semantics | Complete | SMT solving + formula extraction | ~1,177 |

### Language Support Matrix

Each language has complete implementations across all layers:

| Language | Token Classifier | Type System | Parser | Import Resolver |
|----------|-----------------|-------------|--------|-----------------|
| Python | `token_classifier.py` | `languages/python.py` | `languages/python.py` | `resolvers/python.py` |
| TypeScript | `token_classifier_typescript.py` | `languages/typescript.py` | `languages/typescript.py` | `resolvers/typescript.py` |
| Rust | `token_classifier_rust.py` | `languages/rust.py` | `languages/rust.py` | `resolvers/rust.py` |
| Go | `token_classifier_go.py` | `languages/go.py` | `languages/go.py` | `resolvers/go.py` |
| Kotlin | `token_classifier_kotlin.py` | `languages/kotlin.py` | `languages/kotlin.py` | `resolvers/kotlin.py` |
| Swift | `token_classifier_swift.py` | `languages/swift.py` | `languages/swift.py` | `resolvers/swift.py` |
| Zig | `token_classifier_zig.py` | `languages/zig.py` | `languages/zig.py` | `resolvers/zig.py` |

### Recent Development History

```
Phase 19 (latest): Comprehensive tests for Kotlin, Swift, Go
Phase 18: Kotlin and Swift multi-language support
Phase 17: Multi-language hole and constraint tests
Phase 16: TypeScript language support
Phase 15: Three-phase improvement completion
Phase 14: Zig and Rust multi-language support
Phase 13: Context-aware bounds checking for SemanticDomain
Phases 1-12: Core infrastructure (domains, masks, holes, propagation, benchmarks)
```

### Current Git Status

```
Branch: main (up-to-date with origin/main)
Working tree: Clean (no staged or modified files)
Untracked:
  - ananke-performance-optimization.md (this document)
  - python/sglang/srt/constrained/ananke/docs/PLAN-go-swift-kotlin.md
```

---

## Git Workflow

### Branch Strategy

All performance optimization work MUST be done in a dedicated branch using git worktree for isolation:

```bash
# Create the optimization branch and worktree
cd /Users/rand/src/sglang
git worktree add ../sglang-perf-opt -b ananke/performance-optimization

# Work in the isolated worktree
cd ../sglang-perf-opt

# Verify you're on the correct branch
git branch --show-current  # Should show: ananke/performance-optimization
```

### Worktree Structure

```
/Users/rand/src/
├── sglang/                    # Main development (stay on main)
│   └── python/sglang/srt/constrained/ananke/
└── sglang-perf-opt/           # Performance optimization worktree
    └── python/sglang/srt/constrained/ananke/
```

### Development Workflow

```bash
# 1. Always work in the worktree
cd /Users/rand/src/sglang-perf-opt

# 2. Implement changes in phases (see Implementation Priority below)

# 3. Run tests before committing
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
    python/sglang/srt/constrained/ananke/tests/ -v

# 4. Run benchmarks to validate improvements
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
    python/sglang/srt/constrained/ananke/tests/benchmark/ -v

# 5. Commit with descriptive messages
git add -A
git commit -m "perf(ananke): Phase 1.1 - Vectorized mask application

- Replace O(vocab_size) Python loops with vectorized PyTorch ops
- Expected: ~10x speedup (200μs → 20μs per domain)
- Benchmark: bench_mask_computation.py passes

🤖 Generated with Claude Code"

# 6. Push to remote for CI
git push -u origin ananke/performance-optimization

# 7. When complete, create PR to main
gh pr create --title "perf(ananke): Performance optimization phases 1-3" \
    --body "## Summary
- Phase 1: Pure Python optimizations (vectorization, caching, pooling)
- Phase 2: Data structure improvements (immutable CFG, extended precomputation)
- Phase 3: Zig native rewrites (SIMD mask fusion, token classifier)

## Benchmarks
[Include before/after benchmark results]

🤖 Generated with Claude Code"
```

### Cleanup After Merge

```bash
# After PR is merged, remove the worktree
cd /Users/rand/src/sglang
git worktree remove ../sglang-perf-opt
git branch -d ananke/performance-optimization
```

---

## Executive Summary

### Key Findings from Analysis

The Ananke system has **well-designed architecture** but contains **underutilized optimization infrastructure**:

1. **Mask fusion bottleneck**: `_apply_domain_mask()` uses O(vocab_size) Python loops
2. **Eager checkpointing**: Creates checkpoint every token with deep copies
3. **Cache invalidation**: Mask cache cleared every token despite incremental infrastructure
4. **Unused optimizations**: `LazyConstraintEvaluator` and `IncrementalMaskComputer` exist but aren't integrated

### Optimization Impact Summary

| Phase | Focus | Expected Impact | Effort |
|-------|-------|-----------------|--------|
| 1 | Pure Python | 2-10x on hot paths | Low (1-2 weeks) |
| 2 | Data Structures | 5-10x on checkpointing | Medium (2-3 weeks) |
| 3 | Zig Rewrites | 50-100x on mask fusion | High (4-6 weeks) |

### Quantitative Targets

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| `_apply_domain_mask()` | ~200μs/domain | <20μs/domain | 10x |
| Checkpoint creation | O(tokens) | O(1) | Linear → Constant |
| Cold start (classifier) | ~500ms | <10ms (cached) | 50x |
| Full pipeline | ~2ms/token | <500μs/token | 4x |

---

## Architecture Overview

### System Components (82,000 lines)

```
python/sglang/srt/constrained/ananke/
├── core/                          # Constraint algebra foundation
│   ├── constraint.py              # Bounded meet-semilattice (334 lines)
│   ├── domain.py                  # Domain abstraction (341 lines)
│   ├── unified.py                 # Product constraint (240 lines)
│   ├── checkpoint.py              # State rollback (279 lines)
│   ├── token_classifier.py        # Base classifier (876 lines)
│   ├── token_classifier_*.py      # Language-specific (6 files, ~400 lines each)
│   └── context_extraction.py      # Context gathering (662 lines)
├── domains/                       # 5 constraint domains
│   ├── syntax/domain.py           # Grammar via llguidance (~270 lines)
│   ├── types/                     # Bidirectional type checking
│   │   ├── domain.py              # Main domain (958 lines)
│   │   ├── bidirectional/         # Type analysis, synthesis, subsumption
│   │   ├── incremental/           # Delta typing, invalidation
│   │   ├── marking/               # AST marking, provenance
│   │   └── languages/             # 7 language type systems
│   ├── imports/                   # Module availability
│   │   ├── domain.py              # Main domain (646 lines)
│   │   └── resolvers/             # 7 language resolvers
│   ├── controlflow/               # CFG-based reachability
│   │   ├── domain.py              # Main domain (753 lines)
│   │   ├── cfg.py                 # CFG construction
│   │   └── reachability.py        # Reachability analysis
│   └── semantics/                 # SMT-based constraints
│       ├── domain.py              # Main domain (1,177 lines)
│       ├── smt.py                 # SMT solver integration
│       └── extractors.py          # Formula extraction
├── backend/                       # SGLang integration
│   ├── backend.py                 # AnankeBackend orchestrator
│   └── grammar.py                 # Token stream processor (495 lines) [HOT PATH]
├── masks/                         # Token mask computation
│   ├── fuser.py                   # Mask fusion coordination (422 lines) [HOT PATH]
│   ├── cache.py                   # LRU mask cache [UNDERUTILIZED]
│   ├── lazy.py                    # Lazy evaluation [UNDERUTILIZED]
│   └── incremental.py             # Incremental computation [UNDERUTILIZED]
├── propagation/                   # Cross-domain constraint flow
├── holes/                         # Typed hole management
├── parsing/                       # Incremental multi-language parsing
│   └── languages/                 # 7 language parsers
└── tests/                         # Comprehensive test suite
    ├── unit/                      # 44 test files
    ├── integration/               # 4 test files
    ├── property/                  # 6 property-based test files
    └── benchmark/                 # 3 benchmark files
```

### Hot Path Data Flow

```
Token Generation Loop (per token):
┌─────────────────────────────────────────────────────────────────────┐
│ fill_vocab_mask(vocab_mask, idx)                                    │
│   ├─ syntax_grammar.fill_vocab_mask()     [~50μs, llguidance]       │
│   └─ for each domain in [types, imports, controlflow, semantics]:   │
│       ├─ domain.token_mask()              [100-800μs per domain]    │
│       └─ _apply_domain_mask()             [~200μs per domain] ←HOT  │
├─────────────────────────────────────────────────────────────────────┤
│ Model samples token                                                 │
├─────────────────────────────────────────────────────────────────────┤
│ accept_token(token)                                                 │
│   ├─ _maybe_create_checkpoint()           [O(tokens)] ←BOTTLENECK   │
│   │   ├─ for each domain: checkpoint()    [Deep copies]             │
│   │   └─ context_snapshot.copy()          [O(tokens) list copy]     │
│   ├─ for each domain: observe_token()     [5-50μs per domain]       │
│   ├─ _domain_mask_cache.clear()           [Defeats incrementality]  │
│   └─ satisfiability check                 [Early termination]       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Performance Baselines

### Current Targets (from benchmark tests)

| Operation | Target | Benchmark File | Assertion |
|-----------|--------|----------------|-----------|
| Syntax mask | ~50μs | llguidance (external) | N/A |
| Type mask | <500μs | `bench_type_checking.py` | `assert mean < 500` |
| Type mask p99 | <500μs | `test_token_masking.py:520` | `assert p99_us < 500` |
| Import mask | <100μs | `bench_mask_computation.py` | Implicit |
| ControlFlow mask | <200μs | `bench_mask_computation.py` | Implicit |
| Semantic mask | <1ms | `bench_mask_computation.py` | Implicit |
| Semantic mask p99 | <500μs | `test_token_masking.py:547` | `assert p99_us < 500` |
| Fused mask (sequential) | <5ms | `bench_mask_computation.py:193` | `assert mean_ms < 5` |
| Single domain mask | <500μs | `bench_mask_computation.py:172` | `assert mean_us < 500` |
| Cache put | <100μs | `bench_mask_computation.py:268` | `assert mean_us < 100` |
| Cache get (hit) | <10μs | `bench_mask_computation.py:283` | `assert mean_us < 10` |
| Change detection | <50μs | `bench_mask_computation.py:355` | `assert mean_us < 50` |
| Incremental compute | <500μs | `bench_mask_computation.py:384` | `assert mean_us < 500` |

### Estimated Latency Breakdown (per token)

```
fill_vocab_mask() breakdown:
  Syntax mask (llguidance):      50μs
  Type mask computation:        400μs
  Type mask application:        200μs  ← Target for optimization
  Import mask computation:       80μs
  Import mask application:      200μs  ← Target for optimization
  ControlFlow mask:             150μs
  ControlFlow mask application: 200μs  ← Target for optimization
  Semantic mask:                800μs
  Semantic mask application:    200μs  ← Target for optimization
  Cache management:              10μs
  ─────────────────────────────────────
  Total (estimated):          ~2.3ms

accept_token() breakdown:
  Checkpoint creation:          ???μs  ← O(tokens), growing
  Domain observe_token (×5):  25-250μs
  Context extension:            10μs
  Cache invalidation:           10μs
  ─────────────────────────────────────
  Total (estimated):         50-300μs + O(tokens)
```

---

## Critical Findings

### Finding 1: Mask Application is O(vocab_size)

**Location**: `backend/grammar.py:464-495`

```python
def _apply_domain_mask(self, vocab_mask, idx, domain_mask):
    vocab_size = domain_mask.shape[0]
    mask_size = vocab_mask.shape[1]

    for word_idx in range(mask_size):           # O(vocab_size/32)
        start_bit = word_idx * 32
        end_bit = min(start_bit + 32, vocab_size)

        word = 0
        for bit_offset, token_idx in enumerate(range(start_bit, end_bit)):  # O(32)
            if domain_mask[token_idx]:          # Tensor index per iteration
                word |= 1 << bit_offset

        vocab_mask[idx, word_idx] &= word
```

**Impact**: For 32K vocabulary × 4 domains = 4,000+ word iterations per token

### Finding 2: Eager Checkpointing with Deep Copies

**Location**: `backend/grammar.py:421-440`

```python
def _maybe_create_checkpoint(self):
    # NOTE: "For now, checkpoint every token (can be optimized later)"
    domain_checkpoints = {}
    for domain_name, domain in self.domains.items():
        domain_checkpoints[domain_name] = domain.checkpoint()  # Deep copy

    context_snapshot = {
        "generated_tokens": self.context.generated_tokens.copy(),  # O(n) copy
        "metadata": self.context.metadata.copy(),                  # O(n) copy
    }
```

**Impact**: O(tokens²) cumulative overhead as generation progresses

### Finding 3: Cache Invalidation Defeats Incrementality

**Location**: `backend/grammar.py:211, 261`

```python
# Line 211 (in accept_token):
self._domain_mask_cache.clear()

# Line 261 (in rollback):
self._domain_mask_cache.clear()
```

**Available but unused**: `masks/incremental.py` has `IncrementalMaskComputer` with change detection

### Finding 4: Lazy Evaluation Infrastructure Unused

**Location**: `masks/lazy.py` (full implementation exists)

```python
class LazyConstraintEvaluator:
    """Budget-aware domain evaluation with priority ordering."""
    # NOT integrated into fill_vocab_mask()
```

**Impact**: Expensive domains (semantic @800μs) always evaluated even when budget exceeded

### Finding 5: No Tensor Pooling

**Location**: `backend/grammar.py:306-330`

```python
def allocate_vocab_mask(self, vocab_size, batch_size, device):
    # Fresh allocation every time
    return torch.zeros((batch_size, mask_size), dtype=torch.int32, device=device)
```

**Impact**: CUDA allocation overhead per generation step

---

## Phase 1: Pure Python Optimizations

### 1.1 Vectorized Mask Application

**Target**: `backend/grammar.py:464-495`
**Priority**: 1 (Highest)
**Effort**: Low (2-4 hours)

Replace O(vocab_size) Python loops with vectorized PyTorch operations:

```python
def _apply_domain_mask_vectorized(self, vocab_mask, idx, domain_mask):
    """Apply domain mask using vectorized PyTorch operations.

    Complexity: O(vocab_size / 32) tensor operations instead of
    O(vocab_size) Python loop iterations.
    """
    vocab_size = domain_mask.shape[0]
    device = domain_mask.device

    # Pad to multiple of 32
    padded_size = ((vocab_size + 31) // 32) * 32
    if vocab_size < padded_size:
        padded = torch.zeros(padded_size, dtype=torch.bool, device=device)
        padded[:vocab_size] = domain_mask
    else:
        padded = domain_mask

    # Reshape to [num_words, 32] and pack bits
    reshaped = padded.view(-1, 32)

    # Pre-compute powers of 2 (cache this at class level)
    if not hasattr(self, '_powers_of_2') or self._powers_of_2.device != device:
        self._powers_of_2 = (2 ** torch.arange(32, device=device, dtype=torch.int32))

    # Vectorized bit packing: sum of (bit_value * 2^position)
    packed = (reshaped.int() * self._powers_of_2).sum(dim=1).int()

    # Apply via vectorized AND
    mask_size = vocab_mask.shape[1]
    vocab_mask[idx, :packed.shape[0]] &= packed[:mask_size]
```

**Expected Impact**: ~10x speedup (200μs → 20μs per domain)

### 1.2 TokenClassifier Persistent Disk Cache

**Target**: `core/token_classifier.py`
**Priority**: 2
**Effort**: Low (2-4 hours)

```python
import hashlib
import pickle
from pathlib import Path
from typing import Optional

CACHE_DIR = Path.home() / ".cache" / "ananke" / "classifiers"

def get_or_create_classifier(
    tokenizer,
    language: str = "python",
    cache_dir: Optional[Path] = None
) -> "TokenClassifier":
    """Get cached classifier or create and cache a new one.

    Cache key is deterministic hash of vocabulary + language.
    """
    cache_dir = cache_dir or CACHE_DIR

    # Create deterministic hash of vocabulary
    vocab = tokenizer.get_vocab()
    vocab_str = str(sorted(vocab.items()))
    vocab_hash = hashlib.sha256(vocab_str.encode()).hexdigest()[:16]

    cache_path = cache_dir / f"classifier_{language}_{vocab_hash}.pkl"

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                classifier = pickle.load(f)
                # Validate cached classifier
                if (classifier._vocab_size == len(vocab) and
                    classifier._language == language):
                    return classifier
        except (pickle.UnpicklingError, AttributeError, EOFError):
            cache_path.unlink()  # Remove corrupted cache

    # Create new classifier
    classifier = TokenClassifier(tokenizer, language)
    classifier.initialize()

    # Cache for future use
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(classifier, f, protocol=pickle.HIGHEST_PROTOCOL)

    return classifier
```

**Expected Impact**: Eliminates ~500ms cold start for cached tokenizers

### 1.3 Pre-allocated Mask Tensor Pool

**Target**: `backend/grammar.py`, `core/domain.py`
**Priority**: 3
**Effort**: Low (2-4 hours)

```python
class MaskPool:
    """Pre-allocated tensor pool to avoid per-token CUDA allocations."""

    def __init__(self, vocab_size: int, device: str, pool_size: int = 8):
        self._vocab_size = vocab_size
        self._device = device
        self._pool = [
            torch.ones(vocab_size, dtype=torch.bool, device=device)
            for _ in range(pool_size)
        ]
        self._available = list(range(pool_size))

    def acquire(self, fill_value: bool = True) -> tuple[torch.Tensor, int]:
        """Acquire a mask tensor from the pool."""
        if not self._available:
            # Fallback: allocate new tensor (should be rare)
            return torch.full(
                (self._vocab_size,), fill_value,
                dtype=torch.bool, device=self._device
            ), -1

        handle = self._available.pop()
        mask = self._pool[handle]
        mask.fill_(fill_value)
        return mask, handle

    def release(self, handle: int) -> None:
        """Return a mask tensor to the pool."""
        if handle >= 0:
            self._available.append(handle)
```

**Expected Impact**: ~100μs savings per token (eliminates CUDA allocation)

### 1.4 Integrate Existing Incremental Infrastructure

**Target**: `backend/grammar.py:211`
**Priority**: 4
**Effort**: Medium (4-8 hours)

Replace blind cache invalidation with change detection:

```python
from ..masks.incremental import IncrementalMaskComputer, ConstraintChange

class AnankeGrammar:
    def __init__(self, ...):
        # ... existing init ...
        self._incremental_computer = IncrementalMaskComputer(
            domains=self.domains,
            vocab_size=self.vocab_size,
            device=self.device,
        )

    def accept_token(self, token: int) -> None:
        # ... existing token observation ...

        # Instead of: self._domain_mask_cache.clear()
        # Use change detection:
        changes = self._detect_constraint_changes(old_constraint, new_constraint)
        if changes:
            self._incremental_computer.invalidate_domains(
                [c.domain_name for c in changes]
            )
```

**Expected Impact**: Skip recomputation for unchanged domains (~50% cache hits)

### 1.5 Integrate Lazy Evaluation

**Target**: `backend/grammar.py:fill_vocab_mask`
**Priority**: 5
**Effort**: Medium (4-8 hours)

```python
from ..masks.lazy import LazyConstraintEvaluator, EvaluationBudget

def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
    # Syntax first (required)
    if self.syntax_grammar is not None:
        self.syntax_grammar.fill_vocab_mask(vocab_mask, idx)

    # Budget-aware evaluation for other domains
    budget = EvaluationBudget(
        max_time_us=1500,  # 1.5ms budget
        max_domains=4,
    )

    evaluator = LazyConstraintEvaluator(
        domains=self.domains,
        budget=budget,
        priority_order=["types", "imports", "controlflow", "semantics"],
    )

    result = evaluator.evaluate(self.constraint, self.context)

    for domain_name, domain_mask in result.evaluated_masks.items():
        self._apply_domain_mask(vocab_mask, idx, domain_mask)
```

**Expected Impact**: Skip expensive domains when budget exceeded

---

## Phase 2: Data Structure Improvements

### 2.1 Immutable CFG with Structural Sharing

**Target**: `domains/controlflow/domain.py`
**Priority**: 6
**Effort**: Medium (2-3 days)

```python
from immutables import Map
from dataclasses import dataclass
from typing import FrozenSet, Optional

@dataclass(frozen=True)
class ImmutableCFGSketch:
    """Persistent CFG with O(log n) updates and O(1) copy."""

    blocks: Map[str, "BasicBlock"]
    edges: FrozenSet["CFGEdge"]
    entry_id: Optional[str] = None
    exit_ids: FrozenSet[str] = frozenset()

    def add_block(self, block: "BasicBlock") -> "ImmutableCFGSketch":
        return ImmutableCFGSketch(
            blocks=self.blocks.set(block.id, block),
            edges=self.edges,
            entry_id=block.id if block.is_entry else self.entry_id,
            exit_ids=self.exit_ids | ({block.id} if block.is_exit else frozenset()),
        )

    def add_edge(self, edge: "CFGEdge") -> "ImmutableCFGSketch":
        return ImmutableCFGSketch(
            blocks=self.blocks,
            edges=self.edges | {edge},
            entry_id=self.entry_id,
            exit_ids=self.exit_ids,
        )

    def __hash__(self) -> int:
        # Efficient hash for change detection
        return hash((id(self.blocks), self.edges, self.entry_id, self.exit_ids))
```

**Expected Impact**: Checkpoint cost from O(blocks) to O(1)

### 2.2 Sparse Checkpointing

**Target**: `backend/grammar.py:421-440`
**Priority**: 7
**Effort**: Medium (1-2 days)

```python
class CheckpointStrategy:
    """Configurable checkpoint creation strategy."""

    def __init__(
        self,
        interval: int = 10,           # Checkpoint every N tokens
        on_constraint_change: bool = True,  # Also checkpoint on significant changes
        max_checkpoints: int = 200,
    ):
        self.interval = interval
        self.on_constraint_change = on_constraint_change
        self.max_checkpoints = max_checkpoints
        self._token_count = 0
        self._last_constraint_hash = None

    def should_checkpoint(
        self,
        position: int,
        constraint: "UnifiedConstraint"
    ) -> bool:
        self._token_count += 1

        # Interval-based
        if self._token_count % self.interval == 0:
            return True

        # Change-based
        if self.on_constraint_change:
            constraint_hash = hash(constraint)
            if constraint_hash != self._last_constraint_hash:
                self._last_constraint_hash = constraint_hash
                return True

        return False
```

**Expected Impact**: 10x fewer checkpoints (every 10 tokens vs every token)

### 2.3 Extended Type Mask Precomputation

**Target**: `domains/types/domain.py`
**Priority**: 8
**Effort**: Medium (1-2 days)

```python
def _initialize_extended_type_masks(self):
    """Precompute masks for common compound types."""
    # Primitives (existing)
    self._type_masks[PrimitiveType("int")] = self._create_int_mask()
    self._type_masks[PrimitiveType("float")] = self._create_float_mask()
    self._type_masks[PrimitiveType("str")] = self._create_str_mask()
    self._type_masks[PrimitiveType("bool")] = self._create_bool_mask()

    # Common list types
    self._type_masks[ListType(PrimitiveType("int"))] = self._create_list_mask("int")
    self._type_masks[ListType(PrimitiveType("str"))] = self._create_list_mask("str")
    self._type_masks[ListType(PrimitiveType("float"))] = self._create_list_mask("float")

    # Common dict types
    self._type_masks[DictType(STR, ANY)] = self._create_dict_mask()
    self._type_masks[DictType(STR, INT)] = self._create_typed_dict_mask("str", "int")

    # Callable (matches any function/lambda)
    self._type_masks[FunctionType(params=[], returns=ANY)] = self._create_callable_mask()

    # Optional types
    self._type_masks[UnionType([INT, NoneType()])] = self._create_optional_mask("int")
    self._type_masks[UnionType([STR, NoneType()])] = self._create_optional_mask("str")
```

**Expected Impact**: O(1) coverage from ~60% to ~85% of type contexts

---

## Phase 3: Zig Native Rewrites

### 3.1 SIMD Mask Fusion Engine

**Priority**: HIGHEST in Phase 3
**Location**: New `zig/` directory
**Effort**: High (1-2 weeks)

```zig
// zig/src/mask_fusion.zig
const std = @import("std");

pub export fn fuse_masks_simd(
    vocab_mask_ptr: [*]i32,
    domain_mask_ptr: [*]const bool,
    vocab_size: usize,
    batch_idx: usize,
    mask_stride: usize,
) callconv(.C) void {
    const mask_size = (vocab_size + 31) / 32;

    // Use 256-bit vectors (AVX2) - process 8 words at a time
    const Vec8 = @Vector(8, i32);

    var word_idx: usize = 0;
    while (word_idx + 8 <= mask_size) : (word_idx += 8) {
        var packed: Vec8 = @splat(0);

        inline for (0..8) |w| {
            const word_start = (word_idx + w) * 32;
            var word: i32 = 0;

            inline for (0..32) |bit| {
                const token_idx = word_start + bit;
                if (token_idx < vocab_size and domain_mask_ptr[token_idx]) {
                    word |= @as(i32, 1) << @intCast(bit);
                }
            }
            packed[w] = word;
        }

        const row_offset = batch_idx * mask_stride + word_idx;
        const existing: Vec8 = vocab_mask_ptr[row_offset..][0..8].*;
        vocab_mask_ptr[row_offset..][0..8].* = existing & packed;
    }

    // Scalar remainder
    while (word_idx < mask_size) : (word_idx += 1) {
        const word_start = word_idx * 32;
        var word: i32 = 0;

        for (0..32) |bit| {
            const token_idx = word_start + bit;
            if (token_idx < vocab_size and domain_mask_ptr[token_idx]) {
                word |= @as(i32, 1) << @intCast(bit);
            }
        }

        const offset = batch_idx * mask_stride + word_idx;
        vocab_mask_ptr[offset] &= word;
    }
}
```

**Python FFI**:

```python
# zig/ffi.py
import ctypes
from pathlib import Path

def _load_zig_library():
    lib_path = Path(__file__).parent / "zig-out" / "lib" / "libananke_zig.so"
    if not lib_path.exists():
        lib_path = lib_path.with_suffix(".dylib")  # macOS
    return ctypes.CDLL(str(lib_path))

_lib = _load_zig_library()

_lib.fuse_masks_simd.argtypes = [
    ctypes.c_void_p,   # vocab_mask_ptr
    ctypes.c_void_p,   # domain_mask_ptr
    ctypes.c_size_t,   # vocab_size
    ctypes.c_size_t,   # batch_idx
    ctypes.c_size_t,   # mask_stride
]
_lib.fuse_masks_simd.restype = None

def apply_domain_mask_simd(vocab_mask, idx, domain_mask):
    """Apply domain mask using Zig SIMD implementation."""
    _lib.fuse_masks_simd(
        vocab_mask.data_ptr(),
        domain_mask.data_ptr(),
        domain_mask.shape[0],
        idx,
        vocab_mask.shape[1],
    )
```

**Expected Impact**: 50-100x speedup (200μs → 2-4μs)

### 3.2 Zig Token Classifier

**Location**: `zig/src/classifier.zig`
**Effort**: High (1 week)

```zig
// Compile-time perfect hash for Python keywords
const PythonKeywords = std.ComptimeStringMap(TokenCategory, .{
    .{ "if", .KEYWORD },
    .{ "else", .KEYWORD },
    .{ "elif", .KEYWORD },
    .{ "for", .KEYWORD },
    .{ "while", .KEYWORD },
    .{ "def", .KEYWORD },
    .{ "class", .KEYWORD },
    .{ "return", .KEYWORD },
    .{ "import", .KEYWORD },
    .{ "from", .KEYWORD },
    // ... all 35+ Python keywords
});

pub fn classifyToken(text: []const u8) TokenClassification {
    const trimmed = std.mem.trim(u8, text, " \t\n\r");

    if (trimmed.len == 0) return .{ .category = .WHITESPACE };

    // O(1) keyword lookup
    if (PythonKeywords.get(trimmed)) |category| {
        return .{ .category = category };
    }

    // Fast literal detection (no regex)
    if (isDigit(trimmed[0])) {
        return classifyNumeric(trimmed);
    }

    if (trimmed[0] == '"' or trimmed[0] == '\'') {
        return .{ .category = .STRING_LITERAL };
    }

    return .{ .category = .IDENTIFIER };
}
```

**Expected Impact**: 10-20x faster vocabulary classification

### 3.3 Build Configuration

**File**: `zig/build.zig`

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addSharedLibrary(.{
        .name = "ananke_zig",
        .root_source_file = .{ .path = "src/ffi.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Enable SIMD based on target architecture
    if (target.result.cpu.arch == .x86_64) {
        lib.root_module.addCpuFeature(.avx2);
    } else if (target.result.cpu.arch == .aarch64) {
        lib.root_module.addCpuFeature(.neon);
    }

    b.installArtifact(lib);

    // Tests
    const tests = b.addTest(.{
        .root_source_file = .{ .path = "src/tests.zig" },
        .target = target,
        .optimize = optimize,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
```

---

## Deployment Targets

### Primary: Linux x86_64 (Production)

- CUDA/GPU inference servers
- AVX2/AVX-512 SIMD support
- Focus on throughput and p99 latency

### Secondary: macOS (Development)

- Intel Macs: AVX2 support
- Apple Silicon: NEON intrinsics
- Focus on developer experience

### Runtime Feature Detection

```zig
pub const CpuFeatures = struct {
    has_avx2: bool,
    has_avx512: bool,
    has_neon: bool,

    pub fn detect() CpuFeatures {
        const info = std.Target.current;
        return .{
            .has_avx2 = info.cpu.arch == .x86_64 and
                std.Target.x86.featureSetHas(info.cpu.features, .avx2),
            .has_avx512 = info.cpu.arch == .x86_64 and
                std.Target.x86.featureSetHas(info.cpu.features, .avx512f),
            .has_neon = info.cpu.arch == .aarch64,
        };
    }
};
```

---

## Benchmarking Strategy

### Existing Infrastructure

```bash
# Run all benchmarks
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
    python/sglang/srt/constrained/ananke/tests/benchmark/ -v

# Run specific benchmark
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
    python/sglang/srt/constrained/ananke/tests/benchmark/bench_mask_computation.py -v
```

### New Benchmarks to Add

```python
# tests/benchmark/bench_optimization_comparison.py

import pytest
import torch
import time

class TestOptimizationComparison:
    """Compare baseline vs optimized implementations."""

    def test_mask_application_baseline_vs_vectorized(self, benchmark_context):
        """Compare Python loop vs PyTorch vectorized."""
        vocab_size = 32000
        domain_mask = torch.rand(vocab_size) > 0.5
        vocab_mask = torch.ones((1, (vocab_size + 31) // 32), dtype=torch.int32)

        # Baseline
        start = time.perf_counter_ns()
        _apply_domain_mask_baseline(vocab_mask.clone(), 0, domain_mask)
        baseline_ns = time.perf_counter_ns() - start

        # Vectorized
        start = time.perf_counter_ns()
        _apply_domain_mask_vectorized(vocab_mask.clone(), 0, domain_mask)
        vectorized_ns = time.perf_counter_ns() - start

        speedup = baseline_ns / vectorized_ns
        assert speedup > 5, f"Expected >5x speedup, got {speedup:.1f}x"

    def test_checkpoint_baseline_vs_sparse(self, benchmark_context):
        """Compare every-token vs sparse checkpointing."""
        pass

    def test_classifier_cold_vs_warm(self, benchmark_context):
        """Compare cold start vs cached classifier."""
        pass
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Zig FFI overhead negates gains | Low | High | Profile carefully; only use for >5μs operations |
| Memory safety at FFI boundary | Medium | High | Use DLPack for tensor sharing; validate pointers |
| Build complexity | Medium | Medium | Single `build.zig` with cross-platform support |
| Regression in correctness | Low | High | Property-based tests verify algebraic laws |
| ARM64 performance regression | Low | Medium | Test NEON fallback on Apple Silicon |
| Cache coherence issues | Low | Medium | Document cache invalidation patterns |

---

## Implementation Priority

| Phase | Priority | Optimization | Impact | Effort | Status |
|-------|----------|-------------|--------|--------|--------|
| 1 | 1 | Vectorized mask application | HIGH | LOW | Ready |
| 1 | 2 | TokenClassifier disk cache | HIGH | LOW | Ready |
| 1 | 3 | Pre-allocated mask tensors | MEDIUM | LOW | Ready |
| 1 | 4 | Integrate incremental infrastructure | MEDIUM | MEDIUM | Ready |
| 1 | 5 | Integrate lazy evaluation | MEDIUM | MEDIUM | Ready |
| 2 | 6 | Immutable CFG | HIGH | MEDIUM | Ready |
| 2 | 7 | Sparse checkpointing | HIGH | MEDIUM | Ready |
| 2 | 8 | Extended type mask precomputation | MEDIUM | MEDIUM | Ready |
| 3 | 9 | Zig SIMD mask fusion | HIGHEST | HIGH | Ready |
| 3 | 10 | Zig token classifier | HIGH | HIGH | Ready |

---

## Appendix: Implementation Details

### Key Files to Modify

| File | Phase | Changes |
|------|-------|---------|
| `backend/grammar.py:464-495` | 1.1 | Replace `_apply_domain_mask` with vectorized version |
| `core/token_classifier.py` | 1.2 | Add `get_or_create_classifier` with disk cache |
| `backend/grammar.py` | 1.3 | Add `MaskPool` and integrate |
| `backend/grammar.py:211` | 1.4 | Replace cache clear with change detection |
| `backend/grammar.py:fill_vocab_mask` | 1.5 | Integrate `LazyConstraintEvaluator` |
| `domains/controlflow/cfg.py` | 2.1 | Add `ImmutableCFGSketch` |
| `backend/grammar.py:421-440` | 2.2 | Add `CheckpointStrategy` |
| `domains/types/domain.py` | 2.3 | Extend `_initialize_type_masks` |
| New: `zig/src/mask_fusion.zig` | 3.1 | SIMD mask fusion |
| New: `zig/src/classifier.zig` | 3.2 | Token classifier |
| New: `zig/build.zig` | 3.x | Build configuration |

### Test Coverage Requirements

Each optimization must include:
1. Unit tests for correctness
2. Benchmark comparison (before/after)
3. Property-based tests where applicable
4. Integration test with full pipeline

### Commit Message Format

```
perf(ananke): Phase X.Y - Brief description

- Detailed change 1
- Detailed change 2
- Expected impact: Nx speedup

Benchmark results:
- Before: XXXμs
- After: XXXμs
- Speedup: X.Xx

🤖 Generated with Claude Code
```
