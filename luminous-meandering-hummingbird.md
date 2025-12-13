# Ananke Implementation Plan
## A Compositional Constraint System for Verified Code Generation

**Location**: `python/sglang/srt/constrained/ananke/`
**Target Languages**: Python, TypeScript, Rust, Zig, Go
**Type Domain**: Full incremental bidirectional type checker with marked lambda calculus
**Constraint Domains**: All 5 (Syntax, Types, Imports, Control Flow, Semantics)
**Estimated Files**: ~75 Python files across 12 subpackages
**Estimated Lines**: ~15,000-20,000 LOC
**Key Dependencies**: tree-sitter, immutables, z3-solver, llguidance

---

## 1. Executive Summary

Ananke extends SGLang's constrained decoding to support **multi-domain constraint fusion** across syntax, types, imports, control flow, and semantics. The system treats code generation as a constraint satisfaction problem where each token must satisfy all active constraints.

**Key Innovation**: Token masks from multiple domains are fused via bitwise AND, with cross-domain constraint propagation ensuring consistency.

### Theoretical Foundations

Ananke draws heavily from the Hazel research program (Cyrus Omar et al.) which provides rigorous foundations for:

1. **Typed Holes as First-Class Citizens**: Following [Live Functional Programming with Typed Holes](https://arxiv.org/abs/1805.00155), incomplete programs containing holes remain both statically and dynamically well-defined. Holes serve as membranes around missing code and type inconsistencies.

2. **Marked Lambda Calculus**: From [Total Type Error Localization and Recovery with Holes (POPL 2024)](https://dl.acm.org/doi/10.1145/3632910), we adopt the bidirectionally-typed marking system that localizes errors to specific holes via traced provenances. This ensures every partial program has a meaningful type, even during generation.

3. **Incremental Bidirectional Typing**: From [Incremental Bidirectional Typing via Order Maintenance (OOPSLA 2025)](https://arxiv.org/abs/2504.08946), we adopt order maintenance data structures for ~275x speedup over naive re-analysis. Updates propagate through the typed AST via small-step dynamics.

4. **ChatLSP Integration Pattern**: From [Statically Contextualizing LLMs with Typed Holes (OOPSLA 2024)](https://arxiv.org/abs/2409.00921), we adopt the language server integration approach where type context is extracted from holes and used to guide generation.

### Design Principles from Hazel

1. **Totality**: Every editor state (partial program) must be well-defined—no meaningless intermediate states
2. **Bidirectionality**: Type information flows both up (synthesis) and down (analysis) through the AST
3. **Incrementality**: Only recompute what changes when a token is added
4. **Error as Data**: Type errors are first-class values that can be inspected and recovered from
5. **Fill-and-Resume**: Generation can continue even after encountering constraints, with backtracking available

### Mathematical Foundations

**Constraint Semilattice**: All constraints form bounded meet-semilattices ⟨C, ⊓, ⊤, ⊥⟩ where:
- **C** is the set of constraints
- **⊓** (meet) is constraint conjunction
- **⊤** (top) is the trivial constraint (always satisfied)
- **⊥** (bottom) is the absurd constraint (never satisfied)

**Required Properties** (verified by property-based tests):
```
c ⊓ ⊤ = c                    (identity)
c ⊓ ⊥ = ⊥                    (annihilation)
c ⊓ c = c                    (idempotence)
c₁ ⊓ c₂ = c₂ ⊓ c₁            (commutativity)
(c₁ ⊓ c₂) ⊓ c₃ = c₁ ⊓ (c₂ ⊓ c₃)  (associativity)
```

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        AnankeBackend                             │
│                  (registers via grammar_backend_registry)        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     PropagationNetwork                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────┐│
│  │  Syntax  │◄─┤  Types   │◄─┤ Imports  │◄─┤CtrlFlow  │◄─┤Seman││
│  │  Domain  │  │  Domain  │  │  Domain  │  │  Domain  │  │tics ││
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──┬──┘│
│       │             │             │             │            │   │
│       ▼             ▼             ▼             ▼            ▼   │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │                    TokenMaskFuser                            ││
│  │           (bitwise AND with short-circuit)                   ││
│  └────────────────────────────┬─────────────────────────────────┘│
└───────────────────────────────┼──────────────────────────────────┘
                                ▼
                    int32 bitmask tensor → logits
```

---

## 3. Directory Structure

```
python/sglang/srt/constrained/ananke/
├── __init__.py
├── py.typed
│
├── core/                              # Constraint algebra
│   ├── __init__.py
│   ├── constraint.py                  # Constraint ABC, Satisfiability enum
│   ├── domain.py                      # ConstraintDomain ABC, GenerationContext
│   ├── unified.py                     # UnifiedConstraint (product of 5 domains)
│   ├── checkpoint.py                  # State checkpointing for rollback
│   └── context_extraction.py          # ChatLSP-style context extraction methods
│
├── domains/                           # 5 constraint domains
│   ├── __init__.py
│   │
│   ├── syntax/                        # Wraps llguidance
│   │   ├── __init__.py
│   │   ├── constraint.py              # SyntaxConstraint
│   │   ├── domain.py                  # SyntaxDomain
│   │   └── grammars/                  # Language-specific grammar definitions
│   │       ├── __init__.py
│   │       ├── python.py              # Python grammar (Lark format)
│   │       ├── typescript.py          # TypeScript grammar
│   │       ├── rust.py                # Rust grammar
│   │       ├── zig.py                 # Zig grammar
│   │       └── go.py                  # Go grammar
│   │
│   ├── types/                         # Full incremental type checker
│   │   ├── __init__.py
│   │   ├── constraint.py              # TypeConstraint, Type hierarchy
│   │   ├── domain.py                  # TypeDomain
│   │   ├── unification.py             # Robinson's unification + occurs check
│   │   ├── checker.py                 # IncrementalTypeChecker
│   │   ├── environment.py             # TypeEnvironment (immutable map)
│   │   │
│   │   ├── marking/                   # Marked lambda calculus (POPL 2024)
│   │   │   ├── __init__.py
│   │   │   ├── marks.py               # Mark types: hole, inconsistent, error
│   │   │   ├── provenance.py          # Traced error provenances
│   │   │   ├── marked_ast.py          # AST with mark annotations
│   │   │   └── totalization.py        # Total type assignment for partial programs
│   │   │
│   │   ├── incremental/               # Order maintenance (OOPSLA 2025)
│   │   │   ├── __init__.py
│   │   │   ├── order_maintenance.py   # O(1) amortized sequence ordering
│   │   │   ├── dependency_graph.py    # Fine-grained type dependencies
│   │   │   ├── delta_typing.py        # Small-step type updates
│   │   │   └── invalidation.py        # Minimal recomputation on change
│   │   │
│   │   ├── bidirectional/             # Bidirectional type checking
│   │   │   ├── __init__.py
│   │   │   ├── synthesis.py           # Type synthesis (up-flow)
│   │   │   ├── analysis.py            # Type analysis (down-flow)
│   │   │   └── subsumption.py         # Subtype checking with holes
│   │   │
│   │   └── languages/                 # Language-specific type systems
│   │       ├── __init__.py
│   │       ├── base.py                # LanguageTypeSystem ABC
│   │       ├── python.py              # Python type system (mypy-compatible)
│   │       ├── typescript.py          # TypeScript type system
│   │       ├── rust.py                # Rust type system (ownership-aware)
│   │       ├── zig.py                 # Zig type system (comptime)
│   │       └── go.py                  # Go type system (interfaces)
│   │
│   ├── imports/                       # Module/package constraints
│   │   ├── __init__.py
│   │   ├── constraint.py              # ImportConstraint
│   │   ├── domain.py                  # ImportDomain
│   │   └── resolvers/                 # Language-specific import resolution
│   │       ├── __init__.py
│   │       ├── base.py                # ImportResolver ABC
│   │       ├── python.py              # Python (pip packages, stdlib)
│   │       ├── typescript.py          # TypeScript (npm packages)
│   │       ├── rust.py                # Rust (cargo crates)
│   │       ├── zig.py                 # Zig (zon packages)
│   │       └── go.py                  # Go (go modules)
│   │
│   ├── controlflow/                   # CFG constraints
│   │   ├── __init__.py
│   │   ├── constraint.py              # ControlFlowConstraint
│   │   ├── domain.py                  # ControlFlowDomain
│   │   ├── cfg.py                     # CFGSketch representation
│   │   └── reachability.py            # Reachability analysis
│   │
│   └── semantics/                     # SMT-based semantic constraints
│       ├── __init__.py
│       ├── constraint.py              # SemanticConstraint
│       ├── domain.py                  # SemanticDomain
│       ├── smt.py                     # Z3 integration with incremental solving
│       └── extractors.py              # Extract formulas from assertions/contracts
│
├── propagation/                       # Cross-domain constraint flow
│   ├── __init__.py
│   ├── network.py                     # PropagationNetwork (worklist algorithm)
│   ├── edges.py                       # Standard propagation edge definitions
│   ├── worklist.py                    # Priority worklist with iteration limit
│   └── builder.py                     # build_standard_propagation_network()
│
├── holes/                             # Typed hole management (Hazel foundations)
│   ├── __init__.py
│   ├── hole.py                        # Hole, HoleId, HoleGranularity
│   ├── registry.py                    # HoleRegistry with parent-child hierarchy
│   ├── factory.py                     # Dynamic hole creation from AST
│   ├── closure.py                     # Hole closures for fill-and-resume
│   ├── environment_capture.py         # Capture typing environment at hole site
│   └── fill_resume.py                 # Live evaluation around holes
│
├── masks/                             # Token mask computation and fusion
│   ├── __init__.py
│   ├── fuser.py                       # TokenMaskFuser (selectivity-ordered AND)
│   ├── incremental.py                 # IncrementalMaskComputer
│   ├── cache.py                       # LRU mask cache with context-aware keys
│   └── lazy.py                        # LazyConstraintEvaluator
│
├── parsing/                           # Incremental parsing per language
│   ├── __init__.py
│   ├── partial_ast.py                 # PartialAST abstraction
│   ├── base.py                        # IncrementalParser ABC
│   └── languages/
│       ├── __init__.py
│       ├── python.py                  # tree-sitter-python
│       ├── typescript.py              # tree-sitter-typescript
│       ├── rust.py                    # tree-sitter-rust
│       ├── zig.py                     # tree-sitter-zig
│       └── go.py                      # tree-sitter-go
│
├── backend/                           # SGLang integration
│   ├── __init__.py
│   ├── grammar.py                     # AnankeGrammar (BaseGrammarObject)
│   └── backend.py                     # AnankeBackend (BaseGrammarBackend)
│
└── tests/                             # Comprehensive test suite
    ├── __init__.py
    ├── conftest.py                    # Shared fixtures
    │
    ├── unit/                          # Unit tests
    │   ├── test_constraint_algebra.py # Constraint ABC tests
    │   ├── test_unification.py        # Unification algorithm tests
    │   ├── test_marking.py            # Marked lambda calculus tests
    │   ├── test_incremental.py        # Order maintenance tests
    │   ├── test_bidirectional.py      # Bidirectional typing tests
    │   ├── test_propagation.py        # Propagation network tests
    │   ├── test_holes.py              # Hole management tests
    │   └── test_mask_fusion.py        # Token mask fusion tests
    │
    ├── integration/                   # Integration tests
    │   ├── test_sglang_backend.py     # SGLang integration tests
    │   ├── test_python_types.py       # Python type-constrained generation
    │   ├── test_typescript_types.py   # TypeScript type-constrained generation
    │   ├── test_rust_types.py         # Rust type-constrained generation
    │   ├── test_zig_types.py          # Zig type-constrained generation
    │   ├── test_go_types.py           # Go type-constrained generation
    │   └── test_multi_language.py     # Cross-language tests
    │
    ├── property/                      # Property-based tests (Hypothesis)
    │   ├── test_semilattice_laws.py   # All 5 semilattice laws
    │   ├── test_totality_invariant.py # Every partial program gets a type
    │   └── test_propagation_monotonicity.py
    │
    └── benchmark/                     # Performance benchmarks
        ├── bench_mask_computation.py
        ├── bench_type_checking.py
        └── bench_propagation.py
```

---

## 4. Critical Files to Modify/Create

### 4.1 SGLang Integration Points

| File | Modification |
|------|--------------|
| `constrained/base_grammar_backend.py:199` | Import ananke backend in `create_grammar_backend()` |
| `constrained/__init__.py` | Export AnankeBackend |
| `server_args.py` | Add `grammar_backend="ananke"` option |

### 4.2 Core Implementation Files (Create)

| Priority | File | Purpose |
|----------|------|---------|
| P0 | `ananke/core/constraint.py` | Base Constraint ABC with semilattice operations |
| P0 | `ananke/core/domain.py` | ConstraintDomain ABC and GenerationContext |
| P0 | `ananke/core/unified.py` | UnifiedConstraint product type |
| P0 | `ananke/core/context_extraction.py` | ChatLSP-style context extraction (5 methods) |
| P0 | `ananke/backend/grammar.py` | AnankeGrammar implementing BaseGrammarObject |
| P0 | `ananke/backend/backend.py` | AnankeBackend implementing BaseGrammarBackend |
| P1 | `ananke/domains/syntax/domain.py` | SyntaxDomain wrapping llguidance |
| P1 | `ananke/domains/types/domain.py` | TypeDomain with incremental checker |
| P1 | `ananke/domains/types/unification.py` | Type unification engine |
| P1 | `ananke/domains/types/marking/marks.py` | Mark types: HoleMark, InconsistentMark |
| P1 | `ananke/domains/types/marking/provenance.py` | Error provenance tracking |
| P1 | `ananke/domains/types/marking/marked_ast.py` | AST with mark annotations |
| P1 | `ananke/domains/types/marking/totalization.py` | Total type assignment for partials |
| P1 | `ananke/domains/types/incremental/order_maintenance.py` | O(1) amortized ordering |
| P1 | `ananke/domains/types/incremental/dependency_graph.py` | Type dependency tracking |
| P1 | `ananke/domains/types/bidirectional/synthesis.py` | Type synthesis (up-flow) |
| P1 | `ananke/domains/types/bidirectional/analysis.py` | Type analysis (down-flow) |
| P1 | `ananke/propagation/network.py` | PropagationNetwork |
| P2 | `ananke/holes/hole.py` | Typed hole with environment capture |
| P2 | `ananke/holes/fill_resume.py` | Fill-and-resume evaluation |
| P2 | `ananke/masks/fuser.py` | TokenMaskFuser |
| P2 | `ananke/domains/types/languages/*.py` | Per-language type systems (5 files)

---

## 5. Implementation Phases

### Phase 1: Core Foundation (Week 1)

**Goal**: Establish constraint algebra and SGLang backend skeleton

**Tasks**:
1. Create `core/constraint.py`:
   - `Satisfiability` enum (SAT, UNSAT, UNKNOWN)
   - `Constraint` ABC with `meet()`, `is_top()`, `is_bottom()`, `satisfiability()`
   - Semilattice property tests

2. Create `core/domain.py`:
   - `GenerationContext` dataclass
   - `ConstraintDomain[C]` ABC with `token_mask()`, `observe_token()`, `checkpoint()`, `restore()`

3. Create `core/unified.py`:
   - `UnifiedConstraint` as product of 5 domain constraints
   - Component-wise `meet()` operation

4. Create `backend/grammar.py`:
   - `AnankeGrammar(BaseGrammarObject)` skeleton
   - Delegate to wrapped llguidance grammar for now

5. Create `backend/backend.py`:
   - `AnankeBackend(BaseGrammarBackend)` skeleton
   - Register with `register_grammar_backend("ananke", ...)`

**Milestone**: `--grammar-backend ananke` works identically to `--grammar-backend llguidance`

---

### Phase 2: Syntax Domain (Week 2)

**Goal**: Proper syntax domain wrapper with constraint algebra support

**Tasks**:
1. Create `domains/syntax/constraint.py`:
   - `SyntaxConstraint` with grammar string and state hash
   - `SYNTAX_TOP` and `SYNTAX_BOTTOM` singletons

2. Create `domains/syntax/domain.py`:
   - `SyntaxDomain` wrapping llguidance backend
   - Forward `token_mask()` to llguidance
   - State checkpointing for rollback

3. Implement grammar intersection approximation:
   - Track grammar string for equality
   - Delegate actual parsing to llguidance

**Milestone**: Syntax constraints work through Ananke with proper semilattice semantics

---

### Phase 3: Type Domain Foundation — Marked Lambda Calculus

**Goal**: Full incremental type checker with marked lambda calculus (Hazel foundations)

**Tasks**:
1. Create `domains/types/constraint.py`:
   - Type representation hierarchy:
     - `TypeVar` (unification variable)
     - `PrimitiveType` (int, str, bool, float, None)
     - `FunctionType` (params, returns)
     - `ListType`, `DictType`, `TupleType`
     - `UnionType`, `OptionalType`
     - `ClassType` (nominal types)
     - `AnyType` (⊤), `NeverType` (⊥)
     - `HoleType` (typed hole placeholder)
   - `TypeEquation` for unification constraints
   - `TypeConstraint` with expected type, environment, marked AST

2. Create `domains/types/unification.py`:
   - `Substitution` class with `apply()` and `compose()`
   - `occurs_check()` function
   - `unify(t1, t2)` implementing Robinson's algorithm
   - `solve_unification(equations)` worklist algorithm
   - Handle `HoleType` in unification (holes unify with anything)

3. Create `domains/types/environment.py`:
   - `TypeEnvironment` as immutable Map[str, Type] using `immutables.Map`
   - `bind()`, `lookup()`, `merge()` operations
   - `snapshot()` for checkpoint/restore

4. Create `domains/types/marking/marks.py`:
   - `Mark` base class
   - `HoleMark(hole_id, expected_type)` — empty hole awaiting term
   - `InconsistentMark(synthesized, expected, provenance)` — type mismatch
   - `NonEmptyHoleMark(hole_id, inner)` — hole with partial content

5. Create `domains/types/marking/provenance.py`:
   - `Provenance(location, context, parent)` — error trace
   - `SourceSpan(start, end, file)` — source location
   - Provenance chaining for nested errors

6. Create `domains/types/marking/marked_ast.py`:
   - `MarkedAST(node, mark, synthesized_type, children)`
   - `collect_errors()` → all `InconsistentMark`s with provenances
   - `find_hole(hole_id)` → locate specific hole
   - `find_first_unfilled_hole()` → next hole to fill

7. Create `domains/types/marking/totalization.py`:
   - `totalize(ast, expected, env)` → `MarkedAST`
   - Every partial AST gets a well-defined type
   - Type mismatches become `InconsistentMark`s, not failures

8. Create `domains/types/bidirectional/synthesis.py`:
   - `synthesize(ast, env)` → `(Type, MarkedAST)`
   - Type flows UP from leaves to root
   - Handles: literals, variables, function applications

9. Create `domains/types/bidirectional/analysis.py`:
   - `analyze(ast, expected, env)` → `MarkedAST`
   - Type flows DOWN from context to expression
   - Handles: lambdas, let-bindings, conditionals

10. Create `domains/types/bidirectional/subsumption.py`:
    - `subsumes(sub, super)` → bool (is sub <: super?)
    - Handle variance for function types
    - Subsumption with holes (holes satisfy any constraint)

**Milestone**: Marked lambda calculus passes totalization tests — every partial program has a type

---

### Phase 3b: Incremental Bidirectional Typing (Order Maintenance)

**Goal**: O(1) amortized type updates using order maintenance (OOPSLA 2025)

**Tasks**:
1. Create `domains/types/incremental/order_maintenance.py`:
   - `OrderMaintenanceList` with O(1) amortized insert/query
   - Dietz & Sleator algorithm for label maintenance
   - `insert_after(element)`, `query(a, b)` → is a before b?

2. Create `domains/types/incremental/dependency_graph.py`:
   - `DependencyGraph` tracking type dependencies
   - `add_dependency(from_node, to_node)`
   - `affected_by(edit_span)` → nodes whose types may change
   - Bidirectional dependency tracking (synthesis and analysis)

3. Create `domains/types/incremental/delta_typing.py`:
   - `DeltaTypingEngine` for incremental updates
   - `apply_edit(old_ast, new_ast, edit_span)` → minimal recheck
   - Cache type results at AST nodes
   - Track which nodes are "dirty" after edits

4. Create `domains/types/incremental/invalidation.py`:
   - `InvalidationEngine` for cache management
   - `invalidate(node)` → mark node and dependents as dirty
   - Transitive closure over dependency graph
   - Stop propagation at unchanged type boundaries

5. Create `domains/types/checker.py`:
   - `IncrementalBidirectionalChecker` combining all components
   - `check_full(ast, expected, env)` → initial full check
   - `check_incremental(old, new, edit_span)` → incremental update
   - `observe_token(token, context)` → per-token update

**Milestone**: Incremental type checking achieves ~275x speedup over naive reanalysis

---

### Phase 4: Language-Specific Type Systems (Week 5-6)

**Goal**: Type systems for Python, TypeScript, Rust, Zig, Go

**Tasks**:
1. Create `domains/types/languages/base.py`:
   - `LanguageTypeSystem` ABC:
     - `parse_type_annotation(str) -> Type`
     - `infer_literal_type(literal) -> Type`
     - `get_builtin_types() -> Dict[str, Type]`
     - `check_assignable(source, target) -> bool`

2. Create `domains/types/languages/python.py`:
   - Python type system (compatible with mypy/pyright):
     - Primitive types: int, str, bool, float, bytes, None
     - Generic types: List[T], Dict[K,V], Set[T], Tuple[...]
     - Callable[[Args...], Return]
     - Union[T1, T2], Optional[T]
     - TypeVar support
     - Protocol support (structural subtyping)

3. Create `domains/types/languages/typescript.py`:
   - TypeScript type system:
     - Primitive: number, string, boolean, null, undefined
     - Array<T>, Record<K,V>, Map<K,V>
     - Union types, literal types
     - Interface types
     - Generic constraints

4. Create `domains/types/languages/rust.py`:
   - Rust type system:
     - Primitive: i32, u64, f64, bool, char, &str, String
     - Ownership annotations: &T, &mut T, Box<T>, Rc<T>, Arc<T>
     - Lifetime parameters (simplified)
     - Result<T,E>, Option<T>
     - Trait bounds

5. Create `domains/types/languages/zig.py`:
   - Zig type system:
     - Primitive: i32, u64, f64, bool, []u8
     - Comptime types
     - Optional: ?T
     - Error unions: E!T
     - Slice types: []T, [*]T

6. Create `domains/types/languages/go.py`:
   - Go type system:
     - Primitive: int, int64, float64, string, bool, byte, rune
     - Slice: []T, map[K]V
     - Interface types
     - Channel types: chan T
     - Pointer: *T

**Milestone**: All 5 language type systems pass conformance tests

---

### Phase 5: Type Domain Integration with ChatLSP Context

**Goal**: Type domain produces token masks, observes tokens, and provides rich context

**Tasks**:
1. Implement `TypeDomain.token_mask()`:
   - Get syntactically valid tokens from syntax domain
   - For each token, predict resulting AST change
   - Use incremental type checker to verify type validity
   - Budget-limited: check top-K most likely tokens individually
   - Heuristic masks for tokens above budget
   - Return boolean mask

2. Implement `TypeDomain.observe_token()`:
   - Update partial AST via incremental parser
   - Run incremental type check (using order maintenance)
   - Update marked AST with new marks/types
   - Refine expected type for next hole position
   - Trigger constraint propagation if type changes

3. Create `core/context_extraction.py` (ChatLSP-style methods):
   - `ContextExtractor` class implementing 5 ChatLSP methods:
     - `expected_type(hole)` → structured type info
     - `relevant_types(hole, env)` → ranked relevant bindings
     - `relevant_headers(hole, env)` → function signatures
     - `error_report(marked_ast)` → actionable type errors
     - `ai_tutorial(hole, env)` → natural language guidance
   - These methods extract rich context from typed holes
   - Context can be used to guide generation or debug errors

4. Create `parsing/base.py`:
   - `IncrementalParser` ABC with `extend_with_token()`
   - `PartialAST` representation with holes
   - `HoleDetector` finding holes in partial AST

5. Create `parsing/languages/*.py`:
   - tree-sitter integration per language
   - `PythonIncrementalParser` using tree-sitter-python
   - `TypeScriptIncrementalParser` using tree-sitter-typescript
   - `RustIncrementalParser` using tree-sitter-rust
   - `ZigIncrementalParser` using tree-sitter-zig (or custom)
   - `GoIncrementalParser` using tree-sitter-go
   - Each maps tree-sitter nodes to typed AST nodes

6. Implement `TypeDomain.get_context()`:
   - Returns `GenerationContext` with:
     - `expected_type`: what type is needed
     - `relevant_bindings`: available names and types
     - `function_signatures`: callable functions
     - `errors`: current type errors (if any)
   - Used by sampler to inform generation

**Milestone**: Type-constrained generation works for Python with rich context extraction

---

### Phase 6: Propagation Network (Week 8)

**Goal**: Cross-domain constraint propagation

**Tasks**:
1. Create `propagation/network.py`:
   - `PropagationNetwork` class
   - Domain registration
   - Constraint storage per domain
   - Worklist-based propagation loop

2. Create `propagation/edges.py`:
   - `PropagationEdge` dataclass (source, target, propagate function, priority)
   - `syntax_to_types()`: Syntactic structure → type expectations
   - `types_to_syntax()`: Type expectations → grammar restrictions
   - `types_to_imports()`: Type usage → required imports
   - `imports_to_types()`: Available imports → type environment

3. Create `propagation/worklist.py`:
   - Priority queue implementation
   - Fixpoint detection
   - Iteration limit protection

**Milestone**: Constraint changes in one domain propagate to others

---

### Phase 7: Import Domain (Week 9)

**Goal**: Track required/forbidden imports with version constraints

**Tasks**:
1. Create `domains/imports/constraint.py`:
   - `ModuleSpec` (name, version constraint)
   - `ImportConstraint` with required, forbidden, versions

2. Create `domains/imports/domain.py`:
   - `ImportDomain` tracking import statements
   - Detect new imports from generated code
   - Validate against constraints

3. Create `domains/imports/resolver.py`:
   - Language-specific import resolution
   - Python: pip packages, stdlib
   - TypeScript: npm packages
   - Rust: cargo crates
   - Zig: zon packages
   - Go: go modules

**Milestone**: Import constraints affect type availability

---

### Phase 8: Control Flow Domain (Week 10)

**Goal**: CFG-level constraints (reachability, termination)

**Tasks**:
1. Create `domains/controlflow/constraint.py`:
   - `CFGSketch` with basic blocks and edges
   - `ReachabilityConstraint` (must-reach, must-not-reach)
   - `TerminationRequirement` enum

2. Create `domains/controlflow/domain.py`:
   - `ControlFlowDomain` building CFG from AST
   - Detect control flow patterns

3. Create `domains/controlflow/reachability.py`:
   - Reachability analysis
   - Dead code detection
   - Return path analysis

**Milestone**: Control flow constraints enforced during generation

---

### Phase 9: Semantic Domain (Week 11)

**Goal**: SMT-based semantic constraints via Z3

**Tasks**:
1. Create `domains/semantics/constraint.py`:
   - `SMTFormula` wrapper for Z3 expressions
   - `SemanticConstraint` as formula set

2. Create `domains/semantics/domain.py`:
   - `SemanticDomain` with incremental solver
   - Extract formulas from assertions/contracts

3. Create `domains/semantics/smt.py`:
   - Z3 integration
   - Incremental satisfiability checking
   - Model extraction for debugging

**Milestone**: Semantic constraints (assertions, contracts) enforced

---

### Phase 10: Hole Management — Fill-and-Resume Semantics

**Goal**: Typed holes with Hazel-style fill-and-resume evaluation

**Tasks**:
1. Create `holes/hole.py`:
   - `HoleGranularity` enum (TOKEN, EXPRESSION, STATEMENT, BLOCK, FUNCTION, MODULE, SYSTEM)
   - `HoleId` (namespace, name, index, depth)
   - `Hole[C]` generic class with:
     - `id: HoleId`
     - `expected_type: Type`
     - `environment: TypeEnvironment` (captured at hole site)
     - `constraint: C` (unified constraint at hole)
     - `parent: Optional[HoleId]` (for nested holes)

2. Create `holes/registry.py`:
   - `HoleRegistry` managing hole hierarchy
   - `register(hole)`, `lookup(hole_id)`, `children(hole_id)`
   - Parent-child constraint inheritance (child ⊑ parent)
   - `next_hole()` → hole selection strategy (depth-first, breadth-first, priority)
   - `fill(hole_id, term)` → replace hole with term
   - `unfill(hole_id)` → restore hole (for backtracking)

3. Create `holes/closure.py`:
   - `HoleClosure` representing evaluation state around a hole
   - Following Hazel: "Holes serve as membranes around missing code"
   - `HoleClosure(hole_id, env_snapshot, continuation)`
   - Captures what would happen after hole is filled

4. Create `holes/environment_capture.py`:
   - `capture_environment(ast, position)` → `TypeEnvironment`
   - Extract all bindings visible at hole position
   - Include: local variables, parameters, imports, globals
   - Exclude: shadowed names, out-of-scope bindings

5. Create `holes/fill_resume.py`:
   - `FillAndResumeEngine` for live evaluation around holes
   - `evaluate_with_holes(ast)` → partial result + hole closures
   - Following Hazel: evaluation proceeds around holes
   - `fill_and_continue(hole_id, term, closures)` → updated result
   - Enable: try a fill, see result, backtrack if wrong

6. Create `holes/factory.py`:
   - `HoleFactory` creating holes from context:
     - `from_ast_gap(ast, position)` → hole from parsing gap
     - `from_type_error(mark)` → hole from inconsistency
     - `from_incomplete_expression(ast)` → hole from partial expr
   - Automatic constraint inference for new holes

7. Create `holes/strategy.py`:
   - `HoleSelectionStrategy` ABC
   - `DepthFirstStrategy` — complete inner holes first
   - `BreadthFirstStrategy` — complete outer holes first
   - `TypeGuidedStrategy` — fill most-constrained holes first
   - `PriorityStrategy` — user-specified order

**Milestone**: Multi-hole generation with fill-and-resume enabling backtracking

---

### Phase 11: Token Mask Fusion (Week 13)

**Goal**: Optimized multi-domain mask computation

**Tasks**:
1. Create `masks/fuser.py`:
   - `TokenMaskFuser` with selectivity ordering
   - Short-circuit on all-false mask
   - Lazy domain evaluation

2. Create `masks/incremental.py`:
   - `IncrementalMaskComputer`
   - Track changed domains
   - Recompute only affected masks

3. Create `masks/cache.py`:
   - LRU cache for domain masks
   - Cache key: (domain, constraint_hash, position)

**Milestone**: <2ms total mask computation

---

### Phase 12: Performance Optimization (Week 14)

**Goal**: Meet performance targets

**Tasks**:
1. Profile and optimize hot paths
2. Implement parallel domain evaluation (optional)
3. Add Triton kernel for multi-mask fusion
4. Benchmark against baseline llguidance

**Performance Targets**:
| Operation | Target |
|-----------|--------|
| Syntax mask | ~50μs (llguidance baseline) |
| Type mask | <500μs |
| Fused mask | <1ms |
| Total overhead | <2ms per token |

---

### Phase 13: Testing and Documentation (Week 15-16)

**Goal**: Production-ready implementation

**Tasks**:
1. Comprehensive unit tests for all components
2. Integration tests with SGLang
3. Language-specific conformance tests
4. Performance benchmarks
5. API documentation
6. Usage examples

---

## 6. Key Implementation Details

### 6.1 AnankeGrammar (BaseGrammarObject Implementation)

```python
class AnankeGrammar(BaseGrammarObject):
    def __init__(self, syntax_grammar, constraint, network, ...):
        self.syntax_grammar = syntax_grammar  # Wrapped llguidance
        self.constraint = constraint           # UnifiedConstraint
        self.network = network                 # PropagationNetwork
        self.fuser = TokenMaskFuser(network.domains)

    def accept_token(self, token: int) -> None:
        # 1. Update syntax grammar (delegation)
        self.syntax_grammar.accept_token(token)

        # 2. Update each domain constraint
        for domain_name, domain in self.network.domains.items():
            old = getattr(self.constraint, domain_name)
            new = domain.observe_token(old, token, self.context)
            setattr(self.constraint, domain_name, new)

        # 3. Propagate cross-domain
        self.network.propagate()

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        # 1. Get syntax mask from wrapped grammar
        self.syntax_grammar.fill_vocab_mask(vocab_mask, idx)

        # 2. Compute additional domain masks (types, imports, etc.)
        additional_mask = self.fuser.compute_fused_mask(
            self.constraint, self.context, exclude_syntax=True
        )

        # 3. Apply additional restrictions via bitwise AND
        self._apply_additional_mask(vocab_mask, idx, additional_mask)
```

### 6.2 Type Unification Core

```python
def unify(t1: Type, t2: Type) -> Optional[Tuple[Type, FrozenSet[TypeEquation]]]:
    """Robinson's unification with occurs check."""
    if t1 == t2:
        return (t1, frozenset())

    if isinstance(t1, AnyType):
        return (t2, frozenset())
    if isinstance(t2, AnyType):
        return (t1, frozenset())

    if isinstance(t1, TypeVar):
        if occurs_check(t1, t2):
            return None  # Infinite type
        return (t2, frozenset({TypeEquation(t1, t2)}))

    if isinstance(t2, TypeVar):
        if occurs_check(t2, t1):
            return None
        return (t1, frozenset({TypeEquation(t2, t1)}))

    # Structural unification for compound types
    if isinstance(t1, FunctionType) and isinstance(t2, FunctionType):
        return _unify_function_types(t1, t2)

    # Type mismatch
    return None
```

### 6.3 Marked Lambda Calculus Core

Following POPL 2024, we implement a marking system that localizes type errors to specific holes:

```python
@dataclass(frozen=True)
class Mark:
    """A mark indicates a type inconsistency location."""
    pass

@dataclass(frozen=True)
class HoleMark(Mark):
    """Empty hole awaiting a term."""
    hole_id: HoleId
    expected_type: Type

@dataclass(frozen=True)
class InconsistentMark(Mark):
    """Type inconsistency between synthesized and expected."""
    synthesized: Type
    expected: Type
    provenance: Provenance

@dataclass(frozen=True)
class Provenance:
    """Traces error back to source location."""
    location: SourceSpan
    context: str  # e.g., "function argument", "return type"
    parent: Optional['Provenance'] = None

class MarkedAST:
    """AST annotated with marks at each node."""
    def __init__(self, node: ASTNode, mark: Optional[Mark] = None):
        self.node = node
        self.mark = mark
        self.synthesized_type: Optional[Type] = None
        self.children: List[MarkedAST] = []

    def collect_errors(self) -> List[Tuple[Mark, Provenance]]:
        """Collect all error marks with their provenances."""
        errors = []
        if isinstance(self.mark, InconsistentMark):
            errors.append((self.mark, self.mark.provenance))
        for child in self.children:
            errors.extend(child.collect_errors())
        return errors

def totalize(ast: PartialAST, expected: Type, env: TypeEnvironment) -> MarkedAST:
    """
    Totalization: assign a type to every partial program.

    Even incomplete programs get well-defined types via marking.
    This is the key insight from Hazel - no meaningless states.
    """
    if is_hole(ast):
        return MarkedAST(ast, HoleMark(ast.hole_id, expected))

    # Bidirectional: try analysis first, then synthesis
    analyzed = try_analyze(ast, expected, env)
    if analyzed.is_consistent():
        return analyzed

    synthesized = try_synthesize(ast, env)
    if can_subsume(synthesized.type, expected):
        return synthesized

    # Type mismatch: mark as inconsistent, but continue
    return MarkedAST(
        ast,
        InconsistentMark(synthesized.type, expected,
                        Provenance(ast.span, "expression"))
    )
```

### 6.4 Incremental Bidirectional Typing (Order Maintenance)

Following OOPSLA 2025, we use order maintenance for O(1) amortized updates:

```python
class OrderMaintenanceList:
    """
    Maintains a total order over elements with O(1) amortized:
    - insert_after(element)
    - order_query(a, b) -> bool (is a before b?)

    Based on Dietz & Sleator, enables ~275x speedup over naive reanalysis.
    """
    def __init__(self):
        self._elements: List[OrderedElement] = []
        self._labels: Dict[OrderedElement, int] = {}

    def insert_after(self, after: Optional[OrderedElement], element: OrderedElement):
        """Insert element after 'after' in the order."""
        # ... O(1) amortized implementation with relabeling

    def query(self, a: OrderedElement, b: OrderedElement) -> bool:
        """Is a before b in the total order?"""
        return self._labels[a] < self._labels[b]

class IncrementalBidirectionalChecker:
    """
    Incremental type checker using order maintenance.

    Key insight: bidirectional typing creates dependencies between
    nodes. When a node changes, we only recheck nodes that depend on it.
    """
    def __init__(self, language: LanguageTypeSystem):
        self.language = language
        self.order = OrderMaintenanceList()
        self.dependencies = DependencyGraph()
        self.type_cache: Dict[NodeId, Type] = {}

    def check_incremental(self, old_ast: MarkedAST, new_ast: MarkedAST,
                          edit_span: SourceSpan) -> MarkedAST:
        """
        Incrementally recheck after an edit.

        1. Find affected nodes (those whose types may change)
        2. Invalidate their cached types
        3. Recheck only those nodes in dependency order
        """
        affected = self.dependencies.affected_by(edit_span)

        # Sort by order maintenance list (respects dependency order)
        affected_sorted = sorted(affected,
                                 key=lambda n: self.order.position(n))

        for node in affected_sorted:
            self._recheck_node(node, new_ast)

        return new_ast

    def observe_token(self, token: int, context: GenerationContext) -> TypeConstraint:
        """
        Called when a new token is generated.

        1. Update partial AST
        2. Incrementally recheck types (only affected portions)
        3. Update expected type for next hole
        """
        new_ast = context.parser.extend_with_token(token)
        edit_span = SourceSpan(context.position, context.position + 1)

        marked_ast = self.check_incremental(context.marked_ast, new_ast, edit_span)

        # Extract constraint for next position
        current_hole = marked_ast.find_first_unfilled_hole()
        if current_hole:
            expected = current_hole.mark.expected_type
            env = self._capture_environment(current_hole)
            return TypeConstraint(expected, env, marked_ast)

        return TYPE_TOP
```

### 6.5 ChatLSP-Style Context Extraction

Following OOPSLA 2024, we provide methods to extract rich type context:

```python
class ContextExtractor:
    """
    Extracts type context from holes for LLM guidance.

    Based on ChatLSP protocol from "Statically Contextualizing LLMs with Typed Holes"
    """

    def expected_type(self, hole: Hole, marked_ast: MarkedAST) -> ExpectedTypeInfo:
        """
        ChatLSP method 1: Get the expected type at a hole.

        Returns structured type info the LLM can use to constrain generation.
        """
        hole_mark = marked_ast.find_hole(hole.id)
        if not hole_mark:
            return ExpectedTypeInfo.unknown()

        return ExpectedTypeInfo(
            type=hole_mark.expected_type,
            type_string=self._format_type(hole_mark.expected_type),
            constraints=self._extract_constraints(hole_mark),
            examples=self._generate_examples(hole_mark.expected_type)
        )

    def relevant_types(self, hole: Hole, env: TypeEnvironment,
                       limit: int = 20) -> List[RelevantType]:
        """
        ChatLSP method 2: Get types relevant to filling this hole.

        Ranks available types by relevance to expected type.
        """
        expected = hole.expected_type
        available = env.all_bindings()

        # Score by type compatibility and usage proximity
        scored = []
        for name, typ in available.items():
            score = self._relevance_score(typ, expected, hole.context)
            scored.append((name, typ, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return [
            RelevantType(name=n, type=t, relevance=s)
            for n, t, s in scored[:limit]
        ]

    def relevant_headers(self, hole: Hole, env: TypeEnvironment,
                         limit: int = 10) -> List[FunctionSignature]:
        """
        ChatLSP method 3: Get function signatures relevant to this hole.

        Useful when the hole expects a function type or function call.
        """
        expected = hole.expected_type

        if isinstance(expected, FunctionType):
            # Hole expects a function - find compatible signatures
            return self._find_compatible_functions(expected, env, limit)

        # Hole is in expression position - find functions returning expected type
        return self._find_functions_returning(expected, env, limit)

    def error_report(self, marked_ast: MarkedAST) -> List[TypeErrorReport]:
        """
        ChatLSP method 4: Get actionable error reports.

        Errors are first-class data that can inform generation.
        """
        errors = marked_ast.collect_errors()

        return [
            TypeErrorReport(
                location=err.provenance.location,
                message=self._format_error_message(err),
                expected=err.expected,
                got=err.synthesized,
                suggestions=self._generate_suggestions(err)
            )
            for mark, err in errors
            if isinstance(mark, InconsistentMark)
        ]

    def ai_tutorial(self, hole: Hole, env: TypeEnvironment) -> str:
        """
        ChatLSP method 5: Generate natural language guidance.

        Explains what the hole needs in human-readable form.
        """
        expected = hole.expected_type
        relevant = self.relevant_types(hole, env, limit=5)

        tutorial = f"Fill this hole with an expression of type {expected}.\n"

        if relevant:
            tutorial += "\nAvailable bindings that might help:\n"
            for r in relevant:
                tutorial += f"  - {r.name}: {r.type}\n"

        if isinstance(expected, FunctionType):
            tutorial += f"\nThis function takes {len(expected.params)} arguments "
            tutorial += f"and returns {expected.returns}.\n"

        return tutorial

@dataclass
class ExpectedTypeInfo:
    type: Type
    type_string: str
    constraints: List[str]
    examples: List[str]

@dataclass
class RelevantType:
    name: str
    type: Type
    relevance: float

@dataclass
class TypeErrorReport:
    location: SourceSpan
    message: str
    expected: Type
    got: Type
    suggestions: List[str]
```

### 6.6 Propagation Edge Example

```python
def types_to_syntax(types: TypeConstraint, ctx: PropagationContext) -> SyntaxConstraint:
    """Type expectations restrict valid syntactic forms."""
    if types.is_top():
        return SYNTAX_TOP

    # Get forms that could produce expected type
    expected = types.expected

    if isinstance(expected, PrimitiveType):
        if expected.name == "int":
            # Restrict to: int literals, int-returning expressions
            return SyntaxConstraint.restrict_to_int_forms()
        elif expected.name == "str":
            return SyntaxConstraint.restrict_to_str_forms()

    return SYNTAX_TOP  # No restriction
```

---

## 7. Key Invariants and Correctness Guarantees

### 7.1 Totality Invariant (from Hazel)

**Every partial program has a well-defined type.**

```
∀ e ∈ PartialAST, ∀ Γ ∈ TypeEnvironment, ∀ τ ∈ Type:
  totalize(e, τ, Γ) returns a MarkedAST with synthesized_type ≠ ⊥
```

This means:
- Generation never gets "stuck" on type errors
- Errors are localized to marks, not propagated as failures
- The LLM always has type context, even in error states

### 7.2 Incrementality Guarantee (from OOPSLA 2025)

**Type checking a single-token edit is O(k) where k = affected nodes.**

```
If edit affects k AST nodes out of n total:
  Time(check_incremental) = O(k) not O(n)
```

For token-by-token generation, typically k << n, yielding ~275x speedup.

### 7.3 Bidirectional Consistency

**Synthesis and analysis agree on types for well-typed terms.**

```
If synthesize(e, Γ) = τ and analyze(e, τ', Γ) succeeds:
  τ <: τ'  (subtyping holds)
```

### 7.4 Semilattice Laws for Constraints

All constraint domains satisfy:

```python
# Idempotent
c.meet(c) == c

# Commutative
c1.meet(c2) == c2.meet(c1)

# Associative
c1.meet(c2.meet(c3)) == (c1.meet(c2)).meet(c3)

# Top is identity
c.meet(TOP) == c

# Bottom is absorbing
c.meet(BOTTOM) == BOTTOM
```

### 7.5 Propagation Monotonicity

**Constraint propagation only refines, never loosens.**

```
After propagation:
  ∀ domain d: d.constraint ⊑ d.constraint_before
```

This ensures fixpoint convergence.

### 7.6 Fill-and-Resume Soundness

**Filling a hole preserves typing for the filled region.**

```
If hole h has expected_type τ and term t : τ:
  fill(h, t) preserves all marks outside h
```

---

## 8. Dependencies

```toml
# Additional dependencies for ananke/
[project.optional-dependencies]
ananke = [
    # Immutable collections for TypeEnvironment (Hazel-style snapshots)
    "immutables>=0.20",

    # Incremental parsing (tree-sitter bindings)
    "tree-sitter>=0.22.0",
    "tree-sitter-python>=0.21.0",
    "tree-sitter-typescript>=0.21.0",
    "tree-sitter-rust>=0.21.0",
    "tree-sitter-go>=0.21.0",
    # Note: tree-sitter-zig may need manual compilation from source
    # https://github.com/maxxnino/tree-sitter-zig

    # SMT solver for semantic domain
    "z3-solver>=4.12.0",

    # Property-based testing for semilattice laws
    "hypothesis>=6.0.0",
]

# Development dependencies
[project.optional-dependencies.ananke-dev]
ananke-dev = [
    "pytest>=8.0.0",
    "pytest-benchmark>=4.0.0",
    "mypy>=1.8.0",
]
```

---

## 9. Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Syntax mask (llguidance) | ~50μs | Delegated to llguidance |
| Type mask | <500μs | Budget-limited token checking |
| Import mask | <100μs | Simple set operations |
| Control flow mask | <200μs | CFG analysis |
| Semantic mask | <1ms | Z3 query (optional) |
| Fused mask | <2ms | All domains combined |
| Constraint propagation | <3ms | Full network |
| Token observation | <5ms | Including AST update |

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Performance overhead > 10% | Lazy evaluation, caching, profiling-driven optimization |
| Type checker too slow | Budget-limited token checking, heuristic fallback |
| Language type system complexity | Start with Python, add languages incrementally |
| Z3 latency for semantic domain | Make semantic domain optional, batch queries |
| Propagation non-termination | Iteration limit, monotonicity enforcement |

---

## 11. Success Criteria

### Core Metrics

1. **Correctness**: All semilattice laws pass property-based tests (Hypothesis)
2. **Compatibility**: Existing SGLang constrained decoding unchanged when `--grammar-backend` is not `ananke`
3. **Performance**: <2ms per-token overhead for full 5-domain constraint checking
4. **Type Safety**: >50% reduction in type errors vs unconstrained generation
5. **Multi-language**: All 5 target languages (Python, TypeScript, Rust, Zig, Go) supported

### Hazel-Derived Criteria

6. **Totality**: Every partial program gets a type (no `None` or failures during generation)
7. **Error Localization**: All type errors have traced provenances pointing to specific source locations
8. **Incrementality**: Single-token type check <500μs after warm-up (demonstrating O(k) not O(n))
9. **Context Richness**: All 5 ChatLSP methods return meaningful results for typed holes
10. **Fill-and-Resume**: Backtracking via `rollback()` correctly restores type state

### Test Coverage

11. **Unit Tests**: >90% coverage for `core/`, `domains/types/marking/`, `domains/types/incremental/`
12. **Integration Tests**: End-to-end tests for each language generating 100+ token programs
13. **Property Tests**: Semilattice laws, bidirectional consistency, totality invariant
14. **Benchmark Suite**: Per-token latency, total generation time, memory usage vs baseline

---

## 12. Research References

### Hazel Research Program (Cyrus Omar et al.)

These papers provide the theoretical foundations for Ananke's type system:

- **[Statically Contextualizing Large Language Models with Typed Holes (OOPSLA 2024)](https://arxiv.org/abs/2409.00921)** — ChatLSP protocol for extracting type context from holes. Defines 5 methods: `expectedType`, `retrieveRelevantTypes`, `retrieveRelevantHeaders`, `errorReport`, `aiTutorial`. Core inspiration for Ananke's context extraction.

- **[Total Type Error Localization and Recovery with Holes (POPL 2024)](https://dl.acm.org/doi/10.1145/3632910)** — Marked lambda calculus. Bidirectional type system that assigns meaningful types to ALL partial programs via marks. Errors are localized to specific holes with traced provenances.

- **[Incremental Bidirectional Typing via Order Maintenance (OOPSLA 2025)](https://arxiv.org/abs/2504.08946)** — Order maintenance data structures for ~275x speedup. Enables O(1) amortized type updates as tokens are generated.

- **[Live Functional Programming with Typed Holes (ICFP 2019)](https://arxiv.org/abs/1805.00155)** — Original typed holes paper. Defines holes as "membranes around missing code." Establishes fill-and-resume evaluation semantics.

- **[Grove: A Bidirectionally Typed Collaborative Structure Editor Calculus (POPL 2025)](https://arxiv.org/abs/2412.18116)** — Structure editor foundations. Addresses the "edit-time encoding problem" for multi-cursor editing.

### Constrained Generation Research

- **[Type-Constrained Code Generation (PLDI 2025)](https://arxiv.org/abs/2504.09246)** — Prefix automata for type-safe generation. Proves type checking decidable for prefixes.

- **[XGrammar: Flexible and Efficient Structured Generation](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar)** — Context-free grammar masks with rollback. max_rollback_tokens=200.

- **[llguidance](https://github.com/guidance-ai/llguidance)** — Dynamic mask computation (~50μs/token). Lazy automata for efficient CFG parsing.

- **[JSONSchemaBench](https://arxiv.org/html/2501.10868v1)** — Structured output benchmarks for JSON schema conformance.

### Implementation References

- **[tree-sitter](https://tree-sitter.github.io/tree-sitter/)** — Incremental parsing library used for all 5 target languages
- **[immutables](https://github.com/MagicStack/immutables)** — Immutable collections for TypeEnvironment
- **[Z3 Theorem Prover](https://github.com/Z3Prover/z3)** — SMT solver for semantic domain

---

## 13. Implementation Checklist

### Phase 1: Core Foundations
- [ ] `core/constraint.py`
- [ ] `core/domain.py`
- [ ] `core/unified.py`
- [ ] `core/checkpoint.py`
- [ ] `core/context_extraction.py`
- [ ] `backend/grammar.py`
- [ ] `backend/backend.py`
- [ ] Modify `constrained/__init__.py`
- [ ] Modify `constrained/base_grammar_backend.py`

### Phase 2: Syntax Domain
- [ ] `domains/syntax/constraint.py`
- [ ] `domains/syntax/domain.py`
- [ ] `domains/syntax/grammars/*.py` (5 files)

### Phase 3: Type Domain — Marked Lambda Calculus
- [ ] `domains/types/constraint.py`
- [ ] `domains/types/unification.py`
- [ ] `domains/types/environment.py`
- [ ] `domains/types/marking/marks.py`
- [ ] `domains/types/marking/provenance.py`
- [ ] `domains/types/marking/marked_ast.py`
- [ ] `domains/types/marking/totalization.py`
- [ ] `domains/types/bidirectional/synthesis.py`
- [ ] `domains/types/bidirectional/analysis.py`
- [ ] `domains/types/bidirectional/subsumption.py`

### Phase 3b: Incremental Bidirectional Typing
- [ ] `domains/types/incremental/order_maintenance.py`
- [ ] `domains/types/incremental/dependency_graph.py`
- [ ] `domains/types/incremental/delta_typing.py`
- [ ] `domains/types/incremental/invalidation.py`
- [ ] `domains/types/domain.py`
- [ ] `domains/types/checker.py`

### Phase 4: Language Type Systems
- [ ] `domains/types/languages/base.py`
- [ ] `domains/types/languages/python.py`
- [ ] `domains/types/languages/typescript.py`
- [ ] `domains/types/languages/rust.py`
- [ ] `domains/types/languages/zig.py`
- [ ] `domains/types/languages/go.py`

### Phase 5: Type Domain Integration
- [ ] `parsing/partial_ast.py`
- [ ] `parsing/base.py`
- [ ] `parsing/languages/python.py`
- [ ] `parsing/languages/typescript.py`
- [ ] `parsing/languages/rust.py`
- [ ] `parsing/languages/zig.py`
- [ ] `parsing/languages/go.py`

### Phase 6: Propagation Network
- [ ] `propagation/network.py`
- [ ] `propagation/edges.py`
- [ ] `propagation/worklist.py`
- [ ] `propagation/builder.py`

### Phase 7: Import Domain
- [ ] `domains/imports/constraint.py`
- [ ] `domains/imports/domain.py`
- [ ] `domains/imports/resolvers/base.py`
- [ ] `domains/imports/resolvers/python.py`
- [ ] `domains/imports/resolvers/typescript.py`
- [ ] `domains/imports/resolvers/rust.py`
- [ ] `domains/imports/resolvers/zig.py`
- [ ] `domains/imports/resolvers/go.py`

### Phase 8: Control Flow Domain
- [ ] `domains/controlflow/constraint.py`
- [ ] `domains/controlflow/domain.py`
- [ ] `domains/controlflow/cfg.py`
- [ ] `domains/controlflow/reachability.py`

### Phase 9: Semantic Domain
- [ ] `domains/semantics/constraint.py`
- [ ] `domains/semantics/domain.py`
- [ ] `domains/semantics/smt.py`
- [ ] `domains/semantics/extractors.py`

### Phase 10: Hole Management
- [ ] `holes/hole.py`
- [ ] `holes/registry.py`
- [ ] `holes/factory.py`
- [ ] `holes/closure.py`
- [ ] `holes/environment_capture.py`
- [ ] `holes/fill_resume.py`
- [ ] `holes/strategy.py`

### Phase 11: Token Mask Fusion
- [ ] `masks/fuser.py`
- [ ] `masks/incremental.py`
- [ ] `masks/cache.py`
- [ ] `masks/lazy.py`

### Phase 12: Testing
- [ ] `tests/conftest.py`
- [ ] `tests/unit/test_constraint_algebra.py`
- [ ] `tests/unit/test_unification.py`
- [ ] `tests/unit/test_marking.py`
- [ ] `tests/unit/test_incremental.py`
- [ ] `tests/unit/test_bidirectional.py`
- [ ] `tests/unit/test_propagation.py`
- [ ] `tests/unit/test_holes.py`
- [ ] `tests/unit/test_mask_fusion.py`
- [ ] `tests/integration/test_sglang_backend.py`
- [ ] `tests/integration/test_python_types.py`
- [ ] `tests/integration/test_typescript_types.py`
- [ ] `tests/integration/test_rust_types.py`
- [ ] `tests/integration/test_zig_types.py`
- [ ] `tests/integration/test_go_types.py`
- [ ] `tests/integration/test_multi_language.py`
- [ ] `tests/property/test_semilattice_laws.py`
- [ ] `tests/property/test_totality_invariant.py`
- [ ] `tests/property/test_propagation_monotonicity.py`
- [ ] `tests/benchmark/bench_mask_computation.py`
- [ ] `tests/benchmark/bench_type_checking.py`
- [ ] `tests/benchmark/bench_propagation.py`

---

**Total Estimated Files**: ~75 Python files
**Total Estimated Lines**: ~15,000-20,000 LOC
