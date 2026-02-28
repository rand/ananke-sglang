> ---
> **STATUS: HISTORICAL DOCUMENT - IMPLEMENTATION COMPLETE**
> 
> This is the comprehensive implementation plan (v2.0) for integrating Ananke into SGLang.
> All features described in this plan have been implemented, including:
> - All 5 constraint domains (Syntax, Types, Imports, ControlFlow, Semantics)
> - Full multi-language support (7 languages)
> - Incremental bidirectional type checking
> - Typed holes based on Hazel research
> 
> For current documentation, see:
> - [ARCHITECTURE.md](../ARCHITECTURE.md) - System overview
> - [ARCHITECTURE_DEEP_DIVE.md](../ARCHITECTURE_DEEP_DIVE.md) - Technical deep dive
> 
> This document is preserved for historical reference.
> 
> ---
> 
# Ananke: Compositional Constraint System for Verified Code Generation
## Comprehensive Implementation Plan for Claude Code

**Version**: 2.0  
**Target Location**: `python/sglang/srt/constrained/ananke/`  
**Target Languages**: Python, TypeScript, Rust, Zig, Go  
**Type Domain**: Full incremental bidirectional type checker with marked lambda calculus  
**Constraint Domains**: All 5 (Syntax, Types, Imports, Control Flow, Semantics)  
**Estimated Files**: ~75 Python files across 12 subpackages  
**Key Dependencies**: tree-sitter, immutables, z3-solver, llguidance  

---

## 1. Executive Summary

Ananke extends SGLang's constrained decoding infrastructure to support **multi-domain constraint fusion** across syntax, types, imports, control flow, and semantics. The system treats code generation as a constraint satisfaction problem where each token must satisfy all active constraints simultaneously.

### 1.1 Core Innovation

Token masks from multiple domains are fused via bitwise AND, with cross-domain constraint propagation ensuring consistency. Unlike existing grammar-only constrained decoding (llguidance, xgrammar), Ananke provides:

1. **Type-Aware Generation**: Generated code is type-checked incrementally as each token is produced
2. **Multi-Domain Fusion**: Syntax, types, imports, control flow, and semantic constraints work in concert
3. **Typed Holes**: Incomplete programs remain well-defined (statically and dynamically) via Hazel foundations
4. **Progressive Refinement**: Constraints narrow as generation proceeds

### 1.2 Theoretical Foundations

Ananke draws heavily from the Hazel research program (Cyrus Omar et al.):

| Paper | Contribution to Ananke | Key Insight |
|-------|----------------------|-------------|
| **ChatLSP (OOPSLA 2024)** | Context extraction from typed holes | `expectedType`, `relevantTypes`, `relevantHeaders`, `errorReport`, `aiTutorial` methods |
| **Marked Lambda Calculus (POPL 2024)** | Total type assignment for partial programs | Every incomplete program has a well-defined type via marks |
| **Incremental Bidirectional Typing (OOPSLA 2025)** | O(1) amortized type updates | Order maintenance data structures yield ~275x speedup |
| **Live Functional Programming (ICFP 2019)** | Fill-and-resume evaluation semantics | Holes serve as membranes around missing code |
| **Grove (POPL 2025)** | Commutative structure editing | Bidirectionally typed collaborative editing |

### 1.3 Mathematical Foundations

**Constraint Semilattice**: All constraints form bounded meet-semilattices ⟨C, ⊓, ⊤, ⊥⟩ where:
- **C** is the set of constraints
- **⊓** (meet) is constraint conjunction
- **⊤** (top) is the trivial constraint (always satisfied)
- **⊥** (bottom) is the absurd constraint (never satisfied)

**Properties** (must be verified by property-based tests):
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
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AnankeBackend                                   │
│                (extends BaseGrammarBackend, registered via grammar_backend_registry)         │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────────┐
│                          PropagationNetwork                                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │
│  │  Syntax   │◄─┤   Types   │◄─┤  Imports  │◄─┤ CtrlFlow  │◄─┤ Semantics │ │
│  │  Domain   │  │  Domain   │  │  Domain   │  │  Domain   │  │  Domain   │ │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘ │
│        │              │              │              │              │        │
│        ▼              ▼              ▼              ▼              ▼        │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │                        TokenMaskFuser                                  ││
│  │              (bitwise AND with selectivity ordering)                   ││
│  └───────────────────────────────┬────────────────────────────────────────┘│
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   ▼
                       int32 bitmask tensor → logits
```

---

## 3. Directory Structure

```
python/sglang/srt/constrained/ananke/
├── __init__.py
├── py.typed                           # PEP 561 marker
│
├── core/                              # Constraint algebra foundations
│   ├── __init__.py
│   ├── constraint.py                  # Constraint ABC, Satisfiability enum
│   ├── domain.py                      # ConstraintDomain ABC, GenerationContext
│   ├── unified.py                     # UnifiedConstraint (product of 5 domains)
│   ├── checkpoint.py                  # State checkpointing for backtracking
│   └── context_extraction.py          # ChatLSP-style context extraction (5 methods)
│
├── domains/                           # 5 constraint domain implementations
│   ├── __init__.py
│   │
│   ├── syntax/                        # Wraps llguidance for grammar constraints
│   │   ├── __init__.py
│   │   ├── constraint.py              # SyntaxConstraint dataclass
│   │   ├── domain.py                  # SyntaxDomain (delegates to llguidance)
│   │   └── grammars/                  # Language-specific grammar definitions
│   │       ├── __init__.py
│   │       ├── python.py              # Python grammar (Lark format)
│   │       ├── typescript.py          # TypeScript grammar
│   │       ├── rust.py                # Rust grammar
│   │       ├── zig.py                 # Zig grammar
│   │       └── go.py                  # Go grammar
│   │
│   ├── types/                         # Full incremental bidirectional type checker
│   │   ├── __init__.py
│   │   ├── constraint.py              # TypeConstraint, Type hierarchy
│   │   ├── domain.py                  # TypeDomain
│   │   ├── unification.py             # Robinson's unification + occurs check
│   │   ├── environment.py             # TypeEnvironment (immutable Map)
│   │   │
│   │   ├── marking/                   # Marked lambda calculus (POPL 2024)
│   │   │   ├── __init__.py
│   │   │   ├── marks.py               # Mark types: HoleMark, InconsistentMark, NonEmptyHoleMark
│   │   │   ├── provenance.py          # Traced error provenances with source spans
│   │   │   ├── marked_ast.py          # MarkedAST with mark annotations per node
│   │   │   └── totalization.py        # Total type assignment for partial programs
│   │   │
│   │   ├── incremental/               # Order maintenance (OOPSLA 2025)
│   │   │   ├── __init__.py
│   │   │   ├── order_maintenance.py   # Dietz-Sleator O(1) amortized sequence ordering
│   │   │   ├── dependency_graph.py    # Fine-grained bidirectional type dependencies
│   │   │   ├── delta_typing.py        # Small-step type update propagation
│   │   │   └── invalidation.py        # Minimal recomputation on change
│   │   │
│   │   ├── bidirectional/             # Bidirectional type checking modes
│   │   │   ├── __init__.py
│   │   │   ├── synthesis.py           # Type synthesis (bottom-up inference)
│   │   │   ├── analysis.py            # Type analysis (top-down checking)
│   │   │   └── subsumption.py         # Subtype checking with variance
│   │   │
│   │   ├── checker.py                 # IncrementalBidirectionalChecker (main entry point)
│   │   │
│   │   └── languages/                 # Language-specific type systems
│   │       ├── __init__.py
│   │       ├── base.py                # LanguageTypeSystem ABC
│   │       ├── python.py              # Python type system (mypy/pyright compatible)
│   │       ├── typescript.py          # TypeScript type system
│   │       ├── rust.py                # Rust type system (ownership-aware)
│   │       ├── zig.py                 # Zig type system (comptime-aware)
│   │       └── go.py                  # Go type system (interfaces)
│   │
│   ├── imports/                       # Module/package constraint domain
│   │   ├── __init__.py
│   │   ├── constraint.py              # ImportConstraint (required, forbidden, versions)
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
│   ├── controlflow/                   # Control flow graph constraints
│   │   ├── __init__.py
│   │   ├── constraint.py              # ControlFlowConstraint
│   │   ├── domain.py                  # ControlFlowDomain
│   │   ├── cfg.py                     # CFGSketch representation with holes
│   │   └── reachability.py            # Reachability and termination analysis
│   │
│   └── semantics/                     # SMT-based semantic constraints
│       ├── __init__.py
│       ├── constraint.py              # SemanticConstraint (SMT formula set)
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
│   ├── factory.py                     # Dynamic hole creation from AST gaps
│   ├── closure.py                     # HoleClosure for fill-and-resume
│   ├── environment_capture.py         # Capture typing environment at hole site
│   ├── fill_resume.py                 # Live evaluation around holes
│   └── strategy.py                    # Hole selection strategies
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
│   └── languages/                     # tree-sitter integrations
│       ├── __init__.py
│       ├── python.py                  # PythonIncrementalParser
│       ├── typescript.py              # TypeScriptIncrementalParser
│       ├── rust.py                    # RustIncrementalParser
│       ├── zig.py                     # ZigIncrementalParser
│       └── go.py                      # GoIncrementalParser
│
├── backend/                           # SGLang integration layer
│   ├── __init__.py
│   ├── grammar.py                     # AnankeGrammar (BaseGrammarObject)
│   └── backend.py                     # AnankeBackend (BaseGrammarBackend)
│
└── tests/                             # Comprehensive test suite
    ├── __init__.py
    ├── conftest.py                    # Shared fixtures
    │
    ├── unit/                          # Unit tests
    │   ├── test_constraint_algebra.py # Semilattice property tests
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

## 4. SGLang Integration Points

### 4.1 Files to Modify

| File | Modification |
|------|--------------|
| `constrained/__init__.py` | Add `from .ananke import AnankeBackend` |
| `constrained/base_grammar_backend.py:~199` | Add `"ananke": create_ananke_backend` to `create_grammar_backend()` dispatch |
| `server_args.py` | Document `--grammar-backend ananke` option |

### 4.2 BaseGrammarBackend Protocol

Ananke must implement the `BaseGrammarBackend` protocol:

```python
class AnankeBackend(BaseGrammarBackend):
    """Ananke backend for multi-domain constrained generation."""
    
    def __init__(
        self,
        tokenizer_path: str,
        tokenizer_args_dict: Dict[str, Any],
        skip_tokenizer_init: bool = False,
        whitespace_pattern: Optional[str] = None,
        allow_jump_forward: bool = True,
    ):
        # Initialize llguidance as syntax backend
        # Initialize type domain with incremental checker
        # Initialize propagation network
        ...
    
    async def init_value_async(
        self,
        grammar_str: str,
        constraint_type: ConstraintType,
    ) -> AnankeGrammar:
        """Create grammar object from constraint specification."""
        ...
    
    def dispatch_fill_vocab_mask(
        self,
        grammar: AnankeGrammar,
        vocab_mask: torch.Tensor,
        idx: int,
    ) -> None:
        """Fill vocabulary mask with allowed tokens."""
        ...
```

### 4.3 AnankeGrammar Protocol

```python
class AnankeGrammar(BaseGrammarObject):
    """Grammar object wrapping unified constraints."""
    
    def __init__(
        self,
        syntax_grammar: LLGuidanceGrammar,
        constraint: UnifiedConstraint,
        network: PropagationNetwork,
        hole_registry: HoleRegistry,
        fuser: TokenMaskFuser,
    ):
        self.syntax_grammar = syntax_grammar
        self.constraint = constraint
        self.network = network
        self.hole_registry = hole_registry
        self.fuser = fuser
    
    def accept_token(self, token: int) -> bool:
        """Accept a token and update all domain constraints."""
        # 1. Update syntax grammar
        self.syntax_grammar.accept_token(token)
        
        # 2. Update each domain constraint
        context = self._make_context()
        new_constraints = {}
        for domain_name, domain in self.network.domains.items():
            old = getattr(self.constraint, domain_name)
            new = domain.observe_token(old, token, context)
            new_constraints[domain_name] = new
        
        # 3. Rebuild unified constraint
        self.constraint = UnifiedConstraint(**new_constraints)
        
        # 4. Propagate cross-domain
        return self.network.propagate()
    
    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        """Fill vocabulary mask with tokens allowed by all domains."""
        # 1. Get syntax mask from llguidance
        self.syntax_grammar.fill_vocab_mask(vocab_mask, idx)
        
        # 2. Compute additional domain masks
        additional_mask = self.fuser.compute_fused_mask(
            self.constraint,
            self._make_context(),
            exclude_syntax=True,
        )
        
        # 3. Apply additional restrictions via bitwise AND
        self._apply_additional_mask(vocab_mask, idx, additional_mask)
    
    def try_jump_forward(self) -> Optional[List[int]]:
        """Try to determine next tokens without mask computation."""
        return self.syntax_grammar.try_jump_forward()
    
    def is_terminated(self) -> bool:
        """Check if generation is complete."""
        return self.syntax_grammar.is_terminated()
    
    def rollback(self, num_tokens: int) -> None:
        """Rollback state by num_tokens."""
        # Restore from checkpoint
        ...
```

---

## 5. Implementation Phases

### Phase 1: Core Foundations (Week 1)

**Goal**: Establish constraint algebra and SGLang backend skeleton that passes through to llguidance

**Files to Create**:

1. **`core/constraint.py`** — Base constraint ABC
```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TypeVar, final

class Satisfiability(Enum):
    """Three-valued satisfiability result."""
    SAT = auto()      # Definitely satisfiable
    UNSAT = auto()    # Definitely unsatisfiable
    UNKNOWN = auto()  # Cannot determine

C = TypeVar("C", bound="Constraint")

class Constraint(ABC):
    """
    Base class for all constraints in Ananke.
    
    Constraints form a bounded meet-semilattice:
    - meet(⊤) = identity
    - meet(⊥) = ⊥
    - idempotent, commutative, associative
    
    Implementations must be immutable, hashable, and comparable.
    """
    
    __slots__ = ()
    
    @abstractmethod
    def meet(self: C, other: C) -> C:
        """Compute greatest lower bound (conjunction)."""
        ...
    
    @abstractmethod
    def satisfiability(self) -> Satisfiability:
        """Check satisfiability of this constraint."""
        ...
    
    @abstractmethod
    def is_top(self) -> bool:
        """Is this the trivial (unconstrained) element?"""
        ...
    
    @abstractmethod
    def is_bottom(self) -> bool:
        """Is this the absurd (unsatisfiable) element?"""
        ...
    
    @abstractmethod
    def __hash__(self) -> int:
        """Required for caching."""
        ...
    
    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Required for change detection."""
        ...
    
    @final
    def __le__(self, other: Constraint) -> bool:
        """Partial order: self ⊑ other iff self ⊓ other = self."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.meet(other) == self
    
    @final
    def __and__(self: C, other: C) -> C:
        """Syntactic sugar: c1 & c2 = c1.meet(c2)."""
        return self.meet(other)
```

2. **`core/domain.py`** — Constraint domain ABC
```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar
import numpy as np

if TYPE_CHECKING:
    from ananke.core.constraint import Constraint
    from ananke.parsing.partial_ast import PartialAST
    from tokenizers import Tokenizer

C = TypeVar("C", bound="Constraint")

@dataclass(frozen=True)
class GenerationContext:
    """Context for token mask computation."""
    tokenizer: Tokenizer
    vocab_size: int
    position: int
    partial_ast: PartialAST
    generated_tokens: tuple[int, ...] = field(default_factory=tuple)
    language: str = "python"
    
    def advance(self, token: int) -> GenerationContext:
        """Create new context after generating a token."""
        return GenerationContext(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            position=self.position + 1,
            partial_ast=self.partial_ast,
            generated_tokens=(*self.generated_tokens, token),
            language=self.language,
        )
    
    def recent_tokens(self, n: int) -> tuple[int, ...]:
        """Get n most recent tokens."""
        return self.generated_tokens[-n:]
    
    def with_ast(self, ast: PartialAST) -> GenerationContext:
        """Create new context with updated AST."""
        return GenerationContext(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            position=self.position,
            partial_ast=ast,
            generated_tokens=self.generated_tokens,
            language=self.language,
        )

@dataclass
class DomainCheckpoint:
    """Opaque checkpoint for domain state."""
    domain_name: str
    state: object

class ConstraintDomain(ABC, Generic[C]):
    """
    A constraint domain with its own semilattice structure.
    
    Each domain D is a functor from contexts to constraint semilattices:
    D : Context → ConstraintSemilattice
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Domain identifier."""
        ...
    
    @abstractmethod
    def top(self) -> C:
        """The trivial constraint (everything allowed)."""
        ...
    
    @abstractmethod
    def bottom(self) -> C:
        """The absurd constraint (nothing allowed)."""
        ...
    
    @abstractmethod
    def token_mask(self, constraint: C, context: GenerationContext) -> np.ndarray:
        """
        Project constraint to a token mask.
        
        Returns boolean array of shape (vocab_size,) where True = allowed.
        Performance target: <1ms typical, <50ms worst case.
        """
        ...
    
    @abstractmethod
    def observe_token(
        self, constraint: C, token: int, context: GenerationContext
    ) -> C:
        """
        Update constraint after observing a generated token.
        
        Returns refined constraint incorporating new information.
        """
        ...
    
    @abstractmethod
    def checkpoint(self) -> DomainCheckpoint:
        """Save current state for backtracking."""
        ...
    
    @abstractmethod
    def restore(self, checkpoint: DomainCheckpoint) -> None:
        """Restore to a previous state."""
        ...
    
    def token_valid(
        self, constraint: C, token: int, context: GenerationContext
    ) -> bool:
        """Check if a single token is valid."""
        return self.token_mask(constraint, context)[token]
```

3. **`core/unified.py`** — Product domain
4. **`backend/grammar.py`** — AnankeGrammar skeleton
5. **`backend/backend.py`** — AnankeBackend skeleton

**Milestone**: `--grammar-backend ananke` works identically to `--grammar-backend llguidance`

---

### Phase 2: Syntax Domain (Week 2)

**Goal**: Proper syntax domain wrapper with constraint algebra semantics

**Files to Create**:
1. `domains/syntax/constraint.py` — SyntaxConstraint with grammar + state
2. `domains/syntax/domain.py` — SyntaxDomain wrapping llguidance
3. `domains/syntax/grammars/python.py` — Python grammar in Lark format
4. `domains/syntax/grammars/typescript.py` — TypeScript grammar
5. `domains/syntax/grammars/rust.py` — Rust grammar
6. `domains/syntax/grammars/zig.py` — Zig grammar
7. `domains/syntax/grammars/go.py` — Go grammar

**Key Implementation Details**:

```python
@dataclass(frozen=True, slots=True)
class SyntaxConstraint(Constraint):
    """
    Constraint on syntactic structure.
    
    Wraps llguidance grammar string and parser state hash.
    Actual parser state is managed by SyntaxDomain.
    """
    grammar: str
    state_hash: int
    _is_dead: bool = False
    _is_accepting: bool = False
    
    def meet(self, other: SyntaxConstraint) -> SyntaxConstraint:
        """Grammar intersection via tracking both."""
        if self._is_dead or other._is_dead:
            return SYNTAX_BOTTOM
        
        if self.grammar == other.grammar:
            # Same grammar: take more constrained state
            return self if self.state_hash <= other.state_hash else other
        
        # Different grammars: combine (approximation)
        combined = f"{self.grammar}\n---\n{other.grammar}"
        return SyntaxConstraint(
            grammar=combined,
            state_hash=hash((self.state_hash, other.state_hash)),
        )
```

**Milestone**: Syntax constraints work through Ananke with proper semilattice semantics

---

### Phase 3: Type Domain — Marked Lambda Calculus (Weeks 3-4)

**Goal**: Full incremental type checker based on Hazel foundations

#### Phase 3a: Core Type System (Week 3)

**Files to Create**:
1. `domains/types/constraint.py` — Type hierarchy and TypeConstraint
2. `domains/types/unification.py` — Robinson's unification algorithm
3. `domains/types/environment.py` — Immutable type environment

**Type Hierarchy**:
```python
@dataclass(frozen=True, slots=True)
class TypeVar:
    """Unification variable."""
    name: str
    id: int

@dataclass(frozen=True, slots=True)
class PrimitiveType:
    """Primitive type: int, str, bool, float, None."""
    name: str

@dataclass(frozen=True, slots=True)
class FunctionType:
    """Function type: (T1, T2, ...) -> R."""
    params: tuple[tuple[str, Type], ...]
    returns: Type

@dataclass(frozen=True, slots=True)
class ListType:
    """List type: List[T]."""
    element: Type

@dataclass(frozen=True, slots=True)
class DictType:
    """Dict type: Dict[K, V]."""
    key: Type
    value: Type

@dataclass(frozen=True, slots=True)
class UnionType:
    """Union type: T1 | T2."""
    alternatives: frozenset[Type]

@dataclass(frozen=True, slots=True)
class ClassType:
    """Nominal class type."""
    name: str
    module: str
    type_params: tuple[Type, ...] = ()

@dataclass(frozen=True, slots=True)
class HoleType:
    """Type of a typed hole (unifies with anything)."""
    hole_id: HoleId

# Lattice elements
class AnyType:
    """Top of type lattice (⊤)."""
    pass

class NeverType:
    """Bottom of type lattice (⊥)."""
    pass

Type = TypeVar | PrimitiveType | FunctionType | ListType | DictType | UnionType | ClassType | HoleType | AnyType | NeverType
```

#### Phase 3b: Marked Lambda Calculus (Week 3)

**Files to Create**:
1. `domains/types/marking/marks.py` — Mark types
2. `domains/types/marking/provenance.py` — Error provenance tracking
3. `domains/types/marking/marked_ast.py` — AST with mark annotations
4. `domains/types/marking/totalization.py` — Total type assignment

**Mark Types** (from POPL 2024):
```python
@dataclass(frozen=True)
class Mark:
    """Base class for marks."""
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
class NonEmptyHoleMark(Mark):
    """Hole with partial content."""
    hole_id: HoleId
    inner: MarkedAST

@dataclass(frozen=True)
class Provenance:
    """Traces error back to source location."""
    location: SourceSpan
    context: str  # e.g., "function argument", "return type"
    parent: Optional[Provenance] = None

def totalize(ast: PartialAST, expected: Type, env: TypeEnvironment) -> MarkedAST:
    """
    CRITICAL: Total type assignment for partial programs.
    
    Every partial AST gets a well-defined type via marking.
    Type mismatches become InconsistentMarks, not failures.
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
    
    # Type mismatch: mark as inconsistent but CONTINUE
    return MarkedAST(
        ast,
        InconsistentMark(
            synthesized.type, expected,
            Provenance(ast.span, "expression")
        )
    )
```

#### Phase 3c: Bidirectional Type Checking (Week 3-4)

**Files to Create**:
1. `domains/types/bidirectional/synthesis.py` — Type synthesis (bottom-up)
2. `domains/types/bidirectional/analysis.py` — Type analysis (top-down)
3. `domains/types/bidirectional/subsumption.py` — Subtype checking

**Bidirectional Modes**:
```python
def synthesize(ast: PartialAST, env: TypeEnvironment) -> tuple[Type, MarkedAST]:
    """
    Synthesis mode: infer type from expression (bottom-up).
    
    Handles: literals, variables, function applications
    """
    match ast:
        case IntLiteral(_):
            return (PrimitiveType("int"), MarkedAST(ast))
        case StringLiteral(_):
            return (PrimitiveType("str"), MarkedAST(ast))
        case Identifier(name):
            typ = env.lookup(name)
            return (typ, MarkedAST(ast)) if typ else (AnyType(), MarkedAST(ast, HoleMark(...)))
        case FunctionCall(func, args):
            func_type, func_marked = synthesize(func, env)
            # Check argument types...
            return (func_type.returns, MarkedAST(ast, children=[func_marked, ...]))
        case _:
            # Unknown: mark as hole
            return (AnyType(), MarkedAST(ast, HoleMark(...)))

def analyze(ast: PartialAST, expected: Type, env: TypeEnvironment) -> MarkedAST:
    """
    Analysis mode: check expression against expected type (top-down).
    
    Handles: lambdas, let-bindings, conditionals
    """
    match ast:
        case Lambda(params, body):
            if not isinstance(expected, FunctionType):
                return MarkedAST(ast, InconsistentMark(...))
            # Extend env with parameter types
            new_env = env.extend(dict(zip(params, expected.params)))
            body_marked = analyze(body, expected.returns, new_env)
            return MarkedAST(ast, children=[body_marked])
        case _:
            # Fall back to synthesis + subsumption check
            synth_type, marked = synthesize(ast, env)
            if not subsumes(synth_type, expected):
                marked.mark = InconsistentMark(synth_type, expected, ...)
            return marked
```

#### Phase 3d: Incremental Order Maintenance (Week 4)

**Files to Create**:
1. `domains/types/incremental/order_maintenance.py` — Dietz-Sleator algorithm
2. `domains/types/incremental/dependency_graph.py` — Type dependency tracking
3. `domains/types/incremental/delta_typing.py` — Small-step updates
4. `domains/types/incremental/invalidation.py` — Cache invalidation

**Order Maintenance** (from OOPSLA 2025):
```python
class OrderMaintenanceList:
    """
    Maintains total order with O(1) amortized operations.
    
    Based on Dietz & Sleator algorithm.
    Enables ~275x speedup over naive reanalysis.
    """
    
    def __init__(self):
        self._elements: list[OrderedElement] = []
        self._labels: dict[OrderedElement, int] = {}
        self._label_space = 2**62  # Large label space
    
    def insert_after(
        self, after: Optional[OrderedElement], element: OrderedElement
    ) -> None:
        """Insert element after 'after' in the order. O(1) amortized."""
        if after is None:
            # Insert at beginning
            new_label = self._labels[self._elements[0]] // 2 if self._elements else self._label_space // 2
        else:
            after_idx = self._elements.index(after)
            after_label = self._labels[after]
            if after_idx + 1 < len(self._elements):
                next_label = self._labels[self._elements[after_idx + 1]]
                new_label = (after_label + next_label) // 2
                if new_label == after_label:
                    # Labels exhausted: relabel
                    self._relabel()
                    return self.insert_after(after, element)
            else:
                new_label = after_label + (self._label_space - after_label) // 2
        
        # Insert
        insert_idx = self._elements.index(after) + 1 if after else 0
        self._elements.insert(insert_idx, element)
        self._labels[element] = new_label
    
    def query(self, a: OrderedElement, b: OrderedElement) -> bool:
        """Is a before b in the total order? O(1)."""
        return self._labels[a] < self._labels[b]
    
    def _relabel(self) -> None:
        """Relabel all elements with evenly spaced labels."""
        n = len(self._elements)
        step = self._label_space // (n + 1)
        for i, elem in enumerate(self._elements):
            self._labels[elem] = (i + 1) * step

class IncrementalBidirectionalChecker:
    """
    Incremental type checker using order maintenance.
    
    Key insight: bidirectional typing creates dependencies.
    When a node changes, only recheck nodes that depend on it.
    """
    
    def __init__(self, language: LanguageTypeSystem):
        self.language = language
        self.order = OrderMaintenanceList()
        self.dependencies = DependencyGraph()
        self.type_cache: dict[NodeId, Type] = {}
    
    def check_incremental(
        self, old_ast: MarkedAST, new_ast: MarkedAST, edit_span: SourceSpan
    ) -> MarkedAST:
        """
        Incrementally recheck after an edit.
        
        1. Find affected nodes (those whose types may change)
        2. Invalidate their cached types
        3. Recheck only those nodes in dependency order
        """
        affected = self.dependencies.affected_by(edit_span)
        
        # Sort by order maintenance (respects dependency order)
        affected_sorted = sorted(affected, key=lambda n: self.order.position(n))
        
        for node in affected_sorted:
            self._recheck_node(node, new_ast)
        
        return new_ast
    
    def observe_token(self, token: int, context: GenerationContext) -> TypeConstraint:
        """
        Called when a new token is generated.
        
        1. Update partial AST via incremental parser
        2. Run incremental type check (using order maintenance)
        3. Update expected type for next hole position
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

#### Phase 3e: Type Domain Integration (Week 4)

**Files to Create**:
1. `domains/types/domain.py` — TypeDomain with token_mask
2. `domains/types/checker.py` — Main IncrementalBidirectionalChecker

**TypeDomain.token_mask Implementation**:
```python
class TypeDomain(ConstraintDomain[TypeConstraint]):
    """Type constraint domain with incremental checking."""
    
    def __init__(self, budget: int = 100):
        self._checker = IncrementalBidirectionalChecker()
        self._budget = budget
    
    @property
    def name(self) -> str:
        return "types"
    
    def token_mask(
        self, constraint: TypeConstraint, context: GenerationContext
    ) -> np.ndarray:
        """
        Compute token mask based on type constraints.
        
        Strategy:
        1. If constraint is top: allow all
        2. Get syntactically valid tokens (from syntax mask in context)
        3. For each valid token up to budget, check type compatibility
        4. Fall back to allowing all syntactically valid if over budget
        """
        if constraint.is_top():
            return np.ones(context.vocab_size, dtype=bool)
        
        if constraint.is_bottom():
            return np.zeros(context.vocab_size, dtype=bool)
        
        # Get syntax mask (computed by syntax domain first)
        syntax_mask = getattr(context, "syntax_mask", np.ones(context.vocab_size, dtype=bool))
        
        valid_tokens = np.where(syntax_mask)[0]
        
        if len(valid_tokens) > self._budget:
            # Over budget: fall back to syntax mask only
            return syntax_mask
        
        # Check each token for type validity
        mask = syntax_mask.copy()
        partial_ast = context.partial_ast
        
        for token_id in valid_tokens:
            token_str = context.tokenizer.decode([token_id])
            
            # Try extending AST
            extended_ast = partial_ast.extend_tentative(token_str)
            if extended_ast is None:
                mask[token_id] = False
                continue
            
            # Check if extension could be well-typed
            result = self._checker.check_partial(
                extended_ast,
                constraint.expected,
                constraint.environment,
            )
            
            if result == TypeCheckResult.DEFINITELY_ILL_TYPED:
                mask[token_id] = False
        
        return mask
```

**Milestone**: Marked lambda calculus passes totality tests — every partial program has a type

---

### Phase 4: Language-Specific Type Systems (Weeks 5-6)

**Goal**: Type systems for Python, TypeScript, Rust, Zig, Go

**Files to Create**:
1. `domains/types/languages/base.py` — LanguageTypeSystem ABC
2. `domains/types/languages/python.py` — Python type system
3. `domains/types/languages/typescript.py` — TypeScript type system
4. `domains/types/languages/rust.py` — Rust type system (ownership)
5. `domains/types/languages/zig.py` — Zig type system (comptime)
6. `domains/types/languages/go.py` — Go type system (interfaces)

**Language Type System ABC**:
```python
class LanguageTypeSystem(ABC):
    """Abstract base for language-specific type systems."""
    
    @abstractmethod
    def parse_type_annotation(self, annotation: str) -> Type:
        """Parse a type annotation string."""
        ...
    
    @abstractmethod
    def infer_literal_type(self, literal: AST) -> Type:
        """Infer type from a literal value."""
        ...
    
    @abstractmethod
    def get_builtin_types(self) -> dict[str, Type]:
        """Get built-in type definitions."""
        ...
    
    @abstractmethod
    def check_assignable(self, source: Type, target: Type) -> bool:
        """Check if source is assignable to target."""
        ...
    
    @abstractmethod
    def get_member_type(self, receiver: Type, member: str) -> Optional[Type]:
        """Get type of a member access."""
        ...
    
    @abstractmethod
    def get_call_return_type(
        self, callee: Type, args: list[Type]
    ) -> Optional[Type]:
        """Get return type of a function call."""
        ...
```

**Python Type System** (mypy/pyright compatible):
```python
class PythonTypeSystem(LanguageTypeSystem):
    """Python type system compatible with mypy/pyright."""
    
    def __init__(self):
        self._builtins = {
            "int": PrimitiveType("int"),
            "str": PrimitiveType("str"),
            "bool": PrimitiveType("bool"),
            "float": PrimitiveType("float"),
            "bytes": PrimitiveType("bytes"),
            "None": PrimitiveType("None"),
            "object": ClassType("object", "builtins"),
        }
        
        # Generic types
        self._generic_aliases = {
            "List": lambda t: ListType(t),
            "Dict": lambda k, v: DictType(k, v),
            "Set": lambda t: ClassType("set", "builtins", (t,)),
            "Tuple": lambda *ts: ClassType("tuple", "builtins", ts),
            "Optional": lambda t: UnionType(frozenset([t, PrimitiveType("None")])),
            "Union": lambda *ts: UnionType(frozenset(ts)),
            "Callable": lambda params, ret: FunctionType(
                tuple(("_", p) for p in params), ret
            ),
        }
    
    def parse_type_annotation(self, annotation: str) -> Type:
        """Parse Python type annotation."""
        # Handle generics: List[int], Dict[str, int], etc.
        if "[" in annotation:
            base, args = annotation.split("[", 1)
            args = args.rstrip("]")
            # Recursive parsing of type arguments...
            ...
        
        # Handle simple types
        if annotation in self._builtins:
            return self._builtins[annotation]
        
        # Handle class types
        return ClassType(annotation, "__main__")
    
    def check_assignable(self, source: Type, target: Type) -> bool:
        """Check assignability with Python semantics."""
        # Any is universally assignable
        if isinstance(target, AnyType) or isinstance(source, AnyType):
            return True
        
        # Same type
        if source == target:
            return True
        
        # Union types
        if isinstance(target, UnionType):
            return any(self.check_assignable(source, alt) for alt in target.alternatives)
        
        # Structural subtyping for protocols (simplified)
        if isinstance(source, ClassType) and isinstance(target, ClassType):
            # Check inheritance or protocol conformance
            ...
        
        return False
```

**Rust Type System** (ownership-aware):
```python
class RustTypeSystem(LanguageTypeSystem):
    """Rust type system with ownership tracking."""
    
    def __init__(self):
        self._builtins = {
            "i8": PrimitiveType("i8"),
            "i16": PrimitiveType("i16"),
            "i32": PrimitiveType("i32"),
            "i64": PrimitiveType("i64"),
            "i128": PrimitiveType("i128"),
            "isize": PrimitiveType("isize"),
            "u8": PrimitiveType("u8"),
            "u16": PrimitiveType("u16"),
            "u32": PrimitiveType("u32"),
            "u64": PrimitiveType("u64"),
            "u128": PrimitiveType("u128"),
            "usize": PrimitiveType("usize"),
            "f32": PrimitiveType("f32"),
            "f64": PrimitiveType("f64"),
            "bool": PrimitiveType("bool"),
            "char": PrimitiveType("char"),
            "str": PrimitiveType("str"),
            "String": ClassType("String", "std::string"),
        }
    
    def check_borrow_valid(
        self, borrower: Type, borrowed: Type, mutable: bool
    ) -> bool:
        """Check if borrow is valid."""
        # Simplified ownership checking
        # Real implementation would track lifetimes
        ...
```

**Milestone**: All 5 language type systems pass conformance tests

---

### Phase 5: ChatLSP Context Extraction (Week 6)

**Goal**: Implement all 5 ChatLSP methods for rich context extraction

**File to Create**: `core/context_extraction.py`

```python
class ContextExtractor:
    """
    Extracts type context from holes for LLM guidance.
    
    Based on ChatLSP protocol from OOPSLA 2024:
    "Statically Contextualizing LLMs with Typed Holes"
    """
    
    def expected_type(self, hole: Hole, marked_ast: MarkedAST) -> ExpectedTypeInfo:
        """
        ChatLSP Method 1: Get expected type at a hole.
        
        Returns structured type info the LLM can use to constrain generation.
        """
        hole_mark = marked_ast.find_hole(hole.id)
        if not hole_mark:
            return ExpectedTypeInfo.unknown()
        
        return ExpectedTypeInfo(
            type=hole_mark.expected_type,
            type_string=self._format_type(hole_mark.expected_type),
            constraints=self._extract_constraints(hole_mark),
            examples=self._generate_examples(hole_mark.expected_type),
        )
    
    def relevant_types(
        self, hole: Hole, env: TypeEnvironment, limit: int = 20
    ) -> list[RelevantType]:
        """
        ChatLSP Method 2: Get types relevant to filling this hole.
        
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
    
    def relevant_headers(
        self, hole: Hole, env: TypeEnvironment, limit: int = 10
    ) -> list[FunctionSignature]:
        """
        ChatLSP Method 3: Get function signatures relevant to this hole.
        
        Useful when hole expects a function type or function call.
        """
        expected = hole.expected_type
        
        if isinstance(expected, FunctionType):
            # Hole expects a function - find compatible signatures
            return self._find_compatible_functions(expected, env, limit)
        
        # Hole is in expression position - find functions returning expected type
        return self._find_functions_returning(expected, env, limit)
    
    def error_report(self, marked_ast: MarkedAST) -> list[TypeErrorReport]:
        """
        ChatLSP Method 4: Get actionable error reports.
        
        Errors are first-class data that can inform generation.
        """
        errors = marked_ast.collect_errors()
        
        return [
            TypeErrorReport(
                location=err.provenance.location,
                message=self._format_error_message(err),
                expected=err.expected,
                got=err.synthesized,
                suggestions=self._generate_suggestions(err),
            )
            for mark, err in errors
            if isinstance(mark, InconsistentMark)
        ]
    
    def ai_tutorial(self, hole: Hole, env: TypeEnvironment) -> str:
        """
        ChatLSP Method 5: Generate natural language guidance.
        
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
    
    def _relevance_score(
        self, typ: Type, expected: Type, context: HoleContext
    ) -> float:
        """Score type relevance to expected type."""
        # Exact match: highest score
        if typ == expected:
            return 1.0
        
        # Subtype: high score
        if self._is_subtype(typ, expected):
            return 0.9
        
        # Function returning expected type
        if isinstance(typ, FunctionType) and typ.returns == expected:
            return 0.8
        
        # Contains expected type (e.g., List[expected])
        if self._contains_type(typ, expected):
            return 0.6
        
        # Lexically close to hole
        return 0.3 * context.lexical_distance_factor(typ)
```

**Milestone**: All 5 ChatLSP methods return meaningful results for typed holes

---

### Phase 6: Propagation Network (Week 7)

**Goal**: Cross-domain constraint propagation

**Files to Create**:
1. `propagation/network.py` — PropagationNetwork
2. `propagation/edges.py` — Propagation edge definitions
3. `propagation/worklist.py` — Priority worklist
4. `propagation/builder.py` — Network builder

**PropagationNetwork**:
```python
class PropagationNetwork:
    """
    Constraint propagation network using worklist algorithm.
    
    Ensures constraints flow between domains:
    - Syntax → Types: structure constrains types
    - Types → Syntax: types restrict valid syntax
    - Types → Imports: using a type requires its import
    - Imports → Types: available imports determine available types
    - Syntax → ControlFlow: syntax determines CFG shape
    - ControlFlow → Semantics: control flow induces semantic constraints
    - Semantics → Types: semantic constraints become refinement types
    """
    
    def __init__(self):
        self.domains: dict[str, ConstraintDomain] = {}
        self.edges: list[PropagationEdge] = []
        self.constraints: dict[str, Constraint] = {}
        self._worklist: list[tuple[int, str]] = []
    
    def add_constraint(self, domain: str, constraint: Constraint) -> bool:
        """
        Add constraint to domain and propagate.
        
        Returns False if contradiction detected.
        """
        current = self.constraints[domain]
        new = current.meet(constraint)
        
        if new.is_bottom():
            return False
        
        if new != current:
            self.constraints[domain] = new
            self._enqueue_dependents(domain)
        
        return self._propagate()
    
    def _propagate(self) -> bool:
        """Run propagation until fixpoint or contradiction."""
        iterations = 0
        max_iterations = 1000  # Prevent infinite loops
        
        while self._worklist and iterations < max_iterations:
            iterations += 1
            _, target = heapq.heappop(self._worklist)
            
            # Compute induced constraint from all sources
            induced = self.domains[target].top()
            
            for edge in self.edges:
                if edge.target == target:
                    source_constraint = self.constraints[edge.source]
                    ctx = self._make_context()
                    propagated = edge.propagate(source_constraint, ctx)
                    induced = induced.meet(propagated)
            
            # Meet with current
            current = self.constraints[target]
            new = current.meet(induced)
            
            if new.is_bottom():
                return False
            
            if new != current:
                self.constraints[target] = new
                self._enqueue_dependents(target)
        
        return True
```

**Standard Propagation Edges**:
```python
def build_standard_propagation_network() -> PropagationNetwork:
    """Construct standard propagation network for code generation."""
    network = PropagationNetwork()
    
    # Register domains
    network.register_domain(SyntaxDomain())
    network.register_domain(TypeDomain())
    network.register_domain(ImportDomain())
    network.register_domain(ControlFlowDomain())
    network.register_domain(SemanticDomain())
    
    # Syntax → Types (priority 0 - highest)
    network.register_edge(PropagationEdge(
        source="syntax", target="types",
        propagate=syntax_to_types, priority=0
    ))
    
    # Types → Syntax (priority 1)
    network.register_edge(PropagationEdge(
        source="types", target="syntax",
        propagate=types_to_syntax, priority=1
    ))
    
    # Types → Imports (priority 2)
    network.register_edge(PropagationEdge(
        source="types", target="imports",
        propagate=types_to_imports, priority=2
    ))
    
    # Imports → Types (priority 2)
    network.register_edge(PropagationEdge(
        source="imports", target="types",
        propagate=imports_to_types, priority=2
    ))
    
    # Syntax → ControlFlow (priority 3)
    network.register_edge(PropagationEdge(
        source="syntax", target="control_flow",
        propagate=syntax_to_controlflow, priority=3
    ))
    
    # ControlFlow → Semantics (priority 4)
    network.register_edge(PropagationEdge(
        source="control_flow", target="semantics",
        propagate=controlflow_to_semantics, priority=4
    ))
    
    # Semantics → Types (priority 5 - lowest)
    network.register_edge(PropagationEdge(
        source="semantics", target="types",
        propagate=semantics_to_types, priority=5
    ))
    
    return network
```

**Milestone**: Constraint changes in one domain propagate to others correctly

---

### Phase 7: Import Domain (Week 8)

**Goal**: Track required/forbidden imports with version constraints

**Files to Create**:
1. `domains/imports/constraint.py` — ImportConstraint
2. `domains/imports/domain.py` — ImportDomain
3. `domains/imports/resolvers/base.py` — ImportResolver ABC
4. `domains/imports/resolvers/python.py` — Python import resolver
5. `domains/imports/resolvers/typescript.py` — TypeScript resolver
6. `domains/imports/resolvers/rust.py` — Rust resolver
7. `domains/imports/resolvers/zig.py` — Zig resolver
8. `domains/imports/resolvers/go.py` — Go resolver

**ImportConstraint**:
```python
@dataclass(frozen=True, slots=True)
class ModuleSpec:
    """Specification for a module."""
    name: str
    submodules: frozenset[str] = frozenset()

@dataclass(frozen=True, slots=True)
class VersionConstraint:
    """Semantic version constraint."""
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    
    def intersect(self, other: VersionConstraint) -> VersionConstraint:
        """Compute intersection of version ranges."""
        new_min = max(self.min_version, other.min_version) if both else self.min_version or other.min_version
        new_max = min(self.max_version, other.max_version) if both else self.max_version or other.max_version
        
        if new_min and new_max and new_min > new_max:
            return VersionConstraint.empty()
        
        return VersionConstraint(new_min, new_max)

@dataclass(frozen=True, slots=True)
class ImportConstraint(Constraint):
    """Constraint on import graph structure."""
    required: frozenset[ModuleSpec]
    forbidden: frozenset[ModuleSpec]
    versions: FrozenDict[str, VersionConstraint]
    
    def meet(self, other: ImportConstraint) -> ImportConstraint:
        new_required = self.required | other.required
        new_forbidden = self.forbidden | other.forbidden
        
        # Check contradiction: required ∩ forbidden ≠ ∅
        if new_required & new_forbidden:
            return IMPORT_BOTTOM
        
        # Merge version constraints
        new_versions = {}
        all_modules = set(self.versions.keys()) | set(other.versions.keys())
        for module in all_modules:
            v1 = self.versions.get(module, VersionConstraint.any())
            v2 = other.versions.get(module, VersionConstraint.any())
            merged = v1.intersect(v2)
            if merged.is_empty():
                return IMPORT_BOTTOM
            new_versions[module] = merged
        
        return ImportConstraint(
            required=new_required,
            forbidden=new_forbidden,
            versions=FrozenDict(new_versions),
        )
```

**Milestone**: Import constraints affect type availability

---

### Phase 8: Control Flow Domain (Week 9)

**Goal**: CFG-level constraints (reachability, termination)

**Files to Create**:
1. `domains/controlflow/constraint.py` — ControlFlowConstraint
2. `domains/controlflow/domain.py` — ControlFlowDomain
3. `domains/controlflow/cfg.py` — CFGSketch representation
4. `domains/controlflow/reachability.py` — Reachability analysis

**ControlFlowConstraint**:
```python
@dataclass(frozen=True, slots=True)
class CFGSketch:
    """Sketch of control flow graph with holes."""
    blocks: frozenset[BasicBlock]
    edges: frozenset[CFGEdge]
    holes: frozenset[CFGHole]

@dataclass(frozen=True, slots=True)
class ReachabilityConstraint:
    """Constraints on code reachability."""
    must_reach: frozenset[BlockId]      # Must be reachable from entry
    must_not_reach: frozenset[BlockId]  # Must be unreachable

@dataclass(frozen=True, slots=True)
class TerminationRequirement(Enum):
    MUST_TERMINATE = auto()      # Function must always terminate
    MAY_NOT_TERMINATE = auto()   # Function may loop forever
    NO_REQUIREMENT = auto()      # No termination requirement

@dataclass(frozen=True, slots=True)
class ControlFlowConstraint(Constraint):
    """Constraint on control flow structure."""
    cfg_sketch: CFGSketch
    reachability: ReachabilityConstraint
    termination: TerminationRequirement
    
    def meet(self, other: ControlFlowConstraint) -> ControlFlowConstraint:
        # Unify CFG sketches
        unified_sketch = self.cfg_sketch.unify(other.cfg_sketch)
        if unified_sketch is None:
            return CONTROLFLOW_BOTTOM
        
        # Intersect reachability constraints
        unified_reach = ReachabilityConstraint(
            must_reach=self.reachability.must_reach | other.reachability.must_reach,
            must_not_reach=self.reachability.must_not_reach | other.reachability.must_not_reach,
        )
        
        # Check contradiction
        if unified_reach.must_reach & unified_reach.must_not_reach:
            return CONTROLFLOW_BOTTOM
        
        # Combine termination requirements
        unified_term = self._combine_termination(self.termination, other.termination)
        
        return ControlFlowConstraint(
            cfg_sketch=unified_sketch,
            reachability=unified_reach,
            termination=unified_term,
        )
```

**Milestone**: Control flow constraints enforced during generation

---

### Phase 9: Semantic Domain (Week 10)

**Goal**: SMT-based semantic constraints via Z3

**Files to Create**:
1. `domains/semantics/constraint.py` — SemanticConstraint
2. `domains/semantics/domain.py` — SemanticDomain
3. `domains/semantics/smt.py` — Z3 integration
4. `domains/semantics/extractors.py` — Formula extraction

**SemanticDomain**:
```python
@dataclass(frozen=True, slots=True)
class SMTFormula:
    """Wrapper for Z3 expressions."""
    expr: Any  # z3.ExprRef
    
    def to_z3(self) -> z3.ExprRef:
        return self.expr

@dataclass(frozen=True, slots=True)
class SemanticConstraint(Constraint):
    """Constraint on semantic properties (SMT formulas)."""
    formulas: frozenset[SMTFormula]
    
    def meet(self, other: SemanticConstraint) -> SemanticConstraint:
        return SemanticConstraint(formulas=self.formulas | other.formulas)
    
    def satisfiability(self) -> Satisfiability:
        """Query Z3 for satisfiability."""
        from z3 import Solver, sat, unsat
        
        solver = Solver()
        for formula in self.formulas:
            solver.add(formula.to_z3())
        
        result = solver.check()
        if result == sat:
            return Satisfiability.SAT
        elif result == unsat:
            return Satisfiability.UNSAT
        else:
            return Satisfiability.UNKNOWN

class SemanticDomain(ConstraintDomain[SemanticConstraint]):
    """Semantic constraint domain backed by Z3."""
    
    def __init__(self):
        self._solver = IncrementalSMTSolver()
    
    def token_mask(
        self, constraint: SemanticConstraint, context: GenerationContext
    ) -> np.ndarray:
        """
        Semantic constraints typically don't directly constrain tokens.
        
        They constrain via projection to other domains.
        Return all-true mask and let propagation handle it.
        """
        return np.ones(context.vocab_size, dtype=bool)
    
    def observe_token(
        self, constraint: SemanticConstraint, token: int, context: GenerationContext
    ) -> SemanticConstraint:
        """Extract new semantic constraints from generated code."""
        new_formulas = extract_semantic_formulas(context.partial_ast)
        return SemanticConstraint(
            formulas=constraint.formulas | frozenset(new_formulas)
        )
```

**Milestone**: Semantic constraints (assertions, contracts) enforced

---

### Phase 10: Typed Hole Management (Week 11)

**Goal**: Full Hazel-style fill-and-resume evaluation

**Files to Create**:
1. `holes/hole.py` — Hole, HoleId, HoleGranularity
2. `holes/registry.py` — HoleRegistry
3. `holes/factory.py` — HoleFactory
4. `holes/closure.py` — HoleClosure
5. `holes/environment_capture.py` — Environment capture
6. `holes/fill_resume.py` — Fill-and-resume engine
7. `holes/strategy.py` — Hole selection strategies

**Hole and Registry**:
```python
class HoleGranularity(Enum):
    """Hierarchy of hole granularities."""
    TOKEN = 0
    EXPRESSION = 1
    STATEMENT = 2
    BLOCK = 3
    FUNCTION = 4
    MODULE = 5
    SYSTEM = 6

@dataclass(frozen=True, slots=True)
class HoleId:
    """Unique identifier for a hole."""
    namespace: str
    name: str
    index: int = 0
    depth: int = 0
    
    def __hash__(self):
        return hash((self.namespace, self.name, self.index, self.depth))

@dataclass
class Hole(Generic[C]):
    """
    A typed hole representing an unknown program fragment.
    
    Following Hazel: "Holes serve as membranes around missing code."
    """
    id: HoleId
    granularity: HoleGranularity
    expected_type: Type
    environment: TypeEnvironment  # Captured at hole site
    constraint: C
    parent: Optional[HoleId] = None
    children: set[HoleId] = field(default_factory=set)
    provenance: Optional[str] = None
    
    def refine(self, additional: C) -> bool:
        """Add a constraint, narrowing this hole."""
        new_constraint = self.constraint.meet(additional)
        if new_constraint.is_bottom():
            return False
        self.constraint = new_constraint
        return True

class HoleRegistry:
    """
    Manages the hierarchy of typed holes.
    
    Responsibilities:
    1. Track all active holes
    2. Maintain parent-child relationships
    3. Propagate refinements through hierarchy
    4. Select holes for resolution
    """
    
    def __init__(self, network: PropagationNetwork):
        self.network = network
        self.holes: dict[HoleId, Hole] = {}
        self.resolution_order: list[HoleId] = []
    
    def create_hole(
        self,
        id: HoleId,
        granularity: HoleGranularity,
        expected_type: Type,
        environment: TypeEnvironment,
        initial_constraint: Constraint,
        parent: Optional[HoleId] = None,
        provenance: Optional[str] = None,
    ) -> Hole:
        """Create a new hole in the registry."""
        hole = Hole(
            id=id,
            granularity=granularity,
            expected_type=expected_type,
            environment=environment,
            constraint=initial_constraint,
            parent=parent,
            provenance=provenance,
        )
        
        self.holes[id] = hole
        
        if parent is not None:
            self.holes[parent].children.add(id)
            # Inherit constraints from parent
            parent_constraint = self.holes[parent].constraint
            hole.refine(self._project_to_child(parent_constraint, granularity))
        
        return hole
    
    def fill(self, hole_id: HoleId, term: AST) -> bool:
        """Fill a hole with a term."""
        hole = self.holes[hole_id]
        
        # Type check the term against expected type
        result = self._type_check(term, hole.expected_type, hole.environment)
        if result.is_ill_typed:
            return False
        
        # Record resolution
        self.resolution_order.append(hole_id)
        
        return True
    
    def select_next_hole(self) -> Optional[HoleId]:
        """
        Select the next hole to resolve.
        
        Strategy: most constrained first (smallest solution space).
        Ties broken by granularity (finer first).
        """
        unresolved = [
            (id, hole) for id, hole in self.holes.items()
            if not hole.is_resolved()
        ]
        
        if not unresolved:
            return None
        
        def score(item):
            id, hole = item
            constraint_score = self._estimate_solution_space(hole.constraint)
            granularity_score = hole.granularity.value
            return (constraint_score, granularity_score)
        
        unresolved.sort(key=score)
        return unresolved[0][0]
```

**Fill-and-Resume Engine**:
```python
class FillAndResumeEngine:
    """
    Enables live evaluation around holes.
    
    Following Hazel: evaluation proceeds around holes,
    and filling a hole continues from where evaluation left off.
    """
    
    def __init__(self, registry: HoleRegistry):
        self.registry = registry
        self._closures: dict[HoleId, HoleClosure] = {}
    
    def evaluate_with_holes(self, ast: MarkedAST) -> tuple[PartialResult, dict[HoleId, HoleClosure]]:
        """
        Evaluate as far as possible, returning:
        - Partial result (may contain hole values)
        - Closures capturing evaluation state at each hole
        """
        closures = {}
        
        def eval_node(node: MarkedAST, env: EvalEnvironment) -> Value:
            if isinstance(node.mark, HoleMark):
                # Reached a hole: capture closure and return hole value
                closure = HoleClosure(
                    hole_id=node.mark.hole_id,
                    env_snapshot=env.snapshot(),
                    continuation=self._capture_continuation(),
                )
                closures[node.mark.hole_id] = closure
                return HoleValue(node.mark.hole_id)
            
            # Continue evaluation...
            match node.node:
                case IntLiteral(v):
                    return IntValue(v)
                case FunctionCall(func, args):
                    func_val = eval_node(func, env)
                    arg_vals = [eval_node(a, env) for a in args]
                    return self._apply(func_val, arg_vals)
                # ...
        
        result = eval_node(ast, EvalEnvironment.empty())
        return PartialResult(result), closures
    
    def fill_and_continue(
        self, hole_id: HoleId, term: AST, closures: dict[HoleId, HoleClosure]
    ) -> PartialResult:
        """
        Fill a hole and continue evaluation from the captured closure.
        """
        closure = closures[hole_id]
        
        # Evaluate the fill term in the captured environment
        fill_value = self._evaluate(term, closure.env_snapshot)
        
        # Continue with the captured continuation
        return closure.continuation(fill_value)
```

**Milestone**: Multi-hole generation with fill-and-resume enabling backtracking

---

### Phase 11: Token Mask Fusion (Week 12)

**Goal**: Optimized multi-domain mask computation

**Files to Create**:
1. `masks/fuser.py` — TokenMaskFuser
2. `masks/incremental.py` — IncrementalMaskComputer
3. `masks/cache.py` — LRU mask cache
4. `masks/lazy.py` — LazyConstraintEvaluator

**TokenMaskFuser**:
```python
class TokenMaskFuser:
    """
    Fuses token masks from multiple constraint domains.
    
    The fused mask is the intersection (conjunction) of individual masks.
    Optimizations:
    - Selectivity ordering (most selective first)
    - Short-circuit on all-false
    - Lazy evaluation for expensive domains
    """
    
    def __init__(self, domains: dict[str, ConstraintDomain]):
        self.domains = domains
        self._cache = MaskCache(max_size=10000)
    
    def compute_fused_mask(
        self,
        constraint: UnifiedConstraint,
        context: GenerationContext,
        exclude_syntax: bool = False,
    ) -> np.ndarray:
        """Compute fused token mask from all domains."""
        fused = np.ones(context.vocab_size, dtype=bool)
        
        # Order domains by expected selectivity
        domain_order = self._selectivity_order(constraint)
        
        if exclude_syntax:
            domain_order = [d for d in domain_order if d != "syntax"]
        
        for domain_name in domain_order:
            if not fused.any():
                # No tokens left, short-circuit
                break
            
            domain = self.domains[domain_name]
            domain_constraint = getattr(constraint, domain_name)
            
            if domain_constraint.is_top():
                continue  # Skip unconstrained domains
            
            # Check cache
            cache_key = (domain_name, hash(domain_constraint), context.position)
            if cache_key in self._cache:
                mask = self._cache[cache_key]
            else:
                mask = domain.token_mask(domain_constraint, context)
                self._cache[cache_key] = mask
            
            # Intersect
            fused &= mask
        
        return fused
    
    def _selectivity_order(self, constraint: UnifiedConstraint) -> list[str]:
        """Order domains by expected selectivity (most selective first)."""
        # Syntax is usually most selective, semantics least
        base_order = ["syntax", "types", "imports", "control_flow", "semantics"]
        
        # Could be made adaptive based on constraint specificity
        return base_order
```

**IncrementalMaskComputer**:
```python
class IncrementalMaskComputer:
    """
    Efficiently update masks as generation proceeds.
    
    Key insight: after generating token t, most of the mask computation
    from the previous step can be reused.
    """
    
    def __init__(self, fuser: TokenMaskFuser):
        self.fuser = fuser
        self._previous_masks: dict[str, np.ndarray] = {}
        self._previous_constraint: Optional[UnifiedConstraint] = None
    
    def compute_mask(
        self,
        constraint: UnifiedConstraint,
        context: GenerationContext,
    ) -> np.ndarray:
        """Compute mask incrementally when possible."""
        if self._previous_constraint is None:
            # First call: compute from scratch
            mask = self.fuser.compute_fused_mask(constraint, context)
            self._previous_masks = self._domain_masks(constraint, context)
            self._previous_constraint = constraint
            return mask
        
        # Determine which domains have changed
        changed_domains = self._changed_domains(
            self._previous_constraint, constraint
        )
        
        # Recompute only changed domains
        current_masks = dict(self._previous_masks)
        for domain_name in changed_domains:
            domain = self.fuser.domains[domain_name]
            domain_constraint = getattr(constraint, domain_name)
            current_masks[domain_name] = domain.token_mask(domain_constraint, context)
        
        # Fuse
        fused = np.ones(context.vocab_size, dtype=bool)
        for mask in current_masks.values():
            fused &= mask
        
        # Update state
        self._previous_masks = current_masks
        self._previous_constraint = constraint
        
        return fused
    
    def _changed_domains(
        self, old: UnifiedConstraint, new: UnifiedConstraint
    ) -> set[str]:
        """Identify which domains have different constraints."""
        changed = set()
        for domain in ["syntax", "types", "imports", "control_flow", "semantics"]:
            if getattr(old, domain) != getattr(new, domain):
                changed.add(domain)
        return changed
```

**Milestone**: <2ms total mask computation

---

### Phase 12: Incremental Parsing (Week 12)

**Goal**: Tree-sitter integration for all 5 languages

**Files to Create**:
1. `parsing/partial_ast.py` — PartialAST abstraction
2. `parsing/base.py` — IncrementalParser ABC
3. `parsing/languages/python.py` — PythonIncrementalParser
4. `parsing/languages/typescript.py` — TypeScriptIncrementalParser
5. `parsing/languages/rust.py` — RustIncrementalParser
6. `parsing/languages/zig.py` — ZigIncrementalParser
7. `parsing/languages/go.py` — GoIncrementalParser

**IncrementalParser**:
```python
class IncrementalParser(ABC):
    """Abstract base for language-specific incremental parsers."""
    
    @abstractmethod
    def extend_with_token(self, token: str) -> PartialAST:
        """Extend the AST with a new token."""
        ...
    
    @abstractmethod
    def extend_tentative(self, token: str) -> Optional[PartialAST]:
        """Try extending without committing (for type checking)."""
        ...
    
    @abstractmethod
    def find_holes(self) -> list[ASTHole]:
        """Find incomplete nodes that represent holes."""
        ...
    
    @abstractmethod
    def checkpoint(self) -> ParserCheckpoint:
        """Save parser state."""
        ...
    
    @abstractmethod
    def restore(self, checkpoint: ParserCheckpoint) -> None:
        """Restore parser state."""
        ...

class PythonIncrementalParser(IncrementalParser):
    """Python incremental parser using tree-sitter."""
    
    def __init__(self):
        import tree_sitter_python
        from tree_sitter import Parser
        
        self._parser = Parser()
        self._parser.language = tree_sitter_python.language()
        self._source = b""
        self._tree = None
    
    def extend_with_token(self, token: str) -> PartialAST:
        """Extend AST with new token using incremental parsing."""
        old_source = self._source
        new_source = old_source + token.encode()
        
        if self._tree:
            # Incremental update
            edit = self._tree.edit(
                start_byte=len(old_source),
                old_end_byte=len(old_source),
                new_end_byte=len(new_source),
                start_point=(self._line, self._col),
                old_end_point=(self._line, self._col),
                new_end_point=self._new_position(token),
            )
            self._tree = self._parser.parse(new_source, self._tree)
        else:
            self._tree = self._parser.parse(new_source)
        
        self._source = new_source
        return self._to_partial_ast(self._tree.root_node)
```

**Milestone**: All 5 languages have working incremental parsers

---

### Phase 13: Testing and Documentation (Weeks 13-14)

**Goal**: Comprehensive test suite and documentation

**Test Categories**:

1. **Property-Based Tests** (Hypothesis):
```python
from hypothesis import given, strategies as st

@given(st.builds(SyntaxConstraint), st.builds(SyntaxConstraint))
def test_syntax_constraint_commutativity(c1, c2):
    """Test commutativity: c1 ⊓ c2 = c2 ⊓ c1"""
    assert c1.meet(c2) == c2.meet(c1)

@given(st.builds(TypeConstraint), st.builds(TypeConstraint), st.builds(TypeConstraint))
def test_type_constraint_associativity(c1, c2, c3):
    """Test associativity: (c1 ⊓ c2) ⊓ c3 = c1 ⊓ (c2 ⊓ c3)"""
    assert c1.meet(c2).meet(c3) == c1.meet(c2.meet(c3))

@given(st.builds(PartialAST), st.builds(Type), st.builds(TypeEnvironment))
def test_totality_invariant(ast, expected, env):
    """Test totality: every partial program has a type."""
    marked = totalize(ast, expected, env)
    assert marked.synthesized_type is not None
    assert not isinstance(marked.synthesized_type, NeverType)
```

2. **Integration Tests**:
```python
def test_python_type_constrained_generation():
    """Test type-constrained generation for Python."""
    backend = AnankeBackend(tokenizer_path="...")
    
    # Create constraint for: def f(x: int) -> str: ...
    constraint = UnifiedConstraint(
        syntax=python_function_grammar,
        types=TypeConstraint(
            expected=FunctionType([("x", INT)], STR),
            environment=TypeEnvironment.empty(),
        ),
        ...
    )
    
    grammar = await backend.init_value_async(constraint_str, ConstraintType.ANANKE)
    
    # Generate and verify
    for _ in range(100):
        mask = grammar.fill_vocab_mask(...)
        # Verify mask is not all-false
        assert mask.any()
        # Sample and accept
        token = sample(mask)
        grammar.accept_token(token)
    
    # Verify result is well-typed
    result = grammar.get_generated_code()
    assert is_well_typed(result)
```

3. **Benchmarks**:
```python
def bench_mask_computation():
    """Benchmark token mask computation."""
    import time
    
    backend = AnankeBackend(...)
    grammar = ...
    
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        grammar.fill_vocab_mask(...)
        times.append(time.perf_counter() - start)
    
    avg_ms = sum(times) / len(times) * 1000
    p99_ms = sorted(times)[990] * 1000
    
    assert avg_ms < 1.0, f"Average mask computation too slow: {avg_ms}ms"
    assert p99_ms < 5.0, f"P99 mask computation too slow: {p99_ms}ms"
```

---

## 6. Performance Targets

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

## 7. Key Invariants and Correctness Guarantees

### 7.1 Totality Invariant (from Hazel)

**Every partial program has a well-defined type.**

```
∀ e ∈ PartialAST, ∀ Γ ∈ TypeEnvironment, ∀ τ ∈ Type:
  totalize(e, τ, Γ) returns a MarkedAST with synthesized_type ≠ ⊥
```

### 7.2 Incrementality Guarantee (from OOPSLA 2025)

**Type checking a single-token edit is O(k) where k = affected nodes.**

```
If edit affects k AST nodes out of n total:
  Time(check_incremental) = O(k) not O(n)
```

### 7.3 Semilattice Laws

All constraint domains satisfy:
- `c.meet(c) == c` (idempotent)
- `c1.meet(c2) == c2.meet(c1)` (commutative)
- `c1.meet(c2.meet(c3)) == (c1.meet(c2)).meet(c3)` (associative)
- `c.meet(TOP) == c` (identity)
- `c.meet(BOTTOM) == BOTTOM` (absorbing)

### 7.4 Propagation Monotonicity

**Constraint propagation only refines, never loosens.**

```
After propagation:
  ∀ domain d: d.constraint ⊑ d.constraint_before
```

---

## 8. Dependencies

```toml
[project.optional-dependencies]
ananke = [
    # Immutable collections for TypeEnvironment
    "immutables>=0.20",
    
    # Incremental parsing
    "tree-sitter>=0.22.0",
    "tree-sitter-python>=0.21.0",
    "tree-sitter-typescript>=0.21.0",
    "tree-sitter-rust>=0.21.0",
    "tree-sitter-go>=0.21.0",
    # tree-sitter-zig may need manual compilation
    
    # SMT solver for semantic domain
    "z3-solver>=4.12.0",
    
    # Property-based testing
    "hypothesis>=6.0.0",
]
```

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Performance overhead > 10% | Lazy evaluation, caching, selectivity ordering |
| Type checker too slow | Budget-limited token checking, heuristic fallback |
| Language type system complexity | Start with Python, add languages incrementally |
| Z3 latency for semantic domain | Make semantic domain optional, batch queries |
| Propagation non-termination | Iteration limit, monotonicity enforcement |
| tree-sitter-zig unavailable | Fall back to regex-based parsing |

---

## 10. Success Criteria

### Core Metrics

1. **Correctness**: All semilattice laws pass property-based tests
2. **Compatibility**: Existing SGLang constrained decoding unchanged when `--grammar-backend` is not `ananke`
3. **Performance**: <5ms per-token overhead for full 5-domain constraint checking
4. **Type Safety**: >50% reduction in type errors vs unconstrained generation
5. **Multi-language**: All 5 target languages supported

### Hazel-Derived Criteria

6. **Totality**: Every partial program gets a type (no failures during generation)
7. **Error Localization**: All type errors have traced provenances
8. **Incrementality**: Single-token type check <500μs after warm-up
9. **Context Richness**: All 5 ChatLSP methods return meaningful results
10. **Fill-and-Resume**: Backtracking correctly restores type state

---

## 11. Research References

### Hazel Research Program (Cyrus Omar et al.)

- **[ChatLSP (OOPSLA 2024)](https://arxiv.org/abs/2409.00921)** — Statically Contextualizing LLMs with Typed Holes
- **[Marked Lambda Calculus (POPL 2024)](https://dl.acm.org/doi/10.1145/3632910)** — Total Type Error Localization and Recovery with Holes
- **[Incremental Bidirectional Typing (OOPSLA 2025)](https://arxiv.org/abs/2504.08946)** — Order Maintenance for ~275x Speedup
- **[Live Functional Programming (ICFP 2019)](https://arxiv.org/abs/1805.00155)** — Original Typed Holes Paper
- **[Grove (POPL 2025)](https://arxiv.org/abs/2412.18116)** — Collaborative Structure Editing

### Constrained Generation Research

- **[XGrammar](https://arxiv.org/abs/2411.15100)** — Efficient Structured Generation
- **[llguidance](https://github.com/guidance-ai/llguidance)** — Dynamic Mask Computation
- **[JSONSchemaBench](https://arxiv.org/html/2501.10868v1)** — Structured Output Benchmarks

---

## 12. Implementation Checklist for Claude Code

### Phase 1: Core (Week 1)
- [ ] `core/constraint.py`
- [ ] `core/domain.py`
- [ ] `core/unified.py`
- [ ] `core/checkpoint.py`
- [ ] `backend/grammar.py`
- [ ] `backend/backend.py`
- [ ] Modify `constrained/__init__.py`
- [ ] Modify `constrained/base_grammar_backend.py`

### Phase 2: Syntax Domain (Week 2)
- [ ] `domains/syntax/constraint.py`
- [ ] `domains/syntax/domain.py`
- [ ] `domains/syntax/grammars/*.py` (5 files)

### Phase 3: Type Domain (Weeks 3-4)
- [ ] `domains/types/constraint.py`
- [ ] `domains/types/unification.py`
- [ ] `domains/types/environment.py`
- [ ] `domains/types/marking/*.py` (4 files)
- [ ] `domains/types/incremental/*.py` (4 files)
- [ ] `domains/types/bidirectional/*.py` (3 files)
- [ ] `domains/types/domain.py`
- [ ] `domains/types/checker.py`

### Phase 4: Language Type Systems (Weeks 5-6)
- [ ] `domains/types/languages/base.py`
- [ ] `domains/types/languages/python.py`
- [ ] `domains/types/languages/typescript.py`
- [ ] `domains/types/languages/rust.py`
- [ ] `domains/types/languages/zig.py`
- [ ] `domains/types/languages/go.py`

### Phase 5: Context Extraction (Week 6)
- [ ] `core/context_extraction.py`

### Phase 6: Propagation (Week 7)
- [ ] `propagation/network.py`
- [ ] `propagation/edges.py`
- [ ] `propagation/worklist.py`
- [ ] `propagation/builder.py`

### Phase 7: Import Domain (Week 8)
- [ ] `domains/imports/constraint.py`
- [ ] `domains/imports/domain.py`
- [ ] `domains/imports/resolvers/*.py` (6 files)

### Phase 8: Control Flow Domain (Week 9)
- [ ] `domains/controlflow/constraint.py`
- [ ] `domains/controlflow/domain.py`
- [ ] `domains/controlflow/cfg.py`
- [ ] `domains/controlflow/reachability.py`

### Phase 9: Semantic Domain (Week 10)
- [ ] `domains/semantics/constraint.py`
- [ ] `domains/semantics/domain.py`
- [ ] `domains/semantics/smt.py`
- [ ] `domains/semantics/extractors.py`

### Phase 10: Holes (Week 11)
- [ ] `holes/hole.py`
- [ ] `holes/registry.py`
- [ ] `holes/factory.py`
- [ ] `holes/closure.py`
- [ ] `holes/environment_capture.py`
- [ ] `holes/fill_resume.py`
- [ ] `holes/strategy.py`

### Phase 11: Masks (Week 12)
- [ ] `masks/fuser.py`
- [ ] `masks/incremental.py`
- [ ] `masks/cache.py`
- [ ] `masks/lazy.py`

### Phase 12: Parsing (Week 12)
- [ ] `parsing/partial_ast.py`
- [ ] `parsing/base.py`
- [ ] `parsing/languages/*.py` (5 files)

### Phase 13: Testing (Weeks 13-14)
- [ ] `tests/conftest.py`
- [ ] `tests/unit/*.py`
- [ ] `tests/integration/*.py`
- [ ] `tests/property/*.py`
- [ ] `tests/benchmark/*.py`

---

**Total Estimated Files**: ~75 Python files
**Total Estimated Lines**: ~15,000-20,000 LOC
**Estimated Implementation Time**: 14 weeks

This implementation plan provides a complete roadmap for implementing Ananke as a fully-featured multi-domain constraint system within SGLang's constrained decoding infrastructure, drawing on the rigorous foundations of Cyrus Omar's Hazel research program.
