> ---
> **STATUS: HISTORICAL DOCUMENT - IMPLEMENTATION COMPLETE**
> 
> This is the original implementation plan (v1.0) for the Ananke system.
> All features described in this plan have been implemented.
> 
> For current documentation, see:
> - [ARCHITECTURE.md](../ARCHITECTURE.md) - System overview
> - [REFERENCE.md](../REFERENCE.md) - API reference
> - [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guide
> 
> This document is preserved for historical reference.
> 
> ---
> 
# Ananke Implementation Plan
## A Compositional Constraint System for Verified Code Generation

**Version**: 1.0  
**Target Executor**: Claude Code (Opus 4.5)  
**Primary Language**: Python (typed, using uv)  
**Grammar Backend**: llguidance (preferred) with XGrammar compatibility layer

---

## Executive Summary

Ananke is a constraint-driven code generation system that treats AI code generation as constrained search through valid programs rather than probabilistic text completion. This plan provides a phased implementation approach optimized for execution by Claude Code.

**Key Insight**: The system extends SGLang's constrained decoding infrastructure (which currently handles syntax-only constraints via FSM/grammar backends) to support multi-domain constraint fusion across syntax, types, imports, control flow, and semantics—with progressive hole refinement during generation.

---

## Part 1: Foundation Architecture

### 1.1 Project Structure

```
ananke/
├── pyproject.toml              # uv-managed project
├── src/
│   └── ananke/
│       ├── __init__.py
│       ├── py.typed             # PEP 561 marker
│       │
│       ├── core/                # Core constraint algebra
│       │   ├── __init__.py
│       │   ├── constraint.py    # Base Constraint ABC + semilattice
│       │   ├── satisfiability.py
│       │   ├── domain.py        # ConstraintDomain ABC
│       │   └── unified.py       # UnifiedConstraint (product domain)
│       │
│       ├── domains/             # Constraint domain implementations
│       │   ├── __init__.py
│       │   ├── syntax/
│       │   │   ├── __init__.py
│       │   │   ├── constraint.py
│       │   │   ├── domain.py
│       │   │   └── backends/
│       │   │       ├── __init__.py
│       │   │       ├── base.py      # Backend ABC
│       │   │       ├── llguidance.py
│       │   │       └── xgrammar.py
│       │   ├── types/
│       │   │   ├── __init__.py
│       │   │   ├── constraint.py
│       │   │   ├── domain.py
│       │   │   ├── checker.py       # Incremental bidirectional checker
│       │   │   ├── unification.py
│       │   │   └── environment.py
│       │   ├── imports/
│       │   │   ├── __init__.py
│       │   │   ├── constraint.py
│       │   │   └── domain.py
│       │   ├── controlflow/
│       │   │   ├── __init__.py
│       │   │   ├── constraint.py
│       │   │   ├── domain.py
│       │   │   └── cfg.py           # CFG sketch representation
│       │   └── semantics/
│       │       ├── __init__.py
│       │       ├── constraint.py
│       │       ├── domain.py
│       │       └── smt.py           # Z3 integration
│       │
│       ├── propagation/         # Constraint propagation network
│       │   ├── __init__.py
│       │   ├── network.py
│       │   ├── edges.py         # Cross-domain propagation edges
│       │   └── worklist.py
│       │
│       ├── holes/               # Typed hole management
│       │   ├── __init__.py
│       │   ├── hole.py
│       │   ├── registry.py
│       │   ├── factory.py       # Dynamic hole creation
│       │   └── granularity.py
│       │
│       ├── masks/               # Token mask computation
│       │   ├── __init__.py
│       │   ├── fuser.py         # Multi-domain mask fusion
│       │   ├── incremental.py   # Incremental updates
│       │   ├── cache.py
│       │   └── lazy.py          # Lazy evaluation
│       │
│       ├── parsing/             # Incremental parsing infrastructure
│       │   ├── __init__.py
│       │   ├── partial_ast.py
│       │   ├── treesitter.py    # tree-sitter integration
│       │   └── incremental.py
│       │
│       ├── backend/             # SGLang integration
│       │   ├── __init__.py
│       │   ├── ananke_backend.py
│       │   ├── sglang_expr.py   # AnankeConstrainedGen expression
│       │   └── checkpoint.py
│       │
│       └── api/                 # High-level API
│           ├── __init__.py
│           ├── ananke.py        # Main Ananke class
│           ├── holes.py         # Convenience constructors
│           └── result.py
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_constraint.py
│   │   ├── test_domains/
│   │   ├── test_propagation.py
│   │   ├── test_holes.py
│   │   └── test_masks.py
│   └── integration/
│       ├── test_sglang_integration.py
│       └── test_generation.py
│
└── examples/
    ├── simple_type_constraint.py
    ├── multi_hole_generation.py
    └── progressive_refinement.py
```

### 1.2 Dependencies (pyproject.toml)

```toml
[project]
name = "ananke"
version = "0.1.0"
description = "Compositional constraint system for verified code generation"
requires-python = ">=3.11"
dependencies = [
    # Core
    "numpy>=1.26.0",
    
    # Grammar backends
    "llguidance>=1.0.0",  # Primary backend (dynamic, ~50μs/token)
    "xgrammar>=0.1.0",    # Fallback with pre-computation
    
    # Parsing
    "tree-sitter>=0.22.0",
    "tree-sitter-python>=0.21.0",
    "tree-sitter-typescript>=0.21.0",
    
    # Type checking
    "libcst>=1.1.0",      # Python CST manipulation
    
    # SMT solving
    "z3-solver>=4.12.0",
    
    # SGLang integration
    "sglang>=0.4.0",
    
    # Utilities
    "immutables>=0.20",   # Immutable collections
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "mypy>=1.8.0",
    "ruff>=0.2.0",
]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.ruff]
target-version = "py311"
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]
```

---

## Part 2: Core Constraint Algebra Implementation

### Phase 2.1: Constraint Semilattice (Week 1, Days 1-2)

**File**: `src/ananke/core/constraint.py`

```python
"""
Core constraint algebra implementing bounded meet-semilattice.

Mathematical foundation:
- ⟨C, ⊓, ⊤, ⊥⟩ where:
  - C is the set of constraints
  - ⊓ (meet) is constraint conjunction
  - ⊤ (top) is the trivial constraint
  - ⊥ (bottom) is the absurd constraint
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Generic, TypeVar, final

if TYPE_CHECKING:
    import numpy as np
    from ananke.core.domain import GenerationContext


class Satisfiability(Enum):
    """Three-valued satisfiability result."""
    SAT = auto()      # Definitely satisfiable
    UNSAT = auto()    # Definitely unsatisfiable
    UNKNOWN = auto()  # Cannot determine (approximation)


C = TypeVar("C", bound="Constraint")


class Constraint(ABC):
    """
    Base class for all constraints.
    
    Constraints are immutable values in a semilattice.
    Implementations must be hashable and comparable for caching.
    """
    
    __slots__ = ()  # Enforce no instance dict for memory efficiency
    
    @abstractmethod
    def meet(self: C, other: C) -> C:
        """
        Compute the greatest lower bound (conjunction).
        
        Properties that must hold:
        - c ⊓ ⊤ = c (identity)
        - c ⊓ ⊥ = ⊥ (annihilation)
        - c ⊓ c = c (idempotence)
        - c₁ ⊓ c₂ = c₂ ⊓ c₁ (commutativity)
        - (c₁ ⊓ c₂) ⊓ c₃ = c₁ ⊓ (c₂ ⊓ c₃) (associativity)
        """
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
        return self.meet(other) == self  # type: ignore[arg-type]
    
    def __and__(self: C, other: C) -> C:
        """Syntactic sugar: c1 & c2 = c1.meet(c2)."""
        return self.meet(other)
```

**Implementation Notes for Claude Code**:
1. Use `@dataclass(frozen=True, slots=True)` for all concrete constraint classes
2. Implement `__hash__` via `hash(tuple(self.__slots__))` pattern
3. Test semilattice laws exhaustively in unit tests

### Phase 2.2: Constraint Domain ABC (Week 1, Days 2-3)

**File**: `src/ananke/core/domain.py`

```python
"""
Constraint domain abstraction.

Each domain D is modeled as a functor from contexts to constraint semilattices:
D : Context → ConstraintSemilattice
"""
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
    """
    Context for token mask computation.
    
    Captures all state needed to compute valid next tokens.
    """
    tokenizer: Tokenizer
    vocab_size: int
    position: int
    partial_ast: PartialAST
    generated_tokens: tuple[int, ...] = field(default_factory=tuple)
    
    def advance(self, token: int) -> GenerationContext:
        """Create new context after generating a token."""
        return GenerationContext(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            position=self.position + 1,
            partial_ast=self.partial_ast,  # Updated separately
            generated_tokens=(*self.generated_tokens, token),
        )
    
    def recent_tokens(self, n: int) -> tuple[int, ...]:
        """Get the n most recent tokens."""
        return self.generated_tokens[-n:]
    
    def with_ast(self, ast: PartialAST) -> GenerationContext:
        """Create new context with updated AST."""
        return GenerationContext(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            position=self.position,
            partial_ast=ast,
            generated_tokens=self.generated_tokens,
        )


@dataclass
class DomainCheckpoint:
    """Opaque checkpoint for domain state."""
    domain_name: str
    state: object


class ConstraintDomain(ABC, Generic[C]):
    """
    A constraint domain with its own semilattice structure.
    
    Domains are responsible for:
    1. Representing constraints in their formalism
    2. Computing token masks from constraints
    3. Updating constraints given new tokens
    4. Projecting to/from other domains
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
        
        Returns a boolean array of shape (vocab_size,) where True means allowed.
        
        Performance target: <1ms for most cases, <50ms worst case.
        """
        ...
    
    @abstractmethod
    def observe_token(
        self, constraint: C, token: int, context: GenerationContext
    ) -> C:
        """
        Update constraint after observing a generated token.
        
        Returns the refined constraint incorporating the new information.
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
        """
        Check if a single token is valid.
        
        Default implementation uses token_mask, but domains can override
        for efficiency when checking individual tokens.
        """
        return self.token_mask(constraint, context)[token]
```

### Phase 2.3: Unified Constraint (Product Domain) (Week 1, Days 3-4)

**File**: `src/ananke/core/unified.py`

```python
"""
Unified constraint as product of all domains.

Ω(Γ) = Syntax(Γ) × Types(Γ) × Imports(Γ) × ControlFlow(Γ) × Semantics(Γ)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ananke.core.constraint import Constraint, Satisfiability

if TYPE_CHECKING:
    from ananke.domains.controlflow.constraint import ControlFlowConstraint
    from ananke.domains.imports.constraint import ImportConstraint
    from ananke.domains.semantics.constraint import SemanticConstraint
    from ananke.domains.syntax.constraint import SyntaxConstraint
    from ananke.domains.types.constraint import TypeConstraint


@dataclass(frozen=True, slots=True)
class UnifiedConstraint(Constraint):
    """
    Product of all constraint domains.
    
    This is the constraint type that the generation loop operates on.
    Component-wise meet preserves the semilattice structure.
    """
    syntax: SyntaxConstraint
    types: TypeConstraint
    imports: ImportConstraint
    control_flow: ControlFlowConstraint
    semantics: SemanticConstraint
    
    def meet(self, other: UnifiedConstraint) -> UnifiedConstraint:
        """Component-wise meet."""
        return UnifiedConstraint(
            syntax=self.syntax.meet(other.syntax),
            types=self.types.meet(other.types),
            imports=self.imports.meet(other.imports),
            control_flow=self.control_flow.meet(other.control_flow),
            semantics=self.semantics.meet(other.semantics),
        )
    
    def satisfiability(self) -> Satisfiability:
        """All domains must be satisfiable."""
        results = [
            self.syntax.satisfiability(),
            self.types.satisfiability(),
            self.imports.satisfiability(),
            self.control_flow.satisfiability(),
            self.semantics.satisfiability(),
        ]
        
        if Satisfiability.UNSAT in results:
            return Satisfiability.UNSAT
        if all(r == Satisfiability.SAT for r in results):
            return Satisfiability.SAT
        return Satisfiability.UNKNOWN
    
    def is_top(self) -> bool:
        return all([
            self.syntax.is_top(),
            self.types.is_top(),
            self.imports.is_top(),
            self.control_flow.is_top(),
            self.semantics.is_top(),
        ])
    
    def is_bottom(self) -> bool:
        return any([
            self.syntax.is_bottom(),
            self.types.is_bottom(),
            self.imports.is_bottom(),
            self.control_flow.is_bottom(),
            self.semantics.is_bottom(),
        ])
    
    def __hash__(self) -> int:
        return hash((
            self.syntax, self.types, self.imports,
            self.control_flow, self.semantics
        ))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnifiedConstraint):
            return NotImplemented
        return (
            self.syntax == other.syntax
            and self.types == other.types
            and self.imports == other.imports
            and self.control_flow == other.control_flow
            and self.semantics == other.semantics
        )
```

---

## Part 3: Syntax Domain with llguidance Integration

### Phase 3.1: Understanding llguidance Architecture

**Key llguidance Features for Ananke**:

1. **Dynamic Mask Computation**: Unlike XGrammar's pre-computation approach, llguidance computes token masks on-the-fly (~50μs per token for 128k vocabulary). This is critical for Ananke because our type/semantic constraints can dynamically modify which syntactic forms are valid.

2. **Lazy Automata Construction**: llguidance builds lexer automata lazily, making it suitable for grammars that evolve during generation (e.g., when hole refinement restricts valid syntax).

3. **Context-Free Grammar Support**: llguidance supports full CFGs via Earley parsing on top of derivative-based lexing. This is more powerful than FSM-only approaches.

4. **Lark-like Grammar Format**: Use the `%llguidance {}` prefix format for grammar specifications.

### Phase 3.2: Syntax Domain Implementation (Week 1, Days 4-7)

**File**: `src/ananke/domains/syntax/backends/llguidance.py`

```python
"""
llguidance backend for syntax constraint domain.

llguidance provides:
- Dynamic token mask computation (~50μs/token)
- Lazy automata construction (no significant startup cost)
- CFG support via Earley parsing
- Jump-ahead for deterministic token sequences
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tokenizers import Tokenizer

# llguidance imports
import llguidance
from llguidance import ParserFactory, TokenParser


@dataclass
class LLGuidanceState:
    """Encapsulates llguidance parser state."""
    parser: TokenParser
    grammar_str: str
    _checkpoint_stack: list[bytes] = field(default_factory=list)
    
    def checkpoint(self) -> int:
        """Save parser state, return checkpoint ID."""
        state_bytes = self.parser.save_state()
        self._checkpoint_stack.append(state_bytes)
        return len(self._checkpoint_stack) - 1
    
    def restore(self, checkpoint_id: int) -> None:
        """Restore to checkpoint."""
        if checkpoint_id < len(self._checkpoint_stack):
            state_bytes = self._checkpoint_stack[checkpoint_id]
            self.parser.restore_state(state_bytes)
            self._checkpoint_stack = self._checkpoint_stack[:checkpoint_id + 1]


class LLGuidanceBackend:
    """
    llguidance backend adapter for Ananke syntax domain.
    
    Key design decisions:
    1. Use Lark-like grammar format with %llguidance prefix
    2. Support dynamic grammar updates via grammar_stack
    3. Expose jump-ahead tokens for RadixAttention integration
    """
    
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self._factory = ParserFactory(
            tokenizer=tokenizer,
            # Enable jump-ahead optimization
            enable_ff_tokens=True,
        )
        self._grammar_cache: dict[str, TokenParser] = {}
    
    def create_parser(self, grammar: str) -> LLGuidanceState:
        """
        Create a parser for the given grammar.
        
        Grammar should be in Lark-like format:
        ```
        %llguidance {}
        start: statement+
        statement: assignment | expression
        ...
        ```
        """
        if grammar not in self._grammar_cache:
            parser = self._factory.create_parser(grammar)
            self._grammar_cache[grammar] = parser
        else:
            parser = self._grammar_cache[grammar].clone()
        
        return LLGuidanceState(parser=parser, grammar_str=grammar)
    
    def compute_mask(
        self,
        state: LLGuidanceState,
        vocab_size: int,
    ) -> np.ndarray:
        """
        Compute token mask from current parser state.
        
        Returns boolean array where True = token allowed.
        """
        # llguidance returns a set of allowed token IDs
        allowed_tokens = state.parser.get_allowed_tokens()
        
        mask = np.zeros(vocab_size, dtype=bool)
        for token_id in allowed_tokens:
            if token_id < vocab_size:
                mask[token_id] = True
        
        return mask
    
    def advance(
        self,
        state: LLGuidanceState,
        token: int,
    ) -> LLGuidanceState:
        """
        Advance parser state with a new token.
        
        Returns updated state (llguidance mutates in place, so we
        return the same state object but semantically it's "new").
        """
        state.parser.consume_token(token)
        return state
    
    def get_jump_ahead_tokens(self, state: LLGuidanceState) -> list[int]:
        """
        Get tokens that can be deterministically inserted.
        
        These are tokens where only one valid choice exists.
        SGLang's RadixAttention can skip forward passes for these.
        """
        return state.parser.get_ff_tokens()
    
    def is_accepting(self, state: LLGuidanceState) -> bool:
        """Check if current state is in an accepting configuration."""
        return state.parser.is_accepting()
    
    def is_dead(self, state: LLGuidanceState) -> bool:
        """Check if parser has reached a dead state (no valid continuation)."""
        return state.parser.is_dead()
    
    def update_grammar(
        self,
        state: LLGuidanceState,
        grammar_delta: str,
    ) -> LLGuidanceState:
        """
        Dynamically update grammar constraints.
        
        This is the key extension point for Ananke's multi-domain constraints.
        When type checking reveals that certain syntactic forms are invalid,
        we can restrict the grammar dynamically.
        
        Implementation: Create new parser with combined grammar and
        transfer state via prefix replay.
        """
        # Combine grammars (grammar_delta restricts original)
        combined = self._combine_grammars(state.grammar_str, grammar_delta)
        
        # Create new parser
        new_parser = self._factory.create_parser(combined)
        
        # Replay tokens to reconstruct state
        # (llguidance doesn't directly support grammar hot-swap)
        for token in state.parser.get_consumed_tokens():
            new_parser.consume_token(token)
        
        return LLGuidanceState(parser=new_parser, grammar_str=combined)
    
    def _combine_grammars(self, base: str, restriction: str) -> str:
        """
        Combine base grammar with restriction grammar.
        
        The restriction grammar specifies additional constraints.
        Result is the intersection of languages.
        """
        # Simple approach: embed restriction as additional rules
        # More sophisticated: compute grammar intersection
        return f"{base}\n\n// Restrictions\n{restriction}"
```

**File**: `src/ananke/domains/syntax/constraint.py`

```python
"""
Syntax constraint representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ananke.core.constraint import Constraint, Satisfiability

if TYPE_CHECKING:
    from ananke.domains.syntax.backends.llguidance import LLGuidanceState


@dataclass(frozen=True, slots=True)
class SyntaxConstraint(Constraint):
    """
    Constraint on syntactic structure.
    
    Internally represented as a grammar string + parser state hash.
    The actual parser state is managed by the SyntaxDomain.
    """
    grammar: str
    state_hash: int  # Hash of parser state for equality/caching
    _is_dead: bool = False
    _is_accepting: bool = False
    
    def meet(self, other: SyntaxConstraint) -> SyntaxConstraint:
        """
        Grammar intersection.
        
        For CFGs this is computable but potentially expensive.
        We use an approximation: track both grammars and let the
        domain handle the actual intersection.
        """
        if self._is_dead or other._is_dead:
            return SYNTAX_BOTTOM
        
        if self.grammar == other.grammar:
            # Same grammar, take more constrained state
            return self if self.state_hash <= other.state_hash else other
        
        # Different grammars: combine them
        combined_grammar = f"{self.grammar}\n---\n{other.grammar}"
        return SyntaxConstraint(
            grammar=combined_grammar,
            state_hash=hash((self.state_hash, other.state_hash)),
            _is_dead=False,
            _is_accepting=self._is_accepting and other._is_accepting,
        )
    
    def satisfiability(self) -> Satisfiability:
        if self._is_dead:
            return Satisfiability.UNSAT
        if self._is_accepting:
            return Satisfiability.SAT
        return Satisfiability.UNKNOWN
    
    def is_top(self) -> bool:
        return self.grammar == "" and not self._is_dead
    
    def is_bottom(self) -> bool:
        return self._is_dead
    
    def __hash__(self) -> int:
        return hash((self.grammar, self.state_hash))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SyntaxConstraint):
            return NotImplemented
        return self.grammar == other.grammar and self.state_hash == other.state_hash


# Sentinel values
SYNTAX_TOP = SyntaxConstraint(grammar="", state_hash=0)
SYNTAX_BOTTOM = SyntaxConstraint(grammar="", state_hash=-1, _is_dead=True)
```

---

## Part 4: Type Domain with Incremental Bidirectional Checking

### Phase 4.1: Type System Architecture (Week 2, Days 1-3)

**Key Design Decisions**:

1. **Bidirectional Type Checking**: Use synthesis (infer type from expression) and checking (verify expression has type) modes. This enables:
   - Pushing type expectations down through the AST
   - Generating fewer type variables
   - Better error messages

2. **Incremental Updates**: Track which parts of the AST have been typed. When new tokens arrive, only re-check affected subtrees.

3. **Type Holes**: Represent unknown types as unification variables. Constraints on holes become unification equations.

**File**: `src/ananke/domains/types/constraint.py`

```python
"""
Type constraint representation with unification support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

from immutables import Map

from ananke.core.constraint import Constraint, Satisfiability

if TYPE_CHECKING:
    from ananke.domains.types.environment import TypeEnvironment


# Type representation
@dataclass(frozen=True, slots=True)
class TypeVar:
    """A type variable (unification variable)."""
    name: str
    id: int  # Unique ID for this variable


@dataclass(frozen=True, slots=True)
class ConcreteType:
    """Base for concrete types."""
    pass


@dataclass(frozen=True, slots=True)
class PrimitiveType(ConcreteType):
    """Primitive types: int, str, bool, float, None."""
    name: str


@dataclass(frozen=True, slots=True)
class FunctionType(ConcreteType):
    """Function type: (T1, T2, ...) -> R."""
    params: tuple[tuple[str, Type], ...]  # (name, type) pairs
    returns: Type


@dataclass(frozen=True, slots=True)
class ListType(ConcreteType):
    """List type: List[T]."""
    element: Type


@dataclass(frozen=True, slots=True)
class DictType(ConcreteType):
    """Dict type: Dict[K, V]."""
    key: Type
    value: Type


@dataclass(frozen=True, slots=True)
class UnionType(ConcreteType):
    """Union type: T1 | T2 | ..."""
    alternatives: frozenset[Type]


@dataclass(frozen=True, slots=True)
class ClassType(ConcreteType):
    """Class/nominal type."""
    name: str
    module: str
    type_params: tuple[Type, ...] = field(default_factory=tuple)


# Any type (top of type lattice)
@dataclass(frozen=True, slots=True)
class AnyType(ConcreteType):
    """The Any type (unconstrained)."""
    pass


# Never type (bottom of type lattice)
@dataclass(frozen=True, slots=True)
class NeverType(ConcreteType):
    """The Never type (no values)."""
    pass


Type = Union[TypeVar, ConcreteType]


@dataclass(frozen=True, slots=True)
class TypeEquation:
    """An equation T1 = T2 that must be solved by unification."""
    lhs: Type
    rhs: Type


@dataclass(frozen=True, slots=True)
class TypeConstraint(Constraint):
    """
    Constraint on types.
    
    Represented as:
    - expected: The type this expression should have
    - environment: Types currently in scope
    - unification: Equations that must hold
    """
    expected: Type
    environment: TypeEnvironment
    unification: frozenset[TypeEquation]
    
    def meet(self, other: TypeConstraint) -> TypeConstraint:
        """Type constraint conjunction via unification."""
        from ananke.domains.types.unification import unify, solve_unification
        
        # Merge environments (later bindings shadow earlier)
        merged_env = self.environment.merge(other.environment)
        
        # Unify expected types
        result = unify(self.expected, other.expected)
        if result is None:
            return TYPE_BOTTOM
        
        unified_expected, new_equations = result
        
        # Combine all unification constraints
        all_equations = self.unification | other.unification | new_equations
        
        # Check satisfiability
        solution = solve_unification(all_equations)
        if solution is None:
            return TYPE_BOTTOM
        
        return TypeConstraint(
            expected=unified_expected,
            environment=merged_env,
            unification=all_equations,
        )
    
    def satisfiability(self) -> Satisfiability:
        from ananke.domains.types.unification import solve_unification
        
        solution = solve_unification(self.unification)
        if solution is None:
            return Satisfiability.UNSAT
        return Satisfiability.SAT
    
    def is_top(self) -> bool:
        return (
            isinstance(self.expected, AnyType)
            and len(self.unification) == 0
        )
    
    def is_bottom(self) -> bool:
        from ananke.domains.types.unification import solve_unification
        return solve_unification(self.unification) is None
    
    def __hash__(self) -> int:
        return hash((self.expected, self.environment, self.unification))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypeConstraint):
            return NotImplemented
        return (
            self.expected == other.expected
            and self.environment == other.environment
            and self.unification == other.unification
        )


# Sentinel values
TYPE_TOP = TypeConstraint(
    expected=AnyType(),
    environment=Map(),  # Empty environment
    unification=frozenset(),
)
TYPE_BOTTOM = TypeConstraint(
    expected=NeverType(),
    environment=Map(),
    unification=frozenset({TypeEquation(AnyType(), NeverType())}),  # Unsolvable
)
```

### Phase 4.2: Unification Engine (Week 2, Days 3-4)

**File**: `src/ananke/domains/types/unification.py`

```python
"""
Type unification engine with occurs check.

Implements Robinson's unification algorithm with extensions for:
- Subtyping (covariance/contravariance)
- Union types
- Generic type parameters
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ananke.domains.types.constraint import (
    AnyType,
    ClassType,
    ConcreteType,
    DictType,
    FunctionType,
    ListType,
    NeverType,
    PrimitiveType,
    Type,
    TypeEquation,
    TypeVar,
    UnionType,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass
class Substitution:
    """A mapping from type variables to types."""
    mapping: dict[TypeVar, Type]
    
    def apply(self, typ: Type) -> Type:
        """Apply substitution to a type."""
        if isinstance(typ, TypeVar):
            if typ in self.mapping:
                # Recursively apply in case of chains
                return self.apply(self.mapping[typ])
            return typ
        
        if isinstance(typ, FunctionType):
            return FunctionType(
                params=tuple(
                    (name, self.apply(t)) for name, t in typ.params
                ),
                returns=self.apply(typ.returns),
            )
        
        if isinstance(typ, ListType):
            return ListType(element=self.apply(typ.element))
        
        if isinstance(typ, DictType):
            return DictType(
                key=self.apply(typ.key),
                value=self.apply(typ.value),
            )
        
        if isinstance(typ, UnionType):
            return UnionType(
                alternatives=frozenset(self.apply(t) for t in typ.alternatives)
            )
        
        if isinstance(typ, ClassType):
            return ClassType(
                name=typ.name,
                module=typ.module,
                type_params=tuple(self.apply(t) for t in typ.type_params),
            )
        
        return typ
    
    def compose(self, other: Substitution) -> Substitution:
        """Compose two substitutions: (self ∘ other)."""
        # Apply self to all of other's mappings
        new_mapping = {v: self.apply(t) for v, t in other.mapping.items()}
        # Add self's mappings for variables not in other
        for v, t in self.mapping.items():
            if v not in new_mapping:
                new_mapping[v] = t
        return Substitution(mapping=new_mapping)


def occurs_check(var: TypeVar, typ: Type) -> bool:
    """Check if var occurs in typ (prevents infinite types)."""
    if isinstance(typ, TypeVar):
        return typ == var
    
    if isinstance(typ, FunctionType):
        return any(occurs_check(var, t) for _, t in typ.params) or \
               occurs_check(var, typ.returns)
    
    if isinstance(typ, ListType):
        return occurs_check(var, typ.element)
    
    if isinstance(typ, DictType):
        return occurs_check(var, typ.key) or occurs_check(var, typ.value)
    
    if isinstance(typ, UnionType):
        return any(occurs_check(var, t) for t in typ.alternatives)
    
    if isinstance(typ, ClassType):
        return any(occurs_check(var, t) for t in typ.type_params)
    
    return False


def unify(t1: Type, t2: Type) -> tuple[Type, frozenset[TypeEquation]] | None:
    """
    Unify two types.
    
    Returns (unified_type, new_equations) or None if unification fails.
    """
    # Trivial cases
    if t1 == t2:
        return (t1, frozenset())
    
    # Any unifies with anything
    if isinstance(t1, AnyType):
        return (t2, frozenset())
    if isinstance(t2, AnyType):
        return (t1, frozenset())
    
    # Never is bottom - unifies only with itself
    if isinstance(t1, NeverType) or isinstance(t2, NeverType):
        return None
    
    # Type variable cases
    if isinstance(t1, TypeVar):
        if occurs_check(t1, t2):
            return None  # Infinite type
        return (t2, frozenset({TypeEquation(t1, t2)}))
    
    if isinstance(t2, TypeVar):
        if occurs_check(t2, t1):
            return None
        return (t1, frozenset({TypeEquation(t2, t1)}))
    
    # Structural unification
    if isinstance(t1, PrimitiveType) and isinstance(t2, PrimitiveType):
        if t1.name == t2.name:
            return (t1, frozenset())
        return None
    
    if isinstance(t1, FunctionType) and isinstance(t2, FunctionType):
        if len(t1.params) != len(t2.params):
            return None
        
        equations: set[TypeEquation] = set()
        
        # Contravariant in parameters
        for (_, p1), (_, p2) in zip(t1.params, t2.params):
            result = unify(p2, p1)  # Note: reversed for contravariance
            if result is None:
                return None
            equations.update(result[1])
        
        # Covariant in return
        result = unify(t1.returns, t2.returns)
        if result is None:
            return None
        equations.update(result[1])
        
        return (t1, frozenset(equations))
    
    if isinstance(t1, ListType) and isinstance(t2, ListType):
        result = unify(t1.element, t2.element)
        if result is None:
            return None
        return (ListType(element=result[0]), result[1])
    
    if isinstance(t1, DictType) and isinstance(t2, DictType):
        key_result = unify(t1.key, t2.key)
        if key_result is None:
            return None
        value_result = unify(t1.value, t2.value)
        if value_result is None:
            return None
        return (
            DictType(key=key_result[0], value=value_result[0]),
            key_result[1] | value_result[1],
        )
    
    # Type mismatch
    return None


def solve_unification(
    equations: frozenset[TypeEquation],
) -> Substitution | None:
    """
    Solve a set of type equations.
    
    Returns a substitution that makes all equations hold, or None if unsolvable.
    """
    substitution = Substitution(mapping={})
    worklist = list(equations)
    
    while worklist:
        eq = worklist.pop()
        lhs = substitution.apply(eq.lhs)
        rhs = substitution.apply(eq.rhs)
        
        if lhs == rhs:
            continue
        
        result = unify(lhs, rhs)
        if result is None:
            return None
        
        _, new_equations = result
        
        # Add new equations to worklist
        for new_eq in new_equations:
            if isinstance(new_eq.lhs, TypeVar):
                # Add to substitution
                substitution.mapping[new_eq.lhs] = new_eq.rhs
            elif isinstance(new_eq.rhs, TypeVar):
                substitution.mapping[new_eq.rhs] = new_eq.lhs
            else:
                worklist.append(new_eq)
    
    return substitution
```

### Phase 4.3: Type Domain with Incremental Checking (Week 2, Days 4-5)

**File**: `src/ananke/domains/types/domain.py`

```python
"""
Type constraint domain with incremental bidirectional type checking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ananke.core.domain import ConstraintDomain, DomainCheckpoint, GenerationContext
from ananke.domains.types.checker import IncrementalTypeChecker, TypeCheckResult
from ananke.domains.types.constraint import (
    TYPE_BOTTOM,
    TYPE_TOP,
    TypeConstraint,
    TypeEquation,
)
from ananke.domains.types.unification import unify

if TYPE_CHECKING:
    pass


class TypeDomain(ConstraintDomain[TypeConstraint]):
    """
    Type constraint domain with incremental type checking.
    
    Strategy for token_mask:
    1. Fast path: If syntax already rejects token, skip type check
    2. For remaining tokens, check if they could lead to well-typed completion
    3. Use budgeted enumeration for complex cases
    """
    
    def __init__(self, budget: int = 100):
        self._type_checker = IncrementalTypeChecker()
        self._budget = budget  # Max tokens to check individually
    
    @property
    def name(self) -> str:
        return "types"
    
    def top(self) -> TypeConstraint:
        return TYPE_TOP
    
    def bottom(self) -> TypeConstraint:
        return TYPE_BOTTOM
    
    def token_mask(
        self,
        constraint: TypeConstraint,
        context: GenerationContext,
    ) -> np.ndarray:
        """
        Compute token mask based on type constraints.
        
        This is the most complex part of type-constrained generation.
        We use a multi-phase approach:
        
        Phase 1: All tokens allowed (if constraint is top)
        Phase 2: Check tokens individually up to budget
        Phase 3: Fall back to allowing all syntactically valid tokens
        """
        if constraint.is_top():
            return np.ones(context.vocab_size, dtype=bool)
        
        if constraint.is_bottom():
            return np.zeros(context.vocab_size, dtype=bool)
        
        mask = np.ones(context.vocab_size, dtype=bool)
        partial_ast = context.partial_ast
        
        # Get syntactically valid tokens first (from syntax domain)
        # This is an optimization - we only type-check valid tokens
        syntax_mask = self._get_syntax_mask(context)
        
        valid_tokens = np.where(syntax_mask)[0]
        
        if len(valid_tokens) > self._budget:
            # Too many tokens to check individually
            # Fall back to allowing all syntactically valid
            return syntax_mask
        
        # Check each token
        for token_id in valid_tokens:
            token_str = context.tokenizer.decode([token_id])
            
            # Try extending AST with this token
            extended_ast = partial_ast.extend_tentative(token_str)
            if extended_ast is None:
                mask[token_id] = False
                continue
            
            # Check if extension could be well-typed
            result = self._type_checker.check_partial(
                extended_ast,
                constraint.expected,
                constraint.environment,
            )
            
            if result == TypeCheckResult.DEFINITELY_ILL_TYPED:
                mask[token_id] = False
        
        return mask
    
    def observe_token(
        self,
        constraint: TypeConstraint,
        token: int,
        context: GenerationContext,
    ) -> TypeConstraint:
        """
        Update type constraint after observing a token.
        
        This may:
        1. Narrow the expected type based on what was generated
        2. Add new bindings to the environment
        3. Introduce new unification constraints
        """
        token_str = context.tokenizer.decode([token])
        new_ast = context.partial_ast.extend(token_str)
        
        if new_ast is None:
            return constraint
        
        # Extract new type information
        info = self._type_checker.extract_type_info(new_ast)
        
        # Update environment with new bindings
        new_env = constraint.environment
        for name, typ in info.new_bindings.items():
            new_env = new_env.set(name, typ)
        
        # Add new unification constraints
        new_unification = set(constraint.unification)
        new_unification.update(info.new_equations)
        
        # Narrow expected type if we learned something
        new_expected = constraint.expected
        if info.inferred_type is not None:
            result = unify(constraint.expected, info.inferred_type)
            if result is not None:
                new_expected, equations = result
                new_unification.update(equations)
        
        return TypeConstraint(
            expected=new_expected,
            environment=new_env,
            unification=frozenset(new_unification),
        )
    
    def checkpoint(self) -> DomainCheckpoint:
        return DomainCheckpoint(
            domain_name=self.name,
            state=self._type_checker.checkpoint(),
        )
    
    def restore(self, checkpoint: DomainCheckpoint) -> None:
        self._type_checker.restore(checkpoint.state)
    
    def _get_syntax_mask(self, context: GenerationContext) -> np.ndarray:
        """Get syntax mask from context (assumes syntax domain runs first)."""
        # This would be set by the mask fuser
        if hasattr(context, "syntax_mask"):
            return context.syntax_mask
        return np.ones(context.vocab_size, dtype=bool)
```

---

## Part 5: Constraint Propagation Network

### Phase 5.1: Propagation Network (Week 2, Days 5-7)

**File**: `src/ananke/propagation/network.py`

```python
"""
Constraint propagation network using worklist algorithm.

Propagation ensures that constraints flow between domains:
- Syntax → Types: Syntactic structure constrains types
- Types → Syntax: Type expectations restrict valid syntax
- Types → Imports: Using a type requires its import
- Imports → Types: Available imports determine available types
- Syntax → ControlFlow: Syntax determines CFG shape
- ControlFlow → Semantics: Control flow induces semantic constraints
- Semantics → Types: Semantic constraints become refinement types
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from ananke.core.constraint import Constraint
from ananke.core.domain import ConstraintDomain

if TYPE_CHECKING:
    from ananke.core.unified import UnifiedConstraint


@dataclass
class PropagationEdge:
    """
    An edge in the propagation graph.
    
    Defines how constraints in the source domain induce constraints
    in the target domain.
    """
    source: str  # Domain name
    target: str  # Domain name
    propagate: Callable[[Constraint, PropagationContext], Constraint]
    priority: int = 0  # Lower = higher priority
    
    def __lt__(self, other: PropagationEdge) -> bool:
        return self.priority < other.priority


@dataclass
class PropagationContext:
    """Context available during propagation."""
    constraints: dict[str, Constraint]
    domains: dict[str, ConstraintDomain]
    partial_ast: object | None = None
    type_environment: object | None = None


@dataclass
class NetworkCheckpoint:
    """Checkpoint for network state."""
    constraints: dict[str, Constraint]
    domain_checkpoints: dict[str, object]


class PropagationNetwork:
    """
    The constraint propagation network.
    
    Manages cross-domain constraint flow using a priority worklist.
    """
    
    def __init__(self):
        self.domains: dict[str, ConstraintDomain] = {}
        self.edges: list[PropagationEdge] = []
        self.constraints: dict[str, Constraint] = {}
        self._worklist: list[tuple[int, str]] = []
    
    def register_domain(self, domain: ConstraintDomain) -> None:
        """Register a constraint domain."""
        self.domains[domain.name] = domain
        self.constraints[domain.name] = domain.top()
    
    def register_edge(self, edge: PropagationEdge) -> None:
        """Register a propagation edge."""
        self.edges.append(edge)
        self.edges.sort(key=lambda e: e.priority)
    
    def add_constraint(self, domain: str, constraint: Constraint) -> bool:
        """
        Add a constraint to a domain and propagate.
        
        Returns False if adding caused unsatisfiability.
        """
        current = self.constraints[domain]
        new = current.meet(constraint)
        
        if new.is_bottom():
            return False
        
        if new != current:
            self.constraints[domain] = new
            self._enqueue_dependents(domain)
        
        return self._propagate()
    
    def _enqueue_dependents(self, source: str) -> None:
        """Add all domains that depend on source to the worklist."""
        for edge in self.edges:
            if edge.source == source:
                heapq.heappush(self._worklist, (edge.priority, edge.target))
    
    def _propagate(self) -> bool:
        """
        Run propagation until fixpoint or contradiction.
        
        Uses a priority worklist to process high-priority edges first.
        """
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
            
            # Meet with current constraint
            current = self.constraints[target]
            new = current.meet(induced)
            
            if new.is_bottom():
                return False
            
            if new != current:
                self.constraints[target] = new
                self._enqueue_dependents(target)
        
        return True
    
    def _make_context(self) -> PropagationContext:
        return PropagationContext(
            constraints=dict(self.constraints),
            domains=self.domains,
        )
    
    def checkpoint(self) -> NetworkCheckpoint:
        """Save network state for backtracking."""
        return NetworkCheckpoint(
            constraints=dict(self.constraints),
            domain_checkpoints={
                name: domain.checkpoint()
                for name, domain in self.domains.items()
            },
        )
    
    def restore(self, checkpoint: NetworkCheckpoint) -> None:
        """Restore to a previous state."""
        self.constraints = dict(checkpoint.constraints)
        for name, cp in checkpoint.domain_checkpoints.items():
            self.domains[name].restore(cp)
        self._worklist.clear()
```

### Phase 5.2: Standard Propagation Edges (Week 3, Days 1-2)

**File**: `src/ananke/propagation/edges.py`

```python
"""
Standard propagation edges between constraint domains.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ananke.domains.imports.constraint import IMPORT_TOP, ImportConstraint, ModuleSpec
from ananke.domains.syntax.constraint import SYNTAX_TOP, SyntaxConstraint
from ananke.domains.types.constraint import (
    TYPE_TOP,
    AnyType,
    ClassType,
    TypeConstraint,
    TypeEquation,
)
from ananke.propagation.network import PropagationContext, PropagationEdge

if TYPE_CHECKING:
    pass


def syntax_to_types(
    syntax: SyntaxConstraint,
    ctx: PropagationContext,
) -> TypeConstraint:
    """
    Propagate syntactic structure to type constraints.
    
    Example: if syntax forces "x + _", the hole must have a type
    that supports addition with x's type.
    """
    if syntax.is_top():
        return TYPE_TOP
    
    partial_ast = ctx.partial_ast
    if partial_ast is None:
        return TYPE_TOP
    
    # Find type expectations induced by syntactic position
    hole_contexts = partial_ast.find_hole_contexts()
    
    equations: set[TypeEquation] = set()
    for hole_ctx in hole_contexts:
        if hole_ctx.expected_type is not None:
            equations.add(TypeEquation(
                hole_ctx.hole_type_var,
                hole_ctx.expected_type,
            ))
    
    return TypeConstraint(
        expected=AnyType(),
        environment=ctx.type_environment or {},
        unification=frozenset(equations),
    )


def types_to_syntax(
    types: TypeConstraint,
    ctx: PropagationContext,
) -> SyntaxConstraint:
    """
    Propagate type constraints to syntactic constraints.
    
    Example: if the expected type is int, rule out string literals.
    """
    if types.is_top():
        return SYNTAX_TOP
    
    # Get forms that could produce the expected type
    valid_forms = _enumerate_forms_for_type(types.expected, types.environment)
    
    if not valid_forms:
        # No valid syntactic forms - but don't bottom out yet
        # The type might be satisfiable by other means
        return SYNTAX_TOP
    
    # Create grammar restriction
    grammar_restriction = _forms_to_grammar(valid_forms)
    
    return SyntaxConstraint(
        grammar=grammar_restriction,
        state_hash=hash(grammar_restriction),
    )


def types_to_imports(
    types: TypeConstraint,
    ctx: PropagationContext,
) -> ImportConstraint:
    """
    Propagate type constraints to import constraints.
    
    Using a type may require importing its definition.
    """
    if types.is_top():
        return IMPORT_TOP
    
    required_modules: set[ModuleSpec] = set()
    
    # Extract required modules from types
    _collect_required_imports(types.expected, required_modules)
    
    if not required_modules:
        return IMPORT_TOP
    
    return ImportConstraint(
        required=frozenset(required_modules),
        forbidden=frozenset(),
        versions={},
    )


def imports_to_types(
    imports: ImportConstraint,
    ctx: PropagationContext,
) -> TypeConstraint:
    """
    Propagate import constraints to type constraints.
    
    Available imports determine what types are in scope.
    """
    if imports.is_top():
        return TYPE_TOP
    
    # Build type environment from available imports
    new_env = {}
    for module in imports.required:
        module_types = _get_types_from_module(module)
        new_env.update(module_types)
    
    return TypeConstraint(
        expected=AnyType(),
        environment=new_env,
        unification=frozenset(),
    )


def _enumerate_forms_for_type(typ, env) -> list[str]:
    """Enumerate syntactic forms that could produce the given type."""
    # This is language-specific
    # For Python, we'd enumerate:
    # - Literals (for primitive types)
    # - Variable references (from environment)
    # - Function calls (for return types)
    # - etc.
    return []  # Placeholder


def _forms_to_grammar(forms: list[str]) -> str:
    """Convert syntactic forms to a grammar restriction."""
    return ""  # Placeholder


def _collect_required_imports(typ, modules: set[ModuleSpec]) -> None:
    """Collect modules required to use a type."""
    if isinstance(typ, ClassType):
        modules.add(ModuleSpec(name=typ.module))


def _get_types_from_module(module: ModuleSpec) -> dict:
    """Get types exported by a module."""
    return {}  # Placeholder - would use static analysis


def build_standard_edges() -> list[PropagationEdge]:
    """Build the standard set of propagation edges."""
    return [
        PropagationEdge(
            source="syntax",
            target="types",
            propagate=syntax_to_types,
            priority=0,
        ),
        PropagationEdge(
            source="types",
            target="syntax",
            propagate=types_to_syntax,
            priority=1,
        ),
        PropagationEdge(
            source="types",
            target="imports",
            propagate=types_to_imports,
            priority=2,
        ),
        PropagationEdge(
            source="imports",
            target="types",
            propagate=imports_to_types,
            priority=2,
        ),
        # Additional edges for control flow and semantics...
    ]
```

---

## Part 6: Typed Holes and Registry

### Phase 6.1: Hole Management (Week 3, Days 2-4)

**File**: `src/ananke/holes/hole.py`

```python
"""
Typed hole representation and management.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from ananke.core.constraint import Constraint

C = TypeVar("C", bound="Constraint")


class HoleGranularity(IntEnum):
    """
    Hierarchy of hole granularities.
    
    Coarser holes constrain finer holes:
    SYSTEM > LIBRARY > MODULE > FUNCTION > BLOCK > TERM > TOKEN
    """
    TOKEN = 0
    TERM = 1
    BLOCK = 2
    FUNCTION = 3
    MODULE = 4
    LIBRARY = 5
    SYSTEM = 6


@dataclass(frozen=True, slots=True)
class HoleId:
    """Unique identifier for a hole."""
    namespace: str
    name: str
    index: int = 0
    
    def __hash__(self) -> int:
        return hash((self.namespace, self.name, self.index))
    
    def __str__(self) -> str:
        if self.index == 0:
            return f"{self.namespace}:{self.name}"
        return f"{self.namespace}:{self.name}#{self.index}"


@dataclass
class Hole(Generic[C]):
    """
    A typed hole representing an unknown program fragment.
    
    Holes are mutable containers whose constraints narrow over time.
    They form a refinement hierarchy where coarser holes constrain finer ones.
    """
    id: HoleId
    granularity: HoleGranularity
    constraint: C
    parent: HoleId | None = None
    children: set[HoleId] = field(default_factory=set)
    
    # Provenance tracking for debugging and NL integration
    provenance: str | None = None
    
    # Resolution state
    _resolved_value: str | None = field(default=None, repr=False)
    
    def refine(self, additional: C) -> bool:
        """
        Add a constraint, narrowing this hole.
        
        Returns True if refinement succeeded, False if it caused unsatisfiability.
        """
        new_constraint = self.constraint.meet(additional)
        if new_constraint.is_bottom():
            return False
        self.constraint = new_constraint
        return True
    
    def is_resolved(self) -> bool:
        """A hole is resolved when it has a determined value."""
        return self._resolved_value is not None
    
    def resolve(self, value: str) -> None:
        """Mark this hole as resolved with a concrete value."""
        self._resolved_value = value
    
    @property
    def resolved_value(self) -> str | None:
        """Get the resolved value, if any."""
        return self._resolved_value
```

**File**: `src/ananke/holes/registry.py`

```python
"""
Registry for managing typed holes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ananke.holes.hole import Hole, HoleGranularity, HoleId

if TYPE_CHECKING:
    from ananke.core.constraint import Constraint
    from ananke.propagation.network import PropagationNetwork


@dataclass
class RegistryCheckpoint:
    """Checkpoint for registry state."""
    holes: dict[HoleId, Hole]
    resolution_order: list[HoleId]


class HoleRegistry:
    """
    Manages the hierarchy of typed holes.
    
    Responsibilities:
    1. Track all active holes
    2. Maintain parent-child relationships
    3. Propagate refinements through the hierarchy
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
        initial_constraint: Constraint,
        parent: HoleId | None = None,
        provenance: str | None = None,
    ) -> Hole:
        """Create a new hole in the registry."""
        hole: Hole = Hole(
            id=id,
            granularity=granularity,
            constraint=initial_constraint,
            parent=parent,
            provenance=provenance,
        )
        
        self.holes[id] = hole
        
        if parent is not None and parent in self.holes:
            self.holes[parent].children.add(id)
            # Inherit constraints from parent
            parent_constraint = self.holes[parent].constraint
            projected = self._project_to_child(parent_constraint, granularity)
            hole.refine(projected)
        
        return hole
    
    def refine_hole(self, id: HoleId, constraint: Constraint) -> bool:
        """
        Refine a hole with additional constraints.
        
        Propagates to:
        1. Children (they inherit the new constraint)
        2. Siblings (via the propagation network)
        3. Parent (if all children have common constraints)
        """
        if id not in self.holes:
            return False
        
        hole = self.holes[id]
        
        # Refine this hole
        if not hole.refine(constraint):
            return False
        
        # Propagate to children
        for child_id in hole.children:
            child = self.holes.get(child_id)
            if child is None:
                continue
            child_constraint = self._project_to_child(
                constraint, child.granularity
            )
            if not self.refine_hole(child_id, child_constraint):
                return False
        
        # Propagate through network
        domain = self._domain_for_granularity(hole.granularity)
        if domain and not self.network.add_constraint(domain, constraint):
            return False
        
        # Try lifting to parent
        if hole.parent is not None:
            self._try_lift_to_parent(hole.parent)
        
        return True
    
    def select_next_hole(self) -> HoleId | None:
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
        
        def score(item: tuple[HoleId, Hole]) -> tuple[float, int]:
            _, hole = item
            constraint_score = self._estimate_solution_space(hole.constraint)
            granularity_score = hole.granularity.value
            return (constraint_score, granularity_score)
        
        unresolved.sort(key=score)
        return unresolved[0][0]
    
    def mark_resolved(self, id: HoleId, value: str) -> None:
        """Mark a hole as resolved with a value."""
        if id in self.holes:
            self.holes[id].resolve(value)
            self.resolution_order.append(id)
    
    def checkpoint(self) -> RegistryCheckpoint:
        """Save registry state."""
        return RegistryCheckpoint(
            holes={id: Hole(
                id=h.id,
                granularity=h.granularity,
                constraint=h.constraint,
                parent=h.parent,
                children=set(h.children),
                provenance=h.provenance,
            ) for id, h in self.holes.items()},
            resolution_order=list(self.resolution_order),
        )
    
    def restore(self, checkpoint: RegistryCheckpoint) -> None:
        """Restore to a checkpoint."""
        self.holes = checkpoint.holes
        self.resolution_order = checkpoint.resolution_order
    
    def _project_to_child(
        self, constraint: Constraint, child_granularity: HoleGranularity
    ) -> Constraint:
        """Project a constraint to a finer granularity."""
        # Default: inherit as-is
        # Subclasses can override for domain-specific projection
        return constraint
    
    def _try_lift_to_parent(self, parent_id: HoleId) -> None:
        """Try to lift common constraints from children to parent."""
        if parent_id not in self.holes:
            return
        
        parent = self.holes[parent_id]
        children = [self.holes[cid] for cid in parent.children if cid in self.holes]
        
        if not children:
            return
        
        # Find common constraint (meet of all children)
        # Note: This finds the GLB, which is the strongest constraint
        # that all children satisfy
        common = children[0].constraint
        for child in children[1:]:
            common = common.meet(child.constraint)
        
        parent.refine(common)
    
    def _domain_for_granularity(self, granularity: HoleGranularity) -> str | None:
        """Map granularity to primary constraint domain."""
        mapping = {
            HoleGranularity.TOKEN: "syntax",
            HoleGranularity.TERM: "types",
            HoleGranularity.BLOCK: "control_flow",
            HoleGranularity.FUNCTION: "types",
            HoleGranularity.MODULE: "imports",
        }
        return mapping.get(granularity)
    
    def _estimate_solution_space(self, constraint: Constraint) -> float:
        """
        Estimate the size of the solution space.
        
        Lower values indicate more constrained (higher priority).
        """
        if constraint.is_bottom():
            return float('inf')
        if constraint.is_top():
            return float('inf')  # Too unconstrained
        
        # Heuristic based on constraint complexity
        # More sophisticated: use entropy or cardinality estimation
        return 1.0
```

---

## Part 7: Token Mask Fusion and SGLang Integration

### Phase 7.1: Token Mask Fusion (Week 3, Days 4-5)

**File**: `src/ananke/masks/fuser.py`

```python
"""
Token mask fusion from multiple constraint domains.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ananke.masks.cache import MaskCache

if TYPE_CHECKING:
    from ananke.core.domain import ConstraintDomain, GenerationContext
    from ananke.core.unified import UnifiedConstraint


class TokenMaskFuser:
    """
    Fuses token masks from multiple constraint domains.
    
    The fused mask is the intersection (conjunction) of individual masks.
    
    Optimization strategies:
    1. Order domains by expected selectivity (most selective first)
    2. Short-circuit when mask becomes all-false
    3. Cache masks per domain/constraint/position
    """
    
    def __init__(self, domains: dict[str, ConstraintDomain]):
        self.domains = domains
        self._cache = MaskCache(max_size=10000)
    
    def compute_fused_mask(
        self,
        constraint: UnifiedConstraint,
        context: GenerationContext,
    ) -> np.ndarray:
        """
        Compute the fused token mask from all domains.
        """
        # Start with all tokens allowed
        fused = np.ones(context.vocab_size, dtype=bool)
        
        # Order domains by expected selectivity
        domain_order = self._selectivity_order(constraint)
        
        for domain_name in domain_order:
            if not fused.any():
                # No tokens left, short-circuit
                break
            
            domain = self.domains[domain_name]
            domain_constraint = getattr(constraint, domain_name)
            
            # Skip if constraint is top (allows everything)
            if domain_constraint.is_top():
                continue
            
            # Check cache
            cache_key = (domain_name, hash(domain_constraint), context.position)
            cached = self._cache.get(cache_key)
            
            if cached is not None:
                mask = cached
            else:
                mask = domain.token_mask(domain_constraint, context)
                self._cache.put(cache_key, mask)
            
            # Intersect
            fused &= mask
        
        return fused
    
    def _selectivity_order(self, constraint: UnifiedConstraint) -> list[str]:
        """
        Order domains by expected selectivity.
        
        Syntax is usually most selective, semantics least.
        Could be made adaptive based on constraint specificity.
        """
        # Static ordering based on typical selectivity
        base_order = ["syntax", "types", "imports", "control_flow", "semantics"]
        
        # Boost domains with non-trivial constraints
        def priority(domain: str) -> int:
            c = getattr(constraint, domain)
            if c.is_top():
                return 100  # Low priority
            if c.is_bottom():
                return 0  # Highest priority (will short-circuit)
            return base_order.index(domain)
        
        return sorted(base_order, key=priority)
```

### Phase 7.2: SGLang Backend Integration (Week 3-4)

**File**: `src/ananke/backend/ananke_backend.py`

```python
"""
SGLang backend that provides multi-domain constrained generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from ananke.core.unified import UnifiedConstraint
from ananke.holes.factory import HoleFactory
from ananke.holes.registry import HoleRegistry
from ananke.masks.fuser import TokenMaskFuser
from ananke.masks.incremental import IncrementalMaskComputer
from ananke.parsing.partial_ast import PartialAST
from ananke.propagation.edges import build_standard_edges
from ananke.propagation.network import PropagationNetwork

if TYPE_CHECKING:
    from tokenizers import Tokenizer

    from ananke.core.domain import GenerationContext


@dataclass
class AnankeCheckpoint:
    """Complete checkpoint of Ananke state."""
    constraint: UnifiedConstraint
    context: GenerationContext
    network_state: object
    registry_state: object


class AnankeBackend:
    """
    SGLang backend that provides multi-domain constrained generation.
    
    Integrates with SGLang's generation loop to provide:
    1. Token masks fused from multiple constraint domains
    2. Constraint propagation after each token
    3. Hole refinement during generation
    4. Jump-ahead token detection for RadixAttention
    """
    
    def __init__(
        self,
        syntax_backend: Literal["llguidance", "xgrammar"] = "llguidance",
    ):
        # Build propagation network with all domains
        self.network = self._build_network(syntax_backend)
        
        # Hole management
        self.registry = HoleRegistry(self.network)
        self.factory = HoleFactory(self.registry, self.network)
        
        # Mask computation
        self.fuser = TokenMaskFuser(self.network.domains)
        self.mask_computer = IncrementalMaskComputer(self.fuser)
        
        # Generation state
        self._context: GenerationContext | None = None
        self._constraint: UnifiedConstraint | None = None
        self._checkpoints: list[AnankeCheckpoint] = []
    
    def _build_network(self, syntax_backend: str) -> PropagationNetwork:
        """Build the propagation network with all domains."""
        from ananke.domains.controlflow.domain import ControlFlowDomain
        from ananke.domains.imports.domain import ImportDomain
        from ananke.domains.semantics.domain import SemanticDomain
        from ananke.domains.syntax.domain import SyntaxDomain
        from ananke.domains.types.domain import TypeDomain
        
        network = PropagationNetwork()
        
        # Register domains
        network.register_domain(SyntaxDomain(backend=syntax_backend))
        network.register_domain(TypeDomain())
        network.register_domain(ImportDomain())
        network.register_domain(ControlFlowDomain())
        network.register_domain(SemanticDomain())
        
        # Register propagation edges
        for edge in build_standard_edges():
            network.register_edge(edge)
        
        return network
    
    def init_generation(
        self,
        initial_constraint: UnifiedConstraint,
        tokenizer: Tokenizer,
    ) -> None:
        """Initialize a new generation session."""
        from ananke.core.domain import GenerationContext
        
        self._constraint = initial_constraint
        self._context = GenerationContext(
            tokenizer=tokenizer,
            vocab_size=tokenizer.vocab_size,
            position=0,
            partial_ast=PartialAST.empty(),
        )
        
        # Propagate initial constraints
        for domain_name in ["syntax", "types", "imports", "control_flow", "semantics"]:
            domain_constraint = getattr(initial_constraint, domain_name)
            self.network.add_constraint(domain_name, domain_constraint)
        
        # Clear checkpoints
        self._checkpoints.clear()
    
    def get_token_mask(self) -> np.ndarray:
        """
        Get the current token mask.
        
        Called by SGLang before each sampling step.
        """
        if self._constraint is None or self._context is None:
            raise RuntimeError("Generation not initialized")
        
        return self.mask_computer.compute_mask(self._constraint, self._context)
    
    def get_jump_ahead_tokens(self) -> list[int]:
        """
        Get tokens that can be deterministically inserted.
        
        SGLang's RadixAttention can skip forward passes for these.
        """
        # Delegate to syntax domain's backend
        syntax_domain = self.network.domains["syntax"]
        if hasattr(syntax_domain, "get_jump_ahead_tokens"):
            return syntax_domain.get_jump_ahead_tokens()
        return []
    
    def observe_token(self, token: int) -> bool:
        """
        Update state after a token is generated.
        
        Called by SGLang after each sampling step.
        Returns False if the token leads to unsatisfiability.
        """
        if self._constraint is None or self._context is None:
            raise RuntimeError("Generation not initialized")
        
        # Update context
        self._context = self._context.advance(token)
        
        # Update each domain
        new_constraints = {}
        for domain_name, domain in self.network.domains.items():
            old_constraint = getattr(self._constraint, domain_name)
            new_constraint = domain.observe_token(
                old_constraint, token, self._context
            )
            new_constraints[domain_name] = new_constraint
        
        # Build new unified constraint
        self._constraint = UnifiedConstraint(**new_constraints)
        
        # Propagate
        if not self._propagate_all():
            return False
        
        # Update AST and create new holes if needed
        old_ast = self._context.partial_ast
        new_ast = old_ast.extend(
            self._context.tokenizer.decode([token])
        )
        if new_ast is not None:
            self._context = self._context.with_ast(new_ast)
            self.factory.on_ast_update(old_ast, new_ast)
        
        return True
    
    def _propagate_all(self) -> bool:
        """Propagate constraints through all domains."""
        if self._constraint is None:
            return True
        
        for domain_name in ["syntax", "types", "imports", "control_flow", "semantics"]:
            constraint = getattr(self._constraint, domain_name)
            if not self.network.add_constraint(domain_name, constraint):
                return False
        return True
    
    def checkpoint(self) -> int:
        """Save current state for backtracking."""
        if self._constraint is None or self._context is None:
            raise RuntimeError("Generation not initialized")
        
        cp = AnankeCheckpoint(
            constraint=self._constraint,
            context=self._context,
            network_state=self.network.checkpoint(),
            registry_state=self.registry.checkpoint(),
        )
        self._checkpoints.append(cp)
        return len(self._checkpoints) - 1
    
    def restore(self, checkpoint_id: int) -> None:
        """Restore to a previous checkpoint."""
        cp = self._checkpoints[checkpoint_id]
        self._constraint = cp.constraint
        self._context = cp.context
        self.network.restore(cp.network_state)
        self.registry.restore(cp.registry_state)
        self._checkpoints = self._checkpoints[:checkpoint_id + 1]
```

---

## Part 8: Implementation Schedule

### Phase Summary

| Phase | Week | Focus | Deliverables |
|-------|------|-------|--------------|
| 1 | 1 | Core algebra | Constraint ABC, Domain ABC, UnifiedConstraint |
| 2 | 1-2 | Syntax domain | llguidance integration, SyntaxConstraint |
| 3 | 2 | Type domain | TypeConstraint, unification, incremental checking |
| 4 | 2-3 | Propagation | PropagationNetwork, standard edges |
| 5 | 3 | Holes | HoleRegistry, HoleFactory, granularity hierarchy |
| 6 | 3-4 | Masks | TokenMaskFuser, incremental computation |
| 7 | 4 | SGLang | AnankeBackend, generation loop |
| 8 | 4-5 | Testing | Unit tests, integration tests |
| 9 | 5 | Optimization | Performance tuning, benchmarks |

### Critical Path Dependencies

```
Core Algebra (Phase 1)
    ├── Syntax Domain (Phase 2)
    │   └── llguidance integration
    ├── Type Domain (Phase 3)
    │   └── Unification engine
    └── Other Domains (Phases 2-3)
            │
            v
    Propagation Network (Phase 4)
            │
            ├── Hole Registry (Phase 5)
            │
            v
    Token Mask Fusion (Phase 6)
            │
            v
    SGLang Backend (Phase 7)
```

### Testing Strategy

1. **Unit Tests** (continuous):
   - Semilattice property tests for all constraint types
   - Unification algorithm correctness
   - Propagation network fixpoint convergence

2. **Integration Tests** (Week 4-5):
   - SGLang generation with type constraints
   - Multi-hole resolution scenarios
   - Backtracking correctness

3. **Benchmarks** (Week 5):
   - Token mask computation latency (<1ms target)
   - Constraint propagation overhead
   - Comparison with unconstrained generation

---

## Part 9: Extension Points for Future Work

### 9.1 Natural Language to Constraint Compilation

The `ConstraintSource` interface in the design document provides the extension point:

```python
class NLConstraintSource(ConstraintSource):
    """Future: Extract constraints from NL specifications."""
    
    def extract_constraints(self, nl_spec: str) -> list[Constraint]:
        # Use LLM to extract:
        # - Type signatures from descriptions
        # - Import requirements from library mentions
        # - Control flow patterns from behavior descriptions
        # - Semantic constraints from invariants
        ...
```

### 9.2 Additional Constraint Domains

The architecture supports adding new domains:
- **Security Domain**: Track taint flow, sanitization requirements
- **Performance Domain**: Complexity constraints, allocation limits
- **Style Domain**: Naming conventions, formatting rules

### 9.3 Probabilistic Constraints

Future extension to support soft constraints:
- Weighted preferences rather than hard requirements
- Integration with sampling temperature
- Constraint relaxation when unsatisfiable

---

## Part 10: Claude Code Execution Notes

### Environment Setup

```bash
# Initialize project with uv
uv init ananke
cd ananke

# Install dependencies
uv add numpy llguidance xgrammar tree-sitter z3-solver sglang
uv add --dev pytest mypy ruff

# Create source structure
mkdir -p src/ananke/{core,domains,propagation,holes,masks,parsing,backend,api}
touch src/ananke/__init__.py src/ananke/py.typed
```

### Implementation Order for Claude Code

1. **Start with core abstractions** (`src/ananke/core/`):
   - `constraint.py` - Base Constraint ABC
   - `satisfiability.py` - Satisfiability enum
   - `domain.py` - ConstraintDomain ABC, GenerationContext
   - `unified.py` - UnifiedConstraint

2. **Implement syntax domain** (`src/ananke/domains/syntax/`):
   - Start with llguidance backend
   - Test with simple grammars before complex ones

3. **Implement type domain** (`src/ananke/domains/types/`):
   - Unification first (most complex)
   - Then TypeConstraint
   - Finally TypeDomain

4. **Build propagation network** (`src/ananke/propagation/`):
   - Network first
   - Then edges incrementally

5. **Implement holes** (`src/ananke/holes/`):
   - Hole and HoleId first
   - Then Registry
   - Finally Factory

6. **Build mask fusion** (`src/ananke/masks/`):
   - Fuser first
   - Then incremental computation

7. **SGLang integration** (`src/ananke/backend/`):
   - AnankeBackend last
   - Test with simple generation scenarios

### Key Files to Create First

1. `src/ananke/core/constraint.py`
2. `src/ananke/core/domain.py`
3. `src/ananke/domains/syntax/backends/llguidance.py`
4. `src/ananke/domains/types/unification.py`
5. `src/ananke/propagation/network.py`

---

## Appendix A: llguidance vs XGrammar Decision

**Why llguidance as primary backend**:

1. **Dynamic Mask Computation**: llguidance computes masks on-the-fly (~50μs), while XGrammar requires pre-computation that can take seconds for complex grammars. For Ananke, where type constraints dynamically modify valid syntax, dynamic computation is essential.

2. **No Startup Cost**: llguidance has negligible startup cost, making it suitable for grammars that evolve during generation.

3. **CFG Support**: llguidance supports full context-free grammars via Earley parsing, not just regular expressions.

4. **Production Proven**: llguidance powers OpenAI's structured outputs and is integrated in vLLM, SGLang, llama.cpp, and Chromium.

**XGrammar as fallback**:
- For scenarios with repeated identical schemas
- Where pre-computation cost can be amortized
- When ~8μs per-token is needed (vs ~50μs for llguidance)

---

## Appendix B: Performance Targets

| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| Single domain mask | <100μs | Syntax via llguidance |
| Type constraint check | <500μs | Per token, budgeted |
| Fused mask computation | <1ms | All domains combined |
| Constraint propagation | <2ms | Full network |
| Token observation | <3ms | Including AST update |

These targets ensure that constrained generation adds <10% overhead to a typical 30ms forward pass.
