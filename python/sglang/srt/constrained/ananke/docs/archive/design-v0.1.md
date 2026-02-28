> ---
> **STATUS: HISTORICAL DOCUMENT**
> 
> This is the original design document (v0.1) for the Ananke system.
> The system has been fully implemented. For current architecture documentation, see:
> - [ARCHITECTURE.md](../ARCHITECTURE.md) - System overview
> - [ARCHITECTURE_DEEP_DIVE.md](../ARCHITECTURE_DEEP_DIVE.md) - Mathematical foundations
> 
> This document is preserved for historical reference and design rationale.
> 
> ---
> 
# Ananke: A Compositional Constraint System for Verified Code Generation

## Design Document v0.1

-----

## 1. Foundational Principles

### 1.1 Core Insight

The key observation is that constrained code generation is fundamentally a **constraint satisfaction problem with incremental observation**. Each generated token is both:

- An observation that narrows the solution space
- A decision that must satisfy all active constraints

This dual nature suggests modeling the system as a **constraint propagation network** where token generation interleaves with constraint solving.

### 1.2 Design Goals

1. **Algebraic compositionality**: Constraints combine via well-defined operations with predictable semantics
1. **Domain independence**: The propagation mechanism is agnostic to specific constraint domains
1. **Incremental computation**: All operations are efficient under single-token updates
1. **Backtracking support**: The system can checkpoint and restore state for speculative decoding
1. **Observable refinement**: Holes expose their constraint state for introspection and debugging

### 1.3 Non-Goals (Deferred)

- Natural language to constraint compilation (accommodation only)
- Probabilistic constraints (future extension)
- Distributed constraint solving (single-node focus)

-----

## 2. Algebraic Foundations

### 2.1 Constraint Semilattice

We model constraints as elements of a **bounded meet-semilattice** ⟨C, ⊓, ⊤, ⊥⟩ where:

- **C** is the set of constraints
- **⊓** (meet) is constraint conjunction: c₁ ⊓ c₂ represents “both c₁ and c₂ must hold”
- **⊤** (top) is the trivial constraint: always satisfied
- **⊥** (bottom) is the absurd constraint: never satisfied

**Properties:**

```
c ⊓ ⊤ = c                    (identity)
c ⊓ ⊥ = ⊥                    (annihilation)
c ⊓ c = c                    (idempotence)
c₁ ⊓ c₂ = c₂ ⊓ c₁            (commutativity)
(c₁ ⊓ c₂) ⊓ c₃ = c₁ ⊓ (c₂ ⊓ c₃)  (associativity)
```

The partial order c₁ ⊑ c₂ (“c₁ is at least as constraining as c₂”) is defined as c₁ ⊓ c₂ = c₁.

### 2.2 Constraint Domains as Functors

Each constraint domain D (syntax, types, imports, control flow, semantics) is modeled as a **functor** from a category of program contexts to the category of constraint semilattices:

```
D : Context → ConstraintSemilattice
```

This captures that the same constraint may have different meanings (and satisfying solutions) in different contexts.

### 2.3 Cross-Domain Morphisms

Domains are connected by **constraint morphisms** that translate constraints between domains:

```
φ : D₁(Γ) → D₂(Γ)
```

These morphisms must be **monotonic** (preserve the partial order) and **⊥-preserving** (contradictions propagate):

```
c₁ ⊑ c₂  ⟹  φ(c₁) ⊑ φ(c₂)
φ(⊥) = ⊥
```

### 2.4 The Product Domain

The unified constraint system is the **product** of individual domains:

```
Ω(Γ) = Syntax(Γ) × Types(Γ) × Imports(Γ) × ControlFlow(Γ) × Semantics(Γ)
```

With component-wise meet:

```
(s₁, t₁, i₁, cf₁, sem₁) ⊓ (s₂, t₂, i₂, cf₂, sem₂) = 
  (s₁ ⊓ s₂, t₁ ⊓ t₂, i₁ ⊓ i₂, cf₁ ⊓ cf₂, sem₁ ⊓ sem₂)
```

-----

## 3. Core Type Definitions

### 3.1 Constraint Representation

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Set, FrozenSet, Optional, Callable
from enum import Enum, auto
import numpy as np

# Type variables for generic constraint handling
C = TypeVar('C', bound='Constraint')
D = TypeVar('D', bound='ConstraintDomain')

class Satisfiability(Enum):
    SAT = auto()        # Definitely satisfiable
    UNSAT = auto()      # Definitely unsatisfiable  
    UNKNOWN = auto()    # Cannot determine (approximation)

@dataclass(frozen=True)
class Constraint(ABC):
    """
    Base class for all constraints.
    
    Constraints are immutable values in a semilattice.
    """
    
    @abstractmethod
    def meet(self, other: Constraint) -> Constraint:
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

### 3.2 Typed Holes

```python
class HoleGranularity(Enum):
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

@dataclass
class HoleId:
    """Unique identifier for a hole."""
    namespace: str
    name: str
    index: int = 0
    
    def __hash__(self):
        return hash((self.namespace, self.name, self.index))

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
    parent: Optional[HoleId] = None
    children: Set[HoleId] = field(default_factory=set)
    
    # Provenance tracking for debugging and NL integration
    provenance: Optional[str] = None
    
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
        """A hole is resolved when its constraint determines a unique value."""
        # Domain-specific: override in concrete implementations
        return False
```

### 3.3 Constraint Domains

```python
@dataclass
class ConstraintDomain(ABC, Generic[C]):
    """
    A constraint domain with its own semilattice structure.
    
    Domains are responsible for:
    1. Representing constraints in their formalism
    2. Computing token masks from constraints
    3. Updating constraints given new tokens
    4. Projecting to/from other domains
    """
    
    name: str
    
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
        """
        ...
    
    @abstractmethod
    def observe_token(self, constraint: C, token: int, context: GenerationContext) -> C:
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
```

-----

## 4. Constraint Domains

### 4.1 Syntax Domain

The syntax domain wraps an existing grammar backend (llguidance or XGrammar) and exposes it through our interface.

```python
@dataclass(frozen=True)
class SyntaxConstraint(Constraint):
    """
    Constraint on syntactic structure.
    
    Internally represented as a grammar + parser state.
    """
    grammar: Grammar
    parser_state: ParserState
    
    def meet(self, other: SyntaxConstraint) -> SyntaxConstraint:
        """
        Grammar intersection.
        
        For CFGs this is computable but potentially expensive.
        We use an approximation: the more specific grammar wins
        if one is a subset of the other, otherwise we track both.
        """
        if self.grammar.is_subset_of(other.grammar):
            return self
        if other.grammar.is_subset_of(self.grammar):
            return other
        # Neither is a subset: compute intersection or track conjunction
        return SyntaxConstraint(
            grammar=self.grammar.intersect(other.grammar),
            parser_state=self.parser_state.merge(other.parser_state)
        )
    
    def satisfiability(self) -> Satisfiability:
        if self.parser_state.is_dead():
            return Satisfiability.UNSAT
        if self.parser_state.is_accepting():
            return Satisfiability.SAT
        return Satisfiability.UNKNOWN
    
    def is_top(self) -> bool:
        return self.grammar.is_universal()
    
    def is_bottom(self) -> bool:
        return self.parser_state.is_dead()


class SyntaxDomain(ConstraintDomain[SyntaxConstraint]):
    """
    Syntax constraint domain backed by llguidance or XGrammar.
    """
    
    def __init__(self, backend: Literal["llguidance", "xgrammar"] = "llguidance"):
        self.name = "syntax"
        self.backend = backend
        self._backend_impl = self._init_backend(backend)
    
    def _init_backend(self, backend: str):
        if backend == "llguidance":
            from llguidance import LLGuidance
            return LLGuidance()
        elif backend == "xgrammar":
            from xgrammar import GrammarMatcher
            return GrammarMatcher()
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def token_mask(self, constraint: SyntaxConstraint, context: GenerationContext) -> np.ndarray:
        """Delegate to backend for efficient mask computation."""
        return self._backend_impl.compute_mask(
            constraint.grammar,
            constraint.parser_state,
            context.tokenizer
        )
    
    def observe_token(self, constraint: SyntaxConstraint, token: int, context: GenerationContext) -> SyntaxConstraint:
        """Advance parser state with new token."""
        new_state = self._backend_impl.advance(
            constraint.parser_state,
            token,
            context.tokenizer
        )
        return SyntaxConstraint(
            grammar=constraint.grammar,
            parser_state=new_state
        )
```

### 4.2 Type Domain

The type domain implements bidirectional type checking with support for typed holes.

```python
@dataclass(frozen=True)
class TypeConstraint(Constraint):
    """
    Constraint on types.
    
    Represented as a conjunction of:
    - Expected type (what type should this expression have?)
    - Type environment (what types are in scope?)
    - Unification constraints (what type equalities must hold?)
    """
    expected: Type
    environment: TypeEnvironment
    unification: FrozenSet[TypeEquation]
    
    def meet(self, other: TypeConstraint) -> TypeConstraint:
        """
        Type constraint conjunction via unification.
        """
        # Merge environments (later bindings shadow earlier)
        merged_env = self.environment.merge(other.environment)
        
        # Unify expected types
        unified_expected, new_equations = unify(self.expected, other.expected)
        
        if unified_expected is None:
            return TYPE_BOTTOM
        
        # Combine all unification constraints
        all_equations = self.unification | other.unification | new_equations
        
        # Check satisfiability via unification
        solution = solve_unification(all_equations)
        if solution is None:
            return TYPE_BOTTOM
        
        return TypeConstraint(
            expected=apply_substitution(unified_expected, solution),
            environment=apply_substitution(merged_env, solution),
            unification=all_equations
        )
    
    def satisfiability(self) -> Satisfiability:
        solution = solve_unification(self.unification)
        if solution is None:
            return Satisfiability.UNSAT
        return Satisfiability.SAT
    
    def is_top(self) -> bool:
        return self.expected == AnyType and len(self.unification) == 0
    
    def is_bottom(self) -> bool:
        return solve_unification(self.unification) is None


class TypeDomain(ConstraintDomain[TypeConstraint]):
    """
    Type constraint domain with incremental type checking.
    """
    
    def __init__(self):
        self.name = "types"
        self._type_checker = IncrementalTypeChecker()
    
    def token_mask(self, constraint: TypeConstraint, context: GenerationContext) -> np.ndarray:
        """
        Compute token mask based on type constraints.
        
        Strategy: enumerate type-valid continuations up to a budget.
        """
        mask = np.ones(context.vocab_size, dtype=bool)
        
        # Get current partial AST
        partial_ast = context.partial_ast
        
        # For each token, check if it could lead to a well-typed completion
        for token_id in range(context.vocab_size):
            token_str = context.tokenizer.decode([token_id])
            
            # Fast path: if token is clearly invalid syntactically, skip type check
            if not context.syntax_allows(token_id):
                mask[token_id] = False
                continue
            
            # Check if extending with this token could be well-typed
            extended_ast = partial_ast.extend_tentative(token_str)
            if extended_ast is not None:
                type_result = self._type_checker.check_partial(
                    extended_ast,
                    constraint.expected,
                    constraint.environment
                )
                mask[token_id] = type_result != TypeCheckResult.DEFINITELY_ILL_TYPED
        
        return mask
    
    def observe_token(self, constraint: TypeConstraint, token: int, context: GenerationContext) -> TypeConstraint:
        """
        Update type constraint after observing a token.
        
        This may:
        1. Narrow the expected type based on what was generated
        2. Add new bindings to the environment
        3. Introduce new unification constraints
        """
        token_str = context.tokenizer.decode([token])
        new_ast = context.partial_ast.extend(token_str)
        
        # Extract any new type information from the extended AST
        new_info = self._type_checker.extract_type_info(new_ast)
        
        # Update environment with new bindings
        new_env = constraint.environment
        for name, typ in new_info.new_bindings.items():
            new_env = new_env.bind(name, typ)
        
        # Add any new unification constraints
        new_unification = constraint.unification | frozenset(new_info.new_equations)
        
        # Narrow expected type if we've learned something
        new_expected = constraint.expected
        if new_info.inferred_type is not None:
            _, equations = unify(constraint.expected, new_info.inferred_type)
            new_unification = new_unification | equations
        
        return TypeConstraint(
            expected=new_expected,
            environment=new_env,
            unification=new_unification
        )
```

### 4.3 Import Domain

```python
@dataclass(frozen=True)
class ImportConstraint(Constraint):
    """
    Constraint on import graph structure.
    
    Tracks:
    - Required imports (must be present)
    - Forbidden imports (must not be present)
    - Version constraints on dependencies
    """
    required: FrozenSet[ModuleSpec]
    forbidden: FrozenSet[ModuleSpec]
    versions: FrozenDict[str, VersionConstraint]
    
    def meet(self, other: ImportConstraint) -> ImportConstraint:
        new_required = self.required | other.required
        new_forbidden = self.forbidden | other.forbidden
        
        # Check for contradiction: required ∩ forbidden ≠ ∅
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
            versions=FrozenDict(new_versions)
        )
```

### 4.4 Control Flow Domain

```python
@dataclass(frozen=True)
class ControlFlowConstraint(Constraint):
    """
    Constraint on control flow structure.
    
    Represented as a sketch of the CFG with holes for unknown blocks.
    """
    cfg_sketch: CFGSketch
    reachability: ReachabilityConstraints
    termination: Optional[TerminationRequirement]
    
    def meet(self, other: ControlFlowConstraint) -> ControlFlowConstraint:
        # Unify CFG sketches
        unified_sketch = self.cfg_sketch.unify(other.cfg_sketch)
        if unified_sketch is None:
            return CONTROLFLOW_BOTTOM
        
        # Intersect reachability constraints
        unified_reach = self.reachability.intersect(other.reachability)
        if unified_reach.is_contradictory():
            return CONTROLFLOW_BOTTOM
        
        # Combine termination requirements
        unified_term = combine_termination(self.termination, other.termination)
        
        return ControlFlowConstraint(
            cfg_sketch=unified_sketch,
            reachability=unified_reach,
            termination=unified_term
        )
```

### 4.5 Semantic Domain

```python
@dataclass(frozen=True)  
class SemanticConstraint(Constraint):
    """
    Constraint on semantic properties.
    
    Represented as SMT formulas that must be satisfiable.
    """
    formulas: FrozenSet[SMTFormula]
    
    def meet(self, other: SemanticConstraint) -> SemanticConstraint:
        combined = self.formulas | other.formulas
        return SemanticConstraint(formulas=combined)
    
    def satisfiability(self) -> Satisfiability:
        """Query SMT solver for satisfiability."""
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
    """
    Semantic constraint domain backed by Z3.
    """
    
    def __init__(self):
        self.name = "semantics"
        self._solver = IncrementalSMTSolver()
    
    def token_mask(self, constraint: SemanticConstraint, context: GenerationContext) -> np.ndarray:
        """
        Semantic constraints typically don't directly constrain tokens.
        
        Instead, they constrain via projection to other domains.
        Return all-true mask and let propagation handle it.
        """
        return np.ones(context.vocab_size, dtype=bool)
    
    def observe_token(self, constraint: SemanticConstraint, token: int, context: GenerationContext) -> SemanticConstraint:
        """
        Extract any new semantic constraints from the generated code.
        """
        # Parse assertions, contracts, invariants from generated code
        new_formulas = extract_semantic_formulas(context.partial_ast)
        return SemanticConstraint(
            formulas=constraint.formulas | frozenset(new_formulas)
        )
```

-----

## 5. Constraint Propagation Network

### 5.1 Propagation Graph

The propagation network defines how constraints flow between domains.

```python
@dataclass
class PropagationEdge:
    """
    An edge in the propagation graph.
    
    Defines how constraints in the source domain induce constraints
    in the target domain.
    """
    source: str  # Domain name
    target: str  # Domain name
    propagate: Callable[[Constraint, ConstraintContext], Constraint]
    priority: int = 0  # Lower = higher priority

class PropagationNetwork:
    """
    The constraint propagation network.
    
    Manages cross-domain constraint flow using a worklist algorithm.
    """
    
    def __init__(self):
        self.domains: Dict[str, ConstraintDomain] = {}
        self.edges: List[PropagationEdge] = []
        self.constraints: Dict[str, Constraint] = {}
        self._worklist: List[Tuple[int, str]] = []  # (priority, domain)
    
    def register_domain(self, domain: ConstraintDomain) -> None:
        self.domains[domain.name] = domain
        self.constraints[domain.name] = domain.top()
    
    def register_edge(self, edge: PropagationEdge) -> None:
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
        while self._worklist:
            _, target = heapq.heappop(self._worklist)
            
            # Compute induced constraint from all sources
            induced = self.domains[target].top()
            for edge in self.edges:
                if edge.target == target:
                    source_constraint = self.constraints[edge.source]
                    propagated = edge.propagate(source_constraint, self._context())
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
    
    def _context(self) -> ConstraintContext:
        return ConstraintContext(
            constraints=dict(self.constraints),
            domains=self.domains
        )
```

### 5.2 Standard Propagation Edges

```python
def build_standard_propagation_network() -> PropagationNetwork:
    """
    Construct the standard propagation network for code generation.
    """
    network = PropagationNetwork()
    
    # Register domains
    network.register_domain(SyntaxDomain())
    network.register_domain(TypeDomain())
    network.register_domain(ImportDomain())
    network.register_domain(ControlFlowDomain())
    network.register_domain(SemanticDomain())
    
    # Syntax → Types
    # Syntactic structure constrains what types are possible
    network.register_edge(PropagationEdge(
        source="syntax",
        target="types",
        propagate=syntax_to_types,
        priority=0
    ))
    
    # Types → Syntax
    # Type constraints can rule out syntactic forms
    # (e.g., if we need type T, only expressions that could have type T are valid)
    network.register_edge(PropagationEdge(
        source="types",
        target="syntax",
        propagate=types_to_syntax,
        priority=1
    ))
    
    # Types → Imports
    # Using a type may require importing its definition
    network.register_edge(PropagationEdge(
        source="types",
        target="imports",
        propagate=types_to_imports,
        priority=2
    ))
    
    # Imports → Types
    # Available imports determine what types are in scope
    network.register_edge(PropagationEdge(
        source="imports",
        target="types",
        propagate=imports_to_types,
        priority=2
    ))
    
    # Syntax → ControlFlow
    # Syntactic structure determines CFG shape
    network.register_edge(PropagationEdge(
        source="syntax",
        target="control_flow",
        propagate=syntax_to_controlflow,
        priority=3
    ))
    
    # ControlFlow → Semantics
    # Control flow structure induces semantic constraints
    # (e.g., reachability requirements become assertions)
    network.register_edge(PropagationEdge(
        source="control_flow",
        target="semantics",
        propagate=controlflow_to_semantics,
        priority=4
    ))
    
    # Semantics → Types (via refinement types)
    # Semantic constraints can be lifted to refinement types
    network.register_edge(PropagationEdge(
        source="semantics",
        target="types",
        propagate=semantics_to_types,
        priority=5
    ))
    
    return network


def syntax_to_types(syntax: SyntaxConstraint, ctx: ConstraintContext) -> TypeConstraint:
    """
    Propagate syntactic structure to type constraints.
    
    Example: if syntax forces "x + _", the hole must have a type
    that supports addition with x's type.
    """
    partial_ast = ctx.partial_ast
    
    if partial_ast is None:
        return ctx.domains["types"].top()
    
    # Find type expectations induced by syntactic position
    hole_contexts = partial_ast.find_hole_contexts()
    
    equations = set()
    for hole_ctx in hole_contexts:
        if hole_ctx.expected_type is not None:
            equations.add(TypeEquation(
                hole_ctx.hole_type_var,
                hole_ctx.expected_type
            ))
    
    return TypeConstraint(
        expected=AnyType,
        environment=ctx.type_environment,
        unification=frozenset(equations)
    )


def types_to_syntax(types: TypeConstraint, ctx: ConstraintContext) -> SyntaxConstraint:
    """
    Propagate type constraints to syntactic constraints.
    
    Example: if the expected type is Int, rule out string literals.
    """
    if types.is_top():
        return ctx.domains["syntax"].top()
    
    # Enumerate syntactic forms that could produce the expected type
    valid_forms = enumerate_forms_for_type(types.expected, types.environment)
    
    if not valid_forms:
        return ctx.domains["syntax"].bottom()
    
    # Construct grammar that only allows these forms
    restricted_grammar = Grammar.from_alternatives(valid_forms)
    
    return SyntaxConstraint(
        grammar=restricted_grammar,
        parser_state=ParserState.initial(restricted_grammar)
    )
```

-----

## 6. Hole Management

### 6.1 Hole Registry

```python
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
        self.holes: Dict[HoleId, Hole] = {}
        self.resolution_order: List[HoleId] = []
    
    def create_hole(
        self,
        id: HoleId,
        granularity: HoleGranularity,
        initial_constraint: Constraint,
        parent: Optional[HoleId] = None,
        provenance: Optional[str] = None
    ) -> Hole:
        """Create a new hole in the registry."""
        hole = Hole(
            id=id,
            granularity=granularity,
            constraint=initial_constraint,
            parent=parent,
            provenance=provenance
        )
        
        self.holes[id] = hole
        
        if parent is not None:
            self.holes[parent].children.add(id)
            # Inherit constraints from parent
            parent_constraint = self.holes[parent].constraint
            hole.refine(self._project_to_child(parent_constraint, granularity))
        
        return hole
    
    def refine_hole(self, id: HoleId, constraint: Constraint) -> bool:
        """
        Refine a hole with additional constraints.
        
        Propagates to:
        1. Children (they inherit the new constraint)
        2. Siblings (via the propagation network)
        3. Parent (if all children have common constraints, lift to parent)
        """
        hole = self.holes[id]
        
        # Refine this hole
        if not hole.refine(constraint):
            return False
        
        # Propagate to children
        for child_id in hole.children:
            child_constraint = self._project_to_child(constraint, self.holes[child_id].granularity)
            if not self.refine_hole(child_id, child_constraint):
                return False
        
        # Propagate through network (handles sibling propagation)
        domain = self._domain_for_granularity(hole.granularity)
        if not self.network.add_constraint(domain, constraint):
            return False
        
        # Check if we can lift to parent
        if hole.parent is not None:
            self._try_lift_to_parent(hole.parent)
        
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
        
        # Score by constraint strength and granularity
        def score(item):
            id, hole = item
            constraint_score = self._estimate_solution_space(hole.constraint)
            granularity_score = hole.granularity.value
            return (constraint_score, granularity_score)
        
        unresolved.sort(key=score)
        return unresolved[0][0]
    
    def _project_to_child(self, constraint: Constraint, child_granularity: HoleGranularity) -> Constraint:
        """Project a constraint to a finer granularity."""
        # Implementation depends on constraint type
        # Generally: keep constraints that are relevant at the finer level
        return constraint  # Default: inherit as-is
    
    def _try_lift_to_parent(self, parent_id: HoleId) -> None:
        """
        Try to lift common constraints from children to parent.
        
        If all children have constraint C, parent should also have C.
        """
        parent = self.holes[parent_id]
        children = [self.holes[cid] for cid in parent.children]
        
        if not children:
            return
        
        # Find common constraint (greatest lower bound of all children)
        common = children[0].constraint
        for child in children[1:]:
            # This is actually finding the join (least upper bound)
            # since we want the weakest constraint all children satisfy
            common = self._join_constraints(common, child.constraint)
        
        # Refine parent with common constraint
        parent.refine(common)
```

### 6.2 Dynamic Hole Creation During Generation

```python
class HoleFactory:
    """
    Creates holes dynamically as generation reveals program structure.
    """
    
    def __init__(self, registry: HoleRegistry, network: PropagationNetwork):
        self.registry = registry
        self.network = network
        self._hole_counter = 0
    
    def on_ast_update(self, old_ast: PartialAST, new_ast: PartialAST) -> List[Hole]:
        """
        Called when the partial AST is updated with new tokens.
        
        Detects new hole sites and creates holes for them.
        """
        new_holes = []
        
        # Find new incomplete nodes in the AST
        new_incomplete = new_ast.incomplete_nodes() - old_ast.incomplete_nodes()
        
        for node in new_incomplete:
            # Determine granularity from node type
            granularity = self._granularity_for_node(node)
            
            # Extract initial constraint from context
            initial_constraint = self._constraint_for_node(node)
            
            # Find parent hole if any
            parent = self._find_enclosing_hole(node)
            
            # Create the hole
            hole = self.registry.create_hole(
                id=self._next_id(node),
                granularity=granularity,
                initial_constraint=initial_constraint,
                parent=parent
            )
            
            new_holes.append(hole)
        
        return new_holes
    
    def _granularity_for_node(self, node: ASTNode) -> HoleGranularity:
        """Map AST node types to hole granularities."""
        mapping = {
            'program': HoleGranularity.MODULE,
            'function_declaration': HoleGranularity.FUNCTION,
            'block_statement': HoleGranularity.BLOCK,
            'expression': HoleGranularity.TERM,
            'identifier': HoleGranularity.TOKEN,
        }
        return mapping.get(node.type, HoleGranularity.TERM)
    
    def _constraint_for_node(self, node: ASTNode) -> Constraint:
        """Extract initial constraint from AST context."""
        # Get type expectation from parent
        type_constraint = self._type_expectation(node)
        
        # Get syntactic constraint from grammar
        syntax_constraint = self._syntax_expectation(node)
        
        # Combine into unified constraint
        return UnifiedConstraint(
            syntax=syntax_constraint,
            types=type_constraint,
            imports=IMPORT_TOP,
            control_flow=CONTROLFLOW_TOP,
            semantics=SEMANTIC_TOP
        )
```

-----

## 7. Unified Constraint and Token Mask Fusion

### 7.1 Unified Constraint Type

```python
@dataclass(frozen=True)
class UnifiedConstraint(Constraint):
    """
    Product of all constraint domains.
    
    This is the constraint type that the generation loop operates on.
    """
    syntax: SyntaxConstraint
    types: TypeConstraint
    imports: ImportConstraint
    control_flow: ControlFlowConstraint
    semantics: SemanticConstraint
    
    def meet(self, other: UnifiedConstraint) -> UnifiedConstraint:
        return UnifiedConstraint(
            syntax=self.syntax.meet(other.syntax),
            types=self.types.meet(other.types),
            imports=self.imports.meet(other.imports),
            control_flow=self.control_flow.meet(other.control_flow),
            semantics=self.semantics.meet(other.semantics)
        )
    
    def satisfiability(self) -> Satisfiability:
        # All domains must be satisfiable
        results = [
            self.syntax.satisfiability(),
            self.types.satisfiability(),
            self.imports.satisfiability(),
            self.control_flow.satisfiability(),
            self.semantics.satisfiability()
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
            self.semantics.is_top()
        ])
    
    def is_bottom(self) -> bool:
        return any([
            self.syntax.is_bottom(),
            self.types.is_bottom(),
            self.imports.is_bottom(),
            self.control_flow.is_bottom(),
            self.semantics.is_bottom()
        ])
```

### 7.2 Token Mask Fusion

```python
class TokenMaskFuser:
    """
    Fuses token masks from multiple constraint domains.
    
    The fused mask is the intersection (conjunction) of individual masks.
    """
    
    def __init__(self, domains: Dict[str, ConstraintDomain]):
        self.domains = domains
        self._cache = MaskCache()
    
    def compute_fused_mask(
        self,
        constraint: UnifiedConstraint,
        context: GenerationContext
    ) -> np.ndarray:
        """
        Compute the fused token mask from all domains.
        
        Optimization: compute in order of expected selectivity,
        short-circuit when mask becomes all-false.
        """
        # Start with all tokens allowed
        fused = np.ones(context.vocab_size, dtype=bool)
        
        # Order domains by expected selectivity (most selective first)
        domain_order = self._selectivity_order(constraint)
        
        for domain_name in domain_order:
            if not fused.any():
                # No tokens left, short-circuit
                break
            
            domain = self.domains[domain_name]
            domain_constraint = getattr(constraint, domain_name)
            
            # Check cache
            cache_key = (domain_name, domain_constraint, context.position)
            if cache_key in self._cache:
                mask = self._cache[cache_key]
            else:
                mask = domain.token_mask(domain_constraint, context)
                self._cache[cache_key] = mask
            
            # Intersect
            fused &= mask
        
        return fused
    
    def _selectivity_order(self, constraint: UnifiedConstraint) -> List[str]:
        """
        Order domains by expected selectivity.
        
        Syntax is usually most selective, semantics least.
        """
        # Could be made adaptive based on constraint specificity
        return ["syntax", "types", "imports", "control_flow", "semantics"]
```

### 7.3 Incremental Mask Updates

```python
class IncrementalMaskComputer:
    """
    Efficiently update masks as generation proceeds.
    
    Key insight: after generating token t, most of the mask computation
    from the previous step can be reused.
    """
    
    def __init__(self, fuser: TokenMaskFuser):
        self.fuser = fuser
        self._previous_masks: Dict[str, np.ndarray] = {}
        self._previous_constraint: Optional[UnifiedConstraint] = None
    
    def compute_mask(
        self,
        constraint: UnifiedConstraint,
        context: GenerationContext
    ) -> np.ndarray:
        """
        Compute mask incrementally when possible.
        """
        if self._previous_constraint is None:
            # First call: compute from scratch
            mask = self.fuser.compute_fused_mask(constraint, context)
            self._previous_masks = self._domain_masks(constraint, context)
            self._previous_constraint = constraint
            return mask
        
        # Determine which domains have changed
        changed_domains = self._changed_domains(
            self._previous_constraint,
            constraint
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
        self,
        old: UnifiedConstraint,
        new: UnifiedConstraint
    ) -> Set[str]:
        """Identify which domains have different constraints."""
        changed = set()
        for domain in ["syntax", "types", "imports", "control_flow", "semantics"]:
            if getattr(old, domain) != getattr(new, domain):
                changed.add(domain)
        return changed
```

-----

## 8. SGLang Integration

### 8.1 Ananke Backend for SGLang

```python
class AnankeBackend:
    """
    SGLang backend that provides multi-domain constrained generation.
    
    Integrates with SGLang's generation loop to provide:
    1. Token masks fused from multiple constraint domains
    2. Constraint propagation after each token
    3. Hole refinement during generation
    """
    
    def __init__(
        self,
        syntax_backend: Literal["llguidance", "xgrammar"] = "llguidance"
    ):
        self.network = build_standard_propagation_network()
        self.registry = HoleRegistry(self.network)
        self.factory = HoleFactory(self.registry, self.network)
        self.fuser = TokenMaskFuser(self.network.domains)
        self.mask_computer = IncrementalMaskComputer(self.fuser)
        
        # Initialize syntax domain with chosen backend
        self.network.domains["syntax"] = SyntaxDomain(backend=syntax_backend)
        
        # Current generation state
        self._context: Optional[GenerationContext] = None
        self._constraint: Optional[UnifiedConstraint] = None
        self._checkpoints: List[AnankeCheckpoint] = []
    
    def init_generation(
        self,
        initial_constraint: UnifiedConstraint,
        tokenizer: Tokenizer
    ) -> None:
        """Initialize a new generation session."""
        self._constraint = initial_constraint
        self._context = GenerationContext(
            tokenizer=tokenizer,
            vocab_size=tokenizer.vocab_size,
            position=0,
            partial_ast=PartialAST.empty()
        )
        
        # Propagate initial constraints
        for domain_name in ["syntax", "types", "imports", "control_flow", "semantics"]:
            domain_constraint = getattr(initial_constraint, domain_name)
            self.network.add_constraint(domain_name, domain_constraint)
    
    def get_token_mask(self) -> np.ndarray:
        """
        Get the current token mask.
        
        Called by SGLang before each sampling step.
        """
        return self.mask_computer.compute_mask(self._constraint, self._context)
    
    def observe_token(self, token: int) -> bool:
        """
        Update state after a token is generated.
        
        Called by SGLang after each sampling step.
        Returns False if the token leads to unsatisfiability.
        """
        # Update context
        self._context = self._context.advance(token)
        
        # Update each domain
        new_constraints = {}
        for domain_name, domain in self.network.domains.items():
            old_constraint = getattr(self._constraint, domain_name)
            new_constraint = domain.observe_token(old_constraint, token, self._context)
            new_constraints[domain_name] = new_constraint
        
        # Build new unified constraint
        self._constraint = UnifiedConstraint(**new_constraints)
        
        # Propagate
        if not self._propagate_all():
            return False
        
        # Create new holes if AST reveals them
        old_ast = self._context.partial_ast
        new_ast = self._context.update_ast(token)
        self._context = self._context.with_ast(new_ast)
        self.factory.on_ast_update(old_ast, new_ast)
        
        return True
    
    def _propagate_all(self) -> bool:
        """Propagate constraints through all domains."""
        for domain_name in ["syntax", "types", "imports", "control_flow", "semantics"]:
            constraint = getattr(self._constraint, domain_name)
            if not self.network.add_constraint(domain_name, constraint):
                return False
        return True
    
    def checkpoint(self) -> int:
        """
        Save current state for backtracking.
        
        Returns checkpoint ID.
        """
        cp = AnankeCheckpoint(
            constraint=self._constraint,
            context=self._context,
            network_state=self.network.checkpoint(),
            registry_state=self.registry.checkpoint()
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
        # Discard later checkpoints
        self._checkpoints = self._checkpoints[:checkpoint_id + 1]
```

### 8.2 SGLang Function Integration

```python
import sglang as sgl
from sglang.lang.ir import SglExpr

@dataclass
class AnankeConstrainedGen(SglExpr):
    """
    SGLang expression for Ananke-constrained generation.
    """
    name: str
    constraint: UnifiedConstraint
    max_tokens: int = 256
    stop: Optional[List[str]] = None
    
    def interpret(self, backend: AnankeBackend, state: SglState) -> SglState:
        """Execute constrained generation."""
        backend.init_generation(
            initial_constraint=self.constraint,
            tokenizer=state.tokenizer
        )
        
        generated = []
        for _ in range(self.max_tokens):
            # Get constrained token mask
            mask = backend.get_token_mask()
            
            if not mask.any():
                # No valid tokens - generation stuck
                break
            
            # Sample with mask
            token = state.sample_with_mask(mask)
            
            # Check for stop condition
            if self._should_stop(token, generated, state):
                break
            
            # Update backend state
            if not backend.observe_token(token):
                # Constraint became unsatisfiable
                break
            
            generated.append(token)
        
        # Store result
        state[self.name] = state.tokenizer.decode(generated)
        return state


def ananke_gen(
    name: str,
    constraint: UnifiedConstraint,
    max_tokens: int = 256,
    stop: Optional[List[str]] = None
) -> AnankeConstrainedGen:
    """
    Create an Ananke-constrained generation expression.
    
    Usage:
        @sgl.function
        def my_program(s):
            s += "def fibonacci(n):"
            s += ananke_gen(
                "body",
                constraint=UnifiedConstraint(
                    syntax=python_function_body_grammar,
                    types=TypeConstraint(
                        expected=FunctionType(Int, Int),
                        environment=TypeEnv.empty()
                    ),
                    ...
                ),
                max_tokens=500
            )
    """
    return AnankeConstrainedGen(
        name=name,
        constraint=constraint,
        max_tokens=max_tokens,
        stop=stop
    )
```

### 8.3 High-Level API

```python
class Ananke:
    """
    High-level API for Ananke-constrained code generation.
    """
    
    def __init__(
        self,
        model: str,
        syntax_backend: Literal["llguidance", "xgrammar"] = "llguidance"
    ):
        self.backend = AnankeBackend(syntax_backend=syntax_backend)
        self.sgl_backend = sgl.RuntimeEndpoint(model)
    
    def generate(
        self,
        prompt: str,
        holes: List[HoleSpec],
        max_tokens: int = 1024
    ) -> GenerationResult:
        """
        Generate code with typed holes.
        
        Args:
            prompt: Initial code/prompt
            holes: Specifications for holes to fill
            max_tokens: Maximum tokens to generate
            
        Returns:
            GenerationResult with filled code and resolution trace
        """
        # Register holes
        for spec in holes:
            self.backend.registry.create_hole(
                id=spec.id,
                granularity=spec.granularity,
                initial_constraint=spec.constraint,
                provenance=spec.provenance
            )
        
        # Build constraint from holes
        constraint = self._constraint_from_holes(holes)
        
        @sgl.function
        def generate_program(s):
            s += prompt
            s += ananke_gen("code", constraint=constraint, max_tokens=max_tokens)
        
        # Run generation
        state = generate_program.run(backend=self.sgl_backend)
        
        return GenerationResult(
            code=state["code"],
            holes=self.backend.registry.holes,
            trace=self.backend.registry.resolution_order
        )
    
    def _constraint_from_holes(self, holes: List[HoleSpec]) -> UnifiedConstraint:
        """Build unified constraint from hole specifications."""
        syntax = SYNTAX_TOP
        types = TYPE_TOP
        imports = IMPORT_TOP
        control_flow = CONTROLFLOW_TOP
        semantics = SEMANTIC_TOP
        
        for spec in holes:
            c = spec.constraint
            if isinstance(c, UnifiedConstraint):
                syntax = syntax.meet(c.syntax)
                types = types.meet(c.types)
                imports = imports.meet(c.imports)
                control_flow = control_flow.meet(c.control_flow)
                semantics = semantics.meet(c.semantics)
        
        return UnifiedConstraint(
            syntax=syntax,
            types=types,
            imports=imports,
            control_flow=control_flow,
            semantics=semantics
        )


# Convenience constructors for holes
def term_hole(
    name: str,
    expected_type: Type,
    env: Optional[TypeEnvironment] = None
) -> HoleSpec:
    """Create a term-level hole with type constraint."""
    return HoleSpec(
        id=HoleId("term", name),
        granularity=HoleGranularity.TERM,
        constraint=UnifiedConstraint(
            syntax=SYNTAX_TOP,
            types=TypeConstraint(
                expected=expected_type,
                environment=env or TypeEnvironment.empty(),
                unification=frozenset()
            ),
            imports=IMPORT_TOP,
            control_flow=CONTROLFLOW_TOP,
            semantics=SEMANTIC_TOP
        )
    )

def function_hole(
    name: str,
    signature: FunctionType,
    preconditions: Optional[List[SMTFormula]] = None,
    postconditions: Optional[List[SMTFormula]] = None
) -> HoleSpec:
    """Create a function-level hole with signature and contracts."""
    return HoleSpec(
        id=HoleId("function", name),
        granularity=HoleGranularity.FUNCTION,
        constraint=UnifiedConstraint(
            syntax=SYNTAX_TOP,
            types=TypeConstraint(
                expected=signature,
                environment=TypeEnvironment.empty(),
                unification=frozenset()
            ),
            imports=IMPORT_TOP,
            control_flow=CONTROLFLOW_TOP,
            semantics=SemanticConstraint(
                formulas=frozenset(preconditions or []) | frozenset(postconditions or [])
            )
        )
    )
```

-----

## 9. Performance Optimizations

### 9.1 Lazy Domain Evaluation

```python
class LazyConstraintEvaluator:
    """
    Evaluates constraints lazily, only computing what's needed.
    
    Key insight: many tokens are ruled out by syntax alone.
    Type/semantic checking is only needed for syntactically valid tokens.
    """
    
    def __init__(self, domains: Dict[str, ConstraintDomain]):
        self.domains = domains
        self._syntax_domain = domains["syntax"]
    
    def compute_mask(
        self,
        constraint: UnifiedConstraint,
        context: GenerationContext
    ) -> np.ndarray:
        # Phase 1: Syntax (fast, rules out most tokens)
        syntax_mask = self._syntax_domain.token_mask(constraint.syntax, context)
        
        # If syntax rules out everything, done
        valid_count = syntax_mask.sum()
        if valid_count == 0:
            return syntax_mask
        
        # Phase 2: Types (only for syntactically valid tokens)
        # If few tokens remain, check them individually
        if valid_count < 100:
            return self._check_individually(syntax_mask, constraint, context)
        
        # Otherwise, compute type mask in bulk
        type_mask = self.domains["types"].token_mask(constraint.types, context)
        combined = syntax_mask & type_mask
        
        # Phase 3: Other domains only if still many candidates
        if combined.sum() > 50:
            for domain_name in ["imports", "control_flow", "semantics"]:
                domain_constraint = getattr(constraint, domain_name)
                if not domain_constraint.is_top():
                    domain_mask = self.domains[domain_name].token_mask(
                        domain_constraint, context
                    )
                    combined &= domain_mask
        
        return combined
    
    def _check_individually(
        self,
        syntax_mask: np.ndarray,
        constraint: UnifiedConstraint,
        context: GenerationContext
    ) -> np.ndarray:
        """Check remaining tokens one by one through all domains."""
        result = syntax_mask.copy()
        valid_indices = np.where(syntax_mask)[0]
        
        for token_id in valid_indices:
            # Check each domain
            for domain_name in ["types", "imports", "control_flow", "semantics"]:
                domain = self.domains[domain_name]
                domain_constraint = getattr(constraint, domain_name)
                
                if domain_constraint.is_top():
                    continue
                
                if not domain.token_valid(domain_constraint, token_id, context):
                    result[token_id] = False
                    break  # No need to check other domains
        
        return result
```

### 9.2 Constraint Caching

```python
class ConstraintCache:
    """
    Caches constraint computations across generation steps.
    
    Exploits the fact that constraints often have local structure:
    the constraint at position N+1 differs from position N only in
    the most recently generated part.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[CacheKey, CacheEntry] = {}
        self._access_order: List[CacheKey] = []
    
    @dataclass(frozen=True)
    class CacheKey:
        domain: str
        constraint_hash: int
        context_hash: int
    
    @dataclass
    class CacheEntry:
        mask: np.ndarray
        timestamp: int
    
    def get(
        self,
        domain: str,
        constraint: Constraint,
        context: GenerationContext
    ) -> Optional[np.ndarray]:
        key = self.CacheKey(
            domain=domain,
            constraint_hash=hash(constraint),
            context_hash=self._context_hash(context)
        )
        
        entry = self._cache.get(key)
        if entry is not None:
            # Move to end of access order (LRU)
            self._access_order.remove(key)
            self._access_order.append(key)
            return entry.mask
        
        return None
    
    def put(
        self,
        domain: str,
        constraint: Constraint,
        context: GenerationContext,
        mask: np.ndarray
    ) -> None:
        key = self.CacheKey(
            domain=domain,
            constraint_hash=hash(constraint),
            context_hash=self._context_hash(context)
        )
        
        # Evict if full
        while len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = self.CacheEntry(
            mask=mask,
            timestamp=len(self._access_order)
        )
        self._access_order.append(key)
    
    def _context_hash(self, context: GenerationContext) -> int:
        # Hash based on position and recent tokens
        # (full token history would be too expensive)
        return hash((context.position, tuple(context.recent_tokens(10))))
```

### 9.3 Parallel Domain Evaluation

```python
import concurrent.futures
from threading import Lock

class ParallelMaskComputer:
    """
    Compute domain masks in parallel.
    
    Useful when multiple domains have expensive mask computations.
    """
    
    def __init__(
        self,
        domains: Dict[str, ConstraintDomain],
        max_workers: int = 4
    ):
        self.domains = domains
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._result_lock = Lock()
    
    def compute_fused_mask(
        self,
        constraint: UnifiedConstraint,
        context: GenerationContext
    ) -> np.ndarray:
        # Submit all domain computations
        futures = {}
        for domain_name, domain in self.domains.items():
            domain_constraint = getattr(constraint, domain_name)
            if not domain_constraint.is_top():
                future = self.executor.submit(
                    domain.token_mask,
                    domain_constraint,
                    context
                )
                futures[domain_name] = future
        
        # Collect results and fuse
        fused = np.ones(context.vocab_size, dtype=bool)
        for domain_name, future in futures.items():
            mask = future.result()
            with self._result_lock:
                fused &= mask
        
        return fused
```

-----

## 10. Extension Points for NL Integration

### 10.1 Constraint Source Interface

```python
class ConstraintSource(ABC):
    """
    Abstract source of constraints.
    
    Implemented by:
    - DirectConstraintSource: programmatic constraints
    - NLConstraintSource: constraints extracted from natural language
    - HybridConstraintSource: combination of both
    """
    
    @abstractmethod
    def extract_constraints(self, input: Any) -> List[Constraint]:
        """Extract constraints from input."""
        ...
    
    @abstractmethod
    def supported_domains(self) -> Set[str]:
        """Which constraint domains does this source produce?"""
        ...


class NLConstraintSource(ConstraintSource):
    """
    Extract constraints from natural language specifications.
    
    This is a placeholder for future NL-to-constraint compilation.
    """
    
    def __init__(self, nl_compiler: Optional[NLConstraintCompiler] = None):
        self.compiler = nl_compiler
    
    def extract_constraints(self, nl_spec: str) -> List[Constraint]:
        if self.compiler is None:
            raise NotImplementedError(
                "NL constraint compilation not yet implemented. "
                "Please provide constraints programmatically."
            )
        
        return self.compiler.compile(nl_spec)
    
    def supported_domains(self) -> Set[str]:
        # NL can potentially produce constraints for any domain
        return {"syntax", "types", "imports", "control_flow", "semantics"}
```

### 10.2 Provenance Tracking

```python
@dataclass
class ConstraintProvenance:
    """
    Tracks where a constraint came from.
    
    Useful for:
    1. Debugging: why was this token rejected?
    2. NL integration: linking constraints back to specifications
    3. Explanation: generating human-readable constraint descriptions
    """
    source: str  # "programmatic", "nl", "inferred", etc.
    location: Optional[SourceLocation] = None
    nl_fragment: Optional[str] = None
    parent_constraint: Optional[ConstraintProvenance] = None
    
    def explain(self) -> str:
        """Generate human-readable explanation."""
        if self.nl_fragment:
            return f"From specification: '{self.nl_fragment}'"
        if self.source == "inferred":
            return f"Inferred from {self.parent_constraint.explain()}"
        return f"Programmatic constraint from {self.location}"


@dataclass
class ProvenancedConstraint(Generic[C]):
    """Constraint with provenance information."""
    constraint: C
    provenance: ConstraintProvenance
    
    def meet(self, other: ProvenancedConstraint[C]) -> ProvenancedConstraint[C]:
        return ProvenancedConstraint(
            constraint=self.constraint.meet(other.constraint),
            provenance=ConstraintProvenance(
                source="combined",
                parent_constraint=self.provenance  # Could track both parents
            )
        )
```

### 10.3 Constraint Explanation

```python
class ConstraintExplainer:
    """
    Explains why a token was accepted or rejected.
    
    Useful for debugging and for NL integration (explaining
    generation decisions in natural language).
    """
    
    def __init__(self, domains: Dict[str, ConstraintDomain]):
        self.domains = domains
    
    def explain_rejection(
        self,
        token: int,
        constraint: UnifiedConstraint,
        context: GenerationContext
    ) -> str:
        """Explain why a token was rejected."""
        reasons = []
        
        for domain_name, domain in self.domains.items():
            domain_constraint = getattr(constraint, domain_name)
            mask = domain.token_mask(domain_constraint, context)
            
            if not mask[token]:
                reason = domain.explain_rejection(
                    token, domain_constraint, context
                )
                reasons.append(f"[{domain_name}] {reason}")
        
        if not reasons:
            return "Token was accepted by all domains"
        
        return "Token rejected:\n" + "\n".join(reasons)
    
    def explain_acceptance(
        self,
        token: int,
        constraint: UnifiedConstraint,
        context: GenerationContext
    ) -> str:
        """Explain why a token was accepted."""
        explanations = []
        
        for domain_name, domain in self.domains.items():
            domain_constraint = getattr(constraint, domain_name)
            explanation = domain.explain_acceptance(
                token, domain_constraint, context
            )
            if explanation:
                explanations.append(f"[{domain_name}] {explanation}")
        
        return "Token accepted:\n" + "\n".join(explanations)
```

-----

## 11. Example Usage

### 11.1 Simple Type-Constrained Generation

```python
from ananke import Ananke, term_hole, function_hole
from ananke.types import Int, String, List, FunctionType

# Initialize Ananke
ananke = Ananke(model="meta-llama/Llama-3-8B-Instruct")

# Generate a function with type constraints
result = ananke.generate(
    prompt="def process_items(items: List[str]) -> int:",
    holes=[
        function_hole(
            name="body",
            signature=FunctionType(
                params=[("items", List(String))],
                returns=Int
            )
        )
    ],
    max_tokens=200
)

print(result.code)
# Output: well-typed function body
```

### 11.2 Multi-Hole Generation with Propagation

```python
from ananke import Ananke, HoleSpec, HoleGranularity
from ananke.constraints import (
    UnifiedConstraint, TypeConstraint, ImportConstraint,
    SemanticConstraint
)

ananke = Ananke(model="meta-llama/Llama-3-8B-Instruct")

# Define interconnected holes
result = ananke.generate(
    prompt="""
class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
    
    def process(self, data: DataFrame) -> DataFrame:
        # HOLE: processing_logic
        pass
    
    def validate(self, result: DataFrame) -> bool:
        # HOLE: validation_logic
        pass
""",
    holes=[
        HoleSpec(
            id=HoleId("function", "processing_logic"),
            granularity=HoleGranularity.FUNCTION,
            constraint=UnifiedConstraint(
                syntax=SYNTAX_TOP,
                types=TypeConstraint(
                    expected=FunctionType(
                        params=[("self", "DataProcessor"), ("data", "DataFrame")],
                        returns="DataFrame"
                    ),
                    environment=TypeEnvironment.empty()
                ),
                imports=ImportConstraint(
                    required=frozenset([ModuleSpec("pandas")]),
                    forbidden=frozenset(),
                    versions=FrozenDict({"pandas": VersionConstraint(">=1.0.0")})
                ),
                control_flow=CONTROLFLOW_TOP,
                semantics=SEMANTIC_TOP
            ),
            provenance="Process input DataFrame and return transformed DataFrame"
        ),
        HoleSpec(
            id=HoleId("function", "validation_logic"),
            granularity=HoleGranularity.FUNCTION,
            constraint=UnifiedConstraint(
                syntax=SYNTAX_TOP,
                types=TypeConstraint(
                    expected=FunctionType(
                        params=[("self", "DataProcessor"), ("result", "DataFrame")],
                        returns="bool"
                    ),
                    environment=TypeEnvironment.empty()
                ),
                imports=IMPORT_TOP,
                control_flow=CONTROLFLOW_TOP,
                semantics=SemanticConstraint(
                    formulas=frozenset([
                        # Validation must be pure (no side effects)
                        SMTFormula.parse("no_side_effects(validation_logic)")
                    ])
                )
            ),
            provenance="Validate the processed result"
        )
    ],
    max_tokens=500
)

# Constraints propagate: if processing_logic uses a pandas method,
# validation_logic can use it too (import is shared)
```

### 11.3 Progressive Refinement Example

```python
from ananke import Ananke
from ananke.refinement import RefinementSession

ananke = Ananke(model="meta-llama/Llama-3-8B-Instruct")

# Start a refinement session
session = RefinementSession(ananke)

# Initial generation with broad constraints
session.generate(
    prompt="def sort_and_filter(items):",
    initial_constraint=UnifiedConstraint(
        syntax=python_function_grammar,
        types=TYPE_TOP,  # No type constraints yet
        imports=IMPORT_TOP,
        control_flow=CONTROLFLOW_TOP,
        semantics=SEMANTIC_TOP
    )
)

# After seeing the generated structure, refine with more constraints
# (This simulates what would happen automatically during generation)
session.refine(
    hole_id=HoleId("term", "filter_predicate"),
    constraint=TypeConstraint(
        expected=FunctionType([("x", "Any")], "bool"),
        environment=session.current_environment
    )
)

# Continue generation with narrowed constraints
result = session.continue_generation(max_tokens=100)
```

-----

## 12. Summary

Ananke provides a compositional constraint system for verified code generation with:

1. **Algebraic foundations**: Constraints form semilattices with well-defined meet operations
1. **Multi-domain fusion**: Syntax, types, imports, control flow, and semantics unified through product construction
1. **Progressive refinement**: Holes narrow as generation proceeds via constraint propagation
1. **SGLang integration**: Clean backend API for constrained generation
1. **Performance**: Lazy evaluation, caching, and parallelism for efficient mask computation
1. **Extensibility**: Clear extension points for NL-to-constraint compilation

The system is designed to be backend-agnostic (llguidance or XGrammar for syntax), domain-extensible (new constraint domains can be added), and integration-ready (SGLang, but adaptable to other frameworks).​​​​​​​​​​​​​​​​