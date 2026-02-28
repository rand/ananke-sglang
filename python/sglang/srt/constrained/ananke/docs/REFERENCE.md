# Ananke API Reference

Complete API reference for the Ananke multi-domain constrained generation system.

---

## Table of Contents

1. [Backend Classes](#backend-classes)
2. [Grammar Classes](#grammar-classes)
3. [Constraint Specification](#constraint-specification)
4. [Core Constraint System](#core-constraint-system)
5. [Domain Constraints](#domain-constraints)
6. [Checkpointing](#checkpointing)
7. [Typed Holes](#typed-holes)
8. [Performance Utilities](#performance-utilities)
9. [Type System](#type-system)
10. [Configuration Reference](#configuration-reference)
11. [Error Handling](#error-handling)

---

## Backend Classes

### AnankeBackend

**Module:** `sglang.srt.constrained.ananke.backend.backend`

Multi-domain constrained decoding backend for SGLang.

```python
class AnankeBackend(BaseGrammarBackend):
    """Orchestrates constraint checking across multiple domains."""

    def __init__(
        self,
        tokenizer: Any,
        vocab_size: int,
        model_eos_token_ids: Optional[List[int]] = None,
        any_whitespace: bool = True,
        whitespace_pattern: Optional[str] = None,
        language: str = "python",
        enabled_domains: Optional[Set[str]] = None,
        max_rollback_tokens: int = 200,
    ) -> None: ...
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenizer` | `Any` | required | Model tokenizer |
| `vocab_size` | `int` | required | Vocabulary size |
| `model_eos_token_ids` | `Optional[List[int]]` | `None` | EOS token IDs |
| `any_whitespace` | `bool` | `True` | Allow flexible JSON whitespace |
| `whitespace_pattern` | `Optional[str]` | `None` | Custom whitespace regex |
| `language` | `str` | `"python"` | Target programming language |
| `enabled_domains` | `Optional[Set[str]]` | all | Domains to enable |
| `max_rollback_tokens` | `int` | `200` | Maximum rollback depth |

#### Methods

```python
def dispatch_json(self, key_string: str) -> Optional[AnankeGrammar]:
    """Create grammar from JSON schema.

    Args:
        key_string: JSON schema string

    Returns:
        AnankeGrammar or INVALID_GRAMMAR_OBJ on error
    """

def dispatch_regex(self, key_string: str) -> Optional[AnankeGrammar]:
    """Create grammar from regex pattern.

    Args:
        key_string: Regex pattern

    Returns:
        AnankeGrammar or INVALID_GRAMMAR_OBJ on error
    """

def dispatch_ebnf(self, key_string: str) -> Optional[AnankeGrammar]:
    """Create grammar from EBNF specification.

    Args:
        key_string: EBNF grammar string

    Returns:
        AnankeGrammar or INVALID_GRAMMAR_OBJ on error
    """

def dispatch_structural_tag(self, key_string: str) -> Optional[AnankeGrammar]:
    """Create grammar from structural tag.

    Args:
        key_string: Structural tag JSON

    Returns:
        AnankeGrammar or INVALID_GRAMMAR_OBJ on error
    """
```

#### Factory Function

```python
def create_ananke_backend(
    server_args: Any,
    tokenizer: Any,
    vocab_size: int,
    eos_token_ids: Optional[Set[int]] = None,
) -> AnankeBackend:
    """Factory function for SGLang integration.

    Registered with grammar backend registry.
    Enable via --grammar-backend=ananke
    """
```

---

## Grammar Classes

### AnankeGrammar

**Module:** `sglang.srt.constrained.ananke.backend.grammar`

Multi-domain constrained decoding grammar object.

```python
class AnankeGrammar(BaseGrammarObject):
    """Combines syntax, types, imports, control flow, and semantic constraints."""

    def __init__(
        self,
        syntax_grammar: Optional[GuidanceGrammar],
        domains: Dict[str, ConstraintDomain],
        constraint: UnifiedConstraint = UNIFIED_TOP,
        vocab_size: int = 0,
        device: str = "cuda",
        tokenizer: Optional[Any] = None,
        language: str = "python",
        max_rollback_tokens: int = 200,
        checkpoint_interval: int = 1,
        mask_pool_size: int = 8,
    ) -> None: ...
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `syntax_grammar` | `Optional[GuidanceGrammar]` | required | Wrapped llguidance grammar |
| `domains` | `Dict[str, ConstraintDomain]` | required | Constraint domains |
| `constraint` | `UnifiedConstraint` | `UNIFIED_TOP` | Initial constraint |
| `vocab_size` | `int` | `0` | Vocabulary size |
| `device` | `str` | `"cuda"` | PyTorch device |
| `tokenizer` | `Optional[Any]` | `None` | Tokenizer for decode |
| `language` | `str` | `"python"` | Programming language |
| `max_rollback_tokens` | `int` | `200` | Maximum rollback depth |
| `checkpoint_interval` | `int` | `1` | Checkpoint every N tokens |
| `mask_pool_size` | `int` | `8` | Pre-allocated mask tensors |

#### Methods

```python
def accept_token(self, token: int) -> None:
    """Accept generated token, update all constraints.

    Args:
        token: Generated token ID
    """

def rollback(self, k: int) -> None:
    """Roll back k tokens.

    Args:
        k: Number of tokens to roll back
    """

def is_terminated(self) -> bool:
    """Check if grammar reached terminal state.

    Returns:
        True if finished (constraint BOTTOM or EOS reached)
    """

def fill_vocab_mask(
    self,
    vocab_mask: torch.Tensor,
    idx: int,
    use_lazy_evaluation: bool = True,
) -> None:
    """Fill vocabulary mask with valid tokens.

    Args:
        vocab_mask: Bitmask tensor [batch_size, mask_size]
        idx: Batch index
        use_lazy_evaluation: Use budget-limited evaluation
    """

def get_cache_stats(self) -> Dict[str, int]:
    """Get cache performance statistics.

    Returns:
        Dictionary with cache_size and domains_cached
    """

def log_cache_summary(self) -> None:
    """Log cache performance summary."""
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `syntax_grammar` | `Optional[GuidanceGrammar]` | Wrapped syntax grammar |
| `constraint` | `UnifiedConstraint` | Current unified constraint |
| `domains` | `Dict[str, ConstraintDomain]` | Active domains |
| `context` | `GenerationContext` | Generation context |
| `checkpoint_manager` | `CheckpointManager` | Checkpoint manager |
| `finished` | `bool` | Terminal state flag |

---

## Constraint Specification

**Module:** `sglang.srt.constrained.ananke.spec.constraint_spec`

The Constraint Specification system enables rich, context-aware constrained generation.

> See [Constraint Specification Guide](./CONSTRAINT_SPEC.md) for detailed usage.

### ConstraintSpec

```python
@dataclass
class ConstraintSpec:
    """Rich constraint specification for context-aware generation."""

    # Version
    version: str = "1.0"

    # Core Syntax Constraint (one required)
    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    structural_tag: Optional[str] = None

    # Language Configuration
    language: Optional[str] = None
    language_detection: LanguageDetection = LanguageDetection.AUTO
    language_stack: List[LanguageFrame] = field(default_factory=list)

    # Type Context
    type_bindings: List[TypeBinding] = field(default_factory=list)
    function_signatures: List[FunctionSignature] = field(default_factory=list)
    class_definitions: List[ClassDefinition] = field(default_factory=list)
    expected_type: Optional[str] = None
    type_aliases: Dict[str, str] = field(default_factory=dict)

    # Import Context
    imports: List[ImportBinding] = field(default_factory=list)
    available_modules: Set[str] = field(default_factory=set)
    forbidden_imports: Set[str] = field(default_factory=set)
    module_stubs: Dict[str, ModuleStub] = field(default_factory=dict)

    # Control Flow Context
    control_flow: Optional[ControlFlowContext] = None

    # Semantic Constraints
    semantic_constraints: List[SemanticConstraint] = field(default_factory=list)

    # Domain Configuration
    enabled_domains: Optional[Set[str]] = None
    disabled_domains: Optional[Set[str]] = None
    domain_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Cache Control
    cache_scope: CacheScope = CacheScope.SYNTAX_ONLY
    context_hash: Optional[str] = None
```

#### Methods

```python
def to_dict(self) -> Dict[str, Any]:
    """Serialize to dictionary."""

def to_json(self) -> str:
    """Serialize to JSON string."""

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "ConstraintSpec":
    """Deserialize from dictionary."""

@classmethod
def from_json(cls, json_str: str) -> "ConstraintSpec":
    """Deserialize from JSON string."""

@classmethod
def from_legacy(
    cls,
    json_schema: Optional[str] = None,
    regex: Optional[str] = None,
    ebnf: Optional[str] = None,
    structural_tag: Optional[str] = None,
) -> "ConstraintSpec":
    """Create from legacy parameters."""

def compute_cache_key(self) -> str:
    """Compute cache key based on cache_scope."""
```

### Supporting Types

```python
class LanguageDetection(Enum):
    """Language detection strategy."""
    AUTO = "auto"          # Tree-sitter based detection
    EXPLICIT = "explicit"  # Use language field only
    STACK = "stack"        # Use language_stack for polyglot

class CacheScope(Enum):
    """What's included in cache key."""
    SYNTAX_ONLY = "syntax_only"          # Maximum reuse (default)
    SYNTAX_AND_LANG = "syntax_and_lang"  # Include language
    FULL_CONTEXT = "full_context"        # Include all context

@dataclass(frozen=True)
class TypeBinding:
    """Variable-to-type binding."""
    name: str
    type_expr: str
    scope: Optional[str] = None
    mutable: bool = True
    origin: Optional[str] = None

@dataclass(frozen=True)
class FunctionSignature:
    """Function type signature."""
    name: str
    params: Tuple[TypeBinding, ...]
    return_type: str
    type_params: Tuple[str, ...] = ()
    decorators: Tuple[str, ...] = ()
    is_async: bool = False
    is_generator: bool = False

@dataclass(frozen=True)
class ImportBinding:
    """Import statement representation."""
    module: str
    name: Optional[str] = None
    alias: Optional[str] = None
    is_wildcard: bool = False

@dataclass(frozen=True)
class ControlFlowContext:
    """Control flow context at generation point."""
    function_name: Optional[str] = None
    expected_return_type: Optional[str] = None
    loop_depth: int = 0
    loop_variables: Tuple[str, ...] = ()
    in_try_block: bool = False
    exception_types: Tuple[str, ...] = ()
    in_async_context: bool = False
    in_generator: bool = False
    reachable: bool = True

@dataclass(frozen=True)
class SemanticConstraint:
    """Semantic constraint for SMT-based checking."""
    kind: str  # "precondition", "postcondition", "invariant", "assertion"
    expression: str
    scope: Optional[str] = None
    variables: Tuple[str, ...] = ()
```

---

## Core Constraint System

### Constraint (Abstract Base)

**Module:** `sglang.srt.constrained.ananke.core.constraint`

```python
class Constraint(ABC, Generic[C]):
    """Abstract base for bounded meet-semilattice constraints.

    Semilattice Laws:
    1. Identity: c ⊓ ⊤ = c
    2. Annihilation: c ⊓ ⊥ = ⊥
    3. Idempotence: c ⊓ c = c
    4. Commutativity: c₁ ⊓ c₂ = c₂ ⊓ c₁
    5. Associativity: (c₁ ⊓ c₂) ⊓ c₃ = c₁ ⊓ (c₂ ⊓ c₃)
    """

    @abstractmethod
    def meet(self, other: C) -> C:
        """Compute meet (conjunction) of constraints."""

    @abstractmethod
    def is_top(self) -> bool:
        """Check if this is TOP (unconstrained)."""

    @abstractmethod
    def is_bottom(self) -> bool:
        """Check if this is BOTTOM (unsatisfiable)."""

    @abstractmethod
    def satisfiability(self) -> Satisfiability:
        """Determine satisfiability status."""

    def __and__(self, other: C) -> C:
        """Operator alias: c1 & c2 == c1.meet(c2)"""

    def is_satisfiable(self) -> bool:
        """True if definitely satisfiable."""

    def is_unsatisfiable(self) -> bool:
        """True if definitely unsatisfiable."""
```

### Satisfiability

```python
class Satisfiability(Enum):
    """Constraint satisfiability status."""

    SAT = auto()      # Satisfiable (at least one solution)
    UNSAT = auto()    # Unsatisfiable (no solution)
    UNKNOWN = auto()  # Undecidable or timeout

    def __and__(self, other: Satisfiability) -> Satisfiability:
        """Conservative AND: SAT & UNSAT = UNSAT"""

    def __or__(self, other: Satisfiability) -> Satisfiability:
        """Conservative OR: UNSAT | SAT = SAT"""
```

### Singleton Constraints

```python
# Trivial constraint (always satisfied)
TOP: TopConstraint

# Absurd constraint (never satisfied)
BOTTOM: BottomConstraint
```

### UnifiedConstraint

**Module:** `sglang.srt.constrained.ananke.core.unified`

```python
@dataclass(frozen=True, slots=True)
class UnifiedConstraint(Constraint["UnifiedConstraint"]):
    """Product type combining all domain constraints."""

    syntax: Constraint = TOP
    types: Constraint = TOP
    imports: Constraint = TOP
    controlflow: Constraint = TOP
    semantics: Constraint = TOP

    def with_syntax(self, syntax: Constraint) -> UnifiedConstraint: ...
    def with_types(self, types: Constraint) -> UnifiedConstraint: ...
    def with_imports(self, imports: Constraint) -> UnifiedConstraint: ...
    def with_controlflow(self, controlflow: Constraint) -> UnifiedConstraint: ...
    def with_semantics(self, semantics: Constraint) -> UnifiedConstraint: ...

# Singleton instances
UNIFIED_TOP: UnifiedConstraint    # All domains unconstrained
UNIFIED_BOTTOM: UnifiedConstraint  # All domains unsatisfiable
```

### Verification Utility

```python
def verify_semilattice_laws(
    c1: Constraint,
    c2: Constraint,
    c3: Constraint,
) -> bool:
    """Verify all semilattice laws hold for three constraints.

    Returns:
        True if all laws verified

    Raises:
        AssertionError: If any law is violated
    """
```

---

## Domain Constraints

### ConstraintDomain (Interface)

**Module:** `sglang.srt.constrained.ananke.core.domain`

```python
class ConstraintDomain(ABC, Generic[C]):
    """Abstract base for constraint domains."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique domain name."""

    @property
    @abstractmethod
    def top(self) -> C:
        """Unconstrained element."""

    @property
    @abstractmethod
    def bottom(self) -> C:
        """Unsatisfiable element."""

    @abstractmethod
    def token_mask(
        self,
        constraint: C,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Compute valid token mask.

        Returns:
            Boolean tensor (vocab_size,)
        """

    @abstractmethod
    def observe_token(
        self,
        constraint: C,
        token_id: int,
        context: GenerationContext,
    ) -> C:
        """Update constraint after token generation."""

    @abstractmethod
    def checkpoint(self) -> Checkpoint:
        """Create state checkpoint."""

    @abstractmethod
    def restore(self, checkpoint: Checkpoint) -> None:
        """Restore from checkpoint."""
```

### GenerationContext

```python
@dataclass
class GenerationContext:
    """Context during token generation."""

    generated_text: str = ""
    generated_tokens: List[int] = field(default_factory=list)
    position: int = 0
    vocab_size: int = 0
    device: str = "cuda"
    language: str = "python"
    tokenizer: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    mask_pool: Optional[MaskPool] = None

    def extend(self, token_id: int, token_text: str) -> GenerationContext:
        """Create extended context (immutable)."""

    def acquire_mask(self, fill_value: bool = True) -> Tuple[torch.Tensor, int]:
        """Acquire mask from pool."""

    def release_mask(self, handle: int) -> None:
        """Release mask to pool."""
```

### TypeDomain

**Module:** `sglang.srt.constrained.ananke.domains.types.domain`

```python
class TypeDomain(ConstraintDomain[TypeConstraint]):
    """Incremental bidirectional type checking domain."""

    def __init__(self, language: str = "python") -> None: ...

    @property
    def name(self) -> str:  # "types"
```

### ImportDomain

**Module:** `sglang.srt.constrained.ananke.domains.imports.domain`

```python
class ImportDomain(ConstraintDomain[ImportConstraint]):
    """Module/package constraint tracking domain."""

    def __init__(self, language: str = "python") -> None: ...

    @property
    def name(self) -> str:  # "imports"
```

### ControlFlowDomain

**Module:** `sglang.srt.constrained.ananke.domains.controlflow.domain`

```python
class ControlFlowDomain(ConstraintDomain[ControlFlowConstraint]):
    """CFG-based reachability analysis domain."""

    def __init__(self, language: str = "python") -> None: ...

    @property
    def name(self) -> str:  # "controlflow"
```

### SemanticDomain

**Module:** `sglang.srt.constrained.ananke.domains.semantics.domain`

```python
class SemanticDomain(ConstraintDomain[SemanticConstraint]):
    """SMT-based semantic constraint domain."""

    def __init__(self, language: str = "python") -> None: ...

    @property
    def name(self) -> str:  # "semantics"
```

---

## Checkpointing

### Checkpoint

**Module:** `sglang.srt.constrained.ananke.core.checkpoint`

```python
@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Immutable checkpoint for single domain state."""

    domain_name: str
    state: Dict[str, Any]
    position: int = 0
    constraint_hash: int = 0

    def validate(
        self,
        current_position: int,
        current_constraint: Constraint,
    ) -> bool:
        """Validate checkpoint compatibility."""
```

### UnifiedCheckpoint

```python
@dataclass(slots=True)
class UnifiedCheckpoint:
    """Combined checkpoint for all domains."""

    position: int
    unified_constraint: UnifiedConstraint
    domain_checkpoints: Dict[str, Checkpoint]
    context_snapshot: Optional[Dict[str, Any]] = None

    def get_domain_checkpoint(self, domain_name: str) -> Optional[Checkpoint]: ...
```

### CheckpointManager

```python
class CheckpointManager:
    """Bounded checkpoint history manager."""

    def __init__(
        self,
        max_checkpoints: int = 200,
        checkpoint_interval: int = 1,
    ) -> None: ...

    @property
    def checkpoint_count(self) -> int: ...

    @property
    def oldest_position(self) -> Optional[int]: ...

    @property
    def newest_position(self) -> Optional[int]: ...

    def create_checkpoint(
        self,
        position: int,
        unified_constraint: UnifiedConstraint,
        domain_checkpoints: Dict[str, Checkpoint],
        context_snapshot: Optional[Dict[str, Any]] = None,
    ) -> UnifiedCheckpoint: ...

    def get_checkpoint_at(self, position: int) -> Optional[UnifiedCheckpoint]: ...

    def get_checkpoint_before(self, position: int) -> Optional[UnifiedCheckpoint]: ...

    def rollback_to(self, position: int) -> Optional[UnifiedCheckpoint]: ...

    def clear(self) -> None: ...
```

---

## Typed Holes

### Hole

**Module:** `sglang.srt.constrained.ananke.holes.hole`

```python
class HoleState(Enum):
    """Hole state in generation."""
    EMPTY = auto()      # Not yet filled
    PARTIAL = auto()    # Partially filled
    FILLED = auto()     # Completely filled
    VALIDATED = auto()  # Filled and type-checked

@dataclass
class Hole:
    """A typed hole in generated code."""

    name: str
    expected_type: Optional[Type] = None
    environment: TypeEnvironment = field(default_factory=TypeEnvironment)
    constraint: Constraint = TOP
    state: HoleState = HoleState.EMPTY
    fill: Optional[str] = None
    children: List["Hole"] = field(default_factory=list)

    @property
    def is_empty(self) -> bool: ...

    @property
    def is_filled(self) -> bool: ...

    @property
    def is_valid(self) -> bool: ...

    @property
    def is_nested(self) -> bool: ...

    def with_fill(self, fill: str) -> Hole: ...

    def with_constraint(self, constraint: Constraint) -> Hole: ...
```

### TypeEnvironment

```python
class TypeEnvironment:
    """Immutable type variable bindings."""

    def bind(self, name: str, type_: Type) -> TypeEnvironment: ...

    def lookup(self, name: str) -> Optional[Type]: ...

    def extend(self, bindings: Dict[str, Type]) -> TypeEnvironment: ...

    def all_bindings(self) -> Dict[str, Type]: ...
```

---

## Performance Utilities

### MaskPool

**Module:** `sglang.srt.constrained.ananke.core.domain`

```python
class MaskPool:
    """Pre-allocated tensor pool."""

    def __init__(
        self,
        vocab_size: int,
        device: str,
        pool_size: int = 8,
    ) -> None: ...

    def acquire(self, fill_value: bool = True) -> Tuple[torch.Tensor, int]: ...

    def release(self, handle: int) -> None: ...

    @property
    def available_count(self) -> int: ...

    @property
    def pool_size(self) -> int: ...
```

### MaskCache

**Module:** `sglang.srt.constrained.ananke.masks.cache`

```python
class MaskCache:
    """LRU cache for token masks."""

    def __init__(
        self,
        max_size: int = 1000,
        max_age_seconds: float = 60.0,
    ) -> None: ...

    def get(self, key: CacheKey) -> Optional[torch.Tensor]: ...

    def put(
        self,
        key: CacheKey,
        mask: torch.Tensor,
        compute_time_ns: int = 0,
    ) -> None: ...

    def invalidate(self, key: CacheKey) -> bool: ...

    def invalidate_domain(self, domain: str) -> int: ...

    def invalidate_position(self, position: int) -> int: ...

    def clear(self) -> None: ...

    @property
    def stats(self) -> CacheStats: ...

@dataclass
class CacheStats:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_compute_time_saved_ns: int = 0

    @property
    def hit_rate(self) -> float: ...

    @property
    def hit_rate_percent(self) -> float: ...

    @property
    def total_requests(self) -> int: ...

    @property
    def time_saved_ms(self) -> float: ...

    def summary(self) -> str: ...
```

### EvaluationBudget

**Module:** `sglang.srt.constrained.ananke.masks.lazy`

```python
@dataclass
class EvaluationBudget:
    """Budget for lazy constraint evaluation."""

    max_time_ns: int = 2_000_000  # 2ms default
    max_domains: int = 5
    min_selectivity: float = 0.99
```

---

## Type System

### Type Hierarchy

**Module:** `sglang.srt.constrained.ananke.domains.types.types`

```python
class Type(ABC):
    """Abstract type base."""

    def free_type_vars(self) -> FrozenSet[str]: ...
    def substitute(self, substitution: Dict[str, Type]) -> Type: ...

# Primitive types
INT = PrimitiveType("int")
STR = PrimitiveType("str")
BOOL = PrimitiveType("bool")
FLOAT = PrimitiveType("float")
NONE = PrimitiveType("None")

# Special types
ANY: AnyType      # Compatible with everything
NEVER: NeverType  # No values

# Compound types
@dataclass(frozen=True, slots=True)
class FunctionType(Type):
    parameter_type: Type
    return_type: Type

@dataclass(frozen=True, slots=True)
class ListType(Type):
    element_type: Type

@dataclass(frozen=True, slots=True)
class DictType(Type):
    key_type: Type
    value_type: Type

@dataclass(frozen=True, slots=True)
class TupleType(Type):
    element_types: Tuple[Type, ...]
```

### Unification

```python
@dataclass
class Substitution:
    """Type variable substitution."""
    bindings: Dict[str, Type]

EMPTY_SUBSTITUTION = Substitution({})

def unify(type1: Type, type2: Type) -> Optional[Substitution]:
    """Attempt to unify two types.

    Returns:
        Substitution if unifiable, None otherwise
    """
```

---

## Configuration Reference

### Server CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--grammar-backend` | `str` | `"llguidance"` | Grammar backend to use |
| `--ananke-language` | `str` | `"python"` | Target programming language |
| `--ananke-max-rollback-tokens` | `int` | `200` | Maximum rollback depth |
| `--constrained-json-disable-any-whitespace` | `flag` | `False` | Strict JSON whitespace |
| `--constrained-json-whitespace-pattern` | `str` | `None` | Custom whitespace regex |

### Supported Languages

| Language | Type System | Import Resolution | Status |
|----------|-------------|-------------------|--------|
| Python | Full | pip/stdlib | Complete |
| TypeScript | Full | npm/stdlib | Complete |
| Go | Full | go modules | Complete |
| Rust | Full | cargo | Complete |
| Kotlin | Full | maven/gradle | Complete |
| Swift | Full | SPM | Complete |
| Zig | Full | build.zig | Complete |

### Domain Enable/Disable

```python
# Enable all domains (default)
backend = AnankeBackend(tokenizer, vocab_size)

# Enable specific domains
backend = AnankeBackend(
    tokenizer,
    vocab_size,
    enabled_domains={"syntax", "types", "imports"},  # Disable controlflow, semantics
)
```

### Performance Tuning

```python
# Sparse checkpointing (10x fewer checkpoints)
grammar = AnankeGrammar(
    ...,
    checkpoint_interval=10,  # Default: 1
)

# Disable mask pooling
grammar = AnankeGrammar(
    ...,
    mask_pool_size=0,  # Default: 8
)

# Custom evaluation budget
from sglang.srt.constrained.ananke.masks.lazy import EvaluationBudget

budget = EvaluationBudget(
    max_time_ns=1_000_000,  # 1ms (default: 2ms)
    max_domains=3,          # Skip slow domains
    min_selectivity=0.90,   # Stop early if selective
)
```

---

## Error Handling

### Constraint Errors

Ananke signals errors through constraint satisfiability:

```python
# Check if constraint is unsatisfiable
if grammar.constraint.satisfiability() == Satisfiability.UNSAT:
    print("Generation cannot satisfy all constraints")
    # grammar.finished will be True

# Check specific domain
if grammar.constraint.types.is_bottom():
    print("Type error detected")
```

### Common Error Conditions

| Condition | Detection | Behavior |
|-----------|-----------|----------|
| Type mismatch | `TypeConstraint.is_bottom()` | Grammar terminates |
| Missing import | `ImportConstraint.satisfiability() == UNSAT` | Blocks import tokens |
| Unreachable code | `ControlFlowConstraint.is_bottom()` | Grammar terminates |
| SMT contradiction | `SemanticConstraint.satisfiability() == UNSAT` | Grammar terminates |
| Grammar invalid | `dispatch_*() returns INVALID_GRAMMAR_OBJ` | Request fails |

### Debug Logging

```python
import logging

# Enable debug logging
logging.getLogger("sglang.srt.constrained.ananke").setLevel(logging.DEBUG)

# Log cache performance
grammar.log_cache_summary()

# Get detailed stats
stats = grammar.get_cache_stats()
print(f"Cache size: {stats['cache_size']}")
```

---

## Performance Targets

| Component | Target Latency | Notes |
|-----------|---------------|-------|
| Syntax domain (llguidance) | ~50μs/token | Delegated to llguidance |
| Type domain | <500μs/token | Incremental type checking |
| Import domain | <100μs/token | Import statement detection |
| Control flow domain | <200μs/token | CFG analysis |
| Semantics domain | <1ms/token | Z3 dependent |
| Mask fusion (SIMD) | ~10μs | With Zig native library |
| **Total per-token** | <2-3ms | Real-time performance |

---

## Version Information

- Ananke: 0.1.0
- Requires SGLang with grammar backend support
- Optional: Z3 for semantic constraint solving
- Optional: Zig for native SIMD acceleration
