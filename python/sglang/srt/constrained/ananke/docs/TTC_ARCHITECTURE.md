# Ananke Test-Time Compute (TTC) Architecture

This document describes the test-time compute improvements to the Ananke constrained generation backend.

## Overview

Test-time compute (TTC) refers to techniques that improve generation quality by trading additional compute at inference time. Unlike training-time improvements, TTC techniques can be applied dynamically based on task complexity.

The Ananke TTC system provides:

1. **Adaptive Constraint Intensity** - Dynamically adjust constraint overhead based on task complexity
2. **Best-of-N Verification** - Generate multiple candidates and select the best
3. **Sudoku-Style Hole Filling** - CSP-based code generation with MCV heuristic and backtracking
4. **Trajectory Tracking** - Efficient trie-based state management for search

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Request Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│   │   Task Input    │───>│    Complexity   │───>│    Intensity    │        │
│   │   (prompt)      │    │    Assessor     │    │    Selection    │        │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│                                                         │                   │
│                                                         ▼                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    Generation Strategy Selection                     │  │
│   │                                                                       │  │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │  │
│   │  │ Per-Token    │  │ Best-of-N    │  │ Sudoku-Style Hole       │   │  │
│   │  │ Constraints  │  │ Verification │  │ Filling (CSP Search)    │   │  │
│   │  │              │  │              │  │                          │   │  │
│   │  │ (Low TTC)    │  │ (Medium TTC) │  │ (High TTC)              │   │  │
│   │  └──────────────┘  └──────────────┘  └──────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Adaptive Constraint Intensity (`adaptive/intensity.py`)

Dynamically assesses task complexity and selects appropriate constraint intensity.

```python
from adaptive.intensity import ConstraintIntensity, assess_complexity

# Assess complexity
intensity = assess_complexity(prompt="def add(x: int, y: int) -> int:", expected_tokens=50)

# Get enabled domains for intensity
domains = domains_for_intensity(intensity)
# Returns: {'syntax', 'types'} for STANDARD intensity
```

**Intensity Levels:**

| Level | Domains Enabled | Typical Latency | Use Case |
|-------|-----------------|-----------------|----------|
| NONE | None | ~0μs | Unconstrained generation |
| SYNTAX_ONLY | syntax | ~50μs | Simple completions |
| STANDARD | syntax, types | ~550μs | Function bodies |
| FULL | syntax, types, imports, controlflow | ~1.5ms | Complex code |
| EXHAUSTIVE | All + semantics | ~2.5ms | Critical paths |

**Assessment Factors:**
- Expected token count
- Code structure (class, function, block)
- Type annotation density
- Import complexity

### 2. Best-of-N Verification (`verification/`)

Generate multiple candidates and select the best using constraint verification.

```python
from verification.verifier import ConstraintVerifier
from verification.selector import BestOfNSelector, SelectionStrategy

# Create verifier and selector
verifier = ConstraintVerifier(language="python")
selector = BestOfNSelector(
    verifier=verifier,
    strategy=SelectionStrategy.BEST_SCORE,
)

# Verify and select
candidates = [
    "def add(x, y): return x + y",
    "def add(x: int, y: int) -> int: return x + y",  # Best
]
result = selector.select_best(candidates)
print(result.selected)  # The typed version
```

**Selection Strategies:**
- `BEST_SCORE`: Verify all, select highest score
- `FIRST_VALID`: Stop at first valid candidate
- `THRESHOLD`: Stop at first candidate above threshold
- `WEIGHTED`: Custom domain weights

**Verification Result:**
```python
@dataclass
class VerificationResult:
    candidate: str           # Code that was verified
    valid: bool              # True if all domains pass
    overall_score: float     # 0.0 to 1.0
    domain_scores: Dict[str, DomainScore]  # Per-domain scores
    latency_ns: int          # Verification time
```

### 3. Sudoku-Style Hole Filling (`search/`)

CSP-based code generation using typed holes and constraint propagation.

```python
from search.sudoku_filler import SudokuStyleHoleFiller, FillStrategy, HoledCode
from search.generators import TypeAwareFillGenerator
from holes.hole import Hole, HoleId

# Create holed code
hole_id = HoleId(name="value")
hole = Hole(id=hole_id, expected_type=INT)
code = HoledCode(
    template="x: int = ?default:value[0]",
    holes={hole_id: hole},
    hole_markers={hole_id: "?default:value[0]"},
)

# Create filler with type-aware generator
generator = TypeAwareFillGenerator()
filler = SudokuStyleHoleFiller(
    fill_generator=generator,
    strategy=FillStrategy.MCV,  # Most Constrained Variable
)

# Fill holes
result = filler.fill(code)
print(result.filled_code)  # "x: int = 0"
```

**Key Algorithms:**

1. **Most Constrained Variable (MCV) Heuristic**
   - Select hole with fewest valid fill options
   - Fail-fast: detect conflicts early
   - Based on Sudoku-solving techniques

2. **Constraint Propagation**
   - When a hole is filled, propagate constraints to dependent holes
   - Uses TypeConstraintInferencer for type relationship analysis
   - Reduces search space

3. **Backtracking**
   - Trie-based trajectory tracking for efficient rollback
   - O(1) checkpoint/restore operations
   - Configurable maximum backtracks

**Fill Strategies:**
- `MCV`: Most Constrained Variable (default)
- `MRV`: Minimum Remaining Values (alias for MCV)
- `LARGEST_DEGREE`: Most dependent holes
- `SEQUENTIAL`: Fill in declaration order
- `RANDOM`: Random selection (for diversity)

### 4. Trajectory Tracking (`search/trajectory.py`)

Efficient state management for search using trie-based trajectory tracking.

```python
from search.trajectory import TrajectoryTrie, create_trajectory_trie

# Create trajectory trie
trie, traj = create_trajectory_trie(initial_state=code)

# Extend trajectory
traj1 = traj.extend(hole_id, "value1", state=state1, score=0.9)

# Create checkpoint
checkpoint = trie.checkpoint(traj1)

# Extend further
traj2 = traj1.extend(hole2_id, "value2", state=state2, score=0.5)

# Backtrack to checkpoint
restored = trie.restore(checkpoint)

# Find best leaf
best = trie.best_leaf()
```

**Complexity:**
- Checkpoint: O(1)
- Restore: O(1)
- Extend: O(1)
- Best leaf: O(n) where n is number of leaves

## Integration

### Complete Pipeline Example

```python
from adaptive.intensity import assess_complexity, domains_for_intensity
from verification.verifier import ConstraintVerifier
from verification.selector import BestOfNSelector
from search.sudoku_filler import SudokuStyleHoleFiller, FillStrategy, HoledCode
from search.generators import TypeAwareFillGenerator, UnifiedConstraintChecker
from holes.hole import Hole, HoleId

# 1. Assess complexity
prompt = "def process(data: List[Dict]) -> Result:"
intensity = assess_complexity(prompt, expected_tokens=100)

# 2. Get enabled domains
domains = domains_for_intensity(intensity)

# 3. Create holed code
hole_id = HoleId(name="body")
hole = Hole(id=hole_id, expected_type=ANY)
code = HoledCode(
    template=prompt + "\n    return ?default:body[0]",
    holes={hole_id: hole},
    hole_markers={hole_id: "?default:body[0]"},
)

# 4. Generate multiple candidates
generator = TypeAwareFillGenerator()
checker = UnifiedConstraintChecker(enabled_domains=domains)
filler = SudokuStyleHoleFiller(
    fill_generator=generator,
    constraint_checker=checker,
)

candidates = []
for strategy in [FillStrategy.MCV, FillStrategy.SEQUENTIAL]:
    filler.strategy = strategy
    result = filler.fill(code)
    if result.success:
        candidates.append(result.filled_code)

# 5. Select best candidate
verifier = ConstraintVerifier(language="python", enabled_domains=domains)
selector = BestOfNSelector(verifier=verifier)
best = selector.select_best(candidates)

print(f"Selected: {best.selected}")
print(f"Score: {best.selected_result.overall_score}")
```

## Soundness Guarantees

All TTC techniques preserve Ananke's soundness invariants:

1. **Never block valid tokens**: Constraints are permissive (soundness > completeness)
2. **Semilattice laws**: Constraint meet operation is commutative and associative
3. **Monotonicity**: Constraint propagation never loosens constraints

Property-based tests in `tests/property/test_ttc_soundness.py` verify these invariants.

## Performance Characteristics

| Technique | Overhead | Quality Improvement | Best For |
|-----------|----------|---------------------|----------|
| Per-token constraints | ~500μs/token | High (syntax, types) | All code |
| Best-of-N (N=3) | ~3x generation | Medium (ranking) | Short completions |
| Sudoku filling | ~1-10ms total | High (structured code) | Multi-hole templates |
| Combined | Variable | Highest | Complex generation |

## References

1. **XGrammar**: Vocabulary partitioning for fast token classification
2. **DOMINO**: Speculative decoding with constraints
3. **Hazel**: Typed holes for gradual type checking
4. **GenCP**: LLM meets constraint propagation
5. **ROCODE**: Backtracking for code generation
6. **BoNBoN**: Best-of-N sampling for alignment
