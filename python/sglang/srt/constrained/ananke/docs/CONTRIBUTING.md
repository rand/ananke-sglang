# Contributing to Ananke

This guide covers how to set up a development environment, run tests, and extend Ananke with new features.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Running Tests](#running-tests)
4. [Adding a New Language](#adding-a-new-language)
5. [Extending Domains](#extending-domains)
6. [Code Style](#code-style)
7. [Submitting Changes](#submitting-changes)

---

## Development Setup

### Prerequisites

- Python 3.10+
- Zig 0.11+ (optional, for native SIMD)
- Z3 theorem prover (optional, for semantics domain)

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev]"

# Navigate to Ananke
cd python/sglang/srt/constrained/ananke
```

### Optional: Build Zig Native Library

```bash
cd zig
zig build -Doptimize=ReleaseFast
# Library will be at zig-out/lib/libananke_simd.so (or .dylib on macOS)
```

### Optional: Install Z3

```bash
# macOS
brew install z3

# Ubuntu/Debian
sudo apt-get install z3 libz3-dev

# pip (cross-platform)
pip install z3-solver
```

---

## Project Structure

```
ananke/
├── backend/           # SGLang integration
│   ├── backend.py     # AnankeBackend class
│   └── grammar.py     # AnankeGrammar class
├── core/              # Constraint algebra foundations
│   ├── constraint.py  # Base Constraint class
│   ├── domain.py      # ConstraintDomain interface
│   ├── unified.py     # UnifiedConstraint product type
│   └── checkpoint.py  # Checkpoint management
├── domains/           # Constraint domain implementations
│   ├── syntax/        # CFG-based syntax constraints
│   ├── types/         # Type checking domain
│   │   ├── languages/ # Language-specific type systems
│   │   ├── bidirectional/ # Type synthesis/analysis
│   │   └── incremental/   # Delta typing
│   ├── imports/       # Module resolution domain
│   │   └── resolvers/ # Language-specific resolvers
│   ├── controlflow/   # CFG construction and analysis
│   └── semantics/     # SMT-based constraints
├── holes/             # Typed holes system
├── masks/             # Mask computation and fusion
├── parsing/           # Incremental parsers
│   └── languages/     # Language-specific parsers
├── propagation/       # Cross-domain propagation
├── spec/              # Constraint specification
├── zig/               # Native SIMD acceleration
├── tests/             # Test suite
│   ├── unit/          # Unit tests
│   ├── integration/   # Integration tests
│   ├── property/      # Property-based tests
│   └── benchmark/     # Performance benchmarks
└── docs/              # Documentation (you are here)
```

---

## Running Tests

### All Tests

```bash
# From repository root
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
    python/sglang/srt/constrained/ananke/tests/ -v
```

### Unit Tests Only

```bash
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
    python/sglang/srt/constrained/ananke/tests/unit/ -v
```

### Integration Tests

```bash
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
    python/sglang/srt/constrained/ananke/tests/integration/ -v
```

### Property-Based Tests

```bash
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
    python/sglang/srt/constrained/ananke/tests/property/ -v
```

### Benchmarks

```bash
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
    python/sglang/srt/constrained/ananke/tests/benchmark/ -v -s
```

### Language-Specific Tests

```bash
# Type system tests for a specific language
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
    python/sglang/srt/constrained/ananke/tests/unit/test_go_types.py -v

# Control flow tests
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
    python/sglang/srt/constrained/ananke/tests/unit/test_controlflow*.py -v
```

---

## Adding a New Language

To add support for a new programming language, you need to implement:

### 1. Type System (`domains/types/languages/`)

Create `domains/types/languages/{lang}.py`:

```python
from ..constraint import Type, TypeConstraint
from ..base import LanguageTypeSystem

class MyLangTypeSystem(LanguageTypeSystem):
    """Type system for MyLang."""

    def __init__(self):
        super().__init__("mylang")

    def parse_type(self, type_str: str) -> Type:
        """Parse a type annotation string."""
        ...

    def is_assignable(self, source: Type, target: Type) -> bool:
        """Check if source type is assignable to target."""
        ...

    def format_type(self, typ: Type) -> str:
        """Format a type for display."""
        ...
```

### 2. Import Resolver (`domains/imports/resolvers/`)

Create `domains/imports/resolvers/{lang}.py`:

```python
from ..base import ImportResolver, ModuleInfo

class MyLangImportResolver(ImportResolver):
    """Import resolver for MyLang."""

    def __init__(self):
        super().__init__("mylang")
        self._stdlib_modules = self._load_stdlib()

    def resolve(self, module_name: str) -> Optional[ModuleInfo]:
        """Resolve a module name to module information."""
        ...

    def get_standard_library_modules(self) -> Set[str]:
        """Return set of standard library module names."""
        ...
```

### 3. Token Classifier (`core/token_classifier_{lang}.py`)

Create `core/token_classifier_{lang}.py`:

```python
from .token_classifier import TokenClassifier, TokenCategory

class MyLangTokenClassifier(TokenClassifier):
    """Token classifier for MyLang."""

    KEYWORDS = {"if", "else", "while", "for", ...}
    OPERATORS = {"+", "-", "*", "/", ...}
    BUILTINS = {"print", "len", ...}

    def classify(self, token: str) -> TokenCategory:
        """Classify a token string."""
        ...
```

### 4. Parser (`parsing/languages/`)

Create `parsing/languages/{lang}.py`:

```python
from ..base import IncrementalParser, ParseState

class MyLangParser(IncrementalParser):
    """Incremental parser for MyLang."""

    def __init__(self):
        super().__init__("mylang")

    def is_expression_start(self, token: str) -> bool:
        """Check if token can start an expression."""
        ...

    def detect_holes(self, partial_code: str) -> List[HoleInfo]:
        """Detect incomplete constructs in partial code."""
        ...
```

### 5. Control Flow Detection

Add detection method to `domains/controlflow/domain.py`:

```python
def _detect_mylang_control_flow(self, text: str) -> Optional[ControlFlowInfo]:
    """Detect MyLang control flow constructs."""
    ...
```

### 6. Tests

Create tests in `tests/unit/`:
- `test_mylang_types.py` - Type system tests
- `test_mylang_imports.py` - Import resolver tests
- `test_mylang_token_classifier.py` - Token classification tests

---

## Extending Domains

### Adding a New Constraint Domain

1. Create domain class implementing `ConstraintDomain`:

```python
from ananke.core.domain import ConstraintDomain
from ananke.core.constraint import Constraint

class MyDomain(ConstraintDomain[MyConstraint]):
    """My custom constraint domain."""

    @property
    def name(self) -> str:
        return "mydomain"

    @property
    def top(self) -> MyConstraint:
        return MY_TOP

    @property
    def bottom(self) -> MyConstraint:
        return MY_BOTTOM

    def token_mask(
        self,
        constraint: MyConstraint,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Compute token mask from constraint."""
        ...

    def observe_token(
        self,
        constraint: MyConstraint,
        token_id: int,
        context: GenerationContext,
    ) -> MyConstraint:
        """Update constraint after token generation."""
        ...

    def checkpoint(self) -> Checkpoint:
        """Save state for rollback."""
        ...

    def restore(self, checkpoint: Checkpoint) -> None:
        """Restore from checkpoint."""
        ...
```

2. Register domain in `backend/backend.py`:

```python
from mydomain import MyDomain

# In AnankeBackend.__init__
if "mydomain" in enabled_domains:
    self.domains["mydomain"] = MyDomain(language=language)
```

3. Add to `UnifiedConstraint` in `core/unified.py`:

```python
@dataclass(frozen=True, slots=True)
class UnifiedConstraint(Constraint["UnifiedConstraint"]):
    syntax: Constraint = TOP
    types: Constraint = TOP
    imports: Constraint = TOP
    controlflow: Constraint = TOP
    semantics: Constraint = TOP
    mydomain: Constraint = TOP  # Add new domain
```

---

## Code Style

### Python

- **Type hints**: All public functions must have type annotations
- **Docstrings**: Google-style docstrings for public classes/functions
- **Formatting**: Black with 88 character line length
- **Imports**: isort with SGLang configuration

### Zig

- **Style**: Follow Zig standard library conventions
- **Documentation**: Doc comments for public functions
- **Safety**: No undefined behavior, graceful error handling

### Testing

- **Coverage**: New features require tests
- **Property tests**: Use Hypothesis for invariant testing
- **Integration**: E2E tests for user-facing changes

---

## Submitting Changes

### Before Submitting

1. Run all tests and ensure they pass
2. Add tests for new functionality
3. Update documentation if needed
4. Run formatting tools:
   ```bash
   black python/sglang/srt/constrained/ananke/
   isort python/sglang/srt/constrained/ananke/
   ```

### Pull Request Guidelines

1. **Clear title**: Describe the change in present tense
2. **Description**: Explain what and why
3. **Tests**: Include test coverage
4. **Documentation**: Update docs if user-facing

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`

Examples:
```
feat(types): add Swift protocol conformance checking
fix(controlflow): handle unreachable code after return
docs: update language support matrix
```

---

## See Also

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System overview
- [ARCHITECTURE_DEEP_DIVE.md](./ARCHITECTURE_DEEP_DIVE.md) - Technical details
- [REFERENCE.md](./REFERENCE.md) - API reference
