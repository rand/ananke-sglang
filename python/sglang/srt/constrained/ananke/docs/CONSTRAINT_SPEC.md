# Rich Constraint Specification Guide

This document describes Ananke's Rich Constraint Specification system for passing type environments, import contexts, control flow information, and semantic constraints to the constrained generation system.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [ConstraintSpec Structure](#constraintspec-structure)
4. [Type Context](#type-context)
5. [Import Context](#import-context)
6. [Control Flow Context](#control-flow-context)
7. [Semantic Constraints](#semantic-constraints)
8. [Language Configuration](#language-configuration)
9. [Cache Control](#cache-control)
10. [Input Formats](#input-formats)
11. [API Integration](#api-integration)

---

## Overview

The Constraint Specification system enables rich, context-aware constrained generation by allowing callers to provide:

- **Type Bindings**: Variable-to-type mappings (e.g., `x: int`, `users: List[User]`)
- **Function Signatures**: Available functions with their type signatures
- **Class Definitions**: Class types with methods and attributes
- **Import Context**: Available modules, forbidden imports, type stubs
- **Control Flow**: Function context, loop depth, reachability
- **Semantic Constraints**: SMT formulas (preconditions, postconditions, invariants)

```
┌─────────────────────────────────────────────────────────────────┐
│                    ConstraintSpec                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Type Context │  │Import Context│  │ Semantic Constraints │  │
│  │ - bindings   │  │ - imports    │  │ - preconditions      │  │
│  │ - functions  │  │ - modules    │  │ - postconditions     │  │
│  │ - classes    │  │ - forbidden  │  │ - invariants         │  │
│  │ - expected   │  │ - stubs      │  │                      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         └─────────────────┼─────────────────────┘              │
│                           ▼                                     │
│              Domain Context Seeding                             │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌───────────┐       │
│  │TypeDomain│ │ImportDom │ │CtrlFlowDom │ │SemanticDom│       │
│  └──────────┘ └──────────┘ └────────────┘ └───────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Basic Usage

```python
from sglang.srt.constrained.ananke.spec.constraint_spec import (
    ConstraintSpec,
    TypeBinding,
    FunctionSignature,
    ImportBinding,
)

# Create a constraint specification with type context
spec = ConstraintSpec(
    # Core syntax constraint
    json_schema='{"type": "object", "properties": {"result": {"type": "string"}}}',

    # Language configuration
    language="python",

    # Type context
    type_bindings=[
        TypeBinding(name="x", type_expr="int"),
        TypeBinding(name="users", type_expr="List[User]"),
    ],
    expected_type="bool",

    # Import context
    imports=[
        ImportBinding(module="typing", name="List"),
        ImportBinding(module="models", name="User"),
    ],
)
```

### Using with AnankeBackend

```python
from sglang.srt.constrained.ananke.backend.backend import AnankeBackend

backend = AnankeBackend(tokenizer, vocab_size, language="python")

# Create grammar using constraint spec
grammar = backend.dispatch_with_spec(spec)

# Use grammar for generation
grammar.accept_token(token_id)
grammar.fill_vocab_mask(mask, idx)
```

---

## ConstraintSpec Structure

### Core Fields

```python
@dataclass
class ConstraintSpec:
    # Version
    version: str = "1.0"

    # Core Syntax Constraint (one required)
    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    structural_tag: Optional[str] = None

    # Language
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

    # Control Flow
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

---

## Type Context

### TypeBinding

Bind variables to types in the generation context:

```python
from sglang.srt.constrained.ananke.spec.constraint_spec import TypeBinding

# Simple bindings
bindings = [
    TypeBinding(name="x", type_expr="int"),
    TypeBinding(name="name", type_expr="str"),
    TypeBinding(name="items", type_expr="List[int]"),
]

# With scope
bindings = [
    TypeBinding(name="self", type_expr="MyClass", scope="parameter"),
    TypeBinding(name="count", type_expr="int", scope="local"),
    TypeBinding(name="CONFIG", type_expr="Dict[str, Any]", scope="global"),
]

# Mutable/immutable
bindings = [
    TypeBinding(name="data", type_expr="List[str]", mutable=True),
    TypeBinding(name="MAX_SIZE", type_expr="int", mutable=False),
]
```

### FunctionSignature

Register available functions:

```python
from sglang.srt.constrained.ananke.spec.constraint_spec import FunctionSignature

signatures = [
    FunctionSignature(
        name="process",
        params=(
            TypeBinding(name="items", type_expr="List[T]"),
            TypeBinding(name="key", type_expr="Callable[[T], int]"),
        ),
        return_type="List[T]",
        type_params=("T",),
    ),
    FunctionSignature(
        name="fetch_data",
        params=(TypeBinding(name="url", type_expr="str"),),
        return_type="Optional[Dict[str, Any]]",
        is_async=True,
    ),
]
```

### ClassDefinition

Register class types:

```python
from sglang.srt.constrained.ananke.spec.constraint_spec import ClassDefinition

classes = [
    ClassDefinition(
        name="User",
        bases=("BaseModel",),
        instance_vars=(
            TypeBinding(name="id", type_expr="int"),
            TypeBinding(name="name", type_expr="str"),
            TypeBinding(name="email", type_expr="Optional[str]"),
        ),
        methods=(
            FunctionSignature(
                name="validate",
                params=(),
                return_type="bool",
            ),
        ),
    ),
]
```

### Expected Type

Set the expected return type for the generated code:

```python
spec = ConstraintSpec(
    json_schema="...",
    expected_type="int",  # Generation should produce an int
)
```

### Type Aliases

Define type aliases:

```python
spec = ConstraintSpec(
    json_schema="...",
    type_aliases={
        "UserId": "int",
        "UserDict": "Dict[str, User]",
        "Callback": "Callable[[int], bool]",
    },
)
```

---

## Import Context

### ImportBinding

Declare available imports:

```python
from sglang.srt.constrained.ananke.spec.constraint_spec import ImportBinding

imports = [
    # import module
    ImportBinding(module="os"),

    # from module import name
    ImportBinding(module="typing", name="List"),
    ImportBinding(module="typing", name="Dict"),

    # from module import name as alias
    ImportBinding(module="pandas", name="DataFrame", alias="pd"),

    # from module import *
    ImportBinding(module="constants", is_wildcard=True),
]
```

### Available Modules

Specify modules that can be imported:

```python
spec = ConstraintSpec(
    json_schema="...",
    available_modules={
        "typing", "collections", "itertools",
        "numpy", "pandas",  # Allow data science libs
    },
)
```

### Forbidden Imports

Block specific imports:

```python
spec = ConstraintSpec(
    json_schema="...",
    forbidden_imports={
        "os", "subprocess", "shutil",  # Block filesystem access
        "socket", "urllib",             # Block network access
    },
)
```

### Module Stubs

Provide type information for modules:

```python
from sglang.srt.constrained.ananke.spec.constraint_spec import ModuleStub

spec = ConstraintSpec(
    json_schema="...",
    module_stubs={
        "mylib": ModuleStub(
            module_name="mylib",
            exports={
                "process": "Callable[[List[int]], int]",
                "Config": "Type[Config]",
            },
            submodules=("mylib.utils", "mylib.core"),
        ),
    },
)
```

---

## Control Flow Context

### ControlFlowContext

Provide control flow information:

```python
from sglang.srt.constrained.ananke.spec.constraint_spec import ControlFlowContext

# Inside a function
spec = ConstraintSpec(
    json_schema="...",
    control_flow=ControlFlowContext(
        function_name="process_items",
        expected_return_type="int",
        in_async_context=False,
    ),
)

# Inside a loop
spec = ConstraintSpec(
    json_schema="...",
    control_flow=ControlFlowContext(
        loop_depth=2,
        loop_variables=("i", "j"),
    ),
)

# In try/except block
spec = ConstraintSpec(
    json_schema="...",
    control_flow=ControlFlowContext(
        in_try_block=True,
        exception_types=("ValueError", "TypeError"),
    ),
)

# Unreachable code detection
spec = ConstraintSpec(
    json_schema="...",
    control_flow=ControlFlowContext(
        reachable=True,  # Will detect unreachable code
        dominators=("if_check", "loop_entry"),
    ),
)
```

---

## Semantic Constraints

### SemanticConstraint

Add SMT-based semantic constraints:

```python
from sglang.srt.constrained.ananke.spec.constraint_spec import SemanticConstraint

spec = ConstraintSpec(
    json_schema="...",
    semantic_constraints=[
        # Precondition
        SemanticConstraint(
            kind="precondition",
            expression="n >= 0",
            variables=("n",),
        ),

        # Postcondition
        SemanticConstraint(
            kind="postcondition",
            expression="result >= 0",
            variables=("result",),
        ),

        # Invariant
        SemanticConstraint(
            kind="invariant",
            expression="count <= max_size",
            scope="MyClass",
            variables=("count", "max_size"),
        ),

        # Assertion
        SemanticConstraint(
            kind="assertion",
            expression="len(items) > 0",
            variables=("items",),
        ),
    ],
)
```

---

## Language Configuration

### Explicit Language

```python
spec = ConstraintSpec(
    json_schema="...",
    language="python",  # or "typescript", "go", "rust", etc.
)
```

### Auto-Detection

```python
from sglang.srt.constrained.ananke.spec.constraint_spec import LanguageDetection

spec = ConstraintSpec(
    json_schema="...",
    language_detection=LanguageDetection.AUTO,  # Detect from generated text
)
```

### Polyglot Support (Language Stack)

For generating code that contains multiple languages:

```python
from sglang.srt.constrained.ananke.spec.constraint_spec import (
    LanguageDetection,
    LanguageFrame,
)

# TypeScript with embedded Python in template literals
spec = ConstraintSpec(
    json_schema="...",
    language_detection=LanguageDetection.STACK,
    language_stack=[
        LanguageFrame(
            language="typescript",
            start_position=0,
        ),
        LanguageFrame(
            language="python",
            start_position=150,
            delimiter="'''",
            end_delimiter="'''",
        ),
    ],
)
```

---

## Cache Control

### Cache Scope

Control what's included in cache keys:

```python
from sglang.srt.constrained.ananke.spec.constraint_spec import CacheScope

# Maximum reuse (default) - only syntax constraint in key
spec = ConstraintSpec(
    json_schema="...",
    cache_scope=CacheScope.SYNTAX_ONLY,
)

# Include language in cache key
spec = ConstraintSpec(
    json_schema="...",
    language="python",
    cache_scope=CacheScope.SYNTAX_AND_LANG,
)

# Include all context in cache key
spec = ConstraintSpec(
    json_schema="...",
    type_bindings=[...],
    cache_scope=CacheScope.FULL_CONTEXT,
)
```

### Pre-computed Context Hash

For efficiency with large contexts:

```python
import hashlib

# Pre-compute hash
context_data = json.dumps({"types": [...], "imports": [...]})
context_hash = hashlib.sha256(context_data.encode()).hexdigest()[:16]

spec = ConstraintSpec(
    json_schema="...",
    context_hash=context_hash,  # Skip recomputation
    cache_scope=CacheScope.FULL_CONTEXT,
)
```

---

## Input Formats

### JSON Inline (Primary)

```python
spec_dict = {
    "version": "1.0",
    "json_schema": '{"type": "object"}',
    "language": "python",
    "type_bindings": [
        {"name": "x", "type_expr": "int"},
    ],
    "expected_type": "bool",
}

spec = ConstraintSpec.from_dict(spec_dict)
```

### Serialization

```python
# To dict
d = spec.to_dict()

# To JSON
json_str = spec.to_json()

# From JSON
spec = ConstraintSpec.from_json(json_str)
```

---

## API Integration

### Using with dispatch_with_spec

```python
from sglang.srt.constrained.ananke.backend.backend import AnankeBackend
from sglang.srt.constrained.ananke.spec.constraint_spec import ConstraintSpec

backend = AnankeBackend(tokenizer, vocab_size, language="python")

spec = ConstraintSpec(
    json_schema='{"type": "object"}',
    type_bindings=[
        TypeBinding(name="user", type_expr="User"),
    ],
    expected_type="Dict[str, Any]",
)

# Dispatch with full constraint spec
grammar = backend.dispatch_with_spec(spec)

# Use grammar normally
while not grammar.is_terminated():
    grammar.fill_vocab_mask(mask, idx)
    token = sample_from_mask(mask)
    grammar.accept_token(token)
```

### Context Injection

Cached grammars can have fresh context injected:

```python
# Get cached grammar
cached_grammar = cache.get(cache_key)

# Inject fresh context (without recompiling)
if cached_grammar and hasattr(cached_grammar, 'inject_context'):
    cached_grammar.inject_context(new_spec)
```

### Domain Seeding

The spec automatically seeds all enabled domains:

```python
# TypeDomain receives:
# - type_bindings → bind_variable()
# - function_signatures → register_function()
# - class_definitions → register_class()
# - expected_type → set_expected_type()
# - type_aliases → add_type_alias()

# ImportDomain receives:
# - imports → add_import()
# - available_modules → set_available_modules()
# - forbidden_imports → set_forbidden_imports()
# - module_stubs → add_module_stub()

# ControlFlowDomain receives:
# - control_flow → set_control_flow_context()

# SemanticDomain receives:
# - semantic_constraints → add_constraint()
```

---

## Examples

### Type-Safe Function Generation

```python
spec = ConstraintSpec(
    regex=r"def \w+\([^)]*\) -> \w+:\n    .*",
    language="python",
    type_bindings=[
        TypeBinding(name="items", type_expr="List[int]"),
    ],
    function_signatures=[
        FunctionSignature(
            name="sum",
            params=(TypeBinding(name="iterable", type_expr="Iterable[int]"),),
            return_type="int",
        ),
    ],
    expected_type="int",
)
```

### Import-Aware Code Generation

```python
spec = ConstraintSpec(
    ebnf="...",  # Python expression grammar
    language="python",
    imports=[
        ImportBinding(module="typing", name="List"),
        ImportBinding(module="numpy", alias="np"),
    ],
    available_modules={"typing", "numpy", "pandas"},
    forbidden_imports={"os", "subprocess"},
)
```

### Async Function with Semantic Constraints

```python
spec = ConstraintSpec(
    json_schema="...",
    language="python",
    control_flow=ControlFlowContext(
        function_name="fetch_data",
        in_async_context=True,
        expected_return_type="Optional[Dict[str, Any]]",
    ),
    semantic_constraints=[
        SemanticConstraint(
            kind="precondition",
            expression="url.startswith('https://')",
            variables=("url",),
        ),
        SemanticConstraint(
            kind="postcondition",
            expression="result is None or 'data' in result",
            variables=("result",),
        ),
    ],
)
```

---

## Best Practices

1. **Start with SYNTAX_ONLY cache scope** for maximum reuse
2. **Use explicit language** when you know the target language
3. **Provide type bindings** for variables that will be referenced
4. **Use available_modules** to whitelist safe imports
5. **Pre-compute context_hash** for large type environments
6. **Separate syntax constraints from semantic context** when possible

---

## See Also

- [Getting Started Guide](./GETTING_STARTED.md) - Basic Ananke usage
- [API Reference](./REFERENCE.md) - Complete API documentation
- [Architecture Overview](./ARCHITECTURE.md) - System design
