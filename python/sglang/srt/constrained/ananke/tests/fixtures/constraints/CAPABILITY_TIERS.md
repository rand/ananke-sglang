# Ananke Constraint Capability Tiers

This document describes the capability tiers for Ananke constraint examples, explaining how different levels of constraint sophistication affect token masking behavior.

## Overview

Constraint examples in the Ananke test fixtures demonstrate progressively sophisticated constraint capabilities. Each tier builds on the previous, adding more context for the grammar system to make informed masking decisions.

## Tier 1: Syntax Constraints (Foundation)

**Description**: Basic pattern matching using one of the core syntax constraint types.

**Constraint Fields Used**:
- `json_schema`: JSON Schema for structured output
- `regex`: Regular expression pattern
- `ebnf`: Extended Backus-Naur Form grammar
- `structural_tag`: Predefined structural patterns

**Masking Behavior**: Masks tokens that would violate the syntactic pattern.

**Example**:
```python
ConstraintSpec(
    language="python",
    regex=r"^def\s+\w+\s*\([^)]*\)\s*->\s*\w+:",
)
```

**Use Cases**:
- Version string validation (semver)
- Identifier naming conventions
- Template literal patterns
- JSON/YAML structure enforcement

---

## Tier 2: Type Context

**Description**: Adds type-aware constraints beyond pure syntax.

**Constraint Fields Used**:
- `type_bindings`: Variable-to-type mappings in scope
- `function_signatures`: Function type signatures
- `class_definitions`: Class structure definitions
- `expected_type`: Expected return/expression type
- `type_aliases`: Custom type alias definitions

**Masking Behavior**: Masks tokens that would produce type-incompatible code.

**Example**:
```python
ConstraintSpec(
    language="typescript",
    regex=r"function\s+\w+<T>",
    type_bindings=[
        TypeBinding(name="items", type_expr="Array<T>", scope="parameter"),
    ],
    expected_type="T | undefined",
)
```

**Use Cases**:
- Generic type inference
- Return type enforcement
- Type-safe collection operations
- Interface implementation validation

---

## Tier 3: Import Context

**Description**: Adds module and import awareness.

**Constraint Fields Used**:
- `imports`: Required import bindings
- `available_modules`: Set of modules that can be imported
- `forbidden_imports`: Set of modules that cannot be used
- `module_stubs`: Type stubs for external modules

**Masking Behavior**: Masks tokens that would use unavailable or forbidden imports.

**Example**:
```python
ConstraintSpec(
    language="rust",
    imports=[
        ImportBinding(module="std::sync", name="Arc"),
        ImportBinding(module="std::sync", name="Mutex"),
    ],
    forbidden_imports={"std::rc::Rc"},  # Prevent non-thread-safe alternatives
)
```

**Use Cases**:
- Ensuring required dependencies
- Blocking insecure modules
- Enforcing import style (wildcard vs explicit)
- Cross-module type consistency

---

## Tier 4: Control Flow Context

**Description**: Adds awareness of control flow state at the generation point.

**Constraint Fields Used**:
- `control_flow.function_name`: Containing function
- `control_flow.function_signature`: Full function signature
- `control_flow.expected_return_type`: Expected return type
- `control_flow.loop_depth`: Nesting depth of loops
- `control_flow.loop_variables`: Variables from enclosing loops
- `control_flow.in_try_block`: Whether inside try block
- `control_flow.exception_types`: Exception types being caught
- `control_flow.in_async_context`: Whether in async context
- `control_flow.in_generator`: Whether in generator function
- `control_flow.reachable`: Whether code point is reachable

**Masking Behavior**: Masks tokens inappropriate for the control flow context.

**Example**:
```python
ConstraintSpec(
    language="python",
    control_flow=ControlFlowContext(
        function_name="handle_request",
        in_async_context=True,
        expected_return_type="Response",
    ),
)
```

**Use Cases**:
- Enforcing `await` in async functions
- Allowing `break`/`continue` only in loops
- Specific exception handling (not bare `except:`)
- Early return patterns with guards
- Exhaustive match/switch handling

---

## Tier 5: Semantic Constraints

**Description**: Adds logical predicates that must hold during generation.

**Constraint Fields Used**:
- `semantic_constraints`: List of `SemanticConstraint` objects

**SemanticConstraint Fields**:
- `kind`: One of `"precondition"`, `"postcondition"`, `"invariant"`, `"assertion"`, `"assume"`
- `expression`: Boolean expression in predicate form
- `scope`: Where the constraint applies
- `variables`: Free variables referenced

**Masking Behavior**: Masks tokens that would violate semantic invariants.

**Expression Vocabulary** (standardized predicates):
```python
# Mutation predicates
not_mutated(var)
immutable(var)

# Coverage predicates
exhaustive(match_expr)
all_variants_covered(enum_var)

# Control flow predicates
precedes(check, access)
followed_by_defer(acquire, release)
all_exits_cleanup(resource)

# Async predicates
all_awaited(async_calls)
has_terminal_operator(flow)

# Type predicates
implements_all(type, methods)
is_sendable(type)
pointer_receiver(method)

# Resource predicates
paired(acquire, release)
cleanup_in_all_exits(resource)

# Safety predicates
has_safety_comment(unsafe_block)
no_panic_sources(fn)
```

**Example**:
```python
ConstraintSpec(
    language="go",
    semantic_constraints=[
        SemanticConstraint(
            kind="precondition",
            expression="precedes(nil_check(user), access(user))",
            scope="processUser",
            variables=("user",),
        ),
        SemanticConstraint(
            kind="postcondition",
            expression="cleanup_in_all_exits(resource)",
            scope="critical_section",
            variables=("resource",),
        ),
    ],
)
```

**Use Cases**:
- Null/nil checks before dereference
- Resource cleanup guarantees (defer, errdefer)
- Readonly/immutability enforcement
- Cancellation handling in coroutines
- Memory safety invariants in unsafe code

---

## Tier Combinations in Examples

Most constraint examples combine multiple tiers for realistic scenarios:

| Example ID | Syntax | Type | Import | ControlFlow | Semantic |
|------------|--------|------|--------|-------------|----------|
| py-types-001 | regex | bindings, expected_type | - | - | - |
| ts-imports-002 | regex | - | imports | - | - |
| go-controlflow-001 | regex | bindings | - | function, loop | - |
| rust-semantics-001 | ebnf | bindings | - | - | invariant |
| kt-coro-003 | regex | bindings | imports | async | invariant, postcondition |

---

## Domain Configuration

Each tier can be further customized with domain-specific configurations:

```python
ConstraintSpec(
    domain_configs={
        "syntax": {
            "naming_convention": "snake_case",
            "validate_format_strings": True,
        },
        "types": {
            "infer_generics": True,
            "strict_null_checks": True,
        },
        "controlflow": {
            "require_exhaustive_switch": True,
            "require_errdefer": True,
        },
        "semantic": {
            "require_paired_alloc_free": True,
            "track_struct_ownership": True,
        },
    },
)
```

---

## Cache Scope by Tier

The `cache_scope` field controls caching behavior based on constraint complexity:

| Cache Scope | Tier Compatibility | Description |
|-------------|-------------------|-------------|
| `SYNTAX_ONLY` | Tier 1 only | Maximum cache reuse, ignores context |
| `SYNTAX_AND_LANG` | Tiers 1-2 | Includes language in cache key |
| `FULL_CONTEXT` | Tiers 3-5 | All context in cache key |

---

## Implementation Status

| Tier | Status | Notes |
|------|--------|-------|
| 1. Syntax | Implemented | regex, json_schema, ebnf via llguidance |
| 2. Type | Partial | TypeDomain exists, needs LSP integration |
| 3. Import | Partial | ImportDomain exists, needs resolver |
| 4. Control Flow | Partial | ControlFlowDomain exists |
| 5. Semantic | Prototype | SemanticDomain needs predicate evaluator |

---

## Adding New Examples

When creating new constraint examples:

1. **Start minimal**: Use lowest tier that achieves the constraint
2. **Add context progressively**: Only add higher tiers when needed
3. **Use actionable predicates**: Semantic expressions should be machine-verifiable
4. **Include valid/invalid outputs**: Demonstrate expected masking behavior
5. **Tag appropriately**: Include tier-relevant tags

```python
ConstraintExample(
    id="lang-domain-NNN",
    name="Descriptive Name",
    description="What this constraint enforces",
    scenario="Real-world developer scenario",
    spec=ConstraintSpec(
        language="python",
        regex=r"...",  # Tier 1
        type_bindings=[...],  # Tier 2
        semantic_constraints=[...],  # Tier 5
    ),
    expected_effect="What tokens are masked and why",
    valid_outputs=["code that passes"],
    invalid_outputs=["code that is masked"],
    tags=["domain", "tier-features", "use-case"],
    language="python",
    domain="semantics",
)
```
