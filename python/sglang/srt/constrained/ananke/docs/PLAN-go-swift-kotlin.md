# Principled Addition of Go, Swift, and Kotlin to Ananke

## Executive Summary

This plan details the systematic addition of Go, Swift, and Kotlin as fully supported languages in the Ananke constrained decoding system. Each language requires implementation of four core components (type system, token classifier, import resolver, parser) plus comprehensive test coverage, following the established patterns from Python, Zig, Rust, and TypeScript.

**Estimated scope**: ~15,000 lines of code across 24 files (8 per language)

---

## 1. Architecture Overview

### 1.1 Required Components Per Language

| Component | File Location | Purpose | LOC (est.) |
|-----------|---------------|---------|------------|
| Type System | `domains/types/languages/{lang}.py` | Type hierarchy, parsing, assignability | 600-900 |
| Token Classifier | `core/token_classifier_{lang}.py` | Lexical classification | 400-500 |
| Import Resolver | `domains/imports/resolvers/{lang}.py` | Module resolution | 150-250 |
| Parser | `parsing/languages/{lang}.py` | Incremental parsing, hole detection | 500-700 |
| Type Tests | `tests/unit/test_{lang}_types.py` | Type system verification | 400-600 |
| Classifier Tests | `tests/unit/test_{lang}_token_classifier.py` | Token classification tests | 300-400 |
| Import Tests | `tests/unit/test_{lang}_imports.py` | Import resolution tests | 150-200 |
| Parser Tests | `tests/unit/test_{lang}_parsing.py` | Parser behavior tests | 400-500 |

### 1.2 Registration Points

Each language must be registered in:
1. `domains/types/languages/__init__.py` - Type system exports and `get_type_system()`
2. `domains/imports/resolvers/__init__.py` - Resolver exports
3. `parsing/__init__.py` - Parser factory registration
4. `core/token_classifier.py` - Language keyword/builtin integration
5. `tests/integration/test_multilang_generation.py` - Integration tests

---

## 2. Go Language Implementation

### 2.1 Type System Characteristics

**Philosophy**: Go prioritizes simplicity. No inheritance, no generics until 1.18, interfaces are structural.

#### 2.1.1 Primitive Types

```python
# Signed integers
GO_INT = PrimitiveType("int")      # Platform-dependent (32 or 64 bit)
GO_INT8 = PrimitiveType("int8")
GO_INT16 = PrimitiveType("int16")
GO_INT32 = PrimitiveType("int32")
GO_INT64 = PrimitiveType("int64")

# Unsigned integers
GO_UINT = PrimitiveType("uint")
GO_UINT8 = PrimitiveType("uint8")   # alias: byte
GO_UINT16 = PrimitiveType("uint16")
GO_UINT32 = PrimitiveType("uint32")
GO_UINT64 = PrimitiveType("uint64")
GO_UINTPTR = PrimitiveType("uintptr")

# Floating point
GO_FLOAT32 = PrimitiveType("float32")
GO_FLOAT64 = PrimitiveType("float64")

# Complex numbers
GO_COMPLEX64 = PrimitiveType("complex64")
GO_COMPLEX128 = PrimitiveType("complex128")

# Other
GO_BOOL = PrimitiveType("bool")
GO_STRING = PrimitiveType("string")
GO_BYTE = GO_UINT8  # alias
GO_RUNE = GO_INT32  # alias (Unicode code point)
GO_ERROR = InterfaceType("error", [("Error", FunctionType([], GO_STRING))])
```

#### 2.1.2 Composite Types

```python
@dataclass(frozen=True)
class GoArrayType(Type):
    """Fixed-size array: [N]T"""
    length: int
    element: Type

@dataclass(frozen=True)
class GoSliceType(Type):
    """Dynamic slice: []T"""
    element: Type

@dataclass(frozen=True)
class GoMapType(Type):
    """Map type: map[K]V"""
    key: Type
    value: Type

@dataclass(frozen=True)
class GoPointerType(Type):
    """Pointer type: *T"""
    pointee: Type

@dataclass(frozen=True)
class GoChannelType(Type):
    """Channel type: chan T, chan<- T, <-chan T"""
    element: Type
    direction: Literal["bidirectional", "send", "receive"]

@dataclass(frozen=True)
class GoFunctionType(Type):
    """Function type: func(params) (returns)"""
    parameters: Tuple[Tuple[str, Type], ...]
    returns: Tuple[Type, ...]  # Multiple return values
    variadic: bool = False

@dataclass(frozen=True)
class GoInterfaceType(Type):
    """Interface type with method set"""
    name: str
    methods: Tuple[Tuple[str, GoFunctionType], ...]
    embedded: Tuple["GoInterfaceType", ...] = ()

@dataclass(frozen=True)
class GoStructType(Type):
    """Struct type with fields"""
    name: Optional[str]  # None for anonymous structs
    fields: Tuple[Tuple[str, Type, str], ...]  # (name, type, tag)
    embedded: Tuple[Type, ...] = ()
```

#### 2.1.3 Go 1.18+ Generics

```python
@dataclass(frozen=True)
class GoTypeParameter(Type):
    """Type parameter: T, K comparable"""
    name: str
    constraint: Optional[Type] = None  # interface constraint

@dataclass(frozen=True)
class GoGenericType(Type):
    """Generic instantiation: List[int]"""
    base: str
    type_args: Tuple[Type, ...]
```

#### 2.1.4 TypeSystemCapabilities

```python
GO_CAPABILITIES = TypeSystemCapabilities(
    supports_generics=True,           # Go 1.18+
    supports_union_types=False,       # No union types (use interfaces)
    supports_optional_types=False,    # Pointers used instead
    supports_type_inference=True,     # := short declaration
    supports_protocols=True,          # Structural interface matching
    supports_variance=False,          # No variance annotations
    supports_overloading=False,       # No function overloading
    supports_ownership=False,         # GC-managed
    supports_comptime=False,          # No compile-time evaluation
    supports_error_unions=False,      # Uses (T, error) returns
    supports_lifetime_bounds=False,   # GC-managed
    supports_sentinels=False,
    supports_allocators=False,
)
```

### 2.2 Token Classification

```python
GO_CONTROL_KEYWORDS = frozenset({
    "if", "else", "for", "switch", "case", "default",
    "break", "continue", "goto", "return", "fallthrough",
    "select", "range",
})

GO_DEFINITION_KEYWORDS = frozenset({
    "func", "var", "const", "type", "struct", "interface",
    "package", "import", "map", "chan",
})

GO_SPECIAL_KEYWORDS = frozenset({
    "defer", "go",  # Goroutines and deferred execution
})

GO_LITERAL_KEYWORDS = frozenset({
    "true", "false", "nil", "iota",
})

GO_BUILTIN_FUNCTIONS = frozenset({
    "append", "cap", "close", "complex", "copy", "delete",
    "imag", "len", "make", "new", "panic", "print", "println",
    "real", "recover",
})

GO_BUILTIN_TYPES = frozenset({
    "bool", "byte", "complex64", "complex128", "error",
    "float32", "float64", "int", "int8", "int16", "int32", "int64",
    "rune", "string", "uint", "uint8", "uint16", "uint32", "uint64",
    "uintptr", "any", "comparable",
})

GO_OPERATORS = frozenset({
    "+", "-", "*", "/", "%",           # Arithmetic
    "&", "|", "^", "&^", "<<", ">>",   # Bitwise
    "==", "!=", "<", ">", "<=", ">=",  # Comparison
    "&&", "||", "!",                   # Logical
    "<-",                              # Channel
    "...",                             # Variadic
    ":=",                              # Short declaration
    "=", "+=", "-=", "*=", "/=", "%=", # Assignment
    "&=", "|=", "^=", "&^=", "<<=", ">>=",
})

GO_DELIMITERS = frozenset({
    "(", ")", "[", "]", "{", "}",
    ",", ".", ";", ":",
})
```

### 2.3 Import Resolution

```python
GO_STANDARD_LIBRARY = {
    "fmt": ["Print", "Println", "Printf", "Sprintf", "Errorf", ...],
    "os": ["Open", "Create", "ReadFile", "WriteFile", "Stdin", "Stdout", ...],
    "io": ["Reader", "Writer", "ReadAll", "Copy", ...],
    "strings": ["Contains", "Split", "Join", "Replace", ...],
    "strconv": ["Atoi", "Itoa", "ParseInt", "FormatInt", ...],
    "encoding/json": ["Marshal", "Unmarshal", "Encoder", "Decoder", ...],
    "net/http": ["Get", "Post", "ListenAndServe", "Handler", ...],
    "context": ["Context", "Background", "WithCancel", "WithTimeout", ...],
    "sync": ["Mutex", "RWMutex", "WaitGroup", "Once", ...],
    "errors": ["New", "Is", "As", "Unwrap", ...],
    # ... comprehensive standard library
}
```

### 2.4 Hole Detection Patterns

```python
# Go-specific hole indicators
GO_INCOMPLETE_PATTERNS = [
    # Function without body
    r"func\s+\w+\s*\([^)]*\)\s*(?:\([^)]*\)|[\w*]+)?\s*$",
    # Incomplete struct
    r"type\s+\w+\s+struct\s*\{[^}]*$",
    # Incomplete interface
    r"type\s+\w+\s+interface\s*\{[^}]*$",
    # Trailing operators
    r"[+\-*/%&|^<>=!]+\s*$",
    # Incomplete assignment
    r"(?:var|const|\w+)\s*:?=\s*$",
    # Unclosed brackets
    r"\([^)]*$", r"\[[^\]]*$", r"\{[^}]*$",
]
```

---

## 3. Swift Language Implementation

### 3.1 Type System Characteristics

**Philosophy**: Swift combines safety, performance, and expressiveness with powerful type inference and protocol-oriented programming.

#### 3.1.1 Primitive Types

```python
# Integers
SWIFT_INT = PrimitiveType("Int")      # Platform-dependent
SWIFT_INT8 = PrimitiveType("Int8")
SWIFT_INT16 = PrimitiveType("Int16")
SWIFT_INT32 = PrimitiveType("Int32")
SWIFT_INT64 = PrimitiveType("Int64")
SWIFT_UINT = PrimitiveType("UInt")
SWIFT_UINT8 = PrimitiveType("UInt8")
SWIFT_UINT16 = PrimitiveType("UInt16")
SWIFT_UINT32 = PrimitiveType("UInt32")
SWIFT_UINT64 = PrimitiveType("UInt64")

# Floating point
SWIFT_FLOAT = PrimitiveType("Float")    # 32-bit
SWIFT_DOUBLE = PrimitiveType("Double")  # 64-bit
SWIFT_FLOAT80 = PrimitiveType("Float80")

# Other primitives
SWIFT_BOOL = PrimitiveType("Bool")
SWIFT_CHARACTER = PrimitiveType("Character")
SWIFT_STRING = PrimitiveType("String")
SWIFT_VOID = PrimitiveType("Void")
SWIFT_NEVER = PrimitiveType("Never")  # Bottom type
SWIFT_ANY = PrimitiveType("Any")      # Universal base
SWIFT_ANYOBJECT = PrimitiveType("AnyObject")  # Class instances only
```

#### 3.1.2 Composite Types

```python
@dataclass(frozen=True)
class SwiftArrayType(Type):
    """Array type: [T] or Array<T>"""
    element: Type

@dataclass(frozen=True)
class SwiftDictionaryType(Type):
    """Dictionary type: [K: V] or Dictionary<K, V>"""
    key: Type
    value: Type

@dataclass(frozen=True)
class SwiftSetType(Type):
    """Set type: Set<T>"""
    element: Type

@dataclass(frozen=True)
class SwiftOptionalType(Type):
    """Optional type: T? or Optional<T>"""
    wrapped: Type

@dataclass(frozen=True)
class SwiftImplicitlyUnwrappedOptionalType(Type):
    """Implicitly unwrapped: T!"""
    wrapped: Type

@dataclass(frozen=True)
class SwiftTupleType(Type):
    """Tuple type: (T1, T2, ...) with optional labels"""
    elements: Tuple[Tuple[Optional[str], Type], ...]

@dataclass(frozen=True)
class SwiftFunctionType(Type):
    """Function type: (P1, P2) -> R"""
    parameters: Tuple[SwiftParameter, ...]
    return_type: Type
    is_throwing: bool = False
    is_async: bool = False
    is_escaping: bool = False

@dataclass(frozen=True)
class SwiftProtocolType(Type):
    """Protocol type with associated types and requirements"""
    name: str
    requirements: Tuple[Tuple[str, Type], ...]
    associated_types: Tuple[Tuple[str, Optional[Type]], ...] = ()

@dataclass(frozen=True)
class SwiftOpaqueType(Type):
    """Opaque return type: some Protocol"""
    constraint: SwiftProtocolType

@dataclass(frozen=True)
class SwiftExistentialType(Type):
    """Existential type: any Protocol"""
    constraint: SwiftProtocolType

@dataclass(frozen=True)
class SwiftResultType(Type):
    """Result type: Result<Success, Failure>"""
    success: Type
    failure: Type

@dataclass(frozen=True)
class SwiftActorType(Type):
    """Actor type for concurrency"""
    name: str
    methods: Tuple[Tuple[str, SwiftFunctionType], ...]
```

#### 3.1.3 Generic System

```python
@dataclass(frozen=True)
class SwiftTypeParameter(Type):
    """Type parameter with constraints: T: Protocol"""
    name: str
    constraints: Tuple[Type, ...] = ()

@dataclass(frozen=True)
class SwiftGenericType(Type):
    """Generic instantiation: Array<Int>"""
    base: str
    type_args: Tuple[Type, ...]

@dataclass(frozen=True)
class SwiftWhereClause:
    """Where clause constraints"""
    constraints: Tuple[Tuple[Type, str, Type], ...]  # (T, ":", Protocol)
```

#### 3.1.4 TypeSystemCapabilities

```python
SWIFT_CAPABILITIES = TypeSystemCapabilities(
    supports_generics=True,
    supports_union_types=False,       # Use enums instead
    supports_optional_types=True,     # First-class optionals
    supports_type_inference=True,     # Strong type inference
    supports_protocols=True,          # Protocol-oriented
    supports_variance=True,           # Covariance/contravariance
    supports_overloading=True,        # Function overloading
    supports_ownership=False,         # ARC-managed
    supports_comptime=False,
    supports_error_unions=False,      # Uses throws/Result
    supports_lifetime_bounds=False,   # ARC-managed
    supports_sentinels=False,
    supports_allocators=False,
)
```

### 3.2 Token Classification

```python
SWIFT_CONTROL_KEYWORDS = frozenset({
    "if", "else", "guard", "switch", "case", "default",
    "for", "in", "while", "repeat",
    "break", "continue", "fallthrough", "return",
    "throw", "throws", "rethrows", "try", "catch", "do",
    "defer", "where",
})

SWIFT_DEFINITION_KEYWORDS = frozenset({
    "func", "var", "let", "class", "struct", "enum",
    "protocol", "extension", "typealias", "associatedtype",
    "init", "deinit", "subscript", "operator", "precedencegroup",
    "import", "actor", "macro",
})

SWIFT_MODIFIER_KEYWORDS = frozenset({
    "public", "private", "internal", "fileprivate", "open",
    "static", "final", "override", "required", "convenience",
    "lazy", "weak", "unowned", "mutating", "nonmutating",
    "dynamic", "optional", "indirect", "inout", "some", "any",
    "async", "await", "nonisolated", "isolated",
    "@escaping", "@autoclosure", "@discardableResult",
})

SWIFT_LITERAL_KEYWORDS = frozenset({
    "true", "false", "nil",
    "self", "Self", "super",
})

SWIFT_BUILTIN_TYPES = frozenset({
    "Int", "Int8", "Int16", "Int32", "Int64",
    "UInt", "UInt8", "UInt16", "UInt32", "UInt64",
    "Float", "Double", "Float80",
    "Bool", "Character", "String",
    "Void", "Never", "Any", "AnyObject",
    "Optional", "Result", "Array", "Dictionary", "Set",
})

SWIFT_OPERATORS = frozenset({
    "+", "-", "*", "/", "%",           # Arithmetic
    "==", "!=", "<", ">", "<=", ">=",  # Comparison
    "===", "!==",                      # Identity
    "&&", "||", "!",                   # Logical
    "&", "|", "^", "~", "<<", ">>",    # Bitwise
    "??",                              # Nil coalescing
    "?.", "!.",                        # Optional chaining
    "...", "..<",                      # Ranges
    "->", "=>",                        # Arrow operators
    "=", "+=", "-=", "*=", "/=", "%=", # Assignment
    "is", "as", "as?", "as!",          # Type casting
})
```

### 3.3 Import Resolution

```python
SWIFT_STANDARD_FRAMEWORKS = {
    "Foundation": ["URL", "Data", "Date", "UUID", "JSONEncoder", ...],
    "SwiftUI": ["View", "Text", "Button", "State", "Binding", ...],
    "UIKit": ["UIView", "UIViewController", "UITableView", ...],
    "Combine": ["Publisher", "Subscriber", "Subject", "AnyCancellable", ...],
    "Concurrency": ["Task", "Actor", "AsyncSequence", ...],
    "Swift": ["Array", "Dictionary", "Set", "String", "Optional", ...],
}
```

---

## 4. Kotlin Language Implementation

### 4.1 Type System Characteristics

**Philosophy**: Kotlin emphasizes null safety, conciseness, and Java interoperability with modern language features.

#### 4.1.1 Primitive Types

```python
# Integers (map to Java primitives on JVM)
KOTLIN_BYTE = PrimitiveType("Byte")
KOTLIN_SHORT = PrimitiveType("Short")
KOTLIN_INT = PrimitiveType("Int")
KOTLIN_LONG = PrimitiveType("Long")

# Unsigned integers
KOTLIN_UBYTE = PrimitiveType("UByte")
KOTLIN_USHORT = PrimitiveType("UShort")
KOTLIN_UINT = PrimitiveType("UInt")
KOTLIN_ULONG = PrimitiveType("ULong")

# Floating point
KOTLIN_FLOAT = PrimitiveType("Float")
KOTLIN_DOUBLE = PrimitiveType("Double")

# Other primitives
KOTLIN_BOOLEAN = PrimitiveType("Boolean")
KOTLIN_CHAR = PrimitiveType("Char")
KOTLIN_STRING = PrimitiveType("String")
KOTLIN_UNIT = PrimitiveType("Unit")     # Like void
KOTLIN_NOTHING = PrimitiveType("Nothing")  # Bottom type
KOTLIN_ANY = PrimitiveType("Any")       # Root type (like Object)
```

#### 4.1.2 Nullable Types

```python
@dataclass(frozen=True)
class KotlinNullableType(Type):
    """Nullable type: T?"""
    inner: Type

@dataclass(frozen=True)
class KotlinPlatformType(Type):
    """Platform type from Java: T! (nullable status unknown)"""
    inner: Type
```

#### 4.1.3 Composite Types

```python
@dataclass(frozen=True)
class KotlinArrayType(Type):
    """Array type: Array<T> or IntArray, etc."""
    element: Type
    is_primitive_array: bool = False

@dataclass(frozen=True)
class KotlinListType(Type):
    """List type: List<T> or MutableList<T>"""
    element: Type
    is_mutable: bool = False

@dataclass(frozen=True)
class KotlinSetType(Type):
    """Set type: Set<T> or MutableSet<T>"""
    element: Type
    is_mutable: bool = False

@dataclass(frozen=True)
class KotlinMapType(Type):
    """Map type: Map<K, V> or MutableMap<K, V>"""
    key: Type
    value: Type
    is_mutable: bool = False

@dataclass(frozen=True)
class KotlinFunctionType(Type):
    """Function type: (P1, P2) -> R"""
    parameters: Tuple[Type, ...]
    return_type: Type
    is_suspend: bool = False
    receiver: Optional[Type] = None  # Extension receiver

@dataclass(frozen=True)
class KotlinPairType(Type):
    """Pair<A, B>"""
    first: Type
    second: Type

@dataclass(frozen=True)
class KotlinTripleType(Type):
    """Triple<A, B, C>"""
    first: Type
    second: Type
    third: Type
```

#### 4.1.4 Class System

```python
@dataclass(frozen=True)
class KotlinClassType(Type):
    """Class type with modifiers"""
    name: str
    type_parameters: Tuple["KotlinTypeParameter", ...] = ()
    superclass: Optional[Type] = None
    interfaces: Tuple[Type, ...] = ()
    is_data: bool = False
    is_sealed: bool = False
    is_inline: bool = False  # value class
    is_object: bool = False  # singleton

@dataclass(frozen=True)
class KotlinSealedType(Type):
    """Sealed class/interface with known subclasses"""
    base: KotlinClassType
    subclasses: Tuple[Type, ...]
```

#### 4.1.5 Generic System

```python
@dataclass(frozen=True)
class KotlinTypeParameter(Type):
    """Type parameter with variance and bounds"""
    name: str
    variance: Optional[Literal["in", "out"]] = None
    upper_bound: Optional[Type] = None
    is_reified: bool = False

@dataclass(frozen=True)
class KotlinGenericType(Type):
    """Generic instantiation: List<String>"""
    base: str
    type_args: Tuple[Type, ...]

@dataclass(frozen=True)
class KotlinStarProjection(Type):
    """Star projection: List<*>"""
    pass
```

#### 4.1.6 TypeSystemCapabilities

```python
KOTLIN_CAPABILITIES = TypeSystemCapabilities(
    supports_generics=True,
    supports_union_types=False,       # Use sealed classes
    supports_optional_types=True,     # Nullable types T?
    supports_type_inference=True,     # Strong inference
    supports_protocols=False,         # Interfaces are nominal
    supports_variance=True,           # in/out variance
    supports_overloading=True,        # Function overloading
    supports_ownership=False,         # GC-managed (JVM)
    supports_comptime=False,
    supports_error_unions=False,      # Uses exceptions/Result
    supports_lifetime_bounds=False,
    supports_sentinels=False,
    supports_allocators=False,
)
```

### 4.2 Token Classification

```python
KOTLIN_CONTROL_KEYWORDS = frozenset({
    "if", "else", "when", "for", "while", "do",
    "break", "continue", "return", "throw",
    "try", "catch", "finally",
})

KOTLIN_DEFINITION_KEYWORDS = frozenset({
    "fun", "val", "var", "class", "interface", "object",
    "enum", "sealed", "data", "annotation", "typealias",
    "constructor", "init", "companion", "import", "package",
})

KOTLIN_MODIFIER_KEYWORDS = frozenset({
    "public", "private", "protected", "internal",
    "open", "final", "abstract", "override",
    "lateinit", "inline", "noinline", "crossinline",
    "reified", "suspend", "tailrec", "external",
    "const", "vararg", "operator", "infix",
    "inner", "out", "in", "value",
})

KOTLIN_SPECIAL_KEYWORDS = frozenset({
    "this", "super", "null", "true", "false",
    "is", "as", "in", "!in", "as?",
    "by", "where", "get", "set", "field",
    "it",  # Implicit lambda parameter
})

KOTLIN_BUILTIN_TYPES = frozenset({
    "Byte", "Short", "Int", "Long",
    "UByte", "UShort", "UInt", "ULong",
    "Float", "Double",
    "Boolean", "Char", "String",
    "Unit", "Nothing", "Any",
    "Array", "List", "Set", "Map",
    "MutableList", "MutableSet", "MutableMap",
    "Pair", "Triple", "Sequence",
})

KOTLIN_OPERATORS = frozenset({
    "+", "-", "*", "/", "%",           # Arithmetic
    "==", "!=", "<", ">", "<=", ">=",  # Structural equality
    "===", "!==",                      # Referential equality
    "&&", "||", "!",                   # Logical
    "in", "!in",                       # Containment
    "is", "!is",                       # Type check
    "as", "as?",                       # Type cast
    "..", "..<", "downTo", "step", "until",  # Ranges
    "?:", "?.",                        # Null safety
    "!!",                              # Not-null assertion
    "=", "+=", "-=", "*=", "/=", "%=", # Assignment
    "->", "::",                        # Function/reference
})
```

### 4.3 Import Resolution

```python
KOTLIN_STANDARD_LIBRARY = {
    "kotlin": ["Any", "Unit", "Nothing", "Array", "Boolean", ...],
    "kotlin.collections": ["List", "Set", "Map", "listOf", "mutableListOf", ...],
    "kotlin.io": ["print", "println", "readLine", ...],
    "kotlin.text": ["String", "StringBuilder", "Regex", ...],
    "kotlin.coroutines": ["Continuation", "CoroutineContext", ...],
    "kotlinx.coroutines": ["launch", "async", "runBlocking", "Flow", ...],
}

KOTLIN_JAVA_INTEROP = {
    "java.lang": ["Object", "String", "Integer", "Exception", ...],
    "java.util": ["List", "Map", "Set", "ArrayList", "HashMap", ...],
    "java.io": ["File", "InputStream", "OutputStream", ...],
}
```

---

## 5. Implementation Phases

### Phase 1: Go (Weeks 1-2)

**Rationale**: Simplest type system, clear semantics, good baseline.

1. **Week 1: Core Implementation**
   - Day 1-2: `token_classifier_go.py` (keywords, operators, literals)
   - Day 3-4: `go.py` type system (primitives, composites)
   - Day 5: `go.py` resolver (standard library)

2. **Week 2: Parser and Tests**
   - Day 1-2: `go.py` parser (incremental, holes)
   - Day 3-4: All test files
   - Day 5: Integration tests, registration

**Deliverables**:
- `core/token_classifier_go.py` (~450 LOC)
- `domains/types/languages/go.py` (~700 LOC)
- `domains/imports/resolvers/go.py` (~200 LOC)
- `parsing/languages/go.py` (~600 LOC)
- 4 test files (~1500 LOC)

### Phase 2: Kotlin (Weeks 3-4)

**Rationale**: JVM target with familiar patterns, null safety adds complexity.

1. **Week 3: Core Implementation**
   - Day 1-2: `token_classifier_kotlin.py`
   - Day 3-4: `kotlin.py` type system (nullable types, variance)
   - Day 5: `kotlin.py` resolver (stdlib + Java interop)

2. **Week 4: Parser and Tests**
   - Day 1-2: `kotlin.py` parser
   - Day 3-4: All test files
   - Day 5: Integration tests, registration

**Deliverables**:
- `core/token_classifier_kotlin.py` (~500 LOC)
- `domains/types/languages/kotlin.py` (~800 LOC)
- `domains/imports/resolvers/kotlin.py` (~250 LOC)
- `parsing/languages/kotlin.py` (~650 LOC)
- 4 test files (~1700 LOC)

### Phase 3: Swift (Weeks 5-6)

**Rationale**: Most complex - optionals, protocols, actors, async/await.

1. **Week 5: Core Implementation**
   - Day 1-2: `token_classifier_swift.py`
   - Day 3-4: `swift.py` type system (optionals, protocols, actors)
   - Day 5: `swift.py` resolver (Foundation, SwiftUI)

2. **Week 6: Parser and Tests**
   - Day 1-2: `swift.py` parser (complex syntax)
   - Day 3-4: All test files
   - Day 5: Integration tests, registration

**Deliverables**:
- `core/token_classifier_swift.py` (~550 LOC)
- `domains/types/languages/swift.py` (~900 LOC)
- `domains/imports/resolvers/swift.py` (~250 LOC)
- `parsing/languages/swift.py` (~700 LOC)
- 4 test files (~1800 LOC)

### Phase 4: Integration and Polish (Week 7)

1. **Cross-language tests** - Verify all languages work together
2. **Property-based tests** - Hypothesis tests for all three
3. **Documentation** - Update docstrings and comments
4. **Performance testing** - Verify <500μs token processing

---

## 6. Critical Design Decisions

### 6.1 Go: Error Handling

Go uses `(T, error)` return patterns, not sum types. We need:
- Special handling for `error` interface
- Multi-return type support in `GoFunctionType`
- Hole detection for incomplete error checks

### 6.2 Swift: Optional Chaining

Swift's `?.` operator requires tracking optional unwrapping depth:
```swift
let x = obj?.property?.method()  // Result is T??
```

Implementation:
```python
def resolve_optional_chain(self, chain: List[Access]) -> Type:
    """Resolve optional chaining, accumulating optional levels."""
    result_type = self.base_type
    for access in chain:
        if access.is_optional:
            if isinstance(result_type, SwiftOptionalType):
                result_type = SwiftOptionalType(self.resolve_access(result_type.wrapped, access))
            else:
                result_type = SwiftOptionalType(self.resolve_access(result_type, access))
    return result_type
```

### 6.3 Kotlin: Platform Types

Java interop introduces platform types (T!) with unknown nullability:
```kotlin
val s: String = javaMethod()  // May be null at runtime!
```

Implementation:
```python
def check_assignable_platform(self, source: Type, target: Type) -> bool:
    """Platform types are assignable to both T and T?."""
    if isinstance(source, KotlinPlatformType):
        return (self.check_assignable(source.inner, target) or
                self.check_assignable(KotlinNullableType(source.inner), target))
    return False
```

### 6.4 Common: Variance

Both Swift and Kotlin have variance annotations. Implementation:
```python
def check_generic_variance(self, source: GenericType, target: GenericType) -> bool:
    """Check variance-aware generic assignability."""
    if source.base != target.base:
        return False

    for s_arg, t_arg, param in zip(source.type_args, target.type_args, self.get_type_params(source.base)):
        if param.variance == "out":  # Covariant
            if not self.check_assignable(s_arg, t_arg):
                return False
        elif param.variance == "in":  # Contravariant
            if not self.check_assignable(t_arg, s_arg):
                return False
        else:  # Invariant
            if s_arg != t_arg:
                return False
    return True
```

---

## 7. Test Strategy

### 7.1 Unit Tests Per Language (~400 tests each)

| Category | Tests | Description |
|----------|-------|-------------|
| Primitive parsing | 30 | Each primitive type parses correctly |
| Composite parsing | 50 | Arrays, maps, functions, etc. |
| Generic parsing | 40 | Type parameters, constraints |
| Assignability | 80 | Subtyping, covariance, null safety |
| Literal inference | 30 | Infer types from literals |
| Formatting | 30 | Round-trip type formatting |
| Token classification | 60 | Keywords, operators, literals |
| Import resolution | 30 | Standard library, packages |
| Hole detection | 50 | Various incomplete patterns |

### 7.2 Property-Based Tests

Using Hypothesis to verify:
- Assignability reflexivity: `T` assignable to `T`
- Formatting round-trip: `parse(format(T)) == T`
- Checkpoint/restore: state preserved after rollback
- Hole monotonicity: holes decrease as code completes

### 7.3 Integration Tests

- Multi-language comparison tests
- End-to-end generation simulation
- Cross-domain constraint propagation
- Performance benchmarks

---

## 8. Risk Assessment

### 8.1 High Risk

| Risk | Mitigation |
|------|------------|
| Swift optionals complexity | Start with basic `T?`, add chaining later |
| Kotlin Java interop | Focus on Kotlin stdlib first |
| Go generics (1.18+) | Implement basic generics, defer advanced constraints |

### 8.2 Medium Risk

| Risk | Mitigation |
|------|------------|
| Parser complexity | Use existing Rust/TypeScript as templates |
| Test coverage | Set 90% coverage requirement |
| Performance | Profile after each language, optimize as needed |

### 8.3 Low Risk

| Risk | Mitigation |
|------|------------|
| Token classification | Well-defined by language specs |
| Import resolution | Focus on standard libraries |
| Registration | Follow established patterns |

---

## 9. Success Criteria

### 9.1 Functional Requirements

- [ ] All primitive types parse correctly
- [ ] Composite types (arrays, maps, functions) work
- [ ] Generics with constraints supported
- [ ] Null safety (Kotlin, Swift) properly tracked
- [ ] Hole detection works for all common patterns
- [ ] Import resolution for standard libraries
- [ ] Token classification accurate for keywords/operators

### 9.2 Quality Requirements

- [ ] 90%+ test coverage per component
- [ ] <500μs per token type checking
- [ ] <50μs per token classification
- [ ] All property tests pass
- [ ] Integration tests pass
- [ ] No regressions in existing languages

### 9.3 Documentation Requirements

- [ ] Module docstrings with examples
- [ ] Type system capabilities documented
- [ ] Registration instructions updated
- [ ] Test patterns documented

---

## 10. File Creation Checklist

### Go
- [ ] `core/token_classifier_go.py`
- [ ] `domains/types/languages/go.py`
- [ ] `domains/imports/resolvers/go.py`
- [ ] `parsing/languages/go.py`
- [ ] `tests/unit/test_go_types.py`
- [ ] `tests/unit/test_go_token_classifier.py`
- [ ] `tests/unit/test_go_imports.py`
- [ ] `tests/unit/test_go_parsing.py`

### Kotlin
- [ ] `core/token_classifier_kotlin.py`
- [ ] `domains/types/languages/kotlin.py`
- [ ] `domains/imports/resolvers/kotlin.py`
- [ ] `parsing/languages/kotlin.py`
- [ ] `tests/unit/test_kotlin_types.py`
- [ ] `tests/unit/test_kotlin_token_classifier.py`
- [ ] `tests/unit/test_kotlin_imports.py`
- [ ] `tests/unit/test_kotlin_parsing.py`

### Swift
- [ ] `core/token_classifier_swift.py`
- [ ] `domains/types/languages/swift.py`
- [ ] `domains/imports/resolvers/swift.py`
- [ ] `parsing/languages/swift.py`
- [ ] `tests/unit/test_swift_types.py`
- [ ] `tests/unit/test_swift_token_classifier.py`
- [ ] `tests/unit/test_swift_imports.py`
- [ ] `tests/unit/test_swift_parsing.py`

### Registration Updates
- [ ] `domains/types/languages/__init__.py`
- [ ] `domains/imports/resolvers/__init__.py`
- [ ] `parsing/__init__.py`
- [ ] `core/token_classifier.py`
- [ ] `tests/integration/test_multilang_generation.py`
- [ ] `tests/property/test_type_system_properties.py`
