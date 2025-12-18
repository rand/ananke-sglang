"""
Principled Domain-Specific Ananke Evaluation

Tests Ananke's unique value proposition across 4 constraint domains:
- Types: Prevent type errors (bidirectional type inference)
- Imports: Prevent hallucinated imports (module dependency tracking)
- ControlFlow: Prevent dead code (CFG + reachability analysis)
- Semantics: Enforce invariants (SMT constraint solving)

Key Design: 4-condition comparison
1. Unconstrained: Raw LLM output (baseline)
2. Syntax-only: llguidance CFG only
3. Syntax+Types: TypeDomain enabled
4. Full Ananke: All domains enabled

Run with:
    modal run deploy/modal/eval/ananke_domain_eval.py
    modal run deploy/modal/eval/ananke_domain_eval.py --full-comparison
"""

import ast
import json
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import modal

# =============================================================================
# Configuration
# =============================================================================

APP_NAME = "ananke-domain-eval"
DEPLOYED_APP = "qwen3-coder-ananke"
DEPLOYED_CLASS = "Qwen3CoderAnanke"

# All 7 languages supported by Ananke
LANGUAGES = ["python", "typescript", "go", "rust", "kotlin", "swift", "zig"]

# Evaluation parameters
SAMPLES_PER_CONDITION = 20  # Reduced for faster iteration
MAX_TOKENS = 500
TEMPERATURES = [0.3, 0.6, 0.9]


class ConstraintCondition(Enum):
    """The 4 conditions for comparison."""
    UNCONSTRAINED = "unconstrained"
    SYNTAX_ONLY = "syntax_only"
    SYNTAX_TYPES = "syntax_types"
    FULL_ANANKE = "full_ananke"


# =============================================================================
# ConstraintSpec Builder
# =============================================================================

def build_type_binding(name: str, type_expr: str, scope: str = "parameter") -> dict:
    """Build a TypeBinding dictionary."""
    return {"name": name, "type_expr": type_expr, "scope": scope}


def build_function_signature(
    name: str,
    params: list[dict],
    return_type: str,
    type_params: list[str] | None = None,
    is_async: bool = False,
) -> dict:
    """Build a FunctionSignature dictionary."""
    sig = {
        "name": name,
        "params": params,
        "return_type": return_type,
    }
    if type_params:
        sig["type_params"] = type_params
    if is_async:
        sig["is_async"] = is_async
    return sig


def build_import_binding(module: str, name: str | None = None, alias: str | None = None) -> dict:
    """Build an ImportBinding dictionary."""
    binding = {"module": module}
    if name:
        binding["name"] = name
    if alias:
        binding["alias"] = alias
    return binding


def build_control_flow(
    function_name: str,
    expected_return_type: str | None = None,
    loop_depth: int = 0,
    in_async_context: bool = False,
) -> dict:
    """Build a ControlFlowContext dictionary."""
    ctx = {"function_name": function_name}
    if expected_return_type:
        ctx["expected_return_type"] = expected_return_type
    if loop_depth > 0:
        ctx["loop_depth"] = loop_depth
    if in_async_context:
        ctx["in_async_context"] = in_async_context
    return ctx


def build_semantic_constraint(
    kind: str,
    expression: str,
    scope: str | None = None,
    variables: list[str] | None = None,
) -> dict:
    """Build a SemanticConstraint dictionary."""
    constraint = {"kind": kind, "expression": expression}
    if scope:
        constraint["scope"] = scope
    if variables:
        constraint["variables"] = variables
    return constraint


# =============================================================================
# Test Case Data Structures
# =============================================================================

@dataclass
class DomainTestCase:
    """A test case for domain-specific evaluation."""
    id: str
    language: str
    domain: str  # "types", "imports", "controlflow", "semantics"
    description: str
    prompt: str

    # Full ConstraintSpec components
    type_bindings: list[dict] = field(default_factory=list)
    function_signatures: list[dict] = field(default_factory=list)
    expected_type: str | None = None
    imports: list[dict] = field(default_factory=list)
    available_modules: list[str] = field(default_factory=list)
    forbidden_imports: list[str] = field(default_factory=list)
    control_flow: dict | None = None
    semantic_constraints: list[dict] = field(default_factory=list)

    # Validation
    expected_patterns: list[str] = field(default_factory=list)
    forbidden_patterns: list[str] = field(default_factory=list)  # Patterns that indicate failure
    custom_validator: Callable[[str], tuple[bool, str | None]] | None = None

    max_tokens: int = 300
    temperature: float = 0.3

    def build_constraint_spec(self, condition: ConstraintCondition) -> dict | None:
        """Build ConstraintSpec for the given condition.

        Note: The SGLang API ConstraintSpecFormat doesn't have 'enabled_domains'.
        We differentiate conditions by what context is provided:
        - UNCONSTRAINED: No constraint_spec at all
        - SYNTAX_ONLY: Only language + regex (syntax constraint required)
        - SYNTAX_TYPES: Language + regex + type context
        - FULL_ANANKE: All context provided

        IMPORTANT: constraint_spec MUST include at least one syntax constraint
        (json_schema, regex, ebnf, or structural_tag) to be valid. Without
        a syntax constraint, the backend returns None and generation is unconstrained.
        """
        if condition == ConstraintCondition.UNCONSTRAINED:
            return None

        # All constrained conditions need a syntax constraint.
        # Use a permissive regex that allows any code generation but activates
        # the Ananke grammar backend for constraint checking.
        spec = {
            "version": "1.0",
            "language": self.language,
            "language_detection": "explicit",
            "cache_scope": "full_context",
            # Required: syntax constraint activates the grammar backend
            "regex": r"[\s\S]*",  # Match any characters including newlines
        }

        if condition == ConstraintCondition.SYNTAX_ONLY:
            # Only syntax validation via language + regex
            # No type/import/semantic context
            return spec

        if condition == ConstraintCondition.SYNTAX_TYPES:
            # Add type context only
            if self.type_bindings:
                spec["type_bindings"] = self.type_bindings
            if self.function_signatures:
                spec["function_signatures"] = self.function_signatures
            if self.expected_type:
                spec["expected_type"] = self.expected_type
            return spec

        # FULL_ANANKE: all context enabled
        if self.type_bindings:
            spec["type_bindings"] = self.type_bindings
        if self.function_signatures:
            spec["function_signatures"] = self.function_signatures
        if self.expected_type:
            spec["expected_type"] = self.expected_type
        if self.imports:
            spec["imports"] = self.imports
        if self.available_modules:
            spec["available_modules"] = self.available_modules
        if self.forbidden_imports:
            spec["forbidden_imports"] = list(self.forbidden_imports)
        if self.control_flow:
            spec["control_flow"] = self.control_flow
        if self.semantic_constraints:
            spec["semantic_constraints"] = self.semantic_constraints

        return spec


@dataclass
class EvalResult:
    """Result of evaluating a single test case under a single condition."""
    test_id: str
    condition: str
    language: str
    domain: str
    success: bool
    valid_syntax: bool
    expected_patterns_found: int
    forbidden_patterns_found: int
    total_expected_patterns: int
    generation_time_ms: float
    output_preview: str
    error: str | None = None
    custom_validation: str | None = None


# =============================================================================
# Domain-Specific Test Suites
# =============================================================================

def get_type_domain_tests() -> list[DomainTestCase]:
    """Test cases that specifically test TypeDomain value-add."""
    cases = []

    # Test 1: Return type mismatch prevention (Python)
    cases.append(DomainTestCase(
        id="type_return_mismatch_py",
        language="python",
        domain="types",
        description="TypeDomain should enforce list[int] return type, not list[str]",
        prompt='''def get_user_ids(users: list[dict]) -> list[int]:
    """Extract user IDs from user objects. Returns list of integer IDs."""
''',
        function_signatures=[
            build_function_signature(
                name="get_user_ids",
                params=[build_type_binding("users", "list[dict]", "parameter")],
                return_type="list[int]",
            ),
        ],
        expected_type="list[int]",
        expected_patterns=["return", "[", "int"],
        forbidden_patterns=["str(", "['"],  # Common error: returning strings
    ))

    # Test 2: Generic type preservation (TypeScript)
    cases.append(DomainTestCase(
        id="type_generic_preserve_ts",
        language="typescript",
        domain="types",
        description="TypeDomain should preserve generic T, not return 'any'",
        prompt='''function first<T>(items: T[]): T | undefined {
    // Return the first item or undefined if empty
''',
        function_signatures=[
            build_function_signature(
                name="first",
                params=[build_type_binding("items", "T[]", "parameter")],
                return_type="T | undefined",
                type_params=["T"],
            ),
        ],
        expected_type="T | undefined",
        expected_patterns=["return", "items[0]", "undefined"],
        forbidden_patterns=[": any", "as any"],  # Loss of type safety
    ))

    # Test 3: Argument type confusion prevention (Python)
    cases.append(DomainTestCase(
        id="type_arg_confusion_py",
        language="python",
        domain="types",
        description="TypeDomain should ensure amount is used as float, currency as str",
        prompt='''def format_price(amount: float, currency: str) -> str:
    """Format a price with currency symbol. Example: format_price(19.99, 'USD') -> '$19.99'"""
''',
        function_signatures=[
            build_function_signature(
                name="format_price",
                params=[
                    build_type_binding("amount", "float", "parameter"),
                    build_type_binding("currency", "str", "parameter"),
                ],
                return_type="str",
            ),
        ],
        type_bindings=[
            build_type_binding("amount", "float", "parameter"),
            build_type_binding("currency", "str", "parameter"),
        ],
        expected_type="str",
        expected_patterns=["return", "amount", "currency"],
    ))

    # Test 4: Option type handling (Rust)
    cases.append(DomainTestCase(
        id="type_option_rust",
        language="rust",
        domain="types",
        description="TypeDomain should enforce Option<i32> return, with proper Some/None",
        prompt='''fn find_max(numbers: &[i32]) -> Option<i32> {
    // Return the maximum value, or None if empty
''',
        function_signatures=[
            build_function_signature(
                name="find_max",
                params=[build_type_binding("numbers", "&[i32]", "parameter")],
                return_type="Option<i32>",
            ),
        ],
        expected_type="Option<i32>",
        expected_patterns=["Some", "None", "return"],
        forbidden_patterns=["unwrap()", "expect("],  # Unsafe unwrap defeats the purpose
    ))

    # Test 5: Null safety (Kotlin)
    cases.append(DomainTestCase(
        id="type_null_safety_kt",
        language="kotlin",
        domain="types",
        description="TypeDomain should enforce null-safe handling of String?",
        prompt='''fun getNameLength(name: String?): Int {
    // Return length of name, or 0 if null
''',
        function_signatures=[
            build_function_signature(
                name="getNameLength",
                params=[build_type_binding("name", "String?", "parameter")],
                return_type="Int",
            ),
        ],
        type_bindings=[build_type_binding("name", "String?", "parameter")],
        expected_type="Int",
        expected_patterns=["return", "?.", "?:", "0"],
        forbidden_patterns=["name.length", "name!!"],  # Unsafe null handling
    ))

    return cases


def get_import_domain_tests() -> list[DomainTestCase]:
    """Test cases that specifically test ImportDomain value-add."""
    cases = []

    # Test 1: Prevent hallucinated Python module
    cases.append(DomainTestCase(
        id="import_hallucination_py",
        language="python",
        domain="imports",
        description="ImportDomain should prevent hallucinated email_validator import",
        prompt='''# Validate email addresses using only standard library
def validate_email(email: str) -> bool:
    """Check if email is valid using only re module."""
''',
        imports=[build_import_binding("re")],
        available_modules=["re", "typing", "string"],
        forbidden_imports=["email_validator", "validators", "pydantic", "email"],
        expected_patterns=["re.", "return", "match", "@"],
        forbidden_patterns=["import email_validator", "from validators", "from pydantic"],
    ))

    # Test 2: Prevent hallucinated npm package (TypeScript)
    cases.append(DomainTestCase(
        id="import_hallucination_ts",
        language="typescript",
        domain="imports",
        description="ImportDomain should use built-in URL, not hallucinated url-parse",
        prompt='''// Parse and validate a URL using built-in URL API
function parseUrl(urlString: string): URL | null {
    // Use the built-in URL constructor
''',
        available_modules=["url"],
        forbidden_imports=["url-parse", "valid-url", "whatwg-url"],
        expected_patterns=["new URL", "try", "catch", "null"],
        forbidden_patterns=["require('url-parse')", "from 'url-parse'"],
    ))

    # Test 3: Enforce specific import (Python JSON)
    cases.append(DomainTestCase(
        id="import_specific_py",
        language="python",
        domain="imports",
        description="ImportDomain should use stdlib json, not orjson/ujson",
        prompt='''# Read JSON from a file using standard library only
def read_json_file(path: str) -> dict:
    """Read and parse JSON file. Use only standard library."""
''',
        imports=[
            build_import_binding("json"),
            build_import_binding("pathlib", "Path"),
        ],
        available_modules=["json", "pathlib", "typing"],
        forbidden_imports=["orjson", "ujson", "simplejson", "rapidjson"],
        function_signatures=[
            build_function_signature(
                name="read_json_file",
                params=[build_type_binding("path", "str", "parameter")],
                return_type="dict",
            ),
        ],
        expected_patterns=["json.load", "open", "return"],
        forbidden_patterns=["import orjson", "import ujson", "import simplejson"],
    ))

    # Test 4: Go import validation
    cases.append(DomainTestCase(
        id="import_go_http",
        language="go",
        domain="imports",
        description="ImportDomain should use net/http, not third-party HTTP libs",
        prompt='''// Make an HTTP GET request using standard library
func fetchURL(url string) (string, error) {
    // Use net/http only
''',
        imports=[
            build_import_binding("net/http"),
            build_import_binding("io"),
        ],
        available_modules=["net/http", "io", "fmt", "errors"],
        forbidden_imports=["github.com/go-resty", "github.com/parnurzeal/gorequest"],
        expected_patterns=["http.Get", "resp", "Body", "return"],
    ))

    return cases


def get_controlflow_domain_tests() -> list[DomainTestCase]:
    """Test cases that specifically test ControlFlowDomain value-add."""
    cases = []

    # Test 1: Prevent dead code after return (Python)
    cases.append(DomainTestCase(
        id="controlflow_dead_code_py",
        language="python",
        domain="controlflow",
        description="ControlFlowDomain should prevent code after exhaustive return",
        prompt='''def classify_number(x: int) -> str:
    """Classify as negative, zero, or positive."""
    if x < 0:
        return "negative"
    elif x == 0:
        return "zero"
    else:
        return "positive"
''',
        control_flow=build_control_flow(
            function_name="classify_number",
            expected_return_type="str",
        ),
        function_signatures=[
            build_function_signature(
                name="classify_number",
                params=[build_type_binding("x", "int", "parameter")],
                return_type="str",
            ),
        ],
        expected_patterns=["return", "negative", "zero", "positive"],
        # Check that there's no unreachable code after the function
    ))

    # Test 2: Ensure all paths return (Go)
    cases.append(DomainTestCase(
        id="controlflow_all_paths_go",
        language="go",
        domain="controlflow",
        description="ControlFlowDomain should ensure all paths return a value",
        prompt='''func classify(n int) string {
    // Classify n as "negative", "positive", or "zero"
    // All paths must return
''',
        control_flow=build_control_flow(
            function_name="classify",
            expected_return_type="string",
        ),
        function_signatures=[
            build_function_signature(
                name="classify",
                params=[build_type_binding("n", "int", "parameter")],
                return_type="string",
            ),
        ],
        expected_patterns=["return", "negative", "positive", "zero"],
    ))

    # Test 3: Prevent unreachable branch (Rust)
    cases.append(DomainTestCase(
        id="controlflow_unreachable_rust",
        language="rust",
        domain="controlflow",
        description="ControlFlowDomain should handle all match arms",
        prompt='''fn day_type(day: u8) -> &'static str {
    // Return "weekday" for 1-5, "weekend" for 6-7, "invalid" otherwise
    match day {
''',
        control_flow=build_control_flow(
            function_name="day_type",
            expected_return_type="&'static str",
        ),
        function_signatures=[
            build_function_signature(
                name="day_type",
                params=[build_type_binding("day", "u8", "parameter")],
                return_type="&'static str",
            ),
        ],
        expected_patterns=["weekday", "weekend", "invalid", "=>"],
    ))

    # Test 4: Loop control flow (TypeScript)
    cases.append(DomainTestCase(
        id="controlflow_loop_ts",
        language="typescript",
        domain="controlflow",
        description="ControlFlowDomain should handle early return from loop",
        prompt='''function findIndex<T>(arr: T[], predicate: (item: T) => boolean): number {
    // Return index of first matching item, or -1 if not found
    for (let i = 0; i < arr.length; i++) {
''',
        control_flow=build_control_flow(
            function_name="findIndex",
            expected_return_type="number",
            loop_depth=1,
        ),
        function_signatures=[
            build_function_signature(
                name="findIndex",
                params=[
                    build_type_binding("arr", "T[]", "parameter"),
                    build_type_binding("predicate", "(item: T) => boolean", "parameter"),
                ],
                return_type="number",
                type_params=["T"],
            ),
        ],
        expected_patterns=["return", "if", "predicate", "-1"],
    ))

    return cases


def get_semantic_domain_tests() -> list[DomainTestCase]:
    """Test cases that specifically test SemanticDomain value-add."""
    cases = []

    # Test 1: Non-negative precondition (Python factorial)
    cases.append(DomainTestCase(
        id="semantic_nonneg_py",
        language="python",
        domain="semantics",
        description="SemanticDomain should generate guard for n >= 0 precondition",
        prompt='''def factorial(n: int) -> int:
    """Return n! (factorial). Precondition: n >= 0."""
''',
        function_signatures=[
            build_function_signature(
                name="factorial",
                params=[build_type_binding("n", "int", "parameter")],
                return_type="int",
            ),
        ],
        semantic_constraints=[
            build_semantic_constraint(
                kind="precondition",
                expression="n >= 0",
                scope="factorial",
                variables=["n"],
            ),
            build_semantic_constraint(
                kind="postcondition",
                expression="result >= 1",
                scope="factorial",
                variables=["result"],
            ),
        ],
        expected_patterns=["if", "n < 0", "raise", "return", "factorial"],
        forbidden_patterns=[],  # Don't forbid anything specific
    ))

    # Test 2: Bounds checking (Python safe_index)
    cases.append(DomainTestCase(
        id="semantic_bounds_py",
        language="python",
        domain="semantics",
        description="SemanticDomain should enforce 0 <= i < len(arr)",
        prompt='''def safe_get(arr: list[int], i: int) -> int:
    """Return arr[i] safely. Precondition: 0 <= i < len(arr)."""
''',
        function_signatures=[
            build_function_signature(
                name="safe_get",
                params=[
                    build_type_binding("arr", "list[int]", "parameter"),
                    build_type_binding("i", "int", "parameter"),
                ],
                return_type="int",
            ),
        ],
        semantic_constraints=[
            build_semantic_constraint(
                kind="precondition",
                expression="0 <= i < len(arr)",
                scope="safe_get",
                variables=["arr", "i"],
            ),
        ],
        expected_patterns=["if", "i < 0", "i >= len", "raise", "return", "arr[i]"],
    ))

    # Test 3: Division by zero (Rust)
    cases.append(DomainTestCase(
        id="semantic_div_zero_rust",
        language="rust",
        domain="semantics",
        description="SemanticDomain should enforce b != 0 for division",
        prompt='''fn safe_divide(a: i32, b: i32) -> Result<i32, &'static str> {
    // Safe division. Returns error if b == 0.
''',
        function_signatures=[
            build_function_signature(
                name="safe_divide",
                params=[
                    build_type_binding("a", "i32", "parameter"),
                    build_type_binding("b", "i32", "parameter"),
                ],
                return_type="Result<i32, &'static str>",
            ),
        ],
        semantic_constraints=[
            build_semantic_constraint(
                kind="precondition",
                expression="b != 0",
                scope="safe_divide",
                variables=["b"],
            ),
        ],
        expected_patterns=["if", "b == 0", "Err", "Ok", "return"],
    ))

    # Test 4: Invariant maintenance (Go)
    cases.append(DomainTestCase(
        id="semantic_invariant_go",
        language="go",
        domain="semantics",
        description="SemanticDomain should maintain capacity > 0 invariant",
        prompt='''// NewBoundedQueue creates a queue with given capacity.
// Precondition: capacity > 0
func NewBoundedQueue(capacity int) (*BoundedQueue, error) {
''',
        function_signatures=[
            build_function_signature(
                name="NewBoundedQueue",
                params=[build_type_binding("capacity", "int", "parameter")],
                return_type="(*BoundedQueue, error)",
            ),
        ],
        semantic_constraints=[
            build_semantic_constraint(
                kind="precondition",
                expression="capacity > 0",
                scope="NewBoundedQueue",
                variables=["capacity"],
            ),
        ],
        expected_patterns=["if", "capacity <= 0", "error", "return", "nil"],
    ))

    return cases


def get_all_test_cases() -> list[DomainTestCase]:
    """Get all domain-specific test cases."""
    cases = []
    cases.extend(get_type_domain_tests())
    cases.extend(get_import_domain_tests())
    cases.extend(get_controlflow_domain_tests())
    cases.extend(get_semantic_domain_tests())
    return cases


# =============================================================================
# Syntax Validation
# =============================================================================

def validate_syntax(code: str, language: str) -> tuple[bool, str | None]:
    """Validate syntax using tree-sitter where available, fallback otherwise."""
    if language == "python":
        return validate_python_syntax(code)

    # For other languages, use basic structural validation
    # In production, would use tree-sitter
    return validate_structural(code, language)


def validate_python_syntax(code: str) -> tuple[bool, str | None]:
    """Validate Python syntax using ast.parse."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def validate_structural(code: str, language: str) -> tuple[bool, str | None]:
    """Basic structural validation (balanced braces, not empty)."""
    if not code.strip():
        return False, "Empty code"

    # Check balanced delimiters
    braces = code.count('{') - code.count('}')
    parens = code.count('(') - code.count(')')
    brackets = code.count('[') - code.count(']')

    if braces != 0:
        return False, f"Unbalanced braces (delta={braces})"
    if parens != 0:
        return False, f"Unbalanced parentheses (delta={parens})"
    if brackets != 0:
        return False, f"Unbalanced brackets (delta={brackets})"

    return True, None


def extract_code(text: str) -> str:
    """Extract code from response, removing markdown."""
    if "```" in text:
        match = re.search(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return text.strip()


# =============================================================================
# Statistical Functions
# =============================================================================

def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)

    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    p = successes / n

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

    return (max(0, center - spread), min(1, center + spread))


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for proportion comparison."""
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    return abs(phi1 - phi2)


def effect_size_label(h: float) -> str:
    """Interpret Cohen's h."""
    if h < 0.2:
        return "negligible"
    elif h < 0.5:
        return "small"
    elif h < 0.8:
        return "medium"
    else:
        return "large"


# =============================================================================
# Modal App
# =============================================================================

app = modal.App(APP_NAME)


def evaluate_single_case(
    test_case: DomainTestCase,
    condition: ConstraintCondition,
    server: Any,
) -> EvalResult:
    """Evaluate a single test case under a single condition (called within run_domain_evaluation)."""
    start_ms = time.time() * 1000
    try:
        constraint_spec = test_case.build_constraint_spec(condition)

        if condition == ConstraintCondition.UNCONSTRAINED or constraint_spec is None:
            output = server.generate.remote(
                prompt=test_case.prompt,
                max_tokens=test_case.max_tokens,
                temperature=test_case.temperature,
            )
        else:
            response = server.generate_constrained.remote(
                prompt=test_case.prompt,
                constraint_spec=constraint_spec,
                max_tokens=test_case.max_tokens,
                temperature=test_case.temperature,
            )
            output = response.get("text", "") if isinstance(response, dict) else str(response)

        gen_time_ms = time.time() * 1000 - start_ms

        # Extract code and validate
        code = extract_code(output)
        valid_syntax, syntax_error = validate_syntax(code, test_case.language)

        # Check patterns
        code_lower = code.lower()
        expected_found = sum(1 for p in test_case.expected_patterns if p.lower() in code_lower)
        forbidden_found = sum(1 for p in test_case.forbidden_patterns if p.lower() in code_lower)

        # Custom validation
        custom_result = None
        if test_case.custom_validator:
            ok, msg = test_case.custom_validator(code)
            custom_result = msg if not ok else None

        # Success = valid syntax AND most expected patterns AND no forbidden patterns
        success = (
            valid_syntax
            and expected_found >= len(test_case.expected_patterns) // 2
            and forbidden_found == 0
            and custom_result is None
        )

        return EvalResult(
            test_id=test_case.id,
            condition=condition.value,
            language=test_case.language,
            domain=test_case.domain,
            success=success,
            valid_syntax=valid_syntax,
            expected_patterns_found=expected_found,
            forbidden_patterns_found=forbidden_found,
            total_expected_patterns=len(test_case.expected_patterns),
            generation_time_ms=gen_time_ms,
            output_preview=code[:300],
            error=syntax_error,
            custom_validation=custom_result,
        )

    except Exception as e:
        return EvalResult(
            test_id=test_case.id,
            condition=condition.value,
            language=test_case.language,
            domain=test_case.domain,
            success=False,
            valid_syntax=False,
            expected_patterns_found=0,
            forbidden_patterns_found=0,
            total_expected_patterns=len(test_case.expected_patterns),
            generation_time_ms=time.time() * 1000 - start_ms,
            output_preview="",
            error=str(e)[:200],
        )


# =============================================================================
# Cold-Start Handling
# =============================================================================

WARMUP_TIMEOUT = 1500  # 25 minutes - matches MODEL_LOAD_TIMEOUT
WARMUP_INITIAL_DELAY = 10  # Start with 10s delay
WARMUP_MAX_DELAY = 60  # Cap at 60s between retries


def warm_up_server(server, timeout: int = WARMUP_TIMEOUT) -> bool:
    """Wait for server to be ready with exponential backoff.

    The Qwen3-Coder 30B MoE model takes 15-20 minutes to load on cold start.
    This function waits with exponential backoff until the server is ready.

    Args:
        server: The Modal server instance
        timeout: Maximum time to wait in seconds (default 25 min)

    Returns:
        True if server is ready, raises RuntimeError otherwise
    """
    print(f"\nWarming up server (timeout: {timeout}s)...")
    print("Note: First cold start may take 15-20 minutes for 60GB model")

    start_time = time.time()
    delay = WARMUP_INITIAL_DELAY
    attempt = 0

    while time.time() - start_time < timeout:
        attempt += 1
        elapsed = time.time() - start_time

        try:
            print(f"  [{elapsed:.0f}s] Attempt {attempt}: checking health...")
            result = server.health.remote()

            if result.get("status") == "healthy":
                print(f"  [{elapsed:.0f}s] Server healthy! Checking readiness...")

                # Also verify it can generate
                ready = server.health_generate.remote()
                if ready.get("ready"):
                    print(f"  [{elapsed:.0f}s] Server ready for generation!")
                    return True
                else:
                    print(f"  [{elapsed:.0f}s] Server healthy but not ready to generate yet")
            else:
                print(f"  [{elapsed:.0f}s] Server not healthy: {result}")

        except Exception as e:
            error_msg = str(e)[:100]
            print(f"  [{elapsed:.0f}s] Connection error (expected during cold start): {error_msg}")

        # Exponential backoff with cap
        print(f"  [{elapsed:.0f}s] Waiting {delay}s before next attempt...")
        time.sleep(delay)
        delay = min(delay * 1.5, WARMUP_MAX_DELAY)

    raise RuntimeError(
        f"Server failed to become ready within {timeout}s. "
        f"Check Modal logs: modal logs qwen3-coder-ananke"
    )


@app.function(timeout=3600)
def run_domain_evaluation(
    conditions: list[str] | None = None,
    domains: list[str] | None = None,
) -> dict:
    """Run domain-specific evaluation."""
    print("=" * 70)
    print("Ananke Domain-Specific Evaluation")
    print("=" * 70)

    # Default to all conditions
    if conditions is None:
        conditions = [c.value for c in ConstraintCondition]
    condition_enums = [ConstraintCondition(c) for c in conditions]

    # Get test cases
    all_cases = get_all_test_cases()

    # Filter by domain if specified
    if domains:
        all_cases = [tc for tc in all_cases if tc.domain in domains]

    print(f"\nTest cases: {len(all_cases)}")
    print(f"Conditions: {conditions}")

    # Group by domain
    by_domain = {}
    for tc in all_cases:
        by_domain.setdefault(tc.domain, []).append(tc)

    print("\nBy domain:")
    for domain, cases in by_domain.items():
        print(f"  {domain}: {len(cases)}")

    # Connect to deployed model
    print("\nConnecting to deployed model...")
    Qwen3CoderAnanke = modal.Cls.from_name(DEPLOYED_APP, DEPLOYED_CLASS)
    server = Qwen3CoderAnanke()

    # Wait for server to be ready (handles cold start)
    warm_up_server(server)

    # Run evaluation
    results: list[EvalResult] = []

    for condition in condition_enums:
        print(f"\n{'='*60}")
        print(f"CONDITION: {condition.value.upper()}")
        print("=" * 60)

        for i, tc in enumerate(all_cases, 1):
            print(f"  [{i}/{len(all_cases)}] {tc.id} ({tc.language}/{tc.domain})")

            result = evaluate_single_case(tc, condition, server)
            results.append(result)

            status = "PASS" if result.success else "FAIL"
            syntax = "OK" if result.valid_syntax else "ERR"
            patterns = f"{result.expected_patterns_found}/{result.total_expected_patterns}"
            forbidden = f"forbidden={result.forbidden_patterns_found}" if result.forbidden_patterns_found else ""

            print(f"    {status} | syntax={syntax} | patterns={patterns} {forbidden} | {result.generation_time_ms:.0f}ms")
            if result.error:
                print(f"    Error: {result.error[:60]}")

    # Compute summary
    summary = compute_domain_summary(results)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    # Print by condition
    print("\nBy Condition:")
    for cond, stats in summary["by_condition"].items():
        rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        ci_low, ci_high = wilson_ci(stats["success"], stats["total"])
        print(f"  {cond:15s}: {stats['success']}/{stats['total']} ({rate:.1%}) [95% CI: {ci_low:.1%}-{ci_high:.1%}]")

    # Print by domain
    print("\nBy Domain:")
    for domain, stats in summary["by_domain"].items():
        rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {domain:12s}: {stats['success']}/{stats['total']} ({rate:.1%})")

    # Print condition comparisons
    if len(condition_enums) > 1:
        print("\nCondition Comparisons (vs Unconstrained):")
        uc_stats = summary["by_condition"].get("unconstrained", {"success": 0, "total": 1})
        uc_rate = uc_stats["success"] / uc_stats["total"] if uc_stats["total"] > 0 else 0

        for cond in ["syntax_only", "syntax_types", "full_ananke"]:
            if cond in summary["by_condition"]:
                stats = summary["by_condition"][cond]
                rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
                delta = rate - uc_rate
                h = cohens_h(rate, uc_rate)
                effect = effect_size_label(h)
                print(f"  {cond:15s}: {delta:+.1%} (Cohen's h={h:.2f}, {effect})")

    return {
        "summary": summary,
        "results": [vars(r) for r in results],
        "config": {
            "conditions": conditions,
            "domains": domains or ["all"],
            "total_cases": len(all_cases),
        },
    }


def compute_domain_summary(results: list[EvalResult]) -> dict:
    """Compute summary statistics by condition and domain."""
    by_condition: dict[str, dict] = {}
    by_domain: dict[str, dict] = {}
    by_language: dict[str, dict] = {}

    for r in results:
        # By condition
        if r.condition not in by_condition:
            by_condition[r.condition] = {"total": 0, "success": 0, "valid_syntax": 0}
        by_condition[r.condition]["total"] += 1
        if r.success:
            by_condition[r.condition]["success"] += 1
        if r.valid_syntax:
            by_condition[r.condition]["valid_syntax"] += 1

        # By domain
        if r.domain not in by_domain:
            by_domain[r.domain] = {"total": 0, "success": 0}
        by_domain[r.domain]["total"] += 1
        if r.success:
            by_domain[r.domain]["success"] += 1

        # By language
        if r.language not in by_language:
            by_language[r.language] = {"total": 0, "success": 0}
        by_language[r.language]["total"] += 1
        if r.success:
            by_language[r.language]["success"] += 1

    return {
        "by_condition": by_condition,
        "by_domain": by_domain,
        "by_language": by_language,
    }


@app.function(timeout=7200)
def run_full_comparison() -> dict:
    """Run full 4-condition comparison across all domains."""
    print("=" * 70)
    print("Ananke: Full 4-Condition Domain Comparison")
    print("=" * 70)

    # Run all 4 conditions
    return run_domain_evaluation.local(
        conditions=[c.value for c in ConstraintCondition],
        domains=None,  # All domains
    )


@app.local_entrypoint()
def main(
    full_comparison: bool = False,
    domain: str | None = None,
):
    """Run Ananke domain evaluation.

    Args:
        full_comparison: Run all 4 conditions
        domain: Specific domain to test (types, imports, controlflow, semantics)
    """
    if full_comparison:
        results = run_full_comparison.remote()
    else:
        domains = [domain] if domain else None
        results = run_domain_evaluation.remote(
            conditions=["unconstrained", "full_ananke"],
            domains=domains,
        )

    print("\nEvaluation complete!")

    # Save results
    timestamp = int(time.time())
    output_file = f"ananke_domain_eval_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
