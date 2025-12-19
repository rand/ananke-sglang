# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Deep dive example: Dataclass with field validation.

This module provides a comprehensive example that combines multiple constraint
domains to demonstrate how Ananke handles complex, real-world scenarios. This
example shows a dataclass with field validators, post-init validation, and
type constraints working together.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        ClassDefinition,
        SemanticConstraint,
        ImportBinding,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        FunctionSignature,
        ClassDefinition,
        SemanticConstraint,
        ImportBinding,
    )

# This is a single deep-dive example that combines multiple domains
PYTHON_DATACLASS_VALIDATION_EXAMPLE = ConstraintExample(
    id="py-deep-001",
    name="Validated User Dataclass",
    description="Dataclass with field validators and __post_init__ validation",
    scenario=(
        "Developer creating a User dataclass with validation constraints:\n"
        "1. Email must match regex pattern\n"
        "2. Age must be between 0 and 150 (semantic constraint)\n"
        "3. Username must be 3-20 alphanumeric characters\n"
        "4. Created_at must use datetime.datetime type\n"
        "5. __post_init__ validates all fields and raises on invalid data\n\n"
        "This example demonstrates cross-domain constraint propagation where:\n"
        "- TypeDomain ensures proper dataclass field types\n"
        "- ImportDomain provides dataclasses and datetime modules\n"
        "- SemanticDomain enforces age range invariants\n"
        "- SyntaxDomain validates email and username patterns\n"
        "- ControlFlowDomain ensures __post_init__ properly validates"
    ),
    prompt="""Create a User dataclass with validation in __post_init__.
Requirements: username (3-20 chars), email (valid format), age (0-150), created_at (datetime).
Raise ValueError for invalid data.

from dataclasses import dataclass
from datetime import datetime
import re

""",
    spec=ConstraintSpec(
        language="python",
        # EBNF enforces @dataclass with typed fields and __post_init__ validation
        ebnf=r'''
root ::= "@dataclass\nclass User:\n" fields "\n\n    def __post_init__(self):\n" body
fields ::= field ("\n" field)*
field ::= "    " identifier ": " type_expr (" = " [^\n]+)?
identifier ::= [a-zA-Z_]+
type_expr ::= "str" | "int" | "datetime"
body ::= [^\x00]+
''',
        # Import constraints: need dataclasses, datetime, and re
        imports=[
            ImportBinding(module="dataclasses", name="dataclass"),
            ImportBinding(module="dataclasses", name="field"),
            ImportBinding(module="datetime", name="datetime"),
            ImportBinding(module="re"),
        ],
        available_modules={"dataclasses", "datetime", "re", "typing"},
        # Type constraints: define the User class structure
        class_definitions=[
            ClassDefinition(
                name="User",
                bases=(),
                instance_vars=(
                    TypeBinding(name="username", type_expr="str"),
                    TypeBinding(name="email", type_expr="str"),
                    TypeBinding(name="age", type_expr="int"),
                    TypeBinding(name="created_at", type_expr="datetime"),
                ),
                methods=(
                    FunctionSignature(
                        name="__post_init__",
                        params=(TypeBinding(name="self", type_expr="User"),),
                        return_type="None",
                    ),
                ),
            )
        ],
        # Semantic constraints: age must be valid
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="0 <= self.age <= 150",
                scope="User",
                variables=("self.age",),
            ),
            SemanticConstraint(
                kind="precondition",
                expression="len(self.username) >= 3 and len(self.username) <= 20",
                scope="User.__post_init__",
                variables=("self.username",),
            ),
        ],
        # Type bindings for field validation
        type_bindings=[
            TypeBinding(name="self", type_expr="User", scope="class:User"),
            TypeBinding(name="username", type_expr="str", scope="class:User"),
            TypeBinding(name="email", type_expr="str", scope="class:User"),
            TypeBinding(name="age", type_expr="int", scope="class:User"),
            TypeBinding(name="created_at", type_expr="datetime", scope="class:User"),
        ],
    ),
    expected_effect=(
        "Combined masking from all domains:\n"
        "1. TypeDomain masks non-str for username/email, non-int for age, non-datetime for created_at\n"
        "2. ImportDomain ensures dataclasses/datetime imports are available\n"
        "3. SemanticDomain masks age assignments outside 0-150 range\n"
        "4. SyntaxDomain (via regex in validation) ensures email format\n"
        "5. ControlFlowDomain ensures __post_init__ raises ValueError on validation failure\n\n"
        "The domains propagate constraints to each other:\n"
        "- Type constraints inform semantic checks (age is int, so numeric comparisons work)\n"
        "- Semantic constraints refine type domain (age is not just int, but 0-150)\n"
        "- Import availability affects what validators can be written"
    ),
    valid_outputs=[
        """@dataclass
class User:
    username: str
    email: str
    age: int
    created_at: datetime

    def __post_init__(self):
        if not (3 <= len(self.username) <= 20):
            raise ValueError("Username must be 3-20 characters")
        if not re.match(r'^[a-zA-Z0-9._+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,6}$', self.email):
            raise ValueError("Invalid email format")
        if not (0 <= self.age <= 150):
            raise ValueError("Age must be between 0 and 150")""",
        """@dataclass
class User:
    username: str
    email: str
    age: int
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        # Validate username length
        if len(self.username) < 3 or len(self.username) > 20:
            raise ValueError(f"Username '{self.username}' must be 3-20 characters")

        # Validate email pattern
        email_pattern = r'^[a-zA-Z0-9._+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,6}$'
        if not re.match(email_pattern, self.email):
            raise ValueError(f"Invalid email: {self.email}")

        # Validate age range
        if self.age < 0 or self.age > 150:
            raise ValueError(f"Age {self.age} is out of valid range (0-150)")""",
    ],
    invalid_outputs=[
        # Type violations
        """@dataclass
class User:
    username: int  # Wrong type
    email: str
    age: int
    created_at: datetime""",
        # Missing validation in __post_init__
        """@dataclass
class User:
    username: str
    email: str
    age: int
    created_at: datetime

    def __post_init__(self):
        pass  # No validation""",
        # Semantic constraint violation
        """@dataclass
class User:
    username: str
    email: str
    age: int
    created_at: datetime

    def __post_init__(self):
        if self.age < 0 or self.age > 200:  # Wrong upper bound
            raise ValueError("Invalid age")""",
        # Missing required imports
        """# No dataclass import
class User:
    def __init__(self, username: str, email: str, age: int, created_at: datetime):
        self.username = username
        self.email = email
        self.age = age
        self.created_at = created_at""",
        # Insufficient username validation
        """@dataclass
class User:
    username: str
    email: str
    age: int
    created_at: datetime

    def __post_init__(self):
        if len(self.username) < 3:  # Missing upper bound check
            raise ValueError("Username too short")""",
    ],
    tags=[
        "types",
        "semantics",
        "syntax",
        "imports",
        "dataclass",
        "validation",
        "cross-domain",
        "deep-dive",
    ],
    language="python",
    domain="types",  # Primary domain, but uses all domains
)

# Export as list for consistency with other modules
PYTHON_DEEP_DIVE_EXAMPLES = [PYTHON_DATACLASS_VALIDATION_EXAMPLE]
