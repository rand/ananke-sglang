# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Type domain for incremental bidirectional type checking.

This domain implements a full type system based on Hazel research:
- Marked lambda calculus for error localization
- Order maintenance for incremental updates
- Bidirectional type synthesis and analysis

Key Components:
- constraint: Type hierarchy and TypeConstraint
- domain: TypeDomain implementing ConstraintDomain
- unification: Robinson's unification algorithm
- environment: Immutable type environment
- marking: Marks, provenances, MarkedAST
- bidirectional: Synthesis and analysis
"""

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from .constraint import (
        # Type hierarchy
        Type,
        TypeVar,
        PrimitiveType,
        FunctionType,
        ListType,
        DictType,
        TupleType,
        SetType,
        UnionType,
        ClassType,
        AnyType,
        NeverType,
        HoleType,
        # Primitive type singletons
        INT,
        STR,
        BOOL,
        FLOAT,
        NONE,
        BYTES,
        ANY,
        NEVER,
        # Type equations
        TypeEquation,
        # Type constraint
        TypeConstraint,
        TYPE_TOP,
        TYPE_BOTTOM,
        # Factory functions
        OptionalType,
        type_expecting,
        type_with_equation,
    )
    from .domain import (
        TypeDomain,
        TypeDomainCheckpoint,
    )
    from .unification import (
        Substitution,
        EMPTY_SUBSTITUTION,
        UnificationResult,
        UnificationError,
        unify,
        occurs_check,
        solve_equations,
        unify_types,
    )
    from .environment import (
        TypeEnvironment,
        TypeEnvironmentSnapshot,
        EMPTY_ENVIRONMENT,
        create_environment,
        merge_environments,
    )
except ImportError:
    from domains.types.constraint import (
        # Type hierarchy
        Type,
        TypeVar,
        PrimitiveType,
        FunctionType,
        ListType,
        DictType,
        TupleType,
        SetType,
        UnionType,
        ClassType,
        AnyType,
        NeverType,
        HoleType,
        # Primitive type singletons
        INT,
        STR,
        BOOL,
        FLOAT,
        NONE,
        BYTES,
        ANY,
        NEVER,
        # Type equations
        TypeEquation,
        # Type constraint
        TypeConstraint,
        TYPE_TOP,
        TYPE_BOTTOM,
        # Factory functions
        OptionalType,
        type_expecting,
        type_with_equation,
    )
    from domains.types.domain import (
        TypeDomain,
        TypeDomainCheckpoint,
    )
    from domains.types.unification import (
        Substitution,
        EMPTY_SUBSTITUTION,
        UnificationResult,
        UnificationError,
        unify,
        occurs_check,
        solve_equations,
        unify_types,
    )
    from domains.types.environment import (
        TypeEnvironment,
        TypeEnvironmentSnapshot,
        EMPTY_ENVIRONMENT,
        create_environment,
        merge_environments,
    )

__all__ = [
    # Type hierarchy
    "Type",
    "TypeVar",
    "PrimitiveType",
    "FunctionType",
    "ListType",
    "DictType",
    "TupleType",
    "SetType",
    "UnionType",
    "ClassType",
    "AnyType",
    "NeverType",
    "HoleType",
    # Primitive type singletons
    "INT",
    "STR",
    "BOOL",
    "FLOAT",
    "NONE",
    "BYTES",
    "ANY",
    "NEVER",
    # Type equations
    "TypeEquation",
    # Type constraint
    "TypeConstraint",
    "TYPE_TOP",
    "TYPE_BOTTOM",
    # Factory functions
    "OptionalType",
    "type_expecting",
    "type_with_equation",
    # Domain
    "TypeDomain",
    "TypeDomainCheckpoint",
    # Unification
    "Substitution",
    "EMPTY_SUBSTITUTION",
    "UnificationResult",
    "UnificationError",
    "unify",
    "occurs_check",
    "solve_equations",
    "unify_types",
    # Environment
    "TypeEnvironment",
    "TypeEnvironmentSnapshot",
    "EMPTY_ENVIRONMENT",
    "create_environment",
    "merge_environments",
]
