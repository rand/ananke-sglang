# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Import constraint examples for Python.

This module contains realistic examples of import-level constraints that
demonstrate how Ananke's ImportDomain masks tokens to enforce module
availability and security policies during code generation.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        ImportBinding,
        TypeBinding,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        ImportBinding,
        TypeBinding,
    )

PYTHON_IMPORT_EXAMPLES = [
    ConstraintExample(
        id="py-imports-001",
        name="Security Sandbox Environment",
        description="Forbid dangerous imports in sandboxed execution",
        scenario=(
            "Developer writing code for a sandboxed environment where filesystem, "
            "subprocess, and system operations are forbidden for security. The "
            "generator must not produce imports like os, subprocess, sys, __import__, "
            "or eval/exec calls."
        ),
        prompt="""Write imports for a sandboxed environment. Only safe modules allowed: json, math, datetime, re, typing, dataclasses.
No os, sys, subprocess, or filesystem access.

""",
        spec=ConstraintSpec(
            language="python",
            # Regex enforces only allowed module imports
            # Pattern matches: import <allowed> or from <allowed> import ...
            regex=r"^(?:import|from)\s+(?:json|math|datetime|re|typing|dataclasses)(?:\s|$|\.)",
            forbidden_imports={
                "os",
                "sys",
                "subprocess",
                "shutil",
                "__builtin__",
                "builtins",
                "importlib",
            },
            available_modules={
                "json",
                "math",
                "datetime",
                "re",
                "typing",
                "dataclasses",
            },
        ),
        expected_effect=(
            "Masks tokens that would import forbidden modules. Blocks 'import os', "
            "'from subprocess import run', 'import sys', etc. Also blocks dynamic "
            "import patterns like __import__() and importlib.import_module()."
        ),
        valid_outputs=[
            "import json",
            "from typing import List, Dict",
            "import math",
            "from dataclasses import dataclass",
        ],
        invalid_outputs=[
            "import os",
            "from subprocess import run",
            "import sys",
            "from os.path import exists",
            "__import__('os')",
            "import importlib; importlib.import_module('subprocess')",
        ],
        tags=["imports", "security", "sandbox", "policy"],
        language="python",
        domain="imports",
    ),
    ConstraintExample(
        id="py-imports-002",
        name="ML Environment Compatibility",
        description="Ensure numpy compatibility, forbid tensorflow for version conflicts",
        scenario=(
            "Developer writing code for an ML environment that uses numpy 2.x. "
            "TensorFlow is forbidden because it requires numpy 1.x and would create "
            "version conflicts. PyTorch is available and compatible."
        ),
        prompt="""Write ML imports for a numpy 2.x environment. Available: numpy, scipy, sklearn, pandas, torch, matplotlib.
Do not use tensorflow or keras (version conflict with numpy 2.x).

""",
        spec=ConstraintSpec(
            language="python",
            # Regex allows ML modules but blocks tensorflow/keras
            # Matches numpy, scipy, sklearn, pandas, torch, matplotlib imports
            regex=r"^(?:import|from)\s+(?:numpy|scipy|sklearn|pandas|torch|matplotlib)(?:\s|$|\.)|^(?:arr|data|model)\s*=",
            available_modules={
                "numpy",
                "scipy",
                "sklearn",
                "pandas",
                "torch",
                "matplotlib",
            },
            forbidden_imports={
                "tensorflow",
                "tf",
                "keras",
            },
            imports=[
                ImportBinding(module="numpy", alias="np"),
                ImportBinding(module="torch", alias="torch"),
            ],
        ),
        expected_effect=(
            "Masks tokens that would import tensorflow or keras. Allows numpy, "
            "scipy, torch, and other compatible packages. Ensures that aliases "
            "like 'np' are recognized for numpy operations."
        ),
        valid_outputs=[
            "import numpy as np",
            "from scipy import optimize",
            "import torch",
            "from sklearn.linear_model import LinearRegression",
            "arr = np.array([1, 2, 3])",
        ],
        invalid_outputs=[
            "import tensorflow as tf",
            "from tensorflow import keras",
            "import tf",
            "from keras.models import Sequential",
        ],
        tags=["imports", "ml", "compatibility", "environment"],
        language="python",
        domain="imports",
    ),
    ConstraintExample(
        id="py-imports-003",
        name="TYPE_CHECKING Import Pattern",
        description="Type-only imports to avoid circular dependencies",
        scenario=(
            "Developer using the TYPE_CHECKING pattern to import types only for "
            "type annotations, avoiding circular import issues at runtime. Imports "
            "inside if TYPE_CHECKING: block should only be used in annotations, "
            "not runtime code."
        ),
        prompt="""Use the TYPE_CHECKING pattern to import types without circular dependency issues.
Import User from 'module' for annotation only - don't use it at runtime.

from typing import TYPE_CHECKING

""",
        spec=ConstraintSpec(
            language="python",
            # EBNF enforces TYPE_CHECKING pattern: imports in guard, types in annotations
            # Note: Uses ws rule for newlines to work with Lark conversion
            ebnf=r'''
root ::= if_line from_line nls def_block
if_line ::= "if TYPE_CHECKING:" nl
from_line ::= indent "from " module " import " name
module ::= [a-zA-Z_]+
name ::= [a-zA-Z_]+
def_block ::= func_def | class_block
func_def ::= "def process(user: 'User') -> None: pass"
class_block ::= class_line parent_line
class_line ::= "class Tree:" nl
parent_line ::= indent "parent: Optional['Node']"
indent ::= "    "
nl ::= "\n"
nls ::= "\n"+
''',
            imports=[
                ImportBinding(module="typing", name="TYPE_CHECKING"),
                ImportBinding(module="typing", name="Optional"),
            ],
            type_bindings=[
                TypeBinding(name="TYPE_CHECKING", type_expr="bool", scope="global"),
            ],
        ),
        expected_effect=(
            "Masks runtime usage of type-only imports. Allows ForwardRef types "
            "in annotations but blocks instantiation or runtime checks. Inside "
            "if TYPE_CHECKING: block, imports are marked as annotation-only."
        ),
        valid_outputs=[
            "if TYPE_CHECKING:\n    from module import User\n\ndef process(user: 'User') -> None: pass",
            "if TYPE_CHECKING:\n    from circular import Node\n\nclass Tree:\n    parent: Optional['Node']",
        ],
        invalid_outputs=[
            "if TYPE_CHECKING:\n    from module import User\n\nuser = User()",  # Runtime use
            "if TYPE_CHECKING:\n    from module import helper\n\nhelper()",  # Runtime call
        ],
        tags=["imports", "type-checking", "circular-deps", "annotations"],
        language="python",
        domain="imports",
    ),
]
