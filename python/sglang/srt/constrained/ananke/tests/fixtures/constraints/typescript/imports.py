# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Import constraint examples for TypeScript.

This module contains realistic examples of TypeScript import constraints
demonstrating ESM vs CommonJS module resolution, type-only imports, and
path alias resolution patterns.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        ImportBinding,
        ModuleStub,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        ImportBinding,
        ModuleStub,
    )

TYPESCRIPT_IMPORT_EXAMPLES = [
    ConstraintExample(
        id="ts-imports-001",
        name="ESM vs CommonJS Module Resolution",
        description="Enforce proper ESM import syntax for TypeScript modules",
        scenario=(
            "Developer working in a TypeScript project with 'type': 'module' in "
            "package.json, requiring ESM import syntax. Must use 'import' statements "
            "rather than 'require()' and properly handle default vs named exports. "
            "The constraint ensures generated imports follow ESM conventions."
        ),
        prompt="""Write an import statement using ESM syntax (not CommonJS require).
Use "import X from 'module'" for default exports, "import { X } from 'module'"
for named exports, or "import * as X from 'module'" for namespace imports.

""",
        spec=ConstraintSpec(
            language="typescript",
            # Regex enforces ESM import syntax (not require)
            regex=r"^import\s+(?:type\s+)?(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)\s+from\s+['\"]",
            ebnf=r'''
root ::= default_import | named_import | fs_import | namespace_import | type_import
default_import ::= "import express from 'express';"
named_import ::= "import { Router } from 'express';"
fs_import ::= "import { readFile, writeFile } from 'fs/promises';"
namespace_import ::= "import * as fs from 'fs/promises';"
type_import ::= "import type { Express } from 'express';"
''',
            imports=[
                ImportBinding(module="fs", name="promises", alias="fs"),
                ImportBinding(module="express"),
            ],
            available_modules={"fs", "fs/promises", "express", "path", "url"},
            module_stubs={
                "express": ModuleStub(
                    module_name="express",
                    exports={"default": "() => Express", "Router": "() => Router"},
                ),
                "fs/promises": ModuleStub(
                    module_name="fs/promises",
                    exports={
                        "readFile": "(path: string) => Promise<Buffer>",
                        "writeFile": "(path: string, data: string) => Promise<void>",
                    },
                ),
            },
        ),
        expected_effect=(
            "Masks tokens that would generate CommonJS syntax (require, module.exports) "
            "instead of ESM syntax (import, export). Ensures proper handling of default "
            "exports with 'import X from' vs named exports with 'import { X } from'."
        ),
        valid_outputs=[
            "import express from 'express';",
            "import { Router } from 'express';",
            "import { readFile, writeFile } from 'fs/promises';",
            "import * as fs from 'fs/promises';",
            "import type { Express } from 'express';",
        ],
        invalid_outputs=[
            "const express = require('express');",  # CommonJS, not ESM
            "import express = require('express');",  # TypeScript namespace import
            "const { Router } = require('express');",  # CommonJS destructure
            "module.exports = express;",  # CommonJS export
        ],
        tags=["imports", "esm", "commonjs", "module-resolution"],
        language="typescript",
        domain="imports",
    ),
    ConstraintExample(
        id="ts-imports-002",
        name="Type-Only Import Distinction",
        description="Enforce type vs value import separation for tree-shaking",
        scenario=(
            "Developer optimizing bundle size by separating type imports from value "
            "imports. Types should use 'import type' syntax to ensure they're erased "
            "at runtime and don't affect bundle size. This is especially important "
            "when importing large type definition files."
        ),
        prompt="""Write React imports separating types from values. Use "import type { ComponentType }"
for types that are erased at runtime, and "import { useState }" for actual values.
Can also use inline: "import React, { type ComponentType } from 'react'".

""",
        spec=ConstraintSpec(
            language="typescript",
            # Regex enforces proper React imports (type or value, including mixed inline type)
            # Matches: import type { X }, import { X }, import React, { type X }
            regex=r"import\s+(?:type\s+)?(?:\{[^}]*\}|React|type\s+\*\s+as\s+React|\w+,\s*\{[^}]*\})\s+from\s+['\"]react['\"]",
            ebnf=r'''
root ::= type_import | value_import | mixed_import | namespace_import
type_import ::= "import type { ComponentType, FC } from 'react';"
value_import ::= "import { useState, useEffect } from 'react';"
mixed_import ::= "import React, { type ComponentType } from 'react';"
namespace_import ::= "import type * as React from 'react';"
''',
            imports=[
                ImportBinding(module="react", name="ComponentType"),
                ImportBinding(module="react", name="FC"),
            ],
            available_modules={"react", "@types/react"},
            type_aliases={
                "ComponentType": "Type[React.ComponentType<any>]",
                "FC": "Type[React.FunctionComponent<P>]",
            },
            module_stubs={
                "react": ModuleStub(
                    module_name="react",
                    exports={
                        "useState": "<T>(initial: T) => [T, (value: T) => void]",
                        "useEffect": "(effect: () => void, deps?: any[]) => void",
                        "ComponentType": "Type",
                        "FC": "Type",
                    },
                ),
            },
        ),
        expected_effect=(
            "Masks tokens that import types as values or vice versa. Ensures type "
            "imports use 'import type { X }' syntax for tree-shaking. Allows value "
            "imports to use regular 'import { X }' syntax. Blocks mixing type and "
            "value imports in the same statement when isolatedModules is enabled."
        ),
        valid_outputs=[
            "import type { ComponentType, FC } from 'react';",
            "import { useState, useEffect } from 'react';",
            "import React, { type ComponentType } from 'react';",
            "import type * as React from 'react';",
        ],
        invalid_outputs=[
            "import { ComponentType, useState } from 'react';",  # Mixing type and value
            "import type { useState } from 'react';",  # useState is a value
            "import ComponentType from 'react';",  # Not a default export
            "import { type useState } from 'react';",  # useState is not a type
        ],
        tags=["imports", "type-only", "tree-shaking", "optimization"],
        language="typescript",
        domain="imports",
    ),
    ConstraintExample(
        id="ts-imports-003",
        name="Path Alias Resolution",
        description="Resolve TypeScript path aliases from tsconfig.json",
        scenario=(
            "Developer using path aliases configured in tsconfig.json to avoid "
            "relative import hell. Common patterns include '@/' for src root, "
            "'@components/' for components directory. Must ensure imports resolve "
            "to actual file paths and respect the module resolution strategy."
        ),
        prompt="""Write import statements using the @/ path alias configured in tsconfig.json.
Use "@/components/Button" instead of "../../../components/Button".
The @/ alias maps to the src/ directory.

""",
        spec=ConstraintSpec(
            language="typescript",
            # Regex enforces path alias imports with @/ prefix
            regex=r"^import\s+(?:type\s+)?(?:\{[^}]*\}|\*\s+as\s+\w+)\s+from\s+['\"]@/",
            ebnf=r'''
root ::= button_import | format_import | type_import | namespace_import
button_import ::= "import { Button } from '@/components/Button';"
format_import ::= "import { formatDate } from '@/utils/format';"
type_import ::= "import type { ButtonProps } from '@/components/Button';"
namespace_import ::= "import * as utils from '@/utils/format';"
''',
            available_modules={
                "@/components/Button",
                "@/utils/format",
                "@/types/user",
                "src/components/Button",
                "src/utils/format",
            },
            imports=[
                ImportBinding(module="@/components/Button", name="Button"),
                ImportBinding(module="@/utils/format", name="formatDate"),
            ],
            type_aliases={
                "@/*": "src/*",
                "@components/*": "src/components/*",
                "@utils/*": "src/utils/*",
            },
            module_stubs={
                "@/components/Button": ModuleStub(
                    module_name="@/components/Button",
                    exports={"Button": "React.FC<ButtonProps>", "ButtonProps": "Type"},
                ),
                "@/utils/format": ModuleStub(
                    module_name="@/utils/format",
                    exports={"formatDate": "(date: Date) => string"},
                ),
            },
        ),
        expected_effect=(
            "Masks tokens that would create invalid import paths or fail to use "
            "configured aliases. Ensures imports use the @/ alias pattern rather "
            "than relative paths like '../../../components/Button'. Validates that "
            "aliased paths resolve to actual modules."
        ),
        valid_outputs=[
            "import { Button } from '@/components/Button';",
            "import { formatDate } from '@/utils/format';",
            "import type { ButtonProps } from '@/components/Button';",
            "import * as utils from '@/utils/format';",
        ],
        invalid_outputs=[
            "import { Button } from '../../../components/Button';",  # Should use alias
            "import { Button } from 'components/Button';",  # Missing @ prefix
            "import { Button } from '@components/Button';",  # Wrong alias pattern
            "import { Button } from '@/Button';",  # Incorrect path structure
        ],
        tags=["imports", "path-aliases", "module-resolution", "tsconfig"],
        language="typescript",
        domain="imports",
    ),
]
