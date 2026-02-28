# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Import constraint examples for Kotlin.

This module contains realistic examples of import-level constraints that
demonstrate how Ananke's ImportDomain masks tokens to enforce Kotlin's
import requirements, including platform-specific imports, coroutine scope
requirements, and annotation processing.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        ImportBinding,
        ModuleStub,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        ImportBinding,
        ModuleStub,
    )

KOTLIN_IMPORT_EXAMPLES = [
    ConstraintExample(
        id="kt-imports-001",
        name="Platform-Specific Imports (JVM vs JS vs Native)",
        description="Enforce platform-specific import availability",
        scenario=(
            "Developer writing multiplatform Kotlin code that uses platform-specific "
            "APIs. On JVM, java.util.* is available. On JS, kotlin.js.* is available. "
            "On Native, kotlinx.cinterop.* is available. Using the wrong platform "
            "API causes compilation errors in multiplatform projects."
        ),
        prompt="""Write an import statement for a Kotlin/JVM project.
Available packages include java.util.*, java.time.*, kotlin.*, and kotlinx.coroutines.*.
Kotlin/JS packages like kotlin.js.* are not available on JVM.

""",
        spec=ConstraintSpec(
            language="kotlin",
            # Regex enforces JVM-compatible imports (not JS or Native)
            regex=r"^import\s+(?:java\.|kotlin\.|kotlinx\.coroutines)",
            ebnf=r'''
root ::= uuid_import | time_import | flow_import | list_import
uuid_import ::= "import java.util.UUID"
time_import ::= "import java.time.Instant"
flow_import ::= "import kotlinx.coroutines.flow.Flow"
list_import ::= "import kotlin.collections.List"
''',
            available_modules={
                "kotlin.collections",
                "kotlin.text",
                "kotlinx.coroutines",
                "kotlinx.coroutines.flow",
                # JVM-specific
                "java.util",
                "java.io",
                "java.time",
            },
            forbidden_imports={
                "kotlin.js",  # JS platform not available
                "kotlinx.cinterop",  # Native platform not available
                "platform.posix",  # Native-only
            },
        ),
        expected_effect=(
            "Masks import tokens for platform-specific modules not available on "
            "the target platform. Allows JVM imports (java.*) but blocks JS "
            "(kotlin.js.*) and Native (kotlinx.cinterop.*) imports."
        ),
        valid_outputs=[
            "import java.util.UUID",
            "import java.time.Instant",
            "import kotlinx.coroutines.flow.Flow",
            "import kotlin.collections.List",
        ],
        invalid_outputs=[
            "import kotlin.js.Date",  # JS-only
            "import kotlinx.cinterop.CPointer",  # Native-only
            "import platform.posix.getpid",  # Native-only
        ],
        tags=["imports", "multiplatform", "jvm", "js", "native", "kotlin"],
        language="kotlin",
        domain="imports",
    ),
    ConstraintExample(
        id="kt-imports-002",
        name="Coroutine Scope Requirements",
        description="Enforce kotlinx.coroutines imports for suspend functions",
        scenario=(
            "Developer using coroutines features (Flow, CoroutineScope, launch, etc.) "
            "which require kotlinx.coroutines imports. Without the proper imports, "
            "coroutine builders and Flow operators are unresolved references."
        ),
        prompt="""Add the import statement needed to use Flow and its operators in Kotlin.
The kotlinx.coroutines.flow package contains Flow, map, filter, collect.

""",
        spec=ConstraintSpec(
            language="kotlin",
            # Regex enforces kotlinx.coroutines imports
            regex=r"^import\s+kotlinx\.coroutines",
            ebnf=r'''
root ::= flow_import | map_import | scope_import | star_import
flow_import ::= "import kotlinx.coroutines.flow.Flow"
map_import ::= "import kotlinx.coroutines.flow.map"
scope_import ::= "import kotlinx.coroutines.CoroutineScope"
star_import ::= "import kotlinx.coroutines.*"
''',
            available_modules={
                "kotlinx.coroutines",
                "kotlinx.coroutines.flow",
                "kotlin.coroutines",
            },
            module_stubs={
                "kotlinx.coroutines": ModuleStub(
                    module_name="kotlinx.coroutines",
                    exports={
                        "CoroutineScope": "interface",
                        "launch": "function",
                        "async": "function",
                        "runBlocking": "function",
                        "Dispatchers": "object",
                    },
                ),
                "kotlinx.coroutines.flow": ModuleStub(
                    module_name="kotlinx.coroutines.flow",
                    exports={
                        "Flow": "interface",
                        "flow": "function",
                        "collect": "extension",
                        "map": "extension",
                        "filter": "extension",
                    },
                ),
            },
        ),
        expected_effect=(
            "Masks tokens that use coroutine APIs without importing them. "
            "Requires 'import kotlinx.coroutines.flow.Flow' before using Flow, "
            "'import kotlinx.coroutines.launch' before using launch, etc."
        ),
        valid_outputs=[
            "import kotlinx.coroutines.flow.Flow",
            "import kotlinx.coroutines.flow.map",
            "import kotlinx.coroutines.CoroutineScope",
            "import kotlinx.coroutines.*",
        ],
        invalid_outputs=[
            "import java.util.concurrent.Flow",  # Wrong Flow
            "import reactor.core.publisher.Flux",  # Different library
        ],
        tags=["imports", "coroutines", "flow", "async", "kotlin"],
        language="kotlin",
        domain="imports",
    ),
    ConstraintExample(
        id="kt-imports-003",
        name="Annotation Processing Imports",
        description="Enforce annotation processor imports (Room, Serialization, etc.)",
        scenario=(
            "Developer using kotlinx.serialization with @Serializable annotation "
            "which requires specific imports. Annotation processors like Room, "
            "Serialization, Parcelize generate code that depends on correct imports "
            "being present."
        ),
        prompt="""Add the import needed to use @Serializable annotation from kotlinx.serialization.
Don't use java.io.Serializable - we need the Kotlin serialization library.

""",
        spec=ConstraintSpec(
            language="kotlin",
            # Regex enforces kotlinx.serialization imports
            regex=r"^import\s+kotlinx\.serialization",
            ebnf=r'''
root ::= serializable_import | serial_name_import | json_import
serializable_import ::= "import kotlinx.serialization.Serializable"
serial_name_import ::= "import kotlinx.serialization.SerialName"
json_import ::= "import kotlinx.serialization.json.Json"
''',
            available_modules={
                "kotlinx.serialization",
                "kotlinx.serialization.json",
                "kotlinx.serialization.Serializable",
            },
            imports=[
                ImportBinding(
                    module="kotlinx.serialization",
                    name="Serializable",
                ),
            ],
            module_stubs={
                "kotlinx.serialization": ModuleStub(
                    module_name="kotlinx.serialization",
                    exports={
                        "Serializable": "annotation class",
                        "SerialName": "annotation class",
                        "Transient": "annotation class",
                    },
                ),
                "kotlinx.serialization.json": ModuleStub(
                    module_name="kotlinx.serialization.json",
                    exports={
                        "Json": "object",
                        "encodeToString": "function",
                        "decodeFromString": "function",
                    },
                ),
            },
        ),
        expected_effect=(
            "Masks tokens that use serialization annotations without proper imports. "
            "Requires '@Serializable' to have 'import kotlinx.serialization.Serializable'. "
            "Enforces that annotation processor requirements are met."
        ),
        valid_outputs=[
            "import kotlinx.serialization.Serializable",
            "import kotlinx.serialization.SerialName",
            "import kotlinx.serialization.json.Json",
        ],
        invalid_outputs=[
            "import com.google.gson.annotations.Serializable",  # Wrong library
            "import java.io.Serializable",  # Java serialization, not kotlinx
        ],
        tags=["imports", "annotations", "serialization", "codegen", "kotlin"],
        language="kotlin",
        domain="imports",
    ),
]
