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
"""Kotlin import resolution.

This module provides import resolution for Kotlin packages, including
Kotlin standard library and common third-party packages.

References:
    - Kotlin Standard Library: https://kotlinlang.org/api/latest/jvm/stdlib/
    - Kotlinx Libraries: https://github.com/Kotlin/kotlinx.coroutines
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from domains.imports.resolvers.base import (
    ImportResolver,
    ImportResolution,
    ResolvedModule,
    ResolutionStatus,
)


# =============================================================================
# Kotlin Standard Library
# =============================================================================

KOTLIN_STANDARD_LIBRARY: Dict[str, List[str]] = {
    # Core kotlin package
    "kotlin": [
        "Any", "Unit", "Nothing", "Boolean", "Byte", "Short", "Int", "Long",
        "Float", "Double", "Char", "String", "Array",
        "IntArray", "LongArray", "ShortArray", "ByteArray",
        "FloatArray", "DoubleArray", "BooleanArray", "CharArray",
        "UInt", "ULong", "UByte", "UShort",
        "UIntArray", "ULongArray", "UByteArray", "UShortArray",
        "Pair", "Triple", "Comparable", "Comparator",
        "Throwable", "Exception", "Error", "RuntimeException",
        "IllegalArgumentException", "IllegalStateException",
        "IndexOutOfBoundsException", "NullPointerException",
        "NoSuchElementException", "UnsupportedOperationException",
        "NumberFormatException", "ArithmeticException",
        "assert", "check", "checkNotNull", "require", "requireNotNull",
        "error", "TODO", "run", "let", "with", "apply", "also",
        "takeIf", "takeUnless", "repeat", "lazy", "lazyOf",
        "arrayOf", "arrayOfNulls", "emptyArray",
        "intArrayOf", "longArrayOf", "floatArrayOf", "doubleArrayOf",
        "booleanArrayOf", "charArrayOf", "byteArrayOf", "shortArrayOf",
        "println", "print", "readln", "readLine",
        "maxOf", "minOf", "compareValues", "compareValuesBy",
    ],
    "kotlin.collections": [
        "List", "MutableList", "Set", "MutableSet", "Map", "MutableMap",
        "Collection", "MutableCollection", "Iterable", "MutableIterable",
        "Iterator", "MutableIterator", "ListIterator", "MutableListIterator",
        "ArrayList", "HashSet", "LinkedHashSet", "HashMap", "LinkedHashMap",
        "listOf", "listOfNotNull", "mutableListOf", "arrayListOf",
        "setOf", "mutableSetOf", "hashSetOf", "linkedSetOf",
        "mapOf", "mutableMapOf", "hashMapOf", "linkedMapOf",
        "emptyList", "emptySet", "emptyMap",
        "plus", "minus", "contains", "containsAll",
        "isEmpty", "isNotEmpty", "orEmpty",
        "first", "firstOrNull", "last", "lastOrNull",
        "single", "singleOrNull", "elementAt", "elementAtOrNull",
        "find", "findLast", "indexOf", "lastIndexOf",
        "filter", "filterNot", "filterNotNull", "filterIsInstance",
        "map", "mapNotNull", "mapIndexed", "flatMap",
        "forEach", "forEachIndexed", "onEach",
        "reduce", "reduceOrNull", "fold", "foldIndexed",
        "sum", "sumOf", "average", "count",
        "any", "all", "none",
        "sorted", "sortedBy", "sortedDescending", "sortedByDescending",
        "reversed", "shuffled", "distinct", "distinctBy",
        "groupBy", "groupingBy", "partition", "chunked", "windowed",
        "zip", "zipWithNext", "unzip",
        "joinToString", "associate", "associateBy", "associateWith",
        "toList", "toMutableList", "toSet", "toMutableSet",
        "toMap", "toMutableMap", "toTypedArray",
    ],
    "kotlin.sequences": [
        "Sequence", "sequenceOf", "emptySequence", "generateSequence",
        "asSequence", "constrainOnce",
        "filter", "map", "mapNotNull", "flatMap",
        "take", "drop", "takeWhile", "dropWhile",
        "chunked", "windowed", "zipWithNext",
        "distinct", "sorted", "sortedBy",
        "toList", "toSet", "first", "last", "count",
    ],
    "kotlin.text": [
        "String", "StringBuilder", "Appendable", "CharSequence",
        "Regex", "MatchResult", "MatchGroup",
        "buildString", "buildStringBuilder",
        "isBlank", "isNotBlank", "isEmpty", "isNotEmpty",
        "isNullOrBlank", "isNullOrEmpty",
        "trim", "trimStart", "trimEnd",
        "padStart", "padEnd",
        "lowercase", "uppercase", "capitalize", "decapitalize",
        "replace", "replaceFirst", "replaceBefore", "replaceAfter",
        "split", "splitToSequence", "lines", "lineSequence",
        "contains", "startsWith", "endsWith",
        "indexOf", "lastIndexOf", "indexOfAny", "lastIndexOfAny",
        "substring", "substringBefore", "substringAfter",
        "take", "takeLast", "drop", "dropLast",
        "toIntOrNull", "toLongOrNull", "toFloatOrNull", "toDoubleOrNull",
        "toInt", "toLong", "toFloat", "toDouble",
        "toCharArray", "toByteArray",
    ],
    "kotlin.ranges": [
        "IntRange", "LongRange", "CharRange",
        "IntProgression", "LongProgression", "CharProgression",
        "ClosedRange", "OpenEndRange",
        "rangeTo", "rangeUntil", "downTo", "step",
        "coerceIn", "coerceAtLeast", "coerceAtMost",
    ],
    "kotlin.math": [
        "PI", "E",
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "hypot", "sqrt", "cbrt", "exp", "expm1",
        "log", "log10", "log2", "ln", "ln1p",
        "ceil", "floor", "truncate", "round", "roundToInt", "roundToLong",
        "abs", "sign", "min", "max",
        "pow", "withSign", "IEEErem",
    ],
    "kotlin.io": [
        "print", "println", "readln", "readLine",
        "File", "BufferedReader", "BufferedWriter",
        "use", "useLines", "forEachLine",
        "readText", "readLines", "readBytes",
        "writeText", "writeBytes", "appendText", "appendBytes",
    ],
    "kotlin.random": [
        "Random", "nextInt", "nextLong", "nextDouble", "nextFloat",
        "nextBoolean", "nextBytes",
    ],
    "kotlin.time": [
        "Duration", "DurationUnit",
        "days", "hours", "minutes", "seconds", "milliseconds",
        "microseconds", "nanoseconds",
        "measureTime", "measureTimedValue", "TimedValue",
        "TimeSource", "TimeMark",
    ],
    "kotlin.reflect": [
        "KClass", "KType", "KFunction", "KProperty",
        "KMutableProperty", "KParameter", "KTypeParameter",
        "typeOf", "createInstance",
    ],
}

# Kotlinx Coroutines
KOTLINX_COROUTINES: Dict[str, List[str]] = {
    "kotlinx.coroutines": [
        "CoroutineScope", "CoroutineContext", "CoroutineDispatcher",
        "Job", "SupervisorJob", "Deferred",
        "launch", "async", "runBlocking",
        "coroutineScope", "supervisorScope", "withContext",
        "delay", "yield", "isActive", "ensureActive",
        "cancel", "cancelAndJoin", "cancelChildren",
        "Dispatchers", "MainScope", "GlobalScope",
        "CancellationException", "TimeoutCancellationException",
        "withTimeout", "withTimeoutOrNull",
        "awaitAll", "joinAll",
        "CoroutineExceptionHandler",
    ],
    "kotlinx.coroutines.flow": [
        "Flow", "StateFlow", "SharedFlow", "MutableStateFlow", "MutableSharedFlow",
        "flow", "flowOf", "emptyFlow", "asFlow",
        "channelFlow", "callbackFlow",
        "collect", "collectLatest", "first", "firstOrNull", "single",
        "toList", "toSet",
        "map", "mapNotNull", "filter", "filterNotNull",
        "transform", "flatMapConcat", "flatMapMerge", "flatMapLatest",
        "take", "drop", "takeWhile", "dropWhile",
        "onEach", "onStart", "onCompletion", "catch",
        "stateIn", "shareIn", "launchIn",
        "combine", "zip", "merge",
        "debounce", "sample", "conflate", "buffer",
        "distinctUntilChanged", "distinctUntilChangedBy",
        "emit", "emitAll",
    ],
    "kotlinx.coroutines.channels": [
        "Channel", "SendChannel", "ReceiveChannel",
        "BroadcastChannel", "ConflatedBroadcastChannel",
        "produce", "actor", "ticker",
        "BufferOverflow", "ChannelResult",
    ],
    "kotlinx.coroutines.sync": [
        "Mutex", "Semaphore", "withLock", "withPermit",
    ],
}

# Kotlinx Serialization
KOTLINX_SERIALIZATION: Dict[str, List[str]] = {
    "kotlinx.serialization": [
        "Serializable", "Serializer", "KSerializer",
        "Contextual", "Transient", "Required",
        "SerialName", "SerialInfo",
        "encodeToString", "decodeFromString",
    ],
    "kotlinx.serialization.json": [
        "Json", "JsonElement", "JsonObject", "JsonArray",
        "JsonPrimitive", "JsonNull",
        "jsonObject", "jsonArray", "buildJsonObject", "buildJsonArray",
        "JsonBuilder", "JsonConfiguration",
    ],
}

# Popular third-party libraries
KOTLIN_POPULAR_PACKAGES: Dict[str, List[str]] = {
    "io.ktor.client": [
        "HttpClient", "HttpClientEngine", "HttpClientConfig",
        "HttpRequestBuilder", "HttpResponse", "HttpStatement",
        "get", "post", "put", "delete", "patch", "head", "options",
    ],
    "io.ktor.server": [
        "Application", "ApplicationCall", "ApplicationEngine",
        "embeddedServer", "Netty", "CIO", "Jetty",
        "routing", "route", "get", "post", "put", "delete",
    ],
    "org.jetbrains.exposed": [
        "Database", "Table", "Column", "EntityID",
        "select", "selectAll", "insert", "update", "delete",
        "transaction", "SchemaUtils",
    ],
    "com.squareup.okhttp3": [
        "OkHttpClient", "Request", "Response", "Call",
        "RequestBody", "ResponseBody", "MediaType",
        "Headers", "HttpUrl", "Interceptor",
    ],
    "com.google.gson": [
        "Gson", "GsonBuilder", "JsonElement", "JsonObject", "JsonArray",
        "TypeAdapter", "TypeToken",
        "SerializedName", "Expose",
    ],
    "org.junit.jupiter.api": [
        "Test", "BeforeEach", "AfterEach", "BeforeAll", "AfterAll",
        "Assertions", "assertEquals", "assertTrue", "assertFalse",
        "assertNull", "assertNotNull", "assertThrows",
        "DisplayName", "Nested", "Disabled", "Tag",
    ],
    "io.mockk": [
        "mockk", "spyk", "slot", "every", "verify", "coEvery", "coVerify",
        "just", "runs", "returns", "returnsMany", "throws",
        "MockKAnnotations", "MockK", "SpyK", "InjectMockKs",
    ],
    "arrow.core": [
        "Either", "Left", "Right", "Option", "Some", "None",
        "Validated", "Nel", "Ior",
        "raise", "recover", "fold", "getOrElse", "getOrHandle",
    ],
}


# =============================================================================
# Kotlin Import Resolver
# =============================================================================

class KotlinImportResolver(ImportResolver):
    """Kotlin import resolver."""

    @property
    def language(self) -> str:
        return "kotlin"

    def resolve(self, import_path: str) -> ImportResolution:
        """Resolve a Kotlin import path."""
        # Check standard library
        if import_path in KOTLIN_STANDARD_LIBRARY:
            exports = set(KOTLIN_STANDARD_LIBRARY[import_path])
            return ImportResolution(
                status=ResolutionStatus.RESOLVED,
                module=ResolvedModule(
                    name=import_path,
                    path=import_path,
                    exports=exports,
                    is_builtin=True,
                    is_available=True,
                ),
                module_name=import_path,
                exports=exports,
            )

        # Check kotlinx.coroutines
        if import_path in KOTLINX_COROUTINES:
            exports = set(KOTLINX_COROUTINES[import_path])
            return ImportResolution(
                status=ResolutionStatus.RESOLVED,
                module=ResolvedModule(
                    name=import_path,
                    path=import_path,
                    exports=exports,
                    is_builtin=True,
                    is_available=True,
                ),
                module_name=import_path,
                exports=exports,
            )

        # Check kotlinx.serialization
        if import_path in KOTLINX_SERIALIZATION:
            exports = set(KOTLINX_SERIALIZATION[import_path])
            return ImportResolution(
                status=ResolutionStatus.RESOLVED,
                module=ResolvedModule(
                    name=import_path,
                    path=import_path,
                    exports=exports,
                    is_builtin=True,
                    is_available=True,
                ),
                module_name=import_path,
                exports=exports,
            )

        # Check popular packages
        if import_path in KOTLIN_POPULAR_PACKAGES:
            exports = set(KOTLIN_POPULAR_PACKAGES[import_path])
            return ImportResolution(
                status=ResolutionStatus.RESOLVED,
                module=ResolvedModule(
                    name=import_path,
                    path=import_path,
                    exports=exports,
                    is_builtin=False,
                    is_available=True,
                ),
                module_name=import_path,
                exports=exports,
            )

        # Check for wildcard or subpackage
        # In Kotlin, you can import a.b.* or a.b.SomeClass
        base_pkg = import_path.rsplit(".", 1)[0] if "." in import_path else import_path
        all_pkgs = {
            **KOTLIN_STANDARD_LIBRARY,
            **KOTLINX_COROUTINES,
            **KOTLINX_SERIALIZATION,
            **KOTLIN_POPULAR_PACKAGES,
        }

        for pkg in all_pkgs:
            if pkg.startswith(base_pkg) or base_pkg.startswith(pkg):
                return ImportResolution(
                    status=ResolutionStatus.PARTIAL,
                    module=ResolvedModule(
                        name=import_path,
                        path=import_path,
                        exports=set(),
                        is_builtin=False,
                        is_available=True,
                    ),
                    module_name=import_path,
                    exports=set(),
                )

        # Unknown package
        return ImportResolution(
            status=ResolutionStatus.PARTIAL,
            module=ResolvedModule(
                name=import_path,
                path=import_path,
                exports=set(),
                is_builtin=False,
                is_available=True,
            ),
            module_name=import_path,
            exports=set(),
        )

    def get_module_exports(self, import_path: str) -> List[str]:
        """Get exports for a module."""
        if import_path in KOTLIN_STANDARD_LIBRARY:
            return KOTLIN_STANDARD_LIBRARY[import_path]
        if import_path in KOTLINX_COROUTINES:
            return KOTLINX_COROUTINES[import_path]
        if import_path in KOTLINX_SERIALIZATION:
            return KOTLINX_SERIALIZATION[import_path]
        if import_path in KOTLIN_POPULAR_PACKAGES:
            return KOTLIN_POPULAR_PACKAGES[import_path]
        return []

    def get_completion_candidates(self, prefix: str) -> List[str]:
        """Get import path completion candidates."""
        candidates = []
        all_pkgs = {
            **KOTLIN_STANDARD_LIBRARY,
            **KOTLINX_COROUTINES,
            **KOTLINX_SERIALIZATION,
            **KOTLIN_POPULAR_PACKAGES,
        }

        for pkg in all_pkgs:
            if pkg.startswith(prefix):
                candidates.append(pkg)

        return sorted(candidates)

    def is_available(self, module_name: str) -> bool:
        """Check if a Kotlin module is available."""
        return module_name in (
            KOTLIN_STANDARD_LIBRARY |
            KOTLINX_COROUTINES |
            KOTLINX_SERIALIZATION |
            KOTLIN_POPULAR_PACKAGES
        )

    def get_version(self) -> Optional[str]:
        """Get Kotlin version."""
        return "2.0"  # Current stable version


def create_kotlin_resolver() -> KotlinImportResolver:
    """Factory function to create a Kotlin resolver."""
    return KotlinImportResolver()
