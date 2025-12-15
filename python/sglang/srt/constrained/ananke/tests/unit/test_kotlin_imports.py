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
"""Tests for Kotlin import resolver."""

import pytest

from domains.imports.resolvers.base import ResolutionStatus
from domains.imports.resolvers.kotlin import (
    KotlinImportResolver,
    create_kotlin_resolver,
    KOTLIN_STANDARD_LIBRARY,
    KOTLINX_COROUTINES,
    KOTLINX_SERIALIZATION,
    KOTLIN_POPULAR_PACKAGES,
)


class TestKotlinImportResolverCreation:
    """Tests for KotlinImportResolver creation."""

    def test_create_resolver(self):
        """Should create a Kotlin import resolver."""
        resolver = KotlinImportResolver()
        assert resolver.language == "kotlin"

    def test_factory_function(self):
        """Should create resolver via factory."""
        resolver = create_kotlin_resolver()
        assert isinstance(resolver, KotlinImportResolver)
        assert resolver.language == "kotlin"

    def test_get_version(self):
        """Should return Kotlin version."""
        resolver = KotlinImportResolver()
        version = resolver.get_version()
        assert version is not None
        assert version == "2.0"


class TestKotlinStandardLibrary:
    """Tests for Kotlin standard library resolution."""

    @pytest.fixture
    def resolver(self):
        return KotlinImportResolver()

    def test_resolve_kotlin_package(self, resolver):
        """Should resolve kotlin package."""
        result = resolver.resolve("kotlin")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.module is not None
        assert result.module.is_builtin is True
        assert "Int" in result.exports
        assert "String" in result.exports
        assert "println" in result.exports

    def test_resolve_kotlin_collections(self, resolver):
        """Should resolve kotlin.collections package."""
        result = resolver.resolve("kotlin.collections")
        assert result.status == ResolutionStatus.RESOLVED
        assert "List" in result.exports
        assert "MutableList" in result.exports
        assert "Map" in result.exports
        assert "listOf" in result.exports

    def test_resolve_kotlin_sequences(self, resolver):
        """Should resolve kotlin.sequences package."""
        result = resolver.resolve("kotlin.sequences")
        assert result.status == ResolutionStatus.RESOLVED
        assert "Sequence" in result.exports
        assert "sequenceOf" in result.exports

    def test_resolve_kotlin_text(self, resolver):
        """Should resolve kotlin.text package."""
        result = resolver.resolve("kotlin.text")
        assert result.status == ResolutionStatus.RESOLVED
        assert "StringBuilder" in result.exports
        assert "Regex" in result.exports

    def test_resolve_kotlin_math(self, resolver):
        """Should resolve kotlin.math package."""
        result = resolver.resolve("kotlin.math")
        assert result.status == ResolutionStatus.RESOLVED
        assert "PI" in result.exports
        assert "sin" in result.exports
        assert "cos" in result.exports

    def test_resolve_kotlin_io(self, resolver):
        """Should resolve kotlin.io package."""
        result = resolver.resolve("kotlin.io")
        assert result.status == ResolutionStatus.RESOLVED
        assert "print" in result.exports
        assert "println" in result.exports

    def test_resolve_kotlin_time(self, resolver):
        """Should resolve kotlin.time package."""
        result = resolver.resolve("kotlin.time")
        assert result.status == ResolutionStatus.RESOLVED
        assert "Duration" in result.exports
        assert "measureTime" in result.exports

    def test_resolve_kotlin_reflect(self, resolver):
        """Should resolve kotlin.reflect package."""
        result = resolver.resolve("kotlin.reflect")
        assert result.status == ResolutionStatus.RESOLVED
        assert "KClass" in result.exports
        assert "KType" in result.exports


class TestKotlinxCoroutines:
    """Tests for kotlinx.coroutines resolution."""

    @pytest.fixture
    def resolver(self):
        return KotlinImportResolver()

    def test_resolve_coroutines(self, resolver):
        """Should resolve kotlinx.coroutines package."""
        result = resolver.resolve("kotlinx.coroutines")
        assert result.status == ResolutionStatus.RESOLVED
        assert "CoroutineScope" in result.exports
        assert "launch" in result.exports
        assert "async" in result.exports
        assert "delay" in result.exports

    def test_resolve_coroutines_flow(self, resolver):
        """Should resolve kotlinx.coroutines.flow package."""
        result = resolver.resolve("kotlinx.coroutines.flow")
        assert result.status == ResolutionStatus.RESOLVED
        assert "Flow" in result.exports
        assert "StateFlow" in result.exports
        assert "collect" in result.exports

    def test_resolve_coroutines_channels(self, resolver):
        """Should resolve kotlinx.coroutines.channels package."""
        result = resolver.resolve("kotlinx.coroutines.channels")
        assert result.status == ResolutionStatus.RESOLVED
        assert "Channel" in result.exports

    def test_resolve_coroutines_sync(self, resolver):
        """Should resolve kotlinx.coroutines.sync package."""
        result = resolver.resolve("kotlinx.coroutines.sync")
        assert result.status == ResolutionStatus.RESOLVED
        assert "Mutex" in result.exports


class TestKotlinxSerialization:
    """Tests for kotlinx.serialization resolution."""

    @pytest.fixture
    def resolver(self):
        return KotlinImportResolver()

    def test_resolve_serialization(self, resolver):
        """Should resolve kotlinx.serialization package."""
        result = resolver.resolve("kotlinx.serialization")
        assert result.status == ResolutionStatus.RESOLVED
        assert "Serializable" in result.exports

    def test_resolve_serialization_json(self, resolver):
        """Should resolve kotlinx.serialization.json package."""
        result = resolver.resolve("kotlinx.serialization.json")
        assert result.status == ResolutionStatus.RESOLVED
        assert "Json" in result.exports
        assert "JsonElement" in result.exports


class TestKotlinPopularPackages:
    """Tests for popular Kotlin packages."""

    @pytest.fixture
    def resolver(self):
        return KotlinImportResolver()

    def test_resolve_ktor_client(self, resolver):
        """Should resolve io.ktor.client package."""
        result = resolver.resolve("io.ktor.client")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.module.is_builtin is False
        assert "HttpClient" in result.exports

    def test_resolve_ktor_server(self, resolver):
        """Should resolve io.ktor.server package."""
        result = resolver.resolve("io.ktor.server")
        assert result.status == ResolutionStatus.RESOLVED
        assert "Application" in result.exports

    def test_resolve_exposed(self, resolver):
        """Should resolve org.jetbrains.exposed package."""
        result = resolver.resolve("org.jetbrains.exposed")
        assert result.status == ResolutionStatus.RESOLVED
        assert "Database" in result.exports

    def test_resolve_okhttp(self, resolver):
        """Should resolve com.squareup.okhttp3 package."""
        result = resolver.resolve("com.squareup.okhttp3")
        assert result.status == ResolutionStatus.RESOLVED
        assert "OkHttpClient" in result.exports

    def test_resolve_junit(self, resolver):
        """Should resolve org.junit.jupiter.api package."""
        result = resolver.resolve("org.junit.jupiter.api")
        assert result.status == ResolutionStatus.RESOLVED
        assert "Test" in result.exports
        assert "BeforeEach" in result.exports

    def test_resolve_mockk(self, resolver):
        """Should resolve io.mockk package."""
        result = resolver.resolve("io.mockk")
        assert result.status == ResolutionStatus.RESOLVED
        assert "mockk" in result.exports

    def test_resolve_arrow(self, resolver):
        """Should resolve arrow.core package."""
        result = resolver.resolve("arrow.core")
        assert result.status == ResolutionStatus.RESOLVED
        assert "Either" in result.exports
        assert "Option" in result.exports


class TestKotlinUnknownPackages:
    """Tests for unknown Kotlin packages."""

    @pytest.fixture
    def resolver(self):
        return KotlinImportResolver()

    def test_resolve_unknown_package(self, resolver):
        """Should return partial for unknown packages."""
        result = resolver.resolve("com.unknown.package")
        assert result.status == ResolutionStatus.PARTIAL

    def test_resolve_partial_package(self, resolver):
        """Should return partial for partial known packages."""
        result = resolver.resolve("kotlin.unknown")
        assert result.status == ResolutionStatus.PARTIAL


class TestKotlinModuleExports:
    """Tests for module export retrieval."""

    @pytest.fixture
    def resolver(self):
        return KotlinImportResolver()

    def test_get_module_exports_known(self, resolver):
        """Should return exports for known modules."""
        exports = resolver.get_module_exports("kotlin")
        assert "Int" in exports
        assert "String" in exports

    def test_get_module_exports_unknown(self, resolver):
        """Should return empty for unknown modules."""
        exports = resolver.get_module_exports("com.unknown.package")
        assert exports == []


class TestKotlinCompletionCandidates:
    """Tests for import completion."""

    @pytest.fixture
    def resolver(self):
        return KotlinImportResolver()

    def test_completion_kotlin(self, resolver):
        """Should complete kotlin prefix."""
        candidates = resolver.get_completion_candidates("kotlin")
        assert "kotlin" in candidates
        assert "kotlin.collections" in candidates
        assert "kotlin.text" in candidates

    def test_completion_kotlinx(self, resolver):
        """Should complete kotlinx prefix."""
        candidates = resolver.get_completion_candidates("kotlinx")
        assert "kotlinx.coroutines" in candidates
        assert "kotlinx.serialization" in candidates


class TestKotlinModuleAvailability:
    """Tests for module availability checking."""

    @pytest.fixture
    def resolver(self):
        return KotlinImportResolver()

    def test_is_available_stdlib(self, resolver):
        """Should report stdlib as available."""
        assert resolver.is_available("kotlin")
        assert resolver.is_available("kotlin.collections")

    def test_is_available_coroutines(self, resolver):
        """Should report coroutines as available."""
        assert resolver.is_available("kotlinx.coroutines")
        assert resolver.is_available("kotlinx.coroutines.flow")

    def test_is_available_unknown(self, resolver):
        """Should report unknown as unavailable."""
        assert not resolver.is_available("com.unknown.package")
