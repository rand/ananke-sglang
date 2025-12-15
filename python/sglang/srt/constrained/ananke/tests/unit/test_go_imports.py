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
"""Tests for Go import resolver."""

import pytest

from domains.imports.resolvers.go import (
    GoImportResolver,
    GO_STANDARD_LIBRARY,
    GO_POPULAR_PACKAGES,
)


class TestGoImportResolverCreation:
    """Tests for GoImportResolver creation."""

    def test_create_resolver(self):
        """Should create a Go import resolver."""
        resolver = GoImportResolver()
        assert resolver.language == "go"


class TestGoStandardLibraryResolution:
    """Tests for Go standard library import resolution."""

    @pytest.fixture
    def resolver(self):
        return GoImportResolver()

    def test_resolve_fmt(self, resolver):
        """Should resolve fmt package."""
        result = resolver.resolve("fmt")
        assert result is not None
        assert result.success
        assert result.module.name == "fmt"
        assert result.module.is_builtin
        assert len(result.exports) > 0
        assert "Println" in result.exports
        assert "Printf" in result.exports
        assert "Sprintf" in result.exports

    def test_resolve_os(self, resolver):
        """Should resolve os package."""
        result = resolver.resolve("os")
        assert result is not None
        assert result.success
        assert "Open" in result.exports
        assert "Create" in result.exports
        assert "Exit" in result.exports

    def test_resolve_io(self, resolver):
        """Should resolve io package."""
        result = resolver.resolve("io")
        assert result is not None
        assert result.success
        assert "Reader" in result.exports
        assert "Writer" in result.exports
        assert "EOF" in result.exports

    def test_resolve_strings(self, resolver):
        """Should resolve strings package."""
        result = resolver.resolve("strings")
        assert result is not None
        assert result.success
        assert "Contains" in result.exports
        assert "Split" in result.exports
        assert "Join" in result.exports

    def test_resolve_strconv(self, resolver):
        """Should resolve strconv package."""
        result = resolver.resolve("strconv")
        assert result is not None
        assert "Atoi" in result.exports
        assert "Itoa" in result.exports
        assert "ParseInt" in result.exports

    def test_resolve_bytes(self, resolver):
        """Should resolve bytes package."""
        result = resolver.resolve("bytes")
        assert result is not None
        assert "Buffer" in result.exports
        assert "NewBuffer" in result.exports

    def test_resolve_errors(self, resolver):
        """Should resolve errors package."""
        result = resolver.resolve("errors")
        assert result is not None
        assert "New" in result.exports
        assert "Is" in result.exports
        assert "As" in result.exports

    def test_resolve_context(self, resolver):
        """Should resolve context package."""
        result = resolver.resolve("context")
        assert result is not None
        assert "Background" in result.exports
        assert "TODO" in result.exports
        assert "WithCancel" in result.exports

    def test_resolve_sync(self, resolver):
        """Should resolve sync package."""
        result = resolver.resolve("sync")
        assert result is not None
        assert "Mutex" in result.exports
        assert "WaitGroup" in result.exports

    def test_resolve_time(self, resolver):
        """Should resolve time package."""
        result = resolver.resolve("time")
        assert result is not None
        assert "Now" in result.exports
        assert "Sleep" in result.exports
        assert "Duration" in result.exports

    def test_resolve_math(self, resolver):
        """Should resolve math package."""
        result = resolver.resolve("math")
        assert result is not None
        assert "Abs" in result.exports
        assert "Sqrt" in result.exports
        assert "Pi" in result.exports


class TestGoNestedPackageResolution:
    """Tests for nested Go package resolution."""

    @pytest.fixture
    def resolver(self):
        return GoImportResolver()

    def test_resolve_encoding_json(self, resolver):
        """Should resolve encoding/json package."""
        result = resolver.resolve("encoding/json")
        assert result is not None
        assert result.success
        assert "Marshal" in result.exports
        assert "Unmarshal" in result.exports
        assert "NewDecoder" in result.exports

    def test_resolve_encoding_xml(self, resolver):
        """Should resolve encoding/xml package."""
        result = resolver.resolve("encoding/xml")
        assert result is not None

    def test_resolve_net_http(self, resolver):
        """Should resolve net/http package."""
        result = resolver.resolve("net/http")
        assert result is not None
        assert "Get" in result.exports
        assert "ListenAndServe" in result.exports
        assert "StatusOK" in result.exports

    def test_resolve_net_url(self, resolver):
        """Should resolve net/url package."""
        result = resolver.resolve("net/url")
        assert result is not None
        assert "Parse" in result.exports
        assert "URL" in result.exports

    def test_resolve_path_filepath(self, resolver):
        """Should resolve path/filepath package."""
        result = resolver.resolve("path/filepath")
        assert result is not None
        assert "Join" in result.exports
        assert "Walk" in result.exports

    def test_resolve_math_rand(self, resolver):
        """Should resolve math/rand package."""
        result = resolver.resolve("math/rand")
        assert result is not None
        assert "Int" in result.exports
        assert "Intn" in result.exports

    def test_resolve_sync_atomic(self, resolver):
        """Should resolve sync/atomic package."""
        result = resolver.resolve("sync/atomic")
        assert result is not None
        assert "AddInt64" in result.exports
        assert "LoadInt64" in result.exports

    def test_resolve_database_sql(self, resolver):
        """Should resolve database/sql package."""
        result = resolver.resolve("database/sql")
        assert result is not None
        assert "Open" in result.exports
        assert "DB" in result.exports


class TestGoPopularPackageResolution:
    """Tests for popular Go package resolution."""

    @pytest.fixture
    def resolver(self):
        return GoImportResolver()

    def test_resolve_gin(self, resolver):
        """Should resolve gin-gonic/gin package."""
        result = resolver.resolve("github.com/gin-gonic/gin")
        assert result is not None
        assert result.success
        assert not result.module.is_builtin
        assert "Default" in result.exports
        assert "Engine" in result.exports

    def test_resolve_gorilla_mux(self, resolver):
        """Should resolve gorilla/mux package."""
        result = resolver.resolve("github.com/gorilla/mux")
        assert result is not None
        assert "NewRouter" in result.exports

    def test_resolve_testify_assert(self, resolver):
        """Should resolve testify/assert package."""
        result = resolver.resolve("github.com/stretchr/testify/assert")
        assert result is not None
        assert "Equal" in result.exports
        assert "NotNil" in result.exports

    def test_resolve_zap(self, resolver):
        """Should resolve zap package."""
        result = resolver.resolve("go.uber.org/zap")
        assert result is not None
        assert "NewProduction" in result.exports
        assert "Logger" in result.exports

    def test_resolve_gorm(self, resolver):
        """Should resolve gorm package."""
        result = resolver.resolve("gorm.io/gorm")
        assert result is not None
        assert "Open" in result.exports
        assert "DB" in result.exports


class TestGoUnknownPackageResolution:
    """Tests for unknown Go package resolution."""

    @pytest.fixture
    def resolver(self):
        return GoImportResolver()

    def test_resolve_unknown_package(self, resolver):
        """Should resolve unknown package with empty exports."""
        result = resolver.resolve("github.com/unknown/package")
        assert result is not None
        # Unknown packages still resolve (PARTIAL status) but with empty exports
        assert not result.module.is_builtin
        assert result.exports == set()

    def test_resolve_local_package(self, resolver):
        """Should resolve local package path."""
        result = resolver.resolve("myproject/internal/utils")
        assert result is not None
        # Local packages resolve with PARTIAL status


class TestGoModuleExports:
    """Tests for Go module exports retrieval."""

    @pytest.fixture
    def resolver(self):
        return GoImportResolver()

    def test_get_fmt_exports(self, resolver):
        """Should get fmt exports."""
        exports = resolver.get_module_exports("fmt")
        assert "Println" in exports
        assert "Printf" in exports

    def test_get_unknown_exports(self, resolver):
        """Should return empty for unknown package."""
        exports = resolver.get_module_exports("unknown/package")
        assert exports == []


class TestGoCompletionCandidates:
    """Tests for Go import completion candidates."""

    @pytest.fixture
    def resolver(self):
        return GoImportResolver()

    def test_complete_empty_prefix(self, resolver):
        """Should return candidates for empty prefix."""
        candidates = resolver.get_completion_candidates("")
        assert len(candidates) > 0
        assert "fmt" in candidates
        assert "os" in candidates

    def test_complete_encoding_prefix(self, resolver):
        """Should return encoding candidates."""
        candidates = resolver.get_completion_candidates("encoding")
        assert "encoding/json" in candidates
        assert "encoding/xml" in candidates
        assert "encoding/base64" in candidates

    def test_complete_net_prefix(self, resolver):
        """Should return net candidates."""
        candidates = resolver.get_completion_candidates("net")
        assert "net/http" in candidates
        assert "net/url" in candidates

    def test_complete_github_prefix(self, resolver):
        """Should return github candidates."""
        candidates = resolver.get_completion_candidates("github.com")
        assert any("gin-gonic" in c for c in candidates)
        assert any("gorilla" in c for c in candidates)

    def test_complete_no_match(self, resolver):
        """Should return empty for no matches."""
        candidates = resolver.get_completion_candidates("zzzznotapackage")
        assert candidates == []


class TestGoStandardLibraryCoverage:
    """Tests for Go standard library coverage."""

    def test_all_standard_packages_present(self):
        """Should have common standard library packages."""
        expected_packages = [
            "fmt", "os", "io", "strings", "strconv", "bytes",
            "errors", "context", "sync", "time", "math", "sort",
            "regexp", "path", "log", "testing", "reflect", "unsafe",
            "runtime",
        ]
        for pkg in expected_packages:
            assert pkg in GO_STANDARD_LIBRARY, f"Missing standard package: {pkg}"

    def test_nested_standard_packages_present(self):
        """Should have nested standard library packages."""
        expected_packages = [
            "encoding/json", "encoding/xml", "encoding/base64",
            "net/http", "net/url", "path/filepath",
            "math/rand", "sync/atomic", "database/sql",
        ]
        for pkg in expected_packages:
            assert pkg in GO_STANDARD_LIBRARY, f"Missing nested package: {pkg}"
