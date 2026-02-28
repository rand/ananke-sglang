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
"""Tests for Swift import resolver."""

import pytest

from domains.imports.resolvers.base import ResolutionStatus
from domains.imports.resolvers.swift import (
    SwiftImportResolver,
    create_swift_resolver,
    SWIFT_STANDARD_LIBRARY,
    FOUNDATION_FRAMEWORK,
    UIKIT_FRAMEWORK,
    SWIFTUI_FRAMEWORK,
    COMBINE_FRAMEWORK,
    SWIFT_POPULAR_PACKAGES,
)


class TestSwiftImportResolverCreation:
    """Tests for SwiftImportResolver creation."""

    def test_create_resolver(self):
        """Should create a Swift import resolver."""
        resolver = SwiftImportResolver()
        assert resolver.language == "swift"

    def test_factory_function(self):
        """Should create resolver via factory."""
        resolver = create_swift_resolver()
        assert isinstance(resolver, SwiftImportResolver)
        assert resolver.language == "swift"

    def test_get_version(self):
        """Should return Swift version."""
        resolver = SwiftImportResolver()
        version = resolver.get_version()
        assert version is not None
        assert version == "5.9"


class TestSwiftStandardLibrary:
    """Tests for Swift standard library resolution."""

    @pytest.fixture
    def resolver(self):
        return SwiftImportResolver()

    def test_resolve_swift(self, resolver):
        """Should resolve Swift module."""
        result = resolver.resolve("Swift")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.module is not None
        assert result.module.is_builtin is True
        assert "Int" in result.exports
        assert "String" in result.exports
        assert "Array" in result.exports
        assert "print" in result.exports


class TestFoundationFramework:
    """Tests for Foundation framework resolution."""

    @pytest.fixture
    def resolver(self):
        return SwiftImportResolver()

    def test_resolve_foundation(self, resolver):
        """Should resolve Foundation module."""
        result = resolver.resolve("Foundation")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.module.is_builtin is True
        assert "Data" in result.exports
        assert "Date" in result.exports
        assert "URL" in result.exports
        assert "JSONEncoder" in result.exports
        assert "JSONDecoder" in result.exports
        assert "FileManager" in result.exports


class TestUIKitFramework:
    """Tests for UIKit framework resolution."""

    @pytest.fixture
    def resolver(self):
        return SwiftImportResolver()

    def test_resolve_uikit(self, resolver):
        """Should resolve UIKit module."""
        result = resolver.resolve("UIKit")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.module.is_builtin is True
        assert "UIView" in result.exports
        assert "UIViewController" in result.exports
        assert "UITableView" in result.exports
        assert "UIButton" in result.exports
        assert "UILabel" in result.exports
        assert "UIColor" in result.exports


class TestSwiftUIFramework:
    """Tests for SwiftUI framework resolution."""

    @pytest.fixture
    def resolver(self):
        return SwiftImportResolver()

    def test_resolve_swiftui(self, resolver):
        """Should resolve SwiftUI module."""
        result = resolver.resolve("SwiftUI")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.module.is_builtin is True
        assert "View" in result.exports
        assert "Text" in result.exports
        assert "Button" in result.exports
        assert "List" in result.exports
        assert "State" in result.exports
        assert "Binding" in result.exports
        assert "ObservableObject" in result.exports


class TestCombineFramework:
    """Tests for Combine framework resolution."""

    @pytest.fixture
    def resolver(self):
        return SwiftImportResolver()

    def test_resolve_combine(self, resolver):
        """Should resolve Combine module."""
        result = resolver.resolve("Combine")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.module.is_builtin is True
        assert "Publisher" in result.exports
        assert "Subscriber" in result.exports
        assert "Subject" in result.exports
        assert "CurrentValueSubject" in result.exports
        assert "PassthroughSubject" in result.exports


class TestSwiftPopularPackages:
    """Tests for popular Swift packages."""

    @pytest.fixture
    def resolver(self):
        return SwiftImportResolver()

    def test_resolve_alamofire(self, resolver):
        """Should resolve Alamofire package."""
        result = resolver.resolve("Alamofire")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.module.is_builtin is False
        assert "AF" in result.exports
        assert "Session" in result.exports

    def test_resolve_rxswift(self, resolver):
        """Should resolve RxSwift package."""
        result = resolver.resolve("RxSwift")
        assert result.status == ResolutionStatus.RESOLVED
        assert "Observable" in result.exports
        assert "DisposeBag" in result.exports

    def test_resolve_snapkit(self, resolver):
        """Should resolve SnapKit package."""
        result = resolver.resolve("SnapKit")
        assert result.status == ResolutionStatus.RESOLVED
        assert "ConstraintMaker" in result.exports

    def test_resolve_kingfisher(self, resolver):
        """Should resolve Kingfisher package."""
        result = resolver.resolve("Kingfisher")
        assert result.status == ResolutionStatus.RESOLVED
        assert "KingfisherManager" in result.exports

    def test_resolve_realm(self, resolver):
        """Should resolve Realm package."""
        result = resolver.resolve("Realm")
        assert result.status == ResolutionStatus.RESOLVED
        assert "Realm" in result.exports
        assert "Object" in result.exports

    def test_resolve_moya(self, resolver):
        """Should resolve Moya package."""
        result = resolver.resolve("Moya")
        assert result.status == ResolutionStatus.RESOLVED
        assert "MoyaProvider" in result.exports


class TestSwiftUnknownPackages:
    """Tests for unknown Swift packages."""

    @pytest.fixture
    def resolver(self):
        return SwiftImportResolver()

    def test_resolve_unknown_package(self, resolver):
        """Should return partial for unknown packages."""
        result = resolver.resolve("UnknownPackage")
        assert result.status == ResolutionStatus.PARTIAL


class TestSwiftModuleExports:
    """Tests for module export retrieval."""

    @pytest.fixture
    def resolver(self):
        return SwiftImportResolver()

    def test_get_module_exports_known(self, resolver):
        """Should return exports for known modules."""
        exports = resolver.get_module_exports("Swift")
        assert "Int" in exports
        assert "String" in exports

    def test_get_module_exports_foundation(self, resolver):
        """Should return exports for Foundation."""
        exports = resolver.get_module_exports("Foundation")
        assert "Data" in exports
        assert "URL" in exports

    def test_get_module_exports_unknown(self, resolver):
        """Should return empty for unknown modules."""
        exports = resolver.get_module_exports("UnknownPackage")
        assert exports == []


class TestSwiftCompletionCandidates:
    """Tests for import completion."""

    @pytest.fixture
    def resolver(self):
        return SwiftImportResolver()

    def test_completion_swift(self, resolver):
        """Should complete Swift prefix."""
        candidates = resolver.get_completion_candidates("Swift")
        assert "Swift" in candidates
        assert "SwiftUI" in candidates

    def test_completion_ui(self, resolver):
        """Should complete UI prefix."""
        candidates = resolver.get_completion_candidates("UI")
        assert "UIKit" in candidates

    def test_completion_a(self, resolver):
        """Should complete A prefix."""
        candidates = resolver.get_completion_candidates("A")
        assert "Alamofire" in candidates


class TestSwiftModuleAvailability:
    """Tests for module availability checking."""

    @pytest.fixture
    def resolver(self):
        return SwiftImportResolver()

    def test_is_available_stdlib(self, resolver):
        """Should report stdlib as available."""
        assert resolver.is_available("Swift")

    def test_is_available_foundation(self, resolver):
        """Should report Foundation as available."""
        assert resolver.is_available("Foundation")

    def test_is_available_uikit(self, resolver):
        """Should report UIKit as available."""
        assert resolver.is_available("UIKit")

    def test_is_available_swiftui(self, resolver):
        """Should report SwiftUI as available."""
        assert resolver.is_available("SwiftUI")

    def test_is_available_combine(self, resolver):
        """Should report Combine as available."""
        assert resolver.is_available("Combine")

    def test_is_available_unknown(self, resolver):
        """Should report unknown as unavailable."""
        assert not resolver.is_available("UnknownPackage")
