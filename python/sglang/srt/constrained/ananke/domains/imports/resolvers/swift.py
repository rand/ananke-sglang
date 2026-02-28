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
"""Swift import resolution.

This module provides import resolution for Swift frameworks and packages,
including Apple frameworks and Swift Package Manager dependencies.

References:
    - Swift Standard Library: https://developer.apple.com/documentation/swift
    - Foundation: https://developer.apple.com/documentation/foundation
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
# Swift Standard Library
# =============================================================================

SWIFT_STANDARD_LIBRARY: Dict[str, List[str]] = {
    "Swift": [
        # Types
        "Int", "Int8", "Int16", "Int32", "Int64",
        "UInt", "UInt8", "UInt16", "UInt32", "UInt64",
        "Float", "Double", "Float16", "Float80",
        "Bool", "String", "Character",
        "Array", "Dictionary", "Set", "Optional",
        "Range", "ClosedRange", "PartialRangeFrom",
        "Result", "Never", "Void",
        "AnyObject", "AnyHashable",
        "Sequence", "Collection", "RandomAccessCollection",
        "IteratorProtocol", "Comparable", "Equatable", "Hashable",
        "Codable", "Encodable", "Decodable",
        "Error", "CustomStringConvertible", "CustomDebugStringConvertible",
        # Functions
        "print", "debugPrint", "dump", "readLine",
        "abs", "min", "max", "stride",
        "zip", "sequence", "repeatElement",
        "precondition", "preconditionFailure", "assert", "assertionFailure",
        "fatalError",
    ],
}

# Foundation Framework
FOUNDATION_FRAMEWORK: Dict[str, List[str]] = {
    "Foundation": [
        # Core types
        "NSObject", "NSString", "NSNumber", "NSArray", "NSDictionary", "NSSet",
        "NSData", "NSDate", "NSURL", "NSUUID",
        # Swift overlays
        "Data", "Date", "URL", "UUID", "URLComponents", "URLRequest",
        "DateFormatter", "NumberFormatter", "ISO8601DateFormatter",
        "JSONEncoder", "JSONDecoder", "PropertyListEncoder", "PropertyListDecoder",
        "FileManager", "Bundle", "ProcessInfo",
        "NotificationCenter", "Notification", "NotificationQueue",
        "UserDefaults",
        "Timer", "RunLoop",
        "DispatchQueue", "DispatchGroup", "DispatchSemaphore",
        "NSError", "NSException",
        "NSLock", "NSRecursiveLock", "NSCondition",
        "CharacterSet", "Scanner",
        "Locale", "TimeZone", "Calendar",
        "Measurement", "Unit", "Dimension",
    ],
}

# UIKit Framework
UIKIT_FRAMEWORK: Dict[str, List[str]] = {
    "UIKit": [
        # Application
        "UIApplication", "UIApplicationDelegate", "UIScene", "UISceneDelegate",
        # View Controllers
        "UIViewController", "UINavigationController", "UITabBarController",
        "UITableViewController", "UICollectionViewController",
        "UIPageViewController", "UISplitViewController",
        # Views
        "UIView", "UILabel", "UIButton", "UITextField", "UITextView",
        "UIImageView", "UIScrollView", "UITableView", "UICollectionView",
        "UIStackView", "UIActivityIndicatorView", "UIProgressView",
        "UISwitch", "UISlider", "UIStepper", "UISegmentedControl",
        "UIPickerView", "UIDatePicker",
        # Navigation
        "UINavigationBar", "UITabBar", "UIToolbar", "UIBarButtonItem",
        # Alerts and Actions
        "UIAlertController", "UIAlertAction",
        # Table/Collection View
        "UITableViewCell", "UICollectionViewCell",
        "UITableViewDataSource", "UITableViewDelegate",
        "UICollectionViewDataSource", "UICollectionViewDelegate",
        "UICollectionViewLayout", "UICollectionViewFlowLayout",
        # Gestures
        "UIGestureRecognizer", "UITapGestureRecognizer", "UISwipeGestureRecognizer",
        "UIPanGestureRecognizer", "UIPinchGestureRecognizer", "UILongPressGestureRecognizer",
        # Layout
        "NSLayoutConstraint", "UILayoutGuide",
        # Graphics
        "UIColor", "UIImage", "UIFont", "UIBezierPath",
        # Animation
        "UIViewPropertyAnimator",
        # Events
        "UIControl", "UIEvent", "UITouch",
    ],
}

# SwiftUI Framework
SWIFTUI_FRAMEWORK: Dict[str, List[str]] = {
    "SwiftUI": [
        # App Structure
        "App", "Scene", "WindowGroup",
        # Views
        "View", "Text", "Image", "Button", "Label",
        "TextField", "SecureField", "TextEditor",
        "Toggle", "Slider", "Stepper", "Picker", "DatePicker", "ColorPicker",
        "List", "ScrollView", "LazyVStack", "LazyHStack", "LazyVGrid", "LazyHGrid",
        "NavigationView", "NavigationLink", "TabView",
        "Form", "Section", "Group", "GroupBox",
        "Spacer", "Divider",
        "ZStack", "VStack", "HStack",
        "GeometryReader", "Canvas",
        "Alert", "ActionSheet", "Sheet", "Popover",
        "ProgressView", "Gauge",
        # State Management
        "State", "Binding", "ObservableObject", "Published", "StateObject",
        "ObservedObject", "EnvironmentObject", "Environment", "EnvironmentKey",
        "AppStorage", "SceneStorage", "FocusState",
        # Modifiers (as protocols)
        "ViewModifier", "AnimatableModifier",
        # Navigation
        "NavigationPath", "NavigationStack", "NavigationSplitView",
        # Layout
        "Layout", "GridItem", "Alignment", "Edge",
        # Animation
        "Animation", "withAnimation", "Transaction",
        # Shapes
        "Shape", "Rectangle", "Circle", "Ellipse", "Capsule", "RoundedRectangle",
        "Path",
        # Colors and styles
        "Color", "Gradient", "LinearGradient", "RadialGradient",
        "Font", "FontWeight",
        "ShapeStyle", "ForegroundStyle",
        # Gestures
        "Gesture", "TapGesture", "LongPressGesture", "DragGesture",
        "MagnificationGesture", "RotationGesture",
    ],
}

# Combine Framework
COMBINE_FRAMEWORK: Dict[str, List[str]] = {
    "Combine": [
        # Core types
        "Publisher", "Subscriber", "Subscription", "Cancellable", "AnyCancellable",
        "Subject", "CurrentValueSubject", "PassthroughSubject",
        # Publishers
        "Just", "Empty", "Fail", "Deferred", "Future", "Record",
        "AnyPublisher", "Published",
        # Operators (as methods, but types too)
        "Publishers",
        # Schedulers
        "Scheduler", "ImmediateScheduler", "RunLoop", "DispatchQueue",
    ],
}

# Popular Swift Packages
SWIFT_POPULAR_PACKAGES: Dict[str, List[str]] = {
    "Alamofire": [
        "AF", "Session", "Request", "DataRequest", "DownloadRequest", "UploadRequest",
        "HTTPMethod", "HTTPHeaders", "URLEncoding", "JSONEncoding",
        "AFError", "ResponseSerializer",
    ],
    "RxSwift": [
        "Observable", "Observer", "Single", "Completable", "Maybe",
        "PublishSubject", "BehaviorSubject", "ReplaySubject", "AsyncSubject",
        "DisposeBag", "Disposable", "CompositeDisposable",
        "MainScheduler", "SerialDispatchQueueScheduler",
    ],
    "SnapKit": [
        "ConstraintMaker", "ConstraintItem",
        "snp", "makeConstraints", "updateConstraints", "remakeConstraints",
    ],
    "Kingfisher": [
        "KingfisherManager", "ImageDownloader", "ImageCache",
        "KFImage", "ImageResource", "DownloadTask",
    ],
    "Realm": [
        "Realm", "Object", "List", "Results", "LinkingObjects",
        "RealmSwift", "RealmConfiguration",
    ],
    "Moya": [
        "MoyaProvider", "TargetType", "Task", "Method",
        "Endpoint", "Response", "MoyaError",
    ],
}


# =============================================================================
# Swift Import Resolver
# =============================================================================

class SwiftImportResolver(ImportResolver):
    """Swift import resolver."""

    @property
    def language(self) -> str:
        return "swift"

    def resolve(self, import_path: str) -> ImportResolution:
        """Resolve a Swift import path."""
        # Check Swift standard library
        if import_path in SWIFT_STANDARD_LIBRARY:
            exports = set(SWIFT_STANDARD_LIBRARY[import_path])
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

        # Check Foundation
        if import_path in FOUNDATION_FRAMEWORK:
            exports = set(FOUNDATION_FRAMEWORK[import_path])
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

        # Check UIKit
        if import_path in UIKIT_FRAMEWORK:
            exports = set(UIKIT_FRAMEWORK[import_path])
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

        # Check SwiftUI
        if import_path in SWIFTUI_FRAMEWORK:
            exports = set(SWIFTUI_FRAMEWORK[import_path])
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

        # Check Combine
        if import_path in COMBINE_FRAMEWORK:
            exports = set(COMBINE_FRAMEWORK[import_path])
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
        if import_path in SWIFT_POPULAR_PACKAGES:
            exports = set(SWIFT_POPULAR_PACKAGES[import_path])
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

        # Unknown module
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
        all_frameworks = {
            **SWIFT_STANDARD_LIBRARY,
            **FOUNDATION_FRAMEWORK,
            **UIKIT_FRAMEWORK,
            **SWIFTUI_FRAMEWORK,
            **COMBINE_FRAMEWORK,
            **SWIFT_POPULAR_PACKAGES,
        }
        return all_frameworks.get(import_path, [])

    def get_completion_candidates(self, prefix: str) -> List[str]:
        """Get import path completion candidates."""
        candidates = []
        all_frameworks = {
            **SWIFT_STANDARD_LIBRARY,
            **FOUNDATION_FRAMEWORK,
            **UIKIT_FRAMEWORK,
            **SWIFTUI_FRAMEWORK,
            **COMBINE_FRAMEWORK,
            **SWIFT_POPULAR_PACKAGES,
        }

        for pkg in all_frameworks:
            if pkg.startswith(prefix):
                candidates.append(pkg)

        return sorted(candidates)

    def is_available(self, module_name: str) -> bool:
        """Check if a Swift module is available."""
        all_frameworks = {
            **SWIFT_STANDARD_LIBRARY,
            **FOUNDATION_FRAMEWORK,
            **UIKIT_FRAMEWORK,
            **SWIFTUI_FRAMEWORK,
            **COMBINE_FRAMEWORK,
            **SWIFT_POPULAR_PACKAGES,
        }
        return module_name in all_frameworks

    def get_version(self) -> Optional[str]:
        """Get Swift version."""
        return "5.9"  # Current stable version


def create_swift_resolver() -> SwiftImportResolver:
    """Factory function to create a Swift resolver."""
    return SwiftImportResolver()
