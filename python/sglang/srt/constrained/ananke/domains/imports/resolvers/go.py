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
"""Go import resolution.

This module provides import resolution for Go packages, including
standard library packages and common third-party packages.

References:
    - Go Packages: https://pkg.go.dev/
    - Go Standard Library: https://pkg.go.dev/std
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set

from domains.imports.resolvers.base import (
    ImportResolver,
    ImportResolution,
    ResolvedModule,
    ResolutionStatus,
)


# =============================================================================
# Go Standard Library
# =============================================================================

GO_STANDARD_LIBRARY: Dict[str, List[str]] = {
    # Core packages
    "fmt": [
        "Print", "Println", "Printf", "Sprintf", "Fprintf",
        "Scan", "Scanln", "Scanf", "Sscan", "Sscanf",
        "Errorf", "Fprint", "Fprintln",
        "Stringer", "GoStringer", "Formatter", "State",
    ],
    "os": [
        "Open", "Create", "OpenFile", "Remove", "RemoveAll",
        "Mkdir", "MkdirAll", "Rename", "Stat", "Lstat",
        "ReadFile", "WriteFile", "ReadDir",
        "Stdin", "Stdout", "Stderr",
        "Args", "Environ", "Getenv", "Setenv",
        "Exit", "Getwd", "Chdir",
        "File", "FileInfo", "FileMode", "DirEntry",
        "PathError", "LinkError", "SyscallError",
        "ErrExist", "ErrNotExist", "ErrPermission",
    ],
    "io": [
        "Reader", "Writer", "Closer", "Seeker",
        "ReadWriter", "ReadCloser", "WriteCloser",
        "ReadWriteCloser", "ReadSeeker", "WriteSeeker",
        "ReadAll", "Copy", "CopyN", "CopyBuffer",
        "Pipe", "TeeReader", "LimitReader", "MultiReader",
        "NopCloser", "ReadAtLeast", "ReadFull",
        "EOF", "ErrUnexpectedEOF", "ErrShortWrite",
        "Discard",
    ],
    "strings": [
        "Contains", "ContainsAny", "ContainsRune",
        "Count", "EqualFold",
        "HasPrefix", "HasSuffix",
        "Index", "IndexAny", "IndexByte", "IndexRune",
        "Join", "Split", "SplitN", "SplitAfter",
        "Fields", "FieldsFunc",
        "Map", "Repeat", "Replace", "ReplaceAll",
        "ToLower", "ToUpper", "ToTitle",
        "Trim", "TrimLeft", "TrimRight", "TrimSpace",
        "TrimPrefix", "TrimSuffix",
        "Builder", "Reader", "Replacer",
        "NewReader", "NewReplacer",
    ],
    "strconv": [
        "Atoi", "Itoa",
        "ParseInt", "ParseUint", "ParseFloat", "ParseBool",
        "FormatInt", "FormatUint", "FormatFloat", "FormatBool",
        "AppendInt", "AppendUint", "AppendFloat", "AppendBool",
        "Quote", "QuoteRune", "Unquote", "UnquoteChar",
        "IntSize",
    ],
    "bytes": [
        "Buffer", "Reader",
        "Contains", "Count", "Equal", "EqualFold",
        "HasPrefix", "HasSuffix",
        "Index", "IndexAny", "IndexByte",
        "Join", "Split", "SplitN",
        "Trim", "TrimLeft", "TrimRight", "TrimSpace",
        "ToLower", "ToUpper",
        "NewBuffer", "NewBufferString", "NewReader",
    ],
    "errors": [
        "New", "Is", "As", "Unwrap", "Join",
    ],
    "context": [
        "Context", "CancelFunc",
        "Background", "TODO",
        "WithCancel", "WithDeadline", "WithTimeout", "WithValue",
        "Canceled", "DeadlineExceeded",
    ],
    "sync": [
        "Mutex", "RWMutex", "Cond",
        "Once", "WaitGroup",
        "Pool", "Map",
        "Locker",
    ],
    "sync/atomic": [
        "AddInt32", "AddInt64", "AddUint32", "AddUint64",
        "LoadInt32", "LoadInt64", "LoadUint32", "LoadUint64", "LoadPointer",
        "StoreInt32", "StoreInt64", "StoreUint32", "StoreUint64", "StorePointer",
        "SwapInt32", "SwapInt64", "SwapUint32", "SwapUint64", "SwapPointer",
        "CompareAndSwapInt32", "CompareAndSwapInt64",
        "Value", "Bool", "Int32", "Int64", "Uint32", "Uint64", "Uintptr", "Pointer",
    ],
    "time": [
        "Time", "Duration", "Location", "Month", "Weekday",
        "Now", "Date", "Unix", "UnixMilli", "UnixMicro",
        "Parse", "ParseDuration", "ParseInLocation",
        "Sleep", "After", "Tick", "NewTicker", "NewTimer",
        "Since", "Until",
        "Nanosecond", "Microsecond", "Millisecond", "Second", "Minute", "Hour",
        "UTC", "Local",
        "Timer", "Ticker",
    ],
    "math": [
        "Abs", "Ceil", "Floor", "Round", "Trunc",
        "Max", "Min", "Mod", "Pow", "Sqrt",
        "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Atan2",
        "Log", "Log10", "Log2", "Exp", "Exp2",
        "Inf", "IsInf", "IsNaN", "NaN",
        "Pi", "E", "Phi",
        "MaxFloat32", "MaxFloat64", "MaxInt", "MinInt",
    ],
    "math/rand": [
        "Int", "Intn", "Int31", "Int31n", "Int63", "Int63n",
        "Uint32", "Uint64",
        "Float32", "Float64",
        "Perm", "Shuffle",
        "Seed", "New", "NewSource",
        "Rand", "Source",
    ],
    "sort": [
        "Sort", "Stable", "IsSorted", "Search",
        "Ints", "Float64s", "Strings",
        "IntsAreSorted", "Float64sAreSorted", "StringsAreSorted",
        "Slice", "SliceStable", "SliceIsSorted",
        "Interface", "Reverse",
    ],
    "regexp": [
        "Compile", "MustCompile", "CompilePOSIX", "MustCompilePOSIX",
        "Match", "MatchString", "MatchReader",
        "QuoteMeta",
        "Regexp",
    ],
    "path": [
        "Base", "Clean", "Dir", "Ext", "IsAbs", "Join", "Match", "Split",
    ],
    "path/filepath": [
        "Abs", "Base", "Clean", "Dir", "Ext", "FromSlash", "ToSlash",
        "Glob", "IsAbs", "Join", "Match", "Rel", "Split", "SplitList",
        "Walk", "WalkDir",
        "Separator", "ListSeparator",
    ],
    "encoding/json": [
        "Marshal", "MarshalIndent", "Unmarshal",
        "NewDecoder", "NewEncoder",
        "Decoder", "Encoder",
        "RawMessage", "Number",
        "Marshaler", "Unmarshaler",
        "Valid", "Compact", "Indent",
    ],
    "encoding/xml": [
        "Marshal", "MarshalIndent", "Unmarshal",
        "NewDecoder", "NewEncoder",
        "Decoder", "Encoder",
        "Marshaler", "Unmarshaler",
    ],
    "encoding/base64": [
        "StdEncoding", "URLEncoding", "RawStdEncoding", "RawURLEncoding",
        "NewEncoding", "Encoding",
    ],
    "net/http": [
        "Get", "Post", "PostForm", "Head",
        "NewRequest", "NewRequestWithContext",
        "ListenAndServe", "ListenAndServeTLS",
        "Handle", "HandleFunc", "Serve", "ServeTLS",
        "Client", "Server", "Transport",
        "Request", "Response", "Header",
        "Handler", "HandlerFunc", "ServeMux",
        "Cookie", "CookieJar",
        "Error", "NotFound", "Redirect",
        "StatusOK", "StatusCreated", "StatusNotFound", "StatusInternalServerError",
        "DefaultClient", "DefaultServeMux", "DefaultTransport",
        "MethodGet", "MethodPost", "MethodPut", "MethodDelete",
        "FileServer", "StripPrefix", "TimeoutHandler",
    ],
    "net/url": [
        "Parse", "ParseRequestURI",
        "URL", "Userinfo", "Values",
        "QueryEscape", "QueryUnescape", "PathEscape", "PathUnescape",
    ],
    "log": [
        "Print", "Println", "Printf",
        "Fatal", "Fatalln", "Fatalf",
        "Panic", "Panicln", "Panicf",
        "New", "Default",
        "Logger", "Flags", "SetFlags", "SetOutput", "SetPrefix",
    ],
    "testing": [
        "T", "B", "M", "TB", "F",
        "Main", "Short", "Verbose",
        "Cover", "CoverMode",
    ],
    "reflect": [
        "TypeOf", "ValueOf",
        "Type", "Value", "Kind",
        "DeepEqual", "Copy",
        "MakeSlice", "MakeMap", "MakeChan", "MakeFunc",
        "New", "NewAt", "Zero",
    ],
    "unsafe": [
        "Sizeof", "Alignof", "Offsetof",
        "Pointer", "Add", "Slice",
    ],
    "runtime": [
        "GOOS", "GOARCH", "GOROOT",
        "NumCPU", "NumGoroutine", "GOMAXPROCS",
        "GC", "KeepAlive", "SetFinalizer",
        "Caller", "Callers", "Stack",
    ],
    "database/sql": [
        "Open", "OpenDB",
        "DB", "Tx", "Conn", "Stmt", "Rows", "Row",
        "Result", "Scanner", "Driver",
        "ErrNoRows", "ErrConnDone", "ErrTxDone",
        "Named", "NullBool", "NullInt64", "NullFloat64", "NullString", "NullTime",
    ],
}

# Popular third-party packages
GO_POPULAR_PACKAGES: Dict[str, List[str]] = {
    "github.com/gin-gonic/gin": [
        "Default", "New", "Engine", "Context",
        "H", "RouterGroup", "HandlerFunc",
    ],
    "github.com/gorilla/mux": [
        "NewRouter", "Router", "Route",
        "Vars", "Walk",
    ],
    "github.com/stretchr/testify/assert": [
        "Equal", "NotEqual", "Nil", "NotNil",
        "True", "False", "Error", "NoError",
        "Contains", "Len", "Empty", "NotEmpty",
    ],
    "github.com/stretchr/testify/require": [
        "Equal", "NotEqual", "Nil", "NotNil",
        "True", "False", "Error", "NoError",
    ],
    "github.com/pkg/errors": [
        "New", "Errorf", "Wrap", "Wrapf",
        "Cause", "WithStack", "WithMessage",
    ],
    "go.uber.org/zap": [
        "NewProduction", "NewDevelopment", "NewExample",
        "Logger", "SugaredLogger", "Config",
        "L", "S", "ReplaceGlobals",
    ],
    "gorm.io/gorm": [
        "Open", "DB", "Model", "Config",
        "ErrRecordNotFound",
    ],
}


class GoImportResolver(ImportResolver):
    """Go import resolver."""

    @property
    def language(self) -> str:
        return "go"

    def resolve(self, import_path: str) -> ImportResolution:
        """Resolve a Go import path."""
        # Check standard library
        if import_path in GO_STANDARD_LIBRARY:
            exports = set(GO_STANDARD_LIBRARY[import_path])
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

        # Check nested standard library paths
        for pkg in GO_STANDARD_LIBRARY:
            if import_path.startswith(pkg + "/"):
                # Sub-package
                return ImportResolution(
                    status=ResolutionStatus.RESOLVED,
                    module=ResolvedModule(
                        name=import_path,
                        path=import_path,
                        exports=set(),
                        is_builtin=True,
                        is_available=True,
                    ),
                    module_name=import_path,
                    exports=set(),
                )

        # Check popular packages
        if import_path in GO_POPULAR_PACKAGES:
            exports = set(GO_POPULAR_PACKAGES[import_path])
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

        # Unknown package - still return resolution to allow generation
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
        if import_path in GO_STANDARD_LIBRARY:
            return GO_STANDARD_LIBRARY[import_path]
        if import_path in GO_POPULAR_PACKAGES:
            return GO_POPULAR_PACKAGES[import_path]
        return []

    def get_completion_candidates(self, prefix: str) -> List[str]:
        """Get import path completion candidates."""
        candidates = []

        # Standard library
        for pkg in GO_STANDARD_LIBRARY:
            if pkg.startswith(prefix):
                candidates.append(pkg)

        # Popular packages
        for pkg in GO_POPULAR_PACKAGES:
            if pkg.startswith(prefix):
                candidates.append(pkg)

        return sorted(candidates)

    def is_available(self, module_name: str) -> bool:
        """Check if a Go module is available for import.

        Args:
            module_name: Module path to check

        Returns:
            True if the module can be imported (always True for known packages)
        """
        # Standard library is always available
        if module_name in GO_STANDARD_LIBRARY:
            return True
        # Check nested standard library
        for pkg in GO_STANDARD_LIBRARY:
            if module_name.startswith(pkg + "/"):
                return True
        # Popular packages are considered available
        if module_name in GO_POPULAR_PACKAGES:
            return True
        # Unknown packages are still allowed
        return True

    def get_version(self, module_name: str) -> Optional[str]:
        """Get the version of a Go module.

        Args:
            module_name: Module path

        Returns:
            Version string (None for standard library, would require go.mod parsing)
        """
        # Standard library doesn't have separate versions
        if module_name in GO_STANDARD_LIBRARY:
            return None
        # Would need to parse go.mod for actual version info
        return None


def create_go_resolver() -> GoImportResolver:
    """Factory function to create a Go import resolver."""
    return GoImportResolver()
