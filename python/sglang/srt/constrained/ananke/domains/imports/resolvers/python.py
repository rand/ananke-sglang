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
"""Python import resolver for stdlib and pip packages.

Resolves Python imports by checking:
1. Standard library modules
2. Installed pip packages
3. Built-in modules
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Optional, Set

from .base import ImportResolver, ImportResolution, ResolvedModule


# Python standard library modules (Python 3.9+)
PYTHON_STDLIB = {
    # Built-in
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
    "asyncore", "atexit", "audioop", "base64", "bdb", "binascii",
    "binhex", "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb",
    "chunk", "cmath", "cmd", "code", "codecs", "codeop", "collections",
    "colorsys", "compileall", "concurrent", "configparser", "contextlib",
    "contextvars", "copy", "copyreg", "cProfile", "crypt", "csv",
    "ctypes", "curses", "dataclasses", "datetime", "dbm", "decimal",
    "difflib", "dis", "distutils", "doctest", "email", "encodings",
    "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput",
    "fnmatch", "fractions", "ftplib", "functools", "gc", "getopt",
    "getpass", "gettext", "glob", "graphlib", "grp", "gzip", "hashlib",
    "heapq", "hmac", "html", "http", "idlelib", "imaplib", "imghdr",
    "imp", "importlib", "inspect", "io", "ipaddress", "itertools",
    "json", "keyword", "lib2to3", "linecache", "locale", "logging",
    "lzma", "mailbox", "mailcap", "marshal", "math", "mimetypes",
    "mmap", "modulefinder", "multiprocessing", "netrc", "nis", "nntplib",
    "numbers", "operator", "optparse", "os", "ossaudiodev", "pathlib",
    "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
    "plistlib", "poplib", "posix", "posixpath", "pprint", "profile",
    "pstats", "pty", "pwd", "py_compile", "pyclbr", "pydoc", "queue",
    "quopri", "random", "re", "readline", "reprlib", "resource",
    "rlcompleter", "runpy", "sched", "secrets", "select", "selectors",
    "shelve", "shlex", "shutil", "signal", "site", "smtpd", "smtplib",
    "sndhdr", "socket", "socketserver", "spwd", "sqlite3", "ssl",
    "stat", "statistics", "string", "stringprep", "struct", "subprocess",
    "sunau", "symtable", "sys", "sysconfig", "syslog", "tabnanny",
    "tarfile", "telnetlib", "tempfile", "termios", "test", "textwrap",
    "threading", "time", "timeit", "tkinter", "token", "tokenize",
    "trace", "traceback", "tracemalloc", "tty", "turtle", "turtledemo",
    "types", "typing", "unicodedata", "unittest", "urllib", "uu",
    "uuid", "venv", "warnings", "wave", "weakref", "webbrowser",
    "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc",
    "zipapp", "zipfile", "zipimport", "zlib",
    # Added in Python 3.9+
    "zoneinfo", "graphlib",
    # Added in Python 3.10+
    "tomllib",
    # Common submodules
    "collections.abc", "concurrent.futures", "email.mime",
    "http.client", "http.server", "importlib.metadata",
    "importlib.resources", "logging.handlers", "os.path",
    "typing_extensions", "unittest.mock", "urllib.parse",
    "urllib.request", "xml.etree", "xml.etree.ElementTree",
}


class PythonImportResolver(ImportResolver):
    """Import resolver for Python modules.

    Checks:
    1. Standard library (using stdlib list)
    2. Installed packages (using importlib.util)
    3. Built-in modules (sys.builtin_module_names)
    """

    def __init__(self, check_installed: bool = True):
        """Initialize the resolver.

        Args:
            check_installed: Whether to check for installed packages
        """
        self._check_installed = check_installed
        self._cache: dict[str, ImportResolution] = {}

    @property
    def language(self) -> str:
        return "python"

    def resolve(self, module_name: str) -> ImportResolution:
        """Resolve a Python module.

        Args:
            module_name: Module name (e.g., "numpy", "os.path")

        Returns:
            ImportResolution with module info
        """
        # Check cache
        if module_name in self._cache:
            return self._cache[module_name]

        # Check if it's a stdlib module
        base_module = module_name.split(".")[0]
        is_stdlib = (
            module_name in PYTHON_STDLIB
            or base_module in PYTHON_STDLIB
            or base_module in sys.builtin_module_names
        )

        if is_stdlib:
            result = ImportResolution(
                success=True,
                module=ResolvedModule(
                    name=module_name,
                    is_builtin=base_module in sys.builtin_module_names,
                    is_available=True,
                ),
            )
            self._cache[module_name] = result
            return result

        # Check if installed (if enabled)
        if self._check_installed:
            try:
                spec = importlib.util.find_spec(base_module)
                if spec is not None:
                    version = self._get_package_version(base_module)
                    result = ImportResolution(
                        success=True,
                        module=ResolvedModule(
                            name=module_name,
                            version=version,
                            path=str(spec.origin) if spec.origin else None,
                            is_available=True,
                        ),
                    )
                    self._cache[module_name] = result
                    return result
            except (ImportError, ModuleNotFoundError, ValueError):
                pass

        # Not found
        result = ImportResolution(
            success=False,
            error=f"Module '{module_name}' not found",
            alternatives=self.suggest_alternatives(module_name),
        )
        self._cache[module_name] = result
        return result

    def is_available(self, module_name: str) -> bool:
        """Check if a Python module is available."""
        resolution = self.resolve(module_name)
        return resolution.success

    def get_version(self, module_name: str) -> Optional[str]:
        """Get the installed version of a module."""
        resolution = self.resolve(module_name)
        if resolution.success and resolution.module:
            return resolution.module.version
        return None

    def _get_package_version(self, package_name: str) -> Optional[str]:
        """Get version of an installed package."""
        try:
            import importlib.metadata
            return importlib.metadata.version(package_name)
        except Exception:
            return None

    def get_exports(self, module_name: str) -> Set[str]:
        """Get exported names from a module.

        Note: This actually imports the module, which may have side effects.
        Use with caution.
        """
        if not self._check_installed:
            return set()

        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "__all__"):
                return set(module.__all__)
            return {
                name for name in dir(module)
                if not name.startswith("_")
            }
        except Exception:
            return set()

    def suggest_alternatives(self, module_name: str) -> list[str]:
        """Suggest alternative modules."""
        # Common misspellings and alternatives
        alternatives_map = {
            "numpy": ["scipy", "pandas"],
            "np": ["numpy"],
            "pd": ["pandas"],
            "plt": ["matplotlib.pyplot"],
            "tf": ["tensorflow"],
            "torch": ["pytorch"],
            "sklearn": ["scikit-learn"],
            "cv2": ["opencv-python"],
        }

        suggestions = alternatives_map.get(module_name, [])

        # Check for similar stdlib modules
        for stdlib_mod in PYTHON_STDLIB:
            if stdlib_mod.startswith(module_name[:3]) and stdlib_mod != module_name:
                suggestions.append(stdlib_mod)
                if len(suggestions) > 5:
                    break

        return suggestions[:5]

    def is_stdlib(self, module_name: str) -> bool:
        """Check if a module is part of the standard library."""
        base = module_name.split(".")[0]
        return (
            module_name in PYTHON_STDLIB
            or base in PYTHON_STDLIB
            or base in sys.builtin_module_names
        )


def create_python_resolver(check_installed: bool = True) -> PythonImportResolver:
    """Factory function to create a Python import resolver.

    Args:
        check_installed: Whether to check for installed packages

    Returns:
        Configured PythonImportResolver
    """
    return PythonImportResolver(check_installed=check_installed)
