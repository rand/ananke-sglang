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
"""Unit tests for language detection."""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spec.language_detector import (
    DetectionResult,
    LanguageDetector,
    LanguageStackManager,
)
from spec.constraint_spec import LanguageFrame


class TestLanguageDetector:
    """Tests for LanguageDetector."""

    def test_detect_python(self) -> None:
        detector = LanguageDetector(use_tree_sitter=False)
        result = detector.detect_with_confidence("""
def foo(x: int) -> str:
    return str(x)

class MyClass:
    def __init__(self):
        self.value = 0
""")
        assert result.language == "python"
        assert result.confidence > 0.5

    def test_detect_typescript(self) -> None:
        detector = LanguageDetector(use_tree_sitter=False)
        result = detector.detect_with_confidence("""
interface User {
    name: string;
    age: number;
}

const getUserName = (user: User): string => {
    return user.name;
}
""")
        assert result.language == "typescript"
        assert result.confidence > 0.5

    def test_detect_go(self) -> None:
        detector = LanguageDetector(use_tree_sitter=False)
        result = detector.detect_with_confidence("""
package main

import "fmt"

func main() {
    x := 42
    fmt.Println(x)
}

type User struct {
    Name string
    Age  int
}
""")
        assert result.language == "go"
        assert result.confidence > 0.5

    def test_detect_rust(self) -> None:
        detector = LanguageDetector(use_tree_sitter=False)
        result = detector.detect_with_confidence("""
use std::collections::HashMap;

fn main() {
    let mut map: HashMap<String, i32> = HashMap::new();
    map.insert("key".to_string(), 42);
}

struct User {
    name: String,
    age: u32,
}

impl User {
    fn new(name: String) -> Self {
        Self { name, age: 0 }
    }
}
""")
        assert result.language == "rust"
        assert result.confidence > 0.5

    def test_detect_kotlin(self) -> None:
        detector = LanguageDetector(use_tree_sitter=False)
        result = detector.detect_with_confidence("""
data class User(val name: String, val age: Int)

fun processUser(user: User): String {
    return user.name
}

suspend fun fetchData(): List<String> {
    return listOf("a", "b", "c")
}
""")
        assert result.language == "kotlin"
        assert result.confidence > 0.5

    def test_detect_swift(self) -> None:
        detector = LanguageDetector(use_tree_sitter=False)
        result = detector.detect_with_confidence("""
protocol Drawable {
    func draw()
}

struct User {
    var name: String
    let age: Int

    func greet() -> String {
        return "Hello, \\(name)"
    }
}

extension User: Drawable {
    func draw() {
        print(name)
    }
}
""")
        assert result.language == "swift"
        assert result.confidence > 0.5

    def test_detect_zig(self) -> None:
        detector = LanguageDetector(use_tree_sitter=False)
        result = detector.detect_with_confidence("""
const std = @import("std");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit();

    try list.append(42);
}

fn add(a: i32, b: i32) i32 {
    return a + b;
}
""")
        assert result.language == "zig"
        assert result.confidence > 0.5

    def test_detect_empty_returns_default(self) -> None:
        detector = LanguageDetector(default_language="python")
        result = detector.detect_with_confidence("")
        assert result.language == "python"
        assert result.confidence == 0.0

    def test_detect_with_candidates(self) -> None:
        detector = LanguageDetector(use_tree_sitter=False)
        result = detector.detect_with_confidence(
            "def foo(): pass",
            candidates={"python", "typescript"},
        )
        assert result.language == "python"

    def test_default_language(self) -> None:
        detector = LanguageDetector(default_language="typescript")
        assert detector.detect("") == "typescript"


class TestLanguageStackManager:
    """Tests for LanguageStackManager."""

    def test_initial_state(self) -> None:
        manager = LanguageStackManager("python")
        assert manager.current_language == "python"
        assert len(manager.stack) == 1

    def test_push_pop(self) -> None:
        manager = LanguageStackManager("python")
        manager.push("sql", 100, delimiter="'''")

        assert manager.current_language == "sql"
        assert len(manager.stack) == 2

        popped = manager.pop()
        assert popped is not None
        assert popped.language == "sql"
        assert manager.current_language == "python"

    def test_cannot_pop_base(self) -> None:
        manager = LanguageStackManager("python")
        popped = manager.pop()
        assert popped is None
        assert manager.current_language == "python"

    def test_nested_contexts(self) -> None:
        manager = LanguageStackManager("python")
        manager.push("sql", 100)
        manager.push("javascript", 200)

        assert manager.current_language == "javascript"
        assert len(manager.stack) == 3

        manager.pop()
        assert manager.current_language == "sql"

        manager.pop()
        assert manager.current_language == "python"

    def test_language_at_position(self) -> None:
        manager = LanguageStackManager("python")
        manager.push("sql", 100)
        manager.push("javascript", 200)

        assert manager.language_at_position(50) == "python"
        assert manager.language_at_position(150) == "sql"
        assert manager.language_at_position(250) == "javascript"

    def test_current_frame(self) -> None:
        manager = LanguageStackManager("python")
        frame = manager.current_frame
        assert frame is not None
        assert frame.language == "python"
        assert frame.start_position == 0

        manager.push("sql", 100, delimiter="'''", end_delimiter="'''")
        frame = manager.current_frame
        assert frame is not None
        assert frame.language == "sql"
        assert frame.delimiter == "'''"
        assert frame.end_delimiter == "'''"


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_basic_construction(self) -> None:
        result = DetectionResult(
            language="python",
            confidence=0.85,
        )
        assert result.language == "python"
        assert result.confidence == 0.85
        assert result.method == "heuristic"

    def test_with_scores(self) -> None:
        result = DetectionResult(
            language="python",
            confidence=0.85,
            scores={"python": 0.85, "typescript": 0.15},
            method="tree_sitter",
        )
        assert result.scores["python"] == 0.85
        assert result.method == "tree_sitter"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
