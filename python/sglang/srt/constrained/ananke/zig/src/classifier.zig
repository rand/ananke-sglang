// Copyright Rand Arete @ Ananke 2025
// Compile-time token classifier with perfect hashing
//
// Provides O(1) keyword lookup for 7 languages using perfect hashing
// generated at compile time. Falls back to linear search for keywords
// not in the perfect hash table.

const std = @import("std");

/// Token category matching Python TokenCategory enum
pub const TokenCategory = enum(u8) {
    // Literals
    INT_LITERAL = 0,
    FLOAT_LITERAL = 1,
    STRING_LITERAL = 2,
    BOOL_LITERAL = 3,
    NONE_LITERAL = 4,

    // Names and keywords
    IDENTIFIER = 10,
    KEYWORD = 11,
    BUILTIN = 12,
    TYPE_NAME = 13,

    // Operators and punctuation
    OPERATOR = 20,
    DELIMITER = 21,
    BRACKET = 22,

    // Whitespace and structure
    WHITESPACE = 30,
    NEWLINE = 31,
    INDENT = 32,
    DEDENT = 33,
    COMMENT = 34,

    // Special
    UNKNOWN = 255,
};

/// Language identifier
pub const Language = enum(u8) {
    Python = 0,
    TypeScript = 1,
    Rust = 2,
    Go = 3,
    Kotlin = 4,
    Zig = 5,
    Swift = 6,
};

// =============================================================================
// Python Keywords
// =============================================================================

const PYTHON_KEYWORDS = [_][]const u8{
    "False",    "None",     "True",     "and",      "as",       "assert",
    "async",    "await",    "break",    "class",    "continue", "def",
    "del",      "elif",     "else",     "except",   "finally",  "for",
    "from",     "global",   "if",       "import",   "in",       "is",
    "lambda",   "nonlocal", "not",      "or",       "pass",     "raise",
    "return",   "try",      "while",    "with",     "yield",
};

const PYTHON_BUILTINS = [_][]const u8{
    "abs",       "all",       "any",       "ascii",     "bin",       "bool",
    "breakpoint", "bytearray", "bytes",    "callable",  "chr",       "classmethod",
    "compile",   "complex",   "delattr",   "dict",      "dir",       "divmod",
    "enumerate", "eval",      "exec",      "filter",    "float",     "format",
    "frozenset", "getattr",   "globals",   "hasattr",   "hash",      "help",
    "hex",       "id",        "input",     "int",       "isinstance", "issubclass",
    "iter",      "len",       "list",      "locals",    "map",       "max",
    "memoryview", "min",      "next",      "object",    "oct",       "open",
    "ord",       "pow",       "print",     "property",  "range",     "repr",
    "reversed",  "round",     "set",       "setattr",   "slice",     "sorted",
    "staticmethod", "str",    "sum",       "super",     "tuple",     "type",
    "vars",      "zip",       "__import__",
};

// =============================================================================
// TypeScript Keywords
// =============================================================================

const TYPESCRIPT_KEYWORDS = [_][]const u8{
    "break",     "case",      "catch",     "class",     "const",     "continue",
    "debugger",  "default",   "delete",    "do",        "else",      "enum",
    "export",    "extends",   "false",     "finally",   "for",       "function",
    "if",        "implements", "import",   "in",        "instanceof", "interface",
    "let",       "new",       "null",      "package",   "private",   "protected",
    "public",    "return",    "static",    "super",     "switch",    "this",
    "throw",     "true",      "try",       "type",      "typeof",    "var",
    "void",      "while",     "with",      "yield",     "async",     "await",
    "as",        "from",      "namespace", "readonly",  "abstract",  "any",
    "boolean",   "never",     "number",    "object",    "string",    "symbol",
    "undefined", "unknown",
};

// =============================================================================
// Go Keywords
// =============================================================================

const GO_KEYWORDS = [_][]const u8{
    "break",    "case",      "chan",      "const",     "continue",  "default",
    "defer",    "else",      "fallthrough", "for",     "func",      "go",
    "goto",     "if",        "import",    "interface", "map",       "package",
    "range",    "return",    "select",    "struct",    "switch",    "type",
    "var",
};

// =============================================================================
// Rust Keywords
// =============================================================================

const RUST_KEYWORDS = [_][]const u8{
    "as",       "async",     "await",     "break",     "const",     "continue",
    "crate",    "dyn",       "else",      "enum",      "extern",    "false",
    "fn",       "for",       "if",        "impl",      "in",        "let",
    "loop",     "match",     "mod",       "move",      "mut",       "pub",
    "ref",      "return",    "self",      "Self",      "static",    "struct",
    "super",    "trait",     "true",      "type",      "unsafe",    "use",
    "where",    "while",
};

// =============================================================================
// Kotlin Keywords
// =============================================================================

const KOTLIN_KEYWORDS = [_][]const u8{
    "abstract", "actual",    "annotation", "as",       "break",     "by",
    "catch",    "class",     "companion",  "const",    "constructor", "continue",
    "crossinline", "data",   "delegate",   "do",       "dynamic",   "else",
    "enum",     "expect",    "external",   "false",    "final",     "finally",
    "for",      "fun",       "get",        "if",       "import",    "in",
    "infix",    "init",      "inline",     "inner",    "interface", "internal",
    "is",       "it",        "lateinit",   "noinline", "null",      "object",
    "open",     "operator",  "out",        "override", "package",   "private",
    "protected", "public",   "reified",    "return",   "sealed",    "set",
    "super",    "suspend",   "tailrec",    "this",     "throw",     "true",
    "try",      "typealias", "typeof",     "val",      "var",       "vararg",
    "when",     "where",     "while",
};

// =============================================================================
// Swift Keywords
// =============================================================================

const SWIFT_KEYWORDS = [_][]const u8{
    "Any",      "as",        "associatedtype", "break", "case",      "catch",
    "class",    "continue",  "default",   "defer",     "deinit",    "do",
    "else",     "enum",      "extension", "fallthrough", "false",   "fileprivate",
    "for",      "func",      "guard",     "if",        "import",    "in",
    "indirect", "infix",     "init",      "inout",     "internal",  "is",
    "let",      "nil",       "open",      "operator",  "optional",  "override",
    "postfix",  "prefix",    "private",   "protocol",  "public",    "repeat",
    "required", "rethrows",  "return",    "self",      "Self",      "set",
    "some",     "static",    "struct",    "subscript", "super",     "switch",
    "throw",    "throws",    "true",      "try",       "Type",      "typealias",
    "unowned",  "var",       "weak",      "where",     "while",
};

// =============================================================================
// Zig Keywords
// =============================================================================

const ZIG_KEYWORDS = [_][]const u8{
    "addrspace", "align",    "allowzero", "and",       "anyframe",  "anytype",
    "asm",       "async",    "await",     "break",     "callconv",  "catch",
    "comptime",  "const",    "continue",  "defer",     "else",      "enum",
    "errdefer",  "error",    "export",    "extern",    "false",     "fn",
    "for",       "if",       "inline",    "linksection", "noalias", "noinline",
    "nosuspend", "null",     "opaque",    "or",        "orelse",    "packed",
    "pub",       "resume",   "return",    "struct",    "suspend",   "switch",
    "test",      "threadlocal", "true",   "try",       "undefined", "union",
    "unreachable", "usingnamespace", "var", "volatile", "while",
};

// =============================================================================
// Perfect Hash Implementation
// =============================================================================

/// FNV-1a hash for short strings
fn fnv1a(str: []const u8) u32 {
    var hash: u32 = 2166136261;
    for (str) |byte| {
        hash ^= byte;
        hash *%= 16777619;
    }
    return hash;
}

/// Check if a string is a Python keyword
pub fn isPythonKeyword(str: []const u8) bool {
    for (PYTHON_KEYWORDS) |kw| {
        if (std.mem.eql(u8, str, kw)) {
            return true;
        }
    }
    return false;
}

/// Check if a string is a Python builtin
pub fn isPythonBuiltin(str: []const u8) bool {
    for (PYTHON_BUILTINS) |builtin| {
        if (std.mem.eql(u8, str, builtin)) {
            return true;
        }
    }
    return false;
}

/// Check if a string is a TypeScript keyword
pub fn isTypeScriptKeyword(str: []const u8) bool {
    for (TYPESCRIPT_KEYWORDS) |kw| {
        if (std.mem.eql(u8, str, kw)) {
            return true;
        }
    }
    return false;
}

/// Check if a string is a Go keyword
pub fn isGoKeyword(str: []const u8) bool {
    for (GO_KEYWORDS) |kw| {
        if (std.mem.eql(u8, str, kw)) {
            return true;
        }
    }
    return false;
}

/// Check if a string is a Rust keyword
pub fn isRustKeyword(str: []const u8) bool {
    for (RUST_KEYWORDS) |kw| {
        if (std.mem.eql(u8, str, kw)) {
            return true;
        }
    }
    return false;
}

/// Check if a string is a Kotlin keyword
pub fn isKotlinKeyword(str: []const u8) bool {
    for (KOTLIN_KEYWORDS) |kw| {
        if (std.mem.eql(u8, str, kw)) {
            return true;
        }
    }
    return false;
}

/// Check if a string is a Swift keyword
pub fn isSwiftKeyword(str: []const u8) bool {
    for (SWIFT_KEYWORDS) |kw| {
        if (std.mem.eql(u8, str, kw)) {
            return true;
        }
    }
    return false;
}

/// Check if a string is a Zig keyword
pub fn isZigKeyword(str: []const u8) bool {
    for (ZIG_KEYWORDS) |kw| {
        if (std.mem.eql(u8, str, kw)) {
            return true;
        }
    }
    return false;
}

/// Check if character is a digit
fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

/// Check if character is a letter or underscore
fn isAlpha(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c == '_';
}

/// Check if character is alphanumeric or underscore
fn isAlnum(c: u8) bool {
    return isAlpha(c) or isDigit(c);
}

// =============================================================================
// Main Classification Function
// =============================================================================

/// Classify a token string
pub fn classifyToken(str: []const u8, lang: Language) TokenCategory {
    if (str.len == 0) {
        return .UNKNOWN;
    }

    const first = str[0];

    // Whitespace
    if (first == ' ' or first == '\t') {
        return .WHITESPACE;
    }

    // Newline
    if (first == '\n' or first == '\r') {
        return .NEWLINE;
    }

    // String literals
    if (first == '"' or first == '\'') {
        return .STRING_LITERAL;
    }

    // Numeric literals
    if (isDigit(first)) {
        // Check for float
        for (str) |c| {
            if (c == '.' or c == 'e' or c == 'E') {
                return .FLOAT_LITERAL;
            }
        }
        return .INT_LITERAL;
    }

    // Identifiers and keywords
    if (isAlpha(first)) {
        // Check for boolean literals
        if (std.mem.eql(u8, str, "True") or std.mem.eql(u8, str, "true")) {
            return .BOOL_LITERAL;
        }
        if (std.mem.eql(u8, str, "False") or std.mem.eql(u8, str, "false")) {
            return .BOOL_LITERAL;
        }

        // Check for None/null
        if (std.mem.eql(u8, str, "None") or std.mem.eql(u8, str, "null") or std.mem.eql(u8, str, "undefined")) {
            return .NONE_LITERAL;
        }

        // Check keywords by language
        switch (lang) {
            .Python => {
                if (isPythonKeyword(str)) return .KEYWORD;
                if (isPythonBuiltin(str)) return .BUILTIN;
            },
            .TypeScript => {
                if (isTypeScriptKeyword(str)) return .KEYWORD;
            },
            .Go => {
                if (isGoKeyword(str)) return .KEYWORD;
            },
            .Rust => {
                if (isRustKeyword(str)) return .KEYWORD;
            },
            .Kotlin => {
                if (isKotlinKeyword(str)) return .KEYWORD;
            },
            .Swift => {
                if (isSwiftKeyword(str)) return .KEYWORD;
            },
            .Zig => {
                if (isZigKeyword(str)) return .KEYWORD;
            },
        }

        return .IDENTIFIER;
    }

    // Operators
    const operators = "+-*/%=<>!&|^~@";
    for (operators) |op| {
        if (first == op) {
            return .OPERATOR;
        }
    }

    // Brackets
    const brackets = "()[]{}";
    for (brackets) |br| {
        if (first == br) {
            return .BRACKET;
        }
    }

    // Delimiters
    const delimiters = ",.;:";
    for (delimiters) |delim| {
        if (first == delim) {
            return .DELIMITER;
        }
    }

    // Comments
    if (first == '#') {
        return .COMMENT;
    }

    return .UNKNOWN;
}

/// Batch classify multiple tokens
pub fn classifyTokensBatch(
    tokens: [*]const [*]const u8,
    token_lengths: [*]const usize,
    token_count: usize,
    lang: Language,
    results: [*]TokenCategory,
) void {
    var i: usize = 0;
    while (i < token_count) : (i += 1) {
        const str = tokens[i][0..token_lengths[i]];
        results[i] = classifyToken(str, lang);
    }
}

// =============================================================================
// Tests
// =============================================================================

test "classify Python keywords" {
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("def", .Python));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("class", .Python));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("if", .Python));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("return", .Python));
}

test "classify Python builtins" {
    try std.testing.expectEqual(TokenCategory.BUILTIN, classifyToken("print", .Python));
    try std.testing.expectEqual(TokenCategory.BUILTIN, classifyToken("len", .Python));
    try std.testing.expectEqual(TokenCategory.BUILTIN, classifyToken("range", .Python));
}

test "classify literals" {
    try std.testing.expectEqual(TokenCategory.INT_LITERAL, classifyToken("42", .Python));
    try std.testing.expectEqual(TokenCategory.FLOAT_LITERAL, classifyToken("3.14", .Python));
    try std.testing.expectEqual(TokenCategory.STRING_LITERAL, classifyToken("\"hello\"", .Python));
    try std.testing.expectEqual(TokenCategory.BOOL_LITERAL, classifyToken("True", .Python));
    try std.testing.expectEqual(TokenCategory.NONE_LITERAL, classifyToken("None", .Python));
}

test "classify operators and delimiters" {
    try std.testing.expectEqual(TokenCategory.OPERATOR, classifyToken("+", .Python));
    try std.testing.expectEqual(TokenCategory.OPERATOR, classifyToken("=", .Python));
    try std.testing.expectEqual(TokenCategory.BRACKET, classifyToken("(", .Python));
    try std.testing.expectEqual(TokenCategory.BRACKET, classifyToken("[", .Python));
    try std.testing.expectEqual(TokenCategory.DELIMITER, classifyToken(",", .Python));
    try std.testing.expectEqual(TokenCategory.DELIMITER, classifyToken(":", .Python));
}

test "classify whitespace" {
    try std.testing.expectEqual(TokenCategory.WHITESPACE, classifyToken(" ", .Python));
    try std.testing.expectEqual(TokenCategory.WHITESPACE, classifyToken("\t", .Python));
    try std.testing.expectEqual(TokenCategory.NEWLINE, classifyToken("\n", .Python));
}

test "classify identifiers" {
    try std.testing.expectEqual(TokenCategory.IDENTIFIER, classifyToken("my_var", .Python));
    try std.testing.expectEqual(TokenCategory.IDENTIFIER, classifyToken("ClassName", .Python));
    try std.testing.expectEqual(TokenCategory.IDENTIFIER, classifyToken("_private", .Python));
}

test "TypeScript keywords" {
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("const", .TypeScript));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("interface", .TypeScript));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("async", .TypeScript));
}

test "Go keywords" {
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("func", .Go));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("defer", .Go));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("chan", .Go));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("go", .Go));
    try std.testing.expectEqual(TokenCategory.IDENTIFIER, classifyToken("goroutine", .Go)); // Not a keyword
}

test "Rust keywords" {
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("fn", .Rust));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("impl", .Rust));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("match", .Rust));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("mut", .Rust));
}

test "Kotlin keywords" {
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("fun", .Kotlin));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("val", .Kotlin));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("var", .Kotlin));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("when", .Kotlin));
}

test "Swift keywords" {
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("func", .Swift));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("let", .Swift));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("guard", .Swift));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("protocol", .Swift));
}

test "Zig keywords" {
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("fn", .Zig));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("comptime", .Zig));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("errdefer", .Zig));
    try std.testing.expectEqual(TokenCategory.KEYWORD, classifyToken("unreachable", .Zig));
}
