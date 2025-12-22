# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Rust ownership deep dive constraint examples for Ananke.

This module contains comprehensive examples demonstrating advanced Rust ownership,
lifetimes, and borrowing patterns including Pin, 'static bounds, and Arc<Mutex<T>>.
"""

from __future__ import annotations

from typing import List

try:
    from ..base import ConstraintExample
    from ....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        ImportBinding,
        SemanticConstraint,
        ControlFlowContext,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        ImportBinding,
        SemanticConstraint,
        ControlFlowContext,
    )

# =============================================================================
# Ownership Deep Dive Examples
# =============================================================================

RUST_OWNERSHIP_001 = ConstraintExample(
    id="rust-ownership-001",
    name="Pin<Box<T>> for Self-Referential Structs",
    description="Constraint generation for pinned heap allocations with self-references",
    scenario=(
        "Developer implementing an async Future with internal references. The struct "
        "contains a pointer to its own data, requiring Pin to prevent moves. "
        "The constraint ensures proper Pin::new_unchecked usage with safety invariants."
    ),
    prompt="""Create a pinned self-referential struct. Heap-allocate first, establish the pointer,
then pin with unsafe. Include SAFETY comment explaining why the unsafe is sound.

use std::pin::Pin;
use std::marker::PhantomPinned;

struct SelfRef { data: String, ptr: *const String, _pin: PhantomPinned }

fn new(data: String) -> Pin<Box<SelfRef>> {
    """,
    spec=ConstraintSpec(
        language="rust",
        imports=[
            ImportBinding(module="std::pin", name="Pin"),
            ImportBinding(module="std::marker", name="PhantomPinned"),
        ],
        type_bindings=[
            TypeBinding(
                name="data",
                type_expr="String",
                scope="local",
                mutable=False,
            ),
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="self_reference_valid_after_pin()",
                scope="new",
                variables=(),
            ),
            SemanticConstraint(
                kind="invariant",
                expression="unsafe_block_has_safety_doc()",
                scope="new",
                variables=(),
            ),
        ],
        ebnf=r'''
root ::= "let mut selfish = Box::new(SelfRef { data: data, ptr: std::ptr::null(), _pin: PhantomPinned });\nlet ptr = &selfish.data as *const String;\nlet selfish_pinned = unsafe {\n// SAFETY: selfish is heap-allocated and will not be moved after pinning\n    Pin::new_unchecked(selfish)\n};"
''',
    ),
    expected_effect=(
        "Masks tokens that create self-references without Pin or violate pinning guarantees. "
        "Ensures struct is heap-allocated, pointer established, then pinned with unsafe and SAFETY doc."
    ),
    valid_outputs=[
        """let mut selfish = Box::new(SelfRef { data: data, ptr: std::ptr::null(), _pin: PhantomPinned });
let ptr = &selfish.data as *const String;
let selfish_pinned = unsafe {
// SAFETY: selfish is heap-allocated and will not be moved after pinning
    Pin::new_unchecked(selfish)
};""",
    ],
    invalid_outputs=[
        """let selfish = SelfRef { data, ptr: &data, _pin: PhantomPinned };""",  # Stack allocated, can move
        """let selfish = Box::new(SelfRef { data, ptr: &data, _pin: PhantomPinned });""",  # Ptr established too early
        """let selfish_pinned = unsafe { Pin::new_unchecked(Box::new(SelfRef { data, ptr: std::ptr::null(), _pin: PhantomPinned })) };""",  # Missing SAFETY comment
    ],
    tags=["ownership", "pin", "self-referential", "unsafe"],
    language="rust",
    domain="types",
)

RUST_OWNERSHIP_002 = ConstraintExample(
    id="rust-ownership-002",
    name="Lifetime Inference with 'static Bounds",
    description="Constraint generation for functions requiring 'static lifetime on generic types",
    scenario=(
        "Developer implementing a spawn_task function that sends data to another thread. "
        "The generic type T must be 'static (no borrowed data) to safely move across threads. "
        "The constraint ensures generated function signature has 'static bound."
    ),
    prompt="""Write a spawn_task function that takes a callable and runs it in a new thread.
The generic type T must have 'static bound (no borrowed data) to safely move across threads.
Also needs Send since it crosses thread boundaries.

use std::thread;

""",
    spec=ConstraintSpec(
        language="rust",
        imports=[
            ImportBinding(module="std::thread"),
        ],
        type_bindings=[
            TypeBinding(
                name="task",
                type_expr="T",
                scope="parameter",
                mutable=False,
            ),
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="precondition",
                expression="T: 'static + Send",
                scope="spawn_task",
                variables=("T",),
            ),
        ],
        ebnf=r'''
root ::= "fn spawn_task<T: " bounds ">(task: T) -> thread::JoinHandle<()> {" nl "    " body nl "}"
bounds ::= "'static + Send + " trait
trait ::= "FnOnce()" | "Fn()"
body ::= "thread::spawn(move || task())"
nl ::= "\n"
''',
    ),
    expected_effect=(
        "Masks tokens producing function signatures without 'static bound. Ensures T cannot "
        "contain borrowed data that would become invalid when moved to spawned thread."
    ),
    valid_outputs=[
        "fn spawn_task<T: 'static + Send + FnOnce()>(task: T) -> thread::JoinHandle<()> {\n    thread::spawn(move || task())\n}",
        "fn spawn_task<T: 'static + Send + Fn()>(task: T) -> thread::JoinHandle<()> {\n    thread::spawn(move || task())\n}",
    ],
    invalid_outputs=[
        "fn spawn_task<T: Send + FnOnce()>(task: T) -> thread::JoinHandle<()> {\n    thread::spawn(move || task())\n}",  # Missing 'static
        "fn spawn_task<T: FnOnce()>(task: T) -> thread::JoinHandle<()> {\n    thread::spawn(move || task())\n}",  # Missing 'static and Send
        "fn spawn_task<T>(task: T) -> thread::JoinHandle<()> {\n    thread::spawn(move || task())\n}",  # No bounds at all
    ],
    tags=["ownership", "lifetimes", "static", "threading"],
    language="rust",
    domain="types",
)

RUST_OWNERSHIP_003 = ConstraintExample(
    id="rust-ownership-003",
    name="Arc<Mutex<T>> Shared Mutable State",
    description="Constraint generation for thread-safe shared mutable state pattern",
    scenario=(
        "Developer implementing a shared counter accessed by multiple threads. "
        "Arc provides shared ownership, Mutex enables interior mutability. "
        "The constraint ensures proper clone for Arc and lock/unwrap for access."
    ),
    prompt="""Implement a shared counter incremented by multiple threads. Use Arc for shared ownership
and Mutex for interior mutability. Clone Arc before moving to threads, lock before mutating.

use std::sync::{Arc, Mutex};
use std::thread;

fn increment_counter() {
    """,
    spec=ConstraintSpec(
        language="rust",
        imports=[
            ImportBinding(module="std::sync", name="Arc"),
            ImportBinding(module="std::sync", name="Mutex"),
            ImportBinding(module="std::thread"),
        ],
        type_bindings=[
            TypeBinding(
                name="counter",
                type_expr="Arc<Mutex<i32>>",
                scope="local",
                mutable=False,
            ),
        ],
        control_flow=ControlFlowContext(
            function_name="increment_counter",
        ),
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="no_deadlocks_possible()",
                scope="increment_counter",
                variables=(),
            ),
        ],
        ebnf=r'''
root ::= setup nl threads nl joins
setup ::= "let counter = Arc::new(Mutex::new(0));"
threads ::= thread (nl thread)*
thread ::= "let counter" digit " = Arc::clone(&counter);" nl "let handle" digit " = thread::spawn(move || {" nl "    let mut num = counter" digit ".lock().unwrap();" nl "    *num += 1;" nl "});"
digit ::= [0-9]
joins ::= join (nl join)*
join ::= "handle" digit ".join().unwrap();"
nl ::= "\n"
''',
    ),
    expected_effect=(
        "Masks tokens that access Arc without clone or Mutex without lock. Ensures "
        "Arc::clone creates new reference counts and lock() is called before mutation."
    ),
    valid_outputs=[
        """let counter = Arc::new(Mutex::new(0));
let counter1 = Arc::clone(&counter);
let handle1 = thread::spawn(move || {
    let mut num = counter1.lock().unwrap();
    *num += 1;
});
let counter2 = Arc::clone(&counter);
let handle2 = thread::spawn(move || {
    let mut num = counter2.lock().unwrap();
    *num += 1;
});
handle1.join().unwrap();
handle2.join().unwrap();""",
    ],
    invalid_outputs=[
        """let counter = Mutex::new(0);
thread::spawn(move || { counter.lock().unwrap(); });""",  # Missing Arc, can't share
        """let counter = Arc::new(Mutex::new(0));
thread::spawn(move || { *counter += 1; });""",  # Missing lock()
        """let counter = Arc::new(Mutex::new(0));
let handle = thread::spawn(|| { counter.lock().unwrap(); });""",  # Missing move, won't compile
    ],
    tags=["ownership", "arc", "mutex", "concurrency"],
    language="rust",
    domain="types",
)

RUST_OWNERSHIP_004 = ConstraintExample(
    id="rust-ownership-004",
    name="Lifetime Elision and Explicit Annotations",
    description="Constraint generation demonstrating when explicit lifetime annotations are required",
    scenario=(
        "Developer writing a function returning the longer of two string slices. "
        "Rust's lifetime elision rules don't apply when multiple input lifetimes exist. "
        "The constraint ensures explicit lifetime 'a is present to tie inputs to output."
    ),
    prompt="""Write a function that returns the longer of two string slices.
You need explicit lifetime annotations because there are two input references -
Rust's elision rules don't apply here. Use 'a for all three references.

""",
    spec=ConstraintSpec(
        language="rust",
        type_bindings=[
            TypeBinding(name="x", type_expr="&'a str", scope="parameter"),
            TypeBinding(name="y", type_expr="&'a str", scope="parameter"),
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="lifetime_valid(result, min_lifetime(x, y))",
                scope="longest",
                variables=("x", "y", "result"),
            ),
        ],
        ebnf=r'''
root ::= "fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {" nl "    " body nl "}"
body ::= "if x.len() > y.len() { x } else { y }" | "if x.len() >= y.len() { x } else { y }"
nl ::= "\n"
''',
    ),
    expected_effect=(
        "Masks tokens producing function signatures without lifetime parameters. "
        "Ensures 'a is declared and consistently used on both inputs and output."
    ),
    valid_outputs=[
        "fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {\n    if x.len() > y.len() { x } else { y }\n}",
        "fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {\n    if x.len() >= y.len() { x } else { y }\n}",
    ],
    invalid_outputs=[
        "fn longest(x: &str, y: &str) -> &str {\n    if x.len() > y.len() { x } else { y }\n}",  # Missing lifetime annotations
        "fn longest<'a>(x: &'a str, y: &str) -> &'a str {\n    if x.len() > y.len() { x } else { y }\n}",  # y missing 'a
        "fn longest<'a, 'b>(x: &'a str, y: &'b str) -> &'a str {\n    if x.len() > y.len() { x } else { y }\n}",  # Can't return y with 'a
    ],
    tags=["ownership", "lifetimes", "elision", "borrowing"],
    language="rust",
    domain="types",
)

RUST_OWNERSHIP_005 = ConstraintExample(
    id="rust-ownership-005",
    name="Interior Mutability with RefCell",
    description="Constraint generation for runtime borrow checking with RefCell",
    scenario=(
        "Developer implementing a graph node with internal caching that mutates through shared refs. "
        "RefCell allows mutation through &self by deferring borrow checks to runtime. "
        "The constraint ensures borrow_mut() is used correctly and panics are documented."
    ),
    prompt="""Implement a get_cached method that returns cached value or computes and caches it.
Use RefCell for interior mutability through &self. Make sure borrow() is dropped before
calling borrow_mut() - holding both would panic at runtime.

use std::cell::RefCell;

impl Node {
    fn get_cached(&self) -> Value {
        """,
    spec=ConstraintSpec(
        language="rust",
        imports=[
            ImportBinding(module="std::cell", name="RefCell"),
        ],
        type_bindings=[
            TypeBinding(
                name="cache",
                type_expr="RefCell<Option<Value>>",
                scope="class:Node",
            ),
        ],
        semantic_constraints=[
            SemanticConstraint(
                kind="invariant",
                expression="no_overlapping_borrows()",
                scope="get_cached",
                variables=(),
            ),
        ],
        ebnf=r'''
root ::= if_block nl compute nl update nl return_new
if_block ::= "if let cached = self.cache.borrow(); cached.is_some() {" nl "    " return_cached nl "}"
return_cached ::= "return cached.as_ref().unwrap().clone();"
compute ::= "let value = self.compute_expensive();"
update ::= "*self.cache.borrow_mut() = Some(value.clone());"
return_new ::= "value"
nl ::= "\n"
''',
    ),
    expected_effect=(
        "Masks tokens that hold borrow() while calling borrow_mut(), causing runtime panic. "
        "Ensures borrows are dropped (scoped) before mutation via borrow_mut()."
    ),
    valid_outputs=[
        """if let cached = self.cache.borrow(); cached.is_some() {
    return cached.as_ref().unwrap().clone();
}
let value = self.compute_expensive();
*self.cache.borrow_mut() = Some(value.clone());
value""",
    ],
    invalid_outputs=[
        """let cached = self.cache.borrow();
if cached.is_some() {
    return cached.as_ref().unwrap().clone();
}
*self.cache.borrow_mut() = Some(value);""",  # Borrow still held during borrow_mut
        """self.cache.borrow_mut().get_or_insert_with(|| self.compute_expensive())""",  # Calling method while holding mutable borrow
    ],
    tags=["ownership", "interior-mutability", "refcell", "borrowing"],
    language="rust",
    domain="types",
)

# =============================================================================
# Exports
# =============================================================================

RUST_OWNERSHIP_EXAMPLES: List[ConstraintExample] = [
    RUST_OWNERSHIP_001,
    RUST_OWNERSHIP_002,
    RUST_OWNERSHIP_003,
    RUST_OWNERSHIP_004,
    RUST_OWNERSHIP_005,
]

__all__ = ["RUST_OWNERSHIP_EXAMPLES"]
