# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Control flow constraint examples for Python.

This module contains realistic examples of control flow constraints that
demonstrate how Ananke's ControlFlowDomain masks tokens based on control
flow context like loops, async/await, try/except blocks.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        ControlFlowContext,
        TypeBinding,
        FunctionSignature,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        ControlFlowContext,
        TypeBinding,
        FunctionSignature,
    )

PYTHON_CONTROLFLOW_EXAMPLES = [
    ConstraintExample(
        id="py-controlflow-001",
        name="Async Handler Required",
        description="Must use await for async operations in async function",
        scenario=(
            "Developer writing an async HTTP handler that must await all async "
            "operations. The function is declared async, so calling other async "
            "functions without await would create unawaited coroutines."
        ),
        prompt="""I'm writing an async HTTP handler. All async operations must use await.
Don't use blocking calls like time.sleep().

async def handle_request(request: Request) -> Response:
    """,
        spec=ConstraintSpec(
            language="python",
            # Regex enforces await usage - must have 'await' keyword
            regex=r"\bawait\s+\w+",
            control_flow=ControlFlowContext(
                function_name="handle_request",
                function_signature=FunctionSignature(
                    name="handle_request",
                    params=(TypeBinding(name="request", type_expr="Request"),),
                    return_type="Response",
                    is_async=True,
                ),
                expected_return_type="Response",
                in_async_context=True,
            ),
            type_bindings=[
                TypeBinding(name="request", type_expr="Request", scope="parameter"),
                TypeBinding(name="db", type_expr="AsyncDatabase", scope="local"),
            ],
        ),
        expected_effect=(
            "Masks tokens that would call async functions without await. Blocks "
            "patterns like 'db.query()' and requires 'await db.query()'. Also "
            "blocks synchronous blocking calls like time.sleep() in favor of "
            "asyncio.sleep()."
        ),
        valid_outputs=[
            "data = await db.query(request.user_id)",
            "result = await asyncio.gather(task1, task2)",
            "await asyncio.sleep(1.0)",
            "response = await fetch_data(request.url)",
        ],
        invalid_outputs=[
            "data = db.query(request.user_id)",  # Missing await
            "result = asyncio.gather(task1, task2)",  # Missing await
            "time.sleep(1.0)",  # Blocking call in async context
        ],
        tags=["controlflow", "async", "await", "concurrency"],
        language="python",
        domain="controlflow",
    ),
    ConstraintExample(
        id="py-controlflow-002",
        name="Loop Context with break/continue",
        description="Allow break/continue only inside loops",
        scenario=(
            "Developer writing code inside a loop where break and continue are "
            "valid control flow statements. Outside loops, these statements would "
            "be syntax errors."
        ),
        prompt="""I need to skip certain items and stop early based on conditions inside this loop.
Use break and continue for flow control.

for index, item in enumerate(items):
    """,
        spec=ConstraintSpec(
            language="python",
            # Regex enforces loop control statements (continue/break)
            regex=r"^if\s+.+:\s*(continue|break)$",
            control_flow=ControlFlowContext(
                function_name="process_items",
                loop_depth=1,
                loop_variables=("item", "index"),
                in_async_context=False,
            ),
            type_bindings=[
                TypeBinding(name="items", type_expr="List[str]", scope="local"),
                TypeBinding(name="item", type_expr="str", scope="local"),
                TypeBinding(name="index", type_expr="int", scope="local"),
            ],
        ),
        expected_effect=(
            "When loop_depth > 0, allows break and continue statements. Masks "
            "these tokens when loop_depth == 0. Also tracks loop variables "
            "for scope analysis."
        ),
        valid_outputs=[
            "if item == 'skip': continue",
            "if item == 'stop': break",
            "if index > 100: break",
        ],
        invalid_outputs=[
            "return item",  # Returning from loop instead of using break
            "raise StopIteration",  # Using exception instead of break
            "items.remove(item)",  # Mutating collection during iteration
        ],
        tags=["controlflow", "loops", "break", "continue"],
        language="python",
        domain="controlflow",
    ),
    ConstraintExample(
        id="py-controlflow-003",
        name="Try Block Exception Handling",
        description="Handle specific exception types in try/except block",
        scenario=(
            "Developer writing error handling code inside a try block where specific "
            "exception types are expected. The except clauses should only catch "
            "IOError and ValueError, not broad Exception."
        ),
        prompt="""I need to handle file and parsing errors specifically - only catch IOError and ValueError.
Don't use broad Exception or bare except.

try:
    config = load_config(config_path)
""",
        spec=ConstraintSpec(
            language="python",
            # Regex enforces specific exception types only
            regex=r"^except\s+(?:IOError|ValueError|\(IOError,\s*ValueError\))(?:\s+as\s+\w+)?:",
            control_flow=ControlFlowContext(
                function_name="read_config",
                in_try_block=True,
                exception_types=("IOError", "ValueError"),
                expected_return_type="Optional[Config]",
            ),
            type_bindings=[
                TypeBinding(name="config_path", type_expr="str", scope="parameter"),
            ],
        ),
        expected_effect=(
            "Masks overly broad exception catching. Blocks 'except Exception:', "
            "'except BaseException:', and bare 'except:'. Allows only the specific "
            "exception types listed in exception_types tuple."
        ),
        valid_outputs=[
            "except IOError as e:\n    return None",
            "except ValueError as e:\n    logging.error(f'Invalid config: {e}')\n    return None",
            "except (IOError, ValueError) as e:\n    return None",
        ],
        invalid_outputs=[
            "except Exception as e: pass",  # Too broad
            "except BaseException as e: pass",  # Too broad
            "except: pass",  # Bare except
            "except KeyError as e: pass",  # Not in allowed exception_types
        ],
        tags=["controlflow", "exceptions", "error-handling", "try-except"],
        language="python",
        domain="controlflow",
    ),
]
