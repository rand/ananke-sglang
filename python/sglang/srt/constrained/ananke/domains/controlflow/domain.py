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
"""Control flow domain for CFG-based reachability constraints.

The ControlFlowDomain tracks:
- Control flow constructs detected in generated code
- Reachability requirements and violations
- Termination requirements

It builds a CFG sketch incrementally as code is generated
and validates against control flow constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set

import torch

try:
    from ...core.constraint import Satisfiability
    from ...core.domain import ConstraintDomain, GenerationContext
    from ...core.token_classifier import (
        TokenClassifier,
        TokenCategory,
        get_or_create_classifier,
        PYTHON_CONTROL_KEYWORDS,
    )
    from .constraint import (
        CONTROLFLOW_TOP,
        CONTROLFLOW_BOTTOM,
        ControlFlowConstraint,
        CodePoint,
        TerminationRequirement,
    )
    from .cfg import BasicBlock, CFGBuilder, CFGEdge, CFGSketch, EdgeKind
    from .reachability import ReachabilityAnalyzer, ReachabilityResult
except ImportError:
    from core.constraint import Satisfiability
    from core.domain import ConstraintDomain, GenerationContext
    from core.token_classifier import (
        TokenClassifier,
        TokenCategory,
        get_or_create_classifier,
        PYTHON_CONTROL_KEYWORDS,
    )
    from domains.controlflow.constraint import (
        CONTROLFLOW_TOP,
        CONTROLFLOW_BOTTOM,
        ControlFlowConstraint,
        CodePoint,
        TerminationRequirement,
    )
    from domains.controlflow.cfg import BasicBlock, CFGBuilder, CFGEdge, CFGSketch, EdgeKind
    from domains.controlflow.reachability import ReachabilityAnalyzer, ReachabilityResult


@dataclass
class ControlFlowDomainCheckpoint:
    """Checkpoint for ControlFlowDomain state.

    Attributes:
        cfg: Copy of the CFG sketch
        current_block_id: Current block being built
        block_counter: Block counter value
        control_stack: Copy of control structure stack
    """

    cfg: CFGSketch
    current_block_id: Optional[str]
    block_counter: int
    control_stack: List[Dict[str, Any]]


class ControlFlowDomain(ConstraintDomain[ControlFlowConstraint]):
    """Control flow domain for CFG-based constraint tracking.

    The control flow domain:
    1. Detects control flow constructs (if, while, for, return, etc.)
    2. Builds a CFG sketch incrementally
    3. Validates reachability constraints
    4. Provides token masks based on control flow requirements

    Example:
        >>> domain = ControlFlowDomain(language="python")
        >>> constraint = domain.create_constraint(
        ...     must_reach=["function_exit"],
        ...     requires_termination=True,
        ... )
        >>> # As code is generated, constraint is validated against CFG
    """

    def __init__(self, language: str = "python", tokenizer: Optional[Any] = None):
        """Initialize the control flow domain.

        Args:
            language: Programming language (affects construct detection)
            tokenizer: Optional tokenizer for precise masking
        """
        self._language = language
        self._cfg = CFGSketch()
        self._current_block_id: Optional[str] = None
        self._block_counter = 0
        self._control_stack: List[Dict[str, Any]] = []
        self._token_buffer = ""

        # Lazy-initialized classifier
        self._tokenizer = tokenizer
        self._classifier: Optional[TokenClassifier] = None

        # Precomputed keyword token sets (populated on initialization)
        self._return_tokens: FrozenSet[int] = frozenset()
        self._break_tokens: FrozenSet[int] = frozenset()
        self._continue_tokens: FrozenSet[int] = frozenset()

    def _ensure_classifier_initialized(self, context: GenerationContext) -> None:
        """Ensure classifier is initialized and keyword sets are precomputed.

        Args:
            context: Generation context with tokenizer
        """
        tokenizer = context.tokenizer or self._tokenizer
        if tokenizer is None:
            return

        if self._classifier is None:
            self._classifier = get_or_create_classifier(tokenizer, self._language)
            # Precompute keyword token sets
            self._return_tokens = self._classifier.by_keyword("return")
            self._break_tokens = self._classifier.by_keyword("break")
            self._continue_tokens = self._classifier.by_keyword("continue")

    @property
    def name(self) -> str:
        """Return the domain name."""
        return "controlflow"

    @property
    def top(self) -> ControlFlowConstraint:
        """Return the TOP constraint (no restrictions)."""
        return CONTROLFLOW_TOP

    @property
    def bottom(self) -> ControlFlowConstraint:
        """Return the BOTTOM constraint (unsatisfiable)."""
        return CONTROLFLOW_BOTTOM

    @property
    def language(self) -> str:
        """Return the target language."""
        return self._language

    @property
    def cfg(self) -> CFGSketch:
        """Return the current CFG sketch."""
        return self._cfg

    def token_mask(
        self,
        constraint: ControlFlowConstraint,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Compute a token mask based on control flow constraints.

        Implements context-aware keyword blocking:
        1. Blocks break/continue outside of loops
        2. Blocks return if must_not_reach includes function exit
        3. Uses precomputed keyword token sets for O(1) lookups

        Performance target: <20μs typical, <100μs worst case.

        Args:
            constraint: Current control flow constraint
            context: Generation context

        Returns:
            Boolean tensor of valid tokens
        """
        # Handle TOP/BOTTOM
        if constraint.is_top():
            return context.create_mask(fill_value=True)
        if constraint.is_bottom():
            return context.create_mask(fill_value=False)

        # Ensure classifier is initialized
        self._ensure_classifier_initialized(context)

        # Create base mask (all True)
        mask = context.create_mask(fill_value=True)

        # Apply context-aware blocking
        mask = self._apply_context_blocking(mask, constraint, context)

        return mask

    def _apply_context_blocking(
        self,
        mask: torch.Tensor,
        constraint: ControlFlowConstraint,
        context: GenerationContext,
    ) -> torch.Tensor:
        """Apply context-aware keyword blocking.

        Args:
            mask: Current mask to modify
            constraint: Control flow constraint
            context: Generation context

        Returns:
            Modified mask
        """
        # Block break/continue if not in loop context
        if not self._in_loop_context():
            for token_id in self._break_tokens:
                if token_id < context.vocab_size:
                    mask[token_id] = False
            for token_id in self._continue_tokens:
                if token_id < context.vocab_size:
                    mask[token_id] = False

        # Block return if must_not_reach includes function exits
        if self._return_forbidden(constraint):
            for token_id in self._return_tokens:
                if token_id < context.vocab_size:
                    mask[token_id] = False

        return mask

    def _in_loop_context(self) -> bool:
        """Check if currently inside a loop.

        Returns:
            True if in loop context
        """
        for ctx in self._control_stack:
            if ctx.get("type") == "loop":
                return True
        return False

    def _in_function_context(self) -> bool:
        """Check if currently inside a function.

        Returns:
            True if in function context
        """
        for ctx in self._control_stack:
            if ctx.get("type") == "function":
                return True
        return False

    def _return_forbidden(self, constraint: ControlFlowConstraint) -> bool:
        """Check if return is forbidden by the constraint.

        Args:
            constraint: The control flow constraint

        Returns:
            True if return would violate the constraint
        """
        # Check if any must_not_reach point is a return/exit
        for point in constraint.must_not_reach:
            label = point.label.lower()
            if "return" in label or "exit" in label or "function_exit" in label:
                return True

        return False

    def observe_token(
        self,
        constraint: ControlFlowConstraint,
        token_id: int,
        context: GenerationContext,
    ) -> ControlFlowConstraint:
        """Update the control flow constraint after observing a token.

        Detects control flow constructs and updates the CFG.

        Args:
            constraint: Current control flow constraint
            token_id: The generated token
            context: Generation context

        Returns:
            Updated control flow constraint
        """
        if constraint.is_top() or constraint.is_bottom():
            return constraint

        # Get token text
        token_text = ""
        if context.tokenizer is not None:
            try:
                token_text = context.tokenizer.decode([token_id])
            except Exception:
                pass

        # Accumulate token text
        self._token_buffer += token_text

        # Detect control flow constructs
        self._detect_control_flow()

        # Validate constraint against current CFG
        constraint = self._validate_constraint(constraint)

        return constraint

    def _detect_control_flow(self) -> None:
        """Detect control flow constructs in the token buffer."""
        if self._language == "python":
            self._detect_python_control_flow()
        elif self._language == "typescript":
            self._detect_typescript_control_flow()
        elif self._language == "go":
            self._detect_go_control_flow()
        elif self._language == "rust":
            self._detect_rust_control_flow()
        elif self._language == "kotlin":
            self._detect_kotlin_control_flow()
        elif self._language == "swift":
            self._detect_swift_control_flow()
        elif self._language == "zig":
            self._detect_zig_control_flow()
        else:
            self._detect_generic_control_flow()

    def _detect_python_control_flow(self) -> None:
        """Detect Python control flow constructs."""
        # Check for complete statements
        has_newline = "\n" in self._token_buffer

        if not has_newline:
            return

        lines = self._token_buffer.split("\n")
        for line in lines[:-1]:  # Process complete lines
            stripped = line.strip()

            if stripped.startswith("def "):
                self._start_function(stripped)
            elif stripped.startswith("if ") and stripped.endswith(":"):
                self._start_if(stripped)
            elif stripped == "else:" or stripped.startswith("elif "):
                self._continue_if(stripped)
            elif stripped.startswith("while ") and stripped.endswith(":"):
                self._start_loop("while", stripped)
            elif stripped.startswith("for ") and stripped.endswith(":"):
                self._start_loop("for", stripped)
            elif stripped.startswith("return"):
                self._add_return(stripped)
            elif stripped == "break":
                self._add_break()
            elif stripped == "continue":
                self._add_continue()
            elif stripped.startswith("try:"):
                self._start_try()
            elif stripped.startswith("except") or stripped == "finally:":
                self._continue_try(stripped)

        # Keep last incomplete line in buffer
        self._token_buffer = lines[-1]

    def _detect_typescript_control_flow(self) -> None:
        """Detect TypeScript control flow constructs."""
        # Check for complete statements (semicolon or closing brace)
        has_terminator = ";" in self._token_buffer or "}" in self._token_buffer

        if not has_terminator:
            return

        buffer = self._token_buffer
        # Simplified detection - would need proper parsing for full support
        if "function " in buffer:
            self._start_function(buffer)
        elif "if (" in buffer:
            self._start_if(buffer)
        elif "} else {" in buffer or "} else if" in buffer:
            self._continue_if(buffer)
        elif "while (" in buffer:
            self._start_loop("while", buffer)
        elif "for (" in buffer:
            self._start_loop("for", buffer)
        elif "return " in buffer or "return;" in buffer:
            self._add_return(buffer)

        # Clear buffer after processing
        self._token_buffer = ""

    def _detect_generic_control_flow(self) -> None:
        """Generic control flow detection for unsupported languages.

        Detects common control flow patterns that appear across many languages.
        """
        has_terminator = ";" in self._token_buffer or "}" in self._token_buffer or "\n" in self._token_buffer
        if not has_terminator:
            return

        buffer = self._token_buffer

        # Common patterns across languages
        if "function " in buffer or "func " in buffer or "fn " in buffer or "def " in buffer:
            self._start_function(buffer)
        elif "if " in buffer or "if(" in buffer:
            self._start_if(buffer)
        elif "else" in buffer:
            self._continue_if(buffer)
        elif "while " in buffer or "while(" in buffer:
            self._start_loop("while", buffer)
        elif "for " in buffer or "for(" in buffer:
            self._start_loop("for", buffer)
        elif "return" in buffer:
            self._add_return(buffer)

        self._token_buffer = ""

    def _detect_go_control_flow(self) -> None:
        """Detect Go control flow constructs.

        Go uses brace-delimited blocks with optional semicolons.
        Special constructs: defer, go, fallthrough, select, switch/case.
        """
        has_terminator = ";" in self._token_buffer or "}" in self._token_buffer or "\n" in self._token_buffer
        if not has_terminator:
            return

        buffer = self._token_buffer.strip()

        # Function definition: func name(...) or func (receiver) name(...)
        if buffer.startswith("func ") or "func(" in buffer:
            self._start_function(buffer)
        # If statement
        elif buffer.startswith("if ") or "if " in buffer and "{" in buffer:
            self._start_if(buffer)
        # Else/else if
        elif "} else {" in buffer or "} else if" in buffer or buffer.strip() == "} else {":
            self._continue_if(buffer)
        # Switch statement
        elif buffer.startswith("switch ") or buffer.startswith("switch{"):
            self._start_switch(buffer)
        # Case in switch
        elif buffer.startswith("case ") or buffer.strip() == "default:":
            self._add_case(buffer)
        # Fallthrough in switch
        elif "fallthrough" in buffer:
            self._add_fallthrough()
        # Select statement (channel multiplexer)
        elif buffer.startswith("select ") or buffer.startswith("select{"):
            self._start_select(buffer)
        # For loop (Go only has for, no while)
        elif buffer.startswith("for ") or buffer.startswith("for{"):
            self._start_loop("for", buffer)
        # Defer statement
        elif buffer.startswith("defer "):
            self._add_defer(buffer)
        # Goroutine spawn
        elif buffer.startswith("go "):
            self._start_goroutine(buffer)
        # Return statement
        elif buffer.startswith("return") or " return" in buffer:
            self._add_return(buffer)
        # Break statement
        elif buffer.strip() == "break" or buffer.endswith("break"):
            self._add_break()
        # Continue statement
        elif buffer.strip() == "continue" or buffer.endswith("continue"):
            self._add_continue()
        # Panic (non-local exit)
        elif "panic(" in buffer:
            self._add_panic(buffer)

        self._token_buffer = ""

    def _detect_rust_control_flow(self) -> None:
        """Detect Rust control flow constructs.

        Rust uses brace-delimited blocks, expression-based.
        Special constructs: match, loop (infinite), ? operator.
        """
        has_terminator = ";" in self._token_buffer or "}" in self._token_buffer
        if not has_terminator:
            return

        buffer = self._token_buffer.strip()

        # Function definition: fn name(...) or pub fn name(...)
        if buffer.startswith("fn ") or " fn " in buffer or buffer.startswith("pub fn "):
            self._start_function(buffer)
        # Async function
        elif "async fn " in buffer:
            self._start_function(buffer)
        # If expression
        elif buffer.startswith("if ") or " if " in buffer and "{" in buffer:
            self._start_if(buffer)
        # If let pattern
        elif "if let " in buffer:
            self._start_if(buffer)
        # Else/else if
        elif "} else {" in buffer or "} else if" in buffer:
            self._continue_if(buffer)
        # Match expression (pattern matching)
        elif buffer.startswith("match ") or " match " in buffer:
            self._start_match(buffer)
        # Match arm
        elif "=>" in buffer and not buffer.startswith("//"):
            self._add_match_arm(buffer)
        # Infinite loop
        elif buffer.strip() == "loop {" or buffer.startswith("loop {"):
            self._start_loop("loop", buffer)
        # While loop
        elif buffer.startswith("while ") or " while " in buffer:
            self._start_loop("while", buffer)
        # While let pattern
        elif "while let " in buffer:
            self._start_loop("while", buffer)
        # For loop
        elif buffer.startswith("for ") and " in " in buffer:
            self._start_loop("for", buffer)
        # Return statement
        elif buffer.startswith("return") or " return" in buffer:
            self._add_return(buffer)
        # Break with optional value
        elif buffer.strip().startswith("break") or " break" in buffer:
            self._add_break()
        # Continue statement
        elif buffer.strip() == "continue" or " continue" in buffer:
            self._add_continue()
        # Panic macro (non-local exit)
        elif "panic!" in buffer or "unreachable!" in buffer:
            self._add_panic(buffer)

        self._token_buffer = ""

    def _detect_kotlin_control_flow(self) -> None:
        """Detect Kotlin control flow constructs.

        Kotlin uses brace-delimited blocks, expression-oriented.
        Special constructs: when (pattern matching), do-while, labeled returns.
        """
        has_terminator = ";" in self._token_buffer or "}" in self._token_buffer or "\n" in self._token_buffer
        if not has_terminator:
            return

        buffer = self._token_buffer.strip()

        # Function definition: fun name(...) or suspend fun name(...)
        if buffer.startswith("fun ") or " fun " in buffer or "suspend fun " in buffer:
            self._start_function(buffer)
        # If expression
        elif buffer.startswith("if ") or buffer.startswith("if("):
            self._start_if(buffer)
        # Else/else if
        elif "} else {" in buffer or "} else if" in buffer or buffer.strip() == "else {":
            self._continue_if(buffer)
        # When expression (pattern matching)
        elif buffer.startswith("when ") or buffer.startswith("when(") or buffer.startswith("when {"):
            self._start_when(buffer)
        # When branch (like case)
        elif "->" in buffer and self._in_when_context():
            self._add_when_branch(buffer)
        # Do-while loop
        elif buffer.startswith("do {") or buffer.strip() == "do {":
            self._start_do_loop(buffer)
        # While part of do-while
        elif buffer.startswith("} while") and self._in_do_context():
            self._complete_do_loop(buffer)
        # While loop
        elif buffer.startswith("while ") or buffer.startswith("while("):
            self._start_loop("while", buffer)
        # For loop
        elif buffer.startswith("for ") or buffer.startswith("for("):
            self._start_loop("for", buffer)
        # Try-catch
        elif buffer.startswith("try {") or buffer.strip() == "try {":
            self._start_try()
        # Catch block
        elif buffer.startswith("} catch") or buffer.startswith("catch ") or buffer.startswith("catch("):
            self._continue_try(buffer)
        # Finally block
        elif buffer.startswith("} finally") or buffer.strip() == "finally {":
            self._continue_try(buffer)
        # Return (including labeled returns like return@label)
        elif buffer.startswith("return") or " return" in buffer:
            self._add_return(buffer)
        # Break (including labeled breaks)
        elif buffer.strip().startswith("break") or " break" in buffer:
            self._add_break()
        # Continue (including labeled continues)
        elif buffer.strip().startswith("continue") or " continue" in buffer:
            self._add_continue()
        # Throw exception
        elif buffer.startswith("throw ") or " throw " in buffer:
            self._add_throw(buffer)

        self._token_buffer = ""

    def _detect_swift_control_flow(self) -> None:
        """Detect Swift control flow constructs.

        Swift uses brace-delimited blocks.
        Special constructs: guard (early exit), repeat-while, do-catch, defer.
        """
        has_terminator = ";" in self._token_buffer or "}" in self._token_buffer or "\n" in self._token_buffer
        if not has_terminator:
            return

        buffer = self._token_buffer.strip()

        # Function definition: func name(...) or async func name(...)
        if buffer.startswith("func ") or " func " in buffer:
            self._start_function(buffer)
        # If statement
        elif buffer.startswith("if ") or buffer.startswith("if let ") or buffer.startswith("if case "):
            self._start_if(buffer)
        # Else/else if
        elif "} else {" in buffer or "} else if" in buffer or buffer.strip() == "} else {":
            self._continue_if(buffer)
        # Guard statement (early exit)
        elif buffer.startswith("guard "):
            self._start_guard(buffer)
        # Switch statement
        elif buffer.startswith("switch "):
            self._start_switch(buffer)
        # Case in switch
        elif buffer.startswith("case ") or buffer.strip() == "default:":
            self._add_case(buffer)
        # Fallthrough
        elif buffer.strip() == "fallthrough":
            self._add_fallthrough()
        # Repeat-while loop (like do-while)
        elif buffer.startswith("repeat {") or buffer.strip() == "repeat {":
            self._start_repeat_loop(buffer)
        # While part of repeat-while
        elif buffer.startswith("} while") and self._in_repeat_context():
            self._complete_repeat_loop(buffer)
        # While loop
        elif buffer.startswith("while ") or buffer.startswith("while let "):
            self._start_loop("while", buffer)
        # For-in loop
        elif buffer.startswith("for ") and " in " in buffer:
            self._start_loop("for", buffer)
        # Do-catch (error handling)
        elif buffer.startswith("do {") or buffer.strip() == "do {":
            self._start_do_catch(buffer)
        # Catch block
        elif buffer.startswith("} catch") or buffer.startswith("catch "):
            self._add_catch(buffer)
        # Defer block
        elif buffer.startswith("defer {") or buffer.startswith("defer "):
            self._add_defer(buffer)
        # Return statement
        elif buffer.startswith("return") or " return" in buffer:
            self._add_return(buffer)
        # Break statement
        elif buffer.strip() == "break" or buffer.endswith("break"):
            self._add_break()
        # Continue statement
        elif buffer.strip() == "continue" or buffer.endswith("continue"):
            self._add_continue()
        # Throw error
        elif buffer.startswith("throw ") or " throw " in buffer:
            self._add_throw(buffer)
        # fatalError (non-local exit)
        elif "fatalError(" in buffer:
            self._add_panic(buffer)

        self._token_buffer = ""

    def _detect_zig_control_flow(self) -> None:
        """Detect Zig control flow constructs.

        Zig uses brace-delimited blocks, expression-based with explicit error handling.
        Special constructs: inline try-catch, orelse, defer/errdefer, unreachable.
        """
        has_terminator = ";" in self._token_buffer or "}" in self._token_buffer
        if not has_terminator:
            return

        buffer = self._token_buffer.strip()

        # Function definition: fn name(...) or pub fn name(...)
        if buffer.startswith("fn ") or " fn " in buffer or buffer.startswith("pub fn "):
            self._start_function(buffer)
        # Exported function
        elif "export fn " in buffer:
            self._start_function(buffer)
        # If expression
        elif buffer.startswith("if ") or buffer.startswith("if("):
            self._start_if(buffer)
        # Else
        elif "} else {" in buffer or "} else " in buffer:
            self._continue_if(buffer)
        # Switch expression
        elif buffer.startswith("switch ") or buffer.startswith("switch("):
            self._start_switch(buffer)
        # Switch prong (case)
        elif "=>" in buffer and self._in_switch_context():
            self._add_case(buffer)
        # While loop
        elif buffer.startswith("while ") or buffer.startswith("while("):
            self._start_loop("while", buffer)
        # For loop
        elif buffer.startswith("for ") or buffer.startswith("for("):
            self._start_loop("for", buffer)
        # Inline catch (error handling)
        elif " catch " in buffer:
            self._add_catch(buffer)
        # Orelse (null/optional handling)
        elif " orelse " in buffer:
            self._add_orelse(buffer)
        # Defer (cleanup)
        elif buffer.startswith("defer "):
            self._add_defer(buffer)
        # Errdefer (error-path cleanup)
        elif buffer.startswith("errdefer "):
            self._add_errdefer(buffer)
        # Return statement
        elif buffer.startswith("return") or " return" in buffer:
            self._add_return(buffer)
        # Break statement
        elif buffer.strip() == "break" or " break" in buffer:
            self._add_break()
        # Continue statement
        elif buffer.strip() == "continue" or " continue" in buffer:
            self._add_continue()
        # Unreachable
        elif "unreachable" in buffer:
            self._add_unreachable(buffer)
        # @panic builtin
        elif "@panic(" in buffer:
            self._add_panic(buffer)

        self._token_buffer = ""

    # =========================================================================
    # Helper methods for language-specific constructs
    # =========================================================================

    def _in_when_context(self) -> bool:
        """Check if currently inside a when expression (Kotlin)."""
        for ctx in self._control_stack:
            if ctx.get("type") == "when":
                return True
        return False

    def _in_do_context(self) -> bool:
        """Check if currently inside a do block (Kotlin do-while)."""
        for ctx in self._control_stack:
            if ctx.get("type") == "do_loop":
                return True
        return False

    def _in_repeat_context(self) -> bool:
        """Check if currently inside a repeat block (Swift repeat-while)."""
        for ctx in self._control_stack:
            if ctx.get("type") == "repeat_loop":
                return True
        return False

    def _in_switch_context(self) -> bool:
        """Check if currently inside a switch statement."""
        for ctx in self._control_stack:
            if ctx.get("type") in ("switch", "select"):
                return True
        return False

    def _start_switch(self, line: str) -> None:
        """Handle switch/select statement start."""
        switch_id = self._new_block_id("switch_cond")
        switch_block = BasicBlock(id=switch_id, kind="switch_condition")
        self._cfg.add_block(switch_block)

        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=switch_id,
                kind=EdgeKind.SEQUENTIAL,
            ))

        self._control_stack.append({
            "type": "switch",
            "condition": switch_id,
            "cases": [],
            "default": None,
            "exit": None,
        })

        self._current_block_id = switch_id

    def _start_select(self, line: str) -> None:
        """Handle select statement start (Go channel multiplexer)."""
        select_id = self._new_block_id("select")
        select_block = BasicBlock(id=select_id, kind="select")
        self._cfg.add_block(select_block)

        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=select_id,
                kind=EdgeKind.SEQUENTIAL,
            ))

        self._control_stack.append({
            "type": "select",
            "condition": select_id,
            "cases": [],
            "default": None,
        })

        self._current_block_id = select_id

    def _add_case(self, line: str) -> None:
        """Handle case/default in switch or select."""
        if not self._control_stack:
            return

        ctx = self._control_stack[-1]
        if ctx["type"] not in ("switch", "select"):
            return

        is_default = "default" in line.lower()
        case_id = self._new_block_id("default" if is_default else "case")
        case_block = BasicBlock(id=case_id, kind="case")
        self._cfg.add_block(case_block)

        # Connect from switch condition
        self._cfg.add_edge(CFGEdge(
            source=ctx["condition"],
            target=case_id,
            kind=EdgeKind.CONDITIONAL_TRUE if not is_default else EdgeKind.CONDITIONAL_FALSE,
            condition=line,
        ))

        if is_default:
            ctx["default"] = case_id
        else:
            ctx["cases"].append(case_id)

        self._current_block_id = case_id

    def _add_fallthrough(self) -> None:
        """Handle fallthrough in switch (Go, Swift)."""
        if not self._control_stack:
            return

        ctx = self._control_stack[-1]
        if ctx["type"] != "switch":
            return

        # Fallthrough creates edge to next case (will be added when next case is created)
        # For now, just mark that fallthrough occurred
        ctx["has_fallthrough"] = True

    def _start_match(self, line: str) -> None:
        """Handle match expression start (Rust)."""
        match_id = self._new_block_id("match_expr")
        match_block = BasicBlock(id=match_id, kind="match_expression")
        self._cfg.add_block(match_block)

        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=match_id,
                kind=EdgeKind.SEQUENTIAL,
            ))

        self._control_stack.append({
            "type": "match",
            "expression": match_id,
            "arms": [],
            "exhaustive": True,
        })

        self._current_block_id = match_id

    def _add_match_arm(self, line: str) -> None:
        """Handle match arm (Rust pattern => expression)."""
        if not self._control_stack:
            return

        ctx = self._control_stack[-1]
        if ctx["type"] != "match":
            return

        arm_id = self._new_block_id("match_arm")
        arm_block = BasicBlock(id=arm_id, kind="match_arm")
        self._cfg.add_block(arm_block)

        # Connect from match expression
        self._cfg.add_edge(CFGEdge(
            source=ctx["expression"],
            target=arm_id,
            kind=EdgeKind.CONDITIONAL_TRUE,
            condition=line,
        ))

        ctx["arms"].append(arm_id)
        self._current_block_id = arm_id

    def _start_when(self, line: str) -> None:
        """Handle when expression start (Kotlin)."""
        when_id = self._new_block_id("when_expr")
        when_block = BasicBlock(id=when_id, kind="when_expression")
        self._cfg.add_block(when_block)

        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=when_id,
                kind=EdgeKind.SEQUENTIAL,
            ))

        self._control_stack.append({
            "type": "when",
            "expression": when_id,
            "branches": [],
            "else_branch": None,
        })

        self._current_block_id = when_id

    def _add_when_branch(self, line: str) -> None:
        """Handle when branch (Kotlin condition -> result)."""
        if not self._control_stack:
            return

        ctx = self._control_stack[-1]
        if ctx["type"] != "when":
            return

        is_else = "else ->" in line.lower()
        branch_id = self._new_block_id("when_else" if is_else else "when_branch")
        branch_block = BasicBlock(id=branch_id, kind="when_branch")
        self._cfg.add_block(branch_block)

        # Connect from when expression
        self._cfg.add_edge(CFGEdge(
            source=ctx["expression"],
            target=branch_id,
            kind=EdgeKind.CONDITIONAL_TRUE,
            condition=line,
        ))

        if is_else:
            ctx["else_branch"] = branch_id
        else:
            ctx["branches"].append(branch_id)

        self._current_block_id = branch_id

    def _start_do_loop(self, line: str) -> None:
        """Handle do-while loop start (Kotlin)."""
        body_id = self._new_block_id("do_body")
        body_block = BasicBlock(id=body_id, kind="do_body")
        self._cfg.add_block(body_block)

        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=body_id,
                kind=EdgeKind.SEQUENTIAL,
            ))

        self._control_stack.append({
            "type": "do_loop",
            "body": body_id,
            "condition": None,
            "exit": None,
        })

        self._current_block_id = body_id

    def _complete_do_loop(self, line: str) -> None:
        """Complete do-while loop with condition (Kotlin)."""
        if not self._control_stack:
            return

        ctx = self._control_stack[-1]
        if ctx["type"] != "do_loop":
            return

        # Create condition block
        cond_id = self._new_block_id("do_cond")
        cond_block = BasicBlock(id=cond_id, kind="do_condition", is_loop_header=True)
        self._cfg.add_block(cond_block)

        # Connect body to condition
        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=cond_id,
                kind=EdgeKind.SEQUENTIAL,
            ))

        # Connect condition back to body (loop back)
        self._cfg.add_edge(CFGEdge(
            source=cond_id,
            target=ctx["body"],
            kind=EdgeKind.CONDITIONAL_TRUE,
            condition=line,
        ))

        # Create exit block
        exit_id = self._new_block_id("do_exit")
        exit_block = BasicBlock(id=exit_id, kind="loop_exit")
        self._cfg.add_block(exit_block)

        self._cfg.add_edge(CFGEdge(
            source=cond_id,
            target=exit_id,
            kind=EdgeKind.CONDITIONAL_FALSE,
        ))

        ctx["condition"] = cond_id
        ctx["exit"] = exit_id
        self._current_block_id = exit_id

    def _start_repeat_loop(self, line: str) -> None:
        """Handle repeat-while loop start (Swift)."""
        body_id = self._new_block_id("repeat_body")
        body_block = BasicBlock(id=body_id, kind="repeat_body")
        self._cfg.add_block(body_block)

        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=body_id,
                kind=EdgeKind.SEQUENTIAL,
            ))

        self._control_stack.append({
            "type": "repeat_loop",
            "body": body_id,
            "condition": None,
            "exit": None,
        })

        self._current_block_id = body_id

    def _complete_repeat_loop(self, line: str) -> None:
        """Complete repeat-while loop with condition (Swift)."""
        if not self._control_stack:
            return

        ctx = self._control_stack[-1]
        if ctx["type"] != "repeat_loop":
            return

        # Create condition block
        cond_id = self._new_block_id("repeat_cond")
        cond_block = BasicBlock(id=cond_id, kind="repeat_condition", is_loop_header=True)
        self._cfg.add_block(cond_block)

        # Connect body to condition
        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=cond_id,
                kind=EdgeKind.SEQUENTIAL,
            ))

        # Connect condition back to body (loop back)
        self._cfg.add_edge(CFGEdge(
            source=cond_id,
            target=ctx["body"],
            kind=EdgeKind.CONDITIONAL_TRUE,
            condition=line,
        ))

        # Create exit block
        exit_id = self._new_block_id("repeat_exit")
        exit_block = BasicBlock(id=exit_id, kind="loop_exit")
        self._cfg.add_block(exit_block)

        self._cfg.add_edge(CFGEdge(
            source=cond_id,
            target=exit_id,
            kind=EdgeKind.CONDITIONAL_FALSE,
        ))

        ctx["condition"] = cond_id
        ctx["exit"] = exit_id
        self._current_block_id = exit_id

    def _start_guard(self, line: str) -> None:
        """Handle guard statement start (Swift early exit)."""
        guard_id = self._new_block_id("guard_cond")
        guard_block = BasicBlock(id=guard_id, kind="guard_condition")
        self._cfg.add_block(guard_block)

        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=guard_id,
                kind=EdgeKind.SEQUENTIAL,
            ))

        # Guard creates two paths: success continues, failure must exit
        success_id = self._new_block_id("guard_success")
        success_block = BasicBlock(id=success_id, kind="guard_success")
        self._cfg.add_block(success_block)

        self._cfg.add_edge(CFGEdge(
            source=guard_id,
            target=success_id,
            kind=EdgeKind.CONDITIONAL_TRUE,
            condition=line,
        ))

        # The else block will be created when we see the else clause
        self._control_stack.append({
            "type": "guard",
            "condition": guard_id,
            "success": success_id,
            "failure": None,  # Will contain return/throw/break
        })

        self._current_block_id = success_id

    def _start_do_catch(self, line: str) -> None:
        """Handle do-catch block start (Swift error handling)."""
        do_id = self._new_block_id("do_block")
        do_block = BasicBlock(id=do_id, kind="do_block")
        self._cfg.add_block(do_block)

        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=do_id,
                kind=EdgeKind.SEQUENTIAL,
            ))

        self._control_stack.append({
            "type": "do_catch",
            "do_block": do_id,
            "catch_blocks": [],
        })

        self._current_block_id = do_id

    def _add_catch(self, line: str) -> None:
        """Handle catch block (Swift, Zig inline)."""
        # Find the appropriate context
        for ctx in reversed(self._control_stack):
            if ctx["type"] in ("do_catch", "try"):
                catch_id = self._new_block_id("catch")
                catch_block = BasicBlock(id=catch_id, kind="catch_block")
                self._cfg.add_block(catch_block)

                # Connect from do/try block via exception edge
                source_block = ctx.get("do_block") or ctx.get("try_block")
                if source_block:
                    self._cfg.add_edge(CFGEdge(
                        source=source_block,
                        target=catch_id,
                        kind=EdgeKind.EXCEPTION,
                        condition=line,
                    ))

                if "catch_blocks" in ctx:
                    ctx["catch_blocks"].append(catch_id)
                elif "handlers" in ctx:
                    ctx["handlers"].append(catch_id)

                self._current_block_id = catch_id
                break

    def _add_defer(self, line: str) -> None:
        """Handle defer statement (Go, Swift, Zig)."""
        defer_id = self._new_block_id("defer")
        defer_block = BasicBlock(id=defer_id, kind="defer_block")
        self._cfg.add_block(defer_block)

        # Defer blocks execute at scope exit, but we track them
        # The actual edge will be to the exit point
        self._control_stack.append({
            "type": "defer",
            "block": defer_id,
        })

        # Don't change current block - defer doesn't interrupt flow

    def _add_errdefer(self, line: str) -> None:
        """Handle errdefer statement (Zig - runs only on error path)."""
        errdefer_id = self._new_block_id("errdefer")
        errdefer_block = BasicBlock(id=errdefer_id, kind="errdefer_block")
        self._cfg.add_block(errdefer_block)

        self._control_stack.append({
            "type": "errdefer",
            "block": errdefer_id,
        })

        # Don't change current block

    def _add_orelse(self, line: str) -> None:
        """Handle orelse operator (Zig null/optional handling)."""
        # orelse creates a branch: if value is null, use fallback
        orelse_id = self._new_block_id("orelse")
        orelse_block = BasicBlock(id=orelse_id, kind="orelse_fallback")
        self._cfg.add_block(orelse_block)

        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=orelse_id,
                kind=EdgeKind.CONDITIONAL_FALSE,  # null case
                condition=line,
            ))

        # Continue with normal flow (non-null case handled inline)

    def _add_unreachable(self, line: str) -> None:
        """Handle unreachable statement (Zig)."""
        unreachable_id = self._new_block_id("unreachable")
        unreachable_block = BasicBlock(id=unreachable_id, kind="unreachable", is_exit=True)
        self._cfg.add_block(unreachable_block)

        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=unreachable_id,
                kind=EdgeKind.SEQUENTIAL,
            ))

        self._current_block_id = None  # No successor after unreachable

    def _start_goroutine(self, line: str) -> None:
        """Handle goroutine spawn (Go)."""
        goroutine_id = self._new_block_id("goroutine")
        goroutine_block = BasicBlock(id=goroutine_id, kind="goroutine_spawn")
        self._cfg.add_block(goroutine_block)

        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=goroutine_id,
                kind=EdgeKind.SEQUENTIAL,
            ))

        # Goroutine spawns concurrent execution - doesn't block current flow
        # Continue block after spawn
        continue_id = self._new_block_id("after_goroutine")
        continue_block = BasicBlock(id=continue_id, kind="after_goroutine")
        self._cfg.add_block(continue_block)

        self._cfg.add_edge(CFGEdge(
            source=goroutine_id,
            target=continue_id,
            kind=EdgeKind.SEQUENTIAL,
        ))

        self._current_block_id = continue_id

    def _add_throw(self, line: str) -> None:
        """Handle throw statement (Kotlin, Swift)."""
        throw_id = self._new_block_id("throw")
        throw_block = BasicBlock(id=throw_id, kind="throw", is_exit=True)
        self._cfg.add_block(throw_block)

        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=throw_id,
                kind=EdgeKind.EXCEPTION,
            ))

        self._current_block_id = None  # No normal successor after throw

    def _add_panic(self, line: str) -> None:
        """Handle panic/fatalError (Go, Rust, Swift, Zig)."""
        panic_id = self._new_block_id("panic")
        panic_block = BasicBlock(id=panic_id, kind="panic", is_exit=True)
        self._cfg.add_block(panic_block)

        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=panic_id,
                kind=EdgeKind.SEQUENTIAL,
            ))

        self._current_block_id = None  # No successor after panic

    def _new_block_id(self, prefix: str = "block") -> str:
        """Generate a new unique block ID."""
        self._block_counter += 1
        return f"{prefix}_{self._block_counter}"

    def _start_function(self, line: str) -> None:
        """Handle function definition start."""
        # Create entry block
        entry_id = self._new_block_id("func_entry")
        entry = BasicBlock(id=entry_id, kind="function_entry", is_entry=True)
        self._cfg.add_block(entry)
        self._current_block_id = entry_id

        # Push function context onto stack
        self._control_stack.append({
            "type": "function",
            "entry": entry_id,
        })

    def _start_if(self, line: str) -> None:
        """Handle if statement start."""
        # Create blocks for if structure
        cond_id = self._new_block_id("if_cond")
        true_id = self._new_block_id("if_true")

        cond = BasicBlock(id=cond_id, kind="if_condition")
        true_block = BasicBlock(id=true_id, kind="if_true_branch")

        self._cfg.add_block(cond)
        self._cfg.add_block(true_block)

        # Connect from current block
        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=cond_id,
                kind=EdgeKind.SEQUENTIAL,
            ))

        # Connect condition to true branch
        self._cfg.add_edge(CFGEdge(
            source=cond_id,
            target=true_id,
            kind=EdgeKind.CONDITIONAL_TRUE,
            condition=line,
        ))

        # Push if context
        self._control_stack.append({
            "type": "if",
            "condition": cond_id,
            "true_branch": true_id,
            "false_branch": None,
            "join": None,
        })

        self._current_block_id = true_id

    def _continue_if(self, line: str) -> None:
        """Handle else/elif in if statement."""
        if not self._control_stack:
            return

        ctx = self._control_stack[-1]
        if ctx["type"] != "if":
            return

        # Create false branch block
        false_id = self._new_block_id("if_false")
        false_block = BasicBlock(id=false_id, kind="if_false_branch")
        self._cfg.add_block(false_block)

        # Connect condition to false branch
        self._cfg.add_edge(CFGEdge(
            source=ctx["condition"],
            target=false_id,
            kind=EdgeKind.CONDITIONAL_FALSE,
        ))

        ctx["false_branch"] = false_id
        self._current_block_id = false_id

    def _start_loop(self, loop_type: str, line: str) -> None:
        """Handle loop start."""
        # Create blocks for loop structure
        header_id = self._new_block_id(f"{loop_type}_header")
        body_id = self._new_block_id(f"{loop_type}_body")

        header = BasicBlock(id=header_id, kind=f"{loop_type}_header", is_loop_header=True)
        body = BasicBlock(id=body_id, kind=f"{loop_type}_body")

        self._cfg.add_block(header)
        self._cfg.add_block(body)

        # Connect from current block to header
        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=header_id,
                kind=EdgeKind.SEQUENTIAL,
            ))

        # Connect header to body
        self._cfg.add_edge(CFGEdge(
            source=header_id,
            target=body_id,
            kind=EdgeKind.CONDITIONAL_TRUE,
            condition=line,
        ))

        # Push loop context
        self._control_stack.append({
            "type": "loop",
            "loop_type": loop_type,
            "header": header_id,
            "body": body_id,
            "exit": None,
        })

        self._current_block_id = body_id

    def _add_return(self, line: str) -> None:
        """Handle return statement."""
        # Create exit block
        exit_id = self._new_block_id("return")
        exit_block = BasicBlock(id=exit_id, kind="return", is_exit=True)
        self._cfg.add_block(exit_block)

        # Connect from current block
        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=exit_id,
                kind=EdgeKind.RETURN,
            ))

        self._current_block_id = None  # No successor after return

    def _add_break(self) -> None:
        """Handle break statement."""
        # Find enclosing loop
        for ctx in reversed(self._control_stack):
            if ctx["type"] == "loop":
                # Create break target if not exists
                if ctx.get("exit") is None:
                    exit_id = self._new_block_id("loop_exit")
                    exit_block = BasicBlock(id=exit_id, kind="loop_exit")
                    self._cfg.add_block(exit_block)
                    ctx["exit"] = exit_id

                # Connect from current to exit
                if self._current_block_id:
                    self._cfg.add_edge(CFGEdge(
                        source=self._current_block_id,
                        target=ctx["exit"],
                        kind=EdgeKind.BREAK,
                    ))
                break

    def _add_continue(self) -> None:
        """Handle continue statement."""
        # Find enclosing loop
        for ctx in reversed(self._control_stack):
            if ctx["type"] == "loop":
                # Connect from current to header
                if self._current_block_id:
                    self._cfg.add_edge(CFGEdge(
                        source=self._current_block_id,
                        target=ctx["header"],
                        kind=EdgeKind.CONTINUE,
                    ))
                break

    def _start_try(self) -> None:
        """Handle try statement start."""
        try_id = self._new_block_id("try")
        try_block = BasicBlock(id=try_id, kind="try_block")
        self._cfg.add_block(try_block)

        if self._current_block_id:
            self._cfg.add_edge(CFGEdge(
                source=self._current_block_id,
                target=try_id,
                kind=EdgeKind.SEQUENTIAL,
            ))

        self._control_stack.append({
            "type": "try",
            "try_block": try_id,
            "handlers": [],
        })

        self._current_block_id = try_id

    def _continue_try(self, line: str) -> None:
        """Handle except/finally in try statement."""
        if not self._control_stack:
            return

        ctx = self._control_stack[-1]
        if ctx["type"] != "try":
            return

        handler_id = self._new_block_id("except" if "except" in line else "finally")
        handler = BasicBlock(id=handler_id, kind="exception_handler")
        self._cfg.add_block(handler)

        # Connect from try block via exception edge
        self._cfg.add_edge(CFGEdge(
            source=ctx["try_block"],
            target=handler_id,
            kind=EdgeKind.EXCEPTION,
        ))

        ctx["handlers"].append(handler_id)
        self._current_block_id = handler_id

    def _validate_constraint(
        self,
        constraint: ControlFlowConstraint,
    ) -> ControlFlowConstraint:
        """Validate constraint against current CFG.

        Args:
            constraint: Current constraint

        Returns:
            Updated constraint (BOTTOM if violated)
        """
        if constraint.is_top() or constraint.is_bottom():
            return constraint

        if self._cfg.block_count() == 0:
            return constraint

        # Run reachability analysis
        analyzer = ReachabilityAnalyzer(self._cfg)
        result = analyzer.analyze()

        # Check must-reach constraints
        for point in constraint.must_reach:
            if point.label in self._cfg.blocks:
                if not analyzer.is_reachable(point.label):
                    return CONTROLFLOW_BOTTOM

        # Check must-not-reach constraints
        for point in constraint.must_not_reach:
            if point.label in self._cfg.blocks:
                if analyzer.is_reachable(point.label):
                    return CONTROLFLOW_BOTTOM

        # Check termination requirement
        if constraint.requires_termination():
            if result.has_infinite_loop:
                return CONTROLFLOW_BOTTOM

        return constraint

    def analyze_reachability(self) -> ReachabilityResult:
        """Analyze reachability of the current CFG.

        Returns:
            ReachabilityResult with analysis
        """
        analyzer = ReachabilityAnalyzer(self._cfg)
        return analyzer.analyze()

    def checkpoint(self) -> ControlFlowDomainCheckpoint:
        """Create a checkpoint of the current state.

        Returns:
            Checkpoint for restoration
        """
        # Deep copy the CFG
        cfg_copy = CFGSketch(
            blocks={k: BasicBlock(
                id=v.id,
                kind=v.kind,
                statements=v.statements.copy(),
                predecessors=v.predecessors.copy(),
                successors=v.successors.copy(),
                is_entry=v.is_entry,
                is_exit=v.is_exit,
                is_loop_header=v.is_loop_header,
                source_lines=v.source_lines,
            ) for k, v in self._cfg.blocks.items()},
            edges=self._cfg.edges.copy(),
            entry_id=self._cfg.entry_id,
            exit_ids=self._cfg.exit_ids.copy(),
        )

        return ControlFlowDomainCheckpoint(
            cfg=cfg_copy,
            current_block_id=self._current_block_id,
            block_counter=self._block_counter,
            control_stack=[ctx.copy() for ctx in self._control_stack],
        )

    def restore(self, checkpoint: Any) -> None:
        """Restore state from a checkpoint.

        Args:
            checkpoint: Previously created checkpoint
        """
        if not isinstance(checkpoint, ControlFlowDomainCheckpoint):
            raise TypeError(
                f"Expected ControlFlowDomainCheckpoint, got {type(checkpoint).__name__}"
            )

        self._cfg = checkpoint.cfg
        self._current_block_id = checkpoint.current_block_id
        self._block_counter = checkpoint.block_counter
        self._control_stack = [ctx.copy() for ctx in checkpoint.control_stack]

    def satisfiability(self, constraint: ControlFlowConstraint) -> Satisfiability:
        """Check satisfiability of a control flow constraint.

        Args:
            constraint: The constraint to check

        Returns:
            Satisfiability status
        """
        return constraint.satisfiability()

    def create_constraint(
        self,
        must_reach: Optional[List[str]] = None,
        must_not_reach: Optional[List[str]] = None,
        requires_termination: bool = False,
    ) -> ControlFlowConstraint:
        """Create a control flow constraint.

        Args:
            must_reach: List of labels that must be reachable
            must_not_reach: List of labels that must be unreachable
            requires_termination: Whether termination is required

        Returns:
            New ControlFlowConstraint
        """
        if must_reach is None and must_not_reach is None and not requires_termination:
            return CONTROLFLOW_TOP

        reach_points = frozenset(CodePoint(label=lbl) for lbl in (must_reach or []))
        not_reach_points = frozenset(CodePoint(label=lbl) for lbl in (must_not_reach or []))

        # Check for conflict
        reach_labels = {p.label for p in reach_points}
        not_reach_labels = {p.label for p in not_reach_points}
        if reach_labels & not_reach_labels:
            return CONTROLFLOW_BOTTOM

        termination = (
            TerminationRequirement.MUST_TERMINATE
            if requires_termination
            else TerminationRequirement.UNKNOWN
        )

        return ControlFlowConstraint(
            must_reach=reach_points,
            must_not_reach=not_reach_points,
            termination=termination,
        )

    def set_function_context(
        self,
        function_name: str,
        expected_return_type: Optional[str] = None,
        is_async: bool = False,
        is_generator: bool = False,
    ) -> None:
        """Set the function context for control flow tracking.

        Args:
            function_name: Name of the current function
            expected_return_type: Expected return type expression
            is_async: Whether the function is async
            is_generator: Whether the function is a generator
        """
        # Create entry block for function
        entry_id = self._new_block_id("func_entry")
        entry = BasicBlock(id=entry_id, kind="function_entry", is_entry=True)
        self._cfg.add_block(entry)
        self._current_block_id = entry_id

        # Push function context
        self._control_stack.append({
            "type": "function",
            "name": function_name,
            "entry": entry_id,
            "is_async": is_async,
            "is_generator": is_generator,
            "expected_return_type": expected_return_type,
        })

    def set_loop_context(
        self,
        loop_depth: int = 0,
        loop_variables: Optional[List[str]] = None,
    ) -> None:
        """Set the loop context for control flow tracking.

        Args:
            loop_depth: Current nesting depth of loops
            loop_variables: Variables used as loop iterators
        """
        # Create nested loop contexts based on depth
        for i in range(loop_depth):
            header_id = self._new_block_id(f"loop_header_{i}")
            body_id = self._new_block_id(f"loop_body_{i}")

            header = BasicBlock(id=header_id, kind="loop_header", is_loop_header=True)
            body = BasicBlock(id=body_id, kind="loop_body")

            self._cfg.add_block(header)
            self._cfg.add_block(body)

            if self._current_block_id:
                self._cfg.add_edge(CFGEdge(
                    source=self._current_block_id,
                    target=header_id,
                    kind=EdgeKind.SEQUENTIAL,
                ))

            self._cfg.add_edge(CFGEdge(
                source=header_id,
                target=body_id,
                kind=EdgeKind.CONDITIONAL_TRUE,
            ))

            self._control_stack.append({
                "type": "loop",
                "loop_type": "for",
                "header": header_id,
                "body": body_id,
                "exit": None,
            })

            self._current_block_id = body_id

    def set_try_context(self, in_try_block: bool = False, exception_types: Optional[List[str]] = None) -> None:
        """Set the try/except context for control flow tracking.

        Args:
            in_try_block: Whether we're inside a try block
            exception_types: Types of exceptions being caught
        """
        if in_try_block:
            try_id = self._new_block_id("try")
            try_block = BasicBlock(id=try_id, kind="try_block")
            self._cfg.add_block(try_block)

            if self._current_block_id:
                self._cfg.add_edge(CFGEdge(
                    source=self._current_block_id,
                    target=try_id,
                    kind=EdgeKind.SEQUENTIAL,
                ))

            self._control_stack.append({
                "type": "try",
                "try_block": try_id,
                "handlers": [],
                "exception_types": exception_types or [],
            })

            self._current_block_id = try_id

    def inject_context(self, spec: Any) -> None:
        """Inject context from a ConstraintSpec.

        Called when a cached grammar object needs fresh context.
        This re-seeds the control flow state with data from the spec.

        Args:
            spec: A ConstraintSpec object (typed as Any to avoid circular import)
        """
        # Import locally to avoid circular dependency
        # Try relative import first, fall back to absolute
        try:
            from ...spec.constraint_spec import ConstraintSpec
        except ImportError:
            try:
                from spec.constraint_spec import ConstraintSpec
            except ImportError:
                # If we can't import, check by class name
                if spec.__class__.__name__ != "ConstraintSpec":
                    return
                ConstraintSpec = spec.__class__

        if not isinstance(spec, ConstraintSpec):
            return

        # Clear existing state
        self._cfg = CFGSketch()
        self._current_block_id = None
        self._block_counter = 0
        self._control_stack.clear()
        self._token_buffer = ""

        # Apply control flow context if provided
        cf_ctx = spec.control_flow
        if cf_ctx is None:
            return

        # Set function context if provided
        if cf_ctx.function_name:
            self.set_function_context(
                function_name=cf_ctx.function_name,
                expected_return_type=cf_ctx.expected_return_type,
                is_async=cf_ctx.in_async_context,
                is_generator=cf_ctx.in_generator,
            )

        # Set loop context if in loop
        if cf_ctx.loop_depth > 0:
            self.set_loop_context(
                loop_depth=cf_ctx.loop_depth,
                loop_variables=list(cf_ctx.loop_variables),
            )

        # Set try context if in try block
        if cf_ctx.in_try_block:
            self.set_try_context(
                in_try_block=True,
                exception_types=list(cf_ctx.exception_types),
            )

    def get_reachability_status(self) -> bool:
        """Get whether current position is reachable.

        Returns:
            True if current code position is reachable
        """
        if self._current_block_id is None:
            return False

        analyzer = ReachabilityAnalyzer(self._cfg)
        result = analyzer.analyze()
        return analyzer.is_reachable(self._current_block_id)
