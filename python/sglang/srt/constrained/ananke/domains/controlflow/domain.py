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
from typing import Any, Dict, List, Optional, Set

import torch

try:
    from ...core.constraint import Satisfiability
    from ...core.domain import ConstraintDomain, GenerationContext
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

    def __init__(self, language: str = "python"):
        """Initialize the control flow domain.

        Args:
            language: Programming language (affects construct detection)
        """
        self._language = language
        self._cfg = CFGSketch()
        self._current_block_id: Optional[str] = None
        self._block_counter = 0
        self._control_stack: List[Dict[str, Any]] = []
        self._token_buffer = ""

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

        Current implementation is conservative (allows all tokens).
        A full implementation would:
        - Block tokens that would make must-reach points unreachable
        - Block tokens that would create paths to must-not-reach points

        Args:
            constraint: Current control flow constraint
            context: Generation context

        Returns:
            Boolean tensor of valid tokens
        """
        # Handle TOP/BOTTOM
        if constraint.is_top():
            return torch.ones(context.vocab_size, dtype=torch.bool, device=context.device)
        if constraint.is_bottom():
            return torch.zeros(context.vocab_size, dtype=torch.bool, device=context.device)

        # Conservative: allow all tokens
        # Full implementation would analyze control flow implications
        return torch.ones(context.vocab_size, dtype=torch.bool, device=context.device)

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
        # Other languages can be added

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
