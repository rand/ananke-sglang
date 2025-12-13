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
"""Tests for the Control Flow Domain.

Tests cover:
- ControlFlowConstraint semilattice laws
- CodePoint creation and properties
- CFGSketch building and querying
- Reachability analysis
- ControlFlowDomain functionality
"""

from __future__ import annotations

import pytest
import torch
from typing import Dict, List, Optional

try:
    from ...domains.controlflow import (
        ControlFlowConstraint,
        CodePoint,
        ReachabilityKind,
        TerminationRequirement,
        CONTROLFLOW_TOP,
        CONTROLFLOW_BOTTOM,
        controlflow_requiring_reach,
        controlflow_forbidding_reach,
        controlflow_requiring_termination,
        BasicBlock,
        CFGEdge,
        CFGSketch,
        CFGBuilder,
        EdgeKind,
        ControlFlowDomain,
        ControlFlowDomainCheckpoint,
        ReachabilityAnalyzer,
        ReachabilityResult,
        compute_dominators,
        compute_post_dominators,
        find_loop_bodies,
    )
    from ...core.constraint import Satisfiability
    from ...core.domain import GenerationContext
except ImportError:
    from domains.controlflow import (
        ControlFlowConstraint,
        CodePoint,
        ReachabilityKind,
        TerminationRequirement,
        CONTROLFLOW_TOP,
        CONTROLFLOW_BOTTOM,
        controlflow_requiring_reach,
        controlflow_forbidding_reach,
        controlflow_requiring_termination,
        BasicBlock,
        CFGEdge,
        CFGSketch,
        CFGBuilder,
        EdgeKind,
        ControlFlowDomain,
        ControlFlowDomainCheckpoint,
        ReachabilityAnalyzer,
        ReachabilityResult,
        compute_dominators,
        compute_post_dominators,
        find_loop_bodies,
    )
    from core.constraint import Satisfiability
    from core.domain import GenerationContext


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def generation_context() -> GenerationContext:
    """Create a generation context for testing."""
    return GenerationContext(
        vocab_size=100,
        device=torch.device("cpu"),
        tokenizer=None,
        generated_text="",
        position=0,
    )


@pytest.fixture
def python_domain() -> ControlFlowDomain:
    """Create a Python control flow domain."""
    return ControlFlowDomain(language="python")


@pytest.fixture
def simple_cfg() -> CFGSketch:
    """Create a simple linear CFG: entry -> body -> exit."""
    return (
        CFGBuilder()
        .entry("entry")
        .block("body")
        .exit("exit")
        .edge("entry", "body")
        .edge("body", "exit")
        .build()
    )


@pytest.fixture
def branching_cfg() -> CFGSketch:
    """Create a branching CFG with if-else."""
    return (
        CFGBuilder()
        .entry("entry")
        .block("condition")
        .block("then_branch")
        .block("else_branch")
        .block("join")
        .exit("exit")
        .edge("entry", "condition")
        .edge("condition", "then_branch", EdgeKind.CONDITIONAL_TRUE, "x > 0")
        .edge("condition", "else_branch", EdgeKind.CONDITIONAL_FALSE, "x <= 0")
        .edge("then_branch", "join")
        .edge("else_branch", "join")
        .edge("join", "exit")
        .build()
    )


@pytest.fixture
def loop_cfg() -> CFGSketch:
    """Create a CFG with a loop."""
    return (
        CFGBuilder()
        .entry("entry")
        .block("header", is_loop_header=True)
        .block("body")
        .exit("exit")
        .edge("entry", "header")
        .edge("header", "body", EdgeKind.CONDITIONAL_TRUE, "i < 10")
        .edge("header", "exit", EdgeKind.LOOP_EXIT, "i >= 10")
        .edge("body", "header", EdgeKind.LOOP_BACK)
        .build()
    )


# =============================================================================
# CodePoint Tests
# =============================================================================


class TestCodePoint:
    """Tests for CodePoint dataclass."""

    def test_create_simple(self) -> None:
        """Test creating simple CodePoint."""
        point = CodePoint(label="entry")
        assert point.label == "entry"
        assert point.kind is None
        assert point.line is None

    def test_create_full(self) -> None:
        """Test creating CodePoint with all fields."""
        point = CodePoint(label="loop_header", kind="loop", line=42)
        assert point.label == "loop_header"
        assert point.kind == "loop"
        assert point.line == 42

    def test_equality(self) -> None:
        """Test CodePoint equality."""
        p1 = CodePoint(label="entry", kind="start")
        p2 = CodePoint(label="entry", kind="start")
        p3 = CodePoint(label="exit", kind="end")

        assert p1 == p2
        assert p1 != p3

    def test_hashable(self) -> None:
        """Test CodePoint is hashable."""
        point = CodePoint(label="test")
        s = {point}
        assert point in s

    def test_repr(self) -> None:
        """Test CodePoint string representation."""
        point = CodePoint(label="block1", line=10)
        assert "block1" in repr(point)
        assert "L10" in repr(point)


# =============================================================================
# ControlFlowConstraint Tests
# =============================================================================


class TestControlFlowConstraint:
    """Tests for ControlFlowConstraint."""

    def test_top_is_top(self) -> None:
        """Test TOP constraint is_top."""
        assert CONTROLFLOW_TOP.is_top()
        assert not CONTROLFLOW_TOP.is_bottom()

    def test_bottom_is_bottom(self) -> None:
        """Test BOTTOM constraint is_bottom."""
        assert CONTROLFLOW_BOTTOM.is_bottom()
        assert not CONTROLFLOW_BOTTOM.is_top()

    def test_empty_constraint_satisfiable(self) -> None:
        """Test empty constraint is satisfiable."""
        c = ControlFlowConstraint()
        assert c.satisfiability() == Satisfiability.SAT
        assert not c.is_top()
        assert not c.is_bottom()

    def test_require_reach(self) -> None:
        """Test creating constraint with must-reach."""
        c = controlflow_requiring_reach("entry", "exit")
        assert c.is_must_reach(CodePoint(label="entry"))
        assert c.is_must_reach(CodePoint(label="exit"))
        assert not c.is_must_reach(CodePoint(label="other"))

    def test_forbid_reach(self) -> None:
        """Test creating constraint with must-not-reach."""
        c = controlflow_forbidding_reach("dead_code")
        assert c.is_must_not_reach(CodePoint(label="dead_code"))
        assert not c.is_must_not_reach(CodePoint(label="live"))

    def test_require_termination(self) -> None:
        """Test creating constraint requiring termination."""
        c = controlflow_requiring_termination()
        assert c.requires_termination()

    def test_conflict_is_bottom(self) -> None:
        """Test that same point must-reach and must-not-reach = BOTTOM."""
        c1 = controlflow_requiring_reach("point")
        c2 = controlflow_forbidding_reach("point")
        result = c1.meet(c2)
        assert result.is_bottom()

    def test_meet_with_top(self) -> None:
        """Test meet with TOP returns other constraint."""
        c = controlflow_requiring_reach("entry")
        assert c.meet(CONTROLFLOW_TOP) == c
        assert CONTROLFLOW_TOP.meet(c) == c

    def test_meet_with_bottom(self) -> None:
        """Test meet with BOTTOM returns BOTTOM."""
        c = controlflow_requiring_reach("entry")
        assert c.meet(CONTROLFLOW_BOTTOM).is_bottom()
        assert CONTROLFLOW_BOTTOM.meet(c).is_bottom()

    def test_meet_combines_must_reach(self) -> None:
        """Test meet combines must_reach sets."""
        c1 = controlflow_requiring_reach("entry")
        c2 = controlflow_requiring_reach("exit")
        result = c1.meet(c2)

        assert result.is_must_reach(CodePoint(label="entry"))
        assert result.is_must_reach(CodePoint(label="exit"))

    def test_meet_combines_must_not_reach(self) -> None:
        """Test meet combines must_not_reach sets."""
        c1 = controlflow_forbidding_reach("dead1")
        c2 = controlflow_forbidding_reach("dead2")
        result = c1.meet(c2)

        assert result.is_must_not_reach(CodePoint(label="dead1"))
        assert result.is_must_not_reach(CodePoint(label="dead2"))


class TestControlFlowConstraintSemilattice:
    """Verify semilattice laws for ControlFlowConstraint."""

    def test_idempotent(self) -> None:
        """Test c.meet(c) == c."""
        c = controlflow_requiring_reach("entry").meet(controlflow_forbidding_reach("dead"))
        assert c.meet(c) == c

    def test_commutative(self) -> None:
        """Test c1.meet(c2) == c2.meet(c1)."""
        c1 = controlflow_requiring_reach("entry")
        c2 = controlflow_forbidding_reach("dead")

        r1 = c1.meet(c2)
        r2 = c2.meet(c1)

        assert r1.must_reach == r2.must_reach
        assert r1.must_not_reach == r2.must_not_reach

    def test_associative(self) -> None:
        """Test (c1.meet(c2)).meet(c3) == c1.meet(c2.meet(c3))."""
        c1 = controlflow_requiring_reach("entry")
        c2 = controlflow_requiring_reach("exit")
        c3 = controlflow_forbidding_reach("dead")

        left = (c1.meet(c2)).meet(c3)
        right = c1.meet(c2.meet(c3))

        assert left.must_reach == right.must_reach
        assert left.must_not_reach == right.must_not_reach

    def test_top_identity(self) -> None:
        """Test c.meet(TOP) == c."""
        c = controlflow_requiring_reach("entry")
        assert c.meet(CONTROLFLOW_TOP) == c

    def test_bottom_absorbing(self) -> None:
        """Test c.meet(BOTTOM) == BOTTOM."""
        c = controlflow_requiring_reach("entry")
        assert c.meet(CONTROLFLOW_BOTTOM).is_bottom()


# =============================================================================
# CFG Tests
# =============================================================================


class TestBasicBlock:
    """Tests for BasicBlock."""

    def test_create_simple(self) -> None:
        """Test creating simple BasicBlock."""
        block = BasicBlock(id="b1")
        assert block.id == "b1"
        assert block.kind == "block"
        assert not block.is_entry
        assert not block.is_exit

    def test_create_entry(self) -> None:
        """Test creating entry block."""
        block = BasicBlock(id="entry", kind="entry", is_entry=True)
        assert block.is_entry
        assert not block.is_exit

    def test_create_exit(self) -> None:
        """Test creating exit block."""
        block = BasicBlock(id="exit", kind="exit", is_exit=True)
        assert block.is_exit
        assert not block.is_entry

    def test_to_code_point(self) -> None:
        """Test converting to CodePoint."""
        block = BasicBlock(id="loop", kind="loop_header", source_lines=(10, 20))
        point = block.to_code_point()

        assert point.label == "loop"
        assert point.kind == "loop_header"
        assert point.line == 10

    def test_add_statement(self) -> None:
        """Test adding statements."""
        block = BasicBlock(id="b1")
        block.add_statement("x = 1")
        block.add_statement("y = 2")

        assert len(block.statements) == 2
        assert "x = 1" in block.statements


class TestCFGEdge:
    """Tests for CFGEdge."""

    def test_create_simple(self) -> None:
        """Test creating simple edge."""
        edge = CFGEdge(source="a", target="b")
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.kind == EdgeKind.SEQUENTIAL

    def test_create_conditional(self) -> None:
        """Test creating conditional edge."""
        edge = CFGEdge(
            source="cond",
            target="then",
            kind=EdgeKind.CONDITIONAL_TRUE,
            condition="x > 0",
        )
        assert edge.kind == EdgeKind.CONDITIONAL_TRUE
        assert edge.condition == "x > 0"

    def test_repr(self) -> None:
        """Test string representation."""
        edge = CFGEdge(source="a", target="b", kind=EdgeKind.LOOP_BACK)
        assert "a" in repr(edge)
        assert "b" in repr(edge)
        assert "LOOP_BACK" in repr(edge)


class TestCFGSketch:
    """Tests for CFGSketch."""

    def test_simple_cfg(self, simple_cfg: CFGSketch) -> None:
        """Test simple linear CFG."""
        assert simple_cfg.block_count() == 3
        assert simple_cfg.edge_count() == 2
        assert simple_cfg.entry_id == "entry"
        assert "exit" in simple_cfg.exit_ids

    def test_get_block(self, simple_cfg: CFGSketch) -> None:
        """Test getting block by ID."""
        block = simple_cfg.get_block("entry")
        assert block is not None
        assert block.id == "entry"
        assert block.is_entry

    def test_get_entry(self, simple_cfg: CFGSketch) -> None:
        """Test getting entry block."""
        entry = simple_cfg.get_entry()
        assert entry is not None
        assert entry.is_entry

    def test_get_exits(self, simple_cfg: CFGSketch) -> None:
        """Test getting exit blocks."""
        exits = simple_cfg.get_exits()
        assert len(exits) == 1
        assert exits[0].is_exit

    def test_successors(self, simple_cfg: CFGSketch) -> None:
        """Test getting successor blocks."""
        succs = simple_cfg.get_successors("entry")
        assert len(succs) == 1
        assert succs[0].id == "body"

    def test_predecessors(self, simple_cfg: CFGSketch) -> None:
        """Test getting predecessor blocks."""
        preds = simple_cfg.get_predecessors("body")
        assert len(preds) == 1
        assert preds[0].id == "entry"

    def test_outgoing_edges(self, branching_cfg: CFGSketch) -> None:
        """Test getting outgoing edges."""
        edges = branching_cfg.get_outgoing_edges("condition")
        assert len(edges) == 2
        kinds = {e.kind for e in edges}
        assert EdgeKind.CONDITIONAL_TRUE in kinds
        assert EdgeKind.CONDITIONAL_FALSE in kinds

    def test_loop_headers(self, loop_cfg: CFGSketch) -> None:
        """Test getting loop headers."""
        headers = loop_cfg.get_loop_headers()
        assert len(headers) == 1
        assert headers[0].id == "header"

    def test_back_edges(self, loop_cfg: CFGSketch) -> None:
        """Test getting back edges."""
        back_edges = loop_cfg.get_back_edges()
        assert len(back_edges) == 1
        assert back_edges[0].source == "body"
        assert back_edges[0].target == "header"


class TestCFGBuilder:
    """Tests for CFGBuilder."""

    def test_fluent_interface(self) -> None:
        """Test fluent builder interface."""
        cfg = (
            CFGBuilder()
            .entry("start")
            .block("middle")
            .exit("end")
            .edge("start", "middle")
            .edge("middle", "end")
            .build()
        )

        assert cfg.block_count() == 3
        assert cfg.entry_id == "start"

    def test_conditional_helper(self) -> None:
        """Test conditional() helper method."""
        cfg = (
            CFGBuilder()
            .entry("entry")
            .block("cond")
            .block("then")
            .block("else")
            .edge("entry", "cond")
            .conditional("cond", "then", "else", "x > 0")
            .build()
        )

        edges = cfg.get_outgoing_edges("cond")
        assert len(edges) == 2

    def test_loop_helper(self) -> None:
        """Test loop() helper method."""
        cfg = (
            CFGBuilder()
            .entry("entry")
            .block("header", is_loop_header=True)
            .block("body")
            .block("after")
            .edge("entry", "header")
            .loop("header", "body", "after", "i < 10")
            .build()
        )

        # Should have header -> body (true), header -> after (exit), body -> header (back)
        header_edges = cfg.get_outgoing_edges("header")
        assert len(header_edges) == 2

        body_edges = cfg.get_outgoing_edges("body")
        assert len(body_edges) == 1
        assert body_edges[0].kind == EdgeKind.LOOP_BACK


# =============================================================================
# Reachability Tests
# =============================================================================


class TestReachabilityAnalyzer:
    """Tests for ReachabilityAnalyzer."""

    def test_simple_reachability(self, simple_cfg: CFGSketch) -> None:
        """Test reachability on simple CFG."""
        analyzer = ReachabilityAnalyzer(simple_cfg)
        result = analyzer.analyze()

        assert "entry" in result.reachable_from_entry
        assert "body" in result.reachable_from_entry
        assert "exit" in result.reachable_from_entry
        assert not result.dead_blocks
        assert result.all_paths_return

    def test_branching_reachability(self, branching_cfg: CFGSketch) -> None:
        """Test reachability on branching CFG."""
        analyzer = ReachabilityAnalyzer(branching_cfg)
        result = analyzer.analyze()

        assert "then_branch" in result.reachable_from_entry
        assert "else_branch" in result.reachable_from_entry
        assert result.all_paths_return

    def test_dead_code_detection(self) -> None:
        """Test detecting dead code."""
        cfg = (
            CFGBuilder()
            .entry("entry")
            .block("live")
            .block("dead")  # Not connected
            .exit("exit")
            .edge("entry", "live")
            .edge("live", "exit")
            .build()
        )

        analyzer = ReachabilityAnalyzer(cfg)
        result = analyzer.analyze()

        assert "dead" in result.dead_blocks
        assert analyzer.is_dead("dead")
        assert not analyzer.is_dead("live")

    def test_blocked_path_detection(self) -> None:
        """Test detecting blocked paths."""
        cfg = (
            CFGBuilder()
            .entry("entry")
            .block("blocked")  # Reachable but can't reach exit
            .exit("exit")
            .edge("entry", "blocked")
            .edge("entry", "exit")
            # blocked has no outgoing edges
            .build()
        )

        analyzer = ReachabilityAnalyzer(cfg)
        result = analyzer.analyze()

        assert not result.all_paths_return
        assert analyzer.is_path_blocked("blocked")

    def test_infinite_loop_detection(self, loop_cfg: CFGSketch) -> None:
        """Test detecting potential infinite loops."""
        # Normal loop has exit, so no infinite loop
        analyzer = ReachabilityAnalyzer(loop_cfg)
        result = analyzer.analyze()
        assert not result.has_infinite_loop

    def test_infinite_loop_present(self) -> None:
        """Test detecting actual infinite loop."""
        cfg = (
            CFGBuilder()
            .entry("entry")
            .block("loop1", is_loop_header=True)
            .block("loop2")
            .exit("exit")  # Entry block but unreachable from loop
            .edge("entry", "loop1")
            # entry -> exit is NOT connected
            .edge("loop1", "loop2")
            .edge("loop2", "loop1", EdgeKind.LOOP_BACK)
            .build()
        )

        analyzer = ReachabilityAnalyzer(cfg)
        result = analyzer.analyze()

        # Loop1 and loop2 are reachable but can't reach exit
        assert result.has_infinite_loop


class TestDominators:
    """Tests for dominator computation."""

    def test_simple_dominators(self, simple_cfg: CFGSketch) -> None:
        """Test dominators on simple CFG."""
        doms = compute_dominators(simple_cfg)

        # Entry dominates itself
        assert "entry" in doms["entry"]
        # Entry dominates all
        assert "entry" in doms["body"]
        assert "entry" in doms["exit"]

    def test_branching_dominators(self, branching_cfg: CFGSketch) -> None:
        """Test dominators on branching CFG."""
        doms = compute_dominators(branching_cfg)

        # Entry and condition dominate branches
        assert "entry" in doms["then_branch"]
        assert "condition" in doms["then_branch"]
        # Branches don't dominate each other
        assert "then_branch" not in doms["else_branch"]


class TestPostDominators:
    """Tests for post-dominator computation."""

    def test_simple_post_dominators(self, simple_cfg: CFGSketch) -> None:
        """Test post-dominators on simple CFG."""
        pdoms = compute_post_dominators(simple_cfg)

        # Exit post-dominates all
        assert "exit" in pdoms["entry"]
        assert "exit" in pdoms["body"]


class TestLoopBodies:
    """Tests for loop body detection."""

    def test_simple_loop(self, loop_cfg: CFGSketch) -> None:
        """Test loop body detection."""
        loops = find_loop_bodies(loop_cfg)

        assert "header" in loops
        body = loops["header"]
        assert "header" in body
        assert "body" in body


# =============================================================================
# ControlFlowDomain Tests
# =============================================================================


class TestControlFlowDomain:
    """Tests for ControlFlowDomain."""

    def test_domain_name(self, python_domain: ControlFlowDomain) -> None:
        """Test domain name is 'controlflow'."""
        assert python_domain.name == "controlflow"

    def test_domain_language(self, python_domain: ControlFlowDomain) -> None:
        """Test domain language."""
        assert python_domain.language == "python"

    def test_top_and_bottom(self, python_domain: ControlFlowDomain) -> None:
        """Test domain provides TOP and BOTTOM."""
        assert python_domain.top.is_top()
        assert python_domain.bottom.is_bottom()

    def test_create_constraint_empty(self, python_domain: ControlFlowDomain) -> None:
        """Test creating empty constraint returns TOP."""
        c = python_domain.create_constraint()
        assert c.is_top()

    def test_create_constraint_must_reach(self, python_domain: ControlFlowDomain) -> None:
        """Test creating constraint with must_reach."""
        c = python_domain.create_constraint(must_reach=["exit"])
        assert c.is_must_reach(CodePoint(label="exit"))

    def test_create_constraint_must_not_reach(self, python_domain: ControlFlowDomain) -> None:
        """Test creating constraint with must_not_reach."""
        c = python_domain.create_constraint(must_not_reach=["dead"])
        assert c.is_must_not_reach(CodePoint(label="dead"))

    def test_create_constraint_conflict(self, python_domain: ControlFlowDomain) -> None:
        """Test creating conflicting constraint returns BOTTOM."""
        c = python_domain.create_constraint(
            must_reach=["point"],
            must_not_reach=["point"],
        )
        assert c.is_bottom()


class TestControlFlowDomainTokenMask:
    """Tests for ControlFlowDomain.token_mask()."""

    def test_top_allows_all(
        self,
        python_domain: ControlFlowDomain,
        generation_context: GenerationContext,
    ) -> None:
        """Test TOP constraint allows all tokens."""
        mask = python_domain.token_mask(CONTROLFLOW_TOP, generation_context)
        assert mask.all()

    def test_bottom_blocks_all(
        self,
        python_domain: ControlFlowDomain,
        generation_context: GenerationContext,
    ) -> None:
        """Test BOTTOM constraint blocks all tokens."""
        mask = python_domain.token_mask(CONTROLFLOW_BOTTOM, generation_context)
        assert not mask.any()

    def test_regular_constraint_conservative(
        self,
        python_domain: ControlFlowDomain,
        generation_context: GenerationContext,
    ) -> None:
        """Test regular constraint currently allows all (conservative)."""
        c = controlflow_requiring_reach("exit")
        mask = python_domain.token_mask(c, generation_context)
        # Current implementation is conservative
        assert mask.all()


class TestControlFlowDomainCheckpoint:
    """Tests for ControlFlowDomain checkpoint/restore."""

    def test_checkpoint_preserves_state(self, python_domain: ControlFlowDomain) -> None:
        """Test checkpoint preserves domain state."""
        # Add some blocks to the CFG
        python_domain._cfg.add_block(BasicBlock(id="test1", is_entry=True))
        python_domain._cfg.add_block(BasicBlock(id="test2"))
        python_domain._block_counter = 5

        checkpoint = python_domain.checkpoint()

        assert "test1" in checkpoint.cfg.blocks
        assert "test2" in checkpoint.cfg.blocks
        assert checkpoint.block_counter == 5

    def test_restore_reverts_state(self, python_domain: ControlFlowDomain) -> None:
        """Test restore reverts domain state."""
        python_domain._cfg.add_block(BasicBlock(id="before", is_entry=True))
        checkpoint = python_domain.checkpoint()

        python_domain._cfg.add_block(BasicBlock(id="after"))
        assert "after" in python_domain._cfg.blocks

        python_domain.restore(checkpoint)

        assert "before" in python_domain._cfg.blocks
        assert "after" not in python_domain._cfg.blocks

    def test_restore_wrong_type_raises(self, python_domain: ControlFlowDomain) -> None:
        """Test restore with wrong type raises TypeError."""
        with pytest.raises(TypeError):
            python_domain.restore("not a checkpoint")  # type: ignore

    def test_satisfiability_check(self, python_domain: ControlFlowDomain) -> None:
        """Test satisfiability returns constraint's satisfiability."""
        c = controlflow_requiring_reach("exit")
        assert python_domain.satisfiability(c) == Satisfiability.SAT

        assert python_domain.satisfiability(CONTROLFLOW_BOTTOM) == Satisfiability.UNSAT


# =============================================================================
# Integration Tests
# =============================================================================


class TestControlFlowDomainIntegration:
    """Integration tests for Control Flow Domain."""

    def test_cfg_to_constraint_validation(self) -> None:
        """Test CFG validation against constraint."""
        domain = ControlFlowDomain(language="python")

        # Build a simple CFG
        domain._cfg.add_block(BasicBlock(id="entry", is_entry=True))
        domain._cfg.add_block(BasicBlock(id="exit", is_exit=True))
        domain._cfg.add_edge(CFGEdge(source="entry", target="exit"))

        # Create constraint requiring exit reachable
        c = domain.create_constraint(must_reach=["exit"])

        # Validate - should pass
        result = domain._validate_constraint(c)
        assert not result.is_bottom()

    def test_checkpoint_restore_cycle(self, python_domain: ControlFlowDomain) -> None:
        """Test multiple checkpoint/restore cycles."""
        python_domain._cfg.add_block(BasicBlock(id="b1", is_entry=True))
        cp1 = python_domain.checkpoint()

        python_domain._cfg.add_block(BasicBlock(id="b2"))
        cp2 = python_domain.checkpoint()

        python_domain._cfg.add_block(BasicBlock(id="b3"))

        # Restore to cp2
        python_domain.restore(cp2)
        assert "b1" in python_domain._cfg.blocks
        assert "b2" in python_domain._cfg.blocks
        assert "b3" not in python_domain._cfg.blocks

        # Restore to cp1
        python_domain.restore(cp1)
        assert "b1" in python_domain._cfg.blocks
        assert "b2" not in python_domain._cfg.blocks

    def test_analyze_reachability(self, python_domain: ControlFlowDomain) -> None:
        """Test analyze_reachability method."""
        python_domain._cfg.add_block(BasicBlock(id="entry", is_entry=True))
        python_domain._cfg.add_block(BasicBlock(id="exit", is_exit=True))
        python_domain._cfg.add_edge(CFGEdge(source="entry", target="exit"))

        result = python_domain.analyze_reachability()

        assert "entry" in result.reachable_from_entry
        assert "exit" in result.reachable_from_entry
        assert result.all_paths_return
