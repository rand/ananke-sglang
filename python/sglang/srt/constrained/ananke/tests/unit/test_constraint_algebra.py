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
"""Unit tests for constraint algebra (semilattice laws)."""

import pytest

# Import from core modules directly (conftest.py sets up the path)
from core.constraint import (
    BOTTOM,
    TOP,
    BottomConstraint,
    Constraint,
    Satisfiability,
    TopConstraint,
    verify_semilattice_laws,
)
from core.unified import (
    UNIFIED_BOTTOM,
    UNIFIED_TOP,
    UnifiedConstraint,
)


class TestSatisfiability:
    """Tests for Satisfiability enum operations."""

    def test_sat_and_sat(self):
        """SAT ∧ SAT = SAT"""
        assert Satisfiability.SAT & Satisfiability.SAT == Satisfiability.SAT

    def test_sat_and_unsat(self):
        """SAT ∧ UNSAT = UNSAT"""
        assert Satisfiability.SAT & Satisfiability.UNSAT == Satisfiability.UNSAT

    def test_unsat_and_sat(self):
        """UNSAT ∧ SAT = UNSAT"""
        assert Satisfiability.UNSAT & Satisfiability.SAT == Satisfiability.UNSAT

    def test_sat_and_unknown(self):
        """SAT ∧ UNKNOWN = UNKNOWN"""
        assert Satisfiability.SAT & Satisfiability.UNKNOWN == Satisfiability.UNKNOWN

    def test_unknown_and_unsat(self):
        """UNKNOWN ∧ UNSAT = UNSAT"""
        assert Satisfiability.UNKNOWN & Satisfiability.UNSAT == Satisfiability.UNSAT

    def test_sat_or_unsat(self):
        """SAT ∨ UNSAT = SAT"""
        assert Satisfiability.SAT | Satisfiability.UNSAT == Satisfiability.SAT

    def test_unsat_or_unknown(self):
        """UNSAT ∨ UNKNOWN = UNKNOWN"""
        assert Satisfiability.UNSAT | Satisfiability.UNKNOWN == Satisfiability.UNKNOWN


class TestTopConstraint:
    """Tests for TOP constraint singleton."""

    def test_singleton(self):
        """TOP is a singleton."""
        assert TopConstraint() is TOP
        assert TopConstraint() is TopConstraint()

    def test_is_top(self):
        """TOP.is_top() returns True."""
        assert TOP.is_top() is True

    def test_is_bottom(self):
        """TOP.is_bottom() returns False."""
        assert TOP.is_bottom() is False

    def test_satisfiability(self):
        """TOP is satisfiable."""
        assert TOP.satisfiability() == Satisfiability.SAT

    def test_meet_identity(self):
        """c ⊓ ⊤ = c (identity law with TOP as right operand)."""
        assert TOP.meet(TOP) == TOP
        assert TOP.meet(BOTTOM) == BOTTOM

    def test_equality(self):
        """TOP equals itself."""
        assert TOP == TOP
        assert TOP == TopConstraint()

    def test_hash(self):
        """TOP has consistent hash."""
        assert hash(TOP) == hash(TopConstraint())


class TestBottomConstraint:
    """Tests for BOTTOM constraint singleton."""

    def test_singleton(self):
        """BOTTOM is a singleton."""
        assert BottomConstraint() is BOTTOM
        assert BottomConstraint() is BottomConstraint()

    def test_is_top(self):
        """BOTTOM.is_top() returns False."""
        assert BOTTOM.is_top() is False

    def test_is_bottom(self):
        """BOTTOM.is_bottom() returns True."""
        assert BOTTOM.is_bottom() is True

    def test_satisfiability(self):
        """BOTTOM is unsatisfiable."""
        assert BOTTOM.satisfiability() == Satisfiability.UNSAT

    def test_meet_annihilation(self):
        """c ⊓ ⊥ = ⊥ (annihilation law)."""
        assert BOTTOM.meet(TOP) == BOTTOM
        assert BOTTOM.meet(BOTTOM) == BOTTOM

    def test_equality(self):
        """BOTTOM equals itself."""
        assert BOTTOM == BOTTOM
        assert BOTTOM == BottomConstraint()

    def test_hash(self):
        """BOTTOM has consistent hash."""
        assert hash(BOTTOM) == hash(BottomConstraint())


class TestSemilattice:
    """Tests for semilattice laws."""

    def test_top_bottom_laws(self):
        """Basic semilattice laws with TOP and BOTTOM."""
        # Identity: c ⊓ ⊤ = c
        assert TOP.meet(TOP) == TOP
        assert BOTTOM.meet(TOP) == BOTTOM

        # Annihilation: c ⊓ ⊥ = ⊥
        assert TOP.meet(BOTTOM) == BOTTOM
        assert BOTTOM.meet(BOTTOM) == BOTTOM

    def test_verify_semilattice_laws(self):
        """Test the verification function itself."""
        # Should pass for TOP, TOP, TOP
        assert verify_semilattice_laws(TOP, TOP, TOP) is True

        # Should pass for combinations with BOTTOM
        assert verify_semilattice_laws(BOTTOM, TOP, TOP) is True
        assert verify_semilattice_laws(TOP, BOTTOM, TOP) is True
        assert verify_semilattice_laws(TOP, TOP, BOTTOM) is True


class TestUnifiedConstraint:
    """Tests for UnifiedConstraint (product type)."""

    def test_default_is_top(self):
        """Default UnifiedConstraint is TOP."""
        uc = UnifiedConstraint()
        assert uc.is_top() is True
        assert uc.is_bottom() is False
        assert uc.satisfiability() == Satisfiability.SAT

    def test_unified_top_singleton(self):
        """UNIFIED_TOP has all TOP components."""
        assert UNIFIED_TOP.is_top() is True
        assert UNIFIED_TOP.syntax.is_top() is True
        assert UNIFIED_TOP.types.is_top() is True

    def test_unified_bottom(self):
        """UNIFIED_BOTTOM has all BOTTOM components."""
        assert UNIFIED_BOTTOM.is_bottom() is True
        assert UNIFIED_BOTTOM.syntax.is_bottom() is True
        assert UNIFIED_BOTTOM.satisfiability() == Satisfiability.UNSAT

    def test_meet_identity(self):
        """c ⊓ ⊤ = c for UnifiedConstraint."""
        uc = UnifiedConstraint(syntax=BOTTOM)  # syntax is BOTTOM
        result = uc.meet(UNIFIED_TOP)
        assert result == uc

    def test_meet_annihilation(self):
        """c ⊓ ⊥ = ⊥ for UnifiedConstraint."""
        uc = UnifiedConstraint()  # all TOP
        result = uc.meet(UNIFIED_BOTTOM)
        assert result.is_bottom() is True

    def test_partial_bottom(self):
        """UnifiedConstraint with one BOTTOM component is BOTTOM."""
        uc = UnifiedConstraint(types=BOTTOM)  # only types is BOTTOM
        assert uc.is_bottom() is True
        assert uc.satisfiability() == Satisfiability.UNSAT

    def test_with_methods(self):
        """Test with_* methods for immutable updates."""
        uc = UnifiedConstraint()
        uc2 = uc.with_syntax(BOTTOM)
        assert uc.syntax == TOP  # Original unchanged
        assert uc2.syntax == BOTTOM  # New one has BOTTOM

    def test_component_wise_meet(self):
        """Meet is component-wise."""
        uc1 = UnifiedConstraint(syntax=BOTTOM)
        uc2 = UnifiedConstraint(types=BOTTOM)
        result = uc1.meet(uc2)
        # Both BOTTOM components should be present
        assert result.syntax.is_bottom() is True
        assert result.types.is_bottom() is True

    def test_equality(self):
        """Equality is component-wise."""
        uc1 = UnifiedConstraint(syntax=TOP, types=TOP)
        uc2 = UnifiedConstraint(syntax=TOP, types=TOP)
        assert uc1 == uc2

    def test_hash_consistency(self):
        """Hash is consistent with equality."""
        uc1 = UnifiedConstraint(syntax=TOP, types=TOP)
        uc2 = UnifiedConstraint(syntax=TOP, types=TOP)
        assert hash(uc1) == hash(uc2)

    def test_repr(self):
        """Repr shows non-TOP components."""
        uc = UnifiedConstraint()
        assert "TOP" in repr(uc)

        uc2 = UnifiedConstraint(syntax=BOTTOM)
        assert "syntax" in repr(uc2)


class TestConstraintOperator:
    """Tests for the & operator alias."""

    def test_and_operator(self):
        """& operator is alias for meet."""
        assert (TOP & TOP) == TOP.meet(TOP)
        assert (TOP & BOTTOM) == TOP.meet(BOTTOM)
        assert (BOTTOM & TOP) == BOTTOM.meet(TOP)

    def test_is_satisfiable_helper(self):
        """Test is_satisfiable() helper method."""
        assert TOP.is_satisfiable() is True
        assert BOTTOM.is_satisfiable() is False

    def test_is_unsatisfiable_helper(self):
        """Test is_unsatisfiable() helper method."""
        assert TOP.is_unsatisfiable() is False
        assert BOTTOM.is_unsatisfiable() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
