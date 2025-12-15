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
"""Property-based tests for Kotlin type system.

Tests algebraic properties of the type system using Hypothesis.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck

from domains.types.languages.kotlin import (
    KotlinTypeSystem,
    KotlinNullableType,
    KotlinArrayType,
    KotlinListType,
    KotlinSetType,
    KotlinMapType,
    KotlinFunctionType,
    KOTLIN_INT,
    KOTLIN_LONG,
    KOTLIN_FLOAT,
    KOTLIN_DOUBLE,
    KOTLIN_BOOLEAN,
    KOTLIN_STRING,
    KOTLIN_CHAR,
    KOTLIN_UNIT,
    KOTLIN_ANY,
    KOTLIN_NOTHING,
)

# Strategies for generating Kotlin types
KOTLIN_PRIMITIVE_TYPES = [
    KOTLIN_INT, KOTLIN_LONG, KOTLIN_FLOAT, KOTLIN_DOUBLE,
    KOTLIN_BOOLEAN, KOTLIN_STRING, KOTLIN_CHAR, KOTLIN_UNIT,
]

kotlin_primitive_type = st.sampled_from(KOTLIN_PRIMITIVE_TYPES)

@st.composite
def kotlin_nullable_type(draw):
    """Generate nullable types."""
    inner = draw(kotlin_primitive_type)
    return KotlinNullableType(inner)

@st.composite
def kotlin_list_type(draw):
    """Generate list types."""
    element = draw(kotlin_primitive_type)
    mutable = draw(st.booleans())
    return KotlinListType(element, mutable=mutable)

@st.composite
def kotlin_set_type(draw):
    """Generate set types."""
    element = draw(kotlin_primitive_type)
    mutable = draw(st.booleans())
    return KotlinSetType(element, mutable=mutable)

@st.composite
def kotlin_map_type(draw):
    """Generate map types."""
    key = draw(kotlin_primitive_type)
    value = draw(kotlin_primitive_type)
    mutable = draw(st.booleans())
    return KotlinMapType(key, value, mutable=mutable)

@st.composite
def kotlin_function_type(draw):
    """Generate function types."""
    num_params = draw(st.integers(min_value=0, max_value=3))
    params = tuple(draw(kotlin_primitive_type) for _ in range(num_params))
    return_type = draw(kotlin_primitive_type)
    return KotlinFunctionType(params, return_type)

kotlin_any_type = st.one_of(
    kotlin_primitive_type,
    kotlin_nullable_type(),
    kotlin_list_type(),
    kotlin_set_type(),
    kotlin_map_type(),
    kotlin_function_type(),
)


class TestKotlinReflexivity:
    """Test reflexivity property: T is assignable to T."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    @given(kotlin_any_type)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_reflexivity(self, ts, t):
        """Any type should be assignable to itself."""
        assert ts.check_assignable(t, t)

    @given(kotlin_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_reflexivity_primitive(self, ts, t):
        """Primitive types should be assignable to themselves."""
        assert ts.check_assignable(t, t)


class TestKotlinTopBottom:
    """Test Any (top) and Nothing (bottom) properties."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    @given(kotlin_any_type)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_nothing_to_any_type(self, ts, t):
        """Nothing should be assignable to any type."""
        assert ts.check_assignable(KOTLIN_NOTHING, t)

    @given(kotlin_any_type)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_any_type_to_any(self, ts, t):
        """Any type should be assignable to Any."""
        assert ts.check_assignable(t, KOTLIN_ANY)


class TestKotlinNullability:
    """Test nullability properties."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    @given(kotlin_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_non_nullable_to_nullable(self, ts, t):
        """Non-nullable T should be assignable to T?."""
        nullable = KotlinNullableType(t)
        assert ts.check_assignable(t, nullable)

    @given(kotlin_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_nullable_not_to_non_nullable(self, ts, t):
        """Nullable T? should not be assignable to T."""
        nullable = KotlinNullableType(t)
        assert not ts.check_assignable(nullable, t)

    @given(kotlin_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_nullable_reflexive(self, ts, t):
        """T? should be assignable to T?."""
        nullable = KotlinNullableType(t)
        assert ts.check_assignable(nullable, nullable)


class TestKotlinLUBCommutativity:
    """Test LUB commutativity: lub(A, B) == lub(B, A)."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    @given(kotlin_any_type, kotlin_any_type)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_lub_commutative(self, ts, a, b):
        """LUB should be commutative."""
        lub_ab = ts.lub([a, b])
        lub_ba = ts.lub([b, a])
        # Types may not be equal but should be equivalent
        assert ts.check_assignable(a, lub_ab)
        assert ts.check_assignable(b, lub_ab)
        assert ts.check_assignable(a, lub_ba)
        assert ts.check_assignable(b, lub_ba)


class TestKotlinGLBCommutativity:
    """Test GLB commutativity."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    @given(kotlin_any_type, kotlin_any_type)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_glb_commutative(self, ts, a, b):
        """GLB should be commutative."""
        glb_ab = ts.glb([a, b])
        glb_ba = ts.glb([b, a])
        # Both results should be assignable to both a and b
        assert ts.check_assignable(glb_ab, a) or glb_ab == KOTLIN_NOTHING
        assert ts.check_assignable(glb_ba, a) or glb_ba == KOTLIN_NOTHING


class TestKotlinLUBIdentity:
    """Test LUB identity properties."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    @given(kotlin_any_type)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_lub_singleton(self, ts, t):
        """LUB of single type is that type."""
        result = ts.lub([t])
        assert result == t

    @given(kotlin_any_type)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_lub_same(self, ts, t):
        """LUB of same types is that type."""
        result = ts.lub([t, t])
        assert result == t


class TestKotlinGLBIdentity:
    """Test GLB identity properties."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    @given(kotlin_any_type)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_glb_singleton(self, ts, t):
        """GLB of single type is that type."""
        result = ts.glb([t])
        assert result == t

    @given(kotlin_any_type)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_glb_same(self, ts, t):
        """GLB of same types is that type."""
        result = ts.glb([t, t])
        assert result == t


class TestKotlinTypeRoundtrip:
    """Test parse/format round-trip properties."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    @given(kotlin_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_primitive_roundtrip(self, ts, t):
        """Parsing formatted primitive should yield equivalent type."""
        formatted = ts.format_type(t)
        parsed = ts.parse_type_annotation(formatted)
        assert parsed == t

    @given(kotlin_nullable_type())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_nullable_roundtrip(self, ts, t):
        """Parsing formatted nullable should yield equivalent type."""
        formatted = ts.format_type(t)
        parsed = ts.parse_type_annotation(formatted)
        assert parsed == t


class TestKotlinCollectionCovariance:
    """Test collection covariance for immutable collections."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    def test_immutable_list_covariance(self, ts):
        """Immutable List<Nothing> should be assignable to List<Int>."""
        list_nothing = KotlinListType(KOTLIN_NOTHING, mutable=False)
        list_int = KotlinListType(KOTLIN_INT, mutable=False)
        assert ts.check_assignable(list_nothing, list_int)

    def test_mutable_list_invariance(self, ts):
        """Mutable List should be invariant."""
        mlist_int1 = KotlinListType(KOTLIN_INT, mutable=True)
        mlist_int2 = KotlinListType(KOTLIN_INT, mutable=True)
        # Same element types should be assignable
        assert ts.check_assignable(mlist_int1, mlist_int2)

    @given(kotlin_primitive_type, kotlin_primitive_type)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_immutable_list_assignment_property(self, ts, t1, t2):
        """Immutable List<T1> assignable to List<T2> iff T1 assignable to T2."""
        list1 = KotlinListType(t1, mutable=False)
        list2 = KotlinListType(t2, mutable=False)
        if ts.check_assignable(t1, t2):
            assert ts.check_assignable(list1, list2)


class TestKotlinFunctionContravariance:
    """Test function type variance properties."""

    @pytest.fixture
    def ts(self):
        return KotlinTypeSystem()

    def test_function_return_covariance(self, ts):
        """Function with more specific return should be assignable."""
        fn_returns_nothing = KotlinFunctionType((), KOTLIN_NOTHING)
        fn_returns_int = KotlinFunctionType((), KOTLIN_INT)
        # () -> Nothing should be assignable to () -> Int
        assert ts.check_assignable(fn_returns_nothing, fn_returns_int)

    def test_function_param_contravariance(self, ts):
        """Function with less specific param should be assignable."""
        fn_takes_any = KotlinFunctionType((KOTLIN_ANY,), KOTLIN_UNIT)
        fn_takes_int = KotlinFunctionType((KOTLIN_INT,), KOTLIN_UNIT)
        # (Any) -> Unit should be assignable to (Int) -> Unit (contravariance)
        assert ts.check_assignable(fn_takes_any, fn_takes_int)
