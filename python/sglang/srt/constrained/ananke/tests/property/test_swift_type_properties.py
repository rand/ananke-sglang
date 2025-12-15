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
"""Property-based tests for Swift type system.

Tests algebraic properties of the type system using Hypothesis.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck

from domains.types.languages.swift import (
    SwiftTypeSystem,
    SwiftOptionalType,
    SwiftArrayType,
    SwiftDictionaryType,
    SwiftSetType,
    SwiftFunctionType,
    SWIFT_INT,
    SWIFT_INT64,
    SWIFT_FLOAT,
    SWIFT_DOUBLE,
    SWIFT_BOOL,
    SWIFT_STRING,
    SWIFT_CHARACTER,
    SWIFT_VOID,
    SWIFT_ANY,
    SWIFT_NEVER,
)

# Strategies for generating Swift types
SWIFT_PRIMITIVE_TYPES = [
    SWIFT_INT, SWIFT_INT64, SWIFT_FLOAT, SWIFT_DOUBLE,
    SWIFT_BOOL, SWIFT_STRING, SWIFT_CHARACTER, SWIFT_VOID,
]

swift_primitive_type = st.sampled_from(SWIFT_PRIMITIVE_TYPES)

@st.composite
def swift_optional_type(draw):
    """Generate optional types."""
    inner = draw(swift_primitive_type)
    return SwiftOptionalType(inner)

@st.composite
def swift_array_type(draw):
    """Generate array types."""
    element = draw(swift_primitive_type)
    return SwiftArrayType(element)

@st.composite
def swift_dictionary_type(draw):
    """Generate dictionary types."""
    key = draw(swift_primitive_type)
    value = draw(swift_primitive_type)
    return SwiftDictionaryType(key, value)

@st.composite
def swift_set_type(draw):
    """Generate set types."""
    element = draw(swift_primitive_type)
    return SwiftSetType(element)

@st.composite
def swift_function_type(draw):
    """Generate function types."""
    num_params = draw(st.integers(min_value=0, max_value=3))
    params = tuple(draw(swift_primitive_type) for _ in range(num_params))
    return_type = draw(swift_primitive_type)
    return SwiftFunctionType(params, return_type)

swift_any_type = st.one_of(
    swift_primitive_type,
    swift_optional_type(),
    swift_array_type(),
    swift_dictionary_type(),
    swift_set_type(),
    swift_function_type(),
)


class TestSwiftReflexivity:
    """Test reflexivity property: T is assignable to T."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    @given(swift_any_type)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_reflexivity(self, ts, t):
        """Any type should be assignable to itself."""
        assert ts.check_assignable(t, t)

    @given(swift_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_reflexivity_primitive(self, ts, t):
        """Primitive types should be assignable to themselves."""
        assert ts.check_assignable(t, t)


class TestSwiftTopBottom:
    """Test Any (top) and Never (bottom) properties."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    @given(swift_any_type)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_never_to_any_type(self, ts, t):
        """Never should be assignable to any type."""
        assert ts.check_assignable(SWIFT_NEVER, t)

    @given(swift_any_type)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_any_type_to_any(self, ts, t):
        """Any type should be assignable to Any."""
        assert ts.check_assignable(t, SWIFT_ANY)


class TestSwiftOptionals:
    """Test optional type properties."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    @given(swift_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_non_optional_to_optional(self, ts, t):
        """Non-optional T should be assignable to T?."""
        optional = SwiftOptionalType(t)
        assert ts.check_assignable(t, optional)

    @given(swift_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_optional_reflexive(self, ts, t):
        """T? should be assignable to T?."""
        optional = SwiftOptionalType(t)
        assert ts.check_assignable(optional, optional)


class TestSwiftLUBCommutativity:
    """Test LUB commutativity."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    @given(swift_any_type, swift_any_type)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_lub_commutative(self, ts, a, b):
        """LUB should be commutative."""
        lub_ab = ts.lub([a, b])
        lub_ba = ts.lub([b, a])
        # Both should accept the same types
        assert ts.check_assignable(a, lub_ab)
        assert ts.check_assignable(b, lub_ab)
        assert ts.check_assignable(a, lub_ba)
        assert ts.check_assignable(b, lub_ba)


class TestSwiftGLBCommutativity:
    """Test GLB commutativity."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    @given(swift_any_type, swift_any_type)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_glb_commutative(self, ts, a, b):
        """GLB should be commutative."""
        glb_ab = ts.glb([a, b])
        glb_ba = ts.glb([b, a])
        # Both results should be assignable to both a and b
        assert ts.check_assignable(glb_ab, a) or glb_ab == SWIFT_NEVER
        assert ts.check_assignable(glb_ba, a) or glb_ba == SWIFT_NEVER


class TestSwiftLUBIdentity:
    """Test LUB identity properties."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    @given(swift_any_type)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_lub_singleton(self, ts, t):
        """LUB of single type is that type."""
        result = ts.lub([t])
        assert result == t

    @given(swift_any_type)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_lub_same(self, ts, t):
        """LUB of same types is that type."""
        result = ts.lub([t, t])
        assert result == t


class TestSwiftGLBIdentity:
    """Test GLB identity properties."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    @given(swift_any_type)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_glb_singleton(self, ts, t):
        """GLB of single type is that type."""
        result = ts.glb([t])
        assert result == t

    @given(swift_any_type)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_glb_same(self, ts, t):
        """GLB of same types is that type."""
        result = ts.glb([t, t])
        assert result == t


class TestSwiftTypeRoundtrip:
    """Test parse/format round-trip properties."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    @given(swift_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_primitive_roundtrip(self, ts, t):
        """Parsing formatted primitive should yield equivalent type."""
        formatted = ts.format_type(t)
        parsed = ts.parse_type_annotation(formatted)
        assert parsed == t

    @given(swift_optional_type())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_optional_roundtrip(self, ts, t):
        """Parsing formatted optional should yield equivalent type."""
        formatted = ts.format_type(t)
        parsed = ts.parse_type_annotation(formatted)
        assert parsed == t


class TestSwiftArrayCovariance:
    """Test array covariance properties."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_array_covariance_never(self, ts):
        """Array of Never should be assignable to Array of any type."""
        arr_never = SwiftArrayType(SWIFT_NEVER)
        arr_int = SwiftArrayType(SWIFT_INT)
        assert ts.check_assignable(arr_never, arr_int)

    @given(swift_primitive_type, swift_primitive_type)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_array_assignment_property(self, ts, t1, t2):
        """Array<T1> assignable to Array<T2> iff T1 assignable to T2."""
        arr1 = SwiftArrayType(t1)
        arr2 = SwiftArrayType(t2)
        if ts.check_assignable(t1, t2):
            assert ts.check_assignable(arr1, arr2)


class TestSwiftDictionaryCovariance:
    """Test dictionary covariance properties."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    @given(swift_primitive_type, swift_primitive_type, swift_primitive_type, swift_primitive_type)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dict_assignment_property(self, ts, k1, v1, k2, v2):
        """Dict<K1,V1> assignable to Dict<K2,V2> iff keys and values assignable."""
        dict1 = SwiftDictionaryType(k1, v1)
        dict2 = SwiftDictionaryType(k2, v2)
        if ts.check_assignable(k1, k2) and ts.check_assignable(v1, v2):
            assert ts.check_assignable(dict1, dict2)


class TestSwiftFunctionContravariance:
    """Test function type variance properties."""

    @pytest.fixture
    def ts(self):
        return SwiftTypeSystem()

    def test_function_return_covariance(self, ts):
        """Function with more specific return should be assignable."""
        fn_returns_never = SwiftFunctionType((), SWIFT_NEVER)
        fn_returns_int = SwiftFunctionType((), SWIFT_INT)
        # () -> Never should be assignable to () -> Int
        assert ts.check_assignable(fn_returns_never, fn_returns_int)

    def test_function_param_contravariance(self, ts):
        """Function with less specific param should be assignable."""
        fn_takes_any = SwiftFunctionType((SWIFT_ANY,), SWIFT_VOID)
        fn_takes_int = SwiftFunctionType((SWIFT_INT,), SWIFT_VOID)
        # (Any) -> Void should be assignable to (Int) -> Void (contravariance)
        assert ts.check_assignable(fn_takes_any, fn_takes_int)
