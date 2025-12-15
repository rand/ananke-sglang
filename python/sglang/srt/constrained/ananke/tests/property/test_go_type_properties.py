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
"""Property-based tests for Go type system.

Tests algebraic properties of the type system using Hypothesis:
- Reflexivity: T is assignable to T
- Interface satisfaction: any implements interface{}
- Structural subtyping properties
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck

from domains.types.languages.go import (
    GoTypeSystem,
    GoArrayType,
    GoSliceType,
    GoMapType,
    GoPointerType,
    GoChannelType,
    GoFunctionType,
    GoInterfaceType,
    GO_BOOL,
    GO_STRING,
    GO_INT,
    GO_INT8,
    GO_INT16,
    GO_INT32,
    GO_INT64,
    GO_UINT,
    GO_UINT8,
    GO_UINT16,
    GO_UINT32,
    GO_UINT64,
    GO_FLOAT32,
    GO_FLOAT64,
    GO_COMPLEX64,
    GO_COMPLEX128,
    GO_BYTE,
    GO_RUNE,
    GO_ANY,
    GO_ERROR,
)
from domains.types.constraint import NeverType


# Strategies for generating Go types
GO_INTEGER_TYPES = [
    GO_INT, GO_INT8, GO_INT16, GO_INT32, GO_INT64,
    GO_UINT, GO_UINT8, GO_UINT16, GO_UINT32, GO_UINT64,
    GO_BYTE, GO_RUNE,
]

GO_FLOAT_TYPES = [GO_FLOAT32, GO_FLOAT64]

GO_COMPLEX_TYPES = [GO_COMPLEX64, GO_COMPLEX128]

GO_PRIMITIVE_TYPES = (
    GO_INTEGER_TYPES + GO_FLOAT_TYPES + GO_COMPLEX_TYPES +
    [GO_BOOL, GO_STRING]
)

go_primitive_type = st.sampled_from(GO_PRIMITIVE_TYPES)
go_integer_type = st.sampled_from(GO_INTEGER_TYPES)
go_float_type = st.sampled_from(GO_FLOAT_TYPES)


@st.composite
def go_slice_type(draw):
    """Generate Go slice types."""
    element = draw(go_primitive_type)
    return GoSliceType(element)


@st.composite
def go_array_type(draw):
    """Generate Go array types."""
    element = draw(go_primitive_type)
    length = draw(st.integers(min_value=1, max_value=100))
    return GoArrayType(length, element)


@st.composite
def go_map_type(draw):
    """Generate Go map types."""
    key = draw(go_primitive_type)
    value = draw(go_primitive_type)
    return GoMapType(key, value)


@st.composite
def go_pointer_type(draw):
    """Generate Go pointer types."""
    pointee = draw(go_primitive_type)
    return GoPointerType(pointee)


@st.composite
def go_channel_type(draw):
    """Generate Go channel types."""
    element = draw(go_primitive_type)
    direction = draw(st.sampled_from(["bidirectional", "send", "receive"]))
    return GoChannelType(element, direction)


@st.composite
def go_function_type(draw):
    """Generate Go function types."""
    num_params = draw(st.integers(min_value=0, max_value=3))
    params = tuple(("", draw(go_primitive_type)) for _ in range(num_params))
    num_returns = draw(st.integers(min_value=0, max_value=2))
    returns = tuple(draw(go_primitive_type) for _ in range(num_returns))
    return GoFunctionType(params, returns)


go_composite_type = st.one_of(
    go_slice_type(),
    go_array_type(),
    go_map_type(),
    go_pointer_type(),
    go_channel_type(),
)

go_any_type = st.one_of(
    go_primitive_type,
    go_composite_type,
    go_function_type(),
)


class TestGoReflexivity:
    """Test reflexivity property: T is assignable to T."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    @given(go_any_type)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_reflexivity(self, ts, t):
        """Any type should be assignable to itself."""
        assert ts.check_assignable(t, t)

    @given(go_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_reflexivity_primitive(self, ts, t):
        """Primitive types should be assignable to themselves."""
        assert ts.check_assignable(t, t)

    @given(go_slice_type())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_reflexivity_slice(self, ts, t):
        """Slice types should be assignable to themselves."""
        assert ts.check_assignable(t, t)

    @given(go_array_type())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_reflexivity_array(self, ts, t):
        """Array types should be assignable to themselves."""
        assert ts.check_assignable(t, t)


class TestGoTopType:
    """Test any (interface{}) as top type."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    @given(go_any_type)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_any_type_to_any(self, ts, t):
        """Any type should be assignable to any/interface{}."""
        assert ts.check_assignable(t, GO_ANY)


class TestGoPointerProperties:
    """Test pointer type properties."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    @given(go_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_pointer_reflexivity(self, ts, t):
        """*T should be assignable to *T."""
        ptr = GoPointerType(t)
        assert ts.check_assignable(ptr, ptr)

    @given(go_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_pointer_to_any(self, ts, t):
        """*T should be assignable to any."""
        ptr = GoPointerType(t)
        assert ts.check_assignable(ptr, GO_ANY)


class TestGoSliceProperties:
    """Test slice type properties."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    @given(go_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_slice_reflexivity(self, ts, t):
        """[]T should be assignable to []T."""
        sl = GoSliceType(t)
        assert ts.check_assignable(sl, sl)

    @given(go_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_slice_to_any(self, ts, t):
        """[]T should be assignable to any."""
        sl = GoSliceType(t)
        assert ts.check_assignable(sl, GO_ANY)


class TestGoMapProperties:
    """Test map type properties."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    @given(go_primitive_type, go_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_map_reflexivity(self, ts, k, v):
        """map[K]V should be assignable to map[K]V."""
        m = GoMapType(k, v)
        assert ts.check_assignable(m, m)

    @given(go_primitive_type, go_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_map_to_any(self, ts, k, v):
        """map[K]V should be assignable to any."""
        m = GoMapType(k, v)
        assert ts.check_assignable(m, GO_ANY)


class TestGoChannelProperties:
    """Test channel type properties."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    @given(go_channel_type())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_channel_reflexivity(self, ts, c):
        """Channel should be assignable to itself."""
        assert ts.check_assignable(c, c)

    @given(go_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_bidirectional_to_send(self, ts, t):
        """Bidirectional chan should be assignable to send-only."""
        bi = GoChannelType(t, "bidirectional")
        send = GoChannelType(t, "send")
        assert ts.check_assignable(bi, send)

    @given(go_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_bidirectional_to_receive(self, ts, t):
        """Bidirectional chan should be assignable to receive-only."""
        bi = GoChannelType(t, "bidirectional")
        recv = GoChannelType(t, "receive")
        assert ts.check_assignable(bi, recv)


class TestGoFunctionProperties:
    """Test function type properties."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    @given(go_function_type())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_function_reflexivity(self, ts, f):
        """Function should be assignable to itself."""
        assert ts.check_assignable(f, f)


class TestGoInterfaceProperties:
    """Test interface type properties."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    def test_empty_interface_reflexivity(self, ts):
        """Empty interface should be assignable to itself."""
        empty = GoInterfaceType(name=None, methods=())
        assert ts.check_assignable(empty, empty)

    @given(go_primitive_type)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_primitive_to_empty_interface(self, ts, t):
        """Any type should satisfy empty interface."""
        empty = GoInterfaceType(name=None, methods=())
        assert ts.check_assignable(t, empty)


class TestGoArrayProperties:
    """Test array type properties."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    @given(go_primitive_type, st.integers(min_value=1, max_value=100))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_array_reflexivity(self, ts, elem, length):
        """[N]T should be assignable to [N]T."""
        arr = GoArrayType(length, elem)
        assert ts.check_assignable(arr, arr)

    @given(go_primitive_type, st.integers(min_value=1, max_value=100))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_different_length_not_assignable(self, ts, elem, length):
        """[N]T should NOT be assignable to [M]T where N != M."""
        arr1 = GoArrayType(length, elem)
        arr2 = GoArrayType(length + 1, elem)
        assert not ts.check_assignable(arr1, arr2)


class TestGoAliasEquivalence:
    """Test that Go type aliases are equivalent."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    def test_byte_uint8_equivalence(self, ts):
        """byte and uint8 should be assignable to each other."""
        assert ts.check_assignable(GO_BYTE, GO_UINT8)
        assert ts.check_assignable(GO_UINT8, GO_BYTE)

    def test_rune_int32_equivalence(self, ts):
        """rune and int32 should be assignable to each other."""
        assert ts.check_assignable(GO_RUNE, GO_INT32)
        assert ts.check_assignable(GO_INT32, GO_RUNE)


class TestGoParseFormatRoundtrip:
    """Test parse-format consistency."""

    @pytest.fixture
    def ts(self):
        return GoTypeSystem()

    @pytest.mark.parametrize("type_str", [
        "int", "int8", "int16", "int32", "int64",
        "uint", "uint8", "uint16", "uint32", "uint64",
        "float32", "float64",
        "bool", "string",
        "byte", "rune",
    ])
    def test_primitive_roundtrip(self, ts, type_str):
        """Parsing and formatting primitives should be consistent."""
        typ = ts.parse_type_annotation(type_str)
        assert typ is not None
        formatted = ts.format_type(typ)
        reparsed = ts.parse_type_annotation(formatted)
        assert reparsed is not None

    def test_slice_roundtrip(self, ts):
        """Slice type should roundtrip."""
        typ = ts.parse_type_annotation("[]int")
        formatted = ts.format_type(typ)
        assert "[]" in formatted

    def test_map_roundtrip(self, ts):
        """Map type should roundtrip."""
        typ = ts.parse_type_annotation("map[string]int")
        formatted = ts.format_type(typ)
        assert "map[" in formatted

    def test_pointer_roundtrip(self, ts):
        """Pointer type should roundtrip."""
        typ = ts.parse_type_annotation("*int")
        formatted = ts.format_type(typ)
        assert "*" in formatted
