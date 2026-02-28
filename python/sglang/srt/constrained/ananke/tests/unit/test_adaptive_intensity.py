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
"""Unit tests for adaptive constraint intensity module.

Tests verify:
1. ConstraintIntensity enum behavior
2. TaskComplexityAssessor heuristics
3. Domain selection for each intensity level
4. Integration with ConstraintSpec

Soundness Property:
    Lower intensity = more permissive = SOUND
    Tests verify that intensity reduction never adds restrictions.
"""

import pytest
import sys
from pathlib import Path

# Add the ananke package root to sys.path for standalone testing
_ANANKE_ROOT = Path(__file__).parent.parent.parent
if str(_ANANKE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANANKE_ROOT))

from adaptive.intensity import (
    ConstraintIntensity,
    TaskComplexityAssessor,
    IntensityConfig,
    assess_complexity,
    domains_for_intensity,
    DEFAULT_CONFIG,
)


class TestConstraintIntensity:
    """Tests for ConstraintIntensity enum."""

    def test_ordering(self):
        """Verify intensity levels are properly ordered."""
        assert ConstraintIntensity.NONE < ConstraintIntensity.SYNTAX_ONLY
        assert ConstraintIntensity.SYNTAX_ONLY < ConstraintIntensity.STANDARD
        assert ConstraintIntensity.STANDARD < ConstraintIntensity.FULL
        assert ConstraintIntensity.FULL < ConstraintIntensity.EXHAUSTIVE

    def test_from_string_case_insensitive(self):
        """Test parsing from various string formats."""
        assert ConstraintIntensity.from_string("none") == ConstraintIntensity.NONE
        assert ConstraintIntensity.from_string("NONE") == ConstraintIntensity.NONE
        assert ConstraintIntensity.from_string("None") == ConstraintIntensity.NONE
        assert ConstraintIntensity.from_string("syntax_only") == ConstraintIntensity.SYNTAX_ONLY
        assert ConstraintIntensity.from_string("STANDARD") == ConstraintIntensity.STANDARD
        assert ConstraintIntensity.from_string("full") == ConstraintIntensity.FULL
        assert ConstraintIntensity.from_string("exhaustive") == ConstraintIntensity.EXHAUSTIVE

    def test_from_string_numeric(self):
        """Test parsing from numeric strings."""
        assert ConstraintIntensity.from_string("0") == ConstraintIntensity.NONE
        assert ConstraintIntensity.from_string("1") == ConstraintIntensity.SYNTAX_ONLY
        assert ConstraintIntensity.from_string("2") == ConstraintIntensity.STANDARD
        assert ConstraintIntensity.from_string("3") == ConstraintIntensity.FULL
        assert ConstraintIntensity.from_string("4") == ConstraintIntensity.EXHAUSTIVE

    def test_from_string_invalid_defaults_to_standard(self):
        """Invalid strings should default to STANDARD."""
        assert ConstraintIntensity.from_string("invalid") == ConstraintIntensity.STANDARD
        assert ConstraintIntensity.from_string("") == ConstraintIntensity.STANDARD
        assert ConstraintIntensity.from_string("foo") == ConstraintIntensity.STANDARD

    def test_str_representation(self):
        """Test string representation."""
        assert str(ConstraintIntensity.NONE) == "none"
        assert str(ConstraintIntensity.SYNTAX_ONLY) == "syntax_only"
        assert str(ConstraintIntensity.STANDARD) == "standard"
        assert str(ConstraintIntensity.FULL) == "full"
        assert str(ConstraintIntensity.EXHAUSTIVE) == "exhaustive"


class TestDomainsForIntensity:
    """Tests for domain selection by intensity level."""

    def test_none_has_no_domains(self):
        """NONE intensity should have no domains."""
        domains = domains_for_intensity(ConstraintIntensity.NONE)
        assert domains == frozenset()

    def test_syntax_only_has_syntax(self):
        """SYNTAX_ONLY should have only syntax domain."""
        domains = domains_for_intensity(ConstraintIntensity.SYNTAX_ONLY)
        assert domains == frozenset({"syntax"})

    def test_standard_has_syntax_and_types(self):
        """STANDARD should have syntax and types domains."""
        domains = domains_for_intensity(ConstraintIntensity.STANDARD)
        assert domains == frozenset({"syntax", "types"})

    def test_full_has_all_except_semantics(self):
        """FULL should have syntax, types, imports, controlflow."""
        domains = domains_for_intensity(ConstraintIntensity.FULL)
        assert domains == frozenset({"syntax", "types", "imports", "controlflow"})

    def test_exhaustive_has_all_domains(self):
        """EXHAUSTIVE should have all five domains."""
        domains = domains_for_intensity(ConstraintIntensity.EXHAUSTIVE)
        assert domains == frozenset({"syntax", "types", "imports", "controlflow", "semantics"})

    def test_domain_inclusion_monotonic(self):
        """Higher intensity should include all domains from lower intensities.

        This is the soundness property: lower intensity is more permissive.
        """
        for i in range(len(ConstraintIntensity) - 1):
            lower = ConstraintIntensity(i)
            higher = ConstraintIntensity(i + 1)
            lower_domains = domains_for_intensity(lower)
            higher_domains = domains_for_intensity(higher)
            assert lower_domains <= higher_domains, (
                f"Higher intensity {higher} should include all domains from {lower}"
            )


class TestIntensityConfig:
    """Tests for IntensityConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = IntensityConfig()
        assert config.min_tokens_for_types == 20
        assert config.min_tokens_for_full == 100
        assert config.high_temp_threshold == 1.0
        assert "def " in config.function_keywords
        assert "class " in config.class_keywords
        assert "try:" in config.complex_keywords

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        config = IntensityConfig(
            min_tokens_for_types=30,
            min_tokens_for_full=150,
            high_temp_threshold=0.8,
        )
        d = config.to_dict()
        restored = IntensityConfig.from_dict(d)
        assert restored.min_tokens_for_types == 30
        assert restored.min_tokens_for_full == 150
        assert restored.high_temp_threshold == 0.8

    def test_frozen_immutability(self):
        """Config should be immutable (frozen dataclass)."""
        config = IntensityConfig()
        with pytest.raises(AttributeError):
            config.min_tokens_for_types = 100


class TestTaskComplexityAssessor:
    """Tests for TaskComplexityAssessor."""

    @pytest.fixture
    def assessor(self):
        """Create assessor with default config."""
        return TaskComplexityAssessor()

    def test_short_prompt_gets_syntax_only(self, assessor):
        """Short prompts should get SYNTAX_ONLY."""
        intensity = assessor.assess(prompt="x = ", expected_tokens=5)
        assert intensity == ConstraintIntensity.SYNTAX_ONLY

    def test_function_definition_gets_full(self, assessor):
        """Function definitions should trigger FULL intensity."""
        intensity = assessor.assess(prompt="def process(data):", expected_tokens=50)
        assert intensity == ConstraintIntensity.FULL

    def test_async_function_gets_full(self, assessor):
        """Async function definitions should trigger FULL intensity."""
        intensity = assessor.assess(prompt="async def fetch_data():", expected_tokens=50)
        assert intensity == ConstraintIntensity.FULL

    def test_class_definition_gets_full(self, assessor):
        """Class definitions should trigger FULL intensity."""
        intensity = assessor.assess(prompt="class MyClass:", expected_tokens=50)
        assert intensity == ConstraintIntensity.FULL

    def test_dataclass_gets_full(self, assessor):
        """Dataclass definitions should trigger FULL intensity."""
        intensity = assessor.assess(prompt="@dataclass\nclass User:", expected_tokens=50)
        assert intensity == ConstraintIntensity.FULL

    def test_complex_control_flow_escalates(self, assessor):
        """Complex control flow should escalate intensity."""
        # Without complexity indicators
        base_intensity = assessor.assess(prompt="result = ", expected_tokens=30)
        # With complexity indicators
        complex_intensity = assessor.assess(prompt="try:\n    result = ", expected_tokens=30)
        assert complex_intensity >= base_intensity

    def test_high_temperature_escalates(self, assessor):
        """High temperature should escalate intensity."""
        low_temp = assessor.assess(prompt="x = ", expected_tokens=30, temperature=0.5)
        high_temp = assessor.assess(prompt="x = ", expected_tokens=30, temperature=1.5)
        assert high_temp >= low_temp

    def test_semantic_keywords_trigger_exhaustive(self, assessor):
        """Semantic keywords should trigger EXHAUSTIVE intensity."""
        intensity = assessor.assess(
            prompt="def abs(x):\n    assert x >= 0",
            expected_tokens=50,
        )
        assert intensity == ConstraintIntensity.EXHAUSTIVE

    def test_explicit_intensity_overrides(self, assessor):
        """Explicit intensity should override assessment."""
        intensity = assessor.assess(
            prompt="def very_complex_function():",
            expected_tokens=500,
            explicit_intensity=ConstraintIntensity.SYNTAX_ONLY,
        )
        assert intensity == ConstraintIntensity.SYNTAX_ONLY

    def test_long_generation_gets_full(self, assessor):
        """Long expected generations should get FULL intensity."""
        intensity = assessor.assess(prompt="# Module code", expected_tokens=200)
        assert intensity == ConstraintIntensity.FULL

    def test_medium_generation_gets_standard(self, assessor):
        """Medium length generations should get STANDARD intensity."""
        intensity = assessor.assess(prompt="result = ", expected_tokens=50)
        assert intensity == ConstraintIntensity.STANDARD

    def test_stats_tracking(self, assessor):
        """Test that statistics are tracked correctly."""
        assessor.reset_stats()
        assessor.assess(prompt="x = 1", expected_tokens=5)
        assessor.assess(prompt="def foo():", expected_tokens=50)

        stats = assessor.get_stats()
        total = sum(stats.values())
        assert total == 2


class TestModuleLevelConvenienceFunction:
    """Tests for the module-level assess_complexity function."""

    def test_assess_complexity_function_works(self):
        """Test the convenience function."""
        intensity = assess_complexity(prompt="x = 1", expected_tokens=5)
        assert isinstance(intensity, ConstraintIntensity)

    def test_assess_complexity_with_explicit(self):
        """Test with explicit intensity override."""
        intensity = assess_complexity(
            prompt="def complex():",
            expected_tokens=100,
            explicit_intensity=ConstraintIntensity.NONE,
        )
        assert intensity == ConstraintIntensity.NONE


class TestSoundnessProperty:
    """Tests verifying soundness: lower intensity = more permissive.

    This is the critical property that ensures adaptive intensity
    never blocks valid tokens.
    """

    def test_lower_intensity_never_adds_domains(self):
        """Lower intensity should never have domains that higher doesn't."""
        intensities = list(ConstraintIntensity)
        for i, lower in enumerate(intensities[:-1]):
            for higher in intensities[i + 1:]:
                lower_domains = domains_for_intensity(lower)
                higher_domains = domains_for_intensity(higher)
                extra_domains = lower_domains - higher_domains
                assert not extra_domains, (
                    f"Lower intensity {lower} has domains {extra_domains} "
                    f"that higher {higher} doesn't have"
                )

    def test_none_is_most_permissive(self):
        """NONE intensity should have no constraints."""
        domains = domains_for_intensity(ConstraintIntensity.NONE)
        assert len(domains) == 0, "NONE should have no domains"

    def test_domain_count_increases_with_intensity(self):
        """Domain count should monotonically increase with intensity."""
        prev_count = 0
        for intensity in ConstraintIntensity:
            domains = domains_for_intensity(intensity)
            assert len(domains) >= prev_count, (
                f"Intensity {intensity} has fewer domains than lower level"
            )
            prev_count = len(domains)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
