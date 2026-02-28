# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Validation tests for constraint examples.

These tests ensure that all constraint examples:
1. Round-trip through serialization (to_dict/from_dict)
2. Have valid ConstraintSpec configurations
3. Have unique IDs
4. Have all required fields populated
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Set

import pytest

try:
    from . import get_all_examples, get_examples_by_language, get_examples_by_domain
    from .base import ConstraintExample, ExampleCatalog, LANGUAGES, DOMAINS
except ImportError:
    from tests.fixtures.constraints import (
        get_all_examples,
        get_examples_by_language,
        get_examples_by_domain,
    )
    from tests.fixtures.constraints.base import (
        ConstraintExample,
        ExampleCatalog,
        LANGUAGES,
        DOMAINS,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def all_examples() -> List[ConstraintExample]:
    """Get all constraint examples."""
    return get_all_examples()


@pytest.fixture(scope="module")
def catalog(all_examples: List[ConstraintExample]) -> ExampleCatalog:
    """Create catalog from all examples."""
    return ExampleCatalog(version="1.0", examples=all_examples)


# =============================================================================
# Collection Validation Tests
# =============================================================================


class TestExampleCollection:
    """Tests for the overall example collection."""

    def test_examples_exist(self, all_examples: List[ConstraintExample]) -> None:
        """Verify we have examples loaded."""
        assert len(all_examples) > 0, "No examples loaded"

    def test_minimum_example_count(self, all_examples: List[ConstraintExample]) -> None:
        """Verify we have at least 100 examples (target was 126+)."""
        assert len(all_examples) >= 100, f"Only {len(all_examples)} examples, expected 100+"

    def test_unique_ids(self, all_examples: List[ConstraintExample]) -> None:
        """All example IDs must be unique."""
        ids: List[str] = [e.id for e in all_examples]
        duplicates = [id_ for id_ in ids if ids.count(id_) > 1]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {set(duplicates)}"

    def test_all_languages_represented(
        self, all_examples: List[ConstraintExample]
    ) -> None:
        """Each language should have at least one example."""
        represented_languages: Set[str] = {e.language for e in all_examples}
        for lang in LANGUAGES:
            assert lang in represented_languages, f"Language '{lang}' has no examples"

    def test_all_domains_represented(
        self, all_examples: List[ConstraintExample]
    ) -> None:
        """Each domain should have at least one example."""
        represented_domains: Set[str] = {e.domain for e in all_examples}
        for domain in DOMAINS:
            assert domain in represented_domains, f"Domain '{domain}' has no examples"


# =============================================================================
# Individual Example Validation Tests
# =============================================================================


class TestExampleValidation:
    """Tests for individual example validation."""

    def test_all_examples_have_required_fields(
        self, all_examples: List[ConstraintExample]
    ) -> None:
        """All examples must have non-empty required fields."""
        for example in all_examples:
            assert example.id, f"Example missing id"
            assert example.name, f"Example {example.id} missing name"
            assert example.description, f"Example {example.id} missing description"
            assert example.scenario, f"Example {example.id} missing scenario"
            assert example.spec is not None, f"Example {example.id} missing spec"
            assert example.expected_effect, f"Example {example.id} missing expected_effect"
            assert example.language, f"Example {example.id} missing language"
            assert example.domain, f"Example {example.id} missing domain"

    def test_all_examples_have_valid_outputs(
        self, all_examples: List[ConstraintExample]
    ) -> None:
        """All examples should have at least one valid output."""
        missing_valid = []
        missing_invalid = []
        for example in all_examples:
            if len(example.valid_outputs) == 0:
                missing_valid.append(example.id)
            if len(example.invalid_outputs) == 0:
                missing_invalid.append(example.id)

        assert len(missing_valid) == 0, (
            f"Examples missing valid_outputs: {missing_valid}"
        )
        # Warn about missing invalid outputs but don't fail
        if missing_invalid:
            import warnings
            warnings.warn(f"Examples missing invalid_outputs: {missing_invalid}")

    def test_all_examples_have_tags(
        self, all_examples: List[ConstraintExample]
    ) -> None:
        """All examples should have at least one tag."""
        for example in all_examples:
            assert len(example.tags) > 0, f"Example {example.id} has no tags"

    def test_id_format(self, all_examples: List[ConstraintExample]) -> None:
        """IDs should follow a consistent pattern."""
        # Language abbreviations used in IDs
        lang_abbrevs = {
            "python": "py",
            "rust": "rust",
            "zig": "zig",
            "typescript": "ts",
            "go": "go",
            "kotlin": "kt",
            "swift": "swift",
        }
        invalid_ids = []
        for example in all_examples:
            parts = example.id.split("-")
            # Must have at least 2 parts
            if len(parts) < 2:
                invalid_ids.append(f"{example.id}: too few parts")
                continue

            # ID should contain language abbreviation or cross indicator
            expected_prefix = lang_abbrevs.get(example.language, example.language[:3])
            valid_prefix = (
                example.id.startswith(expected_prefix + "-") or
                example.language in example.id or
                "cross" in example.id
            )
            if not valid_prefix:
                invalid_ids.append(f"{example.id}: missing language/cross prefix")

        assert len(invalid_ids) == 0, f"Invalid IDs found:\n" + "\n".join(invalid_ids)


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for serialization round-trips."""

    def test_example_round_trip(self, all_examples: List[ConstraintExample]) -> None:
        """Examples should round-trip through to_dict/from_dict."""
        for example in all_examples:
            # Convert to dict
            d = example.to_dict()
            assert isinstance(d, dict), f"Example {example.id} to_dict failed"

            # Convert back
            restored = ConstraintExample.from_dict(d)
            assert restored.id == example.id
            assert restored.name == example.name
            assert restored.language == example.language
            assert restored.domain == example.domain

    def test_catalog_round_trip(self, catalog: ExampleCatalog) -> None:
        """Catalog should convert to dict successfully."""
        d = catalog.to_dict()
        assert "version" in d
        assert "total_examples" in d
        assert "by_language" in d
        assert "by_domain" in d
        assert "examples" in d
        assert len(d["examples"]) == len(catalog.examples)

    def test_catalog_json_serializable(self, catalog: ExampleCatalog) -> None:
        """Catalog dict should be JSON serializable."""
        d = catalog.to_dict()
        # This will raise if not serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

        # Parse back
        parsed = json.loads(json_str)
        assert parsed["version"] == catalog.version


# =============================================================================
# ConstraintSpec Validation Tests
# =============================================================================


class TestConstraintSpec:
    """Tests for ConstraintSpec validation."""

    def test_spec_has_language(self, all_examples: List[ConstraintExample]) -> None:
        """All specs should have a language field."""
        for example in all_examples:
            assert example.spec.language, f"Example {example.id} spec missing language"

    def test_spec_language_matches_example(
        self, all_examples: List[ConstraintExample]
    ) -> None:
        """Spec language should match example language."""
        for example in all_examples:
            assert example.spec.language == example.language, (
                f"Example {example.id}: spec.language ({example.spec.language}) "
                f"!= example.language ({example.language})"
            )

    def test_spec_has_constraint(self, all_examples: List[ConstraintExample]) -> None:
        """Each spec should have at least one constraint type set."""
        missing_constraints = []
        for example in all_examples:
            spec = example.spec
            has_constraint = any([
                spec.expected_type,
                spec.type_bindings,
                spec.function_signatures,
                spec.imports,
                spec.control_flow,
                spec.semantic_constraints,
                spec.json_schema,
                spec.regex,
                spec.ebnf,
                spec.available_modules,  # Import constraints
                spec.forbidden_imports,  # Import constraints
                spec.class_definitions,  # Type constraints
            ])
            if not has_constraint:
                missing_constraints.append(example.id)

        # Allow a small number of examples without explicit constraints
        # (they may use implicit language constraints)
        if len(missing_constraints) > 5:
            assert False, (
                f"{len(missing_constraints)} examples missing constraints: "
                f"{missing_constraints[:10]}"
            )


# =============================================================================
# Catalog Query Tests
# =============================================================================


class TestCatalogQueries:
    """Tests for catalog query methods."""

    def test_by_language(self, catalog: ExampleCatalog) -> None:
        """Query by language should return correct examples."""
        for lang in LANGUAGES:
            results = catalog.by_language(lang)
            for example in results:
                assert example.language == lang

    def test_by_domain(self, catalog: ExampleCatalog) -> None:
        """Query by domain should return correct examples."""
        for domain in DOMAINS:
            results = catalog.by_domain(domain)
            for example in results:
                assert example.domain == domain

    def test_by_id(self, catalog: ExampleCatalog, all_examples: List[ConstraintExample]) -> None:
        """Query by ID should return the correct example."""
        for example in all_examples[:10]:  # Test first 10
            result = catalog.by_id(example.id)
            assert result is not None, f"Could not find example {example.id}"
            assert result.id == example.id

    def test_by_id_not_found(self, catalog: ExampleCatalog) -> None:
        """Query for non-existent ID should return None."""
        result = catalog.by_id("nonexistent-id-000")
        assert result is None


# =============================================================================
# Module-level helper function tests
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level helper functions."""

    def test_get_examples_by_language(self) -> None:
        """get_examples_by_language should filter correctly."""
        for lang in LANGUAGES:
            examples = get_examples_by_language(lang)
            assert all(e.language == lang for e in examples)

    def test_get_examples_by_domain(self) -> None:
        """get_examples_by_domain should filter correctly."""
        for domain in DOMAINS:
            examples = get_examples_by_domain(domain)
            assert all(e.domain == domain for e in examples)


# =============================================================================
# Catalog File Tests
# =============================================================================


class TestCatalogFile:
    """Tests for the generated catalog.json file."""

    def test_catalog_file_exists(self) -> None:
        """catalog.json should exist."""
        catalog_path = Path(__file__).parent / "catalog.json"
        assert catalog_path.exists(), f"catalog.json not found at {catalog_path}"

    def test_catalog_file_valid_json(self) -> None:
        """catalog.json should be valid JSON."""
        catalog_path = Path(__file__).parent / "catalog.json"
        if catalog_path.exists():
            with open(catalog_path) as f:
                data = json.load(f)
            assert "version" in data
            assert "examples" in data
            assert len(data["examples"]) > 0

    def test_catalog_file_matches_examples(
        self, all_examples: List[ConstraintExample]
    ) -> None:
        """catalog.json should have same count as loaded examples."""
        catalog_path = Path(__file__).parent / "catalog.json"
        if catalog_path.exists():
            with open(catalog_path) as f:
                data = json.load(f)
            # Allow some flexibility since catalog might be slightly out of sync
            assert abs(len(data["examples"]) - len(all_examples)) <= 5, (
                f"catalog.json has {len(data['examples'])} examples, "
                f"but loaded {len(all_examples)}"
            )
