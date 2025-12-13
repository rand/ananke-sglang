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
"""Extractors for semantic formulas from code.

Extracts SMT formulas from:
- Assert statements
- Pre/post conditions in docstrings
- Type annotations with constraints
- Loop invariants
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set

from .constraint import SMTFormula, FormulaKind


@dataclass
class ExtractionResult:
    """Result of formula extraction.

    Attributes:
        formulas: Extracted formulas
        source: Source text that was analyzed
        errors: Any extraction errors
    """

    formulas: List[SMTFormula]
    source: str
    errors: List[str]


class FormulaExtractor:
    """Base class for formula extractors."""

    def extract(self, source: str) -> ExtractionResult:
        """Extract formulas from source text.

        Args:
            source: Source code text

        Returns:
            ExtractionResult with extracted formulas
        """
        raise NotImplementedError


class PythonAssertExtractor(FormulaExtractor):
    """Extract formulas from Python assert statements."""

    # Pattern for assert statements
    _ASSERT_PATTERN = re.compile(
        r'assert\s+(.+?)(?:,\s*["\'](.+?)["\'])?(?:\s*#|$|\n)',
        re.MULTILINE,
    )

    def extract(self, source: str) -> ExtractionResult:
        """Extract formulas from Python assert statements."""
        formulas: List[SMTFormula] = []
        errors: List[str] = []

        for match in self._ASSERT_PATTERN.finditer(source):
            expression = match.group(1).strip()
            message = match.group(2)

            # Clean up the expression
            expression = self._clean_expression(expression)

            if expression:
                formula = SMTFormula(
                    expression=expression,
                    kind=FormulaKind.ASSERTION,
                    name=message,
                )
                formulas.append(formula)

        return ExtractionResult(formulas=formulas, source=source, errors=errors)

    def _clean_expression(self, expr: str) -> str:
        """Clean up an assertion expression."""
        # Remove trailing comments
        if "#" in expr:
            expr = expr.split("#")[0]
        return expr.strip()


class DocstringContractExtractor(FormulaExtractor):
    """Extract pre/post conditions from docstrings.

    Supports common docstring formats:
    - Google style: Args:/Returns: with :pre: and :post: markers
    - NumPy style: Parameters/Returns sections with Preconditions/Postconditions
    - Sphinx style: :pre: and :post: directives
    """

    # Patterns for contract markers
    # Note: Patterns are ordered from most specific to least specific
    # The (?<![:\w]) negative lookbehind prevents matching within longer words
    _PRECONDITION_PATTERNS = [
        re.compile(r':pre:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.MULTILINE),
        re.compile(r'(?<![:\w])Precondition:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.MULTILINE),
        re.compile(r'(?<![:\w])Requires:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.MULTILINE),
        # Pre: pattern - only match at word boundary, not after colon
        re.compile(r'(?<![:\w])Pre:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.MULTILINE),
    ]

    _POSTCONDITION_PATTERNS = [
        re.compile(r':post:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.MULTILINE),
        re.compile(r'(?<![:\w])Postcondition:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.MULTILINE),
        re.compile(r'(?<![:\w])Ensures:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.MULTILINE),
        # Post: pattern - only match at word boundary, not after colon
        re.compile(r'(?<![:\w])Post:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.MULTILINE),
    ]

    _INVARIANT_PATTERNS = [
        re.compile(r':invariant:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.MULTILINE),
        re.compile(r'(?<![:\w])Loop invariant:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.MULTILINE),
        re.compile(r'(?<![:\w])Invariant:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.MULTILINE),
    ]

    def extract(self, source: str) -> ExtractionResult:
        """Extract contracts from docstrings."""
        formulas: List[SMTFormula] = []
        errors: List[str] = []
        # Track matched positions to avoid duplicates
        matched_positions: Set[int] = set()

        # Extract preconditions
        for pattern in self._PRECONDITION_PATTERNS:
            for match in pattern.finditer(source):
                if match.start() in matched_positions:
                    continue
                expression = match.group(1).strip()
                if expression:
                    formula = SMTFormula(
                        expression=expression,
                        kind=FormulaKind.PRECONDITION,
                    )
                    formulas.append(formula)
                    matched_positions.add(match.start())

        # Extract postconditions
        for pattern in self._POSTCONDITION_PATTERNS:
            for match in pattern.finditer(source):
                if match.start() in matched_positions:
                    continue
                expression = match.group(1).strip()
                if expression:
                    formula = SMTFormula(
                        expression=expression,
                        kind=FormulaKind.POSTCONDITION,
                    )
                    formulas.append(formula)
                    matched_positions.add(match.start())

        # Extract invariants
        for pattern in self._INVARIANT_PATTERNS:
            for match in pattern.finditer(source):
                if match.start() in matched_positions:
                    continue
                expression = match.group(1).strip()
                if expression:
                    formula = SMTFormula(
                        expression=expression,
                        kind=FormulaKind.INVARIANT,
                    )
                    formulas.append(formula)
                    matched_positions.add(match.start())

        return ExtractionResult(formulas=formulas, source=source, errors=errors)


class TypeAnnotationExtractor(FormulaExtractor):
    """Extract constraints from type annotations.

    Handles annotations like:
    - x: int  # Implicit x is integer
    - x: Literal[1, 2, 3]  # x in {1, 2, 3}
    - x: Annotated[int, Gt(0)]  # x > 0
    """

    # Pattern for type annotations
    _ANNOTATION_PATTERN = re.compile(
        r'(\w+)\s*:\s*(\w+(?:\[.+?\])?)',
        re.MULTILINE,
    )

    # Patterns for constraint annotations
    _LITERAL_PATTERN = re.compile(r'Literal\[(.+?)\]')
    _GT_PATTERN = re.compile(r'Gt\((.+?)\)')
    _GE_PATTERN = re.compile(r'Ge\((.+?)\)')
    _LT_PATTERN = re.compile(r'Lt\((.+?)\)')
    _LE_PATTERN = re.compile(r'Le\((.+?)\)')

    def extract(self, source: str) -> ExtractionResult:
        """Extract constraints from type annotations."""
        formulas: List[SMTFormula] = []
        errors: List[str] = []

        for match in self._ANNOTATION_PATTERN.finditer(source):
            var_name = match.group(1)
            type_ann = match.group(2)

            # Check for Literal
            literal_match = self._LITERAL_PATTERN.search(type_ann)
            if literal_match:
                values = literal_match.group(1)
                # Create constraint: var in {values}
                formula = SMTFormula(
                    expression=f"{var_name} in ({values})",
                    kind=FormulaKind.ASSERTION,
                    name=f"{var_name}_literal",
                )
                formulas.append(formula)
                continue

            # Check for Gt, Ge, Lt, Le
            for pattern, op in [
                (self._GT_PATTERN, ">"),
                (self._GE_PATTERN, ">="),
                (self._LT_PATTERN, "<"),
                (self._LE_PATTERN, "<="),
            ]:
                constraint_match = pattern.search(type_ann)
                if constraint_match:
                    bound = constraint_match.group(1)
                    formula = SMTFormula(
                        expression=f"{var_name} {op} {bound}",
                        kind=FormulaKind.ASSERTION,
                        name=f"{var_name}_bound",
                    )
                    formulas.append(formula)

        return ExtractionResult(formulas=formulas, source=source, errors=errors)


class CompositeExtractor(FormulaExtractor):
    """Composite extractor that combines multiple extractors."""

    def __init__(self, extractors: Optional[List[FormulaExtractor]] = None):
        """Initialize with a list of extractors.

        Args:
            extractors: List of extractors to use
        """
        self._extractors = extractors or [
            PythonAssertExtractor(),
            DocstringContractExtractor(),
            TypeAnnotationExtractor(),
        ]

    def extract(self, source: str) -> ExtractionResult:
        """Extract formulas using all configured extractors."""
        all_formulas: List[SMTFormula] = []
        all_errors: List[str] = []

        for extractor in self._extractors:
            result = extractor.extract(source)
            all_formulas.extend(result.formulas)
            all_errors.extend(result.errors)

        return ExtractionResult(
            formulas=all_formulas,
            source=source,
            errors=all_errors,
        )


def extract_formulas(source: str) -> List[SMTFormula]:
    """Convenience function to extract formulas from source.

    Args:
        source: Source code text

    Returns:
        List of extracted SMTFormula instances
    """
    extractor = CompositeExtractor()
    result = extractor.extract(source)
    return result.formulas


def extract_assertions(source: str) -> List[SMTFormula]:
    """Extract only assertion formulas from source.

    Args:
        source: Source code text

    Returns:
        List of assertion SMTFormula instances
    """
    extractor = PythonAssertExtractor()
    result = extractor.extract(source)
    return result.formulas


def extract_contracts(source: str) -> List[SMTFormula]:
    """Extract only contract formulas from source.

    Args:
        source: Source code text

    Returns:
        List of contract SMTFormula instances
    """
    extractor = DocstringContractExtractor()
    result = extractor.extract(source)
    return result.formulas
