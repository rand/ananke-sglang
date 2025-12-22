# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Cross-language constraint examples.

These examples demonstrate the same workflow implemented across all 7 languages,
showing how ConstraintSpec adapts to different language idioms.
"""

from __future__ import annotations

from .api_response import CROSS_LANGUAGE_API_EXAMPLES
from .error_handling import CROSS_LANGUAGE_ERROR_EXAMPLES

ALL_CROSS_LANGUAGE_EXAMPLES = (
    CROSS_LANGUAGE_API_EXAMPLES + CROSS_LANGUAGE_ERROR_EXAMPLES
)

__all__ = [
    "CROSS_LANGUAGE_API_EXAMPLES",
    "CROSS_LANGUAGE_ERROR_EXAMPLES",
    "ALL_CROSS_LANGUAGE_EXAMPLES",
]
