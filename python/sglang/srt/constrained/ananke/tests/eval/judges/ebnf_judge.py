# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""EBNF-based judge for syntax constraint satisfaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..metrics import EvalResult, SatisfactionLevel


@dataclass
class EbnfMatch:
    """Details of an EBNF grammar match."""

    matched: bool
    grammar: str
    parse_tree: Optional[str] = None
    error: Optional[str] = None


# Check for required dependencies
try:
    import lark
    from llguidance.gbnf_to_lark import any_to_lark

    HAS_EBNF_SUPPORT = True
except ImportError:
    HAS_EBNF_SUPPORT = False
    lark = None  # type: ignore
    any_to_lark = None  # type: ignore


class EbnfJudge:
    """Judge for EBNF-based syntax constraints.

    This judge evaluates whether generated output matches the EBNF grammar
    specified in a ConstraintSpec. It uses:
    - llguidance's gbnf_to_lark for grammar conversion
    - Lark parser for string matching

    The EBNF format supported is GBNF (::= syntax) which is commonly used
    in constrained generation systems.
    """

    def __init__(self):
        """Initialize the EBNF judge.

        Raises:
            ImportError: If lark or llguidance are not available
        """
        if not HAS_EBNF_SUPPORT:
            raise ImportError(
                "EBNF judge requires 'lark' and 'llguidance' packages. "
                "Install with: pip install lark llguidance"
            )

    def _normalize_gbnf(self, ebnf: str) -> str:
        """Normalize GBNF grammar for llguidance parsing.

        Handles:
        - Multi-line rules with continuation (leading whitespace before |)
        - Single-quoted terminals at start of rule RHS -> double-quoted
        - Extra whitespace between tokens

        Args:
            ebnf: Raw GBNF grammar

        Returns:
            Normalized GBNF grammar
        """
        import re as regex_module

        # Convert single-quoted terminals to double-quoted, but ONLY when they
        # are used as terminal delimiters (not inside double-quoted strings).
        # Strategy: Only replace 'content' when preceded by ::= or | or whitespace
        # and followed by whitespace or end of line (i.e., at terminal position)
        def replace_single_quotes(match: "regex_module.Match") -> str:
            content = match.group(1)
            # Escape any double quotes in content
            escaped = content.replace('"', '\\"')
            return f'"{escaped}"'

        # Match single-quoted strings that appear as standalone terminals
        # (after ::= or | or at start of line, with whitespace or EOL after)
        # This pattern looks for 'content' NOT inside double quotes
        # We use a simpler approach: only replace if the quote starts a token
        ebnf = regex_module.sub(
            r"(?<=::=\s)'([^']*)'" r"|" r"(?<=\|\s)'([^']*)'",
            lambda m: f'"{m.group(1) or m.group(2)}"',
            ebnf,
        )

        lines = ebnf.strip().split("\n")
        normalized_lines = []
        current_rule = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if this is a continuation line (starts with |)
            if stripped.startswith("|"):
                # Append to current rule
                if current_rule:
                    current_rule.append(" " + stripped)
                else:
                    # Orphan continuation - skip or error
                    normalized_lines.append(stripped)
            elif "::=" in stripped:
                # New rule - flush current if any
                if current_rule:
                    normalized_lines.append("".join(current_rule))
                current_rule = [stripped]
            else:
                # Continuation of current rule (no ::= or |)
                if current_rule:
                    current_rule.append(" " + stripped)
                else:
                    normalized_lines.append(stripped)

        # Flush final rule
        if current_rule:
            normalized_lines.append("".join(current_rule))

        return "\n".join(normalized_lines)

    def _convert_to_lark(self, ebnf: str) -> str:
        """Convert EBNF (GBNF format) to Lark grammar.

        Args:
            ebnf: EBNF grammar in GBNF format (::= syntax)

        Returns:
            Lark grammar string
        """
        import re as regex_module

        # First normalize the grammar
        normalized = self._normalize_gbnf(ebnf)
        lark_grammar = any_to_lark(normalized)
        # Remove llguidance directive which Lark doesn't understand
        lark_grammar = lark_grammar.replace("%llguidance {}", "").strip()

        # Fix specific zero-width terminal patterns that Lark rejects.
        # Only fix truly problematic cases where a terminal can match empty string:
        # - /[\s]/* or similar character-class-star patterns
        # - WS: '(?:\ )*' type patterns that can be empty
        #
        # Don't touch group patterns like (...)* which are usually fine.

        # Fix zero-width character class patterns: /[...]/*, /[...]/?
        # Replace /x/* with /x/+ where x is a simple character class
        lark_grammar = regex_module.sub(
            r"/(\[[^\]]+\])/\*",  # Match /[...]/
            r"/\1/+",  # Replace with /[...]/+
            lark_grammar,
        )

        # Fix Lark-specific zero-width patterns: '(?:\ )*' etc.
        # These are generated by llguidance for optional whitespace
        lark_grammar = regex_module.sub(
            r"'\(\?:[^)]+\)\*'",  # Match '(?:...)*'
            r"' '",  # Replace with single space (safe approximation)
            lark_grammar,
        )

        return lark_grammar

    def _create_parser(self, lark_grammar: str) -> "lark.Lark":
        """Create a Lark parser from grammar.

        Args:
            lark_grammar: Lark format grammar

        Returns:
            Lark parser instance
        """
        # Try LALR first (handles zero-width regexes, faster)
        # Fall back to Earley if LALR can't handle the grammar
        try:
            return lark.Lark(lark_grammar, start="start", parser="lalr")
        except lark.exceptions.GrammarError:
            # LALR can't handle all grammars, fall back to Earley
            return lark.Lark(lark_grammar, start="start")

    def matches(self, output: str, ebnf: str) -> bool:
        """Simple check if output matches EBNF grammar.

        Args:
            output: Text to check
            ebnf: EBNF grammar

        Returns:
            True if output parses successfully
        """
        try:
            lark_grammar = self._convert_to_lark(ebnf)
            parser = self._create_parser(lark_grammar)
            parser.parse(output)
            return True
        except Exception:
            return False

    def evaluate(
        self,
        example_id: str,
        output: str,
        ebnf: str,
    ) -> EvalResult:
        """Evaluate whether output matches the EBNF grammar.

        Args:
            example_id: Unique identifier for the example
            output: Generated output to evaluate
            ebnf: EBNF grammar that should match

        Returns:
            EvalResult with satisfaction status and details
        """
        # Convert EBNF to Lark
        try:
            lark_grammar = self._convert_to_lark(ebnf)
        except Exception as e:
            return EvalResult(
                example_id=example_id,
                satisfied=False,
                satisfaction_level=SatisfactionLevel.ERROR,
                output=output,
                error=f"EBNF conversion error: {e}",
                metadata={"ebnf": ebnf[:200]},
            )

        # Create parser
        try:
            parser = self._create_parser(lark_grammar)
        except lark.exceptions.LarkError as e:
            return EvalResult(
                example_id=example_id,
                satisfied=False,
                satisfaction_level=SatisfactionLevel.ERROR,
                output=output,
                error=f"Grammar parse error: {e}",
                metadata={"lark_grammar": lark_grammar[:200]},
            )

        # Try to parse the output
        try:
            parse_tree = parser.parse(output)
            return EvalResult(
                example_id=example_id,
                satisfied=True,
                satisfaction_level=SatisfactionLevel.FULL,
                output=output,
                metadata={
                    "ebnf": ebnf[:200],
                    "parse_tree": str(parse_tree)[:200] if parse_tree else None,
                },
            )
        except lark.exceptions.LarkError as e:
            return EvalResult(
                example_id=example_id,
                satisfied=False,
                satisfaction_level=SatisfactionLevel.NONE,
                output=output,
                error=f"Parse failed: {str(e)[:200]}",
                metadata={"ebnf": ebnf[:200]},
            )

    def validate_grammar(self, ebnf: str) -> Tuple[bool, Optional[str]]:
        """Validate that an EBNF grammar is well-formed.

        Args:
            ebnf: EBNF grammar to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            lark_grammar = self._convert_to_lark(ebnf)
            self._create_parser(lark_grammar)
            return True, None
        except Exception as e:
            return False, str(e)

    def evaluate_valid_outputs(
        self,
        example_id: str,
        valid_outputs: List[str],
        ebnf: str,
    ) -> Tuple[int, int, List[EvalResult]]:
        """Evaluate a list of valid outputs against the EBNF grammar.

        Args:
            example_id: Base example ID
            valid_outputs: List of known-valid outputs
            ebnf: EBNF grammar that should match

        Returns:
            Tuple of (passed_count, total_count, results)
        """
        results = []
        passed = 0

        for i, output in enumerate(valid_outputs):
            result = self.evaluate(
                example_id=f"{example_id}-valid-{i}",
                output=output,
                ebnf=ebnf,
            )
            results.append(result)
            if result.satisfied:
                passed += 1

        return passed, len(valid_outputs), results

    def evaluate_invalid_outputs(
        self,
        example_id: str,
        invalid_outputs: List[str],
        ebnf: str,
    ) -> Tuple[int, int, List[EvalResult]]:
        """Evaluate a list of invalid outputs against the EBNF grammar.

        Args:
            example_id: Base example ID
            invalid_outputs: List of known-invalid outputs
            ebnf: EBNF grammar

        Returns:
            Tuple of (correctly_rejected_count, total_count, results)
        """
        results = []
        rejected = 0

        for i, output in enumerate(invalid_outputs):
            result = self.evaluate(
                example_id=f"{example_id}-invalid-{i}",
                output=output,
                ebnf=ebnf,
            )
            results.append(result)
            # Invalid outputs SHOULD NOT satisfy the grammar
            if not result.satisfied:
                rejected += 1

        return rejected, len(invalid_outputs), results
