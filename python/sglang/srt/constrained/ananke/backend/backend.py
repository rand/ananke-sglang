# Copyright 2023-2024 SGLang Team
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
"""AnankeBackend: Multi-domain constrained decoding backend for SGLang.

This module implements AnankeBackend, which extends SGLang's BaseGrammarBackend
to support compositional constraint checking across multiple domains.

AnankeBackend creates AnankeGrammar instances that combine:
1. Syntax constraints via wrapped llguidance grammar
2. Type constraints via incremental bidirectional type checking
3. Import constraints via module resolution
4. Control flow constraints via CFG analysis
5. Semantic constraints via SMT solving

The backend integrates with SGLang's grammar backend registry, enabling
selection via --grammar-backend=ananke.

References:
    - Hazel: "Statically Contextualizing LLMs with Typed Holes" (OOPSLA 2024)
    - llguidance: Dynamic mask computation
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

from sglang.srt.constrained.base_grammar_backend import (
    INVALID_GRAMMAR_OBJ,
    BaseGrammarBackend,
    BaseGrammarObject,
    register_grammar_backend,
)

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from .grammar import AnankeGrammar
    from ..core.constraint import TOP, BOTTOM
    from ..core.domain import ConstraintDomain, PassthroughDomain
    from ..core.unified import UNIFIED_TOP, UnifiedConstraint
except ImportError:
    from backend.grammar import AnankeGrammar
    from core.constraint import TOP, BOTTOM
    from core.domain import ConstraintDomain, PassthroughDomain
    from core.unified import UNIFIED_TOP, UnifiedConstraint

logger = logging.getLogger(__name__)


class AnankeBackend(BaseGrammarBackend):
    """Multi-domain constrained decoding backend for SGLang.

    AnankeBackend orchestrates multiple constraint domains to provide
    comprehensive code validity checking during generation. It wraps
    llguidance for efficient syntax checking and adds type, import,
    control flow, and semantic constraints.

    The backend follows a delegation pattern:
    - Syntax constraints → llguidance (GuidanceBackend)
    - Additional domains → Ananke constraint system

    Attributes:
        tokenizer: The tokenizer for this model
        vocab_size: Size of the tokenizer vocabulary
        device: PyTorch device for tensor operations
        syntax_backend: Wrapped llguidance backend for syntax
        domains: Dictionary of active constraint domains
        language: Default programming language
    """

    def __init__(
        self,
        tokenizer: Any,
        vocab_size: int,
        model_eos_token_ids: Optional[List[int]] = None,
        any_whitespace: bool = True,
        whitespace_pattern: Optional[str] = None,
        language: str = "python",
        enabled_domains: Optional[Set[str]] = None,
        max_rollback_tokens: int = 200,
    ):
        """Initialize AnankeBackend.

        Args:
            tokenizer: The model's tokenizer
            vocab_size: Vocabulary size
            model_eos_token_ids: EOS token IDs for the model
            any_whitespace: Allow flexible whitespace in JSON schemas
            whitespace_pattern: Custom whitespace pattern for JSON
            language: Default programming language
            enabled_domains: Set of domain names to enable (default: all)
            max_rollback_tokens: Maximum tokens for rollback
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.model_eos_token_ids = model_eos_token_ids
        self.any_whitespace = any_whitespace
        self.whitespace_pattern = whitespace_pattern
        self.language = language
        self.max_rollback_tokens = max_rollback_tokens

        # Initialize syntax backend (llguidance)
        self.syntax_backend = self._create_syntax_backend()

        # Initialize constraint domains
        self.enabled_domains = enabled_domains or {
            "syntax",
            "types",
            "imports",
            "controlflow",
            "semantics",
        }
        self.domains = self._create_domains()

    def _create_syntax_backend(self) -> Optional[BaseGrammarBackend]:
        """Create the underlying llguidance backend for syntax constraints.

        Returns:
            GuidanceBackend instance or None if unavailable
        """
        try:
            from sglang.srt.constrained.llguidance_backend import GuidanceBackend

            return GuidanceBackend(
                tokenizer=self.tokenizer,
                any_whitespace=self.any_whitespace,
                whitespace_pattern=self.whitespace_pattern,
                n_vocab=self.vocab_size,
            )
        except ImportError:
            logger.warning(
                "llguidance not available, syntax constraints will be disabled"
            )
            return None

    def _create_domains(self) -> Dict[str, ConstraintDomain]:
        """Create constraint domain instances.

        For Phase 1, this creates passthrough domains that impose no
        constraints. Full domain implementations will be added in later phases.

        Returns:
            Dictionary mapping domain names to domain instances
        """
        domains: Dict[str, ConstraintDomain] = {}

        # Types domain - Phase 3 will implement full type checking
        if "types" in self.enabled_domains:
            domains["types"] = PassthroughDomain(
                domain_name="types",
                top_constraint=TOP,
                bottom_constraint=BOTTOM,
            )

        # Imports domain - Phase 7 will implement import resolution
        if "imports" in self.enabled_domains:
            domains["imports"] = PassthroughDomain(
                domain_name="imports",
                top_constraint=TOP,
                bottom_constraint=BOTTOM,
            )

        # Control flow domain - Phase 8 will implement CFG analysis
        if "controlflow" in self.enabled_domains:
            domains["controlflow"] = PassthroughDomain(
                domain_name="controlflow",
                top_constraint=TOP,
                bottom_constraint=BOTTOM,
            )

        # Semantics domain - Phase 9 will implement SMT solving
        if "semantics" in self.enabled_domains:
            domains["semantics"] = PassthroughDomain(
                domain_name="semantics",
                top_constraint=TOP,
                bottom_constraint=BOTTOM,
            )

        return domains

    def dispatch_json(self, key_string: str) -> Optional[AnankeGrammar]:
        """Dispatch JSON schema constraint.

        Creates an AnankeGrammar that enforces the JSON schema via
        syntax constraints, with additional domain constraints applied.

        Args:
            key_string: JSON schema string

        Returns:
            AnankeGrammar instance or INVALID_GRAMMAR_OBJ on error
        """
        # Get syntax grammar from llguidance
        syntax_grammar = None
        if self.syntax_backend is not None:
            syntax_grammar = self.syntax_backend.dispatch_json(key_string)
            if syntax_grammar is INVALID_GRAMMAR_OBJ:
                return INVALID_GRAMMAR_OBJ

        return self._create_ananke_grammar(syntax_grammar)

    def dispatch_regex(self, key_string: str) -> Optional[AnankeGrammar]:
        """Dispatch regex constraint.

        Args:
            key_string: Regex pattern string

        Returns:
            AnankeGrammar instance or INVALID_GRAMMAR_OBJ on error
        """
        syntax_grammar = None
        if self.syntax_backend is not None:
            syntax_grammar = self.syntax_backend.dispatch_regex(key_string)
            if syntax_grammar is INVALID_GRAMMAR_OBJ:
                return INVALID_GRAMMAR_OBJ

        return self._create_ananke_grammar(syntax_grammar)

    def dispatch_ebnf(self, key_string: str) -> Optional[AnankeGrammar]:
        """Dispatch EBNF grammar constraint.

        Args:
            key_string: EBNF grammar string

        Returns:
            AnankeGrammar instance or INVALID_GRAMMAR_OBJ on error
        """
        syntax_grammar = None
        if self.syntax_backend is not None:
            syntax_grammar = self.syntax_backend.dispatch_ebnf(key_string)
            if syntax_grammar is INVALID_GRAMMAR_OBJ:
                return INVALID_GRAMMAR_OBJ

        return self._create_ananke_grammar(syntax_grammar)

    def dispatch_structural_tag(self, key_string: str) -> Optional[AnankeGrammar]:
        """Dispatch structural tag constraint.

        Args:
            key_string: Structural tag JSON string

        Returns:
            AnankeGrammar instance or INVALID_GRAMMAR_OBJ on error
        """
        syntax_grammar = None
        if self.syntax_backend is not None:
            syntax_grammar = self.syntax_backend.dispatch_structural_tag(key_string)
            if syntax_grammar is INVALID_GRAMMAR_OBJ:
                return INVALID_GRAMMAR_OBJ

        return self._create_ananke_grammar(syntax_grammar)

    def _create_ananke_grammar(
        self,
        syntax_grammar: Optional[BaseGrammarObject],
    ) -> AnankeGrammar:
        """Create an AnankeGrammar instance with the given syntax grammar.

        Args:
            syntax_grammar: Underlying syntax grammar (llguidance)

        Returns:
            Configured AnankeGrammar instance
        """
        return AnankeGrammar(
            syntax_grammar=syntax_grammar,
            domains=self.domains,
            constraint=UNIFIED_TOP,
            vocab_size=self.vocab_size,
            device="cuda",
            tokenizer=self.tokenizer,
            language=self.language,
            max_rollback_tokens=self.max_rollback_tokens,
        )


def create_ananke_backend(
    server_args: Any,
    tokenizer: Any,
    vocab_size: int,
    eos_token_ids: Optional[Set[int]] = None,
) -> AnankeBackend:
    """Factory function to create AnankeBackend from server args.

    This function is registered with SGLang's grammar backend registry
    to enable selection via --grammar-backend=ananke.

    Args:
        server_args: SGLang server arguments
        tokenizer: Model tokenizer
        vocab_size: Vocabulary size
        eos_token_ids: Set of EOS token IDs

    Returns:
        Configured AnankeBackend instance
    """
    eos_list = list(eos_token_ids) if eos_token_ids else None

    return AnankeBackend(
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        model_eos_token_ids=eos_list,
        any_whitespace=not getattr(
            server_args, "constrained_json_disable_any_whitespace", False
        ),
        whitespace_pattern=getattr(
            server_args, "constrained_json_whitespace_pattern", None
        ),
        language=getattr(server_args, "ananke_language", "python"),
        max_rollback_tokens=getattr(server_args, "ananke_max_rollback_tokens", 200),
    )


# Register with SGLang's grammar backend registry
register_grammar_backend("ananke", create_ananke_backend)
