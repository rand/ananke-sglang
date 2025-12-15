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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from sglang.srt.constrained.base_grammar_backend import (
    INVALID_GRAMMAR_OBJ,
    BaseGrammarBackend,
    BaseGrammarObject,
    register_grammar_backend,
)

if TYPE_CHECKING:
    from ..spec.constraint_spec import ConstraintSpec

# Support both relative imports (when used as subpackage) and absolute imports (standalone testing)
try:
    from .grammar import AnankeGrammar
    from ..core.constraint import TOP, BOTTOM
    from ..core.domain import ConstraintDomain, PassthroughDomain
    from ..core.unified import UNIFIED_TOP, UnifiedConstraint
    from ..domains.types.domain import TypeDomain
    from ..domains.imports.domain import ImportDomain
    from ..domains.controlflow.domain import ControlFlowDomain
    from ..domains.semantics.domain import SemanticDomain
except ImportError:
    from backend.grammar import AnankeGrammar
    from core.constraint import TOP, BOTTOM
    from core.domain import ConstraintDomain, PassthroughDomain
    from core.unified import UNIFIED_TOP, UnifiedConstraint
    from domains.types.domain import TypeDomain
    from domains.imports.domain import ImportDomain
    from domains.controlflow.domain import ControlFlowDomain
    from domains.semantics.domain import SemanticDomain

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

        Creates full domain implementations for each enabled domain:
        - TypeDomain: Incremental bidirectional type checking (Hazel-inspired)
        - ImportDomain: Module/package constraint tracking
        - ControlFlowDomain: CFG-based reachability analysis
        - SemanticDomain: SMT-based constraint checking

        Returns:
            Dictionary mapping domain names to domain instances
        """
        domains: Dict[str, ConstraintDomain] = {}

        # Types domain - Incremental bidirectional type checking
        if "types" in self.enabled_domains:
            domains["types"] = TypeDomain(language=self.language)

        # Imports domain - Module/package constraint tracking
        if "imports" in self.enabled_domains:
            domains["imports"] = ImportDomain(language=self.language)

        # Control flow domain - CFG-based reachability analysis
        if "controlflow" in self.enabled_domains:
            domains["controlflow"] = ControlFlowDomain(language=self.language)

        # Semantics domain - SMT-based constraint checking
        if "semantics" in self.enabled_domains:
            domains["semantics"] = SemanticDomain(language=self.language)

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
        constraint_spec: Optional["ConstraintSpec"] = None,
    ) -> AnankeGrammar:
        """Create an AnankeGrammar instance with the given syntax grammar.

        Args:
            syntax_grammar: Underlying syntax grammar (llguidance)
            constraint_spec: Optional rich constraint specification

        Returns:
            Configured AnankeGrammar instance
        """
        # Determine effective language
        language = self.language
        if constraint_spec is not None and constraint_spec.language:
            language = constraint_spec.language

        return AnankeGrammar(
            syntax_grammar=syntax_grammar,
            domains=self.domains,
            constraint=UNIFIED_TOP,
            vocab_size=self.vocab_size,
            device="cuda",
            tokenizer=self.tokenizer,
            language=language,
            max_rollback_tokens=self.max_rollback_tokens,
            constraint_spec=constraint_spec,
        )

    def dispatch_with_spec(
        self,
        spec: "ConstraintSpec",
    ) -> Optional[AnankeGrammar]:
        """Create AnankeGrammar with full constraint specification.

        This is the primary dispatch method for rich constraint specs,
        enabling:
        - Per-request language configuration
        - Type environment context
        - Import context
        - Control flow context
        - Semantic constraints

        The method:
        1. Resolves the effective language from spec and backend defaults
        2. Creates syntax grammar from the core constraint
        3. Creates domains with context seeded from the spec
        4. Returns configured AnankeGrammar

        Args:
            spec: Rich constraint specification

        Returns:
            AnankeGrammar instance or INVALID_GRAMMAR_OBJ on error
        """
        # 1. Resolve effective language
        effective_language = self._resolve_language(spec)

        # 2. Create syntax grammar from core constraint
        syntax_grammar = self._create_syntax_grammar_from_spec(spec)
        if syntax_grammar is INVALID_GRAMMAR_OBJ:
            return INVALID_GRAMMAR_OBJ

        # 3. Create domains with context from spec
        domains = self._create_domains_with_spec(spec, effective_language)

        # 4. Create and return grammar with spec
        return AnankeGrammar(
            syntax_grammar=syntax_grammar,
            domains=domains,
            constraint=UNIFIED_TOP,
            vocab_size=self.vocab_size,
            device="cuda",
            tokenizer=self.tokenizer,
            language=effective_language,
            max_rollback_tokens=self.max_rollback_tokens,
            constraint_spec=spec,
        )

    def _resolve_language(self, spec: "ConstraintSpec") -> str:
        """Resolve effective language from spec and backend defaults.

        Args:
            spec: Constraint specification

        Returns:
            Effective language string
        """
        # Import LanguageDetection enum
        try:
            from ..spec.constraint_spec import LanguageDetection
        except ImportError:
            from spec.constraint_spec import LanguageDetection

        if spec.language:
            return spec.language
        if spec.language_detection == LanguageDetection.AUTO:
            # Will be determined during generation via tree-sitter
            # Start with backend default
            return self.language
        return self.language

    def _create_syntax_grammar_from_spec(
        self,
        spec: "ConstraintSpec",
    ) -> Optional[BaseGrammarObject]:
        """Create syntax grammar from constraint spec.

        Args:
            spec: Constraint specification

        Returns:
            Syntax grammar or INVALID_GRAMMAR_OBJ on error
        """
        if self.syntax_backend is None:
            return None

        if spec.json_schema is not None:
            return self.syntax_backend.dispatch_json(spec.json_schema)
        elif spec.regex is not None:
            return self.syntax_backend.dispatch_regex(spec.regex)
        elif spec.ebnf is not None:
            return self.syntax_backend.dispatch_ebnf(spec.ebnf)
        elif spec.structural_tag is not None:
            return self.syntax_backend.dispatch_structural_tag(spec.structural_tag)

        return None

    def _create_domains_with_spec(
        self,
        spec: "ConstraintSpec",
        language: str,
    ) -> Dict[str, ConstraintDomain]:
        """Create domains initialized with context from spec.

        Args:
            spec: Constraint specification
            language: Effective language

        Returns:
            Dictionary of initialized domains
        """
        # Compute which domains to enable
        enabled = self._compute_enabled_domains(spec)
        domains: Dict[str, ConstraintDomain] = {}

        # Create each enabled domain with context
        if "types" in enabled:
            domains["types"] = self._create_type_domain_with_spec(spec, language)

        if "imports" in enabled:
            domains["imports"] = self._create_import_domain_with_spec(spec, language)

        if "controlflow" in enabled:
            domains["controlflow"] = self._create_cf_domain_with_spec(spec, language)

        if "semantics" in enabled:
            domains["semantics"] = self._create_semantic_domain_with_spec(spec, language)

        return domains

    def _compute_enabled_domains(self, spec: "ConstraintSpec") -> Set[str]:
        """Compute which domains to enable based on spec and backend config.

        Args:
            spec: Constraint specification

        Returns:
            Set of enabled domain names
        """
        # Start with backend defaults
        enabled = set(self.enabled_domains)

        # Apply spec overrides
        if spec.enabled_domains is not None:
            enabled = spec.enabled_domains
        if spec.disabled_domains is not None:
            enabled = enabled - spec.disabled_domains

        return enabled

    def _create_type_domain_with_spec(
        self,
        spec: "ConstraintSpec",
        language: str,
    ) -> TypeDomain:
        """Create TypeDomain seeded with type context from spec.

        Args:
            spec: Constraint specification
            language: Effective language

        Returns:
            Configured TypeDomain instance
        """
        domain = TypeDomain(language=language)

        # Use inject_context if available (preferred path)
        if hasattr(domain, "inject_context"):
            domain.inject_context(spec)
            return domain

        # Fallback: seed manually using individual methods
        # Seed type bindings (parse type expressions)
        for binding in spec.type_bindings:
            if hasattr(domain, "_parse_type_expr"):
                parsed_type = domain._parse_type_expr(binding.type_expr)
                domain.bind_variable(binding.name, parsed_type)

        # Seed function signatures
        for sig in spec.function_signatures:
            if hasattr(domain, "register_function") and hasattr(domain, "_parse_type_expr"):
                params = [
                    (p.name, domain._parse_type_expr(p.type_expr))
                    for p in sig.params
                ]
                ret_type = domain._parse_type_expr(sig.return_type)
                domain.register_function(
                    name=sig.name,
                    params=params,
                    return_type=ret_type,
                    type_params=list(sig.type_params) if sig.type_params else None,
                    is_async=sig.is_async,
                    is_generator=sig.is_generator,
                )

        # Seed class definitions
        for cls_def in spec.class_definitions:
            if hasattr(domain, "register_class"):
                domain.register_class(
                    name=cls_def.name,
                    bases=list(cls_def.bases) if cls_def.bases else None,
                    type_params=list(cls_def.type_params) if cls_def.type_params else None,
                )

        # Set expected type
        if spec.expected_type and hasattr(domain, "_parse_type_expr"):
            parsed_type = domain._parse_type_expr(spec.expected_type)
            domain.set_expected_type(parsed_type)

        # Apply type aliases
        for alias, target in spec.type_aliases.items():
            if hasattr(domain, "_parse_type_expr") and hasattr(domain, "add_type_alias"):
                parsed_type = domain._parse_type_expr(target)
                domain.add_type_alias(alias, parsed_type)

        return domain

    def _create_import_domain_with_spec(
        self,
        spec: "ConstraintSpec",
        language: str,
    ) -> ImportDomain:
        """Create ImportDomain seeded with import context from spec.

        Args:
            spec: Constraint specification
            language: Effective language

        Returns:
            Configured ImportDomain instance
        """
        domain = ImportDomain(language=language)

        # Use inject_context if available (preferred path)
        if hasattr(domain, "inject_context"):
            domain.inject_context(spec)
            return domain

        # Fallback: seed manually using individual methods
        # Seed imports
        for imp in spec.imports:
            if hasattr(domain, "add_import"):
                domain.add_import(
                    module=imp.module,
                    name=imp.name,
                    alias=imp.alias,
                    is_wildcard=imp.is_wildcard,
                )

        # Set available modules
        if spec.available_modules and hasattr(domain, "set_available_modules"):
            domain.set_available_modules(spec.available_modules)

        # Note: set_forbidden_imports returns a constraint, not modifies domain
        # The forbidden imports are handled via constraint creation
        # if spec.forbidden_imports and hasattr(domain, "set_forbidden_imports"):
        #     domain.set_forbidden_imports(spec.forbidden_imports)

        return domain

    def _create_cf_domain_with_spec(
        self,
        spec: "ConstraintSpec",
        language: str,
    ) -> ControlFlowDomain:
        """Create ControlFlowDomain seeded with control flow context from spec.

        Args:
            spec: Constraint specification
            language: Effective language

        Returns:
            Configured ControlFlowDomain instance
        """
        domain = ControlFlowDomain(language=language)

        # Use inject_context if available (preferred path)
        if hasattr(domain, "inject_context"):
            domain.inject_context(spec)
            return domain

        # Fallback: seed manually using individual methods
        cf_ctx = spec.control_flow
        if cf_ctx is None:
            return domain

        # Set function context if provided
        if cf_ctx.function_name and hasattr(domain, "set_function_context"):
            domain.set_function_context(
                function_name=cf_ctx.function_name,
                expected_return_type=cf_ctx.expected_return_type,
                is_async=cf_ctx.in_async_context,
                is_generator=cf_ctx.in_generator,
            )

        # Set loop context if in loop
        if cf_ctx.loop_depth > 0 and hasattr(domain, "set_loop_context"):
            domain.set_loop_context(
                loop_depth=cf_ctx.loop_depth,
                loop_variables=list(cf_ctx.loop_variables) if cf_ctx.loop_variables else None,
            )

        # Set try context if in try block
        if cf_ctx.in_try_block and hasattr(domain, "set_try_context"):
            domain.set_try_context(
                in_try_block=True,
                exception_types=list(cf_ctx.exception_types) if cf_ctx.exception_types else None,
            )

        return domain

    def _create_semantic_domain_with_spec(
        self,
        spec: "ConstraintSpec",
        language: str,
    ) -> SemanticDomain:
        """Create SemanticDomain seeded with semantic constraints from spec.

        Args:
            spec: Constraint specification
            language: Effective language

        Returns:
            Configured SemanticDomain instance
        """
        domain = SemanticDomain(language=language)

        # Use inject_context if available (preferred path)
        if hasattr(domain, "inject_context"):
            domain.inject_context(spec)
            return domain

        # Fallback: seed manually using individual methods
        for constraint in spec.semantic_constraints:
            if hasattr(domain, "add_semantic_constraint"):
                domain.add_semantic_constraint(
                    kind=constraint.kind,
                    expression=constraint.expression,
                    scope=constraint.scope,
                    variables=list(constraint.variables) if constraint.variables else None,
                )

        return domain


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

    # Parse enabled_domains from comma-separated string to set
    enabled_domains_str = getattr(server_args, "ananke_enabled_domains", None)
    enabled_domains: Optional[Set[str]] = None
    if enabled_domains_str:
        enabled_domains = {d.strip() for d in enabled_domains_str.split(",")}
        # Validate domain names
        valid_domains = {"syntax", "types", "imports", "controlflow", "semantics"}
        invalid = enabled_domains - valid_domains
        if invalid:
            logger.warning(
                f"Unknown Ananke domains ignored: {invalid}. "
                f"Valid domains: {valid_domains}"
            )
            enabled_domains = enabled_domains & valid_domains

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
        enabled_domains=enabled_domains,
        max_rollback_tokens=getattr(server_args, "ananke_max_rollback_tokens", 200),
    )


# Register with SGLang's grammar backend registry
register_grammar_backend("ananke", create_ananke_backend)
