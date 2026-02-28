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
"""Pytest configuration for Ananke tests.

This conftest.py sets up the import paths so tests can run either:
1. Standalone (with ananke as the root package)
2. As part of the full sglang test suite
"""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

# Add the ananke directory to path so we can import core, domains, etc. directly
ananke_dir = Path(__file__).parent.parent
if str(ananke_dir) not in sys.path:
    sys.path.insert(0, str(ananke_dir))

# Also add the sglang python directory for full integration tests
sglang_python_dir = ananke_dir.parent.parent.parent.parent.parent
if str(sglang_python_dir) not in sys.path:
    sys.path.insert(0, str(sglang_python_dir))


def _create_mock_sglang():
    """Create mock sglang modules for standalone testing.

    This creates a minimal mock of the SGLang grammar backend interfaces
    so that Ananke can be tested without the full SGLang installation.
    """

    # Create BaseGrammarObject mock class
    class BaseGrammarObject:
        """Mock of SGLang's BaseGrammarObject for standalone testing."""

        def __init__(self):
            self.finished = False

        def accept_token(self, token: int) -> None:
            raise NotImplementedError

        def fill_vocab_mask(self, vocab_mask, idx: int) -> None:
            raise NotImplementedError

        def allocate_vocab_mask(self, vocab_size: int, batch_size: int, device):
            raise NotImplementedError

        @staticmethod
        def move_vocab_mask(vocab_mask, device):
            raise NotImplementedError

        @staticmethod
        def apply_vocab_mask(logits, vocab_mask) -> None:
            raise NotImplementedError

        def is_terminated(self) -> bool:
            raise NotImplementedError

        def copy(self):
            raise NotImplementedError

        def try_jump_forward(self, tokenizer):
            return None

        def jump_forward_str_state(self, helper):
            return "", -1

        def jump_and_retokenize(self, old_output_ids, new_output_ids, next_state):
            pass

        def rollback(self, k: int) -> None:
            pass

    # Create BaseGrammarBackend mock class
    class BaseGrammarBackend:
        """Mock of SGLang's BaseGrammarBackend for standalone testing."""

        def __init__(self):
            pass

        def dispatch_json(self, key_string: str):
            raise NotImplementedError

        def dispatch_regex(self, key_string: str):
            raise NotImplementedError

        def dispatch_ebnf(self, key_string: str):
            raise NotImplementedError

        def dispatch_structural_tag(self, key_string: str):
            raise NotImplementedError

    # INVALID_GRAMMAR_OBJ sentinel
    INVALID_GRAMMAR_OBJ = object()

    # Create register_grammar_backend function
    _registry = {}

    def register_grammar_backend(name: str, factory):
        _registry[name] = factory

    # Create module structure
    sglang_module = ModuleType("sglang")
    srt_module = ModuleType("sglang.srt")
    constrained_module = ModuleType("sglang.srt.constrained")
    base_grammar_backend_module = ModuleType("sglang.srt.constrained.base_grammar_backend")

    # Set up attributes
    base_grammar_backend_module.BaseGrammarObject = BaseGrammarObject
    base_grammar_backend_module.BaseGrammarBackend = BaseGrammarBackend
    base_grammar_backend_module.INVALID_GRAMMAR_OBJ = INVALID_GRAMMAR_OBJ
    base_grammar_backend_module.register_grammar_backend = register_grammar_backend

    # Link modules
    constrained_module.base_grammar_backend = base_grammar_backend_module
    srt_module.constrained = constrained_module
    sglang_module.srt = srt_module

    # Register in sys.modules
    sys.modules["sglang"] = sglang_module
    sys.modules["sglang.srt"] = srt_module
    sys.modules["sglang.srt.constrained"] = constrained_module
    sys.modules["sglang.srt.constrained.base_grammar_backend"] = base_grammar_backend_module


# Try to import sglang, if it fails, create mock
try:
    import sglang.srt.constrained.base_grammar_backend
except ImportError:
    _create_mock_sglang()
