"""Debug the constraint_spec grammar path.

This script directly tests the grammar creation and dispatch paths
to identify where constraint_spec is breaking.
"""

import modal
import os

DEPLOYED_APP = "qwen3-coder-ananke"
DEPLOYED_CLASS = "Qwen3CoderAnanke"

app = modal.App("debug-constraint-path")

# Build same image as deployment
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "curl", "wget", "build-essential", "cmake")
    .run_commands("pip install --upgrade pip setuptools wheel")
    .add_local_dir(
        os.path.join(REPO_ROOT, "python"),
        remote_path="/sglang/python",
        copy=True,
    )
    .add_local_file(
        os.path.join(REPO_ROOT, "README.md"),
        remote_path="/sglang/python/README.md",
        copy=True,
    )
    .run_commands("cd /sglang/python && pip install -e '.[srt]'")
    .pip_install("z3-solver>=4.12.0", "tree-sitter>=0.22.0", "immutables>=0.20")
)


@app.function(timeout=300, image=image, gpu="T4")
def debug_grammar_dispatch():
    """Debug the grammar dispatch paths."""
    import sys
    sys.path.insert(0, "/sglang/python")

    print("=" * 60)
    print("DEBUG: GRAMMAR DISPATCH PATHS")
    print("=" * 60)

    # 1. Check imports
    print("\n1. Checking imports...")
    try:
        from sglang.srt.constrained.ananke.spec.constraint_spec import ConstraintSpec
        print("   ConstraintSpec: IMPORTED ✓")
    except ImportError as e:
        print(f"   ConstraintSpec: IMPORT FAILED - {e}")
        return "FAILED at import"

    try:
        from sglang.srt.constrained.ananke.backend.backend import AnankeBackend
        print("   AnankeBackend: IMPORTED ✓")
    except ImportError as e:
        print(f"   AnankeBackend: IMPORT FAILED - {e}")
        return "FAILED at import"

    try:
        from sglang.srt.constrained.llguidance_backend import GuidanceBackend
        print("   GuidanceBackend: IMPORTED ✓")
    except ImportError as e:
        print(f"   GuidanceBackend: IMPORT FAILED - {e}")
        return "FAILED at import"

    # 2. Create mock tokenizer
    print("\n2. Creating mock tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"   Tokenizer loaded: {type(tokenizer).__name__}")

    # 3. Create AnankeBackend
    print("\n3. Creating AnankeBackend...")
    try:
        backend = AnankeBackend(
            tokenizer=tokenizer,
            vocab_size=50257,  # GPT-2 vocab size
            model_eos_token_ids=[50256],
            any_whitespace=True,
            whitespace_pattern=None,
            language="python",
            max_rollback_tokens=200,
        )
        print(f"   Backend created: {type(backend).__name__}")
        print(f"   syntax_backend: {backend.syntax_backend}")
        print(f"   enabled_domains: {backend.enabled_domains}")
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()
        return "FAILED at backend creation"

    # 4. Test dispatch_regex (legacy path)
    print("\n4. Testing dispatch_regex (legacy path)...")
    try:
        grammar1 = backend.dispatch_regex("[0-9]+")
        print(f"   Result type: {type(grammar1).__name__}")
        print(f"   syntax_grammar: {grammar1.syntax_grammar}")
        print(f"   Has fill_vocab_mask: {hasattr(grammar1, 'fill_vocab_mask')}")
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()

    # 5. Test dispatch_with_spec (constraint_spec path)
    print("\n5. Testing dispatch_with_spec (constraint_spec path)...")
    try:
        spec = ConstraintSpec.from_dict({"regex": "[0-9]+"})
        print(f"   Spec created: {type(spec).__name__}")
        print(f"   spec.regex: {spec.regex}")
        print(f"   spec.get_syntax_constraint_type(): {spec.get_syntax_constraint_type()}")

        grammar2 = backend.dispatch_with_spec(spec)
        print(f"   Result type: {type(grammar2).__name__}")
        print(f"   syntax_grammar: {grammar2.syntax_grammar}")
        print(f"   Has fill_vocab_mask: {hasattr(grammar2, 'fill_vocab_mask')}")
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()

    # 6. Compare the grammars
    print("\n6. Comparing grammars...")
    if grammar1 and grammar2:
        print(f"   Legacy grammar type: {type(grammar1).__name__}")
        print(f"   Spec grammar type: {type(grammar2).__name__}")
        print(f"   Same type: {type(grammar1) == type(grammar2)}")

        # Check syntax grammars
        print(f"   Legacy syntax_grammar type: {type(grammar1.syntax_grammar).__name__ if grammar1.syntax_grammar else None}")
        print(f"   Spec syntax_grammar type: {type(grammar2.syntax_grammar).__name__ if grammar2.syntax_grammar else None}")

    # 7. Test fill_vocab_mask
    print("\n7. Testing fill_vocab_mask...")
    import torch
    vocab_size = 50257
    mask_size = (vocab_size + 31) // 32

    try:
        # Legacy grammar
        mask1 = grammar1.allocate_vocab_mask(vocab_size, 1, "cpu")
        grammar1.fill_vocab_mask(mask1, 0)
        popcount1 = mask1.sum().item() if mask1.dtype == torch.bool else torch.sum(mask1 != 0).item()
        print(f"   Legacy mask populated: shape={mask1.shape}, nonzero={popcount1}")
    except Exception as e:
        print(f"   Legacy mask FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Spec grammar
        mask2 = grammar2.allocate_vocab_mask(vocab_size, 1, "cpu")
        grammar2.fill_vocab_mask(mask2, 0)
        popcount2 = mask2.sum().item() if mask2.dtype == torch.bool else torch.sum(mask2 != 0).item()
        print(f"   Spec mask populated: shape={mask2.shape}, nonzero={popcount2}")
    except Exception as e:
        print(f"   Spec mask FAILED: {e}")
        import traceback
        traceback.print_exc()

    # 8. Test get_cached_or_future_with_spec
    print("\n8. Testing get_cached_or_future_with_spec...")
    try:
        spec2 = ConstraintSpec.from_dict({"regex": "[a-z]+"})  # Different regex for cache miss
        value, cache_hit = backend.get_cached_or_future_with_spec(spec2, require_reasoning=False)
        print(f"   Value type: {type(value).__name__}")
        print(f"   Cache hit: {cache_hit}")

        if hasattr(value, 'result'):
            # It's a Future
            print("   Value is a Future, resolving...")
            resolved = value.result(timeout=10)
            print(f"   Resolved type: {type(resolved).__name__}")
            print(f"   Resolved syntax_grammar: {resolved.syntax_grammar}")
        else:
            print(f"   Value is already resolved: {value}")
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)

    return "Done"


@app.local_entrypoint()
def main():
    result = debug_grammar_dispatch.remote()
    print(f"\n{result}")
