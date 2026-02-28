"""Test pydantic parsing of constraint_spec locally on Modal."""

import modal
import os

app = modal.App("test-pydantic-local")

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
)


@app.function(timeout=300, image=image, gpu="T4")
def test_pydantic():
    """Test pydantic parsing of constraint_spec."""
    import sys
    sys.path.insert(0, "/sglang/python")

    print("=" * 60)
    print("TESTING PYDANTIC PARSING")
    print("=" * 60)

    # 1. Test ConstraintSpecFormat parsing
    print("\n1. Testing ConstraintSpecFormat directly...")
    from sglang.srt.entrypoints.openai.protocol import ConstraintSpecFormat

    spec_dict = {"language": "python"}
    try:
        spec = ConstraintSpecFormat(**spec_dict)
        print(f"   ConstraintSpecFormat parsed: {spec}")
        print(f"   Type: {type(spec)}")
        print(f"   model_dump(): {spec.model_dump()}")
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()

    # 2. Test CompletionRequest parsing with constraint_spec
    print("\n2. Testing CompletionRequest with constraint_spec dict...")
    from sglang.srt.entrypoints.openai.protocol import CompletionRequest

    req_dict = {
        "model": "default",
        "prompt": "test",
        "max_tokens": 10,
        "constraint_spec": {"language": "python"},
    }
    try:
        req = CompletionRequest(**req_dict)
        print(f"   CompletionRequest parsed successfully")
        print(f"   constraint_spec type: {type(req.constraint_spec)}")
        print(f"   constraint_spec: {req.constraint_spec}")
        if req.constraint_spec:
            print(f"   model_dump: {req.constraint_spec.model_dump()}")
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()

    # 3. Test _build_sampling_params
    print("\n3. Testing _build_sampling_params...")
    from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion

    # We can't easily instantiate OpenAIServingCompletion without a tokenizer_manager,
    # so let's just check the method signature and manually replicate what it does
    print("   Checking what _build_sampling_params does with constraint_spec...")

    # This is what _build_sampling_params does:
    if req.constraint_spec:
        constraint_spec_dump = req.constraint_spec.model_dump()
        print(f"   constraint_spec.model_dump() = {constraint_spec_dump}")
    else:
        print("   constraint_spec is None")

    # 4. Test what the scheduler expects
    print("\n4. Testing ConstraintSpec.from_dict (internal Ananke type)...")
    try:
        from sglang.srt.constrained.ananke.spec.constraint_spec import ConstraintSpec

        # This is what the scheduler does with the dict
        spec_dict = {"language": "python", "regex": "[0-9]+"}
        spec = ConstraintSpec.from_dict(spec_dict)
        print(f"   ConstraintSpec.from_dict succeeded")
        print(f"   Type: {type(spec)}")
        print(f"   language: {spec.language}")
        print(f"   regex: {spec.regex}")
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)

    return "Done"


@app.local_entrypoint()
def main():
    result = test_pydantic.remote()
    print(f"\n{result}")
