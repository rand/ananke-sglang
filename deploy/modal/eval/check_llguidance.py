"""Check if llguidance is available on the Modal deployment."""

import modal

DEPLOYED_APP = "qwen3-coder-ananke"
DEPLOYED_CLASS = "Qwen3CoderAnanke"

app = modal.App("check-llguidance")

# Use the same image as the deployed model
import os
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
def check_imports():
    """Check what's available on the server."""
    print("=" * 60)
    print("CHECKING IMPORTS ON MODAL")
    print("=" * 60)

    # Check llguidance
    print("\n1. Checking llguidance...")
    try:
        import llguidance
        print(f"   llguidance: AVAILABLE (version: {getattr(llguidance, '__version__', 'unknown')})")
    except ImportError as e:
        print(f"   llguidance: NOT AVAILABLE ({e})")

    # Check Ananke imports
    print("\n2. Checking Ananke imports...")
    try:
        from sglang.srt.constrained.ananke.backend.backend import AnankeBackend
        print("   AnankeBackend: AVAILABLE")
    except ImportError as e:
        print(f"   AnankeBackend: NOT AVAILABLE ({e})")

    try:
        from sglang.srt.constrained.ananke.spec.constraint_spec import ConstraintSpec
        print("   ConstraintSpec: AVAILABLE")
    except ImportError as e:
        print(f"   ConstraintSpec: NOT AVAILABLE ({e})")

    # Check guidance backend
    print("\n3. Checking GuidanceBackend...")
    try:
        from sglang.srt.constrained.llguidance_backend import GuidanceBackend
        print("   GuidanceBackend: AVAILABLE")
    except ImportError as e:
        print(f"   GuidanceBackend: NOT AVAILABLE ({e})")

    # Check registered backends
    print("\n4. Checking registered grammar backends...")
    try:
        from sglang.srt.constrained.base_grammar_backend import get_grammar_backend_cls

        for backend_name in ["ananke", "llguidance", "outlines", "xgrammar"]:
            try:
                cls = get_grammar_backend_cls(backend_name)
                print(f"   {backend_name}: REGISTERED ({cls.__name__})")
            except Exception as e:
                print(f"   {backend_name}: NOT REGISTERED ({e})")
    except Exception as e:
        print(f"   Error checking backends: {e}")

    print("\n" + "=" * 60)
    return "Done"


@app.local_entrypoint()
def main():
    result = check_imports.remote()
    print(f"\n{result}")
