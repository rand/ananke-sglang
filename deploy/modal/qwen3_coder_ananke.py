"""
Qwen3-Coder-30B-A3B-Instruct Deployment on Modal.com

Production deployment of the Qwen3-Coder MoE model (30.5B total, 3.3B active)
with SGLang and Ananke constrained generation backend.

Model Details:
    - Total Parameters: 30.5B
    - Active Parameters: 3.3B per token (MoE with 128 experts, 8 active)
    - Context Length: 262K tokens
    - Optimized for code generation

Usage:
    # Pre-download model (recommended for faster cold starts)
    modal run deploy/modal/qwen3_coder_ananke.py::download_model

    # Deploy the service
    modal deploy deploy/modal/qwen3_coder_ananke.py

    # Run E2E tests
    modal run deploy/modal/tests/test_qwen3_coder.py

    # Check logs
    modal logs qwen3-coder-ananke

Requirements:
    - Modal.com account with A100-80GB access
    - HuggingFace token for model access: modal secret create huggingface HF_TOKEN=<token>
"""

import os
import subprocess
import time
from typing import Optional

import modal

# =============================================================================
# Configuration
# =============================================================================

APP_NAME = "qwen3-coder-ananke"
MODEL_PATH = "unsloth/Qwen3-Coder-30B-A3B-Instruct"

# MoE model optimizations
# - mem-fraction-static 0.85: Leave headroom for expert routing tables
# - max-running-requests 32: Conservative for large model
# - chunked-prefill-size 8192: Balance throughput with memory
MEM_FRACTION_STATIC = 0.85
MAX_RUNNING_REQUESTS = 32
CHUNKED_PREFILL_SIZE = 8192

# Default language for Ananke (coding model)
ANANKE_LANGUAGE = "python"
ANANKE_MAX_ROLLBACK_TOKENS = 200

# Scaling configuration
CONTAINER_IDLE_TIMEOUT = 600  # 10 minutes - longer for expensive model
MIN_CONTAINERS = 0
MAX_CONTAINERS = 5

# Timeout for model loading (MoE is large - 60GB takes 15-20 min)
MODEL_LOAD_TIMEOUT = 1500  # 25 minutes

# =============================================================================
# Modal App Setup
# =============================================================================

app = modal.App(APP_NAME)

# Create volume for model caching (MoE model is ~60GB)
model_volume = modal.Volume.from_name(
    "qwen3-coder-model-cache",
    create_if_missing=True
)

# Get the repo root (deploy/modal/qwen3_coder_ananke.py -> repo root)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Build configuration
CUDA_VERSION = "12.4"

# Define the container image - build SGLang from this repo with Ananke
qwen3_image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    # System dependencies
    .apt_install(
        "git",
        "curl",
        "wget",
        "build-essential",
        "cmake",
        "libopenmpi-dev",
        "libnuma-dev",
        "patchelf",
    )
    # Upgrade pip and install build tools
    .run_commands(
        "pip install --upgrade pip setuptools wheel",
    )
    # Copy local sglang repo - need python/ dir and README.md for pyproject.toml
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
    # Install sglang from local repo - let pip resolve all dependencies including flashinfer
    .run_commands(
        "cd /sglang/python && pip install -e '.[srt]'",
    )
    # Install Ananke dependencies
    .pip_install(
        "z3-solver>=4.12.0",
        "tree-sitter>=0.22.0",
        "immutables>=0.20",
    )
    .env({
        "SGLANG_GRAMMAR_BACKEND": "ananke",
        "ANANKE_LANGUAGE": ANANKE_LANGUAGE,
        "ANANKE_MAX_ROLLBACK_TOKENS": str(ANANKE_MAX_ROLLBACK_TOKENS),
        "SGLANG_ALLOW_OVERWRITE": "1",
        "CUDA_HOME": "/usr/local/cuda",
    })
)


# =============================================================================
# SGLang Server Class
# =============================================================================

@app.cls(
    # A100-80GB required for 30B MoE model
    gpu="A100-80GB",
    image=qwen3_image,
    volumes={"/models": model_volume},
    scaledown_window=CONTAINER_IDLE_TIMEOUT,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MODEL_LOAD_TIMEOUT,
)
@modal.concurrent(max_inputs=100)
class Qwen3CoderAnanke:
    """Qwen3-Coder-30B server with Ananke backend on Modal."""

    model_path: str = MODEL_PATH
    server_process: Optional[subprocess.Popen] = None
    server_url: str = "http://localhost:30000"

    @modal.enter()
    def start_server(self):
        """Start the SGLang server when container starts."""
        import requests

        # Build server command with MoE optimizations
        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--host", "0.0.0.0",
            "--port", "30000",
            # Ananke configuration (from local repo)
            "--grammar-backend", "ananke",
            "--ananke-language", ANANKE_LANGUAGE,
            "--ananke-max-rollback-tokens", str(ANANKE_MAX_ROLLBACK_TOKENS),
            # MoE optimizations
            "--mem-fraction-static", str(MEM_FRACTION_STATIC),
            "--max-running-requests", str(MAX_RUNNING_REQUESTS),
            "--chunked-prefill-size", str(CHUNKED_PREFILL_SIZE),
            # Trust remote code for custom model
            "--trust-remote-code",
        ]

        # Set HF cache to persistent volume
        env = os.environ.copy()
        env["HF_HOME"] = "/models"
        env["TRANSFORMERS_CACHE"] = "/models"
        env["HF_HUB_CACHE"] = "/models/hub"

        # Start server
        print(f"Starting SGLang server with Qwen3-Coder MoE model...")
        print(f"Model: {self.model_path}")
        print(f"GPU Memory Fraction: {MEM_FRACTION_STATIC}")
        print(f"Max Concurrent Requests: {MAX_RUNNING_REQUESTS}")

        # Start server with output going to a log file for debugging
        import threading

        log_file = open("/tmp/sglang_server.log", "w")
        self.server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        self._log_file = log_file

        # Wait for server to be ready (MoE model takes longer to load)
        print(f"Waiting up to {MODEL_LOAD_TIMEOUT}s for model to load...")
        start_time = time.time()

        while time.time() - start_time < MODEL_LOAD_TIMEOUT:
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    print(f"Server ready in {elapsed:.1f}s")
                    return
            except requests.exceptions.RequestException:
                pass

            # Check if process died
            if self.server_process.poll() is not None:
                # Read any output
                stdout, _ = self.server_process.communicate()
                raise RuntimeError(
                    f"Server process died. Output:\n{stdout.decode()[:2000]}"
                )

            time.sleep(10)  # Check every 10s for large model

        raise RuntimeError(
            f"Server failed to start within {MODEL_LOAD_TIMEOUT}s timeout"
        )

    @modal.exit()
    def stop_server(self):
        """Stop the server when container shuts down."""
        if self.server_process:
            print("Stopping SGLang server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.server_process.kill()

    def _get_server_logs(self, lines: int = 100) -> str:
        """Get recent server logs for debugging."""
        try:
            with open("/tmp/sglang_server.log", "r") as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except Exception as e:
            return f"Could not read logs: {e}"

    def _check_server_alive(self) -> bool:
        """Check if server process is still running."""
        if self.server_process is None:
            return False
        return self.server_process.poll() is None

    @modal.method()
    def get_server_logs(self, lines: int = 200) -> dict:
        """Get server logs for debugging."""
        return {
            "logs": self._get_server_logs(lines),
            "process_alive": self._check_server_alive(),
        }

    @modal.method()
    def health(self) -> dict:
        """Health check endpoint."""
        import requests
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "model": self.model_path,
                "process_alive": self._check_server_alive(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "process_alive": self._check_server_alive(),
                "recent_logs": self._get_server_logs(50),
            }

    @modal.method()
    def health_generate(self) -> dict:
        """Readiness check - model can generate."""
        import requests
        try:
            response = requests.get(
                f"{self.server_url}/health_generate",
                timeout=30
            )
            return {
                "ready": response.status_code == 200,
                "model": self.model_path,
            }
        except Exception as e:
            return {"ready": False, "error": str(e)}

    @modal.method()
    def get_models(self) -> dict:
        """List available models."""
        import requests
        response = requests.get(f"{self.server_url}/v1/models", timeout=10)
        response.raise_for_status()
        return response.json()

    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 200, **kwargs) -> str:
        """Simple text generation."""
        import requests

        payload = {
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
        }

        response = requests.post(
            f"{self.server_url}/v1/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"]

    @modal.method()
    def generate_constrained(
        self,
        prompt: str,
        constraint_spec: dict,
        max_tokens: int = 200,
        **kwargs
    ) -> dict:
        """Generate text with Ananke constraint specification.

        Args:
            prompt: The input prompt
            constraint_spec: Ananke constraint specification, e.g.:
                {
                    "language": "python",
                    "type_bindings": [{"name": "x", "type_expr": "int"}],
                    "expected_type": "int"
                }
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            dict with "text" and "constraint_info"
        """
        import requests

        # constraint_spec is a top-level field in SGLang's OpenAI-compatible API
        payload = {
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "constraint_spec": constraint_spec,
        }

        response = requests.post(
            f"{self.server_url}/v1/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        result = response.json()
        return {
            "text": result["choices"][0]["text"],
            "usage": result.get("usage", {}),
        }

    @modal.method()
    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 200,
        constraint_spec: Optional[dict] = None,
        **kwargs
    ) -> dict:
        """Chat completion with optional constraints.

        Args:
            messages: List of chat messages [{"role": "user", "content": "..."}]
            max_tokens: Maximum tokens to generate
            constraint_spec: Optional Ananke constraint specification
            **kwargs: Additional generation parameters

        Returns:
            dict with "content" and "usage"
        """
        import requests

        payload = {
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
        }

        if constraint_spec:
            # constraint_spec is a top-level field in SGLang's OpenAI-compatible API
            payload["constraint_spec"] = constraint_spec

        response = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        result = response.json()
        return {
            "content": result["choices"][0]["message"]["content"],
            "usage": result.get("usage", {}),
        }

    @modal.method()
    def chat_completions(self, request: dict) -> dict:
        """OpenAI-compatible chat completions method."""
        import requests

        response = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=request,
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    @modal.method()
    def completions(self, request: dict) -> dict:
        """OpenAI-compatible completions method."""
        import requests

        response = requests.post(
            f"{self.server_url}/v1/completions",
            json=request,
            timeout=180,
        )
        response.raise_for_status()
        return response.json()


# =============================================================================
# Model Download Function
# =============================================================================

@app.function(
    image=qwen3_image,
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,  # 1 hour for large model download
)
def download_model():
    """Pre-download Qwen3-Coder model to volume for faster cold starts.

    Run this once before deploying:
        modal run deploy/modal/qwen3_coder_ananke.py::download_model
    """
    from huggingface_hub import snapshot_download
    import os

    print(f"Downloading model: {MODEL_PATH}")
    print("This may take 30-60 minutes for the 60GB model...")

    # Ensure HF token is available
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Using HuggingFace token for authentication")

    snapshot_download(
        MODEL_PATH,
        local_dir=f"/models/hub/models--{MODEL_PATH.replace('/', '--')}",
        local_dir_use_symlinks=False,
        token=hf_token,
    )

    model_volume.commit()
    print("Model downloaded and cached successfully!")
    print(f"Volume size: ~60GB")


# =============================================================================
# CLI Entry Points
# =============================================================================

@app.local_entrypoint()
def main():
    """Test the Qwen3-Coder deployment."""
    print("=" * 60)
    print("Testing Qwen3-Coder-30B Ananke Deployment")
    print("=" * 60)

    server = Qwen3CoderAnanke()

    # Test health
    print("\n[1] Testing health endpoint...")
    health = server.health.remote()
    print(f"Health: {health}")
    assert health["status"] == "healthy", f"Health check failed: {health}"

    # Test readiness
    print("\n[2] Testing readiness endpoint...")
    ready = server.health_generate.remote()
    print(f"Readiness: {ready}")
    assert ready["ready"], f"Readiness check failed: {ready}"

    # Test models endpoint
    print("\n[3] Testing models endpoint...")
    models = server.get_models.remote()
    print(f"Models: {[m['id'] for m in models.get('data', [])]}")

    # Test simple generation
    print("\n[4] Testing simple generation...")
    result = server.generate.remote(
        prompt="def fibonacci(n: int) -> int:",
        max_tokens=100,
    )
    print(f"Generated:\n{result}")

    # Test constrained generation with regex
    print("\n[5] Testing syntax-constrained generation (regex)...")
    result = server.generate_constrained.remote(
        prompt="The answer is: ",
        constraint_spec={
            "language": "python",
            "regex": "[0-9]+",
        },
        max_tokens=10,
    )
    print(f"Constrained (regex=[0-9]+):\n{result['text']}")

    # Test constraint_spec with syntax + domain context (fully supported)
    print("\n[5b] Testing constraint_spec with syntax + type context...")
    result = server.generate_constrained.remote(
        prompt="x = ",
        constraint_spec={
            "language": "python",
            "regex": "[0-9]+",  # Syntax constraint required for domain constraints
            "type_bindings": [{"name": "x", "type_expr": "int"}],
            "expected_type": "int",
        },
        max_tokens=10,
    )
    print(f"Constrained (regex + type context):\n{result['text']}")

    # Test chat endpoint
    print("\n[6] Testing chat completion...")
    result = server.chat.remote(
        messages=[
            {"role": "user", "content": "Write a Python function to check if a number is prime."}
        ],
        max_tokens=200,
    )
    print(f"Chat response:\n{result['content']}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
