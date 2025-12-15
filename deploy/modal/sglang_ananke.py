"""
SGLang Ananke Deployment on Modal.com

Deploy SGLang with the Ananke constrained generation backend on Modal's
serverless GPU infrastructure.

Usage:
    # Deploy the service
    modal deploy sglang_ananke.py

    # Run locally for testing
    modal run sglang_ananke.py

    # Serve with hot-reload for development
    modal serve sglang_ananke.py

Environment Variables:
    MODAL_TOKEN_ID - Modal API token ID
    MODAL_TOKEN_SECRET - Modal API token secret
    HF_TOKEN - HuggingFace API token (for gated models)
"""

import os
import subprocess
import time
from typing import Optional

import modal

# =============================================================================
# Configuration
# =============================================================================

APP_NAME = os.getenv("MODAL_APP_NAME", "sglang-ananke")
MODEL_PATH = os.getenv("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
ANANKE_LANGUAGE = os.getenv("ANANKE_LANGUAGE", "python")
ANANKE_MAX_ROLLBACK_TOKENS = int(os.getenv("ANANKE_MAX_ROLLBACK_TOKENS", "200"))
ANANKE_ENABLED_DOMAINS = os.getenv("ANANKE_ENABLED_DOMAINS", "")

# GPU configuration
GPU_TYPE = os.getenv("MODAL_GPU_TYPE", "a100")  # a100, a10g, h100, t4
GPU_COUNT = int(os.getenv("MODAL_GPU_COUNT", "1"))
GPU_MEMORY = os.getenv("MODAL_GPU_MEMORY", "40GB")  # For A100: 40GB or 80GB; For H100: 80GB


def get_gpu_config() -> str:
    """Get the appropriate GPU configuration string for Modal.

    Returns a string like "A100-80GB" or "H100" based on environment variables.
    """
    gpu_type = GPU_TYPE.lower()
    if gpu_type == "a100":
        # A100 supports 40GB and 80GB variants
        return f"A100-{GPU_MEMORY}"
    elif gpu_type == "a10g":
        return "A10G"
    elif gpu_type == "h100":
        return "H100"
    elif gpu_type == "t4":
        return "T4"
    elif gpu_type == "l4":
        return "L4"
    else:
        raise ValueError(f"Unknown GPU type: {gpu_type}. Supported: a100, a10g, h100, t4, l4")

# Scaling configuration
CONTAINER_IDLE_TIMEOUT = int(os.getenv("MODAL_IDLE_TIMEOUT", "300"))  # seconds
MIN_CONTAINERS = int(os.getenv("MODAL_MIN_CONTAINERS", "0"))
MAX_CONTAINERS = int(os.getenv("MODAL_MAX_CONTAINERS", "10"))

# =============================================================================
# Modal App Setup
# =============================================================================

app = modal.App(APP_NAME)

# Create volume for model caching
model_volume = modal.Volume.from_name("sglang-model-cache", create_if_missing=True)

# Define the container image
sglang_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "libopenmpi-dev",
    )
    .pip_install(
        "torch>=2.4.0",
        "sglang[all]>=0.5.0",
        # Ananke optional dependencies
        "z3-solver>=4.12.0",
        "tree-sitter>=0.22.0",
        "immutables>=0.20",
        # Utilities
        "requests",
        "httpx",
    )
    .env({
        "SGLANG_GRAMMAR_BACKEND": "ananke",
        "ANANKE_LANGUAGE": ANANKE_LANGUAGE,
        "ANANKE_MAX_ROLLBACK_TOKENS": str(ANANKE_MAX_ROLLBACK_TOKENS),
    })
)


# =============================================================================
# SGLang Server Class
# =============================================================================

@app.cls(
    gpu=get_gpu_config(),
    image=sglang_image,
    volumes={"/models": model_volume},
    container_idle_timeout=CONTAINER_IDLE_TIMEOUT,
    allow_concurrent_inputs=100,
    secrets=[modal.Secret.from_name("huggingface")],
)
class SGLangAnanke:
    """SGLang server with Ananke backend on Modal."""

    model_path: str = MODEL_PATH
    ananke_language: str = ANANKE_LANGUAGE
    ananke_max_rollback_tokens: int = ANANKE_MAX_ROLLBACK_TOKENS
    ananke_enabled_domains: str = ANANKE_ENABLED_DOMAINS
    server_process: Optional[subprocess.Popen] = None
    server_url: str = "http://localhost:30000"

    @modal.enter()
    def start_server(self):
        """Start the SGLang server when container starts."""
        import requests

        # Build server command
        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--host", "0.0.0.0",
            "--port", "30000",
            "--grammar-backend", "ananke",
            "--ananke-language", self.ananke_language,
            "--ananke-max-rollback-tokens", str(self.ananke_max_rollback_tokens),
        ]

        if self.ananke_enabled_domains:
            cmd.extend(["--ananke-enabled-domains", self.ananke_enabled_domains])

        # Set HF cache to persistent volume
        env = os.environ.copy()
        env["HF_HOME"] = "/models"
        env["TRANSFORMERS_CACHE"] = "/models"

        # Start server
        print(f"Starting SGLang server with model: {self.model_path}")
        self.server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Wait for server to be ready
        print("Waiting for server to be ready...")
        start_time = time.time()
        timeout = 300  # 5 minutes for model loading

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                if response.status_code == 200:
                    print("Server is ready!")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(5)

        raise RuntimeError("Server failed to start within timeout")

    @modal.exit()
    def stop_server(self):
        """Stop the server when container shuts down."""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait(timeout=10)

    @modal.method()
    def health(self) -> dict:
        """Health check endpoint."""
        import requests
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return {"status": "healthy" if response.status_code == 200 else "unhealthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> str:
        """Simple text generation."""
        import requests

        payload = {
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            **kwargs,
        }

        response = requests.post(
            f"{self.server_url}/v1/completions",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"]

    @modal.method()
    def generate_constrained(
        self,
        prompt: str,
        constraint_spec: dict,
        max_tokens: int = 100,
        **kwargs
    ) -> str:
        """Generate text with Ananke constraint specification."""
        import requests

        payload = {
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "extra_body": {
                "constraint_spec": constraint_spec,
            },
            **kwargs,
        }

        response = requests.post(
            f"{self.server_url}/v1/completions",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"]

    @modal.web_endpoint(method="POST")
    def chat_completions(self, request: dict) -> dict:
        """OpenAI-compatible chat completions endpoint."""
        import requests

        response = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=request,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    @modal.web_endpoint(method="POST")
    def completions(self, request: dict) -> dict:
        """OpenAI-compatible completions endpoint."""
        import requests

        response = requests.post(
            f"{self.server_url}/v1/completions",
            json=request,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    @modal.web_endpoint(method="GET")
    def models(self) -> dict:
        """List available models."""
        import requests

        response = requests.get(f"{self.server_url}/v1/models", timeout=10)
        response.raise_for_status()
        return response.json()


# =============================================================================
# CLI Entry Points
# =============================================================================

@app.local_entrypoint()
def main():
    """Test the deployment locally."""
    print("Testing SGLang Ananke deployment...")

    sglang = SGLangAnanke()

    # Test health
    health = sglang.health.remote()
    print(f"Health: {health}")

    # Test simple generation
    result = sglang.generate.remote(
        prompt="def fibonacci(n):",
        max_tokens=50,
    )
    print(f"Generated:\n{result}")

    # Test constrained generation
    result = sglang.generate_constrained.remote(
        prompt="def calculate_sum(numbers: list[int]) -> int:",
        constraint_spec={
            "language": "python",
            "domains": ["syntax", "types"],
        },
        max_tokens=100,
    )
    print(f"Constrained:\n{result}")


# =============================================================================
# Deployment Scripts
# =============================================================================

@app.function()
def download_model(model_path: str = MODEL_PATH):
    """Pre-download model to volume for faster cold starts."""
    from huggingface_hub import snapshot_download

    print(f"Downloading model: {model_path}")
    snapshot_download(
        model_path,
        local_dir=f"/models/{model_path.replace('/', '_')}",
        local_dir_use_symlinks=False,
    )
    model_volume.commit()
    print("Model downloaded and cached!")


if __name__ == "__main__":
    # For local testing
    main()
