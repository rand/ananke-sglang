#!/usr/bin/env python3
"""
Unified Deployment CLI for SGLang Ananke

A single entry point for deploying SGLang with Ananke backend across
multiple platforms: Docker, Kubernetes, Modal, AWS, and GCP.

Usage:
    # Docker
    python deploy.py docker run --model meta-llama/Llama-3.1-8B-Instruct
    python deploy.py docker build --variant full

    # Docker Compose
    python deploy.py compose up
    python deploy.py compose down

    # Kubernetes (Helm)
    python deploy.py helm install my-release
    python deploy.py helm upgrade my-release

    # Modal.com
    python deploy.py modal deploy
    python deploy.py modal run

    # AWS SageMaker
    python deploy.py sagemaker deploy --model-name sglang-ananke

    # GCP Vertex AI
    python deploy.py vertex deploy --model-name sglang-ananke

Environment Variables:
    See .env.example for full configuration options.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Get deploy directory path
DEPLOY_DIR = Path(__file__).parent.parent.resolve()
REPO_ROOT = DEPLOY_DIR.parent


class DeployCLI:
    """Unified deployment CLI for SGLang Ananke."""

    def __init__(self):
        self.deploy_dir = DEPLOY_DIR
        self.docker_dir = self.deploy_dir / "docker"
        self.helm_dir = self.deploy_dir / "helm" / "sglang-ananke"
        self.modal_dir = self.deploy_dir / "modal"
        self.aws_dir = self.deploy_dir / "aws"
        self.gcp_dir = self.deploy_dir / "gcp"

    def run_command(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
    ) -> int:
        """Run a command and return exit code."""
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd, env=full_env)
        return result.returncode

    # =========================================================================
    # Docker Commands
    # =========================================================================

    def docker_build(
        self,
        variant: str = "full",
        tag: str = "sglang-ananke:latest",
        build_zig: bool = False,
        no_cache: bool = False,
    ) -> int:
        """Build SGLang Ananke Docker image."""
        dockerfile = f"Dockerfile.ananke{'- minimal' if variant == 'minimal' else ''}"

        cmd = ["docker", "build"]
        if no_cache:
            cmd.append("--no-cache")
        cmd.extend(["-f", str(self.docker_dir / dockerfile)])
        cmd.extend(["-t", tag])

        if build_zig:
            cmd.extend(["--build-arg", "BUILD_ZIG_NATIVE=1"])

        cmd.append(str(REPO_ROOT))

        return self.run_command(cmd)

    def docker_run(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        gpus: int = 1,
        port: int = 30000,
        image: str = "sglang-ananke:latest",
        detach: bool = False,
        name: str = "sglang-ananke",
    ) -> int:
        """Run SGLang Ananke Docker container."""
        cmd = ["docker", "run"]

        if detach:
            cmd.append("-d")
        cmd.extend(["--name", name])
        cmd.extend(["--gpus", f'"device={",".join(str(i) for i in range(gpus))}"'])
        cmd.extend(["-p", f"{port}:{port}"])
        cmd.extend(["-v", f"{os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface"])
        cmd.extend(["--shm-size", "10g"])
        cmd.extend(["--ipc", "host"])

        # Environment variables
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            cmd.extend(["-e", f"HF_TOKEN={hf_token}"])

        cmd.append(image)
        cmd.extend([
            "--model-path", model,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--grammar-backend", "ananke",
        ])

        return self.run_command(cmd)

    # =========================================================================
    # Docker Compose Commands
    # =========================================================================

    def compose_up(
        self,
        detach: bool = True,
        with_monitoring: bool = False,
        build: bool = False,
    ) -> int:
        """Start services with Docker Compose."""
        cmd = ["docker", "compose", "-f", str(self.docker_dir / "compose.ananke.yaml")]

        if with_monitoring:
            cmd.extend(["-f", str(self.docker_dir / "compose.monitoring.yaml")])

        cmd.append("up")
        if detach:
            cmd.append("-d")
        if build:
            cmd.append("--build")

        return self.run_command(cmd, cwd=self.deploy_dir)

    def compose_down(self, volumes: bool = False) -> int:
        """Stop services with Docker Compose."""
        cmd = ["docker", "compose", "-f", str(self.docker_dir / "compose.ananke.yaml"), "down"]
        if volumes:
            cmd.append("-v")
        return self.run_command(cmd, cwd=self.deploy_dir)

    def compose_logs(self, follow: bool = True) -> int:
        """View Docker Compose logs."""
        cmd = ["docker", "compose", "-f", str(self.docker_dir / "compose.ananke.yaml"), "logs"]
        if follow:
            cmd.append("-f")
        return self.run_command(cmd, cwd=self.deploy_dir)

    # =========================================================================
    # Helm Commands
    # =========================================================================

    def helm_install(
        self,
        release_name: str,
        namespace: str = "default",
        values_file: Optional[str] = None,
        set_values: Optional[dict] = None,
        dry_run: bool = False,
    ) -> int:
        """Install Helm chart."""
        cmd = ["helm", "install", release_name, str(self.helm_dir)]
        cmd.extend(["-n", namespace])

        if values_file:
            cmd.extend(["-f", values_file])

        if set_values:
            for key, value in set_values.items():
                cmd.extend(["--set", f"{key}={value}"])

        if dry_run:
            cmd.append("--dry-run")

        return self.run_command(cmd)

    def helm_upgrade(
        self,
        release_name: str,
        namespace: str = "default",
        values_file: Optional[str] = None,
        set_values: Optional[dict] = None,
    ) -> int:
        """Upgrade Helm release."""
        cmd = ["helm", "upgrade", release_name, str(self.helm_dir)]
        cmd.extend(["-n", namespace])

        if values_file:
            cmd.extend(["-f", values_file])

        if set_values:
            for key, value in set_values.items():
                cmd.extend(["--set", f"{key}={value}"])

        return self.run_command(cmd)

    def helm_uninstall(self, release_name: str, namespace: str = "default") -> int:
        """Uninstall Helm release."""
        cmd = ["helm", "uninstall", release_name, "-n", namespace]
        return self.run_command(cmd)

    def helm_template(
        self,
        release_name: str = "test",
        values_file: Optional[str] = None,
    ) -> int:
        """Render Helm templates for inspection."""
        cmd = ["helm", "template", release_name, str(self.helm_dir)]
        if values_file:
            cmd.extend(["-f", values_file])
        return self.run_command(cmd)

    # =========================================================================
    # Modal Commands
    # =========================================================================

    def modal_deploy(self) -> int:
        """Deploy to Modal.com."""
        cmd = ["modal", "deploy", str(self.modal_dir / "sglang_ananke.py")]
        return self.run_command(cmd)

    def modal_run(self) -> int:
        """Run Modal app locally."""
        cmd = ["modal", "run", str(self.modal_dir / "sglang_ananke.py")]
        return self.run_command(cmd)

    def modal_serve(self) -> int:
        """Serve Modal app with hot-reload."""
        cmd = ["modal", "serve", str(self.modal_dir / "sglang_ananke.py")]
        return self.run_command(cmd)

    # =========================================================================
    # AWS SageMaker Commands
    # =========================================================================

    def sagemaker_deploy(
        self,
        model_name: str,
        endpoint_name: Optional[str] = None,
        image_uri: Optional[str] = None,
        instance_type: str = "ml.g5.xlarge",
    ) -> int:
        """Deploy to AWS SageMaker."""
        deploy_script = self.aws_dir / "sagemaker" / "deploy.py"

        endpoint_name = endpoint_name or model_name

        cmd = [
            sys.executable, str(deploy_script),
            "deploy",
            "--endpoint-name", endpoint_name,
            "--model-name", model_name,
            "--instance-type", instance_type,
        ]

        return self.run_command(cmd)

    def sagemaker_delete(self, endpoint_name: str) -> int:
        """Delete SageMaker endpoint."""
        deploy_script = self.aws_dir / "sagemaker" / "deploy.py"
        cmd = [sys.executable, str(deploy_script), "delete", "--endpoint-name", endpoint_name]
        return self.run_command(cmd)

    # =========================================================================
    # GCP Vertex AI Commands
    # =========================================================================

    def vertex_deploy(
        self,
        model_name: str,
        endpoint_name: Optional[str] = None,
        image_uri: Optional[str] = None,
        gpu_type: str = "l4",
    ) -> int:
        """Deploy to GCP Vertex AI."""
        deploy_script = self.gcp_dir / "vertexai" / "deploy.py"

        endpoint_name = endpoint_name or model_name

        cmd = [
            sys.executable, str(deploy_script),
            "deploy",
            "--endpoint-name", endpoint_name,
            "--model-name", model_name,
            "--gpu-type", gpu_type,
            "--create-endpoint",
        ]

        return self.run_command(cmd)

    def vertex_delete(self, endpoint_name: str, force: bool = False) -> int:
        """Delete Vertex AI endpoint."""
        deploy_script = self.gcp_dir / "vertexai" / "deploy.py"
        cmd = [sys.executable, str(deploy_script), "delete", "--endpoint-name", endpoint_name]
        if force:
            cmd.append("--force")
        return self.run_command(cmd)


def main():
    cli = DeployCLI()

    parser = argparse.ArgumentParser(
        description="Unified deployment CLI for SGLang Ananke",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="platform", required=True)

    # =========================================================================
    # Docker
    # =========================================================================
    docker_parser = subparsers.add_parser("docker", help="Docker commands")
    docker_subparsers = docker_parser.add_subparsers(dest="command", required=True)

    # docker build
    docker_build_parser = docker_subparsers.add_parser("build", help="Build Docker image")
    docker_build_parser.add_argument("--variant", choices=["full", "minimal"], default="full")
    docker_build_parser.add_argument("--tag", default="sglang-ananke:latest")
    docker_build_parser.add_argument("--build-zig", action="store_true")
    docker_build_parser.add_argument("--no-cache", action="store_true")

    # docker run
    docker_run_parser = docker_subparsers.add_parser("run", help="Run Docker container")
    docker_run_parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    docker_run_parser.add_argument("--gpus", type=int, default=1)
    docker_run_parser.add_argument("--port", type=int, default=30000)
    docker_run_parser.add_argument("--image", default="sglang-ananke:latest")
    docker_run_parser.add_argument("-d", "--detach", action="store_true")
    docker_run_parser.add_argument("--name", default="sglang-ananke")

    # =========================================================================
    # Compose
    # =========================================================================
    compose_parser = subparsers.add_parser("compose", help="Docker Compose commands")
    compose_subparsers = compose_parser.add_subparsers(dest="command", required=True)

    # compose up
    compose_up_parser = compose_subparsers.add_parser("up", help="Start services")
    compose_up_parser.add_argument("--no-detach", action="store_true")
    compose_up_parser.add_argument("--monitoring", action="store_true")
    compose_up_parser.add_argument("--build", action="store_true")

    # compose down
    compose_down_parser = compose_subparsers.add_parser("down", help="Stop services")
    compose_down_parser.add_argument("-v", "--volumes", action="store_true")

    # compose logs
    compose_logs_parser = compose_subparsers.add_parser("logs", help="View logs")
    compose_logs_parser.add_argument("--no-follow", action="store_true")

    # =========================================================================
    # Helm
    # =========================================================================
    helm_parser = subparsers.add_parser("helm", help="Kubernetes Helm commands")
    helm_subparsers = helm_parser.add_subparsers(dest="command", required=True)

    # helm install
    helm_install_parser = helm_subparsers.add_parser("install", help="Install chart")
    helm_install_parser.add_argument("release_name")
    helm_install_parser.add_argument("-n", "--namespace", default="default")
    helm_install_parser.add_argument("-f", "--values")
    helm_install_parser.add_argument("--dry-run", action="store_true")

    # helm upgrade
    helm_upgrade_parser = helm_subparsers.add_parser("upgrade", help="Upgrade release")
    helm_upgrade_parser.add_argument("release_name")
    helm_upgrade_parser.add_argument("-n", "--namespace", default="default")
    helm_upgrade_parser.add_argument("-f", "--values")

    # helm uninstall
    helm_uninstall_parser = helm_subparsers.add_parser("uninstall", help="Uninstall release")
    helm_uninstall_parser.add_argument("release_name")
    helm_uninstall_parser.add_argument("-n", "--namespace", default="default")

    # helm template
    helm_template_parser = helm_subparsers.add_parser("template", help="Render templates")
    helm_template_parser.add_argument("release_name", nargs="?", default="test")
    helm_template_parser.add_argument("-f", "--values")

    # =========================================================================
    # Modal
    # =========================================================================
    modal_parser = subparsers.add_parser("modal", help="Modal.com commands")
    modal_subparsers = modal_parser.add_subparsers(dest="command", required=True)

    modal_subparsers.add_parser("deploy", help="Deploy to Modal")
    modal_subparsers.add_parser("run", help="Run locally")
    modal_subparsers.add_parser("serve", help="Serve with hot-reload")

    # =========================================================================
    # SageMaker
    # =========================================================================
    sagemaker_parser = subparsers.add_parser("sagemaker", help="AWS SageMaker commands")
    sagemaker_subparsers = sagemaker_parser.add_subparsers(dest="command", required=True)

    # sagemaker deploy
    sm_deploy_parser = sagemaker_subparsers.add_parser("deploy", help="Deploy endpoint")
    sm_deploy_parser.add_argument("--model-name", required=True)
    sm_deploy_parser.add_argument("--endpoint-name")
    sm_deploy_parser.add_argument("--image-uri")
    sm_deploy_parser.add_argument("--instance-type", default="ml.g5.xlarge")

    # sagemaker delete
    sm_delete_parser = sagemaker_subparsers.add_parser("delete", help="Delete endpoint")
    sm_delete_parser.add_argument("--endpoint-name", required=True)

    # =========================================================================
    # Vertex AI
    # =========================================================================
    vertex_parser = subparsers.add_parser("vertex", help="GCP Vertex AI commands")
    vertex_subparsers = vertex_parser.add_subparsers(dest="command", required=True)

    # vertex deploy
    vertex_deploy_parser = vertex_subparsers.add_parser("deploy", help="Deploy endpoint")
    vertex_deploy_parser.add_argument("--model-name", required=True)
    vertex_deploy_parser.add_argument("--endpoint-name")
    vertex_deploy_parser.add_argument("--image-uri")
    vertex_deploy_parser.add_argument("--gpu-type", default="l4")

    # vertex delete
    vertex_delete_parser = vertex_subparsers.add_parser("delete", help="Delete endpoint")
    vertex_delete_parser.add_argument("--endpoint-name", required=True)
    vertex_delete_parser.add_argument("--force", action="store_true")

    # =========================================================================
    # Parse and Execute
    # =========================================================================
    args = parser.parse_args()

    exit_code = 0

    # Docker
    if args.platform == "docker":
        if args.command == "build":
            exit_code = cli.docker_build(
                variant=args.variant,
                tag=args.tag,
                build_zig=args.build_zig,
                no_cache=args.no_cache,
            )
        elif args.command == "run":
            exit_code = cli.docker_run(
                model=args.model,
                gpus=args.gpus,
                port=args.port,
                image=args.image,
                detach=args.detach,
                name=args.name,
            )

    # Compose
    elif args.platform == "compose":
        if args.command == "up":
            exit_code = cli.compose_up(
                detach=not args.no_detach,
                with_monitoring=args.monitoring,
                build=args.build,
            )
        elif args.command == "down":
            exit_code = cli.compose_down(volumes=args.volumes)
        elif args.command == "logs":
            exit_code = cli.compose_logs(follow=not args.no_follow)

    # Helm
    elif args.platform == "helm":
        if args.command == "install":
            exit_code = cli.helm_install(
                release_name=args.release_name,
                namespace=args.namespace,
                values_file=args.values,
                dry_run=args.dry_run,
            )
        elif args.command == "upgrade":
            exit_code = cli.helm_upgrade(
                release_name=args.release_name,
                namespace=args.namespace,
                values_file=args.values,
            )
        elif args.command == "uninstall":
            exit_code = cli.helm_uninstall(
                release_name=args.release_name,
                namespace=args.namespace,
            )
        elif args.command == "template":
            exit_code = cli.helm_template(
                release_name=args.release_name,
                values_file=args.values,
            )

    # Modal
    elif args.platform == "modal":
        if args.command == "deploy":
            exit_code = cli.modal_deploy()
        elif args.command == "run":
            exit_code = cli.modal_run()
        elif args.command == "serve":
            exit_code = cli.modal_serve()

    # SageMaker
    elif args.platform == "sagemaker":
        if args.command == "deploy":
            exit_code = cli.sagemaker_deploy(
                model_name=args.model_name,
                endpoint_name=args.endpoint_name,
                image_uri=args.image_uri,
                instance_type=args.instance_type,
            )
        elif args.command == "delete":
            exit_code = cli.sagemaker_delete(args.endpoint_name)

    # Vertex AI
    elif args.platform == "vertex":
        if args.command == "deploy":
            exit_code = cli.vertex_deploy(
                model_name=args.model_name,
                endpoint_name=args.endpoint_name,
                image_uri=args.image_uri,
                gpu_type=args.gpu_type,
            )
        elif args.command == "delete":
            exit_code = cli.vertex_delete(args.endpoint_name, force=args.force)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
