"""Tests for deploy.py unified CLI."""

import unittest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add deploy/scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from deploy import DeployCLI


class TestDeployCLI(unittest.TestCase):
    """Test DeployCLI class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cli = DeployCLI()

    def test_deploy_dir_exists(self):
        """Test that deploy directory is correctly identified."""
        self.assertTrue(self.cli.deploy_dir.exists())
        self.assertTrue((self.cli.deploy_dir / "docker").exists())
        self.assertTrue((self.cli.deploy_dir / "helm").exists())

    def test_docker_dir_structure(self):
        """Test Docker directory structure."""
        self.assertTrue((self.cli.docker_dir / "compose.ananke.yaml").exists())
        self.assertTrue((self.cli.docker_dir / "Dockerfile.ananke").exists())

    def test_helm_dir_structure(self):
        """Test Helm chart structure."""
        helm_chart = self.cli.helm_dir
        self.assertTrue((helm_chart / "Chart.yaml").exists())
        self.assertTrue((helm_chart / "values.yaml").exists())
        self.assertTrue((helm_chart / "templates").exists())

    def test_modal_dir_structure(self):
        """Test Modal directory structure."""
        self.assertTrue((self.cli.modal_dir / "sglang_ananke.py").exists())

    def test_aws_dir_structure(self):
        """Test AWS directory structure."""
        sagemaker_dir = self.cli.aws_dir / "sagemaker"
        self.assertTrue((sagemaker_dir / "deploy.py").exists())
        self.assertTrue((sagemaker_dir / "serve").exists())
        self.assertTrue((sagemaker_dir / "Dockerfile").exists())

    def test_gcp_dir_structure(self):
        """Test GCP directory structure."""
        vertexai_dir = self.cli.gcp_dir / "vertexai"
        self.assertTrue((vertexai_dir / "deploy.py").exists())


class TestDeployCLICommands(unittest.TestCase):
    """Test CLI command construction."""

    def setUp(self):
        """Set up test fixtures."""
        self.cli = DeployCLI()

    @patch('subprocess.run')
    def test_docker_build_command(self, mock_run):
        """Test Docker build command construction."""
        mock_run.return_value = MagicMock(returncode=0)

        self.cli.docker_build(
            variant="full",
            tag="test:latest",
            build_zig=False,
            no_cache=False,
        )

        # Verify subprocess was called
        self.assertTrue(mock_run.called)
        cmd = mock_run.call_args[0][0]
        self.assertIn("docker", cmd)
        self.assertIn("build", cmd)
        self.assertIn("-t", cmd)
        self.assertIn("test:latest", cmd)

    @patch('subprocess.run')
    def test_docker_build_with_zig(self, mock_run):
        """Test Docker build with Zig native library."""
        mock_run.return_value = MagicMock(returncode=0)

        self.cli.docker_build(
            variant="full",
            tag="test:native",
            build_zig=True,
            no_cache=False,
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--build-arg", cmd)
        self.assertIn("BUILD_ZIG_NATIVE=1", cmd)

    @patch('subprocess.run')
    def test_compose_up_command(self, mock_run):
        """Test Docker Compose up command."""
        mock_run.return_value = MagicMock(returncode=0)

        self.cli.compose_up(detach=True, with_monitoring=False, build=False)

        cmd = mock_run.call_args[0][0]
        self.assertIn("docker", cmd)
        self.assertIn("compose", cmd)
        self.assertIn("up", cmd)
        self.assertIn("-d", cmd)

    @patch('subprocess.run')
    def test_compose_up_with_monitoring(self, mock_run):
        """Test Docker Compose with monitoring overlay."""
        mock_run.return_value = MagicMock(returncode=0)

        self.cli.compose_up(detach=True, with_monitoring=True, build=False)

        cmd = mock_run.call_args[0][0]
        # Should have two -f flags for the overlay
        f_count = cmd.count("-f")
        self.assertEqual(f_count, 2)

    @patch('subprocess.run')
    def test_helm_install_command(self, mock_run):
        """Test Helm install command."""
        mock_run.return_value = MagicMock(returncode=0)

        self.cli.helm_install(
            release_name="test-release",
            namespace="default",
            values_file=None,
            dry_run=False,
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("helm", cmd)
        self.assertIn("install", cmd)
        self.assertIn("test-release", cmd)

    @patch('subprocess.run')
    def test_helm_install_with_values(self, mock_run):
        """Test Helm install with custom values."""
        mock_run.return_value = MagicMock(returncode=0)

        self.cli.helm_install(
            release_name="test-release",
            namespace="production",
            values_file="/path/to/values.yaml",
            dry_run=True,
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("-f", cmd)
        self.assertIn("/path/to/values.yaml", cmd)
        self.assertIn("--dry-run", cmd)
        self.assertIn("-n", cmd)
        self.assertIn("production", cmd)

    @patch('subprocess.run')
    def test_modal_deploy_command(self, mock_run):
        """Test Modal deploy command."""
        mock_run.return_value = MagicMock(returncode=0)

        self.cli.modal_deploy()

        cmd = mock_run.call_args[0][0]
        self.assertIn("modal", cmd)
        self.assertIn("deploy", cmd)


class TestHelmTemplates(unittest.TestCase):
    """Test Helm template files."""

    def setUp(self):
        """Set up test fixtures."""
        self.cli = DeployCLI()
        self.templates_dir = self.cli.helm_dir / "templates"

    def test_required_templates_exist(self):
        """Test that all required templates exist."""
        required = [
            "_helpers.tpl",
            "configmap.yaml",
            "deployment.yaml",
            "service.yaml",
            "serviceaccount.yaml",
        ]
        for template in required:
            self.assertTrue(
                (self.templates_dir / template).exists(),
                f"Missing template: {template}"
            )

    def test_optional_templates_exist(self):
        """Test that optional templates exist."""
        optional = [
            "ingress.yaml",
            "hpa.yaml",
            "pvc.yaml",
            "secret.yaml",
            "servicemonitor.yaml",
            "statefulset.yaml",
            "NOTES.txt",
        ]
        for template in optional:
            self.assertTrue(
                (self.templates_dir / template).exists(),
                f"Missing template: {template}"
            )

    def test_values_files_exist(self):
        """Test that values files exist."""
        values_files = [
            "values.yaml",
            "values-gcp.yaml",
            "values-aws.yaml",
        ]
        for values_file in values_files:
            self.assertTrue(
                (self.cli.helm_dir / values_file).exists(),
                f"Missing values file: {values_file}"
            )


if __name__ == '__main__':
    unittest.main()
