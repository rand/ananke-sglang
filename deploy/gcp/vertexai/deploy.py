#!/usr/bin/env python3
"""
Deploy SGLang Ananke to GCP Vertex AI

This script handles:
- Building and pushing Docker images to Artifact Registry
- Creating Vertex AI model resources
- Deploying endpoints with GPU accelerators
- Managing endpoint configurations

Usage:
    python deploy.py upload-model --model-name sglang-ananke --image-uri $AR_URI
    python deploy.py deploy --endpoint-name sglang-ananke --model-name sglang-ananke
    python deploy.py delete --endpoint-name sglang-ananke

Environment Variables:
    GOOGLE_CLOUD_PROJECT - GCP project ID
    GOOGLE_CLOUD_REGION - GCP region (default: us-central1)
    VERTEX_AI_STAGING_BUCKET - GCS bucket for staging
"""

import argparse
import os
import sys
import time
from typing import Optional

try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform import gapic
except ImportError:
    print("Please install google-cloud-aiplatform: pip install google-cloud-aiplatform")
    sys.exit(1)


class VertexAIDeployer:
    """Deploy SGLang Ananke to Vertex AI."""

    # GPU accelerator types available in Vertex AI
    ACCELERATOR_TYPES = {
        "a100-40gb": "NVIDIA_TESLA_A100",
        "a100-80gb": "NVIDIA_A100_80GB",
        "t4": "NVIDIA_TESLA_T4",
        "v100": "NVIDIA_TESLA_V100",
        "l4": "NVIDIA_L4",
    }

    # Machine types for GPU instances
    MACHINE_TYPES = {
        "a100-40gb": "a2-highgpu-1g",
        "a100-80gb": "a2-ultragpu-1g",
        "t4": "n1-standard-8",
        "v100": "n1-standard-8",
        "l4": "g2-standard-8",
    }

    def __init__(
        self,
        project: Optional[str] = None,
        region: str = "us-central1",
        staging_bucket: Optional[str] = None,
    ):
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project:
            raise ValueError("Project ID required (set GOOGLE_CLOUD_PROJECT)")

        self.region = region
        self.staging_bucket = staging_bucket or os.getenv("VERTEX_AI_STAGING_BUCKET")

        # Initialize Vertex AI
        aiplatform.init(
            project=self.project,
            location=self.region,
            staging_bucket=self.staging_bucket,
        )

    def upload_model(
        self,
        model_name: str,
        image_uri: str,
        model_description: str = "SGLang with Ananke constrained generation backend",
        serving_container_predict_route: str = "/v1/completions",
        serving_container_health_route: str = "/health",
        serving_container_ports: list = None,
        environment_variables: dict = None,
    ) -> aiplatform.Model:
        """Upload model to Vertex AI Model Registry."""

        if serving_container_ports is None:
            serving_container_ports = [30000]

        if environment_variables is None:
            environment_variables = {
                "SGLANG_GRAMMAR_BACKEND": "ananke",
                "ANANKE_LANGUAGE": "python",
                "ANANKE_MAX_ROLLBACK_TOKENS": "200",
            }

        print(f"Uploading model: {model_name}")
        model = aiplatform.Model.upload(
            display_name=model_name,
            description=model_description,
            serving_container_image_uri=image_uri,
            serving_container_predict_route=serving_container_predict_route,
            serving_container_health_route=serving_container_health_route,
            serving_container_ports=serving_container_ports,
            serving_container_environment_variables=environment_variables,
        )

        print(f"Model uploaded: {model.resource_name}")
        return model

    def create_endpoint(
        self,
        endpoint_name: str,
        description: str = "SGLang Ananke endpoint",
    ) -> aiplatform.Endpoint:
        """Create Vertex AI endpoint."""
        print(f"Creating endpoint: {endpoint_name}")
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_name,
            description=description,
        )
        print(f"Endpoint created: {endpoint.resource_name}")
        return endpoint

    def deploy_model(
        self,
        endpoint: aiplatform.Endpoint,
        model: aiplatform.Model,
        gpu_type: str = "l4",
        gpu_count: int = 1,
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        machine_type: Optional[str] = None,
        traffic_percentage: int = 100,
        deployed_model_display_name: Optional[str] = None,
    ) -> None:
        """Deploy model to endpoint."""

        accelerator_type = self.ACCELERATOR_TYPES.get(gpu_type)
        if not accelerator_type:
            raise ValueError(f"Unknown GPU type: {gpu_type}. Options: {list(self.ACCELERATOR_TYPES.keys())}")

        if machine_type is None:
            machine_type = self.MACHINE_TYPES.get(gpu_type, "n1-standard-8")

        print(f"Deploying model to endpoint...")
        print(f"  Machine type: {machine_type}")
        print(f"  Accelerator: {accelerator_type} x {gpu_count}")
        print(f"  Replicas: {min_replica_count}-{max_replica_count}")

        endpoint.deploy(
            model=model,
            deployed_model_display_name=deployed_model_display_name,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=gpu_count,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            traffic_percentage=traffic_percentage,
            sync=True,
        )

        print("Deployment complete!")

    def undeploy_model(
        self,
        endpoint: aiplatform.Endpoint,
        deployed_model_id: Optional[str] = None,
    ) -> None:
        """Undeploy model from endpoint."""
        if deployed_model_id:
            endpoint.undeploy(deployed_model_id=deployed_model_id)
        else:
            endpoint.undeploy_all()
        print("Model undeployed")

    def delete_endpoint(
        self,
        endpoint_name: str,
        force: bool = False,
    ) -> None:
        """Delete endpoint."""
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_name}"'
        )

        if not endpoints:
            print(f"Endpoint not found: {endpoint_name}")
            return

        endpoint = endpoints[0]

        if force:
            # Undeploy all models first
            self.undeploy_model(endpoint)

        endpoint.delete()
        print(f"Endpoint deleted: {endpoint_name}")

    def predict(
        self,
        endpoint_name: str,
        instances: list,
    ) -> list:
        """Send prediction request to endpoint."""
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_name}"'
        )

        if not endpoints:
            raise ValueError(f"Endpoint not found: {endpoint_name}")

        endpoint = endpoints[0]
        response = endpoint.predict(instances=instances)
        return response.predictions

    def get_model_by_name(self, model_name: str) -> Optional[aiplatform.Model]:
        """Get model by display name."""
        models = aiplatform.Model.list(
            filter=f'display_name="{model_name}"'
        )
        return models[0] if models else None

    def get_endpoint_by_name(self, endpoint_name: str) -> Optional[aiplatform.Endpoint]:
        """Get endpoint by display name."""
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_name}"'
        )
        return endpoints[0] if endpoints else None


def main():
    parser = argparse.ArgumentParser(description="Deploy SGLang Ananke to Vertex AI")
    parser.add_argument("--project", help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--staging-bucket", help="GCS staging bucket")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Upload model
    upload_parser = subparsers.add_parser("upload-model", help="Upload model to registry")
    upload_parser.add_argument("--model-name", required=True)
    upload_parser.add_argument("--image-uri", required=True)
    upload_parser.add_argument("--description", default="SGLang with Ananke backend")

    # Create endpoint
    create_ep_parser = subparsers.add_parser("create-endpoint", help="Create endpoint")
    create_ep_parser.add_argument("--endpoint-name", required=True)

    # Deploy
    deploy_parser = subparsers.add_parser("deploy", help="Deploy model to endpoint")
    deploy_parser.add_argument("--endpoint-name", required=True)
    deploy_parser.add_argument("--model-name", required=True)
    deploy_parser.add_argument("--gpu-type", default="l4", choices=VertexAIDeployer.ACCELERATOR_TYPES.keys())
    deploy_parser.add_argument("--gpu-count", type=int, default=1)
    deploy_parser.add_argument("--min-replicas", type=int, default=1)
    deploy_parser.add_argument("--max-replicas", type=int, default=1)
    deploy_parser.add_argument("--create-endpoint", action="store_true", help="Create endpoint if not exists")

    # Delete
    delete_parser = subparsers.add_parser("delete", help="Delete endpoint")
    delete_parser.add_argument("--endpoint-name", required=True)
    delete_parser.add_argument("--force", action="store_true", help="Force delete with deployed models")

    # Test
    test_parser = subparsers.add_parser("test", help="Test endpoint")
    test_parser.add_argument("--endpoint-name", required=True)
    test_parser.add_argument("--prompt", default="def hello():")

    args = parser.parse_args()

    deployer = VertexAIDeployer(
        project=args.project,
        region=args.region,
        staging_bucket=args.staging_bucket,
    )

    if args.command == "upload-model":
        deployer.upload_model(
            model_name=args.model_name,
            image_uri=args.image_uri,
            model_description=args.description,
        )

    elif args.command == "create-endpoint":
        deployer.create_endpoint(endpoint_name=args.endpoint_name)

    elif args.command == "deploy":
        # Get or create endpoint
        endpoint = deployer.get_endpoint_by_name(args.endpoint_name)
        if not endpoint:
            if args.create_endpoint:
                endpoint = deployer.create_endpoint(args.endpoint_name)
            else:
                print(f"Endpoint not found: {args.endpoint_name}")
                print("Use --create-endpoint to create it automatically")
                sys.exit(1)

        # Get model
        model = deployer.get_model_by_name(args.model_name)
        if not model:
            print(f"Model not found: {args.model_name}")
            sys.exit(1)

        # Deploy
        deployer.deploy_model(
            endpoint=endpoint,
            model=model,
            gpu_type=args.gpu_type,
            gpu_count=args.gpu_count,
            min_replica_count=args.min_replicas,
            max_replica_count=args.max_replicas,
        )

    elif args.command == "delete":
        deployer.delete_endpoint(
            endpoint_name=args.endpoint_name,
            force=args.force,
        )

    elif args.command == "test":
        result = deployer.predict(
            endpoint_name=args.endpoint_name,
            instances=[{
                "prompt": args.prompt,
                "max_tokens": 50,
            }],
        )
        print(result)


if __name__ == "__main__":
    main()
