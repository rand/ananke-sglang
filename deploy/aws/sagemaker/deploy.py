#!/usr/bin/env python3
"""
Deploy SGLang Ananke to AWS SageMaker

This script handles:
- Building and pushing Docker images to ECR
- Creating SageMaker models
- Deploying endpoints with GPU instances
- Managing endpoint configurations

Usage:
    python deploy.py create-model --model-name my-model --image-uri $ECR_URI
    python deploy.py deploy --endpoint-name my-endpoint --model-name my-model
    python deploy.py update --endpoint-name my-endpoint --model-name new-model
    python deploy.py delete --endpoint-name my-endpoint

Environment Variables:
    AWS_REGION - AWS region (default: us-west-2)
    AWS_ACCOUNT_ID - AWS account ID (auto-detected if not set)
    SAGEMAKER_ROLE - SageMaker execution role ARN
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("Please install boto3: pip install boto3")
    sys.exit(1)


class SageMakerDeployer:
    """Deploy SGLang Ananke to SageMaker."""

    def __init__(
        self,
        region: str = "us-west-2",
        role_arn: Optional[str] = None,
    ):
        self.region = region
        self.sm_client = boto3.client("sagemaker", region_name=region)
        self.ecr_client = boto3.client("ecr", region_name=region)
        self.sts_client = boto3.client("sts", region_name=region)

        # Get account ID
        self.account_id = os.getenv("AWS_ACCOUNT_ID")
        if not self.account_id:
            self.account_id = self.sts_client.get_caller_identity()["Account"]

        # Get or create role
        self.role_arn = role_arn or os.getenv("SAGEMAKER_ROLE")
        if not self.role_arn:
            self.role_arn = f"arn:aws:iam::{self.account_id}:role/SageMakerExecutionRole"

    def create_ecr_repository(self, repo_name: str) -> str:
        """Create ECR repository if it doesn't exist."""
        try:
            response = self.ecr_client.create_repository(
                repositoryName=repo_name,
                imageScanningConfiguration={"scanOnPush": True},
            )
            print(f"Created ECR repository: {repo_name}")
            return response["repository"]["repositoryUri"]
        except ClientError as e:
            if e.response["Error"]["Code"] == "RepositoryAlreadyExistsException":
                response = self.ecr_client.describe_repositories(
                    repositoryNames=[repo_name]
                )
                return response["repositories"][0]["repositoryUri"]
            raise

    def create_model(
        self,
        model_name: str,
        image_uri: str,
        model_data_url: Optional[str] = None,
        environment: Optional[dict] = None,
    ) -> str:
        """Create SageMaker model."""
        container_def = {
            "Image": image_uri,
            "Mode": "SingleModel",
        }

        if model_data_url:
            container_def["ModelDataUrl"] = model_data_url

        if environment:
            container_def["Environment"] = environment
        else:
            # Default Ananke configuration
            container_def["Environment"] = {
                "SGLANG_GRAMMAR_BACKEND": "ananke",
                "ANANKE_LANGUAGE": "python",
                "ANANKE_MAX_ROLLBACK_TOKENS": "200",
            }

        try:
            self.sm_client.create_model(
                ModelName=model_name,
                PrimaryContainer=container_def,
                ExecutionRoleArn=self.role_arn,
            )
            print(f"Created model: {model_name}")
            return model_name
        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                if "already exists" in str(e):
                    print(f"Model already exists: {model_name}")
                    return model_name
            raise

    def create_endpoint_config(
        self,
        config_name: str,
        model_name: str,
        instance_type: str = "ml.g5.xlarge",
        instance_count: int = 1,
    ) -> str:
        """Create endpoint configuration."""
        try:
            self.sm_client.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        "VariantName": "primary",
                        "ModelName": model_name,
                        "InstanceType": instance_type,
                        "InitialInstanceCount": instance_count,
                        "ContainerStartupHealthCheckTimeoutInSeconds": 600,
                        "ModelDataDownloadTimeoutInSeconds": 600,
                    }
                ],
            )
            print(f"Created endpoint config: {config_name}")
            return config_name
        except ClientError as e:
            if "already exists" in str(e):
                print(f"Endpoint config already exists: {config_name}")
                return config_name
            raise

    def deploy_endpoint(
        self,
        endpoint_name: str,
        config_name: str,
        wait: bool = True,
    ) -> str:
        """Deploy SageMaker endpoint."""
        try:
            self.sm_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name,
            )
            print(f"Creating endpoint: {endpoint_name}")
        except ClientError as e:
            if "already exists" in str(e):
                print(f"Updating existing endpoint: {endpoint_name}")
                self.sm_client.update_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=config_name,
                )
            else:
                raise

        if wait:
            self._wait_for_endpoint(endpoint_name)

        return endpoint_name

    def _wait_for_endpoint(self, endpoint_name: str, timeout: int = 1800):
        """Wait for endpoint to be in service."""
        print("Waiting for endpoint to be in service...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
            status = response["EndpointStatus"]

            if status == "InService":
                print(f"Endpoint {endpoint_name} is in service!")
                return
            elif status == "Failed":
                raise RuntimeError(f"Endpoint creation failed: {response.get('FailureReason')}")

            print(f"Status: {status}...")
            time.sleep(30)

        raise TimeoutError(f"Endpoint did not become ready within {timeout}s")

    def delete_endpoint(self, endpoint_name: str, delete_config: bool = True):
        """Delete endpoint and optionally its configuration."""
        try:
            # Get config name before deleting endpoint
            response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
            config_name = response["EndpointConfigName"]

            # Delete endpoint
            self.sm_client.delete_endpoint(EndpointName=endpoint_name)
            print(f"Deleted endpoint: {endpoint_name}")

            # Wait for deletion
            print("Waiting for endpoint deletion...")
            while True:
                try:
                    self.sm_client.describe_endpoint(EndpointName=endpoint_name)
                    time.sleep(10)
                except ClientError:
                    break

            # Delete config
            if delete_config:
                self.sm_client.delete_endpoint_config(EndpointConfigName=config_name)
                print(f"Deleted endpoint config: {config_name}")

        except ClientError as e:
            if "Could not find endpoint" in str(e):
                print(f"Endpoint not found: {endpoint_name}")
            else:
                raise

    def invoke_endpoint(
        self,
        endpoint_name: str,
        payload: dict,
    ) -> dict:
        """Invoke endpoint for testing."""
        runtime_client = boto3.client(
            "sagemaker-runtime",
            region_name=self.region,
        )

        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        return json.loads(response["Body"].read().decode())


def main():
    parser = argparse.ArgumentParser(description="Deploy SGLang Ananke to SageMaker")
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    parser.add_argument("--role", help="SageMaker execution role ARN")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Create model
    create_model_parser = subparsers.add_parser("create-model", help="Create SageMaker model")
    create_model_parser.add_argument("--model-name", required=True)
    create_model_parser.add_argument("--image-uri", required=True)
    create_model_parser.add_argument("--model-data", help="S3 URI for model data")

    # Deploy
    deploy_parser = subparsers.add_parser("deploy", help="Deploy endpoint")
    deploy_parser.add_argument("--endpoint-name", required=True)
    deploy_parser.add_argument("--model-name", required=True)
    deploy_parser.add_argument("--instance-type", default="ml.g5.xlarge")
    deploy_parser.add_argument("--instance-count", type=int, default=1)
    deploy_parser.add_argument("--no-wait", action="store_true")

    # Update
    update_parser = subparsers.add_parser("update", help="Update endpoint")
    update_parser.add_argument("--endpoint-name", required=True)
    update_parser.add_argument("--model-name", required=True)
    update_parser.add_argument("--instance-type", default="ml.g5.xlarge")

    # Delete
    delete_parser = subparsers.add_parser("delete", help="Delete endpoint")
    delete_parser.add_argument("--endpoint-name", required=True)

    # Test
    test_parser = subparsers.add_parser("test", help="Test endpoint")
    test_parser.add_argument("--endpoint-name", required=True)
    test_parser.add_argument("--prompt", default="def hello():")

    args = parser.parse_args()

    deployer = SageMakerDeployer(region=args.region, role_arn=args.role)

    if args.command == "create-model":
        deployer.create_model(
            model_name=args.model_name,
            image_uri=args.image_uri,
            model_data_url=args.model_data,
        )

    elif args.command == "deploy":
        config_name = f"{args.endpoint_name}-config"
        deployer.create_endpoint_config(
            config_name=config_name,
            model_name=args.model_name,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
        )
        deployer.deploy_endpoint(
            endpoint_name=args.endpoint_name,
            config_name=config_name,
            wait=not args.no_wait,
        )

    elif args.command == "update":
        config_name = f"{args.endpoint_name}-config-{int(time.time())}"
        deployer.create_endpoint_config(
            config_name=config_name,
            model_name=args.model_name,
            instance_type=args.instance_type,
        )
        deployer.deploy_endpoint(
            endpoint_name=args.endpoint_name,
            config_name=config_name,
        )

    elif args.command == "delete":
        deployer.delete_endpoint(args.endpoint_name)

    elif args.command == "test":
        result = deployer.invoke_endpoint(
            endpoint_name=args.endpoint_name,
            payload={
                "prompt": args.prompt,
                "max_tokens": 50,
            },
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
