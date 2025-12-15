#!/usr/bin/env python3
"""
SGLang health check utilities for deployment.

Provides health check functions for use in deployment scripts,
Kubernetes probes, and monitoring systems.

Usage:
    python healthcheck.py [--url URL] [--timeout TIMEOUT] [--check CHECK]

Examples:
    python healthcheck.py --check liveness
    python healthcheck.py --check readiness --timeout 30
    python healthcheck.py --url http://localhost:30000 --check full
"""

import argparse
import json
import sys
import time
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


def check_health(base_url: str, timeout: int = 10) -> bool:
    """Basic liveness check - server is responding."""
    try:
        req = Request(f"{base_url}/health", method="GET")
        with urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except (URLError, TimeoutError, ValueError, OSError):
        return False


def check_health_generate(base_url: str, timeout: int = 30) -> bool:
    """Readiness check - model is loaded and ready to generate."""
    try:
        req = Request(f"{base_url}/health_generate", method="GET")
        with urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except (URLError, TimeoutError, ValueError, OSError):
        return False


def check_models(base_url: str, timeout: int = 10) -> Optional[dict]:
    """Check available models endpoint."""
    try:
        req = Request(f"{base_url}/v1/models", method="GET")
        with urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                return json.loads(response.read().decode())
    except (URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError):
        pass
    return None


def check_ananke(base_url: str, timeout: int = 30) -> bool:
    """Check Ananke backend is functional with a simple constraint test."""
    try:
        payload = json.dumps({
            "model": "default",
            "messages": [{"role": "user", "content": "def hello():"}],
            "max_tokens": 10,
            "extra_body": {
                "constraint_spec": {
                    "language": "python",
                    "domains": ["syntax"]
                }
            }
        }).encode()

        req = Request(
            f"{base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except (URLError, TimeoutError, ValueError, OSError):
        return False


def full_health_check(
    base_url: str,
    timeout: int = 30,
    check_ananke_backend: bool = True
) -> dict:
    """Comprehensive health check with detailed status."""
    results = {
        "healthy": True,
        "checks": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    # Liveness
    results["checks"]["liveness"] = check_health(base_url, timeout)
    if not results["checks"]["liveness"]:
        results["healthy"] = False
        return results

    # Readiness
    results["checks"]["readiness"] = check_health_generate(base_url, timeout)
    if not results["checks"]["readiness"]:
        results["healthy"] = False

    # Models
    models = check_models(base_url, timeout)
    results["checks"]["models"] = models is not None
    if models:
        results["models"] = [m.get("id") for m in models.get("data", [])]

    # Ananke (optional)
    if check_ananke_backend:
        results["checks"]["ananke"] = check_ananke(base_url, timeout)

    return results


def wait_for_ready(
    base_url: str,
    timeout: int = 300,
    interval: int = 5
) -> bool:
    """Wait for server to become ready."""
    start = time.time()
    while time.time() - start < timeout:
        if check_health_generate(base_url, timeout=10):
            return True
        time.sleep(interval)
    return False


def main():
    parser = argparse.ArgumentParser(description="SGLang health check utilities")
    parser.add_argument(
        "--url",
        default="http://localhost:30000",
        help="Base URL of SGLang server"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--check",
        choices=["liveness", "readiness", "models", "ananke", "full", "wait"],
        default="liveness",
        help="Type of health check to perform"
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=300,
        help="Timeout for wait check in seconds"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    if args.check == "liveness":
        result = check_health(args.url, args.timeout)
        if args.json:
            print(json.dumps({"healthy": result}))
        sys.exit(0 if result else 1)

    elif args.check == "readiness":
        result = check_health_generate(args.url, args.timeout)
        if args.json:
            print(json.dumps({"ready": result}))
        sys.exit(0 if result else 1)

    elif args.check == "models":
        result = check_models(args.url, args.timeout)
        if args.json:
            print(json.dumps(result or {"error": "failed"}))
        else:
            if result:
                for model in result.get("data", []):
                    print(model.get("id", "unknown"))
        sys.exit(0 if result else 1)

    elif args.check == "ananke":
        result = check_ananke(args.url, args.timeout)
        if args.json:
            print(json.dumps({"ananke_functional": result}))
        sys.exit(0 if result else 1)

    elif args.check == "full":
        result = full_health_check(args.url, args.timeout)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["healthy"] else 1)

    elif args.check == "wait":
        result = wait_for_ready(args.url, args.wait_timeout)
        if args.json:
            print(json.dumps({"ready": result}))
        else:
            print("Ready" if result else "Timeout waiting for server")
        sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
