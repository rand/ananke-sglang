# SGLang Ananke SageMaker Container
# For deployment on AWS SageMaker endpoints
#
# Build:
#   docker build -t sglang-ananke-sagemaker .
#
# Push to ECR:
#   aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REPO
#   docker tag sglang-ananke-sagemaker:latest $ECR_REPO/sglang-ananke-sagemaker:latest
#   docker push $ECR_REPO/sglang-ananke-sagemaker:latest

ARG BASE_IMAGE=lmsysorg/sglang:latest

FROM ${BASE_IMAGE}

# Install Ananke optional dependencies
RUN pip install --no-cache-dir \
    "z3-solver>=4.12.0" \
    "tree-sitter>=0.22.0" \
    "immutables>=0.20" \
    boto3 \
    sagemaker

# SageMaker requires specific directory structure
ENV SAGEMAKER_PROGRAM=serve \
    SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code \
    SAGEMAKER_CONTAINER_LOG_LEVEL=20 \
    PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PATH="/opt/ml/code:${PATH}"

# Create required directories
RUN mkdir -p /opt/ml/code /opt/ml/model /opt/ml/input/data

# Copy serve script
COPY serve /opt/ml/code/serve
RUN chmod +x /opt/ml/code/serve

# Copy inference script
COPY inference.py /opt/ml/code/inference.py

# Default environment for Ananke
ENV SGLANG_GRAMMAR_BACKEND=ananke \
    ANANKE_LANGUAGE=python \
    ANANKE_MAX_ROLLBACK_TOKENS=200

# SageMaker uses port 8080
EXPOSE 8080

WORKDIR /opt/ml/code

# Health check for SageMaker
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--grammar-backend", "ananke"]
