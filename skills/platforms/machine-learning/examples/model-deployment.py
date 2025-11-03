#!/usr/bin/env python3
"""
Complete Model Deployment Example
Demonstrates production ML deployment with:
- Model serving via FastAPI
- Model versioning and registry
- A/B testing support
- Monitoring and logging
- Docker containerization
- Performance optimization
"""

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pickle
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: List[float]
    model_version: Optional[str] = "v1"


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int
    probability: float
    model_version: str
    timestamp: str
    confidence: float


class ModelMetadata(BaseModel):
    """Model metadata"""
    version: str
    created_at: str
    accuracy: float
    metrics: Dict[str, float]


# Model Registry
class ModelRegistry:
    """
    Manages multiple model versions
    Supports A/B testing and rollback
    """

    def __init__(self, registry_path: str = "models/registry"):
        """
        Initialize model registry

        Args:
            registry_path: Path to model registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.preprocessors = {}
        self.metadata = {}
        self.active_version = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def register_model(
        self,
        model: nn.Module,
        preprocessor: Any,
        version: str,
        metadata: Dict[str, Any]
    ):
        """
        Register a new model version

        Args:
            model: PyTorch model
            preprocessor: Data preprocessor
            version: Version string (e.g., "v1", "v2")
            metadata: Model metadata (accuracy, metrics, etc.)
        """
        # Save model
        model_path = self.registry_path / f"model_{version}.pth"
        torch.save(model.state_dict(), model_path)

        # Save preprocessor
        preprocessor_path = self.registry_path / f"preprocessor_{version}.pkl"
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)

        # Save metadata
        metadata_path = self.registry_path / f"metadata_{version}.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Load into memory
        self.models[version] = model
        self.preprocessors[version] = preprocessor
        self.metadata[version] = metadata

        # Set as active if first model
        if self.active_version is None:
            self.active_version = version

        self.logger.info(f"Registered model version: {version}")

    def get_model(self, version: Optional[str] = None) -> tuple:
        """
        Get model and preprocessor by version

        Args:
            version: Model version (uses active if None)

        Returns:
            Tuple of (model, preprocessor, metadata)
        """
        version = version or self.active_version

        if version not in self.models:
            raise ValueError(f"Model version {version} not found")

        return (
            self.models[version],
            self.preprocessors[version],
            self.metadata[version]
        )

    def set_active_version(self, version: str):
        """Set active model version"""
        if version not in self.models:
            raise ValueError(f"Model version {version} not found")

        self.active_version = version
        self.logger.info(f"Active model version set to: {version}")

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all registered model versions"""
        return [
            {
                "version": version,
                "is_active": version == self.active_version,
                **metadata
            }
            for version, metadata in self.metadata.items()
        ]


# Example Model for Deployment
class SimpleClassifier(nn.Module):
    """Simple neural network for classification"""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def predict_proba(self, x):
        """Get probabilities"""
        logits = self.forward(x)
        return self.softmax(logits)


# FastAPI Application
def create_app(registry: ModelRegistry) -> FastAPI:
    """
    Create FastAPI application for model serving

    Args:
        registry: Model registry instance

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="ML Model API",
        description="Production ML model serving with versioning and A/B testing",
        version="1.0.0"
    )

    # Request counter for A/B testing
    request_counter = {"count": 0}

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "ML Model API",
            "status": "active",
            "active_version": registry.active_version
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "active_model": registry.active_version
        }

    @app.get("/models")
    async def list_models():
        """List all registered models"""
        return {
            "models": registry.list_versions()
        }

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """
        Make prediction

        Args:
            request: Prediction request with features

        Returns:
            Prediction response with class and probability
        """
        try:
            # Get model (support version routing)
            model, preprocessor, metadata = registry.get_model(request.model_version)

            # Preprocess features
            features = np.array(request.features).reshape(1, -1)

            # Convert to tensor
            features_tensor = torch.FloatTensor(features)

            # Make prediction
            model.eval()
            with torch.no_grad():
                probabilities = model.predict_proba(features_tensor)
                predicted_class = probabilities.argmax(dim=1).item()
                confidence = probabilities.max(dim=1).values.item()

            # Log prediction
            logging.info(
                f"Prediction: class={predicted_class}, "
                f"confidence={confidence:.4f}, "
                f"version={request.model_version}"
            )

            return PredictionResponse(
                prediction=predicted_class,
                probability=confidence,
                model_version=request.model_version,
                timestamp=datetime.now().isoformat(),
                confidence=confidence
            )

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/ab-test")
    async def predict_ab_test(request: PredictionRequest):
        """
        Prediction with A/B testing
        Routes 50% of traffic to each model version
        """
        request_counter["count"] += 1

        # A/B split: even requests to v1, odd to v2
        if request_counter["count"] % 2 == 0:
            request.model_version = "v1"
        else:
            request.model_version = "v2" if "v2" in registry.models else "v1"

        return await predict(request)

    @app.post("/models/activate/{version}")
    async def activate_model(version: str):
        """Activate a specific model version"""
        try:
            registry.set_active_version(version)
            return {
                "message": f"Activated model version: {version}",
                "active_version": registry.active_version
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/metrics")
    async def get_metrics():
        """Get model metrics"""
        return {
            "models": {
                version: metadata.get("metrics", {})
                for version, metadata in registry.metadata.items()
            }
        }

    return app


# Deployment Examples
def example_local_deployment():
    """
    Example 1: Local deployment
    Deploy model locally with FastAPI
    """
    print("=" * 80)
    print("EXAMPLE 1: Local Model Deployment")
    print("=" * 80)

    # Initialize registry
    registry = ModelRegistry()

    # Create and register model v1
    model_v1 = SimpleClassifier(input_dim=10, hidden_dim=20, num_classes=2)
    preprocessor_v1 = {"scaler": "standard", "version": "v1"}

    registry.register_model(
        model=model_v1,
        preprocessor=preprocessor_v1,
        version="v1",
        metadata={
            "created_at": datetime.now().isoformat(),
            "accuracy": 0.92,
            "metrics": {
                "precision": 0.91,
                "recall": 0.93,
                "f1": 0.92
            }
        }
    )

    # Create and register model v2 (improved)
    model_v2 = SimpleClassifier(input_dim=10, hidden_dim=30, num_classes=2)
    preprocessor_v2 = {"scaler": "standard", "version": "v2"}

    registry.register_model(
        model=model_v2,
        preprocessor=preprocessor_v2,
        version="v2",
        metadata={
            "created_at": datetime.now().isoformat(),
            "accuracy": 0.95,
            "metrics": {
                "precision": 0.94,
                "recall": 0.96,
                "f1": 0.95
            }
        }
    )

    print("\nRegistered models:")
    for model_info in registry.list_versions():
        print(f"  {model_info['version']}: accuracy={model_info['accuracy']:.2%}")

    # Create FastAPI app
    app = create_app(registry)

    print("\nFastAPI application created")
    print("To run the server, execute:")
    print("  uvicorn model-deployment:app --host 0.0.0.0 --port 8000")
    print("\nAPI Documentation available at:")
    print("  http://localhost:8000/docs")


def example_docker_deployment():
    """
    Example 2: Docker deployment
    Generate Dockerfile and deployment configs
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Docker Deployment Configuration")
    print("=" * 80)

    # Create deployment directory
    deploy_dir = Path("deployment")
    deploy_dir.mkdir(exist_ok=True)

    # Generate Dockerfile
    dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "model-deployment:app", "--host", "0.0.0.0", "--port", "8000"]
"""

    with open(deploy_dir / "Dockerfile", "w") as f:
        f.write(dockerfile_content)

    # Generate requirements.txt
    requirements_content = """
torch==2.0.0
fastapi==0.100.0
uvicorn==0.23.0
pydantic==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
"""

    with open(deploy_dir / "requirements.txt", "w") as f:
        f.write(requirements_content)

    # Generate docker-compose.yml
    docker_compose_content = """
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_VERSION=v1
      - LOG_LEVEL=info
    volumes:
      - ./models:/app/models
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped
"""

    with open(deploy_dir / "docker-compose.yml", "w") as f:
        f.write(docker_compose_content)

    print("\nDeployment files created in 'deployment/' directory:")
    print("  - Dockerfile")
    print("  - requirements.txt")
    print("  - docker-compose.yml")

    print("\nTo build and run:")
    print("  cd deployment/")
    print("  docker-compose up -d")


def example_kubernetes_deployment():
    """
    Example 3: Kubernetes deployment
    Generate K8s manifests for production deployment
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Kubernetes Deployment Manifests")
    print("=" * 80)

    # Create k8s directory
    k8s_dir = Path("k8s")
    k8s_dir.mkdir(exist_ok=True)

    # Generate deployment manifest
    deployment_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-api
  labels:
    app: ml-model-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model-api
  template:
    metadata:
      labels:
        app: ml-model-api
    spec:
      containers:
      - name: api
        image: ml-model-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
"""

    with open(k8s_dir / "deployment.yaml", "w") as f:
        f.write(deployment_yaml)

    # Generate service manifest
    service_yaml = """
apiVersion: v1
kind: Service
metadata:
  name: ml-model-api-service
spec:
  selector:
    app: ml-model-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
"""

    with open(k8s_dir / "service.yaml", "w") as f:
        f.write(service_yaml)

    print("\nKubernetes manifests created in 'k8s/' directory:")
    print("  - deployment.yaml")
    print("  - service.yaml")

    print("\nTo deploy:")
    print("  kubectl apply -f k8s/")


# Main function
def main():
    """
    Run all deployment examples
    Demonstrates different deployment approaches
    """
    print("\n" + "#" * 80)
    print("# MODEL DEPLOYMENT EXAMPLES")
    print("#" * 80)

    try:
        # Example 1: Local deployment
        example_local_deployment()

        # Example 2: Docker deployment
        example_docker_deployment()

        # Example 3: Kubernetes deployment
        example_kubernetes_deployment()

        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during deployment setup: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
