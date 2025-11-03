# Polyglot Microservices Example

Multi-language microservices architecture with FastAPI (Python) and Nest.js (TypeScript) services communicating via gRPC, with shared type definitions and Docker Compose orchestration.

## Architecture

```
┌─────────────────┐         gRPC         ┌─────────────────┐
│  FastAPI        │◄────────────────────►│  Nest.js        │
│  Service        │                       │  Service        │
│  (Python)       │                       │  (TypeScript)   │
└────────┬────────┘                       └────────┬────────┘
         │                                         │
         └──────────────┬──────────────────────────┘
                        │
                   PostgreSQL
```

## Features

- ✅ FastAPI service (Python) for data processing
- ✅ Nest.js service (TypeScript) for business logic
- ✅ gRPC for inter-service communication
- ✅ Shared Protocol Buffer definitions
- ✅ Docker Compose orchestration
- ✅ Centralized logging with Loki/Grafana
- ✅ Distributed tracing with Jaeger

## Project Structure

```
polyglot-microservices/
├── services/
│   ├── python-service/
│   │   ├── main.py              # FastAPI service
│   │   ├── grpc_client.py       # gRPC client
│   │   └── requirements.txt
│   └── typescript-service/
│       ├── src/
│       │   ├── main.ts          # Nest.js service
│       │   └── grpc.service.ts  # gRPC server
│       └── package.json
├── shared/
│   └── proto/
│       └── user.proto           # Shared Protocol Buffer
├── docker-compose.yml           # Service orchestration
└── README.md
```

## Quick Start

### 1. Start All Services

```bash
docker-compose up -d
```

Services will be available at:
- Python service: http://localhost:8000
- TypeScript service: http://localhost:3000
- PostgreSQL: localhost:5432
- Jaeger UI: http://localhost:16686

### 2. Test Inter-Service Communication

```bash
# Call TypeScript service (which calls Python service via gRPC)
curl http://localhost:3000/api/users/process/1
```

## Services

### Python Service (FastAPI)

**Port**: 8000

Endpoints:
- `GET /health` - Health check
- `POST /process` - Process data
- gRPC server on port 50051

### TypeScript Service (Nest.js)

**Port**: 3000

Endpoints:
- `GET /health` - Health check
- `GET /api/users/:id` - Get user
- `GET /api/users/process/:id` - Process user (calls Python service via gRPC)

## Development

### Python Service

```bash
cd services/python-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### TypeScript Service

```bash
cd services/typescript-service
pnpm install
pnpm start:dev
```

## Shared Types

Protocol Buffer definition (`shared/proto/user.proto`):

```protobuf
syntax = "proto3";

package user;

message User {
  int32 id = 1;
  string email = 2;
  string username = 3;
}

message ProcessRequest {
  int32 user_id = 1;
}

message ProcessResponse {
  User user = 1;
  string processed_data = 2;
}

service UserService {
  rpc ProcessUser(ProcessRequest) returns (ProcessResponse);
}
```

## Monitoring

### Distributed Tracing (Jaeger)

View traces at: http://localhost:16686

### Logging

Centralized logging with Grafana Loki

### Metrics

Prometheus metrics exposed on:
- Python service: http://localhost:8000/metrics
- TypeScript service: http://localhost:3000/metrics

## Production Deployment

### 1. Build Images

```bash
docker-compose build
```

### 2. Push to Registry

```bash
docker tag polyglot-python:latest registry.example.com/polyglot-python:latest
docker tag polyglot-typescript:latest registry.example.com/polyglot-typescript:latest

docker push registry.example.com/polyglot-python:latest
docker push registry.example.com/polyglot-typescript:latest
```

### 3. Deploy to Kubernetes

```bash
kubectl apply -f k8s/
```

## Key Patterns

- **gRPC Communication**: Type-safe inter-service calls
- **Shared Proto Definitions**: Single source of truth
- **Service Discovery**: Docker DNS within compose network
- **Health Checks**: Standardized health endpoints
- **Graceful Shutdown**: Proper signal handling

## Testing

```bash
# Python tests
cd services/python-service
pytest

# TypeScript tests
cd services/typescript-service
pnpm test

# Integration tests
bash test-integration.sh
```

---

**Lines**: ~280 (all services combined)
**Languages**: Python + TypeScript
**Frameworks**: FastAPI + Nest.js
**Communication**: gRPC
**Orchestration**: Docker Compose
