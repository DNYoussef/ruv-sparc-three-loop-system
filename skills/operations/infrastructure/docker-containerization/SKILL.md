---
name: docker-containerization
description: Docker containerization specialist for multi-stage builds, layer caching optimization, security scanning with Trivy, Docker Compose orchestration, BuildKit advanced features, and production-grade Dockerfile best practices. Use when containerizing applications, optimizing image size, implementing CI/CD pipelines, or requiring Docker best practices. Handles secrets management, health checks, resource limits, and registry operations.
category: Infrastructure
complexity: Medium
triggers: ["docker", "dockerfile", "docker compose", "containerization", "docker build", "buildkit", "trivy", "docker security", "multi-stage build"]
---

# Docker Containerization Specialist

Expert Docker containerization for production-grade, secure, and optimized container images.

## Purpose

Comprehensive Docker expertise including multi-stage builds, layer caching, security scanning, Docker Compose, BuildKit features, and best practices. Ensures containers are small, fast, secure, and production-ready.

## When to Use

- Creating optimized Dockerfiles
- Implementing multi-stage builds
- Optimizing build caching
- Scanning images for vulnerabilities
- Orchestrating multi-container apps with Docker Compose
- Implementing CI/CD with Docker
- Troubleshooting container performance

## Prerequisites

**Required**: Basic Docker commands, understanding of containers vs VMs

**Agents**: `cicd-engineer`, `security-manager`, `code-analyzer`, `backend-dev`

## Core Workflows

### Workflow 1: Multi-Stage Node.js Build

```dockerfile
# syntax=docker/dockerfile:1
# Stage 1: Dependencies
FROM node:18-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Stage 2: Build
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 3: Production
FROM node:18-alpine AS runner
WORKDIR /app

# Security: Run as non-root
RUN addgroup -g 1001 -S nodejs && adduser -S nodejs -u 1001

# Copy only necessary files
COPY --from=deps --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --chown=nodejs:nodejs package.json ./

USER nodejs
EXPOSE 3000
ENV NODE_ENV=production

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/health', (res) => { process.exit(res.statusCode === 200 ? 0 : 1); })"

CMD ["node", "dist/index.js"]
```

### Workflow 2: Python Multi-Stage Build

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Security: Run as non-root
RUN useradd -m -u 1001 appuser
USER appuser

# Add .local/bin to PATH
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Workflow 3: Docker Compose Multi-Service App

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
      target: runner
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://user:password@db:5432/mydb
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    networks:
      - app-network
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydb
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - app-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    networks:
      - app-network
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data

networks:
  app-network:
    driver: bridge

volumes:
  postgres-data:
  redis-data:
```

### Workflow 4: BuildKit Advanced Features

**Enable BuildKit**

```bash
# Set environment variable
export DOCKER_BUILDKIT=1

# Or in daemon.json
{
  "features": {
    "buildkit": true
  }
}
```

**Cache Mounts for Package Managers**

```dockerfile
# syntax=docker/dockerfile:1
FROM node:18-alpine

WORKDIR /app

# Cache npm cache across builds
RUN --mount=type=cache,target=/root/.npm \
    npm ci

COPY . .
RUN npm run build
```

**Secrets Management**

```dockerfile
# syntax=docker/dockerfile:1
FROM alpine

# Use secret without storing in image layer
RUN --mount=type=secret,id=aws_credentials \
    cat /run/secrets/aws_credentials > ~/.aws/credentials && \
    aws s3 cp s3://bucket/file . && \
    rm ~/.aws/credentials
```

```bash
# Build with secret
docker build --secret id=aws_credentials,src=$HOME/.aws/credentials -t myapp .
```

### Workflow 5: Security Scanning with Trivy

```bash
# Install Trivy
brew install trivy  # macOS
# or
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/trivy.list
sudo apt-get update && sudo apt-get install trivy

# Scan Docker image for vulnerabilities
trivy image myapp:latest

# Fail build on HIGH/CRITICAL vulnerabilities
trivy image --exit-code 1 --severity HIGH,CRITICAL myapp:latest

# Generate JSON report
trivy image --format json --output results.json myapp:latest

# Scan filesystem before building
trivy fs --severity HIGH,CRITICAL .
```

### Workflow 6: Layer Caching Optimization

```dockerfile
# ❌ BAD: Invalidates cache on any code change
FROM node:18
COPY . .
RUN npm install
RUN npm run build

# ✅ GOOD: Separate dependency and code layers
FROM node:18
WORKDIR /app

# Cache dependencies separately
COPY package*.json ./
RUN npm ci

# Code changes don't invalidate dependency layer
COPY . .
RUN npm run build
```

**BuildKit Cache Mount (even better)**

```dockerfile
FROM node:18
WORKDIR /app

COPY package*.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci

COPY . .
RUN npm run build
```

## Best Practices

**1. Minimal Base Images**
```dockerfile
# ✅ GOOD: Alpine for small size (5MB)
FROM node:18-alpine

# ✅ GOOD: Distroless for security (no shell)
FROM gcr.io/distroless/nodejs18

# ❌ BAD: Full OS (1GB+)
FROM ubuntu:22.04
RUN apt-get install nodejs
```

**2. .dockerignore File**
```
# .dockerignore
node_modules
npm-debug.log
.git
.env
.DS_Store
*.md
Dockerfile
.dockerignore
```

**3. Run as Non-Root**
```dockerfile
# ✅ GOOD: Non-root user
RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001
USER appuser

# ❌ BAD: Running as root
USER root
```

**4. Health Checks**
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1
```

**5. Resource Limits**
```yaml
# docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
```

## Quality Criteria

- ✅ Image size <200MB (Node.js apps)
- ✅ Build time <5 minutes
- ✅ Zero HIGH/CRITICAL vulnerabilities (Trivy)
- ✅ Multi-stage builds used
- ✅ Running as non-root user
- ✅ Health check defined
- ✅ .dockerignore configured

## Common Commands

```bash
# Build image
docker build -t myapp:latest .

# Build with BuildKit cache
docker build --cache-from myapp:latest -t myapp:latest .

# Run container
docker run -d -p 3000:3000 --name myapp myapp:latest

# View logs
docker logs -f myapp

# Execute command in container
docker exec -it myapp sh

# Inspect image
docker inspect myapp:latest

# Tag and push to registry
docker tag myapp:latest myregistry.io/myapp:v1.0.0
docker push myregistry.io/myapp:v1.0.0

# Clean up
docker system prune -a --volumes
```

## Troubleshooting

**Issue**: Image size too large
**Solution**: Use multi-stage builds, Alpine/distroless base images, optimize layers

**Issue**: Build cache not working
**Solution**: Ensure COPY commands are in correct order (dependencies before code)

**Issue**: Container exits immediately
**Solution**: Check logs (`docker logs`), ensure CMD/ENTRYPOINT is correct

## Related Skills

- `kubernetes-specialist`: Deploying containers to K8s
- `aws-specialist`: ECR and ECS
- `cicd-intelligent-recovery`: Docker in CI/CD

## Tools

- Docker Desktop: Local Docker environment
- BuildKit: Advanced build engine
- Trivy: Vulnerability scanner
- Dive: Image layer analysis tool
- Hadolint: Dockerfile linter

## MCP Tools

- `mcp__flow-nexus__sandbox_execute` for Docker commands
- `mcp__memory-mcp__memory_store` for Dockerfile patterns

## Success Metrics

- Image size: <200MB
- Build time: <5 minutes
- Vulnerabilities: 0 HIGH/CRITICAL
- Startup time: <10 seconds

---

**Skill Version**: 1.0.0
**Last Updated**: 2025-11-02
