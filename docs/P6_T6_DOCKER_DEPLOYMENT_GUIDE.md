# Docker Image Build & Deployment Guide

**Task**: P6_T6 - Final Release & Retrospective
**Version**: v1.0.0
**Date**: 2025-11-08

---

## Overview

This guide provides instructions for building, tagging, and pushing Docker images for the Ruv-SPARC Three-Loop System to Docker Hub.

**Note**: The ruv-sparc-three-loop-system repository is primarily a development framework and orchestration system. Docker deployment applies to specific sub-projects like `ruv-sparc-ui-dashboard` if they exist.

---

## Prerequisites

### 1. Docker Installation

Verify Docker is installed:
```bash
docker --version
# Expected: Docker version 20.10.0 or higher
```

If not installed:
- **Windows**: Download from https://www.docker.com/products/docker-desktop
- **macOS**: `brew install --cask docker`
- **Linux**: `sudo apt-get install docker.io` (Ubuntu/Debian)

### 2. Docker Hub Account

Create an account at https://hub.docker.com if you don't have one.

### 3. Docker Login

```bash
docker login
# Enter your Docker Hub username and password
```

Verify login:
```bash
docker info | grep Username
```

---

## Docker Image Build Instructions

### Project Structure Check

First, verify if Docker infrastructure exists:

```bash
# Check for Docker Compose file
ls -la ruv-sparc-ui-dashboard/docker-compose.yml 2>/dev/null || echo "No docker-compose.yml found"

# Check for Dockerfiles
ls -la ruv-sparc-ui-dashboard/backend/Dockerfile 2>/dev/null || echo "No backend Dockerfile"
ls -la ruv-sparc-ui-dashboard/frontend/Dockerfile 2>/dev/null || echo "No frontend Dockerfile"
```

**If Docker infrastructure exists** (e.g., in `ruv-sparc-ui-dashboard/`), proceed with the following steps.

**If Docker infrastructure does NOT exist**, this guide serves as a template for future Docker deployment.

---

## Build Process (if Docker infrastructure exists)

### 1. Backend Image (FastAPI)

**Navigate to project directory**:
```bash
cd ruv-sparc-ui-dashboard
```

**Build backend image**:
```bash
docker build \
  -t ruv-sparc-backend:1.0.0 \
  -t ruv-sparc-backend:latest \
  -f backend/Dockerfile \
  ./backend
```

**Verify build**:
```bash
docker images | grep ruv-sparc-backend
# Expected:
# ruv-sparc-backend   1.0.0    <image-id>   X seconds ago   340MB
# ruv-sparc-backend   latest   <image-id>   X seconds ago   340MB
```

**Test backend image**:
```bash
docker run --rm -p 8000:8000 ruv-sparc-backend:1.0.0
# Visit http://localhost:8000/docs (should show FastAPI Swagger UI)
# Press Ctrl+C to stop
```

---

### 2. Frontend Image (React + Nginx)

**Build frontend image**:
```bash
docker build \
  -t ruv-sparc-frontend:1.0.0 \
  -t ruv-sparc-frontend:latest \
  -f frontend/Dockerfile \
  ./frontend
```

**Verify build**:
```bash
docker images | grep ruv-sparc-frontend
# Expected:
# ruv-sparc-frontend   1.0.0    <image-id>   X seconds ago   180MB
# ruv-sparc-frontend   latest   <image-id>   X seconds ago   180MB
```

**Test frontend image**:
```bash
docker run --rm -p 80:80 ruv-sparc-frontend:1.0.0
# Visit http://localhost (should show React app)
# Press Ctrl+C to stop
```

---

### 3. PostgreSQL Image (Optional Custom Build)

**If using custom PostgreSQL configuration**:
```bash
docker build \
  -t ruv-sparc-postgres:1.0.0 \
  -t ruv-sparc-postgres:latest \
  -f config/postgres/Dockerfile \
  ./config/postgres
```

**Note**: Most deployments use the official `postgres:15-alpine` image with custom configuration files. Custom build is only needed for specialized setups.

---

### 4. Redis Image (Optional Custom Build)

**If using custom Redis configuration**:
```bash
docker build \
  -t ruv-sparc-redis:1.0.0 \
  -t ruv-sparc-redis:latest \
  -f config/redis/Dockerfile \
  ./config/redis
```

**Note**: Most deployments use the official `redis:7-alpine` image with custom configuration. Custom build is only needed for specialized setups.

---

## Docker Hub Push Instructions

### 1. Tag Images for Docker Hub

Replace `<your-dockerhub-username>` with your Docker Hub username:

```bash
# Backend
docker tag ruv-sparc-backend:1.0.0 <your-dockerhub-username>/ruv-sparc-backend:1.0.0
docker tag ruv-sparc-backend:latest <your-dockerhub-username>/ruv-sparc-backend:latest

# Frontend
docker tag ruv-sparc-frontend:1.0.0 <your-dockerhub-username>/ruv-sparc-frontend:1.0.0
docker tag ruv-sparc-frontend:latest <your-dockerhub-username>/ruv-sparc-frontend:latest
```

**Verify tags**:
```bash
docker images | grep <your-dockerhub-username>
```

---

### 2. Push Images to Docker Hub

**Backend**:
```bash
docker push <your-dockerhub-username>/ruv-sparc-backend:1.0.0
docker push <your-dockerhub-username>/ruv-sparc-backend:latest
```

**Frontend**:
```bash
docker push <your-dockerhub-username>/ruv-sparc-frontend:1.0.0
docker push <your-dockerhub-username>/ruv-sparc-frontend:latest
```

**Progress output**:
```
The push refers to repository [docker.io/<username>/ruv-sparc-backend]
abc123def456: Pushed
789ghijkl012: Pushed
...
1.0.0: digest: sha256:abc123... size: 4567
latest: digest: sha256:abc123... size: 4567
```

---

### 3. Verify Docker Hub Upload

1. Visit https://hub.docker.com/u/<your-dockerhub-username>
2. Verify repositories exist:
   - `<your-dockerhub-username>/ruv-sparc-backend`
   - `<your-dockerhub-username>/ruv-sparc-frontend`
3. Check tags:
   - `1.0.0` (version tag)
   - `latest` (latest tag)

---

## Production Deployment

### 1. Update docker-compose.yml for Production

**Original (development)**:
```yaml
services:
  backend:
    build: ./backend
    image: ruv-sparc-backend:latest
```

**Production (using Docker Hub images)**:
```yaml
services:
  backend:
    image: <your-dockerhub-username>/ruv-sparc-backend:1.0.0
    # No 'build' directive in production
```

**Create production docker-compose file**:
```bash
cp docker-compose.yml docker-compose.prod.yml
```

**Edit docker-compose.prod.yml**:
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    # ... (rest of config)

  redis:
    image: redis:7-alpine
    # ... (rest of config)

  backend:
    image: <your-dockerhub-username>/ruv-sparc-backend:1.0.0
    restart: always
    environment:
      - ENVIRONMENT=production
    # ... (rest of config)

  frontend:
    image: <your-dockerhub-username>/ruv-sparc-frontend:1.0.0
    restart: always
    # ... (rest of config)
```

---

### 2. Deploy to Production Server

**On production server**:
```bash
# Pull latest images
docker-compose -f docker-compose.prod.yml pull

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Verify all services are running
docker-compose -f docker-compose.prod.yml ps
# Expected: All services "Up (healthy)"
```

**Validate deployment**:
```bash
# Run validation script (if exists)
./scripts/validate-deployment.sh

# Check health endpoints
curl http://localhost:8000/health  # Backend
curl http://localhost/health       # Frontend
```

---

### 3. Security Scan Before Deployment

**Run Trivy scan on images**:
```bash
# Install Trivy (if not installed)
# Linux: sudo apt-get install trivy
# macOS: brew install trivy
# Windows: choco install trivy

# Scan backend image
trivy image <your-dockerhub-username>/ruv-sparc-backend:1.0.0

# Scan frontend image
trivy image <your-dockerhub-username>/ruv-sparc-frontend:1.0.0
```

**Automated scan script** (if exists):
```bash
./scripts/trivy-scan.sh
```

**Verify zero CRITICAL vulnerabilities**:
```
Total: 0 (CRITICAL: 0, HIGH: 0, MEDIUM: 2, LOW: 5)
```

---

## Rollback Procedures

### Quick Rollback (if v1.0.0 has issues)

**Rollback to previous tag** (e.g., v0.9.0):
```bash
# Update docker-compose.prod.yml
sed -i 's/:1.0.0/:0.9.0/g' docker-compose.prod.yml

# Restart services
docker-compose -f docker-compose.prod.yml up -d
```

---

### Emergency Rollback (if major failure)

**Stop all services**:
```bash
docker-compose -f docker-compose.prod.yml down
```

**Restore from backup**:
```bash
# Restore PostgreSQL data
docker run --rm -v postgres_data:/data -v /backup:/backup \
  postgres:15-alpine \
  sh -c "cd /data && tar xvf /backup/postgres_backup.tar"

# Restart with previous version
docker-compose -f docker-compose.prod.v0.9.0.yml up -d
```

---

## CI/CD Integration (Future Enhancement)

### GitHub Actions Workflow (Example)

Create `.github/workflows/docker-publish.yml`:

```yaml
name: Docker Build & Push

on:
  push:
    tags:
      - 'v*.*.*'  # Trigger on version tags (e.g., v1.0.0)

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Extract version from tag
        id: vars
        run: echo "VERSION=${GITHUB_REF#refs/tags/v}" >> $GITHUB_OUTPUT

      - name: Build and push backend
        uses: docker/build-push-action@v4
        with:
          context: ./ruv-sparc-ui-dashboard/backend
          push: true
          tags: |
            ${{ secrets.DOCKERHUB_USERNAME }}/ruv-sparc-backend:${{ steps.vars.outputs.VERSION }}
            ${{ secrets.DOCKERHUB_USERNAME }}/ruv-sparc-backend:latest

      - name: Build and push frontend
        uses: docker/build-push-action@v4
        with:
          context: ./ruv-sparc-ui-dashboard/frontend
          push: true
          tags: |
            ${{ secrets.DOCKERHUB_USERNAME }}/ruv-sparc-frontend:${{ steps.vars.outputs.VERSION }}
            ${{ secrets.DOCKERHUB_USERNAME }}/ruv-sparc-frontend:latest

      - name: Run Trivy vulnerability scan
        run: |
          trivy image ${{ secrets.DOCKERHUB_USERNAME }}/ruv-sparc-backend:${{ steps.vars.outputs.VERSION }}
          trivy image ${{ secrets.DOCKERHUB_USERNAME }}/ruv-sparc-frontend:${{ steps.vars.outputs.VERSION }}
```

**Configure GitHub Secrets**:
1. Go to repository Settings → Secrets and variables → Actions
2. Add:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username
   - `DOCKERHUB_TOKEN`: Docker Hub access token (not password)

---

## Best Practices

### 1. Image Tagging Strategy

**Always use semantic versioning**:
```bash
# Good
docker tag image:1.0.0
docker tag image:1.0
docker tag image:latest

# Bad
docker tag image:prod
docker tag image:stable
```

**Tag hierarchy**:
- `1.0.0` → Specific version (immutable, for production)
- `1.0` → Minor version (updated with patches)
- `latest` → Latest stable release (updated with each release)

---

### 2. Multi-Stage Build Optimization

**Backend Dockerfile** (example):
```dockerfile
# Stage 1: Builder
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY --from=builder /root/.local /home/appuser/.local
COPY . .
USER appuser
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Benefits**:
- 50-70% smaller image size
- Faster deployment
- Reduced attack surface

---

### 3. Security Hardening

**Non-root user**:
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

**No secrets in images**:
```bash
# Bad
ENV DATABASE_PASSWORD=secret123

# Good (use Docker secrets or env vars at runtime)
ENV DATABASE_PASSWORD=${DATABASE_PASSWORD}
```

**Minimal base images**:
```dockerfile
# Good
FROM python:3.11-slim
FROM node:18-alpine
FROM postgres:15-alpine

# Bad (larger attack surface)
FROM ubuntu:latest
FROM python:3.11
```

---

## Monitoring & Maintenance

### 1. Image Updates

**Check for base image updates monthly**:
```bash
docker pull python:3.11-slim
docker pull node:18-alpine
docker pull postgres:15-alpine
docker pull redis:7-alpine
```

**Rebuild with updated base images**:
```bash
docker build --no-cache -t ruv-sparc-backend:1.0.1 ./backend
```

---

### 2. Vulnerability Scanning

**Automated scanning in CI/CD**:
```bash
trivy image <image>:<tag> --severity CRITICAL,HIGH --exit-code 1
# Exit code 1 if CRITICAL or HIGH vulnerabilities found
```

**Schedule weekly scans**:
```bash
# Cron job (every Sunday at 2 AM)
0 2 * * 0 /path/to/trivy-scan.sh
```

---

### 3. Disk Space Management

**Remove unused images**:
```bash
docker image prune -a --filter "until=720h"  # Remove images older than 30 days
```

**Monitor disk usage**:
```bash
docker system df
# Shows space used by images, containers, volumes
```

---

## Troubleshooting

### Issue: "unauthorized: authentication required"

**Solution**:
```bash
docker login
# Re-enter Docker Hub credentials
```

---

### Issue: "denied: requested access to the resource is denied"

**Solution**: Check repository name matches Docker Hub username:
```bash
# Incorrect
docker push someone-else/ruv-sparc-backend:1.0.0

# Correct
docker push <your-dockerhub-username>/ruv-sparc-backend:1.0.0
```

---

### Issue: Build fails with "no space left on device"

**Solution**: Clean up Docker disk space:
```bash
docker system prune -a --volumes
# WARNING: This removes ALL unused images, containers, and volumes
```

---

### Issue: Image is too large (>1GB)

**Solution**: Use multi-stage builds and .dockerignore:

**Create .dockerignore**:
```
node_modules
.git
*.log
*.md
tests
.env
```

**Verify size reduction**:
```bash
docker images | grep ruv-sparc
# Before: 1.2GB
# After multi-stage + .dockerignore: 340MB
```

---

## Success Criteria

✅ Backend image built successfully (<500MB)
✅ Frontend image built successfully (<250MB)
✅ Images tagged with version and latest
✅ Images pushed to Docker Hub
✅ Zero CRITICAL vulnerabilities (Trivy scan)
✅ Images verified on Docker Hub
✅ Production deployment tested

---

## Summary

This guide covered:
1. ✅ Docker image build process (backend, frontend)
2. ✅ Docker Hub push workflow (tagging, authentication)
3. ✅ Production deployment (docker-compose.prod.yml)
4. ✅ Security scanning (Trivy)
5. ✅ Rollback procedures
6. ✅ CI/CD integration example (GitHub Actions)
7. ✅ Best practices (multi-stage builds, security hardening)
8. ✅ Monitoring and maintenance

---

**Deployment Status**: Ready for Docker Hub publication
**Next Step**: Execute build and push commands for v1.0.0
