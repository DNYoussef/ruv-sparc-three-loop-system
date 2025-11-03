---
name: docker-build
category: deployment
version: 1.0.0
---

# /docker-build

Build optimized Docker container images with multi-stage builds and security scanning.

## Usage
```bash
/docker-build [dockerfile_path] [options]
```

## Parameters
- `dockerfile_path` - Path to Dockerfile (default: ./Dockerfile)
- `--tag` - Image tag (default: latest)
- `--platform` - Target platform: linux/amd64|linux/arm64|multi (default: linux/amd64)
- `--build-args` - Build arguments (key=value pairs)
- `--no-cache` - Disable build cache (default: false)
- `--scan-security` - Security scan after build (default: true)
- `--optimize` - Apply optimization strategies (default: true)
- `--push` - Push to registry after build (default: false)
- `--registry` - Container registry URL (default: docker.io)

## What It Does

**Intelligent Docker Build Process**:
1. 🔍 Analyze Dockerfile for best practices
2. 🏗️ Multi-stage build optimization
3. 📦 Layer caching strategy
4. 🔒 Security scanning (vulnerabilities)
5. 📏 Image size optimization
6. 🎯 Platform-specific builds
7. 🚀 Parallel multi-arch builds
8. 📊 Build metrics and recommendations

**Optimization Strategies**:
- **Layer Caching**: Intelligent cache invalidation
- **Multi-Stage**: Separate build and runtime stages
- **Size Reduction**: Remove build dependencies, use alpine/distroless
- **Security**: Non-root user, minimal attack surface
- **Performance**: BuildKit, parallel stages
- **.dockerignore**: Auto-generate exclusion patterns

## Examples

```bash
# Basic Docker build
/docker-build

# Build with custom tag
/docker-build --tag myapp:v1.2.3

# Multi-platform build
/docker-build --platform multi --tag myapp:latest

# Build with security scan and push
/docker-build --scan-security true --push true --registry ghcr.io

# Build with custom args
/docker-build --build-args "NODE_ENV=production,API_URL=https://api.example.com"

# No cache build for clean slate
/docker-build --no-cache true

# Build specific Dockerfile
/docker-build docker/production.Dockerfile --tag myapp:prod
```

## Output

```
🐳 Docker Build Started

Dockerfile Analysis:
  ✅ Multi-stage build detected
  ✅ Non-root user configured
  ⚠️  Consider using .dockerignore
  ⚠️  Base image not pinned (use node:18.19.0-alpine)

Building: myapp:v1.2.3
Platform: linux/amd64
BuildKit: Enabled

[Stage 1/3] Build Dependencies
  Step 1/8: FROM node:18-alpine AS builder
  Step 2/8: WORKDIR /build
  Step 3/8: COPY package*.json ./
  Step 4/8: RUN npm ci --only=production
  ✅ Cached (12.3s saved)

[Stage 2/3] Build Application
  Step 5/8: COPY src/ ./src/
  Step 6/8: RUN npm run build
  ✅ Build successful (23.4s)

[Stage 3/3] Runtime Image
  Step 7/8: FROM node:18-alpine
  Step 8/8: COPY --from=builder /build/dist ./
  ✅ Image created (2.1s)

Security Scan:
  🔍 Scanning for vulnerabilities...
  ✅ No critical vulnerabilities found
  ⚠️  2 medium severity issues
     - openssl@3.0.1 → Update to 3.0.13
     - npm@9.5.0 → Update to 10.2.4

Image Details:
  Size: 142 MB (optimized from ~580 MB)
  Layers: 8
  Compression: gzip
  Created: 2025-11-01T12:34:56Z

Recommendations:
  ✅ Use specific version tags (node:18.19.0-alpine)
  ✅ Add healthcheck instruction
  ✅ Consider distroless base for runtime
  ✅ Pin npm dependencies with exact versions

Build Summary:
  Total Time: 37.8s
  Cache Hit Rate: 67%
  Size Reduction: 75%
  Security Score: 92/100

Image Tagged:
  myapp:v1.2.3
  myapp:latest

✅ Build Complete!
```

## Chains With

```bash
# Build → scan → push
/docker-build --scan-security true && /docker-deploy --push

# Build → test → deploy
/docker-build && /smoke-test && /k8s-deploy

# Full CI/CD pipeline
/security-audit && /docker-build --tag \$VERSION && /docker-deploy

# Multi-stage workflow
/workflow:cicd --stage build --docker true
```

## See Also
- `/docker-deploy` - Deploy Docker containers
- `/k8s-deploy` - Kubernetes deployment
- `/security-audit` - Pre-build security scan
- `/workflow:cicd` - CI/CD workflow orchestration
- `/smoke-test` - Quick container validation
