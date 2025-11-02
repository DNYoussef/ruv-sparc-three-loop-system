# Docker Compose Orchestration Command

**Category**: External Integrations
**Phase**: 4 - Deployment & Integration
**Complexity**: Medium

## Purpose

Orchestrate multi-container Docker applications with development, testing, and production environments using Docker Compose.

## Usage

```bash
/docker-compose [action] [options]

Actions:
  init                Generate docker-compose.yml
  up                  Start services
  down                Stop services
  build               Build images
  logs                View logs
  exec                Execute command in container
  scale               Scale services
  validate            Validate compose file

Options:
  --env <environment> Environment (dev|test|prod)
  --service <name>    Target specific service
  --detach            Run in background
  --build             Force rebuild
  --scale <count>     Number of instances
  --file <path>       Custom compose file
```

## Examples

```bash
# Initialize multi-service stack
/docker-compose init --env dev

# Start all services
/docker-compose up --detach

# Start specific service
/docker-compose up --service api --build

# View logs
/docker-compose logs --service api --follow

# Execute command in container
/docker-compose exec --service api -- npm run migrate

# Scale service
/docker-compose scale --service worker --scale 3

# Validate compose file
/docker-compose validate --file docker-compose.prod.yml

# Stop and remove
/docker-compose down --volumes
```

## Docker Compose Templates

### 1. Full-Stack Application (Node.js + PostgreSQL + Redis)

```yaml
# docker-compose.yml
version: '3.8'

services:
  # API Service
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
      target: ${BUILD_TARGET:-production}
    ports:
      - "3000:3000"
    environment:
      NODE_ENV: ${NODE_ENV:-production}
      DATABASE_URL: postgresql://postgres:password@postgres:5432/myapp
      REDIS_URL: redis://redis:6379
      JWT_SECRET: ${JWT_SECRET}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./api:/app
      - /app/node_modules
    networks:
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # PostgreSQL Database
  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      PGDATA: /var/lib/postgresql/data/pgdata
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    networks:
      - backend
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - backend
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  # Frontend (React)
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      args:
        REACT_APP_API_URL: http://localhost:3000
    ports:
      - "80:80"
    depends_on:
      - api
    networks:
      - frontend
      - backend
    restart: unless-stopped

  # Background Worker
  worker:
    build:
      context: ./api
      dockerfile: Dockerfile
    command: npm run worker
    environment:
      NODE_ENV: ${NODE_ENV:-production}
      DATABASE_URL: postgresql://postgres:password@postgres:5432/myapp
      REDIS_URL: redis://redis:6379
    depends_on:
      - postgres
      - redis
    networks:
      - backend
    restart: unless-stopped
    deploy:
      replicas: 2

volumes:
  postgres_data:
  redis_data:

networks:
  frontend:
  backend:
```

### 2. Development Override

```yaml
# docker-compose.override.yml (auto-loaded in dev)
version: '3.8'

services:
  api:
    build:
      target: development
    environment:
      NODE_ENV: development
      DEBUG: "app:*"
    volumes:
      - ./api:/app
      - /app/node_modules
    command: npm run dev
    ports:
      - "9229:9229"  # Node debugger

  frontend:
    build:
      target: development
    volumes:
      - ./frontend:/app
      - /app/node_modules
    command: npm start
    environment:
      REACT_APP_API_URL: http://localhost:3000
      CHOKIDAR_USEPOLLING: "true"
```

### 3. Production Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    image: myregistry.com/myapp-api:${VERSION}
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
      restart_policy:
        condition: on-failure
        max_attempts: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  postgres:
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password
    volumes:
      - /mnt/data/postgres:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
      - frontend

secrets:
  db_password:
    external: true
```

## Multi-Environment Setup

```bash
# Development (auto-loads docker-compose.override.yml)
docker-compose up

# Testing
docker-compose -f docker-compose.yml -f docker-compose.test.yml up

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Service Scaling

```bash
# Scale workers horizontally
docker-compose up -d --scale worker=5

# Scale down
docker-compose up -d --scale worker=1

# Auto-scaling based on load (use docker swarm or K8s)
```

## Networking

```yaml
# Custom networks for isolation
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No external access

# Bridge to existing network
networks:
  default:
    external:
      name: my_existing_network
```

## Health Checks

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s

# Wait for healthy services
depends_on:
  postgres:
    condition: service_healthy
  redis:
    condition: service_healthy
```

## Secrets Management

```yaml
# Using Docker secrets (Swarm/K8s)
services:
  api:
    secrets:
      - db_password
      - jwt_secret
    environment:
      DB_PASSWORD_FILE: /run/secrets/db_password
      JWT_SECRET_FILE: /run/secrets/jwt_secret

secrets:
  db_password:
    file: ./secrets/db_password.txt
  jwt_secret:
    external: true
```

## Logging

```yaml
# Centralized logging
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service=api"

  # Alternative: Syslog
  worker:
    logging:
      driver: syslog
      options:
        syslog-address: "tcp://192.168.0.42:514"
```

## Resource Limits

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 1G
        reservations:
          cpus: '1'
          memory: 512M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s
```

## Docker Compose Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Execute command
docker-compose exec api sh
docker-compose exec -T postgres pg_dump -U postgres myapp > backup.sql

# Stop services
docker-compose stop
docker-compose down

# Remove volumes
docker-compose down -v

# Build without cache
docker-compose build --no-cache

# Pull latest images
docker-compose pull

# Validate configuration
docker-compose config

# List services
docker-compose ps

# Top processes
docker-compose top
```

## GitHub Actions Integration

```yaml
name: Docker Compose CI

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Build and start services
      run: |
        docker-compose -f docker-compose.yml -f docker-compose.test.yml up -d
        docker-compose exec -T api npm run test

    - name: Cleanup
      if: always()
      run: docker-compose down -v
```

## Best Practices

1. **Use .env files**: Never commit secrets to Git
2. **Health checks**: Ensure services are ready before accepting traffic
3. **Resource limits**: Prevent container resource exhaustion
4. **Named volumes**: Persist data across container restarts
5. **Multi-stage builds**: Reduce image size
6. **Layer caching**: Optimize Dockerfile for faster builds
7. **Networks**: Isolate services (frontend/backend separation)
8. **Logging drivers**: Centralize logs for debugging
9. **Secrets management**: Use Docker secrets or vault
10. **Version pinning**: Pin image versions for reproducibility

## Troubleshooting

```bash
# Check service status
docker-compose ps

# View service logs
docker-compose logs --tail=100 api

# Restart service
docker-compose restart api

# Remove orphan containers
docker-compose down --remove-orphans

# Rebuild service
docker-compose up -d --build --force-recreate api

# Check network connectivity
docker-compose exec api ping postgres
docker-compose exec api curl http://redis:6379
```

## Integration Points

- **/github-actions** - CI/CD automation
- **/aws-deploy** - Cloud deployment
- **/terraform-apply** - Infrastructure provisioning
- **/ansible-deploy** - Configuration management

## Agent Integration

```javascript
// Orchestrator: Manage container lifecycle
mcp__ruv-swarm__task_orchestrate({
  task: "Deploy multi-container stack with health checks",
  strategy: "sequential"
})

// DevOps: Optimize container configuration
mcp__ruv-swarm__agent_spawn({
  type: "optimizer",
  capabilities: ["docker", "containerization"]
})
```

## Related Commands

- `/aws-deploy` - Cloud deployment
- `/terraform-apply` - Infrastructure as code
- `/github-actions` - CI/CD workflows

---

**Implementation Status**: Production-Ready
**Last Updated**: 2025-11-01
**Maintainer**: Integration Specialist
