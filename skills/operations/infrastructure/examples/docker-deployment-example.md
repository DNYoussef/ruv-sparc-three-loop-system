# Docker Microservices Deployment Example

**Scenario**: Deploy a complete microservices application using Docker Compose with service discovery, health checks, auto-restart policies, and monitoring.

**Components**: Frontend (React), API (Node.js), Database (PostgreSQL), Cache (Redis), Message Queue (RabbitMQ), Monitoring (Prometheus + Grafana)

**Lines**: ~250

---

## Prerequisites

```bash
# Required tools
docker --version          # Docker 20.10+
docker-compose --version  # Docker Compose 2.0+

# Optional but recommended
make --version           # For automation
```

## Project Structure

```
microservices-app/
├── docker-compose.yml
├── docker-compose.override.yml  # Local development overrides
├── .env                          # Environment variables
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── src/
├── api/
│   ├── Dockerfile
│   ├── package.json
│   └── src/
├── worker/
│   ├── Dockerfile
│   └── src/
├── nginx/
│   ├── nginx.conf
│   └── conf.d/
└── monitoring/
    ├── prometheus/
    │   └── prometheus.yml
    └── grafana/
        └── provisioning/
```

## Step 1: Environment Configuration

Create `.env` file:

```bash
# Project
PROJECT_NAME=myapp
ENVIRONMENT=production
VERSION=v1.2.0

# Ports
FRONTEND_PORT=80
API_PORT=3000
DB_PORT=5432
REDIS_PORT=6379
RABBITMQ_PORT=5672
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001

# Database
DB_USER=postgres
DB_PASSWORD=secure_password_here
DB_NAME=myapp_production

# Redis
REDIS_PASSWORD=redis_password_here

# RabbitMQ
RABBITMQ_USER=admin
RABBITMQ_PASSWORD=rabbitmq_password_here
RABBITMQ_VHOST=/

# Application
JWT_SECRET=your_jwt_secret_here
API_URL=http://api:3000
LOG_LEVEL=info

# Worker
WORKER_CONCURRENCY=5
WORKER_REPLICAS=2

# Monitoring
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin
```

## Step 2: Multi-Stage Dockerfile for API

Create `api/Dockerfile`:

```dockerfile
# Stage 1: Build dependencies
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies (including devDependencies for build)
RUN npm ci

# Copy source code
COPY . .

# Build TypeScript (if applicable)
RUN npm run build

# Prune dev dependencies
RUN npm prune --production

# Stage 2: Production image
FROM node:18-alpine AS production

WORKDIR /app

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

# Copy built artifacts from builder
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/package*.json ./

# Health check script
COPY --chown=nodejs:nodejs healthcheck.js ./

# Switch to non-root user
USER nodejs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD node healthcheck.js

# Start application
CMD ["node", "dist/server.js"]
```

## Step 3: Frontend Dockerfile with Nginx

Create `frontend/Dockerfile`:

```dockerfile
# Stage 1: Build React application
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine AS production

# Copy built React app
COPY --from=builder /app/build /usr/share/nginx/html

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://localhost/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
```

Create `frontend/nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript
               application/x-javascript application/xml+rss
               application/json application/javascript;

    server {
        listen 80;
        server_name _;

        root /usr/share/nginx/html;
        index index.html;

        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }

        # SPA routing
        location / {
            try_files $uri $uri/ /index.html;
        }

        # API proxy
        location /api/ {
            proxy_pass http://api:3000/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_cache_bypass $http_upgrade;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Cache static assets
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

## Step 4: Worker Dockerfile

Create `worker/Dockerfile`:

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --production

COPY . .

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001 && \
    chown -R nodejs:nodejs /app

USER nodejs

CMD ["node", "src/worker.js"]
```

## Step 5: Complete Docker Compose Configuration

The `docker-compose.yml` is already provided in the templates. Here's how to use it:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps

# Scale worker service
docker-compose up -d --scale worker=5

# Stop all services
docker-compose down

# Stop and remove volumes (destructive!)
docker-compose down -v
```

## Step 6: Monitoring Configuration

Create `monitoring/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'api'
    static_configs:
      - targets: ['api:9090']
    metrics_path: '/metrics'

  - job_name: 'frontend'
    static_configs:
      - targets: ['frontend:9090']

  - job_name: 'database'
    static_configs:
      - targets: ['database:9187']  # postgres_exporter

  - job_name: 'redis'
    static_configs:
      - targets: ['cache:9121']  # redis_exporter
```

## Step 7: Health Check Implementation

Create `api/healthcheck.js`:

```javascript
const http = require('http');

const options = {
  host: 'localhost',
  port: 3000,
  path: '/health',
  timeout: 2000,
};

const request = http.request(options, (res) => {
  console.log(`Health check status: ${res.statusCode}`);
  process.exit(res.statusCode === 200 ? 0 : 1);
});

request.on('error', (err) => {
  console.error('Health check failed:', err.message);
  process.exit(1);
});

request.end();
```

## Step 8: Database Initialization

Create `database/init/01_schema.sql`:

```sql
-- Create tables
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(500) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_token ON sessions(token);

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

## Step 9: Deployment Commands

Create `Makefile` for automation:

```makefile
.PHONY: build up down logs restart clean

build:
	docker-compose build --parallel

up:
	docker-compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 10
	docker-compose ps

down:
	docker-compose down

logs:
	docker-compose logs -f

restart:
	docker-compose restart

clean:
	docker-compose down -v
	docker system prune -f

health:
	@echo "Checking service health..."
	@curl -f http://localhost/health || echo "Frontend: FAIL"
	@curl -f http://localhost:3000/health || echo "API: FAIL"
	@curl -f http://localhost:9090/-/healthy || echo "Prometheus: FAIL"
	@curl -f http://localhost:3001/api/health || echo "Grafana: FAIL"

deploy: build up
	@echo "Deployment complete!"
	@$(MAKE) health
```

## Step 10: Production Deployment Workflow

```bash
# 1. Build optimized images
docker-compose build --parallel --no-cache

# 2. Tag images for registry
docker tag myapp-frontend:latest registry.example.com/myapp-frontend:v1.2.0
docker tag myapp-api:latest registry.example.com/myapp-api:v1.2.0
docker tag myapp-worker:latest registry.example.com/myapp-worker:v1.2.0

# 3. Push to registry
docker push registry.example.com/myapp-frontend:v1.2.0
docker push registry.example.com/myapp-api:v1.2.0
docker push registry.example.com/myapp-worker:v1.2.0

# 4. Deploy on production server
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 5. Verify deployment
docker-compose ps
docker-compose logs --tail=50

# 6. Run health checks
make health

# 7. Monitor metrics
open http://localhost:3001  # Grafana
```

## Step 11: Backup Strategy

Create `scripts/backup.sh`:

```bash
#!/bin/bash
BACKUP_DIR="./backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
docker-compose exec -T database pg_dump -U postgres myapp_production > \
    "$BACKUP_DIR/db_backup_$DATE.sql"

# Backup volumes
docker run --rm \
    -v myapp-database-data:/data \
    -v $(pwd)/backups:/backup \
    alpine tar czf /backup/volumes_backup_$DATE.tar.gz /data

echo "Backup completed: $DATE"
```

## Troubleshooting

### Issue: Services not starting
```bash
# Check logs
docker-compose logs api
docker-compose logs database

# Restart individual service
docker-compose restart api

# Rebuild and restart
docker-compose up -d --build api
```

### Issue: Database connection errors
```bash
# Verify database is healthy
docker-compose exec database pg_isready -U postgres

# Check connection from API container
docker-compose exec api nc -zv database 5432

# Check environment variables
docker-compose exec api env | grep DATABASE
```

### Issue: High memory usage
```bash
# Check container stats
docker stats

# Set resource limits in docker-compose.yml (already configured)
```

## Performance Optimization

1. **Multi-stage builds**: Reduce image size by 60-70%
2. **Layer caching**: Order Dockerfile commands to maximize cache hits
3. **Alpine base images**: Smaller attack surface and faster pulls
4. **Health checks**: Enable zero-downtime deployments
5. **Resource limits**: Prevent resource exhaustion
6. **Horizontal scaling**: Scale workers based on queue depth

## Security Best Practices

✅ Non-root users in containers
✅ Secrets via environment variables (use Docker secrets in swarm mode)
✅ Network segmentation with custom networks
✅ Regular security updates for base images
✅ Minimal base images (Alpine)
✅ No hardcoded credentials

---

**Result**: Production-ready Docker Compose deployment with:
- 9 microservices
- Service discovery
- Health checks
- Auto-restart
- Monitoring stack
- Resource limits
- Security hardening
