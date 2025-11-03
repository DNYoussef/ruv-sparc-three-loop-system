# Example 2: GCP Cloud Run Container Deployment

Deploy a containerized web application using Google Cloud Run with Cloud SQL for a production-ready, fully managed serverless platform.

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                  GCP Cloud Run Application                 │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Internet                                                  │
│     │                                                      │
│     ▼                                                      │
│  ┌──────────────────┐                                     │
│  │  Cloud CDN       │                                     │
│  │  (Global Edge)   │                                     │
│  └────────┬─────────┘                                     │
│           │                                                │
│           ▼                                                │
│  ┌──────────────────┐                                     │
│  │  Load Balancer   │                                     │
│  │  (Global HTTPS)  │                                     │
│  └────────┬─────────┘                                     │
│           │                                                │
│           ▼                                                │
│  ┌──────────────────┐     ┌──────────────────┐           │
│  │  Cloud Run       │     │  Cloud SQL       │           │
│  │  (Containers)    │────▶│  (PostgreSQL)    │           │
│  │  • Autoscaling   │     │  • Private IP    │           │
│  │  • 0-N instances │     │  • High Avail.   │           │
│  └──────────┬───────┘     └──────────────────┘           │
│           │                                                │
│           ▼                                                │
│  ┌──────────────────┐                                     │
│  │  Cloud Logging   │                                     │
│  │  Cloud Monitoring│                                     │
│  └──────────────────┘                                     │
└────────────────────────────────────────────────────────────┘
```

## Use Case

Full-stack web application for an e-commerce platform:

- RESTful API backend built with Node.js/Express
- PostgreSQL database for product catalog and orders
- Automatic scaling from 0 to N instances
- Global deployment with low latency
- Fully managed infrastructure (no servers to maintain)

## Prerequisites

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize and authenticate
gcloud init
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  sqladmin.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com
```

## Project Structure

```
cloud-run-app/
├── terraform/
│   ├── main.tf              # Main Terraform configuration
│   ├── variables.tf         # Input variables
│   ├── outputs.tf           # Output values
│   ├── cloud-run.tf         # Cloud Run service
│   └── cloud-sql.tf         # Cloud SQL instance
├── src/
│   ├── server.js            # Express server
│   ├── routes/
│   │   ├── products.js      # Product endpoints
│   │   └── orders.js        # Order endpoints
│   ├── models/              # Database models
│   ├── middleware/          # Express middleware
│   └── lib/
│       ├── database.js      # Database connection
│       └── cache.js         # Redis caching
├── Dockerfile               # Container image
├── .dockerignore
├── package.json
├── cloudbuild.yaml          # CI/CD configuration
└── README.md
```

## Implementation

### Step 1: Terraform Infrastructure

**terraform/main.tf**:
```hcl
terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "my-terraform-state"
    prefix = "cloud-run-app"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# VPC Network for Cloud SQL
resource "google_compute_network" "private_network" {
  name                    = "cloud-run-network"
  auto_create_subnetworks = false
}

resource "google_compute_global_address" "private_ip_address" {
  name          = "private-ip-address"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.private_network.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.private_network.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

# Secret Manager for database credentials
resource "google_secret_manager_secret" "db_password" {
  secret_id = "db-password"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = random_password.db_password.result
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}
```

**terraform/cloud-sql.tf**:
```hcl
resource "google_sql_database_instance" "postgres" {
  name             = "ecommerce-db-${var.environment}"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier              = "db-f1-micro"  # Use db-custom-2-8192 for production
    availability_type = "REGIONAL"     # High availability
    disk_size         = 20
    disk_type         = "PD_SSD"
    disk_autoresize   = true

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 30
      }
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.private_network.id
    }

    insights_config {
      query_insights_enabled  = true
      query_plans_per_minute  = 5
      query_string_length     = 1024
      record_application_tags = true
    }

    maintenance_window {
      day          = 7  # Sunday
      hour         = 3
      update_track = "stable"
    }
  }

  deletion_protection = true

  depends_on = [google_service_networking_connection.private_vpc_connection]
}

resource "google_sql_database" "app_database" {
  name     = "ecommerce"
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "app_user" {
  name     = "appuser"
  instance = google_sql_database_instance.postgres.name
  password = random_password.db_password.result
}
```

**terraform/cloud-run.tf**:
```hcl
resource "google_cloud_run_v2_service" "app" {
  name     = "ecommerce-api-${var.environment}"
  location = var.region

  template {
    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }

    vpc_access {
      network_interfaces {
        network    = google_compute_network.private_network.name
        subnetwork = google_compute_subnetwork.cloud_run_subnet.name
      }
    }

    containers {
      image = "gcr.io/${var.project_id}/ecommerce-api:latest"

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1000m"
          memory = "512Mi"
        }
        cpu_idle = true
      }

      env {
        name  = "NODE_ENV"
        value = var.environment
      }

      env {
        name  = "DB_HOST"
        value = "/cloudsql/${google_sql_database_instance.postgres.connection_name}"
      }

      env {
        name  = "DB_NAME"
        value = google_sql_database.app_database.name
      }

      env {
        name  = "DB_USER"
        value = google_sql_user.app_user.name
      }

      env {
        name = "DB_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_password.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 10
        period_seconds        = 10
        timeout_seconds       = 5
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        period_seconds    = 30
        timeout_seconds   = 5
        failure_threshold = 3
      }
    }

    service_account = google_service_account.cloud_run.email
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

resource "google_service_account" "cloud_run" {
  account_id   = "cloud-run-sa-${var.environment}"
  display_name = "Cloud Run Service Account"
}

resource "google_project_iam_member" "cloud_run_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.cloud_run.email}"
}

resource "google_secret_manager_secret_iam_member" "cloud_run_secret_accessor" {
  secret_id = google_secret_manager_secret.db_password.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run.email}"
}

# Public access policy (adjust for production)
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  location = google_cloud_run_v2_service.app.location
  name     = google_cloud_run_v2_service.app.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
```

### Step 2: Application Code

**Dockerfile**:
```dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

FROM node:18-alpine

WORKDIR /app

# Install Cloud SQL Proxy
RUN apk add --no-cache curl && \
    curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.linux.amd64 && \
    chmod +x cloud-sql-proxy && \
    mv cloud-sql-proxy /usr/local/bin/

# Copy node_modules from builder
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/src ./src
COPY --from=builder /app/package*.json ./

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001
USER nodejs

EXPOSE 8080

CMD ["node", "src/server.js"]
```

**src/server.js**:
```javascript
const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const compression = require('compression');
const { Pool } = require('pg');

const app = express();
const PORT = process.env.PORT || 8080;

// Middleware
app.use(helmet());
app.use(cors());
app.use(compression());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Database connection
const pool = new Pool({
  host: process.env.DB_HOST,
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  max: 10,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    await pool.query('SELECT 1');
    res.status(200).json({ status: 'healthy', timestamp: new Date().toISOString() });
  } catch (error) {
    console.error('Health check failed:', error);
    res.status(503).json({ status: 'unhealthy', error: error.message });
  }
});

// Product routes
app.get('/api/products', async (req, res) => {
  try {
    const { rows } = await pool.query('SELECT * FROM products ORDER BY created_at DESC');
    res.json({ products: rows });
  } catch (error) {
    console.error('Error fetching products:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/api/products', async (req, res) => {
  const { name, description, price, inventory } = req.body;

  try {
    const { rows } = await pool.query(
      'INSERT INTO products (name, description, price, inventory) VALUES ($1, $2, $3, $4) RETURNING *',
      [name, description, price, inventory]
    );
    res.status(201).json({ product: rows[0] });
  } catch (error) {
    console.error('Error creating product:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Order routes
app.post('/api/orders', async (req, res) => {
  const { user_id, items } = req.body;
  const client = await pool.connect();

  try {
    await client.query('BEGIN');

    // Calculate total
    const total = items.reduce((sum, item) => sum + (item.price * item.quantity), 0);

    // Create order
    const { rows: [order] } = await client.query(
      'INSERT INTO orders (user_id, total, status) VALUES ($1, $2, $3) RETURNING *',
      [user_id, total, 'pending']
    );

    // Create order items
    for (const item of items) {
      await client.query(
        'INSERT INTO order_items (order_id, product_id, quantity, price) VALUES ($1, $2, $3, $4)',
        [order.id, item.product_id, item.quantity, item.price]
      );

      // Update inventory
      await client.query(
        'UPDATE products SET inventory = inventory - $1 WHERE id = $2',
        [item.quantity, item.product_id]
      );
    }

    await client.query('COMMIT');
    res.status(201).json({ order });
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Error creating order:', error);
    res.status(500).json({ error: 'Internal server error' });
  } finally {
    client.release();
  }
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, closing server...');
  await pool.end();
  process.exit(0);
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

### Step 3: CI/CD with Cloud Build

**cloudbuild.yaml**:
```yaml
steps:
  # Build container image
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - 'gcr.io/$PROJECT_ID/ecommerce-api:$COMMIT_SHA'
      - '-t'
      - 'gcr.io/$PROJECT_ID/ecommerce-api:latest'
      - '.'

  # Push container image to GCR
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - '--all-tags'
      - 'gcr.io/$PROJECT_ID/ecommerce-api'

  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'ecommerce-api-${_ENVIRONMENT}'
      - '--image=gcr.io/$PROJECT_ID/ecommerce-api:$COMMIT_SHA'
      - '--region=${_REGION}'
      - '--platform=managed'

  # Run database migrations
  - name: 'gcr.io/cloud-builders/gcloud'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        gcloud run jobs execute migrate-db \
          --region=${_REGION} \
          --wait

images:
  - 'gcr.io/$PROJECT_ID/ecommerce-api:$COMMIT_SHA'
  - 'gcr.io/$PROJECT_ID/ecommerce-api:latest'

substitutions:
  _ENVIRONMENT: 'production'
  _REGION: 'us-central1'

options:
  logging: CLOUD_LOGGING_ONLY
  machineType: 'N1_HIGHCPU_8'
```

### Step 4: Deployment

```bash
# Deploy infrastructure with Terraform
cd terraform
terraform init
terraform apply -var="project_id=YOUR_PROJECT_ID"

# Build and deploy application with Cloud Build
gcloud builds submit --config=cloudbuild.yaml

# Or manual deployment
docker build -t gcr.io/YOUR_PROJECT_ID/ecommerce-api:latest .
docker push gcr.io/YOUR_PROJECT_ID/ecommerce-api:latest

gcloud run deploy ecommerce-api-production \
  --image=gcr.io/YOUR_PROJECT_ID/ecommerce-api:latest \
  --region=us-central1 \
  --platform=managed
```

## Monitoring and Observability

### Cloud Monitoring Dashboards

```bash
# Create custom dashboard
gcloud monitoring dashboards create --config-from-file=dashboard.json
```

**dashboard.json**:
```json
{
  "displayName": "Cloud Run Application Metrics",
  "mosaicLayout": {
    "columns": 12,
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Request Count",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\"",
                  "aggregation": {
                    "alignmentPeriod": "60s",
                    "perSeriesAligner": "ALIGN_RATE"
                  }
                }
              }
            }]
          }
        }
      }
    ]
  }
}
```

## Cost Optimization

### Estimated Monthly Cost

| Service | Usage | Cost |
|---------|-------|------|
| Cloud Run | 1M requests, 2 vCPU, 512MB | $24.00 |
| Cloud SQL | db-f1-micro, 20GB SSD | $7.67 |
| Networking | 10GB egress | $1.20 |
| Cloud Build | 120 build minutes | $0.00 (free tier) |
| **Total** | | **~$32.87/month** |

### Cost-Saving Tips

1. **Scale to zero** when not in use (Cloud Run auto-scales to 0)
2. **Right-size Cloud SQL** instances based on actual usage
3. **Use committed use discounts** for predictable workloads (up to 57% savings)
4. **Enable request coalescing** to reduce cold starts
5. **Cache static assets** with Cloud CDN

## Production Readiness Checklist

- [ ] Multi-region deployment with load balancing
- [ ] Automated backups for Cloud SQL (daily, retained for 30 days)
- [ ] Monitoring dashboards and alerting policies
- [ ] Load testing with realistic traffic (use Locust or k6)
- [ ] Security scanning (Container Analysis, vulnerability scanning)
- [ ] Cost budget alerts
- [ ] Runbook for incident response
- [ ] Database migration strategy (blue-green deployments)

## Related Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud SQL Best Practices](https://cloud.google.com/sql/docs/postgres/best-practices)
- [Cloud Build CI/CD](https://cloud.google.com/build/docs)
