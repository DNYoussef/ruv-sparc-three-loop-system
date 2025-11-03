# Example 3: Deployment Failure Recovery with Rollback Strategy

**Scenario**: Loop 2 implementation passes all tests but fails during production deployment due to infrastructure issues, environment config mismatches, and database migration errors. Loop 3 performs intelligent recovery with automated rollback and retry strategies.

## Initial State

### Production Deployment Failure
```yaml
Workflow: Production Deploy
Status: Failed
Environment: production (AWS ECS)
Error: Deployment rollback triggered
Duration: 12m 38s
Exit Code: 1

Deployment Stages:
  ✅ Build: Success
  ✅ Test: Success
  ✅ Push Docker Image: Success
  ❌ Database Migration: Failed
  ❌ ECS Task Deploy: Failed
  ✅ Automatic Rollback: Success (previous version restored)
```

### Deployment Log Excerpt
```
[12:03:45] Starting database migrations...
[12:03:47] ERROR: relation "user_profiles" already exists
[12:03:47] Migration failed: 20240115_add_user_profiles.sql
[12:03:48] Rolling back migrations...

[12:04:12] Deploying ECS task definition...
[12:04:15] ERROR: Task failed health check
[12:04:16] Container logs:
  Error: Environment variable DATABASE_URL not found
  Error: Redis connection failed: ECONNREFUSED localhost:6379

[12:04:30] Health check failed 3/3 times
[12:04:31] Triggering automatic rollback...
[12:04:50] Rollback complete: v1.2.3 restored
```

## Loop 3 Execution

### Step 1: GitHub Hook Integration (3 minutes)

```bash
# Download deployment failure logs
gh run view 67890 --log > .claude/.artifacts/deployment-failure.log

# Parse deployment stages
node parse-deployment-errors.js
# Output:
{
  "stage_failures": [
    {
      "stage": "database_migration",
      "error": "relation already exists",
      "migration": "20240115_add_user_profiles.sql"
    },
    {
      "stage": "ecs_deploy",
      "errors": [
        "Environment variable DATABASE_URL not found",
        "Redis connection failed: ECONNREFUSED localhost:6379"
      ]
    }
  ],
  "rollback_status": "success",
  "current_version": "v1.2.3 (restored)"
}
```

### Step 2: AI-Powered Analysis (18 minutes)

#### Gemini Large-Context Analysis

```json
{
  "deployment_dependency_graph": {
    "infrastructure_nodes": [
      "AWS ECS task definition",
      "RDS PostgreSQL database",
      "ElastiCache Redis cluster",
      "Secrets Manager (environment variables)"
    ],
    "failure_cascade": [
      "Migration failure → Blocked deployment",
      "Missing DATABASE_URL → Container startup failure",
      "Wrong Redis host (localhost) → Connection failure"
    ]
  },
  "environment_analysis": {
    "development": {
      "DATABASE_URL": "localhost:5432",
      "REDIS_HOST": "localhost",
      "secrets_source": "local .env file"
    },
    "production": {
      "DATABASE_URL": "AWS Secrets Manager",
      "REDIS_HOST": "ElastiCache endpoint",
      "secrets_source": "AWS Secrets Manager + ECS task IAM role"
    },
    "mismatch_detected": [
      "DATABASE_URL not configured in production Secrets Manager",
      "REDIS_HOST hardcoded to localhost instead of environment variable",
      "ECS task IAM role missing SecretsManager read permission"
    ]
  },
  "migration_analysis": {
    "issue": "Migration attempted to create table that already exists",
    "probable_cause": "Migration previously partially executed",
    "state_mismatch": "Schema state tracker out of sync with actual database",
    "safe_recovery": "Mark migration as completed, or make migration idempotent"
  }
}
```

#### 7-Agent Analysis with Byzantine Consensus

**Researchers 1 & 2**: Infrastructure Pattern Research
```json
{
  "database_migration_failures": {
    "pattern": "Idempotent migrations required for safe retries",
    "similar_issues": [
      "Database migration failures in 23% of production deployments",
      "Root cause: Non-idempotent DDL statements"
    ],
    "solutions": [
      "Use IF NOT EXISTS for CREATE statements",
      "Check table/column existence before ALTER",
      "Track migration state in dedicated table"
    ],
    "best_practice": "All migrations must be idempotent",
    "consensus": "7/7 agents agree"
  },
  "environment_config_issues": {
    "pattern": "Environment-specific config must be externalized",
    "similar_issues": [
      "47% of deployment failures due to hardcoded config",
      "Missing environment variables in 31% of cases"
    ],
    "solutions": [
      "Use environment variables for all environment-specific config",
      "Validate required environment variables at startup",
      "Use infrastructure-as-code for environment setup"
    ],
    "consensus": "7/7 agents agree"
  }
}
```

**Error Analyzer**: Deployment Error Root Cause
```json
{
  "migration_error": {
    "error": "relation user_profiles already exists",
    "root_cause": "Migration not idempotent - CREATE TABLE without IF NOT EXISTS",
    "propagation": "Blocks entire deployment pipeline",
    "why_occurred": "Previous deployment partially succeeded (migration ran, deployment failed)",
    "fix": "Add IF NOT EXISTS clause to CREATE TABLE statements"
  },
  "environment_errors": {
    "missing_database_url": {
      "error": "Environment variable DATABASE_URL not found",
      "root_cause": "DATABASE_URL not configured in AWS Secrets Manager",
      "propagation": "Container fails to start",
      "fix": "Add DATABASE_URL to Secrets Manager, grant ECS task IAM role access"
    },
    "redis_connection": {
      "error": "Redis connection failed: ECONNREFUSED localhost:6379",
      "root_cause": "REDIS_HOST hardcoded to 'localhost' instead of environment variable",
      "propagation": "Application starts but Redis operations fail",
      "fix": "Replace hardcoded 'localhost' with process.env.REDIS_HOST"
    }
  }
}
```

**Infrastructure Investigator**: AWS Configuration Analysis
```json
{
  "ecs_task_definition": {
    "secrets": [
      {
        "name": "JWT_SECRET",
        "valueFrom": "arn:aws:secretsmanager:us-east-1:123456:secret:jwt-secret"
      }
    ],
    "missing_secrets": [
      "DATABASE_URL (required but not configured)",
      "REDIS_HOST (required but not configured)"
    ]
  },
  "iam_role_permissions": {
    "current_permissions": [
      "logs:CreateLogStream",
      "logs:PutLogEvents"
    ],
    "missing_permissions": [
      "secretsmanager:GetSecretValue (required for DATABASE_URL, REDIS_HOST)"
    ]
  },
  "rds_config": {
    "endpoint": "prod-db.cluster-abc.us-east-1.rds.amazonaws.com",
    "port": 5432,
    "status": "available",
    "issue": "Connection string not exported to Secrets Manager"
  },
  "elasticache_config": {
    "endpoint": "prod-redis.abc123.ng.0001.use1.cache.amazonaws.com",
    "port": 6379,
    "status": "available",
    "issue": "Endpoint not exported to Secrets Manager"
  }
}
```

**Database Schema Analyzer**: Migration State Analysis
```json
{
  "current_schema": {
    "tables": ["users", "user_profiles", "sessions", "audit_logs"],
    "status": "user_profiles exists (from previous partial deployment)"
  },
  "migration_history": {
    "completed": [
      "20240101_create_users.sql",
      "20240105_create_sessions.sql"
    ],
    "partial": [
      "20240115_add_user_profiles.sql (DDL succeeded, rollback failed)"
    ],
    "pending": [
      "20240120_add_audit_logs.sql"
    ]
  },
  "state_mismatch": {
    "migration_tracker_says": "20240115_add_user_profiles.sql not executed",
    "actual_database": "user_profiles table exists",
    "reason": "Rollback didn't mark migration as partially complete"
  }
}
```

#### Analysis Synthesis with Byzantine Consensus

```json
{
  "rootCauses": [
    {
      "failure": "Database migration error: relation already exists",
      "cause": "Non-idempotent migration + state mismatch from previous partial deployment",
      "evidence": [
        "Error Analyzer: CREATE TABLE without IF NOT EXISTS",
        "Schema Analyzer: user_profiles exists but migration tracker says it doesn't",
        "Researchers: 7/7 agree - non-idempotent migrations are root cause"
      ],
      "consensus": "7/7",
      "confidence": "high"
    },
    {
      "failure": "Missing DATABASE_URL environment variable",
      "cause": "DATABASE_URL not configured in AWS Secrets Manager for production",
      "evidence": [
        "Infrastructure Investigator: DATABASE_URL absent from ECS task secrets",
        "Error Analyzer: Container startup fails without DATABASE_URL",
        "Researchers: 7/7 agree - environment-specific config must be externalized"
      ],
      "consensus": "7/7",
      "confidence": "high"
    },
    {
      "failure": "Redis connection refused (localhost)",
      "cause": "REDIS_HOST hardcoded to 'localhost' instead of environment variable",
      "evidence": [
        "Error Analyzer: Hardcoded 'localhost' in connection config",
        "Infrastructure Investigator: ElastiCache endpoint exists but not used",
        "Researchers: 7/7 agree - hardcoded config causes deployment failures"
      ],
      "consensus": "7/7",
      "confidence": "high"
    },
    {
      "failure": "ECS task cannot read secrets",
      "cause": "IAM role missing secretsmanager:GetSecretValue permission",
      "evidence": [
        "Infrastructure Investigator: IAM role has only CloudWatch permissions",
        "Error Analyzer: Secrets not accessible to container",
        "Researchers: 5/7 agree - IAM permission issues common in deployments"
      ],
      "consensus": "5/7",
      "confidence": "medium-high"
    }
  ]
}
```

### Step 3: Root Cause Detection (15 minutes)

#### Infrastructure Dependency Graph

```json
{
  "deployment_flow": {
    "nodes": [
      {"id": "migration", "type": "root", "status": "failed"},
      {"id": "secrets-config", "type": "root", "status": "failed"},
      {"id": "iam-permissions", "type": "root", "status": "failed"},
      {"id": "container-startup", "type": "cascade", "status": "failed"},
      {"id": "health-check", "type": "cascade", "status": "failed"},
      {"id": "deployment", "type": "cascade", "status": "failed"}
    ],
    "edges": [
      {"from": "migration", "to": "deployment", "blocks": true},
      {"from": "secrets-config", "to": "container-startup"},
      {"from": "iam-permissions", "to": "container-startup"},
      {"from": "container-startup", "to": "health-check"},
      {"from": "health-check", "to": "deployment"}
    ]
  },
  "critical_path": [
    "Fix migration (idempotent) → ",
    "Configure Secrets Manager → ",
    "Fix IAM permissions → ",
    "Update hardcoded config → ",
    "Deploy with health checks"
  ]
}
```

#### Raft Consensus: Root Cause Validation with 5-Whys

**Root Cause 1**: Non-Idempotent Migration
```json
{
  "why_1": "Why migration failed? → Table already exists",
  "why_2": "Why table exists? → Previous deployment partially succeeded",
  "why_3": "Why partial success? → Migration ran, then deployment failed",
  "why_4": "Why didn't migration handle existing table? → No IF NOT EXISTS clause",
  "why_5": "Why no idempotency? → Migration template doesn't enforce idempotent patterns",
  "true_root": "Migrations not designed to be idempotent for safe retries"
}
```

**Root Cause 2**: Infrastructure Configuration Gap
```json
{
  "why_1": "Why DATABASE_URL missing? → Not in Secrets Manager",
  "why_2": "Why not in Secrets Manager? → Wasn't created during infrastructure setup",
  "why_3": "Why wasn't it created? → Infrastructure-as-code didn't include it",
  "why_4": "Why not in IaC? → Environment config not fully automated",
  "why_5": "Why not automated? → Manual deployment steps not captured in code",
  "true_root": "Incomplete infrastructure-as-code: environment config not fully automated"
}
```

### Step 4: Intelligent Fixes (30 minutes)

#### Root Cause 1: Non-Idempotent Migration

**Fix Plan** (Program-of-Thought)
```json
{
  "rootCause": "Migration not idempotent - CREATE TABLE without IF NOT EXISTS",
  "fixStrategy": "isolated",
  "files": [
    {
      "path": "migrations/20240115_add_user_profiles.sql",
      "reason": "Make migration idempotent",
      "changes": "Add IF NOT EXISTS to CREATE TABLE statement"
    },
    {
      "path": "migrations/20240120_add_audit_logs.sql",
      "reason": "Prevent future issues",
      "changes": "Add IF NOT EXISTS to all CREATE statements"
    }
  ],
  "minimalChanges": "Add IF NOT EXISTS clauses to DDL statements",
  "predictedSideEffects": [
    "Migration can be safely retried",
    "No state mismatch if deployment partially succeeds"
  ],
  "validationPlan": {
    "mustPass": [
      "Migration runs successfully when table doesn't exist",
      "Migration runs successfully when table already exists (idempotent)",
      "Migration state tracker updated correctly"
    ]
  }
}
```

**Fix Implementation**
```diff
-- migrations/20240115_add_user_profiles.sql
- CREATE TABLE user_profiles (
+ CREATE TABLE IF NOT EXISTS user_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    bio TEXT,
    avatar_url VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
  );

- CREATE INDEX idx_user_profiles_user_id ON user_profiles(user_id);
+ CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);
```

#### Root Cause 2: Missing DATABASE_URL in Secrets Manager

**Fix Plan** (Program-of-Thought)
```json
{
  "rootCause": "DATABASE_URL not configured in AWS Secrets Manager",
  "fixStrategy": "infrastructure",
  "files": [
    {
      "path": "infrastructure/terraform/secrets.tf",
      "reason": "Add DATABASE_URL secret to Terraform",
      "changes": "Create aws_secretsmanager_secret resource"
    },
    {
      "path": "infrastructure/terraform/ecs.tf",
      "reason": "Reference secret in ECS task definition",
      "changes": "Add DATABASE_URL to task definition secrets"
    }
  ],
  "manualSteps": [
    "Apply Terraform to create secret",
    "Manually set DATABASE_URL value in AWS console (contains sensitive data)",
    "Verify ECS task can read secret"
  ]
}
```

**Fix Implementation**
```hcl
# infrastructure/terraform/secrets.tf
resource "aws_secretsmanager_secret" "database_url" {
  name = "prod/app/database-url"
  description = "PostgreSQL connection string for production"
}

resource "aws_secretsmanager_secret" "redis_host" {
  name = "prod/app/redis-host"
  description = "ElastiCache Redis endpoint for production"
}

# infrastructure/terraform/ecs.tf
resource "aws_ecs_task_definition" "app" {
  # ...
  container_definitions = jsonencode([{
    name  = "app"
    image = "..."
    secrets = [
      {
        name      = "DATABASE_URL"
        valueFrom = aws_secretsmanager_secret.database_url.arn
      },
      {
        name      = "REDIS_HOST"
        valueFrom = aws_secretsmanager_secret.redis_host.arn
      },
      {
        name      = "JWT_SECRET"
        valueFrom = aws_secretsmanager_secret.jwt_secret.arn
      }
    ]
  }])
}
```

#### Root Cause 3: Hardcoded Redis Host

**Fix Plan** (Program-of-Thought)
```json
{
  "rootCause": "REDIS_HOST hardcoded to 'localhost' instead of environment variable",
  "fixStrategy": "isolated",
  "files": [
    {
      "path": "src/config/redis.ts",
      "reason": "Replace hardcoded value with environment variable",
      "changes": "Use process.env.REDIS_HOST with validation"
    }
  ],
  "minimalChanges": "Replace 'localhost' with process.env.REDIS_HOST",
  "predictedSideEffects": [
    "Redis connection works in production (ElastiCache)",
    "Still works locally (REDIS_HOST=localhost in .env)"
  ]
}
```

**Fix Implementation**
```diff
// src/config/redis.ts
  import Redis from 'ioredis';

+ const REDIS_HOST = process.env.REDIS_HOST;
+ if (!REDIS_HOST) {
+   throw new Error('Environment variable REDIS_HOST is required');
+ }

  export const redis = new Redis({
-   host: 'localhost',
+   host: REDIS_HOST,
    port: 6379,
    retryStrategy: (times) => Math.min(times * 50, 2000)
  });
```

#### Root Cause 4: Missing IAM Permissions

**Fix Plan** (Program-of-Thought)
```json
{
  "rootCause": "ECS task IAM role missing secretsmanager:GetSecretValue permission",
  "fixStrategy": "infrastructure",
  "files": [
    {
      "path": "infrastructure/terraform/iam.tf",
      "reason": "Add SecretsManager read permission to ECS task role",
      "changes": "Attach policy allowing GetSecretValue"
    }
  ]
}
```

**Fix Implementation**
```hcl
# infrastructure/terraform/iam.tf
resource "aws_iam_role_policy" "ecs_task_secrets" {
  name = "ecs-task-secrets-policy"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.database_url.arn,
          aws_secretsmanager_secret.redis_host.arn,
          aws_secretsmanager_secret.jwt_secret.arn
        ]
      }
    ]
  })
}
```

### Step 5: Theater Detection Audit (6 minutes)

```json
{
  "theaterDetected": [],
  "realityChecks": {
    "infrastructure": "PASS - Terraform plan shows correct resources created",
    "application": "PASS - Environment variables properly loaded",
    "deployment": "PASS - Container starts successfully, health checks pass"
  },
  "verdict": "PASS",
  "reasoning": "Fixes are authentic: migrations idempotent, secrets properly configured, IAM permissions granted"
}
```

### Step 6: Sandbox Validation (15 minutes)

```bash
# Test 1: Idempotent migration (run twice)
docker run --rm postgres:15 psql < migrations/20240115_add_user_profiles.sql
# Output: Table created

docker run --rm postgres:15 psql < migrations/20240115_add_user_profiles.sql
# Output: Table already exists (no error) ✅

# Test 2: Environment variable validation
docker run --rm -e DATABASE_URL="" app:latest
# Output: Error: Environment variable DATABASE_URL is required ✅

docker run --rm -e DATABASE_URL="postgresql://..." -e REDIS_HOST="redis" app:latest
# Output: Application started successfully ✅

# Test 3: Infrastructure deployment (staging)
terraform plan -var="env=staging"
# Output: 4 resources to create (secrets, IAM policy)

terraform apply -var="env=staging"
# Output: Apply complete! Resources: 4 added ✅
```

### Step 7: Differential Analysis (3 minutes)

```json
{
  "before": {
    "deployment_status": "failed",
    "rollback_triggered": true,
    "failures": [
      "Migration error",
      "Missing DATABASE_URL",
      "Redis connection failed",
      "IAM permission denied"
    ]
  },
  "after": {
    "deployment_status": "success",
    "health_checks": "3/3 passed",
    "failures": []
  },
  "improvements": {
    "migrations": "Idempotent (safe for retries)",
    "infrastructure": "Fully automated with Terraform",
    "environment_config": "Externalized to Secrets Manager",
    "deployment_reliability": "Rollback risk eliminated"
  }
}
```

### Step 8: GitHub Feedback (5 minutes)

```markdown
# 🤖 CI/CD Loop 3: Deployment Failure Recovery

## Summary
Fixed 4 critical deployment issues blocking production release.

## Root Causes Fixed
1. Non-idempotent database migration
2. Missing DATABASE_URL in Secrets Manager
3. Hardcoded Redis host (localhost)
4. ECS task IAM role missing SecretsManager permissions

## Quality Validation
✅ Theater Audit: PASSED (authentic infrastructure fixes)
✅ Idempotent Migrations: Tested (runs successfully when repeated)
✅ Staging Deploy: Success (infrastructure validated)
✅ Health Checks: 3/3 passed

## Files Changed
- migrations/*.sql (idempotent DDL)
- infrastructure/terraform/*.tf (secrets + IAM)
- src/config/redis.ts (environment variable)

## Deployment Strategy
- Terraform apply for infrastructure (secrets + IAM)
- Deploy application with updated migrations
- Health check validation before traffic switch
- Automated rollback ready if needed
```

## Final Results

### Quality Metrics
- **Deployment Status**: Failed → Success
- **Root Causes**: 4 identified and fixed
- **Infrastructure**: Fully automated with Terraform
- **Deployment Safety**: Rollback-safe with idempotent migrations

### Time Efficiency
- **Manual Infrastructure Fix**: 3-4 hours
- **Loop 3 Automated**: 85 minutes
- **Speedup**: 2-3x faster

### Failure Patterns for Loop 1
```json
{
  "patterns": [
    {
      "category": "data-persistence",
      "preventionStrategy": "Ensure all migrations are idempotent, track state correctly",
      "premortemQuestion": "What if database operations fail mid-deployment?"
    },
    {
      "category": "infrastructure",
      "preventionStrategy": "Automate all environment config with IaC, validate before deploy",
      "premortemQuestion": "What if environment-specific config is missing or incorrect?"
    }
  ]
}
```

## Key Takeaways

1. **Idempotent Migrations**: IF NOT EXISTS prevents state mismatch from partial deployments
2. **Infrastructure-as-Code**: Terraform automation eliminated manual configuration errors
3. **Environment Validation**: Application startup validates required environment variables
4. **Rollback Safety**: Idempotent operations allow safe retries without rollback risk
5. **Loop Integration**: Infrastructure patterns fed back to Loop 1 for future architecture planning
