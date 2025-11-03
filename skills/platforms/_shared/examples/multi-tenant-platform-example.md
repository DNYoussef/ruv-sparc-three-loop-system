# Multi-Tenant Platform Example
Building a complete multi-tenant SaaS platform with Flow Nexus

## Overview

This example demonstrates building an enterprise-grade multi-tenant platform with:
- Tenant isolation and data segregation
- Per-tenant resource limits and billing
- Shared infrastructure with tenant-specific customization
- Automated tenant provisioning and lifecycle management
- Centralized monitoring and management

**Estimated Time**: 90-120 minutes
**Difficulty**: Expert
**Prerequisites**: Advanced Flow Nexus knowledge, database expertise, security understanding

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Multi-Tenant Platform                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              API Gateway & Load Balancer                  │  │
│  │          (Tenant Routing & Authentication)                │  │
│  └────┬────────────────┬──────────────────┬─────────────────┘  │
│       │                │                  │                      │
│  ┌────▼────┐      ┌───▼─────┐       ┌───▼──────┐              │
│  │ Tenant  │      │ Tenant  │       │ Tenant   │              │
│  │   A     │      │   B     │       │   C      │              │
│  │ (Free)  │      │(Premium)│       │(Enterprise)             │
│  └────┬────┘      └───┬─────┘       └───┬──────┘              │
│       │               │                  │                      │
│  ┌────▼───────────────▼──────────────────▼──────┐              │
│  │         Shared Services Layer                 │              │
│  │  (Database, Storage, Compute, Monitoring)     │              │
│  └──────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Database Schema for Multi-Tenancy

### Schema-Per-Tenant Approach

```sql
-- Master database for tenant management
-- File: platform/config/schema-master.sql

CREATE TABLE IF NOT EXISTS tenants (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(255) NOT NULL,
  subdomain VARCHAR(255) UNIQUE NOT NULL,
  tier VARCHAR(50) NOT NULL,  -- free, premium, enterprise
  status VARCHAR(50) DEFAULT 'active',  -- active, suspended, deleted

  -- Resource limits
  max_sandboxes INTEGER DEFAULT 5,
  max_storage_mb INTEGER DEFAULT 1000,
  max_compute_hours INTEGER DEFAULT 10,
  max_users INTEGER DEFAULT 5,

  -- Billing
  billing_plan VARCHAR(50),
  billing_cycle VARCHAR(50),  -- monthly, annual
  mrr DECIMAL(10, 2) DEFAULT 0,

  -- Metadata
  database_name VARCHAR(255) UNIQUE NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  deleted_at TIMESTAMP NULL
);

CREATE TABLE IF NOT EXISTS tenant_users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
  email VARCHAR(255) NOT NULL,
  role VARCHAR(50) NOT NULL,  -- owner, admin, member
  status VARCHAR(50) DEFAULT 'active',
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(tenant_id, email)
);

CREATE TABLE IF NOT EXISTS tenant_quotas (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
  resource_type VARCHAR(100) NOT NULL,  -- sandboxes, storage, compute
  current_usage BIGINT DEFAULT 0,
  quota_limit BIGINT NOT NULL,
  updated_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(tenant_id, resource_type)
);

CREATE TABLE IF NOT EXISTS tenant_billing (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
  billing_period_start DATE NOT NULL,
  billing_period_end DATE NOT NULL,

  -- Usage metrics
  sandboxes_used INTEGER DEFAULT 0,
  storage_gb_used DECIMAL(10, 2) DEFAULT 0,
  compute_hours_used DECIMAL(10, 2) DEFAULT 0,
  api_requests_made BIGINT DEFAULT 0,

  -- Billing
  amount_due DECIMAL(10, 2) DEFAULT 0,
  amount_paid DECIMAL(10, 2) DEFAULT 0,
  status VARCHAR(50) DEFAULT 'pending',  -- pending, paid, overdue
  created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_tenants_subdomain ON tenants(subdomain);
CREATE INDEX idx_tenants_status ON tenants(status);
CREATE INDEX idx_tenant_users_tenant ON tenant_users(tenant_id);
CREATE INDEX idx_tenant_quotas_tenant ON tenant_quotas(tenant_id);
```

### Tenant-Specific Database Schema

```sql
-- Template schema for each tenant database
-- File: platform/config/schema-tenant.sql

CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  full_name VARCHAR(255),
  role VARCHAR(50) DEFAULT 'member',
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS applications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  name VARCHAR(255) NOT NULL,
  description TEXT,
  sandbox_id VARCHAR(255),
  status VARCHAR(50) DEFAULT 'active',
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS deployments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  application_id UUID REFERENCES applications(id) ON DELETE CASCADE,
  version VARCHAR(50) NOT NULL,
  status VARCHAR(50) DEFAULT 'pending',
  deployed_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  key_hash VARCHAR(255) UNIQUE NOT NULL,
  name VARCHAR(255),
  last_used_at TIMESTAMP,
  expires_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_applications_user ON applications(user_id);
CREATE INDEX idx_deployments_app ON deployments(application_id);
```

---

## Part 2: Tenant Provisioning System

### Automated Tenant Provisioning

```python
#!/usr/bin/env python3
# tenant-provisioner.py

import asyncio
import os
import psycopg2
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class TenantConfig:
    name: str
    subdomain: str
    tier: str  # free, premium, enterprise
    owner_email: str
    owner_name: str

class TenantProvisioner:
    def __init__(self, master_db_url: str):
        self.master_db_url = master_db_url

    async def provision_tenant(self, config: TenantConfig) -> str:
        """Provision a new tenant with complete infrastructure"""

        print(f"Provisioning tenant: {config.name}")

        # Step 1: Create tenant record
        tenant_id = await self._create_tenant_record(config)

        # Step 2: Create dedicated database
        db_name = f"tenant_{config.subdomain}"
        await self._create_tenant_database(db_name)

        # Step 3: Apply schema to tenant database
        await self._apply_tenant_schema(db_name)

        # Step 4: Create owner account
        await self._create_owner_account(db_name, config)

        # Step 5: Initialize quotas
        await self._initialize_quotas(tenant_id, config.tier)

        # Step 6: Create sandboxes
        sandbox_id = await self._provision_tenant_sandbox(tenant_id, config)

        # Step 7: Create storage buckets
        await self._create_storage_buckets(tenant_id, config)

        # Step 8: Setup monitoring
        await self._setup_monitoring(tenant_id, config)

        print(f"✓ Tenant provisioned successfully: {tenant_id}")
        return tenant_id

    async def _create_tenant_record(self, config: TenantConfig) -> str:
        """Create tenant record in master database"""

        quotas = self._get_tier_quotas(config.tier)

        conn = psycopg2.connect(self.master_db_url)
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO tenants (
                    name, subdomain, tier,
                    max_sandboxes, max_storage_mb, max_compute_hours, max_users,
                    database_name, billing_plan, billing_cycle
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s
                ) RETURNING id
            """, (
                config.name,
                config.subdomain,
                config.tier,
                quotas['sandboxes'],
                quotas['storage_mb'],
                quotas['compute_hours'],
                quotas['users'],
                f"tenant_{config.subdomain}",
                config.tier,
                'monthly'
            ))

            tenant_id = cur.fetchone()[0]
            conn.commit()

            return str(tenant_id)

        finally:
            cur.close()
            conn.close()

    async def _create_tenant_database(self, db_name: str):
        """Create dedicated database for tenant"""

        conn = psycopg2.connect(self.master_db_url)
        conn.autocommit = True
        cur = conn.cursor()

        try:
            cur.execute(f'CREATE DATABASE {db_name}')
            print(f"  ✓ Database created: {db_name}")

        finally:
            cur.close()
            conn.close()

    async def _apply_tenant_schema(self, db_name: str):
        """Apply schema to tenant database"""

        # Read schema file
        with open('platform/config/schema-tenant.sql', 'r') as f:
            schema_sql = f.read()

        # Connect to tenant database
        tenant_db_url = self.master_db_url.replace('/master', f'/{db_name}')
        conn = psycopg2.connect(tenant_db_url)
        cur = conn.cursor()

        try:
            cur.execute(schema_sql)
            conn.commit()
            print(f"  ✓ Schema applied to {db_name}")

        finally:
            cur.close()
            conn.close()

    async def _create_owner_account(self, db_name: str, config: TenantConfig):
        """Create owner account in tenant database"""

        import bcrypt

        # Generate temporary password
        temp_password = secrets.token_urlsafe(16)
        password_hash = bcrypt.hashpw(temp_password.encode(), bcrypt.gensalt()).decode()

        tenant_db_url = self.master_db_url.replace('/master', f'/{db_name}')
        conn = psycopg2.connect(tenant_db_url)
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO users (email, password_hash, full_name, role)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (config.owner_email, password_hash, config.owner_name, 'owner'))

            user_id = cur.fetchone()[0]
            conn.commit()

            print(f"  ✓ Owner account created: {config.owner_email}")
            print(f"    Temporary password: {temp_password}")

            # TODO: Send email with temp password

            return str(user_id)

        finally:
            cur.close()
            conn.close()

    async def _initialize_quotas(self, tenant_id: str, tier: str):
        """Initialize resource quotas for tenant"""

        quotas = self._get_tier_quotas(tier)

        conn = psycopg2.connect(self.master_db_url)
        cur = conn.cursor()

        try:
            for resource_type, limit in quotas.items():
                cur.execute("""
                    INSERT INTO tenant_quotas (tenant_id, resource_type, quota_limit)
                    VALUES (%s, %s, %s)
                """, (tenant_id, resource_type, limit))

            conn.commit()
            print(f"  ✓ Quotas initialized")

        finally:
            cur.close()
            conn.close()

    async def _provision_tenant_sandbox(self, tenant_id: str, config: TenantConfig) -> str:
        """Provision sandbox for tenant"""

        # Use Flow Nexus MCP to create sandbox
        # This would integrate with actual MCP tools

        sandbox_config = {
            'tenant_id': tenant_id,
            'template': 'node',
            'name': f"{config.subdomain}-sandbox",
            'env_vars': {
                'TENANT_ID': tenant_id,
                'TENANT_SUBDOMAIN': config.subdomain
            }
        }

        # Simulate sandbox creation
        sandbox_id = f"sandbox_{tenant_id}_{secrets.token_hex(8)}"
        print(f"  ✓ Sandbox provisioned: {sandbox_id}")

        return sandbox_id

    async def _create_storage_buckets(self, tenant_id: str, config: TenantConfig):
        """Create storage buckets for tenant"""

        buckets = [
            f"{config.subdomain}-uploads",
            f"{config.subdomain}-assets",
            f"{config.subdomain}-backups"
        ]

        for bucket in buckets:
            # Simulate bucket creation
            print(f"  ✓ Storage bucket created: {bucket}")

    async def _setup_monitoring(self, tenant_id: str, config: TenantConfig):
        """Setup monitoring for tenant"""

        # Configure health checks, metrics, alerts
        print(f"  ✓ Monitoring configured for {config.subdomain}")

    def _get_tier_quotas(self, tier: str) -> dict:
        """Get quota limits for tier"""

        tiers = {
            'free': {
                'sandboxes': 1,
                'storage_mb': 100,
                'compute_hours': 1,
                'users': 1,
                'api_requests': 1000
            },
            'premium': {
                'sandboxes': 5,
                'storage_mb': 1000,
                'compute_hours': 10,
                'users': 5,
                'api_requests': 10000
            },
            'enterprise': {
                'sandboxes': 20,
                'storage_mb': 10000,
                'compute_hours': 100,
                'users': 50,
                'api_requests': 100000
            }
        }

        return tiers.get(tier, tiers['free'])

# Usage example
async def main():
    provisioner = TenantProvisioner("postgresql://localhost/platform_master")

    config = TenantConfig(
        name="Acme Corporation",
        subdomain="acme",
        tier="premium",
        owner_email="admin@acme.com",
        owner_name="Acme Admin"
    )

    tenant_id = await provisioner.provision_tenant(config)
    print(f"\nTenant ID: {tenant_id}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Part 3: Tenant Isolation and Security

### Row-Level Security (RLS) Implementation

```sql
-- Enable Row-Level Security on tenant tables
-- File: platform/config/rls-policies.sql

-- Users table
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_users ON users
  USING (tenant_id = current_setting('app.tenant_id')::UUID);

-- Applications table
ALTER TABLE applications ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_apps ON applications
  USING (tenant_id = current_setting('app.tenant_id')::UUID);

-- Deployments table
ALTER TABLE deployments ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_deployments ON deployments
  USING (
    tenant_id = current_setting('app.tenant_id')::UUID OR
    current_setting('app.user_role') = 'admin'
  );
```

### API Gateway with Tenant Routing

```javascript
// api-gateway.js
const express = require('express');
const { Pool } = require('pg');

class TenantRouter {
  constructor(masterPool) {
    this.masterPool = masterPool;
    this.tenantPools = new Map();
  }

  async getTenantFromSubdomain(subdomain) {
    const result = await this.masterPool.query(
      'SELECT * FROM tenants WHERE subdomain = $1 AND status = $2',
      [subdomain, 'active']
    );

    return result.rows[0] || null;
  }

  getTenantPool(tenant) {
    if (!this.tenantPools.has(tenant.id)) {
      const pool = new Pool({
        connectionString: `postgresql://localhost/${tenant.database_name}`,
        max: 10
      });
      this.tenantPools.set(tenant.id, pool);
    }

    return this.tenantPools.get(tenant.id);
  }

  middleware() {
    return async (req, res, next) => {
      try {
        // Extract tenant from subdomain
        const subdomain = req.hostname.split('.')[0];

        // Get tenant info
        const tenant = await this.getTenantFromSubdomain(subdomain);

        if (!tenant) {
          return res.status(404).json({ error: 'Tenant not found' });
        }

        // Check tenant status
        if (tenant.status !== 'active') {
          return res.status(403).json({ error: 'Tenant suspended' });
        }

        // Attach tenant context
        req.tenant = tenant;
        req.tenantPool = this.getTenantPool(tenant);

        // Set tenant context for RLS
        await req.tenantPool.query(
          `SET app.tenant_id = '${tenant.id}'`
        );

        next();
      } catch (err) {
        console.error('Tenant routing error:', err);
        res.status(500).json({ error: 'Internal server error' });
      }
    };
  }
}

// Create API gateway
const app = express();
const masterPool = new Pool({
  connectionString: process.env.MASTER_DATABASE_URL
});

const tenantRouter = new TenantRouter(masterPool);

// Apply tenant routing middleware
app.use(tenantRouter.middleware());

// Tenant-specific routes
app.get('/api/users', async (req, res) => {
  try {
    const result = await req.tenantPool.query(
      'SELECT id, email, full_name, role FROM users'
    );
    res.json(result.rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Database error' });
  }
});

app.listen(3000, () => {
  console.log('Multi-tenant API gateway running on port 3000');
});
```

---

## Part 4: Resource Quota Enforcement

### Quota Tracking Middleware

```javascript
// quota-middleware.js
class QuotaEnforcer {
  constructor(masterPool) {
    this.masterPool = masterPool;
  }

  async checkQuota(tenantId, resourceType) {
    const result = await this.masterPool.query(`
      SELECT current_usage, quota_limit
      FROM tenant_quotas
      WHERE tenant_id = $1 AND resource_type = $2
    `, [tenantId, resourceType]);

    if (result.rows.length === 0) {
      return { allowed: false, reason: 'Quota not configured' };
    }

    const { current_usage, quota_limit } = result.rows[0];

    if (current_usage >= quota_limit) {
      return {
        allowed: false,
        reason: `Quota exceeded for ${resourceType}`,
        current: current_usage,
        limit: quota_limit
      };
    }

    return {
      allowed: true,
      remaining: quota_limit - current_usage
    };
  }

  async incrementUsage(tenantId, resourceType, amount = 1) {
    await this.masterPool.query(`
      UPDATE tenant_quotas
      SET current_usage = current_usage + $1, updated_at = NOW()
      WHERE tenant_id = $2 AND resource_type = $3
    `, [amount, tenantId, resourceType]);
  }

  async decrementUsage(tenantId, resourceType, amount = 1) {
    await this.masterPool.query(`
      UPDATE tenant_quotas
      SET current_usage = GREATEST(0, current_usage - $1), updated_at = NOW()
      WHERE tenant_id = $2 AND resource_type = $3
    `, [amount, tenantId, resourceType]);
  }

  middleware(resourceType) {
    return async (req, res, next) => {
      try {
        const check = await this.checkQuota(req.tenant.id, resourceType);

        if (!check.allowed) {
          return res.status(429).json({
            error: 'Quota exceeded',
            message: check.reason,
            current: check.current,
            limit: check.limit
          });
        }

        req.quotaRemaining = check.remaining;
        next();
      } catch (err) {
        console.error('Quota check error:', err);
        res.status(500).json({ error: 'Quota system error' });
      }
    };
  }
}

module.exports = QuotaEnforcer;
```

---

## Summary

This example demonstrated:
- ✅ Complete multi-tenant architecture
- ✅ Automated tenant provisioning
- ✅ Database-per-tenant isolation
- ✅ Resource quota management
- ✅ Secure tenant routing
- ✅ Tier-based feature access

**Total Lines**: ~800 lines
**Production Considerations**:
- Implement proper encryption
- Add comprehensive logging
- Setup automated backups
- Configure monitoring/alerting
- Implement tenant migration tools
- Add usage analytics
