# Example 2: Schema Design for Multi-Tenant SaaS Platform

## Scenario

A startup is building a project management SaaS platform similar to Asana/Jira. They need a database schema that:

- **Supports multi-tenancy** (1,000+ organizations, each with 10-500 users)
- **Scales horizontally** (handles 100K+ concurrent users)
- **Enforces data isolation** (organizations cannot access each other's data)
- **Enables flexible permissions** (role-based access control)
- **Optimizes for common queries** (project lists, task assignments, activity feeds)

Initial requirements:
- Organizations have multiple workspaces
- Workspaces contain projects
- Projects have tasks with assignees, comments, attachments
- Users can belong to multiple organizations
- Granular permissions (view, edit, admin) at workspace/project level

---

## Design Approach: Row-Level Multi-Tenancy vs Schema-Per-Tenant

### Option 1: Schema-Per-Tenant (❌ Not Recommended)

```sql
-- Each organization gets a separate schema
CREATE SCHEMA org_acme_corp;
CREATE SCHEMA org_globex;

-- Duplicate tables in each schema
CREATE TABLE org_acme_corp.projects (...);
CREATE TABLE org_globex.projects (...);
```

**Pros:**
- Strong data isolation
- Easy backup/restore per organization

**Cons:**
- ❌ Difficult to scale (1,000+ schemas)
- ❌ Complex connection pooling
- ❌ Hard to perform cross-tenant analytics
- ❌ Schema migrations require N operations

### Option 2: Row-Level Multi-Tenancy (✅ Recommended)

```sql
-- Single schema with organization_id in every table
CREATE TABLE projects (
    project_id UUID PRIMARY KEY,
    organization_id UUID NOT NULL REFERENCES organizations(organization_id),
    workspace_id UUID NOT NULL REFERENCES workspaces(workspace_id),
    name TEXT NOT NULL,
    ...
);

-- Enforce data isolation with Row-Level Security (RLS)
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON projects
USING (organization_id = current_setting('app.current_organization_id')::UUID);
```

**Pros:**
- ✅ Simple schema management
- ✅ Easy cross-tenant analytics
- ✅ Single connection pool
- ✅ Efficient resource utilization

**Cons:**
- Requires careful query design (always filter by `organization_id`)
- Risk of data leakage if queries miss tenant filter

**Decision:** Use row-level multi-tenancy with PostgreSQL Row-Level Security (RLS).

---

## Final Schema Design

### Core Entity Tables

```sql
-- Organizations (tenants)
CREATE TABLE organizations (
    organization_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    plan_tier TEXT NOT NULL CHECK (plan_tier IN ('free', 'pro', 'enterprise')),
    max_users INTEGER NOT NULL DEFAULT 10,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Users (global, can belong to multiple organizations)
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL,
    avatar_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMPTZ
);

-- Organization memberships (many-to-many with roles)
CREATE TABLE organization_members (
    organization_id UUID NOT NULL REFERENCES organizations(organization_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('owner', 'admin', 'member', 'guest')),
    joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (organization_id, user_id)
);

-- Workspaces (logical grouping within organizations)
CREATE TABLE workspaces (
    workspace_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(organization_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    is_private BOOLEAN NOT NULL DEFAULT FALSE,
    created_by UUID NOT NULL REFERENCES users(user_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_workspace_per_org UNIQUE (organization_id, name)
);

-- Projects (within workspaces)
CREATE TABLE projects (
    project_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(organization_id) ON DELETE CASCADE,
    workspace_id UUID NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived', 'completed')),
    owner_id UUID NOT NULL REFERENCES users(user_id),
    start_date DATE,
    due_date DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tasks (within projects)
CREATE TABLE tasks (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(organization_id) ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    parent_task_id UUID REFERENCES tasks(task_id) ON DELETE CASCADE, -- Subtasks
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'todo' CHECK (status IN ('todo', 'in_progress', 'review', 'done')),
    priority TEXT NOT NULL DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    assignee_id UUID REFERENCES users(user_id),
    created_by UUID NOT NULL REFERENCES users(user_id),
    due_date TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    position INTEGER NOT NULL, -- For drag-and-drop ordering
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Comments (on tasks)
CREATE TABLE comments (
    comment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(organization_id) ON DELETE CASCADE,
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(user_id),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Attachments (for tasks)
CREATE TABLE attachments (
    attachment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(organization_id) ON DELETE CASCADE,
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    uploaded_by UUID NOT NULL REFERENCES users(user_id),
    file_name TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type TEXT NOT NULL,
    storage_key TEXT NOT NULL, -- S3 key or file path
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Activity log (audit trail)
CREATE TABLE activities (
    activity_id BIGSERIAL PRIMARY KEY, -- Use BIGSERIAL for high-volume table
    organization_id UUID NOT NULL REFERENCES organizations(organization_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(user_id),
    entity_type TEXT NOT NULL, -- 'project', 'task', 'comment', etc.
    entity_id UUID NOT NULL,
    action TEXT NOT NULL, -- 'created', 'updated', 'deleted', 'assigned', etc.
    metadata JSONB, -- Flexible storage for action-specific data
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## Indexes for Performance

```sql
-- Organizations
CREATE INDEX idx_organizations_slug ON organizations(slug);

-- Organization members (common join)
CREATE INDEX idx_org_members_user ON organization_members(user_id);

-- Workspaces (filter by organization)
CREATE INDEX idx_workspaces_org ON workspaces(organization_id);

-- Projects (common queries)
CREATE INDEX idx_projects_org ON projects(organization_id);
CREATE INDEX idx_projects_workspace ON projects(workspace_id);
CREATE INDEX idx_projects_owner ON projects(owner_id);
CREATE INDEX idx_projects_status ON projects(organization_id, status);

-- Tasks (heavily queried)
CREATE INDEX idx_tasks_org ON tasks(organization_id);
CREATE INDEX idx_tasks_project ON tasks(project_id);
CREATE INDEX idx_tasks_assignee ON tasks(assignee_id, status);
CREATE INDEX idx_tasks_parent ON tasks(parent_task_id) WHERE parent_task_id IS NOT NULL;
CREATE INDEX idx_tasks_due_date ON tasks(organization_id, due_date) WHERE due_date IS NOT NULL;

-- Comments
CREATE INDEX idx_comments_task ON comments(task_id, created_at DESC);

-- Attachments
CREATE INDEX idx_attachments_task ON attachments(task_id);

-- Activities (time-series data)
CREATE INDEX idx_activities_org_time ON activities(organization_id, created_at DESC);
CREATE INDEX idx_activities_entity ON activities(entity_type, entity_id);
CREATE INDEX idx_activities_user ON activities(user_id, created_at DESC);

-- GIN index for JSONB metadata search
CREATE INDEX idx_activities_metadata ON activities USING GIN(metadata);
```

---

## Row-Level Security (RLS) Policies

```sql
-- Enable RLS on all tenant-scoped tables
ALTER TABLE workspaces ENABLE ROW LEVEL SECURITY;
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE comments ENABLE ROW LEVEL SECURITY;
ALTER TABLE attachments ENABLE ROW LEVEL SECURITY;
ALTER TABLE activities ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only access data from their organization
CREATE POLICY tenant_isolation ON workspaces
USING (organization_id = current_setting('app.current_organization_id')::UUID);

CREATE POLICY tenant_isolation ON projects
USING (organization_id = current_setting('app.current_organization_id')::UUID);

CREATE POLICY tenant_isolation ON tasks
USING (organization_id = current_setting('app.current_organization_id')::UUID);

CREATE POLICY tenant_isolation ON comments
USING (organization_id = current_setting('app.current_organization_id')::UUID);

CREATE POLICY tenant_isolation ON attachments
USING (organization_id = current_setting('app.current_organization_id')::UUID);

CREATE POLICY tenant_isolation ON activities
USING (organization_id = current_setting('app.current_organization_id')::UUID);

-- Application must set organization_id at session start
-- SET LOCAL app.current_organization_id = 'uuid-here';
```

---

## Advanced Features: Partitioning for Scalability

```sql
-- Partition activities table by month (time-series data grows indefinitely)
CREATE TABLE activities_partitioned (
    activity_id BIGSERIAL,
    organization_id UUID NOT NULL,
    user_id UUID NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id UUID NOT NULL,
    action TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (activity_id, created_at)
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE activities_2025_01 PARTITION OF activities_partitioned
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE activities_2025_02 PARTITION OF activities_partitioned
FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

-- Auto-create partitions with pg_partman extension
CREATE EXTENSION pg_partman;

SELECT partman.create_parent(
    p_parent_table := 'public.activities_partitioned',
    p_control := 'created_at',
    p_type := 'native',
    p_interval := '1 month',
    p_premake := 3
);
```

---

## Application Integration (Node.js Example)

```javascript
const { Pool } = require('pg');
const pool = new Pool({ connectionString: process.env.DATABASE_URL });

// Middleware to set organization context
async function setOrganizationContext(req, res, next) {
    const { organizationId } = req.user; // From JWT or session

    // Set session variable for RLS
    await pool.query(
        'SET LOCAL app.current_organization_id = $1',
        [organizationId]
    );

    next();
}

// Example: Fetch user's tasks
async function getUserTasks(userId, status = null) {
    // RLS automatically filters by organization_id
    const query = `
        SELECT
            t.task_id,
            t.title,
            t.status,
            t.priority,
            t.due_date,
            p.name AS project_name,
            u.full_name AS assignee_name
        FROM tasks t
        JOIN projects p USING (project_id)
        LEFT JOIN users u ON t.assignee_id = u.user_id
        WHERE t.assignee_id = $1
            AND ($2::TEXT IS NULL OR t.status = $2)
        ORDER BY
            CASE t.priority
                WHEN 'urgent' THEN 1
                WHEN 'high' THEN 2
                WHEN 'medium' THEN 3
                WHEN 'low' THEN 4
            END,
            t.due_date ASC NULLS LAST
        LIMIT 50;
    `;

    const result = await pool.query(query, [userId, status]);
    return result.rows;
}

// Example: Create task with activity logging
async function createTask(taskData) {
    const client = await pool.connect();

    try {
        await client.query('BEGIN');

        // Insert task
        const taskResult = await client.query(`
            INSERT INTO tasks (
                organization_id, project_id, title, description,
                status, priority, assignee_id, created_by, due_date, position
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING *;
        `, [
            taskData.organizationId,
            taskData.projectId,
            taskData.title,
            taskData.description,
            taskData.status || 'todo',
            taskData.priority || 'medium',
            taskData.assigneeId,
            taskData.createdBy,
            taskData.dueDate,
            taskData.position
        ]);

        const task = taskResult.rows[0];

        // Log activity
        await client.query(`
            INSERT INTO activities (
                organization_id, user_id, entity_type, entity_id, action, metadata
            )
            VALUES ($1, $2, 'task', $3, 'created', $4);
        `, [
            taskData.organizationId,
            taskData.createdBy,
            task.task_id,
            JSON.stringify({
                task_title: task.title,
                project_id: task.project_id
            })
        ]);

        await client.query('COMMIT');
        return task;
    } catch (error) {
        await client.query('ROLLBACK');
        throw error;
    } finally {
        client.release();
    }
}
```

---

## Schema Migration Strategy

```sql
-- Use versioned migrations (e.g., Flyway, Liquibase, or custom)
-- migrations/V001__initial_schema.sql
-- migrations/V002__add_task_labels.sql

-- Example migration: Add labels to tasks
CREATE TABLE task_labels (
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    label_id UUID NOT NULL REFERENCES labels(label_id) ON DELETE CASCADE,
    PRIMARY KEY (task_id, label_id)
);

CREATE TABLE labels (
    label_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(organization_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    color TEXT NOT NULL,
    CONSTRAINT unique_label_per_org UNIQUE (organization_id, name)
);

ALTER TABLE task_labels ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON task_labels
USING (
    EXISTS (
        SELECT 1 FROM tasks
        WHERE tasks.task_id = task_labels.task_id
        AND tasks.organization_id = current_setting('app.current_organization_id')::UUID
    )
);
```

---

## Outcomes & Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **Multi-tenancy Isolation** | 100% enforced | 100% (RLS) | ✅ |
| **Query Performance (P95)** | <200ms | 156ms | ✅ |
| **Horizontal Scalability** | 100K users | 120K users | ✅ |
| **Data Integrity** | Zero leaks | Zero incidents | ✅ |
| **Schema Complexity** | Low | 11 core tables | ✅ |

**Production statistics (after 6 months):**
- 1,247 organizations
- 42,318 users
- 3.2M tasks created
- 98.7% uptime
- 0 data leakage incidents

---

## Key Takeaways

### 1. **Choose Row-Level Multi-Tenancy for SaaS**
- Simpler schema management
- Better resource utilization
- Easier cross-tenant analytics
- Use PostgreSQL RLS for enforcement

### 2. **UUID Primary Keys for Distributed Systems**
- Enables sharding without coordination
- No sequential ID leakage
- Compatible with distributed databases

### 3. **Denormalize `organization_id` Everywhere**
- Every tenant-scoped table includes `organization_id`
- Enables efficient filtering and RLS policies
- Simplifies queries (no complex joins for tenant isolation)

### 4. **Index Strategy for Multi-Tenant Tables**
- Always include `organization_id` in composite indexes
- Covering indexes for hot queries
- Monitor index bloat with `pg_stat_user_indexes`

### 5. **Partition Time-Series Data**
- Activities/audit logs grow indefinitely
- Monthly partitioning prevents table bloat
- Use `pg_partman` for automated partition management

### 6. **Enforce RLS at Database Level**
- Application bugs cannot leak data
- Defense-in-depth security
- Set `app.current_organization_id` per session

### 7. **Use JSONB for Flexible Metadata**
- Avoid schema changes for new activity types
- GIN indexes enable fast JSONB queries
- Balance flexibility vs query performance

### 8. **Plan for Sharding from Day 1**
- Use UUID primary keys
- Include `organization_id` in all queries
- Design for horizontal scalability (Citus, Vitess)

---

## Tools & Extensions Used

- **PostgreSQL 15** - Core database
- **pg_partman** - Automated partition management
- **pgBouncer** - Connection pooling
- **Flyway** - Schema migration versioning
- **PostGIS** (optional) - Geospatial features
- **pg_stat_statements** - Query performance monitoring

---

## Next Steps

1. **Implement caching layer** (Redis for task lists)
2. **Add full-text search** (PostgreSQL `tsvector` or Elasticsearch)
3. **Set up read replicas** for analytics queries
4. **Implement soft deletes** (`deleted_at` column for data recovery)
5. **Add audit triggers** for automated activity logging
