---
name: sql-database-specialist
description: SQL database specialist for PostgreSQL/MySQL optimization, EXPLAIN plan analysis, index optimization, query rewriting, partitioning strategies, connection pooling, and database performance tuning. Use when optimizing slow queries, designing efficient database schemas, implementing replication/HA, or requiring SQL best practices. Handles JSONB queries, full-text search, vacuum maintenance, and migration strategies.
category: Database Specialists
complexity: High
triggers: ["sql", "postgresql", "mysql", "database optimization", "query optimization", "indexes", "explain plan", "database performance", "postgres", "mariadb"]
---

# SQL Database Specialist

Expert SQL database optimization, schema design, and performance tuning for PostgreSQL and MySQL.

## Purpose

Comprehensive SQL expertise including EXPLAIN plan analysis, index optimization, query rewriting, partitioning, replication, and performance tuning. Ensures databases are fast, scalable, and maintainable.

## When to Use

- Optimizing slow database queries
- Designing efficient database schemas
- Analyzing EXPLAIN plans
- Creating optimal indexes
- Implementing database partitioning
- Setting up replication and high availability
- Migrating data with zero downtime
- Troubleshooting performance issues

## Prerequisites

**Required**: SQL basics, understanding of relational databases, familiarity with PostgreSQL or MySQL

**Agents**: `backend-dev`, `perf-analyzer`, `system-architect`, `code-analyzer`

## Core Workflows

### Workflow 1: Query Optimization with EXPLAIN

**Step 1: Analyze EXPLAIN Plan (PostgreSQL)**

```sql
-- EXPLAIN shows query plan
EXPLAIN
SELECT u.name, o.total
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE o.created_at > '2024-01-01';

-- EXPLAIN ANALYZE executes and shows actual timings
EXPLAIN (ANALYZE, BUFFERS)
SELECT u.name, o.total
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE o.created_at > '2024-01-01';
```

**Key Metrics to Check**:
- **Seq Scan** (bad): Full table scan, add index
- **Index Scan** (good): Using index
- **Bitmap Index Scan** (good): Efficient for multiple conditions
- **Nested Loop** (watch out): Can be slow for large datasets
- **Hash Join** (usually good): Efficient join method
- **Cost**: Estimated cost (lower is better)
- **Actual time**: Real execution time

**Step 2: Create Optimal Index**

```sql
-- ❌ SLOW: No index on created_at
SELECT * FROM orders WHERE created_at > '2024-01-01';

-- ✅ FAST: Create index
CREATE INDEX idx_orders_created_at ON orders (created_at);

-- ✅ COMPOUND INDEX: For multiple columns
CREATE INDEX idx_orders_user_created
ON orders (user_id, created_at);

-- ✅ PARTIAL INDEX: For filtered queries
CREATE INDEX idx_orders_pending
ON orders (created_at)
WHERE status = 'pending';

-- ✅ COVERING INDEX: Include frequently queried columns
CREATE INDEX idx_orders_covering
ON orders (user_id, created_at)
INCLUDE (total, status);
```

**Step 3: Rewrite Query for Performance**

```sql
-- ❌ SLOW: N+1 query pattern
SELECT id, name FROM users;
-- Then for each user:
SELECT * FROM orders WHERE user_id = ?;

-- ✅ FAST: Single query with JOIN
SELECT u.id, u.name, o.*
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- ❌ SLOW: NOT IN with subquery
SELECT * FROM users
WHERE id NOT IN (SELECT user_id FROM orders);

-- ✅ FAST: LEFT JOIN with NULL check
SELECT u.*
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.user_id IS NULL;

-- ❌ SLOW: OR conditions prevent index use
SELECT * FROM orders
WHERE user_id = 123 OR status = 'pending';

-- ✅ FAST: UNION ALL with indexes
SELECT * FROM orders WHERE user_id = 123
UNION ALL
SELECT * FROM orders WHERE status = 'pending' AND user_id != 123;
```

### Workflow 2: Table Partitioning (PostgreSQL)

**Step 1: Create Partitioned Table**

```sql
-- Range partitioning by date
CREATE TABLE orders (
  id BIGSERIAL,
  user_id INT NOT NULL,
  created_at DATE NOT NULL,
  total DECIMAL(10, 2),
  status VARCHAR(20)
) PARTITION BY RANGE (created_at);

-- Create partitions
CREATE TABLE orders_2024_q1 PARTITION OF orders
FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE orders_2024_q2 PARTITION OF orders
FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

-- Create index on each partition
CREATE INDEX idx_orders_2024_q1_user_id
ON orders_2024_q1 (user_id);

-- Queries automatically use correct partition
SELECT * FROM orders
WHERE created_at >= '2024-02-01'
  AND created_at < '2024-03-01';
-- Only scans orders_2024_q1 partition
```

**Step 2: List Partitioning by Status**

```sql
CREATE TABLE orders_by_status (
  id BIGSERIAL,
  status VARCHAR(20) NOT NULL,
  -- ... other columns
) PARTITION BY LIST (status);

CREATE TABLE orders_pending PARTITION OF orders_by_status
FOR VALUES IN ('pending', 'processing');

CREATE TABLE orders_completed PARTITION OF orders_by_status
FOR VALUES IN ('completed', 'shipped');
```

### Workflow 3: PostgreSQL-Specific Optimizations

**Step 1: JSONB Queries with GIN Index**

```sql
-- JSONB column for flexible data
CREATE TABLE events (
  id BIGSERIAL PRIMARY KEY,
  data JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- GIN index for JSONB containment queries
CREATE INDEX idx_events_data_gin ON events USING GIN (data);

-- Query JSONB efficiently
SELECT * FROM events
WHERE data @> '{"user_id": 123}';

SELECT * FROM events
WHERE data -> 'metadata' ->> 'source' = 'web';

-- Extract and index specific JSONB field
CREATE INDEX idx_events_user_id
ON events ((data->>'user_id'));
```

**Step 2: Full-Text Search**

```sql
-- Add tsvector column
ALTER TABLE articles
ADD COLUMN search_vector tsvector;

-- Populate tsvector
UPDATE articles
SET search_vector = to_tsvector('english', title || ' ' || content);

-- Create GIN index for full-text search
CREATE INDEX idx_articles_search
ON articles USING GIN (search_vector);

-- Search query
SELECT * FROM articles
WHERE search_vector @@ to_tsquery('english', 'database & optimization');

-- Trigger to auto-update search_vector
CREATE TRIGGER articles_search_update
BEFORE INSERT OR UPDATE ON articles
FOR EACH ROW EXECUTE FUNCTION
  tsvector_update_trigger(search_vector, 'pg_catalog.english', title, content);
```

**Step 3: Vacuum and Maintenance**

```sql
-- Manual VACUUM to reclaim space
VACUUM VERBOSE ANALYZE orders;

-- VACUUM FULL for maximum space reclamation (locks table)
VACUUM FULL orders;

-- Auto-vacuum settings (postgresql.conf)
autovacuum = on
autovacuum_vacuum_scale_factor = 0.1  -- Vacuum when 10% of rows change
autovacuum_analyze_scale_factor = 0.05

-- Monitor bloat
SELECT schemaname, tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
  n_live_tup, n_dead_tup
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;
```

### Workflow 4: Connection Pooling

**PostgreSQL with PgBouncer**

```ini
# pgbouncer.ini
[databases]
mydb = host=localhost dbname=mydb

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction  # or session, statement
max_client_conn = 1000
default_pool_size = 25
```

**Node.js with pg-pool**

```javascript
const { Pool } = require('pg');

const pool = new Pool({
  host: 'localhost',
  port: 5432,
  database: 'mydb',
  user: 'dbuser',
  password: 'password',
  max: 20,  // Max connections in pool
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

// Query with automatic connection management
const result = await pool.query(
  'SELECT * FROM users WHERE id = $1',
  [userId]
);

// Transaction
const client = await pool.connect();
try {
  await client.query('BEGIN');
  await client.query('UPDATE accounts SET balance = balance - $1 WHERE id = $2', [100, 1]);
  await client.query('UPDATE accounts SET balance = balance + $1 WHERE id = $2', [100, 2]);
  await client.query('COMMIT');
} catch (e) {
  await client.query('ROLLBACK');
  throw e;
} finally {
  client.release();
}
```

### Workflow 5: Zero-Downtime Migration

```sql
-- Step 1: Add new column (non-blocking)
ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT false;

-- Step 2: Backfill data in batches
DO $$
DECLARE
  batch_size INT := 1000;
  last_id INT := 0;
BEGIN
  LOOP
    UPDATE users
    SET email_verified = (email IS NOT NULL AND email != '')
    WHERE id > last_id AND id <= last_id + batch_size;

    EXIT WHEN NOT FOUND;
    last_id := last_id + batch_size;
    COMMIT;  -- Commit each batch
    PERFORM pg_sleep(0.1);  -- Pause to reduce load
  END LOOP;
END $$;

-- Step 3: Add NOT NULL constraint (after backfill)
ALTER TABLE users ALTER COLUMN email_verified SET NOT NULL;
```

## Best Practices

**1. Always Use Parameterized Queries**
```sql
-- ✅ GOOD: Prevents SQL injection
SELECT * FROM users WHERE id = $1;

-- ❌ BAD: SQL injection vulnerability
SELECT * FROM users WHERE id = ' + userId + ';
```

**2. Index Foreign Keys**
```sql
-- ✅ GOOD: Index foreign keys for JOIN performance
CREATE INDEX idx_orders_user_id ON orders (user_id);
```

**3. Use Appropriate Data Types**
```sql
-- ✅ GOOD: Efficient data types
CREATE TABLE users (
  id BIGSERIAL PRIMARY KEY,
  email VARCHAR(255) NOT NULL,
  age SMALLINT,
  balance DECIMAL(10, 2),
  is_active BOOLEAN DEFAULT true
);

-- ❌ BAD: Wasteful data types
CREATE TABLE users (
  id VARCHAR(255),  -- Should be BIGSERIAL
  age INT,  -- Should be SMALLINT
  balance FLOAT  -- Should be DECIMAL for money
);
```

**4. Limit Large Result Sets**
```sql
-- ✅ GOOD: Pagination with LIMIT/OFFSET
SELECT * FROM orders
ORDER BY created_at DESC
LIMIT 20 OFFSET 0;

-- Better: Cursor-based pagination
SELECT * FROM orders
WHERE id > last_seen_id
ORDER BY id
LIMIT 20;
```

**5. Monitor Query Performance**
```sql
-- Enable pg_stat_statements (PostgreSQL)
CREATE EXTENSION pg_stat_statements;

-- Find slow queries
SELECT
  query,
  calls,
  total_exec_time / 1000 AS total_seconds,
  mean_exec_time / 1000 AS mean_seconds
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;
```

## Quality Criteria

- ✅ EXPLAIN plan analyzed for all queries
- ✅ Indexes created for all foreign keys
- ✅ No sequential scans on large tables
- ✅ Query execution time <100ms for simple queries
- ✅ Connection pooling configured
- ✅ Vacuum running automatically
- ✅ Backups automated and tested

## Troubleshooting

**Issue**: Query slow despite index
**Solution**: Check EXPLAIN plan, ensure index is being used, update table statistics with ANALYZE

**Issue**: Deadlocks occurring
**Solution**: Always acquire locks in the same order, use explicit locking with NOWAIT

**Issue**: Disk space growing rapidly
**Solution**: Run VACUUM, check for bloat, archive old data

## Related Skills

- `backend-dev`: Database integration
- `terraform-iac`: Database infrastructure
- `docker-containerization`: PostgreSQL in containers

## Tools

- pgAdmin: PostgreSQL GUI
- DBeaver: Universal database tool
- pg_stat_statements: Query performance tracking
- EXPLAIN Visualizer: Query plan visualization

## MCP Tools

- `mcp__flow-nexus__sandbox_execute` for SQL scripts
- `mcp__memory-mcp__memory_store` for SQL patterns

## Success Metrics

- Query p95 latency: <100ms
- Index usage: ≥95% of queries
- Connection pool utilization: 60-80%
- Database uptime: 99.99%

---

**Skill Version**: 1.0.0
**Last Updated**: 2025-11-02
