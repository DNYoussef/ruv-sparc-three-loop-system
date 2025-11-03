# Example 3: Zero-Downtime Migration Patterns

## Scenario

A fintech company needs to migrate their production database from a monolithic `accounts` table to a microservices-ready architecture with separate `users`, `wallets`, and `transactions` tables. Requirements:

- **Zero downtime** (24/7 uptime SLA)
- **Zero data loss** (financial transactions)
- **Backward compatibility** (gradual rollout across 50+ microservices)
- **Rollback capability** (instant revert if issues arise)
- **Audit compliance** (full migration trail)

Current system:
- **accounts** table: 10M rows, 500GB
- **Peak traffic**: 50K writes/sec, 200K reads/sec
- **Deployment window**: None (continuous deployment)

---

## Migration Strategy: Expand-Contract Pattern

The **Expand-Contract** pattern consists of 4 phases:

1. **Expand**: Add new schema alongside old (dual-write)
2. **Migrate**: Backfill historical data
3. **Contract**: Switch reads to new schema
4. **Cleanup**: Remove old schema after validation

---

## Phase 1: Expand - Introduce New Schema

### Step 1.1: Create New Tables

```sql
-- New normalized schema
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    kyc_status TEXT NOT NULL CHECK (kyc_status IN ('pending', 'verified', 'rejected')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE wallets (
    wallet_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    currency TEXT NOT NULL CHECK (currency IN ('USD', 'EUR', 'GBP', 'BTC', 'ETH')),
    balance NUMERIC(20, 8) NOT NULL DEFAULT 0 CHECK (balance >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, currency)
);

CREATE TABLE transactions (
    transaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    wallet_id UUID NOT NULL REFERENCES wallets(wallet_id),
    type TEXT NOT NULL CHECK (type IN ('deposit', 'withdrawal', 'transfer_in', 'transfer_out')),
    amount NUMERIC(20, 8) NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'failed')),
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Indexes for performance
CREATE INDEX idx_wallets_user ON wallets(user_id);
CREATE INDEX idx_transactions_wallet ON transactions(wallet_id, created_at DESC);
CREATE INDEX idx_transactions_status ON transactions(status, created_at);
```

### Step 1.2: Dual-Write Layer (Application Changes)

```javascript
// transactions-service.js
const { Pool } = require('pg');
const pool = new Pool();

// Feature flag for gradual rollout
const NEW_SCHEMA_ENABLED = process.env.FEATURE_NEW_SCHEMA === 'true';

async function createTransaction(userId, amount, currency, type) {
    const client = await pool.connect();

    try {
        await client.query('BEGIN');

        // WRITE TO OLD SCHEMA (backward compatibility)
        const oldResult = await client.query(`
            INSERT INTO accounts (user_id, amount, currency, transaction_type, status)
            VALUES ($1, $2, $3, $4, 'pending')
            RETURNING account_id, created_at;
        `, [userId, amount, currency, type]);

        // DUAL-WRITE TO NEW SCHEMA (if enabled)
        if (NEW_SCHEMA_ENABLED) {
            // Find or create wallet
            let wallet = await client.query(`
                SELECT wallet_id FROM wallets
                WHERE user_id = $1 AND currency = $2;
            `, [userId, currency]);

            if (wallet.rows.length === 0) {
                wallet = await client.query(`
                    INSERT INTO wallets (user_id, currency, balance)
                    VALUES ($1, $2, 0)
                    RETURNING wallet_id;
                `, [userId, currency]);
            }

            const walletId = wallet.rows[0].wallet_id;

            // Insert transaction
            await client.query(`
                INSERT INTO transactions (wallet_id, type, amount, status, metadata)
                VALUES ($1, $2, $3, 'pending', $4);
            `, [walletId, type, amount, JSON.stringify({
                migrated_from_account_id: oldResult.rows[0].account_id
            })]);
        }

        await client.query('COMMIT');
        return oldResult.rows[0];

    } catch (error) {
        await client.query('ROLLBACK');

        // Log discrepancy for monitoring
        console.error('Dual-write failed:', error);
        throw error;
    } finally {
        client.release();
    }
}
```

**Key principles:**
- ✅ Write to BOTH old and new schemas
- ✅ Old schema is source of truth (reads still use it)
- ✅ Feature flag enables gradual rollout
- ✅ Errors in new schema don't fail transactions

---

## Phase 2: Migrate - Backfill Historical Data

### Step 2.1: Create Backfill Job with Batching

```sql
-- Create tracking table for migration progress
CREATE TABLE migration_progress (
    batch_id SERIAL PRIMARY KEY,
    start_id BIGINT NOT NULL,
    end_id BIGINT NOT NULL,
    rows_processed INTEGER,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

-- Insert batches (10K rows per batch for rate limiting)
INSERT INTO migration_progress (start_id, end_id)
SELECT
    (batch - 1) * 10000 + 1 AS start_id,
    batch * 10000 AS end_id
FROM generate_series(1, 1000) AS batch; -- 10M rows / 10K = 1000 batches
```

### Step 2.2: Backfill Script (Node.js)

```javascript
// backfill-migration.js
const { Pool } = require('pg');
const pool = new Pool({ max: 5 }); // Limit concurrency

async function backfillBatch(batchId) {
    const client = await pool.connect();

    try {
        // Mark batch as running
        await client.query(`
            UPDATE migration_progress
            SET status = 'running', started_at = NOW()
            WHERE batch_id = $1;
        `, [batchId]);

        // Get batch range
        const batch = await client.query(`
            SELECT start_id, end_id
            FROM migration_progress
            WHERE batch_id = $1;
        `, [batchId]);

        const { start_id, end_id } = batch.rows[0];

        await client.query('BEGIN');

        // Migrate users
        await client.query(`
            INSERT INTO users (user_id, email, full_name, kyc_status, created_at, updated_at)
            SELECT DISTINCT
                user_id::UUID,
                email,
                full_name,
                kyc_status,
                created_at,
                updated_at
            FROM accounts
            WHERE account_id BETWEEN $1 AND $2
            ON CONFLICT (user_id) DO NOTHING;
        `, [start_id, end_id]);

        // Migrate wallets
        await client.query(`
            INSERT INTO wallets (user_id, currency, balance, created_at, updated_at)
            SELECT
                user_id::UUID,
                currency,
                SUM(
                    CASE
                        WHEN transaction_type IN ('deposit', 'transfer_in') THEN amount
                        WHEN transaction_type IN ('withdrawal', 'transfer_out') THEN -amount
                        ELSE 0
                    END
                ) AS balance,
                MIN(created_at) AS created_at,
                MAX(updated_at) AS updated_at
            FROM accounts
            WHERE account_id BETWEEN $1 AND $2
            GROUP BY user_id, currency
            ON CONFLICT (user_id, currency) DO UPDATE
            SET balance = EXCLUDED.balance, updated_at = EXCLUDED.updated_at;
        `, [start_id, end_id]);

        // Migrate transactions
        await client.query(`
            INSERT INTO transactions (transaction_id, wallet_id, type, amount, status, created_at, completed_at)
            SELECT
                a.transaction_id::UUID,
                w.wallet_id,
                a.transaction_type,
                a.amount,
                a.status,
                a.created_at,
                a.completed_at
            FROM accounts a
            JOIN wallets w ON a.user_id = w.user_id AND a.currency = w.currency
            WHERE a.account_id BETWEEN $1 AND $2
            ON CONFLICT (transaction_id) DO NOTHING;
        `, [start_id, end_id]);

        const result = await client.query(`SELECT COUNT(*) FROM accounts WHERE account_id BETWEEN $1 AND $2`, [start_id, end_id]);
        const rowsProcessed = parseInt(result.rows[0].count);

        await client.query('COMMIT');

        // Mark batch as completed
        await client.query(`
            UPDATE migration_progress
            SET status = 'completed', rows_processed = $2, completed_at = NOW()
            WHERE batch_id = $1;
        `, [batchId, rowsProcessed]);

        console.log(`Batch ${batchId} completed: ${rowsProcessed} rows`);

    } catch (error) {
        await client.query('ROLLBACK');

        // Mark batch as failed
        await client.query(`
            UPDATE migration_progress
            SET status = 'failed', error_message = $2
            WHERE batch_id = $1;
        `, [batchId, error.message]);

        console.error(`Batch ${batchId} failed:`, error);
        throw error;

    } finally {
        client.release();
    }
}

// Process batches with rate limiting
async function runMigration() {
    const CONCURRENCY = 3; // Process 3 batches in parallel
    const RATE_LIMIT_MS = 1000; // 1 second between batch starts

    const batches = await pool.query(`
        SELECT batch_id FROM migration_progress
        WHERE status IN ('pending', 'failed')
        ORDER BY batch_id;
    `);

    for (let i = 0; i < batches.rows.length; i += CONCURRENCY) {
        const chunk = batches.rows.slice(i, i + CONCURRENCY);

        await Promise.all(
            chunk.map(({ batch_id }) => backfillBatch(batch_id))
        );

        // Rate limiting to avoid overwhelming database
        await new Promise(resolve => setTimeout(resolve, RATE_LIMIT_MS));
    }

    console.log('Migration completed!');
}

runMigration().catch(console.error);
```

### Step 2.3: Validation Queries

```sql
-- Verify row counts match
SELECT
    (SELECT COUNT(*) FROM accounts) AS old_count,
    (SELECT COUNT(*) FROM users) AS users_count,
    (SELECT COUNT(*) FROM wallets) AS wallets_count,
    (SELECT COUNT(*) FROM transactions) AS transactions_count;

-- Verify balances match
SELECT
    a.user_id,
    a.currency,
    SUM(
        CASE
            WHEN a.transaction_type IN ('deposit', 'transfer_in') THEN a.amount
            WHEN a.transaction_type IN ('withdrawal', 'transfer_out') THEN -a.amount
            ELSE 0
        END
    ) AS old_balance,
    w.balance AS new_balance,
    ABS(
        SUM(
            CASE
                WHEN a.transaction_type IN ('deposit', 'transfer_in') THEN a.amount
                WHEN a.transaction_type IN ('withdrawal', 'transfer_out') THEN -a.amount
                ELSE 0
            END
        ) - w.balance
    ) AS diff
FROM accounts a
JOIN wallets w ON a.user_id = w.user_id AND a.currency = w.currency
GROUP BY a.user_id, a.currency, w.balance
HAVING ABS(
    SUM(
        CASE
            WHEN a.transaction_type IN ('deposit', 'transfer_in') THEN a.amount
            WHEN a.transaction_type IN ('withdrawal', 'transfer_out') THEN -a.amount
            ELSE 0
        END
    ) - w.balance
) > 0.00000001 -- Tolerance for floating point
ORDER BY diff DESC
LIMIT 100;
```

---

## Phase 3: Contract - Switch Reads to New Schema

### Step 3.1: Dual-Read with Comparison (Shadow Validation)

```javascript
// Enable dual-read for validation
const VALIDATE_READS = process.env.FEATURE_VALIDATE_READS === 'true';

async function getWalletBalance(userId, currency) {
    // Read from OLD schema (source of truth)
    const oldBalance = await pool.query(`
        SELECT SUM(
            CASE
                WHEN transaction_type IN ('deposit', 'transfer_in') THEN amount
                WHEN transaction_type IN ('withdrawal', 'transfer_out') THEN -amount
                ELSE 0
            END
        ) AS balance
        FROM accounts
        WHERE user_id = $1 AND currency = $2 AND status = 'completed';
    `, [userId, currency]);

    // Read from NEW schema (validation)
    if (VALIDATE_READS) {
        const newBalance = await pool.query(`
            SELECT balance FROM wallets
            WHERE user_id = $1 AND currency = $2;
        `, [userId, currency]);

        const oldValue = parseFloat(oldBalance.rows[0]?.balance || 0);
        const newValue = parseFloat(newBalance.rows[0]?.balance || 0);

        // Log discrepancies
        if (Math.abs(oldValue - newValue) > 0.00000001) {
            console.error('Balance mismatch detected:', {
                userId,
                currency,
                oldValue,
                newValue,
                diff: oldValue - newValue
            });

            // Send alert to monitoring system
            sendAlert('MIGRATION_DISCREPANCY', { userId, currency, oldValue, newValue });
        }
    }

    return oldBalance.rows[0]?.balance || 0;
}
```

### Step 3.2: Gradual Cutover with Feature Flags

```javascript
// Percentage-based rollout
const READ_FROM_NEW_SCHEMA_PERCENTAGE = parseInt(process.env.NEW_SCHEMA_ROLLOUT_PCT || '0');

function shouldUseNewSchema(userId) {
    // Deterministic routing based on user ID hash
    const hash = userId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    return (hash % 100) < READ_FROM_NEW_SCHEMA_PERCENTAGE;
}

async function getWalletBalance(userId, currency) {
    if (shouldUseNewSchema(userId)) {
        // Read from NEW schema
        const result = await pool.query(`
            SELECT balance FROM wallets
            WHERE user_id = $1 AND currency = $2;
        `, [userId, currency]);

        return result.rows[0]?.balance || 0;
    } else {
        // Read from OLD schema
        const result = await pool.query(`
            SELECT SUM(
                CASE
                    WHEN transaction_type IN ('deposit', 'transfer_in') THEN amount
                    WHEN transaction_type IN ('withdrawal', 'transfer_out') THEN -amount
                    ELSE 0
                END
            ) AS balance
            FROM accounts
            WHERE user_id = $1 AND currency = $2 AND status = 'completed';
        `, [userId, currency]);

        return result.rows[0]?.balance || 0;
    }
}

// Rollout schedule:
// Week 1: 0% → 5% (canary)
// Week 2: 5% → 25%
// Week 3: 25% → 50%
// Week 4: 50% → 100%
```

---

## Phase 4: Cleanup - Remove Old Schema

### Step 4.1: Final Validation

```sql
-- Run comprehensive validation queries
SELECT COUNT(*) AS discrepancies
FROM (
    SELECT
        a.user_id,
        a.currency,
        SUM(
            CASE
                WHEN a.transaction_type IN ('deposit', 'transfer_in') THEN a.amount
                WHEN a.transaction_type IN ('withdrawal', 'transfer_out') THEN -a.amount
                ELSE 0
            END
        ) AS old_balance,
        w.balance AS new_balance
    FROM accounts a
    JOIN wallets w ON a.user_id = w.user_id AND a.currency = w.currency
    WHERE a.status = 'completed'
    GROUP BY a.user_id, a.currency, w.balance
    HAVING ABS(
        SUM(
            CASE
                WHEN a.transaction_type IN ('deposit', 'transfer_in') THEN a.amount
                WHEN a.transaction_type IN ('withdrawal', 'transfer_out') THEN -a.amount
                ELSE 0
            END
        ) - w.balance
    ) > 0.00000001
) AS mismatches;

-- Expected: 0 discrepancies
```

### Step 4.2: Soft Delete Old Schema

```sql
-- Rename old table instead of dropping (safety)
ALTER TABLE accounts RENAME TO accounts_deprecated;

-- Create view for legacy queries (temporary)
CREATE VIEW accounts AS
SELECT
    t.transaction_id::BIGINT AS account_id,
    w.user_id::TEXT AS user_id,
    w.currency,
    t.type AS transaction_type,
    t.amount,
    t.status,
    t.created_at,
    t.completed_at AS updated_at
FROM transactions t
JOIN wallets w USING (wallet_id);

-- Monitor view usage for 30 days
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

SELECT query, calls, total_exec_time
FROM pg_stat_statements
WHERE query LIKE '%accounts%'
ORDER BY calls DESC;
```

### Step 4.3: Archive and Drop

```sql
-- After 30 days of stable operation:

-- Archive old table to S3/cold storage
COPY accounts_deprecated TO PROGRAM 'gzip > /backups/accounts_deprecated_2025_03_15.csv.gz' CSV HEADER;

-- Verify archive integrity
\! gunzip -c /backups/accounts_deprecated_2025_03_15.csv.gz | wc -l
-- Expected: 10,000,001 lines (10M rows + header)

-- Drop view
DROP VIEW accounts;

-- Drop old table
DROP TABLE accounts_deprecated;

-- Drop migration tracking table
DROP TABLE migration_progress;
```

---

## Rollback Strategy

### Instant Rollback (During Dual-Write Phase)

```javascript
// Emergency rollback: set feature flag to false
// process.env.FEATURE_NEW_SCHEMA = 'false';

// All writes revert to old schema only
// No data loss (old schema still authoritative)
```

### Rollback After Cutover (Read Switch)

```javascript
// Reduce rollout percentage
// process.env.NEW_SCHEMA_ROLLOUT_PCT = '0';

// All reads revert to old schema
// Continue dual-writes to keep new schema in sync
```

### Nuclear Rollback (After Cleanup)

```sql
-- Restore from archive
CREATE TABLE accounts_restored (LIKE accounts_deprecated);

COPY accounts_restored FROM PROGRAM 'gunzip -c /backups/accounts_deprecated_2025_03_15.csv.gz' CSV HEADER;

-- Rename to original
ALTER TABLE accounts_restored RENAME TO accounts;

-- Rebuild indexes
CREATE INDEX idx_accounts_user ON accounts(user_id, currency);
```

---

## Monitoring and Alerts

### Key Metrics to Track

```javascript
// Prometheus metrics
const promClient = require('prom-client');

// Dual-write success rate
const dualWriteSuccessRate = new promClient.Gauge({
    name: 'migration_dual_write_success_rate',
    help: 'Percentage of successful dual-writes'
});

// Data discrepancy count
const discrepancyCount = new promClient.Counter({
    name: 'migration_discrepancy_count',
    help: 'Number of data mismatches between old and new schema'
});

// Migration progress
const migrationProgress = new promClient.Gauge({
    name: 'migration_progress_percentage',
    help: 'Percentage of data migrated'
});

// Set alerts:
// - dualWriteSuccessRate < 99.9% → Page on-call engineer
// - discrepancyCount > 100/hour → Halt migration
// - migrationProgress stalled for 1 hour → Investigate backfill job
```

---

## Outcomes & Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **Downtime** | 0 seconds | 0 seconds | ✅ |
| **Data Loss** | 0 records | 0 records | ✅ |
| **Migration Duration** | 30 days | 28 days | ✅ |
| **Discrepancy Rate** | <0.001% | 0.0003% | ✅ |
| **Rollback Count** | 0 | 2 (both successful) | ✅ |
| **Performance Impact** | <5% latency | +2.3% latency (dual-write) | ✅ |

**Production statistics:**
- **10M rows migrated** in 28 days
- **2 rollbacks** during canary phase (caught by monitoring)
- **Zero customer-facing incidents**
- **3 discrepancies detected** (all resolved within 24 hours)

---

## Key Takeaways

### 1. **Expand-Contract is the Safest Pattern**
- Expand: Add new schema alongside old
- Migrate: Backfill data incrementally
- Contract: Switch reads gradually
- Cleanup: Remove old schema after validation

### 2. **Dual-Write from Day 1**
- Write to BOTH old and new schemas
- Old schema remains authoritative during migration
- Feature flags enable instant rollback

### 3. **Batch Backfills with Rate Limiting**
- Process 10K-100K rows per batch
- Track progress in dedicated table
- Rate limit to avoid overwhelming database
- Retry failed batches automatically

### 4. **Shadow Validation Before Cutover**
- Read from both schemas and compare
- Log discrepancies for investigation
- Fix data issues before switching traffic

### 5. **Gradual Rollout with Canary Analysis**
- Start with 1-5% traffic (canary)
- Monitor for 48-72 hours
- Increase gradually: 5% → 25% → 50% → 100%
- Deterministic routing (hash-based) for reproducibility

### 6. **Always Have a Rollback Plan**
- Feature flags for instant revert
- Archive old schema before dropping
- Test rollback procedures in staging
- Document rollback steps in runbook

### 7. **Comprehensive Monitoring is Critical**
- Track dual-write success rate
- Alert on data discrepancies
- Monitor migration progress
- Page on-call for anomalies

### 8. **Validate, Validate, Validate**
- SQL validation queries (row counts, sums)
- Application-level validation (shadow reads)
- Manual spot checks (random sampling)
- Automated regression tests

---

## Common Pitfalls to Avoid

❌ **Dropping old schema too early** → Archive for 30+ days first
❌ **No rollback plan** → Always have instant revert capability
❌ **Big-bang cutover** → Use gradual percentage-based rollout
❌ **Ignoring discrepancies** → Investigate and fix immediately
❌ **Backfill without rate limiting** → Causes production outages
❌ **No monitoring** → Flying blind during critical migration
❌ **Skipping validation** → Data corruption discovered too late

---

## Tools & Technologies Used

- **PostgreSQL 15** - Database
- **Node.js + pg** - Application layer
- **Feature flags** (LaunchDarkly or custom)
- **Prometheus + Grafana** - Monitoring and alerting
- **PgBouncer** - Connection pooling
- **pg_stat_statements** - Query performance tracking
- **S3 / Glacier** - Archival storage

---

## Next Steps

1. **Monitor for 30 days** before dropping old schema
2. **Document lessons learned** in runbook
3. **Automate future migrations** with reusable scripts
4. **Train team on migration patterns**
5. **Set up automated testing** for future schema changes
