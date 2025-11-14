# PostgreSQL Disaster Recovery Runbook
**CF001 Mitigation: Complete Recovery Procedures**

**Project**: Ruv-Sparc UI Dashboard
**Created**: 2025-11-08
**Agent**: database-design-specialist
**RTO Target**: <4 hours
**RPO Target**: <1 hour

---

## Table of Contents
1. [Emergency Contact Information](#emergency-contact-information)
2. [Backup Strategy Overview](#backup-strategy-overview)
3. [Recovery Scenarios](#recovery-scenarios)
4. [Step-by-Step Recovery Procedures](#step-by-step-recovery-procedures)
5. [Transaction-Level Recovery](#transaction-level-recovery)
6. [Corruption Detection](#corruption-detection)
7. [Failover Procedures](#failover-procedures)
8. [Testing & Validation](#testing--validation)

---

## Emergency Contact Information

**Database Administrator**: [Your Name]
**Email**: [admin@example.com]
**Phone**: [+1-XXX-XXX-XXXX]

**Backup Contact**: [Backup Admin]
**Email**: [backup@example.com]
**Phone**: [+1-XXX-XXX-XXXX]

**PostgreSQL Support**: https://www.postgresql.org/support/
**Internal Wiki**: https://wiki.example.com/db-recovery

---

## Backup Strategy Overview

### Automated Backups
- **Frequency**: Hourly (using `backup-automation.ps1`)
- **Retention**: 7 days (168 hourly backups)
- **Location**: `C:\Users\17175\backups\postgresql\`
- **Format**: Compressed SQL dumps (`ruv_sparc_YYYYMMDD_HHMMSS.sql.gz`)
- **Verification**: Automatic integrity checks after each backup

### Backup Types
1. **Logical Backups** (pg_dump): Full schema + data snapshots
2. **WAL Archives**: Continuous archiving for point-in-time recovery (PITR)
3. **Physical Backups**: Full data directory snapshots (manual)

### Recovery Point Objective (RPO)
- **Target**: <1 hour (hourly backups)
- **Actual**: Maximum 60 minutes of data loss in worst case

### Recovery Time Objective (RTO)
- **Target**: <4 hours
- **Components**:
  - Backup restore: ~30 minutes (for 1GB database)
  - Schema validation: ~15 minutes
  - Application reconnection: ~15 minutes
  - Testing & verification: ~30 minutes
  - Contingency buffer: ~2 hours

---

## Recovery Scenarios

### Scenario 1: Database Corruption (CF001)
**Symptoms**:
- PostgreSQL crashes with "invalid page header" errors
- Data inconsistencies detected by CHECK constraints
- Replication lag exceeds threshold

**Impact**: Complete database unavailability
**Priority**: CRITICAL
**Procedure**: [See Corruption Recovery](#corruption-recovery)

---

### Scenario 2: Accidental Data Deletion
**Symptoms**:
- User reports missing data
- Row counts don't match expected values
- Audit logs show unexpected DELETE operations

**Impact**: Partial data loss
**Priority**: HIGH
**Procedure**: [See Point-in-Time Recovery](#point-in-time-recovery)

---

### Scenario 3: Failed Migration
**Symptoms**:
- Alembic migration fails mid-execution
- Schema changes incomplete
- Application errors after deployment

**Impact**: Application downtime
**Priority**: HIGH
**Procedure**: [See Migration Rollback](#migration-rollback)

---

### Scenario 4: Hardware Failure
**Symptoms**:
- Disk failures
- Server unresponsive
- Network connectivity lost

**Impact**: Complete service outage
**Priority**: CRITICAL
**Procedure**: [See Failover to Standby](#failover-to-standby)

---

## Step-by-Step Recovery Procedures

### Corruption Recovery

**When to Use**: Database corruption detected, server crashes with data integrity errors

#### Step 1: Assess Damage
```powershell
# Check PostgreSQL logs for corruption indicators
Get-Content "C:\Program Files\PostgreSQL\15\data\log\postgresql-*.log" | Select-String "invalid page|corruption|checksum mismatch"

# Check database connectivity
psql -U postgres -d ruv_sparc_db -c "SELECT version();"

# Verify data integrity
psql -U postgres -d ruv_sparc_db -c "SELECT COUNT(*) FROM users;"
```

**Expected Output**: Error messages indicating corruption
**Decision Point**: If database is accessible, attempt VACUUM FULL. If not, proceed to restore.

---

#### Step 2: Stop Application Services
```powershell
# Stop Ruv-Sparc UI Dashboard
Stop-Service -Name "RuvSparcDashboard" -Force

# Verify no active connections
psql -U postgres -c "SELECT COUNT(*) FROM pg_stat_activity WHERE datname='ruv_sparc_db';"

# Kill remaining connections
psql -U postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='ruv_sparc_db' AND pid <> pg_backend_pid();"
```

**Expected Output**: 0 active connections
**Time Estimate**: 2-5 minutes

---

#### Step 3: Identify Latest Valid Backup
```powershell
# List all backups sorted by date
Get-ChildItem "C:\Users\17175\backups\postgresql\ruv_sparc_*.sql.gz" | Sort-Object LastWriteTime -Descending | Select-Object -First 10

# Verify backup integrity
.\scripts\backup-automation.ps1 -VerifyBackup $true -BackupPath "C:\Users\17175\backups\postgresql"
```

**Expected Output**: List of backups with sizes and timestamps
**Selection Criteria**: Choose latest backup BEFORE corruption occurred
**Time Estimate**: 5 minutes

---

#### Step 4: Create Safety Snapshot (If Possible)
```powershell
# If database is still accessible, dump current state
pg_dump -U postgres -d ruv_sparc_db -F custom -f "C:\Users\17175\backups\postgresql\emergency_snapshot_$(Get-Date -Format 'yyyyMMdd_HHmmss').dump"
```

**Purpose**: Preserve potentially recoverable data
**Time Estimate**: 10-30 minutes (depending on size)

---

#### Step 5: Drop Corrupted Database
```powershell
# Drop corrupted database
psql -U postgres -c "DROP DATABASE ruv_sparc_db;"

# Verify dropped
psql -U postgres -c "\l" | Select-String "ruv_sparc"
```

**Expected Output**: Database no longer listed
**Time Estimate**: 1 minute

⚠️ **CRITICAL**: Ensure backup is verified before dropping!

---

#### Step 6: Restore from Backup
```powershell
# Create new database
psql -U postgres -c "CREATE DATABASE ruv_sparc_db WITH ENCODING='UTF8' LC_COLLATE='en_US.UTF-8' LC_CTYPE='en_US.UTF-8';"

# Decompress backup
$backupFile = "C:\Users\17175\backups\postgresql\ruv_sparc_20251108_120000.sql.gz"
$sqlFile = "C:\Users\17175\backups\postgresql\restore_temp.sql"

# Decompress
$source = [System.IO.File]::OpenRead($backupFile)
$destination = [System.IO.File]::Create($sqlFile)
$gzip = New-Object System.IO.Compression.GzipStream($source, [System.IO.Compression.CompressionMode]::Decompress)
$gzip.CopyTo($destination)
$gzip.Close()
$destination.Close()
$source.Close()

# Restore SQL dump
psql -U postgres -d ruv_sparc_db -f $sqlFile 2>&1 | Tee-Object -FilePath "C:\Users\17175\backups\postgresql\restore.log"

# Cleanup temp file
Remove-Item $sqlFile
```

**Expected Output**: SQL commands executing without errors
**Time Estimate**: 15-45 minutes (depending on database size)

---

#### Step 7: Verify Restoration
```powershell
# Check table counts
psql -U postgres -d ruv_sparc_db -c "SELECT 'users' AS table_name, COUNT(*) FROM users
UNION ALL
SELECT 'projects', COUNT(*) FROM projects
UNION ALL
SELECT 'scheduled_tasks', COUNT(*) FROM scheduled_tasks;"

# Verify schema completeness
psql -U postgres -d ruv_sparc_db -c "\dt"

# Test critical queries
psql -U postgres -d ruv_sparc_db -c "SELECT u.username, COUNT(p.project_id) FROM users u LEFT JOIN projects p ON u.user_id = p.user_id GROUP BY u.username;"
```

**Expected Output**: All tables present, data counts match expectations
**Time Estimate**: 10 minutes

---

#### Step 8: Run Alembic Migrations (If Needed)
```powershell
# Check current migration version
alembic current

# Upgrade to latest if needed
alembic upgrade head

# Verify migration success
psql -U postgres -d ruv_sparc_db -c "SELECT version_num FROM alembic_version;"
```

**Expected Output**: Database at latest migration version
**Time Estimate**: 5-15 minutes

---

#### Step 9: Restart Application Services
```powershell
# Restart Ruv-Sparc Dashboard
Start-Service -Name "RuvSparcDashboard"

# Verify service health
Get-Service -Name "RuvSparcDashboard" | Select-Object Status

# Check application logs
Get-Content "C:\path\to\application\logs\app.log" | Select-Object -Last 20
```

**Expected Output**: Service running, no connection errors
**Time Estimate**: 5 minutes

---

#### Step 10: Post-Recovery Validation
```powershell
# Run health checks
Invoke-WebRequest -Uri "http://localhost:3000/health" -UseBasicParsing

# Test user login
# Test task creation
# Test scheduler execution
```

**Expected Output**: All features functional
**Time Estimate**: 30 minutes

---

#### Step 11: Document Incident
```markdown
# Incident Report: Database Corruption Recovery

**Date**: 2025-11-08 14:30:00
**Detected By**: Monitoring system
**Severity**: CRITICAL
**Downtime**: 2 hours 15 minutes

## Timeline
- 14:30 - Corruption detected (pg_checksums failure)
- 14:35 - Services stopped
- 14:40 - Backup identified (20251108_120000)
- 14:45 - Database dropped
- 14:50 - Restore initiated
- 16:20 - Restore completed
- 16:30 - Services restarted
- 16:45 - Full validation complete

## Root Cause
[Describe cause: disk failure, power loss, etc.]

## Data Loss
[Describe lost data: 2 hours 30 minutes of transactions]

## Recovery Actions Taken
[List all steps performed]

## Preventive Measures
[Recommendations to prevent recurrence]
```

**Time Estimate**: 30 minutes

---

### Point-in-Time Recovery (PITR)

**When to Use**: Recover to specific timestamp before data loss event

⚠️ **Prerequisite**: WAL archiving must be enabled

#### Step 1: Enable WAL Archiving (If Not Enabled)
```ini
# postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'copy "%p" "C:\\Users\\17175\\backups\\postgresql\\wal\\%f"'
max_wal_senders = 3
```

**Restart Required**: Yes
**Time Estimate**: 10 minutes

---

#### Step 2: Restore Base Backup
```powershell
# Follow steps 1-6 from Corruption Recovery
# Stop at base backup restoration
```

---

#### Step 3: Configure Recovery Target
```powershell
# Create recovery.conf (PostgreSQL <12) or recovery.signal + postgresql.auto.conf (PostgreSQL >=12)
$recoveryTime = "2025-11-08 14:30:00"

# PostgreSQL 12+
New-Item -Path "C:\Program Files\PostgreSQL\15\data\recovery.signal" -ItemType File

# Add to postgresql.auto.conf
@"
restore_command = 'copy "C:\\Users\\17175\\backups\\postgresql\\wal\\%f" "%p"'
recovery_target_time = '$recoveryTime'
recovery_target_action = 'promote'
"@ | Out-File "C:\Program Files\PostgreSQL\15\data\postgresql.auto.conf" -Append
```

---

#### Step 4: Start PostgreSQL in Recovery Mode
```powershell
# Start PostgreSQL service
Start-Service -Name "postgresql-x64-15"

# Monitor recovery progress
Get-Content "C:\Program Files\PostgreSQL\15\data\log\postgresql-*.log" -Wait | Select-String "recovery|consistent|promote"
```

**Expected Output**: Recovery completes to target time, database promotes to normal operation

---

### Migration Rollback

**When to Use**: Alembic migration fails or causes issues

#### Step 1: Identify Failed Migration
```powershell
# Check current migration
alembic current

# Check migration history
alembic history

# Review migration error
Get-Content "alembic_migration.log" | Select-Object -Last 50
```

---

#### Step 2: Rollback to Previous Revision
```powershell
# Downgrade one step
alembic downgrade -1

# Verify rollback
alembic current

# Check database state
psql -U postgres -d ruv_sparc_db -c "\dt"
```

**Expected Output**: Database at previous migration version
**Time Estimate**: 5 minutes

---

#### Step 3: Fix Migration Script
```python
# Edit migration file in src/database/migrations/versions/
# Fix SQL errors, add missing dependencies, etc.
```

---

#### Step 4: Re-Apply Migration
```powershell
# Retry upgrade
alembic upgrade head

# Verify success
alembic current
```

---

### Failover to Standby

**When to Use**: Primary server hardware failure

⚠️ **Prerequisite**: Streaming replication configured with standby server

#### Step 1: Verify Standby Server Status
```powershell
# On standby server
psql -U postgres -c "SELECT pg_is_in_recovery();"

# Check replication lag
psql -U postgres -c "SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) AS lag_seconds;"
```

**Expected Output**: `pg_is_in_recovery = true`, lag < 10 seconds

---

#### Step 2: Promote Standby to Primary
```powershell
# On standby server
pg_ctl promote -D "C:\Program Files\PostgreSQL\15\data"

# Verify promotion
psql -U postgres -c "SELECT pg_is_in_recovery();"
```

**Expected Output**: `pg_is_in_recovery = false`
**Time Estimate**: 30 seconds

---

#### Step 3: Update Application Connection Strings
```powershell
# Update DATABASE_URL environment variable
[System.Environment]::SetEnvironmentVariable("DATABASE_URL", "postgresql://postgres:password@standby-server:5432/ruv_sparc_db", "Machine")

# Restart application
Restart-Service -Name "RuvSparcDashboard"
```

**Time Estimate**: 5 minutes

---

## Transaction-Level Recovery

### Identify Inconsistent Data
```sql
-- Check for orphaned records
SELECT * FROM scheduled_tasks WHERE project_id NOT IN (SELECT project_id FROM projects);

-- Check constraint violations
SELECT * FROM execution_results WHERE ended_at < started_at;

-- Find duplicate keys (should not exist with UNIQUE constraints)
SELECT user_id, COUNT(*) FROM users GROUP BY user_id HAVING COUNT(*) > 1;
```

### Manual Data Repair
```sql
-- Delete orphaned records
DELETE FROM scheduled_tasks WHERE project_id NOT IN (SELECT project_id FROM projects);

-- Fix timestamp inconsistencies
UPDATE execution_results SET ended_at = started_at + INTERVAL '1 second' WHERE ended_at < started_at;

-- Merge duplicate records
-- (Complex - requires business logic, handle case-by-case)
```

---

## Corruption Detection

### Enable Data Checksums (New Databases)
```powershell
# During database creation
initdb --data-checksums -D "C:\Program Files\PostgreSQL\15\data"
```

### Verify Checksums (Existing Databases)
```powershell
# Offline verification
pg_checksums --check -D "C:\Program Files\PostgreSQL\15\data"

# Online verification (PostgreSQL 12+)
psql -U postgres -c "SELECT pg_checksums_enabled();"
```

### VACUUM FULL for Corruption Recovery
```sql
-- Attempt to repair minor corruption
VACUUM FULL VERBOSE users;
VACUUM FULL VERBOSE projects;
VACUUM FULL VERBOSE scheduled_tasks;

-- Reindex all tables
REINDEX DATABASE ruv_sparc_db;
```

---

## Testing & Validation

### Quarterly Disaster Recovery Drill
```powershell
# 1. Create test database from production backup
# 2. Simulate corruption
# 3. Execute full recovery procedure
# 4. Measure RTO/RPO
# 5. Document lessons learned
```

### Backup Verification Checklist
- [ ] Backup file created with correct timestamp
- [ ] Backup compressed successfully (gzip)
- [ ] Backup file size reasonable (not 0 bytes, not unexpectedly small)
- [ ] Backup decompresses without errors
- [ ] SQL dump contains expected schema elements
- [ ] Test restore to temporary database succeeds
- [ ] Row counts match production (±1%)

### Recovery Testing Schedule
- **Monthly**: Restore latest backup to test server
- **Quarterly**: Full disaster recovery drill with simulated failure
- **Annually**: Review and update runbook procedures

---

## Appendix A: Common Error Messages

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `invalid page header` | Disk corruption | Full restore from backup |
| `could not read block` | File system issue | Check disk health, restore |
| `database is not accepting commands to avoid wraparound data loss` | Transaction ID wraparound | Run `VACUUM FREEZE` |
| `FATAL: the database system is starting up` | Recovery in progress | Wait for recovery to complete |
| `connection refused` | PostgreSQL not running | Start PostgreSQL service |

---

## Appendix B: Performance Tuning for Faster Recovery

### Parallel Restore
```powershell
# Use pg_restore with parallel jobs (requires custom format dump)
pg_restore -U postgres -d ruv_sparc_db -j 4 backup.dump
```

### Disable Constraints During Restore
```sql
-- Temporarily disable constraints for faster restore
ALTER TABLE scheduled_tasks DISABLE TRIGGER ALL;
-- ... restore data ...
ALTER TABLE scheduled_tasks ENABLE TRIGGER ALL;
```

### Increase Shared Buffers for Recovery
```ini
# postgresql.conf (temporary during recovery)
shared_buffers = 4GB
maintenance_work_mem = 1GB
```

---

**Last Updated**: 2025-11-08
**Next Review**: 2025-12-08
**Document Owner**: Database Administrator
