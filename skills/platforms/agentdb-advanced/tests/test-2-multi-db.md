# Test 2: Multi-Database Management

## Objective
Verify multi-database sharding, backup/restore, and management operations for distributed AgentDB deployments.

## Prerequisites
- AgentDB v1.0.7+ installed
- Bash shell (Git Bash on Windows)
- SQLite3 CLI
- At least 1GB free disk space

## Test Setup

### Initialize Test Environment
```bash
# Set test directory
export AGENTDB_BASE_DIR=".agentdb-test"
export BACKUP_DIR="$AGENTDB_BASE_DIR/backups"

# Make script executable
chmod +x resources/scripts/multi_db_manage.sh
```

## Test Cases

### TC2.1: Initialize Multi-Database Shards
**Description**: Create multiple database shards with consistent structure

**Steps**:
1. Initialize 5 database shards
   ```bash
   ./resources/scripts/multi_db_manage.sh init 5 shard
   ```

2. Verify shard creation
   ```bash
   ls -lh $AGENTDB_BASE_DIR/*.db
   ```

3. Check database structure
   ```bash
   sqlite3 $AGENTDB_BASE_DIR/shard-0.db ".schema"
   ```

**Expected Result**:
- 5 database files created: shard-0.db through shard-4.db
- Each database has patterns table
- All databases initialized with 0 patterns

**Pass Criteria**: ✅ 5 shards created with correct schema

---

### TC2.2: Domain-Based Sharding
**Description**: Verify patterns are routed to correct shards based on domain

**Steps**:
1. Define shard routing function
   ```bash
   get_shard() {
     domain=$1
     num_shards=5

     # Hash domain to shard number
     hash=$(echo -n "$domain" | md5sum | cut -c1-8)
     shard_id=$((0x$hash % num_shards))
     echo $shard_id
   }
   ```

2. Insert patterns into correct shards
   ```bash
   for domain in "knowledge-ai" "knowledge-ml" "code-python" "code-java" "docs-api"; do
     shard_id=$(get_shard "$domain")
     npx agentdb@latest insert "$AGENTDB_BASE_DIR/shard-${shard_id}.db" \
       --domain "$domain" \
       --embedding "$(python -c 'import random; print([random.random() for _ in range(384)])')"
   done
   ```

3. Verify pattern distribution
   ```bash
   for i in {0..4}; do
     count=$(sqlite3 "$AGENTDB_BASE_DIR/shard-$i.db" "SELECT COUNT(*) FROM patterns")
     echo "Shard $i: $count patterns"
   done
   ```

**Expected Result**:
- Patterns distributed across shards
- Same domain always routes to same shard
- No shard has more than 2 patterns (roughly even distribution)

**Pass Criteria**: ✅ Patterns correctly sharded, deterministic routing

---

### TC2.3: List Managed Databases
**Description**: Verify database listing shows correct information

**Steps**:
1. Insert test patterns
   ```bash
   for i in {0..4}; do
     for j in {1..10}; do
       npx agentdb@latest insert "$AGENTDB_BASE_DIR/shard-$i.db" \
         --domain "test-$i" \
         --embedding "$(python -c 'import random; print([random.random() for _ in range(384)])')"
     done
   done
   ```

2. List all databases
   ```bash
   ./resources/scripts/multi_db_manage.sh list
   ```

**Expected Result**:
- Shows all 5 databases
- Displays file size for each
- Shows pattern count (10 per shard)
- Total: 5 databases, 50 patterns

**Pass Criteria**: ✅ Correct database count and statistics

---

### TC2.4: Backup All Databases
**Description**: Verify comprehensive backup functionality

**Steps**:
1. Create backup
   ```bash
   backup_path=$(./resources/scripts/multi_db_manage.sh backup)
   echo "Backup created at: $backup_path"
   ```

2. Verify backup contents
   ```bash
   ls -lh "$backup_path"
   ```

3. Check backup includes both .db and .json.gz files
   ```bash
   file_count=$(ls "$backup_path" | wc -l)
   echo "Backup files: $file_count"
   ```

**Expected Result**:
- Backup directory created with timestamp
- Contains 5 .db files (SQLite backups)
- Contains 5 .json.gz files (portable JSON exports)
- Total: 10 files in backup

**Pass Criteria**: ✅ Backup contains 10 files, all databases backed up

---

### TC2.5: Restore from Backup
**Description**: Verify database restoration from backup

**Steps**:
1. Corrupt one database
   ```bash
   rm "$AGENTDB_BASE_DIR/shard-2.db"
   ```

2. Verify database is missing
   ```bash
   ls "$AGENTDB_BASE_DIR/shard-2.db" 2>/dev/null || echo "Database missing"
   ```

3. Restore from backup
   ```bash
   ./resources/scripts/multi_db_manage.sh restore "$backup_path"
   ```

4. Verify restoration
   ```bash
   sqlite3 "$AGENTDB_BASE_DIR/shard-2.db" "SELECT COUNT(*) FROM patterns"
   ```

**Expected Result**:
- Missing database restored
- Pattern count matches original (10 patterns)
- Database fully functional

**Pass Criteria**: ✅ Database restored with correct data

---

### TC2.6: Merge Multiple Databases
**Description**: Verify merging multiple shards into single database

**Steps**:
1. Merge shards 0, 1, 2 into combined database
   ```bash
   ./resources/scripts/multi_db_manage.sh merge \
     "$AGENTDB_BASE_DIR/merged.db" \
     "$AGENTDB_BASE_DIR/shard-0.db" \
     "$AGENTDB_BASE_DIR/shard-1.db" \
     "$AGENTDB_BASE_DIR/shard-2.db"
   ```

2. Verify merged database
   ```bash
   merged_count=$(sqlite3 "$AGENTDB_BASE_DIR/merged.db" "SELECT COUNT(*) FROM patterns")
   echo "Merged database: $merged_count patterns"
   ```

3. Verify no duplicates
   ```bash
   duplicates=$(sqlite3 "$AGENTDB_BASE_DIR/merged.db" \
     "SELECT id, COUNT(*) FROM patterns GROUP BY id HAVING COUNT(*) > 1")

   if [ -z "$duplicates" ]; then
     echo "No duplicates found"
   else
     echo "Duplicates detected!"
   fi
   ```

**Expected Result**:
- Merged database created
- Contains 30 patterns (10 from each shard)
- No duplicate patterns
- All domains preserved

**Pass Criteria**: ✅ Merged database has 30 patterns, no duplicates

---

### TC2.7: Optimize All Databases
**Description**: Verify database optimization (VACUUM, ANALYZE, reindex)

**Steps**:
1. Check database sizes before optimization
   ```bash
   du -sh "$AGENTDB_BASE_DIR"/*.db
   ```

2. Insert and delete patterns to create fragmentation
   ```bash
   for i in {0..4}; do
     # Insert 100 patterns
     for j in {1..100}; do
       npx agentdb@latest insert "$AGENTDB_BASE_DIR/shard-$i.db" \
         --domain "temp" --embedding "[0.1,0.2,...]"
     done

     # Delete all temp patterns
     sqlite3 "$AGENTDB_BASE_DIR/shard-$i.db" "DELETE FROM patterns WHERE domain = 'temp'"
   done
   ```

3. Run optimization
   ```bash
   ./resources/scripts/multi_db_manage.sh optimize
   ```

4. Check database sizes after optimization
   ```bash
   du -sh "$AGENTDB_BASE_DIR"/*.db
   ```

**Expected Result**:
- Database sizes reduced (reclaimed free space)
- VACUUM completes without errors
- ANALYZE updates query planner statistics
- Databases remain functional

**Pass Criteria**: ✅ Optimization completes, database sizes reduced

---

### TC2.8: Database Statistics
**Description**: Verify comprehensive statistics reporting

**Steps**:
1. Generate statistics report
   ```bash
   ./resources/scripts/multi_db_manage.sh stats
   ```

**Expected Result**:
- Shows statistics for each database:
  - File size (human-readable)
  - Pattern count
- Shows total statistics:
  - Total size across all databases
  - Total pattern count
- Format is clear and readable

**Pass Criteria**: ✅ Statistics accurate and complete

---

### TC2.9: Resharding
**Description**: Verify resharding from 5 to 8 shards

**Steps**:
1. Run resharding operation
   ```bash
   ./resources/scripts/multi_db_manage.sh reshard 5 8
   ```

2. Verify new shards created
   ```bash
   ls -lh "$AGENTDB_BASE_DIR"/new-shard-*.db
   ```

3. Verify pattern redistribution
   ```bash
   for i in {0..7}; do
     count=$(sqlite3 "$AGENTDB_BASE_DIR/new-shard-$i.db" "SELECT COUNT(*) FROM patterns")
     echo "New shard $i: $count patterns"
   done
   ```

**Expected Result**:
- 8 new shards created (new-shard-0 through new-shard-7)
- Patterns redistributed based on new shard count
- Total pattern count preserved
- Distribution roughly even

**Pass Criteria**: ✅ 8 new shards, patterns redistributed

---

### TC2.10: Concurrent Database Access
**Description**: Verify multiple processes can access different shards simultaneously

**Steps**:
1. Start 5 concurrent insertions (one per shard)
   ```bash
   for i in {0..4}; do
     (
       for j in {1..100}; do
         npx agentdb@latest insert "$AGENTDB_BASE_DIR/shard-$i.db" \
           --domain "concurrent-$i" \
           --embedding "$(python -c 'import random; print([random.random() for _ in range(384)])')"
       done
     ) &
   done
   wait
   ```

2. Verify all insertions completed
   ```bash
   for i in {0..4}; do
     count=$(sqlite3 "$AGENTDB_BASE_DIR/shard-$i.db" \
       "SELECT COUNT(*) FROM patterns WHERE domain = 'concurrent-$i'")
     echo "Shard $i: $count concurrent insertions"
   done
   ```

**Expected Result**:
- All 5 processes complete successfully
- Each shard has 100 new patterns
- No database locks or errors
- Total: 500 new patterns

**Pass Criteria**: ✅ 500 patterns inserted, no errors

---

## Performance Benchmarks

### Backup Performance
```bash
# Measure backup time
time ./resources/scripts/multi_db_manage.sh backup
```

**Target Metrics**:
- Backup time: < 10 seconds for 5 shards
- Compression ratio: > 5x for JSON exports
- No data loss

### Merge Performance
```bash
# Measure merge time
time ./resources/scripts/multi_db_manage.sh merge \
  merged.db shard-*.db
```

**Target Metrics**:
- Merge time: < 5 seconds for 5 shards
- No memory issues
- Accurate pattern count

---

## Troubleshooting

### Issue: Database Locked
**Solution**:
```bash
# Check for locks
lsof | grep .agentdb

# Kill blocking processes
pkill -f agentdb

# Retry operation
```

### Issue: Backup Failed
**Solution**:
```bash
# Check disk space
df -h

# Verify write permissions
ls -ld $BACKUP_DIR

# Check backup directory exists
mkdir -p $BACKUP_DIR
```

### Issue: Merge Produces Duplicates
**Solution**:
```bash
# Remove duplicates after merge
sqlite3 merged.db "
DELETE FROM patterns
WHERE rowid NOT IN (
  SELECT MIN(rowid)
  FROM patterns
  GROUP BY id
)"
```

---

## Test Report Template

```markdown
## Multi-Database Management Test Report

**Test Date**: 2025-01-01
**Tester**: [Name]
**Environment**: [OS, Shell version]

### Test Results Summary
- Total Test Cases: 10
- Passed: [X]
- Failed: [Y]
- Skipped: [Z]

### Shard Statistics
- Number of Shards: [N]
- Total Patterns: [X]
- Total Size: [Y] MB
- Avg Patterns per Shard: [Z]

### Performance Metrics
- Backup Time: [X]s
- Restore Time: [Y]s
- Merge Time: [Z]s

### Issues Found
1. [Issue description]
2. [Issue description]

### Recommendations
- [Recommendation 1]
- [Recommendation 2]
```

---

## Cleanup

```bash
# Remove test databases
rm -rf $AGENTDB_BASE_DIR

# Remove backups
rm -rf $BACKUP_DIR

# Reset environment variables
unset AGENTDB_BASE_DIR
unset BACKUP_DIR
```
