# Test 1: QUIC Synchronization

## Objective
Verify QUIC synchronization works correctly between multiple AgentDB nodes with sub-millisecond latency.

## Prerequisites
- AgentDB v1.0.7+ installed
- Python 3.8+ with aioquic library
- Network connectivity between test nodes
- SSL certificates generated

## Test Setup

### Generate Test Certificates
```bash
# Generate self-signed certificates for testing
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

### Configure QUIC Sync
```yaml
# quic-test-config.yaml
server:
  host: "0.0.0.0"
  port: 4433
  tls:
    cert_file: "./cert.pem"
    key_file: "./key.pem"

peers:
  - host: "127.0.0.1"
    port: 4434
    name: "node-2"

sync:
  interval: 500  # 500ms for testing
  batch_size: 10
  compression: true
```

## Test Cases

### TC1.1: Basic QUIC Connection
**Description**: Verify QUIC connection establishment between two nodes

**Steps**:
1. Start Node 1 on port 4433
   ```bash
   python resources/scripts/quic_sync.py --config quic-test-config.yaml --port 4433
   ```

2. Start Node 2 on port 4434
   ```bash
   python resources/scripts/quic_sync.py --config quic-test-config.yaml --port 4434
   ```

3. Verify connection in logs
   ```bash
   tail -f .agentdb/logs/quic-sync.log
   ```

**Expected Result**:
- Nodes establish QUIC connection within 100ms
- TLS 1.3 handshake completes successfully
- Logs show "Connected to peer" messages

**Pass Criteria**: ✅ Connection established, no errors

---

### TC1.2: Pattern Synchronization
**Description**: Verify patterns sync across nodes

**Steps**:
1. Insert pattern on Node 1
   ```bash
   npx agentdb@latest insert .agentdb/node1.db \
     --domain "test-sync" \
     --embedding "[0.1,0.2,0.3,...]" \
     --metadata '{"test_id": "TC1.2", "timestamp": "2025-01-01T00:00:00Z"}'
   ```

2. Wait 1 second for synchronization

3. Query pattern on Node 2
   ```bash
   npx agentdb@latest query .agentdb/node2.db \
     "[0.1,0.2,0.3,...]" \
     --domain "test-sync" \
     --k 1
   ```

**Expected Result**:
- Pattern appears on Node 2 within 1 second
- Pattern data matches exactly (ID, embedding, metadata)
- Sync latency < 100ms (check logs)

**Pass Criteria**: ✅ Pattern synchronized, latency < 100ms

---

### TC1.3: Batch Synchronization
**Description**: Verify batch synchronization performance

**Steps**:
1. Insert 100 patterns on Node 1
   ```bash
   for i in {1..100}; do
     npx agentdb@latest insert .agentdb/node1.db \
       --domain "batch-test" \
       --embedding "$(python -c 'import random; print([random.random() for _ in range(384)])')" \
       --metadata "{\"batch_id\": $i}"
   done
   ```

2. Wait for synchronization (monitor logs)

3. Verify pattern count on Node 2
   ```bash
   sqlite3 .agentdb/node2.db "SELECT COUNT(*) FROM patterns WHERE domain = 'batch-test'"
   ```

**Expected Result**:
- All 100 patterns synchronized
- Total sync time < 5 seconds
- No packet loss or retries

**Pass Criteria**: ✅ 100/100 patterns synced, time < 5s

---

### TC1.4: Network Interruption Recovery
**Description**: Verify QUIC sync recovers from network interruptions

**Steps**:
1. Start Node 1 and Node 2 with sync active

2. Simulate network interruption
   ```bash
   # Block traffic between nodes
   sudo iptables -A INPUT -s 127.0.0.1 -p udp --dport 4434 -j DROP
   ```

3. Insert pattern on Node 1 during interruption

4. Restore network after 5 seconds
   ```bash
   sudo iptables -D INPUT -s 127.0.0.1 -p udp --dport 4434 -j DROP
   ```

5. Verify pattern eventually syncs to Node 2

**Expected Result**:
- Sync pauses during network interruption
- Automatic retry with exponential backoff
- Pattern syncs within 3 retries after recovery
- No data loss

**Pass Criteria**: ✅ Pattern synced after recovery, no data loss

---

### TC1.5: Three-Node Mesh Synchronization
**Description**: Verify mesh topology with 3 nodes

**Steps**:
1. Configure 3-node mesh
   ```yaml
   # Node 1: peers = [node-2, node-3]
   # Node 2: peers = [node-1, node-3]
   # Node 3: peers = [node-1, node-2]
   ```

2. Start all 3 nodes

3. Insert pattern on Node 1

4. Verify pattern appears on both Node 2 and Node 3

**Expected Result**:
- Pattern syncs to all nodes via mesh topology
- Each node receives pattern within 500ms
- No duplicate synchronization

**Pass Criteria**: ✅ Pattern on all 3 nodes, no duplicates

---

### TC1.6: Conflict Resolution
**Description**: Verify conflict resolution when same pattern updated on multiple nodes

**Steps**:
1. Insert pattern with ID "conflict-test" on Node 1
   ```javascript
   { id: "conflict-test", confidence: 0.7, data: "version-1" }
   ```

2. Update same pattern ID on Node 2 simultaneously
   ```javascript
   { id: "conflict-test", confidence: 0.9, data: "version-2" }
   ```

3. Wait for synchronization

4. Verify conflict resolution on both nodes

**Expected Result**:
- Conflict detected via vector clock
- Higher confidence version wins (0.9 > 0.7)
- Both nodes converge to same state

**Pass Criteria**: ✅ Conflict resolved, data: "version-2"

---

## Performance Benchmarks

### Latency Measurement
```bash
# Measure sync latency
python -c "
import time
import json

# Insert pattern on Node 1
start = time.time()
# ... insert code ...

# Poll Node 2 until pattern appears
while True:
    # ... query code ...
    if pattern_found:
        latency_ms = (time.time() - start) * 1000
        print(f'Sync latency: {latency_ms:.2f}ms')
        break
    time.sleep(0.01)
"
```

**Target Metrics**:
- P50 latency: < 10ms
- P95 latency: < 50ms
- P99 latency: < 100ms

### Throughput Measurement
```bash
# Measure patterns per second
patterns_synced=1000
time_seconds=5
throughput=$((patterns_synced / time_seconds))
echo "Throughput: $throughput patterns/sec"
```

**Target Metrics**:
- Throughput: > 100 patterns/sec
- CPU usage: < 50%
- Memory usage: < 200MB per node

---

## Troubleshooting

### Issue: Connection Refused
**Solution**:
```bash
# Check if port is available
netstat -an | grep 4433

# Verify firewall allows UDP
sudo ufw allow 4433/udp

# Check SSL certificates
openssl verify cert.pem
```

### Issue: High Latency
**Solution**:
```bash
# Enable QUIC debug logging
export DEBUG=agentdb:quic

# Check network latency
ping -c 10 <peer-host>

# Verify congestion control
# Switch to BBR if high latency
```

### Issue: Patterns Not Syncing
**Solution**:
```bash
# Check sync queue
tail -f .agentdb/logs/quic-sync.log | grep "queue_pattern"

# Verify peer reachability
telnet <peer-host> 4433

# Check filter configuration
# Ensure patterns match sync filters
```

---

## Test Report Template

```markdown
## QUIC Sync Test Report

**Test Date**: 2025-01-01
**Tester**: [Name]
**AgentDB Version**: 1.0.7

### Test Results Summary
- Total Test Cases: 6
- Passed: [X]
- Failed: [Y]
- Skipped: [Z]

### Performance Metrics
- P50 Latency: [X]ms
- P95 Latency: [Y]ms
- Throughput: [Z] patterns/sec

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
# Stop all nodes
pkill -f quic_sync.py

# Remove test databases
rm -rf .agentdb/node*.db

# Remove test certificates
rm cert.pem key.pem

# Clear logs
rm -rf .agentdb/logs/quic-sync.log
```
