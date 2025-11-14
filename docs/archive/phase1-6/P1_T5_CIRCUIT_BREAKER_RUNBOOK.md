# Circuit Breaker Runbook - Manual Operations

**Project**: Ruv-Sparc UI Dashboard
**Component**: Memory MCP Circuit Breaker
**Task**: P1_T5 - CF003 Mitigation
**Date**: 2025-11-08

---

## Table of Contents

1. [Emergency Procedures](#emergency-procedures)
2. [Manual Circuit Control](#manual-circuit-control)
3. [Health Check Operations](#health-check-operations)
4. [Queue Management](#queue-management)
5. [Troubleshooting](#troubleshooting)
6. [Monitoring](#monitoring)

---

## Emergency Procedures

### Scenario 1: Memory MCP Service Down

**Symptoms**:
- Circuit breaker state: OPEN
- UI shows degraded mode banner
- Alerts fired for consecutive failures

**Immediate Actions**:
1. **Verify circuit opened** (prevents cascade failures)
   ```bash
   curl http://localhost:8000/api/memory-mcp/circuit-state
   # Expected: {"state": "OPEN"}
   ```

2. **Confirm fallback mode active**
   ```bash
   # Check Redis cache
   redis-cli KEYS "project_metadata:*"

   # Check PostgreSQL fallback
   psql -d sparc_ui -c "SELECT COUNT(*) FROM projects;"
   ```

3. **Check Memory MCP service logs**
   ```bash
   # Docker logs
   docker logs memory-mcp --tail 100

   # Or systemd logs
   journalctl -u memory-mcp -n 100
   ```

4. **Attempt service restart**
   ```bash
   # Docker
   docker restart memory-mcp

   # Or systemd
   sudo systemctl restart memory-mcp
   ```

5. **Monitor recovery**
   ```bash
   # Watch circuit state
   watch -n 5 'curl -s http://localhost:8000/api/memory-mcp/circuit-state | jq .state'

   # Expected transitions:
   # OPEN → HALF_OPEN (after 60s)
   # HALF_OPEN → CLOSED (after 3 successful health checks)
   ```

**Timeline**:
- Circuit opens: Immediate (after 5 failures)
- Fallback activates: Immediate
- Health checks resume: Every 30 seconds
- Recovery complete: ~90 seconds (if service restored)

---

### Scenario 2: Circuit Stuck OPEN

**Symptoms**:
- Memory MCP service is healthy
- Circuit remains OPEN for >5 minutes
- Health checks failing

**Actions**:
1. **Verify Memory MCP is actually healthy**
   ```bash
   curl http://localhost:8000/health
   # Expected: 200 OK
   ```

2. **Check circuit breaker statistics**
   ```bash
   curl http://localhost:8000/api/memory-mcp/stats | jq
   ```

3. **Manual health check**
   ```python
   from utils.memory_mcp_circuit_breaker import get_memory_mcp_client

   client = get_memory_mcp_client(...)
   is_healthy = await client.health_check()
   print(f"Health check: {is_healthy}")
   ```

4. **Force circuit close** (last resort)
   ```bash
   curl -X POST http://localhost:8000/api/memory-mcp/force-close
   ```

   **⚠️ WARNING**: Only force-close if you're certain Memory MCP is healthy!

5. **Process queued items**
   ```python
   await client.process_queued_items()
   ```

---

## Manual Circuit Control

### Check Circuit State

```bash
# Via API
curl http://localhost:8000/api/memory-mcp/circuit-state

# Via Python
python -c "
from utils.memory_mcp_circuit_breaker import get_memory_mcp_client
client = get_memory_mcp_client(...)
print(f'State: {client.get_circuit_state()}')
print(f'Stats: {client.get_stats()}')
"
```

### Force Circuit OPEN (for testing)

```python
# Use this for testing fallback mode
from utils.memory_mcp_circuit_breaker import get_memory_mcp_client

client = get_memory_mcp_client(...)
client.circuit_breaker._state = "open"
print("Circuit manually opened for testing")
```

### Force Circuit CLOSED (emergency recovery)

```python
# ⚠️ USE WITH CAUTION - Only if service is confirmed healthy
from utils.memory_mcp_circuit_breaker import get_memory_mcp_client

client = get_memory_mcp_client(...)
client.circuit_breaker._state = "closed"
client.stats.consecutive_failures = 0
client.stats.consecutive_successes = 3
print("Circuit manually closed")
```

### Reset Circuit Breaker

```python
from utils.memory_mcp_circuit_breaker import get_memory_mcp_client

client = get_memory_mcp_client(...)

# Reset all statistics
client.stats.consecutive_failures = 0
client.stats.consecutive_successes = 0
client.stats.circuit_opens = 0
client.stats.circuit_closes = 0

# Close circuit
client.circuit_breaker._state = "closed"

print("Circuit breaker reset")
```

---

## Health Check Operations

### View Health Check Status

```bash
# Check monitoring status
curl http://localhost:8000/api/memory-mcp/health-monitor/stats | jq

# Expected output:
{
  "is_running": true,
  "total_checks": 150,
  "failed_checks": 5,
  "success_rate": 96.7,
  "consecutive_failures": 0,
  "consecutive_successes": 10
}
```

### Restart Health Monitoring

```python
from utils.health_check_monitor import initialize_health_monitor

# Stop existing monitor
await monitor.stop()

# Start new monitor
monitor = await initialize_health_monitor(
    memory_mcp_client=client,
    enable_email=True,
    enable_websocket=True
)

print("Health monitoring restarted")
```

### View Recent Alerts

```python
from utils.health_check_monitor import HealthCheckMonitor

monitor = HealthCheckMonitor(...)
stats = monitor.get_stats()

print("Recent alerts:")
for alert in stats['recent_alerts']:
    print(f"- [{alert['severity']}] {alert['message']}")
```

### Test Alert Callbacks

```python
from utils.health_check_monitor import HealthCheckAlert, send_email_alert

# Create test alert
test_alert = HealthCheckAlert(
    timestamp=datetime.now(),
    message="Test alert",
    severity="warning",
    consecutive_failures=3,
    service_url="http://localhost:8000"
)

# Test email alert
await send_email_alert(test_alert)
print("Email alert sent")
```

---

## Queue Management

### View Queued Items

```bash
# Check Redis queue length
redis-cli LLEN memory_mcp_queue

# View queued items (first 10)
redis-cli LRANGE memory_mcp_queue 0 9
```

### Process Queue Manually

```python
from utils.memory_mcp_circuit_breaker import get_memory_mcp_client

client = get_memory_mcp_client(...)

# Ensure circuit is CLOSED first
if client.get_circuit_state().value == "CLOSED":
    await client.process_queued_items()
    print("Queue processed")
else:
    print(f"Cannot process queue - circuit is {client.get_circuit_state().value}")
```

### Clear Queue (emergency)

```bash
# ⚠️ WARNING: This deletes all queued items!
redis-cli DEL memory_mcp_queue
echo "Queue cleared"
```

### View Sync Queue

```bash
# Check items waiting for Memory MCP sync
redis-cli LLEN memory_mcp_sync_queue

# View sync queue
redis-cli LRANGE memory_mcp_sync_queue 0 9
```

---

## Troubleshooting

### Problem: High Fallback Request Rate

**Diagnosis**:
```bash
# Check fallback request percentage
curl http://localhost:8000/api/memory-mcp/stats | jq '.fallback_requests / .total_requests * 100'

# If >10%, investigate why requests are failing
```

**Possible Causes**:
1. Memory MCP service degraded (slow responses)
2. Network issues between services
3. Circuit breaker threshold too sensitive

**Actions**:
1. Check Memory MCP performance
   ```bash
   curl http://localhost:8000/metrics | grep response_time
   ```

2. Increase timeouts (if justified)
   ```python
   config = CircuitBreakerConfig(
       vector_search_timeout=5.0,  # Increase from 2.0s
       memory_store_timeout=3.0    # Increase from 1.0s
   )
   ```

3. Adjust failure threshold
   ```python
   config = CircuitBreakerConfig(
       failure_threshold=10  # Increase from 5
   )
   ```

---

### Problem: Circuit Breaker Not Opening

**Diagnosis**:
```bash
# Check consecutive failures
curl http://localhost:8000/api/memory-mcp/stats | jq '.consecutive_failures'

# If high but circuit still CLOSED, circuit breaker may be misconfigured
```

**Actions**:
1. Verify circuit breaker configuration
   ```python
   print(f"Failure threshold: {client.config.failure_threshold}")
   print(f"Current state: {client.circuit_breaker.current_state}")
   ```

2. Check for exceptions being caught
   ```python
   # Ensure exceptions are being raised
   try:
       await client.vector_search_with_fallback("test")
   except Exception as e:
       print(f"Exception: {e}")
   ```

---

### Problem: Redis Cache Not Working

**Diagnosis**:
```bash
# Check Redis connection
redis-cli PING
# Expected: PONG

# Check cache entries
redis-cli KEYS "project_metadata:*"

# Check cache TTL
redis-cli TTL "project_metadata:proj-123"
```

**Actions**:
1. Verify Redis connection
   ```python
   from utils.fallback_mode_implementation import FallbackMode

   fallback = FallbackMode(...)
   await fallback.redis.ping()
   print("Redis connected")
   ```

2. Manually cache data
   ```python
   await fallback.cache_project_metadata(
       project_id="proj-123",
       metadata={"name": "Test Project"}
   )
   ```

3. Check cache retrieval
   ```python
   metadata = await fallback.get_cached_project_metadata("proj-123")
   print(f"Cached: {metadata}")
   ```

---

## Monitoring

### Dashboard Metrics

**Key Metrics to Monitor**:
1. **Circuit State**: CLOSED/OPEN/HALF_OPEN
2. **Success Rate**: Target >95%
3. **Fallback Request Rate**: Target <5%
4. **Consecutive Failures**: Alert if >3
5. **Queue Length**: Alert if >100

### Grafana Queries

```promql
# Circuit breaker state (1=CLOSED, 2=HALF_OPEN, 3=OPEN)
circuit_breaker_state{service="memory_mcp"}

# Success rate
rate(circuit_breaker_successful_requests[5m]) / rate(circuit_breaker_total_requests[5m]) * 100

# Fallback request rate
rate(circuit_breaker_fallback_requests[5m]) / rate(circuit_breaker_total_requests[5m]) * 100

# Queue length
circuit_breaker_queue_length{queue="memory_mcp_queue"}
```

### Alert Rules

```yaml
# Circuit breaker opened
- alert: CircuitBreakerOpen
  expr: circuit_breaker_state{service="memory_mcp"} == 3
  for: 1m
  annotations:
    summary: "Memory MCP circuit breaker opened"
    description: "Circuit breaker has been OPEN for >1 minute"

# High failure rate
- alert: HighFailureRate
  expr: rate(circuit_breaker_failed_requests[5m]) / rate(circuit_breaker_total_requests[5m]) > 0.1
  for: 2m
  annotations:
    summary: "High failure rate (>10%)"

# Queue growing
- alert: QueueGrowing
  expr: circuit_breaker_queue_length > 100
  for: 5m
  annotations:
    summary: "Circuit breaker queue growing (>100 items)"
```

---

## Maintenance Tasks

### Daily:
- [ ] Check circuit breaker statistics
- [ ] Review alert history
- [ ] Verify queue is empty (or <10 items)

### Weekly:
- [ ] Review success rate trends
- [ ] Check for recurring failure patterns
- [ ] Update failure thresholds if needed

### Monthly:
- [ ] Review circuit breaker configuration
- [ ] Analyze downtime incidents
- [ ] Update runbook with lessons learned

---

## Contact Information

**On-Call Engineer**: [Your Contact]
**Escalation**: [Team Lead Contact]
**Documentation**: C:\Users\17175\docs\P1_T5_CF003_MITIGATION_IMPLEMENTATION.md

---

**Last Updated**: 2025-11-08
**Agent**: backend-dev
**Task**: P1_T5 - CF003 Mitigation
