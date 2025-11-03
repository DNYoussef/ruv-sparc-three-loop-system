---
name: agent-health-check
category: monitoring
version: 1.0.0
---

# /agent-health-check

Monitor agent health status, performance, and resource utilization in real-time.

## Usage
```bash
/agent-health-check [agent_id] [options]
```

## Parameters
- `agent_id` - Specific agent to check (default: all active agents)
- `--interval` - Check interval in seconds (default: 5)
- `--duration` - Monitoring duration in minutes (default: continuous)
- `--metrics` - Metrics to collect: cpu|memory|tasks|all (default: all)
- `--threshold-cpu` - CPU usage alert threshold % (default: 80)
- `--threshold-memory` - Memory usage alert threshold % (default: 85)
- `--alert` - Enable alerting (default: true)
- `--export` - Export metrics: none|prometheus|json (default: none)

## What It Does

**Comprehensive Agent Monitoring**:
1. 💓 **Health Status**: Alive, responding, stuck
2. 🧠 **Resource Usage**: CPU, memory, network
3. 📊 **Task Metrics**: Active, queued, completed, failed
4. ⚡ **Performance**: Response time, throughput
5. 🔍 **Error Detection**: Crashes, hangs, timeouts
6. 🚨 **Alerting**: Threshold violations
7. 📈 **Trend Analysis**: Performance over time
8. 🔄 **Auto-Recovery**: Restart unhealthy agents

**Monitored Agents**:
- Swarm coordination agents
- SPARC methodology agents
- Integration agents (GitHub, multi-model)
- Custom workflow agents
- Background task agents

## Examples

```bash
# Check all active agents
/agent-health-check

# Monitor specific agent
/agent-health-check coder-agent-123

# Continuous monitoring with 10s interval
/agent-health-check --interval 10 --duration 60

# CPU and memory only
/agent-health-check --metrics cpu,memory

# Custom thresholds
/agent-health-check --threshold-cpu 70 --threshold-memory 80

# Export to Prometheus
/agent-health-check --export prometheus

# Quiet mode (alerts only)
/agent-health-check --alert true --quiet true
```

## Output

```
💓 Agent Health Check Started

Monitoring: All active agents (12 total)
Interval: 5 seconds
Thresholds: CPU 80%, Memory 85%
Alert: Enabled

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Active Agents Overview
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ID                Type              Status    Uptime    CPU    Memory   Tasks
  ──────────────────────────────────────────────────────────────────────────────
  coder-001         coder             healthy   2h 34m    45%    1.2GB    3/10
  reviewer-002      reviewer          healthy   2h 34m    32%    890MB    2/5
  tester-003        tester            healthy   1h 12m    67%    1.8GB    5/8
  researcher-004    researcher        healthy   45m       23%    650MB    1/3
  planner-005       planner           healthy   2h 10m    18%    520MB    0/2
  coordinator-006   swarm-coord       healthy   2h 34m    12%    340MB    8/15
  github-007        github-swarm      healthy   30m       41%    720MB    2/4
  gemini-008        gemini-search     warning   15m       88%    2.1GB    1/1  ⚠️
  codex-009         codex-auto        healthy   8m        54%    1.1GB    1/2
  neural-010        neural-train      critical  5m        95%    7.2GB    0/1  ❌
  analyzer-011      perf-analyzer     healthy   1h 5m     28%    680MB    1/3
  monitor-012       swarm-monitor     healthy   2h 34m    15%    420MB    4/10

  Summary:
    ✅ Healthy: 10 agents (83.3%)
    ⚠️  Warning: 1 agent (8.3%)
    ❌ Critical: 1 agent (8.3%)
    💀 Dead: 0 agents (0%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Health Check Details
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[00:00] Checking all agents...

  ✅ coder-001 (coder)
     Status: Healthy
     CPU: 45% (normal)
     Memory: 1.2GB / 4GB (30%)
     Tasks: 3 active, 7 queued, 245 completed, 2 failed
     Avg Response: 234ms
     Last Heartbeat: 2s ago
     Errors (24h): 2 (0.8%)

  ✅ reviewer-002 (reviewer)
     Status: Healthy
     CPU: 32% (normal)
     Memory: 890MB / 4GB (22%)
     Tasks: 2 active, 3 queued, 187 completed, 0 failed
     Avg Response: 189ms
     Last Heartbeat: 1s ago
     Errors (24h): 0 (0%)

  ✅ tester-003 (tester)
     Status: Healthy
     CPU: 67% (moderate)
     Memory: 1.8GB / 4GB (45%)
     Tasks: 5 active, 3 queued, 567 completed, 12 failed
     Avg Response: 456ms
     Last Heartbeat: 3s ago
     Errors (24h): 12 (2.1%)
     Note: Running comprehensive test suite

  ⚠️  gemini-008 (gemini-search)
     Status: WARNING - High CPU
     CPU: 88% ⚠️ (threshold: 80%)
     Memory: 2.1GB / 4GB (52%)
     Tasks: 1 active, 0 queued, 34 completed, 1 failed
     Avg Response: 1,234ms (slow)
     Last Heartbeat: 2s ago
     Errors (24h): 1 (2.9%)
     Alert: CPU usage above threshold
     Recommendation: Scale up or optimize search queries

  ❌ neural-010 (neural-train)
     Status: CRITICAL - Near OOM
     CPU: 95% ❌ (threshold: 80%)
     Memory: 7.2GB / 8GB (90%) ❌ (threshold: 85%)
     Tasks: 0 active, 1 queued, 5 completed, 3 failed
     Avg Response: 3,456ms (very slow)
     Last Heartbeat: 12s ago (delayed)
     Errors (24h): 3 (37.5%)
     Alert: Memory critical, possible OOM kill imminent
     Recommendation: Restart agent or increase memory limit

[00:05] Checking all agents...

  ⚠️  gemini-008 (gemini-search)
     CPU: 91% ⚠️ (increased from 88%)
     Memory: 2.3GB / 4GB (57%)
     Alert: CPU usage trending up

  ❌ neural-010 (neural-train)
     CPU: 98% ❌ (increased from 95%)
     Memory: 7.8GB / 8GB (97%) ❌ CRITICAL
     Last Heartbeat: 17s ago (timeout soon)
     Alert: Agent unresponsive, memory at 97%
     Action: Auto-restart triggered

  🔄 Restarting neural-010...
     ✅ Agent stopped gracefully
     ✅ Memory released: 7.8GB → 0GB
     ✅ Agent restarted with PID 45678
     ✅ Health check: Passed
     ✅ New metrics: CPU 12%, Memory 450MB

[00:10] Checking all agents...

  ✅ gemini-008 (gemini-search)
     CPU: 76% (improved from 91%)
     Memory: 2.1GB / 4GB (52%)
     Status: Healthy (recovered)
     Note: Search query completed

  ✅ neural-010 (neural-train)
     CPU: 15% (recovered from 98%)
     Memory: 520MB / 8GB (6%)
     Status: Healthy (post-restart)
     Tasks: 1 active (resumed training)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Resource Usage Trends (Last 15 minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CPU Usage:
    Peak: 98% (neural-010 at 00:05)
    Average: 42%
    Current: 38%
    Trend: ↓ Decreasing

  Memory Usage:
    Peak: 7.8GB (neural-010 at 00:05)
    Average: 1.8GB per agent
    Current: 1.2GB per agent
    Trend: → Stable

  Task Throughput:
    Total Completed: 1,234 tasks
    Success Rate: 97.3%
    Avg Task Duration: 3.2 minutes
    Trend: ↑ Increasing

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Alerts & Actions (Last 15 minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [00:00] ⚠️  WARNING: gemini-008 CPU at 88% (threshold: 80%)
  [00:00] ❌ CRITICAL: neural-010 Memory at 90% (threshold: 85%)
  [00:05] ❌ CRITICAL: neural-010 CPU at 98%, Memory at 97%
  [00:05] 🔄 ACTION: Auto-restart neural-010 (unresponsive)
  [00:05] ✅ SUCCESS: neural-010 restarted successfully
  [00:10] ✅ RESOLVED: gemini-008 CPU normalized at 76%
  [00:10] ✅ RESOLVED: neural-010 healthy post-restart

  Total Alerts: 4
  Critical: 2 (resolved)
  Warnings: 2 (1 resolved, 1 auto-resolved)
  Actions Taken: 1 auto-restart

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent Performance Metrics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Response Times (p95):
    coder-001: 345ms
    reviewer-002: 267ms
    tester-003: 678ms
    researcher-004: 456ms
    planner-005: 234ms
    coordinator-006: 189ms
    github-007: 512ms
    gemini-008: 1,456ms (slow)
    codex-009: 789ms
    neural-010: 234ms (post-restart)
    analyzer-011: 398ms
    monitor-012: 156ms

  Task Success Rates:
    coder-001: 99.2% (2/247 failed)
    reviewer-002: 100% (0/187 failed)
    tester-003: 97.9% (12/579 failed)
    researcher-004: 97.1% (1/35 failed)
    neural-010: 62.5% (3/8 failed) ⚠️
      Note: Failures occurred before restart

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Health Check Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ✅ ALL HEALTHY (post-recovery)
Duration: 15 minutes
Checks Performed: 180 (12 agents × 15 intervals)

Agent Status:
  ✅ Healthy: 12 agents (100%)
  ⚠️  Warning: 0 agents
  ❌ Critical: 0 agents
  💀 Dead: 0 agents

Resource Summary:
  Total CPU: 38% average
  Total Memory: 14.5GB / 48GB (30%)
  Active Tasks: 28
  Queued Tasks: 34
  Completed Tasks: 1,234
  Failed Tasks: 34 (2.7%)

Actions Taken:
  Auto-restarts: 1 (neural-010)
  Manual interventions: 0
  Alerts sent: 4 (all resolved)

Recommendations:
  ✅ System operating normally
  ✅ All agents healthy
  ⚠️  Consider increasing memory for neural-train agents
  ⚠️  Monitor gemini-search for CPU spikes during large queries
  ✅ Auto-recovery working as expected

Export:
  Prometheus: http://localhost:9090/metrics
  JSON: /tmp/agent-health-2025-11-01.json
  Dashboard: http://grafana.example.com/d/agent-health

✅ Agent Health Check Complete!
```

## Chains With

```bash
# Monitor → alert → recover
/agent-health-check --alert true

# Health check → metrics → analyze
/agent-health-check && /agent-metrics && /performance-report

# Continuous monitoring
/agent-health-check --duration 1440 --export prometheus

# Pre-deployment agent check
/agent-health-check && /swarm-status && /k8s-deploy
```

## See Also
- `/agent-metrics` - Detailed agent performance metrics
- `/monitoring-configure` - Setup monitoring infrastructure
- `/alert-configure` - Configure alert thresholds
- `/swarm-monitor` - Swarm-level monitoring
- `/agent-rca` - Root cause analysis for agent failures
