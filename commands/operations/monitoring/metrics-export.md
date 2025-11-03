---
name: metrics-export
description: Export performance metrics to external monitoring systems
version: 2.0.0
category: monitoring
complexity: medium
tags: [metrics, export, monitoring, integration, observability, analytics]
author: ruv-SPARC Monitoring Team
created: 2025-11-01
last_updated: 2025-11-01
dependencies: [agent-metrics, task-status, memory-stats]
chains_with: [agent-benchmark, performance-analysis, memory-stats]
evidence_based_techniques: [self-consistency]
---

# /metrics-export - External Metrics Integration

## Overview

The `/metrics-export` command exports performance metrics, system statistics, and monitoring data to external systems for analysis, alerting, and long-term storage.

## Usage

```bash
# Export to JSON file
npx claude-flow@alpha metrics export --output "metrics.json"

# Export to Prometheus format
npx claude-flow@alpha metrics export \
  --format prometheus \
  --output "metrics.prom"

# Export to InfluxDB
npx claude-flow@alpha metrics export \
  --target influxdb \
  --url "http://localhost:8086" \
  --database "ruv-sparc"

# Export to CloudWatch
npx claude-flow@alpha metrics export \
  --target cloudwatch \
  --namespace "RuvSPARC/Production"

# Stream metrics continuously
npx claude-flow@alpha metrics export \
  --stream \
  --target datadog \
  --api-key $DD_API_KEY

# Export specific metrics
npx claude-flow@alpha metrics export \
  --metrics "agent-performance,task-completion,memory-usage" \
  --output "selected-metrics.json"
```

## Parameters

### Output Options
- `--output <path>` - Output file path
- `--format <format>` - Export format: `json`, `prometheus`, `csv`, `influx`, `statsd`
- `--target <system>` - Target system: `file`, `prometheus`, `influxdb`, `cloudwatch`, `datadog`, `grafana`

### Filtering
- `--metrics <list>` - Specific metrics (comma-separated)
- `--time-range <range>` - Time range: `1h`, `24h`, `7d`, `30d`
- `--interval <duration>` - Aggregation interval: `1m`, `5m`, `1h`
- `--agents <ids>` - Specific agents

### Streaming
- `--stream` - Continuous streaming mode
- `--push-interval <duration>` - Push interval (default: 60s)

### Integration
- `--url <url>` - Target system URL
- `--api-key <key>` - API key for authentication
- `--database <name>` - Database/namespace name
- `--tags <tags>` - Additional tags (key=value)

## Supported Metrics

### Agent Metrics
- `agent.tasks.completed` - Total tasks completed
- `agent.tasks.success_rate` - Success rate %
- `agent.tasks.avg_time` - Average task time
- `agent.memory.usage` - Memory usage
- `agent.cpu.usage` - CPU usage
- `agent.api.calls` - API call count
- `agent.tokens.used` - Token consumption

### Task Metrics
- `task.queue.length` - Tasks in queue
- `task.execution.time` - Task execution time
- `task.error.rate` - Task error rate
- `task.throughput` - Tasks per hour

### Memory Metrics
- `memory.total.size` - Total memory size
- `memory.keys.count` - Number of keys
- `memory.fragmentation` - Fragmentation %
- `memory.growth.rate` - Growth rate

### System Metrics
- `system.uptime` - System uptime
- `system.swarms.active` - Active swarms
- `system.agents.active` - Active agents
- `system.performance.score` - Overall performance score

## Export Formats

### JSON
```json
{
  "timestamp": "2025-11-01T10:30:45Z",
  "metrics": {
    "agent.coder-123.tasks.completed": 145,
    "agent.coder-123.tasks.success_rate": 95.2,
    "agent.coder-123.memory.usage": 124000000,
    "task.queue.length": 23
  }
}
```

### Prometheus
```
# HELP agent_tasks_completed Total tasks completed by agent
# TYPE agent_tasks_completed counter
agent_tasks_completed{agent_id="coder-123",type="coder"} 145

# HELP agent_memory_usage Memory usage in bytes
# TYPE agent_memory_usage gauge
agent_memory_usage{agent_id="coder-123"} 124000000
```

### InfluxDB Line Protocol
```
agent_metrics,agent_id=coder-123,type=coder tasks_completed=145,success_rate=95.2,memory_usage=124000000 1730458245000000000
```

### CSV
```csv
timestamp,metric,value,tags
2025-11-01T10:30:45Z,agent.tasks.completed,145,agent_id=coder-123;type=coder
2025-11-01T10:30:45Z,agent.memory.usage,124000000,agent_id=coder-123
```

## Integration Examples

### Prometheus

```bash
# Export Prometheus metrics
npx claude-flow@alpha metrics export \
  --format prometheus \
  --output "/var/lib/prometheus/ruv-sparc-metrics.prom"

# Configure Prometheus scraping
# prometheus.yml:
# scrape_configs:
#   - job_name: 'ruv-sparc'
#     static_configs:
#       - targets: ['localhost:9090']
#     file_sd_configs:
#       - files:
#           - '/var/lib/prometheus/ruv-sparc-metrics.prom'
```

### Grafana Dashboard

```bash
# Stream to InfluxDB for Grafana
npx claude-flow@alpha metrics export \
  --stream \
  --target influxdb \
  --url "http://localhost:8086" \
  --database "ruv_sparc" \
  --push-interval 30s

# Grafana will auto-discover metrics from InfluxDB
```

### CloudWatch

```bash
# Export to AWS CloudWatch
npx claude-flow@alpha metrics export \
  --target cloudwatch \
  --namespace "RuvSPARC/Production" \
  --region us-east-1 \
  --tags "Environment=Production,Team=Engineering"
```

### Datadog

```bash
# Stream to Datadog
npx claude-flow@alpha metrics export \
  --stream \
  --target datadog \
  --api-key $DD_API_KEY \
  --site datadoghq.com \
  --tags "env:production,service:ruv-sparc"
```

## Automation

### Scheduled Exports

```bash
# Export metrics every hour
npx claude-flow@alpha metrics export \
  --schedule "hourly" \
  --output "/var/metrics/ruv-sparc-{timestamp}.json"

# Cron syntax
npx claude-flow@alpha metrics export \
  --cron "0 * * * *" \
  --target influxdb
```

### Continuous Streaming

```bash
# Stream in background
npx claude-flow@alpha metrics export \
  --stream \
  --background \
  --target prometheus \
  --push-interval 60s
```

## See Also

- `/agent-metrics` - Agent performance metrics
- `/agent-benchmark` - Benchmark agents
- `/memory-stats` - Memory statistics
- `/performance-analysis` - Performance analysis

---

**Version**: 2.0.0
**Last Updated**: 2025-11-01
**Maintained By**: ruv-SPARC Monitoring Team
