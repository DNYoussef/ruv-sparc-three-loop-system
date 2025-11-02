---
name: monitoring-configure
category: monitoring
version: 1.0.0
---

# /monitoring-configure

Configure comprehensive monitoring infrastructure with metrics, logging, and alerting.

## Usage
```bash
/monitoring-configure [options]
```

## Parameters
- `--stack` - Monitoring stack: prometheus|datadog|newrelic|custom (default: prometheus)
- `--environment` - Target environment: dev|staging|production (required)
- `--metrics` - Enable metrics collection (default: true)
- `--logging` - Enable centralized logging (default: true)
- `--tracing` - Enable distributed tracing (default: true)
- `--alerting` - Configure alerting rules (default: true)
- `--dashboards` - Create pre-built dashboards (default: true)
- `--retention` - Metrics retention in days (default: 30)

## What It Does

**Complete Monitoring Setup**:
1. 📊 **Metrics Collection**: Prometheus, StatsD, custom metrics
2. 📝 **Centralized Logging**: ELK stack, Loki, CloudWatch
3. 🔍 **Distributed Tracing**: Jaeger, Zipkin, AWS X-Ray
4. 🚨 **Alerting**: Alert rules, notification channels
5. 📈 **Dashboards**: Grafana, DataDog, custom viz
6. 🎯 **Service Discovery**: Auto-discovery of services
7. 🔔 **Notification Channels**: Slack, PagerDuty, email
8. 📊 **SLO/SLA Tracking**: Service level objectives

**Monitored Components**:
- Application metrics (request rate, latency, errors)
- Infrastructure (CPU, memory, disk, network)
- Database (queries, connections, slow queries)
- Cache (hit ratio, evictions, memory)
- Message queues (throughput, lag, errors)
- External services (API calls, latency, errors)

## Examples

```bash
# Configure monitoring for staging
/monitoring-configure --environment staging

# Full stack with custom retention
/monitoring-configure --environment production --retention 90

# Metrics and alerting only (no tracing)
/monitoring-configure --environment dev --tracing false

# DataDog stack
/monitoring-configure --stack datadog --environment production

# Custom monitoring stack
/monitoring-configure --stack custom --metrics true --logging true

# Quick setup with defaults
/monitoring-configure --environment staging --dashboards true
```

## Output

```
📊 Monitoring Configuration Started

Environment: production
Stack: Prometheus + Grafana + Loki + Jaeger
Retention: 30 days

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metrics Collection Setup (Prometheus)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔍 Discovering services...
     ✅ API servers: 3 instances found
     ✅ Database: 1 primary, 2 replicas
     ✅ Redis: 1 master, 1 replica
     ✅ Load balancer: 1 instance
     ✅ Message queue: 3 brokers
     ✅ Background workers: 5 instances

  📊 Configuring Prometheus...
     ✅ Scrape configs created:
        - api-servers (interval: 15s)
        - postgres-exporter (interval: 30s)
        - redis-exporter (interval: 30s)
        - node-exporter (interval: 15s)
        - rabbitmq-exporter (interval: 30s)

     ✅ Retention: 30 days
     ✅ Storage: 50GB allocated
     ✅ Query timeout: 2m

  📈 Metrics Endpoints:
     ✅ /metrics exposed on all services
     ✅ Custom metrics registered:
        - http_request_duration_seconds
        - http_requests_total
        - database_query_duration_seconds
        - cache_hit_ratio
        - task_queue_size
        - external_api_calls_total

  🔄 Service Discovery:
     ✅ Kubernetes SD configured
     ✅ Auto-discovery enabled for:
        - Pod annotations (prometheus.io/scrape)
        - Service monitors
        - Pod monitors

  ✅ Prometheus deployed: http://prometheus.prod.svc:9090
  Duration: 45s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Centralized Logging Setup (Loki)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📝 Configuring Loki...
     ✅ Storage: S3 bucket (prod-logs)
     ✅ Retention: 30 days
     ✅ Compression: gzip
     ✅ Index: 24h chunks

  🔄 Log Aggregation:
     ✅ Promtail deployed to all nodes
     ✅ Log sources configured:
        - Application logs (/var/log/app/*.log)
        - Container logs (stdout/stderr)
        - System logs (/var/log/syslog)
        - Audit logs (/var/log/audit.log)

  🏷️ Log Labels:
     ✅ environment=production
     ✅ namespace={k8s_namespace}
     ✅ pod={k8s_pod}
     ✅ container={k8s_container}
     ✅ level={log_level}
     ✅ service={service_name}

  📊 Log Parsing:
     ✅ JSON logs: Automatic parsing
     ✅ Structured logs: Field extraction
     ✅ Error detection: Regex patterns
     ✅ Sampling: 10% for debug logs

  ✅ Loki deployed: http://loki.prod.svc:3100
  Duration: 38s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Distributed Tracing Setup (Jaeger)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔍 Configuring Jaeger...
     ✅ Storage: Elasticsearch
     ✅ Retention: 7 days
     ✅ Sampling: Adaptive (1% baseline, 100% errors)

  🎯 Instrumentation:
     ✅ OpenTelemetry SDK injected
     ✅ Automatic instrumentation:
        - HTTP requests
        - Database queries
        - Redis operations
        - Message queue operations
        - External API calls

  📊 Trace Context:
     ✅ Propagation: W3C Trace Context
     ✅ Baggage: Custom attributes
     ✅ Span attributes:
        - service.name
        - http.method
        - http.url
        - http.status_code
        - db.statement
        - error (if applicable)

  🔗 Service Map:
     ✅ Auto-generated service dependency graph
     ✅ Latency percentiles per edge
     ✅ Error rate per service

  ✅ Jaeger deployed:
     Query UI: http://jaeger.prod.svc:16686
     Collector: http://jaeger.prod.svc:14268
  Duration: 52s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Alerting Rules Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🚨 Creating alert rules...

  Critical Alerts:
    ✅ HighErrorRate (>5% for 5m)
       Severity: critical
       Channels: PagerDuty, Slack
       Notification: Immediate

    ✅ ServiceDown (all instances down)
       Severity: critical
       Channels: PagerDuty, Slack, Phone
       Notification: Immediate

    ✅ DatabaseDown (primary unreachable)
       Severity: critical
       Channels: PagerDuty, Slack
       Notification: Immediate

  High Priority Alerts:
    ✅ HighLatency (p95 >1s for 10m)
       Severity: high
       Channels: Slack
       Notification: 5-minute delay

    ✅ HighCPU (>80% for 15m)
       Severity: high
       Channels: Slack, Email
       Notification: 10-minute delay

    ✅ HighMemory (>85% for 15m)
       Severity: high
       Channels: Slack, Email
       Notification: 10-minute delay

  Warning Alerts:
    ✅ DiskSpaceLow (<15% free)
       Severity: warning
       Channels: Email
       Notification: 1-hour delay

    ✅ CertificateExpiry (<30 days)
       Severity: warning
       Channels: Email
       Notification: Daily digest

  📋 Alert Manager:
     ✅ Grouping: By alertname, cluster
     ✅ Inhibition rules: Critical inhibits warnings
     ✅ Silences: Manual and auto-silence support
     ✅ Routing tree configured

  ✅ Alert Manager deployed: http://alertmanager.prod.svc:9093
  Duration: 28s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Notification Channels
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔔 Configuring notification channels...

    ✅ Slack Integration
       Workspace: mycompany.slack.com
       Channels:
         - #alerts-critical (critical, high)
         - #alerts-warning (warning)
         - #deployments (deployment events)
       Webhook: https://hooks.slack.com/services/...

    ✅ PagerDuty Integration
       Service: Production Alerts
       Integration key: ••••••••
       Escalation policy: On-call rotation
       Auto-resolve: Enabled

    ✅ Email Integration
       SMTP: smtp.sendgrid.com:587
       From: alerts@example.com
       To:
         - ops-team@example.com (critical, high)
         - engineering@example.com (all)

    ✅ Webhook Integration
       Custom webhooks:
         - https://api.example.com/webhooks/alerts
         - https://ops-dashboard.example.com/alerts

  Duration: 18s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dashboard Creation (Grafana)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📈 Configuring Grafana...
     ✅ Data sources added:
        - Prometheus (metrics)
        - Loki (logs)
        - Jaeger (traces)
        - PostgreSQL (application DB)

  📊 Creating dashboards...

    ✅ Application Overview
       Panels:
         - Request rate (RPS)
         - Response time (p50, p95, p99)
         - Error rate
         - Active users
         - Throughput
       URL: http://grafana.prod.svc/d/app-overview

    ✅ Infrastructure Health
       Panels:
         - CPU usage (per node)
         - Memory usage (per node)
         - Disk I/O
         - Network traffic
         - Pod status
       URL: http://grafana.prod.svc/d/infra-health

    ✅ Database Performance
       Panels:
         - Query duration (p95, p99)
         - Connections (active, idle)
         - Lock wait time
         - Slow queries
         - Replication lag
       URL: http://grafana.prod.svc/d/db-performance

    ✅ API Endpoints
       Panels:
         - Top endpoints (by volume)
         - Slowest endpoints
         - Error rates per endpoint
         - Request/response sizes
       URL: http://grafana.prod.svc/d/api-endpoints

    ✅ Real-time Monitoring
       Panels:
         - Live request stream
         - Active alerts
         - Service health map
         - Recent deployments
       URL: http://grafana.prod.svc/d/realtime

  🎨 Dashboard Features:
     ✅ Variable filters (environment, service, pod)
     ✅ Time range selector
     ✅ Auto-refresh (30s)
     ✅ Dark theme
     ✅ Alert annotations
     ✅ Deployment markers

  ✅ Grafana deployed: http://grafana.prod.svc:3000
  Default login: admin / (auto-generated password)
  Duration: 67s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLO/SLA Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎯 Defining SLOs...

    ✅ Availability SLO
       Target: 99.9% uptime
       Error budget: 43m downtime/month
       Current: 99.94% (within target)

    ✅ Latency SLO
       Target: p95 <500ms
       Error budget: 5% of requests can exceed
       Current: p95 287ms (within target)

    ✅ Error Rate SLO
       Target: <0.1% errors
       Error budget: 1 error per 1000 requests
       Current: 0.04% (within target)

  📊 SLO Dashboards:
     ✅ Error budget burn rate
     ✅ SLO compliance over time
     ✅ Remaining error budget
     ✅ Alerting on budget depletion

  Duration: 22s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Validation & Testing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🧪 Testing monitoring stack...

    ✅ Prometheus scraping metrics
       - API servers: 234 metrics/target
       - Database: 145 metrics/target
       - Redis: 89 metrics/target

    ✅ Loki receiving logs
       - Log rate: 1,234 logs/sec
       - Ingestion delay: 1.2s

    ✅ Jaeger receiving traces
       - Trace rate: 456 spans/sec
       - Sampling: 1.2% (adaptive)

    ✅ Alert Manager routing
       - Test alert sent to Slack: ✅ Received
       - Test alert sent to PagerDuty: ✅ Received

    ✅ Grafana dashboards loading
       - All 5 dashboards: ✅ Loading <2s
       - Data sources: ✅ Connected

  Duration: 34s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Monitoring Configuration Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ✅ COMPLETE
Environment: production
Total Duration: 5m 24s

Components Deployed:
  ✅ Prometheus (metrics)
  ✅ Loki (logging)
  ✅ Jaeger (tracing)
  ✅ Alert Manager (alerting)
  ✅ Grafana (visualization)

Metrics Collection:
  - Targets: 15 services
  - Metrics: 1,234 unique metrics
  - Scrape interval: 15-30s
  - Retention: 30 days
  - Storage: 50GB

Logging:
  - Log sources: 23 pods
  - Log rate: 1,234 logs/sec
  - Retention: 30 days
  - Storage: S3 (prod-logs)

Tracing:
  - Instrumented services: 8
  - Trace rate: 456 spans/sec
  - Sampling: Adaptive (1% baseline)
  - Retention: 7 days

Alerting:
  - Alert rules: 8 (3 critical, 3 high, 2 warning)
  - Notification channels: 4 (Slack, PagerDuty, Email, Webhook)
  - Alert Manager: ✅ Running

Dashboards:
  - Grafana dashboards: 5
  - Data sources: 4 (Prometheus, Loki, Jaeger, PostgreSQL)
  - URL: http://grafana.prod.svc:3000

Access URLs:
  📊 Prometheus: http://prometheus.prod.svc:9090
  📝 Loki: http://loki.prod.svc:3100
  🔍 Jaeger: http://jaeger.prod.svc:16686
  🚨 Alert Manager: http://alertmanager.prod.svc:9093
  📈 Grafana: http://grafana.prod.svc:3000

Next Steps:
  1. Access Grafana: http://grafana.prod.svc:3000
  2. Review dashboards and customize as needed
  3. Configure additional alert rules via Alert Manager
  4. Set up log retention policies
  5. Review SLO targets and adjust if needed

✅ Monitoring Infrastructure Ready!
```

## Chains With

```bash
# Configure monitoring → set alerts
/monitoring-configure --environment production && /alert-configure

# Deploy → configure monitoring
/k8s-deploy && /monitoring-configure --environment production

# Configure → verify → test alerts
/monitoring-configure && /agent-health-check && /alert-configure

# Full observability stack
/monitoring-configure && /log-stream && /trace-request
```

## See Also
- `/alert-configure` - Configure alert thresholds
- `/log-stream` - Real-time log streaming
- `/trace-request` - Distributed request tracing
- `/agent-health-check` - Agent monitoring
- `/profiler-start` - Performance profiling
