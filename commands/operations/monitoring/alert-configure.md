---
name: alert-configure
category: monitoring
version: 1.0.0
---

# /alert-configure

Configure alerting rules, thresholds, and notification channels for proactive monitoring.

## Usage
```bash
/alert-configure [alert_type] [options]
```

## Parameters
- `alert_type` - Alert category: performance|errors|infrastructure|custom (default: all)
- `--severity` - Alert severity: critical|high|warning|info (default: all)
- `--threshold-error-rate` - Error rate threshold % (default: 5)
- `--threshold-latency` - Latency threshold in ms (default: 1000)
- `--threshold-cpu` - CPU usage threshold % (default: 80)
- `--threshold-memory` - Memory usage threshold % (default: 85)
- `--channels` - Notification channels (comma-separated, default: slack)
- `--environment` - Target environment (default: current)

## What It Does

**Comprehensive Alerting Setup**:
1. 🚨 **Alert Rules**: Define conditions and thresholds
2. 📢 **Notification Routing**: Channel selection per severity
3. 🔔 **Escalation Policies**: Progressive escalation
4. 🎯 **Threshold Tuning**: Environment-specific thresholds
5. 📊 **Alert Grouping**: Reduce noise with intelligent grouping
6. ⏰ **Schedule-Based**: Time windows and maintenance modes
7. 🔕 **Silencing**: Manual and auto-silence capabilities
8. 📈 **Alert Analytics**: Track alert trends

## Examples

```bash
# Configure all alerts
/alert-configure

# Performance alerts only
/alert-configure performance --threshold-latency 500

# Critical alerts with custom thresholds
/alert-configure --severity critical --threshold-error-rate 1

# Multi-channel notification
/alert-configure --channels slack,pagerduty,email

# Infrastructure alerts
/alert-configure infrastructure --threshold-cpu 70 --threshold-memory 80

# Custom alert rules
/alert-configure custom --environment production
```

## Output

```
🚨 Alert Configuration Started

Environment: production
Alert Types: all
Severity Levels: all
Channels: slack, pagerduty, email

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Critical Alerts (Immediate Response)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ ServiceDown
     Condition: ALL instances unavailable for 1m
     Threshold: 0 healthy instances
     Channels: PagerDuty, Slack (#incidents), Phone
     Escalation: Immediate → On-call engineer
     Auto-page: Yes

  ✅ CriticalErrorRate
     Condition: Error rate >10% for 5m
     Threshold: 10% errors
     Channels: PagerDuty, Slack (#incidents)
     Escalation: 5m → Engineering lead
     Current: 0.4% (OK)

  ✅ DatabaseDown
     Condition: Primary database unreachable for 30s
     Threshold: Connection timeout
     Channels: PagerDuty, Slack (#database), Phone
     Escalation: Immediate → DBA team
     Auto-failover: Enabled

  ✅ OutOfMemory
     Condition: Container OOM killed
     Threshold: Immediate on OOM event
     Channels: PagerDuty, Slack (#infrastructure)
     Escalation: Immediate → Platform team
     Auto-restart: Enabled

  ✅ DiskFull
     Condition: Disk usage >95% for 2m
     Threshold: 95% usage
     Channels: PagerDuty, Slack (#infrastructure)
     Escalation: 10m → Platform team
     Auto-cleanup: Enabled (logs, temp files)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
High Priority Alerts (Fast Response)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ HighLatency
     Condition: p95 latency >1s for 10m
     Threshold: 1000ms (p95)
     Channels: Slack (#performance)
     Escalation: 15m → Performance team
     Current: 287ms (OK)

  ✅ HighErrorRate
     Condition: Error rate >5% for 10m
     Threshold: 5% errors
     Channels: Slack (#alerts)
     Escalation: 15m → On-call
     Current: 0.4% (OK)

  ✅ HighCPU
     Condition: CPU >80% for 15m
     Threshold: 80% usage
     Channels: Slack (#infrastructure), Email
     Escalation: 30m → Platform team
     Auto-scale: Triggered at 85%
     Current: 45% (OK)

  ✅ HighMemory
     Condition: Memory >85% for 15m
     Threshold: 85% usage
     Channels: Slack (#infrastructure), Email
     Escalation: 30m → Platform team
     Auto-scale: Triggered at 90%
     Current: 62% (OK)

  ✅ SlowDatabaseQueries
     Condition: Query duration >5s for 5m
     Threshold: 5000ms
     Channels: Slack (#database)
     Escalation: 20m → DBA team
     Auto-log: Slow query logging enabled

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Warning Alerts (Monitor & Plan)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ ElevatedErrorRate
     Condition: Error rate >1% for 30m
     Threshold: 1% errors
     Channels: Email
     Escalation: None (informational)
     Current: 0.4% (OK)

  ✅ DiskSpaceLow
     Condition: Disk usage >85% for 1h
     Threshold: 85% usage
     Channels: Email (daily digest)
     Escalation: None
     Current: 67% (OK)

  ✅ CertificateExpiry
     Condition: SSL cert expires in <30 days
     Threshold: 30 days
     Channels: Email (weekly)
     Escalation: None
     Renewal: Auto-renewal enabled (Let's Encrypt)

  ✅ LowCacheHitRatio
     Condition: Cache hit ratio <70% for 2h
     Threshold: 70% hit ratio
     Channels: Slack (#performance)
     Escalation: None
     Current: 76% (OK)

  ✅ PodRestartLoop
     Condition: Pod restarted >3 times in 1h
     Threshold: 3 restarts
     Channels: Slack (#infrastructure)
     Escalation: None
     Auto-debug: Logs collected

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Notification Channel Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📱 Slack Integration
     Workspace: mycompany.slack.com
     Channels configured:
       ✅ #incidents (critical alerts)
       ✅ #alerts (high priority)
       ✅ #performance (latency, throughput)
       ✅ #infrastructure (CPU, memory, disk)
       ✅ #database (DB-related)
     Webhook: https://hooks.slack.com/services/T...
     Format: Rich formatting with graphs
     Mentions: @oncall for critical

  📞 PagerDuty Integration
     Service: Production Alerts
     Integration key: ••••••••
     Escalation policy:
       L1: On-call engineer (immediate)
       L2: Engineering lead (+5m)
       L3: CTO (+15m)
     Auto-resolve: Yes
     Acknowledge timeout: 15m

  📧 Email Integration
     SMTP: smtp.sendgrid.com:587
     From: alerts@example.com
     Recipients:
       Critical: ops-team@example.com, oncall@example.com
       High: engineering@example.com
       Warning: devops@example.com (daily digest)
     HTML formatting: Enabled
     Inline graphs: Yes

  📲 Webhook Integration
     Endpoints:
       ✅ https://api.example.com/webhooks/alerts
       ✅ https://ops-dashboard.example.com/alerts
     Format: JSON
     Authentication: Bearer token
     Retry: 3 attempts with backoff

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Alert Routing & Grouping
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔀 Routing Rules:
     ✅ severity=critical → PagerDuty + Slack
     ✅ severity=high → Slack + Email
     ✅ severity=warning → Email (digest)
     ✅ alertname=DatabaseDown → DBA team
     ✅ namespace=production → Immediate escalation

  📦 Grouping Configuration:
     ✅ Group by: alertname, cluster, namespace
     ✅ Group wait: 30s (collect similar alerts)
     ✅ Group interval: 5m (send grouped alerts)
     ✅ Repeat interval: 4h (re-notify if unresolved)

  🔕 Inhibition Rules:
     ✅ Critical alerts inhibit warnings
     ✅ ServiceDown inhibits HighLatency
     ✅ DatabaseDown inhibits SlowQueries
     ✅ Reduce noise by 60-70%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Schedule & Maintenance Windows
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⏰ Business Hours Routing:
     Weekdays 9am-5pm EST:
       ✅ Critical → PagerDuty + Slack
       ✅ High → Slack
       ✅ Warning → Email

     After hours & weekends:
       ✅ Critical → PagerDuty (immediate)
       ✅ High → PagerDuty (+15m escalation)
       ✅ Warning → Email (next business day)

  🔧 Maintenance Windows:
     ✅ Scheduled maintenance: Auto-silence alerts
     ✅ Deployment windows: Suppress deployment-related alerts
     ✅ Backup windows: Suppress DB alerts during backups
     Example: Every Sunday 2am-4am EST

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Alert Testing & Validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🧪 Sending test alerts...

    ✅ Test alert: HighErrorRate (warning)
       Slack: ✅ Received in #alerts (3.2s)
       Email: ✅ Delivered to engineering@example.com (4.5s)

    ✅ Test alert: ServiceDown (critical)
       PagerDuty: ✅ Incident created (2.1s)
       Slack: ✅ Posted to #incidents with @oncall (2.8s)
       Phone: ✅ Call initiated to on-call (5.4s)

    ✅ Test alert: HighCPU (high)
       Slack: ✅ Received in #infrastructure (3.1s)
       Email: ✅ Delivered to devops@example.com (4.2s)

  ✅ All notification channels working
  ✅ Routing rules validated
  ✅ Escalation policies tested

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Alert Configuration Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ✅ COMPLETE
Environment: production
Total Duration: 2m 34s

Alert Rules Created:
  Critical: 5 rules
  High: 5 rules
  Warning: 5 rules
  Total: 15 rules

Notification Channels:
  ✅ Slack (4 channels)
  ✅ PagerDuty (1 service)
  ✅ Email (3 distribution lists)
  ✅ Webhooks (2 endpoints)

Routing Configuration:
  ✅ Severity-based routing
  ✅ Alert grouping (30s wait, 5m interval)
  ✅ Inhibition rules (3 rules)
  ✅ Business hours scheduling

Current Alert Status:
  Active alerts: 0
  Silenced alerts: 0
  Firing rate (24h): 12 alerts
  Resolution rate (24h): 100%
  False positive rate: 2.1%

Thresholds Configured:
  Error rate: >5% (high), >10% (critical)
  Latency: >1000ms p95 (high)
  CPU: >80% (high)
  Memory: >85% (high)
  Disk: >85% (warning), >95% (critical)

Access:
  Alert Manager UI: http://alertmanager.prod.svc:9093
  Grafana Alerts: http://grafana.prod.svc:3000/alerting
  Slack workspace: mycompany.slack.com
  PagerDuty: https://mycompany.pagerduty.com

Next Steps:
  1. Monitor alert firing rates and adjust thresholds
  2. Review escalation policies after first week
  3. Tune alert grouping to reduce noise
  4. Add custom alerts for business metrics
  5. Schedule maintenance windows for deployments

✅ Alerting Configuration Complete!
```

## Chains With

```bash
# Configure monitoring → set alerts
/monitoring-configure && /alert-configure

# Configure alerts → test with health check
/alert-configure && /agent-health-check

# Full observability setup
/monitoring-configure && /alert-configure && /log-stream

# Update alert thresholds after load test
/load-test && /alert-configure --threshold-latency 500
```

## See Also
- `/monitoring-configure` - Setup monitoring infrastructure
- `/agent-health-check` - Monitor agent health
- `/log-stream` - Real-time log streaming
- `/trace-request` - Distributed tracing
- `/profiler-start` - Performance profiling
