---
name: log-stream
category: monitoring
version: 1.0.0
---

# /log-stream

Real-time log streaming and filtering from distributed services.

## Usage
```bash
/log-stream [service] [options]
```

## Parameters
- `service` - Service name to stream logs from (default: all)
- `--level` - Log level filter: debug|info|warn|error|fatal (default: info)
- `--namespace` - Kubernetes namespace (default: default)
- `--pod` - Specific pod name (optional)
- `--container` - Container name in pod (optional)
- `--grep` - Grep pattern to filter logs (regex supported)
- `--tail` - Number of lines to show initially (default: 100)
- `--follow` - Follow log output (default: true)
- `--timestamps` - Show timestamps (default: true)
- `--format` - Output format: text|json|pretty (default: pretty)

## What It Does

**Real-Time Log Aggregation**:
1. 📜 **Live Streaming**: Tail logs from multiple services
2. 🔍 **Pattern Matching**: Filter logs with regex
3. 🎨 **Syntax Highlighting**: Color-coded log levels
4. 📊 **Structured Logs**: Parse JSON logs
5. 🔗 **Multi-Service**: Stream from multiple pods/containers
6. ⏱️ **Timestamp Parsing**: Local and UTC timestamps
7. 📈 **Rate Display**: Log volume metrics
8. 🚨 **Error Detection**: Highlight errors and stack traces

## Examples

```bash
# Stream logs from all services
/log-stream

# Stream from specific service
/log-stream api-server

# Error logs only
/log-stream --level error

# Grep for specific pattern
/log-stream --grep "payment.*failed"

# Stream from specific pod
/log-stream --pod api-server-abc123

# Show last 500 lines
/log-stream --tail 500 --follow false

# JSON output for parsing
/log-stream --format json

# Stream from namespace
/log-stream --namespace production --level warn
```

## Output

```
📜 Log Stream Started

Service: api-server
Namespace: production
Level: info+
Follow: true
Initial lines: 100

Streaming from 3 pods:
  - api-server-abc123 (container: api)
  - api-server-def456 (container: api)
  - api-server-ghi789 (container: api)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Live Log Stream
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[2025-11-01 12:34:22.123] [INFO] [api-abc123] Server started on port 8080
[2025-11-01 12:34:22.145] [INFO] [api-def456] Server started on port 8080
[2025-11-01 12:34:22.167] [INFO] [api-ghi789] Server started on port 8080
[2025-11-01 12:34:23.234] [INFO] [api-abc123] Connected to database: postgresql://db:5432
[2025-11-01 12:34:23.256] [INFO] [api-def456] Connected to database: postgresql://db:5432
[2025-11-01 12:34:23.278] [INFO] [api-ghi789] Connected to database: postgresql://db:5432

[2025-11-01 12:34:30.456] [INFO] [api-abc123] GET /api/users - 200 OK (45ms)
  Request ID: req-1234567890
  User ID: user-abc
  IP: 192.168.1.100

[2025-11-01 12:34:31.234] [INFO] [api-def456] POST /api/orders - 201 Created (234ms)
  Request ID: req-1234567891
  User ID: user-xyz
  Order ID: ord-9876543210
  IP: 192.168.1.101

[2025-11-01 12:34:32.567] [WARN] [api-ghi789] Slow query detected
  Query: SELECT * FROM users WHERE email = $1
  Duration: 1,234ms (threshold: 1,000ms)
  Request ID: req-1234567892

[2025-11-01 12:34:35.123] [ERROR] [api-abc123] Payment processing failed
  Request ID: req-1234567893
  User ID: user-def
  Error: Stripe API timeout
  Status code: 504
  Stack trace:
    at PaymentService.processPayment (/app/services/payment.js:156)
    at OrderController.createOrder (/app/controllers/order.js:89)
    at Layer.handle (/app/node_modules/express/lib/router/layer.js:95)

[2025-11-01 12:34:36.789] [INFO] [api-def456] Cache miss for key: user:profile:user-abc
  Fetching from database...
  Duration: 23ms

[2025-11-01 12:34:37.456] [INFO] [api-abc123] POST /api/auth/login - 200 OK (189ms)
  Request ID: req-1234567894
  User ID: user-ghi
  Token issued: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
  IP: 192.168.1.102

[2025-11-01 12:34:40.234] [ERROR] [api-ghi789] Database connection lost
  Error: ECONNREFUSED 10.0.1.45:5432
  Retrying in 5 seconds...
  Attempt: 1/3

[2025-11-01 12:34:41.567] [INFO] [api-def456] GET /api/products?category=electronics - 200 OK (78ms)
  Request ID: req-1234567895
  Results: 24 products
  Cache: HIT

[2025-11-01 12:34:45.123] [INFO] [api-ghi789] Database connection restored
  Connected to: postgresql://db:5432/myapp
  Connection pool: 5/20 active

[2025-11-01 12:34:46.789] [WARN] [api-abc123] High memory usage detected
  Memory: 1.8GB / 2GB (90%)
  Heap used: 1.6GB
  Garbage collection triggered

[2025-11-01 12:34:50.456] [INFO] [api-def456] PUT /api/users/user-abc - 200 OK (123ms)
  Request ID: req-1234567896
  Fields updated: name, email
  User ID: user-abc

[2025-11-01 12:34:55.234] [ERROR] [api-abc123] Uncaught exception
  Error: Cannot read property 'id' of undefined
  Stack trace:
    at UserService.getProfile (/app/services/user.js:234)
    at async UserController.getProfile (/app/controllers/user.js:45)
  Process: Restarting...

[2025-11-01 12:35:00.567] [INFO] [api-abc123] Server started on port 8080 (restarted)
  Uptime: 0s
  Memory: 450MB

[2025-11-01 12:35:02.123] [INFO] [api-ghi789] POST /api/webhooks/stripe - 200 OK (45ms)
  Request ID: req-1234567897
  Event: payment_intent.succeeded
  Signature verified: true

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Log Stream Statistics (Last 5 minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Logs: 1,234
  Rate: 4.1 logs/second

  By Level:
    INFO: 1,100 (89.1%)
    WARN: 89 (7.2%)
    ERROR: 45 (3.6%)
    DEBUG: 0 (0%)

  By Pod:
    api-abc123: 412 logs (33.4%)
    api-def456: 398 logs (32.3%)
    api-ghi789: 424 logs (34.3%)

  Top Log Sources:
    1. GET /api/users: 234 requests
    2. POST /api/orders: 187 requests
    3. GET /api/products: 145 requests
    4. POST /api/auth/login: 89 requests
    5. PUT /api/users/*: 67 requests

  Errors Detected:
    Payment processing failed: 12 occurrences
    Database connection lost: 3 occurrences
    Uncaught exception: 2 occurrences
    Stripe API timeout: 8 occurrences

  Performance Warnings:
    Slow queries (>1s): 15 occurrences
    High memory usage: 4 occurrences

Press Ctrl+C to stop streaming
```

## Chains With

```bash
# Stream logs → analyze errors
/log-stream --level error | grep "payment"

# Stream logs → export for analysis
/log-stream --format json > logs.json

# Monitor deployment logs
/k8s-deploy && /log-stream --pod api-server-new-* --follow true

# Debug failed requests
/log-stream --grep "request_id:req-123" --timestamps true
```

## See Also
- `/monitoring-configure` - Setup logging infrastructure
- `/trace-request` - Distributed tracing for requests
- `/agent-health-check` - Monitor agent health
- `/alert-configure` - Configure log-based alerts
