---
name: trace-request
category: monitoring
version: 1.0.0
---

# /trace-request

Distributed request tracing to track requests across microservices.

## Usage
```bash
/trace-request [trace_id] [options]
```

## Parameters
- `trace_id` - Specific trace ID to analyze (optional)
- `--service` - Filter by service name (optional)
- `--operation` - Filter by operation/endpoint (optional)
- `--min-duration` - Minimum trace duration in ms (optional)
- `--error` - Show only traces with errors (default: false)
- `--limit` - Number of traces to show (default: 20)
- `--time-range` - Time range: 15m|1h|6h|24h (default: 1h)
- `--format` - Output format: tree|timeline|json (default: tree)

## What It Does

**Distributed Tracing Analysis**:
1. 🔍 **Trace Lookup**: Find traces by ID or filters
2. 📊 **Span Visualization**: Show service call hierarchy
3. ⏱️ **Timing Analysis**: Identify slow spans
4. 🚨 **Error Detection**: Highlight errors in trace
5. 🔗 **Dependency Map**: Service interaction graph
6. 📈 **Performance Insights**: Bottleneck identification
7. 🎯 **Critical Path**: Longest execution path
8. 📝 **Metadata**: Headers, tags, logs per span

## Examples

```bash
# Show recent traces
/trace-request

# Analyze specific trace
/trace-request abc123def456

# Find slow traces (>1s)
/trace-request --min-duration 1000

# Show error traces only
/trace-request --error true

# Filter by service
/trace-request --service order-service --limit 10

# Timeline view
/trace-request abc123def456 --format timeline

# Export as JSON
/trace-request --time-range 24h --format json > traces.json
```

## Output

```
🔍 Distributed Request Tracing

Trace ID: abc123def456789
Started: 2025-11-01 12:34:56.123 UTC
Duration: 2,456ms
Services: 5
Spans: 12
Status: ❌ ERROR

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trace Tree Visualization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─ [api-gateway] POST /api/orders (2,456ms) ❌
│  Trace ID: abc123def456789
│  Span ID: span-001
│  Start: 12:34:56.123
│  Tags: http.method=POST, http.url=/api/orders, user_id=user-123
│  Error: Payment processing failed
│
├──┬─ [auth-service] POST /auth/verify (89ms) ✅
│  │  Span ID: span-002
│  │  Parent: span-001
│  │  Tags: http.status_code=200, token_valid=true
│  │
│  └─── [database] SELECT FROM users (23ms) ✅
│       Span ID: span-003
│       Parent: span-002
│       Tags: db.statement=SELECT * FROM users WHERE id = $1
│       Rows: 1
│
├──┬─ [order-service] POST /internal/orders/create (2,234ms) ❌
│  │  Span ID: span-004
│  │  Parent: span-001
│  │  Tags: http.status_code=500, order_id=ord-789
│  │  Error: Payment timeout
│  │
│  ├─── [inventory-service] POST /internal/inventory/reserve (156ms) ✅
│  │    Span ID: span-005
│  │    Parent: span-004
│  │    Tags: product_id=prod-456, quantity=2
│  │    Reserved: true
│  │
│  ├─── [database] BEGIN TRANSACTION (12ms) ✅
│  │    Span ID: span-006
│  │    Parent: span-004
│  │
│  ├─── [database] INSERT INTO orders (45ms) ✅
│  │    Span ID: span-007
│  │    Parent: span-004
│  │    Tags: db.statement=INSERT INTO orders (...) VALUES (...)
│  │
│  ├──┬─ [payment-service] POST /internal/payments/charge (1,892ms) ❌
│  │  │  Span ID: span-008
│  │  │  Parent: span-004
│  │  │  Tags: amount=99.99, currency=USD
│  │  │  Error: Stripe API timeout
│  │  │
│  │  ├─── [cache] GET payment:user-123 (8ms) ✅
│  │  │    Span ID: span-009
│  │  │    Parent: span-008
│  │  │    Tags: cache_hit=false
│  │  │
│  │  └─── [external] POST https://api.stripe.com/v1/charges (1,856ms) ❌
│  │       Span ID: span-010
│  │       Parent: span-008
│  │       Tags: http.status_code=504, gateway_timeout=true
│  │       Error: Gateway Timeout
│  │       Retries: 3
│  │
│  └─── [database] ROLLBACK (34ms) ✅
│       Span ID: span-011
│       Parent: span-004
│       Reason: Payment failed
│
└─── [notification-service] POST /internal/notifications/send (123ms) ✅
     Span ID: span-012
     Parent: span-001
     Tags: type=email, template=order_failed
     Sent: true

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Timeline View
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0ms         500ms       1000ms      1500ms      2000ms      2500ms
│           │           │           │           │           │
├─────────────────────────────────────────────────────────────┤ api-gateway (2,456ms)
│ ├──┤                                                         auth-service (89ms)
│ │ ├┤                                                         database (23ms)
│ ├────────────────────────────────────────────────────────┤  order-service (2,234ms)
│ │├─┤                                                        inventory-service (156ms)
│ │├┤                                                         database BEGIN (12ms)
│ │ ├┤                                                        database INSERT (45ms)
│ │ ├──────────────────────────────────────────────────────┤ payment-service (1,892ms)
│ │ │├┤                                                       cache GET (8ms)
│ │ │ ├────────────────────────────────────────────────────┤ Stripe API (1,856ms) ❌
│ │ │                                          ├┤            database ROLLBACK (34ms)
│ ├──┤                                                       notification (123ms)

Critical Path: api-gateway → order-service → payment-service → Stripe API
Total: 2,456ms (75% spent in Stripe API call)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Performance Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Slowest Spans:
    1. Stripe API call: 1,856ms (75.6%)
    2. order-service: 2,234ms (91.0%)
    3. payment-service: 1,892ms (77.0%)
    4. inventory-service: 156ms (6.3%)
    5. notification-service: 123ms (5.0%)

  Database Queries:
    Total: 4 queries
    Total time: 114ms (4.6% of trace)
    Slowest: INSERT INTO orders (45ms)

  Cache Operations:
    Total: 1 operation
    Total time: 8ms (0.3% of trace)
    Hit ratio: 0% (1 miss)

  External API Calls:
    Total: 1 call
    Total time: 1,856ms (75.6% of trace)
    Failures: 1 (100%)

  Network Overhead:
    Total latency: 234ms (9.5%)
    Service-to-service calls: 11

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Error Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Root Cause: Stripe API timeout

  Error Chain:
    1. [external] Stripe API timeout (504 Gateway Timeout)
       ↓
    2. [payment-service] Payment processing failed
       ↓
    3. [order-service] Transaction rolled back
       ↓
    4. [api-gateway] Order creation failed (500 Internal Server Error)

  Impact:
    - User: Order not placed
    - Inventory: Reserved items released
    - Payment: Not charged
    - Database: Transaction rolled back (data consistent)

  Recommendations:
    ✅ Inventory correctly released (no leaks)
    ✅ Transaction rolled back (data integrity preserved)
    ⚠️  Consider circuit breaker for Stripe API
    ⚠️  Add retry with exponential backoff
    ⚠️  Implement payment queue for resilience

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metadata & Tags
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Request Headers:
    X-Request-ID: req-1234567890
    User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1)
    Authorization: Bearer eyJ... (truncated)
    Content-Type: application/json

  Custom Tags:
    user_id: user-123
    session_id: sess-abc789
    experiment: checkout-v2
    ab_test_variant: variant-b
    client_version: 2.1.4
    platform: ios

  Logs (span-008: payment-service):
    [12:34:57.234] Initiating Stripe charge
    [12:34:57.456] Stripe API request sent
    [12:34:59.000] Timeout threshold reached (1500ms)
    [12:34:59.312] Retry attempt 1/3
    [12:35:00.890] Retry attempt 2/3
    [12:35:02.456] Retry attempt 3/3
    [12:35:03.789] All retries exhausted, marking as failed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Service Dependency Graph
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

             api-gateway
                 │
        ┌────────┼────────┐
        │        │        │
   auth-service  │   notification
        │        │
     database    │
                 │
           order-service
                 │
        ┌────────┼────────┐
        │        │        │
   inventory  database  payment
    -service            -service
                            │
                       ┌────┴────┐
                    cache     Stripe API

  Total Services: 6
  Database Connections: 3
  External Dependencies: 1 (Stripe)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trace Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ❌ FAILED
Trace ID: abc123def456789
Total Duration: 2,456ms
Services Involved: 5
Total Spans: 12
Errors: 2

Performance:
  Network latency: 234ms (9.5%)
  Database time: 114ms (4.6%)
  External APIs: 1,856ms (75.6%)
  Service logic: 252ms (10.3%)

Bottleneck: Stripe API call (1,856ms, 75.6% of trace)

Recommendations:
  1. Implement circuit breaker for Stripe API
  2. Add payment processing queue
  3. Use idempotency keys to enable safe retries
  4. Consider alternative payment provider as fallback
  5. Add timeout alerts for >1s external API calls

Related Traces:
  ✅ Similar successful trace: def456abc789 (345ms)
  ❌ Similar failed trace: ghi789def123 (2,234ms)

View in Jaeger: http://jaeger.prod.svc:16686/trace/abc123def456789
```

## Chains With

```bash
# Stream logs → find trace ID → analyze
/log-stream --grep "trace_id" | /trace-request <trace_id>

# Monitor slow requests
/trace-request --min-duration 1000 --time-range 1h

# Error analysis
/trace-request --error true --limit 50

# Performance debugging
/load-test && /trace-request --min-duration 500
```

## See Also
- `/log-stream` - Real-time log streaming
- `/monitoring-configure` - Setup tracing infrastructure
- `/profiler-start` - Performance profiling
- `/bottleneck-detect` - Bottleneck detection
- `/performance-report` - Performance analysis
