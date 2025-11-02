---
name: opentelemetry-observability
description: OpenTelemetry specialist for distributed tracing, metrics collection, log correlation, auto-instrumentation, custom spans, trace context propagation, and sampling strategies. Use when implementing observability in microservices, debugging production issues, monitoring performance, or requiring OpenTelemetry best practices. Handles integration with Jaeger/Zipkin/Tempo, Prometheus/Grafana, and cloud-native observability platforms.
category: Observability
complexity: High
triggers: ["opentelemetry", "otel", "distributed tracing", "observability", "metrics", "spans", "tracing", "jaeger", "zipkin", "tempo", "instrumentation"]
---

# OpenTelemetry Observability Specialist

Expert distributed tracing, metrics, and logging with OpenTelemetry for production observability.

## Purpose

Comprehensive OpenTelemetry expertise including auto-instrumentation, custom spans, metrics collection, log correlation, trace context propagation, and sampling. Ensures applications are fully observable with actionable telemetry data.

## When to Use

- Implementing distributed tracing in microservices
- Monitoring application performance (APM)
- Debugging production issues across services
- Setting up metrics collection and dashboards
- Correlating logs with traces
- Optimizing sampling strategies for cost/performance
- Migrating from proprietary APM to OpenTelemetry

## Prerequisites

**Required**: Understanding of distributed systems, HTTP, basic observability concepts

**Agents**: `cicd-engineer`, `perf-analyzer`, `backend-dev`, `system-architect`

## Core Workflows

### Workflow 1: Node.js Auto-Instrumentation

**Step 1: Install OpenTelemetry Packages**

```bash
npm install @opentelemetry/sdk-node \
  @opentelemetry/auto-instrumentations-node \
  @opentelemetry/exporter-trace-otlp-http \
  @opentelemetry/exporter-metrics-otlp-http
```

**Step 2: Initialize OpenTelemetry**

```javascript
// instrumentation.js
const { NodeSDK } = require('@opentelemetry/sdk-node');
const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node');
const { OTLPTraceExporter } = require('@opentelemetry/exporter-trace-otlp-http');
const { OTLPMetricExporter } = require('@opentelemetry/exporter-metrics-otlp-http');
const { Resource } = require('@opentelemetry/resources');
const { SemanticResourceAttributes } = require('@opentelemetry/semantic-conventions');

const sdk = new NodeSDK({
  resource: new Resource({
    [SemanticResourceAttributes.SERVICE_NAME]: 'my-service',
    [SemanticResourceAttributes.SERVICE_VERSION]: '1.0.0',
    [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: 'production',
  }),
  traceExporter: new OTLPTraceExporter({
    url: 'http://localhost:4318/v1/traces',
  }),
  metricReader: new PeriodicExportingMetricReader({
    exporter: new OTLPMetricExporter({
      url: 'http://localhost:4318/v1/metrics',
    }),
    exportIntervalMillis: 60000,
  }),
  instrumentations: [
    getNodeAutoInstrumentations({
      '@opentelemetry/instrumentation-http': { enabled: true },
      '@opentelemetry/instrumentation-express': { enabled: true },
      '@opentelemetry/instrumentation-pg': { enabled: true },
      '@opentelemetry/instrumentation-redis': { enabled: true },
    }),
  ],
});

sdk.start();

process.on('SIGTERM', () => {
  sdk.shutdown().then(
    () => console.log('Tracing terminated'),
    (err) => console.log('Error terminating tracing', err)
  );
});
```

**Step 3: Start Application with Instrumentation**

```bash
node --require ./instrumentation.js app.js
```

### Workflow 2: Custom Spans and Attributes

```javascript
const { trace } = require('@opentelemetry/api');

const tracer = trace.getTracer('my-service', '1.0.0');

async function processOrder(orderId) {
  const span = tracer.startSpan('processOrder', {
    attributes: {
      'order.id': orderId,
      'order.priority': 'high',
    },
  });

  try {
    // Set span status
    span.setStatus({ code: SpanStatusCode.OK });

    // Add event to span
    span.addEvent('order_validated', {
      'validation.result': 'success',
    });

    // Child span
    const childSpan = tracer.startSpan('calculateTotal', {
      parent: span,
    });
    const total = await calculateTotal(orderId);
    childSpan.setAttribute('order.total', total);
    childSpan.end();

    return total;
  } catch (error) {
    // Record exception
    span.recordException(error);
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: error.message,
    });
    throw error;
  } finally {
    span.end();
  }
}
```

### Workflow 3: Custom Metrics

```javascript
const { metrics } = require('@opentelemetry/api');

const meter = metrics.getMeter('my-service', '1.0.0');

// Counter: Monotonically increasing value
const orderCounter = meter.createCounter('orders.processed', {
  description: 'Total number of orders processed',
});

orderCounter.add(1, {
  'order.type': 'online',
  'order.status': 'completed',
});

// Histogram: Statistical distribution
const requestDuration = meter.createHistogram('http.server.duration', {
  description: 'HTTP request duration in milliseconds',
  unit: 'ms',
});

requestDuration.record(150, {
  'http.method': 'POST',
  'http.route': '/api/orders',
  'http.status_code': 200,
});

// UpDownCounter: Value can go up or down
const activeConnections = meter.createUpDownCounter('db.connections.active', {
  description: 'Number of active database connections',
});

activeConnections.add(1); // Connection opened
activeConnections.add(-1); // Connection closed

// ObservableGauge: Current value snapshot
const memoryUsage = meter.createObservableGauge('process.memory.usage', {
  description: 'Process memory usage in bytes',
  unit: 'bytes',
});

memoryUsage.addCallback((result) => {
  result.observe(process.memoryUsage().heapUsed, {
    'memory.type': 'heap',
  });
});
```

### Workflow 4: Context Propagation (W3C Trace Context)

```javascript
// Propagate context between services
const { propagation, context } = require('@opentelemetry/api');

// Client-side: Inject trace context into HTTP headers
async function callExternalService(url, data) {
  const span = tracer.startSpan('external_api_call');
  const headers = {};

  // Inject trace context into headers (W3C Trace Context)
  propagation.inject(context.active(), headers);

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...headers, // traceparent, tracestate headers
      },
      body: JSON.stringify(data),
    });
    return response.json();
  } finally {
    span.end();
  }
}

// Server-side: Extract trace context from HTTP headers
app.post('/api/process', (req, res) => {
  // Extract context from incoming headers
  const extractedContext = propagation.extract(context.active(), req.headers);

  context.with(extractedContext, () => {
    const span = tracer.startSpan('process_request');
    // This span will be a child of the parent trace from the caller
    // ...
    span.end();
  });

  res.json({ status: 'ok' });
});
```

### Workflow 5: Sampling Strategies

```javascript
const { ParentBasedSampler, AlwaysOnSampler, AlwaysOffSampler, TraceIdRatioBasedSampler } = require('@opentelemetry/sdk-trace-base');

// Probability-based sampling (10% of traces)
const sampler = new TraceIdRatioBasedSampler(0.1);

// Parent-based sampling with rate limiting
const parentBasedSampler = new ParentBasedSampler({
  root: new TraceIdRatioBasedSampler(0.1), // 10% for root spans
  remoteParentSampled: new AlwaysOnSampler(), // Always sample if parent sampled
  remoteParentNotSampled: new AlwaysOffSampler(), // Never sample if parent not sampled
  localParentSampled: new AlwaysOnSampler(),
  localParentNotSampled: new AlwaysOffSampler(),
});

const sdk = new NodeSDK({
  sampler: parentBasedSampler,
  // ... other config
});
```

## Best Practices

**1. Use Semantic Conventions**
```javascript
// ✅ GOOD: Standard semantic conventions
const { SemanticAttributes } = require('@opentelemetry/semantic-conventions');

span.setAttributes({
  [SemanticAttributes.HTTP_METHOD]: 'POST',
  [SemanticAttributes.HTTP_URL]: '/api/users',
  [SemanticAttributes.HTTP_STATUS_CODE]: 200,
  [SemanticAttributes.DB_SYSTEM]: 'postgresql',
  [SemanticAttributes.DB_NAME]: 'mydb',
});

// ❌ BAD: Custom attributes without namespace
span.setAttributes({
  method: 'POST',
  url: '/api/users',
});
```

**2. Keep Span Names Concise**
```javascript
// ✅ GOOD: Generic operation name (use attributes for details)
const span = tracer.startSpan('GET /api/users/:id', {
  attributes: { 'user.id': userId },
});

// ❌ BAD: High cardinality span names
const span = tracer.startSpan(`GET /api/users/${userId}`);
```

**3. Always End Spans**
```javascript
// ✅ GOOD: Use try/finally to ensure span ends
const span = tracer.startSpan('operation');
try {
  await doWork();
} finally {
  span.end();
}

// ❌ BAD: Span might never end
const span = tracer.startSpan('operation');
await doWork();
span.end();
```

**4. Use Baggage for Cross-Cutting Concerns**
```javascript
const { propagation, baggageUtils } = require('@opentelemetry/api');

// Set baggage (propagates across service boundaries)
const baggage = propagation.createBaggage({
  'user.id': { value: '12345' },
  'request.id': { value: 'req-abc-123' },
});

context.with(propagation.setBaggage(context.active(), baggage), () => {
  // Baggage available in all child spans
  const userId = propagation.getBaggage(context.active())?.getEntry('user.id')?.value;
});
```

**5. Log Correlation**
```javascript
const { trace } = require('@opentelemetry/api');
const winston = require('winston');

const logger = winston.createLogger({
  format: winston.format.combine(
    winston.format((info) => {
      const span = trace.getActiveSpan();
      if (span) {
        const spanContext = span.spanContext();
        info.trace_id = spanContext.traceId;
        info.span_id = spanContext.spanId;
      }
      return info;
    })(),
    winston.format.json()
  ),
  transports: [new winston.transports.Console()],
});

logger.info('Order processed', { order_id: '123' });
// Output: { "message": "Order processed", "order_id": "123", "trace_id": "...", "span_id": "..." }
```

## Quality Criteria

- ✅ All HTTP requests automatically traced
- ✅ Database queries instrumented
- ✅ Custom business logic spans added
- ✅ Metrics exported every 60 seconds
- ✅ Sampling rate configured (not 100% in production)
- ✅ Trace context propagated across services
- ✅ Logs correlated with traces

## Backend Setup (Jaeger)

```bash
# Run Jaeger all-in-one (for development)
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest

# Access Jaeger UI: http://localhost:16686
```

## Troubleshooting

**Issue**: No traces appearing in Jaeger
**Solution**: Check exporter URL, ensure OTLP collector is running, verify network connectivity

**Issue**: High memory usage
**Solution**: Reduce sampling rate, use batch span processor with smaller queue size

**Issue**: Missing trace context between services
**Solution**: Ensure W3C Trace Context headers (traceparent, tracestate) are propagated

## Related Skills

- `kubernetes-specialist`: Deploying OTel Collector in K8s
- `aws-specialist`: AWS X-Ray integration
- `backend-dev`: Application instrumentation

## Tools

- Jaeger: Open-source tracing backend
- Zipkin: Distributed tracing system
- Grafana Tempo: High-scale tracing backend
- Prometheus: Metrics collection
- Grafana: Visualization

## MCP Tools

- `mcp__flow-nexus__execution_stream_subscribe` for real-time trace monitoring
- `mcp__flow-nexus__realtime_subscribe` for live metrics
- `mcp__memory-mcp__memory_store` for OTel patterns

## Success Metrics

- Trace coverage: ≥95% of requests
- Sampling rate: 5-10% (production)
- Metrics export interval: 60 seconds
- Span drop rate: <1%
- Log-trace correlation: 100%

---

**Skill Version**: 1.0.0
**Last Updated**: 2025-11-02
