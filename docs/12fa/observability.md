# Observability with Structured Logging

## Overview

This guide explains how to use structured JSON logging with OpenTelemetry compatibility for comprehensive observability in Claude Flow hook operations.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Structured Logging](#structured-logging)
3. [Correlation IDs](#correlation-ids)
4. [OpenTelemetry Integration](#opentelemetry-integration)
5. [Querying Logs](#querying-logs)
6. [Log Aggregation](#log-aggregation)
7. [Best Practices](#best-practices)

---

## Getting Started

### Installation

All observability components are included in the hooks directory:

```
hooks/12fa/
  ├── structured-logger.js         # Core logging utility
  ├── correlation-id-manager.js    # Correlation tracking
  └── opentelemetry-adapter.js     # OpenTelemetry integration
```

### Configuration

Configure logging in `config/logging-config.json`:

```json
{
  "production": {
    "level": "INFO",
    "outputFormat": "json",
    "outputDestination": "file",
    "outputFile": "logs/hooks.log",
    "includeStack": false
  }
}
```

---

## Structured Logging

### Basic Usage

```javascript
const { getLogger } = require('./hooks/12fa/structured-logger');

const logger = getLogger();

// Simple log
logger.info('Operation started', {
  trace_id: 'trace-123',
  agent_id: 'researcher-42',
  operation: 'research_analysis'
});

// Log with metadata
logger.info('Analysis complete', {
  trace_id: 'trace-123',
  agent_id: 'researcher-42',
  operation: 'research_analysis',
  duration_ms: 1523,
  metadata: {
    documents_processed: 15,
    key_findings: 8
  }
});
```

### Log Levels

Four log levels in order of severity:

```javascript
logger.debug('Detailed diagnostic info');  // DEBUG
logger.info('General information');        // INFO
logger.warn('Warning messages');           // WARN
logger.error('Error conditions');          // ERROR
```

### Structured Log Format

Every log entry follows this structure:

```typescript
interface StructuredLog {
  timestamp: string;        // ISO 8601
  level: string;            // DEBUG, INFO, WARN, ERROR
  message: string;          // Human-readable message
  trace_id: string;         // Correlation ID
  span_id: string;          // Span identifier
  parent_span_id?: string;  // Parent span (if child)
  agent_id: string;         // Agent identifier
  agent_type: string;       // Agent type
  operation: string;        // Operation name
  duration_ms?: number;     // Duration (if applicable)
  status: string;           // success | error
  metadata: object;         // Additional context
  error?: {                 // Error details (if error)
    message: string;
    stack: string;
  };
}
```

### Example Output

```json
{
  "timestamp": "2025-11-01T14:30:00.000Z",
  "level": "INFO",
  "message": "Agent task completed",
  "trace_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "span_id": "span-1730472600000-abc123def",
  "agent_id": "researcher-42",
  "agent_type": "researcher",
  "operation": "task_execution",
  "duration_ms": 1523,
  "status": "success",
  "metadata": {
    "tokens_used": 8500,
    "cost_usd": 0.085,
    "files_created": 3
  }
}
```

---

## Correlation IDs

### Purpose

Correlation IDs enable tracking requests across multiple agents and operations:

- **Trace entire workflows** from start to finish
- **Debug distributed operations** across agent boundaries
- **Analyze performance** of multi-agent collaborations
- **Correlate errors** with upstream causes

### Generating IDs

```javascript
const { getManager } = require('./hooks/12fa/correlation-id-manager');

const manager = getManager();

// Generate new correlation ID
const traceId = manager.generate('uuid_v4');
// Result: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

// Short format
const shortId = manager.generate('short');
// Result: "abc123def456"

// Prefixed format
const prefixedId = manager.generate('prefixed');
// Result: "trace-abc123def456"
```

### Propagating Across Agents

```javascript
// Parent agent creates trace
const parentContext = 'parent-agent-task';
const traceId = manager.getOrCreate(parentContext);

// Child agent inherits trace
const childContext = 'child-agent-task';
manager.propagate(parentContext, childContext);

// Both now share the same trace ID
const childTraceId = manager.get(childContext);
console.log(traceId === childTraceId); // true
```

### Child Spans

```javascript
// Create hierarchical span structure
const parentSpanId = 'span-001';
const { spanId, parentSpanId: parent } = manager.createChildSpan(parentSpanId);

console.log(spanId);        // "span-001.abc123def"
console.log(parent);        // "span-001"
```

### Persistence

Correlation IDs persist across sessions:

```javascript
// Automatically saved to .swarm/correlation.db
manager.set('persistent-context', 'trace-123');

// Survives process restart
const manager2 = getManager();
const id = manager2.get('persistent-context');
console.log(id); // "trace-123"
```

---

## OpenTelemetry Integration

### Overview

OpenTelemetry provides distributed tracing with spans representing operations:

```
[Parent Span: Task Execution]
  ├── [Child Span: Research Phase]
  ├── [Child Span: Implementation Phase]
  └── [Child Span: Testing Phase]
```

### Creating Spans

```javascript
const { getAdapter } = require('./hooks/12fa/opentelemetry-adapter');

const adapter = getAdapter();

// Start root span
const span = adapter.startSpan('task-execution', {
  kind: SpanKind.INTERNAL,
  attributes: {
    'agent.id': 'coder-42',
    'agent.type': 'coder',
    'task.type': 'implementation'
  }
});

// Do work...

// End span
adapter.endSpan(span);
```

### Child Spans

```javascript
// Create parent span
const parentSpan = adapter.startSpan('full-workflow');

// Create child spans
const context = adapter.getContext();
const childSpan1 = context.createChildSpan(parentSpan, 'research-phase');
const childSpan2 = context.createChildSpan(parentSpan, 'implementation-phase');

// Work...

adapter.endSpan(childSpan1);
adapter.endSpan(childSpan2);
adapter.endSpan(parentSpan);
```

### Automatic Tracing

```javascript
// Wrap async functions for automatic span management
await adapter.trace('database-query', async (span) => {
  span.setAttribute('query.type', 'SELECT');
  span.setAttribute('query.table', 'users');

  const result = await db.query('SELECT * FROM users');

  span.setAttribute('query.rows', result.length);
  return result;
});
```

### Error Handling

```javascript
const span = adapter.startSpan('risky-operation');

try {
  await performRiskyOperation();
  span.setStatus(SpanStatus.OK);
} catch (error) {
  span.recordException(error);
  // Automatically sets status to ERROR and captures stack
} finally {
  adapter.endSpan(span);
}
```

### Trace Context Propagation

```javascript
// Inject trace context into HTTP headers
const span = adapter.startSpan('http-request');
const headers = context.injectIntoHeaders(span);

// Headers now contain W3C Trace Context:
// {
//   "traceparent": "00-a1b2c3d4...-span123-01"
// }

await fetch('https://api.example.com', { headers });
```

---

## Querying Logs

### Command Line (jq)

```bash
# Get all ERROR logs
cat logs/hooks.log | jq 'select(.level == "ERROR")'

# Find logs for specific trace
cat logs/hooks.log | jq 'select(.trace_id == "trace-123")'

# Get logs from specific agent
cat logs/hooks.log | jq 'select(.agent_id == "researcher-42")'

# Find slow operations (>1s)
cat logs/hooks.log | jq 'select(.duration_ms > 1000)'

# Extract operation statistics
cat logs/hooks.log | jq -s 'group_by(.operation) | map({operation: .[0].operation, count: length, avg_duration: (map(.duration_ms) | add / length)})'
```

### Node.js Queries

```javascript
const fs = require('fs');
const readline = require('readline');

async function queryLogs(filter) {
  const fileStream = fs.createReadStream('logs/hooks.log');
  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity
  });

  const results = [];
  for await (const line of rl) {
    const log = JSON.parse(line);
    if (filter(log)) {
      results.push(log);
    }
  }

  return results;
}

// Find all logs for a trace
const traceLogs = await queryLogs(log => log.trace_id === 'trace-123');

// Find errors in last hour
const recentErrors = await queryLogs(log => {
  return log.level === 'ERROR' &&
         new Date(log.timestamp) > new Date(Date.now() - 3600000);
});
```

### Trace Analysis

```javascript
// Reconstruct complete trace
async function reconstructTrace(traceId) {
  const logs = await queryLogs(log => log.trace_id === traceId);

  // Sort chronologically
  logs.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

  // Build span tree
  const spans = {};
  const rootSpans = [];

  logs.forEach(log => {
    if (!spans[log.span_id]) {
      spans[log.span_id] = { ...log, children: [] };
    }

    if (log.parent_span_id) {
      if (!spans[log.parent_span_id]) {
        spans[log.parent_span_id] = { children: [] };
      }
      spans[log.parent_span_id].children.push(spans[log.span_id]);
    } else {
      rootSpans.push(spans[log.span_id]);
    }
  });

  return rootSpans;
}
```

---

## Log Aggregation

### ELK Stack (Elasticsearch, Logstash, Kibana)

**1. Configure Filebeat:**

```yaml
# filebeat.yml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /path/to/logs/hooks.log
    json.keys_under_root: true
    json.add_error_key: true

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "claude-flow-logs-%{+yyyy.MM.dd}"
```

**2. Query in Kibana:**

```
# Find all errors for specific agent
level: "ERROR" AND agent_id: "coder-42"

# Trace-based search
trace_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

# Performance analysis
operation: "task_execution" AND duration_ms: >1000
```

### Splunk

**1. Configure log input:**

```bash
# Add to inputs.conf
[monitor:///path/to/logs/hooks.log]
sourcetype = _json
index = claude_flow
```

**2. Query in Splunk:**

```spl
# Error rate over time
index=claude_flow level=ERROR
| timechart count by agent_type

# Trace reconstruction
index=claude_flow trace_id="trace-123"
| sort timestamp
| table timestamp level agent_id operation duration_ms

# Performance percentiles
index=claude_flow operation="task_execution"
| stats perc50(duration_ms) perc95(duration_ms) perc99(duration_ms)
```

### Datadog

**1. Configure Datadog Agent:**

```yaml
# datadog.yaml
logs:
  - type: file
    path: /path/to/logs/hooks.log
    service: claude-flow
    source: nodejs
    sourcecategory: hooks
```

**2. Query in Datadog:**

```
# Service map
service:claude-flow

# Trace view
trace_id:a1b2c3d4-e5f6-7890-abcd-ef1234567890

# APM queries
@operation:task_execution @duration_ms:>1000
```

---

## Best Practices

### 1. Always Use Correlation IDs

```javascript
// ❌ Bad: No correlation
logger.info('Task started');

// ✅ Good: With correlation
const traceId = correlationManager.getOrCreate('task-123');
logger.info('Task started', {
  trace_id: traceId,
  agent_id: 'coder-42',
  operation: 'task_start'
});
```

### 2. Include Relevant Context

```javascript
// ❌ Bad: Minimal context
logger.info('Query completed');

// ✅ Good: Rich context
logger.info('Database query completed', {
  trace_id: traceId,
  agent_id: 'coder-42',
  operation: 'db_query',
  duration_ms: 145,
  metadata: {
    query_type: 'SELECT',
    table: 'users',
    rows_returned: 25,
    cache_hit: false
  }
});
```

### 3. Use Appropriate Log Levels

```javascript
// DEBUG: Detailed diagnostic information
logger.debug('Cache lookup', { key: 'user:123', hit: true });

// INFO: General informational events
logger.info('Task completed successfully', { duration_ms: 1500 });

// WARN: Warning conditions (recoverable)
logger.warn('Retry attempt', { attempt: 2, max_attempts: 3 });

// ERROR: Error conditions requiring attention
logger.error('Task failed', { error: err, trace_id: traceId });
```

### 4. Measure Performance

```javascript
const startTime = Date.now();

// Do work...

logger.info('Operation completed', {
  trace_id: traceId,
  operation: 'expensive_operation',
  duration_ms: Date.now() - startTime
});
```

### 5. Create Child Loggers for Context

```javascript
// Create child logger with inherited context
const logger = getLogger().child({
  agent_id: 'coder-42',
  agent_type: 'coder',
  trace_id: traceId
});

// All logs now include inherited context
logger.info('Starting implementation');
logger.info('Code generated');
logger.info('Tests passing');
```

### 6. Handle Errors Properly

```javascript
try {
  await riskyOperation();
} catch (error) {
  logger.error('Operation failed', {
    trace_id: traceId,
    operation: 'risky_operation',
    error: error,  // Automatically extracts message and stack
    metadata: {
      recovery_attempted: true,
      fallback_used: 'default_behavior'
    }
  });

  // Re-throw or handle
  throw error;
}
```

### 7. Sanitize Sensitive Data

```javascript
// ❌ Bad: Logging sensitive data
logger.info('User authenticated', {
  username: user.username,
  password: user.password,  // DON'T DO THIS
  api_key: user.apiKey      // OR THIS
});

// ✅ Good: Sanitized logging
logger.info('User authenticated', {
  username: user.username,
  user_id: user.id,
  auth_method: 'password'
  // Sensitive data omitted
});
```

---

## Performance Considerations

### Logging Overhead

Structured logging adds approximately **<2ms overhead per log entry**:

- JSON serialization: ~0.5ms
- File I/O: ~1ms
- Correlation lookup: ~0.1ms

### Optimization Tips

1. **Use appropriate log levels** - Don't log DEBUG in production
2. **Batch file writes** - Configure output buffering
3. **Async logging** - Use non-blocking I/O where possible
4. **Sampling** - For high-volume operations, sample logs

```javascript
// Example: Sample 10% of high-volume logs
if (Math.random() < 0.1) {
  logger.debug('High-volume operation', { ... });
}
```

---

## Troubleshooting

### Logs Not Appearing

1. Check log level configuration
2. Verify output destination
3. Ensure log directory exists and is writable

```javascript
const logger = getLogger();
logger.debug('Test log');  // May not appear if level is INFO
```

### Correlation IDs Not Propagating

1. Verify manager is shared (singleton pattern)
2. Check context key consistency
3. Ensure persistence path is writable

```javascript
// Always use same manager instance
const manager = getManager();  // Singleton
```

### Large Log Files

Implement log rotation:

```bash
# Using logrotate (Linux)
/path/to/logs/hooks.log {
  daily
  rotate 7
  compress
  missingok
  notifempty
}
```

---

## Examples

### Complete Hook Integration

```javascript
const { getLogger } = require('./hooks/12fa/structured-logger');
const { getManager } = require('./hooks/12fa/correlation-id-manager');
const { getAdapter } = require('./hooks/12fa/opentelemetry-adapter');

// Pre-task hook
async function preTaskHook(description) {
  const logger = getLogger();
  const correlationManager = getManager();
  const otelAdapter = getAdapter();

  // Generate correlation ID
  const traceId = correlationManager.generate();
  const contextKey = `task-${Date.now()}`;
  correlationManager.set(contextKey, traceId);

  // Start span
  const span = otelAdapter.startSpan('task-execution', {
    traceId,
    attributes: {
      'task.description': description,
      'task.start_time': new Date().toISOString()
    }
  });

  // Log task start
  logger.info('Task started', {
    trace_id: traceId,
    span_id: span.spanId,
    operation: 'task_start',
    metadata: { description }
  });

  return { traceId, span, contextKey };
}

// Post-task hook
async function postTaskHook(context, result) {
  const logger = getLogger();
  const otelAdapter = getAdapter();

  // End span
  otelAdapter.endSpan(context.span);

  // Log completion
  logger.info('Task completed', {
    trace_id: context.traceId,
    span_id: context.span.spanId,
    operation: 'task_complete',
    duration_ms: context.span.getDuration(),
    status: 'success',
    metadata: { result }
  });
}
```

---

## Resources

- [OpenTelemetry Specification](https://opentelemetry.io/docs/specs/otel/)
- [W3C Trace Context](https://www.w3.org/TR/trace-context/)
- [Structured Logging Best Practices](https://www.structlog.org/)

---

**Questions or Issues?**

File issues at: https://github.com/ruvnet/claude-flow/issues
