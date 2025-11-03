#!/usr/bin/env node
/**
 * Unit tests for OpenTelemetry Trace Analyzer
 */

const assert = require('assert');
const { describe, it, before, after } = require('node:test');
const { trace, SpanStatusCode } = require('@opentelemetry/api');
const { NodeSDK } = require('@opentelemetry/sdk-node');
const { Resource } = require('@opentelemetry/resources');
const { SemanticResourceAttributes } = require('@opentelemetry/semantic-conventions');
const { InMemorySpanExporter } = require('@opentelemetry/sdk-trace-base');

describe('OpenTelemetry Trace Analyzer Tests', () => {
  let sdk;
  let tracer;
  let spanExporter;

  before(async () => {
    // Setup in-memory exporter for testing
    spanExporter = new InMemorySpanExporter();

    sdk = new NodeSDK({
      resource: new Resource({
        [SemanticResourceAttributes.SERVICE_NAME]: 'test-service',
      }),
      traceExporter: spanExporter,
    });

    await sdk.start();
    tracer = trace.getTracer('test-tracer', '1.0.0');
  });

  after(async () => {
    await sdk.shutdown();
  });

  it('should create a basic span', () => {
    const span = tracer.startSpan('test.operation');
    span.end();

    const exportedSpans = spanExporter.getFinishedSpans();
    assert.strictEqual(exportedSpans.length > 0, true, 'Span should be exported');
    assert.strictEqual(exportedSpans[exportedSpans.length - 1].name, 'test.operation');
  });

  it('should create nested spans (parent-child)', () => {
    const parentSpan = tracer.startSpan('parent.operation');

    const childSpan = tracer.startSpan('child.operation', {
      parent: parentSpan,
    });

    childSpan.end();
    parentSpan.end();

    const exportedSpans = spanExporter.getFinishedSpans();
    const parent = exportedSpans.find(s => s.name === 'parent.operation');
    const child = exportedSpans.find(s => s.name === 'child.operation');

    assert.ok(parent, 'Parent span should exist');
    assert.ok(child, 'Child span should exist');
    assert.strictEqual(
      child.parentSpanId,
      parent.spanContext().spanId,
      'Child should reference parent'
    );
  });

  it('should add attributes to spans', () => {
    const span = tracer.startSpan('operation.with.attributes');

    span.setAttributes({
      'http.method': 'POST',
      'http.url': '/api/users',
      'http.status_code': 201,
      'user.id': 'user-123',
    });

    span.end();

    const exportedSpans = spanExporter.getFinishedSpans();
    const testSpan = exportedSpans.find(s => s.name === 'operation.with.attributes');

    assert.strictEqual(testSpan.attributes['http.method'], 'POST');
    assert.strictEqual(testSpan.attributes['http.status_code'], 201);
  });

  it('should record span events', () => {
    const span = tracer.startSpan('operation.with.events');

    span.addEvent('validation_started', {
      'validation.type': 'schema',
    });

    span.addEvent('validation_completed', {
      'validation.result': 'success',
      'validation.duration_ms': 50,
    });

    span.end();

    const exportedSpans = spanExporter.getFinishedSpans();
    const testSpan = exportedSpans.find(s => s.name === 'operation.with.events');

    assert.strictEqual(testSpan.events.length, 2);
    assert.strictEqual(testSpan.events[0].name, 'validation_started');
    assert.strictEqual(testSpan.events[1].name, 'validation_completed');
  });

  it('should record exceptions in spans', () => {
    const span = tracer.startSpan('operation.with.error');

    try {
      throw new Error('Simulated error for testing');
    } catch (error) {
      span.recordException(error);
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: error.message,
      });
    } finally {
      span.end();
    }

    const exportedSpans = spanExporter.getFinishedSpans();
    const testSpan = exportedSpans.find(s => s.name === 'operation.with.error');

    assert.strictEqual(testSpan.status.code, SpanStatusCode.ERROR);
    assert.ok(testSpan.events.find(e => e.name === 'exception'));
  });

  it('should track span status (OK, ERROR)', () => {
    // Test OK status
    const okSpan = tracer.startSpan('operation.ok');
    okSpan.setStatus({ code: SpanStatusCode.OK });
    okSpan.end();

    // Test ERROR status
    const errorSpan = tracer.startSpan('operation.error');
    errorSpan.setStatus({
      code: SpanStatusCode.ERROR,
      message: 'Something went wrong',
    });
    errorSpan.end();

    const exportedSpans = spanExporter.getFinishedSpans();
    const ok = exportedSpans.find(s => s.name === 'operation.ok');
    const error = exportedSpans.find(s => s.name === 'operation.error');

    assert.strictEqual(ok.status.code, SpanStatusCode.OK);
    assert.strictEqual(error.status.code, SpanStatusCode.ERROR);
  });

  it('should use semantic conventions for HTTP', () => {
    const { SemanticAttributes } = require('@opentelemetry/semantic-conventions');

    const span = tracer.startSpan('http.request');

    span.setAttributes({
      [SemanticAttributes.HTTP_METHOD]: 'GET',
      [SemanticAttributes.HTTP_URL]: 'https://api.example.com/users',
      [SemanticAttributes.HTTP_STATUS_CODE]: 200,
      [SemanticAttributes.HTTP_ROUTE]: '/users',
    });

    span.end();

    const exportedSpans = spanExporter.getFinishedSpans();
    const testSpan = exportedSpans.find(s => s.name === 'http.request');

    assert.strictEqual(testSpan.attributes[SemanticAttributes.HTTP_METHOD], 'GET');
    assert.strictEqual(testSpan.attributes[SemanticAttributes.HTTP_STATUS_CODE], 200);
  });

  it('should measure span duration', () => {
    const span = tracer.startSpan('timed.operation');

    // Simulate work
    const start = Date.now();
    while (Date.now() - start < 10) {
      // Busy wait 10ms
    }

    span.end();

    const exportedSpans = spanExporter.getFinishedSpans();
    const testSpan = exportedSpans.find(s => s.name === 'timed.operation');

    const durationMs = (testSpan.endTime[0] - testSpan.startTime[0]) * 1000;
    assert.ok(durationMs >= 10, 'Span duration should be at least 10ms');
  });
});

console.log('Running OpenTelemetry Trace Analyzer tests...');
