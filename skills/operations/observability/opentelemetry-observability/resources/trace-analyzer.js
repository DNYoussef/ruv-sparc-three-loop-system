#!/usr/bin/env node
/**
 * OpenTelemetry Trace Analyzer
 * Analyze distributed traces for bottlenecks, errors, and performance issues
 */

const { trace, SpanStatusCode } = require('@opentelemetry/api');
const { NodeSDK } = require('@opentelemetry/sdk-node');
const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node');
const { Resource } = require('@opentelemetry/resources');
const { SemanticResourceAttributes } = require('@opentelemetry/semantic-conventions');
const { ConsoleSpanExporter } = require('@opentelemetry/sdk-trace-base');

class TraceAnalyzer {
  /**
   * Initialize OpenTelemetry SDK with auto-instrumentation
   */
  constructor(serviceName = 'trace-analyzer', serviceVersion = '1.0.0') {
    this.serviceName = serviceName;
    this.serviceVersion = serviceVersion;
    this.spans = [];

    // Initialize SDK
    this.sdk = new NodeSDK({
      resource: new Resource({
        [SemanticResourceAttributes.SERVICE_NAME]: serviceName,
        [SemanticResourceAttributes.SERVICE_VERSION]: serviceVersion,
        [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: 'analysis',
      }),
      traceExporter: new ConsoleSpanExporter(),
      instrumentations: [getNodeAutoInstrumentations()],
    });

    this.sdk.start();
    this.tracer = trace.getTracer(serviceName, serviceVersion);
  }

  /**
   * Create a trace with nested spans for analysis
   */
  async createSampleTrace() {
    const rootSpan = this.tracer.startSpan('api.processOrder', {
      attributes: {
        'order.id': 'ORD-12345',
        'order.priority': 'high',
        'customer.id': 'CUST-789',
      },
    });

    try {
      // Simulate multiple service calls
      await this._validateOrder(rootSpan);
      await this._checkInventory(rootSpan);
      await this._processPayment(rootSpan);
      await this._updateDatabase(rootSpan);

      rootSpan.setStatus({ code: SpanStatusCode.OK });
      return rootSpan;
    } catch (error) {
      rootSpan.recordException(error);
      rootSpan.setStatus({
        code: SpanStatusCode.ERROR,
        message: error.message,
      });
      throw error;
    } finally {
      rootSpan.end();
    }
  }

  /**
   * Validate order details
   */
  async _validateOrder(parentSpan) {
    const span = this.tracer.startSpan('validation.validateOrder', {
      parent: parentSpan,
      attributes: {
        'validation.type': 'order',
      },
    });

    try {
      await this._sleep(50); // Simulate validation
      span.addEvent('order_validated', {
        'validation.result': 'success',
        'validation.duration_ms': 50,
      });
      span.setStatus({ code: SpanStatusCode.OK });
    } finally {
      span.end();
    }
  }

  /**
   * Check inventory availability
   */
  async _checkInventory(parentSpan) {
    const span = this.tracer.startSpan('inventory.checkAvailability', {
      parent: parentSpan,
      attributes: {
        'inventory.warehouse': 'WAREHOUSE-US-EAST',
        'inventory.sku': 'PROD-456',
      },
    });

    try {
      // Simulate slow database query
      await this._sleep(250);
      span.addEvent('inventory_checked', {
        'inventory.available': true,
        'inventory.quantity': 42,
      });
      span.setStatus({ code: SpanStatusCode.OK });
    } finally {
      span.end();
    }
  }

  /**
   * Process payment
   */
  async _processPayment(parentSpan) {
    const span = this.tracer.startSpan('payment.processPayment', {
      parent: parentSpan,
      attributes: {
        'payment.method': 'credit_card',
        'payment.provider': 'stripe',
        'payment.amount': 99.99,
        'payment.currency': 'USD',
      },
    });

    try {
      // Simulate external API call
      await this._sleep(150);

      // 10% chance of payment failure for analysis
      if (Math.random() < 0.1) {
        throw new Error('Payment declined by provider');
      }

      span.addEvent('payment_processed', {
        'payment.transaction_id': 'TXN-ABC-123',
        'payment.status': 'authorized',
      });
      span.setStatus({ code: SpanStatusCode.OK });
    } catch (error) {
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

  /**
   * Update database
   */
  async _updateDatabase(parentSpan) {
    const span = this.tracer.startSpan('database.updateOrder', {
      parent: parentSpan,
      attributes: {
        'db.system': 'postgresql',
        'db.name': 'orders',
        'db.operation': 'UPDATE',
        'db.statement': 'UPDATE orders SET status = $1 WHERE id = $2',
      },
    });

    try {
      await this._sleep(100);
      span.addEvent('database_updated', {
        'db.rows_affected': 1,
      });
      span.setStatus({ code: SpanStatusCode.OK });
    } finally {
      span.end();
    }
  }

  /**
   * Analyze trace for bottlenecks
   */
  analyzeBottlenecks(spans) {
    console.log('\n=== TRACE BOTTLENECK ANALYSIS ===\n');

    const spanDurations = spans.map(span => ({
      name: span.name,
      duration: span.duration,
      percentage: (span.duration / spans[0].duration) * 100,
    })).sort((a, b) => b.duration - a.duration);

    console.log('Top Bottlenecks:');
    spanDurations.slice(0, 5).forEach((span, index) => {
      console.log(`${index + 1}. ${span.name}: ${span.duration.toFixed(2)}ms (${span.percentage.toFixed(1)}%)`);
    });

    // Identify critical path
    console.log('\nCritical Path Recommendation:');
    if (spanDurations[0].percentage > 50) {
      console.log(`⚠️  Optimize "${spanDurations[0].name}" - takes ${spanDurations[0].percentage.toFixed(1)}% of total time`);
    } else {
      console.log('✅ No single bottleneck dominates the trace');
    }
  }

  /**
   * Analyze errors in trace
   */
  analyzeErrors(spans) {
    console.log('\n=== ERROR ANALYSIS ===\n');

    const errors = spans.filter(span => span.status === SpanStatusCode.ERROR);

    if (errors.length === 0) {
      console.log('✅ No errors detected in trace');
      return;
    }

    console.log(`Found ${errors.length} error(s):`);
    errors.forEach(error => {
      console.log(`- ${error.name}: ${error.statusMessage}`);
      if (error.exception) {
        console.log(`  Exception: ${error.exception}`);
      }
    });
  }

  /**
   * Calculate trace statistics
   */
  calculateStatistics(spans) {
    console.log('\n=== TRACE STATISTICS ===\n');

    const totalDuration = spans[0].duration;
    const spanCount = spans.length;
    const avgSpanDuration = spans.reduce((sum, s) => sum + s.duration, 0) / spanCount;

    console.log(`Total Duration: ${totalDuration.toFixed(2)}ms`);
    console.log(`Span Count: ${spanCount}`);
    console.log(`Average Span Duration: ${avgSpanDuration.toFixed(2)}ms`);
    console.log(`Max Span Duration: ${Math.max(...spans.map(s => s.duration)).toFixed(2)}ms`);
    console.log(`Min Span Duration: ${Math.min(...spans.map(s => s.duration)).toFixed(2)}ms`);
  }

  /**
   * Sleep utility for simulations
   */
  async _sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Shutdown SDK gracefully
   */
  async shutdown() {
    await this.sdk.shutdown();
  }
}

// Main execution
async function main() {
  const analyzer = new TraceAnalyzer('order-service', '2.0.0');

  console.log('Creating sample distributed trace...\n');

  try {
    // Create multiple sample traces
    for (let i = 0; i < 3; i++) {
      console.log(`\n--- Trace ${i + 1} ---`);
      await analyzer.createSampleTrace();
      await analyzer._sleep(1000); // Wait between traces
    }

    // Simulate trace analysis (in production, fetch from backend)
    const mockSpans = [
      { name: 'api.processOrder', duration: 550, status: SpanStatusCode.OK },
      { name: 'validation.validateOrder', duration: 50, status: SpanStatusCode.OK },
      { name: 'inventory.checkAvailability', duration: 250, status: SpanStatusCode.OK },
      { name: 'payment.processPayment', duration: 150, status: SpanStatusCode.OK },
      { name: 'database.updateOrder', duration: 100, status: SpanStatusCode.OK },
    ];

    analyzer.analyzeBottlenecks(mockSpans);
    analyzer.analyzeErrors(mockSpans);
    analyzer.calculateStatistics(mockSpans);

    console.log('\n=== RECOMMENDATIONS ===\n');
    console.log('1. Add caching to inventory.checkAvailability (250ms)');
    console.log('2. Consider async payment processing');
    console.log('3. Add database connection pooling');
    console.log('4. Implement request timeouts (SLO: 500ms)');

  } catch (error) {
    console.error('Analysis failed:', error);
  } finally {
    await analyzer.shutdown();
    console.log('\nTrace analysis complete!');
  }
}

// Execute if run directly
if (require.main === module) {
  main().catch(console.error);
}

module.exports = TraceAnalyzer;
