#!/usr/bin/env node
/**
 * Distributed Tracing Example with OpenTelemetry
 * Demonstrates complete microservices instrumentation with context propagation
 *
 * Architecture:
 * API Gateway → Order Service → Payment Service → Database
 *                            ↓
 *                      Inventory Service
 */

const { trace, context, propagation, SpanStatusCode } = require('@opentelemetry/api');
const { NodeSDK } = require('@opentelemetry/sdk-node');
const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node');
const { Resource } = require('@opentelemetry/resources');
const { SemanticResourceAttributes, SemanticAttributes } = require('@opentelemetry/semantic-conventions');
const { ConsoleSpanExporter } = require('@opentelemetry/sdk-trace-base');
const { W3CTraceContextPropagator } = require('@opentelemetry/core');

// ============================================================================
// Service Configuration
// ============================================================================

const services = {
  'api-gateway': {
    name: 'api-gateway',
    version: '1.0.0',
    port: 8080,
  },
  'order-service': {
    name: 'order-service',
    version: '2.1.0',
    port: 8081,
  },
  'payment-service': {
    name: 'payment-service',
    version: '1.5.0',
    port: 8082,
  },
  'inventory-service': {
    name: 'inventory-service',
    version: '1.2.0',
    port: 8083,
  },
};

// ============================================================================
// Initialize OpenTelemetry SDK
// ============================================================================

function initializeTracing(serviceName, serviceVersion) {
  const sdk = new NodeSDK({
    resource: new Resource({
      [SemanticResourceAttributes.SERVICE_NAME]: serviceName,
      [SemanticResourceAttributes.SERVICE_VERSION]: serviceVersion,
      [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: 'production',
    }),
    traceExporter: new ConsoleSpanExporter(),
    instrumentations: [
      getNodeAutoInstrumentations({
        '@opentelemetry/instrumentation-http': {
          enabled: true,
          ignoreIncomingRequestHook: (req) => {
            // Ignore health check requests
            return req.url === '/health';
          },
        },
      }),
    ],
  });

  sdk.start();

  // Set W3C Trace Context propagator
  propagation.setGlobalPropagator(new W3CTraceContextPropagator());

  return {
    sdk,
    tracer: trace.getTracer(serviceName, serviceVersion),
  };
}

// ============================================================================
// API Gateway Service
// ============================================================================

class APIGateway {
  constructor() {
    const { sdk, tracer } = initializeTracing(
      services['api-gateway'].name,
      services['api-gateway'].version
    );
    this.sdk = sdk;
    this.tracer = tracer;
  }

  async processRequest(orderId, customerId, items) {
    // Root span for the entire distributed transaction
    const span = this.tracer.startSpan('api.processOrder', {
      attributes: {
        [SemanticAttributes.HTTP_METHOD]: 'POST',
        [SemanticAttributes.HTTP_ROUTE]: '/api/orders',
        [SemanticAttributes.HTTP_TARGET]: `/api/orders/${orderId}`,
        'order.id': orderId,
        'customer.id': customerId,
        'order.item_count': items.length,
      },
    });

    try {
      span.addEvent('request_received', {
        'request.timestamp': new Date().toISOString(),
      });

      // Call Order Service
      const orderResult = await this._callOrderService(span, orderId, customerId, items);

      span.setAttributes({
        'order.total_amount': orderResult.totalAmount,
        'order.status': orderResult.status,
      });

      span.setStatus({ code: SpanStatusCode.OK });
      return orderResult;
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

  async _callOrderService(parentSpan, orderId, customerId, items) {
    // Simulate HTTP call to Order Service with context propagation
    const span = this.tracer.startSpan('http.client.order_service', {
      parent: parentSpan,
      attributes: {
        [SemanticAttributes.HTTP_METHOD]: 'POST',
        [SemanticAttributes.HTTP_URL]: `http://order-service:8081/orders`,
        'peer.service': 'order-service',
      },
    });

    try {
      // Inject trace context into headers (W3C Trace Context)
      const headers = {};
      propagation.inject(context.active(), headers);

      console.log('📤 API Gateway → Order Service (trace context:', headers.traceparent, ')');

      // Simulate service call
      await this._sleep(50);

      const orderService = new OrderService();
      const result = await orderService.createOrder(headers, orderId, customerId, items);

      span.setAttributes({
        [SemanticAttributes.HTTP_STATUS_CODE]: 200,
      });
      span.setStatus({ code: SpanStatusCode.OK });

      return result;
    } finally {
      span.end();
    }
  }

  async _sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async shutdown() {
    await this.sdk.shutdown();
  }
}

// ============================================================================
// Order Service
// ============================================================================

class OrderService {
  constructor() {
    const { sdk, tracer } = initializeTracing(
      services['order-service'].name,
      services['order-service'].version
    );
    this.sdk = sdk;
    this.tracer = tracer;
  }

  async createOrder(incomingHeaders, orderId, customerId, items) {
    // Extract trace context from incoming headers
    const extractedContext = propagation.extract(context.active(), incomingHeaders);

    return context.with(extractedContext, async () => {
      const span = this.tracer.startSpan('order.createOrder', {
        attributes: {
          'order.id': orderId,
          'customer.id': customerId,
        },
      });

      try {
        console.log('📥 Order Service received request (continuing trace)');

        // Validate order
        await this._validateOrder(span, orderId, items);

        // Check inventory (parallel call)
        const inventoryCheck = await this._checkInventory(span, items);

        // Process payment
        const paymentResult = await this._processPayment(span, orderId, customerId, inventoryCheck.totalAmount);

        // Save to database
        await this._saveToDatabase(span, orderId, paymentResult);

        span.setStatus({ code: SpanStatusCode.OK });

        return {
          orderId,
          status: 'confirmed',
          totalAmount: inventoryCheck.totalAmount,
          transactionId: paymentResult.transactionId,
        };
      } catch (error) {
        span.recordException(error);
        span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
        throw error;
      } finally {
        span.end();
      }
    });
  }

  async _validateOrder(parentSpan, orderId, items) {
    const span = this.tracer.startSpan('order.validateOrder', { parent: parentSpan });
    await this._sleep(30);
    span.addEvent('validation_completed', { 'validation.result': 'success' });
    span.end();
  }

  async _checkInventory(parentSpan, items) {
    const span = this.tracer.startSpan('http.client.inventory_service', {
      parent: parentSpan,
      attributes: {
        [SemanticAttributes.HTTP_METHOD]: 'POST',
        [SemanticAttributes.HTTP_URL]: 'http://inventory-service:8083/check',
        'peer.service': 'inventory-service',
      },
    });

    try {
      const headers = {};
      propagation.inject(context.active(), headers);

      console.log('📤 Order Service → Inventory Service');

      await this._sleep(100);

      const totalAmount = items.reduce((sum, item) => sum + item.price * item.quantity, 0);

      span.setAttributes({
        [SemanticAttributes.HTTP_STATUS_CODE]: 200,
        'inventory.total_amount': totalAmount,
      });
      span.setStatus({ code: SpanStatusCode.OK });

      return { totalAmount, available: true };
    } finally {
      span.end();
    }
  }

  async _processPayment(parentSpan, orderId, customerId, amount) {
    const span = this.tracer.startSpan('http.client.payment_service', {
      parent: parentSpan,
      attributes: {
        [SemanticAttributes.HTTP_METHOD]: 'POST',
        [SemanticAttributes.HTTP_URL]: 'http://payment-service:8082/charge',
        'peer.service': 'payment-service',
        'payment.amount': amount,
      },
    });

    try {
      const headers = {};
      propagation.inject(context.active(), headers);

      console.log('📤 Order Service → Payment Service');

      const paymentService = new PaymentService();
      const result = await paymentService.processPayment(headers, orderId, amount);

      span.setStatus({ code: SpanStatusCode.OK });
      return result;
    } finally {
      span.end();
    }
  }

  async _saveToDatabase(parentSpan, orderId, paymentResult) {
    const span = this.tracer.startSpan('db.saveOrder', {
      parent: parentSpan,
      attributes: {
        [SemanticAttributes.DB_SYSTEM]: 'postgresql',
        [SemanticAttributes.DB_NAME]: 'orders',
        [SemanticAttributes.DB_OPERATION]: 'INSERT',
        [SemanticAttributes.DB_STATEMENT]: 'INSERT INTO orders (id, status, transaction_id) VALUES ($1, $2, $3)',
      },
    });

    await this._sleep(75);
    span.addEvent('order_persisted', { 'db.row_id': orderId });
    span.end();
  }

  async _sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ============================================================================
// Payment Service
// ============================================================================

class PaymentService {
  constructor() {
    const { sdk, tracer } = initializeTracing(
      services['payment-service'].name,
      services['payment-service'].version
    );
    this.sdk = sdk;
    this.tracer = tracer;
  }

  async processPayment(incomingHeaders, orderId, amount) {
    const extractedContext = propagation.extract(context.active(), incomingHeaders);

    return context.with(extractedContext, async () => {
      const span = this.tracer.startSpan('payment.processPayment', {
        attributes: {
          'payment.order_id': orderId,
          'payment.amount': amount,
          'payment.currency': 'USD',
          'payment.method': 'credit_card',
        },
      });

      try {
        console.log('📥 Payment Service received request');

        // Simulate payment gateway call
        await this._callPaymentGateway(span, amount);

        const transactionId = `TXN-${Date.now()}`;

        span.addEvent('payment_authorized', {
          'payment.transaction_id': transactionId,
          'payment.status': 'authorized',
        });

        span.setStatus({ code: SpanStatusCode.OK });

        return { transactionId, status: 'authorized' };
      } catch (error) {
        span.recordException(error);
        span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
        throw error;
      } finally {
        span.end();
      }
    });
  }

  async _callPaymentGateway(parentSpan, amount) {
    const span = this.tracer.startSpan('http.client.stripe_api', {
      parent: parentSpan,
      attributes: {
        [SemanticAttributes.HTTP_METHOD]: 'POST',
        [SemanticAttributes.HTTP_URL]: 'https://api.stripe.com/v1/charges',
        'peer.service': 'stripe',
      },
    });

    await this._sleep(120);
    span.setStatus({ code: SpanStatusCode.OK });
    span.end();
  }

  async _sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ============================================================================
// Main Execution
// ============================================================================

async function main() {
  console.log('============================================');
  console.log('  Distributed Tracing Example');
  console.log('  Microservices: API Gateway → Order → Payment → Inventory');
  console.log('============================================\n');

  const gateway = new APIGateway();

  try {
    // Simulate order request
    const orderId = 'ORD-12345';
    const customerId = 'CUST-789';
    const items = [
      { sku: 'PROD-001', price: 29.99, quantity: 2 },
      { sku: 'PROD-002', price: 49.99, quantity: 1 },
    ];

    console.log(`🛒 Processing order: ${orderId}\n`);

    const result = await gateway.processRequest(orderId, customerId, items);

    console.log('\n✅ Order completed successfully!');
    console.log('Result:', JSON.stringify(result, null, 2));

    console.log('\n📊 Check the exported spans above to see distributed trace!');
  } catch (error) {
    console.error('❌ Order failed:', error.message);
  } finally {
    await gateway.shutdown();
  }
}

// Run the example
main().catch(console.error);
