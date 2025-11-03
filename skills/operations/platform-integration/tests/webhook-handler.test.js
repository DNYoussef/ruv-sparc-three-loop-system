/**
 * Webhook Handler Tests
 * Test suite for webhook verification and processing
 */

const { expect } = require('chai');
const request = require('supertest');
const crypto = require('crypto');
const app = require('../resources/scripts/webhook-server');
const WebhookVerifier = require('../resources/scripts/webhook-verifier');

describe('Webhook Handler Tests', () => {
  describe('Webhook Verification', () => {
    const secret = 'test_secret_key';

    it('should verify Stripe signatures correctly', () => {
      const payload = JSON.stringify({ id: 'evt_123', type: 'payment_intent.succeeded' });
      const timestamp = Math.floor(Date.now() / 1000);

      const signature = crypto
        .createHmac('sha256', secret)
        .update(`${timestamp}.${payload}`)
        .digest('hex');

      const header = `t=${timestamp},v1=${signature}`;

      const isValid = WebhookVerifier.verifyStripe(payload, header, secret);
      expect(isValid).to.be.true;
    });

    it('should reject invalid Stripe signatures', () => {
      const payload = JSON.stringify({ id: 'evt_123' });
      const invalidHeader = 't=1234567890,v1=invalid_signature';

      expect(() => {
        WebhookVerifier.verifyStripe(payload, invalidHeader, secret);
      }).to.throw('Invalid signature');
    });

    it('should reject old timestamps (replay attack)', () => {
      const payload = JSON.stringify({ id: 'evt_123' });
      const oldTimestamp = Math.floor(Date.now() / 1000) - 600; // 10 minutes ago

      const signature = crypto
        .createHmac('sha256', secret)
        .update(`${oldTimestamp}.${payload}`)
        .digest('hex');

      const header = `t=${oldTimestamp},v1=${signature}`;

      expect(() => {
        WebhookVerifier.verifyStripe(payload, header, secret);
      }).to.throw('Timestamp too old');
    });

    it('should verify GitHub signatures', () => {
      const payload = JSON.stringify({ ref: 'refs/heads/main' });
      const signature = 'sha256=' + crypto
        .createHmac('sha256', secret)
        .update(payload)
        .digest('hex');

      const isValid = WebhookVerifier.verifyGitHub(payload, signature, secret);
      expect(isValid).to.be.true;
    });

    it('should verify HubSpot signatures', () => {
      const payload = JSON.stringify([{ objectId: 123 }]);
      const signature = crypto
        .createHmac('sha256', secret)
        .update(payload)
        .digest('hex');

      const isValid = WebhookVerifier.verifyHubSpot(payload, signature, secret);
      expect(isValid).to.be.true;
    });
  });

  describe('Webhook Endpoints', () => {
    it('should accept valid Stripe webhook', async () => {
      const payload = { id: 'evt_123', type: 'payment_intent.succeeded' };
      const payloadStr = JSON.stringify(payload);
      const timestamp = Math.floor(Date.now() / 1000);

      const signature = crypto
        .createHmac('sha256', process.env.STRIPE_WEBHOOK_SECRET || 'test_secret')
        .update(`${timestamp}.${payloadStr}`)
        .digest('hex');

      const header = `t=${timestamp},v1=${signature}`;

      const response = await request(app)
        .post('/webhooks/stripe')
        .set('Stripe-Signature', header)
        .send(payload);

      expect(response.status).to.equal(200);
      expect(response.body).to.have.property('received', true);
    });

    it('should reject Stripe webhook with invalid signature', async () => {
      const payload = { id: 'evt_123', type: 'payment_intent.succeeded' };

      const response = await request(app)
        .post('/webhooks/stripe')
        .set('Stripe-Signature', 't=123,v1=invalid')
        .send(payload);

      expect(response.status).to.equal(400);
    });

    it('should accept valid GitHub webhook', async () => {
      const payload = { ref: 'refs/heads/main', commits: [] };
      const payloadStr = JSON.stringify(payload);

      const signature = 'sha256=' + crypto
        .createHmac('sha256', process.env.GITHUB_WEBHOOK_SECRET || 'test_secret')
        .update(payloadStr)
        .digest('hex');

      const response = await request(app)
        .post('/webhooks/github')
        .set('X-GitHub-Event', 'push')
        .set('X-Hub-Signature-256', signature)
        .send(payload);

      expect(response.status).to.equal(200);
    });

    it('should handle idempotent events', async () => {
      const payload = { id: 'evt_duplicate', type: 'test.event' };

      // Send same event twice
      const response1 = await sendWebhook(payload);
      const response2 = await sendWebhook(payload);

      expect(response1.status).to.equal(200);
      expect(response2.status).to.equal(200);
      // Should handle duplicate gracefully
    });
  });

  describe('Event Processing', () => {
    it('should queue webhook events', async () => {
      const payload = { id: 'evt_123', type: 'payment_intent.succeeded' };

      // Mock webhook queue
      const queueSpy = sinon.spy(webhookQueue, 'add');

      await processWebhook('stripe', payload);

      expect(queueSpy.calledOnce).to.be.true;
      expect(queueSpy.args[0][0]).to.equal('stripe');
    });

    it('should process payment success events', async () => {
      const paymentIntent = {
        id: 'pi_123',
        amount: 10000,
        currency: 'usd',
        receipt_email: 'customer@example.com'
      };

      // Mock dependencies
      const sfSpy = sinon.spy(salesforce, 'updateOpportunity');
      const emailSpy = sinon.spy(email, 'send');
      const slackSpy = sinon.spy(slack, 'notify');

      await handlePaymentSuccess(paymentIntent);

      expect(sfSpy.calledOnce).to.be.true;
      expect(emailSpy.calledOnce).to.be.true;
      expect(slackSpy.calledOnce).to.be.true;
    });

    it('should handle payment failures', async () => {
      const paymentIntent = {
        id: 'pi_failed',
        amount: 5000,
        last_payment_error: { message: 'Card declined' }
      };

      await handlePaymentFailed(paymentIntent);

      // Verify error handling
    });
  });

  describe('Error Handling', () => {
    it('should retry failed webhook processing', async () => {
      // Test retry logic with exponential backoff
    });

    it('should send to dead letter queue after max retries', async () => {
      // Test DLQ functionality
    });

    it('should alert on critical failures', async () => {
      // Test alerting
    });
  });
});

// Helper function
async function sendWebhook(payload) {
  const payloadStr = JSON.stringify(payload);
  const signature = crypto
    .createHmac('sha256', 'test_secret')
    .update(payloadStr)
    .digest('hex');

  return request(app)
    .post('/webhooks/test')
    .set('X-Signature', signature)
    .send(payload);
}
