# Example 2: Multi-Platform Webhook Automation

## Overview

This example demonstrates enterprise webhook automation across multiple platforms:
- Stripe payment webhooks → Salesforce & Email
- GitHub code push webhooks → Slack notifications
- HubSpot contact updates → CRM sync
- Webhook signature verification
- Event routing and processing
- Error handling and retry logic

## Architecture

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Stripe  │     │ GitHub  │     │HubSpot  │
└────┬────┘     └────┬────┘     └────┬────┘
     │               │               │
     │  Webhooks     │  Webhooks     │  Webhooks
     ▼               ▼               ▼
┌────────────────────────────────────────┐
│      Webhook Handler Service           │
│  ┌──────────┐  ┌──────────┐  ┌──────┐ │
│  │Verifier  │  │ Router   │  │Queue │ │
│  └──────────┘  └──────────┘  └──────┘ │
└────────────────────────────────────────┘
     │               │               │
     ▼               ▼               ▼
┌──────────┐   ┌──────────┐   ┌──────────┐
│Salesforce│   │  Slack   │   │  Email   │
└──────────┘   └──────────┘   └──────────┘
```

## Prerequisites

```bash
# Install dependencies
npm install express crypto axios bull redis dotenv

# Environment variables
export STRIPE_WEBHOOK_SECRET="whsec_..."
export GITHUB_WEBHOOK_SECRET="your_github_secret"
export HUBSPOT_WEBHOOK_SECRET="your_hubspot_secret"
export SALESFORCE_API_KEY="your_salesforce_key"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
export REDIS_URL="redis://localhost:6379"
```

## Implementation

### 1. Webhook Server with Express

```javascript
#!/usr/bin/env node
/**
 * Multi-Platform Webhook Handler
 */

const express = require('express');
const crypto = require('crypto');
const Queue = require('bull');
const axios = require('axios');
require('dotenv').config();

const app = express();

// Raw body parser for signature verification
app.use(express.json({
  verify: (req, res, buf) => {
    req.rawBody = buf.toString('utf8');
  }
}));

// Event queue for async processing
const webhookQueue = new Queue('webhooks', process.env.REDIS_URL);

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    queue: {
      waiting: webhookQueue.getWaitingCount(),
      active: webhookQueue.getActiveCount()
    }
  });
});

// Metrics endpoint
app.get('/metrics', async (req, res) => {
  const [waiting, active, completed, failed] = await Promise.all([
    webhookQueue.getWaitingCount(),
    webhookQueue.getActiveCount(),
    webhookQueue.getCompletedCount(),
    webhookQueue.getFailedCount()
  ]);

  res.json({
    queue: { waiting, active, completed, failed },
    uptime: process.uptime(),
    memory: process.memoryUsage()
  });
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🎣 Webhook server listening on port ${PORT}`);
  console.log(`📊 Health: http://localhost:${PORT}/health`);
  console.log(`📈 Metrics: http://localhost:${PORT}/metrics`);
});

module.exports = app;
```

### 2. Webhook Verification Utilities

```javascript
/**
 * Webhook signature verification
 */

const crypto = require('crypto');

class WebhookVerifier {
  /**
   * Verify Stripe webhook signature
   */
  static verifyStripe(payload, header, secret) {
    const elements = header.split(',');
    const signatures = {};

    elements.forEach(element => {
      const [key, value] = element.split('=');
      signatures[key] = value;
    });

    const timestamp = signatures.t;
    const expectedSignature = signatures.v1;

    // Check timestamp (prevent replay attacks)
    const now = Math.floor(Date.now() / 1000);
    if (Math.abs(now - parseInt(timestamp)) > 300) {
      throw new Error('Timestamp too old');
    }

    // Verify signature
    const signedPayload = `${timestamp}.${payload}`;
    const computedSignature = crypto
      .createHmac('sha256', secret)
      .update(signedPayload)
      .digest('hex');

    if (!crypto.timingSafeEqual(
      Buffer.from(expectedSignature),
      Buffer.from(computedSignature)
    )) {
      throw new Error('Invalid signature');
    }

    return true;
  }

  /**
   * Verify GitHub webhook signature
   */
  static verifyGitHub(payload, signature, secret) {
    const expectedSignature = 'sha256=' + crypto
      .createHmac('sha256', secret)
      .update(payload)
      .digest('hex');

    return crypto.timingSafeEqual(
      Buffer.from(signature),
      Buffer.from(expectedSignature)
    );
  }

  /**
   * Verify HubSpot webhook signature
   */
  static verifyHubSpot(payload, signature, secret) {
    const computedSignature = crypto
      .createHmac('sha256', secret)
      .update(payload)
      .digest('hex');

    return crypto.timingSafeEqual(
      Buffer.from(signature),
      Buffer.from(computedSignature)
    );
  }
}

module.exports = WebhookVerifier;
```

### 3. Stripe Payment Webhooks

```javascript
/**
 * Stripe webhook handler
 */

const WebhookVerifier = require('./webhook-verifier');

// Stripe webhook endpoint
app.post('/webhooks/stripe', async (req, res) => {
  const signature = req.headers['stripe-signature'];

  try {
    // Verify signature
    WebhookVerifier.verifyStripe(
      req.rawBody,
      signature,
      process.env.STRIPE_WEBHOOK_SECRET
    );

    const event = req.body;
    console.log(`📧 Stripe webhook: ${event.type}`);

    // Queue for async processing
    await webhookQueue.add('stripe', {
      eventType: event.type,
      eventId: event.id,
      data: event.data.object,
      timestamp: new Date().toISOString()
    });

    res.json({ received: true });

  } catch (error) {
    console.error('Stripe webhook error:', error);
    res.status(400).json({ error: error.message });
  }
});

// Process Stripe events
webhookQueue.process('stripe', async (job) => {
  const { eventType, data } = job.data;

  switch (eventType) {
    case 'payment_intent.succeeded':
      await handlePaymentSuccess(data);
      break;

    case 'payment_intent.payment_failed':
      await handlePaymentFailed(data);
      break;

    case 'charge.refunded':
      await handleRefund(data);
      break;

    case 'customer.created':
      await handleNewCustomer(data);
      break;

    default:
      console.log(`Unhandled event type: ${eventType}`);
  }
});

async function handlePaymentSuccess(paymentIntent) {
  console.log(`✅ Payment succeeded: ${paymentIntent.id}`);

  // 1. Update Salesforce
  await updateSalesforceOpportunity({
    stripe_payment_id: paymentIntent.id,
    amount: paymentIntent.amount / 100,
    status: 'Closed Won',
    close_date: new Date().toISOString()
  });

  // 2. Send confirmation email
  await sendEmail({
    to: paymentIntent.receipt_email,
    subject: 'Payment Confirmation',
    template: 'payment-success',
    data: {
      amount: paymentIntent.amount / 100,
      currency: paymentIntent.currency.toUpperCase(),
      payment_id: paymentIntent.id
    }
  });

  // 3. Notify Slack
  await notifySlack({
    text: `💰 Payment received: $${paymentIntent.amount / 100}`,
    channel: '#sales'
  });
}

async function handlePaymentFailed(paymentIntent) {
  console.log(`❌ Payment failed: ${paymentIntent.id}`);

  // 1. Update Salesforce
  await updateSalesforceOpportunity({
    stripe_payment_id: paymentIntent.id,
    status: 'Payment Failed',
    failure_reason: paymentIntent.last_payment_error?.message
  });

  // 2. Send failure notification
  await sendEmail({
    to: paymentIntent.receipt_email,
    subject: 'Payment Failed',
    template: 'payment-failed',
    data: {
      amount: paymentIntent.amount / 100,
      reason: paymentIntent.last_payment_error?.message
    }
  });

  // 3. Alert team
  await notifySlack({
    text: `🚨 Payment failed: ${paymentIntent.last_payment_error?.message}`,
    channel: '#alerts'
  });
}

async function handleRefund(charge) {
  console.log(`💸 Refund processed: ${charge.id}`);

  // Update Salesforce and notify customer
  await Promise.all([
    updateSalesforceOpportunity({
      stripe_charge_id: charge.id,
      status: 'Refunded',
      refund_amount: charge.amount_refunded / 100
    }),
    sendEmail({
      to: charge.billing_details?.email,
      subject: 'Refund Processed',
      template: 'refund-confirmation',
      data: {
        amount: charge.amount_refunded / 100,
        charge_id: charge.id
      }
    })
  ]);
}

async function handleNewCustomer(customer) {
  console.log(`👤 New customer: ${customer.id}`);

  // Create Salesforce account
  await createSalesforceAccount({
    name: customer.name || customer.email,
    email: customer.email,
    stripe_customer_id: customer.id,
    phone: customer.phone
  });
}
```

### 4. GitHub Push Webhooks

```javascript
/**
 * GitHub webhook handler
 */

// GitHub webhook endpoint
app.post('/webhooks/github', async (req, res) => {
  const signature = req.headers['x-hub-signature-256'];
  const event = req.headers['x-github-event'];

  try {
    // Verify signature
    if (!WebhookVerifier.verifyGitHub(
      req.rawBody,
      signature,
      process.env.GITHUB_WEBHOOK_SECRET
    )) {
      return res.status(401).json({ error: 'Invalid signature' });
    }

    console.log(`🐙 GitHub webhook: ${event}`);

    // Queue for processing
    await webhookQueue.add('github', {
      eventType: event,
      data: req.body,
      timestamp: new Date().toISOString()
    });

    res.json({ received: true });

  } catch (error) {
    console.error('GitHub webhook error:', error);
    res.status(400).json({ error: error.message });
  }
});

// Process GitHub events
webhookQueue.process('github', async (job) => {
  const { eventType, data } = job.data;

  switch (eventType) {
    case 'push':
      await handleGitHubPush(data);
      break;

    case 'pull_request':
      await handlePullRequest(data);
      break;

    case 'issues':
      await handleIssue(data);
      break;

    default:
      console.log(`Unhandled GitHub event: ${eventType}`);
  }
});

async function handleGitHubPush(data) {
  const { repository, pusher, commits, ref } = data;
  const branch = ref.replace('refs/heads/', '');

  console.log(`📝 Push to ${repository.full_name}/${branch} by ${pusher.name}`);

  // Notify Slack
  const commitList = commits
    .slice(0, 3)
    .map(c => `• ${c.message} - ${c.author.name}`)
    .join('\n');

  await notifySlack({
    text: `🚀 New push to *${repository.full_name}*\n` +
          `Branch: \`${branch}\`\n` +
          `Commits:\n${commitList}`,
    channel: '#engineering'
  });

  // Trigger CI/CD if main branch
  if (branch === 'main' || branch === 'master') {
    await triggerCICD(repository.full_name, ref);
  }
}

async function handlePullRequest(data) {
  const { action, pull_request } = data;

  if (action === 'opened') {
    await notifySlack({
      text: `🔀 New PR: *${pull_request.title}*\n` +
            `Author: ${pull_request.user.login}\n` +
            `${pull_request.html_url}`,
      channel: '#code-review'
    });
  }
}
```

### 5. HubSpot Contact Update Webhooks

```javascript
/**
 * HubSpot webhook handler
 */

// HubSpot webhook endpoint
app.post('/webhooks/hubspot', async (req, res) => {
  const signature = req.headers['x-hubspot-signature'];

  try {
    // Verify signature
    if (!WebhookVerifier.verifyHubSpot(
      req.rawBody,
      signature,
      process.env.HUBSPOT_WEBHOOK_SECRET
    )) {
      return res.status(401).json({ error: 'Invalid signature' });
    }

    const events = req.body;
    console.log(`📇 HubSpot webhook: ${events.length} events`);

    // Queue each event
    for (const event of events) {
      await webhookQueue.add('hubspot', {
        eventType: 'contact.propertyChange',
        data: event,
        timestamp: new Date().toISOString()
      });
    }

    res.json({ received: true });

  } catch (error) {
    console.error('HubSpot webhook error:', error);
    res.status(400).json({ error: error.message });
  }
});

// Process HubSpot events
webhookQueue.process('hubspot', async (job) => {
  const { data } = job.data;

  await syncContactToSalesforce(data);
});

async function syncContactToSalesforce(event) {
  const { objectId, propertyName, propertyValue } = event;

  console.log(`🔄 Syncing HubSpot contact ${objectId} to Salesforce`);

  // Fetch full contact details from HubSpot
  const contact = await fetchHubSpotContact(objectId);

  // Transform and sync to Salesforce
  await updateSalesforceContact({
    hubspot_id: objectId,
    email: contact.email,
    first_name: contact.firstname,
    last_name: contact.lastname,
    phone: contact.phone,
    company: contact.company
  });
}
```

### 6. Helper Functions

```javascript
/**
 * Integration helper functions
 */

async function updateSalesforceOpportunity(data) {
  // Implementation using Salesforce API
  console.log('Updating Salesforce opportunity:', data);
}

async function createSalesforceAccount(data) {
  console.log('Creating Salesforce account:', data);
}

async function updateSalesforceContact(data) {
  console.log('Updating Salesforce contact:', data);
}

async function sendEmail(options) {
  // Implementation using SendGrid/AWS SES
  console.log('Sending email:', options);
}

async function notifySlack(message) {
  try {
    await axios.post(process.env.SLACK_WEBHOOK_URL, message);
  } catch (error) {
    console.error('Slack notification failed:', error.message);
  }
}

async function triggerCICD(repo, ref) {
  console.log(`Triggering CI/CD for ${repo} @ ${ref}`);
}

async function fetchHubSpotContact(contactId) {
  // Fetch from HubSpot API
  return {
    email: 'example@test.com',
    firstname: 'John',
    lastname: 'Doe'
  };
}
```

## Usage

```bash
# Start webhook server
node webhook-automation.js

# Register webhooks with platforms
node register-webhooks.js

# Test with ngrok for local development
ngrok http 3000

# Update webhook URLs in platform dashboards
# Stripe: https://dashboard.stripe.com/webhooks
# GitHub: Repository Settings → Webhooks
# HubSpot: Settings → Integrations → Webhooks
```

## Production Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  webhook-handler:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  redis-data:
```

## Monitoring

```javascript
// Add Prometheus metrics
const prometheus = require('prom-client');

const webhooksReceived = new prometheus.Counter({
  name: 'webhooks_received_total',
  help: 'Total webhooks received',
  labelNames: ['platform', 'event_type']
});

const webhookProcessingTime = new prometheus.Histogram({
  name: 'webhook_processing_duration_seconds',
  help: 'Webhook processing duration',
  labelNames: ['platform']
});

app.get('/metrics', async (req, res) => {
  res.set('Content-Type', prometheus.register.contentType);
  res.end(await prometheus.register.metrics());
});
```

---

**Generated with Platform Integration Skill v2.0.0**
