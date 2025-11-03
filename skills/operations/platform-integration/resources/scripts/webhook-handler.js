#!/usr/bin/env node
/**
 * Webhook Handler Generator
 * Generates Express/Fastify webhook handlers with signature verification
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

class WebhookHandlerGenerator {
  constructor(config) {
    this.config = config;
    this.framework = config.framework || 'express';
  }

  generate(outputDir) {
    console.log(`🔧 Generating webhook handlers (${this.framework})...`);

    // Create directory structure
    fs.mkdirSync(outputDir, { recursive: true });

    // Generate main handler
    this.generateMainHandler(outputDir);

    // Generate platform-specific handlers
    const platforms = this.config.platforms || [];
    platforms.forEach(platform => {
      if (platform.webhooks) {
        this.generatePlatformHandler(outputDir, platform);
      }
    });

    // Generate verification utilities
    this.generateVerificationUtils(outputDir);

    // Generate registration script
    this.generateRegistrationScript(outputDir);

    console.log('✅ Webhook handlers generated');
  }

  generateMainHandler(outputDir) {
    const template = this.framework === 'express'
      ? this.getExpressTemplate()
      : this.getFastifyTemplate();

    const code = template.replace('{{ROUTES}}', this.generateRoutes());

    fs.writeFileSync(
      path.join(outputDir, 'webhook-server.js'),
      code
    );
  }

  getExpressTemplate() {
    return `const express = require('express');
const crypto = require('crypto');
const { WebhookVerifier } = require('./webhook-verifier');
const { handleWebhookEvent } = require('./event-processor');

const app = express();

// Raw body parser for signature verification
app.use(express.json({
  verify: (req, res, buf) => {
    req.rawBody = buf.toString('utf8');
  }
}));

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Webhook routes
{{ROUTES}}

// Error handler
app.use((err, req, res, next) => {
  console.error('Webhook error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(\`🎣 Webhook server listening on port \${PORT}\`);
});

module.exports = app;
`;
  }

  getFastifyTemplate() {
    return `const fastify = require('fastify')({ logger: true });
const crypto = require('crypto');
const { WebhookVerifier } = require('./webhook-verifier');
const { handleWebhookEvent } = require('./event-processor');

// Health check
fastify.get('/health', async (request, reply) => {
  return { status: 'healthy', timestamp: new Date().toISOString() };
});

// Webhook routes
{{ROUTES}}

// Start server
const start = async () => {
  try {
    await fastify.listen({ port: process.env.PORT || 3000 });
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

start();

module.exports = fastify;
`;
  }

  generateRoutes() {
    const platforms = this.config.platforms || [];
    const routes = [];

    platforms.forEach(platform => {
      if (platform.webhooks) {
        const route = this.generatePlatformRoute(platform);
        routes.push(route);
      }
    });

    return routes.join('\n\n');
  }

  generatePlatformRoute(platform) {
    const platformName = platform.name;
    const signatureHeader = platform.webhooks.signature_header || 'X-Signature';
    const secret = platform.webhooks.secret || 'WEBHOOK_SECRET';

    if (this.framework === 'express') {
      return `// ${platformName} webhook
app.post('/webhooks/${platformName}', async (req, res) => {
  try {
    const signature = req.headers['${signatureHeader.toLowerCase()}'];
    const verifier = new WebhookVerifier(process.env.${secret});

    // Verify signature
    if (!verifier.verifySignature(req.rawBody, signature)) {
      return res.status(401).json({ error: 'Invalid signature' });
    }

    // Process event
    const result = await handleWebhookEvent('${platformName}', req.body);

    res.json({ received: true, event: req.body.type });
  } catch (error) {
    console.error('${platformName} webhook error:', error);
    res.status(500).json({ error: 'Processing failed' });
  }
});`;
    } else {
      return `// ${platformName} webhook
fastify.post('/webhooks/${platformName}', async (request, reply) => {
  const signature = request.headers['${signatureHeader.toLowerCase()}'];
  const verifier = new WebhookVerifier(process.env.${secret});

  if (!verifier.verifySignature(JSON.stringify(request.body), signature)) {
    return reply.status(401).send({ error: 'Invalid signature' });
  }

  const result = await handleWebhookEvent('${platformName}', request.body);
  return { received: true, event: request.body.type };
});`;
    }
  }

  generatePlatformHandler(outputDir, platform) {
    const code = `/**
 * ${platform.name} Webhook Handler
 * Processes ${platform.name} webhook events
 */

class ${this.toPascalCase(platform.name)}WebhookHandler {
  constructor() {
    this.eventHandlers = {
${this.generateEventHandlers(platform.webhooks.events || [])}
    };
  }

  async handle(event) {
    const eventType = event.type || event.event_type;
    const handler = this.eventHandlers[eventType];

    if (handler) {
      return await handler(event);
    } else {
      console.warn(\`Unhandled event type: \${eventType}\`);
      return { handled: false };
    }
  }

${this.generateHandlerMethods(platform.webhooks.events || [])}
}

module.exports = ${this.toPascalCase(platform.name)}WebhookHandler;
`;

    fs.writeFileSync(
      path.join(outputDir, `${platform.name}-handler.js`),
      code
    );
  }

  generateEventHandlers(events) {
    return events.map(event =>
      `      '${event}': this.handle${this.toPascalCase(event)}.bind(this)`
    ).join(',\n');
  }

  generateHandlerMethods(events) {
    return events.map(event => `  async handle${this.toPascalCase(event)}(event) {
    console.log('Processing ${event}:', event.id || event.event_id);

    // TODO: Implement ${event} processing logic
    // - Validate event data
    // - Transform data if needed
    // - Store in database
    // - Trigger downstream actions

    return { handled: true, event_type: '${event}' };
  }`).join('\n\n');
  }

  generateVerificationUtils(outputDir) {
    const code = `const crypto = require('crypto');

class WebhookVerifier {
  constructor(secret) {
    this.secret = secret;
  }

  /**
   * Verify HMAC signature
   */
  verifySignature(payload, signature, algorithm = 'sha256') {
    if (!signature) return false;

    // Remove algorithm prefix if present (e.g., "sha256=...")
    const actualSignature = signature.split('=').pop();

    const expectedSignature = crypto
      .createHmac(algorithm, this.secret)
      .update(payload)
      .digest('hex');

    return crypto.timingSafeEqual(
      Buffer.from(actualSignature),
      Buffer.from(expectedSignature)
    );
  }

  /**
   * Verify timestamp to prevent replay attacks
   */
  verifyTimestamp(timestamp, maxAge = 300) {
    const now = Math.floor(Date.now() / 1000);
    return Math.abs(now - timestamp) <= maxAge;
  }

  /**
   * Stripe-style signature verification
   */
  verifyStripeSignature(payload, header) {
    const elements = header.split(',');
    const signatures = {};

    elements.forEach(element => {
      const [key, value] = element.split('=');
      signatures[key] = value;
    });

    const timestamp = signatures.t;
    const expectedSignature = signatures.v1;

    // Check timestamp
    if (!this.verifyTimestamp(parseInt(timestamp))) {
      throw new Error('Timestamp too old');
    }

    // Verify signature
    const signedPayload = \`\${timestamp}.\${payload}\`;
    const computedSignature = crypto
      .createHmac('sha256', this.secret)
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
}

module.exports = { WebhookVerifier };
`;

    fs.writeFileSync(
      path.join(outputDir, 'webhook-verifier.js'),
      code
    );
  }

  generateRegistrationScript(outputDir) {
    const code = `#!/usr/bin/env node
/**
 * Webhook Registration Script
 * Registers webhooks with platform APIs
 */

const axios = require('axios');

async function registerWebhook(platform, config) {
  console.log(\`📝 Registering webhook for \${platform}...\`);

  try {
    const response = await axios.post(
      config.registration_url,
      {
        url: config.webhook_url,
        events: config.events,
        secret: config.secret
      },
      {
        headers: {
          'Authorization': \`Bearer \${config.api_token}\`,
          'Content-Type': 'application/json'
        }
      }
    );

    console.log(\`✅ Webhook registered: \${response.data.id || 'success'}\`);
    return response.data;
  } catch (error) {
    console.error(\`❌ Registration failed:\`, error.message);
    throw error;
  }
}

async function main() {
  const platforms = [
    // Add your platform configurations here
    // {
    //   name: 'stripe',
    //   registration_url: 'https://api.stripe.com/v1/webhook_endpoints',
    //   webhook_url: 'https://your-domain.com/webhooks/stripe',
    //   events: ['payment_intent.succeeded', 'charge.failed'],
    //   api_token: process.env.STRIPE_API_KEY,
    //   secret: process.env.STRIPE_WEBHOOK_SECRET
    // }
  ];

  for (const platform of platforms) {
    await registerWebhook(platform.name, platform);
  }
}

if (require.main === module) {
  main().catch(console.error);
}

module.exports = { registerWebhook };
`;

    fs.writeFileSync(
      path.join(outputDir, 'register-webhooks.js'),
      code
    );
    fs.chmodSync(path.join(outputDir, 'register-webhooks.js'), 0o755);
  }

  toPascalCase(str) {
    return str
      .split(/[-_.]/)
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join('');
  }
}

// CLI
if (require.main === module) {
  const args = require('minimist')(process.argv.slice(2));

  if (!args.config || !args.output) {
    console.error('Usage: webhook-handler.js --config <file> --output <dir> [--framework express|fastify]');
    process.exit(1);
  }

  const config = JSON.parse(fs.readFileSync(args.config, 'utf8'));
  config.framework = args.framework || 'express';

  const generator = new WebhookHandlerGenerator(config);
  generator.generate(args.output);
}

module.exports = WebhookHandlerGenerator;
