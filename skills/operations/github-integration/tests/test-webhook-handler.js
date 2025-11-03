/**
 * Test Suite for GitHub Webhook Handler
 * Comprehensive tests for webhook processing, validation, and routing
 */

const http = require('http');
const crypto = require('crypto');
const assert = require('assert');

// Test configuration
const WEBHOOK_PORT = 3001;
const WEBHOOK_SECRET = 'test-secret-key';
const BASE_URL = `http://localhost:${WEBHOOK_PORT}`;

/**
 * Generate webhook signature
 */
function generateSignature(payload, secret) {
  const hmac = crypto.createHmac('sha256', secret);
  return 'sha256=' + hmac.update(JSON.stringify(payload)).digest('hex');
}

/**
 * Send webhook request
 */
function sendWebhook(event, payload, signature) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(payload);

    const options = {
      hostname: 'localhost',
      port: WEBHOOK_PORT,
      path: '/webhook',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': data.length,
        'X-GitHub-Event': event,
        'X-Hub-Signature-256': signature
      }
    };

    const req = http.request(options, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        resolve({
          statusCode: res.statusCode,
          body: JSON.parse(body)
        });
      });
    });

    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

/**
 * Test: Health check endpoint
 */
async function testHealthCheck() {
  console.log('\n[TEST] Health check endpoint');

  return new Promise((resolve, reject) => {
    http.get(`${BASE_URL}/health`, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        const data = JSON.parse(body);
        assert.strictEqual(res.statusCode, 200);
        assert.strictEqual(data.status, 'healthy');
        console.log('✅ Health check passed');
        resolve();
      });
    }).on('error', reject);
  });
}

/**
 * Test: Pull request opened event
 */
async function testPullRequestOpened() {
  console.log('\n[TEST] Pull request opened event');

  const payload = {
    action: 'opened',
    pull_request: {
      number: 123,
      title: 'Test PR',
      state: 'open',
      user: { login: 'testuser' },
      mergeable_state: 'clean'
    },
    repository: {
      full_name: 'owner/repo'
    }
  };

  const signature = generateSignature(payload, WEBHOOK_SECRET);
  const response = await sendWebhook('pull_request', payload, signature);

  assert.strictEqual(response.statusCode, 200);
  assert.strictEqual(response.body.event, 'pull_request');
  assert.strictEqual(response.body.action, 'opened');
  console.log('✅ PR opened event handled correctly');
}

/**
 * Test: Issue created event
 */
async function testIssueCreated() {
  console.log('\n[TEST] Issue created event');

  const payload = {
    action: 'opened',
    issue: {
      number: 456,
      title: 'Test Issue',
      user: { login: 'testuser' },
      labels: [{ name: 'bug' }]
    },
    repository: {
      full_name: 'owner/repo'
    }
  };

  const signature = generateSignature(payload, WEBHOOK_SECRET);
  const response = await sendWebhook('issues', payload, signature);

  assert.strictEqual(response.statusCode, 200);
  assert.strictEqual(response.body.event, 'issues');
  assert.strictEqual(response.body.action, 'opened');
  console.log('✅ Issue created event handled correctly');
}

/**
 * Test: Push event
 */
async function testPushEvent() {
  console.log('\n[TEST] Push event');

  const payload = {
    ref: 'refs/heads/main',
    commits: [
      { id: 'abc123', message: 'feat: add feature' },
      { id: 'def456', message: 'fix: fix bug' }
    ],
    pusher: { name: 'testuser' },
    repository: {
      full_name: 'owner/repo'
    }
  };

  const signature = generateSignature(payload, WEBHOOK_SECRET);
  const response = await sendWebhook('push', payload, signature);

  assert.strictEqual(response.statusCode, 200);
  assert.strictEqual(response.body.event, 'push');
  assert.strictEqual(response.body.branch, 'main');
  console.log('✅ Push event handled correctly');
}

/**
 * Test: Release published event
 */
async function testReleasePublished() {
  console.log('\n[TEST] Release published event');

  const payload = {
    action: 'published',
    release: {
      tag_name: 'v1.0.0',
      name: 'Release 1.0.0',
      prerelease: false
    },
    repository: {
      full_name: 'owner/repo'
    }
  };

  const signature = generateSignature(payload, WEBHOOK_SECRET);
  const response = await sendWebhook('release', payload, signature);

  assert.strictEqual(response.statusCode, 200);
  assert.strictEqual(response.body.event, 'release');
  assert.strictEqual(response.body.action, 'published');
  console.log('✅ Release published event handled correctly');
}

/**
 * Test: Workflow run completed
 */
async function testWorkflowRunCompleted() {
  console.log('\n[TEST] Workflow run completed event');

  const payload = {
    action: 'completed',
    workflow_run: {
      name: 'CI',
      status: 'completed',
      conclusion: 'success',
      run_number: 42
    },
    repository: {
      full_name: 'owner/repo'
    }
  };

  const signature = generateSignature(payload, WEBHOOK_SECRET);
  const response = await sendWebhook('workflow_run', payload, signature);

  assert.strictEqual(response.statusCode, 200);
  assert.strictEqual(response.body.event, 'workflow_run');
  console.log('✅ Workflow run event handled correctly');
}

/**
 * Test: Invalid signature rejection
 */
async function testInvalidSignature() {
  console.log('\n[TEST] Invalid signature rejection');

  const payload = {
    action: 'opened',
    pull_request: { number: 1 },
    repository: { full_name: 'owner/repo' }
  };

  const invalidSignature = 'sha256=invalid';
  const response = await sendWebhook('pull_request', payload, invalidSignature);

  assert.strictEqual(response.statusCode, 401);
  assert.strictEqual(response.body.error, 'Invalid signature');
  console.log('✅ Invalid signature rejected correctly');
}

/**
 * Test: Unknown event handling
 */
async function testUnknownEvent() {
  console.log('\n[TEST] Unknown event handling');

  const payload = {
    action: 'test',
    repository: { full_name: 'owner/repo' }
  };

  const signature = generateSignature(payload, WEBHOOK_SECRET);
  const response = await sendWebhook('unknown_event', payload, signature);

  assert.strictEqual(response.statusCode, 200);
  assert.strictEqual(response.body.status, 'ignored');
  console.log('✅ Unknown event ignored correctly');
}

/**
 * Test: Comment slash command detection
 */
async function testSlashCommand() {
  console.log('\n[TEST] Slash command in comment');

  const payload = {
    action: 'created',
    issue: { number: 789 },
    comment: {
      user: { login: 'testuser' },
      body: '/deploy staging'
    },
    repository: {
      full_name: 'owner/repo'
    }
  };

  const signature = generateSignature(payload, WEBHOOK_SECRET);
  const response = await sendWebhook('issue_comment', payload, signature);

  assert.strictEqual(response.statusCode, 200);
  assert.strictEqual(response.body.event, 'issue_comment');
  console.log('✅ Slash command detected correctly');
}

/**
 * Run all tests
 */
async function runTests() {
  console.log('=================================');
  console.log('GitHub Webhook Handler Test Suite');
  console.log('=================================');

  // Set environment variables for webhook handler
  process.env.WEBHOOK_PORT = WEBHOOK_PORT;
  process.env.WEBHOOK_SECRET = WEBHOOK_SECRET;

  // Start webhook handler
  console.log('\n[INFO] Starting webhook handler...');
  const handler = require('../resources/scripts/webhook-handler.js');

  // Wait for server to start
  await new Promise(resolve => setTimeout(resolve, 1000));

  try {
    // Run tests
    await testHealthCheck();
    await testPullRequestOpened();
    await testIssueCreated();
    await testPushEvent();
    await testReleasePublished();
    await testWorkflowRunCompleted();
    await testInvalidSignature();
    await testUnknownEvent();
    await testSlashCommand();

    console.log('\n=================================');
    console.log('✅ All tests passed!');
    console.log('=================================\n');
    process.exit(0);
  } catch (error) {
    console.error('\n=================================');
    console.error('❌ Test failed:', error.message);
    console.error('=================================\n');
    process.exit(1);
  }
}

// Run tests if executed directly
if (require.main === module) {
  runTests();
}

module.exports = { runTests };
