#!/usr/bin/env node
/**
 * GitHub Webhook Handler
 * Processes incoming GitHub webhooks with event routing and validation
 * Supports PR events, issue events, release events, and workflow runs
 */

const crypto = require('crypto');
const http = require('http');

// Configuration
const PORT = process.env.WEBHOOK_PORT || 3000;
const SECRET = process.env.WEBHOOK_SECRET || '';
const VERBOSE = process.env.VERBOSE === 'true';

// Event handlers registry
const handlers = {
  'pull_request': handlePullRequest,
  'issues': handleIssue,
  'push': handlePush,
  'release': handleRelease,
  'workflow_run': handleWorkflowRun,
  'check_run': handleCheckRun,
  'issue_comment': handleComment,
};

/**
 * Verify webhook signature using HMAC-SHA256
 */
function verifySignature(payload, signature) {
  if (!SECRET) {
    console.warn('[WARN] WEBHOOK_SECRET not set - skipping signature validation');
    return true;
  }

  const hmac = crypto.createHmac('sha256', SECRET);
  const digest = 'sha256=' + hmac.update(payload).digest('hex');

  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(digest)
  );
}

/**
 * Log event information
 */
function logEvent(event, action, repo, metadata = {}) {
  const timestamp = new Date().toISOString();
  const logEntry = {
    timestamp,
    event,
    action,
    repository: repo,
    ...metadata
  };

  console.log(JSON.stringify(logEntry));
}

/**
 * Handle Pull Request events
 */
async function handlePullRequest(payload) {
  const { action, pull_request, repository } = payload;
  const pr = pull_request;

  logEvent('pull_request', action, repository.full_name, {
    pr_number: pr.number,
    pr_title: pr.title,
    author: pr.user.login,
    state: pr.state,
    mergeable: pr.mergeable_state
  });

  switch (action) {
    case 'opened':
      console.log(`[PR OPENED] #${pr.number}: ${pr.title}`);
      // Trigger: Auto-assign reviewers, add labels
      break;

    case 'synchronize':
      console.log(`[PR UPDATED] #${pr.number}: New commits pushed`);
      // Trigger: Re-run CI checks, invalidate approvals
      break;

    case 'review_requested':
      console.log(`[REVIEW REQUESTED] #${pr.number} assigned to reviewers`);
      // Trigger: Notify reviewers
      break;

    case 'closed':
      if (pr.merged) {
        console.log(`[PR MERGED] #${pr.number}: ${pr.title}`);
        // Trigger: Update project boards, close related issues
      } else {
        console.log(`[PR CLOSED] #${pr.number}: ${pr.title}`);
      }
      break;
  }

  return { status: 'processed', event: 'pull_request', action };
}

/**
 * Handle Issue events
 */
async function handleIssue(payload) {
  const { action, issue, repository } = payload;

  logEvent('issues', action, repository.full_name, {
    issue_number: issue.number,
    issue_title: issue.title,
    author: issue.user.login,
    labels: issue.labels.map(l => l.name)
  });

  switch (action) {
    case 'opened':
      console.log(`[ISSUE OPENED] #${issue.number}: ${issue.title}`);
      // Trigger: Auto-triage, add to project board
      break;

    case 'labeled':
      console.log(`[ISSUE LABELED] #${issue.number}: Added label ${payload.label.name}`);
      // Trigger: Update project board status
      break;

    case 'assigned':
      console.log(`[ISSUE ASSIGNED] #${issue.number} to ${payload.assignee.login}`);
      // Trigger: Notify assignee
      break;

    case 'closed':
      console.log(`[ISSUE CLOSED] #${issue.number}: ${issue.title}`);
      // Trigger: Update sprint metrics
      break;
  }

  return { status: 'processed', event: 'issues', action };
}

/**
 * Handle Push events
 */
async function handlePush(payload) {
  const { ref, repository, commits, pusher } = payload;
  const branch = ref.replace('refs/heads/', '');

  logEvent('push', 'pushed', repository.full_name, {
    branch,
    commits: commits.length,
    pusher: pusher.name
  });

  console.log(`[PUSH] ${commits.length} commit(s) to ${branch} by ${pusher.name}`);

  if (branch === 'main' || branch === 'master') {
    console.log('[PUSH] Main branch updated - triggering production checks');
    // Trigger: Production deployment checks
  }

  return { status: 'processed', event: 'push', branch };
}

/**
 * Handle Release events
 */
async function handleRelease(payload) {
  const { action, release, repository } = payload;

  logEvent('release', action, repository.full_name, {
    tag: release.tag_name,
    name: release.name,
    prerelease: release.prerelease
  });

  switch (action) {
    case 'published':
      console.log(`[RELEASE PUBLISHED] ${release.tag_name}: ${release.name}`);
      // Trigger: Deployment pipeline, notifications
      break;

    case 'created':
      console.log(`[RELEASE CREATED] ${release.tag_name} (draft)`);
      // Trigger: Release validation checks
      break;
  }

  return { status: 'processed', event: 'release', action };
}

/**
 * Handle Workflow Run events
 */
async function handleWorkflowRun(payload) {
  const { action, workflow_run, repository } = payload;

  logEvent('workflow_run', action, repository.full_name, {
    workflow: workflow_run.name,
    status: workflow_run.status,
    conclusion: workflow_run.conclusion,
    run_number: workflow_run.run_number
  });

  if (action === 'completed') {
    if (workflow_run.conclusion === 'failure') {
      console.log(`[WORKFLOW FAILED] ${workflow_run.name} #${workflow_run.run_number}`);
      // Trigger: Failure analysis, notifications
    } else if (workflow_run.conclusion === 'success') {
      console.log(`[WORKFLOW SUCCESS] ${workflow_run.name} #${workflow_run.run_number}`);
      // Trigger: Deployment if applicable
    }
  }

  return { status: 'processed', event: 'workflow_run', action };
}

/**
 * Handle Check Run events
 */
async function handleCheckRun(payload) {
  const { action, check_run, repository } = payload;

  logEvent('check_run', action, repository.full_name, {
    check_name: check_run.name,
    status: check_run.status,
    conclusion: check_run.conclusion
  });

  return { status: 'processed', event: 'check_run', action };
}

/**
 * Handle Comment events
 */
async function handleComment(payload) {
  const { action, issue, comment, repository } = payload;

  logEvent('issue_comment', action, repository.full_name, {
    issue_number: issue.number,
    commenter: comment.user.login
  });

  // Check for slash commands in comments
  const commandMatch = comment.body.match(/^\/(\w+)/);
  if (commandMatch) {
    const command = commandMatch[1];
    console.log(`[COMMAND] /${command} triggered by ${comment.user.login}`);
    // Trigger: Process slash command
  }

  return { status: 'processed', event: 'issue_comment', action };
}

/**
 * HTTP server request handler
 */
const server = http.createServer(async (req, res) => {
  if (req.method === 'POST' && req.url === '/webhook') {
    let body = '';

    req.on('data', chunk => {
      body += chunk.toString();
    });

    req.on('end', async () => {
      try {
        // Verify signature
        const signature = req.headers['x-hub-signature-256'] || '';
        if (!verifySignature(body, signature)) {
          console.error('[ERROR] Invalid webhook signature');
          res.writeHead(401, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Invalid signature' }));
          return;
        }

        // Parse payload
        const payload = JSON.parse(body);
        const event = req.headers['x-github-event'];

        if (VERBOSE) {
          console.log(`[WEBHOOK] Received ${event} event`);
          console.log(JSON.stringify(payload, null, 2));
        }

        // Route to appropriate handler
        const handler = handlers[event];
        if (handler) {
          const result = await handler(payload);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(result));
        } else {
          console.warn(`[WARN] No handler for event: ${event}`);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ status: 'ignored', event }));
        }
      } catch (error) {
        console.error('[ERROR] Processing webhook:', error);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: error.message }));
      }
    });
  } else if (req.method === 'GET' && req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'healthy' }));
  } else {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not found' }));
  }
});

// Start server
server.listen(PORT, () => {
  console.log(`[INFO] GitHub webhook handler listening on port ${PORT}`);
  console.log(`[INFO] Webhook URL: http://localhost:${PORT}/webhook`);
  console.log(`[INFO] Health check: http://localhost:${PORT}/health`);

  if (!SECRET) {
    console.warn('[WARN] Running without webhook secret - set WEBHOOK_SECRET for production');
  }
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('[INFO] Received SIGTERM - shutting down gracefully');
  server.close(() => {
    console.log('[INFO] Server closed');
    process.exit(0);
  });
});
