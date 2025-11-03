#!/usr/bin/env node
/**
 * GitHub PR Review Metrics Tracker
 *
 * Tracks and analyzes review effectiveness metrics including review time,
 * issues found, fix rates, and agent performance.
 *
 * Usage:
 *   node review-metrics.js --pr 123
 *   node review-metrics.js --period 30d --format json
 *   node review-metrics.js --export-dashboard
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);

if (args.includes('--help') || args.length === 0) {
  console.log('Usage: node review-metrics.js [options]');
  console.log('');
  console.log('Options:');
  console.log('  --pr <number>          Analyze specific PR');
  console.log('  --period <duration>    Analysis period (7d, 30d, 90d)');
  console.log('  --format <type>        Output format (text|json|markdown)');
  console.log('  --export-dashboard     Export metrics dashboard');
  process.exit(0);
}

/**
 * Get argument value
 */
function getArg(flag) {
  const idx = args.indexOf(flag);
  return idx !== -1 && idx + 1 < args.length ? args[idx + 1] : null;
}

const prNumber = getArg('--pr');
const period = getArg('--period') || '30d';
const format = getArg('--format') || 'text';
const exportDashboard = args.includes('--export-dashboard');

/**
 * Execute GitHub CLI command
 */
function ghExec(command) {
  try {
    return execSync(`gh ${command}`, { encoding: 'utf-8' });
  } catch (error) {
    console.error(`Error executing gh command: ${error.message}`);
    return null;
  }
}

/**
 * Collect metrics for a specific PR
 */
function collectPRMetrics(prNumber) {
  const pr = JSON.parse(ghExec(`pr view ${prNumber} --json number,title,createdAt,mergedAt,comments,reviews,labels`));

  const metrics = {
    prNumber: pr.number,
    title: pr.title,
    createdAt: new Date(pr.createdAt),
    mergedAt: pr.mergedAt ? new Date(pr.mergedAt) : null,
    reviewTime: null,
    commentsCount: pr.comments?.length || 0,
    reviewsCount: pr.reviews?.length || 0,
    issuesFound: 0,
    criticalIssues: 0,
    fixedIssues: 0,
    labels: pr.labels?.map(l => l.name) || []
  };

  // Calculate review time (if merged)
  if (metrics.mergedAt) {
    const timeDiff = metrics.mergedAt - metrics.createdAt;
    metrics.reviewTime = Math.round(timeDiff / (1000 * 60)); // minutes
  }

  // Analyze comments for issues
  if (pr.comments) {
    pr.comments.forEach(comment => {
      const body = comment.body.toLowerCase();

      // Count issues based on keywords
      if (body.includes('issue') || body.includes('problem') || body.includes('bug')) {
        metrics.issuesFound++;
      }
      if (body.includes('critical') || body.includes('security') || body.includes('vulnerability')) {
        metrics.criticalIssues++;
      }
      if (body.includes('fixed') || body.includes('resolved') || body.includes('addressed')) {
        metrics.fixedIssues++;
      }
    });
  }

  return metrics;
}

/**
 * Collect aggregate metrics for a period
 */
function collectPeriodMetrics(period) {
  const days = parseInt(period) || 30;
  const sinceDate = new Date();
  sinceDate.setDate(sinceDate.getDate() - days);

  // Get all merged PRs in period
  const prsJson = ghExec(`pr list --state merged --limit 100 --json number,mergedAt`);
  if (!prsJson) return null;

  const prs = JSON.parse(prsJson).filter(pr => {
    return new Date(pr.mergedAt) >= sinceDate;
  });

  const metrics = {
    period: `${days}d`,
    totalPRs: prs.length,
    avgReviewTime: 0,
    totalIssuesFound: 0,
    totalCriticalIssues: 0,
    fixRate: 0,
    prMetrics: []
  };

  // Collect metrics for each PR
  let totalTime = 0;
  let totalFixed = 0;
  let totalIssues = 0;

  prs.forEach(pr => {
    const prMetrics = collectPRMetrics(pr.number);
    metrics.prMetrics.push(prMetrics);

    if (prMetrics.reviewTime) {
      totalTime += prMetrics.reviewTime;
    }
    totalIssues += prMetrics.issuesFound;
    totalFixed += prMetrics.fixedIssues;
    metrics.totalCriticalIssues += prMetrics.criticalIssues;
  });

  metrics.avgReviewTime = prs.length > 0 ? Math.round(totalTime / prs.length) : 0;
  metrics.totalIssuesFound = totalIssues;
  metrics.fixRate = totalIssues > 0 ? Math.round((totalFixed / totalIssues) * 100) : 0;

  return metrics;
}

/**
 * Format metrics output
 */
function formatOutput(metrics, format) {
  if (format === 'json') {
    console.log(JSON.stringify(metrics, null, 2));
    return;
  }

  if (format === 'markdown') {
    if (metrics.prNumber) {
      // Single PR markdown
      console.log(`# PR #${metrics.prNumber} Review Metrics\n`);
      console.log(`**Title**: ${metrics.title}\n`);
      console.log(`**Created**: ${metrics.createdAt.toISOString()}`);
      if (metrics.mergedAt) {
        console.log(`**Merged**: ${metrics.mergedAt.toISOString()}`);
        console.log(`**Review Time**: ${metrics.reviewTime} minutes\n`);
      }
      console.log(`## Issue Detection\n`);
      console.log(`- Total Issues Found: ${metrics.issuesFound}`);
      console.log(`- Critical Issues: ${metrics.criticalIssues}`);
      console.log(`- Fixed Issues: ${metrics.fixedIssues}`);
    } else {
      // Period metrics markdown
      console.log(`# Review Metrics - ${metrics.period}\n`);
      console.log(`## Summary\n`);
      console.log(`- Total PRs Reviewed: ${metrics.totalPRs}`);
      console.log(`- Average Review Time: ${metrics.avgReviewTime} minutes`);
      console.log(`- Total Issues Found: ${metrics.totalIssuesFound}`);
      console.log(`- Critical Issues: ${metrics.totalCriticalIssues}`);
      console.log(`- Fix Rate: ${metrics.fixRate}%\n`);
    }
    return;
  }

  // Text format
  if (metrics.prNumber) {
    console.log(`\n📊 PR #${metrics.prNumber} Review Metrics\n`);
    console.log(`Title: ${metrics.title}`);
    console.log(`Created: ${metrics.createdAt.toLocaleString()}`);
    if (metrics.mergedAt) {
      console.log(`Merged: ${metrics.mergedAt.toLocaleString()}`);
      console.log(`Review Time: ${metrics.reviewTime} minutes`);
    }
    console.log(`\n📈 Issue Detection:`);
    console.log(`  Total Issues: ${metrics.issuesFound}`);
    console.log(`  Critical: ${metrics.criticalIssues}`);
    console.log(`  Fixed: ${metrics.fixedIssues}`);
    console.log('');
  } else {
    console.log(`\n📊 Review Metrics - Last ${metrics.period}\n`);
    console.log(`Total PRs Reviewed: ${metrics.totalPRs}`);
    console.log(`Average Review Time: ${metrics.avgReviewTime} minutes`);
    console.log(`\n📈 Issue Detection:`);
    console.log(`  Total Issues Found: ${metrics.totalIssuesFound}`);
    console.log(`  Critical Issues: ${metrics.totalCriticalIssues}`);
    console.log(`  Fix Rate: ${metrics.fixRate}%`);
    console.log('');
  }
}

/**
 * Export dashboard HTML
 */
function exportDashboardHTML(metrics) {
  const html = `
<!DOCTYPE html>
<html>
<head>
  <title>Review Metrics Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
    h1 { color: #333; }
    .metric-card { display: inline-block; margin: 10px; padding: 20px; background: #f9f9f9; border-radius: 4px; min-width: 200px; }
    .metric-value { font-size: 32px; font-weight: bold; color: #0066cc; }
    .metric-label { color: #666; margin-top: 5px; }
    table { width: 100%; border-collapse: collapse; margin-top: 20px; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #f0f0f0; }
  </style>
</head>
<body>
  <div class="container">
    <h1>📊 Review Metrics Dashboard</h1>
    <p>Period: ${metrics.period}</p>

    <div class="metric-card">
      <div class="metric-value">${metrics.totalPRs}</div>
      <div class="metric-label">Total PRs</div>
    </div>

    <div class="metric-card">
      <div class="metric-value">${metrics.avgReviewTime}m</div>
      <div class="metric-label">Avg Review Time</div>
    </div>

    <div class="metric-card">
      <div class="metric-value">${metrics.totalIssuesFound}</div>
      <div class="metric-label">Issues Found</div>
    </div>

    <div class="metric-card">
      <div class="metric-value">${metrics.fixRate}%</div>
      <div class="metric-label">Fix Rate</div>
    </div>

    <h2>Recent PRs</h2>
    <table>
      <thead>
        <tr>
          <th>PR</th>
          <th>Title</th>
          <th>Review Time</th>
          <th>Issues</th>
          <th>Critical</th>
        </tr>
      </thead>
      <tbody>
        ${metrics.prMetrics.slice(0, 10).map(pr => `
          <tr>
            <td>#${pr.prNumber}</td>
            <td>${pr.title}</td>
            <td>${pr.reviewTime || 'N/A'}m</td>
            <td>${pr.issuesFound}</td>
            <td>${pr.criticalIssues}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  </div>
</body>
</html>
`;

  const filename = `review-metrics-dashboard-${Date.now()}.html`;
  fs.writeFileSync(filename, html);
  console.log(`\n✅ Dashboard exported to: ${filename}\n`);
}

// Main execution
try {
  if (prNumber) {
    const metrics = collectPRMetrics(prNumber);
    formatOutput(metrics, format);
  } else {
    const metrics = collectPeriodMetrics(period);
    if (!metrics) {
      console.error('Error collecting metrics');
      process.exit(1);
    }
    formatOutput(metrics, format);

    if (exportDashboard) {
      exportDashboardHTML(metrics);
    }
  }
} catch (error) {
  console.error(`Error: ${error.message}`);
  process.exit(1);
}
