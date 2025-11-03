#!/usr/bin/env node
/**
 * GitHub PR Analysis Script
 *
 * Analyzes pull request complexity and recommends appropriate review strategies
 * based on PR size, file types, and changed areas.
 *
 * Usage:
 *   node pr-analysis.js <pr-number>
 *   node pr-analysis.js 123 --json
 *   node pr-analysis.js 123 --detailed
 */

const { execSync } = require('child_process');
const fs = require('fs');

// Parse command line arguments
const args = process.argv.slice(2);
const prNumber = args[0];
const jsonOutput = args.includes('--json');
const detailed = args.includes('--detailed');

if (!prNumber || prNumber === '--help') {
  console.log('Usage: node pr-analysis.js <pr-number> [--json] [--detailed]');
  console.log('');
  console.log('Options:');
  console.log('  --json      Output results in JSON format');
  console.log('  --detailed  Include detailed file-by-file analysis');
  process.exit(prNumber ? 0 : 1);
}

/**
 * Execute GitHub CLI command and parse JSON output
 */
function ghExec(command) {
  try {
    const result = execSync(`gh ${command}`, { encoding: 'utf-8' });
    return JSON.parse(result);
  } catch (error) {
    console.error(`Error executing gh command: ${error.message}`);
    process.exit(1);
  }
}

/**
 * Analyze PR complexity and recommend review strategy
 */
function analyzePR(prNumber) {
  // Get PR data with gh CLI
  const pr = ghExec(`pr view ${prNumber} --json files,additions,deletions,title,body,labels,author`);

  const analysis = {
    prNumber: parseInt(prNumber),
    title: pr.title,
    author: pr.author?.login || 'unknown',
    complexity: 'unknown',
    riskLevel: 'unknown',
    recommendedTopology: 'mesh',
    recommendedAgents: [],
    requiredReviewers: 0,
    estimatedReviewTime: 0,
    fileStats: {
      total: pr.files?.length || 0,
      additions: pr.additions || 0,
      deletions: pr.deletions || 0,
      totalChanges: (pr.additions || 0) + (pr.deletions || 0)
    },
    fileTypes: {},
    criticalAreas: [],
    labels: pr.labels?.map(l => l.name) || []
  };

  // Analyze file types
  if (pr.files) {
    pr.files.forEach(file => {
      const ext = file.path.split('.').pop();
      analysis.fileTypes[ext] = (analysis.fileTypes[ext] || 0) + 1;

      // Identify critical areas
      if (file.path.match(/auth|security|payment|admin/i)) {
        analysis.criticalAreas.push({ path: file.path, reason: 'security-sensitive' });
      }
      if (file.path.match(/api|database|migration/i)) {
        analysis.criticalAreas.push({ path: file.path, reason: 'infrastructure' });
      }
      if (file.path.match(/config|\.env|secret/i)) {
        analysis.criticalAreas.push({ path: file.path, reason: 'configuration' });
      }
    });
  }

  // Determine complexity
  const totalChanges = analysis.fileStats.totalChanges;
  if (totalChanges < 100) {
    analysis.complexity = 'simple';
    analysis.estimatedReviewTime = 15; // minutes
  } else if (totalChanges < 500) {
    analysis.complexity = 'moderate';
    analysis.estimatedReviewTime = 45;
  } else if (totalChanges < 1000) {
    analysis.complexity = 'complex';
    analysis.estimatedReviewTime = 90;
  } else {
    analysis.complexity = 'very-complex';
    analysis.estimatedReviewTime = 180;
  }

  // Determine risk level
  if (analysis.criticalAreas.length > 3) {
    analysis.riskLevel = 'high';
  } else if (analysis.criticalAreas.length > 0) {
    analysis.riskLevel = 'medium';
  } else {
    analysis.riskLevel = 'low';
  }

  // Recommend topology based on PR size
  if (analysis.fileStats.total < 5) {
    analysis.recommendedTopology = 'ring';
    analysis.requiredReviewers = 2;
  } else if (analysis.fileStats.total < 15) {
    analysis.recommendedTopology = 'mesh';
    analysis.requiredReviewers = 3;
  } else {
    analysis.recommendedTopology = 'hierarchical';
    analysis.requiredReviewers = 5;
  }

  // Recommend agents based on file types and critical areas
  analysis.recommendedAgents = determineAgents(analysis);

  // Add detailed file analysis if requested
  if (detailed && pr.files) {
    analysis.fileDetails = pr.files.map(file => ({
      path: file.path,
      additions: file.additions,
      deletions: file.deletions,
      changes: file.additions + file.deletions
    })).sort((a, b) => b.changes - a.changes);
  }

  return analysis;
}

/**
 * Determine which review agents should be assigned
 */
function determineAgents(analysis) {
  const agents = new Set(['security', 'style']); // Always include these

  // Security-sensitive files
  if (analysis.criticalAreas.some(a => a.reason === 'security-sensitive')) {
    agents.add('security');
    agents.add('authentication');
    agents.add('audit');
  }

  // Infrastructure files
  if (analysis.criticalAreas.some(a => a.reason === 'infrastructure')) {
    agents.add('performance');
    agents.add('database');
  }

  // Configuration files
  if (analysis.criticalAreas.some(a => a.reason === 'configuration')) {
    agents.add('security');
  }

  // Frontend files
  if (analysis.fileTypes.jsx || analysis.fileTypes.tsx || analysis.fileTypes.vue) {
    agents.add('accessibility');
    agents.add('i18n');
  }

  // Backend files
  if (analysis.fileTypes.js || analysis.fileTypes.ts || analysis.fileTypes.py) {
    agents.add('architecture');
  }

  // Testing files
  if (analysis.fileTypes.test || analysis.fileTypes.spec) {
    agents.add('tester');
  }

  // Documentation
  if (analysis.fileTypes.md || analysis.fileTypes.txt) {
    agents.add('docs');
  }

  return Array.from(agents);
}

/**
 * Format output based on flags
 */
function formatOutput(analysis) {
  if (jsonOutput) {
    console.log(JSON.stringify(analysis, null, 2));
    return;
  }

  console.log(`\n📊 PR Analysis: #${analysis.prNumber} - ${analysis.title}\n`);
  console.log(`Author: ${analysis.author}`);
  console.log(`Complexity: ${analysis.complexity.toUpperCase()}`);
  console.log(`Risk Level: ${analysis.riskLevel.toUpperCase()}`);
  console.log(`\n📈 File Statistics:`);
  console.log(`  Total Files: ${analysis.fileStats.total}`);
  console.log(`  Additions: +${analysis.fileStats.additions}`);
  console.log(`  Deletions: -${analysis.fileStats.deletions}`);
  console.log(`  Total Changes: ${analysis.fileStats.totalChanges}`);

  console.log(`\n🔧 File Types:`);
  Object.entries(analysis.fileTypes).forEach(([type, count]) => {
    console.log(`  ${type}: ${count} file(s)`);
  });

  if (analysis.criticalAreas.length > 0) {
    console.log(`\n⚠️  Critical Areas (${analysis.criticalAreas.length}):`);
    analysis.criticalAreas.slice(0, 5).forEach(area => {
      console.log(`  - ${area.path} (${area.reason})`);
    });
    if (analysis.criticalAreas.length > 5) {
      console.log(`  ... and ${analysis.criticalAreas.length - 5} more`);
    }
  }

  console.log(`\n🎯 Recommendations:`);
  console.log(`  Topology: ${analysis.recommendedTopology}`);
  console.log(`  Required Reviewers: ${analysis.requiredReviewers}`);
  console.log(`  Estimated Review Time: ${analysis.estimatedReviewTime} minutes`);
  console.log(`  Recommended Agents: ${analysis.recommendedAgents.join(', ')}`);

  if (analysis.labels.length > 0) {
    console.log(`\n🏷️  Labels: ${analysis.labels.join(', ')}`);
  }

  if (detailed && analysis.fileDetails) {
    console.log(`\n📁 Top Changed Files:`);
    analysis.fileDetails.slice(0, 10).forEach((file, idx) => {
      console.log(`  ${idx + 1}. ${file.path} (+${file.additions}/-${file.deletions})`);
    });
  }

  console.log('');
}

// Main execution
try {
  const analysis = analyzePR(prNumber);
  formatOutput(analysis);
} catch (error) {
  console.error(`Error analyzing PR: ${error.message}`);
  process.exit(1);
}
