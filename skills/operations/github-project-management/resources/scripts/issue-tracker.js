#!/usr/bin/env node
/**
 * GitHub Issue Tracker
 *
 * Intelligent issue triage, decomposition, and swarm coordination for
 * automated issue management and progress tracking.
 *
 * Features:
 * - Automated issue triage with ML-based labeling
 * - Task decomposition into manageable subtasks
 * - Swarm agent assignment and coordination
 * - Progress tracking and automated updates
 * - Duplicate detection and linking
 *
 * Usage:
 *   node issue-tracker.js triage --repo owner/repo --auto-label
 *   node issue-tracker.js decompose --issue 456 --max-subtasks 10
 *   node issue-tracker.js progress --issue 456 --auto-update
 */

const { execSync } = require('child_process');
const fs = require('fs');

class IssueTracker {
  constructor(config = {}) {
    this.config = {
      repo: config.repo || process.env.GITHUB_REPOSITORY,
      autoLabel: config.autoLabel !== false,
      autoAssign: config.autoAssign !== false,
      ...config
    };
  }

  /**
   * Automated issue triage
   */
  async triage(options = {}) {
    console.log('🔍 Triaging GitHub issues...');

    const filter = options.filter || 'state:open no:label';

    try {
      // Get unlabeled issues
      const issues = JSON.parse(
        execSync(`gh issue list --repo ${this.config.repo} --json number,title,body,labels --limit 100`,
          { encoding: 'utf8' }
        )
      );

      const unlabeled = issues.filter(issue => issue.labels.length === 0);
      console.log(`Found ${unlabeled.length} unlabeled issues`);

      // Analyze and triage each issue
      for (const issue of unlabeled) {
        await this.triageIssue(issue);
      }

      console.log('✅ Triage completed');
      return { processed: unlabeled.length };
    } catch (error) {
      console.error('Triage failed:', error.message);
      throw error;
    }
  }

  /**
   * Triage a single issue
   */
  async triageIssue(issue) {
    console.log(`  Analyzing #${issue.number}: ${issue.title}`);

    // Auto-label based on content
    const labels = this.analyzeLabelingRules(issue);

    if (labels.length > 0) {
      try {
        execSync(`gh issue edit ${issue.number} --repo ${this.config.repo} --add-label "${labels.join(',')}"`,
          { stdio: 'pipe' }
        );
        console.log(`    ✓ Added labels: ${labels.join(', ')}`);
      } catch (error) {
        console.error(`    ✗ Failed to add labels: ${error.message}`);
      }
    }

    // Assign to swarm if ready
    if (labels.includes('swarm-ready')) {
      await this.assignToSwarm(issue.number);
    }
  }

  /**
   * Analyze issue content and suggest labels
   */
  analyzeLabelingRules(issue) {
    const content = `${issue.title} ${issue.body}`.toLowerCase();
    const labels = [];

    // Load labeling rules
    const rules = {
      'bug': ['bug', 'error', 'broken', 'crash', 'fail'],
      'enhancement': ['feature', 'implement', 'add', 'enhance'],
      'documentation': ['docs', 'documentation', 'readme'],
      'performance': ['slow', 'performance', 'optimize', 'speed'],
      'security': ['security', 'vulnerability', 'exploit', 'cve'],
      'swarm-ready': ['complex', 'large', 'refactor', 'integration']
    };

    for (const [label, keywords] of Object.entries(rules)) {
      if (keywords.some(keyword => content.includes(keyword))) {
        labels.push(label);
      }
    }

    return labels;
  }

  /**
   * Decompose issue into subtasks
   */
  async decompose(issueNumber, options = {}) {
    console.log(`📝 Decomposing issue #${issueNumber} into subtasks...`);

    const maxSubtasks = options.maxSubtasks || 10;

    try {
      // Get issue details
      const issue = JSON.parse(
        execSync(`gh issue view ${issueNumber} --repo ${this.config.repo} --json body`,
          { encoding: 'utf8' }
        )
      );

      // Decompose using swarm
      const subtasks = JSON.parse(
        execSync(`npx ruv-swarm github issue-decompose ${issueNumber} \
          --body '${issue.body}' \
          --max-subtasks ${maxSubtasks} \
          --assign-priorities`,
          { encoding: 'utf8' }
        )
      );

      // Update issue with checklist
      await this.updateIssueWithChecklist(issueNumber, issue.body, subtasks);

      // Create linked issues for high-priority subtasks
      await this.createLinkedIssues(issueNumber, subtasks);

      console.log(`✅ Created ${subtasks.tasks.length} subtasks`);
      return subtasks;
    } catch (error) {
      console.error('Decomposition failed:', error.message);
      throw error;
    }
  }

  /**
   * Update issue body with subtask checklist
   */
  async updateIssueWithChecklist(issueNumber, originalBody, subtasks) {
    const checklist = subtasks.tasks
      .map(task => `- [ ] ${task.description}`)
      .join('\n');

    const updatedBody = `${originalBody}\n\n## Subtasks\n${checklist}\n\n---\n🤖 Managed by AI swarm`;

    try {
      execSync(`gh issue edit ${issueNumber} --repo ${this.config.repo} --body "${updatedBody.replace(/"/g, '\\"')}"`,
        { stdio: 'pipe' }
      );
      console.log('  ✓ Updated issue with subtask checklist');
    } catch (error) {
      console.error('  ✗ Failed to update issue:', error.message);
    }
  }

  /**
   * Create linked issues for major subtasks
   */
  async createLinkedIssues(parentIssue, subtasks) {
    const highPriorityTasks = subtasks.tasks.filter(task => task.priority === 'high');

    for (const task of highPriorityTasks) {
      try {
        const result = execSync(`gh issue create --repo ${this.config.repo} \
          --title "${task.title}" \
          --body "${task.description}\n\nParent issue: #${parentIssue}" \
          --label "subtask"`,
          { encoding: 'utf8' }
        );
        console.log(`  ✓ Created linked issue: ${result.trim()}`);
      } catch (error) {
        console.error(`  ✗ Failed to create linked issue: ${error.message}`);
      }
    }
  }

  /**
   * Track issue progress and update
   */
  async progress(issueNumber, options = {}) {
    console.log(`📊 Tracking progress for issue #${issueNumber}...`);

    try {
      // Get current issue state
      const current = JSON.parse(
        execSync(`gh issue view ${issueNumber} --repo ${this.config.repo} --json body,labels`,
          { encoding: 'utf8' }
        )
      );

      // Get swarm progress
      const progress = JSON.parse(
        execSync(`npx ruv-swarm github issue-progress ${issueNumber}`,
          { encoding: 'utf8' }
        )
      );

      // Post progress summary as comment
      if (options.autoUpdate !== false) {
        await this.postProgressComment(issueNumber, progress);
      }

      // Update labels based on progress
      if (progress.completion === 100) {
        await this.updateProgressLabels(issueNumber, 'completed');
      } else if (progress.completion > 0) {
        await this.updateProgressLabels(issueNumber, 'in-progress');
      }

      console.log(`  Progress: ${progress.completion}% | ETA: ${progress.eta}`);
      return progress;
    } catch (error) {
      console.error('Progress tracking failed:', error.message);
      throw error;
    }
  }

  /**
   * Post progress summary as comment
   */
  async postProgressComment(issueNumber, progress) {
    const summary = `## 📊 Progress Update

**Completion**: ${progress.completion}%
**ETA**: ${progress.eta}

### Completed Tasks
${progress.completed.map(task => `- ✅ ${task}`).join('\n')}

### In Progress
${progress.in_progress.map(task => `- 🔄 ${task}`).join('\n')}

### Remaining
${progress.remaining.map(task => `- ⏳ ${task}`).join('\n')}

---
🤖 Automated update by swarm agent`;

    try {
      execSync(`gh issue comment ${issueNumber} --repo ${this.config.repo} --body "${summary.replace(/"/g, '\\"')}"`,
        { stdio: 'pipe' }
      );
      console.log('  ✓ Posted progress comment');
    } catch (error) {
      console.error('  ✗ Failed to post comment:', error.message);
    }
  }

  /**
   * Update issue labels based on progress
   */
  async updateProgressLabels(issueNumber, status) {
    try {
      const labelMap = {
        'in-progress': { add: 'in-progress', remove: 'pending' },
        'completed': { add: 'ready-for-review', remove: 'in-progress' }
      };

      const labels = labelMap[status];
      if (labels) {
        execSync(`gh issue edit ${issueNumber} --repo ${this.config.repo} \
          --add-label "${labels.add}" --remove-label "${labels.remove}"`,
          { stdio: 'pipe' }
        );
      }
    } catch (error) {
      // Labels might not exist
      console.log('  ℹ Label update skipped');
    }
  }

  /**
   * Assign issue to swarm
   */
  async assignToSwarm(issueNumber) {
    console.log(`  🐝 Assigning issue #${issueNumber} to swarm...`);

    try {
      execSync(`npx ruv-swarm github issue-init ${issueNumber} \
        --auto-decompose \
        --assign-agents`,
        { stdio: 'pipe' }
      );
      console.log('    ✓ Assigned to swarm');
    } catch (error) {
      console.error('    ✗ Swarm assignment failed:', error.message);
    }
  }
}

// CLI Interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const command = args[0];

  const tracker = new IssueTracker();

  const options = {};
  for (let i = 1; i < args.length; i += 2) {
    const key = args[i].replace(/^--/, '');
    const value = args[i + 1];
    options[key] = value === 'true' ? true : value === 'false' ? false : value;
  }

  switch (command) {
    case 'triage':
      tracker.triage(options);
      break;
    case 'decompose':
      tracker.decompose(options.issue, options);
      break;
    case 'progress':
      tracker.progress(options.issue, options);
      break;
    default:
      console.log(`
GitHub Issue Tracker

Commands:
  triage     Automated issue triage and labeling
  decompose  Break down issue into subtasks
  progress   Track and update issue progress

Options:
  --repo <owner/repo>    GitHub repository
  --issue <number>       Issue number
  --auto-label           Enable auto-labeling (triage)
  --max-subtasks <n>     Maximum subtasks (decompose)
  --auto-update          Auto-post updates (progress)

Examples:
  node issue-tracker.js triage --repo owner/repo --auto-label
  node issue-tracker.js decompose --issue 456 --max-subtasks 10
  node issue-tracker.js progress --issue 456 --auto-update
      `);
  }
}

module.exports = IssueTracker;
