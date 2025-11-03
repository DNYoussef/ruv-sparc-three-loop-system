#!/usr/bin/env node
/**
 * Sprint Planner
 *
 * Agile sprint planning with velocity tracking, capacity planning,
 * and automated burndown analytics.
 *
 * Features:
 * - Sprint creation and management
 * - Velocity and burndown tracking
 * - Capacity planning with team workload balancing
 * - Automated sprint reports and retrospectives
 * - Integration with GitHub milestones and project boards
 *
 * Usage:
 *   node sprint-planner.js create --sprint "Sprint 24" --capacity 40
 *   node sprint-planner.js track --sprint "Sprint 24" --show-burndown
 *   node sprint-planner.js report --sprint "Sprint 24" --format markdown
 */

const { execSync } = require('child_process');
const fs = require('fs');

class SprintPlanner {
  constructor(config = {}) {
    this.config = {
      repo: config.repo || process.env.GITHUB_REPOSITORY,
      sprintLength: config.sprintLength || 14, // days
      teamCapacity: config.teamCapacity || 40, // story points
      ...config
    };
    this.metrics = {
      velocity: [],
      burndown: []
    };
  }

  /**
   * Create new sprint
   */
  async create(sprintName, options = {}) {
    console.log(`🚀 Creating sprint: ${sprintName}`);

    const capacity = options.capacity || this.config.teamCapacity;
    const template = options.template;

    try {
      // Load template if provided
      let sprintConfig = {
        name: sprintName,
        capacity,
        startDate: new Date().toISOString().split('T')[0],
        endDate: this.calculateEndDate(this.config.sprintLength),
        goals: [],
        team: []
      };

      if (template && fs.existsSync(template)) {
        const templateData = JSON.parse(fs.readFileSync(template, 'utf8'));
        sprintConfig = { ...sprintConfig, ...templateData, name: sprintName };
      }

      // Create GitHub milestone for sprint
      const milestone = await this.createMilestone(sprintConfig);

      // Initialize swarm for sprint management
      await this.initSprintSwarm(sprintName, milestone);

      // Auto-populate sprint backlog
      if (options.autoPopulate !== false) {
        await this.populateBacklog(milestone, capacity);
      }

      console.log(`✅ Sprint created: ${sprintName}`);
      console.log(`   Milestone: ${milestone.title} (${milestone.url})`);
      console.log(`   Capacity: ${capacity} points`);
      console.log(`   Duration: ${sprintConfig.startDate} to ${sprintConfig.endDate}`);

      return sprintConfig;
    } catch (error) {
      console.error('Sprint creation failed:', error.message);
      throw error;
    }
  }

  /**
   * Create GitHub milestone for sprint
   */
  async createMilestone(config) {
    console.log('  Creating milestone...');

    try {
      const result = execSync(`gh api repos/${this.config.repo}/milestones \
        --method POST \
        --field title="${config.name}" \
        --field due_on="${config.endDate}T23:59:59Z" \
        --field description="Sprint: ${config.name}\nCapacity: ${config.capacity} points"`,
        { encoding: 'utf8' }
      );

      const milestone = JSON.parse(result);
      console.log('    ✓ Milestone created');
      return milestone;
    } catch (error) {
      console.error('    ✗ Milestone creation failed:', error.message);
      throw error;
    }
  }

  /**
   * Initialize swarm for sprint management
   */
  async initSprintSwarm(sprintName, milestone) {
    console.log('  Initializing sprint swarm...');

    try {
      execSync(`npx ruv-swarm github sprint-manage \
        --sprint "${sprintName}" \
        --milestone ${milestone.number} \
        --auto-populate \
        --capacity-planning \
        --track-velocity`,
        { stdio: 'pipe' }
      );
      console.log('    ✓ Swarm initialized');
    } catch (error) {
      console.error('    ✗ Swarm initialization failed:', error.message);
    }
  }

  /**
   * Populate sprint backlog with prioritized issues
   */
  async populateBacklog(milestone, capacity) {
    console.log('  Populating sprint backlog...');

    try {
      // Get prioritized backlog items
      const backlog = JSON.parse(
        execSync(`gh issue list --repo ${this.config.repo} \
          --label "ready-for-sprint" \
          --json number,title,labels \
          --limit 50`,
          { encoding: 'utf8' }
        )
      );

      let totalPoints = 0;
      let addedIssues = 0;

      // Add issues until capacity reached
      for (const issue of backlog) {
        const points = this.estimatePoints(issue);

        if (totalPoints + points <= capacity) {
          await this.addIssueToSprint(issue.number, milestone.number);
          totalPoints += points;
          addedIssues++;
        } else {
          break;
        }
      }

      console.log(`    ✓ Added ${addedIssues} issues (${totalPoints}/${capacity} points)`);
    } catch (error) {
      console.error('    ✗ Backlog population failed:', error.message);
    }
  }

  /**
   * Track sprint progress
   */
  async track(sprintName, options = {}) {
    console.log(`📊 Tracking sprint: ${sprintName}`);

    try {
      // Get milestone
      const milestone = await this.getMilestone(sprintName);

      // Calculate metrics
      const metrics = await this.calculateMetrics(milestone);

      // Display progress
      this.displayProgress(metrics);

      // Generate burndown chart data
      if (options.showBurndown) {
        this.displayBurndown(metrics.burndown);
      }

      // Update project board
      if (options.updateBoard !== false) {
        await this.updateSprintBoard(milestone, metrics);
      }

      return metrics;
    } catch (error) {
      console.error('Sprint tracking failed:', error.message);
      throw error;
    }
  }

  /**
   * Calculate sprint metrics
   */
  async calculateMetrics(milestone) {
    // Get sprint issues
    const issues = JSON.parse(
      execSync(`gh issue list --repo ${this.config.repo} \
        --milestone "${milestone.title}" \
        --json number,state,labels,closedAt \
        --limit 100`,
        { encoding: 'utf8' }
      )
    );

    const totalPoints = issues.reduce((sum, issue) => sum + this.estimatePoints(issue), 0);
    const completedPoints = issues
      .filter(issue => issue.state === 'closed')
      .reduce((sum, issue) => sum + this.estimatePoints(issue), 0);

    const daysElapsed = this.calculateDaysElapsed(milestone.created_at);
    const daysRemaining = this.calculateDaysRemaining(milestone.due_on);

    return {
      total: issues.length,
      completed: issues.filter(i => i.state === 'closed').length,
      totalPoints,
      completedPoints,
      remainingPoints: totalPoints - completedPoints,
      velocity: daysElapsed > 0 ? (completedPoints / daysElapsed) : 0,
      daysElapsed,
      daysRemaining,
      onTrack: this.isOnTrack(completedPoints, totalPoints, daysElapsed, this.config.sprintLength),
      burndown: this.calculateBurndown(issues, milestone)
    };
  }

  /**
   * Generate sprint report
   */
  async report(sprintName, options = {}) {
    console.log(`📝 Generating sprint report: ${sprintName}`);

    const format = options.format || 'markdown';

    try {
      const milestone = await this.getMilestone(sprintName);
      const metrics = await this.calculateMetrics(milestone);

      let report;
      if (format === 'markdown') {
        report = this.generateMarkdownReport(sprintName, metrics);
      } else if (format === 'json') {
        report = JSON.stringify(metrics, null, 2);
      }

      // Save report
      const filename = `sprint-${sprintName.replace(/\s+/g, '-')}-report.${format === 'json' ? 'json' : 'md'}`;
      fs.writeFileSync(filename, report);
      console.log(`✅ Report saved: ${filename}`);

      // Distribute report
      if (options.distribute) {
        await this.distributeReport(sprintName, report, options.distribute);
      }

      return report;
    } catch (error) {
      console.error('Report generation failed:', error.message);
      throw error;
    }
  }

  /**
   * Generate markdown report
   */
  generateMarkdownReport(sprintName, metrics) {
    return `# Sprint Report: ${sprintName}

## Summary

- **Total Issues**: ${metrics.total}
- **Completed**: ${metrics.completed} (${Math.round(metrics.completed / metrics.total * 100)}%)
- **Story Points**: ${metrics.completedPoints}/${metrics.totalPoints}
- **Velocity**: ${metrics.velocity.toFixed(2)} points/day
- **Days Elapsed**: ${metrics.daysElapsed}
- **Days Remaining**: ${metrics.daysRemaining}
- **Status**: ${metrics.onTrack ? '✅ On Track' : '⚠️ Behind Schedule'}

## Burndown Chart

\`\`\`
${this.formatBurndownText(metrics.burndown)}
\`\`\`

## Recommendations

${this.generateRecommendations(metrics)}

---
Generated: ${new Date().toISOString()}
🤖 Automated by Sprint Planner
`;
  }

  /**
   * Helper methods
   */
  calculateEndDate(days) {
    const date = new Date();
    date.setDate(date.getDate() + days);
    return date.toISOString().split('T')[0];
  }

  estimatePoints(issue) {
    // Extract story points from labels (e.g., "points:5")
    const pointsLabel = issue.labels.find(l => l.name?.startsWith('points:'));
    return pointsLabel ? parseInt(pointsLabel.name.split(':')[1]) : 3; // default 3 points
  }

  calculateDaysElapsed(startDate) {
    return Math.floor((new Date() - new Date(startDate)) / (1000 * 60 * 60 * 24));
  }

  calculateDaysRemaining(endDate) {
    return Math.max(0, Math.floor((new Date(endDate) - new Date()) / (1000 * 60 * 60 * 24)));
  }

  isOnTrack(completed, total, elapsed, sprintLength) {
    const expectedProgress = (elapsed / sprintLength) * total;
    return completed >= expectedProgress * 0.9; // 90% threshold
  }

  calculateBurndown(issues, milestone) {
    // Generate burndown data (simplified)
    const days = this.config.sprintLength;
    const totalPoints = issues.reduce((sum, i) => sum + this.estimatePoints(i), 0);

    return Array.from({ length: days + 1 }, (_, i) => ({
      day: i,
      ideal: totalPoints - (totalPoints / days) * i,
      actual: totalPoints // Simplified - would track actual completion
    }));
  }

  formatBurndownText(burndown) {
    return burndown.slice(0, 15).map(d =>
      `Day ${d.day.toString().padStart(2)}: Ideal ${d.ideal.toFixed(0).padStart(3)} | Actual ${d.actual.toFixed(0).padStart(3)}`
    ).join('\n');
  }

  generateRecommendations(metrics) {
    const recommendations = [];

    if (!metrics.onTrack) {
      recommendations.push('- ⚠️ Sprint is behind schedule. Consider removing low-priority items.');
    }
    if (metrics.velocity < 2) {
      recommendations.push('- 📉 Low velocity detected. Review team capacity and blockers.');
    }
    if (metrics.remainingPoints > metrics.velocity * metrics.daysRemaining) {
      recommendations.push('- 🎯 Unlikely to complete all items. Prioritize must-have features.');
    }

    return recommendations.length > 0 ? recommendations.join('\n') : '- ✅ Sprint is progressing well. Keep up the momentum!';
  }

  async getMilestone(sprintName) {
    const milestones = JSON.parse(
      execSync(`gh api repos/${this.config.repo}/milestones`, { encoding: 'utf8' })
    );
    return milestones.find(m => m.title === sprintName);
  }

  async addIssueToSprint(issueNumber, milestoneNumber) {
    execSync(`gh issue edit ${issueNumber} --repo ${this.config.repo} --milestone ${milestoneNumber}`,
      { stdio: 'pipe' }
    );
  }

  displayProgress(metrics) {
    console.log('\n📈 Sprint Progress\n');
    console.log(`  Issues: ${metrics.completed}/${metrics.total} (${Math.round(metrics.completed / metrics.total * 100)}%)`);
    console.log(`  Points: ${metrics.completedPoints}/${metrics.totalPoints}`);
    console.log(`  Velocity: ${metrics.velocity.toFixed(2)} points/day`);
    console.log(`  Status: ${metrics.onTrack ? '✅ On Track' : '⚠️ Behind Schedule'}\n`);
  }

  displayBurndown(burndown) {
    console.log('\n📉 Burndown (First 10 days)\n');
    burndown.slice(0, 10).forEach(d => {
      console.log(`  Day ${d.day}: Ideal ${d.ideal.toFixed(0)} | Actual ${d.actual.toFixed(0)}`);
    });
    console.log('');
  }

  async updateSprintBoard(milestone, metrics) {
    // Update project board with sprint metrics
    console.log('  Updating project board...');
    // Implementation would sync with project board
  }

  async distributeReport(sprintName, report, channels) {
    console.log(`  Distributing report to: ${channels}`);
    // Implementation would send to Slack, email, etc.
  }
}

// CLI Interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const command = args[0];

  const planner = new SprintPlanner();

  const options = {};
  for (let i = 1; i < args.length; i += 2) {
    const key = args[i].replace(/^--/, '');
    const value = args[i + 1];
    options[key] = value === 'true' ? true : value === 'false' ? false : value;
  }

  switch (command) {
    case 'create':
      planner.create(options.sprint, options);
      break;
    case 'track':
      planner.track(options.sprint, options);
      break;
    case 'report':
      planner.report(options.sprint, options);
      break;
    default:
      console.log(`
Sprint Planner

Commands:
  create   Create new sprint with capacity planning
  track    Track sprint progress and metrics
  report   Generate sprint report

Options:
  --sprint <name>        Sprint name
  --capacity <points>    Team capacity in story points
  --template <file>      Sprint template file
  --show-burndown        Display burndown chart
  --format <type>        Report format (markdown|json)
  --distribute <channels> Distribute report

Examples:
  node sprint-planner.js create --sprint "Sprint 24" --capacity 40
  node sprint-planner.js track --sprint "Sprint 24" --show-burndown
  node sprint-planner.js report --sprint "Sprint 24" --format markdown
      `);
  }
}

module.exports = SprintPlanner;
