#!/usr/bin/env node

/**
 * Flow Nexus Platform Health Monitor
 *
 * Monitors platform health, tracks credit usage, manages payments,
 * and provides system diagnostics via Flow Nexus MCP tools.
 *
 * Usage:
 *   node platform-health.js check [--detailed]
 *   node platform-health.js credits [--history]
 *   node platform-health.js payment-link <amount>
 *   node platform-health.js auto-refill <enable|disable> [--threshold N] [--amount N]
 *   node platform-health.js audit-log [--limit N]
 *   node platform-health.js user-stats <userId>
 *   node platform-health.js market-data
 */

const { execSync } = require('child_process');

class PlatformHealth {
  constructor() {
    this.mcpPrefix = 'mcp__flow-nexus__';
  }

  /**
   * Execute MCP tool and return parsed result
   */
  executeMCP(toolName, params = {}) {
    const command = `claude mcp call ${this.mcpPrefix}${toolName} '${JSON.stringify(params)}'`;

    try {
      const result = execSync(command, { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] });
      return JSON.parse(result);
    } catch (error) {
      console.error(`MCP call failed: ${error.message}`);
      if (error.stderr) {
        console.error(`Error details: ${error.stderr.toString()}`);
      }
      process.exit(1);
    }
  }

  /**
   * Check system health
   */
  checkHealth(detailed = false) {
    console.log('Checking Flow Nexus platform health...\n');

    const result = this.executeMCP('system_health');

    if (result.status) {
      console.log(`Overall Status: ${result.status === 'operational' ? '✓ Operational' : '⚠ Issues Detected'}`);

      if (result.components) {
        console.log('\nComponents:');
        Object.entries(result.components).forEach(([name, status]) => {
          const icon = status === 'operational' ? '✓' : '✗';
          console.log(`  ${icon} ${name}: ${status}`);
        });
      }

      if (detailed && result.metrics) {
        console.log('\nMetrics:');
        console.log(`  API Response Time: ${result.metrics.api_response_time}ms`);
        console.log(`  Active Sandboxes: ${result.metrics.active_sandboxes}`);
        console.log(`  Active Swarms: ${result.metrics.active_swarms}`);
        console.log(`  Total Users: ${result.metrics.total_users}`);
        console.log(`  Uptime: ${result.metrics.uptime}`);
      }

      if (result.incidents && result.incidents.length > 0) {
        console.log('\n⚠ Active Incidents:');
        result.incidents.forEach(incident => {
          console.log(`  - ${incident.title} (${incident.severity})`);
          console.log(`    ${incident.description}`);
        });
      }

      return result;
    } else {
      console.error('✗ Unable to fetch system health');
      process.exit(1);
    }
  }

  /**
   * Check credit balance and history
   */
  checkCredits(showHistory = false) {
    console.log('Checking credit balance...\n');

    const balance = this.executeMCP('check_balance');

    if (balance) {
      console.log(`Current Balance: ${balance.credits} credits`);
      console.log(`Last Updated: ${new Date(balance.last_updated).toLocaleString()}`);

      if (balance.auto_refill) {
        console.log('\nAuto-Refill: Enabled');
        console.log(`  Threshold: ${balance.auto_refill.threshold} credits`);
        console.log(`  Amount: $${balance.auto_refill.amount}`);
      } else {
        console.log('\nAuto-Refill: Disabled');
      }

      if (showHistory) {
        console.log('\nFetching transaction history...');
        const history = this.executeMCP('get_payment_history', { limit: 20 });

        if (history.transactions && history.transactions.length > 0) {
          console.log('\nRecent Transactions:');
          history.transactions.forEach(tx => {
            const sign = tx.type === 'credit' ? '+' : '-';
            console.log(`  ${new Date(tx.timestamp).toLocaleDateString()} | ${sign}${tx.amount} credits | ${tx.description}`);
          });
        } else {
          console.log('\nNo transaction history found');
        }
      }

      return balance;
    } else {
      console.error('✗ Unable to fetch balance');
      process.exit(1);
    }
  }

  /**
   * Create payment link for purchasing credits
   */
  createPaymentLink(amount) {
    if (amount < 10) {
      console.error('Error: Minimum payment amount is $10');
      process.exit(1);
    }

    console.log(`Creating payment link for $${amount}...\n`);

    const result = this.executeMCP('create_payment_link', { amount: parseFloat(amount) });

    if (result.payment_url) {
      console.log('✓ Payment link created successfully!\n');
      console.log(`Amount: $${amount}`);
      console.log(`Credits: ${result.credits_amount || Math.floor(amount * 100)}`);
      console.log(`\nPayment URL:\n${result.payment_url}`);
      console.log('\nOpen this URL in your browser to complete payment');

      return result;
    } else {
      console.error('✗ Failed to create payment link:', result.error);
      process.exit(1);
    }
  }

  /**
   * Configure auto-refill settings
   */
  configureAutoRefill(action, options = {}) {
    const enabled = action === 'enable';

    console.log(`${enabled ? 'Enabling' : 'Disabling'} auto-refill...`);

    const params = { enabled };

    if (enabled) {
      if (!options.threshold || !options.amount) {
        console.error('Error: enable requires --threshold and --amount');
        process.exit(1);
      }
      params.threshold = parseInt(options.threshold);
      params.amount = parseInt(options.amount);
    }

    const result = this.executeMCP('configure_auto_refill', params);

    if (result.success) {
      console.log(`\n✓ Auto-refill ${enabled ? 'enabled' : 'disabled'}`);

      if (enabled) {
        console.log(`Threshold: ${params.threshold} credits`);
        console.log(`Refill Amount: $${params.amount}`);
      }

      return result;
    } else {
      console.error('✗ Configuration failed:', result.error);
      process.exit(1);
    }
  }

  /**
   * Get audit log
   */
  getAuditLog(limit = 50) {
    console.log(`Fetching audit log (last ${limit} entries)...\n`);

    const result = this.executeMCP('audit_log', { limit: parseInt(limit) });

    if (result.logs && result.logs.length > 0) {
      console.log(`Found ${result.logs.length} log entries:\n`);

      result.logs.forEach(log => {
        console.log(`[${new Date(log.timestamp).toLocaleString()}] ${log.action}`);
        console.log(`  User: ${log.user_id || 'System'}`);
        console.log(`  Resource: ${log.resource_type}/${log.resource_id}`);
        if (log.details) {
          console.log(`  Details: ${log.details}`);
        }
        console.log('  ---');
      });

      return result.logs;
    } else {
      console.log('No audit logs found');
      return [];
    }
  }

  /**
   * Get user statistics
   */
  getUserStats(userId) {
    console.log(`Fetching statistics for user: ${userId}\n`);

    const result = this.executeMCP('user_stats', { user_id: userId });

    if (result.stats) {
      const s = result.stats;
      console.log('User Statistics:');
      console.log(`  User ID: ${userId}`);
      console.log(`  Tier: ${s.tier}`);
      console.log(`  Member Since: ${new Date(s.member_since).toLocaleDateString()}`);

      console.log('\nUsage:');
      console.log(`  Sandboxes Created: ${s.sandboxes_created || 0}`);
      console.log(`  Active Sandboxes: ${s.active_sandboxes || 0}`);
      console.log(`  Swarms Created: ${s.swarms_created || 0}`);
      console.log(`  Deployments: ${s.deployments || 0}`);
      console.log(`  API Calls: ${s.api_calls || 0}`);

      console.log('\nCredits:');
      console.log(`  Total Earned: ${s.credits_earned || 0}`);
      console.log(`  Total Spent: ${s.credits_spent || 0}`);
      console.log(`  Current Balance: ${s.credits_balance || 0}`);

      if (s.achievements && s.achievements.length > 0) {
        console.log('\nAchievements:');
        s.achievements.forEach(achievement => {
          console.log(`  ⭐ ${achievement.name} - ${achievement.description}`);
        });
      }

      return s;
    } else {
      console.error('✗ Unable to fetch user statistics');
      process.exit(1);
    }
  }

  /**
   * Get market statistics
   */
  getMarketData() {
    console.log('Fetching market statistics...\n');

    const result = this.executeMCP('market_data');

    if (result.market) {
      const m = result.market;
      console.log('Market Statistics:');
      console.log(`  Total Apps: ${m.total_apps || 0}`);
      console.log(`  Total Templates: ${m.total_templates || 0}`);
      console.log(`  Total Downloads: ${m.total_downloads || 0}`);
      console.log(`  Active Deployments: ${m.active_deployments || 0}`);

      if (m.top_categories) {
        console.log('\nTop Categories:');
        m.top_categories.forEach(cat => {
          console.log(`  - ${cat.name}: ${cat.count} apps`);
        });
      }

      if (m.trending_apps) {
        console.log('\nTrending Apps:');
        m.trending_apps.slice(0, 5).forEach((app, i) => {
          console.log(`  ${i + 1}. ${app.name} (${app.downloads} downloads)`);
        });
      }

      return m;
    } else {
      console.error('✗ Unable to fetch market data');
      process.exit(1);
    }
  }
}

// CLI Interface
function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log(`
Flow Nexus Platform Health Monitor

Usage:
  node platform-health.js check [--detailed]
  node platform-health.js credits [--history]
  node platform-health.js payment-link <amount>
  node platform-health.js auto-refill <enable|disable> [--threshold N] [--amount N]
  node platform-health.js audit-log [--limit N]
  node platform-health.js user-stats <userId>
  node platform-health.js market-data

Examples:
  node platform-health.js check --detailed
  node platform-health.js credits --history
  node platform-health.js payment-link 50
  node platform-health.js auto-refill enable --threshold 100 --amount 50
  node platform-health.js audit-log --limit 100
    `);
    process.exit(0);
  }

  const monitor = new PlatformHealth();
  const command = args[0];

  try {
    switch (command) {
      case 'check':
        monitor.checkHealth(args.includes('--detailed'));
        break;

      case 'credits':
        monitor.checkCredits(args.includes('--history'));
        break;

      case 'payment-link':
        if (args.length < 2) {
          console.error('Error: payment-link requires <amount>');
          process.exit(1);
        }
        monitor.createPaymentLink(args[1]);
        break;

      case 'auto-refill': {
        if (args.length < 2) {
          console.error('Error: auto-refill requires <enable|disable>');
          process.exit(1);
        }

        const action = args[1];
        if (!['enable', 'disable'].includes(action)) {
          console.error('Error: action must be enable or disable');
          process.exit(1);
        }

        const options = {};
        const thresholdIdx = args.indexOf('--threshold');
        const amountIdx = args.indexOf('--amount');

        if (thresholdIdx !== -1 && args[thresholdIdx + 1]) {
          options.threshold = args[thresholdIdx + 1];
        }
        if (amountIdx !== -1 && args[amountIdx + 1]) {
          options.amount = args[amountIdx + 1];
        }

        monitor.configureAutoRefill(action, options);
        break;
      }

      case 'audit-log': {
        const limitIdx = args.indexOf('--limit');
        const limit = limitIdx !== -1 && args[limitIdx + 1] ? args[limitIdx + 1] : 50;
        monitor.getAuditLog(limit);
        break;
      }

      case 'user-stats':
        if (args.length < 2) {
          console.error('Error: user-stats requires <userId>');
          process.exit(1);
        }
        monitor.getUserStats(args[1]);
        break;

      case 'market-data':
        monitor.getMarketData();
        break;

      default:
        console.error(`Unknown command: ${command}`);
        console.error('Run without arguments to see usage');
        process.exit(1);
    }
  } catch (error) {
    console.error(`Fatal error: ${error.message}`);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

module.exports = PlatformHealth;
