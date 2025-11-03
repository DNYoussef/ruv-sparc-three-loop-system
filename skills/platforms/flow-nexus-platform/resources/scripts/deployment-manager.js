#!/usr/bin/env node

/**
 * Flow Nexus Deployment Manager
 *
 * Manages application deployment workflows including template browsing,
 * deployment, and application publishing via Flow Nexus MCP tools.
 *
 * Usage:
 *   node deployment-manager.js list-templates [--category <cat>] [--featured]
 *   node deployment-manager.js template-info <templateName>
 *   node deployment-manager.js deploy <templateName> <deploymentName> [--var KEY=VALUE]
 *   node deployment-manager.js publish <name> <description> <category> <sourceFile> [--tags tag1,tag2]
 *   node deployment-manager.js app-info <appId>
 *   node deployment-manager.js search <query> [--category <cat>]
 *   node deployment-manager.js analytics <appId> [--timeframe 24h|7d|30d|90d]
 */

const { execSync } = require('child_process');
const fs = require('fs');

class DeploymentManager {
  constructor() {
    this.mcpPrefix = 'mcp__flow-nexus__';
    this.validCategories = [
      'web-api', 'frontend', 'full-stack', 'cli-tools',
      'data-processing', 'ml-models', 'blockchain', 'mobile', 'backend'
    ];
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
   * List available templates
   */
  listTemplates(options = {}) {
    console.log('Fetching available templates...');

    const params = { limit: options.limit || 50 };
    if (options.category) params.category = options.category;
    if (options.featured) params.featured = true;

    const result = this.executeMCP('template_list', params);

    if (result.templates && result.templates.length > 0) {
      console.log(`\nFound ${result.templates.length} template(s):\n`);

      result.templates.forEach(template => {
        console.log(`  Name: ${template.name}`);
        console.log(`  Category: ${template.category}`);
        console.log(`  Description: ${template.description}`);
        if (template.tags) {
          console.log(`  Tags: ${template.tags.join(', ')}`);
        }
        if (template.featured) {
          console.log(`  ⭐ Featured`);
        }
        console.log('  ---');
      });

      return result.templates;
    } else {
      console.log('No templates found');
      return [];
    }
  }

  /**
   * Get template information
   */
  templateInfo(templateName) {
    console.log(`Fetching template info: ${templateName}`);

    const result = this.executeMCP('template_get', { template_name: templateName });

    if (result.template) {
      const t = result.template;
      console.log('\nTemplate Details:');
      console.log(`  Name: ${t.name}`);
      console.log(`  Category: ${t.category}`);
      console.log(`  Version: ${t.version || 'N/A'}`);
      console.log(`  Description: ${t.description}`);

      if (t.tags) {
        console.log(`  Tags: ${t.tags.join(', ')}`);
      }

      if (t.variables) {
        console.log('\n  Required Variables:');
        Object.entries(t.variables).forEach(([key, desc]) => {
          console.log(`    - ${key}: ${desc}`);
        });
      }

      if (t.env_vars_example) {
        console.log('\n  Environment Variables:');
        Object.entries(t.env_vars_example).forEach(([key, val]) => {
          console.log(`    - ${key}=${val}`);
        });
      }

      return t;
    } else {
      console.error('✗ Template not found');
      process.exit(1);
    }
  }

  /**
   * Deploy template
   */
  deploy(templateName, deploymentName, options = {}) {
    console.log(`Deploying template: ${templateName} as ${deploymentName}`);

    const params = {
      template_name: templateName,
      deployment_name: deploymentName
    };

    if (options.variables) {
      params.variables = options.variables;
    }

    if (options.env_vars) {
      params.env_vars = options.env_vars;
    }

    const result = this.executeMCP('template_deploy', params);

    if (result.success) {
      console.log('\n✓ Deployment successful!');
      console.log(`Deployment ID: ${result.deployment_id}`);
      console.log(`URL: ${result.url || 'N/A'}`);

      if (result.sandbox_id) {
        console.log(`Sandbox ID: ${result.sandbox_id}`);
      }

      return result;
    } else {
      console.error('✗ Deployment failed:', result.error);
      process.exit(1);
    }
  }

  /**
   * Publish application to store
   */
  publish(name, description, category, sourceFile, options = {}) {
    if (!this.validCategories.includes(category)) {
      console.error(`Invalid category: ${category}`);
      console.error(`Valid categories: ${this.validCategories.join(', ')}`);
      process.exit(1);
    }

    if (!fs.existsSync(sourceFile)) {
      console.error(`Source file not found: ${sourceFile}`);
      process.exit(1);
    }

    console.log(`Publishing application: ${name}`);

    const sourceCode = fs.readFileSync(sourceFile, 'utf-8');

    const params = {
      name,
      description,
      category,
      source_code: sourceCode,
      version: options.version || '1.0.0'
    };

    if (options.tags) {
      params.tags = options.tags;
    }

    if (options.metadata) {
      params.metadata = options.metadata;
    }

    const result = this.executeMCP('app_store_publish_app', params);

    if (result.success) {
      console.log('\n✓ Application published successfully!');
      console.log(`App ID: ${result.app_id}`);
      console.log(`Status: ${result.status}`);

      if (result.approval_required) {
        console.log('Note: Application pending approval');
      }

      return result;
    } else {
      console.error('✗ Publishing failed:', result.error);
      process.exit(1);
    }
  }

  /**
   * Get application information
   */
  appInfo(appId) {
    console.log(`Fetching application info: ${appId}`);

    const result = this.executeMCP('app_get', { app_id: appId });

    if (result.app) {
      const app = result.app;
      console.log('\nApplication Details:');
      console.log(`  ID: ${app.id}`);
      console.log(`  Name: ${app.name}`);
      console.log(`  Category: ${app.category}`);
      console.log(`  Version: ${app.version}`);
      console.log(`  Description: ${app.description}`);
      console.log(`  Author: ${app.author || 'N/A'}`);
      console.log(`  Downloads: ${app.download_count || 0}`);
      console.log(`  Rating: ${app.rating || 'N/A'}`);

      if (app.tags) {
        console.log(`  Tags: ${app.tags.join(', ')}`);
      }

      return app;
    } else {
      console.error('✗ Application not found');
      process.exit(1);
    }
  }

  /**
   * Search applications
   */
  search(query, options = {}) {
    console.log(`Searching for: ${query}`);

    const params = {
      search: query,
      limit: options.limit || 20
    };

    if (options.category) params.category = options.category;
    if (options.featured) params.featured = true;

    const result = this.executeMCP('app_search', params);

    if (result.apps && result.apps.length > 0) {
      console.log(`\nFound ${result.apps.length} application(s):\n`);

      result.apps.forEach(app => {
        console.log(`  Name: ${app.name}`);
        console.log(`  ID: ${app.id}`);
        console.log(`  Category: ${app.category}`);
        console.log(`  Description: ${app.description.substring(0, 100)}...`);
        console.log(`  Rating: ${app.rating || 'N/A'} | Downloads: ${app.download_count || 0}`);
        console.log('  ---');
      });

      return result.apps;
    } else {
      console.log('No applications found');
      return [];
    }
  }

  /**
   * Get application analytics
   */
  analytics(appId, timeframe = '30d') {
    console.log(`Fetching analytics for: ${appId} (${timeframe})`);

    const result = this.executeMCP('app_analytics', {
      app_id: appId,
      timeframe
    });

    if (result.analytics) {
      const a = result.analytics;
      console.log('\nAnalytics:');
      console.log(`  Total Downloads: ${a.total_downloads || 0}`);
      console.log(`  Total Deploys: ${a.total_deploys || 0}`);
      console.log(`  Active Installations: ${a.active_installations || 0}`);
      console.log(`  Average Rating: ${a.average_rating || 'N/A'}`);
      console.log(`  Revenue (rUv): ${a.revenue || 0}`);

      if (a.downloads_by_day) {
        console.log('\n  Downloads Trend:');
        a.downloads_by_day.slice(0, 7).forEach(day => {
          console.log(`    ${day.date}: ${day.count}`);
        });
      }

      return a;
    } else {
      console.error('✗ Analytics not available');
      process.exit(1);
    }
  }
}

// CLI Interface
function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log(`
Flow Nexus Deployment Manager

Usage:
  node deployment-manager.js list-templates [--category <cat>] [--featured]
  node deployment-manager.js template-info <templateName>
  node deployment-manager.js deploy <templateName> <deploymentName> [--var KEY=VALUE]
  node deployment-manager.js publish <name> <description> <category> <sourceFile> [--tags tag1,tag2]
  node deployment-manager.js app-info <appId>
  node deployment-manager.js search <query> [--category <cat>]
  node deployment-manager.js analytics <appId> [--timeframe 24h|7d|30d|90d]

Categories: web-api, frontend, full-stack, cli-tools, data-processing, ml-models, blockchain, mobile, backend

Examples:
  node deployment-manager.js list-templates --category web-api --featured
  node deployment-manager.js deploy express-api-starter my-production-api --var database_url=postgres://...
  node deployment-manager.js publish "JWT Auth Service" "Production JWT auth" backend ./auth-service.js --tags auth,jwt,security
  node deployment-manager.js search "authentication" --category backend
    `);
    process.exit(0);
  }

  const manager = new DeploymentManager();
  const command = args[0];

  try {
    switch (command) {
      case 'list-templates': {
        const options = {};
        const catIdx = args.indexOf('--category');
        if (catIdx !== -1 && args[catIdx + 1]) {
          options.category = args[catIdx + 1];
        }
        if (args.includes('--featured')) {
          options.featured = true;
        }
        manager.listTemplates(options);
        break;
      }

      case 'template-info':
        if (args.length < 2) {
          console.error('Error: template-info requires <templateName>');
          process.exit(1);
        }
        manager.templateInfo(args[1]);
        break;

      case 'deploy': {
        if (args.length < 3) {
          console.error('Error: deploy requires <templateName> <deploymentName>');
          process.exit(1);
        }

        const options = { variables: {}, env_vars: {} };
        for (let i = 3; i < args.length; i++) {
          if (args[i] === '--var' && args[i + 1]) {
            const [key, value] = args[i + 1].split('=');
            options.variables[key] = value;
            i++;
          } else if (args[i] === '--env' && args[i + 1]) {
            const [key, value] = args[i + 1].split('=');
            options.env_vars[key] = value;
            i++;
          }
        }

        manager.deploy(args[1], args[2], options);
        break;
      }

      case 'publish': {
        if (args.length < 5) {
          console.error('Error: publish requires <name> <description> <category> <sourceFile>');
          process.exit(1);
        }

        const options = {};
        const tagsIdx = args.indexOf('--tags');
        if (tagsIdx !== -1 && args[tagsIdx + 1]) {
          options.tags = args[tagsIdx + 1].split(',');
        }

        manager.publish(args[1], args[2], args[3], args[4], options);
        break;
      }

      case 'app-info':
        if (args.length < 2) {
          console.error('Error: app-info requires <appId>');
          process.exit(1);
        }
        manager.appInfo(args[1]);
        break;

      case 'search': {
        if (args.length < 2) {
          console.error('Error: search requires <query>');
          process.exit(1);
        }

        const options = {};
        const catIdx = args.indexOf('--category');
        if (catIdx !== -1 && args[catIdx + 1]) {
          options.category = args[catIdx + 1];
        }

        manager.search(args[1], options);
        break;
      }

      case 'analytics': {
        if (args.length < 2) {
          console.error('Error: analytics requires <appId>');
          process.exit(1);
        }

        const tfIdx = args.indexOf('--timeframe');
        const timeframe = tfIdx !== -1 && args[tfIdx + 1] ? args[tfIdx + 1] : '30d';

        manager.analytics(args[1], timeframe);
        break;
      }

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

module.exports = DeploymentManager;
