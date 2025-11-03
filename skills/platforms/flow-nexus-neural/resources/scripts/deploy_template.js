#!/usr/bin/env node
/**
 * Flow Nexus Neural - Template Deployment Script
 *
 * Deploy pre-trained models from the Flow Nexus marketplace.
 * Browse, search, and deploy templates with custom configurations.
 *
 * Usage:
 *   node deploy_template.js --search "sentiment analysis"
 *   node deploy_template.js --template sentiment-analysis-v2 --config custom.json
 *
 * Features:
 *   - Search marketplace by category, tags, or keywords
 *   - Filter by tier (free/paid) and accuracy metrics
 *   - Deploy with custom training configurations
 *   - Rate and review deployed templates
 */

const { Command } = require('commander');
const axios = require('axios');
const Table = require('cli-table3');

const program = new Command();

program
  .name('deploy-template')
  .description('Deploy pre-trained neural network templates from Flow Nexus marketplace')
  .option('-s, --search <query>', 'Search templates by keyword')
  .option('-t, --template <id>', 'Template ID to deploy')
  .option('-c, --config <path>', 'Custom configuration JSON (optional)')
  .option('--category <type>', 'Filter by category (classification|regression|nlp|vision|timeseries|anomaly|generative)')
  .option('--tier <type>', 'Filter by tier (free|paid)', 'free')
  .option('--limit <count>', 'Maximum number of search results', '20')
  .option('--list-categories', 'List all available categories', false)
  .option('--verbose', 'Enable verbose logging', false)
  .parse(process.argv);

const opts = program.opts();

// Flow Nexus API configuration
const FLOW_NEXUS_API = process.env.FLOW_NEXUS_API || 'https://api.flow-nexus.ruv.io';
const API_KEY = process.env.FLOW_NEXUS_API_KEY;
const USER_ID = process.env.FLOW_NEXUS_USER_ID;

if (!API_KEY || !USER_ID) {
  console.error('Error: FLOW_NEXUS_API_KEY and FLOW_NEXUS_USER_ID environment variables required');
  process.exit(1);
}

const api = axios.create({
  baseURL: FLOW_NEXUS_API,
  headers: {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
  }
});

const CATEGORIES = [
  'classification',
  'regression',
  'nlp',
  'vision',
  'timeseries',
  'anomaly',
  'generative',
  'reinforcement',
  'custom'
];

/**
 * List available categories
 */
function listCategories() {
  console.log('\n📂 Available Template Categories:\n');
  CATEGORIES.forEach(cat => {
    console.log(`  • ${cat}`);
  });
  console.log('');
}

/**
 * Search marketplace templates
 */
async function searchTemplates() {
  console.log('\n🔍 Searching marketplace...\n');

  try {
    const params = {
      limit: parseInt(opts.limit)
    };

    if (opts.search) {
      params.search = opts.search;
    }
    if (opts.category) {
      params.category = opts.category;
    }
    if (opts.tier) {
      params.tier = opts.tier;
    }

    const response = await api.get('/neural/templates', { params });
    const templates = response.data.templates;

    if (templates.length === 0) {
      console.log('No templates found matching your criteria.');
      return [];
    }

    // Display results in table
    const table = new Table({
      head: ['ID', 'Name', 'Category', 'Accuracy', 'Downloads', 'Tier'],
      colWidths: [30, 40, 15, 12, 12, 8]
    });

    templates.forEach(t => {
      table.push([
        t.id,
        t.name,
        t.category,
        t.accuracy ? (t.accuracy * 100).toFixed(1) + '%' : 'N/A',
        t.downloads.toLocaleString(),
        t.tier
      ]);
    });

    console.log(table.toString());
    console.log(`\nFound ${templates.length} template(s)`);
    console.log(`\nTo deploy: node deploy_template.js --template <ID>\n`);

    return templates;
  } catch (error) {
    console.error('✗ Search failed:', error.response?.data || error.message);
    throw error;
  }
}

/**
 * Get template details
 */
async function getTemplateDetails(templateId) {
  console.log(`\n📋 Fetching template details: ${templateId}...\n`);

  try {
    const response = await api.get(`/neural/templates/${templateId}`);
    const template = response.data;

    console.log(`Name: ${template.name}`);
    console.log(`Category: ${template.category}`);
    console.log(`Description: ${template.description}`);
    console.log(`Accuracy: ${template.accuracy ? (template.accuracy * 100).toFixed(1) + '%' : 'N/A'}`);
    console.log(`Downloads: ${template.downloads.toLocaleString()}`);
    console.log(`Tier: ${template.tier}`);
    console.log(`Author: ${template.author || 'Flow Nexus'}`);
    console.log(`Version: ${template.version || '1.0.0'}`);
    console.log(`Created: ${new Date(template.created_at).toLocaleDateString()}`);

    if (template.tags && template.tags.length > 0) {
      console.log(`Tags: ${template.tags.join(', ')}`);
    }

    if (template.requirements) {
      console.log(`\nRequirements:`);
      Object.entries(template.requirements).forEach(([key, value]) => {
        console.log(`  • ${key}: ${value}`);
      });
    }

    return template;
  } catch (error) {
    console.error('✗ Failed to fetch template:', error.response?.data || error.message);
    throw error;
  }
}

/**
 * Deploy template
 */
async function deployTemplate(templateId, customConfig) {
  console.log(`\n🚀 Deploying template: ${templateId}...\n`);

  try {
    const payload = {
      template_id: templateId,
      user_id: USER_ID
    };

    // Add custom config if provided
    if (customConfig) {
      payload.custom_config = customConfig;
      console.log('Using custom configuration:');
      console.log(JSON.stringify(customConfig, null, 2));
      console.log('');
    }

    const response = await api.post('/neural/templates/deploy', payload);
    const deployment = response.data;

    console.log('✓ Template deployed successfully!');
    console.log(`\nDeployment Details:`);
    console.log(`  Model ID: ${deployment.model_id}`);
    console.log(`  Job ID: ${deployment.job_id}`);
    console.log(`  Status: ${deployment.status}`);
    console.log(`  Created: ${new Date(deployment.created_at).toLocaleString()}`);

    if (deployment.estimated_completion) {
      console.log(`  ETA: ${new Date(deployment.estimated_completion).toLocaleString()}`);
    }

    return deployment;
  } catch (error) {
    console.error('✗ Deployment failed:', error.response?.data || error.message);
    throw error;
  }
}

/**
 * Monitor deployment progress
 */
async function monitorDeployment(jobId) {
  console.log('\n📊 Monitoring deployment...\n');

  const startTime = Date.now();

  const interval = setInterval(async () => {
    try {
      const response = await api.get('/neural/training/status', {
        params: { job_id: jobId }
      });

      const status = response.data;
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);

      if (status.status === 'trained' || status.status === 'ready') {
        clearInterval(interval);
        console.log(`✓ Deployment completed in ${elapsed}s`);
        console.log(`  Model ID: ${status.model_id}`);
        console.log(`\nNext steps:`);
        console.log(`  1. Run inference: node predict_model.js --model ${status.model_id}`);
        console.log(`  2. Benchmark: node benchmark_model.js --model ${status.model_id}`);
        console.log(`  3. Rate template: node rate_template.js --template ${opts.template} --rating 5`);
      } else if (status.status === 'failed') {
        clearInterval(interval);
        console.log(`✗ Deployment failed after ${elapsed}s`);
      } else {
        const progress = status.progress ? (status.progress * 100).toFixed(1) : '0.0';
        console.log(`Progress: ${progress}% | Status: ${status.status} | Elapsed: ${elapsed}s`);
      }
    } catch (error) {
      if (opts.verbose) {
        console.error('Status check error:', error.message);
      }
    }
  }, 3000); // Check every 3 seconds
}

/**
 * Main execution
 */
async function main() {
  try {
    // List categories if requested
    if (opts.listCategories) {
      listCategories();
      return;
    }

    // Search mode
    if (opts.search || opts.category) {
      await searchTemplates();
      return;
    }

    // Deploy mode
    if (!opts.template) {
      console.error('Error: Either --search, --category, or --template required');
      program.help();
      return;
    }

    // Get template details
    const template = await getTemplateDetails(opts.template);

    // Load custom config if provided
    let customConfig = null;
    if (opts.config) {
      const fs = require('fs');
      customConfig = JSON.parse(fs.readFileSync(opts.config, 'utf8'));
    }

    // Deploy
    const deployment = await deployTemplate(opts.template, customConfig);

    // Monitor
    if (deployment.job_id) {
      await monitorDeployment(deployment.job_id);
    }

    console.log('\n✓ Template deployment completed!');

  } catch (error) {
    console.error('\n✗ Operation failed:', error.message);
    if (opts.verbose) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

main();
