#!/usr/bin/env node
/**
 * Flow Nexus Neural - Template Deployment Test Suite
 *
 * Tests for marketplace template browsing, deployment, and rating.
 * Validates template search, filtering, deployment with custom configs, and user feedback.
 *
 * Usage:
 *   npm test -- tests/test_template_deployment.js
 *   node tests/test_template_deployment.js --verbose
 */

const assert = require('assert');
const axios = require('axios');

const FLOW_NEXUS_API = process.env.FLOW_NEXUS_API || 'https://api.flow-nexus.ruv.io';
const API_KEY = process.env.FLOW_NEXUS_API_KEY;
const USER_ID = process.env.FLOW_NEXUS_USER_ID || 'test-user';

if (!API_KEY) {
  console.error('Error: FLOW_NEXUS_API_KEY environment variable required for tests');
  process.exit(1);
}

const api = axios.create({
  baseURL: FLOW_NEXUS_API,
  headers: {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
  }
});

/**
 * Test Suite: Template Deployment
 */
class TemplateDeploymentTests {
  constructor() {
    this.testResults = [];
    this.templateIds = [];
    this.deployedModels = [];
  }

  /**
   * Test 1: List All Templates
   */
  async testListTemplates() {
    console.log('\n🧪 Test 1: List All Templates');

    try {
      const response = await api.get('/neural/templates', {
        params: { limit: 20 }
      });

      assert(response.data.templates, 'Templates array should be returned');
      assert(Array.isArray(response.data.templates), 'Templates should be an array');

      console.log(`    ✓ Retrieved ${response.data.templates.length} template(s)`);

      // Save template IDs for later tests
      if (response.data.templates.length > 0) {
        this.templateIds = response.data.templates.slice(0, 3).map(t => t.id);
      }

      this.testResults.push({ test: 'List Templates', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'List Templates', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 2: Search Templates by Category
   */
  async testSearchByCategory() {
    console.log('\n🧪 Test 2: Search Templates by Category');

    try {
      const categories = ['classification', 'nlp', 'timeseries'];
      const results = [];

      for (const category of categories) {
        const response = await api.get('/neural/templates', {
          params: { category: category, limit: 10 }
        });

        assert(response.data.templates, `Templates should be returned for ${category}`);
        results.push({
          category: category,
          count: response.data.templates.length
        });

        console.log(`    ✓ ${category}: ${response.data.templates.length} template(s)`);
      }

      assert.strictEqual(results.length, categories.length, 'All categories should be searched');

      this.testResults.push({ test: 'Search by Category', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Search by Category', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 3: Search Templates by Keyword
   */
  async testSearchByKeyword() {
    console.log('\n🧪 Test 3: Search Templates by Keyword');

    try {
      const keywords = ['sentiment', 'image', 'forecasting'];
      const results = [];

      for (const keyword of keywords) {
        const response = await api.get('/neural/templates', {
          params: { search: keyword, limit: 10 }
        });

        assert(response.data.templates, `Templates should be returned for keyword: ${keyword}`);
        results.push(keyword);

        console.log(`    ✓ "${keyword}": ${response.data.templates.length} match(es)`);
      }

      assert.strictEqual(results.length, keywords.length, 'All keywords should be searched');

      this.testResults.push({ test: 'Search by Keyword', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Search by Keyword', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 4: Filter by Tier
   */
  async testFilterByTier() {
    console.log('\n🧪 Test 4: Filter by Tier');

    try {
      const tiers = ['free', 'paid'];
      const results = [];

      for (const tier of tiers) {
        const response = await api.get('/neural/templates', {
          params: { tier: tier, limit: 20 }
        });

        assert(response.data.templates, `Templates should be returned for tier: ${tier}`);

        // Verify all returned templates match the tier
        response.data.templates.forEach(template => {
          assert.strictEqual(template.tier, tier, `Template tier should be ${tier}`);
        });

        results.push({
          tier: tier,
          count: response.data.templates.length
        });

        console.log(`    ✓ ${tier}: ${response.data.templates.length} template(s)`);
      }

      assert.strictEqual(results.length, tiers.length, 'All tiers should be tested');

      this.testResults.push({ test: 'Filter by Tier', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Filter by Tier', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 5: Get Template Details
   */
  async testGetTemplateDetails() {
    console.log('\n🧪 Test 5: Get Template Details');

    if (this.templateIds.length === 0) {
      console.log('  ⊘ Skipped: No templates available');
      this.testResults.push({ test: 'Template Details', status: 'SKIP' });
      return false;
    }

    try {
      const templateId = this.templateIds[0];
      const response = await api.get(`/neural/templates/${templateId}`);

      assert(response.data.id, 'Template ID should be returned');
      assert(response.data.name, 'Template name should be returned');
      assert(response.data.category, 'Template category should be returned');
      assert(response.data.description, 'Template description should be returned');

      console.log(`    ✓ Template details retrieved: ${response.data.name}`);

      this.testResults.push({ test: 'Template Details', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Template Details', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 6: Deploy Template (Default Config)
   */
  async testDeployTemplateDefault() {
    console.log('\n🧪 Test 6: Deploy Template (Default Config)');

    if (this.templateIds.length === 0) {
      console.log('  ⊘ Skipped: No templates available');
      this.testResults.push({ test: 'Deploy Default', status: 'SKIP' });
      return false;
    }

    try {
      const templateId = this.templateIds[0];
      const response = await api.post('/neural/templates/deploy', {
        template_id: templateId,
        user_id: USER_ID
      });

      assert(response.data.model_id || response.data.job_id, 'Model/Job ID should be returned');
      assert(response.data.status, 'Status should be returned');

      console.log(`    ✓ Template deployed: ${response.data.model_id || response.data.job_id}`);
      this.deployedModels.push(response.data.model_id || response.data.job_id);

      this.testResults.push({ test: 'Deploy Default', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Deploy Default', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 7: Deploy Template (Custom Config)
   */
  async testDeployTemplateCustom() {
    console.log('\n🧪 Test 7: Deploy Template (Custom Config)');

    if (this.templateIds.length === 0) {
      console.log('  ⊘ Skipped: No templates available');
      this.testResults.push({ test: 'Deploy Custom', status: 'SKIP' });
      return false;
    }

    try {
      const templateId = this.templateIds[0];
      const customConfig = {
        training: {
          epochs: 20,
          batch_size: 64,
          learning_rate: 0.0005
        }
      };

      const response = await api.post('/neural/templates/deploy', {
        template_id: templateId,
        custom_config: customConfig,
        user_id: USER_ID
      });

      assert(response.data.model_id || response.data.job_id, 'Model/Job ID should be returned');
      console.log(`    ✓ Template deployed with custom config: ${response.data.model_id || response.data.job_id}`);

      this.testResults.push({ test: 'Deploy Custom', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Deploy Custom', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 8: Rate Template
   */
  async testRateTemplate() {
    console.log('\n🧪 Test 8: Rate Template');

    if (this.templateIds.length === 0) {
      console.log('  ⊘ Skipped: No templates available');
      this.testResults.push({ test: 'Rate Template', status: 'SKIP' });
      return false;
    }

    try {
      const templateId = this.templateIds[0];
      const response = await api.post('/neural/templates/rate', {
        template_id: templateId,
        rating: 5,
        review: 'Excellent template! Works great for my use case.',
        user_id: USER_ID
      });

      assert(response.data.rated || response.data.success, 'Rating should be acknowledged');
      console.log(`    ✓ Template rated successfully`);

      this.testResults.push({ test: 'Rate Template', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Rate Template', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 9: Publish Model as Template
   */
  async testPublishTemplate() {
    console.log('\n🧪 Test 9: Publish Model as Template');

    if (this.deployedModels.length === 0) {
      console.log('  ⊘ Skipped: No deployed models available');
      this.testResults.push({ test: 'Publish Template', status: 'SKIP' });
      return false;
    }

    try {
      const modelId = this.deployedModels[0];
      const response = await api.post('/neural/templates/publish', {
        model_id: modelId,
        name: 'Test Custom Template',
        description: 'Test template created during automated testing',
        category: 'custom',
        price: 0,
        user_id: USER_ID
      });

      assert(response.data.template_id || response.data.success, 'Template publication should be acknowledged');
      console.log(`    ✓ Model published as template`);

      this.testResults.push({ test: 'Publish Template', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Publish Template', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 10: Performance Benchmark
   */
  async testTemplateBenchmark() {
    console.log('\n🧪 Test 10: Template Performance Benchmark');

    if (this.deployedModels.length === 0) {
      console.log('  ⊘ Skipped: No deployed models available');
      this.testResults.push({ test: 'Benchmark Template', status: 'SKIP' });
      return false;
    }

    try {
      const modelId = this.deployedModels[0];
      const response = await api.post('/neural/benchmark', {
        model_id: modelId,
        benchmark_type: 'comprehensive'
      });

      assert(response.data.benchmarks, 'Benchmark results should be returned');
      console.log(`    ✓ Benchmark completed`);

      this.testResults.push({ test: 'Benchmark Template', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Benchmark Template', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Print test summary
   */
  printSummary() {
    console.log('\n' + '='.repeat(80));
    console.log('TEST SUMMARY');
    console.log('='.repeat(80));

    const passed = this.testResults.filter(r => r.status === 'PASS').length;
    const failed = this.testResults.filter(r => r.status === 'FAIL').length;
    const skipped = this.testResults.filter(r => r.status === 'SKIP').length;

    this.testResults.forEach(result => {
      const icon = result.status === 'PASS' ? '✓' : result.status === 'FAIL' ? '✗' : '⊘';
      console.log(`${icon} ${result.test}: ${result.status}`);
      if (result.error) {
        console.log(`  Error: ${result.error}`);
      }
    });

    console.log('\n' + '-'.repeat(80));
    console.log(`Total: ${this.testResults.length} | Passed: ${passed} | Failed: ${failed} | Skipped: ${skipped}`);
    console.log('='.repeat(80) + '\n');

    return failed === 0;
  }

  /**
   * Run all tests
   */
  async runAll() {
    console.log('\n🚀 Starting Template Deployment Test Suite');
    console.log('='.repeat(80));

    await this.testListTemplates();
    await this.testSearchByCategory();
    await this.testSearchByKeyword();
    await this.testFilterByTier();
    await this.testGetTemplateDetails();
    await this.testDeployTemplateDefault();
    await this.testDeployTemplateCustom();
    await this.testRateTemplate();
    await this.testPublishTemplate();
    await this.testTemplateBenchmark();

    const allPassed = this.printSummary();
    process.exit(allPassed ? 0 : 1);
  }
}

/**
 * Main execution
 */
if (require.main === module) {
  const tests = new TemplateDeploymentTests();
  tests.runAll().catch(error => {
    console.error('\n✗ Test suite failed:', error);
    process.exit(1);
  });
}

module.exports = TemplateDeploymentTests;
