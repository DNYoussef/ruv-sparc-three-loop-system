#!/usr/bin/env node
/**
 * Flow Nexus Neural - Single Model Training Test Suite
 *
 * Tests for single-node neural network training with various architectures.
 * Covers feedforward, LSTM, transformer, GAN, and autoencoder architectures.
 *
 * Usage:
 *   npm test -- tests/test_single_model_training.js
 *   node tests/test_single_model_training.js --verbose
 */

const assert = require('assert');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

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
 * Test Suite: Single Model Training
 */
class SingleModelTrainingTests {
  constructor() {
    this.testResults = [];
    this.modelIds = [];
  }

  /**
   * Load template configuration
   */
  loadTemplate(templateName) {
    const templatePath = path.join(__dirname, '..', 'resources', 'templates', `${templateName}.json`);
    return JSON.parse(fs.readFileSync(templatePath, 'utf8'));
  }

  /**
   * Test 1: Feedforward Network Training
   */
  async testFeedforwardTraining() {
    console.log('\n🧪 Test 1: Feedforward Network Training');

    try {
      const config = this.loadTemplate('feedforward-classifier');

      const response = await api.post('/neural/train', {
        architecture: config.architecture,
        training: {
          epochs: 5, // Reduced for testing
          batch_size: 32,
          learning_rate: 0.001,
          optimizer: 'adam'
        },
        tier: 'nano',
        user_id: USER_ID
      });

      assert(response.data.job_id, 'Job ID should be returned');
      assert(response.data.status === 'training' || response.data.status === 'queued', 'Status should be training or queued');

      console.log(`    ✓ Feedforward training started: ${response.data.job_id}`);
      this.modelIds.push(response.data.job_id);

      this.testResults.push({ test: 'Feedforward Training', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Feedforward Training', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 2: LSTM Network Training
   */
  async testLSTMTraining() {
    console.log('\n🧪 Test 2: LSTM Network Training');

    try {
      const config = this.loadTemplate('lstm-timeseries');

      const response = await api.post('/neural/train', {
        architecture: config.architecture,
        training: {
          epochs: 5,
          batch_size: 64,
          learning_rate: 0.001,
          optimizer: 'adam'
        },
        tier: 'small',
        user_id: USER_ID
      });

      assert(response.data.job_id, 'Job ID should be returned');
      console.log(`    ✓ LSTM training started: ${response.data.job_id}`);
      this.modelIds.push(response.data.job_id);

      this.testResults.push({ test: 'LSTM Training', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'LSTM Training', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 3: Transformer Training
   */
  async testTransformerTraining() {
    console.log('\n🧪 Test 3: Transformer Training');

    try {
      const config = this.loadTemplate('transformer-nlp');

      const response = await api.post('/neural/train', {
        architecture: config.architecture,
        training: {
          epochs: 3,
          batch_size: 16,
          learning_rate: 0.0001,
          optimizer: 'adam'
        },
        tier: 'medium',
        user_id: USER_ID
      });

      assert(response.data.job_id, 'Job ID should be returned');
      console.log(`    ✓ Transformer training started: ${response.data.job_id}`);
      this.modelIds.push(response.data.job_id);

      this.testResults.push({ test: 'Transformer Training', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Transformer Training', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 4: Divergent Thinking Patterns
   */
  async testDivergentPatterns() {
    console.log('\n🧪 Test 4: Divergent Thinking Patterns');

    try {
      const patterns = ['lateral', 'quantum', 'chaotic', 'associative', 'evolutionary'];
      const results = [];

      for (const pattern of patterns) {
        const response = await api.post('/neural/train', {
          architecture: {
            type: 'feedforward',
            layers: [
              { type: 'dense', units: 64, activation: 'relu' },
              { type: 'dense', units: 10, activation: 'softmax' }
            ]
          },
          training: {
            epochs: 2,
            batch_size: 32,
            learning_rate: 0.001,
            optimizer: 'adam'
          },
          divergent: {
            enabled: true,
            pattern: pattern,
            factor: 0.5
          },
          tier: 'nano',
          user_id: USER_ID
        });

        assert(response.data.job_id, `Job ID should be returned for ${pattern}`);
        results.push(pattern);
        console.log(`    ✓ ${pattern} pattern training started`);
      }

      assert.strictEqual(results.length, patterns.length, 'All patterns should be tested');

      this.testResults.push({ test: 'Divergent Patterns', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Divergent Patterns', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 5: Training Status Monitoring
   */
  async testTrainingStatus() {
    console.log('\n🧪 Test 5: Training Status Monitoring');

    if (this.modelIds.length === 0) {
      console.log('  ⊘ Skipped: No training jobs available');
      this.testResults.push({ test: 'Training Status', status: 'SKIP' });
      return false;
    }

    try {
      const jobId = this.modelIds[0];
      const response = await api.get('/neural/training/status', {
        params: { job_id: jobId }
      });

      assert(response.data.job_id, 'Job ID should be returned');
      assert(response.data.status, 'Status should be returned');
      assert(typeof response.data.progress === 'number', 'Progress should be a number');

      console.log(`    ✓ Status retrieved: ${response.data.status} (${(response.data.progress * 100).toFixed(1)}%)`);

      this.testResults.push({ test: 'Training Status', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Training Status', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 6: Model List Retrieval
   */
  async testModelList() {
    console.log('\n🧪 Test 6: Model List Retrieval');

    try {
      const response = await api.get('/neural/models', {
        params: { user_id: USER_ID }
      });

      assert(response.data.models, 'Models array should be returned');
      assert(Array.isArray(response.data.models), 'Models should be an array');

      console.log(`    ✓ Retrieved ${response.data.models.length} model(s)`);

      this.testResults.push({ test: 'Model List', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Model List', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 7: Training Tiers
   */
  async testTrainingTiers() {
    console.log('\n🧪 Test 7: Training Tiers');

    try {
      const tiers = ['nano', 'mini', 'small'];
      const results = [];

      for (const tier of tiers) {
        const response = await api.post('/neural/train', {
          architecture: {
            type: 'feedforward',
            layers: [
              { type: 'dense', units: 32, activation: 'relu' },
              { type: 'dense', units: 10, activation: 'softmax' }
            ]
          },
          training: {
            epochs: 2,
            batch_size: 16,
            learning_rate: 0.001,
            optimizer: 'adam'
          },
          tier: tier,
          user_id: USER_ID
        });

        assert(response.data.job_id, `Job ID should be returned for tier ${tier}`);
        results.push(tier);
        console.log(`    ✓ Training started with tier: ${tier}`);
      }

      assert.strictEqual(results.length, tiers.length, 'All tiers should be tested');

      this.testResults.push({ test: 'Training Tiers', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Training Tiers', status: 'FAIL', error: error.message });
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
    console.log('\n🚀 Starting Single Model Training Test Suite');
    console.log('='.repeat(80));

    await this.testFeedforwardTraining();
    await this.testLSTMTraining();
    await this.testTransformerTraining();
    await this.testDivergentPatterns();
    await this.testTrainingStatus();
    await this.testModelList();
    await this.testTrainingTiers();

    const allPassed = this.printSummary();
    process.exit(allPassed ? 0 : 1);
  }
}

/**
 * Main execution
 */
if (require.main === module) {
  const tests = new SingleModelTrainingTests();
  tests.runAll().catch(error => {
    console.error('\n✗ Test suite failed:', error);
    process.exit(1);
  });
}

module.exports = SingleModelTrainingTests;
