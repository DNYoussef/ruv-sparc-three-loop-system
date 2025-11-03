#!/usr/bin/env node
/**
 * Flow Nexus Neural - Distributed Training Test Suite
 *
 * Comprehensive tests for distributed neural network training across E2B clusters.
 * Tests cluster initialization, node deployment, training coordination, and fault tolerance.
 *
 * Usage:
 *   npm test -- tests/test_distributed_training.js
 *   node tests/test_distributed_training.js --verbose
 */

const assert = require('assert');
const axios = require('axios');

const FLOW_NEXUS_API = process.env.FLOW_NEXUS_API || 'https://api.flow-nexus.ruv.io';
const API_KEY = process.env.FLOW_NEXUS_API_KEY;
const TEST_TIMEOUT = 300000; // 5 minutes for distributed tests

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
 * Test Suite: Distributed Training
 */
class DistributedTrainingTests {
  constructor() {
    this.testResults = [];
    this.clusterId = null;
  }

  /**
   * Test 1: Cluster Initialization
   */
  async testClusterInitialization() {
    console.log('\n🧪 Test 1: Cluster Initialization');

    try {
      const topologies = ['mesh', 'ring', 'star', 'hierarchical'];

      for (const topology of topologies) {
        console.log(`  Testing ${topology} topology...`);

        const response = await api.post('/neural/cluster/init', {
          name: `test-cluster-${topology}-${Date.now()}`,
          architecture: 'transformer',
          topology: topology,
          consensus: 'proof-of-learning',
          daaEnabled: true,
          wasmOptimization: true
        });

        assert(response.data.cluster_id, 'Cluster ID should be returned');
        assert.strictEqual(response.data.topology, topology, `Topology should be ${topology}`);
        assert.strictEqual(response.data.status, 'initializing', 'Status should be initializing');

        // Save one cluster for subsequent tests
        if (topology === 'mesh' && !this.clusterId) {
          this.clusterId = response.data.cluster_id;
        }

        console.log(`    ✓ ${topology} topology initialized successfully`);
      }

      this.testResults.push({ test: 'Cluster Initialization', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Cluster Initialization', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 2: Node Deployment
   */
  async testNodeDeployment() {
    console.log('\n🧪 Test 2: Node Deployment');

    if (!this.clusterId) {
      console.log('  ⊘ Skipped: No cluster available');
      this.testResults.push({ test: 'Node Deployment', status: 'SKIP' });
      return false;
    }

    try {
      // Deploy parameter server
      console.log('  Deploying parameter server...');
      const paramServer = await api.post('/neural/node/deploy', {
        cluster_id: this.clusterId,
        node_type: 'parameter_server',
        model: 'large',
        capabilities: ['parameter_management', 'gradient_aggregation']
      });

      assert(paramServer.data.node_id, 'Parameter server node ID should be returned');
      assert.strictEqual(paramServer.data.type, 'parameter_server', 'Node type should match');
      console.log('    ✓ Parameter server deployed');

      // Deploy workers
      console.log('  Deploying worker nodes...');
      const workerPromises = [];
      for (let i = 0; i < 3; i++) {
        workerPromises.push(
          api.post('/neural/node/deploy', {
            cluster_id: this.clusterId,
            node_type: 'worker',
            model: 'xl',
            capabilities: ['training', 'inference']
          })
        );
      }

      const workers = await Promise.all(workerPromises);
      assert.strictEqual(workers.length, 3, 'Should deploy 3 workers');
      workers.forEach((w, idx) => {
        assert(w.data.node_id, `Worker ${idx + 1} should have node ID`);
      });
      console.log('    ✓ Worker nodes deployed');

      // Deploy aggregator
      console.log('  Deploying aggregator...');
      const aggregator = await api.post('/neural/node/deploy', {
        cluster_id: this.clusterId,
        node_type: 'aggregator',
        model: 'large',
        capabilities: ['gradient_aggregation', 'model_synchronization']
      });

      assert(aggregator.data.node_id, 'Aggregator node ID should be returned');
      console.log('    ✓ Aggregator deployed');

      this.testResults.push({ test: 'Node Deployment', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Node Deployment', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 3: Cluster Connection
   */
  async testClusterConnection() {
    console.log('\n🧪 Test 3: Cluster Connection');

    if (!this.clusterId) {
      console.log('  ⊘ Skipped: No cluster available');
      this.testResults.push({ test: 'Cluster Connection', status: 'SKIP' });
      return false;
    }

    try {
      const response = await api.post('/neural/cluster/connect', {
        cluster_id: this.clusterId,
        topology: 'mesh'
      });

      assert(response.data.connected, 'Cluster should be connected');
      assert(response.data.connections > 0, 'Should have active connections');

      console.log(`    ✓ Cluster connected with ${response.data.connections} connections`);

      this.testResults.push({ test: 'Cluster Connection', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Cluster Connection', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 4: Distributed Training Start
   */
  async testDistributedTraining() {
    console.log('\n🧪 Test 4: Distributed Training Start');

    if (!this.clusterId) {
      console.log('  ⊘ Skipped: No cluster available');
      this.testResults.push({ test: 'Distributed Training', status: 'SKIP' });
      return false;
    }

    try {
      const response = await api.post('/neural/train/distributed', {
        cluster_id: this.clusterId,
        dataset: 'test-dataset',
        epochs: 5,
        batch_size: 32,
        learning_rate: 0.001,
        optimizer: 'adam',
        federated: false
      });

      assert(response.data.job_id, 'Training job ID should be returned');
      assert.strictEqual(response.data.status, 'training', 'Status should be training');

      console.log(`    ✓ Distributed training started: ${response.data.job_id}`);

      this.testResults.push({ test: 'Distributed Training', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Distributed Training', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 5: Cluster Status Monitoring
   */
  async testClusterStatus() {
    console.log('\n🧪 Test 5: Cluster Status Monitoring');

    if (!this.clusterId) {
      console.log('  ⊘ Skipped: No cluster available');
      this.testResults.push({ test: 'Cluster Status', status: 'SKIP' });
      return false;
    }

    try {
      const response = await api.get('/neural/cluster/status', {
        params: { cluster_id: this.clusterId }
      });

      assert(response.data.cluster_id, 'Cluster ID should be returned');
      assert(response.data.nodes, 'Nodes array should exist');
      assert(Array.isArray(response.data.nodes), 'Nodes should be an array');
      assert(response.data.nodes.length > 0, 'Should have at least one node');

      console.log(`    ✓ Cluster status retrieved: ${response.data.nodes.length} nodes`);

      // Verify node types
      const nodeTypes = response.data.nodes.map(n => n.type);
      assert(nodeTypes.includes('parameter_server'), 'Should have parameter server');
      assert(nodeTypes.includes('worker'), 'Should have worker nodes');
      console.log(`    ✓ Node types validated: ${[...new Set(nodeTypes)].join(', ')}`);

      this.testResults.push({ test: 'Cluster Status', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Cluster Status', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 6: Federated Learning Mode
   */
  async testFederatedLearning() {
    console.log('\n🧪 Test 6: Federated Learning Mode');

    try {
      // Create new cluster for federated learning
      const cluster = await api.post('/neural/cluster/init', {
        name: `test-federated-${Date.now()}`,
        architecture: 'cnn',
        topology: 'star',
        consensus: 'byzantine',
        daaEnabled: true
      });

      const federatedClusterId = cluster.data.cluster_id;

      // Start federated training
      const response = await api.post('/neural/train/distributed', {
        cluster_id: federatedClusterId,
        dataset: 'federated-test-dataset',
        epochs: 10,
        batch_size: 64,
        learning_rate: 0.0001,
        optimizer: 'adam',
        federated: true,
        aggregation_rounds: 5
      });

      assert(response.data.job_id, 'Federated job ID should be returned');
      assert(response.data.federated === true, 'Federated mode should be enabled');

      console.log(`    ✓ Federated learning started: ${response.data.job_id}`);

      this.testResults.push({ test: 'Federated Learning', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Federated Learning', status: 'FAIL', error: error.message });
      return false;
    }
  }

  /**
   * Test 7: Cluster Termination
   */
  async testClusterTermination() {
    console.log('\n🧪 Test 7: Cluster Termination');

    if (!this.clusterId) {
      console.log('  ⊘ Skipped: No cluster available');
      this.testResults.push({ test: 'Cluster Termination', status: 'SKIP' });
      return false;
    }

    try {
      const response = await api.post('/neural/cluster/terminate', {
        cluster_id: this.clusterId
      });

      assert(response.data.terminated, 'Cluster should be terminated');

      console.log(`    ✓ Cluster terminated successfully`);

      this.testResults.push({ test: 'Cluster Termination', status: 'PASS' });
      return true;
    } catch (error) {
      console.error(`    ✗ Test failed: ${error.message}`);
      this.testResults.push({ test: 'Cluster Termination', status: 'FAIL', error: error.message });
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
    console.log('\n🚀 Starting Distributed Training Test Suite');
    console.log('='.repeat(80));

    await this.testClusterInitialization();
    await this.testNodeDeployment();
    await this.testClusterConnection();
    await this.testDistributedTraining();
    await this.testClusterStatus();
    await this.testFederatedLearning();
    await this.testClusterTermination();

    const allPassed = this.printSummary();
    process.exit(allPassed ? 0 : 1);
  }
}

/**
 * Main execution
 */
if (require.main === module) {
  const tests = new DistributedTrainingTests();
  tests.runAll().catch(error => {
    console.error('\n✗ Test suite failed:', error);
    process.exit(1);
  });
}

module.exports = DistributedTrainingTests;
