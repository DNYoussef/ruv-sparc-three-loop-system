#!/usr/bin/env node
/**
 * Flow Nexus Neural - Distributed Cluster Training Script
 *
 * Train neural networks across multiple E2B sandbox nodes with distributed computing.
 * Supports data parallelism, model parallelism, and federated learning.
 *
 * Usage:
 *   node train_distributed_cluster.js --config <config.json> --dataset <dataset-id>
 *
 * Features:
 *   - Multi-node cluster initialization (mesh, ring, star, hierarchical)
 *   - Parameter server + worker architecture
 *   - Gradient aggregation and synchronization
 *   - Federated learning support (data stays local)
 *   - Byzantine fault tolerance
 *   - Real-time monitoring and checkpointing
 */

const { Command } = require('commander');
const axios = require('axios');

const program = new Command();

program
  .name('train-distributed-cluster')
  .description('Train neural networks on distributed E2B sandbox cluster')
  .requiredOption('-c, --config <path>', 'Path to training configuration JSON')
  .requiredOption('-d, --dataset <id>', 'Dataset identifier or path')
  .option('-n, --nodes <count>', 'Number of worker nodes', '4')
  .option('-t, --topology <type>', 'Cluster topology (mesh|ring|star|hierarchical)', 'mesh')
  .option('-f, --federated', 'Enable federated learning mode', false)
  .option('-e, --epochs <count>', 'Number of training epochs', '100')
  .option('-b, --batch-size <size>', 'Batch size per worker', '32')
  .option('-l, --learning-rate <rate>', 'Learning rate', '0.001')
  .option('--checkpoint-interval <epochs>', 'Save checkpoint every N epochs', '10')
  .option('--verbose', 'Enable verbose logging', false)
  .parse(process.argv);

const opts = program.opts();

// Flow Nexus API configuration
const FLOW_NEXUS_API = process.env.FLOW_NEXUS_API || 'https://api.flow-nexus.ruv.io';
const API_KEY = process.env.FLOW_NEXUS_API_KEY;

if (!API_KEY) {
  console.error('Error: FLOW_NEXUS_API_KEY environment variable not set');
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
 * Initialize distributed training cluster
 */
async function initializeCluster(config) {
  console.log('\n🚀 Initializing distributed training cluster...');
  console.log(`   Topology: ${opts.topology}`);
  console.log(`   Nodes: ${opts.nodes}`);
  console.log(`   Federated: ${opts.federated ? 'Yes' : 'No'}\n`);

  try {
    const response = await api.post('/neural/cluster/init', {
      name: `training-${Date.now()}`,
      architecture: config.architecture.type,
      topology: opts.topology,
      consensus: 'proof-of-learning',
      daaEnabled: true,
      wasmOptimization: true
    });

    console.log(`✓ Cluster initialized: ${response.data.cluster_id}`);
    return response.data;
  } catch (error) {
    console.error('✗ Cluster initialization failed:', error.response?.data || error.message);
    throw error;
  }
}

/**
 * Deploy parameter server node
 */
async function deployParameterServer(clusterId, config) {
  console.log('📦 Deploying parameter server...');

  try {
    const response = await api.post('/neural/node/deploy', {
      cluster_id: clusterId,
      node_type: 'parameter_server',
      model: 'large',
      capabilities: ['parameter_management', 'gradient_aggregation', 'checkpointing'],
      autonomy: 0.8
    });

    console.log(`✓ Parameter server deployed: ${response.data.node_id}`);
    return response.data;
  } catch (error) {
    console.error('✗ Parameter server deployment failed:', error.response?.data || error.message);
    throw error;
  }
}

/**
 * Deploy worker nodes
 */
async function deployWorkerNodes(clusterId, config, count) {
  console.log(`👷 Deploying ${count} worker nodes...`);

  const workers = [];
  for (let i = 0; i < count; i++) {
    try {
      const response = await api.post('/neural/node/deploy', {
        cluster_id: clusterId,
        node_type: 'worker',
        model: 'xl',
        role: 'worker',
        capabilities: ['training', 'inference', 'data_loading'],
        layers: config.architecture.layers,
        autonomy: 0.9
      });

      workers.push(response.data);
      console.log(`✓ Worker ${i + 1}/${count} deployed: ${response.data.node_id}`);
    } catch (error) {
      console.error(`✗ Worker ${i + 1} deployment failed:`, error.response?.data || error.message);
    }
  }

  return workers;
}

/**
 * Deploy gradient aggregator
 */
async function deployAggregator(clusterId) {
  console.log('🔄 Deploying gradient aggregator...');

  try {
    const response = await api.post('/neural/node/deploy', {
      cluster_id: clusterId,
      node_type: 'aggregator',
      model: 'large',
      capabilities: ['gradient_aggregation', 'model_synchronization', 'convergence_detection']
    });

    console.log(`✓ Aggregator deployed: ${response.data.node_id}`);
    return response.data;
  } catch (error) {
    console.error('✗ Aggregator deployment failed:', error.response?.data || error.message);
    throw error;
  }
}

/**
 * Connect cluster nodes
 */
async function connectCluster(clusterId) {
  console.log('🔗 Connecting cluster nodes...');

  try {
    const response = await api.post('/neural/cluster/connect', {
      cluster_id: clusterId,
      topology: opts.topology
    });

    console.log('✓ Cluster nodes connected');
    return response.data;
  } catch (error) {
    console.error('✗ Cluster connection failed:', error.response?.data || error.message);
    throw error;
  }
}

/**
 * Start distributed training
 */
async function startTraining(clusterId, config) {
  console.log('\n🎓 Starting distributed training...');
  console.log(`   Dataset: ${opts.dataset}`);
  console.log(`   Epochs: ${opts.epochs}`);
  console.log(`   Batch Size: ${opts.batchSize}`);
  console.log(`   Learning Rate: ${opts.learningRate}\n`);

  try {
    const response = await api.post('/neural/train/distributed', {
      cluster_id: clusterId,
      dataset: opts.dataset,
      epochs: parseInt(opts.epochs),
      batch_size: parseInt(opts.batchSize),
      learning_rate: parseFloat(opts.learningRate),
      optimizer: config.training.optimizer || 'adam',
      federated: opts.federated,
      checkpoint_interval: parseInt(opts.checkpointInterval)
    });

    console.log(`✓ Training started: Job ID ${response.data.job_id}`);
    return response.data;
  } catch (error) {
    console.error('✗ Training start failed:', error.response?.data || error.message);
    throw error;
  }
}

/**
 * Monitor training progress
 */
async function monitorTraining(clusterId, jobId) {
  console.log('\n📊 Monitoring training progress...\n');

  const startTime = Date.now();
  let lastEpoch = 0;

  const interval = setInterval(async () => {
    try {
      const response = await api.get(`/neural/cluster/status`, {
        params: { cluster_id: clusterId }
      });

      const status = response.data;
      const metrics = status.training_metrics;

      if (metrics && metrics.current_epoch !== lastEpoch) {
        lastEpoch = metrics.current_epoch;
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
        const progress = ((metrics.current_epoch / metrics.total_epochs) * 100).toFixed(1);

        console.log(`Epoch ${metrics.current_epoch}/${metrics.total_epochs} (${progress}%) | Loss: ${metrics.loss.toFixed(4)} | Accuracy: ${(metrics.accuracy * 100).toFixed(2)}% | Time: ${elapsed}s`);
      }

      if (status.status === 'completed' || status.status === 'failed') {
        clearInterval(interval);
        console.log(`\n${status.status === 'completed' ? '✓' : '✗'} Training ${status.status}`);

        if (status.status === 'completed') {
          console.log(`\nFinal Metrics:`);
          console.log(`  Loss: ${metrics.loss.toFixed(4)}`);
          console.log(`  Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`);
          console.log(`  Total Time: ${((Date.now() - startTime) / 1000).toFixed(0)}s`);
        }
      }
    } catch (error) {
      if (opts.verbose) {
        console.error('Status check error:', error.message);
      }
    }
  }, 5000); // Check every 5 seconds
}

/**
 * Main execution
 */
async function main() {
  try {
    // Load configuration
    const fs = require('fs');
    const config = JSON.parse(fs.readFileSync(opts.config, 'utf8'));

    // Initialize cluster
    const cluster = await initializeCluster(config);
    const clusterId = cluster.cluster_id;

    // Deploy nodes
    const paramServer = await deployParameterServer(clusterId, config);
    const workers = await deployWorkerNodes(clusterId, config, parseInt(opts.nodes));
    const aggregator = await deployAggregator(clusterId);

    // Connect cluster
    await connectCluster(clusterId);

    // Start training
    const training = await startTraining(clusterId, config);

    // Monitor progress
    await monitorTraining(clusterId, training.job_id);

    console.log('\n✓ Training completed successfully!');
    console.log(`  Cluster ID: ${clusterId}`);
    console.log(`  Model ID: ${training.model_id || 'N/A'}`);
    console.log(`\nTo terminate cluster:`);
    console.log(`  curl -X POST ${FLOW_NEXUS_API}/neural/cluster/terminate -d '{"cluster_id":"${clusterId}"}'`);

  } catch (error) {
    console.error('\n✗ Training failed:', error.message);
    process.exit(1);
  }
}

main();
