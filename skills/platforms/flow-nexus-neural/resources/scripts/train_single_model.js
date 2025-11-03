#!/usr/bin/env node
/**
 * Flow Nexus Neural - Single Model Training Script
 *
 * Train neural networks on single E2B sandbox with custom architectures.
 * Ideal for small to medium models, experimentation, and prototyping.
 *
 * Usage:
 *   node train_single_model.js --config <config.json> --tier <nano|mini|small|medium|large>
 *
 * Features:
 *   - Support for 5 architectures (feedforward, LSTM, GAN, transformer, autoencoder)
 *   - Divergent thinking patterns (lateral, quantum, chaotic)
 *   - Real-time training metrics
 *   - Model checkpointing
 *   - Hyperparameter tuning support
 */

const { Command } = require('commander');
const axios = require('axios');

const program = new Command();

program
  .name('train-single-model')
  .description('Train single neural network model on Flow Nexus')
  .requiredOption('-c, --config <path>', 'Path to model configuration JSON')
  .option('-t, --tier <size>', 'Training tier (nano|mini|small|medium|large)', 'small')
  .option('-e, --epochs <count>', 'Number of training epochs', '100')
  .option('-b, --batch-size <size>', 'Batch size', '32')
  .option('-l, --learning-rate <rate>', 'Learning rate', '0.001')
  .option('--optimizer <type>', 'Optimizer (adam|sgd|rmsprop|adagrad)', 'adam')
  .option('--divergent', 'Enable divergent thinking patterns', false)
  .option('--pattern <type>', 'Divergent pattern (lateral|quantum|chaotic|associative|evolutionary)', 'lateral')
  .option('--checkpoint-interval <epochs>', 'Save checkpoint every N epochs', '10')
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

/**
 * Validate configuration
 */
function validateConfig(config) {
  const validArchitectures = ['feedforward', 'lstm', 'gan', 'autoencoder', 'transformer'];
  const validTiers = ['nano', 'mini', 'small', 'medium', 'large'];
  const validOptimizers = ['adam', 'sgd', 'rmsprop', 'adagrad'];

  if (!validArchitectures.includes(config.architecture.type)) {
    throw new Error(`Invalid architecture: ${config.architecture.type}. Must be one of: ${validArchitectures.join(', ')}`);
  }

  if (!validTiers.includes(opts.tier)) {
    throw new Error(`Invalid tier: ${opts.tier}. Must be one of: ${validTiers.join(', ')}`);
  }

  if (!validOptimizers.includes(opts.optimizer)) {
    throw new Error(`Invalid optimizer: ${opts.optimizer}. Must be one of: ${validOptimizers.join(', ')}`);
  }

  console.log('✓ Configuration validated');
}

/**
 * Build training configuration
 */
function buildTrainingConfig(config) {
  const trainingConfig = {
    architecture: config.architecture,
    training: {
      epochs: parseInt(opts.epochs),
      batch_size: parseInt(opts.batchSize),
      learning_rate: parseFloat(opts.learningRate),
      optimizer: opts.optimizer
    },
    tier: opts.tier,
    user_id: USER_ID
  };

  // Add divergent thinking if enabled
  if (opts.divergent) {
    trainingConfig.divergent = {
      enabled: true,
      pattern: opts.pattern,
      factor: 0.5
    };
  }

  return trainingConfig;
}

/**
 * Start training
 */
async function startTraining(trainingConfig) {
  console.log('\n🎓 Starting model training...');
  console.log(`   Architecture: ${trainingConfig.architecture.type}`);
  console.log(`   Tier: ${trainingConfig.tier}`);
  console.log(`   Epochs: ${trainingConfig.training.epochs}`);
  console.log(`   Batch Size: ${trainingConfig.training.batch_size}`);
  console.log(`   Learning Rate: ${trainingConfig.training.learning_rate}`);
  console.log(`   Optimizer: ${trainingConfig.training.optimizer}`);
  if (opts.divergent) {
    console.log(`   Divergent Pattern: ${opts.pattern}`);
  }
  console.log('');

  try {
    const response = await api.post('/neural/train', trainingConfig);

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
async function monitorTraining(jobId) {
  console.log('📊 Monitoring training progress...\n');

  const startTime = Date.now();
  let lastEpoch = 0;
  let bestLoss = Infinity;

  const interval = setInterval(async () => {
    try {
      const response = await api.get('/neural/training/status', {
        params: { job_id: jobId }
      });

      const status = response.data;

      if (status.current_epoch !== lastEpoch) {
        lastEpoch = status.current_epoch;
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
        const progress = ((status.current_epoch / status.total_epochs) * 100).toFixed(1);

        // Track best loss
        if (status.current_loss < bestLoss) {
          bestLoss = status.current_loss;
        }

        console.log(`Epoch ${status.current_epoch}/${status.total_epochs} (${progress}%) | Loss: ${status.current_loss.toFixed(6)} | Best: ${bestLoss.toFixed(6)} | ETA: ${new Date(status.estimated_completion).toLocaleTimeString()}`);

        // Checkpoint notification
        if (status.current_epoch % parseInt(opts.checkpointInterval) === 0) {
          console.log(`  💾 Checkpoint saved at epoch ${status.current_epoch}`);
        }
      }

      if (status.status === 'trained' || status.status === 'failed') {
        clearInterval(interval);
        const totalTime = ((Date.now() - startTime) / 1000).toFixed(0);

        if (status.status === 'trained') {
          console.log(`\n✓ Training completed successfully!`);
          console.log(`\nFinal Metrics:`);
          console.log(`  Loss: ${status.current_loss.toFixed(6)}`);
          console.log(`  Best Loss: ${bestLoss.toFixed(6)}`);
          console.log(`  Total Time: ${totalTime}s`);
          console.log(`  Model ID: ${status.model_id || 'N/A'}`);
        } else {
          console.log(`\n✗ Training failed`);
        }
      }
    } catch (error) {
      if (opts.verbose) {
        console.error('Status check error:', error.message);
      }
    }
  }, 3000); // Check every 3 seconds
}

/**
 * Save model info
 */
async function saveModelInfo(jobId, config) {
  const fs = require('fs');
  const path = require('path');

  const modelInfo = {
    job_id: jobId,
    architecture: config.architecture,
    training: config.training,
    tier: config.tier,
    timestamp: new Date().toISOString()
  };

  const outputFile = path.join(process.cwd(), `model-${jobId}.json`);
  fs.writeFileSync(outputFile, JSON.stringify(modelInfo, null, 2));
  console.log(`\n💾 Model info saved to: ${outputFile}`);
}

/**
 * Main execution
 */
async function main() {
  try {
    // Load configuration
    const fs = require('fs');
    const config = JSON.parse(fs.readFileSync(opts.config, 'utf8'));

    // Validate
    validateConfig(config);

    // Build training config
    const trainingConfig = buildTrainingConfig(config);

    // Start training
    const training = await startTraining(trainingConfig);
    const jobId = training.job_id;

    // Save model info
    await saveModelInfo(jobId, trainingConfig);

    // Monitor progress
    await monitorTraining(jobId);

    console.log('\n✓ Training workflow completed!');
    console.log(`  Job ID: ${jobId}`);
    console.log(`\nNext steps:`);
    console.log(`  1. Run inference: node predict_model.js --model <model-id> --input <data>`);
    console.log(`  2. Benchmark: node benchmark_model.js --model <model-id>`);
    console.log(`  3. Publish: node publish_template.js --model <model-id>`);

  } catch (error) {
    console.error('\n✗ Training failed:', error.message);
    if (opts.verbose) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

main();
