#!/usr/bin/env node
/**
 * Flow Nexus Neural - Model Benchmarking Script
 *
 * Run comprehensive performance benchmarks on trained models.
 * Measure inference latency, throughput, memory usage, and accuracy.
 *
 * Usage:
 *   node benchmark_model.js --model <model-id> --type comprehensive
 *
 * Features:
 *   - Inference latency benchmarking (p50, p95, p99)
 *   - Throughput testing (queries per second)
 *   - Memory profiling (peak usage, leaks)
 *   - GPU utilization monitoring
 *   - Accuracy validation on test sets
 *   - Export results to JSON/CSV
 */

const { Command } = require('commander');
const axios = require('axios');
const Table = require('cli-table3');

const program = new Command();

program
  .name('benchmark-model')
  .description('Run performance benchmarks on neural network models')
  .requiredOption('-m, --model <id>', 'Model ID to benchmark')
  .option('-t, --type <type>', 'Benchmark type (inference|throughput|memory|comprehensive)', 'comprehensive')
  .option('-i, --iterations <count>', 'Number of benchmark iterations', '1000')
  .option('--warmup <count>', 'Warmup iterations before benchmarking', '100')
  .option('--export <format>', 'Export format (json|csv)', 'json')
  .option('--output <path>', 'Output file path')
  .option('--verbose', 'Enable verbose logging', false)
  .parse(process.argv);

const opts = program.opts();

// Flow Nexus API configuration
const FLOW_NEXUS_API = process.env.FLOW_NEXUS_API || 'https://api.flow-nexus.ruv.io';
const API_KEY = process.env.FLOW_NEXUS_API_KEY;

if (!API_KEY) {
  console.error('Error: FLOW_NEXUS_API_KEY environment variable required');
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
 * Run benchmark
 */
async function runBenchmark(modelId, type) {
  console.log('\n⚡ Running performance benchmarks...');
  console.log(`   Model: ${modelId}`);
  console.log(`   Type: ${type}`);
  console.log(`   Iterations: ${opts.iterations}`);
  console.log(`   Warmup: ${opts.warmup}\n`);

  try {
    const response = await api.post('/neural/benchmark', {
      model_id: modelId,
      benchmark_type: type,
      iterations: parseInt(opts.iterations),
      warmup_iterations: parseInt(opts.warmup)
    });

    return response.data;
  } catch (error) {
    console.error('✗ Benchmark failed:', error.response?.data || error.message);
    throw error;
  }
}

/**
 * Display benchmark results
 */
function displayResults(results) {
  console.log('📊 Benchmark Results\n');
  console.log('=' .repeat(80));

  // Inference Latency
  if (results.benchmarks.inference_latency_ms !== undefined) {
    console.log('\n🚀 Inference Latency:');
    const table = new Table({
      head: ['Metric', 'Value'],
      colWidths: [30, 50]
    });

    table.push(
      ['Mean Latency', `${results.benchmarks.inference_latency_ms.toFixed(2)} ms`],
      ['P50 Latency', `${results.benchmarks.p50_latency_ms?.toFixed(2) || 'N/A'} ms`],
      ['P95 Latency', `${results.benchmarks.p95_latency_ms?.toFixed(2) || 'N/A'} ms`],
      ['P99 Latency', `${results.benchmarks.p99_latency_ms?.toFixed(2) || 'N/A'} ms`],
      ['Min Latency', `${results.benchmarks.min_latency_ms?.toFixed(2) || 'N/A'} ms`],
      ['Max Latency', `${results.benchmarks.max_latency_ms?.toFixed(2) || 'N/A'} ms`]
    );

    console.log(table.toString());
  }

  // Throughput
  if (results.benchmarks.throughput_qps !== undefined) {
    console.log('\n📈 Throughput:');
    const table = new Table({
      head: ['Metric', 'Value'],
      colWidths: [30, 50]
    });

    table.push(
      ['Queries Per Second', `${results.benchmarks.throughput_qps.toLocaleString()} QPS`],
      ['Batch Throughput', `${results.benchmarks.batch_throughput?.toLocaleString() || 'N/A'} samples/sec`]
    );

    console.log(table.toString());
  }

  // Memory Usage
  if (results.benchmarks.memory_usage_mb !== undefined) {
    console.log('\n💾 Memory Usage:');
    const table = new Table({
      head: ['Metric', 'Value'],
      colWidths: [30, 50]
    });

    table.push(
      ['Peak Memory', `${results.benchmarks.memory_usage_mb.toFixed(2)} MB`],
      ['Average Memory', `${results.benchmarks.avg_memory_mb?.toFixed(2) || 'N/A'} MB`],
      ['Memory Efficiency', `${results.benchmarks.memory_efficiency?.toFixed(2) || 'N/A'}%`]
    );

    console.log(table.toString());
  }

  // GPU Utilization
  if (results.benchmarks.gpu_utilization !== undefined) {
    console.log('\n🎮 GPU Metrics:');
    const table = new Table({
      head: ['Metric', 'Value'],
      colWidths: [30, 50]
    });

    table.push(
      ['GPU Utilization', `${(results.benchmarks.gpu_utilization * 100).toFixed(2)}%`],
      ['GPU Memory Used', `${results.benchmarks.gpu_memory_mb?.toFixed(2) || 'N/A'} MB`],
      ['Compute Efficiency', `${results.benchmarks.compute_efficiency?.toFixed(2) || 'N/A'}%`]
    );

    console.log(table.toString());
  }

  // Accuracy Metrics
  if (results.benchmarks.accuracy !== undefined) {
    console.log('\n🎯 Accuracy Metrics:');
    const table = new Table({
      head: ['Metric', 'Value'],
      colWidths: [30, 50]
    });

    table.push(
      ['Accuracy', `${(results.benchmarks.accuracy * 100).toFixed(2)}%`],
      ['F1 Score', `${(results.benchmarks.f1_score * 100).toFixed(2)}%`],
      ['Precision', `${((results.benchmarks.precision || 0) * 100).toFixed(2)}%`],
      ['Recall', `${((results.benchmarks.recall || 0) * 100).toFixed(2)}%`]
    );

    console.log(table.toString());
  }

  console.log('\n' + '='.repeat(80));
  console.log(`\nTimestamp: ${results.timestamp}`);
  console.log(`Model ID: ${results.model_id}`);
}

/**
 * Export results
 */
function exportResults(results, format, outputPath) {
  const fs = require('fs');
  const path = require('path');

  const filename = outputPath || `benchmark-${results.model_id}-${Date.now()}.${format}`;

  if (format === 'json') {
    fs.writeFileSync(filename, JSON.stringify(results, null, 2));
  } else if (format === 'csv') {
    const rows = [];
    rows.push('Metric,Value');

    const benchmarks = results.benchmarks;
    Object.entries(benchmarks).forEach(([key, value]) => {
      const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      rows.push(`${formattedKey},${value}`);
    });

    fs.writeFileSync(filename, rows.join('\n'));
  }

  console.log(`\n💾 Results exported to: ${filename}`);
}

/**
 * Compare with baseline
 */
function compareWithBaseline(results) {
  // Common performance baselines
  const baselines = {
    inference_latency_ms: 50, // 50ms target
    throughput_qps: 1000, // 1000 QPS target
    memory_usage_mb: 500, // 500MB target
    accuracy: 0.90 // 90% accuracy target
  };

  console.log('\n📊 Performance vs. Baselines:\n');

  const table = new Table({
    head: ['Metric', 'Actual', 'Baseline', 'Status'],
    colWidths: [30, 15, 15, 10]
  });

  Object.entries(baselines).forEach(([metric, baseline]) => {
    const actual = results.benchmarks[metric];
    if (actual !== undefined) {
      let status = '✓';
      let comparison;

      // Lower is better for latency and memory
      if (metric.includes('latency') || metric.includes('memory')) {
        comparison = actual <= baseline;
        status = comparison ? '✓' : '✗';
      } else {
        // Higher is better for throughput and accuracy
        comparison = actual >= baseline;
        status = comparison ? '✓' : '✗';
      }

      table.push([
        metric.replace(/_/g, ' '),
        actual.toFixed(2),
        baseline.toFixed(2),
        status
      ]);
    }
  });

  console.log(table.toString());
}

/**
 * Main execution
 */
async function main() {
  try {
    // Run benchmark
    const results = await runBenchmark(opts.model, opts.type);

    // Display results
    displayResults(results);

    // Compare with baselines
    compareWithBaseline(results);

    // Export if requested
    if (opts.export) {
      exportResults(results, opts.export, opts.output);
    }

    console.log('\n✓ Benchmark completed successfully!\n');

  } catch (error) {
    console.error('\n✗ Benchmark failed:', error.message);
    if (opts.verbose) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

main();
