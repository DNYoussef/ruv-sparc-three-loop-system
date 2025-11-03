#!/usr/bin/env node
/**
 * Overfitting Detector
 *
 * Detects overfitting by analyzing train vs validation metrics and provides
 * diagnostic insights and regularization recommendations.
 *
 * Usage:
 *   node overfitting-detector.js --metrics metrics.json
 *   node overfitting-detector.js --train-log train.csv --val-log val.csv
 */

const fs = require('fs');
const path = require('path');

class OverfittingDetector {
  constructor(options = {}) {
    this.gapThreshold = options.gapThreshold || 0.1;
    this.windowSize = options.windowSize || 5;
    this.earlyStoppingPatience = options.earlyStoppingPatience || 10;
  }

  /**
   * Load metrics from JSON file
   */
  loadMetrics(filepath) {
    const data = JSON.parse(fs.readFileSync(filepath, 'utf8'));
    return data;
  }

  /**
   * Calculate train/val gap over time
   */
  calculateGap(trainMetrics, valMetrics) {
    const gaps = [];
    const minLength = Math.min(trainMetrics.length, valMetrics.length);

    for (let i = 0; i < minLength; i++) {
      const gap = Math.abs(trainMetrics[i] - valMetrics[i]);
      const relativeGap = gap / (Math.abs(valMetrics[i]) + 1e-8);

      gaps.push({
        epoch: i,
        absolute: gap,
        relative: relativeGap,
        train: trainMetrics[i],
        val: valMetrics[i]
      });
    }

    return gaps;
  }

  /**
   * Detect overfitting based on train/val gap
   */
  detectOverfitting(gaps) {
    const analysis = {
      isOverfitting: false,
      severity: 'none',
      onsetEpoch: null,
      maxGap: 0,
      avgGap: 0,
      trends: []
    };

    // Calculate statistics
    const relativeGaps = gaps.map(g => g.relative);
    analysis.avgGap = relativeGaps.reduce((a, b) => a + b, 0) / relativeGaps.length;
    analysis.maxGap = Math.max(...relativeGaps);

    // Detect onset
    for (let i = this.windowSize; i < gaps.length; i++) {
      const windowGaps = relativeGaps.slice(i - this.windowSize, i);
      const avgWindowGap = windowGaps.reduce((a, b) => a + b, 0) / windowGaps.length;

      if (avgWindowGap > this.gapThreshold && !analysis.isOverfitting) {
        analysis.isOverfitting = true;
        analysis.onsetEpoch = i - this.windowSize;
        break;
      }
    }

    // Determine severity
    if (analysis.maxGap > 0.5) {
      analysis.severity = 'severe';
    } else if (analysis.maxGap > 0.3) {
      analysis.severity = 'high';
    } else if (analysis.maxGap > 0.15) {
      analysis.severity = 'moderate';
    } else if (analysis.isOverfitting) {
      analysis.severity = 'mild';
    }

    // Analyze trends
    if (gaps.length > 10) {
      const recentGaps = relativeGaps.slice(-10);
      const trend = this._calculateTrend(recentGaps);
      analysis.trends.push({
        period: 'recent',
        direction: trend > 0.01 ? 'increasing' : trend < -0.01 ? 'decreasing' : 'stable',
        slope: trend
      });
    }

    return analysis;
  }

  /**
   * Detect early stopping opportunity
   */
  detectEarlyStopping(valMetrics) {
    const analysis = {
      shouldStop: false,
      bestEpoch: 0,
      bestMetric: valMetrics[0],
      patience: 0,
      improvement: []
    };

    let epochsSinceImprovement = 0;

    for (let i = 0; i < valMetrics.length; i++) {
      const current = valMetrics[i];
      const improved = current < analysis.bestMetric; // Assuming lower is better

      if (improved) {
        analysis.bestEpoch = i;
        analysis.bestMetric = current;
        epochsSinceImprovement = 0;
        analysis.improvement.push({
          epoch: i,
          value: current,
          delta: analysis.bestMetric - current
        });
      } else {
        epochsSinceImprovement++;
      }

      if (epochsSinceImprovement >= this.earlyStoppingPatience) {
        analysis.shouldStop = true;
        analysis.patience = epochsSinceImprovement;
        break;
      }
    }

    return analysis;
  }

  /**
   * Generate regularization recommendations
   */
  generateRecommendations(overfittingAnalysis, earlyStoppingAnalysis, trainMetrics, valMetrics) {
    const recommendations = [];

    // Overfitting recommendations
    if (overfittingAnalysis.isOverfitting) {
      recommendations.push({
        priority: 'high',
        category: 'overfitting',
        message: `⚠️ Overfitting detected starting at epoch ${overfittingAnalysis.onsetEpoch} (${overfittingAnalysis.severity} severity)`
      });

      if (overfittingAnalysis.severity === 'severe' || overfittingAnalysis.severity === 'high') {
        recommendations.push({
          priority: 'critical',
          category: 'regularization',
          message: '🛡️ Strong regularization needed:',
          actions: [
            'Increase dropout rate (try 0.3-0.5)',
            'Add L2 regularization (weight_decay=1e-4)',
            'Reduce model capacity (fewer layers/units)',
            'Use data augmentation',
            'Collect more training data if possible'
          ]
        });
      } else if (overfittingAnalysis.severity === 'moderate') {
        recommendations.push({
          priority: 'medium',
          category: 'regularization',
          message: '🔧 Moderate regularization recommended:',
          actions: [
            'Add dropout (0.1-0.3) or increase existing rate',
            'Apply L2 regularization (weight_decay=1e-5)',
            'Try early stopping',
            'Use batch normalization',
            'Implement label smoothing'
          ]
        });
      }
    }

    // Early stopping recommendations
    if (earlyStoppingAnalysis.shouldStop) {
      recommendations.push({
        priority: 'high',
        category: 'early_stopping',
        message: `⏹️ Training should stop - no improvement for ${earlyStoppingAnalysis.patience} epochs`,
        actions: [
          `Restore checkpoint from epoch ${earlyStoppingAnalysis.bestEpoch}`,
          `Best validation metric: ${earlyStoppingAnalysis.bestMetric.toFixed(6)}`
        ]
      });
    }

    // Learning rate recommendations
    const recentTrainLoss = trainMetrics.slice(-5);
    const trainImprovement = this._calculateTrend(recentTrainLoss);

    if (trainImprovement > -0.001 && !overfittingAnalysis.isOverfitting) {
      recommendations.push({
        priority: 'medium',
        category: 'learning_rate',
        message: '📉 Training plateau detected',
        actions: [
          'Reduce learning rate by 50%',
          'Try learning rate scheduling (cosine annealing)',
          'Check if model capacity is sufficient'
        ]
      });
    }

    // Data quality recommendations
    if (trainMetrics[trainMetrics.length - 1] > trainMetrics[0] * 0.8) {
      recommendations.push({
        priority: 'medium',
        category: 'data',
        message: '📊 Limited training progress',
        actions: [
          'Check data preprocessing and normalization',
          'Verify data quality and labels',
          'Ensure batch shuffling is enabled',
          'Review data augmentation strategy'
        ]
      });
    }

    // Success case
    if (recommendations.length === 0) {
      recommendations.push({
        priority: 'info',
        category: 'status',
        message: '✅ Training metrics look healthy',
        actions: [
          'Continue monitoring train/val gap',
          'Consider saving checkpoints regularly',
          'Track additional metrics (accuracy, F1, etc.)'
        ]
      });
    }

    return recommendations;
  }

  /**
   * Calculate linear trend
   */
  _calculateTrend(values) {
    const n = values.length;
    if (n < 2) return 0;

    const x = Array.from({ length: n }, (_, i) => i);
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = values.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * values[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    return slope;
  }

  /**
   * Generate comprehensive analysis
   */
  analyze(trainMetrics, valMetrics) {
    console.log('🔍 Analyzing overfitting patterns...\n');

    // Calculate gaps
    const gaps = this.calculateGap(trainMetrics, valMetrics);

    // Detect overfitting
    const overfittingAnalysis = this.detectOverfitting(gaps);

    // Detect early stopping opportunity
    const earlyStoppingAnalysis = this.detectEarlyStopping(valMetrics);

    // Generate recommendations
    const recommendations = this.generateRecommendations(
      overfittingAnalysis,
      earlyStoppingAnalysis,
      trainMetrics,
      valMetrics
    );

    return {
      gaps,
      overfitting: overfittingAnalysis,
      earlyStopping: earlyStoppingAnalysis,
      recommendations,
      summary: {
        totalEpochs: trainMetrics.length,
        finalTrainMetric: trainMetrics[trainMetrics.length - 1],
        finalValMetric: valMetrics[valMetrics.length - 1],
        finalGap: gaps[gaps.length - 1].relative,
        bestValEpoch: earlyStoppingAnalysis.bestEpoch,
        bestValMetric: earlyStoppingAnalysis.bestMetric
      }
    };
  }

  /**
   * Print analysis report
   */
  printReport(analysis) {
    console.log('📊 Overfitting Detection Report');
    console.log('=' .repeat(60));

    console.log('\n📈 Summary:');
    console.log(`  Total epochs: ${analysis.summary.totalEpochs}`);
    console.log(`  Final train metric: ${analysis.summary.finalTrainMetric.toFixed(6)}`);
    console.log(`  Final val metric: ${analysis.summary.finalValMetric.toFixed(6)}`);
    console.log(`  Final train/val gap: ${(analysis.summary.finalGap * 100).toFixed(2)}%`);
    console.log(`  Best val metric: ${analysis.summary.bestValMetric.toFixed(6)} (epoch ${analysis.summary.bestValEpoch})`);

    console.log('\n🔍 Overfitting Analysis:');
    console.log(`  Status: ${analysis.overfitting.isOverfitting ? '⚠️ DETECTED' : '✅ Not detected'}`);
    if (analysis.overfitting.isOverfitting) {
      console.log(`  Severity: ${analysis.overfitting.severity}`);
      console.log(`  Onset epoch: ${analysis.overfitting.onsetEpoch}`);
      console.log(`  Max gap: ${(analysis.overfitting.maxGap * 100).toFixed(2)}%`);
      console.log(`  Avg gap: ${(analysis.overfitting.avgGap * 100).toFixed(2)}%`);
    }

    console.log('\n⏹️ Early Stopping:');
    console.log(`  Should stop: ${analysis.earlyStopping.shouldStop ? '✅ Yes' : '❌ No'}`);
    if (analysis.earlyStopping.shouldStop) {
      console.log(`  Patience exceeded: ${analysis.earlyStopping.patience} epochs`);
      console.log(`  Best epoch: ${analysis.earlyStopping.bestEpoch}`);
    }

    console.log('\n💡 Recommendations:');
    for (const rec of analysis.recommendations) {
      console.log(`\n  [${rec.priority.toUpperCase()}] ${rec.message}`);
      if (rec.actions) {
        for (const action of rec.actions) {
          console.log(`    • ${action}`);
        }
      }
    }
  }
}

// CLI
function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help')) {
    console.log(`
Overfitting Detector

Usage:
  node overfitting-detector.js --metrics <file.json>
  node overfitting-detector.js --train-log <train.csv> --val-log <val.csv>

Options:
  --metrics <file>       JSON file with train/val metrics
  --train-log <file>     CSV file with training metrics
  --val-log <file>       CSV file with validation metrics
  --gap-threshold <num>  Threshold for overfitting detection (default: 0.1)
  --window <num>         Window size for gap analysis (default: 5)
  --patience <num>       Early stopping patience (default: 10)
  --output <file>        Save analysis as JSON
  --help                 Show this message
    `);
    process.exit(0);
  }

  const options = {};
  let metricsFile, trainLogFile, valLogFile, outputFile;

  for (let i = 0; i < args.length; i += 2) {
    const flag = args[i];
    const value = args[i + 1];

    switch (flag) {
      case '--metrics':
        metricsFile = value;
        break;
      case '--train-log':
        trainLogFile = value;
        break;
      case '--val-log':
        valLogFile = value;
        break;
      case '--gap-threshold':
        options.gapThreshold = parseFloat(value);
        break;
      case '--window':
        options.windowSize = parseInt(value);
        break;
      case '--patience':
        options.earlyStoppingPatience = parseInt(value);
        break;
      case '--output':
        outputFile = value;
        break;
    }
  }

  const detector = new OverfittingDetector(options);

  let trainMetrics, valMetrics;

  if (metricsFile) {
    const data = detector.loadMetrics(metricsFile);
    trainMetrics = data.train || data.train_loss || data.trainLoss;
    valMetrics = data.val || data.val_loss || data.valLoss;
  } else if (trainLogFile && valLogFile) {
    // Simple CSV parsing (assumes single column of loss values)
    trainMetrics = fs.readFileSync(trainLogFile, 'utf8')
      .trim().split('\n').map(parseFloat);
    valMetrics = fs.readFileSync(valLogFile, 'utf8')
      .trim().split('\n').map(parseFloat);
  } else {
    console.error('Error: Must provide --metrics or --train-log and --val-log');
    process.exit(1);
  }

  // Run analysis
  const analysis = detector.analyze(trainMetrics, valMetrics);

  // Print report
  detector.printReport(analysis);

  // Save JSON if requested
  if (outputFile) {
    fs.writeFileSync(outputFile, JSON.stringify(analysis, null, 2));
    console.log(`\n💾 Analysis saved to ${outputFile}`);
  }
}

if (require.main === module) {
  main();
}

module.exports = OverfittingDetector;
