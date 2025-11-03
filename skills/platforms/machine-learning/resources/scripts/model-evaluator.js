#!/usr/bin/env node
/**
 * Model Evaluation Framework
 * Comprehensive evaluation with metrics, fairness analysis, and interpretability
 */

const fs = require('fs');
const path = require('path');

class ModelEvaluator {
  /**
   * Initialize model evaluator
   * @param {Object} config - Evaluation configuration
   */
  constructor(config = {}) {
    this.config = {
      metricsPath: config.metricsPath || 'resources/templates/evaluation-metrics.yaml',
      outputDir: config.outputDir || 'evaluation_results',
      fairnessEnabled: config.fairnessEnabled !== false,
      interpretabilityEnabled: config.interpretabilityEnabled !== false,
      ...config
    };

    this.results = {
      performance: {},
      fairness: {},
      interpretability: {},
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Calculate classification metrics
   * @param {Array} predictions - Model predictions
   * @param {Array} groundTruth - Ground truth labels
   * @returns {Object} Metrics object
   */
  calculateClassificationMetrics(predictions, groundTruth) {
    if (predictions.length !== groundTruth.length) {
      throw new Error('Predictions and ground truth must have same length');
    }

    // Initialize confusion matrix
    const uniqueLabels = [...new Set([...predictions, ...groundTruth])];
    const numClasses = uniqueLabels.length;
    const confusionMatrix = Array(numClasses).fill(0).map(() => Array(numClasses).fill(0));

    // Build confusion matrix
    predictions.forEach((pred, idx) => {
      const predIdx = uniqueLabels.indexOf(pred);
      const trueIdx = uniqueLabels.indexOf(groundTruth[idx]);
      confusionMatrix[trueIdx][predIdx]++;
    });

    // Calculate metrics per class
    const perClassMetrics = uniqueLabels.map((label, idx) => {
      const tp = confusionMatrix[idx][idx];
      const fp = confusionMatrix.reduce((sum, row, i) =>
        i !== idx ? sum + row[idx] : sum, 0);
      const fn = confusionMatrix[idx].reduce((sum, val, i) =>
        i !== idx ? sum + val : sum, 0);
      const tn = confusionMatrix.reduce((sum, row, i) =>
        i !== idx ? sum + row.reduce((s, v, j) => j !== idx ? s + v : s, 0) : sum, 0);

      const precision = tp / (tp + fp) || 0;
      const recall = tp / (tp + fn) || 0;
      const f1 = 2 * (precision * recall) / (precision + recall) || 0;
      const specificity = tn / (tn + fp) || 0;

      return {
        label,
        precision,
        recall,
        f1,
        specificity,
        support: tp + fn
      };
    });

    // Calculate macro and weighted averages
    const totalSupport = perClassMetrics.reduce((sum, m) => sum + m.support, 0);

    const macroAvg = {
      precision: perClassMetrics.reduce((sum, m) => sum + m.precision, 0) / numClasses,
      recall: perClassMetrics.reduce((sum, m) => sum + m.recall, 0) / numClasses,
      f1: perClassMetrics.reduce((sum, m) => sum + m.f1, 0) / numClasses
    };

    const weightedAvg = {
      precision: perClassMetrics.reduce((sum, m) => sum + m.precision * m.support, 0) / totalSupport,
      recall: perClassMetrics.reduce((sum, m) => sum + m.recall * m.support, 0) / totalSupport,
      f1: perClassMetrics.reduce((sum, m) => sum + m.f1 * m.support, 0) / totalSupport
    };

    // Calculate overall accuracy
    const accuracy = predictions.filter((pred, idx) => pred === groundTruth[idx]).length / predictions.length;

    return {
      accuracy,
      confusionMatrix,
      perClassMetrics,
      macroAvg,
      weightedAvg,
      labels: uniqueLabels
    };
  }

  /**
   * Calculate regression metrics
   * @param {Array} predictions - Model predictions
   * @param {Array} groundTruth - Ground truth values
   * @returns {Object} Metrics object
   */
  calculateRegressionMetrics(predictions, groundTruth) {
    if (predictions.length !== groundTruth.length) {
      throw new Error('Predictions and ground truth must have same length');
    }

    const n = predictions.length;

    // Mean Squared Error (MSE)
    const mse = predictions.reduce((sum, pred, idx) => {
      const error = pred - groundTruth[idx];
      return sum + error * error;
    }, 0) / n;

    // Root Mean Squared Error (RMSE)
    const rmse = Math.sqrt(mse);

    // Mean Absolute Error (MAE)
    const mae = predictions.reduce((sum, pred, idx) => {
      return sum + Math.abs(pred - groundTruth[idx]);
    }, 0) / n;

    // R-squared
    const meanY = groundTruth.reduce((sum, val) => sum + val, 0) / n;
    const ssTotal = groundTruth.reduce((sum, val) => {
      const diff = val - meanY;
      return sum + diff * diff;
    }, 0);
    const ssResidual = predictions.reduce((sum, pred, idx) => {
      const error = groundTruth[idx] - pred;
      return sum + error * error;
    }, 0);
    const r2 = 1 - (ssResidual / ssTotal);

    // Mean Absolute Percentage Error (MAPE)
    const mape = predictions.reduce((sum, pred, idx) => {
      if (groundTruth[idx] === 0) return sum;
      return sum + Math.abs((groundTruth[idx] - pred) / groundTruth[idx]);
    }, 0) / n * 100;

    return {
      mse,
      rmse,
      mae,
      r2,
      mape
    };
  }

  /**
   * Analyze fairness across demographic groups
   * @param {Array} predictions - Model predictions
   * @param {Array} groundTruth - Ground truth labels
   * @param {Array} sensitiveAttributes - Sensitive attribute values (e.g., gender, race)
   * @returns {Object} Fairness metrics
   */
  analyzeFairness(predictions, groundTruth, sensitiveAttributes) {
    if (!this.config.fairnessEnabled) {
      return { enabled: false };
    }

    const groups = [...new Set(sensitiveAttributes)];
    const fairnessMetrics = {};

    groups.forEach(group => {
      const groupIndices = sensitiveAttributes
        .map((attr, idx) => attr === group ? idx : -1)
        .filter(idx => idx !== -1);

      const groupPredictions = groupIndices.map(idx => predictions[idx]);
      const groupGroundTruth = groupIndices.map(idx => groundTruth[idx]);

      // Positive prediction rate
      const positivePredictionRate = groupPredictions.filter(p => p === 1).length / groupPredictions.length;

      // True positive rate (TPR)
      const truePositives = groupIndices.filter(idx =>
        predictions[idx] === 1 && groundTruth[idx] === 1
      ).length;
      const actualPositives = groupGroundTruth.filter(gt => gt === 1).length;
      const tpr = actualPositives > 0 ? truePositives / actualPositives : 0;

      // False positive rate (FPR)
      const falsePositives = groupIndices.filter(idx =>
        predictions[idx] === 1 && groundTruth[idx] === 0
      ).length;
      const actualNegatives = groupGroundTruth.filter(gt => gt === 0).length;
      const fpr = actualNegatives > 0 ? falsePositives / actualNegatives : 0;

      fairnessMetrics[group] = {
        size: groupIndices.length,
        positivePredictionRate,
        truePositiveRate: tpr,
        falsePositiveRate: fpr
      };
    });

    // Calculate fairness disparities
    const pprs = Object.values(fairnessMetrics).map(m => m.positivePredictionRate);
    const tprs = Object.values(fairnessMetrics).map(m => m.truePositiveRate);

    const disparities = {
      demographicParity: Math.max(...pprs) - Math.min(...pprs),
      equalizedOdds: Math.max(...tprs) - Math.min(...tprs)
    };

    return {
      perGroup: fairnessMetrics,
      disparities,
      groups
    };
  }

  /**
   * Generate feature importance analysis
   * @param {Object} model - Model object with feature importance
   * @param {Array} featureNames - Names of features
   * @returns {Object} Feature importance rankings
   */
  analyzeFeatureImportance(model, featureNames) {
    if (!this.config.interpretabilityEnabled) {
      return { enabled: false };
    }

    // Mock implementation - in practice, this would use SHAP or LIME
    const importance = featureNames.map((name, idx) => ({
      feature: name,
      importance: model.featureImportances ? model.featureImportances[idx] : Math.random(),
      rank: idx + 1
    }));

    // Sort by importance
    importance.sort((a, b) => b.importance - a.importance);

    // Update ranks
    importance.forEach((item, idx) => {
      item.rank = idx + 1;
    });

    return {
      rankings: importance,
      topFeatures: importance.slice(0, 10)
    };
  }

  /**
   * Evaluate model comprehensively
   * @param {Object} params - Evaluation parameters
   * @returns {Object} Complete evaluation results
   */
  async evaluate(params) {
    const {
      predictions,
      groundTruth,
      taskType = 'classification',
      sensitiveAttributes = null,
      model = null,
      featureNames = null
    } = params;

    console.log('Starting model evaluation...');

    // Performance metrics
    if (taskType === 'classification') {
      this.results.performance = this.calculateClassificationMetrics(predictions, groundTruth);
      console.log(`Accuracy: ${(this.results.performance.accuracy * 100).toFixed(2)}%`);
    } else if (taskType === 'regression') {
      this.results.performance = this.calculateRegressionMetrics(predictions, groundTruth);
      console.log(`R²: ${this.results.performance.r2.toFixed(4)}`);
      console.log(`RMSE: ${this.results.performance.rmse.toFixed(4)}`);
    }

    // Fairness analysis
    if (sensitiveAttributes && this.config.fairnessEnabled) {
      this.results.fairness = this.analyzeFairness(predictions, groundTruth, sensitiveAttributes);
      console.log('Fairness analysis complete');
    }

    // Interpretability
    if (model && featureNames && this.config.interpretabilityEnabled) {
      this.results.interpretability = this.analyzeFeatureImportance(model, featureNames);
      console.log(`Top feature: ${this.results.interpretability.topFeatures[0].feature}`);
    }

    // Save results
    await this.saveResults();

    return this.results;
  }

  /**
   * Save evaluation results to file
   */
  async saveResults() {
    const outputDir = this.config.outputDir;

    // Create output directory if it doesn't exist
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const outputPath = path.join(outputDir, `evaluation_${timestamp}.json`);

    fs.writeFileSync(outputPath, JSON.stringify(this.results, null, 2));
    console.log(`Results saved to: ${outputPath}`);

    // Generate summary report
    this.generateReport(outputPath.replace('.json', '_report.txt'));
  }

  /**
   * Generate human-readable evaluation report
   * @param {string} outputPath - Path to save report
   */
  generateReport(outputPath) {
    let report = '='.repeat(80) + '\n';
    report += 'MODEL EVALUATION REPORT\n';
    report += '='.repeat(80) + '\n\n';
    report += `Timestamp: ${this.results.timestamp}\n\n`;

    // Performance section
    report += 'PERFORMANCE METRICS\n';
    report += '-'.repeat(80) + '\n';
    if (this.results.performance.accuracy !== undefined) {
      report += `Accuracy: ${(this.results.performance.accuracy * 100).toFixed(2)}%\n`;
      report += `Macro F1: ${(this.results.performance.macroAvg.f1 * 100).toFixed(2)}%\n`;
      report += `Weighted F1: ${(this.results.performance.weightedAvg.f1 * 100).toFixed(2)}%\n`;
    } else if (this.results.performance.r2 !== undefined) {
      report += `R²: ${this.results.performance.r2.toFixed(4)}\n`;
      report += `RMSE: ${this.results.performance.rmse.toFixed(4)}\n`;
      report += `MAE: ${this.results.performance.mae.toFixed(4)}\n`;
    }
    report += '\n';

    // Fairness section
    if (this.results.fairness && this.results.fairness.perGroup) {
      report += 'FAIRNESS ANALYSIS\n';
      report += '-'.repeat(80) + '\n';
      report += `Demographic Parity Gap: ${(this.results.fairness.disparities.demographicParity * 100).toFixed(2)}%\n`;
      report += `Equalized Odds Gap: ${(this.results.fairness.disparities.equalizedOdds * 100).toFixed(2)}%\n\n`;

      Object.entries(this.results.fairness.perGroup).forEach(([group, metrics]) => {
        report += `Group: ${group}\n`;
        report += `  Size: ${metrics.size}\n`;
        report += `  Positive Prediction Rate: ${(metrics.positivePredictionRate * 100).toFixed(2)}%\n`;
        report += `  True Positive Rate: ${(metrics.truePositiveRate * 100).toFixed(2)}%\n\n`;
      });
    }

    // Interpretability section
    if (this.results.interpretability && this.results.interpretability.topFeatures) {
      report += 'TOP 10 IMPORTANT FEATURES\n';
      report += '-'.repeat(80) + '\n';
      this.results.interpretability.topFeatures.forEach((item, idx) => {
        report += `${idx + 1}. ${item.feature}: ${item.importance.toFixed(4)}\n`;
      });
    }

    fs.writeFileSync(outputPath, report);
    console.log(`Report saved to: ${outputPath}`);
  }
}

// Example usage
async function main() {
  const evaluator = new ModelEvaluator({
    fairnessEnabled: true,
    interpretabilityEnabled: true
  });

  // Example classification data
  const predictions = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1];
  const groundTruth = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1];
  const sensitiveAttributes = ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'B'];

  const results = await evaluator.evaluate({
    predictions,
    groundTruth,
    taskType: 'classification',
    sensitiveAttributes,
    model: { featureImportances: [0.3, 0.2, 0.15, 0.1, 0.25] },
    featureNames: ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
  });

  console.log('\nEvaluation complete!');
}

if (require.main === module) {
  main().catch(console.error);
}

module.exports = ModelEvaluator;
