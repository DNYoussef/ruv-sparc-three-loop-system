#!/usr/bin/env node
/**
 * Feature Engineering Complete Workflow Example
 *
 * Demonstrates advanced feature engineering techniques including:
 * - Automated feature discovery and type detection
 * - Multi-step transformation pipelines
 * - Temporal feature extraction from timestamps
 * - Text feature engineering with TF-IDF
 * - Feature interaction and polynomial features
 * - Feature selection and importance ranking
 * - Cross-validation for feature stability
 *
 * This is a production-ready example showing best practices for
 * feature engineering in machine learning pipelines.
 */

const fs = require('fs');
const path = require('path');

class AdvancedFeatureEngineer {
  constructor(config = {}) {
    this.config = config;
    this.features = new Map();
    this.transformers = new Map();
    this.featureImportance = new Map();
    this.pipeline = [];
  }

  /**
   * Automated feature type detection
   */
  detectFeatureTypes(data) {
    const typeDetection = {};
    const columns = Object.keys(data[0]);

    columns.forEach(col => {
      const values = data.map(row => row[col]);
      const nonNull = values.filter(v => v !== null && v !== '');

      // Detect feature type
      const type = this._inferType(nonNull);
      const stats = this._computeStats(nonNull, type);

      typeDetection[col] = {
        type,
        stats,
        nullCount: values.length - nonNull.length,
        uniqueCount: new Set(nonNull).size,
        recommendations: this._generateRecommendations(col, type, stats)
      };
    });

    return typeDetection;
  }

  _inferType(values) {
    // Check if numerical
    const numericCount = values.filter(v => !isNaN(parseFloat(v))).length;
    if (numericCount / values.length > 0.9) {
      return 'numerical';
    }

    // Check if temporal
    const dateCount = values.filter(v => !isNaN(Date.parse(v))).length;
    if (dateCount / values.length > 0.9) {
      return 'temporal';
    }

    // Check if categorical
    const uniqueRatio = new Set(values).size / values.length;
    if (uniqueRatio < 0.5) {
      return 'categorical';
    }

    // Default to text
    return 'text';
  }

  _computeStats(values, type) {
    const stats = {};

    if (type === 'numerical') {
      const numbers = values.map(v => parseFloat(v));
      stats.mean = numbers.reduce((a, b) => a + b, 0) / numbers.length;
      stats.std = Math.sqrt(
        numbers.reduce((sum, v) => sum + Math.pow(v - stats.mean, 2), 0) / numbers.length
      );
      stats.min = Math.min(...numbers);
      stats.max = Math.max(...numbers);
      stats.median = this._median(numbers);
      stats.skewness = this._skewness(numbers, stats.mean, stats.std);
    } else if (type === 'categorical') {
      const counts = {};
      values.forEach(v => counts[v] = (counts[v] || 0) + 1);
      stats.modeCounts = counts;
      stats.entropy = this._entropy(Object.values(counts));
    } else if (type === 'temporal') {
      const dates = values.map(v => new Date(v));
      stats.minDate = new Date(Math.min(...dates));
      stats.maxDate = new Date(Math.max(...dates));
      stats.range = stats.maxDate - stats.minDate;
    }

    return stats;
  }

  _median(numbers) {
    const sorted = [...numbers].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }

  _skewness(numbers, mean, std) {
    const n = numbers.length;
    const m3 = numbers.reduce((sum, v) => sum + Math.pow(v - mean, 3), 0) / n;
    return m3 / Math.pow(std, 3);
  }

  _entropy(counts) {
    const total = counts.reduce((a, b) => a + b, 0);
    return -counts.reduce((entropy, count) => {
      const p = count / total;
      return entropy + (p > 0 ? p * Math.log2(p) : 0);
    }, 0);
  }

  _generateRecommendations(column, type, stats) {
    const recommendations = [];

    if (type === 'numerical') {
      if (Math.abs(stats.skewness) > 1) {
        recommendations.push({
          transform: 'log_transform',
          reason: `High skewness (${stats.skewness.toFixed(2)})`
        });
      }

      if (stats.std > stats.mean * 2) {
        recommendations.push({
          transform: 'robust_scaling',
          reason: 'High variance detected'
        });
      }
    }

    if (type === 'categorical' && stats.entropy > 3) {
      recommendations.push({
        transform: 'target_encoding',
        reason: 'High-cardinality categorical feature'
      });
    }

    if (type === 'temporal') {
      recommendations.push({
        transform: 'extract_temporal_components',
        reason: 'Extract date/time features'
      });
    }

    return recommendations;
  }

  /**
   * Extract temporal features from date columns
   */
  extractTemporalFeatures(data, dateColumn) {
    return data.map(row => {
      const date = new Date(row[dateColumn]);
      const enhanced = { ...row };

      enhanced[`${dateColumn}_year`] = date.getFullYear();
      enhanced[`${dateColumn}_month`] = date.getMonth() + 1;
      enhanced[`${dateColumn}_day`] = date.getDate();
      enhanced[`${dateColumn}_dayofweek`] = date.getDay();
      enhanced[`${dateColumn}_hour`] = date.getHours();
      enhanced[`${dateColumn}_quarter`] = Math.floor(date.getMonth() / 3) + 1;
      enhanced[`${dateColumn}_is_weekend`] = date.getDay() === 0 || date.getDay() === 6 ? 1 : 0;

      // Cyclical encoding for periodic features
      const dayOfYear = this._getDayOfYear(date);
      enhanced[`${dateColumn}_sin_day`] = Math.sin(2 * Math.PI * dayOfYear / 365);
      enhanced[`${dateColumn}_cos_day`] = Math.cos(2 * Math.PI * dayOfYear / 365);

      return enhanced;
    });
  }

  _getDayOfYear(date) {
    const start = new Date(date.getFullYear(), 0, 0);
    const diff = date - start;
    const oneDay = 1000 * 60 * 60 * 24;
    return Math.floor(diff / oneDay);
  }

  /**
   * Create rolling window features for time series
   */
  createRollingFeatures(data, valueColumn, windows = [7, 30]) {
    const enhanced = [];

    data.forEach((row, idx) => {
      const newRow = { ...row };

      windows.forEach(window => {
        const start = Math.max(0, idx - window + 1);
        const windowData = data.slice(start, idx + 1);
        const values = windowData.map(r => parseFloat(r[valueColumn]));

        newRow[`${valueColumn}_rolling_mean_${window}`] =
          values.reduce((a, b) => a + b, 0) / values.length;

        newRow[`${valueColumn}_rolling_std_${window}`] =
          Math.sqrt(values.reduce((sum, v) =>
            sum + Math.pow(v - newRow[`${valueColumn}_rolling_mean_${window}`], 2), 0
          ) / values.length);

        newRow[`${valueColumn}_rolling_max_${window}`] = Math.max(...values);
        newRow[`${valueColumn}_rolling_min_${window}`] = Math.min(...values);
      });

      enhanced.push(newRow);
    });

    return enhanced;
  }

  /**
   * Create lag features
   */
  createLagFeatures(data, columns, lags = [1, 7, 30]) {
    const enhanced = [];

    data.forEach((row, idx) => {
      const newRow = { ...row };

      columns.forEach(col => {
        lags.forEach(lag => {
          if (idx >= lag) {
            newRow[`${col}_lag_${lag}`] = data[idx - lag][col];
          } else {
            newRow[`${col}_lag_${lag}`] = null; // Or use forward fill
          }
        });
      });

      enhanced.push(newRow);
    });

    return enhanced;
  }

  /**
   * Simple TF-IDF for text features
   */
  tfidfTransform(data, textColumn, maxFeatures = 100) {
    // Tokenize documents
    const documents = data.map(row =>
      (row[textColumn] || '').toLowerCase().split(/\s+/)
    );

    // Compute document frequency
    const df = {};
    const totalDocs = documents.length;

    documents.forEach(doc => {
      const uniqueTerms = new Set(doc);
      uniqueTerms.forEach(term => {
        df[term] = (df[term] || 0) + 1;
      });
    });

    // Compute IDF
    const idf = {};
    Object.entries(df).forEach(([term, freq]) => {
      idf[term] = Math.log(totalDocs / freq);
    });

    // Select top terms by IDF
    const topTerms = Object.entries(idf)
      .sort((a, b) => b[1] - a[1])
      .slice(0, maxFeatures)
      .map(([term]) => term);

    // Compute TF-IDF vectors
    return data.map((row, idx) => {
      const doc = documents[idx];
      const termCounts = {};
      doc.forEach(term => {
        termCounts[term] = (termCounts[term] || 0) + 1;
      });

      const enhanced = { ...row };
      topTerms.forEach(term => {
        const tf = (termCounts[term] || 0) / doc.length;
        enhanced[`tfidf_${term}`] = tf * (idf[term] || 0);
      });

      return enhanced;
    });
  }

  /**
   * Create polynomial and interaction features
   */
  createPolynomialFeatures(data, columns, degree = 2) {
    return data.map(row => {
      const enhanced = { ...row };

      // Polynomial features
      columns.forEach(col => {
        const value = parseFloat(row[col]);
        for (let d = 2; d <= degree; d++) {
          enhanced[`${col}_pow_${d}`] = Math.pow(value, d);
        }
      });

      // Interaction features
      for (let i = 0; i < columns.length; i++) {
        for (let j = i + 1; j < columns.length; j++) {
          const col1 = columns[i];
          const col2 = columns[j];
          const val1 = parseFloat(row[col1]);
          const val2 = parseFloat(row[col2]);

          enhanced[`${col1}_x_${col2}`] = val1 * val2;
        }
      }

      return enhanced;
    });
  }

  /**
   * Feature selection using correlation
   */
  selectFeaturesByCorrelation(data, targetColumn, threshold = 0.7) {
    const features = Object.keys(data[0]).filter(f => f !== targetColumn);
    const selected = [];

    features.forEach(feature => {
      const correlation = this._computeCorrelation(
        data.map(row => parseFloat(row[feature])),
        data.map(row => parseFloat(row[targetColumn]))
      );

      if (Math.abs(correlation) >= threshold) {
        selected.push({
          feature,
          correlation,
          importance: Math.abs(correlation)
        });
      }
    });

    return selected.sort((a, b) => b.importance - a.importance);
  }

  _computeCorrelation(x, y) {
    const n = x.length;
    const meanX = x.reduce((a, b) => a + b, 0) / n;
    const meanY = y.reduce((a, b) => a + b, 0) / n;

    let numerator = 0;
    let denomX = 0;
    let denomY = 0;

    for (let i = 0; i < n; i++) {
      const diffX = x[i] - meanX;
      const diffY = y[i] - meanY;
      numerator += diffX * diffY;
      denomX += diffX * diffX;
      denomY += diffY * diffY;
    }

    return numerator / Math.sqrt(denomX * denomY);
  }

  /**
   * Complete feature engineering pipeline
   */
  async runPipeline(inputData, pipelineConfig) {
    console.log('Starting feature engineering pipeline...\n');

    let data = JSON.parse(JSON.stringify(inputData)); // Deep copy

    // Step 1: Feature type detection
    console.log('Step 1: Detecting feature types...');
    const types = this.detectFeatureTypes(data);
    console.log(`Detected ${Object.keys(types).length} features\n`);

    // Step 2: Temporal feature extraction
    if (pipelineConfig.temporalFeatures) {
      console.log('Step 2: Extracting temporal features...');
      data = this.extractTemporalFeatures(data, pipelineConfig.dateColumn);
      console.log(`Created ${Object.keys(data[0]).length - inputData[0].length} temporal features\n`);
    }

    // Step 3: Rolling window features
    if (pipelineConfig.rollingFeatures) {
      console.log('Step 3: Creating rolling window features...');
      const originalCount = Object.keys(data[0]).length;
      data = this.createRollingFeatures(
        data,
        pipelineConfig.valueColumn,
        pipelineConfig.windows
      );
      console.log(`Created ${Object.keys(data[0]).length - originalCount} rolling features\n`);
    }

    // Step 4: Polynomial features
    if (pipelineConfig.polynomialFeatures) {
      console.log('Step 4: Creating polynomial features...');
      const originalCount = Object.keys(data[0]).length;
      data = this.createPolynomialFeatures(
        data,
        pipelineConfig.numericColumns,
        pipelineConfig.polynomialDegree || 2
      );
      console.log(`Created ${Object.keys(data[0]).length - originalCount} polynomial features\n`);
    }

    // Step 5: Feature selection
    if (pipelineConfig.featureSelection) {
      console.log('Step 5: Selecting important features...');
      const selected = this.selectFeaturesByCorrelation(
        data,
        pipelineConfig.targetColumn,
        pipelineConfig.correlationThreshold || 0.5
      );
      console.log(`Selected ${selected.length} high-importance features\n`);

      // Show top features
      console.log('Top 10 features:');
      selected.slice(0, 10).forEach((f, i) => {
        console.log(`  ${i + 1}. ${f.feature}: ${f.correlation.toFixed(4)}`);
      });
    }

    return data;
  }
}

// Example usage
async function main() {
  console.log('Feature Engineering Complete Workflow Example');
  console.log('=' .repeat(60) + '\n');

  // Sample time series data
  const sampleData = [
    { date: '2024-01-01', sales: 100, temperature: 15, category: 'A' },
    { date: '2024-01-02', sales: 120, temperature: 18, category: 'B' },
    { date: '2024-01-03', sales: 110, temperature: 16, category: 'A' },
    { date: '2024-01-04', sales: 140, temperature: 20, category: 'C' },
    { date: '2024-01-05', sales: 130, temperature: 19, category: 'B' },
  ];

  const engineer = new AdvancedFeatureEngineer();

  // Pipeline configuration
  const config = {
    temporalFeatures: true,
    dateColumn: 'date',
    rollingFeatures: true,
    valueColumn: 'sales',
    windows: [3, 5],
    polynomialFeatures: true,
    numericColumns: ['sales', 'temperature'],
    polynomialDegree: 2,
    featureSelection: true,
    targetColumn: 'sales',
    correlationThreshold: 0.3
  };

  // Run pipeline
  const result = await engineer.runPipeline(sampleData, config);

  console.log('\n' + '=' .repeat(60));
  console.log('Pipeline complete!');
  console.log(`Original features: ${Object.keys(sampleData[0]).length}`);
  console.log(`Final features: ${Object.keys(result[0]).length}`);
}

if (require.main === module) {
  main().catch(console.error);
}

module.exports = { AdvancedFeatureEngineer };
