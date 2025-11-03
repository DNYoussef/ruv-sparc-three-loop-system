#!/usr/bin/env node
/**
 * Feature Engineering Pipeline Automation
 *
 * Automates feature extraction, transformation, selection, and validation
 * for machine learning pipelines with support for numerical, categorical,
 * temporal, and text features.
 *
 * Usage:
 *   node feature-engineer.js --config feature-config.json --input data.csv --output features.csv
 *   node feature-engineer.js --analyze data.csv --report feature-analysis.json
 *   node feature-engineer.js --optimize features.csv --target accuracy --method recursive
 */

const fs = require('fs');
const path = require('path');

class FeatureEngineer {
  constructor(config) {
    this.config = config;
    this.features = [];
    this.transformers = new Map();
    this.metadata = {
      created: new Date().toISOString(),
      version: '1.0.0',
      transforms: []
    };
  }

  /**
   * Load and parse input data
   */
  async loadData(filepath) {
    const content = fs.readFileSync(filepath, 'utf-8');
    const lines = content.split('\n').filter(l => l.trim());

    // Parse CSV
    const headers = lines[0].split(',');
    const data = lines.slice(1).map(line => {
      const values = line.split(',');
      const row = {};
      headers.forEach((header, i) => {
        row[header] = values[i];
      });
      return row;
    });

    return { headers, data };
  }

  /**
   * Analyze data and suggest feature engineering strategies
   */
  analyzeData(data) {
    const analysis = {
      rowCount: data.length,
      features: {},
      suggestions: []
    };

    // Get unique column names
    const columns = Object.keys(data[0]);

    columns.forEach(col => {
      const values = data.map(row => row[col]);
      const uniqueValues = new Set(values);
      const nullCount = values.filter(v => !v || v === 'null' || v === 'undefined').length;

      // Detect feature type
      let featureType = 'unknown';
      let isNumeric = values.every(v => !isNaN(parseFloat(v)));
      let isDate = values.some(v => !isNaN(Date.parse(v)));

      if (isNumeric) {
        featureType = 'numerical';
      } else if (uniqueValues.size < 20) {
        featureType = 'categorical';
      } else if (isDate) {
        featureType = 'temporal';
      } else {
        featureType = 'text';
      }

      analysis.features[col] = {
        type: featureType,
        uniqueCount: uniqueValues.size,
        nullCount: nullCount,
        nullPercentage: (nullCount / values.length * 100).toFixed(2) + '%'
      };

      // Generate suggestions
      if (featureType === 'numerical') {
        analysis.suggestions.push({
          feature: col,
          transform: 'standardize',
          reason: 'Numerical features benefit from standardization'
        });
      }

      if (featureType === 'categorical' && uniqueValues.size > 5) {
        analysis.suggestions.push({
          feature: col,
          transform: 'one_hot_encode',
          reason: 'High-cardinality categorical feature'
        });
      }

      if (featureType === 'temporal') {
        analysis.suggestions.push({
          feature: col,
          transform: 'extract_temporal_features',
          reason: 'Extract year, month, day, hour components'
        });
      }

      if (nullCount > 0) {
        analysis.suggestions.push({
          feature: col,
          transform: 'impute',
          reason: `${nullCount} missing values detected`
        });
      }
    });

    return analysis;
  }

  /**
   * Apply feature transformations
   */
  transform(data) {
    let transformedData = JSON.parse(JSON.stringify(data)); // Deep copy

    this.config.transformations.forEach(transform => {
      const { feature, method, params } = transform;

      switch (method) {
        case 'standardize':
          transformedData = this.standardize(transformedData, feature);
          break;
        case 'normalize':
          transformedData = this.normalize(transformedData, feature);
          break;
        case 'one_hot_encode':
          transformedData = this.oneHotEncode(transformedData, feature);
          break;
        case 'label_encode':
          transformedData = this.labelEncode(transformedData, feature);
          break;
        case 'bin':
          transformedData = this.bin(transformedData, feature, params.bins);
          break;
        case 'log_transform':
          transformedData = this.logTransform(transformedData, feature);
          break;
        case 'polynomial':
          transformedData = this.polynomial(transformedData, feature, params.degree);
          break;
        case 'interaction':
          transformedData = this.interaction(transformedData, params.features);
          break;
        default:
          console.warn(`Unknown transform: ${method}`);
      }

      this.metadata.transforms.push({ feature, method, params: params || {} });
    });

    return transformedData;
  }

  /**
   * Standardize numerical feature (z-score normalization)
   */
  standardize(data, feature) {
    const values = data.map(row => parseFloat(row[feature]));
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);

    data.forEach((row, i) => {
      row[`${feature}_standardized`] = ((values[i] - mean) / std).toFixed(4);
    });

    return data;
  }

  /**
   * Normalize numerical feature (min-max scaling)
   */
  normalize(data, feature) {
    const values = data.map(row => parseFloat(row[feature]));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;

    data.forEach((row, i) => {
      row[`${feature}_normalized`] = ((values[i] - min) / range).toFixed(4);
    });

    return data;
  }

  /**
   * One-hot encode categorical feature
   */
  oneHotEncode(data, feature) {
    const uniqueValues = [...new Set(data.map(row => row[feature]))];

    data.forEach(row => {
      uniqueValues.forEach(value => {
        row[`${feature}_${value}`] = row[feature] === value ? 1 : 0;
      });
    });

    return data;
  }

  /**
   * Label encode categorical feature
   */
  labelEncode(data, feature) {
    const uniqueValues = [...new Set(data.map(row => row[feature]))];
    const encoding = {};
    uniqueValues.forEach((value, i) => {
      encoding[value] = i;
    });

    data.forEach(row => {
      row[`${feature}_encoded`] = encoding[row[feature]];
    });

    return data;
  }

  /**
   * Bin numerical feature into categories
   */
  bin(data, feature, numBins) {
    const values = data.map(row => parseFloat(row[feature]));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const binSize = (max - min) / numBins;

    data.forEach(row => {
      const value = parseFloat(row[feature]);
      const binIndex = Math.min(Math.floor((value - min) / binSize), numBins - 1);
      row[`${feature}_bin`] = binIndex;
    });

    return data;
  }

  /**
   * Log transform numerical feature
   */
  logTransform(data, feature) {
    data.forEach(row => {
      const value = parseFloat(row[feature]);
      row[`${feature}_log`] = Math.log(value + 1).toFixed(4);
    });

    return data;
  }

  /**
   * Create polynomial features
   */
  polynomial(data, feature, degree) {
    for (let d = 2; d <= degree; d++) {
      data.forEach(row => {
        const value = parseFloat(row[feature]);
        row[`${feature}_pow${d}`] = Math.pow(value, d).toFixed(4);
      });
    }

    return data;
  }

  /**
   * Create interaction features
   */
  interaction(data, features) {
    data.forEach(row => {
      const product = features.reduce((prod, f) => prod * parseFloat(row[f]), 1);
      row[`interaction_${features.join('_')}`] = product.toFixed(4);
    });

    return data;
  }

  /**
   * Select top features based on importance
   */
  selectFeatures(data, targetFeature, method = 'correlation', topK = 10) {
    // Simplified feature selection - in production use scikit-learn
    const features = Object.keys(data[0]).filter(f => f !== targetFeature);

    // Calculate correlation with target (for numerical targets)
    const correlations = features.map(feature => {
      const correlation = this.calculateCorrelation(data, feature, targetFeature);
      return { feature, correlation: Math.abs(correlation) };
    });

    // Sort by correlation and take top K
    correlations.sort((a, b) => b.correlation - a.correlation);
    const selectedFeatures = correlations.slice(0, topK).map(item => item.feature);

    return {
      selected: selectedFeatures,
      scores: correlations
    };
  }

  /**
   * Calculate Pearson correlation coefficient
   */
  calculateCorrelation(data, feature1, feature2) {
    const values1 = data.map(row => parseFloat(row[feature1]));
    const values2 = data.map(row => parseFloat(row[feature2]));

    const mean1 = values1.reduce((sum, v) => sum + v, 0) / values1.length;
    const mean2 = values2.reduce((sum, v) => sum + v, 0) / values2.length;

    let numerator = 0;
    let denominator1 = 0;
    let denominator2 = 0;

    for (let i = 0; i < values1.length; i++) {
      const diff1 = values1[i] - mean1;
      const diff2 = values2[i] - mean2;
      numerator += diff1 * diff2;
      denominator1 += diff1 * diff1;
      denominator2 += diff2 * diff2;
    }

    return numerator / Math.sqrt(denominator1 * denominator2);
  }

  /**
   * Save transformed data to CSV
   */
  saveData(data, filepath) {
    const headers = Object.keys(data[0]);
    const csv = [headers.join(',')];

    data.forEach(row => {
      const values = headers.map(h => row[h]);
      csv.push(values.join(','));
    });

    fs.writeFileSync(filepath, csv.join('\n'));
  }

  /**
   * Save metadata and transformation log
   */
  saveMetadata(filepath) {
    fs.writeFileSync(filepath, JSON.stringify(this.metadata, null, 2));
  }
}

// CLI execution
async function main() {
  const args = process.argv.slice(2);
  const options = {};

  for (let i = 0; i < args.length; i += 2) {
    options[args[i].replace('--', '')] = args[i + 1];
  }

  if (options.analyze) {
    // Analyze data
    const engineer = new FeatureEngineer({ transformations: [] });
    const { data } = await engineer.loadData(options.analyze);
    const analysis = engineer.analyzeData(data);

    if (options.report) {
      fs.writeFileSync(options.report, JSON.stringify(analysis, null, 2));
      console.log(`Analysis saved to ${options.report}`);
    } else {
      console.log(JSON.stringify(analysis, null, 2));
    }
  } else if (options.config && options.input && options.output) {
    // Transform data
    const config = JSON.parse(fs.readFileSync(options.config, 'utf-8'));
    const engineer = new FeatureEngineer(config);
    const { data } = await engineer.loadData(options.input);

    const transformed = engineer.transform(data);
    engineer.saveData(transformed, options.output);
    engineer.saveMetadata(options.output.replace('.csv', '_metadata.json'));

    console.log(`Transformed data saved to ${options.output}`);
  } else {
    console.log('Usage:');
    console.log('  Analyze: node feature-engineer.js --analyze data.csv --report analysis.json');
    console.log('  Transform: node feature-engineer.js --config config.json --input data.csv --output features.csv');
  }
}

if (require.main === module) {
  main().catch(console.error);
}

module.exports = { FeatureEngineer };
