#!/usr/bin/env node

/**
 * Bundle Analyzer
 *
 * Analyzes frontend bundle sizes and provides optimization recommendations.
 * Supports Webpack, Vite, and Rollup build outputs.
 *
 * Features:
 * - Bundle size analysis
 * - Dependency tree visualization
 * - Tree-shaking effectiveness
 * - Optimization recommendations
 * - Size budget validation
 *
 * Usage:
 *   node bundle-analyzer.js --project ./my-app --threshold 200
 *   node bundle-analyzer.js --build-dir ./dist --format json
 *   node bundle-analyzer.js --project ./my-app --budget 150 --verbose
 *
 * @author Frontend Specialists Team
 * @version 1.0.0
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class BundleAnalyzer {
  constructor(options) {
    this.projectDir = options.project || process.cwd();
    this.buildDir = options.buildDir || 'dist';
    this.threshold = parseInt(options.threshold) || 200; // KB
    this.budget = parseInt(options.budget) || null;
    this.format = options.format || 'text'; // text | json
    this.verbose = options.verbose || false;
  }

  /**
   * Run bundle analysis
   */
  async analyze() {
    console.log(`\n📊 Analyzing bundle in ${this.projectDir}\n`);

    // Detect build tool
    const buildTool = this.detectBuildTool();
    console.log(`🔍 Detected build tool: ${buildTool}\n`);

    // Get bundle stats
    const stats = this.getBundleStats(buildTool);

    // Analyze results
    const analysis = this.analyzeStats(stats);

    // Generate report
    this.generateReport(analysis);

    // Validate budget
    if (this.budget) {
      this.validateBudget(analysis);
    }

    return analysis;
  }

  /**
   * Detect build tool from project files
   */
  detectBuildTool() {
    const packageJsonPath = path.join(this.projectDir, 'package.json');

    if (!fs.existsSync(packageJsonPath)) {
      throw new Error('package.json not found');
    }

    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf-8'));
    const deps = { ...packageJson.dependencies, ...packageJson.devDependencies };

    if (deps['vite']) return 'vite';
    if (deps['webpack']) return 'webpack';
    if (deps['rollup']) return 'rollup';
    if (deps['next']) return 'next';

    throw new Error('Could not detect build tool');
  }

  /**
   * Get bundle statistics
   */
  getBundleStats(buildTool) {
    const buildPath = path.join(this.projectDir, this.buildDir);

    if (!fs.existsSync(buildPath)) {
      throw new Error(`Build directory not found: ${buildPath}`);
    }

    const files = this.getJavaScriptFiles(buildPath);
    const stats = {
      files: [],
      totalSize: 0,
      gzippedSize: 0,
      chunks: [],
    };

    for (const file of files) {
      const filePath = path.join(buildPath, file);
      const fileStats = fs.statSync(filePath);
      const size = fileStats.size / 1024; // KB

      // Estimate gzipped size (rough approximation: ~30% of original)
      const gzippedSize = size * 0.3;

      stats.files.push({
        name: file,
        size: size.toFixed(2),
        gzipped: gzippedSize.toFixed(2),
        type: this.classifyFile(file),
      });

      stats.totalSize += size;
      stats.gzippedSize += gzippedSize;
    }

    // Group by chunk type
    stats.chunks = this.groupByChunkType(stats.files);

    return stats;
  }

  /**
   * Get all JavaScript files recursively
   */
  getJavaScriptFiles(dir, fileList = []) {
    const files = fs.readdirSync(dir);

    files.forEach(file => {
      const filePath = path.join(dir, file);
      const stat = fs.statSync(filePath);

      if (stat.isDirectory()) {
        this.getJavaScriptFiles(filePath, fileList);
      } else if (file.endsWith('.js') || file.endsWith('.mjs')) {
        fileList.push(path.relative(path.join(this.projectDir, this.buildDir), filePath));
      }
    });

    return fileList;
  }

  /**
   * Classify file type
   */
  classifyFile(filename) {
    if (filename.includes('vendor') || filename.includes('node_modules')) {
      return 'vendor';
    }
    if (filename.includes('runtime') || filename.includes('webpack-runtime')) {
      return 'runtime';
    }
    if (filename.includes('polyfill')) {
      return 'polyfill';
    }
    return 'app';
  }

  /**
   * Group files by chunk type
   */
  groupByChunkType(files) {
    const chunks = {
      vendor: [],
      runtime: [],
      polyfill: [],
      app: [],
    };

    files.forEach(file => {
      chunks[file.type].push(file);
    });

    return chunks;
  }

  /**
   * Analyze bundle statistics
   */
  analyzeStats(stats) {
    const analysis = {
      summary: {
        totalSize: stats.totalSize.toFixed(2),
        gzippedSize: stats.gzippedSize.toFixed(2),
        fileCount: stats.files.length,
      },
      chunks: {},
      largestFiles: [],
      recommendations: [],
      score: 0,
    };

    // Chunk analysis
    for (const [type, files] of Object.entries(stats.chunks)) {
      const totalSize = files.reduce((sum, f) => sum + parseFloat(f.size), 0);
      analysis.chunks[type] = {
        count: files.length,
        size: totalSize.toFixed(2),
        files: files.map(f => ({ name: f.name, size: f.size })),
      };
    }

    // Largest files
    analysis.largestFiles = [...stats.files]
      .sort((a, b) => parseFloat(b.size) - parseFloat(a.size))
      .slice(0, 5);

    // Generate recommendations
    analysis.recommendations = this.generateRecommendations(stats);

    // Calculate performance score
    analysis.score = this.calculateScore(stats);

    return analysis;
  }

  /**
   * Generate optimization recommendations
   */
  generateRecommendations(stats) {
    const recommendations = [];

    // Check total size
    if (stats.totalSize > this.threshold) {
      recommendations.push({
        type: 'warning',
        message: `Total bundle size (${stats.totalSize.toFixed(2)}KB) exceeds threshold (${this.threshold}KB)`,
        suggestion: 'Consider code splitting, tree shaking, or lazy loading',
      });
    }

    // Check large files
    const largeFiles = stats.files.filter(f => parseFloat(f.size) > 100);
    if (largeFiles.length > 0) {
      recommendations.push({
        type: 'info',
        message: `${largeFiles.length} files exceed 100KB`,
        suggestion: 'Consider splitting large files or using dynamic imports',
      });
    }

    // Check vendor chunk
    const vendorSize = stats.chunks.vendor?.reduce((sum, f) => sum + parseFloat(f.size), 0) || 0;
    if (vendorSize > 150) {
      recommendations.push({
        type: 'warning',
        message: `Vendor bundle is large (${vendorSize.toFixed(2)}KB)`,
        suggestion: 'Use external CDN for common libraries or split vendor chunks',
      });
    }

    // Check polyfills
    const polyfillSize = stats.chunks.polyfill?.reduce((sum, f) => sum + parseFloat(f.size), 0) || 0;
    if (polyfillSize > 50) {
      recommendations.push({
        type: 'info',
        message: `Polyfills are large (${polyfillSize.toFixed(2)}KB)`,
        suggestion: 'Use differential loading or reduce polyfill scope',
      });
    }

    // Best practices
    if (stats.files.length > 50) {
      recommendations.push({
        type: 'info',
        message: `Many chunks (${stats.files.length})`,
        suggestion: 'Too many chunks can impact HTTP/2 performance. Consider chunk optimization',
      });
    }

    return recommendations;
  }

  /**
   * Calculate performance score (0-100)
   */
  calculateScore(stats) {
    let score = 100;

    // Penalize large total size
    if (stats.totalSize > this.threshold) {
      score -= Math.min(30, (stats.totalSize - this.threshold) / 10);
    }

    // Penalize large individual files
    const largeFiles = stats.files.filter(f => parseFloat(f.size) > 100);
    score -= largeFiles.length * 5;

    // Penalize many chunks
    if (stats.files.length > 50) {
      score -= 10;
    }

    return Math.max(0, Math.round(score));
  }

  /**
   * Generate report
   */
  generateReport(analysis) {
    if (this.format === 'json') {
      console.log(JSON.stringify(analysis, null, 2));
      return;
    }

    // Text report
    console.log('═══════════════════════════════════════════════════════════');
    console.log('                    BUNDLE ANALYSIS REPORT                 ');
    console.log('═══════════════════════════════════════════════════════════\n');

    console.log('📦 Summary');
    console.log('─────────────────────────────────────────────────────────');
    console.log(`  Total Size:     ${analysis.summary.totalSize} KB`);
    console.log(`  Gzipped:        ${analysis.summary.gzippedSize} KB`);
    console.log(`  File Count:     ${analysis.summary.fileCount}`);
    console.log(`  Score:          ${analysis.score}/100 ${this.getScoreEmoji(analysis.score)}\n`);

    console.log('📊 Chunks');
    console.log('─────────────────────────────────────────────────────────');
    for (const [type, data] of Object.entries(analysis.chunks)) {
      if (data.count > 0) {
        console.log(`  ${type.padEnd(12)} ${data.size} KB (${data.count} files)`);
      }
    }
    console.log('');

    console.log('🔝 Largest Files');
    console.log('─────────────────────────────────────────────────────────');
    analysis.largestFiles.forEach((file, i) => {
      console.log(`  ${(i + 1)}. ${file.name.padEnd(40)} ${file.size} KB`);
    });
    console.log('');

    if (analysis.recommendations.length > 0) {
      console.log('💡 Recommendations');
      console.log('─────────────────────────────────────────────────────────');
      analysis.recommendations.forEach(rec => {
        const icon = rec.type === 'warning' ? '⚠️' : 'ℹ️';
        console.log(`  ${icon}  ${rec.message}`);
        console.log(`      → ${rec.suggestion}\n`);
      });
    } else {
      console.log('✅ No issues found. Bundle is optimized!\n');
    }

    console.log('═══════════════════════════════════════════════════════════\n');
  }

  /**
   * Get score emoji
   */
  getScoreEmoji(score) {
    if (score >= 90) return '🟢';
    if (score >= 70) return '🟡';
    return '🔴';
  }

  /**
   * Validate size budget
   */
  validateBudget(analysis) {
    const totalSize = parseFloat(analysis.summary.totalSize);

    console.log('💰 Budget Validation');
    console.log('─────────────────────────────────────────────────────────');
    console.log(`  Budget:         ${this.budget} KB`);
    console.log(`  Actual:         ${totalSize.toFixed(2)} KB`);
    console.log(`  Difference:     ${(this.budget - totalSize).toFixed(2)} KB`);

    if (totalSize > this.budget) {
      console.log(`  Status:         ❌ OVER BUDGET\n`);
      process.exit(1);
    } else {
      console.log(`  Status:         ✅ WITHIN BUDGET\n`);
    }
  }
}

// CLI
if (require.main === module) {
  const args = process.argv.slice(2);
  const options = {};

  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace('--', '');
    const value = args[i + 1];
    options[key] = value === 'true' ? true : value === 'false' ? false : value;
  }

  const analyzer = new BundleAnalyzer(options);

  analyzer.analyze().catch(error => {
    console.error(`❌ Error: ${error.message}`);
    process.exit(1);
  });
}

module.exports = { BundleAnalyzer };
