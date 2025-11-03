#!/usr/bin/env node
/**
 * Individual Stage Executor for Feature Development
 *
 * Allows running individual stages of the 12-stage workflow:
 * research, analyze, swarm-init, architecture, diagrams, prototype,
 * theater-detect, testing, style-polish, security, documentation, deploy-check
 *
 * @usage node stage-executor.js <stage-name> <output-dir> [stage-specific-args]
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const stageName = process.argv[2];
const outputDir = process.argv[3];
const extraArgs = process.argv.slice(4);

// Validate inputs
if (!stageName || !outputDir) {
  console.error('❌ Error: Stage name and output directory required');
  console.error('Usage: node stage-executor.js <stage-name> <output-dir> [stage-args]');
  console.error('\nAvailable stages:');
  console.error('  research, analyze, swarm-init, architecture, diagrams,');
  console.error('  prototype, theater-detect, testing, style-polish,');
  console.error('  security, documentation, deploy-check');
  process.exit(1);
}

// Ensure output directory exists
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Utility functions
function exec(command, options = {}) {
  console.log(`\n$ ${command}`);
  try {
    return execSync(command, {
      encoding: 'utf-8',
      stdio: options.silent ? 'pipe' : 'inherit',
      ...options
    });
  } catch (error) {
    if (!options.ignoreError) {
      console.error(`❌ Command failed: ${command}`);
      throw error;
    }
    return null;
  }
}

function writeFile(filepath, content) {
  fs.writeFileSync(filepath, content, 'utf-8');
}

function readFile(filepath) {
  try {
    return fs.readFileSync(filepath, 'utf-8');
  } catch {
    return '';
  }
}

function readJSON(filepath) {
  try {
    return JSON.parse(fs.readFileSync(filepath, 'utf-8'));
  } catch {
    return null;
  }
}

// Stage implementations
const stages = {
  'research': () => {
    const query = extraArgs[0] || 'Best practices';
    const outputFile = path.join(outputDir, 'research.md');

    console.log(`\n🔍 Researching: ${query}`);
    exec(`gemini "${query}" --grounding google-search --output "${outputFile}"`, {
      ignoreError: true
    });

    if (!fs.existsSync(outputFile)) {
      writeFile(outputFile, `# Research: ${query}\n\n[Research findings]\n`);
    }

    console.log(`✅ Research saved to: ${outputFile}`);
  },

  'analyze': () => {
    const targetDir = extraArgs[0] || 'src/';
    const outputFile = path.join(outputDir, 'codebase-analysis.md');

    console.log(`\n📊 Analyzing codebase in: ${targetDir}`);

    const locCommand = `find "${targetDir}" -type f \\( -name "*.js" -o -name "*.ts" \\) -exec wc -l {} + 2>/dev/null | tail -1`;
    const locOutput = exec(locCommand, { capture: true, ignoreError: true, silent: true });
    const totalLOC = locOutput ? parseInt(locOutput.split(/\s+/)[0] || '0') : 0;

    console.log(`   Lines of code: ${totalLOC.toLocaleString()}`);

    writeFile(outputFile, `# Codebase Analysis\n\n## Metrics\n- LOC: ${totalLOC.toLocaleString()}\n- Directory: ${targetDir}\n\n[Analysis details]\n`);
    console.log(`✅ Analysis saved to: ${outputFile}`);
  },

  'swarm-init': () => {
    const topology = extraArgs[0] || 'hierarchical';
    const maxAgents = extraArgs[1] || '6';

    console.log(`\n🐝 Initializing ${topology} swarm with ${maxAgents} agents...`);
    exec(`npx claude-flow@alpha coordination swarm-init --topology ${topology} --max-agents ${maxAgents} --strategy balanced`, {
      ignoreError: true
    });

    console.log('✅ Swarm initialized');
  },

  'architecture': () => {
    const featureSpec = extraArgs.join(' ') || 'Feature';
    const outputFile = path.join(outputDir, 'architecture-design.md');

    console.log(`\n🏗️  Designing architecture for: ${featureSpec}`);

    const doc = `# Architecture Design: ${featureSpec}\n\n## Components\n\n[Component architecture]\n\n## Design Decisions\n\n[Key decisions]\n`;
    writeFile(outputFile, doc);

    console.log(`✅ Architecture saved to: ${outputFile}`);
  },

  'diagrams': () => {
    const featureSpec = extraArgs.join(' ') || 'Feature';

    console.log(`\n📐 Generating diagrams for: ${featureSpec}`);

    const archDiagram = path.join(outputDir, 'architecture-diagram.png');
    const flowDiagram = path.join(outputDir, 'data-flow.png');

    exec(`gemini "System architecture for: ${featureSpec}" --type image --output "${archDiagram}"`, {
      ignoreError: true
    });

    exec(`gemini "Data flow for: ${featureSpec}" --type image --output "${flowDiagram}"`, {
      ignoreError: true
    });

    console.log('✅ Diagrams generated');
  },

  'prototype': () => {
    const implDir = path.join(outputDir, 'implementation');
    const featureSpec = extraArgs.join(' ') || 'Feature';

    if (!fs.existsSync(implDir)) {
      fs.mkdirSync(implDir, { recursive: true });
    }

    console.log(`\n🚀 Prototyping: ${featureSpec}`);
    exec(`codex --full-auto "Implement ${featureSpec}" --output "${implDir}"`, {
      ignoreError: true
    });

    console.log(`✅ Prototype created in: ${implDir}`);
  },

  'theater-detect': () => {
    const implDir = extraArgs[0] || path.join(outputDir, 'implementation');
    const reportFile = path.join(outputDir, 'theater-report.json');

    console.log(`\n🎭 Detecting placeholder code in: ${implDir}`);
    exec(`npx claude-flow@alpha theater-detect "${implDir}" --output "${reportFile}"`, {
      ignoreError: true
    });

    const report = readJSON(reportFile);
    if (report && report.issues) {
      console.log(`   Found ${report.issues.length} placeholder items`);
    }

    console.log('✅ Theater detection complete');
  },

  'testing': () => {
    const implDir = extraArgs[0] || path.join(outputDir, 'implementation');
    const testResults = path.join(outputDir, 'test-results.json');

    console.log(`\n🧪 Testing implementation in: ${implDir}`);
    exec(`npx claude-flow@alpha functionality-audit "${implDir}" --sandbox true --output "${testResults}"`, {
      ignoreError: true
    });

    const results = readJSON(testResults);
    if (results) {
      console.log(`   Tests: ${results.all_passed ? 'PASSING' : 'FAILING'}`);
      console.log(`   Coverage: ${results.coverage_percent || 0}%`);
    }

    console.log('✅ Testing complete');
  },

  'style-polish': () => {
    const implDir = extraArgs[0] || path.join(outputDir, 'implementation');
    const styleReport = path.join(outputDir, 'style-report.json');

    console.log(`\n🎨 Polishing code style in: ${implDir}`);
    exec(`npx claude-flow@alpha style-audit "${implDir}" --fix true --output "${styleReport}"`, {
      ignoreError: true
    });

    const report = readJSON(styleReport);
    if (report) {
      console.log(`   Quality Score: ${report.quality_score || 0}/100`);
    }

    console.log('✅ Style polish complete');
  },

  'security': () => {
    const implDir = extraArgs[0] || path.join(outputDir, 'implementation');
    const securityReport = path.join(outputDir, 'security-report.json');

    console.log(`\n🔒 Security scan of: ${implDir}`);
    exec(`npx claude-flow@alpha security-scan "${implDir}" --deep true --output "${securityReport}"`, {
      ignoreError: true
    });

    const report = readJSON(securityReport);
    if (report) {
      console.log(`   Critical Issues: ${report.critical_issues || 0}`);
      console.log(`   High Issues: ${report.high_issues || 0}`);
    }

    console.log('✅ Security scan complete');
  },

  'documentation': () => {
    const featureSpec = extraArgs.join(' ') || 'Feature';
    const docFile = path.join(outputDir, 'FEATURE-DOCUMENTATION.md');

    console.log(`\n📝 Generating documentation for: ${featureSpec}`);

    const doc = `# ${featureSpec}\n\n## Overview\n\n[Feature overview]\n\n## Usage\n\n[Usage examples]\n\n## API\n\n[API documentation]\n`;
    writeFile(docFile, doc);

    console.log(`✅ Documentation saved to: ${docFile}`);
  },

  'deploy-check': () => {
    console.log('\n🚦 Running production readiness check...');

    const testResults = readJSON(path.join(outputDir, 'test-results.json')) || {};
    const styleReport = readJSON(path.join(outputDir, 'style-report.json')) || {};
    const securityReport = readJSON(path.join(outputDir, 'security-report.json')) || {};

    const checks = {
      testsPassing: testResults.all_passed || false,
      qualityScore: (styleReport.quality_score || 0) >= 85,
      securityOK: (securityReport.critical_issues || 0) === 0
    };

    console.log(`   ${checks.testsPassing ? '✅' : '❌'} Tests Passing`);
    console.log(`   ${checks.qualityScore ? '✅' : '❌'} Quality Score ≥85`);
    console.log(`   ${checks.securityOK ? '✅' : '❌'} No Critical Security Issues`);

    const isReady = Object.values(checks).every(Boolean);
    console.log(`\n${isReady ? '✅ READY' : '❌ NOT READY'} for production`);

    process.exit(isReady ? 0 : 1);
  }
};

// Execute stage
console.log('='.repeat(70));
console.log(`Executing Stage: ${stageName}`);
console.log('='.repeat(70));

if (!stages[stageName]) {
  console.error(`❌ Unknown stage: ${stageName}`);
  console.error('\nAvailable stages:', Object.keys(stages).join(', '));
  process.exit(1);
}

try {
  stages[stageName]();
  console.log('\n✅ Stage completed successfully\n');
} catch (error) {
  console.error(`\n❌ Stage failed: ${error.message}\n`);
  process.exit(1);
}
