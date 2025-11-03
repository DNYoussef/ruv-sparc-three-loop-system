#!/usr/bin/env node

/**
 * Binary Search Debugging Script
 *
 * Automates binary search debugging by dividing the codebase/commit history
 * to isolate the exact location or commit that introduced a bug.
 *
 * Usage:
 *   node binary-search-debug.js --mode commits --start <commit> --end <commit> --test <test-command>
 *   node binary-search-debug.js --mode code --file <filepath> --test <test-command>
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class BinarySearchDebugger {
  constructor(options) {
    this.mode = options.mode || 'commits';
    this.startCommit = options.start;
    this.endCommit = options.end;
    this.testCommand = options.test;
    this.filePath = options.file;
    this.results = [];
  }

  /**
   * Execute test command and return result
   */
  runTest() {
    try {
      execSync(this.testCommand, { stdio: 'pipe', timeout: 30000 });
      return { passed: true, error: null };
    } catch (error) {
      return { passed: false, error: error.message };
    }
  }

  /**
   * Binary search through git commits
   */
  async searchCommits() {
    console.log('🔍 Starting binary search through commits...');

    try {
      // Get commit list
      const commitList = execSync(
        `git rev-list ${this.startCommit}..${this.endCommit}`,
        { encoding: 'utf-8' }
      ).trim().split('\n');

      console.log(`📊 Total commits to search: ${commitList.length}`);

      let left = 0;
      let right = commitList.length - 1;
      let buggyCommit = null;

      while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        const currentCommit = commitList[mid];

        console.log(`\n🧪 Testing commit ${mid + 1}/${commitList.length}: ${currentCommit.substring(0, 8)}`);

        // Checkout commit
        execSync(`git checkout ${currentCommit}`, { stdio: 'pipe' });

        // Run test
        const result = this.runTest();
        this.results.push({ commit: currentCommit, ...result });

        if (result.passed) {
          console.log('✅ Test passed - bug not present here');
          right = mid - 1;
        } else {
          console.log('❌ Test failed - bug present here');
          buggyCommit = currentCommit;
          left = mid + 1;
        }
      }

      // Return to original branch
      execSync('git checkout -', { stdio: 'pipe' });

      if (buggyCommit) {
        console.log(`\n🎯 Bug introduced in commit: ${buggyCommit}`);
        const commitInfo = execSync(`git show --stat ${buggyCommit}`, { encoding: 'utf-8' });
        console.log('\n📝 Commit details:\n', commitInfo);
        return buggyCommit;
      } else {
        console.log('\n⚠️  Could not isolate buggy commit');
        return null;
      }
    } catch (error) {
      console.error('❌ Error during binary search:', error.message);
      // Ensure we return to original branch
      try {
        execSync('git checkout -', { stdio: 'pipe' });
      } catch (e) {
        // Ignore checkout error
      }
      throw error;
    }
  }

  /**
   * Binary search through code sections (line-based)
   */
  async searchCode() {
    console.log('🔍 Starting binary search through code...');

    if (!fs.existsSync(this.filePath)) {
      throw new Error(`File not found: ${this.filePath}`);
    }

    const originalContent = fs.readFileSync(this.filePath, 'utf-8');
    const lines = originalContent.split('\n');

    console.log(`📊 Total lines to search: ${lines.length}`);

    // Binary search to find problematic line range
    let left = 0;
    let right = lines.length - 1;

    while (left < right) {
      const mid = Math.floor((left + right) / 2);

      console.log(`\n🧪 Testing lines 1-${mid + 1} (${Math.round((mid + 1) / lines.length * 100)}%)`);

      // Comment out second half
      const testContent = [
        ...lines.slice(0, mid + 1),
        ...lines.slice(mid + 1).map(line => `// BISECT: ${line}`)
      ].join('\n');

      fs.writeFileSync(this.filePath, testContent, 'utf-8');

      const result = this.runTest();

      if (result.passed) {
        console.log('✅ Test passed - bug in second half');
        left = mid + 1;
      } else {
        console.log('❌ Test failed - bug in first half');
        right = mid;
      }
    }

    // Restore original content
    fs.writeFileSync(this.filePath, originalContent, 'utf-8');

    console.log(`\n🎯 Bug likely on/near line: ${left + 1}`);
    console.log(`\n📝 Line content:\n${lines[left]}`);

    return { line: left + 1, content: lines[left] };
  }

  /**
   * Generate debug report
   */
  generateReport() {
    const reportPath = path.join(process.cwd(), 'binary-search-debug-report.json');
    const report = {
      timestamp: new Date().toISOString(),
      mode: this.mode,
      results: this.results,
      summary: {
        totalTests: this.results.length,
        passed: this.results.filter(r => r.passed).length,
        failed: this.results.filter(r => !r.passed).length
      }
    };

    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), 'utf-8');
    console.log(`\n📄 Report saved to: ${reportPath}`);
  }
}

// CLI argument parsing
function parseArgs() {
  const args = process.argv.slice(2);
  const options = {};

  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace('--', '');
    const value = args[i + 1];
    options[key] = value;
  }

  return options;
}

// Main execution
if (require.main === module) {
  const options = parseArgs();

  if (!options.test) {
    console.error('❌ Error: --test command is required');
    process.exit(1);
  }

  const debugger = new BinarySearchDebugger(options);

  (async () => {
    try {
      if (debugger.mode === 'commits') {
        if (!options.start || !options.end) {
          console.error('❌ Error: --start and --end commits required for commit mode');
          process.exit(1);
        }
        await debugger.searchCommits();
      } else if (debugger.mode === 'code') {
        if (!options.file) {
          console.error('❌ Error: --file path required for code mode');
          process.exit(1);
        }
        await debugger.searchCode();
      } else {
        console.error('❌ Error: Invalid mode. Use "commits" or "code"');
        process.exit(1);
      }

      debugger.generateReport();
      console.log('\n✅ Binary search debugging complete!');
    } catch (error) {
      console.error('\n❌ Fatal error:', error.message);
      process.exit(1);
    }
  })();
}

module.exports = BinarySearchDebugger;
