#!/usr/bin/env node
/**
 * Test suite for network scanner
 * Tests network-scanner.js functionality
 */

const assert = require('assert');
const fs = require('fs').promises;
const path = require('path');
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);

// Test configuration
const TEST_DIR = __dirname;
const RESOURCES_DIR = path.join(path.dirname(TEST_DIR), 'resources');
const SCRIPT_DIR = path.join(RESOURCES_DIR, 'scripts');
const TEMP_DIR = `/tmp/network-security-test-${process.pid}`;

// Test counters
let testsRun = 0;
let testsPassed = 0;
let testsFailed = 0;

// ANSI colors
const colors = {
  green: '\x1b[32m',
  red: '\x1b[31m',
  reset: '\x1b[0m',
};

/**
 * Setup test environment
 */
async function setup() {
  console.log('Setting up test environment...');

  // Create temp directories
  await fs.mkdir(path.join(TEMP_DIR, 'config'), { recursive: true });

  // Create test configuration file
  const configContent = `# Test trusted domains
*.npmjs.org
registry.npmjs.org
*.github.com
api.github.com
`;

  await fs.writeFile(
    path.join(TEMP_DIR, 'config', 'trusted-domains.conf'),
    configContent
  );

  // Set environment variables
  process.env.CONFIG_FILE = path.join(TEMP_DIR, 'config', 'trusted-domains.conf');
  process.env.TIMEOUT = '5000';
  process.env.PARALLEL = '2';
}

/**
 * Cleanup test environment
 */
async function cleanup() {
  console.log('Cleaning up test environment...');

  try {
    await fs.rm(TEMP_DIR, { recursive: true, force: true });
  } catch (error) {
    // Ignore cleanup errors
  }
}

/**
 * Test helper: assert equals
 */
function assertEquals(expected, actual, message) {
  testsRun++;

  try {
    assert.strictEqual(actual, expected);
    console.log(`${colors.green}✓${colors.reset} ${message}`);
    testsPassed++;
  } catch (error) {
    console.log(`${colors.red}✗${colors.reset} ${message}`);
    console.log(`  Expected: ${expected}`);
    console.log(`  Actual: ${actual}`);
    testsFailed++;
  }
}

/**
 * Test helper: assert contains
 */
function assertContains(haystack, needle, message) {
  testsRun++;

  if (haystack.includes(needle)) {
    console.log(`${colors.green}✓${colors.reset} ${message}`);
    testsPassed++;
  } else {
    console.log(`${colors.red}✗${colors.reset} ${message}`);
    console.log(`  Haystack: ${haystack}`);
    console.log(`  Needle: ${needle}`);
    testsFailed++;
  }
}

/**
 * Test helper: assert file exists
 */
async function assertFileExists(filePath, message) {
  testsRun++;

  try {
    await fs.access(filePath);
    console.log(`${colors.green}✓${colors.reset} ${message}`);
    testsPassed++;
  } catch (error) {
    console.log(`${colors.red}✗${colors.reset} ${message}`);
    testsFailed++;
  }
}

/**
 * Test: Script file exists
 */
async function testScriptExists() {
  console.log('\nTest: Script file exists');

  const scriptPath = path.join(SCRIPT_DIR, 'network-scanner.js');
  await assertFileExists(scriptPath, 'network-scanner.js should exist');
}

/**
 * Test: Configuration file parsing
 */
async function testConfigFileParsing() {
  console.log('\nTest: Configuration file parsing');

  try {
    // Import NetworkScanner
    const NetworkScanner = require(path.join(SCRIPT_DIR, 'network-scanner.js'));

    const scanner = new NetworkScanner({
      configFile: process.env.CONFIG_FILE,
      verbose: false,
    });

    const domains = await scanner.parseTrustedDomains();

    assertEquals(4, domains.length, 'Should parse 4 domains');
    assert(domains.includes('*.npmjs.org'), 'Should contain *.npmjs.org');
    assert(domains.includes('*.github.com'), 'Should contain *.github.com');

    testsPassed++;
  } catch (error) {
    console.log(`${colors.red}✗${colors.reset} Configuration parsing failed`);
    console.log(`  Error: ${error.message}`);
    testsFailed++;
  }

  testsRun++;
}

/**
 * Test: Domain resolution
 */
async function testDomainResolution() {
  console.log('\nTest: Domain resolution');

  try {
    const NetworkScanner = require(path.join(SCRIPT_DIR, 'network-scanner.js'));

    const scanner = new NetworkScanner({
      configFile: process.env.CONFIG_FILE,
    });

    // Test resolving github.com
    const addresses = await scanner.resolveDomain('github.com');

    assert(addresses.length > 0, 'Should resolve github.com to at least one IP');
    console.log(`${colors.green}✓${colors.reset} Domain resolution works`);
    testsPassed++;
  } catch (error) {
    console.log(`${colors.red}✗${colors.reset} Domain resolution failed`);
    console.log(`  Error: ${error.message}`);
    testsFailed++;
  }

  testsRun++;
}

/**
 * Test: HTTP connectivity test (mock)
 */
async function testHttpConnectivity() {
  console.log('\nTest: HTTP connectivity test');

  try {
    const NetworkScanner = require(path.join(SCRIPT_DIR, 'network-scanner.js'));

    const scanner = new NetworkScanner({
      configFile: process.env.CONFIG_FILE,
      timeout: 5000,
    });

    // Test HTTP connectivity to github.com
    const result = await scanner.testHttpConnectivity('github.com', 'https');

    assert(typeof result.success === 'boolean', 'Should return success boolean');
    assert(result.url.includes('github.com'), 'Should test correct URL');

    console.log(`${colors.green}✓${colors.reset} HTTP connectivity test works`);
    testsPassed++;
  } catch (error) {
    console.log(`${colors.red}✗${colors.reset} HTTP connectivity test failed`);
    console.log(`  Error: ${error.message}`);
    testsFailed++;
  }

  testsRun++;
}

/**
 * Test: Results export
 */
async function testResultsExport() {
  console.log('\nTest: Results export');

  try {
    const NetworkScanner = require(path.join(SCRIPT_DIR, 'network-scanner.js'));

    const scanner = new NetworkScanner({
      configFile: process.env.CONFIG_FILE,
    });

    // Mock results
    scanner.results = {
      trusted: [],
      blocked: [],
      errors: [],
      summary: {
        total_tested: 0,
        accessible: 0,
        blocked: 0,
      },
    };

    const outputPath = path.join(TEMP_DIR, 'results.json');
    await scanner.exportResults(outputPath);

    await assertFileExists(outputPath, 'Results file should be created');

    // Verify JSON content
    const content = await fs.readFile(outputPath, 'utf-8');
    const results = JSON.parse(content);

    assert(results.summary, 'Results should contain summary');
    console.log(`${colors.green}✓${colors.reset} Results export works`);
    testsPassed++;
  } catch (error) {
    console.log(`${colors.red}✗${colors.reset} Results export failed`);
    console.log(`  Error: ${error.message}`);
    testsFailed++;
  }

  testsRun++;
}

/**
 * Test: CLI interface
 */
async function testCLI() {
  console.log('\nTest: CLI interface');

  try {
    const scriptPath = path.join(SCRIPT_DIR, 'network-scanner.js');

    // Run script with help flag (should fail gracefully without --help support)
    try {
      const { stdout, stderr } = await execPromise(
        `node "${scriptPath}" --verbose --output "${TEMP_DIR}/cli-results.json"`,
        { timeout: 30000 }
      );

      console.log(`${colors.green}✓${colors.reset} CLI executed successfully`);
      testsPassed++;
    } catch (error) {
      // CLI might fail due to network issues, but should handle gracefully
      assertContains(
        error.message,
        'stdout\\|stderr\\|timeout',
        'CLI should handle execution'
      );
    }
  } catch (error) {
    console.log(`${colors.red}✗${colors.reset} CLI test failed`);
    console.log(`  Error: ${error.message}`);
    testsFailed++;
  }

  testsRun++;
}

/**
 * Test: Invalid configuration file
 */
async function testInvalidConfigFile() {
  console.log('\nTest: Invalid configuration file');

  try {
    const NetworkScanner = require(path.join(SCRIPT_DIR, 'network-scanner.js'));

    const scanner = new NetworkScanner({
      configFile: '/nonexistent/file.conf',
    });

    try {
      await scanner.parseTrustedDomains();
      console.log(`${colors.red}✗${colors.reset} Should have thrown error for missing file`);
      testsFailed++;
    } catch (error) {
      console.log(`${colors.green}✓${colors.reset} Correctly handles missing config file`);
      testsPassed++;
    }
  } catch (error) {
    console.log(`${colors.red}✗${colors.reset} Invalid config test failed`);
    console.log(`  Error: ${error.message}`);
    testsFailed++;
  }

  testsRun++;
}

/**
 * Run all tests
 */
async function runTests() {
  console.log('=========================================');
  console.log('Network Scanner Tests');
  console.log('=========================================');

  try {
    await setup();

    await testScriptExists();
    await testConfigFileParsing();
    await testDomainResolution();
    await testHttpConnectivity();
    await testResultsExport();
    await testCLI();
    await testInvalidConfigFile();

    await cleanup();

    console.log('\n=========================================');
    console.log('Test Summary');
    console.log('=========================================');
    console.log(`Tests Run: ${testsRun}`);
    console.log(`${colors.green}Passed: ${testsPassed}${colors.reset}`);
    console.log(`${colors.red}Failed: ${testsFailed}${colors.reset}`);
    console.log('=========================================');

    if (testsFailed === 0) {
      console.log(`${colors.green}All tests passed!${colors.reset}`);
      process.exit(0);
    } else {
      console.log(`${colors.red}Some tests failed!${colors.reset}`);
      process.exit(1);
    }
  } catch (error) {
    console.error(`Fatal error: ${error.message}`);
    process.exit(1);
  }
}

// Run tests
runTests();
