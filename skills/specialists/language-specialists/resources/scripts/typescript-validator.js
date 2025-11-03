#!/usr/bin/env node
/**
 * TypeScript Validator - Automated code quality checks for TypeScript projects.
 *
 * Runs tsc --noEmit, ESLint, Prettier, and detects unused exports.
 *
 * Usage:
 *   node typescript-validator.js --dir src/ --strict
 *   node typescript-validator.js --dir . --fix
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

/**
 * Execute shell command and return exit code.
 */
function runCommand(cmd, options = {}) {
  try {
    execSync(cmd, { stdio: 'inherit', ...options });
    return 0;
  } catch (error) {
    return error.status || 1;
  }
}

/**
 * Check TypeScript compilation without emitting files.
 */
function checkTypeScript(directory, strict = false) {
  console.log('🔍 Checking TypeScript compilation...');

  const tsconfigPath = path.join(directory, 'tsconfig.json');
  if (!fs.existsSync(tsconfigPath)) {
    console.log('⚠️  No tsconfig.json found, skipping tsc check');
    return true;
  }

  const cmd = `npx tsc --noEmit --project ${tsconfigPath}`;
  const exitCode = runCommand(cmd);

  if (exitCode === 0) {
    console.log('✅ TypeScript compilation: PASSED\n');
    return true;
  } else {
    console.log('❌ TypeScript compilation: FAILED\n');
    return false;
  }
}

/**
 * Run ESLint on TypeScript files.
 */
function checkESLint(directory, fix = false) {
  console.log('🔍 Running ESLint...');

  const eslintConfigPath = path.join(directory, '.eslintrc.js');
  const hasConfig = fs.existsSync(eslintConfigPath) ||
                    fs.existsSync(path.join(directory, '.eslintrc.json'));

  if (!hasConfig) {
    console.log('⚠️  No ESLint config found, skipping ESLint check');
    return true;
  }

  let cmd = `npx eslint "${directory}/**/*.{ts,tsx}" --max-warnings 0`;
  if (fix) {
    cmd += ' --fix';
  }

  const exitCode = runCommand(cmd);

  if (exitCode === 0) {
    console.log('✅ ESLint: PASSED\n');
    return true;
  } else {
    console.log('❌ ESLint: FAILED\n');
    if (!fix) {
      console.log('💡 Run with --fix to auto-fix ESLint issues\n');
    }
    return false;
  }
}

/**
 * Check code formatting with Prettier.
 */
function checkPrettier(directory, fix = false) {
  console.log('🔍 Checking code formatting with Prettier...');

  const prettierConfigPath = path.join(directory, '.prettierrc');
  const hasConfig = fs.existsSync(prettierConfigPath) ||
                    fs.existsSync(path.join(directory, '.prettierrc.json'));

  if (!hasConfig) {
    console.log('⚠️  No Prettier config found, skipping Prettier check');
    return true;
  }

  let cmd = `npx prettier --check "${directory}/**/*.{ts,tsx,json}"`;
  if (fix) {
    cmd = `npx prettier --write "${directory}/**/*.{ts,tsx,json}"`;
  }

  const exitCode = runCommand(cmd);

  if (exitCode === 0) {
    console.log('✅ Prettier formatting: PASSED\n');
    return true;
  } else {
    console.log('❌ Prettier formatting: FAILED\n');
    if (!fix) {
      console.log('💡 Run with --fix to auto-format files\n');
    }
    return false;
  }
}

/**
 * Detect unused exports (simple heuristic).
 */
function checkUnusedExports(directory) {
  console.log('🔍 Checking for unused exports...');

  // This is a simplified check - for production, use ts-prune or similar
  const cmd = `npx ts-prune ${directory} --error`;
  const exitCode = runCommand(cmd);

  if (exitCode === 0) {
    console.log('✅ No unused exports detected\n');
    return true;
  } else {
    console.log('⚠️  Unused exports detected (warning only)\n');
    // Don't fail on unused exports, just warn
    return true;
  }
}

/**
 * Main entry point.
 */
function main() {
  const args = process.argv.slice(2);
  const directory = args.find(arg => arg.startsWith('--dir='))?.split('=')[1] || '.';
  const strict = args.includes('--strict');
  const fix = args.includes('--fix');

  if (!fs.existsSync(directory)) {
    console.error(`❌ Error: Directory ${directory} does not exist`);
    process.exit(1);
  }

  console.log(`🚀 Running TypeScript validator on: ${path.resolve(directory)}\n`);

  const results = [
    checkTypeScript(directory, strict),
    checkESLint(directory, fix),
    checkPrettier(directory, fix),
    checkUnusedExports(directory)
  ];

  const passed = results.filter(Boolean).length;
  const total = results.length;

  console.log('='.repeat(50));
  console.log(`📊 Results: ${passed}/${total} checks passed`);
  console.log('='.repeat(50));

  if (passed === total) {
    console.log('✅ All checks passed!');
    process.exit(0);
  } else {
    console.log(`❌ ${total - passed} check(s) failed`);
    process.exit(1);
  }
}

main();
