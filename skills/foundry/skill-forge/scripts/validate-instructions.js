#!/usr/bin/env node

/**
 * Skill Instructions Validation Script
 *
 * Purpose: Check Phase 5 instructions for anti-patterns and completeness
 * Usage: node validate-instructions.js path/to/SKILL.md
 * Exit Codes: 0 = pass, 1 = fail, 2 = warnings
 *
 * Validation Rules:
 * - Every step has explicit success criteria
 * - No vague verbs (handle, process, deal with, manage, work on)
 * - Error handling present for critical operations
 * - At least 3 edge cases documented
 * - Clear error codes and recovery strategies
 * - No TODO or PLACEHOLDER text in instructions
 */

const fs = require('fs');

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

// Anti-pattern detection
const VAGUE_VERBS = [
  'handle', 'process', 'deal with', 'manage', 'work on',
  'take care of', 'sort out', 'figure out', 'check out'
];

const REQUIRED_ERROR_KEYWORDS = [
  'if error', 'try-catch', 'error handling', 'on failure',
  'exit code', 'handle error', 'catch', 'throw'
];

function validateInstructions(filePath) {
  console.log(`${colors.cyan}Validating instructions: ${filePath}${colors.reset}\n`);

  // Read markdown file
  let content;
  try {
    content = fs.readFileSync(filePath, 'utf8');
  } catch (error) {
    console.error(`${colors.red}✗ Failed to read file: ${error.message}${colors.reset}`);
    return { passed: false, warnings: 0 };
  }

  let passCount = 0;
  let failCount = 0;
  let warnCount = 0;
  const failures = [];
  const warnings = [];

  // Helper functions
  const check = (condition, testName, errorMessage, isWarning = false) => {
    if (condition) {
      console.log(`${colors.green}✓ ${testName}${colors.reset}`);
      passCount++;
      return true;
    } else {
      if (isWarning) {
        console.log(`${colors.yellow}⚠ ${testName}${colors.reset}`);
        warnings.push(`  - ${errorMessage}`);
        warnCount++;
      } else {
        console.log(`${colors.red}✗ ${testName}${colors.reset}`);
        failures.push(`  - ${errorMessage}`);
        failCount++;
      }
      return false;
    }
  };

  console.log(`${colors.blue}=== Anti-Pattern Detection ===${colors.reset}`);

  // Check for vague verbs
  const vaguVerbsFound = VAGUE_VERBS.filter(verb => {
    const regex = new RegExp(`\\b${verb}\\b`, 'i');
    return regex.test(content);
  });

  check(
    vaguVerbsFound.length === 0,
    'No vague verbs detected',
    `Found vague verbs: ${vaguVerbsFound.join(', ')}. Replace with specific actions.`
  );

  // Check for placeholders
  const hasPlaceholders = content.includes('[PLACEHOLDER]') ||
                          content.includes('TODO:') ||
                          content.includes('[TODO]');

  check(
    !hasPlaceholders,
    'No placeholders or TODOs',
    'Instructions contain [PLACEHOLDER] or TODO markers'
  );

  console.log(`\n${colors.blue}=== Success Criteria ===${colors.reset}`);

  // Count steps with success criteria
  const stepRegex = /### Step \d+:/g;
  const successCriteriaRegex = /\*\*Success Criteria\*\*:/g;

  const stepMatches = content.match(stepRegex) || [];
  const successMatches = content.match(successCriteriaRegex) || [];

  check(
    stepMatches.length > 0,
    'Instructions have numbered steps',
    'No numbered steps found (### Step N:)'
  );

  check(
    successMatches.length >= stepMatches.length,
    'All steps have success criteria',
    `Only ${successMatches.length}/${stepMatches.length} steps have explicit success criteria`
  );

  // Check for checkmark lists in success criteria
  const successCheckmarks = (content.match(/- ✓/g) || []).length;
  check(
    successCheckmarks >= stepMatches.length * 2,
    'Success criteria use checkmark lists',
    `Expected at least ${stepMatches.length * 2} checkmarks (2+ per step), found ${successCheckmarks}`,
    true // Warning, not failure
  );

  console.log(`\n${colors.blue}=== Error Handling ===${colors.reset}`);

  // Check for error handling sections
  const errorHandlingMatches = content.match(/\*\*Error Handling\*\*:/g) || [];

  check(
    errorHandlingMatches.length >= stepMatches.length,
    'All steps have error handling',
    `Only ${errorHandlingMatches.length}/${stepMatches.length} steps have error handling sections`
  );

  // Check for specific error handling keywords
  const hasErrorKeywords = REQUIRED_ERROR_KEYWORDS.some(keyword =>
    content.toLowerCase().includes(keyword.toLowerCase())
  );

  check(
    hasErrorKeywords,
    'Error handling uses proper techniques',
    'No error handling keywords found (if error, try-catch, exit code, etc.)'
  );

  // Check for exit codes or error codes table
  const hasErrorCodes = content.includes('Error Code') ||
                        content.includes('exit code') ||
                        content.includes('Exit Code');

  check(
    hasErrorCodes,
    'Error codes documented',
    'No error codes table or exit code documentation found',
    true // Warning
  );

  console.log(`\n${colors.blue}=== Edge Cases ===${colors.reset}`);

  // Count edge case sections
  const edgeCaseRegex = /### Edge Case \d+:/g;
  const edgeCaseMatches = content.match(edgeCaseRegex) || [];

  check(
    edgeCaseMatches.length >= 3,
    'Minimum 3 edge cases documented',
    `Only ${edgeCaseMatches.length}/3 edge cases documented`
  );

  // Check for handling instructions in edge cases
  const handlingRegex = /\*\*Handling\*\*:/g;
  const handlingMatches = content.match(handlingRegex) || [];

  check(
    handlingMatches.length >= edgeCaseMatches.length,
    'All edge cases have handling instructions',
    `Only ${handlingMatches.length}/${edgeCaseMatches.length} edge cases have handling instructions`
  );

  console.log(`\n${colors.blue}=== Code Examples ===${colors.reset}`);

  // Count code blocks
  const codeBlockRegex = /```[\s\S]*?```/g;
  const codeBlocks = content.match(codeBlockRegex) || [];

  check(
    codeBlocks.length >= stepMatches.length,
    'Code examples present for steps',
    `Expected at least ${stepMatches.length} code examples, found ${codeBlocks.length}`,
    true // Warning
  );

  // Check for bash/shell code blocks (common in skills)
  const bashCodeRegex = /```(?:bash|shell|sh)/g;
  const bashBlocks = content.match(bashCodeRegex) || [];

  check(
    bashBlocks.length > 0,
    'Executable code examples present',
    'No bash/shell code blocks found',
    true // Warning
  );

  console.log(`\n${colors.blue}=== Completeness ===${colors.reset}`);

  // Check for performance expectations
  const hasPerformance = content.includes('Performance') ||
                         content.includes('Execution Time') ||
                         content.includes('Timeout');

  check(
    hasPerformance,
    'Performance expectations defined',
    'No performance/timeout expectations documented',
    true // Warning
  );

  // Check for verification/testing section
  const hasVerification = content.includes('Verification') ||
                          content.includes('Success Verification') ||
                          content.includes('Test');

  check(
    hasVerification,
    'Verification section present',
    'No verification or testing checklist found',
    true // Warning
  );

  console.log(`\n${colors.blue}=== Actionability Metrics ===${colors.reset}`);

  // Calculate actionability score
  const totalInstructions = stepMatches.length + edgeCaseMatches.length;
  const instructionsWithCriteria = successMatches.length + handlingMatches.length;
  const actionabilityPercent = totalInstructions > 0
    ? Math.round((instructionsWithCriteria / totalInstructions) * 100)
    : 0;

  check(
    actionabilityPercent >= 80,
    `Actionability score: ${actionabilityPercent}%`,
    `Actionability below 80% (target for Phase 8 metrics)`
  );

  // Summary
  console.log(`\n${colors.blue}=== Validation Summary ===${colors.reset}`);
  console.log(`${colors.green}Passed: ${passCount}${colors.reset}`);
  console.log(`${colors.red}Failed: ${failCount}${colors.reset}`);
  console.log(`${colors.yellow}Warnings: ${warnCount}${colors.reset}`);

  const passed = failCount === 0;
  const hasWarnings = warnCount > 0;

  if (passed && !hasWarnings) {
    console.log(`\n${colors.green}✓ All validations passed!${colors.reset}`);
    console.log(`${colors.cyan}Instructions are high quality and ready for deployment.${colors.reset}`);
  } else if (passed && hasWarnings) {
    console.log(`\n${colors.yellow}⚠ Validation passed with ${warnCount} warnings:${colors.reset}`);
    warnings.forEach(w => console.log(`${colors.yellow}${w}${colors.reset}`));
    console.log(`\n${colors.cyan}Instructions are acceptable but could be improved.${colors.reset}`);
  } else {
    console.log(`\n${colors.red}✗ Validation failed with ${failCount} errors:${colors.reset}`);
    failures.forEach(f => console.log(`${colors.red}${f}${colors.reset}`));
    if (hasWarnings) {
      console.log(`\n${colors.yellow}Additionally, ${warnCount} warnings:${colors.reset}`);
      warnings.forEach(w => console.log(`${colors.yellow}${w}${colors.reset}`));
    }
    console.log(`\n${colors.yellow}Fix the errors above before proceeding to Phase 6.${colors.reset}`);
  }

  return { passed, warnings: warnCount };
}

// Main execution
const args = process.argv.slice(2);
if (args.length !== 1) {
  console.error('Usage: node validate-instructions.js path/to/SKILL.md');
  process.exit(1);
}

const filePath = args[0];
if (!fs.existsSync(filePath)) {
  console.error(`Error: File not found: ${filePath}`);
  process.exit(1);
}

const result = validateInstructions(filePath);
if (result.passed) {
  process.exit(result.warnings > 0 ? 2 : 0); // 2 = warnings, 0 = perfect
} else {
  process.exit(1); // 1 = failed
}
