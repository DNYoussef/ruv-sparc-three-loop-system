#!/usr/bin/env node

/**
 * Skill Intake Validation Script
 *
 * Purpose: Validate Phase 1 intake template for completeness and quality
 * Usage: node validate-intake.js path/to/skill-intake.yaml
 * Exit Codes: 0 = pass, 1 = fail
 *
 * Validation Rules:
 * - All required fields must be non-empty
 * - Minimum 3 examples (nominal, edge, error)
 * - Minimum 5 trigger keywords
 * - Minimum 3 constraints
 * - Minimum 3 success criteria
 * - No TODO or PLACEHOLDER text
 */

const fs = require('fs');
const yaml = require('js-yaml');

// ANSI color codes for terminal output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

function validateIntake(filePath) {
  console.log(`${colors.cyan}Validating intake: ${filePath}${colors.reset}\n`);

  // Read and parse YAML
  let intake;
  try {
    const fileContent = fs.readFileSync(filePath, 'utf8');
    intake = yaml.load(fileContent);
  } catch (error) {
    console.error(`${colors.red}✗ Failed to read/parse YAML: ${error.message}${colors.reset}`);
    return false;
  }

  let passCount = 0;
  let failCount = 0;
  const failures = [];

  // Helper functions
  const check = (condition, testName, errorMessage) => {
    if (condition) {
      console.log(`${colors.green}✓ ${testName}${colors.reset}`);
      passCount++;
      return true;
    } else {
      console.log(`${colors.red}✗ ${testName}${colors.reset}`);
      failures.push(`  - ${errorMessage}`);
      failCount++;
      return false;
    }
  };

  const isNonEmpty = (value) => {
    if (typeof value === 'string') {
      return value.trim().length > 0 &&
             !value.includes('TODO') &&
             !value.includes('PLACEHOLDER') &&
             !value.includes('[') &&
             !value.includes(']');
    }
    if (Array.isArray(value)) {
      return value.length > 0 && value.every(item => isNonEmpty(item));
    }
    if (typeof value === 'object' && value !== null) {
      return Object.keys(value).length > 0;
    }
    return false;
  };

  console.log(`${colors.blue}=== Required Fields ===${colors.reset}`);

  // Required field validation
  check(
    isNonEmpty(intake.skill_name),
    'Skill name present',
    'skill_name is missing or contains placeholders'
  );

  check(
    isNonEmpty(intake.skill_category),
    'Skill category present',
    'skill_category is missing or contains placeholders'
  );

  check(
    ['simple', 'medium', 'complex'].includes(intake.complexity_level),
    'Complexity level valid',
    'complexity_level must be: simple, medium, or complex'
  );

  check(
    isNonEmpty(intake.problem_solved),
    'Problem solved described',
    'problem_solved is missing or contains placeholders'
  );

  check(
    isNonEmpty(intake.desired_outcome),
    'Desired outcome described',
    'desired_outcome is missing or contains placeholders'
  );

  console.log(`\n${colors.blue}=== Users & Triggers ===${colors.reset}`);

  check(
    Array.isArray(intake.primary_users) && intake.primary_users.length >= 1,
    'Primary users defined',
    'primary_users must have at least 1 user persona'
  );

  const triggerKeywords = intake.trigger_keywords || [];
  check(
    triggerKeywords.length >= 5 && triggerKeywords.every(k => isNonEmpty(k)),
    'Trigger keywords (minimum 5)',
    `Only ${triggerKeywords.filter(k => isNonEmpty(k)).length}/5 trigger keywords defined`
  );

  const negativeTriggers = intake.negative_triggers || [];
  check(
    negativeTriggers.length >= 3 && negativeTriggers.every(k => isNonEmpty(k)),
    'Negative triggers (minimum 3)',
    `Only ${negativeTriggers.filter(k => isNonEmpty(k)).length}/3 negative triggers defined`
  );

  console.log(`\n${colors.blue}=== Use Cases & Examples ===${colors.reset}`);

  const validateExample = (example, name) => {
    if (!example) return false;
    return isNonEmpty(example.scenario) &&
           isNonEmpty(example.description) &&
           isNonEmpty(example.user_request) &&
           isNonEmpty(example.expected_behavior) &&
           isNonEmpty(example.expected_output);
  };

  check(
    validateExample(intake.example_usage_1, 'example_usage_1'),
    'Example 1 (nominal case) complete',
    'example_usage_1 is incomplete or contains placeholders'
  );

  check(
    validateExample(intake.example_usage_2, 'example_usage_2'),
    'Example 2 (edge case) complete',
    'example_usage_2 is incomplete or contains placeholders'
  );

  check(
    validateExample(intake.example_usage_3, 'example_usage_3'),
    'Example 3 (error case) complete',
    'example_usage_3 is incomplete or contains placeholders'
  );

  console.log(`\n${colors.blue}=== Constraints & Requirements ===${colors.reset}`);

  const constraints = (intake.constraints || []).filter(c => isNonEmpty(c));
  check(
    constraints.length >= 3,
    'Constraints (minimum 3)',
    `Only ${constraints.length}/3 constraints defined`
  );

  const mustHave = (intake.must_have_features || []).filter(f => isNonEmpty(f));
  check(
    mustHave.length >= 3,
    'Must-have features (minimum 3)',
    `Only ${mustHave.length}/3 must-have features defined`
  );

  console.log(`\n${colors.blue}=== Success Criteria ===${colors.reset}`);

  const successCriteria = (intake.success_criteria || []).filter(c => isNonEmpty(c));
  check(
    successCriteria.length >= 3,
    'Success criteria (minimum 3)',
    `Only ${successCriteria.length}/3 success criteria defined`
  );

  const failureConditions = (intake.failure_conditions || []).filter(c => isNonEmpty(c));
  check(
    failureConditions.length >= 3,
    'Failure conditions (minimum 3)',
    `Only ${failureConditions.length}/3 failure conditions defined`
  );

  // Summary
  console.log(`\n${colors.blue}=== Validation Summary ===${colors.reset}`);
  console.log(`${colors.green}Passed: ${passCount}${colors.reset}`);
  console.log(`${colors.red}Failed: ${failCount}${colors.reset}`);

  const passed = failCount === 0;

  if (passed) {
    console.log(`\n${colors.green}✓ All validations passed!${colors.reset}`);
    console.log(`${colors.cyan}Ready to proceed to Phase 2 (Use Case Crystallization)${colors.reset}`);
  } else {
    console.log(`\n${colors.red}✗ Validation failed with ${failCount} errors:${colors.reset}`);
    failures.forEach(f => console.log(`${colors.red}${f}${colors.reset}`));
    console.log(`\n${colors.yellow}Fix the issues above and re-run validation.${colors.reset}`);
  }

  return passed;
}

// Main execution
const args = process.argv.slice(2);
if (args.length !== 1) {
  console.error('Usage: node validate-intake.js path/to/skill-intake.yaml');
  process.exit(1);
}

const filePath = args[0];
if (!fs.existsSync(filePath)) {
  console.error(`Error: File not found: ${filePath}`);
  process.exit(1);
}

const passed = validateIntake(filePath);
process.exit(passed ? 0 : 1);
