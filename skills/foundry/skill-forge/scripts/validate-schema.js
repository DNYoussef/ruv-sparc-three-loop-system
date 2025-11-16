#!/usr/bin/env node

/**
 * Skill Schema Validation Script
 *
 * Purpose: Validate Phase 0 schema for completeness and correctness
 * Usage: node validate-schema.js path/to/skill-schema.json
 * Exit Codes: 0 = pass, 1 = fail, 2 = warnings
 *
 * Validation Rules:
 * - All required sections present (metadata, input_contract, output_contract, error_conditions)
 * - At least 2 output examples (nominal + edge case)
 * - All dependencies have installation checks
 * - Error conditions include recovery strategies
 * - Success conditions are measurable
 * - Schema follows frozen structure (80% locked, 20% flexible)
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

function validateSchema(filePath) {
  console.log(`${colors.cyan}Validating schema: ${filePath}${colors.reset}\n`);

  // Read and parse JSON
  let schema;
  try {
    const fileContent = fs.readFileSync(filePath, 'utf8');
    schema = JSON.parse(fileContent);
  } catch (error) {
    console.error(`${colors.red}✗ Failed to read/parse JSON: ${error.message}${colors.reset}`);
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

  const isNonEmpty = (obj) => {
    if (!obj) return false;
    if (typeof obj === 'string') return obj.trim().length > 0;
    if (Array.isArray(obj)) return obj.length > 0;
    if (typeof obj === 'object') return Object.keys(obj).length > 0;
    return false;
  };

  console.log(`${colors.blue}=== Required Sections ===${colors.reset}`);

  // Check required top-level sections
  check(
    isNonEmpty(schema.skill_metadata),
    'Skill metadata section present',
    'skill_metadata is missing or empty'
  );

  check(
    isNonEmpty(schema.input_contract),
    'Input contract section present',
    'input_contract is missing or empty'
  );

  check(
    isNonEmpty(schema.output_contract),
    'Output contract section present',
    'output_contract is missing or empty'
  );

  check(
    isNonEmpty(schema.error_conditions),
    'Error conditions section present',
    'error_conditions is missing or empty'
  );

  console.log(`\n${colors.blue}=== Skill Metadata ===${colors.reset}`);

  if (schema.skill_metadata) {
    const meta = schema.skill_metadata;

    check(
      isNonEmpty(meta.skill_name),
      'Skill name defined',
      'skill_metadata.skill_name is missing'
    );

    check(
      isNonEmpty(meta.version),
      'Version defined',
      'skill_metadata.version is missing'
    );

    check(
      ['simple', 'medium', 'complex'].includes(meta.complexity),
      'Complexity level valid',
      'skill_metadata.complexity must be: simple, medium, or complex'
    );

    check(
      ['quick', 'expert'].includes(meta.track),
      'Track defined',
      'skill_metadata.track must be: quick or expert'
    );
  }

  console.log(`\n${colors.blue}=== Input Contract ===${colors.reset}`);

  if (schema.input_contract) {
    const input = schema.input_contract;

    check(
      Array.isArray(input.required) && input.required.length > 0,
      'Required inputs defined',
      'input_contract.required must have at least 1 required input'
    );

    // Validate each required input has necessary fields
    if (Array.isArray(input.required)) {
      const validInputs = input.required.filter(inp =>
        isNonEmpty(inp.name) &&
        isNonEmpty(inp.type) &&
        isNonEmpty(inp.description)
      );

      check(
        validInputs.length === input.required.length,
        'All required inputs have name, type, description',
        `${input.required.length - validInputs.length} required inputs are incomplete`
      );
    }

    check(
      Array.isArray(input.constraints),
      'Input constraints array present',
      'input_contract.constraints should be an array',
      true // Warning
    );
  }

  console.log(`\n${colors.blue}=== Output Contract ===${colors.reset}`);

  if (schema.output_contract) {
    const output = schema.output_contract;

    check(
      isNonEmpty(output.format),
      'Output format defined',
      'output_contract.format is missing (text, json, yaml, markdown, code, file)'
    );

    check(
      isNonEmpty(output.schema),
      'Output schema defined',
      'output_contract.schema is missing or empty'
    );

    check(
      Array.isArray(output.success_conditions) && output.success_conditions.length > 0,
      'Success conditions defined',
      'output_contract.success_conditions must have at least 1 condition'
    );

    // Check for minimum 2 output examples
    const examples = output.output_examples || [];
    check(
      examples.length >= 2,
      'Minimum 2 output examples (nominal + edge)',
      `Only ${examples.length}/2 output examples provided`
    );

    // Validate examples have required scenarios
    const scenarios = examples.map(ex => ex.scenario);
    check(
      scenarios.includes('nominal_case'),
      'Nominal case example present',
      'output_examples must include a "nominal_case" scenario',
      true // Warning
    );

    check(
      scenarios.includes('edge_case'),
      'Edge case example present',
      'output_examples must include an "edge_case" scenario',
      true // Warning
    );
  }

  console.log(`\n${colors.blue}=== Error Conditions ===${colors.reset}`);

  if (schema.error_conditions) {
    const errors = schema.error_conditions;

    check(
      Array.isArray(errors.error_types) && errors.error_types.length >= 3,
      'Minimum 3 error types defined',
      `Only ${(errors.error_types || []).length}/3 error types defined`
    );

    // Validate each error has recovery strategy
    if (Array.isArray(errors.error_types)) {
      const errorsWithRecovery = errors.error_types.filter(err =>
        isNonEmpty(err.recovery_strategy)
      );

      check(
        errorsWithRecovery.length === errors.error_types.length,
        'All errors have recovery strategies',
        `${errors.error_types.length - errorsWithRecovery.length} errors missing recovery_strategy`
      );

      // Check for severity levels
      const errorsWithSeverity = errors.error_types.filter(err =>
        ['critical', 'high', 'medium', 'low'].includes(err.severity)
      );

      check(
        errorsWithSeverity.length === errors.error_types.length,
        'All errors have valid severity',
        `${errors.error_types.length - errorsWithSeverity.length} errors missing or invalid severity`
      );
    }
  }

  console.log(`\n${colors.blue}=== Dependencies ===${colors.reset}`);

  if (schema.dependencies) {
    const deps = schema.dependencies;

    // Check tools have installation checks
    if (Array.isArray(deps.tools_required) && deps.tools_required.length > 0) {
      const toolsWithChecks = deps.tools_required.filter(tool =>
        isNonEmpty(tool.installation_check)
      );

      check(
        toolsWithChecks.length === deps.tools_required.length,
        'All required tools have installation checks',
        `${deps.tools_required.length - toolsWithChecks.length} tools missing installation_check`,
        deps.tools_required.length - toolsWithChecks.length > 0 ? false : true
      );
    }

    // Warn if no dependencies defined for complex skills
    if (schema.skill_metadata?.complexity === 'complex') {
      const hasDeps = (deps.tools_required && deps.tools_required.length > 0) ||
                      (deps.files_required && deps.files_required.length > 0) ||
                      (deps.api_dependencies && deps.api_dependencies.length > 0);

      check(
        hasDeps,
        'Complex skill has dependencies documented',
        'Complex skills typically have dependencies (tools, files, or APIs)',
        true // Warning
      );
    }
  }

  console.log(`\n${colors.blue}=== Performance Contract ===${colors.reset}`);

  if (schema.performance_contract) {
    const perf = schema.performance_contract;

    check(
      isNonEmpty(perf.time_complexity),
      'Time complexity documented',
      'performance_contract.time_complexity should specify best/average/worst case',
      true // Warning
    );

    check(
      isNonEmpty(perf.space_complexity),
      'Space complexity documented',
      'performance_contract.space_complexity should specify memory/disk usage',
      true // Warning
    );

    check(
      isNonEmpty(perf.scalability),
      'Scalability limits documented',
      'performance_contract.scalability should specify max input/output sizes',
      true // Warning
    );
  }

  console.log(`\n${colors.blue}=== Testing Contract ===${colors.reset}`);

  if (schema.testing_contract) {
    const testing = schema.testing_contract;

    check(
      Array.isArray(testing.test_cases) && testing.test_cases.length >= 3,
      'Minimum 3 test cases defined',
      `Only ${(testing.test_cases || []).length}/3 test cases defined`,
      true // Warning
    );

    // Check for diverse test case types
    if (Array.isArray(testing.test_cases) && testing.test_cases.length > 0) {
      const types = testing.test_cases.map(tc => tc.type);
      const uniqueTypes = [...new Set(types)];

      check(
        uniqueTypes.length >= 3,
        'Test cases cover diverse types',
        `Only ${uniqueTypes.length} test case types (need nominal, edge, error, boundary, integration)`,
        true // Warning
      );
    }

    check(
      isNonEmpty(testing.coverage_requirements),
      'Coverage requirements defined',
      'testing_contract.coverage_requirements should specify >=80% code, >=95% use case',
      true // Warning
    );
  }

  console.log(`\n${colors.blue}=== Frozen Structure Compliance ===${colors.reset}`);

  // Check frozen structure section exists
  check(
    isNonEmpty(schema.frozen_structure),
    'Frozen structure section present',
    'frozen_structure section documents which elements are locked vs flexible',
    true // Warning
  );

  // Check versioning contract
  check(
    isNonEmpty(schema.versioning_contract),
    'Versioning contract present',
    'versioning_contract defines how schema changes are versioned',
    true // Warning
  );

  console.log(`\n${colors.blue}=== Schema Quality Score ===${colors.reset}`);

  // Calculate completeness score
  const requiredSections = ['skill_metadata', 'input_contract', 'output_contract', 'error_conditions'];
  const optionalSections = ['behavior_contract', 'dependencies', 'performance_contract', 'testing_contract'];

  const requiredPresent = requiredSections.filter(sec => isNonEmpty(schema[sec])).length;
  const optionalPresent = optionalSections.filter(sec => isNonEmpty(schema[sec])).length;

  const completenessScore = Math.round(
    ((requiredPresent / requiredSections.length) * 60) +
    ((optionalPresent / optionalSections.length) * 40)
  );

  check(
    completenessScore >= 80,
    `Schema completeness: ${completenessScore}%`,
    `Schema completeness below 80% (missing optional sections improve quality)`,
    completenessScore >= 60 // Error if <60%, warning if 60-79%
  );

  // Summary
  console.log(`\n${colors.blue}=== Validation Summary ===${colors.reset}`);
  console.log(`${colors.green}Passed: ${passCount}${colors.reset}`);
  console.log(`${colors.red}Failed: ${failCount}${colors.reset}`);
  console.log(`${colors.yellow}Warnings: ${warnCount}${colors.reset}`);
  console.log(`${colors.cyan}Completeness: ${completenessScore}%${colors.reset}`);

  const passed = failCount === 0;
  const hasWarnings = warnCount > 0;

  if (passed && !hasWarnings) {
    console.log(`\n${colors.green}✓ Schema validation passed!${colors.reset}`);
    console.log(`${colors.cyan}Ready to proceed with schema-first development (Phase 1+).${colors.reset}`);
  } else if (passed && hasWarnings) {
    console.log(`\n${colors.yellow}⚠ Schema passed with ${warnCount} warnings:${colors.reset}`);
    warnings.forEach(w => console.log(`${colors.yellow}${w}${colors.reset}`));
    console.log(`\n${colors.cyan}Schema is acceptable but could be more comprehensive.${colors.reset}`);
  } else {
    console.log(`\n${colors.red}✗ Schema validation failed with ${failCount} errors:${colors.reset}`);
    failures.forEach(f => console.log(`${colors.red}${f}${colors.reset}`));
    if (hasWarnings) {
      console.log(`\n${colors.yellow}Additionally, ${warnCount} warnings:${colors.reset}`);
      warnings.forEach(w => console.log(`${colors.yellow}${w}${colors.reset}`));
    }
    console.log(`\n${colors.yellow}Fix the errors above before proceeding to Phase 1.${colors.reset}`);
  }

  return { passed, warnings: warnCount };
}

// Main execution
const args = process.argv.slice(2);
if (args.length !== 1) {
  console.error('Usage: node validate-schema.js path/to/skill-schema.json');
  process.exit(1);
}

const filePath = args[0];
if (!fs.existsSync(filePath)) {
  console.error(`Error: File not found: ${filePath}`);
  process.exit(1);
}

const result = validateSchema(filePath);
if (result.passed) {
  process.exit(result.warnings > 0 ? 2 : 0); // 2 = warnings, 0 = perfect
} else {
  process.exit(1); // 1 = failed
}
