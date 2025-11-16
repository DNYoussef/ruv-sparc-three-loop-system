#!/usr/bin/env node

/**
 * Complete Skill Validation Script
 *
 * Purpose: Comprehensive validation of complete skill (all phases)
 * Usage: node validate-skill.js path/to/skill/directory
 * Exit Codes: 0 = pass, 1 = fail, 2 = warnings
 *
 * Validates:
 * - YAML frontmatter (metadata)
 * - Skill structure and organization
 * - Naming conventions
 * - File organization
 * - Resource references
 * - Overall quality
 */

const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  bold: '\x1b[1m',
};

function validateSkill(skillDir) {
  console.log(`${colors.bold}${colors.cyan}==============================================`);
  console.log(`Comprehensive Skill Validation`);
  console.log(`==============================================` + colors.reset);
  console.log(`${colors.cyan}Validating: ${skillDir}${colors.reset}\n`);

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

  const isNonEmpty = (value) => {
    if (typeof value === 'string') return value.trim().length > 0;
    if (Array.isArray(value)) return value.length > 0;
    if (typeof value === 'object' && value !== null) return Object.keys(value).length > 0;
    return false;
  };

  console.log(`${colors.blue}=== File Structure ===${colors.reset}`);

  // Check SKILL.md exists
  const skillMdPath = path.join(skillDir, 'SKILL.md');
  const hasSkillMd = fs.existsSync(skillMdPath);

  check(
    hasSkillMd,
    'SKILL.md file exists',
    'SKILL.md file is missing'
  );

  if (!hasSkillMd) {
    console.log(`\n${colors.red}Cannot continue validation without SKILL.md${colors.reset}`);
    return { passed: false, warnings: 0 };
  }

  // Read SKILL.md
  let skillContent;
  try {
    skillContent = fs.readFileSync(skillMdPath, 'utf8');
  } catch (error) {
    console.error(`${colors.red}✗ Failed to read SKILL.md: ${error.message}${colors.reset}`);
    return { passed: false, warnings: 0 };
  }

  console.log(`\n${colors.blue}=== YAML Frontmatter ===${colors.reset}`);

  // Extract YAML frontmatter
  const frontmatterRegex = /^---\n([\s\S]*?)\n---/;
  const frontmatterMatch = skillContent.match(frontmatterRegex);

  check(
    frontmatterMatch !== null,
    'YAML frontmatter present',
    'SKILL.md must start with YAML frontmatter between --- delimiters'
  );

  let metadata = null;
  if (frontmatterMatch) {
    try {
      metadata = yaml.load(frontmatterMatch[1]);
    } catch (error) {
      check(false, 'YAML frontmatter valid', `YAML parsing error: ${error.message}`);
    }
  }

  if (metadata) {
    check(
      isNonEmpty(metadata.name),
      'Skill name defined',
      'Frontmatter must include "name" field'
    );

    check(
      isNonEmpty(metadata.description),
      'Description defined',
      'Frontmatter must include "description" field'
    );

    check(
      metadata.description && metadata.description.length >= 50,
      'Description is detailed (≥50 chars)',
      `Description is too short: ${metadata.description?.length || 0} chars (minimum 50)`,
      true // Warning
    );

    check(
      isNonEmpty(metadata.author),
      'Author defined',
      'Frontmatter should include "author" field',
      true // Warning
    );

    // Check for version or created_date
    check(
      isNonEmpty(metadata.version) || isNonEmpty(metadata.created_date),
      'Version or created_date present',
      'Frontmatter should include "version" or "created_date" for tracking',
      true // Warning
    );
  }

  console.log(`\n${colors.blue}=== Content Structure ===${colors.reset}`);

  // Check for section headers
  const hasOverview = skillContent.includes('## Overview') || skillContent.includes('# Overview');
  check(
    hasOverview,
    'Overview section present',
    'SKILL.md should include an Overview section',
    true // Warning
  );

  const hasWhenToUse = skillContent.includes('When to Use') || skillContent.includes('When to use');
  check(
    hasWhenToUse,
    'When to Use section present',
    'SKILL.md should explain when to use this skill',
    true // Warning
  );

  const hasInstructions = skillContent.includes('Instructions') || skillContent.includes('## How');
  check(
    hasInstructions,
    'Instructions section present',
    'SKILL.md must include instructions for Claude'
  );

  // Check for examples
  const exampleCount = (skillContent.match(/```/g) || []).length / 2; // Code blocks come in pairs
  check(
    exampleCount >= 2,
    'At least 2 code examples present',
    `Only ${Math.floor(exampleCount)} code examples found (recommended: 2+)`,
    exampleCount < 1 ? false : true // Error if 0, warning if 1
  );

  console.log(`\n${colors.blue}=== Naming Conventions ===${colors.reset}`);

  const skillName = metadata?.name || '';
  const dirName = path.basename(skillDir);

  // Check kebab-case
  const isKebabCase = /^[a-z0-9]+(-[a-z0-9]+)*$/.test(dirName);
  check(
    isKebabCase,
    'Directory name uses kebab-case',
    `Directory name "${dirName}" should use kebab-case (lowercase-with-hyphens)`
  );

  // Check name matches directory
  const nameSlug = skillName.toLowerCase().replace(/\s+/g, '-');
  check(
    nameSlug === dirName || skillName === dirName,
    'Skill name matches directory',
    `Skill name "${skillName}" doesn't match directory "${dirName}"`,
    true // Warning
  );

  console.log(`\n${colors.blue}=== File Organization ===${colors.reset}`);

  // Check for optional but recommended files
  const hasReadme = fs.existsSync(path.join(skillDir, 'README.md'));
  check(
    hasReadme,
    'README.md present (optional)',
    'README.md provides additional documentation for users',
    true // Warning
  );

  // Check for subdirectories
  const subdirs = [];
  try {
    const entries = fs.readdirSync(skillDir, { withFileTypes: true });
    entries.forEach(entry => {
      if (entry.isDirectory()) {
        subdirs.push(entry.name);
      }
    });
  } catch (error) {
    // Ignore
  }

  // Recommended subdirectories
  const hasScripts = subdirs.includes('scripts');
  const hasTemplates = subdirs.includes('templates');
  const hasReferences = subdirs.includes('references');

  if (subdirs.length > 0) {
    check(
      hasScripts || hasTemplates || hasReferences,
      'Organized subdirectories present',
      'Subdirectories found but none are standard (scripts/, templates/, references/)',
      true // Warning
    );
  }

  console.log(`\n${colors.blue}=== Resource References ===${colors.reset}`);

  // Check for script references in SKILL.md
  const scriptRefs = skillContent.match(/scripts\/[a-zA-Z0-9_-]+\.(js|sh|py)/g) || [];
  if (scriptRefs.length > 0) {
    const scriptsExist = scriptRefs.every(ref => {
      const scriptPath = path.join(skillDir, ref);
      return fs.existsSync(scriptPath);
    });

    check(
      scriptsExist,
      'All referenced scripts exist',
      `${scriptRefs.length - scriptRefs.filter(ref => fs.existsSync(path.join(skillDir, ref))).length} referenced scripts not found`
    );
  }

  // Check for template references
  const templateRefs = skillContent.match(/templates\/[a-zA-Z0-9_-]+\.(yaml|json|md)/g) || [];
  if (templateRefs.length > 0) {
    const templatesExist = templateRefs.every(ref => {
      const templatePath = path.join(skillDir, ref);
      return fs.existsSync(templatePath);
    });

    check(
      templatesExist,
      'All referenced templates exist',
      `${templateRefs.length - templateRefs.filter(ref => fs.existsSync(path.join(skillDir, ref))).length} referenced templates not found`
    );
  }

  console.log(`\n${colors.blue}=== Quality Indicators ===${colors.reset}`);

  // Check for anti-patterns
  const hasVagueVerbs = /\b(handle|process|deal with|manage|work on)\b/i.test(skillContent);
  check(
    !hasVagueVerbs,
    'No vague verbs in instructions',
    'Instructions contain vague verbs (handle, process, deal with, manage)',
    true // Warning
  );

  // Check for success criteria
  const hasSuccessCriteria = skillContent.includes('Success') ||
                             skillContent.includes('success') ||
                             skillContent.includes('✓');
  check(
    hasSuccessCriteria,
    'Success criteria present',
    'Instructions should include explicit success criteria',
    true // Warning
  );

  // Check for error handling
  const hasErrorHandling = skillContent.includes('Error') ||
                           skillContent.includes('error') ||
                           skillContent.includes('fail');
  check(
    hasErrorHandling,
    'Error handling present',
    'Instructions should include error handling guidance',
    true // Warning
  );

  // Check for edge cases
  const hasEdgeCases = skillContent.includes('edge case') ||
                       skillContent.includes('Edge Case') ||
                       skillContent.includes('Edge case');
  check(
    hasEdgeCases,
    'Edge cases documented',
    'Instructions should document edge cases',
    true // Warning
  );

  console.log(`\n${colors.blue}=== Completeness Score ===${colors.reset}`);

  // Calculate completeness
  const requiredElements = [
    hasSkillMd,
    metadata !== null,
    metadata?.name,
    metadata?.description,
    hasInstructions
  ];

  const optionalElements = [
    hasOverview,
    hasWhenToUse,
    exampleCount >= 2,
    hasReadme,
    hasSuccessCriteria,
    hasErrorHandling,
    hasEdgeCases,
    hasScripts || hasTemplates || hasReferences
  ];

  const requiredScore = requiredElements.filter(Boolean).length / requiredElements.length;
  const optionalScore = optionalElements.filter(Boolean).length / optionalElements.length;
  const completenessPercent = Math.round((requiredScore * 60) + (optionalScore * 40));

  check(
    completenessPercent >= 70,
    `Skill completeness: ${completenessPercent}%`,
    `Completeness below 70% (missing recommended elements)`,
    completenessPercent >= 50 // Error if <50%, warning if 50-69%
  );

  // Summary
  console.log(`\n${colors.bold}${colors.blue}=== Validation Summary ===${colors.reset}`);
  console.log(`${colors.green}Passed: ${passCount}${colors.reset}`);
  console.log(`${colors.red}Failed: ${failCount}${colors.reset}`);
  console.log(`${colors.yellow}Warnings: ${warnCount}${colors.reset}`);
  console.log(`${colors.cyan}Completeness: ${completenessPercent}%${colors.reset}`);

  const passed = failCount === 0;
  const hasWarnings = warnCount > 0;

  if (passed && !hasWarnings) {
    console.log(`\n${colors.green}${colors.bold}✓ All validations passed!${colors.reset}`);
    console.log(`${colors.cyan}Skill is high quality and ready for deployment.${colors.reset}`);
  } else if (passed && hasWarnings) {
    console.log(`\n${colors.yellow}⚠ Validation passed with ${warnCount} warnings:${colors.reset}`);
    warnings.forEach(w => console.log(`${colors.yellow}${w}${colors.reset}`));
    console.log(`\n${colors.cyan}Skill is acceptable but could be improved.${colors.reset}`);
  } else {
    console.log(`\n${colors.red}✗ Validation failed with ${failCount} errors:${colors.reset}`);
    failures.forEach(f => console.log(`${colors.red}${f}${colors.reset}`));
    if (hasWarnings) {
      console.log(`\n${colors.yellow}Additionally, ${warnCount} warnings:${colors.reset}`);
      warnings.forEach(w => console.log(`${colors.yellow}${w}${colors.reset}`));
    }
    console.log(`\n${colors.yellow}Fix the errors above before deploying the skill.${colors.reset}`);
  }

  return { passed, warnings: warnCount };
}

// Main execution
const args = process.argv.slice(2);
if (args.length !== 1) {
  console.error('Usage: node validate-skill.js path/to/skill/directory');
  process.exit(1);
}

const skillDir = args[0];
if (!fs.existsSync(skillDir)) {
  console.error(`Error: Directory not found: ${skillDir}`);
  process.exit(1);
}

const result = validateSkill(skillDir);
if (result.passed) {
  process.exit(result.warnings > 0 ? 2 : 0); // 2 = warnings, 0 = perfect
} else {
  process.exit(1); // 1 = failed
}
