#!/usr/bin/env node

/**
 * Test suite for requirements-collector.js
 * Validates answer processing, requirements synthesis, and specification generation.
 */

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { promisify } = require('util');
const writeFile = promisify(fs.writeFile);
const readFile = promisify(fs.readFile);
const unlink = promisify(fs.unlink);

// Import modules under test
const { RequirementsSpec, QuestionAnswer } = require('../resources/requirements-collector.js');

/**
 * Test QuestionAnswer class
 */
function testQuestionAnswer() {
  console.log('Testing QuestionAnswer class...');

  // Test single selection
  const singleAnswer = new QuestionAnswer(
    'q1',
    'What framework?',
    'React/Next.js',
    false
  );

  assert.strictEqual(singleAnswer.selectedOptions.length, 1);
  assert.strictEqual(singleAnswer.selectedOptions[0], 'React/Next.js');
  assert.strictEqual(singleAnswer.isMultiSelect, false);

  // Test multi-selection
  const multiAnswer = new QuestionAnswer(
    'q2',
    'Which features?',
    ['User management', 'Real-time updates'],
    true
  );

  assert.strictEqual(multiAnswer.selectedOptions.length, 2);
  assert.strictEqual(multiAnswer.isMultiSelect, true);

  // Test "Other" detection
  const otherAnswer = new QuestionAnswer(
    'q3',
    'What database?',
    'Other',
    false
  );

  assert.strictEqual(otherAnswer.hasOtherSelected(), true);
  assert.strictEqual(otherAnswer.isComplete(), false);

  // Test completeness
  assert.strictEqual(singleAnswer.isComplete(), true);
  assert.strictEqual(multiAnswer.isComplete(), true);

  console.log('✓ QuestionAnswer tests passed');
}

/**
 * Test RequirementsSpec class
 */
function testRequirementsSpec() {
  console.log('Testing RequirementsSpec class...');

  const spec = new RequirementsSpec();

  // Test adding answers
  const answer1 = new QuestionAnswer(
    'purpose',
    'What is the primary purpose?',
    'New feature',
    false
  );

  const answer2 = new QuestionAnswer(
    'stack',
    'What technology stack?',
    'React/Next.js',
    false
  );

  spec.addAnswer(answer1);
  spec.addAnswer(answer2);

  assert.strictEqual(spec.answers.length, 2);

  // Test synthesize
  spec.synthesize();

  assert.ok(spec.projectScope);
  assert.ok(spec.technicalDecisions);

  console.log('✓ RequirementsSpec tests passed');
}

/**
 * Test project scope extraction
 */
function testProjectScopeExtraction() {
  console.log('Testing project scope extraction...');

  const spec = new RequirementsSpec();

  // Add purpose answer
  spec.addAnswer(new QuestionAnswer(
    'purpose',
    'What is the primary purpose of this project?',
    'New feature',
    false
  ));

  // Add project type answer
  spec.addAnswer(new QuestionAnswer(
    'type',
    'What project type is this?',
    'Web application',
    false
  ));

  // Add complexity answer
  spec.addAnswer(new QuestionAnswer(
    'complexity',
    'What is the project complexity?',
    'Moderate',
    false
  ));

  spec.synthesize();

  assert.strictEqual(spec.projectScope.purpose, 'New feature');
  assert.strictEqual(spec.projectScope.projectType, 'Web application');
  assert.strictEqual(spec.projectScope.complexity, 'Moderate');

  console.log('✓ Project scope extraction tests passed');
}

/**
 * Test technical decisions extraction
 */
function testTechnicalDecisionsExtraction() {
  console.log('Testing technical decisions extraction...');

  const spec = new RequirementsSpec();

  // Add framework answer
  spec.addAnswer(new QuestionAnswer(
    'framework',
    'What framework should we use?',
    'React/Next.js',
    false
  ));

  // Add database answer
  spec.addAnswer(new QuestionAnswer(
    'database',
    'What database type?',
    'PostgreSQL',
    false
  ));

  // Add authentication answer
  spec.addAnswer(new QuestionAnswer(
    'auth',
    'What authentication method?',
    'OAuth2',
    false
  ));

  // Add backend patterns (multi-select)
  spec.addAnswer(new QuestionAnswer(
    'backend',
    'Which backend patterns?',
    ['REST API', 'WebSockets'],
    true
  ));

  spec.synthesize();

  assert.strictEqual(spec.technicalDecisions.stack, 'React/Next.js');
  assert.strictEqual(spec.technicalDecisions.database, 'PostgreSQL');
  assert.strictEqual(spec.technicalDecisions.authentication, 'OAuth2');
  assert.ok(Array.isArray(spec.technicalDecisions.backendPatterns));
  assert.strictEqual(spec.technicalDecisions.backendPatterns.length, 2);

  console.log('✓ Technical decisions extraction tests passed');
}

/**
 * Test feature list extraction
 */
function testFeatureListExtraction() {
  console.log('Testing feature list extraction...');

  const spec = new RequirementsSpec();

  // Add features (multi-select)
  spec.addAnswer(new QuestionAnswer(
    'features',
    'Which features are needed?',
    ['User management', 'Real-time updates', 'File handling'],
    true
  ));

  // Add another features question
  spec.addAnswer(new QuestionAnswer(
    'functionality',
    'What functionality should we add?',
    ['Search/filtering', 'Analytics'],
    true
  ));

  spec.synthesize();

  assert.ok(Array.isArray(spec.featureList));
  assert.strictEqual(spec.featureList.length, 5);
  assert.ok(spec.featureList.includes('User management'));
  assert.ok(spec.featureList.includes('Search/filtering'));

  console.log('✓ Feature list extraction tests passed');
}

/**
 * Test quality requirements extraction
 */
function testQualityRequirementsExtraction() {
  console.log('Testing quality requirements extraction...');

  const spec = new RequirementsSpec();

  // Add testing answer (multi-select)
  spec.addAnswer(new QuestionAnswer(
    'testing',
    'What testing coverage?',
    ['Unit tests', 'Integration tests', 'E2E tests'],
    true
  ));

  // Add quality level
  spec.addAnswer(new QuestionAnswer(
    'quality',
    'What quality level?',
    'Production MVP',
    false
  ));

  // Add performance
  spec.addAnswer(new QuestionAnswer(
    'performance',
    'What performance requirements?',
    'Medium scale',
    false
  ));

  spec.synthesize();

  assert.ok(Array.isArray(spec.qualityRequirements.testing));
  assert.strictEqual(spec.qualityRequirements.testing.length, 3);
  assert.strictEqual(spec.qualityRequirements.level, 'Production MVP');
  assert.strictEqual(spec.qualityRequirements.performance, 'Medium scale');

  console.log('✓ Quality requirements extraction tests passed');
}

/**
 * Test constraints extraction
 */
function testConstraintsExtraction() {
  console.log('Testing constraints extraction...');

  const spec = new RequirementsSpec();

  // Add timeline
  spec.addAnswer(new QuestionAnswer(
    'timeline',
    'What is the timeline?',
    'This month',
    false
  ));

  // Add budget
  spec.addAnswer(new QuestionAnswer(
    'budget',
    'What is the budget?',
    'Moderate',
    false
  ));

  spec.synthesize();

  assert.ok(Array.isArray(spec.constraints));
  assert.strictEqual(spec.constraints.length, 2);
  assert.ok(spec.constraints[0].includes('Timeline'));
  assert.ok(spec.constraints[1].includes('Budget'));

  console.log('✓ Constraints extraction tests passed');
}

/**
 * Test confidence level calculation
 */
function testConfidenceLevel() {
  console.log('Testing confidence level calculation...');

  // High confidence (all complete)
  const highConfSpec = new RequirementsSpec();
  highConfSpec.addAnswer(new QuestionAnswer('q1', 'Question 1?', 'Answer 1', false));
  highConfSpec.addAnswer(new QuestionAnswer('q2', 'Question 2?', 'Answer 2', false));
  highConfSpec.addAnswer(new QuestionAnswer('q3', 'Question 3?', 'Answer 3', false));

  assert.strictEqual(highConfSpec.getConfidenceLevel(), 'high');

  // Medium confidence (some incomplete)
  const medConfSpec = new RequirementsSpec();
  medConfSpec.addAnswer(new QuestionAnswer('q1', 'Question 1?', 'Answer 1', false));
  medConfSpec.addAnswer(new QuestionAnswer('q2', 'Question 2?', 'Other', false));
  medConfSpec.addAnswer(new QuestionAnswer('q3', 'Question 3?', 'Answer 3', false));
  medConfSpec.addAnswer(new QuestionAnswer('q4', 'Question 4?', 'Answer 4', false));

  assert.strictEqual(medConfSpec.getConfidenceLevel(), 'medium');

  // Low confidence (many incomplete)
  const lowConfSpec = new RequirementsSpec();
  lowConfSpec.addAnswer(new QuestionAnswer('q1', 'Question 1?', 'Answer 1', false));
  lowConfSpec.addAnswer(new QuestionAnswer('q2', 'Question 2?', 'Other', false));
  lowConfSpec.addAnswer(new QuestionAnswer('q3', 'Question 3?', 'Other', false));

  assert.strictEqual(lowConfSpec.getConfidenceLevel(), 'low');

  console.log('✓ Confidence level tests passed');
}

/**
 * Test markdown generation
 */
async function testMarkdownGeneration() {
  console.log('Testing markdown generation...');

  const spec = new RequirementsSpec();

  spec.addAnswer(new QuestionAnswer('purpose', 'What is the purpose?', 'New feature', false));
  spec.addAnswer(new QuestionAnswer('stack', 'What stack?', 'React/Next.js', false));
  spec.addAnswer(new QuestionAnswer('testing', 'Testing?', ['Unit tests', 'E2E tests'], true));

  spec.synthesize();

  const markdown = spec.toMarkdown();

  assert.ok(markdown.includes('# Requirements Specification'));
  assert.ok(markdown.includes('## Project Scope'));
  assert.ok(markdown.includes('## Technical Decisions'));
  assert.ok(markdown.includes('New feature'));
  assert.ok(markdown.includes('React/Next.js'));

  console.log('✓ Markdown generation tests passed');
}

/**
 * Test JSON export
 */
async function testJSONExport() {
  console.log('Testing JSON export...');

  const spec = new RequirementsSpec();

  spec.addAnswer(new QuestionAnswer('purpose', 'What is the purpose?', 'New feature', false));
  spec.addAnswer(new QuestionAnswer('stack', 'What stack?', 'React/Next.js', false));

  spec.synthesize();

  const json = spec.toJSON();

  assert.ok(json.projectScope);
  assert.ok(json.technicalDecisions);
  assert.ok(json.featureList);
  assert.strictEqual(json.totalQuestions, 2);
  assert.strictEqual(json.completeAnswers, 2);

  console.log('✓ JSON export tests passed');
}

/**
 * Test missing information tracking
 */
async function testMissingInformationTracking() {
  console.log('Testing missing information tracking...');

  const spec = new RequirementsSpec();

  // Add complete answer
  spec.addAnswer(new QuestionAnswer('q1', 'Complete question?', 'Answer', false));

  // Add incomplete answer (Other selected)
  spec.addAnswer(new QuestionAnswer('q2', 'Incomplete question?', 'Other', false));

  spec.synthesize();

  assert.strictEqual(spec.missingInformation.length, 1);
  assert.strictEqual(spec.missingInformation[0].question, 'Incomplete question?');

  console.log('✓ Missing information tracking tests passed');
}

/**
 * Run all tests
 */
async function runTests() {
  console.log('\n=== Running Requirements Collector Tests ===\n');

  try {
    // Synchronous tests
    testQuestionAnswer();
    testRequirementsSpec();
    testProjectScopeExtraction();
    testTechnicalDecisionsExtraction();
    testFeatureListExtraction();
    testQualityRequirementsExtraction();
    testConstraintsExtraction();
    testConfidenceLevel();

    // Async tests
    await testMarkdownGeneration();
    await testJSONExport();
    await testMissingInformationTracking();

    console.log('\n✓✓✓ All tests passed! ✓✓✓\n');
    return true;

  } catch (error) {
    console.error('\n✗✗✗ Test failed:', error.message);
    console.error(error.stack);
    return false;
  }
}

// Run tests if executed directly
if (require.main === module) {
  runTests().then(success => {
    process.exit(success ? 0 : 1);
  });
}

module.exports = { runTests };
