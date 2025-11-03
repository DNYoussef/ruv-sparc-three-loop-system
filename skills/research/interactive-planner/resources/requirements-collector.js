#!/usr/bin/env node

/**
 * Requirements Collector - Interactive Requirements Gathering
 *
 * Collects and validates user responses from AskUserQuestion tool,
 * synthesizes requirements into actionable specifications,
 * and identifies gaps for follow-up questions.
 *
 * Usage:
 *   node requirements-collector.js --answers answers.json --output spec.md
 */

const fs = require('fs');
const path = require('path');
const { promisify } = require('util');
const readFile = promisify(fs.readFile);
const writeFile = promisify(fs.writeFile);

/**
 * Question answer structure from AskUserQuestion tool
 */
class QuestionAnswer {
  constructor(questionId, question, selectedOptions, isMultiSelect = false) {
    this.questionId = questionId;
    this.question = question;
    this.selectedOptions = Array.isArray(selectedOptions) ? selectedOptions : [selectedOptions];
    this.isMultiSelect = isMultiSelect;
    this.timestamp = new Date().toISOString();
  }

  /**
   * Check if "Other" was selected (requires follow-up)
   */
  hasOtherSelected() {
    return this.selectedOptions.some(opt =>
      opt.toLowerCase() === 'other' || opt.toLowerCase().includes('other')
    );
  }

  /**
   * Validate answer completeness
   */
  isComplete() {
    return this.selectedOptions.length > 0 && !this.hasOtherSelected();
  }
}

/**
 * Requirements specification document builder
 */
class RequirementsSpec {
  constructor() {
    this.projectScope = {};
    this.technicalDecisions = {};
    this.featureList = [];
    this.constraints = [];
    this.qualityRequirements = {};
    this.missingInformation = [];
    this.answers = [];
  }

  /**
   * Add question answer to specification
   */
  addAnswer(answer) {
    this.answers.push(answer);

    // Track incomplete answers
    if (!answer.isComplete()) {
      this.missingInformation.push({
        question: answer.question,
        reason: answer.hasOtherSelected() ? 'Other selected - needs clarification' : 'No answer provided'
      });
    }
  }

  /**
   * Synthesize answers into structured requirements
   */
  synthesize() {
    // Extract project scope
    this.projectScope = this._extractProjectScope();

    // Extract technical decisions
    this.technicalDecisions = this._extractTechnicalDecisions();

    // Extract feature list
    this.featureList = this._extractFeatures();

    // Extract constraints
    this.constraints = this._extractConstraints();

    // Extract quality requirements
    this.qualityRequirements = this._extractQualityRequirements();
  }

  /**
   * Extract project scope from answers
   */
  _extractProjectScope() {
    const scope = {};

    // Find purpose/goal answers
    const purposeAnswers = this.answers.filter(a =>
      a.question.toLowerCase().includes('purpose') ||
      a.question.toLowerCase().includes('goal')
    );

    if (purposeAnswers.length > 0) {
      scope.purpose = purposeAnswers[0].selectedOptions[0];
    }

    // Find project type answers
    const typeAnswers = this.answers.filter(a =>
      a.question.toLowerCase().includes('project type') ||
      a.question.toLowerCase().includes('application type')
    );

    if (typeAnswers.length > 0) {
      scope.projectType = typeAnswers[0].selectedOptions[0];
    }

    // Find complexity answers
    const complexityAnswers = this.answers.filter(a =>
      a.question.toLowerCase().includes('complexity') ||
      a.question.toLowerCase().includes('scale')
    );

    if (complexityAnswers.length > 0) {
      scope.complexity = complexityAnswers[0].selectedOptions[0];
    }

    return scope;
  }

  /**
   * Extract technical decisions from answers
   */
  _extractTechnicalDecisions() {
    const decisions = {};

    // Framework/stack
    const stackAnswers = this.answers.filter(a =>
      a.question.toLowerCase().includes('framework') ||
      a.question.toLowerCase().includes('stack') ||
      a.question.toLowerCase().includes('technology')
    );

    if (stackAnswers.length > 0) {
      decisions.stack = stackAnswers[0].selectedOptions[0];
    }

    // Database
    const dbAnswers = this.answers.filter(a =>
      a.question.toLowerCase().includes('database') ||
      a.question.toLowerCase().includes('storage')
    );

    if (dbAnswers.length > 0) {
      decisions.database = dbAnswers[0].selectedOptions[0];
    }

    // Authentication
    const authAnswers = this.answers.filter(a =>
      a.question.toLowerCase().includes('authentication') ||
      a.question.toLowerCase().includes('auth')
    );

    if (authAnswers.length > 0) {
      decisions.authentication = authAnswers[0].selectedOptions[0];
    }

    // Backend patterns (may be multi-select)
    const backendAnswers = this.answers.filter(a =>
      a.question.toLowerCase().includes('backend') ||
      a.question.toLowerCase().includes('api')
    );

    if (backendAnswers.length > 0) {
      decisions.backendPatterns = backendAnswers[0].selectedOptions;
    }

    return decisions;
  }

  /**
   * Extract feature list from answers
   */
  _extractFeatures() {
    const features = [];

    // Find feature-related multi-select answers
    const featureAnswers = this.answers.filter(a =>
      (a.question.toLowerCase().includes('feature') ||
       a.question.toLowerCase().includes('functionality')) &&
      a.isMultiSelect
    );

    featureAnswers.forEach(answer => {
      features.push(...answer.selectedOptions.filter(opt => opt !== 'Other'));
    });

    return [...new Set(features)]; // Remove duplicates
  }

  /**
   * Extract constraints from answers
   */
  _extractConstraints() {
    const constraints = [];

    // Timeline constraints
    const timelineAnswers = this.answers.filter(a =>
      a.question.toLowerCase().includes('timeline') ||
      a.question.toLowerCase().includes('deadline')
    );

    if (timelineAnswers.length > 0) {
      constraints.push(`Timeline: ${timelineAnswers[0].selectedOptions[0]}`);
    }

    // Budget/resource constraints
    const budgetAnswers = this.answers.filter(a =>
      a.question.toLowerCase().includes('budget') ||
      a.question.toLowerCase().includes('resource')
    );

    if (budgetAnswers.length > 0) {
      constraints.push(`Budget: ${budgetAnswers[0].selectedOptions[0]}`);
    }

    return constraints;
  }

  /**
   * Extract quality requirements from answers
   */
  _extractQualityRequirements() {
    const quality = {};

    // Testing requirements
    const testAnswers = this.answers.filter(a =>
      a.question.toLowerCase().includes('test') ||
      a.question.toLowerCase().includes('coverage')
    );

    if (testAnswers.length > 0) {
      quality.testing = testAnswers[0].selectedOptions;
    }

    // Quality level
    const qualityAnswers = this.answers.filter(a =>
      a.question.toLowerCase().includes('quality level') ||
      a.question.toLowerCase().includes('quality standard')
    );

    if (qualityAnswers.length > 0) {
      quality.level = qualityAnswers[0].selectedOptions[0];
    }

    // Performance requirements
    const perfAnswers = this.answers.filter(a =>
      a.question.toLowerCase().includes('performance') ||
      a.question.toLowerCase().includes('speed')
    );

    if (perfAnswers.length > 0) {
      quality.performance = perfAnswers[0].selectedOptions[0];
    }

    return quality;
  }

  /**
   * Calculate confidence level based on completeness
   */
  getConfidenceLevel() {
    const totalAnswers = this.answers.length;
    const completeAnswers = this.answers.filter(a => a.isComplete()).length;

    if (totalAnswers === 0) return 'low';

    const completeness = completeAnswers / totalAnswers;

    if (completeness >= 0.9) return 'high';
    if (completeness >= 0.7) return 'medium';
    return 'low';
  }

  /**
   * Generate markdown specification document
   */
  toMarkdown() {
    let markdown = '# Requirements Specification\n\n';
    markdown += `**Generated**: ${new Date().toISOString()}\n`;
    markdown += `**Confidence Level**: ${this.getConfidenceLevel()}\n\n`;

    // Project Scope
    markdown += '## Project Scope\n\n';
    if (this.projectScope.purpose) {
      markdown += `**Purpose**: ${this.projectScope.purpose}\n\n`;
    }
    if (this.projectScope.projectType) {
      markdown += `**Project Type**: ${this.projectScope.projectType}\n\n`;
    }
    if (this.projectScope.complexity) {
      markdown += `**Complexity**: ${this.projectScope.complexity}\n\n`;
    }

    // Technical Decisions
    markdown += '## Technical Decisions\n\n';
    Object.entries(this.technicalDecisions).forEach(([key, value]) => {
      const formattedKey = key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1');
      if (Array.isArray(value)) {
        markdown += `**${formattedKey}**:\n`;
        value.forEach(item => {
          markdown += `- ${item}\n`;
        });
        markdown += '\n';
      } else {
        markdown += `**${formattedKey}**: ${value}\n\n`;
      }
    });

    // Features
    if (this.featureList.length > 0) {
      markdown += '## Core Features\n\n';
      this.featureList.forEach(feature => {
        markdown += `- ${feature}\n`;
      });
      markdown += '\n';
    }

    // Quality Requirements
    markdown += '## Quality Requirements\n\n';
    if (this.qualityRequirements.level) {
      markdown += `**Quality Level**: ${this.qualityRequirements.level}\n\n`;
    }
    if (this.qualityRequirements.testing) {
      markdown += '**Testing Coverage**:\n';
      this.qualityRequirements.testing.forEach(test => {
        markdown += `- ${test}\n`;
      });
      markdown += '\n';
    }
    if (this.qualityRequirements.performance) {
      markdown += `**Performance**: ${this.qualityRequirements.performance}\n\n`;
    }

    // Constraints
    if (this.constraints.length > 0) {
      markdown += '## Constraints\n\n';
      this.constraints.forEach(constraint => {
        markdown += `- ${constraint}\n`;
      });
      markdown += '\n';
    }

    // Missing Information
    if (this.missingInformation.length > 0) {
      markdown += '## Missing Information (Follow-up Required)\n\n';
      this.missingInformation.forEach(missing => {
        markdown += `- **${missing.question}**: ${missing.reason}\n`;
      });
      markdown += '\n';
    }

    // Raw Answers Appendix
    markdown += '## Appendix: Raw Answers\n\n';
    this.answers.forEach((answer, idx) => {
      markdown += `### ${idx + 1}. ${answer.question}\n\n`;
      markdown += `**Selected**: ${answer.selectedOptions.join(', ')}\n\n`;
      markdown += `**Multi-select**: ${answer.isMultiSelect ? 'Yes' : 'No'}\n\n`;
    });

    return markdown;
  }

  /**
   * Export to JSON
   */
  toJSON() {
    return {
      projectScope: this.projectScope,
      technicalDecisions: this.technicalDecisions,
      featureList: this.featureList,
      constraints: this.constraints,
      qualityRequirements: this.qualityRequirements,
      missingInformation: this.missingInformation,
      confidenceLevel: this.getConfidenceLevel(),
      totalQuestions: this.answers.length,
      completeAnswers: this.answers.filter(a => a.isComplete()).length
    };
  }
}

/**
 * Parse command line arguments
 */
function parseArgs() {
  const args = process.argv.slice(2);
  const parsed = {
    answersFile: null,
    outputFile: null,
    format: 'markdown'
  };

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--answers' && i + 1 < args.length) {
      parsed.answersFile = args[i + 1];
      i++;
    } else if (args[i] === '--output' && i + 1 < args.length) {
      parsed.outputFile = args[i + 1];
      i++;
    } else if (args[i] === '--format' && i + 1 < args.length) {
      parsed.format = args[i + 1];
      i++;
    }
  }

  return parsed;
}

/**
 * Main execution
 */
async function main() {
  const args = parseArgs();

  if (!args.answersFile || !args.outputFile) {
    console.error('Usage: node requirements-collector.js --answers <file> --output <file> [--format markdown|json]');
    process.exit(1);
  }

  // Read answers file
  const answersData = JSON.parse(await readFile(args.answersFile, 'utf-8'));

  // Build requirements specification
  const spec = new RequirementsSpec();

  // Process answers
  if (Array.isArray(answersData.answers)) {
    answersData.answers.forEach(answerData => {
      const answer = new QuestionAnswer(
        answerData.questionId || answerData.question,
        answerData.question,
        answerData.selectedOptions || answerData.selected,
        answerData.isMultiSelect || false
      );
      spec.addAnswer(answer);
    });
  }

  // Synthesize requirements
  spec.synthesize();

  // Generate output
  let output;
  if (args.format === 'json') {
    output = JSON.stringify(spec.toJSON(), null, 2);
  } else {
    output = spec.toMarkdown();
  }

  // Write output file
  await writeFile(args.outputFile, output, 'utf-8');

  console.log(`Requirements specification written to ${args.outputFile}`);
  console.log(`Confidence level: ${spec.getConfidenceLevel()}`);
  console.log(`Total questions: ${spec.answers.length}`);
  console.log(`Complete answers: ${spec.answers.filter(a => a.isComplete()).length}`);
  console.log(`Missing information: ${spec.missingInformation.length} items`);
}

// Run if executed directly
if (require.main === module) {
  main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
  });
}

module.exports = { RequirementsSpec, QuestionAnswer };
