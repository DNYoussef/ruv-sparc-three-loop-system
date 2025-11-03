#!/usr/bin/env node
/**
 * GitHub PR Review Comment Generator
 *
 * Generates structured, actionable review comments based on code analysis.
 * Supports multiple comment types (security, performance, style, etc.).
 *
 * Usage:
 *   node comment-generator.js --type security --file auth.js --line 42 --issue "SQL injection vulnerability"
 *   node comment-generator.js --type performance --file api.js --line 100 --issue "N+1 query detected"
 */

const args = process.argv.slice(2);

if (args.includes('--help') || args.length === 0) {
  console.log('Usage: node comment-generator.js [options]');
  console.log('');
  console.log('Required Options:');
  console.log('  --type <category>      Comment category (security|performance|style|architecture)');
  console.log('  --file <path>          File path');
  console.log('  --line <number>        Line number');
  console.log('  --issue <description>  Issue description');
  console.log('');
  console.log('Optional:');
  console.log('  --severity <level>     Severity (critical|high|medium|low)');
  console.log('  --suggestion <text>    Suggested fix');
  console.log('  --code <snippet>       Code example for fix');
  console.log('  --references <urls>    Comma-separated reference URLs');
  console.log('  --json                 Output as JSON');
  process.exit(0);
}

/**
 * Parse command line arguments into object
 */
function parseArgs(args) {
  const parsed = {};
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace('--', '');
    const value = args[i + 1];
    parsed[key] = value;
  }
  return parsed;
}

const options = parseArgs(args);

// Validate required options
const required = ['type', 'file', 'line', 'issue'];
const missing = required.filter(opt => !options[opt]);
if (missing.length > 0) {
  console.error(`Error: Missing required options: ${missing.join(', ')}`);
  console.error('Use --help for usage information');
  process.exit(1);
}

/**
 * Comment templates by type
 */
const templates = {
  security: {
    emoji: '🔒',
    severityMap: {
      critical: '🔴 Critical',
      high: '🟠 High',
      medium: '🟡 Medium',
      low: '🟢 Low'
    },
    template: (data) => `
${data.emoji} **Security Issue: ${data.issue}**

**Severity**: ${data.severity}

**Description**:
${data.description || 'Security vulnerability detected that requires immediate attention.'}

**Impact**:
${data.impact || 'This vulnerability could be exploited to compromise system security.'}

**Suggested Fix**:
${data.suggestion || 'Review and apply security best practices.'}
${data.code ? `\n\`\`\`${data.language || 'javascript'}\n${data.code}\n\`\`\`` : ''}

${data.references ? `**References**:\n${data.references.split(',').map(r => `- ${r.trim()}`).join('\n')}` : ''}
`
  },
  performance: {
    emoji: '⚡',
    severityMap: {
      critical: '🔴 Critical Performance Issue',
      high: '🟠 Major Performance Impact',
      medium: '🟡 Performance Concern',
      low: '🟢 Minor Optimization'
    },
    template: (data) => `
${data.emoji} **Performance Issue: ${data.issue}**

**Impact**: ${data.severity}

**Analysis**:
${data.description || 'Performance bottleneck detected that may impact system responsiveness.'}

**Metrics**:
${data.metrics || '- Execution time impact: TBD\n- Memory impact: TBD\n- Scalability impact: TBD'}

**Optimization Strategy**:
${data.suggestion || 'Apply performance optimization techniques.'}
${data.code ? `\n\`\`\`${data.language || 'javascript'}\n${data.code}\n\`\`\`` : ''}

${data.references ? `**Best Practices**:\n${data.references.split(',').map(r => `- ${r.trim()}`).join('\n')}` : ''}
`
  },
  style: {
    emoji: '🎨',
    severityMap: {
      critical: '🔴 Style Violation',
      high: '🟠 Convention Mismatch',
      medium: '🟡 Style Improvement',
      low: '🟢 Formatting Suggestion'
    },
    template: (data) => `
${data.emoji} **Code Style: ${data.issue}**

**Category**: ${data.severity}

**Issue**:
${data.description || 'Code style does not follow project conventions.'}

**Suggestion**:
${data.suggestion || 'Apply consistent coding style.'}
${data.code ? `\n\`\`\`${data.language || 'javascript'}\n${data.code}\n\`\`\`` : ''}

${data.autoFix ? '**Auto-fix Available**: ✅ Run `npm run lint:fix` to automatically correct this issue.' : ''}

${data.references ? `**Style Guide**:\n${data.references.split(',').map(r => `- ${r.trim()}`).join('\n')}` : ''}
`
  },
  architecture: {
    emoji: '🏗️',
    severityMap: {
      critical: '🔴 Critical Design Flaw',
      high: '🟠 Major Architecture Issue',
      medium: '🟡 Design Improvement',
      low: '🟢 Architectural Suggestion'
    },
    template: (data) => `
${data.emoji} **Architecture Review: ${data.issue}**

**Impact**: ${data.severity}

**Analysis**:
${data.description || 'Architectural pattern detected that may impact maintainability.'}

**Design Principles**:
${data.principles || '- SOLID principles\n- Separation of concerns\n- DRY (Don\'t Repeat Yourself)'}

**Recommendation**:
${data.suggestion || 'Refactor to improve architectural quality.'}
${data.code ? `\n\`\`\`${data.language || 'javascript'}\n${data.code}\n\`\`\`` : ''}

${data.references ? `**Design Patterns**:\n${data.references.split(',').map(r => `- ${r.trim()}`).join('\n')}` : ''}
`
  },
  accessibility: {
    emoji: '♿',
    severityMap: {
      critical: '🔴 WCAG Violation',
      high: '🟠 A11y Blocker',
      medium: '🟡 A11y Improvement',
      low: '🟢 A11y Enhancement'
    },
    template: (data) => `
${data.emoji} **Accessibility Issue: ${data.issue}**

**WCAG Level**: ${data.severity}

**Issue**:
${data.description || 'Accessibility barrier detected that may prevent users from accessing content.'}

**Affected Users**:
${data.affectedUsers || '- Screen reader users\n- Keyboard-only users\n- Users with visual impairments'}

**Fix**:
${data.suggestion || 'Apply WCAG 2.1 accessibility guidelines.'}
${data.code ? `\n\`\`\`${data.language || 'html'}\n${data.code}\n\`\`\`` : ''}

${data.references ? `**WCAG Guidelines**:\n${data.references.split(',').map(r => `- ${r.trim()}`).join('\n')}` : ''}
`
  }
};

/**
 * Generate comment based on type and options
 */
function generateComment(options) {
  const template = templates[options.type];
  if (!template) {
    throw new Error(`Unknown comment type: ${options.type}`);
  }

  const data = {
    emoji: template.emoji,
    severity: template.severityMap[options.severity || 'medium'],
    issue: options.issue,
    description: options.description,
    suggestion: options.suggestion,
    code: options.code,
    language: options.language,
    references: options.references,
    impact: options.impact,
    metrics: options.metrics,
    principles: options.principles,
    autoFix: options.autoFix,
    affectedUsers: options.affectedUsers
  };

  const comment = {
    path: options.file,
    line: parseInt(options.line),
    body: template.template(data).trim(),
    type: options.type,
    severity: options.severity || 'medium'
  };

  return comment;
}

// Generate comment
try {
  const comment = generateComment(options);

  if (args.includes('--json')) {
    console.log(JSON.stringify(comment, null, 2));
  } else {
    console.log('\n--- Review Comment ---\n');
    console.log(`File: ${comment.path}:${comment.line}`);
    console.log(`Type: ${comment.type}`);
    console.log(`Severity: ${comment.severity}\n`);
    console.log(comment.body);
    console.log('\n--- End Comment ---\n');
  }
} catch (error) {
  console.error(`Error generating comment: ${error.message}`);
  process.exit(1);
}
