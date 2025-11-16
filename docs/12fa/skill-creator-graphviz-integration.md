# Skill Creator - Graphviz Integration Guide

**Version**: 1.0.0
**Status**: Implementation Guide
**Target**: skill-creator developers
**Last Updated**: 2025-11-01

---

## Overview

This guide describes how `skill-creator` automatically generates Graphviz `.dot` diagrams when creating new skills. This ensures every skill has visual process documentation that AI agents can parse unambiguously.

---

## Integration Goals

1. **Automatic Generation**: Every skill creation generates at least one `.dot` file
2. **Template-Based**: Use workflow type to select appropriate template
3. **Customization**: Populate diagram with skill-specific steps
4. **Validation**: Ensure syntactically correct diagrams
5. **Integration**: Reference diagrams in `skill.yaml`

---

## Workflow Integration Points

### 1. Skill Creation Initialization

When `skill-creator` starts, prompt user for workflow type:

```javascript
// During skill creation prompt sequence
const workflowType = await prompt({
  type: 'select',
  name: 'workflowType',
  message: 'Select skill workflow type:',
  choices: [
    { title: 'Linear (Sequential steps)', value: 'linear' },
    { title: 'Branching (Conditional logic)', value: 'branching' },
    { title: 'Cyclical (Iterative processes)', value: 'cyclical' },
    { title: 'Parallel (Concurrent operations)', value: 'parallel' },
    { title: 'Custom (Manual diagram creation)', value: 'custom' }
  ]
});
```

### 2. Template Selection

Based on workflow type, select appropriate `.dot` template:

```javascript
const templateMap = {
  linear: 'templates/skill-process-linear.dot.template',
  branching: 'templates/skill-process-branching.dot.template',
  cyclical: 'templates/skill-process-cyclical.dot.template',
  parallel: 'templates/skill-process-parallel.dot.template',
  custom: 'templates/skill-process.dot.template' // Full template
};

const templatePath = templateMap[workflowType];
const dotTemplate = await fs.readFile(templatePath, 'utf8');
```

### 3. Step Extraction from Specification

Parse skill specification to extract process steps:

```javascript
// Extract steps from skill description
function extractSkillSteps(specification) {
  const steps = [];

  // Parse from user input or specification
  // Example: "First, validate input. Then, process data. Finally, return result."

  const sentences = specification.split(/\.\s+/);

  sentences.forEach((sentence, index) => {
    // Identify action verbs (validate, process, transform, etc.)
    const actionMatch = sentence.match(/^(First|Then|Next|Finally),?\s+(.+)/i);

    if (actionMatch) {
      steps.push({
        id: `step_${index + 1}`,
        label: actionMatch[2],
        order: index + 1
      });
    }
  });

  return steps;
}
```

### 4. Diagram Generation

Populate template with extracted steps:

```javascript
function generateSkillDiagram(template, skillName, steps, workflowType) {
  let diagram = template;

  // Replace skill name placeholder
  diagram = diagram.replace(/SKILL_NAME_WORKFLOW/g, `${skillName.toUpperCase()}_WORKFLOW`);

  // Generate step nodes
  const stepNodes = steps.map((step, index) => {
    return `    ${step.id} [label="${step.label}", fillcolor=purple];`;
  }).join('\n');

  // Generate step connections based on workflow type
  const stepConnections = generateStepConnections(steps, workflowType);

  // Insert into execution cluster
  diagram = diagram.replace(
    /process_step_1.*process_step_3/s,
    stepNodes + '\n' + stepConnections
  );

  return diagram;
}

function generateStepConnections(steps, workflowType) {
  if (workflowType === 'linear') {
    // Sequential connections
    return steps.slice(0, -1).map((step, index) => {
      return `    ${step.id} -> ${steps[index + 1].id};`;
    }).join('\n');
  } else if (workflowType === 'branching') {
    // Add decision nodes
    return steps.map((step, index) => {
      if (index < steps.length - 1) {
        return `    ${step.id} -> decision_${index};\n` +
               `    decision_${index} [shape=diamond, label="Success?"];\n` +
               `    decision_${index} -> ${steps[index + 1].id} [label="Yes"];`;
      }
      return '';
    }).join('\n');
  }
  // Additional workflow types...
}
```

### 5. Validation

Validate generated `.dot` file syntax:

```javascript
async function validateDotSyntax(dotContent) {
  const { execSync } = require('child_process');

  try {
    // Use Graphviz's dot command to validate
    execSync('dot -Tsvg', {
      input: dotContent,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    return { valid: true };
  } catch (error) {
    return {
      valid: false,
      error: error.message,
      stderr: error.stderr.toString()
    };
  }
}
```

### 6. File System Integration

Save diagram alongside skill files:

```javascript
async function saveSkillDiagram(skillPath, skillName, diagram) {
  const diagramPath = path.join(skillPath, `${skillName}-process.dot`);

  // Ensure not in root directory
  if (path.dirname(diagramPath) === process.cwd()) {
    throw new Error('BLOCKED: Rule #1 - No files in root directory');
  }

  await fs.writeFile(diagramPath, diagram, 'utf8');

  return diagramPath;
}
```

### 7. skill.yaml Integration

Add reference to generated diagram in `skill.yaml`:

```javascript
function updateSkillYaml(skillYaml, diagramPath) {
  skillYaml.documentation = skillYaml.documentation || {};
  skillYaml.documentation.process_diagrams = skillYaml.documentation.process_diagrams || [];

  skillYaml.documentation.process_diagrams.push({
    name: 'workflow',
    path: `./${path.basename(diagramPath)}`,
    type: 'workflow',
    description: 'Skill execution workflow',
    version: skillYaml.version,
    last_updated: new Date().toISOString().split('T')[0]
  });

  skillYaml.visualization = {
    enabled: true,
    auto_generate: true,
    formats: ['svg', 'dot'],
    output_dir: './docs/diagrams',
    validate_syntax: true
  };

  return skillYaml;
}
```

---

## Complete Integration Example

```javascript
// skill-creator main workflow
async function createSkill(skillName, specification) {
  // 1. Prompt for workflow type
  const workflowType = await promptWorkflowType();

  // 2. Extract steps from specification
  const steps = extractSkillSteps(specification);

  // 3. Load template
  const template = await loadDotTemplate(workflowType);

  // 4. Generate diagram
  const diagram = generateSkillDiagram(template, skillName, steps, workflowType);

  // 5. Validate syntax
  const validation = await validateDotSyntax(diagram);
  if (!validation.valid) {
    throw new Error(`Invalid .dot syntax: ${validation.error}`);
  }

  // 6. Create skill directory
  const skillPath = path.join('skills', skillName);
  await fs.mkdir(skillPath, { recursive: true });

  // 7. Save diagram
  const diagramPath = await saveSkillDiagram(skillPath, skillName, diagram);

  // 8. Generate skill.yaml
  let skillYaml = generateSkillYaml(skillName, specification);
  skillYaml = updateSkillYaml(skillYaml, diagramPath);

  // 9. Save skill.yaml
  await fs.writeFile(
    path.join(skillPath, 'skill.yaml'),
    yaml.stringify(skillYaml),
    'utf8'
  );

  // 10. Generate rendered diagram (optional)
  if (skillYaml.visualization.enabled) {
    await renderDiagram(diagramPath, skillPath);
  }

  console.log(`✅ Skill created with workflow diagram: ${diagramPath}`);
}
```

---

## Workflow Type Templates

### Linear Workflow

**Use Case**: Sequential steps without branching

**Template Structure**:
```dot
step_1 -> step_2 -> step_3 -> step_4 -> success
```

**Example**: Data transformation pipeline, file processing

### Branching Workflow

**Use Case**: Conditional logic with multiple paths

**Template Structure**:
```dot
step_1 -> decision_1
decision_1 -> step_2a [label="Path A"]
decision_1 -> step_2b [label="Path B"]
```

**Example**: Input validation with error handling, multi-path processing

### Cyclical Workflow

**Use Case**: Iterative processes with retry logic

**Template Structure**:
```dot
step_1 -> step_2 -> check_condition
check_condition -> step_1 [label="Retry"]
check_condition -> success [label="Complete"]
```

**Example**: Polling, batch processing with retries

### Parallel Workflow

**Use Case**: Concurrent operations with synchronization

**Template Structure**:
```dot
coordinator -> step_1
coordinator -> step_2
coordinator -> step_3
step_1 -> sync
step_2 -> sync
step_3 -> sync
sync -> success
```

**Example**: Multi-agent coordination, parallel data processing

---

## Post-Generation Tasks

### Render SVG Preview

```javascript
async function renderDiagram(dotPath, outputDir) {
  const { execSync } = require('child_process');
  const outputPath = dotPath.replace('.dot', '.svg');

  execSync(`dot -Tsvg "${dotPath}" -o "${outputPath}"`);

  console.log(`📊 Rendered diagram: ${outputPath}`);
}
```

### Display to User

```javascript
async function displayDiagramPreview(svgPath) {
  console.log(`\n📊 Workflow Diagram Generated:`);
  console.log(`   View: ${svgPath}`);
  console.log(`   Edit: ${svgPath.replace('.svg', '.dot')}`);
  console.log(`\n💡 Tip: Iterate ~12 times to perfect the flow\n`);
}
```

---

## User Interaction Flow

1. **Prompt for Workflow Type**
   - Display clear examples for each type
   - Provide visual previews if possible

2. **Review Generated Diagram**
   - Show rendered SVG or ASCII representation
   - Allow user to approve or regenerate

3. **Iterative Refinement**
   - Offer to add decision nodes
   - Offer to add error handling paths
   - Offer to add quality gates

4. **Finalization**
   - Validate syntax
   - Save to skill directory
   - Update skill.yaml
   - Display success summary

---

## Error Handling

### Invalid Syntax

```javascript
if (!validation.valid) {
  console.error('❌ Generated .dot file has syntax errors:');
  console.error(validation.stderr);

  // Offer to fix or use default template
  const action = await prompt({
    type: 'select',
    message: 'How would you like to proceed?',
    choices: [
      { title: 'Use default template', value: 'default' },
      { title: 'Edit manually', value: 'edit' },
      { title: 'Skip diagram generation', value: 'skip' }
    ]
  });

  // Handle action...
}
```

### Missing Graphviz Installation

```javascript
function checkGraphvizInstalled() {
  try {
    execSync('dot -V', { stdio: 'ignore' });
    return true;
  } catch (error) {
    console.warn('⚠️  Graphviz not installed. Diagrams will not be rendered.');
    console.warn('   Install: https://graphviz.org/download/');
    return false;
  }
}
```

---

## Testing Integration

### Unit Tests

```javascript
describe('Graphviz Integration', () => {
  test('generates valid .dot file for linear workflow', async () => {
    const diagram = generateSkillDiagram(template, 'test-skill', steps, 'linear');
    const validation = await validateDotSyntax(diagram);
    expect(validation.valid).toBe(true);
  });

  test('extracts steps from specification', () => {
    const spec = 'First, validate input. Then, process data. Finally, return result.';
    const steps = extractSkillSteps(spec);
    expect(steps).toHaveLength(3);
  });

  test('updates skill.yaml with diagram reference', () => {
    const skillYaml = { name: 'test', version: '1.0.0' };
    const updated = updateSkillYaml(skillYaml, './test-process.dot');
    expect(updated.documentation.process_diagrams).toHaveLength(1);
  });
});
```

---

## Best Practices

1. **Always Validate**: Run syntax validation before saving
2. **Version Diagrams**: Update diagram version in skill.yaml on changes
3. **Provide Templates**: Maintain library of workflow templates
4. **Enable Iteration**: Allow users to regenerate diagrams easily
5. **Document Conventions**: Include legend in generated diagrams
6. **Check Dependencies**: Verify Graphviz installation on first run
7. **Store in Subdirectories**: Never save diagrams to root

---

## Future Enhancements

1. **Interactive Diagram Builder**: Visual editor for .dot files
2. **AI-Assisted Generation**: Use LLM to generate diagrams from natural language
3. **Diagram Diffing**: Show visual diffs when updating workflows
4. **Template Library**: Community-contributed workflow templates
5. **Validation Rules**: Custom validation for project-specific conventions

---

## Summary

By integrating Graphviz generation into `skill-creator`, we ensure:

- ✅ Every skill has visual documentation
- ✅ AI agents can parse workflows unambiguously
- ✅ Developers see process flows immediately
- ✅ Documentation stays in sync with implementation
- ✅ Workflows are self-documenting

This reduces ambiguity and improves both human and AI comprehension of skill behavior.

---

**Next Steps**:
1. Implement workflow type prompts
2. Create workflow-specific templates
3. Add syntax validation
4. Integrate into skill.yaml generation
5. Add unit tests for diagram generation

---

**References**:
- [Graphviz Process Documentation Guide](./graphviz-process-documentation.md)
- [Skill Process Template](../../templates/skill-process.dot.template)
- [Agent Manifest Schema](../../schemas/agent-manifest-v1-graphviz.json)
