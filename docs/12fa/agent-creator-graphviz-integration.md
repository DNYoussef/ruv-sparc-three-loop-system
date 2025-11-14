# Agent Creator - Graphviz Integration Guide

**Version**: 1.0.0
**Status**: Implementation Guide
**Target**: agent-creator developers
**Last Updated**: 2025-11-01

---

## Overview

This guide describes how `agent-creator` automatically generates Graphviz `.dot` diagrams when creating new agents. Agents require more complex documentation than skills, typically including:

1. **Execution Workflow**: Single-agent process flow
2. **Coordination Diagram**: Multi-agent coordination (for coordinators)
3. **Decision Tree**: Agent decision logic

---

## Agent Type Detection

Based on agent type, generate appropriate diagrams:

```javascript
const diagramsNeeded = {
  'single': ['execution'],
  'coordinator': ['execution', 'coordination'],
  'worker': ['execution', 'decision'],
  'specialized': ['execution', 'decision', 'coordination']
};

function getDiagramTypes(agentType) {
  return diagramsNeeded[agentType] || ['execution'];
}
```

---

## Execution Workflow Generation

### Template Selection

```javascript
async function generateExecutionDiagram(agentName, responsibilities) {
  const template = await fs.readFile('templates/agent-execution.dot.template', 'utf8');

  let diagram = template.replace(/AGENT_NAME/g, agentName.toUpperCase());

  // Extract responsibilities and map to execution steps
  const steps = parseResponsibilities(responsibilities);

  // Generate execution nodes
  const executionNodes = steps.map((step, i) => {
    return `    exec_step_${i + 1} [label="${step}", fillcolor=purple];`;
  }).join('\n');

  // Generate connections
  const connections = steps.slice(0, -1).map((_, i) => {
    return `    exec_step_${i + 1} -> exec_step_${i + 2};`;
  }).join('\n');

  diagram = diagram.replace(
    /\/\/ EXECUTION_STEPS_PLACEHOLDER/,
    executionNodes + '\n' + connections
  );

  return diagram;
}
```

---

## Coordination Diagram Generation

### For Coordinator Agents

```javascript
async function generateCoordinationDiagram(agentName, coordinationSpec) {
  const template = await fs.readFile('templates/agent-coordination.dot.template', 'utf8');

  let diagram = template.replace(/AGENT_COORDINATION/g, `${agentName.toUpperCase()}_COORDINATION`);

  // Determine topology
  const topology = coordinationSpec.topology || 'mesh';

  // Generate worker nodes
  const workers = coordinationSpec.workers || [];
  const workerNodes = workers.map((worker, i) => {
    return `    worker_${i + 1} [label="${worker.type} Agent\\n(${worker.role})", fillcolor=yellow];`;
  }).join('\n');

  // Generate coordination edges based on topology
  let coordinationEdges = '';
  if (topology === 'mesh') {
    // All-to-all connections
    for (let i = 0; i < workers.length; i++) {
      for (let j = i + 1; j < workers.length; j++) {
        coordinationEdges += `    worker_${i + 1} -> worker_${j + 1} [dir=both, label="Sync"];\n`;
      }
    }
  } else if (topology === 'hierarchical') {
    // Tree structure
    coordinationEdges = `    coordinator -> worker_1;\n`;
    coordinationEdges += `    coordinator -> worker_2;\n`;
    // ...
  }

  diagram = diagram.replace(/\/\/ WORKER_NODES_PLACEHOLDER/, workerNodes);
  diagram = diagram.replace(/\/\/ COORDINATION_EDGES_PLACEHOLDER/, coordinationEdges);

  return diagram;
}
```

---

## Decision Tree Generation

### Extract Decision Logic

```javascript
async function generateDecisionDiagram(agentName, decisionSpec) {
  const template = await fs.readFile('templates/decision-tree.dot.template', 'utf8');

  let diagram = template.replace(/DECISION_TREE/g, `${agentName.toUpperCase()}_DECISIONS`);

  // Parse decision conditions
  const decisions = extractDecisions(decisionSpec);

  // Generate decision nodes
  const decisionNodes = decisions.map((decision, i) => {
    return `    decision_${i + 1} [shape=diamond, label="${decision.condition}?", fillcolor=lightyellow];`;
  }).join('\n');

  // Generate action nodes
  const actionNodes = decisions.flatMap((decision, i) => {
    return [
      `    action_${i + 1}_yes [label="${decision.actionYes}", fillcolor=green];`,
      `    action_${i + 1}_no [label="${decision.actionNo}", fillcolor=orange];`
    ];
  }).join('\n');

  // Generate edges
  const edges = decisions.map((_, i) => {
    return `    decision_${i + 1} -> action_${i + 1}_yes [label="Yes"];\n` +
           `    decision_${i + 1} -> action_${i + 1}_no [label="No"];`;
  }).join('\n');

  diagram = diagram.replace(/\/\/ DECISION_NODES_PLACEHOLDER/, decisionNodes);
  diagram = diagram.replace(/\/\/ ACTION_NODES_PLACEHOLDER/, actionNodes);
  diagram = diagram.replace(/\/\/ EDGES_PLACEHOLDER/, edges);

  return diagram;
}

function extractDecisions(spec) {
  // Parse specification for conditional logic
  // Example: "If input valid, proceed; otherwise, return error"
  const decisions = [];

  const patterns = [
    /if\s+(.+?),\s+(.+?);?\s+otherwise,?\s+(.+)/gi,
    /when\s+(.+?),\s+(.+?);?\s+else,?\s+(.+)/gi
  ];

  patterns.forEach(pattern => {
    let match;
    while ((match = pattern.exec(spec)) !== null) {
      decisions.push({
        condition: match[1].trim(),
        actionYes: match[2].trim(),
        actionNo: match[3].trim()
      });
    }
  });

  return decisions;
}
```

---

## Complete agent.yaml Integration

```javascript
async function createAgentWithDiagrams(agentName, agentSpec) {
  const agentPath = path.join('agents', agentName);
  await fs.mkdir(agentPath, { recursive: true });

  // Determine agent type
  const agentType = agentSpec.type || 'single';
  const diagramTypes = getDiagramTypes(agentType);

  // Generate diagrams
  const diagrams = [];

  if (diagramTypes.includes('execution')) {
    const executionDiagram = await generateExecutionDiagram(agentName, agentSpec.responsibilities);
    const executionPath = path.join(agentPath, `${agentName}-execution.dot`);
    await fs.writeFile(executionPath, executionDiagram, 'utf8');

    diagrams.push({
      name: 'execution_process',
      path: `./${path.basename(executionPath)}`,
      type: 'workflow',
      description: 'Agent execution workflow'
    });
  }

  if (diagramTypes.includes('coordination')) {
    const coordinationDiagram = await generateCoordinationDiagram(agentName, agentSpec.coordination);
    const coordinationPath = path.join(agentPath, `${agentName}-coordination.dot`);
    await fs.writeFile(coordinationPath, coordinationDiagram, 'utf8');

    diagrams.push({
      name: 'coordination_flow',
      path: `./${path.basename(coordinationPath)}`,
      type: 'coordination',
      description: 'Multi-agent coordination pattern'
    });
  }

  if (diagramTypes.includes('decision')) {
    const decisionDiagram = await generateDecisionDiagram(agentName, agentSpec.decisions);
    const decisionPath = path.join(agentPath, `${agentName}-decisions.dot`);
    await fs.writeFile(decisionPath, decisionDiagram, 'utf8');

    diagrams.push({
      name: 'decision_tree',
      path: `./${path.basename(decisionPath)}`,
      type: 'decision_tree',
      description: 'Agent decision logic'
    });
  }

  // Generate agent.yaml
  const agentYaml = {
    name: agentName,
    version: agentSpec.version || '1.0.0',
    type: agentType,
    documentation: {
      markdown: `./agent.md`,
      process_diagrams: diagrams.map(d => ({
        ...d,
        version: agentSpec.version || '1.0.0',
        last_updated: new Date().toISOString().split('T')[0]
      }))
    },
    visualization: {
      enabled: true,
      auto_generate: true,
      formats: ['svg', 'dot'],
      output_dir: './docs/diagrams',
      render_on_build: false,
      validate_syntax: true
    },
    capabilities: agentSpec.capabilities || [],
    twelve_factor: agentSpec.twelve_factor || {},
    policies: agentSpec.policies || {},
    commands: agentSpec.commands || [],
    mcp_tools: agentSpec.mcp_tools || []
  };

  // Save agent.yaml
  await fs.writeFile(
    path.join(agentPath, 'agent.yaml'),
    yaml.stringify(agentYaml),
    'utf8'
  );

  // Render diagrams
  for (const diagram of diagrams) {
    const dotPath = path.join(agentPath, path.basename(diagram.path));
    await renderDiagram(dotPath, path.join(agentPath, 'docs/diagrams'));
  }

  console.log(`✅ Agent created with ${diagrams.length} workflow diagrams`);

  return { agentPath, diagrams };
}
```

---

## CLI Integration

### Interactive Prompts

```javascript
async function promptForCoordinationDetails() {
  console.log('\n📊 This agent is a coordinator. Let\'s define coordination:');

  const topology = await prompt({
    type: 'select',
    name: 'topology',
    message: 'Select coordination topology:',
    choices: [
      { title: 'Mesh (Peer-to-Peer)', value: 'mesh' },
      { title: 'Hierarchical (Tree)', value: 'hierarchical' },
      { title: 'Ring (Circular)', value: 'ring' },
      { title: 'Star (Centralized)', value: 'star' }
    ]
  });

  const workerCount = await prompt({
    type: 'number',
    name: 'workerCount',
    message: 'How many worker agents?',
    initial: 3,
    min: 2,
    max: 10
  });

  const workers = [];
  for (let i = 0; i < workerCount; i++) {
    const workerType = await prompt({
      type: 'text',
      name: 'type',
      message: `Worker ${i + 1} type (e.g., researcher, coder):`,
      initial: 'worker'
    });

    workers.push({ type: workerType, role: `worker_${i + 1}` });
  }

  return { topology, workers };
}
```

---

## Validation & Testing

### Syntax Validation

```javascript
async function validateAllDiagrams(agentPath) {
  const dotFiles = await glob(`${agentPath}/*.dot`);
  const results = [];

  for (const dotFile of dotFiles) {
    const content = await fs.readFile(dotFile, 'utf8');
    const validation = await validateDotSyntax(content);

    results.push({
      file: path.basename(dotFile),
      valid: validation.valid,
      error: validation.error
    });

    if (!validation.valid) {
      console.error(`❌ ${path.basename(dotFile)}: ${validation.error}`);
    }
  }

  return results;
}
```

---

## Best Practices

1. **Match Agent Complexity**: More complex agents → more diagrams
2. **Update on Changes**: Regenerate diagrams when responsibilities change
3. **Version Consistently**: Diagram version should match agent version
4. **Validate Before Save**: Always check syntax before committing
5. **Render for Review**: Show rendered SVGs to user
6. **Store Metadata**: Track diagram generation settings in agent.yaml

---

## Summary

Agent creation automatically generates:
- ✅ Execution workflow diagram (all agents)
- ✅ Coordination pattern diagram (coordinators)
- ✅ Decision tree diagram (complex logic agents)
- ✅ Updated agent.yaml with references
- ✅ Rendered SVG previews

This ensures agents are self-documenting from creation.

---

**References**:
- [Graphviz Process Documentation Guide](./graphviz-process-documentation.md)
- [Agent Coordination Template](../../templates/agent-coordination.dot.template)
- [Agent Manifest Schema](../../schemas/agent-manifest-v1-graphviz.json)
