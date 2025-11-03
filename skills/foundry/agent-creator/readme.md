# Agent Creator - Silver Tier Documentation

## Overview

The **Agent Creator** skill provides the official comprehensive framework for creating specialized AI agents using the proven 4-phase SOP methodology from Desktop `.claude-flow`. This skill combines evidence-based prompting techniques, Claude Agent SDK implementation, and systematic domain analysis to produce production-ready agents with deeply embedded expertise.

## Quick Start

### Basic Usage

```bash
# Via Claude Code
"Create a new marketing specialist agent"

# Via Skill tool
Skill("agent-creator")
```

### Speed-Run (Experienced Users)

For users familiar with the methodology, create production-ready agents in **2 hours**:

1. **Combined Phase 1+2** (30 min): Domain analysis + specification
2. **Phase 3** (30 min): Base system prompt from template
3. **Phase 4** (45 min): Code patterns + failure modes
4. **Testing** (15 min): Quick validation suite

**Total**: 2 hours with templates

### First-Time Complete Process

For comprehensive agent creation with full documentation:

**Total Time**: 3.5-5.5 hours

1. **Phase 1: Domain Analysis** (30-60 min)
2. **Phase 2: Expertise Extraction** (30-45 min)
3. **Phase 3: Architecture Design** (45-60 min)
4. **Phase 4: Technical Enhancement** (60-90 min)
5. **SDK Implementation** (30-60 min)
6. **Testing & Validation** (30-45 min)
7. **Documentation** (15-30 min)

## Key Features

### 4-Phase Methodology

#### Phase 1: Initial Analysis & Intent Decoding
- Deep domain understanding through systematic research
- Technology stack mapping
- Integration point identification
- **Output**: Domain analysis document

#### Phase 2: Meta-Cognitive Extraction
- Expertise domain identification
- Agent specification creation
- Decision framework documentation
- **Output**: Complete agent specification

#### Phase 3: Agent Architecture Design
- System prompt structure design
- Evidence-based technique integration
- Quality standards & guardrails
- **Output**: Base system prompt v1.0

#### Phase 4: Deep Technical Enhancement
- Code pattern extraction
- Critical failure mode documentation
- MCP integration patterns
- Performance metrics definition
- **Output**: Enhanced system prompt v2.0

### Evidence-Based Prompting Techniques

Integrated throughout the methodology:

- **Self-Consistency Validation**: Multi-angle verification before deliverable finalization
- **Program-of-Thought Decomposition**: Complex task breakdown before execution
- **Plan-and-Solve Execution**: Standard workflow with validation at each step

### Claude Agent SDK Integration

Production-ready implementation in TypeScript and Python:

```typescript
// TypeScript implementation
import { query, tool } from '@anthropic-ai/claude-agent-sdk';

for await (const message of query('Task', {
  model: 'claude-sonnet-4-5',
  systemPrompt: enhancedPromptV2,
  permissionMode: 'acceptEdits',
  allowedTools: ['Read', 'Write', 'Bash'],
  mcpServers: [{ command: 'npx', args: ['claude-flow@alpha', 'mcp', 'start'] }]
})) {
  console.log(message);
}
```

### Agent Specialization Support

Built-in patterns for four agent types:

1. **Analytical Agents**: Evidence evaluation, data quality standards
2. **Generative Agents**: Quality criteria, template patterns, refinement
3. **Diagnostic Agents**: Problem patterns, debugging, hypothesis testing
4. **Orchestration Agents**: Workflow patterns, dependency management, coordination

## Examples

Comprehensive examples available in `examples/` directory:

- **[example-1-basic.md](examples/example-1-basic.md)**: Simple specialist agent creation (Marketing Specialist)
- **[example-2-coordinator.md](examples/example-2-coordinator.md)**: Multi-agent coordinator (DevOps Coordinator)
- **[example-3-hybrid.md](examples/example-3-hybrid.md)**: Hybrid multi-domain agent (Full-Stack Developer)

## Integration

### Claude Code Task Tool

Primary execution method for spawning agents:

```javascript
Task("Marketing Specialist", "Analyze market trends and create campaign strategy", "marketing-specialist")
```

### MCP Tools Coordination

For complex multi-agent workflows:

```javascript
// Setup coordination (optional)
mcp__claude-flow__swarm_init({ topology: "mesh", maxAgents: 6 })
mcp__claude-flow__agent_spawn({ type: "specialist" })

// Execute with Claude Code Task tool
Task("Specialist agent", "Complete domain-specific task", "specialist")
```

### Memory MCP Integration

Cross-session persistence and cross-agent data sharing:

```javascript
// Store results with auto-tagging
mcp__memory-mcp__memory_store({
  text: "Campaign analysis results: target audience identified...",
  metadata: {
    key: "marketing-specialist/campaign-123/audience-analysis",
    namespace: "agents/marketing",
    layer: "mid-term",
    category: "analysis"
  }
})

// Retrieve context
mcp__memory-mcp__vector_search({
  query: "previous campaign targeting strategies",
  limit: 10
})
```

## Validation & Quality

### Validation Gates

Each phase includes validation criteria:

**Phase 1 Gate**:
- [ ] Can describe domain in specific, technical terms
- [ ] Identified 5+ key challenges
- [ ] Mapped technology stack comprehensively
- [ ] Clear on integration requirements

**Phase 2 Gate**:
- [ ] Identified 3+ expertise domains
- [ ] Documented 5+ decision heuristics
- [ ] Created complete agent specification
- [ ] Examples demonstrate quality standards

**Phase 3 Gate**:
- [ ] System prompt follows template structure
- [ ] All Phase 2 expertise embedded
- [ ] Evidence-based techniques integrated
- [ ] Guardrails cover identified failure modes
- [ ] 2+ workflow examples with exact commands

**Phase 4 Gate**:
- [ ] Code patterns include file/line references
- [ ] Failure modes have detection + prevention
- [ ] MCP patterns show exact syntax
- [ ] Performance metrics defined
- [ ] Agent can self-improve through metrics

### Testing Checklist

Complete validation before production deployment:

- [ ] **Identity**: Agent maintains consistent role
- [ ] **Commands**: Uses universal commands correctly
- [ ] **Specialist Skills**: Demonstrates domain expertise
- [ ] **MCP Integration**: Coordinates via memory and tools
- [ ] **Guardrails**: Prevents identified failure modes
- [ ] **Workflows**: Executes examples successfully
- [ ] **Metrics**: Tracks performance data
- [ ] **Code Patterns**: Applies exact patterns from Phase 4
- [ ] **Error Handling**: Escalates appropriately
- [ ] **Consistency**: Produces stable outputs on repeat

## References

Supporting documentation in `references/` directory:

- **[best-practices.md](references/best-practices.md)**: Evidence-based prompting principles and optimization techniques
- **[agent-types.md](references/agent-types.md)**: Detailed specifications for Specialist, Coordinator, and Hybrid patterns
- **[integration-patterns.md](references/integration-patterns.md)**: MCP tool usage patterns and memory coordination

## Workflow Visualization

GraphViz diagram showing the complete 4-phase workflow available at:
`graphviz/workflow.dot`

To generate visualization:
```bash
dot -Tpng graphviz/workflow.dot -o workflow.png
```

## Performance Metrics

Track agent performance with built-in metrics:

```yaml
Task Completion:
  - tasks-completed: [count]
  - task-duration: [milliseconds]

Quality:
  - validation-passes: [count]
  - escalations: [count when needed help]
  - error-rate: [failures / attempts]

Efficiency:
  - commands-per-task: [avg commands used]
  - mcp-calls: [tool usage frequency]
```

## Continuous Improvement

### Maintenance Cycle

1. **Metrics Review**: Weekly review of agent performance metrics
2. **Failure Analysis**: Document and fix new failure modes
3. **Pattern Updates**: Add newly discovered code patterns
4. **Workflow Optimization**: Refine based on usage patterns

### Version Control

- **v1.0**: Base prompt from Phase 3
- **v1.x**: Minor refinements from testing
- **v2.0**: Enhanced with Phase 4 patterns
- **v2.x**: Production iterations and improvements

## Support & Resources

- **Full Skill Documentation**: `skill.md`
- **Official SOP Source**: Desktop `.claude-flow/` documentation
- **Claude Agent SDK**: https://github.com/anthropics/claude-agent-sdk
- **Claude Flow MCP**: https://github.com/ruvnet/claude-flow

## When to Use This Skill

Use **agent-creator** for:

- ✅ Creating project-specialized agents with deeply embedded domain knowledge
- ✅ Building agents for recurring tasks requiring consistent behavior
- ✅ Rewriting existing agents to optimize performance
- ✅ Creating multi-agent workflows with sequential or parallel coordination
- ✅ Agents that will integrate with MCP servers and Claude Flow

## Summary

The Agent Creator skill delivers:

- **Official 4-phase SOP methodology** from Desktop `.claude-flow`
- **Evidence-based prompting techniques** (self-consistency, PoT, plan-and-solve)
- **Claude Agent SDK implementation** (TypeScript + Python)
- **Production validation** and testing frameworks
- **Continuous improvement** through metrics

Create all 90+ specialist agents with:
- Deeply embedded domain knowledge
- Exact command and MCP tool specifications
- Production-ready failure prevention
- Measurable performance tracking

**Result**: Production-ready agents that consistently deliver high-quality results across diverse domains and workflows.
