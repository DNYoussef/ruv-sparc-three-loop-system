# Agent Creation - Systematic Agent Design with Evidence-Based Prompting

**Tier**: Silver (7+ files) | **Status**: Production Ready | **Version**: 1.0.0

## Overview

The Agent Creation skill provides a systematic 4-phase methodology for designing, implementing, testing, and deploying specialized AI agents using evidence-based prompting principles. This skill combines best practices from prompt engineering research with practical agent development workflows to create high-quality, reliable, and maintainable agents.

## Why Agent Creation Matters

Effective agent design is critical for:
- **Specialization**: Focused expertise in specific domains (Python, React, Database Design)
- **Coordination**: Multi-agent systems that work together effectively
- **Reliability**: Consistent, predictable agent behavior
- **Maintainability**: Clear agent roles and responsibilities
- **Performance**: Optimized prompts that reduce token usage and improve accuracy

## Quick Start

### Basic Agent Creation

```bash
# 1. Define agent specification
# Who: Data analyst with SQL expertise
# What: Query optimization and database analysis
# When: Database performance issues arise
# Why: Optimize query performance and reduce latency

# 2. Use Claude Code's Task tool to spawn the agent
Task("SQL Optimizer", "You are an expert SQL database optimizer specializing in PostgreSQL query performance. Analyze query execution plans using EXPLAIN ANALYZE, identify bottlenecks like missing indexes or inefficient joins, and provide concrete optimization recommendations with before/after benchmarks.", "code-analyzer")

# 3. Test with diverse inputs
# - Complex JOIN queries
# - Subquery optimization
# - Index recommendations
# - Execution plan analysis

# 4. Deploy with coordination hooks
npx claude-flow@alpha hooks pre-task --description "SQL query optimization"
npx claude-flow@alpha hooks post-task --task-id "sql-opt-001"
```

### Using the 4-Phase SOP

The Agent Creation SOP follows these phases:

1. **Specification** (Define the agent)
2. **Prompt Engineering** (Craft effective prompts)
3. **Testing & Validation** (Verify agent quality)
4. **Integration** (Deploy with coordination)

See [graphviz/workflow.dot](graphviz/workflow.dot) for a visual workflow diagram.

## 4-Phase Agent Creation SOP

### Phase 1: Specification

Define the agent's purpose, domain, and capabilities.

**Key Questions**:
- What domain expertise does the agent need?
- What are the core capabilities required?
- What inputs will the agent receive?
- What outputs should the agent produce?
- What quality criteria define success?

**Example**:
```yaml
Agent: Python Performance Optimizer
Domain: Python code optimization
Capabilities:
  - Profile Python code for bottlenecks
  - Apply algorithmic optimizations
  - Recommend data structure improvements
  - Use Cython/NumPy for acceleration
Inputs: Python source code files
Outputs: Optimized code with performance benchmarks
Quality: 2x+ performance improvement, maintain correctness
```

**Deliverables**:
- Agent specification document
- Capability requirements list
- Input/output format definitions
- Success criteria and metrics

### Phase 2: Prompt Engineering

Apply evidence-based prompting principles to craft effective agent prompts.

**Evidence-Based Principles**:

1. **Role Definition**: Clearly define the agent's identity and expertise
   ```
   You are an expert Python performance engineer with 10+ years optimizing production systems.
   ```

2. **Context Provision**: Provide relevant background information
   ```
   Your focus is algorithmic optimization, data structure selection, and profiling-driven improvements.
   ```

3. **Task Decomposition**: Break complex tasks into steps
   ```
   1. Profile the code to identify bottlenecks
   2. Analyze algorithmic complexity
   3. Apply targeted optimizations
   4. Benchmark improvements
   ```

4. **Chain-of-Thought**: Use reasoning for complex decisions
   ```
   Before recommending optimizations, explain your reasoning:
   - Why is this a bottleneck?
   - What optimization technique applies?
   - What are the trade-offs?
   ```

5. **Few-Shot Learning**: Provide concrete examples
   ```
   Example 1: Replacing list iteration with NumPy vectorization
   Example 2: Using dict.get() instead of try/except KeyError
   Example 3: Implementing memoization for recursive functions
   ```

6. **Output Formatting**: Define structured response templates
   ```
   Output Format:
   1. **Bottleneck Analysis**: [findings]
   2. **Optimization Strategy**: [approach]
   3. **Implementation**: [code]
   4. **Benchmarks**: [before/after metrics]
   ```

7. **Quality Constraints**: Set explicit success criteria
   ```
   - Maintain 100% functional correctness
   - Achieve minimum 2x performance improvement
   - Preserve code readability
   - Include unit tests for optimized code
   ```

**Deliverables**:
- Complete agent prompt with all principles applied
- Few-shot examples (3-5 examples)
- Output format template
- Quality constraint checklist

### Phase 3: Testing & Validation

Test the agent with diverse inputs and validate output quality.

**Testing Strategy**:

1. **Diverse Input Testing**:
   - Simple cases (baseline functionality)
   - Complex cases (edge cases, error handling)
   - Adversarial cases (stress testing)

2. **Output Quality Validation**:
   - Correctness (functional validation)
   - Completeness (all requirements met)
   - Consistency (reproducible results)
   - Format compliance (structured output)

3. **Performance Metrics**:
   - Response time
   - Token usage
   - Accuracy/success rate
   - Error rate

4. **Iterative Refinement**:
   - Analyze failures and edge cases
   - Adjust prompt engineering
   - Add examples for weak areas
   - Re-test until quality criteria met

**Testing Example**:
```javascript
// Test Cases for Python Optimizer Agent
const testCases = [
  {
    name: "Simple Loop Optimization",
    input: "for i in range(1000000): result.append(i*2)",
    expected: "List comprehension or NumPy vectorization"
  },
  {
    name: "Nested Dict Lookup",
    input: "try: value = data[key1][key2][key3] except KeyError: value = None",
    expected: "Use dict.get() with chaining"
  },
  {
    name: "Recursive Fibonacci",
    input: "def fib(n): return fib(n-1) + fib(n-2) if n > 1 else n",
    expected: "Memoization or iterative approach"
  }
];
```

**Deliverables**:
- Test suite with 10+ test cases
- Performance benchmarks
- Quality validation report
- Iteration log with improvements

### Phase 4: Integration

Deploy the agent with coordination protocols and monitoring.

**Integration Steps**:

1. **Coordination Protocol Setup**:
   ```bash
   # Pre-task hook: Initialize agent state
   npx claude-flow@alpha hooks pre-task --description "[agent task]"

   # Session restore: Load prior context
   npx claude-flow@alpha hooks session-restore --session-id "swarm-[id]"
   ```

2. **Memory Integration**:
   ```javascript
   // Store agent outputs in Memory-MCP
   const { taggedMemoryStore } = require('./hooks/12fa/memory-mcp-tagging-protocol.js');

   taggedMemoryStore(
     'python-optimizer',
     'Optimized sorting algorithm with 3.2x speedup',
     {
       task_id: 'OPT-42',
       file: 'src/data_processor.py',
       improvement: '3.2x',
       technique: 'TimSort to RadixSort'
     }
   );
   ```

3. **Communication Patterns**:
   ```bash
   # Notify other agents of completion
   npx claude-flow@alpha hooks notify --message "Python optimization complete"

   # Update shared memory
   npx claude-flow@alpha hooks post-edit --file "src/optimized.py" \
     --memory-key "swarm/python-optimizer/output"
   ```

4. **Monitoring & Metrics**:
   ```bash
   # Post-task hook: Export performance metrics
   npx claude-flow@alpha hooks post-task --task-id "[task-id]"

   # Session end: Generate summary
   npx claude-flow@alpha hooks session-end --export-metrics true
   ```

**Deliverables**:
- Coordination protocol documentation
- Memory integration code
- Communication pattern definitions
- Monitoring dashboard configuration

## Agent Types

### 1. Specialist Agents

**Characteristics**:
- Single domain expertise (Python, React, SQL)
- Deep knowledge in specific area
- Optimized prompts for narrow tasks
- High accuracy in specialization

**Use Cases**:
- Language-specific code generation
- Framework-specific implementation
- Domain-specific analysis
- Specialized optimization

**Example**: See [examples/example-1-specialist.md](examples/example-1-specialist.md)

### 2. Coordinator Agents

**Characteristics**:
- Multi-agent orchestration
- Task delegation and routing
- Progress monitoring
- Conflict resolution

**Use Cases**:
- Swarm coordination
- Workflow management
- Resource allocation
- Quality assurance

**Example**: See [examples/example-2-coordinator.md](examples/example-2-coordinator.md)

### 3. Hybrid Agents

**Characteristics**:
- Multi-domain capabilities
- Adaptive role switching
- Complex workflow handling
- Context-aware behavior

**Use Cases**:
- Full-stack development
- End-to-end feature implementation
- Cross-domain problem solving
- Adaptive task handling

## Evidence-Based Prompting Techniques

### Chain-of-Thought (CoT)

Improves reasoning for complex tasks by requiring step-by-step thinking.

**Research**: Wei et al. (2022) - "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"

**Application**:
```
Task: Optimize database query performance

Think step-by-step:
1. Analyze the query execution plan
2. Identify performance bottlenecks (sequential scans, missing indexes)
3. Evaluate optimization strategies (indexing, query rewriting)
4. Estimate improvement impact
5. Recommend specific optimizations with rationale
```

### Few-Shot Learning

Provides concrete examples to guide agent behavior and output format.

**Research**: Brown et al. (2020) - "Language Models are Few-Shot Learners"

**Application**:
```
Example 1:
Input: SELECT * FROM users WHERE age > 30 AND city = 'NYC'
Output: Add composite index on (city, age) for 10x speedup

Example 2:
Input: SELECT COUNT(*) FROM orders JOIN users ON orders.user_id = users.id
Output: Use COUNT with EXISTS for 3x faster aggregation

Example 3:
Input: SELECT * FROM products ORDER BY price LIMIT 10
Output: Create index on price column for efficient top-N query
```

### Role-Based Prompting

Defines clear agent identity to activate domain-specific knowledge.

**Research**: Zhou et al. (2023) - "Large Language Models Are Human-Level Prompt Engineers"

**Application**:
```
You are a senior database performance engineer at a high-traffic e-commerce platform.
You have 15+ years of experience optimizing PostgreSQL databases serving 10M+ queries/day.
Your expertise includes index design, query optimization, and database tuning.
```

For more details, see [references/evidence-based-prompting.md](references/evidence-based-prompting.md)

## Integration with Claude Code & MCP

### Claude Code Task Tool

Claude Code's Task tool is the primary way to spawn agents:

```javascript
// Single message with parallel agent spawning
Task("Python Optimizer", "Optimize data processing code...", "code-analyzer")
Task("Test Engineer", "Create performance benchmarks...", "tester")
Task("Code Reviewer", "Review optimization changes...", "reviewer")
```

### MCP Coordination (Optional)

For complex multi-agent tasks, use MCP tools for coordination setup:

```javascript
// Step 1: Initialize coordination topology
mcp__claude-flow__swarm_init({ topology: "mesh", maxAgents: 4 })

// Step 2: Spawn agents via Claude Code Task tool
Task("Agent 1", "...", "agent-type-1")
Task("Agent 2", "...", "agent-type-2")
```

### Memory-MCP Integration

All agents should use Memory-MCP for persistent state:

```javascript
const { taggedMemoryStore } = require('./hooks/12fa/memory-mcp-tagging-protocol.js');

// Store with WHO/WHEN/PROJECT/WHY tags
taggedMemoryStore('agent-name', 'Agent output or state', {
  task_id: 'TASK-123',
  file: 'path/to/file.py'
});
```

## Examples

### Example 1: Python Performance Specialist

A domain-specific specialist agent focused on Python code optimization.

**Full example**: [examples/example-1-specialist.md](examples/example-1-specialist.md)

**Key Features**:
- Profiling-driven optimization
- Algorithmic complexity analysis
- Data structure recommendations
- Benchmark-driven validation

### Example 2: Multi-Agent Coordinator

A coordinator agent that orchestrates multiple specialist agents.

**Full example**: [examples/example-2-coordinator.md](examples/example-2-coordinator.md)

**Key Features**:
- Task delegation to specialists
- Progress monitoring
- Result aggregation
- Conflict resolution

### Example 3: Prompt Engineering Best Practices

A comprehensive guide to evidence-based prompting techniques.

**Full example**: [examples/example-3-prompt-engineering.md](examples/example-3-prompt-engineering.md)

**Key Features**:
- Research-backed techniques
- Before/after examples
- Common pitfalls
- Optimization strategies

## References

- [Evidence-Based Prompting](references/evidence-based-prompting.md) - Research-backed prompting techniques
- [Agent Patterns](references/agent-patterns.md) - Specialist, Coordinator, and Hybrid patterns
- [Workflow Diagram](graphviz/workflow.dot) - Visual representation of 4-phase SOP

## Best Practices

### Do's

✅ Define clear agent roles and responsibilities
✅ Use evidence-based prompting principles
✅ Test with diverse inputs including edge cases
✅ Implement coordination hooks for multi-agent systems
✅ Store agent state in Memory-MCP with proper tagging
✅ Monitor agent performance and iterate
✅ Document agent capabilities and limitations

### Don'ts

❌ Create agents with vague or undefined roles
❌ Skip testing and validation phases
❌ Ignore coordination protocols in multi-agent systems
❌ Forget to implement error handling
❌ Neglect performance monitoring
❌ Use overly complex prompts without testing
❌ Deploy agents without integration testing

## Metrics & Success Criteria

### Agent Quality Metrics

- **Accuracy**: Correctness of agent outputs (target: 95%+)
- **Consistency**: Reproducibility of results (target: 90%+)
- **Performance**: Response time and token usage
- **Reliability**: Error rate (target: <5%)

### Integration Metrics

- **Coordination Efficiency**: Time from task assignment to completion
- **Memory Persistence**: State recovery success rate
- **Communication Latency**: Inter-agent message delay
- **System Throughput**: Tasks completed per hour

## Related Skills

- **skill-builder**: Create new skills with YAML frontmatter
- **prompt-architect**: Optimize and improve existing prompts
- **swarm-orchestration**: Coordinate multi-agent systems
- **agent-creator**: Enhanced agent creation with specification generation

## Version History

- **1.0.0** (2025-11-02): Initial Silver tier release
  - 4-phase SOP methodology
  - Evidence-based prompting principles
  - 3 comprehensive examples
  - GraphViz workflow diagram
  - Integration with Claude Code Task tool

## Contributing

To improve this skill:

1. Add more agent examples to `examples/`
2. Document new prompting techniques in `references/`
3. Create additional workflow diagrams in `graphviz/`
4. Update test cases based on real-world usage
5. Share agent performance metrics and optimization insights

---

**Next Steps**: Review the examples and references, then start creating your first agent using the 4-phase SOP!
