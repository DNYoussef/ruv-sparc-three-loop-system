# Agent Patterns: Specialist, Coordinator, and Hybrid

**Purpose**: Catalog common agent architectural patterns with use cases, strengths, weaknesses, and implementation guidance.

## Overview

AI agents can be categorized into three primary architectural patterns based on their scope, coordination requirements, and specialization level. Understanding these patterns helps in selecting the right agent type for a given task and designing effective multi-agent systems.

---

## 1. Specialist Agents

### Definition

**Specialist agents** have deep expertise in a narrow domain and are optimized for specific, well-defined tasks within that domain.

### Characteristics

- **Single Domain**: Focused on one technology, framework, or problem space
- **Deep Knowledge**: Expert-level understanding of domain-specific concepts
- **Optimized Prompts**: Prompts tuned for specific task patterns
- **High Accuracy**: 95%+ correctness in specialization
- **Limited Scope**: Do not handle tasks outside their domain

### Examples

1. **Python Performance Optimizer**
   - Domain: Python code optimization
   - Tasks: Profiling, algorithmic improvements, vectorization
   - Techniques: Cython, NumPy, multiprocessing

2. **React Component Developer**
   - Domain: React UI development
   - Tasks: Component creation, state management, hooks
   - Techniques: Memoization, code splitting, performance optimization

3. **SQL Query Optimizer**
   - Domain: Database query optimization
   - Tasks: EXPLAIN analysis, index design, query rewriting
   - Techniques: Execution plan analysis, statistics-driven optimization

4. **Security Auditor**
   - Domain: Application security
   - Tasks: Vulnerability scanning, code review, threat modeling
   - Techniques: OWASP Top 10, penetration testing, secure coding

### When to Use Specialist Agents

✅ **Use When**:
- Task requires deep domain expertise
- Clear task boundaries exist
- High accuracy is critical
- Domain-specific best practices must be followed
- Task repeats frequently with similar patterns

❌ **Avoid When**:
- Task spans multiple domains
- Requirements are ambiguous or evolving
- Coordination with other agents is primary concern
- Task is one-time and exploratory

### Strengths

- **High Accuracy**: Expert-level performance in specialization
- **Efficiency**: Optimized prompts reduce token usage
- **Consistency**: Predictable behavior for similar tasks
- **Best Practices**: Built-in domain-specific knowledge
- **Fast Execution**: Narrow scope enables quick responses

### Weaknesses

- **Limited Scope**: Cannot handle multi-domain tasks
- **No Coordination**: Require external orchestration
- **Context Blind**: May miss broader system implications
- **Inflexible**: Not adaptive to changing requirements

### Implementation Pattern

```markdown
You are a [expertise level] [domain] specialist with [years] years of experience in [specific areas].

## Your Expertise
- [Skill 1]: [description]
- [Skill 2]: [description]
- [Skill 3]: [description]

## Your Approach
[Step-by-step methodology specific to domain]

## Output Format
[Domain-specific structured output]

## Quality Constraints
[Domain-specific requirements]

## Few-Shot Examples
[3-5 domain-specific examples]
```

### Performance Metrics

| Metric | Target | Typical Achievable |
|--------|--------|-------------------|
| Domain Accuracy | 95%+ | 92-98% |
| Response Time | <5s | 2-8s |
| Format Compliance | 90%+ | 85-95% |
| Token Efficiency | High | 20-40% better than general agents |

---

## 2. Coordinator Agents

### Definition

**Coordinator agents** orchestrate multiple specialist agents, managing task delegation, progress monitoring, result aggregation, and quality validation in multi-agent systems.

### Characteristics

- **Multi-Agent Orchestration**: Manage 3-10+ specialist agents
- **Task Decomposition**: Break complex features into sub-tasks
- **Dependency Management**: Track and enforce task dependencies
- **Progress Monitoring**: Detect blockers and intervene
- **Result Aggregation**: Combine outputs from multiple agents
- **Quality Validation**: Enforce gates and checkpoints

### Examples

1. **Full-Stack Feature Coordinator**
   - Manages: backend-dev, frontend-dev, tester, reviewer
   - Tasks: Feature decomposition, delegation, integration
   - Workflow: Plan → Delegate → Monitor → Validate

2. **Data Pipeline Coordinator**
   - Manages: data-ingestion, data-transformation, data-validation, data-loading
   - Tasks: Pipeline orchestration, error handling, monitoring
   - Workflow: Extract → Transform → Load → Validate

3. **DevOps Deployment Coordinator**
   - Manages: build-agent, test-agent, security-scan-agent, deploy-agent
   - Tasks: CI/CD orchestration, rollback handling
   - Workflow: Build → Test → Scan → Deploy → Monitor

4. **Research Project Coordinator**
   - Manages: literature-review, method-design, evaluation, writing
   - Tasks: Research lifecycle management, quality gates
   - Workflow: Review → Design → Evaluate → Publish

### When to Use Coordinator Agents

✅ **Use When**:
- Task requires multiple specialist agents
- Dependencies exist between sub-tasks
- Parallel execution opportunities exist
- Quality gates must be enforced
- Progress monitoring is critical
- Complex workflows span multiple domains

❌ **Avoid When**:
- Single specialist agent sufficient
- Task has no clear sub-tasks
- Linear sequential workflow (use pipeline instead)
- No coordination overhead justifiable

### Strengths

- **Parallelization**: 2-4x speedup through concurrent execution
- **Quality Assurance**: Enforces validation gates
- **Adaptability**: Adjusts to blockers and failures
- **Completeness**: Ensures all sub-tasks completed
- **Metrics Tracking**: Provides visibility into progress

### Weaknesses

- **Overhead**: 5-15% coordination overhead
- **Complexity**: Difficult to debug multi-agent interactions
- **Single Point of Failure**: Coordinator failure blocks entire workflow
- **Context Management**: Challenging to maintain shared state

### Implementation Pattern

```markdown
You are an expert [workflow type] coordinator with [years] years of experience orchestrating distributed teams.

## Your Role
1. **Decompose** complex tasks into sub-tasks
2. **Delegate** to appropriate specialist agents
3. **Monitor** progress and handle blockers
4. **Aggregate** results from multiple agents
5. **Validate** quality through checkpoints

## Delegation Template (Claude Code Task tool)
Task(
  "[Agent Name]",
  `Task: [description]
   Context: [background]
   Inputs: [dependencies]
   Expected Output: [deliverables]
   Quality Criteria: [requirements]
   Coordination: [hooks, memory keys]`,
  "[agent-type]"
)

## Progress Monitoring
[Commands for checking status, metrics, blockers]

## Quality Gates
1. Gate 1: [checkpoint 1]
2. Gate 2: [checkpoint 2]
3. Gate 3: [checkpoint 3]

## Output Format
1. Task Decomposition: [sub-tasks, dependencies]
2. Delegation Plan: [agent assignments]
3. Progress Report: [status, blockers]
4. Quality Validation: [gate results]
5. Final Deliverable: [aggregated output]
```

### Performance Metrics

| Metric | Target | Typical Achievable |
|--------|--------|-------------------|
| Parallelization Speedup | 2-4x | 1.8-3.5x |
| Task Completion Rate | 95%+ | 90-97% |
| Quality Gate Pass Rate | 90%+ | 85-95% |
| Blocker Resolution Time | <30min | 10-45min |
| Coordination Overhead | <10% | 5-15% |

---

## 3. Hybrid Agents

### Definition

**Hybrid agents** combine specialist expertise across multiple domains with coordination capabilities, enabling adaptive behavior and end-to-end ownership of complex, cross-domain tasks.

### Characteristics

- **Multi-Domain Expertise**: Competent in 2-4 related domains
- **Adaptive Role Switching**: Adjust behavior based on task phase
- **Self-Coordination**: Manage own workflow without external orchestrator
- **Context-Aware**: Understand broader system implications
- **End-to-End Ownership**: Complete features independently

### Examples

1. **Full-Stack Developer Agent**
   - Domains: Backend (Node.js), Frontend (React), Database (PostgreSQL)
   - Roles: API developer, UI developer, database designer
   - Workflow: Design → Implement → Test → Deploy

2. **DevSecOps Agent**
   - Domains: DevOps (CI/CD), Security (scanning), Operations (monitoring)
   - Roles: Build engineer, security auditor, SRE
   - Workflow: Build → Scan → Deploy → Monitor

3. **ML Engineer Agent**
   - Domains: Data engineering, model training, deployment
   - Roles: Data pipeline builder, ML researcher, MLOps engineer
   - Workflow: Data → Train → Evaluate → Deploy

4. **Technical Writer + Developer Agent**
   - Domains: Documentation, code generation, API design
   - Roles: Technical writer, code generator, API architect
   - Workflow: Design → Code → Document → Review

### When to Use Hybrid Agents

✅ **Use When**:
- Task spans 2-4 related domains
- Strong coupling between domain tasks
- Coordination overhead not justified
- Context preservation critical across phases
- Independent end-to-end ownership needed

❌ **Avoid When**:
- Deep expertise in single domain required
- Many independent specialists available
- Parallel execution across domains needed
- Clear domain boundaries exist

### Strengths

- **End-to-End Ownership**: Complete features independently
- **Context Preservation**: No information loss between specialists
- **Flexibility**: Adapt to changing requirements
- **Reduced Coordination**: No external orchestrator needed
- **Holistic Decisions**: Consider cross-domain implications

### Weaknesses

- **Jack-of-All-Trades**: Less deep expertise than pure specialists
- **Prompt Complexity**: Harder to optimize multi-domain prompts
- **Token Usage**: Longer prompts due to multi-domain knowledge
- **Debugging**: Harder to isolate issues across domains

### Implementation Pattern

```markdown
You are a [expertise level] [multi-domain] engineer with [years] years of experience across [domain 1], [domain 2], and [domain 3].

## Your Multi-Domain Expertise

### Domain 1: [name]
- [Skill 1]
- [Skill 2]

### Domain 2: [name]
- [Skill 1]
- [Skill 2]

### Domain 3: [name]
- [Skill 1]
- [Skill 2]

## Adaptive Workflow

### Phase 1: [domain 1 focus]
[Approach for domain 1]

### Phase 2: [domain 2 focus]
[Approach for domain 2, considering domain 1 output]

### Phase 3: [domain 3 focus]
[Approach for domain 3, integrating 1 and 2]

## Context Management
- Track state across domains: [mechanism]
- Maintain consistency: [approach]
- Handle conflicts: [resolution strategy]

## Output Format
[Structured output covering all domains]

## Quality Constraints
- Domain 1: [requirements]
- Domain 2: [requirements]
- Domain 3: [requirements]
- Integration: [cross-domain requirements]
```

### Performance Metrics

| Metric | Target | Typical Achievable |
|--------|--------|-------------------|
| Cross-Domain Consistency | 90%+ | 85-93% |
| End-to-End Completion | 85%+ | 80-90% |
| Context Preservation | 95%+ | 90-96% |
| vs. Specialist Accuracy | -10% | -5% to -15% |
| vs. Coordinator Speed | +20% | +10% to +30% |

---

## Pattern Selection Decision Tree

```
Start: What type of task do you have?

├─ Single domain, well-defined task?
│  ├─ Yes → **Specialist Agent**
│  │  └─ Example: Python code optimization
│  └─ No → Continue
│
├─ Multiple specialists needed, clear sub-tasks?
│  ├─ Yes → **Coordinator Agent**
│  │  └─ Example: Full-stack feature with backend, frontend, tests
│  └─ No → Continue
│
├─ Spans 2-4 related domains, strong coupling?
│  ├─ Yes → **Hybrid Agent**
│  │  └─ Example: Full-stack CRUD feature by single agent
│  └─ No → Consider task decomposition or human involvement
```

---

## Pattern Comparison Matrix

| Dimension | Specialist | Coordinator | Hybrid |
|-----------|-----------|-------------|--------|
| **Domains Covered** | 1 | N/A (manages others) | 2-4 |
| **Accuracy** | 95-98% | N/A (validates) | 85-93% |
| **Speed** | Fast (2-5s) | Depends on parallel (1.8-3.5x) | Medium (5-15s) |
| **Token Usage** | Low | Medium (delegation) | High |
| **Coordination** | None | High | Self-managed |
| **Context Preservation** | N/A | Low (hand-offs) | High |
| **Use Case** | Deep expertise | Multi-agent workflows | Cross-domain features |
| **Scalability** | High (parallel) | Medium (overhead) | Low (monolithic) |

---

## Design Patterns for Each Type

### Specialist Agent Design Pattern

**Template**:
```
Role: [Expert in X]
Input: [Specific data format]
Process: [Domain-specific methodology]
Output: [Structured result]
Quality: [Domain metrics]
```

**Best Practices**:
- Optimize prompts for repetitive tasks
- Include 5+ few-shot examples
- Define narrow scope clearly
- Use domain-specific terminology

### Coordinator Agent Design Pattern

**Template**:
```
Role: [Orchestrator of Y workflow]
Phase 1: Decompose task into sub-tasks
Phase 2: Delegate to specialists via Task tool
Phase 3: Monitor progress and handle blockers
Phase 4: Aggregate results and validate
Output: [Comprehensive report + deliverables]
```

**Best Practices**:
- Use Claude Code Task tool for parallel delegation
- Implement quality gates at checkpoints
- Define clear dependency graphs
- Use Memory-MCP for shared state

### Hybrid Agent Design Pattern

**Template**:
```
Role: [Multi-domain expert in X, Y, Z]
Phase 1: [Domain X approach]
Phase 2: [Domain Y approach, considering X output]
Phase 3: [Domain Z approach, integrating X and Y]
Context: [Cross-domain state management]
Output: [Integrated multi-domain result]
```

**Best Practices**:
- Explicitly manage context across domains
- Define phase transitions clearly
- Balance breadth vs depth in prompts
- Use adaptive role switching

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Specialist Doing Coordination
❌ **Problem**: Python specialist agent trying to coordinate frontend and backend
✅ **Solution**: Use coordinator agent to manage specialists

### Anti-Pattern 2: Coordinator Writing Code
❌ **Problem**: Coordinator agent implementing code instead of delegating
✅ **Solution**: Coordinator should ONLY delegate, monitor, aggregate

### Anti-Pattern 3: Hybrid Doing Deep Specialization
❌ **Problem**: Full-stack agent attempting advanced ML research
✅ **Solution**: Use specialist ML agent for deep technical work

### Anti-Pattern 4: Too Many Hybrids
❌ **Problem**: 10 hybrid agents each doing full-stack work independently
✅ **Solution**: Use specialists + coordinator for better parallelization

### Anti-Pattern 5: Specialist with Vague Scope
❌ **Problem**: "General developer agent" without clear domain
✅ **Solution**: Define narrow specialization (Python, React, SQL)

---

## Migration Paths Between Patterns

### Specialist → Hybrid
**When**: Task scope expands to adjacent domains
**Approach**: Add secondary domain expertise while maintaining primary specialization

### Hybrid → Specialist
**When**: Multi-domain complexity too high, accuracy suffers
**Approach**: Split into 2-3 specialist agents + coordinator

### Specialists → Coordinator + Specialists
**When**: 3+ specialists need orchestration
**Approach**: Create coordinator agent to manage existing specialists

### Coordinator + Specialists → Hybrid
**When**: Coordination overhead > 15%, strong domain coupling
**Approach**: Merge 2-3 tightly coupled specialists into hybrid agent

---

## Real-World Examples

### Example 1: E-Commerce Checkout Feature

**Pattern**: Coordinator + Specialists

**Agents**:
- Coordinator: Full-Stack Feature Coordinator
- Specialists:
  - backend-dev: Payment API integration
  - frontend-dev: Checkout UI
  - database-design-specialist: Order schema
  - security-testing-agent: Payment security audit
  - tester: E2E checkout tests

**Why This Pattern**: Multiple specialists needed, clear sub-tasks, parallel opportunities

---

### Example 2: Python Script Optimization

**Pattern**: Specialist

**Agent**:
- python-optimizer: Python Performance Optimizer

**Why This Pattern**: Single domain, well-defined task, high accuracy critical

---

### Example 3: Landing Page Creation

**Pattern**: Hybrid

**Agent**:
- frontend-fullstack: HTML + CSS + JavaScript developer

**Why This Pattern**: Tightly coupled frontend domains, end-to-end ownership, no coordination overhead

---

## Further Reading

- [Example 1: Python Performance Specialist](../examples/example-1-specialist.md)
- [Example 2: Multi-Agent Coordinator](../examples/example-2-coordinator.md)
- [Evidence-Based Prompting](evidence-based-prompting.md)

---

**Next Steps**: Use this pattern catalog to select the right agent architecture for your task!
