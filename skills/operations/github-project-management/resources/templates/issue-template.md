# Issue Templates

This file contains standardized issue templates for various types of work items.

---

## Feature Request

```markdown
## ✨ Feature Request

### Feature Description
[Clear description of the proposed feature]

### Use Cases
1. [Use case 1]
2. [Use case 2]
3. [Use case 3]

### Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

### Implementation Approach

#### Design
- [ ] Architecture design
- [ ] API design
- [ ] UI/UX mockups

#### Development
- [ ] Core implementation
- [ ] Integration with existing features
- [ ] Performance optimization

#### Testing
- [ ] Unit tests
- [ ] Integration tests
- [ ] User acceptance testing

### Swarm Coordination
- **Architect**: Design and planning
- **Coder**: Implementation
- **Tester**: Quality assurance
- **Documenter**: Documentation

### Estimated Complexity
[Low | Medium | High | Critical]

### Story Points
[1 | 2 | 3 | 5 | 8 | 13]

---
🤖 Generated with Claude Code
```

---

## Bug Report

```markdown
## 🐛 Bug Report

### Problem Description
[Clear description of the issue]

### Expected Behavior
[What should happen]

### Actual Behavior
[What actually happens]

### Reproduction Steps
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Environment
- **Package**: [package name and version]
- **Node.js**: [version]
- **OS**: [operating system]
- **Browser**: [if applicable]

### Error Logs
```
[Paste error logs or stack traces here]
```

### Investigation Plan
- [ ] Root cause analysis
- [ ] Fix implementation
- [ ] Testing and validation
- [ ] Regression testing
- [ ] Documentation update

### Swarm Assignment
- **Debugger**: Issue investigation
- **Coder**: Fix implementation
- **Tester**: Validation and testing

### Priority
[🔴 Critical | 🟡 High | 🟢 Medium | ⚪ Low]

---
🤖 Generated with Claude Code
```

---

## Integration Task

```markdown
## 🔄 Integration Task

### Overview
[Brief description of integration requirements]

### Objectives
- [ ] Component A integration
- [ ] Component B validation
- [ ] Testing and verification
- [ ] Documentation updates

### Integration Areas

#### Dependencies
- [ ] Package.json updates
- [ ] Version compatibility
- [ ] Import statements

#### Functionality
- [ ] Core feature integration
- [ ] API compatibility
- [ ] Performance validation

#### Testing
- [ ] Unit tests
- [ ] Integration tests
- [ ] End-to-end validation

### Affected Components
- [Component 1]
- [Component 2]
- [Component 3]

### Swarm Coordination
- **Coordinator**: Overall progress tracking
- **Analyst**: Technical validation
- **Coder**: Implementation
- **Tester**: Quality assurance
- **Documenter**: Documentation updates

### Dependencies
- Depends on: #[issue numbers]
- Blocks: #[issue numbers]

### Estimated Timeline
[X days/weeks]

---
🤖 Generated with Claude Code
```

---

## Swarm Task Template

```yaml
name: Swarm Task
description: Create a task for AI swarm processing
body:
  - type: dropdown
    id: topology
    attributes:
      label: Swarm Topology
      description: Choose the coordination pattern
      options:
        - mesh (peer-to-peer collaboration)
        - hierarchical (coordinator-led)
        - ring (sequential processing)
        - star (centralized control)
    validations:
      required: true

  - type: input
    id: agents
    attributes:
      label: Required Agents
      description: Comma-separated list of agent types
      placeholder: "coder, tester, analyst"
    validations:
      required: true

  - type: dropdown
    id: complexity
    attributes:
      label: Complexity
      options:
        - low
        - medium
        - high
        - critical
    validations:
      required: true

  - type: number
    id: storyPoints
    attributes:
      label: Story Points
      description: Effort estimation
      placeholder: "5"
    validations:
      required: false

  - type: textarea
    id: tasks
    attributes:
      label: Task Breakdown
      description: List of subtasks
      placeholder: |
        1. Task one description
        2. Task two description
        3. Task three description
    validations:
      required: true

  - type: textarea
    id: acceptanceCriteria
    attributes:
      label: Acceptance Criteria
      description: Definition of done
      placeholder: |
        - [ ] Criterion 1
        - [ ] Criterion 2
        - [ ] Criterion 3
    validations:
      required: true

  - type: dropdown
    id: priority
    attributes:
      label: Priority
      options:
        - 🔴 Critical
        - 🟡 High
        - 🟢 Medium
        - ⚪ Low
    validations:
      required: true

  - type: checkboxes
    id: swarmFeatures
    attributes:
      label: Swarm Features
      description: Enable advanced swarm capabilities
      options:
        - label: Auto-decomposition
        - label: Progress tracking
        - label: Memory persistence
        - label: Neural learning
```

---

## Documentation Task

```markdown
## 📚 Documentation Task

### Scope
[What needs to be documented]

### Objectives
- [ ] API reference documentation
- [ ] User guides
- [ ] Code examples
- [ ] Architecture diagrams
- [ ] Troubleshooting guides

### Target Audience
[Developers | Users | Contributors | Administrators]

### Deliverables
- [ ] README updates
- [ ] API documentation
- [ ] Tutorial guides
- [ ] Example code
- [ ] Diagrams/visuals

### Swarm Assignment
- **Researcher**: Gather information
- **Technical Writer**: Create documentation
- **Reviewer**: Technical review
- **Designer**: Visual assets

### Related Issues
- Related to: #[issue numbers]

---
🤖 Generated with Claude Code
```

---

## Performance Optimization

```markdown
## ⚡ Performance Optimization

### Current State
[Describe current performance metrics]

### Target State
[Desired performance improvements]

### Profiling Results
```
[Paste profiling data, benchmarks]
```

### Optimization Plan
- [ ] Identify bottlenecks
- [ ] Implement optimizations
- [ ] Benchmark improvements
- [ ] Validate correctness
- [ ] Document changes

### Performance Metrics
- **Before**: [metric values]
- **Target**: [target values]

### Swarm Assignment
- **Analyst**: Profiling and analysis
- **Optimizer**: Implementation
- **Tester**: Performance validation

### Story Points
[5 | 8 | 13]

---
🤖 Generated with Claude Code
```
