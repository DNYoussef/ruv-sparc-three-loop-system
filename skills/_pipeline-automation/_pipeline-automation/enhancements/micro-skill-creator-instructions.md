# Enhancement Instructions for micro-skill-creator

**Target Tier**: Gold
**Current Tier**: Incomplete
**Estimated Time**: 4.5 hours

## ⚠️ CRITICAL: Preserve Existing Files

**BEFORE starting any work**:
1. [!] DO NOT modify or delete existing skill.md/SKILL.md
2. [!] DO NOT modify or delete existing README.md (unless improving it)
3. [!] CREATE new files in proper MECE directories only
4. [!] FOLLOW file naming conventions (lowercase for all .md files except SKILL.md, README.md in root)

## Skill Purpose
--- name: micro-skill-creator description: Rapidly creates atomic, focused skills optimized with evidence-based prompting, specialist agents, and systematic testing. Each micro-skill does one thing ex...

## Tasks to Complete

### Task 1: create_readme (Agent: technical-writer)

Create README.md following MECE universal template:

**Location**: `micro-skill-creator/README.md`

**Required Sections**:
1. Title and one-line description
2. Quick Start (2-3 steps)
3. When to Use This Skill
4. Structure Overview
5. Examples (link to examples/)
6. Quality Tier: Gold

**Template**: Reference skill-forge/README.md as example

### Task 2: create_examples (Agent: researcher)

Create 3 example(s) in examples/ directory:

**Location**: `micro-skill-creator/examples/`

**Requirements**:
- example-1-basic.md: Simple, straightforward use case
- example-2-advanced.md: Complex scenario with agent integration
- example-3-edge-case.md: Edge case or specialized usage

**Format**: Step-by-step walkthrough with code samples

**Template**: Reference skill-forge/examples/ as model

### Task 3: create_graphviz (Agent: architect)

Create GraphViz process diagram(s):

**Location**: `micro-skill-creator/graphviz/`

**Suggested Diagrams**:
- workflow.dot: Main process flow
- orchestration.dot (if multi-agent): Agent coordination pattern

**Format**: GraphViz DOT language with clear labels

**Template**: Reference skill-forge/graphviz/ as model

### Task 4: create_resources (Agent: coder)

Create executable resources:

**Location**: `micro-skill-creator/resources/`

**Suggested Structure**:
- scripts/: Python/Shell scripts for deterministic operations
- templates/: YAML/JSON boilerplate files
- assets/: Images, configs, data files

**Quality**: Production-ready, well-documented code

### Task 5: create_resources (Agent: coder)

Create executable resources:

**Location**: `micro-skill-creator/resources/`

**Suggested Structure**:
- scripts/: Python/Shell scripts for deterministic operations
- templates/: YAML/JSON boilerplate files
- assets/: Images, configs, data files

**Quality**: Production-ready, well-documented code

### Task 6: create_tests (Agent: tester)

Create test cases:

**Location**: `micro-skill-creator/tests/`

**Suggested Tests**:
- test-basic.md: Basic functionality validation
- test-edge-cases.md: Edge case scenarios
- test-integration.md (if applicable): Integration testing

**Format**: Markdown with expected outcomes


## Coordination Protocol

**BEFORE starting**:
```bash
npx claude-flow@alpha hooks pre-task --description "Enhance micro-skill-creator to Gold tier"
```

**DURING work**:
```bash
npx claude-flow@alpha hooks post-edit --file "{file}" --memory-key "skill-enhancement-pipeline/micro-skill-creator/{component}"
```

**AFTER completion**:
```bash
npx claude-flow@alpha hooks post-task --task-id "enhance-micro-skill-creator"
npx claude-flow@alpha memory store --key "skill-enhancement-pipeline/micro-skill-creator/status" --value "enhanced-to-Gold"
```

## Quality Checklist

- [ ] All missing components created
- [ ] MECE structure validated
- [ ] File naming conventions followed
- [ ] Examples are concrete and actionable
- [ ] References provide value-added context
- [ ] GraphViz diagrams are clear
- [ ] Resources are production-ready
- [ ] Tests cover key scenarios

## Success Criteria

**Tier Achievement**: Gold
- File count: 12+ files
- All required directories present
- Quality validation passes (≥85%)

