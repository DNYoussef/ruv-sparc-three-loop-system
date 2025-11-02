# Enhancement Instructions for api-docs

**Target Tier**: Silver
**Current Tier**: Incomplete
**Estimated Time**: 2.5 hours

## Skill Purpose
...

## Tasks to Complete

### Task 1: create_readme (Agent: technical-writer)

Create README.md following MECE universal template:

**Location**: `api-docs/README.md`

**Required Sections**:
1. Title and one-line description
2. Quick Start (2-3 steps)
3. When to Use This Skill
4. Structure Overview
5. Examples (link to examples/)
6. Quality Tier: Silver

**Template**: Reference skill-forge/README.md as example

### Task 2: create_examples (Agent: researcher)

Create 1 example(s) in examples/ directory:

**Location**: `api-docs/examples/`

**Requirements**:
- example-1-basic.md: Simple, straightforward use case

**Format**: Step-by-step walkthrough with code samples

**Template**: Reference skill-forge/examples/ as model

### Task 3: create_references (Agent: technical-writer)

Create reference documentation in references/ directory:

**Location**: `api-docs/references/`

**Suggested Files**:
- best-practices.md: Guidelines and recommendations
- troubleshooting.md: Common issues and solutions
- related-skills.md: Links to complementary skills

**Content**: Abstract concepts, background knowledge, design decisions

### Task 4: create_graphviz (Agent: architect)

Create GraphViz process diagram(s):

**Location**: `api-docs/graphviz/`

**Suggested Diagrams**:
- workflow.dot: Main process flow
- orchestration.dot (if multi-agent): Agent coordination pattern

**Format**: GraphViz DOT language with clear labels

**Template**: Reference skill-forge/graphviz/ as model


## Coordination Protocol

**BEFORE starting**:
```bash
npx claude-flow@alpha hooks pre-task --description "Enhance api-docs to Silver tier"
```

**DURING work**:
```bash
npx claude-flow@alpha hooks post-edit --file "{file}" --memory-key "skill-enhancement-pipeline/api-docs/{component}"
```

**AFTER completion**:
```bash
npx claude-flow@alpha hooks post-task --task-id "enhance-api-docs"
npx claude-flow@alpha memory store --key "skill-enhancement-pipeline/api-docs/status" --value "enhanced-to-Silver"
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

**Tier Achievement**: Silver
- File count: 7+ files
- All required directories present
- Quality validation passes (≥85%)

