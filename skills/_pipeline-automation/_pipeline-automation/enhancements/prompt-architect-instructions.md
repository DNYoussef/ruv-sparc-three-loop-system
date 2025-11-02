# Enhancement Instructions for prompt-architect

**Target Tier**: Bronze
**Current Tier**: Incomplete
**Estimated Time**: 1.5 hours

## Skill Purpose
--- name: prompt-architect description: Comprehensive framework for analyzing, creating, and refining prompts for AI systems. Use when creating prompts for Claude, ChatGPT, or other language models, i...

## Tasks to Complete

### Task 1: create_readme (Agent: technical-writer)

Create README.md following MECE universal template:

**Location**: `prompt-architect/README.md`

**Required Sections**:
1. Title and one-line description
2. Quick Start (2-3 steps)
3. When to Use This Skill
4. Structure Overview
5. Examples (link to examples/)
6. Quality Tier: Bronze

**Template**: Reference skill-forge/README.md as example

### Task 2: create_examples (Agent: researcher)

Create 1 example(s) in examples/ directory:

**Location**: `prompt-architect/examples/`

**Requirements**:
- example-1-basic.md: Simple, straightforward use case

**Format**: Step-by-step walkthrough with code samples

**Template**: Reference skill-forge/examples/ as model


## Coordination Protocol

**BEFORE starting**:
```bash
npx claude-flow@alpha hooks pre-task --description "Enhance prompt-architect to Bronze tier"
```

**DURING work**:
```bash
npx claude-flow@alpha hooks post-edit --file "{file}" --memory-key "skill-enhancement-pipeline/prompt-architect/{component}"
```

**AFTER completion**:
```bash
npx claude-flow@alpha hooks post-task --task-id "enhance-prompt-architect"
npx claude-flow@alpha memory store --key "skill-enhancement-pipeline/prompt-architect/status" --value "enhanced-to-Bronze"
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

**Tier Achievement**: Bronze
- File count: 3+ files
- All required directories present
- Quality validation passes (≥85%)

