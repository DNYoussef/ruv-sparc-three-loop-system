# Agent Creator Resources - Gold Tier Enhancement

This directory contains production-ready resources for the 4-phase agent creation SOP.

## Directory Structure

```
resources/
├── scripts/               # Executable automation scripts
│   ├── 4_phase_sop.py    # Complete 4-phase SOP automation
│   ├── validate_prompt.sh # System prompt quality validation
│   └── test_agent.py     # Agent testing framework
├── templates/             # Reusable templates
│   ├── system-prompt-template.md     # Markdown prompt template
│   └── evidence-based-prompt.yaml    # YAML specification template
└── README.md             # This file
```

## Scripts Overview

### 4_phase_sop.py
**Purpose**: Automate the 4-phase agent creation methodology

**Usage**:
```bash
# Interactive mode (all phases)
python 4_phase_sop.py --agent-name marketing-specialist --mode interactive

# Run specific phase
python 4_phase_sop.py --agent-name backend-dev --phase 1 --output-dir ./outputs

# Batch mode (from YAML input)
python 4_phase_sop.py --agent-name api-designer --mode batch --input spec.yaml
```

**Features**:
- **Phase 1**: Initial analysis & intent decoding (30-60 min)
  - Domain breakdown with 5+ key challenges
  - Technology stack mapping
  - Integration points identification
  - Validation gates with automatic checking

- **Phase 2**: Meta-cognitive extraction (30-45 min)
  - Expertise domain identification (3+ domains)
  - Decision framework documentation (5+ heuristics)
  - Agent specification creation
  - Supporting artifacts (good/bad examples, edge cases)

- **Phase 3**: Agent architecture design (45-60 min)
  - Base system prompt v1.0 generation
  - Evidence-based technique integration
  - Quality guardrails definition
  - Workflow examples with exact commands

**Outputs**:
- `{agent-name}-phase1-analysis.json` - Phase 1 domain analysis
- `{agent-name}-phase2-extraction.json` - Phase 2 expertise extraction
- `{agent-name}-specification.md` - Agent specification document
- `{agent-name}-phase3-architecture.json` - Phase 3 architecture
- `{agent-name}-base-prompt-v1.md` - Base system prompt
- `{agent-name}-4phase-sop-complete.json` - Complete results

### validate_prompt.sh
**Purpose**: Validate system prompt quality against evidence-based standards

**Usage**:
```bash
# Basic validation
bash validate_prompt.sh marketing-specialist-base-prompt-v1.md

# Verbose mode with detailed analysis
bash validate_prompt.sh -v backend-dev-enhanced-prompt-v2.md

# Custom minimum score
bash validate_prompt.sh -s 90 api-security-auditor-enhanced-prompt-v2.md
```

**Validation Checks** (100 points total):
1. Core Identity Section (15 pts)
2. Universal Commands (10 pts)
3. Specialist Commands (10 pts)
4. MCP Server Tools (15 pts)
5. Cognitive Framework (15 pts)
6. Guardrails (10 pts)
7. Success Criteria (10 pts)
8. Workflow Examples (15 pts)

**Tier Classification**:
- **Gold** (90-100%): Production-ready with excellent evidence-based patterns
- **Silver** (75-89%): Well-structured, recommended minor enhancements
- **Bronze** (70-74%): Functional, consider adding more patterns
- **Failing** (<70%): Requires significant improvements

**Exit Codes**:
- 0: Validation passed
- 1: Validation failed
- 2: Invalid arguments or file not found

### test_agent.py
**Purpose**: Test agent system prompts against typical cases, edge cases, and integration scenarios

**Usage**:
```bash
# Basic test suite
python test_agent.py --agent marketing-specialist --test-suite basic

# Comprehensive tests
python test_agent.py --agent devops-orchestrator --test-suite comprehensive

# Integration tests
python test_agent.py --agent api-security-auditor --test-suite integration

# Custom prompt file
python test_agent.py --agent custom-agent --prompt-file /path/to/prompt.md --test-suite basic
```

**Test Suites**:

**Basic Tests** (4 tests):
1. Identity Consistency - Validates core identity and role
2. Command Coverage - Checks essential commands documented
3. Evidence Patterns - Tests for prompting techniques
4. Structural Quality - Validates organization and examples

**Comprehensive Tests** (7 tests, includes Basic):
5. Edge Case Handling - Tests edge case documentation
6. Error Handling - Validates error patterns
7. Workflow Completeness - Checks workflow documentation

**Integration Tests** (10 tests, includes Comprehensive):
8. MCP Integration - Tests MCP tool patterns
9. Cross-Agent Coordination - Validates coordination patterns
10. Memory Patterns - Tests memory usage specifications

**Outputs**:
- `{agent-name}-test-report.json` - Detailed test results
- Console output with pass/fail status per test
- Success rate percentage

**Success Criteria**:
- Basic: 80%+ pass rate
- Comprehensive: 90%+ pass rate
- Integration: 95%+ pass rate

## Templates Overview

### system-prompt-template.md
**Purpose**: Markdown template for agent system prompts

**Sections**:
1. Core Identity - Agent role and expertise
2. Universal Commands - Standard operations
3. Specialist Commands - Domain-specific commands
4. MCP Server Tools - Integration patterns
5. Cognitive Framework - Evidence-based techniques
6. Guardrails - Failure prevention
7. Success Criteria - Completion checklist
8. Workflow Examples - Concrete usage patterns

**Variables** (replace with actual values):
- `{AGENT_NAME}`, `{VERSION}`, `{ROLE_TITLE}`
- `{DOMAIN_AREAS}`, `{PRIMARY_OBJECTIVE}`
- `{SPECIALIST_COMMANDS_LIST}`
- `{VALIDATION_1}`, `{DECOMPOSITION_1}`, `{PLAN_STEP}`
- `{FAILURE_CATEGORY_1}`, `{DANGEROUS_PATTERN_1}`
- `{WORKFLOW_NAME_1}`, `{WORKFLOW_OBJECTIVE_1}`

**Usage**: Copy template and replace variables with Phase 3 outputs.

### evidence-based-prompt.yaml
**Purpose**: YAML specification template for structured agent design

**Structure**:
```yaml
agent_name: "{agent-name}"
version: "1.0"

core_identity:
  role_title: "..."
  domain_areas: [...]
  primary_objective: "..."

universal_commands:
  file_operations: {...}
  git_operations: {...}
  communication: {...}

specialist_commands: [...]

mcp_tools:
  claude_flow: [...]
  domain_specific: [...]

cognitive_framework:
  self_consistency: {...}
  program_of_thought: {...}
  plan_and_solve: {...}

guardrails: [...]

success_criteria: [...]

workflows: [...]

metrics: {...}
```

**Usage**: Use as structured input for batch mode agent creation.

## Workflow Examples

### Example 1: Create Basic Agent
```bash
# Step 1: Run Phase 1-3
python scripts/4_phase_sop.py --agent-name file-organizer --mode interactive

# Step 2: Validate prompt
bash scripts/validate_prompt.sh agent-outputs/file-organizer/file-organizer-base-prompt-v1.md

# Step 3: Test agent
python scripts/test_agent.py --agent file-organizer --test-suite basic

# Expected: 70%+ validation, 80%+ tests pass
```

### Example 2: Create Complex Agent
```bash
# Step 1: Run all phases
python scripts/4_phase_sop.py --agent-name devops-orchestrator --mode interactive

# Step 2: Manual Phase 4 enhancement (add technical patterns)
# Edit: agent-outputs/devops-orchestrator/devops-orchestrator-enhanced-prompt-v2.md

# Step 3: Validate enhanced prompt
bash scripts/validate_prompt.sh -v -s 85 agent-outputs/devops-orchestrator/devops-orchestrator-enhanced-prompt-v2.md

# Step 4: Run comprehensive tests
python scripts/test_agent.py --agent devops-orchestrator --prompt-file agent-outputs/devops-orchestrator/devops-orchestrator-enhanced-prompt-v2.md --test-suite comprehensive

# Expected: 85%+ validation, 90%+ tests pass
```

### Example 3: Production Agent with Full 4-Phase SOP
```bash
# Step 1-3: Automated phases
python scripts/4_phase_sop.py --agent-name api-security-auditor --mode interactive

# Step 4: Manual technical enhancement
# Add code patterns, failure modes, MCP integrations
# Create: agent-outputs/api-security-auditor/api-security-auditor-enhanced-prompt-v2.md

# Validation
bash scripts/validate_prompt.sh -v -s 90 agent-outputs/api-security-auditor/api-security-auditor-enhanced-prompt-v2.md

# Testing
python scripts/test_agent.py --agent api-security-auditor --prompt-file agent-outputs/api-security-auditor/api-security-auditor-enhanced-prompt-v2.md --test-suite integration

# Expected: 90%+ validation (Gold tier), 95%+ tests pass
```

## Quality Tiers

### Bronze Tier (70-74%)
- Basic structure in place
- Core sections present
- Functional but minimal
- **Action**: Add evidence-based patterns, more examples

### Silver Tier (75-89%)
- Well-structured prompt
- Good command coverage
- Some evidence-based techniques
- **Action**: Minor enhancements for production readiness

### Gold Tier (90-100%)
- Production-ready prompt
- Comprehensive evidence-based patterns
- Extensive examples and guardrails
- Complete MCP integration
- **Action**: Deploy with confidence

## Best Practices

### During Phase 1
- Take time for deep domain analysis
- Identify 5+ real challenges, not generic ones
- Map technology stack comprehensively
- Think about integration patterns early

### During Phase 2
- Be specific about expertise domains
- Document decision heuristics with "When X, do Y because Z"
- Create concrete good/bad examples
- Document edge cases from real experience

### During Phase 3
- Use templates as starting point, not constraint
- Integrate evidence-based techniques naturally
- Create 2+ workflow examples with exact commands
- Make guardrails specific and actionable

### During Phase 4 (Manual Enhancement)
- Extract exact code patterns from real implementations
- Include file/line references for patterns
- Document failure modes with detection code
- Add MCP integration with exact syntax
- Define performance metrics for continuous improvement

### Validation
- Always validate before deployment
- Target 70%+ for basic agents
- Target 85%+ for complex agents
- Target 90%+ for production agents

### Testing
- Run appropriate test suite for agent complexity
- Basic suite: Simple, single-domain agents
- Comprehensive suite: Multi-domain agents
- Integration suite: Production-ready agents
- Fix issues until success rate meets target

## Troubleshooting

### Validation Fails
**Problem**: Prompt scores below minimum
**Solution**:
1. Check for missing required sections
2. Add evidence-based technique sections
3. Include 2+ workflow examples
4. Define 3+ guardrails with examples
5. Add specialist commands

### Tests Fail
**Problem**: Test suite pass rate too low
**Solution**:
1. Review identity consistency
2. Add missing universal commands
3. Document MCP integration patterns
4. Include memory usage specifications
5. Add cross-agent coordination patterns

### Phase 1 Validation Fails
**Problem**: Not identifying enough challenges
**Solution**: Research domain more deeply, ask "What makes this hard?"

### Phase 2 Validation Fails
**Problem**: Insufficient expertise domains or heuristics
**Solution**: Think about cognitive patterns activated when reasoning about the domain

### Phase 3 Output Incomplete
**Problem**: Prompt missing sections
**Solution**: Follow template structure exactly, ensure all sections present

## Support

For issues or questions:
1. Check test files in `../tests/` for examples
2. Review main skill documentation in `../SKILL.md`
3. Examine successful agent outputs in `agent-outputs/`
4. Consult the 4-phase SOP documentation in Desktop `.claude-flow/`

## Version History

- **v2.0** (Gold Tier): Complete 4-phase SOP automation, validation, testing
- **v1.0** (Silver Tier): Basic agent creation workflow
